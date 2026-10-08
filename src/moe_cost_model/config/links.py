"""stage 之间那条边: 消费者等多少、中间结果放哪、片上能存几块.

为什么收成一个概念: 原先这三件事是三个各自为政的旋钮
(`gmm2_k_segments` / `act_to_gmm2` / `InstancePolicy.gmm1_activation_depth`),
名字只对一条特定的边说话。可它们问的是同一组问题, 对**任意**一条生产者->消费者的
边都成立:

  1. 消费者要等生产者产出多少才能开工?   -> readiness (沿共享轴分几段就绪)
  2. 中间结果放哪?                      -> location  ("gm" / "onchip")
  3. 片上能同时存几块?                  -> depth     (计数信号量, 容量不是程序序)

共享轴 = 生产者切分的轴 ∩ 消费者的某条轴, 且消费者能沿它增量消费。这不是一句定性
的话: 每条边的共享轴、自然块、以及"分段在这条边上有没有意义"都在 EDGE_AXES 里
逐条写出, 校验照着它拒绝 —— 写得出的取值模型就必须买账, 否则报错, 不静默忽略。

readiness 与 granularity 的分工 (两者都"看起来在切事件", 必须分清):
    readiness   消费者**多早能开始** —— 沿共享轴把一个消费者拆成几段 (时序)
    granularity 一个事件**覆盖多少** —— 把多份工作并成一个事件 (打包)
两者正交只在"轴不同"时成立: GMM2 的 granularity 沿 M/N 打包 tile, readiness 沿 K
分段, 互不干涉。而 gmm2->combine 这条边的共享轴**就是** combine 的打包单元,
在那儿分段等于少打包 —— 该用 granularity, 所以 readiness 在那条边上被拒绝。

算子工程师拿它做什么: 一条边一行, 改一个字段跑一次, 差值就是那个选择值多少钱。
见 analysis.design_space。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .hardware import DAV3510_NONINTERLEAVED_GMM1_ACTIVATION_DEPTH
from .readiness import Readiness, parse_readiness

#: 落点取值
LOC_GM = "gm"
LOC_ONCHIP = "onchip"
LOCATIONS = (LOC_GM, LOC_ONCHIP)


@dataclass(frozen=True)
class SharedAxis:
    """一条边上的共享轴: 它是什么、自然块是什么、在这条边上能不能分段.

    axis / chunk 只用于报错与文档 (模型不按名字办事)。真正有后果的是
    segmentable: False 时任何非 "whole" 的 readiness 直接报错。

    segmentable=False 有两种成因, 分开写, 因为下一步的做法不同:
      packed_by_granularity=True  共享轴就是消费者的打包单元 -> 该调 granularity
      packed_by_granularity=False 一个消费者事件一次只吃一个块 (1:1) -> 没有可分的段,
                                  要更细得先改 tile 几何或 stage 划分
    consumed_by: segmentable=True 时**谁真的读它** —— 这一栏空着就等于声称
    "可以写但没人看", 校验会拦住 (见 validate_links)。
    """

    axis: str
    chunk: str
    segmentable: bool
    consumed_by: str = ""
    packed_by_granularity: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.segmentable and not self.consumed_by:
            raise ValueError(
                f"共享轴 {self.axis!r} 声称可分段, 但没写谁消费 —— "
                "可写而无人读的旋钮等于没有")


#: 本模型的四条 stage 边各自的共享轴。表里没有的边, links 里写它就报错 ——
#: 模型只有这五个 stage, 别的边名多半是拼错。
EDGE_AXES: Mapping[Tuple[str, str], SharedAxis] = {
    ("dispatch", "gmm1"): SharedAxis(
        axis="token 行", chunk="m-group", segmentable=False,
        note=("一个 GMM1 事件的行落在一个 m-group 内, 它等的就是那一个组的就绪标记 "
              "(builders/gmm1 的 dispatch_ready_event) —— 一次只吃一个块, "
              "没有可分的段; 要更细得先改 tile_m 或 m-group 的划分")),
    ("gmm1", "activation"): SharedAxis(
        axis="N (GMM1 的输出列)", chunk="GMM1 tile", segmentable=False,
        note=("一个 ACT 对一个 GMM1 tile (1:1), 一次只吃一个块。ACT 内部按 "
              "epilogue 行块再分是**粒度**而不是就绪 (builders/activation)")),
    ("activation", "gmm2"): SharedAxis(
        axis="K (GMM2 的归约轴 = GMM1 的输出列)", chunk="kL1 块", segmentable=True,
        consumed_by="builders/gmm2.add_gmm2_wave",
        note=("L0C 本来就沿 K 分块累加, 所以第 j 段只等覆盖自己那段 K 的 ACT。"
              "块大小由 select_kl1 定 (options.gmm2_kl1 可覆盖)")),
    ("gmm2", "combine"): SharedAxis(
        axis="切片内的 tile", chunk="GMM2 tile", segmentable=False,
        packed_by_granularity=True,
        note=("这条边的共享轴就是 combine 的打包单元: 一个 combine 事件要写的那些行, "
              "由它覆盖的 GMM2 tile 给齐。分段 = 少打包")),
}


@dataclass(frozen=True)
class StageLink:
    """一条 stage 边的编排选择.

    producer / consumer: stage 名 (事件 meta 的 "stage")
    readiness: 消费者沿共享轴分几段独立就绪。取值见 config/readiness:
        "whole" (缺省) / >= 2 的整数 (均分) / "per_chunk" / "first_chunk"。
        只在 EDGE_AXES 说这条边可分段时才允许非 "whole"。
    segment_sync_us: 每多一段, 消费者多付的同步开销 (除首段外每段加这么多时长)。
        **缺省 0 = 未标定**, 不是"量过是零" —— 所以缺省下分得越细越不差,
        那是模型没算这笔钱, 不是硬件上的结论。要标定它, 取的窗口是
        "上一段算完 -> 本段标志等待返回", 不含生产者还在算的那一段。
    location: 中间结果落哪
        "gm"     物化。生产者写回 GM, 消费者读回来。tile->核 自由。
        "onchip" 不物化。留在产它的那个核的片上 (硬件有 UB->L1 通路), 不付 GM 字节,
                 代价是消费者必须与生产者同核 —— 消费者要吃整条共享轴时, 这会把
                 一整组工作钉在一个核上, 并行度上限掉到"组数"。
    depth: 片上能同时存几块 (计数信号量; location="gm" 时无意义)
        0 = 不建这个约束 (假设不构成瓶颈 —— 是上界, 不是物理)
    colocated_by_hardware: 同核是**硬件强制**, 不是 location 推出来的选择。
        gmm1->activation 就是这种: GMM1 的结果经 L0C->UB 的 Fixpipe 直给配对的 AIV0,
        这条通路只在绑定对内存在。把它单列出来, 是为了不让 location 去推它 ——
        否则把这条边改成 "gm" 会连硬件约束一起删掉。
    """

    producer: str
    consumer: str
    readiness: Readiness = Readiness.whole()
    location: str = LOC_GM
    depth: int = 0
    colocated_by_hardware: bool = False
    segment_sync_us: float = 0.0

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "readiness", parse_readiness(self.readiness))
        except ValueError as exc:
            raise ValueError(
                f"{self.producer}->{self.consumer}: readiness: {exc}") from None
        if self.location not in LOCATIONS:
            raise ValueError(
                f"{self.key}: location 只能是 {'/'.join(LOCATIONS)}, 收到 {self.location!r}")
        if self.depth < 0:
            raise ValueError(f"{self.key}: depth 不能为负")
        if self.segment_sync_us < 0:
            raise ValueError(f"{self.key}: segment_sync_us 不能为负")

    @property
    def key(self) -> Tuple[str, str]:
        return (self.producer, self.consumer)

    @property
    def materialised(self) -> bool:
        return self.location == LOC_GM

    @property
    def axis(self) -> Optional[SharedAxis]:
        """这条边的共享轴声明 (表里没有则 None)."""
        return EDGE_AXES.get(self.key)


#: 缺省: 每条边都是"最少假设"。
#:
#: gmm1->activation 的 location 不是选择 —— Fixpipe 直给配对 AIV0, 本来就在片上;
#: depth=1 是 UB 里能同时存几块 GMM1 结果, 这是硬件容量 (不是实现风格), 取单槽
#: 是其中最紧的那个真实取值; 交织路径是 2 (见 KernelConfig.gmm1_interleaved)。
DEFAULT_LINKS: Tuple[StageLink, ...] = (
    StageLink("gmm1", "activation", location=LOC_ONCHIP,
              depth=int(DAV3510_NONINTERLEAVED_GMM1_ACTIVATION_DEPTH),
              colocated_by_hardware=True),
    StageLink("activation", "gmm2", location=LOC_GM),
)


def resolve_link(links: Sequence[StageLink], producer: str, consumer: str) -> StageLink:
    """取这条边的设置; 没给就回落到缺省表, 缺省表也没有就给"等齐 + 落 GM"。"""
    for lk in links:
        if lk.producer == producer and lk.consumer == consumer:
            return lk
    for lk in DEFAULT_LINKS:
        if lk.producer == producer and lk.consumer == consumer:
            return lk
    return StageLink(producer, consumer)


def validate_links(links: Sequence[StageLink], granularity=None) -> None:
    """拒绝三类写法, 而不是静默忽略它们.

    1. 重复边 —— 两条说法会让"谁生效"变成实现细节;
    2. 不认识的边 —— 模型只有 EDGE_AXES 那几条, 别的边名多半是拼错 (而且
       location/depth/readiness 在那儿一个都不会被读);
    3. 这条边上表达不出来的 readiness / segment_sync_us —— 共享轴不可分段时,
       非 "whole" 的就绪与非零的分段开销都没有作用对象。静默忽略的后果是
       算子工程师在那儿扫一圈得到"0 收益", 还以为硬件上也没收益。
    """
    seen = set()
    for lk in links:
        if lk.key in seen:
            raise ValueError(f"links 里有重复的边 {lk.producer}->{lk.consumer}")
        seen.add(lk.key)
        axis = lk.axis
        if axis is None:
            raise ValueError(
                f"links 里有模型没有的边 {lk.producer}->{lk.consumer}; "
                f"只有这几条: "
                f"{', '.join(f'{p}->{c}' for p, c in EDGE_AXES)}")
        if axis.segmentable:
            continue
        where = f"{lk.producer}->{lk.consumer}"
        if lk.segment_sync_us:
            raise ValueError(
                f"{where}: 这条边不分段, segment_sync_us 没有作用对象 "
                f"({axis.note})")
        if lk.readiness.is_whole:
            continue
        if axis.packed_by_granularity:
            have = granularity.items(lk.consumer) if granularity is not None else None
            raise ValueError(
                f"{where}: readiness={lk.readiness} 与 granularity 是同一个维度 —— "
                f"这条边的共享轴 ({axis.axis}) 就是 {lk.consumer} 的打包单元, "
                f"分段 = 少打包。要改就改 granularity[\"{lk.consumer}\"]"
                + (f" (现在 = {have})" if have is not None else ""))
        raise ValueError(
            f"{where}: readiness={lk.readiness} 在这条边上没有作用对象 —— "
            f"共享轴 {axis.axis} 不可分段: {axis.note}")


def effective_gmm1_act_link(links: Sequence[StageLink], kernel) -> StageLink:
    """gmm1->activation 这条边在给定编译点下的样子.

    TopkWeightsPrefetch=false (缺省): 就是 ``resolve_link`` 给的那条 —— GMM1 的结果经
    Fixpipe L0C->UB 直给配对的 AIV0, 片上, 占一个 UB 槽。

    TopkWeightsPrefetch=true: kernel 换了一条通路 —— AIC 把 tile 落 GM 并置
    ``gmm1TileStatus``, AIV0 等这个 GM 标志再 ``CopyGM2UB`` 读回
    (stage/mega_moe_gmm1_activation.h:618-645)。于是:
      * location = "gm": 中间结果物化, 字节要申报;
      * depth = 0: 那个 UB ping-pong 槽 (``vecSetSyncCom``) 在这条通路上不存在,
        AIV 等的是 GM 标志而不是配对 AIC 的 UB 交接。留着它等于凭空多一条约束。
      * colocated_by_hardware = False: 读回走 GM, Fixpipe 的同核要求没了。
        **同核仍然成立**, 但那是实现的分工 (prefetch 的 epilogue 循环按
        ``loopIdx += config.blockNum`` 分核, 与 GMM1 同一个 block 号), 不是硬件强制。
      * readiness 照抄: 这条边上它只能是 "whole" (1:1, validate_links 拦住别的),
        所以"照抄"不会把一个没人读的取值带进图里。

    为什么不是旋钮: 这是编译期模板参数的后果, 不是算法工程师在 links 里能另选的
    编排。给了 location="onchip" 又开 prefetch, 两种说法会同时出现在一张图里。
    """
    base = resolve_link(links, "gmm1", "activation")
    if not bool(getattr(kernel, "topk_weights_prefetch", False)):
        return base
    return StageLink("gmm1", "activation", readiness=base.readiness,
                     location=LOC_GM, depth=0, colocated_by_hardware=False)
