"""stage 之间那条边: 消费者等多少、中间结果放哪、片上能存几块.

为什么收成一个概念: 原先这三件事是三个各自为政的旋钮
(`gmm2_k_segments` / `act_to_gmm2` / `InstancePolicy.gmm1_activation_depth`),
名字只对一条特定的边说话。可它们问的是同一组问题, 对**任意**一条生产者->消费者的
边都成立:

  1. 消费者要等生产者产出多少才能开工?   -> readiness (沿共享轴分几段就绪)
  2. 中间结果放哪?                      -> location  ("gm" / "onchip")
  3. 片上能同时存几块?                  -> depth     (计数信号量, 容量不是程序序)

共享轴 = 生产者切分的轴 ∩ 消费者的某条轴。activation->gmm2 这条边上, 生产者切的 N 轴
就是消费者的归约轴 K, 所以"分段就绪"沿 K 分; gmm1->activation 这条边上共享轴是 N,
一个 ACT 对一个 GMM1 tile, 粒度天然是 1。

算子工程师拿它做什么: 一条边一行, 改一个字段跑一次, 差值就是那个选择值多少钱。
见 analysis.design_space。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .hardware import DAV3510_NONINTERLEAVED_GMM1_ACTIVATION_DEPTH

#: 落点取值
LOC_GM = "gm"
LOC_ONCHIP = "onchip"
LOCATIONS = (LOC_GM, LOC_ONCHIP)


@dataclass(frozen=True)
class StageLink:
    """一条 stage 边的编排选择.

    producer / consumer: stage 名 (事件 meta 的 "stage")
    readiness: 消费者沿共享轴分几段独立就绪
        1 = 等齐 (缺省, 最少假设 —— 不声称实现能在只拿到部分轴时起步)
        0 = 最细 (每个 L1 块一段)
        N = 均分 N 段
      物理上可行的依据: L0C 本来就沿 K 分块累加, 所以第 j 段只等覆盖自己那段的生产者。
      代价: 段越多, 实现侧的 flag 轮询越多 —— 本模型不计这项, 所以细粒度的收益是上界。
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
    readiness: int = 1
    location: str = LOC_GM
    depth: int = 0
    colocated_by_hardware: bool = False

    def __post_init__(self) -> None:
        if self.readiness < 0:
            raise ValueError(f"{self.key}: readiness 不能为负 (0 = 最细, 1 = 等齐)")
        if self.location not in LOCATIONS:
            raise ValueError(
                f"{self.key}: location 只能是 {'/'.join(LOCATIONS)}, 收到 {self.location!r}")
        if self.depth < 0:
            raise ValueError(f"{self.key}: depth 不能为负")

    @property
    def key(self) -> Tuple[str, str]:
        return (self.producer, self.consumer)

    @property
    def materialised(self) -> bool:
        return self.location == LOC_GM


#: 缺省: 每条边都是"最少假设"。
#:
#: gmm1->activation 的 location 不是选择 —— Fixpipe 直给配对 AIV0, 本来就在片上;
#: depth=1 是 UB 里能同时存几块 GMM1 结果, 这是硬件容量 (不是实现风格), 取单槽
#: 是其中最紧的那个真实取值; 交织路径是 2 (见 KernelConfig.gmm1_interleaved)。
DEFAULT_LINKS: Tuple[StageLink, ...] = (
    StageLink("gmm1", "activation", readiness=1, location=LOC_ONCHIP,
              depth=int(DAV3510_NONINTERLEAVED_GMM1_ACTIVATION_DEPTH),
              colocated_by_hardware=True),
    StageLink("activation", "gmm2", readiness=1, location=LOC_GM),
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


def validate_links(links: Sequence[StageLink]) -> None:
    """重复边直接报错 (两条说法会让"谁生效"变成实现细节)."""
    seen = set()
    for lk in links:
        if lk.key in seen:
            raise ValueError(f"links 里有重复的边 {lk.producer}->{lk.consumer}")
        seen.add(lk.key)


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

    为什么不是旋钮: 这是编译期模板参数的后果, 不是算法工程师在 links 里能另选的
    编排。给了 location="onchip" 又开 prefetch, 两种说法会同时出现在一张图里。
    """
    base = resolve_link(links, "gmm1", "activation")
    if not bool(getattr(kernel, "topk_weights_prefetch", False)):
        return base
    return StageLink("gmm1", "activation", readiness=base.readiness,
                     location=LOC_GM, depth=0, colocated_by_hardware=False)
