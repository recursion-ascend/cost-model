"""第 1 层: 一份实现的 **stage 词汇表** —— 有哪些 stage、它们之间有哪些边.

为什么要有这一层
----------------
``scheduler/`` 不认识任何 stage 名: 它只看 ``Event`` 的 resources / deps /
acquires / releases。但 ``config/`` 与 ``analysis/`` 里有七处把 MegaMoE 的五个
stage 名写成了字面量 (粒度表、边的共享轴、角色缺省表、必经链、搬运组、比对口径、
排空归集)。于是"换一个算子"这件事在代码里没有落脚处: 另一份实现 (例如多一个权重
解压阶段, 或 MoonEP 那种 stage 划分) 要么改七个文件, 要么被 ``STAGES`` 校验直接
拒掉。

本模块把那七处的字面量收到一个值对象里。``MEGAMOE`` 这一份声明的内容**与原先
逐处写死的完全相同** —— 这一步只是归位, 不动行为 (判据: golden 指纹零差异)。
第二份实现要做的事就是再写一个 ``StageVocabulary``, 把它传给
``GranularityAssignment`` / ``RoleAssignment`` / ``validate_links``。

词汇表里放什么、不放什么
------------------------
放: **stage 的存在与关系** —— 名字、数据流顺序、工作单元名、执行角色、边的共享轴。
不放: 时长公式 (``costs``)、tile 几何 (``KernelConfig``)、建图顺序 (适配器)。
判据是"换一个算子时这条信息会不会变, 且与具体时长无关"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

#: 三个执行角色 (与 config/roles 同一组名字, 在那里有物理说明)。
#: 放在这里是因为词汇表要给每个 stage 指一个角色, 而 roles 反过来要读词汇表。
AIC = "AIC"
AIV0 = "AIV0"
AIV1 = "AIV1"


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
    "可以写但没人看", 校验会拦住 (见 config.links.validate_links)。
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
                "可写而无人读的参数等于没有")


@dataclass(frozen=True)
class StageVocabulary:
    """一份实现有哪些 stage —— 下游七处校验与缺省表的唯一出处.

    pipeline      主链上的 stage, 顺序即数据流顺序。粒度 / 边 / 比对只认这些。
    unit_of       每个 pipeline stage 的自然工作单元名 (报错与 meta 用)。
    default_items 粒度缺省不取 1 的 stage (见 config/granularity 对 dispatch 的说明)。
    roles         stage -> 执行角色。键集合比 pipeline 大: 辅助 stage
                  (shared_*, dispatch_call, mask_scan ...) 不在主链上但要落核。
    cube_only     只能跑在 Cube 上的 stage (矩阵乘, 物理)。
    edges         有哪些 stage 边, 每条边的共享轴。表里没有的边在 links 里写就报错。
    compared      与实测比对覆盖哪些 stage (见 validation/compare 对 dispatch_call 的说明)。
    drain         每个引擎上跑哪些 stage (排空节点按此归集)。
    steal_groups  搬运时必须随动的配对 (驱动 stage, 随动 stage) —— 同核关系决定的,
                  不是策略。
    default_steal_drivers 其中缺省就开的那几个驱动 stage (其余是 opt-in;
                  为什么只开一部分由声明处写出理由)。
    chain         一份工作的必经链, 用于依赖链下界。缺省 = pipeline。
    """

    operator: str
    pipeline: Tuple[str, ...]
    unit_of: Mapping[str, str]
    roles: Mapping[str, str]
    cube_only: Tuple[str, ...]
    edges: Mapping[Tuple[str, str], SharedAxis]
    compared: Tuple[str, ...]
    drain: Tuple[Tuple[str, Tuple[str, ...]], ...]
    steal_groups: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    default_steal_drivers: Tuple[str, ...] = ()
    default_items: Mapping[str, int] = field(default_factory=dict)
    chain: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.pipeline:
            raise ValueError(f"{self.operator}: pipeline 不能为空")
        if len(set(self.pipeline)) != len(self.pipeline):
            raise ValueError(f"{self.operator}: pipeline 里有重复的 stage")
        if set(self.unit_of) != set(self.pipeline):
            raise ValueError(
                f"{self.operator}: unit_of 必须且只能覆盖 pipeline 的 stage; "
                f"缺 {sorted(set(self.pipeline) - set(self.unit_of))}, "
                f"多 {sorted(set(self.unit_of) - set(self.pipeline))}")
        missing_role = [s for s in self.pipeline if s not in self.roles]
        if missing_role:
            raise ValueError(
                f"{self.operator}: 这些 stage 没给执行角色: {missing_role}")
        for s in self.cube_only:
            if s not in self.roles:
                raise ValueError(f"{self.operator}: cube_only 里的 {s!r} 不在 roles 表里")
            if self.roles[s] != AIC:
                raise ValueError(
                    f"{self.operator}: {s!r} 声明只能跑 Cube, 但缺省角色是 {self.roles[s]!r}")
        for p, c in self.edges:
            for s in (p, c):
                if s not in self.pipeline:
                    raise ValueError(
                        f"{self.operator}: 边 {p}->{c} 提到了不在 pipeline 里的 {s!r}")
        for s in self.compared:
            if s not in self.pipeline:
                raise ValueError(f"{self.operator}: compared 里的 {s!r} 不在 pipeline 里")
        for _engine, sts in self.drain:
            for s in sts:
                if s not in self.roles:
                    raise ValueError(f"{self.operator}: drain 里的 {s!r} 不在 roles 表里")
        for driver, paired in self.steal_groups:
            for s in (driver,) + tuple(paired):
                if s not in self.pipeline:
                    raise ValueError(
                        f"{self.operator}: steal_groups 里的 {s!r} 不在 pipeline 里")
        drivers = {d for d, _ in self.steal_groups}
        for d in self.default_steal_drivers:
            if d not in drivers:
                raise ValueError(
                    f"{self.operator}: default_steal_drivers 里的 {d!r} 不是任何搬运组的驱动")
        for s, n in self.default_items.items():
            if s not in self.pipeline:
                raise ValueError(f"{self.operator}: default_items 里的 {s!r} 不在 pipeline 里")
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                raise ValueError(
                    f"{self.operator}: default_items[{s!r}] 必须是非负整数, 得到 {n!r}")
        if not self.chain:
            object.__setattr__(self, "chain", tuple(self.pipeline))
        for s in self.chain:
            if s not in self.pipeline:
                raise ValueError(f"{self.operator}: chain 里的 {s!r} 不在 pipeline 里")

    @property
    def all_stages(self) -> Tuple[str, ...]:
        """pipeline + 只在 roles 表里出现的辅助 stage (按 pipeline 优先的稳定顺序)."""
        extra = [s for s in self.roles if s not in self.pipeline]
        return tuple(self.pipeline) + tuple(extra)

    @property
    def default_steal_groups(self) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """缺省开启的搬运组 (按 steal_groups 的顺序)."""
        keep = set(self.default_steal_drivers)
        return tuple(g for g in self.steal_groups if g[0] in keep)

    def items_default(self, stage: str) -> int:
        """这个 stage 的缺省事件粒度 (没特别声明的都是 1 = 最细, 最少假设)."""
        return int(self.default_items.get(stage, 1))

    def require(self, stage: str) -> str:
        """stage 必须在 pipeline 里, 否则报错而不是猜一个."""
        if stage not in self.pipeline:
            raise ValueError(
                f"未知 stage {stage!r}; 只能是 {tuple(self.pipeline)}")
        return stage

    def axis_of(self, producer: str, consumer: str):
        """这条边的共享轴声明 (表里没有则 None)."""
        return self.edges.get((producer, consumer))


def default_vocabulary() -> StageVocabulary:
    """缺省词汇表 = 仓内 MegaMoE 实现声明的那一份.

    这是**向后兼容的缺省**, 不是框架假设: 核心 (scheduler/timing) 不认识 stage 名,
    而 config/analysis 里的校验与缺省表需要一份词汇表才能工作。不传 vocab 时取这一份,
    于是仓内现有的调用与 golden 保持不变; 另一份实现把自己的 StageVocabulary 传进
    GranularityAssignment / RoleAssignment / validate_links 即可, 不必改这些模块。

    懒导入是为了让依赖方向成立: 具体实现依赖 config, 不是反过来。
    """
    from ..implementations.megamoe_stages import MEGAMOE
    return MEGAMOE
