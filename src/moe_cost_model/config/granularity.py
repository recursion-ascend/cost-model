"""第 2 层: 每个 stage 的**事件粒度** —— 一个事件覆盖多少份该 stage 的自然工作单元.

为什么要有这一层
----------------
粒度是五个 stage **共有**的编排维度, 不是 combine 的特性。它与 tile 几何是两件事:

  * ``KernelConfig.tile_m`` / ``tile_n`` 受 L1/L0C 容量约束 —— **物理**;
  * "一个事件覆盖几个 tile" 是 **同步点密度 <-> 并行度** 的交换 —— **纯编排**。

粗粒度买到的是更少的同步/派发次数与更好的 B 流复用, 付出的是更粗的并行度 (一个
事件只能落一个核, 粒度越粗可参与的核越少) 与更晚的下游就绪。这笔交换该由算子
工程师来扫, 所以它是参数。

2026-10-04 之前只有 ``combine_granularity`` 一个参数 (缺口 11 的产物, 为回答一个
具体问题就地加的), dispatch 的粒度叫 ``dispatch_rows_per_item``, 而 GMM1 / SwiGLU /
GMM2 的粒度写死为 1。那是提问历史留下的洞, 不是物理。本模块把这个维度统一起来。

自然工作单元
------------
=============  ==========================  ======================================
stage          单元                        ``items_per_event`` 的物理上界
=============  ==========================  ======================================
dispatch       一行 (token x 目的专家)     窗口 / UB 容量
gmm1           一个 GMM1 tile              items x tile 的 A+B 字节 <= L1
activation      一个 GMM1 tile 的输出       items x tile 输出字节 <= UB;
                                           且必须与产它的 GMM1 同核 (Fixpipe)
gmm2           一个 GMM2 tile              items x tile 的 A+B 字节 <= L1
combine        一个 GMM2 tile 的输出       items x tile 输出字节 <= UB
=============  ==========================  ======================================

``items_per_event = 0`` 表示"整个专家切片一个事件" (combine 的 per_expert 就是它)。

**上表的"物理上界"目前只强制了一条**: activation 的粒度不得超过 UB 槽数
(``StageLink("gmm1","activation").depth``), 在 ``builders/activation.ActBatcher`` 构造时查
—— 槽不够会死锁, 所以必须拦。**按字节的容量上界 (items x tile 字节 <= L1/UB) 没有强制**:
它要在建图时按形状算字节, 还没接线。原先这里放了一个 ``validate_capacity`` 函数,
但它从未被任何地方调用 (2026-10-05 审计删除) —— 一个没人调的校验函数比没有更糟,
它让人以为这条约束已经在查了。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

#: 五个 stage. 顺序即数据流顺序。
STAGES: Tuple[str, ...] = ("dispatch", "gmm1", "activation", "gmm2", "combine")

#: 每个 stage 的工作单元名 (只用于报错与 meta, 不参与计算)。
UNIT_OF: Mapping[str, str] = {
    "dispatch": "row",
    "gmm1": "tile",
    "activation": "tile",
    "gmm2": "tile",
    "combine": "tile",
}


@dataclass(frozen=True)
class StageGranularity:
    """一个 stage 的事件粒度.

    items_per_event: 一个事件覆盖多少个单元。1 = 最细 (缺省, 最少假设:
        不预设任何合并)。0 = 整个专家切片。>1 = 攒这么多再发一次。
    """

    stage: str
    items_per_event: int = 1

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(
                f"未知 stage {self.stage!r}; 只能是 {STAGES}")
        if not isinstance(self.items_per_event, int) or isinstance(self.items_per_event, bool):
            raise ValueError(
                f"{self.stage} 的 items_per_event 必须是整数, 得到 {self.items_per_event!r}")
        if self.items_per_event < 0:
            raise ValueError(
                f"{self.stage} 的 items_per_event 不能为负 (0 = 整个切片一个事件)")


#: 缺省粒度. 四个计算/通信 stage 最细 (1 个单元一个事件 = 最少假设);
#: dispatch 取 0 = "沿用 tiling 算出的 routeItemsPerBatch" —— dispatch 的单元是行,
#: 一行一个事件既不是任何实现的做法也不是合理缺省, 所以这里的 0 不表示"整片",
#: 而表示"没有覆盖, 用 tiling 的值" (与历史参数 dispatch_rows_per_item 同义)。
DEFAULT_GRANULARITY: Tuple[StageGranularity, ...] = (
    StageGranularity("dispatch", 0),
    StageGranularity("gmm1", 1),
    StageGranularity("activation", 1),
    StageGranularity("gmm2", 1),
    StageGranularity("combine", 1),
)


@dataclass(frozen=True)
class GranularityAssignment:
    """stage -> 事件粒度. 与 RoleAssignment / StageLink 平行 (每 stage 一条)."""

    stages: Tuple[StageGranularity, ...] = DEFAULT_GRANULARITY
    _by_stage: Dict[str, StageGranularity] = field(
        default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        seen: Dict[str, StageGranularity] = {}
        for g in self.stages:
            if not isinstance(g, StageGranularity):
                raise ValueError(
                    f"granularity 的每一项必须是 StageGranularity, 得到 {g!r}")
            if g.stage in seen:
                raise ValueError(f"stage {g.stage!r} 的粒度给了两遍")
            seen[g.stage] = g
        for g in DEFAULT_GRANULARITY:
            seen.setdefault(g.stage, g)
        object.__setattr__(self, "_by_stage", seen)

    def of(self, stage: str) -> StageGranularity:
        try:
            return self._by_stage[stage]
        except KeyError:
            raise ValueError(f"未知 stage {stage!r}; 只能是 {STAGES}") from None

    def items(self, stage: str) -> int:
        """该 stage 一个事件覆盖多少单元 (0 = 整个切片)."""
        return self.of(stage).items_per_event

    def with_stage(self, stage: str, items_per_event: int) -> "GranularityAssignment":
        rest = tuple(g for g in self.stages if g.stage != stage)
        return GranularityAssignment(
            rest + (StageGranularity(stage, items_per_event),))

    def coarser_than_default(self) -> Tuple[str, ...]:
        """哪些 stage 偏离了缺省粒度 (供报告与 design_space 标注)."""
        base = {g.stage: g.items_per_event for g in DEFAULT_GRANULARITY}
        return tuple(s for s in STAGES if self.items(s) != base[s])


DEFAULT_GRANULARITIES = GranularityAssignment()


def resolve_granularity(value: Optional[object]) -> GranularityAssignment:
    """接受 GranularityAssignment / StageGranularity 序列 / {stage: items} 映射."""
    if value is None:
        return DEFAULT_GRANULARITIES
    if isinstance(value, GranularityAssignment):
        return value
    if isinstance(value, Mapping):
        return GranularityAssignment(tuple(
            StageGranularity(s, int(n)) for s, n in value.items()))
    return GranularityAssignment(tuple(value))
