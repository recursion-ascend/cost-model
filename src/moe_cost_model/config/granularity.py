"""第 2 层: 每个 stage 的**事件粒度** —— 一个事件覆盖多少份该 stage 的自然工作单元.

为什么要有这一层
----------------
粒度是**每个** stage 共有的编排维度 (哪些 stage 由 config/stages 的词汇表给),
不是 combine 的特性。它与 tile 几何是两件事:

  * ``KernelConfig.tile_m`` / ``tile_n`` 受 L1/L0C 容量约束 —— **物理**;
  * "一个事件覆盖几个 tile" 是 **同步点密度 <-> 并行度** 的交换 —— **纯编排**。

粗粒度买到的是更少的同步/派发次数与更好的 B 流复用, 付出的是更粗的并行度 (一个
事件只能落一个核, 粒度越粗可参与的核越少) 与更晚的下游就绪。这笔交换该由算子
工程师来扫, 所以它是参数。

每个 stage 的粒度都是同一个维度上的取值, 不该每个 stage 一个专名参数 (只对一条边
说话), 也不该把 GMM1 / SwiGLU / GMM2 写死成 1 —— 那是提问历史留下的洞, 不是物理。

自然工作单元 (下表是仓内 MegaMoE 实现声明的那五个 stage, 见
implementations/megamoe_stages.py; 另一份实现声明自己的单元与上界)
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
它要在建图时按形状算字节, 还没接线。这里**不放** ``validate_capacity`` 这类没人调
的校验函数 —— 它比没有更糟, 让人以为这条约束已经在查了。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

from .stages import StageVocabulary, default_vocabulary

#: 哪些 stage、每个 stage 的单元叫什么, 由 **实现声明的词汇表** 给 (config/stages.py)。
#: 这里的两个模块级别名是缺省词汇表 (仓内 MegaMoE) 的视图, 保留是为了现有调用不改;
#: 另一份实现把自己的 StageVocabulary 传进 GranularityAssignment 即可。
STAGES: Tuple[str, ...] = default_vocabulary().pipeline

#: 每个 stage 的工作单元名 (只用于报错与 meta, 不参与计算)。
UNIT_OF: Mapping[str, str] = default_vocabulary().unit_of


@dataclass(frozen=True)
class StageGranularity:
    """一个 stage 的事件粒度.

    items_per_event: 一个事件覆盖多少个单元。1 = 最细 (缺省, 最少假设:
        不预设任何合并)。0 = 整个专家切片。>1 = 攒这么多再发一次。
    """

    stage: str
    items_per_event: int = 1
    #: 用哪份词汇表判"这个 stage 存不存在"。不参与相等/repr: 它是校验的出处,
    #: 不是这条粒度设置的内容。
    vocab: StageVocabulary = field(
        default_factory=default_vocabulary, compare=False, repr=False)

    def __post_init__(self) -> None:
        self.vocab.require(self.stage)
        if not isinstance(self.items_per_event, int) or isinstance(self.items_per_event, bool):
            raise ValueError(
                f"{self.stage} 的 items_per_event 必须是整数, 得到 {self.items_per_event!r}")
        if self.items_per_event < 0:
            raise ValueError(
                f"{self.stage} 的 items_per_event 不能为负 (0 = 整个切片一个事件)")


def default_granularity(vocab: Optional[StageVocabulary] = None
                        ) -> Tuple[StageGranularity, ...]:
    """这份词汇表的缺省粒度: 每个 stage 取 ``vocab.items_default``.

    缺省是"最细" (1 个单元一个事件 = 最少假设, 不预设任何合并); 偏离 1 的
    stage 由词汇表自己声明并写出理由 (MegaMoE 的 dispatch 取 0, 见
    implementations/megamoe_stages)。
    """
    v = vocab or default_vocabulary()
    return tuple(StageGranularity(s, v.items_default(s), vocab=v) for s in v.pipeline)


#: 缺省词汇表的缺省粒度 (现有调用与测试沿用这个名字)。
DEFAULT_GRANULARITY: Tuple[StageGranularity, ...] = default_granularity()


@dataclass(frozen=True)
class GranularityAssignment:
    """stage -> 事件粒度. 与 RoleAssignment / StageLink 平行 (每 stage 一条)."""

    stages: Optional[Tuple[StageGranularity, ...]] = None
    vocab: StageVocabulary = field(
        default_factory=default_vocabulary, compare=False, repr=False)
    _by_stage: Dict[str, StageGranularity] = field(
        default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.stages is None:
            object.__setattr__(self, "stages", default_granularity(self.vocab))
        seen: Dict[str, StageGranularity] = {}
        for g in self.stages:
            if not isinstance(g, StageGranularity):
                raise ValueError(
                    f"granularity 的每一项必须是 StageGranularity, 得到 {g!r}")
            if g.stage in seen:
                raise ValueError(f"stage {g.stage!r} 的粒度给了两遍")
            seen[g.stage] = g
        for g in default_granularity(self.vocab):
            seen.setdefault(g.stage, g)
        object.__setattr__(self, "_by_stage", seen)

    def of(self, stage: str) -> StageGranularity:
        try:
            return self._by_stage[stage]
        except KeyError:
            raise ValueError(
                f"未知 stage {stage!r}; 只能是 {tuple(self.vocab.pipeline)}") from None

    def items(self, stage: str) -> int:
        """该 stage 一个事件覆盖多少单元 (0 = 整个切片)."""
        return self.of(stage).items_per_event

    def with_stage(self, stage: str, items_per_event: int) -> "GranularityAssignment":
        rest = tuple(g for g in self.stages if g.stage != stage)
        return GranularityAssignment(
            rest + (StageGranularity(stage, items_per_event, vocab=self.vocab),),
            vocab=self.vocab)

    def coarser_than_default(self) -> Tuple[str, ...]:
        """哪些 stage 偏离了缺省粒度 (供报告与 design_space 标注)."""
        v = self.vocab
        return tuple(s for s in v.pipeline if self.items(s) != v.items_default(s))


DEFAULT_GRANULARITIES = GranularityAssignment()


def resolve_granularity(value: Optional[object],
                        vocab: Optional[StageVocabulary] = None) -> GranularityAssignment:
    """接受 GranularityAssignment / StageGranularity 序列 / {stage: items} 映射."""
    v = vocab or default_vocabulary()
    if value is None:
        return GranularityAssignment(vocab=v) if vocab else DEFAULT_GRANULARITIES
    if isinstance(value, GranularityAssignment):
        return value
    if isinstance(value, Mapping):
        return GranularityAssignment(tuple(
            StageGranularity(s, int(n), vocab=v) for s, n in value.items()), vocab=v)
    return GranularityAssignment(tuple(value), vocab=v)
