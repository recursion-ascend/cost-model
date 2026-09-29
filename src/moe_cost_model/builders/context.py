"""第 4 层: 建图跨阶段共享状态.

一个 BuildContext 的生命周期 = 一次 build() 调用. 五个状态字典是
stage 之间的接口:
  dispatch_ready_event   dispatch 写 → gmm1 读        (expert, group) → 就绪标记名
  activation_ready       gmm1 写  → gmm2 读           (expert, group) → [(ntile, ACT 名)]
  gmm1_act_history       gmm1 写  → 同核 gmm1 读      每核 ACT 名序列 (UB 缓冲依赖)
  gmm2_combine_history   gmm2 写  → combine credit 读 每核 COMBINE 名序列
  last_combine_by_core   gmm2 写  → 下一波 dispatch 读 核 → 该核最后 COMBINE 名
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..shape import BlockCursor


@dataclass
class BuildContext:
    cursor: BlockCursor
    dispatch_ready_event: Dict[Tuple[int, int], str] = field(default_factory=dict)
    activation_ready: Dict[Tuple[int, int], List[Tuple[int, str]]] = field(default_factory=dict)
    gmm1_act_history: List[List[str]] = field(default_factory=list)
    gmm2_combine_history: List[List[str]] = field(default_factory=list)
    last_combine_by_core: Dict[int, str] = field(default_factory=dict)

    @classmethod
    def fresh(cls, p: int, cursor: BlockCursor = None) -> "BuildContext":
        return cls(
            cursor=cursor if cursor is not None else BlockCursor(p, 0),
            gmm1_act_history=[[] for _ in range(p)],
            gmm2_combine_history=[[] for _ in range(p)],
        )
