"""第 2 层: ready 集选择策略 — EarliestStart / CriticalPathFirst / PriorityByStage.

默认值 = 当前 kernel 行为. 可替换, simulate 结果随之改变.
"""
from __future__ import annotations

from typing import Dict, Sequence

from .events import Event


# ---------------------------------------------------------------------------
# 1. Ready 集选择 (dag.py 调度器)
# ---------------------------------------------------------------------------

class SchedulingPolicy:
    """从 ready 集中选择下一个调度事件的排序键.

    返回 tuple, 越小越优先. 调度器取 min.
    prune_by_start: 排序键首项是否为开始时刻. True 时调度器可用
    tbase ≤ best.start 剪枝; False (优先级/slack 等非时间首项) 时
    剪枝无效, 全体 ready 候选都会被比较.
    """

    prune_by_start: bool = False

    def event_key(self, ev: Event, start: float,
                  tbase: Dict[str, float], end_by_name: Dict[str, float]) -> tuple:
        raise NotImplementedError


class EarliestStart(SchedulingPolicy):
    """默认: 最早启动, 同刻按建图序, 再按字典序."""

    prune_by_start = True

    def event_key(self, ev, start, tbase, end_by_name):
        return (start, ev.order, ev.name)


class CriticalPathFirst(SchedulingPolicy):
    """关键路径优先: 下游总时差最小的先跑.

    需要预计算 downstream_slack (从 DAG 反向遍历一次).
    若 meta 无 slack 信息则退化为 EarliestStart.
    """

    def event_key(self, ev, start, tbase, end_by_name):
        slack = ev.meta.get("downstream_slack")
        if slack is None:
            return (start, ev.order, ev.name)
        return (slack, start, ev.order, ev.name)


class PriorityByStage(SchedulingPolicy):
    """按阶段优先级: 指定 stage 排序, 同级按最早启动."""

    def __init__(self, stage_order: Sequence[str] = ("dispatch", "gmm1", "act", "gmm2", "combine")):
        self.prio = {s: i for i, s in enumerate(stage_order)}

    def event_key(self, ev, start, tbase, end_by_name):
        p = self.prio.get(str(ev.meta.get("stage", "")), 99)
        return (p, start, ev.order, ev.name)
