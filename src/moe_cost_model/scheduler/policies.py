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
    work_conserving: 这个策略是否承诺"有就绪的活就不让核空着". 以优先级为首项的
    策略会**主动**让核空着去等高优先级的 stage, 那是它的语义, 不是调度器的缺陷 ——
    所以模型层的 work-conservation 护栏对声明 False 的策略不生效
    (model.A8W8WaveCostModel.check_work_conservation)。
    """

    prune_by_start: bool = False
    work_conserving: bool = True

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


class WorkConservingCriticalPath(SchedulingPolicy):
    """零空闲 + 关键路径优先: 先按最早可开始, 同刻按剩余关键链长降序.

    为什么首项仍是 start: 把 start 放第一位, 任何能立刻开始的事件都优先于只能更晚
    开始的事件 —— 核不会因为"等一个更关键但还没就绪的事件"而空闲, work-conservation
    由构造保证。关键路径只在**同样能立刻开始**的候选之间决定谁先上核, 这正是贪心
    (EarliestStart 按建图序打破平手) 会把关键 tile 挤到后面的那一步。

    remaining_path_us 由调度器在 needs_remaining_path 时反向拓扑算出 (事件自身时长 +
    下游最长链, 含边延迟)。
    """

    prune_by_start = True
    needs_remaining_path = True

    def event_key(self, ev, start, tbase, end_by_name):
        return (start, -float(ev.meta.get("remaining_path_us", 0.0)), ev.order, ev.name)


class PriorityByStage(SchedulingPolicy):
    """按阶段优先级: 指定 stage 排序, 同级按最早启动.

    名字必须与建图器真正发出的 stage 对齐。2026-10-05 之前缺省写的是 "act", 而没有任何
    建图器发这个名字 (发的是 "activation", builders/activation.py) —— 于是 ACT 瓦片全部
    落到兜底优先级 99, **排在 combine 之后**, 与这个策略声称的顺序相反, 而且一声不响。

    这一层不该知道 MegaMoE 的 stage 词表 (调度器要与 kernel 无关), 所以这里不校验名字;
    缺省列表与建图器的对齐由 tests/test_scheduler.py 守 —— 它从一次真实建图里取出实际
    发出的 stage 集合, 断言缺省列表是它的子集。未列出的 stage 仍按兜底排在后面, 那是
    **有意**的: stage_order 只排它关心的那几个。
    """

    #: 按优先级排意味着"宁可让核空着也先排高优先级的 stage" —— 实测 golden 的
    #: policy_priority_by_stage 形状在派发时刻绑定下留下 41879 核·us 的可避免空闲。
    #: 那是这个策略的语义, 所以护栏跳过它。
    work_conserving = False

    def __init__(self, stage_order: Sequence[str] = ("dispatch", "gmm1", "activation",
                                                     "gmm2", "combine")):
        self.prio = {s: i for i, s in enumerate(stage_order)}

    def event_key(self, ev, start, tbase, end_by_name):
        p = self.prio.get(str(ev.meta.get("stage", "")), 99)
        return (p, start, ev.order, ev.name)
