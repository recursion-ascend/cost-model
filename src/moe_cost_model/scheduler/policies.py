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
    #: 要不要状态视图。True 时引擎每步构造一个只读 SchedulerView 并改调
    #: event_key_with_view —— 策略因此能做一步前瞻 (见 scheduler/view.py)。
    #: 缺省 False: 不构造, 不走那条分支。
    wants_view: bool = False
    #: 要不要"从这个事件起到 sink 的最长链"。True 时引擎反向拓扑算一遍写进 meta。
    needs_remaining_path: bool = False

    def event_key(self, ev: Event, start: float,
                  tbase: Dict[str, float], end_by_name: Dict[str, float]) -> tuple:
        raise NotImplementedError

    def event_key_with_view(self, ev: Event, start: float,
                            tbase: Dict[str, float], end_by_name: Dict[str, float],
                            view) -> tuple:
        """wants_view = True 的策略实现这个; 缺省回落到 event_key.

        view 是只读的 (scheduler/view.SchedulerView): 能问状态、能做一步试探,
        改不了状态。引擎不认识任何具体规则 —— 规则全在策略这一侧。
        """
        return self.event_key(ev, start, tbase, end_by_name)


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

    名字必须与建图器真正发出的 stage 对齐。写成 "act" 这类不存在的名字 (建图器发的是
    "activation", builders/activation.py) 会让该 stage 的瓦片全部落到兜底优先级 99,
    **排在 combine 之后**, 与这个策略声称的顺序相反, 而且一声不响。

    这一层不该知道 MegaMoE 的 stage 词表 (调度器要与 kernel 无关), 所以这里不校验名字;
    缺省列表与建图器的对齐由 tests/test_scheduler.py 守 —— 它从一次真实建图里取出实际
    发出的 stage 集合, 断言缺省列表是它的子集。未列出的 stage 仍按兜底排在后面, 那是
    **有意**的: stage_order 只排它关心的那几个。
    """

    #: 按优先级排意味着"宁可让核空着也先排高优先级的 stage" —— 实测在派发时刻绑定下
    #: 留下 41879 核·us 的可避免空闲。
    #: 那是这个策略的语义, 所以护栏跳过它。
    work_conserving = False

    def __init__(self, stage_order: Sequence[str] = ("dispatch", "gmm1", "activation",
                                                     "gmm2", "combine")):
        self.prio = {s: i for i, s in enumerate(stage_order)}

    def event_key(self, ev, start, tbase, end_by_name):
        p = self.prio.get(str(ev.meta.get("stage", "")), 99)
        return (p, start, ev.order, ev.name)


class LookaheadOneStep(SchedulingPolicy):
    """一步前瞻破平. **实测不如缺省策略 —— 留在仓里是为了跑通契约, 不是推荐用法.**

    动机: 表调度对输入不单调 (Graham 1969 的时序异常) —— 本仓实测一条依赖边加 0.01us
    让总时长变化 -2.81%, 就绪均分 3 段比 2 段差 1.0%。根因看起来是"此刻最早开始"这个
    判据不看它挡住了谁, 同刻并列时只按建图序与名字破平。所以试了前瞻。

    规则: 每个候选算一个下界

        bound(c) = max( start(c) + 剩余最长链(c),
                        max_{d != c, d 就绪} ( start_if(d, c 占住的资源到 end(c))
                                               + 剩余最长链(d) ) )

    排序键 = (start, bound, order, name)。两项都不含资源争用与按核信号量的后效, 所以是
    下界, 评分 admissible。

    **实测结论 (三个形状, 单 rank 直达 model)**:

        形状              earliest    bound 当首项    (start, bound)    同刻按链最长
        skewed             247.466     364.277 (+47%)   247.466 (±0)     247.625 (+0.06%)
        uniform 2x16x64    607.254       --             609.698 (+0.40%) 607.418 (+0.03%)
        uniform 2x6x256    602.578       --             610.155 (+1.26%) 605.104 (+0.42%)
        耗时               0.2-0.3s    160.6s           25-57s           0.3-0.6s

    三条都读得出来:

      1. **把 bound 放首项是错的** (+47%): 最小化一个松的下界不等于最小化墙钟。剩余最长链
         那一项会主导, 于是它总挑链最长的事件, 哪怕那个事件要晚得多才能开工 —— 核白闲着。
         贪心的"最早开始"虽然短视, 但它保证工作守恒, 那是实打实的收益。
      2. **退成同刻破平后仍不赢**: 持平或 +0.4%~+1.3%。按"挡住别人最少"破平会系统性地
         偏向剩余链**短**的事件 (自己那条链进 max 里, 链长的 bound 就大), 等于推迟关键链;
         反过来按"链最长优先"破平也不赢 (+0.03%~+0.42%)。现有证据下, 同刻并列这件事上
         没有比建图序更好的规则 —— 至少这两个方向都不是。
      3. **代价是 100-200 倍**: 每步 O(|ready|) 次试探, 且关掉了 EarliestStart 的快路径。

    所以它的用途只有一个: 这是 wants_view 契约 (scheduler/view.py) 上的第一个实现, 把
    "策略能试探"这条路跑通, 并且作为后续多步搜索 (beam) 的对照基线。要用它做结论得先
    有新的证据。

    它给的是"这个事件图换一条决策规则能快多少", 不是"硬件会跑多快"; 结果能不能实现取决于
    绑定方式 (静态分核下要把顺序编译进去 = 一个 CoreAssignment 算法; 派发时刻绑定下要先
    把运行时取活的开销算进来), 并且必须与**同一种绑定方式**的基线比。
    """

    prune_by_start = True           # 首项仍是开始时刻, 引擎的时间剪枝继续有效
    wants_view = True
    needs_remaining_path = True

    def __init__(self, max_candidates: int = 0):
        #: 前瞻时最多看 ready 集里 t_base 最小的这几个 (0 = 全部)。限制它是为了控成本。
        self.max_candidates = int(max_candidates)

    def event_key(self, ev, start, tbase, end_by_name):
        # 没有视图时退化成最早启动 —— 任何入口下都给得出一个确定的键
        return (start, 0.0, ev.order, ev.name)

    def event_key_with_view(self, ev, start, tbase, end_by_name, view):
        if start == float("inf"):
            return (float("inf"), 0.0, ev.order, ev.name)
        end = start + max(0.0, ev.duration_us)
        bound = end + view.remaining_path_us(ev.name)
        busy = view.occupied_by(ev.name, end)
        if busy:
            others = view.ready_names()
            if self.max_candidates > 0 and len(others) > self.max_candidates:
                others = tuple(sorted(others, key=view.t_base))[:self.max_candidates]
            for other in others:
                if other == ev.name:
                    continue
                t = view.start_if(other, busy)
                if t == float("inf"):
                    continue
                # 剩余最长链已含该事件自身的时长, 所以这里只加它的开始时刻
                got = t + view.remaining_path_us(other)
                if got > bound:
                    bound = got
        return (start, bound, ev.order, ev.name)
