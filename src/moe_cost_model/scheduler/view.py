"""第 2 层: 调度状态的**只读视图** —— 策略能问"现在怎样"与"如果这样会怎样".

为什么要这一层
--------------
只给 ``SchedulingPolicy`` 一个排序键的四个入参 (事件、它的最早开始时刻、全体 ready
的 t_base 表、已完成事件的结束时刻) 时, 策略看得见"现在", 看不见
"如果我选它, 之后会怎样"。于是只写得出**静态优先级**类规则 (按剩余关键路径、slack
排序), 写不出**试探**类规则 (前瞻 D 步再决定、beam)。

但调度引擎本身不该认识任何一种派发规则。所以把"可以试探"做成一个只读契约:

  ``SchedulerView``  问状态: 谁就绪、各自最早能什么时候开始、某个资源何时空出来
                     做试探: ``start_if(name, busy_until)`` —— 假设某些资源被占到
                     某个时刻, 这个事件最早能什么时候开始

视图**不能写状态**: 没有提交、没有回退。它暴露的两个计算 (t_base 查询、start_if)
都是纯函数, 由引擎里同一批不写状态的闭式判据实现。策略因此无法把引擎带到一个
不一致的状态里去。

它不是什么
----------
``start_if`` 只往前推**一步**的占用: 它回答"这些资源被占到这些时刻, 你最早何时能开始",
不回答"再往后 D 步会怎样"。多步试探要能复制整个调度状态 (资源空闲时刻、计数信号量
的归还时间线、就绪集), 那是下一步的事 —— 在那之前, 基于本视图的策略只能做一步前瞻。

共享资源的判据 (``start_if`` 的保守性)
--------------------------------------
一个事件被另一个事件的占用推迟, 只在两者**真的抢同一个资源**时发生:
  具体资源名相同            -> 推迟
  同一个资源池的占位符       -> 池里有多个成员时**不**推迟 (对端可以落别的核);
                             池只剩一个成员时等同于具体资源
按核的计数信号量 (UB 槽、L1 槽) 不在本判据内 —— 它们由引擎的容量准入处理, 视图给出的
``t_base`` 已经含了当前台账下的准入时刻, 但不含"假设那个候选占了某个核的槽"之后的变化。
所以 ``start_if`` 是**下界**: 真实开始时刻不会比它早, 可能比它晚。用它做评分是
admissible 的 (不会因为高估而剪掉好分支), 这正是前瞻评分需要的方向。
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from .events import Event, pool_key


class SchedulerView:
    """一次调度步的只读状态视图. 引擎每步构造一个 (只在策略索要时构造)."""

    __slots__ = ("_by_name", "_ready", "_tbase", "_end", "_free", "_members",
                 "_start_of")

    def __init__(self, *, by_name: Mapping[str, Event], ready: Mapping[str, None],
                 tbase: Mapping[str, float], end_by_name: Mapping[str, float],
                 resource_free: Mapping[str, float],
                 pool_members: Mapping[str, Sequence[str]],
                 start_of: Callable[[str], float]):
        self._by_name = by_name
        self._ready = ready
        self._tbase = tbase
        self._end = end_by_name
        self._free = resource_free
        self._members = pool_members
        self._start_of = start_of

    # ---- 问状态 ----

    def ready_names(self) -> Tuple[str, ...]:
        """此刻就绪 (依赖齐备) 的事件名, 按入集序."""
        return tuple(self._ready)

    def event(self, name: str) -> Event:
        return self._by_name[name]

    def t_base(self, name: str) -> float:
        """max(依赖就绪, 所需资源空闲). 不含容量准入 —— 那在 start() 里。"""
        return self._tbase[name]

    def start(self, name: str) -> float:
        """这个就绪事件此刻最早能开始的时刻 (含计数信号量准入). 不可行则 +inf."""
        return self._start_of(name)

    def end_of(self, name: str) -> Optional[float]:
        """已完成事件的结束时刻; 没完成则 None."""
        return self._end.get(name)

    def resource_free_at(self, resource: str) -> float:
        """这个具体资源何时空出来 (占位符请自己按 pool_members 展开)."""
        return self._free.get(resource, 0.0)

    def pool_members(self, pool: str) -> Tuple[str, ...]:
        return tuple(self._members.get(pool, ()))

    def remaining_path_us(self, name: str) -> float:
        """从这个事件开始算起到任一 sink 的最长链 (含边延迟).

        只有策略声明 needs_remaining_path = True 时引擎才算它, 否则恒 0。
        """
        return float(self._by_name[name].meta.get("remaining_path_us", 0.0) or 0.0)

    # ---- 做试探 ----

    def occupied_by(self, name: str, until: float) -> Dict[str, float]:
        """这个事件若在 [.., until) 占着它的资源, 哪些**具体资源**被占到何时.

        占位符只在池里仅剩一个成员时才算占住 (见模块开头的共享资源判据)。
        """
        out: Dict[str, float] = {}
        ev = self._by_name[name]
        for r in ev.resources:
            pk = pool_key(r)
            if pk is None:
                out[r] = max(out.get(r, 0.0), until)
                continue
            mem = self._members.get(pk, ())
            if len(mem) == 1:
                out[mem[0]] = max(out.get(mem[0], 0.0), until)
        return out

    def start_if(self, name: str, busy_until: Mapping[str, float]) -> float:
        """假设 busy_until 里的资源被占到那些时刻, 这个事件最早何时能开始.

        返回 max(此刻的最早开始时刻, 与它抢同一资源的那些占用的结束时刻)。
        是**下界** (不含假设占用对按核信号量台账的影响), 见模块开头。
        """
        t = self._start_of(name)
        if t == float("inf") or not busy_until:
            return t
        ev = self._by_name[name]
        for r in ev.resources:
            pk = pool_key(r)
            if pk is None:
                got = busy_until.get(r)
                if got is not None and got > t:
                    t = got
                continue
            mem = self._members.get(pk, ())
            if len(mem) == 1:
                got = busy_until.get(mem[0])
                if got is not None and got > t:
                    t = got
        return t
