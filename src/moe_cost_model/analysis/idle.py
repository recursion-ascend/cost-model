"""第 6 层: AIC/AIV 空闲分解 — 区分"DAG 逼出来的"与"真浪费".

为什么要分开
------------
"核不能有空闲"作为绝对约束在逻辑上不可能: 事件必须等前置完成才能开始, 而 t=0 时刻
GMM1 还在等 dispatch_ready (AIV1 产出), 所以开头那段**所有 AIC 必须空着**。这是依赖
约束直接推出来的, 不是模型缺陷。

能成立的不变量只有一个 —— **work-conserving**: 核不得在"存在已就绪的活"时空闲。
于是空闲要分成两类:

  forced    此刻该资源池里**没有**已就绪未开始的事件 → 空闲无法消除, 只能靠改 DAG
            结构 (加深流水、改波的组成) 来减少
  avoidable 此刻**有**已就绪未开始的事件, 却还有核空着 → work-conservation 违规,
            换 tile→核 的绑定方式 (如派发时刻晚绑定) 就能回收

avoidable 是**上界**
--------------------
它只检查"有就绪的活 + 有空核", 没有检查那个活是否真能落到那个空核上:
  - AIC/AIV 共位: GMM1 落核 X 时 ACT 必须落 AIV0:X, 而 AIV0:X 可能正忙
  - 队列计数信号量 (Q:aic:c*) 可能另有约束
所以真实可回收量 <= avoidable_idle_us。把它当"值不值得动绑定方式"的量级判断, 不要
当承诺。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: 时间比较的容差. 调度器的时刻是浮点, 相邻事件的 end 与下一个 start 常常只差 ulp。
_EPS = 1e-9


@dataclass(frozen=True)
class IdleSegment:
    """一段 avoidable 空闲: [t_begin, t_end) 内有核空着而有活已就绪."""
    t_begin: float
    t_end: float
    idle_resources: Tuple[str, ...]
    waiting_ready: Tuple[str, ...]      # 已就绪但还没开始的事件名 (最多留几条做样本)

    @property
    def core_us(self) -> float:
        return len(self.idle_resources) * (self.t_end - self.t_begin)


@dataclass(frozen=True)
class IdleReport:
    """一个资源池 (如某个 rank 的全部 AIC) 的空闲分解."""
    pool: Tuple[str, ...]
    horizon_us: float                   # 统计区间 [0, horizon)
    capacity_us: float                  # len(pool) * horizon, 单位 核·us
    busy_us: float
    forced_idle_us: float
    avoidable_idle_us: float
    segments: Tuple[IdleSegment, ...]   # 只收 avoidable 的段

    @property
    def idle_us(self) -> float:
        return self.forced_idle_us + self.avoidable_idle_us

    @property
    def utilization(self) -> float:
        return self.busy_us / self.capacity_us if self.capacity_us > 0 else 0.0

    @property
    def work_conserving(self) -> bool:
        """work-conserving 不变量: 没有一寸空闲是"有活却空着"."""
        return self.avoidable_idle_us <= _EPS


def _split_rank(resource: str) -> Tuple[str, str]:
    if "." in resource:
        head, tail = resource.split(".", 1)
        return head, tail
    return "", resource


def idle_decomposition(scheduled: Sequence, resource_prefix: str = "AIC:",
                       horizon_us: Optional[float] = None,
                       max_segments: int = 64,
                       max_waiting_sample: int = 4) -> Dict[str, IdleReport]:
    """按 rank 分组给出资源池的空闲分解.

    scheduled:        ScheduledEvent 序列 (rank_results[r]["events"])
    resource_prefix:  池的资源前缀, "AIC:" / "AIV0:" / "AIV1:"
    horizon_us:       统计区间上界; 缺省取**传入全部事件**的最大 end_us (≈ dag_end)
    max_segments:     最多收集几段 avoidable (只影响 segments, 不影响统计量)
    """
    # horizon 缺省取**传入的全部事件**的最大 end_us (≈ dag_end), 不是池内事件的 ——
    # 尾段 (epilogue/unpermute 在 AIV 上) 期间 AIC 确实没活干, 那段空闲属于 forced,
    # 工程上要算进核利用率里。只按池内事件取 horizon 会把尾段整段藏掉。
    all_end = max((e.end_us for e in scheduled if getattr(e, "end_us", None) is not None),
                  default=0.0)
    pools: Dict[str, List[str]] = {}
    members: Dict[str, List] = {}       # rank -> 该池的事件
    for e in scheduled:
        if not e.resources:
            continue
        # 池事件只占一个池资源; 多占的 (如 moe_stage_done 同时占 AIC 与 AIV) 按首个算
        hit = [r for r in e.resources if _split_rank(r)[1].startswith(resource_prefix)]
        if not hit:
            continue
        rank = _split_rank(hit[0])[0]
        pools.setdefault(rank, [])
        members.setdefault(rank, []).append((e, hit[0]))
        for r in hit:
            if r not in pools[rank]:
                pools[rank].append(r)

    out: Dict[str, IdleReport] = {}
    for rank, pool in pools.items():
        pool_sorted = tuple(sorted(pool))
        evs = members[rank]
        horizon = float(horizon_us) if horizon_us is not None else all_end
        if horizon <= 0:
            out[rank] = IdleReport(pool_sorted, 0.0, 0.0, 0.0, 0.0, 0.0, ())
            continue

        # 时间断点: 事件起止 + 就绪时刻 (就绪时刻会改变"有无就绪活"的真值)
        pts = {0.0, horizon}
        for e, _ in evs:
            pts.update((e.start_us, e.end_us, e.dependency_ready_us))
        marks = sorted(t for t in pts if 0.0 <= t <= horizon)

        # 事件在 [start, end) 占资源; 在 [dependency_ready, start) 处于"就绪未开始"
        starts: Dict[float, List] = {}
        ends: Dict[float, List] = {}
        ready_delta: Dict[float, int] = {}
        for e, res in evs:
            # 零时长事件 (dispatch_ready / moe_stage_done 的 duration 都是 0) 不进占用
            # 统计: 它们占的是测度零的时间, 对 busy/idle 的积分无贡献。而且同一时刻
            # "先处理 end 再处理 start"的次序会让它们入集后永不移除 —— 实测把 AIC busy
            # 从 5455.0 虚报到 6438.7。
            if e.end_us - e.start_us <= _EPS:
                continue
            starts.setdefault(e.start_us, []).append((e, res))
            ends.setdefault(e.end_us, []).append((e, res))
            if e.start_us - e.dependency_ready_us > _EPS:
                ready_delta[e.dependency_ready_us] = ready_delta.get(e.dependency_ready_us, 0) + 1
                ready_delta[e.start_us] = ready_delta.get(e.start_us, 0) - 1

        busy_res: set = set()
        ready_cnt = 0
        busy_us = forced = avoidable = 0.0
        segs: List[IdleSegment] = []
        for i, t in enumerate(marks[:-1]):
            # 先处理在 t 结束的 (资源即时释放), 再处理在 t 开始的
            for e, res in ends.get(t, ()):
                busy_res.discard(res)
            for e, res in starts.get(t, ()):
                busy_res.add(res)
            ready_cnt += ready_delta.get(t, 0)

            t_next = marks[i + 1]
            dt = t_next - t
            if dt <= _EPS:
                continue
            busy_us += len(busy_res) * dt
            idle = [r for r in pool_sorted if r not in busy_res]
            if not idle:
                continue
            core_us = len(idle) * dt
            if ready_cnt > 0:
                avoidable += core_us
                if len(segs) < max_segments:
                    waiting = tuple(
                        e.name for e, _ in evs
                        if e.dependency_ready_us <= t + _EPS < e.start_us
                    )[:max_waiting_sample]
                    segs.append(IdleSegment(t, t_next, tuple(idle), waiting))
            else:
                forced += core_us

        out[rank] = IdleReport(
            pool=pool_sorted, horizon_us=horizon,
            capacity_us=len(pool_sorted) * horizon,
            busy_us=busy_us, forced_idle_us=forced, avoidable_idle_us=avoidable,
            segments=tuple(segs))
    return out
