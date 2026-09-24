"""DAG events and multi-resource scheduler.

三层扩展 (默认全关, 行为与旧版逐事件一致):
  L0 逐边依赖延迟: Event.dep_latency_overrides [(dep_name, us)]
  L1 容量令牌:     Event.acquires/releases + schedule(capacities={...})
                   acquire 于事件 start 计入, release 于配对事件 end 归还;
                   容量不足时事件推迟到下一个归还时刻.
  L2 速率服务器信道: Event.channel_bytes [(name, bytes, entitled_rate)] +
                   schedule(channels={name: Channel}); 事件按应得速率名义时长
                   准入, 窗口内剩余带宽不足则降速或推迟, 无抢占无重定价.

性能结构 (不改变任何数值语义):
  - _TimeCounter: heap+cold 双栈按时间清算, count_le(t) 均摊 O(活跃集)
  - _ChannelState: 活跃区间按 end 清算, probe 只扫在飞区间 (~并发数) 而非全部历史
  - 前沿惰性探测: 先探 min t_base 事件得上界 A, 只探 t_base ≤ A 的候选
    (被剪枝事件 adjusted ≥ t_base > A, 选择结果精确等价)
"""
from __future__ import annotations

import heapq
from bisect import bisect_right, insort
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Event:
    name: str
    resources: Tuple[str, ...]
    duration_us: float
    deps: Tuple[str, ...] = ()
    order: int = 0
    meta: Dict[str, object] = field(default_factory=dict)
    # 依赖边传播延迟(us): flag 握手 RTT, 物理量(实测 WAIT_GMM1_BUFFER median)
    dep_latency_us: float = 0.0
    # L0 逐边覆盖: [(dep_name, latency_us)], 优先于 dep_latency_us
    dep_latency_overrides: Tuple[Tuple[str, float], ...] = ()
    # L1 容量令牌: (资源名, 个数); 本事件 start 占用, 由配对 release 事件 end 归还
    acquires: Tuple[Tuple[str, int], ...] = ()
    releases: Tuple[Tuple[str, int], ...] = ()
    # L2 信道需求: (信道名, 字节数, 应得速率 B/µs)
    channel_bytes: Tuple[Tuple[str, float, float], ...] = ()


@dataclass(frozen=True)
class Channel:
    """共享带宽信道: 聚合带宽 + 并发端口上限 + 单事件速率上限.

    bw_total:          聚合带宽 (B/µs), 所有并发事件速率之和的上界
    ports:             最大并发事件数 (0 = 不限)
    max_rate_per_event: 单事件速率上限 (B/µs, 0 = 不限).
                       物理含义: 单核 MTE/访存流不可能占满聚合带宽
                       (内存控制器多通道并行). 设为 bw_total/N 即 N 路均分.

    语义: 先到先得速率预留, 无抢占无重定价. 无争用时传输时长 = bytes/entitled
    (与闭式公式一致); 争用时按窗口内剩余带宽降速或推迟.
    """
    name: str
    bw_total: float
    ports: int = 0
    max_rate_per_event: float = 0.0

    def __post_init__(self) -> None:
        if self.bw_total <= 0:
            raise ValueError(f"channel {self.name}: bw_total must be positive")
        if self.ports < 0:
            raise ValueError(f"channel {self.name}: ports must be >= 0")
        if self.max_rate_per_event < 0:
            raise ValueError(f"channel {self.name}: max_rate_per_event must be >= 0")
        cap = self.max_rate_per_event or self.bw_total
        if cap > self.bw_total:
            raise ValueError(
                f"channel {self.name}: max_rate_per_event {cap} > bw_total {self.bw_total}"
            )


@dataclass(frozen=True)
class RestructureContext:
    """重构钩子的只读上下文 (策略函数不得修改返回的容器)."""
    time_us: float
    resource_free: Dict[str, float]      # 每资源当前空闲时刻
    resource_pending: Dict[str, int]     # 每资源未发射事件数
    pending: Dict[str, "Event"]          # 未发射事件视图 (name -> Event)
    committed_tail: Dict[str, str]       # 每资源最后提交的事件名
    channel_inflight: Dict[str, float]   # 每信道在飞速率和


@dataclass
class RestructureAction:
    """钩子返回的重构动作: inject/cancel/add_dep.
    契约: cancel 的每个事件必须以同名重新 inject (消费者依赖才能最终满足),
    否则调度以死图错误终止."""
    inject: List["Event"] = field(default_factory=list)
    cancel: List[str] = field(default_factory=list)
    add_dep: List[Tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ScheduledEvent:
    name: str
    resources: Tuple[str, ...]
    start_us: float
    end_us: float
    dependency_ready_us: float
    resource_ready_us: float
    dependency_wait_us: float
    resource_queue_us: float
    critical_parent: Optional[str]
    critical_reason: str
    order: int
    meta: Dict[str, object]
    # L1/L2 归因: 容量等待/信道等待/实际分得速率
    capacity_wait_us: float = 0.0
    channel_wait_us: float = 0.0
    channel_rate: Dict[str, float] = field(default_factory=dict)


def edge_latency(ev: "Event", dep: str) -> float:
    """L0: 逐边延迟, override 优先于事件级 dep_latency_us."""
    for name, lat in ev.dep_latency_overrides:
        if name == dep:
            return lat
    return ev.dep_latency_us


class _TimeCounter:
    """(time, k) 计数器: count_le(t) = Σ{k: time ≤ t}, 均摊 O(活跃集).

    heap 存未清算项; cold 按 time 升序存已清算项; 查询 t 回退时从 cold 尾部恢复.
    """

    __slots__ = ("heap", "cold", "total_k", "heap_k")

    def __init__(self) -> None:
        self.heap: List[Tuple[float, int]] = []
        self.cold: List[Tuple[float, int]] = []
        self.total_k = 0
        self.heap_k = 0

    def add(self, t: float, k: int) -> None:
        heapq.heappush(self.heap, (t, k))
        self.total_k += k
        self.heap_k += k

    def count_le(self, t: float) -> int:
        while self.cold and self.cold[-1][0] > t:
            item = self.cold.pop()
            heapq.heappush(self.heap, item)
            self.heap_k += item[1]
        while self.heap and self.heap[0][0] <= t:
            item = heapq.heappop(self.heap)
            self.cold.append(item)
            self.heap_k -= item[1]
        return self.total_k - self.heap_k


class _ChannelState:
    """速率服务器: 活跃区间 (start, end, rate), 按 end 清算.

    不变量: active 中所有区间 end > watermark; cold 按 end 升序.
    probe 只扫 active (≈ 在飞并发数), 而非全部历史区间.
    """

    def __init__(self, channel: Channel):
        self.ch = channel
        self.active: List[Tuple[float, float, float]] = []
        self.cold: List[Tuple[float, float, float]] = []
        self.watermark = float("-inf")
        # probe 记忆化: (t0, nbytes, entitled) -> (start, dur).
        # commit 时只失效窗口与新区间相交的表项 (probe 结果只依赖相交区间,
        # 提交只增争用 → 未相交的结果不变).
        self._memo: Dict[Tuple[float, float, float], Tuple[float, float]] = {}

    def commit(self, s: float, e: float, r: float) -> None:
        self.active.append((s, e, r))
        if self._memo:
            dead = [k for k, (st, d) in self._memo.items() if k[0] < e and st + d > s]
            for k in dead:
                del self._memo[k]

    def _prep(self, t0: float) -> None:
        if t0 < self.watermark:
            # 查询时间回退: 恢复 end > t0 的清算项 (cold 尾部是最大 end)
            while self.cold and self.cold[-1][1] > t0:
                self.active.append(self.cold.pop())
            self.watermark = t0
            return
        if t0 > self.watermark:
            keep: List[Tuple[float, float, float]] = []
            moved: List[Tuple[float, float, float]] = []
            for iv in self.active:
                (moved if iv[1] <= t0 else keep).append(iv)
            if moved:
                moved.sort(key=lambda iv: iv[1])
                self.cold.extend(moved)
                self.active = keep
            self.watermark = t0

    def _window_stats(self, t0: float, t1: float) -> Tuple[float, int]:
        """[t0,t1] 内最小剩余带宽与最大并发数 (保守: 任一重叠即全额计入)."""
        if t1 <= t0:
            t1 = t0 + 1e-9
        self._prep(t0)
        cands = [iv for iv in self.active if iv[0] < t1]
        if not cands:
            return self.ch.bw_total, 0
        pts = {t0, t1}
        for s, e, _ in cands:
            if t0 < s < t1:
                pts.add(s)
            if t0 < e < t1:
                pts.add(e)
        bounds = sorted(pts)
        f_min = self.ch.bw_total
        c_max = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            mid = (a + b) / 2.0
            rate_sum = 0
            cnt = 0
            for s, e, r in cands:
                if s < mid < e:
                    rate_sum += r
                    cnt += 1
            if rate_sum > 0 or cnt > c_max:
                f_min = min(f_min, self.ch.bw_total - rate_sum)
                c_max = max(c_max, cnt)
        return f_min, c_max

    def _min_active_end(self, t: float) -> Optional[float]:
        self._prep(t)
        return min((iv[1] for iv in self.active), default=None)

    def probe(self, t0: float, nbytes: float, entitled: float) -> Tuple[float, float]:
        """求 (start, dur): 不写状态. 名义时长 = nbytes/有效应得速率."""
        key = (t0, nbytes, entitled)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        result = self._probe_uncached(t0, nbytes, entitled)
        self._memo[key] = result
        return result

    def _probe_uncached(self, t0: float, nbytes: float, entitled: float) -> Tuple[float, float]:
        cap = self.ch.max_rate_per_event or self.ch.bw_total
        rate_cap = min(entitled, self.ch.bw_total, cap)
        if nbytes <= 0:
            return t0, 0.0
        t = t0
        dur = nbytes / rate_cap
        for _ in range(8):
            f_min, c_max = self._window_stats(t, t + dur)
            if self.ch.ports and c_max >= self.ch.ports:
                nxt = self._min_active_end(t)
                if nxt is None:
                    break
                t = nxt
                continue
            r = min(rate_cap, f_min)
            if r <= 1e-9:
                nxt = self._min_active_end(t)
                if nxt is None:
                    break
                t = nxt
                continue
            new_dur = nbytes / r
            if abs(new_dur - dur) < 1e-9:
                return t, new_dur
            dur = new_dur
        return t, dur


class MultiResourceScheduler:
    """Precedence-aware serial schedule generator.

    Program order on AIC/AIV0/AIV1 is represented explicitly as dependencies.
    Among currently ready events, schedule the event with the smallest earliest
    feasible start.  Thus the result does not depend on the phase in which
    events happened to be appended to the Python list.

    capacities: {容量资源名: 令牌容量}. 事件 acquires 超容量时推迟到归还时刻.
    channels:   {信道名: Channel}. 事件 channel_bytes 按速率服务器准入.
    """

    def schedule(
        self,
        events: Sequence[Event],
        capacities: Optional[Dict[str, int]] = None,
        channels: Optional[Dict[str, Channel]] = None,
        restructure=None,
        restructure_limit: Optional[int] = None,
        policy=None,
    ) -> Tuple[float, List[ScheduledEvent]]:
        """restructure: 每次事件提交后调用的重构钩子
        hook(RestructureContext) -> RestructureAction.
        cancel 的事件必须同名 reinject (见 RestructureAction 契约);
        注入事件数上限 restructure_limit (默认 4×初始事件数, 防策略失控).
        """
        capacities = dict(capacities or {})
        channels = dict(channels or {})
        if policy is None:
            from .policies import EarliestStart
            policy = EarliestStart()
        chan_state = {name: _ChannelState(ch) for name, ch in channels.items()}

        by_name: Dict[str, Event] = {}
        for ev in events:
            if ev.name in by_name:
                raise ValueError(f"duplicate event name: {ev.name}")
            by_name[ev.name] = ev

        indegree: Dict[str, int] = {name: 0 for name in by_name}
        children: Dict[str, List[str]] = {name: [] for name in by_name}
        for ev in events:
            for dep in ev.deps:
                if dep not in by_name:
                    raise ValueError(f"event {ev.name} depends on missing {dep}")
                indegree[ev.name] += 1
                children[dep].append(ev.name)

        # 静态校验: acquire 不得超过容量, 引用必须已声明
        for ev in events:
            for res, k in ev.acquires:
                if res not in capacities:
                    raise ValueError(f"event {ev.name} acquires unknown capacity resource {res}")
                if k > capacities[res]:
                    raise ValueError(
                        f"event {ev.name} acquires {k} > capacity {capacities[res]} of {res}"
                    )
            for cname, _, _ in ev.channel_bytes:
                if cname not in chan_state:
                    raise ValueError(f"event {ev.name} references unknown channel {cname}")

        ready = {name for name, deg in indegree.items() if deg == 0}
        end_by_name: Dict[str, float] = {}
        resource_free: Dict[str, float] = {}
        resource_last_event: Dict[str, str] = {}
        # L1 台账: 计数器按时间清算
        acq_ctr: Dict[str, _TimeCounter] = {}
        rel_ctr: Dict[str, _TimeCounter] = {}
        rel_times: Dict[str, List[float]] = {}   # 有序归还时刻 (供容量推进枚举)
        scheduled: List[ScheduledEvent] = []
        pending_names: set = set(by_name.keys())
        canceled: set = set()
        injected_count = [0]
        limit = restructure_limit if restructure_limit is not None else 4 * len(events)

        def _register(ev: Event) -> None:
            by_name[ev.name] = ev
            indegree[ev.name] = sum(1 for d in ev.deps if d not in end_by_name)
            for d in ev.deps:
                if d not in by_name:
                    raise ValueError(f"event {ev.name} depends on missing {d}")
                children[d].append(ev.name)
            if indegree[ev.name] == 0:
                ready.add(ev.name)
            pending_names.add(ev.name)

        def _apply_restructure(t_now: float, last_ev: Optional[Event]) -> None:
            if restructure is None:
                return
            res_free = dict(resource_free)
            pend_cnt = defaultdict(int)
            pend_view = {n: by_name[n] for n in pending_names if n not in end_by_name and n not in canceled}
            for n, e in pend_view.items():
                for r in e.resources:
                    pend_cnt[r] += 1
            ctx = RestructureContext(
                time_us=t_now, resource_free=res_free, resource_pending=dict(pend_cnt),
                pending=pend_view,
                committed_tail=dict(resource_last_event),
                channel_inflight={name: sum(r for _, r, _ in st_.active)
                                  for name, st_ in chan_state.items()})
            act = restructure(ctx)
            for nm in act.cancel:
                if nm in end_by_name:
                    raise ValueError(f"restructure: 不能取消已提交事件 {nm}")
                if nm in ready:
                    ready.discard(nm)
                pending_names.discard(nm)
                canceled.add(nm)
            for d_pair in act.add_dep:
                tgt, dep = d_pair
                if tgt in end_by_name:
                    raise ValueError(f"restructure: 不能给已提交事件 {tgt} 追加依赖")
                if dep not in by_name:
                    raise ValueError(f"restructure: add_dep 引用未知事件 {dep}")
                by_name[tgt].deps = tuple(dict.fromkeys(by_name[tgt].deps + (dep,)))
                if tgt in pending_names:
                    indegree[tgt] += 1
                    children[dep].append(tgt)
                    if tgt in ready:
                        ready.discard(tgt)
            for ev in act.inject:
                if injected_count[0] >= limit:
                    raise ValueError("restructure: 注入事件数超上限 (策略失控保护)")
                if ev.name in by_name and ev.name not in canceled:
                    raise ValueError(f"restructure: 注入与现存事件重名 {ev.name}")
                was_canceled = ev.name in canceled
                if was_canceled:
                    canceled.discard(ev.name)
                    for d in by_name[ev.name].deps:
                        if ev.name in children[d]:
                            children[d].remove(ev.name)
                injected_count[0] += 1
                _register(ev)

        def outstanding(res: str, t: float) -> int:
            a = acq_ctr.get(res)
            r = rel_ctr.get(res)
            return (a.count_le(t) if a else 0) - (r.count_le(t) if r else 0)

        def capacity_feasible(ev: Event, t: float) -> Tuple[bool, float]:
            """t 时刻容量是否够; 不够时沿未来归还时刻找最早可行点."""
            for res, k in ev.acquires:
                if outstanding(res, t) + k <= capacities[res]:
                    continue
                times = rel_times.get(res, ())
                ok_t = float("inf")
                for i in range(bisect_right(times, t), len(times)):
                    te = times[i]
                    if outstanding(res, te) + k <= capacities[res]:
                        ok_t = te
                        break
                if ok_t == float("inf"):
                    return False, float("inf")
                t = ok_t
            return True, t

        def probe(ev: Event, t_base: float) -> Tuple[float, float, float, float, Dict[str, float]]:
            """联合准入探测 (不写状态). 返回 (start, duration, ch_wait, cap_wait, rates)."""
            t = t_base
            ch_wait = 0.0
            cap_wait = 0.0
            rates: Dict[str, float] = {}
            for _ in range(4):
                if ev.channel_bytes:
                    ch_t = t
                    ch_dur = max(0.0, ev.duration_us)
                    for cname, nbytes, entitled in ev.channel_bytes:
                        st, d = chan_state[cname].probe(ch_t, nbytes, entitled)
                        ch_t = max(ch_t, st)
                        ch_dur = max(ch_dur, d)
                        rates[cname] = nbytes / d if d > 0 else entitled
                    if ch_t > t:
                        ch_wait += ch_t - t
                        t = ch_t
                    dur = ch_dur
                else:
                    dur = max(0.0, ev.duration_us)
                ok, t_cap = capacity_feasible(ev, t)
                if not ok:
                    return float("inf"), dur, ch_wait, cap_wait, rates
                if t_cap > t:
                    cap_wait += t_cap - t
                    t = t_cap
                    continue
                if ev.channel_bytes and any(
                    chan_state[c].probe(t, nb, en)[0] > t for c, nb, en in ev.channel_bytes
                ):
                    continue
                return t, dur, ch_wait, cap_wait, rates
            return t, dur, ch_wait, cap_wait, rates

        while ready:
            # ---- pass 1: 全体 ready 的 t_base, 找最小 ----
            tbase: Dict[str, float] = {}
            min_name: Optional[str] = None
            min_tb = float("inf")
            for name in ready:
                ev = by_name[name]
                dep_ready = max(
                    (end_by_name[d] + edge_latency(ev, d) for d in ev.deps), default=0.0
                )
                res_ready = max((resource_free.get(r, 0.0) for r in ev.resources), default=0.0)
                tb = max(dep_ready, res_ready)
                tbase[name] = tb
                if tb < min_tb:
                    min_tb = tb
                    min_name = name

            # ---- pass 2: 前沿惰性探测, 只探 t_base ≤ 当前最优 start 的候选 ----
            best_key: Optional[Tuple[float, int, str]] = None
            best: Optional[Tuple[str, float, float, float, float, Dict[str, float]]] = None

            def consider(name: str) -> None:
                nonlocal best_key, best
                ev = by_name[name]
                if not ev.acquires and not ev.channel_bytes:
                    start, dur = tbase[name], ev.duration_us
                    payload = (0.0, 0.0, {})
                else:
                    start, dur, ch_w, cap_w, rates = probe(ev, tbase[name])
                    if start == float("inf"):
                        return
                    payload = (ch_w, cap_w, rates)
                key = policy.event_key(ev, start, tbase, end_by_name)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (name, start, dur, payload[0], payload[1], payload[2])

            assert min_name is not None
            consider(min_name)
            if best_key is None:
                # min t_base 事件容量死锁: 无剪枝上界, 全量探测
                for name in ready:
                    if name != min_name:
                        consider(name)
            else:
                for name in ready:
                    if name != min_name and tbase[name] <= best_key[0]:
                        consider(name)

            if best is None:
                blocked = sorted(ready)
                raise ValueError(f"capacity deadlock: no feasible event among ready={blocked[:8]}")

            name, start, dur, ch_wait, cap_wait, rates = best
            ev = by_name[name]
            ready.remove(name)
            end = start + max(0.0, dur)

            dep_parent = max(ev.deps, key=lambda d: end_by_name[d], default=None)
            dep_ready = (
                max(end_by_name[d] + edge_latency(ev, d) for d in ev.deps) if ev.deps else 0.0
            )
            blocking_resource = (
                max(ev.resources, key=lambda r: resource_free.get(r, 0.0), default=None)
                if ev.resources else None
            )
            res_ready = resource_free.get(blocking_resource, 0.0) if blocking_resource else 0.0

            if dep_ready >= res_ready and dep_parent is not None:
                critical_parent = dep_parent
                critical_reason = "dependency"
            elif blocking_resource is not None and blocking_resource in resource_last_event:
                critical_parent = resource_last_event[blocking_resource]
                critical_reason = f"resource:{blocking_resource}"
            else:
                critical_parent = None
                critical_reason = "root"
            if cap_wait > 0.0:
                critical_reason = "capacity"
            elif ch_wait > 0.0:
                critical_reason = "channel"

            end_by_name[name] = end
            for resource in ev.resources:
                resource_free[resource] = end
                resource_last_event[resource] = name
            for res, k in ev.acquires:
                acq_ctr.setdefault(res, _TimeCounter()).add(start, k)
            for res, k in ev.releases:
                rel_ctr.setdefault(res, _TimeCounter()).add(end, k)
                insort(rel_times.setdefault(res, []), end)
            for cname, nbytes, _ in ev.channel_bytes:
                rate = rates.get(cname, 0.0)
                chan_state[cname].commit(start, start + max(dur, 1e-12),
                                         min(nbytes / max(dur, 1e-12), self._rate_cap(channels[cname])))

            scheduled.append(
                ScheduledEvent(
                    name=name,
                    resources=ev.resources,
                    start_us=start,
                    end_us=end,
                    dependency_ready_us=dep_ready,
                    resource_ready_us=res_ready,
                    dependency_wait_us=max(0.0, dep_ready - res_ready),
                    resource_queue_us=max(0.0, res_ready - dep_ready),
                    critical_parent=critical_parent,
                    critical_reason=critical_reason,
                    order=ev.order,
                    meta=dict(ev.meta),
                    capacity_wait_us=cap_wait,
                    channel_wait_us=ch_wait,
                    channel_rate=rates,
                )
            )

            for child in children[name]:
                indegree[child] -= 1
                if indegree[child] == 0 and child not in canceled:
                    ready.add(child)
            pending_names.discard(name)
            _apply_restructure(start, ev)

        # 重构契约: cancel 的一切必须同名 reinject (canceled 集合最终必须为空)
        if canceled:
            raise ValueError(f"restructure: cancel 后未同名 reinject: {sorted(canceled)[:8]}")
        # 事件守恒: 所有入图事件 (含注入) 必须全部提交
        if len(scheduled) != len(by_name):
            remaining = [name for name, deg in indegree.items() if deg > 0]
            raise ValueError(f"event DAG contains a cycle; unresolved={remaining[:12]}")

        scheduled.sort(key=lambda e: (e.start_us, e.end_us, e.order, e.name))
        total = max((e.end_us for e in scheduled), default=0.0)
        return total, scheduled

    @staticmethod
    def _rate_cap(ch: Channel) -> float:
        return min(ch.bw_total, ch.max_rate_per_event or ch.bw_total)
