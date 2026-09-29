"""第 2 层: 通用离散事件调度引擎 — 多资源互斥 + 计数信号量 + 速率服务器.

输入: 事件列表 + 容量表 + 信道表 + 调度策略.

性能结构:
  - _TimeCounter: heap+cold 双栈按时间清算, count_le(t) 均摊 O(活跃集)
  - _ChannelState: 活跃区间按 end 清算, probe 只扫在飞区间 (~并发数) 而非全部历史
  - 前沿惰性探测: 先探 min t_base 事件得上界 A, 只探 t_base ≤ A 的候选
    (被剪枝事件 adjusted ≥ t_base > A, 选择结果精确等价)
  - 就绪集增量维护: t_base 在事件入就绪集时算一次, 此后只在其占用的资源被
    提交时重算; 最小 t_base 由惰性失效堆给出, 不再每轮全量扫描就绪集
  - 探测结果留存: 联合准入探测只取决于 t_base 与事件引用的容量/信道台账,
    二者未变则结果沿用; 提交只让引用了被写台账的事件重新探测
"""
from __future__ import annotations

import heapq
from bisect import bisect_right, insort
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .events import Channel, Event, RestructureAction, RestructureContext, ScheduledEvent, edge_latency


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
        """[t0,t1] 内最小剩余带宽与最大并发数 """
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

    capacities: {资源名: 计数信号量容量}. 事件 acquires 超容量时推迟到归还时刻.
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

        # ---- 就绪集 (增量维护) ----
        # ready 用 dict 保持入集顺序: 遍历序与哈希种子无关, 探测次序确定.
        # tbase[name] = max(依赖就绪, 资源空闲), 键集合恒等于 ready.
        # 堆项 (t_base, order, name, 代次); 代次与 ready_gen 不符即失效, 取堆顶时丢弃.
        # 无容量/信道需求的事件 start = t_base, 进 simple_heap; 其余需联合准入
        # 探测, 进 probe_heap (待探测, 按 t_base 排序).
        # EarliestStart 快路径下, 探测过的事件移入 resolved_heap (按 start 排序),
        # 结果存 resolved; 其 t_base 或所引用的台账 (容量计数/信道) 变化时退回
        # probe_heap 重新探测. 不可行 (start=inf) 的只存 resolved, 不进堆.
        ready: Dict[str, None] = {}
        tbase: Dict[str, float] = {}
        dep_ready_at: Dict[str, float] = {}
        ready_gen: Dict[str, int] = {}
        ready_by_res: Dict[str, set] = {}
        simple_heap: List[Tuple[float, int, str, int]] = []
        probe_heap: List[Tuple[float, int, str, int]] = []
        resolved_heap: List[Tuple[float, int, str, int]] = []
        resolved: Dict[str, Tuple[float, float, float, float, Dict[str, float]]] = {}
        ledger_watch: Dict[str, Dict[str, None]] = {}   # 台账名 -> 引用它的就绪事件
        gen_counter = [0]

        def _ledgers(ev: Event) -> List[str]:
            return [res for res, _ in ev.acquires] + [c[0] for c in ev.channel_bytes]

        def _push_ready(name: str, ev: Event, tb: float) -> None:
            gen_counter[0] += 1
            ready_gen[name] = gen_counter[0]
            tbase[name] = tb
            resolved.pop(name, None)
            heapq.heappush(probe_heap if (ev.acquires or ev.channel_bytes) else simple_heap,
                           (tb, ev.order, name, gen_counter[0]))

        def _ledger_written(ledger: str) -> None:
            """台账有新写入: 引用它的已探测事件结果失效, 退回待探测."""
            for n in ledger_watch.get(ledger, ()):
                if n in resolved:
                    _push_ready(n, by_name[n], tbase[n])

        def _enter_ready(name: str) -> None:
            ev = by_name[name]
            dep_ready = max(
                (end_by_name[d] + edge_latency(ev, d) for d in ev.deps), default=0.0
            )
            res_ready = max((resource_free.get(r, 0.0) for r in ev.resources), default=0.0)
            ready[name] = None
            dep_ready_at[name] = dep_ready
            for r in ev.resources:
                ready_by_res.setdefault(r, set()).add(name)
            for led in _ledgers(ev):
                ledger_watch.setdefault(led, {})[name] = None
            _push_ready(name, ev, max(dep_ready, res_ready))

        def _leave_ready(name: str) -> None:
            if name not in ready:
                return
            del ready[name]
            del tbase[name]
            del dep_ready_at[name]
            del ready_gen[name]
            resolved.pop(name, None)
            ev = by_name[name]
            for r in ev.resources:
                ready_by_res[r].discard(name)
            for led in _ledgers(ev):
                ledger_watch[led].pop(name, None)

        def _resource_committed(resource: str) -> None:
            """resource 的空闲时刻已变: 重算占用它的就绪事件的 t_base."""
            for n in ready_by_res.get(resource, ()):
                ev_n = by_name[n]
                res_ready = max((resource_free.get(r, 0.0) for r in ev_n.resources),
                                default=0.0)
                tb = max(dep_ready_at[n], res_ready)
                if tb != tbase[n]:
                    _push_ready(n, ev_n, tb)

        def _heap_top(heap: List[Tuple[float, int, str, int]]):
            while heap:
                top = heap[0]
                if ready_gen.get(top[2]) == top[3]:
                    return top
                heapq.heappop(heap)
            return None

        def _register(ev: Event) -> None:
            by_name[ev.name] = ev
            indegree[ev.name] = sum(1 for d in ev.deps if d not in end_by_name)
            for d in ev.deps:
                if d not in by_name:
                    raise ValueError(f"event {ev.name} depends on missing {d}")
                children[d].append(ev.name)
            if indegree[ev.name] == 0:
                _enter_ready(ev.name)
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
                _leave_ready(nm)
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
                    if dep in end_by_name:
                        # dep 已提交: 依赖已满足. 不加 indegree — 已提交事件的
                        # children 不会再被访问, 加了计数就永远减不回 (假环).
                        if tgt in ready:
                            # 依赖集变了: 缓存的依赖就绪时刻失效, 重新入集
                            _leave_ready(tgt)
                            _enter_ready(tgt)
                        continue
                    indegree[tgt] += 1
                    children[dep].append(tgt)
                    _leave_ready(tgt)
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

        from .policies import EarliestStart
        # 快路径只对 EarliestStart 本类启用: 排序键 (start, order, name) 与堆序一致.
        # 子类/其他策略的键由 event_key 决定, 走通用路径.
        fast = type(policy) is EarliestStart
        for name, deg in indegree.items():
            if deg == 0:
                _enter_ready(name)

        while ready:
            best_key: Optional[tuple] = None
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

            top_simple = _heap_top(simple_heap)
            top_probe = _heap_top(probe_heap)
            if fast:
                # 排序键 (start, order, name). 无探测事件 start = t_base, 已探测事件
                # start 留存, 两个堆顶各是本类最优. 待探测事件 start ≥ t_base, 只有
                # t_base ≤ 当前最优 start 的才可能胜出, 逐个探测并收紧上界;
                # 尚无可行候选时上界为 +inf, 全量探测 (容量死锁判定).
                top_resolved = _heap_top(resolved_heap)
                bound = float("inf")
                if top_simple is not None:
                    bound = top_simple[0]
                if top_resolved is not None and top_resolved[0] < bound:
                    bound = top_resolved[0]
                while top_probe is not None and top_probe[0] <= bound:
                    heapq.heappop(probe_heap)
                    cand = top_probe[2]
                    result = probe(by_name[cand], tbase[cand])
                    resolved[cand] = result
                    if result[0] != float("inf"):
                        heapq.heappush(resolved_heap, (result[0],) + top_probe[1:])
                        if result[0] < bound:
                            bound = result[0]
                    top_probe = _heap_top(probe_heap)
                top_resolved = _heap_top(resolved_heap)
                if top_simple is not None and (
                        top_resolved is None or top_simple[:3] < top_resolved[:3]):
                    best = (top_simple[2], top_simple[0],
                            by_name[top_simple[2]].duration_us, 0.0, 0.0, {})
                elif top_resolved is not None:
                    best = (top_resolved[2],) + resolved[top_resolved[2]]
            else:
                # ---- 通用策略: 先探 min t_base 事件得上界, 再比较候选 ----
                if top_simple is None or (top_probe is not None and top_probe < top_simple):
                    min_name = top_probe[2]
                else:
                    min_name = top_simple[2]
                consider(min_name)
                if best_key is None:
                    # min t_base 事件容量死锁: 无剪枝上界, 全量探测
                    for name in list(ready):
                        if name != min_name:
                            consider(name)
                elif getattr(policy, "prune_by_start", False):
                    # 剪枝仅对首项=开始时刻的策略有效:
                    # tbase 是 start 的下界, tbase > best.start 的候选不可能更早.
                    best_start = best[1]
                    for name in list(ready):
                        if name != min_name and tbase[name] <= best_start:
                            consider(name)
                else:
                    # 优先级/slack 等非时间首项: 时间剪枝无效, 全量比较
                    for name in list(ready):
                        if name != min_name:
                            consider(name)

            if best is None:
                blocked = sorted(ready)
                raise ValueError(f"capacity deadlock: no feasible event among ready={blocked[:8]}")

            name, start, dur, ch_wait, cap_wait, rates = best
            ev = by_name[name]
            _leave_ready(name)
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
            for resource in ev.resources:
                _resource_committed(resource)
            for res, k in ev.acquires:
                acq_ctr.setdefault(res, _TimeCounter()).add(start, k)
            for res, k in ev.releases:
                rel_ctr.setdefault(res, _TimeCounter()).add(end, k)
                insort(rel_times.setdefault(res, []), end)
            for cname, nbytes, _ in ev.channel_bytes:
                rate = rates.get(cname, 0.0)
                chan_state[cname].commit(start, start + max(dur, 1e-12),
                                         min(nbytes / max(dur, 1e-12), self._rate_cap(channels[cname])))
            for ledger in dict.fromkeys([res for res, _ in ev.acquires]
                                        + [res for res, _ in ev.releases]
                                        + [c[0] for c in ev.channel_bytes]):
                _ledger_written(ledger)

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
                    _enter_ready(child)
            pending_names.discard(name)
            _apply_restructure(start, ev)

        # cancel 的一切必须同名 reinject (canceled 集合最终必须为空)
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
