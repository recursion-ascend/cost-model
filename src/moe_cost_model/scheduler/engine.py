"""第 2 层: 通用离散事件调度引擎 — 多资源互斥 + 计数信号量.

输入: 事件列表 + 容量表 + 调度策略.

信道 (速率服务器) 已于 2026-10-03 停用: Event.channel_bytes 仍然申报字节, 但只做
访存量统计, **不参与准入, 不影响任何时长**。去掉的理由见 README 的信道一节。

性能结构:
  - _TimeCounter: heap+cold 双栈按时间清算, count_le(t) 均摊 O(活跃集)
  - 前沿惰性探测: 先探 min t_base 事件得上界 A, 只探 t_base ≤ A 的候选
    (被剪枝事件 adjusted ≥ t_base > A, 选择结果精确等价)
  - 就绪集增量维护: t_base 在事件入就绪集时算一次, 此后只在其占用的资源被
    提交时重算; 最小 t_base 由惰性失效堆给出, 不再每轮全量扫描就绪集
  - 探测结果留存: 联合准入探测只取决于 t_base 与事件引用的容量台账,
    未变则结果沿用; 提交只让引用了被写台账的事件重新探测
"""
from __future__ import annotations

import heapq
import re
from bisect import bisect_right, insort
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .events import (Event, RestructureContext, ScheduledEvent,
                     edge_latency, pool_key)

#: 按核 token 的核号后缀 ("R0.UB:gmm1act:c7" 里的 "c7")
_TOK_CORE_SUFFIX = re.compile(r"c\d+$")


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


class MultiResourceScheduler:
    """Precedence-aware serial schedule generator.

    Program order on AIC/AIV0/AIV1 is represented explicitly as dependencies.
    Among currently ready events, schedule the event with the smallest earliest
    feasible start.  Thus the result does not depend on the phase in which
    events happened to be appended to the Python list.

    capacities: {资源名: 计数信号量容量}. 事件 acquires 超容量时推迟到归还时刻.
    事件的 channel_bytes 只是访存量申报, 不参与准入 (信道模型已停用)。
    """

    def schedule(
        self,
        events: Sequence[Event],
        capacities: Optional[Dict[str, int]] = None,
        restructure=None,
        restructure_limit: Optional[int] = None,
        policy=None,
        pools: Optional[Dict[str, Sequence[str]]] = None,
    ) -> Tuple[float, List[ScheduledEvent]]:
        """pools: {池名: 成员资源名序列}. 事件 resources 里写 "<池名>:*" 的占位符,
        调度器在**派发时刻**解析成最早空闲的成员 (晚绑定); Event.colocate_with 指定
        必须同核号的锚点事件。pools 为空时行为与静态绑定逐位一致。

        restructure: 每次事件提交后调用的重构钩子
        hook(RestructureContext) -> RestructureAction.
        cancel 的事件必须同名 reinject (见 RestructureAction 契约);
        注入事件数上限 restructure_limit (默认 4×初始事件数, 防策略失控).
        """
        capacities = dict(capacities or {})
        if policy is None:
            from .policies import EarliestStart
            policy = EarliestStart()

        # ---- L3 晚绑定: 池成员表 + 已绑定核号 ----
        pool_members: Dict[str, Tuple[str, ...]] = {
            k: tuple(v) for k, v in (pools or {}).items()}
        for pk, mem in pool_members.items():
            if not mem:
                raise ValueError(f"pool {pk} 的成员表为空")
        bound_core: Dict[str, str] = {}      # 事件名 -> 绑定的核号后缀
        charged_once: set = set()            # 已计过的 (once_per_core 键, 核号)

        def _core_of(resource: str) -> str:
            return resource.rsplit(":", 1)[1] if ":" in resource else resource

        def _candidates(ev: Event, resource: str) -> Tuple[str, ...]:
            """resource 的候选成员. 具体资源返回自身; 占位符返回池成员, 若本事件
            colocate_with 的锚点已绑定则收窄为那一个核号。"""
            pk = pool_key(resource)
            if pk is None:
                return (resource,)
            mem = pool_members.get(pk)
            if mem is None:
                raise ValueError(f"event {ev.name} 引用未声明的资源池 {pk}")
            anchor = ev.colocate_with
            if anchor is not None and anchor in bound_core:
                want = bound_core[anchor]
                same = tuple(r for r in mem if _core_of(r) == want)
                if not same:
                    raise ValueError(
                        f"event {ev.name} 要与 {anchor} 共位于核 {want}, 但池 {pk} 无此成员")
                return same
            return mem

        def _res_ready(ev: Event) -> float:
            """事件的资源就绪时刻. 具体资源取 max (全部都要空闲);
            占位符取候选里的 min (任一成员空闲即可) —— 这就是晚绑定的全部语义差别。"""
            pick = pool_pick.get(ev.name)
            t = 0.0
            for r in ev.resources:
                cand = _candidates(ev, r)
                if len(cand) == 1 and pool_key(r) is None:
                    t = max(t, resource_free.get(cand[0], 0.0))
                elif pick is not None and pool_key(r) is not None:
                    # 已选定核: 用它的空闲时刻, 与容量准入/绑定保持同一个核
                    same = [c for c in cand if _core_of(c) == pick]
                    t = max(t, min(resource_free.get(c, 0.0) for c in (same or cand)))
                else:
                    t = max(t, min(resource_free.get(c, 0.0) for c in cand))
            return t

        def _pick_core(ev: Event, t_dep: float) -> Optional[str]:
            """在候选核里选**能最早开始**的那一个.

            为什么要一次选定: 核空闲时刻与按核槽余量是两个约束, 分别取"最早空的核"和
            "最早有槽的核"可能指向不同的核, 于是算出的 start 没有任何单个核真能满足。
            这里按核算出各自的 max(依赖, 该核空闲, 该核槽可用), 再取最小的那个核。
            """
            pooled = None
            for r in ev.resources:
                if pool_key(r) is not None:
                    pooled = r
                    break
            if pooled is None:
                return None
            cand = _candidates(ev, pooled)
            best_t, best_c = float("inf"), None
            for member in cand:
                c = _core_of(member)
                t = max(t_dep, resource_free.get(member, 0.0))
                for res0, k in ev.acquires:
                    if res0.endswith("c*"):
                        t = max(t, _earliest_ok(res0[:-1] + c, k, t))
                if t < best_t or (t == best_t and (best_c is None or member < best_c)):
                    best_t, best_c = t, member
            return _core_of(best_c) if best_c is not None else None

        def _bind(ev: Event, t: Optional[float] = None) -> Tuple[str, ...]:
            """派发时刻把占位符绑定到最早空闲的候选成员 (并列时取名字序, 保证确定性).

            给了 t 就先筛掉"该核的按核信号量在 t 时刻没余量"的候选 —— 否则会绑到一个
            放不下这个活的核上, 与 capacity_feasible 的判断不一致。
            """
            out = []
            for r in ev.resources:
                cand = _candidates(ev, r)
                if len(cand) == 1:
                    out.append(cand[0])
                    continue
                pick = pool_pick.get(ev.name)
                if pick is not None:
                    same = [c for c in cand if _core_of(c) == pick]
                    if same:
                        cand = same
                elif t is not None:
                    ok = [c for c in cand if _slots_ok(ev, _core_of(c), t)]
                    if ok:
                        cand = ok
                out.append(min(cand, key=lambda c: (resource_free.get(c, 0.0), c)))
            return tuple(out)

        def _remap_tokens(toks, core: Optional[str]):
            """队列 token 的核号占位符 "c*" 换成绑定核号 ("Q:aic:c*" -> "Q:aic:c7")."""
            if core is None:
                return tuple(toks)
            return tuple((t[:-1] + core if t.endswith("c*") else t, k) for t, k in toks)

        #: 池化事件已选定的核号 (事件名 -> 核号). 与 tbase 同生命周期: _push_ready 时
        #: 重算。选定后 res_ready / 容量准入 / _bind 全用它, 三者因此必然一致。
        pool_pick: Dict[str, str] = {}

        def _tok_cores(ev: Event) -> Tuple[str, ...]:
            """本事件的 "c*" token 可以落到哪些核号上 —— 与它的核资源候选一致.

            共位事件 (ACT 跟随 GMM1) 只有锚点那一个核号可选。
            容量准入发生在绑核之前, 所以这里给出**全部**候选, 由 capacity_feasible
            判"任一个有余量即可", 再由 _bind(ev, start) 绑到确实有余量的那个。
            """
            if not has_pools:
                return ()
            anchor = ev.colocate_with
            if anchor is not None and anchor in bound_core:
                return (bound_core[anchor],)
            pick = pool_pick.get(ev.name)
            if pick is not None:
                return (pick,)
            for r in ev.resources:
                if pool_key(r) is not None:
                    return tuple(_core_of(c) for c in _candidates(ev, r))
            return ()

        has_pools = bool(pool_members)

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

        # 关键路径长度 (事件自身开始算起, 到任一 sink 的最长链, 含边延迟):
        # 只在策略索要时算一次, 反向拓扑一遍, O(V+E)。写进 meta 供策略读取。
        if getattr(policy, "needs_remaining_path", False):
            order_rev: List[str] = []
            deg = dict(indegree)
            stack = [n for n, d in deg.items() if d == 0]
            while stack:
                n = stack.pop()
                order_rev.append(n)
                for ch in children[n]:
                    deg[ch] -= 1
                    if deg[ch] == 0:
                        stack.append(ch)
            remaining: Dict[str, float] = {}
            for n in reversed(order_rev):
                ev = by_name[n]
                tail = 0.0
                for ch in children[n]:
                    tail = max(tail, remaining[ch] + edge_latency(by_name[ch], n))
                remaining[n] = max(0.0, ev.duration_us) + tail
            for n, v in remaining.items():
                by_name[n].meta["remaining_path_us"] = v

        # 静态校验: acquire 不得超过容量, 引用必须已声明
        for ev in events:
            for res, k in ev.acquires:
                if res.endswith("c*"):
                    # 晚绑定占位: 真核号派发时才定, 各核同容量 -> 用 c0 做静态校验
                    res = res[:-1] + "0"
                if res not in capacities:
                    raise ValueError(f"event {ev.name} acquires unknown capacity resource {res}")
                if k > capacities[res]:
                    raise ValueError(
                        f"event {ev.name} acquires {k} > capacity {capacities[res]} of {res}"
                    )

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
        # 无容量需求的事件 start = t_base, 进 simple_heap; 其余需容量准入探测,
        # 进 probe_heap (待探测, 按 t_base 排序).
        # EarliestStart 快路径下, 探测过的事件移入 resolved_heap (按 start 排序),
        # 结果存 resolved; 其 t_base 或所引用的容量台账变化时退回 probe_heap
        # 重新探测. 不可行 (start=inf) 的只存 resolved, 不进堆.
        ready: Dict[str, None] = {}
        tbase: Dict[str, float] = {}
        dep_ready_at: Dict[str, float] = {}
        ready_gen: Dict[str, int] = {}
        ready_by_res: Dict[str, set] = {}
        simple_heap: List[Tuple[float, int, str, int]] = []
        probe_heap: List[Tuple[float, int, str, int]] = []
        resolved_heap: List[Tuple[float, int, str, int]] = []
        resolved: Dict[str, Tuple[float, float, float]] = {}
        ledger_watch: Dict[str, Dict[str, None]] = {}   # 台账名 -> 引用它的就绪事件
        gen_counter = [0]

        def _ledgers(ev: Event) -> List[str]:
            return [res for res, _ in ev.acquires]

        def _push_ready(name: str, ev: Event, tb: float) -> None:
            gen_counter[0] += 1
            ready_gen[name] = gen_counter[0]
            tbase[name] = tb
            resolved.pop(name, None)
            heapq.heappush(probe_heap if ev.acquires else simple_heap,
                           (tb, ev.order, name, gen_counter[0]))

        def _ledger_written(ledger: str) -> None:
            """台账有新写入: 引用它的已探测事件结果失效, 退回待探测.

            按核槽的余量变了, 最优核也可能换人 —— 所以同时重选并重算 t_base。

            注意键的两种形态: 就绪登记用的是**占位名** (UB:gmm1act:c*, 来自 Event),
            而写入通知用的是**绑定名** (UB:gmm1act:c7, 来自提交路径)。只查绑定名会让
            watch 永不触发 —— 等槽的事件再也不被重新探测, 表现为假的 capacity deadlock。
            """
            watchers = dict(ledger_watch.get(ledger, ()))
            core = _TOK_CORE_SUFFIX.search(ledger)
            if core is not None:
                watchers.update(ledger_watch.get(ledger[: core.start()] + "c*", ()))
            for n in list(watchers):
                if n not in ready:
                    continue
                if has_pools:
                    pick = _pick_core(by_name[n], dep_ready_at[n])
                    if pick is not None:
                        pool_pick[n] = pick
                    tb = max(dep_ready_at[n], _res_ready(by_name[n]))
                    _push_ready(n, by_name[n], tb)
                elif n in resolved:
                    _push_ready(n, by_name[n], tbase[n])

        def _enter_ready(name: str) -> None:
            ev = by_name[name]
            dep_ready = max(
                (end_by_name[d] + edge_latency(ev, d) for d in ev.deps), default=0.0
            )
            if has_pools:
                pick = _pick_core(ev, dep_ready)
                if pick is not None:
                    pool_pick[name] = pick
                res_ready = _res_ready(ev)
            else:
                res_ready = max((resource_free.get(r, 0.0) for r in ev.resources),
                                default=0.0)
            ready[name] = None
            dep_ready_at[name] = dep_ready
            # 占位资源要登记到**全部**候选成员上: 任一成员空出来都应触发重算 t_base。
            for r in ev.resources:
                for c in (_candidates(ev, r) if has_pools else (r,)):
                    ready_by_res.setdefault(c, set()).add(name)
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
                for c in (_candidates(ev, r) if has_pools else (r,)):
                    ready_by_res.get(c, set()).discard(name)
            for led in _ledgers(ev):
                ledger_watch[led].pop(name, None)

        def _resource_committed(resource: str) -> None:
            """resource 的空闲时刻已变: 重算占用它的就绪事件的 t_base."""
            for n in list(ready_by_res.get(resource, ())):
                if n not in ready:
                    # 登记与注销用的候选集可能不同: 共位锚点在这期间被绑定后,
                    # _candidates 收窄到一个核, _leave_ready 只摘得到那一个,
                    # 别的核的集合里留下陈迹。在这里顺手清掉。
                    ready_by_res[resource].discard(n)
                    continue
                ev_n = by_name[n]
                if has_pools:
                    # 核的空闲时刻变了 -> 最优核可能换人, 重选 (三处用同一个选择)
                    pick = _pick_core(ev_n, dep_ready_at[n])
                    if pick is not None:
                        pool_pick[n] = pick
                    res_ready = _res_ready(ev_n)
                else:
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
                end_by_name=dict(end_by_name))
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

        def _earliest_ok(res: str, k: int, t: float) -> float:
            """>= t 的最早时刻, res 的在途量容得下 k; 没有则 +inf."""
            cap = capacities.get(res)
            if cap is None:
                return t
            if outstanding(res, t) + k <= cap:
                return t
            times = rel_times.get(res, ())
            for i in range(bisect_right(times, t), len(times)):
                te = times[i]
                if outstanding(res, te) + k <= cap:
                    return te
            return float("inf")

        def _slots_ok(ev: Event, core: str, t: float) -> bool:
            """把 ev 放到 core 上, t 时刻它的按核 token 还有余量吗."""
            for res0, k in ev.acquires:
                if not res0.endswith("c*"):
                    continue
                if _earliest_ok(res0[:-1] + core, k, t) > t + 1e-12:
                    return False
            return True

        def capacity_feasible(ev: Event, t: float) -> Tuple[bool, float]:
            """t 时刻容量是否够; 不够时沿未来归还时刻找最早可行点.

            占位 token ("...c*") 的语义是"**任一**候选核有余量即可" —— 取各核里最早
            可行的那个时刻。若按单个核解析, 所有就绪事件会一起撞在同一个核上,
            而那个核的归还时刻可能还没登记, 于是全体不可行 -> 假死锁。
            """
            cores = _tok_cores(ev)
            for res0, k in ev.acquires:
                if res0.endswith("c*") and cores:
                    ok_t = min((_earliest_ok(res0[:-1] + c, k, t) for c in cores),
                               default=float("inf"))
                else:
                    ok_t = _earliest_ok(res0, k, t)
                if ok_t == float("inf"):
                    return False, float("inf")
                t = max(t, ok_t)
            return True, t

        def probe(ev: Event, t_base: float) -> Tuple[float, float, float]:
            """容量准入探测 (不写状态). 返回 (start, duration, cap_wait)."""
            t = t_base
            cap_wait = 0.0
            dur = max(0.0, ev.duration_us)
            for _ in range(4):
                ok, t_cap = capacity_feasible(ev, t)
                if not ok:
                    return float("inf"), dur, cap_wait
                if t_cap > t:
                    cap_wait += t_cap - t
                    t = t_cap
                    continue
                return t, dur, cap_wait
            return t, dur, cap_wait

        from .policies import EarliestStart
        # 快路径只对 EarliestStart 本类启用: 排序键 (start, order, name) 与堆序一致.
        # 子类/其他策略的键由 event_key 决定, 走通用路径.
        fast = type(policy) is EarliestStart
        for name, deg in indegree.items():
            if deg == 0:
                _enter_ready(name)

        while ready:
            best_key: Optional[tuple] = None
            best: Optional[Tuple[str, float, float, float]] = None

            def consider(name: str) -> None:
                nonlocal best_key, best
                ev = by_name[name]
                if not ev.acquires:
                    start, dur = tbase[name], ev.duration_us
                    payload = (0.0,)
                else:
                    start, dur, cap_w = probe(ev, tbase[name])
                    if start == float("inf"):
                        return
                    payload = (cap_w,)
                key = policy.event_key(ev, start, tbase, end_by_name)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (name, start, dur, payload[0])

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
                            by_name[top_simple[2]].duration_us, 0.0)
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

            name, start, dur, cap_wait = best
            ev = by_name[name]
            _leave_ready(name)
            end = start + max(0.0, dur)

            dep_parent = max(ev.deps, key=lambda d: end_by_name[d], default=None)
            dep_ready = (
                max(end_by_name[d] + edge_latency(ev, d) for d in ev.deps) if ev.deps else 0.0
            )
            # "真能动"的最早时刻: 从依赖齐备起算的容量可行点 (不是从 t_base 起算的
            # capacity_wait —— 后者已经把"等自己的核"那一段算在前面了, 加到 dep_ready
            # 上会严重低估)。台账此刻还没写入本事件, 所以这里看到的正是它当初面对的
            # 状态。供 analysis/idle.py 判"这个活到底能不能动"。
            _, t_act = capacity_feasible(ev, dep_ready) if ev.acquires else (True, dep_ready)
            actionable = min(max(dep_ready, t_act), start)
            # L3 晚绑定: 派发时刻才把占位资源绑到具体成员。之后一律用 bound_res,
            # 不再碰 ev.resources —— ScheduledEvent 里记的也是绑定后的名字, 这样
            # 下游 (利用率、空闲分解、stealing) 看到的都是真实落核。
            bound_res = _bind(ev, start) if has_pools else ev.resources
            if has_pools and bound_res:
                bound_core[name] = _core_of(bound_res[0])
            # C5: 一次性开销在静态绑定下同样生效 —— 核号直接取绑定到的资源, 不依赖池。
            once_us = 0.0
            if ev.once_per_core is not None and bound_res:
                ck = (ev.once_per_core[0], _core_of(bound_res[0]))
                if ck not in charged_once:
                    charged_once.add(ck)
                    once_us = max(0.0, ev.once_per_core[1])
                    end += once_us
            acq = _remap_tokens(ev.acquires, bound_core.get(name)) if has_pools else ev.acquires
            rel = _remap_tokens(ev.releases, bound_core.get(name)) if has_pools else ev.releases
            blocking_resource = (
                max(bound_res, key=lambda r: resource_free.get(r, 0.0), default=None)
                if bound_res else None
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

            end_by_name[name] = end
            for resource in bound_res:
                resource_free[resource] = end
                resource_last_event[resource] = name
            for resource in bound_res:
                _resource_committed(resource)
            for res, k in acq:
                acq_ctr.setdefault(res, _TimeCounter()).add(start, k)
            for res, k in rel:
                rel_ctr.setdefault(res, _TimeCounter()).add(end, k)
                insort(rel_times.setdefault(res, []), end)
            for ledger in dict.fromkeys([res for res, _ in acq]
                                        + [res for res, _ in rel]):
                _ledger_written(ledger)

            scheduled.append(
                ScheduledEvent(
                    name=name,
                    resources=bound_res,
                    start_us=start,
                    end_us=end,
                    dependency_ready_us=dep_ready,
                    resource_ready_us=res_ready,
                    dependency_wait_us=max(0.0, dep_ready - res_ready),
                    resource_queue_us=max(0.0, res_ready - dep_ready),
                    critical_parent=critical_parent,
                    critical_reason=critical_reason,
                    order=ev.order,
                    meta=dict(ev.meta, once_per_core_us=once_us) if once_us else dict(ev.meta),
                    capacity_wait_us=cap_wait,
                    actionable_us=actionable,
                    acquires=acq,
                    releases=rel,
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
