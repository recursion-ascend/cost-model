"""Wave cost model 编排层 — 建图 + 调度 + 后处理.

"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import registry
from .builders.mte import MteEventBuilder
from .config.hardware import KernelConfig, ceil_div
from .scheduler.events import POOL_WILDCARD, Event, ScheduledEvent
from .scheduler.engine import MultiResourceScheduler
from .scheduler.policies import (CriticalPathFirst, EarliestStart, PriorityByStage,
                                 WorkConservingCriticalPath)
from .builders.barriers import apply_barriers
from .builders.pipeline_expand import apply_pipeline
from .costs import PrimitiveCosts
from .shape import (
    CursorTrace, EngineQueueDepths, MegaMoeShape, ModelOptions,
)
from .planning.waves import (
    Wave, calc_m_groups_per_wave, plan_waves, plan_layered_waves,
)


#: 可晚绑定的角色池。moe_stage_done 之类"代表某个核"的零时长栅栏不参与。
POOLABLE_ROLES = ("AIC", "AIV0", "AIV1")
_CORE_SUFFIX = re.compile(r"c\d+$")


def _rewrite_for_late_binding(events: List[Event], roles: Sequence[str], aic_num: int,
                             pre: str, pools: Dict[str, Tuple[str, ...]]) -> None:
    """把指定角色的核资源从"建图时定死"改成"派发时绑定最早空闲的核"。

    做法: 事件声明 "AIC:*" 而不是 "AIC:7", 调度器在准入那一刻挑成员 (engine 的
    _candidates/_bind)。每核的自取自还队列信号量 (Q:aic:c7) 在独占核资源下是空约束,
    直接去掉 (否则落到核 3 的 tile 会去扣核 7 的队列深度)。

    **不删不改任何已有依赖边**: 事件名、deps、dep_latency 原样, 变的只有"哪个核来做"。

    物理约束用 colocate_with 保住:
      GMM1 -> ACT 同核 (L0C->UB Fixpipe 直给配对 AIV0), 所以 AIC 入池即隐含 AIV0 入池,
      且每个 ACT 必须与它的 GMM1 落同一核号。
      dispatch 的每波调用开销改成"该核本波第一段 dispatch 的一次性开销"
      (Event.once_per_core), dispatch_call 事件本身归 0 —— 不钉核, 不加边。
      GMM2 -> combine 不是物理共位 (GMM2 写 GM, combine 读 GM), 故不加约束。
    """
    pooled = set(roles)
    if "AIC" in pooled:
        pooled.add("AIV0")
    bad = sorted(pooled - set(POOLABLE_ROLES))
    if bad:
        raise ValueError(f"late_bind_pools 只支持 {POOLABLE_ROLES}, 收到 {bad}")
    for role in sorted(pooled):
        pools[f"{pre}{role}"] = tuple(f"{pre}{role}:{c}" for c in range(aic_num))

    stage_of = {e.name: str(e.meta.get("stage", "")) for e in events}
    dur_of = {e.name: e.duration_us for e in events}
    for ev in events:
        if str(ev.meta.get("stage", "")) == "moe_stage_done":
            continue  # 每核排空栅栏: 它代表的就是那个核, 不能漂移
        new_res, touched = [], False
        for r in ev.resources:
            role, _, core_s = r.partition(":")
            if core_s and role in pooled:
                new_res.append(role + POOL_WILDCARD)
                touched = True
            else:
                new_res.append(r)
        if not touched:
            continue
        ev.resources = tuple(new_res)
        core = ev.meta.get("core")
        if isinstance(core, int):
            # 按核的计数信号量分两类:
            #
            # 1) 自取自还 (Q:aic:c7 之类的引擎队列): 事件同时独占该核资源, 同核在途数
            #    恒 <= 1, 只要深度 >= 1 就不起约束 (见 EngineQueueDepths 文档)。
            #    **去掉**, 语义不变, 也省掉一次核号解析。
            # 2) 跨事件持有 (UB:gmm1act:c7 —— GMM1 取、配对 ACT 还): 真约束, 核号换成
            #    占位 c*, 由引擎在派发时回填 (_tok / _remap_tokens)。取与还必须落同一个
            #    核号, 这由共位保证 (ACT 的 colocate_with 指向它的 GMM1)。
            self_paired = set(ev.acquires) & set(ev.releases)

            def _norm(tok):
                t, k = tok
                if not _CORE_SUFFIX.search(t):
                    return tok
                return (_CORE_SUFFIX.sub("c*", t), k)

            ev.acquires = tuple(_norm(a) for a in ev.acquires if a not in self_paired
                                or not _CORE_SUFFIX.search(a[0]))
            ev.releases = tuple(_norm(r) for r in ev.releases if r not in self_paired
                                or not _CORE_SUFFIX.search(r[0]))
        if str(ev.meta.get("stage", "")) == "activation":
            anchor = next((d for d in ev.deps if stage_of.get(d) == "gmm1"), None)
            if anchor is not None:
                ev.colocate_with = anchor
        if str(ev.meta.get("stage", "")) == "dispatch" and ".dispatch." in ev.name:
            # 调用开销 = 该核这一波第一段 dispatch 的一次性开销 (once_per_core),
            # 不再靠单独钉核的 dispatch_call 事件承担: 段可落任意空闲核, 开销
            # 落在真正搬数据的那个核上。无新增依赖边。
            call = f"{ev.name.split('.dispatch.', 1)[0]}.dispatch_call.c{core}"
            if stage_of.get(call) == "dispatch_call" and dur_of[call] > 0:
                ev.once_per_core = (call.rsplit(".c", 1)[0], dur_of[call])

    # dispatch_call 的开销已转给各段 dispatch 的 once_per_core, 事件本身留作 0 时长
    # 的标记 (名字仍被 dispatch_ready 的 meta 引用)。
    if "AIV1" in pooled:
        for ev in events:
            if str(ev.meta.get("stage", "")) == "dispatch_call":
                ev.duration_us = 0.0

    # 相位拆分事件 (.ld/.cb/fix) 自己不持核资源, 只靠带核号的队列 token 绑核;
    # 晚绑定下这些 token 无从回填, 先明确拒绝而不是算出一个错数。
    for ev in events:
        if ev.resources:
            continue
        if any(_CORE_SUFFIX.search(t) for t, _ in ev.acquires + ev.releases):
            raise NotImplementedError(
                f"事件 {ev.name} 不持核资源却带按核的队列信号量 (相位流水), "
                "暂不支持与 late_bind_pools 同用")


def completion_event(scheduled: Sequence[ScheduledEvent]) -> Optional[ScheduledEvent]:
    """执行时间的终点事件: 最晚结束的 COMBINE.

    执行时间记到最后一个 COMBINE 结束为止; 其后的尾段 (counts_export /
    core_sync / rank_sync / unpermute / finalize, 以及共享专家的 GMM2) 仍在
    事件图里照常调度, 但不计入执行时间. 没有 COMBINE 事件时退回最晚结束的事件.
    """
    if not scheduled:
        return None
    combines = [e for e in scheduled if e.meta.get("stage") == "combine"]
    return max(combines or scheduled, key=lambda e: (e.end_us, e.order, e.name))


class A8W8WaveCostModel:
    def __init__(self, costs: PrimitiveCosts, options: ModelOptions = ModelOptions()):
        if not options.combine_no_quant:
            raise NotImplementedError("v3 models A8W8 COMBINE_NO_QUANT only")
        if options.topk_weights_prefetch:
            raise NotImplementedError("v3 models TopkWeightsPrefetch=false only")
        self.costs = costs
        self.options = options
        self._order = 0
        self._rank = 0
        self.cursor_traces: Dict[int, List[CursorTrace]] = {}

    def _kernel_cfg(self, shape: MegaMoeShape) -> KernelConfig:
        return shape.kernel if shape.kernel is not None else KernelConfig()

    def m_groups_per_wave(self, shape: MegaMoeShape) -> int:
        km = self._kernel_cfg(shape)
        if km.topo_urma:
            return 0   # Layered 宏 Wave = 专家范围, 无 m-group 波宽概念
        if self.options.m_groups_per_wave > 0:
            return self.options.m_groups_per_wave      # C4: 直接给波宽
        p1 = shape.p1_override if shape.p1_override > 0 else 1
        p2 = shape.p2_override if shape.p2_override > 0 else 1
        return calc_m_groups_per_wave(
            hidden_dim=shape.hidden_dim, h=shape.h, aic_num=shape.aic_num,
            p1=p1, p2=p2, tile_n=km.tile_n)

    def waves(self, shape: MegaMoeShape) -> List[Wave]:
        km = self._kernel_cfg(shape)
        if km.topo_urma:
            return plan_layered_waves(shape.expert_tokens, shape.token_num, shape.topk,
                                      tile_m=km.tile_m)
        wp = getattr(shape, "wave_packing", None)
        if wp is not None:
            return wp.plan(shape.expert_tokens, self.m_groups_per_wave(shape),
                           tile_m=km.tile_m)
        return plan_waves(shape.expert_tokens, self.m_groups_per_wave(shape),
                          tile_m=km.tile_m)

    def build_events(self, shape: MegaMoeShape) -> Tuple[List[Event], List[CursorTrace]]:
        """建图器: shape.orchestration 优先, 未给时按 kernel.topo_urma 自动选."""
        km = self._kernel_cfg(shape)
        cls = registry.builder_class(getattr(shape, "orchestration", None))
        if cls is None:
            if km.topo_urma:
                from .builders.layered import LayeredEventBuilder
                cls = LayeredEventBuilder
            else:
                cls = MteEventBuilder
        return cls(self.costs, self.options).build(shape, self.waves(shape))

    def dispatch_ir(self, shape: MegaMoeShape) -> List:
        return [
            MteEventBuilder(self.costs, self.options)._dispatch_call_ir(shape, w, core)
            for w in self.waves(shape)
            for core in range(shape.aic_num)
        ]

    def simulate(self, shape: MegaMoeShape, restructure=None) -> Dict[str, object]:
        return self.simulate_multi([shape], restructure=restructure)[shape.rank_id]

    def simulate_multi(self, shapes: Sequence[MegaMoeShape],
                       restructure=None) -> Dict[int, Dict[str, object]]:
        """多 rank 调度: 核/队列资源按 rank 前缀隔离.

        信道模型 (速率服务器) 已于 2026-10-03 停用: 事件的 channel_bytes 仍然申报
        字节, 但只在 rank_results["traffic_bytes"] 里汇总成访存量, 不参与准入、
        不影响任何时长。片间 fab 通路同理 —— 它的两个常数本来就不同尺度
        (聚合 BW_WINDOW=33000 是整卡值, 逐事件 BW_REMOTE_GM=31000 是从 28 核并发
        反解的单核值, 已含平均争用), 叠速率服务器会把争用计两遍。

        各 rank 之间无共享资源时逐 rank 独立调度 (结果与合并调度逐位一致,
        见 _ranks_independent); 否则全部事件进同一个调度器.
        """
        per: List[Tuple[MegaMoeShape, List[Event], Dict, List[CursorTrace]]] = []
        self._sched_policy = getattr(shapes[0], 'scheduling_policy', None) if shapes else None
        if shapes:
            mism = [d for d, sh in enumerate(shapes)
                    if getattr(sh, 'scheduling_policy', None) is not self._sched_policy
                    and getattr(sh, 'scheduling_policy', None) != self._sched_policy]
            if mism:
                raise ValueError(
                    f"各 rank 的 scheduling_policy 不一致 (rank {mism} 与 rank 0 不同); "
                    "多 rank 单调度器只支持同一策略")
        for shape in shapes:
            events, trace = self.build_events(shape)
            self.cursor_traces[shape.rank_id] = trace
            if self.options.barriers:
                events = apply_barriers(events, self.options.barriers)
            caps: Dict = {}
            if self.options.pipeline is not None:
                events, caps = apply_pipeline(
                    events, self.options.pipeline,
                    aic_num=shape.aic_num, h=shape.h,
                    gmm1_act_depth=shape.policy.gmm1_activation_depth, kernel=shape.kernel)
            per.append((shape, events, caps, trace))

        # 每 rank 一组 (事件, 容量, 资源池)
        groups: List[Tuple[List[Event], Dict[str, int],
                           Dict[str, Tuple[str, ...]]]] = []
        late = tuple(self.options.late_bind_pools or ())
        for shape, events, caps, trace in per:
            pre = f"R{shape.rank_id}."
            pools: Dict[str, Tuple[str, ...]] = {}
            if late:
                _rewrite_for_late_binding(events, late, shape.aic_num, pre, pools)
            capacities: Dict[str, int] = {}
            for ev in events:
                ev.resources = tuple(pre + r for r in ev.resources)
                ev.acquires = tuple((pre + a, k) for a, k in ev.acquires)
                ev.releases = tuple((pre + r, k) for r, k in ev.releases)
                # 访存量申报全部保留 (不再按"开了哪些信道"过滤): 它只做统计。
                # 片间通路跨 rank 共享, 不加前缀; 其余按 rank 前缀隔离。
                ev.channel_bytes = tuple(
                    (c, b, rt) if c.startswith("fab_") else (pre + c, b, rt)
                    for c, b, rt in ev.channel_bytes)
            qd = self.options.engine_queue_depths or EngineQueueDepths()
            ub_depth = shape.policy.gmm1_activation_depth
            for core in range(shape.aic_num):
                capacities[pre + f"Q:aic:c{core}"] = qd.aic
                capacities[pre + f"Q:vec0:c{core}"] = qd.vec0
                capacities[pre + f"Q:aiv1:c{core}"] = qd.aiv1
                # GMM1->ACT 的 UB 槽位数 (C2: 容量, 不是程序序边)。深度 0 时
                # builders 不申报这个 token, 容量也就不必声明。
                if ub_depth > 0:
                    capacities[pre + f"UB:gmm1act:c{core}"] = ub_depth
            for k, v in caps.items():
                capacities[pre + k] = v
            groups.append((events, capacities, pools))

        sched_pol = getattr(self, '_sched_policy', None)
        if not self._ranks_independent(shapes, restructure, sched_pol):
            # 合并调度: 全部 rank 的事件/容量进同一个调度器
            merged_caps: Dict[str, int] = {}
            merged_pools: Dict[str, Tuple[str, ...]] = {}
            for events, capacities, pools in groups:
                merged_caps.update(capacities)
                merged_pools.update(pools)
            groups = [([ev for events, _, _ in groups for ev in events],
                       merged_caps, merged_pools)]
        scheduled: List[ScheduledEvent] = []
        for events, capacities, pools in groups:
            _, part = MultiResourceScheduler().schedule(
                events, capacities=capacities or None,
                restructure=restructure, policy=sched_pol, pools=pools or None)
            scheduled.extend(part)

        # 访存量汇总: 事件申报的 channel_bytes 不参与准入 (信道模型已停用), 只在
        # 这里按通路累加成字节数, 供"哪条通路搬了多少"的核算用。
        traffic: Dict[int, Dict[str, float]] = {sh.rank_id: {} for sh in shapes}
        for shape, events, _caps, _trace in per:
            tr = traffic[shape.rank_id]
            for ev in events:
                for cname, nbytes, _rate in ev.channel_bytes:
                    tr[cname] = tr.get(cname, 0.0) + float(nbytes)

        # 全部 rank 的容量表 (空闲分解要用它判"这个空闲核的槽还有余量吗")
        all_caps: Dict[str, int] = {}
        for _evs, caps_, _pools in groups:
            all_caps.update(caps_)

        results: Dict[int, Dict[str, object]] = {}
        for shape in shapes:
            rank = shape.rank_id
            evs = [e for e in scheduled if e.meta.get("rank") == rank]
            results[rank] = self._postprocess(shape, evs, all_caps)
            results[rank]["traffic_bytes"] = dict(sorted(traffic[rank].items()))
        return results

    def _ranks_independent(self, shapes: Sequence[MegaMoeShape], restructure,
                           sched_pol) -> bool:
        """各 rank 能否独立调度而不改变结果.

        调度器每步提交就绪集里排序键最小的事件. 资源/容量按 rank 前缀隔离、
        依赖不跨 rank 时, 一个 rank 的就绪集与资源状态只被本 rank 的提交改变,
        合并调度里该 rank 的提交子序列就等于它单独调度的序列.
        以下情况该前提不成立, 走合并调度:
          * 重构钩子 — 上下文是全局视图, 可跨 rank 转移任务;
          * 自定义调度策略 — event_key 能读到全局 tbase/end_by_name;
          * rank_id 重复 — 交给调度器报重名.
        """
        if restructure is not None:
            return False
        if sched_pol is not None and type(sched_pol) not in (
                EarliestStart, CriticalPathFirst, PriorityByStage,
                WorkConservingCriticalPath):
            return False
        ranks = [sh.rank_id for sh in shapes]
        return len(set(ranks)) == len(ranks)

    def _postprocess(self, shape: MegaMoeShape,
                     scheduled: List[ScheduledEvent],
                     capacities: Optional[Dict[str, int]] = None) -> Dict[str, object]:
        waves = self.waves(shape)
        resource_busy: Dict[str, float] = {}
        resource_first: Dict[str, float] = {}
        resource_last: Dict[str, float] = {}
        stage_busy: Dict[str, float] = {}
        stage_first: Dict[str, float] = {}
        stage_last: Dict[str, float] = {}
        stage_dependency_wait: Dict[str, float] = {}
        stage_resource_queue: Dict[str, float] = {}

        for ev in scheduled:
            duration = ev.end_us - ev.start_us
            for resource in ev.resources:
                resource_busy[resource] = resource_busy.get(resource, 0.0) + duration
                resource_first[resource] = min(resource_first.get(resource, ev.start_us), ev.start_us)
                resource_last[resource] = max(resource_last.get(resource, ev.end_us), ev.end_us)
            stage = str(ev.meta.get("stage", "other"))
            # overlapped: 与同 tile 的另一相位并行且较短 (相位流水), 不重复计入
            if not ev.meta.get("overlapped"):
                stage_busy[stage] = stage_busy.get(stage, 0.0) + duration
            else:
                stage_busy.setdefault(stage, 0.0)
            stage_first[stage] = min(stage_first.get(stage, ev.start_us), ev.start_us)
            stage_last[stage] = max(stage_last.get(stage, ev.end_us), ev.end_us)
            stage_dependency_wait[stage] = stage_dependency_wait.get(stage, 0.0) + ev.dependency_wait_us
            stage_resource_queue[stage] = stage_resource_queue.get(stage, 0.0) + ev.resource_queue_us

        resource_span = {r: resource_last[r] - resource_first[r] for r in resource_busy}
        resource_idle = {r: max(0.0, resource_span[r] - resource_busy[r]) for r in resource_busy}
        resource_utilization = {
            r: (resource_busy[r] / resource_span[r] if resource_span[r] > 0 else 0.0)
            for r in resource_busy}

        # 空闲分解 (每个角色池一份): forced = 此刻全局无就绪活, 消不掉;
        # avoidable = 有就绪活却有核空着, 即 work-conservation 违规。约 simulate 的 1%。
        from .analysis.idle import idle_decomposition
        idle_reports = {}
        for role in ("AIC:", "AIV0:", "AIV1:"):
            for rank_tag, rep in idle_decomposition(
                    scheduled, role, capacities=capacities).items():
                idle_reports[f"{rank_tag}.{role.rstrip(':')}" if rank_tag
                             else role.rstrip(":")] = rep

        scheduled_by_name = {ev.name: ev for ev in scheduled}
        gmm1_by_group: Dict[Tuple[int, int], List[ScheduledEvent]] = {}
        for ev in scheduled:
            if ev.meta.get("stage") == "gmm1":
                gmm1_by_group.setdefault(
                    (int(ev.meta["expert"]), int(ev.meta["mgroup"])), []).append(ev)

        dispatch_ready_tiles: List[Dict[str, object]] = []
        for ev in scheduled:
            if ev.meta.get("stage") != "dispatch_ready":
                continue
            expert = int(ev.meta["expert"])
            mgroup = int(ev.meta["mgroup"])
            g1 = gmm1_by_group.get((expert, mgroup), [])
            g1_first = min((x.start_us for x in g1), default=None)
            dispatch_ready_tiles.append({
                "dst_rank": int(ev.meta["dst_rank"]),
                "wave": int(ev.meta["wave"]),
                "expert": expert, "mgroup": mgroup,
                "required_rows": int(ev.meta["required_rows"]),
                "contributed_rows": int(ev.meta["contributed_rows"]),
                "contributor_count": int(ev.meta["contributor_count"]),
                "t_dispatchReady_us": ev.end_us,
                "gmm1_first_start_us": g1_first,
            })
        dispatch_ready_tiles.sort(key=lambda x: (x["t_dispatchReady_us"], x["expert"], x["mgroup"]))

        critical_path: List[Dict[str, object]] = []
        tail = completion_event(scheduled)
        if tail is not None:
            seen = set()
            cur: Optional[ScheduledEvent] = tail
            while cur is not None and cur.name not in seen:
                seen.add(cur.name)
                critical_path.append({
                    "name": cur.name, "stage": str(cur.meta.get("stage", "other")),
                    "start_us": cur.start_us, "end_us": cur.end_us,
                    "duration_us": cur.end_us - cur.start_us,
                    "critical_reason": cur.critical_reason,
                    "critical_parent": cur.critical_parent,
                    "resources": cur.resources})
                cur = scheduled_by_name.get(cur.critical_parent) if cur.critical_parent else None
            critical_path.reverse()

        return {
            "total_us": tail.end_us if tail is not None else 0.0,
            "dag_end_us": max((e.end_us for e in scheduled), default=0.0),
            "m_groups_per_wave": self.m_groups_per_wave(shape),
            "wave_count": len(waves),
            "waves": waves,
            "cursor_trace": self.cursor_traces.get(shape.rank_id, []),
            "events": scheduled,
            "resource_busy_us": resource_busy,
            "resource_span_us": resource_span,
            "resource_idle_us": resource_idle,
            "resource_utilization": resource_utilization,
            "dispatch_ready_tiles": dispatch_ready_tiles,
            "critical_path": critical_path,
            "stage_busy_us": stage_busy,
            "stage_first_start_us": stage_first,
            "stage_last_end_us": stage_last,
            "stage_dependency_wait_us": stage_dependency_wait,
            "stage_resource_queue_us": stage_resource_queue,
            "gmm2_lag_active": (shape.policy.effective_gmm2_lag(shape.token_num) > 0
                                and not self._kernel_cfg(shape).topo_urma),
            # 空闲分解: 区分"DAG 逼出来的"与"有活却空着"。后者才是 work-conservation
            # 违规, 换 tile->核 的绑定方式可回收; 前者只能靠改 DAG 结构 (加深流水、改波
            # 的组成)。见 analysis/idle.py —— avoidable 是上界, 未计共位与队列约束。
            "idle_decomposition": idle_reports,
        }

    def structural_summary(self, shape: MegaMoeShape) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        km = shape.kernel if shape.kernel is not None else KernelConfig()
        g1n = ceil_div(ceil_div(shape.hidden_dim, km.activation_n_half), km.tile_n)
        g2n = ceil_div(shape.h, km.tile_n)
        for w in self.waves(shape):
            rows.append({
                "wave": w.index, "rows": w.rows, "m_groups": w.m_groups,
                "expert_slices": len(w.slices),
                "gmm1_tiles": sum(s.m_groups * g1n for s in w.slices),
                "gmm2_tiles": sum(s.m_groups * g2n for s in w.slices),
                "begin": (w.begin.expert, w.begin.row, w.begin.global_row),
                "end": (w.end.expert, w.end.row, w.end.global_row),
                "slices": tuple((s.expert, s.row_begin, s.row_end, s.m_groups)
                                for s in w.slices)})
        return rows

    def cursor_summary(self, shape: MegaMoeShape) -> List[CursorTrace]:
        _, trace = self.build_events(shape)
        return trace
