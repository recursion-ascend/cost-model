"""Wave cost model 编排层 — 建图 + 调度 + 后处理.

"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import registry
from .builders.mte import MteEventBuilder
from .config.hardware import KernelConfig, BW_WINDOW, ceil_div
from .config.policy import InstancePolicy
from .scheduler.events import Channel, Event, ScheduledEvent
from .scheduler.engine import MultiResourceScheduler
from .scheduler.policies import CriticalPathFirst, EarliestStart, PriorityByStage
from .costs import DispatchDataLayout
from .builders.pipeline_expand import apply_pipeline
from .costs import PrimitiveCosts
from .shape import (
    BlockCursor, CursorTrace, EngineQueueDepths, MegaMoeShape, ModelOptions,
)
from .planning.waves import (
    ExpertSlice, Wave, calc_m_groups_per_wave, plan_waves, plan_layered_waves,
)


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
        from .builders.base import DispatchCallIR
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

        片间信道 (fab_src/fab_dst) 默认关闭 (占位, ModelOptions.fabric_channels):
        常数从真实运行反解、已含平均争用, 无消融证据前不再叠加速率服务器。

        占位为什么还不能当预测用 (开启前必须先重标定这两个常数):
          聚合 bw_total = BW_WINDOW = 33000 B/us 是**整卡**的片间带宽, 而逐事件
          申报速率用 BW_REMOTE_GM = 31000 —— 后者是从真实运行逐段反解的**单核**
          速率, 本身已含 28 核并发的平均争用。于是单个事件就要吃掉整卡片间带宽的
          94%, 28 核并发时每核只剩 1179 B/us, 等于在已含争用的常数上再叠一层
          26 倍降速。争用被计了两遍。
          要用这条路径, 逐事件速率得换成**无争用**的单核带宽 (需要单核独占的
          片间搬运实验), 聚合再保持整卡值 —— 那时 28 核同写一条链路的降速才是
          调度器算出来的, 而不是既藏在常数里又叠在信道上。
          2026-09-30 起 COMBINE 的跨卡行写也申报到这两条信道 (原先完全不占片间
          资源 —— DAG 里一条跨卡写不碰互连是建模漏洞), 所以这个双重计费在
          fabric_channels_on 用例上比原先更显眼 (2344 -> 6830 us)。数值是占位缺陷
          的显形, 不是预测。

        各 rank 之间无共享资源时逐 rank 独立调度 (结果与合并调度逐位一致,
        见 _ranks_independent); 否则全部事件进同一个调度器.
        """
        per: List[Tuple[MegaMoeShape, List[Event], Dict, Dict, List[CursorTrace]]] = []
        world = max((len(sh.expert_source_tokens[0]) for sh in shapes), default=0)
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
            caps: Dict = {}
            chans: Dict = {}
            if self.options.pipeline is not None:
                events, caps, chans = apply_pipeline(
                    events, self.options.pipeline,
                    aic_num=shape.aic_num, h=shape.h,
                    gmm1_act_depth=shape.policy.gmm1_activation_depth, kernel=shape.kernel)
            per.append((shape, events, caps, chans, trace))

        fab_on = bool(self.options.fabric_channels)
        channels: Dict[str, Channel] = {}
        if fab_on:
            for s_ in range(world):
                channels[f"fab_src:{s_}"] = Channel(f"fab_src:{s_}",
                                                    bw_total=BW_WINDOW, max_rate_per_event=BW_WINDOW)
                channels[f"fab_dst:{s_}"] = Channel(f"fab_dst:{s_}",
                                                    bw_total=BW_WINDOW, max_rate_per_event=BW_WINDOW)
        # 每 rank 一组 (事件, 容量, 信道); 片间信道跨 rank 共享, 单列
        groups: List[Tuple[List[Event], Dict[str, int], Dict[str, Channel]]] = []
        for shape, events, caps, chans, trace in per:
            pre = f"R{shape.rank_id}."
            capacities: Dict[str, int] = {}
            rank_channels: Dict[str, Channel] = {}
            for ev in events:
                ev.resources = tuple(pre + r for r in ev.resources)
                ev.acquires = tuple((pre + a, k) for a, k in ev.acquires)
                ev.releases = tuple((pre + r, k) for r, k in ev.releases)
                # 未配置的信道直接丢掉: 建图器无条件申报字节 (它不知道调用方开了
                # 哪些信道), 由此处按实际存在的信道过滤。片间信道跨 rank 共享不加
                # 前缀; 其余按 rank 前缀隔离。
                ev.channel_bytes = tuple(
                    (c, b, rt) if c.startswith("fab_") else (pre + c, b, rt)
                    for c, b, rt in ev.channel_bytes
                    if (fab_on if c.startswith("fab_") else c in chans))
            qd = self.options.engine_queue_depths or EngineQueueDepths()
            for core in range(shape.aic_num):
                capacities[pre + f"Q:aic:c{core}"] = qd.aic
                capacities[pre + f"Q:vec0:c{core}"] = qd.vec0
                capacities[pre + f"Q:aiv1:c{core}"] = qd.aiv1
            for k, v in caps.items():
                capacities[pre + k] = v
            for ch_name, ch in chans.items():
                rank_channels[pre + ch_name] = Channel(pre + ch_name,
                                                       bw_total=ch.bw_total,
                                                       max_rate_per_event=ch.max_rate_per_event)
            groups.append((events, capacities, rank_channels))

        sched_pol = getattr(self, '_sched_policy', None)
        if not self._ranks_independent(shapes, restructure, sched_pol):
            # 合并调度: 全部 rank 的事件/容量/信道进同一个调度器
            merged_caps: Dict[str, int] = {}
            for events, capacities, rank_channels in groups:
                merged_caps.update(capacities)
                channels.update(rank_channels)
            groups = [([ev for events, _, _ in groups for ev in events],
                       merged_caps, channels)]
        scheduled: List[ScheduledEvent] = []
        for events, capacities, rank_channels in groups:
            _, part = MultiResourceScheduler().schedule(
                events, capacities=capacities or None, channels=rank_channels or None,
                restructure=restructure, policy=sched_pol)
            scheduled.extend(part)

        results: Dict[int, Dict[str, object]] = {}
        for shape in shapes:
            rank = shape.rank_id
            evs = [e for e in scheduled if e.meta.get("rank") == rank]
            results[rank] = self._postprocess(shape, evs)
        return results

    def _ranks_independent(self, shapes: Sequence[MegaMoeShape], restructure,
                           sched_pol) -> bool:
        """各 rank 能否独立调度而不改变结果.

        调度器每步提交就绪集里排序键最小的事件. 资源/容量/信道按 rank 前缀
        隔离、依赖不跨 rank 时, 一个 rank 的就绪集与资源状态只被本 rank 的提交
        改变, 合并调度里该 rank 的提交子序列就等于它单独调度的序列.
        以下情况该前提不成立, 走合并调度:
          * 片间信道开启 — fab_src/fab_dst 跨 rank 共享;
          * 重构钩子 — 上下文是全局视图, 可跨 rank 转移任务;
          * 自定义调度策略 — event_key 能读到全局 tbase/end_by_name;
          * rank_id 重复 — 交给调度器报重名.
        """
        if restructure is not None or self.options.fabric_channels:
            return False
        if sched_pol is not None and type(sched_pol) not in (
                EarliestStart, CriticalPathFirst, PriorityByStage):
            return False
        ranks = [sh.rank_id for sh in shapes]
        return len(set(ranks)) == len(ranks)

    def _postprocess(self, shape: MegaMoeShape,
                     scheduled: List[ScheduledEvent]) -> Dict[str, object]:
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
            contributor_events = tuple(ev.meta.get("contributor_events", ()))
            work_events = [scheduled_by_name[n] for n in contributor_events]
            first_row_work = min((x.start_us for x in work_events), default=ev.start_us)
            service_sum = sum(x.end_us - x.start_us for x in work_events)
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
