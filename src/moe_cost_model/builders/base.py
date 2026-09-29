"""第 4 层: 事件图构建公共基类 — IR/公共方法/共享专家/GMM1+ACT/GMM2+COMBINE/尾段.

MTE 与 URMA Layered 两个建图器共享本基类; 路径差异在各自子类.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..config.hardware import (
    BW_UNPERMUTE_AGG, T_CORE_SYNC_BARRIER_US, T_COUNTS_EXPORT_US,
    T_FINALIZE_US, T_OUTPUT_INIT_US, T_RANK_SYNC_RTT_US, ceil_div,
)
from ..scheduler.events import Event
from ..costs import DispatchDataLayout, PrimitiveCosts
from ..shape import BlockCursor, CursorTrace, MegaMoeShape, ModelOptions
from ..planning.waves import ExpertSlice, Wave, swizzle_coord


# =====================================================================
# Dispatch 段枚举 IR 
# =====================================================================


@dataclass(frozen=True)
class DispatchExpertIR:
    expert: int = 0
    dst_rank: int = 0
    row_begin: int = 0
    row_end: int = 0
    segments: Tuple[Tuple[int, int], ...] = ()   # (src_rank, rows) in execution order
    local_segments: int = 0
    remote_segments: int = 0
    rows: int = 0


@dataclass(frozen=True)
class DispatchCallIR:
    dst_rank: int = 0
    wave: int = 0
    aiv1: int = 0
    global_row_begin: int = 0
    global_row_end: int = 0
    experts: Tuple[DispatchExpertIR, ...] = ()


def build_dispatch_expert_ir(*, expert, dst_rank, source_counts, row_begin, row_end, layout):
    """Segments of one (core-local) expert row range, src-major order."""
    segs: List[Tuple[int, int]] = []
    cur = 0
    for src, cnt in enumerate(source_counts):
        nxt = cur + cnt
        lo = max(row_begin, cur)
        hi = min(row_end, nxt)
        if hi > lo:
            segs.append((src, hi - lo))
        cur = nxt
    local = [(s, r) for s, r in segs if s == dst_rank]
    remote = [(s, r) for s, r in segs if s != dst_rank]
    return DispatchExpertIR(
        expert=expert, dst_rank=dst_rank,
        row_begin=row_begin, row_end=row_end,
        segments=tuple(segs),
        local_segments=len(local), remote_segments=len(remote),
        rows=sum(r for _, r in segs))


class EventBuilderBase:
    """持有建图工作状态, 按波序列生成事件图.

    状态生命周期 = 一次 build() 调用; 建成后本对象可丢弃.
    """

    def __init__(self, costs: PrimitiveCosts, options: ModelOptions):
        self.costs = costs
        self.options = options
        self.events: List[Event] = []
        self.cursor_trace: List[CursorTrace] = []
        self._order = 0
        self._rank = 0
        # combine 传输后端 (builders/comm): 由编排循环装配;
        # gmm2 stage 通过 on_gmm2_tile 钩子调用
        self.combine_backend = None
        # gmm2 tail 事件按 (expert, global_group) 归档 — layered combine 的
        # 组级就绪依赖 (kernel: GMM2 sync counter ≥ nTilesPerGroup)
        self.gmm2_tail_by_group: Dict[Tuple[int, int], List[str]] = {}

    

    def _event(self, name: str, resources, duration_us: float,
               deps=(), meta=None, dep_latency_us: float = 0.0,
               acquires=(), releases=(), channel_bytes=()) -> str:
        _rp = f"R{self._rank}."
        dep_tuple = tuple(
            dict.fromkeys(d if d.startswith(_rp) else _rp + d for d in deps if d))
        name = f"R{self._rank}.{name}"
        self.events.append(Event(
            name=name, resources=tuple(resources),
            duration_us=max(0.0, float(duration_us)),
            deps=dep_tuple, order=self._order,
            meta=dict(meta or {}, rank=self._rank),
            dep_latency_us=dep_latency_us,
            acquires=tuple(acquires), releases=tuple(releases),
            channel_bytes=tuple(channel_bytes),
        ))
        self._order += 1
        return name

    @staticmethod
    def _tile_rows(slice_: ExpertSlice, local_mgroup: int, tile_m: int = 256) -> int:
        start = local_mgroup * tile_m
        return max(0, min(tile_m, slice_.rows - start))

    @staticmethod
    def _rotated_balanced_range(total: int, worker: int, workers: int,
                                global_prefix: int) -> Tuple[int, int]:
        if workers <= 0 or worker < 0 or worker >= workers:
            return (0, 0)
        first_owner = global_prefix % workers
        logical = worker - first_owner if worker >= first_owner else worker + workers - first_owner
        base, rem = divmod(total, workers)
        extra_before = logical if logical < rem else rem
        start = logical * base + extra_before
        count = base + (1 if logical < rem else 0)
        return start, count

    @staticmethod
    def _gmm1_device_scheduler_n(shape: MegaMoeShape, act_half: int = 2) -> int:
        return ceil_div(shape.hidden_dim, act_half)

    def _dispatch_call_ir(self, shape: MegaMoeShape, w: Wave, core: int) -> DispatchCallIR:
        if not shape.expert_source_tokens:
            raise ValueError("mechanistic Dispatch requires exact expert_source_tokens")
        layout = shape.dispatch_layout
        if layout is None:
            layout = DispatchDataLayout.from_hidden(shape.h)
        rel_begin, count = self._rotated_balanced_range(w.rows, core, shape.aic_num,
                                                        w.begin.global_row)
        core_global_begin = w.begin.global_row + rel_begin
        core_global_end = core_global_begin + count
        expert_irs = []
        if count:
            for sl in w.slices:
                overlap_begin = max(core_global_begin, sl.global_row_begin)
                overlap_end = min(core_global_end, sl.global_row_end)
                if overlap_begin >= overlap_end:
                    continue
                local_begin = sl.row_begin + (overlap_begin - sl.global_row_begin)
                local_end = sl.row_begin + (overlap_end - sl.global_row_begin)
                expert_irs.append(build_dispatch_expert_ir(
                    expert=sl.expert, dst_rank=shape.rank_id,
                    source_counts=shape.expert_source_tokens[sl.expert],
                    row_begin=local_begin, row_end=local_end, layout=layout))
        return DispatchCallIR(
            dst_rank=shape.rank_id, wave=w.index, aiv1=core,
            global_row_begin=core_global_begin, global_row_end=core_global_end,
            experts=tuple(expert_irs))

    # ---- 主入口 ----


    def _build_shared_expert(self, shape, km, ACT_HALF, TILE_M, TILE_N, p, c):
        if shape.shared_expert_num <= 0:
            return None
        m_tot_s = shape.token_num
        sched_n_s = self._gmm1_device_scheduler_n(shape, ACT_HALF)
        nt1_s = ceil_div(sched_n_s, TILE_N)
        mg_s = ceil_div(m_tot_s, TILE_M)
        g1s, as_events = [], []
        sc1 = BlockCursor(p, 0)
        for ti in range(mg_s * nt1_s):
            mg, nt = swizzle_coord(ti, mg_s, nt1_s, km.swizzle_offset, km.swizzle_direction)
            m_rows = min(TILE_M, m_tot_s - mg * TILE_M)
            logical_n = min(TILE_N, sched_n_s - nt * TILE_N)
            core = sc1.owners(1)[0]
            g1s.append(self._event(
                f"shared.gmm1.m{mg}.n{nt}", (f"AIC:{core}",),
                c.gmm1_tile(m_rows, shape.h, logical_n),
                meta={"stage": "shared_gmm1", "m_rows": m_rows}))
            as_events.append(self._event(
                f"shared.act.m{mg}.n{nt}", (f"AIV0:{core}",),
                c.activation_tile(m_rows, logical_n),
                deps=(g1s[-1],), meta={"stage": "shared_act", "m_rows": m_rows}))
        return self._event("shared.head_done", (), 0.0, deps=tuple(as_events),
                           meta={"stage": "shared_head_done"})

    # stage 建图函数在同包各文件: dispatch.py / gmm1.py / activation.py /
    # gmm2.py / combine.py — 状态经 BuildContext (context.py) 传递.

    # ---- 尾段 ----

    def _add_epilogue(self, shape, km, ACT_HALF, p, c):
        last_combine = {}
        for ev in self.events:
            if str(ev.meta.get("stage", "")) == "combine":
                last_combine[ev.meta.get("core")] = ev.name
        ce_deps = tuple(sorted(last_combine.values()))
        counts_export = self._event("epilogue.counts_export", (), T_COUNTS_EXPORT_US, deps=ce_deps,
                                    meta={"stage": "epilogue", "part": "counts_export"})
        core_sync = self._event("epilogue.output_core_sync", (), T_CORE_SYNC_BARRIER_US, deps=(counts_export,),
                                meta={"stage": "epilogue", "part": "output_core_sync"})
        tail_head = core_sync
        if shape.shared_expert_num > 0:
            w2_bytes = (shape.hidden_dim // ACT_HALF) * shape.h
            shared_gmm2 = self._event("epilogue.shared_gmm2", (), w2_bytes / BW_UNPERMUTE_AGG,
                                      deps=(core_sync,), meta={"stage": "epilogue", "part": "shared_gmm2"})
            tail_head = shared_gmm2
        rank_sync = self._event("epilogue.output_rank_sync", (), T_RANK_SYNC_RTT_US, deps=(tail_head,),
                                meta={"stage": "epilogue", "part": "output_rank_sync"})
        out_init = self._event("epilogue.output_buffer_init", (), T_OUTPUT_INIT_US, deps=(rank_sync,),
                               meta={"stage": "epilogue", "part": "output_buffer_init"})
        unpermute_bytes = shape.token_num * (shape.topk * shape.h * 2 + shape.h * 2)
        if shape.shared_expert_num > 0:
            unpermute_bytes += shape.shared_expert_num * shape.token_num * shape.h * 2
        unpermute = self._event("epilogue.unpermute", (), unpermute_bytes / BW_UNPERMUTE_AGG,
                                deps=(out_init,), meta={"stage": "epilogue", "part": "unpermute"})
        self._event("epilogue.finalize", (), T_FINALIZE_US, deps=(unpermute,),
                    meta={"stage": "epilogue", "part": "finalize"})

    # ---- 完成事件 ----

    def _add_completion(self, p, policy, gmm1_act_history):
        for core in range(p):
            depth = policy.gmm1_activation_depth
            aic_deps = gmm1_act_history[core][-depth:]
            self._event(f"moe_expert_stage_done.aic.c{core}", (f"AIC:{core}",), 0.0,
                        deps=aic_deps,
                        meta={"stage": "moe_stage_done", "role": "aic", "core": core})
            self._event(f"moe_expert_stage_done.aiv0.c{core}", (f"AIV0:{core}",), 0.0,
                        meta={"stage": "moe_stage_done", "role": "aiv0", "core": core})
            self._event(f"moe_expert_stage_done.aiv1.c{core}", (f"AIV1:{core}",), 0.0,
                        meta={"stage": "moe_stage_done", "role": "aiv1", "core": core})