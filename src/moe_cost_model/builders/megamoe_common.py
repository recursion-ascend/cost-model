"""第 4 层: 两份 MegaMoE 建图器共用的实现专属部分.

这些东西原先住在 EventBuilderBase 里, 于是任何新建图器只要继承基类就白得一条
MegaMoE 的尾段 (counts_export / output_core_sync / output_rank_sync /
output_buffer_init / unpermute / finalize 六个事件)、共享专家的两段、dispatch 的段枚举, 以及 GMM1 调度宽度 = hidden_dim/activation_n_half 这个 SwiGLU 假设。它们都不是
"建图"这件事的共性, 是 MegaMoE 这份实现的事实。

用法: ``class MteEventBuilder(MegaMoeBuilderMixin, EventBuilderBase)``。另一份实现不
混入它, 于是不会继承这些。

**只搬位置, 一行逻辑没改** —— 判据是 golden 40 个用例零差异。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..config.hardware import (
    BW_UNPERMUTE_AGG, T_CORE_SYNC_BARRIER_US, T_COUNTS_EXPORT_US,
    T_FINALIZE_US, T_OUTPUT_INIT_US, T_RANK_SYNC_RTT_US, ceil_div,
)
from ..costs import DispatchDataLayout
from ..planning.waves import ExpertSlice, Wave, swizzle_coord
from ..shape import BlockCursor, MegaMoeShape
from .base import rows_by_source_rank


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


class MegaMoeBuilderMixin:
    """MegaMoE 两条路径 (MTE / URMA Layered) 共用的实现专属建图."""

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
        """共享专家前半段: GMM1 + ACT tile, 排在 MoE dispatch 之前.

        返回门控事件名 (全部共享 ACT 完成), dispatch_call 依赖它.
        后半段 (共享 GMM2) 在尾段, 见 _add_shared_gmm2.
        """
        self.shared_act_by_group = {}
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
                f"shared.gmm1.m{mg}.n{nt}", (self.options.role_resource("shared_gmm1", core),),
                c.gmm1_tile(m_rows, shape.h, logical_n),
                meta={"stage": "shared_gmm1", "m_rows": m_rows}))
            as_events.append(self._event(
                f"shared.act.m{mg}.n{nt}", (self.options.role_resource("shared_act", core),),
                c.activation_tile(m_rows, logical_n),
                deps=(g1s[-1],), meta={"stage": "shared_act", "m_rows": m_rows}))
            self.shared_act_by_group.setdefault(mg, []).append(as_events[-1])
        return self._event("shared.head_done", (), 0.0, deps=tuple(as_events),
                           meta={"stage": "shared_head_done"})

    def _add_shared_gmm2(self, shape, km, ACT_HALF, p, c, after: str) -> str:
        """共享专家后半段: GMM2 按 tile 建事件, 占 AIC 核, 时长取 GMM2 公式 (纯计算).

        每个 tile 依赖 after (尾段前序事件) 与本 m-group 的全部共享 ACT
        (GMM2 的输入是该 m-group 的 ACT 产出, 覆盖整个 K).
        返回汇合事件名 (全部共享 GMM2 tile 完成).
        """
        tile_m, tile_n = km.tile_m, km.tile_n
        k_gmm2 = shape.hidden_dim // ACT_HALF
        n_tiles = ceil_div(shape.h, tile_n)
        m_groups = ceil_div(shape.token_num, tile_m)
        cursor = BlockCursor(p, 0)
        tiles = []
        for ti in range(m_groups * n_tiles):
            mg, nt = swizzle_coord(ti, m_groups, n_tiles,
                                   km.swizzle_offset, km.swizzle_direction)
            m_rows = min(tile_m, shape.token_num - mg * tile_m)
            logical_n = min(tile_n, shape.h - nt * tile_n)
            core = cursor.owners(1)[0]
            tiles.append(self._event(
                f"shared.gmm2.m{mg}.n{nt}", (self.options.role_resource("shared_gmm2", core),),
                c.gmm2_tile(m_rows, k_gmm2, logical_n),
                deps=(after, *self.shared_act_by_group.get(mg, ())),
                meta={"stage": "shared_gmm2", "m_rows": m_rows, "mgroup": mg,
                      "ntile": nt, "logical_n": logical_n, "core": core}))
        return self._event("epilogue.shared_gmm2_done", (), 0.0, deps=tuple(tiles),
                           meta={"stage": "epilogue", "part": "shared_gmm2_done"})

    # ---- 尾段 ----

    def _add_epilogue(self, shape, km, ACT_HALF, p, c, drains=()):
        """尾段链. 门是每核三引擎的排空节点 (drains), 不是"每核最后一个 COMBINE".

        内核里 WAIT_GMM_DRAIN (实测 trace 恰好 84 个 = 28 核 x AIC/AIV0/AIV1) 在
        WAIT_OUTPUT_CORE_SYNC → COUNTS_EXPORT 之前, 三个引擎都要各自排空。只等
        COMBINE 会漏掉 AIC 的 GMM2 尾块与 AIV0 的 ACT: bs=36 用例里核 2~5 的最后
        一个 ACT (209.6us) 晚于该核最后一个 COMBINE (149.9us), 只是恰好被核 0/1 的
        晚 COMBINE (231.0us) 盖住 —— 换个路由就会让尾段起得太早。
        """
        # C5: 五项固定开销可配置 (ModelOptions.epilogue_overheads); 缺省沿用实测常数
        oh = getattr(self.options, "epilogue_overheads", None)

        def _oh(field: str, fallback: float) -> float:
            if oh is None:
                return fallback
            v = getattr(oh, field, 0.0)
            return v if (v or getattr(oh, "literal", False)) else fallback

        counts_export = self._event("epilogue.counts_export", (),
                                    _oh("counts_export_us", T_COUNTS_EXPORT_US),
                                    deps=tuple(drains),
                                    meta={"stage": "epilogue", "part": "counts_export"})
        core_sync = self._event("epilogue.output_core_sync", (),
                                _oh("core_sync_us", T_CORE_SYNC_BARRIER_US),
                                deps=(counts_export,),
                                meta={"stage": "epilogue", "part": "output_core_sync"})
        tail_head = core_sync
        if shape.shared_expert_num > 0:
            tail_head = self._add_shared_gmm2(shape, km, ACT_HALF, p, c, after=core_sync)
        rank_sync = self._event("epilogue.output_rank_sync", (),
                                _oh("rank_sync_us", T_RANK_SYNC_RTT_US), deps=(tail_head,),
                                meta={"stage": "epilogue", "part": "output_rank_sync"})
        out_init = self._event("epilogue.output_buffer_init", (),
                               _oh("output_init_us", T_OUTPUT_INIT_US), deps=(rank_sync,),
                               meta={"stage": "epilogue", "part": "output_buffer_init"})
        unpermute_bytes = shape.token_num * (shape.topk * shape.h * 2 + shape.h * 2)
        if shape.shared_expert_num > 0:
            unpermute_bytes += shape.shared_expert_num * shape.token_num * shape.h * 2
        unpermute = self._event("epilogue.unpermute", (), unpermute_bytes / BW_UNPERMUTE_AGG,
                                deps=(out_init,), meta={"stage": "epilogue", "part": "unpermute"})
        self._event("epilogue.finalize", (), _oh("finalize_us", T_FINALIZE_US),
                    deps=(unpermute,), meta={"stage": "epilogue", "part": "finalize"})

