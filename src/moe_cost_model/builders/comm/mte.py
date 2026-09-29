"""MTE 传输: AIV 触发 DataCopyPad 直写目的窗 (源推模型).

MTE 路径. dispatch 前瞻配速读 ctx.last_combine_by_core;
combine 为配对 tile, 每 GMM2 tail 一个同核 AIV1 事件.
"""
from __future__ import annotations

from typing import Dict, Tuple

from ...costs import DispatchDataLayout
from ..context import BuildContext
from .base import CombineTransport, DispatchTransport


class MteDispatch(DispatchTransport):

    def add_wave(self, builder, ctx: BuildContext, w, shape, km, p, c, policy,
                 shared_gates):
        # 配速深度 = dispatch 偏移 + 1, 与显式 wave_offsets 对齐;
        # 缺省 (= dispatch_lookahead) 时与旧值逐字节一致
        la = policy.effective_wave_offsets(shape.token_num).dispatch + 1
        TILE_M = km.tile_m
        wave_contrib: Dict[Tuple[int, int, list], list] = {}
        call_irs = [builder._dispatch_call_ir(shape, w, core) for core in range(p)]
        layout0 = shape.dispatch_layout or DispatchDataLayout.from_hidden(shape.h)
        for core in range(p):
            call_ir = call_irs[core]
            deps = [shared_gates] if shared_gates else []
            pacing_wave = w.index - la
            if pacing_wave >= 0 and core in ctx.last_combine_by_core:
                deps.append(ctx.last_combine_by_core[core])
            q_aiv1 = (f"Q:aiv1:c{core}", 1)
            call_name = builder._event(
                f"W{w.index}.dispatch_call.c{core}", (f"AIV1:{core}",),
                c.dispatch_mechanistic.call_base_us(), deps=deps,
                acquires=(q_aiv1,), releases=(q_aiv1,),
                meta={"stage": "dispatch_call", "wave": w.index, "core": core})

            rel_begin, count = builder._rotated_balanced_range(w.rows, core, p,
                                                               w.begin.global_row)
            if count == 0:
                continue
            cb = w.begin.global_row + rel_begin
            ce = cb + count
            expert_ir_by_id = {e.expert: e for e in call_ir.experts}
            b_row = layout0.bytes_read_per_row()
            first_remote = True
            for sl in w.slices:
                ob = max(cb, sl.global_row_begin)
                oe = min(ce, sl.global_row_end)
                if ob >= oe:
                    continue
                lb = sl.row_begin + (ob - sl.global_row_begin)
                expert_ir = expert_ir_by_id[sl.expert]
                seg_start = lb
                for si, (src, rows) in enumerate(expert_ir.segments):
                    seg_end = seg_start + rows
                    if rows <= 0:
                        continue
                    name = f"W{w.index}.dispatch.c{core}.e{sl.expert}.s{si}.r{seg_start}_{seg_end}"
                    resources = [f"AIV1:{core}"]
                    if builder.options.serialize_dispatch_comm and src != shape.rank_id:
                        resources.append("DISPATCH_COMM")
                    duration = c.dispatch_mechanistic.segment_us(
                        src, shape.rank_id, rows, layout0) + c.dispatch_ready_publish_us
                    ch_bytes = ()
                    if src != shape.rank_id:
                        if first_remote:
                            duration += c.dispatch_mechanistic.gmm1_overlap_us_per_call
                            first_remote = False
                        bx = rows * b_row
                        ch_bytes = (
                            (f"fab_src:{src}", bx, c.dispatch_mechanistic.bw_remote_bytes_per_us),
                            (f"fab_dst:{shape.rank_id}", bx, c.dispatch_mechanistic.bw_remote_bytes_per_us),
                        )
                    meta = {"stage": "dispatch", "wave": w.index, "core": core,
                            "expert": sl.expert, "src_rank": src,
                            "row_begin": seg_start, "row_end": seg_end, "rows": rows}
                    ev = builder._event(name, resources, duration, deps=deps,
                                        meta=meta, channel_bytes=ch_bytes)
                    for grp in range(seg_start // TILE_M, (seg_end - 1) // TILE_M + 1):
                        gb = grp * TILE_M
                        ge = min(gb + TILE_M, shape.expert_tokens[sl.expert])
                        rows_g = max(0, min(seg_end, ge) - max(seg_start, gb))
                        if rows_g:
                            wave_contrib.setdefault((sl.expert, grp), []).append(
                                (ev, rows_g, core, call_name))
                    seg_start = seg_end

        for sl in w.slices:
            fg = sl.row_begin // TILE_M
            for lg in range(sl.m_groups):
                group = fg + lg
                key = (sl.expert, group)
                if key in ctx.dispatch_ready_event:
                    raise ValueError(f"duplicate DispatchReady producer for {key}")
                required = max(0, min(TILE_M, shape.expert_tokens[sl.expert] - group * TILE_M))
                contrib = wave_contrib.get(key, [])
                got = sum(x[1] for x in contrib)
                if got != required:
                    raise ValueError(
                        f"DispatchReady row mismatch for expert={sl.expert}, group={group}: "
                        f"got {got}, expected {required}")
                deps = tuple(x[0] for x in contrib)
                rn = f"W{w.index}.dispatch_ready.e{sl.expert}.g{group}"
                builder._event(rn, (), 0.0, deps=deps, meta={
                    "stage": "dispatch_ready", "dst_rank": shape.rank_id,
                    "wave": w.index, "expert": sl.expert, "mgroup": group,
                    "required_rows": required, "contributed_rows": got,
                    "contributor_count": len(contrib),
                    "contributor_events": tuple(x[0] for x in contrib),
                    "contributor_rows": tuple(x[1] for x in contrib),
                    "contributor_cores": tuple(x[2] for x in contrib),
                    "contributor_call_events": tuple(x[3] for x in contrib)})
                ctx.dispatch_ready_event[key] = rn


class MteCombine(CombineTransport):

    def on_gmm2_tile(self, builder, ctx: BuildContext, w, si, sl, mg, nt, core,
                     m_rows, gmm2_logical_n, gname, global_group, call_iteration):
        c = builder.costs
        q_aiv1 = (f"Q:aiv1:c{core}", 1)
        cname = f"W{w.index}.E{sl.expert}.S{si}.combine.m{mg}.n{nt}.c{core}"
        builder._event(cname, (f"AIV1:{core}",),
                       c.combine_tile(m_rows, gmm2_logical_n) + c.combine_ack_us,
                       deps=[gname], acquires=(q_aiv1,), releases=(q_aiv1,),
                       meta={"stage": "combine", "wave": w.index,
                             "call_iteration": call_iteration, "expert": sl.expert,
                             "slice": si, "mgroup": global_group, "ntile": nt,
                             "logical_n": gmm2_logical_n, "core": core, "m_rows": m_rows})
        ctx.gmm2_combine_history[core].append(cname)
        ctx.last_combine_by_core[core] = cname

    def flush_wave(self, builder, ctx: BuildContext, w, shape, km, p):
        pass
