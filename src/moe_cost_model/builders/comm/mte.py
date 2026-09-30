"""MTE 传输: AIV 触发 DataCopyPad 直写目的窗 (源推模型).

MTE 路径. dispatch 前瞻配速读 ctx.last_combine_by_core;
combine 为配对 tile, 每 GMM2 tail 一个同核 AIV1 事件.
"""
from __future__ import annotations

from typing import Dict, Tuple

from ...config.hardware import ceil_div
from ...costs import DispatchDataLayout
from ..base import build_dispatch_expert_ir
from ..context import BuildContext
from .base import CombineTransport, DispatchTransport


class MteDispatch(DispatchTransport):
    """按 (专家, m-group) 分块, 一块归一个 AIV1 核.

    切分层级: 先按专家, 每个专家内再按 m-group (tile_m 行一组). 一个块的全部行
    由同一个核搬运, 块内再按源卡切段 (不同源卡要从不同的卡上读)。

    核号按全局 m-group 序轮转 —— 同一个块无论落在哪个波, 归属的核都不变。
    块数 = Σ_e ceil(专家 e 的行数 / tile_m); 块数少于核数时, 多出来的核在
    dispatch 阶段没有活。
    """

    @staticmethod
    def _block_owner(shape, km, expert: int, group: int, p: int) -> int:
        """(专家, m-group) → AIV1 核号: 全局 m-group 序对核数取模."""
        prefix = sum(ceil_div(n, km.tile_m) for n in shape.expert_tokens[:expert])
        return (prefix + group) % p

    def add_wave(self, builder, ctx: BuildContext, w, shape, km, p, c, policy,
                 shared_gates):
        # 配速深度 = dispatch 偏移 + 1, 与显式 wave_offsets 对齐;
        # 缺省 (= dispatch_lookahead) 时与旧值逐字节一致
        la = policy.effective_wave_offsets(shape.token_num).dispatch + 1
        TILE_M = km.tile_m
        wave_contrib: Dict[Tuple[int, int], list] = {}
        layout0 = shape.dispatch_layout or DispatchDataLayout.from_hidden(shape.h)
        b_row = layout0.bytes_read_per_row()

        blocks_by_core: Dict[int, list] = {}
        for sl in w.slices:
            fg = sl.row_begin // TILE_M
            for lg in range(sl.m_groups):
                group = fg + lg
                owner = self._block_owner(shape, km, sl.expert, group, p)
                blocks_by_core.setdefault(owner, []).append((sl, group))

        for core in range(p):
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

            first_remote = True
            for sl, group in blocks_by_core.get(core, ()):
                row_begin = group * TILE_M
                row_end = min(row_begin + TILE_M, shape.expert_tokens[sl.expert])
                expert_ir = build_dispatch_expert_ir(
                    expert=sl.expert, dst_rank=shape.rank_id,
                    source_counts=shape.expert_source_tokens[sl.expert],
                    row_begin=row_begin, row_end=row_end, layout=layout0)
                seg_start = row_begin
                for si, (src, rows) in enumerate(expert_ir.segments):
                    seg_end = seg_start + rows
                    if rows <= 0:
                        continue
                    name = (f"W{w.index}.dispatch.c{core}.e{sl.expert}.g{group}"
                            f".s{si}.r{seg_start}_{seg_end}")
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
                            "expert": sl.expert, "mgroup": group, "src_rank": src,
                            "row_begin": seg_start, "row_end": seg_end, "rows": rows}
                    ev = builder._event(name, resources, duration, deps=deps,
                                        meta=meta, channel_bytes=ch_bytes)
                    # 一个块整体归一个核, 段必然落在本 m-group 内
                    wave_contrib.setdefault((sl.expert, group), []).append(
                        (ev, rows, core, call_name))
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
