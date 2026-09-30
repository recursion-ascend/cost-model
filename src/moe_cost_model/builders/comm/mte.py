"""MTE 传输: AIV 触发 DataCopyPad 直写目的窗 (源推模型).

MTE 路径. dispatch 前瞻配速读 ctx.last_combine_by_core;
combine 为配对 tile, 每 GMM2 tail 一个同核 AIV1 事件.
"""
from __future__ import annotations

from typing import Dict, Iterator, Tuple

from ...costs import DispatchDataLayout
from ..base import rows_by_source_rank
from ..context import BuildContext
from ..pipeline_expand import CH_DISPATCH_READ, CH_DISPATCH_WRITE
from .base import CombineTransport, DispatchTransport


def _route_batches(row_begin: int, rows: int, batch_rows: int) -> Iterator[Tuple[int, int]]:
    """把一段的行区间按 routeItemsPerBatch 切批, 产出 (批起始行, 批行数).

    对应内核 DispatchRankTokens 的 while 批循环: 每批一次
    CopyTokensAndMetaForDispatch, 而 PROFILE 区间在那个函数里 —— 所以一批就是
    实测 trace 里的一个 DISPATCH_XFER/LOCAL 事件。段不超过一批时只产出一个,
    起止行与整段相同, 事件名与 meta 因此与未分批时逐字节一致。
    """
    step = max(1, int(batch_rows))
    off = 0
    while off < rows:
        n = min(step, rows - off)
        yield row_begin + off, n
        off += n


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
            b_write = layout0.bytes_written_per_row()
            batch_rows = layout0.route_items_per_batch
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
                    rate_r = (c.dispatch_mechanistic.bw_local_bytes_per_us if src == shape.rank_id
                              else c.dispatch_mechanistic.bw_remote_bytes_per_us)
                    bw_loc = c.dispatch_mechanistic.bw_local_bytes_per_us
                    bw_rem = c.dispatch_mechanistic.bw_remote_bytes_per_us
                    for b_begin, b_rows in _route_batches(seg_start, rows, batch_rows):
                        b_end = b_begin + b_rows
                        name = (f"W{w.index}.dispatch.c{core}.e{sl.expert}.s{si}"
                                f".r{b_begin}_{b_end}")
                        resources = [f"AIV1:{core}"]
                        if builder.options.serialize_dispatch_comm and src != shape.rank_id:
                            resources.append("DISPATCH_COMM")
                        duration = c.dispatch_mechanistic.segment_us(
                            src, shape.rank_id, b_rows, layout0) + c.dispatch_ready_publish_us
                        # 每行: 读 rowBytes (源卡窗口) + 写 b_write (本卡 workspace)。
                        # 读侧 dispatch_read, 写侧 dispatch_write。原先 dispatch 只申报
                        # 远端读的片间字节, 本卡读与全部写在调度器眼里根本不存在 ——
                        # DAG 里少了一整条访存通路。
                        rx, wx = b_rows * b_row, b_rows * b_write
                        ch_bytes = [(CH_DISPATCH_READ, rx, rate_r),
                                    (CH_DISPATCH_WRITE, wx, bw_loc)]
                        if src != shape.rank_id:
                            if first_remote:
                                # T_GMM1_OVERLAP 暂时留着: dispatch 与 GMM1 抢访存的
                                # 机制要靠"两者并到同一条聚合为整卡访存带宽的信道"
                                # 才能算出来, 而整卡聚合带宽尚无实测 (不能拿每核速率
                                # x核数当整卡值)。那天到了这一项必须同时删掉, 否则
                                # 就是双重计费。
                                duration += c.dispatch_mechanistic.gmm1_overlap_us_per_call
                                first_remote = False
                            ch_bytes += [(f"fab_src:{src}", rx, bw_rem),
                                         (f"fab_dst:{shape.rank_id}", rx, bw_rem)]
                        meta = {"stage": "dispatch", "wave": w.index, "core": core,
                                "expert": sl.expert, "src_rank": src,
                                "row_begin": b_begin, "row_end": b_end, "rows": b_rows}
                        ev = builder._event(name, resources, duration, deps=deps,
                                            meta=meta, channel_bytes=tuple(ch_bytes))
                        for grp in range(b_begin // TILE_M, (b_end - 1) // TILE_M + 1):
                            gb = grp * TILE_M
                            ge = min(gb + TILE_M, shape.expert_tokens[sl.expert])
                            rows_g = max(0, min(b_end, ge) - max(b_begin, gb))
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

    def on_gmm2_tile(self, builder, ctx: BuildContext, w, shape, si, sl, t, label,
                     ntile, core, gname, global_group, call_iteration):
        c = builder.costs
        q_aiv1 = (f"Q:aiv1:c{core}", 1)
        cname = f"W{w.index}.E{sl.expert}.S{si}.combine.{label}.c{core}"
        # 本窗每行要写回它的来源卡: 行区间按源卡分段, 逐卡行数精确数出。
        # 跨卡行是 COMBINE 的主导项, 所以 EP 摆放/本地亲和度会直接改 combine 代价。
        abs_begin = sl.row_begin + t.row_begin
        by_dst = rows_by_source_rank(shape.expert_source_tokens[sl.expert],
                                     abs_begin, abs_begin + t.rows)
        remote_rows = sum(n for d, n in enumerate(by_dst) if d != shape.rank_id)
        # 片间信道: 逐目的卡一条边 (fab_src = 流量离开本卡, fab_dst = 到达对端),
        # 与 dispatch 的方向语义一致。争用由速率服务器裁决, 不折进事件时长 ——
        # 时长用无争用带宽, 28 个核同时写同一条 fab 的降速是调度出来的。
        row_bytes = c.combine_write_bytes_per_row(t.cols)
        bw_fab = c.dispatch_mechanistic.bw_remote_bytes_per_us
        ch_bytes = tuple(
            ch for d, n in enumerate(by_dst) if d != shape.rank_id and n
            for ch in ((f"fab_src:{shape.rank_id}", n * row_bytes, bw_fab),
                       (f"fab_dst:{d}", n * row_bytes, bw_fab)))
        builder._event(cname, (f"AIV1:{core}",),
                       c.combine_tile(t.rows, t.cols, remote_rows) + c.combine_ack_us,
                       deps=[gname], acquires=(q_aiv1,), releases=(q_aiv1,),
                       channel_bytes=ch_bytes,
                       meta={"stage": "combine", "wave": w.index,
                             "call_iteration": call_iteration, "expert": sl.expert,
                             "slice": si, "mgroup": global_group, "ntile": ntile,
                             "col_begin": t.col_begin, "col_end": t.col_end,
                             "row_begin": t.row_begin, "row_end": t.row_end,
                             "logical_n": t.cols, "core": core, "m_rows": t.rows,
                             "remote_rows": remote_rows, "rows_by_dst": by_dst})
        ctx.gmm2_combine_history[core].append(cname)
        ctx.last_combine_by_core[core] = cname

    def flush_wave(self, builder, ctx: BuildContext, w, shape, km, p):
        pass
