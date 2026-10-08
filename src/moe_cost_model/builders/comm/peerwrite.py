"""直写对端窗口的传输后端: AIV 发 DataCopyPad 直接写入目的卡的对称窗口 (源推).

**名字按机制取, 不按搬运引擎取。** kernel 把这条拓扑称作 MTE
(`common/mega_moe_peermem.h`: `TOPO_TYPE_MTE = 0`, 与 URMA 对举), 但 MTE1/2/3 是**核内的
搬运单元**, 两条拓扑都在用它 —— 用它命名区分不了两者, 也会被读成一种通信协议。真正的
区别是跨卡怎么做: 这一条直接写对端的对称窗口, URMA 那一条走 GetUrmaCommHandle 的
GET/PUT (见 comm/urma.py)。对称窗口的布局是两条路径共用的, 见 kernel 的 peermem。

dispatch 的前瞻配速读 ctx.last_combine_by_core; combine 与 GMM2 tile 配对, 每个 GMM2
末段一个同核 AIV1 事件。
"""
from __future__ import annotations

from typing import Dict, Iterator, Tuple

from ...costs import DispatchDataLayout
from ..base import build_dispatch_expert_ir, rows_by_source_rank
from ..context import BuildContext
from ...config.hardware import BW_LOCAL_GM
from ..pipeline_expand import (CH_COMBINE_READ, CH_DISPATCH_READ,
                               CH_DISPATCH_WRITE, CH_HBM_WRITE)
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


class PeerWriteDispatch(DispatchTransport):

    def add_wave(self, builder, ctx: BuildContext, w, shape, km, p, c, policy,
                 shared_gates):
        # 配速深度 = dispatch 偏移 + 1, 与显式 wave_offsets 对齐;
        # 缺省 (= dispatch_lookahead) 时与旧值逐字节一致
        la = policy.effective_wave_offsets(shape.token_num).dispatch + 1
        TILE_M = km.tile_m
        wave_contrib: Dict[Tuple[int, int, list], list] = {}
        per_core_deps: Dict[int, Tuple[tuple, str]] = {}
        call_irs = [builder._dispatch_call_ir(shape, w, core) for core in range(p)]
        layout0 = shape.dispatch_layout or DispatchDataLayout.from_hidden(shape.h)
        # "pooled" 切法下没有按核的调用结构, 所以不发 dispatch_call 事件 —— 每波每核
        # 的调用开销由 Event.once_per_core 挂在该核本波第一段 dispatch 上。
        # "precut" 切法保留这个事件: 实测 trace 有 DISPATCH_SCHEDULE 包络,
        # tools/compare_measured.py 按它对齐。
        emit_call_events = builder.options.dispatch_partition == "precut"
        call_us = c.dispatch_mechanistic.call_base_us()
        for core in range(p):
            deps = [shared_gates] if shared_gates else []
            pacing_wave = w.index - la
            pacing = builder.options.dispatch_pacing
            if pacing_wave >= 0 and pacing == "per_core":
                if core in ctx.last_combine_by_core:
                    deps.append(ctx.last_combine_by_core[core])
            elif pacing_wave >= 0 and pacing == "wave":
                deps.extend(ctx.combines_by_wave.get(pacing_wave, ()))
            elif pacing != "none" and pacing not in ("per_core", "wave"):
                raise ValueError(f"dispatch_pacing 只能是 per_core/wave/none, 收到 {pacing!r}")
            if emit_call_events:
                # 令牌跟着角色走 (见 config/roles.queue_token)
                q_aiv1 = (builder.options.role_queue_token("dispatch_call", core), 1)
                call_name = builder._event(
                    f"W{w.index}.dispatch_call.c{core}",
                    (builder.options.role_resource("dispatch_call", core),),
                    call_us, deps=deps,
                    acquires=(q_aiv1,), releases=(q_aiv1,),
                    meta={"stage": "dispatch_call", "wave": w.index, "core": core})
            else:
                call_name = None

            per_core_deps[core] = (tuple(deps), call_name)

        # ---- 一份 dispatch 工作 = (专家切片, 源卡段) 切出来的一批行 ----
        # "谁去取哪些行"是**调度决策**, 不该写在建图里。两种切法:
        #   "pooled" (缺省): 不做按核预切, 只按 rows_per_item 切整个切片; 核号只是
        #                    轮转占位, AIV1 入池后由调度器在派发时刻决定
        #   "precut":        先按均衡+轮转把波的行分给 p 个核, 每核再切批 —— 某实现
        #                    的分工方式, compare_measured 要用它和实测 trace 对齐
        mode = builder.options.dispatch_partition
        if mode not in ("pooled", "precut"):
            raise ValueError(f"dispatch_partition 只能是 pooled/precut, 收到 {mode!r}")
        b_row = layout0.bytes_read_per_row()
        b_write = layout0.bytes_written_per_row()
        batch_rows = builder.options.dispatch_rows_per_item or layout0.route_items_per_batch
        first_remote: Dict[int, bool] = {core: True for core in range(p)}

        def emit(core, sl, si, src, b_begin, b_rows):
            deps_c, call_name_c = per_core_deps[core]
            b_end = b_begin + b_rows
            rate_r = (c.dispatch_mechanistic.bw_local_bytes_per_us if src == shape.rank_id
                      else c.dispatch_mechanistic.bw_remote_bytes_per_us)
            bw_loc = c.dispatch_mechanistic.bw_local_bytes_per_us
            bw_rem = c.dispatch_mechanistic.bw_remote_bytes_per_us
            # C1: 名字不带核号。行区间按核互不重叠, 所以
            # (专家, 段, 行区间) 已经唯一标识这一份搬运工作。
            name = (f"W{w.index}.dispatch.e{sl.expert}.s{si}"
                    f".r{b_begin}_{b_end}")
            resources = [builder.options.role_resource("dispatch", core)]
            if builder.options.serialize_dispatch_comm and src != shape.rank_id:
                resources.append("DISPATCH_COMM")
            duration = c.dispatch_mechanistic.segment_us(
                src, shape.rank_id, b_rows, layout0) + c.dispatch_ready_publish_us
            # 每行: 读 rowBytes (源卡窗口) + 写 b_write (本卡 workspace)。
            rx, wx = b_rows * b_row, b_rows * b_write
            ch_bytes = [(CH_DISPATCH_READ, rx, rate_r),
                        (CH_DISPATCH_WRITE, wx, bw_loc)]
            if src != shape.rank_id:
                if first_remote[core]:
                    # T_GMM1_OVERLAP 暂时留着: dispatch 与 GMM1 抢访存的机制要靠
                    # 整卡访存带宽的实测才能算出来。那天到了这一项必须同时删掉。
                    duration += c.dispatch_mechanistic.gmm1_overlap_us_per_call
                    first_remote[core] = False
                ch_bytes += [(f"fab_src:{src}", rx, bw_rem),
                             (f"fab_dst:{shape.rank_id}", rx, bw_rem)]
            meta = {"stage": "dispatch", "wave": w.index, "core": core,
                    "expert": sl.expert, "src_rank": src,
                    "row_begin": b_begin, "row_end": b_end, "rows": b_rows}
            ev = builder._event(name, resources, duration, deps=deps_c,
                                meta=meta, channel_bytes=tuple(ch_bytes))
            if not emit_call_events and call_us > 0:
                # 每 (波, 核) 只算一次, 落在该核本波真正搬数据的第一段上
                builder.events[-1].once_per_core = (f"W{w.index}.dispatch_call", call_us)
            for grp in range(b_begin // TILE_M, (b_end - 1) // TILE_M + 1):
                gb = grp * TILE_M
                ge = min(gb + TILE_M, shape.expert_tokens[sl.expert])
                rows_g = max(0, min(b_end, ge) - max(b_begin, gb))
                if rows_g:
                    wave_contrib.setdefault((sl.expert, grp), []).append(
                        (ev, rows_g, core, call_name_c))

        if mode == "precut":
            for core in range(p):
                rel_begin, count = builder._rotated_balanced_range(w.rows, core, p,
                                                                   w.begin.global_row)
                if count == 0:
                    continue
                cb = w.begin.global_row + rel_begin
                ce = cb + count
                expert_ir_by_id = {e.expert: e for e in call_irs[core].experts}
                for sl in w.slices:
                    ob = max(cb, sl.global_row_begin)
                    oe = min(ce, sl.global_row_end)
                    if ob >= oe:
                        continue
                    lb = sl.row_begin + (ob - sl.global_row_begin)
                    seg_start = lb
                    for si, (src, rows) in enumerate(expert_ir_by_id[sl.expert].segments):
                        seg_end = seg_start + rows
                        if rows > 0:
                            for b_begin, b_rows in _route_batches(seg_start, rows, batch_rows):
                                emit(core, sl, si, src, b_begin, b_rows)
                        seg_start = seg_end
        else:
            # 轮转只是占位: 它给静态绑定一个中性的分核 (不是 kernel 的那套),
            # AIV1 入池时由调度器覆盖。
            nxt = 0
            for sl in w.slices:
                ir = build_dispatch_expert_ir(
                    expert=sl.expert, dst_rank=shape.rank_id,
                    source_counts=shape.expert_source_tokens[sl.expert],
                    row_begin=sl.row_begin, row_end=sl.row_end, layout=layout0)
                seg_start = sl.row_begin
                for si, (src, rows) in enumerate(ir.segments):
                    seg_end = seg_start + rows
                    if rows > 0:
                        for b_begin, b_rows in _route_batches(seg_start, rows, batch_rows):
                            emit(nxt % p, sl, si, src, b_begin, b_rows)
                            nxt += 1
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
                    # "rows" 切法下没有 dispatch_call 事件, 这一项为空
                    "contributor_call_events": tuple(
                        x[3] for x in contrib if x[3] is not None)})
                ctx.dispatch_ready_event[key] = rn


def _spread_slots(options, shape, rows: int) -> float:
    """这 rows 行的写出落点铺开在多少个槽位里 (ModelOptions.combine_layout).

    token 散射: 落点 = (tokenIdx·topK + topkIdx)·n, 所以整张卡的落点空间是
        token 数 x topk 个槽位, 本窗这几行散落其中。
    按专家连续: 落点连续, 跨度就是行数本身 (= 不散开)。
    """
    layout = options.combine_layout
    if layout == "expert_contiguous":
        return float(rows)
    if layout != "token_scatter":
        raise ValueError(
            f'combine_layout 只能是 "token_scatter" / "expert_contiguous", 收到 {layout!r}')
    return float(shape.token_num * shape.topk)


class PeerWriteCombine(CombineTransport):
    """combine 的两种粒度 (ModelOptions.combine_granularity):

      "per_tile"   与每个 GMM2 tile 1:1 配对、紧跟其后。
      "per_expert" 一个专家切片一个事件, 等该切片全部 GMM2 段做完。

    粒度只改"一个事件覆盖多少工作"; 跑在哪个角色由 ModelOptions.roles 决定, 两者正交。
    """

    def __init__(self):
        #: 整片粒度 (0) 下累计待批: (si, expert) -> [gmm2 末段名], 以及行列范围
        self._pending = {}
        #: 攒 N 个 n-tile 的粒度 (>=2) 下累计待批: (si, expert, 行范围) -> 正在攒的那一项。
        #: **不按核分组**: combine 从 GM 读 GMM2 的输出 (StageLink location="gm"),
        #: 与 GMM2 同核不是物理约束, 所以一个 combine 事件可以吃不同核产的 tile。
        #: 按核分组会让这个参数失效 —— 轮转/晚绑定下同一个核拿到的是**不相邻**的 n-tile。
        self._runs = {}

    def on_gmm2_tile(self, builder, ctx: BuildContext, w, shape, si, sl, t, label,
                     ntile, core, gname, global_group, call_iteration):
        grain = builder.options.grain("combine")
        if grain >= 2:
            self._accumulate(builder, ctx, w, shape, si, sl, t, ntile, core,
                             gname, global_group, call_iteration, grain)
            return
        if grain == 0:
            key = (si, sl.expert)
            slot = self._pending.setdefault(
                key, {"deps": [], "cores": [], "sl": sl, "si": si,
                      "call_iteration": call_iteration, "rows": set()})
            slot["deps"].append(gname)
            slot["cores"].append(core)
            slot["rows"].add((t.row_begin, t.row_end))
            return
        self._emit_tile(builder, ctx, w, shape, si, sl, core, gname,
                        global_group, call_iteration, label=label, ntile=ntile,
                        row_begin=t.row_begin, row_end=t.row_end,
                        col_begin=t.col_begin, col_end=t.col_end,
                        deps=[gname], n_tiles=1)

    def _accumulate(self, builder, ctx, w, shape, si, sl, t, ntile, core, gname,
                    global_group, call_iteration, grain) -> None:
        """攒 N 个相邻 n-tile 再发一个 combine (同核、同行范围、列相邻).

        列必须相邻: combine 的代价是 rows x cols 的字节, 不连续的列并集表达不出来。
        换核/换行范围/列不连续都先把手上那一项发掉 —— 粒度是上界, 不是凑数配额。
        """
        key = (si, sl.expert, t.row_begin, t.row_end)
        cur = self._runs.get(key)
        if cur is not None and cur["col_end"] != t.col_begin:
            # 列不相邻: 先把手上那一项发掉 (粒度是上界, 不是凑数配额)
            self._flush_run(builder, ctx, w, shape, key)
            cur = None
        if cur is None:
            self._runs[key] = {
                "si": si, "expert": sl.expert, "sl": sl, "ntile": ntile,
                "row_begin": t.row_begin, "row_end": t.row_end,
                "col_begin": t.col_begin, "col_end": t.col_end,
                "deps": [gname], "group": global_group, "core": core,
                "call_iteration": call_iteration, "n": 1}
        else:
            cur["col_end"] = t.col_end
            cur["deps"].append(gname)
            cur["n"] += 1
        if self._runs[key]["n"] >= grain:
            self._flush_run(builder, ctx, w, shape, key)

    def _flush_run(self, builder, ctx, w, shape, key) -> None:
        r = self._runs.pop(key, None)
        if r is None:
            return
        # 落核: 取第一个成员所在的核 (晚绑定下调度器会重新决定)
        core = r["core"]
        tag = f"m{r['row_begin']}_{r['row_end']}.n{r['col_begin']}_{r['col_end']}"
        self._emit_tile(builder, ctx, w, shape, r["si"], r["sl"], core,
                        r["deps"][0], r["group"], r["call_iteration"],
                        label=tag, ntile=r["ntile"],
                        row_begin=r["row_begin"], row_end=r["row_end"],
                        col_begin=r["col_begin"], col_end=r["col_end"],
                        deps=sorted(r["deps"]), n_tiles=r["n"])

    def _emit_tile(self, builder, ctx, w, shape, si, sl, core, gname,
                   global_group, call_iteration, *, label, ntile,
                   row_begin, row_end, col_begin, col_end, deps, n_tiles):
        rows, cols = row_end - row_begin, col_end - col_begin
        c = builder.costs
        q_aiv1 = (builder.options.role_queue_token("combine", core), 1)
        # C1: 名字不带核号 (见 gmm1.py 的说明)
        cname = f"W{w.index}.E{sl.expert}.S{si}.combine.{label}"
        # 本窗每行要写回它的来源卡: 行区间按源卡分段, 逐卡行数精确数出。
        # 跨卡行是 COMBINE 的主导项, 所以 EP 摆放/本地亲和度会直接改 combine 代价。
        abs_begin = sl.row_begin + row_begin
        by_dst = rows_by_source_rank(shape.expert_source_tokens[sl.expert],
                                     abs_begin, abs_begin + rows)
        remote_rows = sum(n for d, n in enumerate(by_dst) if d != shape.rank_id)
        # 片间信道: 逐目的卡一条边 (fab_src = 流量离开本卡, fab_dst = 到达对端),
        # 与 dispatch 的方向语义一致。争用由速率服务器裁决, 不折进事件时长 ——
        # 时长用无争用带宽, 28 个核同时写同一条 fab 的降速是调度出来的。
        row_bytes = c.combine_write_bytes_per_row(cols)
        bw_fab = c.dispatch_mechanistic.bw_remote_bytes_per_us
        ch_bytes = tuple(
            ch for d, n in enumerate(by_dst) if d != shape.rank_id and n
            for ch in ((f"fab_src:{shape.rank_id}", n * row_bytes, bw_fab),
                       (f"fab_dst:{d}", n * row_bytes, bw_fab)))
        # 本卡侧的两股也要申报 (与 AnalyticalCombineCosts.tile 的三段逐项对应):
        #   读回 GMM2 tile + 路由元数据 (GM→UB)
        #   目的卡 == 本卡的那些行的写出
        # 2026-10-05 之前这两股没申报, 而相位流水那条路径反而用
        # "base_dur x BW_SCATTER" 从**时长**倒推出一个 hbm_write 字节数 —— 方向是反的,
        # 用的还是标着"旧口径, 已不用"的常数, 而且只在开了相位流水时才出现
        # (换一个编排参数不该改变搬了多少字节)。现在按字节直接申报。
        local_rows = rows - remote_rows
        # combine_read_bytes 是 PrimitiveCosts 的必填字段, 所以这里直接调 ——
        # 原先有个 getattr 兜底, 那正是让两条入口静默分叉的东西。
        read_back = float(c.combine_read_bytes(rows, cols))
        local_write = float(local_rows * row_bytes)
        # 读与写分开申报, 且读走自己的通路名 —— 混进 hbm_write 会让
        # "不物化就不写 GM" (test_onchip_declares_no_act_gm_write) 这类断言失去意义:
        # 那条断言问的是 ACT 写没写, 不是 COMBINE 读没读。
        if read_back:
            ch_bytes = ch_bytes + ((CH_COMBINE_READ, read_back,
                                    float(BW_LOCAL_GM)),)
        if local_write:
            ch_bytes = ch_bytes + ((CH_HBM_WRITE, local_write,
                                    float(BW_LOCAL_GM)),)
        spread = _spread_slots(builder.options, shape, rows)
        builder._event(cname, (builder.options.role_resource("combine", core),),
                       c.combine_tile(rows, cols, remote_rows, spread)
                       + c.combine_ack_us,
                       deps=deps, acquires=(q_aiv1,), releases=(q_aiv1,),
                       channel_bytes=ch_bytes,
                       meta={"stage": "combine", "wave": w.index,
                             "call_iteration": call_iteration, "expert": sl.expert,
                             "slice": si, "mgroup": global_group, "ntile": ntile,
                             "col_begin": col_begin, "col_end": col_end,
                             "row_begin": row_begin, "row_end": row_end,
                             "logical_n": cols, "core": core, "m_rows": rows,
                             "tiles_in_event": n_tiles,
                             "remote_rows": remote_rows, "rows_by_dst": by_dst,
                             "spread_slots": spread})
        ctx.gmm2_combine_history[core].append(cname)
        ctx.last_combine_by_core[core] = cname
        ctx.combines_by_wave.setdefault(w.index, []).append(cname)

    def flush_wave(self, builder, ctx: BuildContext, w, shape, km, p):
        for key in list(self._runs):
            self._flush_run(builder, ctx, w, shape, key)
        """per_expert 粒度: 每个专家切片发一个 combine, 等该切片全部 GMM2 段做完.

        与 per_tile 的实质差别有两处, 都在成本里体现:
          * 路由元数据每行只读一次 (per_tile 下每个 n-tile 都读一遍本窗 m 行);
          * 写出是整片 h 列一次, 而不是按 n-tile 分 20 次。
        代价在 DAG 上: 它等整个切片的 GMM2, 不是等一个 tile。
        """
        if not self._pending:
            return
        c = builder.costs
        for idx, (key, slot) in enumerate(sorted(self._pending.items())):
            si, expert = key
            sl = slot["sl"]
            # 落核: 轮转占位 (晚绑定下由调度器决定; 静态绑定下给一个中性分配)
            core = slot["cores"][0] if slot["cores"] else idx % max(1, p)
            rows = sum(end - begin for begin, end in sorted(slot["rows"]))
            by_dst = rows_by_source_rank(shape.expert_source_tokens[expert],
                                        sl.row_begin, sl.row_begin + rows)
            remote_rows = sum(n for d, n in enumerate(by_dst) if d != shape.rank_id)
            row_bytes = c.combine_write_bytes_per_row(shape.h)
            bw_fab = c.dispatch_mechanistic.bw_remote_bytes_per_us
            ch_bytes = tuple(
                ch for d, n in enumerate(by_dst) if d != shape.rank_id and n
                for ch in ((f"fab_src:{shape.rank_id}", n * row_bytes, bw_fab),
                           (f"fab_dst:{d}", n * row_bytes, bw_fab)))
            q_aiv1 = (builder.options.role_queue_token("combine", core), 1)
            cname = f"W{w.index}.E{expert}.S{si}.combine.expert"
            spread = _spread_slots(builder.options, shape, rows)
            builder._event(
                cname, (builder.options.role_resource("combine", core),),
                c.combine_tile(rows, shape.h, remote_rows, spread)
                + c.combine_ack_us,
                deps=sorted(slot["deps"]), acquires=(q_aiv1,), releases=(q_aiv1,),
                channel_bytes=ch_bytes,
                meta={"stage": "combine", "wave": w.index,
                      "call_iteration": slot["call_iteration"], "expert": expert,
                      "slice": si, "granularity": "per_expert",
                      "col_begin": 0, "col_end": shape.h,
                      "row_begin": 0, "row_end": rows,
                      "logical_n": shape.h, "core": core, "m_rows": rows,
                      "remote_rows": remote_rows, "rows_by_dst": by_dst,
                      "spread_slots": spread})
            ctx.gmm2_combine_history[core].append(cname)
            ctx.last_combine_by_core[core] = cname
            ctx.combines_by_wave.setdefault(w.index, []).append(cname)
        self._pending = {}
