"""第 4 层: GMM2 stage — head/tail 拆分事件 + 按行列范围的 ACT 依赖.

tile 网格由 shape.tile_grid 给出 (缺省 SwizzledTileGrid = kernel 现行为)。
依赖按 ACT 的行列范围挑, 不按构建序 — swizzle 蛇形遍历下构建序与坐标不对应。
head 窗只等覆盖 K[0,kL1) 的 ACT (流水最大化), 主体等覆盖 K[kL1,K) 的.
"""
from __future__ import annotations

from typing import List

from ..config.hardware import BW_L1_GM, select_kl1
from ..costs import gmm2_phase_split
from ..planning.tile_grid import STAGE_GMM2, validate_tiles
from .context import BuildContext
from .pipeline_expand import CH_GM_TO_L1
from .tiling import coalesce_tiles, resolve_grid, tile_label


def add_gmm2_wave(builder, ctx: BuildContext, w, shape, km, p, c, core_assign,
                  policy, TILE_M, TILE_N, ACT_HALF, call_iteration) -> None:
    grid = resolve_grid(shape)
    # 这条 stage 边的编排: 消费者沿 K 分几段就绪 + 中间结果落哪 (config/links.py)
    link = builder.options.link("activation", "gmm2")
    cursor = ctx.cursor
    k_gmm2 = shape.hidden_dim // ACT_HALF
    for si, sl in enumerate(w.slices):
        tiles = grid.plan(stage=STAGE_GMM2, rows=sl.rows, cols=shape.h, kernel=km)
        validate_tiles(tiles, rows=sl.rows, cols=shape.h, tile_m=TILE_M,
                       where=f"GMM2 专家 {sl.expert}")
        # 事件粒度 (ModelOptions.granularity): 一个 GMM2 事件覆盖多少个 tile。
        # 时长按成员逐个算再求和 —— 合并省的是事件间同步, 不省每 tile 的搬运与计算。
        items = coalesce_tiles(tiles, builder.options.grain("gmm2"),
                              where=f"GMM2 专家 {sl.expert}")
        tile_count = len(items)
        tile_costs = [sum(c.gmm2_tile(mt.rows, k_gmm2, mt.cols) for mt in members)
                      for _, members in items]
        if core_assign is not None:
            owners = core_assign.assign(tile_count, p, cursor.start,
                                        tile_costs=tile_costs)
            cursor.set(cursor.start + tile_count)
        else:
            owners = cursor.owners(tile_count)
        first_owned = [True] * p

        kl1 = select_kl1(sl.rows, k_gmm2, builder.options.gmm2_kl1,
                         tile_m=km.tile_m, tile_n=km.tile_n,
                         l1_size=km.l1_size, k_l1_base=km.l1_tile_k,
                         n_windows=km.l1_buf_num)

        for tile_idx, core in enumerate(owners):
            t, members = items[tile_idx]
            duration = tile_costs[tile_idx]
            mg = t.row_begin // TILE_M
            global_group = sl.row_begin // TILE_M + mg
            # GMM2 的 K 就是 GMM1 的输出列: 取行范围相交的 ACT, 它们必须覆盖整个 K
            acts = [a for a in ctx.activation_ready.get((sl.expert, global_group), ())
                    if a.row_begin < t.row_end and a.row_end > t.row_begin]
            _require_full_k(acts, k_gmm2, sl.expert, global_group, t)

            deps: List[str] = []
            q_aic2 = (f"Q:aic:c{core}", 1)
            combine_history = ctx.gmm2_combine_history[core]
            credit = policy.gmm2_combine_credit
            if credit is not None and len(combine_history) >= credit:
                deps.append(combine_history[-credit])

            if first_owned[core]:
                duration += c.gmm2_problem_startup_us
                first_owned[core] = False

            ordered = sorted(acts, key=lambda a: (a.col_begin, a.row_begin))
            bounds = _k_segment_bounds(k_gmm2, kl1, link.readiness)

            ntile = t.col_begin // TILE_N
            g2_res = builder.options.role_resource("gmm2", core)
            label = tile_label(t, sl.rows, shape.h, TILE_M, TILE_N)
            meta = {"stage": "gmm2", "wave": w.index,
                    "call_iteration": call_iteration, "expert": sl.expert,
                    "slice": si, "mgroup": global_group, "ntile": ntile,
                    "col_begin": t.col_begin, "col_end": t.col_end,
                    "row_begin": t.row_begin, "row_end": t.row_end,
                    "logical_n": t.cols, "core": core, "m_rows": t.rows,
                    "cursor_tile": tile_idx, "tiles_in_event": len(members)}
            # 同一个 tile 的各 K 段按时长占比分摊本 tile 的 B 流, 信道字节才不会在
            # 计算绑定时被整段时长放大。段的 K 范围决定它等哪些 ACT。
            phases = _sum_phases(c, members, k_gmm2)
            # GM→L1 访存量: B 流权重 K2·cols (每个 tile 都要读) + A 流激活 m·K2
            # (只在物化编排下存在: ACT 写 GM, GMM2 读回)。
            # 2026-10-05 之前只申报 A 流 —— B 流进了时长公式 (gmm2_phases 的
            # b_load = k2·cols/bw_b) 却没进字节申报, 于是全卡申报量**低于算法必搬的
            # 字节** (scenario_basic: 1660.9MB vs 2420.1MB, 差 759.2MB ≈ GMM2 权重
            # 805.3MB)。申报量低于算法下界在物理上不可能, 那是漏账不是口径差异。
            # GMM1 一直是两股都申报的 (a_bytes + b_bytes), 这里补上对称。
            a_gm = (sum(mt.rows * k_gmm2 for mt in members)
                    if link.materialised else 0)
            b_gm = sum(k_gmm2 * mt.cols for mt in members)
            # C1: 名字不带核号 (见 gmm1.py 的说明)
            gname = f"W{w.index}.E{sl.expert}.S{si}.gmm2.{label}"
            n_seg = len(bounds)
            prev: List[str] = []
            for j, (k_lo, k_hi) in enumerate(bounds):
                frac = (k_hi - k_lo) / k_gmm2
                seg_acts = [a.name for a in ordered
                            if a.col_begin < k_hi and a.col_end > k_lo]
                last = (j == n_seg - 1)
                # 末段用不带后缀的名字: combine 与 gmm2_tail_by_group 按它挂钩。
                # 两段时首段沿用 ".h"/part="head" —— 现有测试与 audit_edges 认这个名字。
                if last:
                    name, part = gname, "tail"
                elif n_seg == 2:
                    name, part = gname + ".h", "head"
                else:
                    name, part = f"{gname}.k{j}", f"k{j}"
                seg_meta = dict(meta, part=part)
                seg_bytes = (a_gm + b_gm) * frac
                seg_ch = (((CH_GM_TO_L1, seg_bytes, float(BW_L1_GM)),)
                          if seg_bytes else ())
                if phases is not None:
                    load_us, compute_us = phases
                    seg_meta["load_us"] = load_us * frac
                    seg_meta["compute_us"] = compute_us * frac
                # 首段持有队列 token; 后续段靠前一段的串接边保序, 不重复占用。
                if j == 0:
                    builder._event(name, (g2_res,), duration * frac,
                                   deps=deps + seg_acts, channel_bytes=seg_ch,
                                   acquires=(q_aic2,), releases=(q_aic2,), meta=seg_meta)
                else:
                    builder._event(name, (g2_res,), duration * frac,
                                   deps=prev + seg_acts, channel_bytes=seg_ch,
                                   meta=seg_meta)
                prev = [name]
            builder.gmm2_tail_by_group.setdefault((sl.expert, global_group), []).append(gname)

            # combine 走传输后端钩子: MTE 配对 tile / URMA 记录待批
            builder.combine_backend.on_gmm2_tile(
                builder, ctx, w, shape, si, sl, t, label, ntile, core,
                gname, global_group, call_iteration)


def _sum_phases(costs, members, k_gmm2: int):
    """粗粒度事件的 load/cube 相位 = 成员逐个拆分再相加 (任一成员不支持则整体不拆)."""
    total_load = total_compute = 0.0
    for mt in members:
        ph = gmm2_phase_split(costs, mt.rows, k_gmm2, mt.cols)
        if ph is None:
            return None
        total_load += ph[0]
        total_compute += ph[1]
    return (total_load, total_compute)


def _k_segment_bounds(k_gmm2: int, kl1: int, segments: int):
    """GMM2 沿 K 的分段边界 [(k_lo, k_hi), ...], 末段到 k_gmm2.

    segments == 2: 首个 kL1 块一段, 其余合成一段 (MEGAMOE_A8W8 用这个)。
        与旧实现逐字节等价: head 等 col_begin < kl1 的 ACT, tail 等 col_end > kl1 的。
    segments == 0: 每个 kL1 块各一段 (最细)。
    segments > 2: 按 kL1 块数均分成 segments 段 (块数不足时退化为块数)。
    """
    if kl1 <= 0:
        raise ValueError("kl1 must be positive")
    n_chunks = -(-k_gmm2 // kl1)
    if segments == 2:
        if n_chunks <= 1:
            return [(0, k_gmm2)]
        return [(0, kl1), (kl1, k_gmm2)]
    if segments == 0 or segments >= n_chunks:
        return [(j * kl1, min((j + 1) * kl1, k_gmm2)) for j in range(n_chunks)]
    if segments < 1:
        raise ValueError("StageLink.readiness must be >= 0")
    if segments == 1:
        return [(0, k_gmm2)]
    # 把 n_chunks 个块尽量均匀地分到 segments 段里
    per, rem = divmod(n_chunks, segments)
    out, lo_chunk = [], 0
    for j in range(segments):
        take = per + (1 if j < rem else 0)
        hi_chunk = lo_chunk + take
        out.append((lo_chunk * kl1, min(hi_chunk * kl1, k_gmm2)))
        lo_chunk = hi_chunk
    return out


def _require_full_k(acts, k_gmm2: int, expert: int, group: int, t) -> None:
    """行范围相交的 ACT 必须无缺口地覆盖 [0, k_gmm2)."""
    cursor = 0
    for a in sorted(acts, key=lambda a: a.col_begin):
        if a.col_begin > cursor:
            break
        cursor = max(cursor, a.col_end)
    if cursor < k_gmm2:
        raise ValueError(
            f"activation readiness incomplete for expert={expert}, group={group}, "
            f"rows {t.row_begin}–{t.row_end}: ACT 只覆盖 K 的 [0,{cursor}), "
            f"需要 [0,{k_gmm2})")
