"""第 4 层: GMM2 stage — head/tail 拆分事件 + 按行列范围的 ACT 依赖.

tile 网格由 shape.tile_grid 给出 (缺省 SwizzledTileGrid = kernel 现行为)。
依赖按 ACT 的行列范围挑, 不按构建序 — swizzle 蛇形遍历下构建序与坐标不对应。
head 窗只等覆盖 K[0,kL1) 的 ACT (流水最大化), 主体等覆盖 K[kL1,K) 的.
"""
from __future__ import annotations

from typing import List

from ..config.hardware import _gmm2_head_tail_fractions, select_kl1
from ..costs import gmm2_phase_split
from ..planning.tile_grid import STAGE_GMM2, validate_tiles
from .context import BuildContext
from .tiling import resolve_grid, tile_label


def add_gmm2_wave(builder, ctx: BuildContext, w, shape, km, p, c, core_assign,
                  policy, TILE_M, TILE_N, ACT_HALF, call_iteration) -> None:
    grid = resolve_grid(shape)
    cursor = ctx.cursor
    k_gmm2 = shape.hidden_dim // ACT_HALF
    for si, sl in enumerate(w.slices):
        tiles = grid.plan(stage=STAGE_GMM2, rows=sl.rows, cols=shape.h, kernel=km)
        validate_tiles(tiles, rows=sl.rows, cols=shape.h, tile_m=TILE_M,
                       where=f"GMM2 专家 {sl.expert}")
        tile_count = len(tiles)
        tile_costs = [c.gmm2_tile(t.rows, k_gmm2, t.cols) for t in tiles]
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
        head_frac, tail_frac = _gmm2_head_tail_fractions(k_gmm2, kl1)

        for tile_idx, core in enumerate(owners):
            t = tiles[tile_idx]
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
            head_acts = [a.name for a in ordered if a.col_begin < kl1]
            tail_acts = [a.name for a in ordered if a.col_end > kl1]

            ntile = t.col_begin // TILE_N
            label = tile_label(t, sl.rows, shape.h, TILE_M, TILE_N)
            meta = {"stage": "gmm2", "wave": w.index,
                    "call_iteration": call_iteration, "expert": sl.expert,
                    "slice": si, "mgroup": global_group, "ntile": ntile,
                    "col_begin": t.col_begin, "col_end": t.col_end,
                    "row_begin": t.row_begin, "row_end": t.row_end,
                    "logical_n": t.cols, "core": core, "m_rows": t.rows,
                    "cursor_tile": tile_idx}
            # head/tail 是同一个 tile 的两段: 各按时长占比分摊本 tile 的 B 流,
            # 信道字节才不会在计算绑定时被整段时长放大。
            phases = gmm2_phase_split(c, t.rows, k_gmm2, t.cols)
            head_meta = dict(meta, part="head")
            tail_meta = dict(meta, part="tail")
            if phases is not None:
                load_us, compute_us = phases
                head_meta["load_us"] = load_us * head_frac
                head_meta["compute_us"] = compute_us * head_frac
                tail_meta["load_us"] = load_us * tail_frac
                tail_meta["compute_us"] = compute_us * tail_frac
            gname = f"W{w.index}.E{sl.expert}.S{si}.gmm2.{label}.c{core}"
            builder._event(gname + ".h", (f"AIC:{core}",), duration * head_frac,
                           deps=deps + head_acts,
                           acquires=(q_aic2,), releases=(q_aic2,), meta=head_meta)
            builder._event(gname, (f"AIC:{core}",), duration * tail_frac,
                           deps=[gname + ".h"] + tail_acts, meta=tail_meta)
            builder.gmm2_tail_by_group.setdefault((sl.expert, global_group), []).append(gname)

            # combine 走传输后端钩子: MTE 配对 tile / URMA 记录待批
            builder.combine_backend.on_gmm2_tile(
                builder, ctx, w, shape, si, sl, t, label, ntile, core,
                gname, global_group, call_iteration)


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
