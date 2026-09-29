"""第 4 层: GMM2 stage — head/tail 拆分事件 + 按 ntile 的 ACT 依赖.

依赖按 meta["ntile"] 选, 不按构建序 — swizzle 蛇形遍历下奇数块的
构建序与 ntile 不对应. head 窗只等覆盖 K[0,kL1) 的 ACT (流水最大化),
主体等覆盖 K[kL1,K) 的 ACT.
"""
from __future__ import annotations

from typing import List

from ..config.hardware import _gmm2_head_tail_fractions, ceil_div, select_kl1
from ..planning.waves import swizzle_coord
from .context import BuildContext


def add_gmm2_wave(builder, ctx: BuildContext, w, shape, km, p, c, core_assign,
                  policy, TILE_M, TILE_N, ACT_HALF, call_iteration) -> None:
    gmm2_n_tiles = ceil_div(shape.h, TILE_N)
    expected_act = ceil_div(builder._gmm1_device_scheduler_n(shape, ACT_HALF), TILE_N)
    cursor = ctx.cursor
    k_gmm2 = shape.hidden_dim // ACT_HALF
    for si, sl in enumerate(w.slices):
        tile_count = sl.m_groups * gmm2_n_tiles
        # 预计算逐 tile 几何与时长 (分核策略按真实代价均衡)
        tile_info = []
        tile_costs = []
        for tile_idx in range(tile_count):
            mg, nt = swizzle_coord(tile_idx, sl.m_groups, gmm2_n_tiles,
                                   km.swizzle_offset, km.swizzle_direction)
            m_rows = builder._tile_rows(sl, mg, tile_m=km.tile_m)
            gmm2_logical_n = min(TILE_N, shape.h - nt * TILE_N)
            if c.gmm2_bw_bytes_per_us is not None:
                dur = k_gmm2 * gmm2_logical_n / c.gmm2_bw_bytes_per_us
            else:
                dur = c.gmm2_tile(m_rows, k_gmm2, gmm2_logical_n)
            tile_info.append((mg, nt, m_rows, gmm2_logical_n, dur))
            tile_costs.append(dur)
        if core_assign is not None:
            owners = core_assign.assign(tile_count, p, cursor.start,
                                        tile_costs=tile_costs)
            cursor.set(cursor.start + tile_count)
        else:
            owners = cursor.owners(tile_count)
        first_owned = [True] * p

        for tile_idx, core in enumerate(owners):
            mg, nt, m_rows, gmm2_logical_n, duration = tile_info[tile_idx]
            global_group = sl.row_begin // TILE_M + mg
            ready = ctx.activation_ready.get((sl.expert, global_group), [])
            if len(ready) != expected_act:
                raise ValueError(
                    f"activation readiness incomplete for expert={sl.expert}, "
                    f"group={global_group}: got {len(ready)}, expected {expected_act}")

            deps: List[str] = []
            q_aic2 = (f"Q:aic:c{core}", 1)
            combine_history = ctx.gmm2_combine_history[core]
            credit = policy.gmm2_combine_credit
            if credit is not None and len(combine_history) >= credit:
                deps.append(combine_history[-credit])

            kl1 = select_kl1(sl.rows, k_gmm2, builder.options.gmm2_kl1,
                             tile_m=km.tile_m, tile_n=km.tile_n,
                             l1_size=km.l1_size, k_l1_base=km.l1_tile_k,
                             n_windows=km.l1_buf_num)
            head_frac, tail_frac = _gmm2_head_tail_fractions(k_gmm2, kl1)
            if first_owned[core]:
                duration += c.gmm2_problem_startup_us
                first_owned[core] = False

            act_by_ntile = dict(ready)
            head_acts = [act_by_ntile[n] for n in sorted(act_by_ntile)
                         if n * TILE_N < kl1]
            tail_acts = [act_by_ntile[n] for n in sorted(act_by_ntile)
                         if n * TILE_N >= kl1]

            gname = f"W{w.index}.E{sl.expert}.S{si}.gmm2.m{mg}.n{nt}.c{core}"
            builder._event(gname + ".h", (f"AIC:{core}",), duration * head_frac,
                           deps=deps + head_acts,
                           acquires=(q_aic2,), releases=(q_aic2,),
                           meta={"stage": "gmm2", "wave": w.index,
                                 "call_iteration": call_iteration, "expert": sl.expert,
                                 "slice": si, "mgroup": global_group, "ntile": nt,
                                 "core": core, "m_rows": m_rows,
                                 "cursor_tile": tile_idx, "part": "head"})
            builder._event(gname, (f"AIC:{core}",), duration * tail_frac,
                           deps=[gname + ".h"] + tail_acts,
                           meta={"stage": "gmm2", "wave": w.index,
                                 "call_iteration": call_iteration, "expert": sl.expert,
                                 "slice": si, "mgroup": global_group, "ntile": nt,
                                 "core": core, "m_rows": m_rows,
                                 "cursor_tile": tile_idx, "part": "tail"})
            builder.gmm2_tail_by_group.setdefault((sl.expert, global_group), []).append(gname)

            # combine 走传输后端钩子: MTE 配对 tile / URMA 记录待批
            builder.combine_backend.on_gmm2_tile(
                builder, ctx, w, si, sl, mg, nt, core, m_rows,
                gmm2_logical_n, gname, global_group, call_iteration)
