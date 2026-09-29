"""第 4 层: GMM1 stage — GMM1 tile 事件 + 游标分核.

依赖: 组就绪标记 (dispatch 产出) + 同核第 i-depth 个 ACT (UB 缓冲).
流水填充经 PrimitiveCosts.gmm1_fill_us 按 tile 均摊.
"""
from __future__ import annotations

from typing import List

from ..config.hardware import ceil_div
from ..planning.waves import swizzle_coord
from .activation import add_activation_tile
from .context import BuildContext


def add_gmm1_wave(builder, ctx: BuildContext, w, shape, km, p, c, core_assign,
                  policy, TILE_M, TILE_N, ACT_HALF) -> None:
    gmm1_sched_n = builder._gmm1_device_scheduler_n(shape, km.activation_n_half)
    gmm1_n_tiles = ceil_div(gmm1_sched_n, TILE_N)
    cursor = ctx.cursor
    for si, sl in enumerate(w.slices):
        tile_count = sl.m_groups * gmm1_n_tiles
        fill_share = c.gmm1_fill_us / tile_count if tile_count else 0.0
        # 预计算逐 tile 几何与时长 (分核策略按真实代价均衡, 不按个数)
        tile_info = []
        tile_costs = []
        for tile_idx in range(tile_count):
            mg, nt = swizzle_coord(tile_idx, sl.m_groups, gmm1_n_tiles,
                                   km.swizzle_offset, km.swizzle_direction)
            m_rows = builder._tile_rows(sl, mg, tile_m=km.tile_m)
            logical_n = min(TILE_N, gmm1_sched_n - nt * TILE_N)
            dur = c.gmm1_tile(m_rows, shape.h, logical_n)
            dur += fill_share
            tile_info.append((mg, nt, m_rows, logical_n, dur))
            tile_costs.append(dur)
        if core_assign is not None:
            owners = core_assign.assign(tile_count, p, cursor.start,
                                        tile_costs=tile_costs)
            cursor.set(cursor.start + tile_count)
        else:
            owners = cursor.owners(tile_count)
        first_owned = [True] * p

        for tile_idx, core in enumerate(owners):
            mg, nt, m_rows, logical_n, duration = tile_info[tile_idx]
            global_group = sl.row_begin // TILE_M + mg

            deps: List[str] = []
            q_aic = (f"Q:aic:c{core}", 1)
            ready_name = ctx.dispatch_ready_event.get((sl.expert, global_group))
            if ready_name is None:
                raise ValueError(
                    f"missing t_dispatchReady for expert={sl.expert}, group={global_group}")
            deps.append(ready_name)
            history = ctx.gmm1_act_history[core]
            depth = policy.gmm1_activation_depth
            if len(history) >= depth:
                deps.append(history[-depth])

            if first_owned[core]:
                duration += c.gmm1_problem_startup_us
                first_owned[core] = False

            gname = f"W{w.index}.E{sl.expert}.S{si}.gmm1.m{mg}.n{nt}.c{core}"
            builder._event(gname, (f"AIC:{core}",), duration, deps=deps,
                           acquires=(q_aic,), releases=(q_aic,),
                           meta={"stage": "gmm1", "wave": w.index, "expert": sl.expert,
                                 "slice": si, "mgroup": global_group, "ntile": nt,
                                 "logical_n": logical_n, "core": core, "m_rows": m_rows,
                                 "cursor_tile": tile_idx,
                                 "dispatch_ready_event": ready_name})

            add_activation_tile(builder, ctx, w, si, sl, mg, nt, core,
                                m_rows, logical_n, global_group, gname)
