"""第 4 层: GMM1 stage — GMM1 tile 事件 + 游标分核.

tile 网格由 shape.tile_grid 给出 (缺省 SwizzledTileGrid = kernel 现行为)。
依赖: 组就绪标记 (dispatch 产出) + 同核第 i-depth 个 ACT (UB 缓冲).
流水填充经 PrimitiveCosts.gmm1_fill_us 按 tile 均摊.
"""
from __future__ import annotations

from typing import List

from ..costs import gmm1_phase_split
from ..planning.tile_grid import STAGE_GMM1, validate_tiles
from .activation import add_activation_tile
from .context import BuildContext
from .tiling import resolve_grid, tile_label


def add_gmm1_wave(builder, ctx: BuildContext, w, shape, km, p, c, core_assign,
                  policy, TILE_M, TILE_N, ACT_HALF) -> None:
    # 交织路径的调度宽度是整个 hidden_dim (gate/up 在 tile 内按列交织), 非交织是
    # hidden_dim/activation_n_half —— 于是 n-tile 数翻倍, 每 tile 的 B 流减半。
    # 出处: mega_moe_wave_a8w8.h:446。
    act_half = 1 if km.gmm1_interleaved else km.activation_n_half
    gmm1_sched_n = builder._gmm1_device_scheduler_n(shape, act_half)
    # 一个 GMM1 tile 产出多少输出列: 交织时 epilogueN = tileN/activation_n_half。
    out_div = km.activation_n_half if km.gmm1_interleaved else 1
    grid = resolve_grid(shape)
    cursor = ctx.cursor
    for si, sl in enumerate(w.slices):
        tiles = grid.plan(stage=STAGE_GMM1, rows=sl.rows, cols=gmm1_sched_n, kernel=km)
        validate_tiles(tiles, rows=sl.rows, cols=gmm1_sched_n, tile_m=TILE_M,
                       where=f"GMM1 专家 {sl.expert}")
        tile_count = len(tiles)
        fill_share = c.gmm1_fill_us / tile_count if tile_count else 0.0
        # 预计算逐 tile 时长 (分核策略按真实代价均衡, 不按个数)
        # B 复用: 切片内只有首个 m-group 付自己列块的 B 流。
        # 未开复用时按三参调用 —— 自定义 gmm1_tile callable 只需接三个参数。
        b_load = [not km.gmm1_b_reuse or t.row_begin < TILE_M for t in tiles]
        if km.gmm1_b_reuse:
            tile_costs = [c.gmm1_tile(t.rows, shape.h, t.cols, b_load[i]) + fill_share
                          for i, t in enumerate(tiles)]
        else:
            tile_costs = [c.gmm1_tile(t.rows, shape.h, t.cols) + fill_share for t in tiles]
        if core_assign is not None:
            owners = core_assign.assign(tile_count, p, cursor.start,
                                        tile_costs=tile_costs)
            cursor.set(cursor.start + tile_count)
        else:
            owners = cursor.owners(tile_count)
        first_owned = [True] * p

        for tile_idx, core in enumerate(owners):
            t = tiles[tile_idx]
            duration = tile_costs[tile_idx]
            mg = t.row_begin // TILE_M
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
            # depth = UB 里能同时存几块 GMM1 结果。GMM1 的结果经 L0C->UB 的 Fixpipe
            # 直给配对 AIV0 (不落 GM), 所以 AIV0 没把第 i-depth 块读走, AIC 就没地方
            # 写第 i 块。depth=0 = 不建这条边 (假设 UB 不构成约束)。
            # 注意 history[-0] 是 history[0] 而不是"无依赖", 所以 0 必须单独判。
            if depth > 0 and len(history) >= depth:
                deps.append(history[-depth])

            if first_owned[core]:
                duration += c.gmm1_problem_startup_us
                first_owned[core] = False

            ntile = t.col_begin // TILE_N
            meta = {"stage": "gmm1", "wave": w.index, "expert": sl.expert,
                    "slice": si, "mgroup": global_group, "ntile": ntile,
                    "col_begin": t.col_begin, "col_end": t.col_end,
                    "row_begin": t.row_begin, "row_end": t.row_end,
                    "logical_n": t.cols, "core": core, "m_rows": t.rows,
                    "cursor_tile": tile_idx,
                    "dispatch_ready_event": ready_name}
            phases = gmm1_phase_split(c, t.rows, shape.h, t.cols, b_load[tile_idx])
            if phases is not None:
                # 相位流水按这组数拆 load/cube 相位并折算 GM→L1 信道字节
                meta["load_us"], meta["compute_us"] = phases
            label = tile_label(t, sl.rows, gmm1_sched_n, TILE_M, TILE_N)
            gname = f"W{w.index}.E{sl.expert}.S{si}.gmm1.{label}.c{core}"
            builder._event(gname, (f"AIC:{core}",), duration, deps=deps,
                           acquires=(q_aic,), releases=(q_aic,), meta=meta)

            add_activation_tile(builder, ctx, w, si, sl, t, label, ntile, core,
                                global_group, gname, out_div)
