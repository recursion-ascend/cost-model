"""第 4 层: GMM1 stage — GMM1 tile 事件 + 游标分核.

tile 网格由 shape.tile_grid 给出 (缺省 SwizzledTileGrid = kernel 现行为)。
依赖: 组就绪标记 (dispatch 产出) + 同核第 i-depth 个 ACT (UB 缓冲).
流水填充经 PrimitiveCosts.gmm1_fill_us 按 tile 均摊.
"""
from __future__ import annotations

from typing import List

from ..config.hardware import BW_L1_GM
from ..costs import gmm1_phase_split
from ..planning.tile_grid import STAGE_GMM1, validate_tiles
from .activation import add_activation_tile
from .context import BuildContext
from .pipeline_expand import CH_GM_TO_L1
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
        # B 复用: 切片内首个 m-group 的 tile 付整份 B 流, 其余各付 gmm1_b_reuse_frac。
        # 未开复用 (frac == 1.0) 时按三参调用 —— 自定义 gmm1_tile callable 只需接三个参数。
        frac = float(km.gmm1_b_reuse_frac)
        reuse = frac != 1.0
        b_load = [1.0 if (not reuse or t.row_begin < TILE_M) else frac for t in tiles]
        if reuse:
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
            # UB 槽位 (C2): GMM1 的结果经 L0C->UB 的 Fixpipe 直给配对 AIV0 (不落 GM),
            # UB 里能同时存 depth 块。物理上这是**容量**, 不是程序序 —— 所以用计数
            # 信号量表达: GMM1 开始时占一个槽, 它配对的 ACT 读完后归还。
            #
            # 原先写成"距离依赖" (gmm1[i] deps act[i-depth]), 那等于把 kernel 在这个核
            # 上的发射顺序钉进了图里: 换一种顺序就得换一条边。改成容量之后调度器可以
            # 自由换序, 约束照样成立。
            #
            # 不会死锁: 归还者 (ACT) 自己不占 UB 槽, 且只依赖它的 GMM1 —— 而 GMM1 持槽
            # 时已经跑完了, 所以持槽者的归还者永远最终可运行, 不可能成环。
            # (当初改成距离依赖的理由是"信号量会与 act 的程序序依赖成环", 那个前提
            #  已不存在: 现在 ACT 之间没有任何程序序边, 同核 ACT 的先后由资源互斥定。)
            # depth=0 = 不建这个约束 (假设 UB 不构成瓶颈)。
            ub_slot = (f"UB:gmm1act:c{core}", 1)
            depth = builder.options.link("gmm1", "activation").depth
            acq = (q_aic, ub_slot) if depth > 0 else (q_aic,)

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
            # C1: 事件名**不带核号** —— 事件的身份是"哪一份工作", 核是调度的产出。
            # 名字带核号的后果是依赖边也带: 换一种分核方式, 图的结构就跟着变, 那是在
            # 复现 kernel 的记账而不是建模。(URMA 路径一直就是这样命名的。)
            gname = f"W{w.index}.E{sl.expert}.S{si}.gmm1.{label}"
            # GM→L1 访存量: A 流 m·K (dispatch 落 GM 的激活) + B 流 wb·K·cols。
            # max(A,B) 口径只让较慢的一股决定**时长**, 但两股字节都真实发生,
            # 所以这里按字节申报 (统计用, 不参与准入)。
            a_bytes = t.rows * shape.h
            b_bytes = _gmm1_b_bytes(c, shape.h, t.cols) * b_load[tile_idx]
            builder._event(gname, (f"AIC:{core}",), duration, deps=deps,
                           acquires=acq, releases=(q_aic,), meta=meta,
                           channel_bytes=((CH_GM_TO_L1, float(a_bytes + b_bytes),
                                           float(BW_L1_GM)),))

            add_activation_tile(builder, ctx, w, si, sl, t, label, ntile, core,
                                global_group, gname, out_div,
                                ub_slot if depth > 0 else None)


def _gmm1_b_bytes(costs, k: int, cols: int) -> int:
    """一个 GMM1 tile 的 B 流字节 = wb·K·cols (wb = 非交织 2 / 交织 1).

    wb 只有解析公式对象知道; 自定义 callable 按 2 算 (与缺省 KernelConfig 一致)。
    """
    owner = getattr(costs.gmm1_tile, "__self__", None)
    wb = int(getattr(owner, "wb", 2))
    return wb * k * cols
