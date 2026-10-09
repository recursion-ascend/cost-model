"""Outer API: routing-count tensor -> per-rank schedules -> kernel time."""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

from .costs import (AnalyticalCombineCosts, AnalyticalGmmCosts,
                    DispatchDataLayout, PrimitiveCosts)
from .model import A8W8WaveCostModel, MegaMoeShape, ModelOptions
from .config.policy import InstancePolicy
from .analysis.bounds import attach_bounds
from .config.provenance import run_provenance


def _rebind_costs_to_kernel(costs: PrimitiveCosts, kernel) -> PrimitiveCosts:
    """编译期参数以 KernelConfig 为唯一事实源, 公式自动重绑.

    l1_buf_num / l1_tile_k / combine_quant_mode 影响公式结构, 手工拼装
    Analytical* 时公式类看不到 KernelConfig. 此处在入口按 kernel 重建
    公式 (标定参数 — 带宽/速率/重启停顿 — 保留公式自身的), 三个参数
    因此在任何拼装方式下都生效.

    自定义 callable 无法内省, 原样使用 (调用方自行保证一致).
    """
    if kernel is None:
        return costs
    g1 = getattr(costs.gmm1_tile, "__self__", None)
    if isinstance(g1, AnalyticalGmmCosts):
        want_wb = 1 if kernel.gmm1_interleaved else kernel.activation_n_half
        if (g1.serial != (kernel.l1_buf_num == 1)
                or g1._k_l1 != kernel.l1_tile_k
                or g1.weight_nz != kernel.weight_nz
                or g1.wb != want_wb):
            if kernel.weight_nz and not g1.weight_nz:
                raise ValueError(
                    "KernelConfig.weight_nz=True 但 gmm1 公式按 Z 布局构造, "
                    "NZ 路径带宽无法从 Z 标定推导. 显式给: "
                    "build_analytical_costs(bw_l1_gm_b_nz=...) 或 "
                    "AnalyticalGmmCosts(weight_nz=True, bw_b_nz_bytes_per_us=...)")
            new_g = AnalyticalGmmCosts(
                bw_bytes_per_us=g1.bw,
                weight_nz=kernel.weight_nz,
                bw_b_nz_bytes_per_us=(g1.bw_b if g1.weight_nz else 0.0),
                l1_buf_num=kernel.l1_buf_num,
                cube_mac_per_us=g1.cube_rate,
                tile_restart_us=g1.chunk_restart,
                l1_tile_k=kernel.l1_tile_k,
                gmm1_weight_blocks=want_wb,
                gmm2_a_from_gm=g1.gmm2_a_from_gm,
                load_overlap=g1.load_overlap)
            costs = dataclasses.replace(costs, gmm1_tile=new_g.gmm1_tile,
                                        gmm2_tile=new_g.gmm2_tile)
    comb = getattr(costs.combine_tile, "__self__", None)
    if (isinstance(comb, AnalyticalCombineCosts)
            and (comb.combine_quant_mode != kernel.combine_quant_mode
                 or comb.meta_bytes != kernel.combine_meta_bytes_per_row)):
        new_c = AnalyticalCombineCosts(
            combine_quant_mode=kernel.combine_quant_mode,
            bw_local_bytes_per_us=comb.bw_local,
            bw_remote_bytes_per_us=comb.bw_remote,
            meta_bytes_per_row=kernel.combine_meta_bytes_per_row)
        costs = dataclasses.replace(costs, combine_tile=new_c.tile)
    return costs


def simulate_routing_counts(
    *,
    routing_counts: Sequence[Sequence[Sequence[int]]],
    token_num_per_rank: int,
    h: int,
    hidden_dim: int,
    aic_num: int,
    costs: PrimitiveCosts,
    topk: int = 8,
    shared_expert_num: int = 0,
    kernel=None,
    options: ModelOptions = ModelOptions(),
    policy: InstancePolicy = None,
    restructure=None,
    p1_override: int = 0,
    p2_override: int = 0,
    dispatch_layout: Optional[DispatchDataLayout] = None,
    wave_packing=None,
    core_assignment=None,
    scheduling_policy=None,
    tile_grid=None,
    orchestration=None,
    platform=None,
    # 下界断言: 缺省**抛异常**。穿透物理下界的数说明模型漏算了某项代价, 不该让人
    # 拿去做决策。给 False 可降级为只记录 (rank_results[i]["bounds"]["violation"]),
    # 用于排查而不是用于出结论。见 analysis/bounds.py 与
    # analysis/bounds.py 的三条下界。
    check_bounds: bool = True,
) -> Dict[str, object]:
    """Simulate directly from C[dst_rank][local_expert][src_rank].

    The model never asks the caller for segment/batch counts.  Those are derived
    internally from routing counts and source control flow.  Ranks are scheduled
    independently because AIC/AIV resources are per-rank; the operator kernel
    latency is the max-rank completion time, measured to the end of the last
    COMBINE (kernel_total_us); kernel_dag_end_us additionally covers the
    epilogue.  Cross-rank fabric contention is
    not modelled: the rate-server channel model is not enabled (its two fabric
    constants are on different scales and stacking them would double-count
    contention; see model.simulate_multi). Events still declare
    channel_bytes, which are only summed into rank_results["traffic_bytes"].

    wave_packing / core_assignment / scheduling_policy / tile_grid: 策略对象,
    对全部 rank 生效; None = 模型缺省 (SequentialGreedy / StaticRoundRobin /
    EarliestStart / SwizzledTileGrid).
    orchestration: 建图器类; None = 按 KernelConfig.topo_urma 自动选 MTE/Layered.
    """
    world = len(routing_counts)
    if world == 0:
        raise ValueError("routing_counts must contain at least one destination rank")
    local_experts = len(routing_counts[0])
    if local_experts == 0:
        raise ValueError("routing_counts must contain local experts")
    for dst, expert_rows in enumerate(routing_counts):
        if len(expert_rows) != local_experts:
            raise ValueError("all destination ranks must have the same local expert count")
        for expert, src_counts in enumerate(expert_rows):
            if len(src_counts) != world:
                raise ValueError(
                    f"routing_counts[{dst}][{expert}] must have world-size={world} source counts"
                )
            if any(int(x) < 0 for x in src_counts):
                raise ValueError("routing counts must be non-negative")
    # 逐 src 守恒 (每源 rank 发出 token_num_per_rank x topk 行) 在 model.simulate_multi
    # 里查 —— 唯一的地方, 所有入口都经过它 (原先这里是个只记了 TODO 的空白)。

    costs = _rebind_costs_to_kernel(costs, kernel)
    model = A8W8WaveCostModel(costs, options)
    shapes: List[MegaMoeShape] = []
    for dst in range(world):
        source_rows = tuple(tuple(int(x) for x in row) for row in routing_counts[dst])
        expert_tokens = tuple(sum(row) for row in source_rows)
        shapes.append(MegaMoeShape(
            expert_tokens=expert_tokens,
            token_num=token_num_per_rank,
            h=h,
            hidden_dim=hidden_dim,
            aic_num=aic_num,
            rank_id=dst,
            p1_override=p1_override,
            p2_override=p2_override,
            expert_source_tokens=source_rows,
            dispatch_layout=dispatch_layout or DispatchDataLayout.from_hidden(h),
            topk=topk,
            shared_expert_num=shared_expert_num,
            kernel=kernel,
            policy=policy if policy is not None else InstancePolicy(),
            wave_packing=wave_packing,
            core_assignment=core_assignment,
            scheduling_policy=scheduling_policy,
            tile_grid=tile_grid,
            orchestration=orchestration,
        ))
    rank_results = model.simulate_multi(shapes, restructure=restructure)
    # 下界: 只用算法 + 物理 + 硬件事实, 不含编排。墙钟低于它一定是模型漏算了代价,
    # 所以这里算完就断言 (见 analysis/bounds.py)。
    for dst in range(world):
        # platform 只有 api 这一层知道 (model 层不收), 所以这里用它**重算**一遍:
        # model.simulate_multi 已经按 platform=None 挂过, 给了 platform 才能把
        # 聚合 HBM 这条规格算进带宽下界。
        rank_results[dst]["bounds"] = attach_bounds(
            shapes[dst], rank_results[dst], costs=costs, kernel=kernel,
            platform=platform, active_cores=aic_num, check=check_bounds)
    all_ready: List[Dict[str, object]] = []
    for dst in range(world):
        all_ready.extend(rank_results[dst]["dispatch_ready_tiles"])

    slowest_rank = max(rank_results, key=lambda r: float(rank_results[r]["total_us"]))
    return {
        "provenance": run_provenance(costs, kernel),
        "kernel_total_us": float(rank_results[slowest_rank]["total_us"]),
        "kernel_dag_end_us": max(float(r["dag_end_us"]) for r in rank_results.values()),
        "slowest_rank": slowest_rank,
        "rank_results": rank_results,
        "dispatch_ready_tiles": sorted(
            all_ready, key=lambda x: (x["t_dispatchReady_us"], x["dst_rank"], x["expert"], x["mgroup"])
        ),
    }
