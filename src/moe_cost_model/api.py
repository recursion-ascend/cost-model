"""Outer API: routing-count tensor -> per-rank schedules -> kernel time."""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

from .costs import (AnalyticalCombineCosts, AnalyticalGmmCosts,
                    DispatchDataLayout, PrimitiveCosts)
from .model import A8W8WaveCostModel, MegaMoeShape, ModelOptions
from .config.policy import InstancePolicy
from .config.provenance import collect_provenance, provenance_report


def _rebind_costs_to_kernel(costs: PrimitiveCosts, kernel) -> PrimitiveCosts:
    """编译期旋钮以 KernelConfig 为唯一事实源, 公式自动重绑.

    l1_buf_num / l1_tile_k / combine_quant_mode 影响公式结构, 手工拼装
    Analytical* 时公式类看不到 KernelConfig. 此处在入口按 kernel 重建
    公式 (标定参数 — 带宽/速率/重启停顿 — 保留公式自身的), 三个旋钮
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
                gmm2_a_from_gm=g1.gmm2_a_from_gm)
            costs = dataclasses.replace(costs, gmm1_tile=new_g.gmm1_tile,
                                        gmm2_tile=new_g.gmm2_tile)
    comb = getattr(costs.combine_tile, "__self__", None)
    if (isinstance(comb, AnalyticalCombineCosts)
            and comb.combine_quant_mode != kernel.combine_quant_mode):
        new_c = AnalyticalCombineCosts(
            combine_quant_mode=kernel.combine_quant_mode,
            bw_local_bytes_per_us=comb.bw_local,
            bw_remote_bytes_per_us=comb.bw_remote,
            meta_bytes_per_row=comb.meta_bytes)
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
) -> Dict[str, object]:
    """Simulate directly from C[dst_rank][local_expert][src_rank].

    The model never asks the caller for segment/batch counts.  Those are derived
    internally from routing counts and source control flow.  Ranks are scheduled
    independently because AIC/AIV resources are per-rank; the operator kernel
    latency is the max-rank completion time, measured to the end of the last
    COMBINE (kernel_total_us); kernel_dag_end_us additionally covers the
    epilogue.  Cross-rank fabric contention is
    not invented here: the rate-server mechanism exists but is a placeholder
    (ModelOptions.fabric_channels, default off) until an ablation benchmark
    justifies enabling it (see README 信道占位状态).

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
    # TODO(输入校验缺失): 逐 src 总量守恒未校验 —
    #   Σ_{dst,e} routing_counts[dst][e][src] == token_num_per_rank * topk
    # C 与 token_num_per_rank 不自洽时静默产生分裂预测: MoE 主 stage 按 C 推导的
    # M_e 计, 而 GMM2 lag 阈值 (model.py gmm2_lag_threshold 判断) / 共享专家规模
    # (m_tot_s) / 尾段 unpermute 字节却按 token_num_per_rank 计, 无任何报错.

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
    all_ready: List[Dict[str, object]] = []
    for dst in range(world):
        all_ready.extend(rank_results[dst]["dispatch_ready_tiles"])

    slowest_rank = max(rank_results, key=lambda r: float(rank_results[r]["total_us"]))
    from .config import hardware as _hw, policy as _pol
    prov = collect_provenance(vars(_hw))
    prov.update(collect_provenance(vars(_pol)))
    prov.update({f"costs.{k}": (float(v), 'user-supplied')
                 for k, v in vars(costs).items() if isinstance(v, (int, float))})
    prov.update({f"kernel.{k}": (float(v), 'user-supplied')
                 for k, v in (vars(kernel) if kernel else {}).items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return {
        "provenance": provenance_report(prov),
        "kernel_total_us": float(rank_results[slowest_rank]["total_us"]),
        "kernel_dag_end_us": max(float(r["dag_end_us"]) for r in rank_results.values()),
        "slowest_rank": slowest_rank,
        "rank_results": rank_results,
        "dispatch_ready_tiles": sorted(
            all_ready, key=lambda x: (x["t_dispatchReady_us"], x["dst_rank"], x["expert"], x["mgroup"])
        ),
    }
