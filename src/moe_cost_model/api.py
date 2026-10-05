"""Outer API: routing-count tensor -> per-rank schedules -> kernel time."""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

from .costs import (AnalyticalCombineCosts, AnalyticalGmmCosts,
                    DispatchDataLayout, PrimitiveCosts)
from .model import A8W8WaveCostModel, MegaMoeShape, ModelOptions
from .config.policy import InstancePolicy
from .analysis.bounds import (Bounds, BoundViolation, bandwidth_bound_us,
                              check_wall_clock,
                              compute_bound_us, dependency_bound_us,
                              idle_split_us, workload_facts)
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
    # docs/design_space_gaps.md 的"下界与漏账"。
    check_bounds: bool = True,
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
    # 下界: 只用算法 + 物理 + 硬件事实, 不含编排。墙钟低于它一定是模型漏算了代价,
    # 所以这里算完就断言 (见 analysis/bounds.py)。
    for dst in range(world):
        rank_results[dst]["bounds"] = _attach_bounds(
            shapes[dst], rank_results[dst], costs=costs, kernel=kernel,
            platform=platform, active_cores=aic_num, check=check_bounds)
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


def _attach_bounds(shape, rank_result, *, costs, kernel, platform, active_cores,
                   check: bool = True):
    """给一个 rank 算三个下界并断言墙钟没穿透它们.

    算法事实 (乘加数、必搬字节) 只由形状决定; 速率取硬件规格。所以同一形状换任何
    编排, 下界都不变 —— 它是用来**检查**编排结果的尺子, 不是预测。
    """
    link = None
    opts = getattr(shape, "options", None)
    if opts is not None:
        try:
            link = opts.link("activation", "gmm2")
        except Exception:
            link = None
    facts = workload_facts(
        shape, kernel=kernel,
        gmm2_a_from_gm=True if link is None else bool(link.materialised))
    cube = _cube_rate_of(costs)
    bw = _load_bw_of(costs)
    agg = getattr(platform, "hbm_bytes_per_us", None) if platform is not None else None
    bw_us, rate, who = bandwidth_bound_us(
        facts, bw_per_core_bytes_per_us=bw, active_cores=active_cores,
        aggregate_bytes_per_us=agg)
    b = Bounds(
        compute_us=compute_bound_us(facts, cube_mac_per_us=cube,
                                    active_cores=active_cores),
        bandwidth_us=bw_us,
        dependency_us=_dependency_bound_of(rank_result),
        facts=facts, bandwidth_rate=rate, bandwidth_limited_by=who)
    out = b.as_dict()
    out["violation"] = None
    if check:
        try:
            check_wall_clock(b, float(rank_result["total_us"]),
                             where=f"rank{getattr(shape, 'rank_id', '?')} 墙钟")
        except BoundViolation as exc:
            # 记进结果再抛: 调用方给 check_bounds=False 时能拿到同一条诊断
            out["violation"] = str(exc)
            raise
    else:
        low = b.lower_us
        wall = float(rank_result["total_us"])
        if low > 0 and wall < low * (1.0 - 1e-6):
            out["violation"] = (
                f"墙钟 {wall:.3f}us 低于 {b.binding} 下界 {low:.3f}us "
                f"({(low - wall) / low * 100:.1f}%)")
    out["idle_split_us"] = idle_split_us(
        (rank_result.get("idle_decomposition") or {}).get(
            f"R{getattr(shape, 'rank_id', 0)}.AIC"), active_cores)
    return out


def _cube_rate_of(costs) -> float:
    """从代价对象上取每核 Cube 速率 (MAC/us); 自定义 callable 取不到则返回 0."""
    owner = getattr(getattr(costs, "gmm1_tile", None), "__self__", None)
    return float(getattr(owner, "cube_rate", 0.0) or 0.0)


def _load_bw_of(costs) -> float:
    """从代价对象上取每核 GM->L1 载入带宽 (B/us)."""
    owner = getattr(getattr(costs, "gmm1_tile", None), "__self__", None)
    return float(getattr(owner, "bw", 0.0) or 0.0)


def _dependency_bound_of(rank_result) -> float:
    """一个 token 必经链 dispatch->GMM1->ACT->GMM2->COMBINE 上各 stage 的最小一份.

    从已排出的事件里取每个 stage 的最短事件时长 —— 那就是"这个 stage 最小一份工作"
    的时长。链上的事不能并行, 所以它们的和是硬下界 (弱, 但不会错)。
    """
    CHAIN = ("dispatch", "gmm1", "activation", "gmm2", "combine")
    best = {}
    for e in rank_result.get("events", ()):
        st = (e.meta or {}).get("stage")
        if st in CHAIN:
            d = e.end_us - e.start_us
            if d > 0 and (st not in best or d < best[st]):
                best[st] = d
    return dependency_bound_us(best)
