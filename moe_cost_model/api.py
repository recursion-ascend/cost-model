"""Outer API: routing-count tensor -> per-rank schedules -> kernel time."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .dispatch import DispatchDataLayout
from .model import A8W8WaveCostModel, MegaMoeShape, ModelOptions
from .primitives import PrimitiveCosts
from .provenance import collect_provenance, provenance_report


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
    p1_override: int = 0,
    p2_override: int = 0,
    dispatch_layout: Optional[DispatchDataLayout] = None,
) -> Dict[str, object]:
    """Simulate directly from C[dst_rank][local_expert][src_rank].

    The model never asks the caller for segment/batch counts.  Those are derived
    internally from routing counts and source control flow.  Ranks are scheduled
    independently because AIC/AIV resources are per-rank; the operator kernel
    latency is the max-rank completion time.  Cross-rank fabric contention is not
    invented here and must be added as a calibrated shared-resource model if data
    shows it is prediction-relevant.
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

    model = A8W8WaveCostModel(costs, options)
    rank_results: Dict[int, Dict[str, object]] = {}
    all_ready: List[Dict[str, object]] = []
    for dst in range(world):
        source_rows = tuple(tuple(int(x) for x in row) for row in routing_counts[dst])
        expert_tokens = tuple(sum(row) for row in source_rows)
        shape = MegaMoeShape(
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
        )
        result = model.simulate(shape)
        rank_results[dst] = result
        all_ready.extend(result["dispatch_ready_tiles"])

    slowest_rank = max(rank_results, key=lambda r: float(rank_results[r]["total_us"]))
    import moe_cost_model.constants as _const
    prov = collect_provenance(vars(_const))
    prov.update({f"costs.{k}": (float(v), 'user-supplied')
                 for k, v in vars(costs).items() if isinstance(v, (int, float))})
    prov.update({f"kernel.{k}": (float(v), 'user-supplied')
                 for k, v in (vars(kernel) if kernel else {}).items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return {
        "provenance": provenance_report(prov),
        "kernel_total_us": float(rank_results[slowest_rank]["total_us"]),
        "slowest_rank": slowest_rank,
        "rank_results": rank_results,
        "dispatch_ready_tiles": sorted(
            all_ready, key=lambda x: (x["t_dispatchReady_us"], x["dst_rank"], x["expert"], x["mgroup"])
        ),
    }

