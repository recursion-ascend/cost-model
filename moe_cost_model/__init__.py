"""
moe_cost_model  

Dispatch-ready-centered structural + calibrated cost model for the current mainline MegaMoE path:
  hicann/ops-transformer/mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h
  Ascend 950 / DAV_3510, MTE Wave, A8W8, TopkWeightsPrefetch=false,
  COMBINE_NO_QUANT (paired AIV1 tile combine).

What is source-faithful in this version
---------------------------------------
1. Host mGroupsPerWave formula from mega_moe_tiling.cpp.
2. Dynamic 256-row M-group wave packing across expert token counts.
3. Rotating shared GMM startBlockIdx cursor, advanced in the real call order:
      GMM1(current wave) -> GMM2(current wave or previous wave when lagged).
   The fixed-role-resonance cursor correction in mega_moe_wave_a8w8.h is included.
4. AIV1 Dispatch lookahead:
      first outer iteration prepares W0 and W1; later iterations prepare W(i+1).
5. Dispatch ownership uses GetRotatedBalancedWorkRange semantics.  Readiness is
   tracked per expert / 256-row group and is released after every contributing
   AIV1 core has finished its expert-local range, matching PublishGmm1TileReady.
   Optional per-expert/per-source-rank counts can be supplied to expose the exact
   number of source-rank segments touched by a dispatch range.
6. GMM1 -> Activation uses the non-prefetch A8W8 UB double buffer.  AIC tile i
   cannot reuse its UB slot until paired AIV0 has completed tile i-2.
7. Activation -> GMM2 uses the Wave path's IsWaveFlagGrained=true protocol:
   every GMM2 M-group waits for all GMM1/Activation N-tiles of that same 256-row
   group, not for the entire expert's wave-slice matmul.
8. GMM2 -> no-quant Combine uses paired AIV1. Fixed producer credit is an optional
   counterfactual knob; DAV_3510 validation does not support a fixed credit=15 edge.
   AIC tile i is allowed to lead Combine by at most 15 tiles, so tile i>=15
   depends on the Combine ACK for tile i-15.
9. tokenNum >= 4096 uses the source one-wave GMM2 lag.  This is represented by
   the actual per-core program order rather than by an artificial global barrier.
10. AIC/AIV0/AIV1 program order is explicit in the DAG, so a dependency-aware
    scheduler cannot reorder calls in ways the kernel cannot execute.

Intentional abstractions
------------------------
* BlockScheduler tile-coordinate order is represented as M-group-major then
  N-tile-major.  Tile counts, startBlockIdx rotation and per-core ownership are
  preserved.  Validate coordinate order with kernel trace/msprof if it becomes
  prediction-sensitive.
* Dispatch supports two calibration paths: v0 OLS (segments/rows), and a v1
  source-derived IR that explicitly counts source-rank segments, route batches,
  requested bytes, per-row fetch/store operations, and ready-counter atomics.
  Timing coefficients remain empirical; no hardware constants are invented.
* Absolute microseconds are never invented.  Default primitive coefficients are
  zero and must be calibrated from profiler/microbenchmark data.
"""

from .constants import *  # noqa: F401,F403
from .constants import KernelConfig, InstancePolicy
from .model import EngineQueueDepths
from .provenance import SourcedInt, SourcedValue, collect_provenance, provenance_report
from .dag import Channel, Event, MultiResourceScheduler, ScheduledEvent
from .dispatch import (
    DispatchCallIR, DispatchDataLayout, DispatchMechanisticLatency,
    build_dispatch_expert_ir,
)
from .primitives import (
    AnalyticalActCosts, AnalyticalCombineCosts, AnalyticalGmmCosts,
    PrimitiveCosts, build_analytical_costs,
)
from .waves import (
    ExpertSlice, Position, Wave, calc_m_groups_per_wave, plan_waves,
    resolve_gmm1_min_logical_tiles_per_core,
)
from .model import (
    A8W8WaveCostModel, BlockCursor, CursorTrace, MegaMoeShape, ModelOptions,
)
from .pipeline import (
    BufferSlots, PhaseRates, PipelineConstraints, QueueDepths, SyncLatency,
    default_channels, parse_tiling,
)
from .api import simulate_routing_counts
from .policies import idle_core_stealing

__all__ = [n for n in dir() if not n.startswith('_')]
