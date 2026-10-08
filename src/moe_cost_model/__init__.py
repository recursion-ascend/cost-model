"""moe_cost_model: 昇腾 NPU 算子 cost model —— 用硬件执行过程的模拟得到时间.

核心建模的是机制: 计算、数据搬运、存储层级与片上容量、调度、资源竞争。
事件图与调度器不认识任何算子或 stage 名 —— 一份具体实现 (有哪些 stage、
哪些边、落哪个执行角色) 由 config.stages.StageVocabulary 声明, 由
implementations/ 下的适配器降解成事件图。仓内的 MegaMoE 两份 kernel 级适配器
是其中一种实现路径 (可与 profiler trace 逐 stage 对账), 不是框架的缺省假设。

输入 → 波规划 → 事件图 → 多资源调度 → 执行时间.
分层: config(0) / shape+costs(1) / scheduler(2) / planning(3) /
builders(4) / model(5) / analysis(6).
"""
from .config.hardware import (
    ACT_BYTES_PER_VEC, ACTIVATION_N_HALF, BW_L1_GM, BW_LOCAL_GM, BW_REMOTE_GM,
    BW_REMOTE_WRITE, BW_SCATTER, BW_UB, BW_UNPERMUTE_AGG, BW_WINDOW,
    DAV3510_NONINTERLEAVED_GMM1_ACTIVATION_DEPTH,
    EpilogueOverheads,
    GMM1_LARGE_BATCH_TOKEN_THRESHOLD, GMM1_MIN_LOGICAL_TILES_PER_CORE,
    GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE, GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL,
    GMM1_SMALL_BATCH_TOKEN_THRESHOLD, GMM2_LAG_MIN_TOKEN_NUM,
    GMM2_MIN_LOGICAL_TILES_PER_CORE, KernelConfig, L1_TILE_K,
    LAYERED_BALANCED_WAVE_COUNT, LAYERED_FEW_EXPERT_THRESHOLD,
    LAYERED_FIRST_WAVE_ROWS, LAYERED_LATENCY_ROWS_PER_EXPERT,
    LAYERED_LATENCY_WAVE_COUNT, LAYERED_META_BYTES_PER_ROW,
    LAYERED_THROUGHPUT_ROWS_PER_EXPERT, LAYERED_THROUGHPUT_WAVE_COUNT,
    MXFP_DIVISOR_SIZE, MXFP_MULTI_BASE_SIZE, MXFP_MULTI_BASE_SIZE_K,
    SCALE_TRANSFER_BYTES, T_CALL_OH, T_CORE_SYNC_BARRIER_US, T_COUNT_GATE,
    T_COUNTS_EXPORT_US, T_FINALIZE_US, T_FILL_GMM1, T_GMM1_OVERLAP, T_INIT_US,
    T_INPUT_QUANT_FIXED_US, T_INPUT_QUANT_PER_TOKEN_US, T_LAT_LOCAL,
    T_LAT_REMOTE, T_OUTPUT_INIT_US, T_RANK_SYNC_RTT_US, T_STARTUP_VEC,
    TILE_M, TILE_N, TOTAL_L0C_SIZE, TOTAL_L1_SIZE, TOTAL_UB_SIZE,
    URMA_FLAG_BYTES, URMA_FLAG_WINDOW_TOKENS, URMA_GET_BW_SINGLE,
    URMA_GET_LAT_US, URMA_PUT_BW_SINGLE, URMA_PUT_LAT_US, VEC_ELEM_FP32,
    VEC_REG_WIDTH, _gmm2_head_tail_fractions, ceil_div, select_kl1,
)
from .config.granularity import (DEFAULT_GRANULARITIES, GranularityAssignment,
                                 StageGranularity)
from .analysis.bounds import (BoundViolation, Bounds, WorkloadFacts,
                              bandwidth_bound_us, check_wall_clock,
                              compute_bound_us, dependency_bound_us,
                              workload_facts)
from .analysis.idle import (WorkConservationViolation, idle_decomposition,
                            work_conservation_violations)
from .analysis.sensitivity import (RANGED, UNCERTAIN_INPUTS, UNKNOWNS, Interval,
                                   Ranged, Unknown, propagate)
from .config.links import DEFAULT_LINKS, EDGE_AXES, SharedAxis, StageLink
from .config.stages import StageVocabulary, default_vocabulary
from .config.readiness import Readiness
from .config.roles import (DEFAULT_ROLES, DEFAULT_STAGE_ROLES, ROLES,
                           RoleAssignment, VECTOR_ROLES)
from .config.platform import (ASCEND_950DT, ASCEND_950PR, PlatformSpec,
                              cube_mac_per_us, resolve_platform)
from .config.policy import InstancePolicy, StageWaveOffsets
from .config.pipeline import (
    BufferSlots, PhaseRates, PipelineConstraints, QueueDepths, SyncLatency,
    parse_tiling,
)
from .config.provenance import (
    SourcedInt, SourcedValue, collect_provenance, provenance_report,
    run_provenance,
)
from .scheduler.events import (
    Event, RestructureAction, RestructureContext, ScheduledEvent,
    edge_latency,
)
from .scheduler.engine import MultiResourceScheduler
from .scheduler.policies import (
    CriticalPathFirst, EarliestStart, PriorityByStage, SchedulingPolicy,
    WorkConservingCriticalPath,
)
from .planning.waves import (
    ExpertSlice, Position, Wave, calc_layered_first_wave_expert_count,
    calc_layered_steady_wave_expert_count, calc_layered_target_wave_count,
    calc_m_groups_per_wave, plan_layered_waves, plan_waves,
    resolve_gmm1_min_logical_tiles_per_core, swizzle_coord,
)
from .planning.core_assignment import (
    ContiguousBlock, CoreAssignment, GreedyLeastBusy, StaticRoundRobin,
)
from .planning.tile_grid import (
    RowMajorTileGrid, SplitRowsTileGrid, SwizzledTileGrid, Tile, TileGrid,
    validate_tiles,
)
from .planning.wave_packing import (
    BalancedWaves, LongestExpertFirst, SequentialGreedy, WavePacking,
)
from .shape import (
    BlockCursor, CursorTrace, MegaMoeShape, ModelOptions,
)
from .costs import (
    AnalyticalActCosts, AnalyticalCombineCosts, AnalyticalGmmCosts,
    DispatchDataLayout, DispatchMechanisticLatency, LayeredDispatchLayout,
    PrimitiveCosts, UrmaMechanisticLatency, build_analytical_costs,
)
from .builders.base import (
    DispatchCallIR, DispatchExpertIR, EventBuilderBase,
    build_dispatch_expert_ir,
)
from .builders.mte import MteEventBuilder
from .builders.pipeline_expand import apply_pipeline
from .model import A8W8WaveCostModel
from .analysis import (
    bottleneck_report, compare_variants, critical_path_breakdown, design_space,
    extract_critical_path, format_design_space, idle_core_stealing,
    resource_utilization,
)
from .api import simulate_routing_counts
from .profiles import MEGAMOE_A8W8, ReferenceProfile
from .implementations import (CalibrationDomain, CompileConfig,
                              ImplementationId, RuntimeConfig,
                              RuntimeTopology, ShapeDomain, Unsupported)
from .registry import register
from . import guardrails
from .scenario import (Calibration, Scenario, TilingSource, Workload,
                       load_scenario, simulate)

__all__ = [name for name in dir() if not name.startswith('_')]
