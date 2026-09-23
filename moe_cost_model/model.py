"""Wave cost model: shape/options/cursor + DAG 事件构建."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .constants import (
    DAV3510_NONINTERLEAVED_GMM1_ACTIVATION_DEPTH, GMM2_KL1,
    GMM2_LAG_MIN_TOKEN_NUM, GMM2_MIN_LOGICAL_TILES_PER_CORE,
    GMM1_MIN_LOGICAL_TILES_PER_CORE, GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL,
    GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE,
    GMM1_SMALL_BATCH_TOKEN_THRESHOLD, GMM1_LARGE_BATCH_TOKEN_THRESHOLD,
    LEGACY_GMM_MAX_PENDING_TILES, ACTIVATION_N_HALF, L1_TILE_K,
    TILE_M, TILE_N, _gmm2_head_tail_fractions, ceil_div,
)
from .dag import Event, MultiResourceScheduler, ScheduledEvent
from .dispatch import (
    DispatchCallIR, DispatchDataLayout, DispatchMechanisticLatency,
    build_dispatch_expert_ir,
)
from .phases import apply_pipeline
from .pipeline import PipelineConstraints
from .primitives import PrimitiveCosts
from .constants import (
    BW_UNPERMUTE_AGG, KernelConfig, T_CORE_SYNC_BARRIER_US, T_COUNTS_EXPORT_US,
    T_DISPATCH_PREPARE_US, T_FINALIZE_US, T_INIT_US, T_INPUT_QUANT_FIXED_US,
    T_INPUT_QUANT_PER_TOKEN_US, T_OUTPUT_INIT_US, T_RANK_SYNC_RTT_US,
    select_kl1,
)
from .waves import (
    ExpertSlice, Wave, calc_m_groups_per_wave, plan_waves,
    resolve_gmm1_min_logical_tiles_per_core, swizzle_coord,
)


@dataclass
class BlockCursor:
    jobs: int
    start: int = 0

    def owners(self, tile_count: int) -> List[int]:
        if self.jobs <= 0:
            return []
        owners = [((self.start + i) % self.jobs) for i in range(tile_count)]
        self.start = (self.start + tile_count) % self.jobs
        return owners

    def set(self, value: int) -> None:
        if self.jobs > 0:
            self.start = value % self.jobs


@dataclass(frozen=True)
class CursorTrace:
    iteration: int
    gmm1_wave: Optional[int]
    cursor_before_gmm1: int
    cursor_after_gmm1: int
    gmm2_wave: Optional[int]
    cursor_after_gmm2: int
    resonance_fix_applied: bool
    cursor_after_fix: int


@dataclass(frozen=True)
class MegaMoeShape:
    expert_tokens: Tuple[int, ...]
    token_num: int
    h: int
    hidden_dim: int
    aic_num: int
    rank_id: int = 0
    p1_override: int = 0
    p2_override: int = 0
    # Optional [expert][source_rank] counts.  When supplied, row sums must equal
    # expert_tokens and Dispatch's source-rank segment count becomes exact.
    expert_source_tokens: Tuple[Tuple[int, ...], ...] = ()
    # Exact Dispatch tiling/data geometry for the mechanistic model.  Keep this
    # separate from workload shape because routeItemsPerBatch is a tiling result.
    dispatch_layout: Optional[DispatchDataLayout] = None
    # topk (UNPERMUTE 读放大) 与共享专家数 (尾段共享 GMM2), tiling 真值提供
    topk: int = 8
    shared_expert_num: int = 0
    # kernel 编译期参数 (CMake cost-sweep knobs), 默认 = 源码值
    kernel: object = None   # KernelConfig; None → 模块默认

    def __post_init__(self) -> None:
        if self.h <= 0 or self.hidden_dim <= 0 or self.aic_num <= 0:
            raise ValueError("h, hidden_dim and aic_num must be positive")
        if self.token_num < 0:
            raise ValueError("token_num must be non-negative")
        if self.rank_id < 0:
            raise ValueError("rank_id must be non-negative")
        if self.p1_override < 0 or self.p2_override < 0:
            raise ValueError("p1/p2 overrides must be non-negative; zero means default")
        if any(x < 0 for x in self.expert_tokens):
            raise ValueError("expert token counts must be non-negative")
        if self.expert_source_tokens:
            if len(self.expert_source_tokens) != len(self.expert_tokens):
                raise ValueError("expert_source_tokens must have one row per expert")
            widths = {len(row) for row in self.expert_source_tokens}
            if len(widths) > 1:
                raise ValueError("all expert_source_tokens rows must have the same world-size")
            for expert, (row, total) in enumerate(zip(self.expert_source_tokens, self.expert_tokens)):
                if any(x < 0 for x in row) or sum(row) != total:
                    raise ValueError(
                        f"expert_source_tokens[{expert}] must be non-negative and sum to expert_tokens[{expert}]"
                    )


@dataclass(frozen=True)
class ModelOptions:
    combine_no_quant: bool = True
    topk_weights_prefetch: bool = False
    gmm2_lag_threshold: int = GMM2_LAG_MIN_TOKEN_NUM
    serialize_dispatch_comm: bool = False

    # Execution-regime parameters. These are intentionally configurable: the
    # DAV_3510 values below describe the validated non-interleaved path, not a
    # universal hardware constant.
    gmm1_activation_depth: int = DAV3510_NONINTERLEAVED_GMM1_ACTIVATION_DEPTH

    # Evidence chain for the current default:
    # gmm1_activation.h non-interleaved/non-prefetch path profiles
    # WAIT_GMM1_BUFFER around WaitForVector(); WaitForVector is the cross-core
    # handshake waiting for paired AIV0 to consume the previous UB tile and
    # NotifyCube.  Therefore the marker is a depth-1 structural handshake.
    # IMPORTANT: event incidence is not time fraction.  In the B64 capture,
    # 1936/2080 instances carry the marker because each participating AIC's
    # first GMM1 tile is unmarked; this count is structurally invariant to mgw.
    gmm2_combine_credit: Optional[int] = None

    # L0/L1/L2 流水线约束 (None = 全关闭, 现行为)
    pipeline: Optional[PipelineConstraints] = None
    # GMM2 K-窗 (kL1): None = kernel 自适应规则移植 (select_kl1), int = 显式覆盖
    gmm2_kl1: Optional[int] = None

    def __post_init__(self) -> None:
        if self.gmm1_activation_depth <= 0:
            raise ValueError("gmm1_activation_depth must be positive")
        if self.gmm2_combine_credit is not None and self.gmm2_combine_credit <= 0:
            raise ValueError("gmm2_combine_credit must be positive or None")


class A8W8WaveCostModel:
    def __init__(self, costs: PrimitiveCosts, options: ModelOptions = ModelOptions()):
        if not options.combine_no_quant:
            raise NotImplementedError("v3 models A8W8 COMBINE_NO_QUANT only")
        if options.topk_weights_prefetch:
            raise NotImplementedError("v3 models TopkWeightsPrefetch=false only")
        self.costs = costs
        self.options = options
        self._order = 0

    def m_groups_per_wave(self, shape: MegaMoeShape) -> int:
        p1 = shape.p1_override or resolve_gmm1_min_logical_tiles_per_core(shape.token_num)
        p2 = shape.p2_override or GMM2_MIN_LOGICAL_TILES_PER_CORE
        return calc_m_groups_per_wave(
            hidden_dim=shape.hidden_dim, h=shape.h, aic_num=shape.aic_num, p1=p1, p2=p2
        )

    def waves(self, shape: MegaMoeShape) -> List[Wave]:
        km0 = shape.kernel if shape.kernel is not None else None
        kmc = km0 if km0 is not None else KernelConfig()
        return plan_waves(shape.expert_tokens, self.m_groups_per_wave(shape),
                          tile_m=kmc.tile_m)

    @staticmethod
    def _tile_rows(slice_: ExpertSlice, local_mgroup: int, tile_m: int = TILE_M) -> int:
        start = local_mgroup * tile_m
        return max(0, min(tile_m, slice_.rows - start))

    @staticmethod
    def _rotated_balanced_range(total: int, worker: int, workers: int, global_prefix: int) -> Tuple[int, int]:
        """Exact GetRotatedBalancedWorkRange start/count."""
        if workers <= 0 or worker < 0 or worker >= workers:
            return (0, 0)
        first_owner = global_prefix % workers
        logical = worker - first_owner if worker >= first_owner else worker + workers - first_owner
        base, rem = divmod(total, workers)
        extra_before = logical if logical < rem else rem
        start = logical * base + extra_before
        count = base + (1 if logical < rem else 0)
        return start, count

    def _dispatch_call_ir(self, shape: MegaMoeShape, w: Wave, core: int) -> DispatchCallIR:
        """Reconstruct one source-faithful DispatchTokenRange call."""
        if not shape.expert_source_tokens:
            raise ValueError("mechanistic Dispatch requires exact expert_source_tokens")
        layout = shape.dispatch_layout
        if layout is None:
            layout = DispatchDataLayout.from_hidden(shape.h)
        rel_begin, count = self._rotated_balanced_range(w.rows, core, shape.aic_num, w.begin.global_row)
        core_global_begin = w.begin.global_row + rel_begin
        core_global_end = core_global_begin + count
        expert_irs = []
        if count:
            for sl in w.slices:
                overlap_begin = max(core_global_begin, sl.global_row_begin)
                overlap_end = min(core_global_end, sl.global_row_end)
                if overlap_begin >= overlap_end:
                    continue
                local_begin = sl.row_begin + (overlap_begin - sl.global_row_begin)
                local_end = sl.row_begin + (overlap_end - sl.global_row_begin)
                expert_irs.append(
                    build_dispatch_expert_ir(
                        expert=sl.expert,
                        dst_rank=shape.rank_id,
                        source_counts=shape.expert_source_tokens[sl.expert],
                        row_begin=local_begin,
                        row_end=local_end,
                        layout=layout,
                    )
                )
        return DispatchCallIR(
            dst_rank=shape.rank_id,
            wave=w.index,
            aiv1=core,
            global_row_begin=core_global_begin,
            global_row_end=core_global_end,
            experts=tuple(expert_irs),
        )

    def dispatch_ir(self, shape: MegaMoeShape) -> List[DispatchCallIR]:
        """Return one IR record per (wave, AIV1) DispatchTokenRange call."""
        return [
            self._dispatch_call_ir(shape, w, core)
            for w in self.waves(shape)
            for core in range(shape.aic_num)
        ]

    @staticmethod
    def _gmm1_device_scheduler_n(shape: MegaMoeShape,
                                 act_half: int = ACTIVATION_N_HALF) -> int:
        # Mapping validation: non-interleaved A8W8 schedules outputN=hiddenDim/act_half.
        # Host CalcMGroupsPerWave still uses full hiddenDim and must NOT use this.
        return ceil_div(shape.hidden_dim, act_half)

    def _event(
        self,
        events: List[Event],
        name: str,
        resources: Sequence[str],
        duration_us: float,
        deps: Iterable[str] = (),
        meta: Optional[Mapping[str, object]] = None,
        dep_latency_us: float = 0.0,
    ) -> str:
        dep_tuple = tuple(dict.fromkeys(d for d in deps if d))
        events.append(
            Event(
                name=name,
                resources=tuple(resources),
                duration_us=max(0.0, float(duration_us)),
                deps=dep_tuple,
                order=self._order,
                meta=dict(meta or {}),
                dep_latency_us=dep_latency_us,
            )
        )
        self._order += 1
        return name

    def build_events(self, shape: MegaMoeShape) -> Tuple[List[Event], List[CursorTrace]]:
        km = shape.kernel if shape.kernel is not None else KernelConfig()
        TILE_M = km.tile_m      # 局部遮蔽模块常数 (结构点全部走 km)
        TILE_N = km.tile_n
        ACT_HALF = km.activation_n_half
        self._order = 0
        waves = self.waves(shape)
        p = shape.aic_num
        c = self.costs
        events: List[Event] = []
        cursor = BlockCursor(p, 0)
        cursor_trace: List[CursorTrace] = []

        # Kernel program-order tails for each physical role/core.
        aic_prev: List[Optional[str]] = [None] * p
        aiv0_prev: List[Optional[str]] = [None] * p
        aiv1_prev: List[Optional[str]] = [None] * p

        # Slot/credit histories persist across expert and wave calls.
        gmm1_activation_history: List[List[str]] = [[] for _ in range(p)]
        gmm2_combine_history: List[List[str]] = [[] for _ in range(p)]

        # Dispatch readiness is an explicit zero-duration join event per GMM1 M-tile.
        # This is the primary Dispatch->GMM1 interface of the model:
        #   t_dispatchReady(dst_rank, expert, mgroup) = schedule time of this join.
        dispatch_ready_event: Dict[Tuple[int, int], str] = {}
        activation_ready: Dict[Tuple[int, int], List[str]] = {}

        # TOKEN_COUNT_PREPARE: 跨卡专家计数表的准备, 是一次性的启动门控.
        # 物理执行: 仅 block 0 的 AIV1 做实际 cumsum; 其余核只是等待同一就绪信号.
        # 实测: 84 个事件各 ~1.3µs (非 AIV1 核的函数进出开销), 总 busy ~112µs.
        # 53.9µs 是首个事件 begin 到最后事件 end 的 span (含跨卡等待), 是墙钟门控,
        # 不是每核 busy. 模型应表示为: 单事件阻塞所有后续 dispatch (时长 = span).
        # 每核另加 ~1.3µs 的进入/退出开销 (分派给 dispatch_call 的 T_call_oh 已覆盖).
        # ---- 前导链 (arch35.h:660-676): INIT → INPUT_QUANT → COUNT_GATE → 调度准备 ----
        init_name = "preamble.init"
        self._event(events, init_name, (), T_INIT_US, deps=(),
                    meta={"stage": "input_quant", "part": "init"})
        aiv_n = km.aiv_num or (2 * shape.aic_num)
        aiv_per_core = -(-shape.token_num // aiv_n) if shape.token_num else 1
        quant_name = "preamble.input_quant"
        self._event(
            events, quant_name, (),
            T_INPUT_QUANT_FIXED_US + aiv_per_core * T_INPUT_QUANT_PER_TOKEN_US,
            deps=(init_name,),
            meta={"stage": "input_quant", "part": "quant"})
        token_count_name = "token_count_prepare.gate"
        self._event(
            events,
            token_count_name,
            (),  # 无资源占用 (纯等待)
            c.count_table_prepare_us + T_DISPATCH_PREPARE_US,  # 跨卡门控 + 调度准备
            deps=(quant_name,),
            meta={"stage": "count_table_prepare", "role": "gate"},
        )
        for core in range(p):
            # 每核的串行链头指向同一个门控事件
            aiv1_prev[core] = token_count_name
            aic_prev[core] = token_count_name

        def add_dispatch_wave(w: Wave) -> None:
            # Source executes one DispatchTokenRange(w) call on every AIV1.
            # Low-level service-time calibration remains per AIV1/expert work, but
            # the externally meaningful result is the M-tile ready join built below.
            # Each tuple is (dispatch_event, contributed_rows, core, dispatch_call_event).
            wave_contrib: Dict[Tuple[int, int], List[Tuple[str, int, int, str]]] = {}
            # Mechanistic FCFS window queueing: durations depend on the whole
            # wave's stream set, so build all call IRs first, then simulate.
            call_irs = [self._dispatch_call_ir(shape, w, core) for core in range(p)]
            slice_durs = c.dispatch_mechanistic.simulate_wave(
                call_irs, shape.dispatch_layout or DispatchDataLayout.from_hidden(shape.h)
            )
            for core in range(p):
                call_ir = call_irs[core]
                call_name = f"W{w.index}.dispatch_call.c{core}"
                deps = [aiv1_prev[core]] if aiv1_prev[core] else []
                call_duration = c.dispatch_mechanistic.call_base_us()
                call_meta = {"stage": "dispatch_call", "wave": w.index, "core": core}
                call_meta.update({f"mech_{k}": v for k, v in call_ir.features().items()})
                self._event(
                    events,
                    call_name,
                    (f"AIV1:{core}",),
                    call_duration,
                    deps=deps,
                    meta=call_meta,
                )
                aiv1_prev[core] = call_name

                rel_begin, count = self._rotated_balanced_range(w.rows, core, p, w.begin.global_row)
                if count == 0:
                    continue
                core_global_begin = w.begin.global_row + rel_begin
                core_global_end = core_global_begin + count

                expert_ir_by_id = {e.expert: e for e in call_ir.experts}
                for sl in w.slices:
                    overlap_begin = max(core_global_begin, sl.global_row_begin)
                    overlap_end = min(core_global_end, sl.global_row_end)
                    if overlap_begin >= overlap_end:
                        continue
                    local_begin = sl.row_begin + (overlap_begin - sl.global_row_begin)
                    local_end = sl.row_begin + (overlap_end - sl.global_row_begin)
                    name = (
                        f"W{w.index}.dispatch.c{core}.e{sl.expert}."
                        f"r{local_begin}_{local_end}"
                    )
                    resources = [f"AIV1:{core}"]
                    deps = [aiv1_prev[core]] if aiv1_prev[core] else []

                    expert_ir = expert_ir_by_id[sl.expert]
                    ef = expert_ir.features()
                    if ef["remote_segments"] and self.options.serialize_dispatch_comm:
                        resources.append("DISPATCH_COMM")
                    # FCFS-simulated slice duration (base service + window wait)
                    duration = slice_durs[core][sl.expert] + c.dispatch_ready_publish_us
                    meta = {
                        "stage": "dispatch", "wave": w.index, "core": core,
                        "expert": sl.expert, "row_begin": local_begin, "row_end": local_end,
                        **{f"mech_{k}": v for k, v in ef.items()},
                        # Keep legacy aliases for diagnostics.
                        "rows": ef["rows"],
                        "local_segments": ef["local_segments"],
                        "remote_segments": ef["remote_segments"],
                        "local_rows": ef["local_source_row_fetch_ops"],
                        "remote_rows": ef["remote_source_row_fetch_ops"],
                        "source_segments": ef["local_segments"] + ef["remote_segments"],
                    }

                    self._event(events, name, resources, duration, deps=deps, meta=meta)
                    aiv1_prev[core] = name

                    first_group = local_begin // TILE_M
                    last_group = (local_end - 1) // TILE_M
                    for group in range(first_group, last_group + 1):
                        group_begin = group * TILE_M
                        group_end = min(group_begin + TILE_M, shape.expert_tokens[sl.expert])
                        contributed_rows = max(0, min(local_end, group_end) - max(local_begin, group_begin))
                        if contributed_rows:
                            wave_contrib.setdefault((sl.expert, group), []).append(
                                (name, contributed_rows, core, call_name)
                            )

            # Materialize one readiness join for every GMM1 M-tile in this wave.
            # PublishGmm1TileReady increments the counter only after one AIV1 has
            # completed its entire expert-local range, so the contributor event's
            # finish time is exactly the publication time represented here.
            for sl in w.slices:
                first_group = sl.row_begin // TILE_M
                for local_group in range(sl.m_groups):
                    group = first_group + local_group
                    key = (sl.expert, group)
                    if key in dispatch_ready_event:
                        raise ValueError(f"duplicate DispatchReady producer for {key}")
                    group_begin = group * TILE_M
                    required_rows = max(0, min(TILE_M, shape.expert_tokens[sl.expert] - group_begin))
                    contrib = wave_contrib.get(key, [])
                    contributed_rows = sum(x[1] for x in contrib)
                    if contributed_rows != required_rows:
                        raise ValueError(
                            f"DispatchReady row mismatch for expert={sl.expert}, group={group}: "
                            f"got {contributed_rows}, expected {required_rows}"
                        )
                    deps = tuple(x[0] for x in contrib)
                    ready_name = f"W{w.index}.dispatch_ready.e{sl.expert}.g{group}"
                    self._event(
                        events, ready_name, (), 0.0, deps=deps,
                        meta={
                            "stage": "dispatch_ready",
                            "dst_rank": shape.rank_id,
                            "wave": w.index,
                            "expert": sl.expert,
                            "mgroup": group,
                            "required_rows": required_rows,
                            "contributed_rows": contributed_rows,
                            "contributor_count": len(contrib),
                            "contributor_events": tuple(x[0] for x in contrib),
                            "contributor_rows": tuple(x[1] for x in contrib),
                            "contributor_cores": tuple(x[2] for x in contrib),
                            "contributor_call_events": tuple(x[3] for x in contrib),
                        },
                    )
                    dispatch_ready_event[key] = ready_name

        def add_gmm1_wave(w: Wave) -> None:
            gmm1_scheduler_n = self._gmm1_device_scheduler_n(shape, km.activation_n_half)
            gmm1_n_tiles = ceil_div(gmm1_scheduler_n, TILE_N)
            for si, sl in enumerate(w.slices):
                tile_count = sl.m_groups * gmm1_n_tiles
                owners = cursor.owners(tile_count)
                first_owned_on_core = [True] * p

                # B 矩阵 L2 复用: 同一专家在同一波内的矩阵乘任务中,
                # B 从 GM 只需加载一次/列 (N-tile), 后续 M-group 从 L2 命中.
                # A 矩阵每行唯一, 每 (M-group, N-tile) 从 GM 加载.
                # 源码: SetWaveWeightL2CacheHint (gmm_common.h:395-423) 在
                # 专家行数 > tileM 时保留 B 在 L2.
                #
                # 载入字节 = m_total*K + n_tiles*n_half*K*tileN  (GM 唯一字节)
                # 计算量   = 2*m_total*schedulerN*K            (总 MACs)
                # T_task   = max(载入/BW, 计算/R_cube) + T_fill
                m_total = sl.rows  # 该专家在本波的总行数
                n_half = ACT_HALF
                k = shape.h
                a_bytes = m_total * k
                b_bytes = gmm1_n_tiles * 2 * k * TILE_N  # 2 = SwiGLU 双投影
                compute_macs = 2 * m_total * gmm1_scheduler_n * k
                # 解析 GMM1 模型: 需要带宽和计算速率都已标定
                if c.gmm1_bw_bytes_per_us is not None and c.gmm1_mac_per_us is not None:
                    if km.weight_nz:
                        if c.gmm1_bw_b_nz_bytes_per_us is None:
                            raise ValueError(
                                "KernelConfig.weight_nz=True 需要 PrimitiveCosts."
                                "gmm1_bw_b_nz_bytes_per_us (NZ 路径实测带宽)")
                        load_time = (a_bytes / c.gmm1_bw_bytes_per_us
                                     + b_bytes / c.gmm1_bw_b_nz_bytes_per_us)
                    else:
                        load_time = (a_bytes + b_bytes) / c.gmm1_bw_bytes_per_us
                    if km.l1_buf_num == 1:
                        t_task = (load_time + compute_macs / c.gmm1_mac_per_us
                                  + getattr(c, 'gmm1_tile_restart_us', 0.0) * tile_count)
                    else:
                        t_task = max(
                            load_time,
                            compute_macs / c.gmm1_mac_per_us,
                        )
                    t_task += c.gmm1_fill_us
                    per_tile = t_task / tile_count if tile_count > 0 else 0.0
                else:
                    t_task = None  # 回退到逐 tile callable

                for tile_idx, core in enumerate(owners):
                    mg = tile_idx // gmm1_n_tiles
                    nt = tile_idx % gmm1_n_tiles
                    m_rows = self._tile_rows(sl, mg, tile_m=km.tile_m)
                    global_group = sl.row_begin // TILE_M + mg

                    deps: List[str] = []
                    if aic_prev[core]:
                        deps.append(aic_prev[core])
                    ready_name = dispatch_ready_event.get((sl.expert, global_group))
                    if ready_name is None:
                        raise ValueError(
                            f"missing t_dispatchReady for expert={sl.expert}, group={global_group}"
                        )
                    deps.append(ready_name)

                    # GMM1->Activation backpressure. Current DAV_3510
                    # non-interleaved measurements give effective depth=1.
                    history = gmm1_activation_history[core]
                    depth = self.options.gmm1_activation_depth
                    if len(history) >= depth:
                        deps.append(history[-depth])

                    # 逐 tile 时长: 优先用专家波内任务的解析模型 (含 B 复用),
                    # 无则回退到逐 tile callable
                    if t_task is not None:
                        duration = per_tile
                    else:
                        duration = c.gmm1_tile(m_rows, shape.h)
                    if first_owned_on_core[core]:
                        duration += c.gmm1_problem_startup_us
                        first_owned_on_core[core] = False

                    gname = f"W{w.index}.E{sl.expert}.S{si}.gmm1.m{mg}.n{nt}.c{core}"
                    self._event(
                        events,
                        gname,
                        (f"AIC:{core}",),
                        duration,
                        deps=deps,
                        meta={
                            "stage": "gmm1",
                            "wave": w.index,
                            "expert": sl.expert,
                            "slice": si,
                            "mgroup": global_group,
                            "ntile": nt,
                            "core": core,
                            "m_rows": m_rows,
                            "cursor_tile": tile_idx,
                            "dispatch_ready_event": ready_name,
                        },
                    )
                    aic_prev[core] = gname

                    adeps = [gname]
                    if aiv0_prev[core]:
                        adeps.append(aiv0_prev[core])
                    aname = f"W{w.index}.E{sl.expert}.S{si}.act.m{mg}.n{nt}.c{core}"
                    logical_n = min(TILE_N, gmm1_scheduler_n - nt * TILE_N)
                    self._event(
                        events,
                        aname,
                        (f"AIV0:{core}",),
                        c.activation_tile(m_rows, logical_n / TILE_N) + c.activation_ready_publish_us,
                        deps=adeps,
                        meta={
                            "stage": "activation",
                            "wave": w.index,
                            "expert": sl.expert,
                            "slice": si,
                            "mgroup": global_group,
                            "ntile": nt,
                            "core": core,
                            "m_rows": m_rows,
                        },
                    )
                    aiv0_prev[core] = aname
                    history.append(aname)
                    activation_ready.setdefault((sl.expert, global_group), []).append(aname)

        def add_gmm2_wave(w: Wave, call_iteration: int) -> None:
            gmm2_n_tiles = ceil_div(shape.h, TILE_N)
            expected_activation_tiles = ceil_div(
                        self._gmm1_device_scheduler_n(shape, ACT_HALF), TILE_N)

            for si, sl in enumerate(w.slices):
                tile_count = sl.m_groups * gmm2_n_tiles
                owners = cursor.owners(tile_count)
                first_owned_on_core = [True] * p

                for tile_idx, core in enumerate(owners):
                    mg, nt = swizzle_coord(tile_idx, sl.m_groups, gmm2_n_tiles,
                                  km.swizzle_offset, km.swizzle_direction)
                    m_rows = self._tile_rows(sl, mg, tile_m=km.tile_m)
                    global_group = sl.row_begin // TILE_M + mg
                    ready = activation_ready.get((sl.expert, global_group), [])
                    if len(ready) != expected_activation_tiles:
                        raise ValueError(
                            f"activation readiness incomplete for expert={sl.expert}, group={global_group}: "
                            f"got {len(ready)}, expected {expected_activation_tiles}"
                        )

                    deps: List[str] = []
                    if aic_prev[core]:
                        deps.append(aic_prev[core])
                    # k-window 粒度就绪: GMM2 的 K=intermediate=8个act tile拼接,
                    # k-window w 只消费 act(g,w)。head 只等 act(g,0) 即可启动,
                    # tail 等 act(g,7) (最后一个 k-window 的输入)。
                    # 旧组屏障(等全部8个)过保守, 实测 WAIT_GMM2_INPUT 高估 1.6x。
                    combine_history = gmm2_combine_history[core]
                    credit = self.options.gmm2_combine_credit
                    if credit is not None and len(combine_history) >= credit:
                        deps.append(combine_history[-credit])

                    # GMM2 输出宽 = H（回 hidden）；B 矩阵的 K = intermediate = hiddenDim / n_half
                    k_gmm2 = shape.hidden_dim // ACT_HALF
                    # GMM2 解析路径 (参数已设) 或回退 callable
                    if c.gmm2_bw_bytes_per_us is not None:
                        duration = k_gmm2 * TILE_N / c.gmm2_bw_bytes_per_us
                    else:
                        duration = c.gmm2_tile(m_rows, k_gmm2)
                    kl1 = select_kl1(sl.rows, k_gmm2, self.options.gmm2_kl1,
                                tile_m=km.tile_m, tile_n=km.tile_n,
                                l1_size=km.l1_size)
                    head_frac, tail_frac = _gmm2_head_tail_fractions(k_gmm2, kl1)
                    if first_owned_on_core[core]:
                        duration += c.gmm2_problem_startup_us
                        first_owned_on_core[core] = False

                    gname = f"W{w.index}.E{sl.expert}.S{si}.gmm2.m{mg}.n{nt}.c{core}"
                    head_deps = deps + [ready[0]]
                    tail_deps = [gname + ".h", ready[-1]]
                    self._event(
                        events,
                        gname + ".h",
                        (f"AIC:{core}",),
                        duration * head_frac,
                        deps=head_deps,
                        meta={
                            "stage": "gmm2",
                            "wave": w.index,
                            "call_iteration": call_iteration,
                            "expert": sl.expert,
                            "slice": si,
                            "mgroup": global_group,
                            "ntile": nt,
                            "core": core,
                            "m_rows": m_rows,
                            "cursor_tile": tile_idx,
                            "part": "head",
                        },
                    )
                    self._event(
                        events,
                        gname,
                        (f"AIC:{core}",),
                        duration * tail_frac,
                        deps=tail_deps,
                        meta={
                            "stage": "gmm2",
                            "wave": w.index,
                            "call_iteration": call_iteration,
                            "expert": sl.expert,
                            "slice": si,
                            "mgroup": global_group,
                            "ntile": nt,
                            "core": core,
                            "m_rows": m_rows,
                            "cursor_tile": tile_idx,
                            "part": "tail",
                        },
                    )
                    aic_prev[core] = gname

                    cdeps = [gname]
                    # AIV1 source program order: Dispatch call for this outer iteration
                    # precedes the GMM2/Combine call; previous Combine calls also precede
                    # next iteration's Dispatch call.
                    if aiv1_prev[core]:
                        cdeps.append(aiv1_prev[core])
                    cname = f"W{w.index}.E{sl.expert}.S{si}.combine.m{mg}.n{nt}.c{core}"
                    logical_n = min(TILE_N, shape.h - nt * TILE_N)
                    self._event(
                        events,
                        cname,
                        (f"AIV1:{core}",),
                        c.combine_tile(m_rows, logical_n / TILE_N) + c.combine_ack_us,
                        deps=cdeps,
                        meta={
                            "stage": "combine",
                            "wave": w.index,
                            "call_iteration": call_iteration,
                            "expert": sl.expert,
                            "slice": si,
                            "mgroup": global_group,
                            "ntile": nt,
                            "core": core,
                            "m_rows": m_rows,
                        },
                    )
                    aiv1_prev[core] = cname
                    combine_history.append(cname)

        if not waves:
            return events, cursor_trace

        # ---- 尾段链 (arch35.h:735-775 AIV 序列) ----
        # counts_export(每核最后 combine 后) → 核间 barrier → [共享专家GMM2]
        # → 跨卡同步 → 输出缓冲初始化 → UNPERMUTE → FINALIZE
        last_combine_per_core = {}
        for ev in events:
            if str(ev.meta.get("stage", "")) == "combine":
                core = ev.meta.get("core")
                last_combine_per_core[core] = ev.name
        ce_deps = tuple(sorted(last_combine_per_core.values()))
        counts_export = "epilogue.counts_export"
        self._event(events, counts_export, (), T_COUNTS_EXPORT_US, deps=ce_deps,
                    meta={"stage": "epilogue", "part": "counts_export"})
        core_sync = "epilogue.output_core_sync"
        self._event(events, core_sync, (), T_CORE_SYNC_BARRIER_US, deps=(counts_export,),
                    meta={"stage": "epilogue", "part": "output_core_sync"})
        tail_head = core_sync
        if getattr(shape, "shared_expert_num", 0):
            # 共享专家 GMM2 (尾段内, W2 权重流读主导): I×H FP8 / 聚合带宽
            w2_bytes = (shape.hidden_dim // ACT_HALF) * shape.h
            shared_gmm2 = "epilogue.shared_gmm2"
            self._event(events, shared_gmm2, (), w2_bytes / BW_UNPERMUTE_AGG,
                        deps=(core_sync,), meta={"stage": "epilogue", "part": "shared_gmm2"})
            tail_head = shared_gmm2
        rank_sync = "epilogue.output_rank_sync"
        self._event(events, rank_sync, (), T_RANK_SYNC_RTT_US, deps=(tail_head,),
                    meta={"stage": "epilogue", "part": "output_rank_sync"})
        out_init = "epilogue.output_buffer_init"
        self._event(events, out_init, (), T_OUTPUT_INIT_US, deps=(rank_sync,),
                    meta={"stage": "epilogue", "part": "output_buffer_init"})
        # UNPERMUTE: 读 topk×h×BF16 + 写 h×BF16 (peermem 流式, 聚合带宽)
        unpermute_bytes = shape.token_num * (shape.topk * shape.h * 2 + shape.h * 2)
        unpermute = "epilogue.unpermute"
        self._event(events, unpermute, (), unpermute_bytes / BW_UNPERMUTE_AGG,
                    deps=(out_init,), meta={"stage": "epilogue", "part": "unpermute"})
        fin = "epilogue.finalize"
        self._event(events, fin, (), T_FINALIZE_US, deps=(unpermute,),
                    meta={"stage": "epilogue", "part": "finalize"})

        lag = shape.token_num >= self.options.gmm2_lag_threshold
        dispatched: set[int] = set()

        for iteration, w in enumerate(waves):
            # AIV1 dispatch call order from PrepareDispatchWave:
            # first iteration executes W0 then prefetches W1; later iterations
            # prefetch W(i+1) before GMM2/Combine of this iteration.
            if iteration == 0:
                for dispatch_index in (0, 1):
                    if dispatch_index < len(waves) and dispatch_index not in dispatched:
                        add_dispatch_wave(waves[dispatch_index])
                        dispatched.add(dispatch_index)
            else:
                dispatch_index = iteration + 1
                if dispatch_index < len(waves) and dispatch_index not in dispatched:
                    add_dispatch_wave(waves[dispatch_index])
                    dispatched.add(dispatch_index)

            cursor_before = cursor.start
            add_gmm1_wave(w)
            cursor_after_gmm1 = cursor.start

            gmm2_wave_idx: Optional[int]
            if lag:
                gmm2_wave_idx = iteration - 1 if iteration > 0 else None
            else:
                gmm2_wave_idx = iteration

            if gmm2_wave_idx is not None:
                add_gmm2_wave(waves[gmm2_wave_idx], call_iteration=iteration)
            cursor_after_gmm2 = cursor.start

            has_next_wave = iteration + 1 < len(waves)
            fixed_role_resonance = (
                has_next_wave
                and cursor_after_gmm2 == cursor_before
                and cursor_after_gmm1 != cursor_before
            )
            if fixed_role_resonance:
                cursor.set(cursor_after_gmm1)

            cursor_trace.append(
                CursorTrace(
                    iteration=iteration,
                    gmm1_wave=w.index,
                    cursor_before_gmm1=cursor_before,
                    cursor_after_gmm1=cursor_after_gmm1,
                    gmm2_wave=gmm2_wave_idx,
                    cursor_after_gmm2=cursor_after_gmm2,
                    resonance_fix_applied=fixed_role_resonance,
                    cursor_after_fix=cursor.start,
                )
            )

        # Source drains the final pending GMM2 wave after the main loop when lag is active.
        if lag:
            drain_before = cursor.start
            add_gmm2_wave(waves[-1], call_iteration=len(waves))
            cursor_trace.append(
                CursorTrace(
                    iteration=len(waves),
                    gmm1_wave=None,
                    cursor_before_gmm1=drain_before,
                    cursor_after_gmm1=drain_before,
                    gmm2_wave=waves[-1].index,
                    cursor_after_gmm2=cursor.start,
                    resonance_fix_applied=False,
                    cursor_after_fix=cursor.start,
                )
            )

        # Explicit role-return control points make this expert-stage DAG composable
        # with later shared-expert / output-sync / unpermute stages.  AIC returns
        # from ProcessMoeExpertStages only after the GMM1 UB ping-pong is drained.
        for core in range(p):
            aic_deps: List[str] = []
            if aic_prev[core]:
                aic_deps.append(aic_prev[core])
            # EndSync waits the still-live ping-pong ACKs; depending on all of the
            # last two Activation events is equivalent and harmless if one already
            # completed much earlier.
            depth = self.options.gmm1_activation_depth
            aic_deps.extend(gmm1_activation_history[core][-depth:])
            done = f"moe_expert_stage_done.aic.c{core}"
            self._event(
                events, done, (f"AIC:{core}",), 0.0, deps=aic_deps,
                meta={"stage": "moe_stage_done", "role": "aic", "core": core},
            )
            aic_prev[core] = done

            v0deps = [aiv0_prev[core]] if aiv0_prev[core] else []
            done0 = f"moe_expert_stage_done.aiv0.c{core}"
            self._event(
                events, done0, (f"AIV0:{core}",), 0.0, deps=v0deps,
                meta={"stage": "moe_stage_done", "role": "aiv0", "core": core},
            )
            aiv0_prev[core] = done0

            v1deps = [aiv1_prev[core]] if aiv1_prev[core] else []
            done1 = f"moe_expert_stage_done.aiv1.c{core}"
            self._event(
                events, done1, (f"AIV1:{core}",), 0.0, deps=v1deps,
                meta={"stage": "moe_stage_done", "role": "aiv1", "core": core},
            )
            aiv1_prev[core] = done1

        # All wave groups used by GMM1 must have one explicit DispatchReady join.
        missing_dispatch: List[Tuple[int, int]] = []
        for w in waves:
            for sl in w.slices:
                first_group = sl.row_begin // TILE_M
                for local_group in range(sl.m_groups):
                    key = (sl.expert, first_group + local_group)
                    if key not in dispatch_ready_event:
                        missing_dispatch.append(key)
        if missing_dispatch:
            raise ValueError(f"missing DispatchReady joins for groups: {missing_dispatch[:12]}")

        return events, cursor_trace

    def simulate(self, shape: MegaMoeShape) -> Dict[str, object]:
        events, cursor_trace = self.build_events(shape)
        capacities: Dict[str, int] = {}
        channels: Dict[str, object] = {}
        if self.options.pipeline is not None:
            events, capacities, channels = apply_pipeline(
                events, self.options.pipeline,
                aic_num=shape.aic_num, h=shape.h,
                gmm1_act_depth=self.options.gmm1_activation_depth,
                kernel=shape.kernel,
            )
        total_us, scheduled = MultiResourceScheduler().schedule(
            events, capacities=capacities or None, channels=channels or None
        )
        waves = self.waves(shape)

        resource_busy: Dict[str, float] = {}
        resource_first: Dict[str, float] = {}
        resource_last: Dict[str, float] = {}
        stage_busy: Dict[str, float] = {}
        stage_first: Dict[str, float] = {}
        stage_last: Dict[str, float] = {}
        stage_dependency_wait: Dict[str, float] = {}
        stage_resource_queue: Dict[str, float] = {}

        for ev in scheduled:
            duration = ev.end_us - ev.start_us
            for resource in ev.resources:
                resource_busy[resource] = resource_busy.get(resource, 0.0) + duration
                resource_first[resource] = min(resource_first.get(resource, ev.start_us), ev.start_us)
                resource_last[resource] = max(resource_last.get(resource, ev.end_us), ev.end_us)
            stage = str(ev.meta.get("stage", "other"))
            stage_busy[stage] = stage_busy.get(stage, 0.0) + duration
            stage_first[stage] = min(stage_first.get(stage, ev.start_us), ev.start_us)
            stage_last[stage] = max(stage_last.get(stage, ev.end_us), ev.end_us)
            stage_dependency_wait[stage] = stage_dependency_wait.get(stage, 0.0) + ev.dependency_wait_us
            stage_resource_queue[stage] = stage_resource_queue.get(stage, 0.0) + ev.resource_queue_us

        resource_span = {
            r: resource_last[r] - resource_first[r]
            for r in resource_busy
        }
        resource_idle = {
            r: max(0.0, resource_span[r] - resource_busy[r])
            for r in resource_busy
        }
        resource_utilization = {
            r: (resource_busy[r] / resource_span[r] if resource_span[r] > 0 else 0.0)
            for r in resource_busy
        }

        scheduled_by_name = {ev.name: ev for ev in scheduled}
        gmm1_by_group: Dict[Tuple[int, int], List[ScheduledEvent]] = {}
        for ev in scheduled:
            if ev.meta.get("stage") == "gmm1":
                gmm1_by_group.setdefault(
                    (int(ev.meta["expert"]), int(ev.meta["mgroup"])), []
                ).append(ev)

        dispatch_ready_tiles: List[Dict[str, object]] = []
        for ev in scheduled:
            if ev.meta.get("stage") != "dispatch_ready":
                continue
            expert = int(ev.meta["expert"])
            mgroup = int(ev.meta["mgroup"])
            contributor_events = tuple(ev.meta.get("contributor_events", ()))
            contributor_calls = tuple(ev.meta.get("contributor_call_events", ()))
            work_events = [scheduled_by_name[n] for n in contributor_events]
            call_events = [scheduled_by_name[n] for n in contributor_calls]
            first_row_work = min((x.start_us for x in work_events), default=ev.start_us)
            first_call_start = min((x.start_us for x in call_events), default=first_row_work)
            service_sum = sum(x.end_us - x.start_us for x in work_events)
            g1 = gmm1_by_group.get((expert, mgroup), [])
            g1_first = min((x.start_us for x in g1), default=None)
            g1_last_end = max((x.end_us for x in g1), default=None)
            dispatch_ready_tiles.append({
                "dst_rank": int(ev.meta["dst_rank"]),
                "wave": int(ev.meta["wave"]),
                "expert": expert,
                "mgroup": mgroup,
                "required_rows": int(ev.meta["required_rows"]),
                "contributed_rows": int(ev.meta["contributed_rows"]),
                "contributor_count": int(ev.meta["contributor_count"]),
                "contributor_cores": tuple(ev.meta.get("contributor_cores", ())),
                "contributor_rows": tuple(ev.meta.get("contributor_rows", ())),
                "t_dispatchReady_us": ev.end_us,
                "dispatch_call_window_start_us": first_call_start,
                "first_row_work_start_us": first_row_work,
                "dispatch_ready_from_call_start_us": ev.end_us - first_call_start,
                "dispatch_row_work_span_us": ev.end_us - first_row_work,
                "dispatch_contributor_service_sum_us": service_sum,
                "gmm1_first_start_us": g1_first,
                "gmm1_last_end_us": g1_last_end,
                "ready_to_gmm1_start_wait_us": (g1_first - ev.end_us) if g1_first is not None else None,
            })
        dispatch_ready_tiles.sort(key=lambda x: (x["t_dispatchReady_us"], x["expert"], x["mgroup"]))

        # Reconstruct one realized critical chain through explicit dependencies and
        # serial resource order.  This is schedule-specific, not a hardware theorem.
        critical_path: List[Dict[str, object]] = []
        if scheduled:
            tail = max(scheduled, key=lambda x: (x.end_us, x.order))
            seen = set()
            cur: Optional[ScheduledEvent] = tail
            while cur is not None and cur.name not in seen:
                seen.add(cur.name)
                critical_path.append({
                    "name": cur.name,
                    "stage": str(cur.meta.get("stage", "other")),
                    "start_us": cur.start_us,
                    "end_us": cur.end_us,
                    "duration_us": cur.end_us - cur.start_us,
                    "critical_reason": cur.critical_reason,
                    "critical_parent": cur.critical_parent,
                    "resources": cur.resources,
                })
                cur = scheduled_by_name.get(cur.critical_parent) if cur.critical_parent else None
            critical_path.reverse()

        return {
            "total_us": total_us,
            "m_groups_per_wave": self.m_groups_per_wave(shape),
            "wave_count": len(waves),
            "waves": waves,
            "cursor_trace": cursor_trace,
            "events": scheduled,
            "resource_busy_us": resource_busy,
            "resource_span_us": resource_span,
            "resource_idle_us": resource_idle,
            "resource_utilization": resource_utilization,
            "dispatch_ready_tiles": dispatch_ready_tiles,
            "critical_path": critical_path,
            "stage_busy_us": stage_busy,
            "stage_first_start_us": stage_first,
            "stage_last_end_us": stage_last,
            "stage_dependency_wait_us": stage_dependency_wait,
            "stage_resource_queue_us": stage_resource_queue,
            "gmm2_lag_active": shape.token_num >= self.options.gmm2_lag_threshold,
        }

    def structural_summary(self, shape: MegaMoeShape) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        g1n = ceil_div(self._gmm1_device_scheduler_n(shape), TILE_N)
        g2n = ceil_div(shape.h, TILE_N)
        for w in self.waves(shape):
            rows.append(
                {
                    "wave": w.index,
                    "rows": w.rows,
                    "m_groups": w.m_groups,
                    "expert_slices": len(w.slices),
                    "gmm1_tiles": sum(s.m_groups * g1n for s in w.slices),
                    "gmm2_tiles": sum(s.m_groups * g2n for s in w.slices),
                    "begin": (w.begin.expert, w.begin.row, w.begin.global_row),
                    "end": (w.end.expert, w.end.row, w.end.global_row),
                    "slices": tuple(
                        (s.expert, s.row_begin, s.row_end, s.m_groups)
                        for s in w.slices
                    ),
                }
            )
        return rows

    def cursor_summary(self, shape: MegaMoeShape) -> List[CursorTrace]:
        _, trace = self.build_events(shape)
        return trace

