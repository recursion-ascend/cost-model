"""形状/选项/游标: 模型的输入数据结构与建图工作状态中的轻量类型."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config.hardware import EpilogueOverheads
from .config.policy import InstancePolicy
from .costs import DispatchDataLayout
from .config.pipeline import PipelineConstraints


@dataclass
class BlockCursor:
    """轮转游标: 相位 start ∈ [0, jobs), 发牌规则 core(idx) = (start + idx) mod jobs."""

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
    """一个 dst rank 的完整形状: 工作量 + 绑定 + 策略接口."""

    expert_tokens: Tuple[int, ...]
    token_num: int
    h: int
    hidden_dim: int
    aic_num: int
    rank_id: int = 0
    p1_override: int = 0
    p2_override: int = 0
    expert_source_tokens: Tuple[Tuple[int, ...], ...] = ()
    dispatch_layout: Optional[DispatchDataLayout] = None
    topk: int = 8
    shared_expert_num: int = 0
    kernel: object = None       # KernelConfig; None → 模块默认
    core_assignment: object = None  # CoreAssignment; None → StaticRoundRobin
    wave_packing: object = None     # WavePacking; None → SequentialGreedy
    scheduling_policy: object = None  # SchedulingPolicy; None → EarliestStart
    tile_grid: object = None    # TileGrid; None → SwizzledTileGrid (kernel 现行为)
    orchestration: object = None  # 建图器类; None → 按 kernel.topo_urma 自动选
    policy: object = None       # InstancePolicy; None → 默认实例

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
        if self.policy is None:
            object.__setattr__(self, "policy", InstancePolicy())
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
class EngineQueueDepths:
    """引擎 FIFO 队列深度 (每核).

    注意: 不开相位拆分 (ModelOptions.pipeline=None) 时, 深度 >1 无行为
    差异 — 事件同时占独占资源 AIC/AIV0/AIV1, 引擎本就串行. 流水重叠
    需配合 PipelineConstraints 相位拆分 (load 相位不占核资源, 深度
    放开后 load 与 cube 才能跨 tile 重叠).
    """

    aic: int = 1
    vec0: int = 1
    aiv1: int = 1


# 纯物理基线: 五项尾段开销按字面取 0, 不回落到实测常数。
_ZERO_OVERHEADS = EpilogueOverheads(literal=True)


@dataclass(frozen=True)
class ModelOptions:
    combine_no_quant: bool = True
    topk_weights_prefetch: bool = False
    serialize_dispatch_comm: bool = False
    # C3 全核栅栏 (编排选择): () = 不加 (缺省, 逐核推进 = 融合算子);
    #   ("wave",)  波间全核对齐 —— 下一波的任何事件都等上一波全做完
    #   ("stage",) 波内每个 stage 之后对齐 —— 最彻底的分段式执行
    # 用来回答"融合 vs 分段", 见 builders/barriers.py。
    # 尾段固定开销. 缺省 = 全 0 (纯物理基线): 这五项是某一版实现的实测残留,
    # 模型不替任何实现预设它们。要那份实现的值就给 EpilogueOverheads()
    # (零值回落到模块常数 = 实测), profiles.MEGAMOE_A8W8 就是这么给的。
    epilogue_overheads: object = field(
        default_factory=lambda: _ZERO_OVERHEADS)   # config.hardware.EpilogueOverheads
    # C4 波宽: 每波装几个 m-group。0 = 由 p1/p2 经 calc_m_groups_per_wave 推导
    # (p1/p2 未给时取理论下限 1/1 —— kernel 那张按 token 数分档的表
    #  resolve_gmm1_min_logical_tiles_per_core 只留给复现工具, 模型不用)。
    # 给了正整数就**直接**当波宽用: 它是算子工程师要扫的决策变量, 不该只能经由
    # p1/p2 间接表达。
    m_groups_per_wave: int = 0
    # ACT -> GMM2 的交接编排 (C7). ACT 产出的是 GMM2 的 A (量化激活)。
    #
    #   "gm" (缺省) 物化: ACT 写回 GM, GMM2 的 A 从 GM 读回来。这是最少假设 ——
    #            不声称实现能把中间结果留在片上。某实现走这条
    #            (epilogue 写 workspaceInfo.activationQuantDataPtr,
    #             GMM2 从 Location::GM 取同一个指针)。
    #            代价: GMM2 每 tile 多付 A 流 m·K2 字节 (与 B 流取 max)。
    #            收益: tile->核 自由 —— 一个 m-group 的 18 个 ACT 可以散在 18 个核上,
    #            它的 40 个 GMM2 tile 也可以散开。
    #   "onchip" 不物化: A 留在片上 (硬件有 UB->L1 通路,
    #            Blaze::Gemm::Tile::CopyUB2L1Weight8Bit)。A 流不付 GM 字节,
    #            ACT 也不申报 GM 写。代价是**共位**: 一个 GMM2 tile 要吃该 m-group
    #            整个 K (= 全部 18 个 ACT), 所以这个 m-group 的 GMM1/ACT/GMM2 必须
    #            全落在同一个核上 -> 并行度上限 = m-group 数, 不是核数。
    #            需要 late_bind_pools 含 "AIC" (静态绑定下这个共位无从表达)。
    #
    # 不设第三种"跨核 K-split": 那要把 18 份 fp32 部分和落 GM 再归约
    # (每份 m·TILE_N·4B), 当前事件图没有归约事件 (见 docs 的 gap 1)。
    act_to_gmm2: str = "gm"
    # dispatch 的"谁取哪些行":
    #   "pooled" (缺省): 不按核预切, 只按 dispatch_rows_per_item 把切片切成若干份,
    #                    哪个核去取由调度器在派发时刻定。最少假设 —— 不预设分工。
    #   "precut":        建图时就把波的行按核分好 (均衡分配 + 轮转), 每核再切批。
    #                    一种具体实现的分工方式; 对齐实测 trace 要用它。
    # 见 builders/comm/mte.py。
    dispatch_partition: str = "pooled"
    # 一份 dispatch 工作覆盖多少行; 0 = 用 tiling 的 routeItemsPerBatch
    dispatch_rows_per_item: int = 0
    barriers: Tuple[str, ...] = ()
    pipeline: Optional[PipelineConstraints] = None
    gmm2_kl1: Optional[int] = None
    # GMM2 沿 K 维分几段独立就绪 (K = GMM1 的输出列 = ACT 的列范围)。
    #
    # GMM2 的 K 就是 GMM1 切分的那个 N 轴, 所以一个 ACT tile 只产出 GMM2 在 K 上
    # 1/ceil(k/TILE_N) 的部分; GMM2 要累完整个 K 才有结果。分几段就绪是**编排选择**,
    # 不是物理约束 —— L0C 本来就沿 kL1 分块累加 (block_mmad 的 ProcessTileL1), 所以
    # 每段只等覆盖自己 K 范围的 ACT 在物理上可行。
    #
    #   1 (缺省): 不分段 —— 一个 GMM2 tile 等齐覆盖整个 K 的全部 ACT 再开工。
    #       最少假设: 不声称实现能在只拿到部分 K 时就起步。
    #   2: 首个 kL1 块一段 (只等 1 个 ACT), 其余合成一段。k=4608/kl1=256 时第一段只
    #      占 1/18 = 5.6% 时长, 94.4% 仍等满 18 个 ACT。
    #   0: 每个 kL1 块各一段, 第 j 段只等覆盖第 j 块的 ACT —— 最细粒度。
    #   N>2: 均分成 N 段。
    #
    # 代价: 段数越多, flag 轮询次数越多 (kernel 侧每段一次 WaitUntilGmFlagEquals)。
    # 本模型不计这项开销, 所以细粒度的收益是**上界**。
    gmm2_k_segments: int = 1
    # L3 晚绑定: 哪些角色池的 tile->核 绑定推迟到**派发时刻**。
    #
    # 缺省 = 三个池全入 (派发时刻绑定)。理由是模型的不变量: 决不允许"某 tile 的前置
    # 依赖已完成、又有核空闲, 它却还在等" (analysis/idle.py 的 avoidable_idle_us)。
    # 静态绑定 (() ) 做不到这一点 —— 已就绪的 tile 会被困在忙核上而别的核空着;
    # 它是一种具体实现的分核方式, 要评估就显式给 ()。
    #
    # 取值: ("AIC",) / ("AIC", "AIV1") / ("AIC", "AIV0", "AIV1")
    # 共位约束自动加: ACT 跟随它的 GMM1 落核 (L0C->UB Fixpipe 只在绑定对内), 所以
    # "AIC" 入池隐含 "AIV0" 入池, 整对一起漂移。
    # combine **不**跟随它的 GMM2: GMM2 写 GM、combine 从 GM 读, 同核不是物理约束,
    # combine 可落任意空闲 AIV1。
    #
    # 残留保守项: 按核索引的 L1 回压边 (activation->gmm1, 由 gmm1_activation_depth
    # 产生) 在建图时按静态核号生成, 晚绑定后会指向别的核的 ACT。这不违反不变量
    # (那段空闲会计成 forced), 但墙钟会略微高估。该类边占总边数 1.4%~3.0%。
    late_bind_pools: Tuple[str, ...] = ("AIC", "AIV1")
    # 下一波 dispatch 的配速边 (等第 w-lookahead 波的 combine), 由算子工程师选:
    #   "none" (缺省) 不等 combine, 跨波连续 dispatch —— 依赖关系之外不加序边,
    #                 这是最少假设。
    #   "per_core"    等本核该波最后一个 combine (某实现的 AIV1 循环体: 先 combine
    #                 再下一波 dispatch)。与 AIV1 晚绑定同用时偏保守: 边指向原核号
    #                 的 combine, 那段等待计 forced。
    #   "wave"        等该波全部 combine
    dispatch_pacing: str = "none"
    engine_queue_depths: Optional[EngineQueueDepths] = None
