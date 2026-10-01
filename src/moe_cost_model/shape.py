"""形状/选项/游标: 模型的输入数据结构与建图工作状态中的轻量类型."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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


@dataclass(frozen=True)
class ModelOptions:
    combine_no_quant: bool = True
    topk_weights_prefetch: bool = False
    serialize_dispatch_comm: bool = False
    # 片间 fab 信道争用: 占位开关, 默认关。机制 (速率服务器) 已实现,
    # 但无可引用的标定证据 (bw_remote 常数从真实运行反解, 已含平均争用,
    # 叠加有双重计费风险) — 开启前需消融 benchmark, 见 README 信道 TODO。
    fabric_channels: bool = False
    pipeline: Optional[PipelineConstraints] = None
    gmm2_kl1: Optional[int] = None
    # GMM2 沿 K 维分几段独立就绪 (K = GMM1 的输出列 = ACT 的列范围)。
    #
    # GMM2 的 K 就是 GMM1 切分的那个 N 轴, 所以一个 ACT tile 只产出 GMM2 在 K 上
    # 1/ceil(k/TILE_N) 的部分; GMM2 要累完整个 K 才有结果。分几段就绪是**编排选择**,
    # 不是物理约束 —— L0C 本来就沿 kL1 分块累加 (block_mmad 的 ProcessTileL1), 所以
    # 每段只等覆盖自己 K 范围的 ACT 在物理上可行。
    #
    #   2 (缺省, = 现有 kernel): 首个 kL1 块一段 (只等 1 个 ACT), 其余合成一段 (等其余全部)。
    #       实测 k=4608/kl1=256 时, 第一段只占 1/18 = 5.6% 时长, 94.4% 仍等满 18 个 ACT。
    #   0: 每个 kL1 块各一段, 第 j 段只等覆盖第 j 块的 ACT —— 最细粒度。
    #   N>2: 均分成 N 段。
    #
    # 代价: 段数越多, flag 轮询次数越多 (kernel 侧每段一次 WaitUntilGmFlagEquals)。
    # 本模型不计这项开销, 所以细粒度的收益是**上界**。
    gmm2_k_segments: int = 2
    engine_queue_depths: Optional[EngineQueueDepths] = None
