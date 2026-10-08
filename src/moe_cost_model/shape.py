"""形状/选项/游标: 模型的输入数据结构与建图工作状态中的轻量类型."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config.hardware import EpilogueOverheads
from .config.granularity import (DEFAULT_GRANULARITIES, GranularityAssignment,
                                 resolve_granularity)
from .config.links import (DEFAULT_LINKS, StageLink, effective_gmm1_act_link,
                           resolve_link, validate_links)
from .config.roles import DEFAULT_ROLES, RoleAssignment
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


# 纯物理基线: 五项尾段开销按字面取 0, 不回落到实测常数。
_ZERO_OVERHEADS = EpilogueOverheads(literal=True)


@dataclass(frozen=True)
class ModelOptions:
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
    # dispatch 的"谁取哪些行":
    #   "pooled" (缺省): 不按核预切, 只按 dispatch_rows_per_item 把切片切成若干份,
    #                    哪个核去取由调度器在派发时刻定。最少假设 —— 不预设分工。
    #   "precut":        建图时就把波的行按核分好 (均衡分配 + 轮转), 每核再切批。
    #                    一种具体实现的分工方式; 对齐实测 trace 要用它。
    # 见 builders/comm/peerwrite.py。
    dispatch_partition: str = "pooled"
    # 一份 dispatch 工作覆盖多少行; 0 = 用 tiling 的 routeItemsPerBatch
    dispatch_rows_per_item: int = 0
    barriers: Tuple[str, ...] = ()
    pipeline: Optional[PipelineConstraints] = None
    gmm2_kl1: Optional[int] = None
    # stage 之间那条边: 消费者等多少 / 中间结果放哪 / 片上存几块。
    # 一条边一个 StageLink, 见 config/links.py。取代原先三个各自为政的参数
    # (gmm2_k_segments / act_to_gmm2 / InstancePolicy.gmm1_activation_depth)。
    links: Tuple[object, ...] = DEFAULT_LINKS     # config.links.StageLink
    # 哪个 stage 跑在哪个执行角色上 (AIC / AIV0 / AIV1), 见 config/roles.py。
    # 缺省: 矩阵乘上 Cube, ACT 占一个向量角色, 通信与 combine 占另一个。
    # 换角色是编排选择 —— 实测 A8W8 下两个向量核利用率都不到 6%, 而
    # GMM2 -> combine 的同核**不是**物理约束 (过 GM), 所以 combine 可以挪。
    roles: object = DEFAULT_ROLES                 # config.roles.RoleAssignment
    # 每个 stage 的**事件粒度** (一个事件覆盖多少份该 stage 的自然工作单元),
    # 见 config/granularity.py。与 links (每 stage 一条边)、roles (每 stage 一个角色)
    # 平行 —— 粒度是每个 stage 共有的维度, 不是 combine 的特性。
    # 缺省全 1 = 最细 = 最少假设 (不预设任何合并)。
    # 下面 combine_granularity / dispatch_rows_per_item 是它的**兼容视图**,
    # __post_init__ 会把两边对齐, 冲突直接报错 —— 只允许一个真相。
    granularity: object = DEFAULT_GRANULARITIES    # config.granularity.GranularityAssignment
    # combine 的**粒度**: 一个 combine 事件覆盖多少工作。角色由 roles 决定, 两者正交。
    #   "per_tile" (缺省) 与每个 GMM2 tile 1:1 配对, 紧跟其后 —— 延迟低, 但每个
    #       n-tile 都要把本窗 m 行的路由元数据读一遍 (读 n_tile 次)。
    #   "per_expert" 一个专家切片一个 combine 事件, 等该切片**全部** GMM2 段做完 ——
    #       合并: 元数据每行只读一次, 写侧落点跨度也更可控; 代价是等整片。
    # 两种都是合理编排; 参考实现把它与量化模板参数绑在一起, 那是它的耦合 (见 docs 缺口 11)。
    combine_granularity: str = "per_tile"
    # combine 写出的**落点布局**: 这 m 行写到目的卡窗口的哪里。
    #   "token_scatter" (缺省) 落点 = (tokenIdx·topK + topkIdx)·n, 由 token 全局编号决定。
    #       好处: UNPERMUTE 可以顺序读。代价: 写侧按 token 散射, 跨度 = token 数 x topk。
    #   "expert_contiguous" 按专家连续写 (跨度 = 本窗行数), UNPERMUTE 侧改成 gather。
    #       好处: 写侧局部性好。代价: 读侧变散。
    # 这笔交换**两侧都还没算全**:
    #   写侧代价要靠 AnalyticalCombineCosts 的 scatter_us_per_row, 它缺省 0 (只按字节算),
    #     所以缺省下换布局只改申报的跨度、不改时长;
    #   读侧代价 (UNPERMUTE 从顺序读变 gather) **完全没建模** —— UNPERMUTE 现在是
    #     字节量 / BW_UNPERMUTE_AGG 一个除法 (builders/base.py), 与落点布局无关。
    # 所以填了 scatter 系数之后 "expert_contiguous" 会显得单方面变好, 那是模型的偏置,
    # 不是结论。
    combine_layout: str = "token_scatter"
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
    # 残留保守项: 按核索引的 L1 回压边 (activation->gmm1, 由 gmm1->activation 的 depth
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "granularity", resolve_granularity(self.granularity))
        self._reconcile_granularity_views()
        # granularity 先定下来, 校验才报得出"该改的是 granularity, 现在是几"。
        # 用哪份 stage 词汇表校验边, 跟着 granularity 走 —— 它已经带着那份声明
        # (config/stages.py), 所以换算子不必在这里再加一个参数。
        validate_links(self.links, self.granularity,
                       vocab=self.granularity.vocab)

    def _reconcile_granularity_views(self) -> None:
        """把两个历史参数折进统一的 granularity, 保证只有一个真相.

        combine_granularity: "per_tile" <-> combine 粒度 1; "per_expert" <-> 0 (整片)。
        dispatch_rows_per_item: 就是 dispatch 的粒度 (0 = 用 tiling 的 routeItemsPerBatch)。

        谁赢: 只有一边离开缺省就那一边赢; 两边都离开缺省且矛盾就报错。
        """
        g: GranularityAssignment = self.granularity
        view = {"per_tile": 1, "per_expert": 0}.get(self.combine_granularity)
        if view is None:
            raise ValueError(
                'combine_granularity 只能是 "per_tile" / "per_expert", '
                f"收到 {self.combine_granularity!r}")
        have = g.items("combine")
        if have != view:
            if view == 1:                    # 视图是缺省 -> granularity 说话
                # have >= 2 (攒 N 个 tile) 两个字符串取值都表达不了 —— 视图只能
                # 近似记成 per_tile, 真相在 granularity 里。
                object.__setattr__(
                    self, "combine_granularity",
                    "per_expert" if have == 0 else "per_tile")
            elif have == 1:                  # granularity 是缺省 -> 视图说话
                object.__setattr__(self, "granularity", g.with_stage("combine", view))
            else:
                raise ValueError(
                    f"combine_granularity={self.combine_granularity!r} 与 "
                    f"granularity 里 combine 的 {have} 矛盾 —— 只给一个")
        g = self.granularity
        if g.items("dispatch") != self.dispatch_rows_per_item:
            if self.dispatch_rows_per_item:
                object.__setattr__(
                    self, "granularity",
                    g.with_stage("dispatch", self.dispatch_rows_per_item))
            else:
                object.__setattr__(
                    self, "dispatch_rows_per_item", g.items("dispatch"))

    def grain(self, stage: str) -> int:
        """该 stage 一个事件覆盖多少单元 (0 = 整个专家切片)."""
        return self.granularity.items(stage)

    def role_resource(self, stage: str, core: int) -> str:
        """该 stage 在 core 号核上的资源名 —— 建图器用它代替写死的 f-string."""
        return self.roles.resource(stage, core)

    def role_queue_token(self, stage: str, core: int) -> str:
        """该 stage 在 core 号核上的引擎队列令牌 —— 同样跟着角色走, 见 roles.queue_token."""
        return self.roles.queue_token(stage, core)

    def link(self, producer: str, consumer: str) -> StageLink:
        """取这条 stage 边的设置 (没给就回落到缺省)."""
        return resolve_link(self.links, producer, consumer)

    def gmm1_act_link(self, kernel) -> StageLink:
        """gmm1->activation 这条边在该编译点下的样子.

        单列出来是因为它不只取决于 links: TopkWeightsPrefetch 开着时 kernel 走 GM
        往返而不是 Fixpipe 直给, 那条边的 location/depth/同核性质都随之改变
        (见 config/links.effective_gmm1_act_link)。所有读这条边的地方都走这里,
        免得一半代码按 links 的说法、另一半按编译点的说法。
        """
        return effective_gmm1_act_link(self.links, kernel)
