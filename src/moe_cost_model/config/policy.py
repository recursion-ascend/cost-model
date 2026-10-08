"""第 0 层: 实例层策略绑定 (运行时策略取值)."""
from dataclasses import dataclass




@dataclass(frozen=True)
class InstancePolicy:
    """实例层绑定 (运行时策略): 一个具体 kernel 实现的策略取值.

    三层定位: 理论层 (物理公式/依赖图, 不引用此处) → 策略层 (参数化函数)
    → 实例层 = 本绑定 + KernelConfig (编译期).
    """
    dispatch_lookahead: int = 2            # 迭代0预取 W0..W(la-1); 迭代 i 预取 W(i+la-1)
    gmm2_lag_threshold: int = 4096         # tokenNum ≥ 阈值 → GMM2 lag 一波
    gmm2_lag_waves: object = None          # 显式 GMM2 滞后波数 (None=按阈值两档缺省);
                                           # kernel 实例只用 0/1, 其他取值为模型参数空间的未验证点
    # (原 gmm1_activation_depth 已移到 ModelOptions.links:
    #  StageLink("gmm1","activation", location="onchip", depth=N) —— 它和"落哪、等多少"
    #  是同一组问题, 见 config/links.py)
    gmm2_combine_credit: object = None     # kernel 实例未启用 (None=关); 取值为模型参数空间的未验证点
    wave_offsets: object = None            # 显式 StageWaveOffsets; None=由 dispatch_lookahead
                                           # 与 gmm2_lag 推导 (kernel 实例语义, 缺省)
    cursor_resonance_fix: bool = True      # cursor 共振修正启用


    def effective_gmm2_lag(self, token_num: int) -> int:
        """GMM2 滞后波数: 显式参数优先, 缺省按 token 阈值两档 (0/1)."""
        if self.gmm2_lag_waves is not None:
            return max(0, int(self.gmm2_lag_waves))
        return 1 if token_num >= self.gmm2_lag_threshold else 0

    def effective_wave_offsets(self, token_num: int) -> "StageWaveOffsets":
        """stage 波偏移: 显式 wave_offsets 优先, 缺省由 la/lag 推导.

        缺省推导: dispatch = dispatch_lookahead - 1, gmm2 = -effective_gmm2_lag.
        与旧硬编码循环逐字节一致.
        """
        if self.wave_offsets is not None:
            return self.wave_offsets
        return StageWaveOffsets(dispatch=self.dispatch_lookahead - 1,
                                gmm2=-self.effective_gmm2_lag(token_num))

@dataclass(frozen=True)
class StageWaveOffsets:
    """MTE 编排循环的 stage 波偏移, 以 GMM1 的波为锚 (GMM1 偏移恒 0).

    dispatch: ≥ 0. 迭代 0 预取 W0..W(dispatch), 迭代 i 预取 W(i+dispatch).
              0 = 本波搬运, 1 = 超前一波 (对应 dispatch_lookahead=2).
    gmm2:     ≤ 0. 迭代 i 执行 W(i+gmm2) 的 GMM2. -1 = 滞后一波.
              > 0 无效: GMM2 的波超前于其 ACT 产出, 建图期就绪校验必失败.

    更高阶评估: 偏移组合映射到假设的 kernel 编排变体 (未验证取值).
    """

    dispatch: int = 1
    gmm2: int = 0

    def __post_init__(self):
        if self.dispatch < 0:
            raise ValueError("dispatch 偏移必须 ≥ 0 (建图期就绪标记先于 GMM1)")
        if self.gmm2 > 0:
            raise ValueError("gmm2 偏移必须 ≤ 0 (GMM2 不能超前于其 ACT 产出)")

