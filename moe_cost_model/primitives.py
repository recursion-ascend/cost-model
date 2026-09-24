"""Cost primitives: 必填物理公式容器 + 三个解析公式实现.

PrimitiveCosts 五个延迟字段全部必填; 无回归拟合, 无零值默认.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .constants import (
    ACT_BYTES_PER_VEC, BW_L1_GM, BW_SCATTER, BW_UB, T_COUNT_GATE,
    T_STARTUP_VEC, VEC_ELEM_FP32,
)
from .dispatch import DispatchMechanisticLatency
from .constants import KernelConfig


@dataclass(frozen=True)
class PrimitiveCosts:
    # 全部 tile 延迟 callable 必填: 只接受物理公式实现 (AnalyticalGmmCosts /
    # AnalyticalActCosts / AnalyticalCombineCosts / DispatchMechanisticLatency),
    # 不提供回归拟合或零值默认, 防止静默回退到错误预测.
    dispatch_mechanistic: DispatchMechanisticLatency

    gmm1_tile: Callable[[int, int], float]
    gmm2_tile: Callable[[int, int], float]

    # x = valid M rows; y = logical scheduler columns / TILE_N.
    activation_tile: Callable[[int, float], float]
    combine_tile: Callable[[int, float], float]

    # One-time per physical AIV1 before MoE waves.  Optional because profiler may
    # already fold it into another stage fit.
    count_table_prepare_us: float = T_COUNT_GATE

    # 每个专家波内任务的每核首次 tile 的启动开销
    gmm1_problem_startup_us: float = 0.0
    gmm2_problem_startup_us: float = 0.0

    # GMM1 解析模型参数 (激活 B 复用路径; 不设则回退到 gmm1_tile callable)
    # gmm1_bw_bytes_per_us: GM→L1 载入带宽 (H 扫描差分实测)
    # gmm1_mac_per_us: AIC 立方计算速率 (m 差分实测)
    # gmm1_fill_us: 流水线填充延迟 (残差实测)
    gmm1_bw_bytes_per_us: Optional[float] = None
    gmm1_mac_per_us: Optional[float] = None
    gmm1_fill_us: float = 0.0

    # GMM2 解析模型参数 (不设则回退到 gmm2_tile callable)
    gmm2_bw_bytes_per_us: Optional[float] = None

    # NZ 权重布局: B 流 (权重) GM→L1 的 NZ 分形路径带宽; None = 走 bw (Z 标定域)
    gmm1_bw_b_nz_bytes_per_us: Optional[float] = None

    # Explicit ready/ACK overheads if they are visible after calibration.
    dispatch_ready_publish_us: float = 0.0
    activation_ready_publish_us: float = 0.0
    combine_ack_us: float = 0.0

    # L1 无 ping-pong (b=1) 时每 K-chunk 边界的流水重启停顿, 仅供 model.py
    # 解析路径使用 (旧实现经 getattr 取值恒为 0, 属死代码, 现转正为字段).
    # callable 路径由 AnalyticalGmmCosts.tile_restart_us 承载, 二者标定方法一致
    # (b1/b2 A/B 差分, 惩罚 ∝ K-chunk 数 × tile 数).
    gmm1_tile_restart_us: float = 0.0


class AnalyticalGmmCosts:
    """GMM1/GMM2 的纯物理公式工厂: tileN 用模块常量, SwiGLU 的 2 写死.

    GMM1 每 tile:
        T = (m*K + 2*K*tileN) / BW
        A: m 行 × K 列 (FP8 1B/元素)
        B: 2 × cols 列 × K 行 (2 = SwiGLU gate+up 双投影, 算法结构常数)
        cols: 每窗实际列数, 调用期传入 (= KernelConfig.tile_n, 尾 N-tile 时更小)

    GMM2 每 tile:
        T = K2 * cols / BW
        B: cols 列 × K2 行 (K2 = intermediate, 单投影无双半)
    """

    def __init__(self, bw_bytes_per_us: float = BW_L1_GM,
                 weight_nz: bool = False, bw_b_nz_bytes_per_us: float = 0.0,
                 l1_buf_num: int = 2, cube_mac_per_us: float = 0.0,
                 tile_restart_us: float = 0.0, l1_tile_k: int = 256):
        """weight_nz=True 时 B 流 (权重 GM→L1) 用 NZ 分形路径带宽, A 流不变.

        NZ 带宽无标定数据时必须显式给出 (零猜测原则: Z 常数不通用,
        分形搬运的突发效率与线性流不同).
        l1_tile_k: K-chunk 基线, 必须与 KernelConfig.l1_tile_k 一致
        (串行路径的 chunk 重启次数由它决定).
        """
        if weight_nz and bw_b_nz_bytes_per_us <= 0:
            raise ValueError(
                "weight_nz=True 需要 bw_b_nz_bytes_per_us (NZ 路径 GM→L1 实测带宽, "
                "标定法: NZ run 的 GMM1 tile 时长差分, 同 H 扫描)")
        if l1_tile_k <= 0:
            raise ValueError("l1_tile_k must be positive")
        self.bw = bw_bytes_per_us
        self.bw_b = bw_b_nz_bytes_per_us if weight_nz else bw_bytes_per_us
        # b=1 (L1 无 ping-pong): 每 K-chunk 边界流水重启停顿 (chunk 级常数,
        # b1/b2 A/B 差分标定: 惩罚 ∝ tile 数, 计算串行项不显著)
        self.serial = (l1_buf_num == 1)
        self.cube_rate = cube_mac_per_us
        self.chunk_restart = tile_restart_us
        self._k_l1 = int(l1_tile_k)

    def _chunks(self, k: int) -> int:
        # K-chunk 数 = ceil(k / kL1), kL1 来自 KernelConfig.l1_tile_k (不再硬编码)
        return -(-k // self._k_l1)

    def gmm1_tile(self, m: int, k: int, cols: int) -> float:
        """m 行 × cols 列输出 tile: A 流 m·k (FP8) + B 流 2·k·cols (SwiGLU 双投影)."""
        t = (m * k) / self.bw + (2 * k * cols) / self.bw_b
        if self.serial:
            t += self._chunks(k) * self.chunk_restart
            if self.cube_rate > 0:
                t += 2.0 * m * cols * k / self.cube_rate
        return t

    def gmm2_tile(self, m: int, k2: int, cols: int) -> float:
        """B 流主导 (k2×cols, 源码注释与标定域一致); NZ 时 B 流走分形带宽."""
        t = k2 * cols / self.bw_b
        if self.serial:
            t += self._chunks(k2) * self.chunk_restart
            if self.cube_rate > 0:
                t += m * cols * k2 / self.cube_rate
        return t


# PrimitiveCosts 


class AnalyticalActCosts:
    """ACT (SwiGLU + MX量化) 的物理公式.

    每 tile (m行 × tileN列) 的向量操作数:
        n_vec = m * tileN / VEC_ELEM  (VEC_ELEM = 64 FP32/向量)

    每向量的 UB 流量 (从源码精确计数):
        读: gate(BF16) 128B + up(BF16) 128B + bf16中间重读 128B = 384B
        写: bf16中间 128B + fp8 64B + scale 4B = 196B
        总: 580B/向量

    公式:
        T = T_startup + n_vec * 580 / BW_ub

    硬件参数 (单点测量, 非拟合):
        BW_ub: UB 读+写饱和带宽 (从大 m tile 一次实测)
        T_startup: 向量流水启动+排空 (从空/极小 tile 一次实测)
    """
    VEC_ELEM = VEC_ELEM_FP32
    BYTES_PER_VEC = ACT_BYTES_PER_VEC

    def __init__(self, bw_ub_bytes_per_us=BW_UB, t_startup_us=T_STARTUP_VEC):
        # 几何量 (每窗列数) 调用期传入 — 同 GMM, 消除 tile_n 双份来源
        self.bw_ub = bw_ub_bytes_per_us
        self.t_startup = t_startup_us

    def tile(self, m: int, cols: int) -> float:
        n_vec = m * cols / self.VEC_ELEM
        return self.t_startup + n_vec * self.BYTES_PER_VEC / self.bw_ub


COMBINE_NO_QUANT = 0
COMBINE_QUANT = 1


class AnalyticalCombineCosts:
    """COMBINE (AIV1 配对消费 GMM2 tile) 物理公式, 按量化模式参数化.

    每 (m-group, N-tile) 窗 (m 行 × logical_n 列) 的 GM 流量 = m×(e·logical_n + meta):
      读侧 e_in: GMM2 输出 tile 经 workspace 读回 — 两种模式均 BF16 = 2B/元素
        (配对握手 gmmToEpilogueFlag, mega_moe_arch35.h RunGmm2CombineForExpert)
      写侧 e_out: per-slot 部分和 (topk 归约在 UNPERMUTE, 非累加 rmw):
        NO_QUANT: BF16 = 2B/元素
        QUANT:    FP8 = 1B/元素 + MX scale 1B/32元素 = 1/32 B/元素
      meta: 8B/行 = token 位置 int32 4B (metaInfo, constants.h INT32_PER_256B=8)
                 + topk 权重 fp32 4B (probsGm)
    → e = e_in + e_out + scale: NO_QUANT 推导值 4 (= 2读+2写); QUANT 推导值 3.03125
    公式: T = m × (e×logical_n + meta) / BW_scatter
    (旧公式 m×(4h+16) 的两处错误: 列数误用全 H ×24; meta 16B 无源码依据 → 8B)
    待声明未建模: 散射写凸型 m 依赖 (超线性, 见 2026-09 对比记录)。
    """
    META_BYTES_PER_ROW = 8

    def __init__(self, combine_quant_mode: int = COMBINE_NO_QUANT,
                 bw_scatter_bytes_per_us: float = BW_SCATTER,
                 meta_bytes_per_row: int = META_BYTES_PER_ROW):
        if combine_quant_mode == COMBINE_NO_QUANT:
            self.in_elem_bytes = 2.0        # GMM2 输出 BF16
            self.out_elem_bytes = 2.0       # 部分和 BF16
            self.scale_bytes_per_elem = 0.0
        elif combine_quant_mode == COMBINE_QUANT:
            self.in_elem_bytes = 2.0        # GMM2 输出 BF16 (UB 内量化)
            self.out_elem_bytes = 1.0       # 部分和 FP8
            self.scale_bytes_per_elem = 1.0 / 32.0   # MX scale 每 32 元素 1B
        else:
            raise ValueError(f"未知 combine_quant_mode: {combine_quant_mode}")
        self.combine_quant_mode = combine_quant_mode
        self.bw = bw_scatter_bytes_per_us
        self.meta_bytes = meta_bytes_per_row

    @property
    def per_elem_bytes(self) -> float:
        """推导的每元素流量系数: NO_QUANT=4 (2读+2写); QUANT=3.03125 (2读+1写+1/32 scale)."""
        return self.in_elem_bytes + self.out_elem_bytes + self.scale_bytes_per_elem

    def tile(self, m: int, logical_n: int = 256) -> float:
        # logical_n = 本窗实际列数 (尾 N-tile 时 < 256), 调用期传入
        data = m * (self.per_elem_bytes * logical_n + self.meta_bytes)
        return data / self.bw


def build_analytical_costs(
    *,
    h: int,
    dispatch_mechanistic: DispatchMechanisticLatency,
    kernel: KernelConfig = None,
    bw_l1_gm: Optional[float] = None,
    bw_l1_gm_b_nz: float = 0.0,
    cube_mac_per_us: float = 0.0,
    gmm1_fill_us: float = 0.0,
    gmm1_tile_restart_us: float = 0.0,
    gmm2_bw_bytes_per_us: Optional[float] = None,
    bw_ub: Optional[float] = None,
    t_startup_us: Optional[float] = None,
    bw_scatter: Optional[float] = None,
    count_table_prepare_us: float = T_COUNT_GATE,
) -> PrimitiveCosts:
    """按 (h, KernelConfig) 一致构建解析公式族, 消除 tile_n/l1_tile_k/h 漏配.

    修复的漏配类: 手工构造 Analytical* 用默认 tile_n=256 / h=6144, 而
    KernelConfig.tile_n / shape.h 被覆盖 → n_frac 换算与字节计数静默出错.
    几何量 (tile_m/tile_n/每窗列数) 属于 KernelConfig, 调用期由模型传入公式,
    容器只承载硬件常数与 L1 组织参数 — 结构上消除 tile_n/h 双份来源的漏配.

    带宽缺省 = constants 实测值; NZ 布局必须显式给 bw_l1_gm_b_nz (零猜测).
    """
    # 几何量 (tile_n/列数) 不进容器: 调用方按 KernelConfig 调用期传入.
    # 容器只承载硬件常数; kernel 参数中仅 L1 组织 (l1_buf_num/l1_tile_k/weight_nz) 影响公式形态.
    km = kernel if kernel is not None else KernelConfig()
    gmm = AnalyticalGmmCosts(
        bw_bytes_per_us=bw_l1_gm if bw_l1_gm is not None else BW_L1_GM,
        weight_nz=km.weight_nz,
        bw_b_nz_bytes_per_us=bw_l1_gm_b_nz,
        l1_buf_num=km.l1_buf_num,
        cube_mac_per_us=cube_mac_per_us,
        tile_restart_us=gmm1_tile_restart_us,
        l1_tile_k=km.l1_tile_k,
    )
    act = AnalyticalActCosts(
        bw_ub_bytes_per_us=bw_ub if bw_ub is not None else BW_UB,
        t_startup_us=t_startup_us if t_startup_us is not None else T_STARTUP_VEC,
    )
    comb = AnalyticalCombineCosts(
        combine_quant_mode=km.combine_quant_mode,
        bw_scatter_bytes_per_us=bw_scatter if bw_scatter is not None else BW_SCATTER,
    )
    return PrimitiveCosts(
        dispatch_mechanistic=dispatch_mechanistic,
        gmm1_tile=gmm.gmm1_tile,
        gmm2_tile=gmm.gmm2_tile,
        activation_tile=act.tile,
        combine_tile=comb.tile,
        count_table_prepare_us=count_table_prepare_us,
        gmm1_bw_bytes_per_us=bw_l1_gm,
        gmm1_mac_per_us=(cube_mac_per_us if cube_mac_per_us > 0 else None),
        gmm1_fill_us=gmm1_fill_us,
        gmm2_bw_bytes_per_us=gmm2_bw_bytes_per_us,
        gmm1_bw_b_nz_bytes_per_us=(bw_l1_gm_b_nz if (km.weight_nz and bw_l1_gm_b_nz > 0) else None),
        gmm1_tile_restart_us=gmm1_tile_restart_us,
    )
