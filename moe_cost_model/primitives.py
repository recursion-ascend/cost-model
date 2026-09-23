"""Cost primitives: 必填物理公式容器 + 三个解析公式实现.

PrimitiveCosts 五个延迟字段全部必填; 无回归拟合, 无零值默认.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .constants import (
    ACT_BYTES_PER_VEC, BW_L1_GM, BW_SCATTER, BW_UB, T_COUNT_GATE,
    T_STARTUP_VEC, TILE_N, VEC_ELEM_FP32,
)
from .dispatch import DispatchMechanisticLatency


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


class AnalyticalGmmCosts:
    """GMM1/GMM2 的纯物理公式工厂: tileN 用模块常量, SwiGLU 的 2 写死.

    GMM1 每 tile:
        T = (m*K + 2*K*tileN) / BW
        A: m 行 × K 列 (FP8 1B/元素)
        B: 2 × tileN 列 × K 行 (2 = SwiGLU gate+up 双投影, 算法结构常数)
        tileN: v6 模块常量 TILE_N = 256

    GMM2 每 tile:
        T = K2 * tileN / BW
        B: tileN 列 × K2 行 (K2 = intermediate, 单投影无双半)
    """

    def __init__(self, bw_bytes_per_us: float = BW_L1_GM, tile_n: int = TILE_N,
                 weight_nz: bool = False, bw_b_nz_bytes_per_us: float = 0.0,
                 l1_buf_num: int = 2, cube_mac_per_us: float = 0.0,
                 tile_restart_us: float = 0.0):
        """weight_nz=True 时 B 流 (权重 GM→L1) 用 NZ 分形路径带宽, A 流不变.

        NZ 带宽无标定数据时必须显式给出 (零猜测原则: Z 常数不通用,
        分形搬运的突发效率与线性流不同).
        """
        if weight_nz and bw_b_nz_bytes_per_us <= 0:
            raise ValueError(
                "weight_nz=True 需要 bw_b_nz_bytes_per_us (NZ 路径 GM→L1 实测带宽, "
                "标定法: NZ run 的 GMM1 tile 时长差分, 同 H 扫描)")
        self.bw = bw_bytes_per_us
        self.tile_n = tile_n
        self.bw_b = bw_b_nz_bytes_per_us if weight_nz else bw_bytes_per_us
        # b=1 (L1 无 ping-pong): 每 K-chunk 边界流水重启停顿 (chunk 级常数,
        # b1/b2 A/B 差分标定: 惩罚 ∝ tile 数, 计算串行项不显著)
        self.serial = (l1_buf_num == 1)
        self.cube_rate = cube_mac_per_us
        self.chunk_restart = tile_restart_us

    def _chunks(self, k: int) -> int:
        return -(-k // 256)   # kL1 基线 256; 自适应由 select_kl1 独立处理

    def gmm1_tile(self, m: int, k: int) -> float:
        t = (m * k) / self.bw + (2 * k * self.tile_n) / self.bw_b
        if self.serial:
            t += self._chunks(k) * self.chunk_restart
            if self.cube_rate > 0:
                t += 2.0 * m * self.tile_n * k / self.cube_rate
        return t

    def gmm2_tile(self, m: int, k2: int) -> float:
        # B 流主导 (k2×tile_n, 源码注释与标定域一致); NZ 时 B 流走分形带宽
        t = k2 * self.tile_n / self.bw_b
        if self.serial:
            t += self._chunks(k2) * self.chunk_restart
            if self.cube_rate > 0:
                t += m * self.tile_n * k2 / self.cube_rate
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

    def __init__(self, bw_ub_bytes_per_us=BW_UB, t_startup_us=T_STARTUP_VEC,
                 tile_n: int = TILE_N):
        self.bw_ub = bw_ub_bytes_per_us
        self.t_startup = t_startup_us
        self.tile_n = tile_n

    def tile(self, m: int, n_frac: float = 1.0) -> float:
        n_vec = m * self.tile_n * n_frac / self.VEC_ELEM
        return self.t_startup + n_vec * self.BYTES_PER_VEC / self.bw_ub


class AnalyticalCombineCosts:
    """COMBINE 物理公式: 接口匹配combine_tile(m_rows, n_frac).

    每 event (m 行 × n_frac×tileN 列, H 归约) 的数据流量:
        读 GMM2 输出: m × H × 2B (BF16, 顺序读)
        写回 token 位置: m × H × 2B (BF16, 散射写)
        元数据: m × 16B (token 位置 + topk weight)

    公式: T = (m × h × 4 + m × 16) × n_frac / BW_scatter
    """
    def __init__(self, h: int = 6144, bw_scatter_bytes_per_us: float = BW_SCATTER):
        self.h = h
        self.bw = bw_scatter_bytes_per_us

    def tile(self, m: int, n_frac: float = 1.0) -> float:
        data = m * (self.h * 4 + 16) * n_frac
        return data / self.bw
