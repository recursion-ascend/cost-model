"""全部 stage 时长公式: dispatch + GMM1/GMM2 + ACT + COMBINE.

PrimitiveCosts 容器的全部延迟字段必填; 无回归拟合, 无零值默认.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .config.hardware import (
    ACT_BYTES_PER_VEC, BW_L1_GM, BW_LOCAL_GM, BW_REMOTE_GM, BW_SCATTER, BW_UB,
    T_CALL_OH, T_COUNT_GATE, T_GMM1_OVERLAP, T_LAT_LOCAL, T_LAT_REMOTE,
    T_STARTUP_VEC,
    MXFP_DIVISOR_SIZE, MXFP_MULTI_BASE_SIZE_K, URMA_FLAG_BYTES, URMA_FLAG_WINDOW_TOKENS,
    URMA_GET_BW_SINGLE, URMA_GET_LAT_US, URMA_PUT_BW_SINGLE, URMA_PUT_LAT_US,
    VEC_ELEM_FP32,
)
from .config.hardware import KernelConfig


# =====================================================================
# Dispatch: 数据布局 + 机制延迟
# =====================================================================

BYTES_ALIGN_QUANT = 256
BYTES_ALIGN_SCALE = 32


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@dataclass(frozen=True)
class DispatchDataLayout:
    route_items_per_batch: int = 256
    rev_token_elem_cnt: int = 6144   # = H, 1 byte/elem after E5M2 quant
    rev_scale_elem_cnt: int = 192    # = ceil(H/32)

    def bytes_read_per_row(self) -> int:
        return _align(self.rev_token_elem_cnt, BYTES_ALIGN_QUANT) + \
            _align(self.rev_scale_elem_cnt, BYTES_ALIGN_SCALE)

    @staticmethod
    def from_hidden(h: int) -> "DispatchDataLayout":
        return DispatchDataLayout(rev_token_elem_cnt=h, rev_scale_elem_cnt=(h + 31) // 32)


@dataclass
class DispatchMechanisticLatency:
    """Uncontended base service; contention via scheduler Channel rate server.

    Constants recalibrated from direct per-segment DISPATCH event timing
    (dispatch_transfer_raw.csv, B=64 real run) + MTE probe large-size fit.
    T_lat decomposes as: MTE startup (~78 cyc local / ~740 cyc remote
    incl. interconnect hop) + dispatch path overhead (~1687 cyc).
    """
    # 缺省值取 constants.py 单一事实源 (SourcedValue, 出处标签随值传播);
    # 数值与 2026-09 dispatch_transfer_raw.csv 逐段反解一致.
    t_call_oh_us: float = T_CALL_OH
    t_lat_local_us: float = T_LAT_LOCAL
    bw_local_bytes_per_us: float = BW_LOCAL_GM
    t_lat_remote_us: float = T_LAT_REMOTE
    bw_remote_bytes_per_us: float = BW_REMOTE_GM
    gmm1_overlap_us_per_call: float = T_GMM1_OVERLAP

    def call_base_us(self) -> float:
        return self.t_call_oh_us

    def segment_us(self, src: int, dst: int, rows: int,
                   layout: DispatchDataLayout = None) -> float:
        """单段基础服务 (无争用): λ_src + rows·b_row/BW_src."""
        if layout is None:
            layout = DispatchDataLayout()
        if src == dst:
            return self.t_lat_local_us + rows * self._bytes_row(layout) / self.bw_local_bytes_per_us
        return self.t_lat_remote_us + rows * self._bytes_row(layout) / self.bw_remote_bytes_per_us

    def _bytes_row(self, layout: DispatchDataLayout) -> int:
        return layout.bytes_read_per_row()


# =====================================================================
# URMA (Layered) 路径
# =====================================================================


@dataclass(frozen=True)
class LayeredDispatchLayout:
    """URMA Layered 二级接收的逐 token 记录布局.

    每 token GET = width_a (FP8 数据) + width_a_scale (e8m0 scale) 两条 WQE;
    width_a_scale = CeilDiv(h, 32) × MXFP_MULTI_BASE_SIZE(2) 个 e8m0 元素
    (kernel InitDispatchLayout / ReceiveRemoteDispatchBatch 的 widthA/widthAScale).
    本地 token 拷贝 = 读 win 记录 + 写 dispatchRevData/Scale (读+写双份流量).
    """

    width_a: int = 6144
    width_a_scale: int = 384

    @staticmethod
    def from_hidden(h: int) -> "LayeredDispatchLayout":
        return LayeredDispatchLayout(
            width_a=h,
            width_a_scale=(h + MXFP_DIVISOR_SIZE - 1) // MXFP_DIVISOR_SIZE * MXFP_MULTI_BASE_SIZE_K,
        )

    def get_bytes_per_token(self) -> int:
        return self.width_a + self.width_a_scale

    def local_copy_bytes_per_token(self) -> int:
        return 2 * self.get_bytes_per_token()


@dataclass
class UrmaMechanisticLatency:
    """URMA 批量 GET/PUT 机制延迟 (fab_arbitration_probe 实测标定).

    每批 = 一次 BatchCommit+Drain: λ + bytes / 单流 BW。
    并发纪律: 实测并行不降速 (3 流保持率 0.96)、无 FCFS 预留 → 事件间独立,
    不叠加速率服务器 (与 MTE fab 信道占位状态的处理一致, 且有 probe 证据)。
    flag 轮询: 每批一次 256-token 窗口的 ReadNbi+Drain (2048B)。
    """

    t_get_lat_us: float = URMA_GET_LAT_US
    bw_get_single_bytes_per_us: float = URMA_GET_BW_SINGLE
    t_put_lat_us: float = URMA_PUT_LAT_US
    bw_put_single_bytes_per_us: float = URMA_PUT_BW_SINGLE

    def get_batch_us(self, nbytes: float) -> float:
        return self.t_get_lat_us + nbytes / self.bw_get_single_bytes_per_us

    def put_batch_us(self, nbytes: float) -> float:
        return self.t_put_lat_us + nbytes / self.bw_put_single_bytes_per_us

    def flag_poll_us(self) -> float:
        window_bytes = int(URMA_FLAG_WINDOW_TOKENS) * int(URMA_FLAG_BYTES)
        return self.t_get_lat_us + window_bytes / self.bw_get_single_bytes_per_us


# =====================================================================
# 容器: 全部 stage 公式必填
# =====================================================================


@dataclass(frozen=True)
class PrimitiveCosts:
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

    # GMM1 流水线填充延迟 (残差实测): builder 按 tile 均摊到切片的每个 tile.
    # 原解析路径参数, 解析路径删除后归并到逐 tile 路径. 缺省 0 (零假设).
    gmm1_fill_us: float = 0.0

    # Explicit ready/ACK overheads if they are visible after calibration.
    dispatch_ready_publish_us: float = 0.0
    activation_ready_publish_us: float = 0.0
    combine_ack_us: float = 0.0

    # URMA Layered 路径机制延迟 (topo_urma=True 时使用; None → constants 默认值)
    urma_mechanistic: Optional[UrmaMechanisticLatency] = None


class AnalyticalGmmCosts:
    """GMM1/GMM2 的物理公式工厂.

    GMM1 每 tile:
        A 流 = m·K 字节 (FP8 激活, GM→L1)
        计算 = 2·m·cols·K MACs (SwiGLU 双投影)
        b=2: T = max(A流/BW, 计算/R_cube)        载入与计算重叠, 慢侧绑定
        b=1: T = A流/BW + 计算/R_cube + restart   串行相加

    GMM2 每 tile:
        计算 = m·cols·K2 MACs
        T = 计算/R_cube                          与 L1 缓冲数无关, 无 restart
        A 从 UB 直达 L0A, 片上带宽约 500 GB/s 且每核独占, 搬运时间忽略不计,
        所以 GMM2 只剩计算. restart 是 L1 换块的停顿, A 不经 L1 就没有这一项.

    B 流 (权重搬运) 两个 stage 都不建模: 认为被其他任务的执行掩盖.
    cube_mac_per_us 必填, 无缺省: 计算项是两个公式的主体, 而 Cube 速率没有
    标定常数, 给缺省值等于替调用方编一个数.
    """

    def __init__(self, bw_bytes_per_us: float = BW_L1_GM, *,
                 cube_mac_per_us: float,
                 l1_buf_num: int = 2,
                 tile_restart_us: float = 0.0, l1_tile_k: int = 256):
        if not cube_mac_per_us or cube_mac_per_us <= 0:
            raise ValueError(
                "cube_mac_per_us 必须为正 (Cube 计算速率, MAC/µs): "
                "GMM1 = max(A流, 计算), GMM2 = 纯计算, 无此速率无法计时")
        if l1_tile_k <= 0:
            raise ValueError("l1_tile_k must be positive")
        self.bw = bw_bytes_per_us
        self.serial = (l1_buf_num == 1)
        self.cube_rate = cube_mac_per_us
        self.chunk_restart = tile_restart_us
        self._k_l1 = int(l1_tile_k)

    def _chunks(self, k: int) -> int:
        return -(-k // self._k_l1)

    def gmm1_phases(self, m: int, k: int, cols: int) -> Tuple[float, float]:
        """GMM1 tile 的 (载入, 计算) 时长. 载入 = A 流, 单缓冲时含 restart.

        闭式时长由二者合成 (双缓冲取 max, 单缓冲相加); 相位流水按同一组数
        拆 load / cube 相位, 两边口径因此一致.
        """
        load = (m * k) / self.bw
        if self.serial:
            load += self._chunks(k) * self.chunk_restart
        return load, 2.0 * m * cols * k / self.cube_rate

    def gmm1_tile(self, m: int, k: int, cols: int) -> float:
        """m 行 × cols 列输出 tile: A 流载入与 SwiGLU 双投影计算."""
        load, compute = self.gmm1_phases(m, k, cols)
        return load + compute if self.serial else max(load, compute)

    def gmm2_tile(self, m: int, k2: int, cols: int) -> float:
        """GMM2: A 从 UB 直达 L0A 不计时, 时长 = 计算; 计算量 = m·cols·K2."""
        return m * cols * k2 / self.cube_rate


# PrimitiveCosts 


def gmm1_phase_split(costs: PrimitiveCosts, m: int, k: int,
                     cols: int) -> Optional[Tuple[float, float]]:
    """costs.gmm1_tile 的 (载入, 计算) 分解; 自定义 callable 无法分解, 返回 None."""
    owner = getattr(costs.gmm1_tile, "__self__", None)
    if isinstance(owner, AnalyticalGmmCosts):
        return owner.gmm1_phases(m, k, cols)
    return None


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
        (配对握手 gmmToEpilogueFlag)
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
    cube_mac_per_us: float,
    urma_mechanistic: Optional[UrmaMechanisticLatency] = None,
    kernel: KernelConfig = None,
    bw_l1_gm: Optional[float] = None,
    gmm1_fill_us: float = 0.0,
    gmm1_tile_restart_us: float = 0.0,
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

    带宽缺省 = constants 实测值; cube_mac_per_us (Cube 计算速率) 必填, 无缺省.
    """
    # 几何量 (tile_n/列数) 不进容器: 调用方按 KernelConfig 调用期传入.
    # 容器只承载硬件常数; kernel 参数中仅 L1 组织 (l1_buf_num/l1_tile_k) 影响公式形态.
    km = kernel if kernel is not None else KernelConfig()
    gmm = AnalyticalGmmCosts(
        bw_bytes_per_us=bw_l1_gm if bw_l1_gm is not None else BW_L1_GM,
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
        urma_mechanistic=urma_mechanistic,
        gmm1_tile=gmm.gmm1_tile,
        gmm2_tile=gmm.gmm2_tile,
        activation_tile=act.tile,
        combine_tile=comb.tile,
        count_table_prepare_us=count_table_prepare_us,
        gmm1_fill_us=gmm1_fill_us,
    )
