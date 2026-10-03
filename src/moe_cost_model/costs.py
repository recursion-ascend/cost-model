"""全部 stage 时长公式: dispatch + GMM1/GMM2 + ACT + COMBINE.

PrimitiveCosts 容器的全部延迟字段必填; 无回归拟合, 无零值默认.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .config.hardware import (
    ACT_BYTES_PER_VEC, BW_L1_GM, BW_LOCAL_GM, BW_REMOTE_GM, BW_REMOTE_WRITE,
    BW_UB, DISPATCH_BUFFER_COUNT,
    T_COUNT_GATE, T_GMM1_OVERLAP, T_LAT_LOCAL, T_LAT_REMOTE,
    T_STARTUP_VEC,
    MXFP_DIVISOR_SIZE, MXFP_MULTI_BASE_SIZE, MXFP_MULTI_BASE_SIZE_K,
    URMA_FLAG_BYTES, URMA_FLAG_WINDOW_TOKENS,
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
    #: 内核 dispatchBufferConfig.routeItemsPerBatch (tiling 真值 256).
    #: DispatchRankTokens 把一个 (核,专家,源卡) 段再按它分批, 每批一次
    #: CopyTokensAndMetaForDispatch —— 而 MOE_PROFILE_BEGIN/END 就在那个函数里,
    #: 所以**一批就是一个 trace 事件**, 各付自己的 λ 与流水填充/排空。
    route_items_per_batch: int = 256
    rev_token_elem_cnt: int = 6144   # = H, 1 byte/elem after E5M2 quant
    rev_scale_elem_cnt: int = 192    # = ceil(H/32)

    #: metaInfo 每行 INT32_PER_256B(8) x int32 = 32B (内核 StoreDispatch... 的第三次搬运)
    META_BYTES_PER_ROW = 32

    def bytes_read_per_row(self) -> int:
        """Fetch 一次 DataCopy 的字节 = context.quantTokenScaleAlignBytes.

        内核 FetchDispatchTokenAndMetaInfo 从源卡窗口读这么多进 UB; 也正是 trace
        payload 里 [23:12] 那个 rowBytes/32 的来源。
        """
        return _align(self.rev_token_elem_cnt, BYTES_ALIGN_QUANT) + \
            _align(self.rev_scale_elem_cnt, BYTES_ALIGN_SCALE)

    def bytes_written_per_row(self) -> int:
        """Store 三次搬运的字节, 全部落在**本卡** workspace.

        内核 StoreDispatchTokenAndMetaInfo: DataCopyPad(token) +
        DataCopyPad(scale) + DataCopy(metaInfo)。前两个是未对齐的精确元素数
        (revTokenElemCnt = H/A_ELEMS_PER_BYTE, revScaleElemCnt =
        CeilDiv(H, MXFP_DIVISOR_SIZE) x MXFP_MULTI_BASE_SIZE), 都是 1B/元素。
        """
        return self.rev_token_elem_cnt + self.rev_scale_elem_cnt + self.META_BYTES_PER_ROW

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
    # 每波每核 dispatch 调用的固定开销。缺省 0: 由算子工程师按自己的实现填
    # (实测参考值 T_CALL_OH = 1.006us, 见 config/hardware.py)。
    t_call_oh_us: float = 0.0
    t_lat_local_us: float = T_LAT_LOCAL
    bw_local_bytes_per_us: float = BW_LOCAL_GM
    t_lat_remote_us: float = T_LAT_REMOTE
    bw_remote_bytes_per_us: float = BW_REMOTE_GM
    gmm1_overlap_us_per_call: float = T_GMM1_OVERLAP
    # 行级软流水槽数 (tiling dispatchBufferCount); 槽内的行重叠, 不串行累加
    buffer_count: int = DISPATCH_BUFFER_COUNT

    def call_base_us(self) -> float:
        return self.t_call_oh_us

    def segment_us(self, src: int, dst: int, rows: int,
                   layout: DispatchDataLayout = None) -> float:
        """单段基础服务 (无争用), 按内核的行级软流水算.

        CopyTokensAndMetaForDispatch 用 buffer_count 个槽做 Fetch/Store 双段流水:
        前 buffer_count 行的 Fetch 背靠背发出 (Wait=false), 只有 issueIdx >=
        buffer_count 才等 MTE3_MTE2 腾槽。于是
          rows <= buffer_count: 行在流水里重叠, 段时长 = max(一次往返 λ, 字节时间)
          rows >  buffer_count: 多出来的行按每行稳态节拍串上去
        字节仍要过互连, 但并发争用由片间信道的速率服务器裁决 —— 不折进这里
        (这个容器的契约就是"无争用基础服务")。

        旧式 λ + rows·b_row/BW 把重叠的行也串行计了, 对 20260930 run 的 dispatch
        内层高估 32%: 实测 1~3 行是平台 (远端 2.17-2.42us, 与 T_LAT_REMOTE=2.43
        吻合), 不是线性上升。同一份数据若直接按行数拟合, 会把争用当成每行常数。
        """
        if layout is None:
            layout = DispatchDataLayout()
        row_us = self._bytes_row(layout) / (self.bw_local_bytes_per_us if src == dst
                                            else self.bw_remote_bytes_per_us)
        lat = self.t_lat_local_us if src == dst else self.t_lat_remote_us
        depth = max(1, int(self.buffer_count))
        overlapped = min(max(rows, 0), depth)
        serial = max(0, rows - depth)
        return max(lat, overlapped * row_us) + serial * row_us

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
    # 本窗写出到 GM 的字节 (fp8 + MX scale), 供 builder 折算写信道流量
    activation_store_bytes: Callable[[int, float], float]
    # (m 行, 本窗列数, 其中目的卡 != 本卡的行数) -> us
    # 第三参必填: 跨卡行是 COMBINE 的主导项 (实测占 95%), 缺了它公式会低估 10 倍.
    combine_tile: Callable[[int, float, int], float]
    # 本窗每行写出的字节数, 供 builder 折算片间信道流量.
    # 必填 (不给零值缺省): 缺省 0 会让 COMBINE 的跨卡写悄悄不占片间资源 ——
    # 手工构造 PrimitiveCosts 的调用点会与 build_analytical_costs 静默分叉。
    combine_write_bytes_per_row: Callable[[float], float]

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
        载入 = A 流 m·K (FP8 激活) + B 流 2·K·cols (SwiGLU gate/up 两块权重)
        计算 = 2·m·cols·K MACs
        b=2: T = max(载入/BW, 计算/R_cube)     载入与计算重叠, 慢侧绑定
        b=1: T = 载入/BW + 计算/R + restart     串行相加

    GMM2 每 tile:
        载入 = max(A 流, B 流), 口径同 GMM1
            B 流 = K2·cols (权重 GM→L1)
            A 流 = m·K2 —— **只在 gmm2_a_from_gm=True (物化编排) 下计**:
                ACT 把结果写回 GM, GMM2 的 A 再从 GM 读回来。
                gmm2_a_from_gm=False = 不物化编排: A 留在片上 (UB→L1),
                A 流不付 GM 字节。这不是硬件常数, 是编排选择
                (ModelOptions.act_to_gmm2), 两种编排的 DAG 约束不同,
                由 model 层统一摆平。
        计算 = m·cols·K2 MACs
        b=2: T = max(载入/BW, 计算/R_cube)
        b=1: T = 载入/BW + 计算/R + restart

    **B 流是小 batch 下的主导项**: bs=36 实测 (20260930 run) 每 tile
    GMM1 56.9 µs / GMM2 23.9 µs, 与 B 流载入时间 (50.5 / 22.7) 同量级;
    反解 Cube 速率只有 3.3e6 MAC/µs, 远低于硬件能力 —— 说明两个 stage 都是
    权重载入绑定, 不是计算绑定。每个专家只有 72 行而一个 tile 的权重是
    256 列 × K, 权重流量是激活的 7 倍, 没有东西可以掩盖它。

    cube_mac_per_us = 0 时不计计算项 (实测域内计算远小于载入; 大 m 时应给出).
    """

    def __init__(self, bw_bytes_per_us: float = BW_L1_GM,
                 weight_nz: bool = False, bw_b_nz_bytes_per_us: float = 0.0,
                 l1_buf_num: int = 2, cube_mac_per_us: float = 0.0,
                 gmm1_weight_blocks: int = 2,
                 tile_restart_us: float = 0.0, l1_tile_k: int = 256,
                 gmm2_a_from_gm: bool = True):
        if weight_nz and bw_b_nz_bytes_per_us <= 0:
            raise ValueError(
                "weight_nz=True 需要 bw_b_nz_bytes_per_us (NZ 路径 GM→L1 实测带宽)")
        if l1_tile_k <= 0:
            raise ValueError("l1_tile_k must be positive")
        self.bw = bw_bytes_per_us
        self.weight_nz = bool(weight_nz)
        self.bw_b = bw_b_nz_bytes_per_us if weight_nz else bw_bytes_per_us
        self.serial = (l1_buf_num == 1)
        # 一个 GMM1 tile 要载入几个权重块: 非交织 = ACTIVATION_N_HALF (gate+up 两遍
        # mmad), 交织 = 1 (gate/up 在 tile 内按列交织, 一遍 mmad)。
        # 两种口径下整层 B 流总量相同 (18 tile x 2 == 36 tile x 1)。
        self.wb = int(gmm1_weight_blocks)
        if self.wb <= 0:
            raise ValueError("gmm1_weight_blocks must be positive")
        self.cube_rate = cube_mac_per_us
        self.chunk_restart = tile_restart_us
        # GMM2 的 A 是否要从 GM 读回 (物化编排); 见类 docstring 与
        # ModelOptions.act_to_gmm2。
        self.gmm2_a_from_gm = bool(gmm2_a_from_gm)
        self._k_l1 = int(l1_tile_k)

    def _chunks(self, k: int) -> int:
        return -(-k // self._k_l1)

    def gmm1_phases(self, m: int, k: int, cols: int,
                    b_load: bool = True) -> Tuple[float, float]:
        """GMM1 tile 的 (载入, 计算) 时长; 单缓冲时载入含 restart.

        载入 = max(A流, B流) —— 两股并发, 搬运事件取较慢的那一股。
        结果写出 (L0C -> GM/UB 的 Fixpipe) 不计入: 按"数据释放事件忽略不计"的口径。

        口径沿革 (这一项反复过两次, 都记下来):
          2026-09-30(1) 先用 bs36 (m=72) 与 bs8192 (m=256) 两个 run 比, 看到实测单
            tile 55.0 / 53.8 us 几乎不随 m 变, 于是用了 max(A,B)。当时判断那是混淆
            变量: bs8192 同时变了 m (72->256) 与每专家 m-group 数 (1->12)。
          2026-09-30(2) 改回相加。依据是 bs128 (m=256 但仍 1 个 m-group) 把两者分开:
            bs36   m= 72, 1 组: 实测 55.645   A+B = 57.61 (+3.5%)   max = 50.51 ( -9.2%)
            bs128  m=256, 1 组: 实测 74.810   A+B = 75.76 (+1.3%)   max = 50.51 (-32.5%)
            bs8192 m=256,12 组: 实测 53.810   A+B = 75.76 (+40.8%)  max = 50.51 ( -6.1%)
          2026-10-03 按口径决定改为 max。**与上面两个单 m-group 实测点冲突**:
            B 流恒大于 A 流时 max 口径的 tile 时长完全不随 m 变, 而 bs36->bs128 是
            干净的单变量对比 (只有 m 变), 实测从 55.645 升到 74.810, 斜率
            0.10416 us/行。max 预测 0 斜率。这个冲突没有消解, 是已知的建模取舍:
            采用 max 口径就意味着在 1 个 m-group 的形状上低估 9%~32%。
            反过来 max 口径在 12 个 m-group 的 bs8192 上只差 -6.1% (相加口径 +40.8%),
            而 bs8192 的偏差本来要靠 gmm1_b_reuse 解释 —— 换 max 后那个待定的
            "B 流付几次" 规律不再是解释 bs8192 的必要条件。
        """
        a_load = (m * k) / self.bw
        b_load_us = (self.wb * k * cols) / self.bw_b if b_load else 0.0
        # 搬运口径: A 流与 B 流并发, 事件时长取较慢的一股。
        load = max(a_load, b_load_us)
        if self.serial:
            load += self._chunks(k) * self.chunk_restart
        compute = (2.0 * m * cols * k / self.cube_rate) if self.cube_rate > 0 else 0.0
        return load, compute

    def gmm1_tile(self, m: int, k: int, cols: int, b_load: bool = True) -> float:
        """m 行 × cols 列输出 tile.

        b_load: 是否计 B 流 (权重) 的 GM→L1 字节. B 复用模式下切片内只有首个
        m-group 的 tile 付自己列块的 B, 后续 m-group 假设命中 L2。
        """
        load, compute = self.gmm1_phases(m, k, cols, b_load)
        return load + compute if self.serial else max(load, compute)

    def gmm2_phases(self, m: int, k2: int, cols: int) -> Tuple[float, float]:
        """GMM2 tile 的 (载入, 计算) 时长.

        载入 = max(A 流, B 流): B 流是权重 GM→L1; A 流只在物化编排下存在
        (ACT 写 GM, GMM2 读回), 不物化时为 0。
        """
        b_load = k2 * cols / self.bw_b
        a_load = (m * k2) / self.bw if self.gmm2_a_from_gm else 0.0
        load = max(a_load, b_load)
        if self.serial:
            load += self._chunks(k2) * self.chunk_restart
        compute = (m * cols * k2 / self.cube_rate) if self.cube_rate > 0 else 0.0
        return load, compute

    def gmm2_tile(self, m: int, k2: int, cols: int) -> float:
        """GMM2: 载入 = max(A 流, B 流) (A 流见 gmm2_phases); 计算量 = m·cols·K2."""
        load, compute = self.gmm2_phases(m, k2, cols)
        return load + compute if self.serial else max(load, compute)


# PrimitiveCosts 


def gmm1_phase_split(costs: PrimitiveCosts, m: int, k: int, cols: int,
                     b_load: bool = True) -> Optional[Tuple[float, float]]:
    """costs.gmm1_tile 的 (载入, 计算) 分解; 自定义 callable 无法分解, 返回 None."""
    owner = getattr(costs.gmm1_tile, "__self__", None)
    if isinstance(owner, AnalyticalGmmCosts):
        return owner.gmm1_phases(m, k, cols, b_load)
    return None


def gmm2_phase_split(costs: PrimitiveCosts, m: int, k2: int,
                     cols: int) -> Optional[Tuple[float, float]]:
    """costs.gmm2_tile 的 (载入, 计算) 分解; 自定义 callable 返回 None."""
    owner = getattr(costs.gmm2_tile, "__self__", None)
    if isinstance(owner, AnalyticalGmmCosts):
        return owner.gmm2_phases(m, k2, cols)
    return None


class AnalyticalActCosts:
    """ACT (SwiGLU + MX量化) 的物理公式.

    profile 区间 = 内核 Gmm1Aiv0EpilogueTileGeneric 一次调用 (stage/
    mega_moe_gmm1_activation.h 的 MOE_PROFILE_BEGIN(ACT_QUANT)), 纯计算+写出:
    WaitForCube 在前一个 WAIT_ACT_INPUT 区间里, NotifyCube 在区间外。
    非 prefetch 路径下输入已由 AIC 落在 UB, 区间内**没有 GM→UB 读**。

    每 tile (m行 × tileN列) 的向量操作数:
        n_vec = m * tileN / VEC_ELEM  (VEC_ELEM = 64 FP32/向量)

    公式:
        T = T_startup + n_vec * BYTES_PER_VEC / BW_ub

    ---- 已知的三处偏差 (2026-09-30 对 20260930 run 审计; 都没改, 理由见下) ----

    1. BYTES_PER_VEC = 580 与源码不符, 但**单改它是变相拟合**。
       源码真值 (blaze/epilogue/block_epilogue_activation_mx_quant.h): bf16 中间
       缓冲被**整体流三遍** —— SwiGLU 写一遍, ComputeMaxExp 读一遍,
       ComputeFp8Data 再读一遍。模型只算了一次重读, 漏了 128B/向量; 另有
       maxExp/inverseMxScale 各 uint16 的往返约 14B。UB 侧真值 722B/向量, 其中
       66B (fp8 64 + scale 2) 其实是 GM 流量而非 UB。
       但 (T_startup, BW_ub) 当初是**用 580 在两个 ACT 点上联合拟合**的 (截距取
       小 m tile, 斜率取大 m tile), 所以只有比值 BYTES_PER_VEC/BW_ub 可观测:
       把字节改成 722 再同两点重拟合会得到 BW_ub = 93000x722/580 = 115769,
       预测**逐位不变**。故字节数的错是"结构上错、数值上惰性", 单改它没有信息,
       要分开只能扫 m 或扫 tileN (本 run 全部 54 个 tile 形状相同, 零信息)。

    2. 写出是 GM 而非 UB, 且代价随 m 走而不是随 m*cols 走 (待建模)。
       StoreQuantOutput 发 blockCount = m 次、每次 cols*1B(fp8) 的带 stride 突发;
       StoreQuantScaleCompact 发 m 次、每次 ceil(cols/MXFP_DIVISOR)*MXFP_MULTI_BASE
       = 8B 的突发。8B 远低于任何 GM 突发粒度, 那一路的代价由 m 次请求发射决定,
       不由 576 字节决定。现行公式把两者都按 m*cols 记在 BW_ub 上。
       正确形态多一项 m*(每行发射代价), 它让 T 依赖 tile 的行列长宽比 —— 本配置
       m 恒为 72, 与截距不可分, 要扫 m 才能测出来。

    3. 缺并发项 —— 这是本 run 能证明的那一条, 但它的归宿是 DAG 不是公式。
       实测同一个专家的 tile: 28 个 ACT 并发时中位 3.899us, 8~10 个并发时 3.554us
       (+9.7%)。受控对比: 专家1 的 tile 0~9 落在 28 并发窗、tile 10~17 落在 8 并发
       窗, 形状与输出区域完全相同。区间内唯一的片外流量就是那两次 UB→GM
       DataCopyPad, 所以本模型把这两次的字节申报到写信道上 (store_bytes), 让降速
       由速率服务器算出来, 而不是在公式里加常数 —— 与 dispatch 的处理同口径。
       信道缺省不启用时字节被过滤, 预测不变。
    """
    VEC_ELEM = VEC_ELEM_FP32
    BYTES_PER_VEC = ACT_BYTES_PER_VEC

    def __init__(self, bw_ub_bytes_per_us=BW_UB, t_startup_us=T_STARTUP_VEC,
                 mxfp_divisor: int = MXFP_DIVISOR_SIZE,
                 mxfp_scale_bytes: int = MXFP_MULTI_BASE_SIZE):
        # 几何量 (每窗列数) 调用期传入 — 同 GMM, 消除 tile_n 双份来源
        self.bw_ub = bw_ub_bytes_per_us
        self.t_startup = t_startup_us
        self.mxfp_divisor = int(mxfp_divisor)
        self.mxfp_scale_bytes = int(mxfp_scale_bytes)

    def tile(self, m: int, cols: int) -> float:
        n_vec = m * cols / self.VEC_ELEM
        return self.t_startup + n_vec * self.BYTES_PER_VEC / self.bw_ub

    def store_bytes(self, m: int, cols: int) -> float:
        """本 tile 写出到 GM 的字节: m 行 x (fp8 cols x 1B + MX scale).

        对应 StoreQuantOutput + StoreQuantScaleCompact 两次 UB→GM DataCopyPad,
        各 blockCount = m 次带 stride 的突发。只供信道折算流量; tile 时长仍走
        上面的 UB 口径 (两者的关系见类注释第 2、3 条)。
        """
        scale = -(-int(cols) // self.mxfp_divisor) * self.mxfp_scale_bytes
        return m * (int(cols) * 1.0 + scale)


COMBINE_NO_QUANT = 0
COMBINE_QUANT = 1


class AnalyticalCombineCosts:
    """COMBINE (AIV1 配对消费 GMM2 tile) 物理公式, 按量化模式参数化.

    对应内核 Gmm2Aiv1EpilogueA8W4 (A8W8 + COMBINE_NO_QUANT 走 AIV1 epilogue,
    与 trace 里 COMBINE↔GMM2 tile 1:1 吻合)。每 (m-group, N-tile) 窗三段流量:

      1. 读回: Copy(copyGM2UB) 把整个 GMM2 tile 从 GM 读回 UB, 加 metaInfo。
         m × (e_in·logical_n + meta) 字节, 本卡 HBM → BW_LOCAL_GM
         e_in = BF16 = 2B/元素 (ElementC 一路追到 RunGmm2ByMode, 四个分支
         全部显式传 bfloat16_t)
      2. 本卡行写: CombineTokens 里目的卡 == 本卡的那些行 → BW_LOCAL_GM
      3. 跨卡行写: 目的卡 != 本卡的行 → BW_REMOTE_WRITE (每核约 5.1 GB/s)

    CombineTokens 对**每一行**发一次 DataCopyPad (blockLen = logical_n×e_out),
    目标是 route.dstRankId 那张卡窗口里的 (tokenIdx·topK+topkIdx)·n + nLoc。
    每行代价随字节走, 不是随次数走 —— 若按每行固定开销 (dispatch 远端段
    0.9us/行) 算, 54 行跨卡要 50us, 与实测 5.7us 差 10 倍, 假设被否。

    写侧每元素 e_out (per-slot 部分和, topk 归约在 UNPERMUTE, 非累加 rmw):
        NO_QUANT: BF16 = 2B/元素
        QUANT:    FP8 = 1B/元素 + MX scale 1B/32元素 = 1/32 B/元素
    meta: 8B/行 = token 位置 int32 4B (metaInfo, constants.h INT32_PER_256B=8)
               + topk 权重 fp32 4B (probsGm)

    公式: T = m·(e_in·n + meta)/BW_local
            + (m-remote)·e_out'·n/BW_local
            + remote·e_out'·n/BW_remote        (e_out' = e_out + scale)

    修订史: 旧公式 T = m×(4·n + 8)/BW_SCATTER 把跨卡写按本卡 HBM 计价
    (BW_SCATTER 是 B=64 随机路由标定的本地口径), 且元素宽度用了 4B(读+写合并),
    对 20260930 bs=36 run 低估 90.6%。见 memory/combine-cost-root-cause。
    """
    META_BYTES_PER_ROW = 8

    def __init__(self, combine_quant_mode: int = COMBINE_NO_QUANT,
                 bw_local_bytes_per_us: float = BW_LOCAL_GM,
                 bw_remote_bytes_per_us: float = BW_REMOTE_WRITE,
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
        if bw_local_bytes_per_us <= 0 or bw_remote_bytes_per_us <= 0:
            raise ValueError("COMBINE 的本卡/跨卡带宽必须为正")
        self.combine_quant_mode = combine_quant_mode
        self.bw_local = bw_local_bytes_per_us
        self.bw_remote = bw_remote_bytes_per_us
        self.meta_bytes = meta_bytes_per_row

    @property
    def write_bytes_per_elem(self) -> float:
        """写侧每元素字节: NO_QUANT=2 (BF16); QUANT=1.03125 (FP8 + 1/32 scale)."""
        return self.out_elem_bytes + self.scale_bytes_per_elem

    def read_us(self, m: int, logical_n: int) -> float:
        """GMM2 tile 从 GM 读回 UB + metaInfo, 全在本卡."""
        return m * (self.in_elem_bytes * logical_n + self.meta_bytes) / self.bw_local

    def write_bytes_per_row(self, logical_n: float) -> float:
        """一行写出的字节 (CombineTokens 的一次 DataCopyPad); 片间信道按它折算."""
        return self.write_bytes_per_elem * logical_n

    def tile(self, m: int, logical_n: int = 256, remote_rows: int = 0) -> float:
        """logical_n = 本窗实际列数 (尾 N-tile 时 < 256), 调用期传入.

        remote_rows = 本窗 m 行里目的卡 != 本卡的行数, 由 routing 精确算出
        (第 r 卡第 e 专家的行按源卡分段, 源卡 != r 的就要跨卡写回)。
        """
        if not 0 <= remote_rows <= m:
            raise ValueError(f"remote_rows={remote_rows} 必须落在 [0, m={m}]")
        row_bytes = self.write_bytes_per_elem * logical_n
        return (self.read_us(m, logical_n)
                + (m - remote_rows) * row_bytes / self.bw_local
                + remote_rows * row_bytes / self.bw_remote)


def build_analytical_costs(
    *,
    h: int,
    dispatch_mechanistic: DispatchMechanisticLatency,
    urma_mechanistic: Optional[UrmaMechanisticLatency] = None,
    kernel: KernelConfig = None,
    bw_l1_gm: Optional[float] = None,
    bw_l1_gm_b_nz: float = 0.0,
    cube_mac_per_us: float = 0.0,
    gmm1_fill_us: float = 0.0,
    gmm1_tile_restart_us: float = 0.0,
    bw_ub: Optional[float] = None,
    t_startup_us: Optional[float] = None,
    bw_combine_local: Optional[float] = None,
    bw_combine_remote: Optional[float] = None,
    count_table_prepare_us: float = T_COUNT_GATE,
    gmm2_a_from_gm: bool = True,
) -> PrimitiveCosts:
    """按 (h, KernelConfig) 一致构建解析公式族, 消除 tile_n/l1_tile_k/h 漏配.

    修复的漏配类: 手工构造 Analytical* 用默认 tile_n=256 / h=6144, 而
    KernelConfig.tile_n / shape.h 被覆盖 → n_frac 换算与字节计数静默出错.
    几何量 (tile_m/tile_n/每窗列数) 属于 KernelConfig, 调用期由模型传入公式,
    容器只承载硬件常数与 L1 组织参数 — 结构上消除 tile_n/h 双份来源的漏配.

    带宽缺省 = constants 实测值; NZ 布局必须显式给 bw_l1_gm_b_nz (零猜测).
    cube_mac_per_us 缺省 0 = 不计计算项 (实测域内两个 GMM 都是权重载入绑定).
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
        gmm1_weight_blocks=1 if km.gmm1_interleaved else km.activation_n_half,
        gmm2_a_from_gm=gmm2_a_from_gm,
    )
    act = AnalyticalActCosts(
        bw_ub_bytes_per_us=bw_ub if bw_ub is not None else BW_UB,
        t_startup_us=t_startup_us if t_startup_us is not None else T_STARTUP_VEC,
    )
    comb = AnalyticalCombineCosts(
        combine_quant_mode=km.combine_quant_mode,
        bw_local_bytes_per_us=bw_combine_local if bw_combine_local is not None else BW_LOCAL_GM,
        bw_remote_bytes_per_us=(bw_combine_remote if bw_combine_remote is not None
                                else BW_REMOTE_WRITE),
    )
    return PrimitiveCosts(
        dispatch_mechanistic=dispatch_mechanistic,
        urma_mechanistic=urma_mechanistic,
        gmm1_tile=gmm.gmm1_tile,
        gmm2_tile=gmm.gmm2_tile,
        activation_tile=act.tile,
        activation_store_bytes=act.store_bytes,
        combine_tile=comb.tile,
        combine_write_bytes_per_row=comb.write_bytes_per_row,
        count_table_prepare_us=count_table_prepare_us,
        gmm1_fill_us=gmm1_fill_us,
    )
