
from dataclasses import dataclass

from .provenance import SourcedInt, SourcedValue

TILE_M = SourcedInt(256, 'kernel:mega_moe_constants.h:83 MEGAMOE_TILE_M 缺省')
TILE_N = SourcedInt(256, 'kernel:mega_moe_constants.h:86 MEGAMOE_TILE_N 缺省')
L1_TILE_K = SourcedInt(256, 'kernel:mega_moe_constants.h:82 基线 K-chunk')

# ---- 硬件物理参数----
# 内存子系统带宽
BW_L1_GM = SourcedValue(51900.0, 'measured:H 扫描差分 (gmm1_map 系列); 单点, B=64 域')
BW_UB = SourcedValue(93000.0, 'measured:ACT 大 m tile 单点')
BW_WINDOW = SourcedValue(33000.0, 'measured:dispatch 窗排空差分 ×4')
BW_SCATTER = SourcedValue(139500.0, 'measured:B=64 随机路由反推; 域受限 — B=1024 实测 +168~246% 域外失效')
BW_LOCAL_GM = SourcedValue(158000.0, 'measured:本卡 GM 读单点')

# 流水线启动/填充延迟
T_FILL_GMM1 = SourcedValue(0.0, 'assumed:零假设 — 已知缺失 (n 系差分实测 68ns/tile, 待机制推导)')
T_STARTUP_VEC = SourcedValue(1.48, 'measured:ACT 小 m 截距')

# Dispatch 机制参数
T_CALL_OH = SourcedValue(1.006, 'measured:空调用均值 n=2133')
T_LAT_LOCAL = SourcedValue(3.78, 'measured:asym 差分')
T_LAT_REMOTE = SourcedValue(3.57, 'measured:asym 首位流')
T_GMM1_OVERLAP = SourcedValue(0.9, 'measured:w≥1 首位流差分')
T_COUNT_GATE = SourcedValue(53.9, 'measured:COUNTS_EXPORT→首 dispatch span')

# 向量引擎
VEC_REG_WIDTH = SourcedInt(256, 'kernel:kernel_utils_constants.h:171 VECTOR_REG_WIDTH')
VEC_ELEM_FP32 = VEC_REG_WIDTH // 4   # FP32 元素/向量
# SwiGLU 每向量的 UB 流量 (源码精确计数):
#   读: gate(BF16 128B) + up(BF16 128B) + bf16重读(128B) = 384B
#   写: bf16中间(128B) + fp8(64B) + scale(4B) = 196B
#   总: 580B/向量
ACT_BYTES_PER_VEC = SourcedValue((128 + 128 + 128) + (128 + 64 + 4),
                                'derived:SwiGLU 源码逐项字节计数 = 580B/向量')

# GMM2 K-window 
GMM2_KL1 = SourcedInt(256, 'kernel:gmm_common.h K-chunk 基线 (自适应见 select_kl1)')
GMM1_MIN_LOGICAL_TILES_PER_CORE = SourcedInt(4, 'kernel:mega_moe_constants.h:99')
GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL = SourcedInt(2, 'kernel:mega_moe_constants.h:100')
GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE = SourcedInt(6, 'kernel:mega_moe_constants.h:101')
GMM1_SMALL_BATCH_TOKEN_THRESHOLD = SourcedInt(2048, 'kernel:mega_moe_constants.h:102')
GMM1_LARGE_BATCH_TOKEN_THRESHOLD = SourcedInt(16384, 'kernel:mega_moe_constants.h:103')
GMM2_MIN_LOGICAL_TILES_PER_CORE = SourcedInt(1, 'kernel:mega_moe_constants.h:104')
GMM2_LAG_MIN_TOKEN_NUM = SourcedInt(4096, 'kernel:mega_moe_constants.h lag 阈值')
LEGACY_GMM_MAX_PENDING_TILES = SourcedInt(15, 'kernel:legacy 常数')
DAV3510_NONINTERLEAVED_GMM1_ACTIVATION_DEPTH = SourcedInt(1, 'kernel:非交织路径 WaitForVector 结构深度')
ACTIVATION_N_HALF = SourcedInt(2, 'kernel:mega_moe_constants.h:90 SwiGLU 双投影')

def _gmm2_head_tail_fractions(k_gmm2: int, kl1: int = 0) -> tuple:
    """GMM2 head/tail by K-window physics: head = first kL1 chunk (starts after
    first ACT tile), tail = remaining K (starts after last ACT tile).
    kl1=0 → L1_TILE_K (256) 兼容旧调用; kL1 由 select_kl1 或显式参数给出."""
    window = kl1 or L1_TILE_K
    head = window / k_gmm2 if k_gmm2 > 0 else 0.5
    return (head, 1.0 - head)


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("divisor must be positive")
    return (a + b - 1) // b


# ---- Ascend 950 (DAV_3510) 物理容量 ----
TOTAL_L1_SIZE = SourcedInt(512 * 1024, 'kernel:kernel_utils_constants.h:168 __NPU_ARCH__==3510')
TOTAL_UB_SIZE = SourcedInt(248 * 1024, 'kernel:kernel_utils_constants.h:167')
TOTAL_L0C_SIZE = SourcedInt(256 * 1024, 'kernel:kernel_utils_constants.h:170')
L1_HALF_SIZE = int(TOTAL_L1_SIZE) // 2
MXFP_DIVISOR_SIZE = SourcedInt(64, 'kernel:mega_moe_constants.h:50')
MXFP_MULTI_BASE_SIZE = SourcedInt(2, 'kernel:mega_moe_constants.h:52')
SCALE_TRANSFER_BYTES = SourcedInt(64 * 1024, 'kernel:gmm_common.h:179')


def select_kl1(m_rows: int, k: int, override=None, tile_m: int = TILE_M,
               tile_n: int = TILE_N, l1_size: int = TOTAL_L1_SIZE,
               k_l1_base: int = L1_TILE_K) -> int:
    """SelectBlockMmadTilingConfig 的 kL1 选择移植 (gmm_common.h:225).

    正向规则: 整 tile (m>=tile_m) 或 K<=基线 → k_l1_base;
    部分 tile: blockM=align16(m), 2 个数据窗 + scale 窗能放进半片 L1
    且单侧 scale ≤ 64KiB → kL1 = 2×k_l1_base. (kL1 不变仅放大 scale 窗的
    分支不影响 K 窗结构, 未移植.)

    k_l1_base: K-chunk 基线, KernelConfig.l1_tile_k 可设; 缺省 = 源码 256.
    """
    if override is not None:
        return override
    base = int(k_l1_base)
    if base <= 0:
        raise ValueError("k_l1_base must be positive")
    if m_rows == 0 or m_rows >= tile_m or k <= base:
        return base
    block_m = ((m_rows + 15) // 16) * 16
    data_per_unit = block_m * base + tile_n * base             # A+B, FP8 1B/元素
    scale_k_per_unit = ((base + MXFP_DIVISOR_SIZE - 1)
                        // MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE
    scale_a = block_m * scale_k_per_unit
    scale_b = tile_n * scale_k_per_unit
    can_double = (2 * data_per_unit + 2 * (scale_a + scale_b) <= l1_size // 2
                  and 2 * scale_a <= SCALE_TRANSFER_BYTES
                  and 2 * scale_b <= SCALE_TRANSFER_BYTES)
    return base * 2 if can_double else base


# ---- 前导/尾段段常数 ----
# INPUT_QUANT: 每核向量化 MX 量化, per-token ~0.64µs + 固定 0.35µs
# (B=64: 1.1 token/核→1.03µs; B=1024: 18.3 token/核→12.2µs, 两尺度线性一致)
T_INPUT_QUANT_FIXED_US = SourcedValue(0.35, 'measured:INPUT_QUANT span 两尺度截距')
T_INPUT_QUANT_PER_TOKEN_US = SourcedValue(0.64, 'measured:每核每 token 线性斜率 (B=64→1024 两尺度)')
T_INIT_US = SourcedValue(2.0, 'measured:INIT span 中位')
T_DISPATCH_PREPARE_US = SourcedValue(20.0, 'assumed:两尺度残差 8~32µs 取中, ±12µs 不确定 — 弱常数')
#                                        (两尺度残差 8~32µs 取中, 不确定度 ±12µs)
T_COUNTS_EXPORT_US = SourcedValue(1.0, 'measured:span 中位 ~1µs, B 无关')
T_CORE_SYNC_BARRIER_US = SourcedValue(2.0, 'measured:WAIT_OUTPUT_CORE_SYNC p5 下限')
T_RANK_SYNC_RTT_US = SourcedValue(2.2, 'measured:跨 rank 最小 p5 (最晚到达核 floor)')
#                                        (最晚到达核的 floor: default r1/r3=1.6~2.0,
#                                         h6144 r1/r3=2.2~2.5; 早到核多出的 9~33µs 是
#                                         跨卡偏斜等待, 独立 rank 仿真不覆盖, 已量化声明)
T_OUTPUT_INIT_US = SourcedValue(1.0, 'measured:span 中位 ~0.94µs')
T_FINALIZE_US = SourcedValue(1.0, 'measured:span 中位 ~1.1µs')
# UNPERMUTE 聚合流式带宽: 读 topk×h×BF16 (peermem combineSend) + 写 h×BF16
# 双尺度: B=64 7.08MB→~6µs / B=1024 88.1MB→93µs → 0.95TB/s (h6144 命中, default +18%)
BW_UNPERMUTE_AGG = SourcedValue(950000.0, 'measured:UNPERMUTE 双尺度 (B=1024 命中, B=64 +18%)')


# ---------------------------------------------------------------------------
# kernel 编译期参数 (CMake cost-sweep knobs / 模板参数), Python 可设
# 对应 megamoe_profile/CMakeLists.txt:28-32 与 mega_moe_constants.h:82-90
# ---------------------------------------------------------------------------





@dataclass(frozen=True)
class KernelConfig:
    """kernel 编译期参数. 默认值 = 源码/CMake 缺省.

    对应: MEGAMOE_TILE_M/TILE_N/L1_BUF_NUM/TOPO_URMA/TOPK_PREFETCH
    (megamoe_profile/CMakeLists.txt:28-32) 与 Blaze BlockSchedulerSwizzle<3,0>
    模板参数. 修改后模型的 wave 规划/tile 网格/分核轮转随之改变.
    """

    weight_nz: bool = False           # 权重 GM 布局: Z(线性) / NZ(分形, 需 NZ 带宽标定)
    tile_m: int = 256                 # MEGAMOE_TILE_M: m-group 行高
    tile_n: int = 256                 # MEGAMOE_TILE_N: scheduler N tile
    l1_buf_num: int = 2               # MEGAMOE_L1_BUF_NUM: L1 ping-pong (1=禁用)
    topk_weights_prefetch: bool = False  # MEGAMOE_TOPK_PREFETCH
    topo_urma: bool = False           # MEGAMOE_TOPO_URMA (URMA 路径未建模, 仅声明)
    swizzle_offset: int = 3           # Blaze BlockSchedulerSwizzle<Offset,Dir>
    swizzle_direction: int = 0
    activation_n_half: int = 2        # SwiGLU 双投影
    l1_tile_k: int = 256              # K-chunk 基线 (select_kl1 自适应)
    combine_quant_mode: int = 0       # CombineQuantMode 模板参数: 0=NO_QUANT, 1=QUANT(FP8+scale)
    l1_size: int = 512 * 1024         # arch 3510 (kernel_utils_constants.h)
    aiv_num: int = 0                  # 0 = 2×aic_num


@dataclass(frozen=True)
class InstancePolicy:
    """实例层绑定 (运行时策略): 一个具体 kernel 实现的策略取值.

    三层定位: 理论层 (物理公式/依赖图, 不引用此处) → 策略层 (参数化函数)
    → 实例层 = 本绑定 + KernelConfig (编译期). 换 kernel 实现 = 换此绑定.
    全部取值来自当前实例 (mega_moe_wave_a8w8.h / mega_moe_constants.h).
    """
    dispatch_lookahead: int = 2            # 迭代0预取 W0..W(la-1); 迭代 i 预取 W(i+la-1)
    gmm2_lag_threshold: int = 4096         # tokenNum ≥ 阈值 → GMM2 lag 一波
    gmm1_activation_depth: int = 1         # 非交织路径 GMM1→ACT UB 握手深度
    gmm2_combine_credit: object = None     # 反事实旋钮 (None=关), 本实例未启用
    cursor_resonance_fix: bool = True      # cursor 共振修正启用
    wave_policy_p1: int = 2                # kernel 默认波策略 @bs<2048 (constants.h:99)
    wave_policy_p2: int = 1                # (constants.h:104; 中/大档 4/6 详见分层参考)

    def default_p1(self, token_num: int) -> int:
        """kernel 默认分层策略参考 (非模型结构): <2048→2, ≥16384→6, 其余→4."""
        if token_num < GMM1_SMALL_BATCH_TOKEN_THRESHOLD:
            return GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL
        if token_num >= GMM1_LARGE_BATCH_TOKEN_THRESHOLD:
            return GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE
        return GMM1_MIN_LOGICAL_TILES_PER_CORE
