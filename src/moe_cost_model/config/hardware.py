"""第 0 层: 硬件实测常数 + kernel 编译期参数. 出处系统: SourcedValue 标签.

字符串 = 类别前缀 + 物理含义; 测量方法与来源在注释里.
"""

from dataclasses import dataclass

from .provenance import SourcedInt, SourcedValue

# 来自 kernel: MEGAMOE_TILE_M
TILE_M = SourcedInt(256, 'kernel:每个 m-group 的行数')
# 来自 kernel: MEGAMOE_TILE_N
TILE_N = SourcedInt(256, 'kernel:每个 N-tile 的列数')
# 来自 kernel
L1_TILE_K = SourcedInt(256, 'kernel:一次载入覆盖的 K 维行数')

# ---- 硬件物理参数----
# 内存子系统带宽
# 标定: H 扫描差分 (gmm1_map 系列); B=64 域单点, 并发数未扫.
# 常数在旧口径 (激活 A 流 + 权重 B 流一起计费) 下折算得出. 现行 GMM 公式只有
# GMM1 的 A 流用到它, B 流不建模 — 口径已变, 数值待按新公式重新标定.
BW_L1_GM = SourcedValue(51900.0, 'measured:片外内存到片上 L1 的载入带宽; 旧口径 (A+B 流) 下标定, 待重标')
# 标定: ACT 大 m tile 单点.
# ACT 从 UB 读输入、向 UB 写输出, 读写流量都按此速率折算成时长.
BW_UB = SourcedValue(93000.0, 'measured:ACT 搬移 UB 数据的带宽')
# 标定: dispatch 窗排空差分 ×4.
BW_WINDOW = SourcedValue(33000.0, 'measured:跨卡读数据的片间带宽, dispatch 用')
# 标定: B=64 随机路由反推;
# 只剩本卡侧口径: COMBINE 现在把 GM→UB 读回与本卡行写按 BW_LOCAL_GM 计,
# 跨卡行写按 BW_REMOTE_WRITE 计, 本常数不再进 COMBINE 公式 (留给旧标定复现).
BW_SCATTER = SourcedValue(139500.0, 'measured:COMBINE 散射写带宽 (旧口径, 已不用); 域受限')
# COMBINE 的跨卡行写 (CombineTokens 每行一次 DataCopyPad 直写目的卡窗口) 无直测:
# 现有跨卡常数都是读侧 (BW_REMOTE_GM 是 dispatch 远端段的读)。同引擎同互连,
# 暂按对称假设取读侧值 —— 与 URMA_PUT_BW_SINGLE 取 GET 对称值同一处理。
# 标定待办: 20260930 bs=36 run 只有一种 tile 形状 (m=72, 每行 512B), 单点不足以
# 分离"每行固定开销"与"按字节"两项; 需要扫 m 或扫 tile_n 的 run 才能定这个常数。
BW_REMOTE_WRITE = SourcedValue(31000.0, 'assumed:跨卡写目的卡窗口的带宽 (每核), 取远端读对称值; COMBINE 用')

# 流水线启动/填充延迟
# 暂取 0; 实测存在 68ns/tile 的缺口 (n 系差分), 机制待推导.
T_FILL_GMM1 = SourcedValue(0.0, 'assumed:GMM1 流水填充延迟, 暂取 0')
# 标定: ACT 小 m tile 截距; ACT 公式把它作为与数据量无关的固定项.
T_STARTUP_VEC = SourcedValue(1.48, 'measured:向量流水启动与排空的固定时长')

# Dispatch 机制参数 
# 2026-09 重标定: 旧值 T_LAT_LOCAL=3.78/T_LAT_REMOTE=3.57 为 call 级差分,
# 新值从 dispatch_transfer_raw.csv 逐段直接反解 (含 MTE 启动 + dispatch 路径开销).
# 标定: 不搬任何行的调用均值, n=2133 (asym 数据).
T_CALL_OH = SourcedValue(1.006, 'measured:dispatch 零行调用的固定开销')
# 标定: dispatch_transfer_raw.csv 1行(1805cyc)/2行(1868cyc) 段差分;
# 2026-09 重标定, 旧 call 级差分值 3.78.
T_LAT_LOCAL = SourcedValue(1.78, 'measured:本地段固定延迟 = MTE 启动 78cyc + dispatch 软件路径 1687cyc')
# 标定: dispatch_transfer_raw.csv 1行(2629cyc)/2行(2831cyc) 段差分; 旧值 3.57.
T_LAT_REMOTE = SourcedValue(2.43, 'measured:远端段固定延迟 = MTE 加片间互连 740cyc + dispatch 软件路径 1687cyc')
# 标定: MTE 大尺寸拟合, 单核载入无其他搬运.
BW_LOCAL_GM = SourcedValue(157000.0, 'measured:读本卡内存的带宽, dispatch 本地段用')
# 标定: dispatch_transfer_raw.csv 1→2 行段差分.
BW_REMOTE_GM = SourcedValue(31000.0, 'measured:读远端卡内存的带宽, dispatch 远端段用')
# GMM1 计算与首个远端段部分重叠, 此为补偿项. 标定: w≥1 首位流差分.
T_GMM1_OVERLAP = SourcedValue(0.9, 'measured:dispatch 调用首个远端段的附加时长')
# 标定: COUNTS_EXPORT→首 dispatch span.
T_COUNT_GATE = SourcedValue(53.9, 'measured:COUNTS_EXPORT 到首个 dispatch 的最短间隔')

# ---- URMA 机制参数 (Layered 路径) ----
# 来源: fab_arbitration_probe/results/report.md (2026-09-29, 4x Ascend950PR,
# CANN 9.1.0, URMA GET = Hcomm::ReadNbi 默认 WQE, 行 6336B 512B 对齐槽位,
# 单核 channel-owner 流语义, cycle 标定 ~310-353 ticks/us)
# pair 两点实验 OLS 拟合 (λ + bytes/BW): pair_r0.csv 逐 (rows) 中位 4 点,
# 6336B→12.63us / 25344B→19.88us / 101376B→51.52us / 405504B→188.92us
# 标定: pair 两点实验 OLS 拟合; 报告 8.5µs, 复算 8.47µs.
URMA_GET_LAT_US = SourcedValue(8.5, 'measured:URMA GET 单次读的固定延迟')
# 标定: pair 两点实验的差分速率; drain-chunk=8 下界.
URMA_GET_BW_SINGLE = SourcedValue(2253.0, 'measured:URMA GET 读带宽, 2.25 GB/s')
# PUT (WriteNbi) 无直测数据: fab probe 仅测 ReadNbi(GET); Combine 写走 PUT。
# GET/PUT 同引擎同互连, 暂按对称假设取 GET 常数 — assumed 显式暴露。
URMA_PUT_LAT_US = SourcedValue(8.5, 'assumed:URMA PUT 单次写的固定延迟, 取 GET 对称值')
URMA_PUT_BW_SINGLE = SourcedValue(2253.0, 'assumed:URMA PUT 写带宽, 取 GET 对称值')
# 并发纪律 (mdst/stagger/gran 三组判定, 见报告 [2][4][5]): 并行不降速
# (3 流保持率 0.96-0.97), 晚发射不被在飞流推迟 (无 FCFS 整传输预留),
# 仲裁量子 << 25KB (细粒度交叉) → 每事件独立 λ+bytes/BW, 不叠加速率服务器。
# 实测域: 4 卡 / 3 流; world-1 > 3 的并发外推未验证。
# 来自 kernel: DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY
URMA_FLAG_WINDOW_TOKENS = SourcedInt(256, 'kernel:flag 轮询窗槽数')
URMA_FLAG_BYTES = SourcedInt(8, 'kernel:relay flag 槽位字节数, uint64')

# ---- URMA Layered 宏 Wave 策略 ----
LAYERED_FIRST_WAVE_ROWS = SourcedInt(1024, 'kernel:Layered 首波行数上限, 小 batch 单波判定')
LAYERED_LATENCY_WAVE_COUNT = SourcedInt(2, 'kernel:Layered 延迟档波数')
LAYERED_BALANCED_WAVE_COUNT = SourcedInt(6, 'kernel:Layered 均衡档波数')
LAYERED_THROUGHPUT_WAVE_COUNT = SourcedInt(4, 'kernel:Layered 大批量优先档的波数')
LAYERED_LATENCY_ROWS_PER_EXPERT = SourcedInt(256, 'kernel:Layered 延迟档每专家行数')
LAYERED_THROUGHPUT_ROWS_PER_EXPERT = SourcedInt(2048, 'kernel:Layered 大批量优先档的每专家行数')
LAYERED_FEW_EXPERT_THRESHOLD = SourcedInt(8, 'kernel:Layered 少专家判定阈值, ≤8 走延迟档')
# Layered 每行元数据: META_INFO_SIZE = 8 × int32
LAYERED_META_BYTES_PER_ROW = SourcedInt(32, 'kernel:Layered 每行元数据字节数')
# 来自 kernel: MXFP_MULTI_BASE_SIZE
MXFP_MULTI_BASE_SIZE_K = SourcedInt(2, 'kernel:MX scale K 侧每 32 组字节数')

# 向量引擎
# 来自 kernel: VECTOR_REG_WIDTH
VEC_REG_WIDTH = SourcedInt(256, 'kernel:向量寄存器位宽, bit')
VEC_ELEM_FP32 = VEC_REG_WIDTH // 4   # FP32 元素/向量
# SwiGLU 每向量的 UB 流量 :
#   读: gate(BF16 128B) + up(BF16 128B) + bf16重读(128B) = 384B
#   写: bf16中间(128B) + fp8(64B) + scale(4B) = 196B
#   总: 580B/向量
ACT_BYTES_PER_VEC = SourcedValue((128 + 128 + 128) + (128 + 64 + 4),
                                'derived:ACT 每处理一个向量的 UB 字节数, 读 384B + 写 196B')

# GMM2 K-window 
GMM1_MIN_LOGICAL_TILES_PER_CORE = SourcedInt(4, 'kernel:p1 中档缺省: 每核最少 GMM1 逻辑 tile 数')
GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL = SourcedInt(2, 'kernel:p1 小批量档: token<2048 时每核最少 GMM1 tile 数')
GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE = SourcedInt(6, 'kernel:p1 大批量档: token≥16384 时每核最少 GMM1 tile 数')
GMM1_SMALL_BATCH_TOKEN_THRESHOLD = SourcedInt(2048, 'kernel:p1 小批量档 token 阈值')
GMM1_LARGE_BATCH_TOKEN_THRESHOLD = SourcedInt(16384, 'kernel:p1 大批量档 token 阈值')
# 来自 kernel: p1/p2 分档与阈值
GMM2_MIN_LOGICAL_TILES_PER_CORE = SourcedInt(1, 'kernel:p2 缺省: 每核最少 GMM2 tile 数')
GMM2_LAG_MIN_TOKEN_NUM = SourcedInt(4096, 'kernel:GMM2 滞后一波的 token 阈值')
# 来自 kernel
ACTIVATION_N_HALF = SourcedInt(2, 'kernel:SwiGLU 投影数, gate+up 共 2')
# InstancePolicy.gmm1_activation_depth 的缺省来源
DAV3510_NONINTERLEAVED_GMM1_ACTIVATION_DEPTH = SourcedInt(1, 'kernel:GMM1→ACT UB 握手深度, 非交织路径')

def _gmm2_head_tail_fractions(k_gmm2: int, kl1: int = 0) -> tuple:
    """GMM2 head/tail by K-window physics: head = first kL1 chunk (starts after
    first ACT tile), tail = remaining K (starts after last ACT tile).
    kl1=0 → L1_TILE_K (256) 兼容旧调用; kL1 由 select_kl1 或显式参数给出.
    K < kL1 时 head 覆盖全部 K: 窗口截断到 min(kL1, K), tail 比例 ≥ 0."""
    window = min(kl1 or L1_TILE_K, k_gmm2) if k_gmm2 > 0 else 0
    if k_gmm2 <= 0:
        return (0.5, 0.5)
    head = window / k_gmm2
    return (head, 1.0 - head)


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("divisor must be positive")
    return (a + b - 1) // b


# ---- Ascend 950 (DAV_3510) 物理容量 ----
# 来自 kernel: __NPU_ARCH__==3510
TOTAL_L1_SIZE = SourcedInt(512 * 1024, 'kernel:L1 容量, 字节')
# 来自 kernel
TOTAL_UB_SIZE = SourcedInt(248 * 1024, 'kernel:UB 容量, 字节')
# 来自 kernel
TOTAL_L0C_SIZE = SourcedInt(256 * 1024, 'kernel:L0C 容量, 字节')
# 来自 kernel
MXFP_DIVISOR_SIZE = SourcedInt(64, 'kernel:MX 量化组大小, 元素/组')
# 来自 kernel
MXFP_MULTI_BASE_SIZE = SourcedInt(2, 'kernel:MX scale 每组字节数')
# 来自 kernel
SCALE_TRANSFER_BYTES = SourcedInt(64 * 1024, 'kernel:scale 载入窗单侧上限, 字节')


def select_kl1(m_rows: int, k: int, override=None, tile_m: int = TILE_M,
               tile_n: int = TILE_N, l1_size: int = TOTAL_L1_SIZE,
               k_l1_base: int = L1_TILE_K, n_windows: int = 2) -> int:
    """kL1 选择: 部分 tile 时容量允许则 kL1 翻倍.

    正向规则: 整 tile (m>=tile_m) 或 K<=基线 → k_l1_base;
    部分 tile: blockM=align16(m), n_windows 个数据窗 + scale 窗能放进
    半片 L1 且单侧 scale ≤ 64KiB → kL1 = 2×k_l1_base. (kL1 不变仅放大
    scale 窗的分支不影响 K 窗结构, 未移植.)

    k_l1_base: K-chunk 基线, KernelConfig.l1_tile_k 可设; 缺省 = 源码 256.
    n_windows: L1 缓冲窗数, 缺省 2 (kernel 双缓冲). 多缓冲变体
    (l1_buf_num=3) 传入 3 — 窗数增加时 kL1 倾向不翻倍, 这是多缓冲
    的容量代价.
    """
    if override is not None:
        return override
    base = int(k_l1_base)
    if base <= 0:
        raise ValueError("k_l1_base must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    if m_rows == 0 or m_rows >= tile_m or k <= base:
        return base
    block_m = ((m_rows + 15) // 16) * 16
    data_per_unit = block_m * base + tile_n * base             # A+B, FP8 1B/元素
    scale_k_per_unit = ((base + MXFP_DIVISOR_SIZE - 1)
                        // MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE
    scale_a = block_m * scale_k_per_unit
    scale_b = tile_n * scale_k_per_unit
    can_double = (n_windows * data_per_unit + n_windows * (scale_a + scale_b)
                  <= l1_size // 2
                  and n_windows * scale_a <= SCALE_TRANSFER_BYTES
                  and n_windows * scale_b <= SCALE_TRANSFER_BYTES)
    return base * 2 if can_double else base


# ---- 前导/尾段段常数 ----
# 前导已移出 DAG. T_INPUT_QUANT_* / T_INIT_US 为实测记录, 保留 —
# 前导若建回 DAG 可直接用作标定值; 尾段在用的见下方 T_COUNTS_EXPORT_US 等
# INPUT_QUANT: 每核向量化 MX 量化, per-token ~0.64µs + 固定 0.35µs
# (B=64: 1.1 token/核→1.03µs; B=1024: 18.3 token/核→12.2µs, 两尺度线性一致)
T_INPUT_QUANT_FIXED_US = SourcedValue(0.35, 'measured:输入量化固定开销')
T_INPUT_QUANT_PER_TOKEN_US = SourcedValue(0.64, 'measured:输入量化每 token 每核时长')
T_INIT_US = SourcedValue(2.0, 'measured:INIT 阶段时长')
#                                        (两尺度残差 8~32µs 取中, 不确定度 ±12µs)
# 标定: span 中位 ~1µs, 与 B 无关.
T_COUNTS_EXPORT_US = SourcedValue(1.0, 'measured:COUNTS_EXPORT 阶段时长')
# 标定: WAIT_OUTPUT_CORE_SYNC p5 下限.
T_CORE_SYNC_BARRIER_US = SourcedValue(2.0, 'measured:核间同步屏障时长')
# 标定: 跨 rank 最小 p5 (最晚到达核 floor).
T_RANK_SYNC_RTT_US = SourcedValue(2.2, 'measured:跨 rank 同步往返时长')
#                                        (最晚到达核的 floor: default r1/r3=1.6~2.0,
#                                         h6144 r1/r3=2.2~2.5; 早到核多出的 9~33µs 是
#                                         跨卡偏斜等待, 独立 rank 仿真不覆盖, 已量化声明)
# 标定: span 中位 ~0.94µs.
T_OUTPUT_INIT_US = SourcedValue(1.0, 'measured:输出缓冲初始化时长')
# 标定: span 中位 ~1.1µs.
T_FINALIZE_US = SourcedValue(1.0, 'measured:FINALIZE 阶段时长')
# UNPERMUTE 聚合流式带宽: 读 topk×h×BF16 (peermem combineSend) + 写 h×BF16
# 双尺度: B=64 7.08MB→~6µs / B=1024 88.1MB→93µs → 0.95TB/s (h6144 命中, default +18%)
# UNPERMUTE 每个 token 读 topk 份、写 1 份, 读写字节都计在此带宽内.
BW_UNPERMUTE_AGG = SourcedValue(950000.0, 'measured:UNPERMUTE 阶段读加写的总带宽')


# ---------------------------------------------------------------------------
# kernel 编译期参数 (CMake cost-sweep knobs / 模板参数), Python 可设
# 对应 kernel 工程的 CMake 编译旋钮
# ---------------------------------------------------------------------------





@dataclass(frozen=True)
class KernelConfig:
    """kernel 编译期参数. 默认值 = 源码/CMake 缺省.

    对应: MEGAMOE_TILE_M/TILE_N/L1_BUF_NUM/TOPO_URMA/TOPK_PREFETCH
    (megamoe_profile/CMakeLists.txt:28-32) 与 Blaze BlockSchedulerSwizzle<3,0>
    模板参数. 修改后模型的 wave 规划/tile 网格/分核轮转随之改变.
    """

    weight_nz: bool = False           # 权重 GM 布局: Z(线性) / NZ(分形). 不影响时长:
                                      #   B 流 (权重载入) 不进 GMM tile 公式
    tile_m: int = 256                 # MEGAMOE_TILE_M: 每个 m-group 的行数
    tile_n: int = 256                 # MEGAMOE_TILE_N: scheduler N tile
    l1_buf_num: int = 2               # MEGAMOE_L1_BUF_NUM: L1 ping-pong (1=禁用)
    topk_weights_prefetch: bool = False  # MEGAMOE_TOPK_PREFETCH (未使用: 硬门查的是
                                          #   ModelOptions.topk_weights_prefetch, 本字段无读者)
    topo_urma: bool = False           # MEGAMOE_TOPO_URMA: True → URMA Layered 路径
                                      #   (MegaMoeLayered); 建模见
                                      #   layered.py — 单 Server 假设, PUT 复用 GET 常数
    # Blaze BlockSchedulerSwizzle<Offset, Direction>; kernel 实例化为 <3, 1>
    # (common/mega_moe_gmm_common.h: using BlockScheduler = BlockSchedulerSwizzle<3, 1>),
    # GMM1 与 GMM2 都用它。Direction=1 = N 维在外层, 连续 tile 共享同一 A 行块。
    swizzle_offset: int = 3
    swizzle_direction: int = 1
    activation_n_half: int = ACTIVATION_N_HALF   # SwiGLU 双投影
    l1_tile_k: int = 256              # K-chunk 基线 (select_kl1 自适应)
    # GMM1 B 复用. 不影响时长: B 流 (权重载入) 不进 GMM tile 公式,
    # 复用与否都不计费. 字段保留以对应 kernel 选项.
    gmm1_b_reuse: bool = False
    combine_quant_mode: int = 0       # CombineQuantMode 模板参数: 0=NO_QUANT, 1=QUANT(FP8+scale)
    l1_size: int = 512 * 1024         # DAV_3510 平台
    # 核数 aic_num 是可调场景输入 (MegaMoeShape.aic_num); 向量核数恒为 2×aic_num
    # (每 block 1 AIC + 2 AIV, 平台结构常数), 模型以每核 AIV0/AIV1 两角色表达,
    # 不设独立旋钮。
