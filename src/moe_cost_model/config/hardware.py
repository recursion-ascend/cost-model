"""第 0 层: 硬件实测常数 + kernel 编译期参数. 出处系统: SourcedValue 标签.

字符串 = 类别前缀 + 物理含义; 测量方法与来源在注释里.
"""

from dataclasses import dataclass

from .provenance import SourcedInt, SourcedValue

# 来自 kernel: MEGAMOE_TILE_M
TILE_M = SourcedInt(256, 'impl:每个 m-group 的行数')
# 来自 kernel: MEGAMOE_TILE_N
TILE_N = SourcedInt(256, 'impl:每个 N-tile 的列数')
# 来自 kernel
L1_TILE_K = SourcedInt(256, 'impl:一次载入覆盖的 K 维行数')

# ---- 硬件物理参数----
# 内存子系统带宽
# 标定: H 扫描差分 (gmm1_map 系列); B=64 域单点, 并发数未扫.
# 常数在旧口径 (激活 A 流 + 权重 B 流一起计费) 下折算得出. 现行 GMM 公式只有
# GMM1 的 A 流用到它, B 流不建模 — 口径已变, 数值待按新公式重新标定.
# 单核值。两条独立标定互相不一致, 都记下来:
#   旧口径 (B=64 H 扫描差分单点) 反解 51900;
#   2026-10-04 按"固定并发核数只变 m"的 A 流斜率反解: 28 核 45300, 18 核 37000
#   —— 随并发核数变, 所以它不是一个常数。
# 聚合上界另有约束: 单核值 x 活跃核数 不得超过平台聚合 HBM 带宽 (config/platform.py
# 的 PlatformSpec.gm_bw_per_core)。950PR 在 28 核上 51900 已占聚合的 91%。
BW_L1_GM = SourcedValue(51900.0, 'measured:片外内存到片上 L1 的载入带宽 (单核); 随并发核数变, 待按并发分档')
# 标定: ACT 大 m tile 单点.
# ACT 从 UB 读输入、向 UB 写输出, 读写流量都按此速率折算成时长.
#
# 出处待重标 (2026-09-30): 这个值是**用 ACT_BYTES_PER_VEC=580 在单点上反解**的,
# 而 580 已被证明与源码不符 (真值 722, 漏了 ComputeFp8Data 那一遍 bf16 重读)。
# 同一个标定点按 722 重算会给 93000x722/580 = 115769; 而 20260930 的两点 m 扫
# (bs36 m=72 / bs8192 m=256) 定出斜率 0.007760 us/向量, 对应 bw = 93041。
# 两者差 24% —— 说明老标定点与新 run 的条件不同 (并发度/形状), 不能互换。
# 现在保留 93000: 它与两点 m 扫一致, 而老标定点的原始数据已不可得, 无法重算。
# 要彻底定死这个常数, 需要一次**已知并发度**下的 ACT 扫 m / 扫 tileN。
BW_UB = SourcedValue(93000.0, 'measured:ACT 搬移 UB 数据的带宽; 出处按 580 反解, 待重标')
# 标定: dispatch 窗排空差分 ×4.
BW_WINDOW = SourcedValue(33000.0, 'measured:跨卡读数据的片间带宽, dispatch 用')
# 标定: B=64 随机路由反推;
# 只剩本卡侧口径: COMBINE 现在把 GM→UB 读回与本卡行写按 BW_LOCAL_GM 计,
# 跨卡行写按 BW_REMOTE_WRITE 计, 本常数不再进 COMBINE 公式 (留给旧标定复现).
# 2026-10-05: 确认**真的没有任何公式再用它**。此前标着"已不用", 但
# builders/pipeline_expand.py 还在用 "base_dur x BW_SCATTER" 从时长倒推 COMBINE 的
# hbm_write 字节 —— 方向反了, 而且那股字节只在开相位流水时出现 (换编排旋钮不该改变
# 搬了多少字节)。现在 COMBINE 的本卡读回与本卡行写由 builders/comm/mte.py 按字节直接
# 申报, 这个常数只作为旧标定的复现记录保留, 不进任何公式、不进任何申报。
BW_SCATTER = SourcedValue(139500.0, 'measured:COMBINE 散射写带宽 (旧口径); 域受限 (B=64 随机路由标定); 已退役, 不进任何公式, 仅留复现记录')
# COMBINE 的跨卡行写 (CombineTokens 每行一次 DataCopyPad 直写目的卡窗口) 没有直测,
# 只能从 COMBINE 事件的总时长里反扣。2026-10-04 之前这里取的是"与读侧对称"的假设值
# 31000 —— 现已被现有 trace 排除, 见下。
BW_REMOTE_WRITE = SourcedValue(8600.0, 'measured:跨卡写目的卡窗口的带宽 (每核); COMBINE 用')
# 8600 B/us 的来历 (2026-10-04, 20260930 三个 noshared run, rank0, 去掉 pid 重复记录):
#   取每个形状**最快**的 COMBINE tile (未被别的阶段挤的那一条), 把读回与本卡行写按
#   BW_LOCAL_GM 扣掉, 余下时间除以跨卡字节:
#     bs8192  m=256 floor 11.46us (206 个样本, p10=11.30, 分布极紧 -> 是硬速率)  -> 9.5 GB/s
#     bs36    m= 72 min   3.86us                                                -> 7.8 GB/s
#     bs128   m=256 min  22.74us (该 run 全程被挤, 这个"min"仍含排队)           -> 4.5 GB/s
#   前两个取中 ~8.6 GB/s; 第三个是下界。**原先的 31000 是"取远端读对称值"的假设,
#   现有 trace 已经把它排除掉了 (差 3-7 倍)**。要定准仍需 R4 (按每行字节扫)。

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
#
# 待清理 (2026-09-30): 这一项有两个问题, 都是框架层面的, 不是数值层面的。
#   1) 它把**跨 stage 的调度交互**写成了加在单事件时长上的常数。本项目的口径是
#      把执行过程建成事件 DAG, 让 GMM1 与 dispatch 的重叠由依赖边和核资源互斥
#      在离散事件引擎里自己走出来 —— 而不是折进 dispatch 事件的时长里。这是
#      roofline 式补偿项的残留。
#   2) 它的标定基于 segment_us 的旧形态 (λ + rows·b_row/BW, 把流水里重叠的行也
#      串行计了)。2026-09-30 segment_us 改成 buffer_count 槽的行级软流水后, 它当初
#      吸收的残差已经变了, 标定失效。
#   实际影响不小: 20260930 bs=36 run 里它落在 61 个 dispatch 事件中的 44 个 (72%),
#   模型单事件中位 3.330 vs 实测 2.779 (+19.8%); 去掉它是 2.430 (-12.6%)。
#   两个数都不对 —— 说明要的是把重叠建成边, 不是换个常数。故先不动值, 只标明。
T_GMM1_OVERLAP = SourcedValue(0.9, 'measured:dispatch 调用首个远端段的附加时长')
# 来自 kernel/tiling: dispatchBufferConfig.bufferCount.
# CopyTokensAndMetaForDispatch 是 bufferCount 槽的行级软流水: 前 bufferCount 行的
# Fetch 背靠背发出 (模板参数 Wait=false), 只有 issueIdx >= bufferCount 才等
# MTE3_MTE2 让槽腾出来。所以一段里不超过 bufferCount 行是重叠的, 段时长由一次
# 往返延迟封底, 不随行数线性增长。缺省 6 = 20260930 run 的 tiling 真值。
DISPATCH_BUFFER_COUNT = SourcedInt(6, 'impl:dispatchBufferConfig.bufferCount, 行级软流水槽数')
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
URMA_FLAG_WINDOW_TOKENS = SourcedInt(256, 'impl:flag 轮询窗槽数')
URMA_FLAG_BYTES = SourcedInt(8, 'impl:relay flag 槽位字节数, uint64')

# ---- URMA Layered 宏 Wave 策略 ----
LAYERED_FIRST_WAVE_ROWS = SourcedInt(1024, 'impl:Layered 首波行数上限, 小 batch 单波判定')
LAYERED_LATENCY_WAVE_COUNT = SourcedInt(2, 'impl:Layered 延迟档波数')
LAYERED_BALANCED_WAVE_COUNT = SourcedInt(6, 'impl:Layered 均衡档波数')
LAYERED_THROUGHPUT_WAVE_COUNT = SourcedInt(4, 'impl:Layered 大批量优先档的波数')
LAYERED_LATENCY_ROWS_PER_EXPERT = SourcedInt(256, 'impl:Layered 延迟档每专家行数')
LAYERED_THROUGHPUT_ROWS_PER_EXPERT = SourcedInt(2048, 'impl:Layered 大批量优先档的每专家行数')
LAYERED_FEW_EXPERT_THRESHOLD = SourcedInt(8, 'impl:Layered 少专家判定阈值, ≤8 走延迟档')
# Layered 每行元数据: META_INFO_SIZE = 8 × int32
LAYERED_META_BYTES_PER_ROW = SourcedInt(32, 'impl:Layered 每行元数据字节数')
# 来自 kernel: MXFP_MULTI_BASE_SIZE
MXFP_MULTI_BASE_SIZE_K = SourcedInt(2, 'spec:MX scale K 侧每 32 组字节数 (MX 格式标准)')

# 向量引擎
# 来自 kernel: VECTOR_REG_WIDTH
VEC_REG_WIDTH = SourcedInt(256, 'spec:向量寄存器位宽, bit')
VEC_ELEM_FP32 = VEC_REG_WIDTH // 4   # FP32 元素/向量
# SwiGLU + MX 量化每向量 (64 个 FP32 元素) 的 UB 流量, 从源码逐句计数:
#   bf16 中间缓冲被**整体流三遍** (blaze/epilogue/block_epilogue_activation_mx_quant.h):
#     SwiGLU 写一遍, ComputeMaxExp 读一遍, ComputeFp8Data 再读一遍
#   读: gate(BF16 128B) + up(BF16 128B) + ComputeMaxExp 重读(128B)
#       + ComputeFp8Data 重读(128B) + maxExp/inverseMxScale 回读(8B) = 520B
#   写: bf16 中间(128B) + maxExp(4B) + inverseMxScale(4B) + fp8(64B) + scale(2B) = 202B
#   总: 722B/向量
# 旧值 580 漏了 ComputeFp8Data 那一遍重读 (128B) 与 scale 中间量 (~14B)。
# 两点 m 扫独立验证 (2026-09-30): bs36 (m=72, n_vec=288) 与 bs8192 (m=256, n_vec=1024)
# 两个 run 定出实测斜率 0.007760 us/向量; 722/BW_UB = 0.007763 (+0.05%), 580 低 19.6%。
# 截距实测 1.425 us 对 T_STARTUP_VEC=1.48 (差 3.8%)。改后 ACT 误差 -10.5%/-16.1%
# -> +1.5%/+0.6%。注意: 单个 run 里 54 个 tile 形状全同, 只有比值可观测, 所以这个
# 修正必须靠两个不同 m 的 run 才能与 BW_UB 分开 —— 单 run 改它是变相拟合。
ACT_BYTES_PER_VEC = SourcedValue((128 + 128 + 128 + 128 + 8) + (128 + 4 + 4 + 64 + 2),
                                'derived:ACT 每处理一个向量的 UB 字节数, 读 520B + 写 202B')

# GMM2 K-window 
GMM1_MIN_LOGICAL_TILES_PER_CORE = SourcedInt(4, 'impl:p1 中档缺省: 每核最少 GMM1 逻辑 tile 数')
GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL = SourcedInt(2, 'impl:p1 小批量档: token<2048 时每核最少 GMM1 tile 数')
GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE = SourcedInt(6, 'impl:p1 大批量档: token≥16384 时每核最少 GMM1 tile 数')
GMM1_SMALL_BATCH_TOKEN_THRESHOLD = SourcedInt(2048, 'impl:p1 小批量档 token 阈值')
GMM1_LARGE_BATCH_TOKEN_THRESHOLD = SourcedInt(16384, 'impl:p1 大批量档 token 阈值')
# 来自 kernel: p1/p2 分档与阈值
GMM2_MIN_LOGICAL_TILES_PER_CORE = SourcedInt(1, 'impl:p2 缺省: 每核最少 GMM2 tile 数')
GMM2_LAG_MIN_TOKEN_NUM = SourcedInt(4096, 'impl:GMM2 滞后一波的 token 阈值')
# 来自 kernel
ACTIVATION_N_HALF = SourcedInt(2, 'algo:SwiGLU 投影数, gate+up 共 2 (算法定义, 换算法才变)')
# config.links 里 gmm1->activation 那条边 depth 的缺省来源
DAV3510_NONINTERLEAVED_GMM1_ACTIVATION_DEPTH = SourcedInt(1, 'impl:GMM1→ACT UB 握手深度, 非交织路径')

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
TOTAL_L1_SIZE = SourcedInt(512 * 1024, 'spec:L1 容量, 字节 (平台物理容量)')
# 来自 kernel
TOTAL_UB_SIZE = SourcedInt(248 * 1024, 'spec:UB 容量, 字节 (平台物理容量)')
# 来自 kernel
TOTAL_L0C_SIZE = SourcedInt(256 * 1024, 'spec:L0C 容量, 字节 (平台物理容量)')
# 来自 kernel
MXFP_DIVISOR_SIZE = SourcedInt(64, 'spec:MX 量化组大小, 元素/组 (MX 格式标准)')
# 来自 kernel
MXFP_MULTI_BASE_SIZE = SourcedInt(2, 'spec:MX scale 每组字节数 (MX 格式标准)')
# 来自 kernel
SCALE_TRANSFER_BYTES = SourcedInt(64 * 1024, 'impl:scale 载入窗单侧上限, 字节')


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


@dataclass(frozen=True)
class EpilogueOverheads:
    """尾段的固定开销 (C5): 全是**某一版实现**的实测残留, 不是物理定律.

    counts_export / output_core_sync / output_rank_sync / output_buffer_init /
    finalize: 这五项是尾段的固定耗时, 换一版 kernel 就会变。注意 unpermute **不在这里**
    —— 它是真实的数据搬运, 由 token 数 x topk x h 的字节量除以 BW_UNPERMUTE_AGG 算出来,
    属于物理。

    **哪个缺省是什么, 2026-10-05 把话说清** (原先这段写"缺省沿用实测值, 这样默认结果
    不变", 那句描述的是本类自己的行为, 却被读成"模型缺省沿用实测值" —— 我自己照着它
    误报过好几轮"实测残留藏在缺省值里"):

      literal=False (本类的缺省)   五项里为 0 的回落到模块实测常数
                                   (T_COUNTS_EXPORT_US / T_CORE_SYNC_BARRIER_US / ...)。
                                   `profiles.MEGAMOE_A8W8` 用这个 —— 复现那份实现。
      literal=True                 五项按字面取 (含 0), 不回落。
                                   **`ModelOptions.epilogue_overheads` 的缺省是这个**
                                   (shape._ZERO_OVERHEADS), 所以**模型缺省下尾段固定开销
                                   就是 0**, 实测残留只出现在 profile 里。
                                   这才符合"缺省值不引用任何实现"。

    换句话说: 实测残留**没有**藏在模型缺省里; 要它得显式用 profile 或自己填值。
    """
    counts_export_us: float = 0.0        # 0 = 用模块常数 T_COUNTS_EXPORT_US
    core_sync_us: float = 0.0            # 0 = T_CORE_SYNC_BARRIER_US
    rank_sync_us: float = 0.0            # 0 = T_RANK_SYNC_RTT_US
    output_init_us: float = 0.0          # 0 = T_OUTPUT_INIT_US
    finalize_us: float = 0.0             # 0 = T_FINALIZE_US
    #: True 时上面五项按字面取 (含 0), 不回落到模块常数 —— 用来跑"纯物理"基线
    literal: bool = False
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

    # 权重 GM 布局: Z(线性) / NZ(分形)。**影响时长**: NZ 时 GMM 的 B 流 (权重载入) 改用
    # NZ 路径的 GM->L1 带宽 (Calibration.bw_l1_gm_b_nz, 必须实测给出, 不给直接报错 ——
    # 不声称 NZ 与 Z 同速)。2026-10-05 之前这里写"不影响时长: B 流不进 GMM tile 公式",
    # 那是 B 流还没进公式时的旧话; 现在 b_load = k·cols / bw_b, bw_b 由本字段选。
    weight_nz: bool = False
    tile_m: int = 256                 # MEGAMOE_TILE_M: 每个 m-group 的行数
    tile_n: int = 256                 # MEGAMOE_TILE_N: scheduler N tile
    l1_buf_num: int = 2               # MEGAMOE_L1_BUF_NUM: L1 ping-pong (1=禁用)
    topo_urma: bool = False           # MEGAMOE_TOPO_URMA: True → URMA Layered 路径
                                      #   (MegaMoeLayered); 建模见
                                      #   layered.py — 单 Server 假设, PUT 复用 GET 常数
    # Blaze BlockSchedulerSwizzle<Offset, Direction>; 仓内 kernel 实例化为 **<3, 0>**
    # (common/mega_moe_gmm_common.h:33: using BlockScheduler = BlockSchedulerSwizzle<3, 0>),
    # GMM1 与 GMM2 共用这一个别名 (stage/mega_moe_gmm1_activation.h:878,951 与
    # stage/mega_moe_gmm2_combine.h:510,892,980 都取 GmmKernel::BlockScheduler)。
    # Direction=0 = "m first": loopFirst = ceil(M/tileM) = m 组数, loopSecond = n-tile 数
    # (block_scheduler_swizzle.h:41-47 构造函数, :85-93 GetBlockCoord 的返回分支)。
    #
    # 2026-10-05 之前这里缺省 1, 注释也声称 kernel 用 <3, 1> —— 与仓内源码相反。
    # 它**影响时长**: tile->核 的轮转顺序变了 (planning/tile_grid.py:96 ->
    # planning/waves.py:154)。planning/waves.py:155 的函数缺省一直是 0 (与源码一致),
    # 所以两个 Python 缺省自己也不一致。
    swizzle_offset: int = 3
    swizzle_direction: int = 0
    activation_n_half: int = ACTIVATION_N_HALF   # SwiGLU 双投影
    # MegaMoeA8W8Wave 的 IsGmm1Interleaved 模板参数 (kernel 两条路径都已实现):
    #   False (缺省, 非交织): 调度宽度 = hidden_dim/activation_n_half -> 18 个 n-tile,
    #     每 tile 跑 ACTIVATION_N_HALF 遍权重块 (gate + up 各一次 mmad),
    #     GMM1 每算完一个 tile 就等配对 AIV0 读走 UB (vecSetSyncCom = 1, 深度 1)。
    #   True (交织): 调度宽度 = hidden_dim -> 36 个 n-tile, gate/up 在 tile 内按列交织,
    #     每 tile 一遍 mmad, 产出的 epilogue 宽度是 tile 的一半 (epilogueN = N/2);
    #     UB 两块缓冲 ping-pong, 攒到 2 个在飞才等 (vecSetSyncCom >= 2, 深度 2)。
    # 出处: stage/mega_moe_gmm1_activation.h:251-285 (同步分支),
    #       mega_moe_wave_a8w8.h:446 (调度宽度), 同文件 377-392 (epilogueN = N/2)。
    gmm1_interleaved: bool = False
    l1_tile_k: int = 256              # K-chunk 基线 (select_kl1 自适应)
    # GMM1 B 复用: 切片内首个 m-group 的 tile 付整份 B 流, 其余 m-group 的 tile 各付
    # **本比例**。1.0 = 不复用 (缺省, 不声称 L2 会命中); 0.0 = 完全复用 (只有首个付)。
    #
    # **影响时长** —— 进 gmm1_tile 的 b_load (costs.py)。实测只有一个点:
    #   bs128  m=256, 1 个 m-group, 28 核: 74.81 us
    #   bs8192 m=256,12 个 m-group, 28 核: 53.80 us   <- 几何相同, 只差 m-group 数
    # 反解每 tile 的 B 流只有整份的 56.5%; 按"首个付整份、其余各付 f"算, G=12 时
    # f ≈ 0.53。**一个点不是规律** (f 可能随 G、随列块数变), 所以缺省不声称复用。
    # 注意: 每专家只有 1 个 m-group 时本旋钮无效 (没有可复用的对象)。
    gmm1_b_reuse_frac: float = 1.0
    combine_quant_mode: int = 0       # CombineQuantMode: 0=NO_QUANT, 1=QUANT(FP8+scale)。
                                      # 只管**数据格式**; combine 跑在哪个角色/什么粒度
                                      # 是编排, 模型只能表达一种 (见 docs 缺口 11)
    # COMBINE 每行要读的路由元数据字节。**搬几个字段是编排选择**:
    #   12 = 算法下界 (combine 只需 dstRankId / tokenIdx / topkIdx 三项)
    #   16 = 缺省 (四个具名字段)
    #   32 = 某实现的取值 (DataCopy 搬满 META_INFO_SIZE=8 个 int32 槽;
    #        与同仓 DispatchDataLayout.META_BYTES_PER_ROW 一致)
    combine_meta_bytes_per_row: int = 16
    l1_size: int = 512 * 1024         # DAV_3510 平台
    # 核数 aic_num 是可调场景输入 (MegaMoeShape.aic_num); 向量核数恒为 2×aic_num
    # (每 block 1 AIC + 2 AIV, 平台结构常数), 模型以每核 AIV0/AIV1 两角色表达,
    # 不设独立旋钮。
