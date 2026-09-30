/*
 * 访存常数微基准 (单卡, 无通信)。给 moe_cost_model 定死几个目前只能"假设"的常数。
 *
 * 设计原则 —— 每个 case 只隔离一条数据通路, 并且**同时扫两个轴**:
 *   1. 尺寸轴: 分离"每次请求的固定开销"与"每字节的代价" (截距 vs 斜率)
 *   2. 并发轴 (activeCores): 分离"单核独占速率"与"整卡聚合带宽"
 * 第 2 个轴是关键 —— 现有常数全都是在 28 核并发下反解的单核值, 已含平均争用,
 * 再叠到速率服务器上就是双重计费。只有显式扫并发才能把两者分开。
 *
 * 不做任何拟合: 内核只吐每 (case, 核, 重复) 的原始 cycle, 分析在 analyze.py 里做。
 *
 * 输出布局 (uint64 数组, 行主序):
 *   out[((caseIdx * MAX_REPS) + rep) * MAX_CORES + core] = cycles
 * cycle 为 0 表示该核未参与 (blockIdx >= activeCores)。
 *
 * 注意: 本文件未在 NPU 上编译验证过 (写它的机器没有 CANN)。若有编译错, 大概率是
 * DataCopyPad 的 ExtParams 字段名/顺序或 LocalTensor 直址构造在你的 CANN 版本上略有
 * 差异 —— 这两处都照抄了 mega_moe/op_kernel/arch35 里的现成用法, 以它为准。
 */
#include "kernel_operator.h"

using namespace AscendC;

// ---------------------------------------------------------------- 常量
constexpr uint32_t MAX_CORES = 64;
constexpr uint32_t MAX_REPS = 8;
constexpr uint32_t N_CASES = 6;

// UB 直址布局 (照抄 arch35 里 LocalTensor(TPosition::VECCALC, 字节偏移, 元素数) 的用法)
constexpr uint32_t UB_SRC = 0;              // 源缓冲
constexpr uint32_t UB_DST = 96 * 1024;      // 目的缓冲
constexpr uint32_t UB_BYTES_MAX = 64 * 1024;

// L1 直址 (A1 位置), 给 GM->L1 用
constexpr uint32_t L1_DST = 0;

struct BenchParams {
    uint32_t activeCores;    // 参与计时的核数; 其余核只参与栅栏
    uint32_t reps;           // 每 case 重复次数 (取中位, 首次含冷启动)
    uint32_t bytes;          // case 0/1/2/4: 连续搬运字节数
    uint32_t rows;           // case 3: blockCount (突发次数)
    uint32_t rowBytes;       // case 3: 每次突发字节数
    uint32_t rowStride;      // case 3: 目的地址相邻突发的字节间距
    uint32_t vecElems;       // case 5: 向量元素数 (fp32)
    uint32_t pad;
};

// ---------------------------------------------------------------- 跨核栅栏
// GM 计数器 + 自旋。每个 case 前后各一次, 保证"并发"是真的并发。
__aicore__ inline void Barrier(__gm__ uint32_t *counter, uint32_t generation,
                               uint32_t totalCores)
{
    if (GetSubBlockIdx() != 0U) {
        return;
    }
    GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(counter));
    SetAtomicAdd<int32_t>();
    // 每代用一个独立槽, 避免复用时的 ABA
    LocalTensor<int32_t> one(TPosition::VECCALC, UB_DST, 8);
    one.SetValue(0, 1);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopyExtParams p{1U, sizeof(int32_t), 0U, 0U, 0U};
    DataCopyPad(g[generation * 32], one, p);
    SetAtomicNone();
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    __gm__ int32_t *slot = reinterpret_cast<__gm__ int32_t *>(counter) + generation * 32;
    while (static_cast<uint32_t>(ReadGmByPassDCache(slot)) < totalCores) {
    }
}

// ---------------------------------------------------------------- 各 case
// case 0: GM -> UB 连续读 (MTE2)。定"读本卡内存"的单核速率与整卡聚合。
__aicore__ inline void CaseGmToUb(GlobalTensor<int8_t> &src, uint32_t bytes)
{
    LocalTensor<int8_t> dst(TPosition::VECCALC, UB_SRC, UB_BYTES_MAX);
    DataCopyExtParams p{1U, bytes, 0U, 0U, 0U};
    DataCopyPadExtParams<int8_t> pad{false, 0U, 0U, 0U};
    DataCopyPad(dst, src, p, pad);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

// case 1: GM -> L1 连续读 (MTE2)。GMM 的权重/激活载入通路, 定 BW_L1_GM。
__aicore__ inline void CaseGmToL1(GlobalTensor<int8_t> &src, uint32_t bytes)
{
    LocalTensor<int8_t> dst(TPosition::A1, L1_DST, bytes);
    DataCopy(dst, src, bytes);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

// case 2: UB -> GM 连续写 (MTE3)。定本卡写带宽 (hbm_write 聚合的来源)。
__aicore__ inline void CaseUbToGm(GlobalTensor<int8_t> &dst, uint32_t bytes)
{
    LocalTensor<int8_t> src(TPosition::VECCALC, UB_SRC, UB_BYTES_MAX);
    DataCopyExtParams p{1U, bytes, 0U, 0U, 0U};
    DataCopyPad(dst, src, p);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

// case 3: UB -> GM 带 stride 的多次突发 (MTE3)。
// 扫 rows 与 rowBytes 两个轴, 分离"每次突发的固定开销"与"每字节代价" ——
// 这正是 ACT 的 StoreQuantScaleCompact (每次只 8 字节!) 与 COMBINE 的
// CombineTokens (每行一次) 的形态。
__aicore__ inline void CaseStridedStore(GlobalTensor<int8_t> &dst, uint32_t rows,
                                        uint32_t rowBytes, uint32_t rowStride)
{
    LocalTensor<int8_t> src(TPosition::VECCALC, UB_SRC, UB_BYTES_MAX);
    // blockCount = rows, blockLen = rowBytes, dstStride = 相邻突发的间距 - rowBytes
    DataCopyExtParams p{static_cast<uint16_t>(rows), rowBytes, 0U,
                        rowStride > rowBytes ? (rowStride - rowBytes) : 0U, 0U};
    DataCopyPad(dst, src, p);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

// case 4: UB -> GM 逐次单发 (rows 次独立 DataCopyPad, 每次 rowBytes)。
// 与 case 3 对照: case 3 是一条指令里 blockCount 次突发, case 4 是 rows 条指令。
// CombineTokens 是 case 4 的形态 (每行一次 DataCopyPad, 目标地址不规则)。
__aicore__ inline void CaseScatterStore(GlobalTensor<int8_t> &dst, uint32_t rows,
                                         uint32_t rowBytes, uint32_t rowStride)
{
    LocalTensor<int8_t> src(TPosition::VECCALC, UB_SRC, UB_BYTES_MAX);
    DataCopyExtParams p{1U, rowBytes, 0U, 0U, 0U};
    for (uint32_t r = 0; r < rows; ++r) {
        DataCopyPad(dst[static_cast<uint64_t>(r) * rowStride], src, p);
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

// case 5: 纯向量运算 (无访存)。判 ACT 到底是 UB 带宽绑定还是 FLOP 绑定 ——
// 子代理审计留下的未决问题。做 SwiGLU 里的那串: Exp/Div/Mul。
__aicore__ inline void CaseVectorOnly(uint32_t elems)
{
    LocalTensor<float> a(TPosition::VECCALC, UB_SRC, UB_BYTES_MAX / 4);
    LocalTensor<float> b(TPosition::VECCALC, UB_DST, UB_BYTES_MAX / 4);
    Exp(b, a, elems);
    PipeBarrier<PIPE_V>();
    Div(b, b, a, elems);
    PipeBarrier<PIPE_V>();
    Mul(b, b, a, elems);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
}

// ---------------------------------------------------------------- 入口
extern "C" __global__ __aicore__ void BenchMem(GM_ADDR paramsGm, GM_ADDR srcGm,
                                               GM_ADDR dstGm, GM_ADDR outGm,
                                               GM_ADDR barrierGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__NPU_ARCH__)
    InitSocState();
    BenchParams prm;
    {
        auto s = reinterpret_cast<__gm__ uint32_t *>(paramsGm);
        auto d = reinterpret_cast<uint32_t *>(&prm);
        for (uint32_t i = 0; i < sizeof(prm) / 4; ++i) {
            d[i] = s[i];
        }
    }
    const uint32_t core = static_cast<uint32_t>(GetBlockIdx() / GetTaskRation());
    const uint32_t totalCores = static_cast<uint32_t>(GetBlockNum());
    const bool active = core < prm.activeCores;
    const uint32_t reps = prm.reps < MAX_REPS ? prm.reps : MAX_REPS;

    GlobalTensor<int8_t> src, dst;
    // 每核一块独立区域, 避免地址冲突掩盖真实带宽
    src.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(srcGm)
                        + static_cast<uint64_t>(core) * 8 * 1024 * 1024);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(dstGm)
                        + static_cast<uint64_t>(core) * 8 * 1024 * 1024);
    GlobalTensor<uint64_t> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(outGm));
    auto barrier = reinterpret_cast<__gm__ uint32_t *>(barrierGm);

    uint32_t gen = 0;
    for (uint32_t c = 0; c < N_CASES; ++c) {
        for (uint32_t rep = 0; rep < reps; ++rep) {
            Barrier(barrier, gen++, totalCores);
            uint64_t t0 = GetSystemCycle();
            if (active && GetSubBlockIdx() == 0U) {
                if (c == 0) {
                    CaseGmToUb(src, prm.bytes);
                } else if (c == 1) {
                    CaseGmToL1(src, prm.bytes);
                } else if (c == 2) {
                    CaseUbToGm(dst, prm.bytes);
                } else if (c == 3) {
                    CaseStridedStore(dst, prm.rows, prm.rowBytes, prm.rowStride);
                } else if (c == 4) {
                    CaseScatterStore(dst, prm.rows, prm.rowBytes, prm.rowStride);
                } else {
                    CaseVectorOnly(prm.vecElems);
                }
            }
            uint64_t t1 = GetSystemCycle();
            if (GetSubBlockIdx() == 0U && core < MAX_CORES) {
                uint64_t idx = (static_cast<uint64_t>(c) * MAX_REPS + rep) * MAX_CORES + core;
                __gm__ int64_t *slot = reinterpret_cast<__gm__ int64_t *>(outGm) + idx;
                WriteGmByPassDCache(slot, static_cast<int64_t>(active ? (t1 - t0) : 0));
            }
        }
    }
    Barrier(barrier, gen, totalCores);
#endif
}

extern "C" void bench_mem_launch(uint32_t blocks, void *stream, uint8_t *params,
                                 uint8_t *src, uint8_t *dst, uint8_t *out,
                                 uint8_t *barrier)
{
    BenchMem<<<blocks, nullptr, stream>>>(params, src, dst, out, barrier);
}
