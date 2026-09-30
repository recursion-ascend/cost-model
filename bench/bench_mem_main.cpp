/*
 * bench_mem 的主机驱动: 扫参数网格, 每个点跑一次内核, 把原始 cycle 写成 CSV。
 * 不做任何拟合 —— 分析在 bench/analyze.py 里做。
 *
 * 用法:  ./bench_mem [device] > bench_mem.csv
 *
 * 网格设计 (为什么这么扫):
 *   连续搬运 (case 0/1/2): 尺寸 x 并发 —— 尺寸给截距/斜率, 并发给单核 vs 整卡
 *   突发/散射 (case 3/4):  rows x rowBytes x 并发 —— 分离每次请求的固定开销与每字节
 *                          代价。ACT 的 scale 写每次只 8B, COMBINE 每行一次 512B,
 *                          都落在这个网格里
 *   纯向量  (case 5):      元素数 x 并发 —— 判 ACT 是带宽绑定还是 FLOP 绑定
 */
#include <acl/acl.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void bench_mem_launch(uint32_t blocks, void *stream, uint8_t *params,
                                 uint8_t *src, uint8_t *dst, uint8_t *out,
                                 uint8_t *barrier);

namespace {

constexpr uint32_t MAX_CORES = 64;
constexpr uint32_t MAX_REPS = 8;
constexpr uint32_t N_CASES = 6;
constexpr uint32_t BLOCKS = 28;          // 与 aic 数一致
constexpr uint64_t PER_CORE_REGION = 8ull * 1024 * 1024;
constexpr uint64_t BUF_BYTES = PER_CORE_REGION * MAX_CORES;
constexpr uint64_t OUT_SLOTS = static_cast<uint64_t>(N_CASES) * MAX_REPS * MAX_CORES;
constexpr uint64_t BARRIER_SLOTS = 32ull * (N_CASES * MAX_REPS + 4);

struct BenchParams {
    uint32_t activeCores;
    uint32_t reps;
    uint32_t bytes;
    uint32_t rows;
    uint32_t rowBytes;
    uint32_t rowStride;
    uint32_t vecElems;
    uint32_t pad;
};

#define CHECK(expr)                                                              \
    do {                                                                         \
        aclError _e = (expr);                                                    \
        if (_e != ACL_SUCCESS) {                                                 \
            std::fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #expr, \
                         static_cast<int>(_e));                                  \
            return 1;                                                            \
        }                                                                        \
    } while (0)

const char *CASE_NAME[N_CASES] = {
    "gm_to_ub",        // 读本卡内存 -> UB
    "gm_to_l1",        // 读本卡内存 -> L1 (GMM 载入通路)
    "ub_to_gm",        // 写本卡内存, 连续
    "strided_store",   // 一条指令 blockCount 次突发
    "scatter_store",   // rows 条独立指令
    "vector_only",     // 纯向量 Exp/Div/Mul
};

}  // namespace

int main(int argc, char **argv) {
    int32_t device = (argc > 1) ? std::atoi(argv[1]) : 0;
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(device));
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream));

    void *dParams = nullptr, *dSrc = nullptr, *dDst = nullptr, *dOut = nullptr,
         *dBar = nullptr;
    CHECK(aclrtMalloc(&dParams, sizeof(BenchParams), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dSrc, BUF_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dDst, BUF_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dOut, OUT_SLOTS * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dBar, BARRIER_SLOTS * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemset(dSrc, BUF_BYTES, 0, BUF_BYTES));

    std::vector<uint64_t> host(OUT_SLOTS);
    std::printf("case,active_cores,bytes,rows,row_bytes,row_stride,vec_elems,rep,core,cycles\n");

    // ---- 扫描网格 ----
    const uint32_t CORES[] = {1, 2, 4, 7, 14, 28};
    const uint32_t BYTES[] = {512, 2048, 5280, 8192, 32768, 65536};
    // (rows, rowBytes): ACT scale 写是 (m, 8); COMBINE 每行是 (m, 512);
    // ACT fp8 写是 (m, 256)。固定 rowBytes 扫 rows, 再固定 rows 扫 rowBytes。
    const uint32_t ROWS[] = {1, 8, 32, 72, 128, 256};
    const uint32_t ROWB[] = {8, 32, 128, 256, 512, 2048};
    const uint32_t VEC[] = {64, 256, 1024, 4096, 16384};

    auto run = [&](BenchParams p) -> int {
        CHECK(aclrtMemset(dBar, BARRIER_SLOTS * sizeof(uint32_t), 0,
                          BARRIER_SLOTS * sizeof(uint32_t)));
        CHECK(aclrtMemset(dOut, OUT_SLOTS * sizeof(uint64_t), 0,
                          OUT_SLOTS * sizeof(uint64_t)));
        CHECK(aclrtMemcpy(dParams, sizeof(p), &p, sizeof(p),
                          ACL_MEMCPY_HOST_TO_DEVICE));
        bench_mem_launch(BLOCKS, stream, static_cast<uint8_t *>(dParams),
                         static_cast<uint8_t *>(dSrc), static_cast<uint8_t *>(dDst),
                         static_cast<uint8_t *>(dOut), static_cast<uint8_t *>(dBar));
        CHECK(aclrtSynchronizeStream(stream));
        CHECK(aclrtMemcpy(host.data(), OUT_SLOTS * sizeof(uint64_t), dOut,
                          OUT_SLOTS * sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST));
        for (uint32_t c = 0; c < N_CASES; ++c) {
            for (uint32_t rep = 0; rep < p.reps; ++rep) {
                for (uint32_t core = 0; core < p.activeCores; ++core) {
                    uint64_t v = host[(static_cast<uint64_t>(c) * MAX_REPS + rep) *
                                      MAX_CORES + core];
                    if (v == 0) continue;
                    std::printf("%s,%u,%u,%u,%u,%u,%u,%u,%u,%llu\n", CASE_NAME[c],
                                p.activeCores, p.bytes, p.rows, p.rowBytes,
                                p.rowStride, p.vecElems, rep, core,
                                static_cast<unsigned long long>(v));
                }
            }
        }
        return 0;
    };

    BenchParams base{};
    base.reps = 5;
    base.bytes = 5280;
    base.rows = 72;
    base.rowBytes = 512;
    base.rowStride = 10240;
    base.vecElems = 1024;

    // 1) 尺寸 x 并发 (连续搬运)
    for (uint32_t nc : CORES) {
        for (uint32_t b : BYTES) {
            BenchParams p = base;
            p.activeCores = nc;
            p.bytes = b;
            if (run(p)) return 1;
        }
    }
    // 2) rows 扫 (固定 rowBytes) x 并发
    for (uint32_t nc : CORES) {
        for (uint32_t r : ROWS) {
            BenchParams p = base;
            p.activeCores = nc;
            p.rows = r;
            if (run(p)) return 1;
        }
    }
    // 3) rowBytes 扫 (固定 rows) x 并发
    for (uint32_t nc : CORES) {
        for (uint32_t rb : ROWB) {
            BenchParams p = base;
            p.activeCores = nc;
            p.rowBytes = rb;
            p.rowStride = rb * 2 > 64 ? rb * 2 : 64;
            if (run(p)) return 1;
        }
    }
    // 4) 向量元素数 x 并发
    for (uint32_t nc : CORES) {
        for (uint32_t v : VEC) {
            BenchParams p = base;
            p.activeCores = nc;
            p.vecElems = v;
            if (run(p)) return 1;
        }
    }

    aclrtFree(dParams); aclrtFree(dSrc); aclrtFree(dDst);
    aclrtFree(dOut); aclrtFree(dBar);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return 0;
}
