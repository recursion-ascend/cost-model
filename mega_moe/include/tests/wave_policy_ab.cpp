/*
 * A/B regression: patched CalcMGroupsPerWave (real tiling_helpers.h, parameterized
 * p1/p2) vs the HEAD@ebf6f3a reference formula (verbatim copy below), over the same
 * grid as scripts/wave_policy_regression.py. CPU-only; no NPU required.
 *
 * Build with the same include set as the megamoe_host CMake target (see
 * megamoe_profile/CMakeLists.txt INCS). Run: exit 0 iff no mismatch.
 */
#include <cstdint>
#include <cstdio>

#include "mc2/mega_moe/op_kernel/arch35/common/mega_moe_workspace.h"

using namespace MegaMoeImpl;
namespace ops = Ops::Base;

#ifndef MEGAMOE_TILE_N
#define MEGAMOE_TILE_N 256U
#endif
constexpr uint32_t GMM_TILE_N = MEGAMOE_TILE_N;
// 与 op_host/op_tiling/arch35/mega_moe_tiling.cpp 保持同步（含 p2 显式化）。
constexpr uint32_t GMM1_MIN_LOGICAL_TILES_PER_CORE = 4;
constexpr uint32_t GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL = 2;
constexpr uint32_t GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE = 6;
constexpr uint32_t GMM1_SMALL_BATCH_TOKEN_THRESHOLD = 2048;
constexpr uint32_t GMM1_LARGE_BATCH_TOKEN_THRESHOLD = 16384;
constexpr uint32_t GMM2_MIN_LOGICAL_TILES_PER_CORE = 1;
#include "tiling_helpers.h"

namespace {
/*
 * HEAD@ebf6f3a 参考实现（逐字复制，作为回归基准）：
 * 分档 p1 + 隐含 p2=1（G2 = ceil(aicNum / gmm2TilesPerMGroup)）。
 */
uint32_t CalcMGroupsPerWaveRef(const MegaMoeTilingData *tilingData, uint32_t aicNum)
{
    if (tilingData->hiddenDim == 0U || tilingData->h == 0U || aicNum == 0U) {
        return 1U;
    }

    uint32_t minTilesPerCore = GMM1_MIN_LOGICAL_TILES_PER_CORE;
    if (tilingData->bs > 0U && tilingData->bs < GMM1_SMALL_BATCH_TOKEN_THRESHOLD) {
        minTilesPerCore = GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL;
    } else if (tilingData->bs >= GMM1_LARGE_BATCH_TOKEN_THRESHOLD) {
        minTilesPerCore = GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE;
    }

    uint64_t gmm1LogicalTilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->hiddenDim, GMM_TILE_N);
    uint64_t gmm2TilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->h, GMM_TILE_N);
    uint64_t gmm1RequiredMGroups = ops::CeilDiv<uint64_t>(
        static_cast<uint64_t>(aicNum) * minTilesPerCore, gmm1LogicalTilesPerMGroup);
    uint64_t gmm2RequiredMGroups = ops::CeilDiv<uint64_t>(static_cast<uint64_t>(aicNum), gmm2TilesPerMGroup);
    return static_cast<uint32_t>(std::max(gmm1RequiredMGroups, gmm2RequiredMGroups));
}

const uint32_t BS_VALUES[] = {1U, 64U, 1024U, 2047U, 2048U, 2049U, 5000U, 8192U, 16383U, 16384U, 16385U, 65536U, 0U};
const uint32_t H_VALUES[] = {1024U, 2048U, 4096U, 5120U, 6144U, 7168U, 8192U, 1056U};
const uint32_t HIDDEN_DIM_VALUES[] = {512U, 1024U, 2048U, 2560U, 4096U, 5120U, 6144U, 8192U, 576U};
const uint32_t AIC_VALUES[] = {1U, 2U, 4U, 8U, 16U, 20U, 24U, 28U, 32U, 48U, 56U, 64U};
} // namespace

int main()
{
    uint32_t cases = 0U;
    uint32_t bad = 0U;
    for (uint32_t bs : BS_VALUES) {
        for (uint32_t h : H_VALUES) {
            for (uint32_t hiddenDim : HIDDEN_DIM_VALUES) {
                for (uint32_t aic : AIC_VALUES) {
                    MegaMoeTilingData td{};
                    td.bs = bs;
                    td.h = h;
                    td.hiddenDim = hiddenDim;
                    uint32_t want = CalcMGroupsPerWaveRef(&td, aic);
                    uint32_t got = CalcMGroupsPerWave(&td, aic, ResolveGmm1MinLogicalTilesPerCore(td.bs),
                                                      GMM2_MIN_LOGICAL_TILES_PER_CORE);
                    ++cases;
                    if (want != got) {
                        ++bad;
                        if (bad <= 20U) {
                            std::printf("MISMATCH bs=%u h=%u hiddenDim=%u aic=%u: ref=%u new=%u\n", bs, h, hiddenDim,
                                        aic, want, got);
                        }
                    }
                }
            }
        }
    }

    // 显式 p1/p2 覆盖路径抽查：p1=4,p2=1 必须与“固定 4”基准一致（与 Python check3 对应）。
    uint32_t badOverride = 0U;
    for (uint32_t bs : BS_VALUES) {
        for (uint32_t h : H_VALUES) {
            for (uint32_t hiddenDim : HIDDEN_DIM_VALUES) {
                for (uint32_t aic : AIC_VALUES) {
                    MegaMoeTilingData td{};
                    td.bs = bs;
                    td.h = h;
                    td.hiddenDim = hiddenDim;
                    // 固定 p1=4 基准：不分档直接用默认档常量。
                    uint32_t t1 = static_cast<uint32_t>(ops::CeilDiv<uint64_t>(hiddenDim, GMM_TILE_N));
                    uint32_t t2 = static_cast<uint32_t>(ops::CeilDiv<uint64_t>(h, GMM_TILE_N));
                    uint32_t base = static_cast<uint32_t>(std::max(ops::CeilDiv<uint64_t>(aic * 4U, t1),
                                                                   ops::CeilDiv<uint64_t>(aic, t2)));
                    uint32_t got = CalcMGroupsPerWave(&td, aic, 4U, 1U);
                    if (base != got) {
                        ++badOverride;
                    }
                }
            }
        }
    }

    // 历史锚点：bs=8192/H=4096/aic=28 -> 7；固定 case 默认 bs=64/H=6144/hiddenDim=4096 -> 4。
    MegaMoeTilingData anchor1{};
    anchor1.bs = 8192U;
    anchor1.h = 4096U;
    anchor1.hiddenDim = 4096U;
    uint32_t anchor1Got = CalcMGroupsPerWave(&anchor1, 28U, ResolveGmm1MinLogicalTilesPerCore(anchor1.bs),
                                             GMM2_MIN_LOGICAL_TILES_PER_CORE);
    MegaMoeTilingData anchor2{};
    anchor2.bs = 64U;
    anchor2.h = 6144U;
    anchor2.hiddenDim = 4096U;
    uint32_t anchor2Got = CalcMGroupsPerWave(&anchor2, 28U, ResolveGmm1MinLogicalTilesPerCore(anchor2.bs),
                                             GMM2_MIN_LOGICAL_TILES_PER_CORE);

    std::printf("C++ A/B regression: grid=%u default_mismatches=%u override_mismatches=%u\n", cases, bad, badOverride);
    std::printf("anchors: bs8192_H4096_aic28 -> %u (expect 7); bs64_H6144_hd4096_aic28 -> %u (expect 4)\n", anchor1Got,
                anchor2Got);
    bool ok = bad == 0U && badOverride == 0U && anchor1Got == 7U && anchor2Got == 4U;
    std::printf("result: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
