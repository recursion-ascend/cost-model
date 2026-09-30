#include <algorithm>
#include "case_config.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include "mc2/mega_moe/op_kernel/arch35/common/mega_moe_workspace.h"
#include "tiling/platform/platform_ascendc.h"
using namespace MegaMoeImpl;
namespace ops = Ops::Base;
#ifndef MEGAMOE_TILE_N
#define MEGAMOE_TILE_N 256U
#endif
constexpr uint32_t GMM_TILE_N = MEGAMOE_TILE_N;
#ifndef MEGAMOE_TOPO_URMA
#define MEGAMOE_TOPO_URMA 0
#endif
constexpr int64_t PROFILE_TOPO_TYPE = MEGAMOE_TOPO_URMA ? TOPO_TYPE_URMA : TOPO_TYPE_MTE;
// 每核保底逻辑 N tile 数：随 bs 分档（与 op_host/op_tiling/arch35/mega_moe_tiling.cpp 保持同步）。
constexpr uint32_t GMM1_MIN_LOGICAL_TILES_PER_CORE = 4;
constexpr uint32_t GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL = 2;
constexpr uint32_t GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE = 6;
constexpr uint32_t GMM1_SMALL_BATCH_TOKEN_THRESHOLD = 2048;
constexpr uint32_t GMM1_LARGE_BATCH_TOKEN_THRESHOLD = 16384;
// p2：GMM2 每核保底逻辑 N tile 数，历史公式隐含 1，显式化（与 op_host 保持同步）。
constexpr uint32_t GMM2_MIN_LOGICAL_TILES_PER_CORE = 1;
#include "tiling_helpers.h"
extern "C" uint64_t megamoe_tiling_size() {return sizeof(MegaMoeTilingData);}
extern "C" uint64_t megamoe_peermem_size() {
    PeermemSizeParams params{PROFILE_TOKENS, PROFILE_TOPK, PROFILE_HIDDEN,
        PROFILE_EXPERTS / PROFILE_EP, PROFILE_EP, 2, 1, false, false, PROFILE_TOPO_TYPE, 1};
    return EXCEPTION_DUMP_REGION_SIZE + CalcPeermemLeastSize(params);
}
extern "C" uint64_t megamoe_make_tiling(void *out, uint64_t prof) {
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(PROFILE_PLATFORM);
    if (!platform) return 0;
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    uint32_t aic=platform->GetCoreNumAic(), aiv=platform->GetCoreNumAiv();
    std::printf("Platform aic=%u aiv=%u UB=%lu\n",aic,aiv,ub);
    if (aic != 28 || aiv != 56 || ub == 0) return 0;
    MegaMoeTilingData td{};
    td.bs=PROFILE_TOKENS; td.numMaxTokensPerRank=PROFILE_TOKENS; td.h=PROFILE_HIDDEN; td.hiddenDim=2*PROFILE_INTERMEDIATE;
    td.moeExpertPerRank=PROFILE_EXPERTS/PROFILE_EP; td.epWorldSize=PROFILE_EP; td.topK=PROFILE_TOPK;
    td.sharedExpertNum=PROFILE_SHARED_EXPERTS;
    td.aicNum=aic;td.blockAivNum=aiv;td.blockNumPerEP=aic/td.epWorldSize;
    td.maxOutputSize=td.bs*td.epWorldSize*std::min(td.topK,td.moeExpertPerRank);
    td.combineQuantMode=COMBINE_NO_QUANT; td.clampLimit=std::numeric_limits<float>::max();
    td.groupedMatmulMode=GROUPED_MATMUL_MODE_GENERAL; td.topoType=PROFILE_TOPO_TYPE;
    td.actMode=static_cast<uint8_t>(MegaMoeActMode::SWIGLU);
    td.activationAlpha=1;td.activationBeta=1;td.rankNumPerServer=PROFILE_EP;
    // cost model 标定用：环境变量显式覆盖每核保底逻辑 N tile 数 p1/p2（>0 生效）。
    // 不设置时 p1 走 bs 分档、p2 取默认 1，与 op_host 源码默认行为一致；
    // requested=0 表示 use default；effective 为最终生效值。
    int requestedP1 = 0;
    int requestedP2 = 0;
    if (const char *p1Env = std::getenv("MEGAMOE_P1_OVERRIDE")) { requestedP1 = std::atoi(p1Env); }
    if (const char *p2Env = std::getenv("MEGAMOE_P2_OVERRIDE")) { requestedP2 = std::atoi(p2Env); }
    uint32_t effectiveP1 = requestedP1 > 0 ? static_cast<uint32_t>(requestedP1)
                                           : ResolveGmm1MinLogicalTilesPerCore(td.bs);
    uint32_t effectiveP2 = requestedP2 > 0 ? static_cast<uint32_t>(requestedP2)
                                           : GMM2_MIN_LOGICAL_TILES_PER_CORE;
    MegaMoeWavePolicy wave{};
    td.mGroupsPerWave=CalcMGroupsPerWave(&td,aic,effectiveP1,effectiveP2,&wave);
    const char *dominantSide = wave.gmm1RequiredMGroups > wave.gmm2RequiredMGroups ? "GMM1"
                               : (wave.gmm2RequiredMGroups > wave.gmm1RequiredMGroups ? "GMM2" : "TIE");
    std::printf("MegaMoe wave policy: aicNum=%u hiddenDim=%u h=%u "
                "requested_p1=%d requested_p2=%d effective_p1=%u effective_p2=%u "
                "gmm1TilesPerMGroup=%lu gmm2TilesPerMGroup=%lu "
                "gmm1RequiredMGroups=%lu gmm2RequiredMGroups=%lu "
                "mGroupsPerWave=%u dominant_side=%s\n",
                aic,td.hiddenDim,td.h,requestedP1,requestedP2,effectiveP1,effectiveP2,
                wave.gmm1LogicalTilesPerMGroup,wave.gmm2TilesPerMGroup,
                wave.gmm1RequiredMGroups,wave.gmm2RequiredMGroups,
                td.mGroupsPerWave,dominantSide);
    // cost model 标定用：环境变量覆盖自动 wave 容量（>0 生效），最终值以下方打印为准。
    // p1/p2 实验不使用该机制，保留给 wave 容量直接标定。
    if (const char *mgwEnv = std::getenv("MEGAMOE_MGW_OVERRIDE")) {
        int mgw = std::atoi(mgwEnv);
        if (mgw > 0) {
            td.mGroupsPerWave = static_cast<uint32_t>(mgw);
            std::printf("MGW_OVERRIDE=%d\n", mgw);
        }
    }
    td.dispatchBufferConfig=CalcDispatchBufferConfig(&td,1,ub);
    td.combineSyncSlotCountPerExpert=CalcCombineSyncSlotCountPerExpert(&td);
    SetTopkValidIndexBufferConfigs(&td,1,ub);
    SetUnpermuteBufferConfigs(&td,2,ub);
    td.profBufGm=prof;
    std::memcpy(out,&td,sizeof(td));
    WorkspaceLayout layout(&td);
    std::printf("Tiling bytes=%lu workspace=%ld maxOutput=%u mGroupsPerWave=%u\n",sizeof(td),layout.workspaceSize,td.maxOutputSize,td.mGroupsPerWave);
    return layout.workspaceSize;
}
