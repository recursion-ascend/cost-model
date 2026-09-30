#include "kernel_operator.h"
#if defined(__NPU_ARCH__)
#include "profiler.h"
#include "case_config.h"
#include "mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h"
#endif
using namespace AscendC;
extern "C" __global__ __aicore__ void MegaMoeProfile(
    GM_ADDR context, GM_ADDR x, GM_ADDR ids, GM_ADDR gates, GM_ADDR w1, GM_ADDR w2,
    GM_ADDR s1, GM_ADDR s2, GM_ADDR y, GM_ADDR counts, GM_ADDR workspace, GM_ADDR tiling,
    GM_ADDR sw1, GM_ADDR sw2, GM_ADDR ss1, GM_ADDR ss2)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__NPU_ARCH__)
    InitSocState();
    MegaMoeTilingData td;
    static_assert(sizeof(td) % 4 == 0);
    auto src = reinterpret_cast<__gm__ uint32_t*>(tiling);
    auto dst = reinterpret_cast<uint32_t*>(&td);
    for (uint32_t i = 0; i < sizeof(td) / 4; ++i) dst[i] = src[i];
    MOE_PROFILE_BIND(td.profBufGm);
    PROF_INIT(td.profBufGm);
    MOE_PROFILE_BEGIN(KERNEL, 0);
#ifndef MEGAMOE_TOPK_PREFETCH
#define MEGAMOE_TOPK_PREFETCH 0
#endif
    MegaMoeImpl::MegaMoeA8W8Wave<bfloat16_t,bfloat16_t,bfloat16_t,PROFILE_WEIGHT_TYPE,
        MegaMoeImpl::PROFILE_QUANT,MegaMoeImpl::COMBINE_NO_QUANT,MEGAMOE_TOPK_PREFETCH,false> op;
    MOE_PROFILE_BEGIN(INIT, 0);
    op.Init(context,x,ids,gates,w1,w2,nullptr,s1,s2,nullptr,sw1,sw2,ss1,ss2,
            y,counts,workspace,&td,tiling);
    MOE_PROFILE_END(INIT, 0);
    op.Process();
    MOE_PROFILE_END(KERNEL, 0);
#endif
}
extern "C" void megamoe_profile_launch(uint32_t blocks, void *stream,
    uint8_t *context,uint8_t *x,uint8_t *ids,uint8_t *gates,uint8_t *w1,uint8_t *w2,
    uint8_t *s1,uint8_t *s2,uint8_t *y,uint8_t *counts,uint8_t *workspace,uint8_t *tiling,
    uint8_t *sw1,uint8_t *sw2,uint8_t *ss1,uint8_t *ss2)
{
    MegaMoeProfile<<<blocks,nullptr,stream>>>(context,x,ids,gates,w1,w2,s1,s2,y,counts,workspace,tiling,sw1,sw2,ss1,ss2);
}
