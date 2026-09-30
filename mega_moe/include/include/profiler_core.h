/**
 * \file profiler_core.h
 * \brief Device-side profiling mechanism — GM ring buffer write + DCCI flush.
 *
 * Contains buffer layout constants, GmWrite helpers, Profiler::Init/Mark,
 * and PROF_INIT/PROF_MARK macros. NO event IDs — those belong in your
 * operator-specific profiler.h alongside this file.
 *
 * Include chain: profiler.h → profiler_core.h → profiler_config.h
 *
 * When ENABLE_PROFILING is not defined, PROF_INIT and PROF_MARK expand
 * to ((void)0) — zero overhead.
 */

#pragma once

#include "kernel_basic_intf.h"

#ifdef ENABLE_PROFILING

#include "profiler_config.h"

static_assert(PROF_N_CORES > 0, "PROF_N_CORES must be > 0. Check profiler_config.h");

constexpr uint32_t PROF_RINGIDX_REGION = PROF_N_CORES * PROF_RINGIDX_ALIGN;
constexpr uint32_t PROF_DATA_REGION    = PROF_N_CORES * PROF_SLOTS_PER_CORE * PROF_RECORD_BYTES;
constexpr uint32_t PROF_BUF_TOTAL      = PROF_RINGIDX_REGION + PROF_DATA_REGION;


namespace Profiler {

__aicore__ inline uint32_t GetCoreId() {
    uint32_t id = AscendC::GetBlockIdx();
    // AIC cores are numbered after AIV cores in the profiler ring buffer.
    // PROF_AIV_CORES = 0 means AIV-only mode (no AIC cores).
#if PROF_AIV_CORES > 0
    if ASCEND_IS_AIC {
        id += PROF_AIV_CORES;
    }
#endif
    return id;
}

// Write a 32-bit value to GM, then flush DCCI to make it visible to host.
// Same pattern as HCCL ReadAndFlipWinFlag.
__aicore__ inline void GmWrite32(__gm__ uint32_t* addr, uint32_t value) {
    AscendC::GlobalTensor<uint32_t> gt;
    gt.SetGlobalBuffer(addr, 1);
    gt.SetValue(0, value);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(gt);
}

// Write a 64-bit value to GM. Split into two 32-bit SetValue + DCCI calls
// so the compiler pipeline stays happy.
__aicore__ inline void GmWrite64(__gm__ uint32_t* addr, uint64_t value) {
    uint32_t lo = static_cast<uint32_t>(value);
    uint32_t hi = static_cast<uint32_t>(value >> 32);
    AscendC::GlobalTensor<uint32_t> gt;
    gt.SetGlobalBuffer(addr, 2);
    gt.SetValue(0, lo);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(gt);
    gt.SetValue(1, hi);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(gt);
}

__aicore__ inline void Init(__gm__ uint8_t* profBufGm) {
    if (profBufGm == (__gm__ uint8_t*)0) return;
    uint32_t coreId = GetCoreId();
    __gm__ uint32_t* ringIdxPtr = (__gm__ uint32_t*)(profBufGm + coreId * PROF_RINGIDX_ALIGN);
    GmWrite32(ringIdxPtr, 0);
}

__aicore__ inline void Mark(__gm__ uint8_t* profBufGm, uint32_t eventId, uint32_t payload = 0) {
    if (profBufGm == (__gm__ uint8_t*)0) return;
    uint32_t coreId = GetCoreId();
    __gm__ uint32_t* ringIdxPtr = (__gm__ uint32_t*)(profBufGm + coreId * PROF_RINGIDX_ALIGN);

    AscendC::GlobalTensor<uint32_t> ringGt;
    ringGt.SetGlobalBuffer(ringIdxPtr, 1);
    uint32_t idx = ringGt.GetValue(0);

    if (idx >= PROF_SLOTS_PER_CORE) {
        return;
    }

    uint64_t cycle = AscendC::GetSystemCycle();
    __gm__ uint32_t* slot = (__gm__ uint32_t*)(profBufGm + PROF_RINGIDX_REGION +
                                coreId * PROF_SLOTS_PER_CORE * PROF_RECORD_BYTES +
                                idx * PROF_RECORD_BYTES);

    // Write event record: [eventId, payload, cycle_lo, cycle_hi]
    GmWrite32(&slot[0], eventId);
    GmWrite32(&slot[1], payload);
    GmWrite64(&slot[2], cycle);

    // Increment ringIdx
    GmWrite32(ringIdxPtr, idx + 1);
}

} // namespace Profiler

#define PROF_INIT(buf)           Profiler::Init((__gm__ uint8_t*)(buf))
#define PROF_MARK(buf, evt, ...) Profiler::Mark((__gm__ uint8_t*)(buf), (evt), ##__VA_ARGS__)

#else  // !ENABLE_PROFILING

#define PROF_INIT(buf)           ((void)0)
#define PROF_MARK(buf, evt, ...) ((void)0)

#endif
