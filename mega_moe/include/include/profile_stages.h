#pragma once
#include <cstdint>

// Single source of stage names and stable BEGIN IDs. END = BEGIN + 1,
// except legacy KERNEL whose END ID is 0x00ff. Do not renumber existing IDs.
namespace MoeProfile {
enum MoeProfileStage : uint32_t {
    KERNEL = 0x0001,
    INPUT_QUANT = 0x6001,
    DISPATCH_SCHEDULE = 0x6011,
    GMM1 = 0x6021,
    ACT_QUANT = 0x6031,
    GMM2 = 0x6041,
    COMBINE = 0x6051,
    INPUT_BUFFER_INIT = 0x6101,
    ROUTE_SEND = 0x6111,
    SYNC_RESET = 0x6121,
    WAIT_INPUT_CORE_SYNC = 0x6131,
    WAIT_INPUT_RANK_SYNC = 0x6141,
    WAIT_OUTPUT_RANK_SYNC = 0x6151,
    COUNTS_EXPORT = 0x6161,
    WAIT_OUTPUT_CORE_SYNC = 0x6171,
    OUTPUT_BUFFER_INIT = 0x6181,
    UNPERMUTE = 0x6191,
    FINALIZE = 0x61a1,
    DISPATCH_BUFFER_INIT = 0x61b1,
    TOKEN_COUNT_PREPARE = 0x61c1,
    WAIT_GMM_DRAIN = 0x61d1,
    INIT = 0x61e1,
    WAIT_GMM1_INPUT = 0x6301,
    WAIT_GMM1_BUFFER = 0x6311,
    WAIT_ACT_INPUT = 0x6321,
    WAIT_GMM2_INPUT = 0x6331,
    WAIT_COMBINE_INPUT = 0x6341,
    WAIT_TOKEN_COUNT = 0x6351,
    DISPATCH_XFER = 0x6401,
    DISPATCH_LOCAL = 0x6411,
};
}
