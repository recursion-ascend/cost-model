// Extracted from repository tiling; BF16 dtype size passed explicitly.
// 与 op_host/op_tiling/arch35/mega_moe_tiling.cpp 的 CalcMGroupsPerWave 保持同步：
// p1/p2 由调用方解析后传入；policy 非空时回放全部中间量（cost model 标定日志用）。

/*
 * 解析 p1（GMM1 每核保底逻辑 N tile 数）：无显式策略时按 bs 分档。
 * bs 为本卡本次实际处理的 token 数；小 batch 取细波降低流水一拍滞后的绝对时延，
 * 大 batch 取粗波摊薄波界同步；默认档与历史行为一致。
 */
static uint32_t ResolveGmm1MinLogicalTilesPerCore(uint32_t bs)
{
    uint32_t minTilesPerCore = GMM1_MIN_LOGICAL_TILES_PER_CORE;
    if (bs > 0U && bs < GMM1_SMALL_BATCH_TOKEN_THRESHOLD) {
        minTilesPerCore = GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL;
    } else if (bs >= GMM1_LARGE_BATCH_TOKEN_THRESHOLD) {
        minTilesPerCore = GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE;
    }
    return minTilesPerCore;
}

// Wave-sizing policy 中间量与结果（与 op_host 的 MegaMoeWavePolicy 保持同步）。
struct MegaMoeWavePolicy {
    uint32_t gmm1MinLogicalTilesPerCore; // p1（分档后生效值）
    uint32_t gmm2MinLogicalTilesPerCore; // p2
    uint64_t gmm1LogicalTilesPerMGroup;  // ceil(hiddenDim / GMM_TILE_N)
    uint64_t gmm2TilesPerMGroup;         // ceil(h / GMM_TILE_N)
    uint64_t gmm1RequiredMGroups;        // G1
    uint64_t gmm2RequiredMGroups;        // G2
    uint32_t mGroupsPerWave;             // max(G1, G2)
};

uint32_t CalcMGroupsPerWave(const MegaMoeTilingData *tilingData, uint32_t aicNum,
                            uint32_t gmm1MinLogicalTilesPerCore, uint32_t gmm2MinLogicalTilesPerCore,
                            MegaMoeWavePolicy *policy = nullptr)
{
    if (tilingData->hiddenDim == 0U || tilingData->h == 0U || aicNum == 0U) {
        if (policy != nullptr) {
            *policy = MegaMoeWavePolicy{gmm1MinLogicalTilesPerCore, gmm2MinLogicalTilesPerCore, 0U, 0U, 0U, 0U, 1U};
        }
        return 1U;
    }

    /*
     * hiddenDim 包含 gate/up 两部分。交织模式每核至少调度 4 个独立 N tile；非交织模式
     * 每个物理任务成对处理 gate/up，原先每核 2 个物理任务同样等价于 4 个逻辑 N tile。
     * hiddenDim 已校验为 GMM_TILE_N(256) 的倍数，CeilDiv 对
     * 非交织半宽产生的尾 tile 只影响 wave 粒度估算，不影响正确性，两种编译模式共用本公式。
     */
    uint64_t gmm1LogicalTilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->hiddenDim, GMM_TILE_N);
    uint64_t gmm2TilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->h, GMM_TILE_N);
    uint64_t gmm1RequiredMGroups = ops::CeilDiv<uint64_t>(
        static_cast<uint64_t>(aicNum) * gmm1MinLogicalTilesPerCore, gmm1LogicalTilesPerMGroup);
    uint64_t gmm2RequiredMGroups = ops::CeilDiv<uint64_t>(
        static_cast<uint64_t>(aicNum) * gmm2MinLogicalTilesPerCore, gmm2TilesPerMGroup);
    uint32_t mGroupsPerWave = static_cast<uint32_t>(std::max(gmm1RequiredMGroups, gmm2RequiredMGroups));
    if (policy != nullptr) {
        *policy = MegaMoeWavePolicy{gmm1MinLogicalTilesPerCore, gmm2MinLogicalTilesPerCore,
                                    gmm1LogicalTilesPerMGroup,    gmm2TilesPerMGroup,
                                    gmm1RequiredMGroups,          gmm2RequiredMGroups,
                                    mGroupsPerWave};
    }
    return mGroupsPerWave;
}

const static uint32_t WEIGHT_MATRIX_ROW_DIM_INDEX = 0U;
const static uint32_t WEIGHT_MATRIX_COLUMN_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MATRIX_DIM_INDEX = 0U;
const static uint32_t WEIGHT_SCALE_GROUP_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MULTI_BASE_DIM_INDEX = 2U;

static uint32_t CalcDispatchCopyBufferBytes(const MegaMoeTilingData *tilingData, uint32_t activationElementsPerByte)
{
    // copyBufferBytes 是每个 dispatch slot 中量化 token 和 scale 的拷贝区大小。
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    uint32_t copyBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        uint32_t weightBytes =
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        copyBufferBytes += weightBytes;
    }
    return copyBufferBytes;
}

/*
 * 计算 dispatch 阶段不随 ring 深度与 route batch 变化的固定 UB 占用。
 */
static uint32_t CalcDispatchFixedBufferBytes(const MegaMoeTilingData *tilingData)
{
    // fixedBufferBytes 包含 cumsumInfoTensor_ 和 expertTokenNumsOutTensor_。
    uint32_t fixedBufferBytes =
        static_cast<uint32_t>(ops::CeilAlign(
            static_cast<uint64_t>(tilingData->epWorldSize) * tilingData->moeExpertPerRank * sizeof(int32_t),
            static_cast<uint64_t>(ALIGN_32))) +
        static_cast<uint32_t>(ops::CeilAlign(static_cast<uint64_t>(tilingData->moeExpertPerRank) * sizeof(int32_t),
                                             static_cast<uint64_t>(ALIGN_32)));
    return fixedBufferBytes;
}

/*
 * 分两步定下 dispatch 阶段的 UB 分配：
 *   第一步：先按一个保守的 route batch 大小起步，把剩下的 UB 全拿去开 ring slot，slot 越多流水越深；
 *   第二步：ring 深度定死之后，UB 若还有富余，再反过来把 route batch 撑大，这样总批数更少。
 * 预算减法使用饱和计算，避免 UB 不足时无符号回绕；bufferCount 必须先限制上限，再保证最小 ring 深度。
 */
static void SelectDispatchRingAndRouteBatch(MegaMoeDispatchBufferConfig &bufferConfig, uint64_t sendTotalNum,
                                            uint64_t alignedTotalRouteItems, uint32_t fixedBufferBytes,
                                            uint32_t dispatchSlotBytes, uint32_t availableUbBytes)
{
    // 第一步：用基准 batch 把 ring 深度定下来。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_RECV_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>(ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));

    // MTE 接收侧只保留一个 int32 topK 有效下标 batch，不再分配 mask 和第二个 index tensor。
    // routeIndexBufferBytes 是该有效下标 tensor 的大小。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    // bytesWithoutDispatchSlots 包含 count/prefix 固定区和一份有效下标 tensor。
    uint32_t bytesWithoutDispatchSlots = fixedBufferBytes + routeIndexBufferBytes;
    // dispatchSlotBudgetBytes 是扣除非 ring tensor 后可用于分配 dispatch slot 的 UB。
    uint32_t dispatchSlotBudgetBytes =
        availableUbBytes > bytesWithoutDispatchSlots ? availableUbBytes - bytesWithoutDispatchSlots : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(dispatchSlotBudgetBytes / dispatchSlotBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_DISPATCH_BUFFER_COUNT);
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_DISPATCH_BUFFER_COUNT);

    // 第二步：ring 深度已定，剩余 UB 用来扩 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // fixedBytesWithDispatchSlots 包含固定 tensor 和已选中的全部 dispatch slot。
        uint32_t fixedBytesWithDispatchSlots =
            fixedBufferBytes + static_cast<uint32_t>(bufferConfig.bufferCount) * dispatchSlotBytes;
        // routeItemBudgetBytes 是有效下标 tensor 可使用的 UB。
        uint32_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithDispatchSlots ? availableUbBytes - fixedBytesWithDispatchSlots : 0U;
        uint32_t expandedRouteItems = routeItemBudgetBytes / static_cast<uint32_t>(sizeof(int32_t));
        expandedRouteItems = expandedRouteItems / static_cast<uint32_t>(ALIGN_256) * ALIGN_256;
        expandedRouteItems =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(expandedRouteItems), alignedTotalRouteItems));
        if (expandedRouteItems > static_cast<uint32_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>(
                ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
        }
    }
}

static MegaMoeDispatchBufferConfig CalcDispatchBufferConfig(const MegaMoeTilingData *tilingData,
                                                            uint32_t activationElementsPerByte,
                                                            uint32_t availableUbBytes)
{
    MegaMoeDispatchBufferConfig bufferConfig{};
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->numMaxTokensPerRank);
    uint64_t alignedTotalRouteItems = ops::CeilAlign(sendTotalNum, static_cast<uint64_t>(ALIGN_256));
    uint32_t copyBufferBytes = CalcDispatchCopyBufferBytes(tilingData, activationElementsPerByte);
    bufferConfig.copyBufferBytes = copyBufferBytes;
    uint32_t fixedBufferBytes = CalcDispatchFixedBufferBytes(tilingData);
    // 一个 dispatch ring slot 包含 token/scale copy buffer 和一条 32B triple。
    uint32_t dispatchSlotBytes = copyBufferBytes + static_cast<uint32_t>(ALIGN_32);

    SelectDispatchRingAndRouteBatch(bufferConfig, sendTotalNum, alignedTotalRouteItems, fixedBufferBytes,
                                    dispatchSlotBytes, availableUbBytes);
    return bufferConfig;
}

static uint64_t CalcTopkValidIndexRingSlotBytes(uint32_t routeItemsPerBatch, uint32_t topK)
{
    // routeItemsPerBatch 按 256 个 item 对齐，但 topK 不一定整除 256，因此后续 batch 可能从某个
    // token 的 topK 段中间开始。同一 token 的 topK 专家不重复，单个专家每个 token 至多匹配一个下标；
    // 一个 batch 最多跨越 CeilDiv(routeItemsPerBatch + topK - 1, topK) 个 token，slot 按此上界预留空间。
    uint64_t maxMatchedRouteItems =
        ops::CeilDiv(static_cast<uint64_t>(routeItemsPerBatch) + topK - 1U, static_cast<uint64_t>(topK));
    uint64_t validIndexBytes = ops::CeilAlign(maxMatchedRouteItems * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32));
    return static_cast<uint64_t>(routeItemsPerBatch) / BITS_PER_BYTE + validIndexBytes;
}

// MTE Wave producer：ring slot 同时保存临时 compare mask 与本专家的 topK 有效下标。
static MegaMoeSendMaskBufferConfig CalcTopkValidIndexBufferConfig(const MegaMoeTilingData *tilingData,
                                                                  uint32_t fixedBufferBytes, uint32_t ownedExpertCount,
                                                                  uint32_t availableUbBytes)
{
    MegaMoeSendMaskBufferConfig bufferConfig{};
    // sendTotalNum 表示所有专家合计最多发送的 topK 有效下标数，不是 token 数。发送批网格按
    // numMaxTokensPerRank * topK 的容量上界划分，确保各 Rank 使用一致批次数；Kernel 再按本 Rank
    // 的实际 bs * topK 对每批有效长度进行裁剪。
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->topK;
    uint64_t alignedTotalRouteItems = ops::CeilAlign(sendTotalNum, static_cast<uint64_t>(ALIGN_256));

    // Stage 1：使用基准 batch 确定 route ring 深度。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_SEND_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>(ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
    bufferConfig.bufferBytes = static_cast<uint32_t>(
        CalcTopkValidIndexRingSlotBytes(static_cast<uint32_t>(bufferConfig.routeItemsPerBatch), tilingData->topK));

    // topkIdsTensor 和 gather 输出 tensor 各占一份 int32 route batch。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    uint32_t bytesWithoutRouteBuffers = fixedBufferBytes + 2U * routeIndexBufferBytes;
    // routeBufferBudgetBytes 是扣除固定 tensor 和两份 route tensor 后可用于 ring slot 的 UB。
    uint32_t routeBufferBudgetBytes =
        availableUbBytes > bytesWithoutRouteBuffers ? availableUbBytes - bytesWithoutRouteBuffers : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(routeBufferBudgetBytes / bufferConfig.bufferBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_SEND_MASK_BUFFER_COUNT);

    // ring 深度超过当前核的实际发送次数不会增加流水重叠。
    uint64_t routePushCount = static_cast<uint64_t>(bufferConfig.routeBatchCount) * ownedExpertCount;
    if (routePushCount > 0U && static_cast<uint64_t>(bufferConfig.bufferCount) > routePushCount) {
        bufferConfig.bufferCount = static_cast<int32_t>(routePushCount);
    }
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_SEND_MASK_BUFFER_COUNT);

    // Stage 2：固定 ring 深度，用剩余 UB 扩大 route batch。容量公式包含 topK 有效下标的占用。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // 每个 slot 预留有效下标的 32B 对齐余量和 batch 边界余量，其余 UB 用于随 batch 线性增长的数据区。
        uint64_t fixedBytesWithRoutePadding =
            static_cast<uint64_t>(fixedBufferBytes) +
            static_cast<uint64_t>(bufferConfig.bufferCount) * (ALIGN_32 + 2U * sizeof(int32_t));
        uint64_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithRoutePadding ? availableUbBytes - fixedBytesWithRoutePadding : 0U;
        // 两份 route tensor 各占 32bit/item；每个 ring slot 占 1bit mask 和约 32/topK bit 有效下标。
        uint64_t expandedRouteItems =
            routeItemBudgetBytes * BITS_PER_BYTE /
            (2U * sizeof(int32_t) * BITS_PER_BYTE + static_cast<uint64_t>(bufferConfig.bufferCount) +
             ops::CeilDiv(static_cast<uint64_t>(bufferConfig.bufferCount) * sizeof(int32_t) * BITS_PER_BYTE,
                          static_cast<uint64_t>(tilingData->topK)));
        expandedRouteItems = expandedRouteItems / ALIGN_256 * ALIGN_256;
        expandedRouteItems = std::min(expandedRouteItems, alignedTotalRouteItems);
        if (expandedRouteItems > static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>(
                ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
            bufferConfig.bufferBytes = static_cast<uint32_t>(
                CalcTopkValidIndexRingSlotBytes(static_cast<uint32_t>(expandedRouteItems), tilingData->topK));
        }
    }
    return bufferConfig;
}

/*
 * 计算 unpermute 的 slot 与 scale 尺寸：单 token 的 BF16 搬入区 + FP32 计算区，以及 combine quant 的 scale 展开区。
 * dataSlotBytes 与 scaleBytes 经出参带出，供后续两个阶段共用。
 */
static void CalcUnpermuteSlotAndScaleBytes(const MegaMoeTilingData *tilingData,
                                           MegaMoeUnpermuteBufferConfig &bufferConfig, uint32_t &dataSlotBytes,
                                           uint32_t &scaleBytes)
{
    uint32_t bf16SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(uint16_t), static_cast<uint64_t>(ALIGN_32)));
    uint32_t fp32SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    // dataSlotBytes 是同一 token 的 BF16 搬入区和 FP32 计算区之和。
    dataSlotBytes = bf16SlotBytes + fp32SlotBytes;
    bufferConfig.bf16SlotElementCount = bf16SlotBytes / sizeof(uint16_t);
    bufferConfig.fp32SlotElementCount = fp32SlotBytes / sizeof(float);

    // scaleBytes 是 combine quant 使用的 BF16/FP32 scale 展开区大小。
    scaleBytes = 0U;
    if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleElementCount = (tilingData->h + ALIGN_32 - 1U) / ALIGN_32;
        scaleBytes = static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(uint16_t) * DEQUANT_BF16_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32))) +
                     static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(float) * DEQUANT_FP32_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32)));
    }
}

/*
 * 按 weight 元素个数刷新 unpermute 的 weight 区大小：FP32 主区必算，转换中转区只有需要 dtype
 * 转换时才占 UB。两个阶段定完 tokensPerBatch 后都要重算一次，所以抽出来复用。
 */
static void SetUnpermuteWeightBufferBytes(MegaMoeUnpermuteBufferConfig &bufferConfig, uint32_t weightElementCount,
                                          uint32_t topKWeightsConversionElementBytes)
{
    bufferConfig.topKWeightsBufferBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    if (topKWeightsConversionElementBytes > 0U) {
        bufferConfig.topKWeightsConversionBufferBytes = static_cast<uint32_t>(
            ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * topKWeightsConversionElementBytes,
                           static_cast<uint64_t>(ALIGN_32)));
    }
}

static MegaMoeUnpermuteBufferConfig CalcUnpermuteBufferConfig(const MegaMoeTilingData *tilingData,
                                                              uint32_t coreTokenCount,
                                                              uint32_t topKWeightsConversionElementBytes,
                                                              uint32_t availableUbBytes)
{
    MegaMoeUnpermuteBufferConfig bufferConfig{};
    if (coreTokenCount == 0U) {
        return bufferConfig;
    }

    uint32_t dataSlotBytes = 0U;
    uint32_t scaleBytes = 0U;
    CalcUnpermuteSlotAndScaleBytes(tilingData, bufferConfig, dataSlotBytes, scaleBytes);

    // 第一步：先按基准 weight batch 定住每批 token 数和 weight 区，再拿剩余 UB 开输入 ring。
    uint32_t baseTokensPerBatch = UNPERMUTE_WEIGHT_ITEMS_PER_BATCH / tilingData->topK;
    bufferConfig.tokensPerBatch = static_cast<int32_t>(std::min(baseTokensPerBatch, coreTokenCount));
    uint32_t weightElementCount = static_cast<uint32_t>(bufferConfig.tokensPerBatch) * tilingData->topK;
    SetUnpermuteWeightBufferBytes(bufferConfig, weightElementCount, topKWeightsConversionElementBytes);

    // bytesBeforeInputBuffers 包含 weight、scale 和一个累加/输出 data slot。
    uint32_t bytesBeforeInputBuffers = bufferConfig.topKWeightsBufferBytes +
                                       bufferConfig.topKWeightsConversionBufferBytes + scaleBytes + dataSlotBytes;
    uint32_t inputBufferBudgetBytes =
        availableUbBytes > bytesBeforeInputBuffers ? availableUbBytes - bytesBeforeInputBuffers : 0U;
    bufferConfig.inputBufferCount = static_cast<int32_t>(inputBufferBudgetBytes / dataSlotBytes);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, MAX_UNPERMUTE_INPUT_BUFFER_COUNT);
    int32_t accumulationItemCount =
        bufferConfig.tokensPerBatch * static_cast<int32_t>(tilingData->topK + tilingData->sharedExpertNum);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, accumulationItemCount);
    bufferConfig.inputBufferCount = std::max(bufferConfig.inputBufferCount, MIN_UNPERMUTE_INPUT_BUFFER_COUNT);

    // 第二步：ring 深度已定，UB 还有富余就把 weight batch 撑大。
    if (baseTokensPerBatch < coreTokenCount) {
        // fixedBytes 包含 scale、一个累加/输出 slot 和已经选中的所有输入 slot。
        uint32_t fixedBytes = scaleBytes + (static_cast<uint32_t>(bufferConfig.inputBufferCount) + 1U) * dataSlotBytes;
        // weightBudgetBytes 是 FP32 weight 及可选转换中转区可使用的 UB。
        uint32_t weightBudgetBytes = availableUbBytes - fixedBytes - UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES;
        uint32_t weightBytesPerToken =
            tilingData->topK * (static_cast<uint32_t>(sizeof(float)) + topKWeightsConversionElementBytes);
        uint32_t expandedTokensPerBatch = std::min(weightBudgetBytes / weightBytesPerToken, coreTokenCount);
        if (expandedTokensPerBatch > static_cast<uint32_t>(bufferConfig.tokensPerBatch)) {
            bufferConfig.tokensPerBatch = static_cast<int32_t>(expandedTokensPerBatch);
            weightElementCount = expandedTokensPerBatch * tilingData->topK;
            SetUnpermuteWeightBufferBytes(bufferConfig, weightElementCount, topKWeightsConversionElementBytes);
        }
    }
    return bufferConfig;
}

static uint64_t CalcCombineSyncSlotCountPerExpert(const MegaMoeTilingData *tilingData)
{
    // MTE 统一使用 per-expert AIC ready 表；group counter 服务 URMA layered Combine，量化与非量化均需要。
    if (tilingData->topoType != TOPO_TYPE_URMA || tilingData->moeExpertPerRank == 0U) {
        return 0U;
    }

    // 上述 guard 保证这里只剩 URMA；layered Combine 仅由 subBlockIdx=1 的半数 AIV 执行。
    uint64_t combineCoreCount = tilingData->blockAivNum / 2U;
    // 同一 token 的 topK expert id 不重复，因此单 expert 从每张卡最多接收 bs 个 token。
    uint64_t maxTokenCountForOneExpert =
        static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->epWorldSize;
    uint64_t maxTokenGroupCountForOneExpert =
        ops::CeilDiv(maxTokenCountForOneExpert, static_cast<uint64_t>(COMBINE_TOKEN_GROUP_SIZE));
    // Workspace 在路由结果产生前分配，因此每个本卡 MoE expert 都按独立最坏情况预留 slot。
    return std::max(maxTokenGroupCountForOneExpert, combineCoreCount);
}

static uint64_t CalcHostFlagElementCount(const MegaMoeTilingData *tilingData)
{
    uint64_t maxWavesPerExpert = ops::CeilDiv<uint64_t>(tilingData->maxOutputSize, L1_TILE_M_256);
    uint64_t waveFlagSlotsPerExpert = maxWavesPerExpert * INT_CACHELINE;
    uint64_t activationFlagSlotsPerExpert =
        tilingData->topoType == TOPO_TYPE_MTE ? waveFlagSlotsPerExpert : INT_CACHELINE;
    uint64_t moeExpertCount = tilingData->moeExpertPerRank;

    uint64_t flagElementCount = moeExpertCount * (activationFlagSlotsPerExpert + waveFlagSlotsPerExpert +
                                                  static_cast<uint64_t>(INT_CACHELINE) * tilingData->aicNum);
    bool isW4Mode = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
                    tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
                    tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ;
    if (isW4Mode || (tilingData->topoType == TOPO_TYPE_MTE && tilingData->combineQuantMode == COMBINE_NO_QUANT)) {
        flagElementCount += static_cast<uint64_t>(tilingData->aicNum) * INT_CACHELINE;
    }
    if (tilingData->topoType == TOPO_TYPE_MTE && tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        flagElementCount += moeExpertCount * tilingData->aicNum * INT_CACHELINE;
    }
    if (tilingData->topoType == TOPO_TYPE_URMA) {
        flagElementCount += tilingData->combineSyncSlotCountPerExpert * moeExpertCount * INT_CACHELINE;
    }
    if (tilingData->sharedExpertNum > 0 && tilingData->topoType == TOPO_TYPE_MTE) {
        uint64_t tokenGroupCount = ops::CeilDiv<uint64_t>(tilingData->bs, L1_TILE_M_256);
        flagElementCount += tokenGroupCount * tilingData->sharedExpertNum * INT_CACHELINE;
        flagElementCount +=
            static_cast<uint64_t>(CalcSharedActivationFlagElementsPerExpert(static_cast<int64_t>(tilingData->bs))) *
            tilingData->sharedExpertNum;
    }
    return flagElementCount;
}

/*
 * 设置 topK 有效下标发送的两套 UB 配置：先算固定占用，再按 expert 分核的两类 core 各算一套。
 * 本函数读 tilingData->combineSyncSlotCountPerExpert（经 CalcHostFlagElementCount），
 * 该字段必须在调用前写好。
 */
static void SetTopkValidIndexBufferConfigs(MegaMoeTilingData *tilingData, uint32_t activationElementsPerByte,
                                           uint32_t availableUbBytes)
{
    uint64_t totalFlagElementCount = CalcHostFlagElementCount(tilingData);
    uint32_t resetElementCountPerCore =
        static_cast<uint32_t>(ops::CeilDiv(totalFlagElementCount, static_cast<uint64_t>(tilingData->blockAivNum)));
    uint32_t resetBatchElementCount = std::min(resetElementCountPerCore, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    uint32_t resetTensorBytes =
        ops::CeilAlign(resetBatchElementCount, static_cast<uint32_t>(INT32_PER_256B)) * sizeof(int32_t);
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    // 与 kernel xOutTensorSize 一致：token 和 scale 分别对齐，prefetch 模式再追加对齐后的 weight。
    uint32_t quantOutputBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        quantOutputBufferBytes +=
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    }
    uint32_t quantInputBufferBytes = ops::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_128)) * sizeof(uint16_t);
    // sendCntAccTensor_ 按本卡 MoE 专家数分配，与 kernel 地址布局一致。
    uint32_t maxExpertCountPerCore =
        ops::CeilDiv(tilingData->epWorldSize * tilingData->moeExpertPerRank, tilingData->blockAivNum);
    uint32_t sendCountAccumulatorBytes = static_cast<uint32_t>(ops::CeilAlign(
        static_cast<uint64_t>(maxExpertCountPerCore) * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32)));
    // mxTempTensor_ 占 2KB，xOutTensor_ 和 xInTensor_ 各使用双 buffer。
    uint32_t sendMaskFixedBufferBytes = resetTensorBytes + 2U * 1024U + 2U * quantOutputBufferBytes +
                                        2U * quantInputBufferBytes + sendCountAccumulatorBytes;

    /*
     * 与 kernel topK 有效下标发送的 expert 连续均衡分核一一对应。totalExpertCount 除以 blockAivNum 后，
     * 前 remainder 个 AIV job 各多处理一个 expert，其余 job 处理 quotient 个 expert，因此这里只需
     * 预计算两套配置。Dispatch/Combine 的 wave 内 token 轮转由各自阶段完成，与这里的一次性发送分核无关。
     *
     * 若修改发送阶段的 expert 分核方式、ownedExpertCount 或 routePushCount 计算，必须同步更新
     * 这里的两类 core 划分和 kernel 配置选择条件。
     */
    uint32_t totalExpertCount = tilingData->epWorldSize * tilingData->moeExpertPerRank;
    uint32_t expertCountPerCoreWithoutExtraExpert = totalExpertCount / tilingData->blockAivNum;
    tilingData->sendMaskCoreCountWithExtraExpert = totalExpertCount % tilingData->blockAivNum;
    uint32_t expertCountPerCoreWithExtraExpert = expertCountPerCoreWithoutExtraExpert + 1U;
    tilingData->sendMaskConfigForCoreWithExtraExpert = CalcTopkValidIndexBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithExtraExpert, availableUbBytes);
    tilingData->sendMaskConfigForCoreWithoutExtraExpert = CalcTopkValidIndexBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithoutExtraExpert, availableUbBytes);
}

/*
 * 设置 Unpermute 的完整 chunk 与 tail chunk 两套 UB 配置，并记录完整 chunk 对应的 core 数。
 */
static void SetUnpermuteBufferConfigs(MegaMoeTilingData *tilingData, uint32_t topKWeightsConversionElementBytes,
                                      uint32_t availableUbBytes)
{
    /*
     * 与 kernel Unpermute 开头的 TilingByCore(m_, ..., align=1) 一一对应。TilingByCore 使用：
     *   fullTokenChunkSize = ceil(bs / blockAivNum)
     * 为连续 core 分配等长完整 chunk，最后一个活跃 core 可能只处理 tail，后续 core 的 coreLen 为 0
     * 并在读取配置前返回。因此 host 只需预计算“完整 chunk”和“tail chunk”两套配置，并记录完整
     * chunk 对应的 core 数作为 kernel 选择边界。
     *
     * 若修改 TilingByCore、Unpermute 的 align 参数或分核方式，必须同步更新下面的 chunk 推导以及
     * kernel 中 UnpermuteBuffInit 的配置选择条件。
     */
    // FP32 可直接搬入计算 buffer；其他已支持类型按实际元素大小预留转换中转区。
    // This harness uses BF16 top-k weights: caller passes sizeof(uint16_t).
    uint32_t fullTokenChunkSize = ops::CeilDiv(tilingData->bs, tilingData->blockAivNum);
    uint32_t activeCoreCount = ops::CeilDiv(tilingData->bs, fullTokenChunkSize);
    uint32_t tailTokenChunkSize = tilingData->bs - (activeCoreCount - 1U) * fullTokenChunkSize;
    bool tailIsFullTokenChunk = tailTokenChunkSize == fullTokenChunkSize;
    tilingData->unpermuteFullTokenChunkCoreCount = tailIsFullTokenChunk ? activeCoreCount : activeCoreCount - 1U;
    tilingData->unpermuteConfigForFullTokenChunk =
        CalcUnpermuteBufferConfig(tilingData, fullTokenChunkSize, topKWeightsConversionElementBytes, availableUbBytes);
    tilingData->unpermuteConfigForTailTokenChunk =
        tailIsFullTokenChunk ? MegaMoeUnpermuteBufferConfig{} :
                               CalcUnpermuteBufferConfig(tilingData, tailTokenChunkSize,
                                                         topKWeightsConversionElementBytes, availableUbBytes);
}
