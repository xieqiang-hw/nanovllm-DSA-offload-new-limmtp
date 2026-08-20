/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Phase-1 MTP4 LightningIndexer + TopK kernel.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_li_manage_mtp_template_tiling_key.h"
#include "lightning_indexer_kernel.h"

using namespace LIKernel;

__aicore__ inline void InitMtpPlaceholderOutputs(__gm__ uint8_t *topkSlots,
                                                  __gm__ uint8_t *missCount,
                                                  uint32_t batchSize)
{
    if ASCEND_IS_AIV {
        constexpr uint32_t MAX_INIT_ELEMENTS = 4096U;
        uint32_t aivCoreNum = GetBlockNum() * 2U;
        uint32_t aivCoreIdx = GetBlockIdx();
        uint64_t totalSlotElements = static_cast<uint64_t>(batchSize) * 4U * 2048U;
        uint64_t elementsPerCore = (totalSlotElements + aivCoreNum - 1U) / aivCoreNum;
        uint64_t coreStart = static_cast<uint64_t>(aivCoreIdx) * elementsPerCore;
        uint64_t remainingElements = coreStart < totalSlotElements ? totalSlotElements - coreStart : 0U;
        uint64_t coreElements = elementsPerCore < remainingElements ? elementsPerCore : remainingElements;
        GlobalTensor<int32_t> topkSlotsGm;
        topkSlotsGm.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
        for (uint64_t offset = 0; offset < coreElements; offset += MAX_INIT_ELEMENTS) {
            uint64_t remainingCoreElements = coreElements - offset;
            uint64_t initElements = MAX_INIT_ELEMENTS < remainingCoreElements
                ? static_cast<uint64_t>(MAX_INIT_ELEMENTS)
                : remainingCoreElements;
            AscendC::InitGlobalMemory(topkSlotsGm[coreStart + offset], initElements, -1);
        }
        if (GetBlockIdx() == 0U) {
            GlobalTensor<int32_t> missCountGm;
            missCountGm.SetGlobalBuffer((__gm__ int32_t *)missCount);
            AscendC::InitGlobalMemory(missCountGm, batchSize, 0);
        }
    }
}

#define LI_MTP_COPY_TILING()                                                                                           \
    GET_TILING_DATA_WITH_STRUCT(FusedLiManageTilingData, tiling_data_in, tiling);                                      \
    const FusedLiManageTilingData *__restrict tiling_data = &tiling_data_in

#define INVOKE_LI_MTP_TOPK(...)                                                                                        \
    do {                                                                                                               \
        LI_MTP_COPY_TILING();                                                                                           \
        LIPreload<LIType<__VA_ARGS__, int32_t, true, LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND>> op;                          \
        op.Init(query, key, weights, nullptr, actualSeqLengths, blockTable, topkIndex, topkSlots,                      \
                user, tiling_data, &tPipe);                                                                             \
        op.Process();                                                                                                  \
        InitMtpPlaceholderOutputs(topkSlots, missCount, tiling_data->bSize);                                            \
    } while (0)

template <int DT>
__global__ __aicore__ void nanovllm_fused_li_manage_mtp(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
    __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
    __gm__ uint8_t *cacheTokens, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *topkIndex,
    __gm__ uint8_t *topkSlots, __gm__ uint8_t *missSrcIds,
    __gm__ uint8_t *missDstSlots, __gm__ uint8_t *missCount,
    __gm__ uint8_t *cacheSlotsOut, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)
#else
    TPipe tPipe;
    (void)missSrcIds;
    (void)missDstSlots;
    (void)cacheSlotsOut;
    (void)reqPoolEntries;
    (void)cacheSlots;
    (void)cacheTokens;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (DT == LI_MTP_TPL_FP16) {
        INVOKE_LI_MTP_TOPK(half, half);
    } else {
        INVOKE_LI_MTP_TOPK(bfloat16_t, bfloat16_t);
    }
#endif
}
