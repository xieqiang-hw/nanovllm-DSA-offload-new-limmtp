/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Phase-1 MTP4 LightningIndexer + TopK kernel.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_li_manage_mtp_template_tiling_key.h"
#include "lightning_indexer_kernel.h"

using namespace LIKernel;

#define LI_MTP_COPY_TILING()                                                                                           \
    GET_TILING_DATA_WITH_STRUCT(FusedLiManageTilingData, tiling_data_in, tiling);                                      \
    const FusedLiManageTilingData *__restrict tiling_data = &tiling_data_in

#define INVOKE_LI_MTP_TOPK(...)                                                                                        \
    do {                                                                                                               \
        LI_MTP_COPY_TILING();                                                                                           \
        LIPreload<LIType<__VA_ARGS__, int32_t, true, LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND>> op;                          \
        op.Init(query, key, weights, reqPoolEntries, cacheSlots, nullptr, actualSeqLengths, blockTable,              \
                topkIndex, topkSlots, missSrcIds, missDstSlots, scoreScratch, thresholdScratch,                       \
                user, tiling_data, &tPipe);                                                                             \
        op.Process();                                                                                                  \
    } while (0)

template <int DT>
__global__ __aicore__ void nanovllm_fused_li_manage_mtp(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
    __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
    __gm__ uint8_t *cacheTokens, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *topkIndex,
    __gm__ uint8_t *topkSlots, __gm__ uint8_t *missSrcIds,
    __gm__ uint8_t *missDstSlots, __gm__ uint8_t *missCount,
    __gm__ uint8_t *cacheSlotsOut, __gm__ uint8_t *scoreScratch,
    __gm__ uint8_t *thresholdScratch, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)
#else
    TPipe tPipe;
    (void)missCount;
    (void)cacheSlotsOut;
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
