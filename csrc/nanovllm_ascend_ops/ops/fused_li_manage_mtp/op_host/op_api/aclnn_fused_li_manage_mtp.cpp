#include "aclnn_fused_li_manage_mtp.h"
extern "C" {
extern aclnnStatus aclnnInnerNanovllmFusedLiManageMtpGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
extern aclnnStatus aclnnInnerNanovllmFusedLiManageMtp(
    void *, uint64_t, aclOpExecutor *, const aclrtStream);

aclnnStatus aclnnNanovllmFusedLiManageMtpGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *reqPoolEntries, const aclTensor *cacheSlots,
    const aclTensor *cacheTokens, const aclTensor *actualSeqLengthsKey,
    const aclTensor *blockTable, const aclTensor *topkIndexOut,
    const aclTensor *topkSlotsOut, const aclTensor *missSrcIdsOut,
    const aclTensor *missDstSlotsOut, const aclTensor *missCountOut,
    const aclTensor *cacheSlotsOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerNanovllmFusedLiManageMtpGetWorkspaceSize(
        query, key, weights, reqPoolEntries, cacheSlots, cacheTokens,
        actualSeqLengthsKey, blockTable, topkIndexOut, topkSlotsOut,
        missSrcIdsOut, missDstSlotsOut, missCountOut, cacheSlotsOut,
        workspaceSize, executor);
}
aclnnStatus aclnnNanovllmFusedLiManageMtp(void *workspace, uint64_t workspaceSize,
                                          aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerNanovllmFusedLiManageMtp(workspace, workspaceSize, executor, stream);
}
}
