#include "aclnn_fused_li_manage_mtp_union.h"
extern "C" {
extern aclnnStatus aclnnInnerNanovllmFusedLiManageMtpUnionGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
extern aclnnStatus aclnnInnerNanovllmFusedLiManageMtpUnion(void *, uint64_t, aclOpExecutor *, const aclrtStream);
aclnnStatus aclnnNanovllmFusedLiManageMtpUnionGetWorkspaceSize(
    const aclTensor *a, const aclTensor *b, const aclTensor *c, const aclTensor *d,
    uint64_t *size, aclOpExecutor **executor)
{ return aclnnInnerNanovllmFusedLiManageMtpUnionGetWorkspaceSize(a, b, c, d, size, executor); }
aclnnStatus aclnnNanovllmFusedLiManageMtpUnion(void *workspace, uint64_t size,
                                               aclOpExecutor *executor, const aclrtStream stream)
{ return aclnnInnerNanovllmFusedLiManageMtpUnion(workspace, size, executor, stream); }
}
