#include "aclnn_fused_li_manage_mtp_union.h"
extern "C" {
extern aclnnStatus aclnnInnerNanovllmFusedLiManageMtpUnionGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
extern aclnnStatus aclnnInnerNanovllmFusedLiManageMtpUnion(void *, uint64_t, aclOpExecutor *, const aclrtStream);
aclnnStatus aclnnNanovllmFusedLiManageMtpUnionGetWorkspaceSize(
    const aclTensor *a, const aclTensor *b, const aclTensor *c, const aclTensor *d,
    const aclTensor *e, const aclTensor *f, const aclTensor *g, const aclTensor *h,
    const aclTensor *i, const aclTensor *j, const aclTensor *k, const aclTensor *l,
    const aclTensor *m, const aclTensor *n,
    uint64_t *size, aclOpExecutor **executor)
{ return aclnnInnerNanovllmFusedLiManageMtpUnionGetWorkspaceSize(
      a, b, c, d, e, f, g, h, i, j, k, l, m, n, size, executor); }
aclnnStatus aclnnNanovllmFusedLiManageMtpUnion(void *workspace, uint64_t size,
                                               aclOpExecutor *executor, const aclrtStream stream)
{ return aclnnInnerNanovllmFusedLiManageMtpUnion(workspace, size, executor, stream); }
}
