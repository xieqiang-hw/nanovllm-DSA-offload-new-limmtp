#ifndef ACLNN_NANOVLLM_FUSED_LI_MANAGE_MTP_UNION_H
#define ACLNN_NANOVLLM_FUSED_LI_MANAGE_MTP_UNION_H
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
extern "C" {
__attribute__((visibility("default"))) aclnnStatus aclnnNanovllmFusedLiManageMtpUnionGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
__attribute__((visibility("default"))) aclnnStatus aclnnNanovllmFusedLiManageMtpUnion(
    void *, uint64_t, aclOpExecutor *, const aclrtStream);
}
#endif
