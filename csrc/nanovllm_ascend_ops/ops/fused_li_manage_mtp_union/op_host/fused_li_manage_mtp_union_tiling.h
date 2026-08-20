#ifndef FUSED_LI_MANAGE_MTP_UNION_TILING_H_
#define FUSED_LI_MANAGE_MTP_UNION_TILING_H_
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(FusedLiManageMtpUnionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(NanovllmFusedLiManageMtpUnion, FusedLiManageMtpUnionTilingData)
struct FusedLiManageMtpUnionCompileInfo {};
} // namespace optiling
#endif
