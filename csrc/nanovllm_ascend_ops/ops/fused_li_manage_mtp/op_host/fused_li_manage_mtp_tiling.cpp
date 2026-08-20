/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#define FUSED_LI_MANAGE_MTP_TILING_REGISTRATION
#include "../../fused_li_manage/op_host/fused_li_manage_tiling.h"

namespace optiling {
static ge::graphStatus TilingPrepareForNanovllmFusedLiManageMtp(
    gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForNanovllmFusedLiManageMtp(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR("NanovllmFusedLiManageMtp", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    FusedLiManageTilingInfo info;
    FusedLiManageTiling tiling(context, true);
    if (tiling.ParseAndCheck(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tiling.DoTiling(&info);
}

IMPL_OP_OPTILING(NanovllmFusedLiManageMtp)
    .Tiling(TilingForNanovllmFusedLiManageMtp)
    .TilingParse<FusedLiManageCompileInfo>(TilingPrepareForNanovllmFusedLiManageMtp);
} // namespace optiling
