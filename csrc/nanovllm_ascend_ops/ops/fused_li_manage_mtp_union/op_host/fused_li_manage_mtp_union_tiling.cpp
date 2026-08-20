#include "fused_li_manage_mtp_union_tiling.h"
#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;
    const gert::StorageShape *shape = context->GetInputShape(0);
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() != 3) return ge::GRAPH_FAILED;
    uint32_t batch = static_cast<uint32_t>(shape->GetStorageShape().GetDim(0) / 4);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aiv = platform.GetCoreNumAiv();
    context->SetBlockDim(platform.CalcTschBlockDim(std::min(batch, aiv), 0U, aiv));
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    FusedLiManageMtpUnionTilingData data;
    data.set_batchSize(batch);
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    context->SetTilingKey(0U);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Prepare(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(NanovllmFusedLiManageMtpUnion)
    .Tiling(Tiling).TilingParse<FusedLiManageMtpUnionCompileInfo>(Prepare);
} // namespace optiling
