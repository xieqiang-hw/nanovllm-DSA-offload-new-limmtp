#include <register/op_impl_registry.h>
#include "error/ops_error.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;
    const gert::Shape *topk = context->GetInputShape(0);
    if (topk == nullptr || topk->GetDimNum() != 2)
        return ge::GRAPH_FAILED;
    gert::Shape *ids = context->GetOutputShape(0);
    gert::Shape *counts = context->GetOutputShape(1);
    if (ids == nullptr || counts == nullptr) return ge::GRAPH_FAILED;
    ids->SetDimNum(2); ids->SetDim(0, topk->GetDim(0)); ids->SetDim(1, 8192);
    counts->SetDimNum(1); counts->SetDim(0, topk->GetDim(0));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(NanovllmFusedLiManageMtpUnion).InferShape(InferShape).InferDataType(InferType);
} // namespace ops
