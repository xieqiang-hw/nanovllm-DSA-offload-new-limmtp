/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

namespace ops {
static ge::graphStatus InferShapeNanovllmFusedLiManageMtp(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("NanovllmFusedLiManageMtp", "context is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *query = context->GetInputShape(0);
    const gert::Shape *cache = context->GetInputShape(4);
    const gert::Shape *metadata = context->GetInputShape(3);
    OPS_LOG_E_IF_NULL(context, query, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cache, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, metadata, return ge::GRAPH_FAILED);
    OPS_ERR_IF(query->GetDimNum() != 3 || query->GetDim(0) % 4 != 0,
               OPS_LOG_E(context, "query must be [B*4, 32, 128]"), return ge::GRAPH_FAILED);

    for (uint32_t out = 0; out < 2; ++out) {
        gert::Shape *shape = context->GetOutputShape(out);
        OPS_LOG_E_IF_NULL(context, shape, return ge::GRAPH_FAILED);
        shape->SetDimNum(3);
        shape->SetDim(0, query->GetDim(0));
        shape->SetDim(1, 1);
        shape->SetDim(2, 2048);
    }
    for (uint32_t out = 2; out < 4; ++out) {
        gert::Shape *shape = context->GetOutputShape(out);
        OPS_LOG_E_IF_NULL(context, shape, return ge::GRAPH_FAILED);
        shape->SetDimNum(2);
        shape->SetDim(0, metadata->GetDim(0));
        shape->SetDim(1, 8192);
    }
    gert::Shape *missCount = context->GetOutputShape(4);
    gert::Shape *cacheOut = context->GetOutputShape(5);
    OPS_LOG_E_IF_NULL(context, missCount, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cacheOut, return ge::GRAPH_FAILED);
    missCount->SetDimNum(1);
    missCount->SetDim(0, metadata->GetDim(0));
    *cacheOut = *cache;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeNanovllmFusedLiManageMtp(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("NanovllmFusedLiManageMtp", "context is nullptr"),
               return ge::GRAPH_FAILED);
    for (uint32_t out = 0; out < 6; ++out) {
        context->SetOutputDataType(out, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NanovllmFusedLiManageMtp)
    .InferShape(InferShapeNanovllmFusedLiManageMtp)
    .InferDataType(InferDataTypeNanovllmFusedLiManageMtp);
} // namespace ops
