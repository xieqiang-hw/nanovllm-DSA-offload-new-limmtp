#include "register/op_def_registry.h"
namespace ops {
class NanovllmFusedLiManageMtpUnion : public OpDef {
public:
    explicit NanovllmFusedLiManageMtpUnion(const char *name) : OpDef(name)
    {
        this->Input("union_pair0").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("union_pair1").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("topk_dst_slots").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("num_candidate_tokens").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Output("miss_src_ids").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("miss_counts").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};
OP_ADD(NanovllmFusedLiManageMtpUnion);
} // namespace ops
