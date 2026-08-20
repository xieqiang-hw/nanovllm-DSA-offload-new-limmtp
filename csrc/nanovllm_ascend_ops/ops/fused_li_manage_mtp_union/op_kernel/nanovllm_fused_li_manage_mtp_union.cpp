#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "fused_li_manage_mtp_union_tiling.h"
using namespace AscendC;
namespace {
constexpr uint32_t ROUTES=4U, TOPK=2048U, PAIR_WORDS=4096U, CAPACITY=8192U;
constexpr uint32_t SHORT_MASK=(1U<<18U)-1U, LONG_MASK=(1U<<21U)-1U;
constexpr uint32_t MISS_KEY_BASE_BITS=0x40000000U;
template <HardEvent event> __aicore__ inline void Sync(HardEvent e) {
    event_t id=static_cast<event_t>(GetTPipePtr()->FetchEventID(e));
    AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
}
class MtpMissUnion {
public:
    __aicore__ inline void Init(GM_ADDR p0, GM_ADDR p1, GM_ADDR slots, GM_ADDR candidates,
                                GM_ADDR out, GM_ADDR counts,
                                const optiling::FusedLiManageMtpUnionTilingData *t, TPipe *pipe) {
        pair0Gm.SetGlobalBuffer((__gm__ float*)p0);
        pair1Gm.SetGlobalBuffer((__gm__ float*)p1);
        slotsGm.SetGlobalBuffer((__gm__ int32_t*)slots);
        candidatesGm.SetGlobalBuffer((__gm__ int32_t*)candidates);
        outGm.SetGlobalBuffer((__gm__ int32_t*)out);
        countsGm.SetGlobalBuffer((__gm__ int32_t*)counts);
        batchSize=t->batchSize;
        pipe->InitBuffer(pairInBuf,CAPACITY*2U*sizeof(float));
        pipe->InitBuffer(pairOutBuf,CAPACITY*2U*sizeof(float));
        pipe->InitBuffer(slotsBuf,TOPK*sizeof(int32_t));
        pipe->InitBuffer(countBuf,32U);
    }
    __aicore__ inline void Process() {
        for(uint32_t b=GetBlockIdx();b<batchSize;b+=GetBlockNum()) ProcessBatch(b);
    }
private:
    __aicore__ inline void ProcessBatch(uint32_t b) {
        LocalTensor<float> input=pairInBuf.Get<float>(), merged=pairOutBuf.Get<float>();
        LocalTensor<int32_t> result=input.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> slots=slotsBuf.Get<int32_t>(), countLocal=countBuf.Get<int32_t>();
        uint32_t len[ROUTES]={0U,0U,0U,0U};
        for(uint32_t r=0;r<ROUTES;++r) {
            uint64_t base=(static_cast<uint64_t>(b)*ROUTES+r)*TOPK;
            DataCopy(slots,slotsGm[base],TOPK); Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            while(len[r]<TOPK && slots.GetValue(len[r])<0) ++len[r];
        }
        uint64_t base=static_cast<uint64_t>(b)*CAPACITY;
        DataCopy(input,pair0Gm[base],CAPACITY);
        DataCopy(input[CAPACITY],pair1Gm[base],CAPACITY);
        Sync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        MrgSort4Info p;
        p.elementLengths[0]=len[0]; p.elementLengths[1]=len[1];
        p.elementLengths[2]=len[2]; p.elementLengths[3]=len[3];
        p.ifExhaustedSuspension=false; p.validBit=0b1111; p.repeatTimes=1;
        MrgSortSrcList<float> src;
        src.src1=input; src.src2=input[PAIR_WORDS];
        src.src3=input[PAIR_WORDS*2U]; src.src4=input[PAIR_WORDS*3U];
        MrgSort<float>(merged,src,p); Sync<HardEvent::V_S>(HardEvent::V_S);
        uint32_t total=len[0]+len[1]+len[2]+len[3], count=0U;
        uint32_t mask=candidatesGm.GetValue(b)>static_cast<int32_t>(SHORT_MASK+1U)?LONG_MASK:SHORT_MASK;
        int32_t last=-1; LocalTensor<uint32_t> bits=merged.ReinterpretCast<uint32_t>();
        for(uint32_t i=0;i<total;++i) {
            uint32_t key=bits.GetValue(i*2U);
            int32_t id=static_cast<int32_t>(mask-(key-MISS_KEY_BASE_BITS));
            if(id!=last) { result.SetValue(count++,id); last=id; }
        }
        if(count!=0U) {
            Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(outGm[static_cast<uint64_t>(b)*CAPACITY],result,
                        {1,static_cast<uint16_t>(count*sizeof(int32_t)),0,0});
            Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        countLocal.SetValue(0,static_cast<int32_t>(count));
        Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(countsGm[b],countLocal,{1,static_cast<uint16_t>(sizeof(int32_t)),0,0});
    }
    GlobalTensor<float> pair0Gm,pair1Gm;
    GlobalTensor<int32_t> slotsGm,candidatesGm,outGm,countsGm;
    TBuf<TPosition::VECCALC> pairInBuf,pairOutBuf,slotsBuf,countBuf;
    uint32_t batchSize=0U;
};
}
extern "C" __global__ __aicore__ void nanovllm_fused_li_manage_mtp_union(
    GM_ADDR pair0,GM_ADDR pair1,GM_ADDR slots,GM_ADDR candidates,GM_ADDR out,GM_ADDR counts,
    GM_ADDR workspace,GM_ADDR tiling) {
    (void)workspace; KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(optiling::FusedLiManageMtpUnionTilingData,data,tiling);
    TPipe pipe; MtpMissUnion op; op.Init(pair0,pair1,slots,candidates,out,counts,&data,&pipe); op.Process();
}
