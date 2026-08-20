#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "fused_li_manage_mtp_union_tiling.h"
using namespace AscendC;
namespace {
constexpr uint32_t ROUTES=4U, TOPK=2048U, PAIR_WORDS=4096U, CAPACITY=8192U, CHUNK=512U;
constexpr uint32_t SORT_REPEATS=CHUNK/32U, ACC_PAIR_FLOATS=CAPACITY*2U;
constexpr uint32_t MERGE_PAIR_FLOATS=(CAPACITY+CHUNK)*2U;
constexpr uint32_t SHORT_MASK=(1U<<18U)-1U, LONG_MASK=(1U<<21U)-1U;
constexpr uint32_t MISS_KEY_BASE_BITS=0x40000000U;
constexpr int32_t NEG_INF_BITS=static_cast<int32_t>(0xFF800000U);
template <HardEvent event> __aicore__ inline void Sync(HardEvent e) {
    event_t id=static_cast<event_t>(GetTPipePtr()->FetchEventID(e));
    AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
}

__aicore__ inline void SortChunk(LocalTensor<float> dst, LocalTensor<float> key,
                                 LocalTensor<uint32_t> payload, LocalTensor<float> tmp) {
    Sort32(tmp,key,payload,SORT_REPEATS); PipeBarrier<PIPE_V>();
    MrgSort4Info p;
    p.elementLengths[0]=32; p.elementLengths[1]=32; p.elementLengths[2]=32; p.elementLengths[3]=32;
    p.ifExhaustedSuspension=false; p.validBit=0b1111; p.repeatTimes=1;
    for(uint32_t group=0;group<4U;++group) {
        uint32_t off=group*256U;
        MrgSortSrcList<float> src;
        src.src1=tmp[off]; src.src2=tmp[off+64U]; src.src3=tmp[off+128U]; src.src4=tmp[off+192U];
        MrgSort<float>(dst[off],src,p);
    }
    PipeBarrier<PIPE_V>();
    p.elementLengths[0]=128; p.elementLengths[1]=128;
    p.elementLengths[2]=128; p.elementLengths[3]=128;
    MrgSortSrcList<float> src;
    src.src1=dst; src.src2=dst[256U]; src.src3=dst[512U]; src.src4=dst[768U];
    MrgSort<float>(tmp,src,p); PipeBarrier<PIPE_V>();
    DataCopy(dst,tmp,CHUNK*2U); PipeBarrier<PIPE_V>();
}

class MtpMissUnion {
public:
    __aicore__ inline void Init(GM_ADDR p0,GM_ADDR p1,GM_ADDR slots,GM_ADDR candidates,
                                GM_ADDR scores,GM_ADDR thresholds,GM_ADDR cacheSlots,
                                GM_ADDR reqEntries,GM_ADDR cacheTokens,GM_ADDR topkSrc,
                                GM_ADDR out,GM_ADDR missDst,GM_ADDR evictSrc,GM_ADDR counts,
                                const optiling::FusedLiManageMtpUnionTilingData *t, TPipe *pipe) {
        pair0Gm.SetGlobalBuffer((__gm__ float*)p0); pair1Gm.SetGlobalBuffer((__gm__ float*)p1);
        slotsGm.SetGlobalBuffer((__gm__ int32_t*)slots); candidatesGm.SetGlobalBuffer((__gm__ int32_t*)candidates);
        scoresGm.SetGlobalBuffer((__gm__ float*)scores); thresholdsGm.SetGlobalBuffer((__gm__ float*)thresholds);
        cacheSlotsGm.SetGlobalBuffer((__gm__ int32_t*)cacheSlots);
        reqEntriesGm.SetGlobalBuffer((__gm__ int32_t*)reqEntries);
        cacheTokensGm.SetGlobalBuffer((__gm__ int32_t*)cacheTokens);
        topkSrcGm.SetGlobalBuffer((__gm__ int32_t*)topkSrc);
        outGm.SetGlobalBuffer((__gm__ int32_t*)out); missDstGm.SetGlobalBuffer((__gm__ int32_t*)missDst);
        evictSrcGm.SetGlobalBuffer((__gm__ int32_t*)evictSrc); countsGm.SetGlobalBuffer((__gm__ int32_t*)counts);
        batchSize=t->batchSize; sourceCapacity=t->sourceCapacity;
        pipe->InitBuffer(pairInBuf,MERGE_PAIR_FLOATS*sizeof(float));
        pipe->InitBuffer(pairOutBuf,MERGE_PAIR_FLOATS*sizeof(float));
        pipe->InitBuffer(workBuf,7168U*sizeof(float));
        pipe->InitBuffer(countBuf,32U);
    }
    __aicore__ inline void Process() {
        for(uint32_t b=GetBlockIdx();b<batchSize;b+=GetBlockNum()) ProcessBatch(b);
    }
private:
    __aicore__ inline uint32_t BuildUnion(uint32_t b, LocalTensor<float> input,
                                          LocalTensor<float> merged, LocalTensor<int32_t> result,
                                          LocalTensor<int32_t> work) {
        uint32_t len[ROUTES]={0U,0U,0U,0U};
        for(uint32_t r=0;r<ROUTES;++r) {
            uint64_t base=(static_cast<uint64_t>(b)*ROUTES+r)*TOPK;
            DataCopy(work,slotsGm[base],TOPK); Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            while(len[r]<TOPK && work.GetValue(len[r])<0) ++len[r];
        }
        uint32_t total=len[0]+len[1]+len[2]+len[3];
        if(total==0U) return 0U;
        uint64_t base=static_cast<uint64_t>(b)*CAPACITY;
        DataCopy(input,pair0Gm[base],CAPACITY); DataCopy(input[CAPACITY],pair1Gm[base],CAPACITY);
        Sync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        MrgSort4Info p;
        p.elementLengths[0]=len[0]; p.elementLengths[1]=len[1];
        p.elementLengths[2]=len[2]; p.elementLengths[3]=len[3];
        p.ifExhaustedSuspension=false;
        p.validBit=(len[0]>0U?1U:0U)|(len[1]>0U?2U:0U)|(len[2]>0U?4U:0U)|(len[3]>0U?8U:0U);
        p.repeatTimes=1;
        MrgSortSrcList<float> src;
        src.src1=input; src.src2=input[PAIR_WORDS]; src.src3=input[PAIR_WORDS*2U]; src.src4=input[PAIR_WORDS*3U];
        MrgSort<float>(merged,src,p); Sync<HardEvent::V_S>(HardEvent::V_S);
        uint32_t mask=candidatesGm.GetValue(b)>static_cast<int32_t>(SHORT_MASK+1U)?LONG_MASK:SHORT_MASK;
        int32_t last=-1; uint32_t count=0U; LocalTensor<uint32_t> bits=merged.ReinterpretCast<uint32_t>();
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
        return count;
    }

    __aicore__ inline void BuildEvictChunk(uint32_t b,uint32_t row,uint32_t start,uint32_t valid,
                                           LocalTensor<float> chunkPair,LocalTensor<float> work) {
        LocalTensor<float> s0=work, s1=work[CHUNK], s2=work[CHUNK*2U], s3=work[CHUNK*3U];
        LocalTensor<float> key=work[CHUNK*4U], temp=work[CHUNK*5U];
        LocalTensor<int32_t> cache=work[CHUNK*6U].ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> payload=work[CHUNK*7U].ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> mask=work[CHUNK*8U].ReinterpretCast<uint8_t>();
        LocalTensor<float> invalid=work[CHUNK*9U];
        LocalTensor<float> sortTmp=work[CHUNK*10U];
        for(uint32_t r=0;r<ROUTES;++r) {
            LocalTensor<float> dst=(r==0?s0:(r==1?s1:(r==2?s2:s3)));
            DataCopyPad(dst,scoresGm[(static_cast<uint64_t>(b)*ROUTES+r)*sourceCapacity+start],
                        AscendC::DataCopyExtParams{
                            1,static_cast<uint32_t>(valid*sizeof(float)),0,0,0},
                        AscendC::DataCopyPadExtParams<float>{true,0,static_cast<uint8_t>((8U-valid%8U)%8U),0.0f});
        }
        DataCopyPad(cache,cacheSlotsGm[static_cast<uint64_t>(row)*sourceCapacity+start],
                    AscendC::DataCopyExtParams{
                        1,static_cast<uint32_t>(valid*sizeof(int32_t)),0,0,0},
                    AscendC::DataCopyPadExtParams<int32_t>{false,0,0,0});
        Sync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Adds(key,s0,-thresholdsGm.GetValue(static_cast<uint64_t>(b)*ROUTES),valid); PipeBarrier<PIPE_V>();
        Adds(temp,s1,-thresholdsGm.GetValue(static_cast<uint64_t>(b)*ROUTES+1U),valid); PipeBarrier<PIPE_V>();
        Max(key,key,temp,valid); PipeBarrier<PIPE_V>();
        Adds(temp,s2,-thresholdsGm.GetValue(static_cast<uint64_t>(b)*ROUTES+2U),valid); PipeBarrier<PIPE_V>();
        Max(key,key,temp,valid); PipeBarrier<PIPE_V>();
        Adds(temp,s3,-thresholdsGm.GetValue(static_cast<uint64_t>(b)*ROUTES+3U),valid); PipeBarrier<PIPE_V>();
        Max(key,key,temp,valid); PipeBarrier<PIPE_V>();
        Muls(key,key,-1.0f,valid); PipeBarrier<PIPE_V>();
        Duplicate(invalid.ReinterpretCast<int32_t>(),NEG_INF_BITS,CHUNK); PipeBarrier<PIPE_V>();
        CompareScalar(mask,key,0.0f,CMPMODE::LE,valid); PipeBarrier<PIPE_V>();
        Select(key,mask,invalid,key,SELMODE::VSEL_TENSOR_TENSOR_MODE,valid); PipeBarrier<PIPE_V>();
        CompareScalar(mask,cache,-1,CMPMODE::EQ,valid); PipeBarrier<PIPE_V>();
        Select(key,mask,invalid,key,SELMODE::VSEL_TENSOR_TENSOR_MODE,valid); PipeBarrier<PIPE_V>();
        ArithProgression<uint32_t>(payload,start,1U,CHUNK); PipeBarrier<PIPE_V>();
        if(valid<CHUNK) { Duplicate(key.ReinterpretCast<int32_t>()[valid],NEG_INF_BITS,CHUNK-valid); PipeBarrier<PIPE_V>(); }
        SortChunk(chunkPair,key,payload,sortTmp);
    }

    __aicore__ inline void FindEvicts(uint32_t b,uint32_t count,LocalTensor<float> acc,
                                      LocalTensor<float> tmp,LocalTensor<float> work,
                                      LocalTensor<int32_t> result) {
        if(count==0U) return;
        uint32_t candidateCap=((count+CHUNK-1U)/CHUNK)*CHUNK;
        if(candidateCap>CAPACITY) candidateCap=CAPACITY;
        Duplicate(acc.ReinterpretCast<int32_t>(),NEG_INF_BITS,candidateCap*2U); PipeBarrier<PIPE_V>();
        uint32_t actual=static_cast<uint32_t>(candidatesGm.GetValue(b));
        uint32_t chunks=(actual+CHUNK-1U)/CHUNK;
        uint32_t row=static_cast<uint32_t>(reqEntriesGm.GetValue(b));
        uint32_t seed=(actual*2654435761U)^(row*2246822519U);
        uint32_t begin=chunks==0U?0U:seed%chunks;
        uint32_t stop=chunks;
        LocalTensor<float> chunkPair=work[CHUNK*12U];
        for(uint32_t scan=0;scan<chunks;++scan) {
            uint32_t chunk=(begin+scan)%chunks, start=chunk*CHUNK;
            uint32_t valid=start+CHUNK>actual?actual-start:CHUNK;
            BuildEvictChunk(b,row,start,valid,chunkPair,work);
            MrgSort4Info p; p.elementLengths[0]=candidateCap; p.elementLengths[1]=CHUNK;
            p.elementLengths[2]=0; p.elementLengths[3]=0; p.ifExhaustedSuspension=false;
            p.validBit=0b0011; p.repeatTimes=1;
            MrgSortSrcList<float> src; src.src1=acc; src.src2=chunkPair; src.src3=chunkPair; src.src4=chunkPair;
            MrgSort<float>(tmp,src,p); PipeBarrier<PIPE_V>();
            DataCopy(acc,tmp,candidateCap*2U); PipeBarrier<PIPE_V>();
            if(stop==chunks) {
                Sync<HardEvent::V_S>(HardEvent::V_S);
                if(acc.GetValue((count-1U)*2U)>0.0f) stop=(scan+5U<chunks)?scan+5U:chunks;
            } else if(scan+1U>=stop) break;
        }
        Sync<HardEvent::V_S>(HardEvent::V_S);
        LocalTensor<uint32_t> bits=acc.ReinterpretCast<uint32_t>();
        uint32_t budget=static_cast<uint32_t>(cacheTokensGm.GetValue(b));
        for(uint32_t i=0;i<count;++i) {
            int32_t src=static_cast<int32_t>(bits.GetValue(i*2U+1U));
            int32_t slot=(src>=0&&static_cast<uint32_t>(src)<actual)?
                         cacheSlotsGm.GetValue(static_cast<uint64_t>(row)*sourceCapacity+src):-1;
            if(slot<0||static_cast<uint32_t>(slot)>=budget) { src=-1; slot=-1; }
            result.SetValue(i,src); result.SetValue(CAPACITY+i,slot);
        }
        Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(evictSrcGm[static_cast<uint64_t>(b)*CAPACITY],result,
                    {1,static_cast<uint16_t>(count*sizeof(int32_t)),0,0});
        DataCopyPad(missDstGm[static_cast<uint64_t>(b)*CAPACITY],result[CAPACITY],
                    {1,static_cast<uint16_t>(count*sizeof(int32_t)),0,0});
        Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void ProcessBatch(uint32_t b) {
        LocalTensor<float> input=pairInBuf.Get<float>(), merged=pairOutBuf.Get<float>();
        LocalTensor<float> work=workBuf.Get<float>();
        LocalTensor<int32_t> result=input.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> countLocal=countBuf.Get<int32_t>();
        uint32_t count=BuildUnion(b,input,merged,result,work.ReinterpretCast<int32_t>());
        FindEvicts(b,count,merged,input,work,result);
        countLocal.SetValue(0,static_cast<int32_t>(count));
        Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(countsGm[b],countLocal,{1,static_cast<uint16_t>(sizeof(int32_t)),0,0});
    }
    GlobalTensor<float> pair0Gm,pair1Gm,scoresGm,thresholdsGm;
    GlobalTensor<int32_t> slotsGm,candidatesGm,cacheSlotsGm,reqEntriesGm,cacheTokensGm,topkSrcGm;
    GlobalTensor<int32_t> outGm,missDstGm,evictSrcGm,countsGm;
    TBuf<TPosition::VECCALC> pairInBuf,pairOutBuf,workBuf,countBuf;
    uint32_t batchSize=0U,sourceCapacity=0U;
};
}
extern "C" __global__ __aicore__ void nanovllm_fused_li_manage_mtp_union(
    GM_ADDR pair0,GM_ADDR pair1,GM_ADDR slots,GM_ADDR candidates,
    GM_ADDR scores,GM_ADDR thresholds,GM_ADDR cacheSlots,GM_ADDR reqEntries,
    GM_ADDR cacheTokens,GM_ADDR topkSrc,GM_ADDR out,GM_ADDR missDst,
    GM_ADDR evictSrc,GM_ADDR counts,GM_ADDR workspace,GM_ADDR tiling) {
    (void)workspace; KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(optiling::FusedLiManageMtpUnionTilingData,data,tiling);
    TPipe pipe; MtpMissUnion op;
    op.Init(pair0,pair1,slots,candidates,scores,thresholds,cacheSlots,reqEntries,
            cacheTokens,topkSrc,out,missDst,evictSrc,counts,&data,&pipe);
    op.Process();
}
