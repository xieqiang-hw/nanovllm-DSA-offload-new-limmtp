#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "fused_li_manage_mtp_union_tiling.h"
using namespace AscendC;
namespace {
constexpr uint32_t ROUTES = 4U;
constexpr uint32_t TOPK = 2048U;
constexpr uint32_t CAPACITY = ROUTES * TOPK;
template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}
class MtpMissUnion {
public:
    __aicore__ inline void Init(GM_ADDR ids, GM_ADDR slots, GM_ADDR out, GM_ADDR counts,
                                const optiling::FusedLiManageMtpUnionTilingData *tiling, TPipe *pipe)
    {
        idsGm.SetGlobalBuffer((__gm__ int32_t *)ids);
        slotsGm.SetGlobalBuffer((__gm__ int32_t *)slots);
        outGm.SetGlobalBuffer((__gm__ int32_t *)out);
        countsGm.SetGlobalBuffer((__gm__ int32_t *)counts);
        batchSize = tiling->batchSize;
        pipe->InitBuffer(idsBuf, CAPACITY * sizeof(int32_t));
        pipe->InitBuffer(keysBuf, (CAPACITY / 2U) * sizeof(float));
        pipe->InitBuffer(pair0Buf, CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(pair1Buf, CAPACITY * 2U * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t b = GetBlockIdx(); b < batchSize; b += GetBlockNum()) ProcessBatch(b);
    }
private:
    __aicore__ inline void ProcessBatch(uint32_t b)
    {
        LocalTensor<int32_t> ids = idsBuf.Get<int32_t>();
        LocalTensor<uint32_t> idsU = ids.template ReinterpretCast<uint32_t>();
        LocalTensor<float> keys = keysBuf.Get<float>();
        LocalTensor<int32_t> slots = keys.template ReinterpretCast<int32_t>();
        LocalTensor<float> pair0 = pair0Buf.Get<float>();
        LocalTensor<float> pair1 = pair1Buf.Get<float>();
        uint32_t lengths[ROUTES] = {0U, 0U, 0U, 0U};
        Duplicate(ids, -1, CAPACITY);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < ROUTES; ++r) {
            uint64_t base = (static_cast<uint64_t>(b) * ROUTES + r) * TOPK;
            DataCopy(slots, slotsGm[base], TOPK);
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            while (lengths[r] < TOPK && slots.GetValue(lengths[r]) < 0) ++lengths[r];
            if (lengths[r] != 0U) {
                DataCopyPad(ids[r * TOPK], idsGm[base],
                    {1, lengths[r] * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0},
                    {false, 0, 0, 0});
            }
        }
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        constexpr uint32_t HALF_CAPACITY = CAPACITY / 2U;
        constexpr uint32_t HALF_SORT_REPEATS = HALF_CAPACITY / 32U;
        for (uint32_t half = 0; half < 2U; ++half) {
            Duplicate(keys, -3.402823466e+38F, HALF_CAPACITY);
            PipeBarrier<PIPE_V>();
            for (uint32_t localRoute = 0; localRoute < 2U; ++localRoute) {
                uint32_t r = half * 2U + localRoute;
                if (lengths[r] == 0U) continue;
                Cast(keys[localRoute * TOPK], ids[r * TOPK], RoundMode::CAST_NONE, lengths[r]);
                PipeBarrier<PIPE_V>();
                Muls(keys[localRoute * TOPK], keys[localRoute * TOPK], -1.0F, lengths[r]);
                PipeBarrier<PIPE_V>();
            }
            Sort32(pair0[half * HALF_CAPACITY * 2U], keys,
                   ids.template ReinterpretCast<uint32_t>()[half * HALF_CAPACITY], HALF_SORT_REPEATS);
            PipeBarrier<PIPE_V>();
        }
        uint32_t groups = CAPACITY / 32U;
        uint32_t elements = 32U;
        LocalTensor<float> src = pair0;
        LocalTensor<float> dst = pair1;
        while (groups > 1U) {
            MrgSort4Info params;
            params.elementLengths[0] = elements;
            params.elementLengths[1] = elements;
            params.elementLengths[2] = elements;
            params.elementLengths[3] = elements;
            params.ifExhaustedSuspension = false;
            params.validBit = 0b1111;
            params.repeatTimes = groups / 4U;
            MrgSortSrcList<float> list;
            list.src1 = src[0];
            list.src2 = src[2U * elements];
            list.src3 = src[4U * elements];
            list.src4 = src[6U * elements];
            MrgSort(dst, list, params);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> swap = src; src = dst; dst = swap;
            groups /= 4U;
            elements *= 4U;
        }

        GatherMaskParams extract{1, static_cast<uint8_t>(CAPACITY / 64U), 8, 0};
        uint64_t ignored = 0;
        GatherMask(idsU, src.template ReinterpretCast<uint32_t>(), static_cast<uint8_t>(2), false, 0U, extract, ignored);
        GatherMask(idsU[CAPACITY / 2U], src.template ReinterpretCast<uint32_t>()[CAPACITY],
                   static_cast<uint8_t>(2), false, 0U, extract, ignored);
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> previous = dst.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> offsets = dst.template ReinterpretCast<int32_t>()[CAPACITY];
        CreateVecIndex(offsets, static_cast<int32_t>(0), CAPACITY);
        PipeBarrier<PIPE_V>();
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(int32_t)), CAPACITY);
        PipeBarrier<PIPE_V>();
        Adds(offsets, offsets, -static_cast<int32_t>(sizeof(int32_t)), CAPACITY);
        PipeBarrier<PIPE_V>();
        offsets.SetValue(0, 0);
        Gather(previous, ids, offsets.template ReinterpretCast<uint32_t>(), 0U, CAPACITY);
        PipeBarrier<PIPE_V>();
        previous.SetValue(0, -1);
        LocalTensor<uint8_t> uniqueMask = keys.template ReinterpretCast<uint8_t>();
        LocalTensor<uint8_t> validMask = uniqueMask[1024U];
        Compare(uniqueMask, ids, previous, CMPMODE::NE, CAPACITY);
        PipeBarrier<PIPE_V>();
        CompareScalar(validMask, ids, -1, CMPMODE::GT, CAPACITY);
        PipeBarrier<PIPE_V>();
        And(uniqueMask.template ReinterpretCast<uint32_t>(), uniqueMask.template ReinterpretCast<uint32_t>(),
            validMask.template ReinterpretCast<uint32_t>(), CAPACITY / 32U);
        PipeBarrier<PIPE_V>();
        uint64_t uniqueCount = 0;
        GatherMask(dst.template ReinterpretCast<int32_t>()[CAPACITY], ids,
                   uniqueMask.template ReinterpretCast<uint32_t>(), true, CAPACITY,
                   {1, 1, 8, 1}, uniqueCount);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(outGm[static_cast<uint64_t>(b) * CAPACITY], dst.template ReinterpretCast<int32_t>()[CAPACITY],
                    {1, static_cast<uint16_t>(uniqueCount * sizeof(int32_t)), 0, 0});
        previous.SetValue(0, static_cast<int32_t>(uniqueCount));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(countsGm[b], previous, {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
    }
    GlobalTensor<int32_t> idsGm, slotsGm, outGm, countsGm;
    TBuf<TPosition::VECCALC> idsBuf, keysBuf, pair0Buf, pair1Buf;
    uint32_t batchSize = 0U;
};
}
extern "C" __global__ __aicore__ void nanovllm_fused_li_manage_mtp_union(
    GM_ADDR ids, GM_ADDR slots, GM_ADDR out, GM_ADDR counts, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(optiling::FusedLiManageMtpUnionTilingData, data, tiling);
    TPipe pipe;
    MtpMissUnion op;
    op.Init(ids, slots, out, counts, &data, &pipe);
    op.Process();
}
