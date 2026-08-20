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
                uint32_t route = half * 2U + localRoute;
                if (lengths[route] == 0U) continue;
                Cast(keys[localRoute * TOPK], ids[route * TOPK], RoundMode::CAST_NONE, lengths[route]);
                PipeBarrier<PIPE_V>();
                Muls(keys[localRoute * TOPK], keys[localRoute * TOPK], -1.0F, lengths[route]);
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

        uint32_t totalMiss = lengths[0] + lengths[1] + lengths[2] + lengths[3];
        LocalTensor<int32_t> mergedPairs = src.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> result = dst.template ReinterpretCast<int32_t>();
        bool globallySorted = true;
        int32_t previous = -1;
        for (uint32_t i = 0; i < totalMiss; ++i) {
            int32_t current = mergedPairs.GetValue(i * 2U + 1U);
            if (current < previous) globallySorted = false;
            previous = current;
        }
        uint32_t count = 0U;
        int32_t last = -1;
        if (globallySorted) {
            for (uint32_t i = 0; i < totalMiss; ++i) {
                int32_t current = mergedPairs.GetValue(i * 2U + 1U);
                if (current != last) {
                    result.SetValue(count++, current);
                    last = current;
                }
            }
        } else {
            uint32_t positions[ROUTES] = {0U, 0U, 0U, 0U};
            int32_t heads[ROUTES] = {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX};
            for (uint32_t r = 0; r < ROUTES; ++r) {
                if (lengths[r] != 0U) heads[r] = ids.GetValue(r * TOPK);
            }
            while (positions[0] < lengths[0] || positions[1] < lengths[1] ||
                   positions[2] < lengths[2] || positions[3] < lengths[3]) {
                int32_t next = INT32_MAX;
                for (uint32_t r = 0; r < ROUTES; ++r) next = heads[r] < next ? heads[r] : next;
                if (next != last) { result.SetValue(count++, next); last = next; }
                for (uint32_t r = 0; r < ROUTES; ++r) {
                    if (heads[r] == next) {
                        ++positions[r];
                        heads[r] = positions[r] < lengths[r]
                            ? ids.GetValue(r * TOPK + positions[r]) : INT32_MAX;
                    }
                }
            }
        }
        if (count != 0U) {
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(outGm[static_cast<uint64_t>(b) * CAPACITY], result,
                        {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        slots.SetValue(0, static_cast<int32_t>(count));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(countsGm[b], slots, {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
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
