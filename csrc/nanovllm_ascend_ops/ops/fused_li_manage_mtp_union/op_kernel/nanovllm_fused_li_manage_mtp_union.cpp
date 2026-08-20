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
        pipe->InitBuffer(outBuf, CAPACITY * sizeof(int32_t));
        pipe->InitBuffer(slotsBuf, TOPK * sizeof(int32_t));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t b = GetBlockIdx(); b < batchSize; b += GetBlockNum()) ProcessBatch(b);
    }
private:
    __aicore__ inline void ProcessBatch(uint32_t b)
    {
        LocalTensor<int32_t> ids = idsBuf.Get<int32_t>();
        LocalTensor<int32_t> result = outBuf.Get<int32_t>();
        LocalTensor<int32_t> slots = slotsBuf.Get<int32_t>();
        uint32_t lengths[ROUTES] = {0U, 0U, 0U, 0U};
        uint32_t positions[ROUTES] = {0U, 0U, 0U, 0U};
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
        uint32_t count = 0U;
        int32_t last = -1;
        while (positions[0] < lengths[0] || positions[1] < lengths[1] ||
               positions[2] < lengths[2] || positions[3] < lengths[3]) {
            int32_t next = INT32_MAX;
            for (uint32_t r = 0; r < ROUTES; ++r) {
                if (positions[r] < lengths[r]) {
                    int32_t value = ids.GetValue(r * TOPK + positions[r]);
                    next = value < next ? value : next;
                }
            }
            if (next != last) { result.SetValue(count++, next); last = next; }
            for (uint32_t r = 0; r < ROUTES; ++r) {
                while (positions[r] < lengths[r] && ids.GetValue(r * TOPK + positions[r]) == next) ++positions[r];
            }
        }
        if (count != 0U) {
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(outGm[static_cast<uint64_t>(b) * CAPACITY], result,
                        {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
        }
        countsGm.SetValue(b, static_cast<int32_t>(count));
    }
    GlobalTensor<int32_t> idsGm, slotsGm, outGm, countsGm;
    TBuf<TPosition::VECCALC> idsBuf, outBuf, slotsBuf;
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
