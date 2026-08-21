#ifndef FUSED_LI_MANAGE_MTP_UNION_H
#define FUSED_LI_MANAGE_MTP_UNION_H

#include "kernel_operator.h"

namespace MtpUnion {
using namespace AscendC;

constexpr uint32_t ROUTES = 4U;
constexpr uint32_t TOPK = 2048U;
constexpr uint32_t PAIR_WORDS = 4096U;
constexpr uint32_t CAPACITY = 8192U;
constexpr uint32_t SHORT_MASK = (1U << 18U) - 1U;
constexpr uint32_t LONG_MASK = (1U << 21U) - 1U;
constexpr uint32_t MISS_KEY_BASE_BITS = 0x40000000U;

template <HardEvent event>
__aicore__ inline void Sync(HardEvent e)
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(e));
    AscendC::SetFlag<event>(id);
    AscendC::WaitFlag<event>(id);
}

class MtpMissUnion {
public:
    __aicore__ inline void Init(GM_ADDR pair0, GM_ADDR pair1,
                                GM_ADDR candidates, GM_ADDR out, GM_ADDR counts,
                                uint32_t batch, TPipe *pipe)
    {
        pair0Gm.SetGlobalBuffer((__gm__ float *)pair0);
        pair1Gm.SetGlobalBuffer((__gm__ float *)pair1);
        candidatesGm.SetGlobalBuffer((__gm__ int32_t *)candidates);
        outGm.SetGlobalBuffer((__gm__ int32_t *)out);
        countsGm.SetGlobalBuffer((__gm__ int32_t *)counts);
        batchSize = batch;
        pipe->InitBuffer(pairInBuf, CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(pairOutBuf, CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(countBuf, 32U);
    }

    __aicore__ inline void Process(uint32_t first, uint32_t stride)
    {
        for (uint32_t batch = first; batch < batchSize; batch += stride) {
            ProcessBatch(batch);
        }
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t batch)
    {
        LocalTensor<float> input = pairInBuf.Get<float>();
        LocalTensor<float> merged = pairOutBuf.Get<float>();
        LocalTensor<int32_t> result = input.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> countLocal = countBuf.Get<int32_t>();
        uint64_t pairBase = static_cast<uint64_t>(batch) * CAPACITY;
        DataCopy(input, pair0Gm[pairBase], CAPACITY);
        DataCopy(input[CAPACITY], pair1Gm[pairBase], CAPACITY);
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        uint32_t lengths[ROUTES] = {0U, 0U, 0U, 0U};
        LocalTensor<uint32_t> pairBits = input.ReinterpretCast<uint32_t>();
        for (uint32_t route = 0; route < ROUTES; ++route) {
            // SortTopkBySlotIndex places miss pairs first.  Miss keys are in
            // [MISS_KEY_BASE_BITS, +inf), while hit keys are below that range,
            // so the route miss prefix can be located with a binary search.
            uint32_t routeOffset = route * PAIR_WORDS;
            uint32_t low = 0U;
            uint32_t high = TOPK;
            while (low < high) {
                uint32_t middle = (low + high) >> 1U;
                uint32_t key = pairBits.GetValue(routeOffset + middle * 2U);
                if (key >= MISS_KEY_BASE_BITS) {
                    low = middle + 1U;
                } else {
                    high = middle;
                }
            }
            lengths[route] = low;
        }

        uint32_t total = lengths[0] + lengths[1] + lengths[2] + lengths[3];
        if (total == 0U) {
            countLocal.SetValue(0, 0);
            Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(countsGm[batch], countLocal,
                        {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
            return;
        }

        Sync<HardEvent::S_V>(HardEvent::S_V);

        MrgSort4Info params;
        params.elementLengths[0] = lengths[0];
        params.elementLengths[1] = lengths[1];
        params.elementLengths[2] = lengths[2];
        params.elementLengths[3] = lengths[3];
        params.ifExhaustedSuspension = false;
        params.validBit = (lengths[0] > 0U ? 0b0001 : 0U) |
                          (lengths[1] > 0U ? 0b0010 : 0U) |
                          (lengths[2] > 0U ? 0b0100 : 0U) |
                          (lengths[3] > 0U ? 0b1000 : 0U);
        params.repeatTimes = 1;
        MrgSortSrcList<float> sources;
        sources.src1 = input;
        sources.src2 = input[PAIR_WORDS];
        sources.src3 = input[PAIR_WORDS * 2U];
        sources.src4 = input[PAIR_WORDS * 3U];
        MrgSort<float>(merged, sources, params);
        Sync<HardEvent::V_S>(HardEvent::V_S);

        uint32_t mask = candidatesGm.GetValue(batch) > static_cast<int32_t>(SHORT_MASK + 1U)
                            ? LONG_MASK
                            : SHORT_MASK;
        uint32_t count = 0U;
        int32_t last = -1;
        LocalTensor<uint32_t> bits = merged.ReinterpretCast<uint32_t>();
        for (uint32_t index = 0; index < total; ++index) {
            uint32_t key = bits.GetValue(index * 2U);
            int32_t sourceId = static_cast<int32_t>(mask - (key - MISS_KEY_BASE_BITS));
            if (sourceId != last) {
                result.SetValue(count++, sourceId);
                last = sourceId;
            }
        }

        if (count != 0U) {
            Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(outGm[static_cast<uint64_t>(batch) * CAPACITY], result,
                        {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
            Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        countLocal.SetValue(0, static_cast<int32_t>(count));
        Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(countsGm[batch], countLocal,
                    {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
    }

    GlobalTensor<float> pair0Gm;
    GlobalTensor<float> pair1Gm;
    GlobalTensor<int32_t> candidatesGm;
    GlobalTensor<int32_t> outGm;
    GlobalTensor<int32_t> countsGm;
    TBuf<TPosition::VECCALC> pairInBuf;
    TBuf<TPosition::VECCALC> pairOutBuf;
    TBuf<TPosition::VECCALC> countBuf;
    uint32_t batchSize = 0U;
};
} // namespace MtpUnion

#endif
