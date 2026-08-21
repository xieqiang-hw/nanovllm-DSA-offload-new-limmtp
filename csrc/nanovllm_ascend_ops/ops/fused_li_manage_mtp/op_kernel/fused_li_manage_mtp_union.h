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
constexpr uint32_t EVICT_CHUNK = 512U;
constexpr uint32_t EVICT_SORT_REPEATS = EVICT_CHUNK / 32U;
constexpr int32_t NEG_INF_BITS = static_cast<int32_t>(0xFF800000U);

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
                                GM_ADDR candidates, GM_ADDR cacheSlots,
                                GM_ADDR reqEntries, GM_ADDR scoreScratch,
                                GM_ADDR thresholdScratch, GM_ADDR out,
                                GM_ADDR counts, uint32_t batch,
                                uint32_t scoreCapacity,
                                uint32_t cacheCapacity, TPipe *pipe)
    {
        pair0Gm.SetGlobalBuffer((__gm__ float *)pair0);
        pair1Gm.SetGlobalBuffer((__gm__ float *)pair1);
        candidatesGm.SetGlobalBuffer((__gm__ int32_t *)candidates);
        cacheSlotsGm.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
        reqEntriesGm.SetGlobalBuffer((__gm__ int32_t *)reqEntries);
        scoreScratchGm.SetGlobalBuffer((__gm__ float *)scoreScratch);
        thresholdScratchGm.SetGlobalBuffer((__gm__ float *)thresholdScratch);
        outGm.SetGlobalBuffer((__gm__ int32_t *)out);
        countsGm.SetGlobalBuffer((__gm__ int32_t *)counts);
        batchSize = batch;
        scoreStride = ((scoreCapacity + EVICT_CHUNK - 1U) / EVICT_CHUNK) * EVICT_CHUNK;
        sourceCapacity = cacheCapacity;
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
    // This is the MTP extension of non-MTP SortEvictCandidateChunk.  It keeps
    // the same 512-entry Sort32/MrgSort pair representation; only construction
    // of the key changes from one score threshold to the intersection of four.
    __aicore__ inline void SortEvictChunk(LocalTensor<float> dst,
                                          LocalTensor<float> key,
                                          LocalTensor<uint32_t> payload,
                                          LocalTensor<float> tmp)
    {
        Sort32(tmp, key, payload, EVICT_SORT_REPEATS);
        PipeBarrier<PIPE_V>();
        MrgSort4Info params;
        params.elementLengths[0] = 32U;
        params.elementLengths[1] = 32U;
        params.elementLengths[2] = 32U;
        params.elementLengths[3] = 32U;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;
        params.repeatTimes = 1;
        for (uint32_t group = 0U; group < 4U; ++group) {
            uint32_t offset = group * 256U;
            MrgSortSrcList<float> sources;
            sources.src1 = tmp[offset];
            sources.src2 = tmp[offset + 64U];
            sources.src3 = tmp[offset + 128U];
            sources.src4 = tmp[offset + 192U];
            MrgSort<float>(dst[offset], sources, params);
        }
        PipeBarrier<PIPE_V>();
        params.elementLengths[0] = 128U;
        params.elementLengths[1] = 128U;
        params.elementLengths[2] = 128U;
        params.elementLengths[3] = 128U;
        MrgSortSrcList<float> sources;
        sources.src1 = dst;
        sources.src2 = dst[256U];
        sources.src3 = dst[512U];
        sources.src4 = dst[768U];
        MrgSort<float>(tmp, sources, params);
        PipeBarrier<PIPE_V>();
        DataCopy(dst, tmp, EVICT_CHUNK * 2U);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildEvictCandidateChunk(
        uint32_t batch, uint32_t cacheRow, uint32_t start, uint32_t valid,
        LocalTensor<float> chunkPair, LocalTensor<float> scratch)
    {
        LocalTensor<float> score0 = scratch;
        LocalTensor<float> score1 = scratch[EVICT_CHUNK];
        LocalTensor<float> score2 = scratch[EVICT_CHUNK * 2U];
        LocalTensor<float> score3 = scratch[EVICT_CHUNK * 3U];
        LocalTensor<float> key = scratch[EVICT_CHUNK * 4U];
        LocalTensor<float> temp = scratch[EVICT_CHUNK * 5U];
        LocalTensor<int32_t> slots = scratch[EVICT_CHUNK * 6U].ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> payload = scratch[EVICT_CHUNK * 7U].ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> invalidMask = scratch[EVICT_CHUNK * 8U].ReinterpretCast<uint8_t>();
        LocalTensor<float> invalidKey = scratch[EVICT_CHUNK * 9U];
        LocalTensor<float> sortTmp = scratch[EVICT_CHUNK * 10U];

        for (uint32_t route = 0U; route < ROUTES; ++route) {
            LocalTensor<float> score = route == 0U ? score0 :
                                       (route == 1U ? score1 :
                                        (route == 2U ? score2 : score3));
            uint64_t routeIndex = static_cast<uint64_t>(batch) * ROUTES + route;
            DataCopyPad(score, scoreScratchGm[routeIndex * scoreStride + start],
                        AscendC::DataCopyExtParams{
                            1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0},
                        AscendC::DataCopyPadExtParams<float>{
                            true, 0, static_cast<uint8_t>((8U - valid % 8U) % 8U), 0.0F});
        }
        DataCopyPad(slots,
                    cacheSlotsGm[static_cast<uint64_t>(cacheRow) * sourceCapacity + start],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(valid * sizeof(int32_t)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        Sync<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        // margin=max_r(score_r-threshold_r). A candidate is valid only when
        // it is cached and margin<0, i.e. below every route's TopK threshold.
        Adds(key, score0, -thresholdScratchGm.GetValue(batch * ROUTES), valid);
        PipeBarrier<PIPE_V>();
        Adds(temp, score1, -thresholdScratchGm.GetValue(batch * ROUTES + 1U), valid);
        PipeBarrier<PIPE_V>();
        Max(key, key, temp, valid);
        PipeBarrier<PIPE_V>();
        Adds(temp, score2, -thresholdScratchGm.GetValue(batch * ROUTES + 2U), valid);
        PipeBarrier<PIPE_V>();
        Max(key, key, temp, valid);
        PipeBarrier<PIPE_V>();
        Adds(temp, score3, -thresholdScratchGm.GetValue(batch * ROUTES + 3U), valid);
        PipeBarrier<PIPE_V>();
        Max(key, key, temp, valid);
        PipeBarrier<PIPE_V>();
        Muls(key, key, -1.0F, valid);
        PipeBarrier<PIPE_V>();

        Duplicate(invalidKey.ReinterpretCast<int32_t>(), NEG_INF_BITS, EVICT_CHUNK);
        PipeBarrier<PIPE_V>();
        CompareScalar(invalidMask, key, 0.0F, CMPMODE::LE, valid);
        PipeBarrier<PIPE_V>();
        Select(key, invalidMask, invalidKey, key,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
        PipeBarrier<PIPE_V>();
        CompareScalar(invalidMask, slots, 0, CMPMODE::LT, valid);
        PipeBarrier<PIPE_V>();
        Select(key, invalidMask, invalidKey, key,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
        PipeBarrier<PIPE_V>();
        ArithProgression<int32_t>(payload.ReinterpretCast<int32_t>(),
                                  static_cast<int32_t>(start), 1, EVICT_CHUNK);
        PipeBarrier<PIPE_V>();
        if (valid < EVICT_CHUNK) {
            Duplicate(key.ReinterpretCast<int32_t>()[valid], NEG_INF_BITS,
                      EVICT_CHUNK - valid);
            PipeBarrier<PIPE_V>();
        }
        SortEvictChunk(chunkPair, key, payload, sortTmp);
    }

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
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> reqEntriesGm;
    GlobalTensor<float> scoreScratchGm;
    GlobalTensor<float> thresholdScratchGm;
    GlobalTensor<int32_t> outGm;
    GlobalTensor<int32_t> countsGm;
    TBuf<TPosition::VECCALC> pairInBuf;
    TBuf<TPosition::VECCALC> pairOutBuf;
    TBuf<TPosition::VECCALC> countBuf;
    uint32_t batchSize = 0U;
    uint32_t scoreStride = 0U;
    uint32_t sourceCapacity = 0U;
};
} // namespace MtpUnion

#endif
