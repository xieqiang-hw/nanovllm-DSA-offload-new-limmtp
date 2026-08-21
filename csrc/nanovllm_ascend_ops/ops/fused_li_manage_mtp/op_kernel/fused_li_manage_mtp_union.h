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
constexpr uint32_t INDEX_BITS = 18U;
constexpr uint32_t INDEX_MASK = (1U << INDEX_BITS) - 1U;
constexpr uint32_t INDEX_HIGH_BITS = 3U;
constexpr uint32_t SCORE_TAG_CLEAR_SHIFT = INDEX_HIGH_BITS;
constexpr uint32_t MISS_KEY_BASE_BITS = 0x40000000U;
constexpr uint32_t EVICT_CHUNK = 512U;
constexpr uint32_t EVICT_SORT_REPEATS = EVICT_CHUNK / 32U;
constexpr uint32_t EVICT_CAPACITY = 2048U;
constexpr uint32_t EVICT_EXTRA_SCAN_CHUNKS = 4U;
constexpr uint32_t THRESHOLD_STRIDE = 8U;
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
                                GM_ADDR cacheTokens, GM_ADDR reqEntries, GM_ADDR scoreScratch,
                                GM_ADDR thresholdScratch, GM_ADDR out,
                                GM_ADDR counts, uint32_t batch,
                                uint32_t scoreCapacity,
                                uint32_t cacheCapacity, TPipe *pipe)
    {
        pair0Gm.SetGlobalBuffer((__gm__ float *)pair0);
        pair1Gm.SetGlobalBuffer((__gm__ float *)pair1);
        evictSlotsGm.SetGlobalBuffer((__gm__ int32_t *)pair1);
        candidatesGm.SetGlobalBuffer((__gm__ int32_t *)candidates);
        cacheSlotsGm.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
        cacheTokensGm.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
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
    __aicore__ inline uint32_t HashEvictScanSeed(uint32_t actual, uint32_t cacheRow)
    {
        uint32_t value = actual ^ ((cacheRow + 1U) * 0x9e3779b9U);
        value ^= value >> 16U;
        value *= 0x7feb352dU;
        value ^= value >> 15U;
        value *= 0x846ca68bU;
        value ^= value >> 16U;
        return value;
    }

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
        bool hasLongIndexTag, LocalTensor<float> chunkPair,
        LocalTensor<float> scratch)
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
        Adds(key, score0,
             -thresholdScratchGm.GetValue(batch * ROUTES * THRESHOLD_STRIDE), valid);
        PipeBarrier<PIPE_V>();
        Adds(temp, score1,
             -thresholdScratchGm.GetValue((batch * ROUTES + 1U) * THRESHOLD_STRIDE), valid);
        PipeBarrier<PIPE_V>();
        Max(key, key, temp, valid);
        PipeBarrier<PIPE_V>();
        Adds(temp, score2,
             -thresholdScratchGm.GetValue((batch * ROUTES + 2U) * THRESHOLD_STRIDE), valid);
        PipeBarrier<PIPE_V>();
        Max(key, key, temp, valid);
        PipeBarrier<PIPE_V>();
        Adds(temp, score3,
             -thresholdScratchGm.GetValue((batch * ROUTES + 3U) * THRESHOLD_STRIDE), valid);
        PipeBarrier<PIPE_V>();
        Max(key, key, temp, valid);
        PipeBarrier<PIPE_V>();
        Muls(key, key, -1.0F, valid);
        PipeBarrier<PIPE_V>();

        CompareScalar(invalidMask, key, 0.0F, CMPMODE::LE, valid);
        PipeBarrier<PIPE_V>();
        if (hasLongIndexTag) {
            ShiftRight(key.ReinterpretCast<uint32_t>(), key.ReinterpretCast<uint32_t>(),
                       SCORE_TAG_CLEAR_SHIFT, valid);
            PipeBarrier<PIPE_V>();
            ShiftLeft(key.ReinterpretCast<uint32_t>(), key.ReinterpretCast<uint32_t>(),
                      SCORE_TAG_CLEAR_SHIFT, valid);
            PipeBarrier<PIPE_V>();
            Adds(key.ReinterpretCast<int32_t>(), key.ReinterpretCast<int32_t>(),
                 static_cast<int32_t>((start >> INDEX_BITS) &
                                      ((1U << INDEX_HIGH_BITS) - 1U)), valid);
            PipeBarrier<PIPE_V>();
        }
        Duplicate(invalidKey.ReinterpretCast<int32_t>(), NEG_INF_BITS, EVICT_CHUNK);
        PipeBarrier<PIPE_V>();
        Select(key, invalidMask, invalidKey, key,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
        PipeBarrier<PIPE_V>();
        CompareScalar(invalidMask, slots, 0, CMPMODE::LT, valid);
        PipeBarrier<PIPE_V>();
        Select(key, invalidMask, invalidKey, key,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
        PipeBarrier<PIPE_V>();
        // Match the non-MTP eviction codec: slot14 occupies the upper bits,
        // index_low18 occupies the payload low bits, and index_high3 is kept
        // in the score key's low three bits for long source sequences.
        ArithProgression<int32_t>(payload.ReinterpretCast<int32_t>(),
                                  static_cast<int32_t>(start & INDEX_MASK), 1,
                                  EVICT_CHUNK);
        PipeBarrier<PIPE_V>();
        ShiftLeft(slots.ReinterpretCast<uint32_t>(), slots.ReinterpretCast<uint32_t>(),
                  INDEX_BITS, valid);
        PipeBarrier<PIPE_V>();
        Add(payload.ReinterpretCast<int32_t>(), payload.ReinterpretCast<int32_t>(),
            slots, valid);
        PipeBarrier<PIPE_V>();
        if (valid < EVICT_CHUNK) {
            Duplicate(key.ReinterpretCast<int32_t>()[valid], NEG_INF_BITS,
                      EVICT_CHUNK - valid);
            PipeBarrier<PIPE_V>();
        }
        SortEvictChunk(chunkPair, key, payload, sortTmp);
    }

    __aicore__ inline void MergeEvictCandidateChunk(
        LocalTensor<float> accumulator, LocalTensor<float> chunkPair,
        uint32_t candidateCap, LocalTensor<float> tmp)
    {
        uint32_t candidateBlocks = candidateCap / EVICT_CHUNK;
        if (candidateBlocks == 2U || candidateBlocks == 3U) {
            LIServiceVec::MrgBasicBlock(tmp, accumulator,
                                        static_cast<int64_t>(candidateBlocks + 1U),
                                        EVICT_CHUNK);
            PipeBarrier<PIPE_V>();
            DataCopy(accumulator, tmp, candidateCap * 2U);
            PipeBarrier<PIPE_V>();
            return;
        }
        LIServiceVec::MergeSort(accumulator, static_cast<int32_t>(candidateCap),
                                chunkPair, EVICT_CHUNK, tmp);
    }

    __aicore__ inline bool FindEvictCandidates(uint32_t batch, uint32_t count,
                                                LocalTensor<float> input,
                                                LocalTensor<float> merged)
    {
        if (count == 0U || count > EVICT_CAPACITY) {
            return count == 0U;
        }
        uint32_t candidateCap =
            ((count + EVICT_CHUNK - 1U) / EVICT_CHUNK) * EVICT_CHUNK;
        LocalTensor<float> accumulator = merged;
        Duplicate(accumulator.ReinterpretCast<int32_t>(), NEG_INF_BITS,
                  candidateCap * 2U);
        PipeBarrier<PIPE_V>();

        uint32_t actual = static_cast<uint32_t>(candidatesGm.GetValue(batch));
        uint32_t cacheRow = static_cast<uint32_t>(reqEntriesGm.GetValue(batch));
        uint32_t chunks = (actual + EVICT_CHUNK - 1U) / EVICT_CHUNK;
        if (chunks == 0U) {
            return false;
        }
        uint32_t startChunk = HashEvictScanSeed(actual, cacheRow) % chunks;
        uint32_t stopScan = chunks;
        bool hasLongIndexTag = actual > INDEX_MASK + 1U;

        if (candidateCap == EVICT_CHUNK) {
            // Same four-chunk batching used by the non-MTP 512 fast path.
            LocalTensor<float> chunkBlocks = input;
            LocalTensor<float> scratch = input[EVICT_CHUNK * 8U];
            LocalTensor<float> batchMerged = input[EVICT_CHUNK * 20U];
            LocalTensor<float> mergeTmp = input[EVICT_CHUNK * 28U];
            uint32_t scan = 0U;
            while (scan < chunks) {
                uint32_t batchChunks = chunks - scan < 4U ? chunks - scan : 4U;
                if (stopScan != chunks && batchChunks > stopScan - scan) {
                    batchChunks = stopScan - scan;
                }
                for (uint32_t item = 0U; item < batchChunks; ++item) {
                    uint32_t chunk = (startChunk + scan + item) % chunks;
                    uint32_t start = chunk * EVICT_CHUNK;
                    uint32_t valid = start + EVICT_CHUNK > actual
                                         ? actual - start
                                         : EVICT_CHUNK;
                    BuildEvictCandidateChunk(batch, cacheRow, start, valid,
                                             hasLongIndexTag,
                                             chunkBlocks[item * EVICT_CHUNK * 2U],
                                             scratch);
                }
                if (batchChunks == 1U) {
                    LIServiceVec::MergeSort(accumulator, EVICT_CHUNK, chunkBlocks,
                                            EVICT_CHUNK, mergeTmp);
                } else {
                    LIServiceVec::MrgBasicBlock(batchMerged, chunkBlocks,
                                                static_cast<int64_t>(batchChunks),
                                                EVICT_CHUNK);
                    PipeBarrier<PIPE_V>();
                    LIServiceVec::MergeSort(accumulator, EVICT_CHUNK, batchMerged,
                                            EVICT_CHUNK, mergeTmp);
                }
                scan += batchChunks;
                Sync<HardEvent::V_S>(HardEvent::V_S);
                if (stopScan == chunks &&
                    accumulator.GetValue((count - 1U) * 2U) > 0.0F) {
                    uint32_t remaining = chunks - scan;
                    uint32_t extra = remaining < EVICT_EXTRA_SCAN_CHUNKS
                                         ? remaining
                                         : EVICT_EXTRA_SCAN_CHUNKS;
                    if (extra == 0U) {
                        break;
                    }
                    stopScan = scan + extra;
                } else if (stopScan != chunks && scan >= stopScan) {
                    break;
                }
            }
        } else {
            LocalTensor<float> scratch = input;
            LocalTensor<float> mergeTmp = input[EVICT_CHUNK * 12U];
            LocalTensor<float> chunkPair = accumulator[candidateCap * 2U];
            for (uint32_t scan = 0U; scan < chunks; ++scan) {
                uint32_t chunk = (startChunk + scan) % chunks;
                uint32_t start = chunk * EVICT_CHUNK;
                uint32_t valid = start + EVICT_CHUNK > actual
                                     ? actual - start
                                     : EVICT_CHUNK;
                BuildEvictCandidateChunk(batch, cacheRow, start, valid,
                                         hasLongIndexTag,
                                         chunkPair, scratch);
                MergeEvictCandidateChunk(accumulator, chunkPair,
                                         candidateCap, mergeTmp);
                Sync<HardEvent::V_S>(HardEvent::V_S);
                if (stopScan == chunks &&
                    accumulator.GetValue((count - 1U) * 2U) > 0.0F) {
                    uint32_t remaining = chunks - scan - 1U;
                    uint32_t extra = remaining < EVICT_EXTRA_SCAN_CHUNKS
                                         ? remaining
                                         : EVICT_EXTRA_SCAN_CHUNKS;
                    if (extra == 0U) {
                        break;
                    }
                    stopScan = scan + 1U + extra;
                } else if (stopScan != chunks && scan + 1U >= stopScan) {
                    break;
                }
            }
        }
        Sync<HardEvent::V_S>(HardEvent::V_S);
        return accumulator.GetValue((count - 1U) * 2U) > 0.0F;
    }

    // Consume the sorted prefix with the same scalar validation pattern as
    // non-MTP UpdateCacheAndWriteTopkSlots.  MTP additionally rechecks the
    // four-route threshold intersection before accepting a packed candidate;
    // the vector filter is an optimization, not the final correctness guard.
    __aicore__ inline void WriteEvictSlots(uint32_t batch, uint32_t count,
                                           bool found,
                                           LocalTensor<float> candidates,
                                           LocalTensor<int32_t> output)
    {
        if (count == 0U) {
            return;
        }
        for (uint32_t index = 0U; index < count; ++index) {
            output.SetValue(index, -1);
        }
        if (found) {
            uint32_t actual = static_cast<uint32_t>(candidatesGm.GetValue(batch));
            uint32_t cacheTokenCount = static_cast<uint32_t>(cacheTokensGm.GetValue(batch));
            uint32_t cacheRow = static_cast<uint32_t>(reqEntriesGm.GetValue(batch));
            uint64_t cacheRowBase = static_cast<uint64_t>(cacheRow) * sourceCapacity;
            uint32_t candidateCap =
                ((count + EVICT_CHUNK - 1U) / EVICT_CHUNK) * EVICT_CHUNK;
            bool hasLongIndexTag = actual > INDEX_MASK + 1U;
            LocalTensor<uint32_t> candidateBits = candidates.ReinterpretCast<uint32_t>();
            uint32_t cursor = 0U;
            uint32_t accepted = 0U;
            while (cursor < candidateCap && accepted < count) {
                float candidateKey = candidates.GetValue(cursor * 2U);
                if (candidateKey <= 0.0F) {
                    break;
                }
                uint32_t keyBits = candidateBits.GetValue(cursor * 2U);
                uint32_t payload = candidateBits.GetValue(cursor * 2U + 1U);
                ++cursor;
                uint32_t sourceIndex = payload & INDEX_MASK;
                if (hasLongIndexTag) {
                    sourceIndex |= (keyBits & ((1U << INDEX_HIGH_BITS) - 1U)) << INDEX_BITS;
                }
                int32_t slot = static_cast<int32_t>(payload >> INDEX_BITS);
                if (sourceIndex >= actual || slot < 0 ||
                    static_cast<uint32_t>(slot) >= cacheTokenCount) {
                    continue;
                }
                // Sort32/MrgSort must carry the packed slot and source index
                // as one payload.  Re-read the persistent map before trusting
                // the pair so a key/payload or payload-field mismatch cannot
                // evict a different token from the one validated below.
                if (cacheSlotsGm.GetValue(cacheRowBase + sourceIndex) != slot) {
                    continue;
                }
                bool belowAllThresholds = true;
                for (uint32_t route = 0U; route < ROUTES; ++route) {
                    uint64_t routeIndex = static_cast<uint64_t>(batch) * ROUTES + route;
                    float score = scoreScratchGm.GetValue(routeIndex * scoreStride + sourceIndex);
                    float threshold = thresholdScratchGm.GetValue(routeIndex * THRESHOLD_STRIDE);
                    if (score >= threshold) {
                        belowAllThresholds = false;
                        break;
                    }
                }
                bool duplicateSlot = false;
                for (uint32_t selected = 0U; selected < accepted; ++selected) {
                    if (output.GetValue(selected) == slot) {
                        duplicateSlot = true;
                        break;
                    }
                }
                if (belowAllThresholds && !duplicateSlot) {
                    output.SetValue(accepted++, slot);
                }
            }
        }
        Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(evictSlotsGm[static_cast<uint64_t>(batch) * CAPACITY], output,
                    {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
        Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
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
        bool found = FindEvictCandidates(batch, count, input, merged);
        WriteEvictSlots(batch, count, found, merged, result);
    }

    GlobalTensor<float> pair0Gm;
    GlobalTensor<float> pair1Gm;
    GlobalTensor<int32_t> evictSlotsGm;
    GlobalTensor<int32_t> candidatesGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> cacheTokensGm;
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
