#ifndef FUSED_LI_MANAGE_MTP_WORKSPACE_H
#define FUSED_LI_MANAGE_MTP_WORKSPACE_H

#include <cstdint>

namespace MtpWorkspace {
constexpr uint64_t S2_BASE_SIZE = 512U;
constexpr uint64_t M_BASE_SIZE = 256U;
constexpr uint64_t MM1_ELEM_BYTES = sizeof(float);
constexpr uint64_t DOUBLE_BUFFER = 2U;
constexpr uint64_t S1_BASE_SIZE = 8U;
constexpr uint64_t LD_HEAD_TAIL = 2U;
constexpr uint64_t VALUE_AND_INDEX = 2U;
constexpr uint64_t TOPK = 2048U;
constexpr uint64_t LD_PARAM_NUM = 16U;
constexpr uint64_t ROUTES = 4U;

__aicore__ inline uint64_t ScoreStride(uint64_t sourceCapacity)
{
    return ((sourceCapacity + S2_BASE_SIZE - 1U) / S2_BASE_SIZE) * S2_BASE_SIZE;
}

__aicore__ inline uint64_t ScoreOffset(uint64_t blockNum)
{
    uint64_t offset = blockNum * DOUBLE_BUFFER * M_BASE_SIZE * S2_BASE_SIZE * MM1_ELEM_BYTES;
    offset += blockNum * S1_BASE_SIZE * LD_HEAD_TAIL * VALUE_AND_INDEX * TOPK * sizeof(float);
    offset += blockNum * S1_BASE_SIZE * LD_HEAD_TAIL * LD_PARAM_NUM * sizeof(int64_t);
    return offset;
}

__aicore__ inline uint64_t ThresholdOffset(uint64_t blockNum, uint64_t batch, uint64_t sourceCapacity)
{
    return ScoreOffset(blockNum) + batch * ROUTES * ScoreStride(sourceCapacity) * sizeof(float);
}
} // namespace MtpWorkspace

#endif
