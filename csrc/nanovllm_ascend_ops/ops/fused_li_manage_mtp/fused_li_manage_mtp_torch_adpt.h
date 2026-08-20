#ifndef NANOVLLM_FUSED_LI_MANAGE_MTP_TORCH_ADPT_H
#define NANOVLLM_FUSED_LI_MANAGE_MTP_TORCH_ADPT_H

namespace vllm_ascend {
inline void npu_fused_li_manage_mtp(
    const at::Tensor& query, const at::Tensor& index_weights,
    const at::Tensor& index_key_cache, const at::Tensor& index_block_table,
    const at::Tensor& num_candidate_tokens, const at::Tensor& num_cache_tokens,
    const at::Tensor& req_pool_entries, at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids, at::Tensor topk_dst_slots,
    at::Tensor miss_src_ids, at::Tensor miss_dst_slots, at::Tensor miss_counts) {
  constexpr int64_t kMtpWidth = 4;
  constexpr int64_t kTopK = 2048;
  constexpr int64_t kUnionCapacity = 8192;
  TORCH_CHECK(query.dim() == 3 && query.size(0) % kMtpWidth == 0 &&
                  query.size(1) == 32 && query.size(2) == 128,
              "Fused LI Manage MTP query must be [B*4, 32, 128].");
  const int64_t batch = query.size(0) / kMtpWidth;
  TORCH_CHECK(index_weights.dim() == 2 && index_weights.size(0) == query.size(0) &&
                  index_weights.size(1) == 32,
              "Fused LI Manage MTP weights must be [B*4, 32].");
  TORCH_CHECK(index_key_cache.dim() == 4 && index_key_cache.size(1) == 128 &&
                  index_key_cache.size(2) == 1 && index_key_cache.size(3) == 128,
              "Fused LI Manage MTP key must be [blocks, 128, 1, 128].");
  TORCH_CHECK(index_block_table.dim() == 2 && index_block_table.size(0) == batch,
              "Fused LI Manage MTP block table must be [B, INDEX_MAX_BLOCKS].");
  TORCH_CHECK(num_candidate_tokens.sizes() == at::IntArrayRef({batch}) &&
                  num_cache_tokens.sizes() == at::IntArrayRef({batch}) &&
                  req_pool_entries.sizes() == at::IntArrayRef({batch}),
              "Fused LI Manage MTP request metadata must be [B].");
  TORCH_CHECK(cache_slots_pool.dim() == 2 &&
                  cache_slots_pool.size(1) == index_block_table.size(1) * 128,
              "Fused LI Manage MTP cache pool width must match block table capacity.");
  TORCH_CHECK(topk_src_ids.sizes() == at::IntArrayRef({batch * 4, 1, kTopK}) &&
                  topk_dst_slots.sizes() == topk_src_ids.sizes(),
              "Fused LI Manage MTP TopK outputs must be [B*4, 1, 2048].");
  TORCH_CHECK(miss_src_ids.sizes() == at::IntArrayRef({batch, kUnionCapacity}) &&
                  miss_dst_slots.sizes() == miss_src_ids.sizes() &&
                  miss_counts.sizes() == at::IntArrayRef({batch}),
              "Fused LI Manage MTP miss outputs must be [B, 8192]/[B].");
  TORCH_CHECK(query.scalar_type() == index_weights.scalar_type() &&
                  query.scalar_type() == index_key_cache.scalar_type() &&
                  (query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16),
              "Fused LI Manage MTP query/key/weights must share fp16 or bf16 dtype.");
  TORCH_CHECK(index_block_table.scalar_type() == at::kInt &&
                  num_candidate_tokens.scalar_type() == at::kInt &&
                  num_cache_tokens.scalar_type() == at::kInt &&
                  req_pool_entries.scalar_type() == at::kInt &&
                  cache_slots_pool.scalar_type() == at::kInt &&
                  topk_src_ids.scalar_type() == at::kInt &&
                  topk_dst_slots.scalar_type() == at::kInt &&
                  miss_src_ids.scalar_type() == at::kInt &&
                  miss_dst_slots.scalar_type() == at::kInt &&
                  miss_counts.scalar_type() == at::kInt,
              "Fused LI Manage MTP metadata and outputs must be int32.");
  TORCH_CHECK(query.is_contiguous() && index_weights.is_contiguous() &&
                  index_key_cache.is_contiguous() && index_block_table.is_contiguous() &&
                  num_candidate_tokens.is_contiguous() && num_cache_tokens.is_contiguous() &&
                  req_pool_entries.is_contiguous() && cache_slots_pool.is_contiguous() &&
                  topk_src_ids.is_contiguous() && topk_dst_slots.is_contiguous() &&
                  miss_src_ids.is_contiguous() && miss_dst_slots.is_contiguous() &&
                  miss_counts.is_contiguous(),
              "Fused LI Manage MTP tensors must be contiguous.");
  TORCH_CHECK(query.device().is_privateuseone(), "Fused LI Manage MTP tensors must be on NPU.");
  const auto device = query.device();
  TORCH_CHECK(index_weights.device() == device && index_key_cache.device() == device &&
                  index_block_table.device() == device && num_candidate_tokens.device() == device &&
                  num_cache_tokens.device() == device && req_pool_entries.device() == device &&
                  cache_slots_pool.device() == device && topk_src_ids.device() == device &&
                  topk_dst_slots.device() == device && miss_src_ids.device() == device &&
                  miss_dst_slots.device() == device && miss_counts.device() == device,
              "Fused LI Manage MTP tensors must be on the same NPU.");

  auto keepalive = std::make_tuple(
      query, index_key_cache, index_weights, req_pool_entries, cache_slots_pool,
      num_cache_tokens, num_candidate_tokens, index_block_table, topk_src_ids,
      topk_dst_slots, miss_src_ids, miss_dst_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnNanovllmFusedLiManageMtp, keepalive,
      query, index_key_cache, index_weights, req_pool_entries,
      cache_slots_pool, num_cache_tokens, num_candidate_tokens,
      index_block_table, topk_src_ids, topk_dst_slots,
      miss_src_ids, miss_dst_slots, miss_counts, cache_slots_pool);
}
} // namespace vllm_ascend
#endif
