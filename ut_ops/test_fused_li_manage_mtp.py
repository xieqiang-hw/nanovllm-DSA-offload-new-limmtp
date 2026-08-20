"""Correctness and latency comparison for the phase-1 MTP4 LI TopK op."""

from __future__ import annotations

import argparse
import statistics

import torch
import torch_npu

import nanovllm.ops  # noqa: F401 - loads torch custom-op registrations
from _op_utils import require_local_opapi


MTP_WIDTH = 4
HEADS = 32
HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 2048
UNION_CAPACITY = MTP_WIDTH * TOPK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-latency-ratio", type=float, default=1.25)
    return parser.parse_args()


def make_case(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    if args.seq_len < TOPK or args.seq_len % BLOCK_SIZE:
        raise ValueError("--seq-len must be a multiple of 128 and >= 2048")
    if args.cache_tokens < TOPK or args.cache_tokens > args.seq_len:
        raise ValueError("--cache-tokens must be in [2048, seq_len]")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    batch = args.batch_size
    blocks_per_request = args.seq_len // BLOCK_SIZE
    total_blocks = batch * blocks_per_request

    query = torch.randn(batch * MTP_WIDTH, HEADS, HEAD_DIM, dtype=dtype, device=device)
    weights = torch.randn(batch * MTP_WIDTH, HEADS, dtype=dtype, device=device)
    key = torch.randn(total_blocks, BLOCK_SIZE, 1, HEAD_DIM, dtype=dtype, device=device)
    block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).view(batch, blocks_per_request)
    candidate_lens = torch.full((batch,), args.seq_len, dtype=torch.int32, device=device)
    cache_tokens = torch.full((batch,), args.cache_tokens, dtype=torch.int32, device=device)
    req_entries = torch.arange(batch, dtype=torch.int32, device=device)
    cache_slots = torch.full((batch, args.seq_len), -1, dtype=torch.int32, device=device)
    topk_src = torch.full((batch * MTP_WIDTH, 1, TOPK), -1, dtype=torch.int32, device=device)
    topk_dst = torch.full(topk_src.shape, -7, dtype=torch.int32, device=device)
    miss_src = torch.full((batch, UNION_CAPACITY), -1, dtype=torch.int32, device=device)
    miss_dst = torch.full(miss_src.shape, -1, dtype=torch.int32, device=device)
    miss_counts = torch.full((batch,), -1, dtype=torch.int32, device=device)
    # TND standard-LI convention: cumulative query lengths for B sequences.
    query_lens = torch.arange(MTP_WIDTH, batch * MTP_WIDTH + 1, MTP_WIDTH,
                              dtype=torch.int32, device=device)
    return locals()


def call_standard(case: dict[str, torch.Tensor]) -> torch.Tensor:
    out = torch_npu.npu_lightning_indexer(
        query=case["query"], key=case["key"], weights=case["weights"],
        actual_seq_lengths_query=case["query_lens"],
        actual_seq_lengths_key=case["candidate_lens"],
        block_table=case["block_table"], layout_query="TND",
        # The custom interface defines num_candidate_tokens as the same
        # prefill full-block range for all four MTP routes, so compare the
        # unmasked LI score/TopK semantics rather than right-down causal S1.
        layout_key="PA_BSND", sparse_count=TOPK, sparse_mode=0,
    )
    return out[0] if isinstance(out, (tuple, list)) else out


def call_mtp(case: dict[str, torch.Tensor]) -> None:
    torch.ops.nanovllm_dsa.fused_li_manage_mtp.default(
        case["query"], case["weights"], case["key"], case["block_table"],
        case["candidate_lens"], case["cache_tokens"], case["req_entries"],
        case["cache_slots"], case["topk_src"], case["topk_dst"],
        case["miss_src"], case["miss_dst"], case["miss_counts"],
    )


def benchmark(fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    return statistics.mean(samples), statistics.median(samples)


def main() -> None:
    args = parse_args()
    require_local_opapi()
    if not callable(getattr(torch_npu, "npu_lightning_indexer", None)):
        raise RuntimeError("torch_npu.npu_lightning_indexer is unavailable")
    case = make_case(args)
    reference = call_standard(case).reshape(args.batch_size * MTP_WIDTH, TOPK)

    # Build a deterministic mixed hit/miss cache from the standard TopK itself.
    # This guarantees coverage of legal slot 0 without making every TopK token a hit.
    cached_per_request = min(args.cache_tokens, TOPK // 2)
    for batch_idx in range(args.batch_size):
        route0 = reference[batch_idx * MTP_WIDTH]
        cached_ids = route0[:cached_per_request]
        case["cache_slots"][batch_idx, cached_ids] = torch.arange(
            cached_per_request, dtype=torch.int32, device=case["query"].device
        )
    old_cache = case["cache_slots"].clone()

    reference_slots = []
    for route_idx in range(args.batch_size * MTP_WIDTH):
        batch_idx = route_idx // MTP_WIDTH
        reference_slots.append(case["cache_slots"][batch_idx, reference[route_idx].long()])
    reference_slots = torch.stack(reference_slots)
    # Lexicographic order: miss group first (slot < 0), then source ID ascending.
    reorder_key = (reference_slots >= 0).to(torch.float32) * float(args.seq_len + 1)
    reorder_key = reorder_key + reference.to(torch.float32)
    reorder = torch.argsort(reorder_key, dim=-1)
    expected_src = torch.gather(reference, 1, reorder)
    expected_slots = torch.gather(reference_slots, 1, reorder)

    call_mtp(case)
    torch.npu.synchronize()

    actual = case["topk_src"].reshape(args.batch_size * MTP_WIDTH, TOPK)
    actual_slots = case["topk_dst"].reshape(args.batch_size * MTP_WIDTH, TOPK)
    src_mismatch = int((expected_src != actual).sum().item())
    slot_mismatch = int((expected_slots != actual_slots).sum().item())
    if src_mismatch or slot_mismatch:
        raise AssertionError(
            "MTP TopK hit/miss differs from the standard-LI-derived reference: "
            f"src_mismatches={src_mismatch}, slot_mismatches={slot_mismatch}"
        )
    if int((actual_slots == 0).sum().item()) == 0:
        raise AssertionError("test case did not exercise legal HBM slot 0")
    expected_union = []
    expected_counts = []
    for batch_idx in range(args.batch_size):
        route_begin = batch_idx * MTP_WIDTH
        route_end = route_begin + MTP_WIDTH
        batch_misses = expected_src[route_begin:route_end][
            expected_slots[route_begin:route_end] < 0
        ]
        batch_union = torch.unique(batch_misses, sorted=True)
        expected_union.append(batch_union)
        expected_counts.append(batch_union.numel())
    actual_counts = case["miss_counts"].cpu().tolist()
    if actual_counts != expected_counts:
        raise AssertionError(
            f"MTP union miss counts differ: actual={actual_counts}, expected={expected_counts}"
        )
    for batch_idx, batch_union in enumerate(expected_union):
        count = expected_counts[batch_idx]
        actual_union = case["miss_src"][batch_idx, :count]
        if not torch.equal(actual_union, batch_union):
            mismatch = int((actual_union != batch_union).sum().item())
            raise AssertionError(
                f"MTP union miss IDs differ for batch {batch_idx}: mismatches={mismatch}"
            )
    if not torch.equal(case["cache_slots"], old_cache):
        raise AssertionError("phase-1 MTP TopK modified cache_slots_pool")
    standard_mean, standard_p50 = benchmark(lambda: call_standard(case), args.warmup, args.iters)
    mtp_mean, mtp_p50 = benchmark(lambda: call_mtp(case), args.warmup, args.iters)
    ratio = mtp_mean / standard_mean
    print(
        "FUSED_LI_MANAGE_MTP_RESULT "
        f"batch={args.batch_size} routes=4 seq_len={args.seq_len} dtype={args.dtype} "
        f"standard_mean_us={standard_mean:.3f} standard_p50_us={standard_p50:.3f} "
        f"mtp_mean_us={mtp_mean:.3f} mtp_p50_us={mtp_p50:.3f} ratio={ratio:.4f} "
        f"warmup={args.warmup} iters={args.iters} topk_hit_miss_reorder_match=1 "
        f"union_miss_match=1 cache_unchanged=1",
        flush=True,
    )
    if ratio > args.max_latency_ratio:
        raise AssertionError(
            f"MTP latency ratio {ratio:.4f} exceeds limit {args.max_latency_ratio:.4f}"
        )
    print("FUSED_LI_MANAGE_MTP_UT_OK", flush=True)


if __name__ == "__main__":
    main()
