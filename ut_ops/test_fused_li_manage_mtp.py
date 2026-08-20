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
    parser.add_argument("--source-len", "--seq-len", dest="seq_len", type=int, default=20992)
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--perf-query-miss-count", type=int, default=200)
    parser.add_argument("--perf-query-noise", type=float, default=0.25)
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument("--skip-boundary-tests", action="store_true",
                        help="skip the small correctness-only edge-case matrix")
    parser.add_argument("--max-latency-ratio", type=float, default=1.5)
    return parser.parse_args()


def make_case(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    if args.seq_len < TOPK or args.seq_len % BLOCK_SIZE:
        raise ValueError("--seq-len must be a multiple of 128 and >= 2048")
    if args.cache_tokens < TOPK or args.cache_tokens > args.seq_len:
        raise ValueError("--cache-tokens must be in [2048, seq_len]")
    if not 0 <= args.perf_query_miss_count <= TOPK:
        raise ValueError("--perf-query-miss-count must be in [0, 2048]")
    if args.perf_query_noise <= 0 or args.graph_replays < 0:
        raise ValueError("--perf-query-noise must be positive and --graph-replays >= 0")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    batch = args.batch_size
    blocks_per_request = args.seq_len // BLOCK_SIZE
    total_blocks = batch * blocks_per_request

    base_query = torch.randn(batch, 1, HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    query_noise = torch.randn(batch, MTP_WIDTH, HEADS, HEAD_DIM,
                              dtype=torch.float32, device=device)
    query = (base_query + args.perf_query_noise * query_noise).reshape(
        batch * MTP_WIDTH, HEADS, HEAD_DIM).to(dtype)
    base_weights = torch.rand(batch, 1, HEADS, dtype=torch.float32, device=device)
    weights = base_weights.expand(-1, MTP_WIDTH, -1).reshape(
        batch * MTP_WIDTH, HEADS).contiguous().to(dtype)
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


def build_balanced_cache(case: dict[str, torch.Tensor], reference: torch.Tensor,
                         per_query_misses: int) -> tuple[list[int], list[int]]:
    """Create exactly N misses per route with shared MTP3 misses where possible."""
    batch = case["batch"]
    source_len = case["args"].seq_len
    budget = case["args"].cache_tokens
    reference_cpu = reference.cpu().to(torch.int64)
    union_sizes: list[int] = []
    union_miss_counts: list[int] = []
    for request in range(batch):
        rows = reference_cpu[request * MTP_WIDTH:(request + 1) * MTP_WIDTH]
        union = torch.unique(rows.flatten(), sorted=True)
        union_sizes.append(int(union.numel()))
        if union.numel() > budget:
            raise AssertionError(
                f"request={request}: TopK union={union.numel()} exceeds cache budget={budget}"
            )
        membership = torch.zeros(source_len, dtype=torch.uint8)
        for route, row in enumerate(rows):
            membership[row] |= 1 << route
        by_mask = {
            mask: torch.nonzero(membership == mask).flatten().to(torch.int64)
            for mask in range(1, 1 << MTP_WIDTH)
        }
        common_count = min(per_query_misses // 2, int(by_mask[0b1111].numel()))
        selected = [by_mask[0b1111][:common_count]]
        pair_degree = per_query_misses - common_count
        opposite_pairs = ((0b0011, 0b1100), (0b0101, 0b1010), (0b1001, 0b0110))
        capacities = [min(int(by_mask[a].numel()), int(by_mask[b].numel()))
                      for a, b in opposite_pairs]
        if sum(capacities) < pair_degree:
            raise AssertionError(
                f"request={request}: insufficient shared-miss capacity={capacities}; "
                "increase --perf-query-noise"
            )
        pair_counts = [min(pair_degree // 3, capacity) for capacity in capacities]
        left = pair_degree - sum(pair_counts)
        while left:
            for idx, capacity in enumerate(capacities):
                if pair_counts[idx] < capacity:
                    pair_counts[idx] += 1
                    left -= 1
                    if left == 0:
                        break
        for (a, b), count in zip(opposite_pairs, pair_counts):
            selected.extend((by_mask[a][:count], by_mask[b][:count]))
        misses = torch.cat(selected) if selected else torch.empty(0, dtype=torch.int64)
        route_counts = [int(torch.isin(row, misses).sum()) for row in rows]
        if route_counts != [per_query_misses] * MTP_WIDTH:
            raise AssertionError(
                f"request={request}: per-route misses={route_counts}, "
                f"expected={[per_query_misses] * MTP_WIDTH}"
            )
        union_miss_counts.append(int(misses.numel()))
        miss_mask = torch.zeros(source_len, dtype=torch.bool)
        miss_mask[misses] = True
        hits = union[~miss_mask[union]]
        union_mask = torch.zeros(source_len, dtype=torch.bool)
        union_mask[union] = True
        fillers = torch.arange(source_len, dtype=torch.int64)[~union_mask]
        needed = budget - int(hits.numel())
        if needed < 0 or needed > fillers.numel():
            raise AssertionError(f"request={request}: cannot construct cache budget={budget}")
        cached = torch.cat((hits, fillers[:needed]))
        # Hits are stored before fillers, so slot 0 is guaranteed to appear in TopK.
        slots = torch.arange(budget, dtype=torch.int32)
        case["cache_slots"][request, cached.to(case["query"].device)] = slots.to(
            case["query"].device)
    return union_sizes, union_miss_counts


def run_graph_check(case: dict[str, torch.Tensor], replays: int) -> None:
    if replays == 0:
        return
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        call_mtp(case)
    torch.npu.synchronize()
    for replay in range(replays):
        # Mutate data in-place while retaining captured addresses. This catches
        # accidental value specialization and stale-output reuse.
        case["query"].add_(torch.tensor((replay + 1) * 1e-3,
                                        dtype=case["query"].dtype,
                                        device=case["query"].device))
        graph.replay()
        torch.npu.synchronize()
        captured = {name: case[name].clone() for name in
                    ("topk_src", "topk_dst", "miss_src", "miss_dst", "miss_counts")}
        call_mtp(case)
        torch.npu.synchronize()
        mismatches = []
        for name in ("topk_src", "topk_dst", "miss_counts"):
            if not torch.equal(captured[name], case[name]):
                mismatches.append(name)
        captured_counts = captured["miss_counts"].cpu().tolist()
        eager_counts = case["miss_counts"].cpu().tolist()
        if captured_counts == eager_counts:
            for request, count in enumerate(captured_counts):
                for name in ("miss_src", "miss_dst"):
                    if not torch.equal(captured[name][request, :count],
                                       case[name][request, :count]):
                        mismatches.append(f"{name}[{request},:{count}]")
        if mismatches:
            raise AssertionError(
                f"ACLGraph replay outputs differ from eager outputs at replay={replay}: "
                f"fields={mismatches}")
    print(f"FUSED_LI_MANAGE_MTP_GRAPH_CHECK replays={replays} dynamic_query=1 ok=1",
          flush=True)


def expected_outputs(case: dict[str, torch.Tensor], reference: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Derive reorder and union results from standard LI and the persistent map."""
    flat_reference = reference.reshape(case["batch"] * MTP_WIDTH, TOPK)
    reference_slots = []
    for route_idx in range(case["batch"] * MTP_WIDTH):
        request = route_idx // MTP_WIDTH
        pool_row = int(case["req_entries"][request].item())
        reference_slots.append(
            case["cache_slots"][pool_row, flat_reference[route_idx].long()])
    reference_slots = torch.stack(reference_slots)
    reorder_key = ((reference_slots >= 0).to(torch.float32) *
                   float(case["args"].seq_len + 1) + flat_reference.to(torch.float32))
    reorder = torch.argsort(reorder_key, dim=-1)
    expected_src = torch.gather(flat_reference, 1, reorder)
    expected_slots = torch.gather(reference_slots, 1, reorder)
    unions = []
    for request in range(case["batch"]):
        begin = request * MTP_WIDTH
        end = begin + MTP_WIDTH
        misses = expected_src[begin:end][expected_slots[begin:end] < 0]
        unions.append(torch.unique(misses, sorted=True))
    return expected_src, expected_slots, unions


def check_case(case: dict[str, torch.Tensor], reference: torch.Tensor,
               label: str, require_slot_zero: bool = False,
               require_evict_slots: bool = False) -> list[int]:
    expected_src, expected_slots, expected_union = expected_outputs(case, reference)
    old_cache = case["cache_slots"].clone()
    call_mtp(case)
    torch.npu.synchronize()
    actual_src = case["topk_src"].reshape(case["batch"] * MTP_WIDTH, TOPK)
    actual_slots = case["topk_dst"].reshape(case["batch"] * MTP_WIDTH, TOPK)
    src_mismatch = int((expected_src != actual_src).sum().item())
    slot_mismatch = int((expected_slots != actual_slots).sum().item())
    if src_mismatch or slot_mismatch:
        raise AssertionError(
            f"{label}: TopK/reorder mismatch: src={src_mismatch}, slots={slot_mismatch}")
    if require_slot_zero and int((actual_slots == 0).sum().item()) == 0:
        raise AssertionError(f"{label}: legal HBM slot 0 was not exercised")
    expected_counts = [int(ids.numel()) for ids in expected_union]
    actual_counts = case["miss_counts"].cpu().tolist()
    if actual_counts != expected_counts:
        raise AssertionError(
            f"{label}: union counts differ: actual={actual_counts}, expected={expected_counts}")
    for request, ids in enumerate(expected_union):
        actual = case["miss_src"][request, :ids.numel()]
        if not torch.equal(actual, ids):
            raise AssertionError(
                f"{label}: union IDs differ for request={request}, "
                f"mismatches={int((actual != ids).sum().item())}")
        if require_evict_slots and ids.numel():
            dst = case["miss_dst"][request, :ids.numel()]
            budget = int(case["cache_tokens"][request].item())
            if bool(((dst < 0) | (dst >= budget)).any().item()):
                raise AssertionError(
                    f"{label}: invalid evict destination slot: "
                    f"dst={dst.cpu().tolist()}, budget={budget}, union={ids.cpu().tolist()}")
            if int(torch.unique(dst).numel()) != int(dst.numel()):
                raise AssertionError(f"{label}: duplicate evict destination slots")
            begin = request * MTP_WIDTH
            end = begin + MTP_WIDTH
            protected = torch.unique(expected_slots[begin:end][expected_slots[begin:end] >= 0])
            if bool(torch.isin(dst, protected).any().item()):
                raise AssertionError(f"{label}: evicted a slot belonging to the four-route TopK union")
    if not torch.equal(case["cache_slots"], old_cache):
        raise AssertionError(f"{label}: cache_slots_pool was modified")
    return expected_counts


def boundary_args(base: argparse.Namespace, **overrides) -> argparse.Namespace:
    values = vars(base).copy()
    values.update(batch_size=1, warmup=0, iters=0, graph_replays=0,
                  perf_query_noise=0.25, perf_query_miss_count=0)
    values.update(overrides)
    return argparse.Namespace(**values)


def run_boundary_tests(args: argparse.Namespace) -> None:
    """Correctness-only coverage for semantics implemented through miss union."""
    cases = (
        ("minimum_len_all_hit_bf16", dict(seq_len=TOPK, cache_tokens=TOPK,
                                           dtype="bf16"), "all_hit"),
        ("fp16_one_miss", dict(seq_len=8192, cache_tokens=8064, dtype="fp16",
                               perf_query_miss_count=1), "balanced"),
        ("partial_budget", dict(seq_len=8192, cache_tokens=6144, dtype="bf16",
                                perf_query_miss_count=64), "balanced"),
        ("all_miss", dict(seq_len=8192, cache_tokens=8192, dtype="bf16",
                          perf_query_noise=2.0), "all_miss"),
        ("identical_routes", dict(seq_len=8192, cache_tokens=8192, dtype="bf16",
                                  perf_query_noise=1e-6), "all_miss"),
        ("union_capacity", dict(seq_len=8192, cache_tokens=8192, dtype="bf16",
                                perf_query_noise=4.0), "all_miss"),
    )
    summaries = []
    for index, (label, overrides, cache_mode) in enumerate(cases):
        print(f"FUSED_LI_MANAGE_MTP_BOUNDARY_BEGIN case={label}", flush=True)
        edge_args = boundary_args(args, seed=args.seed + 100 + index, **overrides)
        case = make_case(edge_args)
        if label == "identical_routes":
            case["query"].copy_(case["query"][0:1].expand_as(case["query"]))
        if label == "union_capacity":
            # Give each route an exclusive 2048-token score band. The four
            # TopKs are disjoint, so the all-miss union reaches capacity 8192.
            case["query"].zero_()
            case["weights"].zero_()
            case["weights"][:, 0] = 1
            case["key"].zero_()
            flat_key = case["key"].reshape(edge_args.seq_len, 1, HEAD_DIM)
            values = torch.linspace(1.0, 2.0, TOPK, dtype=torch.float32,
                                    device=case["query"].device).to(case["key"].dtype)
            for route in range(MTP_WIDTH):
                case["query"][route, 0, route] = 1
                begin = route * TOPK
                flat_key[begin:begin + TOPK, 0, route] = values
            # Also exercise a legal non-sequential physical block mapping.
            case["block_table"].copy_(case["block_table"][:, torch.randperm(
                case["blocks_per_request"], device=case["query"].device)])
        reference = call_standard(case).reshape(MTP_WIDTH, TOPK)
        if cache_mode == "balanced":
            build_balanced_cache(case, reference, edge_args.perf_query_miss_count)
        elif cache_mode == "all_hit":
            case["cache_slots"][0, :edge_args.seq_len] = torch.arange(
                edge_args.seq_len, dtype=torch.int32, device=case["query"].device)
        # all_miss deliberately retains the initial -1 persistent map.
        counts = check_case(case, reference, label,
                            require_slot_zero=cache_mode != "all_miss",
                            require_evict_slots=cache_mode == "balanced")
        if label == "identical_routes" and counts != [TOPK]:
            raise AssertionError(f"{label}: expected union={TOPK}, actual={counts}")
        if label == "union_capacity" and counts != [UNION_CAPACITY]:
            raise AssertionError(
                f"{label}: expected union={UNION_CAPACITY}, actual={counts}")
        summaries.append(f"{label}:{counts[0]}")

    # Request-pool indirection: move the logical row away from row 0.
    print("FUSED_LI_MANAGE_MTP_BOUNDARY_BEGIN case=noncontiguous_request_pool",
          flush=True)
    edge_args = boundary_args(args, seed=args.seed + 200, seq_len=8192,
                              cache_tokens=8064, dtype="bf16",
                              perf_query_miss_count=1)
    case = make_case(edge_args)
    reference = call_standard(case).reshape(MTP_WIDTH, TOPK)
    build_balanced_cache(case, reference, 1)
    expanded = torch.full((3, edge_args.seq_len), -1, dtype=torch.int32,
                          device=case["query"].device)
    expanded[2].copy_(case["cache_slots"][0])
    case["cache_slots"] = expanded
    case["req_entries"].fill_(2)
    counts = check_case(case, reference, "noncontiguous_request_pool", True, True)
    summaries.append(f"noncontiguous_request_pool:{counts[0]}")
    print("FUSED_LI_MANAGE_MTP_BOUNDARY_CHECK " + " ".join(summaries) + " ok=1",
          flush=True)


def main() -> None:
    args = parse_args()
    require_local_opapi()
    if not callable(getattr(torch_npu, "npu_lightning_indexer", None)):
        raise RuntimeError("torch_npu.npu_lightning_indexer is unavailable")
    if not args.skip_boundary_tests:
        run_boundary_tests(args)
    case = make_case(args)
    reference = call_standard(case).reshape(args.batch_size * MTP_WIDTH, TOPK)

    union_sizes, expected_union_miss_counts = build_balanced_cache(
        case, reference, args.perf_query_miss_count)
    union_mean = statistics.mean(union_sizes)
    union_target_ok = 3000.0 <= union_mean <= 4000.0
    print(
        "FUSED_LI_MANAGE_MTP_WORKLOAD_CHECK "
        f"batch={args.batch_size} candidate_len={args.seq_len} "
        f"query_noise={args.perf_query_noise:.4f} "
        f"topk_union_min={min(union_sizes)} topk_union_mean={union_mean:.2f} "
        f"topk_union_max={max(union_sizes)} target_range=[3000,4000] "
        f"target_ok={int(union_target_ok)} per_query_misses={args.perf_query_miss_count} "
        f"unique_union_misses_min={min(expected_union_miss_counts)} "
        f"unique_union_misses_mean={statistics.mean(expected_union_miss_counts):.2f} "
        f"unique_union_misses_max={max(expected_union_miss_counts)}",
        flush=True,
    )
    if not union_target_ok:
        raise AssertionError("TopK union is outside [3000,4000]; adjust --perf-query-noise")
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
    if expected_counts != expected_union_miss_counts:
        raise AssertionError(
            f"constructed union misses={expected_union_miss_counts}, reference={expected_counts}"
        )
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
        evict_slots = case["miss_dst"][batch_idx, :count]
        budget = int(case["cache_tokens"][batch_idx].item())
        if bool(((evict_slots < 0) | (evict_slots >= budget)).any().item()):
            raise AssertionError(f"MTP eviction produced an invalid slot for batch {batch_idx}")
        if int(torch.unique(evict_slots).numel()) != count:
            raise AssertionError(f"MTP eviction produced duplicate slots for batch {batch_idx}")
        route_begin = batch_idx * MTP_WIDTH
        route_end = route_begin + MTP_WIDTH
        protected_slots = torch.unique(reference_slots[route_begin:route_end][
            reference_slots[route_begin:route_end] >= 0])
        if bool(torch.isin(evict_slots, protected_slots).any().item()):
            raise AssertionError(f"MTP eviction selected a TopK-union slot for batch {batch_idx}")
    if not torch.equal(case["cache_slots"], old_cache):
        raise AssertionError("phase-1 MTP TopK modified cache_slots_pool")
    run_graph_check(case, args.graph_replays)
    if not torch.equal(case["cache_slots"], old_cache):
        raise AssertionError("graph replay modified cache_slots_pool")
    standard_mean, standard_p50 = benchmark(lambda: call_standard(case), args.warmup, args.iters)
    mtp_mean, mtp_p50 = benchmark(lambda: call_mtp(case), args.warmup, args.iters)
    ratio = mtp_mean / standard_mean
    print(
        "FUSED_LI_MANAGE_MTP_RESULT "
        f"batch={args.batch_size} routes=4 seq_len={args.seq_len} dtype={args.dtype} "
        f"standard_mean_us={standard_mean:.3f} standard_p50_us={standard_p50:.3f} "
        f"mtp_mean_us={mtp_mean:.3f} mtp_p50_us={mtp_p50:.3f} ratio={ratio:.4f} "
        f"warmup={args.warmup} iters={args.iters} topk_hit_miss_reorder_match=1 "
        f"union_miss_match=1 evict_slots_valid=1 cache_unchanged=1",
        flush=True,
    )
    if ratio > args.max_latency_ratio:
        raise AssertionError(
            f"MTP latency ratio {ratio:.4f} exceeds limit {args.max_latency_ratio:.4f}"
        )
    print("FUSED_LI_MANAGE_MTP_UT_OK", flush=True)


if __name__ == "__main__":
    main()
