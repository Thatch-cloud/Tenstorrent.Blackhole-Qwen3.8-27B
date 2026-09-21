"""Side-by-side device timing: a prefill-style masked SDPA candidate versus the decode-SDPA
fake-head configuration the packed four-user verify actually dispatches today.

BACKGROUND. The four-user 64-row packed verify (Qwen3.5-27B, TP2 on two Blackhole p150a,
32k context, paged bf8 KV) spends 262 ms of its ~1118 ms round in attention: per
full-attention layer (16 of them), 8 calls of
``ttnn.transformer.paged_scaled_dot_product_attention_decode`` at ~2.05 ms each - 4 users x
2 bundles, each user's 16 draft rows folded as fake heads by ``attention_head_fold.py`` (groups
of 4 rows x 12 heads = 48 heads per lane, up to 3 lanes per call - see ``attention_parallel.py``
and ``pooled_attention_replay.py``/``attention_replay.py``, the latter two HASH-PINNED and never
edited by this script). Every call walks the full 32k prefix with a full-width provided mask.
That decode-SDPA kernel is a one-query-row-per-batch flash-decode loop, so cost scales with
rows x KV length; ``docs/experiment-execution.md``'s "Parallel width microbenchmark 34073855612"
already showed merging two bundles into one wider call saves only ~10% at T16/32k
(0.572341 -> 0.517279 ms per matched attention call).

THE QUESTION THIS MEASURES. The draft side already runs a PREFILL-style masked SDPA for its own
16-row attention (``dflash_t16_native_attention.py``, POLICY 'dflash-t16-native-proposal-only-
unqualified'): ``ttnn.transformer.scaled_dot_product_attention`` with an explicit ``attn_mask``,
``is_causal=False``, non-paged interleaved-DRAM Q/K/V, bounded to a 2048-token sliding window
and 1-2 users (``packed_key_limit``). Could the TARGET verify use an analogous masked, non-causal
SDPA call - one per layer, batched over B=4 users - instead of the decode-fake-head calls?

THIS IS NOT A DROP-IN COMPARISON. The one native tt-metal op that genuinely is "prefill-style
AND paged" (``ttnn.transformer.chunked_scaled_dot_product_attention``, used by the grafted
``attention/tp.py:forward_prefill_paged``) takes NO ``attn_mask`` argument at all - it is
unconditionally causal over whatever is resident in the paged cache, and is called once per user
(``paged_fill_cache(..., batch_idx=user_id)``, one page table row per call): no batch axis, no
custom mask, and it requires the query's own rows to already be committed to the paged KV cache.
None of that fits four-user packed verify, where draft rows are speculative (not yet accepted,
so must NOT be written into the shared paged cache before commit) and need a provided mask over
prefix-visible + some-subset-of-draft-rows-visible structure, not plain causality. So the only
pinned op that can even take the required inputs (an explicit mask, a non-causal structure) is
the SAME general op the draft side uses, ``ttnn.transformer.scaled_dot_product_attention`` - which
is NOT paged: it wants contiguous per-user K/V. This script's "prefill-style" arm therefore times
that op against a directly materialized (gathered) K/V buffer, which is optimistic - it excludes
the paged-gather cost a real integration would need to pay every step, since draft rows cannot
be pre-committed to the paged cache the way a real prefill chunk is. That gap is deliberate and
reported alongside the numbers, not hidden in them.

MODEL SHAPES. The grafted ``model_config.py``/``attention/tp.py`` (Qwen3.5-27B, TP2) give
n_local_heads=12, head_dim=256, and n_kv_heads=4 GLOBALLY (attention/tp.py:34, "27B's 4 KV heads
on TP=8") -> n_local_kv_heads = max(1, 4 // 2) = **2** at TP2, not 4: GQA is 12:2 = 6:1, not 3:1.
This is independently confirmed by ``attention_head_fold.fold_query``, which hardcodes
``kv_heads=2`` and folds heads as ``reshape(rows, 2, 6, 256)`` (2 KV-head groups of 6 Q heads
each). The default ``--kv-heads`` below is 2 for this reason; the task brief's "4 KV heads x 256
dim" is kept available via ``--kv-heads 4`` for direct comparison but is not the target's real
number.

Runs entirely inside the serving image on ONE device (paged_scaled_dot_product_attention_decode
and scaled_dot_product_attention are both single-device ops; no CCL). Every ttnn/torch import is
deferred to the device-dependent functions, so this module and its pure helpers import and
unit-test cleanly under plain CPython with no ttnn installed - see
test_sdpa_verify_prefill_style_bench.py.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

TILE = 32

# Target model, TP2 (see module docstring): fixed by the pinned fold_query/replay geometry,
# not swept.
NH_LOCAL = 12
HD = 256
ROWS_PER_FOLD_GROUP = 4
FOLDED_HEADS = ROWS_PER_FOLD_GROUP * NH_LOCAL  # 48

DEFAULT_KV_HEADS = 2  # see module docstring: confirmed local KV heads at TP2, not the 4 assumed
DEFAULT_CONTEXT = 32768
DEFAULT_PAGE_SIZE = 64
DEFAULT_DRAFT_ROWS = 16
DEFAULT_PADDED_ROWS = 32


def capacity_for(context, short_pad=256):
    """The native chunk family's capacity for a `context`-token prefix: `context + short_pad`,
    rounded up to a page-size-friendly multiple of 256 (attention_mask_replay.py's own
    `capacity - 256 == first position` convention). context=32768 -> 33024."""
    if type(context) is not int or context <= 0 or type(short_pad) is not int or short_pad <= 0:
        raise ValueError("Positive integer context and padding required")
    return ((context + short_pad + 255) // 256) * 256


def pages_for(capacity, page_size):
    if type(capacity) is not int or type(page_size) is not int or capacity <= 0 or page_size <= 0:
        raise ValueError("Positive integer capacity and page size required")
    if capacity % page_size:
        raise ValueError("Capacity must be a whole number of pages")
    return capacity // page_size


def decode_fake_head_bundles(rows=DEFAULT_DRAFT_ROWS, capacity=None, page_size=DEFAULT_PAGE_SIZE,
                             max_group_rows=ROWS_PER_FOLD_GROUP, max_batches=3):
    """The exact bundle plan attention_head_fold.parallel_groups gives one user's `rows`-row
    verify block at this capacity - reproduced by calling the pinned host function directly
    (not re-derived), so this stays correct if the replay geometry ever changes. At the default
    32768-token context this returns ((3, 48), (1, 48)): one call at batch 3, one at batch 1,
    each over 48 folded heads - matching the task brief's "B=3 lanes x 48 fake heads + B=1 x 48"
    exactly."""
    from attention_head_fold import parallel_groups

    if capacity is None:
        capacity = capacity_for(DEFAULT_CONTEXT)
    start = capacity - 256
    bundles = parallel_groups(start, rows, max_batches=max_batches, max_group_rows=max_group_rows)
    plan = []
    for bundle in bundles:
        group_rows = bundle[0]["rows"]
        if any(group["rows"] != group_rows for group in bundle):
            raise ValueError("Mixed-width bundle from parallel_groups; replay geometry changed")
        plan.append((len(bundle), group_rows * NH_LOCAL))
    return tuple(plan)


def build_tree_mask(users, capacity, draft_rows=DEFAULT_DRAFT_ROWS, padded_rows=DEFAULT_PADDED_ROWS):
    """Host BF16 provided mask for the prefill-style candidate: (users, 1, padded_rows,
    capacity). The first `draft_rows` (query) rows see the whole `capacity - padded_rows`
    prefix plus a causal triangle over the draft block itself (row i of the draft additionally
    sees draft rows 0..i - a linear speculative chain, the simplest tree); the
    padded_rows - draft_rows filler rows each see exactly one key, matching the padded-row rule
    ``dflash_t16_native_attention.validate_mask`` enforces for the draft's own T16 mask so the
    two mask conventions stay comparable. This mask's VALUES are synthetic (this is a timing
    benchmark, not a correctness one) but its zero/-inf STRUCTURE and shape match what a real
    tree-masked verify call would need to supply."""
    import torch

    if not 1 <= users <= 64:
        raise ValueError("Bounded user count required")
    if not 0 < draft_rows <= padded_rows or capacity < padded_rows:
        raise ValueError("Bounded draft/padded row and capacity geometry required")
    prefix = capacity - padded_rows
    mask = torch.full((users, 1, padded_rows, capacity), float("-inf"), dtype=torch.bfloat16)
    mask[:, :, :draft_rows, :prefix] = 0.0
    draft_rows_idx = torch.arange(draft_rows)
    causal = (draft_rows_idx[:, None] >= draft_rows_idx[None, :])
    mask[:, :, :draft_rows, prefix:prefix + draft_rows] = torch.where(
        causal, torch.zeros(()), torch.full((), float("-inf"))).to(torch.bfloat16)
    mask[:, :, draft_rows:, prefix] = 0.0
    return mask


def format_table(rows):
    header = "%-28s %8s %10s %10s %10s %8s" % ("config", "users", "calls", "mean_us", "min_us", "errors")
    lines = [header]
    for row in rows:
        lines.append("%-28s %8d %10d %10s %10s %8d" % (
            row["label"], row["users"], row["calls"],
            "ERR" if row.get("error") else "%.1f" % row["mean_us"],
            "ERR" if row.get("error") else "%.1f" % row["min_us"],
            1 if row.get("error") else 0))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------
# Device-dependent parts. ttnn/torch imports are local to these functions so the module
# (and every pure function above) stays importable and unit-testable with no ttnn installed.
# ---------------------------------------------------------------------------------

def time_op(ttnn, device, run_once, warmup=3, iters=20):
    """Warm up `warmup` iterations, then time `iters` individually (perf_counter +
    ttnn.synchronize_device around each call), reporting mean/min microseconds. Every output
    is deallocated so repeated calls do not exhaust L1/DRAM across the sweep. A raised
    exception is caught by the caller (safe_time), not here."""
    for _ in range(warmup):
        out = run_once()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        out = run_once()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - started) * 1e6)
        ttnn.deallocate(out)
    return dict(mean_us=statistics.mean(samples), min_us=min(samples), samples_us=samples)


def safe_time(ttnn, device, run_once, warmup, iters):
    try:
        return time_op(ttnn, device, run_once, warmup=warmup, iters=iters)
    except Exception as error:  # noqa: BLE001 - a bad config must not kill the sweep
        return dict(error="%s: %s" % (type(error).__name__, error))


def make_contiguous_kv(ttnn, torch, device, users, kv_heads, capacity, seed=0):
    """Non-paged, interleaved-DRAM bf8 K/V for `users` users - the prefill-style candidate's
    input shape, matching draft_sdpa's own operand placement (draft_attention.py) except at the
    target's head_dim/kv_heads and full 32k length instead of the draft's <=2048-token window.
    This is the optimistic part of the comparison: materializing this tensor from a paged cache
    every verify step is exactly the gather cost a real integration would need to pay and this
    benchmark does not charge for it (see module docstring)."""
    torch.manual_seed(seed)
    host = torch.randn(users, kv_heads, capacity, HD, dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(host, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_query(ttnn, torch, device, users, heads, rows, seed=1):
    torch.manual_seed(seed)
    host = torch.randn(users, heads, rows, HD, dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_folded_query(ttnn, torch, device, batch, heads, seed=1):
    """(1, batch, heads, HD) bf16 - the post-fold shape attention_parallel.execute's `stacked`
    tensor has when it reaches paged_scaled_dot_product_attention_decode (concat of `batch`
    per-group folded queries on dim=1). Built directly in this shape; the on-device DMA fold
    itself (attention_fold_dma.device_layout_dma) is a separate preprocessing cost this
    benchmark does not include, since the task is timing the SDPA call, not the fold."""
    torch.manual_seed(seed)
    host = torch.randn(1, batch, heads, HD, dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def upload_mask(ttnn, torch_mask, device):
    return ttnn.from_torch(torch_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def prefill_style_arm(ttnn, torch, device, users, kv_heads, capacity, warmup, iters,
                      k_chunk_size):
    """One masked, non-causal ttnn.transformer.scaled_dot_product_attention call over `users`
    users at once: Q (users, 12, 32, 256) bf16 (16 valid rows padded to 32), K/V (users,
    kv_heads, capacity, 256) bf8, mask (users, 1, 32, capacity) bf16. This is the layer-equivalent
    candidate cost: ONE call serves every user, versus 2 x users decode-fake-head calls today."""
    query = make_query(ttnn, torch, device, users, NH_LOCAL, DEFAULT_PADDED_ROWS)
    key = make_contiguous_kv(ttnn, torch, device, users, kv_heads, capacity, seed=10)
    value = make_contiguous_kv(ttnn, torch, device, users, kv_heads, capacity, seed=11)
    mask_host = build_tree_mask(users, capacity)
    mask = upload_mask(ttnn, mask_host, device)
    kernel = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    q_chunk = DEFAULT_PADDED_ROWS
    program = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk, k_chunk_size=k_chunk_size, exp_approx_mode=False)

    def once():
        return ttnn.transformer.scaled_dot_product_attention(query, key, value, attn_mask=mask,
            is_causal=False, scale=HD ** -0.5, program_config=program, compute_kernel_config=kernel,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)

    timing = safe_time(ttnn, device, once, warmup, iters)
    for tensor in (query, key, value, mask):
        ttnn.deallocate(tensor)
    calls = 1
    return dict(label="prefill_style_masked_sdpa", users=users, calls=calls, kv_heads=kv_heads,
               capacity=capacity, **timing)


def make_paged_kv_pool(ttnn, torch, device, kv_heads, page_size, num_blocks, seed=20):
    torch.manual_seed(seed)
    k_host = torch.randn(num_blocks, kv_heads, page_size, HD, dtype=torch.bfloat16) * 0.02
    v_host = torch.randn(num_blocks, kv_heads, page_size, HD, dtype=torch.bfloat16) * 0.02
    k = ttnn.from_torch(k_host, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.from_torch(v_host, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return k, v


def make_page_table(ttnn, torch, device, batch, pages, num_blocks, seed=30):
    torch.manual_seed(seed)
    host = torch.randint(0, num_blocks, (batch, pages), dtype=torch.int32)
    return ttnn.from_torch(host, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def decode_fake_head_arm(ttnn, torch, device, users, kv_heads, capacity, page_size, warmup, iters):
    """The decode-SDPA fake-head configuration exactly as pooled_attention_replay.py /
    attention_parallel.py dispatch it today: per user, the bundle plan from
    decode_fake_head_bundles() (default (3, 48) then (1, 48)), each a
    ttnn.transformer.paged_scaled_dot_product_attention_decode call with a full-width provided
    mask, is_causal=False, no cur_pos_tensor (the mask supplies all masking - see
    attention_parallel.execute). `users` separate users' worth of calls are timed back to back
    so the total matches the real per-layer call count (8 calls at users=4)."""
    pages = pages_for(capacity, page_size)
    num_blocks = pages  # synthetic pool; content is unused for timing, only shape/dtype matter
    keys, values = make_paged_kv_pool(ttnn, torch, device, kv_heads, page_size, num_blocks)
    grid = device.compute_with_storage_grid_size()
    bundles = decode_fake_head_bundles(capacity=capacity)
    per_call = []
    owned = [keys, values]
    try:
        for user in range(users):
            for batch, heads in bundles:
                query = make_folded_query(ttnn, torch, device, batch, heads, seed=100 + user)
                owned.append(query)
                page_table = make_page_table(ttnn, torch, device, batch, pages, num_blocks,
                                             seed=200 + user)
                owned.append(page_table)
                mask_host = torch.zeros(batch, 1, heads, capacity, dtype=torch.bfloat16)
                mask = upload_mask(ttnn, mask_host, device)
                owned.append(mask)
                program = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                    exp_approx_mode=False, q_chunk_size=0, k_chunk_size=256)

                def once(q=query, pt=page_table, m=mask, pc=program):
                    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        q, keys, values, page_table_tensor=pt, is_causal=False, attn_mask=m,
                        scale=HD ** -0.5, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                timing = safe_time(ttnn, device, once, warmup, iters)
                per_call.append(dict(batch=batch, heads=heads, **timing))
    finally:
        for tensor in owned:
            ttnn.deallocate(tensor)
    calls = len(per_call)
    errors = sum(1 for entry in per_call if "error" in entry)
    if errors:
        return dict(label="decode_fake_head", users=users, calls=calls, kv_heads=kv_heads,
                   capacity=capacity, error="%d of %d calls failed" % (errors, calls),
                   per_call=per_call)
    total_mean = sum(entry["mean_us"] for entry in per_call)
    total_min = sum(entry["min_us"] for entry in per_call)
    return dict(label="decode_fake_head", users=users, calls=calls, kv_heads=kv_heads,
               capacity=capacity, mean_us=total_mean, min_us=total_min, per_call=per_call)


def run(args):
    import torch
    import ttnn

    capacity = capacity_for(args.context)
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    rows = []
    try:
        for users in args.users:
            rows.append(prefill_style_arm(ttnn, torch, device, users, args.kv_heads, capacity,
                                          args.warmup, args.iters, args.k_chunk_size))
            rows.append(decode_fake_head_arm(ttnn, torch, device, users, args.kv_heads, capacity,
                                             args.page_size, args.warmup, args.iters))
    finally:
        ttnn.close_device(device)
    passed = not any(row.get("error") for row in rows)
    return dict(passed=passed, context=args.context, capacity=capacity, kv_heads=args.kv_heads,
               page_size=args.page_size, users=list(args.users), iters=args.iters,
               warmup=args.warmup, k_chunk_size=args.k_chunk_size, rows=rows,
               scope="Single-device synthetic SDPA timing; not a correctness, coding-quality or "
                     "committed-throughput result. The prefill-style arm excludes the paged-gather "
                     "cost a real integration would pay (see module docstring).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS,
                       help="local KV heads; default 2 matches the confirmed TP2 target config "
                            "(attention_head_fold.fold_query hardcodes kv_heads=2), not the 4 the "
                            "task brief assumed - pass 4 to reproduce that assumption directly")
    parser.add_argument("--k-chunk-size", type=int, default=256)
    parser.add_argument("--users", default="1,4", help="comma-separated user counts to time")
    args = parser.parse_args()
    args.users = [int(value) for value in args.users.split(",")]
    if any(value < 1 for value in args.users):
        parser.error("--users must be positive integers")

    report = dict(passed=False)
    try:
        report = run(args)
    except Exception as error:  # noqa: BLE001
        report["error"] = "%s: %s" % (type(error).__name__, error)
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))

    print(format_table(report.get("rows", [])))
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
