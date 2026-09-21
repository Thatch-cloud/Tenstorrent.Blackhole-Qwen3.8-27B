# Batching four users' paged SDPA decode into one launch: feasibility

Scope: can the 64 `paged_scaled_dot_product_attention_decode` launches/round (4 users x
16 full-attention layers, `scripts/ci/pooled_attention_replay.py:29-33`) become 16 (one
per layer, all 4 users batched)? Read-only investigation, no hardware.

## 1. What the reader passes today

`PackedReplayAttentionReader.__call__` (`scripts/ci/pooled_attention_replay.py:320-345`)
loops the pack's 4 segments (= users), slices that user's rows out of the 64-row query,
and hands each slice to that user's own `PooledReplayAttentionReader` (wraps the pinned
`ReplayAttentionReader`), concatenating the 4 outputs back in row order. Each user is a
fully separate device object: its own positions word, page table, mask, `SDPAProgramConfig`.

Per user per layer, `ReplayAttentionReader.__call__` (`attention_replay.py:118-140`) calls
`attention_parallel.execute` (`attention_parallel.py:6-30`), which issues the SDPA op
**once per "bundle"**, not once per user. For T16 at the qualified capacity 32768
(`docs/batch-spec-tasks-2026-09-19.md:513`), `parallel_groups(32512, 16, max_group_rows=4)`
(`attention_head_fold.py:52-64`) returns 2 bundles - `[3 groups of 4 rows, 1 group of 4
rows]` (verified by running the function locally) - so **every user already issues 2 SDPA
launches per layer**, not 1. 4 users x 16 layers x 2 bundles = 128 physical launches/round,
matching the CSV exactly (see below).

Per launch, `attention_parallel.execute` calls (`attention_parallel.py:17-20`):
```
operations.transformer.paged_scaled_dot_product_attention_decode(stacked, keys, values,
    page_table_tensor=pages, is_causal=False, attn_mask=mask, scale=scale,
    program_config=config, memory_config=memory_config)
```
- `stacked` query: `(1, len(bundle), rows*12, 256)` - dim 1 is the op's batch axis, currently
  1-3, **all groups belonging to the same user** (folded heads: 12 Q heads x 256 head_dim).
- `pages` (page_table_tensor): `(len(bundle), capacity//64)` int32 - the **same** user's
  table repeated `len(bundle)` times (`attention_replay.py:49`), not per-batch-distinct.
- `mask` (attn_mask): `(len(bundle), 1, rows*12, capacity)` bf16, dense (not a scalar
  `cur_pos_tensor` - **`cur_pos_tensor` is never passed to the op**; positions live only in
  the mask, refreshed by the custom `attention_mask_replay` kernel from one positions word).
- `program_config`: `SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
  exp_approx_mode=False, q_chunk_size=0, k_chunk_size=256)` (`attention_replay.py:53-54`) -
  **the full 11x10=110-core mesh grid**, same for every bundle of every user.

Confirmed in `C:/Users/liamb/.claude/jobs/8376c877/tmp/profile6g/cpp_device_perf_report.csv`:
256 `SdpaDecodeDeviceOperation` rows (110 cores, 2 devices x 128), avg kernel duration
2.053 ms/row; 128 dev-0 launches = 4 users x 16 layers x 2 bundles exactly.

## 2. Can one launch take all 4 users?

The op itself already has a batch axis and per-batch page-table/mask rows - nothing in
the *op's* Python signature rules out 4 different users' tables/masks stacked on that
axis. The blocker is **core budget**, not the op's API surface:

`parallel_groups`'s own comment caps bundles at 3 because that is what the 110-core grid
holds: *"At most three TP2 groups preserve sixteen workers per KV head on 110 cores"*
(`attention_head_fold.py:53`) -> 3 groups x 16 workers/KV-head x 2 local KV heads = 96 of
110 cores at 4-row group width. One 64-row, 4-user launch needs 16 batch slots (4 users x
4 groups-of-4) at that same ratio: 16 x 16 x 2 = 512 cores - 4.7x the entire grid.

The other lever - fewer, wider batch slots (e.g. one 16-row group per user, 4 slots total)
- fails on the *other* axis: widening a single user's own group from 4 to 8 rows (still
B=1, no multi-user batching at all) already overflows L1 on the same full grid, per the
run cited in the task brief (35583274784): circular buffers grow to 1,829,824 B against a
1,572,864 B L1 budget. Going to 16 rows/group (needed for a 4-slot, 4-user layout) would
be worse, and going to 64 rows/group (1 slot for all 4 users) worse still.

**Verdict: infeasible at the kernel's current core/CB allocation strategy on either axis.**
This is not a data-plumbing change (stack 4 users' tensors and call once) - it needs a new
core-to-work mapping in the C++ program factory, i.e. the authorized-kernel-change route
`docs/trace-datamov-plan.md:53-58` already names: *"Collapsing this to one 64-row SDPA
call needs a native batch-64 paged-SDPA-decode kernel (C++, authorised)."*

## 3. Where the 2.05 ms/launch goes

KV cache dtype at this frozen recipe's capacity family is `ttnn.bfloat16`, not bf8
(`scripts/ci/frozen_runtime_context.py:45-46`, `full_mtp_request.py:164`,
`full-prefix.py:552`, all `ttnn.bfloat16`; `frozen_target_replay.py:34,63` requires
`kv_dtype == 'bfloat16'`). Full-capacity K+V read for one launch: 32768 keys x 256
head_dim x 2 (K,V) x 2 local KV heads x 2 bytes = 67,108,864 B (64 MiB). At 2.05 ms
that is ~32 GB/s - plausible for one op's DMA, i.e. consistent with **K/V streaming**,
not fixed launch overhead, dominating the measured kernel duration (this column is
on-device time only, already excluding host dispatch).

But the team's own attribution puts SDPA at only ~40 ms of the 157.8 ms round
(`docs/trace-datamov-plan.md:47-58`), while the raw serial sum of the 128 dev-0 launches'
measured durations is ~262 ms (128 x ~2.05 ms). Most of each launch's device time must
therefore overlap with other concurrent device work in the round; only ~40 ms of it sits
on the critical path.

**Consequence for batching**: each user's K/V is distinct DRAM content - batching users
into one launch does not reduce bytes moved, only launch *count*. If the 2.05 ms/launch
is genuinely streaming-bound (the bandwidth estimate above says so), the realistic saving
from 4-user batching is **not** "128 launches x fixed overhead," it is bounded by whatever
fraction of the already-small ~40 ms critical-path slice is fixed per-launch cost rather
than DMA - almost certainly a small fraction, not the full 40 ms. The one place batching
*could* remove real redundant DMA is each user's own 2-bundle split re-reading the same
K/V twice per layer today - but merging that (same-user, not cross-user) hits the same
CB wall as section 2 (8-row group already overflows L1).

## 4. Alternatives

| Option | Core/CB budget | Touches pinned SDPA files? | Notes |
|---|---|---|---|
| One 64-row, 4-user launch | Infeasible both axes (see 2) | Yes, all of them | Ruled out without a new core-mapping strategy |
| Two 2-user launches | 2 x 96 = 192 of 110 cores at today's ratio - also infeasible | Yes | Needs the same CB-shrink that already failed at 8 rows |
| Disjoint core ranges, 4 concurrent launches (GDN pattern) | Fits: GDN already runs 4 users on 4 disjoint 24-core shares of the same 11x10 grid in one `MeshProgramDescriptor`/`generic_op` (`scripts/ci/gdn_user_batch.py:79-96`, docstring 1-19) | No new numerics, but SDPA's kernel is not project-owned the way GDN's fused kernel is - GDN reuses `gdn_multitoken.load_kernels` **verbatim** (`gdn_user_batch.py:19-20`); replicating that for SDPA means re-hosting the pinned reader/writer/compute kernels into a custom multi-`CoreRangeSet` wrapper, not just placement | `gdn_user_batch.py:44-58` itself documents the batched launch measuring **slower** than the per-user split below a 2-user threshold - no guarantee of a win even if ported |
| Overlap SDPA(user N) with MLP(user N-1) in-trace | Pure scheduling change, no kernel edit | No | Lowest risk of the four; ceiling depends on how much MLP-per-user time exists to hide behind - not measured in this investigation |

## 5. Risks and requalification

**fp32-intermediates B==1 gate**: the pinned `sdpa_program_factory.cpp` does carry a
literal `B == 1` condition (`scripts/ci/dflash_combined_sim_runtime.py:17-22`,
identical text at `dspark_fp32_intermediates.py:11-17`). But its co-conditions - `NQH ==
16 && NKH == 4 && DHt == 4` (16 query heads, 4 KV heads, 128 head_dim) - describe a
different geometry than this reader's folded call (12 heads folded into the row axis,
head_dim 256, 2 local KV heads). This looks like it targets a separate SDPA call site
(most likely the draft model's own per-token decode), not `PackedReplayAttentionReader`.
**Open item, not confirmed either way**: before assuming this blocks batching the verify
path, confirm which call sites actually trip `qwen_draft_fp32_intermediates`.

**Pinned surface**: `dflash_t16_native_attention_gate.py:18-25` hashes 14 files for this
op family - `sdpa.cpp/.hpp`, `sdpa_nanobind.cpp`, `sdpa_device_operation.cpp/.hpp/_types.hpp`,
`sdpa_program_factory.cpp`, `sdpa_interleaved_cb_ids.hpp`, `sdpa_subblock_utils.hpp`,
`compute_common.hpp`, `compute_streaming.hpp`, `compute/sdpa.cpp`,
`dataflow/reader_interleaved.cpp`, `dataflow/writer_interleaved.cpp`, plus the tt_llk
packer header. A core-mapping or CB change almost certainly touches the factory, the
device-operation validation, and both kernels - most or all of this set needs new hashes
and a full token-exactness requalification (the gate already refused a run once for a
mere `attention_replay.py` byte diff: run 35495227738, `pooled_attention_replay.py:14-16`).

**Hang precedent**: a batch-64 change on a related op hung the device outright before its
authorized fix - `attn_decode_prep` at batch 64 (run 35507675630, `probe_kernel_sources.py:9-11`),
now fixed (task #40, confirmed live by the `[PINDIAG]` log line quoted in
`docs/trace-datamov-plan.md`). A from-scratch batch-64 SDPA kernel carries the same class
of risk.

**Effort**: infeasible at current core/CB ratios on both axes (2), a 14-file pinned
surface plus full requalification (5), and an unresolved fp32-branch question that needs
answering before work starts - this is a multi-week authorized-kernel effort, comparable
to or larger than task #40, with a real chance of landing net-negative given the CB
overflow already observed at a far smaller widening (8 rows, one user, not four).

## Recommendation

Do not pursue the single 64-row launch. Spend the budget on alternative 4 (SDPA/MLP
overlap in-trace) first - no kernel change, no requalification, and it is the only option
whose ceiling isn't already known to be blocked. If overlap's ceiling proves too small,
disjoint core ranges is the next-cheapest *structurally*, but only after confirming a
ported SDPA kernel would actually run at a favorable users-per-launch threshold, per the
GDN precedent's own documented slowdown risk.
