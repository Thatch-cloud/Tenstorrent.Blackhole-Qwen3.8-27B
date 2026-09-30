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

### CB arithmetic detail (added on request)

**Caveat up front**: `sdpa_program_factory.cpp` is not vendored in this worktree and this
investigation has no rig/hardware access, so nothing below is a verified line-cited read
of the live source - it is reconstructed from the pinned patch's own variable names
(`dflash_combined_sim_runtime.py:17-22`), the reader's fixed call parameters, and the two
known failures, then bounded against a third point (today's working config) rather than
asserted as exact.

`Skt` (the factory's own variable name for total K length in tiles) is fully recoverable:
testing `Skt = capacity//32 + 8` against every value in the pinned `COMBINED_FACTORY`
patch's Skt list (`dflash_combined_sim_runtime.py:19`, values 144/272/528/1040/2064/4112/8208)
reproduces all seven exactly for context rungs 4096 through 262144 (`capacity = context +
256`, verified by running the formula). So the 65536-context failure (`capacity=65792`,
run 35585801822) sits at **Skt=2064** confirmed, and the 32768-family 8-row failure sits
at Skt=1040 (context 32768, same family the packed T16 replay uses).

Two failures, one working baseline, three unknowns (a Skt-scaled term, a q-row-scaled
term, and a fixed floor) - not fully solvable, but boundable:
- **65536 case**: Skt 1040->2064 (+1024 tiles), q rows unchanged (still the production
  4-row group), total 1,572,864(fits, today's 32768 config) -> 1,600,448 (fails by 27,584 B).
  Lower bound: the Skt-scaled term costs >=27,584 B per +1024 tiles, i.e. >=27 B/tile - a
  lower bound because the 32768 baseline is not known to sit exactly at the 1,572,864
  ceiling, only under it.
- **8-row case**: Skt fixed at ~1040 (still 32768), rows 4->8, total 1,572,864(fits) ->
  1,829,824 (fails by 256,960 B). Lower bound: the q-row-scaled term costs >=256,960 B for
  +4 rows, i.e. >=64,240 B/row - again a floor, not the exact per-row cost.

Row width is the dominant lever by a wide margin (>=64,240 B/row vs >=27 B/Skt-tile), which
is consistent with per-row-scaled CBs (QK score tile, output accumulator, softmax stats -
all sized by `Sq_chunk_t`) being far larger per unit than whatever scales with `Skt`. That
a **4-row, unwidened** launch still overflows purely from context growth (the 65536 case)
points at a component that is NOT chunk-bounded by `k_chunk_size` (fixed at 256/8 tiles
regardless of capacity, `attention_replay.py:53-54`) but scales with the *full* `Skt`
instead - the likeliest candidate is the attention mask: `attn_mask` is shaped `(batches, 1, rows*12, capacity)`
(`attention_replay.py:50`) - **full-capacity width, not chunk width** - because
`is_causal=False` with a dense provided mask is used (`attention_parallel.py:18`) instead
of the op's chunked-causal path. The pinned patch's own condition list names a
`!is_chunked` / `compute_use_provided_mask` combination (`dflash_combined_sim_runtime.py:19-20`)
as a real mode switch in the factory, consistent with a full-width mask-staging buffer for
this call pattern - inference from indirect evidence, not a confirmed CB name/line.

**Lowest-numerical-impact lever for 65536 specifically** (context growth alone, group
width already at production minimum): reduce `k_chunk_size` below 256. Chunk size is
normally a pure re-tiling knob in online-softmax kernels - the running max/sum combine is
chunk-size-invariant - so this should carry no extra numerical risk. It only helps if a
chunk-scaled CB is part of the 27,584 B overflow; if the dominant term is the full-`Skt`
mask buffer above, `QWEN_SDPA_TREE_SCRATCH_ROUNDS` (`attention_replay.py:23`, "process-fixed
compact native scratch") is the more direct candidate - its name ties it to trading L1
scratch for more DMA rounds, the same class of fix a full-width buffer needs. Grid size is
**not** expected to help: static per-core CB size is a function of chunk/row parameters,
not core count. None of this is verified against the real factory source.

**Verdict: infeasible at the kernel's current core/CB allocation strategy on either axis.**
This is not a data-plumbing change (stack 4 users' tensors and call once) - it needs a new
core-to-work mapping in the C++ program factory, i.e. the authorized-kernel-change route
`docs/trace-datamov-plan.md:53-58` already names: *"Collapsing this to one 64-row SDPA
call needs a native batch-64 paged-SDPA-decode kernel (C++, authorised)."*

## 3. Where the 2.05 ms/launch goes

**Corrected 2026-09-22** (`docs/sdpa-kv-bf8-plan.md`). This section originally read
the KV dtype as `ttnn.bfloat16` from `frozen_runtime_context.py:45-46` and friends.
That citation belongs to a simulator-gated harness, not to the serving decode launch:
the launch this section is about reads **bfloat8_b** K/V zero-copy from the vLLM
serving cache (`attention_replay.py:118/129`, `attention_parallel.py:17-19`,
`serving_cache_owner.py:26-31`, `packed_cache_writer.py:3`). So there is no bf16
KV copy on this path and no bf16-to-bf8 saving left to claim.

Full-capacity K+V read for one launch is therefore 32768 keys x 256 head_dim x
2 (K,V) x 2 local KV heads x 1 byte = 33,554,432 B (32 MiB), half the figure this
section first gave. At 2.05 ms that is ~16-20 GB/s once the dense bf16 mask
(3-9 MiB, still bf16 and still the one remaining bytes lever here) is counted with
it - lower than the ~32 GB/s originally inferred, and low enough that K/V streaming
alone no longer obviously accounts for the launch duration. The qualitative
conclusion below still holds (per-user K/V is distinct DRAM content, so batching
users does not amortise the read), but the bandwidth headroom argument should be
re-derived before anything is built on it.

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
