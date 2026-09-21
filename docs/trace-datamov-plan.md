# Trace data-movement plan: the ~66 ms non-compute bucket in the 157.8 ms packed round

Scope: data movement (26 ms) + GDN other (25 ms) + fold DMA (15 ms) inside the four-user
64-row packed verify round (runs 35578180747 / 35584039408, `[PACKED-PHASE] trace_ms`
157.71-157.83 ms, `m3native-gate12b`/`m3native-gate-35584039408` JSON/log). This round has
`native_m3` AND `native_attn` engaged (`[PINDIAG] native_m3 binder calls this round:
{'decode norm': 129, 'full-attention forward': 16, 'MLP forward': 0, 'GDN output
projection': 0, 'sliced attn_decode_prep': 0, 'two-tile head concat': 0}` -
`m3native-gate-35584039408/m3native-gate-stdout.log:303`), so the two-tile MLP/GDN-output/
sliced-attn-prep wrappers this doc's predecessor analysis covered are **already retired**
in this trace. The remaining ~66 ms is a different layer of the stack.

## Evidence used

- Device kernel-duration CSV: `C:/Users/liamb/.claude/jobs/8376c877/tmp/profile6g/cpp_device_perf_report.csv`
  and the identical `m3native-gate12b/profile/.logs/cpp_device_perf_report.csv` (52,915/52,423
  device-0+1 rows). **Caveat, stated up front**: this CSV was captured in the profiler's eager
  (non-traced) dispatch mode, so its OP-TO-OP LATENCY column is inflated by Python dispatch and is
  NOT used here. Only DEVICE KERNEL DURATION is used - that number is hardware-accurate regardless
  of capture mode. Per-launch kernel costs measured from it: SliceDeviceOperation 7.3 us,
  CloneOperation 7.2 us, ConcatDeviceOperation 8.5 us, TilizeWithValPaddingDeviceOperation ~38 us,
  UntilizeWithUnpaddingDeviceOperation ~58 us, SdpaDecodeDeviceOperation ~2.05 ms - these per-launch
  averages are used against **current-source** launch-count formulas, not against this CSV's own
  raw totals (its GDN clone/slice counts predate task #46's batching and do not reflect the current
  floor - see "ruled out" below).
- Source of the current worktree (`scripts/ci/`), re-read for this task where the earlier
  inventory didn't cover it: `gdn_records.py`, `gdn_batched_conv.py`, `attention_fold_dma.py`,
  `attention_grouped.py`, `attention_parallel.py`.
- The team's own bucket totals (SDPA 40, GDN recurrence 23, matmul 34, data movement 26,
  GDN other 25, fold DMA 15, rest 13) are taken as ground truth; this doc apportions the 66 ms
  slice, it does not re-derive the total.

## Ranked plan

| # | Item | Call site | Est. ms | Mechanism | Removable | Risk | Kernel change |
|---|------|-----------|--------:|-----------|-----------|------|----------------|
| 1 | `commit_user` re-validates native buffer bindings on every one of a round's 4 packed-user commits | `gdn_records.py:181-190` (`RetainedGDNBlock.commit_user`) | **up to 24** (of a documented ~32 ms total: "~8 ms per user (four users x 48 layers x 5 get_device_tensors host round trips each)... dominates the packed_commit phase's host wall time" - comment at `gdn_records.py:181-183`) | Host-side, not device: each of the 4 users' commit re-walks all 48 GDN layers x 5 native tensors and calls `get_device_tensors` to re-confirm bindings verify() already checked this round | **Yes** - `QWEN_FAST_FAST_COMMIT=1` already exists and does exactly this: "keep the check on this round's FIRST commit_user call only... skip the three repeats that only re-confirm what verify() already established" (`gdn_records.py:183-188`). Confirm this env var is set in the serving arm; if not, it's a one-line config flip, zero code change | Low - the file's own comment frames the skip as safe ("still catches a binding actually broken before any commit of the round") | No - env var only |
| 2 | Per-segment `to_memory_config(norm_w, DRAM)` reissued for every one of the 4 users, every GDN layer | `gdn_batched_conv.py:85` (`run_batched_projected`) | ~1.5-3 (192 launches/round: 4 segments x 48 layers, ~7-8 us kernel+dispatch each) | `norm_w` is the same per-layer weight tensor across all 4 user segments; `to_memory_config` still issues a device op every call even when the address check on the next line shows no move happened | Partial - `gdn_batched_conv.py` is pinned (frozen-recipe route needed to edit it), but the **caller** is not: `DeviceLoopState._recurrence` (`gdn_device_loop_state.py:165-189`) calls `run_batched_projected` once per segment inside `_decode_packed`'s 4-user loop. Converting `norm_w` to DRAM once per layer, before the segment loop, and passing a tensor already in that memory config lets the existing `if addresses(...) != addresses(...)` check on `gdn_batched_conv.py:86` no-op on 3 of 4 calls without touching the pinned file | Low - pure hoist, no numerics change | No |
| 3 | Per-user `resident_piece` slice of the shared input projection | `gdn_device_loop_state.py:259-260` (`_decode_packed`) | ~1.5-3 (192 launches/round: 4 users x 48 layers) | Each user's 16-row recurrence window is cut from the one 64-row projection with its own `operations.slice` call | No, not with an overlay fusion - each user's slice IS the per-user work the recurrence needs; fusing it means changing the recurrence kernel to accept 4 segment boundaries in one call instead of 4 separate 16-row tensors | N/A (not attempted) | Yes - this is a C++ kernel change (authorised), not an overlay change |
| 4 | K/V ordered-cache write, one call per 32-row tile of the block | `packed_cache_writer.py:91,95` (`SegmentedOrderedCacheWriter`, `to_memory_config` once + one DRAM slice per tile) | part of the ~15 ms fold-DMA-adjacent bucket - candidate for what the team is calling "K/V fold DMA"; 32 launches/round (2 tiles x 16 full-attention layers) plus the `ordered_cache.update` kernel itself (compute, not counted here) | The writer is pinned frozen-recipe evidence (`packed_cache_writer.py:1-20` docstring: "Neither is edited... runs the audited kernel exactly as qualified, once per 32-row tile") | No overlay path - the file states plainly it stays at 32 rows because both it and the per-row alternative (`attention_batch.SerialCacheWriter`) are pinned | N/A | Yes, if pursued - needs a batch-64 ordered-cache kernel, frozen-recipe route |
| 5 | `nlp_concat_heads_decode` / `attn_decode_prep` glue | `gdn_records.py` n/a; glue lives in the K64 graft binding, `two_tile_decode.py:404-420` (native_attn branch) | Small now - `NLPConcatHeadsDecodeDeviceOperation` 9.8 ms / `AttnPrepDeviceOperation` 7.4 ms **total kernel** across the whole CSV capture (not per-round; both already run as ONE batch-64 call per layer under native_attn, confirmed by the `[PINDIAG] native_attn engaged: one attn_decode_prep at batch 64 and one nlp_concat_heads_decode at 64 users` log line) | Already done | - | - | - |

## Per-user-4x flag (explicitly requested)

**SDPA is the largest per-user-4x pattern in the round.** `SdpaDecodeDeviceOperation` in the
CSV averages ~2.05 ms/launch and fires 4 times per full-attention layer (once per packed user,
via `PackedReplayAttentionReader` - "one pooled reader per user over that user's own start word
and page tables... slice the user's rows, that user's bundles, concatenate the results in row
order", `pooled_attention_replay.py:29-33`), i.e. 64 calls/round, matching the team's ~40 ms SDPA
bucket almost exactly (64 x ~2.05 ms would overshoot at full serialization; the bucket total
implies most of these 64 launches overlap or are individually smaller than the CSV's mixed-length
average - the launch count, not the per-call ms, is the actionable fact here). This is **not**
overlay-removable: `attention_replay.py` and `pooled_attention_replay.py` are hash-pinned, and the
file's own docstring already explains why per-user is the design (`pooled_attention_replay.py:29-33`
runs 4 pooled per-user SDPA calls specifically to avoid the 32-serial-launch-per-layer alternative
it replaced). Collapsing this to one 64-row SDPA call needs a native batch-64 paged-SDPA-decode
kernel (C++, authorised) and sits alongside item 3 above as the same class of fix - it is compute
(the 40 ms SDPA bucket), not the 66 ms non-compute bucket this doc is scoped to, but it is the
single biggest instance of the exact pattern asked about.

A second candidate exists but is **not confirmed live in this round**: `attention_grouped.py`
(imported conditionally at `model_batch.py:569`, gated by a `grouped_attention` flag) folds
multiple users' queries into fewer, larger SDPA calls via `attention_fold_dma.device_layout_dma`
- this is a plausible source of a "fold DMA" bucket by name, but the CSV's SDPA call count (4/layer)
matches the plain per-user pooled reader, not the grouped reader's chunked-group pattern, so
`grouped_attention` is most likely off in this trace. Flagging as an open question rather than a
finding: worth a one-line check of the `ModelBatch(...)` construction call in the serving arm for
`grouped_attention=`.

## What remains unresolved

TilizeWithValPaddingDeviceOperation/UntilizeWithUnpaddingDeviceOperation are individually the
most expensive tiny-op launches found (~38 us / ~58 us each, 5-8x a slice/clone) but I could not
confirm from static source which specific decode-path call sites produce them in the *current*
(native_m3, native_attn, batched-GDN) configuration - the tilize/untilize call sites I read in
`gdn/tp.py` (lines 308, 317, 375) belong to `forward_prefill`, not the decode path this round
runs. Closing this needs either a profiler wrap around the decode-path candidates (the pattern
`gdn_prefix.py:121` already uses for one function) or a fresh traced-mode CSV filtered to one
packed decode round, not the eager capture used here.
