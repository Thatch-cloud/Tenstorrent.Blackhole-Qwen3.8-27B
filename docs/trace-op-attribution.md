# Per-op attribution of the packed verify round (eager capture)

Source: `cpp_device_perf_report.csv` (52,915 rows), eager-mode device profile of the
`-v12` arm, K64c kernels, four users, `max_tokens=48`, `QWEN_FAST_PROFILE_DUMP_ROUND=4`.
**This capture predates GDN user batching (recurrence ran 4x/layer) and packed
proposals.** Only `DEVICE KERNEL DURATION [ns]` is used below; `OP TO OP LATENCY` is
eager-dispatch noise and is ignored throughout, per the team's standing note.

## 1. Round boundary

Rows are blocked by device, not interleaved: device 0 is rows 0-26482, device 1 is
rows 26483-52914 (26,483 and 26,432 rows). Within a device, `GLOBAL CALL COUNT` is
monotonic but not evenly spaced, so row index (not GCC) is the reliable ordinal.

Two marker ops pin the round: `EmbeddingsDeviceOperation` fires **once** per device,
at row-index 15445 on both (GCC 15,816,704 / 15,816,705) — this is the tied-weight
logits projection, not input embedding (it sits right before the LM head, not at the
top). `ArgMaxDeviceOperation` fires **twice** per device, at indices 20276 and 20280
— one sample per 32-row tile of the 64-row block (`two_tile_decode.py`). Structural
counts confirm layer geometry inside that span: `AttnPrepDeviceOperation` = 32 and
`NLPConcatHeadsDecodeDeviceOperation` = 32, i.e. 16 attention layers × 2 tiles. There
is no discrete "48 GDN layers" marker op — GDN's contribution is the per-token
`slice`/`clone` loop described in §3, not a named per-layer op.

**Round taken = rows 0-20280 inclusive, per device** (GCC 1,024-20,767,744 on device 0,
1,025-20,767,745 on device 1). This is one 64-row verify forward: GDN-recurrence-heavy
rows 0-15444, then embeddings + 16 attention layers + norms rows 15445-20280, ending at
the second `ArgMax`. **Device 0 and device 1 agree closely**: total kernel time in this
span is 1564.299 ms (dev0) vs 1564.575 ms (dev1), a 0.018% difference, and both marker
ops sit at identical indices — a clean symmetric TP=2 split.

Rows 20281-26482 are a **separate pass**, excluded from the table below: `SDPAOperation`
(distinct from `SdpaDecodeDeviceOperation`), `RotaryEmbeddingHfDeviceOperation`,
`PagedFillCacheDeviceOperation`, and `ChunkGdnPrepOperation`/`ChunkGdnScanOperation`
(the chunked-prefill GDN scan) appear *only* here, never in the round. This is the
pipelined draft-proposal + KV-commit pass (task #45), structurally a small
chunked-prefill, not the 64-row decode asked about. It also contains 51 launches of
`AllGatherMinimalMatmulAsyncOp` at ~16.7 ms **each** (~852 ms total, device 0) — do not
read that as per-round CCL cost; see §5.

## 2. Op table (100% of the round, device 0, sorted by total ms)

| OP NAME | launches | ms | mean µs | cum% | dominant core counts |
|---|---:|---:|---:|---:|---|
| GenericOpDeviceOperation | 2096 | 655.11 | 312.6 | 41.9% | 64(×128), 91(×64), 48(×492) |
| MatmulDeviceOperation | 321 | 314.40 | 979.4 | 62.0% | 32(×124), 39(×124), 43(×48) |
| SdpaDecodeDeviceOperation | 128 | 262.26 | 2048.9 | 78.7% | 110(×120) |
| TilizeDeviceOperation | 675 | 69.50 | 103.0 | 83.2% | 1(×241), 4(×192), 110(×96) |
| SliceDeviceOperation | 7794 | 57.05 | 7.3 | 86.8% | 110(×7236) |
| CloneOperation | 6960 | 50.42 | 7.2 | 90.1% | 110(×6960) |
| ReduceScatterMinimalAsyncDeviceOperation | 128 | 42.93 | 335.4 | 92.8% | 10(×124) |
| AllGatherAsyncDeviceOperation | 131 | 42.33 | 323.1 | 95.5% | 10(×129) |
| UntilizeWithUnpaddingDeviceOperation | 290 | 16.82 | 58.0 | 96.6% | 96(×188) |
| TilizeWithValPaddingDeviceOperation | 651 | 15.96 | 24.5 | 97.6% | 80(×437) |
| NLPConcatHeadsDecodeDeviceOperation | 32 | 9.79 | 305.9 | 98.2% | 12(×30) |
| AttnPrepDeviceOperation | 32 | 7.37 | 230.3 | 98.7% | 64(×30) |
| LayerNormDeviceOperation | 129 | 6.20 | 48.0 | 99.1% | 32(×125) |
| BinaryNgDeviceOperation | 208 | 3.96 | 19.0 | 99.4% | 110(×201) |
| ShardedToInterleavedDeviceOperation | 225 | 3.05 | 13.5 | 99.5% | 32(×185) |
| ArgMaxDeviceOperation | 2 | 2.54 | 1270.3 | 99.7% | 109(×2) |
| ConcatDeviceOperation | 272 | 2.30 | 8.5 | 99.9% | 110(×150) |
| UnaryDeviceOperation | 112 | 0.79 | 7.1 | 99.9% | 110(×111) |
| InterleavedToShardedDeviceOperation | 32 | 0.52 | 16.2 | 99.9% | 32(×30) |
| CopyDeviceOperation | 53 | 0.36 | 6.8 | 100.0% | 110(×52) |
| UntilizeCodegenDeviceOperation | 2 | 0.30 | 148.9 | 100.0% | 109(×2) |
| ReshapeViewDeviceOperation | 1 | 0.27 | 265.1 | 100.0% | 110(×1) |
| TypecastDeviceOperation | 6 | 0.09 | 15.3 | 100.0% | 1(×4) |
| EmbeddingsDeviceOperation | 1 | 0.01 | 10.9 | 100.0% | 64(×1) |
| **Total** | **20281** | **1564.30** | | | |

`GenericOpDeviceOperation` is an umbrella name for hand-authored `ttnn.generic_op`
kernels (no per-kernel-source field in this CSV), so it is split by core count instead
— see §3/§4 for the cc=64 and cc=48 attributions.

### Family folding

| Family | launches | ms | % |
|---|---:|---:|---:|
| GenericOp (composite, see below) | 2096 | 655.11 | 41.9% |
| Matmul (QKVO / MLP / LM head) | 321 | 314.40 | 20.1% |
| SDPA (`SdpaDecodeDeviceOperation`) | 128 | 262.26 | 16.8% |
| GDN token-loop (Slice+Clone) | 14754 | 107.47 | 6.9% |
| Tilize/Untilize/pad (layout) | 1618 | 102.57 | 6.6% |
| CCL (AllGather/ReduceScatter, standalone) | 259 | 85.25 | 5.5% |
| Attn housekeeping (concat heads/prep/shard) | 321 | 20.72 | 1.3% |
| Elementwise (Binary/Unary/Ternary/Typecast/Copy/Concat/Reshape) | 652 | 7.77 | 0.5% |
| Norm (LayerNorm family) | 129 | 6.20 | 0.4% |
| LM head (ArgMax/Embeddings) | 3 | 2.55 | 0.2% |

GenericOp by core count (proxy for distinct kernel identity):

| core count | launches | ms | mean µs | likely source (§3) |
|---:|---:|---:|---:|---|
| 64 | 128 | 333.12 | 2602.5 | SDPA-decode companion kernel (native T16 path) |
| 91 | 64 | 66.59 | 1040.5 | attention housekeeping, unconfirmed |
| 32 | 300 | 62.13 | 207.1 | attention_fold_dma.py (rows≈4) / MLP epilogue, mixed |
| 96 | 192 | 61.05 | 318.0 | unconfirmed, attention-region cluster |
| 24 | 191 | 51.48 | 269.5 | attention_fold_dma.py (rows=3) / GDN norm-gate, mixed |
| 48 | 492 | 46.57 | 94.6 | `gdn_state_copy.copy_compact` (48-core DMA grid) |
| 81 | 192 | 16.17 | 84.2 | unconfirmed |
| 8/16 | 492 | 13.18 | — | attention_fold_dma.py (rows=1,2) |
| 31/15/23/47 | 45 | 4.83 | — | tail, unconfirmed |

## 3. Data-movement call sites

- **GDN per-token slice+clone (the largest data-movement bucket, 107.47 ms / 14,754
  launches)** — `scripts/ci/gdn_multitoken_conv.py:146-168`, `run_projected`'s
  `for token in range(rows)` loop: one `operations.slice` (view) and one
  `operations.clone` (row) per token unconditionally, plus a second `clone` of the 5
  conv-state tensors per token when that position is a packed checkpoint
  (`gdn_multitoken_conv.py:165`) — under deferred packed verify, checkpoints likely
  cover every position (any prefix length must be acceptable), so most tokens pay all
  three. Invoked once per (layer × user segment) from
  `gdn_device_loop_state.py:319-321` (`_decode_packed`, per-user `slice` +
  `resident_piece`) and `:207` (`_recurrence` → `run_projected`/`run_batched_projected`).
  **Batchable across users**, not removable per-token (true serial recurrence): the
  batched counterpart already exists — `_recurrence_user_batched`
  (`gdn_device_loop_state.py:344-384`) calls `run_user_batched_projected`
  (`gdn_user_batch_conv.py`) once for all four users' spans. This capture predates that
  path engaging for decode, so slice/clone still fires per-user here.
- **Per-user compact state transfer, folds into the cc=48 GenericOp cluster
  (46.57 ms / 492 launches)** — `scripts/ci/gdn_state_copy.py:34-71`, `_copy_state`
  (`copy_compact`): a hand-written `ttnn.generic_op` DMA kernel on a fixed 48-core grid
  copying 5 small state tensors compact↔compact. Called once per layer per non-last
  packed segment, from `gdn_device_loop_state.py:204` and `:370`. **Batchable**: one
  descriptor list covering all segments in a single `generic_op` launch instead of one
  per user, same pattern as the recurrence batching above; the file's own docstring
  notes it does the transfer "without arithmetic," so this is pure DMA.
- **QKV/MLP two-tile split, `scripts/ci/two_tile_decode.py:186-205`** (QKV raw + RoPE
  cos/sin sliced into two 32-row tiles, concatenated post-projection),
  **`:263-332`** (MLP gate/activation two-tile split+join), **`:571-587`** (a third
  generic split/join shared by more than one caller). This is a deliberate trade
  documented in `gdn_device_loop_state.py`'s module docstring: weights are read twice
  (once per 32-row tile) to stay on the one-tile L1-resident decode matmul arm instead
  of the ~13x-slower prefill 2D arm at M=64. **Hoistable, not removable in place**: the
  native M3 path (`attn_wo_decode_1d_progcfg_64`, gated in
  `gdn_device_loop_state.py:33-44`) already widens the cap to a single 64-row call —
  qualifying it collapses every slice/concat pair from this split at once, across GDN
  projection, attention QKVO, and MLP simultaneously (task #39, already active).
- **Attention head fold/permutation (small, inside the GenericOp cc=8/16/24/32
  cluster)** — `scripts/ci/attention_head_fold.py:83-88` (device-side `slice`+`reshape`
  for TP2 query-head layout) and `scripts/ci/attention_fold_dma.py:17-60` (a hand
  kernel via `ttnn.generic_op`, 8-column grid, `tiles = rows*8`, so core counts 8/16/24/32
  map to `rows`=1/2/3/4). Removable only if the native T16 attention path accepts the
  pre-fold layout directly — that path is inside the frozen SOURCES pin (§4).
- **KV-cache commit write** — `scripts/ci/packed_cache_writer.py:91-95`
  (`to_memory_config` + per-chunk `slice` into the paged cache). Lives in the
  **excluded region B** (matches `PagedFillCacheDeviceOperation` there, 26 launches),
  not in the verify round proper — the call site to target if optimizing the commit
  pass specifically, out of scope for this table.
- **`scripts/ci/model_batch.py:203,652,705,724`** — host-to-device staging
  (`ttnn.copy`) and output reshape/concat around the round boundary. Not per-layer,
  small, no material target.
- **`graft/attention/tp.py`, `graft/gdn/tp.py`, `graft/mlp.py`** — not present in this
  worktree (searched, no matches). The `gdn/tp.py:1036-1042` reference in
  `gdn_device_loop_state.py`'s docstring points at the M3native graft image, a
  different tree; could not inspect directly — a gap, not a conclusion.

## 4. Ranked kernel targets

1. **SDPA decode + its cc=64 GenericOp companion — ~595 ms combined (38% of the
   round), 128 launches each.** `SdpaDecodeDeviceOperation` (262.3 ms) and the cc=64
   GenericOp cluster (333.1 ms) both fire 128 times = 16 layers × 2 tiles × 4
   sub-launches; the 1:1 count match and adjacency in the row order strongly suggest
   the cc=64 kernel is the native-attention mask/softmax-correction companion to each
   SDPA decode call. **Mechanism: batch the 4 sub-launches/layer/tile into 1**, the
   same move already made for GDN recurrence. **Risk: high.** `SdpaDecodeDeviceOperation`
   and the native attention kernel sit under `dflash_t16_native_attention_gate.py:14-19`
   `SOURCES` (which also pins `gdn_multitoken_conv.py`) and its `NATIVE_SOURCES` C++
   list (the `sdpa/` kernel tree, `cpack_common.h`), plus
   `dflash_combined_sim_runtime.py:10` `BINARY_SHA256` pinning the compiled
   `_ttnncpp.so`. Any change here requires rerunning the dflash gate.
2. **GDN per-token slice+clone loop — 107.5 ms (6.9%), 14,754 launches at 7.2-7.3 µs
   mean on a full 110-core grid** (dispatch-bound, not compute-bound).
   **Mechanism: extend user-batching to the token loop** (§3) — `gdn_multitoken_conv.py`
   is also in the SOURCES pin, so **medium risk**: the arithmetic is unchanged
   (launches only reorder), which should preserve token-exactness, but it still
   requires requalification because the pinned file's bytes change.
3. **`copy_compact` state transfer, part of the cc=48 GenericOp cluster — up to
   46.6 ms (3.0%), 492 launches.** **Mechanism: one batched DMA descriptor list across
   all packed users instead of one per segment.** `gdn_state_copy.py` is **not** in the
   SOURCES pin list and does no arithmetic — **lowest risk of the three**, cheapest to
   land first.
4. *(Context, not new)* **Matmul family — 314.4 ms (20.1%)**, inflated by the same
   two-tile weight-re-read trade as target 1's neighbor (§3); the native single-call
   64-row path already tracked as task #39 removes this at the same time as the SDPA
   and GDN splits, making it the single highest-leverage change but already active.
5. *(Secondary)* **Standalone CCL, 85.3 ms (5.5%), 259 launches at ~330 µs, all on
   core count 10** (single-ethernet-link-bound). **Mechanism: fuse with the adjacent
   matmul** via `AllGatherMinimalMatmulAsyncOp`, which the excluded region B's
   draft/commit pass already uses but the region-A decode path does not. Fusion removes
   dispatch overhead, not the fabric transfer itself, so the realistic ceiling is the
   84 GB/s single-link bandwidth floor, not zero. No SOURCES pin found for these ops.

## 5. What the eager capture cannot tell us

- **Overlap.** Each row is a per-launch wall-clock duration on that device's queue;
  eager mode inserts host dispatch between every launch, so this CSV cannot show
  whether SDPA's 4 sub-launches or the GDN slice/clone pairs would overlap with DMA or
  CCL on another engine under a traced, pipelined build. That is almost certainly why
  the round's eager total here (1564.3 ms/device) is ~10x the known 157.8 ms traced
  floor — **read this report as a relative map of where launches and call sites live,
  not as today's absolute ms.**
- **The region A/B split is inferred**, not labeled in the CSV: marker-op position
  agreement across devices and clean op-name partitioning (§1) make it a confident
  read, not a certified one.
- **The 655 ms GenericOp bucket is opaque by name.** The cc=64/91/48/32/24/96/81/8/16
  split (§2-§3) comes from clustering, row-adjacency, and call-site line reads, not
  from a per-launch kernel-source column. A kernel-name-resolved profile (or reading
  the generated `ProgramDescriptor`s) would confirm the SDPA-companion read for cc=64
  and the `copy_compact` read for cc=48 directly instead of by inference.
- **Ops that only exist post-optimization are absent by construction**: the batched
  GDN recurrence launch and packed proposals postdate this capture, so their current
  (much smaller) footprint isn't visible here — only the pre-batch baseline they
  replaced.
- **`ttnn.ReadDeviceProfiler` records no rows for trace replays today**, so the
  157.8 ms traced floor has no CSV of its own to cross-check family shares against;
  this report can only reason from op-count and structural parity (§1), not from a
  second per-op table. A traced-mode measurement needs that gap closed first.
