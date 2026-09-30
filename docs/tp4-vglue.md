# TP4 verify-glue levers (branch tp4/vglue)

The packed 64-row verify at four cards spends most of what is not weights on per-user launches and glue: conv-gates launched four
times per GDN layer (192 launches per replay: 3.97 ms of kernels and 2.25 ms of dispatch stalls), 960 per-user GDN glue ops
(6.07 ms), 37 attention glue ops per layer (2.3 ms), a 2-core Reduce in the sampler (0.79 ms) and, outside the trace, four GDN
commit traces per round that wait on every tile (2.6 ms). Device profile, trace 0, per chip, live-4: 61.8 ms of span.

This branch replaces each of them with a **byte-identical** launch, behind its own flag so an A/B can isolate it. Nothing has run on
a card: every saving below is an estimate from that profile, every exactness claim below is either by construction or waits on the
in-trace audit (`QWEN_FAST_TP4_VGLUE_AUDIT=1`, gate profile only). The audit covers V2, V1 and V4a: it fails when an engaged lever declined on an audited layer (4 piece entries + 1 merge with V2, 8 block entries per user with V1, a value reference with V4a) and when nothing was compared. C1a and V3a have no in-trace audit: they are checked by comparing G2's texts with S3a's offline.

| Lever | Flag | What | Estimate | Exactness |
|---|---|---|---|---|
| C1a | `QWEN_FAST_TP4_COMMIT_LANES` | GDN commit DMA with eight tiles in flight per barrier (`gdn_commit_lanes_tp.cpp`, from the unqualified `gdn_commit_batched_dma.cpp` with the TP4 page counts as defines) | -1.8 to -2.1 ms per round | by construction: the same bytes to the same places |
| V4a | `QWEN_FAST_TP4_SHARD_VALUES` | the shard maximum is `ttnn.gather` at the shard's own argmax id, not a 2-core `ttnn.max` | -0.78 ms, **may be zero or negative**: `ttnn.gather` is tile based and a row-major input may be tilized whole (64 x 62080 per chip) and read a full row per index tile row, about the work of the Reduce it replaces; G3b measures it and the flag stays off if it does not save. Cheaper exact options: gather from the TILE `logits` with a tiny tilized index, or one small DMA launch reading `logits[row, id]` | the maximum's bits (a zero of the other sign at worst: the combine compares them equal); a NaN row differs; falls back to `ttnn.max` if this build's gather refuses row-major |
| V2 | `QWEN_FAST_TP4_GDN_GLUE` | the GDN split (block projection to per-user pieces) and merge (users' outputs to the block) as one half-tile DMA launch each (`gdn_rows_dma_tp`), through a twin `DeviceLoopState` | -4.7 ms | the served untilize/tilize round trip maps a zero-exponent bf16 to +0 (gdn_prefill_conv_exact.py, measured on card M); the kernel applies that rule to exactly the halves the served path round-trips (users 1 and 3, every merged output). **Exact on observed data only until an edge probe injects -0 and denormal rows into the block and the outputs on a card (no such job exists yet; the rule was measured on the prefill conv's x_padded round trip, a different op instance and shape). The audit copies with `ttnn.clone`, which is assumed a plain copy; if it canonicalised, it could not see this rule at all.** |
| V1 | `QWEN_FAST_TP4_GDN_BLOCK_CONV` (needs V2) | one `gdn_decode_conv_gates` at batch 64 over the block with four stacked windows (`gdn_block_conv_tp`), instead of four per layer | -3.8 ms on top of V2 (3.3 to 4.2) | the op is row-parallel; that a row's result does not depend on its tile row and face is the audit's to show; its canon block shares V2's zero-exponent rule, so it is exact on observed data too |
| V3a | `QWEN_FAST_TP4_ATTN_FOLD` | the attention fold-in (block query to each bundle's stacked query) and fold-out (each SDPA result to the block output) as one launch each (`attention_block_fold_tp`); the SDPA launches are untouched | -2.3 ms | by construction: slices, concats and row copies only |

The four verify levers (V2 -4.7, V1 -3.8, V3a -2.3, V4a -0.8) sum to about -11.6 ms of the 61.8 ms verify (about 50 ms); C1a's -1.8 to -2.1 ms is commit time outside the verify trace. Together about -13.4 ms per round: a live-4 round of about 111 ms (124.5 ms today), about 46 tok/s per user at tau 5.1. V2 and V1 add 96 to 192 DMA launches per replay that these estimates do not net out.

## Not here

* One multi-user SDPA launch (V3b) and folding the K/V write into the attention prep: the `share` mode needs every entry of a launch to
  read one page-table row, and eight entries on 110 cores chunk the context differently from two on 55, which changes the order the
  partial results combine. Not byte-identical without a factory change that emulates the B=2 partition.
* A vector argmax (V4b): the `ArgMax` is already at the per-core scalar limit; beating it is a new compute kernel.
* Later kernel work: a windows sibling that writes block windows directly (about -1.5 ms, V1b), K5-A block I/O (removes the split and
  merge entirely, needs card requalification), one commit trace for all users (C1b), the concat-heads DMA (V3c).

## Where things are

* Flags, markers, the in-trace audit compare: `scripts/ci/tp4_vglue.py`. Strict `0`/`1`, read at capture or attach, any flag at two cards raises.
* Twins: `gdn_device_loop_state_tp.DeviceLoopState` (V2, V1; bound by `tp_addresses.TWINS`, delegates to the pinned class with the flag off),
  `extent_attention_replay_tp.PackedExtentReplayReader` (V3a), `gdn_commit_dma_tp.prepare` (C1a), `verify_trace_t1.sample_shards` (V4a).
  `gdn_user_batch_conv.run_user_batched_projected` takes an optional `block_stage` (V1). The pinned sources are untouched.
* Tests (`py -3.11`): `test_tp4_vglue`, `_attention`, `_gdn`, `_twin`, `_block`, `_window`. The kernels are emulated from their own loops over the runtime
  arguments the host planners write, on int16 tiles with -0, denormals and NaN payloads, against the served composition.
* Profiles: `c2-packed-tp4-speed-vglue` (all five, timed), `c2-packed-tp4-gate-vglue` (all five, audited), `c2-packed-tp4-speed-vglue-{c1a,v4a,v2,v1,v3a}` (one lever each).
* Jobs: `scripts/ci/references/tp4-vglue-jobs/` (ORDER.txt): G0 build, G1 timed smoke, G2 the S3a matrix on the audited profile (strict), G3 per-lever A/B.
