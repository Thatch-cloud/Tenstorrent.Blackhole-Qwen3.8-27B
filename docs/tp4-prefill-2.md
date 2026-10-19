# tp4/prefill-2: where long-context prefill time goes on the TP4 stack, what is worth changing, and the one exact change built

**Date:** 2026-10-19. **Base:** `tp4/swift` (27af6902: the ship-4 stack with the checkpoint selection). **Status:** analysis (CPU only), one lever built and tested on CPU (default off), a card pack of
templates, and two kernel-class designs. **Nothing here has run on a card.** Every number is labelled **M** (measured, with its source) or **E** (an estimate, with its arithmetic).

## 0. Summary

1. **What a returning agent turn costs, and why.** A turn of 3.1k new tokens at 120,832 tokens of context is two passes (one 2,048-row chunk and a masked tail padded to a 2,048-row bucket) and costs about
   1.19 s (**M**: the route lines for steps at 120,832 say 1.09-1.16 s, the median of the 15 such steps at 120-160k in the window is 1.26 s; the fitted model 1.19 s). 44 percent of it is attention (SDPA), the only part that grows with context; 50 percent is the rest of the
   device work, which is flat in context (matmuls 119 ms, GDN 49, collectives 49, glue 20, norm 12, eltwise 12 per 2,048-row chunk); about 6 percent is eager dispatch and the end-of-prompt hand-off.
2. **Attention is already at the single-core floor.** The one-q-chunk-per-core call (oneq, in the ship-4 stack) puts each q chunk on its own core, and a core streams its K/V blocks one after another at
   13.51 us a block (**M**, card M K0a): a 120k step spends 2 x 16 layers x 960 blocks x 13.5 us = 0.41 s there whatever else is done. bf8 K/V reads are already used and the pass is compute-bound
   (77 GB/s of DRAM against 512). More cores per stream (a K-split) is the only way down, and it is a numerics change.
3. **The biggest lever is not in the prefill path: exact resume (tp4/exact-end, built).** Every returning turn re-prefills one full chunk it already has (`new tokens >= 2,048` in all 996 prefix-hit
   prompts of a 1.85-hour production window, median 3,545), because the checkpoint sits one chunk below the boundary. Resuming one chunk higher takes **39 percent** of the window's route time (**E**, on the
   fitted model; 49 percent of the step at 120k-160k), exact. Everything in this document is priced before and after it, because after it a typical step is ONE masked tail pass and the tail pass is
   what is left.
4. **Built here (PF2-H, `QWEN_FAST_PREFILL_HANDOFF`, gate only, default off): the final step's GDN state goes to the decode slot from device-side clones.** The ship-4 device profile (PF1, run 38051000905)
   shows what the host round trip costs: after the last chunk the device idles 48.6 ms for 240 blocking reads, and the 48-layer slot write runs 115.2 ms for 8.0 ms of device kernels (1,919 ops of host
   dispatch). The lever removes the reads and the uploads and keeps the graft's own `write_slot` (bytes written are the scratch's: exact by construction, audited on the card). Its worth is **20 ms
   (10-30) a final step, 1.6 percent of the window's route time today (0.8-2.3), 2.6 percent after exact resume (1.3-3.8)** (**E**). On production's prefill share (58 percent of card time, the owner's figure; this log
   window shows 22 percent of wall and 37 percent of busy time, see 3.5) that is 0.5-1.3 percent of card time. It is the only exact change that needs no kernel and has evidence behind it; it is small.
5. **What would be worth a kernel (designs in 7 and 8):** traced masked tails (E0, **5.7 percent today / 9.4 percent after exact resume**, **E**) and a K-split attention for the tail passes (tolerance class, 4.6 / 7.6
   percent, **E**). Neither is built; both are stopped before any build.
6. **Rejected with evidence (section 4):** larger or smaller chunks, finer tail buckets (the eager floor is flat), bf8/bf4 K/V, skipping drafter feature capture on non-final chunks (already free: 0.14 ms
   once), overlapping page writes (0.10 ms a chunk), more cores for the 2,048-row passes (96 of 130 busy, no whole second slice fits).

## 1. Sources

| tag | what | where |
| --- | --- | --- |
| PF1 | the prefill device profile: four cards, 13x10 workers each, the production profile, one 131,072-token prompt as 64 chunks of 2,048 rows, per-chunk kernel time by op class, median of four chips (chips agree within 0.1 percent), fitted a + b x context over chunks 1-63 (R^2 1.000); chunk 0 lost its profiler markers (intercept stands for it); the unprofiled twin of the same prompt measured TTFT 36.77 s and a chunk's wall as 5.5 ms + 1.1286 x kernel ms | run 38051000905 |
| OQ | the card-M record of the oneq call (360 of 360 byte-exact) and its per-layer served/oneq times at 4k-254k | runs 38066157833 (watcher), 38066212228 (Q1); `optimisation/ttnn-op/sdpa_prefill_oneq/README.md` |
| OQ-T | the solo TTFT A/B of oneq on the fusion stack: 131,072 tokens 37.0 / 36.9 s without, 31.2 / 31.1 s with (5.9 s of the 7.8 s the profile-anchored estimate gave: an effective SDPA ratio of 0.63, not 0.505); 32k 7.11 s -> 6.8 s; 4k unmoved | runs 38095202391, 38095525167, 38095893373, 38096201565 |
| PROD | the `lever N route` lines of one 1.85-hour window of the ship-4 stack serving the Swift checkpoint: 1,608 lines, 1,024 finished prompts, 996 of them prefix hits; fitted per pass. Aggregates only | production container log, 2026-10-19 |
| ER | per-forward costs measured on three earlier runs, the exact-resume scoping | `docs/tp4-exact-resume.md` (tp4/exact-end), runs 38084230415, 38083229148, 38080393657 |

`scripts/ci/prefill_cost_model.py` is the whole model as code (stdlib; `chunk`, `step`, `levers` commands) and `test_prefill_cost_model` holds it against these rows.

## 2. The per-op cost of ONE 2,048-token chunk

PF1 kernel milliseconds, with the oneq stack's SDPA (the paired call as profiled is in the model's second table). Collectives are pure communication here: the fused all-gather matmul is counted as a
matmul (the 2026-09-19 regrouping, `docs/prefill-profile-corrected-2026-09-19.md`).

| context (tokens) | matmul | GDN | collective | glue | norm | eltwise | SDPA (oneq) | kernel | wall | SDPA share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 119.3 | 48.7 | 48.9 | 20.4 | 12.2 | 11.7 | 14.8 | 276.0 | 316.9 | 5% |
| 32,768 | 119.3 | 48.7 | 48.9 | 20.4 | 12.2 | 11.7 | 67.3 | 328.4 | 376.2 | 20% |
| 131,072 | 119.3 | 48.7 | 48.9 | 20.4 | 12.2 | 11.7 | 247.6 | 508.8 | 579.7 | 49% |
| 163,840 | 119.3 | 48.7 | 48.9 | 20.4 | 12.2 | 11.7 | 307.9 | 569.1 | 647.8 | 54% |
| 253,952 | 119.3 | 48.7 | 48.9 | 20.4 | 12.2 | 11.7 | 474.1 | 735.2 | 835.3 | 64% |

The first three rows lie inside or at the edge of PF1's measured range (chunks 1-63, to 129k); 163,840 and 253,952 are extrapolations of an R^2 = 1.000 line (**E**). Wall = 5.5 ms + 1.1286 x kernel (**M**, the twin).
The paired call as profiled costs 806 ms of wall at 131k and 1,276 ms at 254k.

The same chunk at 131,072 tokens of context, by the categories asked for (kernel ms; **M** except the in-situ SDPA factor):

| category | ms | share | what it is |
| --- | ---: | ---: | --- |
| attention SDPA | 247.6 | 48.7% | 16 layers; 5.05 + 3.384 ms per 1k tokens of context paired, x 0.55 with oneq in the served stack |
| MLP weights | 64.7 | 12.7% | gate/up (fused all-gather matmul, LoFi, 76 cores) 49.0, down 15.7 |
| projections | 54.6 | 10.7% | GDN in 27.4 and out 15.0, attention in 9.5 and out 2.6 |
| collectives | 48.9 | 9.6% | reduce-scatters after the GDN (fp32) 25.4, MLP 17.4 and attention 2.7 out-projections; norm statistics 3.3 |
| GDN chunked prefill | 43.9 | 8.6% | prep 16.5, scan 26.8 (48 of 130 cores), gates 0.6 |
| gate/up repack | 8.3 | 1.6% | a per-call repack into the shared scratch |
| conv | 4.8 | 0.9% | the prefill conv1d |
| glue (data movement) | 12.1 | 2.4% | GDN split/layout/state 7.7, head and attention splits 3.5, rope 0.8, kv fill 0.1 |
| norms | 12.2 | 2.4% | |
| eltwise | 11.7 | 2.3% | residual, z-gate, attention gate |
| **page writes** | 0.10 | 0.02% | `attn.kvfill`, 64 cores |
| **drafter feature capture** | 0.00 | 0% | `tap.clone`: 0.14 ms in the final chunk only, none in the others |
| **host / dispatch / clock** | 70.9 | 12.2% of wall | wall - kernel: 5.5 ms of host a chunk, the rest the cards' 300 W board limit (loaded AICLK ~1,210 against 1,345 MHz in the profile) and dispatch gaps |

Why the size of the context-dependent part matters: only the first row grows with context (3.384 ms per 1k tokens paired; about 1.9 ms with oneq); all the others are constants of a 2,048-row
chunk. At 4k of context a chunk is 5 percent attention; at 254k, 64 percent.

## 3. The step: why 3.1k tokens at 120k cost 1.1 to 1.2 s

### 3.1 The passes

`prefill_traced_chunked` runs every full 2,048-row chunk and then the tail through the masked bucket path (`prefill_masked_bucket`, buckets 128/256/512/1024/2048). In the served merged route both are
eager (`trace_mode decode_only`): the chunk loop is the eager masked forward at full bucket, the tail the same forward at the bucket that holds it. A step of 3,112 tokens is one chunk and a
1,064-token tail that runs in the 2,048-row bucket (984 padded rows).

### 3.2 What each pass costs (PROD, **M**: milliseconds against the step's start context in k tokens)

| pass | a + b x context |
| --- | --- |
| a 2,048-row chunk (non-final) | 321.1 + 2.108 x ctx |
| the end-of-prompt tail pass, bucket 128 | 324 + 1.39 x ctx |
| bucket 256 | **290 + 3.31 x ctx** |
| bucket 512 | 329 + 1.27 x ctx |
| bucket 1024 | 381 + 1.25 x ctx |
| bucket 2048 | 407 + 1.66 x ctx |

Three facts read off the fits:

* The chunk's slope 2.108 ms per 1k tokens is 1.1286 x 3.3841 x 0.552: the served oneq SDPA is 0.55 of the paired call (card M's 0.505 per layer, with the cards' power limit taking some back).
* **A tail pass costs a flat 320-410 ms before any attention.** Bucket 128 costs 324 ms at no context; bucket 2048 costs 407. The device work of a 128-row forward is a few tens of milliseconds: the rest is
  the eager dispatch of ~3,155 ops (~100 us each) and the end-of-prompt work. A smaller bucket therefore does not make a short tail cheap, and finer buckets (a 1,536) would not either.
* **Bucket 256 is an anomaly: +300 ms at 160k of context.** Its two q chunks per head pair onto six cores (the stock SDPA, not the chain: rows 256 are outside 512/1024/2048), so each core streams two
  K lengths. The 128-row bucket has one q chunk per head, the 512-row one runs the chain with one chunk per core. A tail of 129-256 tokens (6.4 percent of the window's prompts) pays it.

### 3.3 The 120,832 / 3,112 step (**M** model, **E** split)

| part | ms | share |
| --- | ---: | ---: |
| attention, two passes at ~121k and ~123k | 521 | 44% |
| the rest of the device work, two passes | 589 | 50% |
| eager dispatch, end-of-prompt hand-off, other | 76 | 6% |
| **total** (route lines: 1,094-1,161) | **1,186** | |

**The context-dependent part is 44 percent.** The 6 percent includes the final step's host round trip of the GDN scratch (section 5).

### 3.4 The production population

1,024 finished prompts in the window: 29 cold (6 percent of route time), 746 hits of fewer than 4,096 new tokens (63 percent), 209 of 4,096-8,191 (22), 40 of 8,192 and more (9). **All the prefix hits have
at least 2,048 new tokens (minimum 2,065, median 3,545, p75 4,096)**: the resume point sits one chunk below the boundary the cache already publishes (`docs/tp4-exact-resume.md` section 1.1: the `drop_last`
of vLLM's eagle handling), so each returning turn re-prefills a chunk it has. Start contexts are two groups, 0-100k and 160-224k. Tail buckets (all prompts): 128: 61, 256: 66, 512: 126,
1,024: 243, 2,048: 526. 755 prompts are one step, 202 two.

### 3.5 The prefill share

The brief puts prefill at 58 percent of card wall time. This window's route lines sum to 1,437 s of 6,668 s of wall (21.6 percent), against 2,377 s of verify rounds (35.6 percent); counted as busy time
prefill is 37 percent. The log is a sample (one process, 1.85 hours) and the 58 percent may be counted on another basis; **savings are given in milliseconds per step and as a share of this window's route
time, so either basis can be applied**. Rebinds add ~88 s in the window (150 events, median 541 ms), outside the route lines.

## 4. The candidates, ranked

Savings are **E** unless stated; "today" is the window's route time under the fitted model (1,336 s model against 1,437 s measured), "after P0" the same population with the first chunk of every hit gone.
Class: **E0** byte-exact by construction (audited exactness arm), **E1** exact against a split twin, **T** a tolerance-class numerics change (owner waiver), **K** needs a new kernel.

| # | change | class | at 120-160k, a 3.3k step (ms) | today | after P0 | cost (days) | risk |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | **exact resume P0**: resume one chunk higher (tp4/exact-end, built) | E0 | -580 to -670 (49%) | -39% | | built; one card window | low |
| 2 | traced masked tails: the tail pass becomes a replay (valid_len moved off the host) | E0, K | -21 (bucket 2048) to -130 (bucket <= 1024) | -5.7% | -9.4% | 10-15 | high: trace region, GDN op arguments |
| 3 | K-split SDPA for the tail passes (section 7) | T, K | 0 (2,048-row tails) to -110 (<= 1024-row) | -4.6% | -7.6% | 15-25 | high |
| 4 | `QWEN_GDN_PREFILL_MMRS=1` (fused out-projection + reduce-scatter; the flag exists, off in every TP4 profile) | unproven (fp32 sum order) | -12 to -25 a pass | -2.4% | -2.4% | 1 | medium: deadlocked on 1x2, TP4 unproven |
| 5 | **PF2-H device-side hand-off (built)** | E0 | **-20 (10-30)** | **-1.6%** | **-2.6%** | 3 (done) | low |
| 6 | tail bucket 256 -> 512 above 16k of context | E1 | -250 for a 129-256 token tail, 0 otherwise | -0.8% | -1.2% | 1 | low |
| 7 | rebind: the 1,244 zero-fills of the slot, done when the engine parks instead of on the request's path (the PF1 timeline after the hand-off: 287 ms of host dispatch for 2.6 ms of kernels) | E0 | not a prefill pass: -190 to -290 on a prompt that rebinds | | | 3 | outside this programme (engine reuse) |

How the brief's candidates fall out:

* **A larger chunk for short tails / a smaller one.** Closed by section 3.2: a tail pass is a flat 320-410 ms, so neither helps, and `MAX_PREFILL_CHUNK_SIZE` 32 was worse before
  (`docs/prefill-chunk-size-verdict-2026-09-22.md`). A 2,560-row chunk would fit oneq's grid (20 q chunks x 6 heads = 120 cores) and cut the K stream per token by 20 percent (E: 3-4 percent of a
  solo 128k prefill), but it breaks the 2,048-token alignment of every checkpoint, step and drafter window. Not pursued.
* **Attention with bf8 K/V.** Already bf8 (Q, K and V). The call is compute-bound at 13.51 us a block; a chain injector streams 87 MB of K/V per layer at 170k in 18 ms, 77 GB/s of 512.
* **Split-K for small q against many keys.** It helps only passes that leave cores idle (a tail of 1,024 rows or fewer); the 2,048-row passes keep 96 of 130 cores busy and cannot take a second slice.
  Section 7.
* **Not re-reading K/V for the drafter window.** The window features are taps captured in the chunk loop (`tap.clone`: 0.14 ms in the final chunk, none before); nothing re-reads K/V. What is
  expensive is the engine rebind after the last chunk (item 7).
* **Overlapping page writes.** `attn.kvfill` is 0.10 ms a chunk.
* **Skipping drafter feature capture on non-final chunks.** Already free (above).
* **Fusing GDN prefill ops.** GDN is 8.6 percent of a chunk, its glue 1.6; the fusion programme's decode levers have no prefill twin yet. Kernel work, 3-5 percent of a chunk at best; not ranked.
* **Overlapping the eager tail's dispatch with the previous chunk's device time.** Unsafe: the tail's eager ops allocate while the previous pass's buffers are live (the `synchronize_device`
  between passes is what keeps the allocator and the scratch honest), and the gain exists only for buckets under 2,048 rows.

## 5. Built: PF2-H, the device-side hand-off of the final step's GDN state

### 5.1 What is paid today (PF1, chip 0, after the last chunk of the profiled prompt)

| phase | wall (ms) | device kernels (ms) | what |
| --- | ---: | ---: | --- |
| logits to the host, then `_read_scratch` | 48.6 | 0 | 240 blocking reads (48 layers x one recurrent state and four conv taps); the device waits |
| `_write_gdn_slot`: uploads | 18.7 | | 240 host-to-device writes, each followed by a device tilize |
| `_write_gdn_slot`: recurrent state, 3 ops a layer | 7.2 | | slice, concat, copy into the batched buffer |
| `_write_gdn_slot`: conv taps, 32 ops a layer | 80.2 | | the batch row of a TILE tensor cannot be sliced, so each tap untilizes, slices, tilizes, concatenates, copies |
| `_write_gdn_slot`: first-layer gaps and the rest | 9.1 | | |
| **hand-off total** | **163.8** | **8.0** | 1,919 ops, almost all host dispatch |

This is a profiled run, so it is the upper end of what production pays: the production end-of-prompt pass minus a chunk pass is 25-65 ms at 40-200k of context, which bounds the production hand-off
from above (model: 30-90 ms, central 60). The profile's per-tap sequence has no slice of the rows before the slot (8 ops), which is what slot 0 gives and the cheapest case: for other slots `_write_index` adds a second slice and two untilizes per tap.

### 5.2 What the lever does

`QWEN_FAST_PREFILL_HANDOFF=1` replaces the host read-back and the uploads with `ttnn.clone` of each scratch tensor (240 device bit copies, same dtype, layout and DRAM memory config), and hands the clones to the
graft's own `TPGatedDeltaNet.write_slot` (which consumes them). All clones are made before the first write, so a clone that fails writes nothing and the request is served on the host path (`fell back`,
latched off). Holding all 240 clones until the last layer is written costs ~70 MB a chip (48 layers x 1.4 MB: a 0.8 MB fp32 recurrent state and four 164 KB conv taps), against the ~4 GB the traffic arm keeps free after the first round (**M**, `docs/tp4-octo3.md`); the host path held 1.4 MB at a time. The write itself, its ops, its order and its slot rows are unchanged; the remaining cost is the conv taps' 32 ops a layer, which section 5.5 prices as the next step.

| piece | where |
| --- | --- |
| the module: strict flags, clones, the write, the audit, the markers | `scripts/ci/prefill_handoff.py` (an overlay file; the route imports it only when a flag names it) |
| the route: the final step of `route` and of the merged route, and the merged warm, take the device path | `scripts/ci/levern_route.py` (flag unset: the same calls in the same order; nothing is imported) |
| flags | `QWEN_FAST_PREFILL_HANDOFF`: unset off, `0` off with the timing line (the control arm), `1` on; `QWEN_FAST_PREFILL_HANDOFF_AUDIT=1` needs `1`; both need `QWEN_FAST_LEVER_N=1`; anything else raises at the attach |
| markers | `[PINDIAG] prefill handoff engaged mode=device layers=48 clones=240` once; `[PF2] handoff mode=<host|device> ms=.. read_ms=.. [clone_ms=..] write_ms=.. layers=..` per final step; `... fell back reason=..`; `... audit n=.. exact=True`; `... audit mismatch` |
| smoke rule | `c2_smoke_check.py` (placed by `make_fusion_profiles.py` from `fusion-wp/PF2.json`): the engaged line is required when a profile sets the flag to 1, a fell-back line always fails, the audit flag needs an `exact=True` line |
| profiles (gate only, outside the fusion namespace) | `make_pf2_profiles.py`: `...-pf2-s4-ctl` (the ship candidate with the gate marker and `=0`), `...-pf2-s4` (`=1`), `...-pf2-s4-audit` (the ship-4 audited twin plus both flags) |
| the window | `scripts/ci/references/fusion-jobs/PF2` (`make_pf2_jobs.py`; section 6) and the reader `scripts/ci/pf2_report.py` |
| the model | `scripts/ci/prefill_cost_model.py` |

### 5.3 Exactness

`to_torch` followed by `from_torch` to the same dtype is the identity on the values (the old path), and a clone is a bit copy (the new one: `ttnn.clone` without a dtype change is a reader-to-writer copy with no compute kernel in the pinned tt-metal, `clone_program_factory.cpp`, `convert_dtype` false); both feed the same `write_slot`. The audit proves it on the card:
on the first 8 final steps and every 24th it reads the whole batched decode state of all 48 layers before and after the write and requires (a) the slot's rows equal to the host copy of the scratch,
byte for byte, on every chip, and (b) every other row unchanged. A prompt of one step is the stock path (`prefill_paged_slots`) and is not touched; production's returning turns are all route steps.

### 5.4 Tests (CPU, on the real staged `model.py` with the recording fake ttnn)

`test_prefill_handoff` (33): strict settings; the device slot equals the host slot, with the logits and the KV, at six lengths on both routes; on the final step 15 reads and 15 uploads disappear
(three GDN layers x five tensors) and 15 clones appear; non-final steps do nothing new; the slot write is counted for the lifecycle; the scratch survives for the next prompt (and a hand-off that gave the
scratch itself to the consuming write is SEEN: the next prompt fails); flag unset = no handoff and no line, `0` = the same device calls as unset plus the timing line; a corrupting clone is caught by the
audit and goes unnoticed without it; a write that touches another slot is caught; a refused clone falls back once and stays there, and a clone that fails midway writes nothing; the warm compiles what the final
steps need and nothing is compiled after it (stage-1 and merged); the module's ttnn and model calls are a closed list. `test_pf2_profiles` (11), `test_pf2_pack` (23), `test_prefill_cost_model` (20), and the
census tests of every profile-enumerating module learned the three twins. The route's write-site census (`test_levern_prefix_prestage`) is unchanged and passes.

### 5.5 What is next on the same path

The conv taps are 49 percent of the hand-off (80 ms of 164, 32 ops a layer). A blend with a one-hot mask (`ttnn.where`, or a persistent per-slot mask) writes a tap in 2-3 ops; a persistent per-slot
trace of the whole hand-off (8 traces of ~1,900 ops, ~10 ms replay) removes all of it but needs trace-region headroom the memory plan does not show. Both are exact in principle and neither is built:
the first needs a card probe of `where`'s broadcast shapes on TILE tensors, the second a measurement of the free trace region after the attach (A1's memory ledger records the DRAM; the trace region is
a second line to read). Worth: another 20-40 ms a final step (**E**).

## 6. The window pack (`references/fusion-jobs/PF2`, image placeholder `tp4-pf2-1`, templates only)

| block | jobs | minutes (E) |
| --- | --- | ---: |
| gates | B0 window build (ship-4 profile baked, K64j-OQ graft), X0, **A1 audited attach on Swift** (stop), **AP1 audited returning turns** (stop: `tier-timed` at about 32k/128k/254k) | 250 |
| smoke ABAB | T1 control, T2 lever, T3 control, T4 lever on Swift: warmup, coding, parked_turns (40 final steps), prefill_ladder_solo (4k-254k), prefill_few_decoders (a 120k and a 254k arrival beside decoding seats), concurrent8_steady | 220 |
| returning turns | R1 control, R2 lever: `tier-timed`, the hand-off at 125k and 245k of context | 180 |
| end | Z | 10 |

Total 660 minutes (620 on cards). There is no 160k rung: the hand-off does not depend on the context (the state it moves is the same size), and the returning-turn pair reads it at 125k and 245k. Every smoke job carries `C2_MODEL_REPO=ukisai/Swift-1.5-Qwen3.8-27b`, `C2_MODEL_REVISION=b4c84d42903a8646b25857eb2827288d92ed85a4`; the prefix steps do not forward the selection
(`MODEL_ACTIONS`) and serve the base checkpoint, which the hand-off does not depend on. The measurement is the `[PF2] handoff` line both arms print, read by
`python3 -B scripts/ci/pf2_report.py --control <T1 log> --control <T3 log> --arm <T2 log> --arm <T4 log>` (mean/median/p90 of the hand-off per final step, the saving, the route milliseconds by context
band; exit 1 when the pair is not a pair), beside `prefill_ladder_report.py` for the cold ladder (TTFT ratios and equal answers). The expected saving is a few tens of milliseconds a step against ladder TTFTs of
1.8 to 97 s, so the TTFT pair cannot show it above its noise (the control-to-control spread of the 4k rung in OQ-T is 1.8 percent, the saving 1-3); the hand-off line can. Each template parses through `c2_serving_job.py` with rc 0 and no refusal. The CPU suite
must be green on the pushed head before any tag; the ARC runner sets are at zero from A1 to R2.

## 7. Design (not built): K-split attention for the tail passes

**Why it needs a kernel and a waiver.** A q chunk's K stream is serial on its core because the online softmax is a sequential recurrence over blocks (running max, running sum, rescaled output), and
the sequence of roundings is the kernel's result. Sharing a stream between S cores changes the association (each slice starts from an empty state, a merge combines them), so no K-split is
byte-equal to the sequential call. A bit-exact variant exists on paper (cores compute the per-block row maxima, which are order-free, and the partial products P V, which are independent given the
global prefix maxima, and one core replays the rescale-and-add chain in order) but the replay moves one [128 x 256] partial per block over the NoC (~9 us at 14 GB/s) against 13.5 us of compute per
block today: it cannot win. The change is therefore **class T**: same result up to the rounding of a different association; needs the owner's numerics waiver and "real-text exact" (tokens identical on
the corpora) rather than byte equality.

**What it extends.** The K64j-OQ prefill factory (PS0-PS5 over the PF chain factory) and the pinned SDPA compute kernel (`compute_common.hpp`, tt-metal 9f9cd4fd). The pinned kernel already contains
what a merge needs, written for ring attention: with `sdpa_type == RING` a pass emits the normalised output and its log-sum-exp (`cb_out`, `cb_lse_out`: `lse = scale x max + log(sum)`) and a later
pass folds a previous pair in with `out = prev_out - sigmoid(cur_lse - prev_lse) x (prev_out - cur_out)`, `lse = prev_lse - logsigmoid(prev_lse - cur_lse)` (`cb_lse_in`, `cb_prev_out`; chunked
causal masking in the global frame is there too, `ChunkedContext`). So no new arithmetic: the compute kernel is a new INSTANTIATION of the existing template (JIT), and the new code is data movement.

**Core grid and layout (TP4: 6 local q heads on 1 KV head, 130 workers, q/k chunk 128).** A pass of R rows has `busy = 6 x R / 128` q-chunk cores in oneq. Slices `S = min(4, floor(130 / busy))`:
R = 2,048 -> busy 96, S = 1 (the lever does not engage); R = 1,024 -> 48, S = 2; R <= 512 -> <= 24, S = 4. A (head, q chunk) gets S consecutive cores, slice `s` taking the K blocks
`[s x B/S, (s+1) x B/S)` of the chunk's `B = C + q + 1` blocks (the last slice carries the diagonal block). Cores are assigned slice-major so the chain grouping of oneq survives: the six heads that
share a (q chunk, slice) read byte-identical K/V sub-streams, so each K/V block is still read from DRAM once per q chunk and forwarded down a 6-core chain (S times as many chains of the same length).
Slice 0's core is the merge core. Each slice runs the unchanged compute loop over its range with `ring_iter = 0` semantics and writes `(out_s, lse_s)`: `out_s` [128 x 256] as 32 bf16 tiles (64 KB), `lse_s` as 4 fp32 tiles; slices 1..S-1 send both
to the merge core's L1 over the NoC (S-1 partials, <= 3 x 66 KB, against 1.5 MB of L1) and the merge core folds them in with the ring merge, in slice order, then writes the normalised output. The merge
costs ~15-20 us a layer against ~8 ms of K stream saved per layer at 170k for S = 2 (**E**).

**Exactness argument (tolerance).** Each slice is the sequential kernel on a prefix-free sub-range, so its (out_s, lse_s) is exact in the kernel's own arithmetic; the merge is the ring kernel's. The deviation
from the sequential result is the rounding of an extra bf16 round trip of out_s and of the sigmoid form of the merge: of the order of one bf16 ulp relative per merge, comparable to the call's own output rounding.
Acceptance would be: the card-M harness shows max abs and relative error against the oneq call inside a bound taken from the call's own run-to-run difference against the stock path; the model gate
shows identical tokens on real code text at 4k/32k/128k and on the agent replay; the audited attach compares each call with the oneq program within the bound (not bitwise).

**Qualification harness.** `oneq_card_m.py` gains arms `ksplit` (S = 2, 4) beside `stock`, `served`, `oneq`: the same (rows, Q memory, seed, chunk start) matrix restricted to R <= 1,024, comparing error
statistics instead of bytes; the refusals (S > 4, a pass that needs more than 130 cores, R = 2,048, an odd chunk count); the program cache (the flag keys its own program); the timing at 8 context
points; the planted-hang test and the NoC sanitiser pass on the chain protocol model (`pf_protocol_model.py` extended to S chains per q chunk and the partial hand-off).

**Graft plan.** `apply_factory_pk.py` (flag 0x10 in the program word `0x5EFA0000 | flags`, accepted only with 0x1 and 0x8; sha-checked, anchored, inverted, refusing any addition outside `create_descriptor`
as PS does), `build_k64j_oq_ks.sh` into a new directory beside `~/opgraft-K64j-OQ`, `sdpa_pf_oneq_tp.py` ORing 0x10 where `split_k_slices(rows) >= 2`, a flag `QWEN_FAST_SDPA_PF_KSPLIT` (default off) and
its audit. Estimated worth (**E**): 4.6 percent of route time today, 7.6 after exact resume; 44 percent of the tail pass's SDPA at S = 2, 71 percent at S = 4.

**Build resources (no build was started).** The build container with the pinned tt-metal at K64j-OQ's source state (three incremental ninja runs, 30-60 s each, a few GB of memory, no device), then one
p150a (card M) for about 40 minutes under the watcher and 75 minutes for the proof and timing, then one four-card window (an OQ-T-style pair, 25 minutes each, and an audited attach). Nothing runs on
the production host.

## 8. Design (not built): traced masked tails

The eager tail pass is host-bound below the 2,048-row bucket: ~3,155 ops at ~100 us each is the 320 ms floor of section 3.2, against 15-150 ms of device work. A replayed trace would pay the device
time only. The masked tail cannot be replayed today because three things in it are per-call host values baked into a trace:

1. `valid_len`, an argument of the GDN conv op and of the fused chunk op, and the conv-state one-hot (a host write under `from_torch`, which a trace capture refuses);
2. the real-block page table (a different width per call) and the fill widths of `paged_fill_cache`;
3. the logits row selector (a one-hot built per call).

(2) and (3) are persistent-buffer problems solved the way the chunk trace solves its own inputs (`copy_host_to_device_tensor` into a fixed buffer before the replay; a fixed-width page table
whose pad entries point at a block nobody reads). (1) needs one of two things:

* **a kernel change**: `valid_len` read from a device scalar by the conv and chunk ops (a new JIT variant of each, runtime-argument-free); or
* **no new kernel, class E0 if the card agrees**: neutralise the pad rows through the gates. The recurrence over a pad row is the identity when its decay is 1 and its write strength 0, which the gate
  inputs give exactly (`a = -1e4` makes `softplus(a + dt)` underflow to 0, so `g = exp(0) = 1`; `b = -1e4` makes `beta = sigmoid(b) = 0`), added to the `a` and `b` projections by a persistent
  per-call vector; the conv state is the last three real rows, selected by a persistent one-hot that is written, not built, before each replay. The fused op then runs its `valid_len = None` path, the one the full chunk uses. The claim to test on card M
  is that the recurrent state and conv state equal the `valid_len`-masked path's byte for byte (adding exact zeros and multiplying by exact ones leaves the bits alone, but the chunk algorithm's
  cumulative products have not been read against that on a card).

The cost that decides it is memory: a trace holds the dispatch commands of ~3,155 ops. The served profile prefills eagerly (`trace_mode decode_only`) and its trace region (640 MiB with the octo block) is
sized for the decode, octo and drafter traces; whether it has room for prefill traces is unmeasured. One tail trace per bucket (128, 512, 1,024; 256 is remapped to 512, 2,048 is device-bound) is three traces; their size must be measured first (capture one on
card M and read the region's use) before anything else is written. Worth (**E**): 5.7 percent of route time today, 9.4 after exact resume (21 ms for a 2,048-row tail, 120-175 ms below it).

## 9. Order of work

1. **Exact resume P0** (tp4/exact-end): its pack exists; it is half of every step. Stack it with this branch's pack in one window: A1 and AP1 here cost 240 minutes and run on the same image family.
2. **PF2-H** on top (this branch): small, exact, cheap to confirm; it grows from 1.6 to 2.6 percent of route time once P0 has removed the first chunk.
3. **The measurement that decides between 7 and 8:** the free trace region after the attach (A1's ledger records DRAM, not the trace region: add the line), and one card-M probe of the gate-neutralised
   conv/scan against the masked path. Both are about an hour of card M.
4. Item 6 (bucket 256 -> 512) as a one-line profile of the tail path once an owner decision on E1-class exactness exists; item 4 (`QWEN_GDN_PREFILL_MMRS=1`) as a one-job byte A/B on the audited twin.

## 10. Not done

Any card run; the conv-tap blend and the traced hand-off (5.5); the stock path's (`prefill_paged_slots`, a prompt of one step) hand-off, which is the same loop and not on the returning-turn path; the
two kernel designs' implementations; a rebind (item 7) change, which belongs to engine reuse.
