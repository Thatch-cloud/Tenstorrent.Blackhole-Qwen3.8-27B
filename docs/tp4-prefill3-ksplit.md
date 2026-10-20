# PF3-K: K-split SDPA for the tail passes (tp4/prefill-3, kernel design and host plumbing)

**Status.** Designed and built as source on the CPU; **nothing has been compiled by the device toolchain and nothing has run on a card.** The flag `QWEN_FAST_SDPA_PF_KSPLIT` (default off) and
the graft it needs (K64j-OQ-KS) are gate-only. Every number in section 3 is an **ESTIMATE**; every number in section 4 that is not marked otherwise is an a-priori reading or a CPU emulation of the arithmetic,
to be replaced by card M. **The numerics waiver (section 4) is the owner's decision; this document only says exactly what it would waive.** The program word, the sources and the tests are in
`optimisation/ttnn-op/sdpa_prefill_ksplit/`, the model-side hook and the profiles in `scripts/ci/`.

Labels: **M** measured, **E** estimate (arithmetic on measured inputs), **A** assumption a card run can break, **not run** nothing was executed for it.

## 0. Summary

* **What.** A masked tail pass of 1,024 or 512 rows keeps 48 or 24 of the grid's 130 cores busy under the oneq call (flag 0x8: one q chunk per core), and a q chunk's K stream is serial on its core
  (13.51 us a 128x128x256 block, M). Flag 0x10 gives each q chunk S = min(4, 130 / busy) = 2 (1,024 rows) or 4 (512 rows) **slices** on the idle cores; slice s streams the K blocks
  `[s B / E, (s+1) B / E)`, and the slices' partial results (normalised output, log-sum-exp) are folded **in slice order by the pinned ring merge**
  (`out = prev - sigmoid(cur_lse - prev_lse) (prev - cur)`). No new arithmetic: the compute kernel is a new INSTANTIATION of the pinned template; the new code is data movement.
* **Worth (E).** About **12 seconds of a 1,336-second window of route time (0.92 percent; 0.54-1.13 percent over the low-high band; 1.5 percent after exact resume P0)** at the ~59k-token mean prompt of
  production (M, the programme's figure), **about a fifth of what `docs/tp4-prefill-2.md` section 4 priced (4.6 percent)**, for three reasons that section 3 itemises: the kernel can only take the 512- and 1,024-row buckets (369 of 1,022
  prompts; the 128- and 256-row buckets are not chain-qualified), it prices the tail's own measured attention slope (1.25-1.27 ms per 1k tokens) instead of a 2,048-row chunk's SDPA, and it
  charges the clock cost of keeping 96 cores busy. At 120-160k of context a 512-row tail saves about 104-142 ms (22-26 percent of the pass) and a 1,024-row tail 56-76 ms (10-13 percent). If production
  tail passes sit at 120-160k instead of the 57k mean, the same arithmetic gives 2.0-2.7 percent.
* **Numerics (the waiver).** The output of a K-split pass is not bitwise the oneq output. A CPU emulation of the two arithmetics in the formats the served factory uses (section 4.3) says something the
  a-priori reading did not: **the sequential call's own error against a float reference is large and grows with the context** (about 0.2 bf16 ulps rms at 1k, 1.3 at 32k, 2.6 at 131k, from the bf16 accumulator rounded at every K block),
  **and the split is closer to the float reference than it is** (rms ratio 0.5-0.8 from 8k up). So the question "is the split inside a bound around oneq" has a bound that must grow with the context, and the better
  question is "is it no further from the truth than oneq": both are in the harness and the audit. The provisional bound is in `scripts/ci/ksplit_numerics.py` and **is replaced by card M's numbers before the owner decides**.
* **Even the passes with one slice change.** The lever stores the running statistics in fp32 for every pass that carries flag 0x10 (the log-sum-exp lives there; bf16 would be 6x the error of the output's own
  rounding), so a pass whose chunk is too short for a second slice (less than 1k of context) differs from oneq too. 2,048-row passes and every pass of every other bucket carry no 0x10 and stay **bitwise** oneq.
* **Not compiled.** The compute kernel has never seen a compiler (the compute API headers are not in this repository), the host descriptor plumbing is assumed from served call sites, the L1 budget of the added
  CBs is arithmetic, and the NoC hop cost (18 us per hop) is an estimate. Section 2.7 lists exactly what is checked on the CPU and what is not. The first real evidence is the build (section 5, G1) and the
  watcher pass (G2).
* **Card time to a verdict (E).** Build about 10 minutes (no device); card M about 40 minutes under the watcher, 75 minutes for the proof and timing, the planted hang last (about 15); then a four-card pair (25 minutes
  each) and an audited attach (25 minutes). About 2.2 hours on card M and 1.25 hours on four cards, without the real-text gate. Section 5.

## 1. Files (a)

`optimisation/ttnn-op/sdpa_prefill_ksplit/` (new directory; the oneq and chain directories are not edited):

| File | What it is | Status |
|---|---|---|
| `fixtures/*` | the five pinned tt-metal 9f9cd4fd sources the generators start from (writer, compute kernel, compute_common, q_chunk_remapping, chunked_prefill_utils), file names carry their sha prefix | M (shas checked) |
| `ksplit_planner.py` | `split_k_slices(rows)`, the device range arithmetic (`blocks_of`, `effective_slices`, `slice_begin`), the slice-major plan (cores, chains, relay edges), refusals, the time model (block counts, relay hops) | built, tested |
| `ks_common.py` | the one C++ text of the range arithmetic the three kernels carry byte for byte | built, tested |
| `apply_factory_pk.py` | factory patch PK0-PK13 over the PS (oneq) factory: flag 0x10 (accepted only with 0x1 and 0x8) and the test flag 0x400, S slices, slice-major cores, chain key with the slice, extra CBs, semaphores, kernel selection, runtime args | built; **not compiled** |
| `make_ks_reader.py` -> `reader_interleaved_qwen_ksplit.cpp` | the chain reader with the slice's K range and the receive side of the hand-off (K1-K6) | built; syntax-stub-compiled |
| `make_ks_writer.py` -> `writer_interleaved_qwen_ksplit.cpp` | the pinned writer with the send side of the hand-off, the ring merge's scale tile and the skip rules (W1-W7) | built; syntax-stub-compiled |
| `make_ks_compute.py` -> `sdpa_qwen_ksplit.cpp` | the pinned compute kernel with the RING instantiation (C1-C5) | built; **never compiled** |
| `ks_protocol_model.py` | discrete-event model of S chains per q chunk plus the hand-off: deadlock freedom, exact payload coverage, the planted hang, four mutations | built, tested |
| `host_exec_ksplit.py`, `stub_compile_ksplit.py`, `stubcheck/include/*` | the real decision C++ of the patched factory compiled with g++ and stubs (per-core slice, previous, next core); stub compiles of the factory regions, reader and writer | built, tested |
| `ksplit_card_m.py`, `run_card_m_ks.sh` | the card-M qualification harness (arms stock / oneq / ksplit, configs s2 / s4 / s2g, refusals, cache, grid, timing at 8 contexts, the planted hang) and its runner | **template, not run** |
| `build_k64j_oq_ks.sh` | builds the K64j-OQ-KS graft beside K64j-OQ (never overwrites); six publish checks | dry run only |
| `ksplit_estimate.py` | the ESTIMATE of section 3 (and the check that this document's tables are its output) | built, tested |
| `ksplit_emulation.py` | CPU emulation of the two arithmetics in the statistic of the waiver (section 4.3) | built, tested |
| `test_ksplit_planner.py`, `_factory.py`, `_sources.py`, `_protocol.py`, `_card.py`, `_build.py`, `_emulation.py`, `_estimate.py` | the tests (section 6) | pass on CPU |

`scripts/ci/` (shared area; the files marked "append" are one-block edits):

| File | What |
|---|---|
| `sdpa_pf_ksplit_tp.py` (new) | the model-side hook: strict `'1'`/`'0'`/unset flag parse, needs `QWEN_FAST_SDPA_PF_ONEQ=1` and a grafted binary (refuses at attach otherwise), ORs 0x10 into the oneq program word of the 512- and 1,024-row calls, markers inside the patched call (`engaged` / `skipped` / `fell back`), the audit |
| `ksplit_numerics.py` (new) | the error statistic (bf16 ulps of the row scale), the provisional bound and its growth with the context, the noise-floor rule, a float reference, shared by the harness and the audit |
| `ksplit_smoke.py` (new) | the smoke rule: what a gated arm must see in the log (engaged for 512 or 1,024 rows, no `fell back`, an audit line with `within_bound=True`, nothing when the flag is off) |
| `make_pf3k_profiles.py` (new), `qwen_c2_profiles.json` (generated) | the three gate-only twins of the ship-5 candidate: `...-er-pf3k-s5-ctl` (flag 0), `...-pf3k-s5` (flag 1), `...-pf3k-s5-audit` (flag 1 + audit) |
| `fusion-wp/PF3K.json` (new) | the manifest through which `make_fusion_profiles.py --write` placed the smoke-table row, the overlay lines and the CPU allowlist entries |
| `serving_c2_contract.py` (append) | `ksplit_problems`: the switch on a traffic profile is refused unless the owner's traffic waiver names it (the numerics waiver is absent), the audit on a traffic profile is refused, the switch needs the oneq flag |
| `tp_addresses.py` (append) | binds the hook after the oneq block, imported only when either flag is set |
| `c2_smoke_check.py`, `profile_twins.py`, `docker/qwen-c2-overlay.txt`, `.github/workflows/qwen-integration-cpu.yml` (append) | smoke row and rule, the twin list, the two image files, the CPU allowlist (tests named; the op directory discovered) |
| `test_sdpa_ksplit_tp.py`, `test_pf3k_profiles.py` (new) + 23 census tests (exact-count edits) | section 6 |

## 2. Kernel design as built (b)

### 2.1 What a pass looks like

TP4: 6 local Q heads share 1 KV head, head dim 256, q and k chunk 128 (4 tiles), 130 worker cores (13x10; an 11x10 grid is a harness pass), bf8 Q/K/V, HiFi2, fp32 dest, the flexible `chunk_start`
tensor. A pass of R rows has `total = 6 R / 128` q-chunk units (one core each under oneq): 96 at 2,048 rows, 48 at 1,024, 24 at 512. With 0x10:

* `S = min(4, 130 / total)`: 1 (2,048 rows, **refused** with a `TT_FATAL`, the host never sends it), 2 (1,024), 4 (512). A cap of 4 because more hops cost what the split saves; the cap is checked at 256 rows
  (ten slices would fit; the factory must log exactly four).
* **Slice-major cores:** slice s owns cores `[s total, (s+1) total)`, inside a slice the flat position is the oneq one, so the oneq chain grouping survives: the six heads that share a (q chunk, slice)
  read byte-identical K/V sub-streams and form a 6-core chain, **S times as many chains of the same length** (F4's key gains the slice). Each K/V block is still read from DRAM **once per q chunk**
  and forwarded down a chain; the DRAM concurrency of a 512-row split pass (16 chains) equals a 2,048-row oneq chunk's.
* **Device-side K ranges.** A q chunk has `B = C + q + 1` K blocks (C = chunk start / 128, q the q chunk's index in the pass); the chunk start is a *device tensor* (the flexible path), so the host cannot
  know B. Reader, writer and compute kernel therefore carry **the same** integer functions (`ks_common.RANGE_BLOCK`, held byte for byte by `test_ksplit_sources`):
  `E = max(1, min(S, B / 4))` active slices (a slice keeps at least 4 blocks; below that the hop costs more than it saves), `slice_begin(s) = s B / E`, slice s >= E idle (an idle slice, being a whole chain,
  runs an empty loop and does nothing else). The **last active slice carries the diagonal block** (and the remainder) and is the one that writes the output.
* **Hand-off (a relay, not a star).** The brief and `docs/tp4-prefill-2.md` section 7 say "slice 0's core is the merge core". As built it is a **relay**: slice s-1's writer sends its normalised
  output (32 bf16 tiles, 64 KB) and its log-sum-exp (4 fp32 tiles, 16 KB) to slice s's core (same head, core `i + total`) at the same L1 addresses; slice s's pinned ring pass folds it in
  (`ring_iter = 1`, `cb_prev_out` / `cb_lse_in`); the last active slice writes the result to DRAM. Why: the pinned ring path folds exactly ONE previous pair into the current pass, so a star would
  need S-1 receive buffers and a different compute loop (new arithmetic code), while the relay keeps one buffer pair and the pinned code as it is. The cost is E-1 hops on the critical path instead of
  one receive wait; the merge compute is serial in both (planner: `HOP_US = 18` per hop, pessimistic 40, **E**, section 2.6).
* **Credit, then data, then flag.** The receiving slice reserves its two CBs at the start of the unit and credits the sender (an atomic increment on the sender's semaphore 3, then a barrier: the
  Blackhole atomic-barrier hazard the chain already handles); the sender's writer waits for the compute kernel's output and for the credit, writes both payloads, waits for both acks, and only then raises
  the receiver's VALID flag (semaphore 4: data acked, then flag, the chain's own order); the receiver waits for VALID, pushes both CBs and resets the flag. The chain's own semaphores (0/1/2) are untouched.
  The protocol model (`ks_protocol_model.py`) checks termination, exact payload coverage (the final payload of every head is blocks `[0, B)` exactly), clean semaphores and CBs over thousands of
  shapes, the planted hang (test flag 0x400: every receiver withholds its credit; must block exactly the predicted set of waits with no content error) and four mutations the checker must catch
  (VALID before the acks, a write without the credit, a missing dummy K/V pair after an even round count, differing range arithmetic between the kernels).
* **The compute kernel is the ring instantiation.** `sdpa_inner_loop<RING, ...>` with `chunked_enabled` and `Nq = chunk_size = 2^20` tiles (the K frame is then the identity), `logical_nt =
  0xFFFFFFFF` (no padding, no joint), `is_causal` with the lightweight diagonal stamp (applied to the one K chunk of the last slice that overlaps the diagonal), `use_zigzag_balancing = false`,
  `is_last_ring_iter = true`. Slice 0 runs `ring_iter = 0` (emits the normalised output and `lse = scale max + log(sum)`), every later slice `ring_iter = 1`. The pinned loop pops one dummy K/V pair at
  its end after an even number of K chunks, so the reader pushes one (the model's `dummy_missing` mutation deadlocks without it). The `sdpa_ring` wrapper ignores `iter_k_chunk_start`, so the kernel
  calls the inner loop directly with all 60 function parameters / 21 template parameters (counts held against the pinned header by `test_ksplit_sources`).

### 2.2 Factory patch (`apply_factory_pk.py`)

PK0 decode of bit 0x10, S and busy count; PK1 constants and the two semaphore ids; PK2 accepted flag set (0x10 only with 0x8, 0x400 only with 0x10); PK3 0x400 joins the test-only flags
(`QWEN_SDPA_PF_TEST=1`); PK4 fp32 statistics under the lever; PK5 the extra CBs (previous output, previous lse, own lse, scale tile); PK6 the suffix compile-time args of the three kernels; PK7 the two
semaphores over the core grid; PK8 the envelope (at least 2 slices, fp32 dest and exact exp, a softmax scale exact in bf16); PK9 the F4 range of a ksplit core and the slice in the chain key; PK10 the
log line `[QWEN-SDPA-PF] ksplit=1 slices=.. q_chunks=.. cores=.. busy=.. chains=.. members=..`; PK11 the kernel sources; PK12 the per-core slice, previous and next core; PK13 the runtime args.

Rules it keeps (as the oneq patch does): the input must be the PS factory (sha-checked; the PF and the served factory are accepted and brought to PS first); every anchor occurs once at its recorded line; the
result hashes to `PK_FACTORY`; the edits invert; **every new line lies inside `create_descriptor`** (nothing at namespace scope: the unity-build collision trap); the protected
`qwen_draft_fp32_intermediates` block is byte-identical; a call without 0x10 takes the old path through guards on `qwen_ks`. Four new `[QWEN-...]` strings are the only ones the graft adds
(`NEW_QWEN_STRINGS`, checked by the build as the string-diff gate).

### 2.3 Data-movement kernels

Reader K1-K6: the slice arithmetic; the suffix compile-time args; the runtime words (slice, previous core); per unit B, E, whether this slice takes part and its K range; a receiving slice reserves its CBs and
credits; the K loop runs `[begin, end)`; after the loop a receiver waits for VALID and pushes both CBs, and an even round count pushes the dummy pair. Writer W1-W7: the two API headers the chain reader
already includes; `qwen_ks_xfer::send_partial` (bounded credit wait: a hang stays a hang, with waypoint `QKDC` and a watcher assert); the runtime words (slice, next core); the suffix args; the ring
merge's scale tile made by the writer's own `generate_bcast_col_scalar` (the merge reads element [0][0] with a scalar broadcast; the scale must be bf16-exact, PK8 guards it); per unit an idle slice skips,
a slice that is not the last sends its partial and skips the DRAM write; the last slice writes as served and drops its unused lse.

### 2.4 The fp32 statistics finding

In the served factory `stats_df = im_df = Float16_b` unless the draft path asks for fp32 (`sdpa_program_factory.fd8c0676.cpp` lines 712-715); `sum_df` and `qk_im_df` are fp32 with fp32 dest. The ring
merge stores `lse = scale max + log(sum)` in the statistics CBs; with bf16 statistics the max (|max| up to ~30) rounds to 0.125 and the merge weight sigmoid(cur_lse - prev_lse) is wrong by up to ~6 percent,
six times the output's own rounding. PK4 therefore makes `stats_df` fp32 under the lever (the new lse CBs too). Consequences: the added L1 is about 24 KB for the three fp32 statistics CBs plus 98 KB for the new
CBs (prev output 64, lse in 16, lse out 16, scale tile 2) (**E**, arithmetic on tile sizes; the build/JIT is the check, an overflow is a `TT_FATAL` at program build, a refusal rather than a hang), and **every
pass with flag 0x10 differs from oneq, including those with a single active slice** (see the waiver, W3).

### 2.5 Where it deviates from the brief

1. Relay instead of a merge core (2.1). 2. Only the 512- and 1,024-row buckets: the chain-qualified rows with a second slice (128 rows is an odd chunk count; the 256-row bucket is not in the graft's
`_QWEN_PF_ROWS` and its production fit is unreliable, slope 3.3 ms per 1k against 1.3 for its neighbours). 3. The 4-block minimum per slice and idle slices (device-side E). 4. fp32 statistics for every flagged pass.

### 2.6 Time model

Per layer, the planner counts the busiest slice's blocks at 13.51 us, adds the layer overhead and E-1 relay hops; `layer_ratio(rows, chunk_start)` is the ratio to the oneq call at the same rows. Ratios
(isolated, 18 us hop): 1,024 rows 0.60 at 4k of context, 0.52 at 32k, 0.50 at 120k; 512 rows 0.46, 0.28, 0.26. One hop is about 5 us for 80 KB at ~16 GB/s of one unicast write plus about 12 us
for 32 tiles of sub, mul, sub and the fp32 sigmoid / logsigmoid (**E, A**; 16 layers x 1 hop x 18 us = 0.3 ms per pass at S = 2, small against the saving, 0.9 ms at S = 4).

### 2.7 What is verified where

| Check | How | Result |
|---|---|---|
| Patch is reproducible, anchored, inverts, stays inside `create_descriptor`, protected block untouched | `apply_factory_pk.py`, `test_ksplit_factory.py` | pass |
| The per-core decisions of the REAL patched C++ equal the planner's over ~200 shapes | `host_exec_ksplit.py`: the factory's decision regions compiled with g++ and stubs | pass |
| Factory regions, reader and writer parse as C++ against stub headers | `stub_compile_ksplit.py` | pass |
| **Compute kernel compiles** | no compute API headers in the repository; text-anchored against the committed pinned header (60 parameters, 21 template parameters, the ring merge's CB uses, the epilogue's dummy pop) | **not compiled** |
| Host descriptor plumbing (`KernelDescriptor` runtime args, `SemaphoreDescriptor`) | modelled on the served call sites in the same factory | **A, not compiled** |
| L1 budget of the added CBs | tile-size arithmetic (2.4) | **E** |
| Protocol: no deadlock, exact payloads, clean semaphores, planted hang, four mutations | `ks_protocol_model.py` | pass (model, not NoC) |
| NoC order, atomics ordering, barrier hazards | the watcher / NoC sanitiser pass, G2 | **not run** |
| Build script | dry run only (`KS_BUILD_DRY_RUN=1`) | **not run** |

## 3. Expected effect (c): ESTIMATE

All of this section is **E**; nothing ran. `python3 -B ksplit_estimate.py` regenerates the tables (a test holds them identical to this document's).

**Model.** A tail pass of R rows costs `a + b x context` (the production fit, M, `prefill_cost_model.PRODUCTION_TAIL`): 329 ms + 1.274 ms per 1k tokens for 512 rows, 381 + 1.251 for 1,024 rows. The slope
b is the attention, the only part that grows with context. The K-split multiplies it by a ratio r (the planner's block counts and relay hops), and the in-situ ratio multiplies r by a clock penalty P, because a
split pass keeps 96 cores busy (S x the 24 or 48 of its oneq tail), the busy-core count of a 2,048-row chunk, which runs at the 300 W board limit (loaded clock ~1,210 MHz against 1,345 in the profile):

* LOW: P = chunk slope / tail slope = 2.108 / 1.274 = 1.65 (512 rows), 2.108 / 1.251 = 1.68 (1,024 rows) (every split core runs at the chunk's per-block time), 40 us hops.
* CENTRAL: P = 0.63 / 0.505 = 1.25 (the measured in-situ factor of the oneq lever: its card-M ratio 0.505 became an effective 0.63 in the served stack), 18 us hops.
* HIGH: P = 1 (the isolated model), 18 us hops.

Only the 512- and 1,024-row buckets engage. Population (M, `docs/tp4-prefill-2.md` section 3.4): tail buckets 128: 61, 256: 66, 512: 126, 1,024: 243, 2,048: 526 (1,022 of the window's 1,024 finished prompts);
the start contexts are not published per prompt (two groups, 0-100k and 160-224k), so the default puts every tail pass at the mean (the 59k mean prompt less 2k) and prints the sensitivity; the saving is
close to linear in the context, so the mean is what matters.

<!-- ksplit-estimate:begin -->
Per tail pass (ms; E; the pass is the production fit a + b x context; "isolated r" is the planner's per-layer ratio, "in-situ r" multiplies it by the central clock penalty; "older row" = prefill_cost_model.split_k_saving_ms for the same pass):

| bucket | context | the pass today | S | isolated r | in-situ r (central) | saved low | saved central | saved high | central share of the pass | older row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4,096 | 334 | 4 | 0.46 | 0.57 | 0 | 2 | 3 | 0.7% | 12 |
| 512 | 32,768 | 371 | 4 | 0.28 | 0.35 | 21 | 27 | 30 | 7.3% | 54 |
| 512 | 65,536 | 412 | 4 | 0.27 | 0.33 | 45 | 56 | 61 | 13.5% | 101 |
| 512 | 120,832 | 483 | 4 | 0.26 | 0.32 | 87 | 104 | 114 | 21.6% | 182 |
| 512 | 163,840 | 538 | 4 | 0.26 | 0.32 | 119 | 142 | 155 | 26.4% | 245 |
| 512 | 253,952 | 653 | 4 | 0.25 | 0.32 | 186 | 221 | 241 | 33.8% | 378 |
| 1024 | 4,096 | 386 | 2 | 0.60 | 0.75 | 0 | 1 | 2 | 0.3% | 7 |
| 1024 | 32,768 | 422 | 2 | 0.52 | 0.65 | 5 | 15 | 20 | 3.4% | 34 |
| 1024 | 65,536 | 463 | 2 | 0.51 | 0.63 | 11 | 30 | 40 | 6.5% | 64 |
| 1024 | 120,832 | 532 | 2 | 0.50 | 0.63 | 22 | 56 | 75 | 10.5% | 115 |
| 1024 | 163,840 | 586 | 2 | 0.50 | 0.63 | 31 | 76 | 102 | 13.0% | 154 |
| 1024 | 253,952 | 699 | 2 | 0.50 | 0.63 | 48 | 119 | 158 | 17.0% | 238 |

Over the production window (369 of its 1,022 bucketed prompts are in a bucket the lever takes: 512 x 126, 1,024 x 243; every tail pass at the mean context 57,000 tokens: the 59k mean prompt less 2k):

| level | seconds saved in the window | share of today's route time (1336 s) | share after exact resume (815 s) | share of the window's wall time (6668 s) |
|---|---:|---:|---:|---:|
| low | 7.2 | 0.54% | 0.88% | 0.11% |
| central | 12.3 | 0.92% | 1.51% | 0.18% |
| high | 15.1 | 1.13% | 1.86% | 0.23% |

Sensitivity to the mean tail context (central level; the older row prices every bucket below 2,048 rows, 496 prompts):

| mean tail context | saved, this lever (s) | share of today's route time | share after exact resume | the older row, same context (s) | share of today's route time |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 6.9 | 0.52% | 0.85% | 21.8 | 1.63% |
| 57,000 | 12.3 | 0.92% | 1.51% | 36.0 | 2.69% |
| 80,000 | 17.5 | 1.31% | 2.15% | 49.6 | 3.72% |
| 120,832 | 26.7 | 2.00% | 3.28% | 74.0 | 5.54% |
| 160,000 | 35.5 | 2.66% | 4.36% | 97.3 | 7.28% |
<!-- ksplit-estimate:end -->

**Reading.** Per pass the lever is worth 2 ms at 4k (nothing), 15-27 ms at 32k, and at 120k 56 ms (1,024 rows) or 104 ms (512 rows) at the central level, 10.5 or 21.6 percent of the pass. Over the window the
central figure is 12.3 seconds = 0.92 percent of today's route time (0.18 percent of the window's wall time), 1.5 percent after exact resume P0 (which removes 39 percent of the route time and so makes every
remaining second weigh more). Doubling the mean tail context doubles it; **the number that decides is the real per-pass context list of the route lines, which `population_saving_s(level, passes=...)` takes**.

**Against `docs/tp4-prefill-2.md` section 4 ("4.6 percent today, 7.6 after P0").** That row is `prefill_cost_model.split_k_saving_ms` (printed here as "older row"): the SDPA of a 2,048-row chunk at the
context, wall-scaled, times 1 - 1/(1 + 0.8 (S - 1)), for **every** bucket below 2,048 rows. At the same context this document's central estimate is about half of it per pass (0.44-0.57 over the table; column "older row"), because (i)
the tail's measured attention slope is 1.25-1.27 ms per 1k tokens, not the chunk's 2.108 (a 512-row pass at 120k has 154 ms of context-dependent time, the older row prices 260 ms), (ii) the clock penalty, and (iii) the
older row also counts the 128- and 256-row buckets (127 prompts) that the graft's chain-qualified rows exclude. Even priced the older way, the sensitivity table needs a mean tail context
of about 100k to reach the 4.6 percent. **If the owner weighs the waiver against this lever's worth, the corrected figure is about 1 percent of route time**, and widening the qualified rows (not
built; the 256-row bucket would add 66 prompts, but its fit is unreliable) is the only way to add to it.

**What moves the number.** The in-situ clock penalty (LOW to HIGH is 7.2 to 15.1 s; card M's timing pass, G3, measures the isolated ratio and the four-card pair, G5, the in-situ one); the hop cost (HOP_US;
tiny); the real context list; whether the lever composes with traced tails and exact resume (those save host dispatch and the first chunk, this saves attention device time at the tail: additive in
principle, to be confirmed in the combined window; the programme's rule is that overlapping levers never simply add).

## 4. The numerics waiver decision card (d)

**This is the text that would be approved or refused. The owner decides; nothing in the repository treats it as approved** (`serving_c2_contract.ksplit_problems` refuses the switch on a traffic profile
unless the owner's traffic waiver names it; the three profiles that carry it are gate-only).

### 4.1 What would be waived

> **W1.** For every prefill pass of 512 or 1,024 rows (the masked final pass of a prompt) on a profile with `QWEN_FAST_SDPA_PF_KSPLIT=1`, the output of each of the 16 attention layers' SDPA may differ
> from the current (oneq, which is byte-equal to stock) output in the last bits of bf16. The difference is bounded by the statistic and the bound of 4.2, not by a bit comparison.
>
> **W2.** Everything computed from those hidden states in that pass may therefore differ in the last bits: the K/V rows the later layers write for the tail tokens, the recurrent (GDN) state and conv
> state handed to the decode slot, the drafter's tap features, and the logits of the first generated token. SDPA itself writes nothing to the K/V cache, but the cache of the tail tokens is not
> "unaffected": it inherits the differences of the layers above. Greedy text can differ from the flag-off path at near-tie positions; neither is the exact fp32 answer (4.3).
>
> **W3.** The difference exists on every pass that carries the flag, including a pass whose chunk has less than 1k of context and so a single active slice (the statistics are fp32 under the lever).
>
> **W4.** The effect lasts for the decode of that request. By the resume rules of `docs/tp4-exact-resume.md` / `tp4-exact-end.md` a returning turn resumes at a chunk boundary and recomputes the tail, so
> the tail's K/V and states are not reused by a later turn (**to be confirmed with the prefix-tier owner before approval**).

**Not waived:** the 2,048-row passes and every chunk pass of any prompt (bitwise oneq, at every context); the 128-, 256- and every other bucket (no 0x10 in their program word, bitwise); the whole non-tail path;
GDN, MLP, norms, sampling, the verify rule; the prefix-cache keys and the checkpoint contents; the flag-off path, which is **bitwise the s5 path** (the oneq program word without 0x10; the graft's guarded
edits are proven inert by the harness's `--anchor` oneq-equals-stock check on the new binary, G3).

**Rollback:** unset the flag (or `=0`): every call is the oneq call again, bitwise the s5 path. The graft is a separate directory and a new image tag; the shipped image is not touched.

### 4.2 The statistic and the bound

**Statistic** (`ksplit_numerics.error_stats`): the error of an element in units of the bf16 ulp of **its row's scale**: for a row (one head, one query row, 256 columns) with R = max |reference|, the unit is
`u = 2^(floor(log2 R) - 7)`. 1.0 is one bf16 step of the row's largest element; one rounding of any element is at most 0.5. Per call the distribution of |candidate - reference| / u: max, rms, 99.9th percentile,
share above 1 and 2, the worst row.

**Where the difference comes from (A-priori).** (1) each slice's normalised output is rounded to bf16 once (at most 0.5 u; entering the result weighted by its softmax mass, so these sum to at most 0.5 u);
(2) each merge costs up to three bf16 roundings of intermediates: `d = prev - cur` (|d| up to 2R: at most 1 u), `t = sigma d` (at most 1 u) and `out = prev - t` (at most 0.5 u); (3) the weight
`sigma(cur_lse - prev_lse)` is computed from fp32 lse, error ~2^-12 of |d|, negligible. Worst case, all roundings aligned: S = 2 about 3 u; S = 4 (three chained merges, the running result re-rounded at
each hop) about 8 u, which is where the provisional max bound of 8 comes from. Typical (random signs): rms about 0.3 u for S = 2 at 1k of context (the emulation below: 0.32-0.34 rms, 4-5 u max).
**Against that sit the sequential call's own roundings**, which dominate at any longer context (next).

### 4.3 The CPU emulation changes the picture (E, not the card)

`ksplit_emulation.py` runs both arithmetics on random K/V (128 rows, head dim 256, softmax scale 1/16, three seeds, q scale 1 = flat softmax, 4 = peaky), rounding to bf16 where the served factory stores bf16
and keeping fp32 where it does (A: accumulator bf16, running max and correction factor bf16 in the sequential call and fp32 under the lever, sum fp32, P fp32, exact exp, one accumulator rounding per block;
card M may break any of these). Errors in units of u (max / rms / share above 2 u):

| blocks (context) | S (active) | q scale | ksplit vs oneq | ksplit vs float | oneq vs float | rms ratio ksplit / oneq vs float |
|---:|---:|---:|---|---|---|---:|
| 8 (1k) | 2 (2) | 1.0 | 5.0 / 0.32 / 0.000 | 2.9 / 0.24 | 4.0 / 0.22 | 1.10 |
| 8 (1k) | 2 (2) | 4.0 | 4.0 / 0.34 / 0.000 | 2.8 / 0.26 | 2.7 / 0.24 | 1.06 |
| 64 (8k) | 2 (2) | 1.0 | 8.0 / 0.77 / 0.015 | 4.5 / 0.52 | 6.6 / 0.66 | 0.78 |
| 64 (8k) | 4 (4) | 4.0 | 7.0 / 0.61 / 0.007 | 4.3 / 0.37 | 6.7 / 0.54 | 0.68 |
| 256 (32k) | 2 (2) | 1.0 | 13.0 / 1.43 / 0.107 | 8.4 / 0.98 | 12.2 / 1.33 | 0.74 |
| 256 (32k) | 4 (4) | 4.0 | 11.0 / 0.91 / 0.032 | 5.3 / 0.52 | 12.5 / 0.88 | 0.59 |
| 1024 (131k) | 2 (2) | 1.0 | 28.0 / 2.76 / 0.325 | 13.2 / 1.86 | 25.6 / 2.63 | 0.71 |
| 1024 (131k) | 4 (4) | 4.0 | 12.0 / 1.35 / 0.095 | 7.7 / 0.77 | 12.4 / 1.32 | 0.58 |

(The full table, 16 rows, is `python3 -B ksplit_emulation.py --blocks 8 64 256 1024 --slices 2 4 --seeds 3`; with 1k of context S = 4 has two active slices, the 4-block minimum.)

Findings, all **E**: (i) **the sequential call's error against the float reference grows with the number of K blocks** (rms 0.22 u at 1k, 0.66 at 8k, 1.33 at 32k, 2.63 at 131k for a flat softmax; about half that at 131k for
a peaky one), because its accumulator is rounded to bf16 at every block; (ii) the split's error is smaller (fewer sequential steps per slice) and the ratio improves with the context (1.06-1.10 at 1k, 0.5-0.8 from
8k); (iii) so the difference |ksplit - oneq| is mostly the *oneq call's* noise and grows the same way: the absolute bound of the first draft (max 8, rms 1.5, share above 2 u 2 percent) would have rejected a
**correct** split at 8k and beyond. `bound_for(context)` therefore multiplies max and rms by sqrt(blocks / 16) from 2,048 tokens up (x8 at 131k) and the share above 2 u by blocks / 16 up to 0.5; (iv) a
**wrong merge is far outside it**: a lost slice gives rms 68 u, a wrong-way weight 10-50 u on peaky rows. **A flat softmax cannot see a wrong weight** (equal-mass slices give sigma = 0.5 either way), so
the matrix carries the peaky variant, which is what makes the harness sharp.

**Provisional bound** (`BOUND`): max 8, rms 1.5, share above 2 u 0.02 at up to 2,048 tokens, scaled by `bound_for(context)`; and the second half of the rule, `NOISE`: against the float reference the split
may be no further than 1.25x rms and 1.5x max the oneq call's own distance. These are a tripwire at a gate, **not a measurement**.

### 4.4 The numbers that must be measured on card M before approval

All from `ksplit_card_m.py` (G2-G3), in the harness's `numerics` section, over configs s2 (1,024 rows, S = 2), s4 (512 rows, S = 4), s2g (512 rows on an 8x8 grid, S = 2), both Q memories, seeds x {normal,
peaky}, 15 chunk starts from 0 to 251,904:

1. **|ksplit - oneq| per context** (max, rms, p99.9, share above 1 and 2, worst-row p99): replaces `BOUND` by the measured worst case with a margin. Decides the audit's bound.
2. **oneq vs float and ksplit vs float up to 32k** (and one point at 64k and 131k with `--reference-max-start`): the noise floor and the ratio against `NOISE`. **This is the number that decides the question the
   emulation raised**: if card M's oneq-vs-float growth is far smaller than the emulation's, the accumulator is not bf16 and the first-draft bound was right; if it matches, the split is the more accurate
   kernel and the waiver is mostly about *different* bits, not worse ones.
3. **Non-finite count** (must be 0 over the whole matrix) and the oneq-equals-stock anchor on the new binary (must hold: it proves the guarded edits inert).
4. The refusals, the program-cache keying and the 11x10 grid pass (section 5).

### 4.5 How the decision is judged

1. **Per-call statistics (G2-G3):** every case inside `bound_for(context)` after item 1 above has replaced the provisional numbers, and ksplit-vs-float inside `NOISE` of oneq-vs-float wherever the reference exists.
2. **Audited attach (G6, `...-pf3k-s5-audit`):** the first 8 split calls and every 24th afterwards also run the oneq program on the same inputs and judge the pair by the same statistic at the call's own
   context (`[PINDIAG] sdpa prefill ksplit audit n=.. rows=.. context=.. ... within_bound=True`); not a bitwise audit; a call outside raises. Gate only: it doubles those calls.
3. **Real text (G7, not sized here):** greedy continuations on code corpora at 4k, 32k and 128k and on the agent replay, flag on against flag off. Token identity is not expected; the judge is the divergence rate and
   the first-token logit difference against a calibrant of the same kind that already exists, the 128-aligned-resume arm of the split-invariance probe (`docs/tp4-exact-end.md`), which also changes the bits;
   plus no regression in the replay's task outcomes. The thresholds are the owner's to set once the calibrant's numbers are in.
4. **Timing (G3, G5):** KS-WIN needs the median ksplit/oneq layer ratio at 32k and beyond at most 0.75 (KS-PARTIAL at most 0.95); a loss of more than 5 percent at 4k and below fails the lever; the
   four-card pair must show the tail passes' saving near section 3's central figure.

**Abort criteria:** any non-finite value; any case or audited call outside the bound; the noise rule failing at any context; a watcher or NoC-sanitiser report; a hang that is not the planted one; the oneq
anchor failing; KS-NO-WIN (median ratio above 0.95) at 32k and beyond; a loss above 5 percent at 4k and below; the real-text divergence above the calibrant's.

## 5. Card gates (e)

All minutes **E** except where an existing harness has a number. The build and G2-G4 use card M under `qual_card.sh` (card B refused; M needs `ALLOW_SERVING_CARD` and a look at the exclusive
group); a hang resets the target card only (M and A together by the hint the runner prints).

| Gate | What | Minutes | Pass when |
|---|---|---:|---|
| G0 | CPU, done here | n/a | all tests of section 6, every generator `--check` |
| G1 | `build_k64j_oq_ks.sh` dry run (`KS_BUILD_DRY_RUN=1`), then the build: K64j-OQ-KS beside K64j-OQ, three ninja runs of 30-60 s, six publish checks | ~10, no device | the checks pass and the `K64J_OQ_KS_TTNNCPP_SHA256` line is printed. **The first compile of the factory C++** (descriptor plumbing) |
| G2 | `WATCHER=1 bash run_card_m_ks.sh`: `TT_METAL_WATCHER=5` (NoC sanitiser, waypoints), 1,024 and 512 rows, five starts, one seed, no timing | ~40 | no watcher assert, no sanitiser report, no non-finite, the statistics inside the provisional bound. **The first JIT compile of the three kernels** |
| G3 | `bash run_card_m_ks.sh`: statistics (4.4), the 11 refusals (`TT_FATAL`, never a hang), the cap check (256 rows logs exactly 4 slices), program cache, the 11x10 grid, timing at 8 contexts (0 to 251,904) | ~75 | all `problems` empty; refusals matched; the verdict KS-WIN or KS-PARTIAL; the oneq anchor equals stock; the measured bound written back into `ksplit_numerics.py` |
| G4 | `bash run_card_m_ks.sh hang`, LAST in its session: flag 0x41B with `QWEN_SDPA_PF_TEST=1` | ~15 (E) plus the reset | the watchdog fires on the planted call (QKDC waypoint with the watcher) |
| G5 | four-card timed pair `...-pf3k-s5-ctl` vs `...-pf3k-s5` (OQ-T style: solo TTFT at 131k and 32k, tail passes) | 25 each | the tail passes' saving near the central estimate; nothing else moved |
| G6 | `...-pf3k-s5-audit` attach | ~25 | audit lines, `within_bound=True`, no `fell back` |
| G7 | real-text exactness and agent replay, flag on vs off | not sized | section 4.5 item 3 |

Total on hardware to a verdict: about 2.2 hours on card M (G2-G4), then about 1.25 hours on four cards (G5-G6), without G7 (E).

## 6. Tests and checks (CPU)

New: `test_ksplit_planner` (18), `test_ksplit_factory` (30), `test_ksplit_sources` (22), `test_ksplit_protocol` (15), `test_ksplit_card` (26, needs torch), `test_ksplit_build` (11), `test_ksplit_emulation`
(8, needs torch), `test_ksplit_estimate`, `scripts/ci/test_sdpa_ksplit_tp` (51 with the audit and bound tests), `scripts/ci/test_pf3k_profiles` (18). The op directory is discovered by the CI step added to
`qwen-integration-cpu.yml`; the two `scripts/ci` modules are named in it. Existing tests touched: the profile-census tests (23 modules) learned the three twins with exact-count assertions (no loosening);
`test_tp4_closure_literals` classifies `sdpa_pf_ksplit_tp` and `ksplit_numerics` as served; `test_serving_image_copy_closure` the new plain import. All generators: `make_pf3k_profiles.py --check` and every other
`make_*profiles*.py --check` pass; the new profiles are generated, never hand-edited.

Reproduce: `python3 -B optimisation/ttnn-op/sdpa_prefill_ksplit/ksplit_estimate.py --doc docs/tp4-prefill3-ksplit.md` (the tables above are its output); `python3 -B scripts/ci/make_pf3k_profiles.py --check`.

## 7. Risks (f)

1. **The compute kernel is uncompiled** (a 60-parameter template instantiation on the pinned header): the first compile is G2, not G1. Probable failure class: a parameter the text anchors cannot see.
2. **Descriptor plumbing assumed** (runtime-arg lists, semaphore descriptors): a G1 compile error, cheap to fix.
3. **The performance premise.** The in-situ clock penalty may be worse than LOW (a 96-core split pass at the chunk's per-block time is the LOW bound; a worse draw is possible), and the 18 us hop is an estimate. The
   worth is about 1 percent of route time; a card result at the LOW end is 0.5 percent. This is the reason to run G2-G3 before spending a four-card window.
4. **The accumulator format.** If card M shows oneq is far less noisy than the emulation (an fp32 accumulator), the split is not the more accurate kernel and the waiver's honest description becomes "different
   by about 1 u", with the first-draft bound valid.
5. **Coverage.** Only 369 of 1,022 prompts' tails (36 percent) are in a taken bucket; 526 prompts (51 percent) end in a 2,048-row pass the lever cannot touch.
6. **Sharing the grid with other levers.** The 11x10 grid works in the planner and the harness grid pass; traced tails (valid_len as a device scalar) capture the program configuration inside a trace, so
   the 0x10 word must be in the captured configuration; exact resume changes the contexts the tails see but not their rows.
7. **Merge risk.** The lever touches shared files by appending only (section 8), but `qwen_c2_profiles.json` is generated: after merging with tp4/next-2, tp4/upload-1 and tp4/sched-2, rerun the generators in
   the order the headers say (fusion, octo3, ship4, s5, then this) and the census tests' exact counts will need the other branches' twins added.

## 8. Merge reconciliation notes

Shared files this branch edits, all by appending a block or a name: `.github/workflows/qwen-integration-cpu.yml` (one `# PF3K` block with two commands), `docker/qwen-c2-overlay.txt` (two lines),
`scripts/ci/serving_c2_contract.py` (`KSPLIT_*` constants, `ksplit_problems`, one call in `boot()`), `scripts/ci/tp_addresses.py` (one block before `return rebound`), `scripts/ci/profile_twins.py` (one import,
one tuple), `scripts/ci/c2_smoke_check.py` (one row in `FUSION_LEVERS`, one name in `FUSION_RULES`; written by `make_fusion_profiles.py --write` from `fusion-wp/PF3K.json`),
`scripts/ci/qwen_c2_profiles.json` (three generated twins after the last ship-5 twin), and the census tests listed in section 6 (the exact-count assertions: on a merge, recount, do not loosen).
Not edited: `levern_route.py`, `prefill_tail_trace.py`, `make_pf3_profiles.py`, `test_prefill_tail_trace.py`, `docs/tp4-prefill3.md`.
