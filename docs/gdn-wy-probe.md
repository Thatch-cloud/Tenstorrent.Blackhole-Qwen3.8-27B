# GDN window WY probe: a non-exact research track (branch tp4/gdn-wy-probe)

Status: **research only, non-exact, never in a serving or traffic profile.** The CPU numerics (E1) and the one-card harness (E2) are built and CPU-tested. **The Tenstorrent kernel is at zero**: no generator, circular-buffer plan in code, compile arguments or C++ source exists, only the CPU model that is its specification and the interface the harness expects. Nothing here has run on a card. The harness's window arms run only on a fake runtime (with a fake kernel that computes with the CPU model) until `scripts/ci/gdn_wy_block.py` is written (section 6).

Labels: (m) measured, (src) read from the code, (cm) computed from measured numbers, (e) estimate, (U) unknown.

## 0. Design (written first)

**The question.** The verify step's GDN recurrence runs a 16-token dependent chain per (user, value head): K5-A, one core per pair, 193.4 us a layer (m, `gdn_seq_block_split.py` header), 9.29 ms a four-user block over 48 layers (cm), about 18 ms of a 175 ms eight-seat round at 32k context (m, round budget). The chunked WY / UT block form of the same gated delta rule replaces the chain by a handful of tile matmuls plus a 16 x 16 triangular inverse. Verifying a 32-row block as **two aligned 16-row windows** (the second window starts from the first window's committed bf16 state) would cut the per-layer cost if the two windows fit in about 90 us (the kill line), against about 387 us for two sequential K5-A launches (cm). It also moves 4.6x fewer bytes and drops the 16 per-token state snapshots (cm, section 5).

**Why it is a side track.** The window form computes a different arithmetic. Its outputs are not the sequential recurrence's bytes (section 2), so it cannot run under the byte-identity contract that every serving arm is held to. The owner approved it as research only: measure whether it is worth a decision, and say plainly what the decision would cost.

**Three deliverables, three honest limits.**

| Deliverable | What it is | What it is not |
|---|---|---|
| E1 numerics (`scripts/ci/gdn_wy_numerics.py`, workflow `qwen-gdn-wy-numerics`) | CPU: window form vs the served chain vs fp64, per-element error, 1,000-cycle drift with committed lengths from the tau distribution, causality, packed equals solo, segmentation. Machine-readable JSON. | Real layer 0 / 23 / 47 activations: the weights are not in CI. Synthetic regimes unless `--fixture` files are given (section 3). Not a Tensix emulation. |
| E2 card harness (`optimisation/ttnn-op/wy_probe/gdn_wy_card_m.py`, `run_card_m.sh`) | One card, one layer, four users at the TP4 geometry. Timing of A, A2 (two K5-A launches), W1, W2; causality as raw bytes; packing; SRAM. Kill rules, a correctness gate, exit codes, partial reports. Its baseline half and every window path are CPU-tested on a fake runtime; none has run on a card. | A run of the window kernel: there is none. |
| This document | The non-exactness argument, the kill rules, what remains. | A recommendation to build it. |

**What was built, in order.** (1) `gdn_wy_model.py`: the two forms, the rounding models, the cost and SRAM plans, one place. (2) `gdn_wy_numerics.py` on top of it. (3) The card harness and its launcher, patterned on the V5 byte gate (`optimisation/ttnn-op/v5split/gdn_v5_card_m.py`) and importing its input builders and watchdog. (4) The job templates for the combined card window. (5) This document.

## 1. The two forms

Per user, per value head, `S` is a `[dk, dv]` = `[128, 128]` state. Tokens `t = 1..T`, with `k~, q~` the L2-normalised key and query, `beta`, the log decay `g` and `alpha = exp(g)`.

**Served (sequential, K5-A).** Per token: `S <- alpha S`; `r = k~^T S`; `d = beta (v - r)`; `S <- S + k~ d^T`; `o = q~^T S`. The state is rounded to bf16 after every token and each token's snapshot is written; the commit copies snapshot `n` (src `scripts/ci/gdn_seq_block_device_test.py`, the reference).

**Window WY (`gdn_wy_model.wy_window`, `wy_commit`, `wy_verify`).** One window of `W = 16` rows at a time. With `G = cumsum(g)`, `Gamma[t, i] = exp(G_t - G_i)` for `i <= t` (log space), `A = (I + tril_strict(diag(beta) (K K^T o Gamma)))^-1`, `u = A (beta V)`, `w = A (beta gamma K)`:

- pseudo-values `V~ = u - w S0`;
- outputs `O = (gamma Q) S0 + tril(Q K^T o Gamma) V~`;
- state after `n` rows `S_n = gamma_n S0 + sum_{i<=n} exp(G_n - G_i) k_i v~_i^T`, rounded to bf16 **once**, at the window boundary or at the commit.

A 32-row block is two chained windows. Every window computes all 16 of its rows (the fixed shape of a launch); only the committed `n` rows are used. The literal TreeWY ratio `g_t / g_i` is kept only as `ratio=True`: under strong decay (regime R2, cumulative log decay about -180 over 16 tokens) it is 0/0 and every output is non-finite (m, CPU, `single_round.R2.*.ratio_form`). **A TT kernel must use the log-space form.**

## 2. Why it cannot serve: it is not the served bytes

Whatever the numbers in section 4 say, the window form fails the byte-identity contract for three separate reasons, each reproduced by the numerics report:

1. **Different arithmetic.** The served chain rounds the state to bf16 after every token and reads out from the rounded state; the window form keeps fp32 within a window and rounds once. About half of the bf16 gated outputs differ over a drift run (`acceptance.contract.gated_bf16_differing_fraction_wybf_vs_seqbf`, section 4). In fp64 the two forms agree to about 1e-15 (m, CPU), so the difference is rounding structure, not algebra.
2. **Segmentation dependence.** The sequential form is bit-for-bit independent of how a committed stream is split into rounds (m, CPU, the earlier research run: 0 of 16.5 M outputs differ). The window form is not: a different split changes which rows share a window, so changing the proposal count, an acceptance boundary, a lane's cap or a lifecycle event changes the bytes (`segmentation.*.wy_bf16.outputs_bitwise_identical`). Several serving gates rely on those knobs being arithmetic-neutral (`real_text_compare.py`); under the window form they are not.
3. **Precedent.** A last-bit difference already broke exactness once at four cards: a four-way reduce-scatter summed partials in a different order, "differ in the last bit only; that flips a near-tie", and two users diverged in their first verify rounds (`docs/tp4-exact-ring-parity.md`). The window form's differences are orders of magnitude larger than a last bit.

It is the same conclusion the repo already recorded: "WY-form GDN is faster but not token-exact, which is a user decision" (`docs/four-streams-131k-feasibility-2026-09-23.md`). This probe prices that decision; it does not make it.

## 3. E1: the CPU numerics

`scripts/ci/gdn_wy_numerics.py`, run by the workflow `qwen-gdn-wy-numerics` (push a tag `experiment/gdn-wy-numerics-vN`; eight cells: model / R1 / R2 at 16 rows, model at 32 rows as two chained 16-row windows, R1 and model at 32 rows with a long-acceptance commit distribution, and the plan's single W = 32 window for model and for R1 with the long-acceptance distribution). A laptop may flip bits at rest, so a local run is a smoke test only (`--smoke`, seconds) and the report from CI is the result.

- **Geometry.** The TP4 per-card shard: 4 key heads, 12 value heads, `dk = dv = 128`, 8 users, T16 rows, 48 GDN layers.
- **Rounding models** (labels in every report): `fp64` (truth), `fp32`, `bf16` (the served class: bf16 inputs, fp32 arithmetic, bf16 state), `bf16+tf32` (operands also rounded to TF32, the Tensix srcA/srcB width, estimated), `bf16mid` (window form with every intermediate stored as bf16, as bf16 circular buffers would hold it). CPU models of the device class, not a bit-faithful emulation.
- **Inputs.** Real layer 0 / 23 / 47 inputs are **not** obtainable in CI (the weights are private and large). `--fixture DIR` reads `layer<N>.npz` (`q, k, v, z, beta, g`, optional `S0`, `norm_w`; float32, bf16-valued, shape `(users, rows, heads, dim)`) as a dev card or the V5 harness's real-text path (R5) could dump them, and the regime `fixture:layer0` then draws from the file. Without fixtures the regimes are synthetic: `model` (the Qwen3-Next parameterisation shapes with the per-head decay scale spread over three decades), `R1`, `R2` (the regimes of `gdn_seq_block_device_test.py`). **The numbers in a synthetic report say the arithmetic's behaviour, not the model's.**
- **Window size.** `--window 16` (default) runs a 32-row block as two chained 16-row windows, the form the card harness times (two windows in one launch); `--window 32` runs it as the single W = 32 window of the approved plan's E1. Both are in the matrix.
- **Committed lengths.** The lab's per-round histogram is private; the public figure is the pooled tau 4.485 (`docs/tp4-lookup.md`). Drift draws each round's committed `n` from the truncated geometric distribution fitted to that tau (`commit_dist = tau`), or from a supplied histogram (`--commit-hist`), or from `p = 0.97` (`stress`) so that the second window of a 32-row block is exercised.
- **Sections.** `crosscheck` (this module's fp64 sequential form against the repo's own reference function, read as text: zero differing elements); `single` (one round, every model, error per element against fp64 and against the served form, by row, plus the T = 1 decode step and the ratio-form failure); `causality` (rows `0..t` and the committed state **bit for bit** with later rows real / zero / other, and the NaN and Inf poison leak; at 32 rows also n = 17, 24 and 31, which chain the second window through the first's committed bf16 state); `packed_solo` (a batch of users against each user alone, a different committed length per user, at 32 rows lengths past the first window too, and the batch reversed); `drift` (at least 1,000 accept / reject cycles; each form carries its own state; the state error is computed at **every** cycle and the full comparison recorded at checkpoints 1, 2, 5, 10, ... 1,000; the rejected draft rows are fresh random draws, or, from a fixture, the second half of the file, disjoint from the stream in the first half; a fixture too short for the run is refused rather than wrapped, and the report counts the rows each lane drew and wrapped); `segmentation`; `counts` (FLOPs, bytes, 32 x 32 tile operations, the SRAM and DRAM plans).
- **Report format.** One JSON object, `kind: gdn-wy-numerics`, `schema: 1`: `geometry`, `inputs`, `repo_reference_crosscheck`, `single_round`, `causality`, `packed_solo`, `drift`, `segmentation`, `counts`, `acceptance`, `seconds`. The report is strict JSON (non-finite numbers are `null`). The last stdout lines are `GDN_WY_NUMERICS e1=PASS|FAIL ...` and a one-line JSON. Exit 0 when E1 holds, 1 when it does not (a result), 2 on a usage error, 3 when the run crashed (the partial report then has no `acceptance` and `completed` is false). The workflow requires the result line and `acceptance`, so a crash fails the job, and it annotates an `e1=FAIL` result as a warning.
- **The E1 criterion** (the plan's, `acceptance.e1_*`): the window form's error against fp64 is no worse than the served chain's, in the `bf16+tf32` class (TF32 operands, fp32 accumulation, bf16 state at the window boundaries), for the state and the output (max over every cycle of the run), in every drift case, and nothing is non-finite. The bf16-only class (no TF32 rounding of operands) is reported as a second, ungated row (`acceptance.e1_bf16_class`). It is a research criterion: `acceptance.contract.byte_identical_to_served` is `false` whenever any byte differs, `serving_eligible` is always `false`.

**What the CPU numerics can show about causality.** Rows `0..t` of the window form are bit-identical when the later rows are real, zero or other finite data (the triangular masks multiply them by exact zeros, and adding exact zeros changes no partial sum), but **NaN or Inf in a later row leaks into the committed rows** (0 times NaN is NaN in the `A V` and `QK^T` products). A kernel that reads padded or stale tile rows must keep them finite. The harness's `nan` variant is informational for that reason; the three finite variants are gated.

## 4. Results (CI run of tag `experiment/gdn-wy-numerics-v1`, 8 users, 1,000 cycles per cell, synthetic regimes)

Max over the run of the error against fp64 (state, then gated-output core), bf16 class; the last column is the fraction of the window form's gated bf16 outputs that differ from the served chain's. All values finite, `repo_reference_crosscheck` zero differing elements in every cell (m, CPU, synthetic inputs).

| Cell | Served chain vs fp64 (state / output) | Window form vs fp64 (state / output) | E1 | WY vs served, gated bytes differing |
|---|---|---|---|---:|
| model, 16 rows, tau | 0.053 / 0.024 | 0.024 / 0.012 | pass | 62% |
| R1, 16 rows, tau | 0.00500 / 0.00464 | 0.00504 / 0.00330 | **fail by 1.0% on the state** (ratio 1.01) | 51% |
| R2, 16 rows, tau | 0.0033 / 0.0050 | 0.0032 / 0.0023 | pass | 43% |
| model, 32 rows, tau | 0.048 / 0.026 | 0.022 / 0.013 | pass | 62% |
| model, 32 rows, long acceptance (two windows in 60% of rounds) | 0.053 / 0.028 | 0.014 / 0.0076 | pass | 62% |
| R1, 32 rows, long acceptance | 0.0069 / 0.0048 | 0.0038 / 0.0023 | pass | 51% |

- **E1 (error no worse than the served chain's): five of six cells pass, R1 at 16 rows misses by one percent on one metric** while its output error is 29% lower. The criterion is strict and was not loosened after the fact; read it as a tie on the state and a win on the output there. The window form is as good as or better than the chain in every cell, because it rounds the state once per window and the chain sixteen times. That is the form's one numerical advantage and it is not a reason to serve it.
- **Byte identity: not met anywhere** (43% to 62% of gated bf16 outputs differ). `contract.byte_identical_to_served` is false in every cell, as section 2 requires.
- **Segmentation: the served chain is bitwise invariant in every cell (fp32 and bf16, all common commit points); the window form is not** (0 of 68 common commit points identical at 16 rows in model and R1; 10 to 21 of 68 in R2, where strong decay makes later rows matter little).
- **Causality (CPU model):** rows `0..t` and the committed state are bit-identical across real, zero and other later rows for both forms; with a shorter block (absent rows) the window form is *not* bit-identical (different matmul shapes: a CPU-library effect, reported, not gated); NaN or Inf in later rows leaks into the committed rows for the window form only (section 3).
- **Packed equals solo (CPU model):** true for all forms in every cell. A CPU matmul library chose the same blocking for 8 users and 1; this says nothing about the kernel. The card harness checks the kernel's own bytes.
- **Drift:** no divergence over 1,000 cycles; the errors above are the run maxima and the per-checkpoint series are in each report (`drift.*.pairs.*.state`).
- **Not measured:** real layer 0 / 23 / 47 activations (section 3), the device-class numerics of a real kernel, and anything on a card.

These are CPU-model statements about a form that does not exist on the device. The reports are the artifacts of the run named above; section 6 says what remains.

## 5. Cost model (cm)

Per (user, value head, round) at T = 16 (`counts`), bf16 storage:

| | Sequential (served) | Window WY |
|---|---:|---:|
| FLOPs | 1.84 M | 1.57 M at n = 4.485, 1.96 M at n = 16 (0.86x to 1.06x) |
| Bytes | 643 kB (524 kB of per-token snapshots) | 139 kB (86 kB if the commit is fused into the next verify) |
| 32 x 32 tile operations | 124 per token x 16 = 1,984 | about 144 (e; T = 16 padded to one tile row, excludes reader, writer and pack overhead) |
| Per card per layer per round (96 pairs) | 176.6 MFLOP, 61.7 MB | 151.2 MFLOP, 13.4 MB |
| Per round, 48 layers | 2.96 GB (2.25 GiB of snapshots) | 0.64 GB |

K5-A's own measurement says snapshots are about 1.4% of the launch (214.7 us with split-NoC snapshot writes vs 211.8 us without, TP2 shape, m). What the window form removes is not the snapshots: it is the 16-step dependent chain. The upside is capped at the recurrence bucket, about 18 ms of a 175 ms round at 32k (10%, m); its larger benefit is memory (about 1.2 GB of snapshot history per block per chip). At the kill line itself (two windows in 90 us, so one window in about 45 us) a T16 verify would save about 150 us a layer, 7.1 ms a four-user block and 14 ms a two-block round (e): a ceiling that assumes the kernel reaches the kill line and nothing else in the round moves, bought by giving up byte identity. That is a decision for the owner, not an engineering default. The SRAM plan (`gdn_wy_model.sram_plan`) is 196,608 B for one window and 229,376 B for two chained windows, against K5-A's 630,784 B (src, `docs/tp4-recurrence-split.md`) and 1,572,864 B of L1 (plan, not a measurement).

## 6. E2: the card probe, and what remains

**The harness** (`optimisation/ttnn-op/wy_probe/`): `gdn_wy_card_m.py` and `run_card_m.sh`, one card, a 1x1 mesh, four users x 12 value heads, no model, no collective, `QWEN_FAST_TP=4` and `QWEN_FAST_VERIFY_T1=1` so the K5-A control is the coalesced build the model runs. The launcher embeds the canonical card-selection block, refuses the reserved board and (without the override) a serving card, resolves the board at run time and, on a hang, writes the reset hint to a file in the results directory. The public log carries only the card tag and the image tag: not the board id, node, PCI address, results path or resolved image reference (the shared library's lines that name them are sent to /dev/null here; its refusals stay on stderr). Exit codes: 0 CONTINUE, **10 KILL** (1 is what the launcher's refusals and an import-time crash return, so it is no longer a finding), 2 usage, 3 watchdog, 4 NO-DECISION.

| Section | Rule |
|---|---|
| `timing` | Per layer, one trace of 48 distinct-input launches per arm, 25 serpentine rounds: A (one K5-A launch), A2 (two), W1 (one window), W2 (two windows in one launch). **KILL when W2 > 90 us a layer.** |
| `causality` | Rows `0..n-1` of the output and the committed state, as **raw bytes** (`to_torch` hides -0 and denormals), with every later row of the 32-row block real / zero / other; `n` in 1, 4, 8, 15, 16, 17, 24, 31; NaN informational. **KILL on any mismatch.** Controls: K5-A under the same variants must be causal (else a harness fault, NO-DECISION), and a changed earlier row must differ (else a blind compare, NO-DECISION). |
| `packing` | The four-user launch against four one-user launches, outputs and committed states, byte for byte. **KILL on any mismatch.** |
| `sram` | The kernel's circular-buffer bytes a core against L1 (1,572,864 B) and the CPU plan. **KILL over L1.** DRAM bytes: the planned interface has `dram_bytes(windows)` (the reader and writer totals the generator emits), reported as `dram_kernel`; the CPU plan is reported beside it labelled as a plan, never as a measurement. |
| `accuracy` | **The correctness gate.** For both windows, every regime and seed, W's output rows and each committed state against the fp64 sequential reference of the same host inputs (`gdn_wy_model`), with K5-A run twice (the second window from its own committed state) as the yardstick: **KILL when W's error against fp64 exceeds 1.5 x K5-A's** on the output or either state (E1 measured the window form at 1.01 x the served chain at worst), or on non-finite output. A fast kernel that is causal and packing-exact but computes the wrong function therefore cannot reach CONTINUE. A reference that fails (non-finite, or K5-A itself far from fp64) is NO-DECISION. The bf16 model's own error and W's difference from it, the differing fraction against K5-A, max abs and the ulp histogram of window one are reported beside it. |
| inputs | After every launch (and a sample of the timing sets after the replays) the inputs are read back and compared with what was uploaded: **a changed input page is a KILL**. |
| `selftest` | A raw page round trip of known bytes and the environment the process read. |

An unwritten page (a sentinel-filled output page the launch never wrote) or a moved input is a KILL. A reduced scope can never reach CONTINUE, and the scope accounting names the timing arms: a run without the W2 arm (`--timing-arms`) is reduced, and a kernel run whose W2 arm was not measured is NO-DECISION. The timing KILL needs the plan's timing scope (48 launches x 25 rounds): at a reduced timing scope the per-launch overhead is not amortised, so an over-90 us figure is reported and the verdict is NO-DECISION, while the causality, packing, correctness, SRAM, unwritten-page and moved-input KILLs hold at any scope. A kernel module that is present but raises on import is a recorded error and NO-DECISION, never "not built". CONTINUE means only that every kill rule held at full scope: it is the owner's decision whether to start the programme, never a serving licence.

**What was built and CPU-tested:** everything above except the kernel: the verdicts and kill rules, the later-row variants, the raw-byte prefix compare (including signed zero and NaN payload), the timing and SRAM rules, the scope accounting, argument parsing, the launcher's structure and refusals, the job templates, the **baseline flow** end to end on the page-level fake ttnn of the V5 gate's tests (no kernel: selftest, controls, A and A2 timing, NO-DECISION saying why), and the **window flow** on the same fake with K5-A and a fake `gdn_wy_block` that both compute (`gdn_wy_model`, bf16 class): a correct fake reaches CONTINUE at the full plan (4 users, both regimes, both seeds, 8 commit lengths: 219 launches, each followed by an input read-back); a fake that leaks a later row, differs packed against solo, leaves a state page unwritten, overwrites an input (in a causality launch and in a timing replay), computes zeros or a wrong decay sign (causal and exact, so only the correctness gate sees it), or times over 90 us each ends in KILL by the section that must catch it, and a timing KILL at a reduced timing scope is NO-DECISION. This tests the harness, not a kernel: the fake and the reference share the same CPU model, so a model error would pass; the card run is the first test of the kernel's arithmetic.

**What remains (all of it device work):**

1. **The kernel `scripts/ci/gdn_wy_block.py` and its C++ sources** (the planned interface is in the harness docstring; **the kernel is at zero**: not even the generator, the circular-buffer plan in code or the compile arguments exist): per-(user, head) fused window compute from copies of the prep and scan compute code, in the V5 generator style; the 16 x 16 inverse by doubling (about 11 tile operations); log-space `exp(G_t - G_i)`; two chained windows in L1; the commit with `commit_rows`; padded tile rows kept finite and zero. **Not written.** The CPU model (`gdn_wy_model.py`) is its executable specification.
2. **The device-class numerics of the real kernel** (TF32 operands, the bf16 CB rounding points) against the CPU models; the `accuracy` section gates the kernel against fp64 and K5-A, and reports its difference from the bf16 model.
3. **One card-time measurement** of the harness (jobs W1, W2 and, as filler, W0, only once the kernel exists: `scripts/ci/references/tp4-wy-probe-jobs/`, whose ORDER separates `NEEDS` (job names) from `REQUIRES-FILE` (the kernel file)). Without the kernel the probe uses no card time: the baseline half is runnable, but it re-measures what the repo already holds (K5-A is 193.4 us a window), and W2 times A and A2 in the same trace as the window arms. W2's estimate is 60 minutes against a 120-minute container timeout.
4. **If, and only if, E1, the timing and the owner's decision all say go:** the integration (a gate-only profile behind a default-off flag, a different parity contract that does not rely on segmentation invariance, a different real-text comparison for the arm). None of that is designed here.

## 7. Files

- `scripts/ci/gdn_wy_model.py`, `scripts/ci/gdn_wy_numerics.py`, tests `scripts/ci/test_gdn_wy_numerics.py`.

Open item for the owner (not changed here): the shared card-selection library, which every single-card harness embeds byte for byte, names the rig's board identifiers in tracked source. This harness keeps them out of its own log, but the systemic fix is to move the identifiers out of the tracked library into a rig-local file the library reads at run time.
- `optimisation/ttnn-op/wy_probe/gdn_wy_card_m.py`, `run_card_m.sh`, `test_gdn_wy_card_m.py`.
- `scripts/ci/references/tp4-wy-probe-jobs/` (ORDER and three templates), tested by `scripts/ci/test_gdn_wy_probe_window.py`.
- `.github/workflows/qwen-gdn-wy-numerics.yml` (tag `experiment/gdn-wy-numerics-v*`); the tests are named in `qwen-integration-cpu.yml`.
