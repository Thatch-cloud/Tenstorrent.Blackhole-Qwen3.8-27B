# Complete learned DSpark layer: simulator bring-up

**Complete layer executes, but its numerical gate fails. It is not qualified.**
This connects the previously qualified, observed feature-projection output to
every operation in a learned DSpark layer. It does not use a neuron slice or
replace device intermediates with CPU-golden values.

| Part | Complete scope |
| --- | --- |
| Parameters | All 11 pinned layer-zero tensors, BF16 |
| Attention | 32 global query / 8 KV heads, split 16 / 4 per chip |
| History | 32 actual feature rows plus all seven proposal keys; 64 padded keys |
| Q/K preparation | Learned projection, direct RMS weights, absolute YaRN tables |
| Attention policy | Previously qualified precise exponential and 64-key chunks |
| MLP | All 17,408 channels; 8,704 per chip, full gate/up/SiLU/product/down |
| Reductions | FP32 partial sum, explicit BF16 cast, BF16 residual addition |
| Cases | Two frozen input patterns; absolute starts 0 and 8,190 |

The CPU layer reproduces the existing upstream-checked attention residual and
layer output **bitwise at all six checkpoints**. The whole checkpoint is
reverified on close. The native port declares its FP32 normalization, rotary
and SiLU intermediates explicitly; final comparisons retain the existing
0.01 relative plus 0.01 absolute threshold.

## Execution boundary

Three phases have separate changing-input traces: attention/output projection,
MLP, and final residual. The simulator host stages the **observed** FP32 partials
between chips at the two reduction boundaries. Each sum and residual addition
then runs on device. The initial context is the hash-pinned observed output from
the successful complete FC/norm run, not the CPU reference substituted for it.

This is not a fabric test, a complete captured pipeline, all five layers, a
target verifier or a serving result. Those eligibility flags remain false.
The first eager-only diagnostic skips replay to find integration errors sooner;
it cannot qualify the complete matrix, even if its comparisons pass.

## First device result

- Eager-only simulator `20260909T173511Z-440`, code `7d892df`, completes all
  attention, MLP and residual phases. It fails 102 of 192 eager comparisons.
- All 1,154 host tests and 59 simulator-harness tests pass, plus wrapper syntax.
- All 84 input, 44 parameter and 12 padding checks pass, as do six CPU checks.
  Mesh and checkpoint close cleanly; outer exit is 1. There is no replay result.
- All three native grafted files are restored byte-identically and owned locks
  released. Both native binaries remain unchanged. Independent qualification
  rejects the failed report, as required.

Failure report SHA256:
`f81fbb9aaab4eaae9ddadaddb4bbd59f8be0ebab6769eb67f038f192232f965c`.
The observed tensors are retained locally, SHA256
`e65c254e3197d28a888ddc5be194fdccc321a9afc29dd9a08d1d2f11bff31c2d`.

## What the failure isolates

The offline attribution uses each operation's **actual device inputs** rather
than attributing every downstream error to that operation. For case 0 / chip 0:

| Comparison against its own input reference | Result |
| --- | --- |
| Q/K rotary | Bitwise exact |
| Q/K and post-attention normalization | Pass the unchanged numerical threshold |
| Attention | 15 out-of-threshold elements; not qualified |
| Attention output projection | Pass the unchanged numerical threshold |
| MLP gate/up projections | Pass the unchanged numerical threshold |
| MLP down projection | 202 out-of-threshold elements; not qualified |
| Both BF16 residual additions | Within numerical tolerance, but fail required exact bits |

The full layer output has 1.324% relative L2 error against the frozen CPU
backbone in that case; this is diagnostic, not a coding-quality score or an
alternative passing threshold. All six cases/chips remain in
`scripts/ci/dspark-layer-attribution.json`, SHA256
`ca3d46e27f300421516f4255aba6dbd3a7ff2a19eb9b80b2c5609d86429f5ec8`.

A short residual diagnostic reproduces all 12 native controls exactly. Merely
requesting an FP32 output before a BF16 cast still fails all 12 candidate exact
checks, with clean exit 1. Explicitly widening **both BF16 inputs** before the
FP32 addition and final BF16 cast fixes this on all retained operands:
`20260909T180713Z-404` passes 12 exact native controls, 12 exact CPU-reference
candidates and 24 unchanged-input checks (48/48), with clean exit 0 and unchanged
native runtime. No full-weight reload or CPU arithmetic fallback is involved.
Report `scripts/ci/dspark-residual-wide-simulator.json`, SHA256
`850be2b14fc5c1f87e7320e85fc841b008f4ecbf05de1c98b23633fd12f4e116`.

Both layer residual boundaries now explicitly use this helper, and its source
is included in the layer report closure. This is an integrated candidate, not
a new full-layer pass: learned attention and down-projection discrepancies
remain open. Their saved operands are the next short diagnostic fixtures;
the original 102/192 failure and all qualification thresholds are retained.
The integrated helper passes all 1,160 host tests and 59 simulator-harness tests;
the simulator wrapper also passes shell syntax validation.

### Targeted arithmetic follow-up

The retained down projection matches the existing Blackhole grouped-product /
FP32-destination reference **bitwise at all 48 sampled coordinates** when the
fidelity span is 32. The sample includes four failing and four passing CPU-FP32
coordinates from each case/chip, with the entire 8,704-term reduction retained.
Span 16 matches only 3/48. This points to native arithmetic in these samples,
not a demonstrated sharding/layout defect; it is not full-output qualification.
All original per-element failures remain recorded. CPU attribution report
`scripts/ci/dspark-layer-rounding-attribution.json`, SHA256
`a33c11d4de5c524ba6c1284d84e2031e67fc951f5990376e1918b4d4ad482b87`.

Using the native BF16 scale in the CPU attention reference leaves 36 of the
original 37 own-input element failures across the six case/chip pairs. Scale
rounding alone is therefore not a fix.

The existing cached FP32 SFPU dot/row-sum composition now passes the eager-only
learned-attention diagnostic `20260909T182723Z-406`, code `7cd6b20`:

| Saved-input attention checks | Passing |
| --- | ---: |
| Final BF16 output against FP32 CPU attention | 6/6 |
| Scores, probabilities and FP32 outputs | 18/18 |
| Exact borrowed inputs | 24/24 |
| Exact chip-local Q/K/V widening and GQA expansion | 18/18 |
| **Total** | **66/66** |

All final comparisons have zero out-of-threshold elements, versus 37 total in
the retained precise-native output. The final output is not bitwise identical
to CPU: 5-8 BF16 elements differ per case/chip, max absolute difference 0.015625.
The original 0.01/0.01 combined threshold is unchanged. The six retained native
comparisons are diagnostic baselines, not controls rerun in this experiment.

No checkpoint weights are loaded and no native files are changed. Outer exit 0,
clean closure, stable source/native hashes and independent numerical
recalculation from the saved tensors all pass. The 38.9-second simulator wall
time is **not hardware latency**. All 1,165 host tests and 59 harness tests pass.
There is no trace, full-layer, fabric, coding-quality or performance qualification;
the component pass alone does not qualify the integrated layer or serving.

Report `scripts/ci/dspark-attention-captured-simulator.json`, SHA256
`b9f4dae5fa04425143bc8a3a009497adbb605b4a10f567e60e28689d84aad641`.
Saved output tensors remain local, SHA256
`cfe3f2d084c8ed7fe58052e4a8b2986911dd3fb3ae4aab622fe9c86661226f0e`.

### Integrated candidate

`dspark-layer-probe.py --composed-attention` now connects this FP32 attention
composition to the full learned layer and both widened residual boundaries.
Every intermediate allocation is retained through the caller's trace lifetime,
including allocations made before an operation throws. The ordinary native
attention path remains available; no serving defaults change.

The integrated mode requires the **original** packer, SDPA sources and native
binaries, with no active graft. The independent gate requires the same explicit
mode and source closure. All 664 checks and numerical thresholds are unchanged;
the next full matrix includes all three cases, both chips and changed-input
replays of attention, MLP and final reduction. An isolated attention pass or
native arithmetic sample cannot substitute for that matrix.

Full-matrix run `20260909T183658Z-393` uses code `779ae7f` and the original
runtime. It completes the entire matrix, then exits 1 for **78/192 failed eager
comparisons**, down from 102 in the original eager-only diagnostic.

| Complete integrated layer checks | Result |
| --- | ---: |
| Frozen CPU checkpoints | 6/6 pass |
| Eager arithmetic comparisons | 114/192 pass |
| Exact changed-input replay and bindings | 208/208 pass |
| Borrowed inputs | 196/196 pass |
| Complete learned parameters before/after | 44/44 pass |
| Stale-input controls | 6/6 pass |
| Inactive-row zeros | 12/12 pass |

All 42 mandatory exact eager comparisons pass, including both residual
boundaries. MLP gate/up comparisons also pass now. All three phases execute and
replay on both simulated chips, but **the full 664-check qualification rejects
the run**. All 472 non-eager checks pass; that does not overrule the 78 failures.
Mesh/checkpoint close cleanly, sources/native binaries are unchanged, and the
independent reconciliation preserves the failed result. Simulator wall time is
1,620.9 seconds, not hardware latency. No hardware allocation or serving change.

Report `scripts/ci/dspark-layer-composed-simulator-failed.json`, SHA256
`59ab89b185696643ce7422975128a16069fe05093ad09349c1a56258e1c171a6`.
Observed tensors remain local, SHA256
`f064c5ba2edaacbd23d6bb11beba10da5d0bf2e409e4add6ad0e14157cbccb0a`.

Own-input attribution confirms the integrated attention passes all six numerical
comparisons and both residual additions are bitwise exact in all six case/chip
pairs. The final output still differs from the frozen CPU backbone: relative L2
is 1.292%, 1.215% and 1.521% for the three cases. These are numerical diagnostics,
not coding-quality measurements. Attribution SHA256:
`79a7cdeed5b34f1994a30eb321fe0ec07cde2ae9d0b1e895d29b763da9e9f795`.

### Reference-backend mismatch

The reviewed upstream CPU control explicitly selects `_attn_implementation='eager'`.
It rounds QK scores, their scaling and probabilities through BF16; its rotary
also rounds each product before addition. The device candidate instead uses
FP32 attention and FP32 rotary intermediates. Comparing these different
arithmetic policies is not a clean test of kernel implementation error.

A controlled CPU attribution keeps the **same observed context**, all eleven
learned weights and input noise, changing only attention/rotary arithmetic:

| Case | Device versus eager CPU | Versus FP32 attention | Versus FP32 attention + rotary |
| --- | ---: | ---: | ---: |
| Pattern 0, position 0 | 1.187% | 0.832% | 0.711% |
| Pattern 0, position 8,190 | 1.219% | 0.884% | 0.654% |
| Pattern 1, position 8,190 | 1.443% | 0.978% | 0.755% |

Values are final-output relative L2, not an alternative passing threshold.
Matching policies reduces this discrepancy by about 40-48%, but thousands of
elements still fail the original pointwise comparison. It explains part of the
gap, not all of it. The original six frozen CPU checkpoints remain exact.

`scripts/ci/dspark-backend-attribution.json`, SHA256
`8faee2d8b23b14de0ab32b0a2edd7e070422953c47e3f31462286f4072abe662`,
retains both device-versus-variant and variant-versus-frozen comparisons. Source
stability and whole-checkpoint closure pass. **No reference is requalified and
no numerical limit is relaxed by this diagnostic.** A backend-matched upstream
reference is the next step, before another full-layer kernel rewrite.
After these changes, all 1,176 host tests and 59 simulator-harness tests pass;
the simulator wrapper also passes shell syntax validation.

### Long-context preparation

The layer bring-up still has **32 historical rows**, not the 4K benchmark
context. Extending the existing 1D projection naively would assign all 128 M
tiles of a 4K history to each output-column worker. A separate, unqualified
`dspark_history_projection.py` candidate distributes the M dimension across
eight worker rows, while retaining the complete learned 5,120-by-512 local K/V
matrix. At 4K it assigns 16 M tiles and two N tiles per worker on an 8x8 grid.

The installed native binary accepts that 2D program configuration; four host
tests verify geometry, precision, ownership and rejection guards. This is **not
a simulator pass, an L1-capacity measurement or a speedup**. It is not wired into
the active layer run. Learned full-size simulation against the existing tiled
native projection, history masking/rotary and full-context attention remain
required before connecting this candidate to a 4K request.

| Full-matrix requirement | Checks |
| --- | ---: |
| Frozen CPU layer checkpoints | 6 |
| Complete learned eager stages and final references | 192 |
| Exact changing-input replays and stable bindings | 208 |
| Borrowed inputs | 196 |
| All learned parameters before/after | 44 |
| Stale-input controls | 6 |
| Inactive output rows remain zero | 12 |
| **Total** | **664** |

## Next gates

1. Validate a backend-matched upstream CPU reference and reconcile the remaining
   native arithmetic differences; retain the original failures and tensors.
2. Qualify all 664 checks, restore the original native runtime, then independently
   reconcile the source-bound report.
3. Connect all five learned layers and selector, then the real TP collective and
   target-verification path. Measure coding acceptance and committed PP / CTX / TG
   through CI before any serving change.

The measured single-stream hardware result remains **PP 3,324.52 / CTX 4,096 /
TG 74.27**. No layer-component pass changes that result or establishes 200 TG.
