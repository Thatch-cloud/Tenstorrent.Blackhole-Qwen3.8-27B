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

## Backend-matched CPU reference

The complete five-layer CPU reference passes **96/96 exact checks**: 48 retained
eager controls and 48 comparisons against the reviewed upstream topology with an
explicit FP32 attention/rotary policy. BF16 linear, normalization and output
boundaries remain unchanged. This is a declared custom backend, not a claim of
stock SGLang or FlashAttention equivalence.

All 62 checkpoint tensors, source closure, repeated and changed inputs, and the
saved stage tensors were independently reconciled. The original eager reference
and the 78 TT numerical failures remain intact; native matmul is not emulated.
Report `scripts/ci/dspark-backend-cpu-reference.json`, SHA256
`efd8bd223ad7adf373efe41fff93488a309efff278e7efcf138f111b00835d37`.
All 1,181 host tests pass at this checkpoint.

## Device-only TP boundaries

Shared-BDF simulation `20260909T200506Z-380` passes **144/144 exact checks** on
the saved learned attention and MLP partials. Both boundaries now use a device
all-gather and rank-ordered slices, followed by the unchanged residual arithmetic.
There is no host tensor staging inside either captured boundary.

| Boundary requirement | Passing checks |
| --- | ---: |
| Eager outputs against retained native tensors | 36 |
| Changing-input trace replays and stable output bindings | 48 |
| Borrowed inputs unchanged | 56 |
| Omitted-update negative controls | 4 |

Exit 0, source/native closure and clean teardown were independently reconciled.
Report `scripts/ci/dspark-mesh-simulator.json`, SHA256
`b5bf2abecce186a86721b3abfc35b0194e87f777ec6caa5c70da340914915355`.
The simulator explicitly uses one link; this does **not** qualify the physical
four-link fabric or measure speed. The old numerical failures remain open.

### Complete layer in one device trace

Run `20260909T201255Z-393` passes **534/534 exact checks**, with clean exit 0.
All attention, MLP and residual operations execute together, including both TP
collectives; no host copies or CPU arithmetic separate the phases inside the trace.

| Full-layer transport requirement | Passing checks |
| --- | ---: |
| All 26 eager stages, three cases, both chips | 156 |
| Four changing-input trace replays, all stages and both chips | 208 |
| Borrowed inputs unchanged | 112 |
| All eleven learned weights before/after, both chips | 44 |
| Omitted-update negative controls | 2 |
| Inactive attention/output rows stay zero | 12 |

Saved eager tensors match the retained same-arithmetic native control bitwise.
Coordinates, replay hashes, stable bindings, all 40 source hashes, 1,496 native
fingerprints, saved operands and whole-checkpoint closure were independently
reconciled. Report `scripts/ci/dspark-layer-mesh-simulator.json`, SHA256
`1374f6cbd1704a04069d29b8fa0b32db523972298e19699d0b45df2848a8607a`.

This qualifies **transport and capture of one learned layer**, not its failed CPU
numerical gate, all five layers, physical fabric, coding acceptance or throughput.
History remains 32 rows with seven proposal rows. No hardware run follows from
this component result alone.

The next five-layer device chain is also wired, keeping projected context and
rotary inputs shared while passing each layer's actual output into the next.
The complete learned run `20260909T205211Z-403` on code `e83cbdb` was interrupted
by an unclean WSL restart during the fifth layer's weight loading. It recorded
88 passing parameter checks but **no five-layer forward result**. The launcher
ended with status 1; the native wrapper's exit-status file is missing, so this
is not a clean kernel failure or a passing matrix. The raw report and a separate
interruption record are retained.

Windows also reported `Thread failed to start`. Host memory pressure is a
plausible cause, not proven. All 1,496 native fingerprints remain unchanged.
Any future simulator retry requires a **per-run 3-GiB memory / 4-GiB swap cgroup budget**,
with boot identity and OOM counters recorded. Global WSL settings, serving
defaults, arithmetic and the complete test matrix are unchanged.

It uploads and audits one tensor at a time rather than keeping duplicate CPU
copies of all weights. Its matrix includes all 56 backbone parameters, all 131
device stage tensors, three eager cases and four changing-input trace replays:
**2,394 functional checks plus 72 separately reported CPU numerical diagnostics**.
No terminal five-layer result is available yet. All 1,193 host tests and 59
simulator-harness tests pass before this run; those are not device evidence.

Feature projection and the full-vocabulary selector remain separate components.
An unqualified target-head bridge is prepared: borrow the existing TP2 LM head,
gather all 248,320 vocabulary scores on-device, and preserve query rows zero
through six for Markov selection. It does not reuse DFlash2's top-16 shortlist
or discard the anchor query row. Four bridge orchestration tests pass.
The full-size transport probe now passes **50/50 checks**, including changing-input
trace replay, swapped-rank and dropped-row-zero controls. Run
`20260909T213101Z-407` exits cleanly with status 0; report SHA-256
`e95f780502fefcafa5b07c813614450c88609044abf27e606aab11d4b293216c`.
Peak cgroup memory is 2,032,467,968 bytes, with no swap use or OOM events.
All 21 source and 1,496 native fingerprints remain unchanged.
This validates synthetic full-size logits transport, **not target weights,
Markov inference or physical links**.

## Faster experiment cycle

Whole-model TTsim is no longer the prerequisite for integration. The interrupted
run spent roughly half an hour loading weights without a five-layer result.
Keep its optional diagnostic matrix; do not restart it as the next experiment.

| Stage | Where | Required evidence |
| --- | --- | --- |
| Changed arithmetic/layout | Targeted TTsim | Numerical, replay and ownership checks for the change |
| Complete learned backbone | Hardware CI | All five layers, actual device handoffs, changing inputs; separate CPU diagnostics |
| Target integration | Same pinned hardware runtime | Actual features, embeddings, LM head and Markov feedback; exact target verification |
| Fast screen | One loaded model per CI job | Matched PP / CTX / TG, acceptance and verifier/draft/publication time |
| Promotion | Hardware coding/context ladder | Repeatable committed-TG gain and unchanged token/state correctness |

Reuse the existing ABBA and runtime/weight caches rather than rebuilding unchanged
components. Reject regressions before expanding the coding/context matrix.
The approximately 62-ms verifier and useful accepted proposal length are the
priority; small isolated kernel wins are not substitutes for request throughput.

The new opt-in `dspark-backbone` suite uses the existing exclusive hardware
workflow and pinned image. It joins **learned FC + normalization + all five
layers + final normalization** in one trace, with four explicitly selected
physical fabric links. One checkpoint load supports three input cases, four
changing-input replays and 30 warm timed replays. It audits all 58 parameters,
137 stage tensors, borrowed inputs and zero padding. Compilation/loading time
is separate from warm backbone time; **neither is committed TG**.
Its input features and noise are still synthetic. Target embedding/head/Markov
integration, 4K history and committed-token acceptance remain subsequent gates.
All 1,213 host tests and 59 simulator-harness tests pass before dispatch.
Hardware run [`34410668392`](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34410668392)
uses immutable code `b8acf391414efe0a5fab03daaf39255f6d89327c`.
It failed **before opening the mesh**: the pinned image uses Python 3.10.12,
but the simulator-side file hash helper requires Python 3.11. The checkpoint
was successfully staged and verified; no backbone timing or numerical result
was produced. The failed-run record retains exit 1 and its log hash.

The hardware adapter now uses a streaming SHA-256 helper without changing any
frozen arithmetic/reference source. Python/config/source/input preflight runs
**before** compilation. The first attempt spent 260 seconds rebuilding the
runtime; a new content-addressed cache preserves that isolated library for
subsequent runs, keyed by the pinned image, build scripts and registration patch.
Cache restores verify both library paths and reapply the audited CCL source
patches. Cache hits and timing are recorded; savings are not yet measured.
Retry [`34411942931`](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34411942931)
uses code `edf45b0f13638b5526533d5afa59e5748dea29e0`; 1,222 host tests pass
before dispatch. It remains the same complete FC/five-layer scope, not a
reduced matrix or a target-generation benchmark.
This retry also stopped before device execution, but **before rebuilding**:
the container ran for about 4.36 seconds before the source gate rejected its
modified `slice.cpp`. The image contains an additional tile-window narrowing
optimization absent from the simulated runtime. The retained hardware source
and local original share revision `9f9cd4f`; their exact diff is 39 added lines.

For this disposable 32-row DSpark integration only, the launcher now restores
the exact simulator-tested `slice.cpp` from the pinned Git object before build.
Both the image source hash and restored hash are checked; unknown versions
still fail. The strict source gate and full integration matrix are unchanged.
Serving and other experiment runtimes keep their existing slice implementation.
The runtime cache key includes this restoration, so it cannot reuse a differently
built library. Twelve targeted restoration/cache/compatibility checks pass.
The source-aligned retry is
[`34412639592`](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34412639592),
code `d1bd01191ff236994bf1bbc63ceeec1e819b828b`.
It confirms the slice restoration and native numerical-source parity, then
stops at an incorrectly placed **pre-build** library-alignment check. The image's
installed and staged libraries are not identical until the existing build step
synchronizes them. That alignment check now runs after build/cache restore,
still mandatorily before device execution; source checks remain before build.
Seven compatibility tests pass, including rejection of mismatched post-build
libraries and changed sources at either stage. No numerical gate is waived.
The corrected-stage retry is
[`34413173251`](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34413173251),
code `23ce458a9b65ec6b3dc8badd7f905a12032c6262`.

The next target adapter is prepared separately in `dspark_target.py`: borrow
the real target's `.embd`, gather its hidden shards, zero-pad after the seven
query embeddings, execute FC/backbone, borrow the target LM head, then apply
the full-vocabulary Markov feedback. Four host orchestration checks pass.
This adapter is **not in the active hardware run and is not device-qualified**.
It does not discard query row zero or feed mask-token embeddings into inactive
padding rows. Caller validation, actual device execution and target-state
verification are still required before request measurements.

## Combined hardware result

Run [`34413173251`](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34413173251)
on code `23ce458a9b65ec6b3dc8badd7f905a12032c6262` completes cleanly, exit 0.
The complete learned FC/normalization, all five layers and final normalization
execute in **one device trace with four explicitly selected physical links**.

| Measurement | Result | Scope |
| --- | --- | --- |
| Warm replay mean | **10.1900 ms/block** | Three batches of ten queued replays; excludes input update/readback |
| History / query rows | **32 / 7** | Synthetic feature/noise inputs, not a target request |
| Functional checks | **2,398 pass** | All 137 stages, 58 parameters, changing inputs, ownership and padding |
| Separate CPU diagnostics | **6/72 pass** | Numerical gate remains failed; not relabeled as functional success |
| Complete probe wall time | **44.36 s** | CPU references, upload, compile, checks and timing |
| Native runtime build | **262 s** | First cache miss; built library now retained for reuse |
| PP / committed TG | **Not measured** | No target head, Markov feedback or target verification in this run |

All matrix coordinates, 62 checkpoint tensor fingerprints, 55 source hashes,
1,496 native hashes and replay/eager hashes were independently reconciled.
The retained report is `scripts/ci/dspark-pipeline-hardware.json`, SHA-256
`632215b0a7c90420ab53ad06ecabf82e8124ee8262f4d9ca11ccaae4851a7262`.
Final-normalization relative L2 differences versus the declared CPU backend are
1.43–1.56% across the three synthetic cases. Earlier numerical failures remain
open. This is neither coding-acceptance evidence nor a comparison against the
complete DFlash2 proposal time.

The targeted **seven-row embedding lookup, hidden gather and zero-padding**
simulator gate passes all 44 checks, including changing-input replay. It uses
a synthetic 64-row lookup table, not real target weights. Run
`20260909T230225Z-411` closes cleanly, exit 0; simulator execution is 22.7 seconds,
not hardware latency. Peak process-cgroup memory is 1,051,586,560 bytes with no
swap or OOM events. All 27 source and 1,517 native hashes remain unchanged.
Report `scripts/ci/dspark-noise-simulator.json`, SHA-256
`6215f1ee04d8928814ac79b56ae7fe63bd345ebf6aaa83995814c245bb5c1cc0`.

The new [target-bound integration lane](dspark-target-integration-2026-09-10.md)
connects real target features/embeddings/head to all five DSpark layers and the
full-vocabulary Markov selector. It is not yet a hardware result or TG measurement.

## Next gates

1. Connect actual target features/embeddings/LM head and the full-vocabulary
   Markov selector, preserving target state and changing-input trace replay.
2. Extend history to 4K and qualify useful wider proposals; retain mask,
   ownership, numerical diagnostics and exact target-state gates.
3. Measure coding acceptance and committed PP / CTX / TG through CI. Wider useful
   proposals and/or a faster verifier remain necessary for 200 TG; do not infer
   speed or coding quality from a component pass or change serving defaults.

The measured single-stream hardware result remains **PP 3,324.52 / CTX 4,096 /
TG 74.27**. No layer-component pass changes that result or establishes 200 TG.
