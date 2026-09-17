# T32: reuse the T16 score-layout optimisation

The integrated T32 drafter still uses the older native Markov score conversion
path. Comparing it against the fast T16 recipe would therefore change both
proposal width and score-layout policy.

The simulator-only candidate `dspark_t32_score_layout.py` reuses the existing
T16 fused base-row-plus-bias kernel, without changing the native dot product,
argmax, full vocabulary or sequential feedback. It keeps the original T32
7/7/7/7/3 segmentation and per-query observer indices. The T16 kernels and default
T32 implementation remain untouched; no speed improvement is claimed.

`dspark-t32-markov-probe.py --fused-score-layout` selects the candidate explicitly.
Start with the weight-free vocabulary-64 fixture, then qualify learned full
vocabulary and changed-input replay using the existing numerical oracle and
tolerances. The report records which score path executed and fingerprints all
five candidate/score-kernel dependencies. Hardware is rejected by the candidate.

Simulator run **35177332339 passed** in 3 minutes 15 seconds: 186 eager and 248
replay queries, including changed inputs, using the synthetic 64-token vocabulary.
Qualification report digest:
`499f7e7664cf111ff8fa40516eb49818ce4a1f5bbdc122e21e6d5a47e07c005f`.
This is not learned full-vocabulary, target integration, coding-quality or TG evidence.

`TracedDSparkDevice(fused_score_layout=True)` now selects the candidate per instance
for prepared eager warmup, capture and replay-audit execution. Native scoring remains
the default; hardware selection is rejected before model allocation. Local protocol
tests cover selection, isolation and failure-path ownership, not device execution.
Pending: learned full-vocabulary and complete-proposal qualification, followed by
a matched complete-request T16/T32 comparison.

The complete-proposal probe accepts `--fused-score-layout`. It compares the
candidate's captured hidden states, logits and tokens against a separate native-score
eager proposal on both chips, initially and after each changed anchor. This prevents
candidate-versus-itself replay checks from being mistaken for recipe parity.
The probe still uses synthetic cached K/V, so even a pass cannot qualify target
prefill, commit/rollback, coding quality or combined TG. This new comparison has
host protocol coverage but has not yet executed on the simulator.

Do not dispatch the old `t32-combined` lane unchanged: its probe timeout is 9,000
seconds and it uploads the target embedding/head and all five learned draft layers.
The new `qwen-t32-proposal-sim.yml` explicitly sets `QWEN_T32_FUSED_SCORE=1`:
six-minute probe, 450-second launcher and nine-minute whole-job cap. It preserves
the full vocabulary and five learned layers and rejects missing native-reference
checks. Weight loading is included in the cap; a setup timeout is not a numerical
failure. The three-minute weight-free score result does not predict its cost.

Run **35178536604** exited 124 at the six-minute probe cap (whole job 6m48s).
It reached `load_target_embedding_head`, but never proposal execution. The artifact
has `passed=false` and `closed_cleanly=false`; it cannot qualify the candidate.
Added persisted elapsed-time stage markers for metadata, matrix read, finite check,
hash, head transpose, combined TT-NN conversion/upload and synchronization. The
previous log cannot distinguish these costs, and no longer timeout is justified yet.

Run **35179275796** isolated the bottleneck (exit 124, whole job 3m13s):

| Target embedding setup | Seconds |
| --- | ---: |
| Read matrix | 1.004 |
| Finite-value check | 3.960 |
| Hash matrix | 1.027 |
| TT-NN conversion/upload | Did not return before the probe cap |

Conversion/upload began 12.243 seconds after opening the mesh. The head and draft
layers were never reached. This rules out checkpoint reads as the dominant delay
in this run; it does not yet separate host TT-NN conversion from simulated device
transfer. Do not spend another full-proposal run on the same loading route or
attribute this timeout to the fused score kernel. Full-weight simulator execution
is not presently a practical iteration gate; kernel simulator evidence must stay
distinct from subsequent complete-runtime hardware correctness and timing evidence.
The first bounded run, 35177093623, stopped at the host-I/O gate before Docker:
15.63% full I/O stall measured over 15 seconds, against the unchanged 1% limit.
The simulator step was skipped; that run produced no numerical result. The whole job
finished in 35 seconds rather than waiting through container setup.
CPU regression run 35177082045 passed all 144 tests. This is not device evidence.

Follow-up policy correction: host-I/O telemetry is advisory for this weight-free,
correctness-only simulator job. Busy disk can delay it, but does not invalidate
exact score/token comparisons. The job retains its nine-minute cap and reports
no timing result. A failed pressure observation stays in the artifact rather
than being relabelled a pass. Hardware throughput workflows retain their strict
1% I/O gate unchanged; numerical, source, replay and cleanup gates remain mandatory.

## Recipe parity before a T16/T32 speed comparison

`full_dspark_request.measure_dspark_request(..., t32=True)` now connects the
31-query prepared drafter, T32 publication adapter, 32-row verifier request cap,
proposal warmup and final frontier accounting in the shared request lifecycle.
This is an **audit-only simulator route**, covered by host integration fixtures;
it is not a measured full-model run. It retains native target attention/MLP and
explicitly rejects T16-only recipe flags rather than silently treating them as
T32 optimisations. Timed and hardware execution remain rejected. T16 stays the
default. Target fusion/replay parity and hardware admission remain required before
this route can become a fair performance comparison; no full-weight sim retry is
scheduled to test this wiring.

The additional `fused_t32_mlp=True` option connects the existing simulator-covered
`FusedT32Arm` to that lifecycle. It installs before verifier capture, uses the
existing packed weights and restores the original MLP methods before releasing
the drafter. A successful request must report positive fusion invocation counts
for all 64 target layers and unchanged weight bindings; enabling a flag alone is
not accepted as evidence. Host tests cover routing, restoration and rejection of
a missed layer. This has not yet executed as a complete device request; target
attention replay and other T16 recipe optimisations still need integration.

`target_attention_t32=True` plus `target_attention_t32_evidence=<report.json>`
now connects the existing replay reader and family routing without setting the
T16-only attention flag. Admission checks a clean simulator exit, all component
checks and current source fingerprints, and restricts the request to a 4096-token
prompt with at most 224 post-seed tokens (4352-capacity family). It does not extend
the component's evidence to larger contexts. Host tests cover route selection,
missing evidence, source changes, timeouts and geometry limits. Complete-device
correctness, execution coverage and hardware admission are still pending.

## Recovered hardware groundwork

The earlier T32 line did execute a combined hardware request: run **34689543028**
at `d6d5a309263d243f8cef4dad26c5ba88e23555fc` passed. That revision is not an
ancestor of this experiment branch. Its `t32_hardware_kernel.py`,
`t32_proposal_gate.py` and scope-restoration tests have been recovered unchanged,
retaining the original runtime and simulator-report pins. This avoids inventing
a second hardware deployment mechanism. It does not admit the current candidate.

The retained complete-proposal simulator artifact from run **34660555430** is
still available and independently reads `passed=true`, `closed_cleanly=true`.
The restored gate correctly rejects the current dependency closure for exactly
three changed Python files: `attention_mask_replay.py`, `dspark_t32_prepared.py`
and `dspark_t32_score_layout.py`. The mask change adds the separate simulator
context-ladder option; the latter two implement the explicit fused-score path.
Next hardware admission must bind and validate that specific delta against the
retained proposal plus new score evidence, not relabel the old pass as current.

## Keep the 200 TG target tied to measured costs

### Source composition review after the 4K cache result

The retained T32 hardware run `34689543028` was re-read, not inferred from a
component timer: its two timed requests deliver **36.56 TG** at 4K, with 128
committed tokens and 110/316 accepted proposals. Its older recipe and prompt
are not matched to today's T16 result, so it does not establish a T32 speedup.

`t32_score_composition.py` now independently checks the retained full-proposal
simulator artifact, the 31-query fused-feedback artifact, and both 64- and
248,320-vocabulary score-layout gates. It accounts for all **68** current
Python dependencies: 65 match retained proposal evidence and exactly three
have pinned, reviewed source changes. The feedback report has 186 exact eager
queries and 248 changed-input replay queries, plus input/weight/stale controls.
The audit passes against the retained real artifacts locally.

This is **not** full-proposal candidate or hardware qualification. Existing
simulator-only guards remain intact. The next integration must explicitly
combine this evidence with hardware runtime admission, native-score proposal
comparison and the full target-state audit; it must not retry multi-gigabyte
simulator weight upload or silently admit arbitrary changed dependencies.

The current cached-prefill T16 screen measures **123.93 committed TG**, with
**66.39 ms** verifier/readback and **97.60 ms** whole blocks. Prefix reuse fixes
prompt work, not the 200-TG decode shortfall. Wider drafting must be compared
on the same prompt and fused target recipe, rather than using the old 36.56-TG
line as its control.

Re-read the retained **35167726511**, 4096-context hardware artifact rather than
using isolated-kernel timings. Its two non-instrumented `request_checks` contain
20 blocks and 242 committed tokens, with pooled TG **118.2196**:

| Mean per block | ms |
| --- | ---: |
| Draft | 26.650 |
| Verify/readback | 67.249 |
| Selection/commit | 6.839 |
| Whole block | 102.281 |
| Blocking trace call, contained in verification | 65.803 |
| Output readback, contained in verification | 0.433 |

These are host intervals, not isolated device-kernel times; nested columns must
not be added together. At the measured 12.1 committed tokens per block, 200 TG
requires **60.5 ms per block**, roughly **41.8 ms less** than measured. Even free
drafting would leave about 75.6 ms per block, or at most about 160 TG at unchanged
acceptance. Therefore T32 wiring is not enough: the next performance experiment
must attribute the winning T16 verifier's combined execution, or demonstrate that
T32 increases accepted tokens enough to amortize its additional work. Readback
removal alone cannot close this gap. Keep the proven T16 recipe as the control.

| Area | Current T32 path | Required next evidence |
| --- | --- | --- |
| Markov score layout | Opt-in fused adapter; native default | Learned complete-proposal output and replay parity |
| Draft K/V assembly | `dspark_t32_attention.append_queries`: slice, concat, pad | Compare with T16 tail assembly; qualify 31-query geometry before reuse |
| Proposal history | `PreparedDSparkProposal.update_history`: copies every history tensor | Measure bytes and publication time; evaluate incremental updates without stale rows |
| Logits | Full-vocabulary gather, slice, FP32 conversion | Attribute end-to-end cost; preserve exact feedback semantics |
| Target verifier | Separate from proposal integration | Record executed GDN/norm/prefetch routes and source identities in complete requests |
| Fabric | T32 target gathers call `projection_links()` | Record actual links at all collective call sites, not just sampling |

These are source-level gaps or audit items, not measured speedups. No T16 admission
or geometry guard should be bypassed to claim T32 parity. Compare committed tokens
per total decode wall time, with draft, verify and commit phases recorded together.

## Combined hardware audit adapter

The explicit `QWEN_T32_FUSED_SCORE_HARDWARE=1` route now selects the fused
31-query scorer inside `full_dspark_request`, with T32 target attention and
fused target MLP required. It remains audit-only: native token/state checks,
feature checks and proposal replay comparisons stay enabled; TG is not reported
as a performance result. A flag alone cannot enable the route: it requires an
owned mesh scope, the retained source-bound component composition and the
installed simulator-qualified attention arithmetic. Serving defaults are unchanged.

`t32_score_hardware.py` checks AST equality of its feedback math against the
simulator-covered function, validates runtime and source bindings, and restores
the admission scope on failure. The original simulator modules remain unchanged.
Local tests cover selection, missing ownership, partial target recipes, denied
timing, all 31 feedback positions and failure cleanup. These are orchestration
tests, not physical-device acceptance.

Next: connect the outer CI probe to the installer and owned scope, validate the
target-attention evidence against current sources, then run the complete learned
proposal/native-score comparison and target request audit on hardware. No new
hardware result is claimed yet. T32 still needs the T16 optimization parity work
listed above before it can be described as a matched winning-recipe comparison.
