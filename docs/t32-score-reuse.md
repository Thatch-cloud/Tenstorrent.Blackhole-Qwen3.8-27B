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
