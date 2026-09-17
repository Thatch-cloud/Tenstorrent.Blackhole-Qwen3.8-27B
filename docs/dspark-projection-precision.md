# Drafter-only projection precision

**Query-projection simulator execution passes; combined correctness/performance unqualified.**

## First real-weight simulator screen

[Run 35226045287](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35226045287)
at `fbd3486a624ea160c1c59bcd91892ed0596e5cd8` completes in **1m47s**.
It uses layer-zero query weights, two synthetic activation patterns and both
simulated chips. All six poisoned changed-input replays reproduce HiFi2 eager
output exactly; five integrity checks preserve input/weight contents and buffer
bindings. Independent validation and container teardown pass.

| HiFi2 versus HiFi4, FP32 output | Observed range across chips/inputs |
| --- | ---: |
| Maximum absolute difference | 0.06569-0.08042 |
| RMS difference | 0.01147-0.01331 |
| Reference RMS magnitude | 2.851-3.323 |
| RMS difference / reference RMS | approximately 0.4% |

This is not a numerical-equivalence or model-quality pass: outputs differ.
Report SHA-256:
`361ace678704220f0ed138d2aa6878924187a92ef59d9b2dd4b0a6fd0d1e1e6c`.
Remaining projection weights/shapes require their own execution coverage before
enabling the broader drafter policy. Then measure changed proposal acceptance
and verify final tokens/state against the unchanged target in combined testing.
No throughput number or serving change is justified by this component result.

The winning combined runtime spends roughly 25 ms drafting and 66 ms verifying
each block. Small target-reader changes have not produced repeatable committed
TG gains. This candidate explores a separate contributor: DSpark layer linear
projections currently use BF16 weights, HiFi4, FP32 accumulation/output and
explicit BF16 casts. Try HiFi2 only in those projection matmuls.

The operations proxy leaves target kernels, weights, normalization, rotary,
attention, scoring, FP32 accumulation, packer policy and rounding boundaries
unchanged. It is explicitly passed to the existing drafter linear helper;
there is no global TT-NN patch or serving flag. Host tests cover routing,
ownership, policy rejection and failure without mutating shared operations.

## Required gates

The remaining six layer-zero projection screens run serially in CI matrix
`35226751937`, source `ba16283d40bfa5bd59f3e6b6d106c1a723d76674`.
The key-projection job has passed independent replay/integrity validation;
its report hash is
`1cd094769e75979dc687ed06302087c27ca2b095cf3cd4415db0f0e43905bf18`.
The matrix is not yet complete; no broader admission follows from this result.

Value and output projections also pass independent validation, each with six
exact poisoned replays, five integrity checks and clean container teardown.
Their HiFi2/HiFi4 RMS differences are 0.00913–0.00941 and 0.00614–0.00625,
respectively; neither is numerical equivalence or a throughput result.
Report hashes:

| Projection | SHA-256 |
| --- | --- |
| Value | `33d1cebe54648ff48180b228011f48c270a8b0810b3be75a10207c5271c14e3a` |
| Output | `649f2f11c50d691bbb3842da5a541af9fba29ccdd21648163af1a7b40e99a23d` |
| MLP gate | `797d9797e0a806c34eadc6417b26cd9288766d5564c14a0be728f7fbf71a75e1` |
| MLP up | `dcba1c4b0fe1421a59d4a508dc8900bb3eb05a079d3d9bf93d0b9ccd203719fe` |

The gate projection also passes all six replays, five integrity checks and
clean teardown. Its RMS difference is 0.01195–0.01216. Up now passes the same
checks with RMS difference 0.01168–0.01173. Down is running; the complete
projection set is still not admitted.

The report validator now also checks a complete seven-projection set against
independently supplied source and checkpoint fingerprints. Missing or duplicate
projections, source drift and partial replay evidence fail closed. This is only
the component evidence gate; native runtime identity and combined correctness
still require separate admission.

`dspark_layer_precision.py` supplies the pending integration boundary: an explicit
operations proxy for one proposal layer, requiring all seven expected projection
shapes in order. It does not patch shared TT-NN operations, change history setup,
or install itself into serving. Two host tests cover full routing and rejection
of partial, out-of-order or wrong-width execution. Device integration still
needs source-bound admission and complete candidate eager/replay verification.

1. Simulator: compare real-weight HiFi4/HiFi2 projection outputs; record numerical
   differences, require finite output and exact same-policy changed-input replay,
   poisoned-output replacement, input/weight integrity and stable trace bindings.
2. Combined audit: feed candidate proposals to the unchanged target verifier.
   Require final tokens and target state to match native greedy generation,
   including rejection/rollback. Compare candidate eager and replay proposals.
3. Matched timing: retain a native-precision control, record actual acceptance
   and complete-cycle committed TG. Candidate proposals may differ; do not
   silently reuse a comparator requiring identical proposals between arms.
4. Held-out coding and longer-context checks before any promotion. Lower drafter
   precision is not a claim of unchanged acceptance or a 200 TG result.

Existing native-precision numerical gates remain unchanged. A separate candidate
qualification must distinguish expected drafter approximation from target
correctness: no relaxing target checks to accommodate this experiment.

`dspark_precision_comparison.py` now provides the combined comparison boundary:
two independent full audits followed by HiFi4/HiFi2/HiFi2/HiFi4 timing. Each
timed arm must reproduce its own audited proposals; both arms must reproduce
the same native target tokens and exact state. Block counts must reconcile
with acceptance and committed tokens, and TG uses the entire decode loop.
Four host tests cover changed proposals, audit/target failures, false accounting
and a regressing timing pair. Existing identical-proposal comparators remain
unchanged. This comparator is not yet wired into a hardware experiment.

`dspark_precision_device.py` supplies per-instance proposal routing after native
history construction and before trace capture. It keeps the ordinary control
device unchanged, rejects baseline eager fallback, and inherits trace teardown.
Four host tests cover isolation, failed initialization cleanup, closed-device
rejection and successful-call accounting. The caller must still perform source
admission before construction; no global backend or serving default changes.

`dspark_precision_gate.py` now binds all seven reviewed report hashes to the
actual staged source files, checkpoint, simulator runtime, candidate manifests
and clean exits/teardown. It rejects incomplete reviews rather than deriving
trust from whatever artifacts happen to be present. Three host tests cover
successful component-only admission and source/runtime/report tampering. The
remaining down artifact and hardware staging are still required.

`dspark_precision_experiment.py` now wraps the loaded combined runtime with two
independent audits and ABBA timing. It changes only the request-owned proposal
device for HiFi2, retains target admission unchanged, checks executed layer calls
and closed traces, restores routing after every request, and rechecks sources
and evidence afterward. Two host integration tests cover all six requests and
cleanup after candidate failure. No hardware result is implied: the immutable
review manifest and CI launch remain pending.

`dspark_precision_stage.py` now stages only the candidate helpers, explicit
opt-in flag and six-request schedule onto the frozen ladder. It requires a
complete reviewed artifact set before writing an admission manifest. Two host
tests check the schedule, opt-in default and rejection of changed staging
anchors. A fresh checkout at `8c102b2` passes the actual adaptation anchors;
all seven unchanged projection dependency hashes match the simulator report.
The new local checkout is `D:\qwen-precision-combined-stage-20260918`; final
candidate admission awaits the down-projection artifact.

The hardware workflow `qwen-dspark-precision-combined.yml` restores the same
frozen target, downloads query plus the six matrix artifacts, applies the
precision wrapper, and retains exclusive card scheduling and the disk-pressure
gate. It is not launched yet: the complete reviewed report manifest must exist
first. The initial query artifact used `weight-pipeline-candidate.json`; its
compatibility reader accepts only that exact reviewed manifest/report hash pair,
not arbitrary legacy evidence. Four admission tests and the real query artifact
check pass locally.

The matrix has now completed successfully. Down has RMS difference
0.01585–0.01596, six exact poisoned replays and clean teardown; report hash
`0de80d82c1f075013cb5b0b4d67a1b487fddb9a44fe37e2cb422d9b5d3c7312b`.
All seven real reports pass source-bound admission against the staged frozen
dependencies. `dspark-precision-reviewed.json` records the reviewed artifact set.
The complete-set check also caught and fixed an admission bug: `weight_sha256`
identifies each selected tensor, not the whole checkpoint. Each projection now
has its own reviewed tensor digest; the pinned loader still verifies the whole
checkpoint. Ten focused report/admission tests pass. Hardware TG remains unmeasured.

Combined hardware run `35229622229` uses source
`856e1f29bfa69a40b67569a0e249b00936343047` and tag
`experiment/dspark-precision-combined-v1`. Exact-commit CPU CI and fresh local
staging passed before launch. The runner has passed projection admission and
host-pressure checks and is executing the combined audit/timing step.

`dspark_precision_report.py` independently recomputes both arms from complete
requests, checking actual packed target weights, unchanged fusion/norm/history,
per-arm proposal and five-tap feature audits, executed precision-layer identity,
stable sources and complete-cycle TG. Two report tests plus four comparison
tests pass locally; no result is accepted merely because the workflow is green.
