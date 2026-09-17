# Experiment record template

Copy into a new model/experiment-specific Markdown file. Replace each `Not recorded`
entry; use `Not measured` or `Not applicable` explicitly, never silently omit a gate.

## Question and decision

| Field | Record |
|---|---|
| ID, date, owner | Not recorded |
| Status | Planned |
| Hypothesis and measured bottleneck | Not recorded |
| Single changed variable | Not recorded |
| Expected benefit and mechanism | Not recorded |
| Correctness requirements and rejection conditions | Not recorded |
| Decision and evidence | Not recorded |

## Reproduction

| Field | Record |
|---|---|
| Repository, immutable commit/tag | Not recorded |
| Container/runtime/native-library hashes | Not recorded |
| Model, tokenizer, drafter revisions and hashes | Not recorded |
| Precision by operator, intermediate and cache | Not recorded |
| Cards, PCIe, firmware, dispatch, actual grid and fabric links | Not recorded |
| Exact command/flags and environment | Not recorded |
| CI run, attempt, artifact location and report SHA256 | Not recorded |
| Baseline revision and rollback command | Not recorded |

## Workload and timing

Record prompt set/hash, template, sampling/seed, thinking mode, stream/batch count,
prefix-cache policy, cold/warm state, trial ordering and repetitions. State whether
audited requests are excluded from timing, and define every timing boundary.

| Arm | CTX | Output allowance / actual / EOS | PP | Committed TG | TTFT | Total request time |
|---|---:|---|---|---|---|---|
| Control | Not recorded | Not recorded | Not measured | Not measured | Not measured | Not measured |
| Candidate | Not recorded | Not recorded | Not measured | Not measured | Not measured | Not measured |

Include per-request samples and variability, not just a winning average.
For speculation add draft/verify/commit times, proposed/accepted/committed counts,
acceptance definition and cold-reset boundary. Report loading/build/setup separately.

## Acceptance checklist

- [ ] Host guards, shapes and lifetime tests pass.
- [ ] Synthetic eager/replay results cover actual producer shapes and changed inputs.
- [ ] Poisoned padding, stale state, prefix selection and rollback checks pass.
- [ ] Real-weight full-request target tokens, recurrent state and KV checks pass.
- [ ] Both chips, actual links and source/library identities are verified.
- [ ] Matched complete-runtime gain repeats beyond observed variability.
- [ ] Held-out coding quality and context regression checks pass.
- [ ] Serving/streaming/concurrency acceptance is separately recorded if claimed.
- [ ] Cleanup, residual warnings and rollback are documented.

## What the next model should learn

Record the mechanism supported by evidence, portability limits, rejected alternatives,
remaining uncertainty and the next action. Link raw artifacts and the programme entry.
Do not turn a simulator pass, a kernel speedup or an external benchmark into a
whole-model performance claim.
