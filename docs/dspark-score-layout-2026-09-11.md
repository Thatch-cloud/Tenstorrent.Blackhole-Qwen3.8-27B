# Fused Markov score layout

This candidate keeps the learned rank-256 matmul and native argmax unchanged.
It replaces score-row slicing, full tiled addition and untilization with one
kernel that reads the selected FP32 base row and FP32 bias row, adds them on
Tensix, and writes compact row-major FP32 scores.

## Evidence so far

| Gate | State | Coverage |
| --- | --- | --- |
| Host geometry tests | Pass | Exact tile coverage and invalid-input rejection |
| Small TTsim screen | Pass | 24 eager and 18 replay comparisons on both chips |
| Full vocabulary TTsim | Pass | 12 eager and 18 replay comparisons; 248,320 scores on both chips |
| Small complete Markov chain | Pass | 90 eager and 120 replay score/token comparisons |
| Full Markov chain / hardware | Pending | Full-vocabulary chain next; no throughput qualification |

Small run `20260910T122211Z-378` exits zero and closes cleanly in 16 simulated
wall-clock seconds. Scores match both native operations and CPU FP32 addition
bit-for-bit. Native argmax IDs match. Single-worker multi-tile and multi-worker
execution pass; replay replaces poisoned outputs and inputs remain unchanged.
Report: `scripts/ci/dspark-score-layout-small-simulator.json`.

Full run `20260910T122401Z-405` exits zero and closes cleanly. Both changed-input
patterns and the return-to-original replay match native score bits exactly.
All six source hashes are unchanged and the launcher restores the original
packer. Report: `scripts/ci/dspark-score-layout-simulator.json`.

An independent validator requires both complete result matrices, current source
hashes and process exit status. The opt-in Markov backend replaces only score
layout and keeps native rank-256 matmul, argmax and token feedback. Seven host
tests pass; they are not device-chain qualification. No speedup or complete-drafter
qualification is claimed yet.

## Complete feedback-chain test

`dspark-markov-score-layout-probe.py` compares all 15 sequential score vectors
and selected tokens against the unchanged native Markov path on both chips.
The first run uses a 64-token synthetic vocabulary; this is a functional screen,
not evidence for full-vocabulary speed or coding quality.

It exercises two different anchor/logit patterns, an all-zero first-index tie,
and four captured replays including return to the original inputs. Every replay
poisons score and token outputs first, checks fixed buffer addresses, and verifies
caller inputs and weights remain intact. Full-vocabulary and learned-weight
validation still remain before hardware request qualification.

Small chain run `20260910T123844Z-402` exits zero with clean shutdown and
unchanged source hashes: 90 eager, 120 replay, 28 input-integrity and eight
weight-integrity checks pass. The original packer is restored. Evidence is in
`scripts/ci/dspark-markov-score-layout-small-simulator.json`.

## Hardware request wiring

The existing `dspark-target-attention-request` CI suite now accepts
`dspark_score_layout=true`. Both arms use precise native DSpark attention and
folded T16 target attention; only the candidate changes Markov score layout.
Two audited requests precede four timed requests in A/B/B/A order. Comparison
requires identical target tokens, state, proposal tokens and acceptance.

This route remains gated by both full-vocabulary simulator reports. The complete
chain run `20260910T124104Z-619` was interrupted; hardware has not been dispatched.
Shell syntax, workflow YAML parsing and 31 relevant host tests pass. Learned-weight
request audits and end-to-end timing remain unqualified.

The interrupted report retains 30 exact first-pattern comparisons (15 steps on
both chips), four input checks and four initial weight checks. It stops at
`native_1`, with no replay checks, final source snapshot, clean close or exit-status
file. This is partial evidence only, retained as
`scripts/ci/dspark-markov-score-layout-interrupted.json`. No matching simulator
process was found after the session interruption. The simulator packer remained
patched, so its owned runtime state must be recovered before another launch.

Recovery verified no matching simulator/launcher process and no owner PID 618.
Both SDPA sources already matched their original hashes. The exact packer patch
was reversed, its original SHA-256 `87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181`
verified, and the abandoned owner lock removed. This restores runtime state only;
it does not turn the interrupted report into a pass.

Fresh retry `20260910T210038Z-427` runs under the managed WSL service
`qwen-score-feedback-recovery-20260911.service`, independent of the terminal.
The full-vocabulary process timeout is now six hours: initialization plus just
the first native/candidate pair took roughly 45 minutes in the interrupted run,
so the previous two-hour cap was too short for all patterns and replays.
Small-vocabulary runs remain capped at 30 minutes. Kernel sources, comparison
matrices, exactness checks and the bounded memory policy are unchanged.

The first service retry was terminated with exit 143 during a journal-confirmed
WSL shutdown. A waiting Windows `wsl.exe` client is therefore kept alive for the
service lifetime (`systemd-run --wait`, hidden window). A second launch exposed
a separate TTNN import segmentation fault before any kernel test: the service
import failed without `HOME`/`USER` and succeeded when both were provided.

Current retry `20260910T210526Z-378` uses service
`qwen-score-feedback-env-20260911.service`, with `HOME=/root`, `USER=root`, and
a waiting WSL client. It has passed TTNN import and opened the simulated mesh.
These fixes concern launch reliability only; full feedback validation is pending.

## Faster validation sequence

The long synthetic full-chain simulator run is no longer the sole admission
route to hardware correctness. The changed score kernel already has exact
full-vocabulary eager/replay simulator evidence, and the complete feedback path
has the small-vocabulary simulator gate. Those admit **hardware correctness only**.

Before any request timing, the new hardware audit runs both unchanged native and
candidate fifteen-step chains with the actual loaded learned weights, two distinct
anchor/logit patterns, and three changed-input captured replays. It requires all
60 eager and 90 replay score/token comparisons, input integrity, unchanged weight
hashes and stable bindings. Each candidate request requires that audit and the
same learned-weight allocation. Full-request audits and A/B/B/A timing follow.

This moves expensive repeated unchanged matmuls to hardware, not out of validation.
The existing simulator run is left intact. Partial simulator reports remain
unqualified, and no hardware throughput result is implied by admission.

The earlier SFPU **dot-product** prototype remains unqualified and is not used.
This kernel receives already-computed FP32 bias; its only arithmetic is addition.
Serving defaults and learned weights are unchanged.

## First complete hardware result

[Run 34535719147](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34535719147)
passed on immutable revision `83f679f895296df3720df4aad8a17ef0343c5b6a`.
One stream, CTX 4096, fifteen DSpark proposals and T16 target verification;
both arms use precise native drafter attention and folded target attention.

| Arm | PP tok/s | Committed TG tok/s | Timed committed tokens |
| --- | ---: | ---: | ---: |
| Native score layout | 3317.83 | 88.51 | 234 |
| Fused score layout | 3339.11 | 95.36 | 234 |

**Gain: 7.74%.** Each arm has one audited and two timed requests, in A/B/B/A
timing order. Both accept 214 of 330 proposed tokens. Exact target tokens,
state, inactive slots and proposal equivalence pass. The learned-weight hardware
audit passes all 60 eager and 90 replay score/token checks before request timing.
Independent local validation confirms the hardware gate, request summary,
current source hashes and unchanged native sources.

Mean timings below cover the 22 speculative blocks per arm, not setup or prefill.

| Component | Control ms/block | Fused ms/block |
| --- | ---: | ---: |
| Draft | 37.42 | 29.56 |
| Verify and readback | 69.05 | 69.09 |
| Select and commit | 12.54 | 11.77 |
| Input | 0.77 | 0.72 |
| Complete block cycle | 120.12 | 111.49 |

At the observed 10.64 committed tokens/block, 200 TG requires roughly 53.18 ms
per block. Verification alone currently exceeds that budget. Eliminating the
remaining drafting time alone would therefore not reach the target: the next
major optimization must reduce verification/publication cost or increase accepted
tokens per cycle without weakening correctness or coding quality.

Artifact: `dspark-score-layout-request-hardware.json`, SHA-256
`ea0825420e4325d35c7bb62c2b2468e18532f20f0d90ca44336f58d83d8f6aed`.
Hardware audit canonical digest:
`5a242ee9023a84fd9259235f96c1a226d25d94db5477e80601ad5c0854a068f5`.

Unchanged confirmation run:
[34536767330](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34536767330).
The confirmation passed, with independent revision-bound source-hash validation,
hardware-audit validation and exact reconstruction of the request summary.

| Repeat arm | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: |
| Native score layout | 3314.52 | 4096 | 88.35 |
| Fused score layout | 3350.74 | 4096 | 96.82 |

The repeat gain is **9.58%**, with the same 234 timed committed tokens per arm and
214/330 accepted proposals. Pooling committed tokens divided by total decode time
across both matched runs gives **96.08 candidate versus 88.43 control TG** (four
timed requests per arm). These are not arithmetic averages of throughput values.
The repeat report SHA-256 is
`91c0433ee9386df21259be415e0c84b74d29ae8b6bf25f38f9f36580262d8655`;
its canonical hardware-audit digest matches the first run. Source validation uses
the tested immutable revision, not the subsequently updated profiler worktree.

These short offline requests do not certify
held-out coding quality, long-context scaling or serving readiness. The full-size
synthetic feedback simulator remains a separate unfinished check, not a claimed pass.

### Deliberate simulator reprioritization

Run `20260910T210526Z-378` was deliberately stopped at `candidate_0` to release
the simulator for the new publication kernel. Its exit status is 143; its report
remains `passed=false`, `closed_cleanly=false`, with no completed eager/replay
check matrix. It is not a full-chain simulator pass. This was a scheduling decision,
not a diagnosis of a frozen process or a failed numerical comparison.

The exact probe process was terminated while its launcher remained alive to
restore owned runtime sources. The original packer SHA-256 `87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181`
and released owner lock were verified; the service then became terminal with
status 143 and no remaining probe/launcher processes. Existing full-size changed-
kernel simulation, small-chain simulation and both exact learned hardware audits
remain the score-layout evidence. No evidence was upgraded because of cancellation.
