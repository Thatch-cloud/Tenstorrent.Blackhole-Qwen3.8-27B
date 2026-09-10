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

The earlier SFPU **dot-product** prototype remains unqualified and is not used.
This kernel receives already-computed FP32 bias; its only arithmetic is addition.
Serving defaults and learned weights are unchanged.
