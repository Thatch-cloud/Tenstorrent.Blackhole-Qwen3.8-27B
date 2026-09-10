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
| Complete Markov chain / hardware | Pending | Opt-in feedback backend implemented; device-chain validation next |

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

The earlier SFPU **dot-product** prototype remains unqualified and is not used.
This kernel receives already-computed FP32 bias; its only arithmetic is addition.
Serving defaults and learned weights are unchanged.
