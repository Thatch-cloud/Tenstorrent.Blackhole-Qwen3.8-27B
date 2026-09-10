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
| Full vocabulary TTsim | Running | 248,320 scores, rows 0/6/14, changing inputs |
| Complete Markov chain / hardware | Not started | Requires full-vocabulary simulator evidence |

Small run `20260910T122211Z-378` exits zero and closes cleanly in 16 simulated
wall-clock seconds. Scores match both native operations and CPU FP32 addition
bit-for-bit. Native argmax IDs match. Single-worker multi-tile and multi-worker
execution pass; replay replaces poisoned outputs and inputs remain unchanged.
Report: `scripts/ci/dspark-score-layout-small-simulator.json`.

Full run `20260910T122401Z-405` is a separate gate, not inferred from the small
screen. No speedup or complete-drafter qualification is claimed yet.

The earlier SFPU **dot-product** prototype remains unqualified and is not used.
This kernel receives already-computed FP32 bias; its only arithmetic is addition.
Serving defaults and learned weights are unchanged.
