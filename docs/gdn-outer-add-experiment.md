# GDN outer-product/state-add fusion

Status: simulator and full-request exactness pass. No demonstrated trace speedup;
candidate disabled while the native combined runtime remains the reference.

Initial simulator run **34696989003** compiled successfully but failed the first
eager gated-output comparison on chip 0. Devices closed cleanly. The failure
does not yet identify rounding versus dataflow as the cause. A diagnostic retry
records differences for both chips, all prefix states and the FP32 bridge before
rejecting the candidate. Exactness is not relaxed; no hardware test is admitted.

Diagnostic run **34697183901** confirms finite but unequal outputs on both chips.
Chip 0 differs in 17,764/49,152 gated-output values and 1,112,139/6,291,456 prefix
state values. Maximum absolute differences are 0.015625 and 0.00048828125,
respectively. Differences begin at the first token, not only after trace replay.
The FP32 bridge also differs. These observations do not prove the cause; inspect
the pinned image's SFPU addition implementation before changing arithmetic again.

| Question | Experiment |
|---|---|
| What changes? | Compute each outer-product tile, add its decayed-state tile in destination registers, then pack once |
| What disappears? | Outer-product CB write/read, its wait/pop and the separate addition pass |
| What stays? | Native multiply, BF16 recurrence rounding, readers/writers, norm/gate, all state outputs |
| Numerical risk | SFPU addition may not reproduce the native FPU addition bit-for-bit |
| First gate | Synthetic T16 two-chip simulator; output, prefix states, FP32 bridge and input immutability |
| Resource boundary | Runner CPU only, no weights, 15-minute probe limit |
| Hardware gate | Only after exact simulator admission; matched complete combined-runtime requests |

This candidate does not include the paired-copy experiment. That prior change
did not improve the measured blocking trace. The combined reference remains the
native T16 path, approximately 105–106 committed TG at CTX 4096 for the current
merge-intervals request. The 200 TG single-stream objective is unchanged.

## FPU destination-reuse retry

Read-only source export **34697363596** confirms the pinned API provides
`add_reuse_dest_init` and `add_reuse_dest_tiles`. The revised candidate uses
`DEST_TO_SRCB`: decayed state remains operand A and the outer product held in DST
becomes operand B. This removes the SFPU add and the second register copy from
the failed candidate. It still avoids the outer-product CB write/read. Exactness
must be re-established in the simulator; the source inspection alone is not proof.

Simulator run **34697528864**, revision `93a13bd`, passes all 24 exact output,
prefix-state and FP32-bridge comparisons, plus 48 input-immutability checks.
Both chips close cleanly. The eager and changed-input replay matrices and all
738 captured source hashes were independently verified against the worktree and
pinned source export. This qualifies only the synthetic T16 operation.

Retained report SHA256:
`a2a90b18cc56c6c1e2acafc90e559b1890ddb9f13c2484505439f2f6306b81ca`.

`gdn_outer_add_gate.py` rejects any other report, incomplete matrix or changed
numerical dependency, including the pinned arithmetic API headers. The SFPU
variant remains rejected. The next measurement must compare complete requests
with all accepted runtime features, not an isolated recurrence timing claim.

## Complete-request hardware result

Run **34697917056**, revision `b12c10c`, one stream / batch 1, CTX 4096.
Each arm has one audit and two timed EOS-ended responses, 121 committed tokens
per response. Instrumented audit times are excluded from TG.

| Arm | PP tok/s | CTX | Committed TG | Blocking trace ms/block | Draft ms/block |
|---|---:|---:|---:|---:|---:|
| Native recurrence | 3313.36 | 4096 | 104.44 | 68.011 | 28.925 |
| FPU outer-add fusion | 3330.16 | 4096 | 106.07 | 68.101 | 27.653 |

The 1.56% TG increase does not establish a recurrence speedup: blocking trace
time worsens by 0.090 ms while drafting improves by 1.272 ms/block. Setup-inclusive
latency also worsens, from 6103.89 to 6150.25 ms/request. Do not promote this as an
additive kernel gain or a new performance milestone.

All six requests pass exact output, recurrent-state and inactive-slot checks.
Timed arms each commit 242 tokens and accept 222/330 proposals. Each candidate
request records 96 modified T16 program constructions and clean scope restoration.
All 786 script source hashes and both pooled TG values were independently checked.
Program construction evidence is not a per-kernel device profile; further
attribution must establish the executed trace's actual kernel costs.

Hardware report SHA256:
`a02a333de7d8471ed4dc706a41b873543e33f98dfa0bca44ffa77426adf8b1e3`.

## Two-tile arithmetic setup candidate

The next revision computes two outer-product tiles before switching to FPU
destination-reuse addition, then packs both. This halves multiply/add setup and
register handoff groups for the four-tile recurrence partition, retaining tile
order, operand order and conversion boundaries. Only two destination slots are
used; an odd tile count has an explicit tail. This is a new numerical candidate:
the prior simulator report must reject its changed source until a fresh pass.
No hardware admission or speed gain is claimed for this revision yet.
