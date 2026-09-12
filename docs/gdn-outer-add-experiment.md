# GDN outer-product/state-add fusion

Status: numerical candidate only. No performance claim or hardware admission.

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
