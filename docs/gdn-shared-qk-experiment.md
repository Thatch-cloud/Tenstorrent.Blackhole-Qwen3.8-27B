# Shared block Q/K normalization

Status: design hypothesis, not implemented or qualified.

## Why change direction?

The combined device profile puts the 96-worker recurrence group at about
11.75 ms per T16 verifier replay. Copy scheduling and one-/two-tile outer-add
fusion did not improve the complete trace. Historical input-prefetch run
34049169694 also found no useful latency improvement; do not repeat that reader
experiment solely because its estimated DRAM traffic is lower.

The current reader maps workers to Q/K heads using `worker // 12`: three value
heads, each split into four value partitions, share each key head. Each worker
nevertheless repeats the same Q/K normalization for every token. That duplicated
arithmetic is a more substantial target than copy handoffs.

## Candidate and gates

| Stage | Required change or evidence |
|---|---|
| Prepare | Normalize all T16 token rows once for each of eight local Q/K heads, using the pinned native arithmetic |
| Representation | Keep normalized Q/K in FP32; retain native Q scale and epsilon placement |
| Recurrence | Consume prepared Q/K, removing duplicated normalization only; preserve state decay/update, BF16 feedback and all prefix outputs |
| Norm/gate | Keep the current whole-head batched norm/gate unchanged |
| Ownership | Allocate prepared outputs before trace capture; verify stable addresses and no input/state aliases |
| Simulator | Compare prepared normalization and complete recurrence against native, both chips, changed inputs/replays, padding and all prefixes |
| Hardware | Matched complete combined-runtime requests; include preparation, traffic and capture costs, not just the shorter recurrence |
| Acceptance | Exact target outputs/state and a repeatable complete-request improvement; no serving change |

A block preparation operation adds traffic and dispatch, so eliminating duplicate
math does not guarantee a gain. Whole-row normalization must reproduce the
native row-zero computation; do not substitute a differently rounded RMSNorm or
BF16-normalized Q/K. First prove those numerical and ownership boundaries.

This is not a proposed 12x model speedup: only Q/K normalization is duplicated
across those workers. The 200 committed TG objective still requires substantial
improvement across verifier and drafting costs, followed by coding acceptance
and the context ladder.
