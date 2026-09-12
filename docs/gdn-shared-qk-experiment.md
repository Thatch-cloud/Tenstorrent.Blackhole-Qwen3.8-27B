# Shared block Q/K normalization

Status: compute/dataflow generators and eight-head program builder implemented
and host-checked. Simulator comparison and model integration remain unqualified.

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

## Compute source implementation

`gdn_shared_qk_compute.py` extracts the normalization chain from the hash-pinned
native kernel. Block and serial reference variants differ only in loop count;
they retain distinct Q scaling and K normalization, EPS placement, native matmul
reduction, and separate FP32 sum/factor buffers. The reader/writer will determine
whether each input tile contains all token rows or only the current row-zero
token. No model path is changed yet.

Normalized Q/K buffers become writer-consumed outputs. The extracted compute
therefore removes its terminal waits on those two outputs: a concurrent writer
could otherwise pop a buffer before the compute's wait and deadlock. All waits
on compute-owned scratch remain. Five host tests cover the source transform,
ownership rule, coordinate mapping, and invalid inputs. Extraction from the
actual pinned source also passes. This is not compilation or numerical evidence.

## Dataflow implementation

The block reader uses full-page BF16 reads and clears inactive token rows before
normalization. The serial reference caches the same eight Q/K pages, gathers
one token into row zero, and uses exactly the same compute chain. Its writer
assembles every normalized token into distinct FP32 scratch before full-page
writes. No sub-page accessor DMA is used, following the pinned reader's known
restriction. Writes drain before buffers are released.

The program builder uses eight workers per chip and caller-owned, distinct
`[1,16,1024]` FP32 query/key outputs. It rejects wrong shape, dtype, placement,
mesh size or input/output aliasing. Nine host tests check extraction, head/page
mapping, face boundaries and scratch ownership. Device compilation, numerical
comparison, trace replay and integration into recurrence are still required.
