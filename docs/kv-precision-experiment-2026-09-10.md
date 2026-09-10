# KV precision: isolate the benefit

No precision or serving defaults change in this experiment plan.

## What is implemented

| Component | Evidence | Next action |
|---|---|---|
| Target attention KV | Runtime `allocate_kv_caches` selects BF8 when `QWEN_SDPA_BF8=1`; earlier hardware attention profiling used this flag | Record actual dtype and shape on both chips in the next request report |
| Target query | The same flag also selects BF8 queries in TP attention | Do not label a flag toggle as a KV-only comparison |
| DSpark history | `FullHistoryKV` and `StableHistoryKV` explicitly require BF16 | Prototype BF8 storage separately, preserving fixed banks and valid-prefix semantics |
| GDN state | Recurrent/convolution state is not ordinary attention KV | Preserve its precision during cache experiments |

## Experiment order

1. Establish actual tensor formats, not environment flags alone. The request
   report now records all 32 target KV buffers and 240 GDN state tensors on both chips.
2. Simulator-test BF8 draft storage: conversion, partial-tile append, bank swap,
   rejected-prefix discard, fixed-capacity attention and trace replay. Check
   untouched history and compare attention outputs at unchanged tolerances.
3. Compare BF16 versus BF8 draft history on hardware at CTX4096 with identical
   target precision, queries, prompts and proposal width. Charge conversion and
   publication costs to committed TG; record acceptance and setup-inclusive latency.
4. Extend to CTX16384 and CTX32768 only after removing and validating the current
   draft-history capacity limit (8192). Do not silently truncate history to pass.
5. Consider lower-bit target KV only after proving a material attention-bandwidth
   bottleneck and native operator support. Do not assume BF4 is a drop-in SDPA format.

## Results required

| Variant | Streams / batch | PP tok/s | CTX tokens | Committed TG tok/s | Cache bytes | Acceptance | Coding checks |
|---|---|---|---|---|---|---|---|
| Unchanged precision | 1 / 1 | Pending matched run | 4096 | Pending matched run | Pending | Pending | Pending |
| BF8 draft history | 1 / 1 | Not measured | 4096 | Not measured | Not measured | Not measured | Not run |

Report allocated bytes separately from valid-history bytes, including both banks
and trace-owned copies. Smaller storage alone is not a throughput result. Target
KV changes require quality comparisons against the original target, not merely
agreement between speculative decoding and an equally quantized native control.
