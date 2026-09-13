# Exact Markov bias cache: feasibility only

The current 8K combined runtime reaches roughly 101 TG, not 200. Its draft
stage takes about 29.8 ms/block and verifier trace about 67.7 ms/block.

`dspark_markov_score_layout.py` repeats a learned rank-256 dot product at each
of 15 proposal steps. Its bias depends on the previous token and fixed learned
weights, not on the current base-logit row. An exact cached bias could therefore
be reused before adding the new base logits. Do not cache final scores or tokens.

## Measured reuse opportunity

Cold per-request LRU analysis of run 34730226400, without future tokens,
prompt prewarming or cross-request reuse. Each timed request has 165 actual
Markov predecessors, including rejected draft prefixes, with 63 distinct tokens.

| Cache slots | Hits / lookups | Hit rate | FP32 row payload per card |
| ---: | ---: | ---: | ---: |
| 16 | 73 / 165 | 44.24% | 15.16 MiB |
| 32 | 91 / 165 | 55.15% | 30.31 MiB |
| 64 | 102 / 165 | 61.82% | 60.63 MiB |
| 128 | 102 / 165 | 61.82% | 121.25 MiB |

These are host trace counts, not measured device hits or saved milliseconds.
Payload excludes tags, alignment and scratch space. One frozen coding task is
not evidence of general workload hit rates. The reproducible analyzer is
`scripts/ci/dspark_markov_cache_analysis.py`; three host tests pass.

## Required experiment

1. Prototype a bounded device-owned 64-slot cache, with validity and weight
   identity checks. Preserve the original native dot product for misses.
2. Prove cache-hit execution actually skips the dot product. A gather/where
   selection that still executes every matmul is not an optimization.
3. Synthetic simulator checks: exact FP32 bias bits, hits/misses, collision and
   eviction, weight invalidation, changed-input replay, poisoned cache rows,
   and teardown. No model weights in simulation.
4. Matched combined hardware requests: identical proposals, target tokens/state,
   cold/warm setup costs, actual hit counts and PP/CTX/TG. Include multiple coding
   tasks rather than tuning cache membership to this recorded completion.

Reject 128 slots unless new workloads demonstrate useful additional hits.
Do not change learned weights, shorten vocabulary, or move lookup decisions to
host-side per-token synchronization. Serving defaults remain unchanged.

This optimization cannot reach 200 TG alone: even deleting the entire draft
stage leaves the current verifier slower than the roughly 55 ms cycle budget
at the observed eleven committed tokens per block. Verifier and acceptance
improvements remain necessary. The full context ladder remains outstanding.
