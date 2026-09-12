# Captured drafter operation attribution

Run `34474912529`, immutable commit `224800e`, is an attribution-only request
on the two P150A cards: CTX 4096, one stream, fifteen draft queries, folded T16
target attention and the existing precise-native DSpark proposal backend.

The latest uninstrumented comparisons place drafting around 37 ms per block.
That is a stage measurement, not proof of which device operations dominate.
The profile is intended to select larger improvements than another small MLP
layout change.

## What is measured

- Every actual prepared proposal trace, including its capture warmup.
- Both chips, per-operation durations, core counts and operand metadata.
- Per-chip kernel envelopes, covered intervals and overlapping operation sums.
- Exact eager/replay proposals, target tokens/state and feature publication.

Reversible Python hooks observe the existing trace calls; qualified proposal
sources and kernel arithmetic are unchanged. Hooks restore even on failure.
The validator rejects missing/duplicate events, changed sources, incomplete
audits or a throughput claim. Forty-one relevant host tests pass.

## Boundaries

This profile excludes host-to-device proposal staging, correctness audits and
token readback from its marked trace intervals. Those costs remain included in
ordinary committed TG measurements. Instrumented durations are neither TG nor
an end-to-end critical path. There is no serving change or performance promotion.

The run must finish and its artifacts pass independent validation before any
operation is identified as the next bottleneck.

## Validated hardware result

The run passed and its downloaded artifact passed independent validation.
There are **13 actual proposal trace replays** on each chip: one capture warmup
and twelve proposal invocations. Every replay contains 738 captured operations.
All required eager/replay proposal, target-feature and target-state checks pass.

| Per-chip trace estimate | Chip 0 | Chip 1 |
| --- | ---: | ---: |
| Kernel envelope | 34.852 ms | 34.858 ms |
| Covered kernel intervals | 34.132 ms | 34.139 ms |
| Uncovered intervals | 0.718 ms | 0.718 ms |

The repeated final trace sequence matches `dspark_markov_device.execute`:
fifteen matmuls on its explicit 10x10 grid, interleaved with score-row slicing,
addition, layout conversions, argmax and feedback embedding. The backbone's
explicit projection grid is 8x10, not 10x10. This source/order mapping identifies
the Markov suffix; the profiler does not provide source-line labels.

| Markov suffix, chip 0 | Calls | Sum of per-call median durations |
| --- | ---: | ---: |
| Rank-256 full-vocabulary matmul | 15 | 8.100 ms |
| Full-width tilize/padding | 14 | 3.875 ms |
| Full-width untilize/unpadding | 29 | 3.224 ms |
| Add base logits to transition bias | 15 | 3.179 ms |
| Argmax | 15 | 1.061 ms |
| Score-row slices | 15 | 0.723 ms |

Chip 1 agrees closely. These operation sums are not additive critical-path
proof or a speedup prediction. Operand metadata is empty in this runtime's
device CSV; no operand shapes were inferred from nonexistent profiler fields.

## Next kernel experiment

Preserve the existing learned rank-256 dot product and device-feedback order.
Fuse the selected FP32 base-logit row, FP32 transition-bias addition and compact
row-major score output into one operation, then use the existing argmax.
This targets the measured slice/add/layout chain without changing learned
weights or reclassifying the earlier failed SFPU dot-product prototype as valid.

Require exact score bits and token selection, full 248,320-column coverage,
multiple row indices, changed-input replay, poisoned-output replacement and
input integrity in simulation before hardware. Only a complete Markov-chain
and request comparison can establish a benefit.

Request SHA256: `2b94a5c1fbb97d2272edd897f68921c1f06d7dcbe5101e7c050632781b591b51`.
Device CSV SHA256: `17a8d2cfff22f8c82c58bba17a4c2d1b25afe24f9b547559366ad30ecf35f475`.
