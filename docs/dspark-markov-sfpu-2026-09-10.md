# DSpark: repair the Markov score arithmetic

**Prototype only. Not simulated, hardware-qualified or enabled in requests yet.**
This is separate from the captured-proposal / commit-only hardware comparison.

## Why the previous candidate failed

The retained learned Markov diagnostic isolates the error to native grouped
BF16 matmul products, not score addition. In its 64-column slice, the native
result differs from ordinary FP32 matmul by up to 0.0025445. It matches the
Blackhole arithmetic oracle exactly but fails the original FP32 score tolerance.
That failure is not erased by a functional replay pass.

## Proposed repair

| Stage | Existing path | New prototype |
| --- | --- | --- |
| Learned dot product | Padded matrix multiply with grouped FPU products | One-query SFPU FP32 multiply, accumulate and reduction |
| Add base logits | Separate slice and add operations | Read the selected base row and add inside the same kernel |
| Score layout | Tiled, then untilized | Compact row-major output directly |
| Argmax | Native device operation | Same native device operation |

Weights and embeddings remain BF16. The codebook is transposed when packed;
no learned values change. Each worker caches the rank-256 latent and streams
successor tiles. The kernel can use 110 workers per chip. More workers and fewer
operations are design choices, **not a measured speedup**.

## Required tests

1. Reuse the exact saved failing operands, then change both latent and base scores.
2. Check score rows 0 and 6, one-worker multi-tile execution and the normal grid.
3. Require the original `rtol=atol=1e-4`, exact native argmax and changing-input replay.
4. Repeat across the complete 248,320-entry learned vocabulary before hardware.
5. Measure against the existing scorer, then test full draft feedback and target requests.

The old failure remains retained. Fixing this scorer would not clear the separate
learned-backbone numerical failures or establish held-out coding quality.

The repaired 15-query request fully accepts only two of its ten draft windows;
acceptance lengths are 12, 6, 10, 9, 15, 13, 13, 15, 10 and 9. This is why simply
increasing the proposal window is not assumed to deliver 200 committed tok/s.
