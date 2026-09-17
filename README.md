# Qwen3.8-27B on two Blackhole P150A cards

Optimising coding inference on two Tenstorrent cards.
**Target: 200 committed tokens/s for one stream. Not achieved yet.**

## Hardware and runtime

| Item | Current experiment setup |
| --- | --- |
| Accelerators | 2 × P150A, 32 GiB each |
| Host attachment | One PCIe x16; one PCIe x4 behind a switch |
| Inter-card fabric | Two QSFP-DD cables; four configured links |
| Mesh | TP2; explicit P150-pair physical descriptor |
| Active benchmark recipe | T16 verifier, DSpark drafting, fused MLP, shared Q/K, norm prefetch, incremental publication, tail-first draft assembly |
| T32 | Combined correctness passes; 74.68 TG at 4K, not promoted |

## Measured combined results

One stream / batch 1, pinned synthetic coding prompt, two timed requests after
a fresh correctness audit. **PP** is prompt-processing tok/s; **CTX** is actual
input tokens; **TG** is committed output tok/s including drafting, verification
and commit—not draft tokens or isolated kernel throughput.

| CTX | PP tok/s | TG tok/s | Status |
| ---: | ---: | ---: | --- |
| 4,096 | 3,249.81 | **118.22** | Passed |
| 8,192 | 3,293.61 | **106.50** | Passed |
| 16,384 | 3,170.94 | **87.11** | Passed |
| 32,768 | 2,968.87 | **89.96** | Passed |
| 65,536 | 2,564.54 | **50.45** | Passed |
| 131,072 | 2,124.78 | **53.69** | Passed |
| 261,888 | — | — | Failed full-history allocation; 262,144 total window remains unqualified |
| 262,144 | — | — | Prompt + 256 output tokens exceeds position limit |

These are **offline complete-runtime tests**, not a streaming serving benchmark.
Current-recipe concurrent users, sustained generation and held-out coding quality
are not yet qualified. Historical B8 serving results use a different runtime.

**Latest matched 4K comparison** (one stream; separate from the cold ladder):

| Runtime | PP tok/s | Committed TG tok/s | Decision |
| --- | ---: | ---: | --- |
| Unchanged T16 control | 3,322.37 | 124.72 | Retained |
| Compact draft selection | 3,314.59 | 125.99 | Correct; gain below promotion threshold |

Paired TG changes are +0.56% / +1.48%; aggregate gain is 1.02%.
The candidate is not promoted. [Combined evidence](docs/compact-draft-score-selection.md).

The 261,888-token prompt attempt failed during full-history allocation/concat;
there is no accepted 262K-window result. It needs a memory-layout fix, not an
unchanged retry. The T32 result above also lacks full T16 optimisation parity;
it is not a width-only comparison. [T32 evidence](docs/t32-score-reuse.md).

At 64K, mean drafting costs 55.50 ms/block and verification/readback 81.12 ms;
both need improvement. T32 is not automatically faster and is not the default.

## Cached prefill: separate from the cold ladder

| CTX | Reused / new tokens | Cached prefill | Effective cached PP | Committed TG |
| --- | --- | --- | --- | --- |
| 4,096 | 2,048 / 2,048 | 0.684 s | 5,988.86 tok/s | **123.93 tok/s** |

The same-run cold control takes 1.239 s: **1.81x faster prefill**, not a
decode-speedup claim. Changed-suffix output/state/feature checks pass in the
offline combined runtime. Serving caching remains disabled.
[Cache evidence and limits](docs/combined-prefix-cache.md).

## Reproduce and contribute

- [Exact recipe, image pins and CI replay](docs/runtime-reproduction.md)
- [Context ladder evidence and failures](docs/combined-context-ladder.md)
- [T16/T32 integration validation](docs/runtime-integration-validation.md)
- [Tuning playbook](docs/blackhole-tuning-playbook.md) · [Experiment programme](docs/two-card-experiment-programme.md)
- [Historical results](docs/results-history-2026-09-17.md) · [Serving/harness guide](docs/serving-harness-history.md)
- [Upstream contribution history](docs/upstream-status-history.md) · [Gotchas](docs/gotchas.md)

Serving defaults remain unchanged. Do not replace the running service with an
experimental benchmark image or bypass numerical/source admission checks.

Repository glue is [MIT licensed](LICENSE); derived patches retain upstream
licensing. See [NOTICE](NOTICE). Model weights are not redistributed.
