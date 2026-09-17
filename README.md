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
| Active benchmark recipe | T16 verifier, DSpark drafting, fused MLP, shared Q/K, incremental publication, tail-first draft assembly |
| T32 | Code integrated as an experimental path; fresh combined qualification pending |

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
| 261,888 | — | — | 262,144 total window including 256 output; qualification underway |
| 262,144 | — | — | Prompt + 256 output tokens exceeds position limit |

These are **offline complete-runtime tests**, not a streaming serving benchmark.
Current-recipe concurrent users, sustained generation and held-out coding quality
are not yet qualified. Historical B8 serving results use a different runtime.

At 64K, mean drafting costs 55.50 ms/block and verification/readback 81.12 ms;
both need improvement. T32 is not automatically faster and is not the default.

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
