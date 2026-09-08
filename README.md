# Qwen3.8-27B on two Tenstorrent Blackhole cards

Two-card inference and kernel experiments for fast, reliable coding responses.
**Target: 200 committed tokens/s for one coding stream. Not achieved yet.**
Experimental paths are opt-in; serving defaults remain unchanged.

## Setup

| Component | Experiment setup |
| --- | --- |
| Accelerators | 2 × Blackhole P150A, 32 GiB each |
| Placement | Tensor parallel across a `(1, 2)` mesh |
| Card-to-card connection | QSFP-DD fabric; two fabric links |
| Host connections | One PCIe x16 card; one PCIe x4 card behind a switch |
| Compute grid | 110 exposed workers/card on current worker-dispatch path |
| Extra column | Ethernet dispatch could expose 120; not validated on this pair |
| Model | Qwen/Qwen3.8-27B; 64 layers: 48 GDN + 16 attention |
| Dimensions | Hidden 5120; MLP 17408; vocabulary 248320 |
| Target weight formats | Gate/up BF4; down, other projections and LM head BF8 |
| Experiment runtime | TT-Metal `9f9cd4f` plus audited grafts; vLLM plugin `bf77cd6` |

Fabric communication does not establish fabric-based host weight loading.
The model is not uniformly limited to 36 cores: different kernels use different grids.

## Reading the results

- **CTX:** actual input context tokens, including prompt template where applicable.
- **PP:** prefill input tokens/s. `—` means not measured, not zero.
- **TG:** generation tokens/s **per stream**; each table states the timing boundary.
- **Streams (B):** concurrent requests. **Verify rows (T):** speculative positions, not users.
- **TTFT:** time to first token. Includes more than prefill; we do not convert it into PP.

## Serving baseline

September 5: three repetitions, 1024 output tokens/request, container-local streaming
endpoint with host sampling and `logprobs=1`. TG is **client-estimated**, not an
engine commit-timestamp measurement. These are not coding-quality scores.

| CTX | Streams (B) | PP tok/s | TG tok/s per stream | TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 109 | 1 | — | 19.45 | 0.309 s |
| 4078 | 1 | — | 19.27 | 1.203 s |
| 32752 | 1 | — | 18.97 | 10.356 s |
| 64504 | 1 | — | 18.39 | 23.272 s |
| 4078 | 2 | — | 18.56 | 1.837 s |
| 4078 | 8 | — | 17.18 | 10.158 s |

Tested context limit: 65536, reserving room for generation. Concurrency is not
single-stream acceleration; these medians are not aggregate throughput.
[Baseline details and evidence](docs/baseline-2026-09-05.md).

## Speculative request experiments

Synthetic lookup proposals, **not a learned drafter or a coding benchmark**.
TG counts committed tokens during decode, excluding prefill and setup.

| CTX | Streams (B) | Maximum verify rows (T) | PP tok/s | Committed TG tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 4078 | 1 | 8 | — | 23.88 |
| 16363 | 1 | 8 | — | 20.61 |

Including setup, but still excluding prefill, rates fall to 10.75/10.35 tok/s.
Acceptance is poor: 128 committed tokens require 89/99 verification blocks.
[Request run](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34084598829).

## What currently limits 200 tok/s?

Even perfect acceptance needs an eight-token draft/verify/commit cycle within **40 ms**.
Our best static target verification alone still takes longer:

| CTX | Verify rows (T) | Target verification block |
| ---: | ---: | ---: |
| 4095 | 8 | 62.35 ms |
| 16383 | 8 | 64.00 ms |

These are **block timings, not generated throughput**, excluding draft/commit overhead.
[Verifier run](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34150091732).

## Latest kernel and drafter results — September 8

| Experiment | Measured outcome | Meaning |
| --- | --- | --- |
| Learned MLP, T8 captured replay | 1.911 ms median | One isolated MLP branch, not a complete draft |
| Gate/up fusion, T8 eager | Native 0.240 ms; fused 0.761 ms | Correct but slower; not adopted |
| Fusion captured replay, simulator | Changing-input and stale-input controls pass on both chips | Correctness only; no hardware speed claim |
| Fusion captured replay, hardware | First launch failed before card access; exact runtime now restored | Hardware timing retry pending |

Evidence: [MLP trace](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34170293087),
[eager fusion](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34176158014),
[trace launch](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34178476409).
Fusion uses geometry-matched DFlash weights, not a full target-model quality test.

| Workstream | Current position |
| --- | --- |
| DFlash2 | Five-layer hardware correctness passes; full captured drafting and live request integration remain |
| EAGLE3 / DSpark / combined drafters | No validated throughput on this pair |
| KV usage | September 5: no zero occupancy in 4065 active-request samples; idle zero is expected |
| Prefill/decode disaggregation | End-to-end placement, scheduling and responsiveness tests remain |
| Ethernet dispatch / extra column | Hardware grid and fabric gates remain before performance claims |
| Coding quality | Held-out coding evaluation remains; numerical gates alone do not certify quality |

Next: measure captured fusion on the restored exact runtime, then pursue full-model
verifier savings and learned-drafter integration without disturbing other workloads.
PP and committed TG need a matched context/concurrency sweep once that path is ready.

## Guides and evidence

- [Experiment programme](docs/two-card-experiment-programme.md): gates, failures and dated results.
- [August 24 bring-up guide](docs/bringup-2026-08-24.md): historical build/launch instructions and longer-context results, including 256K. Not a current baseline.
- [Gotchas](docs/gotchas.md): configuration and runtime constraints.

Repository glue is [MIT licensed](LICENSE); derived patches retain upstream licensing.
See [NOTICE](NOTICE). Model weights are not redistributed here.
