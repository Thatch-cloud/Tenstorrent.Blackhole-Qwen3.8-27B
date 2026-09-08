# Qwen3.8-27B on two Tenstorrent Blackhole cards

Two-card inference and kernel experiments for fast, reliable coding responses.
**Target: 200 committed tokens/s for one coding stream. Not achieved yet.**
Experimental paths are opt-in; serving defaults remain unchanged.

**Latest completed result: 48.83 committed tok/s**, versus a matched native
**19.74 tok/s**. [Two-card MTP run](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34196777661)
finishes one coding response with all 150 decode tokens and final cache state exact.
This is a 2.47x decode improvement, not 200 TG or a held-out coding-quality score.
Setup is not amortized yet: the request including preparation remains slower than native.

**Running now:** [native-row sampling comparison](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34200129693).
It removes 32-row sampling padding from MTP and verification, with paired complete
requests and exact checks. Simulator reduction checks pass; hardware results are pending.

## Setup

| Component | Experiment setup |
| --- | --- |
| Accelerators | 2 × Blackhole P150A, 32 GiB each |
| Placement | Tensor parallel across a `(1, 2)` mesh |
| Card-to-card connection | Two QSFP-DD cables; four physical Ethernet links |
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

### Native MTP: complete coding response

Seven draft tokens, up to eight target verification rows, full-vocabulary device
argmax and cache repair after rejection. One stream, not eight batched users.
TG includes drafting, verification/readback and commit; excludes prefill/setup.

| CTX | Streams (B) | Path | Verify rows (T) | PP tok/s | Committed TG tok/s |
| ---: | ---: | --- | ---: | ---: | ---: |
| 170 | 1 | Matched native | 1 | 566.82 | 19.74 |
| 170 | 1 | Native MTP K7 | 8 | 570.35 | **48.83** |

PP measures the target prefill helper; MTP feature capture is included, draft
initialization is not. MTP accepts 125/182 proposals over 26 blocks and reaches EOS.
Tokens, active GDN, valid KV and inactive slots match the native reference exactly.
Prefill + unamortized setup + decode: **10.77 s MTP versus 7.90 s native**.
These totals start with the target model already loaded, not a cold process launch.
Trace/setup reuse across requests and longer-context measurements remain to do.

### Earlier lookup experiments

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

MTP now produces useful drafts, but the full cycle averages **118.10 ms** for
**5.77 committed tokens**. Measured mean costs in the completed request:

| Work per block | Time |
| --- | ---: |
| Target verification and readback | 66.29 ms |
| Seven sequential MTP drafts | 38.82 ms |
| Cache repair and commit | 12.04 ms |
| Input staging | 0.54 ms |

At that acceptance, 200 TG requires a cycle around **28.85 ms**, not 118 ms.
Even perfect T8 acceptance and free drafting cannot overcome the current verifier.
Next: remove unnecessary sampler rows and qualify parallel attention at the actual
short context. Both need exact checks and a complete-request hardware comparison.

The earlier September 8 coding requests expose weak lookup proposals.
All four requests reach EOS after 150 committed decode tokens and match native
tokens, active state and inactive slots. This is one task, not held-out coding-quality certification.

| CTX | Streams | Path | PP tok/s | Committed TG tok/s |
| ---: | ---: | --- | ---: | ---: |
| 170 | 1 | Native reference, paired candidate repetitions | Not isolated | 18.96 |
| 170 | 1 | Lookup capped at T32, norm batching on | Not isolated | 15.16 |
| 170 | 1 | Lookup capped at T8, norm batching on | Not isolated | 18.26 |

The T8 cap reduces proposals from 1812 to 572 per request, with the same 22 accepted.
That recovers 20.4% versus T32, but remains slower than the paired native reference.
TG excludes prefill/setup; T8 including setup is 13.67 tok/s.
[Completed coding request evidence](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34183834884).
The subsequent matched T8 request test measured **18.87 tok/s with one sampling
link versus 19.10 with four (+1.26%)**. All four requests completed with identical
tokens and state; setup-inclusive four-link TG was 14.60 tok/s. This is one ABBA
block, not a robust production speedup claim.
[Four-link request evidence](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34185881285).
This is not adoption of a learned drafter or a serving change.

Even perfect acceptance needs an eight-token draft/verify/commit cycle within **40 ms**.
Our best static target verification alone still takes longer:

| CTX | Verify rows (T) | Target verification block |
| ---: | ---: | ---: |
| 4095 | 8 | 62.35 ms |
| 16383 | 8 | 64.00 ms |

These are **block timings, not generated throughput**, excluding draft/commit overhead.
[Verifier run](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34150091732).

## Latest kernel and drafter results — September 8

Four-channel fabric sampling now passes exact argmax and input-preservation checks
on both cards, including ties and changing traced inputs. Three ABBA blocks per
comparison measured the following warmed sampler latency (not model TG):

| Comparison | One-link control | Candidate |
| --- | ---: | ---: |
| Two links | 2.671 ms | 2.098 ms |
| Four links | 2.702 ms | 1.827 ms |

This uses the `p150_x2` descriptor and a sampler-only override of the hardcoded
link helper. Serving defaults are unchanged.
[Fabric sampling evidence](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34185352446).

The five-layer captured-drafter gate is **not passed**. Local tensor-integrity
failures led to a CI simulator retry; that retry was cancelled, not qualified.
Earlier eager hardware checks do not establish a working captured drafter.

The native eager link-discovery bug is fixed in the disposable experiment build.
The [two-card check](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34189506734)
passes without the fallback warning. T8/T32 collectives show no useful speed gain;
this is a correctness fix, not progress to 200 TG by itself.

| Experiment | Measured outcome | Meaning |
| --- | --- | --- |
| Learned MLP, T8 captured replay | 1.911 ms median | One isolated MLP branch, not a complete draft |
| Gate/up fusion, T8 eager | Native 0.240 ms; fused 0.761 ms | Correct but slower; not adopted |
| Fusion captured replay, simulator | Changing-input and stale-input controls pass on both chips | Correctness only; no hardware speed claim |
| Fusion captured replay, T8 hardware | Native 0.211 ms; fused 0.201 ms | About 4.4% lower isolated latency; not adopted in full model |

Evidence: [MLP trace](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34170293087),
[eager fusion](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34176158014),
[captured fusion](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34179917205).
Fusion uses geometry-matched DFlash weights, not a full target-model quality test.
Captured measurements use three ABBA blocks with exact outputs on both chips;
they exclude uploads, allocation, capture and validation. The earlier launch failure
was resolved by restoring the exact runtime image.

| Workstream | Current position |
| --- | --- |
| MTP | Complete exact coding request: 48.83 TG; setup reuse and held-out/context tests remain |
| DFlash2 | Five-layer hardware correctness passes; full captured drafting and live request integration remain |
| EAGLE3 / DSpark / combined drafters | No validated throughput on this pair |
| KV usage | September 5: no zero occupancy in 4065 active-request samples; idle zero is expected |
| Prefill/decode disaggregation | End-to-end placement, scheduling and responsiveness tests remain |
| Ethernet dispatch / extra column | Hardware grid and fabric gates remain before performance claims |
| Coding quality | Held-out coding evaluation remains; numerical gates alone do not certify quality |

Next: reduce the measured verifier and drafting costs; acceptance is now measured.
The modest isolated fusion gain does not close the verifier gap.
PP and committed TG need a matched context/concurrency sweep after qualification.

## Guides and evidence

- [Experiment programme](docs/two-card-experiment-programme.md): gates, failures and dated results.
- [August 24 bring-up guide](docs/bringup-2026-08-24.md): historical build/launch instructions and longer-context results, including 256K. Not a current baseline.
- [Gotchas](docs/gotchas.md): configuration and runtime constraints.

Repository glue is [MIT licensed](LICENSE); derived patches retain upstream licensing.
See [NOTICE](NOTICE). Model weights are not redistributed here.
