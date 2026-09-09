# Qwen3.8-27B on two Tenstorrent Blackhole cards

Two-card inference and kernel experiments for fast, reliable coding responses.
**Target: 200 committed tokens/s for one coding stream. Not achieved yet.**
Experimental paths are opt-in; serving defaults remain unchanged.

## Current position

- **Best single stream: 78.06 TG at CTX 170, PP 510.65.** Captured DFlash2 T8,
  commit-only GDN and fused draft convolution improve the matched control by
  8.10%. [Measured run](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34246322267).
- **Repeat-confirmed 4K result: PP 3,324.52 / CTX 4,096 / TG 74.27**, versus
  61.28 TG control (**+21.21%**, pooling both runs). Native attention changes
  draft proposals only; all 12 requests retain exact target tokens and state.
  [Results](docs/drafter-numerics-experiment-2026-09-09.md).
- **8K:** the uncached repeat reaches 53.48 TG. This uses the older draft path,
  not the new 4K candidate; these are not scaling guarantees.
- **Main bottleneck:** the 4K target verifier takes about 62 ms/block, almost
  entirely on device. [Attribution](docs/current-verifier-profile-2026-09-09.md).
- **Attention result:** live-query attention is 7-10% faster in isolation, but
  the two complete 4K runs pool to **57.64 TG versus 59.27 control**. Both pass
  correctness; all stalls remain included. No promotion. [Details](docs/live-query-attention-2026-09-09.md).
- **Spare-core streaming result:** exact simulation and real-weight hardware
  correctness pass, but the complete MLP takes **0.744 ms versus 0.335 ms native**.
  It loses all nine comparisons and is not promoted. More active cores do not
  guarantee lower latency. [Results](docs/tensix-pooled-mlp-2026-09-09.md).
  [Device attribution](docs/tensix-mlp-device-profile-2026-09-09.md) locates the
  slowdown in gate/up/down, not the copy or collective.
  [Sixteen producers](docs/tensix-sixteen-producers-2026-09-09.md) improve the
  prototype to **0.456 ms**, but remain **36.17% slower than native**. All 188
  simulator and 118 hardware checks pass; no promotion or new TG result.
- **Fixed-packet reader result:** simulator and hardware correctness pass, but
  the complete MLP takes **0.425 ms versus 0.335 ms native: 26.92% slower**.
  It loses all nine comparisons and is not promoted.
  [Results](docs/weight-read-packets-2026-09-10.md).
- **Fabric result:** at CTX 4,096, four target-model links reach **61.41 TG versus
  62.39 control**; correctness passes, but no speed promotion. Sampler and drafter
  are unchanged. [PP / CTX / TG table](docs/target-model-link-counts-2026-09-09.md).

Smaller MLP tiles are [rejected: 4.70% slower](docs/tiny-tile-projections-2026-09-09.md).
Native drafter SDPA remains disabled by default; its exact numerical gate failed. All rates
below are offline coding-task experiments, not held-out quality or serving certification.

[Approximate draft attention](docs/drafter-numerics-experiment-2026-09-09.md)
cuts drafting from **40.32 to 20.35 ms/block** in the complete 4K comparison.
Acceptance stays **88.24%**, despite different proposals. The unchanged target
verifier still costs **61.72 ms/block**: this remains the main obstacle to 200 TG.
This is one coding prompt, not held-out quality or serving certification.

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

## Current lead: PP / CTX / TG

Captured DFlash2 T8 + commit-only GDN + fused convolution; **one stream**.
These are offline complete requests, not endpoint streaming measurements.

| PP tok/s | CTX tokens | Committed TG tok/s | Status |
| ---: | ---: | ---: | --- |
| 510.65 | 170 | **78.06** | Measured: two 150-token decode samples through EOS |
| 3,355.04 | 4,096 | **58.18** | Measured: two 121-token decode samples through EOS |
| 3,149.33 | 8,192 | **46.20** | First run: publication stall included; TG41.94 / 51.43 |
| 3,298.59 | 8,192 | **53.48** | Same-code repeat: two 121-token EOS samples; TG53.40 / 53.57 |
| 3,322.74 | 4,096 | **75.42** | New approximate-drafter candidate: two 121-token EOS samples; matched control61.47 |
| 3,326.31 | 4,096 | **73.16** | Same-code repeat: two 121-token EOS samples; matched control61.08 |
| — | 16,384 | — | Planned |
| — | 32,768 | — | Planned |
| — | 64,504 | — | Planned; reserves generation space below 65,536 |

At CTX 170, mean prefill is **0.333 s**; prefill + fresh setup + decode is
**6.42 s**, excluding model loading. TG excludes prefill/setup; PP includes
target feature capture and first-token selection, but not drafter initialization.
The drafter's rolling 2,048-token history is separate from the target's full KV
context. An opt-in 4K initializer now captures the valid tail and retains absolute
positions. The corrected 4K run passes native token/state and feature audits;
mean prefill is 1.221 s and prefill + fresh setup + decode is 7.05 s.
[4K hardware result](docs/dflash-long-context-2026-09-09.md).
The first 8K run has a 424 ms publication stall; it remains included in TG.
The same-code repeat measures 53.48 TG and 8.70 s mean complete request, versus
46.20 TG and 10.24 s initially. This is repeatability evidence, not a code gain.
[8K result and timing spread](docs/dflash-8k-context-2026-09-09.md).
[Matrix, measurement rules and next gates](docs/pp-ctx-tg-benchmark-matrix.md).

The new 4K approximate-drafter pair pools to **74.27 TG** across four timed
requests, with **6.65 s** mean prefill/setup/decode. It has not run at 8K or
higher.

**DSpark remains experimental.** One learned layer and full-vocabulary transport
pass targeted simulation. The [hardware integration run](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34412639592)
joins learned FC and all five layers in one trace, using four fabric links.
Target integration and the earlier numerical differences remain open; **no DSpark
TG result yet**. Checks now run before compilation, and unchanged native builds
are cached. [Evidence and next gates](docs/dspark-learned-layer-2026-09-10.md).

The separate matched4K cache experiment measures **PP3,307.88 /CTX4,096 /TG60.33**,
against uncached **PP3,293.42 /TG58.81**. Publication overhead consumes most of
the drafting saving; caching is not enabled in serving or the8K ladder rows.
Capturing cache updates separately passes correctness but is effectively flat:
**PP3,292.61 /CTX4,096 /TG62.11**, versus cached eager-update **62.01 TG**.
That0.15% difference is within timing spread; it is not a performance promotion.

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
| 170 | 1 | Native reference, all paired repetitions | 1 | 546.69 | 19.47 |
| 170 | 1 | Host-stepped MTP K7, KV-only repair | Up to 8 | 566.80 | 57.24 |
| 170 | 1 | Device-chained MTP K7, KV-only repair | Up to 8 | 534.26 | **58.33** |

PP measures the target prefill helper; MTP feature capture is included, draft
initialization is not. Both arms accept 125/178 proposals in 26 blocks.
Each produces 150 committed decode tokens
and reaches EOS; two repetitions per arm run in ABBA order. This is one coding task.
Tokens, active GDN, valid KV and inactive slots match the native reference exactly.
Mean prefill + unamortized setup + decode: **7.60 s host-stepped versus 9.27 s chained**.
These totals start with the target model already loaded, not a cold process launch.
Trace/setup reuse across requests and longer-context measurements remain to do.
The chain cuts drafting from 27.15 to 25.15 ms/block. Its first setup takes 3.58 s;
the second takes 0.16 s. Neither is treated as free. Earlier exact KV-only repair
improved 55.07 to 56.85 TG (+3.25%). PP variation is not attributed to decode changes.
Earlier approximate
cache reuse lost 7% TG through lower acceptance; repaired parallel attention was also slower
(54.36 versus 54.90 TG); its mask correctness fix remains separate from speed.
The earlier [sampling comparison](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34200129693)
established +7.7%; changes between separate runs are not attributed to an optimization.

### DFlash2: complete coding response

All five learned layers, shared target embedding/head and committed feature history.
TG includes drafting, verification/readback and publication; excludes prefill/setup.

| CTX | Streams (B) | Verify rows (T) | PP tok/s | Committed TG tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 170 | 1 | Up to 8 | 568.82 | **37.64** |
| 170 | 1 | Up to 32, trained-width extrapolation | 554.23 | **41.03** |
| 170 | 1 | Up to 8, captured drafter | 517.11 | **66.76** |
| 170 | 1 | Up to 32, captured drafter and trained-width extrapolation | 542.94 | **60.74** |
| 170 | 1 | Up to 8, captured ABBA control | 496.89 | **65.50** |
| 170 | 1 | Up to 8, captured + commit-only GDN | 530.32 | **70.34** |
| 170 | 1 | Up to 8, convolution ABBA control | 517.89 | **72.21** |
| 170 | 1 | Up to 8, captured + commit-only GDN + fused convolution | 510.65 | **78.06** |

Two complete 150-token responses reach EOS at 37.11 and 38.18 TG. Each accepts
129/154 proposals in 22 blocks. Tokens, GDN, valid KV and inactive slots are exact;
a separate audited request checks every committed feature row on both chips.
Prefill + unamortized setup + decode takes **8.24 / 8.17 s**, with the target loaded.
This is one coding task, not held-out quality or a matched comparison against MTP.
[Hardware run and artifacts](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34232609121).

The opt-in 32-row test extrapolates beyond the trained eight-token block. It
completes two exact 150-token requests at 40.79/41.26 TG, 17 blocks each.
Complete prefill/setup/decode takes9.82/9.22s. The first attempt exhausted DRAM;
reserving KV pages for one stream fixes that without reducing its65,536-token
capacity. This is not a matched T8 comparison or an eight-stream configuration.
[Wider hardware run](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34236992387).

Captured T8 preserves the same129/154 acceptance and22 blocks in this task.
Its two complete requests take7.34/7.43s including fresh prefill and setup;
their decode-only rates are68.41/65.20 TG. This also changes context padding and
KV allocation, so it is not a matched capture-only attribution.
[Captured hardware run](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34238134003).

Captured T32 completes at61.26/60.24 TG, with18 blocks/request. It costs98.44 ms
per verification block versus65.27 ms at T8, without enough extra accepted tokens
to compensate. It is not promoted over the trained-width T8 candidate.
[Captured T32 run](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34239197088).

Commit-only GDN removes speculative state writes before acceptance. The matched
ABBA test preserves all proposals and outputs: 150 tokens through EOS, 22 blocks,
129/154 accepted drafts. Candidate prefill + fresh setup + decode takes
**6.90 / 7.06 s**, versus control **8.15 / 8.59 s**; setup is not amortized.
Two separate audit requests are excluded from TG. PP/setup variation is not
attributed to the decode change. Serving defaults remain unchanged.
[Matched hardware run](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34242926044).

Fused convolution improves the matched control by 8.10%, with the same 150
tokens through EOS, 22 blocks and 129/154 accepted drafts. Complete prefill +
fresh setup + decode takes **6.56 / 6.29 s**, versus control **7.03 / 8.01 s**.
Its audit checks all 20 learned convolutions per proposal on both chips, in
addition to the existing token, feature and cache checks. Audits are excluded
from TG; no setup amortization or held-out coding-quality claim.
[Fused-convolution hardware run](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34246322267).

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

MTP now produces useful drafts, but the chained candidate cycle averages **98.87 ms** for
**5.77 committed tokens**. Measured mean costs in the completed request:

| Work per block | Time |
| --- | ---: |
| Target verification and readback | 65.04 ms |
| Device-chained MTP drafts | 25.15 ms |
| Cache repair and commit | 7.77 ms |

At that acceptance, 200 TG requires a cycle around **28.85 ms**, not 99 ms.
Even perfect T8 acceptance and free drafting cannot overcome the current verifier.
Next: reduce sequential drafting/repair and develop wider parallel proposals to
amortize target verification. Short-context attention and earlier larger-grid MLP
sweeps did not improve whole-request TG; repeating them is not the next step.

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

The complete captured five-layer drafter now passes on hardware through EOS,
including22 exact eager-versus-trace proposal audits. Earlier cancelled simulator
attempts and local tensor-integrity failures are retained as failed evidence,
not retroactively counted as passes.

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
| MTP | Device chain + KV-only repair: 58.33 TG, +1.90% matched; setup-inclusive latency is worse; wider parallel proposals needed |
| DFlash2 | Complete exact captured T8: 66.76 TG; captured T32 is slower at60.74 TG; target-verifier optimization is next |
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
