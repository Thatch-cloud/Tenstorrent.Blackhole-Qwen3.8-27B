# Two-P150A execution ledger

Updated 2026-09-05. Target: 200 committed tokens/s for one coding stream, not
aggregate throughput. No adoption or serving restart is authorized by a test pass.

## Verified control

- Image: `sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465`.
- TT-Metal: `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9` with recorded graft changes.
- Plugin: `bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c`; vLLM `0.25.1+empty`.
- Weights/tokenizer: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, cache discovered
  and mounted directly read-only; full-model startup and K/M engagement verified.
- PCIe x16/x4 host attachment; inter-chip collectives use QSFP-DD/P300 fabric.
- Operator-confirmed exclusive card allocation; no host process-scan claim.
- Runner repository access and tested-ref exceptions remain enabled at operator request.

## Execution queue

| Track | Current status | Required next evidence |
| --- | --- | --- |
| Hardware prerequisite | Correctness-passed, run 33941853075 | Repeat for changed native kernels |
| E0 baseline | Benchmarked: run 33943034757 | 19.45 to 18.39 client-estimated tok/s at B=1; no-logprobs and engine timing still required |
| E1 cache | Occupancy observed, no active-zero recurrence | 0-12.4893% occupancy; full-model cache lifecycle gate remains required |
| E2 interleaving | Boundary and three-request mixed-traffic gates passed at all ratios | Repeated workload/long-context/load/cancellation sweeps; full device KV lifecycle remains unverified |
| E3 verifier | Historical harness not yet a correctness gate | Assert every accepted-prefix state and output, forced rejections; then measure T=1/2/4/8/16 |
| E4 fusion/pipeline | Planned, needs implementation | E3 verified state contract, one native change per arm, full-path timing |
| E5 drafting | Lookup proposal policy passes 11 host tests; integration dependency-gated | Passing E3 before enabling lookup/MTP; no non-greedy semantic substitution |
| E6 coding quality | Corpus not frozen | 200 independent executable fixtures, isolated code execution and paired outcomes |
| E7 prefix reuse | Dependency-gated | E1/E2 lifecycle gate; hybrid KV plus recurrent/conv state identity and isolation |
| E8 precomputation | Planned audit | Bound removable cost before table prototypes; no unapproved arithmetic changes |
| E9 spare-core work | Three eager layer profiles passed, run 33945221856 | Full-model traced attribution before reader/core-map/L1 changes |
| E10 disaggregation | Eight-slot TP1 pool fails analytical capacity bound | Smaller-pool TP1 and hybrid-state handoff remain unverified; E2 mixed-traffic control first |

The queue is not a claim that all tracks already have runnable implementations.
Run dependent arms only after their entry gates pass. Preserve failed and negative
results; do not convert skipped or blocked experiments into successful test counts.

## Baseline protocol

Manual workflow suite `baseline`, `cards_allocated=true`. Network-disabled disposable
container, loopback-only endpoint, read-only existing weight cache, separate labeled
experimental weight/kernel cache volume, unchanged K-image flags and host sampling.
No production volumes are written. Only the container created by the job is removed.

Warm every workload, then three alternating-order repetitions at concurrency 1 with
target prompt lengths 128/4096/32768/65536, capped below the 65536 context limit to
reserve 1024 output tokens. Actual templated lengths are recorded. Concurrency 2 and
8 at 4096 tokens are separate aggregate controls. Ignore-EOS is verified by output
counts, exact streamed token IDs cross-checked against usage, metrics scraped every 250 ms.
Client timing is labeled an estimate; no engine-token timestamps means this cannot
certify 200 committed tokens/s. Historical run 33943034757 used logprobs and token
strings; new runs request exact token IDs without logprobs by default. The old
results remain valid only for that original host-sampling workload.
Coding quality and greedy token-ID equivalence require their separate gates.

Initial full-model run 33942471680 reached readiness and logged K/M engagement,
but no throughput requests ran: Transformers returned a BatchEncoding rather than
a token list. The client now normalizes the actual input_ids and has regression
tests for both return shapes. Run 33943034757 retries the baseline with the frozen
snapshot and unchanged model flags. Do not count the failed warm-up as a benchmark.

The [payload feasibility estimate](decode-payload-bound.md) gives approximately
19.92 GB of dense projection value payload per ordinary decode step under current
mixed bfloat4/bfloat8 layouts. It is not measured traffic, but establishes the
approximately 3.98 TB/s payload requirement of a non-amortised 200-step/s path.

## Layer profile gate

Run 33944471561 passed GDN and MLP numerical checks with two-chip device timing
and no dropped markers. Some operations use all 110 available workers; this is
not a uniform 36-core workload. These eager fixture timings are not full-model
traced decode timings. Attention failed before completing its numerical check:
the stock fixture allocated BF16 KV while the shipped BF8 flag casts fill inputs
to BF8. The retry stages only the fixture cache allocation dtype to match that
flag, recording original/staged hashes. Model precision and PCC thresholds are
unchanged. The failed attention profile is not accepted as performance evidence.

Retry [33945221856](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33945221856)
passed all three fixtures on both chips with no dropped markers. Attention
paged-versus-concat PCC: prefill 0.999696, decode 0.999757. GDN position-zero
PCC 0.99985; MLP PCC 0.999230. No thresholds were relaxed.

Device-0 eager attribution: the GDN fixture's two steps used 333.358 us in four
matmuls (32/43 cores), 93.131 us in two recurrent kernels (24 cores), and 70.693 us
in two fused conv/gate kernels (81 cores). The MLP fixture's single step used
303.913 us in three matmuls (32/39 cores). Some copy/elementwise operations use
110 cores. Attention includes both prefill and decode, reference and paged paths;
its aggregate operation totals must not be called a single decode latency.

Local validation at commit 35a0d3f: 84 host tests passed (31 CI, 38 continuation/
interleaving, four stream rig, 11 lookup policy). These are not 84 silicon tests.
Interleaving run 33945305689 passed the control's eight boundary prompts and
post-cancellation output-ID check without requesting logprobs. The ratio-1 arm
failed before model execution because vLLM offline mode rewrote the canonical
model ID to the pinned local snapshot path. The guard now accepts only that exact
reviewed snapshot in addition to the canonical model ID; arbitrary model paths
and other revisions remain rejected. Ratio 2/4 did not execute. Historical
zero-cache reports remain observation-only unless current measurements reproduce them.

Retry 33945632554 again passed the control, then failed during ratio-1 API startup:
`Chunked MM input disabled but max_tokens_per_mm_item (16384) is larger than
max_num_batched_tokens (2048)`. This is multimodal admission/encoder budgeting,
not a KV-usage or kernel failure. Interleaving v1 is deliberately text-only but
the launcher had left image/video limits at their defaults. All interleaving
arms, including the control, now set image/video limits to zero and disable MM
embedding inputs. The 2048-token chunk budget and numerical gates are unchanged.
The prior baseline/profile suites retain their original settings.

Run 33945913888 confirmed the encoder-budget fix: the ratio-1 endpoint loaded and
became ready. Its first text request then hit `Continuation v1 is text-only`.
The pinned plugin represents an absent per-request image as `pixel_values=[None]`,
not only `None`. The continuation guard now accepts exactly one absent visual
row, while rejecting tensors, populated/nested rows, malformed row counts and
vision tokens. Regression cases cover both acceptance and rejection. This is
still a failing integration result, not a passed interleaving correctness gate.

Run 33946277920 progressed past the visual-row guard and failed on position
validation. The pinned runner builds prefill positions and prompt lengths as
NumPy integer arrays; our validator accepted only Python integers and torch
integer scalars. It now accepts `numbers.Integral`, normalizes to Python int,
and still rejects floating-point, Boolean and array-valued positions. The
integration regression now feeds the actual plugin visual-metadata gatherer
and NumPy position/prompt-length representations through `submit_prefill`.

Run [33946645240](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33946645240)
cleared all three runtime/configuration errors. Control and ratio-1 each completed
eight boundary requests and the post-cancellation check. Cross-arm comparison
correctly failed: the 2049-token prompt differs at generated token index 1
(the second output token). First four control IDs: `[74830,8,198,727]`;
ratio-1 IDs: `[74830,1590,198,262]`. All 32 IDs match for the other seven lengths
(63/64/65/2047/2048/4096/8193) and after cancellation. This is now a genuine
full-model equivalence failure, not a launcher error. Its cause is not yet
established; the matching first output token does not prove equal logits or state.
Do not relax the exact-output gate or adopt interleaving. Ratios 2/4 did not run.
Next diagnostic: repeat the isolated 2049 case and compare recurrent/conv/KV state
and logits around positions 2048/2049 and the first decode step. The previous
raw startup errors remain preserved in their original artifacts.

Diagnostic run 33947341479 localized the failure: for 2049-token prompts, ratio-1
records one 2048-token intermediate prefill, then enters decode at position 2048.
There is **no final GDN slot write**. The 8193-token case likewise has no final
slot write, despite previously matching output tokens by coincidence. The pinned
runner's `_is_still_prefilling` assumes one remaining token means decode. That
assumption is invalid for this prototype's separate B=1 prefill scratch: even a
one-token final prefill must execute and commit the request's GDN state to its
decode slot. The fix uses the existing request/epoch completion ledger only when
continuation is enabled; ordinary decode and the flag-off path keep their original
classification. A regression executes the transformed nested runner classifier
for a one-token tail, completed request, no remaining tokens, and flag-off mode.
Diagnostic reruns also gate three isolated 2049-token repetitions against control,
in addition to the original nine checks. Host fingerprint instrumentation performs
no additional device operations; its timings must not be used as speed evidence.

Fix verification [33947778844](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33947778844)
passed: ratios 1/2/4 each match the control on all 12 exact-token checks (eight
boundary prompts, three isolated 2049 repetitions, and post-cancellation output).
Each arm also has 13 paired diagnostic request records, including the cancelled
request. All host recurrent/conv snapshots supplied to the decode-slot writer
match the control byte-for-byte, as do the recorded final-prefill and first-three
decode logits. No missing final slot writes remain. This is not a complete direct
device KV snapshot or multi-request isolation certification.

Mixed-traffic run 33948332548 separately tests one 512-token decoder, an injected
8193-token prefill and a short coding request. It requires exact token-ID equality
against isolated requests and verifies actual request overlap. Boundary checks
remain enabled in every arm. Diagnostic hashing is disabled for this run, and
KV metrics continue to be collected. No production serving changes or automatic
adoption follow from these tests.

Mixed-traffic [33948332548](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33948332548)
passed all four arms. The live decoder, long request and short request each match
their isolated output IDs exactly; post-run comparison also matches all three
against the ratio-0 isolated control. Actual overlap and short-request arrival
before the long request's first token were verified. One smoke trial per ratio:

| Decode steps/chunk | Live decoder maximum gap (s) | p99 gap (s) | Client decode tok/s | Long TTFT (s) | Short TTFT (s) |
| --- | --- | --- | --- | --- | --- |
| Control, no interleaving | 3.116 | 0.055 | 17.671 | 2.590 | 2.813 |
| 1 | 0.654 | 0.556 | 17.470 | 2.836 | 3.308 |
| 2 | 0.646 | 0.552 | 17.249 | 3.078 | 3.605 |
| 4 | 0.647 | 0.546 | 17.148 | 3.517 | 4.132 |

The maximum interruption falls approximately 79%, but one long stall becomes
multiple chunk-sized interruptions: p99 worsens, and new-request TTFT increases.
This is a responsiveness tradeoff, **not** a decode-throughput improvement or
evidence of 200 committed tok/s. Ratio 1 is the next candidate for repeated
paired measurements, not an adopted default. Cache occupancy was nonzero
(peak 1.60-1.61%), with zero active-zero-cache samples across 1230 active samples
and zero scrape errors. Production serving and default-off experiment flags
remain unchanged.

## E9f: guarded TP2 greedy device sampling

Native sampler prerequisite [33949480633](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33949480633)
passed all 30 checks: real 248320-token vocabulary, random/boundary/tie/near-tie
cases, eager and two trace replays, on both chips. Median synchronized sampler
latency was 2.607 ms over 20 warm iterations. This is a 32-row sampler-only
measurement, not B=1 full-model throughput.

The pinned model disables generic device sampling on TP2 because each vocabulary
shard exceeds its TopK limit. The opt-in experiment enables only force-argmax,
with a runtime guard that fails before any generic TopK execution. Original
sampling eligibility checks remain intact; non-greedy requests, penalties,
multi-request batches and unsupported requests retain host sampling.

The full-model test uses three ABBA blocks per prompt length (approximately 128
and 4096 tokens), B=1 and 512 output tokens. Both arms omit explicit seeds to
permit internal greedy sampler tracing, and omit logprobs in timed requests.
Every output must match exact token IDs and text; device-path engagement is
recorded per request. A device-labelled logprobs request must fall back to host.
This is separate from the historical logprobs-enabled baseline. KV metrics are
collected throughout; experiment flags and production defaults remain unchanged.

Full-model attempt 33950052748 stopped safely during startup: the shared warmup
sweep requested non-greedy TopK. The experiment now passes the existing
`greedy_only=True` warmup option only for its TP2 model. Host/no-sampler warmup
remains included, and the runtime TopK guard is retained.

Full-model [33950377324](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33950377324)
passed at `dd42e2e`. All 24 measured requests completed 512 tokens and matched
their prompt-length control exactly (token IDs and text). All 14 device requests
(12 measured, two warmup) recorded force-argmax engagement with model trace mode
`all`; both logprobs negative controls correctly selected host sampling.

| Actual prompt tokens | Host mean client tok/s | Device mean client tok/s | Mean paired gain | Three-block gain range |
| --- | --- | --- | --- | --- |
| 109 | 19.850 | 21.317 | +7.389% | +7.334% to +7.455% |
| 4078 | 19.800 | 21.113 | +6.629% | +6.278% to +6.975% |

Means weight the three ABBA blocks equally. These are B=1 client estimates,
not engine-committed timing or evidence of 200 tok/s. Exact equality on these two
coding prompts is not a general coding-quality benchmark. KV occupancy peaked
at 0.8782%; no active-zero occupancy occurred across 2364 active samples out of
2454 scrapes, with zero scrape errors and zero preemptions.

Decision: retain guarded device force-argmax as a promising experimental candidate,
not a production default. Next gates are longer-context paired runs, explicit-seed
and non-greedy/penalty fallback checks on hardware, and full-model traced operation
attribution before changing projection grids or low-level GDN kernels. Multi-request
device sampling and composition with prefill interleaving are not certified here.

The `sampling-extended` suite next runs three ABBA blocks at approximately 32K
and 64K input tokens, still B=1 with 512 output tokens and exact ID/text gates.
Before timing it compares host/device-labelled requests with explicit seeds 123
and 456, seeded non-greedy sampling, each penalty type separately, and a final
greedy recovery request. Non-greedy and penalty cases must actually select host;
seeded greedy and post-fallback greedy must engage device force-argmax. These
checks change request parameters only, not eligibility or sampling mathematics.

Extended [33952227890](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33952227890)
passed at `9ca28d0`, following 47 passing local CI-helper tests. All seven
fallback/seed pairs matched exact IDs and text, with the expected actual sampler
selection; device greedy resumed correctly after penalty requests. Both long-context
logprobs negative controls also selected host and matched greedy control tokens.
All 24 long-context measured requests completed 512 tokens and matched their
prompt-length control exactly across three ABBA blocks:

| Actual prompt tokens | Host mean client tok/s | Device mean client tok/s | Mean paired gain | Three-block gain range |
| --- | --- | --- | --- | --- |
| 32752 | 19.231 | 20.522 | +6.709% | +6.553% to +6.828% |
| 64504 | 18.724 | 19.968 | +6.642% | +6.464% to +6.831% |

KV occupancy peaked at 12.3918%; zero active-zero observations across 2563 active
samples out of 4581 scrapes, zero scrape errors and zero preemptions. This is
exporter evidence, not direct certification of every attention/GDN state tensor.

The sampling improvement now reproduces at all four tested context lengths.
Explicit-seed checks establish correctness and path selection, not seeded throughput
or internal sampler-trace reuse. Timings still exclude logprobs and use client
stream events, not engine commit timestamps. Production defaults remain unchanged.
The next performance investigation is full-model traced operation attribution:
separate projection/GDN math, collectives, sampling and host dispatch before tuning
core grids. Existing eager module profiles cannot establish that critical path.
