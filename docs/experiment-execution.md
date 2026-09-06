# Two-P150A execution ledger

Updated 2026-09-06. Target: 200 committed tokens/s for one coding stream, not
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

### E3 attention shared-page gate (implementation, hardware result pending)

Source-only run 34000023393 exported the actual K-image fused attention source
(`attention/tp.py`, SHA256 `e0c685a43796f6f8a0ba42fd70a9533b502461b50fdda15e51c8753340f3dc3a`).
Its decode batch means independent users, with one KV writer core per user.
The paged update validator explicitly rejects `share_cache`. This is a shared-page
write hazard to avoid, not evidence that production B1 cache writes are broken.

The new `attention-batch` suite batches native fused QKV preparation, attention and
output projection while replacing only the two paged writes with ordered B1 calls
through an instance-local tail function. Installed source and global TTNN functions
are untouched. Two seeds, T=1/2/4/8/16 and start positions 31/63/65 cover tile/page
boundaries with nonzero BF8 caches and a nonidentity shared page map. Eager and trace
outputs plus the entire physical K/V pool must match native sequential B1 exactly
on both chips; omitted writes and wrong-page writes must be detected. Static host
position/page fixtures are not a device-dynamic verifier or serving integration.
This gate deliberately makes no throughput claim; numerical drift blocks promotion.

| Track | Current status | Required next evidence |
| --- | --- | --- |
| Hardware prerequisite | Correctness-passed, run 33941853075 | Repeat for changed native kernels |
| E0 baseline | Benchmarked: run 33943034757 | 19.45 to 18.39 client-estimated tok/s at B=1; no-logprobs and engine timing still required |
| E1 cache | Occupancy observed, no active-zero recurrence | 0-12.4893% occupancy; full-model cache lifecycle gate remains required |
| E2 interleaving | Boundary and three-request mixed-traffic gates passed at all ratios | Repeated workload/long-context/load/cancellation sweeps; full device KV lifecycle remains unverified |
| E3 verifier | Full-model serial rollback/logit/KV oracle passed, run 33999532634; layer GDN batching and compact DMA separately validated | Build alias-safe batched attention/target verification; compare to the oracle, then measure T=1/2/4/8/16 verification and commit cost |
| E4 fusion/pipeline | Full-model exactness passed, run 33996306217; repeatability gate missed | No serving promotion; retain 91-core kernel experiment and E3 state prerequisite |
| E5 drafting | Lookup policy host-tested; MTP historical integration; DFlash2/DSpark checkpoint/config review only, not card-tested | Shared E3 verifier/rollback gate, then matched MTP/lookup/DFlash2/DSpark runs; no non-greedy semantic substitution |
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

The `model-profile` probe loads every model layer through the served Qwen Generator,
retaining max-batch 8, 64K context and the observed 8200-block KV pool. Only the B=1
host-sampling decode bucket is warmed/captured; HTTP, scheduling and other resident
batch traces are excluded. It checks the first 16 tokens against endpoint run
33950377324, flushes device profiler buffers between decode steps, and requires
15 complete trace replay sessions on both chips. Attribution filters by the actual
decode trace ID, excluding eager compilation and prefill. Summed kernel durations
are attribution evidence, not non-overlapping critical-path time or tok/s.

Attempt 33954595293 completed all 16 exact-token checks on the full 64-layer
Generator, but is not a passing profile. Warmup overflowed profiler buffers and
the report fell into device-only Python analysis of an 11.5 GB raw log; it was
cancelled after exceeding its inner timeout. No kernel bottleneck conclusion is
drawn from that attempt. The retry explicitly enables host operation metadata
before Tracy imports, uses partial Python tracing and runtime C++ analysis without
the large raw CSV dump, and adds a hard kill-after timeout. Warmup overflow warnings
are reported separately: any overflow during the explicitly bounded decode region
still fails, as do missing or inconsistent replay rows on either chip.

Retry 33957235279 again passes exact generation and now captures host operation
metadata, but the all-operations report correctly rejects missing warmup data
(operation 11738113, device 1). The next revision scopes the pinned report join to
the recorded decode trace before joining device measurements. Its existing missing
operation assertions remain active for every selected operation; the final checker
also retains both-chip, replay-count, per-replay coverage and measured-overflow gates.
Profiler CSV sidecars are preserved explicitly even on failure, rather than relying
on hidden-directory artifact upload. No model/kernel arithmetic is modified.

Full-model trace [33957724963](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33957724963)
passes at `f62c298`: all 64 layers, exact first-16 endpoint tokens, 15 measured
replays on each chip, and 1433 operations per replay. Native C++ operation counts
also match the joined report by chip, replay and operation name. No marker drops
occur in measured decode; 1100 excluded warmup warnings remain explicitly recorded.
This certifies the selected decode evidence, not the discarded warmup profile.

| Operation group | Chip 0 summed kernel ms/step | Chip 1 summed kernel ms/step |
| --- | --- | --- |
| Matrix multiplications | 32.069 | 32.051 |
| GDN recurrent kernel | 2.256 | 2.255 |
| GDN convolution/gates | 1.699 | 1.699 |
| All-gather | 1.999 | 1.770 |
| Reduce-scatter | 1.413 | 1.606 |
| All operation groups | 42.670 | 42.657 |

Each entry is the median of per-replay summed durations; the last row sums those
group medians. Matmul accounts for approximately 75% of this attribution budget,
GDN recurrence plus convolution/gates for 9.3%, and the two collectives for 8%.
Do not sum chips or equate these sums to non-overlapping critical-path latency.

Largest matrix shapes, chip 0 (chip 1 agrees closely):

| Weight dimensions (padded) | Calls/step | Weight dtype | Cores | Summed kernel ms/step |
| --- | --- | --- | --- | --- |
| 5120 x 8704 | 128 | BFLOAT4_B | 39 | 11.803 |
| 8704 x 5120 | 64 | BFLOAT8_B | 32 | 7.819 |
| 5120 x 8256 (8240 logical) | 48 | BFLOAT8_B | 43 | 5.754 |
| 3072 x 5120 | 64 | BFLOAT8_B | 32 | 2.984 |
| 5120 x 7168 | 16 | BFLOAT8_B | 56 | 1.939 |
| 5120 x 124160 | 1 | BFLOAT8_B | 108 | 1.769 |

Decision: prioritize MLP gate/up and down projection grid/blocking experiments,
then the GDN input projection. Preserve the existing BF4/BF8 weight dtypes and
accumulation fidelity; extra cores are not automatically a bandwidth improvement.
The vocabulary projection already uses 108 cores, disproving a universal 36-core
limit. Next gates are real-shape numerical checks and synchronized traced kernel
A/B timing before any full-model adoption. No speedup beyond the earlier sampling
result is claimed from profiling. The host scheduler and device-sampling path
remain outside this probe. The enormous raw host-times CSV is not preserved in
future runs; final operation reports, native C++ data and host operation metadata
remain available. Local CI-helper validation now includes shape/coverage checks.

The `mlp-sweep` gate loads real layer-0 weights and compares the frozen production
44/33 gate/down grid budgets against separate larger/smaller grids and K-block
sizes 4/16 (control 8). It preserves BF4 gate/up, BF8 down, LoFi, FP32 destination
and packer L1 accumulation. Fifteen configurations run three seeded B=1 vectors
against the existing Torch PCC threshold (0.97) and exact baseline outputs, eager
and traced. All compilation precedes trace capture. Three ABBA blocks per candidate
time 20 replay submissions followed by synchronization per sample. Promotion to
a full-model gate requires exact control equality and over 2% lower latency in
every block; this is only a single-layer screen, with replicated DRAM input and
both-card reduce-scatter, not a serving-throughput or coding-quality benchmark.

Grid/block sweep [33958859849](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33958859849)
passes at `50d63d7`: all 90 eager/traced seeded outputs match control exactly;
Torch PCC ranges 0.999203-0.999417. No candidate clears the predeclared 2% latency
gate in every block. The best, gate grid budget 110, reduces whole-layer latency
only 1.691% (about 0.338 to 0.332 ms). Larger down grids are 0.48-2.43% slower;
K blocks 4 and 16 are 3.96% and 5.72% slower. The frozen control remains unchanged.

Follow-up `mlp-packing` keeps the control grid budgets but separately sets gate/up
or down output tiles per core to 8, 12 or 16, permitting a four-tile FP32 subblock
instead of the control's one-tile subblock. It retains the same precision, seeded
exact-output and paired timing gates. This tests tile packing, not a promise that
more active cores improve bandwidth.

Packing [33959155664](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33959155664)
passes at `86b9d83`, with all 42 seeded eager/traced outputs exactly equal to control.
No candidate meets the promotion gate:

| Change from frozen control | Mean whole-layer latency change |
| --- | --- |
| Gate/up 8 output tiles/core | -0.062% (one block slower) |
| Gate/up 12 output tiles/core | +2.300% |
| Gate/up 16 output tiles/core | +4.204% |
| Down 8 output tiles/core | +1.686% |
| Down 12 output tiles/core | +4.354% |
| Down 16 output tiles/core | +8.817% |

Across the two runs, 20 non-control configurations were screened without changing
weight or accumulation precision. None qualifies for full-model adoption under
the existing rule; the 44/33 control remains the default. These single-layer
results do not prove DRAM saturation, but they do rule out a large gain from the
tested grid/K-block/output-packing changes. They are not a full-model speedup.
Next investigation: combine gate/up work or reduce surrounding memory transfers,
with weight-layout/extra-allocation accounting and exact-output gates before
serving tests. Do not extrapolate a one-layer gain to the 200 tok/s objective.
Local CI-helper suite: 56 tests pass; hardware access and serving remain unchanged.

## GDN upstream parity audit

Read-only source audit [33960203974](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33960203974)
passed. The image includes ctxbot's fused recurrent kernel hardening plus packed
QKV, native norm/gate folding and broadcast rank-one updates. No missing listed
core fusion fix was identified; do not overwrite these extensions with the PR.
See [source comparison and next experiment](gdn-source-audit-2026-09-05.md) for
exact upstream heads, evidence limits, two failed audit attempts and the proposed
active-prefix state-write experiment. B1/Bmax8 currently cannot use the existing
in-place flag. No new throughput result or production change is claimed.

Offline attribution of the existing 33957724963 hardware trace now isolates the
three recurrent-state copy stages across all 48 GDN layers, 15 replays and both
chips. Median summed copy time is 0.548383/0.548504 ms per step on chip 0/1,
about 1.285%/1.286% of summed kernel time (not critical-path time). Native call
coverage, chain adjacency and BF16 state shapes are checked. Active-prefix state
writeback remains a secondary candidate; combined gate/up projection work takes
priority. The analyzer is added to future full-model profiles; eight regression
tests pass. This is new analysis of old measurements, not a new speed benchmark.

## Fused gate/up decode probe

The `mlp-fusion` suite compares the frozen real-weight layer-0 MLP with native
`minimal_matmul(fuse_swiglu=True)` using tile-pair-interleaved gate/up weights.
It retains BF4 gate/up, BF8 down, LoFi FP32 destination and packer L1 accumulation;
fusion can still alter rounding/order, so exact control output remains mandatory
for promotion. Three candidates use M/K blocks 1/8, N blocks 8/16/32 and grid
11x2. The existing minimal matmul partitions M across grid rows, so adding rows
for B1 is not automatically useful N-parallel work. No custom kernel is added.

The timed path includes the original input-to-L1 transfer, unchanged down
projection and TP reduction. All candidates compile before traces; three seeds
are checked eager and traced against Torch PCC >=0.97 and exact control output.
Three ABBA blocks with 20 replays/sample require >2% lower latency in every block
and exact outputs before any full-model gate. A non-exact result is not promoted.
An additional packed layer weight is allocated only inside the experiment;
host packing is checked reversible and no weight cache is written. Full-model
duplicate-weight memory capacity remains unverified. This is not serving adoption.

Attempt 33962274337 passed the three eager control seeds but rejected the first
fused candidate before execution: the operator requires grid dimensions >=2x2.
Retry uses 11x2 rather than 11x1, without relaxing the native validation. No fused
correctness or timing result is claimed from the rejected attempt.

Retry [33962376720](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33962376720)
completed at `460959c`. All three seeds passed Torch PCC >=0.97 eagerly and under
trace for control and every candidate (24 checks total). All 18 candidate
seed/mode comparisons failed bit-exact equality against control. Candidate PCC
was 0.999201477-0.999417543, close to control's 0.999202549-0.999416769;
this is not evidence of coding-quality equivalence or degradation.

| Fused N block | Mean layer latency increase | Three-block range | Approximate candidate ms |
| --- | --- | --- | --- |
| 8 | +70.565% | +70.481% to +70.674% | 0.576 |
| 16 | +68.031% | +67.801% to +68.285% | 0.568 |
| 32 | +57.923% | +57.889% to +57.942% | 0.535 |

Control was approximately 0.338 ms. No candidate qualifies on either exactness
or speed; no full-model arm is warranted for these configurations. Execution
success is not an optimization pass. This does not rule out a decode-specialized
fused kernel: the existing implementation uses a 2D M/N partition instead of the
control's 1D N-parallel map. Its SwiGLU stage multiplies gate/up in destination
registers before the final pack, unlike control's separately rounded BF16 gate
and up intermediates. These source differences are hypotheses for the result,
not experimentally isolated causes.

Next native candidate must preserve the control's 1D N-parallel distribution,
keep complete gate/up tile pairs together, reproduce BF16 intermediate rounding
and account for packed-weight allocation before repeating the same gates. Do not
extend the negative prefill-oriented grid sweep or relax precision/exactness to
claim a win. The 65 local CI-helper tests pass; production remains unchanged.

## Decode-specialized 1D gate/up prototype

`projection-1d` uses a separate `generic_op` program, not the prefill-oriented
minimal matmul. The native 1D compute source is copied into a source-code kernel
descriptor; its final-output branch and a post-loop epilogue change. The original K-block loop and
FP32 packer-L1 partial accumulation remain. No installed source or binary is edited.
Native and generated compute-source SHA256 values are recorded in the result.

The program maps 272 output-tile pairs to the same 39 N-workers (seven pairs/core,
six on the last), within an 11x4 rectangle. An activation-multicast reader includes
the five non-compute receiver cores. Separate weight-reader/output-writer code
streams pair-interleaved BF4 weights, including zero padding on the last worker,
and writes compact BF16 output directly. These readers are new code, not a claim
of byte-identical native reader scheduling. K-block size stays eight; paired
subblocks have two tiles instead of control's single tile.

The epilogue runs native pack-side SiLU on the gate tile only, packs gate and up
to a private fourteen-tile BF16 circular buffer (seven pairs per worker), then multiplies those rounded values
and packs BF16 output. Packed device weights must dequantize bit-identically to
the original gate/up tensors on each chip. This establishes the intended rounding
contract, not a correctness claim before silicon testing.

First gate is projection-only (no down projection or CCL reduction): three real
layer-0 input seeds, both chips, eager and traced exact comparisons, finite/PCC
diagnostics, then three ABBA blocks with 40 trace replays/sample. Inputs already
reside in L1 for both arms. Only exact output and >2% lower latency in every block
permit a whole-MLP experiment. Preserve non-exact/negative results and do not
replace the serving path on the strength of this prerequisite test.

### Initial 1D silicon attempts

All runs below are isolated projection probes, not full-MLP or serving results.
Execution/PCC success is separate from the exact-output promotion gate.

| Run | Outcome |
| --- | --- |
| 33963133460 | Descriptor setup failed: unpack-mode vector constructor not exported; use the exposed property instead. |
| 33963238683 | Descriptor validation required 64 unpack-mode entries rather than 32. |
| 33963346556 | Inline per-pair product interfered with subsequent native matmul state; first comparison PCC 0.95914, stopped before timing. |
| 33963532517 | Deferring product until after all matmul work restored PCC above 0.99988, but LoFi FPU multiply left 8,040–8,121 mismatches per 8,704-element shard. Projection latency increased 83.89–85.39%. |
| 33963794203 | SFPU product JIT failed because its binary API header was missing. |
| 33963923741 | Explicit SFPU API plus waiting for all rounded tiles reduced discrepancies to 52–83 elements per shard, maximum absolute error 0.015625. Still not exact. Control 0.19695–0.19733 ms versus candidate 0.36276–0.36490 ms, 83.83–85.29% slower. Not eligible for whole-MLP testing. |

The next probe isolates rounded gate and up outputs independently before checking
their product. It also corrects the new readers to the native 1D NoC assignments:
input multicast on NoC 1, weight reads on NoC 0. NoC 1 multicast endpoints reverse,
but receiver acknowledgements still target the original sender. Compare only the
ordinary product arm in timing; diagnostic intermediates compile before traces.

Run 33964284985 confirmed all twelve rounded gate/up comparisons exact across
three seeds and both chips. Product mismatches were unchanged. With corrected NoC
routing, candidate projection latency was 0.20523–0.20580 ms versus control
0.19672–0.19693 ms: only 4.33–4.50% slower, rather than approximately 85% slower.
This remains a rejected candidate, but identifies a substantial reader defect.

The native Blackhole `calculate_sfpu_binary_mul` explicitly applies BF16
round-to-nearest-even and zero-input handling when its arithmetic template flag
`is_fp32_dest_acc_en` is false. Calling the ordinary SFPU API inside our FP32
matmul kernel passes true and skips that narrowing. The next candidate keeps
FP32 destination addressing/accumulation for the matmul but calls the same native
multiply calculation with its BF16 narrowing branch enabled. No accumulation
precision is lowered, and the existing exact-output gate remains mandatory.

Run 33964555688 confirmed the correction: every rounded intermediate, eager
product, and traced product comparison was exact on both chips for all three
seeds. Projection latency remained 5.00–5.10% above control (0.20688–0.20733 ms
versus 0.19695–0.19732 ms). Correctness is established only for this bounded gate;
there is still no performance promotion or whole-model quality claim.

The subsequent isolated grid sweep retains this arithmetic, K-block eight,
two-tile subblocks and NoC routing. It compares 39/55/68/91 N-workers using
7/5/4/3 pairs per worker respectively. Only the N partition, corresponding buffers
and multicast receiver rectangle vary. All candidates compile before trace
capture and each receives its own three ABBA timing blocks and exact-output gate.

Run 33964688595 retained exact outputs for every grid but promoted none:
39 cores +13.22–13.30%, 55 cores +96.00–96.54%, 68 cores +11.38–11.70%,
91 cores +14.29–14.56% projection latency versus control. The 39-core arm itself
regressed relative to the preceding single-candidate probe. Generalizing the
reader had inadvertently changed its inner tile-loop bound from compile-time
to runtime; the retry restores compile-time specialization and compact core
ranges before drawing conclusions about the additional cores.

Run 33964808840 restored the 39-core control candidate to +5.07–5.16%. All
four candidates remained exact. The 55-core variant was +85.50–85.57%; the
68-core variant improved 1.94–2.06% but failed the >2% requirement in one block.
The 91-core variant passed the prerequisite: 0.18833–0.18882 ms and 4.07–4.32%
lower projection latency in every block. This is not end-to-end tok/s.

`projection-1d` now conditionally follows its projection probe with the existing
whole-MLP harness. It independently verifies each selected candidate's twelve
eager/traced shard checks and all three timing blocks, then requires an identical
compute manifest. Rejected candidates are not forwarded. Whole-MLP testing keeps
the native BF8 down projection, DRAM input/output and TP2 reduction; three seeds,
Torch PCC checks, exact native-control outputs and paired traces still apply.

Run [33964993645](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33964993645)
repeated the projection gate and completed the conditional whole-MLP test:

| Stage | Result |
| --- | --- |
| 91-core projection | Exact on both chips, eager and traced, all three seeds; 4.24–4.47% lower latency across all three paired blocks. |
| Complete layer-0 MLP, including down projection and TP2 reduction | Exact native-control output for all three seeds, eager and traced; Torch PCC 0.99920–0.99942, unchanged from control. |
| Complete MLP latency | Control 0.33735–0.33777 ms versus fused 0.32892–0.32962 ms; 2.37–2.50% lower latency, mean 2.447%. |
| Promotion | Eligible for the full-model gate only. No serving default changed and no end-to-end tok/s result claimed. |

The 68-core arm again missed the >2% projection requirement in one block and was
not forwarded; 39 and 55 cores remained slower. Local helper validation passed
75 tests, with shell syntax and whitespace checks passing. These results correct
the prior inference from the prefill-oriented 2D kernel: properly routed,
rounding-preserving 1D fusion can improve this decode workload, although the
measured gain is small and does not establish the 200 tok/s target.

Next gate: opt-in full-model integration with pair-packed weights replacing
(not retaining alongside) gate/up allocations, source/precision checks on every
layer, trace-safe B1-only dispatch and native fallback elsewhere. Verify output
tokens and memory headroom at short and long contexts before paired serving
timings. Do not extrapolate the layer microbenchmark to end-to-end speed or
claim coding-quality coverage from three layer-input seeds.

## Full-model fusion gate (2026-09-06)

Source inspection found an important allocation detail: the native TP model already
loads a pair-packed `w_gate_up` tensor for its fused prefill path. The full-model
experiment reuses that existing allocation instead of building another copy.
Native `w1`/`w3` remain available for unmodified fallback, with no weight-memory
increase relative to the current baseline. This replaces the preceding proposed
weight-replacement approach; hardware must verify all 64 layers and both shards.

`full-model-fusion` is an isolated Generator test, not a serving deployment.
It checks pinned compute/reader hashes, BF4 gate/up and BF8 down precision,
dequantized packed-weight equality for every layer/shard, and unchanged weight
addresses. It records per-chip DRAM/L1/trace allocator snapshots. Both arms share
the model and its KV pool but use separate decode trace stores. All decode kernels
compile before trace capture; prefill always uses the native path. Only explicit
B1 decode inputs select the fused path, with capture engagement checked on all
64 layers. No installed source or serving default is patched.

Predeclared work: short (~109 tokens) and long (~64,504 tokens) coding prompts,
16-token eager/traced token and full-logit equality checks, then three ABBA blocks
of 128-token generations per context. Timing includes host argmax and readback,
excludes prefill and HTTP/scheduler overhead, and counts 127 decode steps rather
than counting the prefill-produced first token as decode. Every request resets
state through native prefill. Require exact outputs and >2% lower latency in every
block before marking eligibility for a subsequent serving gate. Preserve failures
and smaller/negative gains; none establishes 200 committed tok/s.

### Completed run 33996306217

[CI evidence](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33996306217),
code `84db0ab`: execution/correctness passed, `eligible_for_serving_gate=false`.
All six eager/traced logit/token checks were exact, as were paired generation
tokens and packed weights on all 64 layers/both chips. Weight addresses were
unchanged and additional weight allocation was zero.

| Context | Control host tok/s, three blocks | Fused host tok/s, three blocks | Decode latency change |
| --- | --- | --- | --- |
| Short (~109 input) | 19.576 / 19.652 / 19.713 | 19.789 / 20.245 / 20.067 | -1.076% / -2.931% / -1.766% |
| Long (~64,504 input) | 18.410 / 18.626 / 18.201 | 18.406 / 18.705 / 18.746 | +0.018% / -0.421% / -2.907% |

Only two of six blocks clear the predeclared >2% latency threshold. The exact
kernel is viable but the full-path improvement is not repeatable enough for
promotion. These are Generator timings with host sampling, not the separate
device-argmax serving arm or engine-committed throughput. No serving change made.

### Multi-drafter groundwork

Added lookup-first routing to one explicitly selected neural callback, with native
target fallback when mode/verifier readiness is unsupported or no proposal exists.
Nine new host tests cover selection, skipped neural work, invalid proposals,
explicit commit, failures and isolation, alongside the existing eleven lookup tests.
Callbacks are test doubles: no DFlash2, DSpark or EAGLE3 device adapter is claimed.
The E5 plan now separates routing, adaptive selection, cascades and tree ensembles.
The shared exact E3 verifier and device state lifecycle remain the next dependency.

## E3a native GDN prefix gate (2026-09-06)

Added the `gdn-prefix` hardware suite. The historical multi-token implementation
used older composed conv/gate/norm operations, so it is not reused as the current
control. The new candidate calls the audited native packed input projection once
for T rows, then supplies independent row copies to the unchanged fused native
decode remainder. Output projections and collectives remain per token in this
first isolation test; it is not a complete batched verifier or optimal kernel.

The gate loads one real GDN layer, retains the pinned K-image BF16 recurrence and
decode settings, and uses B1 within an eight-slot state allocation on both chips.
Nonzero priming precedes three seeds at T=1/2/4/8/16. Both eager and trace replay
must match every native sequential output and recurrent/conv snapshot exactly.
Each prefix 0..T is restored in place, followed by two correction-input steps,
with exact outputs and final state required. Deliberately restoring stale
recurrence or convolution separately must be detected in state and continuation.

Device snapshots are preallocated outside trace capture and all live state
addresses must stay fixed. Timings include snapshot copies and are diagnostic
only. Attention KV, full-model logits, EOS/cancellation/slot epochs and all-layer
verification remain later gates; no host test or layer pass certifies those.

Run [33997659654](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33997659654),
code `b4da5c4`, passed all 216 prefix/continuation cases (both chips, three seeds,
five row widths, eager/trace) and all 30 stale recurrence/convolution negative
controls. This establishes input-projection batching with native fused GDN state
evolution for this layer fixture, not a complete verifier.

Next `gdn-block` arm batches the output projection and TP reduction as well. It
extracts the pre-projection native decode body from the SHA-pinned audited source,
with a structural tail check, preserving the internal fused GDN arithmetic. This
is an instance-local experimental function, not an installed source patch.

The host `greedy_verify.py` contract selects the longest exact proposal prefix
against K+1 target argmax rows. Retain state after seed plus accepted proposals;
the target correction/bonus is emitted but remains the next unconsumed input.
Accepted EOS suppresses later rows; correction EOS terminates without consuming
it. Six tests cover all rejection positions, bonus/off-by-one cases and EOS.
This selector neither calculates target predictions nor commits device state.

Run [33997961604](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33997961604),
code `82b9e3b`, passed the output-batched arm: all 216 exact prefix/continuation
cases and all 30 stale-state negative controls again passed. The layer now reads
input and output projection weights once per T-row block and performs one TP
output reduction; convolution/recurrence/norm-gate still use the native fused
per-token path, not a new time-fused recurrence kernel.

Single synchronized traced diagnostic observations, three seeds:

| T | Input-batched only, snapshot-inclusive ms | Input + output batched, snapshot-inclusive ms |
| --- | --- | --- |
| 1 | 0.411-0.468 | 0.409-0.414 |
| 2 | 0.666-0.675 | 0.703-0.720 |
| 4 | 1.195-1.222 | 1.195-1.199 |
| 8 | 2.247-2.251 | 2.191-2.194 |
| 16 | 4.354-4.389 | 4.176-4.193 |

These are separate correctness runs, not paired ABBA performance trials. Each
row snapshots the entire eight-slot recurrent/conv allocation even though only
one slot is active. Do not use these costs as an optimized verifier curve,
extrapolate them over 48 layers, or claim a serving speedup. Next isolate active-slot
snapshot/restore cost, retain full-state comparisons for correctness, and test
attention KV masks/positions plus all-layer target logits before drafter integration.
Current local validation: 87 CI/helper tests and 26 drafting/selector tests passed;
shell syntax and whitespace checks passed. No serving default or runner access changed.

## E3 active-slot storage (2026-09-06)

The `gdn-active` arm snapshots only logical slot zero: recurrent axis 0,
convolution axis 1. It retains BF16 dtype and padded tile storage, so convolution
snapshots do not necessarily shrink physically. Restore clones the saved tensors
before the native consuming writers and changes only slot zero, preserving live
buffer addresses. No installed source or arithmetic changes.

Repeat the 216 prefix/continuation and 30 stale-state gates, now comparing complete
live state against the full-state reference after each active restore. Additionally
change all eight slots, restore only zero and assert slots 1..7 stay unchanged on
both chips. Compare full versus active save and restore separately with three ABBA
blocks, 30 trace replays per sample, synchronized outside the measured loop.
This is a snapshot-cost experiment, not full-model speculative throughput.

Run [33998325637](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33998325637),
code `da39648`: all 216 prefix cases, 30 stale-state controls and 15 changed-idle-slot
checks passed. Padded snapshot allocation fell from 7,602,176 to 2,097,152 bytes per
chip (72.4% smaller). However active save was 11.1-11.4% slower (~0.0593 vs 0.0533 ms),
and restore was 13.61-13.66 times the full-copy cost (~0.732 vs 0.0537 ms).
Do not promote this native-writer restore as a latency optimization.

The native convolution slot writer reconstructs the whole buffer with slice/concat
and copy. Next `gdn-direct` tests a 48-worker generic-op DMA kernel: full tiles for
slot-zero recurrent state, but only the two 32-byte BF16 row-zero face segments per
convolution tile. It uses no compute kernel or arithmetic, does not overwrite other
rows, and avoids native restore's reconstruction. Both directions use preallocated
snapshot/live addresses and the identical correctness/isolation and ABBA gates.
Shapes, interleaved DRAM placement, dtype, both chips and non-aliasing are guarded.

Direct-copy runs 33998799270 and 33998989061 failed exactness; preserve their timings
as invalid-for-promotion. Diagnostics isolated 2,560 wrong values per convolution
tap/shard, starting at channel 16, before trace capture; recurrence matched. The
second face-row used a scratch offset of 32 against a DRAM face offset of 512.
The retry preserves the same NoC address alignment with scratch offset 512;
do not confuse this with changing the BF16 face layout or arithmetic.

Run [33999114362](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33999114362),
code `2b38da1`, passed 216 exact prefix/continuation cases, 30 stale-state controls
and 15 changed-idle-slot checks. The alignment fix retained exact recurrence and
all convolution channels on both chips. Copy kernel SHA256:
`4f921500b63817a5f26f288725a9da1fee9223841a68d058d96fac0f67a23428`.
In three paired ABBA blocks, save took 0.02266-0.02282 ms versus full-copy
0.05319-0.05346 ms (57.16-57.48% less); restore took 0.02265-0.02272 ms versus
0.05320-0.05336 ms (57.35-57.55% less). Padded allocation remains 72.4% smaller.
This is a validated per-layer snapshot microbenchmark, not a decode-speed claim.
The full-model oracle pins this kernel hash as its prerequisite.

## Full-model rollback oracle

Added `full-prefix`, gated behind the direct-copy layer test. It retains native
sequential target decode, not the experimental multi-row GDN forward. Compare
eager and trace logits, then freshly prefill each branch at lengths 63/64/65.
For retained prefixes 0..16, advance through a deliberately wrong rejected tail,
restore all 48 GDN slot-zero snapshots, rewind the explicit decode position and
compare two corrected steps against an uninterrupted native reference.

Attention KV rollback is logical: future rejected entries remain physically present,
but only positions below the current valid length may affect attention. Compare
dequantized valid KV tensor values on both chips for all 16 attention layers,
including the 64-token page transition, and all active GDN states/full logits.
Wrong logical-to-physical page mapping and omitted GDN restore are negative controls.
Fresh prefill per branch prevents one branch's correction writes contaminating the
next branch's valid prefix. Snapshot buffers are preallocated before trace capture.

This oracle is a prerequisite for the batched target verifier. It does not validate
tree branches, stochastic acceptance, request cancellation/epoch reuse, long contexts,
or a serving speedup. It must not turn the policy's `verifier_ready` on in serving.

Initial full-model run 33999230778 loaded and warmed native decode, then failed in
the harness's KV reader: native `ttnn.Shape` supports integer indexing, not Python
slice indexing. No full-model correctness case ran. Convert the shape to a tuple
before extracting dimensions, add a native-like shape regression fixture and retry.

Run [33999532634](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33999532634),
code `d203018`, completed successfully. Artifact evidence confirms:

- All six eager/trace baseline comparisons matched full logits exactly.
- All 102 rollback cases passed: lengths 63/64/65, eager and trace, every retained
  prefix 0..16. Both correction steps matched full logits, all 48 GDN active states
  and logically valid KV values in all 16 attention layers, on both chips.
- All six negative-control configurations detected both stale GDN state and wrong
  page mapping (12 checks), rather than merely exiting successfully.
- Two reusable all-layer compact snapshot sets occupied 201,326,592 bytes per chip.

This establishes full-model serial rollback correctness for the tested page-boundary
fixtures. Rejected KV entries need not be physically erased in this path: rewinding
the decode position and restoring GDN state kept the rejected suffix from affecting
the checked continuation. It does not establish that concurrent multi-row KV writes
to shared pages are safe. The next gate is batched attention/target verification
against this oracle, including shared-page write hazards, then its measured cost
curve. No speculative serving flag, precision setting or deployment changed.
