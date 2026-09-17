# Combined-runtime trace attribution

## Winning-recipe refresh: recovered device attribution

Run **35185624322** completed the audited 4K request and closed both devices
cleanly. CI failed afterward because the host Tracy export was missing, not
because inference failed. The retained device CSV is usable independently.
`winning_device_attribution.py` pins the request/device hashes, verifies request
correctness and source stability, and checks replay markers/counts on both chips.
Four steady T16 replays per chip pass those checks.

| Median T16 device timing | Chip 0 | Chip 1 |
| --- | ---: | ---: |
| Kernel envelope | 66.158 ms | 66.155 ms |
| Kernel interval union | 64.916 ms | 64.899 ms |
| Uncovered intervals | 1.243 ms | 1.256 ms |

Chip 0's largest groups are the 99-core generic group (11.396 ms), 32-core
matmuls (10.786 ms), and 96-core generic group (9.534 ms). Their geometry is
consistent with fused MLP, down/output projections and GDN recurrence,
respectively; generic labels do not prove kernel identity. Group sums can
overlap. This is attribution, not a new TG result or proof of internal stalls.
No hardware rerun is needed just to recover this evidence. The 131K/262K
combined context ladder takes priority over further short-context profiling.

### Per-RISC check of the retained winning trace

`winning_risc_attribution.py` reuses the exact request/device hash validation,
excludes first replays, and requires every operation in every selected steady
T16 replay. Missing RISC measurements stay missing, not zero. Chip 0 medians:

| Operation/core group | BRISC ms | NCRISC ms | TRISC0 ms | TRISC1 ms | TRISC2 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic / 99 | 11.395 | 10.839 | 11.276 | 11.371 | 11.372 |
| Matmul / 32 | 10.785 | 10.271 | 10.724 | 10.715 | 10.731 |
| Generic / 96 | 9.534 | 9.217 | 9.484 | 9.477 | 9.484 |
| Matmul / 43 | 5.738 | 5.508 | 5.698 | 5.698 | 5.715 |

Reader/writer and compute envelopes are nearly coextensive in the dominant
groups. **This does not distinguish memory stalls from compute saturation**:
RISC durations include waits. It does not justify increasing cores or replacing
SFPU math blindly. The next targeted diagnostic needs internal reader/compute
sections or circular-buffer wait measurements in the winning verifier, retaining
the full combined request as the correctness and performance acceptance test.
No new hardware run or throughput result was needed for this re-analysis.

## Earlier admission attempts

Attempt three passed disk admission (0.53% full I/O stall), entered the audited
request after about 148 seconds and completed five speculative blocks before
the nine-minute launcher timeout. It did not finish correctness validation or
profiler export; it supplies no accepted throughput or complete attribution.

The next profiling-only revision caps output at 64 tokens, while explicitly
retaining 4096 prompt rows and the winning 4352-row draft history allocation.
The full-request token/state/feature audits and minimum replay-count checks
remain enabled. This reduces diagnostic work, not the performance target;
unprofiled throughput acceptance still requires the normal combined workload.
Five host tests and adaptation against the staged frozen recipe pass locally.

Run **35181419754**, revision `09cbc1e`, staged the unchanged winning T16
recipe successfully but stopped before weight loading with exit **75**.
Four 15-second host I/O observations measured full-stall fractions of **15.38%,
3.16%, 2.55%, and 4.25%**, above the existing 1% admission limit. This was not a
kernel failure or a throughput measurement. Do not rerun unchanged until host
contention has subsided; do not stop unrelated jobs to make the gate pass.

The prepared profile runs one complete audited 4K request with shared-Q/K,
norm prefetch, incremental history, draft-tail assembly, target T16 fusion and
captured publication. It changes host instrumentation, not device kernels.
The launcher is capped at nine minutes and the whole job at twelve minutes.
PP/TG remain null because instrumentation perturbs execution.

The retained winning 4K run **35167726511** measures 118.2196 committed TG:
67.249 ms verification/readback per block, including 65.803 ms in the blocking
trace call and only 0.433 ms in output readback. A blocking host call includes
device execution; it is not proof of host overhead. The historical profiles
below suggest internal kernel work is important, but are not a substitute for
attributing this exact recipe. Keep synchronization and correctness checks
until dependency-safe alternatives have their own evidence.

Next: collect both-chip trace attribution, rank current operation costs, then
qualify the selected change in simulation before an uninstrumented combined
control/candidate comparison. Do not convert summed or overlapping profiler
durations into projected committed TG.

Retained admission artifact SHA256:
`04f65acdaf3ba110a244048dc2f1dadedc9206827a9f63c9ab51a27b5d2c8f08`.

## Current 32K shared-Q/K profile

Run **35076649250**, revision `b93f337`, passes in **9m42s**. This profiles
the admitted 32K combined candidate from run 35074425940, retaining shared-Q/K,
scalar draft reciprocal and compact target scratch. It is not a TG benchmark.
One audited request supplies eleven T16 replays, ten steady, on each chip.

| Per-chip median | Chip 0 | Chip 1 |
| --- | ---: | ---: |
| Kernel envelope | 73.688 ms | 73.682 ms |
| Kernel interval union | 72.372 ms | 72.357 ms |
| Uncovered intervals | 1.314 ms | 1.327 ms |

| Chip 0 operation/core group | Calls/replay | Summed kernel ms |
| --- | ---: | ---: |
| Generic, 99 cores (fused MLP mapping) | 64 | 11.389 |
| Matmul, 32 cores (down/output mapping) | 128 | 10.788 |
| Decode SDPA, 110 cores | 32 | 9.882 |
| Generic, 96 cores (recurrence mapping) | 48 | 9.488 |
| Matmul, 43 cores (GDN input mapping) | 48 | 5.738 |
| Generic, 48 cores | 112 | 4.607 |
| Generic, 24 cores (GDN norm mapping) | 48 | 4.487 |

Mappings in parentheses remain source-geometry inferences, not kernel identity
labels. Group sums can overlap. Do not sum across chips or convert profile time
to committed TG. The two chips are balanced at this resolution; uncovered
dispatch intervals cannot explain the required speedup. Investigate internal
matmul/MLP and recurrence work, plus the now larger long-context SDPA cost;
do not restart rejected outer-add scheduling or chase host readback alone.

Request SHA256: `3429a57f7bff88eaa6951cfaddc6fba84080cb09ed5fc9805b2cde342d7fade4`.
Device report SHA256: `6cb93cf904fe36876db72310652579a232b27d6547a2c147a521670c327112fe`.

## Historical 4K profile

Status: completed attribution run 34698584939, revision `6f83b7c`. This is not a TG benchmark.

The most recent GDN changes preserve exactness but do not reduce blocking trace
time. Before another fusion, measure the actual device operations in the current
combined runtime rather than reuse an older pre-fusion profile.

| Preserved feature | Configuration |
|---|---|
| Target | Qwen3.8-27B, two P150A, four fabric links |
| Workload | CTX 4096, one coding stream; output capped at 64 tokens for attribution |
| Drafter | Native DSpark attention, captured proposals, score layout |
| Verifier | T16 folded attention, commit-only GDN, fused T16 MLP |
| Publication | Captured feature projection; unused singleton position uploads skipped |
| Experimental GDN changes | Disabled; native recurrence reference |

Only one audited request runs. The existing request observer brackets real
verification calls and records their trace IDs; device profiler rows must match
those replay IDs on both chips. Feature, output and state audits remain enabled.
The report explicitly marks instrumented/correctness-only and leaves PP/TG null.
This is neither a new throughput measurement nor completed-function acceptance.

The prior profile flag rejects fusion and captured publication. A separate
`combined_profile` option retains those safeguards and admits only the complete,
audited combined configuration. The report validator additionally checks its
publication/fusion route. Serving defaults remain unchanged.

## Measured device costs

Steady T16 trace envelope: **68.722 ms on chip 0 / 68.719 ms on chip 1**.
Kernel interval union: **67.394 / 67.379 ms**. Uncovered intervals: **1.329 /
1.341 ms**. This rules out uncovered launch intervals as the dominant cost in
this trace; it does not measure every cause of stalls inside kernels.

| Chip 0 operation/core group | Calls per replay | Summed kernel ms |
|---|---:|---:|
| Generic, 96 cores (GDN recurrence mapping) | 48 | 11.747 |
| Generic, 99 cores (fused MLP mapping) | 64 | 11.388 |
| Matmul, 32 cores (down/output projections) | 128 | 10.791 |
| Matmul, 43 cores (GDN input projection) | 48 | 5.740 |
| Generic, 48 cores | 112 | 4.608 |
| Generic, 24 cores (GDN norm mapping) | 48 | 4.494 |
| Decode SDPA, 110 cores | 32 | 3.509 |

Parenthesized mappings follow the current source geometry and call counts;
the profiler labels these operations by family/core count, not generated source
identity. Group sums may overlap and must not be added as a critical-path total.
The 99-core/64-call group is consistent with the installed fused MLP path, rather
than the older 39-core native projection profile.

The historical 36-core figure is not the active-core count for this whole
runtime: different kernels use different grids, including 96, 99 and 110 cores.
Simply adding cores or removing host gaps cannot plausibly double this trace's
speed. At 11 committed tokens/block, 200 TG requires a whole block cycle of
55 ms, already less than this verifier trace alone, before drafting/publication.
That acceptance is workload-dependent, not a general guarantee.

Historical follow-up, now completed: the two-tile outer-add candidate passed
simulator correctness but hardware run 34699539483 regressed from 106.47 to
106.16 committed TG, with blocking trace time rising from 68.017 to 68.329 ms.
Do not restart that scheduling family from this older recommendation. See
[the retained comparison](gdn-outer-add-experiment.md#two-tile-arithmetic-setup-candidate).

This profile also predates the shared-Q/K path used by the later winning recipe.
Its operation groups identify historical costs, not measured costs of that newer
kernel composition. Before selecting another recurrence/projection change,
attribute the exact winning runtime rather than treating these group totals as
its current bottleneck breakdown. The winning 8K request's overall verifier
cost is separately established in [the runtime budget](winning-runtime-controls.md#winning-8k-recipe-where-the-200-tg-gap-actually-is).

Evidence validation checks 786 script hashes, exact request/state/feature audits,
trace markers, and request/device-report hashes. PP and TG remain null.
Attribution SHA256:
`47d3d488294042674f2778ebe93e4cddc77d66c14da88ed1fea4b71b00261da0`.
Request SHA256:
`dec755b4254cf0909eb2c12121be580c4d1d1985e224a6377ccd97b1a8e2854c`.
