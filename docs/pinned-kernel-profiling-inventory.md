# Pinned kernel profiling inventory

Run **35092212895**, revision `341059a`, completes in **12 seconds**. It reads
14 source files from the existing pinned image, without model weights, device
access, network access or changes to the image. No expected paths are missing.
The exported GDN compute, reader and writer match all three exact hashes used
by `gdn_multitoken.py`. Evidence is in the run's `qwen-kernel-inventory` artifact.

## What this establishes

| Facility | Pinned source evidence | Still unproven |
| --- | --- | --- |
| Named device zones | `kernel_profiler.hpp` defines `DeviceZoneScopedN` | Marker retention and useful attribution in our request capture |
| Hardware counter definitions | `perf_counters.hpp` declares FPU, PACK, UNPACK, L1 and instruction groups | Which groups the compiled Blackhole runtime enables and exports |
| Original GDN math | Exported native sources match our required hashes | Which internal stage dominates current latency |
| Existing device CSV | Per-RISC duration columns, no CB-wait or counter fields | Useful compute versus time waiting inside a kernel |

Source definitions are not measured counters. Do not report unsupported fields
as zero, infer utilization from kernel duration, or apply upstream features on
the assumption that the pinned profiler exposes them.

## Next measurement boundary

Stop input/weight-buffer capacity sweeps: both the four-block and full-K MLP
variants failed the combined-runtime performance test. Retain incremental
publication and norm prefetch, then add bounded named scopes to a representative
MLP/GDN operation inside that exact combined request. Separate input readiness,
weight transfer completion, compute/pack and output backpressure. Keep numerical
and state audits, label the run instrumented, and reject missing/dropped scopes.
Do not time an isolated kernel and report it as committed TG.

Instrument only selected operations/workers initially to bound marker volume.
The pinned `DeviceZoneScopedN` macro declares fixed local names; each injected
zone needs its own lexical block, and its lifetime must cover the intended wait
or work rather than merely recording an instantaneous marker. These source
changes must preserve buffer ownership and synchronization order.

## Sampled MLP diagnostic qualification

Export inventory run **35093041332** passed in **14 seconds**, but its guessed
`tt_metal/tools/tracy` directory was absent. The pinned upstream tree locates
the exporter under `tools/tracy`; the inventory now uses that path.
At revision `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`,
`tools/tracy/__main__.py` maps `--disable-device-data-dump-to-files` to
`TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1`. `process_device_log.py` consumes
zone name, phase, source location and trace identity fields from raw device CSV.
Our current combined capture disables that raw export; aggregate RISC timings
alone cannot establish internal waits.

`frozen_mlp_wait_zones.py` prepares nine diagnostic scopes around existing waits
and barriers, sampled at K-block 10 on input workers 0/1 and weight worker 0.
Output readiness/write completion use only the first output pair on worker 0.
Each branch executes the original statement once; removing the generated scopes
must recover the complete original source byte-for-byte after newline normalization.
Buffer sizes, transfers, math and semaphore order are unchanged. Four host tests
cover that transformation, duplicate instrumentation and source drift rejection.

The simulator lane requests device profiling and retains eager, changed-input
replay and byte-exact weight checks, with a 12-minute whole-job cap and no model
weight loading. This is not yet simulator or hardware qualified. Raw marker
retention must still be verified before attributing hardware stalls. A barrier
scope measures remaining completion wait, not the whole transfer or bandwidth.
No serving default or performance claim changes.

First simulator attempt **35146501486** failed after **3m59s** (exit 134),
not a timeout. Its T16 changed-input replay matrix passed, but the overall report
remained false: native profiler processing aborted on mismatched TRISC kernel/FW
markers with sentinel trace identities. This attempt enabled device profiling
without trace tracking. The retry adds `TT_METAL_PROFILER_TRACE_TRACKING=1`, as
used by the existing combined capture; kernel sources and correctness checks
remain unchanged. This is a diagnosis to test, not a proven fix. No hardware
admission follows from the partial replay result.

Retry **35147195915** also failed with exit 134 after **4m24s**. The two eager,
twelve replay and four weight-check entries were retained, but the overall report
stayed false and native TRISC kernel/FW marker pairing still aborted at close.
Trace tracking alone is therefore not a fix. The next diagnostic is an exact
unmodified-reader control with the same profiler settings, not another buffer or
kernel variant. It distinguishes a general profiler/replay failure from an
instrumentation-specific failure. Neither failed run admits hardware use.

Control **35147997366** failed in **4m04s** with the same native TRISC marker
pairing abort. Its retained manifest proves both input and weight reader hashes
are identical before and after staging; `unmodified_control=true`. The overall
report is false, despite two eager checks, four weight-check entries and a passed
T16 replay matrix. The new timing scopes are therefore not necessary to reproduce
this failure. Stop changing kernel operations to address this profiler failure.
The remaining diagnostic boundary is native profiler collection/finalization:
the existing combined profile uses mid-run dumping, whereas these fixtures do
not. Inspect that path before another retry; do not disable marker validation or
promote partial reports as successful qualification.

### Mid-run drain result: numerical pass, timing incomplete

Run **35148936342**, revision `c3b56fc`, passed in **5m37s**, exit 0 and clean
container cleanup. Explicit profiler drains follow replays and precede trace
release. Independent artifact validation confirms two exact eager comparisons,
twelve exact changed-input comparisons, four stale-input controls, four exact
43,520-page weight comparisons, and matching reader/replay-adapter hashes.
Numerical report SHA256:
`d246e6e3b23fb6a5615cafe261c90dc5e15db195cb0e78f4f6a47641584088fd`.

The raw CSV is retained, but is **not qualified timing evidence**. Across four
fused trace executions on each chip, it contains 133 sampled endpoints rather
than 160: every input-read and output-write end marker is absent, and input-free
has only five ends for sixteen starts. The strict raw parser rejects this data.
The simulator also reports zero clock frequency; no hardware latency or bandwidth
is inferred. Missing endpoints are not zero-cost waits. The cause of their loss
is still unproven, and mid-run success does not prove profiler event completeness.

This qualifies the tested kernel arithmetic/replay, not the diagnostic timing
path. Next establish complete markers in a bounded hardware fixture before
spending a full model load on combined-request profiling. Keep the same numerical
gates and reject incomplete hardware markers; do not promote an instrumented TG.

The hardware marker lane has a seven-minute whole-job cap and a 210-second probe
limit. It reuses only the cached MLP fixture, opens the explicitly allocated pair,
and refuses detected device owners. Before opening cards, admission pins the
successful simulator report, reader sources, projection code, replay helper and
packed-weight checks. Hardware adaptations change backend selection and record
the actual fused trace ID, not kernel operations or arithmetic.
The final validator requires all numerical checks plus ten matched scopes per
execution, across four fused replays on both chips. Sources are checked before
and after the probe. A green numerical report alone cannot qualify the markers.

### Hardware marker check: passed in 34 seconds

Run **35150494386**, revision `a894c75`, passed all numerical gates and retained
**160 endpoints / 80 matched scope pairs** across four actual fused replays on
each chip. Independent downloaded-artifact validation reproduces the samples
exactly. Container exit is 0 with no OOM. No full model was loaded.

Hardware marker report SHA256:
`08b18726cb651df3536c9143767d52432dca7fe585bfd23cb069d97f7677920e`.
Hardware numerical report SHA256:
`c648ce5795a3a7df682d3f85b1d8651eb123f73f7df41e0085df806627a317f0`.
Raw marker CSV SHA256:
`2f8146e381f9f2ba0f6b7bf42ee593f594a2aaaed03202286d1a44055b3415af`.

This admits the diagnostic scopes, not a performance change. Fixture waits are
not representative evidence for the complete model's bottleneck. The combined
runtime gate pins both reports, retains the original multi-width fusion gate,
and permits only the qualified reader-source differences. Its reversible scope
requires an audited 32K profiling request and rejects throughput claims. The
post-request route requires norm prefetch and incremental publication to remain
enabled. Combined staging/capture wiring and actual full-request marker coverage
are still pending; no combined profiling run is launched by this result alone.

Combined capture wiring now stages on top of the accepted norm/incremental
recipe, not either rejected MLP buffer variant. It runs one audited request at
32K, preserves raw device logs, and validates both the existing verifier profile
and the sampled waits. Every full-row replay must contain all 64 MLP program
identities on each chip, with stable coverage and complete endpoint pairs. Missing
layers, sessions or endpoints fail the run; diagnostics never report committed TG.
The normal whole-job cap remains 20 minutes, including staging and model loading.

The prepared raw-scope validator rejects missing or duplicate endpoints,
substituted replay identities, missing chip coverage and reported marker drops.
It keeps per-core cycle samples separate and never converts their sum to TG or
critical-path time. The hardware marker fixture passed; the combined capture
is wired but has not qualified its wait attribution.

### Combined capture overflow and bounded retry

Run **35151664646** failed: native profiler DRAM buffers filled during setup,
before the first committed block. The wait validator correctly rejected dropped
markers; the outer launcher later returned 124. This is not a new TG result.

The v2 diagnostic adds drains after completed prefill, audited reference decode,
proposal preparation, verifier warm-up, verifier capture and publication capture.
No drain is inserted inside a captured operation. It retains the same qualified
kernel sources and full correctness checks. The proposed 64-token cap was withdrawn:
it changes the history capacity and conflicts with the qualified incremental-history
transaction gate. Run **35154364454** was cancelled rather than relaxing that gate.
The next diagnostic retains the existing 256-token limit and 33,024-row history;
it still requires at least three full-width verifier observations.
Repeated metadata preservation skips unchanged files, and overflow is rejected
before parsing the multi-gigabyte raw CSV. Hardware validation remains pending.

The v3 export retains every `QWEN_MLP_` raw row (including duplicates and unmatched
endpoints), with hashes of both the full source and selected CSV. It does not
filter by replay success or relax completeness checks. This avoids copying and
uploading gigabytes of unrelated raw events. Against the existing hardware
fixture, export preserved all 160 trace endpoints exactly; 200 total MLP rows
include the eager markers. The original source remains in the profiler log directory.

Run **35154804723** failed during host preflight, before model execution: the new
drains accessed device APIs in non-profiled mock tests. The v4 adapter guards all
new drains by explicit combined profiling. Regression tests execute each guard
with profiling disabled and enabled, retaining the original non-profiled behavior.

Run **35155716351** passed host preflight but exhausted its cap with exit 137.
The first prefill ended at 22:10:24 UTC; the next began at 22:18:29. Per-token
reference-decode drains made this diagnostic impractically expensive, and marker
overflow still appeared after the second prefill. No completed request or wait
attribution qualified. Do not repeat this drain configuration or extend its cap.
Continue the independently testable draft-tail candidate while redesigning capture
to exclude reference-generation profiling rather than draining every token.

## Keep the 200 TG objective in view

The accepted 32K combined report from **35087582465**, re-analyzed with
`frozen_cycle_budget.py`, averages **11.7 committed tokens per 131.38 ms block**.
At unchanged acceptance, 200 TG requires **58.50 ms**, removing **72.88 ms**.

| Component | Observed mean | Hypothetical TG if this cost alone vanished |
| --- | ---: | ---: |
| Draft | 51.47 ms | 146.43 |
| Verification plus readback | 72.86 ms | 199.95 |
| Selection and commit | 5.76 ms | 93.14 |

These are block-only bounds with every other cost and acceptance fixed, not
achievable forecasts; request overhead makes end-to-end results lower. Do not
add nested replay/readback timers. The MLP scopes answer a narrow attribution
question: they cannot by themselves deliver 200 TG. Subsequent combined work
must reduce both draft and verifier cost, or demonstrate a higher accepted-token
yield without weakening target verification or coding quality. Avoid more buffer
sweeps without evidence of the specific wait being addressed.
