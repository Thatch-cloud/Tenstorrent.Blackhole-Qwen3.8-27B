# DRAM-sharded projection: partial-reload investigation

Status: hardware correctness passes, but the complete MLP is slower. No promotion.

| Real-weight T16 path | Complete MLP latency |
| --- | ---: |
| Native 1D control | 0.334692 ms |
| Corrected DRAM-sharded candidate | 0.357111 ms |

Run `34466816834` on `8f719ca` passes all correctness checks and clean closure.
Independent result validation reproduces all nine ABBA blocks: the candidate
loses every block, about 6.70% slower overall. Both arms include input transfer
and the native four-link collective. These are layer timings, not PP/CTX/TG.
Report SHA256: `48f1080e51d88550c34065d994401fbc205ecf6e1dee9c73d29fca86a105f4bd`.

The next candidate should remove redundant layout conversions: stage the shared
gate/up input once, keep their product sharded, and reshard directly for down.
It needs fresh complete-MLP simulation and a matched hardware comparison.

That reduced-conversion candidate passes complete-MLP simulation in
`20260910T104357Z-402`. Hardware run `34468231173` stopped at its 4D weight guard:
the native loader retains 2D matrices. The retry uses the existing audited
zero-copy 4D views before sharding, preserving native control tensors and buffers.
No candidate arithmetic ran in the failed attempt. The new build cache was
populated in 260.52 seconds; reuse has not yet been measured.

Earlier hardware run `34466116546` on `a109569`
matched the first real-weight output on both cards, then failed cleanup before
timing. The native TP2 collective consumes its input partial; the harness also
retained that freed tensor, causing `Both chips required` during address lookup.
The retry transfers ownership to the collective instead. No speed result or
serving change resulted from that failed run.

The native DRAM-sharded matmul factory uses Float32 intermediate partials when
FP32 destination accumulation is enabled, but does not set the CB5 FP32 unpack
mode that the native 1D matmul factory sets. The candidate changes that descriptor
field only; it does not change weights, activation precision or arithmetic fidelity.
Gate, up and down now match the native control exactly for two input patterns
on both simulated chips. This is eager projection evidence, not full-model or
hardware qualification.

| Simulator attempt | Result | Next action |
| --- | --- | --- |
| `20260910T092238Z-405` | Wrapper omitted the optional bias slot; factory raised `IndexError` before candidate execution | Supply `[None]` on both mesh and per-chip inputs |
| `20260910T092808Z-436` | Descriptor format getter returned an unbound native enum; raised `TypeError` before candidate execution | Compare the exposed raw format value against a constructed Float32 descriptor |
| `20260910T093131Z-684` | UP projection passes all four eager comparisons exactly; clean closure and unchanged source hashes | Screen gate/down, then replay and integrity checks |
| `20260910T093507Z-403` | Gate projection also passes all four eager comparisons exactly, including fused SiLU; clean closure and unchanged source hashes | Screen down, then replay and integrity checks |
| `20260910T093831Z-405` | Down matches the first pattern, then retained intermediates clash with native circular buffers | Release completed eager intermediates between patterns |
| `20260910T094223Z-404` | Down passes all four eager comparisons exactly; clean closure and unchanged source hashes | Changed-input trace qualification |
| `20260910T094621Z-402` | Down passes six changed-input replay checks; exact 32 physical rows, poisoned logical outputs replaced, input and packed-weight integrity, stable bindings, clean closure | Apply the same replay screen to gate/up; then complete-MLP testing |
| `20260910T095600Z-411` | Gate passes the same six replay and integrity checks, including fused SiLU | Up replay, then complete-MLP testing |
| `20260910T100422Z-402` | Up passes the same six replay and integrity checks; all three component projections now pass | Complete-MLP testing |

The composed T16 local MLP probe passed as `20260910T101303Z-400`: four exact eager
comparisons and six exact changed-input replay comparisons, including all 32
physical rows and poisoned-output replacement. It closed cleanly, retained
unchanged experiment/native fingerprints and exited zero. It compares
gate/up/product/down together against the native control, then replays changed
inputs with poisoned outputs. It does not include the TP2 collective or measure
token throughput. Intermediates are released between projections rather than
retained across the whole composition; ten focused host tests pass.

### Reduced-conversion hardware result

Run `34468926073`, commit `a68ca2e`, passed correctness but did not qualify:
native **0.334674 ms**, shared-staging/sharded-product candidate **0.385219 ms**,
or **15.1% slower**. This includes input transfer and the four-link collective,
not full-model PP or TG. The earlier native 2D versus candidate 4D weight guard
failure was fixed with zero-copy views, preserving native metadata and buffers.

The isolated native build cache hit: setup **2.34 seconds**, versus **260.52
seconds** on its cold run. This reduces experiment setup, not model latency.

Next: instrument eighteen complete MLP replays (three input fixtures, both
arms, one ABBA per fixture), retain per-operation/per-chip timings and exact
output checks. Instrumented results cannot qualify performance. The candidate
kernel is unchanged from its successful simulator pass; this adds observation,
not a new arithmetic implementation or serving default.

### Operation attribution: measured, not inferred

Run `34470085030`, commit `247d2d1`, passed all eighteen instrumented replays,
exact-output/input checks and independent local artifact validation. Both chips
agree closely. Approximate medians below are microseconds, not throughput:

| Operation | Native | Reduced-conversion candidate |
| --- | ---: | ---: |
| Gate projection including SiLU | 95.2 | 127.0 |
| Up projection | 88.3 | 87.2 |
| Elementwise product | 4.0 | 39.8 |
| Down projection | 122.3 | 100.3 |
| Additional staging / reshard / output conversion | — | 1.9 / 5.2 / 3.7 |

The complete device envelope is about 327 versus 382 microseconds, consistent
with the separate uninstrumented regression. Operation durations include waits;
their sums are not an end-to-end critical path or model TG. The profiler reports
110 cores for both multiply operations, so attributing the regression simply to
"too few multiply cores" would not be supported by this evidence.

The next candidate keeps native gate, up and interleaved multiplication, using
only the faster DRAM-sharded down projection. Its extra boundary conversions
must be included, and the complete hybrid must pass simulator replay before
hardware testing. No gain is claimed before those measurements.

Down-only hybrid simulator run `20260910T111727Z-410` passed in 338 seconds:
four eager comparisons and six changed-input replay comparisons, exact across
all 32 physical rows on both simulated chips, with output poisoning and input
integrity checks. Sources were unchanged, closure was clean, wrapper exit was
zero and the original packer was restored. Its separate source-qualified
artifact is `scripts/ci/dram-mlp-down-simulator.json`. Real-weight hardware ABBA
must still establish whether the down-projection saving survives conversions
and the native four-link collective.

First down-only hardware run `34471157896`, immutable commit `7e64401`, passed
the independent gate: **0.321541 ms** versus **0.334778 ms** native, a **3.95%
latency reduction**. All nine matched ABBA blocks exceed the 2% speedup gate;
the six eager and twelve changed-input trace checks are exact and the mesh
closes cleanly. This qualifies a repeat and then full-model integration, not a
new PP/CTX/TG result. An unchanged repeat has been dispatched before promotion.

The unchanged repeat `34471337398` also passed: **0.321349 ms** versus
**0.334666 ms** native, again about **4.0% lower latency**. Its independent local
gate passes. Both full reports are retained under `scripts/ci/dram-mlp-down-*.json`.

The target-integration hook now prepares separate down weights for all 64
layers and routes only exact `[1,1,16,5120]` inputs through the qualified hybrid.
Prefill, T1 and other shapes call the original method. Setup cost is recorded;
native weights stay untouched; methods and temporary weights are restored on
exit or failure. Host tests cover every layer, fallback routing, caller-owned
input, partial allocation failure and failed-kernel cleanup. This hook has not
yet been measured in a complete request and is not a serving change.

### Full-request integration underway

Run `34472208506`, immutable commit `ed9483f`, compares native MLP against the
down-only hybrid with folded T16 target attention enabled in **both** arms.
Both use captured precise-native DSpark proposals and commit-only GDN.
CTX is 4096, one coding stream, fifteen draft queries and sixteen verifier rows.
The schedule is two feature/state audits followed by timed A/B/B/A requests.

The gate requires exact output tokens, target state, feature publication and
matching proposals/acceptance across arms. Candidate routing must execute in
all 64 target layers and restore their original methods and weight bindings.
Weight-resharding setup is included in setup-inclusive request time, but not
mislabelled as steady decode or PP. Thirty-nine relevant host tests pass.
No full-request improvement is claimed until this run completes and its
artifacts are independently validated.

`scripts/ci/dram-projection-binding-check.py` exercises the installed native
bindings without opening devices. It checks that the exact 64-entry unpack
policy survives descriptor mutation and that the explicit no-bias slot survives
the binding. This passed locally, alongside five host regression tests.

The binding check is not a numerical simulator pass. Even a successful eager
screen still needs replay, input/weight integrity, complete-MLP and hardware
request measurements before promotion toward the 200 committed TG target.
