# Bounded MLP wait diagnostic

**Status: simulator and hardware fixture qualified; combined attribution pending.**
This does not change the winning T16 recipe or serving defaults.

## Why this experiment

The winning 4K cached request spends about 66.39 ms in verification per block.
At its observed 12.1 committed tokens/block, the entire cycle must fall below
60.5 ms to reach 200 TG. Caching alone cannot close that gap.

Retained RISC envelopes include waits; they cannot distinguish arithmetic from
memory backpressure. Earlier global-profiler combined runs overflowed buffers
and spent minutes draining reference-generation events. Do not repeat those
runs or extend their timeouts.

## What changes

| Component | Bounded diagnostic |
| --- | --- |
| Arithmetic | Unchanged fused MLP computation and operation order |
| Sample points | K-block 10 input/weight waits, plus first output wait/write |
| Storage per chip/layer | Two input pages and one weight page, 128 bytes each |
| L1 scratch | Separate CB31 input and CB6 weight storage; no cross-RISC alias |
| Clock | 64-bit wall clock, high/low/high read to handle low-word rollover |
| Global profiler | Not used |
| Reporting | Cycles only; diagnostic timing is not committed TG |

`mlp_clock_samples.py` wraps the existing sampled statements without deleting
or reordering original operations. Both branches execute the same statement.
`mlp_clock_projection.py` adds persistent caller-owned DRAM sample bindings
and scratch buffers. Reversing either transformation must reproduce the exact
original source. Sample parsing rejects missing markers, backwards intervals,
excessively large intervals and out-of-order observations.

The adapter deliberately does not allocate or free sample tensors. Its caller
must allocate them before capture, keep separate storage for every live layer,
and retain them until all referencing traces are released. Reader export adds
a small diagnostic cost; these samples cannot be treated as unperturbed timing.

## Remaining acceptance gates

1. Run the new simulator fixture owner: poison every output page before each
   replay, synchronize, execute once, synchronize, then read all three pages on
   both chips. A missing-execution negative control must reject poisoned pages.
   `mlp_clock_capture.py` implements this sequence; host tests also reject stale
   samples after a successful execution, missing-chip output and moved bindings.
2. Compare eager and changed-input replay outputs against the unchanged kernel;
   require complete, fresh samples on both chips. No full-model weight loading.
3. Qualify the same bounded fixture on hardware before using it in the combined
   winning T16 runtime. Do not relax its correctness or source-identity gates.
4. Collect per-layer samples from one combined audited request, outside reference
   generation. No global profiler drains. Use results to select one concrete
   verifier optimization, then measure that candidate with diagnostics disabled.

Host tests verify source restoration and buffer contracts, not compilation,
device ordering, trace lifetime or numerical parity. Nothing here establishes
200 TG, a faster MLP, or a production-ready profiler.

The dedicated `qwen-mlp-clock-sim.yml` stages from the frozen runtime, restricts
the fixture to T16, retains byte-exact packed-weight checks and all changing-input
native comparisons, and requires five complete sample sets (one eager, four
replays). Its process cap is 510 seconds and whole job cap 12 minutes, including
setup and artifacts. It does not load the full target model or enable the global
profiler. These caps do not alter the full context ladder workflow.

First simulator attempt `35207165540` failed in sample allocation validation,
before kernel compilation, and closed cleanly. TT-NN shard `device().id()` exposes
the mesh identity here; it is not a unique chip identifier. Alias validation now
compares addresses within each ordered shard, matching the existing replay
binding checks. Host fixtures reproduce equal mesh IDs and equal cross-chip
addresses while continuing to reject input/weight aliasing on the same shard.

## Simulator acceptance

Run **35207583937**, source `237d38eab89dc24995091918548672e4070afec4`,
passes in **3m54s** including setup/artifacts. Independent report validation
confirms both T16 eager outputs, all 12 native/fused replay outputs, four
byte-exact weight comparisons, stale-input controls and the missing-execution
sample control. Five complete captures contain **100 fresh samples** across
both chips. Process exit and all four container-cleanup statuses are zero.

Report SHA-256:
`6eb70a4126095ce96b14c78ff23453e470e3a78199f2dacdc4d2867038f9753e`.

The hardware lane pins that report and checks staged sources against its
manifest before changing only simulator-versus-hardware CLI admission. The
same diagnostic arithmetic and all numerical/freshness checks remain required.
Hardware timing samples will not be promoted as combined-runtime throughput.

## Hardware fixture acceptance

Run **35208641840**, source `c5e41f44cb319488d4349653d8d0c46d55f53a06`,
passes in **26 seconds** including setup/artifacts. The actual probe step takes
10 seconds. Independent validation confirms the same 100 fresh samples, both
eager outputs, 12 replay outputs, weight checks and negative controls. Source
fingerprints remain unchanged; hardware and simulator kernel manifests match.
The container exits zero without OOM or runtime error.

Report SHA-256:
`b47176bc1397aef8bb9115faad2c0a745c729b0b18eec75c1480fb3616059c22`.

These are geometry-matched single-layer fixture observations, not target-model
bottleneck measurements. The next gate is all 64 MLP layers inside the winning
combined T16 runtime, with distinct persistent sample buffers per layer. Do not
infer a throughput improvement from successful instrumentation qualification.

## Combined ownership adapter (host-tested, not deployed)

`mlp_clock_combined.py` binds one distinct preallocated capture to each of the
64 ordered native target MLP weight tensors. It rejects cross-layer sample
aliasing and T32 substitutions. The scoped verifier wrapper samples only the
first two full T16 verification calls; later blocks and tail widths still run
the unchanged verifier exactly once. No profiling is added to gold decoding.

The caller must create these buffers before trace capture and retain their
ownership through verifier teardown. Engine construction is tracked before
capture begins, so a constructor failure cannot accidentally authorize freeing
buffers still referenced by a partially captured trace. Failed execution does
not retry or publish partial samples. Scoped methods restore on exit.

`mlp_clock_experiment.py` now wires the bank into the winning frozen request
runner. It allocates before request warm-up, binds source-matched qualified
diagnostic kernels, retains native feature/state checks and executes one audited
4K request. PP and TG remain null. Normal teardown requires verifier traces to
be closed; failed construction pins owned pages on the model until mesh shutdown.

Fresh local staging from the frozen revision passes, including runtime component
evidence and exact diagnostic source matching. The dedicated combined workflow
has a 600-second launcher cap and 12-minute whole-job cap. It samples only two
full verifier replays, does not enable the global profiler, and leaves the ladder
workflow untouched. Combined hardware acceptance is still pending.
