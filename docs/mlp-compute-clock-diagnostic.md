# Bounded MLP compute-processor timing

**Simulator and small hardware fixture qualified; combined instrumentation pending.**

[Simulator run 35218187889](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35218187889)
at `bebb0040a29bb81741be4aea31ed1061cc9cc97a` passed in 3m53s.
The independent validator accepts all 180 samples, native eager/replay,
changed-input and packed-weight checks, and the missing-execution negative
control. Process exit and container cleanup are clean. Report SHA-256:
`027af7feffa73ee1d3fca5c47a265d810e1a59ab27bac9ad07c9faad401f7cb3`.
This proves simulator execution, not hardware performance or combined TG.

[Hardware run 35219468061](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35219468061)
at `6027e5b5f3fd372a007b657bddadbb1ac181f491` passes the same numerical and
180-sample checks. Its hardware step takes 11 seconds, without loading the full
model. Staging requires the exact simulator source hashes; the hardware kernel
manifest must match simulation. Independent downloaded-report validation passes,
and the container exits zero without OOM. Hardware report SHA-256:
`f95a0a568250b00e338accfc0904e3f2e98646de52e76e7b0d3b6348db143634`.

The opt-in combined adapter applies these bounded intervals across
all 64 MLP layers and two replay positions. The fixture alone does not establish
which operation dominates the full request; no throughput promotion is justified.

## Combined qualification

`qwen-mlp-compute-clock-combined.yml` stages the unchanged winning 4K T16/DSpark
recipe and runs one native-reference feature/state/token audit. Each MLP layer
owns a separate preallocated sample tensor; no sample allocation occurs during
trace capture or replay. Two actual verifier blocks must yield all 4,608
chip/processor/interval records. Missing writes, aliases and changed bindings
fail closed. Trace ownership must end before sample buffers can be released.

The adapter requires explicit `QWEN_MLP_COMPUTE_CLOCK_COMBINED=1`; ordinary
ladder and serving routes are unchanged. Hardware admission pins both the
simulator and fixture report hashes and the instrumented projection source.
Fresh full-recipe staging and host ownership/report tests pass. Combined
hardware qualification is pending. PP/TG remain null because poisoning,
synchronization and readback intentionally perturb request timing.

The winning verifier's MLP group takes about 11.4 ms per replay. Reader samples
and the rejected bank-order change do not identify whether matmul, packing or
processor handoffs dominate. This diagnostic adds six bounded intervals on
TRISC0/1/2, on logical worker (0,0), without the global profiler.

| Interval | Selected work |
| --- | --- |
| Input/weight wait | K block 10 |
| Matmul issue | K block 10, first output pair |
| Partial handoff/pack | K block 10, first output pair |
| Final partial reload | K block 19, first output pair |
| Gate rounding/handoff/pack | K block 19, first output pair |
| Rounded product epilogue | All three pairs after accumulation |

These are **processor wall-clock intervals**, not active-unit utilisation or
isolated arithmetic cycles. Calls compiled away on a particular TRISC can
measure mostly instrumentation overhead. Matmul issue may include stalls;
packing intervals include handoffs. Different processors overlap, so their
durations must not be added as a critical-path total or converted into TG.

## Storage and safety

- One caller-owned, height-sharded uint32 L1 tensor on core (0,0) per chip.
- Three disjoint 256-byte pages: one per compute processor; no cross-processor
  writes or shared sample cursor. Other workers receive sampling disabled.
- Allocate before traces, preserve addresses, poison before every invocation,
  synchronize before reading and reject missing, malformed or out-of-order data.
- No reader changes, extra CBs, global profiler buffers or new device barriers.
- Removing instrumentation reproduces the original generated compute source
  exactly. Native precision, accumulation, packer and epilogue remain unchanged.

The source adapters and pinned-native round-trip pass locally. Five host tests
cover clock rollover, processor identity, missing records, order, projection
source restoration and preservation of probe controls. Fresh frozen staging
also passes. These checks do not prove the kernel compiles or the sharded sample
storage behaves correctly on devices.

## Gates

`qwen-mlp-compute-clock-sim.yml` runs the retained T16 eager/replay/packed-weight
matrix with poisoned sample pages, plus a missing-execution negative control.
Five sample sets must each contain all 36 processor/chip/interval records.
It uses a 510-second probe budget and 12-minute whole-job cap, without model
weights. A source-bound hardware diagnostic must follow a passing simulation;
only then may it instrument the actual combined verifier. Diagnostic PP/TG stay
null. Serving and the accepted runtime remain unchanged.
