# Bounded MLP compute-processor timing

**Host-tested prototype; no simulator or hardware qualification yet.**

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
