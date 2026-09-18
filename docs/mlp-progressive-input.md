# Progressive MLP activation delivery

**Prototype only. No simulator, hardware or throughput qualification yet.**

The combined control still takes approximately 59 ms in target replay. Earlier
compute-clock evidence showed operand-readiness waits, but did not distinguish
activation from weight stalls. This experiment tests activation coordination;
it is not evidence that activation delivery dominates the current runtime.

| Property | Current reader | Rejected full-K preload | New prototype |
|---|---|---|---|
| Receiver-ready rounds | 20 | 1 | 1 |
| First compute can start | After first block | After all 20 blocks | After first block |
| Activation capacity per core | 32 KiB | 320 KiB | 320 KiB |
| Weight path | Serial contiguous BF4 stream | Older tile reader | Same serial contiguous BF4 stream |

All receivers reserve capacity for the complete input before acknowledging
readiness. The sender reads and multicasts eight tiles at a time, waits for
delivery, then publishes a monotonically increasing completion counter. Receivers
publish eight CB tiles per completed block; a slow receiver may catch up without
missing an intermediate counter value. Signal writes finish before the sender
reuses their source word. No input slot is overwritten within the invocation.

The pinned runtime supplies the required greater-or-equal wait in
[dataflow_api.h](https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/tt_metal/hw/inc/api/dataflow/dataflow_api.h).
The source checks do not prove NoC ordering; changed-input device replay must.

This differs from the rejected full-input preload by retaining early compute
overlap, and from rejected weight pipelines by leaving weight reads unchanged.
It preserves register arithmetic, BF16 rounding, BF4 bytes, output layout and
worker count. Additional declared activation storage is 288 KiB per multicast
core; actual L1 admission remains a device check. Added signaling barriers may
outweigh saved readiness rounds, so no speedup is assumed.

## Gates

1. Host checks: source identity, reader protocol structure, slow-receiver counter
   behavior, projection reversibility and rejection of hardware execution.
2. Existing bounded full-size synthetic MLP simulator: native-reference eager
   output, changed-input trace replay, stale-output controls, packed-weight and
   stream integrity. No model-weight loading. The run is correctness-only.
3. Only after that passes: source-bound admission and one-load combined HTTP
   ABBA comparison, followed by state/feature and held-out quality qualification
   if it produces a repeatable benefit. Do not promote based on component time.

The opt-in CI tag ends in `-progressive-input`; ordinary serving and all prior
qualified kernels remain unchanged. This mechanism alone is not expected to
close the roughly 38-ms whole-cycle gap to 200 committed tokens/s.
