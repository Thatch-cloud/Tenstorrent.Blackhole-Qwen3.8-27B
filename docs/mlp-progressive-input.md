# Progressive MLP activation delivery

**Simulator replay passed. Hardware and throughput remain unqualified.**

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

Run **35344423357** passed in **4m28s**, source `9c1e270`. Independent review
reconstructed both generated sources and checked two exact eager outputs,
12 changed-input replay comparisons, four stale-input negative controls, four
complete packed-weight comparisons and unchanged stream fingerprints. The native
control, compute, rounding, weight reader and geometry match the pinned serial
reference. Simulator exit and all container cleanup statuses are zero.

| Retained evidence | SHA256 |
|---|---|
| Replay report | `8b843215d54c68ee2f7272db1a4c2ec0d4bdab1524cb062bb02239bf9422c72a` |
| Staging manifest | `618ede1c41773d6c9a7f4a2412b21efdf2fd787cce4d90308d753d4dc0581fb4` |
| Progressive input reader | `5d433b22eeab77f80dee3d3ef2a0cf9b34ae21743897b312d23032904684b4a5` |

The report reviewer is `scripts/ci/mlp_progressive_input_report.py`. The new
hardware admission helper pins these artifacts, retains serial/register
qualification and requires the identical reader. Its adapter-source tests pass,
but it is not yet connected to a hardware request scope. Next: complete that
scope, validate it against the actual image and run combined ABBA requests.
There is no isolated hardware timing or serving promotion.

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
