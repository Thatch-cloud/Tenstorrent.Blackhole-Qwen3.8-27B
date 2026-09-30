# Progressive MLP activation delivery

**Combined correctness passed. No demonstrated verifier speedup; not promoted.**

## Combined hardware result

Run **35346008351**, source `a92f10f`, passed in **5m58s**. One loaded model
completed two native-reference audits followed by four timed ABBA requests.
CI's independent staged-source validator passed; local read-only analysis
recomputed pair rates and token totals. All six requests retain native drafting,
serial BF4 streams and direct DMA KV publication. Candidate requests construct
all 64 projections, record 128 calls and restore their scopes. Exit is zero,
devices close cleanly, and experiment sources remain unchanged.

| CTX / streams | Arm | PP tok/s | Complete-cycle TG tok/s | Blocking verifier replay |
|---|---|---:|---:|---:|
| 4096 / 1 | Control | 3337.58 | 122.77 | 59.495 ms |
| 4096 / 1 | Progressive input | 3340.38 | 125.34 | 59.537 ms |

Each timed arm commits 242 tokens across 22 blocks with identical proposals,
acceptance and outputs. Aggregate TG improves **2.10%**, but paired changes are
**+2.26% / +1.93%**, below the two-percent requirement in both pairs. More
importantly, the changed target replay is **0.042 ms slower**, not faster.
The complete-cycle difference comes mainly from host input and commit intervals,
not a demonstrated activation-delivery saving. Do not promote this kernel or
claim its larger buffer brings the model closer to 200 TG by the aggregate gain.
PP variation is not attributed to this decode-only change.

| Mean block interval | Control | Candidate |
|---|---:|---:|
| Draft | 19.236 ms | 19.042 ms |
| Input | 1.415 ms | 0.686 ms |
| Verification/readback | 60.156 ms | 60.191 ms |
| Selection/commit | 8.386 ms | 7.440 ms |
| Complete cycle | 89.492 ms | 87.648 ms |

Blocking replay is nested within verification; do not add it again. At eleven
committed tokens/block, 200 TG requires 55 ms total. Even free drafting would
not make the measured verifier fit that budget. Stop unchanged activation
handshake/buffer experiments; the next mechanism needs to address a different
measured cost. Held-out coding quality and production serving remain unqualified.

Report SHA256:
`cbe24ae0dd2fd6c8fbfe98fa978e6645c24af6aaf9e7a67b9932389967f12f86`.

## Experiment design

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
qualification and requires the identical reader. The combined DFlash runtime now
accepts explicit `progressive_evidence` and requires
`QWEN_PROGRESSIVE_INPUT_EXPERIMENT=1`. It rejects T32, simultaneous weight
pipelining, missing admission, changed bindings and use after scope closure.
All 64 layers must execute, restore their bindings and report the additional L1
capacity. Eighteen focused host tests cover routing, admission/source changes,
full layer ownership and exception cleanup. These are not device results.
The staged comparison now retains native drafting, serial contiguous weights and
direct DMA KV publication in both arms. Two native-reference audits precede
timed control/candidate/candidate/control requests in one loaded model. Report
validation rejects changed proposals, acceptance, missing publication coverage or
an accidental pipeline/control change. Preload checks run before target weights.
The hardware tag ends in `-progressive-direct-dma-kv-slide-block-stream-dflash-native`.
CPU CI **35345791716**, actual-image preload admission and the combined hardware
comparison above have now passed. They do not establish a performance promotion.
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
close the measured control's roughly 34.5-ms whole-cycle gap to 200 committed tokens/s.
