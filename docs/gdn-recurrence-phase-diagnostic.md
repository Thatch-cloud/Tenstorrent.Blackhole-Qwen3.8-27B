# Find the remaining GDN cost inside the combined verifier

**Simulator and combined hardware diagnostics pass.
No throughput improvement or serving change is claimed.**

## Combined hardware finding

[Run 35245533179](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35245533179)
at `ee025f464ba28381d1458cb1dba201b9ae3b7f22` finishes in **4m1s**. The full
4K request passes native token/state/inactive checks, 100 feature checks and
11 proposal checks. All 96 norm-prefetched builds use the admitted recurrence;
4,032 phase samples cover two verifier replays, 48 GDN layers and both chips.
Container exit is zero, no OOM is reported, and source fingerprints are stable.
The independent combined-report validator passes. Report SHA-256:
`e59e1ba57cd5d8e79d6fbbdf18d254ba9a6dde2eb655001909eafd5f460dd1ce`.

Median cycles on chip 0, sampled token 8:

| Phase | UNPACK | MATH | PACK |
| --- | ---: | ---: | ---: |
| Input wait | 5,019 | 19 | 21 |
| Input conversion | 1,181 | 6,079.5 | 6,068.5 |
| State decay | 2,513 | 2,467.5 | 2,410 |
| Value read / delta | 1,162 | 1,224 | 1,163 |
| Rank update | 2,266 | 2,194 | 2,284 |
| Output projection | 372 | 204 | 257 |
| State publication | 842 | 1,204 | 1,158 |

Chip 1 is similar: UNPACK input wait 4,977.5 cycles and MATH conversion
6,046 cycles. These phases overlap across processors: the long MATH conversion
interval includes waiting for UNPACK readiness, not six thousand cycles of
conversion arithmetic. Do not sum the columns or extrapolate one core/token
to an exact model speedup.

### Candidate selected from the evidence

The reader currently reserves normalized Q/K rings **before** reading V, beta
and gate. Q is held until output projection near the end of the current token;
V/beta/gate are consumed and released near its beginning. That ordering delays
the next token's DRAM reads even when their destination rings are already free.

`gdn_input_overlap.py` moves V/beta/gate gathering ahead of Q/K gathering.
It adds no buffers, changes no arithmetic and retains every snapshot. This is
different from the earlier 6 KiB input-cache experiment, which retained the same
Q/K-first order and did not establish repeatable combined throughput improvement.
Host tests, source comparison and simulator replay qualification pass; see
[input-overlap evidence](gdn-input-overlap.md). The reordered reader still needs
its matched uninstrumented combined hardware test.
This targets the measured readiness stall, not an assumed DRAM bandwidth limit,
and cannot by itself establish the 200 TG objective.

## Simulator acceptance

[Run 35243917766](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35243917766)
at `6057beb32e3ef0e0bccd87d4ee5a0bb924bead9d` completes in **3m8s**.
No target weights are loaded. The independently checked artifact contains:

| Check | Result |
| --- | --- |
| Output, every prefix state and FP32 bridge | 24 exact comparisons |
| Original input immutability | 48 exact comparisons |
| Eager and three changed-input replays | 168 valid phase samples |
| Unexecuted poisoned samples | Rejected before and after replay |
| Exit and container cleanup | All zero |
| Staged helpers and generated kernels | Reconstructed locally and matched |

Report SHA-256:
`fa3897309763ea33703cbac7c5455cb114b6746342a78b9b81ebe68a1b88670a`.
Generated control/candidate compute SHA-256:
`ce404cf287f9962243dc65f30d9736f45b0c95cfff503c6c20d68f8981a41153` /
`cc615d298e69867c49c66a5a3c92c099a75386cc86db5e335ba760c4f3028a45`.
The simulator uses pinned TT-Metal
`9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`. Its cycle values must not be
used to rank physical hardware bottlenecks or predict TG.

## Why this is next

The register-resident MLP experiment passed correctness but did not materially
reduce verifier time: about 66.5 ms remains per T16 block. At 12.1 committed
tokens per block, 200 TG needs the **whole cycle below 60.5 ms**, including the
drafter and commit. Another sub-millisecond epilogue change cannot close this gap.

The retained combined profile places the 96-worker generic group at about
9.53 ms on chip 0. Its GDN identity is inferred from the program geometry, not
directly named by the profiler. It is worth investigating, but removing even
this entire group would not be enough on its own. MLP and projection work remain
part of the wider verifier target.

## Source finding

The current shared-Q/K path already removes repeated Q/K normalization. Each
recurrence worker handles four key tiles and one value tile, with BF16 local
state feedback. It still publishes a complete state for every candidate token
so the accepted prefix can be committed exactly.

For the current geometry, those published snapshots total
`16 * 24 * 128 * 128 * 2 = 12,582,912` bytes per layer per chip, or **576 MiB
across 48 layers** per verification block. This is a source-derived payload,
not measured traffic or proof of a bandwidth bottleneck. Replacing snapshots
with low-rank reconstruction would have to reproduce every intermediate BF16
rounding; storing only a final state would break rejection/rollback.

## Diagnostic boundary

`scripts/ci/gdn_recurrence_clock.py` adds removable wall-clock markers around:

| Phase | What its interval includes |
| --- | --- |
| Input wait | Q/K, value, gates and initial/local-feedback readiness |
| Input conversion | BF16-to-FP32 copies and their drain waits |
| State decay | Exponential and state scaling |
| Value read / delta | State matmul, subtraction and beta scaling |
| Rank update | Key transpose, broadcasts, outer product and state addition |
| Output projection | Query/state matmul and output handoff |
| State publication | Snapshot and rounded feedback copies |

UNPACK, MATH and PACK each get a separate 256-byte page. A caller-selected
token allows first, middle and final steps to be inspected separately. Original
runtime argument 0 remains the instance count; diagnostic arguments 1–3 are
the persistent local sample address, enable flag and token index. A future
adapter must supply all arguments on every worker, enable only the owner core,
and keep the allocation alive through trace release.

These are overlapping per-processor intervals, **not additive active-cycle
counts**. A long publication interval alone cannot distinguish pack work from
writer backpressure. Do not remove barriers based on these samples.

## Remaining gates

1. Implemented and host-tested: caller-owned, poisoned sample buffers in the
   unchanged three-stage GDN program. Readers, writers, CB sizes, precision and
   arithmetic are retained. Samples are an additional operand only for
   recurrence; normalization and norm/gate operands remain unchanged.
2. Passed: simulator exact output and every prefix state, changed-input replay,
   immutable inputs, and rejection of unexecuted/poisoned samples.
3. Collect a bounded sample during the **combined loaded T16/DSpark request**,
   without Tracy or full-model simulator loading. Instrumented timings cannot
   qualify TG or replace the unchanged control.
4. Pick the structural change from the measured phase, then require a separate
   uninstrumented combined correctness and ABBA performance result.

Host tests verify lossless source removal against the retained hash-checked
native export, phase order, processor/token identities, clock rollover and
missing-sample rejection. They do not establish compilability or device safety.

The simulator workflow runs the small synthetic recurrence fixture, not model
weights: 24 exact output/state/bridge comparisons, 48 immutable-input checks,
and four poisoned sample captures (eager plus three changed-input replays).
Missing-execution rejection runs before and after the numerical matrix. This
first qualification samples token 8; first/final-token device sampling remains
unqualified. Local fresh staging at the frozen revision imports successfully.

## Combined hardware integration

The combined adapter now preallocates 48 separate sample pages per chip before
request warmup, retains them until verifier traces close, and samples only the
first two complete T16 verifier replays. Warmup/capture construction can make
multiple ordered 48-layer passes; each layer reuses its own diagnostic page.
Incomplete layer sets, source drift, aliases and failed replay are rejected.

The existing winning norm-prefetch builder still runs. Only its recurrence
compute descriptor receives the simulator-qualified markers; reader/writer
programs, norm staging, MLP, attention, drafter and publication are unchanged.
One complete native-reference request audits tokens, state, inactive slots,
features and proposals. This diagnostic publishes neither PP nor TG.

Fresh combined staging passes retained-artifact/source admission locally.
The complete native API check runs against the pinned image on hardware; the
local exported source inventory does not contain every required API header.
The workflow retains exclusive two-card scheduling, the disk-pressure check,
and a bounded 600-second hardware command. No separate isolated hardware
benchmark is inserted between simulator acceptance and this combined request.
