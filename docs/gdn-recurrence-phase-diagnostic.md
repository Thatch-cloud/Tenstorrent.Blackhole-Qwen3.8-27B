# Find the remaining GDN cost inside the combined verifier

**Simulator qualification passes; combined hardware integration is next.
No performance result or serving change.**

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
