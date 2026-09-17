# Find the remaining GDN cost inside the combined verifier

**Host instrumentation tests pass; device integration and simulator qualification
are pending. No performance result or serving change.**

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

1. Wire caller-owned, poisoned sample buffers into the unchanged three-stage
   GDN program; retain reader, writer, CB sizes and arithmetic.
2. Simulator: exact output and every prefix state, changed-input replay,
   immutable inputs, and rejection of unexecuted/poisoned samples.
3. Collect a bounded sample during the **combined loaded T16/DSpark request**,
   without Tracy or full-model simulator loading. Instrumented timings cannot
   qualify TG or replace the unchanged control.
4. Pick the structural change from the measured phase, then require a separate
   uninstrumented combined correctness and ABBA performance result.

Host tests verify lossless source removal against the retained hash-checked
native export, phase order, processor/token identities, clock rollover and
missing-sample rejection. They do not establish compilability or device safety.
