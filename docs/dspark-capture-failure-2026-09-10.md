# Captured DSpark proposal: hardware correctness failure

No captured-proposal speed result is qualified. The eager arm passes, then the
captured arm fails on the first target feature at CTX4096, tap5, chip0.

| Hardware run | New evidence |
|---|---|
| 34438631295 | Eager exact; captured arm fails before any timed request |
| 34439165535 | Proposal capture/replay preserves target GDN and valid KV; 2543/2560 feature values differ |
| 34439562578 | Verifier setup restores initial target state; actual token/position buffers match the ticket on both chips |
| 34440152157 | Failure-only diagnostic cannot restore the independently recorded initial GDN state |
| 34440590613 | Fresh prefill restores exact initial state; native T1 and eager batch match through layers 0–5 on both chips, but captured tap5 matches neither |
| 34441244154 | Proposal phase overwrites 138/480 saved verifier shard hashes before target verification; live target state is unchanged |

## Latest isolation

Run 34440590613 remains a correctness failure, not a speed measurement.
All three diagnostic saved-state restoration attempts differ in 138 of 480
GDN shard hashes (indices 0–137); all 64 valid-KV hashes remain exact.
Fresh prefill restores the independently recorded initial hashes every time.
From that verified frontier, all twelve layer/chip comparisons through layer5
are bit-exact between native T1 and the identical-input eager batch.

This narrows investigation to captured execution and state/buffer lifetime,
not a demonstrated error in eager batched layer math. It does not yet prove
whether saved active snapshots or inactive GDN slots are damaged: the digest
covers whole tensors, while restoration writes only slot zero.
Next inspect snapshot ownership and trace allocation/reuse, separating active
from inactive state before changing kernels. Do not retry this candidate unchanged.

## Saved-buffer corruption confirmed

Run 34441244154 fails earlier at
`proposal_replay_protected_verifier_storage`. The same 138 shard entries now
change in the verifier's saved initial snapshots across the proposal call.
These snapshots must remain immutable for the request. This is direct evidence
of cross-component storage corruption, independent of the later feature mismatch.
The instrumented call includes eager comparison and trace replay; this result
alone does not distinguish those two operations as the writer.

The pinned local allocator implementation explicitly warns that buffers allocated
after a live trace must die before replay. Current setup captures the proposal
before allocating verifier snapshots and fixtures. Retaining Python-visible
proposal intermediates does not establish ownership of internal operation scratch.
The repair needs a safe allocation lifecycle for both traces, not a fresh-prefill
workaround or relaxed numerical acceptance. No new throughput is qualified.

## Allocation-order candidate

The opt-in captured arm now builds verifier initial snapshots, checkpoints,
feature destinations and input fixtures before capturing the proposal. A
pre-capture callback then prepares and warms the proposal before verifier capture.
The eager arm and serving defaults are unchanged. Proposal trace setup now falls
inside verifier setup time; both remain excluded from TG and included in total
request setup. Snapshot protection stays enabled for the audit arm, including
proposal preparation itself. This is a candidate repair, not hardware qualification:
later trace outputs and internal scratch still require the existing exact checks.

The last two runs reproduce identical actual/expected feature hashes:

- Actual: `fa79684034f73743469633c3d96ef3de595d957b7de0698b18b0f8b086f29482`
- Expected: `f544a7f4396a7be7b1c3dc94a6ce27a2e28a4c7f48b10ef665f3f232c063142a`
- Shape `[1,1,1,2560]`, BF16, both finite; maximum absolute difference0.2421875.
- First ticket token71093; positions4096 through4111. Both device shards match.
- Both recorded proposal eager/replay comparisons are exact across six tensors.

These checks do not prove all verifier intermediates, rotary buffers, shared
weights, or collective scratch are correct. Next isolation must compare the
first target layers and distinguish verifier trace replay from an identical-input
eager verifier, preserving initial state and the full valid KV prefix.
Do not spend another hardware run collecting the same terminal mismatch alone.

Target KV inventory confirms BF8 on both chips. Keep this separate from the
drafter's BF16 history and from the unqualified compressed-KV proposals.
