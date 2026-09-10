# Captured DSpark proposal: hardware correctness failure

No captured-proposal speed result is qualified. The eager arm passes, then the
captured arm fails on the first target feature at CTX4096, tap5, chip0.

| Hardware run | New evidence |
|---|---|
| 34438631295 | Eager exact; captured arm fails before any timed request |
| 34439165535 | Proposal capture/replay preserves target GDN and valid KV; 2543/2560 feature values differ |
| 34439562578 | Verifier setup restores initial target state; actual token/position buffers match the ticket on both chips |

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
