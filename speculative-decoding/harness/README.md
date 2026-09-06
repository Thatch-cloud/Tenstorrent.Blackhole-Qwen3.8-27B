# Greedy request coordinator

T32 proposal capacity is an explicit experiment opt-in: construct `GreedySession`
with `verifier_rows=32` and call `propose(..., max_rows=32)`. Standalone lookup,
hybrid and greedy-selection helpers require `max_proposals=31`. Defaults remain
T16/15 proposals, and a wider host contract does not certify a hardware executor.
No proposal is padded with oracle tokens; actual available proposals and the
remaining generation budget still determine the selected bucket.

`GreedySession` connects the existing request-local lookup/neural proposal routing
to greedy selection and a caller-owned, synchronized device publication callback.
It is a host coordinator, **not a TT device executor or a throughput result**.

1. Construct it with the prompt and the already-emitted prefill seed. The seed is
   inserted into draft history but is not counted as decode work.
2. `propose(request_id)` returns one immutable, identity-checked block ticket:
   absolute input position, epoch, source and `[seed, *proposals]` tokens. The
   supported1/2/4/8/16 bucket is bounded by both available proposals and remaining
   generation budget. A request may have only one live ticket.
3. The device executor must update every token/position/RoPE/page-table input,
   verify the ticket and return exactly one target argmax per input row. It must
   retain all recurrent histories until the decision and preserve valid KV rules.
4. `commit(request_id, ticket, predictions, publish)` selects the greedy prefix
   and calls `publish(state_rows)`. That callback must restore/publish the selected
   state on both chips and synchronize before returning `None`. Only then are
   output tokens and draft history advanced. Failures poison the session; they
   cannot be retried as a fresh commit.
5. `abort` requires the same ticket and a synchronized `restore(0)` callback. It
   leaves output/history unchanged. Stale tickets and reentrant draft/commit calls
   are rejected. `close` clears request-local history after commit/abort; a failed
   device operation still requires executor-owned trace/buffer cleanup or recovery.

`committed_decode_tokens` excludes the prefill seed and increments only after
successful publication. It is the numerator for a decode-only rate whose timer
includes drafting, input updates, verification, selection, commit and readback.
TTFT, total generated tokens, aborted work and failed requests must be reported
separately. Proposal counts are explicitly scoped to committed blocks.

Tests use actual lookup proposals and fake device callbacks to verify output
identity, rejection corrections, EOS, exact generation budgets, request isolation,
ticket lifetime, aborts, failures and reentrancy. They do not certify TT trace
reuse, neural drafter weights, coding quality or200 committed tokens/s.

The current lookup algorithm can return fewer tokens than requested: for a short
periodic history, its longest nearest match may have only a short known suffix.
The coordinator respects that actual proposal length rather than padding it with
oracle tokens. Learned drafting and any periodic-extension experiment require
their own measured acceptance/latency gates.

`prefer_full_suffix=True` is an opt-in lookup tie-break experiment. Match length
still wins first; among equally long matches it prefers a longer available
continuation, capped at the requested count, then the nearest match. It copies
only a contiguous slice of committed history and never extrapolates repetitions.
Defaults retain the nearest-match policy. Host exhaustive-rank tests and real
lookup T32 session accounting pass; hardware acceptance and speed remain untested.

Accepted EOS publishes only inputs preceding that terminal output, matching
native autoregressive stopping. Verification failures poison the request through
`fail_verification`; no output or draft history is advanced by a failed block.
