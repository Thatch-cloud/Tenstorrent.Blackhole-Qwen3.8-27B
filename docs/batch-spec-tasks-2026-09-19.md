# Batched speculative decoding: task sheet

Written after four probes located the boundary. Every task has a done-condition that is
an observation, not an edit, because this session's record is that edits made ahead of
observations were wrong four times out of four.

## What is already true

| | |
| --- | --- |
| scheduler admits 2 | **done** - `max_num_seqs` bounded by `NATIVE_GDN_SLOTS`, image `sha256:75e923b9eae8` |
| server reaches readiness with 2 users | **done** - run 35433428038, `ready: True` |
| first request then fails | `dflash_device.__init__:40`, via `_sample` -> `bridge_factory` -> `create_request` -> `from_prefill` |

The architectural statement, in one line: **the same hardware batch dimension is spent on
users in the plain path and on speculation candidates in the fast path.** Plain path gives
8 users at 15-19 tok/s each because a step yields one token per user. Fast path gives 1
user at ~123 tok/s because a step yields ~12 tokens for that user. Both at once needs a
second axis.

## Phase 0 - locate the boundary exactly

**T1. Which clause of `dflash_device.__init__` fires.**
The condition is fourteen `or`-ed terms and the message names none of them. Instrument it
to report the first failing term and the offending value, run two users, read it.
*Done when:* the log names one term and its value.
*Risk:* none. No behaviour change, one hardware slot.

**T2. What `from_prefill` hands it for a second request.**
Extract `serving_request_factory.py` and `serving_runtime.py`. Map where `features`,
`feature_start` and `position` come from, and which of them are per-request against
per-session.
*Done when:* each of the five features is traced to its producer.
*Risk:* none, no device.

**T3. Establish what `native_gdn_slots=8` actually indexes.**
It was read as user capacity and that is now doubtful - the verdict describes verify rows
as positions within one sequence, and tt-metal PR 55548 groups candidates four to a batch
row. Find its consumers.
*Done when:* a consumer is shown indexing it by candidate or by user.
*Risk:* none, no device. **Blocks T5**: the representation depends on this answer.

## Phase 1 - the second axis

**T4. Decide packed or padded.**
Sequences accept different numbers of draft tokens, so the batch goes ragged. Published
options: EXSPEC schedules equal-length sequences to avoid realignment; EQSPEC keeps
rectangular alignment by invariant. Padding to `T16` per user wastes rows but keeps every
existing shape assumption; packing does not but touches every mask.
*Done when:* a one-page argument with the row count each costs at 2, 4 and 8 users.
*Risk:* low effort, high leverage - this decision propagates everywhere.

**T5. `serving_lifecycle`: one slot to N.**
`self.request_id = self.capture = self.hook = None` becomes per-slot. `_execute` tests
membership rather than equality.
*Done when:* two requests can be tracked and released independently, unit-tested on CPU.
*Risk:* contained; the module is 6 KB and has a CPU test already.

**T6. `dflash_device`: accept a user dimension.**
Whatever T1 names, plus the feature-shape assertions.
*Done when:* two requests construct their devices without raising.
*Risk:* **highest in the sheet.** Feature shapes reach the kernels.

**T7. Capture and trace buckets keyed by (users, rows).**
Buckets are keyed `(rows, capacity)` today and trace bucket 1 is singular.
*Done when:* a two-user shape captures and replays.
*Risk:* high. Trace capture is where host writes are forbidden and a wrong shape
TT_FATALs rather than degrading.

## Phase 2 - prove it

**T8. Equality gate.**
Two concurrent users must produce byte-identical tokens to the same two prompts run
sequentially at one user. This is the only thing that makes the result trustworthy, and
it is why the harness already exists.
*Done when:* token ids match for both streams, with the lever-moved marker carrying its
value.

**T9. Throughput, with the regression floor.**
A gate that passes a slowdown is worse than no gate: run 35431417507 passed every
correctness control on a change that made serving 14x slower. The floor is a control.
*Done when:* per-user tok/s at 2 users is reported against 1 user, floor enforced.

## What this does not fix

At 4 users and 131k, the arithmetic puts the cycle at 108.8 ms and 111.2 tok/s even with
batching working perfectly, because batching amortises the **weight** pass and that is
already inside the floor. Reaching 150 needs bf4 KV *and* roughly all 20.1 ms of GDN
machinery removed, on top of everything above. This sheet is necessary for the target and
nowhere near sufficient, and that should stay visible while working through it.
