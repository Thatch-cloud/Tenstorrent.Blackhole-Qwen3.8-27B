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

## T1 result, and a re-scope of T5

**T1 is answered, and it is not `dflash_device`.** Run 35434220264, with a prompt
finally above the 2048-token history window, reached `serving_lifecycle._execute` and
raised there:

```python
if (self.request_id is not None or len(scheduled.scheduled_new_reqs) != 1
        or scheduled.scheduled_cached_reqs.req_ids ...):
    raise ValueError('Fast serving requires one complete fresh prefill')
```

Two clauses fire: the single slot is occupied once user 1 is admitted, and vLLM
schedules both new requests in **one** step so `scheduled_new_reqs` has two.
`from_prefill` and `dflash_device` are downstream and were never reached, so earlier
claims that the device rejected the second request were wrong.

### Two earlier runs were invalid, and said so only after the prompt was fixed

`from_prefill` computes `feature_start = len(prompt) - 2048`. The bench builder emits
one phrase per eight units at a real **5.021 tokens per repetition**, so `prompt: 3072`
produced 1928 tokens and `feature_start = -120`. Runs 35433428038 and 35433989496 both
failed on that arithmetic and **would have failed identically at one user**. `prompt:
4800` gives about 3012 tokens and `feature_start = +964`.

### T5 is bigger than this sheet estimated

The sheet called T5 "contained; the module is 6 KB and has a CPU test". That was wrong,
and an attempt at it was reverted rather than left half-finished. Three separate
problems hide behind "make the slot a dict":

1. **`_execute` needs N simultaneous captures.** It currently does
   `with self.capture.capture():` around one. Whether the capture machinery supports two
   concurrent captures is **unknown and unverified** - it is exactly the kind of
   assumption that has been wrong repeatedly here.
2. **`_sample` assumes one seed**: `result.req_ids != [self.request_id]` and
   `len(result.sampled_token_ids) != 1`. With two prefills it receives two.
3. **`FastWorkerHook` binds to `self.worker`.** Once `self.hook` is set, `_execute`
   simply delegates. A second hook on the same worker is the real architectural
   boundary, and it is not bookkeeping.

Several per-request validations also assume a single request arithmetically, not just
structurally: `scheduled.num_scheduled_tokens != {new.req_id: len(prompt)}` is an
equality against a one-entry dict, and `total_num_scheduled_tokens != len(prompt)` is a
sum over one.

**Revised order**, smallest verifiable step first:

- **T5a** Can the capture factory produce two concurrent captures at all? A CPU test
  against the real factory, no device. *This gates everything else and is currently
  assumed.*
- **T5b** Per-request bookkeeping with the hook still singular, failing by name when a
  second would be needed. CPU-testable.
- **T5c** `_sample` for N seeds and N bridges.
- **T6** The hook. Unscoped until T5a says whether captures can coexist.

## T5a: ANSWERED. Two concurrent captures are structurally impossible

No device needed; the source settles it.

```python
def capture(self):
    if self.started or self.closed             or hasattr(self.model, '_qwen_dflash_prefill_capture')             or hasattr(self.model, '_qwen_target_feature_capture'):
        raise ValueError('One non-nested native prefill capture required')
```

`PrefillWindowCapture.capture()` installs itself **as an attribute on the model**
through `instance_overrides`, and monkey-patches `model._forward_prefill_chunk_masked_tp`
with its own wrapper. There is one model. The guard exists precisely to stop a second.

It is not incidental. The wrapper carries per-capture sequence state:

```python
if not self.active or chunk_start != self.cursor:
    raise ValueError('Prefill chunks must execute once in absolute sequence order')
self.cursor += valid_len
```

One `cursor`, tracking absolute chunk order for one sequence. Two interleaved prefills
break that ordering check even if the attribute collision were solved.

**So T5b as written is unreachable**: it was going to hold N captures, and N captures
cannot exist. The boundary is below `serving_lifecycle`, in how prefill capture attaches
to the model.

Three ways on, none small:

1. **Serialise the prefills.** Admit both, prefill one at a time, then decode both. The
   capture is only live *during* prefill - `_sample` sets it to None once the bridge is
   built - so the constraint may never be violated. Cheapest thing that could work, and
   it would make this a scheduling change rather than a capture-machinery one.
2. **Per-sequence capture.** Key the model attribute and the cursor by request. Touches
   `instance_overrides` and the chunk wrapper.
3. **Capture-free second prefill.** Almost certainly not: the features *are* the DFlash
   input.

Option 1 first, and the question it needs is observable: does `scheduled_new_reqs` ever
carry two, or can the scheduler be made to deliver them in separate `_execute` calls?

## Iteration speed, fixed

The serving CPU suite runs **locally in 15 seconds** with `py -3.10` (see
[[tt-rig-benchmark-tooling]]). It had been running as a 10-minute CI round trip all day
because `python` here is 3.7 and cannot parse the tests' PEP 604 unions, which presents
as a test failure rather than a version problem. Every remaining lifecycle and policy
task - T5b, T5c, most of T2 - is now a seconds-long loop. Only device behaviour, timing
and capture still need the rig.

## T5b: prefills CAN be serialised, and the chain that blocks it is three links

CPU probe, run 35434952352, 50 seconds, real vllm `Scheduler`:

```
max_num_seqs=1, two queued -> new=['A'] cached=[]
A prefill                  -> new=['A'] cached=[]
B arrives                  -> new=['B'] cached=['A']
```

Three results:

- **`max_num_seqs=1` already serialises**: the scheduler admits one prefill and queues
  the other.
- **`max_num_seqs=2` with both queued batches them** into one step. That is the
  simultaneous-arrival case, and the only one that would need two live captures.
- **A later prefill arrives as `new=['B'] cached=['A']`** - B prefills while A decodes,
  with no change to anything.

And the capture constraint from T5a is **not violated** by that shape: `_sample` sets
`self.capture = None` once the bridge is built, so A's capture is already released before
B's is created. Only one is ever live. **Serialisation needs no capture surgery.**

### The three links

Serialised prefills are refused three times, in this order:

1. **`serving_lifecycle._execute`** - `scheduled_cached_reqs.req_ids` must be empty, so a
   new prefill alongside a decoding request raises "one complete fresh prefill".
2. **`serving_vllm_contract.admit_scheduler_output`** - the decode-side contract the hook
   runs under requires `not scheduled.scheduled_new_reqs` and exactly one cached request
   matching the ticket. It refuses B's prefill outright while A decodes.
3. **`FastWorkerHook`** - singular, bound to `self.worker`, and it is what executes decode
   at all.

Nothing here is capture machinery, which is the good news. All three are admission
contracts around a singular session. The bad news is that they are three coordinated
changes rather than the one the sheet assumed, and (2) is the one that decides whether
prefill and decode can interleave at all - which is the same question Lever N's M2 asks.

### Revised remaining work

- **T5b-i** `_execute`: accept a new prefill when cached requests exist. Local test.
- **T5b-ii** `admit_scheduler_output`: admit a step carrying one new prefill plus the
  resident decode. **This is the load-bearing one.**
- **T5c** `_sample` for N seeds and N bridges.
- **T6** the hook: N tickets on one worker.
- Simultaneous arrivals still batch both prefills into one step, so either the scheduler
  is constrained to one new request per step, or (1) and (2) must tolerate two.

### Cycle time, for the record

| question | before | now |
| --- | ---: | ---: |
| lifecycle and policy logic | ~10 min CI | **15 s local**, `py -3.10` |
| anything needing vllm | ~10 min CI | **50 s** CPU probe lane, no card |
| device, timing, capture | ~10 min | unchanged, genuinely needs the rig |

The CPU lane attaches no device, loads no weights, and sits outside the exclusive
concurrency group so it cannot queue behind hardware work.

## T6 redefined: the fix is a batched verifier, not a second hook (2026-09-19)

Three probes settled what two-user serving actually requires. All are CPU-only,
driving the real `Scheduler` and the plugin's `TTScheduler`, ~50 s per lane.

| Question | Run | Measured |
| --- | --- | --- |
| Two prompts arriving together | 35436384975 | TTScheduler batches them: `new=['A','B']` in one step |
| A prompt arriving during decode | 35435453374 | Serialised: `new=['B'] cached=[]` |
| Two established requests decoding | 35436807668 | Batched every step: `cached=['B','A']`, `counts={'B':16,'A':16}`, `total=32`, both proposal sets in `scheduled_spec_decode_tokens` |

Run 35436193682 on hardware admitted both requests and got both into `_execute`,
which no previous run managed, then failed on `len(scheduled_new_reqs) != 1`. That
is the first row, not the routing the previous commit fixed.

### What this means

The decode step carries **two requests at once**, not alternating ones. So the two
candidate designs are:

**Alternation** (Lever N section 3.3 item 2) - one decode per step, each device
execution stays batch 1, every existing shape is reused. **Arithmetically ruled out
for this goal.** Decode is weight-bandwidth bound: 19.92 GB of dense projections per
step against 512 GB/s per card is a ~19.5 ms floor per weight pass, and alternation
spends one weight pass per user per round. Per-user cycle becomes N x the single-user
cycle, so per-user rate is the single-user rate divided by N. Even at the optimised
60 ms cycle that is 100 tok/s at two users and 50 at four, against a 200 tok/s target.
Alternation is a correctness fallback, never a path to the goal.

**Batched verify** - one weight pass serves every user's verifier rows, which is the
only reason the 60 ms budget can hold for four users simultaneously. This is the fix.

### What blocks the batched verify, read from the code

- `dflash_device` already accepts `block_rows in (8, 16, 32)`, so a 32-row block
  exists. It is **one request's 31 proposals** (`max_drafts = block_rows - 1`), not
  two requests' 16 + 16.
- `self.position`, `self.history_rows`, `feature_start` and `window` are all
  single-valued, and the feature tensors are checked as
  `value.shape[2] != window['rows']`. One contiguous captured KV window is shared by
  every row. Two users at different positions need two windows.
- The verifier mask is `(1, 1, 32, K)` with `32 <= K <= 2080`. A block-diagonal mask
  over two users needs both windows in K, and the capture window is up to 2048 rows,
  so two users need up to 4096 against a 2080 cap.
- `serving_vllm_contract.admit_scheduler_output` asserts
  `list(cached.req_ids) != [request_id]`, single resident decode.

So T6 is: give the verifier a batch dimension over independent KV windows - per-row
window tables, a block-diagonal mask wider than 2080, and feature tensors gaining a
batch dim - then widen the admission contract to match. It is a device-side change.

### Still required regardless

The one-in-flight rule from Lever N section 3.3 item 1, because the fast prefill path
takes one capture and executes one prompt, and simultaneous arrivals are batched.
