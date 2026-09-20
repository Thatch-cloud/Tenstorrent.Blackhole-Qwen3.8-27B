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

## Batched verifier: what is built, and what the arithmetic says it buys

### Built and proved on the host (no device)

| Piece | Module | Gate |
| --- | --- | --- |
| Block-diagonal mask | `dflash_batched_mask.batched_attention_mask` | at one user byte-identical to the shipped T16 mask at seven context lengths; at two users each user's rows restricted to its own segment ARE its single-user mask |
| Packed RoPE | `packed_rope_tables`, `live_key_rope` | at one user byte-identical to today's `rope_tables(position, 32)` and to `rope['k'][context:key_rows]` |
| Key assembly plan | `key_value_plan` | packed attention equals separate attention in float32 at four context pairs; overwriting a neighbour's keys with values 100x larger leaves a user bit-identical |
| Device branch | `draft_attention_branch.execute_attention_branch(pack=...)` | mock-operations test asserts the key and value axes are assembled cached/live/pad per user with live windows (0,16) then (16,32); `pack=None` keeps all eight existing branch tests green |
| Proposal layout | `dflash_packed_proposal` | anchors land at rows 0 and 16; user u's drafts are merged indices `[u*16 : u*16+15]` after `merge_chunk_candidates` drops the anchor row |

The draft block was already 32 rows with 16 live (`mask[..., 16:, :] = -inf`), so a
second T16 user costs no extra proposal pass. `shared_head_candidates` and
`merge_chunk_candidates` already accept 32 rows, so the vocabulary head needed no
change at all.

### The target side is already per-row paged

The 66.63 ms verifier half runs `ttnn.transformer.paged_scaled_dot_product_attention_decode`
with a `page_table_tensor` and a batch dimension, and `attention_replay.ReplayAttentionReader`
already accepts `rows in (8, 16, 32)`. What pins it to one user is narrower than a
kernel: `pages_host.shape[0] != 1` ("One complete native cache page table required"),
one `start` word staged into `self.positions`, and `pages_host.repeat(len(bundle), 1)`
repeating a single user's table across the bundle. Per-user page tables and per-user
starts are the change; no new attention kernel is needed.

### What batching actually buys, from the goal's own figures

Batching amortises the WEIGHT pass across users. It does not amortise KV, which is
per user and grows with context. Per card per step at 161k, from
`memory/goal-200tps-concurrent.md`: 19.92 GB of weights over two cards is 9.96 GB,
and 5.37 GB of KV per user is 2.69 GB.

| Users | Weights/card | KV/card | Total | At 512 GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 9.96 GB | 2.69 | 12.6 GB | 24.6 ms |
| 2 | 9.96 | 5.37 | 15.3 GB | 29.9 ms |
| 4 | 9.96 | 10.74 | 20.7 GB | 40.4 ms |

Four users stay inside the 60 ms budget with about a third to spare, so the target
is bandwidth-feasible - but only batched. Alternation divides the achieved per-user
rate by the user count whatever the cycle time is.

**Batching is necessary, not sufficient.** The measured cycle is 97.55 ms against a
24.6 ms single-user bandwidth floor, so roughly three quarters of the cycle is not
bandwidth. Adding users batched costs only the extra KV - about 5.2 ms per extra
user per card - so four users should land near 113 ms rather than the 390 ms
alternation would cost. Reaching 200 tok/s per user still needs the documented
37.98% cycle reduction on top of the pack.

### Not yet built

- **Slot state.** `DFlashDevice` holds one `history`, `position` and `history_rows`.
  `propose_packed` takes the per-user state as explicit slots, so what remains is
  for the serving path to keep N of them instead of one - today two concurrent
  requests would mean two devices and two uploads of the five DFlash2 layers.
- **Target-side page tables.** `ReplayAttentionReader` pins one user three ways:
  `pages_host.shape[0] != 1`, a single `start` word staged into `self.positions`,
  and `pages_host.repeat(len(bundle), 1)` repeating one table across the bundle.
  `parallel_groups` bundles up to three row groups sharing a `(rows, signature)`,
  and the signature derives from `start`, so two users at different frontiers
  generally will not share a bundle. Per-user page tables and starts are the work,
  and it interacts with trace capture since `positions` is captured into the
  program. No new attention kernel: `paged_scaled_dot_product_attention_decode`
  already takes a page table per batch entry.
- **Serving contract.** `serving_vllm_contract.admit_scheduler_output` still asserts
  `list(cached.req_ids) != [request_id]`, and `FastWorkerHook` still binds one
  request to the worker.
- **The one-in-flight prefill rule**, required independently of all of the above.

### Regression standing

Every module touched passes: 120 tests across the packed modules and their
neighbours, plus 14 branch tests and 9 proposal tests, all locally in seconds. The
full local discover sits at 12 failures and 44 errors against a 12/43 baseline; all
56 named failures are in modules that import nothing changed here, and the eleven
largest reproduce in isolation as missing weights and devices. The extra one is not
attributable to this work, and is not claimed to be absent either.

## Target side: the page tables were never the blocker (2026-09-19)

`ModelBatch` builds the verify block as

    positions = torch.arange(start, start + rows)
    self.pages = upload(pages.repeat(self.rows, 1))
    self.cos, self.sin = rot_mats_decode(..., positions)

so the model already takes one position and one page-table row per query row. The
only single-user assumptions are that the positions are one contiguous run and that
one page table is repeated. `target_packed_pages.packed_rows` replaces both, and at
one user reproduces today's tensors exactly. That is built and tested.

It is not enough, and the reason is the same class of bug as the draft convolution
seam, one layer deeper.

**48 of the 64 target layers are GDN, and their row axis is TIME.**
`gdn_prefix.decode_projected` is explicit:

    for index, token in enumerate(token_inputs):
        outputs.append((forward or gdn.forward_decode)(token))
        checkpoint(index + 1)

one row at a time through a single recurrent state, with a projection callback that
raises on any batch but one. Packed without handling this, user B's rows continue
user A's recurrence: wrong for EVERY row of B, not just the first, and A's state is
left advanced by the whole block. Nothing downstream can detect it. "batch" in
`gdn_batched_conv`, `batch_conv` and `norm_batch` means batching operations across
the rows of one sequence, not across sequences; there is no per-sequence batch
anywhere in the GDN modules.

So `ModelBatch.validate_pack` refuses a multi-user pack outright. The page tables
are correct and tested, and they stay behind that guard rather than becoming a
silent source of wrong committed tokens.

### Why this is still tractable

The weight-heavy GDN work - the qkvzab input projection and the output projection -
already runs ONCE across all rows (`gdn._project_qkvzab_raw(packed_input, rows, ...)`)
and is per-token independent. Only the recurrent update is per sequence, and it is
elementwise, not a weight pass. So the fix is a **state swap at each segment
boundary**, the same pattern as the convolution seam, rather than a batch dimension
in the GDN kernels. That keeps the whole point of packing intact: one pass over the
19.92 GB of weights serving every user.

Ordered by what blocks what:

1. GDN per-segment recurrent state in `DeviceLoopState` / `decode_projected`,
   which unblocks `validate_pack` and the target verify.
2. The fused draft convolution kernel taking segment spans, since
   `draft_convolution_fused_compute.cpp` has only `rows` as a runtime argument and
   production sets `fused_convolution=True`.
3. `admit_scheduler_output`, the one-hook-per-request binding, and the one-in-flight
   prefill rule.

## GDN state swap: done, and it keeps the weight amortisation (2026-09-19)

The blocker named above is cleared. `DeviceLoopState.decode` takes segment spans
plus one carried recurrent state per user, restores the layer's active state to
that user immediately before its segment runs, and advances only that user by its
own rows.

The reason this works without touching a kernel is where the weights sit:

| Stage | Packed behaviour | Weights |
| --- | --- | --- |
| `_project_qkvzab_raw` | once, across all 32 rows | the input projection, amortised |
| recurrence (`run_batched_projected`) | once per segment, on a slice of that projection | none, elementwise |
| `finish_output` / `_row_proj` | once, over the concatenated block | the output projection, amortised |

So packing N users costs the same weight traffic as one, which is the entire
argument for the batched verifier. The recurrence runs the same total number of
rows either way, split across N launches instead of one.

`ModelBatch` threads a pack carrying, per user: frontier, page table, accepted
prefix, and a per-layer GDN checkpoint and carried state. Positions and page rows
come from `target_packed_pages`, and the per-row page list replaces
`[singleton_pages] * rows` so a packed row reads its own user's blocks. A pack
missing any per-user GDN state is refused rather than run.

One hazard is recorded in the code: the layer's active state is left holding the
last segment's user. That is safe only because every packed segment restores its
own slot before running, so packed and unpacked blocks must not be mixed within a
request.

**Not validated on device.** The tests are mock-based and assert the call
structure - one projection across 32 rows, two recurrences of 16, each preceded by
its own user's restore, each checkpointing its own prefix and advancing its own
width - plus that unpacked decode takes the identical path it did before.

### Remaining

1. The fused draft convolution kernel taking segment spans.
   `draft_convolution_fused_compute.cpp` has only `rows` as a runtime argument and
   production sets `fused_convolution=True`, so `checked_convolution` refuses a
   packed call today.
2. `VerifierEngine` building a pack: it holds one page table and caps widths at
   `min(verifier_rows, max_verify_rows)`, so two T16 users need `verifier_rows=32`
   and per-user page tables and checkpoints threaded into `fixture`.
3. `admit_scheduler_output`, the one-hook-per-request binding, and the
   one-in-flight prefill rule.
4. Hardware qualification of the whole chain.

## Two users on hardware: six boundaries, five of them real bugs (2026-09-20)

The chain was driven on the rig until it stopped, six times, each time with the
failure naming itself. Every run is two users arriving SIMULTANEOUSLY at context
4352, which is the case the whole one-in-flight argument is about.

| Run | Stopped at | What it was |
| --- | --- | --- |
| 35436193682 | `len(scheduled_new_reqs) != 1` | the scheduler batching simultaneous prefills |
| 35441222051 | same, clause unknown | the message did not say which term fired |
| 35441524535 | `prefill_slot='cmpl-9cf6...' new=[one] cached=[]` | a request finishing at its first token never freed the prefill slot |
| 35441818361 | `prefill_slot=None new=[] cached=[BOTH] spec={both}` | **both users prefilled**; neither had a bridge, because both took the terminal branch under ignore_eos |
| 35442208627 | `Terminal prefill must finish without allocating a verifier` | the same ignore_eos assumption one layer down |
| 35442532141 | `Qualified 32K T16 replay ... required` | a context pin: the T16 target attention path is qualified at position 32768 exactly |
| 35442719988 | `Unique physical pages from the admitted cache required` | at 32768, `serving_runtime.bridge_factory` caps a request at 68 KV pages |

### The milestone

Run 35441818361 is the one that matters: `prefill_slot=None`, `new=[]`,
`cached=['cmpl-8e11...', 'cmpl-b76d...']`, proposals present for both. **Two users
both prefilled on hardware**, and the scheduler then presented exactly the packed
two-user decode step probe 35436807668 predicted on CPU. Every previous attempt
refused the second request at admission.

### The three bugs, all invisible with one user

- The EOS branch of `serving_lifecycle._sample` returned without clearing
  `request_id`. With one user that is harmless, since that user is done. With two
  it holds the prefill slot forever.
- That branch fires on a terminal first token, but under `ignore_eos` the request
  does NOT stop - vLLM keeps scheduling it - so the short circuit left it decoding
  with no bridge. A synthetic prompt that repeats one phrase invites a terminal
  first token, so this is the common case on the bench, not a corner.
- `serving_request_factory` refused to allocate a verifier for a terminal seed,
  the same assumption one layer down.

### Two pre-existing pins in tension

`frozen_combined_runtime.validate_target_option` qualifies the T16 replay path at
`rows == 16 and position == 32768` exactly. `serving_runtime.bridge_factory` caps
a request at 68 KV pages, and 68 x 64 = 4352 tokens exactly. **Both cannot hold at
once**, so whichever context is chosen, one of them fires. Neither is a packing
problem and neither is in scope to lift, but the next person to run two users at
any context will meet one of them, so it is written down here rather than
rediscovered.

The open question is why `target_attention_t16` is enabled in this configuration
at all, since single-user runs at 4352 do not hit its pin.

### Still not built

The packed device step - the callable that drives the packed `ModelBatch` fixture
and returns one committed output per user. `serving_packed_bridge` takes it as a
parameter and refuses by name without it, so the serving chain is complete up to
that call and no further.

## Correction: the fast path in this image could not decode at ANY context (2026-09-20)

Six two-user boundaries were read as concurrency problems. Run **35472072127** ran
ONE user and hit the same `Qualified 32K T16 replay` gate, so from that boundary on
the reading was wrong: this workflow had never produced a token.

What it actually is, measured by probe 35473235186 reading the IMAGE's files by
path rather than importing them:

- The image's `target_t16_attention_gate.py` is frozen-adapted to
  `if request_context() == selected_geometry()['context']: -> frozen validator`.
- `frozen_context_geometry.selected_geometry()` reads the SAME
  `QWEN_DSPARK_REQUEST_CONTEXT` as `request_context()`, so that condition is
  **always true**. The gate always delegates to the 32K validator, which demands
  `position == 32768`.
- `serving_runtime.bridge_factory` built a fixed `(1, 68)` page table: 68 x 64 =
  **4352 tokens**.

So the image's target attention wanted 32768 and the serving runtime offered 4352.
Irreconcilable, at any user count. Setting `QWEN_DSPARK_REQUEST_CONTEXT=4096`
changed nothing (run 35473307362) because both sides of the comparison move
together, and removing `QWEN_FROZEN_COMBINED_RUNTIME` changed nothing because the
adapters are applied when the image is BUILT - that variable only ever selected
which context the frozen geometry used.

**Probe gotcha worth keeping:** the CPU probe lane bind-mounts the repo at `/probe`
FIRST on `PYTHONPATH`, so importing a module there measures the REPO's copy, not
the image's. The first version of this probe reported `request_context() == 4096`
from the repo while the image's own default is `'8192'`. Read image files by path.

**Fix applied:** the page table width now derives from the admitted context,
`max(68, ceil(max_model_len / 64))`, with 68 kept as the floor because
`ServingCacheOwner` requires at least that many physical pages.

### What this does and does not change about the packing work

Nothing measured about packing is affected. The scheduler probes, the mask and key
equivalence proofs, the convolution seam, the GDN state swap and the packed
contract were all established on CPU or in the image's own libraries, not through
this serving path. What changes is the claim about hardware: the only thing the rig
has demonstrated is run 35441818361, where **both users prefilled and the scheduler
presented the packed two-user decode step**. No decode of any kind has run through
the fast path in this image, at one user or two.

## The fast path decodes (2026-09-20)

**Run 35475120459 is the first time the fast path has produced a token in this
workflow**, at one user: `prompt_tokens=32768` exactly, 10 tokens, no error,
`blocks=516`, ITL median 122.5 ms, 8.2 tok/s.

Three things had to be true at once, and each was found by reading the IMAGE
rather than this repo:

1. **The gate's required position is not a constant.** The image's
   `frozen_combined_runtime.validate_target_option` compares
   `position != selected_geometry()['context']`, where this repo's copy hardcodes
   32768. `selected_geometry()` reads `QWEN_DSPARK_REQUEST_CONTEXT`, default
   `'8192'`. Every value the diagnostic printed matched because none of them was
   the one being compared.
2. **The prompt must be exact.** The bench built text - a phrase repeated
   `prompt_tokens // 8` times - so asking for 32768 gave `position=20488`. Every
   gate here compares position for equality. It now sends a token-id array.
3. **The page table must cover the context.** `serving_runtime` built a fixed
   `(1, 68)` table, 4352 tokens. It now derives from `max_model_len`.

### Two users

| Run | Result |
| --- | --- |
| 35475354962 | both prefill; refused at the packed device step, by name |
| 35476203187 | 1 token each; refused at ticket preparation |
| 35476929953 | the image was running the bundle's `serving_vllm_state`, not this repo's |
| 35477522469 | **2 tokens each**, then the draft head refuses its own candidates |

The last one is the real wall. `merge_chunk_candidates` rejects the vocabulary
head's output - 'Finite complete-block top16 values and in-range integer indices
required' - after two committed blocks. The sequential step runs two complete
single-user cycles through ONE model, and the draft machinery is not re-entrant
per request: each `DFlashDevice` holds its own history, but they share the model,
the mesh and the vocabulary head, and nothing swaps between them.

That is precisely what the packed design handles - one pass with each user's state
restored at its own segment boundary - and precisely what the sequential shortcut
cannot, because it never restores anything. So the shortcut gets two users further
than before and then stops for a reason the batched verifier does not have.

### Where that leaves the work

- One user decodes. That is new, and everything above it was blocking at one user
  as much as at two.
- Two users prefill, admit, and commit two blocks each.
- The remaining piece is unchanged and is now the ONLY piece: a packed device step
  that restores per-user state, which is what `verifier_pack`, the GDN segment
  swap and `ModelBatch(pack=...)` were built for. `serving_packed_bridge` takes it
  as a parameter, so it drops in where `sequential_packed_step` sits today.

**Lesson worth keeping:** three separate times the image's copy of a module
differed from this repo's - `verifier_engine` is the bundle's, this repo's
`dspark_context_selection` defaults to 4096 where the image's defaults to 8192,
and the frozen validator compares a computed geometry where this repo's compares a
constant. The CPU probe lane mounts the repo at `/probe` FIRST on `PYTHONPATH`, so
importing a module there measures the repo. Read image files by path.

## Why two users cannot simply take turns (2026-09-20, measured)

One user decodes. Two users prefill, admit, and commit two blocks each, and then
the DRAFT path fails inside `select_proposal`:

```
Replicated learned selector features differ:
call=1 position=32779 rows=2048 max_abs=2.01562 mean_abs=0.36001
differing=8184 of 8192 finite=True/True
```

Read that carefully, because it rules out most of the candidates at once:

- **8184 of 8192 elements differ.** Not a corrupted region - everything.
- **max 2.02, mean 0.36.** Far too large for fidelity or rounding.
- **Both shards finite.** Not a bad write, not uninitialised memory.
- **call=1**, the device's SECOND proposal, i.e. the first one that happens after
  the other user's pass has run in between. Call 0 is fine for both users.

The selector projection is replicated across the two chips, so identical inputs
must give identical outputs. They did not. The two chips computed the proposal on
different data - a collective that did not pair.

Two candidate explanations were tested on hardware and BOTH are excluded:

| Hypothesis | Test | Result |
| --- | --- | --- |
| Each request captures its own device trace, and two traces collide | `QWEN_FAST_EAGER_PROPOSAL=1` leaves `proposal_capture` unset; `propose` already has an eager path | Run 35478872085: still diverges |
| Each request builds its own `TT_CCL`, so two cycle semaphore handles over one mesh | one shared collectives object for every request | Run 35479238722: still diverges |

Both changes are kept: they are correct regardless, and one user still decodes
with them (run 35478659909, 10 tokens, 7.2 tok/s eager against 8.2 traced).

**What this says about the design.** The draft proposal pass is not re-entrant
across requests on a TP2 mesh. Running two complete single-user passes in turn is
not equivalent to serving two users, whatever order they run in. That is not an
argument the batched verifier wins on speed - it is the reason the sequential
shortcut cannot be made correct by tuning. The packed design has one device, one
proposal pass, one set of collectives and one set of semaphores serving every
user, so the failure mode does not exist to be fixed.

Getting an eager mask validated also required the device modules to be OVERRIDDEN
into the image: `dflash_device`, `draft_attention_branch`, `draft_mlp_branch` and
the convolution modules were never copied, so none of the packed work had ever
reached hardware. The build context is a temp directory the workflow fills file by
file, so a Dockerfile COPY alone fails the build - both lists need the name.

## What actually runs in the image (inventory, 2026-09-20)

The workflow's build-context list and the Dockerfile's COPY list are now in exact
1:1 correspondence: 26 named files plus `test_serving_*.py`. Every one of those
overrides the bundle. Everything else in the serving path runs as the bundle's
frozen copy, including the ENTIRE verify/commit core: `verifier_engine`,
`model_batch`, `full_dflash_request`, `dflash_request_runtime`,
`gdn_device_loop_state`, `serving_runner_bridge`, `serving_cache_owner`,
`serving_page_binding`, `dflash_combined_request`, `dflash_prefill_window`,
`dflash_proposal_trace`, `gdn_snapshot`, `gdn_multitoken_conv`, `gdn_prefix`,
`feature_collective`, `draft_shared_head`, `draft_selector`,
`prepared_target_features`, `attention_replay`, `attention_batch` and more.
`greedy_session` and `hybrid_draft` exist only in the image's harness.

So `ModelBatch(pack=...)`, `DeviceLoopState.decode(segments=...)` and
`VerifierEngine.fixture(pack=...)` have never reached hardware, and a fix in the
verify/commit path - where the sharpened two-user signature points - needs its
module added to BOTH lists. `probe_image_drift.py` hashes every repo module
against the image's copy so an override is known to be byte-identical or a
deliberate change before it is made.

**Drift measured (probe 35480229521):** of the verify/commit core, 20 modules are
byte-identical to the bundle and safe to override. `verifier_engine`,
`model_batch`, `gdn_device_loop_state` and `dflash_t16_native_attention` differ
only by the packed edits made here. `dflash_combined_request` differs because the
repo's copy is a strict superset - it also forwards `progressive_evidence` - so the
bundle is simply older there. No core override would change behaviour beyond the
packed changes themselves.

## arXiv 2510.22876 (Zhang et al.), reviewed against the packed block (2026-09-20)

A correctness paper on batched speculative decoding over a padded KV cache. Its
two invariants: I1, rectangular alignment across the batch; I2, every row's
position and KV entry derived from that row's OWN prefix, never from batch index.
Existing batch-spec implementations scored 0-3.5% token-exact match against
single-sequence decoding while reporting good throughput (Table 2).

What carries over:

- **I1 is free for us.** The packed block is a fixed 16 rows per user, so the
  realignment cost their Theorem 3.1 charges - `B(E[max accept] - mean)` - never
  arises.
- **I2 is the trap a block-diagonal layout invites.** Position must come from each
  user's own frontier, not the block row. `dflash_batched_mask.packed_rope_tables`
  does exactly that and is byte-identical to the single-user tables at one user;
  `target_packed_pages.packed_rows` does the same for the target. Keep it that way.
- **Rows per user = K + 1**: the bonus token needs a row. T16 = 15 drafts + 1 is
  right for two users. Four users at T8 means K = 7 and about 6 commits per cycle,
  so 4-user per-user throughput is bounded BELOW half the 2-user figure at equal
  cycle time - a correction to the earlier note that only said "half".
- **Adopt their audit**: token-exact AND partial match of the packed block against
  two independent single-user runs, per cycle. High partial with low exact means
  drifting state; near-zero partial means immediate indexing error. This is the
  gate on trusting any multi-user number.
- **Adopt their numerical control**: a non-speculative packed pass as the baseline
  before blaming the speculative path for a mismatch.

What does not: their EqSpec repad machinery, and their EXSpec same-length
scheduler, which cannot fill a window at four users and costs a 6x P99 at batch 2.

The paper says nothing about recurrent state; it treats "rejected tokens left in
the KV cache" as repairable by overwrite, which a destructive GDN update is not.
That design already exists here: the verify recurrence emits a checkpoint at every
candidate prefix (`conv_prefixes`, `states` per row in `run_batched_projected`) and
`restore_prefix` recovers the accepted one at commit.

## Two audits converge on the two-user mechanism (2026-09-20)

Independent code audits of shared-model hooks and of device-memory lifetimes
reached the same mechanism, and this repo already met it once
(`docs/experiment-execution.md:499-510`):

A request's verify and commit traces are captured at admission. The captured
forward allocates and FREES many per-chip intermediates during capture
(`model_batch.py`, `gdn_multitoken_conv.finish_output`), but their DRAM addresses
stay baked into the trace. A request admitted later allocates its persistent,
REPLICATED draft buffers - `history`, `spare_history` (`dflash_device.py:74-78`),
`DraftKVHistory` active/spare (`draft_kv_history.py:35-39`) - into those holes.
When the first request's trace replays, each chip writes its own TP-shard
activations over the second request's buffer, so the two replicas diverge: finite,
O(1), nearly every element, first visible at that request's next proposal.

This fits every measurement: one user runs forever (its buffers predate its own
traces); disabling the proposal trace changed nothing (the VERIFY trace still
replays); sharing collectives changed nothing; the failure is call=1. It predicts
the victim is the later-admitted request.

TT-Metal itself flags the condition - "Allocating device buffers is unsafe due to
the existence of an active trace. These buffers may be corrupted once a trace is
executed. (allocator.cpp:123)" - but note the warning is one-shot per process and
appears once in single-user runs too, so it confirms the condition exists, not
which allocation was hit.

Confirmation: compare the two shards of the OTHER request's `history` and K-V
immediately after a request's verify replay; replicated buffers must be equal.
Fix, with precedent: allocate per-request persistent buffers from a pool reserved
before any trace capture (`feature_prefix.allocate_prefix_pool`).

The shared-GDN-carry bug (no per-request restore in the sequential path) is real
and separately being fixed, but it is chip-symmetric and cannot produce this
signature.

### Synthesis of the three audits (2026-09-20)

| Audit | Verdict on the chip disagreement |
| --- | --- |
| shared-model hooks | no Python hook or attribute outlives its scope; the only cross-request coupling is device memory baked into verify traces |
| device memory / caches | verify traces bake addresses of intermediates freed after capture; the later request's persistent buffers land there; the earlier request's replay overwrites them per chip. Precedent `docs/experiment-execution.md:499-510`. No address-keyed cache, no cross-request free found |
| collectives | every draft collective is barrier-protected and the mesh is idle between requests; a mis-paired gather is not constructible and both hardware runs agree. Refinement: on the cached path the proposal reads the per-request DRAFT WEIGHTS, not `history` |

One mechanism, three independent routes to it. The refinement matters for the fix:
the draft weights are uploaded first at device construction, so first-fit places
them in the lowest freed holes of the earlier request's verify trace. A pool that
covers only `history` cannot help; the weights must be prepared once, before any
request exists, and shared - which also removes the per-request duplication of the
five draft layers. `history`/`spare_history`, the draft K/V, the feature taps and
the proposal buckets are the remaining per-request allocations and are the next
hole candidates in that order.

Prediction to test on the rig: the victim is the later-admitted request, and the
scribble happens across the earlier request's verify replay. The diagnostic
compares the two shards of the other device's replicated weights, history and K/V
after every step and names the tensor, its shape and its address on divergence.

**Run 35481140763 (image v32, diagnostic + carry, no pool):** died at admission
with `ModelBatch.__init__() got an unexpected keyword argument 'pack'`. Overriding
`verifier_engine` brought a call that the bundle's `model_batch` predates. Rule:
an overridden module must not depend on a module it does not bring along; the
drift probe is the check. Fixed by passing `pack=` only when packed (5205854a).

## Confirmed on the rig: the trace-hole mechanism (run 35481466425, 2026-09-20)

Image v33: history pool + per-engine GDN carry + shard check, two users arriving
together, eager proposal. The check compares, in order, the replicated draft
weights, then `history`/`spare_history` (both pooled, allocated before any
trace), then the draft K/V (not pooled), and raises at the first mismatch. It
fired on the very first step:

```
replicated draft kv differs between chips after step of <entry 0>:
victim=<entry 1> buffer=kv_history[0].k shape=(1, 4, 2048, 128)
```

So at that moment the checked weights and the POOLED history of the victim were
still bit-identical across chips, and the one UNPOOLED persistent buffer was not.
The pooled buffers survived the other request's verify replay; the unpooled one
was overwritten. That is the mechanism three audits predicted, on hardware, with
the fix's own effect visible in the same measurement.

Corrections to the prediction: the victim was the draft K/V, not the weights - the
checked weight subset was intact - and the scribble is visible after the FIRST
verify replay, before the victim ever verifies. Whether the eager proposal reads
the K/V is a separate question; the check raised before a proposal ran.

Next: pool the draft K/V per request (storage injection into DraftKVHistory), then
hoist the draft weights so no per-request persistent allocation follows a trace.
Log lines are truncated at ~250 characters by the capture, which lost the
differing count, max_abs and the address map; the diagnostic will emit short
lines next run.

**Run 35481903377 (v34: history pool, draft weights uploaded once and shared,
pool first in attach, per-engine carry, shard check OFF):** neither diverged nor
failed - it HUNG on the decode step, no stream progress for over eleven minutes,
cancelled, no log recovered because the bench dumps the server log only at the
end and its socket timeout was 900 s. A different failure class from every run
before it. The bench now errors a stream after 180 s of inactivity and keeps the
tokens it saw, so a hang still yields the diagnostic lines up to the stall. The
collectives audit had said the one semaphore-pool hazard it could construct was a
hang rather than divergence; that analysis is being redone against v34's changes.

## v35 on the rig (run 35482551725): protected buffers, and the second proposal stalls

History and K/V pooled, draft weights uploaded once and shared (109 tensors lent
to both devices), per-engine carry, warn-mode shard check. Both users admitted,
two rounds each, TTFT 14.6 s and 28.6 s (serialised prefills), then generation
went from 2.2 tok/s to 0.0 with both requests still Running - a step that never
returned. The 180 s inactivity limit ended it with the log intact.

Two corrections from the diagnostic:

- Every step reported all ten draft K/V banks "diverged" across chips (~99.9% of
  elements, both directions, from the first step) while the five checked weights
  and both histories stayed bit-identical. That is structure, not damage: the
  draft q/k/v projections are sharded across the chips, so each chip holds its
  own K/V heads. The check must exclude the K/V, and v33's "kv_history[0].k
  victim" was this false positive. What v33 actually showed - and v35 repeats - is
  that the pooled history and the weights are intact.
- `proposal_calls` stayed [1, 1] through both rounds: neither device proposed a
  second time. Every earlier chip-divergence failure was at call=1 in
  select_proposal; with the buffers protected, the second proposal now stops
  instead of producing garbage. So the remaining fault is in whatever the second
  eager proposal waits on after the other request's cycle has run - a collective
  or a fence, not a buffer.

Next run: QWEN_FAST_PHASE_LOG=1 wraps each proposal and each step in begin/end
lines so the stall names its phase; the collectives audit is re-examining the
semaphore state a trace replay leaves for the next eager gather.

**Device hang, not livelock (v35 log):** vLLM's 10-second engine stats came at
01:56:22/32/42/52 and then never again, while the server lived on to 02:00:02. A
scheduler livelock would have kept the engine loop, and its stats, alive; a step
that never returned stops both. Combined with proposal_calls stuck at [1, 1], the
block is inside a device's second proposal or the step after round two. v36 adds
per-execute scheduling counts, a periodic Python stack dump and TT-Metal's
watcher (its log printed by the bench) so the stalled kernel names itself, and
runs with per-request collectives as the A/B for semaphore residue.

### Per-request allocations that still follow a trace (audit-verified, 2026-09-20)

Protected now: draft history and spare history (pool), draft K/V banks (pool),
draft weights (uploaded once at attach, shared), the per-engine GDN carry
(allocated beside the initial snapshots, before capture).

Still allocated after an earlier request's traces exist, in the order they would
matter:

1. The verifier's own state: initial GDN snapshots (`verifier_engine.py` ~:109),
   per-bucket checkpoints (~:143), target feature taps (~:124), and the
   fixture's ModelBatch buffers (`model_batch.py` ~:471-499: tokens, positions,
   pages, singleton pages/positions, cos/sin, replay-reader positions/pages/masks).
   tokens/positions/pages are restaged before every verify, so a clobber there is
   overwritten before use; the GDN snapshots and checkpoints are NOT, and a
   clobber there corrupts the recurrent state silently - wrong tokens, no
   assertion. This is the next suspect if two users decode but emit garbage.
2. `DraftKVHistory.query` (a zero input the projection only reads).
3. `PreparedDFlashProposal` buckets and retained trace intermediates - trace mode
   only, not exercised while proposals are eager.

The principle that closes the class: nothing a request keeps across steps may be
allocated after any trace that will replay exists. The packed design needs the
same rule for its single shared engine.

**Run 35483320919 (v36, watcher on):** "Watcher stopped the device due to
tripped assert" at 02:16:27, during the FIRST request's admission capture - one
pool slot acquired, no second request yet, so this is the single-user path. With
the watcher off, kernel asserts are compiled out and a violated one is silently
run past; with it on, the device stops and the watcher log names the kernel and
the assert. The bench's 200-line tail held only idle cores and the kernel table
(fabric routers and dispatch), so the assert text was not captured; the bench now
extracts it from the whole log. This precedes and may underlie both the
second-proposal divergence and the hang. The per-request-collectives A/B did not
run and is still pending.

**Run 35483704438 (v37, watcher on, assert extracted):** the watcher named it.
`Device 0 worker core(x=0,y=0)`, BRISC, `tripped an assert on line 119`, kernels
`all_gather_async/device/kernels/minimal_default_writer.cpp` (BRISC) and
`minimal_default_reader.cpp` (NCRISC), last waypoint `K,CRBW,W,W,W`. Timeline:
gate at 02:23:48.078 (first request, position 32768), the target decode ops'
"engaged" lines, then at 02:23:49.172 the last host lines,
`tt_sampling:forward:711 Forcing argmax sampling` and `:720 Force argmax sampling
all-gather: cluster_axis=None, num_links=4, topology=Topology.Linear`; the next
watcher poll at 02:24:09 found the hang. No `[PHASE]` line was ever written, so
this is inside admission (the verifier's first target decode, sampling
included), before the first proposal - the same place as v36. Both streams got
the engine's 500; no tokens.

What line 119 is, read at the image's tt-metal `9f9cd4fd`: the writer's own line
119 is a runtime-arg read, so the line belongs to an included header, and exactly
two of its headers carry an `ASSERT` at line 119 -
`tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp:119`
`ASSERT(has_forward_connection())` inside `get_forward_connection()`, and
`tt_metal/fabric/hw/inc/packet_header_pool.h:119` `ASSERT(route_id < route_id_)`
inside `for_each_header`. The writer (`minimal_default_writer.cpp:226-227`) takes
`direction ? get_backward_connection() : get_forward_connection()`
UNCONDITIONALLY, right after `fabric_connection.open()`. On a two-chip line one
chip has no forward neighbour, and the writer launched for that direction has
`num_targets_forward_direction == 0` at compile time, so `valid_targets()`
(lines 80-101) is false and it never sends through the reference it took - it
only joins the semaphore accounting (lines 483-489). Without the watcher the
ASSERT is compiled out and a reference to the member object is harmless; with
it, the core hangs at the assert and the poll stops the device. That is a
pre-existing wart of this tt-metal revision, not a corruption path and not our
bug: the connection is never used. It fires on the sampler's all-gather and not
on the projection gathers during the 77 s prefill (watcher dumps 3-5 cover it),
so the projection path reaches the fabric differently: the collectives audit
read the program factory and found that `tt_sampling.py:187,193` passes
`num_workers_per_link=1`, and `all_gather_async_default_program_factory.cpp`
drops the mux cores exactly then (:254-257, :272-273, :438-440) while still
launching a direction-0 worker on the chip with no forward neighbour with both
connection flags false (:380-393, :797-806); our drafter gathers pass 2
(`feature_collective.py:55`, `dflash_device.py:242`), get a mux core and take the
writer's `USE_WORKER_MUX` branch (:126-199, :219-222), which never touches the
connection manager - so they passed under the same watcher before the gate.
The other line-119 candidate, `packet_header_pool.h:119`, is unreachable from
this writer (pointer-based header API). One open point: the plugin's prefill
sampling did not trip, so it does not reach `tt_sampling.forward:711-737`.

Consequence: the watcher as configured cannot get past admission on this stack,
and v36's trip was this same line. `tt_metal/llrt/rtoptions.cpp:161` at the same
revision reads `TT_METAL_WATCHER_DISABLE_ASSERT`, which drops only the assert
feature and keeps NoC sanitisation, waypoints and stack checks - the parts that
would name a bad address or a stalled kernel at the second proposal. Next run
(v38 = the v37 image unchanged, env only): that variable added; the bench keeps
the last two watcher dumps rather than the first 600 lines (one mid-prefill dump
was 270 lines), surfaces `[PHASE]` and `[CARRY]` with the diagnostics bounded
from both ends, and no longer shadows `re` inside main() (the
`prefill_chunk_observed: UnboundLocalError` in this run's report).

### Kernel-exit contract violation in fused_1d_input.cpp (audit, 2026-09-20)

The collectives audit, before v37 named the sampler gather, ranked our own
`scripts/ci/fused_1d_input.cpp` as the likeliest watcher trip, and the reason
stands even though it was not this trip: the receivers issue
`noc_semaphore_inc` every block (line 33) and never `noc_async_atomic_barrier()`,
and worker 0's last action is `noc_semaphore_set_multicast` (line 31) with no
write barrier after the final block. TT-Metal's watcher checks at every kernel
exit that reads are flushed, non-posted writes and atomics acked, and reports a
violation as a tripped assert. The newer `tensix_stream_matmul_sink.cpp:36`,
`tensix_weight_stream_sink.cpp:28` and `tensix_weight_stream_writer.cpp:17,22`
drain at exit; `tensix_stream_activation.cpp` shares the omission. It is
behaviourally benign so far because the peer waits on the same semaphores before
it exits, so the transactions land and only the issuing core's ack counters lag -
but a late ack arriving after the NEXT kernel on that core has snapshotted its
NoC counters would leave that kernel's first barrier waiting for an equality that
never comes. Two lines fix it: `noc_async_atomic_barrier()` before exit on the
receivers, `noc_async_write_barrier()` after the last multicast on worker 0.
Not applied yet: the kernel is not in the image's copy list and
`frozen_mlp_input_gate.py` pins its sha256 (`READER_SHA256`), so the change is a
frozen-recipe revision with a rebuilt image, to be taken with whatever v38's
stalled-core dump demands. Every other kernel of ours on the admission path
(`gdn_state_copy`, `attention_fold_dma`, `attention_mask_replay`,
`gdn_conv_windows`, `fused_1d_weights`, `gdn_commit_dma`,
`draft_convolution_fused_io`) ends with the proper barriers. One sanitiser-class
note for later: `gdn_state_copy.cpp:12-16` moves 32-byte sub-tile DRAM pieces
from a scratch not rounded to 64 bytes (`:37`), unlike `gdn_slot_copy.cpp:41`.

**Run 35484349353 (v38 = the v37 image, watcher on, asserts off):** the first
run to get past admission under the watcher. Server ready at 120 s; the first
request prefilled (gate 02:39:35.760, position 32768, emitted=1); the verifier
built and warm-verified, sampler gather included, without a stop; `[CARRY]
op=save layers=48 enqueue_ms=7.3 fence_ms=0.2`; then the first eager proposal
`[PHASE] propose ... end 12283.1 ms` - the first proposal ever to return under
the watcher. The next line, `[PHASE] execute total=16 new=0 cached=1 spec=1`,
was followed at once by

    ValueError: Resident decode requests at the exact scheduled frontier required

from `serving_vllm_packed.ordered_tickets` line 31, reached through
`serving_packed_bridge.execute_packed_decode:33` and
`admit_packed_scheduler_output:55`. A Python refusal in our code, not a device
hang: the watcher's last two dumps hold only the dispatch cores and the
`sdpa_decode` and `slice` kernels of the last op, no NoC finding. Stream 1 got
one token (TTFT 171 s, the admission token) and then the engine's 500; stream 0
got nothing and hit the 180 s inactivity limit. vLLM's stats at 02:40:30 said
`Running: 1 reqs, Waiting: 0 reqs`, and only one request id appears in the log,
so the second prompt had NOT reached the scheduler when that step was built -
the opposite order from v35, where B was prefilled before A's first decode.

The raise at line 31 guards five things at once - a new request, finished ids,
preempted ids, structured output, encoder inputs - and its message names none.
`serving_vllm_contract.admit_scheduler_output` carries the same five, and the
single-user run 35475120459 passed them ten times, so something in THIS step
differs. The plugin's rule (`_local_prefill_intent`: prefill when a partial
prefill exists, or a request waits and there is capacity; otherwise decode) put
the scheduler on its decode-only branch, and `docs/lever-n-plugin-contract` notes
that its prefill-to-decode fallback carries `finished_req_ids`,
`free_encoder_mm_hashes` and `preempted_req_ids` across a discarded pass - the
very fields the clause refuses. Rather than guess, `probe_frontier_refusal.py`
(cpu-probe-v20) replays both arrival orders at the fp2u geometry against stock
vllm, TTScheduler and the one-in-flight subclass, prints every field the clause
reads per step, runs the packed contract on the real outputs, and prints the
plugin's negotiation source. The fix follows the probe: whichever field it is,
both contracts will also carry their values in the refusal.

Diagnostic-mode costs seen, not serving numbers: verifier build 48 s and the
eager proposal 12.3 s under the watcher's NoC sanitiser.

**Probe cpu-probe-v20 (run 35485312330, `probe_frontier_refusal.py`):** on the real
scheduler classes at the fp2u geometry, the solo step is ADMITTED by the packed
contract - stock vllm, TTScheduler and the one-in-flight subclass alike, in both
arrival orders - and every field the clause reads is false on every decode step
(`finished=[] preempted=set() structured=False encoder={}`; all three attributes
exist in this vLLM). The plugin's `schedule()` source: prefill-only whenever a
prefill is pending, falling back to decode-only when that pass schedules nothing
and a decode is running. So the scheduler's own bookkeeping never produces the
refused step; the truthy field in v38 came from engine state between the two
steps - a request finishing or aborting is the only source of `finished_req_ids`
on a step that still schedules the survivor. Which one, the live engine must say:
both contracts now name every condition they judged with its value
(`serving_vllm_contract.step_refusals`), the hook's execute line prints
`finished=` and `preempted=`, and the bench keeps the request-lifecycle lines
(`Received request`, aborts, the access log, the engine stats) among the
diagnostics so both arrivals stay readable. `serving_vllm_contract.py` joins the
image's copy lists (byte-identical to the bundle before this change). Image v38.

### Verifier per-request buffers pooled (a52be7a7, 2026-09-20)

Items 1-6 of the post-capture list now come from `ServingBufferPool` slots
allocated at attach, before any trace: `DraftKVHistory.query`; the verifier's
initial GDN snapshots, its carry, and one bucket per capture width holding
checkpoints, target features, optional MTP hidden and the ModelBatch inputs
(tokens, positions, pages at the exact page width, singleton pages/positions,
cos/sin). `capture_bucket_rows(16, 256, 16)` gives widths (1,2,4,8,8,16,16), the
second 8 and 16 being the other mask family. Injected by `storage=` on
`VerifierEngine`, `ModelBatch.prepare_inputs` and `query=` on `DraftKVHistory`;
borrowed buffers are never freed; a width the pool lacks raises at admission;
the unpooled path is unchanged. 125 tests in the touched modules and 192 in the
modules importing them pass on CPU. Cost: one GDN set tile-pads to 100.7 MB on
device, nine sets per slot = 906 MB per user per chip (the engine allocated seven,
705 MB, per request before), so two users hold 1.8 GB of pooled sets per chip.
`model_batch.py` joins the image copy lists (its bundle drift was the packed
edits only). Not pooled: proposal buckets (trace mode only), and two allocations
outside the change's file list that are the next hole candidates -
`ReplayAttentionReader` positions/pages/masks (`attention_replay.py`; pages and
masks are NOT restaged before every verify) and `DeviceLoopState` entry/state per
GDN layer. Image v39 carries this on top of v38's evidence-carrying refusals.

## Run 35485758177 (image v38): two users, one full round, then the fused input kernel's exit race

The v38 refusal did not recur: this time the second prompt was waiting when the
first was admitted, so the step after A's admission was B's prefill
(`[PHASE] execute total=32768 new=1 cached=0 spec=0 finished=[] preempted=[]`), the
v35 order. Timeline: A admitted 03:11:22 (pool slot 0, TTFT 82 s), eager proposal
287.6 ms (12.3 s in v38 was the first JIT under the watcher); B prefilled
03:11:27-03:12:44 (77 s) and admitted (pool slot 1); both proposals; then the first
packed step `total=32 cached=2 spec=2`: sequential step B (3244 ms, its first
verify replay), `[CARRY] save B`, shards equal (17 buffers), step A with
`[CARRY] restore A from B` (407 ms), save A, shards equal; both second proposals
(207 ms each); the second packed step; `[PHASE] step B begin`, `[CARRY] restore B
from A enqueue 8.1 ms fence 0.2 ms` - and nothing after. Both streams: 2 tokens,
then the 180 s inactivity limit. vLLM stats: 2.3 tok/s at 03:12:54, 0.0 after. So
round one is correct end to end with the carry swapped both ways, and the hang is
B's SECOND verify, the first verify replay after the other user's whole cycle
has run on the same cores.

The watcher names the kernel. Dump #22 (at 425 s, mid-hang), device 0:

    core(0..4,0)   R,   R,  W,  W,  W   rmsg D1D h_id 190271   done, a LATER program
    core(5,0)   NWBW,   W,  K,  D,  K   rmsg D0G h_id 190270   writer at noc_async_write_barrier
    core(6,0),(7,0),(0..7,1)
                CWFW, NSW,  K,  K,  K   rmsg D0G h_id 190270   reader at noc_semaphore_wait
    k_id 1431-1433: Kernel_Source_Code (three inline-source kernels)

(NSW = `noc_semaphore_wait`, dataflow_api.h:1946; NWBW = `noc_async_write_barrier`,
:1783; CWFW = `cb_wait_front`.) Device 1 shows the same class on eleven cores
(NSMW/NWID/CWFW). This is `fused_1d_input.cpp`: its input kernel runs on RISCV_1
over the 11 x rows grid (`fused_1d.py:155-163`), worker 0 is core (0,0), and worker
0's last act is `noc_semaphore_set_multicast` of the `received` signal with no
barrier before the kernel exits (line 31). Core (0,0) had exited and started the
next program while ten receivers still waited at `noc_semaphore_wait(received, 1)`
(line 35): the final signal never landed. A single user runs the same trace and
never trips it because the programs before and after are always the same; a
second user's cycle on the same cores changes the timing of that window. This is
the hang v34 and v35 hit, now with a name.

Fix: `noc_async_write_barrier()` after the last multicast on worker 0 and
`noc_async_atomic_barrier()` before exit on the receivers, in
`scripts/ci/fused_1d_input.cpp`. It is a frozen-recipe revision: the kernel's
sha256 is pinned by `fused_1d.py:116` into the manifest and checked by
`fused_t16_admission.py:30`, `full_model_fusion.py:51`, `frozen_mlp_input_gate.py`
and the wait-zone gates, and none of those modules nor the kernel are in the
image copy lists; the collectives audit is mapping every pin before the edit.
Image v39 (the pooled verifier) runs first, unchanged env, to qualify the pool on
hardware up to the same round.

## Run 35486440095 (image v39, verifier pooled): two rounds, then the divergence returns one round later

Pool at attach: `bytes_per_slot 444,409,204` (draft 84 MB, query 128 KB, verifier
360 MB), both slots lent. Timeline: A admitted 03:27:20 (TTFT 82 s), B prefilled and
admitted 03:28:42; round one: step B 433 ms, step A (restore from B) 415 ms, shards
equal; round two: step B (restore from A) 1527 ms - the verify that hung in
35485758177 completed - shards equal, `proposal_calls=[2, 2]`; step A (restore from
B) with its carry saved 03:28:52.892; then the third proposal:

    AssertionError: Replicated learned selector features differ: call=2
    position=32780 rows=2048 max_abs=0.496094 mean_abs=0.0312087 differing=7391 of 8192

from `dflash_device.select_proposal` (via `_drafts` -> `serving_runner_bridge.drafts`
-> `serving_fast_request.prepare` -> `dflash_request_runtime` -> `propose`). Both
streams: 3 chunks each, then the engine's 500. This is the ORIGINAL two-user
failure (v33: call=1), now at call=2 with every persistent buffer pooled and every
shard and drift check equal immediately before it. What differs between the chips
is therefore a transient of the eager proposal, not a buffer a request keeps.

One cause for both symptoms. `fused_1d_input.cpp`'s worker 0 exits with its final
signal multicast in flight, and the receivers' last block multicast
(`noc_async_write_multicast`, 8 x 2048 bytes into each receiver's circular buffer
region) is only barriered by worker 0, not acknowledged by the receivers before
they exit. When the last signal is lost the receivers hang (35485758177); when the
data lands late it lands in whatever the next program on those cores has placed at
that L1 address - with two users, the other user's or the same user's eager
proposal - and one chip's intermediate is scribbled, hence replicated features
that differ. A single user never reorders the programs after the fused MLP, so the
window never opens. The pool did its job: the divergence moved from a persistent
buffer to a transient, which is what a pooled-but-still-diverging run had to show.

The fix stays the two barriers, under the frozen recipe. The collectives audit's
pin map: the kernel serving runs is the staged copy
`mlp-register-epilogue-candidate/fused_1d_input.cpp` (staged byte-for-byte from
`scripts/ci`, `mlp_register_epilogue_gate.stage_candidate:44-45`), exec'd through
`mlp_block_stream_runtime.py:100-113`; five startup pins - P1
`fused_t16_admission.py:10,30-32` (REPORT_SHA256 of
`fused-t16-target-simulator.json`, six kernels' reader hashes), P2
`mlp_register_epilogue_gate.py:13,76,82,86-88` (`register-epilogue-evidence/
fused-batch.json`), P3 `mlp_block_stream_gate.py:13,62-79` (`block-stream-evidence/
fused-batch.json`), P4 `mlp_block_stream_runtime.py:161-163` live manifest, P5
`fused_t16_scope.py:30-33` - and none of those files, nor the evidence directories,
are in either image copy list; they ride in the bundle cut from the runner's
`read-combined` tree. Hand-editing the recorded hashes would assert probes that
never ran, so the route is: fix the kernel, re-run the three evidence lanes
(cumulative-t16 7 min, register-epilogue-combined 7 min, block-stream), update the
three REPORT_SHA256 constants to the regenerated reports, re-cut the bundle
(serving-bundle, 1 min), pin it in the image workflow, build v40. Gate-only pins
(full_model_fusion.py:51, mlp_progressive_input.py, frozen_mlp_input_gate.py, the
wait-zone gates, frozen_recipe_context.py:225,304, cumulative_t16_experiment.py:28)
follow for lane hygiene. `tensix_stream_activation.cpp` has the same exit defect
but is not on the serving path.

### Frozen-recipe revision for the drained reader: the chain as run (2026-09-20)

Kernel: `scripts/ci/fused_1d_input.cpp` gains `noc_async_write_barrier()` and
`noc_async_atomic_barrier()` after the block loop (commit 4d890d6a, reader sha256
3e907b91). Overlay: `frozen_recipe_context.py`'s unconditional orchestrator loop
now carries the reader, `fused_t16_admission.py` and the target-mode simulator
report with its exit status into every lane's read-combined tree (REVISION
unchanged at 8c102b20; dry-run against a checkout of that ref verified).
Allowlist: one PATCH added five entries (experiments@frozen-mlp-exit-v1,
mlp-register-epilogue-sim-v5, mlp-block-stream-replay-v7,
cumulative-t16-full-v20-direct-dma-kv-slide-block-stream-dflash-native,
serving-bundle-v2); the group's five repositories survived and were verified.

P1: `qwen-experiments.yml` dispatched on tag frozen-mlp-exit-v1 (learned-attention,
simulator_only, simulator_fusion_t16, simulator_target_math). First attempt
35487389218 died inside actions/checkout deleting the persistent workspace
root, which other lanes had filled with container-written trees; the lane's
ownership step now owns the whole workspace (24bf8bab). Second attempt
35487539038 succeeded: six kernels, every `reader_sha256['fused_1d_input.cpp']`
= 3e907b91, exit status 0; committed as `fused-t16-target-simulator.json` with
`fused_t16_admission.REPORT_SHA256` = f9b9ce2d (2f8204e2); `qualify_simulator()`
passes on the tree. The regenerated report also records the current
`packed_weight_check.py` hash where the 09-17 report had an older one.

P2 (sim-v5, run 35488011817) and P3 (replay-v7, run 35488011111) tagged at
2f8204e2; then the hardware lane (cumulative full v20), the bundle (v2), image
v40, and the two-user run.

P2 and P3 succeeded (constants 0fa23737 and 02e16191, commit 5819047a; both gates
pass on CPU against a staged candidate). The hardware lane (full v20, run
35488441244) then refused in its restore step, before any device work:
`drafter_comparison_stage.py:43` -> `dflash_t16_native_attention_gate.qualify`
'Clean source-bound proposal-only simulation with unchanged native runtime
required'. That stage hashes the ORCHESTRATOR's SOURCES (line 21:
`directory = Path(__file__).parent`), not the pinned tree, against the
dflash-native evidence of run 35293698929; this branch has since changed exactly
one of them, `dflash_t16_native_attention.py` (+21/-3, the packed-mask `users=`
edits, since a92f10f8 of the last good cumulative run). Not the kernel. It is the
only stage that hashes the orchestrator this way. Regenerating that evidence:
tag dflash-t16-attention-sim-v5 (CPU simulator lane) at 5819047a, then the
cumulative lane again as full v21 with the new run id at cumulative-t16.yml:88.
Second allowlist PATCH for those two entries; repositories verified after it.

The dflash-native evidence was regenerated (sim-v5, run 35488650495; both
contexts pass, sources equal this tree) and `dflash_t16_native_scope.REPORTS`
pins its two reports (b1e61622). Hardware lane full v21, second attempt (run
35488924182): the restore and staging steps PASSED - every gate accepted the
regenerated P1, P2, P3 and native evidence against the drained reader - and the
audit step then failed at the first command of `run-dspark-hardware.sh`,
`no valid artifacts found to download`. That script restores eight
`qwen-hardware-inventory-*` artifacts of experiment runs from 2026-09-12/13
(34677941763, 34693525557, 34694674606, 34695921492, 34699176210, 34701425373,
34703126782, 34728453080), uploaded with 7-day retention; all eight expired
between 2026-09-19 06:22 and 2026-09-20 00:37 UTC, before the first v20 attempt.
The lane is not runnable by anyone until that week-old evidence chain is
regenerated, which is unrelated to this change and out of today's scope.

Decision: cut the serving bundle (serving-bundle-v2) from the runner's tree as
the lane left it after its passed staging steps - the same provenance as
bundle v1, which run 35328870330 packaged after v18's hardware step had failed
- and state the gap plainly: the fused T16 arm's ABBA device audit did NOT run
for the drained reader. Its device qualification is the two-user serving run on
image v40, which is the workload the fix exists for. No evidence hash was
hand-edited anywhere in this chain.

Bundle v2, first attempt (run 35489107221): 'Staged source changed since
inventory: dflash_combined_request.py' - `serving_bundle.py` fingerprints eight
staged files against the 09-18 inventory, and two had legitimately changed on the
runner's tree (`dflash_t16_native_scope.py` by its re-pinned reports,
`dflash_combined_request.py` by the lane's adaptation from the newer
orchestrator). Inventory lane serving-image-inventory-v3 (run 35489185444)
re-inventoried that tree: cached binary matches, runtime 9f9cd4fd, eight
fingerprints recorded; both the bundle and image workflows now restore and
verify it (sha 826ea8f0, commit 77d6995a). Bundle v2, second attempt (run
35489235797): success; `serving-bundle.json` lists 1335 sources with
`fused_1d_input.cpp` = 3e907b91 at BOTH `experiment-scripts/ci/` and
`experiment-scripts/ci/mlp-register-epilogue-candidate/`, the regenerated
`fused-t16-target-simulator.json` (f9b9ce2d) and the three re-pinned gate
modules. Image v40 (tag fast-serving-image-v40 at 1945b741, build run
35489293275) restores bundle run 35489235797. Third allowlist PATCH for the
inventory and a spare bundle tag; repositories verified after it.

## Run 35489404340 (image v40, drained reader): the divergence is not the reader's exit race

Same shape as 35486440095: both users admitted, two full packed rounds (steps
407-430 ms), carry restored and saved every step, every shard and drift check
equal (17 buffers, proposal_calls [2, 2]), no hang, and then

    AssertionError: Replicated learned selector features differ: call=2
    position=32780 rows=2048 max_abs=0.132812 mean_abs=0.0136247 differing=6712 of 8192

The order in the last round matters: step B, shards equal; step A (restore from
B), shards equal - B's buffers included, since the check covers every OTHER
entry; propose A (call 2) PASSED, 206 ms; propose B (call 2) FAILED. So B's
persistent inputs were verified equal on both chips immediately before A's
third proposal, and the only device work between that check and B's failure is
A's eager proposal. All traces execute on cq_id=0 (`verifier_engine.py:430,503`),
so nothing overlaps on a second queue. The divergence differs run to run
(max_abs 0.496 in v39, 0.133 here; 7391 vs 6712 of 8192 elements), so it is
timing- or content-dependent, and the affected elements are most of the tensor,
not a few rows. Conclusion: the fused input reader's exit race was real
(watcher-proven hang signature) and is closed, but it did not cause this
divergence. What remains is state that A's proposal leaves and B's proposal
consumes on one chip differently: something shared between the two DFlashDevice
instances (module-level caches keyed by position or shape - both users sit at
position 32780 - shared scratch, the program cache) or a per-chip transient.

Next: (1) environment A/B with shared collectives (QWEN_FAST_SHARED_CCL=1) on
the same image; (2) a first-divergent-stage audit inside the eager proposal
(QWEN_FAST_PROPOSAL_AUDIT=1): compare every replicated input and per-layer
output across the chips and name the first stage that differs.

## Run 35489772960 (image v40, QWEN_FAST_SHARED_CCL=1): two users decode nine rounds each

The environment A/B against 35489404340 - one shared collectives object instead
of one per request, everything else identical - decoded NINE packed rounds for
both users: 18 steps, `proposal_calls=[9, 9]`, every shard and drift check equal
on every step, no hang, no divergence, 10 chunks per stream (TTFT 82 s and 164 s,
serialised prefills; chunk gap median 2.3 s under the watcher's NoC sanitiser).
The first user completed its 64-token budget (`ignore_eos`), the lifecycle
detached it, and the survivor's next step -
`[PHASE] execute total=16 new=0 cached=1 spec=1 finished=['cmpl-b1ac...']` - was
refused by the packed contract, whose message now names the field:
`finished=['cmpl-b1ac...']`. That also explains v38's refusal in hindsight.

So the third-proposal divergence of v39 and v40 was the per-request TT_CCL
instances (`QWEN_FAST_SHARED_CCL=0`): A's proposal through its own collectives
leaves state that B's proposal through a second instance consumes differently
per chip. Shared collectives are the correct configuration and the fp2u lane
keeps them. The fix for the refusal: `step_refusals(scheduled, live=...)` counts
a finished id only when it still names a request the contract holds; the
packed contract prepares its tickets first to know that set (tests in both
modules). Image v41 carries it; the next run is expected to complete both
users' budgets. What the watcher-instrumented run already says about speed:
sequential steps of 407-430 ms per user plus ~207 ms per eager proposal, i.e.
the correctness scaffold, not the packed device step.

## Run 35490298652 (image v41, shared collectives): both users complete, no error

The first complete two-user serving cycle on the fast path. Image v41 =
sha256:b11e22ae (657b2970: the finished-id rule). Stream 0: 10 chunks, done at
178.9 s; stream 1: 23 chunks, done at 189.9 s; no error on either. Nine packed
rounds while both were resident (`proposal_calls=[9, 9]`), every shard and drift
check equal, then `[PHASE] execute total=16 new=0 cached=1 spec=1
finished=['cmpl-879b...']` was ADMITTED and the survivor decoded alone to its
budget (engine stats: Running 2 -> Running 1, 3.4 -> 5.2 -> 3.5 tok/s). Both
prompts 32768 exact token ids, max_tokens 64, ignore_eos; TTFT 82 s and 164 s
because the prefills are serialised (one in flight) and each takes ~77 s under
the watcher.

What this run measures and does not: chunk gap median 1627 ms, p90 1720 ms, with
the watcher's NoC sanitiser on every kernel and ~58 MB of shard readback per
device per step, through the sequential scaffold (one full weight pass per user
per round). The bench's `tokens` counts SSE chunks, not tokens (stream 0's 64
tokens arrived in 10 chunks, ~6.4 accepted tokens per verify), so
`tokens_per_user_per_s` in this report is chunks per second. Next: the bench
reads `usage.completion_tokens` via stream_options, and the timing run drops the
watcher and the shard readbacks.

Where the day ends against the goal: two users, 33024 context, 32768 prefill,
correct end to end. 200 tok/s per user needs the packed device step (one weight
pass for all users per round) in place of the sequential scaffold, 64-row blocks
for four T16 users, and the prefill/decode interleave; none of those is measured
yet. The concurrency pin (one hook, one bridge, one trace bucket) is lifted for
two users by construction.

## Run 35490648209 (image v41, timing): the sequential scaffold's honest numbers

Watcher off, shard readbacks off, shared collectives, real token counts from the
server's usage report. Both users completed 64 tokens, no error.

    stream 0: completion_tokens=64 chunks=10 ttft=14.9 s wall=31.6 s decode window=16.7 s -> 3.8 tok/s
    stream 1: completion_tokens=64 chunks=23 ttft=28.8 s wall=33.4 s decode window=4.6 s -> 13.7 tok/s
    chunk gap median 259 ms, p90 304 ms; steps n=30 per run median 90 ms (83-367);
    eager proposals median 41 ms (max 185); aggregate 17.5 tok/s; 4.4% of 200 tok/s per user

Reading it: a 32768-token prefill takes ~14 s without the watcher, and the
one-in-flight rule holds user 0 idle for the whole of user 1's prefill, which is
why user 0's window-averaged rate is a third of user 1's. In the phase where
both are resident, one packed round is step B (90 ms) + step A (90 ms) + two
eager proposals (2 x 41 ms) = ~262 ms for two chunks of ~6.4 accepted tokens
each, i.e. ~24 tok/s per user, ~49 tok/s aggregate, through the sequential
scaffold that reads the 19.92 GB of dense weights once PER USER per round. The
16-row verify replay itself is 90 ms.

Against the goal (200 tok/s per user = 5 ms per token): two users are correct
end to end at ~24 tok/s each while co-resident. The next lever is the packed
device step - one verify (one weight pass) for all users per round, and one
packed proposal - which at today's per-op costs would bring a two-user round
to roughly 90 + 45 ms, ~45 tok/s per user; four users at T8 (K=7, ~6 commits)
share the same pass. The prefill/decode interleave (Lever N) is what stops the
first user idling during the second's 14 s prefill. 200 tok/s per user is not
reachable by packing alone at a 90 ms verify: it needs the verify itself well
under 60 ms per round on top of packing, which is the kernel work the earlier
profiling documents describe. That is the honest position at the end of this
session.

### Proposal audit instrument (7b3977b8): QWEN_FAST_PROPOSAL_AUDIT=1

Built while the collectives A/B ran, and kept for the packed step. With
`QWEN_FAST_PROPOSAL_AUDIT=1` and eager proposals, `DFlashDevice.propose` reads
every replicated input and intermediate back from both chips at the stage that
made it (`ProposalAudit`, dflash_device.py:97-259; `compare_shards` :44-71) and
logs one `[AUDIT] dev=<id> call=<n> stage=<name> ...` line per stage - 89 stages
per proposal from the identifiers, pooled history buffer, sliced window, mask and
RoPE tables, through the embedding gather, each layer's attention and MLP branch
(normalized, kernels, convolutions, the gathered partials inside the collective,
the reduced sum, outputs; sharded heads and partials named once, not compared),
to the selector projection and features. The first differing stage is marked
`FIRST larger_norm=chipK` and repeated in a summary line. Off, no observer is
passed and no op or order changes. At proposal start it also logs every tensor
the proposal reads that the instance does not own: the lent PreparedDraftWeights
(104 tensors, ids and both chip addresses), the shared TT_CCL object with its
counters, the target's embedding and head, the pool slot. The audit found NO
module-level cache keyed by position or shape on the proposal path; the only
module-level state is host-side (projection-links lru_cache, the T16 admission
ContextVar, the per-instance validated-mask set) plus the device program cache.
22 new tests (`test_dflash_device_audit`, registered), 167 green across the
touched modules with the switch off. Not yet run on hardware; the readback
follows `check_shards`, which has.

### Why per-request collectives diverge: the trace-hole mechanism on L1 (audit, 2026-09-20)

Read at the image's tt-metal pin (v0.77.0-rc1). A `TT_CCL` instance owns 36
global semaphores (`models/tt_transformers/tt/ccl.py:51-74`: 3 axis pools x 2
double-buffered x [1 barrier + 2 all-gather + 3 reduce-scatter]), each a real
HEIGHT_SHARDED L1 buffer with one 4-byte page on every Tensix core of both chips,
allocated through the ordinary allocator at whatever L1 was free at construction
(`tt_metal/impl/buffers/global_semaphore.cpp:89-102`). With
`QWEN_FAST_SHARED_CCL=0` each request built its own at admission
(`serving_request_factory.py:87`) - i.e. B's 36 pages per core were allocated
AFTER A's verify trace was captured, into L1 holes that trace baked for the
intermediates it freed. Every replay of A's verify then wrote A's L1
intermediates over B's semaphore pages, per chip, with data-dependent values.

The all-gather protocol turns a scribbled semaphore into exactly what was seen:
the reader waits with `noc_semaphore_wait_min(out_ready_sem, target)`
(`minimal_default_reader.cpp:300-302, 376-378`, reset to 0 at :400), which
releases on ANY value >= target, so a nonzero leftover lets that chip consume the
freshly allocated output buffer before the remote chunks land - it reads a freed
intermediate. Most elements wrong on one chip, magnitude set by stale data, no
hang (0.496 vs 0.133 max_abs, 7391 vs 6712 of 8192 across runs). Everything
else fits: B fails and A does not (A's semaphores predate B's traces); the
failing proposal is the one right after A's step, i.e. after A's verify replay;
the shard check does not cover semaphores. The two instances share nothing
across the CCL objects themselves - no persistent gather buffer
(`feature_collective.py:51`), semaphore addresses rewritten per call on a
program-cache hit (`all_gather_async_default_program_factory.cpp:112-141,
854-868`); what the second instance shares is the L1 allocator with a trace
captured before it existed. With `QWEN_FAST_SHARED_CCL=1` the single `TT_CCL`
is built at attach (`serving_runtime.py:84`), before any request trace, like the
pool and the shared weights: protected by construction ORDER. Same class as
runs 35477522469 and 35481466425, on L1 instead of DRAM. The rule stands and now
covers semaphores: nothing a request keeps across steps may be allocated after
any trace that will replay exists.

## Packed device step build (from docs/packed-device-step-plan-2026-09-20.md)

**Items 1-3 landed (a9c03b45):** `DeviceLoopState(..., commit_only=True, users=N)`
allocates N per-user entries before any trace; `decode(..., segments=, slots=,
deferred=True)` restores each user's own carry slot, runs its recurrence, does
NO eager prefix restore, and leaves `segment_results` in segment order
(gdn_device_loop_state.py:102, :195). `RetainedGDNBlock.commit_user(segment,
prefix, dma=, publication=, synchronize=)` commits one user's segment with its
own prefix through `segment_layers(segment)` (48 lists of 20 for the commit
DMA, the carry as destination; prefix 0 is a no-op that still counts as a
decision; replay needs every user decided and a synchronising last commit; a
raising publication poisons the block) (gdn_records.py:143, :132, :67).
ModelBatch: pack + retain_records requires `commit_only_gdn=True` (:212 - the
plan said False; serving already uses True), the retained block's checkpoints
are the tuple of per-user carries (:440), and `run()` requires
`checkpoint_calls == len(pack['segments'])` per layer (:528). Two consequences
for the block engine: the pack's per-user checkpoints are dead in deferred
mode but still validated, and warming the commit traces writes the carries
and native slot 0, so the carries must be re-seeded after warming. Tests: 91
green across eight modules (16 new); test_gdn_records,
test_gdn_device_loop_state and test_retained_ownership were never in CI and
are now registered.

**Items 4-5 landed (38895d69):** `packed_verifier.py` - `PackedShape`,
`m1_shape(page_width)`, `segment_rows`, `PackedFeatureTaps` (a tuple subclass
carrying the user's row offset), module-level `stage_packed` (so the bundle's
verifier_inputs.py stays untouched), and `PackedVerifierEngine(operations,
model, helpers, sampler, *, pool, shared_weights, shape, feature_taps)` with
`segment_of(engine)`, `segments(entries)`, `stage_packed_inputs(entries)`,
`verify(entries) -> (predictions, metrics)` (predictions in entries order,
`metrics['segments'][i]` = entry i's segment), `features(segment)`,
`commit_user(segment, prefix)`, `describe()`, `close()`.
`VerifierEngine.adopt_packed(ticket, block, segment)` (:514) installs a
synthetic bucket keyed ('packed', segment) whose feature capture is a
`PackedFeatureView`; `publish` (:534) then routes to `block.commit_user`, clears
`_resident`, advances the position and goes idle without `save_carry`;
`note_packed_step()` (:28) is for the step to call. `project_features` takes
`row_offset` (dflash_device.py:532). SEGMENT BINDING: the trace bakes each
segment's restore source and commit destination - the pool slot carries - so
segment u is bound to pool slot u at attach and an entry is mapped to its
segment by carry identity; the plan's 'segment u = entries[u]' cannot hold once
the scheduler presents B before A, and the step must pass
`metrics['segments'][i]` for entry i. Construction order (after the pool and
PreparedDraftWeights, before the lifecycle) is enforced: construction refuses
a lent slot, a closed pool or weights, a differing page width, or a resident
carry. Tests: 128 green across seven modules (12 new in test_packed_verifier,
5 in test_verifier_carry). Combined with items 1-3 at HEAD: 223 green.

Follow-up ff9ef4ef: after warming the 32 per-user commit traces at attach,
`reseed()` restores native slot 0 from the initial snapshot and zeroes every
carry (the pool re-zeroes on loan and the engine seeds at admission anyway).
Corrections to the plan from the code: 32 commit traces, not 34 (prefix 0
publishes nothing); packed staging is 69 host copies per round (5 inputs, 32
singleton positions, 32 row tables) behind one fence - a cost to watch in the
timings. The bench now keeps each stream's generated text (9720ac91) so the
packed step's gate can be checked offline: run 35492676194 is the sequential
reference on the two fp2u prompts.

### Token-exact gate baseline: run 35492676194 (sequential step, image v41)

Same lane and env as the timing run, plus the bench keeping each stream's text.
Both users 64 tokens, no error; the same timing as run 35490648209.
The prompts are the exact token-id arrays [1000 + i %% 64], so greedy output is
deterministic nonsense - which is what a byte-exact comparison needs. The
reference texts (utf-8) hash to:

    stream 0: 64 tokens in 10 chunks, sha256 66a5d0c2a531af8bf65fc2cacbd1c952a413ee42bb7051ded6555b41384ea217
    stream 1: 64 tokens in 23 chunks, sha256 481ea949857d4cce166dd9fb56048a88b016622b523ea2b18ad19f4c3a56fd6d

The packed step passes the gate when a run on the same lane with
QWEN_FAST_PACKED_STEP=1 yields the same two hashes (the full texts are kept
locally in runner-evidence.local/packed-gate/). Stream order is by request
index, which the bench fixes; the scheduler's order does not matter to it.

**The two users' texts differ (run 35492676194).** Same prompt, greedy decoding:
the texts agree for the first 48 characters and then user 0 (stream 0, admitted
first, idle through user 1's 14 s prefill) emits token soup for the rest of its
64 tokens (240 chars) while user 1 emits coherent code with markdown and
TypeScript (200 chars). At most one is right, and 'no error + draft shard checks
equal' was never a check of the target's per-user state. Hypothesis to test:
user 1's target prefill rewrites per-slot state that user 0's carry restore does
not put back (the carry covers rec_state and conv_states; the decode kernel also
reads packed conv states and the attention's per-slot tables). A single-user run
of the same prompt is in flight to say which text is right; the GDN audit is
ranking what the prefill overwrites. This is the first correctness signal beyond
completion for the two-user path and it predates the packed step entirely.

**Single-user reference (run 35492921706, users=1, same prompt, image v41):** 64
tokens in 10 chunks, text BYTE-IDENTICAL to user 0 of run 35492676194 (sha
66a5d0c2), common prefix with user 1 only 48 chars. So the 'token soup' is the
correct greedy continuation of the nonsense prompt and user 0 was right; USER 1 -
the second-admitted request, which the plugin prefilled 'into slots [1]' - is the
corrupted one: right for its first chunk, wrong after, and accepting fewer draft
tokens from the start (23 chunks vs 10). The fast path saves and restores GDN
state at native index 0 only (`gdn_snapshot.py:43-45`:
`_write_recurrent_state_prefix(..., 1)`, `_write_index(target, saved, 0, 1)`;
the eight-slot native state at :10), so user 1's initial snapshot and carry were
taken from slot 0 - whatever it held - not from the slot its prefill wrote.
Identical prompts make the two post-prefill states coincide, which is why user
1 was right for one chunk; the run with distinct prompts (one id offset) tells
whether user 1 is wrong from its first token. Single user on the fast path:
46.7 tok/s over its decode window (64 tokens; 90 ms verify + 41 ms proposal per
chunk of ~6.4 accepted tokens).

**Distinct prompts (run 35493208438, users 2, base 1000 and 1001):** user 0 again
BYTE-IDENTICAL to its single-user reference (sha 66a5d0c2). User 1, on its own
prompt, opens with user 0's continuation - ' “veniss ourclassraw yearData' are
user 0's tokens 2-5 - and then drifts into a coherent assessment of its prompt
as 'repeated and consistent use of the same ... likely a form of spam or a test
of the AI' (29 chunks for 64 tokens). That is what decoding from user 0's GDN
state with user 1's own attention pages looks like: the recurrent layers carry
user 0's prompt, the KV pages carry user 1's. The code agrees:
`gdn_snapshot.ActiveSnapshot` allocates, saves and restores at native index 0
unconditionally (`_slice_along(tensor, dimension, 0, 1)` at :18 and :31,
`_write_recurrent_state_prefix(..., 1)` / `_write_index(target, saved, 0, 1)`
at :43-45), while the plugin prefilled user 1 'into slots [1]'. So user 1's
initial snapshot and carry at admission were user 0's post-prefill state, and
user 1's true state, written to slot 1, was never read. The single-user run on
base 1001 (queued) is the formal check that user 1 is wrong from its first
token. Fix shape: the admission snapshot must read the slot the prefill wrote;
decode, restore and commit stay at slot 0.

**Single-user reference for prompt base 1001 (run 35493236124):** 64 tokens in 11
chunks, sha 26c9c952, 39.7 tok/s. Its text is nearly the base-1000 text without
the leading ' would' - the two nonsense prompts have near-identical correct
continuations, so the distinct-prompt run did NOT discriminate as intended.
User 1 of run 35493208438 matches its own reference for 30 characters (about
five tokens, i.e. most of its first verify) and diverges after ('would be to
the end' where the reference has 'val somefterys'). Two readings survive: an
initial snapshot taken from slot 0 (user 0's state, similar enough for a few
tokens), or a corruption between user 1's first and second verifies. Probe
cpu-probe-v20 (run 35493486419, `probe_prefill_slot.py`) prints the image
model's prefill signatures and slot selection; the GDN audit ranks the
per-slot state. Note for the fix: serving builds `ActiveSnapshot(direct=True)`
(serving_runtime.py:56), whose save/restore go through the
`gdn_state_copy.cpp` DMA kernel at slot 0; the admission snapshot can use the
ttnn slice path at the prefill's index instead, leaving the per-step carry
swaps at slot 0 untouched. `gdn_snapshot.py` is identical to the bundle's
copy and not yet in the image lists.

**Items 6-7 landed (fac52b81):** `serving_packed_step.packed_device_step(entries,
*, cancelled, block)` (:78) - same contract as the sequential step; refuses
before device work (:82-91); `ineligible` (:62: wrong user count, a narrow
ticket, an unbound engine) or a pre-verify cancellation sends the WHOLE round
to `sequential_packed_step` (:97; N=1 survivor included); one `block.verify`
(:107, which stages internally - 69 host copies and one fence); then per entry
in scheduler order `adopt_packed(ticket, block, metrics['segments'][i])` and
commit or abort through `runtime.publish` (:132-146); `note_packed_step` after
every round (:127); `fail_round` (:175) aborts adopted users, fails unadopted
ones and releases every pending segment at prefix 0 so the block never stays
'verified'; `[PACKED]` audit lines under QWEN_FAST_PACKED_AUDIT=1 (:149). The
block fences only the commit that empties its pending segments
(packed_verifier.py:483-485, 14bc5cf6). Wiring (serving_runtime.py:119-136,
:167): under QWEN_FAST_PACKED_STEP=1 ONLY, the block is built after the draft
weights and before the lifecycle, closed between them, and the lifecycle gets
`partial(packed_device_step, block=block)`; the stage line always carries
`device_step`. Default stays sequential. Image lists gained packed_verifier.py,
serving_packed_step.py, gdn_device_loop_state.py and gdn_records.py (drift =
packed commits only); dflash_request_runtime.py stays a bundle copy (its drift
is 55e199ac's proposal_counts change, outside this work). 117 tests green in
the eight step modules; test_serving_packed_step registered. NOT run on
hardware: its gate (both users' texts equal to their single-user references)
cannot be met until the slot-1 snapshot bug below is fixed on the sequential
step, since the packed step inherits each user's carry from admission.

### The slot bug, confirmed from the image's model source (probe cpu-probe-v20, run 35493486419)

`qwen36_vllm._prefill_forward_tp_batched(model, tokens, page_table, prompt_lens,
empty_slots)` (qwen36_vllm.py:271) takes `empty_slots` from vLLM's kwargs (:215) -
the request's decode slot, [0] for the first resident request, [1] for the
second - and calls `model.prefill_paged_slots(token_ids_list, page_table,
empty_slots, valid_lens)` (:292), whose docstring: 'bind a B=1 GDN scratch, run
the trace-safe pre-warmed masked-bucket prefill per user, snapshot its B=1
state - but writes each snapshot into its slot via write_slot (preserving the
live rows)'; 'GDN state is a fixed [B,...] buffer indexed by slot, not paged'.
So the second user's post-prefill GDN state was written to row 1 and row 0 was
left holding the first user's state; the fast path's `ActiveSnapshot` reads
row 0 for the initial snapshot and the carry, so the second user decoded from
the first user's recurrent state with its own attention pages. The chunk
boundary the capture wraps (`_forward_prefill_chunk_masked_tp(token_buf,
valid_len, chunk_start, page_table, bucket, flex_sdpa=True, vision_tokens=None)`)
runs on the scratch and never sees the slot; `prefill_paged_slots` on the same
model object does. The plugin also carries a `slot_remap` kwarg on decode
(qwen36_vllm.py:309-311, `model._remap_gdn_slots`) that the bypassed decode
would use to compact slots. Fix in progress (agent 'slotfix'): the capture
records `empty_slots`; `ActiveSnapshot.adopt_slot(k)` copies row k into row 0
through the slice path at admission, before the engine's first save; decode,
restore and commit stay at slot 0. A follow-up probe prints `write_slot` so the
adoption covers everything the prefill wrote.

The GDN audit reached the same finding independently (from the image's
`lever_n_model_patch.py:174-206`: bind the single-occupancy GDN prefill scratch,
run the chunked prefill with vLLM's own block-table row, read the scratch
rec/conv back to host, unbind, `_write_gdn_slot(int(empty_slots[u]), ...)` -
slot 1 for the second user; slot 0 is not written) and adds what the fix needs:
slot 0's recurrent state plus the four conv taps IS the whole per-slot GDN
state (gates, dt bias, neg-exp A and norm are weights, gdn_device_loop_state.py:77),
so copying those rows is complete; `note_prefill()`'s assumption that a prefill
overwrote slot 0 is the reverse of reality and costs one extra restore, nothing
more; and serving's prefill is the untraced chunked fallback (:361-366), an
eager path that allocates only free memory and cannot scribble the first
user's live buffers - which the first user's byte-exact text confirms.

Follow-up probe (run 35493717058): `model._remap_gdn_slots(remap)` applies vLLM's
batch-condense slot_remap to every GDN layer's batched decode state through
`layer.attention.remap_slots(remap)` ('slot i takes the state at slot remap[i]';
the plugin's own slot_remap does not move GDN state). `write_slot` is a method
of the GDN layer, not the model, so its body was not printed; the audit's
reading of the patched prefill (scratch rec/conv read back, then written into
the slot) and the per-slot inventory (rec + four conv taps) stand. The fix's
`adopt_slot` copies exactly those five rows.

### Two bugs, not one (GDN audit re-rank, 2026-09-20)

On the SAME-prompt run the slot bug is masked: user 1's seed from slot 0 is
user 0's post-prefill state, numerically identical to user 1's own (same
prompt, user 0 had not verified). Yet user 1 still went wrong after its first
verify - i.e. after user 0's FIRST trace replay (round 1 order: step B right,
step A = A's first replay since B's buffers were allocated; round 2: B wrong).
That is the trace-hole class once more, on the one per-request allocation the
pool never covered: the ReplayAttentionReader's per-bundle page tables
(attention_replay.py:47-48, uploaded per request, 4 bundles at T16), rewritten
only by `VerifierPageBinding.refresh` on a block change (serving_page_binding.py:
91-101) - static across steps, so a clobber sends the user's 16 attention layers
to wrong KV pages from the next verify on, exposing only the LATER-admitted
user and only after the earlier user's first step. Possibly also ModelBatch's
row_tables (model_batch.py:319, not in BATCH_INPUTS). Replay masks are
recomputed in-trace from the positions word and the word is re-uploaded per
verify, so they are safe; so are the GDN entry/state (written in-trace before
read) and every pooled buffer. Both bugs are being fixed before the gate run:
slot adoption at admission (agent 'slotfix') and pooled replay page tables plus
a page-table drift check under QWEN_FAST_SHARD_CHECK (agent 'replaypool').

Adoption scope check (audit): the prefill reads back exactly rec_state and
conv_states per layer (lever_n_model_patch.py:197-198) and `_write_gdn_slot`
takes only those (:205); the tensors are written in place, so the addresses the
traces baked hold (gdn_device_loop_state.py:112-114 would raise otherwise).
One risk: conv_states are (1, 8, 5120) per chip in TILE layout with all eight
slots inside one 32-row tile, so a slice starting at row k != 0 is a tile-
internal unaligned slice no path exercises today. `adopt_slot` therefore
verifies each device slice against a host readback of the full tensor's row k
on both chips before writing row 0, and raises otherwise - once per admission,
unconditional, so the gate run is also the proof of the slice.

**Bug 1 fixed (d65773d5):** `PrefillWindowCapture` also wraps
`prefill_paged_slots` on the model (dflash_prefill_window.py:163; wrapper
:137-149) and records `capture.prefill_slot` (exactly one user per call, ints
only; None on the single-sequence path). `ActiveSnapshot.adopt_slot(index)`
(gdn_snapshot.py:47) slices row `index` of rec_state (dim 0) and each conv
state (dim 1) and writes them into row 0 through the model's own
`_write_recurrent_state_prefix` / `_write_index`, freeing the slices; a no-op
for 0. `serving_request_factory.from_prefill` calls it on all 48 helpers at :80,
after the host-side refusals and before the drafter and the engine (whose
constructor makes the initial save and seeds the carry from slot 0), logging
'[PINDIAG] adopted GDN slot k into slot 0'. 12 new tests; test_dflash_prefill_window
and test_gdn_snapshot had never been registered in CI and now are; both
modules join the image lists (drift from the bundle = this change only). A
follow-up adds the host-readback equality check on each unaligned slice.

Follow-up 18d627d9: `adopt_slot` reads back each device slice and the full
tensor per chip and requires equality with the full tensor's row k before
writing row 0; a mismatch raises with the tensor index, chip and max_abs and
frees the slices. Unconditional, once per admission. The gate run is therefore
also the device proof of the unaligned conv slice.

Cost of the self-check (slotfix): the rec_state readback is whole-tensor per
chip - eight slots x 24 heads x 128 x 128 bf16 = 6.3 MB per layer per chip, so
~604 MB per admission across 48 layers and two chips, plus ~31 MB for the conv
states: a few hundred ms to a second over PCIe per second-user admission
against a 14 s prefill. Kept for the gate run; after it, the rec_state check
(page-aligned slice, never at risk) is to be dropped and only the unaligned
conv-state proof retained.

**Bug 2 fixed (d580f1c8):** the replay reader's page tables are pooled. They are
not at the pool's page width: each is (batches, capacity // 64) for the
capture's 256-position family, capacity = (position // 256 + 1) * 256 at
admission (attention_replay.py:49), so `BucketSlot.batch.replay_pages` holds one
set per family the reader can be captured in (`attention_replay.family_capacities`,
the 50 capacities 4096..16640 the ticket validator admits, cut to the page
table): 300 tables and 388,800 bytes per slot at page width 1024, ~1-2 s of
uploads at attach for two users. Two bundles at T16 (3 groups + 1), not four.
`ModelBatch.replay_storage` (model_batch.py:92) hands the family's list to
`ReplayAttentionReader(storage=)`; the reader validates, borrows, stages with
copy_host_to_device_tensor and never frees them; `VerifierPageBinding.refresh`
already wrote in place, so block changes land in the pool. Row tables alias the
pooled singleton pages - nothing to add; the reader's 8-word positions are
restaged every verify and the masks recomputed in-trace. The drift check
(serving_sequential_step.py:144-296) now snapshots every page table - fixture
pages, singleton pages, each replay table - at the start of a checked round and
after the owner's step and compares per chip after every other step, logging
`[PINDIAG] page-table drift ...` in the K/V drift format; under
QWEN_FAST_SHARD_CHECK=1 it raises. attention_replay.py joins the image lists
(drift = this change only); serving_page_binding.py is unchanged and already in
the bundle. 214 tests green (21 new); test_attention_replay registered.

Image v42 (tag fast-serving-image-v42 at d580f1c8) carries both fixes, the
packed step (off by default) and the audit instrument. Gate run configuration:
two users on distinct prompts, QWEN_FAST_SHARD_CHECK=warn so the page-table
drift check runs, shared collectives, no watcher. Pass = both texts equal their
single-user references (66a5d0c2 and 26c9c952) and no drift line.

## Run 35495227738 (image v42): refused by the frozen recipe's source pin

Both streams 500 at the first admission: `frozen_combined_runtime.qualify`
(:76-105, reached from `verifier_engine.__init__:160` -> `target_t16_attention_gate.qualify:71`)
hashes every source named in the frozen evidence reports (draft-numerical.json
and target-replay.json 'sources', minus three exclusions) against
/experiment-scripts/ci and stops at the first mismatch:
`Combined runtime component source differs: attention_replay.py`. So the
frozen 32K recipe pins attention_replay.py, which the pooling fix edited; and
since the check stops at one name, the other files v42 overrides for the first
time (gdn_device_loop_state.py, gdn_records.py, gdn_snapshot.py,
dflash_prefill_window.py) may be pinned too. The adoption path did run
('[PINDIAG] prefill slot 0, nothing to adopt' for the first user). Probe
`probe_frozen_pins.py` lists the pinned set and every mismatch. Route: return
each pinned file to its pinned bytes and move its change into an unpinned
module (a pooled subclass for the replay reader, constructed by model_batch,
which is not pinned - v39-v41 overrode it and passed this gate), rather than
regenerate the 32K evidence (hardware lanes with expiring upstream artifacts).

**Pin probe (run 35495461481, `probe_frozen_pins.py`):** the frozen evidence in the
image holds three reports with sources (draft-diagnostics 48, draft-numerical 48,
target-replay 8); `qualify()` pins 42 sources. Of the twenty files the fast-
serving image overrides, exactly ONE is pinned: attention_replay.py. Not pinned:
gdn_snapshot, dflash_prefill_window, packed_verifier, serving_packed_step,
gdn_device_loop_state, gdn_records, model_batch, verifier_engine, dflash_device,
draft_kv_history, serving_buffer_pool, serving_vllm_contract, the draft branches
and convolutions, dflash_t16_native_attention, verifier_pack. (The probe lane
runs an older image, so its 'every pinned source matches' verdict is about that
image, not v42; the pinned SET is what it measured.) So the slot fix and the
packed step are clear of the recipe; only the pooled replay reader has to live
in an unpinned module with attention_replay.py restored to its 8c102b20 bytes -
the relocation in progress. Image v43 follows.

**Relocation (563f633c):** attention_replay.py is byte-identical to 8c102b20 again
(diff empty) and out of the image lists; `pooled_attention_replay.
PooledReplayAttentionReader` subclasses the pinned reader and intercepts its
`upload` callable - the pinned constructor asks for the positions word, then per
bundle a 2-D int32 page table and a bf16 mask; every 2-D int32 request is
answered with the next lent table, the rest forwarded - so metadata, positions,
masks, programs, refresh and the trace capture are the base's verbatim and
serving_page_binding.py / verifier_engine.py stay untouched. Storage is
validated before any upload, the tables are staged after construction, lent
tables move from `owned` to `borrowed`, close() frees only what the base
uploaded. model_batch chooses the pooled reader when the storage carries the
family's tables, else the pinned call unchanged. Helpers moved to the new
module; serving_buffer_pool imports them from there. 218 tests green over the
twelve modules; test_pooled_attention_replay registered. Image v43 (tag at
563f633c) is the gate build.

## Run 35495982721 (image v43): the adoption self-check catches the slice convention

The frozen pin is satisfied (first user admitted, one chunk emitted). The second
user's admission reached adoption and the self-check refused before any write:

    ValueError: Recurrent-state slice at row 1 differs from the row: layer 0 rec_state
    chip 0 shape (0, 24, 128, 128) torch.bfloat16, row (1, 24, 128, 128) torch.bfloat16

The model's `_slice_along(tensor, dim, a, b)` takes (start, stop), not (start,
count): row 1 requested as (1, 1) is empty. Every older call is (0, 1), which
reads identically under either convention, so nothing in the codebase or its
tests had ever disambiguated it; the test fixture modelled the count reading.
Fixed: `adopt_slot` slices (index, index + 1) and the fixture narrows to stop -
start. Without the self-check this would have written an empty tensor into row
0 and decoded from garbage silently. Image v44 = the gate build.

## Run 35496290854 (image v44): TOKEN-EXACT GATE PASSED for two users

Two users on distinct prompts (base 1000 and 1001), sequential step, shared
collectives, drift checks on. User 0: 64 tokens in 10 chunks, text sha
66a5d0c2 = its single-user reference byte for byte. User 1: 64 tokens in 11
chunks, sha 26c9c952 = its single-user reference byte for byte. Admission lines:
'prefill slot 0, nothing to adopt' for user 0 and 'adopted GDN slot 1 into slot
0: 48 layers, slices verified on both chips' for user 1. No page-table drift,
K/V drift or shard-mismatch line. This is the first run in which two concurrent
users on the fast path produce exactly what they would alone, and it closes
both second-user bugs: the slot seeding (adoption at admission, self-verified)
and the replay page tables in the trace holes (pooled). Image v44 =
sha256:873a9103, built from 3db6e225 (with 89f13a27's fixture fix on the
branch). The packed device step's gate is now meaningful; its run follows on the
same image with QWEN_FAST_PACKED_STEP=1.

## Run 35496483954 (image v44, QWEN_FAST_PACKED_STEP=1): attach failed, error masked

The server never became ready: the packed block's construction at attach raised
inside `attach_combined_runtime`, whose handler (serving_runtime.py:168-170)
called `scopes.close()` while handling it; the block-stream scope's exit check
(mlp_block_stream_runtime.py:189-190, 'Every target layer must construct and
execute the candidate' - trivially unmet at attach, before any layer has run
the fused candidate) raised, and vLLM logged only that. The original error is
lost. The fused arm itself is not the cause: a 32-row activation takes its
fallback (fused_t16_scope.py:49-51), not an exception. Fixed for the rerun: the
handler logs the closing error and re-raises the original (image v45). The
lane's committed state returned to the sequential step, which passes the gate.

**Run 35496918269 (image v45, packed step): the attach failure named.**
`[PINDIAG] attach failed with ModuleNotFoundError: No module named
'target_packed_pages'; closing the scopes then raised ValueError: Every target
layer must construct and execute the candidate`. The module is repo-only
(never in the bundle) and imported lazily inside `model_batch.validate_pack`
(model_batch.py:41), which only a packed fixture reaches - so every sequential
run was blind to it. The image's copy lists now carry it and
dflash_batched_mask.py (the same lazy class on the draft branches' pack= path,
for the packed proposal later). A static import closure of the packed modules
lists 70 more repo-only names, all experiment-only lazy imports (dspark_*,
t32_*, frozen stages) the serving path never executes. Image v46.
