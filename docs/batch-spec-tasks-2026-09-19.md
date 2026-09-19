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
