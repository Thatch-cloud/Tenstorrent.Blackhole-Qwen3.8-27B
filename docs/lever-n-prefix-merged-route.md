# Lever N with prefix reuse: one prefill route

**Date:** 2026-10-07. **Branch:** `tp4/levern-prefix`. Its base is merge commit `0cab164f`, which is `origin/ship/262k-prefix` (b8041cf4, the production recipe: W1 plus packed prefix reuse, eight seats x 262k, profile `c2-packed-tp4-8x262k-ship-prefix`) merged with `origin/tp4/lever-n` (0be1dfda).

**Status: implemented on CPU (sections 2-11), proved on real vLLM 0.25.1 by `test_levern_prefix_scheduler_vllm`; nothing has run on a card.** Section 15 records where the build departs from the text below and what it left open.
- The merge is committed and was the builder's starting point. Both levers refuse each other at boot only where they are not both on: the two new gate-only profiles (`c2-packed-tp4-8x262k-ship-prefix-levern` and its `-audit` twin) turn both on; every other profile behaves as before.
- Nothing ran on a card. The card gates (12.3) are listed, not run.

**Labels:**
- **m**: measured (the source is named).
- **cm**: computed from measured numbers.
- **e**: estimate. Section 11 uses the timing model in `docs/lever-n-tp4-timing-model.py`.

**Read with:**
- `docs/lever-n-tp4-design-2026-10-04.md` (the Lever N design; sections 3.5, 3.10 and 3.11 already sketch the merge);
- `docs/lever-n-tp4-stage1-2026-10-04.md` (what stage 1 built);
- `docs/tp4-packed-prefix.md` (sticky sessions at TP4: "one route with three call kinds (cold-first, hit-first, continue) is the merge contract").

---

## 0. Decisions

1. **One meaning for `start_pos > 0`.** It means: before this step's first chunk, the B=1 prefill scratch must hold the GDN state after exactly `start_pos` tokens of this request's prompt. Every step names where that state comes from:
   - **COLD**: start 0; zeros.
   - **CHECKPOINT**: a hit's first step; the registry's committed grant at Q.
   - **SCRATCH**: the request's own suspended scratch; the owner token.
   - **PARKED**: the request's own state saved to host while another prefill ran (v2).

   A prefix hit is therefore a resume whose state comes from the registry. The lifecycle names the source in the step announcement. The route resolves the source again from its own evidence and refuses before any device work if the two disagree. The scheduler checks the same table before it allocates (section 2).
2. **The merged route is `levern_route`, extended.** Under the merged profile, every served prefill step goes through it, cold steps, whole prompts and hits included. G1's `_qwen_prefix_prefill_slots` stays the route for every profile without Lever N, so production is untouched. The merged route only calls G1's staged model methods (restore, scratch read, capture, `prefill_traced_chunked(start, capture_at, on_capture)`) and its row check. No sha256 pin moves:
   - not the model graft's `PATCHED_SHA256`;
   - not the plugin's `scheduler.py`;
   - not a kernel.
3. **Exactly five refusal sites change** (section 4):
   - the Lever N contract refusal becomes the merged requirements;
   - the G1 graft accepts chunked prefill only beside an installed Lever N cap, under sticky sessions;
   - the sticky lifecycle's "whole rest of the prompt" clause becomes "a step of the plan from R";
   - the prefix contract's "a prefill must never split" keeps its check and changes its meaning;
   - the route's two `start > 0` refusals become the source resolution.
4. **Split prefills publish the same checkpoints as whole ones.**
   - Under sticky sessions the plan's prompt boundary is `floor2048(P) - 2048`. That is exactly `S_last`, the end of the last non-final Lever N step.
   - The route passes `capture_at = plan ∩ (start, end]` to each step.
   - The registry keeps a split request's capture plan across its steps. Today a grant lives for one step, so a capture in a later step is refused. That is why a split 254k prompt publishes nothing.
   - Claim, tested on CPU and on cards: with Lever N on, the registry ends every turn with the same keys, positions, token ids and state bytes as with it off.
5. **The G1 scheduler graft changes, and the change needs an image rebuild.** The graft and its registry are image-baked:
   - the build's AST stage adds a hook to `TTScheduler.__init__` that imports them from the plugin package;
   - `docker/qwen-c2-overlay.txt` lays both files at the plugin path and in `/experiment-scripts/ci`, and provenance hashes both copies.

   A new serving image is built. Only overlaid module bytes change; no pin moves. The graft also gains a stats-free **peek** of a waiting request's trimmed hit, because the Lever N cap must be computed from `(P, Q)`, not from `(P, 0)`. Today a hit at `Q = S_last` would split the drafter window across two steps (section 6, defect D1).
6. **Page-table width stays the runner's: 4,096 blocks at 262k.**
   - The merged route never calls G1's `_qwen_prefix_fit_page_table`.
   - The attach refuses a chunk-input buffer of any width under the merged profile.
   - The warm runs a restore and a capture at that width before the traces.
7. **The fixture epoch gets a third writer class.** Today a step that wrote no decode slot bumps the global epoch twice, killing both W1 blocks' pre-stage snapshots. Under the new class:
   - an intermediate Lever N step that measured `wrote_slot=0` is a *disjoint writer* and bumps no epoch;
   - this holds only while an attach-time address check shows the route's persistent writes are disjoint from every block's staging destinations, and while a per-step identity check shows the scratch was unbound;
   - the final step and the admission still bump globally, once per prompt instead of about 123 times.

   The audited prestage arm is the hardware gate (section 8).
8. **Admission v2 is a short lane with host parking.** A request whose remaining tokens after its peeked hit are at most `QWEN_FAST_LEVERN_SHORT_TOKENS` (16,384) preempts a long prefill at its next step boundary:
   - the long's scratch is parked to host through G1's own capture path and restored later through G1's own restore path, the path every production hit already runs;
   - one park slot; shortest remaining first among shorts; aging bounds a long's total park time.

   A hit turn arriving mid-cold waits at most one step, about 1.05 s at 250k of context (e), instead of up to about 190 s.
9. **Request-shaped refusals end one request; invariant refusals stay fatal and become unreachable.**
   - The more-than-one-partial refusal and the scratch-owner refusal become pre-allocation, per-request quarantines, through the D2 consumer the image already runs.
   - The "pass admitted another request" refusal and the alignment refusal become unreachable by construction:
     - the capped pass sees only its target;
     - the cap comes from the peeked Q.

     Both stay fatal if they are ever reached.
   - A Lever N kill switch file stops splitting new prompts without a restart.
10. **TTFT at 262k: base share f = 0.5 plus a deadline governor.**
    - The client deadline for a large prompt is 240 s, and the target is T* = 180 s.
    - Whenever any pending prefill would miss T*, the governor raises the in-flight prefill's share, up to 1.0 (no yields, which is today's stall, chunked).
    - A lone cold 253,920-token arrival: 192 s at f = 0.5 becomes about 180 s at f ≈ 0.54 (e).
    - Two simultaneous cold 254k arrivals: the second gets about 197 s with the governor, against about 187 s today and about 291 s at a fixed f = 0.5 (e).
    - f_base is still chosen by the agent-turn sweep. T* is not swept: the client deadline fixes it.
11. **Exactness is proved CPU-first:**
    - unit tests on the real staged `model.py` with the recording fake ttnn and the toy hybrid model: hit equals cold twin byte for byte, the boundary matrix, the capture and restore round trip, park and unpark;
    - two real-vLLM 0.25.1 proofs, one per phase, on experiment tags v19 and v20 (both unused today);
    - the CPU suite once per phase on the dev queue.

    The card gates are listed in section 12.3 and are not run here.

---

## 1. What refuses what (verified on the branches)

| # | Site (origin branch:line; line on this branch) | What it refuses | Effect today if both levers are on |
| --- | --- | --- | --- |
| R1 | `serving_c2_contract.py` `levern_problems` (tp4/lever-n :461-463; here :461-463) | `QWEN_PREFIX_REUSE` or `QWEN_FAST_STICKY_SESSIONS` beside `QWEN_FAST_LEVER_N=1` | boot refused |
| R2 | `qwen_prefix_scheduler_patch.py` `install_problems` :336-338, plus :344-347 (threshold must be 0); image-baked | `enable_chunked_prefill` on the scheduler config. `levern_platform` keeps it on. | the engine dies in `TTScheduler.__init__` |
| R3 | `serving_lifecycle.py` `_granted_resume` (tp4/lever-n :301-302; here :301-302) | a hit whose step does not carry the whole rest of the prompt | a split hit kills the engine at admission |
| R4 | `serving_c2_contract.py` `prefix_reuse_problems` (tp4/packed-prefix :368-371; here :380-383) | `max-num-batched-tokens < max-model-len`: "a prefill must never split" | passes (262,144 = 262,144), but its premise is what Lever N breaks |
| R5 | `levern_route.py` `prefill_forward` :313-316 (an unannounced `start > 0`) and `route` :414-418 (the owner token) | any `start > 0` the route did not split itself | every hit reaches `run_chunk`, even one that fits a single step (`whole()` needs start 0), and the owner refusal kills the engine |

Other facts the design rests on, each verified on this branch:

- **The route captures nothing.** `levern_route.py:431` calls `prefill_traced_chunked(tokens[:, :end], page[0:1], actual_len=end, start=start)` with no `capture_at` or `on_capture`.
- **A grant lives for one step.**
  - `qwen_prefix_registry.begin_step` (:382) clears `committed` at every `schedule()`.
  - `capture` (:436) refuses a position with no committed grant.
  - A Lever N continuation is a running request, never re-admitted, so no grant is ever staged or committed for it.
- **The sticky plan's prompt boundary equals S_last.** The plan's boundary is `floor2048(P) - 2048` (`qwen_prefix_scheduler_patch.plan` :509, sticky with the DFlash drop-last). `levern_policy.final_start` (:190) gives `S_last = F - 2048` when there is a tail and `P - 2048` when there is none, with F = floor2048(P). The two are always equal.
- **Two sites bump the epoch for every Lever N step.**
  - The lifecycle: `note_fixture_writer('prefill')` at :579 for a first step and `('prefill-chunk')` at :647 for a continuation.
  - The worker hook's pass-through: :336 and :356.
  - Both bump the global epoch, which in W1's `blocks` mode kills both blocks' snapshots (`verify_prestage.bump` :229 against `bump_fixture` :254).
- **Lever N's cap is computed before the pass.** `levern_scheduler.LevernRuntime.budget` (:72) reads the waiting request's `num_computed_tokens`, which is 0 before vLLM's trim runs inside the pass.

---

## 2. The contract: one meaning for `start_pos`

### 2.1 The rule

A prefill step `(request X, start S, end E, prompt P)` is legal only if both hold:

- **(a)** the scratch holds the GDN state after exactly S tokens of X's prompt before the first chunk runs;
- **(b)** the KV of positions [0, S) of X is in X's pages, written, either by X's own earlier steps or by vLLM's cached blocks, which the trim and the cap guarantee.

The step then runs the model's own chunk loop from `S/2048` (`prefill_traced_chunked(start=S)`), which resets the GDN state only at S = 0.

| Source | When | Evidence the route checks | Device work before the loop |
| --- | --- | --- | --- |
| `COLD` | S = 0 | no committed grant with Q > 0 for X | the loop's own reset |
| `CHECKPOINT` | X's first step, S = Q > 0 | `registry.grant_for(X)`, checked by the staged `_qwen_prefix_row`: `grant.q == S`, the checkpoint at S, the token ids equal X's [0, S), and the GDN spec equal to the scratch's | `_qwen_prefix_restore(rec, carry)` (G1's warmed path) |
| `SCRATCH` | a continuation | `owner == (X, S)` | none |
| `PARKED` (v2) | a continuation after a park | `park[X] == S` and the scratch's owner is none or already parked | `_qwen_prefix_restore(park rec, carry)`, then the park slot is freed |

Anything else is refused before any device work.

**Two witnesses, both required to agree:**
- the lifecycle announces the source in `Step(req_id, start, end, total, source)`;
- the route resolves it from the evidence above.

**A third check, before allocation (section 10).** The scheduler reads the same owner and park table through a holder under a fixed `sys.modules` key, the `_qwen_prefill_gate` pattern. A request whose state is gone is quarantined before vLLM allocates or publishes anything for it.

### 2.2 Why every exactness argument still holds

- **COLD and SCRATCH** are Lever N's op-sequence identity (design 3.4): the same chunk programs on the same bytes in the same order, at the model's own 2,048-token boundaries.
- **CHECKPOINT** is G1's argument: a byte copy of the fp32 recurrent state and the conv carry the chunk program wrote at an absolute boundary, followed by the same programs.
- **PARKED** is the same capture (`_qwen_prefix_read_scratch`) and the same restore (`_qwen_prefix_restore`), applied to a state the request took itself. A park at S and a checkpoint at S are the same bytes.

The restore path's one point the CPU fixture cannot see, the conv-carry tile padding (`qwen_prefix_model_patch` `_qwen_prefix_restore` docstring), is on the path every production hit already runs. Its hardware check is the ship pack's prefix exactness job (`references/tp4-ship-262k-prefix-jobs`, P1). Parking adds no new restore path; G-NP1 and G-NP3 re-read it with parks.

---

## 3. The merged route

### 3.1 Where it lives and what it calls

- **`levern_route.route`** grows from one source (SCRATCH) to four.
- **`prefill_forward`** (the wrapper on the vLLM model):
  - sends every announced step to the merged route, whole prompts included. Stage 1's deviation 3 is reversed under the merged profile; it stays as it is under Lever N alone;
  - sends only unannounced start-0 steps (the platform and request warms, before serving) to the original;
  - refuses an unannounced start-0 step while the scratch has an owner or a park slot is in use (an invariant: it would reset a suspended state).
- **What the route calls in the staged model tree.** These are instance methods and module functions that the G1 stage already puts into every C2 image's `model.py`: `prefill_traced_chunked(..., start, capture_at, on_capture)`, `_qwen_prefix_restore`, `_qwen_prefix_read_scratch`, `_qwen_prefix_capture`, `_bind_gdn_prefill_scratch`/`_unbind_gdn_prefill_scratch`, `_write_gdn_slot`, and the module functions `_qwen_prefix_row`, `_qwen_prefix_registry`, `_qwen_prefix_declare_mid_loop` and `_qwen_prefix_dram`, which the route reaches through `sys.modules[type(model).__module__]`.

  `levern_route.requirement_problems` adds all of them to its attach-time check, so an image without them fails the attach by name.

### 3.2 One step

1. Resolve the source (2.1) against the announcement, and refuse on any disagreement.
2. If the scratch has an owner other than X, parking is required. Under v2 the announcement carries `park_owner=True` and a free park slot exists: capture the owner's scratch into the slot. Anything else is an invariant refusal (the scheduler never schedules such a step).
3. Bind the scratch. For CHECKPOINT or PARKED, restore. Run `prefill_traced_chunked(tokens[:, :E], page[0:1], actual_len=E, start=S, capture_at=C, on_capture=hook)` with `C = {p in X's plan : S < p <= E}`.
   - The hook is G1's `_qwen_prefix_capture`, so `loop_pos == pos` is enforced by the registry.
   - When C is empty, both keyword arguments are passed as `None`, exactly as stage 1 calls it.
4. On a non-final step, take no snapshot and write no slot. Then:
   - `owner = (X, E)`;
   - assert after unbind that every GDN layer's `rec_state` is the batched buffer again. This is design hazard H3's identity check, which stage 1 dropped and which section 8 needs back;
   - log `wrote_slot=0`.
5. On the final step, snapshot the scratch, write the slot (`_write_gdn_slot`), clear the owner and log `wrote_slot=1`.
6. **Markers.**
   - The existing `[PINDIAG] lever N route ... ` line per step gains `source=`.
   - The final step also emits one `[PREFIX] row=` line in G1's format, so `prefix_markers`, `prefix_judge` and `c2_prefix_gate` read the merged route unchanged. Its fields are: `Q` = the first step's start; `L` = P; `plan` = the grant's plan; `captured=[...]` aggregated over every step of the request; `restored_ms` from the first step; `programs=A->B` over the request; and `slot_sha` and `logits_sha` under `QWEN_PREFIX_DIGESTS`.
   - The G2 DRAM readings (`dram after registry` and `dram after first capture`) come from the same adapter function, so the bring-up rules hold.

### 3.3 The warm

`levern_route.warm` runs before any trace, at the runner's width. It becomes:

- `[0, 2048)`, capturing at 2048 into a warm-local sink;
- `[2048, 4096)` SCRATCH;
- a CHECKPOINT restore of the 2048 state, then `[2048, 4096)` again;
- the final step `[4096, 6208)`;
- under v2, a park and unpark around the final step.

Every program the four sources touch then exists before the packed traces. The rule `B - A - W == 0` is unchanged.

---

## 4. The five refusal sites: exact changes

| # | Change |
| --- | --- |
| R1 | **The contract.** `levern_problems`' loop over `(PREFIX_SWITCH, STICKY_SWITCH)` is replaced: either both are on, or both are off. With both on, `prefix_reuse_problems` must pass, and the image must carry the merged route. That is a capability constant, `levern_route.SOURCES` naming all four sources, read by the contract at boot. It is not a flag, so a stage-1 image cannot boot a merged profile. Everything else in `levern_problems` is unchanged: gate-only until the ladder, `KV_RESERVATION=1`, budget equal to context, no lane, no per-admission credit. |
| R2 | **The G1 graft.** `install_problems` accepts `enable_chunked_prefill` only when all of these hold; any other chunked configuration is refused with today's text: `QWEN_FAST_STICKY_SESSIONS=1`, `QWEN_FAST_LEVER_N=1`, `type(scheduler).schedule` carries `levern_scheduler.WRAPPED`, `_schedule_prefill_only` carries `serving_prefill_admission.WRAPPED`, `max_num_scheduled_tokens` is an int (the cap's own path), `long_prefill_token_threshold == 0` and `max_num_batched_tokens >= max_model_len`. **The threshold refusal (:344-347) stays absolute.** `levern_scheduler.apply_cap`'s fallback through the threshold is refused when a prefix graft is installed, because vLLM would split there without the cap's alignment. The install line gains `chunked=levern`. Section 6 has the rest of the graft change. |
| R3 | **The sticky lifecycle.** In `_granted_resume`, `chunk != prompt - computed` becomes, under Lever N, `not levern_policy.valid_step(P, R, chunk)`: either `R + chunk == P`, or `(R + chunk) % 2048 == 0` and `R + chunk <= S_last(P)`. The other clauses (a committed grant, Q equal to R, R aligned, `R <= P - 2048`) stay as they are. They are graft invariants, so a failure is fatal. The admission announces `source=CHECKPOINT`. |
| R4 | **The prefix contract.** It keeps the check `budget >= context`. The reason text becomes: "vLLM's own budget must never split a prefill; only the Lever N cap may, at 2,048-token boundaries with the drafter window inside the final step". Under Lever N, `levern_problems` already requires `budget == context`. |
| R5 | **The route.** `:313-316` stays as an invariant: every served step is announced, and hits now are. `:414-418` becomes the source resolution of 2.1, and its refusal remains the last line behind the scheduler's pre-allocation check (section 10). |

**Two epoch-bump sites change in the same commit** (section 8): `serving_lifecycle.py:579/647` and `serving_worker_hook.py:336/356`.

---

## 5. Checkpoints from a split prefill

1. **The registry keeps an in-flight plan.**
   - In `commit`, a grant whose admitted request was scheduled fewer tokens than `P - Q` (a split) is also recorded in `registry.inflight[req_id] = (plan, tokens)`.
   - `begin_step` keeps `inflight`, but unpins the checkpoint after the commit step: the restore ran in that step.
   - `capture` accepts a committed grant or an in-flight plan. Every other rule is unchanged: a planned position, `loop_pos == pos`, the budget, the kill switch.
   - An entry is dropped on the request's final step (`commit` sees it among the cached requests with `num_computed + scheduled == P`), on `_free_request` (`forget_request`), on kill and on reset.
   - New counters: `inflight_captures` and `inflight_dropped`.
2. **The route filters.** `capture_at = plan ∩ (S, E]`. A planned position is taken in the step whose loop runs past it, inside the loop (mid-loop: the model declares this at warm) or at the step's drain (`pos == E`). Both cases already go through the G1 stage's loop edits. At the final step, a gap boundary at F runs before the tail (`EAGER_TAIL_NEW`).
3. **What gets captured.**
   - Under sticky sessions, the prompt boundary is `floor2048(P) - 2048 = S_last`. It is captured at the drain of the last non-final step. The next turn trims to it exactly as today: the turn re-prefills from the old `S_last`.
   - The gap boundary (a sibling's shared prefix) lands in whichever step runs past it.
   - Nothing above `floor2048(P)` is planned, as today.
4. **Publication is unchanged.** The cap (`cap` :589) publishes blocks below `floor2048(P)` at each step's allocation. Every such block is written by that step before the next `schedule()` (async scheduling is off). The same-step rule still protects a hit in the step that published.
5. **The equality claim.** With Lever N on and off, the registry at the end of every turn holds the same keys, positions and token ids, and byte-identical states. The same `prefix_judge.Oracle(sticky=True)` predicts every h, Q and plan.

---

## 6. The G1 scheduler graft

**D1, the cap before the hit.**
- *The defect.* For a fresh admission, Lever N computes the budget with `computed = 0`. A hit then runs `[Q, Q + budget)`.
- *Why it breaks the window.* Q is at most `S_last`, and the most common next-turn hit is close to it. With `Q = S_last` and a tail, the cap gives a non-final step `[S_last, F)` and a tail-only final step: the drafter window `[P - 2048, P)` straddles two steps (hazard H1).
- *Why nothing catches it.* `verify` accepts it: the end is aligned and P is at least 4,096.
- *The same with the solo budget.* 16,384 overshoots `S_last` the same way.

**The changes:**

1. **Peek.** `SchedulerGraft.peek(request) -> Q`.
   - It runs the trim for a waiting request with `h` from `coordinator.find_longest_cache_hit(...)`, the same call `get_computed_blocks` makes, but without vLLM's `prefix_cache_stats` record.
   - It memoizes `(blocks, h, Q)` per request for the current `schedule()` call. The in-pass `get_computed_blocks` wrapper returns the memo instead of calling vLLM twice, so stats, staging and token checks are counted once.
   - The memo is cleared in `begin_step`.
2. **The cap from Q.** `LevernRuntime.budget` for a fresh target uses `step_budget(P, Q)`. `verify` gains the window rule: a non-final end must be at most `S_last(P)`.
3. **A single-candidate pass.** During a capped pass, the waiting queues hold only the target, and every other partial is hidden from `running`. Admission already hides queues this way; a v2 parked partial is hidden the same way. vLLM therefore cannot admit another request in that pass.
4. **The install rule (R2) and in-flight plans (section 5).**
5. **Composition.**
   - The order is: the graft's instance `schedule` (`begin_step`, then commit) wraps the class `schedule` that Lever N wrapped, which wraps the plugin's. It holds because the fast path's attach, which installs Lever N on the class, runs in `compile_or_warm_up_model`, before vLLM builds the scheduler (the `serving_startup.prefix_warm` docstring), and the graft binds `scheduler.schedule` at construction.
   - The R2 install check (the class schedule carries Lever N's marker) turns a broken order into a boot refusal instead of a silent bypass of the alternation.

**Why it needs an image rebuild.**
- The graft is not a runtime mount:
  - the image build's `qwen_prefix_stage apply` AST-patches the pinned plugin `scheduler.py` (bf77cd63) with a hook in `TTScheduler.__init__`;
  - that hook imports `.qwen_prefix_scheduler_patch` relatively, from the plugin package;
  - `docker/qwen-c2-overlay.txt` lays the graft and the registry at the plugin path and in `/experiment-scripts/ci`;
  - `c2_image_provenance` (b) hashes the plugin copies against the source;
  - the hook runs inside the EngineCore at scheduler construction.
- Every other file this design touches is an overlay module of the baked evidence tree: `levern_*`, `serving_lifecycle`, `serving_prefill_admission`, `serving_worker_hook`, `verify_prestage`, `dflash_prefill_window`, `serving_runtime` and `serving_c2_contract`.
- So the whole change ships as **one new serving image** (B0 of the card plan).
- **No pin moves:** not the hook bytes (`INIT_HOOK` is unchanged), not `SCHEDULER_SHA256`, not the model stage's `PATCHED_SHA256`.

---

## 7. Page-table width

- The merged route passes the runner's table (`runner.max_num_blocks_per_req`, 4,096 at 262k) unchanged, which is the width `prefill_warm_before_traces` and `levern_route.warm` compile.
- It never calls `_qwen_prefix_fit_page_table`. That fits to `_chunk_full_page_table_buf`, and a different width compiles the paged SDPA after the traces are parked (#48536).
- **At attach, under the merged profile:**
  - `_chunk_full_page_table_buf` must be absent. That is stricter than sticky's "absent or equal" (`serving_runtime.py:262`): the merged route has no fitting branch at all;
  - `_chunked_trace_id` must be `None` (the eager loop only: `trace_mode=decode_only`);
  - the warm (3.3) must show zero program growth for the CHECKPOINT and PARKED sources at that width.

---

## 8. The fixture epoch and W1's per-block pre-stage

**Today.** Every Lever N step bumps the global epoch twice (1.4). In `blocks` mode every external bump kills both blocks' snapshots, so the first round after every step stages both blocks in full. That is about 22.7 ms of device idle a round (m: the tp4/hostgap measurement at 32k).
- At f = 0.5 it is one round in about three.
- At R = 1 it is every round, so W1's two-block saving is lost for the whole prefill.
- For a cold 254k (123 steps) it costs about 2.8 s (cm), plus the loss of the diff path while the prefill runs.

**Change: a third writer class, `disjoint`, next to `global` (external) and `local` (a block's own writes).**

- A writer in this class bumps no epoch. It logs a counter, `verify_prestage.note_disjoint(reason)`, so the hostgap lines still show it.
- **An intermediate Lever N step is in the class only if all of these hold:**
  1. the route measured `wrote_slot=0` for it;
  2. the H3 identity check after unbind passed (3.2 step 4), so no binding the window validated can have moved;
  3. at attach, after the route warm, `verify_prestage.register_external_writer('levern-route', addresses)` registered the chip-local addresses of the route's persistent writes (the scratch's `rec_state`, `conv_carry` and `conv_states` per GDN layer), and they are disjoint from every block's staging destinations. On overlap, here or at a block's later first pre-stage (`register_destinations`), the class disengages and every step bumps globally again (the safe direction), with a `[PINDIAG] verify prestage external writer refused reason=overlap` line;
  4. `QWEN_FAST_LEVERN_EPOCH_SCOPE=route`. The default under the merged gate profiles is `route`, and `global` is today's rule for the A/B.
- **Why these cover everything a step writes.** A step's other device writes are transients, allocated and freed inside the step: they cannot alias a live fixture buffer. The KV pool is data the verify reads through its page tables, not a staged input; the value diff recomputes page tables at verify time anyway. A CPU census of the route's write call sites, on the AST like `test_tp4_packed_prefix_audit`, holds this list. A new write site fails the census.
- **Bump points.** The bump moves after the step (it is decided by the measured `wrote_slot`), and is still before the next verify. The first step of a split prompt (:579, hook :336) and continuations (:647, hook :356) follow the same rule. **The final step, the engine build and the admission stay `global`:** one bump per prompt.
- **Per-block scoping of the admission** (only the block the new seat joins) is *not* proposed. The engine build captures new traces, and 1c's skipped binding check leans on the global bump after it.
- **The hardware gate.** The audited arm with `QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT=1` runs during interleaving:
  - zero `[PACKED-PRESTAGE-FULLAUDIT]` mismatches on rounds after intermediate steps;
  - at least 95% `path=diff` on those rounds, excluding the per-prompt admission bump;
  - the `[PACKED-PRESTAGE-SHADOW]` check runs and passes.

---

## 9. Prefill admission v2: a short lane with host parking

**Problem.** In v1 a request waits behind the whole remaining cold prefill. With prefix reuse most turns are short:
- a hit's own device time is about 3-7 s including the build (e: Q at 40-244k, deltas of 4-9k);
- behind a cold 254k at f = 0.5 it would wait 48-144 s, depending on when it arrived (e).

**The rule (scheduler: `serving_prefill_admission` + `levern_scheduler`; lifecycle; route).**

| Item | Rule |
| --- | --- |
| Classes | For each waiting request, `remaining = P - peek(Q)`. **Short** means `remaining <= QWEN_FAST_LEVERN_SHORT_TOKENS` (16,384, 8 steps); **long** is everything else. A cold prompt that is short is in the short class too. |
| In flight | At most one long and one short at a time (`1 + QWEN_FAST_LEVERN_PARK_SLOTS`, slots = 1). A short never parks another short. |
| Preemption | A short may be admitted while a long is in flight if all of these hold: a park slot is free, a seat is free, the KV reservation holds, the DRAM admits it, the long's park age is below `QWEN_FAST_LEVERN_MAX_PARK_S` (30 s), and the deadline governor (11) does not need the long. The short is admitted at the long's next step boundary. The long is hidden from `running` for the short's steps. |
| Park | The route captures the long's scratch to host (`_qwen_prefix_read_scratch`: about 154 MB of host RAM; e: 50-150 ms) when the short's first step runs, and restores it (`_qwen_prefix_restore`; e: 50-150 ms) at the long's next step (`source=PARKED`). The scheduler checks free host memory before it lets a short preempt. If the host read still fails inside the route, the short has been allocated already and must run: the long prefill is ended per request (D2 quarantine, `park_failures` counted, released on the finished id by the route and the lifecycle) and the engine stays up; with no quarantine consumer installed the failure stays fatal. |
| Order | Shortest remaining first among shorts, FIFO on ties. The long resumes when no short can be admitted. Longs run FIFO, one at a time, as in v1. |
| Short pacing | A short's steps alternate at R = 1, so decoder gaps stay at about one step and the short's TTFT rises about 15% (e). |
| Lifecycle | `request_id`/`capture` becomes `prefills[req_id]`, a dict of capture, refusal and ignore_eos per in-flight prefill. The gate holds a set. `_is_continuation` is keyed on the scheduled cached id. `_release_prefill(req_id)` also frees that request's park slot. |
| Device memory | Nothing new on the device: a park is host memory. A device park slot (38.5 MB per chip, design 3.8) is a later option if measured switch cost matters. |

**Result (e).** A hit turn arriving during a cold 254k waits at most one step (about 1.05 s at 250k of context) plus the park, then takes its own 3-7 s with R = 1 pacing. Today, without Lever N, the same turn waits up to about 94 s behind the cold whole prefill.

---

## 10. Fail-closed refusals: per-request, or fatal and unreachable

**The per-request mechanism already exists:** D2 quarantine (`serving_request_quarantine`). It gains a scheduler-side registration, before the pass, for a request that is hidden from this step:
- the consumer in `update_from_output` finishes it as `FINISHED_ABORTED` with no new tokens (the W5b form);
- the lifecycle releases its capture, owner and park on `finished_req_ids`.

Nothing is allocated or published for the step that would have run it, so no unwritten block can be cached.

**A refusal after allocation is never made per-request.** The cap publishes a step's blocks at allocation, so an aborted step would leave hashed blocks that were never written, and a later checkpoint chain could serve them. Every check that can run before allocation is moved before it.

| Site | Today | Merged |
| --- | --- | --- |
| `levern_scheduler.py:64`: more than one partial | engine dies | Up to `1 + PARK_SLOTS` is legal. Beyond that, the youngest extra partial is quarantined before the pass. Its earlier steps' blocks were written, so nothing is unwritten. |
| `levern_scheduler.py:142-144`: the pass admitted a request other than the target | engine dies after allocation | Unreachable: the capped pass sees only the target (6.3). Stays fatal if reached, because two prefill rows in one step break the route's one-row rule. |
| `levern_scheduler.py:147-150`: a non-final end off the boundary or past `S_last` | engine dies after allocation | Unreachable: the cap comes from the peeked Q (6.2). Stays fatal. |
| `levern_route.py:313-316`: an unannounced `start > 0` | engine dies | Stays fatal. Every served step is announced, hits included, so reaching it means the lifecycle and the route disagree. |
| `levern_route.py:414-418`: the owner | engine dies | Becomes the source resolution (2.1). The scheduler checks the shared owner and park table before the pass. A request whose state is gone (a park lost to an error, an owner taken by an unannounced warm) is quarantined, and the client gets `finish_reason=abort`. A retry is then a cheap prefix hit. The route assertion remains as the invariant behind it. |
| `serving_lifecycle.py:301-304`: sticky | engine dies | The whole-rest clause becomes the plan (R3). The grant clauses stay fatal (graft invariants). |
| `levern_scheduler.py:102`: no cap path | attach fails | Unchanged: boot-time refusal is correct. |
| Capture failure | counted, never fails | Unchanged (S7). |
| Park failure (new) | none | The long prefill is quarantined (FINISHED_ABORTED) at that step; the short runs. No engine failure with the consumer installed. |

**A Lever N kill switch (new).** The file `levern.off` sits next to `prefix-reuse.off` under the image's `.qwen-c2` directory and is polled like the prefix kill switch.
- Once present, new admissions get a whole-prompt cap (no splitting) and the short lane closes.
- In-flight partials finish through the route.
- It latches for the life of the process.
- It turns "Lever N misbehaves" into an operator action without a restart.

---

## 11. The TTFT budget at 262k: which f

**Numbers.** The timing model (e) is chunk t(p) = 420 ms + 2.5 ms per 1k of context, fitted to T3/T4; 8-live round 245 ms; transition 40 ms per step; build 1.9 s. The case is a cold 253,920-token arrival with seven long decoders.

| Prefill share | TTFT | Decoders' rate inside the window |
| --- | ---: | ---: |
| off (today) | 93.6 s | 0 |
| f = 1.0 (no yields) | 98.6 s | 0 |
| R = 1 | 128.5 s | 4.2 tok/s |
| f = 0.6 | 162.3 s | 7.1 tok/s |
| f = 0.54 | 179.9 s | about 8 tok/s |
| f = 0.5 | 192.2 s | 8.7 tok/s |
| f = 0.4 | 240.2 s | 10.6 tok/s |

With agent decoders that have 176-543 tokens left, the arrival's TTFT is 108-129 s at any f < 1 (e).

**A fixed 0.5 is not acceptable at 262k:**
- a lone cold maximum prompt has 48 s of margin to the 240 s client deadline, on a chunk time never measured at 250k;
- a second cold 254k queued behind it gets about 291 s (e). Today that is 187 s.

**Decision: `f_base = 0.5` with a deadline governor in `levern_policy.Alternator`.**

- **Flag:** `QWEN_FAST_LEVERN_TTFT_TARGET_S` (default 180, the 240 s client deadline minus 60 s of margin; 0 turns the governor off).
- **The input:** each pending prefill j (in flight, parked or queued), with its arrival time (vLLM `Request.arrival_time`), its remaining device time D_j and a build B (5 s, measured at v584/v620; the model had 2.5 s).
- **D_j is measured online:** an EWMA of the route's per-step ms against its context, seeded with t(p). Constants are not trusted at 254k.
- **The rule:** `f_eff = clamp(max(f_base, max_j (sum_{i<=j} D_i + B) / (T* - (now - arrival_j))), f_base, 1.0)`, with i running in service order. At `f_eff = 1.0` the share owes nothing, but the **decode-gap floor** still runs: `QWEN_FAST_LEVERN_MAX_DECODE_GAP_S` (G, default 8, 0 = off) gives the decoders one decode round at least every G seconds, and a short prefill step is still followed by its one round. The static share 1.0 control arm keeps no yields. (The first design said "no yields at 1.0"; v584 and v620 showed two cold 254k prompts pin the governor at 1.0 for the whole second prompt, and the decoders stalled 92-98 s.) The floor costs one 167 ms round per G seconds: about 2% of a 97 s long, about +4 s on the second long's TTFT.
- **Logging:** the step line gains `f_eff=`, `need=` (the unclamped demand; above 1.0 the deadline cannot be met) and `gap_ms=` (wall time since the last decode round); the install logs `lever N governor: ttft= gap_floor=`.

**Effect (e):**
- **A lone cold 253,920:** the governor runs it at about f = 0.54, about 180 s.
- **Prompts up to about 240k:** the governor never binds (168.6 s at 229,376 with f = 0.5).
- **Two simultaneous cold 254k:** about 98.6 s and 197 s, against today's 93.6 s and 187 s.
- **The bound:** the governor never makes a cold arrival slower than today plus about 40 ms per step of transition. Three or more simultaneous cold maximum prompts breach 240 s today as well.

**What the sweep decides and what it doesn't.** f_base is still chosen by the agent-turn sweep (G-NP6: 0.33 / 0.5 / R = 1 at T* = 180). T* is fixed by the client deadline, not swept. Shorts (9) are paced at R = 1 and are never governed (also while the governor holds 1.0: `short_owed` forces their round).

---

## 12. Exactness proof plan

### 12.1 CPU unit tests

These run locally, targeted, with `py -3.11 -B -m unittest <modules>` from `scripts/ci`. Every new module goes on the allowlist in `.github/workflows/qwen-integration-cpu.yml` (the `lever-n` job), and none is named `test_serving_*`.

| Module | Holds |
| --- | --- |
| `test_levern_policy` (extended) | `valid_step` and `plan_from(P, Q)` for every P in {2047, 2048, 2049, 4095, 4096, 4097, 6143, 6144, 6145, 8191, 10241, 32785, 253920} and every aligned Q up to `floor2048(P - 2048)`: every non-final end aligned and at most `S_last`, and the window inside the final step. The governor on a fake clock: the f_eff formula, the clamp, and that it never goes below f_base. |
| `test_qwen_prefix_registry` / `_scheduler_patch` (extended) | The in-flight plan's life (commit, steps, the final step, free, kill, reset); a continuation capture at a planned position is stored, and at an unplanned one or a wrong `loop_pos` is refused and counted; the checkpoint is unpinned after the commit step; the peek is idempotent within a step and the in-pass call reuses it; R2 accepts exactly the merged conditions and refuses each one missing with today's text. |
| `test_levern_prefix_route` (new; real staged `model.py`, FakeTTNN and the toy hybrid model of `qwen_prefix_model_fixture`) | **E1, hit equals cold twin:** turn 1 is a cold split prompt that captures at `S_last`; turn 2 extends it and hits at Q (CHECKPOINT, then SCRATCH steps). Its final slot, logits and KV [0, P2) equal a cold whole prefill of P2 and a cold split prefill of P2, byte for byte. **E2, the boundary matrix:** the P and Q set above x step in {2048, 4096, 16384}. **E3, the capture round trip:** each checkpoint stored by the split route (mid-loop and at a step's drain) equals the one the whole-prompt G1 route stores at that position, byte for byte, and the registries of the on and off runs are equal (section 5.5). **E4, park:** A split, parked at step k; B, a hit, runs to completion; A resumes and finishes. A and B each equal their solo runs. **E5, negative controls:** a wrong checkpoint, a skipped park, a foreign scratch, a capture at the wrong `loop_pos`, a CHECKPOINT on a non-first step. Each changes the digests or is refused before FakeTTNN records any device op. **E6:** every call at width 4,096; zero programs after the warm across all four sources; a model with a chunk buffer refused at attach. |
| `test_levern_lifecycle` / `_capture` (extended) | A hit admitted split (`source=CHECKPOINT`, the capture from Q across segments, `records_prefix_route`); continuation; abort during a hit's prefill and during a park; displacement only on a measured slot write; the epoch scope (`disjoint` without a bump, `global` on a final step). |
| `test_verify_prestage` (extended) | `register_external_writer`; overlap disengages; `disjoint` moves neither epoch; the audit flag still forces the full stage. |
| `test_levern_contract` (extended) | The merged profile is accepted; each requirement is refused by name; a stage-1 image is refused; the ship-prefix argv and env are byte-identical with the flags off. |
| `test_levern_scheduler` (extended; reduced scheduler) | The short lane, SRPT, the park cap, aging, the governor, the single-candidate pass, pre-pass quarantine. |
| `test_levern_prefix_census` (new) | The route's device-write census against the audited list in section 8. |

### 12.2 Real-vLLM CPU proofs

These run on vLLM 0.25.1 in `qwen-fast-vllm-cpu.yml`, through experiment/fast-vllm-cpu-vN tags only. v19 to v30 are unused today; check again before tagging. One tag per phase.

- **V1, phase A (`test_levern_prefix_scheduler_vllm`, new).** The real `TTScheduler` staged with the G1 graft (sticky sessions, DFlash lookahead 16, drop-last) plus the Lever N wraps and the KV reservation.
  - A chained conversation: every turn hits at the Q that `prefix_judge.Oracle(sticky=True)` predicts. That is the same sequence as with Lever N off.
  - Every turn splits per `plan_from(P, Q)`.
  - `FakeGdnModel` is extended with SCRATCH: every restored or continued state equals the cold hash chain.
  - Captures at `S_last` land in the right step.
  - The peek leaves `prefix_cache_stats` unchanged.
  - No preemption; the reservation bound holds; every block is free at the end.
  - Negative control: with the old cap (computed from 0), a hit at `S_last` splits the window, and the proof must see it.
- **V2, phase B.** A hit arriving during a cold partial:
  - it is admitted at the next boundary (the cold hidden), finishes, and the cold resumes;
  - at most two partials;
  - no preemption; the reservation holds;
  - a pre-pass quarantine frees every block and publishes nothing unwritten.
- **The CPU suite**, once at the end of each phase, on the dev queue: `gh workflow run qwen-cpu-suite.yml --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B -f ref=tp4/levern-prefix`.

### 12.3 Card gates (listed, not run)

**Image:** one image (B0) built from the phase-B head. **New gate-only profiles:**
- `c2-packed-tp4-8x262k-ship-prefix-levern` = `ship-prefix` plus `QWEN_FAST_LEVER_N=1`, step 2048, solo 16384, share 0.5, T* 180, short 16384, `PARK=host`, `PARK_SLOTS=1`, `MAX_PARK_S=30`, `EPOCH_SCOPE=route`, gate_only;
- its audited twin (`...-levern-audit`), which adds `QWEN_FAST_LEVERN_AUDIT`, `QWEN_PREFIX_DIGESTS`, `QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT` and the verify audits;
- the controls are `ship-prefix` and `ship-prefix-audit`.

| Gate | Shape | READ |
| --- | --- | --- |
| G-NP0 attach | warmup | the graft's install line with `sticky=1` and `chunked=levern`; Lever N installed before it; the route warm with all four sources and program growth 0; no chunk buffer; the external-writer line `engaged` |
| G-NP1 exactness | `levern_equal` at the 12 boundary lengths, single and busy; the prefix gate's `exactness-eager` strict pairs; `agent-turns-prefix` transcripts against the control | hit and salted-cold twin digests identical; `[PREFIX] row` Q equal to the oracle; texts equal the control and solo |
| G-NP2 registry equality | the same conversations, with Lever N on and off | per turn: checkpoint keys, positions and `slot_sha` identical |
| G-NP3 hang shapes (x3) | H1-H3 of tp4/lever-n, plus a hit arriving mid-cold (park), an abort during a parked long, the kill switch mid-prefill, and a flood-eviction of checkpoints (L1-style) | every stream completes; quarantines only where planned; no stall-watch exit |
| G-NP4 pre-stage | the audited arm while interleaving | zero full-audit mismatches; at least 95% `path=diff` after intermediate steps |
| G-NP5 stall/TTFT ABAB | `stall8_cold262k` (control `ship-prefix`); a hit arriving mid-cold; two simultaneous cold 254k arrivals | worst seat gap; seat tok/s inside the window; cold TTFT at most 180 s; hit TTFT at most its solo + 1.5 s; second cold under 240 s; worst seat gap at most G + one atomic step (the final step with its build) |
| G-NP6 `agent8_turns-prefix` | f_base in {0.33, 0.5}, R = 1, and the control | p50/p90 turn latency; choose f_base |
| G-NP7 churn16 | merged arm | zero engine deaths; quarantines counted and attributed |

A traffic twin comes only after the ladder, and only on the owner's ship decision.

---

## 13. Build order

| Phase | Content | e |
| --- | --- | --- |
| A | R1-R5; the peek and cap from Q; in-flight plans; the merged route (COLD, CHECKPOINT, SCRATCH) and its warm; markers; the epoch class and census; V1; the CPU suite once | 6-8 eng-days |
| B | v2 (lifecycle dict, the short lane, host park, PARKED); the governor; pre-pass quarantine; the kill switch; V2; the CPU suite once | 6-9 eng-days |
| C | B0 image and the window G-NP0 to G-NP7 (about 14-18 h of cards) | after A and B |

**New flags** (each defaults off or to the value shown; each is parsed strictly by `levern_policy`): `QWEN_FAST_LEVERN_TTFT_TARGET_S` (180), `QWEN_FAST_LEVERN_SHORT_TOKENS` (16384), `QWEN_FAST_LEVERN_PARK` (0 or host), `QWEN_FAST_LEVERN_PARK_SLOTS` (1), `QWEN_FAST_LEVERN_MAX_PARK_S` (30), `QWEN_FAST_LEVERN_EPOCH_SCOPE` (global or route).

## 14. Risks

1. **The plugin runner with two partial prefills in its persistent batch** (a long hidden while a short runs) has never been run on TT. The CPU proofs cover the scheduler and the route, not the runner's row handling. G-NP3 is the first check; a slot move (v121) is the expected failure mode.
2. **The park and restore costs are estimates.** If a switch costs more than about 0.5 s, a device park slot (design 3.8) replaces host parking.
3. **The 254k chunk time is fitted, not measured.** The governor measures online, but its seed and T* margin assume the fit is within about 25%.
4. **The peek must reproduce vLLM's own hit length exactly.** It uses `find_longest_cache_hit` with the same max length. V1 pins it against `get_computed_blocks` on 0.25.1.
5. **The disjoint-writer class depends on the census staying complete.** A future route write site must fail the census, not pass silently.

## 15. What was built, where it departs from the text above, what is open

Built on `tp4/levern-prefix` (CPU only; no card, no tag but the CI proof tag). File by file:

| File | Change |
| --- | --- |
| `levern_policy.py` | `valid_step`, `plan_from`, `merged_config` and the six merged flags, `StepTimes`, `effective_share` (the governor), `Alternator` share and short pacing, `OffSwitch`, the shared route state (`state_holder`, `state_has`), the merged line formats |
| `levern_route.py` | `SOURCES`; the merged route (`_resolve`, `_merged_route`, `_park`, `_aggregate`, `_warm_merged`); `persistent_writes` and `engage_epoch_scope`; the write census list |
| `levern_scheduler.py` | `PassPlan`, `plan_pass` (peek, single candidate, quarantine, short lane), the governor input, the kill switch, `verify` with the window rule |
| `serving_prefill_admission.py` | the wrapper applies the plan: hidden prefills, the single-candidate queue, the holds asked before anything is hidden |
| `qwen_prefix_scheduler_patch.py` | R2 (`levern_chunking_problems`), `peek`/`select`, the in-pass trim capped at the peek, in-flight notes in `commit`, the `install chunked=levern` line |
| `qwen_prefix_registry.py` | the in-flight plan (`Inflight`, `note_scheduled`, `planned`), counters |
| `serving_lifecycle.py`, `serving_worker_hook.py`, `dflash_prefill_window.py` | R3, source announcements, parking, the epoch scope, the capture ledger's first call through the range entry |
| `verify_prestage.py`, `serving_runtime.py` | the disjoint external-writer class and its registration at attach |
| `serving_c2_contract.py`, `qwen_c2_profiles.json`, `c2_smoke_check.py` | R1, R4, the two profiles, the merged smoke rules |

Departures from the design text:
1. **Profile names.** `c2-packed-tp4-8x262k-ship-prefix-levern` and `c2-packed-tp4-8x262k-ship-prefix-levern-audit`, not `-levern-gate`. They carry no waiver marker: their base is the production profile, which has none.
2. **The peek does not ask vLLM twice for the in-pass answer.** The peek calls vLLM's own `get_computed_blocks` once (with `prefix_cache_stats` swapped for a null object) and memoizes only Q; the in-pass call is vLLM's authoritative answer, capped at the peek (`select(..., ceiling=Q_peek)`). A hit that moved between the two can only lower Q. That keeps the end aligned for an intermediate step, but NOT when the peeked step was the whole rest (a lowered Q with the budget P - Q_peek ends unaligned and non-final): the budget for a peeked whole rest is therefore the whole prompt, valid from any Q' <= Q_peek (review B1, section 16). Nothing depends on a block object that may have been evicted in between.
3. **The install line.** The graft's sticky install line is byte for byte what it was (other tests and tools read it); `install chunked=levern: ...` is its own line, logged only where the merged route accepts chunking.
4. **Scheduler-side state reaches the route and back through one holder** (`_qwen_levern_state`: the scratch owner and the parks), read before any allocation. A partial whose `(request, computed)` is in neither is quarantined through the D2 consumer; with no consumer installed the refusal stays fatal.
5. **The warm** runs four steps and logs one warm line (`steps=4`) and no per-step route lines; the smoke rule counts the steps from the warm line under the merged profile.
6. **Whole prompts under the merged profile** go through the merged route too (one step, COLD or CHECKPOINT) but keep the lifecycle's `note_prefill` (global bump and displacement) exactly as today: the epoch scope changes only intermediate steps.
7. **Kill switch.** `levern.off` next to `prefix-reuse.off`, polled once a second and latched; new prompts then take their whole rest in one step and the short lane closes.

Open, not done here:
- **Card gates G-NP0 to G-NP7** (12.3). The new serving image (B0) is not built.
- **The host-memory check for a park** reads `MemAvailable` before a short may preempt; a park that still fails inside the route (an allocation error) fails the engine, because the short has been admitted by then. A device park slot (design 3.8) is the alternative if the check proves too weak.
- **The runner with two partial prefills in its persistent batch** is untested off-card (risk 1 of section 14); the CPU proof covers the scheduler, the lifecycle and the route, not the plugin runner's row handling.
- **The disjoint-writer class** is gated on the audited prestage arm (G-NP4); until then `QWEN_FAST_LEVERN_EPOCH_SCOPE=global` is the safe A/B arm.

## 16. Review of 5b6c0448 and what changed

- **B1 (blocking).** `peek` now polls the prefix kill switch through `kill_switch_engaged()`, as the trim does, so the two agree inside one `schedule()` call; and `LevernRuntime.budget` returns the whole prompt when the peeked step is the whole rest (`Q_peek >= S_last`) or `levern.off` is engaged for a fresh admission, so any lowered Q' still takes the whole rest, which `valid_step` accepts.
- **S1.** `pending_hint` counts a waiting long only if a seat would be free when it reaches the front, and not once it is already past T* (it cannot be helped, and 1.0 for it only stalls the decoders). An in-flight prefill past T* still gets 1.0.
- **S2.** The step-time model learns only from an intermediate step that continued its own scratch (no new admission, no final step, not the step that resumed a park); the EWMA weight is 0.2 from the first sample.
- **S3.** With no prefill in flight, the oldest waiting long goes before the waiting shorts when the governor needs it or it has waited `QWEN_FAST_LEVERN_MAX_PARK_S`.
- **S4.** A waiting request is classified afresh on every call; its class is fixed from the peek of the admitting call (`pending_class`, promoted once it runs).
- **S5.** A host park failure quarantines the long prefill per request (section 9 and 10 rows corrected).
- **S6.** The lifecycle releases a parked or in-flight prefill that vLLM preempted, as for a finished one; the admission wrapper detects a resumed single through `running` as well as `scheduled_new_reqs`, and hidden prefills return to their original positions in `running` instead of the tail.
- **S7.** Tests: the parked run's logits against the cold twins, a CHECKPOINT short that parks a long, the kill-switch race at the graft and at the budget, the governor inputs, aging, the class, the park failure, the preemption release.
- **S8.** The H3 identity check snapshots `B`, `rec_state`, `conv_states` (and each element), `conv_carry`, `_zero_conv0` and `_stable_state`.
- **Declined (minor).** The route's `request_ids[0]` cross-check: `run_chunk` receives the step's announcement and the runner's `start_pos`/`prompt_lens`, which it already compares; the runner's request ids are not among the arguments the wrapper receives at that entry. The peek is not cached across calls (it is memoized per call; the hint is now only computed when something is pending).
