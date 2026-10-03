# Lever N at TP4, eight seats: resumable chunked prefill interleaved with the packed decode rounds

> **In tree.** Stage 1 of this design is built on branch tp4/lever-n; what it covers, where it deviates from the sections below and what is left is
> docs/lever-n-tp4-stage1-2026-10-04.md. The timing model behind section 6 is docs/lever-n-tp4-timing-model.py. The text below is the design as written.

**Date:** 2026-10-04. **Base:** `origin/tp4/262k8` at 52ad5839 (eight seats, two packed 64-row blocks, 262k windows, all round levers, the eight-seat quad).
**Status:** read-only design. No branch, checkout or rig was touched. The only files written are this note and `levern_model.py` (the timing model behind section 6), both in this directory.

**Labels:**
- **m**: measured (the run is named).
- **cm**: computed from measured numbers.
- **e**: estimate.

**Prior work:** read in full and built on:
- `docs/lever-N-prefill-decode-interleave.md` (branch `lever-n-prefill-decode-interleave`, plus the newer uncommitted copy in the main checkout);
- `optimisation/sim/{INTERLEAVE,CONTINUATION,PREFILL-STATE}.md`;
- on the base: `docs/lever-n-plugin-contract-2026-09-19.md`, `lever-n-fastpath-scope-2026-09-22.md`, `lever-n-build-plan-2026-09-22.md`, `m2-alternation-verdict-2026-09-22.md`, `ttft-decode-stall-2026-09-22.md`, `prefill-chunk-size-verdict-2026-09-22.md`, and `four-streams-131k-feasibility-2026-09-23.md` (v117-v128).

---

## 0. Answer first

1. **Today, a prefill is one engine step, and the decoders stop for all of it.** Three things make it so:
   - the profile passes `no-enable-chunked-prefill` with a 262,144-token budget;
   - the TT platform turns chunking off for `qwen3_5` anyway;
   - the plugin's `TTScheduler` runs a prefill-only step whenever prefill is pending.

   Nothing explicitly pauses the packed rounds. The engine is synchronous (async scheduling off), so no decode round is enqueued until the prefill step returns. That step also builds the new seat's engine (about 2 s), and only then does the seat join its block.

   The cost of a cold 253,920-token arrival is about **92 s of prefill plus about 2 s of build (cm)**. For all of that time the seven other seats emit nothing.
2. **Most of Lever N already exists in pieces on this base.** What exists:
   - the lifecycle's continuation routing;
   - the multi-segment prefill capture;
   - the chunked-prefill frontier in `from_prefill`;
   - the resumable chunk loops: the G1 stage puts them into the image's `model.py` and `qwen36_vllm.py` on every C2 build;
   - the one-fresh-prompt scheduler wrapper.

   This was proven at TP2: v118 (4 x 32k) and v128 (4 x 131k) were byte-exact, and the worst decoder stall fell from 44.8 s to 4.1 s.

   **What is missing:**
   - the platform keeping chunking on;
   - a scheduler-side per-step token cap that lands on 2048-token boundaries;
   - an alternation policy in the scheduler class this deployment actually runs;
   - a model route that continues its own suspended scratch. The image's G1 route accepts start_pos > 0 only with a prefix grant, and refuses M1's tree (F5);
   - skipping the per-chunk GDN slot write;
   - a warm of that route before the traces are captured.
3. **Exactness is by op-sequence identity, not by numerical argument.** Every step boundary lands on the model's own 2048-token outer chunk boundary. At that boundary the stock eager loop already does three things:
   - synchronizes;
   - keeps the fp32 recurrent state and the conv carry in the persistent B=1 scratch;
   - reads earlier chunks' K/V from the paged cache.

   So a split prefill runs the same programs on the same bytes in the same order:
   - the GDN conv is the exact op, with its carry;
   - the GDN scan uses 128-token internal chunks, and 2048 = 16 x 128;
   - the chunked SDPA runs over the pages.

   **Steps that are not 2048-aligned (for example 1024) are not covered by that argument:**
   - they use different matmul M shapes and program configs;
   - they pad the masked bucket differently.

   They are refused.
4. **What it buys, and what it costs (e, section 6):**
   - The worst decoder gap during a cold 254k prefill falls from about 94 s to about 1.1 s inside the prompt. The final step, which merges the last chunk, the tail and the engine build, is about 4 s once.
   - Arrival TTFT depends on the decoders' outputs:
     - with long decoders (the `stall8_cold262k` shape), it rises 37% (one round per chunk) to 105% (prefill share 0.5);
     - with agent-sized decoders (p50 176 tokens left), it rises only about 16% (94 to 108 s) at any share. The decoders finish their turns in 15-32 s instead of 103 s.
   - Lever N is a latency and fairness lever, not a throughput lever.
5. **It does not fix the burst staircase, or the cost of re-prefilling the whole context every turn.**
   - Eight agent seats re-prefilling about 65k per turn need about 18 s of device time per turn each (cm). That saturates the device with prefill.
   - Only prefix reuse removes that (the owner's second question; the sibling design `tp4/packed-prefix`).
   - Lever N v1 keeps one prefill in flight, so a new turn from another seat still waits behind a cold 254k prefill. **v2 (scratch parking plus shortest-remaining-first among prefills)** removes that, and is the part that matters most once prefix reuse makes most turns short (section 3.11).
6. **Build:**
   - v1 is about 10-12 eng-days on CPU (scheduler and platform wrap, model route stage, lifecycle and capture, smoke tooling), then one card window (about 12-14 h).
   - v2 is +6-10 days and half a window.
   - Every piece sits behind `QWEN_FAST_LEVER_N=1`, on new gate-only profiles. With the flag off, the image serves byte-for-byte as today.

---

## 1. How a new request's prefill runs today (TP4, eight seats)

### 1.1 The configuration that forbids chunking

Profile `c2-packed-tp4-8x262k-best-time-gate` (`scripts/ci/qwen_c2_profiles.json`):

| Setting | Value |
| --- | --- |
| `no-enable-chunked-prefill` | true |
| `max-num-batched-tokens` | 262,144 (= `max-model-len`) |
| `no-async-scheduling` | true |
| `no-enable-prefix-caching` | true |
| `max-num-seqs` | 8 |
| `trace_mode` | `decode_only` |
| speculative config | DFlash, 15 tokens, greedy |
| `QWEN_FAST_M3_BLOCKS` | 2 |
| `QWEN_FAST_KV_RESERVATION` | 1, pool 19,968 blocks |
| `QWEN_FAST_DRAM_*_MB` | 800 / 600 / 256 |
| `QWEN_FAST_DECODE_STEPS_PER_ADMISSION` | unset (0) |

The plugin's `platform._apply_chunked_prefill_policy` (fixture `scripts/ci/fixtures/platform_chunked_prefill_policy.py`) logs:

> "Chunked prefill is not supported for model_type=qwen3_5; disabling it"

It then sets `long_prefill_token_threshold = 0`. `check_and_update_config` installs `vllm_tt_plugin.scheduler.TTScheduler`, not the lane coordinator: DP is 1 (proved in run 35707860782).

### 1.2 Who decides: the scheduler (EngineCore process)

`TTScheduler.schedule` (bf77cd63; fixture `scripts/ci/fixtures/plugin_scheduler.py:104-117`), default mode: if prefill work is pending, it calls `_schedule_prefill_only()`. Decode gets a step only when that call schedules zero tokens: the fallback to `_schedule_decode_only()`.

Our runtime wrapper `serving_prefill_admission.wrap` (`scripts/ci/serving_prefill_admission.py:725-814`) is installed on that class by the lifecycle under `QWEN_FAST_ANY_REQUEST=1`. It decides admission in this order:
- partials present: allowed = partials, queues hidden;
- the lifecycle's prefill gate is held: allowed = 0;
- the KV reservation hold;
- the DRAM split hold;
- otherwise one fresh prompt.

Every "allowed = 0" with a decoder running turns into a decode-only pass. `QWEN_FAST_DECODE_STEPS_PER_ADMISSION` (off) is the only existing alternation, and it applies per admission, not per chunk.

**Pausing the decoders is therefore purely a scheduling outcome.** The step is prefill-only. The engine (uniproc, async off) runs `schedule -> execute_model -> sample_tokens -> update_from_output` to completion before the next decode step can exist.

### 1.3 What runs: the worker (same process)

`FastServingLifecycle._execute` (`serving_lifecycle.py:356-549`) is installed over `worker.execute_model`:
1. **Step-shape check:** exactly one new request, text only, uncached (or a sticky grant).
2. **Admission side effects:**
   - `capture_factory(P)` builds the `dflash_prefill_window.PrefillWindowCapture`, wrapped by `serving_runtime.prefill_tripwire` to count programs per segment;
   - `note_prefill()`: the resident verifier engine is displaced, and `note_fixture_writer('prefill')` bumps the pre-stage epoch, so the next packed verify restages in full;
   - the gate is held: `request_id` sets `sys.modules['_qwen_prefill_gate'].held`.
3. **The model call**, inside `capture.capture()`, via `original_execute`:
   - the plugin `TTModelRunner.submit_prefill(start_pos=0, prompt_lens=P, empty_slots=[row])` calls `qwen36_vllm.prefill_forward` (fixture `qwen36_vllm.py:203`), which calls `_prefill_forward_tp_batched`, which calls `model.prefill_paged_slots` (fixture `qwen36_model.py:1719`). The G1 route `_qwen_prefix_prefill_slots` is taken only with `QWEN_PREFIX_REUSE=1`.
   - The model then:
     - binds the persistent B=1 GDN scratch. It is allocated before every trace by `serving_runtime.prefill_warm_before_traces`; this was the v172 fix;
     - `prefill_traced_chunked`: `_build_request_rope`, which for text clears to 1D RoPE;
     - `trace_mode=decode_only` means no prefill trace, so the **eager loop `_prefill_chunked_eager_tp`** runs (fixture `:2442`). It resets the GDN state, then for every full 2048-token chunk calls `_forward_prefill_chunk_masked_tp(...)` and **`synchronize_device`**, then runs the tail in the masked bucket and computes logits.
   - Per chunk, all 64 layers run:
     - **16 attention layers:** QKV, partial RoPE, paged K/V fill, chunked flex-SDPA over the request's pages, output projection and its collective;
     - **48 GDN layers:** projections, the exact prefill conv op (`gdn_prefill_conv_exact`, with its carry), the fused chunk scan (128-token internal chunks, fp32 state carried in the scratch), gated norm, output projection and its collective;
     - **MLP:** C1e packed gate/up, then down.
   - The capture wraps the chunk forward and snapshots the drafter's feature rows of the window [P-2048, P).
   - The model then host-snapshots the scratch: 48 x (rec fp32 plus conv), 154 MB across the four chips. It unbinds and calls `_write_gdn_slot(empty_slots[0])`.
4. **`_sample`** (same engine step, `serving_lifecycle.py:670-746`):
   - the native seed;
   - EOS, max_tokens and D2 quarantine branches;
   - `bridge_factory(state, capture)`, which calls `serving_request_factory.from_prefill` (`:361`):
     - the DRAM backstop;
     - `adopt_prefill_slot`: plugin slot row to slot 0;
     - the DFlash2 device from the window features. **A pool slot is acquired here** (`serving_buffer_pool.acquire`, then `placement_slot` `:893`): the lone-user block first, else the fuller open block;
     - the 2048 proposal bucket;
     - the verifier engine and its 1/2/4-row captures;
   - `hook.attach(bridge)`.

   **That is the seat joining its block:** block A is slots 0-3, block B is slots 4-7.
5. **The next decode step:** `FastWorkerHook` runs the packed round. Both 64-row verify traces are restored inside the trace, after a full restage. Then come the quad drafts per block and the fused commits.

### 1.4 What it costs, measured on this branch's image

| What | Value | Source |
| --- | --- | --- |
| Per-user TTFT step, 8 x ~30k (concurrent8_code_32k) | about 8.6 s per user, including the engine build | m: T3 v388 run 37124226448 and T4 v389 run 37125023011 |
| Per-user TTFT step, 8 x ~119.5k (concurrent8_code_128k) | about 36 s per user | m: same runs |
| Fitted chunk time at TP4 | **t(p) ~ 420 ms + 2.5 ms per 1k tokens of context** (the SDPA slope is half TP2's 5.4) | cm: `levern_model.py` reproduces 8.6 / 35.3 s |
| Cold 253,920-token prompt | prefill **~92 s**, + ~2 s build | cm: never measured; M9a and H4 did not run (window 1009 died on laptop disk-full, window 1010 cut H4) |
| 8-live round | ~245 ms (≈18 tok/s per seat at τ≈4.4); the device idles ~80 ms of it | m: the task brief |
| Today's decoder freeze | the whole prefill plus the build. For 8 x 120k the last user's TTFT is 291-321 s | m: T1-T4 |

---

## 2. What exists, and what is missing on this base

**Present on `origin/tp4/262k8`, inert until chunking is on.**

`serving_lifecycle.py`:
- `_is_continuation` / `_continue_prefill` / `_displace_after_continuation` / `_after_prefill_chunk`. This is the v80 routing fix; the continuation is tested before the hook branch;
- `_release_decoders` and `_release_prefill` are split, so a decoder finishing mid-prefill does not tear down the capture.

`dflash_prefill_window.PrefillWindowCapture.segment()`:
- a capture spans engine steps;
- the cursor ledger validates at the true end;
- the slot may move between segments (v121);
- the prefix route is recorded when `records_prefix_route` is set.

`serving_request_factory.from_prefill` accepts a frontier inside the prompt (run 35696842354).

`serving_prefill_admission`:
- one fresh prompt per step;
- partials hide the queues;
- the KV and DRAM holds;
- per-admission decode credit;
- `carry_finished` (finished ids put back after a held pass).

Image model tree: the G1 stage (`qwen_prefix_stage.STAGES`, applied in every C2 build) already provides:
- `chunk_from`/`chunk_to`/`do_reset`/`do_tail` resumable loops, traced **and eager** (`lever_n_model_patch.patch_tp_replay`);
- `prefill_traced_chunked(start=, capture_at=, on_capture=)`, which asserts a 2048-aligned start and resets only at 0;
- `_qwen_prefix_prefill_slots`, a per-row route with request ids, restore, and checkpoint capture, plus the program-cache marker per row;
- the runner patch carrying `request_ids`.

All of it is switched by `QWEN_PREFIX_REUSE=1`.

**Missing:**

| # | Gap | Where it bites |
| --- | --- | --- |
| A | Chunking stays off for qwen3_5 | the platform policy and the profile argv |
| B | No per-step token cap, so vLLM would schedule the whole prompt anyway | the budget is 262,144 |
| C | No per-chunk alternation in **TTScheduler's default branch** | M2 item 1 lived only in an m3native plugin graft; `TTLaneCoordinator` is never built here (run 35707860782) |
| D | The image's route refuses start_pos > 0 without a G1 grant. M1's `prefill_paged_slots_range` is not in the image, and G1 refuses an M1 tree (F5, `qwen_prefix_model_patch.py:140-145`) | the model |
| E | Every call host-snapshots the scratch and writes a GDN slot (154 MB down and back) and may write slot 0 | the per-chunk cost and the decoder displacement |
| F | The drafter window [P-2048, P) straddles two steps when P mod 2048 != 0. The first part's snapshot is an eager buffer allocated after the traces and live across a decode replay | hazard H1 |
| G | The route has never been warmed before the traces in the fast path, and the G1 route fits the page table to the chunk-buffer width (`_qwen_prefix_fit_page_table`), not the width `prefill_warm_before_traces` compiled | a late compile (#48536 hang class) |
| H | The G1 scheduler graft refuses `enable_chunked_prefill` (`qwen_prefix_scheduler_patch.py:336-338`: "start_pos > 0 would also mean continue my own suspended scratch (Lever N)") | combining with prefix reuse later |

---

## 3. Design

### 3.1 Shape of the solution

```
EngineCore step k   : [prefill step: request X, tokens [s, e), e % 2048 == 0 unless final] -> fence
EngineCore step k+1 : [decode-only: both packed blocks, every live seat]                   -> fence
...                   (policy decides how many decode steps per prefill step)
final prefill step  : [last full chunk + tail (window inside) -> seed -> engine build -> seat joins its block]
```

The building blocks are:
- vLLM chunked prefill, which advances `num_computed_tokens`, sets `is_prefill_chunk`, and samples nothing for intermediate rows;
- a scheduler wrapper that caps each prefill step at an aligned boundary and decides decode-versus-prefill by a time-share policy;
- one resumable model route with an explicit resume source;
- the existing multi-segment capture.

**No plugin file is grafted.**
- The scheduler changes are runtime wraps of `TTScheduler` methods: the proven `serving_prefill_admission.install` pattern, whose positive control is the "live in TTScheduler" marker.
- The platform change is a post-import wrap.
- The model and runner edits are one more AST stage in the existing G1 stage table (image build).

### 3.2 Chunk size policy (scheduler side)

The rule for a prefill step of a request with prompt length P and `num_computed_tokens` c. Constants: CH = 2048 (the model's outer chunk), F = floor(P / CH) x CH, tail = P - F.

```
S_last = max(0, F - CH) if tail else P - CH      # the final step starts here
if P - c <= (P - S_last):        budget = P - c          # final step: last full chunk + tail, window inside it
elif no decode running and nothing waiting:
                                 budget = min(SOLO, S_last - c)     # SOLO = k x CH, default 16384
else:                            budget = min(STEP, S_last - c)     # STEP = k x CH, default 2048
```

- **Alignment.** Every non-final step ends on a multiple of 2048, so every continuation starts on one. The wrapper asserts it, the model entry asserts it (G1: `start % chunk_size`), and the capture ledger asserts the cursor.
- **Final-step coalescing** (fixes F and H1).
  - The window [P-2048, P) lies inside [S_last, P), so every drafter feature snapshot is taken in the step that builds the engine, as today.
  - The final step is at most 4,095 tokens: 2048 plus a tail of up to 2047.
  - A prompt shorter than 4,096 tokens runs whole, as today.
- **Step size.**
  - STEP = 2048 is one model chunk: 425 ms at 2k of context, 750 ms at 131k, 1,050 ms at 250k (e, from t(p)). That bounds a decoder's gap inside the prompt.
  - STEP = 4096 halves the transitions and doubles the gap.
  - SOLO steps (no decoder, nothing waiting) cut the overhead when nobody is hurt. They stay bounded (16,384 tokens = 8 chunks, about 3.5-8.5 s by context) so that an arrival, which v1 makes wait anyway and v2 lets preempt at the next boundary, waits at most one solo step.
  - **Smaller than 2048 is refused:** see section 3.4.
- **Mechanism.** vLLM v1's `Scheduler.schedule()` starts from `token_budget = self.max_num_scheduled_tokens`; the attribute name is to be proved on 0.25.1. The wrapper around `_schedule_prefill_only` sets that attribute to `budget` for the call, and restores it in `finally`, as it already does for `max_num_running_reqs`.
  - Only one prefill request is ever scheduled in a step, so the cap binds exactly that request.
  - The argv keeps `max-num-batched-tokens = max-model-len`. With the flag off and chunking on, nothing would split: chunking happens only through the cap.
  - The CPU test must prove the attribute name and the split against the real vLLM 0.25.1 scheduler, as `test_serving_kv_reservation_vllm` does.

### 3.3 Alternation policy (scheduler side)

This is the time-share policy with a floor, in `serving_prefill_admission.wrap_schedule`. It replaces the per-admission credit, and the two are mutually exclusive.

```
on schedule():
  dt = now - last; last = now
  if last_kind == PREFILL and decodes_were_running: owed += dt * (1 - f) / f ; owed = min(owed, KMAX * round_ms)
  if last_kind == DECODE: owed -= dt
  if no decode running: owed = 0
  if prefill_pending and decode running and (owed > 0 or rounds_since_prefill < RMIN):
        result = self._schedule_decode_only(); last_kind = DECODE      # hides partials and waiting
  else: result = original(); last_kind = PREFILL if result schedules prefill tokens else DECODE
```

**Parameters:**

| Name | Default | Meaning |
| --- | --- | --- |
| f (`QWEN_FAST_LEVERN_PREFILL_SHARE`) | 0.5 | the fraction of wall time the in-flight prefill gets while decoders run |
| RMIN | 1 | at least one decode round per prefill step when f < 1 |
| KMAX | 8 | at most this many rounds owed after one step, so a 4 s final step does not owe 16 rounds |

- f = 1.0 means back-to-back chunks: today's stall, chunked.
- **Static mode** for gates: `QWEN_FAST_LEVERN_ROUNDS=R` gives exactly R decode steps per prefill step. This is M2's r. It was proven on TP2 at v67/v118/v128.

**What counts as a prefill step.** Every one: chunks, whole short prompts, and the final step including the engine build. So the 2-4 s admission freeze is also repaid with decode rounds, which subsumes `QWEN_FAST_DECODE_STEPS_PER_ADMISSION`.

**Wall time is measured by the scheduler itself,** between its own `schedule()` calls. The uniproc engine is synchronous, so that interval is the step's execute plus sample plus update: the host and transition costs are charged where they occur. Nothing crosses from the worker.

**Never idle.** With no decoder there is no yield. A yield always schedules the running decodes. `owed` drops to 0 when the last decoder leaves.

**Final-step DRAM gate.** Before a partial's final step (which builds the engine), the wrapper asks the registered DRAM predicate with the backstop's terms (`backstop_need`, `contiguous_need`).
- If short and a decoder runs, it holds: a decode-only step, logged once.
- If no decoder is left to free memory, it lifts.

Without this, a quad or pair capture taken between chunks could leave a fully prefilled 254k prompt to be refused by the backstop after about 92 s of work.

**Which f.** For decoders with short remaining work (agent turns, p50 76-176 tokens), the arrival's TTFT is the same at any f < 1. The decode rounds get spent either way, so f only decides how soon the decoders finish (section 6). With long decoders, f trades directly. Plan:
- start at 0.5;
- sweep 0.33 / 0.5 / R=1 on `agent8_turns` (section 5, G-N5);
- pick the share that minimizes p90 turn latency, with arrival TTFT at most 1.5x the flag-off arm.

The owner's "standard seats first" points to 0.33-0.5.

### 3.4 Exactness

**The claim.** For every request, the KV pages, the final GDN state (the slot the engine adopts), the last-position logits and the drafter window features after a split prefill are **byte-identical** to the whole prefill's. Every later token then equals the flag-off arm's, under the standing packed == solo qualification.

**Why it holds: op-sequence identity.** The whole prefill is already a sequence of 2048-token outer chunks. Each chunk starts from embeddings and reads only:
- **(a)** the scratch: fp32 `rec_state` and the conv carry (the last K-1 = 3 real inputs);
- **(b)** the request's paged K/V for earlier positions;
- **(c)** the host RoPE for its positions. For text this is 1D RoPE computed per position, independent of call length.

The loop calls `synchronize_device` after every chunk. Splitting at those boundaries changes only what runs *between* chunks, so the same programs run on the same bytes in the same order.

**Component by component:**
- **GDN conv:** the same `gdn_prefill_conv_exact` call with the same carry. Its −0 and denormal canonicalisation is identical because it is the same op.
- **GDN scan:** the 128-token internal chunks align with 2048 (16 x 128), and the cross-chunk state never leaves fp32.
- **Attention:** a chunk's queries see the same K/V bytes (whatever the cache dtype), in the same k-chunk order.
- **Already shown:**
  - the M1 equality gate (run 35416319586: 460 / 3,524 / 5,918 tokens identical);
  - v118 (4 x 32k) exact;
  - v128 (4 x 131k) byte-exact with slot moves;
  - the 112-check TT-Sim state-carry gate.

**What is NOT chunk-invariant, and how the design excludes it:**
1. **Boundaries off 2048.** A 1024-token step would run the chunk programs at another M (other matmul program configs and K-blocking, so other rounding) and a padded masked bucket. That is mathematically equal and not proven bit-equal. Refused: STEP and SOLO must be multiples of 2048, the wrapper asserts every non-final end, and the model asserts the start.
2. **A scratch disturbed between steps.** If another prompt's start-0 call re-zeroes it, or a foreign continuation advances it, the result is wrong tokens with no error: the v73-v80 class. Guarded by:
   - the scheduler's one-in-flight rule;
   - a **scratch owner token** at the point of use: the route continues only when `owner == (request_id, start)`, and refuses anything else before any device work;
   - the capture's cursor ledger.
3. **A restored scratch.** This is v2 parking, or a prefix checkpoint. It is exact only as a bit copy of fp32. Device parking uses `ttnn.copy`. The host route uses G1's restore path, whose warm proves a byte round-trip (`restore_mode`).
4. **Logits on an intermediate step.** These are the logits of the chunk's last row (the "exact multiple" branch of the loop). They are never sampled (the runner zeroes them through `intermediate_prefill_mask`), and they mutate no state.

**Verification:**
- the G-N1 digests at boundary lengths (section 5);
- text equality against the flag-off arm and against solo;
- the drafter's window-tail checks (`snapshot_prefill_tail`'s exact compare, on the audited arm).

Acceptance per round is not compared bitwise. bf8 draft proposals are not run-to-run deterministic (v127), so acceptance is reported as statistics.

### 3.5 The model route (AST stage `qwen_levern_model_patch`, after G1's in `qwen_prefix_stage.STAGES`)

**One route, with the resume source explicit.** Under `QWEN_FAST_LEVER_N=1` (or `QWEN_PREFIX_REUSE=1`) the batched TP prefill goes to `_qwen_prefix_prefill_slots`, with `starts`, `req_ids` and the runner's `intermediate` mask. The runner patch forwards `intermediate_prefill_mask` and the ids under either switch.

`_qwen_prefix_row` resolves start_pos as follows:

| Source | When | Device work before the loop |
| --- | --- | --- |
| COLD | start == 0 | reset (do_reset) |
| CHECKPOINT (G1, prefix hit) | start == Q of this request's committed grant, first step only | restore the checkpoint into the scratch |
| **SCRATCH (Lever N)** | start > 0 and `owner == (req_id, start)` | none: the scratch already holds the state at start |
| PARKED (v2) | a park slot holds (req_id, start) | copy park slot to scratch |
| anything else | | **AssertionError before any device work** |

**Per row:**
- Run `prefill_traced_chunked(tokens[:, :e], pt, actual_len=e, start=s)`. For a non-final step e is a multiple of 2048, so the existing entry runs chunks [s/2048, e/2048) with no tail.
- **Intermediate row:**
  - skip the host snapshot and `_write_gdn_slot`;
  - return the chunk-end logits; the last full chunk computes them anyway, and the runner discards them;
  - set `owner = (req_id, e)` and `_qwen_levern_wrote_slot = False`.
- **Final row:**
  - snapshot, write the slot (as today), clear the owner, and set `wrote_slot = True`.
- **Any start-0 row** sets a new owner. A finish or abort clears it through a lifecycle call (`model._qwen_levern_release(req_id)`).
- **Marker inside the method:** `[PINDIAG] lever N route row req= source= start= end= final= wrote_slot= ms= programs=A->B`. This is the positive control that the route executed, not merely that it was installed.

**Warm (fixes G).** `serving_runtime.prefill_warm_before_traces`, under the flag, runs a chunked warm through the route under the bound scratch: request `__levern_warm__`, steps [0, 2048), [2048, 4096) and [4096, 6208) final. This happens before any packed capture.
- Every program and every persistent buffer the route touches then exists before the traces, at the page-table width the route actually uses.
- The tripwire must show zero growth in every later segment.
- The sticky-sessions branch already runs G1's restore warm in the fast path, and v2 reuses it.

### 3.6 The worker lifecycle and the capture

- **Capture.** `capture_factory(P, start=0, prefix_route=True)` under the flag, so the capture wraps the route by name. `check_resumed_call` already demands `starts == cursor` on every route call.
- **Displacement keyed on writes.** After each prefill step, `note_prefill()` runs only if the route reports a slot written to row 0, or reports nothing. An intermediate chunk writes no slot, so the resident decoder is not displaced and no carry restore follows every chunk. v128 displaced on every slot-0 chunk.

  The pre-stage epoch bump (`note_fixture_writer('prefill-chunk')`) stays in v1. It is conservative, and its cost is measured in G-N3.
- **Finish or abort mid-prefill** (client cancel at chunk k):
  - `_release_prefill` closes the capture and frees the gate;
  - the model owner token is cleared;
  - vLLM frees the KV and the reservation returns.

  This is a G-N2 hang shape.
- **Decoders finishing mid-prefill.** `_release_decoders` closes the hook when the last one leaves, and the final step builds a fresh hook. vLLM's batch condense moves the partial's plugin row (v121). That is harmless: the owner is keyed on the request, and the final write lands in the latest row, which `adopt_prefill_slot` adopts.
- **Preemption** of a partial by vLLM is impossible under the KV reservation. That proof must be re-run with chunking on (section 7). If it is ever seen, the request is quarantined and the owner cleared: fail closed.

### 3.7 Packed block membership

Unchanged by construction. The prefilling request holds no pool slot until its final step, so its block simply has one live seat fewer. Padded rounds (`QWEN_FAST_PADDED_BLOCK`) serve 2-3 live seats, and a lone seat goes sequential as today.

At the final step, `from_prefill` acquires a slot through `placement_slot` (lone-user block first, else the fuller one), and the seat joins that block on the next round. Quad re-formation follows the existing ramp rule: the quad forms once the new member's draft history reaches 2048 rows.

What is new is only timing. The membership changes at a step boundary between two decode steps, exactly as an admission does today.

### 3.8 Memory

- **DRAM, v1:**
  - no new device buffers;
  - the scratch (38.5 MB per chip) is already allocated at warm;
  - each step's transients are a subset of today's whole-prompt prefill, using the same chunk programs;
  - intermediate rows skip the `_write_gdn_slot` from_torch temporaries;
  - the window snapshots exist only in the final step, as today;
  - KV is allocated per step but reserved worst-case at admission. The reservation does not change.
- **Host, v1:** nothing persistent. There is no RoPE table for text and no per-chunk 154 MB snapshot.
- **v2 parking:** 38.5 MB per chip per park slot. That is 69 of the 19,968 KV blocks at 557,056 B per block per chip, allocated before the traces. Host parking instead costs 154 MB of host RAM per parked prompt and about 2 x 50-150 ms per switch (e). Default: one device park slot.

### 3.9 Trace-capture and hang hazards

| # | Hazard | Mitigation | Check |
| --- | --- | --- | --- |
| H1 | An eager buffer allocated after the traces that survives a replay (the Metal "unsafe allocation" class). Only the window snapshots would. | final-step coalescing (3.2) | the audited arm with `QWEN_FAST_TRACE_CENSUS=1`, `_GRAPH=1`: no `[PINDIAG] trace overlap` |
| H2 | A program compiled after the traces (#48536: v159/v188 hangs) | route warm (3.5); same chunk programs | `prefill programs A->B window=W`: B−A−W = 0 for every segment; the route marker's programs |
| H3 | Scratch single-occupancy broken; or "bound" left set (an attribute swap with no restore) | owner token; before each decode round, assert every GDN layer's `rec_state` is the batched buffer (a Python identity check) | negative control `QWEN_FAST_LEVERN_FAULT=foreign` must refuse |
| H4 | Eager prefill collectives reusing a CCL semaphore handle that a replayed trace baked (the documented hang mode) | the stock loop fences each chunk; the round fences at its last commit, so both transitions are fenced | hang arms with `QWEN_FAST_CCL_HANDLE_GUARD=1` |
| H5 | Slot 0 overwritten under a resident engine | final-only slot writes; displacement on actual writes | `[PINDIAG] prefill chunk wrote working GDN slot` only at final steps |
| H6 | A stale pre-staged verify | epoch bump per chunk | text equality |
| H7 | Quad, pair or engine captures between chunks consuming DRAM the final build needs | final-step DRAM gate (3.3); the coordinator's `capture_headroom` | the gate line, and no backstop refusal after a long prefill |
| H8 | Watchdogs | stall-watch scopes per segment, now ≤ ~2.1 s rather than ~92 s | no stall-watch exit |
| H9 | Batch condense or slot remap mid-prefill (v121) | owner keyed on request; remap padding fix in the C2 graft (`gdn/tp.py:923-939`) | the slot-move line followed |
| H10 | A partial preempted, or aborted mid-prefill | reservation proof; release path | G-N2 abort shape |

### 3.10 Composition

**Prefix reuse (`tp4/packed-prefix`, sibling design).**
- One route and one source table (3.5) serve both. A hit's first step is CHECKPOINT at Q; its later steps are SCRATCH.
- The G1 graft's chunking refusal (H in section 2) is relaxed only under `QWEN_FAST_LEVER_N=1`, because start_pos > 0 is now disambiguated by grant versus owner.
- `serving_lifecycle._granted_resume`'s "the step carries the whole rest of the prompt" clause becomes "the first step from R, under the 3.2 cap".
- The G1 checkpoint plan (`capture_at`) works per step: a planned boundary inside [s, e) is captured in that step.
- **Branch order:** land Lever N's route-source enum first; packed-prefix adds CHECKPOINT on top of it.

**Stage E (parked engines).** It cuts the final-step build (about 1.2-1.9 s of the about 4 s worst gap).

**Fast lane (`QWEN_FAST_LANE`).** Not on 262k8. Its lane gate must learn prefill-chunk steps before it is combined with Lever N; until then the contract refuses the combination.

### 3.11 v2: more than one prefill in flight (scratch parking and shortest-remaining-first)

**Why.** In v1 a cold 254k prefill (~92-190 s) blocks every other seat's *next turn* behind it, through the one-in-flight rule. With prefix reuse those turns are 1-5 s deltas. In an agent workload that queueing is a larger loss than the decode stall itself.

**Design:**
- **Lifecycle:**
  - singular `request_id`/`capture` becomes a dict of in-flight prefills (at most 1 + park slots);
  - the gate becomes a set;
  - `_is_continuation` is keyed on the scheduled cached request.
- **Scheduler:**
  - each prefill step runs exactly one request (the others are hidden like `_schedule_decode_only` hides partials);
  - candidates are the partials plus at most one fresh admission while a park slot is free;
  - order is shortest remaining first, with aging: a partial's wait is bounded by `QWEN_FAST_LEVERN_MAX_PARK_S`;
  - the KV and DRAM holds apply per admission;
  - the final-step DRAM gate protects a parked long prompt's eventual build.
- **Model:** when a row's request is not the owner and the owner is suspended, the route parks the owner (device copy into a free park slot, owner moved to that slot) before running the row. With no slot free it refuses: the scheduler must not have scheduled the row.
- **Exactness:** a bit copy (3.4 item 3). G-N1 gains "park A, run B whole, restore A, finish A".
- **Expected effect (e):** a 2-10k delta turn arriving during a cold 254k prefill waits at most about 1 s (one chunk) instead of the cold prefill's remainder.

### 3.12 Optional research, after v2: prefill in the round's device-idle time

The device idles about 80 ms of a 245 ms round, while the host commits and drafts. A chunk can be split at *layer* granularity without breaking op identity: the same per-layer programs run in the same order, and the carried activations stay in DRAM. That could fill those windows (about 8 of a chunk's 64 layers per round).

The hazards are single-command-queue ordering, which delays the round's own readbacks, a second host dispatcher, and buffer liveness across replays. Not proposed for v1 or v2.

---

## 4. Flags and profiles

| Flag | Default | Meaning |
| --- | --- | --- |
| `QWEN_FAST_LEVER_N` | 0 | master switch: the platform keeps chunking, the scheduler cap and policy, the route, the warm, the capture over the route. 0 is byte-for-byte today. |
| `QWEN_FAST_LEVERN_STEP_TOKENS` | 2048 | step while decoders run; a multiple of 2048 or refused |
| `QWEN_FAST_LEVERN_SOLO_STEP_TOKENS` | 16384 | step with no decoder and nothing waiting; a multiple of 2048 |
| `QWEN_FAST_LEVERN_PREFILL_SHARE` | 0.5 | f in 3.3, (0, 1] |
| `QWEN_FAST_LEVERN_ROUNDS` | unset | gate only: static R decode steps per prefill step (overrides f) |
| `QWEN_FAST_LEVERN_MAX_ROUNDS` | 8 | KMAX |
| `QWEN_FAST_LEVERN_AUDIT` | 0 | gate only: digests at the end of every prefill (GDN slot bytes per layer, last logits, the request's KV page bytes, window features); reuses G1's `QWEN_PREFIX_DIGESTS`/`_AUDIT` machinery |
| `QWEN_FAST_LEVERN_FAULT` | unset | gate only negative control: `foreign` (a continuation with a wrong owner must refuse), `final-hold` (force the final-step DRAM hold once) |
| `QWEN_FAST_LEVERN_PARK` / `_PARK_SLOTS` / `_ORDER` / `_MAX_PARK_S` | 0 / 1 / srpt / 30 | v2 |

**Contract** (`serving_c2_contract.levern_problems`, enforced at boot). The flag requires:
- `QWEN_FAST_ANY_REQUEST=1`;
- `no-async-scheduling`;
- `enable-chunked-prefill` (in place of `no-enable-chunked-prefill`);
- `max-num-batched-tokens == max-model-len`;
- block size 64;
- text only;
- `QWEN_FAST_DECODE_STEPS_PER_ADMISSION` unset or 0;
- no `QWEN_FAST_LANE`.

Prefix-reuse profiles are refused with the flag until packed-prefix lands its CHECKPOINT handling. Then the boot hook installs the platform wrap (a PostImportHook on `vllm_tt_plugin.platform`, the same mechanism the contract already uses for the input processor), and it logs once:

> `[PINDIAG] lever N: chunked prefill kept for qwen3_5 (budget=262144 threshold=0)`

The gate also reads vLLM's own "Chunked prefill is enabled" line, and requires the absence of "Chunked prefill is not supported".

**Profiles (gate only, `QWEN_FAST_262K_EVIDENCE_WAIVER=1`, every result unqualified):**
- `c2-packed-tp4-8x262k-best-levern-time-gate` = `-8x262k-best-time-gate` plus the flag and f = 0.5 (the timing arm). Its paired control is `-8x262k-best-time-gate`.
- `c2-packed-tp4-8x262k-best-levern-audit` = `-8x262k-best-audit` plus the flag and `QWEN_FAST_LEVERN_AUDIT=1` (the exactness and hang arm). Hang jobs add `QWEN_FAST_CCL_HANDLE_GUARD=1` and the stall watch. One arm adds `QWEN_FAST_TRACE_CENSUS_GRAPH=1`, which has a capture cost and is therefore its own arm.
- **Later, only after G-N1 to G-N5:** a traffic twin `c2-packed-tp4-8x262k-levern` (not gate only), with f chosen by G-N5.

---

## 5. Hardware tests

The tests run in one window on one image. Paired ABAB for any timing claim, judged per round. Nothing in the window decides production.

| Job | Arms | Shape | READ rule (plus `c2_smoke_check` clean, no stall) |
| --- | --- | --- | --- |
| **G-N0** attach | levern-audit | warmup, coding | the platform line, vLLM's "Chunked prefill is enabled", the route warm line before the first packed capture, `programs` growth 0 |
| **G-N1 `levern_equal`** (exactness against non-interleaved) | A = best-audit (flag off), B = levern-audit; solo reference | (i) single user, P in {2047, 2048, 2049, 4095, 4096, 4097, 6143, 6145, 32,785, 131,077, 253,920}, 256 out; (ii) the same prompts while 7 seats decode 4k ignore_eos prompts, so real decode rounds run between B's chunks (pause, decode, resume) | per P: the end-of-prefill digests (GDN slot per layer, logits, KV pages of the prompt, window features) **identical A = B**; texts identical A = B = solo; route lines: one per step, starts = previous ends, every non-final end % 2048 = 0, final wrote_slot=True, intermediate wrote_slot=False; the capture's window checks exact. Any difference is NO-GO. |
| **G-N2 `levern_hang`** (x3 consecutive, H1-H3 style) | levern-audit, `CCL_HANDLE_GUARD=1`, stall watch | (a) a decoder finishes mid-prefill (slot move); (b) every decoder finishes mid-prefill (the hook closes, the final step rebuilds it); (c) the prefilling client cancels at chunk 5; (d) an arrival during an in-flight prefill (queued, then admitted); (e) EOS at the seed after 17 chunks; (f) max_tokens=1 chunked; (g) `FAULT=final-hold`; (h) `FAULT=foreign` (must refuse; a separate container) | every stream completes; texts equal solo; 0 owner refusals except (h), where the refusal must fire; the abort frees gate, owner and reservation (the next arrival exact); no stall-watch or replay-deadline exit; no handle-guard refusal |
| **G-N3 `stall8_cold262k`** (exists: `c2_serving_smoke.py:560`) | ABAB: A best-time-gate, B levern-time-gate (f = 0.5); plus one C arm with `LEVERN_ROUNDS=1` | 7 seats decode 4k (ignore_eos, 6,000 budget); after 12 chunks each, a cold 253,920 arrives | recorded: arrival TTFT; each seat's longest gap after arrival; each seat's tok/s inside the arrival window (**new field**, from the delta stamps); chunk ms from the route lines; transition overhead (B's prefill wall minus the sum of route ms); A gives the never-measured cold baseline (M9a and H4 never ran) |
| **G-N4** existing shapes | the same ABAB | concurrent8_code_32k, concurrent8_code_128k | the TTFT staircase, live4_min, decode_agg, wall_tok_s; expected: min seat rate up, TTFT max up |
| **G-N5 `agent8_turns`** (new; chooses f) | B with f in {0.33, 0.5}, R=1, and A | 8 seats replay tau-lab agent tapes (thinking on, coding): 5 turns per seat with growing context 16-140k, outputs from the tapes (p50 76-176), tool gaps 2-8 s. Without prefix reuse each turn re-prefills. | turn latency p50/p90, TTFT p50/p90, per-seat client tok/s from the first token, device busy share; choose f |
| **G-N6** churn | levern-time-gate | churn16 (C9 lengths) | zero aborts, refusals or engine deaths; reserved + running ≤ pool on every line; no DRAM hold with a seat free; no backstop refusal after a long prefill |
| **v2 later** | +PARK | G-N1 (ii) with a 4k arrival preempting a 131k prefill; `burst8_mixed` (8 arrivals together, 4k-128k mixed) | park/restore digests equal; the short arrival's TTFT ≤ its solo TTFT + 1.5 s |

**New smoke-check rules** (`c2_smoke_check`, under the flag):
- route lines per chunked prompt = ceil(steps);
- alignment;
- final-only slot writes;
- no `programs` growth;
- the alternation line present whenever decoders and a partial coexist (a marker inside the policy, not at install).

**Window order:** X0, B0, G-N0, G-N1, G-N2 (x3), G-N3 ABAB, G-N4 ABAB, G-N5, G-N6, Z. About 12-14 h (e). The NEEDS lines gate G-N3 onward on G-N1 and G-N2.

---

## 6. Estimates

**Model:** `levern_model.py` in this directory.
- Chunk time is t(p) = 420 ms + 2.5 ms per 1k tokens of context (cm: fitted to T3/T4).
- Build 1.9 s; 8-live round 245 ms; τ 4.4; transition 40 ms per step. Each of these is e.

| Case | Today | Lever N, R=1 | f = 0.5 | f = 0.33 |
| --- | ---: | ---: | ---: | ---: |
| Cold 253,920, 7 long decoders: arrival TTFT | 93.6 s | 128.5 s (+37%) | 192 s (+105%) | 291 s (+210%) |
| same: seat rate inside the window (steady ~18) | 0 tok/s | 4.2 | 8.7 | 11.9 |
| same: worst seat gap | ~94 s | 4.0 s once (final step 2.1 s + build), ≤ 1.1 s otherwise | same | same |
| Cold 253,920, 7 agent decoders with 176 tok left: arrival TTFT / decoders done | 93.6 / 103.4 s | 108.4 / 32.4 s | 108.4 / 20.1 s | 108.4 / 14.7 s |
| same, 543 tok left (p90) | 93.6 / 123.8 s | 128.5 / 128.7 s | 128.7 / 60.4 s | 128.7 / 45.1 s |
| One 65k prompt beside long decoders: TTFT / seat rate | 17.9 s / 0 | 26.5 s / 5.0 | 34.1 s / 7.9 | – |
| One 120k prompt: TTFT / seat rate | 35.3 s / 0 | 51.6 s / 4.9 | 72.4 s / 8.6 | – |

**Per-step times** (e): a 2048 step costs 425 ms at 2k of context, 580 ms at 64k, 750 ms at 131k, and 1,050 ms at 250k. The final step (chunk plus tail) costs up to 2.1 s, plus the build.

**Bursts.** With 8 x 30k together, user 8's TTFT goes from about 72 s (m, T3) to about 95 s at R=1 (e). The staircase is prefill throughput, and Lever N cannot remove it. v2's shortest-first ordering helps only mixed lengths.

**The real agent ceiling.** At about 3.6k prefill tok/s, 8 seats re-prefilling about 65k per turn need about 18 s each per turn: the device is prefill-bound. Prefix reuse (deltas of a few thousand tokens per turn) is the lever there. With it, Lever N's value becomes the bounded gap for cold or compaction turns, plus v2's preemption.

---

## 7. Staged build plan

**Branch and delivery.**
- Branch `tp4/lever-n` from `origin/tp4/262k8`.
- Overlay modules go in `docker/qwen-c2-overlay.txt`: only there, never in the Dockerfile copy list.
- Model and runner edits are AST stages in `qwen_prefix_stage.STAGES`. Both routes ship only by **image build**: the evidence tree is baked, not grafted.
- New tests are allowlisted in `qwen-integration-cpu.yml`. None are named `test_serving_*`: those get copied into the P8 build without their fakes.
- Run the CPU suite in CI (`gh workflow run qwen-cpu-suite.yml -f ref=tp4/lever-n`), not on the laptop.

| Stage | Content | Files | Tests | e |
| --- | --- | --- | --- | --- |
| **S1. Policy core (pure)** | the 3.2 cap rule with coalescing, solo and step sizes; the 3.3 time-share and static-R state machine; flag parsing (refuse malformed) | new `scripts/ci/levern_policy.py` (stdlib) | `test_levern_policy.py`: P in {1, 2047, 2048, 2049, 4095-4097, 6143-6145, 253,920}; every non-final end aligned; window inside the final step; owed and KMAX; no yield without decoders; final step charged | 1.5 d |
| **S2. Scheduler and platform** | the cap applied in `serving_prefill_admission.wrap` (`max_num_scheduled_tokens` around `original`); the alternation in `wrap_schedule`, mutually exclusive with the credit; the final-step DRAM gate; markers; platform keep-chunking wrap; contract `levern_problems`; profiles | `serving_prefill_admission.py`, new `levern_platform.py`, `serving_c2_contract.py`, `qwen_c2_profiles.json` | `test_levern_scheduler_vllm.py` against the **real vLLM 0.25.1 scheduler**: chunking on, the cap splits only at aligned ends, one prefill request per step, decode steps hide the partial, **the KV reservation bound re-proved with chunking** (no preemption, allocations within the reservation, negative control); a platform-wrap test executing the real policy fixture; contract tests (flag off gives an identical argv) | 2.5-3 d |
| **S3. Model route and runner** | `qwen_levern_model_patch` stage pinned to G1's PATCHED_SHA256: the vLLM entry switch, the SCRATCH source, the owner token, intermediate rows skipping snapshot and slot write, the route marker, `_qwen_levern_release`, `_qwen_levern_warm`; the runner patch forwarding ids and the intermediate mask under either switch | new `qwen_levern_model_patch.py`, `qwen_prefix_runner_patch.py`, `qwen_prefix_stage.py` | executes both files against the recording fake ttnn (the `test_qwen_prefix_model_runtime` pattern): COLD/SCRATCH/refusals; a foreign owner refused before any device call; intermediate makes no `_write_gdn_slot` or `to_torch` of state; final identical to the stock path's calls; switch off keeps the stock bytes; stage pins | 3-4 d |
| **S4. Lifecycle, capture, warm** | capture over the route under the flag; displacement on actual writes; owner release on finish and abort; route warm in `prefill_warm_before_traces`; smoke-check rules | `serving_lifecycle.py`, `dflash_prefill_window.py` (`segment_wrote_slot`), `serving_runtime.py`, `c2_smoke_check.py` | lifecycle: chunked admission, continuation, abort mid-prefill, decoders finishing mid-prefill, EOS at seed, D2 quarantine at final; capture: the route ledger across segments and coalescing | 1.5-2 d |
| **S5. Gate tooling** | `levern_equal`, `levern_hang`, `agent8_turns`, the seat-in-window rate in `stall8_cold262k`; job templates `scripts/ci/references/tp4-levern-jobs/` + ORDER.txt | `c2_serving_smoke.py`, `c2_serving_gate.py` (digests read), references | template parse tests, smoke-check fixtures | 1.5 d |
| **S6. Card window** | B0 image `tp4-levern-1`; G-N0 to G-N6 (section 5) | – | – | ~12-14 h of cards |
| **S7. v2** | parking (device slot), multi-prefill lifecycle, shortest-first scheduler, PARKED source | the above | v2 CPU tests; G-N1 (ii) park; `burst8_mixed` | 6-10 d + 0.5 window |

**Review gates.** One adversarial review after S3, aimed at the exactness and owner token, and one after S5, aimed at the read rules and the absence of NOT_EXERCISED paths. Every hardware run has a positive control *inside* each new method: the route marker, the alternation marker, the platform line.

## 8. Risks and open points

1. **The TP4 numbers are fitted, not measured,** for the 254k cold prompt and the transition cost. G-N3's A arm is also the first cold-262k measurement.
2. **TTFT goes up by design.** If the owner rates arrival TTFT above decoder gaps, f near 1 with R=1 is the conservative setting, and G-N5 decides.
3. **Late compile through the route's page-table fit** (gap G). This is the most likely first-run failure; the warm and the tripwire are the guard.
4. **The vLLM 0.25.1 internals** (`max_num_scheduled_tokens`, the chunked path with DFlash lookahead) must be proven on CPU first. They were proven on TP2 hardware with an older image.
5. **Packed-prefix coordination.** Both branches edit `qwen_prefix_*` and the lifecycle, so the route-source enum lands first (3.10).
6. **Lever N alone does not meet the target.** It neither raises tok/s nor fixes per-turn re-prefill. It is worth building because once prefix reuse makes most turns short, v2's preemption keeps every seat's turn latency bounded while cold prefills run.
