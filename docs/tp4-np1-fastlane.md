# tp4/np1-fastlane: the lone-user padded round (NP1)

Status: built and tested on CPU; nothing here has run on a card. The flag is `QWEN_FAST_SOLO_PACKED=1`, **gate only, default off**: with it unset every path below is byte for byte what it was (the flag-off
tests compare against the commit this branch was cut from). The approved ship-4 profile is untouched; the flag lives in two gate-only twins of it (`make_np1_profiles.py`). The window pack is
`scripts/ci/references/fusion-jobs/NP1` (`make_np1_jobs.py`); no tag is pushed from this branch.

Owner rule: **strict exactness**. Greedy output stays byte-identical to the target's own greedy decode; speculation changes how many tokens a round commits, never which.

## 1. The problem, measured on the ship-4 production logs

| live seats | where the round runs | verify rows | step | host gap | tokens committed per round | rate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | the user's own 4-row engine (`engine.verify`) | 4 (3 drafted) | 83.9 ms | 15.3 ms | 3.70 | ~37 tok/s |
| 2 | the 64-row M3 block, padded (2 idle segments) | 16 rows a user | 52.8 ms | 19.6 ms | 13.56 for the pair (6.78 a user) | ~94 tok/s a user |

The packed pass is faster than the per-request one because it carries the fusion programme (page-parallel K/V writer, glue levers, W2, the fused commit); the rows are nearly free, the pass is what costs
(`docs/tp4-octo.md` section 1: 49.1 ms at 2 seats, 49.7 at 4). A lone user would run the same trace as a pair. The target's fast lane (8 x 75 + 1 x 150) needs this first.

## 2. Why live = 1 takes the per-user path (file:line)

1. `serving_worker_hook.py:571` returns the bridge's own `drafts()` only when there is no packed step; production has one, so the round goes through `block_groups` (`:623-625`, `:775-787`) to
   `PackedStep.proposal_groups` (`serving_packed_step.py:722-732`) -> `proposal_groups` (`:357-393`) -> `block_rows_for` (`:396-415`).
2. `block_rows_for` is the decision: `padded = len(group) != shape.users and pads(block, len(group))` and `if len(group) != shape.users and not padded: return None` (`:402-404`). One live request in a
   4-user block is `len(group) == 1`, `pads(block, 1)` is False (`:418-423` -> `packed_verifier.py:1760`: `padded_min_users <= count < users`), so the group's width is `None`.
3. `None` means each member drafts at its engine's own width: `serving_fast_request.py:62` -> `verifier_engine.py:357-373` (`proposal_rows(packed_rows=None)` = the widest capture, 4 rows) and the round
   is served by `sequential_packed_step` (`serving_packed_step.py:1091-1125`, `ineligible` -> `:863-865`; `serving_sequential_step.py:62`).
4. `padded_min_users` is `QWEN_FAST_PADDED_BLOCK_MIN_USERS`, default **2** (`packed_verifier.py:624`, `serving_runtime.py:174-192`, `:451`), because **the block refuses a smaller one at construction**
   (`packed_verifier.py:876-885`: `max(1, users - idle_capacity) <= padded_min_users`) and `MAX_IDLE_SEGMENTS = 2` (`:1847`). The image bakes `QWEN_FAST_PADDED_BLOCK=1` (`docker/qwen-c2-serving.Dockerfile:146`),
   so two and three live users in one block already pad.
5. The octo block (`serving_octo.py:52` `MIN_LIVE_DEFAULT = 6`, `serving_packed_step.py:653-666`) takes six or more live seats; the ship profile sets `QWEN_FAST_M3_BLOCKS=2`, `QWEN_FAST_OCTO=live`,
   `QWEN_FAST_OCTO_MIN_LIVE=6`. New arrivals fill a block with exactly one live user first, then the fuller block (`serving_buffer_pool.py:917-938`), so one to three users sit in block A and a lone user is the
   usual live=1 state; a lone user in block B appears after departures (and at 5 live: block A full, one in B).
6. The flag name already existed (`serving_octo.py:328-366`, `docs/tp4-octo.md` section 6) and **refused to engage**: "a lone user leaves three idle segments and page 0 holds two".

## 3. What the packed block needs to serve one user

* **Capture widths.** Nothing new: the block's verify trace is captured once at 4 segments x 16 rows (`packed_shapes.m3_shape`); a round with fewer live users replays the same trace with the other
  segments staged idle (`packed_verifier.py` module docstring, VARIABLE-USER ROUNDS; `gdn_user_batch.py:50-70`). The commit traces are per (segment, prefix 1..16); an idle segment decides at prefix 0 and writes no state.
* **Idle rows.** `idle_inputs` (`:1849-1888`): tokens `(1,)*16`, start 0 or 32 (the extent block: family E = 256), through an all-zero page table: K/V lands on physical page 0 (vLLM's null block), on tile row
  `j % 2` for the j-th idle segment. The idle carries are read in place (`carries_in_place`, `padded_refusal` `:1805-1845`), so no idle slot's state moves.
* **Page tables.** Live tables must not name page 0 inside the range the round reads or writes (`padded_refusal`, `page_zero_index` `:815-837`); the extent attention reads pages `[0, E // 64)` of every segment.
* **The (page, tile row) rule.** The K/V writers are exact only while no two segments write one (page, tile row) (`verify_trace_t2.py:203-221` `kv_conflict`; the page-parallel writer's header,
  `kv_page_writer_tp4.py` EXACTNESS). Page 0 has two 32-row tile rows, so two idle segments fit and a third does not. **This is the whole obstacle**: a lone user is three idle segments.
* **The drafter's row count.** 16 (seed + 15): `bridge.drafts(packed_rows=16)` (`serving_worker_hook.py:889-891`) -> `session.propose(max_rows=16)`. The draft pass is the request's own single-user capture whatever the
  verify width (the coordinator's `prepare_single`, `dflash_packed_proposal_coordinator.py:1010-1021`): it already runs for any unpaired seat of a padded round. Its cost does not change.
* **The attention extents.** The idle segments are ordinary users at start 0 / 32 in family E = 256 (K64j flags 0x23, two eight-row groups per bundle); the live segment is at its own family. No new geometry,
  no new program: the lone round is an existing replay with different staged values.

## 4. The answer: a third (page, tile row) target that nothing else owns

Options, and why one:

| option | verdict |
| --- | --- |
| lower `padded_min_users` to 1 | refused at construction (`packed_verifier.py:876-885`): the same obstacle |
| let a third idle segment share a page-0 tile row | breaks the writers' exactness precondition (the read-modify-write of one tile row by two units races), and the page-parallel writer's audit (`QWEN_FAST_KV_PAGE_WRITER_AUDIT`, real cache against a served-writer shadow) would be expected to report the race as a mismatch |
| D0 solo lane (`QWEN_FAST_SOLO_LANE`, `serving_solo_lane.py`) | one extra 16-row block bound to pool slot 0 only, refused beside two M3 blocks, and a block's worth of DRAM per slot to cover every lone-user slot |
| octo block for a lone user | 7 idle segments |
| a free vLLM block as the sink | vLLM owns every id below its pool size and keeps free blocks' content for its prefix cache (the profile has `enable-prefix-caching`): scribbling in one is a correctness bug for someone else |
| **one KV block past vLLM's pool** | chosen |

The TT worker allocates the cache tensors from the shape vLLM gives it (`scripts/ci/fixtures/vllm_tt_plugin_bf77cd63/model_runner.py:423-453` -> `Qwen36ForCausalLM.allocate_kv_cache(shape, dtype, layers)`,
`scripts/ci/fixtures/qwen36_vllm.py:146`; the scheduler plans its pool from the plugin's block count in the engine process, `worker.py:479-577`: the worker's tensors and the scheduler's pool are two things that are equal only because the plugin makes them so). `idle_scratch.py` makes that shape **one block longer**
on the worker: block id N (vLLM's pool size) is never handed to a request, is zero-filled like the rest (`qwen36_model.py:2740` `_allocate_kv_caches_tp`, or the device-zeros fill), and is written only by idle segments.

* `idle_scratch.install` (a `serving_c2_contract.PostImportHook` on the model entry, armed at boot by the flag and by nothing else, `serving_c2_contract.py:1691-1704`, `:1775-1776`) wraps
  `allocate_kv_cache` and records `(N, 1)`. It chains beside the prefix profiles' own hook on the same module (tested).
* `ServingCacheOwner` (`serving_cache_owner.py:14-25`) claims the page at the attach (`idle_scratch.claim`: the cache must be exactly the pool plus the recorded pages, else the attach fails by name) and keeps
  `physical_pages` at the POOL, so the bridge factory's own bound check (`serving_runtime.py:1030`, "Unique physical pages from the admitted cache required") refuses any request block id past the pool.
* `serving_octo.solo_packed_admission` counts the page (`idle_capacity(len(pages))` = 2 + 2 per page) and admits; `serving_runtime.py:495-502,850` hands `idle_scratch_pages` to the two M3 blocks (not the octo
  block, which pads from 6 live seats), with `padded_min_users = 1`.
* `PackedVerifierEngine(idle_scratch_pages=...)` (`packed_verifier.py:872-885`): `idle_capacity = 2 + 2 * pages`; `idle_inputs` sends idle segments 0 and 1 to page 0's tile rows 0 and 1 **as before** and
  segment 2 to tile row 0 of the scratch page through a table that holds that page in every entry (the extent reader reads `[0, E // 64)` of it); `padded_refusal` treats a scratch id in a live table as it treats
  page 0 (`page0 scratch page ...`, same marker). The T2 guard, the step's `ineligible` and `kv_guard` read the same fills and see three distinct cells.

No kernel, program, trace or captured address changes. The block, the readers and the writers take the scratch table as a host value, the way they take the zero table.

## 5. Memory cost

One KV block is 16 attention layers x K and V x 64 x 256 bfloat8_b = **557,056 bytes per chip** (the figure `docs/tp4-octo.md` section 4 uses for the pool); the pool itself (16,500 blocks,
`QWEN36_MAX_TOKENS_ALL_USERS` 1,055,488) is untouched, so the reservation admission, the host tier and the prefix store see the same pool. The ship-4 plan's margin (the third block's memory plan) is
unaffected at this order; A1 records the free DRAM per chip after the attach.

## 6. Exactness argument, and what stays a card question

Why a lone user's accepted tokens equal its per-request ones: the round commits only the target's own argmax at each row of the live segment; speculation changes how many accepted rows a round keeps.
The live segment's computation is the one a pair or a triple already runs (the packed trace is row-independent in every matmul, norm, collective and sampler; attention, K/V write and GDN recurrence
are per segment; carries are read in place and idle segments commit prefix 0). What the flag changes is **only where the idle segments' own K/V lands**, and that is host data.

What the CPU tests establish (`test_np1_fastlane.py`, against the real `PackedVerifierEngine` over the fake two-chip device, the real extent readers and the real T2 guard):
* at most two idle segments the staged inputs equal the flag-off block's, byte for byte, for every live pattern; the parent commit's `idle_inputs`, `padded_refusal`, `describe` and admission line are the
  same functions of the same inputs;
* a live segment's staged rows do not depend on how many segments are idle or where they write;
* the three idle segments of a lone user write three distinct (page, tile row) cells, `verify_trace_t2.kv_conflict` passes, and an emulated paged cache written by the staged rows shows idle writes on page 0
  and the scratch page only, never on a page a live row reads or writes (randomised over live patterns, positions and tables);
* a live table holding the scratch id or page 0 is refused before anything is staged; the routing at 1, 2, 3, 4, 5, 6 and 8 live seats (below).

What only a card settles (A1, then the ABAB):
* **K1** a block id past the pool in a page table: the K/V writers and the extent reader have never been given one. The cache tensor really is that long (the allocation is ours), the id is below its
  extent, and the audits (page writer, T2, pre-stage, multi-user SDPA) compare the written bytes; this is the first card to do it.
* **K2** the lone user's text equals its per-request text (E1/E2 style: `octo_compare.py` over the ABAB logs; `concurrent5_split`/`concurrent8_drain` under the audits).
* **K3** the 16-row trace at one live user costs what it costs at two (the arithmetic says 52.8 ms; the idle rows are free).
* **K4** the allocation hook runs in the worker process before the plugin's runner allocates (PostImportHook on the model entry, as the prefix profiles' own): if it does not, the attach refuses by name
  (`claim`), it does not serve a lone user on a block someone owns.

## 7. Routing at each seat count (flag on; `test_np1_fastlane.RoutingTests`, real blocks)

| live seats (slots) | groups |
| --- | --- |
| 1 (any slot) | its block at 16 rows, padded (**changed**; was the engine's 4 rows) |
| 2, 3 in one block | that block at 16 rows, padded (unchanged) |
| 4 in one block | that block at 16 rows, full (unchanged) |
| 1 + 1 (one per block) | both blocks at 16 rows, two passes (**changed**; was two engine passes) |
| 4 + 1 | block A full, the lone user in B at 16 rows (**changed** for the lone user) |
| 6 to 8 | the octo block, 8 rows (unchanged) |

The flag changes exactly the blocks with one live seat.

## 8. Expected gain, and its basis

A lone user commits what a pair-member commits per round of the same trace: 6.78 tokens (the measured 13.56 for two) in about 52.8 + ~18 ms (the host gap is 19.6 ms at two seats and 15.3 at one; call it
17-20) = ~71 ms: **~95 tok/s against ~37, x2.1 to x2.9** (tokens per round 5.5 to 7.5 and the gap bracket it: 79 to 107). The basis is the measured live=2 round, not a measurement of this change; the lone
user's acceptance at 15 drafted rows is the largest unknown (at 3 drafted rows it takes 3.70 of 4). At 4 live seats nothing changes; at 5 the lone user in the other block goes from the engine's ~22 tok/s
share to ~48 (two block passes a round). It does not reach the 150 tok/s fast lane (that needs a shorter round or a deeper accepted prefix), but it is the step the lane stands on.

## 9. Flag, profiles, ship path

* `QWEN_FAST_SOLO_PACKED` is strict `0|1` (`serving_octo.solo_packed_requested`) and a gate instrument: `serving_c2_contract.OCTO_GATE_ONLY` refuses it on a traffic profile and removes an inherited value.
  Admission also needs `QWEN_FAST_PACKED_STEP=1`, `QWEN_FAST_TP=4`, `QWEN_FAST_EXTENT_REPLAY=1`, `QWEN_FAST_PADDED_BLOCK=1`, `QWEN_FAST_SINGLE_GATEUP=1`, no `QWEN_FAST_SOLO_LANE`, and a gate run.
* `c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-s4-np1` (timed arm) and `...-s4-np1-audit` (audited attach) are generated by `make_np1_profiles.py` from the approved `-fx-traffic-s4` and its audited twin `-fx-s4-audit`
  plus exactly the flag (and, for the timed arm, the gate marker in place of the waiver id, as `-s4-nogrid13` does).
* A ship profile needs an owner traffic waiver naming the flag (the way octo-T8 is waived) and a contract change admitting it in a waiver run; this branch does neither. The decision is the owner's after A1 and the ABAB. (tp4/next-1 made the contract change: `serving_c2_contract.NEXT_WAIVABLE_LEVERS`, and `serving_octo.solo_packed_admission` admits a waiver run beside a gate run; docs/tp4-next1.md.)

## 10. The window (`references/fusion-jobs/NP1`, templates only)

B0 build `tp4-np1-1` (a placeholder), X0, **A1** audited attach on the audit twin (warmup, coding, np1_live2, concurrent5_split, concurrent8_drain, concurrent8_steady), **C1 L1 C2 L2** the timed ABAB at one,
two and four live seats with eight beside them (`np1_live1`, `np1_live2`, `np1_live4`, `concurrent8_steady`; production profile against its `-s4-np1` twin, both on the Swift checkpoint production serves), Z.
`np1_report.py` reads the ABAB (GAIN at one seat, no LOSS elsewhere), `octo_compare.py` the texts, `octo_judge.py` the solo rule (a padded round of a lone live user ran; no eligible lone user went to the engines).

## 11. Risks and open questions

* K1 to K4 above. K1 is the real one; the attach and the audits are built to fail loudly on it, not to serve around it.
* The scratch id is the pool size N, taken from the shape the plugin passes; if a future plugin changes how the runner calls the allocation (the per-layer API), `install` raises at the model import and the
  source pins in `test_np1_fastlane` fail first.
* `QWEN_FAST_OCTO_MIN_LIVE` stays 6: the octo block's capacity is still two idle segments. Giving it scratch pages (two more targets a page) would let it pad from fewer seats, but that is a separate decision
  with its own device question (eight-row segments on a scratch table) and is not made here.
* The lone-user round holds the device ~71 ms where the engine held it ~99 ms: Lever N's solo step size (`QWEN_FAST_LEVERN_SOLO_STEP_TOKENS`) was set against the engine's round, not re-tuned.
* Two single-seat blocks run two sequential padded passes a round; a rebalance (migrate the lone user of B into A) would halve that but needs a slot rebind. Not built.
