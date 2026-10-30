# tp4/s7, tp4/s7b and tp4/s7c: the ship-7 candidate

**tp4/s7c (2026-10-22) is tp4/s7b without admission pipelining (`QWEN_FAST_ADMISSION_SEED_FIRST`).** Two findings, in this order.

1. *The S7B audited attach (card run 39649281798, job S7B-A1) died at engine start* with `prefix reuse refused: ... Lever N's schedule wrapper installed on the class before the graft`. `SeedFirst.install` wrapped the scheduler class's `schedule()` with a bare function, which hid Lever N's mark (`_qwen_levern_schedule`) from the prefix graft's install check. That is fixed in `admission_pipeline.py` (`functools.update_wrapper` carries the inner wrappers' marks onto the new wrapper; `test_levern_prefix_graft` and `test_sched2_pipeline` pin it) and the fix stays.
2. *A review then found that the lever's design cannot work with vLLM's engine loop.* In vLLM 0.25.1 `EngineCore._process_engine_step` (`vllm/v1/engine/core.py`, lines 510-517 and 1300-1309 of the image's copy) runs `step()` (schedule, execute, `update_from_output`), queues the outputs, then `post_step()` calls `take_draft_token_ids()`, and only after that comes the next `schedule()`. Seed first completes the deferred bind at the START of the next `schedule()`, so `take_draft_token_ids()` has already run without the new request: no draft rows and no prepared request ticket exist for it, and the next packed step raises `ValueError: Live prepared request ticket required before scheduler admission` (`serving_vllm_contract.prepared_ticket`, reached through `serving_vllm_packed.ordered_tickets`). That would happen on the first real request after boot and on every rebound turn after it. The CPU suite could not see it: `test_sched2_pipeline`'s script calls `schedule()` before `take_draft_token_ids()`, the reverse of vLLM's order. `SeedFirstLifecycleTests` now drives vLLM's order: the eager bind has its drafts at the post-step, a deferred request has none (a test of the documented limitation), and `test_in_vllms_order_seed_first_has_the_eager_orders_drafts_at_the_post_step` is an `expectedFailure` that a redesign of the lever must turn green.

tp4/s7c therefore ships **without seed first**: `QWEN_FAST_ADMISSION_SEED_FIRST` is in neither profile (the candidate `...-fx-traffic-s7` and its audited twin `...-fx-s7-audit`), not in the waiver's `levers`, and item (T) of the waiver is now a note that the lever is NOT IN THIS PROFILE and NOT WAIVED. `make_s7_profiles.py` refuses a parent profile that names it, as it refuses the traced-tail knobs. Every other s7 lever stays, the memory-ledger trim (`QWEN_FAST_ADMISSION_LEDGER_TRIM`) included: nothing in the contract or in `admission_pipeline.problems()` ties the trim to seed first (only seed first needs engine reuse, the lean admission and Lever N), and item (U) does not depend on it. The decision sentence of the waiver names no lever, so it is word for word the one the owner approved (`make_fusion_profiles.py --approve-ship` was not needed); the profile it approves is now the narrower one. The lever's code stays in the tree, default off, in no profile of this chain (the gate-only `-sch-*` arms of tp4/sched-2 still carry it; the contract still lets a waiver name it, and no waiver does). The pack lost its seed-first read rule, and the estimate that went with the lever (a returning turn's first token p50 1.46 s to about 1.19 s) is no longer a claim of the candidate.

**tp4/s7b (2026-10-22) is tp4/s7 without the traced masked tails.** The first S7 window's two audited attaches (`-s7-audit`, `-s7-audit-mid`) both died at engine start with `TT_FATAL: Reads are not supported during trace capture` (`fd_mesh_command_queue.cpp`, `!trace_id_.has_value()`), logged as `[PINDIAG] attach failed with RuntimeError`. The read was the oneq prefill-SDPA audit (`QWEN_FAST_SDPA_PF_ONEQ_AUDIT`, `sdpa_pf_oneq_tp.py`): its wrapper reruns scheduled calls (the first 8 engaged calls, then every 24th) with the chain program and reads both outputs back, and it has no capture check, so the one scheduled call (call index 200) that fell inside the first capture of a 512- or 1,024-row bucket of the traced tail (PF3-T, `QWEN_FAST_PREFILL_TAIL_TRACE=1`) read from the device inside the capture. The tp4/prefill-3 audited twins carry the same flag pair and had never run on a card. The unaudited candidate does not install that wrapper, but PF3-T has still never completed a capture of an engaged bucket on a card. tp4/s7b therefore takes the lever out of the candidate and out of its audited twin (`QWEN_FAST_PREFILL_TAIL_TRACE` and its three knobs are in neither profile), retires the second audited twin and its job (A2: it existed only for the bucket split), and narrows the waiver to items (S) to (V) and the decision sentence to the one audited attach. Everything else is unchanged; the lever's code stays in the tree, default off, in no profile of this chain (the `-pf3-s5-audit*` twins of tp4/prefill-3 still carry the pair that crashes: do not run them without a capture guard in `sdpa_pf_oneq_tp.py`). Sections 1 to 3 below describe the s7c state (s7b with seed first out); the rest of this file is the tp4/s7 text.

Status: **CANDIDATE, never run on a card.** Built on `tp4/s6` (the ship-6 candidate, `docs/tp4-next2.md`, which passed its card gates on 2026-10-21) with four already-built, CPU-tested branches merged in (tp4/s7b ships three of the four, tp4/s7c only the ledger trim of tp4/sched-2), each lever behind its own
switch so that a failing one comes out by deleting one line. Every effect below is an **ESTIMATE**, none is a measurement. The owner approved the ship on 2026-10-21 **conditionally on the S7 window gates** ("Yes approve s7 if it
passes"); the decision sentence in `scripts/ci/fusion-wp/S7.json` records the conditions, and production switches to the image only after the gates are read.

## 1. What is in it

`...-fx-traffic-s7` is the ship-6 profile (`...-fx-traffic-s6`, unchanged, its waiver untouched) plus exactly these lines (and the waiver id, the profile's own name for its waiver; the admission-pipelining row of tp4/s7 and tp4/s7b is gone in tp4/s7c):

| lever (branch) | line(s) | what it does | ESTIMATE |
| --- | --- | --- | --- |
| threaded host-tier upload (`tp4/upload-1`, `docs/tp4-upload1.md`) | `QWEN_PREFIX_TIER_IO_THREADS=4`, `QWEN_PREFIX_TIER_VERIFY_THREADS=4` | the staging copies of the tier's raw block IO run on workers beside the device calls, the sampled digest checks on workers; a failure latches it off and repeats the call serially | a 254k restore of about 1 s falls to 0.4-0.6 s: **under 0.5 s of a 5.4-6.5 s returning first token** (the brief's "8 s to 0.4 s" divided by a conversion rate that the preconverted checkpoints of ship 5 already removed) |
| memory-ledger trim (`tp4/sched-2`) | `QWEN_FAST_ADMISSION_LEDGER_TRIM=1` | the memory ledger ends after the attach (no per-rebind bracket) | about 1.7 s of a 15.4 s burst into a fresh process, nothing warm |
| prompt lookup on the engine-reuse path (`tp4/stream-1`, `docs/tp4-stream-1.md`) | `QWEN_FAST_LOOKUP_DRAFT=n3m12` | a host-side lookup over the user's own tokens replaces the drafter's proposal rows where it matches; the target verifies every row | about +3 percent tokens a second, +8 percent on edits (short-prompt simulation) |

**Left out, default off, in no profile:** admission pipelining (`tp4/sched-2`, `QWEN_FAST_ADMISSION_SEED_FIRST`: its design cannot work with vLLM's engine loop, see the top of this file), the traced masked tails (`tp4/prefill-3`, `QWEN_FAST_PREFILL_TAIL_TRACE`: tp4/s7's audited attaches died on a device read inside the trace capture, see the top of this file), the hold reuse (`QWEN_FAST_ADMISSION_HOLD_REUSE`: its own doc wants a ballast-boundary run before any ship; the contract keeps it a gate's own), the K-split of the tail passes
(`QWEN_FAST_SDPA_PF_KSPLIT`, a tolerance-class numerics change that needs **its own owner waiver**; its code is in the tree because it came in the same branch, behind the flag and a contract rule that refuses it on a traffic profile
unless a waiver names it), the two-candidate tree verify (`tree_verify.py` is imported by nothing in serving) and the lookup shadow (`QWEN_FAST_LOOKUP_SHADOW`).

**The lookup's index cap.** `prompt_lookup.MAX_INDEXED_PROMPT = 32768` indexes only the last 32,768 prompt tokens. A later simulation (`docs/tp4-drafter-long.md` on `tp4/drafter-long`) finds the cap removes the lookup's
gain beyond 32k (n3m12 as built x1.014 [0.999, 1.041] against x1.046 [1.012, 1.089] with the whole prompt; at 128k x0.999 against x1.066). The cap is a module constant, not an environment setting, and the index is built on the
engine step thread at every rebind (about 0.07 s at the cap, 0.27 s for 123k tokens, 0.6 s for 254k, laptop figures, stalling every decoding seat), so raising it is a code change with a host-time cost that nothing has measured
on the rig. It is left as built here and is the follow-up if the lookup passes its gates.

## 2. The profiles and the pack

* `c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s7` (`make_fusion_profiles.py` from `scripts/ci/fusion-wp/S7.json`): the candidate. **Waiver `ship7-fx-262k-2026-10-21`**: the ship-6 text verbatim plus items (S), (U) and (V),
  one per group, and the note (T) that admission pipelining is not in the profile. The contract admits the ledger trim (and, still, seed first) on a traffic profile only under a waiver that names each at this value (`SCHED2_WAIVABLE_LEVERS`, added to `NEXT_WAIVABLE_LEVERS`); this waiver names the trim only; the upload levers ride the
  waiver's naming of the tier; the lookup has no contract mechanism (like PF2-H and the op-fusion levers), so the waiver names it in words.
* `...-fx-s7-audit` (`make_s7_profiles.py`): the audited attach arm, gate-only: the ship-6 audited twin plus every difference of the candidate **except the ledger trim** (the audit's judge reads the
  ledger's points). The second audited twin of tp4/s7 (`...-fx-s7-audit-mid`) is retired (the generator drops a stale copy).
* The control of every timed pair is the ship-6 profile itself. Generator order: `make_fusion_profiles.py`, `make_octo3_profiles.py`, `make_ship4_profiles.py`, `make_np1_profiles.py`, `make_xe_profiles.py`, `make_pf2_profiles.py`,
  `make_s5_profiles.py`, `make_s6_profiles.py`, `make_sched2_profiles.py`, `make_pf3_profiles.py`, `make_pf3k_profiles.py`, `make_s7_profiles.py`, then `make_fusion_jobs.py` and `make_s7_jobs.py`. The generators of this chain
  place their twins after the last twin of the generators before them, so each one's `--check` holds whichever ran last.
* The pack (`scripts/ci/references/fusion-jobs/S7`, `make_s7_jobs.py`), one image `tp4-s7-1` (a placeholder tag):

| job | mode | minutes (ESTIMATE) | what |
| --- | --- | ---: | --- |
| B0 | stop | 40 (build) | build the **ship image** from the pushed head with `...-fx-traffic-s7` baked as the serving default and the K64j-OQ graft (refused while the waiver is PENDING): the window's image is the ship image |
| X0 | stop | 1 | status, rescan, reset |
| A1 | stop | 61 | `-s7-audit`, warmup, coding, np1_live2, np1_live4, prefill_few_decoders, concurrent8_drain, concurrent8_steady, agent_turns |
| T1A / T1B | soft | 18 each | ship-6 control / s7 candidate: warmup, np1_live1, np1_live2, np1_live4, prefill_few_decoders, concurrent8_steady, agent_turns |
| R1 / R2 | soft | 45 each | the restore pair (below) |
| T2A / T2B | opt | 15 each | agent_turns8 pair |
| Z | soft | 5 | status, reset |

Core (X0 A1 T1A T1B Z) 103 minutes of cards; with the restore pair 193; with the agent_turns8 pair 223. The minutes of X0, A1, T1 and T2 are the 2026-10-21 ship-6 window's; R1 and R2 are the upload-1
pack's estimates. The audited A1 hit three agent-turn client timeouts from the audited profile's about 2.6 GB per chip DRAM admission hold (the owner accepted that for ships 5 and 6), so a hold line or a client timeout on A1 is the
audit's weight, not the candidate's.

**The restore test.** No smoke test evicts a returning session to the host tier and restores it (the smoke tests' returning turns restore too, untimed and mixed with prefill). The prefix gate has exactly that cycle as a plan
(`tier-upload-timed` / `tier-upload-timed-control`: cold prefill, device hit, floods the size of the KV pool, the returning hit, its cold twin, at ~64k / 128k / 254k), so R1 and R2 are that plan on the control and on the candidate.
No new harness. `upload_restore_compare.py` wants an ABAB; the single pair is passed twice and its repeat floor reads 1.

## 3. Removing a lever

Delete its line from the ship entry of `scripts/ci/fusion-wp/S7.json` (for the ledger trim also from the waiver's `levers`), run the generators in the order above and rebuild: `QWEN_PREFIX_TIER_IO_THREADS` with
`QWEN_PREFIX_TIER_VERIFY_THREADS`, `QWEN_FAST_ADMISSION_LEDGER_TRIM`, `QWEN_FAST_LOOKUP_DRAFT` (seed first and the traced tails are already out). The audited twin
reads the candidate's differences from ship 6, so it follows.

## 4. Merge notes (tp4/s6 + four branches)

All four branches share the merge base `758264ac` with `tp4/s6`. The merges were clean except where each branch edited the census tests that exempt generated profile families (one-line unions with the s6 terms), `profile_twins.py`,
`qwen_c2_profiles.json` (kept s6's file and regenerated the twins) and, for `tp4/prefill-3`, `serving_c2_contract.py` (the K-split waivable-lever block beside the s6 governor entry) and the fusion-jobs ORDER header. A tracked
`.pyc` that `tp4/prefill-3` added is dropped. The sched-2, prefill-3 and prefill-3-split generators were re-anchored after the s6 audited twin so that all generators commute, and the tests that compared "every profile that existed
before this branch" with a fixed parent now subtract the other branches' generated twins.
