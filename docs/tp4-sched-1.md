# tp4/sched-1: three scheduler-side levers for the production shape

Status: built and unit-tested on the CPU, never run on a card. Every lever is a flag, default off, gate-only (the profile contract refuses each on a traffic profile),
and the flag-off path is the code that ran before (section 8 says how that is held). Every number marked ESTIMATE is a model or a reconstruction, not a measurement;
sections 4 and 6 say which parts of each are measured.

The production shape this answers (the ship-4 stack, the last two hours before the task): 58 % of card wall time in prefill steps, 36 % in decode rounds, 6 % other;
per-user tok/s by live users 1 -> 17.8, 2 -> 37.3, 3 -> 28.8, 4 -> 21.6, 5 -> 16.7, 6-7 (octo) -> 14-16; every returning turn pays an engine rebind (the "sticky engine built"
line: p50 541 ms, p90 721 ms; the parked rebind inside it p50 188 ms); and on the host tier, 3,393 of the 3,905 blocks a 254k restore displaced in the TC1 card run were
dropped by the 512-block spill cap.

## 1. The answer

| | lever | flag(s) | cause | ESTIMATE on the production numbers | owner decision |
| --- | --- | --- | --- | --- | --- |
| 1 | complete spills | `QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete` | `flush()` cuts every flush at 512 blocks and drops the rest for good (`qwen_prefix_scheduler_patch.py:689-692`, cap at `qwen_prefix_registry.py:195`); the bytes cannot wait for a later step | TC1: the 3,393 blocks (87 %) the cap dropped are kept; the step that displaces them pays 0.4 ms a block once the slab is resident and 1.7 ms while it fills (+1.6 s to +6.6 s on a 254k restore, unsliced) | yes: it lengthens the restore step until `tp4/tier-slice` slices the restore |
| 2a | returning-turn pacing | `QWEN_FAST_LEVERN_OWED_CAP_MS=1500`, `_SHORT_SHARE=0.65`, `_SHORT_WAIT_S=5` | the owed time after a prefill step is capped at 8 rounds of the measured round time (0.5 to 0.6 s at one to four seats), and a short step owes the decoders one round, so while returning turns arrive back to back the decoders get one round per 1.8 s step | decoders at 5 to 7 live +9.6 % (17.7 -> 19.4 tok/s), paid by returning-turn TTFT p90 +25 % (6.4 -> 8.0 s), cold TTFT p90 +32 %, turns per minute -2 %; the share moves delay, it removes none | yes: a trade, not a gain |
| 2b | few-decoder policy | `QWEN_FAST_LEVERN_ADAPTIVE=1` (ported from `tp4/prefill-1`) | with one or two decoders live a cold prompt still gets half the device | cold-prompt TTFT -33 % at the median in the quiet-hours shape (84 -> 56 s), the one or two decoders' rate falls in that window; no change with three or more live | yes: the same trade |
| 3 | lean admission | `QWEN_FAST_ADMISSION_LEAN=1` | 541 ms of a returning turn's engine build is a 188 ms rebind plus about 350 ms of host-side accounting with every seat stopped (section 4) | the sticky build p50 541 -> about 240 ms, and 66 ms more outside it: 410 ms (290 ms for the lighter class of rebinds) a returning turn; simulated turns per minute +10 %, seat rate +13 %, returning-turn TTFT p90 -23 % | yes: it takes two verification readings and a diagnostic bracket out of the admission |

The one that returns device time is lever 3, and the existing `QWEN_FAST_ADMISSION_DIAG_TRIM=1` (no code; the ship-4 profile does not set it) returns about 100 ms
more. Put them first. Lever 2a and 2b move delay between the decoders and the arrivals; they are the owner's trade and are not recommended on the evidence alone.
Lever 1 is the right fix for the cap and needs the sliced restore beside it.

## 2. The evidence this note starts from

| | | |
| --- | --- | --- |
| wall time | prefill steps 58 %, decode rounds 36 %, other 6 % | the owner's two-hour read of the ship-4 stack |
| Lever N | `PREFILL_SHARE=0.5`, an 8 s decode-gap floor, 16k solo steps | `levern_policy.py` |
| rebind | sticky engine build p50 541 ms, p90 721 ms; parked rebind p50 188 ms | `serving_runtime.py:1070` |
| host tier | TC1: 3,393 of 3,905 displaced blocks dropped by the cap; spill p50 1.7 ms, p90 4.8 ms an event; production counters `kv_evicted` 480 and `ckpt_evicted` 1,061 a week with the tier full | |

## 3. Item 1: the spill cap

### 3.1 Where the blocks go

`TierHooks.on_evict` (`qwen_prefix_scheduler_patch.py:634`) is called as vLLM evicts a prefix-cached block to reuse it; the block goes to `pending`. `flush()` (`:666`)
runs at the end of the same `schedule()` call, before the model writes the device, and at the top of every restore (`load()` at `:855`, `self.flush()` at `:903`). It sorts
the pending blocks lowest index first within each chain (a restore needs the run from the chain's first block), then
`keep, cut = items[:limit], items[limit:]` at `:690`, with `limit = config.spill_max` and `spill_max` = `DEFAULT_TIER_SPILL_MAX_BLOCKS = 512`
(`qwen_prefix_registry.py:195`, read at `:553-556`). The cut blocks are counted `tier_spill_dropped_cap` and `unhold`ed (their checkpoints dropped).

Why the rest cannot wait for the next step: the blocks vLLM evicted are reused by the very allocation that evicted them, so by the next `schedule()` the device holds
another request's bytes. A deferred spill would have to hold the blocks out of the free queue, which is a change to vLLM's block lifetimes; this note does not make it.
The one flush that exceeds 512 is a restore: `load()` allocates the whole session at once and evicts as many blocks as it restores (a 254k session is 3,905 blocks).

### 3.2 What a block costs

From the `host tier spill blocks=N ms=` lines of the recorded TC1 run: `ms = 2.64 + 1.705 x blocks` while the slab fills (1 block 4.3 ms, 32 blocks 57 ms, 256 blocks 439 ms,
3,905 blocks about 6.7 s), and about 0.4 ms a block once the slab is resident (the comment at `qwen_prefix_registry.py:196-202`). A block is 2,228,224 bytes, so a full
4,096-block flush moves 9.1 GB. `scripts/ci/sched_report.py` fits the same line from any later log.

### 3.3 The change (default off)

`QWEN_PREFIX_HOST_TIER_SPILL_MODE=cap|complete` (default `cap`, today's code). In `complete` mode `flush()` uses `max(spill_max, plan.hard_max)` as its limit
(`QWEN_PREFIX_HOST_TIER_SPILL_HARD_MAX_BLOCKS`, default 4,096: a 262,144-token session) and reads in batches (`flush_batches`, `:740`;
`QWEN_PREFIX_HOST_TIER_SPILL_BATCH_BLOCKS`, default 512), one slab reservation and one device read a batch, so a read that fails loses its batch and not what was stored.
The counters of the cap mode are unchanged and `tier_spill_batches`, `tier_spill_cap_lifted`, `tier_spill_max_blocks`, `tier_spill_max_ms`, `tier_spill_slow` are added only
when the mode is on. A flush that held the engine longer than `QWEN_PREFIX_HOST_TIER_SPILL_WARN_MS` (default 1,000) logs one line (`prefix_markers.TIER_SPILL_SLOW`).
`qwen_prefix_registry.spill_plan` (`:586`) parses the four values strictly: a malformed one or a knob without `complete` is a configuration error, named at the contract.

### 3.4 Expected gain, and the break-even

TC1: 3,393 blocks kept that were dropped. The step that spills them costs the lifted blocks times the per-block cost: +1.4 s resident, +5.8 s while filling. That step stalls
every seat, which is the cost the bounded batch does not remove: the bytes have to be read in that step.

Is it worth it? Re-use is worth `(recompute - restore) - spill` a block. A cold block costs about 23 ms to recompute at a long context (96 s for 3,905 blocks of a 254k
session; `levern_policy.StepTimes`: 420 ms + 2.5 ms per 1,000 tokens of context a 2,048-token chunk, 32 blocks) and about 14 ms at 32k; a restore costs 1.7 ms a block (6.5 s for
the same 254k session). The break-even probability that a spilled block is wanted again is `spill / (recompute - restore)`: 2 % to 8 % at 254k (0.4 to 1.7 ms of spill),
3 % to 13 % at 32k. The production counters do not separate cap drops from capacity drops, so the weekly gain is not known: `tier_spill_cap_lifted` (new) and
`tier_spill_dropped_cap` (existing) will say.

### 3.5 With the sliced restore

`origin/tp4/tier-slice` moves `TierHooks.load` onto a `RestoreJob` with `advance` and `slice_size`, so the restore's own stall is sliced across steps. It does not slice the
spill, which is a read that has to happen in the step that evicts. The two compose: slicing keeps each step's restore work small; the complete mode keeps what the restore
displaced. Merging them conflicts in five files (`prefix_markers.py`, `qwen_prefix_registry.py`, `references/fusion-jobs/ORDER.txt`, `serving_c2_contract.py`,
`test_tp4_fusion_jobs.py`), all additive (a trial merge; section 9). Complete mode should not ship without it.

### 3.6 Tests

`test_sched_spill.py` (23): the cap mode's counters and drops byte-identical with the mode unset; complete mode keeps up to the hard maximum, in batches, lowest index first;
a failed batch loses itself and not the stored ones; a tier that is full drops the rest and unholds it; the latch after three failures; the slow-flush line; strict parsing;
and the `displaced_by_a_restore` scenario (tier 400 blocks, pool 300, cap 16): cap mode gives session A back at position 0 and F at 0, complete mode gives 8,192 and 12,288.
`test_prefix_tier_profiles.py` and `test_qwen_prefix_tiers.py` pin the new names.

## 4. Item 3: where a returning turn's 541 ms goes, and what comes out

### 4.1 The timeline

The "sticky engine built" line (`serving_runtime.py:1070`) is the bridge build's own clock (`began` at `:1044`, `bridge_factory` at `:1015`): for a returning turn,
`from_prefill` (`serving_request_factory.py:499`) with a parked engine to rebind. The audited card run of the engine-reuse window (115 rebinds, the memory ledger on as the
image sets it, `rebind_split.py` segments) gives, in ms:

| segment | what runs | where | rebinds that compiled no program (39) | rebinds that compiled three (35) |
| --- | --- | --- | ---: | ---: |
| A | host checks; the DRAM backstop's allocator reading (4 DRAM + 4 trace views, about 70 ms: the 66 ms "dram after engine" line is the same DRAM reading); the GDN slot adoption's conv-slice readback (48 layers, a geometry proof that is the same for every admission of a slot) | `dram_backstop` `serving_request_factory.py:228`; `adopt_prefill_slot` `:397` | 102 | 216 |
| B | slot placement; the host refusal check with the first `capture.outputs()` (the window join); the ledger's op=rebind reading before | `parked.place` `serving_parked_engines.py:1152`; `rebind_refusal` `:1205`; `rebind_parked` `serving_request_factory.py:443`, `memory_ledger.before` `:464` | 74 | 127 |
| C | the rebind itself: fence, re-zero, the drafter projection from this prefill's taps, the copy, the K/V history, the engine's page tables and carry | `rebind_slot` `serving_parked_engines.py:1241`, `rebind_device` `:549` | 164 | 164 |
| D | the ledger's op=rebind reading after | `memory_ledger.after` | 70 | 65 |
| E | the rest | | 1 | 1 |
| total | | | 446 | 579 |

Production's 541 ms p50 lies between the two classes, near the second. Two classes sit outside this table: 32 of the 115 rebinds were SLOW (the window join compiled kernels:
B of 680 to 711 ms, a total of 1,104 to 1,133; the programme of `tp4/rebind-fast`, not this branch) and the first rebind of a slot took 1,963 ms in C (3 of 115). Outside the sticky line but
inside the stall: the "dram after engine" line (`serving_runtime.py:1080`, 66 ms) and, before the bridge starts, the ledger's prefill points (the "after" reading alone is about 100 ms
after the route line in the audited log; `memory_ledger.admission_diag`, `:132`; the existing trim cuts them to the first admission's).

The memory ledger is on in the image (`QWEN_FAST_MEMORY_LEDGER=1`, `docker/qwen-c2-serving.Dockerfile:144`). Its readings are a diagnostic: `Ledger.before` swallows its
own errors (`memory_ledger.py:489`), and nothing reads the op=rebind bracket per rebind (`parked_markers.py` collects it as `rebind_ops` and no judge uses it).

### 4.2 What `QWEN_FAST_ADMISSION_LEAN=1` removes

| piece | change | where | ms per returning turn |
| --- | --- | --- | ---: |
| the backstop's reading | reuse the reading this very admission's DRAM hold took seconds earlier when it leaves a wide margin (it must clear the backstop's split terms after a 512 MB allowance, be at most 30 s old and be this request's own); a tight reading is read again | `serving_prefill_admission.py:588` `note_reading`, `:606` `reusable_reading`; `serving_request_factory.py:259`; `dram_hold` `:781` notes the reading | about 70 |
| the conv-slice readback | taken at each slot's first adoption only (the rule `QWEN_FAST_ADMISSION_DIAG_TRIM` already uses at this site); the metadata checks and the copy run every time | `serving_request_factory.py:424-425` | 30 to 145 |
| the ledger's op=rebind bracket | taken for a slot's first rebind only | `lean_bracket_skipped` `serving_request_factory.py:123`, used at `:460` (the reading is `:464`) | about 127 |
| the "dram after engine" line | off for a request rebound onto a parked engine (a cold build keeps it) | `serving_runtime.py:1079` | about 66, outside the sticky line |

Sum: 70 + 145 + 127 + 66 = 408 ms for the class that compiled three programs, 70 + 30 + 127 + 66 = 293 ms for the class that compiled none. The sticky line goes from 541 to
about 240 ms (the rest is C, 164 to 188 ms, and the window join). What a rebind leaves for a long-lived slot is exactly what the first one proved: the readback and the
bracket are proofs and records that do not change between admissions of a slot; the admission hold still checks the DRAM before every prefill and the backstop's own terms
still hold on the reused reading.

Nothing was cut that a gate judges: the DRAM hold, the backstop's refusal, the adoption's metadata checks and copy, and the rebind's own refusal check all run. A reading
that does not clear the margin is read again, so the backstop refuses exactly when it would have.

### 4.3 What stays, and why it is not overlapped

C is 164 to 188 ms of device work that depends on the prefill that just ended: the drafter projection reads this prefill's taps (`capture.outputs()`), the copy writes the
slot's history from the projection, the K/V history is seeded over it, and the engine's carry saves this prefill's GDN state. None of that can start before the prefill's
last chunk. The device runs one stream, so work that does not depend on the prefill and is moved earlier gains only the time it takes: the candidate is `ServingBufferPool.rezero`
(phase 2 of `rebind_device`), which depends on nothing but the previous session leaving the slot and could run at park time. Its size is not known: the five device phases
were never timed on a card. `QWEN_FAST_ADMISSION_LEAN` therefore also logs, per rebind, `[PINDIAG] admission lean rebind phases request= slot= fence_ms= rezero_ms= project_ms=
copy_ms= history_ms= engine_ms=` (`serving_parked_engines.py`, `PHASES_MARKER`) and, per admission, `[PINDIAG] admission lean split request= a_ms= b_ms= c_ms= total_ms=`
(`serving_request_factory.py:636`) without the ledger, so the next card run says what is movable. Nothing is moved on a guess.

The other candidate asked for, preparing the returning session's engine while the others decode, has the same limit: the page tables are known at admission, but they are
written inside `engine.rebind` together with the carry save, and what is left after the LEAN pieces is the rebind (C) and the window join, both of which need the taps.

### 4.4 Tests

`test_sched_lean.py` (35): the flag's strictness and the profile contract; the reading reuse (kept, used once, aged out, tight, wrong request, trace term); the hold's notes;
the backstop with and without a parked engine; the adoption (readback at the first adoption of a slot only, per slot); the bracket (every rebind without the flag, a
slot's first with it); the "dram after engine" line; the split and phases lines; and flag off = the same calls (counted allocator reads) as before. The served modules
are in both image copy lists, `docker/qwen-c2-overlay.txt` and `test_tp4_closure_literals`.

## 5. Item 2: the governor

### 5.1 What the production shape asks of it

Many short returning turns (about 3k tokens of tail at a long context: two 2,048-token chunks, about 1.8 s with the 0.54 s admission) beside decoders, and a few long cold
prompts. Today the alternator (`levern_policy.Alternator`, `:893`) owes the decoders `dt x (1 - f) / f` ms of rounds after a prefill step at `f = 0.5`, capped at
`QWEN_FAST_LEVERN_MAX_ROUNDS` (8) times the measured round time, at least one round (`RMIN`); the deadline governor (`effective_share`, `:836`) raises `f` for a long cold
prompt; the 8 s decode-gap floor holds; and a short step (at most 16,384 tokens left to compute) is paced at one round whatever the share. Two consequences. The rounds cap is
0.5 to 0.6 s at one to four seats, below the 0.9 s or more that `f = 0.5` owes after a 1.8 s step, so the decoders get less than the half the share promises exactly where they are
few and fast. And a burst of returning turns costs the decoders one round per step, so at five to seven live seats the arrivals take the device and the decoders' gap is the
sum of the steps.

### 5.2 The policy

`tp4/prefill-1`'s adaptive governor, ported (`QWEN_FAST_LEVERN_ADAPTIVE`, `docs/lever-n-adaptive-governor.md`: with at most two decoders live the base share is 0.9 and the
deadline target 120 s; three or more is today's pacing step for step), and three new knobs for the returning-turn shape, in `levern_policy.py` (`Sched`, `:483`;
`sched_config`, `:540`; `sched_problems`, `:500`; the `Alternator` at `:972` and `:978`) and `levern_scheduler.py` (`oldest_short_wait`, `:424`):

| knob | range | meaning |
| --- | --- | --- |
| `QWEN_FAST_LEVERN_OWED_CAP_MS` | 100 to 10,000 | the most a prefill step may add to the debt, in ms, in place of 8 rounds of the measured round time: the decoders' share is then the share's (up to the cap) whatever the seats |
| `QWEN_FAST_LEVERN_SHORT_SHARE` | decimal in (0, 1) | the decoders' share after a short step that ran beside decoders: the debt is `max(round_ms, dt x (1 - f_s) / f_s)` instead of one round |
| `QWEN_FAST_LEVERN_SHORT_WAIT_S` | 1 to 600 | the guard (needs the share): while the oldest waiting short has waited this many seconds the debt is forgiven at every decision (the one round after a short step stays), so a burst of returning turns cannot queue behind the share for longer than about that |

Chosen: cap 1,500 ms, short share 0.65, guard 5 s. The log gains ` short= cap_ms= guard=` on the step lines and one `lever N sched:` line; `c2_smoke_check` judges them
(`levern_sched_problems`): off means no trace, on means the install line, the fields, the cap honoured, the guard rules, and NOT EXERCISED when no short step ran.
The 8 s floor, the chunking arithmetic, the admission rule, and the text are untouched: only when the decode rounds run between the steps moves.

### 5.3 Ideas checked and not built

| idea | verdict |
| --- | --- |
| batch the returning-turn tails into one step | nothing to save: the first round after a final step costs 77 to 80 ms against 74 to 75 at three and four seats (a card log), so the switch is 5 ms, and a batched step is a longer gap for the same work |
| size the solo step | 8,192 instead of 16,384: +0 % at production load, -2 % in quiet hours (section 6); keep 16,384 |
| re-tune the step for a lone decoding user (`np1-fastlane`: a 72 ms round against 100 ms) | a 4,096-token step while the user decodes, with and without a 32,768 solo step: within noise of the 2,048 step (section 6); the lever for a lone user is the share (adaptive), not the step |
| a larger decoder share for every short | that is lever 2a: it is the trade in section 6, not a free gain |

## 6. The simulation

`scripts/ci/sched_sim.py` (stdlib; `python3 -B scripts/ci/sched_sim.py --seeds 12 --hours 3`) drives the shipped `Alternator`, `step_budget` and the flag parsers on a
simulated clock: ten agents on 8 seats (three in the quiet hours, two for the lone user), a tool wait, a returning turn of about 3k tokens at a long context, a decode of a few
hundred tokens beside the others, again; 3 % of the turns a cold long prompt. The step costs are the repository's own: a 2,048-token chunk at context p costs 420 ms +
2.5 ms per 1,000 tokens of p, scaled to reproduce the wall-time split; a decode round of k live seats costs 105, 74, 83, 116, 136, 95, 95, 100 ms for k = 1 to 8 (the table's
steps plus the host gap); an admission is lognormal with p50 541 ms and p90 721 ms. The sim re-implements only vLLM's waiting queue and the short lane.

Calibration (12 seeds x 3 h): prefill steps 59.5 % of the wall time against 58 %, decode rounds 39.1 % against 36 %, other 1.4 % against 6 %. The per-user rates by live users
run 25 to 40 % low at one to four users (the table: 1 -> 17.8 simulated 11.6, 2 -> 37.3 / 26.0, 3 -> 28.8 / 20.3, 4 -> 21.6 / 15.7) and from 7 % low to 37 % high at five
to seven (16.7 / 15.6, 15.0 / 16.9, 15.0 / 20.6). Read the policy comparisons, not the absolute rates.

Production shape, 12 seeds x 3 h (rate at 5 to 7 live is the mean of those users; TTFT in seconds):

| policy | prefill share | decode share | seat rate tok/s | rate at 5-7 live | returning TTFT p50 / p90 | cold TTFT p50 / p90 | gap p90 / max | turns/min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production (ship-4 pacing) | 59.5 % | 39.1 % | 23.0 | 17.7 | 2.7 / 6.4 | 79 / 135 | 0.46 / 5.3 | 16.0 |
| LEAN (-410 ms a returning turn) | 53.5 % | 45.0 % | 26.0 (+13 %) | 21.1 | 2.0 / 4.9 | 70 / 128 | 0.45 / 3.8 | 17.6 (+10 %) |
| LEAN, low estimate (-290 ms) | 55.3 % | 43.2 % | 25.1 (+9 %) | 20.2 | 2.2 / 5.3 | 73 / 132 | 0.45 / 5.2 | 17.2 (+8 %) |
| `DIAG_TRIM` (existing flag, -100 ms) | 57.7 % | 40.9 % | 23.9 (+4 %) | 18.6 | 2.5 / 5.9 | 75 / 136 | 0.45 / 4.9 | 16.5 (+3 %) |
| LEAN + `DIAG_TRIM` (-510 ms) | 53.0 % | 45.4 % | 26.2 (+14 %) | 21.5 | 1.9 / 4.6 | 70 / 130 | 0.45 / 4.2 | 17.9 (+12 %) |
| owed cap 1,500 ms | 59.2 % | 39.4 % | 23.1 (+1 %) | 17.7 | 2.7 / 6.4 | 80 / 143 | 0.45 / 5.6 | 16.0 |
| short share 0.80 | 58.5 % | 40.1 % | 23.2 (+1 %) | 18.9 | 2.8 / 7.3 | 98 / 162 | 0.45 / 7.7 | 15.9 |
| short share 0.65 | 56.7 % | 41.9 % | 23.9 (+4 %) | 21.4 | 3.3 / 9.6 | 119 / 179 | 0.42 / 9.1 | 15.6 (-2 %) |
| short share 0.50 | 56.0 % | 42.5 % | 24.2 (+6 %) | 22.3 | 3.6 / 11.6 | 139 / 181 | 0.41 / 9.0 | 15.2 (-5 %) |
| short share 0.65 + guard 5 s | 57.8 % | 40.8 % | 23.6 (+3 %) | 19.4 | 3.2 / 8.0 | 101 / 178 | 0.44 / 8.8 | 15.7 (-2 %) |
| short share 0.50 + guard 5 s | 57.4 % | 41.2 % | 23.8 (+4 %) | 20.1 | 3.4 / 8.9 | 111 / 179 | 0.43 / 8.7 | 15.6 (-2 %) |
| solo step 8,192 | 59.5 % | 39.1 % | 23.0 | 17.7 | 2.7 / 6.4 | 78 / 136 | 0.46 / 5.6 | 16.0 |
| adaptive (<= 2 decoders) | 58.9 % | 39.7 % | 23.4 (+2 %) | 17.7 | 2.6 / 6.4 | 77 / 142 | 0.45 / 6.6 | 16.2 (+1 %) |
| LEAN + cap + share 0.65 + guard | 52.1 % | 46.3 % | 26.3 (+15 %) | 22.7 | 2.4 / 6.1 | 90 / 159 | 0.44 / 7.4 | 17.4 (+9 %) |

Quiet hours (3 agents, a third of the turns a cold 64k to 254k prompt):

| policy | seat rate tok/s | returning TTFT p50 / p90 | cold TTFT p50 / p90 | gap p90 / max | turns/min |
| --- | ---: | ---: | ---: | ---: | ---: |
| production | 35.4 | 2.0 / 3.9 | 84 / 142 | 0.39 / 3.3 | 2.4 |
| adaptive (<= 2 decoders) | 37.8 (+7 %) | 1.8 / 3.2 | 56 / 97 | 0.52 / 9.0 | 2.5 (+4 %) |
| owed cap 1,500 ms | 36.0 (+2 %) | 2.0 / 4.1 | 85 / 148 | 0.40 / 4.0 | 2.3 |
| solo step 8,192 | 34.6 (-2 %) | 1.9 / 3.1 | 85 / 144 | 0.40 / 3.8 | 2.3 |

A lone decoding user (2 agents, a quarter of the turns a cold 64k to 200k prompt; relative only, the lone user's rate is the least calibrated figure):

| policy | seat rate tok/s | cold TTFT p50 / p90 | turns/min |
| --- | ---: | ---: | ---: |
| production | 42.7 | 40 / 71 | 1.7 |
| lone-user padded round (`np1-fastlane`) | 81.2 (+90 %) | 35 / 57 | 2.0 (+18 %) |
| np1 + owed cap 1,500 ms | 81.3 | 36 / 59 | 2.1 |
| np1 + adaptive | 79.1 | 33 / 51 | 2.0 |
| np1 + 4,096-token step while the user decodes | 80.1 | 35 / 57 | 2.0 |
| np1 + 4,096 step + 32,768 solo step | 80.3 | 36 / 56 | 2.0 |

How to read it. A work-conserving system moves delay and removes none: every pacing row trades the decoders' gap against the arrivals' wait (the share moves 5 to 7 live
users from 17.7 to 22.3 tok/s for returning-turn TTFT p90 from 6.4 to 11.6 s), and the guard gives part back (0.65 with the guard: +9.6 % for +25 % TTFT p90). Only the lean
admission removes device time, which is why it improves both columns at once. The differences below about 3 % on rates and 15 % on the cold-prompt columns (about eighty
cold prompts in a three-hour run of one seed) are inside the seed noise (four seeds against twelve moved the low-estimate row's cold TTFT p50 from 61 to 73 s).

### Recommendation

1. `QWEN_FAST_ADMISSION_LEAN=1`, and `QWEN_FAST_ADMISSION_DIAG_TRIM=1` beside it (an existing flag the `c2-packed-tp4-f2` and `-f12` traffic twins already set, which cuts the
   ledger's prefill points to the first admission's).
2. Owed cap 1,500 ms in place of the 8-round cap: it gives the few and fast decoders the share the share promised, and the production-shape sim shows it nearly neutral
   (+1 % rate, cold TTFT p90 +6 %); take it with the pacing or leave it.
3. Short share 0.65 with the 5 s guard only if the owner prefers the 5 to 7 live decoders' rate to the returning turns' wait: +9.6 % against +25 % TTFT p90. No evidence here
   says which the users feel.
4. Adaptive for the quiet hours: cold prompts -33 % at the median with at most two decoders, whose rate falls in that window; no effect at three or more.

## 7. The flags, the profiles, the card jobs

| flag | values | default | module |
| --- | --- | --- | --- |
| `QWEN_FAST_ADMISSION_LEAN` | 0, 1 | off | `serving_prefill_admission.py:563`, `serving_request_factory.py`, `serving_runtime.py:1079`, `serving_parked_engines.py` |
| `QWEN_PREFIX_HOST_TIER_SPILL_MODE` (+ `_HARD_MAX_BLOCKS`, `_BATCH_BLOCKS`, `_WARN_MS`) | cap, complete | cap | `qwen_prefix_registry.py:202` |
| `QWEN_FAST_LEVERN_OWED_CAP_MS`, `_SHORT_SHARE`, `_SHORT_WAIT_S` | see 5.2 | unset | `levern_policy.py` |
| `QWEN_FAST_LEVERN_ADAPTIVE` (+ `_ADAPTIVE_DECODERS`, `_ADAPTIVE_SHARE`, `_ADAPTIVE_TTFT_S`) | 0, 1 | off | `levern_policy.py` |

All are gate-only: `serving_c2_contract` refuses each on a traffic profile (`sched_problems`, `levern_problems`, `prefix_policy_problems`) and the gate marker
`QWEN_C2_GATE_PROFILE=1` replaces the owner waiver on the twins.

`make_sched_profiles.py` generates six gate-only twins of the approved profile `c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s4` (the approved profile plus the gate marker
instead of its waiver plus exactly the named flags; the engine, memory plan and every other flag are the parent's byte for byte): `-sch-lean`, `-sch-spill`, `-sch-pace`,
`-sch-adaptive`, `-sch-all`, and `-sch-all-audit` (the ship-4 audited twin plus the four levers: every audit of the candidate and the tier's instruments). Every profile that
existed before this branch is byte-identical (`test_sched_profiles.py` compares the file with the commit the branch was cut from).

`scripts/ci/references/fusion-jobs/SCH/` holds the window (`make_sched_jobs.py`, 17 templates, ORDER.txt, 890 minutes): B0 the build, X0 the status, A1 the audited attach of
the all-audit twin (exactness: every audit line exact, the lever lines present), S1 the prefix tier-concurrent plan on the all-audit twin, C1 L1 C2 L2 the ABAB of the approved
profile against `-sch-all` on the production-shaped `agent_turns` test, LE1 PA1 FD0 FD1 the attribution, T0 to T3 the tier-concurrent-timed control against the spill arm, and Z
the reset. Swift is selected by `C2_MODEL_REPO` / `C2_MODEL_REVISION` on the smoke jobs (the prefix action does not forward them and runs the base weights, which the tier's
mechanics do not depend on). `agent_turns` is built on `tp4/agent-turns` (smoke tests `agent_turns` and `agent_turns_long`, read with `scripts/ci/agent_turns.py report`); it merges
into this branch without a conflict, and the pack says to merge it before the window. `scripts/ci/sched_report.py <container log>` (`--second` for B over A) reads the rebind p50 and p90, the
lean split and phases, the readbacks skipped, the backstop reuses, the brackets, the prefill share of the wall time while a prefill is pending, the rounds after each returning
turn, the longest stretch without a decode round, the guard steps, and the spill fit and counters. Every figure prints `-` when no line is behind it.

## 8. Risks and how the flag-off path is held

- The simulation is calibrated, not measured: low-k rates run 25 to 40 % low, other is 1.4 % against 6 %, and the user and tool-time distributions are chosen to reproduce the
  baseline. The comparisons are the policies', but the sizes are estimates.
- LEAN's 408 ms rests on the segments of one audited card run (ledger on, the production image's default) and a reconstruction of the production class mix from p50 541 ms; the
  readback range (30 to 145 ms) is A minus the 70 ms reading. LE1 measures it; `sched_report.py` prints the prediction's own split.
- LEAN skips two verification readings and a diagnostic bracket after a slot's first. The first of each stays; the DRAM hold and the backstop's terms are unchanged; if a card
  run disagrees, the flag is off by removing one line.
- Complete spills lengthen the restore step until the restore is sliced, and the effect on the tier's churn (it is full) is unknown. S1 and T0 to T3 are the check.
- The pacing and the adaptive policy are trades. They are default off and the owner decides.
- `agent_turns` is on another branch; until it is merged the production-shaped pair cannot run from this branch.
- Flag off: each lever's call shape and log are what they were. Held by tests, not by argument: `test_sched_lean` (counted allocator reads, calls, lines), `test_sched_spill` (stats and
  drops byte-identical), `test_levern_policy` and `test_levern_prefix_scheduler` (two schedules' whole logs equal with the knobs unset), `test_levern_check` (no trace with the flag off), and
  `test_sched_profiles` (every pre-existing profile byte-identical).

## 9. Merge notes

`tp4/tier-slice` conflicts in five files, additive and resolved by union (section 3.5). `tp4/np1-fastlane` conflicts in nineteen (the generated profiles file, `profile_twins.py`, the
census tests, ORDER.txt), also textual: resolve by union and regenerate the twins with `make_sched_profiles.py --write` and `make_fusion_profiles.py --write`. `tp4/agent-turns` merges
cleanly. The served modules are in both image copy lists, `docker/qwen-c2-overlay.txt` and `test_tp4_closure_literals`; `scripts/ci/__pycache__` is tracked, so run tests with `python -B`.

## 10. Tests

New: `test_sched_lean` (35), `test_sched_spill` (23), `test_sched_sim` (14), `test_sched_report`, `test_sched_pack` (18), `test_sched_profiles` (12), all named in
`qwen-integration-cpu.yml`. Extended: `test_levern_policy` (`SchedConfigTests`, `SchedAlternatorTests`, the ported adaptive tests), `test_levern_prefix_scheduler`
(`SchedPacingTests`, the adaptive tests), `test_levern_check` (`SchedPacingRuleTests`), `test_prefix_tier_profiles`, `test_qwen_prefix_tiers`, `test_tp4_fusion_jobs`,
`test_fused_commit` (`without_admission_lean`), `test_tp4_closure_literals`, and the profile census tests that enumerate generated twins. Run narrow, niced, from `scripts/ci`:
`CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 15 python -B -m unittest <module>`. The full suite: `gh workflow run qwen-cpu-suite.yml -f ref=tp4/sched-1`.
