# tp4/sched-2: admission pipelining and the memory-ledger trim (NP3)

Status: built and unit-tested on the CPU, never run on a card. Three switches, all default off and all gate arms (the profile contract refuses each on a traffic profile). With every switch unset
the serving path is the ship-5 path call for call (tests below). Every number marked **ESTIMATE** is arithmetic over the measured segments of section 2, never a card result; section 5 says how.

Base: `tp4/next-1` head (ship 5, `docs/tp4-next1.md`), which already carries the lean admission and the complete spills of `tp4/sched-1` (`docs/tp4-sched-1.md`).

## 1. The answer

| | lever | switch | what it moves | ESTIMATE |
| --- | --- | --- | --- | --- |
| 1 | **seed first** (the pipelining) | `QWEN_FAST_ADMISSION_SEED_FIRST=1` | the seed token leaves the engine before the request's engine is bound; the bind runs at the start of the next `schedule()` call | a returning turn's first token p50 1.46 s -> about 1.19 s (-275 ms, the whole bind); the last of eight fresh prompts -0.5 s (-0.35 s once the ledger trim has taken its part of the bind) |
| 2 | **hold reuse** | `QWEN_FAST_ADMISSION_HOLD_REUSE=1` (+ `_MAX_AGE_MS`, `_MAX_REUSES`, `_DRIFT_MB`) | the DRAM hold takes no allocator reading when a recent one clears its terms after a drift allowance | -84 ms for three holds in four (-63 ms on average) of back-to-back arrivals; -0.65 s on the burst's last first token |
| 3 | **ledger trim** | `QWEN_FAST_ADMISSION_LEDGER_TRIM=1` | the memory ledger ends after the attach: no bracket at a slot's first rebind, no admission point | nothing on a warm process; the burst into a fresh process -1.7 s of 15.4 s (the eight first rebinds' brackets) |
| | all three | | | burst last first token 15.4 s -> about 12.7 s (-17 %); returning turn p50 1.46 s -> about 1.12 s (-23 %) |

What the brief asked for, preparing the next arrival's engine and host state **during** the current prefill, is not available at ship 5 and is not built (section 3). The levers above take its place:
they remove host work (2, 3) and move the first token ahead of the work that no longer has to precede it (1).

## 2. Where the admission time goes at ship 5 (measured)

Source: the container log of the ship-5 timed arm (T1B of the 2026-10-19 window, 93 admissions of a returning mix plus the first burst of eight fresh prompts of about 3.5k tokens into a fresh
process). `scripts/ci/sched2_report.py <container log>` reads any such log; `--burst` prints the first eight arrivals one line each.

A **warm returning turn** (78 of the 93), p50 in ms: the scheduler's DRAM hold 84 (the allocator read between the kv-fit and dram-fit lines) + the prefill 690 + the drain 13 (what the sampler still
waited for after the route's own clock) + **the bind 275** (p90 391): A the slot adoption and host checks 65, B the placement and window join 11 (mean 55), C the rebind 181 (re-zero 30, projection 91,
K/V history 15, engine page tables and carry 29). Its first token leaves at p50 1.46 s, p90 2.33 s. So **about 400 ms (hold + drain + bind, p90 764) is serial between one arrival's prefill and the next's**
and none of it is prefill.

The **burst into a fresh process** (eight slots, all new), ms, with the last first token at 15.39 s:

| | ms | share |
| --- | ---: | ---: |
| prefill (the device) | 5,388 | 35.1 % |
| the decoders' owed rounds (Lever N's pacing between arrivals) | 4,151 | 27.0 % |
| the rebind itself and its host checks | 2,199 | 14.3 % |
| the memory ledger's bracket at each slot's first rebind (170 to 270 ms each) | 1,661 | 10.8 % |
| the conv-slice readback at each native slot's first adoption (85 to 155 ms each) | 868 | 5.7 % |
| the scheduler's hold | 855 | 5.6 % |
| the drain | 229 | 1.5 % |

The sched-1 lean admission skips the bracket and the readback **after a slot's first use**; a burst into a fresh process is eight first uses, which is why 17 % of that burst is first-use cost. A warm process
pays none of it. The readback is a proof (a geometry check on the first adoption of each native slot) and is kept; the bracket is a diagnostic and is what lever 3 removes.

`scripts/ci/sched2_model.py` replays these segments (the eager replay reproduces every one of the eight first-token times within 2 %, `--check`) and is the only source of the estimates here.

## 3. Why the engine is not prepared during the prefill

The prefill is a **synchronous** device call: `levern_route` reads the logits back with `to_torch` after the traced chunk loop, whose chunks are separated by a device synchronize. The host is inside
that call for the whole prefill, and what is left on the device when the route returns is 13 ms at p50 (8 ms over all admissions; the drain). So there is neither a host-idle window during the prefill to
hide host work behind (the only code that could use it would have to live inside the traced chunk loop, which is model code the image patches) nor a device-busy window after it: everything after
the route (PF2-H's hand-off 55 ms, A 65, B, C 181) is dispatched one op at a time onto an idle device. A helper thread for the allocator read is not safe (the allocator is mutated by the dispatch thread).

Whether C's four phases are host-dispatch-bound or device-bound is unknown: the lean phases line is a host clock (`fence_ms=0.1 rezero_ms=30 project_ms=91 copy_ms=0.2 history_ms=15 engine_ms=29`). If they
are dispatch-bound, dispatching them ahead of a fence would hide up to about 270 ms a returning turn; if device-bound, nothing. **A card probe that fences between phases would say**; it is the next step if
the owner wants more, and nothing is moved on a guess. Rejected on the same evidence: hoisting the re-zero (30 ms) to park time (it moves the device work, it does not remove it); `QWEN_FAST_PARKED_PROJECT_ROWS=0`
(one projection call, estimate -40 to -50 ms, costs 250 MB of DRAM headroom and needs the rebind audit again; an existing knob, not this branch's).

## 4. The levers

### 4.1 Seed first (`admission_pipeline.SeedFirst`, `serving_lifecycle`)

`FastServingLifecycle._sample` builds the request's engine (the bridge: slot adoption, rebind, hook) before it returns the sampler's output, so the first token waits for the whole bind. With the switch on,
`_sample` asks the policy (`SeedFirst.defer`); if it says yes it stores the request's state in `deferred`, keeps `request_id`, `capture` and the prefill gate held, and returns the output at once. The bind
(`complete_deferred`: `bridge_factory`, then the same hook create/attach `_sample` ran, now `_admit_bridge`) runs at the start of the **next `schedule()` call**: the policy wraps `schedule` on the scheduler class
the engine runs (outermost, installed after the admission cap and Lever N's wrapper), and the wrapper completes a pending bind before calling it. `_execute` completes one too, as a backstop where no schedule()
ran. Why this order:

- **Same device work, same order.** The bind is the same call with the same arguments, before the next step's anything; nothing is scheduled in between (the gate is held, so the scheduler sees a prefill in flight, as
  between chunks).
- **Same pacing.** The alternator measures a prefill step's time from one `schedule()` call's `begin()` to the next's; the bind runs before that `begin()`, so the step still includes it. Held by
  `PacingIdentityTests` on the real alternator and the plugin scheduler over a fake clock: every decision, every debt and the clock at every scheduling call are equal to the eager order's, and the first token leaves
  earlier by exactly the bind.
- **Same scheduler view.** The DRAM hold, the gate and the parked terms are read inside `schedule()` after the bind, as before.
- **An abort in the gap** is the eager case: the request is bound when the next call begins, and the finished id detaches it in the next `_execute`.

The one real difference is the **refusal path**. The sampler's bind may be refused (`RequestRefused`: the DRAM backstop after a failed rebind), which `_quarantine_request` turns into an abort riding the step's
output; after a seed has gone there is no such output. So the policy defers **only** a request that will rebind a parked engine (`peek()` names a parked slot, no injected fault, parked set open) **and** whose cold
fallback's DRAM terms also clear on the reading its own hold took (`serving_prefill_admission.reading_clears`, the backstop's reuse rule without using the reading up). Anything else binds in the sampler as
before (counted and logged with its reason). A refusal that happens anyway stops the engine with the request named, as any other bind failure does; the gate reads for the line.

### 4.2 Hold reuse (`admission_pipeline.HoldReadings`, `serving_prefill_admission.dram_predicate`)

The hold's predicate reads four DRAM views and four trace views (84 ms at p50, the device idle). With the switch on it keeps the last reading and reuses it while it is younger than `_MAX_AGE_MS` (10 s), has served
fewer than `_MAX_REUSES` (3) holds, was taken under the same parked-set signature (fallback, unpark and re-park counts, each slot's state and whether its single is released) and still clears the **same**
`split_short` terms after a discount of `(uses + 1) x _DRIFT_MB` (256 MB) off the free total and off the largest block, which is halved first (a prefill can split it). Anything else is a real reading, which replaces the kept one. The
backstop's reuse (`QWEN_FAST_ADMISSION_LEAN`) takes the reading's real age and the admissions since it as an extra allowance.

Evidence for the drift (the 94 `dram admission fit` lines of the log): inside the burst the free total fell 4,906.8 -> 4,798.0 -> 4,586.0 -> 4,477.5 MB and the largest block 4,690.9 -> 4,588.3 -> 4,380.5 -> 4,277.3 MB, steps of 103,
208 and 103 MB (the first trace captures of new slots), and in the 80 admissions after the burst under 1 MB an admission. The one larger step (the largest block -562 MB, the free total -319 MB) came between two tests, more than 10 s
later, with the burst's engines closing and their singles rebuilt at the idle moment: a signature change and an age over the limit, so a reuse would not have spanned it. 256 MB covers the worst step inside a burst. The assumption (a
low-address-first allocator takes from the free end) is the weak point; the randomised test holds the arithmetic against a model that also halves the block once, and the gate (section 7) holds the real thing. A stale reuse that admitted a prompt that does not fit would be a prefill allocation
failure, so the limits are small on purpose and a tight reading is never reused.

### 4.3 Ledger trim (`admission_pipeline.trim_ledger`, `serving_request_factory.lean_bracket_skipped`)

`memory_ledger.end()` after the attach: every later `record`, `before`, `after`, `engine_admitted` and `first_packed_round` is a no-op (the module's own contract), and `lean_bracket_skipped` is always true so the rebind
does not even ask the parked set for its terms. The ledger is a diagnostic: `Ledger.before` swallows its own errors and no judge reads a bracket per rebind. What is lost is the first-prefill `model_after_prefill`
check of `c2_smoke_check` (it only fires when buffers are reported), so the audited arm does not trim.

## 5. Estimates

`python3 -B scripts/ci/sched2_model.py` (ESTIMATES, arithmetic over the measured segments; the owed-rounds cap binds in the measured burst, so a shorter chain does not shorten the decoders' gaps):

| | burst last first token | vs eager | warm returning turn p50 |
| --- | ---: | ---: | ---: |
| eager (ship 5) | 15.35 s (measured 15.39) | | 1,460 ms (measured) |
| ledger trim | 13.69 s | -10.8 % | no change |
| hold reuse | 14.70 s | -4.2 % | 1,397 ms (-4.3 %) |
| seed first | 14.83 s | -3.4 % | 1,185 ms (-18.8 %) |
| hold + ledger | 13.04 s | -15.0 % | 1,397 ms |
| all three | 12.70 s | -17.3 % | 1,122 ms (-23.2 %) |
| all three, owed rounds following the step (an upper bound) | 11.79 s | -23.2 % | |

Seed first moves each first token earlier by that arrival's **own** bind and nothing else (the bind still sits before the next arrival's hold), so it helps the single returning turn most and the burst's tail least. The
decoders' tokens a second are not touched by any lever; the gate reads them.

## 6. Merge points with the adaptive Lever N governor (`tp4/next-2`)

This branch edits **none** of `levern_policy.py`, `levern_scheduler.py`, `test_levern_*.py`. Where the two meet:

| file | this branch | likely conflict |
| --- | --- | --- |
| `serving_prefill_admission.py` | `HOLD_REUSE_FLAG`, `dram_predicate` (signature, the reading), `note_reading` / `reusable_reading` (the extra allowance), new `reading_clears` | none with `wrap` / `install` (the governor's wiring); a textual one only if they touch `dram_predicate` |
| `serving_lifecycle.py` | `seed_first` / `deferred` class attributes, `__init__` (kwarg and install), `_release_prefill`, the head of `_execute`, the bind tail of `_sample` (now `_admit_bridge`, `complete_deferred`), `close` | none expected |
| `serving_request_factory.py`, `serving_runtime.py`, `serving_c2_contract.py` | `register_dram_admission`, `lean_bracket_skipped`; the block before `FastServingLifecycle(...)` and the trim after it; `sched2_problems` and one call | contract and runtime blocks are additive |
| profiles, `profile_twins.py`, census tests | five twins named `...-w2-er-sch-s5-*`, outside the fusion namespace | regenerate with `make_sched2_profiles.py --write`; the census exemptions are one-line unions |

Two **semantic** couplings to keep when the governor lands: (a) the seed-first wrapper must stay the **outermost** `schedule` wrapper, so that the bind runs before the alternator's `begin()`; the lifecycle installs it last,
and `PacingIdentityTests` fails if the order flips; (b) the alternator must keep measuring a prefill step from `begin()` to `begin()` (the bind is then inside it). A governor that read the step time from the output's
arrival instead would see the bind move out of the prefill step into the next decode step and owe the decoders less.

## 7. The card gate plan (`scripts/ci/references/fusion-jobs/SCH2`, `make_sched2_jobs.py`; templates only, no tag pushed from the branch)

Image `tp4-sched-2` (a placeholder B0 builds from the pushed head with the K64j-OQ graft; it bakes the ship-5 candidate). ARC runners at zero from A1 to LT1. Shortest useful window **X0, A1, C1, L1, Z: 175 minutes of cards**.

| job | minutes | what |
| --- | ---: | --- |
| B0 | 40 | build (may run while production serves) |
| X0 | 15 | status, rescan, reset |
| **A1** (stop) | 80 | **exactness**: the ship-5 audited twin plus seed first and hold reuse; warmup, coding, `concurrent8_drain`, `concurrent8_steady` (eight fresh prompts of ~3.5k tokens), `agent_turns`. Every audit exact, no mismatch, no fall-back, the lever lines present |
| **C1, L1** (+ C2, L2) | 35 each | **timed ABAB**: the candidate against all three levers; `warmup`, `concurrent8_steady` FIRST (the fresh-process burst), `agent_turns`. Per user and turn, paired by index: the first-token time (the burst's eight arrivals in order, every returning turn), the answers EQUAL, the decoders' rate and longest gap not worse |
| SE1, HO1, LT1 | 35 each | attribution: each lever alone |
| Z | 10 | reset |

**GO** for a lever: the answers equal, no audit mismatch, the lever's lines present, a first-token gain beyond the control-to-control floor, and no loss in the decoders' rate, the agent_turns rate or any gap.
**NO-GO**: a token that differs, an `admission seed first: the bind of ... failed` line, an allocator failure, a first token later than the control's. Adopting a lever needs an owner traffic waiver naming it; the hold
reuse also wants a ballast-boundary run (the engine-reuse ballast ladder, on the ship-5 twin) before any ship.

## 8. Risks

- The estimates rest on one run's segments (one 8-burst, 93 admissions) and on the cap binding; sections 2 and 5 say which part is measured.
- Seed first moves a failure after a token has gone; the preflight makes a refusal unexpected, not impossible. Its premise (the engine core ships a step's output before the next `schedule()`) is the engine core's, read
  from the source of the vLLM the image carries, not run here: **the first A1 line to check is that a first token's client time leads the bind's `built` line**.
- Hold reuse trades an allocator read for an assumption about allocation; tight readings are never reused.
- The ledger trim removes the only per-request check of `model_after_prefill`; the audited arm keeps it.
- `agent_turns` and `concurrent8_steady` are the pack's only timed shapes; a one-user shape (the lone-user lane) gains the same bind and hold but is not timed here.

## 9. Tests

New: `test_sched2_pipeline` (51: the switches' strictness and their equality across the five modules that name them; the reuse rule with a randomised allocator model and the verdict grid; the ledger; seed first on
the real lifecycle fixtures: the whole story of a request equal with and without it, the order, the timing on a fake clock, failure, close, abort; the real alternator), `test_sched2_model` (23: calibration, lever arithmetic,
the report on synthetic logs), `test_sched2_profiles` (12), `test_sched2_pack` (19), all named in `qwen-integration-cpu.yml`. The new module is in `docker/qwen-c2-overlay.txt` and the closure tests. Run narrow, niced, from
`scripts/ci`: `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 15 python -B -m unittest test_sched2_pipeline`.
