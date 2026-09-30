# One fast lane beside standard lanes, at TP4: the mechanism, its gate, and the lanes hardware window

Status 2026-10-01: built on `tp4/lanes`, behind default-off flags and gate-only profiles; nothing here has run on hardware.

**The target (coding workloads):** one *fast lane* for a single user at >= 150 tok/s, running at the same time as several *standard
lanes* at >= 75 tok/s per user, on the four Blackhole cards at TP4 on the S2 fast path (`c2-packed-tp4`, the 64-row M3 block, four
seats). All figures below come from this repository's own measurements and models (`docs/200tg-latency-budget.md`,
`docs/real-text-2026-09-24.md`, `scripts/ci/lanes_sim.py`); each is marked measured (m), derived (d) or estimated (e).

## What decides it, before any code

A lane's rate is `tau x (rounds the lane is in) / (frame time)`, where tau is the tokens a user commits per round. A frame is the
repeating pattern of rounds the scheduler runs. So the target is a statement about two things: the round times, and tau.

- **Round times (d/e).** F is a solo round (one user on a one-user 16-row block), P(n) a packed round with n live users. At TP4 the
  straight port has F = 67-74 ms and P(4) = 108-126 ms; the phase-1 exact levers (device-resident round, drafter fusion, collective
  fusion, value split, weight delivery, and for packed rounds host batching, a data-parallel drafter, a 2-way value split) bring
  them to F = 53-59 and P(4) = 85-102. The conservative anchors (full pessimism gaps, the quad draft and fused commit off in
  `c2-packed-tp4`, a 4 ms lane switch, no value split on the solo block) put phase 1 at F = 58-66 and P(4) = 93-111.
- **Tau (m).** The offline coding fixture commits 12.1 tokens per round. Real coding text committed **4.4-5.3** (the four v157-v160 runs, mean over four
  code tasks: pytest tests, review, explanation) and **7.1-9.5 on refactors** (`docs/real-text-2026-09-24.md`). The served image
  runs the bfloat8 drafter, which the fixture's 12.1 would lower to about 11.6 (e).

**Both bars hold exactly when one packed round plus one solo round fits in the time a standard user needs to write tau tokens at 75
tok/s:** `P(N+1) + F + 2 sigma <= 13.33 tau` ms (161 ms at tau 12.1, 133 at 10, 107 at 8). The minimum tau for both bars with one
fast user and one to three standard users:

| Levers | Mid band | Conservative anchors |
|---|---|---|
| Straight TP4 port + D0 | 12.1-15.1 | 14.5-16.8 |
| Phase 1 | 9.6-12.2 | 11.7-13.8 |
| All levers | 8.1-10.0 | 8.9-10.9 |

**On measured coding tau the two bars do not hold together with any lever set in the plans.** The standard bar alone (no fast
user, every round packed) needs tau >= 7.0-8.3 at phase 1 and >= 5.2-6.2 with all levers (conservative anchors, 4k to 131k), and code tasks sit at 4.4-5.3. What the
build does deliver is (1) **D0**: a lone coding user decodes at F-rate instead of on the 1/2/4-row engines, 152-208 tok/s at
tau 12.1 and 62-163 at tau 4.9-9.5 (conservative F, straight and phase 1), and (2) the lanes machinery and its gate, which measure tau at TP4 and
say exactly which lever or acceptance work moves the target.

## What is built (flags default off; gate-only profiles; TP2 pins untouched)

| Piece | Module | Flag |
|---|---|---|
| **D0**: a one-segment 16-row extent block over pool slot 0, built beside M3 at attach; a lone slot-0 user's rounds run on it | `serving_solo_lane.py`, `packed_shapes.solo_shape`, `serving_packed_step.py` | `QWEN_FAST_SOLO_LANE=1` |
| **Lane marking and admission**: `SamplingParams.extra_args['qwen_lane']` = `fast` or `standard`; exactly one fast lane on slot 0 (reserved); a second fast request is downgraded to standard, never queued; standard requests never take slot 0 | `serving_fast_lane.py`, `serving_buffer_pool.py` (slot order), `serving_lifecycle.py` | `QWEN_FAST_LANE=1` (needs D0) |
| **The lane gate**: the worker plans each round where every live request is known, drafts only its members and publishes the plan; a scheduler wrapper hides the other running decodes for that step (the plugin's own hide-and-restore, decode steps only) so the scheduled set equals the prepared set | `serving_fast_lane_scheduler.py`, `serving_worker_hook.py` | with the lane |
| **The controller**: the fast user rides every packed round and gets k solo rounds between them; k fixed (`QWEN_FAST_LANE_RATIO`), swept (`QWEN_FAST_LANE_SCHEDULE`, gate only) or the closed form on the measured round times, held by the standard floor; deficit counter for fractional k; at most `KMAX` solo rounds between packed rounds and a packed round at least every `GAP_MS`; the controller takes the standard lanes down to the floor (75 tok/s, less the margin) for the fast lane, never below it: a standard user's measured rate binds k, and a stall (a prefill) is not counted in it | `serving_fast_lane.py` | `QWEN_FAST_LANE_RATIO / _SCHEDULE / _KMAX / _GAP_MS / _FLOOR / _MARGIN / _RESERVE` |

Telemetry (server log): `[LANE-ADMIT]` per request, `[LANE-ROUND]` per executed round (kind, block, members, cycle ms, tokens per
member, ratio in force), `[LANE-FRAME]` per packed round, `[LANE-SCHED]` once per distinct scheduler state, `[LANE-DESYNC]` when a
scheduler the gate never ran on scheduled a request the round did not plan (refused by name before any device work).

**Exactness.** The lane changes which round a user is in, never what a round computes. The CPU tests drive the real hook, gate,
controller and packed step over a fake vLLM whose target is a pure function of each user's own history, and assert every user's
committed stream is the model's chain from its prompt whatever frame served it (every ratio, early drafting, tails, departures).
On hardware the gate's `lanes-exact` plan asserts it against each user's text alone on the per-request engines.

**Scheduling simulation.** `python scripts/ci/lanes_sim.py` runs the real controller on a virtual clock at the modelled round times
and prints both bars for every lever set and context; `test_fast_lane_hook` runs it through the whole stack.

## The gate plans and the window (`scripts/ci/references/tp4-lanes-jobs/`)

One image (`tp4-lanes-1`) for the whole window; every arm is a gate-only profile that is unqualified for traffic.

| Job | Mode | Plan and what it measures | Duration (e) |
|---|---|---|---|
| L0 build | stop | status, all-four reset with heal, image build | 10-30 min (cached), 40-90 cold |
| L0b lanes-attach-smoke | stop | all-four reset, the lanes profile's first four-card attach (D0's M=16 programs, the second block's trace region, the solo block's publication warm) and a two-test smoke: fails once, cheaply, before L1's long arms | 15-45 min |
| L1 lanes-exact | stop | every audit on: each lane's text, and D0's, equals that user's text alone on the per-request engines (3 arms: reference, D0, lanes with the ratio swept on the boot) | 35-100 min, up to 180 with a re-run |
| L2 round-timing | soft | audits off: P(4), P(3), P(2) and the lone user on today's engines in one boot per context (4k, 32k); D0 against those engines for a lone coding user (tokens per round and tok/s) | 45-100 min |
| L3 lanes-timing | soft | audits off: one fast user beside 3, 2 and 1 standard users at 4k and 3 at 32k, real text, the ratio swept from the high end down (k = 2, 1.5, 1, 0.5, 0, then the controller; a stretch counts only with the fast user in it); per-lane steady rates, P, F, sigma, tau per lane, MET/MISSED against 150 / 75 | 30-70 min |
| O1 lanes-timing-more | optional | 2 and 1 standard users at 32k, 3 at 120k | 25-60 min |

Required jobs L0-L3 (with L0b): about 2 h 15 min at best, 3 h 45 min expected, 7 h 30 min at worst. A missed bar is a result, not a failure: the timing
plans PASS when every arm was healthy and its cells were read, print the model's frame beside every measured stretch, and say
NOT_EXERCISED where a stretch had too few rounds. The hardware step also needs an image built from the window's own commit and the S2
TP4 window's packed-4 exactness (S3a) to have passed on the same kernels (image tp4-s2-2 or later, which carries the four-card collectives fix).

## Known limits (each is a decision or a measurement, not an oversight)

- **Real coding tau decides it** (above). The window measures it at TP4; a lookup or n-gram drafter aimed at copy-heavy code is the
  acceptance lever the model says matters most.
- **A lone standard user under the reserve runs on the per-request engines**: slot 0 is the fast user's, the padded block cannot
  hold one live user, and D0 is bound to slot 0. `QWEN_FAST_LANE_RESERVE=0` (lend) gives a standard request that arrives with nobody
  else live slot 0 first, so it decodes on D0; later standard users take slots 1-3, and a fast arrival while a standard user holds
  slot 0 is downgraded (`reason=slot-busy`) until it frees.
- **Prefill stalls every lane today** (no chunked prefill on the S2 TP4 path): steady rates exclude them, the report prints the
  client's wall-clock rate beside them, and Lever N (`docs/lever-N-prefill-decode-interleave.md`) is the fix.
- **More than three standard users** needs either a per-round carry swap (rotation) or the 128-row block; neither is built.
- **The hold mechanism is exercised on fakes shaped like the plugin's scheduler**, not on the pinned plugin's own source; the
  `[LANE] engaged`, `[PINDIAG] lane gate installed on ...` and `[LANE-DESYNC]` lines are how a served arm proves it ran.
