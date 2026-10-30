# tp4/stream-1: per-stream speed levers on the ship-5 stack

Branch `tp4/stream-1`, from `tp4/next-1` (the production ship-5 stack). One user's speed is capped by the tokens accepted per round (about 5.5 on code, about 4.2 on our own long sessions) over a round of
about 90 ms; this branch works the first factor from the host side. Everything is default off, byte-identical off, CPU-tested; **nothing here has run on a card**, and every gain quoted is an ESTIMATE
(the earlier simulation, short prompts) until the card plan below has run.

## 1. What is in it

| Piece | What | Flag / entry | Tests |
| --- | --- | --- | --- |
| Prompt lookup on the production path | a rebound request (engine reuse) attaches the host-side prompt lookup like a built one: `serving_request_factory.attach_lookup` after `rebind_parked`; the boot's refusal of the pair and the tau lab's refusal of a reuse base are lifted (a malformed policy is still refused at the boot) | `QWEN_FAST_LOOKUP_DRAFT=n3m12` (existing) | `test_lookup_reuse`, `test_parked_tp4_profiles`, `test_drafter_arms` |
| The shadow | one `[LOOKUP-SHADOW]` line per round (both candidates and the committed tokens); `tree_gain_report.py` replays them: what DFlash2 alone, the lookup alone and the better of the two would have committed on the same text | `QWEN_FAST_LOOKUP_SHADOW=1` (new; the lab's lookup arm sets it) | `test_tree_gain_report` |
| The paired decode rate | `drafter_pair_report.py` reads the lab's own turn timings: pooled completion tokens per decode second, control against candidate, with the cluster-bootstrap interval, beside (never inside) the GO rule | `speed` in the pair summary | `test_drafter_arms` |
| Tree verify, scoped | the design (`docs/tp4-tree-verify.md`) and a CPU prototype of the accept/commit logic and the state routing, plus the planner that gives branches only free segments | `tree_verify.py` (imported by nothing in serving) | `test_tree_verify` |
| Card jobs | the tau-lab pairs on the production profile and checkpoint | `scripts/ci/references/tp4-stream-1-jobs` | `test_tp4_stream_1_jobs` |

Docs: `docs/tp4-lookup.md` (sections "Beside engine reuse" and "The shadow"), `docs/drafter-arms.md`, `docs/tp4-tree-verify.md`.

## 2. The card plan (minutes)

Goal: in the shortest time on the cards, answer three questions about the lookup on the production stack and one about the tree. One image (B0 builds it from the pushed head while production serves),
two tau-lab pairs, no new profile (the lab derives its arms from the ship-5 profile at run time, so the profile registry and its census tests are untouched).

| Job | What | Cards | Answers |
| --- | --- | --- | --- |
| B0 | build the image of the branch head with the ship-5 default and the K64j-OQ graft (build host, production keeps serving) | 40 min, no card | |
| X0 | status, rescan, reset | 15 | the four boards, no holder, the grid |
| **TL0** | tau lab, `control` arm, ship-5 profile, production checkpoint, set A2 (our own sessions, thinking on), 8 in flight (four live seats, the M3 block) | 55 | the control's tau and decode rate on this stack and text |
| **TL1** | the same with the `lookup` arm (+ shadow) | 55 | **does the lookup attach on a rebound engine** (canary in the first minutes), tau ratio with its interval (GO rule), text equal on every paired turn (exactness on real text), **decode tok/s ratio** (`speed`), **TREEGAIN** (what a two-candidate verify would commit) |
| TL2 / TL3 | the same pair with ONE request in flight: the lone-user lane (NP1), the fast-lane shape | 45 + 45 | the lookup alone for a single stream; where a tree would live |
| Z | closing reset | 5 | |

**Fast path: X0 + TL0 + TL1 = 125 minutes of cards (165 with the build), then decide.** Everything runs with the ARC runner sets at zero (the s5 window's pause and restore). Push one tag at a time,
add `C2_TAULAB_PAIR_CONTROL=<run id of the control job>` to TL1 and TL3 (the reader refuses them without it). The lab holds the lookup arm to its markers (a canary that finds no engaged line stops the arm:
if that happens, the rebind path is the first suspect; the job's first six turns tell within minutes) and writes aggregates only: `tau-lab-summary.json`, `drafter-pair-summary.json`, `tree-gain-summary.json`;
the token ids of the SHADOW lines stay in the rig-local container log.

**Timed pair.** The lab's turns carry first-token time, seconds and completion tokens, so the pair report's `speed` line (paired decode tok/s, same text, same order, control against lookup) is the timed pair
at the lab's concurrency (8 in flight, and 1 for the lone user) with no extra boot. It is a within-run, same-stack comparison; it is not the smoke ABAB with the fusion reports. A smoke ABAB needs
two gate-only twins of the ship-5 profile (`...-fx-traffic-s5` with `QWEN_FAST_LOOKUP_DRAFT` off and `n3m12`, gate marker in place of the waiver id, the pf2 pattern) and about fifteen census-test exemptions
(every profile-enumerating test names the twin families); it was left out on purpose because the exemption lines are where this branch would collide with next-2, sched-2 and prefill-3. Build it only if the
`speed` pair is a GO and the owner wants the ship gates (T1A/T1B style: `np1_report.py`, `octo_compare.py`, `agent_turns.py`).

**Decision rules (written before the run).**

* *Lookup.* GO needs: `[DRAFTER-PAIR] arm=lookup verdict=GO` (pooled tau lower 95% bound above 1.00, no 80k+ or p10 regression, identical text on every paired turn) **and** the `speed` ratio's lower bound at or above
  1.00. A tau GO with a speed ratio below the tau ratio is the host cost of the lookup (the index build at the rebind, the proposal's host work): read `build_ms` in the engaged lines (rig-local) for the first.
  Any text difference between the arms is NO-GO until explained.
* *Tree.* Stop unless the `walk` gain over the run in `[TREEGAIN]` is at least 3% on A2 (docs/tp4-tree-verify.md section 7); build phase 1 only if it is 3-8% (recorded above 8%) and the lone-user lane is the
  next target; build D1 if it is above 8%. The checks on the line must read `dflash_replay_mismatches=0 continuation_conflicts=0 walk(run)=run True`, else the numbers are not to be used.
* *Fast lane.* The lone-user pair (TL2/TL3) is the lane's number: its `speed` ratio is what the lookup alone adds to the 61 tok/s of NP1; the owner's 150 is far from every figure in this branch.

## 3. Risks (card side)

1. **First card run of the lookup anywhere.** The lookup has never run on a card (docs/tp4-lookup.md). `RequestLookup` prints its round lines (`flush=True`); the engaged line goes through the plugin logger. If the
   engine-core subprocess's stdout does not reach the container log, the canary stops TL1 within the first turns and nothing else is lost.
2. **The tau lab has never run on the ship-5 profile.** Its two log flags (`QWEN_FAST_PACKED_AUDIT`, `QWEN_FAST_PHASE_TIMING`) are the lab's on every profile; the derived profile passes every contract check on the
   CPU (`test_tp4_stream_1_jobs`), but a boot with the audit flag beside octo, NP1 and exact resume is untested. TL0 shows it in its first minutes.
3. **The Swift checkpoint's tau.** The drafter (DFlash2, the base's) is not trained on Swift's text; the control arm is the reference, the pair is within-run.
4. **The index build is on the step thread** (about 0.07 s at the 32,768-token cap, estimate); a rebind otherwise costs 0.2-0.4 s. `build_ms` is in the engaged line; if it matters the cap is one constant.
5. **Exactness.** The lookup changes only the proposal rows; the target verifies every row. On real text the pair report's text gate is the check (identical output ids on every paired turn).
6. **Lab concurrency is not production's.** Four live seats at 8 in flight (the M3 block, no octo) and one: the two shapes where a free segment or a short block exists. The octo shape (>= 6 live) is not measured.

## 4. Merging onto next-2

Touched, with the likely collisions: `serving_request_factory.py` (one new function above `from_prefill`, two small hunks at its end), `serving_fast_policy.py` (a helper and one `if` in
`parked_engine_problems`), `c2_tau_lab.py` (the lookup arm's env, the `tree_gain` step in `analyze`, one removed refusal), `prompt_lookup.py` (opt-in shadow), `drafter_pair_report.py` (an additive `speed`),
`.github/workflows/qwen-integration-cpu.yml` (one line after `test_parked_tp4_smoke`). Everything else is new files. No profile was added, so `qwen_c2_profiles.json`, `profile_twins.py` and the census tests are
untouched. next-2's W3 WC and governor do not touch these functions; a conflict in the workflow line is a union.

The D-T5 templates of `tp4-drafter-jobs` and `tp4-next2-1-jobs` still say that a profile with engine reuse cannot carry the lookup arm: that sentence is superseded by this branch (the packs belong to other
streams and were left as they are).
