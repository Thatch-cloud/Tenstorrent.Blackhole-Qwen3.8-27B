# Short windows: Lever N + prefix reuse first, in outages of at most 9 hours

**Branch:** `tp4/stage1-windows` (on `tp4/w2-levern-prefix`). **Packs:** `scripts/ci/references/tp4-w1-levern-jobs` (W-1) and `scripts/ci/references/tp4-w2-levern-jobs` (W-2).
**Status:** built and CPU-tested; nothing here has run on cards. The baseline window (W-0, `tp4/baseline-window`) runs first and is not touched by this branch.

## Why short windows

The combined pack (`tp4-w2ln-jobs`) is one outage of 63.6 hours that ships nothing until it ends, and the cutover is a separate pack after it. The measured times are far below the pack's constants: an audited attach takes
2 to 5 minutes (not 15 to 19), and a five-shape timed job 9 to 18 minutes (not 130). The plan therefore splits the work into windows of at most 9 hours each, in value order, and puts Lever N with prefix reuse FIRST:
W-0 the baseline on today's production bytes, W-1 Lever N exactness, W-2 Lever N robustness and the cutover (live after about 21 hours of outage), then research windows W-3 to W-6 and a conditional integrated cutover.

## The two windows

| | W-1 "Lever N exactness" | W-2 "Lever N robustness, then the cutover" |
|---|---|---|
| Image | `tp4-serve-11` (the combined branch's engine, the Lever N traffic profile baked in) | the same |
| Jobs | A0X0, A1-LN (attach, the kill signal), S0-CTL (flags off equals production), P1ab-LN (exactness-shared and lifecycle-evict in one job), E1-LN (eager exactness, only if its box fits), hand-back | A0X0, S0b, L8-LN, HL-LN x3 on fresh boots, HF-LN, C16-LN, the timed block (T0s, TA1, TL1, TA2, TL2, TA3), S1/S2, GG1/GG2, SR10, then the cutover or the hand-back |
| Outage | 471 min = 7.9 h | 445 min = 7.4 h |
| Ends | restoring production | in the cutover, or restoring the pre-cutover production |

Rules for every window: the first job is `A0X0` (agentstop, unserve, rescan, reset, no `status`: production's container would read as stale); a job starts only if `elapsed + its box + the hand-back's upper bound <= 540` (60 min in W-1, 75 in W-2; the central figures are 40, 40 and 45), else it moves to
the head of the next window; nothing is ever cancelled (cancelling lost a board's by-id link once).

## The traffic profile

`c2-packed-tp4-8x262k-ship-prefix-levern-traffic` is the production profile plus the Lever N flags, with no `gate_only`, no evidence waiver, no gate marker and no gate instrument. The contract
(`serving_c2_contract.levern_problems`) now accepts the Lever N master switch on a traffic profile only on the merged route (prefix reuse and sticky sessions both on) and without the digest audit or fault flags; the
stage-1 shape and the instruments stay gate-only. The image `tp4-serve-11` bakes it as its serving default (`C2_BAKE_DEFAULT_PROFILE`, names up to 64 characters), so a platform launch serves it, and the kill switch
`levern.off` on the hub mount turns the lever off with no restart. The audited gate twins (`-levern-audit`, `-levern-audit-nolna`) stay gate-only.

## Tooling changes

- **Job boxes (`C2_BOX_MINUTES`).** A template names its box; the workflow enforces it from inside the step and never cancels a run: the smoke step stops waiting for the API at the box and runs its client under `timeout`,
  the prefix gate takes `--box-seconds` (every arm's docker timeout is clipped to what the box has left, an arm with under five minutes left is not started), the serving gate refuses before any container a plan its box cannot
  hold (its worst case counts every re-run: L8-LN is 368 min, not 186), and the replay runs under an interrupt timeout so its own cleanup removes its container. Without a per-job box the admit-on-box rule would skip S2, GG1 and two timed jobs late in a window, because the smoke step's
  box is 210 minutes and a G plus GH job's is 306.
- **Heal wait 10 s.** In 117 resets the by-id links were back at 0 s (74) or still missing at 90 s (43), never in between; the reset step now waits 10 s before it re-probes.
- **Readiness poll 2 s.** The smoke step polls the API every 2 s instead of 15 s (the wait stays 80 minutes).
- **Box outcomes are greppable.** A box that cuts a job short prints `C2_BOX verdict=TIMEBOX step=...` (a prefix arm it cuts short is NOT_EXERCISED with `boxed`, never FAIL); a refused serving-gate plan prints `C2_BOX verdict=REFUSED step=gate`.
- **Judge dry-run.** `scripts/ci/stage1_judge_dry_run.py` re-judges archived artifacts (the smoke check and the prefix gate's arm judge) with the committed rules and reports any rule-only failure.

## The cutover and the rollback rule

The cutover needs every Lever N gate of W-1 and W-2, the timed read (Lever N's cost inside the A-to-A floor at steady and 32k), one S/G/GH pair, the platform replay, W-0's controls neither FAIL nor NO-VERDICT, and the
owner present with a fresh admin key. After it (owner-approved 2026-10-07):

- **Tier 0, image rollback** to the pre-cutover image (15 to 25 minutes plus the owner's key): one engine death or container restart; a hang (no `[PHASE] execute` line for 300 s with a request live); server errors of at
  least 3 in an hour, or more than max(1%, baseline + 3 sigma) over 24 h.
- **Tier 1, kill switches, no restart and no key:** `levern.off` when any request over 128k has a TTFT above 240 s or any Lever N quarantine line is logged; `prefix-reuse.off` when the prefix grant ratio drops more than
  25% below its baseline over 24 h.
- **Tier 2, statistical, read at 72 h and finally at 7 days:** the round time per (32k or 128k window, live count 5 to 8) with at least 100 load-matched rounds a side, worse by more than max(3%, floor); the TTFT p90 for
  prompts up to 32k with at least 500 requests, worse by more than max(20%, twice the day-to-day spread).

## Tags

The free allowlisted tags after the baseline window's `v538-v557` are `v558-v568`, `v582-v585`, `v589-v592`, `v595-v599` and `v601-v620` (44). W-1 takes 8 and keeps 4 in reserve, W-2 takes 21 and keeps 4, and 7 stay
unused. The combined pack's own tag list (`tp4-w2ln-jobs`, v538 to v640) overlaps these and is superseded for the windows covered here. Stage 2 (W-3 to W-6) needs 59, so at least 52 more must be allowlisted before it (68 with the same reserve of four a window). Adding tags is a runbook step in the private platform repository.
