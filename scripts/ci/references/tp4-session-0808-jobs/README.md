# tp4-session-0808-jobs: the chained card session of 2026-10-08

One outage that chains the two short windows of `docs/tp4-short-windows.md`: W-1 (Lever N exactness) and W-2 (robustness, then the cutover or the hand-back), with the audit op's
qualification first, ONE `A0X0`, no intermediate hand-back and the end decided by the cutover rule. Owner, 2026-10-08 03:40Z: "prep now ready for 12+ hours of testing".
The order, classes, minutes, boxes, tags, dependencies and directives are in `ORDER.txt`; `test_tp4_session_0808` pins them. The packs `tp4-w1-levern-jobs` and `tp4-w2-levern-jobs` are the
two-window route and stay as they are: they share tags with this pack, so never run either beside it.

## Clock (UTC, the start at 07:30Z is minute 0)

| Time | Minute | What |
|---|---|---|
| 07:00Z | -30 | go/no-go (the checklist in the private `RUNBOOK-SESSION-0808.md`) |
| 07:30Z | 0 | `A0X0`: the agent stops, production goes down |
| 18:30Z-19:20Z | 660-710 | the last job ends by minute 690; READY for `/deploy` follows its ZR, LM and TICK |
| 19:30Z | 720 | READY target: the owner's admin key is needed here and only here |
| 20:30Z | 780 | hard cap (a cutover adds about 30 minutes of thin-layer check and publish after the last job, minute 740) |

A job starts only if `E + box (+30 for a serving-gate job) + 60 <= 750` (`CAP`, `RESERVE-MIN` in `ORDER.txt`). Nothing is cancelled.

## What the plan says at the central estimates (the test and the driver's dry run print the same walk)

- 14 of 26 planned jobs run, 642 minutes: `KVQ`, `A1-LN`, `S0-CTL`, `S0b`, `L8-LN`, `P1ab-LN`, `E1-LN`, `P1a-CTL2`, `C16-LN`, `HL-LN` x3, `HF-LN` (and `A0X0`). READY at about 18:32Z.
- 12 jobs are refused for time: `S1`/`S2`, `P1b-CTL`, `GG1`/`GG2`, `SR10` and the six timed jobs (270 minutes). The cutover rule (`CUTOVER-NEEDS`, owner-approved) needs those, so **the session ends in the hand-back
  to `f1dfcc99`, and the cutover goes to a short window** (about 4 to 5.5 h with its own `A0X0`; `results/CARRYOVER.txt` lists the jobs). The full cutover battery is 777 minutes of jobs against 690 available.
- If every job runs to 1.5 times its estimate the exactness core (`P1ab-LN`, `E1-LN`) still runs and the robustness core is cut short at `HL-LN`; READY then lands about 19:06Z.
- If `KVQ` fails, `P1ab-LN`, `E1-LN` and `P1a-CTL2` are skipped (their NEEDS), and the freed time runs `S1`/`S2`, `P1b-CTL`, `GG1`/`GG2` and `SR10`; the timed block misses by two minutes. No cutover (`P1ab-LN` and `E1-LN` are in `CUTOVER-NEEDS`).

Ways to reach the cutover in one session, each an owner decision and none applied: raise READY and the hard cap by about two hours (the battery is 777 job-minutes plus the 4-minute `A0X0` against 690 available), or cut jobs from
`CUTOVER-NEEDS` (the owner-approved rule; the cheapest honest cut is E1-LN, 100 minutes, which W-1's README allows to be carried, but that moves the cutover to the next window by that very rule). None is needed to run the testing.

## The audit op, the controls, the judge

- `KVQ` runs `optimisation/ttnn-op/kv_region_read/kv_region_read_card.py` on the four-card mesh inside `tp4-serve-11` (the fabric step's `kvread` probe, `scripts/ci/tp4_kv_read_probe.py`). `P1ab-LN`, `E1-LN` and `P1a-CTL2` NEED it.
  The op is not compiled yet (docs/prefix-audit-cost.md, "Delivery"): `tp4-serve-11` must be rebuilt with it before the session (B0), or the **fallback** pack conf (`packs/tp4-session-0808-fallback.conf`, private) skips `KVQ` and the three audited jobs.
- `P1a-CTL2` is W-0's P1a-CTL re-run on the window image with Lever N off (the production base has no region read); `P1b-CTL` is W-0's P1b-CTL on `tp4-serve-10` with the fixed lifecycle traffic. Both are `CUTOVER-WAIVABLE`: when they did not
  PASS, the cutover needs the line `WAIVE P1CTL` in the `CUTOVER_OWNER_GO` file, which the owner writes with the evidence in front of them. A FAIL is a production finding either way.
- `scripts/ci/session_cutover_judge.py` reads the timed block (Lever N's cost inside the A-to-A floor at steady and 32k, the production bytes inside TA1's floor; load-voided pairs are never "inside").

## Rules

- `CUTOVER_OWNER_GO` (a file in the run directory, or the environment) is the owner's recorded go; without it the window hands back. The owner is asleep at 07:30Z: it is decided at the 07:00Z go/no-go or not at all.
- `prefix-reuse.off` and `levern.off` must be absent on the hub mount; the operator holds a production switch aside with `rig_telemetry.sh ks-hold` and the hand-back restores it before the agent start.
- No CI dispatch, PR push or other rig workload while the session runs (a timed job read against that load is void).
- The tags are `experiment/c2-serving-v558` to `v605` (29 jobs with a tag, `B0` included), the reserve is `v606` to `v611`, `v612` to `v620` stay unused. All are allowlisted; do not modify the runner group.
