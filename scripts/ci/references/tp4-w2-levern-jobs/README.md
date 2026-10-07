# tp4-w2-levern-jobs: window W-2, "Lever N robustness, then the cutover"

The second short window (`docs/tp4-short-windows.md`): at most 9 h of outage on the window image `tp4-serve-11`, ending in the CUTOVER of Lever N + prefix reuse (the thin layer on `tp4-serve-11` becomes `:latest`)
when every cutover rule in `ORDER.txt` holds, else in the hand-back to the pre-cutover production image. Run it only after W-1 (`tp4-w1-levern-jobs`) passed in full. The order, classes, minutes, boxes, tags, dependencies, the
cutover rules and the rollback rule are in `ORDER.txt`; `test_tp4_stage1_windows` pins them.

## What it answers

| Job | Question | Why it is here |
|---|---|---|
| `S0b` | Does the baked default boot and serve the same bytes as the control? | The image's own default is what a platform launch serves. |
| `HL-LN`, `HL-LN2`, `HL-LN3` | Do the Lever N hang shapes complete three times in a row on fresh boots? | G-NP3: two partial prefills in the runner's persistent batch is the design's first-listed risk. A hang at 10% a boot is missed 73% of the time by three runs, so the rollback rule watches for hangs in production. |
| `HF-LN` | Do the fault drills latch `levern.off` with no restart? | The kill switch is the rollback tier that needs no key. |
| `L8-LN`, `C16-LN` | Concurrent equals solo past 131k, and sixteen sessions of churn? | G-NP7 and the reservation hold with the lever on. |
| `T0s`, `TA1`, `TL1`, `TA2`, `TL2`, `TA3` | Does Lever N's steady and 32k round cost sit inside the A-to-A floor, and is the window image the production bytes' speed? | Six timed jobs of two shapes (ABABA), ARC runners at zero. |
| `S1`, `S2` | Does the stall and TTFT line hold (a cold 254k arrival beside seven decoders, and two at once)? | G-NP5; the point of the lever. |
| `GG1`, `GG2` | Do agent turns and a hit arriving mid-cold-prefill pass their lines? | One G pair and one GH pair, in one prefix job each; the second pair is a later window. |
| `SR10` | Does the thin layer on the window image serve through the platform's own sequence, booting the baked default? | The last gate before `:latest` moves. |

## The cutover and the rollback rule

The cutover is ZR, LM, TICK, Z and the operator's release step with the owner's admin key (never in a script), then the live checks listed in `ORDER.txt`. The owner-approved rollback rule (2026-10-07) is in `ORDER.txt`
and `docs/tp4-short-windows.md`: Tier 0 image rollback, Tier 1 kill switches that need no key, Tier 2 the statistical read at 72 h and 7 days.

## Rules

- The first job is `A0X0` (no `status` step). A job starts only if `E + box + HB <= 540` with HB the cutover's upper bound, 75 min (the central figures are 45 for the cutover and 40 for the hand-back; a gate job adds the 30 min its reset may take); a job that does not fit moves to the head of the next window; nothing is cancelled.
- The ARC runner sets are at zero for the timed block, `S1`/`S2` and `GG1`/`GG2` only (170 minutes).
- Tags: this pack takes 21 tags after W-1's twelve and keeps four in reserve; seven allowlisted tags stay unused, and at least 52 more are needed before Stage 2 (68 with the same reserve of four a window) (see W-1's `ORDER.txt`).
- `SR10` reads the thin layer's image name from a file the operator writes on the runner (`C2_PLATFORM_IMAGE=local:thin-layer`); no registry name or digest is committed.
- `L8-LN` runs right after `S0b`: its box is the gate's own worst case, 368 min, which does not fit later in the window.
- No CI dispatch or PR push while a window runs.
