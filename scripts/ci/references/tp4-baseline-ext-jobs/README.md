# tp4-baseline-ext-jobs: the baseline extension window's job pack

A second block of tests chained directly after the baseline window (`tp4-baseline-jobs`) so the TT outage ends when the owner can attend the release, not in the small hours. It runs ONLY on today's production
ENGINE: the production base image `tp4-serve-10` and the production profile family `c2-packed-tp4-8x262k-ship-prefix` (the `-audit` twin where a gate needs the extent audit). SR10 additionally replays the
production thin layer. Nothing here builds an image, moves `:latest` or places a model. Order, classes, minutes, boxes, tags, dependencies and read rules are in `ORDER.txt`; `test_tp4_baseline_ext_window` pins them.

## What it answers

| Job | Question | Why it is here |
|---|---|---|
| `H1`, `H2` | Does the production traffic profile survive the eight-user hang shapes (warm-up, steady, resend, replay, split, drain), twice after a fresh reset? | Production went live without a hang-shape verdict on these bytes. |
| `SR10` | Does the node agent's serving sequence, running the production thin layer, boot, serve, restart and answer the thinking-budget smoke? | The platform path of production has never been replayed on cards. CAVEAT: with no live container (the first window's A0X0 removed it) the replay copies the tracked 2026-09-26 container record, which carries the pair-era argv; a PASS certifies the thin-layer image and the production profile by name, not today's agent argv. |
| `C16-CTL` | Does production hold 16 sessions of churn at eight seats with prefix caching on? | Skipped at the go-live. |
| `E1-CTL` | Is the sticky chain to the prompt limit and the 4095/4096/4097 boundaries equal to the cold twin at 262k page width? | The production path has never run exactness-eager at 262k. |
| `G0`, `T0` | Agent-turn replay and the timed production-bytes round: the reference numbers for later comparisons. | Data. |

## Rules

- The jobs test the production ENGINE bytes, not the production image, except SR10, whose replay runs the thin layer. The private driver fills `@THIN_LAYER_IMAGE@` in SR10 (the public template carries only the placeholder); the operator's salted two-turn reuse check after the release stays the thin-layer check for reuse.
- The block starts only after the first window's hand-back reached `READY FOR /deploy`, and it stops the node agent that hand-back started (A0X0, no status step: production's container is gone, nobody released). It ends with its own hand-back (ZR, links re-measure, topology wait, kill-switch check, Z). The operator then runs the platform release step with the owner's admin key, which is never in a script.
- The driver admits a job by the dual rule against READY FOR /deploy (not the end of the deploy): NOW + its estimate + 48 min (ZR 15, LM 15, TICK 18) <= 19:45Z AND NOW + its box + 48 <= the hard cap 20:00Z. The 95-minute hand-back figure includes Z's container wait and the operator's deploy, which come after READY, and is for reporting. A job that was started is never cancelled; the boxes are admission worst cases (C16-CTL 70 and G0 40 are twice the measured worst, the gates' own timeouts are 153), and the workflow's smoke step cap (210 min) and replay step cap (180 min) are higher than the smoke boxes, so a hung smoke job can overrun its box and delay READY.
- The driver takes the cards only if production is NOT serving after the first window (no thatch-inference container, no answering endpoint) and no other window driver or qwen-c2-serving run is active; otherwise it halts and the first window's hand-back stands.
- READ rules for H1, H2 and T0 need positive evidence: every named test's result line, a SMOKE_CHECK line that is not FAILED, and a users: 8 aggregate for an eight-user test; a green run without them reads NO-VERDICT NOREAD.
- Every reset renumbers `/dev/tenstorrent`; the agent refuses a TT load unless the links were measured after the newest entry, so the hand-back is reset with a rescan, links re-measure, wait for the topology file, agent start.
- Tags are `experiment/c2-serving-vN`, hardware-allowlisted and never pushed: v601-v610 planned, v611-v615 reserve (the driver uses v611 and v612). Do not modify the runner group.
