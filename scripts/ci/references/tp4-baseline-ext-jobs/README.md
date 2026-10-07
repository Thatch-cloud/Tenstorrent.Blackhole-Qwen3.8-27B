# tp4-baseline-ext-jobs: the baseline extension window's job pack

A second block of tests chained directly after the baseline window (`tp4-baseline-jobs`) so the TT outage ends when the owner can attend the release, not in the small hours. It runs ONLY on today's production
ENGINE: the production base image `tp4-serve-10` and the production profile family `c2-packed-tp4-8x262k-ship-prefix` (the `-audit` twin where a gate needs the extent audit). SR10 additionally replays the
production thin layer. Nothing here builds an image, moves `:latest` or places a model. Order, classes, minutes, boxes, tags, dependencies and read rules are in `ORDER.txt`; `test_tp4_baseline_ext_window` pins them.

## What it answers

| Job | Question | Why it is here |
|---|---|---|
| `H1`, `H2` | Does the production traffic profile survive the eight-user hang shapes (warm-up, steady, resend, replay, split, drain), twice after a fresh reset? | Production went live without a hang-shape verdict on these bytes. |
| `SR10` | Does the node agent's serving sequence, running the production thin layer, boot, serve, restart and answer the thinking-budget smoke? | The platform path of production has never been replayed on cards. |
| `C16-CTL` | Does production hold 16 sessions of churn at eight seats with prefix caching on? | Skipped at the go-live. |
| `E1-CTL` | Is the sticky chain to the prompt limit and the 4095/4096/4097 boundaries equal to the cold twin at 262k page width? | The production path has never run exactness-eager at 262k. |
| `G0`, `T0` | Agent-turn replay and the timed production-bytes round: the reference numbers for later comparisons. | Data. |
| `R4`, `V1a`, `V1b` | The single-card V5 byte gate (only if the first window did not run it). | Optional. |

## Rules

- The jobs test the production ENGINE bytes, not the production image, except SR10, whose replay runs the thin layer. The private driver fills `@THIN_LAYER_IMAGE@` in SR10 (the public template carries only the placeholder); the operator's salted two-turn reuse check after the release stays the thin-layer check for reuse.
- The block starts only after the first window's hand-back reached `READY FOR /deploy`, and it stops the node agent that hand-back started (A0X0, no status step: production's container is gone, nobody released). It ends with its own hand-back (ZR, links re-measure, topology wait, kill-switch check, Z). The operator then runs the platform release step with the owner's admin key, which is never in a script.
- The driver admits a job only when NOW + its BOX + the 95-minute hand-back ends by the target wall clock (19:45Z). A job that was started is never cancelled; the workflow's smoke step cap (210 min) and replay step cap (180 min) are higher than the boxes, so a hung smoke job can overrun its box and delay the hand-back.
- Every reset renumbers `/dev/tenstorrent`; the agent refuses a TT load unless the links were measured after the newest entry, so the hand-back is reset with a rescan, links re-measure, wait for the topology file, agent start.
- Tags are `experiment/c2-serving-vN`, hardware-allowlisted and never pushed: v601-v613 planned, v614-v615 reserve. Do not modify the runner group.
