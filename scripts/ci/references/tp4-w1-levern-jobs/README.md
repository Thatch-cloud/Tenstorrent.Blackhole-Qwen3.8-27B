# tp4-w1-levern-jobs: window W-1, "Lever N exactness"

The first of the short windows (`docs/tp4-short-windows.md`): at most 9 h of outage, Lever N merged with prefix reuse judged on its own, nothing else in the image. The window image `tp4-serve-11` is the combined
branch's engine with the Lever N TRAFFIC profile `c2-packed-tp4-8x262k-ship-prefix-levern-traffic` baked in as its serving default (`B0-build-serve-11`, built while production serves). The window ends by
restoring the pre-cutover production image; the cutover is the last step of the next window (`tp4-w2-levern-jobs`). The order, classes, minutes, boxes, tags, dependencies and read rules are in `ORDER.txt`;
`test_tp4_stage1_windows` pins them.

## What it answers

| Job | Question | Why it is here |
|---|---|---|
| `A1-LN` | Does Lever N + prefix reuse attach, audited, and are its install lines, digests and epoch scope right (G-NP0, G-NP1, G-NP4)? | It is the kill signal, so it runs first: an attach death shows within minutes. |
| `S0-CTL` | With every Lever N flag off, is the window image byte-identical to production? | Production's hashes are the smoke JSON of tag v579 (run 37240205944); a mismatch ends the window, and no T0 job is needed. |
| `P1ab-LN` | Do exactness-shared and lifecycle-evict hold with Lever N on? | The two arms are ONE job (box 336 min, inside the prefix step's 380): one boot fewer and one reset fewer than two jobs. |
| `E1-LN` | Is a hit restored at a large Q and followed by many scratch steps equal to its cold twin at 262k? | The eager exactness arm has never run at 262k; admitted only if its box fits the cap, else it opens W-2. |

## Rules

- The first job is `A0X0`: agentstop, unserve, rescan, reset, in the workflow's fixed order and with NO `status` (production's container is still up and a status read would find it stale).
- A job starts only if `E + box + hand-back <= 540`; a job that does not fit moves to the head of the next window; nothing is cancelled. A box is the job's own timeout (see `ORDER.txt`).
- The hand-back is ZR (all-four reset), LM (links re-measure), TICK (the topology publish), Z (agent start) and the operator's release step with the owner's admin key. The key is never in a script and expires about two hours after issue.
- `prefix-reuse.off` and `levern.off` must be ABSENT on the hub mount before every job that opens a gate, prefix or smoke step.
- Tags are `experiment/c2-serving-vN`, hardware-allowlisted and never pushed; the map is in `ORDER.txt`. This pack takes `v558-v565` and keeps `v566-v568, v582` in reserve. Do not modify the runner group.
- One A1-LN re-run is allowed under a reserve tag with a swapped profile (lean, pool or epoch-global twin); a pass on a swap is a finding for the owner.

## How to run a job

Copy the template over `.github/c2-serving-job.env` on a throwaway commit of `tp4/stage1-windows` and push the job's tag. The private window driver does this one job at a time, reads each run's verdict from its
artifact and applies the admit-on-box rule; this repository holds only the public templates.
