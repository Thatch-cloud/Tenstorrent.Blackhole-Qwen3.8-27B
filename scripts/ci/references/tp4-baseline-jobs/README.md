# tp4-baseline-jobs: the baseline development window's job pack

A SHORT outage (target 6-8 h, hard cap 9 h) that runs ONLY on today's production: the production base image `tp4-serve-10` and the production profile family `c2-packed-tp4-8x262k-ship-prefix`
(the `-audit` twin where a gate needs the extent audit: the same engine and env plus audit flags). It runs BEFORE the big combined window (`tp4-w2ln-jobs`) and builds nothing: no window image, no
thin layer, no `:latest` move, no placement. The window stops the node agent (A0) and starts it again (Z); production's image is untouched, so its rollback is "start the agent". The order, classes,
minutes, boxes, tags, dependencies and read rules are in `ORDER.txt`; `test_tp4_baseline_window` pins them.

## What it answers

| Job | Question | Why it is here |
|---|---|---|
| `P1a-CTL`, `P1b-CTL` | Is production's packed prefix reuse EXACT (eight agents sharing blocks) and does its lifecycle (abort, eviction, kill switch, restart) hold? | Production's prefix reuse has never had a verdict on cards (cancelled at 84 min once, at go-live once). |
| `L8-CTL`, `C16-CTL` | Does production hold the strict ladder past 131k and 16 sessions of churn at eight seats? | Both were skipped at the go-live. |
| `PROF-CTL` | Where do the eight-seat round's device microseconds go on the production profile? | No production profile has ever been profiled. |
| `V1a`, `V1b` (optional) | Does the single-card V5 byte gate pass (K5-A against V5 bit for bit)? | Its only earlier run died on a mesh-descriptor error since fixed in the harness (`QWEN_C2_SERVING=0`, mesh descriptor unset). |

## What does not fit

The four groups together are 11.1 h of jobs by estimate, and with the real hand-back (95 min: reset 15, links re-measure 15, topology publish up to 18, agent start 32, release and engine load 15) none of the combinations
with the profile or the V5 block fits the 9 h cap. The PLANNED window is the CORE (A0, X0, P1a, P1b, L8, C16 and the hand-back): 440 min = 7.3 h. CORE plus PROF-CTL is 570 min and CORE plus the V5 block (R4, V1a, V1b: 135 min)
is 575 min, so both run only when the jobs before them ran short (the driver's clock rule: a job past the 8 h target starts only if its BOX fits the cap). The owner can instead pick, before the window, `SKIP_JOBS=C16-CTL-churn16 TARGET_MIN=540`
for the profile (525 min) or `SKIP_JOBS="L8-CTL-ladder8-past-131k C16-CTL-churn16 PROF-CTL-ops-profile-8x4k" TARGET_MIN=540` for the V5 block (485 min); `TARGET_MIN=540` makes the clock trust the estimates up to the cap, and a job that runs to its box can still pass it.
PROF-CTL profiles the UNSALTED path (no `C2_GATE_SALT`): production's salted capture and publish work is not in the profile.
The 32k shape of the device profile is NOT in `PROF-CTL`: `ops_profile_plan` has only the 4k shape (docs/tp4-profile.md: the exactness divergence at 32k and above is unresolved) and op-support 20000 loses the
packed rounds at 32k (v133); a 32k profile needs a plan change and its own review.

## Rules

- The control jobs test the production ENGINE bytes, not the production image: they serve the base image under production's thin layer, so the layer's request path (the Thatch wrapper, idle admission, prefix metrics, its entrypoint) is not exercised by a gate. The operator's salted two-turn reuse post-check after the release is the thin-layer check.
- Every reset renumbers `/dev/tenstorrent`. The agent refuses a TT load unless the links were measured after the newest entry, so the hand-back is: reset with a rescan (ZR), links re-measure (LM, a Thatch.Server workflow
  the driver dispatches), wait until the topology file shows the new links (TICK, polled; the timer has no fixed clock grid), agent start (Z, which waits up to 30 min for the release). Then the OPERATOR runs the platform release step with the owner's admin key. No script holds the key.
- One-card jobs (V1a, V1b) need an all-four reset first after the quad runs (R4). Their harness mounts this checkout's scripts into the production base image; nothing is built.
- Never cancel a profile job: the cancel skips the hand-back step and leaves root-owned logs under the runner's work directory (the next checkout dies with EACCES). op-support is 20000; 200000 segfaults the dispatch thread
  and `ops_profile_plan` refuses it.
- P1a-CTL's box is the gate's raised 10,800 s. A FAIL of any control is a PRODUCTION FINDING: the driver prints an ALERT and goes on (production is already down); the owner decides afterwards about the kill switch.
- Tags are `experiment/c2-serving-vN`, hardware-allowlisted and never pushed; the map is in `ORDER.txt` (v538-v549 planned, v550-v557 reserve). Do not modify the runner group.

## How to run a job

Copy the template over `.github/c2-serving-job.env` on a throwaway commit of `tp4/baseline-window` and push the job's tag. The private window driver does this one job at a time, reads each run's verdict from its artifact and
applies the clock rule; this repository holds only the public templates.
