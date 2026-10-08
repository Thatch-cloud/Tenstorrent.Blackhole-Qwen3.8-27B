# tp4-session-0808b-jobs: the second chained card session of 2026-10-08 (performance and robustness, no digests)

The first session of the day (`tp4-session-0808-jobs`) chains Lever N exactness and robustness, and its audited arms (`QWEN_PREFIX_DIGESTS`, the lever audits, the prefix audit) are followed after every request by a whole-pool host digest
of 5 to 8 minutes at 8 x 262k (`docs/prefix-audit-cost.md`), so they cannot finish inside their boxes and the session hands the cards back early. This pack uses the idle cards from that hand-back to READY with the work that needs
**no digest and no audit**: every profile it names is the production profile (`c2-packed-tp4-8x262k-ship-prefix`) or the Lever N traffic profile the image bakes in (`c2-packed-tp4-8x262k-ship-prefix-levern-traffic`), so a request costs what
production's costs. The order, classes, minutes, boxes, tags and dependencies are in `ORDER.txt`; `test_tp4_session_0808b` pins them. Never run it beside the first session: they share the cards and the first pack's tags.

## What runs, in priority order

| Block | Jobs | Class | Why |
|---|---|---|---|
| attach | `A0X0` (the agent stops again: the first session's hand-back restarted it), `S0b` (baked default smoke, hashes equal to the production reference) | stop | the kill signals |
| hang shapes | `HL-LN`, `HL-LN2`, `HL-LN3` (three fresh boots, Lever N hang shapes, the skewed eight) | stop | robustness core |
| performance | `T0s` (production bytes), `TA1`/`TL1`, `TA2`/`TL2`, `TA3` (ABABA, steady and 32k eight-live rounds) | opt block | Lever N cost against the A-to-A floor |
| client shapes | `SM` (streamed tool call, streamed reasoning, steady resend, the platform's eight requests, drain) | soft | the agents' real request shapes |
| stall | `S1`/`S2` (seven decoders and one cold 128k or 262k arrival, two simultaneous 254k arrivals) | opt block | the stall Lever N removes |
| agent turns | `GG1`/`GG2` (eight coding agents of eight turns, a hit arriving mid-cold prefill) | opt block | prefix reuse under Lever N |
| late attach | `S0b2` (the S0b smoke again after the whole session's load) | soft | boot repeatability |
| deep | `DA1`/`DL1` (eight users at 32k, then eight at about 120k) | opt block | depth, the shape the 262k window exists for |
| platform replay | `SR10` | soft | needs the thin layer staged on the runner; listed in `SKIP_JOBS` until it is |

Dropped from the first pack: `KVQ`, `A1-LN`, `S0-CTL`, `L8-LN`, `C16-LN`, `P1ab-LN`, `P1-CTLR`, `P1b-CTL`, `E1-LN`, `HF-LN` (audited profiles or the audited fault drill) and the cutover. The end is always the hand-back.

## Clock (UTC)

The launch wrapper turns the absolute times into minutes from the driver's start (`A0X0` = minute 0): READY for `/deploy` 19:30Z, hard cap 20:30Z, the last job ends 60 minutes before 20:00Z. A job starts only if
`E + box + 60 <= CAP`; an opt block is admitted or refused whole on the sum of its boxes. At the central estimates (a start at 11:45Z) the 19 jobs that run take 330 minutes and READY lands near 17:35Z, about two hours inside the target. The boxes, not the
estimates, decide admission, so a start up to 60 minutes later (12:45Z) still runs everything; later starts refuse the blocks whole, last first: the deep pair from 62 minutes late, the agent-turn pair from about 2 h,
the stall pair from about 3 h (the test pins the walk).

## Rules

- ARC runner sets at zero for the timed block (`T0s` to `DL1`, about four hours): the operator scales them before `T0s` and creates `runs/<id>/ARC_ZERO_ACK`; without the file the block is SKIPPED-ARC after the wait and the session hands back early.
- No CI dispatch, PR push or other rig workload while the session runs. `prefix-reuse.off` and `levern.off` must be absent on the hub mount (the operator holds a production switch aside and the hand-back restores it).
- The tags are `experiment/c2-serving-v582` to `v617` as listed in `ORDER.txt`, the reserve `v618`-`v620`. All are allowlisted; do not modify the runner group.
