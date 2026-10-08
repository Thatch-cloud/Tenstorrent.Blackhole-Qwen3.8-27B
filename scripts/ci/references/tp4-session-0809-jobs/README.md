# tp4-session-0809-jobs: the third card session of 2026-10-08/09 (functional tests with CI running)

The owner asked for about three more hours of tests with the production TT down, **CI left running** (no ARC runner scale-down), READY for `/deploy` by 22:30Z and a hard cap of 23:00Z. This pack therefore has **no timed block** (no `T0s`, `TA`, `TL` jobs)
and **no ARC gate** (no `ARC_ZERO_ACK`, no ARC-ZERO directive). **Every result is functional; any timing in it was taken beside the rig's CI load and is noisy** (each job's `load.log` says how noisy). Read hashes, completions, gaps and hangs, not tok/s.
The performance claim stays with the timed block of `tp4-session-0808b-jobs`, for a later quiet window.

## Why it exists

s0808b's `HL-LN` (v584, run 37773733305) failed its smoke check: two prefill steps of one request ran back to back with decoders running, "the alternation did not yield" (the steps had `reason=paid owed_ms=0 owed_rounds=0` and no decode step between them).
`HL-LN-DIAG` runs that hang shape again on the same image with the debug logs the code offers (see its template header: `QWEN_FAST_SEQ_STAGE_LOG`, `VLLM_LOGGING_LEVEL=DEBUG`, `QWEN_PREFIX_STATS_S`, handed in through the new `C2_SMOKE_ENV` job key, which allows only diagnostic log switches).
No digest, no audit, no gate-only profile.

## What runs, in priority order

| Job | Class | Why |
|---|---|---|
| `A0X0` | stop | the agent stops, the cards are reset (the kill signal) |
| `HL-LN-DIAG` | soft | the failing hang shape with the debug logs on; a failure is the expected finding |
| `SM` | soft | the agents' real request shapes (streamed tool call, streamed reasoning, steady resend, replay of eight, drain) |
| `S1` / `S2` | opt block | seven decoders and a cold 128k/262k arrival, two simultaneous 254k arrivals: control against Lever N |
| `GG1` / `GG2` | opt block | eight coding agents of eight turns, a hit arriving mid-cold prefill |
| `DA1` / `DL1` | opt block | eight users at 32k then at about 120k |
| `S0b2` | soft | the baked-default smoke again, late (hashes report-only) |
| `SR10`, `HL-LN-DIAG2` | owner-skipped | block separators; `SR10` needs the thin layer, `DIAG2` is a second diagnostic boot to un-skip if the first reproduced the failure |
| `ZR`, `LM`, `TICK`, `Z` | hand-back | the end is always the hand-back to the production image; `/deploy` is the operator's, with the owner's key |

## Clock (UTC)

The launch wrapper turns the absolute times into minutes from the driver's start: READY 22:30Z, hard cap 23:00Z, soft deadline 22:40Z, end-sequence reserve 40 minutes (the last job may end by 22:00Z for a 19:50Z start). A job starts only if `E + box + 40 <= CAP`;
an opt block is admitted whole on the sum of its boxes. At the central estimates the admitted set is `A0X0`, `HL-LN-DIAG`, `SM`, `S1`/`S2` and `S0b2` (the agent-turn and deep pairs are refused whole for their boxes; they run only when earlier jobs finish well inside their estimates).

## Rules

- Cards: nothing else may open a card while the session runs; CI that does not touch the cards may run. `prefix-reuse.off` and `levern.off` must be absent on the hub mount (the operator holds a production switch aside and the hand-back restores it).
- Tags `experiment/c2-serving-v585`, `v589` to `v603`, `v612` as listed in `ORDER.txt`, reserve `v613`-`v615`; all allowlisted; do not modify the runner group.
