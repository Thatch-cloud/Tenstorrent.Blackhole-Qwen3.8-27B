# The W2 runtime kill switch (`w2.off`)

W2 is `QWEN_FAST_TP4_SDPA=multi` (one SDPA launch for every user of a packed block) plus `QWEN_FAST_TP4_CONV_GATES_SPREAD=1` (F1, the conv-gates launch on spread
gate cores). Lever N has `levern.off`, prefix reuse `prefix-reuse.off` and engine reuse `parked.off`, each a file under the hub mount's `.qwen-c2` directory
that stops the lever with no restart. W2 had none, so any W2 problem meant an image rollback. `w2.off` is the fourth file, and it works differently, because W2
cannot be edited out of a trace.

## Why this one is not a flag read per round

Both W2 levers are baked into the packed blocks' verify traces. The trace is recorded once, at the attach, and a replay re-issues the device program and never comes
back to Python. A second trace without W2 would need a second capture: trace region and DRAM that production does not have free, in code no card has run. So the
switch acts only on the host, only between traces, at two points. `scripts/ci/w2_switch.py` is the whole of it (stdlib only, host only).

## When it takes effect, exactly

1. **Live, at the next round decision (about one round).** The file appears. The next question a round asks (`serving_packed_step.proposal_rows` and
   `block_rows_for` before the drafts, `ineligible` at the step) polls the file (at most one `stat` every 0.25 s), latches, and logs once
   `[PINDIAG] w2 kill switch <path> present: packed rounds on the W2 blocks take the exact sequential step ...`. From then on every round on a 16-row block goes to
   the sequential step whole, the existing fallback for a round the block cannot serve: the served SDPA and the unspread conv gates, each user on its own
   per-request engine, tickets drafted at the engines' own widths. The round whose trace is already replaying finishes untouched. Greedy output is byte-identical
   (a packed round and a sequential round are exact against each other; that is what the packed gates prove). It is **slower**: a brown-out to get safely off W2, not
   a mode to stay in. The latch lasts for the life of the process; removing the file does not bring the packed block back.
2. **At the next attach.** Restart the engine with the file still present and `sdpa_long_tp.apply` leaves `reader.multi` None and `gdn_block_conv_tp.stage` runs the
   served conv-gates call: the packed blocks are captured **without W2** and serve at full packed speed on the pre-W2 paths. One line says so
   (`... present at attach: W2 is not attached`). Removing the file and restarting brings W2 back. An audit flag (`QWEN_FAST_TP4_SDPA_AUDIT`,
   `QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT`) is a gate's own and cannot be skipped: with one set the attach ignores the file (one line) and only the live latch applies.

The octo block (eight-row users) runs no W2 and is never routed away.

## Byte-identical when the file is absent

A process whose environment names neither W2 lever never makes a system call for the switch. With W2 on and the file absent the only cost is one `stat` at most every
0.25 s from the round decisions; no round changes. `QWEN_FAST_W2_OFF_PATH` moves the file (default `/models/.qwen-c2/w2.off`; an empty value disables it), like
`QWEN_FAST_LEVERN_OFF_PATH` and `QWEN_FAST_PARKED_OFF_PATH`.

## The drill (gate only)

`QWEN_FAST_W2_OFF_AFTER=n` (the contract refuses it outside a gate-only profile, without W2, or without a scratch `QWEN_FAST_W2_OFF_PATH`) makes the server write the
flag file itself after `n` packed rounds, so a card job needs no operator mid-run and the real file path is the one exercised. The profile is generated:
`c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-kill` = the production profile as a gate-only twin plus the two names (`scripts/ci/make_w2_kill_profiles.py`).

`c2_smoke_check.w2_kill_problems` judges the log. Outside the drill arm any kill line is a problem (a leftover file would make the arm run without W2 and read clean;
the gate job also refuses to start with a `w2.off` on the hub mount). In the drill arm: one drill line, one latch line after it, `[PINDIAG] packed extent round` lines
before the latch and **none after it** (every packed round replays a trace that carries W2: the W2 markers stop), a routed line per block after it.

## The card jobs (`scripts/ci/references/tp4-w2-kill-jobs`)

`K-W2a` runs the W2-off control (`...-levern-parked`: Lever N + prefix + engine reuse) on `warmup,concurrent8_steady,concurrent8_code_equal,coding`; `K-W2b` runs the
drill profile on the same tests, the kill landing about half way through `concurrent8_code_equal`. Then `python3 scripts/ci/levern_compare.py --control <K-W2a smoke log>
--levern <K-W2b smoke log>` must exit 0: every stream's text, token count and finish reason equal, the packed rounds before the latch and the sequential ones after it
included. Not a timing arm; every result is UNQUALIFIED.

## Operator

`touch <hub mount>/.qwen-c2/w2.off` (the file's content is not read). Watch for the latch line; production is then slower but exact. Restart the engine to get full
packed speed without W2. Remove the file before the next gate job.
