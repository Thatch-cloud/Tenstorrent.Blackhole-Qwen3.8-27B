# Combined split-K audit: timeout, not acceptance

Run **34920122480**, revision **58b127e**, ended on 2026-09-15 with request
subprocess status **124**. The container exited with status 1, not OOM. The
420-second request deadline fired before the correctness screen completed.
No clean device close or new committed-TG result is established.

| Observed phase | Evidence |
|---|---|
| Combined native build | Cache hit; 4 seconds |
| Model setup through decode warmup | 163.32 seconds before request audit |
| Two 65,536-token prefills | Both completed |
| First committed T16 block | 11 committed tokens; 30,159.75 ms draft, 6,579.37 ms selection/commit, 37,846.47 ms cycle |
| Request completion | Missing; deadline interrupted audit |

These block times are **instrumented correctness-audit times**, not a clean
performance regression measurement. `full_request.py` times the entire
`session.propose` call. With auditing enabled,
`PreparedDSparkProposal.propose` performs an additional eager execution,
copies normalized outputs, logits and tokens from both chips, executes the
trace, copies its outputs, and checks exact tensor equality. The current
timer does not separate those costs. This source inspection establishes
included work, not its individual measured contribution.

## Next bounded diagnostic

Keep every correctness check and the existing deadline. Add flushed,
per-phase host measurements around input/history update, eager reference,
reference snapshot, blocking replay, replay snapshot, equality checks and
token readback. Observe publication separately. Persist each completed phase
immediately so a timeout retains useful attribution. Do not fence new regions
or call those host intervals device-kernel measurements.

The existing `dspark_proposal_phase_profile.py` rejects audited proposals;
do not bypass that guard or silently reuse its non-audit timing route.
Use an explicit audit observer, with tests for exception propagation,
restoration, unchanged calls and unavailable/incomplete phases.

After the combined audit passes, admit a separately source-pinned clean
complete-request comparison. Follow the external review's order: split-K
local work versus tree reduction, then verifier recurrence/projections and
layout materialization. Preserve the distinction between provisional
attribution, component correctness, and combined PP / CTX / TG.

Evidence: artifact `qwen-splitk-combined-34920122480`, especially
`dspark-64k-request-hardware.exit-status`, `dspark-runtime-cache.json`,
`dspark-build-time.json`, `dspark-container-state.json`, and request log.
