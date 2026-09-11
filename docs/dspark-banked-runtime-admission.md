# Banked drafting: combined-runtime admission

Status: experimental request option implemented; no hardware performance result yet.

The candidate binds one captured draft graph to each existing history bank. After
publication swaps banks, it selects the matching graph instead of copying all
history into a third bank. At capacity 4352, the avoided destination payload is
42.5 MiB per chip per proposal; this is not a measured time saving.

| Evidence | Result | Does not prove |
| --- | --- | --- |
| Revised TTsim bank lifetime probe | 80 output and 200 bank checks exact; both graphs replay twice | Learned drafting or speed |
| Focused host tests | 27 pass | Accelerator correctness |
| Combined learned request | Not run with banked candidate | Any PP or TG improvement |

Simulator report: `/opt/ttsim/results/20260911T002837Z-443-dspark-banked-trace-probe.json`.
SHA256: `967dcbee569e215993e5fd29d2cadcba85a51f5f6a34137e67b9502bc832f248`.
The independent gate checks every chip, operand, bank transition, post-close
readback, replay audit, exit status, and current source digest. It deliberately
labels this synthetic evidence. The active-trace allocation warning still occurs
during eager auditing; learned request checks must establish whether it is safe.

`measure_dspark_request` accepts explicit `banked_proposal=True` and a
`banked_proposal_evidence` report path. It requires the combined traced native
drafter, commit-only GDN, and folded T16 target path. Defaults are unchanged.
Reports include per-bank replay counts. No workflow currently selects this option.

Next: retain the simulator evidence for CI, wire a distinct matched banked/control
arm, and run learned feature/history/proposal/target-state audits before timed
A/B/B/A requests. Keep score-layout selection identical in both arms. Report
PP / CTX / committed TG and setup-inclusive latency; require both banks to execute
and preserve coding functional tests. Do not promote based on this simulator pass.
