# Split history preparation before changing it

The two completed requests in run **34926705012** contain 40 committed blocks.
Their existing publication timers report these means:

| Publication stage | Host wall ms/block |
| --- | ---: |
| Feature selection | 0.057 |
| Prepare history | 44.109 |
| Publish target state | 1.978 |
| Commit history frontier | 0.009 |

These are nested inside selection/commit, not additional cycle costs. Process
CPU time is not device-kernel time. The current source places captured feature
projection and full-bank history assembly inside `prepare_history`; the existing
record does not separate their costs. Do not attribute all 44 ms to copies.

The opt-in `dspark-64k-publication-profile-request.py` wrapper samples up to 32
publications from one complete control request. It separates
projection, bank assembly and remaining host work, without extra fences, altered
cache contents or reordered calls. It retains the existing lazy-loader and
combined-runtime correctness admissions. Profiling TG is labeled instrumented,
not promoted as a clean speed measurement.

Local tests cover bounded sampling, unchanged results, callback restoration
and error handling. After the exact output/state request returns, an explicit
diagnostic exception unwinds the runtime and closes devices before a second arm
or final parameter audit. A completed diagnostic requires clean shutdown but
keeps `passed=false`, `full_request_passed=false` and all accepted TG fields null.
It is not a substitute for those audits on a performance candidate. The process
budget is 300 seconds, with a 420-second outer limit. Hardware attribution is
reported below. A 44 ms saving alone cannot establish 200 committed TG.

## Hardware diagnostic 35020108364

Completed in **4m12s**. One 135-token request passes exact output, final active
state and inactive-slot checks; device and checkpoint closure pass. This is
diagnostic completion, not full performance acceptance or a final weight audit.

| Sampled stage, 20 blocks | Mean ms | Maximum ms |
| --- | ---: | ---: |
| Captured feature projection | 1.653 | 2.324 |
| Full history-bank assembly | 46.803 | 63.831 |

The multi-second stalls from the previous run do not recur, so their cause is
still unresolved. The persistent publication cost is now attributed to bank
assembly, not projection. Report SHA256:
`d134bd0b278c23873f19a7fa5d628d0adcf772e03b5d7905e4d467882e9084f1`.

Next candidate: replace full-history slice/concat/pad/copy with a bit-preserving
dirty-tile writer. Keep the active bank untouched until commit. Repair the spare
bank's previous accepted/discarded tail, append only accepted rows, and zero
invalid rows within the touched tiles. A bounded planner covers at most 96 rows
per append instead of rebuilding 66,560 rows. CPU tests compare entire banks
through 480 randomized commit/discard transactions and tile boundaries, including
64K positions. This is a transaction plan, **not an implemented accelerator
kernel or a measured speedup**. Next gates: BF16 bit-copy kernel, simulator bank
contents/address lifetime checks, then combined-runtime hardware correctness and
matched PP/CTX/TG. Serving defaults remain unchanged.
