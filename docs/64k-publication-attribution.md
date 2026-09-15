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
**not yet run**. Select a publication optimization only after
the cost is attributed; a 44 ms saving alone cannot establish 200 committed TG.
