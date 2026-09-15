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

The opt-in `dspark-64k-publication-profile-request.py` wrapper samples only the
first three publications from each of two complete paired requests. It separates
projection, bank assembly and remaining host work, without extra fences, altered
cache contents or reordered calls. It retains the existing lazy-loader and
combined-runtime correctness admissions. Profiling TG is labeled instrumented,
not promoted as a clean speed measurement.

Two local tests cover bounded sampling, unchanged results, callback restoration
and error handling. Hardware attribution is **not yet run**. The active clean
paired run remains uninstrumented. Select a publication optimization only after
the cost is attributed; a 44 ms saving alone cannot establish 200 committed TG.
