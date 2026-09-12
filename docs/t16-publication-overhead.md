# Combined-runtime publication overhead

Recomputed from hardware run **34586017906**, fusion arm, two non-audit requests
and 22 speculative cycles. CTX4096, one stream, 15 draft queries / T16 verifier;
committed TG **101.88 tok/s**. These are retained hardware measurements, not simulator timings.

| Selection/commit work | Mean host wall time per cycle |
| --- | ---: |
| Obtain verified features | 0.031 ms |
| Prepare drafter history | 9.612 ms |
| Publish target state | 1.706 ms |
| Commit drafter history | 0.005 ms |
| Remaining selection/commit work | 0.084 ms |
| **Total selection/commit** | **11.438 ms** |

History preparation accounts for about **84%** of this bucket. Its process CPU
time averages 10.312 ms; this is process-wide and can exceed wall time. These
timers add no device fences but include instrumentation. They do not separately
measure kernel execution, dispatch, copies or synchronization.

`StableHistoryKV.prepare_publication` projects accepted target features into all
five learned K/V pairs. `prepare_projected` then slices the old history,
concatenates the new prefix, pads to capacity and copies into the spare bank for
each of ten K/V tensors, followed by device synchronization. Thus the current
path rebuilds full history even when only a few rows were accepted.

Next attribution should separate feature projection from history-bank assembly
inside the same combined request. Then test range-based publication or captured
publication while preserving failed-publication rollback and trace address
stability. Do not attribute all 9.612 ms to copies without that measurement.

Even eliminating the entire 11.438 ms bucket would only reduce this workload's
107.97 ms mean cycle to about 96.53 ms; this alone cannot establish 200 tok/s.
Keep verifier/drafter optimization and acceptance measurement in the programme.
