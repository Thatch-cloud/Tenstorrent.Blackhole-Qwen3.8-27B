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

## Split hardware profile: run 34674716217

Revision `2a692be236b510af5c6f741834b2dedf94752b25` completes cleanly;
all six requests report exact output, target state and inactive state. The two
non-audit fusion requests retain 222/330 accepted drafts and 242 committed tokens.
Their combined PP2900.98 / CTX4096 / TG62.18 is a profiling-run result, not an
improvement or serving qualification.

| Non-audit fusion request | Projection ms/cycle | Bank assembly ms/cycle | Total history ms/cycle | Whole cycle ms |
| --- | ---: | ---: | ---: | ---: |
| First | 13.009 | 3.765 | 17.619 | 122.137 |
| Second | 42.659 | 3.786 | 128.414 | 231.506 |

Each row averages 11 cycles. Total history includes the two subintervals; do not
add these columns together. Bank assembly is stable across these requests,
whereas projection and the unclassified outer interval vary substantially.
The second request has about 81.97 ms/cycle outside both subintervals. This
changes the next investigation: attribute outer validation, tensor ownership
and cleanup, including host/GC pauses, before claiming full-bank copies dominate.
These are host-wall measurements with no extra device fences, not isolated
kernel timings. Preserve both repeats; do not cherry-pick the faster request.

## GC follow-up: run 34675539551

Revision `0c08f72bc6b584be38b772aef633b3ab7b3c6d43` passes and closes cleanly.
The four non-audit requests retain exact output, target and inactive state.
Fusion summary: PP1946.41 / CTX4096 / committed TG93.43, one stream,
222/330 accepted drafts and 242 committed tokens. This is not a new speed record.
Report SHA256: `fdf62e95fa3e0a9d436ee7a88db660a56eef210692de3a4ed9132a0dad920081`.

| Fusion repeat | Projection ms/cycle | Bank assembly ms/cycle | History total ms/cycle | GC within history ms/cycle |
| --- | ---: | ---: | ---: | ---: |
| First | 11.131 | 3.268 | 15.423 | 0.0119 |
| Second | 9.707 | 3.367 | 13.948 | 0.0082 |

GC was observed, not disabled. It is negligible in this run; the previous
81.97 ms unclassified spike did not recur, so this does not explain that older
event. Process CPU time within history averages 14.842 and 13.329 ms/cycle.
Prioritize captured feature projection over blanket GC changes or assuming
bank copies dominate. The current projection builds and uploads rotary tables,
dispatches shared feature projection and all ten learned K/V projections, then
synchronizes for each accepted prefix. A captured path must keep inputs and
outputs at stable addresses, preserve rounding and accepted-prefix masking,
and prove changed-position replay plus rollback before hardware comparison.

## Captured projection simulator gate

Run **34677062667**, revision `0ba59b3`, passes with exit 0 and clean cleanup.
All 20 learned K/V output shards match fixed32 eager execution exactly at
positions 4096, 4111 and 4128, with changing synthetic feature taps and rotary
tables. Entirely stale output is explicitly rejected. Source hashes match the
checked-out revision and remain unchanged during the run; runtime fingerprints
are unchanged. Peak cgroup memory is 8.36 GiB, with no OOM events or swap use.

Report SHA256: `2986b5b4c89b1a30f499d957d50fb2e3993b946f6980877ac924a09f6f07de5e`.

This qualifies the observed captured-versus-eager projection behavior only,
not an independent model oracle, accepted-prefix publication/rollback,
coexisting target traces or combined-request performance. Next integrate the
captured outputs with the existing transactional history banks, preserving
prefix slicing and borrowed-output lifetime before the paired hardware run.
