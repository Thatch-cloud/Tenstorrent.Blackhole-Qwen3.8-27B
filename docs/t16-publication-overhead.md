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

### Transaction simulator pass

Run **34677941763**, revision `abc23a8`, passes with clean exit 0. Captured
learned projection matches eager output across six 20-shard comparisons.
The bank adapter passes a 15-row commit at 4096, a one-row discard at 4111,
and a 32-row commit at the unchanged 4111 frontier. Checks compare complete
active/prepared banks and preserve their allocation addresses. All 803 source
hashes match the revision and before/after snapshots; runtime fingerprints
remain unchanged.

Report SHA256: `4bd749d6381cb7e1f5be69276d5a5011c9e6cfa7c30182b44dd09f3d1b115914`.

Initial history and feature taps remain synthetic; learned projection weights
are real. This is not full-request admission or a TG measurement. The next
gate is opt-in full-request integration with target-derived features and
coexisting target/proposal traces, followed by matched hardware PP/CTX/TG.

## Combined hardware result: captured publication

Run **34679446247**, revision `b09450f`, passes exact outputs, target state and
inactive state on all six requests. The audited candidate records 12 complete
20-shard projection comparisons. The paired summary was independently
recomputed; request and native source snapshots remain unchanged.

| Merge intervals, one stream | PP tok/s | CTX | Committed TG tok/s | Mean select/commit ms/cycle |
| --- | ---: | ---: | ---: | ---: |
| Combined fusion control | 3291.29 | 4096 | 102.12 | 11.441 |
| Captured publication | 3326.66 | 4096 | 106.75 | 7.157 |

Both arms commit 242 tokens across two timed requests, with 222/330 accepted
drafts (67.27%). TG improves **4.53%**, saving about **4.284 ms/cycle** in
selection/publication. Candidate mean whole-cycle time is 103.008 ms versus
107.670 ms control. Setup-inclusive time worsens: 6455.79 versus 6096.87 ms.
This is a useful combined-runtime improvement, not 200 TG or broad coding
quality qualification. Retain the supported context-ladder acceptance step;
do not extrapolate this CTX4096 result to long context or multiple streams.

Report SHA256: `2451fe1ab0470f5a1b82d376d7b6a5d95a1997ca9a46b9d3bb42938fc4bf9ec4`.

### Stable-unique coding screen

Run **34679998171**, unchanged revision `b09450f`, also passes. Its paired
summary was independently recomputed with unchanged request/native source
snapshots. One stream, CTX4096:

| Runtime | PP tok/s | Committed TG tok/s |
| --- | ---: | ---: |
| Combined fusion control | 3344.86 | 119.25 |
| Captured publication | 3361.97 | 126.33 |

TG improves 5.94%; both arms commit 104 tokens with 98/120 accepted drafts
(81.67%) across two timed requests. Candidate setup-inclusive time is 5487.57
ms versus 5552.69 ms control. This is a short task-specific sample, not a
sustained long-generation throughput claim.

CI did not execute functional coding tests. Independent local execution of the
inspected generated function passes four cases: empty input, interleaved
duplicates, negative/zero values and identical values. All preserve the input.
This small screen is not broad held-out coding-quality certification.

Report SHA256: `9bd2c13d26e44859a108203519f7330d2332b8132e020c15b5423a17b642f4a8`.
