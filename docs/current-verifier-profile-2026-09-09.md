# Current request verifier attribution

**Purpose: identify the actual62 ms verifier cost, not generate another TG number.**
At the measured4K acceptance rate,200 TG requires a complete cycle below35.59 ms.
The older static profile used different attention/state options, so its operation
sums are not evidence of the current critical path.

## Profiled configuration

| Item | Selection |
| --- | --- |
| Hardware | Two P150A cards, TP2, audited four-link communication |
| Request | Same4K coding prompt; one complete response through EOS |
| Drafter | Captured T8 DFlash2, fused convolution, historical K/V cache |
| Cache update | Eager projection; the flat captured-update candidate is excluded |
| Verifier | Current commit-only GDN, norm batching, native-row sampling and native serial attention |
| Timing status | Instrumented correctness/attribution only; no PP/TG summary |

`full-dflash-verifier-profile` observes each real `VerifierEngine.verify` call.
It does not replay a different fixture, change proposals or publish extra state.
Profiler dumps and native-state digests sit outside the marked verification
intervals. The full request still has to match native tokens/GDN/valid KV/inactive
slots and pass complete feature, cache, proposal and convolution audits.

## Evidence required

- Every actual verification has an unambiguous block/position/width/trace marker.
- Runtime replay counts match those calls exactly on both physical chips.
- Replays of one captured trace retain the same operation/core-count coverage.
- Keep the first replay visible but exclude it from steady-state medians.
- Report operation/core sums, per-chip kernel envelopes/unions and host call costs.
- Reject missing data, dropped markers inside verification and inconsistent clocks.

Device nanosecond envelopes use a checked conversion from the runtime's own
cycle/duration pairs. They are observed intervals, **not a dependency-graph
critical-path proof**. Kernel durations may include waits and overlap. Do not sum
chips, convert those sums to TG or identify uncovered intervals as host stalls
without further evidence. Python call durations include native-extension waits,
not CPU utilization.

## Gate status

859 CI tests and60 harness tests pass. Tests cover real-call observation, error
handling, profiling-only summaries, markers, both-chip coverage, replay counts,
overlapping intervals and rejected mixed configurations. The parser also reads
the preserved older C++ CSV successfully; that is a format check, not new timing.

This changes instrumentation only: no new kernel math, trace contents or serving
defaults. The underlying runtime already passed learned simulator and full-request
hardware gates. Hardware attribution has not been dispatched yet.
