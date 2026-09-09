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

860 CI tests and60 harness tests pass. Tests cover real-call observation, error
handling, profiling-only summaries, markers, both-chip coverage, replay counts,
overlapping intervals and rejected mixed configurations. The parser also reads
the preserved older C++ CSV successfully; that is a format check, not new timing.

This changes instrumentation only: no new kernel math, trace contents or serving
defaults. The underlying runtime already passed learned simulator and full-request
hardware gates. [Hardware run34296336943](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34296336943)
on `8460b69` failed during profiler finalization, after the complete request audit
passed. It is not a new throughput result.

## Profiler memory failure and correction

| Evidence | Result |
| --- | --- |
| Saved request | Passed, CTX4,096, 121 committed tokens through EOS; all 17 real verifier calls recorded |
| Correctness and marker revalidation | Passed against the downloaded request and console artifacts |
| Container memory | Peak 103,079,215,104 bytes (96 GiB); `oom_kill=1`, `oom=12` |
| Device attribution | Missing C++ CSV; no kernel attribution can be claimed |

Request artifact SHA256:
`a430df3e2934355eb638c7242c15a0d0374d08147581debf762d82cb8f737487`.
The saved host profile places 1.048 s of 1.136 s in the blocking verifier
replay/readback callback across 17 calls. This is not evidence of a Python-only
bottleneck or a replacement for the missing device measurements.

The wrapper previously disabled mid-run dumps. At pinned TT-Metal `9f9cd4f`,
`ReadDeviceProfiler` then retains device markers until final processing.
Incremental dumps instead analyze and append the C++ CSV, then clear those
markers. The retry requires `TT_METAL_PROFILER_MID_RUN_DUMP=1` and passes Tracy's
`--dump-device-data-mid-run` option. It retains every real verifier call and all
native audits, with before/after cgroup evidence; it does not raise the memory
limit or reduce correctness coverage.

The tests also reject missing incremental-dump configuration before opening
hardware or observing a request. This is a source-grounded collection fix, not
yet a successful hardware profile.
The incremental-dump retry is
[run34298049648](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34298049648)
on `da69ef8`; its final device attribution is pending.
Mid-run dumping weakens cross-device clock alignment; attribution remains
per-chip, and no cross-chip critical-path claim is permitted.
