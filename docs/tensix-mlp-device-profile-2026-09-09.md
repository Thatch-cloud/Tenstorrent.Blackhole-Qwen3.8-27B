# Why is spare-core streaming slower?

The uninstrumented [hardware comparison](tensix-pooled-mlp-2026-09-09.md)
measures **0.335 ms native versus 0.744 ms streamed**, with exact outputs.
This diagnostic locates that regression before another kernel change.

[CI 34332550656](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34332550656)
passes on immutable tag `ci-qwen-hardware-9f52675`. All twenty replays and both
chips are accounted for; the independent local re-analysis also passes.

## Result: the projections lose, not the collective

| Operation | Native, microseconds | Streamed, microseconds |
| --- | ---: | ---: |
| Input copy | 1.95 | 1.95 |
| Gate + SiLU | **96.07** | 222.38 |
| Up | **87.89** | 216.95 |
| Product | 3.95 | 3.35 |
| Down | **122.22** | 244.20 |
| Four-link reduction | 8.78 | 8.04 |

Values are the mean of the two chips' six-measurement medians, from instrumented
kernel durations. They are not additive request latency. Projection names follow
the captured native/streamed operation order; all six operations recur unchanged.

Per-chip complete kernel envelopes are approximately **327-328 us native versus
737 us streamed**. Uncovered intervals rise from about 6.94 us to 39.8 us, but
most of the regression is inside the three projections. Input copying, product
and the collective do not account for the slowdown. Long BRISC/NCRISC/TRISC
durations do not distinguish useful work from waiting.

MLP result SHA256: `15284cf7211abd144b62996f4c437e9d507be9c802d05580f0b0681de30fab39`.
Raw device CSV SHA256: `167efc0ccfeba40f3d3f5cb1e8d89724a250b5b1e2e3f71fbf94ad872ec3184d`.
Recomputed attribution SHA256: `e9e5ae049ad155fe838779bb5fe39e6c036e48a674d561a38a5255e85bf5c490`.

## Next isolated test: producer capacity

The streamed path reduces weight-reading workers to eight, versus 39/32 native
matmul workers. Its reader issues individual tile reads. **Insufficient producer
issue capacity is a hypothesis**, not something these aggregate RISC timings prove.

Test sixteen producers while keeping the same 68/80 compute receivers, native
math, compressed weights, two-page FIFOs, activation path and four-link reduction.
Use eight senders on row9 and eight on row8, disjoint from the activation rectangle.
The gate/up operation then uses 93 total active workers; down uses 104, within110.
Maximum receivers per sender fall from9 to5 for gate/up and10 to5 for down.

Before hardware: qualify the new mapping with full-size pooled MLP simulation,
both fixtures/chips, all32 physical rows, changed-input replay and raw-weight audits.
Record the actual producer count in source-bound evidence; reject report/runtime
mismatches. Then run the same real-weight nine-block ABBA against native.
Require a greater-than2% gain in every block before any full-model promotion.
No gain is predicted, and 200 committed TG still requires a complete request result.

| Setting | Attribution run |
| --- | --- |
| Hardware/model | Same two P150A cards and real Layer-0 BF4/BF4/BF8 weights |
| Workload | T8 MLP verification for one stream, not batch-eight serving |
| Operations | Same complete native and pooled streamed traces; four links in both arms |
| Replays | Twenty: audit/stale controls plus three fixture-specific C/A/A/C sequences |
| Numerical checks | Every measured output on both chips; raw weights/input and native metadata unchanged |
| Device evidence | Trace IDs, every replay, operation order, core counts, per-RISC durations |
| Reporting | Per-chip kernel intervals and operation attribution; **no TG or promotion** |

Suite: `tensix-stream-mlp-profile`. The two existing simulator gates run before
the hardware build/device probes. No arithmetic, transport, FIFO or weight-layout
source changes. Host tests cover missing/reordered events, changed captured
operations, dropped markers, incorrect output, and profiler contamination of
performance mode. The current suites pass 992 CI tests and 57 simulator tests.

Profiler dumps and correctness readbacks sit outside replay markers. All twenty
runtime replay sessions must match the recorded calls on both chips. The six
measured replays per chip/arm are retained separately from audits and stale checks.

Long RISC durations can include waits and span different cores. They do not
alone prove DRAM saturation, NoC contention, or credit starvation. Operation
sums can overlap; they are not an end-to-end critical path or a token rate.
