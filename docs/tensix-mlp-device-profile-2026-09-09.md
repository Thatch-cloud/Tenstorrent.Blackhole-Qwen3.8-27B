# Why is spare-core streaming slower?

The uninstrumented [hardware comparison](tensix-pooled-mlp-2026-09-09.md)
measures **0.335 ms native versus 0.744 ms streamed**, with exact outputs.
This diagnostic locates that regression before another kernel change.

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
