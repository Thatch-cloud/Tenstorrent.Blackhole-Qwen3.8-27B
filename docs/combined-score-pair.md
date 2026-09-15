# Combined-runtime paired comparison

The control is the hardware-validated split-K + fused T16 MLP + shared-Q/K
runtime (31.33 committed TG at 64K in run 34926705012). MLP without shared Q/K
previously measured 31.40 TG; that difference was not a paired gain.

One model load serves two identical 65,536-token coding requests, in this order:

1. Control, without score-layout fusion.
2. Same runtime, with the admitted score-layout fusion.

Both retain the 256-token output budget, native gold reference, exact output,
final-state and inactive-slot checks. Existing proposal-repeatability and
component gates remain active. There is no text-only loading change, host
deserialization probe, extra per-block feature audit or profiling fence.

If the control is below 25 committed TG, retain its evidence and stop before
the candidate: a platform/runtime regression should not consume another request.
This floor is a diagnostic stop condition, **not** a success target. Existing
420-second request and 540-second outer limits are unchanged.

Record separate per-arm TG and prefill times plus existing per-block draft,
verify/readback and selection/commit times. Do not pool the two arms into a
throughput number. One observation per arm in fixed order is diagnostic, not
repeatability, an order-independent causal result, held-out coding acceptance
or proof of the 200 committed TG objective.
# Paired timing outcome: 35018609131

The run fails with **exit 124 during final parameter auditing**, after both
complete requests return exact output/state checks. Clean device closure and
final learned-weight checks are incomplete, so this is not performance acceptance.

| Arm | CTX | Committed output | Generation seconds | Diagnostic TG |
| --- | ---: | ---: | ---: | ---: |
| Native-score control | 65536 | 135 | 5.2900 | 25.52 |
| Fused-score candidate | 65536 | 135 | 16.1356 | 8.37 |

Target loading takes 52.92 seconds. The requests finish at process seconds 251.01
and 400.57; final parameter auditing starts at 401.30 and exceeds the 420-second
process budget. Do not extend the timeout and rerun this candidate unchanged.

The candidate has severe history-preparation stalls: 2684.88 ms at position
65558 with only 48.04 ms process CPU, and 1920.69 ms at 65650 with 45.41 ms CPU.
Recorded GC pauses are zero, and cgroup CPU throttling/OOM counters do not
increase. Blocking verifier replays remain around 80.6 ms. These facts do not
prove whether the stalls originate in I/O, runtime compilation, synchronization
or history-bank assembly; nested attribution is needed before changing kernels.
Score fusion is not promoted. Its causal effect cannot be isolated from this
single ordered pair and variable stalls.
