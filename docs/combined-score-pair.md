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
