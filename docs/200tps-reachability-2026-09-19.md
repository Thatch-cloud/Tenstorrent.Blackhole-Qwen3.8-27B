# Is 200 tok/s per user reachable at 4 users and 161k?

**Not in the configuration the programme is currently building for.** At 4 concurrent
users, 161k context and bf8 KV, the DRAM traffic alone needs ~77.6 ms per speculative
cycle against a 60.5 ms budget. The target is missed by 17 ms *before* a single cycle
of compute, dispatch or collective overhead is counted.

This is arithmetic over measured block timings and a byte count, not a new experiment.
It was derived while the M1 gate was on the rig, and it re-points the programme.

## The budget

From `docs/200tg-latency-budget.md`, T16 control run 35258782100:

| Quantity | Measured |
| --- | ---: |
| Mean committed tokens per cycle | 12.1 |
| Draft / verify+readback / selection | 24.67 / 66.63 / 5.34 ms |
| Whole cycle | 97.55 ms |

12.1 tokens per 97.55 ms is **124.0 tok/s per user**. 200 tok/s at the same acceptance
means `1000 x 12.1 / 200` = **60.50 ms per cycle**.

Per-user rate is set by cycle time alone. Batching multiplies aggregate throughput and
leaves the per-user number untouched, so *the user count never helps this target* — it
only adds work inside the same 60.5 ms.

## The floor those 60.5 ms have to cover

Two dense reads happen every cycle and neither is compressible by scheduling.

**Weights.** 19.92 GB across TP2 = 9.96 GB per card. At 512 GB/s that is **19.45 ms**.

**KV cache.** Only the 16 full-attention layers grow with context (the 48 GDN/linear
layers hold ~40 MB/user of state, negligible). At bf8:

```
2 (K,V) x 8 kv heads x 128 head_dim x 16 layers = 32,768 B = 32 KB per token
163,840 tokens x 32 KB = 5.37 GB per user
```

Every decode cycle attends over the whole cache. The 16 speculative rows share one
read, which is the favourable assumption. At 4 users that is 21.5 GB per cycle, 10.74
GB per card, **20.97 ms** at 512 GB/s.

So the verifier's floor at 4 users and 161k is **40.4 ms** at theoretical bandwidth.
Add the measured draft (24.67) and selection (5.34) and the *cycle* floor is 70.4 ms —
already 10 ms over budget with the verifier's overhead set to zero.

## Where that leaves each configuration

Budget minus floor, at an assumed 85% of theoretical DRAM bandwidth. "Overhead
allowed" is what remains for everything that is not DRAM traffic; today the verifier
alone carries about **47 ms** of it.

| 4 users @ 161k | Budget ms | Floor ms | Overhead allowed | |
| --- | ---: | ---: | ---: | --- |
| bf8 KV, acceptance 12.1 *(today, and the programme's target config)* | 60.5 | 77.6 | **-17.1** | impossible |
| bf8 KV, acceptance 14 | 70.0 | 77.6 | -7.6 | impossible |
| bf8 KV, acceptance 16 (perfect T16) | 80.0 | 77.6 | 2.4 | feasible |
| bf4 KV, acceptance 12.1 | 60.5 | 65.2 | -4.7 | impossible |
| bf4 KV, acceptance 14 | 70.0 | 65.2 | 4.8 | feasible |
| bf4 KV, acceptance 16 (perfect T16) | 80.0 | 65.2 | 14.8 | feasible |

A negative figure is not a hard engineering problem; it is a statement that no
software change reaches the target, because the memory system cannot deliver the bytes
in the time available.

**The dominant lever is acceptance, not bandwidth.** The budget scales linearly with
committed tokens, so 12.1 -> 16 buys 32% of budget, more than halving KV precision
buys. Nothing in the current task sheet owns acceptance.

**bf4 KV is required either way.** It was previously filed as a capacity question for
8 users; it is also a bandwidth question at 4. It is the only entry that leaves a
double-digit overhead allowance.

## Lever N does not address any of this

Worth stating plainly, because Lever N is the active build. Its own design doc:

- s4: `decode ITL between stalls | 38-47 ms | unchanged (decode path untouched)`
- s7: `Speculative decoding is off on this endpoint; the design does not consider it.`

Lever N M1/M2 is what makes concurrency and non-blocking 30k prefill possible, and it
remains necessary for the *other* halves of the goal. It contributes exactly zero to
the per-user decode rate. The 200 tok/s number is currently unowned by any task.

## What is assumed rather than measured

The conclusion rests on a spec figure, and that is the weakest link.

1. **512 GB/s is the p150a GDDR6 spec; achieved bandwidth has never been measured
   here.** Every floor above scales inversely with it. This is measurable in an hour
   and should be measured before the programme is re-planned on these numbers.
2. 85% efficiency is an assumption. The table shows 100% and 75% variants in the
   working; at 100% the bf8/12.1 cell is still -10 ms.
3. Draft cost is treated as context-independent, on the strength of
   `draft_kv_history.py` holding per-layer 2K K/V buffers. Unverified at 161k. If the
   draft KV does grow with context, every figure here is optimistic.
4. Acceptance 12.1 was measured at 4096 context. Acceptance at 161k is unknown and
   has no reason to be identical.
5. The verifier is assumed to read the KV cache once per cycle, not once per
   speculative row. If it reads per row the floor is 16x worse and the target is far
   further away.

Assumptions 1 and 4 both make the verdict *softer* if they break favourably, and 3 and
5 make it harder. None of them rescue the bf8/12.1 cell.

## Recommended re-point

1. **Measure achieved DRAM bandwidth and the verifier's internal split** at long
   context. The whole analysis turns on the 512 GB/s figure and on whether the ~47 ms
   of verifier overhead is dispatch, collectives or non-overlap.
2. **Measure acceptance at 161k.** It sets the budget and is the largest single lever.
3. **Re-scope the target with the user.** 4 users at 161k and bf8 looks capped near
   **156 tok/s/user** even with a perfectly bandwidth-bound verifier, and nearer
   120-130 with any realistic overhead. Either bf4 KV enters scope, or the per-user
   number comes down, or the context does.
4. Finish Lever N regardless — concurrency and non-blocking prefill are needed for
   every surviving version of the goal.

Reproduced by the arithmetic in this document; no serving defaults were changed and no
hardware was used to produce it.
