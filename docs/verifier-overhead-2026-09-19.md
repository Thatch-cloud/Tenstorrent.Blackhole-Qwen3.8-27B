# The verifier streams weights at 37% of measured DRAM bandwidth

The 200 tok/s gap is not a memory-system limit and not host overhead. It is the
verifier's traced device program using 2.7x more time than the bytes it moves require.

Derived from run 35258782100 (T16 control, 4096 context, one user) and the bandwidth
measurement in run 35416870419. No new instrumentation: the phase timing this needs has
been emitted all along by `serving_fast_request.step` under `QWEN_FAST_PHASE_TIMING=1`.

## The decomposition

Median over the 40 post-warmup blocks. `cycle_ms` 97.57 reproduces the 97.55 in
`200tg-latency-budget.md`, so this is the same regime that doc measured.

| Phase | ms |
| --- | ---: |
| cycle | 97.57 |
| draft | 24.62 |
| **verify + readback** | **66.89** |
| select / commit | 5.15 |

and inside the verifier:

| Component | ms | |
| --- | ---: | --- |
| `blocking_trace_host_ms` | **66.09** | device execution |
| `input_ms` | 0.44 | host |
| `replay_checks_sync_ms` | 0.69 | host |
| `binding_validation_ms` | 0.40 | host |
| `output_readback_host_ms` | 0.13 | host |

Host-side overhead totals **1.66 ms**. Dispatch, staging and readback are not the
problem; 98.8% of the verifier is time the host spends blocked inside the device trace.

## CORRECTION: weight streaming is efficient; the opportunity is elsewhere

A device profile of this path already exists - run 34298049648, written up in
[current-verifier-profile-2026-09-09.md](current-verifier-profile-2026-09-09.md) - and
it contradicts the first reading below. Keeping both because the error is instructive.

| Per-chip observed interval (T8, 4096 ctx) | Chip 0 |
| --- | ---: |
| Kernel envelope | 61.539 ms |
| Union of kernel intervals | 60.216 ms |
| Uncovered intervals | 1.324 ms |

| Operation group | Calls | Median summed kernel |
| --- | ---: | ---: |
| All matrix multiplications | 321 | 32.050 ms |
| Gate/up, 39 cores | 128 | 11.787 ms |
| Down/output projections, 32 cores | 128 | 10.803 ms |
| GDN input projection, 43 cores | 48 | 5.741 ms |
| GDN recurrence, 96 cores | 48 | 6.290 ms |
| Native decode SDPA, 110 cores | 128 | 5.827 ms |

The weight-bearing matmuls total **28.33 ms**, which against 9.96 GB per card is about
**352 GB/s, 87% of the measured 405 GB/s**. Weight streaming is close to the achievable
rate, and the device is busy for 98% of the interval (1.32 ms uncovered).

So the real shape is:

| | ms | share of the 60.2 ms union |
| --- | ---: | ---: |
| matmul | 32.05 | 53% |
| GDN recurrence | 6.29 | 10% |
| decode SDPA | 5.83 | 10% |
| unattributed | 16.05 | 27% |

**The opportunity is the 28.2 ms that is not matmul**, and the largest single piece of it
is 16 ms that the existing grouping does not name.

### Why the first reading was wrong

Dividing the weight bytes by the whole verifier time assumes the verifier does nothing
but stream weights. It does not: matmuls are roughly half the device time. The 150.7
GB/s figure below is therefore an average over an interval that includes GDN recurrence,
SDPA and other work, not a measure of streaming efficiency. The profile separates them
and the separation changes what to optimise.

Two caveats on the profile itself, from its own text: operation sums can overlap and
durations include waits, so these are observed intervals rather than a dependency-graph
critical path; and it profiled a **T8** verifier on 2026-09-09 with a different option
set, where the current control is T16 on 2026-09-17. A fresh profile of the current
configuration is what settles the 16 ms.

## The first reading, retained: what the time is worth in bytes

Each card must read its 9.96 GB half of the weights once per verification.

```
effective bandwidth = 9.96 GB / 66.09 ms = 150.7 GB/s
measured capability                      = 405.0 GB/s   (run 35416870419)
                                           37.2%, a 2.7x headroom
```

If the verifier streamed at the rate the same memory system delivers to a plain matmul:

| | now | at 405 GB/s |
| --- | ---: | ---: |
| verifier | 66.09 ms | 24.59 ms |
| cycle | 97.57 ms | 56.07 ms |
| per user | 123 tok/s | **214 tok/s** |

That is above the 200 tok/s target **at the current single-user, 4096-context
configuration**, before any concurrency work.

## The draft is the control that makes this credible

The drafter runs 15 sequential steps in 24.62 ms, 1.64 ms per step. At 405 GB/s that is
about 0.66 GB per card per step, a plausible size for five DFlash2 layers. So the draft
appears to run near the bandwidth limit on the same hardware, in the same process, in
the same cycle. The verifier does not. Whatever costs the verifier 2.7x is specific to
its traced program, not to the device, the memory system or the runtime.

## What to profile next

The existing profile is T8, older, and leaves 16.05 ms unattributed. A fresh device
profile of the current T16 configuration should answer, in order:

1. **What is the unattributed 16 ms?** It is the largest single unexplained block.
2. **How much of it is TP2 collectives?** An all-gather or all-reduce per layer across
   64 layers, serialised against the matmuls rather than overlapped.
3. **Does GDN recurrence scale with rows?** 6.29 ms at T8; if it grows with verify rows
   it matters more at T16 and much more at any batched configuration.
4. **Is SDPA at 110 cores near its own limit**, or padded-row waste at 16 rows?

The wrapper needs `TT_METAL_PROFILER_MID_RUN_DUMP=1` and Tracy's
`--dump-device-data-mid-run`; without incremental dumps the earlier attempt hit a 96 GiB
container OOM at final processing (run 34296336943). Memory peaked at 92.871 GiB even
with the fix, so headroom is tight.

## Superseded: what the 41.5 ms is not, and what it might be

Not host overhead (1.66 ms, measured). Not arithmetic: 16 rows over a 27B model is
roughly 0.9 TFLOP, under 3 ms even at pessimistic utilisation. That leaves, in order of
suspicion:

1. **TP2 collectives.** An all-gather or all-reduce per layer across 64 layers, serialised
   against the matmuls rather than overlapped with the next layer's weight fetch.
2. **Non-overlapped weight fetch.** If each layer's DRAM read waits on the previous
   layer's compute, the pipeline is idle for much of every layer.
3. **Matmul shape efficiency at 16 rows.** A 16-row activation against a large weight
   matrix may tile poorly and under-use the DRAM burst.

These are distinguishable. The next step is a device-profiler run
(`TT_METAL_DEVICE_PROFILER`) over one verification, giving per-op device times inside
the trace, which separates collectives from matmuls from waits.

## Why this is the right thing to work on

- It is the largest single term: 41.5 ms of a 97.57 ms cycle.
- It is the only lever that helps *every* configuration, including any future
  multi-session work, which needs verifier overhead under 12.1 ms to reach the target at
  4 users with bf4 KV.
- It needs no architectural change to attempt, unlike multi-session support, which
  requires a batch axis through a single-sequence speculative engine.
- The draft demonstrates the hardware already achieves the rate elsewhere.

Ruled out by measurement: DRAM bandwidth itself. 405 GB/s is 79-80% of the 512 GB/s
spec, which is a normal sustained figure for GDDR6, consistent across access sizes and
across read-only and read-write regimes. There is no meaningful win there; a physically
impossible 100% of spec would save 5.1 ms against the 41.5 ms above.
