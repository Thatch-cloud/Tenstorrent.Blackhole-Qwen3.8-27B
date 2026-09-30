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

## DEVICE ATTRIBUTION, T16: it is the GDN machinery, not matmuls or collectives

No new hardware run was needed. Run 35185624322 - one of four "winning verifier profile"
attempts that CI recorded as failures - completed its request and audits in full and
failed only in profiler finalisation, where `python3 -m tracy` tried to copy a host
`.tracy` file that did not exist and then launch a Tracy WASM web GUI. The
`preserve_metadata` EXIT trap in `dspark-combined-device-profile.sh` had already written
an 81 MB `cpp_device_perf_report.csv`, 355,461 rows, both chips. The data was there the
whole time behind a red tick.

Trace 24 is the T16 verifier (`qwen_request_verify_4_pos4137_t16_trace24`). Chip 0,
median of the four steady replays with the first discarded:

| Operation | ms | calls | share |
| --- | ---: | ---: | ---: |
| **GenericOpDeviceOperation** | **32.59** | 496 | **50%** |
| MatmulDeviceOperation | 20.25 | 193 | 31% |
| SdpaDecodeDeviceOperation | 3.49 | 32 | 5% |
| AllGatherAsyncDeviceOperation | 2.01 | 130 | 3% |
| GdnConvGatesDeviceOperation | 1.70 | 48 | 3% |
| ReduceScatterMinimalAsyncDeviceOperation | 1.43 | 128 | 2% |
| everything else | 3.43 | | 5% |
| **total** | **64.90** | | |

64.90 ms against the 66.09 ms the host observes blocking on the trace, so the attribution
is essentially complete.

**The collectives hypothesis was wrong.** All-gather plus reduce-scatter is 3.44 ms,
5% of the verifier. TP2 communication is not the problem.

`GenericOpDeviceOperation` is the project's own kernels, and splitting it by core count
separates them by which layers they run on:

| cores | ms | calls | |
| ---: | ---: | ---: | --- |
| 99 | 11.40 | 64 | one per layer, all 64 |
| 96 | 9.53 | 48 | GDN recurrence, per the T8 profile's labelling |
| 48 | 4.61 | 112 | |
| 24 | 3.38 | 48 | GDN-shaped |
| 16 | 2.48 | 112 | |
| 8 | 0.86 | 48 | GDN-shaped |
| 32 | 0.33 | 64 | |

The 48-call groups track the 48 GDN layers and total **13.77 ms**; the 64-call groups
track all layers and total 11.73 ms. So the GDN path, not weight streaming, is the
largest identifiable cost in the verifier.

### The scaling result that matters most

GDN recurrence was **6.29 ms at T8** and is **9.53 ms at T16**: 1.52x for twice the
verify rows. It is not free per row. Speculation buys committed tokens by widening the
verifier, and this term grows as it does, so it taxes exactly the mechanism the 60 ms
budget depends on - and it would be taxed again by any batched multi-session work.

### Where this leaves the target

Matmul is 20.25 ms of a 64.90 ms verifier. Even eliminating every non-matmul operation
entirely - collectives, SDPA, the whole GDN path - would leave a verifier near 20 ms and
a cycle near 52 ms, which is 231 tok/s per user at single-user 4096. That is the
optimistic ceiling of this direction, and the realistic target is the 13.77 ms of
GDN-layer work plus the 11.73 ms of all-layer work, both of which are project-owned
kernels rather than vendor ops.

**Next, in order:** identify the 99-core 64-call group, which is the single largest
sub-group at 11.40 ms and is currently unnamed; confirm the GDN recurrence row-scaling
with a T4/T8/T16 sweep from the same CSV family; then decide whether the recurrence can
be made row-parallel.

**Fix the profile job regardless.** It works, and CI calls it a failure. It needs the
tracy finalisation to stop trying to launch a web GUI, so the artifact is produced under
a green tick rather than rescued from a red one.

## REFINEMENT: the 99-core group is MLP gate/up, and weight streaming is near-optimal

Identifying operations by core count alone was not enough. Dumping one replay in
execution order shows a repeating per-layer block:

```
AllGatherAsync (10c, 13 us)  ->  LayerNorm (32c, 6 us)  ->  ShardedToInterleaved
  ->  GenericOp cores=99  178.5 us      MLP gate/up, bfloat4_b
  ->  Matmul    cores=32  122.1 us      MLP down, bfloat8_b
  ->  ReduceScatterMinimalAsync (10c, 11 us)  ->  BinaryNg residual (110c, 2 us)
```

So the 11.40 ms 99-core group is the **MLP gate/up projection**, which is weight-bearing,
not GDN. The previous section assigned it to "all-layer work" and implied it was
overhead. It is not. Corrected split:

| Component | ms | share |
| --- | ---: | ---: |
| weight-bearing (gate/up + all matmuls) | 31.64 | 49% |
| **GDN machinery** | **20.08** | **31%** |
| collectives | 3.44 | 5% |
| SDPA | 3.49 | 5% |
| everything else | 6.25 | 10% |

**Weight streaming is near-optimal.** 9.96 GB per card in 31.64 ms is **315 GB/s, 78% of
the measured 405 GB/s**, consistent with the 87% the older T8 profile showed for its
matmul groups. There is no 2.7x hiding in the weight path; the original reading of this
document was wrong and both corrections point the same way.

The GDN block, per GDN layer:

```
GenericOp 48c 18.0 us -> Matmul 43c 118.6 us -> GenericOp 48c 72.8 us
 -> GdnConvGates 81c 35.4 us -> Slice 96c -> GenericOp 8c 17.9 us
 -> GenericOp 96c 197.3 us   <- the recurrence
 -> GenericOp 24c 70.5 us -> Matmul 32c 46.2 us
```

**GDN recurrence is 197 us per layer, the most expensive single operation in the
verifier**, ahead of MLP gate/up at 178 us, and it runs on all 48 GDN layers for 9.53 ms.

### What is actually on the table

| Scenario | verifier | cycle | tok/s/user |
| --- | ---: | ---: | ---: |
| today | 64.9 | 97.6 | 123 |
| halve the GDN machinery | 54.9 | 87.5 | 137 |
| every non-weight op vanishes (unreachable) | 31.6 | 64.3 | 187 |

Even the unreachable bound is **187 tok/s at single-user 4096**, below the 200 target.
Verifier optimisation alone does not reach it; it has to be combined with higher
acceptance, which is the term that multiplies the budget.

### On the row-scaling claim

The earlier "1.52x for 2x rows" compares 6.29 ms at T8 from run 34298049648 on
2026-09-09 against 9.53 ms at T16 from run 35185624322 on 2026-09-17, which used a
different option set. Within this CSV only trace 24 has enough replays for a steady-state
median; the t2 and t4 verifier traces have a single session each and still include
compilation. **Treat the scaling as suggestive, not established.** A dedicated T4/T8/T16
sweep in one run, with several replays at each width, is what would settle it - and it
matters, because a recurrence that grows with verify rows taxes the exact mechanism
speculation relies on.
