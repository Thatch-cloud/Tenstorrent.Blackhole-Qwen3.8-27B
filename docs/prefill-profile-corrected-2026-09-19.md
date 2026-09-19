# Prefill device profile, regrouped correctly

Run 35422536834, chip 0, three prefills of 2,568 / 10,248 / 20,488 tokens. 37,203 calls
over 36 op types, **3,696.8 ms** of `DEVICE KERNEL DURATION`.

Supersedes the grouping in `prefill-prediction-2026-09-19.md`, which put collectives at
37.2% of device time and arithmetic at 17%, and on that basis argued for a sharding
change. Reproduce with:

```
python scripts/ci/prefill_profile_groups.py --csv .../cpp_device_perf_report.csv
```

## The error

`AllGatherMinimalMatmulAsyncOp` was counted as a collective. It is a **fused all-gather
and matmul** - the name says so - so putting all 1,096.7 ms of it in a communication
bucket charged its matmul time to the link. That one misassignment is the whole inversion:
it is 29.67% of device time on its own, against 7.51% for every op that does nothing but
move bytes.

| bucket | ms | share |
| --- | ---: | ---: |
| `AllGatherMinimalMatmulAsyncOp` (**fused**, comm + matmul) | 1096.7 | 29.67% |
| `MatmulDeviceOperation` | 488.7 | 13.22% |
| `ChunkGdnScanOperation` | 325.7 | 8.81% |
| `ReduceScatterMinimalAsyncDeviceOperation` (pure comm) | 260.7 | 7.05% |
| `SliceDeviceOperation` | 226.0 | 6.11% |
| `ChunkGdnPrepOperation` | 191.3 | 5.17% |
| `SDPAOperation` | 134.0 | 3.62% |
| `AllGatherAsyncDeviceOperation` (pure comm) | 16.9 | 0.46% |

`AllGatherAsyncDeviceOperation` is 898 calls at 18.8 us against `LayerNormPreAllGather`'s
897 calls, so it is the layernorm-statistic gather - a few hundred bytes, not activations.
The activation gather lives inside the fused op.

## Splitting the fused op by traffic

The profile cannot separate the two halves, so bound it instead. TP2 shards hidden, so
each card must receive the other half once per layer per token:

```
33,304 tokens x 64 layers x 2,560 cols x 2 bytes = 10.91 GB per combine-per-layer
```

The reduce-scatter carries exactly one such combine per layer and takes 260.7 ms, so it is
achieving **41.9 GB/s - 50% of the 83.74 GB/s** measured by the fabric probe
(`fabric-bandwidth-2026-09-19.md`). Pricing the fused op's gather at the probe ceiling is
the optimistic end; pricing it at the reduce-scatter's own observed rate is the
pessimistic end.

| | communication | arithmetic |
| --- | ---: | ---: |
| optimistic (gather at 83.7 GB/s) | 407.9 ms, **11.0%** | 1589.5 ms, **43.0%** |
| pessimistic (gather at 41.9 GB/s) | 538.2 ms, **14.6%** | 1459.2 ms, **39.5%** |
| **as previously reported** | 1374.3 ms, 37.2% | 625.2 ms, 16.9% |

Both ends of the bound land in the same place, which is what makes the conclusion safe
despite the estimate: **communication is 11-15%, arithmetic is 39-43%.** The previous
figures were wrong by roughly a factor of three in each direction, and they pointed the
work the wrong way.

## Corrected picture of where prefill time goes

| group | ms | share |
| --- | ---: | ---: |
| arithmetic (matmul, fused matmul, SDPA) | 1459-1590 | **39-43%** |
| layout and data movement | 637.1 | 17.2% |
| GDN family | 528.9 | 14.3% |
| communication | 408-538 | **11-15%** |
| elementwise, norm, everything else | ~600 | ~16% |

What survives from the original reading:

- **The GDN hypothesis was wrong.** Predicted above 50%, measured 14.3%. That stands, and
  the preliminary evidence from the untraced decode rows called it correctly.
- **Layout is 17.2%** and remains a genuine target: 6,643 `Slice` calls, plus tilize and
  untilize, is pure shuffling.

What inverts:

- Prefill is **not** communication-bound. It was never communication-bound.
- Decode and prefill are **not** opposites on this axis. Decode is bandwidth-bound with
  negligible communication; prefill is arithmetic-heavy with modest communication. The
  "an optimisation for one is useless for the other" framing was an artefact of the
  misgrouping.

## RESULT: the matmuls are fine. Counted, not inferred

Run 35426279454 extracted `config.json` and `model_config.py`. Reproduce with
`scripts/ci/model_param_count.py --config <config.json>`.

The architecture is nothing like the one earlier estimates assumed. `config.json` is a
multimodal wrapper and the language model lives under `text_config`:

| | assumed | actual |
| --- | ---: | ---: |
| intermediate_size | 8704 | **17408** |
| head_dim | 128 | **256** |
| num_attention_heads | 40 | 24 |
| vocab_size | 151936 | 248320 |
| full_attention_interval | - | 4, so **16 full / 48 GDN** confirmed |

Counted parameters: **25.36 B** (body 22.82 B), which matches the 27B in the name once the
MTP layer is included. Weights are `bfloat8_b`, activations `bfloat16`, read directly from
`model_config.py`.

### The fidelity question answers itself

Prefill for the profiled 33,304 tokens is **1.52 PFLOP**, so 0.76 PFLOP per card under TP2.

| rate | ideal | vs 3697 ms device time | MFU in the 1459-1590 ms arithmetic window |
| --- | ---: | ---: | ---: |
| LoFi, 774 TFLOPS/card | 982 ms | **3.77x** | **62-67%** |
| HiFi2, 387 TFLOPS/card | 1964 ms | 1.88x | 124-135% - **impossible** |

`model_config.py` carries no explicit `MathFidelity`, but it does not need to: the HiFi2
ideal of 1964 ms **exceeds the 1590 ms that arithmetic ops actually occupied**, and work
cannot take less time than its own floor. So the matmuls demonstrably execute at the
faster rate, and the measurement settles what the config did not state.

### This reverses the recommendation

The roadmap said prefill's problem was matmul efficiency and put "extract model_config.py"
first precisely so that claim could be checked. Checked, it fails:

- **Prefill is 3.77x off compute-bound, not 11.7x.** The 11.7x came from dividing a
  wall-clock serving prefill by a peak derived from the model's *name*. Both halves were
  wrong.
- **The matmuls run at 62-67% of peak when they run.** That is respectable. There is no
  factor of three hiding in them.

The real shape of prefill is that **arithmetic occupies only about 40% of device time**:

| group | share | is it a target? |
| --- | ---: | --- |
| arithmetic | 39-43% | closed again, on better evidence: prefill runs **1.75x a naive `ttnn.matmul`**, so its matmuls are well tuned. The 62-67% MFU figure stays withdrawn. |
| layout | 17.2% | **yes** - 6,643 Slice calls, pure shuffling |
| GDN family | 14.3% | maybe - it is real work, but 48 layers of it |
| communication | 11-15% | no - fabric measured, bounded at ~1% of prefill |
| elementwise and norm | ~16% | **yes** - fusion candidates |

**The opportunity is the ~60% that is not arithmetic**, and layout plus elementwise is a
third of prefill on its own. That is a fusion and data-movement problem, which is a
different programme from the kernel-efficiency one the roadmap had queued.

### What this did not settle

`prefill_pflop` counts the body's projections. It does not count the attention quadratic
term, which matters at long context and not at the 33k profiled here, nor the GDN
recurrence, which is cheap in FLOPs and expensive in wall time. So 62-67% is the MFU of
the projection work, and the GDN recurrence remains unpriced.

## What to do next

Both inputs this section used to ask for have landed, and they closed the question rather
than sharpening it. The matmuls are not the problem, so the order is:

1. **Layout, 17.2%.** 6,643 `Slice` calls plus tilize and untilize is pure shuffling and
   the largest non-arithmetic group. Needs no further measurement to start on.
2. **Elementwise and norm, ~16%.** Fusion candidates in the same vein.
3. **Price the GDN recurrence.** 14.3% of prefill, and the FLOP count above deliberately
   does not cover it, so its efficiency is genuinely unknown rather than assumed good.
4. **Not arithmetic, not communication, not sharding.** All three are measured and closed.

## Caveat carried forward

These are observed intervals. Durations include waits, can overlap, and are not a
dependency-graph critical path. The traffic split is an estimate bounded at both ends, not
an observation.

## Grid occupancy: a third of prefill runs on under a third of the chip

Found by circling back rather than by a new run - the column was in the profile all
along. `AVAILABLE WORKER CORE COUNT` is **110 for all 37,203 calls**, so the whole
grid is available at every moment.

| cores used | ms | % of prefill | biggest contributor |
| --- | ---: | ---: | --- |
| **exactly 1** | 204.0 | **5.5%** | AllGatherMinimalMatmul, 149 ms |
| 2-8 | 294.2 | 8.0% | AllGatherMinimalMatmul, 90 ms |
| 9-32 | 745.7 | 20.2% | ReduceScatterMinimalAsync, 151 ms |
| 33-64 | 486.2 | 13.2% | AllGatherMinimalMatmul, 118 ms |
| 65-99 | 1292.5 | 35.0% | AllGatherMinimalMatmul, 656 ms |
| 100-110 | 674.2 | 18.2% | Matmul, 156 ms |

**33.6% of prefill device time, 1,244 ms, runs on 32 cores or fewer.** Only 18.2%
uses 100 or more. Core-weighted occupancy over the whole of prefill is **52.8%**.

The single largest operation is the worst offender twice over: it averages 51.3
cores, and **13.6% of its time - 148.8 ms over 109 calls at 1.37 ms each - runs on
exactly one core**. A millisecond and a third on 1 of 110 cores is a serialised
step, not a small tensor.

### This puts a recorded conclusion in doubt

Earlier today this document recorded prefill matmuls at 62-67% MFU and closed
arithmetic as a target on that basis. That figure is a ratio of an assumed
full-grid peak to measured time, and it does not survive contact with occupancy:

```
arithmetic ops: 1719 ms, mean 55.8 cores = 50.7% grid occupancy
MFU = occupancy x per-core efficiency
65% / 51% = 127% per-core efficiency
```

Over 100% is impossible. Occupancy is **measured**, from the profiler's own
`CORE COUNT`; the 774 TFLOPS per card peak was **assumed** and never verified. So
the peak is the suspect input, and **"the matmuls are fine" is no longer
established**. It is withdrawn pending a measured peak, not replaced with a
contrary claim.

This is the third time today an assumed constant has produced a confident wrong
reading, after the 400 GB/s fabric aggregate and the model-name FLOP count. The
pattern is specific enough to name: **a derived percentage is only as good as its
denominator, and a spec-sheet denominator has been wrong every time it has been
checked on this rig.**

### Why this is a different lever from everything closed today

Sharding, communication, the causal conv and layout were all about *what work is
done*. Occupancy is about *how much of the chip does it*. Nothing measured so far
touches it, and it is not bounded by the same arguments:

- the conv verdict said anything leaving TILE layout lands where the FIR does -
  irrelevant to spreading work across cores
- the arithmetic verdict said the matmuls are efficient - which is exactly what is
  now in doubt
- a matmul's core grid is a **program-config choice**, not a property of the maths

What is not yet known, and must be measured before any claim: how much of the
low-occupancy time is structural. Communication ops legitimately use few cores, and
`ReduceScatterMinimalAsync` at a mean of 6.7 cores may be correct by design. The
149 ms of single-core time inside a *matmul* op is the part that looks wrong.

## Peak FLOPS: measured, and the measurement did not answer the question

Run 35429739377. All 24 configurations passed the compute-bound guard, so these are
arithmetic rates and not DRAM in disguise.

| shape | weights | LoFi | HiFi2 | HiFi3 | HiFi4 |
| --- | --- | ---: | ---: | ---: | ---: |
| 8192^3 | bfloat8_b | **252.6** | 218.6 | 159.4 | 125.4 |
| 8192^3 | bfloat16 | 252.2 | 219.2 | 159.1 | 125.2 |
| 2048x5120x17408 (the MLP shape) | bfloat8_b | 219.9 | 192.6 | 143.0 | 113.4 |
| 4096^3 | bfloat8_b | 215.6 | 192.5 | 145.1 | 116.6 |

Two things fall out immediately, and both are solid:

- **Fidelity is a 2x lever.** LoFi to HiFi4 is 252.6 to 125.4. model_config.py sets no
  explicit MathFidelity, so whatever the matmuls inherit matters by a factor of two.
- **bfloat8_b buys no speed at all.** 252.6 against 252.2 for bfloat16 weights, and the
  same at every fidelity and shape. The narrower weight dtype is a memory economy here,
  not an arithmetic one. Any reasoning that assumed an fp8 *rate* was wrong on that
  ground alone.

### The number cannot be used as a peak, because prefill beats it

| | TFLOPS/card |
| --- | ---: |
| naive `ttnn.matmul`, best measured | 252.6 |
| **prefill's own arithmetic** (0.76 PFLOP/card in 1719 ms) | **442** |
| prefill, matmul portion only | ~507 |

Prefill runs at **1.75x the benchmark**. A model cannot exceed the hardware peak, so
252.6 is a **floor** on the peak and not the peak: the probe measured an untuned
`ttnn.matmul`, while the model uses tuned program configs. The benchmark was built to
supply a denominator and instead demonstrated it was the wrong instrument.

**So MFU still has no valid denominator**, and the honest position on prefill
arithmetic is:

- the 62-67% MFU figure stays **withdrawn**; it rested on an assumed 774
- the 774 figure is **neither confirmed nor refuted** - a naive benchmark being far
  below peak refutes nothing
- but prefill's matmuls achieve 1.75x what a default matmul does, which is direct
  evidence they are **well tuned**, arrived at without needing any peak at all

That last line reinstates this morning's conclusion - arithmetic is not the prefill
lever - on evidence that does not depend on a spec sheet. It was the right answer for
the wrong reason.

### What this leaves of the occupancy finding

Prefill's arithmetic averages 55.8 cores and achieves 442 TFLOPS/card, which is about
**7.9 TFLOPS per core in use**. The naive benchmark, left to choose its own grid, got
252.6 overall. Nothing here shows the busy cores are underperforming.

So the occupancy number stands as measured - 33.6% of prefill runs on 32 cores or
fewer - but it can no longer be read as "half the chip is being wasted on the
arithmetic". The arithmetic ops are dense and fast. The low-occupancy time is
concentrated in **communication and the serialised single-core step inside the fused
all-gather-matmul, 148.8 ms over 109 calls**, and that remains the one piece that
looks wrong rather than structural.

### Getting a real peak, if it is ever needed

It would take a tuned matmul: an explicit program config with the full core grid and
blocking chosen for the shape, not the library default. That is a real piece of work
and, given prefill already exceeds the default by 1.75x, it would only refine a number
that is no longer blocking any decision.
