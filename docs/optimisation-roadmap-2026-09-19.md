# Optimisation roadmap, built from measurement rather than intuition

Where the remaining time actually is, what has already been tried, and what each
remaining lever would have to deliver. Written after the DRAM bandwidth measurement, the
T16 device attribution and the decode scaling ladder, because before those three the
question could not be answered.

Target for reference: **150 tok/s per user at 4 users, 131k context, bf8 KV**. Current
projected cycle at that configuration is **113.1 ms**, against an **80.7 ms** budget.
**32.4 ms has to come out.**

## What is ruled out, with the measurement that rules it out

### Weight streaming. 78% of achievable bandwidth; six attempts already failed

| Attempt | Outcome |
| --- | --- |
| MLP K-block, larger reduction blocks | regresses, rejected |
| MLP bulk pipeline, two-block overlap | 0.60% regression, rejected |
| MLP weight pipeline, two in-flight blocks | rejected |
| Fused MLP activation prefetch | ~2% regression, original kept |
| DRAM-core prefetcher | 24.7% slower, rejected |
| GDN gate/exp fusion | ~0.x% improvement, rejected |

Measured DRAM is **405 GB/s per card**; weight streaming inside the verifier achieves
**315 GB/s, 78% of it**. There was never headroom. Every one of those attempts was
bounded at a couple of percent before it was written, and the measurement that would have
said so takes an hour.

**Do not spend further effort here.**

### Raw DRAM bandwidth. 79-80% of spec is normal GDDR6

Flat across access sizes and consistent between read-only and read-write regimes, which
is a saturated memory system. A physically impossible 100% of spec would save 5.1 ms.

### Acceptance rate. Already ~76%, and perfect acceptance still misses

Measured on record at 224/300 and 104/126 proposals, so 75-83%; T16 commits 12.1 of 16
rows, 75.6%, consistent.

| committed per cycle | budget at 150 tok/s | vs 113.1 ms cycle |
| ---: | ---: | --- |
| 12.1 (measured) | 80.7 ms | short 32.4 ms |
| 13.0 | 86.7 ms | short 26.4 ms |
| 14.0 | 93.3 ms | short 19.8 ms |
| 16.0 (perfect, unattainable) | 106.7 ms | **short 6.4 ms** |

Even a perfect drafter does not reach the target. Acceptance is a **model-quality lever**
- it is a property of the DFlash2 drafter, not of the runtime - and it is not where this
gap closes.

### Concurrency. Already free, and already exploited

Decode amortises to eight users at 1.02x the latency for 7.9x the aggregate. There is
nothing left to win; the ladder confirms the decode cost model has **no user-count term**.
Worth knowing in the other direction: prefill does **not** amortise, so per-request
time-to-first-token cannot be improved by batching.

## What remains, ranked by measured size

The T16 verifier is 64.90 ms of which 31.64 ms is weight-bearing and near its ceiling.
The remaining **33.26 ms** is the entire remaining opportunity in decode, and nothing has
ever targeted it, because it was not attributed until a device profile was read.

| Target | ms | Notes |
| --- | ---: | --- |
| **GDN machinery** | **20.08** | recurrence alone is 9.53 ms, 197 us x 48 layers, the single most expensive op |
| draft | 24.62 | 15 sequential steps at 1.64 ms; appears near its own bandwidth floor |
| everything else in the verifier | 6.25 | unattributed residue |
| select / commit | 5.15 | |
| decode SDPA | 3.49 | |
| collectives | 3.44 | all-gather plus reduce-scatter; **not** the bottleneck, contrary to the obvious guess |

### 1. GDN machinery, 20.08 ms

The largest single item and the only one big enough to matter on its own. Prior attempts
(gate/exp fusion, outer-product/state-add fusion, norm prefetch) were rejected, but they
predate the attribution and were aimed at specific fusions rather than at the measured
cost. The recurrence at 197 us per layer over 48 layers is now known to be the most
expensive operation in the verifier, which was not known when those were tried.

Open question that changes the approach: does recurrence cost scale with verify rows?
Suggestive evidence puts it at 6.29 ms at T8 and 9.53 ms at T16, 1.52x for twice the
rows, but those are different runs with different option sets. **A T4/T8/T16 sweep in one
run would settle it**, and it matters because a recurrence that grows with rows taxes
speculation itself.

### 2. The draft, 24.62 ms

**Corrected by the profile.** The draft trace (trace 2 in run 35185624322: 23.05 ms
median over nine replays, against 24.62 ms measured) is **63% matmul and 37% other**, so
roughly 8.5 ms of it is not weight streaming. The earlier claim here that the draft "runs
near its own bandwidth floor" came from an arithmetic estimate, not a measurement, and it
was wrong in the same way the first verifier reading was wrong: dividing bytes by a
wall-clock interval that contained other work.

The trace identification rests on the timing match rather than a signpost, so treat the
mapping as probable rather than certain. Reducing the **number** of
proposals is the available lever, and T32 was already tried: it committed fewer tokens
(11.0) at higher latency, so more proposals is the wrong direction. Fewer proposals
lowers both draft time and committed tokens, which roughly cancels.

### 3. The residue, and why it is NOT a target

Now attributed from the profile, and it is a dead end: **3.44 ms spread across fifteen op
types**, the largest being AttnPrep at 0.895 ms and LayerNorm at 0.780 ms over 129 calls.
There is no single item to attack and no fusion that would collect a meaningful amount.

Recording it because a diffuse residue is a genuine finding: it means the verifier is
fully accounted for, and the remaining opportunity really is concentrated in the GDN
machinery rather than hiding in unexamined overhead.

The verifier now attributes to 64.91 ms against the 64.90 ms measured, so the accounting
is complete:

| | ms |
| --- | ---: |
| GenericOp (MLP gate/up + GDN family) | 32.59 |
| Matmul | 20.25 |
| decode SDPA | 3.49 |
| residue, 15 op types | 3.44 |
| all-gather | 2.01 |
| GDN conv gates | 1.70 |
| reduce-scatter | 1.43 |

## Prefill, separately

Prefill is **8.5x off compute-bound in the per-token term**, and 3% of its wall time is
weight traffic, so it is neither bandwidth-bound nor close to compute-bound. A chunk
takes 833 ms against a 71 ms ideal.

The hypothesis was the GDN layers; preliminary evidence from the untraced rows of the
existing profile **argues against it** - `ChunkGdnScanOperation` costs about 50 ms per
2048-token chunk, roughly 6% rather than the 50%+ the hypothesis needs. The dedicated
prefill profile settles it.

One caveat that halves or doubles the prize: peak is taken at the fp8 rate. Activations
are bf16, so if the matmuls execute at the bf16 rate the gap is 5.9x rather than 11.7x.
**`model_config.py` holds the math-fidelity settings and has not been extracted** - it
should be added to the next graft extraction, because it changes how much is on the table.

## Recommended order

1. **Read the prefill profile** when it lands, against the pre-registered prediction in
   `prefill-prediction-2026-09-19.md`. Prefill is the larger multiple off its ceiling
   (8.5x versus decode's 1.3x) and nothing has ever been aimed at it.
2. **Extract `model_config.py`** and settle the fidelity question. An hour, and it halves
   or doubles every prefill estimate here.
3. **T4/T8/T16 recurrence sweep in one run.** Settles whether GDN cost scales with verify
   rows, which decides whether attacking the recurrence helps speculation or merely
   shifts it.
4. **Then, and only then, GDN work in decode**, aimed at whatever the sweep and profile
   name, not at a fusion chosen in advance.

Against the 150 tok/s target specifically: the honest position is that 32.4 ms must come
out of a 113.1 ms cycle, the only pool large enough is the 33.26 ms of non-weight verifier
work, and that would mean eliminating essentially all of it. The target is not reachable
by tuning alone at 4 users and 131k. It is reachable at lower user counts or shorter
context - the decode cost model says exactly where, and that arithmetic is now reliable.

## Sharding: a trade, not a win, and probably the wrong lever

Prompted by the prefill profile putting collectives at 37.2% of device time.

### What TP2 buys and what it costs

TP2 exists to halve the decode weight pass, which is the dominant decode cost:

| | weights read per card per pass | at 405 GB/s |
| --- | ---: | ---: |
| TP2 (current) | 9.96 GB | 24.6 ms |
| data parallel (weights replicated) | 19.92 GB | 49.2 ms |

So dropping TP2 adds about 24.6 ms to a 97.6 ms decode cycle. In exchange it removes
all 37.2% of prefill collectives.

**Data parallel would fit.** Replicated weights plus KV comes to 22.6 GB at 65k context
and 25.2 GB at 131k, against 33.1 GB usable per card.

So the two regimes want opposite sharding, which is the same inversion the collectives
finding showed. **TP2 is a decode optimisation that prefill pays for.**

### Why it is probably still the wrong lever

The links are four QSFP-DD at 800 Gb/s, so 100 GB/s per port and **400 GB/s aggregate**
in theory. The rate implied by the prefill profile is **9.3 GB/s**, which is 2.3% of
aggregate and 9.3% of a single port. DRAM on the same cards achieves 79-80% of its spec,
which is what a saturated resource looks like.

A resource running at 2% of capability is not saturated. That points at a fixed cost per
collective rather than a bandwidth limit, and the remedy for that is **fewer and larger
collectives, which costs decode nothing**, rather than a sharding change that costs
decode 24.6 ms per step.

Two measurements decide it, both queued:

1. **Inter-card collective bandwidth**, swept by size. Rising GB/s with size means a
   fixed per-collective cost and confirms the reading above. Flat near 400 means the
   link really is the limit and sharding becomes the only remedy.
2. **The prefill chunk sweep**, which halves the number of collectives by doubling the
   chunk. If prefill time falls roughly in proportion, the cost is per-collective and
   the fix is a constant rather than an architecture.

### Caveat on the 9.3 GB/s

It is derived, not measured: expected all-gather traffic computed as
`layers x chunks x tokens x (hidden/2) x 2 bytes`, divided by the profiled collective
time. If the traffic model is wrong the ratio moves with it. That is precisely why the
probe exists; the DRAM figure was assumed at 85% of spec and measured at 79%, and the
gap here is 43x rather than a few percent.
