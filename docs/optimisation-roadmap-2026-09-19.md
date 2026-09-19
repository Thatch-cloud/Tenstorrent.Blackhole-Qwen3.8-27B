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

The hypothesis was the GDN layers. **The profile refutes it**: the GDN family is 14.3% of
prefill device time against the 50%+ the hypothesis needed, and the preliminary evidence
from the untraced decode rows had already pointed that way.

Where prefill time actually goes, from run 35422536834 regrouped
(`prefill-profile-corrected-2026-09-19.md`):

| group | share |
| --- | ---: |
| arithmetic - matmul, fused matmul, SDPA | **39-43%** |
| layout and data movement | 17.2% |
| GDN family | 14.3% |
| communication | 11-15% |
| elementwise, norm, the rest | ~16% |

About 40% of prefill is already arithmetic while prefill is 11.7x off compute-bound, so
the matmuls are **slow when they run** rather than crowded out. That is the live question,
and it is not the one this roadmap previously queued.

One caveat that halves or doubles the prize: peak is taken at the fp8 rate. Activations
are bf16, so if the matmuls execute at the bf16 rate the gap is 5.9x rather than 11.7x.
**`model_config.py` holds the math-fidelity settings and has not been extracted** - it
should be added to the next graft extraction, because it changes how much is on the table.

## Recommended order

1. **Extract `model_config.py` and count the parameters.** Now first, not second. Both
   the fidelity setting and a counted FLOP requirement are needed before "prefill matmuls
   are inefficient" is a measurement rather than a reading, and that claim is currently
   carrying the whole prefill case. An hour, and the 9.3 GB/s episode is what happens when
   a derived number is built on instead.
2. **Prefill layout, 17.2%.** 6,643 `Slice` calls plus tilize and untilize is pure
   shuffling, it is the second largest group, and unlike the matmul question it needs no
   further measurement to start on.
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

## Sharding: ruled out, and the argument for it was built on a misreading

Superseded in full. The earlier version of this section argued that TP2 is a decode
optimisation prefill pays for, on the grounds that collectives were 37.2% of prefill
device time while running at 9.3 GB/s, or 2.3% of a 400 GB/s aggregate. A resource that
idle is overhead-bound, and the remedies for overhead are fewer collectives or different
sharding.

**Every input to that argument was wrong**, and the two measurements queued to test it are
what showed it.

| input | as argued | measured |
| --- | ---: | ---: |
| collectives, share of prefill | 37.2% | **11-15%** |
| fabric aggregate | 400 GB/s over four links | **~84 GB/s over one link** |
| rate collectives achieve | 9.3 GB/s, 2.3% of aggregate | **42-84 GB/s, 50-100% of it** |
| fixed cost per collective | assumed to dominate | ~10 us, **under 1% of prefill in total** |

Two independent errors, compounding:

1. `AllGatherMinimalMatmulAsyncOp` is a **fused** all-gather and matmul. Counting it as a
   collective charged 1,096.7 ms of matmul to the link and roughly tripled the apparent
   communication share. Details in `prefill-profile-corrected-2026-09-19.md`.
2. The 9.3 GB/s was traffic divided by that same inflated time bucket, so it understated
   the achieved rate by the same factor. It was flagged as derived rather than measured,
   and the flag was the right instinct - the measurement moved it by 4.5x to 9x.

And the fabric is not what the 400 GB/s figure assumed. The run log reports
`intra-mesh degree histograms mesh0 {1:2}`: two nodes of degree one, so the mesh
descriptor gives this pair **a single link**. `num_links` is accepted and ignored - it is
deprecated in this runtime - which is why the one/two/four sweep returned identical
numbers. See `fabric-bandwidth-2026-09-19.md`.

**Do not pursue a sharding change.** Dropping TP2 costs decode 24.6 ms per step to buy
back at most 15% of prefill, against a resource already running at half to full its
measured ceiling. The trade was never close once the numbers were right.

**Do try the packet size.** The runtime volunteers
`Fabric packet size 4352 B is suboptimal for transporting 2048 B pages. Configure 8192 B`.
That is cheap, low-risk and the only fabric lever the measurement supports - and it is
bounded at a few percent of prefill, so it is a knob rather than a programme.

## What this episode is worth keeping

Three levers this month produced arms that agreed to within 2%, and each time the
agreement was the tell rather than the result:

| lever | why it never moved | what would have caught it |
| --- | --- | --- |
| prefill chunk size | `_chunked_chunk_size or 2048` default never overridden | log the value in the run |
| fabric `num_links` | accepted but deprecated and ignored | grep the log for the new behaviour |
| M1 resumable prefill | dispatch condition never true | assert the path was taken |

**Suspiciously exact agreement between arms means the arms were not different.** Two
genuinely different configurations do not agree to within a percent. Treat it as a defect
in the experiment until the lever is proven to have moved, and prove it from the run's own
output rather than from the code that was supposed to set it.

The companion rule, from the 9.3 GB/s: **a derived number is a hypothesis, not a
measurement.** Both the DRAM figure and this one were assumed before they were measured;
DRAM came back 6% off its assumption and this came back 450-900% off. The cost of
measuring was an hour in each case.
