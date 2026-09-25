# Prefill: a prediction, recorded before the profile lands

Written deliberately in advance. A profile read with an open mind tends to confirm
whatever the reader expected, so the discriminating signature is stated here first and
the profile either matches it or does not.

## What is established

Serving prefill is chunk-outer: `prefill_traced_chunked` replays a captured trace per
2048-token chunk, and every chunk runs all 64 layers.

For the measured 79,368-token prefill at 32.5 s (run 35419117141, 1 user x 131072):

| Quantity | Value |
| --- | ---: |
| chunks | 39 |
| weight traffic | 388 GB/card -> 1.0 s at 405 GB/s, **3% of wall** |
| compute required | 4.29 PFLOP -> 2.8 s at peak |
| measured | 32.5 s, **11.7x off compute-bound** |
| per chunk | ideal 71 ms, actual **833 ms** |

So prefill is neither bandwidth-bound nor close to compute-bound. Something consumes
about 760 ms per chunk that is neither.

Fitting `time = a*N + b*N^2` on two same-configuration points puts the shortfall in the
**per-token term**, not the quadratic one: `a` implies a 3,369 tok/s ceiling even with
attention free, and the quadratic share is 44% at 161k tokens. Falling tok/s at long
context is therefore substantially the quadratic term doing genuinely more work, which
is physics. The defect is in `a`.

## The hypothesis

The gated-delta-net layers. 48 of the 64 layers are GDN, and the scan runs 256-token
sub-chunks inside each 2048-token chunk, so a chunk contains `48 x 8 = 384` sequential
sub-steps. To account for the 833 ms they would need to average 2.17 ms each.

Supporting it, from the decode side: GDN machinery is 31% of the T16 verifier, and the
recurrence is the single most expensive operation there at 197 us per layer.

Against it: GDN is linear attention and cheap in FLOPs, so it would not appear in a FLOP
count at all - it would have to be costing wall time through utilisation rather than
arithmetic. And the projections are an equally good suspect with no prefill-time
efficiency measurement either way.

## The discriminating signature

The prefill device profile should show one of these. Call counts are the cleanest
discriminator because the two families run on different numbers of layers.

| If this is true | Then the prefill trace shows |
| --- | --- |
| **GDN owns the per-token cost** | GDN-family ops above 50% of device time, with call counts in multiples of **48**, and per-chunk sub-step counts near 384 |
| **Projections own it** | `MatmulDeviceOperation` plus the 99-core gate/up above 60%, with call counts in multiples of **64**, and the gap is matmul efficiency at `[2048, 5120] x [5120, 8704]` |
| **Neither** | time concentrated in layout, typecast or data-movement ops, which would point at conversion overhead rather than either compute path |

A fourth outcome is possible and worth naming: the profile may show the device busy
across many small ops with no single owner, which would mean the cost is dispatch or
serialisation rather than any one kernel, and the remedy is fusion rather than a faster
kernel.

## Caveats on the arithmetic

- Peak is taken as 1548 TFLOPS for two cards at the fp8 rate. Weights are bf4/bf8 but
  activations are bf16; if the matmuls execute at the bf16 rate the peak halves, the gap
  becomes 5.9x rather than 11.7x, and the prize is correspondingly smaller. The profile
  will not settle this on its own.
- The `a*N + b*N^2` fit rests on two points from one configuration. An earlier fit that
  mixed in a number from a different deployment mispredicted its own third point by 50%.
- The 833 ms per chunk assumes chunks are uniform. The first chunk of a request includes
  staging the RoPE table and any compilation, so the steady-state figure is lower.

## Why this matters beyond prefill

The 30k-prefill half of the goal depends on it directly, and the decode ladder has now
shown that **prefill does not amortise across users while decode does** - aggregate
prefill throughput is constant in user count. So prefill cost is paid per request in
full, and at 2,400 tok/s a 30k prompt costs 12.5 s of time-to-first-token that
concurrency cannot reduce.

## Preliminary evidence, and it argues against the hypothesis above

Found in the untraced rows of the existing decode profile (run 35185624322) before the
dedicated prefill profile ran. Recorded here rather than quietly dropped, because the
prediction was made in public and this is evidence against it.

| Op | ms | calls | per call |
| --- | ---: | ---: | ---: |
| `ChunkGdnScanOperation` | 200.6 | 192 | 1.04 ms |
| `MatmulDeviceOperation` | 2651.5 | 24,412 | 43.9% of the window |
| `AllGatherMinimalMatmulAsyncOp` | 700.5 | 512 | 1.37 ms |
| `DecodeGatedDeltaRuleDeviceOperation` | 143.1 | 3,072 | |
| `GdnConvGatesDeviceOperation` | 115.4 | 3,264 | |

`ChunkGdnScanOperation` is the GDN prefill scan, and 192 calls is 48 layers x 4 chunks.
At 1.04 ms per call that is about **50 ms of GDN scan per 2048-token chunk**, against a
measured 833 ms per chunk - roughly **6%**, not the more than 50% the hypothesis
requires. On this evidence the GDN layers are not what makes prefill slow.

`AllGatherMinimalMatmulAsyncOp` is new here: a fused collective-and-matmul, 512 calls,
which is 64 layers x 8, so it runs per layer per chunk and belongs to the projection
path rather than the GDN path.

**Why this is indicative rather than conclusive.** The window mixes prefill with every
other untraced operation in the run - setup, capture, untraced decode - so the totals are
not a prefill attribution. And the 833 ms per chunk comes from the plain path at 79k
context, where attention over a long cache is expensive, while this profile served a 4k
request whose chunks are cheaper. The two are not like for like.

The dedicated profile bounds prefill with signposts per prompt length and will give a
clean answer. If it confirms this, the revised hypothesis is the projection path and the
fused collective-matmul, and the remedy is matmul efficiency or fusion rather than
anything to do with the recurrence.

## RESULT: the prediction was wrong about GDN, and the first reading of why
was wrong too

> **CORRECTION, same day.** The section below grouped
> `AllGatherMinimalMatmulAsyncOp` as a collective. It is a *fused* all-gather and
> matmul, so its 1,096.7 ms of matmul was charged to the link. Collectives are
> **11-15%** of prefill, not 37.2%, and arithmetic is **39-43%**, not 16.9%. The
> conclusion that prefill is communication-bound does not survive, and neither
> does the sharding recommendation built on it. See
> `prefill-profile-corrected-2026-09-19.md` for the regrouping and
> `fabric-bandwidth-2026-09-19.md` for the measured link.
>
> What does survive: **the GDN hypothesis was wrong**, at 14.3% against a
> predicted 50%+, and **layout is 17.2%** and remains a target. The original text
> is kept below unedited, because a pre-registered prediction is worth nothing if
> its result is quietly rewritten.

### Original reading, superseded


Run 35422536834, 16 MB device report, three prefills of 2,568 / 10,248 / 20,488 tokens.
Chip 0, all operations, 37,203 calls over 36 op types, 3,697 ms of device time.

| Group | ms | share |
| --- | ---: | ---: |
| **collectives** (AllGatherMinimalMatmul, ReduceScatter, AllGather) | **1374.3** | **37.2%** |
| layout (Slice, Tilize, Untilize, Reshape, Sharded/Interleaved, Concat, Pad) | 636.1 | 17.2% |
| GDN family (ChunkGdnScan, ChunkGdnPrep, GdnConvGates) | 528.9 | 14.3% |
| elementwise and norm | 511.8 | 13.8% |
| plain Matmul | 488.7 | 13.2% |
| SDPA / attention | 136.5 | 3.7% |

**The hypothesis said GDN above 50%. It is 14.3%.** The preliminary evidence from the
untraced rows of the decode profile pointed this way and was right to.

**Actual arithmetic is 16.9% of device time** - matmul plus SDPA together. That is the
9x MFU gap, stated exactly: the chips spend about one sixth of prefill doing the model's
maths, and the rest moving data.

The single largest operation is `AllGatherMinimalMatmulAsyncOp` at 74 cores: 565 ms
across 467 calls, 15.5% on its own. It is already a fused collective-and-matmul, so the
obvious fusion has been done and this is what remains after it.

### What this inverts

Decode and prefill are opposites on this axis, which nothing in the earlier analysis
predicted:

| | collectives | arithmetic |
| --- | ---: | ---: |
| decode (T16 verifier) | 5% | 49% weight-bearing |
| prefill | **37%** | **17%** |

Decode is bandwidth-bound and communication is negligible. Prefill is communication-bound
and arithmetic is a minority of its time. An optimisation aimed at one is close to
useless for the other, and the intuition carried over from decode - that collectives are
cheap here - is exactly wrong for prefill.

### Where prefill work should go

1. **Collectives, 37.2%.** The fused all-gather-matmul is the biggest single item. Fewer,
   larger collectives, or a sharding that needs less of them.
2. **Layout, 17.2%.** 2,556 Slice calls and 636 ms of tilize/untilize/reshape is pure
   data shuffling. Candidate for fusion into neighbouring ops.
3. **Not GDN**, at 14.3%, and not attention at 3.7%.

The caveat from the original profiling work still applies: these are observed intervals,
durations include waits and can overlap, and they are not a dependency-graph critical
path.
