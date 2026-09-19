# Inter-card fabric: measured at 83.74 GB/s, which is one cable fully used

Run 35424379930. Answers a question the prefill profile raised and could not settle:
collectives looked like 37% of prefill device time, and the traffic model implied they
were running at 9.3 GB/s. Either the link is slow, or the reading was wrong.

The reading was wrong. Both of them.

## What the fabric actually delivers

`ttnn.all_gather` across the pair, bfloat16, 30 iterations, hidden sharded on the last
dimension so each card receives the other half:

| shape | bytes per call | us per call | GB/s |
| --- | ---: | ---: | ---: |
| 512 x 5120 | 2.62 MB | 40.6 | 64.49 |
| 2048 x 5120 | 10.49 MB | 137.2 | 76.44 |
| 8192 x 5120 | 41.94 MB | 500.9 | **83.74** |

Fitting `time = fixed + bytes / rate` across those three points gives **about 10 us of
fixed cost per collective and an asymptote near 85 GB/s**. Rising GB/s with size is what a
per-collective overhead looks like; the overhead is real but small.

**83.74 GB/s is 84% of one QSFP-DD800 cable**, which carries two 400 Gb/s links for
100 GB/s. That is a saturated cable, in the same way that 405 GB/s is a saturated GDDR6
subsystem at 79% of spec - and because it exceeds one 50 GB/s link, both links of that
cable are demonstrably in use.

## The Ethernet fabric is up, and at least two links are already carrying traffic

Worth stating because the physical topology invites the opposite guess: the second card
sits behind a gen5 MCIO switch and negotiates x4, so it is reasonable to suspect the
traffic never reaches Ethernet at all.

It does. From the run log:

```
Fabric initialized on 2 devices
Fabric Initialized with config FabricConfig::FABRIC_1D
Using custom mesh graph descriptor: p150_x2_mesh_graph_descriptor.textproto
```

And the arithmetic rules PCIe out independently: gen5 x4 is about 15.8 GB/s and gen5 x16
about 63 GB/s, so a measured 83.74 GB/s cannot have crossed either. PCIe width is not in
this path, consistent with the earlier finding that it costs about a second once at model
load and nothing per step.

**One QSFP-DD800 cable carries two ethernet links of 400 Gb/s each**, so a link is
50 GB/s and a cable is 100 GB/s. That fixes the reading:

- 83.74 GB/s is **above** a single 400 Gb/s link, so **at least two links are already
  active**. "Only one link is in use" is false.
- 83.74 GB/s is **84% of one cable**, which is what one fully-used cable looks like.

So the open question is not whether links are being used, but **how many cables connect
the pair**. One fully used and two half used produce similar bandwidth, so the number
cannot be inferred from it. Run 35425107580 asks the runtime directly, via the per-peer
ethernet socket count.

> **CORRECTION.** An earlier version of this document read
> `intra-mesh degree histograms mesh0 {1:2}` as proof of a single link. That was wrong.
> Degree counts *neighbours*, not edges: with two chips in the mesh each has exactly one
> neighbour however many cables run between them, so the line is true of any two-node
> mesh and says nothing about link count. The measurement above is what constrains the
> answer; that log line never did.

## num_links is deprecated, so the link sweep proves nothing

The sweep returned 76.44 / 77.63 / 77.43 GB/s at one, two and four links: identical to
within 1.6%. Two genuinely different configurations do not agree that closely, so the
lever had not moved.

The probe recorded `num_links_accepted: true` on every arm, which was meant to be the
control. It is too weak a control, because the runtime says:

```
The following ttnn.all_gather args are deprecated and will be removed in
September-2026: num_links, topology, chunks_per_sync, num_workers_per_link, ...
```

The argument is **accepted and ignored**. A `try/except TypeError` guard cannot see that,
because nothing raises. This is a third distinct way for an experimental lever to be
silently inert, alongside a default that never gets overridden and a fallback that
swallows the difference:

| failure | looks like | control that catches it |
| --- | --- | --- |
| argument rejected | arms identical | catch the exception, record it |
| argument accepted but ignored | arms identical | **grep the log for the new behaviour** |
| code path never entered | arms identical | assert the path was taken, in the run |

All three produce the same symptom. **Suspiciously exact agreement between arms means the
arms were not different** - it is never a physical result until the lever is proven to
have moved.

Here the deprecation warning explains the flat sweep on its own: the runtime chooses its
own link count from what the fabric offers, and the argument asking for more is ignored.
Whether more links exist to be chosen is a separate question, measured in run 35425107580.

## A real tuning knob, volunteered by the runtime

```
Fabric packet size 4352 B is suboptimal for transporting 2048 B pages.
Configure 8192 B packet size to maximize throughput.
```

The runtime is naming its own misconfiguration. 2048-byte pages into 4352-byte packets
wastes roughly half of each packet. This is cheap to try and is the only fabric-side lever
this measurement supports.

It is also bounded: communication is 11-15% of prefill device time (see
`prefill-profile-corrected-2026-09-19.md`), so even a perfect fix is worth a few percent.
Worth doing, not worth a programme.

## What this kills

The sharding trade. It was argued from collectives being 37.2% of prefill and running at
2.3% of a 400 GB/s aggregate - a resource that idle must be overhead-bound, and the
remedy for overhead is fewer, larger collectives or a different sharding.

Every input to that argument is now wrong:

- collectives are **11-15%** of prefill, not 37.2%
- delivered bandwidth is **~84 GB/s**, one cable; the 400 GB/s aggregate was never measured
- the collectives that do run achieve **42-84 GB/s**, 50-100% of that measured ceiling
- fixed cost is ~10 us per collective; across ~2,950 collective calls that is about
  **29 ms of 3,697 ms, under 1%**, so "fewer, larger collectives" is also small

A resource at 50-100% of its measured capability is saturated, not idle. Dropping TP2
would cost decode 24.6 ms per step to buy back at most 15% of prefill. **Do not pursue
it.**
