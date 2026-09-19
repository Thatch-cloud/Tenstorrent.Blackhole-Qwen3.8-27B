# Inter-card fabric: four links are up, and only about two links of bandwidth arrives

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

Whether 83.74 GB/s is good depends on how many cables carry the four links the runtime
reports, which is unresolved below. It is 84% of one cable or 42% of two.

## Four ethernet links are up. There is nothing to enable

Run 35425330364 read the cluster descriptor the runtime serialises
(`ttnn.cluster.serialize_cluster_descriptor()` returns a path; the container is `--rm`, so
it has to be read inside the run). Between the two chips:

```
ethernet_connections:
  - [chip 0 chan  8, chip 1 chan 5]
  - [chip 0 chan  9, chip 1 chan 4]
  - [chip 0 chan 10, chip 1 chan 7]
  - [chip 0 chan 11, chip 1 chan 6]
ethernet_connections_to_remote_devices: []
```

**Four links, all up**, between `0000:d1:00.0` and `0000:f3:00.0` - the M+A pair the Qwen
work uses. `get_cluster_type` is `ClusterType.P150_X2`. The mesh descriptor asks for
`channels { count: 4 policy: RELAXED }`, so the runtime is requesting all four and the
hardware is providing all four.

So the answer to "can we enable the other links" is that they are already enabled. The
gap is that the collective delivers far less than four links of bandwidth.

## How much is being left on the table depends on the cabling

| if the four links are | ceiling | measured 83.74 GB/s is |
| --- | ---: | ---: |
| **two cables**, two 400 Gb/s links each | 200 GB/s | **41.9%** - about 116 GB/s unused |
| **one cable**, four channels on one port | 100 GB/s | **83.7%** - near saturation |

The descriptor cannot separate these: chip 0 uses channels 8-11 and chip 1 uses 4-7, and
contiguous runs of four are equally consistent with one four-channel port or two
two-channel ports. **This is a question for whoever can see the back of the machine**, and
it decides whether there is a factor of two waiting or nothing at all.

The earlier reasoning in this document assumed a single cable and treated 83.74 GB/s as
saturation. That assumption is now explicitly unresolved rather than quietly load-bearing.

## RESULT: packet size was real but small, and the prediction was wrong

Runs 35425948829 and 35426073865. Three arms, one device open each.

| arm | packet | pages/packet | warnings | 512 | 2048 | 8192 | % of 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 4352 | 2 | 6 | 61.06 | 76.04 | 83.28 | 41.6% |
| packet 8192 | **8192** | **4** | **0** | 75.00 | 84.51 | **90.43** | 45.2% |
| + 4 planes | 8192 | 4 | 0 | 64.78 | 82.84 | 91.15 | 45.6% |

**The lever demonstrably moved.** `get_tt_fabric_max_payload_size_bytes()`
returned 8192 in the tuned arm against 4352 in the baseline, and the runtime's own
`suboptimal for transporting` warning went from 6 occurrences to 0. This is the first
lever this month that can be shown to have moved rather than assumed to have.

**The pre-registered prediction was 160-170 GB/s and it was wrong.** The result is
90.43 GB/s, +8.6% at the large shape. Packing was not the constraint.

The shape of the gain says what it actually was: **+22.8% at 512 rows, +11.1% at 2048,
+8.6% at 8192**. Largest where per-packet overhead dominates and smallest at the
asymptote is a fixed-cost reduction - it shaved the ~10 us per collective - not a raised
ceiling.

### The planes arm is inconclusive, and that was predicted too

`num_planes=4` moved the large shape from 90.43 to 91.15, which is 0.8% and inside noise,
and it made the small shape worse. There is **no getter for plane count**, so this arm
cannot distinguish "four planes do not help" from "num_planes was ignored", and the
0.8% agreement is the same signature that three inert levers produced this month. The
weakness of the control was recorded before the run rather than discovered after it.

Treat the planes question as **open**, not as answered in the negative.

### ROOT CAUSE: tt-metal hardcodes 2 links, and we filed it ourselves in September

The missing factor of two is **upstream issue
[#55125](https://github.com/tenstorrent/tt-metal/issues/55125), filed by this project on
2026-09-02**, and rediscovered from scratch today.

`models/common/modules/tt_ccl.py:148-181` looks up `link_dict` by a product name derived
purely from **device count** (`device_utils.py:26-32`). Two Blackhole devices resolve to
`"P300"`, and a real p300 is two dies on one package with two links, so `get_num_links()`
returns **2 whatever the cabling is**. The link count is a property of how the boards are
cabled, and UMD already discovers it - the cluster descriptor lists all four connections.

The arithmetic closes exactly. Two links at 50 GB/s is a 100 GB/s path, and the best
measurement is **90.43 GB/s, which is 90% of it**. Not 45% of a four-link fabric that is
running badly; **90% of a two-link fabric that is running well.** Nothing was inefficient
- half the fabric was never asked for.

That also explains every negative result today: `num_links` on the op cannot help because
the cap is applied inside the CCL layer, and `num_planes` cannot help because planes were
never the constraint.

**Overriding the value to 4 was already measured when the issue was filed: +2.4% decode.**
So the remaining fabric win is known, is small, and needs a source patch via the graft
pattern rather than any configuration available to us.

### Is it worth taking the 8.6%?

Honestly: barely, and it is not free to adopt.

| | share | after an 8.6% fabric gain |
| --- | ---: | ---: |
| prefill communication | 408-538 ms of 3697 | saves ~35-46 ms, **~1% of prefill** |
| decode collectives | 3.44 ms of 64.90 | saves 0.27 ms, **0.4% of the cycle** |

And the two-link fix on top of it is worth **+2.4% decode**, already measured. Both are
real, both are small, and neither is a route to the target.

Against a 32.4 ms gap in the decode cycle this is under 1% of what is needed. The packet
size is also a **serving default**, so changing it needs authorisation rather than being
applied quietly. Recommend recording it as a known-good setting and revisiting it if the
fabric ever becomes load-bearing, which on these numbers it is not.

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
- delivered bandwidth is **83.74 GB/s** across four links; the 400 GB/s figure was never measured
- the collectives that do run achieve **42-84 GB/s**, 50-100% of that measured ceiling
- fixed cost is ~10 us per collective; across ~2,950 collective calls that is about
  **29 ms of 3,697 ms, under 1%**, so "fewer, larger collectives" is also small

Dropping TP2 would cost decode 24.6 ms per step to buy back at most 15% of prefill, and
that trade does not come close under either cabling reading. **Do not pursue it.** If the
ceiling really is 200 GB/s, the remedy is the packet size or the collective's link usage,
both of which cost decode nothing.
