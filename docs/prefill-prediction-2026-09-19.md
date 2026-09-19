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
