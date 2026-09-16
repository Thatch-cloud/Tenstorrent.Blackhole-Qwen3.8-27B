# Pinned kernel profiling inventory

Run **35092212895**, revision `341059a`, completes in **12 seconds**. It reads
14 source files from the existing pinned image, without model weights, device
access, network access or changes to the image. No expected paths are missing.
The exported GDN compute, reader and writer match all three exact hashes used
by `gdn_multitoken.py`. Evidence is in the run's `qwen-kernel-inventory` artifact.

## What this establishes

| Facility | Pinned source evidence | Still unproven |
| --- | --- | --- |
| Named device zones | `kernel_profiler.hpp` defines `DeviceZoneScopedN` | Marker retention and useful attribution in our request capture |
| Hardware counter definitions | `perf_counters.hpp` declares FPU, PACK, UNPACK, L1 and instruction groups | Which groups the compiled Blackhole runtime enables and exports |
| Original GDN math | Exported native sources match our required hashes | Which internal stage dominates current latency |
| Existing device CSV | Per-RISC duration columns, no CB-wait or counter fields | Useful compute versus time waiting inside a kernel |

Source definitions are not measured counters. Do not report unsupported fields
as zero, infer utilization from kernel duration, or apply upstream features on
the assumption that the pinned profiler exposes them.

## Next measurement boundary

Stop input/weight-buffer capacity sweeps: both the four-block and full-K MLP
variants failed the combined-runtime performance test. Retain incremental
publication and norm prefetch, then add bounded named scopes to a representative
MLP/GDN operation inside that exact combined request. Separate input readiness,
weight transfer completion, compute/pack and output backpressure. Keep numerical
and state audits, label the run instrumented, and reject missing/dropped scopes.
Do not time an isolated kernel and report it as committed TG.

Instrument only selected operations/workers initially to bound marker volume.
The pinned `DeviceZoneScopedN` macro declares fixed local names; each injected
zone needs its own lexical block, and its lifetime must cover the intended wait
or work rather than merely recording an instantaneous marker. These source
changes must preserve buffer ownership and synchronization order.
