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

## Sampled MLP diagnostic qualification

Export inventory run **35093041332** passed in **14 seconds**, but its guessed
`tt_metal/tools/tracy` directory was absent. The pinned upstream tree locates
the exporter under `tools/tracy`; the inventory now uses that path.
At revision `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`,
`tools/tracy/__main__.py` maps `--disable-device-data-dump-to-files` to
`TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1`. `process_device_log.py` consumes
zone name, phase, source location and trace identity fields from raw device CSV.
Our current combined capture disables that raw export; aggregate RISC timings
alone cannot establish internal waits.

`frozen_mlp_wait_zones.py` prepares nine diagnostic scopes around existing waits
and barriers, sampled at K-block 10 on input workers 0/1 and weight worker 0.
Output readiness/write completion use only the first output pair on worker 0.
Each branch executes the original statement once; removing the generated scopes
must recover the complete original source byte-for-byte after newline normalization.
Buffer sizes, transfers, math and semaphore order are unchanged. Four host tests
cover that transformation, duplicate instrumentation and source drift rejection.

The simulator lane requests device profiling and retains eager, changed-input
replay and byte-exact weight checks, with a 12-minute whole-job cap and no model
weight loading. This is not yet simulator or hardware qualified. Raw marker
retention must still be verified before attributing hardware stalls. A barrier
scope measures remaining completion wait, not the whole transfer or bandwidth.
No serving default or performance claim changes.

First simulator attempt **35146501486** failed after **3m59s** (exit 134),
not a timeout. Its T16 changed-input replay matrix passed, but the overall report
remained false: native profiler processing aborted on mismatched TRISC kernel/FW
markers with sentinel trace identities. This attempt enabled device profiling
without trace tracking. The retry adds `TT_METAL_PROFILER_TRACE_TRACKING=1`, as
used by the existing combined capture; kernel sources and correctness checks
remain unchanged. This is a diagnosis to test, not a proven fix. No hardware
admission follows from the partial replay result.
