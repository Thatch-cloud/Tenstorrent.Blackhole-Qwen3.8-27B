# Contiguous BF4 weight blocks

**Unqualified transport candidate. No throughput result or runtime promotion.**

The latest combined DFlash2 candidate spends about 63 ms in verification per
block. Earlier combined MLP timing found operand-readiness stalls; deeper
buffers, bank ordering and two-inflight reads did not improve verifier time.
This candidate changes data placement instead of repeating those schedules.

| Property | Current fused MLP | Proposed block stream |
| --- | --- | --- |
| Weight precision | BF4 | Same packed bytes, including exponents |
| Per-worker K-block payload | 48 separate 576-byte tiles | One contiguous 27,648-byte span |
| Compute / register epilogue | Current recipe | Unchanged |
| Weight CB | Two blocks | Same two blocks |
| Packing | Native tile order | One-time byte-only gather before traces |

One contiguous span can still require multiple NoC packets. This does not
claim a single physical transaction or a predicted speedup.

The stream is block-major across 91 workers, rather than worker-major, to
avoid concentrating the first block on a small subset of interleaved banks.
The last worker's unused pair is zero-filled. Host tests cover all 87,040
native tiles exactly once, ordered correctly, plus 320 padding tiles.

## Memory and acceptance

One packed layer requires **50,319,360 bytes/card**. Keeping all 64 alongside
native weights adds approximately **3.00 GiB/card**. Do not hide this allocation
or release borrowed native weights. Full-runtime memory admission is required;
packing time belongs in setup measurements, not steady-state TG.

1. Weight-free simulator: check every raw word, padding and source preservation
   for two patterns, both chips, a multi-block tail case and all 91 workers.
2. Actual BF4 fused MLP: validate unchanged register epilogue, eager output,
   changed-input trace replay and byte integrity against the native recipe.
3. Only then run complete-request ABBA with the current combined stack,
   exact target output/state, feature checks, PP/CTX/TG and memory accounting.

The first probe is transport-only. It does not qualify floating-point math,
trace replay, the bulk reader, hardware, coding quality or performance. Neither
the packer nor reader transform is connected to serving or the combined runtime.

## Current evidence

Full T16 MLP simulator run **35297164881** passed in **4m25s**, using
source `b5103f746befd67964893fc0f826ff142c0eefd4`. Downloaded evidence was
independently checked for the complete eager/replay matrix, weight comparisons,
stream integrity, staged projection identity and successful container cleanup.

| Check | Result |
|---|---|
| Eager T16 output | Exact on both chips |
| Changed-input replay (0, 1, 0) | All 12 control/candidate checks exact |
| Stale-input negative controls | All four detected |
| Native paired BF4 weights | Zero mismatches, 43,520 pages per projection/chip |
| Packed stream after eager/capture/replay | Both raw-byte hashes unchanged |
| Extra stream memory | 50,319,360 bytes per chip/layer |

Numerical report SHA256:
`548805b4bc3b64c1441f59c6a7f887ec35df40308f9c10cdbaa57cd4aa62a138`.
This uses pinned DFlash2 layer-zero weights as geometry-matched operands and the
existing simulator packer graft. It is **not** target-model quality validation,
physical-hardware correctness or a throughput measurement. Next is guarded
integration into the combined candidate, including memory admission and exact
runtime/source binding before paired hardware requests.

Run **35296716954** passed: **40 seconds for the job**, including the 15-second
storage gate; the transport simulator step took **18 seconds**. All eight checks
passed (two geometries, two patterns, both chips), with exact raw words,
zero padding and unchanged source buffers. The independent admission check
passed both in CI and against the downloaded artifact locally.

Report SHA256:
`1febb741244d258ba603694dabf0168ec0a61fa010d8b09e2611a0839c93007d`.
Source commit: `578cc1e7293a71b451ad8c1a09d2ea8491a70df5`.
This first run qualifies the packer transport only. The subsequent replay run
above covers the generated bulk reader and arithmetic in simulation; combined
performance remains unqualified.

Run **35295292083** stopped at Docker preflight creation (exit 124 after
30 seconds), before simulator startup. Retained host telemetry recorded
65.08% full I/O pressure (`avg10`), about 95 GiB available RAM, and two active
BuildKit containers. This is storage contention, not an observed kernel failure.
The next launch checks one bounded 15-second I/O observation before Docker
creation; timeouts and the existing 1% admission threshold are unchanged.

Guarded retry **35295976475** measured **13.61%** full I/O stall over 15 seconds
and correctly stopped before Docker creation. No simulator kernels ran in
either attempt. The user subsequently released a clear CI window for the
successful third attempt above; the admission threshold was not relaxed.

The simulator-only projection adapter is prepared and host-tested. It changes
the weight accessor and reader source, retaining the constructor, fused compute,
input distribution and output publication. It rejects hardware, aliased buffers,
partial streams and changed bindings/arithmetic. Caller ownership of the extra
stream remains explicit. Full MLP simulation passed as recorded above;
physical-hardware qualification is still pending.
Its manifest identifies the generated bulk reader rather than incorrectly
retaining the original reader's hash. Host tests also preserve the register
epilogue's constructor/compute source and reject changed reader metadata.
