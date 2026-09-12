# Combined-runtime trace attribution

Status: completed attribution run 34698584939, revision `6f83b7c`. This is not a TG benchmark.

The most recent GDN changes preserve exactness but do not reduce blocking trace
time. Before another fusion, measure the actual device operations in the current
combined runtime rather than reuse an older pre-fusion profile.

| Preserved feature | Configuration |
|---|---|
| Target | Qwen3.8-27B, two P150A, four fabric links |
| Workload | CTX 4096, one coding stream; output capped at 64 tokens for attribution |
| Drafter | Native DSpark attention, captured proposals, score layout |
| Verifier | T16 folded attention, commit-only GDN, fused T16 MLP |
| Publication | Captured feature projection; unused singleton position uploads skipped |
| Experimental GDN changes | Disabled; native recurrence reference |

Only one audited request runs. The existing request observer brackets real
verification calls and records their trace IDs; device profiler rows must match
those replay IDs on both chips. Feature, output and state audits remain enabled.
The report explicitly marks instrumented/correctness-only and leaves PP/TG null.
This is neither a new throughput measurement nor completed-function acceptance.

The prior profile flag rejects fusion and captured publication. A separate
`combined_profile` option retains those safeguards and admits only the complete,
audited combined configuration. The report validator additionally checks its
publication/fusion route. Serving defaults remain unchanged.

## Measured device costs

Steady T16 trace envelope: **68.722 ms on chip 0 / 68.719 ms on chip 1**.
Kernel interval union: **67.394 / 67.379 ms**. Uncovered intervals: **1.329 /
1.341 ms**. This rules out uncovered launch intervals as the dominant cost in
this trace; it does not measure every cause of stalls inside kernels.

| Chip 0 operation/core group | Calls per replay | Summed kernel ms |
|---|---:|---:|
| Generic, 96 cores (GDN recurrence mapping) | 48 | 11.747 |
| Generic, 99 cores (fused MLP mapping) | 64 | 11.388 |
| Matmul, 32 cores (down/output projections) | 128 | 10.791 |
| Matmul, 43 cores (GDN input projection) | 48 | 5.740 |
| Generic, 48 cores | 112 | 4.608 |
| Generic, 24 cores (GDN norm mapping) | 48 | 4.494 |
| Decode SDPA, 110 cores | 32 | 3.509 |

Parenthesized mappings follow the current source geometry and call counts;
the profiler labels these operations by family/core count, not generated source
identity. Group sums may overlap and must not be added as a critical-path total.
The 99-core/64-call group is consistent with the installed fused MLP path, rather
than the older 39-core native projection profile.

The historical 36-core figure is not the active-core count for this whole
runtime: different kernels use different grids, including 96, 99 and 110 cores.
Simply adding cores or removing host gaps cannot plausibly double this trace's
speed. At 11 committed tokens/block, 200 TG requires a whole block cycle of
55 ms, already less than this verifier trace alone, before drafting/publication.
That acceptance is workload-dependent, not a general guarantee.

Next: reduce arithmetic/projection work in the large groups. The first outer-add
fusion reinitializes multiply/add modes for every tile; a multi-tile register
version can test whether mode-switch cost erased the saved buffer pass. Any
candidate still needs exact simulator admission and complete-request timing.

Evidence validation checks 786 script hashes, exact request/state/feature audits,
trace markers, and request/device-report hashes. PP and TG remain null.
Attribution SHA256:
`47d3d488294042674f2778ebe93e4cddc77d66c14da88ed1fea4b71b00261da0`.
Request SHA256:
`dec755b4254cf0909eb2c12121be580c4d1d1985e224a6377ccd97b1a8e2854c`.
