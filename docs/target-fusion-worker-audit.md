# Target replay: fusion and worker audit

Source review against the combined T16 path, 2026-09-18. These are scheduling
constraints and candidate boundaries, not newly measured kernel timings.

## What is already present

| Component | Current construction | Consequence |
|---|---|---|
| Gate/up MLP | 272 output tile-pairs, three per worker, 91 compute workers | Not a 36-worker kernel; a simple larger grid is not a new fix |
| MLP down | 160 output tiles, two per productive worker, 11x8 rectangle | 80 workers produce output; rectangle area is not compute occupancy |
| GDN convolution | Direct-window reader, deferred packed checkpoints | The old standalone-window timing is not a current removable cost |
| Verifier scheduling | Captured multi-operation replay | Capture removes host submission work, not inter-kernel dependencies |

Sources: `scripts/ci/fused_1d.py` (`mapping`, `FusedProjection`),
`mlp_down_grid.py` (`widen`), `fused_t16_scope.py` (`forward`),
`gdn_direct_window_scope.py` (`scoped_direct_windows`). The latest report
35315198247 attempt 2 confirms `wider_down=true` and all 64 layers covered.
The older isolated down-grid rejection must not be mistaken for the current
combined recipe's configuration, nor does inclusion prove its independent gain.

For the existing whole-pair gate/up partition, 272/110 still rounds up to three
pairs on the busiest worker. Two pairs everywhere would require 136 workers.
This is a workload bound, not proof that worker placement cannot help NoC traffic.
Using spare cores effectively needs a different partition or useful overlap,
not just changing the grid rectangle.

## Remaining structural boundaries

The current MLP path is:

`activation -> L1 copy if needed -> fused gate/up -> L1 intermediate -> down -> collective -> DRAM result`

Gate/up writes its output as an interleaved L1 tensor; down consumes it through
a separate native matmul. The existing register epilogue has already removed
one internal rounding handoff, but has not fused the two matrix multiplications.
The small activation intermediate is not automatically the dominant cost: down
depends on all intermediate columns, so a direct producer/consumer design needs
cross-core delivery and an explicit reduction schedule. It must preserve BF16
rounding points, BF4/BF8 weights, immutable inputs and changed-input replay.

| Candidate family | Prerequisite before implementation | Acceptance |
|---|---|---|
| Gate/up-to-down producer/consumer fusion | Review rejected streamed MLP design; demonstrate a different delivery/compute schedule | Bounded simulator, then identical combined request |
| Split-K or mixed producer/consumer workers | Account for partial-sum storage, reduction traffic and rounding | Exact target output/state gate; no precision relaxation |
| Collective/residual/norm fusion | Inspect the pinned native consumer and actual layout contract | Include conversion and communication costs in the combined cycle |

Do not reuse old per-kernel milliseconds as current-source attribution. Current
measured verifier/readback is 60.17 ms; a complete 200-TG cycle allows only 55 ms
at 11 committed tokens/block. The target needs structural savings as well as
drafter/publication improvements. No row in this table is a qualified speedup.
