# Drafter fusion levers (tp4/fx-wp6, work package 6 of the op-fusion programme)

Four default-off levers on the four-card drafter, all **bit-identical by construction** and each with an `_AUDIT` twin that compares it with the served composition on the
device. Nothing here changes a served token (the target verifies every proposal) and nothing here is tau-only, so there is no tau A/B: an audit that disagrees makes the
lever NO-GO, not a tau question.

| Flag | Item | What it does | Files |
| --- | --- | --- | --- |
| `QWEN_FAST_DRAFT_REDUCE` | F-F1 DR | after the unchanged dim-0 all-gather, the four slices and three fp32 adds `((p0 + p1) + p2) + p3` are one launch | `draft_reduce_tp.py`, `draft_reduce_tp_io.cpp`, `draft_reduce_tp_compute.cpp`, `draft_fuse_out.cpp`; dispatcher in `feature_collective_tp.py`, hook in `draft_mlp_branch.py` |
| `QWEN_FAST_DRAFT_TAIL` | F-F3a | the MLP branch's SwiGLU (nine launches) and its residual tail (four) are one launch each | `draft_tail_tp.py`, `draft_tail_tp_io.cpp`, `draft_tail_tp_swiglu_compute.cpp`, `draft_tail_tp_residual_compute.cpp`, `draft_fuse_out.cpp` |
| `QWEN_FAST_DRAFT_GATEUP1` | F-F3c (gate and up) | gate and up as one matmul over a load-time `[gate | up]` weight; the tail kernel reads both halves in place | `draft_gateup_tp.py`, the load and execute hooks in `draft_mlp_branch.py` |
| `QWEN_FAST_DRAFT_MM_GRID` | R2 (added from v678) | the MLP branch's 1D matmuls run on a program grid as wide as the device's, not the fixed 8-wide rectangle | `draft_mmgrid_tp.py`, the `project` hook in `draft_mlp_branch.py` |

Shared: `draft_fusion_tp.py` (strict flags, refused at the pair, markers, core plans from the grid the device reports, the eager audit), `draft_wp6_smoke.py` (the offline
judge of the markers), `wp6_fake_device.py` (test support: a functional fake of the ttnn surface with the kernels transliterated). The manifest is
`scripts/ci/fusion-wp/WP6.json`.

## What was measured (v676, 11x10) and what is estimated

Per quad, from `quadsites.txt` (M676) and the 13x10 re-run (M678, plan section 8.3):

| Site | Measured | The lever |
| --- | --- | --- |
| gather-add glue: 4 slices (7 us), 3 adds (10.4 us), per chain, 10 chains | 0.324 ms (+0.324 for the typecast and the rest of the chain line) | REDUCE: the 7 launches of slices and adds become 1 (kernel time **estimated** at 12-15 us: 320 tiles over 110 cores, 16 KB read a tile) |
| `swiglu_device`: 7 typecasts, silu, multiply, per layer | 0.270 ms | TAIL: 1 launch (**estimated** 10 us) |
| residual tail: 2 typecasts, add, typecast, per branch and layer | 0.137 ms per branch | TAIL: 1 launch (**estimated** 10 us) |
| gate + up: 75.6 + 76.1 us on 68 cores per layer | 0.753 ms (0.881 on 13x10) | GATEUP1: 1 launch, same 68-core shape at twice the per-core columns (**estimated** -20 to -30 us a layer: the cores are not more numerous, so the saving is the launch and the ramp, not bandwidth) |
| gate/up, down, wo on the fixed (8, 10) grid | +17 %, +36 %, +21 % on 13x10 (v678: +0.127, +0.120, +0.026 ms per quad) | MM_GRID: the same programs on (13, k) grids (**estimated** to recover most of it: +0.247 ms per quad for the MLP branch's own, the wo and the fused commit's 80-core projection are other packages' files) |

Estimated saving per 8-live round (two quads), all four levers, with the MLP-branch halves wired as in this branch: REDUCE -0.44, TAIL -0.61 (-0.44 SwiGLU, -0.17 residual),
GATEUP1 -0.30, MM_GRID up to -0.49: **about -1.8 ms**, and about -2.5 ms once the attention branch's chain and residual are hooked too (WP7's three one-line hooks, below). The base costs
are measured; every kernel time is an estimate until a card-M timing says. The plan's WP6 range was -1.2 to -2.4 ms (F-F1 -1.0, F-F3a -0.8, gate|up -0.3).

What 13x10 changes: the launches take min(tiles, grid, 110) cores, so a wider grid does not help them past 110 (the reduce chain has 320 tiles at 64 rows, the residual 320, the SwiGLU 272);
the fused gate|up keeps its 68-core (8, 10) program because 272 output tiles split into whole per-core columns only as 4 x 68 (2 x 136 needs 136 cores, which neither grid has);
MM_GRID is the one lever that exists because of 13x10.

## Exactness

* **REDUCE**: the served chain adds fp32 tiles with binary_ng's SFPU add in chip order, writing each sum to DRAM (fp32, lossless). The kernel does the same adds with `add_binary_tile` on
  tiles read in the same order, keeping the running sum in the destination register. The slices are tile-aligned copies. The all-gather (its order is exactness-locked) is the served call.
* **TAIL**: the served SwiGLU passes every value through DRAM tensors; the steps that change a value are the four fp32 -> bf16 roundings, the silu and the multiply, and the kernel runs those
  same primitives (`typecast_tile<Float32, Float16_b>`, `silu_tile`, `mul_binary_tile`) in the same order. The bf16 -> fp32 widenings and the DRAM round trips are lossless. The same pattern is
  `draft_convolution_fused_compute.cpp`'s, which the card audit holds byte-equal to its composed ops. The silu is the one primitive a CPU cannot reproduce bit for bit against the SFPU: the card audit holds it.
* **GATEUP1**: output column j of a matmul is its own K reduction (the same `in0_block_w`-blocked accumulation into fp32 whatever the other columns of the same core are); `per_core_N` moves
  only which core owns a column. bfloat8_b quantises inside tiles, and the fused matrix concatenates whole tile columns.
* **MM_GRID**: only `compute_with_storage_grid_size` changes; the cores, per-core columns, `in0_block_w`, subblocks and `per_core_M` are the served ones. In a 1D multicast program each core
  accumulates K block by block in the same order whichever core multicasts which in0 block.

Proofs: the CPU suite (below) holds each by transliteration of the kernels against the served composition on a functional fake, with negative controls (a different add order, a missing
rounding point, a swapped gate and up, a reversed K order each DIFFER). The card half is the audit flag of each lever (eager, on the warm pass of each drafter bucket: it reads back, so it can
only run outside a capture; the first `AUDIT_CALLS` = 10 calls of each site and row count, i.e. every layer's chains, on real activations and weights), then the draft-singles audit and the
position-keyed accepted-prefix compare.

## The hooks the attention branch needs (WP7's file, `draft_attention_branch.py`)

Each is lazy and flag-guarded exactly as in `draft_mlp_branch.py`, so a flags-off process never imports the new modules:

1. the gather at the end of the branch: `gather = draft_reduce_tp.choose(gather, quad, 'attention')` under `QWEN_FAST_DRAFT_REDUCE=1` (the quad's 64-row chains; a branch without a quad already reaches the
   dispatcher in `feature_collective_tp.gather_add_projection`);
2. the residual tail: `output = draft_tail_tp.residual(operations, mesh, finished, hidden, retain, site='attention')` in place of its two typecasts, add and typecast, under `QWEN_FAST_DRAFT_TAIL=1`;
3. the wo projection's grid: `grid = draft_mmgrid_tp.grid_for(operations, mesh, value, weight, (8, 10), columns, rows, kernel, site='wo')` under `QWEN_FAST_DRAFT_MM_GRID=1`.

Until they land the engaged line's `chains=<n>` reaches 5 per quad, not 10: `draft_wp6_smoke.py --expect-chains` says so. The fused commit's 80-core projection (`fused_commit_tp.py`, WP10) has the same fixed-grid
problem (92.8 -> 124.7 us in v678; about -0.26 ms a round): `draft_mmgrid_tp.wide_grid()` and `program_config()` are the one-line fix.

**F-F4 (one 64-row drafter LM head instead of two 32-row halves) is not in this package**: the halves are `quad_draft_tp.head_candidates` over `draft_shared_head_tp.py`, both WP7's files.

## Card plan

The pack generated from the manifest (`make_fusion_jobs.py`): per lever an audited attach (`...-fx-<id>-audit`) and a timed control / lever ABAB at eight live. This package's own extra templates are in
`scripts/ci/references/fusion-jobs/WP6/` (see its ORDER.txt): the draft-singles audit (`...-fx-wp6-singles` against its control, the audited lean twin), the combined timed ABAB (`...-fx-wp6`) and the per-op
profile of the combined arm (`ops-twin,ops-trace`), from which each lever's gain is read site by site (the reduce chains, the SwiGLU row, the gate/up row, the down row, the residual) against the v678 control.
Judge every pair paired per round (`w2ln_timing_compare.py pair`), never by unpaired medians; the texts and the position-keyed accepted prefixes must be equal.

```
python3 -B -m unittest test_draft_reduce_tp test_draft_tail_tp test_draft_gateup_tp test_draft_mmgrid_tp test_draft_wp6_branch test_draft_wp6_smoke     (from scripts/ci, python 3.11)
```
