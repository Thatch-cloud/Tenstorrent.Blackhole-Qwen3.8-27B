# M1: matmul program grids for the 13x10 compute grid (tp4/m1-grid13)

The four cards expose a 13x10 compute grid (130 worker cores) since the 2026-10-10 firmware unlock; it was 11x10. This package asks which matmul, and matmul-adjacent, program
grids of the ship stack (`c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic`, base `tp4/fusion-1` 90738c8b) were sized for 11 columns (or fixed at 8) and what 13x10 allows,
and ships the exact ones as ONE default-off lever, `QWEN_FAST_GRID13` (`scripts/ci/grid13_tp.py`), with an `_AUDIT` twin.

Labels: **M** measured in a card profile (v676 on 11x10, v678 on 13x10 decode, v701 solo prefill; run 38034191271 for v678), **cm** computed from measured numbers, **E** estimate (nothing
here has run on a card), **src** read in code.

## What the profiles say

* **Decode (8 live, ~125 ms a round on the ship stack).** The target's matmuls already take their grid width from the device (`decode_grid_w`), so they moved -4 % to +3 % between
  v676 and v678 (M). The big 13x10 effects were two: the drafter's matmuls on FIXED 8-wide grids lost 17-36 % (M), and the quad's K/V-assembly untilizes on 128 cores paid a 28 us dispatch gap
  each (M). The ship stack already carries the answers to both: `QWEN_FAST_DRAFT_MM_GRID` (the MLP branch's gate/up/down and the attention branch's wo on device-wide grids) and
  `QWEN_FAST_DRAFT_PERMUTE` (the assembly untilizes are gone). `QWEN_FAST_MLP_CFG=g3u4d3` was swept ON 13x10 (card M, run 38043752440; widths 8, 10, 11, 12, 13; the 12-wide down is 15 % slower than
  the 13-wide one), so nothing of it is sized for 11. What is left on grids is small: see the table. **Decode has about 1 ms a round of grid headroom (E; 2 with the GDN commit re-plan), not 10.**
* **Solo prefill (v701).** The three fused all-gather matmuls (gdn_in, attn_in, the gate|up SwiGLU: 27.3 + 9.5 + 48.9 = 85.7 ms of the 266 ms non-SDPA kernel time of every 2,048-row chunk, M) are the
  one place where the 8-wide grid is a design constant of the pinned `tp_common` helper (`grid = (8, grid[1])`, 8x9 cores, 76 with the mux cores): the biggest item of the package.

## Ranked table

Per-call times are the v678 (13x10) medians; "per round" is an 8-live round (two verify passes of 64 rows); "per chunk" is one 2,048-row prefill chunk (64 of them at 131,072 tokens).
Savings are **estimates**. Exactness: "same K order" means the K reduction of every output tile is the served one (same `in0_block_w` / `K_block_size`, fidelity, fp32 destination, gather order), so
only WHICH core owns a tile moves; none of these changes a reduction order, so none is flagged as needing a different proof than the audit.

| # | Op | Phase | Now (grid, per call, calls) | Proposed on 13x10 | Estimated saving | Exactness |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | gate\|up AGMM + SwiGLU | prefill | 8x9+4 cores, M block 8, 763.8 us, 64 a chunk = 48.9 ms (M) | 12x9+6, M block 6: 8 M blocks of 8 rows on 8 columns (8 rows a core) become 11 blocks of 6 on 12 columns (6 rows) | 3 to 10 ms a chunk (E; 25 % fewer output rows on the busiest core, less the smaller block's reuse) | same K order; the gather is a copy; audit |
| 2 | gdn_in AGMM | prefill | 8x9+4, M block 4, 568.3 us, 48 a chunk = 27.3 ms (M) | 12x9+6, M block 6 (6 rows a core, was 8) | 2 to 7 ms a chunk (E) | same |
| 3 | attn_in AGMM | prefill | 8x9+4, M block 4, 592.1 us, 16 a chunk = 9.5 ms (M) | same plan | 0.7 to 2.4 ms a chunk (E) | same |
| 4 | output projections (attn wo, GDN out), M=64 | decode verify | 13x3, 32 cores, per_core_N 5, 27.9-28.2 us, 64 a pass = 1.8 ms (M) | 13x5, 54 cores, per_core_N 3 (the shape the MLP down's sweep found 9.4 % faster at the same N) | 0 to 0.35 ms a round (E; by analogy, K is 3x smaller so less) | T1 #11 rule: in0_block_w, per_core_M, fuse_batch, mcast_in0 unchanged (checked in code) |
| 5 | fused commit feature projection (`dflash_device.project_features`) | decode commit | (8,10), 80 cores, 92.8 -> 124.7 us, 8 a round (M) | (13,7), the same 80 cores, per_core_N 2 | ~0.25 ms a round (E: recovers the measured +32 us each) | grid field only (as `QWEN_FAST_DRAFT_MM_GRID`, audited exact in the combined attach); needs the unapplied hook patch |
| 6 | GDN / attn in-projections, M=64 | decode verify | 13x4 (43 cores) 63.2 us and 13x3 (38) 52.6 us, 64 a pass = 3.9 ms (M) | plateau: not changed by default (`gdn_in=` / `attn_in=` name a partition) | 0 to 0.1 ms a pass (E) | T1 #11 rule |
| 7 | 2D prefill matmuls (mlp_down 245.8 us, gdn_out 313.1, attn_out 161.9) | prefill | 10x10 (100 cores), 33.3 ms a chunk (M) | a transposed 13x10 config (M tiles on the 13 columns: 5 rows a core, was 7) | ~4 ms a chunk, mlp_down only (E; gdn_out is bound by its 42 MB fp32 output) | same K order; `transpose_mcast` with interleaved inputs is unproven: NOT built |
| 8 | GDN commit one-op trace | decode commit | 96 cores, 508 us, 8 a round = 4.1 ms (M) | re-plan on 120-130 cores (kernel plan, WP10) | 0.5 to 1.0 ms a round of DEVICE time (E), exposure unknown | data movement; NOT built |
| 9 | all other ops | both | see below | no grid-limited gain | 0 | |

What the table leaves out, and why (all from the same profiles):

* **Already taken by the ship stack:** MLP down/gate/up (`g3u4d3`, -8.3 us a layer, M), the drafter MLP-branch and wo matmuls (`DRAFT_MM_GRID`), the K/V-assembly untilizes (`DRAFT_PERMUTE`), the argmax (S1),
  the fused gate|up (`GATEUP1`, 68 cores: 272 output tiles split into whole per-core columns only as 4 x 68, a 13x10 variant has none).
* **Not grid-limited:** multi-user SDPA (106 us, 16 cores an entry at 4 users, the tree is fixed), the drafter SDPA (175 us; its work units are heads x query chunks, far fewer than the 64 cores of its grid, so more cores add nothing, and a K split is not exact), the
  drafter's D2a conv (75 us, unchanged by the grid in v678: its work unit is a page, and the page count, not the core count, sets the per-core work), K5-A (48 cores, chain-bound), LayerNorm (32/40 cores, reduction order), AttnPrep (128 cores, already moved), the LM head
  (130 cores, 92 % of the measured DRAM rate), the collectives (12 cores, fabric), the prefill norms (64 = one core per tile row), GDN scan (12 heads x 4 column tiles = 48), the prefill SDPA (separate work).
* **Not exact if changed:** the drafter's 2-core hidden norms (81 us, 11 a quad; widening them is the tau-only D2b), SDPA cores per entry, LayerNorm cores, K5 core counts.
* **Not a grid change:** the drafter's q/k/v/conv projections (52, 46, 47, 57 us on 8-wide grids, per_core_N 1) look bound by the 40-block `in0_block_w` = 4 multicast handshake, not by cores
  (E: each core streams its few weight columns at a rate far under what its NoC reads can do); `in0_block_w` 8 would halve the handshakes but is the T1 #11 rule's one field, so it is a card sweep with its own proof, not part of this lever.

Total, as configured by `QWEN_FAST_GRID13=1` with the drafter hook applied: **prefill 6 to 20 ms a chunk (central 14), i.e. 0.4 to 1.5 s (central 1.0 s) of the 36.8 s TTFT at 131,072 tokens (E); decode 0 to 0.6 ms
a round (E), inside the timing floor.**

## The lever

`QWEN_FAST_GRID13` is `1` (every sub-switch) or a comma list of `prefill`, `verify`, `drafter`; `QWEN_FAST_GRID13_AUDIT=1` needs it; `QWEN_FAST_GRID13_CFG` (optional) names a sweep point
(`agmm=<cols>[x<M block>]`, `out=<n>`, `gdn_in=<n>`, `attn_in=<n>`, `width=<n>`). All strict; any of them at the pair raises. Installed once at startup by `tp_addresses.install`
(`fusion-wp/M1-tp-addresses.patch`; after the model is loaded and before the fast path attaches, which is enough because every wrapper acts at the call); with the flag unset nothing is imported or wrapped.

* **prefill.** `tp_common.all_gather_matmul_prefill` and `all_gather_swiglu_prefill` are wrapped, not copied: the pinned function runs as it is, with the `ttnn` global it sees viewed through a shim that
  rewrites two keywords of its one `all_gather_minimal_matmul_async` call (`config`, `num_workers_per_link`) after a census of the rest (2 links, transposed, served grid 8 wide, subblock_h 1, workers x links = width).
  The plan is `agmm_plan`: 12 columns (the widest even width of a 13-wide device; 2 links x 6 workers), rows as served (the mux cores keep row 9), the M block that minimises the busiest core's rows
  (`pick_m_block`: 6 for 64 tiles), and only if that is strictly fewer rows than the served call. K block, N block, subblocks, fidelity, fp32 destination are the served ones.
* **verify.** `tp_common.matmul_1d_decode` is wrapped: the model built its 64-row program configs at ModelArgs, before the lever can be installed (`serving_startup.start` installs after the model is
  loaded), so the swap is at the call, once per served config object (the warm pass and the captured launch of a layer get the same lever config, so no program is new inside a capture). The pinned
  builder makes the lever config (the same field set as the served one), the four fields the T1 #11 rule keeps are checked equal and anything else refuses. Under the audit flag the served config
  runs again on the same operands (eager only).
* **drafter.** `commit_program` returns the commit projection's program with only `compute_with_storage_grid_size` replaced; the hook is `fusion-wp/M1-dflash-device.patch` (unapplied: `dflash_device.py`
  is not this package's).
* **Grid.** Read from the device (`compute_with_storage_grid_size`, or the model's `decode_grid_w` for the builder). A device narrower than 13x10, or one whose grid cannot be read, keeps the served ops and logs
  `[PINDIAG] tp4 grid13 fell back reason=...`.
* **Markers.** `[PINDIAG] tp4 grid13 engaged site=<site> ...`, `[PINDIAG] tp4 grid13 unchanged reason=... site=<site>` (information: a chunk of under 512 rows, or a plan with nothing to gain; the served ops ran and the
  smoke does not fail on it), `[PINDIAG] tp4 grid13 audit n=<n> exact=True site=<site> ...`, and on a difference `... audit n=<n> exact=mismatch ...` then
  `[PINDIAG] tp4 grid13 audit mismatch ...` and an AssertionError. Sites: `gdn_in`, `attn_in`, `gate_up` (or `mlp_gate_or_up`), `verify.out`, `drafter.commit`. Read by `grid13_smoke.py`.
* **Capture rules.** The audit runs only in eager passes: inside a trace capture it is skipped (one logged line, nothing synchronized or read back), and the first four calls of each (site, rows) are audited.
  The verify wrapper hands the SAME program config to the warm pass and the captured launch (`test_grid13_tp` holds it through `capture_rules`), so no program is new inside a capture.

## Exactness

The argument is the one of `draft_mmgrid_tp` and the MLP sweep: a 1D- or 2D-multicast matmul and the minimal matmul give each output tile to one core, which accumulates K block by block in a fixed order into an
fp32 destination, spilling the partial sums at the same K-block boundaries (whatever format the spill has, the boundaries are the served ones because `K_block_size` / `in0_block_w` do not change), whatever the grid, the M and N partition and the output subblock are. The WP4 card-M sweep measured exactly such configs (the builder's own arithmetic, transcribed, on the minimal rectangle as wide as the device) for the MLP's gate, up and down: ~110 partitions, 0 inexact. The all-gather is a copy; the open question for the AGMM is only whether the ring's arrival order of K blocks depends on the workers per link
(6 instead of 4): the audit is what answers it, and nothing is exact until it has. The CPU tests hold the plans, the call shape, the fall-backs, the audit mechanism (with a negative control) and the capture
rules; they cannot hold the device.

## Cards

`scripts/ci/references/fusion-jobs/M1/` (ORDER.txt): the audited attach (warm-up, solo prefill ladder, eight live, telemetry), the decode ABAB at eight live, the solo prefill ladder ABAB. The profiles are
`...-fx-all-grid13` and `...-fx-all-grid13-audit` (`fusion-wp/M1.json`). If the first prefill refuses or hangs the 12-wide plan, the sweep names are in ORDER.txt.
