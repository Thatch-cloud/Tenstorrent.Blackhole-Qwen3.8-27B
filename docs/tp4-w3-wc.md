# tp4/w3 WC: the GDN layer's windows, convolution and gates in one launch

Status: **built on `tp4/w3-wc`, behind `QWEN_FAST_TP4_GDN_WC` (off by default), CPU-tested, never compiled or run on a card.** The kernels have never been through a compiler: the attempt to compile them in a throwaway toolchain container on the rig was refused by the session's permission classifier and not retried, so the first compile is the card-M byte gate. Their API surface is the production kernels' (a test holds every `noc_`/`cb_`/`get_` call to one a production kernel already makes); the one exception is `invalidate_l1_cache()`, which only the non-default branch of the T2 windows kernel calls (its settings matrix ran on a card). The estimate is the fusion
audit's (ESTIMATED, not measured): WC replaces G3-G8 of a GDN layer (the V2 split, the T2 windows, the canon block, the window stack, the conv-gates
launch, the unstack: 139.3 us a layer of kernels and gaps, 6.69 ms a verify) with one launch of 25-40 us, which the audit's review haircuts to
**-4.5 to -5.5 ms a verify, about -9 to -11 ms a two-block round** (the audit's own figure with F1 inside is -6 to -8.5 ms a round at 8 live, after
overlaps). F1 is inside WC, not additional.

Labels: (src) read from the code, (cm) computed, (e) estimate, (U) unknown until a card says.

## Flags

| Flag | Effect |
|---|---|
| `QWEN_FAST_TP4_GDN_WC=1` | One `generic_op` per GDN layer and packed block in place of the split/T2/canon/stack/conv-gates/unstack chain. Needs `QWEN_FAST_TP=4`, `QWEN_FAST_TP4_GDN_GLUE=1`, `QWEN_FAST_TP4_GDN_BLOCK_CONV=1` and `QWEN_FAST_VERIFY_T2=1` with the windows cut; strict 0/1, anything else raises. `QWEN_FAST_TP4_CONV_GATES_SPREAD=1` may be set beside it: F1 stays the launch for a layer WC declines (but `QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT=1` may not: a layer WC serves never calls F1's launch, so F1's audit would fail it). `QWEN_FAST_TP4_VGLUE_AUDIT=1` with the lever needs the WC audit flag (a layer WC serves has no V1 block entries). The profile env is the W2 profile's (glue, block conv, F1) plus the flag; `QWEN_FAST_VERIFY_T2=1` comes from the serving image's own environment. |
| `QWEN_FAST_TP4_GDN_WC_AUDIT=1` | Needs the lever and `QWEN_FAST_TP4_VGLUE_AUDIT=1`. WC's per-user conv, beta, g, z and all sixteen advanced windows are held against the SERVED per-user path in the replay (the V1 audit's `hold_served`), every user, every chip, as int16 bits. |

With the flag unset `gdn_wc` is not imported and the code path is the V1 path byte for byte (`test_tp4_w3_wc.FlagOffIdentityTests`).

## What the launch does (per chip, 110 cores at 64 rows)

| Cores | Count (4 users) | Reader (`gdn_wc_reader.cpp`, RISCV_1) | Compute (served source) | Writer (`gdn_wc_writer.cpp`, RISCV_0) |
|---|---:|---|---|---|
| conv | 80, two instances each (the served partition) | per instance: the projection tile of the block into the conv CB, its second half canonicalised; the history spans; st1..st3 built in place; the taps | `gdn_conv_gates.cpp`, unchanged | conv tile and the four shift tiles split into the two users' halves, ten whole pages written |
| gate | 2, one gate tile each (F1) | a and b gathered by words from the projection tile, second half canonicalised; dt_bias, neg_exp_A | unchanged | beta and g halves, four pages |
| z | 28 (3 or 4 z tiles each) | the z tile of the block with its second half canonical | none (`{0, 0}`) | two pages a tile |

### The bytes (the contract the emulation and the audit hold)

A block tile (tile row `bt`, tile column `c`) holds user `2bt` in faces 0-1 (rows 0-15) and user `2bt+1` in faces 2-3 (rows 16-31). The V1 chain's
numbers, which WC reproduces:

* **x** (the conv CB's slot 3) = the block tile with faces 2-3 canonicalised (the canon block's tile): a bf16 with a zero exponent (-0 and every
  denormal) becomes +0 (`CANON_DENORM`, on, the define `gdn_rows_dma_tp` launches with). Users 0 and 2 stay raw.
* **st_s, s = 1..3** (slots 0..2 of the conv CB) = per face `F` (user `2bt + F/2`, face column `F%2`): rows `r < 4 - s` are row 0 of that face of history `r + s`
  of that user (raw, never canonicalised), rows `4 - s .. 15` are rows `0 .. 11 + s` of x's same face, in one block of `(12 + s) * 32` bytes
  (`gdn_conv_windows_packed.window_copies`). Slot 0 is not built: the conv never reads it, and history 0 is read by nothing in this launch.
* **gate tiles** a and b = the projection tile `bt*129 + 128` gathered at columns `0-11` (a) and `12-23` (b) of the page, rows 0-31, zero past the heads;
  rows 16-31 canonicalised (the gather reads the raw block; the rule is element-wise, so it commutes with the copy, and the zero padding is canonical).
* **outputs**: per user `2bt + h`, the page `c` of `conv` (and of each of the four windows `s`, which are the compute's pass-through of
  `[st1, st2, st3, x]` as the new `[st0, st1, st2, st3]`) is half `h` of the compute's tile in the page's rows 0-15 and zeros in rows 16-31; the gate tiles
  likewise; `z` is half `h` of the canonical block tile `(bt, 80 + c)`.

Everything but the canonical rule is a byte copy; nothing is computed outside the served compute source.

### Core plan, work table, arguments

`gdn_wc.plan(grid_x, grid_y, users)`: F1's plan (conv instances over the served partition, one gate tile per core after the conv cores) plus the z tiles
over the cores that are left, in core order (core `c` is `(c // grid_y, c % grid_y)`). Only 2 or 4 users (an odd count leaves half a tile to
zero; the V1 chain handles it). A grid that cannot hold the gate tiles beside the conv cores, or has no core left for z, is a fall-back.

* per-core runtime words, reader and writer (8, every core): `[inst_start, n_inst, g_n, gate_start, z_start, n_z, 0, 0]`; compute `[n_inst, g_n]`;
* common runtime words (addresses; their count is what generic_op hashes): reader 19 `[x, H[u][1..3] x 4 users, tap0..3, dt_bias, neg_exp_A]`, writer 32
  `[conv x4, beta x4, g x4, z x4, window 4u+s x16]`;
* compile words: reader `[K, Ct, Nvt, users, Wt, ACOL, BCOL, NV, ZFIRST, ZCOLS]` + accessors (projection, history, small) + `RT_WORDS`; writer
  `[K, Ct, Nvt, users, ZCOLS]` + accessor (output) + `RT_WORDS`; both kernels `static_assert` their highest runtime index below `RT_WORDS`;
* circular buffers: the served factory's 0-12 as F1 builds them, 13 the gate gather ring (4 pages), 14 the history spans (1), 15 the z tile queue (2), 16
  the output staging ring (10 user pages and a zero source page): 143,360 bytes a core (cm), far below the SDPA's 796,864.

### Why whole-page writes and local copies

Every DRAM write is a whole 2 KiB page composed in the staging ring (half data from the compute's output tile by a local L1-to-L1 NoC copy, the upper half
zeroed once at start by copying a zeroed source page); DRAM reads are whole pages except the history spans (two 64-byte reads of a page, the T2
`hist_row` setting card M proved). The memory note "sub-page accesses silently no-op on this stack" is respected for writes; the 64-byte reads and the
1 KiB local copies are the primitives the production T2 windows kernel and the V2 row mover already run.

## Binding

`gdn_device_loop_state_tp._recurrence_glued` (the V2 twin) asks `gdn_wc.enabled()` only when the flag is set (a literal check; nothing imported otherwise).
When WC is on the V2 split launch is skipped (the pieces exist only for the audits, `QWEN_FAST_TP4_VGLUE_AUDIT` and `QWEN_FAST_VERIFY_T2_AUDIT`, or when WC
declines) and `gdn_wc.stage(...)` runs after the state moves, from the conv states of `pending`, handing `run_user_batched_projected` a
`BlockStage` that carries the sixteen windows through its new `prebuilt` argument (the T2 window build and the V1 block stage are skipped; the T2
"windows" engaged count and the vglue "block conv" count are noted as before). `stage` answers `None` after ONE logged fall-back line for any call it cannot
take (users, shapes, dtype, layout, placement, mixed history placements, a non-uniform accessor class, a tree whose conv-gates sources are not the pinned
ones, a grid without room); the caller then makes the pieces and runs the V1 chain unchanged, so a declined layer costs nothing but the lines.

## Markers and the smoke rule (`w3_wc_smoke.py`)

* `[PINDIAG] tp4 gdn wc engaged rows=64 conv_cores=80 gate_cores=2 z_cores=28 per_core=2 served_per_core=2 audit=<0|1>` once per shape;
* `[PINDIAG] tp4 gdn wc fell back reason=...` once per reason (any such line fails a gated arm: the V1 chain ran and the timing is not the lever's);
* `[PINDIAG] tp4 gdn wc audit <n> exact=True layers=... entries=...` and `... audit mismatch ...`.

## What only cards can show

1. That the compiled kernels compile, hang-free, and move the bytes the emulation moves: the card-M byte gate `optimisation/ttnn-op/gdn_wc` (WC against the V1
   chain's served ops, random, edge and real-text-shaped inputs, all 16 advanced windows, traced replay, program-cache deltas, timing) and the audited arm.
2. The cost of the launch (U): the audit's 25-40 us (reader canonicalisation of 256 words a tile at about 12 ns a word is about 3 us a conv instance; two
   instances a core), against the 139 us it replaces.
3. That the canonical rule is what the served untilize/tilize round trip does on every pattern: the U2 probe (`optimisation/ttnn-op/canon_probe`) decides
   X-obs vs X-construct for V1, V2 and WC alike.
4. The hang matrix: WC removes about 24 transient tensors per layer from the capture, which changes the allocation pattern the unexplained audits-off hang is
   sensitive to; five hang-shape completions on the WC profile are part of the gate.

## Tests (CPU)

| Module | What it holds |
|---|---|
| `test_gdn_wc` (28) | the core plan (80 conv + 2 gate + 28 z cores at 64 rows, 80 + 1 + 29 at 32), every output page written once, the face-order emulation of the kernels' loops against the four SERVED per-user calls and, tile for tile, the V1 chain's unstack, on edge values (-0, denormals, infinities, NaN payloads), with negative controls (no canonical rule, a wrong history row) |
| `test_tp4_w3_wc` (77) | flag grammar and prerequisites, the kernels' text (argument indices, common-word layout, CB indices, the transcribed window formulas, no new API surface, LF), the launch on a fake ttnn (programs, words, buffers, every fall-back), the stage and audit labels, `run_user_batched_projected(prebuilt=)`, the V2 twin binding (split skipped, stage after the state moves, real pieces on a decline or under the audits) and the flag-off identity against the twin at its base commit, the replay audit, the smoke rule, the overlay lists |
| `optimisation/ttnn-op/gdn_wc` (30) | the card-M probe: host data, compare, sections, verdict, timing arithmetic, whole run on a fake ttnn, the run script |

## Card jobs WC needs (templates are the parent's pack; minutes ESTIMATED from this window's measured jobs)

1. **Card-M byte gate** (`optimisation/ttnn-op/gdn_wc/run_card_m.sh`, cardm step, one card, no quad job running): first JIT compile of the two kernels, then the compare matrix, replay, cache, control and timing. 45-90 min (container timeout 90 min). STOP on a compile error (exit 4) or any difference.
2. **Audited attach** on the WC audit twin (`QWEN_FAST_TP4_GDN_WC=1`, `_AUDIT=1`, the vglue audit): every audited layer's WC entries equal the served path's. Keep it to the short exactness list (audited 8x262k requests cost about 8.4 min each): about 25-50 min.
3. **Hang shapes** five consecutive completions with the audits off on the WC profile (the capture's allocation pattern changes): about 25-40 min each.
4. **Paired ABAB timing** at 4 and 8 live with CI paused: about 10 min a job (this window's timed jobs ran 9-10 min).
