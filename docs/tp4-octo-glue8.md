# tp4/octo-2 lever 2: the verify-glue levers at the octo block's eight-row users (QWEN_FAST_OCTO_GLUE8)

Status: built and tested on CPU; **nothing has run on a card**. Default off, gate only, strict `0`/`1`, byte-identical off (held by `test_octo_glue8_twin`, `test_octo_glue8_windows`). No tt-metal rebuild and no graft: every kernel is a
`generic_op` data-movement source shipped as `scripts/ci/*.cpp` and JIT-compiled at the first launch.

## 1. What the octo run showed

The O3 card run (`c2-packed-tp4-8x262k-ship-prefix-levern-octo`, `QWEN_FAST_OCTO=alternate`, 8 live) logged, for the two kinds of block in one boot:

| capture | verify t2 | vglue engaged | seq_block |
| --- | --- | --- | --- |
| M3 block (4 users x 16 rows) | `windows=48 windows_fallback=0` | `gdn_block_conv=48 gdn_glue=48 gdn_merge=48 attn_fold=16` | 48 of 48 GDN layers |
| octo block (8 users x 8 rows) | `windows=0 windows_fallback=48` | `gdn_split_eight_row_served=48 gdn_merge_eight_row_served=48 attn_fold=16` (no `gdn_block_conv`) | 0 of 48 |

and three `[PINDIAG] tp4 vglue octo 8-row served path site=split|merge|windows` lines. So at the octo block three of the lever family were absent: V2 (the split and the merge), the T2 packed windows, and (because V1 needs the packed windows)
V1, the block conv. V3a (the attention fold, `attn_fold=16`) already works at one eight-row group per bundle and V4a/C1a are per block, so they are not in this lever. K5-A (`seq_block`) is the one NOT made native here (section 5).

## 2. What is native under the flag

`QWEN_FAST_OCTO_GLUE8=1` (module `octo_glue8`) makes each of these sites run natively at the octo block **when its own lever is on**; a lever that is off stays off. `octo_glue8.sites_wanted(environ)` is that table.

| site | lever that turns it on | served path at 8 rows (today) | native |
| --- | --- | --- | --- |
| `split` | V2 `QWEN_FAST_TP4_GDN_GLUE` | per user a tile Slice (users 0, 4) or an untilize / slice / tilize round trip (1, 2, 3, 5, 6, 7), 8 pieces, in L1 | one `gdn_rows_dma8_tp` launch |
| `merge` | V2 | 8 untilizes, a concat, a tilize | one launch, block in interleaved DRAM like the served join |
| `block_conv` | V1 `QWEN_FAST_TP4_GDN_BLOCK_CONV` (needs V2 and the windows) | 8 `gdn_decode_conv_gates` launches at batch 8, 8 `z` slices | canon block, window stack, ONE conv gates at batch 64, unstack (`gdn_block_conv8_tp`) |
| `windows` | T2 `QWEN_FAST_VERIFY_T2` (cut `windows`) | 8 served window launches (72 us each, latency bound) | one `gdn_conv_windows_packed8` launch for the 32 windows |

Files: `octo_glue8.py` (flag, markers, smoke rule), `gdn_rows_dma8_tp.py` + `.cpp` (the mover), `gdn_block_conv8_tp.py`, `gdn_conv_windows_packed8.py` + `.cpp`. Edited existing files, each only on the eight-row geometry behind the flag:
`gdn_device_loop_state_tp.py` (the twin class: split, merge, which block stage), `gdn_user_batch_conv.py` (which windows builder), `packed_verifier.py` (the V1/V2 audit reads the octo block when the flag makes its sites native; before, `served_only`). The
sixteen-row modules (`gdn_rows_dma_tp`, `gdn_block_conv_tp`, `gdn_conv_windows_packed` and its kernel) are untouched: `test_octo_glue8_windows` proves the kernel twin differs from the pinned one in exactly the intended lines.

### The quarter-tile mover

A 32 x 32 bf16 tile is four 16 x 16 faces (face `f` at byte 512 f, row `r` of a face at 32 r). Rows 8q .. 8q + 7 are two 256-byte chunks, in faces 2 (q / 2) and 2 (q / 2) + 1, at offsets `1024 (q / 2) + 256 (q % 2)` and `+ 512`. One task composes ONE
destination tile from four quarter sources (raw, canonical or zero); the kernel reads each source chunk straight to its destination offset, zeroes or canonicalises in L1, and writes the tile. Runtime args carry the source and destination ADDRESS tables once and five packed words a task
(vs eight in the half-tile mover), so the longest launch, the advanced-window unstack of 8 users (3,600 tasks, 8 sources, 64 destinations), fits 110 cores at 33 of 36 slots. A launch that does not fit raises `Unsupported` before any state move and the served path runs under the glue8 FALLBACK
marker. Canonicalisation (zero exponent to +0, `CANON_DENORM`) is the half-tile mover's rule byte for byte.

## 3. Exactness, per site

Everything here changes how bytes move, never what the arithmetic is. The one new fact a card must confirm is the canonicalisation set.

* **Which users are canonical.** The served split cuts the user's rows with `ttnn.slice`. At 16 rows the V2 audit established (and card M measured the value rule): a slice that starts on a tile boundary (users 0 and 2) is a raw tile copy; one that starts inside a tile (users 1 and 3) goes through
  untilize / slice / tilize, which maps a zero-exponent bf16 (-0 and every denormal) to +0. At 8 rows the planner extends this by position: users 0 and 4 start on a boundary (RAW), users 1, 2, 3, 5, 6, 7 start inside a tile (CANONICAL), every merged output is canonical. This is
  an extension by analogy (the slice path does not look at the slice length). **It is exactly what `QWEN_FAST_TP4_VGLUE_AUDIT` compares in-trace** (`piece user k`, `merged output`, `block conv user k ...`), which now runs at the octo block when the flag is on (`packed_verifier` passes `served_only=False`); an edge probe that injects
  -0 and denormal rows into an eight-row block and its outputs would settle it independently. `test_octo_glue8_block` has a negative control that applies the sixteen-row parity rule to eight-row users and fails on users 2 and 6.
* **split / merge / stack / unstack** are bit copies (plus the canonical rule above). Proven on raw face-ordered tiles against the served composition for all eight positions, -0 / denormal / NaN payload / infinity rows included (`test_octo_glue8`).
* **block_conv.** The conv gates op is row-parallel: every output row is the four-tap FIR over that row's own window rows, a silu, and the two gates of that row's own `a` and `b`. The block launch is the M3 block's launch exactly (64 rows, same op, same compile-time shapes); only which rows hold which
  user differs. The data plumbing around it is proven equal to eight served calls at batch 8 (`test_octo_glue8_block`, with a row-leak negative control). That a row's result does not depend on its tile row and quarter is the op's property, shown by the in-trace audit, as for V1.
* **windows.** The trimmed kernel is the pinned one with the window geometry of an eight-row piece: window row `r` of slot `s` is history `r + s` while `r + s < 4`, else piece row `r + s - 4`; the piece block copied is `(4 + s)` rows (rows 0 .. 3 + s, inside the 8 the piece has); rows 8-15 of both left faces and
  faces 2-3 stay zero because each scratch page is zeroed WHOLE once (the sixteen-row kernel zeroes faces 2-3 only, which would leave rows 8-15 undefined: negative control in `test_octo_glue8_windows`). Equal to the served kernel's loop on every byte, padding included, across a sequence of tasks.
  The user cap is `gdn_seq_block.batch.MAX_USERS` (the width-aware module): the pinned op reads the pair's `gdn_user_batch` and refused the octo block with `8 users outside 1..4` in the card log.

## 4. Markers and the smoke rule

```
[PINDIAG] tp4 octo glue8 engaged site=<split|merge|block_conv|windows> users=8 rows=8     once per process per site (first native launch)
[PINDIAG] tp4 octo glue8 fell back site=<site> reason=<why>                                a native site declined; its served path ran (fails a gated arm)
[PINDIAG] tp4 vglue engaged site=packed_verify ... glue8_split=48 glue8_merge=48 glue8_block_conv=48 glue8_windows=48
                                                                                           the existing per-capture count line: one count per GDN layer of the octo block's capture
```

`octo_glue8.glue8_problems(log_text, env)` is the rule (to be called by `octo_judge` / `c2_smoke_check` for a profile that carries the flag): flag off, not one glue8 line and no `glue8_*` count; flag on, for every site `sites_wanted(env)` names, an engaged line at users=8 rows=8,
the count equal to 48 in a vglue engaged line, no `fell back` line, and no `[PINDIAG] tp4 vglue octo 8-row served path` line for that site. `octo_glue8.refusal(env)` is the admission question (needs `QWEN_FAST_TP=4`, `QWEN_FAST_OCTO` live or alternate, and at least one lever to make native).

## 5. What is NOT native, and why

* **K5-A (`QWEN_FAST_GDN_SEQ_BLOCK`)**: `gdn_seq_block` is a recurrence COMPUTE kernel qualified for sixteen-row users (`ROWS = 16`); the octo block keeps the served batched launch (`seq_block` 0 of 48 in the card log). An eight-row K5-A is a new compute-kernel build and
  a new qualification (its own K5 evidence chain), not a data-movement twin: stubbed. It is the largest remaining per-user GDN item at the octo block.
* **The pair slice (`QWEN_FAST_GDN_PAIR_SLICE`)** is the V2-off alternative and half-tile by construction: not extended (V2 replaces it).
* **V3a / V4a / C1a** are not on the list: V3a already takes bundles of one (`attn_fold=16` at the octo block), V4a and C1a are per block.

## 6. Estimate (per 8-live octo pass)

Anchors: the octo pass is 86.5 ms (about 13 ms weights + about 9 ms per user); the M3 pass is 50-54 ms with these levers on; the all-levers verify trace measured 66.69 to 58.31 ms (-8.4 ms, 2.1 ms per user) at four users against estimates summing to -11.6 ms (72% realised), of which V2 (-4.7) and V1 (-3.8) are
73%. Scaling by user count (the savings are per-user launches, independent of the rows each carries):

| site | per user | x 8 users |
| --- | --- | --- |
| split + merge (V2) | 4.7 ms x 0.72 / 4 = 0.85 ms | -6.8 ms |
| block conv (V1) | 3.8 ms x 0.72 / 4 = 0.69 ms | -5.5 ms |
| T2 windows | 8 served launches x 48 layers x 72 us = 27.7 ms serial at best, against one launch of about 40-75 us a layer (3 ms): in-trace overlap makes the real figure smaller; 0.5 to 1.5 ms a user | -4 to -12 ms |

Total about **-16 ms per octo pass (range -9 ms without the windows, -24 ms with the full serial window cost)**, i.e. the pass goes from 86.5 to about 70 ms. On the octo round (step 94.7 + gap 45.0 = 139.7 ms, 29.5 tokens) that is -11% of the round, +12% committed tokens
per second per seat on the octo shape (26.4 to about 29.5 tok/s/seat, +16% over the M3 round's 25.5), which alone would clear the +10% bar the octo arm is judged on. This is an estimate from the M3 anchors, not a measurement; the K5-A gap (section 5) is on top of it.

## 7. Card questions and jobs (templates are the parent's)

1. **Canonical set at eight rows**: the audited attach with the flag on and `QWEN_FAST_TP4_VGLUE_AUDIT=1` (octo audit profile twin carrying the flag, two tests, as the lean O1): every `piece user k`, `merged output` and `block conv user k ...` entry equal on every chip; `glue8_problems` clean; the T2 windows audit (`QWEN_FAST_VERIFY_T2_AUDIT=1`) compares the
   32 windows after the conv gates. Any mismatch: the first suspect is the canonical set; flip `gdn_rows_dma8_tp.served_canon`.
2. **The mover's first launches**: a 256-byte DRAM / L1 read at 256-byte multiples is the new access size (the half-tile mover reads 1,024); the 64-byte alignment rule holds (offsets are multiples of 256) but no card has run it. A watcher pass first if the first audited run hangs.
3. **Timing**: ABAB at eight live with CI paused, control = the same octo twin with the flag off, arm = flag on, text hashes equal (unaudited exactness), `glue8_problems` clean on the arm and no glue8 line on the control, judge with the same pairing rule as the octo pack. Read: committed tokens per second per seat, the `[PACKED-PHASE] trace_ms` of the octo block (86.5 ms before).
