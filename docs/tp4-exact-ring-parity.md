# TP4 exactness: the block's second tile is reduced in another order (branch tp4/exact)

Status: the fix and its spike are built and CPU-tested; nothing here has run on four cards. Every hardware statement below is a
prediction until the X1 verdict line says otherwise (`scripts/ci/references/tp4-exact-jobs`).

## The symptom (S3a at four cards)

Four real-text users (4,096 / 16,384 / 32,768 / 60,000 tokens) served packed (one 64-row verify block, 16 rows a user) and then one
at a time (the sequential engine, four rows, one tile). Users 0 and 1 are byte-identical for 512 tokens. Users 2 and 3 diverge in
their first verify rounds, deterministically, on a near-tie at the same absolute row with identical inputs. At the pair the same
matrix passed.

## What is ruled out

* The core split of the attention program. At one KV head both routes run 16 cores per key head, 256-key chunks, the same chunk
  ranges and the same four-round reduction tree at every extent (`factory_k64j.cpp` `min(110, 16*B*NKV)/B/NKV`,
  `rt_args_common.hpp`), so pinning the split changes nothing.
* The sampler's shard merge (audited exact on every row), the extent staging (audited on all four chips every round), the drafter
  and publication path, and the GDN kernels at 12 heads.
* Attention as a whole is not the suspect: the TP4 program (`0x23`, `reader_decode_qwen.cpp`, PNHt 2) differs from the pair's proven
  `0x27` program only in the q-slice offsets, which are zero without the slice.

## The mechanism

What TP4 changes in arithmetic: every cross-chip sum has four addends, and the model's collective runs on the RING. `tt_all_reduce`
(`models/tt_transformers/tt/ccl.py`) on a `(1, N)` mesh is the `reduce_scatter_minimal_async` alone: it returns straight after it, so
each chip keeps only its own 1280-column slice of the sum (the fabric probe and the MLP already expect that). The ring reduce-scatter sends the
even chunks of a slice forward and the odd chunks backward (`ring_reduce_scatter_minimal_async_reader.cpp`), the parity being
`(tiles_read / tile_granularity) % 2` (`reduce_scatter_common::chunk_ring_parity`), and the last step adds three terms
(`DST = interm2 + (interm + input)`, `ring_reduction.cpp`), so the two directions associate the four partials differently. At two
chips the sum commutes and the order cannot show.

The parity is taken from the tile's flat index in the per-chip slice. The slice of a `(1, 1, 64, 5120)` input at four chips is two
rows of 40 tiles. With `tile_granularity` 8 (bfloat16: `min(4 * min(4, payload / page), dst)`), the second row starts at chunk 5, an
odd one. Every chunk of rows 32..63 therefore runs in the opposite direction, hence in another order, from the same row in a
one-tile call, which is what the sequential engine issues (four rows are one padded tile). The sums differ in the last bit only;
that flips a near-tie. Users 2 and 3 sit in rows 32..63; users 0 and 1 in rows 0..31. The mechanism depends on the row tile, not on
the extent, which is why the 16k user agrees.

This reading is from the tt-metal source at v0.77.0-rc1 (the image's tag; the same code is on main). The one unknown is the fabric
payload size on this image. If the image's `tile_granularity` is 4 (an even 10 chunks a row), nothing
flips and the mechanism is wrong; the spike says which (verdict NOT_REPRODUCED).

The fabric probe (J0) could not have caught it: its inputs are small integers, exact in bfloat16 under any association, and it only
tested 32 rows and 2,048 rows, never a 64-row block.

## The fix

`scripts/ci/tile_collective_tp.py`: inside the block's forward, a `rows`-row all-reduce is issued as one call per 32-row tile,
with the model's own arguments, and the results are joined on the row axis. Every tile then runs the very call the sequential
engine runs, so it has the sequential engine's order. Nothing else changes: prefill, the sequential engine (four rows), the
drafter and every other shape pass straight through.

* Bound by `tp_addresses.install()` (called by `serving_startup.start` at `QWEN_FAST_TP != 2` only): the model's `tt_all_reduce` is
  replaced on the ccl module and in every loaded module that holds the function (`from ccl import tt_all_reduce` binds a copy in
  the graft's attention, GDN and MLP modules), the same identity sweep the other four-card twins use. The graft files, sha-pinned on
  the serving path, are untouched. The pair never installs it.
* Scoped by `tile_collective_tp.scoped_run`, which the same install wraps around `ModelBatch.run` (a class attribute: `model_batch.py`
  is in the serving bundle inventory that `test_quad_draft` holds byte for byte, so it is not edited): nothing at the pair or within
  one tile.
* Guarded: the scope refuses the round unless every block-wide all-reduce was split (128 per forward: 16 wo + 48 GDN out + 64 w2;
  64 when the MLP is not native at the block's rows, its two-tile form already reducing 32 rows at a time). A wrapper that was never
  bound where the layers look it up would otherwise run the inexact reduction silently.
* Cost: per packed round, 128 extra reduce-scatters (there is no gather to add), 256 slices and 128 concats around them; the
  3-6 ms predicted is a guess that may be low; X3 measures it.
  Cheaper variants, exact with no extra launches, left for after the measurement (neither is proven on the image): a direct
  `reduce_scatter_minimal_async` on the unit-major `(1, 2, 32, 5120)` view (its ring count restarts for each unit; only the direct
  call does, `tt_all_reduce` flattens the units first), which X1 records as informational; or Linear topology for the target's
  `tt_all_reduce` at TP4 (its reader has no chunk parity: it always adds local, forward and backward in that order).
* Refused, deliberately: a TP4 block wider than 32 rows that is not a whole number of tiles (`validate_rows`). Serving blocks are 32
  and 64; an experiment harness at another width fails loudly rather than run the inexact reduction.

## The spike (X1, four cards)

`scripts/ci/tp4_rs_tile_spike.py`, selected by `C2_FABRIC_PROBE=rs-tile` on the fabric action. A collective cannot be shown on one
card, so unlike the K64j spikes it opens all four. It reproduces first (the model's call on a whole 64- and 128-row block against the
same call on each tile alone, bit for bit: tile 0 equal, later tiles differing) and then requires the wrapper's output to equal the
one-tile calls everywhere. Controls: the one-tile call twice, four rows against the tile's first four, every chip's copy against
chip 0's, Linear topology on the whole block. The verdict line is `TP4_RS_TILE verdict=PASS|NOT_REPRODUCED|FAIL comparisons=C
differing=D ...`; exit 0 only on PASS.

At `(1, 4)` the outputs of the chips are different column slices, so every comparison joins all four chips' slices of a tile (each
chip starts its ring at a different slice index, so one chip's columns prove nothing about the others). There is no replication
control. Informational rows (Linear on the block, the direct unit-major reduce-scatter) never enter the verdict, and an error in
them is recorded, not fatal. The report records the cluster type (Ring is assumed from P150_X4, the model's own choice).

## The window

`X0` build (no graft: host Python, shipped through all three copy lists), `X1` the spike, `X2` the S3a matrix again, `X3` the smoke
with timings, `X4` (optional) the permuted matrix. X1 stops the window on FAIL and on NOT_REPRODUCED.

## Not done here

* The K64j `k2` section at one KV head (CB2a-TP4 evidence, one card): still pending, and not a reproducer, because the defect is a
  four-device collective. It gates nothing in this window.
* Extents beyond 60,000 at four cards (the ladder to 120000 and 123136) after the matrix passes.
* If X1 reports NOT_REPRODUCED: the discriminator is the permuted arm (X4) on an image without the fix.
