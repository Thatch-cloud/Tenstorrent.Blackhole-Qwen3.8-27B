# nlp_concat_heads_decode at batch 64 (two batch tiles) in one call

Staged patch of the image's
`/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/`
(tt-metal 9f9cd4fd, v0.77.0-rc1). All 13 op sources are here, byte-exact from the image except for
the changes below (each file was verified against the `sha256=` header of the source dump before
patching), so this directory can be copied or bind-mounted over the op wholesale. NOT built, NOT
run on hardware: `build-and-test-b64.sh` does that on the rig.

## What the op does

Input `[1, B, padded_heads=32, HD]`, TILE, HEIGHT_SHARDED with one user per core (validated: shard
shape is `(padded_heads, HD)` and `num_cores == input_shape[1]`). Output
`[1, 1, max(B,32), num_heads*HD]`, WIDTH_SHARDED over `num_heads` cores with shard
`(max(B,32), HD)`: output core `i` owns head `i`'s column block for every user. There is no compute
kernel - two dataflow kernels run the same reader source on both RISCs, risc0 reading the left half
of each row (faces 0/2) and risc1 the right half (faces 1/3). Each output core loops over users and
pulls head `i`'s 16-element subtile lines out of that user's input core over the NOC.

## What `input_shape[1] <= 32` protected

`device/nlp_concat_heads_decode_device_operation.cpp:39` (original numbering):

```
TT_FATAL(input_shape[1] <= 32, "currently only support less than 32 users");
```

It protected exactly one thing: **the reader's output row offset**, which was written for a single
32-row output tile.

- `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp:52` (original):
  `wptr_offset = q < 16 ? q*SUBTILE_LINE_BYTES : (q-16)*SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE`
- `..._subcoregrid.cpp:51` (original):
  `wptr_offset = q < face_h ? q*SUBTILE_LINE_BYTES : (q + face_h)*SUBTILE_LINE_BYTES`

Both map the user index `q` straight onto a row inside ONE 32x32 tile (rows 0..15 in face 0, rows
16..31 in face 2, which starts 512 elements in). At `q = 32` the first form lands at
`16*SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE`, i.e. inside **face 3 of tile 0** - it would overwrite
users 0..15's right-hand halves; from `q = 48` it runs past tile 0 into the next head-dim tile.
Silent cross-user corruption, no out-of-shard write, no hang. That is the whole of the hazard.

Nothing else in the op was one-batch-tile-bound - checked line by line, original numbering:

- `compute_output_specs` (device_operation.cpp:72-111) is already batch-generic:
  `batch = std::max<uint32_t>(batch, 32)` (:86), `Shape({seq, 1, batch, hidden})` (:90),
  `ShardSpec{output_core_grid, {batch, head_dim}}` (:105). At B=64 it already yields
  `(1,1,64,3072)` with a `(64,256)` shard over 12 cores; it needed no change.
- The CB bound to the output buffer is shape-derived:
  `q_num_tiles = q_shard_spec.shape[0]*q_shard_spec.shape[1]/TILE_HW`
  (program_factory.cpp:41, subcoregrids:51) = 16 tiles at B=64. Already right.
- **The per-core work split is per HEAD, not per batch**: output core `i` <-> head `i`, and
  `in_tile_offset_by_batch` is a *head* row offset inside the input shard, with a head-tile skip for
  padded_heads > 32 (program_factory.cpp:114-136, subcoregrids:121-143). Batch does not enter it.
- The reader's walk over input cores is already batch-driven:
  `num_tiles_per_core = (head_size_num_tiles * batch) / total_input_cores` (reader:41,
  subcoregrid:40), which equals `head_tiles` because `num_cores == input_shape[1]` is validated
  (device_operation.cpp:62, :68). So the reader advances exactly one input core per user
  (reader:81-91, subcoregrid:77-83) and user `q` reads input core `q` at any batch.

So: a validation guard over a kernel that was batch-tile-generic everywhere except the output row
offset.

## Changes

New line numbers are this directory's files; "was" is the image's.

### 1. `device/nlp_concat_heads_decode_device_operation.cpp` - lines 39-49 (was line 39)

The `<= 32` fatal becomes "at most 32 users, or a whole number of 32-user batch tiles":

```
constexpr uint32_t batch_tile_height = 32;
TT_FATAL(
    input_shape[1] <= batch_tile_height || input_shape[1] % batch_tile_height == 0,
    "batch (input_shape[1] = {}) must be at most 32 users or a whole number of 32-user batch tiles",
    input_shape[1]);
```

Why a multiple of 32 rather than any batch: the output shard is `(batch, head_dim)` in TILE layout,
so a batch that is not a whole number of 32-row tiles has no tile-aligned output shard. B < 32 stays
allowed and is padded up to one tile by `compute_output_specs` (now :96) - the "always emits batch
padded to 32" the model's `_concat_heads_decode` comment relies on for the B=1 vLLM path.

No artificial upper bound was added: the pre-existing `num_cores == input_shape[1]` checks (now :72,
:78) already cap batch at the number of worker cores the input can be height-sharded across (130 on
a p150a), and the L1 cost is `batch*head_dim*element_size` per head core (32 KB at B=64, bf16,
HD=256). `batch_tile_height` is function-local, so it cannot collide in the unity build.

### 2. `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp` - lines 51-67 (was 51-53)

The standard reader, which is the one the model takes (see "Which factory" below). The row offset is
now computed over `ceil(batch/32)` batch tiles:

```
constexpr uint32_t TILE_ROWS = 32;
...
uint32_t batch_tile  = q / TILE_ROWS;
uint32_t row_in_tile = q - batch_tile * TILE_ROWS;
uint32_t wptr_offset =
    batch_tile * head_size + (row_in_tile < 16
                                  ? row_in_tile * SUBTILE_LINE_BYTES
                                  : (row_in_tile - 16) * SUBTILE_LINE_BYTES + 512 * ELEMENT_SIZE);
```

`head_size` is the batch-tile stride, and that is an identity rather than a coincidence: the output
shard is `(batch x head_dim)` in tile layout with tiles stored row-major, `head_dim/TILE_WIDTH =
head_tiles` tiles per tile row, each `single_tile_size` bytes, and
`head_size = head_tiles * single_tile_size` (program_factory.cpp, now :39). One whole tile row of
the output shard is exactly `head_size` bytes. The kernel already derives its per-tile step from the
same quantity (`tile_size = head_size / head_size_num_tiles`, :48), so **no new compile-time
argument was added** - the compile-arg vector, the runtime args and the program descriptor stay
bit-identical to upstream at every batch.

### 3. `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp` - lines 50-65 (was 50-52)

The same change in the subcoregrid reader, with rows-per-tile as `2 * face_h` (constexpr, from
compile-time arg 8) instead of a literal 32 - the same rows-per-tile its own program factory uses to
place a head row (`i / (2 * face_h)`, subcoregrids:127 original). The face-2 form
`(row_in_tile + face_h) * SUBTILE_LINE_BYTES` is kept verbatim. The model does not reach this path,
but leaving the two readers inconsistent would be a trap for the next caller with a ragged grid.

### 4. `device/nlp_concat_heads_decode_program_factory.cpp` - comment at lines 35-38

Four comment lines above `head_size` recording that it is *also* the output shard's tile-row stride
and that the reader depends on that. No code change, no codegen change.

### 5. `device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp` - comment at lines 45-48

Same comment above its `head_size`. No code change.

### 6. `nlp_concat_heads_decode_nanobind.cpp` - docstring, line 22

`[S=1, B=32, ...]` becomes `[S=1, B, ...]` and states the new contract: B at most 32 (output batch
padded to one 32-row tile) or a whole number of 32-user batch tiles, B=64 being two. Docstring only.

### Which factory the model takes

`ttnn::prim::nlp_concat_heads_decode` sets `on_subcoregrids` only when the input shard grid has more
than one range, does not start at (0,0), or `sub_core_grids` was passed (device_operation.cpp:133-140
original). `attention/tp.py:_concat_heads_decode` builds one rectangle anchored at (0,0) (8x4 at
B=32, 8x8 at B=64) and passes no `sub_core_grids`, so it takes `NLPConcatHeadsDecodeProgramFactory`
and the **standard** reader - change 2. Change 3 keeps the other path in step.

## Why B <= 32 is unchanged

- `batch_tile == 0` and `row_in_tile == q` for every `q < 32` (`q < 2*face_h` in the subcoregrid
  reader), so `wptr_offset` reduces to the original expression term for term; `batch_tile *
  head_size` is 0. Same addresses, same NOC transactions, same order.
- The host side is untouched on that path: same CB descriptor, same two kernels, same compile-time
  argument list (nothing added or reordered), same runtime args (still `Buffer*`, never a raw
  address), same `compute_output_specs`. The only host change is the fatal's predicate, which
  accepts exactly the batches it accepted before (<= 32) and still refuses 33..63 and any other
  non-multiple of 32.
- Changes 4, 5 and 6 are comments and a docstring.

## B=64: per-core split and output indexing

- **Input**: 64 cores, 8x8 anchored at (0,0), ROW_MAJOR, one user per core, shard (32, 256) = 8
  tiles. Shard `k` sits on core `(k % 8, k / 8)`, which is the order the reader walks (x then y,
  reader:82-88).
- **Output**: still `num_heads = 12` cores, each holding `(64, 256)` = 2 tile rows x 8 tiles. The
  split across cores is unchanged (head per core); each core now does two batch tiles' worth of
  work, halved across the two RISCs exactly as before (risc0 phase 1 = left halves, risc1 phase 2 =
  right halves).
- **Indexing**: for user `q` in [0, 64), output core `i` writes at
  `cb_write_ptr_base + (q>>5)*head_size + face_offset(q & 31)` and then steps `tile_size` per
  head-dim tile, 8 times. So user `q`'s head `i` lands at output row `q`, columns
  `i*256 .. i*256+255` - i.e. `out[0,0,b,h*HD+d] == in[0,b,h,d]` for all b in [0,64).
- **Reads**: `num_tiles_per_core = 8*64/64 = 8`, so the reader advances one input core per user and
  user `q` is read from input core `q`.
- **Cost**: 512 32-byte subtile reads per RISC per output core (2x the B=32 call) - the same bytes
  the two-call workaround moves, minus one program launch, on each of the 16 attention layers.

Splitting the batch tiles across `2*num_heads` cores instead was rejected: the output is
width-sharded by head, so a second core would have to NOC-write into the head core's L1 shard with a
completion handshake, which changes the output spec the model consumes and adds a semaphore to a
kernel that currently has none.

## Traps respected

- **Full-tile DRAM pages**: not applicable. Every transfer here is L1 to L1 - the input is an L1
  height shard read over the NOC, the destination is the output's L1 shard aliased through the CB
  (`.buffer = output.buffer()`). No DRAM page is touched, so 32-byte subtile reads stay legal; they
  are the op's existing mechanism.
- **Unity build**: no namespace added or renamed. Every new identifier is function-local
  (`batch_tile_height` in `validate_on_program_cache_miss`; `TILE_ROWS`, `batch_tile`, `row_in_tile`
  inside `kernel_main`).
- **Runtime args**: untouched - still `rt_args.push_back(in_buffer)` with `Buffer*`, never a raw
  address, so program-cache hits keep re-resolving it (the decode trace depends on that).
- **docker cp nesting**: `build-and-test-b64.sh` does `docker exec rm -rf` before every `docker cp`.
- **CRLF**: every file here is LF, UTF-8.

## Unchanged files (7 of 13)

`nlp_concat_heads_decode.cpp`, `nlp_concat_heads_decode.hpp`, `nlp_concat_heads_decode_nanobind.hpp`,
`device/nlp_concat_heads_decode_device_operation.hpp`,
`device/nlp_concat_heads_decode_device_operation_types.hpp`,
`device/nlp_concat_heads_decode_program_factory.hpp`,
`device/nlp_concat_heads_decode_subcoregrids_program_factory.hpp`.

They are present byte-exact so the directory can replace the op wholesale.

## Test and build

- `test_concat_heads_b64.py` - builds `[1, B, 12, 256]` on the model's own memory config and grid
  (shard `(TILE_SIZE, 256)`, HEIGHT, ROW_MAJOR, 8x4 at B=32 and 8x8 at B=64) and checks the
  `[1, 1, B, 3072]` output **bit-exactly** against `x.reshape(1, 1, B, 12*256)`, because the op is a
  pure permutation. Cases: B=8 (the padded-batch regression), B=32 (production), B=64 (new), and
  B=64 against the two 32-user halves joined on the user axis - the answer
  `scripts/ci/two_tile_decode.TwoTileConcatHeads` produces today. Prints `CASE B=64: PASS|FAIL` and
  a final `RESULT`, exit 0 iff everything passed.
- `build-and-test-b64.sh` - stages the 13 sources only, `docker exec ttbuild rm -rf` +
  `docker cp` into
  `/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode`,
  `ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so` (~25 s; the binding is
  `build_Release/ttnn/_ttnn.so`, not the stale source-tree copy), refreshes `~/opgraft-K64`, then
  runs the test through `~/kwork64/test-k64.sh` when present (card M, `timeout 900`,
  `TT_METAL_WATCHER=5`) or directly otherwise. The op is already in the experimental/transformer
  CMake glob, so nothing has to be registered. On a hang: `docker rm -f k64concat` (timeout does not
  stop the container), then `~/.local/bin/tt-smi -r`.

## Residual risks

1. **Shard orientation is still unvalidated.** The reader assumes input shard `k` is on the `k`-th
   core in ROW_MAJOR order. A COL_MAJOR input shard would transpose users. Pre-existing at B=32; the
   model passes ROW_MAJOR (attention/tp.py:396) and the test pins it. Not tightened here because a
   new fatal could refuse a caller we cannot test.
2. **One-past-the-end runtime-arg load.** After the last user's tiles the reader still advances the
   core cursor and loads `in0_mcast_noc_x/y` one entry past the array (reader:82-88,
   subcoregrid:78-80). Pre-existing upstream; the value is never used and no NOC transaction
   results. Left untouched deliberately to keep the diff minimal.
3. **`preallocated_output` bypasses `compute_output_specs`** (device_operation.cpp:74-76 original).
   An output whose shard height is not `max(batch,32)` would make the reader walk off the shard -
   pre-existing, but the blast radius grows with batch. The model does not preallocate.
4. **Batch above 64** (96, 128) is now accepted and is arithmetically covered by the same
   `batch_tile` term, but is untested. The ceiling in practice is one worker core per user plus
   `batch*head_dim*element_size` of L1 per head core.
5. **`batch` is a compile-time argument**, so B=32 and B=64 are separate program-cache entries -
   switching batch recompiles rather than mis-dispatching a cached program.
6. **Nothing here has been built or run.** Compile errors and device behaviour are unverified until
   `build-and-test-b64.sh` runs on the rig.
