# attn_decode_prep at batch 64 — the deadlock and the fix

Staged patch for `/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/attn_prep/`. This directory
mirrors the image's layout exactly and contains **all twelve** source files, so it can be dropped
over the source tree wholesale (`rm -rf` the target first, then `docker cp` — `docker cp` nests
when the target exists). The three non-source files here (`PATCH-NOTES.md`,
`test_attn_prep_b64.py`, `build-and-test-b64.sh`) are deleted inside the container by
`build-and-test-b64.sh` before the build; neither `sources.cmake` (explicit file list) nor the
kernels glob would pick them up anyway.

Every file was reconstructed byte-for-byte from the image dump — the nine unchanged files still
hash to the image's sha256 prefixes:

| file | image sha256 (16) | changed |
| --- | --- | --- |
| `attn_prep.cpp` | `9aeba2de7a4c6936` | no |
| `attn_prep.hpp` | `cbde502845bb67c5` | no |
| `attn_prep_nanobind.cpp` | `8606d7d70ae2c06c` | no |
| `attn_prep_nanobind.hpp` | `6be6099395fc3611` | no |
| `device/attn_prep_device_operation.cpp` | `6a1c85e09b4acab4` | no |
| `device/attn_prep_device_operation.hpp` | `8c7be0de3b72460a` | no |
| `device/attn_prep_device_operation_types.hpp` | `207f1e173a4aecbb` | no |
| `device/attn_prep_program_factory.hpp` | `d83a4624c51871ac` | no |
| `device/kernels/compute/attn_prep.cpp` | `90090093c05df47a` | no |
| `device/attn_prep_program_factory.cpp` | `757880fa91b955b6` | **yes** |
| `device/kernels/dataflow/reader_attn_prep.cpp` | `4617d4bd37c03b64` | **yes** |
| `device/kernels/dataflow/writer_attn_prep.cpp` | `e960d2557a0eafa4` | **yes** |

---

## 1. What actually hangs, with file:line

### 1.1 The work split at B = 64

`device/attn_prep_program_factory.cpp:60,65-75` (original line numbers):

```
60   const uint32_t n_inst = attrs.B * 4;                          // instance = (batch row, kind)
65   const CoreCoord grid = device->compute_with_storage_grid_size();
66   const uint32_t grid_y = grid.y;
67   const uint32_t ncores = grid.x * grid.y;
68   const uint32_t per_core = (n_inst + ncores - 1) / ncores;
69   const uint32_t n_cores  = (n_inst + per_core - 1) / per_core;
73       active_cores.push_back(CoreCoord{c / grid_y, c % grid_y});
165      const uint32_t start = c * per_core;                        // contiguous instance range
```

Kind is `inst % 4` (0 = q, 1 = k, 2 = v, 3 = gate); kinds 0/1 are **computed** (the reader fills
`cb_blk`, the compute produces the finished block into `cb_out`), kinds 2/3 are **copied** (the
reader produced the finished block straight into `cb_out`, the compute skips them at
`device/kernels/compute/attn_prep.cpp:209-211`).

The grid on this card is 12 x 10 (`x = 0..11`, `y = 0..9`) — read straight off the watcher dump of
run 35507675630, which shows live cores at `x = 11` and `y = 9`, and which pins `grid_y = 10`
through the `c / grid_y, c % grid_y` mapping. So `ncores = 120`, and:

| B | `n_inst` | `per_core` | `n_cores` | instances on core c | kinds on core c |
| --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1 | 4 | `{c}` | one kind |
| 3 | 12 | 1 | 12 | `{c}` | one kind |
| 8 | 32 | 1 | 32 | `{c}` | one kind |
| 32 | 128 | **2** | 64 | `{2c, 2c+1}` | `{q,k}` (c even) or `{v,gate}` (c odd) |
| 64 | 256 | **3** | 86 | `{3c, 3c+1, 3c+2}` | **mixed**: see below |

At B = 64 the start instance is `3c`, so `3c mod 4 = 3·(c mod 4) mod 4` and the four residues give:

| `c mod 4` | kinds on the core | cb_out producers on that core |
| --- | --- | --- |
| 0 | q, k, v | compute **and** reader |
| 1 | **gate, q, k** | **reader first, then compute** |
| 2 | v, gate, q | reader **and** compute |
| 3 | k, v, gate | compute **and** reader |

### 1.2 The bug: `cb_out` has two producers

`cb_out` (`attn_prep_program_factory.cpp:33`, `add_cb(cbap::out, HDt, 2, df_io)` at `:89`) is
written by **two different RISCs**:

* `reader_attn_prep.cpp:109` — `const uint32_t target = math ? cb_blk : cb_out;` then
  `:113-115` reserve / `:150` `tcb.push_back(HDt)`. The reader runs on NCRISC
  (`ReaderConfigDescriptor` → RISCV_1).
* `compute/attn_prep.cpp:260` `cb_reserve_back(cb_out, HDt)` / `:313` `cb_push_back(cb_out, HDt)`,
  executed by the compute's PACK thread (TRISC2).

and read by one consumer, `writer_attn_prep.cpp:42-43`:

```
42       CircularBuffer cb(cb_out);
43       cb.wait_front(HDt);          <-- CWFW
```

A Metal circular buffer is a **single-producer, single-consumer** object. The producer's page
count is held in that RISC's *own local* CB interface and is **stored** into the shared L1 word the
consumer polls, not added to it (`llk_push_tiles` writes
`tiles_received_ptr[0] = cb_interface[out].tiles_received`). With two producers each keeping its
own local count, the second producer's store overwrites the first's: any push whose local count
does not exceed what the other producer last stored is **lost outright**, and the consumer's
`cb_wait_front` for the final block never satisfies. (On top of that, each producer also keeps its
own local write pointer, so both start writing at slot 0 and the data is scrambled even when the
counts happen to survive.)

### 1.3 The exact deadlock on the `c ≡ 1 (mod 4)` cores

Let `P = HDt` pages per block (HD = 256 → `HDt = 8`), `R` = the shared received count, `A` = the
shared acked count, and each producer's local count in brackets. Core kinds are **gate, q, k** —
and the ordering is *deterministic*, not a race: the gate instance is the reader's **first** action,
while the compute cannot push anything until the reader has filled `cb_blk` for the *second*
instance.

| step | actor | effect |
| --- | --- | --- |
| 1 | reader pushes gate → `cb_out` | reader local 0→P, `R = P` |
| 2 | writer `wait_front(P)`: `R − 0 = P` ✓ | writes the gate block to the gate output, `pop` → `A = P`, writer acked = P |
| 3 | compute pushes q → `cb_out` | compute local 0→P, **stores P** → `R = P` (unchanged: **one block's receipt lost**) |
| 4 | writer `wait_front(P)`: `R − P = 0` | **blocks** |
| 5 | compute pushes k → `cb_out` | compute local P→2P, stores → `R = 2P` |
| 6 | writer unblocks: `2P − P = P` ✓ | pops, writes to the **q** pages, `A = 2P`, acked = 2P |
| 7 | writer `wait_front(P)` for k: `2P − 2P = 0` | **blocks forever** — reader has exited, compute has exited |

That is exactly the watcher state in run 35507675630, dump #43 at 846 s (both chips identical):

```
Device 0 worker core(x= 0,y= 1) virtual(x= 1,y= 3): CWFW,   W,   W,   W,   W  rmsg:D1G|BNT ...
```

BRISC (the writer) in `CWFW`; NCRISC (the reader) and all three TRISCs (the compute) showing `W`
with `smsg:DDDD` — i.e. **done and waiting for the next go**. 21 attn_prep cores in that state:

```
(0,1)(2,1)(4,1)(6,1)(8,1) (1,3)(3,3)(5,3)(7,3) (0,5)(2,5)(4,5)(6,5)
(1,7)(3,7)(5,7)(7,7) (0,9)(2,9)(4,9)(6,9)
```

Mapped through `c = x*10 + y` these are exactly `c = 1, 5, 9, …, 81` — **every `c ≡ 1 (mod 4)`**,
and only those. `c = 85` (the last core, `(8,5)`) escapes because `n = min(per_core, n_inst − start)
= min(3, 256 − 255) = 1`, so it runs a single copy instance. That is the "checkerboard": even/odd
`c` are vertically adjacent in the `c/10, c%10` mapping, so every fourth `c` draws alternating
columns on alternating rows.

The other three residues also mix producers and are equally broken — they merely resolved benignly
in this run because their reader push landed after the compute's, where uint16 wraparound in
`cb_wait_front` masks the lost store. They write **garbage** (for `c ≡ 3`, all three outputs land in
the wrong tensors) and they can hang on a different interleaving. They are not "the cores that
work"; they are the cores that got lucky.

### 1.4 Why B ≤ 32 is fine, and what it is *not*

`B ≤ 30`: `n_inst ≤ 120 ≤ ncores`, so `per_core = 1` — one instance per core, one producer.
`B = 32`: `n_inst = 128 > 120`, so `per_core = 2` and `start = 2c` is always **even**, so a core's
two instances are `{4m, 4m+1}` = `{q, k}` (compute-only producer) or `{4m+2, 4m+3}` =
`{v, gate}` (reader-only producer). **Never both.** The op has always been one adverse work split
away from this hang; B = 32 is the last batch where the arithmetic happens to keep the producers
apart.

Things that were checked and are **not** the bug (all of them already handle two batch tiles):

* `reader_attn_prep.cpp:111` `const uint32_t row_page0 = (b / 32) * Wt;` — the projection read is
  already batch-tile aware; at B = 64 pages 224..447 of the 448-page qkv are read correctly.
* `reader_attn_prep.cpp:154-155` cos/sin page `b*RDt + t` — correct for `[1,B,1,RD]` TILE
  (128 pages at B = 64).
* `writer_attn_prep.cpp:46` page `b*HDt + t` — correct for `[1,B,NH,HD]` and for the 64-shard
  `[1,B,32,HD]` height-sharded K/V (the accessor is built from the real output buffer at
  `attn_prep_program_factory.cpp:127`, so the 8x8 grid is baked in correctly).
* `attn_prep_device_operation.cpp:22-48` — there is **no** `batch <= 32` / single-tile validation
  to widen. The only `<= 32` assert is `:32`, and it is about *heads per device*
  (`NH <= 32 && NKV <= 32`), not batch. Nothing in validation needed changing, and nothing was
  changed there.
* CB sizing, per-CB page counts and every loop bound are already per-instance and already balance:
  per core, reader pushes = compute pops and compute pushes + reader pushes = writer pops, for any
  instance set. The failure is not a count mismatch — it is two producers on one ring.

---

## 2. The change

Minimal and split-independent: **give the reader's finished blocks their own ring.** `cb_out` then
has exactly one producer (the compute's PACK thread) and the new `cb_cpy` exactly one (the reader);
the writer picks the ring by kind. Three files, eight edits.

### 2.1 `device/attn_prep_program_factory.cpp`

| new lines | change | why |
| --- | --- | --- |
| 9-23 | header comment: `cbap::cpy` described, plus a **ONE PRODUCER PER CIRCULAR BUFFER** paragraph spelling out the local-count / shared-store mechanism and why B ≤ 32 survived | the next person to touch this must not re-merge the rings |
| 65 | `constexpr uint32_t cpy = tt::CBIndex::c_23;` added to `namespace cbap` | a free CB index (the op used c_0..c_22; Blackhole has c_0..c_31) |
| 102 | `add_cb(cbap::cpy, HDt, 2, df_io);` | same geometry as `cb_out`: `HDt` tiles, 2 buffers, Float16_b — so the copy path keeps the same double-buffered pipelining it had |

The namespace is still `cbap`, unique in the unity build — no new `namespace cb {…}` was
introduced. Runtime args are untouched and still pass `Buffer*`, never raw addresses. The work
split, core set, core mapping, compile-time args and every runtime arg are **unchanged**.

### 2.2 `device/kernels/dataflow/reader_attn_prep.cpp`

| new lines | change |
| --- | --- |
| 11-14 | header comment: v/gate go to `cb_cpy`, and why they must not go to `cb_out` |
| 26 | `cb_out = 2` replaced by `cb_cpy = 23` in the CB index list (the reader no longer touches `cb_out`) |
| 112 | `const uint32_t target = math ? cb_blk : cb_cpy;` (was `: cb_out`) |

Nothing else moves: the same `reserve_back(HDt)` / zero-fill / grouped `HG = 4` DMA gather /
row scatter / `push_back(HDt)` sequence, the same full-tile-page DRAM reads, the same cos/sin
reads. Only the destination ring index changes.

### 2.3 `device/kernels/dataflow/writer_attn_prep.cpp`

| new lines | change |
| --- | --- |
| 8-11 | header comment: two rings, one producer each |
| 21 | `constexpr uint32_t cb_out = 2, cb_cpy = 23;` |
| 47 | `CircularBuffer cb(kind < 2 ? cb_out : cb_cpy);` (was `CircularBuffer cb(cb_out);`) |

`tb = get_tile_size(cb_out)` at line 37 is kept: both rings are Float16_b with the same page size,
so one tile size serves both. The write loop, the page arithmetic, the four accessors and
`async_write_barrier()` / `pop_front(HDt)` are untouched.

### 2.4 `device/kernels/compute/attn_prep.cpp` — **unchanged, byte for byte**

It was already a single producer of `cb_out` and a single consumer of `cb_blk`/`cb_cos`/`cb_sin`.
Leaving it identical keeps every numerical property fixed: the rms-norm reduction, the two bf16
round-trips that match the composed chain's rounding points, the rotate-half RoPE, the HiFi4 /
`fp32_dest_acc_en` config. **No math changed.** The Blackhole binary-op rule is untouched too —
every binary op still reads two distinct rings, and no operand spans two separately pushed blocks.

---

## 3. Why B ≤ 32 is unchanged

* Program factory: identical `n_inst`, `per_core`, `n_cores`, `active_cores`, `CoreRangeSet`,
  compile-time args and runtime args. Same cores, same instances per core, same order.
* Reader: identical DMA pattern, identical block contents, identical push order. On a B ≤ 32 core
  the reader was already the *sole* `cb_out` producer whenever it produced at all, so moving those
  pushes to `cb_cpy` is a pure relabelling of a private ring.
* Writer: the block it pops for a given instance is the same block, from the same producer, in the
  same order — only the ring index differs.
* Compute: not touched.
* Outputs are therefore bit-identical at B = 1, 3, 8, 32, and `v_exact` / `g_exact`
  (`torch.equal`) still hold because v and gate still never pass through the FPU.

The only cost anywhere is 32 KB more L1 per participating core for the second ring.

## 4. The batch-64 work split and CB budget after the fix

* `n_inst = 256`, `ncores = 120` (12 x 10), `per_core = 3`, `n_cores = 86`; core `c` runs
  instances `3c … 3c+2` (core 85 runs one). Cores `c = 0 … 85` map to `(x, y) = (c/10, c%10)`,
  i.e. `x = 0..8`.
* Per instance the work is one batch row's one kind: a 32-row x HD block = `HDt = 8` tiles.
  A core gathers at most 2 computed blocks and at most 2 copied blocks.
* Per-core rows: q/gate instances gather `NH = 12` head rows (rows 12..31 zeroed), k/v instances
  gather `NKV = 2` head rows (rows 2..31 zeroed), out of `Wt = 224` projection tiles per batch
  tile; the batch tile is selected by `(b / 32) * Wt`, so rows 32..63 read pages 224..447.
* Critical path: the heaviest core runs **two** compute chains — the same as B = 32, where
  `per_core = 2` already put two computed instances on every even core. B = 64 adds one copy
  instance to that core, not a second math chain. The expected B64/B32 device time is well under
  the 2x gate.
* CB pages per core (page = one tile): `src` 32 bf16, `blk` 16 bf16, `out` 16 bf16,
  **`cpy` 16 bf16 (new)**, `cos`/`sin` 4 bf16 each, `wq`/`wk` 8 bf16 each, `rt` 8 bf16,
  `wqf`/`wkf`/`xf`/`sq`/`xn`/`xw`/`xwr`/`xnr` 8 fp32 each, `cosf`/`sinf` 2 fp32 each,
  `ones`/`sc`/`fac`/`ta`/`tb` 1 fp32 each. Total **516 KB** per core (was 484 KB), plus the
  16 KB K shard + 16 KB V shard on the 64 cores of the 8x8 KV grid = 548 KB worst case, against
  ~1.4 MB of usable Blackhole L1.
* Deadlock freedom now holds for **any** contiguous instance split, not just the lucky ones: with
  one producer per ring, the writer at instance `i` can only be waiting on a ring whose block the
  producer for instance `i` can still make, because the reader is never behind the writer and the
  reader emits `cb_blk`/`cb_cos`/`cb_sin` for an instance before it emits anything for a later one.

## 5. Residual risks — what could still hang, and what the watcher would show

1. **A `cb_cpy` typo / missing CB.** If `add_cb(cbap::cpy, …)` were dropped but the kernels still
   used index 23, the reader would write into an unconfigured CB. Watcher: reader in `CRBW` or a
   bad-address assert; writer in `CWFW`. Mitigation: the index is defined in one place per file and
   all three agree (23 / `c_23`); verified by grep before staging.
2. **Load imbalance, not a hang.** `c ≡ 1 (mod 4)` cores run two math chains while `c ≡ 2` run one.
   That is already true at B = 32 and is bounded by the 2x timing gate in the test.
3. **L1 pressure at larger B.** The 32 KB second ring is charged to every participating core. At
   B = 64 the KV shards land on 64 of the 86 program cores, worst case 548 KB — fine. A future
   config that also height-shards q/gate into L1 on the same cores would need re-checking; the
   symptom would be a *host* `TT_FATAL` about circular buffers growing past the L1 bank, not a hang.
4. **B not a multiple of 32 above 32** (e.g. B = 48). Nothing in this patch assumes it — the reader
   already indexes the batch tile with `(b / 32) * Wt` and the split is per instance. Untested on
   hardware here; `kv_cfg(48)` would build a 6 x 8 grid and the test only covers 1/3/8/32/64.
5. **`nlp_concat_heads_decode` still refuses `input_shape[1] > 32`** downstream
   (`nlp_concat_heads_decode_device_operation.cpp:39`). Fixing attn_prep at 64 moves the failure to
   that host-side `TT_FATAL`; it is a separate patch. Expect the B = 64 *model* path to raise there
   even once this op is green — the standalone test in this directory does not touch it.
6. **If it still hangs**, `TT_METAL_WATCHER=5` writes `generated/watcher/watcher.log`. Read the
   waypoint columns as BRISC (writer), NCRISC (reader), TRISC0/1/2 (compute unpack/math/pack):
   * writer `CWFW` + reader/compute `W` with `smsg:DDDD` → still a lost producer count: some ring
     still has two producers, or a push/pop count no longer balances.
   * writer `CWFW` + compute `UPAD`/`MWDD` → the Blackhole unpack→math→pack stall: a binary op
     whose operands share a ring or span two pushed blocks. Nothing in this patch introduces one.
   * reader `CRBW` → a ring too small for the per-core instance count; raise the `nbuf` on that CB.
   Recovery: `docker rm -f` the test container first (a `timeout` on `docker run` does not stop it),
   then reset that card only with the `tt-smi -r` command `build-and-test-b64.sh` prints. A bare
   `~/.local/bin/tt-smi -r` resets every board, the serving pair included.
