# TP4 recurrence value split, lever V5 (branch tp4/next-3-scope)

Design only. **This is not a host-side change, so no flag and no code were added** (`QWEN_FAST_GDN_SPLIT_V` is reserved here for the build that implements it). The split needs new C++ kernels, and the existing three are hash-qualified. Nothing here has run on a card.

Labels: (m) measured, (src) read from the code, (cm) computed from measured numbers, (e) estimate, (U) unknown.

## Answer first

- **What:** the K5-A packed recurrence runs one (user, head) per core: 4 users x 12 heads = **48 cores** of 110 (src, `gdn_user_batch_tp.core_shares`), 9.29 ms a four-user trace, 193.5 us a layer (m). Each core's 128-column value head is four 32-column tiles. Split the value columns x2: each core runs the chain over two of the four tile columns, **96 cores**, and the chain gets about half as long.
- **Why it is not host-side:** all three K5-A kernels assert four value tiles and use the head's tile count both as the local width and as the stride in global page addresses; and the fused norm at the end reduces over all 128 value columns, so the two halves must meet. Both need kernel code. Placement and runtime-argument plumbing are host-side but useless without it.
- **Exactness:** exact by construction for the chain (every per-token op is per value column and keeps its operands and K order); the one cross-column step, the final RMS norm, is kept byte for byte by running it on one core of each pair over bit copies of both halves. Proof on a card is the existing P0 byte compare against the served launch, extended to the split build.
- **Budget:** circular buffers fit unchanged: 630,784 B (616 KiB) a core on 96 cores, the footprint TP2 ran at (cm).
- **Expected:** the audit's estimate is -4.3 ms a four-user round, -8.6 ms at eight seats (two blocks) (e). Interpolating the TP2 value-split measurements gives -3.1 to -3.5 ms, so I would plan on **-3 to -4 ms** (below).

## 1. Today (src unless marked)

- Placement: `gdn_user_batch_tp.core_shares(grid_x, grid_y, users)` hands each user 12 contiguous workers, worker `w` at `(w // 10, w % 10)` on the 11 x 10 grid. Four users occupy columns x = 0..4 (48 cores); the other 62 cores (columns 5 to 10 and the last two cores of column 4, about 89 MB of L1) are idle during the recurrence (cm: 110 - 48).
- Per core, per 16-token block: a prologue (bf16 to fp32 copies, exp(g), q and k L2 norms over all 16 rows), a 16-token chain, and an epilogue (rms_norm over the head's 128 values times `norm_w` times silu(z), 16 rows). The chain per token is T1 state copy, T2 decay, T3 `v_read = k~ @ h`, `delta = v - v_read`, `delta *= beta`, T4 row broadcast, T5 rank-1 update with the add in DST, T6 one copy packed twice (snapshot plus bf16 feedback), T7 `o = q~ @ h_new`.
- State is 16 tiles (4 K tiles x 4 V tiles), column major in the CBs: tile (i, j) at page `4j + i`. The 32 KiB bf16 snapshot of each token leaves on both NoCs (K rows 0 and 1 from the writer, rows 2 and 3 from the reader), into the served states layout, page `(h + H t) * 16 + 4 i + j`.
- 31 circular buffers, 630,784 B (`gdn_seq_block.cb_bytes()`, checked: 616.0 KiB).
- The TP2 history: a value split x4 (96 workers = 24 heads x 4 columns, `gdn_vsplit.py`) was built, passed native correctness on both chips, and was measured: its recurrence stage 0.194 ms and a separate norm stage 0.053 ms per launch, against 0.526 ms for the then-served fused launch (m, `docs/gdn-user-batch-launches.md`); in the isolated synthetic harness it was about 4% slower than the 24-worker fused arm at T16 (0.546 against 0.523 ms, `docs/experiment-execution.md`) and was not integrated. Two lessons: halving the chain works, and a separate norm stage plus an fp32 DRAM bridge gives most of it back. This design keeps the norm in the launch.

## 2. What splits exactly

Per value column j (a 32-wide tile column), the chain reads and writes only column j of the state plus per-token operands that every core holds in full (k~, q~, the decay, beta):

| Step | Tile ops | Couples columns? |
|---|---|---|
| T1 state copy, T2 decay | per tile | no |
| T3a `v_read = k~ @ h` (`mm_cols`) | one DST per column, K order ki = 0..3 | no |
| T3b/c delta, T4 row broadcast | per column | no |
| T5 rank-1 update (`outer_add_cols`) | per column, windows of two tiles | no |
| T6 snapshot and feedback (`state_out`) | per column | no |
| T7 `o = q~ @ h_new` (`mm_cols`) | one DST per column | no |
| Prologue q and k L2 norms | over the 4 K tiles | no: both cores compute the same bits redundantly |
| **Epilogue rms_norm** | `rowsum_k` over all 4 value tiles of o squared | **yes** |

So a core holding columns {c0, c0+1} runs T1..T7 on 8 state tiles, with identical tile ops, operands, and K order. Nothing about a column's arithmetic changes.

The epilogue is once per block, over 16 rows, not per token. Its sum of squares is an fp32 reduction in a fixed tile order; adding two partial sums would change the order and could change bits. So the exact scheme moves **data, not partial sums**: one core of the pair (the owner) receives the other's two fp32 `o` tiles and runs today's epilogue unchanged at four tiles.

## 3. Design

Split factor s = 2 (flag value `QWEN_FAST_GDN_SPLIT_V=2`; `1` or unset is today's launch; anything else raises; needs `QWEN_FAST_GDN_SEQ_BLOCK=1`, TP 4, and `users * heads * s <= grid cores`). The factor 4 (one value tile a core, 192 cores) does not fit; s = 4 fits only the lone lane (12 heads x 4 = 48 cores).

**Placement.** `core_shares(grid_x, grid_y, users, workers=heads * s)` already takes the worker count; at s = 2 it gives 96 distinct points, x = 0..9 (checked). Worker `w` of a user is head `w // 2`, half `w % 2`, first tile column `c0 = 2 * (w % 2)`. The partners of a head are adjacent cores in one grid column (w even, w + 1), never across a column wrap. Half 0 is the **owner** (epilogue, z, norm_w, output write); half 1 is the **helper**.

**Runtime arguments** (the compute kernel starts to use its runtime arguments (it is handed `[rows]` today and reads none); the launch stays one descriptor per role, so the coalescing and the T1 gate's one `coalesced` per layer are unchanged): head, `c0`, owner flag, and for the helper the owner's NoC coordinates (from `device.worker_core_from_logical_core`, the pattern `ordered_cache` uses). One semaphore (`SemaphoreDescriptor`, initial 0) over the union of the ranges, per the same precedent.

**Per core, owner and helper both:** reader stages state columns `c0, c0+1` (`s0` pages `h*16 + i*4 + (c0 + j)`), v tiles `VOT + h*4 + c0 + j`, and the shared q/k/beta/g operands exactly as today; the compute runs the chain at two local value tiles; the snapshot writer writes its 8 pages a token at page `(h + H t) * 16 + 4 i + (c0 + j)` (4 a NoC, so each token's 32 KiB is now two cores' 16 KiB; per-core DRAM write volume halves).

**Epilogue handoff, once a block.** The helper's writer assembles its two fp32 `o` tiles (rows 0..15, 4 KiB a tile with the faces already zeroed) exactly as the owner's does, then NoC-writes them into the owner's `O` ring pages 2 and 3 and increments the owner's semaphore. The `O` ring has the same L1 address on every core of the launch (CBs are allocated once over the union), so no address is exchanged. The owner's writer assembles pages 0 and 1, waits for the semaphore, then pushes four tiles; the owner's compute runs the existing epilogue (`WAIT(SB_O, 4)` and on) unchanged. The helper skips the epilogue and the output write. Cost per layer: one 8 KiB NoC write and a semaphore round, a few microseconds (e), 0.1 to 0.2 ms over 48 layers.

**Not changed:** the gated output tensor (the owner writes all four tiles to `h*4 + tile`), the states tensor and its layout (every page still written once, by whichever core owns it), `gdn_records`, `gdn_commit_dma`, `restore_prefix`. The audit (`QWEN_FAST_GDN_SEQ_BLOCK_AUDIT`) compares the model's K5-A output and prefix states against the served launch's, so it audits the split build unchanged.

## 4. Circular-buffer and L1 budget

| Quantity | Value |
|---|---|
| Physical L1 a core | 1,572,864 B |
| Firmware base | 111,488 B |
| `L1_SMALL` | 24,576 B |
| Allocatable a core | 1,436,800 B |
| K5-A CBs a core (unchanged plan) | 630,784 B (616 KiB) |
| Left a core | 806,016 B |
| Total on 96 cores | 60.5 MB (30.3 MB today on 48) |

A first build keeps the CB plan as is: the chain's rings (state 16, feedback 16, S 16, H 16 pages) are twice what two tiles need but the owner's epilogue uses them at four tiles, and the same plan on all 96 cores means no per-core-set CB sizes. A leaner plan (helper without the epilogue rings) is a later size cut, not needed to fit. Held outside this launch: interleaved L1 tensors (1.0 MB a chip over the whole trace, at most about 10 KB a core), and the SDPA's 796,864 B of CBs, which does not coexist with K5-A in the trace (ops serialise). tt-metal rejects a static CB that clashes with an L1 buffer when the program is created, naming both addresses: the failure is loud at attach or capture, not silent corruption.

## 5. Expected gain

| Source | Number |
|---|---|
| K5-A today | 193.5 us a layer, 9.29 ms a trace (m) |
| TP2 value split x4, recurrence stage alone | 0.194 ms, plus 0.053 ms for its separate norm stage (m, served kernels) |
| TP2 served fused launch, T16 | 0.526 ms, norm inside (m) |
| Linear fit through those two (time = f + a x value tiles) | f about 0.10 ms, a about 0.093 ms a tile; two tiles give 0.61 of the four-tile chain (e) |
| Applied to K5-A's 193.5 us a layer, chain share 90 to 100% | -68 to -76 us a layer, -3.3 to -3.6 ms over 48 layers (e) |
| Less the pair exchange | 0.1 to 0.2 ms |
| Net a four-user trace | **-3.1 to -3.5 ms** from the fit; the audit's point estimate is -4.3 (e), which needs a 46% cut |
| Eight seats (two verify traces) | -6.2 to -7.0 ms from the fit; -8.6 at the audit's figure (e) |

A single 128-row block at 8 seats (the weights read once) would put 8 users x 12 heads = 96 cores on the unsplit chain at once and capture the same cores without a kernel change; the two are alternatives for 8 seats, not additive. At 4 seats this is the only way to use the idle 62 cores for the chain.

## 6. The exact files

**Do not edit** (their sha256 triple is what the K5-A level-0 qualification and the tests hold): `scripts/ci/gdn_seq_block.py`, `gdn_seq_block_reader.cpp`, `gdn_seq_block_writer.cpp`, `gdn_seq_block_compute.cpp`, `gdn_user_batch.py`, `gdn_multitoken.py`, `gdn_vsplit.py` and the other pinned sources in `tp2_pinned_sources.json`.

**New:**
- `scripts/ci/gdn_seq_block_split_reader.cpp`, `..._writer.cpp`, `..._compute.cpp`: copies of the three kernels with the split below. Reader: drop `static_assert(Kt == 4 && Vt == 4)` for local `LVt`; global stride stays 4 in `s0` (`h*Kt*4 + i*4 + c0 + j`), v and z page ids (`VOT + h*4 + c0 + j`), and token staging (`slot`, the `half` pages); `c0` and the owner flag as runtime arguments; the helper does not read z or `norm_w`. Writer: `base_page` and the snapshot loop at `LVt`, `start = h % half` rotation recomputed for 4 pages a NoC, the helper's `O` write to the owner and the semaphore increment, the owner's wait before `o.push_back`, the helper's skipped output write. Compute: `SB_VT` becomes a local width `LVt` in `mm_cols`, `outer_add_cols`, `state_out` and the chain's `WAIT`/`POP` counts (T3a..T7), with the epilogue at four tiles on the owner only (a runtime owner flag, since the compute has none today).
- `scripts/ci/gdn_seq_block_split.py`: the sibling module (the `gdn_user_batch_tp` pattern): the flag reader, `generate()` over the three new sources and the native compute prefix (hash-checked like `gdn_seq_block.native_prefix`), `build_program` with the placement, runtime arguments, semaphore and 96-core union, its own `QUALIFIED` triple (empty until a card passes P0), and `execute`.
- `scripts/ci/test_gdn_seq_block_split.py`: placement (96 distinct points, partners adjacent), argument table, the page-address maps of the new kernels against `gdn_seq_block`'s for s = 1 (must be identical) and a mirror proof that s = 2 covers every state page exactly once, CB budget, flag grammar, the qualification refusal.
- `scripts/ci/gdn_seq_block_split_device_test.py` (or a `--split` mode of `gdn_seq_block_device_test.py`): the one-card P0 at TP4 shapes.

**Edit:**
- `scripts/ci/gdn_user_batch_conv.py` (lines 178 to 183): choose `gdn_seq_block_split.execute` when the split flag is on. Not pinned.
- `scripts/ci/tp_shapes.py`: `k5_reader_arguments` gains the split's local tile count (the module is not pinned).
- Copy lists the closure tests demand: `docker/qwen-c2-overlay.txt`, `docker/qwen-fast-serving.Dockerfile` (the K5-A line), `.github/workflows/qwen-fast-serving-image.yml` (both loops), `.github/workflows/qwen-integration-cpu.yml` (the new test).
- `scripts/ci/qwen_c2_profiles.json`: a gate-only profile (production recipe plus `QWEN_FAST_GDN_SEQ_BLOCK`'s split flag), added with its twin test, only when the build exists.

## 7. Gate plan (for the build that implements it)

1. CPU: the test module above, plus `test_gdn_seq_block`, `test_tp2_pins`, `test_tp4_closure_literals` unchanged and green.
2. One card, TP4 shapes (12 heads, 4 users): P0 byte compare of the split build against the served launch over the plan's cases (142 at level 0), traced replay exact, ramp of tokens 1 to 16, poisoned helper tiles (a stale helper write must be detected); then the timed launch against K5-A at four users.
3. Four cards: the audited smoke with `QWEN_FAST_GDN_SEQ_BLOCK_AUDIT=0,23,47`: every line `mismatches=0`.
4. Hang shapes, audits off, five of five (it changes no allocation, but it adds a semaphore and a launch shape to a trace the hang fix depends on).
5. Paired ABAB timing against the production recipe, coding prompts; success: the recurrence family -3 ms or better at four users, round paired per round, accepted-prefix sequences identical.

## 8. Risks

- L1 clash on cores 5..9 with something resident there: loud at program creation.
- The helper's remote write against the owner's `O` ring: the ring has a fixed L1 address and no other user, and the owner's zeroing touches only faces 2 and 3 (words 512 to 1023) of each tile, which the helper's tiles hold as zeros too, so either order leaves the same bytes; the ordering to confirm is that the owner never pops or reuses `O` before the semaphore, and that the semaphore is re-initialised each launch (U until a card shows it).
- DST accumulation at LVt = 2: same ops, but a different loop trip count; P0 decides.
- The coalescing of descriptors (one per role) with per-core runtime arguments of different lengths: owner and helper must be given arguments of one length.
