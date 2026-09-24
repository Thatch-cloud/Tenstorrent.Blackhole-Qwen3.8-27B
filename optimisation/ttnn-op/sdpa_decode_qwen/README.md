# sdpa_decode_qwen: the one-pass decode SDPA, stages 1 and 3

This directory holds stages 1 and 3 of `sdpa-onepass-spec.md` (Angle C, staged, with Angle B's
per-call opt-in). It is a graft of tt-metal's `paged_scaled_dot_product_attention_decode` that
the replay attention reader can opt into one call at a time. Legacy calls are unchanged.

- **Stage 1 (tail-only mask)** is `~/opgraft-K64e`, served now. Its kernels are the two `.cpp`
  files in this directory, and `build_k64e.sh` still reproduces it byte for byte.
- **Stage 3 (K/V leader multicast, "share")** is `~/opgraft-K64f`. It is stage 1 plus F9-F12
  in the factory and R4-R5 in the reader. Its kernels are in `stage3/` and it is built by
  `build_k64f.sh`. It is **built and proven on the TT-Sim simulator only, not yet on hardware**
  (see "Evidence" below).

## What it changes and why

Each packed verify at 4 × 131k spends about 128 ms per round in decode SDPA.

- **Stage 1.** Of the 706 MB each chip reads per user-layer, 134 MB is the provided mask: every
  core reads two mask tile rows for every key chunk. The replay mask is `+0.0` everywhere except
  the last 256 columns, because the pinned refresh kernel (`attention_mask_replay.cpp`) writes
  only the last eight column tiles of a zero-initialised mask. Stage 1 stops reading and adding
  the mask on every chunk except each head's final one.
- **Stage 3.** Every entry of a bundle reads the same page-table row, because the replay reader
  repeats one user's row per bundle. So every entry reads the same K and V bytes. With the share
  flag, entry 0 (the leader) reads each K/V chunk from DRAM, then multicasts both CB slots to
  the other entries (its twins). The twins sit directly below the leader in the same grid
  column and never read K or V themselves. At 8-row groups each T16 user is one batch-2 bundle,
  so this halves the K/V bytes again (spec section 2: 286 MB → 143 MB per chip per user-layer
  at 131k).

| Edit | File | What |
|---|---|---|
| F1 | factory | `qwen_mode` / `qwen_flags` from `SDPAProgramConfig.q_chunk_size` (`0x51DEC000 \| flags`). The decode path never reads this field, and the default program hash covers it. |
| F2 | factory | Preconditions (TT_FATAL). Stage 1 refuses flag `0x2` here. |
| F3 | factory | Forces compact tree scratch in qwen mode. This removes only slots that are never used. |
| F4 | factory | One `log_info` line per program: `[QWEN-SDPA] flags=… kv_share=… cb_bytes=…`. |
| F5, F6 | factory | Four suffix reader compile-time args: `mask_tail`, `mask_width_t`, `kv_share` and the `kv_ready` semaphore id (3). They go after every accessor block, so the legacy offsets do not move. |
| F7 | factory | Compute compile-time arg 32 (`mask_tail`). |
| F8 | factory | Selects `reader_decode_qwen.cpp` / `sdpa_flash_decode_qwen.cpp` in qwen mode. |
| **F9** | factory (stage 3) | Replaces stage 1's `KV share is not in this build` refusal with the twin-band fit check. B bands of `ceil(cores_per_batch / grid.x)` rows are needed: B=2 needs 6 of 10 rows, B=3 needs 9, and B=4 (12) is refused. |
| **F10** | factory (stage 3) | Twin placement. Linear index `i = b * cores_per_batch + p` keeps every role (batch, head, reduce index, tree parameters). Only the coordinate moves, to column `p % grid.x`, row `(p / grid.x) * B + b`. The reducer, output and tree tables follow automatically. |
| **F11** | factory (stage 3) | Creates the READY semaphore (id 3). VALID is the existing `k_mcast` semaphore (id 2), which is unused because qwen mode refuses `q_heads_parallel_factor > 1`. |
| **F12** | factory (stage 3) | Fills the existing K-multicast runtime slots 15-19 (no new slots; the idle vector stays at 20). A leader gets `do_k_mcast` and its twins' NoC column span, and TT_FATALs unless that span is vertically contiguous. A twin gets its leader's NoC coordinate. |
| R1–R3 | reader | Read the suffix args. The mask stride becomes the mask's own width. In tail mode the reader reads one mask chunk (the last), on `k_chunk == k_num_chunks - 1` only. |
| **R4** | reader (stage 3) | Under `if constexpr (kv_share)`, one READY/VALID round per chunk.<br>**Leader:** `read_k`/`read_v` (the legacy DRAM reads), wait for READY == B-1, reset it, multicast the K slot and the V slot, `async_write_barrier`, then set VALID and multicast it.<br>**Twin:** reserve both slots, reset VALID, raise READY on the leader, wait for VALID, then push both slots.<br>Every entry then reads its own mask. The `else` branch is the stage-1 code, re-indented. |
| **R5** | reader (stage 3) | At the end of `kernel_main`, the leader does `async_write_barrier` and a twin does `async_atomic_barrier`, then VALID is reset to 0. Nothing is left in flight on exit (the fused_1d_input lesson). |
| C1–C2 | compute | Apply the mask under the same predicate. Unchanged in stage 3. |

- **Legacy calls.** Every factory branch is gated on `qwen_mode`, and every stage-3 branch on `qwen_kv_share`. That flag is `0x2` inside qwen mode with B > 1. The original kernels are untouched.
- **Exactness.**
  - Stage 1 holds by argument (spec section 4.8): the skipped work was `qk + (+0.0)`.
  - Stage 3 holds by construction relative to the same call without `0x2`. The roles are unchanged, the compute and writer binaries are identical, and each twin's K/V bytes are a copy of the leader's `read_k`/`read_v` on the same page-table row.
  - The byte comparisons are what prove both.
- **Precondition of share.** All rows of the page table must be equal, and every replay writer guarantees this. If it is broken the result is wrong attention, not a hang. Card-M check N1 plants exactly that case.
- **Refused.** `narrow` (stage 1b) is refused by name in the Python.

## Files

| File | Purpose |
|---|---|
| `make_qwen_kernels.py` | Builds the kernels from the served originals (the probe dump or the files themselves). `--stage 1` (the default) gives R1–R3/C1–C2; `--stage 3` adds R4–R5. The base shas (49a05926, d24769bd) are enforced, every anchor must occur exactly once, and each stage's output shas are recorded. `--check DIR` verifies a directory. Stage 3 also checks that reverting R5 and R4 gives the stage-1 reader. |
| `reader_decode_qwen.cpp`, `sdpa_flash_decode_qwen.cpp` | Stage 1 (K64e): 55d8fe5e…, 8776fcc7…. |
| `stage3/reader_decode_qwen.cpp`, `stage3/sdpa_flash_decode_qwen.cpp` | Stage 3 (K64f): **280a847f…**, and 8776fcc7… (the compute kernel is the same file). |
| `apply_factory_qwen.py` | Applies F1–F8 (`--stage 1`, the default, output 1b54abd3…) or F1–F12 (`--stage 3`, output **06167779a979ba1f…**) to the 3e0a69af factory and refuses any other input. Both stages invert to the base, and stage 3 also inverts to stage 1. It keeps `.orig-3e0a69af` when patching in place. |
| `build_k64e.sh` | The rig build of `~/opgraft-K64e` (stage 1). Unchanged. |
| `build_k64f.sh` | The rig build of `~/opgraft-K64f` (stage 3). Details below. |
| `run_card_m.sh` | Runs the card test in the serving image on the qualification card (`QUAL_CARD`, default card B), in `reference` or `candidate` mode, with an optional `WATCHER=1` first pass. |
| `test_sdpa_decode_qwen_card_m.py` | The card-M unit test for both stages. The stage is read off the loaded binary. |
| `test_sdpa_decode_qwen_sources.py` | CPU checks: the edits invert to the bases for both stages, the factory patches reproduce their recorded shas, every constant that crosses a file boundary agrees, the card-M host helpers work, and a full dry run of the card-M flow runs against a fake ttnn with the graft's share semantics. That dry run includes a broken fake whose twins read their own rows, which N1, N4 and N5 must catch. |
| `probe_k1_card_b.py` | The S0.K1 probe (K1 design sections 3, 7, 8): per-call timing of the served K64g decode SDPA for the G8 B2 tail+share call and its K1a proxy, G4 B2 tail+share, with controls, per-chunk slopes, output checksums and one `K1_PROBE` go/no-go line. It imports the card test's host helpers. |
| `run_probe_k1.sh` | Runs the probe on the qualification card with K64g mounted as the arm mounts it (including `sdpa/`, as under `M3NATIVE_SDPA_PF=1`). It checks the graft's binary, kernels and manifest before launching. `WATCHER=1` is the first pass; `PROBE_DRY_RUN=1` prints the argv. |
| `test_probe_k1.py` | CPU checks for the probe: the slopes against the design's card-M figures, the decision table at every edge, the contract with the gate and the arm, the runner's dry run and refusals, and a full dry run on a fake ttnn (DRAM reuse, model clock) whose broken variants the controls must catch. |

Python-side wiring lives outside this directory:

- **`scripts/ci/pooled_attention_replay.py`**
  - `QWEN_FAST_SDPA_MODES` accepts `tail` and `share`. `narrow` is still refused by name.
  - `mode_flags` sets `0x2` only on bundles with more than one entry. At 8-row groups each T16 user is one batch-2 bundle (`0x3` with tail). At 4-row groups the bundles (3, 1) get `0x3, 0x1`, and with `share` alone a single-entry bundle keeps its pinned config.
  - `share` also requires the stage-3 factory in the loaded `_ttnncpp.so`: F9's `[QWEN-SDPA] KV-share twin bands` literal must be present. Against K64e the run is refused before any config changes, instead of hitting a TT_FATAL at the first capture.
  - Markers: `[PINDIAG] sdpa qwen-modes binary … carries the [QWEN-SDPA] branch with KV share` (once per process) and `[PINDIAG] sdpa qwen-modes modes=share,tail … flags=['0x3']` (once per reader).
- **`scripts/ci/lever_n_m3native_gate.py`**
  - The required markers follow the modes.
  - `tail` alone keeps the stage-1 markers exactly.
  - `tail,share` requires the `modes=share,tail ` line and the factory's `[QWEN-SDPA] flags=0x3 ` line: a bundle of twins was built with `kv_share=true`. A run that only ever built tail programs (`flags=0x1`) fails the gate.
  - `share` alone requires `flags=0x2 `.
- **`scripts/ci/lever_n_m3native_run_arm.sh`** needs nothing new. `M3NATIVE_SDPA_MODES=tail,share` becomes `-e QWEN_FAST_SDPA_MODES=tail,share` plus the `pooled_attention_replay.py` mount. With `KOPGRAFT64=~/opgraft-K64f` the graft's `sdpa_decode` directory is mounted, and the kernel cache is keyed by the two kernels' bytes, so it never reuses K64e's reader. For stage 3 proper, also set `M3NATIVE_REPLAY_GROUP_ROWS=8`.

## Build (rig)

### K64e (stage 1): unchanged

1. Ship this directory to `~/kwork64/k64e/`.
2. Run `bash ~/kwork64/k64e/build_k64e.sh`.

### K64f (stage 3)

1. Ship this directory, **including `stage3/`**, to `~/kwork64/k64f/` with LF line endings. For anything over about 30 KB, use the base64 tar over stdin pattern.
2. Run `bash ~/kwork64/k64f/build_k64f.sh`. It follows build_k64e.sh step for step:
   - It checks the shipped `stage3/` kernels' shas, that K64d is `06865d8e…` with the tree-scratch factory, and ttbuild's prefill factory (`fd8c0676`), decode kernels, attn_prep and nlp_concat_heads_decode.
   - It regenerates the stage-3 kernels from ttbuild's own originals (`make_qwen_kernels.py --stage 3 --check stage3`).
   - It backs up ttbuild's decode factory and patches the saved 3e0a69af copy with `apply_factory_qwen.py --stage 3` (06167779…). A ttbuild left on the stage-1 factory by a `K64E_KEEP_STAGED` run is recognised and patched from the saved base, not refused.
   - It stages the stage-3 kernels and runs `ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so`.
   - It assembles `~/opgraft-K64f.partial` from K64d's contents, the new `.so` files, and ttbuild's `sdpa_decode` directory with its factory `.cpp` put back to the audited 3e0a69af.
   - It verifies the result:
     - `strings` shows `[QWEN-SDPA] flags=`, `[QWEN-SDPA] KV-share twin bands`, `QWEN_SDPA_TREE_SCRATCH_ROUNDS` and `reader_decode_qwen.cpp`;
     - `strings` shows **no** `KV share is not in this build`;
     - the `qwen_draft_fp32_intermediates` count equals K64d's;
     - the op dirs equal K64d's;
     - every audited kernel sha matches, including the stage-3 reader;
     - `diff -rq` against the serving image's `sdpa_decode` directory shows only the two qwen kernels.
   - It moves the result to `~/opgraft-K64f`, writes `MANIFEST.sha256`, and prints `K64F_TTNNCPP_SHA256=…`.
   - It restores ttbuild: factory 3e0a69af (touched), qwen kernel files removed or put back, then `build_Release` rebuilt and checked for no `[QWEN-SDPA] flags=` and the tree scratch still present. The EXIT trap restores on failure. `K64F_KEEP_STAGED=1` keeps the staged state.

## Test

On CPU, from this directory:

```
py -3.11 -B -m unittest test_sdpa_decode_qwen_sources
```

Under `scripts/ci`:

```
py -3.11 -B -m unittest test_pooled_attention_replay test_lever_n_m3native_gate test_m3native_arm_env
```

On the qualification card (card B unless `QUAL_CARD` says otherwise; references land in
`~/kwork64/k64f/card-b/`, one directory per board):

```
bash run_card_m.sh reference                      # stock image: legacy calls only, records every output's sha256
WATCHER=1 bash run_card_m.sh candidate            # FIRST pass of K64f: NoC sanitiser, watchdog, 900 s cap
bash run_card_m.sh candidate                      # the full sweep, with timing
```

- **Environment.** Both roles set `QWEN_SDPA_TREE_SCRATCH_ROUNDS=1`, as the arm does. The legacy G8 (PNHt=3) call does not fit L1 with the full tree scratch: 1,827,904 B at 2,304 keys, found on the simulator. The test refuses to run the share section without that variable.
- **`WATCHER=1`.** `TT_METAL_WATCHER=5`, capacities 2,304 and 33,024, seed 0, variants `normal` and `zeroq`, starts +0 and +240, no timing. Each device call gets a 120 s watchdog (`WATCHDOG_S`) that prints `WATCHDOG` and `os._exit(3)`s. The container timeout is 900 s, and the watcher log goes to `$RESULTS/watcher-<stamp>/`.
- **Which card.** Every single-card harness here runs on the **qualification card**: `QUAL_CARD`, a board id under `/dev/tenstorrent/by-id`, default card B (`blackhole-F36F768B9A5CAFA0`, PCIe only). Card M and card A are the serving pair; `QUAL_CARD` may name one only with `ALLOW_SERVING_CARD=1`, which prints a loud warning. The node is resolved by board id at launch, and a run is refused only while a container or a host process can reach *that* card, so a CI gate on the serving pair does not block a card-B run (`scripts/ci/qual_card.sh`, embedded in each runner). Only the m3native gate is scoped to the serving pair: `qwen-card-reset.yml` and most other hardware workflows still act on every node, card B included, so check `gh run list` for the qwen-two-p150a-exclusive group before and during a card-B session. The file names keep `card_m` from when card M was the only bench card.
- **On exit 3, 124 or 137.** The container is removed. Reset **that card only** with the command the script prints, `n=$(readlink -e /dev/tenstorrent/by-id/<board id>) && ~/.local/bin/tt-smi -r "$n"`: it resolves the board id when it is run, since nodes renumber, and runs nothing when the board is gone (an empty argument would reset every board). Confirm the card's row in `~/.local/bin/tt-smi -ls` by the PCI address the script prints. Never a bare number, which tt-smi reads as its own board index (that renumbers too), and never a bare `tt-smi -r`, which resets every board. The script never resets anything itself.

`test_sdpa_decode_qwen_card_m.py`, on one p150a:

- **Inputs.**
  - Q is folded `(1, B, rows*12, 256)` with 2 KV heads per chip.
  - **G4** is a batch-3 bundle (groups 0, 4, 8) and a batch-1 bundle (group 12), PNHt=2.
  - **G8** is the same 16 tokens as one batch-2 bundle (offsets 0, 8), PNHt=3. The G4 queries are folded from the G8 query's tokens.
  - One random page table is repeated to every bundle row. The bf8 pool is `capacity // 64 + 64` blocks. The mask is full width and zero except for the refresh formula in its last 256 columns.
- **Sweep.** Capacities 2,304 (9 chunks: the early-return path), 33,024 and 131,328; seeds 0–4; variants `normal`, `peaky` and `zeroq`; starts +0, +7 and +240.
- **Stage 1 (G4).**
  - Tail equals legacy.
  - The planted chunk-0 `-inf` moves legacy but not tail.
  - Narrow equals wide.
  - Legacy outputs match the reference shas.
  - 50 legacy/tail alternations.
- **Stage 3 (spec section 8).**
  - **E5:** for G8 B=2, G4 B=3 and G4 B=1 (where `0x2` must be a no-op), each of `0x0`, `0x2` share, `0x1` tail, `0x3` tail+share, narrow tail and narrow tail+share equals legacy.
  - **E4** (stage 2's question, recorded, not failed on): G8 unfolded equals G4 unfolded, per token, for legacy and tail.
  - **N1:** page-table rows 1.. are other permutations. Share must equal legacy on the leader's row, and each twin must differ from legacy on its own row.
  - **N2:** a planted `-inf` does not move tail+share.
  - **N3:** the stage's refusals: B=4 twin bands, unknown flag, 512-wide mask under tail, narrow without tail, the sentinel on a causal call. A stage-1 binary refuses `0x2` instead.
  - **N4:** 1,000 calls alternating legacy and tail+share on the distinct page table, where the two modes differ.
  - **N5:** one trace (legacy and tail+share G8, legacy and tail+share G4 B3, legacy and tail G4 B1) replayed 200 times, each replay bit-equal to eager.
  - **Log:** exactly the requested qwen programs have factory lines, with `kv_share` true only for `0x2` with B>1, `scratch_slots=4`, and the spec's `cb_bytes`.
  - **Timing:** the spec's acceptance ratios are recorded. G8+tail B2 must be ≤ 0.6 × legacy (B3+B1) at 131k, and share+tail B3 ≤ 1.6 × tail B1.
- **Output.** `<out>.json` and `<out>.json.native.log`, which holds all C++ output.

## S0.K1 probe (K1 head-sliced Q, no build)

The K1 design (`k1-sdpa-head-slice-design.md`, sections 3, 7 and 8) needs one number before any C++:
whether a head-sliced G8 call stays compute-bound. The **G4 batch-2 tail+share** call does exactly the
head-sliced G8 call's per-core work (2 row tiles, 16 cores per head, 32 leaders reading 142.9 MB, one twin
each), so its time on the served binary is K1a's per-call time.

```
PROBE_DRY_RUN=1 bash run_probe_k1.sh     # the argv, nothing launched
WATCHER=1 bash run_probe_k1.sh           # first pass: NoC sanitiser, checksums at 2,304 and 33,024, no timing
bash run_probe_k1.sh                     # the probe: 2,304 / 33,024 / 66,048 / 131,328 keys
```

- **Shapes.** G8 B2 tail+share (the served call, and the control that re-anchors card B against card
  M), G4 B2 tail+share (the proxy), G4 B2 tail (the DRAM reference), G4 B1 tail (the compute
  reference), and G8 B2 tail (G8's DRAM-bound reference).
- **Per capacity and shape.**
  - One poisoned call: a NaN tensor of the output's shape is freed first, so an unwritten row shows as
    NaN (design N-S0).
  - One legacy call. Both outputs' sha256 are recorded.
  - Eager timing: the median of 20 calls per round, over two rounds that interleave the shapes.
  - A 16-call trace, replayed, per call.
- **Slopes.** Per chunk on the busiest core (1 / 9 / 17 / 33 chunks), two-point and least-squares, with
  the rise the design reads for linearity.
- **Verdict.** Section 3's table, relative to card B's own controls at 131,328 keys:
  - **A:** the proxy is ≤ 1.08 × G4 B1 tail, with a slope within 5%. Enable `0x4`.
  - **C:** the proxy is ≥ 0.93 × the served call. Enable `0x4,0x8`, and also time `0xB`.
  - **B:** anything in between. Enable `0x4,0x8`.
  - **NO-DECISION:** the measurement is invalid, the controls do not separate A from C, or the proxy
    failed the spot check (rows left unwritten, or bytes that differ from legacy: a share protocol that
    is wrong can also be fast). The numbers are still printed.
- **Also printed.** K1a alone against section 7's 0.78 / 0.90 rules; the saving per replay (×64 calls):
  on the eager basis as `(card)`, `(trace-scaled)` (×761/815.8, section 3's scaling) and the measured
  `(trace)`, and on `--basis trace` the measured figure only; and the spot check (G4 B2 tail+share ==
  legacy; B=2 share on G4 was never qualified). `equal-unpoisoned(k/n)` means equal, but k outputs did
  not take the poisoned address. The JSON also records section 3's 4 × 32k cross-check at 33,024 keys.
- **Failures.**
  - The loaded binary is not K64g's (134bc834…), or the kernels are not 280a847f / 8776fcc7.
  - No compact scratch.
  - A requested program without its factory line.
  - A qualified shape that differs from legacy.
  - The watchdog.

Results land in `~/kwork64/k1probe/<card tag>/probe-<stamp>.json` (and `.log`, `.json.native.log`).

## Evidence (stage 3, before hardware)

The **TT-Sim Blackhole simulator** (local WSL `TT-Sim`, tt-metal 9f9cd4fd with the same factory 3e0a69af and the same kernel shas as served) was used as follows:

- A **private** `_ttnncpp.so` was compiled from the stage-3 factory (06167779…) with clang-20: the transformer unity TU was recompiled and relinked in a private copy of `build_Release`. The shared tree and its build were not touched.
- The stage-3 kernels were JIT-compiled through a private `TT_METAL_RUNTIME_ROOT` symlink farm.

| Run | Result |
|---|---|
| reference (stock `.so`), 2,304 keys, legacy only | passed; E4 legacy exact |
| candidate K64f-equivalent, 2,304 keys, seed 0, `normal`, start +0 | **passed**: stage-1 checks; E5 for every mode on G8 B2, G4 B3 and G4 B1; E4 legacy and tail exact; N1 (share on the leader's row, twins differ); N2; all five stage-3 refusals; legacy/tail+share alternation; legacy equal to the stock `.so`; the factory lines exactly the 18 requested programs (`kv_share=true` only for `0x2`/`0x3` with B>1) |
| candidate, 8,448 keys (33 chunks: 2-3 chunks per core, so the handshake repeats and the two-slot CB ring wraps), seed 0, `normal`, start +7, share section only | **passed**: E5 for every mode on G8 B2, G4 B3 and G4 B1; E4 exact; N1 (both G8 and G4 B3, share and tail+share); N2; all five refusals; alternation distinguishable, no drift; 18 factory lines, one per requested program |

No run hung. The simulator runs with slow dispatch, so N5 (trace) and the watcher were not exercised there. They are covered on CPU by the dry run only.

## Stage plan

| Stage | Change | Status |
|---|---|---|
| 0 | Checks before the build: the ttbuild factory sha and the K64c `.so` strings | K64d (tree-scratch factory restored) is built |
| **1** | Tail-only mask, full-width mask, forced compact scratch, per-call sentinel | K64e, served; `M3NATIVE_SDPA_MODES=tail`; card M 180/180 |
| 1b | Narrow `(b,1,48,256)` mask. Python only; frees about 269 MB per chip at 4 users | Refused by name. The factory admits it, and card M checks narrow = wide. |
| 2 | 8-row groups (`M3NATIVE_REPLAY_GROUP_ROWS=8`) plus tail | No new code. E4 is recorded by the card-M run; exact on the simulator at 2,304 keys. |
| **3** | K/V leader multicast to twin entries (F9–F12, R4–R5), flag `0x2` | **K64f**: `build_k64f.sh`; `M3NATIVE_SDPA_MODES=tail,share` + `M3NATIVE_REPLAY_GROUP_ROWS=8`. Simulator-proven; hardware next (WATCHER=1 card M first). |

**Hardware gate (spec section 9).** Use `KOPGRAFT64=~/opgraft-K64f` for every arm. The control and `tail` arms run unchanged on it: the legacy and tail branches are the same code as K64e's.

- Run the control first, then S2 (`tail` + 8-row), then S3 (`tail,share` + 8-row).
- S3 is expected at about 150–162 ms at 4×131k, and is accepted at ≤ S2 − 8 ms.
- The pass criterion is unchanged: all four streams byte-identical to the single-stream references.
