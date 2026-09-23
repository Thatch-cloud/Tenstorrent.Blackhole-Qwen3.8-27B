# sdpa_decode_qwen: stage 1 of the one-pass decode SDPA (tail-only mask)

This directory holds the first stage of `sdpa-onepass-spec.md` (Angle C, staged, with
Angle B's per-call opt-in). It is a graft of tt-metal's
`paged_scaled_dot_product_attention_decode` that the replay attention reader can opt into
one call at a time. Legacy calls are unchanged.

## What it changes and why

Each packed verify at 4 × 131k spends about 128 ms per round in decode SDPA. Of the 706 MB
each chip reads per user-layer, 134 MB is the provided mask: every core reads two mask tile
rows for every key chunk. The replay mask is `+0.0` everywhere except the last 256 columns,
because the pinned refresh kernel (`attention_mask_replay.cpp`) writes only the last eight
column tiles of a zero-initialised mask. Stage 1 stops reading and adding the mask on every
chunk except each head's final one.

| Edit | File | What |
|---|---|---|
| F1 | factory | `qwen_mode` / `qwen_flags` from `SDPAProgramConfig.q_chunk_size` (`0x51DEC000 \| flags`). The decode path never reads this field, and the default program hash covers it. |
| F2 | factory | Preconditions (TT_FATAL). Flag `0x2` (KV share, stage 3) is refused in this build. |
| F3 | factory | Forces compact tree scratch in qwen mode. This removes only slots that are never used. |
| F4 | factory | One `log_info` line per program: `[QWEN-SDPA] flags=… cb_bytes=…`. |
| F5, F6 | factory | Four suffix reader compile-time args (`mask_tail`, `mask_width_t`, `kv_share`, `kv_ready` semaphore id). They go after every accessor block, so the legacy offsets do not move. |
| F7 | factory | Compute compile-time arg 32 (`mask_tail`). |
| F8 | factory | Selects `reader_decode_qwen.cpp` / `sdpa_flash_decode_qwen.cpp` in qwen mode. |
| R1–R3 | `reader_decode_qwen.cpp` | Reads the suffix args. The mask stride becomes the mask's own width. In tail mode it reads one mask chunk (the last), on `k_chunk == k_num_chunks - 1` only. |
| C1–C2 | `sdpa_flash_decode_qwen.cpp` | Applies the mask under the same predicate. |

- **Legacy calls.** Every factory branch is gated on `qwen_mode`. The original kernels are untouched.
- **Exactness.** This holds by argument (section 4.8), not by construction. The skipped work was `qk + (+0.0)`. The card-M byte comparison is what proves it.
- **Refused for now.** `narrow` (stage 1b) and `share` (stage 3). The Python refuses both names, and the factory refuses flag `0x2`.

## Files

| File | Purpose |
|---|---|
| `make_qwen_kernels.py` | Builds the two kernels from the served originals. The input is either the probe dump or the files themselves. The base shas (49a05926, d24769bd) are enforced, every anchor must occur exactly once, and the output shas are recorded. `--check DIR` verifies a directory. |
| `reader_decode_qwen.cpp`, `sdpa_flash_decode_qwen.cpp` | The generated kernels: the originals byte for byte plus the edits. They hash to 55d8fe5e… and 8776fcc7…. |
| `apply_factory_qwen.py` | Applies F1–F8 to the 3e0a69af factory and refuses any other input. The output is 1b54abd3fe466a05…. It inverts back to the input, keeps `.orig-3e0a69af` when patching in place, and exits 0 if the file is already patched. |
| `build_k64e.sh` | The rig build that produces `~/opgraft-K64e`. Details below. |
| `run_card_m.sh` | Runs the card-M test in the serving image, in `reference` or `candidate` mode. |
| `test_sdpa_decode_qwen_card_m.py` | The card-M unit test. Details below. |
| `test_sdpa_decode_qwen_sources.py` | CPU checks: the edits invert to the bases, the factory patch reproduces its recorded sha, every constant that crosses a file boundary agrees, and the card-M host helpers work. |

Python-side wiring lives outside this directory:

- **`scripts/ci/pooled_attention_replay.py`**
  - Adds `sdpa_modes`, `mode_flags`, `loaded_binary_has_modes` and `apply_sdpa_modes`.
  - The pooled reader applies the modes at construction, after staging its pages. The packed reader applies them to every user's reader. Both happen before any trace capture.
  - Unset, nothing runs.
  - Markers: `[PINDIAG] sdpa qwen-modes binary …` (once per process) and `[PINDIAG] sdpa qwen-modes modes=tail …` (once per reader).
- **`scripts/ci/lever_n_m3native_run_arm.sh`**
  - Mounts `$KOPGRAFT64/sdpa_decode` over the container's op directory, but only if the graft has one. K64c and K64d do not, so their runs are unchanged.
  - That graft gets its own kernel cache: `/experiment-cache/kernels-qwen-<sha12 of the two kernels>`.
  - `M3NATIVE_SDPA_MODES=tail` becomes `-e QWEN_FAST_SDPA_MODES=tail`, plus a mount of this checkout's `pooled_attention_replay.py` over the baked copy.
- **`scripts/ci/lever_n_m3native_gate.py`**
  - With `tail` in `QWEN_FAST_SDPA_MODES`, the run fails unless `server.log` has both `[PINDIAG] sdpa qwen-modes` lines and the factory's own `[QWEN-SDPA] flags=0x1 ` line.

## Build (rig)

1. Ship this directory to `~/kwork64/k64e/`, with LF line endings. For anything over about 30 KB, use the base64 tar over stdin pattern.
2. Run `bash ~/kwork64/k64e/build_k64e.sh`. It:
   - checks the shipped kernels' shas, and that K64d is `06865d8e…` and carries the tree-scratch factory;
   - checks that ttbuild's prefill factory is the combined `fd8c0676`, that its decode kernels are the served originals, and that its attn_prep and nlp_concat_heads_decode equal K64d's;
   - regenerates the qwen kernels from ttbuild's own originals (`make_qwen_kernels.py --check`);
   - backs up and patches ttbuild's decode factory, stages the two kernels, and runs `ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so`;
   - assembles `~/opgraft-K64e.partial`: the contents of K64d, the new `.so` files, and a copy of ttbuild's whole `sdpa_decode` directory. It drops backup files from that copy and puts its factory `.cpp` back to the audited 3e0a69af, because the compiled factory lives in the `.so` and `sdpa_tree_scratch.audit(patched=True)` hashes the file on disk;
   - verifies the result:
     - `strings` shows `[QWEN-SDPA] flags=`, `QWEN_SDPA_TREE_SCRATCH_ROUNDS`, `reader_decode_qwen.cpp` and `qwen_draft_fp32_intermediates`;
     - the op dirs are identical to K64d's;
     - every audited kernel sha matches;
     - `diff -rq` against the serving image's own `sdpa_decode` directory shows **only** the two new kernels;
   - moves the result to `~/opgraft-K64e`, writes a `MANIFEST.sha256`, and prints `K64E_TTNNCPP_SHA256=…`;
   - restores ttbuild: factory 3e0a69af (touched, because `docker cp` keeps the saved copy's old mtime and ninja would otherwise call the unity TU up to date), qwen kernel files removed, then reruns ninja so `build_Release` is rebuilt from the restored factory and checks its `_ttnncpp.so` has no `[QWEN-SDPA] flags=` and still has the tree scratch. On failure the EXIT trap restores and touches the sources but does not rebuild; it warns that `build_Release` still holds the qwen objects until the next ninja run. `K64E_KEEP_STAGED=1` keeps them.

## Test

On CPU, from this directory:

```
py -3.11 -B -m unittest test_sdpa_decode_qwen_sources
```

Under `scripts/ci`:

```
py -3.11 -B -m unittest test_pooled_attention_replay test_lever_n_m3native_gate test_m3native_arm_env
```

On card M:

```
bash run_card_m.sh reference    # stock image: legacy calls only, records every output's sha256
bash run_card_m.sh candidate    # K64e mounted as the arm mounts it
```

`test_sdpa_decode_qwen_card_m.py` runs on one p150a:

- **Inputs.** Folded Q `(1, B, 48, 256)` with 2 KV heads per chip. A batch-3 bundle (groups 0, 4, 8) and a batch-1 bundle (group 12) share one random page table. The bf8 pool is `capacity // 64 + 64` blocks. The mask is full width and zero, except for the refresh formula in its last 256 columns.
- **Sweep.** Capacities 33,024 and 131,328, seeds 0–4, query variants `normal`, `peaky` and `zeroq`, and block starts +0, +7 and +240.
- **Byte comparisons.**
  - Tail against legacy, for every case.
  - A planted `-inf` in mask chunk 0 must change the legacy output and must not change the tail output (liveness).
  - The narrow tail against the wide tail.
  - Legacy outputs against the reference run's shas.
- **Refusals.** Five TT_FATALs.
- **Program cache.** 50 alternating legacy/tail pairs.
- **Log check.** The factory lines must show `scratch_slots=4` and the spec's `cb_bytes` (790,592 at 33k, 796,736 at 131k).
- **Timing.** Median synchronised host timing, legacy against tail, per bundle and per user-layer.
- **Output.** `<out>.json` and `<out>.json.native.log`, which holds all C++ output.

The spec expects tail to cut B1 by at least 8% at 33,024 and at least 15% at 131,328. The model predicts −11% and −19%.

## Stage plan

| Stage | Change | Status |
|---|---|---|
| 0 | Checks before the build: the ttbuild factory sha and the K64c `.so` strings | K64d (tree-scratch factory restored) is built |
| **1** | Tail-only mask, full-width mask, forced compact scratch, per-call sentinel | **This directory**: K64e, `M3NATIVE_SDPA_MODES=tail` |
| 1b | Narrow `(b,1,48,256)` mask. Python only; frees about 269 MB per chip at 4 users | Refused by name for now. The factory already admits it, and card M checks narrow = wide. |
| 2 | 8-row groups (`M3NATIVE_REPLAY_GROUP_ROWS=8`) plus tail | No new code. Needs the card-M 8-row byte compare (E4). |
| 3 | K/V leader multicast to twin entries (F9–F12, R4–R5), flag `0x2` | Refused by name and by TT_FATAL. Needs a second `.so` build. |

**Hardware gate (spec section 9).** Use `KOPGRAFT64=~/opgraft-K64e` for every arm.

- Run the control arm with no modes first, then `M3NATIVE_SDPA_MODES=tail`.
- Read `trace_ms` against the control. The expected value at 4×131k is about 224 ms, with acceptance at ≤ 236 ms.
- The pass criterion is unchanged: all four streams byte-identical to the single-stream references.
