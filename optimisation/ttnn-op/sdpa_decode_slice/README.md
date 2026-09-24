# sdpa_decode_slice: K1, the head-sliced decode SDPA (graft K64i)

This directory holds K1 of the round-fence plan (`k1-sdpa-head-slice-design.md`, reviewed). It is stage 4 of the
[QWEN-SDPA] decode SDPA (stages 1 and 3 are in `../sdpa_decode_qwen`). Two new per-call flags ride in the
`q_chunk_size` sentinel (`0x51DEC000 | flags`):

- **`0x4` q-slice (K1a).** Every core of a KV head computes only the Q row tiles that hold that head's folded
  rows: KV head 0 computes tiles 0-1 and KV head 1 tiles 1-2, instead of all three. The key partition, the tree
  reduction and the K/V bytes are unchanged. K/V is still read from DRAM once per user-layer.
- **`0x8` K/V read-ahead (K1b).** The share leader reads chunk n+1 while chunk n's multicast is in flight. It needs
  `0x2`: the factory refuses it without `0x2`, and like `0x2` it has no effect at B=1.

The S0.K1 probe on card B (2026-09-24, `~/kwork64/k1probe/card-b/probe-20260924T120742.json`) found scenario B,
so both flags are to be enabled:

`K1_PROBE verdict=GO scenario=B plan=K1a+K1b enable=0x4,0x8`

| Probe reading (131,328 keys) | Value |
|---|---|
| proxy | 696.3 µs |
| compute | 563.8 µs |
| served | 796.5 µs |
| proxy / served | 0.874 |
| saving | 100.2 µs per call |

- **Exactness.** Class B: bit-identical accumulation order (design section 4). The compute kernel is reused byte
  for byte. The card-B harness below proves exactness per kept row of both heads.
- **Legacy and served programs.** Flags `0x0`-`0x3` build exactly K64g's programs.

## Edits

### Factory: `apply_factory_slice.py`

The input is the 3e0a69af base or the stage-3 factory 06167779. The output is recorded as
**`1634369677ae0247…`** (`STAGE4_FACTORY`). The edits invert: stage 4 → 3 → 1 → base.

| Edit | What |
|---|---|
| F13 | Adds `kQwenQSlice = 0x4`, `kQwenKvReadahead = 0x8` and the slice kernels' ABI tag `0x51CE`. |
| F14 | Under `0x4` only, `PNHt` becomes the slice. The slice is `max_h(ceil((h+1)G/32) - floor(hG/32))` with G = `num_q_heads / num_kv_heads`. `qwen_pnht_full` keeps Q's own row tiles. Every later `PNHt` use follows automatically: the CBs, the subblocks, compute CTA 3, writer and reader CTA 1, `MUL_BCAST_GRANULARITY` and the F4 line. |
| F15 | Adds `qwen_kv_readahead = 0x8 && qwen_kv_share`. Admits `0x4` and `0x8`, and refuses: `0x8` without `0x2`. Under `0x4` it also refuses: heads that do not divide or a single KV head; a slice that saves no tile; a head the slice does not cover; a mask whose row tiles are not Q's (validate refuses that one first). |
| F16 | Appends the reader suffix `+4 pnht_full, +5 rows_per_kv, +6 kv_readahead, +7 q_slice, +8 tag` under `0x4` **or** `0x8`. Appends the writer suffix `+0 pnht_full, +1 rows_per_kv, +2 tag` under `0x4` only. This is the review's F16/F17 fix. |
| F17 | Selects `reader_decode_qwen_slice.cpp` under `0x4` or `0x8`, and `writer_decode_qwen_slice.cpp` under `0x4` only. `0xB` keeps the stock writer. |
| F18 | One line per stage-4 program: `[QWEN-SDPA] q-slice rows_per_kv=… pnht_full=… slice_tiles=… readahead=…`. On a read-ahead-only program, `slice_tiles` equals `pnht_full`. |

### Reader: `reader_decode_qwen_slice.cpp`

The stage-3 reader 280a847f plus R6-R9. Output **`0f5a019c…`**.

| Edit | What |
|---|---|
| R6 | Reads the five suffix CTAs. Asserts the tag, "no slice ⇒ full PNHt", "the last head's slice fits", and "read-ahead ⇒ share". Adds `q_tile_start = q_slice ? (cur_head_group * rows_per_kv) >> 5 : 0`. |
| R7 | `q_batch_offset = cur_batch * pnht_full * DHt + q_tile_start * DHt`. |
| R8 | `mask_batch_offset = … * pnht_full * mask_width_t + q_tile_start * mask_width_t`. |
| R9 | Behind `kv_readahead`, the read-ahead leader, in this order: a prologue read of the first chunk; wait READY(n); multicast K(n) and V(n); chunk n's mask; read K(n+1) and V(n+1), never past `k_chunk_end`; write barrier; VALID(n). The shared mask read skips this leader. The stage-3 leader, the twins and R5 are unchanged. |

### Writer: `writer_decode_qwen_slice.cpp`

The stock 734c90c0 plus W1-W3. Output **`ac6cf815…`**.

| Edit | What |
|---|---|
| W1 | Reads the three suffix CTAs. Asserts the tag, `rows_per_kv == num_q_heads / num_kv_heads`, DRAM output with one head per core, and that the slice fits. Computes `q_tile_start`. |
| W2 | `out_tile_id = cur_batch * pnht_full * vDHt`. |
| W3 | `write_partial_tiles_sliced`: the stock `write_partial_tiles_to_memory` with the L1 tile rebased to `(head_tile - q_tile_start) * vDHt + d`. A CPU test diffs the two functions line by line. |

## Files

| File | Purpose |
|---|---|
| `apply_factory_slice.py` | F13-F18, as above. |
| `make_slice_kernels.py` | Builds both kernels from the served originals: the probe dump, or `--reader`/`--writer` taken from ttbuild. Checks the base shas, every anchor once, the recorded outputs, the inversion and `--check DIR`. |
| `reader_decode_qwen_slice.cpp`, `writer_decode_qwen_slice.cpp` | The two stage-4 kernels, as generated. |
| `slice_index_model.py` | A pure-Python mirror of the rule and the index math (R7, R8, W2, W3), plus the CB bytes. The fake ttnn is built on it, including its bug variants: `r7`, `r8_row`, `r8_stride`, `w2`, `w3`. |
| `share_protocol_model.py` | Exhaustive state exploration of READY/VALID with the stage-3 and the R9 leader, over B = 2 and 3, 1-6 chunks, three mask kinds and two compute pop orders. It checks for deadlock, a multicast into an unreserved slot, a DRAM read over a slot still in flight, and the wrong chunk at compute. Two broken variants must be caught. |
| `stubcheck/` | `g++ -fsyntax-only` of the served stage-3 reader and stock writer, then the slice kernels at every buildable flag set, and the static_asserts that must fire. Also compiles the F13-F18 blocks in a typed stub. Uses the prefill chain's API stubs plus `include/` here. |
| `build_k64i.sh` | The rig build of `~/opgraft-K64i`, described below. |
| `run_card_b.sh` | The card-B runner, with roles `reference` (on K64g) and `candidate` (on K64i). `WATCHER=1` gives the first pass and `K64I_DRY_RUN=1` prints the argv. It embeds `qual_card.sh`, which is synced and registered in `scripts/ci/test_qual_card.py`. |
| `sdpa_decode_slice_card_b.py` | The card-B harness (design section 7). It imports the stage-3 card test's host helpers and the probe's watchdog, poison and timing. |
| `test_sdpa_decode_slice_sources.py` | CPU tests: sources, factory, contracts, index model, protocol model, build script and stub compile. |
| `test_sdpa_decode_slice_card_b.py` | CPU tests: harness helpers, runner dry runs, and a full dry run on a fake ttnn whose broken variants the controls must catch. |

## Build (rig): `~/opgraft-K64i`

1. **Ship.** Put this directory and its two siblings side by side, with LF endings, under `~/kwork64/k64i/`: `sdpa_decode_slice/`, `sdpa_decode_qwen/` and `sdpa_prefill_chain/`. The tarball is about 0.5 MB without `__pycache__`, so stream it through plink's stdin (see `tt-rig-access`). From the worktree root:

   ```
   tar --force-local --exclude=__pycache__ --exclude=.pytest_cache -cf k64i.tar -C optimisation/ttnn-op sdpa_decode_slice sdpa_decode_qwen sdpa_prefill_chain
   plink -batch -pw thatch thatch@10.10.11.184 "ssh -o BatchMode=yes thatch@192.168.2.165 'mkdir -p ~/kwork64/k64i && cat > ~/kwork64/k64i/k64i.tar'" < k64i.tar
   # then, as an argument-sized script:
   cd ~/kwork64/k64i && tar -xf k64i.tar && ls sdpa_decode_slice/build_k64i.sh
   ```

2. **Preflight.** Every item must be present:
   - `docker ps` lists `ttbuild`;
   - `~/opgraft-K64g/MANIFEST.sha256` verifies, and `_ttnncpp.so` is `134bc834…`;
   - `~/kwork64/k64f/sdpa_decode_program_factory.cpp.3e0a69af9563ae8899db1363286d6dc6cdc1744e0154887dc91a630b16084a4a` exists;
   - `~/kwork64/k64g/sdpa_program_factory.cpp.fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783` exists;
   - the images P3 `70c27e75…`, P2 `463964d5…` and `e41ef884…` are present. Otherwise set `K64I_IMAGES`, keeping P3.

3. **Build.** Run `bash ~/kwork64/k64i/sdpa_decode_slice/build_k64i.sh > ~/kwork64/k64i/build.log 2>&1; echo exit=$?`. It follows `build_k64g.sh` step for step:
   - **0: inputs.** Checks LF endings, the shipped kernel shas and the images.
   - **1: the base.** K64g must carry `sdpa/`, which inverts build_k64g's check. It records K64g's QWEN strings.
   - **2: ttbuild sources.** Checks the decode factory 3e0a69af, the prefill factory fd8c0676 and the stock kernels. It regenerates all three kernel sets from ttbuild's own originals.
   - **3: stage.** Stages the stage-4 decode factory, the prefill factory bfab8558 and the five kernels.
   - **4: ninja.**
   - **5: assemble.** K64g, plus the new `.so` files, plus the two slice kernels in `sdpa_decode/`.
   - **6: restore.** Restores ttbuild, touches the factories, rebuilds, and checks that neither qwen branch remains.
   - **7: verify.**
     - Every QWEN string of K64g and of each image is still in the new `.so`, plus the stage-4 literals.
     - `sdpa_decode/` equals K64g's plus the two slice files, and each image's plus the four qwen kernels.
     - `sdpa/` equals K64g's, and each image's plus the chain reader.
     - It writes `MANIFEST.sha256`.
4. **Record the hash.** `grep -E '^(ok|FAIL|K64I_TTNNCPP_SHA256)' ~/kwork64/k64i/build.log | tail -40` shows the result. Record `K64I_TTNNCPP_SHA256`: the arm passes it as `QWEN_FAST_RUNTIME_BINARY_SHA256`, and the gate pins it per tag.

## Card B: qualification

First check `gh run list` for the qwen-two-p150a-exclusive group. `qwen-card-reset.yml` and most hardware workflows still act on every node, card B included. Then run:

```
cd ~/kwork64/k64i/sdpa_decode_slice
K64I_DRY_RUN=1 bash run_card_b.sh candidate    # the argv only
bash run_card_b.sh reference                  # K64g: legacy + 0x0-0x3 shas, about 40 min (no controls, no timing)
WATCHER=1 bash run_card_b.sh candidate        # FIRST PASS of K64i: NoC sanitiser, 2,304 / 33,024 keys, 30 min cap
bash run_card_b.sh candidate                  # the full qualification and timing, about 90 min
```

Each step must print `### exit 0`, and its `### K1_CARD` line must say `exact=PASS`. Stop at the first failure. The candidate reads the newest `reference-*.json` in `~/kwork64/k64i/card-b/`, and refuses it (exit 2, before the device opens) unless it is a passing reference-role report. A candidate's `K1_CARD` line must end `reference=reference-<stamp>.json`: `reference=none` means no reference was found, and its `exact=PASS` proves only that the binary agrees with itself, not that legacy and `0x0`-`0x3` are K64g's bytes.

**On a hang** (exit 3, 124 or 137, or exit 1 with `Timeout (` in the log), use the printed reset hint for card B only.

**What the candidate checks** (see the harness docstring for the full list):

| Check | What |
|---|---|
| E-S1 | Every flag set equals its non-slice twin and legacy on both heads' rows. The shapes are G8 at B = 2, 3 and 1, and G7 at B=2. `0xB` builds with the stock writer. |
| E-S1b | G16 per token against G8 legacy. Recorded, not failed on. |
| N-S0 | Poison-and-address before every call. |
| N-S1 | A +320 spike planted at rows 40, 56 and 90, one entry at a time, plus a chunk-0 spike. |
| N-S2 | Twins with distinct page-table rows. |
| N-S3 | Refusals. |
| N-S4 | 3 × 1,000 program-cache alternations. |
| N-S5 | A 200-replay trace soak. |
| Log | The F4 and F18 lines of exactly the requested programs, with the modelled `cb_bytes`. |
| Timing | `0x3` against `0x7` / `0xB` / `0xF` at every capacity, eager and traced. |

**Acceptance** (design section 7): the better of `0x7` and `0xF` must be ≤ 0.78 × `0x3` at 131,328 keys. Anything above 0.90 × `0x3` stops K1. For reference, the probe's K1a-alone proxy is at 0.874.

## Tests (CPU, py -3.11)

```
cd optimisation/ttnn-op/sdpa_decode_slice
py -3.11 -B -m unittest test_sdpa_decode_slice_sources test_sdpa_decode_slice_card_b   # about 60 s
py -3.11 -B stubcheck/stub_compile.py                                                   # the syntax check alone
py -3.11 -B share_protocol_model.py --broken                                           # the full protocol sweep (~100 s)
cd ../../../scripts/ci
py -3.11 -B -m unittest test_pooled_attention_replay test_qual_card
```

The dump, the 3e0a69af factory and the prefill tree are found in the job directory, or via `QWEN_SDPA_SOURCES_DUMP`, `QWEN_SDPA_FACTORY_BASE` and `QWEN_SDPA_PREFILL_SRC`. The tests that need them skip without them.

## Model wiring

**Done here.** `scripts/ci/pooled_attention_replay.py` takes the modes `slice` (`0x4`) and `readahead` (`0x8`):
- `slice` applies only to bundles whose groups the factory's rule saves a tile on (`q_slice_saves`: 6-8 rows).
- `readahead` requires `share`, and only bundles with more than one entry get it.
- Both require the stage-4 literal `[QWEN-SDPA] q-slice rows_per_kv=` in the mapped `.so`.
- The PINDIAG modes line keeps its format.
- At 8-row groups, T16 users get `0xF`.
- The arm already mounts this file whenever `M3NATIVE_SDPA_MODES` is set, so no image build is needed.

**Still to do.** These changes are in H1b's files, which this change does not touch:
- **`lever_n_m3native_gate.py`.** Needs `SDPA_MODE_FLAGS['slice'] = 0x4` and `['readahead'] = 0x8`, and a required `[QWEN-SDPA] q-slice rows_per_kv=` marker under either. Until then the gate fails any arm with those modes: it expects `flags=0x3` and the factory prints `0x7` or `0xf`. It fails closed, not silently.
- **`lever_n_m3native_run_arm.sh`.** The kernel-cache key hashes only the two stage-3 kernels. It must hash every `*qwen*.cpp` under the graft's `sdpa_decode/device/kernels`. The presence check must also cover the slice kernels when the modes include `slice` or `readahead`.
- **`.github/workflows/qwen-integration-cpu.yml`.** Must name `test_sdpa_decode_slice_sources` and `test_sdpa_decode_slice_card_b`. Otherwise they never run in CI.

**The ABAB arms** (design section 8):
- **A:** the current best argv with `KOPGRAFT64=~/opgraft-K64i` and `M3NATIVE_SDPA_MODES=tail,share`.
- **B:** A with `tail,share,slice,readahead`.
- Both keep `QWEN_FAST_VERIFY_T2=1`.
- **Markers required in B:** `flags=0xf `, `q-slice rows_per_kv=` and the PINDIAG modes line.

**Stack order.** K64i is built over K64g. Wave 4 #4's graft (K64h in the verify-trace spec) must be built over K64i, or K1 rebased onto it. The QWEN-string diff at step 7 enforces either order.
