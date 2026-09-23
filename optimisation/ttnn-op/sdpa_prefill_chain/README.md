# sdpa_prefill_chain: the G6 K/V chain for the served chunked prefill SDPA (prefill lever #1)

Built from `sdpa-prefill-share-spec.md` (final spec; sections 2-6 are the build contract).
**Nothing here has run on hardware.** The build is `~/opgraft-K64g` = K64f + this chain.

## What it does

The served prefill SDPA call is causal, flexible-chunked and paged: 12 Q heads, 2 KV heads, 2048 rows,
q/k chunk 128, bf8. On the 11 x 10 grid, core `i < 96` runs Q head `i/8` for chunks `m = i%8` and `15-m`.
The six cores that share a (KV head, m) pair (`i = 48g + 8s + m`) read byte-identical K/V streams. The
chain makes the first of them (the injector) read each K/V chunk from DRAM. It then forwards both CB
slots down a 6-core unicast chain, one credit and one VALID flag per chunk. The five receivers never
read K/V from DRAM, which gives 16 chains of 6.

- **Compute and writer are unchanged**: the same binaries and the same runtime args, so outputs are
  byte-exact by construction.
- **The served default is unchanged.** A call opts in by setting
  `SDPAProgramConfig.max_cores_per_head_batch = 0x5EFA0000 | flags`. The prefill factory never reads
  that field, and its default of 16 never carries the tag.

**K0 on card M** (image A' 1b9b6445, stock slope 0.2838 / 0.2839 ms per 1k keys):

| Probe | Slope (ms per 1k keys) | Notes |
|---|---|---|
| K0a (the compute floor) | 0.2111 | t_c = 13.51 us per step |
| K0b4 (the 16 injectors alone, served read-ahead 4) | 0.2114 | |
| K0b32 | 0.2116 | |
| K0c (every core at read-ahead 32) | 0.3798 | slower; output exact |

One injector per group at the served cadence already reaches the compute floor. **The production
value is therefore `0x5EFA0001` (the chain alone).** Flag 0x2 stays as an A/B knob only (see Deviations).
Expected saving is about 4.9-5.8 s per 131k prompt, because the controlled stock step is 18.2 us, not
the spec table's 21.76 us.

| Flag | Name | Use |
|---|---|---|
| 0x1 | kv_chain | required (the tag with no flags is refused) |
| 0x2 | inj_batch | the injector's K/V barrier every 32 tiles; A/B only |
| 0x4 | noc_order | chain order by NoC distance (exhaustive over the 720 orders; refused above 8 members) instead of raster; A/B at Q2 |
| 0x100 | test_mutate | test only: the injector reads the other KV head for k_chunk 0 (M-A) |
| 0x200 | test_hang | test only: the sink resets its flag but withholds its credit on its last round (Q1.3: the sink waits in QWDV, its upstream in QWDC) |

**Which calls opt in.** `pf_optin` sets the word only for the flexible path, bf8 K/V, q/k chunk 128 and
S in `apply_factory_pf.QUALIFIED_ROWS` = 512, 1024 or 2048: exactly the row counts the card-M Q1 sweep
runs. The CPU tests keep the two lists and the card test's `ROWS` equal. Any other S (a 256-1792-row tail
chunk, or S > 2048 with its unequal groups) takes the served path.

Test flags are refused unless `QWEN_SDPA_PF_TEST=1` is set in the process that builds the program.
Every other bit is refused, including 0x1000 (reserved for Plan B).

## Edits

### Factory: `apply_factory_pf.py`, fd8c0676 -> `bfab8558d889ad21...`

Anchors are checked at these served lines. Each is unique, and `unpatch()` gives fd8c0676 back.

| Edit | Line | Spec | What |
|---|---|---|---|
| F0 | F:21 | F0 | Adds `<algorithm> <cstdlib> <numeric> <vector>` after `<cmath>`. |
| F1 | F:527 | F1 | Sentinel parse after `use_zigzag_balancing`. Two TT_FATALs: unknown or incomplete flags, and test flags without `QWEN_SDPA_PF_TEST=1`. |
| F2 | F:586 | F2 | `if (!is_causal \|\| qwen_kv_chain)`: semaphore ids 0/1/2 go into reader CT 29/30/31. |
| F7 | F:832 | F7 | Pushes the flags as reader CT `cb_arg_offset + 8`, after every accessor block and the CB ids. |
| F3 | F:842 | F3 | `if (!is_causal \|\| qwen_kv_chain)`: semaphores INVALID / INVALID / VALID over core_grid. |
| F4 | F:1347 | F4 | Envelope TT_FATAL (the 18 predicates of spec 3.2 plus `qk_in0_num_subblocks > 1`), the G6 grouping, raster or NoC order (a TT_FATAL above 8 members bounds the n! search), roles, and the log line `[QWEN-SDPA-PF] flags=0x1 kv_chain=1 chains=16 members=96 order=raster`. |
| F6 | F:1353 | F6 | Selects `reader_interleaved_qwen_chain.cpp` in chain mode. |
| F5 | F:1424 | F5 | `if (!is_causal \|\| qwen_kv_chain)`: the 14-word runtime chain block, so every core has 27 reader RT words. |

- All new code is function-local.
- The anonymous namespace and the `qwen_draft_fp32_intermediates` block (F:706-714) are byte-identical,
  and the tests check both.
- The base fd8c0676 is committed as `fixtures/sdpa_program_factory.fd8c0676.cpp` (Apache-2.0, as the other
  tt-metal sources in this repo), so the factory tests (base sha, anchors, reproduction of `PF_FACTORY`,
  inversion, the F5 word order, the semaphore ids) run in CI too. The probe tree, when present, must equal it.

### Reader: `make_pf_reader.py`, f97f5490 -> `reader_interleaved_qwen_chain.cpp` `eecc1166a209e61d...`

The served `reader_interleaved.cpp` is not modified.

| Edit | Line | What |
|---|---|---|
| R1 | after R:13 | `namespace qwen_pf`: `wd_wait_credit` / `wd_wait_valid` poll up to 2^28 times, then `WAYPOINT("QWDC"/"QWDV")`, `ASSERT(false)` and a spin that never falls through; `credit_prev` (always reset INVALID, then, unless `withhold`, an atomic increment upstream); `forward_round` (wait credit == 1, reset 0, write the K and V slots, `async_write_barrier`, relay VALID, flush). |
| R2 | after R:97 | `kQwenKvChain` and a static_assert of the causal flexible-chunked paged envelope. |
| R3 | R:150 | Parses the chain block under `!is_causal \|\| kQwenKvChain`. |
| R4 | after R:196 | The flags CT arg at `cb_arg_offset + 8`, with static_asserts: flag 0x1 set; mutation needs NKH >= 2. |
| R5 | after R:209 | `kv_bt`: `k_chunk_tiles` for the injector under 0x2, else the served `barrier_threshold`; `static_assert(use_q_subblock_push)` (every participant Q read is the subblock read behind the atomic barrier). |
| R6 | R:392-399 | `should_forward = participant && !sink`, `should_receive = participant && !injector`. |
| R7 | R:406-709 | One round per k_chunk: capture both slot pointers, then the receiver (reserve both, `credit_prev(..., pf_test_hang && is_sink && last_round)`, wait VALID) or served K read, the Q subblocks (plus an atomic barrier for participants), the receiver or served V read, and `forward_round`. |
| R8 | before the final `}` | For participants: write and atomic barriers, then receiver and sender set back to INVALID. |
| R9 | 13 lines | `[[maybe_unused]]` on every declaration R6/R7 orphan. |

## Files

| File | Purpose |
|---|---|
| `apply_factory_pf.py` | F0-F7, the cross-file constants, and `decode_word()` (F1 in Python). Accepts `--out`, `--record` and in-place with `.orig-fd8c0676`. |
| `make_pf_reader.py`, `reader_interleaved_qwen_chain.cpp` | R1-R9 and the committed output. Modes: `--check DIR`, `--dump`, `--reader FILE`, `--record`. |
| `pf_protocol_model.py` | The F4 topology port, the round lists, and the discrete-event model checker (spec 5.3 items 2-4). Run it directly for the full grid. |
| `pf_optin.py` | The Python opt-in of spec 3.6: `TP_HELPERS`, and `patch_tp()` / `unpatch_tp()` for the grafted `attention/tp.py`. |
| `build_k64g.sh` | The rig build of `~/opgraft-K64g` (spec 5.2). |
| `run_card_m_pf.sh` | Card M: `reference`, `candidate` (optionally with `WATCHER=1`) and `hang`. |
| `test_sdpa_prefill_chain_card_m.py` | Q1 on card M: the matrix, M-A, cache/hash, refusals, stress, trace, and the planted hang. |
| `test_sdpa_prefill_chain_sources.py` | CPU tests (spec 5.3 items 1-7 and 9, the card-M helpers, a fake-ttnn dry run of all three roles, and the stub g++ check). |
| `stubcheck/` | `stub_compile.py` plus syntax stubs of the dataflow API for `g++ -fsyntax-only`. |
| `fixtures/` | `sdpa_program_factory.fd8c0676.cpp` (the served factory, for the CPU tests). `cardm_worker_coords.json` goes here too: it was captured in the K0 session (`results/k0-<stamp>/cardm_worker_coords-<stamp>.json` on the rig) and **is not yet copied**. |
| `../sdpa_prefill_bench/` | `sdpa_prefill_bench.py`: chain arms, `--program-word`, `--q-memory`, `--page-blocks`, `--verify-log`, `--k0b32-slope` and `M1 Q2:` (rules a-c plus d: every chain arm's output equals the baseline's, so Q2 needs `--sha`). `run_m1.sh`: `KOPGRAFT_PF` (only with `IMAGE` set explicitly to an image `build_k64g.sh` compared, and then `M1_REQUIRE_SOURCES=1`), `WATCHER=1` and `QWEN_SDPA_PF_TEST=1`. `test_prefill_chain_bench.py` tests both. |

## Deviations from the spec

1. **Production flags are 0x1, not 0x3.**
   - The reason is the K0 result above.
   - Affected: the `pf_optin` default, the card-M stress/trace/cache flag, the bench's `chain` arm and the Q2 choice rule. A flag set beyond 0x1 is chosen only if its slope is at least 2% lower.
   - A `chain_o` arm (0x5) was added to Q2, and 0x5 is in the Q1 sweep and the watcher pass (flags 0x1, 0x3, 0x5, 0x7), so every flag set Q2 can choose is exactness-qualified.
2. **R2 and R4 anchor on the lines the spec's parentheses name.** These are the zigzag CT line (R:97) and `cb_id_chunk_start_idx_writer` (R:196). The spec's line numbers, R:99 and R:203, point at the lines after them. The effect is the same.
3. **The F7 anchor is at F:832-833, not F:830-831.** It is the same text.
4. **R9 marks 13 orphaned declarations, not 2.** Removing R:406-709 and the role block also removes the only reads of: `use_padded_mask`, `chain_batch`, `chain_head`, `next_core_q_chunks`, `mcast_num_dests`, `sender_wait_count`, `mask_tile_bytes`, `mask_reader`, `k_tile_shape`, `v_tile_shape` and `cb_mask`. The stub check shows the served reader itself has four unused locals the JIT tolerates, so this is belt and braces.
5. **Small hardening changes:**
   - `forward_round` passes `{}` as the source args, the served form, instead of `{.offset_bytes = 0}`.
   - The give-up spin is `for (;;) { (void)*p; }`, a volatile read, not `while (true) {}`, which C++ may assume terminates.
   - F1 requires `QWEN_SDPA_PF_TEST` to equal `"1"`, not merely to be set.
   - F1 reads the field through `static_cast<uint32_t>`.
6. **K0 files stay in `../sdpa_prefill_bench/`** (`make_k0_readers.py`, `k0/`), where K0 ran. They are not duplicated here.
7. **The Python opt-in is delivered but not wired.**
   - `pf_optin.patch_tp` applies spec 3.6 to `attention/tp.py`. It is tested against the pinned tp.py from the probe 35503727180 dump.
   - Staging it into the arm's graft (`lever_n_m3native_patch.stage`) is not done: that is a scripts/ci change, and the image copy lists must be checked first (bundle provenance).
   - The same applies to spec 5.2 step 8, the arm mounting `$KOPGRAFT64/sdpa`.
   - Both are Q3 prerequisites. Card M reaches the graft through `run_m1.sh KOPGRAFT_PF` and `run_card_m_pf.sh`.
   - `pf_optin` carries its own self-contained marker check because the image has no copy of `pooled_attention_replay`.
8. **The model gate now runs on image A'' eceb2daa** (gate v128+, with K64f).
   - `run_card_m_pf.sh` defaults to A'' ("the image the model gate will use").
   - `build_k64g.sh` compares the graft's `sdpa/` against A'', A' (the K0 image) and e41ef884 (the arm script's default).
9. **Q1.7 (cache) runs first in the candidate role, and Q1.8 (refusals) before Q1.4 (M-A).** Once the
   sweep has built both programs, "served then chain adds one entry" cannot be observed. The factory
   checks only on a program-cache miss, so once M-A has built 0x103 the test-flag refusal would hit the
   cache and return an output; a refusal whose program is already cached is reported as untestable.
10. **The model-checker grid runs 40 seeds in CI; the full 2,000 was run locally** (Evidence below). Set `QWEN_PF_MODEL_SEEDS=2000` to run the full grid under the tests.
11. **A spec claim this build corrects.**
    - The claim (spec 4.2's last paragraph): if group members' lengths diverged, the result would be "a hang, not corruption".
    - That holds only when the lengths differ, and even then consumers see foreign chunks before the hang.
    - Members with the **same** number of rounds but different chunks (for example m and m', which both give 2C + 17 rounds) complete silently with wrong data. The model's `wrong_m` variant shows this.
    - Exactness therefore rests on F4 keying groups by the **whole unit list** (tested) and on mutation M-A on hardware.
12. **The planted hang (0x200) withholds only the credit, not the flag reset** (review fix). The first
    reader skipped the whole `credit_prev` call on the sink's last round, so the sink saw its previous
    round's VALID, pushed a stale K/V slot and exited, and only P4 hung (QWDV never exercised). The model
    had reset the flag outside the hang condition and so hid it; it now calls one `credit_prev(j, withhold)`
    step that mirrors the helper, and a `withhold_skips_reset` hang mutation proves the checker catches the
    old behaviour (720 of 720 runs).
13. **Envelope additions beyond spec 3.2:** `qk_in0_num_subblocks > 1` (with the reader's static_assert),
    because only the subblock Q read sits behind the participant's atomic barrier; and a TT_FATAL bounding
    the 0x4 search to 8 members. Neither changes the served shape (q_num_subblocks 2, groups of 6).
14. **The opt-in admits only S in {512, 1024, 2048}**, the row counts Q1 sweeps (see "Which calls opt in").

## Evidence (CPU, this workstation)

**CPU tests.** Both commands run from this directory.

| Command | Result |
|---|---|
| `py -3.11 -B -m pytest -q test_sdpa_prefill_chain_sources.py` | 89 passed, 1 skipped (the coordinate fixture), in about 48 s |
| `py -3.11 -B -m pytest -q ../sdpa_prefill_bench/test_prefill_chain_bench.py` | 16 passed |

CI-style runs, from the repo root with `PYTHONPATH=scripts/ci`:

- `unittest discover` of this directory: 90 tests OK (1 skipped), about 46 s. With
  `QWEN_SDPA_PREFILL_SRC=/nonexistent` (no probe tree, as in CI): 90 OK, 4 skipped - the factory tests now
  run from the committed fixture; only the probe-tree reader compare, the two stub reader compiles and the
  coordinate fixture skip.
- `unittest discover` of `../sdpa_prefill_bench`: 101 OK, which is the existing 85 plus the new 16. One
  existing assertion in `test_k0_readers.py` was updated for the runner's `$timeout_s`.

CI runs this directory through a new discover line in `.github/workflows/qwen-integration-cpu.yml`.

**Full model-checker grid.** `py -3.11 -B pf_protocol_model.py --seeds 2000 --jobs 8` ran 144,000 runs
(2,000 seeds x C {0, 1, 15, 16, 33, 64} x m {0, 3, 7} x the four variants) in 502 s, after the
planted-hang fix (the model's credit step now mirrors `credit_prev`):

- `ok`, `hang`, `wrong_c` and `wrong_m`: **0 failures**. The hang now blocks exactly the sink's VALID wait,
  P4's credit wait and the sink's consumer, with no content error.
- The checker's teeth: `relay_before_ack` is caught in 30/30 seeds, `credit_before_reserve` in 27/30 and
  `no_reset` in 30/30; the hang mutation `withhold_skips_reset` (the first reader's 0x200 behaviour) in
  720/720 runs (40 seeds x 6 C x 3 m).

**Stub g++ check.** `py -3.11 -B stubcheck/stub_compile.py` uses arm-none-eabi-g++ 13.3.1 with
`-std=c++20 -Wall -Wextra -Werror`:

- The F1-F7 blocks compile inside a stub `create_descriptor` whose locals carry the factory's types.
- The served reader and the chain reader both compile against the real probe-v25 `dataflow_common.hpp` and API stubs, with flags 0x1, 0x3, 0x5, 0x7 and 0x303; with `qk_subblock_h = 4` (one Q subblock) the chain reader fails on its static_assert while the served reader compiles.
- The chain reader adds **no** warning beyond the served reader's four unused locals.
- The harness catches a misspelt API call and an unmarked orphan.
- This is evidence only; the first JIT compile on card M is the proof (spec Q-6 WAYPOINT include, Q-12 code size).

## Build and qualify on the rig (card M only)

### Ship

Ship with LF endings, using the base64 tar over stdin pattern for anything over about 30 KB:

- this directory to `~/kwork64/k64g/`;
- `../sdpa_prefill_bench/` to `~/kwork-m1/` (the updated bench and runner).

`~/kwork64/k64f/` must still hold the K64f build inputs: `apply_factory_qwen.py`,
`make_qwen_kernels.py`, `stage3/`, and the saved `sdpa_decode_program_factory.cpp.3e0a69af…`.

### Build

```
bash ~/kwork64/k64g/build_k64g.sh 2>&1 | tee ~/kwork64/k64g/build-$(date +%Y%m%dT%H%M%S).log
# -> ~/opgraft-K64g, prints K64G_TTNNCPP_SHA256=...; ttbuild restored and rebuilt clean
```

### Q1 (exactness), then Q2 (timing), then the planted hang last

Each step must pass before the next. After any exit 3, 124 or 137 (or 1 with `Timeout (` in the log,
the faulthandler backstop):

1. `docker rm -f` the container (the scripts' EXIT trap does it).
2. Reset card M only: `~/.local/bin/tt-smi -r <index>`. The index is tt-smi's BOARD index, **not** the
   `/dev/tenstorrent` number; the scripts print card M's PCI address and a probable index - confirm it in
   `~/.local/bin/tt-smi -ls` and check `gh run list` for the qwen-two-p150a-exclusive group first.
3. Run a passing stock smoke call.

The smoke call:

```
IMAGE=sha256:eceb2daa744c3345368a638488a804f0ffe8b76f6b0680945a47bb527bdb55a9 \
  M1_ARGS="--arms baseline --starts 0,2048 --rounds 1 --warmup 1 --watchdog-s 120" bash ~/kwork-m1/run_m1.sh
```

The qualification sequence:

```
cd ~/kwork64/k64g
bash run_card_m_pf.sh reference                 # Q1.1 stock served shas (image A'', full matrix)
WATCHER=1 bash run_card_m_pf.sh candidate       # Q1.2 first pass: watcher + NoC sanitiser, 0x1/0x3/0x5/0x7
bash run_card_m_pf.sh candidate                 # Q1.4-Q1.9: cache/hash, sweep, refusals, M-A, stress, trace
# Q2: 9 interleaved rounds, Q in L1 (acceptance) then DRAM; stop at the first failure (a hang needs a reset)
for q in l1 dram; do
  IMAGE=sha256:eceb2daa744c3345368a638488a804f0ffe8b76f6b0680945a47bb527bdb55a9 KOPGRAFT_PF=$HOME/opgraft-K64g \
  M1_ARGS="--arms baseline,chain,chain_b,chain_o,chain_bo --starts 0,2048,8192,32768,65536,126976 --rounds 9 \
           --q-memory $q --page-blocks 2080 --verify-log --sha --watchdog-s 120 --no-fallback-scalar" \
  bash ~/kwork-m1/run_m1.sh || break
done                                            # read 'M1 Q2:' (a-d) and 'M1 PF_LOG' lines
bash run_card_m_pf.sh hang                      # Q1.3 LAST: PASS = the planted call never returns; then reset
WATCHER=1 bash run_card_m_pf.sh hang            # after reset + smoke: PASS = QWDV/QWDC (or a chain-reader assert)
```

**Q3 (model gate) prerequisites, not done here:**

- Copy the coordinate fixture into `fixtures/`.
- Stage `pf_optin.patch_tp` into the arm's `graft/attention/tp.py`.
- Mount `$KOPGRAFT64/sdpa` in `lever_n_m3native_run_arm.sh`: check both image copy lists, and that the path is host-side, not baked.
- Key the arm's JIT cache on the chain reader too. `lever_n_m3native_run_arm.sh` (lines 148-149) names
  `kernel_cache` after the two decode kernels' bytes only, and the kernel cache hash is not known to cover
  file contents, so a revised `reader_interleaved_qwen_chain.cpp` at the same path could reuse a stale
  binary: include it in that hash when the graft has `sdpa/`.
- Add the gate markers: `[PINDIAG] sdpa prefill kvchain flags=0x5efa0001` and `[QWEN-SDPA-PF] flags=0x1 kv_chain=1 chains=16 members=96`.
