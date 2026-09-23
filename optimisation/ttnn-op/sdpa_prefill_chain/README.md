# sdpa_prefill_chain: the G6 K/V chain for the served chunked prefill SDPA (prefill lever #1)

Built from `sdpa-prefill-share-spec.md` (final spec; sections 2-6 are the build contract).
The build is `~/opgraft-K64g` = K64f + this chain (`_ttnncpp.so` 134bc834, chain reader eecc1166,
patched factory bfab8558), **qualified on card M**: the full Q1 exactness sweep passed, the watcher
pass was clean, and the planted hang was caught with card recovery. Q2 timing is below. The model
opt-in is wired into the m3native gate behind `QWEN_FAST_SDPA_PF=1` (see "Model gate (Q3)"); the
model gate itself has not run yet.

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

At K0's bench config one injector per group at the served cadence already reached the compute
floor, which suggested `0x5EFA0001` (the chain alone). **Q2 overturned that at the model's config.**

**Q2 on card M** (graft K64g, Q in L1, the model's 2080-block page table; baseline slope 0.335 ms per
1k keys, against the model's 0.340), per-step time:

| Arm | Flags | Step (us) | vs baseline | Output |
|---|---|---|---|---|
| baseline | - | 21.4 | 1.000 | - |
| chain | 0x1 | 20.7 | 0.967 | exact |
| **chain_b** | **0x3** | **16.5** | **0.771** (0.770 from the unrounded slopes) | **exact** |
| chain_o | 0x5 | 22.3 | 1.042 | exact |
| chain_bo | 0x7 | 18.9 | 0.883 | exact |

**The production value is `0x5EFA0003`** (the chain plus the injector's read-ahead 32): the fastest
arm, more than 2% faster than 0x1, and 0x7 is not 2% faster than it (it is slower). It is the model
opt-in's default. In the model's config the injector's read-ahead is what pays, unlike at K0's.

Predicted saving per 131k prompt: 1,049,600 steps x 4.9 us = **5.1 s** (5.2 s at the model's own
21.76 us step), so the spec's TTFT A/B acceptance, max(3 s, 0.75 x predicted), is about 3.9 s.

Q2's rule (a), "forwarding and lockstep overhead at most 10%", used to compare the chain's absolute
slope with K0b32's 0.2116. K0b32 came from K0's bench config, whose stock slope was 0.284, not 0.335,
so the two slopes do not compare. The rule now compares each run's ratio to its own baseline:
chain / baseline <= 1.10 x K0b32 / K0 stock = 1.10 x 0.746 = 0.820. 0x3's 0.770 passes it; the old
absolute form would have failed it (0.2578 > 0.2328). `sdpa_prefill_bench.py` implements the ratio
form (`--k0b32-slope` over `--k0-stock-slope`, both from one K0 session) and prints the production
choice in its `M1 Q2:` line.

| Flag | Name | Use |
|---|---|---|
| 0x1 | kv_chain | required (the tag with no flags is refused) |
| 0x2 | inj_batch | the injector's K/V barrier every 32 tiles; **production** (with 0x1: Q2) |
| 0x4 | noc_order | chain order by NoC distance (exhaustive over the 720 orders; refused above 8 members) instead of raster; slower at Q2 (0x5, 0x7) |
| 0x100 | test_mutate | test only: the injector reads the other KV head for k_chunk 0 (M-A) |
| 0x200 | test_hang | test only: the sink resets its flag but withholds its credit on its last round (Q1.3: the sink waits in QWDV, its upstream in QWDC) |

**Which calls opt in.** The opt-in sets the word only for the flexible path, bf8 K/V, q/k chunk 128 and
S in `apply_factory_pf.QUALIFIED_ROWS` = 512, 1024 or 2048: exactly the row counts the card-M Q1 sweep
runs. The CPU tests keep the lists (the graft's, the factory's and the card test's `ROWS`) equal. Any
other S (a 256-1792-row tail chunk, or S > 2048 with its unequal groups) takes the served path.

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
| `pf_optin.py` | The Python opt-in of spec 3.6, re-exported from its one copy, `scripts/ci/lever_n_m3native_patch.py` section I (what the gate grafts): `TP_HELPERS`, `patch_tp()` / `unpatch_tp()` / `patch_tp_full()`, and a CLI (`--full` gives the gate's whole `attention/tp.py` graft). |
| `build_k64g.sh` | The rig build of `~/opgraft-K64g` (spec 5.2). |
| `run_card_m_pf.sh` | Card M: `reference`, `candidate` (optionally with `WATCHER=1`) and `hang`. |
| `test_sdpa_prefill_chain_card_m.py` | Q1 on card M: the matrix, M-A, cache/hash, refusals, stress, trace, and the planted hang. |
| `test_sdpa_prefill_chain_sources.py` | CPU tests (spec 5.3 items 1-7 and 9, the card-M helpers, a fake-ttnn dry run of all three roles, and the stub g++ check). |
| `stubcheck/` | `stub_compile.py` plus syntax stubs of the dataflow API for `g++ -fsyntax-only`. |
| `fixtures/` | `sdpa_program_factory.fd8c0676.cpp` (the served factory, for the CPU tests). `cardm_worker_coords.json` goes here too: it was captured in the K0 session (`results/k0-<stamp>/cardm_worker_coords-<stamp>.json` on the rig) and **is not yet copied**. |
| `../sdpa_prefill_bench/` | `sdpa_prefill_bench.py`: chain arms, `--program-word`, `--q-memory`, `--page-blocks`, `--verify-log`, `--k0b32-slope` / `--k0-stock-slope` and `M1 Q2:` (rules a-c plus d: every chain arm's output equals the baseline's, so Q2 needs `--sha`; rule (a) compares ratios to each run's own baseline; the line names the production choice). `run_m1.sh`: `KOPGRAFT_PF` (only with `IMAGE` set explicitly to an image `build_k64g.sh` compared, and then `M1_REQUIRE_SOURCES=1`), `WATCHER=1` and `QWEN_SDPA_PF_TEST=1`. `test_prefill_chain_bench.py` tests both. |

## Deviations from the spec

1. **Production flags: the spec's 0x3, after all.**
   - The build first chose 0x1 from K0. Q2, at the model's config, measured 0x3 fastest (the table
     above), so the model opt-in's default is 0x3 again.
   - Still 0x1 from that choice: the card-M test's stress / trace / cache flag (qualification history;
     the Q1 sweep covered 0x3 too).
   - The bench's choice rule (`q2_production`) is spec 6.3's, generalised to every timed flag set: a
     flag set replaces one with fewer flag bits only with a slope at least 2% lower. That is 0x1 unless
     another set is 2% faster, and 0x7 over 0x3 only if 0x7 is 2% faster than 0x3 (the earlier
     "fastest of those beating 0x1" would have taken a 1%-faster 0x7). One bench run times one Q
     placement; spec 6.3 asks for the same answer in both. Q2's recorded choice is the L1 (acceptance)
     run's.
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
7. **The Python opt-in is wired, under different names than the spec's.**
   - The flag is `QWEN_FAST_SDPA_PF=1` (flags `QWEN_FAST_SDPA_PF_FLAGS`, default 0x3), not the spec's
     `QWEN_SDPA_PREFILL_KVCHAIN(_FLAGS)`: the `QWEN_FAST_*` family the arm translates `M3NATIVE_*` into.
   - The opt-in lives in `scripts/ci/lever_n_m3native_patch.py` section I and nowhere else
     (`pf_optin` re-exports it). It composes with the decode graft: `PATCHES['attention/tp.py']` is
     `patch_attention_tp_full` = `patch_attention_tp`, then `patch_attention_tp_sdpa_pf`.
   - It is an insertion AFTER the served `SDPAProgramConfig` statement (kept verbatim) rather than the
     spec's rewrite of it, so flag off runs the served statement itself.
   - `tp_common.py` is sha-pinned on the serving path and is not touched.
   - The graft's source is the image's own `attention/tp.py` (sha256 e0c685a4, the file every m3native
     graft through v143 staged, and v144 on image A5 126b30df too: run 35917834242's graft artifact
     `attention/tp.py.orig`), committed as `scripts/ci/fixtures/qwen36_attention_tp.py`. On that v144
     file the composed patch gives section I plus exactly v144's staged decode graft.
   - None of this is baked into an image. The graft is staged host-side from the checkout and mounted;
     `lever_n_m3native_patch.py` is in neither image copy list, and must stay importable alone (two rig
     harnesses copy it by itself), so it keeps its own copies of the factory constants and the CPU
     tests hold them equal to `apply_factory_pf.py`.
   - The helpers carry their own `/proc/self/maps` marker check, because the image has no copy of
     `pooled_attention_replay`.
8. **Images.** Card M and `run_card_m_pf.sh` use A'' eceb2daa (gate v128-v137).
   - `build_k64g.sh` step 7 compared the graft's `sdpa/` against A'', A' (the K0 image) and e41ef884
     (the arm script's default).
   - The gate's best config has since moved (v144-v148: A5 126b30df), so the arm no longer relies on
     that list: under `M3NATIVE_SDPA_PF=1` it runs step 7's comparison itself against the run's own
     image before launching (see "Model gate (Q3)").
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
| `py -3.11 -B -m pytest -q test_sdpa_prefill_chain_sources.py` | 91 passed, 1 skipped (the coordinate fixture), in about 30 s |
| `py -3.11 -B -m pytest -q ../sdpa_prefill_bench/test_prefill_chain_bench.py` | 21 passed |
| `py -3.11 -B -m pytest -q test_lever_n_m3native_sdpa_pf.py` (in `scripts/ci`) | 31 passed: the model-gate wiring (graft, arm, gate) |

CI-style runs, from the repo root with `PYTHONPATH=scripts/ci`:

- `unittest discover` of this directory: 92 tests OK (1 skipped). With `QWEN_SDPA_PREFILL_SRC=/nonexistent`
  (no probe tree, as in CI): 92 OK, 4 skipped - the factory tests run from the committed fixture; only the
  probe-tree reader compare, the two stub reader compiles and the coordinate fixture skip.
- `unittest discover` of `../sdpa_prefill_bench`: 106 OK.
- `unittest test_lever_n_m3native_sdpa_pf`: 31 OK, including the whole arm run by bash against a stub
  docker (which also plays the image for the `sdpa/` comparison).

CI runs this directory and the bench through discover lines in `.github/workflows/qwen-integration-cpu.yml`,
and `test_lever_n_m3native_sdpa_pf` through its own line.

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

## Model gate (Q3)

### Wiring (done; CPU-tested by `scripts/ci/test_lever_n_m3native_sdpa_pf.py`)

- **Graft.** `lever_n_m3native_patch.PATCHES['attention/tp.py']` is `patch_attention_tp_full`: the
  decode graft, then the opt-in (section I). The gate workflow's graft job stages it on every tag. Without
  the flag it is inert: the grafted `forward_prefill_paged` makes exactly the decode graft's ttnn calls,
  and the launched `docker run` argv is the arm's without this lever. The grafted *file* is not the
  decode-only graft, though: section I is in every tag's `attention/tp.py`, so every `graft.sha256`
  changes, and if the image's `tp.py` ever drifts at section I's two anchors, every tag's graft job
  fails (loudly, before any hardware), not only tags that set the flag. This follows section H's
  precedent.
- **Arm.** `M3NATIVE_SDPA_PF=1` needs `KOPGRAFT64` to be a K64g graft. `M3NATIVE_SDPA_PF_FLAGS` is optional
  (0x1, 0x3, 0x5 or 0x7; default 0x3) and refused without `M3NATIVE_SDPA_PF=1` (it would pass nothing,
  and an intended "on" arm would measure the served prefill). Before the run, the arm refuses a graft:
  - that lacks `sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp`;
  - whose `_ttnncpp.so` lacks `[QWEN-SDPA-PF] flags=`;
  - whose `MANIFEST.sha256` is missing or does not verify;
  - whose `sdpa/` is not the run image's `sdpa/` plus the chain reader and nothing else. This is
    `build_k64g.sh` step 7's test (`docker create` the image, `docker cp` its op directory out,
    `diff -rq`), run against the image the tag actually launches. It matters because every kernel the
    JIT builds from `sdpa/` then comes from the graft: the served prefill reader and writer, the draft
    SDPA, and the decode kernel's include of `compute_common.hpp`. The attach pins do not cover
    `dataflow_common.hpp` or `chain_link.hpp`.

  It mounts the graft's `sdpa/` over the image's op directory (a single directory, never anything
  under `/experiment-scripts/ci`). The JIT cache is keyed on the whole mounted tree (every file's
  sha256), because the chain reader includes headers from it.
- **Gate.** `required_flag_markers` requires, under the flag:
  - `[PINDIAG] sdpa prefill kvchain flags=0x5efa0003 `: the graft, once per process, after its binary check;
  - `[QWEN-SDPA-PF] flags=0x3 kv_chain=1 chains=16 members=96 `: the factory, once per chain program,
    for the 2048-row topology (spec 6.4). It is required whenever a prompt holds a 2048-token chunk,
    so a run where only 512- or 1024-row tail chunks took the chain does not pass.

  Every factory line must also carry the requested flags and one of the Q1-qualified topologies
  (512 rows 4/24, 1024 rows 8/48, 2048 rows 16/96). Neither the Python eligibility test nor the
  factory's envelope pins the grid or the local head counts, so this is the runtime proof of the
  topology. Without the flag, any `[QWEN-SDPA-PF] flags=` line fails the run, because the off arm of
  an A/B must be the served prefill.

### A model-gate run

Add to the tag's env line in the workflow's arm step, on top of the best config (for example v146's
list, with `KOPGRAFT64` changed from K64f to K64g; K64g carries K64f's decode side):

```
KOPGRAFT64=/home/thatch/opgraft-K64g M3NATIVE_SDPA_PF=1        # M3NATIVE_SDPA_PF_FLAGS=0x3 is the default
```

The launched `docker run` argv then differs from the same arm without `M3NATIVE_SDPA_PF` by exactly:

```
-v /home/thatch/opgraft-K64g/sdpa:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa:ro   (after the sdpa_decode mount)
-e QWEN_FAST_SDPA_PF=1 -e QWEN_FAST_SDPA_PF_FLAGS=0x3                                         (before QWEN_FAST_NATIVE_ATTN)
-e TT_METAL_CACHE=/experiment-cache/kernels-qwen-<K64f's 12>-pf-<12 of the sdpa/ tree's hash>   (every file's sha256)
```

The arm's log then shows `... = image's sdpa/ + reader_interleaved_qwen_chain.cpp; manifest verified`
and the `SDPA prefill K/V chain (lever #1)` line naming the flags and the cache.

**The image.** Any image works whose `sdpa/` equals the graft's minus the chain reader; the arm checks
this against the tag's own image before every flag-on run, and refuses with the `diff -rq` output
otherwise. The v144-v148 best config runs A5 126b30df, which `build_k64g.sh` step 7 never compared, so
the first flag-on tag on A5 is also that comparison. If it refuses, the graft's `sdpa/` and A5's differ:
build a K64g whose `sdpa/` is A5's plus the chain reader (ttbuild's tree synced to A5, `K64G_IMAGES`
naming A5), or tag on an image the check passes for (A'' eceb2daa passed step 7).

Gates, in order (spec 6.4):

1. 32k token-exact against the v95 reference.
2. 131k token-exact against the v96 reference.
3. TTFT A/B at 131k on the same graft, with the flag on and off: 3 alternating runs each. Accept a gain
   of at least max(3 s, 0.75 x 5.1 s) = 3.9 s.
   - Run one discarded priming run with `M3NATIVE_SDPA_PF=1` first. The flag-on arm has its own JIT
     cache (`-pf-<tree>`), so its first run compiles every kernel in the model, while the flag-off
     arm reuses K64f's warm cache. The cache lives in the `qwen-experiments` volume, so one priming
     run warms it for the A/B.
   - Compare the medians of the three runs on each side, because rig CI load skews single runs.
4. 4 x 131k concurrent, token-exact, with the flag on.

**Still not done:** copying the coordinate fixture into `fixtures/`. It is needed only by the 0x4
(NoC-order) CPU test, not by the production flags.
