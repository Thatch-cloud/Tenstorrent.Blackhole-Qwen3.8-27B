# sdpa_prefill_oneq: ONE q chunk per core in the chunked prefill SDPA (prefill lever S1)

The prefill device profile (run 38051000905, 131,072-token prompt, TP4, 2,048-row chunks) found the chunked prefill SDPA to be the only cost
that grows with context: 5.05 + 3.384 ms per 1k tokens of context per chunk, 442 of 707 ms at 126k, and the factory logs
`[QWEN-SDPA-PF] flags=0x3 kv_chain=1 chains=8 members=48`: **48 of the grid's 130 cores do the work.** This directory gives each q chunk its own
core (96 busy cores, 16 chains of 6) with the compute, writer and chain-reader kernels untouched.

## Where the pairing and the chain grouping are decided

Pinned tt-metal 9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9, fetched with
`gh api -H "Accept: application/vnd.github.raw" "repos/tenstorrent/tt-metal/contents/<path>?ref=9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9"`:

| What | Where (pinned file, line) | Effect at TP4 (6 Q heads, 1 KV head, 2048 rows, q chunk 128) |
|---|---|---|
| the work split | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp` 389-410: `global_q_pair_distribute = is_causal && (q_num_chunks % 2 == 0)`; pairs: `base = (total_pairs / num_cores) * 2`, `extra_cores = total_pairs % num_cores`, `extra = 2` | 96 chunks = 48 pairs: cores 0..47 get **2 consecutive flat positions**, the other 82 get none |
| the per-core range | same file 1378-1397 (reader / writer / compute runtime args `global_q_start`, `global_q_count`) | `[2i, 2i + 2)` |
| the pairing itself | `.../sdpa/device/kernels/q_chunk_remapping.hpp` `linear_to_zigzag` (30), called by `decompose_global_q_index` (81) in `kernels/dataflow/reader_interleaved.cpp` 296, `writer_interleaved.cpp` 151 and `kernels/compute/compute_common.hpp` 1612 (`sdpa_inner_loop`) with `use_zigzag_balancing = is_causal` (factory 527, CT arg 33 / 24 / 33) | position 2k -> q chunk k (light), position 2k+1 -> q chunk N-1-k (heavy): `2K+17` blocks per core |
| the Q double buffer | factory 410, 438: `q_buffer_factor = max_global_q_chunks_per_core > 1 ? 2 : 1` | one chunk per core -> single-buffered Q (no pipelining is lost: there is no second chunk) |
| the chain grouping (this repo) | `../sdpa_prefill_chain/apply_factory_pf.py` F4: cores are grouped by their **whole unit list** `(batch, kv_head, q_chunk)...`, so only cores that read byte-identical K/V streams share a chain | paired: key `(0, 0, k, 0, 0, N-1-k)` -> 8 chains of 6; oneq: key `(0, 0, q)` -> 16 chains of 6 |

Every per-q-chunk quantity in the kernels (`q_start_tile`, `k_chunk_end = ceil(q_high / Sk_chunk_t)`, the ping-pong buffers, the diagonal mask) is
recomputed for each `q_iter` from the remapped q chunk (`compute_common.hpp` 1604-1659); nothing depends on how many chunks a core has or on the position
of a chunk inside its core's range.

## The change (apply_factory_ps.py, over the PF factory bfab8558)

Flag **0x8** joins the per-call word `SDPAProgramConfig.max_cores_per_head_batch = 0x5EFA0000 | flags` (0x1 chain, 0x2 inj_batch, 0x4 noc_order; 0x8 oneq
needs 0x1). Six hunks, all inside `create_descriptor`, nothing at namespace scope:

| Edit | What |
|---|---|
| PS0 | before the split: decode `qwen_one_q` (tag, chain bit and oneq bit) from the same word and make `global_q_pair_distribute` false for it. The file's own non-pair branch then gives core i the single flat position i (`base = total / cores = 0`, `extra_cores = total`, `extra = 1`) |
| PS1, PS2 | the constant `kQwenPfOneQ = 0x8` and its place in F1's accepted set (a bare 0x8, or any unknown bit, is still refused: `unknown or incomplete flags`) |
| PS3 | the F4 envelope takes `global_q_pair_distribute || (qwen_one_q && q_num_chunks % 2 == 0)`; every other predicate (B == 1, bf8, flexible, lightweight causal, `qk_in0_num_subblocks > 1`, ...) is unchanged |
| PS4 | `TT_FATAL(max_global_q_chunks_per_core == 1, "[QWEN-SDPA-PF] oneq needs one q chunk per core: {} q chunks on {} cores")`: a shape that does not fit the grid is refused, never silently run with two chunks per core |
| PS5 | `[QWEN-SDPA-PF] oneq=1 q_chunks=96 cores=<N> chunks_per_core=1 chains=16 members=96` after the existing flags line |

Sha chain: served fd8c0676 -> PF bfab8558 (apply_factory_pf.py, K64g/K64i/K64j) -> PS 37b9d966 (apply_factory_ps.py). `apply_factory_ps.py` accepts either of
the first two and refuses any other input, anchors on recorded PF line numbers, checks the result sha, inverts, and refuses any addition outside the function.

### Evidence without a card

* `host_exec_oneq.py` cuts the work split, F1, F4 and the reader range loop **out of the patched file text**, compiles them with g++ against stubs and runs them
  for ~100 (heads, KV heads, rows, grid, word) combinations; `scripts/ci/test_sdpa_oneq_factory.py` compares every core's range, chain flags, coordinates, the
  logged lines and the refusals with `oneq_planner.plan()`. The same executable prints, for the paired word on the 13x10 grid, exactly the profile's log line
  (`flags=0x3 kv_chain=1 chains=8 members=48`).
* `stub_compile_oneq.py`: `g++ -fsyntax-only -Wall -Wextra -Werror` of the three blocks; its teeth (a misspelt name in each edit fails).
* The G6 protocol model (`../sdpa_prefill_chain/pf_protocol_model.py`) run on the oneq round lists (one unit of `C + q + 1` rounds per member, a one-buffer Q CB):
  terminates with exact content for C in {0, 1, 15, 16, 33, 64} x q in {0, 3, 7, 15} x seeds; the planted hang still blocks exactly the sink's waits; the
  checker still catches its three protocol mutations. With one unit per core, two members of a group with different q chunks always have different round
  counts, so a mis-grouped chain **deadlocks instead of silently completing** (the paired scheme's equal-length divergence m / m' cannot occur).

## Exactness

By construction: the compute kernel runs, per q chunk, the same instruction sequence on the same data in the same k-chunk order (0..`C+q`), with the ping-pong
buffers reset per q chunk; the writer writes the same tiles; the chain reader forwards byte-identical K/V chunks to members whose rounds are equal; the only
changes are *which core* runs a q chunk and the Q CB being single-buffered. `oneq_card_m.py` proves it on the card: every arm (served 0x3, oneq 0xB,
oneq_noc 0xF) must equal the stock path (no program word) bit for bit at every (rows, Q memory, seed, Q variant, chunk_start) case.

## Build the graft (on the rig host; no device; outputs to a new directory)

`build_k64j_oq.sh` is `../subdev_h/build_k64j_lo.sh`'s method (branch tp4/subdev-h) for this change: K64j's own source state in ttbuild (the decode factory at K64j,
the four K64j kernels, the chain reader) with the **prefill factory replaced by PS**, `~/opgraft-K64j` read-only as the base, the result in `~/opgraft-K64j-OQ`
(`OQ_GRAFT` / first argument; an existing directory is refused, nothing existing is overwritten or moved). It publishes only if the result is K64j plus the oneq
edits: the base is the recorded K64j binary (packed_any_admission) with a verifying manifest; K64j alone rebuilt in the same ttbuild is compared with it
(`REPRODUCED`, else a warning); every QWEN string of the base survives (a graft `.so` replaces the whole binary: the K64j decode patches, the tree-scratch
factory, the chain factory and the draft fp32 patch are all in it) and the only new ones are the two oneq literals; the exported symbols are the base's;
`_ttnn.so` and every op directory (the chain reader included) are the base's; the manifest differs in one line. ttbuild is then restored and rebuilt clean.

```
# 1. dry run (reads, prints every command, runs none)
QUAL_CARD=<card M board id> ALLOW_SERVING_CARD=1 OQ_BUILD_DRY_RUN=1 bash optimisation/ttnn-op/sdpa_prefill_oneq/build_k64j_oq.sh
# 2. the build (about three ninja runs of 30-60 s); prints K64J_OQ_TTNNCPP_SHA256=<sha>
QUAL_CARD=<card M board id> ALLOW_SERVING_CARD=1 bash optimisation/ttnn-op/sdpa_prefill_oneq/build_k64j_oq.sh
```
(or through CI: `references/fusion-jobs/WPP/OQ-L0-build-dry.env`, `OQ-L1-build.env`.) The card is context only: the build opens no device.

## Run on card M

```
# 3. first hardware pass under the NoC sanitiser (about 40 min)
QUAL_CARD=<card M board id> ALLOW_SERVING_CARD=1 WATCHER=1 IMAGE=<served image> KOPGRAFT64=$HOME/opgraft-K64j-OQ \
  EXPECT_TTNNCPP_SHA256=<the sha step 2 printed> bash optimisation/ttnn-op/sdpa_prefill_oneq/run_card_m_oq.sh
# 4. the proof and the timing (about 75 min); same command without WATCHER=1
# 5. the estimated-vs-measured table from its report
python3 -B optimisation/ttnn-op/sdpa_prefill_oneq/oneq_report.py ~/kwork64/k64j-oq/card-m/oneq-<stamp>.json --markdown
```
(templates: `references/fusion-jobs/WPP/OQ-W1-watcher.env`, `OQ-Q1-sweep.env`, `OQ-Q2-timing-repeat.env`; read rules and stop rules in its ORDER.txt.) Card M is half of
the serving pair: look at `gh run list` for the qwen-two-p150a-exclusive group first; a hang's printed reset hint resets card M and card A together.

`oneq_card_m.py` on ONE card of the mesh, TP4 shape (6 Q heads, 1 KV head, head dim 256, q/k chunk 128, bf8, HiFi2 with fp32 dest, flexible chunk_start tensor,
4,096-block page table): arms `stock`, `served` (0x5EFA0003), `oneq` (0x5EFA000B), `oneq_noc` (0x5EFA000F); rows 2048 / 1024 / 512; Q in DRAM and L1;
seeds 0, 1; Q normal and peaky; 15 chunk starts 0 .. 251,904 (contexts 2k .. 254k). Also: TP2's head counts (12 Q heads, 2 KV heads) at 1024 and 512 rows,
the refusals (oneq at TP2's 2048 rows, a bare 0x8, a legacy int chunk_start, bf16 K/V, an odd chunk count), the program cache (the oneq word keys its own
program; a chunk_start change adds none), 400 alternating served / oneq / oneq_noc calls against the stock sha, an 11x10 grid on the 13x10 device, the factory
log checked against the planner, and the timing (8 starts, 9 interleaved rounds, one attention layer per sample).

## Estimated vs measured

Model (`oneq_planner.py`): 13.51 us per 128x128 block (card-M K0a, the compute floor) + 90 us per layer; the busiest core decides; the paired call at TP4 has `2K+17`
blocks, oneq `K+16` (K = chunk_start / 128). It reproduces the profile's measured paired layer time at chunks 1, 32, 63 within 0.3 percent (modelled 752.0 /
14153.9 / 27555.8 us, measured 750.4 / 14177.3 / 27604.4 us). Solo-TTFT saving = 16 attention layers x sum over the prompt's chunk starts of (paired - oneq) x
1.1286 (unprofiled wall per device second, profile section 5; an estimate for this lever, to be confirmed by the model-gate A/B).

| prompt | chunks | **E** saving, floor step 13.51 us | **E** saving if the chain step is 16.5 us | **M** card-M saving (oneq_report.py) | **M** model-gate TTFT A/B |
|---|---|---|---|---|---|
| 32k (32,768) | 16 | 0.47 s | 0.35 s | pending | pending |
| 128k (131,072) | 64 | 7.88 s | 6.09 s | pending | pending |
| 254k (253,952) | 124 | 29.80 s | 23.10 s | pending | pending |

Per attention layer at TP4 (**E**, us; **M** pending): chunk_start 0: 319.7 -> 306.2; 2,048: 752.0 -> 522.3; 8,192: 2,048.9 -> 1,170.8; 32,768: 7,236.8 -> 3,764.7;
65,536: 14,153.9 -> 7,223.3; 129,024: 27,555.8 -> 13,924.2; 196,608: 41,822.4 -> 21,057.5; 251,904: 53,495.0 -> 26,893.8. The ratio tends to 0.5 (0.503 at 252k).

**The main unknown** is the chain step with 16 chains: the estimate assumes the K/V forwarding stays at the compute floor as it does with 8 chains at TP4 (slope
0.2115 ms per 1k = 13.51 us per step). Q2 on card M measured 16.5 us per step (22 percent above the floor) for the 16-chain TP2 geometry. If oneq lands there, the
saving is the second column (~77 percent of the best case); the `oneq_noc` arm (chain ordered by NoC distance) and the repeat job show whether placement matters.
`timing_verdict`: OQ-WIN when the median oneq / served ratio at 32k and beyond is at most 0.75, OQ-PARTIAL up to 0.95.

## Other chunk sizes, head counts, grids (what happens, so nothing else regresses)

| Case | Result |
|---|---|
| any call without the oneq bits (stock, the chain 0x1..0x7, tail chunks, the draft SDPA, non-causal) | the same ranges, chains and log as PF (host-exec test); PS0 only adds a decode that is false |
| TP4, 2048 / 1024 / 512 rows (96 / 48 / 24 q chunks) on 13x10 or 11x10 | oneq runs: 16 / 8 / 4 chains of 6, 96 / 48 / 24 busy cores (paired: 48 / 24 / 12) |
| TP2 (12 Q heads, 2 KV heads) at 2048 rows = 192 chunks | **refused** by PS4 with the chunk and core counts (more than 130 or 110 cores); the model hook sees the same arithmetic first and serves the plain chain, logging `fell back reason=...` |
| TP2 at 1024 / 512 rows (96 / 48 chunks) | runs (16 / 8 chains of 6); not a served shape, covered by the card sweep |
| 8 Q heads on 1 KV head at 2048 rows (128 chunks) | fits 13x10 (128 cores), refused on 11x10 |
| odd chunk count (1920 rows) | the PF envelope's even-count predicate is kept: refused like the plain chain is |
| q/k chunk 64 or 256 | the chain envelope (`Sq_chunk_t == Sk_chunk_t`, `qk_in0_num_subblocks > 1`) and the Python opt-in (chunk 128, rows 512 / 1024 / 2048) are unchanged: nothing new is admitted |
| 13x10 versus 11x10 | the factory takes the grid from the program config; the 96 chunks use linear cores 0..95 of either; the card sweep runs both |
| a bare 0x8, or 0x8 with an unknown bit | `unknown or incomplete flags` (the Q1 refusal test's `0x5EFA0008` still refuses) |

## The model hook

`scripts/ci/sdpa_pf_oneq_tp.py`: `QWEN_FAST_SDPA_PF_ONEQ=1` (strict 0/1, default off; needs `QWEN_FAST_SDPA_PF=1`, a production `QWEN_FAST_SDPA_PF_FLAGS`, four-card
serving) replaces the graft's `_qwen_pf_program_config` with a wrapper that calls it and ORs 0x8 into the chain's word where the factory accepts the call;
`QWEN_FAST_SDPA_PF_ONEQ_AUDIT=1` also runs the chain's own program on the first 8 and every 24th engaged call and compares both outputs bit for bit on every
chip. The graft file is not edited and with the flag off nothing is imported. **It cannot engage before two integration steps** (WPP.json notes): the image must
carry the K64j-OQ `_ttnncpp.so`, and `packed_any_admission.check_runtime`, which pins every mapped `_ttnncpp.so` to K64j's sha, must accept it.

## Files

| File | Purpose |
|---|---|
| `apply_factory_ps.py` | PS0-PS5, the recorded shas, the word decoder |
| `oneq_planner.py` | the planner (ranges, units, chains), the block counts, the time model and the estimate table |
| `host_exec_oneq.py`, `stub_compile_oneq.py` | the real patched C++ blocks, run and syntax-checked on the host |
| `build_k64j_oq.sh` | the graft build into `~/opgraft-K64j-OQ` |
| `oneq_card_m.py`, `run_card_m_oq.sh` | the card-M proof and timing and its runner (the K64j graft mounted read-only, holder check, node recheck, watchdog) |
| `oneq_report.py` | the estimated-vs-measured table from a card report |
| `../../../scripts/ci/test_sdpa_oneq_{factory,build,card,tp}.py` | the CPU tests (patch, host-executed C++, planner, protocol model, time model, fake-ttbuild build, fake-ttnn harness and hook) |
