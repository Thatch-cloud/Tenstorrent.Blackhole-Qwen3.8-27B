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
(templates: `references/fusion-jobs/WPP/OQ-W1-watcher.env`, `OQ-Q1-sweep.env`, `OQ-Q2-timing-repeat.env`; read rules and stop rules in its ORDER.txt.) The end-to-end number is the solo TTFT pair `OQ-T1..OQ-T4` (four cards: `prefill_ladder_solo` on the production profile and on its `-fx-oneq` twin, ABAB, read with `scripts/ci/prefill_ladder_report.py`; it needs the image with the K64j-OQ binary, see below). Card M is half of
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
| 32k (32,768) | 16 | 0.47 s | 0.35 s | 0.57 s (0.41 s profile-anchored) | pending (OQ-T1..T4) |
| 128k (131,072) | 64 | 7.88 s | 6.09 s | 9.46 s (7.79 s profile-anchored) | pending (OQ-T1..T4) |
| 254k (253,952) | 124 | 29.80 s | 23.10 s | 35.78 s (29.68 s profile-anchored) | pending (OQ-T1..T4) |

**Measured on card M** (W1 run 38066157833: watcher PASS 10/10 exact; Q1 run 38066212228: `ONEQ_CARD_M verdict=PASS cases=360 exact=360 tp2=6 grid=4 failures=0`, binary
`2b81e28f017ccf0ab50028fbae5eb31dd61cfd8a3159233a1ed785712d024a57`). Per attention layer at TP4, 2048 rows, Q in DRAM, served -> oneq (ms): context 4k 0.96 -> 0.69
(0.72x), 34k 8.73 -> 4.58 (0.52x), 67k 17.06 -> 8.73 (0.51x), 131k 33.17 -> 16.80 (0.51x), 199k 50.26 -> 25.30 (0.50x), 254k 64.27 -> 32.34 (0.50x). The `oneq_noc` arm
(chain ordered by NoC distance) measured about 0.68x: worse, dropped (not used by any profile or template). The ratio matches the model; the card-M ABSOLUTE times
(both arms) are 1.20x the profile's in-situ per-layer times (e.g. 64.27 vs 53.5 ms at 252k), which is the 16.5 us chain step of the second column applied to both arms.
"Card-M saving" applies the card-M lines (absolute, 1.20x the estimate); "profile-anchored" applies the measured oneq / served ratio per chunk start to the profile's
paired layer times (the estimate's own base) and is the figure to expect in the TTFT A/B: about -7.8 s at 128k and -29.7 s at 254k. The model-gate A/B (OQ-T1..T4) stays
the end-to-end measurement.

**The unknown was the chain step** with 16 chains (the estimate assumed the K/V forwarding stays at the compute floor, 13.51 us per step, as it does with 8 chains at TP4). Card M
settled it: the 16-chain call sits at the 16.5 us step Q2 measured for the TP2 geometry, in BOTH arms, so the ratio (0.50x from 67k) is the model's and the absolute times are 1.20x
the profile's. Placement does not help (`oneq_noc` 0.68x). `timing_verdict`: OQ-WIN (median oneq / served ratio at 32k and beyond at most 0.75; measured 0.50-0.52).

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

`scripts/ci/sdpa_pf_oneq_tp.py`: `QWEN_FAST_SDPA_PF_ONEQ=1` (strict 0/1, default off; needs `QWEN_FAST_SDPA_PF=1`, which the image environment sets, a production
`QWEN_FAST_SDPA_PF_FLAGS`, four-card serving) replaces the graft's `_qwen_pf_program_config` with a wrapper that calls it and ORs 0x8 into the chain's word where the
factory accepts the call; `QWEN_FAST_SDPA_PF_ONEQ_AUDIT=1` also runs the chain's own program on the first 8 and every 24th engaged call and compares both outputs bit for bit
on every chip. The graft file is not edited. **Bound by `tp_addresses.install()`**, which imports the module and calls `sdpa_pf_oneq_tp.install(environ)` only when
either flag is SET (any value, so a malformed one still reaches the strict parser and fails at attach); with both unset nothing of it is imported, and the pair never reaches
it. `tp_addresses.uninstall()` puts the graft's function back. The module is an overlay-only image file (`docker/qwen-c2-overlay.txt`), not in the P8 copy lists.

## Integration into an image (tp4/prefill-sdpa)

**Which binary: the whole image is switched, with an accept-set, not a flag-gated acceptance.** `build-c2-serving-image.sh` bakes K64j (default, byte-identical to every
earlier image) or, with the job key `C2_BAKE_GRAFT=K64j-OQ`, the K64j-OQ graft (`~/opgraft-K64j-OQ`): the Dockerfile takes `GRAFT_NAME` / `GRAFT_SHA` as build args (defaults
K64j's) and sets `QWEN_FAST_RUNTIME_BINARY_SHA256` to the graft's own digest; G1 (`c2_image_provenance.verify`) holds the image to whichever graft the context holds (its
literals, the K64i superset, the reviewed succession of the runtime pin). Why this and not "accept the new sha only when the profile asks for oneq": the ABAB control arms
run flag-off on the SAME baked image (one image, a read-only rootfs, no boot-time swap), so an acceptance gated on the flag would refuse the control arm, or demand two
images and put the pair on different binaries (the pair would then measure the image, not the lever); and the oneq edits are inert unless a call carries 0x8, which only
the flag produces. `packed_any_admission.admit` accepts exactly the two binaries of `SERVED_BINARIES` (anything else is refused by name; `QWEN_FAST_RUNTIME_BINARY_SHA256` unset
reads as K64j), and K64j-OQ borrows K64j's evidence records **in a gate boot only** (`QWEN_C2_GATE=1`, logged as `binary_equivalence`); a traffic boot on K64j-OQ needs
records made on K64j-OQ (below). With the default build nothing changes: K64j serves exactly as before.

**Evidence and pins bound to the K64j binary sha (`152951c1...`), what a swap voids, and what stays valid:**

| Item | Bound to the sha? | On a K64j-OQ image |
|---|---|---|
| `packed_any_evidence_tp4.json` (131,328 window; CB1, CB2a, CB2b; pinned by `EVIDENCE_TP4_SHA256`) | yes: `binary.ttnncpp_sha256` | **VOID for traffic** (borrowed in a gate boot) |
| `packed_any_evidence_tp4_262144.json` (262,144 window; pinned by `EVIDENCE_TP4_262K_SHA256`) | yes | **VOID for traffic** (borrowed in a gate boot) |
| `packed_any_evidence.json` (the two-chip pair; `EVIDENCE_SHA256`) | yes | void, not served (four cards only) |
| `ordered_writer_evidence_tp4.json` (E1, `ORDERED_WRITER_EVIDENCE_TP4_SHA256`) | NO: bound to the page-writer kernel source, the pinned compute kernel, the geometry (`design_signature`) and its report; it holds no binary field | stays valid (the kernels are JIT sources the swap does not touch; re-running E1w/E1 is optional confidence, 35 min) |
| K64j kernel pins in `packed_any_admission.K64J_KERNELS`, the evidence records' `kernels` / `sources` blocks | no (source shas of the four decode kernels and the python modules) | unchanged: the build proves the four kernels and the chain reader are K64j's |
| `packed_any_admission.K64J_TTNNCPP_SHA256` and the runtime check | yes (the pin) | now `SERVED_BINARIES` (K64j, K64j-OQ); K64j-OQ must also carry the two oneq literals and no K64j-only bytes under its name |
| `docker/qwen-c2-serving.Dockerfile` (`ARG GRAFT_SHA`, the two library re-checks, the `QWEN_FAST_RUNTIME_BINARY_SHA256` env), `build-c2-serving-image.sh` (`graft_sha`) | yes | default K64j, K64j-OQ by `C2_BAKE_GRAFT` |
| `c2_image_provenance.py` (`ENVIRONMENT_SUCCESSIONS`, `GRAFT_*`, the `OQ_*` variant), `test_c2_image_overlay` | yes | updated (variant added, tested both ways) |
| `real_text_compare.REVIEWED_SUCCESSIONS` (K64i -> K64j) | yes | NOT extended: two arms on a K64j and a K64j-OQ image read NOT_COMPARABLE (conservative); the ABAB pairs run on one image |
| `optimisation/ttnn-op/kv_region_read/build_kv_read.sh` (`TTNNCPP_SHA`, names the image's binaries it builds beside) and the `opgraft-KVR` extension pin (`5b2ad8d7...`) | the build script names K64j's sha; the extension is a separate binary | not voided (no ABI change); set `TTNNCPP_SHA` if KVR is ever rebuilt against an OQ image |
| card-M requalification templates (`references/tp4-262k8-jobs/E*`, `references/tp4-s2-serve-jobs/EV-*`, `references/fusion-jobs/WP2/KVPAGE-W*`, `.github/c2-serving-job.env` line 94) | they carry `KOPGRAFT64` / `EXPECT_TTNNCPP_SHA256` (placeholders `@K64J_GRAFT_DIR@` / `@K64J_TTNNCPP_SHA256@`, or K64j's literal in the job file) | re-run with the K64j-OQ directory and sha filled in |
| W1 / W2 / Lever N / engine-reuse / fusion lever evidence, the 262k waiver, `real_text` gates | no (flag-level and source-level) | stand; the fusion pack's gate boots on the OQ image re-prove them on the new binary for free |

A record names ONE binary, so re-recording on K64j-OQ replaces the K64j record in the same file: a K64j image built from that commit is then gate-only. That is the cost of the
wholesale switch and the reason the K64j-OQ evidence is recorded last, once the lever has paid for itself.

**Requalification (only when a K64j-OQ image is to carry TRAFFIC; gate boots need none).** Card M alone (the pair), the existing recorders and templates, placeholders
filled with `@K64J_GRAFT_DIR@=$HOME/opgraft-K64j-OQ` and `@K64J_TTNNCPP_SHA256@=2b81e28f...`; production is down for it (the four-card jobs leave the fabric on the ethernet cores,
hence the quad reset first). Minutes are the templates' own estimates (a quad reset included):

| Step | Templates | Minutes |
|---|---|---|
| 131,328 record | `tp4-s2-serve-jobs`: `EV-R0` 12, `EV-W1` 6, `EV-F1` (CB1) 6, `EV-F2` (CB2a) 6, `EV-W2` 8, `EV-F3` (CB2b) 10 | 48 |
| 262,144 record | `tp4-262k8-jobs`: `E0` 12, `E2w` 8, `E2a` 20, `E2b` 20, `E2xw` 10, `E2c` 30 | 100 (88 after the shared reset: one reset for both records makes the pair 136) |
| both, one window | one `R0`/`E0` reset, the six 131k jobs without it, the five 262k jobs | about 136 |
| E1 ordered writer (optional; the record is not bound to the binary) | `E1w` 10, `E1` 25 | 35 |
| after the cards, no card time | `python3 -B scripts/ci/record_packed_any_evidence_tp4.py --binary k64j-oq [--capacity 262144] --cb1 ... --cb2a ... --cb2b ...` (the recorder checks every report names the K64j-OQ sha); commit each JSON with its re-pinned `EVIDENCE_TP4_SHA256` / `EVIDENCE_TP4_262K_SHA256` together (stage by explicit path: `scripts/ci/__pycache__` is tracked); the CPU suite on the pushed head | no card |

Gate boots (every fusion-pack job, `OQ-A1`, `OQ-T1..T4`, the generated `ONEQ*` jobs) run on the K64j-OQ image today; the ship pack's own jobs (`references/fusion-jobs/ship`) boot
the traffic profile, so they need the records above first (or the K64j image).

## The jobs of the integration (references/fusion-jobs/WPP, one image `tp4-fusion-1`)

`OQ-B0-build-oq` is the fusion pack's `B0` plus `C2_BAKE_GRAFT=K64j-OQ` (use it in place of `B0` for a window that runs a oneq arm: every other arm then runs the K64j-OQ binary
with the lever off); `OQ-A1-audited-attach-all` boots `...-fx-all-oneq-audit` (every fx-all lever with its audit, oneq with its audit); `OQ-T1..T4` are the solo prefill ladder ABAB on
`...-fx-all` (control) and `...-fx-all-oneq` (lever), read with `prefill_ladder_report.py`. The profile twins come from `profiles[]` of `scripts/ci/fusion-wp/WPP.json` (parents
`...-fx-all` and `...-fx-all-audit`, generated after the combined arm), so the lever is measured on top of the combination and on nothing else.

## Files

| File | Purpose |
|---|---|
| `apply_factory_ps.py` | PS0-PS5, the recorded shas, the word decoder |
| `oneq_planner.py` | the planner (ranges, units, chains), the block counts, the time model and the estimate table |
| `host_exec_oneq.py`, `stub_compile_oneq.py` | the real patched C++ blocks, run and syntax-checked on the host |
| `build_k64j_oq.sh` | the graft build into `~/opgraft-K64j-OQ` |
| `oneq_card_m.py`, `run_card_m_oq.sh` | the card-M proof and timing and its runner (the K64j graft mounted read-only, holder check, node recheck, watchdog) |
| `oneq_report.py` | the estimated-vs-measured table from a card report |
| `../../../scripts/ci/test_sdpa_oneq_{factory,build,card,tp,admission}.py` | the CPU tests (patch, host-executed C++, planner, protocol model, time model, fake-ttbuild build, fake-ttnn harness and hook, tp_addresses binding, the admission accept-set and recorder, the `C2_BAKE_GRAFT` job key and build-script switch) |
