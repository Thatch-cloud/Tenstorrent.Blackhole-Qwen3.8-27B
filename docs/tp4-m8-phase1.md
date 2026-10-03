# One 128-row verify block at eight seats: phase 1 (tp4/m8-phase1)

Eight seats run two 64-row packed verify blocks back to back, reading every weight twice. One 128-row block over the same eight users reads
them once (`QWEN_FAST_M8_BLOCK`, default off). The design (block128-design.md, GO WITH CHANGES in block128-review.md) puts the saving at 25 to 33 ms a
round at 4k contexts. Phase 1 builds what a CPU and one card can prove before any of the serving wiring is written: the limits census, the rebinding
twins for the limits that are pure Python or configuration, and the card-M matmul byte compare (B1). The flag is refused everywhere until phase 2:
`serving_startup.start` calls `tp4_m8.refuse_until_phase2` before it attaches anything, at either width.

## What is here

| file | what |
|---|---|
| `scripts/ci/m8_limits.py` | the census: every 64-row / 4-user limit, its file:line (found by a pattern the test re-checks), whether the file is pinned, its status |
| `scripts/ci/test_tp4_m8_limits.py` | the gate: `accepts` rows take 128 rows / 8 users as they stand; `rebound` rows refuse before `tp4_m8.install()` and accept after, with the twin's engaged marker; `phase2` rows still refuse after it |
| `scripts/ci/tp4_m8.py` | the flag grammar, its refusal, and the twins (`install()` / `uninstall()`): rebinds at runtime, so no pinned file is edited; served, so it is in both image copy lists and the overlay |
| `scripts/ci/m8_matmul_plan.py`, `m8_matmul_bytes.py` | B1: the seven 1D configs at M = 32 / 64 / 128 (the same builder, the K order held on the CPU), the LM head's auto config, the byte compare and the timing rules |
| `optimisation/ttnn-op/m8_matmul/run_card_m.sh` | the card-M harness (the cardm step), in the style of the fixed one-card harnesses |
| `scripts/ci/references/tp4-m8-jobs/` | the job pack: one card-M job (B1), then the all-four status / rescan / reset |

## The limits

The design listed about a dozen; the review added seven; building the twins found six more (rows 9, 10, 12, 13, 14, 38), one of which would have
sent conv outputs onto beta slots at eight users. `status` is `accepts` (the code takes it), `rebound` (a twin in `tp4_m8.py`) or `phase2` (still
refuses; the test asserts the refusal). A pinned file is never edited: its fix is a rebind. This table is generated (`python scripts/ci/m8_limits.py`) and
held equal to the code by the test.

<!-- m8-limits-table:begin -->
| # | limit | file:line | pinned | phase-1 action |
|---|---|---|---|---|
| 1 | packed shape: legal block widths | `scripts/ci/packed_shapes.py:27` | no | rebound: table widened to 128 (serving_buffer_pool aliases the same tuple: row 2); the M8 shape is tp4_m8.m8_shape |
| 2 | buffer pool: legal widths (alias of packed_shapes.BLOCK_ROWS) and its top width | `scripts/ci/serving_buffer_pool.py:155` | no | rebound: alias and PACKED_BLOCK_ROWS rebound; the pool still allocates only its replay shapes (row 3) |
| 3 | buffer pool: replay shapes (2, 16) and (4, 16), extent storage and placement per block | `scripts/ci/serving_buffer_pool.py:183` | no | phase2: add (8, 16): extent storage for 8 users (8x the bundle tables and cur_pos words), the lowest free slot, one block built whole |
| 4 | serving shape for a scheduler request count | `scripts/ci/packed_shapes.py:76` | no | phase2: a count of 8 takes M8 only under QWEN_FAST_M8_BLOCK, refused beside QWEN_FAST_M3_BLOCKS=2; wired with the runtime (row 5) |
| 5 | runtime: M3 blocks logic, two-phase build, place_blocks, the eight-seat count | `scripts/ci/serving_runtime.py:39` | no | phase2: one block built whole (no two-phase), carries_in_place kept, request-width warm once before the capture |
| 6 | admission: the extent path is the M3 shape | `scripts/ci/packed_any_admission.py:318` | no | phase2: admit the M8 shape (and re-run CB2b only if an evidence-pinned file changes: it does not, see row 12) |
| 7 | packed step: per-block rounds and the padded rule | `scripts/ci/serving_packed_step.py:401` | no | phase2: one block of 8: packed_device_step; eligibility follows the padded rule of row 8 |
| 8 | idle-segment cap: page 0 holds two idle segments, so one 8-user block serves 6-8 live users | `scripts/ci/packed_verifier.py:1659` | no | phase2: sink pages (S1) or shared idle tile rows (S4), with the v188 idle-exactness gate repeated at 128 rows |
| 9 | packed rows total (literal tuple inside the function) | `scripts/ci/target_packed_pages.py:50` | no | rebound: twin tp4_m8.validate_users (the body with 128 added; the test holds the two sources equal modulo that tuple) |
| 10 | packed participants fill one legal width (literal tuple inside the function) | `scripts/ci/verifier_pack.py:60` | no | rebound: twin tp4_m8.build_pack (same drift guard) |
| 11 | sequential capture rows: the trim is keyed on block_rows == 64 | `scripts/ci/packed_shapes.py:106` | no | rebound: twin answers 4 for the 128-row block, so no per-request engine captures T16 beside it (the v52 OOM pattern) |
| 12 | verifier engine: verify widths and block widths | `scripts/ci/verifier_engine.py:64` | yes: frozen evidence (dspark-*.json, frozen-ladder-corpus.json) | rebound: VERIFY_WIDTHS and BLOCK_WIDTHS widened at runtime, and verifier_engine_tp (which imports VERIFY_WIDTHS by name) follows (the file is in the serving bundle inventory and the frozen evidence) |
| 13 | packed policy: block widths and the native GDN slots | `scripts/ci/serving_fast_policy.py:127` | no | rebound: PACKED_BLOCK_WIDTHS widened (packed_geometry(8, 128) then gives 16 rows and 15 proposals a user) |
| 14 | drafter proposal block widths (32 or 64) | `scripts/ci/dflash_packed_proposal.py:34` | no | phase2: not hit while the quads propose in 64-row passes over rows 0-63 and 64-127 of one tap set; C1 must show it unreached |
| 15 | model_batch: BLOCK_WIDTHS and validate_checkpoint | `scripts/ci/model_batch.py:223` | yes: frozen evidence (dspark-*.json, frozen-ladder-corpus.json) | rebound: table widened at runtime (the file is held byte for byte by test_quad_draft's inventory and the frozen evidence) |
| 16 | model_batch: native_m3 detection by the _64 attribute (the engines, binders and the native_attn guard) | `scripts/ci/model_batch.py:371` | yes: frozen evidence (dspark-*.json, frozen-ladder-corpus.json) | phase2: native_m8 detection by the _128 attributes through a tp_addresses-style twin; the binders log engaged counts |
| 17 | gdn_prefix: ROW_WIDTHS, called by the pinned gdn_device_loop_state.decode | `scripts/ci/gdn_prefix.py:40` | yes: frozen evidence (dspark-*.json, frozen-ladder-corpus.json) | rebound: table widened at runtime (attach refused in the warm forward without it) |
| 18 | gdn_device_loop_state.project_qkvzab_by_tile: cap 64, silently four 32-row calls above it | `scripts/ci/gdn_device_loop_state.py:45` | yes: frozen evidence (dspark-*.json, frozen-ladder-corpus.json) | rebound: twin: one native call when the graft has the _128 configs, else the tiled path AND a fell-back line (+8.4 to 9.1 ms a verify, no marker today) |
| 19 | gdn_records: BLOCK_ROWS (RetainedGDNBlock) | `scripts/ci/gdn_records.py:58` | yes: frozen evidence (dspark-*.json, frozen-ladder-corpus.json) | rebound: table widened at runtime |
| 20 | extent reader: validate_segments limit 64 (bound as a default where it is defined) | `scripts/ci/pooled_attention_replay.py:391` | yes: packed_any_evidence_tp4.json | rebound: twin passes 128 and marks eight segments; the reader (extent_attention_replay_tp) is evidence-pinned and untouched |
| 21 | force_argmax: SAMPLE_WIDTHS (sample_tiles is generic per 32-row tile) | `scripts/ci/force_argmax.py:16` | yes: frozen evidence (dspark-*.json, frozen-ladder-corpus.json) | rebound: table widened at runtime (four in-trace sampler calls at 128 rows) |
| 22 | graft matmul gates: rows <= 2 * TILE select the _64 configs, above it the prefill arms (mlp.py) | `docker/qwen-c2-graft/graft/mlp.py:499` | no | phase2: lever_n_m3native_patch M = 128 section: gates <= 4 * TILE, select _64 or _128 by rows, new graft.sha256 |
| 23 | graft matmul gates (gdn/tp.py) | `docker/qwen-c2-graft/graft/gdn/tp.py:447` | no | phase2: same section |
| 24 | graft matmul gates (attention/tp.py) | `docker/qwen-c2-graft/graft/attention/tp.py:194` | no | phase2: same section |
| 25 | graft model_config: the seven _64 configs (the _128 siblings do not exist) | `docker/qwen-c2-graft/graft/model_config.py:295` | no | phase2: seven *_progcfg_128 (per_core_M 4) from the SAME builder, asserting in0_block_w, fuse_batch, mcast_in0 equal the _64 ones: m8_matmul_plan holds the arithmetic and B1 is the byte gate |
| 26 | LM head: ttnn.linear auto program config (K blocking may change with M) | `image: model.py ttnn.linear(x, lm_head_weight)` | yes: the image's own model source (tp_common / model.py are sha256-pinned on the serving path), not in this repo | phase2: B1 byte-compares M = 128 against 64-row halves and 32-row tiles; pin the 64-row choice if it differs |
| 27 | norms at 128 rows (two_tile_norm: block_h = rows / 32) | `scripts/ci/two_tile_norm.py:65` | no | accepts: none: generic (block_h 4); rms_norm sharded vs per tile is a card byte check (B5) |
| 28 | AttnPrep (K64 graft) is qualified at batch 64 only (512 instances over 110 cores at 128 is an unqualified kind mix) | `scripts/ci/two_tile_decode.py:159` | no | phase2: run the qualified batch-64 op on each 64-row half; q and gate join by one Concat each |
| 29 | NLPConcatHeadsDecode: one input core per user (128 users need 128 cores of 110) | `scripts/ci/two_tile_decode.py:238` | no | phase2: two 64-user calls |
| 30 | K/V write: LAUNCH_ROWS (64, 32) and the rows != 64 refusal | `scripts/ci/packed_ordered_cache.py:47` | no | phase2: halves mode: two chained64 launches per cache over rows 0-63 and 64-127 (disjoint tile rows, per-half metadata); B4 on card M |
| 31 | K/V write launch rows flag grammar | `scripts/ci/verify_trace_t2.py:69` | no | phase2: a halves spelling beside 64 and 32 |
| 32 | tile-split all-reduce at 128 rows (four 32-row tiles) | `scripts/ci/tile_collective_tp.py:49` | no | accepts: none: generic; 512 reduce-scatters a round either way |
| 33 | K5-A users: the pinned pair module | `scripts/ci/gdn_user_batch.py:37` | yes: pair module of the qualified K5 launch (docs/tp4-recurrence-split.md: do not edit) | rebound: the pair file stays 4; the TP4 sibling carries the bound (row 34) |
| 34 | K5-A users: the TP4 sibling inherits the pair's bound | `scripts/ci/gdn_user_batch_tp.py:28` | no | rebound: MAX_USERS = 8 at runtime: 8 x 12 = 96 distinct points on 11 x 10; a new program instance, so the P0 compare at 8 users is a card gate (B2) |
| 35 | K5-A / V1 users bound in the conv path (reads gdn_seq_block.batch, the TP4 sibling at four cards) | `scripts/ci/gdn_user_batch_conv.py:101` | no | accepts: none after row 34 |
| 36 | T2 packed windows bound by the PAIR module's MAX_USERS (and so V1 block conv falls back) | `scripts/ci/gdn_conv_windows_packed.py:331` | no | rebound: twin asks the original about each group of four users and answers the first reason |
| 37 | rows DMA: 256 runtime words a core, 8 a task: at most 31 tasks a core, 3,410 on 110 cores | `scripts/ci/gdn_rows_dma_tp.py:32` | no | rebound: launch twin splits an oversize list into equal launches (the 3,600-task unstack becomes 1,800 + 1,800) |
| 38 | unstack destination layout hard-codes four users (conv 0-3, beta 4-7, g 8-11, z 12-15, windows 16 + 4u): at eight, conv 4-7 lands on beta | `scripts/ci/gdn_block_conv_tp.py:76` | no | rebound: twin derives every base from the user count (the original's list at 1-4 users); in neither the design nor the review |
| 39 | GDN split / canon / merge planners at 8 users (1,032 / 516 / 192 tasks) | `scripts/ci/gdn_rows_dma_tp.py:65` | no | accepts: none: within one launch's capacity; V2 / V1 at 8 users are card-checked (B6) |
| 40 | attention fold planners at 8 bundles | `scripts/ci/attention_block_fold_tp.py:61` | no | accepts: none: within capacity |
| 41 | ops profile plan: users | `scripts/ci/ops_profile_plan.py:167` | no | accepts: none: an eight-seat profile plans eight users (tp4/262k8); C6 profiles the M8 recipe on one block |
| 42 | K5 card harness users (gdn_tp4_card_test USERS = 4; the TP2 gdn_seq_block_device_test --users 1-4) | `scripts/ci/gdn_tp4_card_test.py:48` | no | phase2: extend the UB and K5 sections to --users 8 (B2) |
<!-- m8-limits-table:end -->

Counts: 6 accepts, 18 phase2, 18 rebound.

## B1, the card-M matmul byte compare

Every projection of the verify at its four-card per-chip shape is run at 128 rows (per_core_M 4) and compared bit for bit, tile by tile, with the two
64-row halves (the config built at M = 64) and the four 32-row tiles (the sequential engine's), for the `served` variant (verify-trace T1 #11's attn_qkv on
44 cores and gate on 88) and the `base` variant (the plain `_64` build), then timed at 32, 64 and 128 rows; the MLP projections are re-gridded at per_core_M 4
and the best exact config is named with the builder arguments that reproduce it. The LM head is `ttnn.linear` with no program config: ttnn picks one from M,
and that is the design's one real unknown. Pass is exactness only (`M8_MATMUL verdict=PASS`); the timing rules are the review's (`t128 <= t64 + 1.5 (t64 - t32)`
per projection, `sum t128 <= 1.1 sum t64` over the seven) and are reported, not gating. Nothing is applied by the job.

## What phase 2 needs

Estimate 12 to 16 engineer-days, one card-M window (2 to 4 h) and two four-card windows (the design said 10 to 15 days; the review's additions are in).

| piece | days | notes |
|---|---:|---|
| graft M = 128 section (`lever_n_m3native_patch`) | 1.5 to 2 | seven `*_progcfg_128` from the same builder asserting `in0_block_w`, `fuse_batch`, `mcast_in0` equal the `_64` ones (m8_matmul_plan holds the arithmetic); the gates `<= 4 * TILE` select `_64` or `_128` by rows; new `graft.sha256` and source pins; the LM head pinned if B1 says so (0.5) |
| model forward at 128 rows | 2 to 2.5 | `native_m8` detection by the `_128` attributes (a twin: `model_batch` is pinned); AttnPrep and the concat heads by 64-row halves with the q / gate join; V1 / V2 stage at eight users (the layout and split twins are done here; the stage needs the card, B6) |
| K/V write halves mode | 1 | two `chained64` launches per cache over rows 0-63 and 64-127, per-half metadata staged with the tiles, `verify_trace_t2` flag grammar; B4 on card M |
| host wiring | 2.75 | shapes and the pool (extent storage for eight users), the runtime building one block whole, admission, the packed step on one block; each row 3 to 7 of the table |
| idle-segment cap | 1 to 2 | page 0 holds two idle segments, so one block of eight serves 6 to 8 live users: sink pages (S1, needs a cache-allocation hook in the plugin) or shared idle tile rows (S4), with the v188 idle-exactness gate repeated at 128 rows |
| profiles, copy lists, closure tests, K5 harness at 8 users (B2), B5, B6 | 2 | gate-only profiles `c2-packed-tp4-m8-best[-gate]`; `gdn_tp4_card_test --users 8` |
| analysis | 2 to 3 | C4 paired ABAB against the two-block arm, C6 device profile |

Capture budgets, none observable on this ttnn (Q18): the trace region at 512 MiB holds one trace of about 2,970 ops (the two-block arm captures two of 2,252);
the block DRAM is about 3.6 GB a chip (the C1 gate: at most 3.65 GB); the in-trace sampler (four tiles) holds about 64 MB of scratch; every device buffer the
new path uses must exist before the first capture (the halves' per-half position and page-row tensors, the extent tables for eight users, the sink pages if S1,
the 128-row logits), and the audits-off hang shapes (five of five) are mandatory on the new trace shape.
