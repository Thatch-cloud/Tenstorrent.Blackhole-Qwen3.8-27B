# tp4/octo-3: octo-T8 and its three levers on the fused stack

Branch `tp4/octo-3` = `tp4/oneq-ship` (the fusion ship stack: Lever N, W2, engine reuse, the op-fusion set, the prefill levers) + `tp4/toolcall-whitespace` + `tp4/rebind-fast` + the work below.
Everything here is **default off, gate only or under an owner traffic waiver (PENDING), and unqualified on cards**. Nothing in this document has run on a card on the fused stack.

## 1. Measured and estimated

**Measured, on the Lever N stack** (tp4/octo-2, image tp4-octo2-1, 8 live, ~3.5k-token coding prompts, `QWEN_FAST_OCTO=alternate`: half the rounds are octo, half the old two-quad m3 shape;
paired per round with `octo2_report.py` against the control boot; runs 38096633389 control, 38097257289 draft, 38100444114 glue8, 38101222763 bundle, 38101791976 all three, 38102380592 control repeat):

| arm | committed tokens per second per seat on octo rounds | notes |
| --- | --- | --- |
| octo draft | +6.1% | early draft -12 ms, tokens per seat-round -2% (tau) |
| glue levers | +28.6% | the octo block's verify trace 94.5 to 55 ms with the bundle (see below) |
| attention bundle | +2.5% | trace 94.5 to 89.8 ms at 4k |
| all three | +46.6% | 35.46 against 24.20 |

Text: `octo_compare.py`, 24 of 24 answers identical in every arm.

**Estimated** (this branch): every number about the fused stack (section 4: the octo draft against the fused quads, -4..-8 ms), the memory margin and the pack minutes.
**Not measured anywhere**: any octo lever beside W2, engine reuse or the op-fusion set; the 13x10 grid with the octo block (the octo-2 runs were already on 13x10); the traffic profile.

## 2. Every fused lever's octo status

The octo block is a THIRD packed block of eight users x eight rows (one 64-row verify pass) beside the two M3 blocks of four users x sixteen rows. A lever of the combined arm is `native` (generic in users and
rows), `octo-served` (its design is the M3 shape's: the octo block runs the served composition and says so in a line of its own, which the judge demands), `m3-only`, `drafter`, `host` or `prefill`.
The table is `scripts/ci/octo3_levers.py`; `test_octo3_judge` holds that every flag the combined arm adds over the production profile has a row.

| flag | status | why |
| --- | --- | --- |
| `QWEN_FAST_CCL_OPTIONS` | native | keyed on the block rows (64) and the layer count; the engaged line is per rows, the audit needs two block owners |
| `QWEN_FAST_GRID13` | native | matmul grids for the 13x10 grid: the prefill all-gather matmuls, the verify block's output projections at 64 rows (the octo block is 64 rows) and the fused commit's feature projection |
| `QWEN_FAST_MLP_CFG` | native | bound by block rows (64); needs QWEN_FAST_SINGLE_GATEUP=1, which the octo shape requires |
| `QWEN_FAST_TP4_CONV_GATES_SPREAD` | native | F1 runs in the M3 blocks' conv stage; the octo block reaches it through glue8's block conv (gdn_block_conv8_tp.stage, one launch at 64 rows, the w2.off kill switch checked there too); without glue8 the octo block runs the served per-user conv gates |
| `QWEN_FAST_TP4_SHARD_ARGMAX` | native | scans the 64-row logits shard (rows 1..64, the device grid); no users or rows_per_user in the plan |
| `QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2` | native | the tree fold of the same scan |
| `QWEN_FAST_KV_PAGE_WRITER` | octo-served | the design (design_signature, the card-M record) is four users of sixteen rows; the octo block's eight users of eight rows keep the chained writer and log "[PINDIAG] tp4 kv page writer octo 8-row served path" once (kv_page_writer_tp4_smoke demands it on an octo profile); an 8x8 design is its own card-M record |
| `QWEN_FAST_TP4_SDPA` | m3-only | W2 multi SDPA is the M3 blocks' one launch over sixteen-row users; the octo reader keeps eight single launches or the bundle (extent_attention_fold_tp skips sdpa_long_tp.apply for eight-row segments; w2_switch.applies already excluded the octo block); serving_octo admits it beside the octo shape |
| `QWEN_FAST_DRAFT_GATEUP1` | drafter | row-based (per_core_M = 2 as the quad); the WP6 audit budget is per pass (draft_fusion_tp.set_shape) |
| `QWEN_FAST_DRAFT_HEAD64` | drafter | the octo pass is a QuadPass: its head is the quad's two 32-row halves as one 64-row matmul |
| `QWEN_FAST_DRAFT_MM_GRID` | drafter | row-based grids; audited per pass |
| `QWEN_FAST_DRAFT_PERMUTE` | drafter | the octo pass hooks kv, fold and unfold at shape=octo (halves=2 users=4 block=8); octo_draft_tp._execute tells the audit which pass is eager |
| `QWEN_FAST_DRAFT_QKV1` | drafter | site=octo in draft_attention_branch; the fused weight is shared with the quad |
| `QWEN_FAST_DRAFT_REDUCE` | drafter | row-based; audited per pass |
| `QWEN_FAST_DRAFT_TAIL` | drafter | row-based; audited per pass |
| `QWEN_FAST_PARKED_DRAFTS` | drafter | the draft book holds pairs and block quads; the octo pass's trace is not in it (octo_draft_tp.EXCLUDED_FLAGS no longer refuses the flag; the coordinator closes and rebuilds the octo trace after a rebound member): the one lifecycle no card has run |
| `QWEN_FAST_BATCHED_READS` | host | the verify read-back works at 64 rows for any shape; the octo draft pass's own collect uses its unbatched reads (not applied there: a ~0.5-1 ms gap, no marker, no failure) |
| `QWEN_FAST_LEVERN_BUILD_MS` | host | the governor charges admission per slot; no shape coupling |
| `QWEN_FAST_PARKED_ENGINES` | host | the octo block is built first and the parked set after it; an octo round drafts with the book's quad traces cut to seven rows (or the octo pass, which the coordinator keeps outside the book) |
| `QWEN_FAST_PRESTAGE_DIFF` | host | a per-block record; the pre-stage covers three blocks (the octo block is block C); the destinations overlap check is the open question (hostgap rule fails loudly) |
| `QWEN_FAST_TAP_WINDOW_ALIGNED` | host | the drafter window's tile-aligned join at a returning turn's rebind (tp4/rebind-fast); data movement before any block runs, shape-independent |
| `QWEN_FAST_TP4_ROUND_HOST_KEYED` | host | keyed on the block's own shape and fixture; the keyed-share smoke rule counts one verify for an octo step |
| `QWEN_FAST_TP4_ROUND_HOST_LEAN` | host | drops the per-phase and commit audit lines only (the lean ledger replaces them); no device work |
| `QWEN_FAST_TP4_ROUND_HOST_READ` | host | the octo readback uses round_host.merge_chunk_candidates at block_rows 32 |
| `QWEN_FAST_TP4_ROUND_HOST_SELECT` | host | the batched selection is row-wise for any user count (the octo draft returns into the same path) |
| `QWEN_FAST_WRITE_PACKED_LEAN` | host | a lean address book and mapper per block; the engaged line is per block, no shape assumption |
| `QWEN_FAST_DEVICE_ZEROS` | prefill | engine start: fills the pool's zero buffers (the (8, 8) extent storage is lent by the same filler) |
| `QWEN_FAST_LAZY_SHARD_W` | prefill | engine start: the weight loader |
| `QWEN_FAST_SDPA_PF_ONEQ` | prefill | the chunked prefill SDPA with one q chunk per core; acts in prefill, needs the K64j-OQ binary (the oneq arms) |

Read from code (three read-only audits of the verify-pass, host-side and drafter-side levers), proven by the audited attach A1 of the O3 pack. What had to change to make the stack boot at all:

* **W2 beside the octo block.** `serving_octo.EXCLUDED_OFF_VALUES` refused `QWEN_FAST_TP4_SDPA` of any value, and `extent_attention_fold_tp.PackedExtentReplayReader` called `sdpa_long_tp.apply` for the octo
  reader too (which would raise "qualified at flags 0x23 only" on the 0x21 octo segments). The reader twin now leaves an eight-row reader to the octo readers (`multi` stays `None` for it, or becomes the
  bundle); `serving_octo` and `octo_attn_bundle` admit `multi`. `w2_switch.applies` already excluded the octo block, so `w2.off` does not touch octo rounds; `gdn_block_conv8_tp.stage` now honours the F1
  kill switch as `gdn_block_conv_tp.stage` does.
* **The K/V page writer** (`QWEN_FAST_KV_PAGE_WRITER`): its design and card-M record are four users of sixteen rows (`design_signature` binds them), so the octo block's capture used to log `fell back`, which
  every fusion smoke rule fails on. It now logs `[PINDIAG] tp4 kv page writer octo 8-row served path` once; `kv_page_writer_tp4_smoke` demands it on an octo profile and refuses it elsewhere. An 8x8 writer is
  its own card-M record (64 units on 64 cores at WT=4 would fit) and is not built.
* **The octo draft beside engine reuse.** `octo_draft_tp.EXCLUDED_FLAGS` refused `QWEN_FAST_PARKED_ENGINES` and `_DRAFTS`. They are lifted: the coordinator keeps the octo trace outside the draft book (it follows a
  rebound member and a released device as the quad did before the book, `_rebound` / `_note_generations` / `release_parked` / `close`) and the next all-eight-live octo round rebuilds it. **No card has run that
  lifecycle.** The parked judge's trace-steadiness rule belongs to the churn job, which this window does not run.
* **The WP6 / WP7 audits at the octo pass.** `octo_draft_tp._execute` now tells the permutation audit which pass is eager (as `quad_draft_tp._execute` does) and tags the WP6 audit budget per pass
  (`draft_fusion_tp.set_shape`): before, the two quads' warm passes spent the budget and the octo pass was never audited. `draft_permute_smoke` demands kv, fold and unfold at `shape=octo`.
* **Smoke rules that mis-counted a third block.** The round-host keyed share counted two verifies per eight-live step (an octo step has one: the old rule capped a healthy octo arm at 0.5); the pre-stage
  rules wanted a 4-live diff verify and audit lines of blocks A and B (an octo-live arm runs the M3 blocks only below six live seats): they accept the octo block (`C`) and 8-live diff verifies.

## 3. The three defects of the octo-2 card runs

1. **`[OCTO-ATTN-BUNDLE] engaged` twice a boot.** Not a parked engine or a re-warm: `packed_verifier._warm_and_fixture` builds TWO fixtures per block (the eager warm forward's, then the captured one), each
   with its own `PackedExtentReplayReader` and so its own bundle (the card logs show the second line 0.9 s after the first, with a second `extent replay engaged segments=8 ... G8B1`). The two are the same fact:
   `octo_attn_bundle` logs the engaged and UNQUALIFIED lines once per process (`_log_once`, `reset_logged` for tests); the judge rule (exactly one) stays.
2. **No `[QWEN-SDPA] runtime-extent entries=1 kv_share=false` line on the bundle path.** The factory logs the page table's batch B: the bundle's K64j program has B = 4 users, so the log reads
   `entries=4 kv_share=false` (it is in the card log) and the octo rule demanded `entries=1`. `octo_judge.factory_entries(env)` derives the wanted sizes from the profile (1; 4 for `ATTN_BUNDLE=4`; 1 and 4
   under the bundle audit). On the W2 stack the M3 blocks' own multi launch also logs `entries=4 kv_share=false`, so on a bundle of four that line cannot tell the bundle from W2: the bundle's engaged and
   call lines and the octo reader's `G8B1` line are the proof (`bundle_problems`).
3. **The glue8 audit could not finish** (warm-up 412 s, 3.4 tok/s a seat). The in-trace audit is not the cost (the device pays the served path beside the native one on every replay whatever the host does);
   the host READ-BACK is: `tp4_vglue.audit_round` copies every audited entry of two layers off four chips after EVERY round (an octo layer holds 73 entries: 8 pieces, the merged output, 8 x 8 block-conv
   entries; each read as one tensor per chip, twice, about a thousand `to_torch` calls a round) and all 48 layers on round one. The audited octo-2 twin also sat on the Lever N audited parent, whose
   prefix digests (8.4 minutes a request), singles audit and Lever N audit are instruments of their own. `audit_sampling.py` (`QWEN_FAST_AUDIT_FIRST`, `QWEN_FAST_AUDIT_EVERY`, strict, gate only, unset =
   every round as before): the first N rounds of a block, then every K-th; the layer rotation follows the audited rounds' ordinal so a long run still covers every layer; the vglue, packed-windows,
   F1 and K5-A read-backs sample, the counter audits (page writer, CCL, MLP, shard argmax, multi SDPA, round host, pre-stage) do not. The audited octo3 twin is `-fx-all-audit` (not the Lever N
   audited parent) with `FIRST=1, EVERY=16` (about 1.2 s a full audited round at eight users: an estimate from the 3.4 tok/s, so amortised about 75 ms a round).

**The 32k read.** "At 32k the bundle arm was 4% below the control": no octo round ran in that test, in any arm. `concurrent8_code_32k` prefills eight prompts one after another (about 15 s each) and an
800-token answer is over in 40 s, so the first users finish before the last has prefilled: the log has not one eight-live round after the first test (`live=8` counts: 138, 5, 139, then 0 in every later
150-round window), and the per-user table shows the arrival order is a thread race (user 0's first token at 9.5 s in the control and 38.4 s in the bundle arm; the last arrival decoding alone at 51 or 31
tok/s). The aggregate of that test measures the race. `concurrent8_code_32k_long` / `_128k_long` (budgets of 3,000 / 4,000 tokens, ignore_eos) outlast the prefill queue so eight seats decode together at
depth, which is the only way an octo round runs there.

## 4. Which drafter drafts an octo round

Options: (a) the fused quad drafters, cut to seven rows on the host (today's octo draft), (b) the octo draft pass (`QWEN_FAST_OCTO_DRAFT`, one 64-row pass over 8 seats x 8 rows, a QuadPass at eight users).

* Measured on the Lever N stack: (b) saved 12 ms of the early draft against two UNFUSED quads (14.26 ms of device time each), at -2% tokens per seat-round (the checkpoint is trained at 16 rows: tau).
* Measured since: the fused set made a quad about 9.2 ms (two fused quads 18.4 ms of device time after a 2.25 ms first enqueue, host path 26.6 ms: the host window is the floor in 91% of 8-live steps).
* Estimate: the octo pass costs about 9.5-10.5 ms of device time (a quad's plus up to 1.5 ms for the 24-piece K/V plan and the 8-way fold), so it returns about 8 ms of device time and -4..-8 ms of the round,
  against -2% tokens (+2..+5% rate, central +3%), plus one more capture (about 0.52 GB, estimate) and the lifecycle above.
* **Decision: the fused quad drafters draft an octo round by default**, and the octo draft stays an arm. The pack's `-octo3` carries all three levers (the brief); `-octo3-quaddraft` is the same without the
  draft and Q1/Q2 are the A/B. A tie goes to the quad drafter (no extra capture, no lifecycle code, tau intact). The ship candidates carry the draft today because the owner asked for all three; if Q1/Q2 favour
  the quads, drop `QWEN_FAST_OCTO_DRAFT` from the ship entries (one line each in `fusion-wp/O3.json`).
* Not applied at the octo pass: `QWEN_FAST_BATCHED_READS` for the pass's own collect (`read_octo_outputs` keeps its unbatched reads; about 0.5-1 ms, no marker, no failure).

## 5. Modes and profiles

`QWEN_FAST_OCTO=live` is the non-alternating mode: every eligible round (six or more live seats, every live request bound to the octo block) is octo, none is an m3 round; the control is the fused stack's
own profile on another boot of the same image. `alternate` (a gate arm for in-boot pairs) is not waivable. `make_octo3_profiles.py` generates, on the combined arm `-fx-all` and, for two of them, on
`-fx-all-oneq-grid13`: `-octo3` (all three levers), `-octo3-quaddraft`, `-octo3-audit` (`-fx-all-audit` + the levers with the bundle's, GDN glue, packed-windows, F1 and tile-aligned-window audits + sampling),
each with the third block's memory plan (the KV pool 19,968 to 16,500 blocks, QWEN36_MAX_TOKENS_ALL_USERS 1,055,488, the trace region 512 to 640 MiB; four full 262,144-token windows still fit; 8 x about
120,800 tokens reserve 15,112 of the 16,492 usable blocks). **Memory is the named risk**: the plan's margin is 5 MB, the 8 per-user state sets of the octo block are estimated near 0.4 GB a chip more than the
plan allows, the eight parked engines take 3.35 GB, the octo draft capture is estimated at 520 MiB, and the audited free-DRAM floor after the engines is 3.6 GB: A1 records the ledger before anything is timed.

## 6. The O3 pack

`references/fusion-jobs/O3` (generated by `make_octo3_jobs.py`; image `tp4-octo3-1` built by B0 with `C2_BAKE_GRAFT=K64j-OQ`): B0, X0, **A1** audited attach (`-octo3-audit`: warmup, coding, parked_turns,
concurrent8_steady), **C1 L1 C2 L2** the ABAB (control `-fx-all`, arm `-fx-all-octo3`, each: warmup, steady, 32k, 128k, 32k_long, 128k_long), **Q1 Q2** the draft A/B, **A2** the audited attach of the
prefill combination with the octo levers, **RBC1 RBL1 RBC2 RBL2** the tile-aligned drafter window's turn ABAB (folded in from `references/fusion-jobs/RB` on this image), Z. ARC runners at zero from A1 to RBL2.
810 minutes in all (estimates), the core (B0 to L2) 410. Reads: `octo_judge.py`, `octo3_report.py` (the all-live window of each test, both pairs, the floors: GAIN / PARITY / LOSS; SHIP-GO needs no LOSS and a
steady GAIN), `octo_compare.py` (every answer equal).

The tile-aligned drafter window (`QWEN_FAST_TAP_WINDOW_ALIGNED`, tp4/rebind-fast) acts at a returning turn's rebind, never in a steady decode window, so the timed octo arms do not carry it; the audited twin
does (A1 audits the candidate's whole composition) and the RB turn ABAB times it.

## 7. The ship candidates and their waiver

`fusion-wp/O3.json` generates, with `make_fusion_profiles.py`, two TRAFFIC profiles: `...-fx-traffic-o3` = the approved fusion ship profile (`-fx-traffic`, byte-identical, with its waiver) plus the octo
levers (non-alternate), `QWEN_FAST_TAP_WINDOW_ALIGNED=1`, `QWEN_FAST_OCTO_WAIVER=<id>` and the memory plan; and `...-fx-traffic-o3-oneq` = that plus `QWEN_FAST_SDPA_PF_ONEQ=1` (its image bakes K64j-OQ, whose
packed-any records have landed). Each has its own `owner_traffic_waiver`, decision **PENDING**; their gate packs (B0 refused while PENDING, G1, GX0/GX1 against `-fx-traffic` on the same image, G2, SR) are
`references/fusion-jobs/ship-traffic-o3` and `ship-traffic-o3-oneq`.

**What the octo levers are outside of** (and so why a waiver, not a flag):

1. `serving_octo.octo_admission` refuses the shape outside "a gate run of a gate-only four-card profile" (`QWEN_C2_GATE=1` with `QWEN_C2_GATE_PROFILE=1`): no card run qualified it. A traffic profile never has the
   marker. The waiver is the one other way in: the profile's own `QWEN_FAST_OCTO_WAIVER` carries the waiver's id (the contract, `octo_problems`, refuses it on any profile whose waiver has another id, on a
   gate-only profile, without the shape, and removes an inherited value); the admitted line says `(owner traffic waiver <id>)` and `octo_judge` checks it against the profile.
2. The pinned qualification records (`packed_any_evidence_tp4*.json`) cover the served K64j programs at flags 0x23 (G8B2) and the multi-user launch is the M3 blocks' own. The octo block's attention is the
   same K64j program at flags **0x21 with one eight-row group per bundle (G8B1)**, and with the bundle **B = 4 entries** of it: no record covers either; `packed_any_admission.admit_blocks` checks only that
   the block is the extent block, so it would not notice an octo block (the waiver, and the octo reader's `G8B1` line in the log, stand in).
3. The batched GDN launch at eight users (96 cores, one wave), the quarter-tile glue movers, the third block's fused commit and publication warm have no card-M record either.
4. The octo draft pass changes only how many tokens a round commits, never which; its cost is tau and one more capture, and its trace is outside the draft book.

**What would replace the waiver** (a qualification record): the K64j card jobs `Q1w/Q1a/Q1b` (`k64j_card_b.py --kv-heads 1 --octo`: G8B1 at 0x21 equal to its compile-time twin at every family and start,
one trace over more than 50 families, the stale-writer zone, the native one-row decode against the one-entry call) with a **B = 4 bundle arm (not built)**, `Q2` (`gdn_tp4_card_test.py --users 8 --rows 8`: one
launch equal to eight per-user launches bit for bit) and a quarter-tile mover harness (**not built**), recorded in a new pinned section of the packed-any evidence beside the G8B2 one (**not built**: a
`record_packed_any_evidence_tp4.py --octo` and its pin in `packed_any_admission.py`). Until then the owner's decision stands in for them, and the evidence the waiver cites is the octo-2 card runs above
(text identical, 24 of 24; +46.6%) plus the exactness argument of each lever (docs/tp4-octo-draft.md, -glue8.md, -attn-bundle.md), none of it on the fused stack.

**`QWEN_FAST_ADMISSION_DIAG_TRIM`: excluded.** It skips the second and later admissions' `prefill_before`/`prefill_after`/`before_engine`/`engine` memory-ledger points and the `dram after engine` line, and
makes the GDN slot adoption (`adopt_prefill_slot`) pass `readback=False` after a slot's first adoption: `gdn_snapshot._verify_row` then does shape/dtype checks only and the bit-for-bit comparison of the
conv-state slice against the live row (a ValueError, i.e. a refused admission, on a mismatch) runs once per slot per process instead of on every admission. No production decision reads the ledger (the DRAM
backstop, the dram hold, the parked arrival terms and the `make_room` ladder never import it) and every skipped item is a read or a log line, so it changes no token; but it weakens a runtime assertion
production executes on every adoption, its saving on a returning turn is less than the quoted 0.19 s (the rebind's own before/after points, about 0.13 s, stay), and it has only run on the old `c2-packed-tp4`
profiles (never beside prefix reuse, Lever N, engine reuse or sticky sessions). Include it only as a gate-only twin measured in the RB turn ABAB, or as a strict value that trims the diagnostics and keeps
the adoption readback (about 10 lines in `memory_ledger.trim_enabled`).

## 7a. Card finding: the page writer's octo decline (GX1 of the combined candidate, run 38117538847)

The combined candidate's first card boot died at engine start while capturing the third block's verify trace (`Cannot load new binaries during trace capture`, `mesh_workload.cpp:159`, in
`packed_ordered_cache.ChainedOrderedCacheWriter.__call__`'s `to_memory_config`). Cause (this branch's, not the candidate's other parts): the K/V page writer declined the octo block by its SPANS
(eight users of eight rows), but the block's WARM forward is built over one chain of 64 rows (`model_batch.chained_writer_options(single_chain=True)`: the placeholders write one tile row), so the
warm forward attached the page writer and the captured one did not; the capture then launched the chained writer's conversion and chained launch, which had never run eagerly. The octo-2 runs
never met it (no page writer on the Lever N stack). Fix: `packed_verifier.build_fixture` builds both fixtures inside `verify_trace_t2.block_layout(the block's segments)`, the writer hands that
layout to `kv_page_writer_tp4.attach` (`layout_spans`), and the decision reads it: both forwards take the chained writer. (`model_batch.py` is not edited: test_quad_draft pins it.)
`test_fusion_capture_rules.KVPageTests` builds both writers the way the model does, under the real profiles' env, and runs the warm forward then the capture behind the capture rules; the
negative control (a warm writer without the layout) reproduces the card's failure. Any other lever that decides a block's path from the warm fixture's collapsed spans would fail the same way.

## 7b. Card finding: the audited attach of the combined candidate (run 38129663805)

The candidate booted and served (audit_mismatches 0 over 803 publish rounds, octo rounds executed, no compile on a shape switch); four things read red.

1. **`RecursionError` in every mid-serving draft build, then `octo draft disabled`.** `draft_fusion_tp.track` and `grid13_tp.track` both wrap the ttnn module's `begin_trace_capture` / `end_trace_capture`
   and both run on every audit call; each looked only for its OWN marker on the TOP wrapper, so with the other's wrapper on top each wrapped again. A Lever N cycle alternates prefill chunks (grid13's
   agmm audits) with draft passes (WP6's audits): the chain grew one layer per alternation until the next capture (a new pair's build, the octo draft's rebuild) recursed through it. Neither package alone
   alternates, so neither package's tests saw it. Fix: every tracker lists itself and the trackers beneath it on its wrapper (`capture_trackers`) and finds its own anywhere in the chain
   (test_capture_trackers: 5,000 alternations leave two wrappers; the old code fails it). Only the audited arms call `track`; the traffic arm never ran into this. `QWEN_FAST_OCTO_DRAFT=1` stays in the
   candidate: the +46.6% measured on the Lever N stack was all three levers together, the quad-draft arm (Q1/Q2) is the A/B that decides whether the octo draft's +6.1% is worth its extra capture
   (its tau, one more trace, a lifecycle beside the draft book); with the recursion gone nothing else argues for dropping it.
2. **`only 0 of 225 verifies took the keyed write` and no `kind=keyed` audit line.** `verify_prestage.key_matches` returns False while `QWEN_FAST_VERIFY_T2_AUDIT=1` (an audited verify takes the diff that
   writes the T2 audit's lines), so an arm that sets it takes the value diff on every round, M3 or octo: the audited octo twins set it (the packed-window audit of the third block), `-fx-all-audit` does not.
   Not an octo gap: the smoke rule now exempts the keyed share and the keyed audit line under that flag (`round_host_keyed_exempt`). What the audited arm gives up is the diff's own recompute, about
   2.3 ms of a verify (`diff_ms` of the fences line); the keyed claim is audited by the arms without the T2 audit (M3 blocks, `-fx-all-audit`), and an octo round's keyed share is judged on the traffic
   arm's ledger. Nothing here measures the keyed write ON an octo round: the first unaudited ledger that does is G1's.
3. **Quad rules that assume the M3 pair of quads.** Under `QWEN_FAST_OCTO=live` with the octo draft the eight-seat rounds are one pass over eight slots (`[PACKED-SELECT] pairs=[[0..7]]`), the second quad
   is never built and the first still serves the M3 rounds below the octo threshold. `blocks_problems` is octo-pass aware (`octo_pass_mode`): the first quad is required, the second at most once, a round
   must be drafted by the octo pass or by both quads, a quad fallback or disable still fails. Without the octo draft, under `alternate` and off octo the rules are unchanged (test_octo3_judge).
4. **Free DRAM after the parked engines, 2.186 GB a chip against a 3.6 GB audited floor.** Measured on all four chips; the parked engines took the usual 3.336 GB (0.418 each). The octo block's packed
   block is 3.602 GB against 2.471 GB for an M3 block and the lower KV pool gives back about 1.9 GB, so the octo stack holds about 1.5 GB less than the four-user one; the audits are part of what is held in the audited arm.
   The octo arms have their own floors (`parked_judge.LEDGER_FREE_FLOOR_OCTO_GB`): audited 2.1 GB (the measurement less 4%); audits-off 4.7 GB, the four-user stack's, because the traffic arm MEASURED
   4.907 GB free a chip after the parked engines (G1, run 38131783025; P7 after the attach 8.254 GB): the audits of the octo stack hold 2.72 GB, not the 1.10-1.26 GB of the four-user one, and my first
   estimate of 3.2 GB for this floor was far too loose. The traffic arm needs about 1.47 GB for an arrival and keeps about 4.2 GB after the first round, the quad and the octo draft have built; the audited
   arm measured 1.498 GB there. No change to the host tier (host RAM) or the octo plan is needed for the traffic arm.

## 8. Not done

K5-A at eight rows (the octo block keeps the served batched GDN launch); an 8x8 K/V page writer (own card-M record); the octo draft in the draft book; the batched collect for the octo pass; the Q1/Q2 qualification
records; any card run. The octo pass and the quads' memory beside eight parked engines is the first thing A1 reads.
