# tp4/octo-t8: adaptive drafting rounds at eight seats, and the lone-user padded round

Status: the host side AND the device pieces are built and tested on CPU (the attach builds the third block, the batched GDN launch carries eight users, the octo extent readers take one
eight-row group per bundle, the publication warm and the fused commit cover the block); the flag still engages in a **gate run of a gate-only profile only**. Nothing here has run on a card: the
admission logs every card question as an `[OCTO] UNQUALIFIED` line naming the job that settles it (`serving_octo.UNQUALIFIED_ITEMS`), and the first card job (Q1w) is the one that can kill the shape. Every
card result this document asks for is UNQUALIFIED until it has run. Default off, gate only, byte-identical off. No kernel and no K64j graft changed: see section 5.

Owner rule: **strict exactness**. Greedy output must stay byte-identical to the target's own greedy decode. Speculation may only change how many tokens a round
commits, never which.

## 1. The measured basis

Measured on the eight-seat 262k stack (profile `c2-packed-tp4-8x262k-ship-prefix-levern-traffic`, the X2 timing run):

| round | time |
| --- | --- |
| 4 live (one 64-row M3 block, 4 seats x 16 rows) | 81 ms |
| 8 live (two M3 blocks and two quad draft passes), 4k context | 157-161 ms |
| 8 live, 33k context | 177 ms |

The 8-live round splits as 2 x 49.7 ms trace + 7.2 ms host verify work + 42.9 ms (two quad drafts) + 6.2 ms residual. A single block's trace costs 49.1 ms at 2 seats and
49.7 ms at 4: **the rows are nearly free, the pass is what costs.** A seat commits 3.3-4.8 tokens per round at 8 live and only 5% of seat-rounds emit more than 8
(capping a round at 8 rows keeps 0.85-0.99 of the tokens).

So at 6 or more live seats one 64-row pass of 8 seats x 8 rows (7 proposals + the seed) costs about one trace instead of two. A model, UNVERIFIED on a card: if the second trace (49.7 ms)
and half the host verify work (3.6 ms) go away, the 8-live round falls from about 159 ms to about 106 ms; a seat commits 0.85-0.99 of its tokens; the committed rate per seat moves by
0.85 to 0.99 x 159/106 = **x1.28 to x1.49**. The bar for a GO is +10%.

A lone user today runs the 4-row per-request engine (85 ms a round, 2.5 tokens a round, 29.4 tok/s), slower than not speculating (32.6 tok/s). Two live users in one padded 64-row
T16 block round in 74 ms.

## 2. The policy (host side, built)

`QWEN_FAST_OCTO=off|live|alternate` (default off; any other value, an empty one included, is a configuration error), `QWEN_FAST_OCTO_MIN_LIVE` (default 6).

* `serving_packed_step.PackedStep(blocks, per_block_widths=True, octo=<block>, octo_state=<OctoState>)`.
* `proposal_groups` (called by the worker hook before drafting) plans the coming round's shape. With at least `min_live` live seats, every live request bound to the octo block, and the
  block's own per-member rules holding (`block_rows_for`: the padded count and idle-segment rules, tokens left, the extent block's frontier range, page 0, the K/V tile rows), the round is
  **one group**: the octo block, every live request, 8 rows. Otherwise the groups are exactly the two-M3-block ones, as before.
* The drafts do not change. The pair and quad draft passes run per M3 block (`QWEN_FAST_QUAD_DRAFT_BLOCKS=2`); a seat's 8-row ticket is its draft cut to 7 proposals on the host
  (`GreedySession.propose(max_rows=8)`, the cut a narrowed ticket already takes).
* The hook passes the set of requests its budget narrowing would cut at 8 rows (`FastWorkerHook.octo_blocked`); one such request keeps the round on the M3 blocks, where only its own
  block narrows, instead of sending eight seats down the sequential step.
* `route_octo` (the step) runs the round on the octo block when every entry is an 8-row ticket of a request bound to it, else on the M3 blocks and the sequential step as ever. A round whose
  membership changed between the drafts and the step is narrowed (S2 D1) and served sequentially: the same path an M3 round takes.
* Each switch of shape bumps the fixture write epoch (`announce_shape`, at the plan, before the drafts' fence window, as D0's `announce_round`); the step's own check finds it already noted.
  Every other block's deferred commits are flushed before a round's verify (the ordering invariant of `packed_device_rounds`). The octo block is a third block of the host-gap pre-stage
  (`verify_prestage.engage_two_block` over all three).
* `alternate` switches shape on every eligible round. The turn only advances when an eligible round ran as planned (`OctoState.finish`), so planning twice (an early draft discarded and
  drafted again) advances nothing. Both arms of an alternate boot pay a full pre-stage per switch; the `live` boot against the flag-off parent is the read without it.

### Why 6 and not 5

The design asks for 5 or more live seats. An idle segment writes its K/V through an all-zero page table to physical page 0, and page 0 has two 32-row tile rows; the chained K/V write
(`verify_trace_t2.kv_conflict`) allows one writer per (page, tile row), so a block holds at most two idle segments (`PackedVerifierEngine.MAX_IDLE_SEGMENTS`). Eight segments less two
idle is six live. Five live leaves three idle segments, and the block refuses a `padded_min_users` below `users - 2` at construction. `QWEN_FAST_OCTO_MIN_LIVE=5` is therefore refused at
attach until a third idle target exists (a card question); the default is 6. At 5 live the round is today's: block A packed, one seat on the per-request engine.

## 3. Exactness

Only the target's own argmax is ever committed. The octo round is the M3 round at another row geometry: the GDN state commits at the accepted prefix (the block's per-user commit
traces, 8 users x prefixes 1..8), K/V rows past the frontier are overwritten later, the drafter's history takes verified features only (the block's taps at the segment's eight rows), every
shape switch bumps the fixture epoch, and native GDN slot 0 stays untrusted after a packed step (`verifier_engine.note_packed_step`). `QWEN_FAST_SINGLE_GATEUP=1` is required (one MLP arithmetic at
M = 64 in every shape, exactness E1). The proof is by job: E1 and E2 compare every user's text with its solo run while the rounds alternate.

## 4. Memory: the choice and why

The third block is one more 64-row block: about 224 MB per DRAM bank (the measured need of a packed block), eight banks a chip, 1.79 GB, plus the trace region it captures into (the verify
trace, 64 GDN commit traces, 8 projection and 64 slide traces; the region goes from 512 to 640 MiB, 0.13 GB).

Option A, taken: **lower the KV pool and raise the trace region, in new gate-only profiles.** 19,968 blocks become 16,500: 3,468 blocks x 557,056 bytes = 1.93 GB, the block and the larger
region with 5 MB to spare, and the pool still holds four full 262,144-token windows ((16,500 - 8) x 64 = 1,055,488 tokens; `QWEN36_MAX_TOKENS_ALL_USERS` follows, as the request contract demands).
The pool is tight, not generous: the largest request the contract admits (a 253,920-token prompt with the answer room the window leaves it) reserves 4,097 blocks, so four are resident at once
with 111 blocks to spare (`test_octo_profiles`). The conversion is known, the profiles are gate-only, and the production pool is untouched.

Option B, refused: replace block B with the octo block (memory neutral). Block B serves every round at 4 or fewer live seats whose users sit in slots 4..7 and every fall-back, so the M3
path would no longer be the control the octo arm is measured against.

`scripts/ci/make_octo_profiles.py` generates the twins and refuses a parent whose pool or trace region differs from what this plan was derived under. M1 reads the ledger against M0 (the same plan
on the parent): free DRAM at 8 live at least 3 GB per chip, the floor at least 1 GB, the trace region's largest free block at least 44 MB, and not more than 0.2 GB below the parent's.

## 5. The device pieces: built, and what each card job must establish

`serving_octo.DEVICE_PIECES` lists them and `octo_admission` refuses by name while one is unbuilt (`device_gaps`; `BUILT` declares them, and the GDN piece is also read from the module's own limit).
All five are built on the host; each has CPU tests, and none has run on a card.

1. **attach-build.** `serving_runtime.attach_combined_runtime` builds `PackedVerifierEngine(shape=octo_shape, pool_slots=0..7, defer_capture=True, padded_min_users=min_live)` last, over extent storage the pool lends
   for the (8, 8) shape (`packed_shapes` gains it; `packed_replicas` stays the M3 pair's), and captures it with the two M3 blocks in `complete_blocks_two_phase` (every block's warm and fixture, then every capture, then
   every finish: the construction order the pool's holes require). `carries_in_place` is proven for it too, the step is bound to it with its `OctoState`, `admit_blocks` and the extent check include it, and the
   arrival placement is still by M3 block (`place_blocks`). Tests: `test_serving_runtime` (the attach on its fakes with a third block), `test_octo_block` (the real block at 8 x 8 on the fake device).
2. **gdn-batch.** `gdn_user_batch_tp.MAX_USERS` is 8 (the unpinned TP4 sibling; the pair's `gdn_user_batch` is hash-pinned at 4 and untouched): one launch is 8 users x 12 heads = 96 of the 110 cores, disjoint
   contiguous shares, one wave, the same kernels per worker. The first four users sit on the cores a four-user launch uses. Card question: job Q2 (`gdn_tp4_card_test.py --users 8 --rows 8`: one launch == eight per-user
   launches bit for bit, the head-slice equivalence against the pinned pair launch, the native twin), watcher pass first. Note which launch it is: K5-A (`gdn_seq_block`, what the M3 blocks run under
   `QWEN_FAST_GDN_SEQ_BLOCK=1`) needs every user at 16 rows, so the octo block runs the SERVED batched launch (`gdn_user_batch_tp.execute`, rows as a runtime argument); the octo and M3 recurrences are
   the same arithmetic only through K5-A's qualification against that served launch (the K5 evidence) and E1/E2, not through Q2, whose K5 section is refused at 8 x 8.
3. **attention-8row.** **No kernel and no graft change.** K64j's flag sets are per call: 0x23 (tail | share | extent) is the M3 bundle of two eight-row groups, and 0x21 (tail | extent) is the same program at ONE
   entry (`pooled_attention_replay.mode_flags` drops the share bit when a bundle has one entry; the q-slice is illegal at one KV head). CB1 qualified 0x21 at G8B2 and G4B3 on one KV head and the harness ran a
   one-entry bundle's skip; what no card has run is the bundle of ONE eight-row group (G8B1), and at one entry the cores per entry (16), the chunk split and the reduction tree are each G8B2 entry's, so an octo row is
   the M3 row at another launch geometry, not another sum - an argument, which Q1 measures. The code: `extent_attention_octo_tp.py` (the octo segment reader and the packed constructor, an unpinned twin: the four-card
   reader's bytes are held by the evidence record and stay unedited), dispatched by the fold twin for a block of eight-row segments only (so an M3 attach never imports it), bound whenever `QWEN_FAST_OCTO` is set
   (`tp_addresses._octo`); `ServingBufferPool` lends the (8, 8) storage (`extent_bundle_entries`), `attention_block_fold_tp` accepts bundles of one (a data-movement index map, shown byte for byte equal to the served
   composition at eight one-group segments); `QWEN_FAST_TP4_SDPA` is refused beside the flag. The four-card flag set is 0x23 / 0x21, not the pair's 0x27 / 0x25. Card questions: **Q1w, Q1a, Q1b** (`k64j_card_b.py
   --kv-heads 1 --octo`: G8B1 0x21 == its compile-time twin 0x1 at every family and start, one trace over more than 50 families, the stale-writer zone, and K2, the native one-row decode against the one-entry call,
   row by row), and the extent audit of the audited attach (A1) for the mask and staging at eight segments. The reader-harness `--octo` (CB2b's R1/S/R2/R4 through the real reader class) is NOT built; the audit and the
   strict text gates stand in for it.
4. **publication-8row.** `complete_blocks_two_phase` skips the publication warm for every block after the first except one marked `keep_publication_warm` (the octo block): its plan is 64 packed shapes of which 32 no M3
   warm ran, and they compile at attach, not in the first round after a shape switch. Card question: A1/H1 (no program compiled on a switch, `octo_judge`).
5. **fused-third-block.** The fused commit builds over the third block (8 projection and 64 slide traces; `test_octo_block.FusedThirdBlockTests`) and the pre-stage includes it. Card question: A1/E1.

### What the audit of the verify path found beyond the five pieces

* **GDN glue at eight-row users runs the served path.** The half-tile kernels (V2 split/merge, V1 block conv) and the T2 packed conv windows move 16-row users; an 8-row user is a quarter tile. At the octo block those
  sites take the pinned (served) path - exact, the same arithmetic as before the levers - and say so with `[PINDIAG] tp4 vglue octo 8-row served path` (a distinct marker: a gated arm fails on `FALLBACK`, and
  this is the block's known state). The octo pass therefore pays those launches (V2 saved 4.7 ms and V1 3.8 ms per M3 pass in `docs/tp4-vglue.md`; the packed windows an unmeasured amount) that the estimate in
  section 1 did not include. A quarter-tile glue is the follow-up if the exactness jobs pass and the gain is short; it is a new kernel mode and a new qualification, not built here.
* **The audits at the octo block.** `QWEN_FAST_TP4_VGLUE_AUDIT` and the T2 windows audit read lever entries the octo block does not have (its glue sites run the served path); the vglue audit is told (`served_only`) and the
  windows audit is off because no packed window engaged. The M3 blocks keep both. Every other audit of the audited twin (T1 sampler, extent, fused commit, unit-major reduction, prestage) is generic in users and rows.
* **Attention is eight launches of one entry per layer.** The two M3 blocks make four launches of two shared entries each; the per-user KV bytes are the same, so the octo pass's attention equals the two M3 passes' total.
  The estimate "the second trace disappears" holds for the dense weights, the collectives and the sampler; it does not for the attention, which grows with context. P1 at 32k and above reads it.
* **Memory is the named risk.** The plan sizes the third block as one more M3 block. The octo block holds eight per-user state sets (checkpoints, retained entries) where an M3 block holds four. A reading of the
  allocations (not a measurement) puts the difference near 0.4 GB a chip, against a plan margin of 5 MB. A1 may fail at the block's construction on an allocation; M1 reads the ledger and the twins are regenerated
  from the measured need (`make_octo_profiles.MEMORY`).
* **Exactness of the K/V write.** An eight-row user straddling a 32-row tile row (start % 32 > 24) is the same chain of single-row read-modify-writes as a 16-row user; the T2 chained writer, `kv_conflict` and the idle
  cap of two are generic in users and rows. The reduction order of the 4-way ring depends on the tile column, not on which user sits in a tile (`tile_collective_tp`).

### The attach evidence

The extent path is admitted by `packed_any_admission` against a pinned four-card evidence record (CB1, CB2a, CB2b at G8B2). The octo block is a new bundle geometry (G8B1) that no record covers. A **gate** arm
needs no new record: the profiles carry the gate marker (`QWEN_C2_GATE_PROFILE=1`), which turns a missing record into a logged UNQUALIFIED line (and the octo admission logs its own, `[OCTO] UNQUALIFIED`, six lines).
A **traffic** profile would need a new record for the 8-row bundle, taken on a card (Q1a, Q1b), before anything else; `admit_blocks` checks only `extent` and `runtime_extent`, so it would not notice an octo block - the
octo judge reads the flags (`extent replay engaged segments=8 flags=0x21,...`, the factory's `runtime-extent entries=1 kv_share=false`) and refuses a log of M3 programs under an octo flag.

## 6. The lone-user padded round (`QWEN_FAST_SOLO_PACKED=1`)

The idea: a lone live user (and two or three in a block) take the padded 64-row T16 block instead of the 1/2/4-row engines. Two or three live users are the padded rounds the image already serves
(`QWEN_FAST_PADDED_BLOCK=1` is in the image ENV). A **lone** user in the 4-user block leaves three idle segments, and page 0 holds two: the block refuses `padded_min_users=1` at construction. So the lone
user needs the same third idle target as five live seats in the octo block. The D0 one-user block (`QWEN_FAST_SOLO_LANE`) is the other answer, and is refused beside two M3 blocks. The flag is admitted
only when `idle_capacity() >= 3`; the policy itself (`proposal_groups` routes any group whose block `pads` the count) needs no new code. The lone `coding` stream and the drain (live 8 to 1) are the
tests (L0, L1).

## 7. Gates (references/tp4-octo-jobs)

Templates only; no tag is pushed from this branch. ORDER.txt carries the order, the dependencies and the NO-GO rule. In short: Q1w/Q1a/Q1b the K64j qualification at one eight-row group per bundle and Q2w/Q2 the GDN launch at eight users (one card each, first), A0/A1 the audited attach (the control against the arm,
`octo_compare.py`), E1/E2 exactness while alternating (every concurrent8 and staggered user equal to its solo run), H1-H3 hang shapes (three consecutive completions), M0/M1 the memory ledger,
P1-P4 the paired timing (`octo_judge.py --verdict`: GO at >= +10% committed tokens per second per seat at 8 live, aggregate and median pair, every text exact), R1-R3 on engine reuse, L0/L1 the lone user.

The smoke rules (`octo_judge`, called by `c2_smoke_check.check`): (a) a flagged arm logs its admission once and at least 16 rounds with `shape=octo`, written after the octo block's own round counter moved (a
mounted change is not an executed change); (b) under `alternate`, at least 16 counted rounds of each shape, strictly alternating; (c) no program compiled on the first round of a shape after a switch
(`[OCTO] programs`), and the counter must be readable; a flag-off arm logs no `[OCTO]` line.

## 8. Files

`packed_shapes.octo_shape`; `serving_octo` (flags, admission, policy state, device pieces); `extent_attention_octo_tp` (the octo extent readers); `k64j_card_b.py --octo` and `gdn_tp4_card_test.py --users 8 --rows 8` (the qualification harnesses); `octo_markers`, `octo_judge`, `octo_compare` (the log, the judgement, the arm against its control);
`serving_packed_step` (octo routing), `serving_worker_hook` (the blocked set), `serving_runtime` (the admission and the tripwire); `make_octo_profiles` (the gate-only twins); tests `test_octo`, `test_octo_block`,
`test_octo_judge`, `test_octo_profiles`, `test_tp4_octo_jobs`.
