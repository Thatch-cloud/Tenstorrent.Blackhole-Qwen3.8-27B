# tp4/octo-t8: adaptive drafting rounds at eight seats, and the lone-user padded round

Status: the host side is built and tested on CPU; the flag **refuses to engage** (by name) until the device pieces below exist. Nothing here has run on a card. Every
card result this document asks for is UNQUALIFIED until it has run. Default off, gate only, byte-identical off.

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

## 5. What remains: the device pieces

`serving_octo.DEVICE_PIECES` is the list; `octo_admission` refuses while any piece is unbuilt, naming each. What the CPU work already established about them:

1. **attach-build.** No attach builds the third block: `PackedVerifierEngine(shape=octo_shape, pool_slots=0..7)` captured after blocks A and B inside `complete_blocks_two_phase`, over extent
   storage the pool lends for a (8, 8) shape, with `carries_in_place` proven. The REAL block runs at 8 x 8 on the fake device (`test_octo_block`): construction, 8 live rounds, 6- and 7-live padded
   rounds, per-segment commits at prefixes 1..8, the taps' row offsets, the shared carries beside a real M3 block, the boundary cap. That is host-path evidence only, with the reader's
   qualification patched in.
2. **gdn-batch.** `gdn_user_batch.MAX_USERS` is 4 (a hash-pinned source, `test_tp2_pins`; the TP4 sibling takes it): the batched GDN launch carries four users. Eight segments need a TP4 batch of
   eight (96 cores at 12 heads) or two launches of four. A card question.
3. **attention-8row.** K64j is qualified at G8B2 only (flags 0x27, bundles of two eight-row groups). An 8-row segment bundles as one group: `ExtentSegmentReader` refuses it ("qualified at G8B2 only"),
   and `ServingBufferPool` refuses extent storage for any other shape (`extent_bundle_batches(8, 8)` is `(1,)`; M3's is `(2,)`). One group per bundle runs flags 0x25 (no KV share). Qualifying K64j
   at G8B1, and the masks and the T2 K/V chains at 8-row granularity, is the riskiest part and a card question. The executed refusals are pinned in `test_octo_block`; when K64j is qualified those
   tests fail and this piece's text must change with them.
4. **publication-8row.** `complete_blocks_two_phase` sets `warm_publication = False` on every block after the first (the M3 plan is the same 71 shapes). The octo block's plan is 64 packed shapes of
   which 32 no M3 warm ran (segment offsets 8, 24, 40, 56 x prefixes 1..8): its warm must run at attach or the first octo round compiles them (the smoke judge refuses a program compiled on a shape switch).
5. **fused-third-block.** The fused commit of a third block (8 projection and 64 slide traces) and the pre-stage state of a third fixture: the host wiring is built, the device behaviour is unqualified.

## 6. The lone-user padded round (`QWEN_FAST_SOLO_PACKED=1`)

The idea: a lone live user (and two or three in a block) take the padded 64-row T16 block instead of the 1/2/4-row engines. Two or three live users are the padded rounds the image already serves
(`QWEN_FAST_PADDED_BLOCK=1` is in the image ENV). A **lone** user in the 4-user block leaves three idle segments, and page 0 holds two: the block refuses `padded_min_users=1` at construction. So the lone
user needs the same third idle target as five live seats in the octo block. The D0 one-user block (`QWEN_FAST_SOLO_LANE`) is the other answer, and is refused beside two M3 blocks. The flag is admitted
only when `idle_capacity() >= 3`; the policy itself (`proposal_groups` routes any group whose block `pads` the count) needs no new code. The lone `coding` stream and the drain (live 8 to 1) are the
tests (L0, L1).

## 7. Gates (references/tp4-octo-jobs)

Templates only; no tag is pushed from this branch. ORDER.txt carries the order, the dependencies and the NO-GO rule. In short: A0/A1 the audited attach (the control against the arm,
`octo_compare.py`), E1/E2 exactness while alternating (every concurrent8 and staggered user equal to its solo run), H1-H3 hang shapes (three consecutive completions), M0/M1 the memory ledger,
P1-P4 the paired timing (`octo_judge.py --verdict`: GO at >= +10% committed tokens per second per seat at 8 live, aggregate and median pair, every text exact), R1-R3 on engine reuse, L0/L1 the lone user.

The smoke rules (`octo_judge`, called by `c2_smoke_check.check`): (a) a flagged arm logs its admission once and at least 16 rounds with `shape=octo`, written after the octo block's own round counter moved (a
mounted change is not an executed change); (b) under `alternate`, at least 16 counted rounds of each shape, strictly alternating; (c) no program compiled on the first round of a shape after a switch
(`[OCTO] programs`), and the counter must be readable; a flag-off arm logs no `[OCTO]` line.

## 8. Files

`packed_shapes.octo_shape`; `serving_octo` (flags, admission, policy state, device pieces); `octo_markers`, `octo_judge`, `octo_compare` (the log, the judgement, the arm against its control);
`serving_packed_step` (octo routing), `serving_worker_hook` (the blocked set), `serving_runtime` (the admission and the tripwire); `make_octo_profiles` (the gate-only twins); tests `test_octo`, `test_octo_block`,
`test_octo_judge`, `test_octo_profiles`, `test_tp4_octo_jobs`.
