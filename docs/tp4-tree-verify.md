# tp4/stream-1: two-candidate (tree) verify, scoped

Status: **design plus a CPU prototype of the part that can be settled on a CPU.** Nothing here has run on a card, no flag was added to serving, and nothing in `scripts/ci` imports the prototype
(`tree_verify.py`). Every gain below is an ESTIMATE from the earlier simulation (short prompts); every device cost is an estimate from the measured figures of the documents it cites.

Owner rule, unchanged: **strict exactness**. Greedy output stays byte-identical to the target's own greedy decode; speculation changes how many tokens a round commits, never which.

## 0. The verdict, first

* **The accept/commit logic is settled and tested** (section 2): given several candidate chains for one round, the round commits the chain that emits the most tokens (ties to the primary), the session commits it as
  it always commits, and the output is the target's greedy continuation whatever the candidates were. `tree_verify.py`, `test_tree_verify.py` (19 tests: exactness over 40 random runs with branches winning,
  parity with plain commit for one candidate, end of text, budget caps, a toy block that proves the state routing, the planner).
* **The device half is not a day's work, and the part that decides its price is not the verify.** The verify of a second chain is almost free (rows are nearly free; the pass is what costs, docs/tp4-octo.md
  section 1) and an ordinary segment can carry it. What is hard is **adopting the winner when it is not the request's own segment**: the GDN state, the K/V rows, the drafter's feature publication and its K/V
  history are all committed by traces bound to the segment's own slot (section 3). The design that needs no new kernel (D1, sibling segments, section 4) moves all of that between slots; the design that
  keeps everything in one segment (D2, an in-segment tree, section 5) needs a tree-aware GDN scan, a tree mask for the K64j extent attention and a K/V row remap, all new kernels. Rough size: D1 about three
  engineer-weeks and two card windows, D2 more and riskier (ESTIMATES, section 8).
* **It pays only where the block has free segments: one to three live seats per block, i.e. the lone-user lane and the pair.** At six or more live seats the octo block is full (8 users x 8 rows) and an M3 block
  with four live seats has no free segment either. That is exactly the shape of the owner's "ninth fast lane" and the wrong shape for the eight-seat throughput target.
* **Whether to build it is a measurement that needs no device work:** `QWEN_FAST_LOOKUP_SHADOW=1` on the lookup arm of the tau lab logs both candidates of every round, and `tree_gain_report.py` replays them against
  the committed continuation (section 7). The card plan (docs/tp4-stream-1.md) runs it in the first pair. Rule proposed there: stop here unless the walk (the lower bound) shows the better-of-two above the run's
  rate by at least 3% on the own-session set.

## 1. The round today (what a tree has to fit into)

| | today | file |
| --- | --- | --- |
| Row layout | a verify block is 64 rows = users x rows_per_user: four 16-row segments (M3, `packed_shapes.m3_shape`) or, beside two M3 blocks, eight 8-row segments (octo, `serving_octo`, `OCTO_USERS/ROWS_PER_USER`). Row 0 of a segment is the seed (the last emitted token), rows 1.. the drafter's proposals. A lone or paired round is the same trace with the other segments staged idle. | `packed_shapes.py`, `packed_verifier.py` module docstring |
| Segment <-> slot | segment u is bound to pool slot u **by carry identity**: the verify trace restores segment u's GDN state from slot u's carry inside the trace (`carries_in_place` under `QWEN_FAST_VERIFY_T1`: read in place, never written), and segment u's commit traces DMA the accepted prefix state "straight back into the same carry". Both addresses are baked at attach. The octo block shares the slots (and carries) of the two M3 blocks. | `packed_verifier.py` ("WHICH ROWS ARE WHOSE"), `segment_of` |
| Attention | per user a K64j extent launch (flags `0x21`/`0x23`) over the user's page table: the rows attend to the paged cache below the segment's start and to the block's own rows through the K/V the chained writer just wrote, under a mask the trace builds from the segment's positions word (causal inside the final 256-key chunk). | `docs/tp4-octo-attn-bundle.md`, `sdpa_multi_mask_tp`, `extent_attention_*` |
| K/V writes | the page-parallel writer writes row r of segment u at (page table u, start u + r); exact only while **no two segments write one (page, tile row)** (`verify_trace_t2.kv_conflict`; page 0 has two tile rows, so two idle segments fit, `MAX_IDLE_SEGMENTS`, and every NP1 scratch page adds two, `IDLE_SEGMENTS_PER_PAGE`). | `packed_verifier.idle_inputs`, `docs/tp4-np1-fastlane.md` section 3-4 |
| GDN (48 of 64 layers) | a recurrence over the segment's rows in order; per-prefix checkpoints (users x 48 sets); `commit_user(segment, prefix)` replays one of `rows_per_user` commit traces that copy checkpoint[prefix] into the carry. Row r's state depends on rows 0..r-1 of **its own segment only**. | `gdn_records.RetainedGDNBlock`, `packed_verifier.commit_user` |
| Accept | host: `greedy_verify.select_prefix(proposals, predictions)`; `GreedySession.commit` publishes `state_rows = accepted + 1` and emits `accepted` proposals plus the target's correction; `accept_limit` caps rows at or past a family end. | `greedy_verify.py`, `greedy_session.py` |
| Drafter | DFlash2 proposes 15 rows from the target's features of the last committed prefix; `publish(prefix)` advances the drafter's own K/V history from the **verify block's feature taps of that segment** (five `(1,1,64,5120)` taps; the fused commit's per-segment RoPE tables, deltas and slide traces). Its top-16 candidates per position exist at the draft output (`docs/tp4-next1.md` section 2, W3 MR). | `dflash_request_runtime`, `fused_commit(_tp)`, `packed_verifier.features` |
| vLLM | the scheduler is handed `ticket.tokens[1:]` as the request's draft ids (`serving_vllm_contract.draft_token_ids`), admission compares the scheduler's tokens with the ticket, and the runner returns the **committed tokens as `sampled_token_ids`**; the scheduler takes the rejected count as scheduled minus generated. | `serving_vllm_contract.py`, `serving_runner_bridge.py` |

Measured basis (docs/tp4-octo.md, docs/tp4-next1.md): a 64-row pass costs 49-50 ms whether two or four seats are live ("the rows are nearly free, the pass is what costs"); the octo block's trace is about 55 ms with
the glue levers; a lone user on the NP1 lane runs 61.2 tok/s in the s5 window (about 5.5 tokens a round over a round of about 90 ms).

## 2. The accept/commit logic (built: `tree_verify.py`)

Candidates are linear chains on one seed: index 0 is the **primary** (the chain the scheduler was handed), the others branches. Each is verified as a chain of its own, so each is judged exactly as a one-candidate
round judges it (`select_prefix`), with the cuts `GreedySession.commit` applies to a ticket of its width (the budget left, the commit cap).

* **Exactness.** A chain's accepted proposals equal the target's predictions and its correction is the target's prediction at the first row that differs, so the tokens it emits are a prefix of *the* greedy
  continuation of the seed. Any two chains' emitted tokens agree on their common length; whichever chain wins, the output is the target's own. The rule only decides how many tokens the round commits
  (`test_the_emissions_of_all_candidates_agree_on_their_common_prefix`, `test_the_emitted_tokens_are_the_targets_greedy_continuation_whatever_is_offered`: 40 random streams, candidate sets of one to four chains
  of random quality and width, to the budget; and across an end-of-text row).
* **Choice.** The winner emits the most tokens; ties go to the lowest index, so a branch wins only by **strictly more** and `copy_back` is exactly "the winner's state and K/V are in a sibling segment".
  With one candidate the round is today's, call for call (`test_one_candidate_leaves_the_session_where_plain_commit_does`: tokens, position, seed, epoch, block and accepted counters).
* **Commit.** `commit_tree` makes the winner the session's pending ticket (a new epoch, as `prompt_lookup.replace_proposals` does) and calls `GreedySession.commit` with the winner's predictions and the
  winner's publication callback (`publish_for(branch)`); it checks that the session decided the prefix the tree selected. The drafter publishes the winner's features at the winner's prefix.
* **Width.** A branch may not be wider than the primary (`build_candidates` refuses it): vLLM's rejected count is scheduled minus generated, so a winner that emitted more than the scheduled width would corrupt the
  scheduler's token count. A branch equal to an earlier chain is dropped; one that shares a first row is kept (the tree still wins after the first difference).
* **The state routing, proved on a toy block** (`BlockSim` in the test): each segment forks the request's recurrent state, writes K/V into cells of its own, and the round commits the winner's state at its accepted
  prefix and, for a branch, copies the winner's K/V rows of that prefix into the request's cells. After every round the request's state and K/V equal a plain decode's of the same emitted tokens; the same test
  with the copy-back left out fails (`test_the_copy_back_is_needed...`). That fixes the off-by-ones the device design must keep: the winner commits `state_rows = accepted + 1` rows (seed through the last
  accepted proposal), so K/V rows 0..state_rows-1 of the winner move, and the correction token is **not** consumed (it is the next seed).
* **`plan_branches(segments, live, wants)`** gives branches only the block's *free* segments (round robin in priority order, at most `MAX_CANDIDATES - 1 = 3` extras a request): a full block plans nothing.

What the prototype does not touch: any device buffer, trace or kernel; the drafter; vLLM; the packed step's staging. It is the specification the device half is checked against.

## 3. What the device half needs, by component

| Component | D1: sibling segments (section 4) | D2: in-segment tree (section 5) |
| --- | --- | --- |
| Rows / positions | a branch is an ordinary segment: same start, own tokens (host staging) | rows `[seed, A1..An, B1..Bm]` in one segment; RoPE positions per row (host-staged table: easy) |
| Attention mask | none (causal per segment) | **new**: B rows must not see A rows; the mask the trace builds from the positions word is causal-contiguous; a staged mask tensor or a new mask kernel, and a new capture of the verify trace |
| GDN recurrence | none per segment, but the branch needs a **fork of the request's carry** before the round (240 tensors a chip, about 67 MB: 48 layers x (0.8 MB fp32 state + four 164 KB conv taps)) | **new kernels**: B1 must start from the state after the seed, not after An; the conv window likewise; a tree-aware multi-token scan (the W3 WC kernels, the nearest thing, have never been through a compiler) |
| K/V rows | the branch writes its own cells (a **copy-on-write page**: a scratch page holding a copy of the page the round writes, past vLLM's pool as NP1's scratch is); the winner's rows are copied back into the real page | rows of B land at slots after A's: after a B win, **move B's accepted rows to their logical slots** (in-page row move) |
| Commit | winner state: the commit trace of the winner's (segment, prefix) writes its own slot's carry; then **carry move** to the request's slot (a trace per ordered slot pair) | commit trace per tree-row prefix (checkpoints at tree rows) |
| Drafter publication | **cross-slot**: the fused commit's deltas/slides and the five feature taps are per segment and write the slot's own drafter history; a branch win has to publish *segment s' features into slot s' history* and carry the drafter K/V history rows back (or publish through the request's segment with the branch's tap rows copied in) | feature rows remapped on the host (`features(segment)` row index): small |
| Host staging | per round: branch tokens/pages/positions/start words for free segments; prestage keyed diff must know them | per round: tree tokens, positions, mask |
| New captures | carry-fork and carry-move traces (24 for an M3 block), COW page copy and K/V copy-back ops (about 32 small ops each, host-launched, page ids vary per round) | a second verify trace (the tree mask), tree GDN kernels, commit traces |
| Admission | scratch pages (the K/V cell rule: each branch segment owns a page of two tile rows, enough unless its 16 rows cross a page: then two), a free slot **and** a free segment | none beyond the row budget |
| Gate set | exactness of the fork/COW/copy-back (digest of the request's carry, K/V rows and drafter history against a linear decode, per round, on cards) | kernel qualification (the long pole), tree-mask exactness |

## 4. Design D1: sibling segments (recommended first, lone user and pair only)

**Layout.** A request with a better-than-nothing second chain takes a free segment (and with it that slot's carry, K/V cells and feature taps) of its own block for the round. The block's trace is the one it
already replays: the branch segment is staged like any live segment, with the request's start position and seed, the branch's tokens, a page table of its own and a positions word of its own.
The segment's rows are then verified exactly as a one-candidate round would verify them: **no kernel and no trace of the verify changes.** `plan_branches` decides who gets which free segment
(the user with the longest lookup match first).

**One round, in order.**
1. *Plan* (host, with the drafts): the primary chain from DFlash2 (or the lookup, per the existing policy), branches from the other source (the lookup's chain when it differs from DFlash2's; later a DFlash2
   top-2 swap at its least certain row, which the drafter's top-16 outputs allow without a second pass: DFlash2 is a block drafter, so the rows after a swapped row are not conditioned on it).
   vLLM is handed the primary's tokens only.
2. *Fork* (device): copy the request's carry (48 layers x state + conv taps) into the branch slot's carry; copy-on-write the page(s) the 16 rows will write into the branch's scratch page(s).
   A round whose rows straddle a page (start offset above 48 in the 64-token page) takes two scratch pages or runs without a branch (about a quarter of rounds; ESTIMATE).
3. *Verify* (device, unchanged): the block replays; the readback gives every segment's predictions.
4. *Decide* (host): `tree_verify.select_branch`.
5. *Commit*: the winner's `commit_user(segment, state_rows)` as today. If the winner is the primary, **nothing else happens** (the branch's carry, cells and scratch page are garbage that the next fork overwrites;
   the idle rule that a segment never writes a live slot's carry holds because the branch slot is free). If a branch won (`copy_back`): the carry move (branch slot -> request slot), the K/V copy-back of rows
   0..state_rows-1 from the scratch page into the real page, and the **drafter publication from the branch's segment** (section 4.2).
6. *Admission and abort*: a round that fails before the commit leaves the request's carry, cells and drafter history untouched (everything the branch wrote is in the branch's slot and scratch page): the abort path
   is today's.

### 4.1 K/V

NP1 already established the pattern for cells outside vLLM's pool: `idle_scratch` allocates extra KV blocks on the worker, claimed at the attach, never handed to a request, zero-filled
(`docs/tp4-np1-fastlane.md` section 4, 557,056 bytes a chip a block). A branch needs one such block (two for a straddling round) per free segment it may take: at most three on an M3 block with one live seat.
The copy-on-write copy of the real page into the scratch page is needed because the extent reader of the branch segment reads `[0, E // 64)` through the branch's own page table, and the writers work on whole
tile rows: the rows of the page below the round's start must be readable and must survive the writer's read-modify-write. The copy and the copy-back are page-table-driven data movement over 16 attention
layers x K and V (about 32 small ops each way on a chip). **They cannot be traced** (the real page id changes every round), so they are host-launched: an ESTIMATE of 1-2 ms of host time per branch per round
for the copy-on-write and the same on the rounds a branch wins. The host window is already the floor in 91% of 8-live steps (`docs/tp4-octo3.md`); at one or two live seats the round is 72-90 ms with a
host gap of 15-20 ms (`docs/tp4-np1-fastlane.md` section 1), and the copy-on-write can be enqueued before the verify (overlapping the previous round's tail) while the copy-back, which follows the
readback, cannot: that is why the cost is charged to the rounds a branch wins. This is a data-movement kernel candidate (one launch for all layers is not possible while the caches are separate tensors).

### 4.2 The commit pipeline is per segment, and that is the real cost

The target's GDN commit (the `(segment, prefix)` traces), the fused commit's projection and slide traces (per segment, `FUSED_COMMIT_LIVE_BANKS`), the feature taps `features(segment)` and the drafter's
K/V history banks are bound to the slot of the segment that produced them. When a branch wins, the request's state must end up in the request's slot:

* target GDN: commit in the branch slot, then a **carry move** trace per ordered slot pair of the block (12 for four segments), about 0.3 ms of device time a move (67 MB read and written a chip);
* target K/V: the copy-back above;
* drafter: either (a) publish from the branch segment into the branch slot's drafter history and then move that history (the draft K/V history banks of five layers and the cached projection) into the
  request's, or (b) point the request's publication at the branch segment's feature taps and deltas (`features(segment)` already takes a segment; the fused commit's traces are the obstacle: their destination
  is the segment's own slot). Neither exists. (b) is the smaller change if the slide traces can be captured with a source/destination pair per ordered slot pair, as the carry moves are.

This is the part to prototype first on cards (phase 1, section 8): a gate-only **forced-branch** switch in which a branch segment carries the *same* chain as the primary and always wins, so every round
adopts across slots, and the request's state afterwards is checked against the control run bit for bit. It isolates the cross-slot adoption from every question about candidates.

### 4.3 vLLM and the scheduler

The scheduler sees only the primary's tokens. The runner returns the winner's emitted tokens as `sampled_token_ids`; the scheduler counts the rejected as scheduled minus generated, and the winner is never
wider than the primary (enforced by `build_candidates`). The admission comparison (`admit_scheduler_output`) is made against the primary ticket, so it is unchanged. The KV blocks vLLM allocated cover the
request's 16 rows whichever chain wins. To be held by a CPU test when the device half is built: the bridge round trip with a branch winner (`VllmPathTests` of `test_prompt_lookup` is the pattern).

### 4.4 Cost model (ESTIMATES; the card gate replaces them)

Per round with one branch, extra: carry fork about 0.3 ms device; copy-on-write 32 ops, 1-2 ms host; on a branch win (the fraction the shadow readout measures) the carry move 0.3 ms, the copy-back 1-2 ms and
the drafter adoption (unknown, section 4.2). A branch segment costs no extra verify time when the block has a free segment: the pass is the same trace. So at a lone user the price is a few percent of a 90 ms
round on the rounds that carry a branch, against a gain of (E[max(a_A, a_B)] - E[a_A]) tokens on them. The earlier simulation's +8% overall is the *ceiling* of that product; the cost can eat half of it.

## 5. Design D2: an in-segment tree (not first)

One segment, rows `[seed, A1..An, B1..Bm]` (n + m <= 15; each chain shorter than today's 15, a loss at tau 5.5 where only a few percent of rounds accept more than 8). It needs no free segment, so it works at any
live count that has room in a block, but it needs: a tree attention mask in the K64j extent program's final chunk (a staged additive mask tensor instead of the in-trace causal generator: a new verify capture),
RoPE positions per row (host), a tree-aware GDN scan (B1 starts from the state after the seed; the conv window likewise: new kernels, the W3 WC kernels' kind), per-row checkpoints for the commit at a tree prefix,
and, for a B win, an in-page row move of B's accepted K/V to the logical slots and a row remap of the feature taps. Everything stays in the request's slot, so the cross-slot adoption of D1 does not exist;
the price is kernel qualification, which is the long pole of this repo (see the W3 WC history in docs/tp4-next1.md). It is the right design if D1's adoption proves too costly and the gain at six or more seats
(where there is no free segment and rows per user are 8) is wanted: it is not, since an 8-row segment has no room for two chains.

## 6. Interactions

* **Octo (`QWEN_FAST_OCTO`, six or more live seats).** The octo block is 8 users x 8 rows: every segment is a request. No free segment, no branch; D2 has no room in 8 rows either. The tree is off at >= 6 live
  and the planner returns nothing for a full block, so the round is today's. An 8-row primary ticket (the octo cut of the draft to 7 proposals) is a valid primary for a branch on an M3 block only if the branch is
  no wider than it, which `build_candidates` enforces.
* **M3 blocks and `QWEN_FAST_M3_BLOCKS=2`.** Two blocks of four slots; new arrivals fill a block with exactly one live user first, so one or two live seats means one live segment per block and three free
  ones: the best case. At 4 live (two per block) one free segment per request. At 8 live nothing.
* **NP1 (the lone-user padded round).** Its idle segments are the free segments; a branch replaces an idle segment and needs a scratch page of its own where an idle one shares page 0's two tile rows
  (`MAX_IDLE_SEGMENTS`, `IDLE_SEGMENTS_PER_PAGE`). The K/V cell rule is the admission check: every non-primary segment, idle or branch, owns at least one distinct (page, tile row); a branch owns a whole
  scratch page (two tile rows: its 16 rows cross a 32-row tile boundary about half the time), two pages when they cross a page. `padded_refusal` already refuses a live table that names a scratch page.
* **Engine reuse.** Every slot holds a parked engine, and a parked slot counts as lent. A branch borrows a *parked* slot's carry, cells and taps for one round and leaves them as garbage; the next rebind
  rezeroes the slot's state, reseeds its K/V and rewrites its page table (R1 of docs/tp4-engine-reuse.md), so the rule that matters is that **no rebind of the borrowed slot can interleave with the round**
  (the engine thread runs rounds and rebinds one at a time) and that the borrowed slot's drafter device and its 2c-bound traces are never run by the branch (the branch uses the request's own drafter for its
  chain: it only needs the segment's verify-side buffers). The kill switch (`parked.off`) must also turn the tree off. `PARKED_FAULT` paths: a failed fork aborts the branch, not the round.
* **Exact resume (P0, `QWEN_PREFIX_STICKY_UNDROP`, `QWEN_PREFIX_WINDOW_CARRY`).** The carry is saved and filed at turn ends and the drafter-window rows are uploaded on the resumed turn; both read only the
  *committed* state. A tree round commits the same state a linear round of the winning chain commits (that is the toy block's invariant), so what the tier files is unchanged. The one new obligation:
  the K/V rows the host tier files from the decode-written blocks must be byte-exact, so the copy-back is a bit copy with a digest gate (the engine reuse audit's digest machinery is the pattern).
* **W2 / W3 WC / the fused commit.** WC runs per M3 block of 2 or 4 users and the block's shape does not change (a branch is one more live-looking segment of the same trace), but its audit compares per
  user and must treat a branch segment's checkpoints as another user's. The fused commit's per-segment tables are the section 4.2 obstacle.
* **Lever N / governor.** A branch changes no scheduling; a round with branches is the same trace. The deadline governor charges the same pass.
* **Memory.** Per chip: one 557 KB scratch block per branch segment (three per block at most); the carry-move and carry-fork traces' region (240 small copies each, 24 traces: ESTIMATE tens of MB of trace
  region, against about 80 MB audited free; the ballast ladder of the engine reuse gate is the way to find the floor); nothing in DRAM beyond that.

## 7. How much is it worth, and the measurement that decides

Let a_A be the rows the primary's chain is accepted for and a_B the branch's. A tree round commits `max(a_A, a_B) + 1` tokens against `a_A + 1`. The earlier simulation (short prompts, DFlash2's block plus a
lookup branch) put this at **+8% overall (ESTIMATE)**, against **+3%** for the lookup *replacing* DFlash2's rows by policy (also an estimate); the difference is the rounds on which the policy picks the wrong
chain. Those figures came from prompts of a few thousand tokens; on our own long sessions (tau 4.2) they are unknown, which is why the decision should not be taken on them.

The shadow readout settles it with no device work: the lookup arm of the tau lab logs both candidates per round (`QWEN_FAST_LOOKUP_SHADOW=1`, docs/tp4-lookup.md), `tree_gain_report.py` replays them
against the committed continuation, and the pair's `[TREEGAIN]` lines give the mean tokens per round of DFlash2 alone, the lookup alone, the policy as run and the better of the two, between a lower bound
(the walk) and an upper bound (recorded starts). The reading is self-checking (`dflash_replay_mismatches=0`: the rounds the run took from DFlash2 replay to the commits logged). **Proposed rule:**

* walk gain over the run below 3% on the own-session set: stop; the lookup policy is enough and the device half is not worth three engineer-weeks;
* walk gain 3-8%, recorded above 8%: build phase 1 (the forced-branch gate, section 8) only if the lone-user lane is the owner's next target, since D1 pays at one or two seats;
* walk gain above 8%: build D1.

The report also bins the rounds by the lookup's match length (`by_lookup_match`), which says whether a cheaper gate (branch only when the match is long enough to matter, so the fork cost is paid on few rounds)
would keep most of the gain, and counts how often the two chains share their first row (the second candidate then only matters after the first difference).

## 8. Build plan, gates, size (ESTIMATES)

| Phase | What | Where it runs | Size |
| --- | --- | --- | --- |
| 0 | the measurement above (built) | tau lab pair, 2 x 55 min | done on the CPU side |
| 1 | the **forced-branch gate**: a branch segment in a free slot carries the primary's own chain; the carry fork, the COW page, the commit in the branch slot, the carry move, the K/V copy-back and the drafter adoption run *every round*, and the request must equal the control run bit for bit (carry digests, K/V digests of the written rows, drafter history digests, text) | audited attach, 45 min, lone user and pair | 1-1.5 engineer-weeks (carry fork/move traces, page copy ops, publication from a sibling segment, staging) |
| 2 | real candidates: the planner, the lookup branch, `commit_tree` on the packed step, the scheduler round trip; the paired ABAB (control vs tree) at one and two live seats | timed pair, 2 x 45 min | 1 engineer-week |
| 3 | D1 beyond the lone user (pairs, rounds that straddle a page), the abort and kill-switch paths, memory floor | hang shapes, 3 x 15 min | 0.5-1 engineer-week |

Gate after each: exactness first (text equal to control on every paired turn; the audits above), then the rate. A phase-1 failure ends D1 before any candidate logic is wired.

## 9. Risks and open questions

* **The drafter adoption (4.2) is unsized.** It may force D2, or a cheaper lazy variant (on a branch win commit nothing and replay the winner's tokens as forced rows at the head of the next round, in the request's own
  segment): no cross-slot copy at all, but the next round's draft is conditioned on stale features, so it costs the round after a win. Not evaluated; the shadow log could measure the cost (the committed run after a
  branch win is in the log).
* **Bitwise equality of a segment across slots.** The design assumes a chain verified in a sibling segment gives the same rows as in the request's own (the bundle documents argue this for entries of a
  launch; the K/V page writer is audited exact against a served-writer shadow). Phase 1's forced branch is the check.
* **Host time.** The copy-on-write and copy-back ops are host-launched and the host window is the floor at eight seats; at one or two seats it is not, which is where the tree pays.
* **vLLM.** One assumption to hold with a test when built: the scheduler's bookkeeping when the runner's generated tokens are not a prefix of its scheduled draft ids (it only counts them).
* **The simulation's +8% is a short-prompt ESTIMATE** and its row budget was not the one a 16-row block gives two chains; treat it as an upper bound until the shadow readout has run.
