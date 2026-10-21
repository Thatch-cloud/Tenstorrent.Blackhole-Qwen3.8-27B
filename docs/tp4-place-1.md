# tp4/place-1: compact placement of a new request (QWEN_FAST_TP4_PLACE_COMPACT)

Branch tp4/place-1, on the tp4/s6 head. CPU only: nothing here has run on a card. Default off; with the switch unset every byte of the serving path is what it was.

## 1. Result in five lines

* Why live users sit in two blocks: a seat is chosen once, when a turn is admitted, and cannot change while it decodes; the users who FINISH are not chosen at all, so departures leave holes in
  whichever block they were in. Today's rule then makes it a little worse (section 2). Arrivals alone are already compact (4 + 1 at five live).
* The switch (section 3) is the best that an admission-time choice can do. It is exact by construction (it names a different pool slot; the slots are replicas) and a few lines.
* It is worth about +1.2 to +1.4% aggregate decode throughput (section 4), not the ~5% the round breakdown estimated: the lone-first rule is a small part of the splits, the departures are the rest.
* The rest needs a live user to MOVE between blocks (migration; upper bound +8 to +9%, section 5) or the octo pass to take a split round (+5 to +7%, section 5, no migration). Neither is built here.
* Card gate (section 6): an audited attach, and a pair read by the two-pass share of rounds, not by tokens per second (a 1% lever sits inside the noise of a timed pair).

## 2. Why 3 to 5 live users run two verify passes

Read from the code of this tree and from the round-breakdown numbers (tp4/round-breakdown, s5 rounds of 2026-10-19/20; docs/tp4-round-host.md for the ledger).

1. A round runs one verify pass per M3 block that holds a live user. The step partitions a round by block (`PackedStep` with `per_block_widths`, `proposal_groups`, `packed_device_rounds`); each block
   with a member runs its own 64-row trace, padded when it holds one to three (`QWEN_FAST_SOLO_PACKED=1`: `padded_min_users` 1), and the trace costs the same whether one or four of its segments are
   live (42.4 ms verify at 3.6k for one to four live). Block A is pool slots 0 to 3, block B slots 4 to 7. From six live the octo block runs one pass over all eight (`QWEN_FAST_OCTO_MIN_LIVE=6`).
   Live users in both blocks therefore cost two passes and two drafts: 119.6 against 65.3 ms at 3.6k, 142.0 against 68.5 ms at 64k, 165.9 against 79.4 ms at 128k.
2. A seat is a pool slot, bound at admission and fixed for the life of the request. `ServingBufferPool.acquire` takes it (`placement_slot`), or, with engine reuse, `ParkedEngineSet.place` over
   the serving slots (the same function, design rule R5). The request's engine borrows that slot's carry and the block's captured trace reads that carry (`PackedVerifierEngine.segment_of`
   finds the segment by carry identity), the draft history and K/V banks are the slot's own: nothing can be re-pointed while the request decodes.
3. The rule (`placement_slot`): the block holding exactly ONE live request first, else the fuller block that is not full (the first on a tie), the lowest free slot of it. It was written when a
   block with fewer than four live ran the sequential step ("a lone user is rare and a block runs packed whenever it can"). With padded blocks a block with one user costs a whole pass.
4. What is NOT the cause: seat hold or parked seats (a parked engine's slot is lent but not live in the placement predicate, `ParkedEngineSet.live`); the order of free slots inside a block (all four
   are replicas); the prefix store, host tier and exact resume (they key on tokens and pages, never on a seat: no `pool_slot` in prefix_window_carry or the registry); arrivals alone (they fill block A
   first, so 1 to 4 live are one block and five are 4 + 1).
5. What is: DEPARTURES. Turns end in an order nobody chooses, and each leaves its hole in its own block: five live as 4 + 1, one of block A finishes, four live sit in two blocks. Then the rule
   makes it worse in one case: when one block holds exactly one live user and the other two or three, today's rule sends the arrival to the lone user's block ((2,1) becomes (2,2), (3,1) becomes (3,2))
   where the fuller block would have taken it. The two rules differ ONLY there (pinned by test_place_compact over all 256 sets of lent slots).

How much each part is worth, from the Markov model (section 4, `scripts/ci/place_model.py`, fitted to the production live-count mix): share of rounds in two passes at 4 live, 32% today, 28% under the
fullest-block rule (an admission-time choice cannot do better, section 4), and 0% only if the order of departures were controlled. Measured production: 8.7% at 3 live, 43.8% at 4, 99.9% at 5 (five live need
two M3 passes under ANY placement). The model has 12%, 32% and 100%: it under-reads the 4-live splits and over-reads the 3-live ones, which says production turns are less memoryless than the model, not that
the cause is another.

## 3. The switch

`QWEN_FAST_TP4_PLACE_COMPACT`, strictly `0` or `1` (unset is `0`; anything else is a configuration error naming the flag). In `scripts/ci/serving_buffer_pool.py` only:

* `place_blocks(blocks, compact=None)`: None asks the environment (`place_compact_enabled`), a bool names it. On, the pool stores it and writes the engaged line once:
  `[PINDIAG] placement compact engaged over blocks [[0, 1, 2, 3], [4, 5, 6, 7]] (QWEN_FAST_TP4_PLACE_COMPACT=1): a new request takes the fullest block that is not full`.
* `placement_slot`: the lone-block preference is dropped; the block is the fullest of the open blocks (first on a tie), the slot its lowest free one.
* `ServingBufferPool.__init__` refuses the flag at any pool size but eight (the flag would do nothing), before it allocates anything.

`serving_runtime.py` is untouched (the call `pool.place_blocks(...)` is the one it always made; its source is pinned hunk by hunk by test_fused_commit, and every lever branch edits it, so this one leaves it alone). The module is in both
image copy lists already. No new image file: `place_model.py` and its test are host-side.

Interplay, each read in the code and held by test_place_compact where it can be on CPU:

* Engine reuse (`QWEN_FAST_PARKED_ENGINES`): `ParkedEngineSet.place()` and `pool.acquire()` call the same `placement_slot` (the set passes its serving-slot predicate), so a compact arrival takes the same slot
  either way; tested with every slot lent and three serving.
* Solo packed (`QWEN_FAST_SOLO_PACKED`, the lone-user fast lane): admits a lone user into the padded block wherever the block is; placement only decides which block. Compact placement is meant FOR
  this stack (padded blocks cost a pass for one user); without it blocks with fewer than four live run the sequential step and the effect is not modelled.
* Octo (`QWEN_FAST_OCTO=live`, six or more live): every block is full or the octo pass runs over all eight seats; the placement does not matter once the round is an octo round, and it is the same
  function before and after.
* Prefix store, host tier, exact resume, sticky sessions: seat independent. The carried drafter window is the request's capture, not a slot.
* `QWEN_FAST_LANE` / `pool.slot_order`: an explicit slot order still wins over placement (tested); not in any traffic profile.

### Exactness

The switch changes WHICH pool slot an admitted request takes and nothing a slot does: the same code, the same weights, the same device ops. The eight slots are replicas by construction (one allocation
loop; two blocks of the one M3 shape, captured against different slots), and a request's verify result in a padded or full round does not depend on its neighbours (the packed block's qualified
property: per-segment identical to the request's own sequential engine, idle segments on page 0 / the NP1 scratch, V1/V2/T1 audits per request). Every combination of live segments the new rule produces is one
today's rule produces after some order of departures (any subset of slots can be live today), so no (block, segment) pairing is new. The target's argmax is the only token ever committed; the draft shape (a quad when
all four slots of a block are live, pairs and singles otherwise) moves only tau and time. The gate (section 6) still compares text, per request, byte for byte, with the switch on and off.

## 4. Expected gain, recomputed

`python3 scripts/ci/place_model.py` (stdlib, 2 s; `--exhaustive` and `--simulate` add the checks below). State (a, b) = live users in block A and B; turns arrive at rate rho, take the block the rule names, leave
at rate 1 (one user at a time, uniformly); stationary shares by exact solution of the 25-state chain. The load is a mixture of rho fitted to the production decode-time share by live count, which it reproduces
(production 34.3 / 27.1 / 18.4 / 9.9 / 5.6 / 2.6 / 1.6 / 0.5 %, model 34.3 / 27.2 / 18.1 / 10.3 / 5.3 / 2.7 / 1.4 / 0.7). Walls are the measured ones (section 2); the throughput is the time-weighted
sum of live / wall, the occupancy held (placement does not change who is live).

| against today's rule | 3.6k | 64k | 128k |
| --- | --- | --- | --- |
| `fullest_first` (this switch) | +1.20% | +1.41% | +1.42% |
| share of rounds in two passes at 3 / 4 live, today | 13.5 / 35.2% | 12.1 / 32.4% | 12.0 / 32.2% |
| share of rounds in two passes at 3 / 4 live, `fullest_first` | 10.4 / 30.4% | 9.3 / 27.9% | 9.2 / 27.7% |
| `ideal_compaction`: every round at ceil(n / 4) passes (a free migration; 5 live stay at two) | +7.92% | +9.26% | +9.35% |
| `octo_on_split`: a split round of 5 live or fewer takes the octo pass (8 rows, tau x 0.9), today's placement | +5.25% | +6.18% | +6.13% |
| `octo_on_split` beside `fullest_first` | +5.99% | +7.04% | +7.01% |

* No admission-time policy does better than `fullest_first`: the best of all 4,096 stationary block choices in the model equals it in every state at rho 1.5, 2 and 3 (`--exhaustive`; 150 sampled policies
  are in the test). The reason is structural: the arrival's block can only pick which hole to fill, and filling the fuller block leaves the fewest blocks occupied.
* Sensitivity to the memoryless turn (`--simulate`, rho 2, 64k walls): log-normal turns of sigma 1.0 and 1.5 and arrivals in pairs give +0.84 to +1.47% (exponential single arrivals +1.25%). The sign and size hold.
* The round breakdown's "~5%" for one-block placement at 3 to 5 live is between the placement alone (+1.4%) and the free-migration bound (+9%): it needs the users to move, not only the arrivals to choose.
* One tenth of a percent here is not resolvable on a card; the claim to check is the two-pass share (section 6).

## 5. What would get the rest (not built)

**Migration (bound +8 to +9%).** Move a lone user of the minority block to a free slot of the other at a step boundary when the total live is 4 or fewer. State to move: the GDN carry (48 layers of recurrent
state and convolution; 154 MB across the four chips, about 38 MB each), the slot's draft history and K/V banks, the verifier's per-slot buckets and replay tables (rewritten from the page table each
round), and the parked engine rebound to the destination (`rebind_slot`, which today seeds from a prefill capture, not from a decoding session). The KV pages need no copy (the page pool is shared).
Exactness would be a bit copy plus the rebind rules R1 to R3 (digest equality source against destination, as `audit_rebound_state` does for a rebind). Cost per move: the copies are tens of
megabytes a chip, and one skipped round of that user; a minority user that stays 20 rounds pays back in two. Not cheap to BUILD: R2 (no foreign replay between a verify and its publish), the deferred
commit flush order across the two blocks, the pre-stage epochs and the draft traces bound to the old slot are all per-slot state no CPU test reaches. A separate default-off switch
(`QWEN_FAST_TP4_PLACE_MIGRATE`, name reserved) and its own card window; I would build it only if the octo route below is refused.

**The octo pass for a split round (+5 to +7%, no migration).** At 5 live or fewer with users in both blocks, one octo pass (52.2 ms verify at 3.6k with the idle segments masked, 81.7 ms the round) replaces two M3
passes, and it also serves the five-live rounds (99.9% two-pass today) that no placement or migration can help. It needs what the octo block does not have: idle write targets for 3 to 6 idle segments
(NP1's scratch pages are wired to the M3 blocks only, `serving_runtime`: `idle_scratch_pages ... if not is_octo`), the octo draft pass for fewer than eight live seats (it refuses today), and a policy that
takes the octo pass only for a split round (a one-block round is cheaper on its M3 trace: 65.3 against 81.7 ms). The model charges 8 rows a user (tau x 0.9, round_shape_model K1 at 5 live).

## 6. Card gate plan (the switch is folded into the s7 candidate later; no card time is asked for here)

No profile twin is generated on this branch: a twin costs about twenty census-test edits (round-breakdown's probe twin touched 30 files) and is best written once, by whoever builds s7's twin chain. Spec:
`<ns>-fx-s6-place` = `-fx-traffic-s6` env + `QWEN_FAST_TP4_PLACE_COMPACT=1` (gate-only, the timed arm; its control is `-fx-traffic-s6` itself, as in every s6 pair) and `<ns>-fx-s6-place-audit` =
`-fx-s6-audit` + the flag (the audited attach). Folded into s7's candidate instead, the same two reads below attribute it, because nothing else in s7 changes the split of live users over the blocks.

1. **Audited attach (A).** Engaged line once, no `fell back`, every audit line `exact=True`, as the s6 audited attach. Then an eight-seat CHURN load: 12 or more greedy streams of unequal length started
   in a stagger, so that departures leave splits (the load that makes today's rule and the new one disagree). Pass: per-stream output text byte-identical to the control arm run on the same image,
   stream by stream; no audit mismatch; at least one round of each of 3, 4 and 5 live in the log.
2. **Timed pair (T), ABAB** on the s6 agent_turns arm and a churn arm at fixed 3, 4 and 5 live: control, candidate, control, candidate. Read with the ledger's second-verify field (`v1` in
   `[PACKED-ROUND-HOST]`, QWEN_FAST_TP4_ROUND_HOST_LEAN is in the s6 env) or round_probe_report.py (tp4/round-breakdown; its group key carries the pass count).
   * the share of ROUNDS in two passes by live count: expected, relative to control, -23% at 3 live and -14% at 4 (model: 12.1 to 9.3% and 32.4 to 27.9% at 64k), 0 at 5 (always two passes), and
     no change at 1 and 2 beyond a few tenths;
   * the round wall at a fixed (live, passes): unchanged within the pair's noise (the placement does not change what a round costs);
   * aggregate and slowest-user tokens per second at 3 to 5 live, and the agent_turns completion: NOT below control beyond noise (the expected +1.2 to +1.4% is below it, so this is a no-harm read);
   * the engaged line in the candidate arms only.
   Kill rule: any text difference; a two-pass share above control's at 3 or 4 live over two repeats.

## 7. Files and tests

* `scripts/ci/serving_buffer_pool.py`: the switch (about 40 lines).
* `scripts/ci/place_model.py`: the Markov model, the production calibration, the exhaustive search and the event simulation (host side only).
* `scripts/ci/test_place_compact.py` (in the CPU allowlist, after the fusion-wp block): the strict flag, off equal to the rule of before for every one of the 256 sets of lent slots, the difference only where one block
  holds one user and the other two or three, the engaged line, the pool refusing the flag at any size but eight (before allocating), engine reuse's predicate, `acquire` and a slot order, hand-built
  arrival/leave sequences with the verify passes of each step (today against compact), a seeded process in which both rules see the SAME turns (time at two passes at 2 to 4 live never more under compact, five live
  always two, live count identical), and the model (rules equal to the real function, mixture fit, orderings, simulation against the chain).

## 8. Open questions

1. Octo-on-split or migration, or neither? Octo-on-split is the larger and the cheaper lever on the numbers above and also covers five live; its blockers are the idle write targets and the octo draft pass at fewer than eight seats.
2. Who writes the two twin profiles (section 6), and does s7's pack carry a churn arm? The timed pair of this switch is only informative with a load that has departures at 3 to 5 live.
3. The model under-reads production's 4-live splits (32% against 43.8%); the production turn-length distribution (the rounds files, per request: start and end round, seat) would settle whether turns are stickier than
   exponential, and would replace the fitted load mixture with the measured one.
