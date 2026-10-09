# tp4/round-host: the host work inside a verified round

Base: `tp4/er-drafters` (the integration base: Lever N, W2, engine reuse), profile `c2-packed-tp4-8x262k-ship-prefix-levern-traffic`. Six flags, every one
default off, host only (no kernel, no trace, no allocation, no device command), byte-identical off. CPU work only: nothing here has run on a card.

Labels: (m) measured from a real server log, (src) read from the code, (cm) computed from measured numbers, (e) estimate, (U) unknown until a card says.

## Answer first

1. **The host cost of a verified round is 13.4 ms at eight live users (157 ms) and about 11 ms at four (81 ms), and only part of it holds the device idle** (m):
   the two verifies are 49.7 ms of device trace each, and 7.2 ms of verify host work plus 6.2 ms between them are the "13.4". Inside the 43 ms early draft the host
   also does 4.4 ms of readback, 2.2 ms of FP64 selection and 2 ms of ticket work. The device is idle (e) about 12-14 ms of the 157, **8.5 ms of it in one
   chain: the draft fence returns, the host reads 40 buffers, selects, hands the drafts to vLLM, enters the next step and stages the verify, while the
   only device work queued is the 4.4 ms of GDN commits.** The rest (the verify read backs, the commit enqueue) is one in-order command queue's turnaround and no
   host-only change moves it.
2. **What is built** (default off, behind `QWEN_FAST_TP4_ROUND_HOST_*`): the ledger (`LOG`), a faster exact selection (`SELECT`), a batched merge and a sampled
   replicated-feature guard in the quad readback (`READ`), the keyed verify-time diff (`KEYED`), the dropped per-phase lines (`LEAN`) and a reference-beside-fast audit
   (`AUDIT`). Together **(e) -3.8 ms a round at eight live (2.4%) and -2.5 ms at four live (3.0%)**, all on the chain above except the lean log's share.
3. **What is not built, and why** (section 5): a device-resident round, publication as a trace and a ring-buffer publication. The publication already is a trace (the
   fused commit, `docs/tp4-traced-publish.md`), the commit's host share is hidden under the device's own fused commit, and everything else changes a trace or needs
   a non-blocking read the pinned runtime has never been tried on. Those are kernel- or trace-owner work.
4. The ceiling of host-only work is about 8% of the round (the device idle, e); what is built takes about a third of that. The device floor at eight live is
   two verifies + two quads + the GDN and fused commits = about 143 ms (e: the quads are read from the draft's timing, 29.5 ms for both).

## 1. Where the 13.4 ms goes (m)

Source: the container log of a smoke run of the traffic profile on the integration image (13,000 `[PACKED-PHASE]` lines), eight-live rounds at about 4k context
(238 rounds under 260 ms) and four-live rounds (255), parsed from the `[PHASE]` / `[PACKED-PHASE]` / `[PACKED-SELECT]` / `[PACKED-PRESTAGE-WINDOW]` lines. Milliseconds,
p50 (mean in brackets where it differs by more than 0.3). Log lines are 1 ms stamps; the fields the code times itself are 0.01 ms. **The log is the image's audited one
(`QWEN_FAST_PACKED_AUDIT=1`, `QWEN_FAST_PHASE_LOG=1`: 92 lines an eight-live round), so every host phase carries its own logging.**

| Piece (code) | 8 live | 4 live | On the device-idle chain? |
|---|---:|---:|---|
| Round | 159 | 81 | |
| Device traces (verify A + B) | 49.7 + 50.2 | 50.0 | no (it is the device) |
| Step entry: admit, runner update, 8 x reservation + page refresh (`execute_packed_decode`) | 2.0 | 2.0 | yes: after the GDN commits have drained |
| Verify input stage, the diff write (`verify_prestage.stage`, `input_ms`): A / B | 2.3 / 2.4 | 2.2 | A yes; B hides under fused commits A |
| of which the diff recompute (`diff_ms` p50 1.56, `write_ms` 0.60) | 1.6 | 1.6 | |
| Verify sync + prediction read back + glue: A / B | 0.1 + 0.8 + 0.3 / 0.2 + 0.8 + 0.3 | 0.1 + 0.9 + 0.3 | yes: the fused commit waits on the accepted prefix |
| Commits (publication enqueue + bookkeeping): A / B | 2.0 / 1.0 | 1.0 | A hides under the 4.9 ms fused commit; B nearly |
| Early draft, whole | 43.7 | 23.6 | |
| &nbsp; its host work before the first quad launches (eligibility, tickets) | 2.0 | | mostly hidden (fused commits B) |
| &nbsp; the two quads' launch host work | 2.3 + 1.8 | 2.3 | hidden under the first quad |
| &nbsp; the pre-stage window (hidden by design) | 17.8 | 9.1 | no |
| &nbsp; the readback of the draft outputs (`collect_ms`, 40 blocking reads) | 4.4 | 2.4 | **yes** |
| &nbsp; the FP64 selection (`select_ms`) | 2.2 | 1.5 | **yes** |
| &nbsp; select to the end of the draft; the hook and vLLM to the next entry | 1.0; 1.0 | 1.0; 1.0 | yes |

Closure at eight live: 2 x 49.7 + verify host 7.2 + early draft 43.7 + residual 6.0 (entry 2.0, commits 3.0, hook 1.0) = 157 (the brief's 157 = 2 x 49.7 + 7.2 + 42.9 + 6.2).

**The idle chain** (e, eight live): fence returns, 4.4 collect, 0.1 GDN enqueue, 2.2 selection, 1.0 tail, 1.0 hook, 2.0 entry, 2.3 stage = 13.0 ms of host
before verify A's trace can start, 4.4 of which the deferred GDN commits cover: 8.6 ms of idle. A to B, B to the first quad: the fused commits (4.9 ms each) cover the
host (idle about 0.3-1.3 ms each, and that is the sync + read back turnaround).

## 2. The levers

| Flag | What | Exactness | Estimate, eight / four live |
|---|---|---|---|
| `_LOG` | one `[PACKED-ROUND-HOST]` line per step with the host ms of every phase, `pos` and `live` as pairing keys | log only | the control arm carries it alone |
| `_SELECT` | `draft_selector.select_active_candidates` with the operand checks made once: the gathered codebooks were scanned for non-finite values, and converted to float64, once per 8-position chunk. 45% of a call on a CPU; the codebook scan is kept per tensor (identity, storage address, version) and runs once (about 50 ms a float64 codebook, m: a one-time 0.1 s on the first selection after the attach). | the arithmetic is the reference's, statement for statement; every failed check hands the whole call to the reference | -0.85 / -0.4 (m on float64 codebooks as served: the call 4.2 ms single-threaded, 1.9 ms at 4 threads, the rig's 2.1; 65% and 45% faster) |
| `_READ` | the quad readback's merge of 8 chunk reads as one batch (stacked validation), and the replicated-feature guard (the same tensor read back from all 4 chips and compared) on the first 64 reads and every 64th; chip 0's copy is what the selection uses either way | batched checks accept a subset of what the reference accepts and return its bytes; the audit's flag-off selection is always the reference | -0.6 (merge, m: 44%) -0.7 (3 of 20 reads a quad) / -0.3 -0.3 |
| `_KEYED` | the verify-time diff keyed on (start, page table) per segment: the pre-stage keeps copies of both; at verify, equal values mean every staged value but the tokens is the snapshot's (a pure function of the key), so the 149-destination recompute and comparison is skipped and the tokens buffer alone is written | the T2 K/V guard verdict is taken at the pre-stage; readers still validate their start before any copy; any change (position, an appended page, a replaced reader, a conflict, the T2 audit) takes today's diff | -1.2 on verify A (B hides) / -1.2 |
| `_LEAN` | not written: `[PHASE]` begin/end of propose, prepare_proposals, propose_quad, early_draft, packed_commit; `[PACKED-PUBLISH-SPLIT]`, `[PACKED-COMMIT-HOST]`, `[PACKED-COMMIT]` (52 of the 92 lines); the ledger replaces them | lines only | -0.5 on the chain, 1-1.5 host in all (e: 20-30 us a line) / -0.3 |
| `_AUDIT` | the reference runs beside each fast path on the live data (selection, merge, the keyed round's full values against the snapshot) and the two must agree byte for byte; a difference is logged and the REFERENCE bytes are used | `[ROUND-HOST-AUDIT] kind= equal=` | host cost, not a timing arm |

Totals (e): **-3.8 ms at eight live, -2.5 ms at four live**; each is read, phase by phase, against the control arm by `scripts/ci/round_host_report.py pair`, and in
round time by `w2ln_timing_compare.py pair`. A phase that fell and a round that did not was hidden under a device trace.

LEAN's trade: the begin/end lines of the dropped phases are what names a stalled phase in a hang. Under LEAN a stall is localised by the last ledger line and the
`packed_verify` lines only; run the hang shapes for adoption beside an arm without LEAN if the stall markers matter.

## 3. Exactness

STRICT (the owner's rule): the greedy output is byte-identical to the target's own greedy decode.

- **SELECT and READ** never change an operation's operands or order: the selection's arithmetic is the reference's code on the same float64 operands (the codebooks are
  converted once instead of twice; a conversion is deterministic), and the merge is the reference's sorts and gathers on the stacked same bytes. Both DECLINE to the reference
  (the whole call) on any failed check, so an error is raised by the reference's own code with its own message; a declined call is logged (`[PINDIAG] round host declined`)
  and fails the smoke check. `test_tp4_round_host` pins the equality bit for bit over random operands, ties included, and every refusal's message.
- **KEYED** relies on one claim: every staged value but the tokens is a pure function of (start, page table) per segment and of constants. The values are the
  snapshot's when the key stands because they were computed from that key (`packed_values` at the pre-stage); the CPU tests show it on the real block over the fake device
  model, whose replays compute their predictions from what is staged: random users, page tables, idle sets and T2 on or off, the device state after every round equals the full
  stage's. On a card the existing full read-back audit (`QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT`) covers the keyed rounds too, and `AUDIT` checks the claim itself
  (`changed == [tokens]`).
- **LEAN, LOG** write lines.
- With every flag off: no line, no counter, no new key in a snapshot, the same call order (`FlagOffIdentityTests`, `LedgerTests`, and every earlier suite that touches these modules).

## 4. The smoke rules and the job pack

`c2_smoke_check.round_host_problems`: a profile without the flags logs none of the lines; with any flag the engaged line is once and names the profile's flags; no refused or
declined line; the ledger runs (a line a step with every field); on the eight-seat steady mix SELECT and READ ran on 95% of the drafting steps, KEYED on half of the verifies of
a block that can be keyed, the feature guard was skipped on some steps once 64 reads had passed; LEAN wrote none of the lines it drops; AUDIT logged an `equal=1` line for each audited lever
and no `equal=0`. (`concurrent8_steady` is in every eight-seat quad smoke list: the quad-blocks rule fails the job without it.)

Profiles (generated by `make_round_host_profiles.py`, gate only, the traffic profile plus flags): `-roundhost-log` (the control), `-roundhost` (the arm), `-roundhost-audit`.

`scripts/ci/references/tp4-round-host-jobs` (templates only): X0, B0, E1 (exactness control, unaudited), A1 (audited attach), E2 (exactness arm, unaudited: every hash of every
test equal to E1's), T1-T4 (paired ABAB at eight live), T5-T8 (at four live), Z. CI paused for the timing block; the load rule of the combined window applies.

## 5. Not built, and why

- **Device-resident round / ring-buffer publication.** The publication already is a trace (`QWEN_FAST_FUSED_COMMIT`: 4 x [T_proj 1.0-1.1 ms + a slide trace], 4.9 ms a block, in
  every profile that matters); its host share is the commit enqueue (2.0 and 1.0 ms), hidden under that device work. A ring-buffer history is a change of trace content (lever P2 in
  `docs/tp4-traced-publish.md`), with the trace-capture-order hazards that note lists; the selection that sits between the readback and the next verify is a host FP64 search on
  purpose (`docs/tp4-hostgap.md` stage 3).
- **Joined quad outputs (20 reads to 6 a quad)**: a trace-content change (copies inside the quad's capture), -3 ms (e); with events and non-blocking reads (E0 of the host-gap
  design, never tried on the pinned runtime) it would become -6 more. Neither is host-only.
- **Reads on worker threads**: unknown whether the runtime serialises a mesh's reads; a card probe, not a CPU argument.
- **GC freeze**: no measured stall that GC explains in this log; the 300 ms+ rounds of the log are quad captures at admissions (`[QUAD-DRAFT] built=1`).
