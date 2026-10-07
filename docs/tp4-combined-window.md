# The combined development window: W2, Lever N with prefix reuse, the drafter hooks and the production prefix control

**Branch:** `tp4/w2-levern-prefix` (built on `tp4/levern-prefix`, with `tp4/w2` merged in). **Window image:** `tp4-w2ln-1`. **Pack:** `scripts/ci/references/tp4-w2ln-jobs`.
**Status:** built and CPU-tested. Nothing here has run on a card.

The production profile (`c2-packed-tp4-8x262k-ship-prefix`: W1, eight seats at 262k, sticky sessions with packed prefix reuse, image `tp4-serve-10`) is the control for everything below. The window
takes the cards off production once and gates four things together:

1. **W2** = W1 + `QWEN_FAST_TP4_SDPA=multi` (one K64j SDPA decode launch per attention layer for every user of a 64-row block) + `QWEN_FAST_TP4_CONV_GATES_SPREAD=1` ("F1", the block conv-gates launch with its gate tiles on cores of their own). F1's kernels have never compiled on a card; multi is outside the baked 262k evidence (its one-launch G16 flags-0x21 program is not the served 0x23 program).
2. **Lever N merged with prefix reuse**: a long prefill is split into 2,048-token-aligned steps that alternate with packed decode rounds; a prefix hit resumes through the same route; a short arrival parks a long prefill on the host (`docs/lever-n-prefix-merged-route.md`).
3. **Drafter hooks**: the existing `QWEN_FAST_DRAFTER_BF16` arm with a pool-matched control, and a profile-selectable checkpoint (`scripts/ci/drafter_checkpoint.py`) with provenance pins. No candidate id exists yet.
4. **The production prefix control (P1)**: exactness-shared and lifecycle-evict on the production audited profile and the production base image. It has never produced a verdict on cards (cancelled after 84 minutes once, at go-live once).

## 1. The merge

`origin/tp4/w2` (497bd004) merged into `origin/tp4/levern-prefix` (478e2736): 18 conflicted files, all additive (census lists, profile document, one smoke-rule block). The profile document was merged at the JSON level
(neither side changed an existing profile; W2's seven profiles append after Lever N's last), every list took the union of both sides, and `c2_smoke_check.py` keeps both rule blocks. The prefix route and the baked 262k evidence bytes are untouched.
The packed-prefix writer census (`test_tp4_packed_prefix_audit`) did not list W2's writers, so a plain merge would have passed CPU silently; it now lists `sdpa_multi_tp`, `sdpa_long_tp`, `gdn_conv_gates_spread` and `gdn_block_conv_tp` and classifies their write sites.

## 2. Interactions (read from the code; the card proofs are the pack's gates)

- **Multi x Lever N.** Multi binds three buffers per segment reader by Python identity (`positions`, the lent page table, `cur_pos`) and bakes their addresses into its gather and mask programs at attach. Only the reader constructors assign them; a Lever N step touches the B=1 prefill scratch and the decode slot, never those. Nothing re-checked the identities after a trace is captured (a replay never returns to Python), so the verify now asks multi on the host before every replay (`rebound_reason`), and `test_w2ln_identity_census` holds every site of the image that assigns, mutates in place (`metadata[:] =`, `cur_pos[i] =`, `append`, `setattr`) or constructs the readers.
- **Multi x prefix reuse.** A hit's table names pages another seat may read; multi gathers each user's row into its own stacked table and writes no pool block. Compatible. Sticky agents make skewed blocks (one long seat, three short) the common case, which is where multi can lose: the skewed eight is in every timed job.
- **Multi x the epoch scope.** Multi's persistent buffers are derived in-trace from the lent words every forward; they are not staging destinations. The proof is G-NP4 on the combined audit twin (zero full-audit mismatches, at least 95% of rounds after intermediate steps staged by diff).
- **F1 x checkpoints, chunked prefill and the slot write.** F1's only call site is the packed verify's V1 stage on the block windows; no prefill path calls it.
- **Audits.** `sdpa_multi_tp.attach` refuses `QWEN_FAST_EXTENT_AUDIT=1` without `QWEN_FAST_TP4_SDPA_AUDIT=1`, and both gates add the extent audit to every non-timed arm, so every such arm on a multi profile must name an audit twin. Both gates now refuse it on the host at plan time (naming the twins), reading the DERIVED profile's env.
- **Evidence.** The merged image's 262k evidence check passes with multi on (it pins `extent_attention_replay_tp.py`, which W2 does not edit), so no waiver is needed and none of the window profiles carries one. Multi at 262,144 logs `[PINDIAG] tp4 sdpa multi UNQUALIFIED ...`, the smoke requires the line, and the contract refuses multi, F1 and both audits on any traffic profile.
- **DRAM.** The pool of 19,968 blocks sits 11 blocks below the modelled high band; the F1 audit holds about 65 MB a chip. Fit is a card question, so the audit twins have committed fallbacks: `-lean` (no T1/T2 audits), `-pool` (19,200 blocks) and the `-sdpa`, `-f1`, `-ln` audit slices.

## 3. What was built

| | |
|---|---|
| Profiles | 30 gate-only profiles (`scripts/ci/references/tp4-w2ln-jobs/README.md` lists each as its parent plus exactly its keys; `test_tp4_w2ln_profiles`). |
| Gate guards | multi-without-its-audit refusal on both gates; `lever_engagement_problems` (the smoke's per-lever rules) applied to every gate arm's server log, with Lever N's kill-switch line allowed only in the drill arm; the contract's traffic refusal for multi, F1 and their audits; the UNQUALIFIED marker and its smoke rule; the rebound check; the identity census. |
| Card shapes | `cold2_254k` (six decoders and two simultaneous 254k arrivals: eight seats, never a ninth request; `test_w2ln_smoke_shapes` holds every shape at or under eight); the prefix plans `levern-faults` (hit mid-cold, abort of a parked long prompt, `levern.off` mid-split) and `levern-hit` (a timed hit-mid-cold arrival). The drill flag lives INSIDE the container (`QWEN_FAST_LEVERN_OFF_PATH` points at a container path) and is removed in a `finally`; the workflow also removes an owner-tagged `levern.off` in its trap and refuses to start any gate, prefix or smoke job while `prefix-reuse.off` or `levern.off` exists. |
| Timing | `scripts/ci/w2ln_timing_compare.py`: eight-live rounds of each timed job's server log, paired by mean-context bucket (not exact position), rounds beside a Lever N prefill step dropped, a pair VOID under 200 matched rounds or over the load maximum (the smoke step logs the host load once a minute), GO and NO-GO against the A-to-A drift floor. `speed_window_compare.py` reads matrix-gate trees and cannot do this. |
| Drafter | `drafter_checkpoint.py` and `drafter_checkpoints.json`: the default is today's bytes and paths; a candidate must be pinned (revision, config and manifest hashes, weights hash, geometry equal to the default's: the 2,048-token window, 15 proposals, 16 rows a user, five taps, hidden size and depth). The contract checks the profile's draft paths, the boot verifies the baked bytes and logs `[PINDIAG] drafter checkpoint id=... verified=1`, the smoke requires it, and the build stages candidates from a pinned list (`C2_DRAFTER_CANDIDATES`; a placeholder when none, so the image build never fails on a missing directory). |
| Pack | 47 planned and 15 conditional templates, `ORDER.txt` (classes, minutes, NEEDS, swap rules, read rules, cutover rules), `test_tp4_w2ln_window`. |
| Gate timeout | `exactness-shared` gets 10,800 s (`S2_TIMEOUTS`): it has never finished inside 7,200 s. |

## 4. The pack in one screen

Cheapest kill signals first. B0 builds before the outage (production serving, no card opened). A0 starts the outage. Then: F1 on card M; X0; the CONTROL block (P1a-CTL and P1b-CTL on the production bytes, S0-CTL on the window image with the prefix route's digests on, DK0 for the selector's plumbing, L8-CTL and C16-CTL, which production skipped); A1-C, the combined audited attach; on a pass the combined hangs (HW, HL, HX), the fault drills (HF) and the audited exactness jobs (P1a-C, P1b-C, L8-C, C16-C, E1-C, the eager exactness arm); then the timed blocks: T0 (production bytes), T1-T8 as A B C A B C A B (A production, B W2 alone, C combined), S1-S4 stall and TTFT, G0-G4 agent turns, GH1-GH4 hit mid-cold, D1-D4 BF16 against its pool-matched control; Z hands the cards back. The single-lever attaches and the bisect jobs run only when a combined job fails. Rules in `ORDER.txt`:

- **W2's gain is measured with no Lever N on either arm** (A against B), so a W2 NO-GO cannot be Lever N's. C against B is Lever N's steady-state cost on top of W2 and must sit inside the floor.
- **A prefix-gate arm needs verdict PASS.** NO-VERDICT counts as FAIL for every cutover, and a CONTROL NO-VERDICT queues the same arm on the window image (P1a-CTL2, P1b-CTL2).
- **A fallback cuts over only after PASS on P1a, P1b, L8 and C16 on its exact profile.**
- **P1 CONTROL is a production check.** A FAIL there is reported to the owner at once.
- **Hand-back needs the owner**: the agent does not re-dispatch by itself, so production returns through the platform release step with an admin token supplied inline.

Outage estimate (computed from the lines of `ORDER.txt`; pinned by the test): **full 2,449 min = 40.8 h** (A0 to Z, no conditional job); **core 2,029 min = 33.8 h** (drops D1-D4, GH3-GH4, G3-G4, S3-S4; keeps all three T pairs); **worst case 3,934 min = 65.6 h** (every conditional job and every box hit). An owner checkpoint sits at about 20 hours, after the T block. Tags: 62 of the 64 allowlisted, so re-runs after swaps need more tags requested first.

## 5. Review

Two independent reviews of the design were applied. What happened to each finding:

**Adopted as written.** The W2 round-time statistic with no Lever N on either arm and the A B C attribution; the comparator (the named tool could not compute it); attribution and fallback soundness (A1-LN and A1-W2 are supersets between them of A1-C, conditional bisect jobs, failure classes by audit); P1a-C and P1b-C on a twin without the Lever N KV digest, with `exactness-shared` raised in `S2_TIMEOUTS` and NO-VERDICT treated as not qualified; the `levern.off` leak (container-local drill path, owner-tagged cleanup in the trap, a start-of-job refusal, the kill-line engagement rule); `cold2_254k` at six decoders plus a seat-count test; the DRAM swap (pool twins) and the second level of audit twins (slices); the control digest reference and `levern_equal_long` in S0-CTL; the per-job duration rebuild and the split of the combined hang job by family; the host-side `rebound` call before every replay and the census over every overlay module; the epoch-scope fallback; T0 and G0 on the production bytes; the placeholder directories for candidates; X1 reading the derived profile's env; the pre-registered CTL2 jobs; L8-CTL and C16-CTL; the S and cold2 rules (single-arrival shapes judged on the gap, cold2 on TTFT only); the timed hit-mid-cold scenario; the eager exactness arm on the combined twin; the fault drills on the combined arm; core rules written separately; the pool-matched BF16 control and a selector-plumbing control (DK0); the load logger and VOID rule; the ordering (combined attach first, single levers only on failure); the report-only burst read; the hand-back's release step and the owner's availability; the owner's precondition written into P0.

**Adopted in part.**
- *A merged ROUTE line must exist in every arm where a prompt over 16,384 tokens arrived while decoders were live.* The kill-line half of the extended engagement rule is built; the arrival-conditional ROUTE rule needs the arm's arrival times, which the gates do not hold, so it is left to the smoke's NOT_EXERCISED reading in A1-C, HL-C and HF-C.
- *Per-check verdicts per completed round, so a timebox of P1 still yields PASS or FAIL for what finished.* Not built: it changes the judge's verdict structure. The mitigations are the raised timeout, the lean twin and the CTL2 jobs; NO-VERDICT is never read as a pass.
- *Per-test recomputation from measured W1 and ship times.* Done to the extent measured times exist (attach, hang shapes, timed smoke); the 254k-prefill shapes, the digests and every prefix arm remain estimates, stated as such.

**Not adopted, with the reason.**
- *Mount candidate fixtures at run time so a late candidate needs no rebuild.* The pins are verified at every boot from the baked bytes; a run-time mount would need a workflow change and would put unpinned bytes on the serving path. A candidate joins through the table and a rebuild.
- *Merge `tp4/tau-lab` (M2) so the drafter arms run on our transcripts.* It is the owner's call (one conflict, `c2_serving_job.py`) and it is not needed for the BF16 pair, which reads on the smoke with `drafter_bf16.compare`; the smoke pass rule is written. Recompute the merge-tree on the combined result before taking it.
- *Engine reuse and the GDN prototype in this pack.* Neither exists on the branch; the pack lists them as descoped in P0 until the owner says otherwise.

## 6. Open

- Nothing here has run on a card. The first questions are the card-M F1 compile, the audited attach fit (L1, DRAM, trace region), multi past 131k (L8-C is its first run) and two partial prefills in the plugin's persistent batch (the first sign is HL-C or HF-C).
- The multi qualification record for traffic (X3b) and Lever N on a traffic profile (X4) are owner decisions that land after the window.
- The tau-lab merge, drafter candidate ids and pins, and the three descoped items above.
