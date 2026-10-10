# tp4/octo-2: making octo-T8 pay (three levers behind the octo timed arm)

Branch tp4/octo-2 = tp4/window-next + the three levers below. Every lever is default off, gate only, strict 0/1 (or a count), byte-identical off (a test each), unqualified on cards, and has its own smoke
rule (a log marker proving it ran), CPU tests, a gate-only profile twin (make_octo2_profiles.py) and card jobs (scripts/ci/references/tp4-octo2-jobs). Owner rule throughout: STRICT exactness - greedy
output byte-identical to the target's own greedy decode. Only Lever 1 can change how many tokens a round commits (a different draft); none can change which.

## 1. The measured basis (O3, the octo twin, QWEN_FAST_OCTO=alternate, 8 live, 140 octo + 139 m3 rounds)

| | octo round | m3 round (2 passes) |
| --- | --- | --- |
| verify trace (`[PACKED-PHASE] trace_ms`) | 86.5 ms (one 8-user block) | 2 x 50-54 ms |
| step (verify + commits + host) | 94.7 ms | 112.9 ms |
| gap = the early draft (43 ms at the median in both shapes) + vLLM | 44.9 ms | 46.6 ms |
| committed tokens per seat-round | 3.66 | 4.07 |
| committed tokens per second per seat | 25.9 | 25.2 (paired: x1.031 aggregate, x1.086 median pair; bar x1.10) |

A pass is ~13 ms of weights and ~9 ms per user (attention over the user's KV, GDN recurrence and state, per-user glue, commits): the rows are nearly free, the launches are not. The octo block spends
that ~9 ms on eight users instead of four, and drafts with two quad passes.

## 2. The levers

| | lever | what | est. saved per 8-live octo round | status |
| --- | --- | --- | --- | --- |
| 1 | `QWEN_FAST_OCTO_DRAFT` (docs/tp4-octo-draft.md) | ONE 64-row pass drafts all eight seats at eight rows (T8) instead of two quad passes | -13 ms (range -8 to -18), before any tau change | built, 74 + 14 tests |
| 2 | `QWEN_FAST_OCTO_GLUE8` (docs/tp4-octo-glue8.md) | the V2 split/merge, V1 block conv and T2 windows run natively at eight-row users (quarter-tile mover kernels) | -16 ms (range -9 to -24) | built, 93 tests; K5-A at 8 rows STUBBED (a recurrence compute kernel) |
| 3 | `QWEN_FAST_OCTO_ATTN_BUNDLE=4` (docs/tp4-octo-attn-bundle.md) | the octo block's attention in 2 K64j launches per layer (4 users each) instead of 8 | -6.9 ms at 4k, -19.9 at 32k, -60 at 120k (modelled) | built, 62 tests; no kernel change |

The estimates overlap (a faster verify leaves the draft less to hide under) and do not add: the all-three arm is read after each has passed alone. At the O3 shape (141 ms cycle, 3.66 tokens per
seat-round) -13 ms is +10%, -16 ms +11%, -7 ms +5%; all three, discounted, would take the octo shape from +3.5% to about +20 to +25% over the m3 rounds (bar +10%).

## 3. Why each is safe, in one line

1. The draft: only the target's argmax is ever committed, so a different draft is a different TAU, never a different token; the fold re-indexes each user's single-user T8 computation (bit-equal on two
   SDPA stand-ins with firing controls). The cost is tau (the checkpoint is trained at 16 rows): read by `octo2_report.py`.
2. The glue: bytes move, arithmetic does not; the canonical set at eight rows (users 0 and 4 raw, the rest canonical) extends the sixteen-row rule by position and is exactly what the in-trace vglue audit
   (profile `-octo-glue8-audit`) compares on every chip.
3. The bundle: each entry of a bundled launch IS the single launch's entry (16 cores per entry up to 6 entries, the partition and the tree depend only on cur_pos and the core's index inside its
   entry); the in-trace audit (`-octo-bundle-audit`) compares the eight single launches with the two bundled ones word for word.

## 4. Nothing to build

All device pieces are `generic_op` JIT sources shipped as `scripts/ci/*.cpp` (the quarter-tile mover and windows kernels) or existing ops (the draft pass reuses the quad's, with the sha-pinned conv
kernel untouched; the bundle reuses `sdpa_multi_*`). **No tt-metal rebuild, no graft `.so`, no op build in the ttbuild container.** The new modules ride the image build (B0): `octo_draft_tp.py` and the
glue8 files in both copy lists and the overlay (the table is `tp4_vglue.RUNTIME_FILES` for glue8), `octo_attn_bundle.py` in the overlay only (as `sdpa_long_tp`).

## 5. Profiles and flags

make_octo2_profiles.py (never hand-merge; `--check` in CI) writes six gate-only twins behind the octo block of the profiles file: `P-octo-draft`, `P-octo-bundle`, `P-octo-bundle-audit`, `P-octo-glue8`,
`P-octo-glue8-audit` (on the audited octo twin) and `P-octo-levers`, P = c2-packed-tp4-8x262k-ship-prefix-levern. Each is its parent plus exactly the env of its lever; the memory plan is the octo twins' (the
third block's pool and trace region): a quad capture measured 375 MB and the free DRAM at the O3 steady state was 6.4 GB per chip, so the octo draft pass (one more capture, estimated 520 MiB) fits.
There is no audited draft twin: the octo draft is refused beside the singles audit (nothing else drafts T8 to compare with).

## 6. Smoke rules and reads

`octo_judge.judge` (called by `c2_smoke_check.check`) now also runs `draft_problems` (admitted once, engaged once at the profile's conv mode, >= 16 rounds and half the octo rounds at 8 live, no
fallback / disabled / refusal), `octo_attn_bundle.bundle_problems` and `octo_glue8.glue8_problems`; a profile without a lever's flag that logs any of its lines fails (a leak). `octo2_report.py
--control <log> --arm <log>` is the paired read between boots (octo rate per seat aggregate and median, tokens per seat-round, early-draft ms per planned shape, the m3 rounds of both boots as the control of
the control); `octo_compare.py` is the text read (every answer equal).

## 7. Jobs, in order (scripts/ci/references/tp4-octo2-jobs/ORDER.txt, 615 minutes, 500 without the repeat pair)

B0 build (one image), X0 card state, **OC1** control (`-octo`), **OD1** Lever 1, **OGA** Lever 2 audited attach (nine audited requests), **OG1** Lever 2 timed, **OBA** Lever 3 audited bundle, **OB1**
Lever 3 timed, **OL1** all three, **OC2/OD2** the ABAB repeat of the first pair, Z. Push one tag at a time, CI paused from OC1 to the last timed job, the owner stops production before X0.

## 8. Not done, and why

* K5-A at eight rows (`QWEN_FAST_GDN_SEQ_BLOCK`): the recurrence compute kernel is qualified for sixteen-row users; an eight-row one is a new compute-kernel build and its own qualification chain. The largest
  remaining per-user GDN item at the octo block.
* The draft book / parked engines with the octo draft: their traces are the quad's; the octo draft trace is not in the book. The `-octo-parked` twins therefore carry no lever.
* A card-M mover harness (the quarter-tile kernels run first in OGA, in the model): a hang there needs a person to reset the cards; a one-card harness with the NoC sanitiser (as Q1w for K64j) would be
  the safer first step and is the recommended addition if OGA is to be risked.
