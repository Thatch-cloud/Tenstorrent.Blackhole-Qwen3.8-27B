# tp4/octo-2 Lever 1: one eight-seat draft pass for the octo rounds (`QWEN_FAST_OCTO_DRAFT`)

Status: built and tested on CPU (`test_octo_draft`, `test_octo2_report`, `test_octo2_profiles`); nothing here has run on a card. Default off, gate only, byte-identical off. No kernel, no graft,
no op build: every device piece is an op sequence the quad draft already runs (`generic_op` conv kernel included, sha-pinned and unchanged).

## 1. The measured basis

O3 (the Lever N octo twin, `QWEN_FAST_OCTO=alternate`, 8 live, container log read with `octo2_report.py`):

| | octo round | m3 round |
| --- | --- | --- |
| step (verify + commits) | 94.7 ms | 112.9 ms |
| gap (the early draft, vLLM, the scheduler) | 44.9 ms | 46.6 ms |
| committed tokens per seat per round | 3.66 | 4.07 |
| committed tokens per second per seat | 25.9 | 25.2 |

The gap is the draft: the early draft of an 8-live round takes **43 ms at the median in either shape** (the 30 ms median in the first read mixed ramp and drain drafts at other live counts).
Its timeline (one round of the log): 3 ms of host prelude, two quad launches (2.2 to 6 ms of host each), the verify pre-stage window (11.75 ms of host work, hidden under the device), the
fence, then the readback (`collect_ms` 4.6) and the selection (2.0). The device runs the two 64-row quad passes back to back, about 15 ms each (a single-user pass is 10.5 ms at 32
rows, `docs/tp4-batched-draft.md`: the pass is op-latency bound, a 64-row pass costs 1.4 times a 32-row one). The octo round verifies eight seats in ONE pass and then drafts them with TWO.

## 2. What it does

One 64-row pass over eight seats x eight rows (user u at rows 8u..8u+7), drafted at the round's end for an octo round with all eight seats live:

* **The pass is the quad's, at eight users** (`octo_draft_tp.py`; `quad_draft.py` and `quad_draft_tp.py` are not edited): every row-local op, the nine matmul programs at `per_core_M = 2`,
  the K/V projection and rotary, the fused conv (the sha-pinned `quad_conv_io.cpp` reads a seam bit per row; seams at rows 8, 16 and 24 of a tile row are the same kernel), the collectives, the
  learned norm and selector projection and the 32-row-half LM head are the quad's calls. `OctoPass` is a `QuadPass` with `users = 8` and `block = 8` (the two places that counted four users of
  16 rows, `dflash_device.execute_proposal`'s guard and its MLP seams, read them with the quad's values as defaults).
* **K/V**: 24 pieces. User u's 2048 cached rows (the pool's live banks, as the quad binds them), its 8 live rows `[8u, 8u + 8)` of the block, and 24 pad rows from its 32-row half. Every segment is 2080
  keys with the live keys at 2048..2055: the single-user T8 layout. The pad is masked.
* **The fold** (the quad's four-way fold generalised): each 32-row half holds four users; the half is rotated by 0, 8, 16 and 24 rows (the quarters concatenated in cyclic order) so user k of the half
  sits at rows 0-7, where the single-user T8 trace has its eight query rows; the four rotations are grouped on the head axis and the two halves after them, so folded head
  `8 * group * h + group * u + j` is head `h * group + j` and the GQA group sends it to KV head `8h + u`. The SDPA runs at 8 x the query heads over 8 x the KV heads (**64 / 16 at four cards**, the
  program shape the pair's quad proved on card B) reading the one single-user T8 mask. Every (query head, KV head) unit is the user's single-user unit: the same 8 query rows at the same tile rows, the same
  keys in the same order, the same mask tiles, the same 65 key chunks. `test_octo_draft` proves it bit for bit against each user's single-user SDPA on a per-head float32 stand-in and a
  key-chunk-ordered online-softmax stand-in, with firing controls (a different chunk size, a mis-mapped segment, a missing rotation each change the bits).
* **The readback**: the quad's 18 reads, each 32-row half merged as the quad merges it, stitched `[half 0 | dummy | half 1]`, split by `split_selection(users=8, block_rows=8, block_width=64)`; each seat
  selects 7 tokens. `test_octo_draft` holds every user's features, candidates and scores equal to the selection over its own eight rows.

## 3. What it costs, and why that is the card question

The target verifies every proposal, so the text is the target's own greedy decode whatever the draft holds: **exactness is not at stake, tau is.** A T8 block has eight live keys where the T16 block
has sixteen, and the drafter's attention is bidirectional inside the block (`dflash_batched_mask.batched_attention_mask`: every live query sees every live key), so rows 1-7 of a T8 draft are not rows
1-7 of a T16 draft. The checkpoint is trained at 16 rows. How much acceptance moves is unknown and is job OD1's answer (`octo2_report.py` prints the tokens per seat-round of both boots beside the
draft time). A tau loss above the time gain is a NO-GO; the lever is switched off by unsetting one env var.

## 4. Where it engages

`PackedProposalCoordinator._prepare_octo`, before the quads, for a round the packed step planned for the octo block (the hook passes `octo_draft=True` only then: `octo_draft_round`) with all eight slots live and
packable. Six and seven live octo rounds keep the quads (cut to seven rows on the host, as today). A refusal (`octo_draft_tp.refusal`), a short DRAM headroom or a failure falls the round back to the quads
(`[OCTO-DRAFT] fallback`); two failures in a row block the pass for the process (`[PINDIAG] octo draft disabled`, which the smoke rule fails on). A round that does not form the pass drops an
earlier unconsumed proposal of it (the capture view matches on the seed alone, and a seven-proposal pass cannot answer a wider ticket; `finish` refuses a count above 7). The trace closes with a
member's close (`release_closed`), like the quad's. The draft book / parked engines are refused (their traces are the quad's).

Memory: a fresh capture asks `OCTO_CAPTURE_BYTES_EST` (520 MiB; a quad's measured capture was 375 MB, estimated 472) of the headroom check; the octo twins' free DRAM was 6.4 GB at the O3 steady state, so
the plan needs no pool change. `octo_draft_built` is the ledger point that measures the real figure; the pooled mask and output sets for slots 0-7 are allocated at the attach (a few hundred KB).

## 5. Flags, markers, the smoke rule

`QWEN_FAST_OCTO_DRAFT` strict 0/1 (unset 0), `QWEN_FAST_OCTO_DRAFT_CONV` 110|80|halves (the quad's modes). It needs `QWEN_FAST_OCTO=live|alternate` (`serving_octo.octo_admission` folds the draft's reasons
into its refusal; `serving_runtime` refuses the flag beside no octo block), the quad flags, `QWEN_FAST_PACKED_PROPOSAL`, `_PIPELINED_PROPOSALS`, `_PAIR_ROW_EXACT`, `_ROUND_B1`, and is refused beside the quad
shadow audit, the singles audit, the mask audit and parked engines.

Lines: `[OCTO-DRAFT] admitted ...` and `[OCTO-DRAFT] UNQUALIFIED (gate only) (i/3) ... [job ...]` once; `[PINDIAG] octo draft engaged slots=[0,...,7] heads=64/16 rows=64 block=8 conv=110` once, after the first
capture and replay; `[OCTO-DRAFT] round=<n> built=<0|1> ms=<host enqueue ms>` per round the pass served (**the draft ms per round**; the whole draft's wall time is the `[PACKED-EARLY-DRAFT] draft_ms`
that `octo2_report.py` joins to the shape it drafted for); `[OCTO-DRAFT] fallback`, `[PINDIAG] octo draft disabled`. `octo_judge.draft_problems` (called by `c2_smoke_check` through `octo_judge.judge`) fails
an arm that does not admit once, engage once at the profile's conv mode, serve at least 16 rounds and half the octo rounds at 8 live, or that logs a fallback, a disabled marker or a refusal; and fails
a profile without the flag that logs any of these lines.

## 6. Estimate

Saved per 8-live octo round: the two quad passes run back to back on the device (about 15 ms each, the 43 ms draft = 3 prelude + 30 device + 7 readback and selection + the hidden host window). One 64-row
pass over eight seats costs the quad's 15 ms plus the larger K/V assembly and the fold's extra copies (about 1.5 ms; the SDPA runs on 64 cores in the time the quad's 32 units take on 32) and halves the
readback. Central **-13 ms (range -8 to -18)** of the 141 ms cycle: +10% on the cycle before any tau change, which with O3's +3.5% would clear the +10% bar; the draft phase cannot drop below the host work
it hides (about 18 ms), which bounds the best case at about -25 ms.

## 7. Jobs

`scripts/ci/references/tp4-octo2-jobs`: OC1 the control (`P-octo`), OD1 the arm (`P-octo-draft`), OC2 and OD2 the repeat (ABAB), the text read `octo_compare.py` (every answer equal) and the paired read
`octo2_report.py --control OC --arm OD` (GO at +5% on the octo rate per seat with the m3 rounds of both boots within 5% of each other). No audited job: nothing else drafts T8 to compare with; the
fold's exactness is the CPU proof, and the lever's cost is tau.
