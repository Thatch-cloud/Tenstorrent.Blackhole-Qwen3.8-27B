# The four-card batched draft (tp4/draft)

At four cards (QWEN_FAST_TP=4) the fast path drafted every user on its own single-user capture. This note records what a
live-4 round does between two verifies, what the branch changes and how the change is proven exact.

## The round, at four cards, before this branch

Measured on the speed window's timed arm (audits off, the K/V slide on; four users, two of them in the draft ramp), 173 rounds,
medians:

| Phase | ms | Notes |
|---|---|---|
| verify (`packed_verify`) | 63.3 | the trace is 57.8 |
| commits (`packed_commit` x4) | 37.9 | four serial eager publications, 8-15 ms each |
| early draft | 63.8 | one `prepare_proposals` of 47.2 (four single-user draft passes back to back on the device, about 10-11 ms each, the host enqueue and the next verify's prestage hidden under them) and four serial `propose` calls of about 3 ms |
| gap | 5.0 | vLLM step and scheduler |
| **round** | **170** | |

Every 4-live round logged `[PACKED-PROPOSE] singles ... reasons=['ramp' x4]`: a fixed slot pair (0, 1) or (2, 3) packs only when
BOTH members' draft history is at 2,048 rows, and each pair held one prompt short of that. Nothing was refused for being TP4.

## What the branch does

- `quad_draft_tp.py` is the four-card twin of `quad_draft.py` (one 64-row draft pass for four users, QWEN_FAST_QUAD_DRAFT=1).
  `tp_addresses.MODULE_TWINS` puts it in the pinned module's place at QWEN_FAST_TP=4 only; the pair keeps the pinned module, whose
  bytes a test holds. The width is read from `tp_shapes` at each call: 8 query / 2 KV heads per chip, so the four-way fold runs the
  draft SDPA at 32 query / 8 KV heads over 2,080 keys - the pair fold's program, whose every unit is the user's single-user unit.
- The K/V banks are placeholders the pass fills from each user's ACTIVE bank every round. The four-card slide swaps the active bank
  on every commit and the four-card fused commit (the in-place slide that would let a trace read the live banks) is not ported, so
  the live-banks flag is refused with the reason.
- `draft_singles_audit.py` (QWEN_FAST_DRAFT_SINGLES_AUDIT=all|N): in an audited round every member of every batched group (a
  packed pair or the quad) also drafts on its own single-user capture with the same seed; after the fence the raw per-chip outputs,
  the selector features, the candidates, the scores and the tokens are compared bit for bit. One `[DRAFT-SINGLES-AUDIT]` line per
  group names the first difference. Off, nothing changes.
- `acceptance_report.compare_packed(by_position=True)` and `speed_window_compare.py --batched-vs-singles` compare a user's
  accepted prefixes by the position each round drafted at, so a scheduling difference cannot pass for a draft difference and a real
  one cannot hide behind it.

## Why exactness needs its own proof

The verifier emits the target's own argmax whatever the draft holds: a wrong batched draft changes acceptance, never the text. So
the text comparisons cannot see it, and the proof is about the drafts: the CPU tests hold the fold's head map at 8 / 2 heads, its
bit equality with the pair fold and with each user's single-user SDPA (two stand-ins, one of them ordered by key chunk, with
firing controls), the K/V layout, the readback against four singles' rows; the audit and the per-request prefix comparison hold it
in the model.

## Projection (verify held at 63 ms, four steady users)

| Config | Round (ms) | tok/s per user at 4.2 / 10 tokens per round |
|---|---|---|
| singles (measured) | 170 | 24.7 / 58.8 |
| packed pairs | about 141 | 29.8 / 70.9 |
| quad, this branch | about 134 | 31.3 / 74.6 |
| quad and a four-card fused commit | about 95 | 44 / 105 |

The commits (38 ms) are the next term; 75 tok/s per user at 4.2 tokens per round needs a round of 56 ms, below the verify alone,
and belongs to the verify-side branches.

## Running it

`scripts/ci/references/tp4-draft-jobs` (ORDER.txt): D0 build, D1 the audited quad smoke, D2 the same with the quad off, D3 the
exactness matrix on four steady users, D4 and D5 the timed smokes. The smoke's `concurrent4_steady` test is the mix that can form a
batch (the mixed `concurrent4` keeps two users in the ramp).
