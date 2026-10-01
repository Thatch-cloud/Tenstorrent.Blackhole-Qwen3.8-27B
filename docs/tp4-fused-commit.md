# TP4 fused commit (tp4/fcommit)

The four-card port of the pair's fused commit (round-fence plan H1b). Nothing here has run on a card yet: the CPU proofs are in this branch, the hardware
proofs are the jobs in `scripts/ci/references/tp4-fcommit-jobs`.

## Why

A live-4 round of four steady users at TP4 is about 65 ms of device verify plus about 110 ms of host. Four serial `packed_commit` calls take 37.9 ms of it:
each is the drafter's own eager K/V update (the projection, two RoPE uploads, ten slide dispatches out of place into the spare, a fence, then a bank swap),
8-15 ms per user. The pair's fused commit replaces that with one T_proj trace per segment (feature projection, K/V projection, five layers) and one
in-place slide trace per (segment, prefix): 68 traces, commits of about 2 ms, and the banks never swap, so the packed pairs and the quad can bind the
pool's active banks (live banks) and copy nothing.

Projected round (verify 65.2 ms, gap 5 ms; tau = tokens per user per round):

| Config | Round (ms) | tok/s per user at tau 4.2 / 8 / 12 |
|---|---|---|
| Singles, eager commit (today) | 172 | 24.4 / 46.5 / 69.8 |
| Pairs, eager commit | 144 | 29.2 / 55.7 / 83.6 |
| Quad, eager commit | 136 | 30.9 / 58.8 / 88.2 |
| Pairs + fused commit + live banks | 108 | 38.7 / 73.7 / 111 |
| Quad + fused commit + live banks | 100 | 42.1 / 80.2 / 120 |
| Zero draft, zero commit | 70 | 59.8 / 114 / 171 |

The fused commit is worth about 34 ms a round. These are projections, not measurements.

## What blocks `fused_commit.py` at four cards

`fused_commit.py` is byte-pinned (`test_quad_draft`, `test_fused_commit`) and the pair's evidence is about those bytes, so it is not edited. It refuses at
four cards because of: the slide-scope check (the pair's text-patched `DraftKVHistory.prepare`, which the four-card process never installs); four KV heads
per chip in the bank and delta shapes; 16 workers per bank; two-chip loops in the slide program and the retainer; the pair's slide transport (raises at
TP4); and the pair module's kernel import (not served at four cards).

## What was built

- `scripts/ci/fused_commit_tp.py`: the module twin, installed only at `QWEN_FAST_TP=4` through `tp_addresses.MODULE_TWINS`. Widths come from `tp_shapes`
  (banks `(1, 2, 2048, 128)`, deltas `(1, 2, 32, 128)`, 8 workers per bank, ten banks = 80 workers per program: one program per user per chip). The scope is
  `QWEN_FAST_TP_KV_SLIDE=1` plus the TP4 cache class. The kernel and transport are `draft_kv_slide_tp`, the K/V projection `draft_kv_projection_tp`. Names not
  defined there fall back to the pinned module (one set of flags, markers, log lines and reasons). At two chips the slide program and the retainer are field
  for field the pinned ones.
- `scripts/ci/quad_draft_tp.py`: a live-bank mode. `QWEN_FAST_FUSED_COMMIT_LIVE_BANKS=1` is accepted only with `QWEN_FAST_FUSED_COMMIT=1` and
  `_INPLACE=1`. The quad then binds the pool's active banks and copies only where a device's bank is on the spare side (the pair's normalisation).
- Six profiles (gate-only until qualified): `c2-packed-tp4-gate-fcommit`, `-gate-fcommit-live`, `-gate-fcommit-quad` (audited); `-speed-fcommit`,
  `-speed-fcommit-quad` (timed, audits off); `-speed-fcommit-oop` (fallback: T_proj plus eager out-of-place slides).
- `c2_smoke_check.py`: under `QWEN_FAST_FUSED_COMMIT=1` the H1b gate rules (no refusal, audits == fused publications with no mismatch, only ramp or parity
  today-path reasons, no discard) stop the smoke; in `concurrent4_steady` at least one round with all four users fused, and with live banks the engaged line.
- Image copy lists: the Dockerfile COPY, the workflow context loop and the C2 overlay all carry `fused_commit_tp.py`.
- The one-card M-F0 harness (`optimisation/ttnn-op/draft_slide_inplace`) has a `--heads 2` mode (banks and deltas at two heads, 8 workers per bank, layouts
  of 1 / 2 / 5 / 10 banks per program = 8 / 16 / 40 / 80 workers, the program pinned to `draft_kv_slide_tp.prepare`) and ramp cases at drop 0.

## The proof the port carries

CPU (`test_fused_commit_tp4`, `test_quad_draft_tp4`, `test_c2_smoke_check_fcommit`, `test_c2_packed_tp4_profiles`, `test_tp2_pins`,
`test_tp4_closure_literals`, `test_tp4_fcommit_window`):

- the pair is untouched: `fused_commit.py`, `quad_draft.py`, `draft_kv_slide.{py,cpp}` and `draft_kv_history.py` are byte-pinned, and at TP2 nothing imports
  the twin;
- kernel mirrors at two heads: in place equals out of place equals the eager chain, for ramp and steady histories and every prefix 1..16, with poisoned delta
  tails and a poisoned spare; the in-place hazard model is empty for workers 0..7 and its descending-order control fires;
- a simulated request life per segment 0..3 (ramp, parity, steady, a sequential step): the device's active-bank bytes on all four chips after every commit
  equal the eager path's, and `commit_kv` never swaps under in-place;
- row independence at TP4 shapes (the T_proj sequence at 16 rows gives rows 0..prefix-1 equal to the prefix-count run);
- the guards (scope, mesh, kernel sha, the 80-worker grid), the audit (`checked=80`) and its repair.

Hardware (order in `ORDER.txt`): F0 build; F1a/F1b one card, the in-place slide at two KV heads (watcher, then full; go = every comparison exact, 1 x 80 verified,
traced at or under 0.5 ms a user); F2 audited smoke; F3 plus live banks and the singles audit; F4 plus the quad; F5a-h timed arms, eager and fused alternated
twice per shape, judged paired per round, then the position-keyed compare of every user's accepted-prefix sequence (they must be identical); F6 the exactness
matrix. If F1 fails, run the `-oop` fallback (F5x) instead of F2-F5; if the quad fails, the target is pairs + fused commit + live banks.

## Risks

In place at two heads (low: the per-worker code is identical, F1 decides); the 80-worker layout (fallback two programs of 40); T_proj device time at TP4
(estimated 1-1.5 ms a user, not measured); parity refusals after sequential steps (expected); merging the lanes branch (two blocks would share slot 0's banks,
needs its own test); 68 more traces and about 24 MB a chip of T_proj intermediates.

## Not done

The optional fused ramp (short-prompt users through one eager in-place op each, behind a TP4-only flag) is not built: it needs the F1 ramp cases first and is worth
about 17 ms a round on short-prompt mixes only.
