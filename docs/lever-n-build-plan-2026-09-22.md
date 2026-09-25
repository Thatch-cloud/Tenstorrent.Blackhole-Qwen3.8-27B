# Lever N on the fast path: the build plan, from mechanism

**Date:** 2026-09-22. **Method:** six readers over the mechanisms a fix must change, every
load-bearing claim then handed to independent agents prompted to refute it (30 agents, 0
errors, 3.5M tokens). This supersedes the *sequencing* advice in
`docs/lever-n-fastpath-scope-2026-09-22.md`, which established the blockers; this
establishes how to attack them and corrects three things that scope got wrong.

## 1. The central question, answered

**Can `PrefillWindowCapture` be a no-op on every chunk but the last?** *Yes for features,
no for accounting — and "the last chunk" is the wrong unit.*

- **Features already are conditional.** `dflash_prefill_window.py:129` is literally
  `if piece['rows']:`; a chunk ending at or before `position - 2048` runs the original
  bare, with no `LayerOutputCapture`, no slice, no clone. The optimisation I hoped to make
  is already made.
- **Accounting cannot be skipped.** `validate_prefill_chunks` (:55-70, fired at :174)
  demands two sums: every chunk's `valid_rows` totals `position` (the whole prompt ran),
  and the window-intersecting rows total exactly `prefill_window(position)['rows']`.
- **The window straddles two calls.** At the serving ceiling `position=65504` with
  `chunk_size=2048`, the 2048-row tail is 32 rows from the last *full* chunk plus the
  2016-row tail chunk. Anything keyed on "the last call" drops the first 32 and fails at
  :68-69 and :196-197.

So the change is not "skip the capture" but **"let one capture span engine steps"**, which
is bounded and mechanical rather than a redesign.

**Four independent things break across steps** (all `dflash_prefill_window.py`):
`instance_overrides` un-installs both wrappers at the first step's exit;
`validate_prefill_chunks` fires at that same exit because `cursor < position`;
`self.started` blocks re-entry; and a fresh capture per step restarts `cursor` at 0 so the
next `chunk_start` mismatches.

## 2. Three corrections to the earlier scope

**(a) M2 composes with `OneInFlightScheduler` after all.** I previously reported that M2
"targets a class not in this path". That is wrong. `OneInFlightScheduler`'s base is an
*injected parameter* (`def one_in_flight_scheduler(base=None)`, `serving_one_in_flight.py:34`,
defaulting to `TTScheduler`), and its `_schedule_prefill_only` override is a **cooperative
wrapper that calls the patched body**. So an M2 edit to `TTScheduler._schedule_prefill_only`
*would* take effect through it. What does not apply is the route: M2's `stage()`
(`lever_n_scheduler_patch.py:270-282`) has no caller — the M2 workflow grafts by its own
loop and bypasses it.

**(b) `platform.py` can be carried by the existing graft.** The objection was that
`stage()` resolves keys under a single `root`. But `root / relative` is a pathlib join, and
an absolute right operand **replaces** the left outright, so an absolute key reaches
`/opt/qwen-fast-plugin/...` unchanged. Also `lever_n_model_patch` has no `PATCHES` table at
all — its `stage()` (:392-398) reads and writes one path — so "purely additive to PATCHES"
was answering a question that does not exist.

**(c) Image byte-identity is unverified, not established.** Whether `model.py` and
`qwen36_vllm.py` are identical between `bd878710e15c` and `e41ef884f4c8` **cannot be closed
from this repo**. It needs a probe against both images before M1's anchors are trusted on
the fast-path image.

## 3. The safety finding that outranks the rest

The GDN-scratch safety argument the design rests on — that decode between chunks is safe
because the scratch is unbound and guards would catch a violation — **does not hold as
stated**. The refutation is blunt: the guard count is wrong, two of three cited guards are
not guards (one is a constructor check that reads no address), and

> *the corruption mode Lever N actually introduces is structurally invisible to all of them.*

"Bound" is an ordinary attribute swap (`layer.B`, `layer.rec_state`, `layer.conv_states`),
and paths exist that bind without the helper, one of which never restores.

**Consequence for the build: do not rely on existing guards to catch a mis-sequenced
prefill.** The build must add its own assertion at the point of use, and the first
hardware run must be treated as unproven until a positive control shows the guard fires.

## 4. Build order

Each step is independently landable and independently measurable against the reference
baseline (run `35669428795`: `SERIAL at 14.1 s`, stall `85.8 s of 280.6 s`, 30.6%).

| # | change | cost | why first |
|---|---|---|---|
| 1 | Bind `prefill_paged_slots_range` in the capture's binding list | small | Closes the silent-corruption hazard. Valuable even if the rest is never built. |
| 2 | Make a `None` `prefill_slot` loud when the model exposes a range method | small | Today `None` legitimately means "single-sequence path", so the hazard is indistinguishable from normal. |
| 3 | Own guard for scratch single-occupancy, asserted at point of use | small | §3 — nothing existing catches this. |
| 4 | Capture start/suspend/resume/stop; `validate_prefill_chunks` at true end | medium | The core of blocker 5. |
| 5 | Lifecycle holds the capture per request; continuation branch on `num_computed_tokens > 0` | medium | Blocker 4. Split the admission contract into step-shape and request-shape. |
| 6 | Gate config: chunked on, `max_num_batched_tokens == chunk_size`, `long_prefill_token_threshold` 0 or 2048 | small | Blocker 3. Both are load-bearing: a wrong threshold sub-chunks inside vLLM and breaks `start % chunk_size == 0`. |
| 7 | Extend the m3native graft to `model.py`, `qwen36_vllm.py`, `platform.py` | small | Blocker 1, plus the probe from §2(c) for blocker 2. |
| 8 | Reconcile M2 with `OneInFlightScheduler`: cap at `partials`, not `partials + 1` | small | Stops a fresh prompt being admitted beside a suspended one. |

Steps 1-3 are safety and land first regardless of whether the rest proceeds. Steps 4-5 are
the build. 6-8 are wiring.

## 5. Invariants the build must not break

Collected from the readers, with what each protects — these are the tripwires:

- `chunk_start == self.cursor`, cursor advances by `valid_len`. Protects against a skipped,
  replayed or out-of-order chunk assembling the feature tail from the wrong absolute rows.
- Both `validate_prefill_chunks` sums. The first proves no chunk was lost across a
  suspension; the second proves the tail is exactly 2048 rows with no gap or overlap.
- `32 <= bucket <= 2048`, `bucket % 32 == 0`, `valid_len <= bucket`. The snapshot slices
  chunk-local coordinates out of a padded tensor; a wrong bucket lets padding into the tail.
- One non-nested capture per model. Two would interleave into one cursor.
- `start % chunk_size == 0`, and `max_num_batched_tokens == chunk_size` **exactly**. A
  larger budget hands the model a window it cannot replay as whole traced chunks; a smaller
  one starts a continuation mid-chunk.
- At most one prompt in the GDN prefill scratch between its first and last chunk. A second
  either re-zeroes it (`do_reset` on `start == 0`) or advances it with foreign tokens.
- `capture.prefill_slot` must record the slot before anything reads GDN row 0. The plugin
  writes the admitted user's state into `empty_slots[0]` while `ActiveSnapshot` reads row 0
  unconditionally.

One invariant has **no justification in the code**: `install` refusing when
`scheduler_config.scheduler_cls` is already set — "the code names no failure, it only
refuses to overwrite an existing selection."

## 6. How the build is judged

Not on absolute TTFT seconds — running the identical config twice moved them ~8% (13.0 vs
14.1 s prefill, 79.4 vs 85.8 s stall). Judge on:

- `prefill_serial` flipping to false, or the interval collapsing toward the round time;
- `stall_share` dropping materially — it was stable at 30.8% vs 30.6% across those two runs;
- `gate_passed` staying true and the four-user output staying token-exact.

`scripts/ci/m3native_ttft_profile.py` reports all of these on every run.
