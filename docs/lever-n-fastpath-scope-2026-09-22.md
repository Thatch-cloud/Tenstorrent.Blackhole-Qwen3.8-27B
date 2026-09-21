# Lever N against the four-user fast path: what exists, and the five things in the way

**Date:** 2026-09-22. **Method:** six parallel readers over the Lever N surfaces, each
reader's load-bearing claims then handed to independent agents prompted to *refute* them
(30 agents, 0 errors). Corrections the refutation pass produced are marked below; they
sharpened the picture rather than overturning it, with one exception that matters.

This exists because "do the prefill ramp" turned out not to be a flag flip. M1 and M2 are
built. Neither reaches the arm with the problem.

## 1. What is actually built

**M1, resumable prefill** — `scripts/ci/lever_n_model_patch.py`, an AST-anchored text graft:

- `prefill_paged_slots_range` (:145-207), inserted after `prefill_paged_slots`
- `_prefill_traced_chunked_tp` made resumable via `chunk_from/chunk_to/do_reset/do_tail` (:77-101)
- first-chunk branch as designed: GDN reset and request-RoPE staging both gated on `start == 0` (:85-93, :117-124)
- `start % chunk_size == 0` asserted (:132-135)
- dispatch on a **nonzero** start, with an `[M1]` log marker per branch as a positive control (:256-275)
- a `platform.py` opt-in, `TT_M1_FORCE_CHUNKED_PREFILL`, because
  `platform._apply_chunked_prefill_policy` refuses chunking for `model_type=qwen3_5` outright

It is a **reduced** form of design §3.1. The intermediate branch is deliberately absent:
every step emits logits and writes its GDN slot, with discarding delegated to the vLLM
runner's `intermediate_prefill_mask` (:152-158, justified in
`docs/lever-n-plugin-contract-2026-09-19.md:53-72`). `is_last` is not in the signature, and
a test pins that absence. The `max_num_batched_tokens == chunk_size` invariant is not
asserted in the graft at all — it lives in `lever_n_prefill_range.py:110-119`, a host-side
planning module nothing but its own test imports.

**M2, alternation** — `scripts/ci/lever_n_scheduler_patch.py`, 295 lines, 19 passing tests.
Both edits exist in text form. But it is not the M2 §3.3 describes: neither
`TT_PREFILL_DECODE_INTERLEAVE` nor `TT_DECODE_STEPS_PER_PREFILL_CHUNK` appears anywhere in
it. The ratio is a hardcoded `ALTERNATION_PERIOD = 2` interpolated as the literal
`step % 2 == 1`, and the feature is gated by whether the run-arm bind-mounts the patched
files, not by any environment variable.

**Gates.** M1 has run green on hardware (run `35416319586`, 3/3 lengths, all controls).
**M2 has never run**: its workflow fires on `experiment/lever-n-m2-v*` and no such tag has
ever existed — 39 lever-n tags, all m1 or m3native.

## 2. The five blockers

None of these is a wiring oversight that a mount line fixes. They are independent.

**(1) Not grafted.** The m3native graft job stages exactly four decode-side files —
`model_config.py`, `attention/tp.py`, `gdn/tp.py`, `mlp.py`
(`lever_n_m3native_patch.py:466-471`). `model.py`, `qwen36_vllm.py`, `platform.py`,
`scheduler.py` and `lane_scheduler.py` are never extracted, never patched, never mounted.
`prefill_paged_slots_range` does not exist in that container.

**(2) Three different images.** M1/M2 are pinned to `sha256:bd878710e15c…`; the m3native
lane that produced the ramp to `sha256:e41ef884f4c8…`; fp2u to `sha256:8312d86ad35f…`. M1's
anchors have never been matched against the fast-path image's sources, and an M1 result does
not transfer.

**(3) The engine config forbids it.** The gate starts vLLM with
`--no-enable-chunked-prefill` and `--max-num-batched-tokens` equal to the whole context
(`lever_n_m3native_gate.py:159,162`). The runner therefore never sends a nonzero
`start_pos`, so M1's resumable branch is unreachable; and no request ever carries
`is_prefill_chunk=True`, so M2's `has_partial_prefill` guard can never fire.
`TT_M1_FORCE_CHUNKED_PREFILL` is not set in that arm.

**(4) The fast lifecycle refuses partial prefills by contract.**
`serving_lifecycle._execute` demands `num_computed_tokens == 0` and
`total_num_scheduled_tokens == len(prompt_token_ids)` on a single brand-new request
(:110-114). An M1 continuation chunk is rejected with *"Text-only uncached complete prompt
required"* before it reaches the model.

**(5) `PrefillWindowCapture` forbids the shape M1 produces — and fails silently.** This is
the one that is not merely an incompatibility:

- `wrap()` raises when `chunk_start != self.cursor`, requiring absolute order from 0 (`dflash_prefill_window.py:124`)
- `capture()` refuses re-entry; `close()` raises while active; `validate_prefill_chunks`
  demands contiguous full coverage before `complete` is set (:154-155, :160-181)
- `wrap_slots()` refuses a second batched prefill per capture (:205-207)

A capture is one prompt, one uninterrupted pass. Worse: the capture wraps
`prefill_paged_slots` but **not** `prefill_paged_slots_range`. Under M1 a resumed prompt
dispatches to the range method, so `capture.prefill_slot` stays `None`,
`adopt_prefill_slot` logs "nothing to adopt", and **a second user decodes from whatever sits
in GDN row 0**. Wiring M1 in naively produces wrong output, not an error.

## 3. Corrections from the refutation pass

Three claims were knocked down. Two were overstatements; one is material.

**Material — the scheduler is the wrong class.** The measured arm serves with
`qwen_fast_t16=True` and `--max-num-seqs 4`, so `serving_fast_policy.validate_fast_config`
installs `serving_one_in_flight.OneInFlightScheduler` as `scheduler_cls` whenever
`max_num_seqs > 1` (`serving_fast_policy.py:96-105`). That class subclasses **`TTScheduler`,
not `TTLaneCoordinator`**, and caps prefill-step capacity at `partials + 1 + decodes`. M2's
alternation edit targets `TTLaneCoordinator._negotiate_forced_mode` — a class not in this
path. M2 is aimed at the wrong seam for this arm, independently of blockers 1-4.

**Overstatement — "no lock anywhere".** True as written (no `threading` primitive appears in
any `serving_*.py`), but the grep tokens were the wrong instrument: serialisation here is
built from occupancy flags and scheduler mode votes, not threading objects. The conclusion
stands; the reasoning did not.

**Overstatement — "graft, not image build".** The bind-mount route is correct for the model
and plugin files, but `docker/qwen-fast-serving.Dockerfile` bakes about 30 `scripts/ci/*.py`
into `/experiment-scripts/ci` at build time, and that is the **only** route for the
fast-path overlay (`serving_*.py`, `dflash_*.py`, `two_tile_*.py`, `model_batch.py`) — no
lever-n run-arm bind-mounts over that tree. So a change to blockers 4 or 5 is an image
build, not a mount.

## 4. What no existing gate would catch

Neither M1 nor M2's gate measures TTFT, and neither ever has two prefills in flight. M1
issues its three prompts sequentially in one thread; M2 runs exactly one decode and one
prefill. The 13.5/26.5/39.6/52.5 s staircase has **no gate that would reproduce or detect
it**. A four-user TTFT gate is a prerequisite for calling any of this fixed, and it does not
exist yet.

## 5. Consequences for sequencing

The cheap, unblocked lever is prefill **throughput**, not scheduling: the server's own
startup log records `get_max_prefill_chunk_size` not recognising Qwen3.8-27B on P300 and
falling back to `MAX_PREFILL_CHUNK_SIZE = 4` "for compatibility", with the vendor's next
line inviting powers of two up to 128. The stall is exactly the remaining prefill time, so
cutting 13.1 s cuts the 79.4 s proportionally, with no resumable prefill, no capture
surgery and no scheduler change. `scripts/ci/probe_prefill_chunk_size.py` reads that
function and its call sites before anything is changed, because raising it risks L1
exhaustion.

The expensive path — Lever N on the fast path — is a real build, not a wiring job. In
dependency order it needs: the fast-path overlay changed (image build) so the lifecycle
admits partial prefills; `PrefillWindowCapture` taught to span suspended chunks across
engine steps *and* to wrap the range method; M2 re-aimed at `OneInFlightScheduler`; the
three images reconciled; and a four-user TTFT gate built to prove any of it worked.
