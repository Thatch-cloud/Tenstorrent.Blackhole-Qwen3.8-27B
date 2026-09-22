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

---

## ADDENDUM 2026-09-22 late: what four hardware attempts established

Four arms (v36 35679222511, v37 35681324335, v38 35682236496, v39 35683127469) plus
one CPU probe (35681729538) and one image build (35682784112). None has yet produced a
Lever N measurement. Each failed for a different, now-closed reason, and the sequence
is worth recording because three of the four were avoidable from what this document
already said.

| Run | Reached | Died on | Avoidable? |
| --- | --- | --- | --- |
| v36 | server start | `--no-enable-chunked-prefill`: the host env var never crossed into the container | yes - argv fact |
| v37 | config validation | `disable_chunked_mm_input` inherited from gemma4's branch | yes - argv fact |
| v38 | the fast-path lifecycle | the image's **pre-step-5** `serving_lifecycle.py` | yes - **this document said so** |
| v39 | in flight | control for the image swap | n/a |

### The delivery split, which is the real finding

Lever N's eight build steps do not share a delivery route:

- **Graft (bind-mount) reaches:** step 3 and step 7's model-tree edits - `model.py`,
  `qwen36_vllm.py`, `platform.py`, and the four m3native files.
- **Only an image build reaches:** steps 1, 2 and 4 (`dflash_prefill_window.py`),
  step 5 (`serving_lifecycle.py`), step 8 (`serving_one_in_flight.py`). All three live
  in `/experiment-scripts/ci`, baked by the Dockerfile, and mounting over that tree is
  forbidden.

The overstatement this document corrected - "graft, not image build" - was corrected in
words and then not acted on. Half the build was structurally unable to reach the rig
through the route being used, and three arms ran anyway. `fast-serving-image-v79`
(`sha256:e215968de712`, from commit 85f3fc50) is the build that carries them, with
`test_serving_lifecycle` passing at bake time.

### M1's equality gate was already passed, before any of this

Run **35416319586** (`lever-n-m1-v8`, 2026-09-19) passed it on the plain path:

    baseline_took_one_shot : true
    baseline_avoided_range : true
    resumable_took_range   : true
    chunked: true   chunk_size: 2048   lengths_checked: 3/3   gate_passed: true

Three prompt shapes - 460, 3524 and 5918 tokens, i.e. no full chunk, one chunk plus a
tail, several chunks plus a tail - all `identical: true`. The `fatal: KeyError:
'baseline'` on the resumable report is a reporting artefact of the arm that ran first
with no peer yet; the baseline arm ran second with `--peer` and produced the comparisons.

The fourth gate case, the mid-prefill short prompt, is **deliberately** not covered
there: the gate's own docstring says v1 allows one in-flight prefill per lane, so it
needs the M2 scheduler and is gated with M2.

So the open question for M1 was never "does resumable prefill produce identical
output". It was, and remains, **fast-path delivery**. Two observations sharpen that:

1. The M1 lane is plain-path **by design**. `--fast-path` exists as a flag and is passed
   by no workflow. Run 35413668471 tried it and every request died in
   `dflash_device.__init__` with `bounded prefill required`.
2. That refusal is narrower than "the 4096 pin blocks the fast path". The m3native lane
   serves 32,768-token prompts on the fast path routinely. The fast path accepts
   **qualified rungs**; the M1 gate's arbitrary 460/3524/5918-token prompts are not
   rungs, and the staged recipe is CLI-locked to 32768
   (`frozen_recipe_context.py:196-198`).

Which makes the m3native lane the correct vehicle for the fast-path question - it is the
only lane that reaches the fast path at a qualified rung with four users - and makes
v38's failure the first genuinely informative one: it proved vLLM chunks the prompt
(2048 of 32,768 scheduled) and that the fast-path lifecycle is the next wall.

### Guards added so none of these three recur

- `test_m3native_arm_env` - derives the `/bench` script list from the arm's own mount
  flags, walks each script's AST for `M3NATIVE_*` environment reads (resolving through
  helpers), and fails if one is not passed through with `-e`. Catches v36.
- `test_m3native_engine_argv` - `engine_argv` split out of `start_server` so the argv is
  assertable without launching a server. Pins both arms, the batched-budget-equals-chunk
  invariant, the multimodal zeros, and that chunking moves the prefill budget and
  nothing else. Catches v36 (12 failures) and v37 (2 failures) when either is reverted.
- `test_lever_n_model_patch` platform tests now **execute** the patched policy against
  stub configs instead of asserting on its text. The fixture is the real function,
  copied verbatim from v37's graft artifact. Catches v38's class of fault: the previous
  tests all asked whether the patched source had gained an `or`, so none could see which
  branch it then entered.

---

## ADDENDUM 2, 2026-09-22 late: step 8's delivery mechanism was never viable

Run **35690327326** (v51) carried a marker that fires from *inside*
`_schedule_prefill_only`, added precisely because `install()`'s marker had proved
nothing. It did not fire:

```
one-in-flight policy      0   -
one-in-flight scheduler   3   (the install marker, in all three logs)
```

The policy never ran. And the failure changed shape to two FRESH prompts in one step,
`prefill_slot=None new=['A','B'] cached=[]`, so neither the queue hiding nor the
capacity cap was in the loop.

**Why.** `serving_one_in_flight.install()` sets
`vllm_config.scheduler_config.scheduler_cls` to our subclass. The plugin's own
`platform.check_and_update_config` then writes it again:

```python
vllm_config.scheduler_config.scheduler_cls = TT_LANE_SCHEDULER_CLS   # line 1079
vllm_config.scheduler_config.scheduler_cls = TT_SCHEDULER_CLS        # line 1085
```

Last writer wins, and the plugin writes later. `install()` guards against a scheduler
class *already* being selected, which is true at our call site and irrelevant by the
time the config is resolved.

**This document's own contract said where M2 belongs**, before step 8 was built
(`docs/lever-n-plugin-contract-2026-09-19.md`): "two small, precisely located changes",
**both in the vLLM TT plugin, not our overlay code** -
`LaneScheduler._negotiate_forced_mode` for alternation and
`TTScheduler._schedule_prefill_only` for one-in-flight. Step 8 implemented the second as
an overlay via `scheduler_cls` instead of as a graft, and the overlay cannot survive the
plugin's own config pass.

### What the correct delivery looks like, and it is the pattern already in use

The arm already bind-mounts a patched `platform.py` into
`/opt/qwen-fast-plugin/src/vllm_tt_plugin/`, produced by
`lever_n_model_patch.patch_platform` with anchor-checked `replace_once` edits. The
scheduler needs exactly the same treatment, and both sources are already captured
verbatim by `probe_plugin_scheduler_sources` (cpu-probe run **35665853903**):

| file | sha256 | lines | what M2 edits |
| --- | --- | --- | --- |
| `scheduler.py` | `a1bd6257d3a14c90` | 207 | `TTScheduler._schedule_prefill_only`, line 154 |
| `lane_scheduler.py` | `f8e19e1907c05b24` | 758 | `LaneScheduler._negotiate_forced_mode` |

So M2 is tractable and its inputs are in hand. It is a graft of one or two plugin files,
not an overlay, and `serving_one_in_flight`'s arithmetic (`allowed_prefills`,
`effective_capacity`) is still the right policy - only its delivery was wrong.

### The lesson, which is the session's recurring one

`install()` logged `[PINDIAG] one-in-flight scheduler installed` on every run from v38
onward, and it meant only that a string had been assigned. A positive control has to
observe the behaviour, not the intention. The marker that settled this took ten minutes
to add and should have existed in step 8.
