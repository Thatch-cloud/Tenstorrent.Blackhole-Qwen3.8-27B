# Lever N at TP4, stage 1: what is built, where it departs from the design, what is left

**Date:** 2026-10-04. **Branch:** `tp4/lever-n`, from `origin/tp4/262k8` (52ad5839). **Design:** `docs/lever-n-tp4-design-2026-10-04.md`.
**Status:** CPU side built and tested; nothing has run on a card. Every Lever N profile is gate-only and carries the 262k evidence waiver, so every result of the window is UNQUALIFIED.

## What stage 1 is

A long prefill is split into engine steps that each end on the model's own 2,048-token chunk boundary, with decode rounds between them, behind `QWEN_FAST_LEVER_N=1`. With the flag off nothing
changes: `levern` is `None` in the lifecycle, the platform wrap is never registered, the scheduler class is wrapped exactly as before, and no Lever N module is imported.

| Piece | File | What it does |
| --- | --- | --- |
| The arithmetic and the alternation | `levern_policy.py` | `final_start`, `step_budget`, `plan` (the final step is the last full chunk plus the tail, at most 4,095 tokens, so the drafter window and the engine build are the whole-prompt path's; a prompt under 4,096 tokens never splits); the `Alternator` (time share f, or static R decode steps per prefill step; owed time capped at KMAX rounds; nothing owed without a decoder; the clock is a parameter); every flag parsed strictly (a step that is not a multiple of 2,048 is refused). Stdlib only. |
| The scheduler side | `levern_scheduler.py`, `serving_prefill_admission.py` | The cap: `Scheduler.max_num_scheduled_tokens` is lowered for the one call that advances the partial (or the one fresh prompt), restored in a `finally` (a scheduler without that attribute is capped through `long_prefill_token_threshold`). The alternation: `schedule()` runs the plugin's own `_schedule_decode_only` when the decoders are owed a step. The final-step DRAM gate. One `lever N step` line per decision while a prefill is pending (kind, seats, request, start, tokens, end, prompt, final, reason, the previous step's wall time, what is owed). |
| The platform | `levern_platform.py`, `serving_c2_contract.py` | A post-import wrap of `vllm_tt_plugin.platform._apply_chunked_prefill_policy` (armed by the contract in every process of a Lever N profile) puts `enable_chunked_prefill`, `max_num_batched_tokens` and `long_prefill_token_threshold` back after the platform turned them off; logs `lever N: chunked prefill kept for qwen3_5 (budget=... threshold=0)`. `levern_problems` refuses a profile that breaks any requirement (the contract also refuses it at boot). |
| The model route | `levern_route.py`, `serving_runtime.py` | `prefill_paged_slots_range` installed on the model instance at the attach, and a wrapper on the vLLM model's `prefill_forward`. It continues the persistent B=1 GDN scratch at `start > 0` through the G1 stage's own `prefill_traced_chunked(start=...)`, under an **owner token** (`(request id, next start)`: a continuation with any other owner is an `AssertionError` before any device work), writes the decode slot and snapshots the scratch **only on the final step**, and warms itself over a three-step 6,208-token prompt before any trace is captured. |
| The lifecycle and the capture | `serving_lifecycle.py`, `dflash_prefill_window.py` | The lifecycle announces each step (`Step(request id, start, end, total)`) for the one `execute_model` call and withdraws it in a `finally`; keeps the resident engine resident across a step that wrote no slot; releases the owner when the request finishes or is aborted mid-prefill; refuses to attach a Lever N profile whose scheduler cap could not be installed. The capture reads what the route reports (`segment_wrote_slot`). |
| The audit | `levern_route.py` | `QWEN_FAST_LEVERN_AUDIT=1`: at the end of every prefill a digest line (GDN slot, last-position logits, KV pages [0, P)); on its own it is the non-interleaved control that produces the digests the interleaved arm is compared with. |
| Profiles | `qwen_c2_profiles.json` | Six gate-only eight-seat 262k profiles (`c2-packed-tp4-8x262k-best-levern-*`): `time-gate` (share 0.5, the timed arm), `r1-time-gate` (static R=1), `audit` (the exactness arm), `control-audit` (digests, chunking off), `final-hold-time-gate` and `foreign-time-gate` (negative controls). The control of a timing arm is the existing `best-time-gate`. |
| Smoke tests and rules | `c2_serving_smoke.py`, `c2_smoke_check.py`, `levern_compare.py` | `levern_equal` / `_long` / `_busy` (prompts of EXACTLY 2,047 .. 253,920 tokens as ids, 256 out), the hang shapes (`levern_decoder_finishes`, `_all_decoders_finish`, `_cancel_mid_prefill`, `_arrival_during_prefill`, `_seed_stops`), `stall8_cold128k` and `stall8_cold262k` extended with each seat's chunks and estimated tokens a second inside the arrival's prefill window. `levern_problems` in the check: the engaged markers, the route ledger, the alternation, every seat progressing in the window, the digests. `levern_compare.py`: the two arms answer for answer and digest for digest. |
| The job pack | `references/tp4-lever-n-jobs` | Image `tp4-lever-n-1`: X0 status rescan reset, B0 build, A0c control digests, A1 audited interleaved attach (compared), H1-H3 hang shapes, F1 final-hold control, S1-S4 stall ABAB (control against interleaved, 128k and 254k arrivals), S5 static arm, F2 foreign-owner control, Z. No agent start or stop. |

## Where it departs from the design

1. **The route is installed at runtime, not an AST stage.** The design put the route in a stage after G1's in `qwen_prefix_stage.STAGES`. The G1 stage already puts the resumable loops and `prefill_traced_chunked(start=)` in
   every C2 build's `model.py`, so the route needs only instance attributes: no sha256 pin moves, nothing in the plugin or the model tree is edited, and the route is tested by executing the REAL staged `model.py` on the fake
   ttnn (`test_levern_route`). `levern_route.install` refuses a model tree without those loops by name.
2. **The step identity comes from the lifecycle, not the runner.** The plugin runner passes `start_pos` and `prompt_lens` but no request id and not the prompt's full length (the design assumed a runner patch). The lifecycle
   knows both before it calls the worker, so it announces them; the wrapper checks them against the runner's own `start_pos` and `prompt_lens` before any device work.
3. **A whole prompt takes the scratch owner away.** In the design every row, whole prompts included, goes through the route. Here a whole prompt (and any unannounced prefill, the warm requests) runs the stock path untouched and
   clears the owner, so a suspended prompt's continuation after it is refused instead of running on another prompt's state.
4. **Two negative-control profiles** (`final-hold`, `foreign`) and the audit-only control profile exist because a job cannot override a profile's environment.
5. **The digests include the KV pages** (read one cache tensor at a time) on both arms, in addition to the GDN slot and the logits; the drafter window is checked by the capture's own snapshot comparisons. The KV read
   copies every cache tensor of the whole pool to the host whatever the prompt length, so only prompts of at most 32,785 tokens (the single-user `levern_equal` rows and the busy rows' three prompts) take it; longer
   prompts log `kv_sha` as 32 zeros on both arms ("not taken"). Budget a digest at about a minute and the A0c and A1 estimates accordingly. Each digest line also carries `tokens_sha`, the prompt's own identity: the
   arms are compared per (length, token sha), never by length and log order.
6. **The hang jobs run on their own profile** (`-levern-hang-gate`: the timed arm plus `QWEN_FAST_STALL_DEADLINE_S=120` and `QWEN_FAST_CCL_HANDLE_GUARD=log`; the refusing mode fails passing runs), because a hang on the
   time gate burned the job timeout and recorded nothing.
7. **The scheduler verifies the pass it capped.** After the base scheduler ran under the cap, `LevernRuntime.verify` checks that no request but the capped one was admitted and that a non-final end is a multiple of
   2,048 inside a prompt of at least 4,096 tokens; otherwise it logs a `REFUSED` line and raises (the engine stops before any device work), instead of handing the route a step it cannot continue.
8. **`wrote_slot` and the program rule are measured.** The route counts the real `_write_gdn_slot` calls of each step, and logs the drafter-window snapshot's program count (`window=`): the rule is
   after - before - window == 0 (the four-card tripwire's own B-A-W), the warm requests exempt.

## What is NOT done

- **Nothing has run on a card.** The window (about 19 h of cards, the pack's ORDER.txt) is the next step; B0 needs a fresh tag `tp4-lever-n-1` built from the pushed commit.
- **The real-vLLM proof is written but unrun.** `test_levern_scheduler_vllm` (the attribute name, the exact split of a 9,000-token prompt by the cap alone, the budget put back, no preemption under the KV reservation) skips
  where vLLM is not installed and runs in `qwen-fast-vllm-cpu.yml`, which a tag push triggers; it has not been triggered. Until it passes, `max_num_scheduled_tokens` as the cap is an assumption (the fallback through
  `long_prefill_token_threshold` is tested on the reduced model only).
- **v2** (scratch parking, shortest-remaining-first among prefills, more than one prefill in flight), **composition with prefix reuse** (`tp4/packed-prefix`: its CHECKPOINT source would join the owner token; the contract
  refuses `QWEN_PREFIX_REUSE` and sticky sessions with the flag), **the fast lane** (refused), **`agent8_turns`** (G-N5, the shape that chooses the share f: the pack runs f = 0.5 and R = 1 only, so f stays at its default), the optional layer-granularity research of design 3.12, and three
  design items dropped from this stage on purpose: **G-N6 churn** (the churn16 admission shape; the eight-seat quad's own churn jobs cover admission and the KV reservation on the base, and the hang jobs here cover the
  Lever N specific aborts), **the H3 per-round assertion** that every GDN layer's `rec_state` is the batched buffer (the route binds and unbinds the scratch in one try/finally and the owner token refuses a foreign
  continuation; a decode round that ran on the scratch would show as a digest mismatch in A1, which is the stronger check), and **the trace-census arm** (the route allocates nothing after the warm, which the program rule
  and the existing unsafe-allocation marker read).
- Hardware assumptions the CPU tests cannot see: that the text RoPE a step builds for `tokens[:end]` is the one the whole prompt's staging gives (the toy model has no RoPE; G-N1 reads it), that an intermediate step's
  logits (the exact-multiple branch of the eager loop) are harmless and discarded by the runner, that the device scratch is bitwise what the host fixture's is across a synchronize, and every timing in the design's section 6.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `QWEN_FAST_LEVER_N` | 0 | the master switch (gate-only profiles; strictly 0 or 1) |
| `QWEN_FAST_LEVERN_STEP_TOKENS` | 2048 | the step while decoders run; a multiple of 2,048 or refused |
| `QWEN_FAST_LEVERN_SOLO_STEP_TOKENS` | 16384 | the step with no decoder and nothing waiting; a multiple of 2,048, at least the step |
| `QWEN_FAST_LEVERN_PREFILL_SHARE` | 0.5 | f in (0, 1]: the prefill's share of wall time while decoders run (1.0 is back-to-back chunks) |
| `QWEN_FAST_LEVERN_ROUNDS` | unset | static R in 1..64 decode steps per prefill step (overrides f) |
| `QWEN_FAST_LEVERN_MAX_ROUNDS` | 8 | KMAX, the most rounds owed at once |
| `QWEN_FAST_LEVERN_AUDIT` | 0 | digests at the end of every prefill; also valid alone (the non-interleaved control) |
| `QWEN_FAST_LEVERN_FAULT` | unset | negative control: `final-hold` (hold one final step) or `foreign` (a continuation with a wrong owner must refuse) |

The contract (`serving_c2_contract.levern_problems`) requires with `QWEN_FAST_LEVER_N=1`: a gate-only profile; `QWEN_FAST_ANY_REQUEST=1` and `QWEN_FAST_TP=4` and the fast path; `enable-chunked-prefill` in place of
`no-enable-chunked-prefill`; `max-num-batched-tokens` equal to `max-model-len`; `no-async-scheduling`; block size 64; text only; `QWEN_FAST_KV_RESERVATION=1`; no `QWEN_FAST_DECODE_STEPS_PER_ADMISSION`, no
`QWEN_FAST_LANE`, no prefix reuse or sticky sessions; and no sibling flag without the master switch.

## Markers (what a gate reads)

    [PINDIAG] lever N installed on <class>: step=2048 solo=16384 share=0.5 rounds=- max_rounds=8
    [PINDIAG] lever N: chunked prefill kept for qwen3_5 (budget=262144 threshold=0)
    [PINDIAG] lever N route installed: route=1 audit=<0|1>
    [PINDIAG] lever N route warmed before the packed traces: steps=3 programs=<A>-><B> ms=<..>
    [PINDIAG] lever N step n=<k> kind=<prefill|decode> seats=<d> req=<id> start=<s> tokens=<t> end=<e> prompt=<P> final=<0|1> reason=<..> prev=<kind>:<ms>ms owed_ms=<..> owed_rounds=<..>
    [PINDIAG] lever N route req=<id> start=<s> end=<e> prompt=<P> final=<0|1> wrote_slot=<n> ms=<..> programs=<A>-><B> window=<W>
    [PINDIAG] lever N digest req=<id> prompt=<P> tokens_sha=<32 hex> slot_sha=<32 hex> logits_sha=<32 hex> kv_sha=<32 hex>
    [PINDIAG] lever N final-step dram hold req=<id> prompt=<P> decodes=<d> short=<terms>
    [PINDIAG] lever N REFUSED <why>                      a problem for the check

## Tests (all on the CPU suite; `qwen-integration-cpu.yml` has a `lever-n` job for them)

`test_levern_policy` (the arithmetic at every boundary length, the alternation on a fake clock, the flags), `test_levern_scheduler` (the cap and the alternation on the plugin's own scheduler over a chunking
model of vLLM's, the final gate, the install refusals, flag-off identity), `test_levern_platform` (the pinned platform policy executed), `test_levern_contract` (the profiles, each refusal, the boot hook),
`test_levern_route` (the real staged `model.py`: a split prompt equals the whole one byte for byte, the owner token, the audit, the warm), `test_levern_lifecycle`, `test_levern_capture`, `test_levern_check` (the smoke
rules and `levern_compare`, with negative controls), `test_levern_smoke` (the smoke tests against a fake server), `test_tp4_lever_n_jobs` (the pack), `test_levern_scheduler_vllm` (installed vLLM only).
