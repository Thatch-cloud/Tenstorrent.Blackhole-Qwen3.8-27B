# 131k one-user attach arm: what the env can and cannot do

## The earlier 131k qualification is a different code path

The 53.69 committed TG tok/s single-stream 131,072 result
(`docs/combined-context-ladder.md:34`, `docs/runtime-reproduction.md:38-49`) was produced
by `qwen-combined-ladder.yml`, which checks out a **historically pinned commit**
`8c102b20df22329106955b4006bf4d650bb94e40` (`.github/workflows/qwen-combined-ladder.yml:26`)
and drives it with `dspark_request_experiment.py` directly - no vLLM `api_server`, no
`serving_fast_policy.py`, no `verifier_engine.py`, no `dflash_device.py`
(`scripts/ci/dspark_request_experiment.py:1-12` imports none of them). `frozen_ladder_stage.py`
generalises that harness's pins via `selected_geometry()['context']`
(`scripts/ci/frozen_ladder_stage.py:8-60`), which is why it can run any `CONTEXTS` rung
including 131072. **This is a disjoint lineage from the m3native gate's vLLM+packed-decode
stack**, per memory `serving-image-bundle-provenance` (bundle is `77d6995a`, not `8c102b20`).

## The vLLM-servable T16 recipe is staged CLI-locked to 32768, not selected_geometry()

The tree that becomes `frozen_combined_runtime.py`/`target_t16_attention_gate.py` inside the
actual serving bundle is produced by `frozen_recipe_context.py --combined-runtime`, and its own
argparse guard refuses anything else:

```
scripts/ci/frozen_recipe_context.py:196-198
    if options.combined_runtime and (options.context != 32768 or not options.scalar_reciprocal
            or not options.target_replay or options.eager_only):
        parser.error('Combined candidate requires 32768, scalar reciprocal and target replay, not eager-only')
```

Every literal `adapt_combined_sources` emits (`scripts/ci/frozen_combined_adapters.py:14-86`) is a
hard `32768`/`33024`, not `selected_geometry()`-driven - unlike the ladder's generalised copies of
the *same-named* files. `frozen_runtime_context.py:31-36` does widen `dspark_context_selection.
request_context()`'s admitted **string** set to all of `frozen_context_geometry.CONTEXTS`
(so `QWEN_DSPARK_REQUEST_CONTEXT=131072` parses without raising), but nothing downstream of that
acts on the wider value - so the widening is cosmetic here.

## Exact refusal chain at 131,072 with the packed 64-row block engaged (QWEN_FAST_FOUR_AS_TWO=0)

1. **First, prefill itself**, unless `QWEN_FAST_MAX_POSITION` is raised:
   `dflash_prefill_window.MAX_POSITION` defaults to 65504 (`scripts/ci/dflash_prefill_window.py:15`)
   and `prefill_window(position)` requires `1 <= position <= MAX_POSITION`
   (`scripts/ci/dflash_prefill_window.py:41-42`), raising
   `ValueError: Absolute prefill position within the target allocation and verification headroom required`.
   The run-arm now sets `QWEN_FAST_MAX_POSITION` to the requested context so this clears.

2. **Then, per-request engine construction**, unconditionally, regardless of (1):
   `VerifierEngine.__init__` calls `target_t16_attention_gate.validate_request_option`
   (`scripts/ci/verifier_engine.py:162-166`). In the staged bundle this function's only
   non-4096 branch is `if request_context() == 32768: ... validate_target_option` (staged from
   `scripts/ci/frozen_combined_adapters.py:53-55`). At `request_context()==131072` that branch is
   skipped, so the **base** clause runs
   (`scripts/ci/target_t16_attention_gate.py:25-29`, pre-stage source shown - the staged copy is
   the same shape after the 8192->32768 rename) and raises:
   `ValueError: Qualified T16 experiment requires a bounded 4096..4352 request and native sampling`
   This fires on the FIRST verify-engine build, i.e. after prefill has already spent device time,
   and **before** `serving_runtime.py:261`'s `[PINDIAG] dram after engine` print - so that line
   will not appear for this arm even though attach succeeds. (Redundant if ever reached:
   `frozen_combined_runtime.qualify()` also hard-refuses at `scripts/ci/frozen_combined_runtime.
   py:78`, `selected_geometry()['context'] != 32768` - but (2) always raises first.)

**Not blockers at 131,072** (checked and cleared): `serving_fast_policy.validate_fast_config` only
requires `model_len >= 4352` and a whole 64-page multiple (`scripts/ci/serving_fast_policy.py:
85-90`) - 131,328 % 64 == 0, fine. `page_width` is derived from `max_model_len`
(`scripts/ci/serving_runtime.py:77`), not hardcoded. `capture_widths`/`VERIFY_WIDTHS` are
context-independent (`scripts/ci/verifier_engine.py:64-75`). None of the hash-pinned files
(`dflash_combined_sim_runtime.py`, `dflash_t16_native_scope.py`, `dflash_t16_native_attention.py`,
`attention_replay.py`, `attention_batch.py`, `gdn_multitoken_conv.py`, `gdn_commit_dma.py`,
`verifier_inputs.py`, `packed_cache_writer.py`) contain a literal 32768/33024 pin.

## What this means for the arm

`[PINDIAG] dram after attach` **will** print - attach is context-independent
(`serving_runtime.py:236`, before any request). The per-request `[PINDIAG] dram after engine`
line, `native_m3` marker, `packed_phase`, and any decode round **will not** appear: the request
is refused inside engine construction before a token is verified. This is a real, informative
result (attach headroom at 131k, confirmation the fast T16 recipe is unreachable there today) -
not a partial pass. Reachability needs a new staged combined-runtime tree at 131072 (rerunning
`frozen_recipe_context.py --combined-runtime --context 131072` after relaxing its 32768 guard and
requalifying every downstream literal) - the multi-day rebuild tracked as task #36/#38.
