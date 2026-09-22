"""Why enabling chunked prefill for Qwen3.5 makes vLLM refuse to start.

Run 35681324335 (Lever N v37) proved the chunk flag now reaches the engine - and the
engine died before readiness:

  ValueError: Chunked MM input disabled but max_tokens_per_mm_item (16384) is larger
  than max_num_batched_tokens (2048). Please increase max_num_batched_tokens.

The cause is in the graft, not in vLLM. lever_n_model_patch.patch_platform opens
_apply_chunked_prefill_policy's allowlist:

    if model_type in _CHUNKED_PREFILL_MODEL_TYPES or _m1_chunked_prefill_opt_in():
        scheduler_config.disable_chunked_mm_input = True
        return

so qwen3_5 now takes the branch written for gemma4 - and that branch sets
disable_chunked_mm_input, which is exactly the flag vLLM refuses to honour when one
declared MM item (16384 tokens) exceeds the batched-token budget (2048).

Qwen3_5ForConditionalGeneration resolves to the TT class Qwen36ForCausalLM, which is
TEXT-ONLY and cannot consume an image at all, so the 16384-token item is a phantom of
the HF config's nested vision config. Two candidate fixes, and this probe exists to
pick between them on facts rather than on my reading of vLLM:

  A. Do not set disable_chunked_mm_input on the opt-in path. Note this is CLOSER to
     today's behaviour than setting it: for qwen3_5 the shipped function never touches
     the field, so the vLLM default applies. Needs no CLI change.
  B. Pass --limit-mm-per-prompt with zeros so no modality is nonzero and the budget is
     skipped. Needs a new engine flag on the chunked arm only.

Questions, in the order that decides it:
  1. Is this probe image's vLLM the SAME one that raised? The traceback pins the raise
     to encoder_cache_manager.py:302; if that line is not the raise here, every other
     answer is about the wrong build and the verdict must say so.
  2. What is disable_chunked_mm_input's default? A is only 'closer to today' if the
     default is False.
  3. Does the raise depend on the flag, so that A avoids it outright?
  4. Would zero mm limits empty the nonzero-modality map, so that B avoids it too?
  5. Does this vLLM accept limit_mm_per_prompt at all, and spelled how?
"""

import inspect
import sys

VERDICT = []


def note(line):
    print(line)


def main():
    import vllm
    note('vllm version: %s' % vllm.__version__)
    note('python: %s' % sys.version.split()[0])

    from vllm.v1.core import encoder_cache_manager as ecm
    source_file = inspect.getsourcefile(ecm)
    note('encoder_cache_manager: %s' % source_file)

    lines = open(source_file, encoding='utf-8').read().splitlines()

    # Q1: the traceback pinned the raise to line 302 of this file.
    at_302 = lines[301].strip() if len(lines) > 301 else '<file shorter than 302 lines>'
    note('line 302 here: %r' % at_302)
    same_build = at_302.startswith('raise ValueError')
    note('Q1 same build as the run: %s' % same_build)

    # Q3: the budget function itself, verbatim - the only authority on the condition.
    note('--- compute_mm_encoder_budget ---')
    budget_src = inspect.getsource(ecm.compute_mm_encoder_budget)
    for line in budget_src.splitlines():
        note('  ' + line)

    guarded_by_flag = 'disable_chunked_mm_input' in budget_src
    note('Q3 raise is guarded by disable_chunked_mm_input: %s' % guarded_by_flag)

    nonzero_gated = 'nonzero_modality' in budget_src or 'nonzero' in budget_src
    note('Q4 budget consults a nonzero-modality map: %s' % nonzero_gated)

    # Q2: the default, from the dataclass field rather than from a docstring.
    from vllm.config import SchedulerConfig
    import dataclasses
    default = '<absent>'
    for field in dataclasses.fields(SchedulerConfig):
        if field.name == 'disable_chunked_mm_input':
            default = field.default
    note('Q2 SchedulerConfig.disable_chunked_mm_input default: %r' % (default,))

    # Q4 continued: does an empty limit really empty the map?
    try:
        from vllm.multimodal.registry import MultiModalRegistry
        names = [n for n in dir(MultiModalRegistry) if 'max_tokens' in n]
        note('registry max_tokens helpers: %s' % sorted(names))
        for name in names:
            if 'nonzero' in name:
                note('--- %s ---' % name)
                for line in inspect.getsource(getattr(MultiModalRegistry, name)).splitlines():
                    note('  ' + line)
    except Exception as error:
        note('registry inspection failed: %s: %s' % (type(error).__name__, error))

    # Q5: the CLI surface.
    try:
        from vllm.engine.arg_utils import EngineArgs
        fields = {f.name for f in dataclasses.fields(EngineArgs)}
        note('Q5 limit_mm_per_prompt is an EngineArgs field: %s'
             % ('limit_mm_per_prompt' in fields))
        note('Q5 mm-ish EngineArgs fields: %s'
             % sorted(f for f in fields if 'mm' in f or 'multimodal' in f))
    except Exception as error:
        note('EngineArgs inspection failed: %s: %s' % (type(error).__name__, error))

    if not same_build:
        return ('INCONCLUSIVE - this image is not the build that raised; '
                'answers below describe a different vLLM')
    parts = []
    parts.append('fix A viable' if guarded_by_flag and default is False
                 else 'fix A NOT established')
    parts.append('fix B viable' if nonzero_gated else 'fix B NOT established')
    return '; '.join(parts)


if __name__ == '__main__':
    try:
        outcome = main()
    except Exception as error:
        import traceback
        traceback.print_exc()
        outcome = 'PROBE FAILED: %s: %s' % (type(error).__name__, error)
    print('VERDICT: %s' % outcome)
