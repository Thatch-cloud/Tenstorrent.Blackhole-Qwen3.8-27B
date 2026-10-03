"""Lever N at TP4: keep chunked prefill on for qwen3_5 (QWEN_FAST_LEVER_N=1), as a post-import wrap of the TT platform.

The TT platform (vllm_tt_plugin.platform, plugin bf77cd63) runs _apply_chunked_prefill_policy from check_and_update_config. For every
model type outside a two-entry allowlist it logs "Chunked prefill is not supported for model_type=qwen3_5; disabling it", sets
scheduler_config.enable_chunked_prefill = False, bumps max_num_batched_tokens back up to max_model_len, and zeroes
long_prefill_token_threshold. The lever needs vLLM to keep splitting a prompt (the cap in levern_scheduler lowers the token budget per
step), so this module wraps that function the way serving_c2_contract hooks the other plugin modules: serving_c2_contract.boot registers a
PostImportHook on the platform module in EVERY process (the API server and the EngineCore both run check_and_update_config), and the hook
calls install(module).

The wrap calls the platform's own function first, then, only when that function turned chunked prefill off, puts the request's own
settings back: enable_chunked_prefill True, max_num_batched_tokens as it was (the profile keeps it equal to max-model-len, so nothing
splits except through the cap), long_prefill_token_threshold 0 (the platform's value: the base scheduler applies a nonzero threshold
before it consults the chunking flag). A model type the platform handles itself is left alone. The platform file is not edited and not
grafted; with the flag off the hook is never registered, so the platform is byte for byte what it was.

    [PINDIAG] lever N: chunked prefill kept for qwen3_5 (budget=<max_num_batched_tokens> threshold=<long_prefill_token_threshold>)

is the positive control that the wrap RAN (not merely that it was installed), logged once per process through the platform's own logger.
"""

import levern_policy

MODULE = 'vllm_tt_plugin.platform'
FUNCTION = '_apply_chunked_prefill_policy'
WRAPPED = '_qwen_levern_keeps_chunking'


def wrap(original, logger=None):
    """The wrapper installed as <module>._apply_chunked_prefill_policy."""
    state = dict(logged=False)

    def policy(vllm_config):
        scheduler = vllm_config.scheduler_config
        wanted = bool(getattr(scheduler, 'enable_chunked_prefill', False))
        budget = getattr(scheduler, 'max_num_batched_tokens', None)
        original(vllm_config)
        if wanted and not getattr(scheduler, 'enable_chunked_prefill', False):
            scheduler.enable_chunked_prefill = True
            scheduler.max_num_batched_tokens = budget
            scheduler.long_prefill_token_threshold = 0
            if not state['logged']:
                state['logged'] = True
                message = levern_policy.PLATFORM_LINE.format(budget, scheduler.long_prefill_token_threshold)
                try:
                    (logger or _logger()).info(message)
                except Exception:
                    pass

    setattr(policy, WRAPPED, True)
    policy.__wrapped__ = original
    return policy


def _logger():
    from loguru import logger

    return logger


def install(module, logger=None):
    """Wrap the platform module's chunked-prefill policy. Idempotent. A module without it is refused by name: the
    profile would serve a chunked configuration the platform silently turns off."""
    original = getattr(module, FUNCTION, None)
    if not callable(original):
        raise ValueError('%s has no %s: the TT platform this image carries is not the one Lever N was cut against'
                         % (getattr(module, '__name__', module), FUNCTION))
    if getattr(original, WRAPPED, False):
        return module
    setattr(module, FUNCTION, wrap(original, logger or getattr(module, 'logger', None)))
    return module
