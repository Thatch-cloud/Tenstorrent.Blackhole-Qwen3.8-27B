"""Opt-in configuration contract; importing this module does not enable TT speculation.

The model-length ceiling was originally pinned to exactly 4352 (4096 context plus a
256-token output budget) as an initial profile. Capture planning in
attention_request_plan is parameterised by position rather than tied to that number,
so the pin is relaxed to a floor: any whole number of 64-token pages at or above 4352.
The scheduler request pin is a different matter and stays - session state through
serving_lifecycle, serving_runner_bridge and the trace bucket is singular, so more
than one in-flight request needs code, not a wider contract.
"""

OUTPUT_BUDGET = 256
MINIMUM_MODEL_LEN = 4352


def fast_requested(config):
    extra = config.additional_config
    if extra is None:
        return False
    if not isinstance(extra, dict):
        raise ValueError('Explicit additional_config mapping required')
    enabled = extra.get('qwen_fast_t16', False)
    if type(enabled) is not bool:
        raise ValueError('qwen_fast_t16 must be an explicit boolean')
    return enabled


def validate_fast_config(config):
    if not fast_requested(config):
        raise ValueError('Fast serving requires explicit opt-in')
    scheduler, parallel, cache = config.scheduler_config, config.parallel_config, config.cache_config
    speculative = config.speculative_config
    if (scheduler.max_num_seqs != 1 or scheduler.async_scheduling is not False
            or parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1):
        raise ValueError('Synchronous single-request single-worker serving required; TT mesh supplies TP2')
    model_len = config.model_config.max_model_len
    if (cache.block_size != 64 or cache.enable_prefix_caching is not False
            or config.lora_config is not None or type(model_len) is not int
            or model_len < MINIMUM_MODEL_LEN or model_len % cache.block_size):
        raise ValueError('Fast profile requires 64-token pages, at least %d positions in whole '
                         'pages, no prefix cache or LoRA' % MINIMUM_MODEL_LEN)
    if (speculative is None or speculative.method != 'dflash'
            or speculative.num_speculative_tokens != 15
            or speculative.draft_sample_method != 'greedy'
            or speculative.rejection_sample_method != 'standard'):
        raise ValueError('Explicit greedy 15-proposal DFlash T16 policy required')
    return dict(scheduler_requests=1, native_gdn_slots=8, verifier_rows=16,
        physical_devices=2, context_tokens=model_len - OUTPUT_BUDGET,
        output_budget=OUTPUT_BUDGET, max_model_len=model_len,
        serving_qualified=False, performance_qualified=False)


def internal_batch_capacity(config):
    if fast_requested(config):
        return validate_fast_config(config)['native_gdn_slots']
    return int(config.scheduler_config.max_num_seqs)


def validate_request_sampling(parameters, *, prompt_tokens, eos_ids=()):
    primary_eos = getattr(parameters, '_eos_token_id', None)
    if (any(type(token) is not int or token < 0 for token in eos_ids)
            or any(type(token) is not int or token not in eos_ids for token in (parameters.stop_token_ids or ()))
            or (primary_eos is not None and primary_eos not in eos_ids)):
        raise ValueError('Only target-snapshot EOS stop tokens are supported')
    if (type(prompt_tokens) is not int or prompt_tokens != 4096
            or parameters.temperature != 0 or parameters.n != 1
            or parameters.max_tokens != 256 or parameters.min_tokens != 0
            or parameters.ignore_eos is not False
            or parameters.logprobs is not None or parameters.prompt_logprobs is not None
            or parameters.presence_penalty != 0 or parameters.frequency_penalty != 0
            or parameters.repetition_penalty != 1
            or parameters.stop
            or getattr(parameters, 'structured_outputs', None) is not None
            or getattr(parameters, 'logit_bias', None)
            or getattr(parameters, 'allowed_token_ids', None)
            or getattr(parameters, 'bad_words', None)):
        raise ValueError('Initial serving qualification requires the unmodified greedy 4K/256 coding profile')
