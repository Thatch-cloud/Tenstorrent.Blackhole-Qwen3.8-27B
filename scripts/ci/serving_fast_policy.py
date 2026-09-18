"""Opt-in configuration contract; importing this module does not enable TT speculation."""


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
    if (cache.block_size != 64 or cache.enable_prefix_caching is not False
            or config.lora_config is not None or config.model_config.max_model_len != 4352):
        raise ValueError('Initial fast profile requires 64-token pages, 4352 positions, no prefix cache or LoRA')
    if (speculative is None or speculative.method != 'dflash'
            or speculative.num_speculative_tokens != 15
            or speculative.draft_sample_method != 'greedy'
            or speculative.rejection_sample_method != 'standard'):
        raise ValueError('Explicit greedy 15-proposal DFlash T16 policy required')
    return dict(scheduler_requests=1, native_gdn_slots=8, verifier_rows=16,
        physical_devices=2, context_tokens=4096, output_budget=256,
        serving_qualified=False, performance_qualified=False)


def internal_batch_capacity(config):
    if fast_requested(config):
        return validate_fast_config(config)['native_gdn_slots']
    return int(config.scheduler_config.max_num_seqs)


def validate_request_sampling(parameters, *, prompt_tokens):
    if (type(prompt_tokens) is not int or prompt_tokens != 4096
            or parameters.temperature != 0 or parameters.n != 1
            or parameters.max_tokens != 256 or parameters.min_tokens != 0
            or parameters.ignore_eos is not False
            or parameters.logprobs is not None or parameters.prompt_logprobs is not None
            or parameters.presence_penalty != 0 or parameters.frequency_penalty != 0
            or parameters.repetition_penalty != 1
            or parameters.stop or parameters.stop_token_ids
            or getattr(parameters, 'structured_outputs', None) is not None
            or getattr(parameters, 'logit_bias', None)
            or getattr(parameters, 'allowed_token_ids', None)
            or getattr(parameters, 'bad_words', None)):
        raise ValueError('Initial serving qualification requires the unmodified greedy 4K/256 coding profile')
