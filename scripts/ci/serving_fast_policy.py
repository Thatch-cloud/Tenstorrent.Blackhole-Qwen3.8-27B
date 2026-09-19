"""Opt-in configuration contract; importing this module does not enable TT speculation.

The model-length ceiling was originally pinned to exactly 4352 (4096 context plus a
256-token output budget) as an initial profile. Capture planning in
attention_request_plan is parameterised by position rather than tied to that number,
so the pin is relaxed to a floor: any whole number of 64-token pages at or above 4352.
The scheduler request pin was lifted on 2026-09-19, to a ceiling of NATIVE_GDN_SLOTS
rather than a wider contract in general. Read the next paragraph before assuming that
means concurrent serving works.

It probably does not yet. Session state through serving_lifecycle,
serving_runner_bridge and the trace bucket is STILL singular - serving_lifecycle
holds one request_id, one capture and one hook - so more than one in-flight request
needs code. Widening the contract does not supply that code; it only allows a second
request far enough in to find out where the singular state actually fails, which is
the thing four separate probes could not reach while the contract refused at config
construction. If a run gets past this module and dies in the lifecycle, that is the
expected next result, not a regression.

What did NOT move: the TT mesh supplies TP2 itself, so host-side tensor, pipeline and
data parallel are all still required to be 1, and every sampler equality that makes
output bit-comparable is untouched.
"""

OUTPUT_BUDGET = 256
NATIVE_GDN_SLOTS = 8
# The proposal and verify blocks are 32 rows. capture_widths caps a bucket at 32 and
# dflash_device accepts block_rows in (8, 16, 32), so a packed block divides those 32
# rows among its users: two T16 users, or four T8 users. FOUR T16 USERS DO NOT FIT -
# they need 64 rows - and dropping to T8 to fit four cuts each user from 15 proposals
# to 7, which is a direct cut to committed tokens per cycle.
PACKED_BLOCK_ROWS = 32


def packed_geometry(users):
    """Rows per user in the shared 32-row block, and the proposals that buys."""
    if type(users) is not int or users < 1 or PACKED_BLOCK_ROWS % users:
        raise ValueError('Packed users must divide the %d-row block' % PACKED_BLOCK_ROWS)
    rows = PACKED_BLOCK_ROWS // users
    if rows not in (8, 16, 32):
        raise ValueError('Each packed user needs a supported T8/T16/T32 share of the block')
    return dict(users=users, block_rows=rows, verifier_rows=PACKED_BLOCK_ROWS,
                proposals_per_user=rows - 1)
MINIMUM_MODEL_LEN = 4352
# Prompts are bounded by the served context rather than pinned to 4096; the server
# still rejects anything past max_model_len before a request reaches this check.
MAXIMUM_PROMPT_TOKENS = 1 << 20


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
    # max_num_seqs was pinned to 1. Lifted 2026-09-19 to admit concurrent requests:
    # the bound that matters on the device is native_gdn_slots, which is 8, and
    # internal_batch_capacity already returns it rather than max_num_seqs. The rest
    # of this condition stays exact - the TT mesh supplies TP2 itself, so a second
    # HOST-side parallel axis is a different thing and is still refused.
    _seqs = scheduler.max_num_seqs
    if (not isinstance(_seqs, int) or not 1 <= _seqs <= NATIVE_GDN_SLOTS
            or scheduler.async_scheduling is not False
            or parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1):
        raise ValueError('Synchronous serving of at most %d requests on a single worker '
                         'required; TT mesh supplies TP2' % NATIVE_GDN_SLOTS)
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
    if _seqs > 1:
        # Probe 35436384975: the plugin scheduler batches SIMULTANEOUS prefills into
        # one step, and the fast prefill path takes one capture and executes one
        # prompt, so it cannot serve that. Probe 35440618178 showed the subclass
        # admitting one fresh prompt per step against the plugin's own scheduler.
        # Only at more than one request, because at one there is nothing to serialise.
        from serving_one_in_flight import install as _install_one_in_flight

        if getattr(scheduler, 'scheduler_cls', None) in (None, ''):
            _install_one_in_flight(config)
    return dict(scheduler_requests=_seqs, native_gdn_slots=NATIVE_GDN_SLOTS, verifier_rows=16,
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
    # The 4096/256 equalities were the initial coding profile, not a correctness
    # requirement: they became bounds so longer contexts can be exercised. Everything
    # below them stays exact - temperature, n, penalties, logprobs, stop and the
    # structured-output hooks are what make output bit-comparable against a reference,
    # and every claim downstream of this module depends on that comparability.
    if (type(prompt_tokens) is not int or prompt_tokens < 1
            or prompt_tokens > MAXIMUM_PROMPT_TOKENS
            or parameters.temperature != 0 or parameters.n != 1
            or type(parameters.max_tokens) is not int
            or not 1 <= parameters.max_tokens <= OUTPUT_BUDGET
            or parameters.min_tokens != 0
            or not isinstance(parameters.ignore_eos, bool)
            or parameters.logprobs is not None or parameters.prompt_logprobs is not None
            or parameters.presence_penalty != 0 or parameters.frequency_penalty != 0
            or parameters.repetition_penalty != 1
            or parameters.stop
            or getattr(parameters, 'structured_outputs', None) is not None
            or getattr(parameters, 'logit_bias', None)
            or getattr(parameters, 'allowed_token_ids', None)
            or getattr(parameters, 'bad_words', None)):
        raise ValueError('Fast serving requires a greedy single-sequence request with an '
                         'unpenalised sampler and at most %d output tokens' % OUTPUT_BUDGET)
