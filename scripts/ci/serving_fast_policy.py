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

import os

# The per-request output budget. Every request's verifier captures are allocated for it up
# front, one set per 256-position mask family it can touch (verifier_engine.capture_bucket_rows),
# so it costs device memory per admitted request rather than being a free policy number.
# 256 is the measured C2 contract; a serving image may raise it (QWEN_FAST_OUTPUT_BUDGET, a
# whole number of 256-token families up to 16384) and pay for it with a shorter context.
PACKED_FAMILY_TOKENS = 256


def _output_budget(value):
    if value in (None, ''):
        return PACKED_FAMILY_TOKENS
    try:
        budget = int(value)
    except ValueError:
        budget = 0
    if budget < PACKED_FAMILY_TOKENS or budget > 16384 or budget % PACKED_FAMILY_TOKENS:
        raise ValueError('QWEN_FAST_OUTPUT_BUDGET must be a multiple of %d from %d to 16384, got %r'
                         % (PACKED_FAMILY_TOKENS, PACKED_FAMILY_TOKENS, value))
    return budget


OUTPUT_BUDGET = _output_budget(os.environ.get('QWEN_FAST_OUTPUT_BUDGET'))

# C2-any (plan stage S1; the c2 serving profile sets it, `exact` and every gate leave it
# unset). With it on, the fast path takes any request the edge contract admits instead of
# only the frozen benchmark shape:
# - beside the four-user block, whose engines capture only widths 1, 2 and 4, each engine is
#   built without replay attention and without the per-request T16 gate, which admits only
#   position == the frozen context with <= 256 left (serving_request_factory.from_prefill);
# - the source qualification that gate carried runs once, at attach, instead
#   (serving_request_factory.attach_source_check, called by serving_runtime);
# - each request's budget is its own max_tokens, not the server ceiling OUTPUT_BUDGET;
# - a host-side refusal of one request's own terms (its sampling contract, budget or page
#   table) ends that one request (FINISHED_ABORTED) instead of the engine - every other refusal
#   stays fatal - and ignore_eos and a first token that exhausts max_tokens are served
#   (serving_lifecycle, serving_request_quarantine).
# Unset or '0', every one of those paths is exactly what it was.
ANY_REQUEST_FLAG = 'QWEN_FAST_ANY_REQUEST'


def any_request_enabled(environ=None):
    """Whether QWEN_FAST_ANY_REQUEST=1. Read per call, so a test or a profile can set it;
    anything but unset, '0' or '1' is a configuration error, never a silent off."""
    environ = os.environ if environ is None else environ
    value = environ.get(ANY_REQUEST_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (ANY_REQUEST_FLAG, value))
    return value == '1'


# Sticky sessions, phase 1 (default off; the c2-packed-prefix profiles set it beside
# QWEN_PREFIX_REUSE=1). A turn that extends an earlier one resumes exactly from the last
# 2048-token boundary at or below its prompt minus 2048 whose GDN checkpoint and KV prefix
# the prefix-reuse grafts kept (qwen_prefix_scheduler_patch trims vLLM's hit to it and commits
# a grant; qwen_prefix_model_patch restores it); the fast path then prefills only the tail and
# builds its engine as it always does. On, and only together with QWEN_PREFIX_REUSE=1:
# - validate_fast_config accepts vLLM's prefix cache (sha256 block hashes only);
# - serving_lifecycle admits a new request at num_computed_tokens == R > 0 when the shared
#   registry holds this step's committed grant at Q == R and R is a boundary at or below the
#   prompt minus 2048 (serving_lifecycle.FastServingLifecycle._granted_resume); the prefill
#   capture then counts from R and records the prefix-reuse route's GDN slot;
# - the scheduler graft accepts DFlash's lookahead and plans the checkpoint vLLM's EAGLE-style
#   last-block drop makes reachable (floor2048(P) - 2048);
# - the packed K/V guard maps a row past an engine's bound blocks to a per-request sentinel
#   page instead of the shared page-table pad (serving_packed_step.kv_guard).
# Unset or '0', every one of those paths is exactly what it was.
STICKY_SESSIONS_FLAG = 'QWEN_FAST_STICKY_SESSIONS'
# The prefix-reuse switch the sticky flag needs beside it (serving_c2_contract.PREFIX_SWITCH).
PREFIX_REUSE_FLAG = 'QWEN_PREFIX_REUSE'
# The block-hash algorithm a prefix hit's exactness leans on (serving_c2_contract.PREFIX_HASH_ALGO).
PREFIX_HASH_ALGO = 'sha256'


def sticky_sessions_enabled(environ=None):
    """Whether QWEN_FAST_STICKY_SESSIONS=1. Read per call, like any_request_enabled; anything
    but unset, '0' or '1' is a configuration error, never a silent off."""
    environ = os.environ if environ is None else environ
    value = environ.get(STICKY_SESSIONS_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (STICKY_SESSIONS_FLAG, value))
    return value == '1'


def prefix_cache_admitted(cache, environ=None):
    """Whether the fast path may run with vLLM's prefix cache on: sticky sessions and prefix
    reuse both on, and a sha256 block-hash chain. Only consulted when the cache is on."""
    environ = os.environ if environ is None else environ
    return (sticky_sessions_enabled(environ) and environ.get(PREFIX_REUSE_FLAG) == '1'
            and getattr(cache, 'prefix_caching_hash_algo', None) == PREFIX_HASH_ALGO)

# Engine reuse, parked per-slot engines (default off; the gate-only parked twins of the four-card profiles set it). One engine per pool slot is
# built at attach on a synthetic request and REBOUND to each request instead of built per request (serving_parked_engines). On, and only on the
# four-card shape it was ported to:
# - QWEN_FAST_TP=4 (the pair's Stage E is the TP2 line's own, never merged here), C2-any and the extent readers
#   (QWEN_FAST_ANY_REQUEST=1, QWEN_FAST_EXTENT_REPLAY=1);
# - the packed step beside the sequential captures 1, 2 and 4 (QWEN_FAST_PACKED_STEP=1; capture_rows 4, which the attach itself checks), with
#   four scheduler requests and one block, or eight and QWEN_FAST_M3_BLOCKS=2;
# - captured proposals (no QWEN_FAST_EAGER_PROPOSAL) and the shared collectives (QWEN_FAST_SHARED_CCL=1);
# - the verify trace reading every carry in place (QWEN_FAST_VERIFY_T1=1; the attach checks the block's own carries_in_place too), because a
#   padded round would otherwise overwrite an idle parked slot's carry;
# - QWEN_FAST_M3_REQUEST_WARM=1: the S8 hang class (a program first compiled after a block's first replay) is covered by that warm, not by the
#   attach ordering alone;
# - no lane or solo lane (they drive their own slot orders), no prompt-lookup draft (a rebind attaches none), no gather experiment.
# QWEN_FAST_PARKED_DRAFTS=1 ("2c": the pair and quad traces bound to the slots) additionally needs the engines, the quad over both blocks
# (QWEN_FAST_QUAD_DRAFT=1, QWEN_FAST_QUAD_DRAFT_BLOCKS=2), the live K/V banks (QWEN_FAST_FUSED_COMMIT_LIVE_BANKS=1), and no wide-drafter candidate
# (QWEN_FAST_TP4_DRAFT_WIDE: a persistent L1 buffer allocated after a drafter capture could land on freed shard addresses a bound trace
# overwrites on every replay) until an audited arm passes.
# The names below are the whole QWEN_FAST_PARKED_ family: any other is refused, so a typo never reads as "off". Unset or '0', every path is
# exactly what it was.
PARKED_ENGINES_FLAG = 'QWEN_FAST_PARKED_ENGINES'
PARKED_DRAFTS_FLAG = 'QWEN_FAST_PARKED_DRAFTS'
PARKED_PREFIX = 'QWEN_FAST_PARKED_'
PARKED_NAMES = ('QWEN_FAST_PARKED_ENGINES', 'QWEN_FAST_PARKED_DRAFTS', 'QWEN_FAST_PARKED_PROJECT_ROWS', 'QWEN_FAST_PARKED_AUDIT',
                'QWEN_FAST_PARKED_NEGATIVE', 'QWEN_FAST_PARKED_FAULT')
# Gate-only knobs (serving_c2_contract.parked_problems refuses them outside a gate profile).
PARKED_NEGATIVE_FLAG = 'QWEN_FAST_PARKED_NEGATIVE'
PARKED_NEGATIVES = ('carry', 'drafter', 'pages', 'widths')
PARKED_FAULT_FLAG = 'QWEN_FAST_PARKED_FAULT'
PARKED_FAULTS = ('park', 'rebind')
GATE_DRAM_BALLAST_FLAG = 'QWEN_FAST_GATE_DRAM_BALLAST'


def parked_engines_enabled(environ=None):
    """Whether QWEN_FAST_PARKED_ENGINES=1. Read per call; anything but unset, '0' or '1' is a configuration error, never a silent off."""
    environ = os.environ if environ is None else environ
    value = environ.get(PARKED_ENGINES_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (PARKED_ENGINES_FLAG, value))
    return value == '1'


def parked_engine_problems(environ, users):
    """Every way the parked engines' environment is wrong, [] when none. Unknown QWEN_FAST_PARKED_ names are refused whether the flag is on or
    not; every other check runs only with QWEN_FAST_PARKED_ENGINES=1 (the DRAFTS, AUDIT, NEGATIVE and FAULT values are checked whenever set)."""
    problems = ['%s is not a parked-engines setting (the names are %s)' % (name, ', '.join(PARKED_NAMES))
                for name in sorted(environ) if name.startswith(PARKED_PREFIX) and name not in PARKED_NAMES]
    for name in (PARKED_DRAFTS_FLAG, 'QWEN_FAST_PARKED_AUDIT'):
        if environ.get(name, '0') not in ('0', '1'):
            problems.append('%s must be 0 or 1, got %r' % (name, environ.get(name)))
    try:
        enabled = parked_engines_enabled(environ)
    except ValueError as error:
        return problems + [str(error)]
    if not enabled:
        if environ.get(PARKED_DRAFTS_FLAG, '0') == '1':
            problems.append('%s=1 needs %s=1' % (PARKED_DRAFTS_FLAG, PARKED_ENGINES_FLAG))
        return problems
    for name, wanted, why in (
            (ANY_REQUEST_FLAG, '1', 'the parked engines serve C2-any requests'),
            ('QWEN_FAST_EXTENT_REPLAY', '1', 'the packed blocks beside the parked engines are the S2 extent blocks'),
            ('QWEN_FAST_PACKED_STEP', '1', 'the packed step is the four-user block'),
            ('QWEN_FAST_SHARED_CCL', '1', 'a parked device keeps its collectives for the process'),
            ('QWEN_FAST_TP', '4', 'this port serves the four-card mesh; the pair\'s Stage E is the other line\'s'),
            ('QWEN_FAST_M3_REQUEST_WARM', '1', 'the S8 hang class (an engine program first compiled after a block\'s first replay) is covered '
                                                 'by the request-width warm, not by the attach ordering alone'),
            ('QWEN_FAST_VERIFY_T1', '1', 'a padded round writes the idle slot carry unless the trace reads every '
                                         'carry in place, and parked slots are always lent')):
        if environ.get(name) != wanted:
            problems.append('%s=1 needs %s=%s (%s), not %s' % (PARKED_ENGINES_FLAG, name, wanted, why, environ.get(name, '(unset)')))
    blocks = environ.get('QWEN_FAST_M3_BLOCKS', '')
    if users == 4 and blocks not in ('', '1'):
        problems.append('%s=1 at four scheduler requests needs one block (QWEN_FAST_M3_BLOCKS unset or 1), not %r' % (PARKED_ENGINES_FLAG, blocks))
    elif users == 8 and blocks != '2':
        problems.append('%s=1 at eight scheduler requests needs QWEN_FAST_M3_BLOCKS=2, not %r' % (PARKED_ENGINES_FLAG, blocks or '(unset)'))
    elif users not in (4, 8):
        problems.append('%s=1 needs four or eight scheduler requests (one or two packed blocks), not %r' % (PARKED_ENGINES_FLAG, users))
    if environ.get('QWEN_FAST_EAGER_PROPOSAL') == '1':
        problems.append('%s=1 needs captured proposals: QWEN_FAST_EAGER_PROPOSAL=1 has no trace to keep' % PARKED_ENGINES_FLAG)
    for name in ('QWEN_FAST_LANE', 'QWEN_FAST_SOLO_LANE'):
        if environ.get(name, '0') not in ('', '0'):
            problems.append('%s=1 does not run beside %s: the lanes drive their own slot orders' % (PARKED_ENGINES_FLAG, name))
    if environ.get('QWEN_FAST_LOOKUP_DRAFT', '').strip().lower() not in ('', '0', 'off'):
        problems.append('%s=1 does not run beside QWEN_FAST_LOOKUP_DRAFT: a rebind attaches no prompt lookup' % PARKED_ENGINES_FLAG)
    for name in ('QWEN_GDN_GROUPED_GATHER_ABBA', 'QWEN_GDN_GATE_EXP_ABBA'):
        if environ.get(name, '0') != '0':
            problems.append('%s=1 does not run beside a gather experiment (%s=%s)' % (PARKED_ENGINES_FLAG, name, environ.get(name)))
    rows = environ.get('QWEN_FAST_PARKED_PROJECT_ROWS', '256')
    if not (rows.isdigit() and rows == str(int(rows)) and (int(rows) == 0 or (int(rows) % 32 == 0 and int(rows) <= 2048))):
        problems.append('QWEN_FAST_PARKED_PROJECT_ROWS must be 0 or a multiple of 32 up to 2048, got %r' % (rows,))
    negative = environ.get(PARKED_NEGATIVE_FLAG)
    if negative is not None and negative != '' and negative not in PARKED_NEGATIVES:
        problems.append('%s must be one of %s, got %r' % (PARKED_NEGATIVE_FLAG, ', '.join(PARKED_NEGATIVES), negative))
    fault = environ.get(PARKED_FAULT_FLAG)
    if fault is not None and fault != '' and fault not in PARKED_FAULTS:
        problems.append('%s must be one of %s, got %r' % (PARKED_FAULT_FLAG, ', '.join(PARKED_FAULTS), fault))
    if environ.get(PARKED_DRAFTS_FLAG, '0') == '1':
        for name, wanted, why in (('QWEN_FAST_QUAD_DRAFT', '1', 'the bound traces are the pair and block-quad ones'),
                                  ('QWEN_FAST_QUAD_DRAFT_BLOCKS', '2', 'the book holds the eight-seat per-block quads'),
                                  ('QWEN_FAST_FUSED_COMMIT_LIVE_BANKS', '1', 'a bound trace reads the pool slots\' active K/V banks, which '
                                                                              'never move; its own placeholders would be per request')):
            if environ.get(name) != wanted:
                problems.append('%s=1 needs %s=%s (%s), not %s' % (PARKED_DRAFTS_FLAG, name, wanted, why, environ.get(name, '(unset)')))
        if environ.get('QWEN_FAST_TP4_DRAFT_WIDE', '0') not in ('', '0'):
            problems.append('%s=1 does not run beside QWEN_FAST_TP4_DRAFT_WIDE: an L1 buffer allocated after a drafter capture could land on '
                            'freed shard addresses a bound trace overwrites on every replay (until an audited arm passes)' % PARKED_DRAFTS_FLAG)
    return problems


NATIVE_GDN_SLOTS = 8
# The proposal block is 32 rows and the verify block is 32 or 64. capture_widths caps a
# per-request bucket at 32 and dflash_device accepts block_rows in (8, 16, 32), so the
# 32-row packed block divides its rows among its users: two T16 users (M1), or four T8
# users (M2) - and dropping to T8 to fit four cuts each user from 15 proposals to 7,
# which is a direct cut to committed tokens per cycle. Four T16 users need the 64-row
# verify block (M3, docs/packed-device-step-plan-2026-09-20.md section 5); the draft
# side still proposes in 32-row passes, two T16 users per pass.
PACKED_BLOCK_ROWS = 32
PACKED_BLOCK_WIDTHS = (32, 64)


def packed_geometry(users, block_rows=PACKED_BLOCK_ROWS):
    """Rows per user in the shared verify block, and the proposals that buys."""
    if type(block_rows) is not int or block_rows not in PACKED_BLOCK_WIDTHS:
        raise ValueError('Packed verify block of %s rows required' % ' or '.join(map(str, PACKED_BLOCK_WIDTHS)))
    if type(users) is not int or not 1 <= users <= NATIVE_GDN_SLOTS or block_rows % users:
        raise ValueError('Packed users must divide the %d-row block within the %d native GDN slots'
                         % (block_rows, NATIVE_GDN_SLOTS))
    rows = block_rows // users
    if rows not in (8, 16, 32):
        raise ValueError('Each packed user needs a supported T8/T16/T32 share of the block')
    return dict(users=users, block_rows=rows, verifier_rows=block_rows,
                proposals_per_user=rows - 1)
MINIMUM_MODEL_LEN = 4096 + OUTPUT_BUDGET
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
    # The prefix cache stays refused unless sticky sessions admit it (prefix_cache_admitted,
    # consulted only when the cache is on, so with it off this clause is what it always was).
    if (cache.block_size != 64
            or (cache.enable_prefix_caching is not False
                and not (cache.enable_prefix_caching is True and prefix_cache_admitted(cache)))
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
    # Engine reuse: only names and values the parked engines define, and only on the shape they were built for. Nothing is read from the
    # environment beyond the QWEN_FAST_PARKED_ names unless the flag is on.
    parked = parked_engine_problems(os.environ, _seqs)
    if parked:
        raise ValueError('Parked engines (QWEN_FAST_PARKED_ENGINES): %s' % '; '.join(parked))
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
