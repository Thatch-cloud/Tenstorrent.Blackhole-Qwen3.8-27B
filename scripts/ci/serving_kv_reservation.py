"""KV reservation admission (DF WP-05): admit a request only when its worst-case KV blocks fit what is not already reserved.

WHY. A 262,144-token WINDOW per seat does not fit eight times in the KV cache the cards can hold: eight full windows are 32,768
blocks of 64 tokens, the pool is about 21,760 (the profile's num-gpu-blocks-override), so five full windows are resident at once
and the rest of the arithmetic is the traffic's. The fast path cannot preempt: a request vLLM would preempt for lack of blocks
is a request whose engine, snapshot and traces are live on the cards, and the fast path's lifecycle has no way to park it. So
vLLM must never need to. The only way to keep it from needing to is to never admit a request that could, later in its life,
need a block the pool no longer has.

THE RULE. A request that will hold at most r blocks over its whole life reserves r from the pool when it is admitted and returns
them when it finishes; a fresh prompt is admitted only when the blocks reserved by every running request plus its own r are at
most the pool. Nothing is ever taken back from a running request, so nothing ever preempts.

    r(request) = ceil((prompt_tokens + max_tokens + LOOKAHEAD_TOKENS) / BLOCK_TOKENS) + SPARE_BLOCKS

with LOOKAHEAD_TOKENS = 32 (the drafter's lookahead: vLLM's num_lookahead_tokens is 16 for 15 speculative tokens, and a request
asks the block manager for its tokens plus the lookahead on every step, so its blocks at step t are at most
ceil((computed_t + scheduled_t + 16) / 64), and the last step's computed + scheduled never exceeds prompt + max_tokens + 15;
32 is that bound rounded up to a whole drafter block of 32 rows) and SPARE_BLOCKS = 1 (the block a step's rounding can add beyond
the ceiling: the block table grows before the tokens that need it are accepted). The bound is PROVED against the real vLLM 0.25.1
scheduler on CPU (test_serving_kv_reservation_vllm: random arrivals, lengths, budgets and acceptances of 1 to 16, every step's
preempted request ids empty, every request's allocated blocks within its reservation, and a negative control that preempts when
the rule is off or the margin removed). The pool is vLLM's own count less its null block
(scheduler.kv_cache_manager.block_pool.num_gpu_blocks - NULL_BLOCKS: the null block is block 0 and never allocated).

WHAT IS HELD. The wrapper around the TT scheduler's _schedule_prefill_only (serving_prefill_admission.wrap) asks hold() when a fresh
prompt would be admitted this step (a seat is free, no partial prefill, the gate free): the candidates are the requests the waiting
loop may take (admission_candidates), and the binding one is the one that needs the most blocks (a conservative max, never less
than the one admitted). A hold answers allowed = 0 with both queues hidden, exactly as behind a held gate or a DRAM hold, so the
plugin runs a decode-only pass and the running requests keep decoding; the hold resolves when one of them finishes and returns its
blocks. It is never lifted: with nothing running the pool is empty, so any request the contract admitted (r <= pool, refused at the
edge otherwise, serving_c2_contract) fits. A request whose r exceeds the whole pool is held and logged as too large; the contract's
400 keeps it from ever reaching the scheduler.
    [PINDIAG] kv reservation installed on <class>: pool from block_pool, block=64 lookahead=32 spare=1   once, at install
    [PINDIAG] kv reservation hold request=<id> reserved=<r> running=<sum> pool=<n> decodes=<d>        once per (request, running)
    [PINDIAG] kv reservation released request=<id> reserved=<r> running=<sum> pool=<n>                 once, when it fits
    [PINDIAG] kv reservation fit request=<id> reserved=<r> running=<sum> pool=<n> decodes=<d>          once per request admitted
                                                                                                        without a hold
    [PINDIAG] kv reservation too large request=<id> reserved=<r> pool=<n>: held; the request contract refuses it at the edge
    [PINDIAG] kv reservation carried finished=[<ids>] ...                                              the ids put back past a held pass
Only lines that start with HOLD_PREFIX are holds (what a gate counts). Each state logs once; the state kept to decide that is the
held (request, running) and the last line noted, so it stays bounded whatever the traffic.

OFF (QWEN_FAST_KV_RESERVATION unset or '0'): nothing here is imported by the wrapper and every step is what it was. '1' is on; any
other value is refused (ValueError), never read as off. A profile whose KV pool is smaller than seats x its window must say 1
(serving_c2_contract.request_limits refuses the boot otherwise).

Stdlib only; overlay only (its importers, serving_prefill_admission and serving_c2_contract, are overlay-only).
"""

import os

FLAG = 'QWEN_FAST_KV_RESERVATION'
BLOCK_TOKENS = 64
LOOKAHEAD_TOKENS = 32
SPARE_BLOCKS = 1
NULL_BLOCKS = 1

HOLD_PREFIX = '[PINDIAG] kv reservation hold request='
INSTALLED_LINE = ('[PINDIAG] kv reservation installed on {}: pool from block_pool.num_gpu_blocks less the null block, '
                  'block=%d lookahead=%d spare=%d' % (BLOCK_TOKENS, LOOKAHEAD_TOKENS, SPARE_BLOCKS))
HOLD_LINE = HOLD_PREFIX + '{} reserved={} running={} pool={} decodes={}'
RELEASED_LINE = '[PINDIAG] kv reservation released request={} reserved={} running={} pool={}'
FIT_LINE = '[PINDIAG] kv reservation fit request={} reserved={} running={} pool={} decodes={}'
TOO_LARGE_LINE = ('[PINDIAG] kv reservation too large request={} reserved={} pool={}: held; the request contract refuses it '
                  'at the edge')
UNAVAILABLE_LINE = '[PINDIAG] kv reservation unavailable request={}: {} (the engine stops: it cannot keep the reservation)'
CARRIED_LINE = ('[PINDIAG] kv reservation carried finished={} past the discarded prefill pass into the decode-only step')


def enabled(environ=None):
    """QWEN_FAST_KV_RESERVATION, strictly: unset or '0' is off, '1' is on, anything else is refused."""
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (FLAG, value))
    return value == '1'


def request_blocks(prompt_tokens, max_tokens):
    """r: the blocks a request holds at most over its whole life (the module docstring)."""
    if type(prompt_tokens) is not int or type(max_tokens) is not int or prompt_tokens < 0 or max_tokens < 0:
        raise ValueError('Non-negative integer prompt and max_tokens required, got %r and %r' % (prompt_tokens, max_tokens))
    return -(-(prompt_tokens + max_tokens + LOOKAHEAD_TOKENS) // BLOCK_TOKENS) + SPARE_BLOCKS


def pool_blocks(scheduler):
    """The blocks the pool can hand out: vLLM's block count less its null block. ValueError (named) when the scheduler has no
    such pool: the install checks it once, so a vLLM that moved it fails the install, not a request."""
    total = getattr(getattr(getattr(scheduler, 'kv_cache_manager', None), 'block_pool', None), 'num_gpu_blocks', None)
    if type(total) is not int or total <= NULL_BLOCKS:
        raise ValueError('scheduler.kv_cache_manager.block_pool.num_gpu_blocks is %r: the KV reservation needs vLLM\'s block '
                         'count' % (total,))
    return total - NULL_BLOCKS


def prompt_tokens(request):
    value = getattr(request, 'num_prompt_tokens', None)
    if type(value) is int:
        return value
    try:
        return len(request.prompt_token_ids)
    except Exception:
        return None


def output_budget(request, prompt, window):
    """The request's max_tokens as vLLM carries it (Request.max_tokens), or - when it carries none - the rest of the window."""
    value = getattr(request, 'max_tokens', None)
    if type(value) is int and value >= 0:
        return value
    params = getattr(request, 'sampling_params', None)
    value = getattr(params, 'max_tokens', None)
    if type(value) is int and value >= 0:
        return value
    return max(0, window - prompt)


def request_reservation(request, window):
    """r for a vLLM Request, or None when its prompt length cannot be read (then it counts as the whole window)."""
    prompt = prompt_tokens(request)
    if prompt is None:
        return request_blocks(window, 0)
    return request_blocks(prompt, output_budget(request, prompt, window))


def window_tokens(scheduler):
    """The longest sequence a request may reach: Scheduler.max_model_len (the unknown-budget fallback only)."""
    value = getattr(scheduler, 'max_model_len', None)
    return value if type(value) is int and value > 0 else 0


def running_blocks(scheduler):
    """Sum of r over the requests the scheduler is running (decodes and partial prefills alike): what is reserved."""
    window = window_tokens(scheduler)
    return sum(request_reservation(request, window) for request in getattr(scheduler, 'running', ()))


def binding(candidates, window):
    """(the candidate that needs the most blocks, its r), the first of equals; (None, 0) when there is none."""
    best, need = None, 0
    for request in candidates:
        blocks = request_reservation(request, window)
        if best is None or blocks > need:
            best, need = request, blocks
    return best, need


def _note(state, log, key, template, *values):
    """Log `template` unless the last line noted was for this same key (bounded state, as the DRAM hold's)."""
    if state.get('kv_noted') != key:
        state['kv_noted'] = key
        log(template, *values)


def hold(scheduler, candidates, decodes, state, log):
    """Whether the fresh prompt this step would admit waits for blocks. The wrapper asks only when a seat is free and nothing else
    holds the step. False when nothing waits; ValueError (logged) when the pool cannot be read; True when running + the binding candidate's r exceeds
    the pool (or that r alone does: too large). `state` keeps the held (request, running) and the last line noted."""
    window = window_tokens(scheduler)
    request, need = binding(candidates, window)
    if request is None:
        return False
    request_id = getattr(request, 'request_id', None)
    try:
        pool = pool_blocks(scheduler)
    except ValueError as failure:
        # Fail CLOSED and loud: a pool that cannot be read is a reservation that cannot be kept, and a silent admit is a request
        # vLLM may later have to preempt, which the fast path cannot survive.
        log(UNAVAILABLE_LINE, request_id, str(failure)[:160])
        raise
    running = running_blocks(scheduler)
    held = state.get('kv_held')
    if need > pool:
        _note(state, log, ('large', request_id), TOO_LARGE_LINE, request_id, need, pool)
        return True
    if running + need <= pool:
        if held is not None and held[0] == request_id:
            state['kv_held'] = None
            log(RELEASED_LINE, request_id, need, running, pool)
        else:
            _note(state, log, ('fit', request_id), FIT_LINE, request_id, need, running, pool, decodes)
        return False
    if held != (request_id, running):
        state['kv_held'] = (request_id, running)
        log(HOLD_LINE, request_id, need, running, pool, decodes)
    return True


def install_check(importer=None):
    """At install: the vLLM the rule reads is the one it was proved against - Scheduler.kv_cache_manager.block_pool.num_gpu_blocks
    exist as the pool's count (checked in vLLM's own source: KVCacheManager binds .block_pool, BlockPool binds .num_gpu_blocks) and
    the scheduler carries running, waiting and max_model_len. Returns the problems, one string each ([] when none); the install
    refuses by name when any. Where vLLM is not importable (a CPU test with fakes) nothing is checked: `importer` returns None
    for a module that is not there."""
    import importlib
    import inspect

    importer = importlib.import_module if importer is None else importer
    problems = []
    for module, name, member, needle in (
            ('vllm.v1.core.kv_cache_manager', 'KVCacheManager', '__init__', 'self.block_pool'),
            ('vllm.v1.core.block_pool', 'BlockPool', '__init__', 'self.num_gpu_blocks'),
            ('vllm.v1.core.sched.scheduler', 'Scheduler', '__init__', 'self.kv_cache_manager')):
        try:
            loaded = importer(module)
        except ImportError:
            return []
        cls = getattr(loaded, name, None)
        try:
            source = inspect.getsource(getattr(cls, member))
        except (OSError, TypeError, AttributeError):
            problems.append('%s.%s has no readable %s' % (module, name, member))
            continue
        if needle not in source:
            problems.append('%s.%s.%s no longer binds %s' % (module, name, member, needle))
    return problems
