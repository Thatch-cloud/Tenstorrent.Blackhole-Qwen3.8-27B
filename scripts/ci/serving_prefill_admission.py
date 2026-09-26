"""C2-any: at most one fresh prompt per prefill step, on the scheduler class the engine actually runs.

Run 36211578069 (G4 ladder, profile c2, nine users submitted at once) killed the EngineCore with
`Fast serving requires one complete fresh prefill: prefill_slot=None new=[three ids] cached=[]`
(serving_lifecycle._execute). The fast prefill path takes one capture and runs one prompt; its GDN
prefill scratch and host RoPE table are single-occupancy. The stock TTScheduler admits every waiting
prompt that fits max-num-batched-tokens into one prefill step (131328 under c2), so three short
prompts that arrive together are batched into one step, and the lifecycle refuses that step.

The fast path's own one-in-flight scheduler never runs. serving_fast_policy.validate_fast_config
sets scheduler_cls to serving_one_in_flight.OneInFlightScheduler. Then the TT platform's
check_and_update_config, in its non-lane branch (plugin bf77cd63 platform.py:1080-1081), overwrites
it with 'vllm_tt_plugin.scheduler.TTScheduler'. The last writer wins, which run 35690327326 had
already shown. Before this module, `exact` was safe only structurally (it admits only
staged-context prompts, and two of those cannot fit one step), and the c2 gate was safe only
because it staggered arrivals by 0.25 s.

THE RULE is lever_n_model_patch.patch_scheduler's, the M2 graft that rewrites TTScheduler.
_schedule_prefill_only in the plugin's scheduler.py (the m3native arm mounts it). It is reused
here, not re-invented:
- if partial prefills are running: allowed = their count, and both waiting queues are hidden, so
  nothing new joins a partial (run 35689293766);
- else, if the lifecycle's prefill gate (sys.modules['_qwen_prefill_gate'].held) names a request:
  allowed = 0, and the queues are hidden (run 35714211185);
- else: allowed = 1, and the queues stay visible, so exactly one fresh prompt is admitted
  (run 35690327326, new=[A, B]).
The waiting loop then sees max(0, min(saved_max - decodes, allowed)). A gate that is absent or
unreadable counts as not held.

Here it is applied at runtime, as a wrapper around the class's own _schedule_prefill_only. It is
not a textual graft, because the serving image ships the plugin's scheduler.py unpatched. The
wrapper writes min(saved_max, allowed + decodes) before delegating. The plugin then subtracts the
decodes (max(0, max_num_running_reqs - len(pure_decodes))) and lands on exactly the graft's value
(serving_one_in_flight's compensation). The queues are hidden and merged back with the file's own
idiom: create_request_queue(self.policy) and prepend_requests, as _schedule_decode_only and the
graft do. test_serving_prefill_admission checks that this matches the graft call for call.

WHERE and WHEN: the lifecycle installs it beside the D2 quarantine consumer
(serving_request_quarantine), on the class the worker's own config names
(serving_request_quarantine.scheduler_class, which on the TT platform is TTScheduler). Both share
the EngineCore process (uniproc executor). The lifecycle does this only under
QWEN_FAST_ANY_REQUEST=1 (the c2 profiles), so `exact` schedules exactly as before. On TT a step is
either all prefill or all decode, and that is unchanged. A prefill step still pauses the running
decodes for that step, as the stock scheduler already did. The difference is that three
simultaneous arrivals now take three one-prompt prefill steps instead of one three-prompt step
that kills the engine. Decodes are never dropped from `running`. When nothing can be admitted
(every seat decoding, or the gate held), the plugin's default mode falls back to a decode-only
step (scheduler.py:140-142).

Two markers show whether it ran (memory: graft-mounted-is-not-graft-executed):
    [PINDIAG] one fresh prefill per step installed on <module.Class>   (once, at install)
    [PINDIAG] one fresh prefill per step live in <Class>               (once, the first prefill
                                                                        step: the platform warmup)
There is also one decision line per distinct (partials, decodes, gate held, allowed, hidden) state.
That keeps the log bounded; the graft logged every call.

S2 W6b, THE DRAM ADMISSION HOLD (s2-design.md section 3.2 item 2; QWEN_FAST_EXTENT_REPLAY=1 only). Beside the
64-row block (3.84 GB per chip) four per-request engines leave about 1.17 GB per chip (d); the fourth arrival's
prefill and engine build see about 1.97 GB (d), or about 1.48 GB (d) while a departed user's quad and pair
traces are still held (W6c releases them at detach). A prefill or build that runs out of DRAM kills the engine
and every live user with it, and a refusal raised after the prefill (the bridge's RequestRefused backstop,
serving_request_factory.dram_backstop) has already spent the prefill. So the worker parks a predicate under
DRAM_KEY at attach (serving_request_factory.register_dram_admission), the prefill gate's sys.modules pattern,
and the wrapper asks it about the request the base scheduler would admit next (head_request) before one fresh
prompt is let in. While it refuses, allowed = 0 and both queues are hidden, exactly as behind a held gate: the
plugin falls back to a decode-only step, the running users keep decoding, and the hold resolves when one of
them finishes and its engine is freed. With no decode left to wait for nothing can free DRAM, so the prompt is
admitted anyway ('lifted') and the backstop decides after its prefill: a hold never livelocks the engine.
The need is ONE set of defaults, here (dram_need): the engine build's resident cost and transient margin, the
prefill transient of a prompt of PREFILL_TRANSIENT_FROM tokens or more, and the coordinator's DRAM reserve,
against the SMALLEST largest free block over the chips (largest_free), never the total free. No predicate, an
unreadable one, or a reading that is unavailable holds nothing: the rule above, call for call.
    [PINDIAG] dram hold prompt=<n> largest_free=<MB> need=<MB> request=<id> decodes=<d>   once per held request
    [PINDIAG] dram hold released prompt=<n> ...                                        once, when it fits
    [PINDIAG] dram hold lifted prompt=<n> ...                                          no decode left to wait for
"""

import importlib
import sys
import types

from serving_request_quarantine import scheduler_class

# serving_lifecycle.PREFILL_GATE_KEY. Duplicated rather than imported, so that the scheduler side
# does not import the lifecycle. The test pins the two equal.
GATE_KEY = '_qwen_prefill_gate'
WRAPPED = '_qwen_one_fresh_prefill'
METHOD = '_schedule_prefill_only'
INSTALLED = '[PINDIAG] one fresh prefill per step installed on '
LIVE = '[PINDIAG] one fresh prefill per step live in '

# S2 W6b, the DRAM admission hold (the module docstring): the key the worker parks its predicate under, and the
# lines the hold logs (DRAM_HOLD is the prefix a gate counts).
DRAM_KEY = '_qwen_dram_admission'
DRAM_HOLD = '[PINDIAG] dram hold prompt='
DRAM_HOLD_LINE = DRAM_HOLD + '{} largest_free={} need={} request={} decodes={}'
DRAM_RELEASED_LINE = '[PINDIAG] dram hold released prompt={} largest_free={} need={} request={}'
DRAM_LIFTED_LINE = ('[PINDIAG] dram hold lifted prompt={} largest_free={} need={} request={}: no decode is left to '
                    'free DRAM, so the prompt is admitted and the bridge backstop decides')
DRAM_UNAVAILABLE_LINE = '[PINDIAG] dram hold unavailable request={}: {} (not held)'
MEGABYTE = 10 ** 6
# The need: one set of defaults, per chip (s2-design.md section 3.2 item 2). M8 and M11 calibrate them.
ENGINE_BUILD_BYTES = 800 * MEGABYTE          # (m) an engine with one 2048 proposal bucket (v26, run 36218104858)
ENGINE_BUILD_MARGIN_BYTES = 200 * MEGABYTE   # (e) the build's transient above what stays resident
PREFILL_TRANSIENT_BYTES = 300 * MEGABYTE     # (e, UNVERIFIED Q8) the prefill of a long prompt beside the block
PREFILL_TRANSIENT_FROM = 2048                # prompts shorter than this: no prefill transient (e)


def _log(message, *values):
    try:
        from loguru import logger
    except ImportError:
        print(message.format(*values), flush=True)
        return
    logger.info(message, *values)


def gate_held(modules=None):
    """The request the lifecycle holds in its prefill phase, read the way the graft reads it.

    Absent, unset or unreadable all mean None ("not held"). That is the behaviour before the gate
    existed. The scheduler must never hard-depend on the baked tree."""
    modules = sys.modules if modules is None else modules
    try:
        return getattr(modules.get(GATE_KEY), 'held', None)
    except BaseException:
        return None


def admission(partials, held):
    """(allowed, hide) for one prefill step: lever_n_model_patch.patch_scheduler's rule.

    allowed is how many requests the step may run (the partials, or one fresh prompt, or none
    while the gate is held). hide says whether both waiting queues are blanked for the step."""
    if type(partials) is not int or partials < 0:
        raise ValueError('Non-negative integer partial prefill count required')
    if partials:
        return partials, True
    if held is not None:
        return 0, True
    return 1, False


def engine_build_peak():
    """An engine build's DRAM peak per chip: what stays resident and the build's own transient."""
    return ENGINE_BUILD_BYTES + ENGINE_BUILD_MARGIN_BYTES


def _reserve(reserve):
    if type(reserve) is not int or reserve < 0:
        raise ValueError('A non-negative integer DRAM reserve in bytes is required, got %r' % (reserve,))
    return reserve


def dram_need(prompt_tokens, reserve):
    """The largest free DRAM block per chip a fresh prompt needs before its prefill is scheduled: the engine
    build's peak, the prefill's transient (none below PREFILL_TRANSIENT_FROM tokens; a length that cannot be
    read counts as long) and the reserve."""
    long_prompt = type(prompt_tokens) is not int or prompt_tokens >= PREFILL_TRANSIENT_FROM
    return engine_build_peak() + (PREFILL_TRANSIENT_BYTES if long_prompt else 0) + _reserve(reserve)


def backstop_need(reserve):
    """What the post-prefill backstop requires (serving_request_factory.dram_backstop): the prefill has run,
    so the build's peak and the reserve."""
    return engine_build_peak() + _reserve(reserve)


def largest_free(pool):
    """(the smallest largest-free-DRAM-block over the chips, None), read through the pool's allocator
    statistics (serving_buffer_pool.ServingBufferPool.dram_statistics), or (None, why) when they cannot be read.
    Never the total free: a buffer needs one contiguous block."""
    statistics = getattr(pool, 'dram_statistics', None)
    if not callable(statistics):
        return None, 'pool without device statistics'
    try:
        report = statistics()
        if isinstance(report, dict):
            return None, str(report.get('unavailable', 'no statistics'))
        return min(int(chip['largest_free']) for chip in report), None
    except Exception as failure:
        return None, '%s: %s' % (type(failure).__name__, str(failure)[:120])


def dram_predicate(pool, reserve):
    """The predicate the worker registers, admits(prompt_tokens) -> (ok, detail): ok is False only when the
    pool's reading exists and its largest free block is below dram_need. An unavailable reading admits, and
    detail['unavailable'] says why (the attach refuses a pool without statistics under the flag, W7)."""
    _reserve(reserve)

    def admits(prompt_tokens):
        need = dram_need(prompt_tokens, reserve)
        largest, reason = largest_free(pool)
        if largest is None:
            return True, dict(largest_free=None, need=need, unavailable=reason)
        return largest >= need, dict(largest_free=largest, need=need)

    return admits


def register_dram_predicate(admits, modules=None):
    """Park `admits` under DRAM_KEY. Returns the callable that removes it again, only while it is still this
    one (the attach's scope calls it before the pool the predicate reads is closed)."""
    if not callable(admits):
        raise ValueError('A callable DRAM admission predicate is required')
    modules = sys.modules if modules is None else modules
    holder = types.ModuleType(DRAM_KEY)
    holder.admits = admits
    modules[DRAM_KEY] = holder

    def unregister():
        if modules.get(DRAM_KEY) is holder:
            del modules[DRAM_KEY]

    return unregister


def dram_admits(modules=None):
    """The registered predicate, or None. Absent or unreadable means no hold: the scheduler never
    hard-depends on the worker."""
    modules = sys.modules if modules is None else modules
    try:
        admits = getattr(modules.get(DRAM_KEY), 'admits', None)
    except BaseException:
        return None
    return admits if callable(admits) else None


def head_request(scheduler):
    """The request the base scheduler's waiting loop takes first - skipped_waiting's head, else waiting's
    (vLLM 0.25.1: `queue = self.skipped_waiting or self.waiting`) - or None."""
    for name in ('skipped_waiting', 'waiting'):
        queue = getattr(scheduler, name, None)
        if not queue:
            continue
        try:
            peek = getattr(queue, 'peek_request', None)
            return peek() if callable(peek) else next(iter(queue))
        except Exception:
            continue
    return None


def prompt_tokens(request):
    """A vLLM Request's prompt length (num_prompt_tokens, else len(prompt_token_ids)), or None."""
    value = getattr(request, 'num_prompt_tokens', None)
    if type(value) is int:
        return value
    try:
        return len(request.prompt_token_ids)
    except Exception:
        return None


def _megabytes(value):
    return '%.1fMB' % (value / MEGABYTE)


def dram_hold(scheduler, decodes, state, log, modules=None):
    """Whether the fresh prompt at the head of the queue waits this step (S2 W6b, the module docstring).
    False when no predicate is registered, nothing waits, the predicate raises or reads nothing, the prompt
    fits, or no decode is left to wait for. `state` holds the held request and the lines already logged, so
    each state logs once."""
    admits = dram_admits(modules)
    if admits is None:
        return False
    request = head_request(scheduler)
    if request is None:
        return False
    request_id = getattr(request, 'request_id', None)
    prompt = prompt_tokens(request)
    seen = state.setdefault('dram_seen', set())
    try:
        ok, detail = admits(prompt)
        largest, need, unavailable = detail['largest_free'], detail['need'], detail.get('unavailable')
    except Exception as failure:
        ok, largest, need, unavailable = True, None, None, '%s: %s' % (type(failure).__name__, str(failure)[:120])
    if largest is None:
        if ('unavailable', request_id) not in seen:
            seen.add(('unavailable', request_id))
            log(DRAM_UNAVAILABLE_LINE, request_id, unavailable or 'no reading')
        return False
    if ok:
        if state.get('dram_held') == request_id:
            state['dram_held'] = None
            log(DRAM_RELEASED_LINE, prompt, _megabytes(largest), _megabytes(need), request_id)
        return False
    if not decodes:
        state['dram_held'] = None
        if ('lifted', request_id) not in seen:
            seen.add(('lifted', request_id))
            log(DRAM_LIFTED_LINE, prompt, _megabytes(largest), _megabytes(need), request_id)
        return False
    if state.get('dram_held') != request_id:
        state['dram_held'] = request_id
        log(DRAM_HOLD_LINE, prompt, _megabytes(largest), _megabytes(need), request_id, decodes)
    return True


def waiting_capacity(saved_max, decodes, allowed):
    """The max_num_running_reqs the base scheduler's waiting loop sees: the graft's
    max(0, min(saved_max - decodes, allowed))."""
    return max(0, min(saved_max - decodes, allowed))


def _module_queue_factory(original):
    """create_request_queue(self.policy), by the name the plugin's scheduler.py binds it to (the
    graft's idiom). Falls back to vLLM's own when the method's module does not bind it."""
    create = getattr(original, '__globals__', {}).get('create_request_queue')
    if create is None:
        create = importlib.import_module('vllm.v1.core.sched.request_queue').create_request_queue
    return lambda scheduler: create(scheduler.policy)


def wrap(original, *, queue_factory, log):
    """The wrapper installed as <class>._schedule_prefill_only around `original`."""
    state = dict(live=False, seen=set(), dram_held=None, dram_seen=set())

    def _schedule_prefill_only(self):
        decodes = sum(1 for request in self.running if not request.is_prefill_chunk)
        partials = len(self.running) - decodes
        held = gate_held()
        allowed, hide = admission(partials, held)
        if allowed and not partials and dram_hold(self, decodes, state, log):
            # S2 W6b: the head prompt does not fit the DRAM left, so it waits as behind a held gate.
            allowed, hide = 0, True
        if not state['live']:
            state['live'] = True
            log(LIVE + '{}', type(self).__name__)
        decision = (partials, decodes, held is not None, allowed, hide)
        if decision not in state['seen']:
            state['seen'].add(decision)
            log('[PINDIAG] one fresh prefill per step: partials={} decodes={} gate_held={} allowed={} hidden={}',
                *decision)
        saved_max = self.max_num_running_reqs
        saved_waiting = self.waiting
        saved_skipped = getattr(self, 'skipped_waiting', None)
        # The plugin writes max(0, value - len(pure_decodes)) before calling the base scheduler, so
        # the value written here carries the decodes it is about to remove.
        self.max_num_running_reqs = min(saved_max, allowed + decodes)
        if hide:
            self.waiting = queue_factory(self)
            if saved_skipped is not None:
                self.skipped_waiting = queue_factory(self)
        try:
            return original(self)
        finally:
            if hide:
                # Anything the base scheduler put back (a preemption) is merged ahead of what
                # was hidden, exactly as _schedule_decode_only and the graft merge it.
                if self.waiting:
                    saved_waiting.prepend_requests(self.waiting)
                if saved_skipped is not None:
                    if self.skipped_waiting:
                        saved_skipped.prepend_requests(self.skipped_waiting)
                    self.skipped_waiting = saved_skipped
                self.waiting = saved_waiting
            self.max_num_running_reqs = saved_max

    setattr(_schedule_prefill_only, WRAPPED, True)
    _schedule_prefill_only.__wrapped__ = original
    return _schedule_prefill_only


def install(config, *, importer=importlib.import_module, log=None, queue_factory=None):
    """Wrap the configured scheduler class's _schedule_prefill_only. Returns its qualified name.

    Idempotent. A subclass of an already-wrapped class inherits the wrapper and is not wrapped
    again. Raises if the class cannot be resolved or has no _schedule_prefill_only (vLLM's stock
    Scheduler has none): the caller then serves as it did before this module existed."""
    log = _log if log is None else log
    cls = scheduler_class(config, importer)
    name = '%s.%s' % (cls.__module__, cls.__qualname__)
    original = getattr(cls, METHOD, None)
    if not callable(original):
        raise ValueError('%s has no %s: not a TT scheduler, so there is no prefill step to cap' % (name, METHOD))
    if getattr(original, WRAPPED, False):
        return name
    if queue_factory is None:
        queue_factory = _module_queue_factory(original)
    setattr(cls, METHOD, wrap(original, queue_factory=queue_factory, log=log))
    log(INSTALLED + '{}', name)
    return name
