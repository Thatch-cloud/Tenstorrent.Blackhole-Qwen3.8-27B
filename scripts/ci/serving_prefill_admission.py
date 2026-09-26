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
"""

import importlib
import sys

from serving_request_quarantine import scheduler_class

# serving_lifecycle.PREFILL_GATE_KEY. Duplicated rather than imported, so that the scheduler side
# does not import the lifecycle. The test pins the two equal.
GATE_KEY = '_qwen_prefill_gate'
WRAPPED = '_qwen_one_fresh_prefill'
METHOD = '_schedule_prefill_only'
INSTALLED = '[PINDIAG] one fresh prefill per step installed on '
LIVE = '[PINDIAG] one fresh prefill per step live in '


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
    state = dict(live=False, seen=set())

    def _schedule_prefill_only(self):
        decodes = sum(1 for request in self.running if not request.is_prefill_chunk)
        partials = len(self.running) - decodes
        held = gate_held()
        allowed, hide = admission(partials, held)
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
