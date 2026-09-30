"""The lane gate: the scheduler hides the running decodes a round does not serve (QWEN_FAST_LANE, GATE ONLY).

vLLM schedules EVERY running decode in each step, and the worker must commit exactly one output per scheduled request
(serving_vllm_packed.ordered_tickets: the scheduled set must equal the prepared set). A request cannot "sit a round out" by
returning nothing (serving_vllm_state refuses a non-terminal empty output), so a solo round for the fast user must not schedule
the standard users at all. This is that, in the scheduler: the worker plans the coming round when it drafts
(serving_fast_lane.LaneRuntime.publish parks the members under sys.modules[GATE_KEY]) and this wrapper hides every other running
decode from the base scheduler for that one call, then puts them back in their original order.

The mechanism is not new. TTScheduler._schedule_prefill_only already hides the running decodes (`self.running = partial_prefills`,
then `extend`) whenever a prefill runs beside decodes, and the plugin's runner drops a running request that a step does not schedule
from its persistent batch, keeps its state, and re-adds it when it is next scheduled (model_runner._update_states); the
S2 runs exercise both on every staggered arrival. vLLM clears a request's spec_token_ids when it schedules them and
update_draft_token_ids sets only the ids it is given, so a hidden request carries no stale proposals and is drafted again in the
tick before its next round.

Three wrappers on the class the worker's config names (serving_request_quarantine.scheduler_class - on the TT platform,
TTScheduler), each installed once, none touching the pinned plugin's source:
  schedule                   a decode step with no prefill pending: hide the non-members around the base call
  _schedule_decode_only      the decode fallback (a prefill that could not make progress, a forced mode): the same
  _schedule_prefill_only     the seat hold (QWEN_FAST_LANE_RESERVE=1): a standard request never takes the last seat, which is the
                             fast request's - standard users are capped at seats - 1 - and a second fast request counts as
                             standard (the worker's admission downgrades it the same way)
A prefill step is left alone: it schedules no decode, and the plugin's own hide-and-restore covers the decodes.

With no plan published (`members` None) or with the flag off, nothing is hidden: the scheduler is today's. A plan that names no
running request (its members finished or aborted after the draft) hides every decode for at most STALE_LIMIT steps - an empty
bookkeeping step, after which the worker plans the survivors - and then nothing. Every decision that hides something logs one
[LANE-SCHED] line per distinct state.
"""

import importlib
import sys

# serving_fast_lane.GATE_KEY, duplicated so the scheduler side imports no worker module (test_fast_lane pins them equal).
GATE_KEY = '_qwen_lane_gate'
FAST = 'fast'
WRAPPED = '_qwen_lane_gate_wrapped'
HIDING = '_qwen_lane_hiding'
STALE_RUN = '_qwen_lane_stale_run'
STALE_LIMIT = 2          # empty steps in a row a stale plan may cost before the gate opens
INSTALLED = '[PINDIAG] lane gate installed on '
SCHED_MARKER = '[LANE-SCHED]'
LOG_STATES = 256          # distinct states logged; beyond it the log stays quiet (a bounded set, not a per-round line)


def _log(message, *values):
    try:
        from loguru import logger
    except ImportError:
        print(message.format(*values), flush=True)
        return
    logger.info(message, *values)


def gate_state():
    """The shared holder (or None when the worker never engaged the lanes in this process)."""
    return sys.modules.get(GATE_KEY)


def gate_members():
    """The request ids the coming round serves, or None: no plan, nothing hidden."""
    holder = gate_state()
    members = getattr(holder, 'members', None)
    return None if members is None else frozenset(members)


def short(request_id, holder=None):
    """A request as the [LANE-SCHED] line prints it: the lane book's alias (u1, u2, ...) when the worker gave one."""
    alias = (getattr(holder, 'aliases', None) or {}).get(request_id)
    return alias or str(request_id)[-12:]


def is_chunk(request):
    return bool(getattr(request, 'is_prefill_chunk', False))


class Notes:
    """One line per distinct hidden state, from a bounded set."""

    def __init__(self, log):
        self.log, self.seen = log, set()

    def state(self, members, hidden, holder):
        key = (len(members), tuple(sorted(short(request_id, holder) for request_id in hidden)))
        if key in self.seen or len(self.seen) >= LOG_STATES:
            return
        self.seen.add(key)
        self.log('{} round={} members={} hidden={}', SCHED_MARKER, getattr(holder, 'round', 0), len(members),
                 ','.join(key[1]))

    def once(self, key, template, *values):
        if key in self.seen or len(self.seen) >= LOG_STATES:
            return
        self.seen.add(key)
        self.log(template, *values)


def prefill_pending(scheduler):
    """Whether this schedule() call will run a prefill step (TTScheduler._has_pending_prefill: waiting or skipped requests, or
    a partial prefill). Without the method: the same three questions asked directly."""
    ask = getattr(scheduler, '_has_pending_prefill', None)
    if callable(ask):
        return bool(ask())
    return (bool(getattr(scheduler, 'waiting', None)) or bool(getattr(scheduler, 'skipped_waiting', None))
            or any(is_chunk(request) for request in getattr(scheduler, 'running', ())))


def forced_prefill(scheduler):
    """A forced PREFILL_ONLY mode (the lane coordinator's): no decode step, so nothing to hide."""
    mode = getattr(scheduler, '_forced_mode', None)
    return getattr(mode, 'name', None) == 'PREFILL_ONLY'


def hidden_call(scheduler, members, call, notes):
    """Run `call` with every running non-chunk decode outside `members` removed from scheduler.running, then restore them in their
    original order (dropping what the call itself removed: a finished or preempted request stays out). Nothing is hidden when
    the plan names no running decode - a stale plan must never starve the engine."""
    running = list(scheduler.running)
    decodes = [request for request in running if not is_chunk(request)]
    serving = [request for request in decodes if request.request_id in members]
    hidden = [request for request in decodes if request.request_id not in members]
    if not hidden:
        setattr(scheduler, STALE_RUN, 0)
        return call()
    if not serving:
        # The plan's members are all gone (the fast user finished or aborted after the draft) and every decode still running was
        # NOT drafted for this round, so scheduling them would hand the worker requests with no ticket. Hide them all for this
        # call: the step is empty (a bookkeeping step, the worker passes it through), the finish is reported, the lifecycle
        # releases the member (which clears the plan) and the drafts after the step plan the survivors. Bounded: after
        # STALE_LIMIT empty steps in a row the gate opens (today's scheduler) rather than starve the engine.
        run = getattr(scheduler, STALE_RUN, 0)
        if run >= STALE_LIMIT:
            notes.once(('stale-open', len(members)), '{} gate names no running decode after {} empty step(s): nothing hidden '
                       'members={} running={}', SCHED_MARKER, run, len(members), len(decodes))
            return call()
        setattr(scheduler, STALE_RUN, run + 1)
        notes.once(('stale', tuple(sorted(short(request.request_id) for request in decodes))),
                   '{} gate names no running decode: an empty step, the worker plans again members={} running={}',
                   SCHED_MARKER, len(members), len(decodes))
        hidden = decodes
    else:
        setattr(scheduler, STALE_RUN, 0)
    hidden_ids = {id(request) for request in hidden}
    scheduler.running = [request for request in running if id(request) not in hidden_ids]
    setattr(scheduler, HIDING, True)
    notes.state(members, [request.request_id for request in hidden], gate_state())
    try:
        return call()
    finally:
        setattr(scheduler, HIDING, False)
        present = {id(request) for request in scheduler.running}
        scheduler.running = [request for request in running if id(request) in present or id(request) in hidden_ids]


def wrap_schedule(original, notes):
    def schedule(self, *args, **kwargs):
        members = gate_members()
        if (members is None or getattr(self, HIDING, False) or forced_prefill(self) or prefill_pending(self)):
            return original(self, *args, **kwargs)
        return hidden_call(self, members, lambda: original(self, *args, **kwargs), notes)

    setattr(schedule, WRAPPED, True)
    schedule.__wrapped__ = original
    return schedule


def wrap_decode_only(original, notes):
    def _schedule_decode_only(self, *args, **kwargs):
        members = gate_members()
        if members is None or getattr(self, HIDING, False):
            return original(self, *args, **kwargs)
        return hidden_call(self, members, lambda: original(self, *args, **kwargs), notes)

    setattr(_schedule_decode_only, WRAPPED, True)
    _schedule_decode_only.__wrapped__ = original
    return _schedule_decode_only


def seat_hold(scheduler, holder):
    """Whether standard requests must wait for a seat: the reserve is on and standard users already fill every seat but the
    fast request's. A partial prefill counts as its lane (a fast one in flight holds the fast seat, any other a standard one)."""
    if not getattr(holder, 'reserve', False) or not getattr(holder, 'seats', 0):
        return False
    running = list(getattr(scheduler, 'running', ()))
    fast_id = getattr(holder, 'fast_id', None)
    import serving_fast_lane

    lane_of = serving_fast_lane.lane_of

    standard = 0
    for request in running:
        fast = (fast_id is not None and request.request_id == fast_id) or (
            fast_id is None and is_chunk(request) and lane_of(request) == FAST)
        if not fast:
            standard += 1
    return standard >= holder.seats - 1


def fast_visible(request, holder, running):
    """A waiting request the seat hold lets through: a fast request while the fast seat is free (nobody holds it and no fast
    prefill is in flight). Everything else - a standard request, a second fast one - waits."""
    import serving_fast_lane

    lane_of = serving_fast_lane.lane_of

    if lane_of(request) != FAST or getattr(holder, 'fast_id', None) is not None:
        return False
    return not any(is_chunk(other) and lane_of(other) == FAST for other in running)


def wrap_admission(original, *, queue_factory, notes):
    def _schedule_prefill_only(self, *args, **kwargs):
        holder = gate_state()
        if holder is None or not seat_hold(self, holder):
            return original(self, *args, **kwargs)
        running = list(self.running)
        saved_waiting = self.waiting
        saved_skipped = getattr(self, 'skipped_waiting', None)
        visible, held = queue_factory(self), queue_factory(self)
        for request in list(saved_waiting):
            (visible if fast_visible(request, holder, running) else held).add_request(request)
        if not held:
            return original(self, *args, **kwargs)
        notes.once(('seat-hold', len(held), len(visible)),
                   '{} seat hold: standard users fill {} of {} seats, {} waiting request(s) held, {} visible',
                   SCHED_MARKER, holder.seats - 1, holder.seats, len(held), len(visible))
        self.waiting = visible
        if saved_skipped is not None:
            self.skipped_waiting = queue_factory(self)
        try:
            return original(self, *args, **kwargs)
        finally:
            # Anything the base scheduler put back (a preemption) or left unadmitted goes ahead of what was held.
            if self.waiting:
                held.prepend_requests(self.waiting)
            self.waiting = held
            if saved_skipped is not None:
                if self.skipped_waiting:
                    saved_skipped.prepend_requests(self.skipped_waiting)
                self.skipped_waiting = saved_skipped

    setattr(_schedule_prefill_only, WRAPPED, True)
    _schedule_prefill_only.__wrapped__ = original
    return _schedule_prefill_only


def install(config, *, importer=importlib.import_module, log=None, queue_factory=None):
    """Wrap the configured scheduler class once. Returns its qualified name. Idempotent (a class whose method already carries
    WRAPPED is left alone). Raises when the class has no schedule() - not a scheduler this can gate."""
    import serving_request_quarantine

    log = _log if log is None else log
    cls = serving_request_quarantine.scheduler_class(config, importer)
    name = '%s.%s' % (cls.__module__, cls.__qualname__)
    notes = Notes(log)
    original = getattr(cls, 'schedule', None)
    if not callable(original):
        raise ValueError('%s has no schedule(): there is no decode step to gate' % name)
    if getattr(original, WRAPPED, False):
        return name
    cls.schedule = wrap_schedule(original, notes)
    decode_only = getattr(cls, '_schedule_decode_only', None)
    if callable(decode_only) and not getattr(decode_only, WRAPPED, False):
        cls._schedule_decode_only = wrap_decode_only(decode_only, notes)
    prefill_only = getattr(cls, '_schedule_prefill_only', None)
    if callable(prefill_only) and not getattr(prefill_only, WRAPPED, False):
        if queue_factory is None:
            import serving_prefill_admission

            queue_factory = serving_prefill_admission._module_queue_factory(prefill_only)
        cls._schedule_prefill_only = wrap_admission(prefill_only, queue_factory=queue_factory, notes=notes)
    log(INSTALLED + '{}', name)
    return name
