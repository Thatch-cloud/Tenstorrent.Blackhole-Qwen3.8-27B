"""tp4/sched-2 (docs/tp4-sched2.md): NP3, admission pipelining and the memory-ledger trim. Three levers, each a switch that defaults off, each in this one module so that
the files the other branches change keep their hunks small. Stdlib only. With every switch unset nothing here is imported by the serving path and every call is what it was.

WHERE THE ADMISSION TIME GOES AT SHIP 5 (measured: docs/tp4-sched2.md section 2, from the 93 admissions of the ship-5 timed arm). A returning turn spends, between one arrival's
prefill and the next arrival's, about 400 ms (p50) that is not device-bound prefill: the scheduler's DRAM hold reads the allocator (84 ms), the prefill's device work ends
(13 ms), the slot adoption (65 ms), the placement and window join (11 ms), and the rebind itself (181 ms); and the token the prefill already sampled waits for ALL of that,
because the sampler returns the seed only after the engine for the request exists. A cold slot costs far more: the first rebind of a slot also pays the memory ledger's
bracket (about 170 to 310 ms) and the first adoption of a native slot the conv-slice readback (about 85 to 160 ms), which an eight-user burst after a restart pays eight times.

  QWEN_FAST_ADMISSION_SEED_FIRST=1    PIPELINING. The seed token leaves the engine BEFORE the request's engine is bound: the lifecycle returns the sampler's output at once and
                                      builds the bridge (the slot adoption, the rebind, the hook) at the start of the next schedule() call, ahead of everything that call decides, so
                                      the device work, the scheduler's view and the pacing are the ones of the eager order, and only the moment the first token reaches the output
                                      queue moves: earlier by the whole bind. Engages only when the request will rebind a parked engine and its cold fallback would clear the DRAM
                                      too (a refusal cannot be raised after a token has gone); anything else binds eagerly, as it always did.
  QWEN_FAST_ADMISSION_HOLD_REUSE=1    The scheduler's DRAM hold reuses a recent allocator reading (84 ms, device idle) when it clears the hold's terms after a drift allowance per
                                      admission since the reading (_MAX_AGE_MS, _MAX_REUSES, _DRIFT_MB); else it reads, and every reading is kept for the next.
  QWEN_FAST_ADMISSION_LEDGER_TRIM=1   The memory ledger stops at the end of the attach: no reading at any admission, not even the first of each kind or the first rebind of a slot.

All three are gate arms until the owner decides (serving_c2_contract.sched2_problems). Their log lines (the report, scripts/ci/sched2_report.py, reads them):

    [PINDIAG] admission pipeline installed: seed_first=<0|1> hold_reuse=<0|1> ledger_trim=<0|1> ...
    [PINDIAG] admission hold reading reused prompt=<n> age_ms=<ms> reuses=<k> free=<bytes> largest_free=<bytes>       one per reused hold reading
    [PINDIAG] admission seed first deferred request=<id> slot=<k>                                                     the seed went out before the bind
    [PINDIAG] admission seed first eager request=<id> reason=<why>                                                    the bind stayed in the sampler
    [PINDIAG] admission seed first built request=<id> ms=<bind ms> after_seed_ms=<ms since the seed was returned>      the bind, when it ran (ms is what the first token did not wait for)
    [PINDIAG] admission ledger trim: the memory ledger ended after the attach
"""

import functools
import os
import sys
import time
import types

SEED_FLAG = 'QWEN_FAST_ADMISSION_SEED_FIRST'
HOLD_FLAG = 'QWEN_FAST_ADMISSION_HOLD_REUSE'
LEDGER_FLAG = 'QWEN_FAST_ADMISSION_LEDGER_TRIM'
HOLD_AGE_FLAG = 'QWEN_FAST_ADMISSION_HOLD_REUSE_MAX_AGE_MS'
HOLD_REUSES_FLAG = 'QWEN_FAST_ADMISSION_HOLD_REUSE_MAX_REUSES'
HOLD_DRIFT_FLAG = 'QWEN_FAST_ADMISSION_HOLD_REUSE_DRIFT_MB'
SWITCHES = (SEED_FLAG, HOLD_FLAG, LEDGER_FLAG)
HOLD_KNOBS = (HOLD_AGE_FLAG, HOLD_REUSES_FLAG, HOLD_DRIFT_FLAG)
NAMES = SWITCHES + HOLD_KNOBS

# The hold's knobs: (default, low, high). The age bounds how long a reading may stand in for the allocator; the reuses bound how many holds one reading may serve; the drift is what
# one admission between the reading and the hold may take from the free total and from the largest free block (the discount is (reuses + 1) times it, and the largest block is also
# halved, as serving_prefill_admission.reusable_reading does, since a prefill can split it). The measured ratchet of the ship-5 timed arm: a first-time trace capture took 108 to 212 MB
# in one admission of the cold burst, and a warm admission 0.1 MB; 256 MB covers the worst one seen.
HOLD_DEFAULTS = {HOLD_AGE_FLAG: (10000, 100, 60000), HOLD_REUSES_FLAG: (3, 1, 16), HOLD_DRIFT_FLAG: (256, 16, 1024)}
MEGABYTE = 10 ** 6

INSTALLED_LINE = '[PINDIAG] admission pipeline installed: seed_first={} hold_reuse={} ledger_trim={} max_age_ms={} max_reuses={} drift_mb={}'
HOLD_REUSED_LINE = '[PINDIAG] admission hold reading reused prompt={} age_ms={:.0f} reuses={} free={} largest_free={}'
SEED_DEFERRED_LINE = '[PINDIAG] admission seed first deferred request={} slot={}'
SEED_EAGER_LINE = '[PINDIAG] admission seed first eager request={} reason={}'
SEED_BUILT_LINE = '[PINDIAG] admission seed first built request={} ms={:.1f} after_seed_ms={:.1f}'
LEDGER_TRIM_LINE = '[PINDIAG] admission ledger trim: the memory ledger ended after the attach'
HOLDER_KEY = '_qwen_seed_first'
SCHEDULE_WRAPPED = '_qwen_seed_first_schedule'


def _log(message, *values):
    try:
        from loguru import logger
    except ImportError:
        print(message.format(*values), flush=True)
        return
    logger.info(message, *values)


# -- the switches ---------------------------------------------------------------------------------------------------------------------------------

def _switch(name, environ=None):
    return (os.environ if environ is None else environ).get(name) == '1'


def seed_first_enabled(environ=None):
    """QWEN_FAST_ADMISSION_SEED_FIRST=1. Lenient (a malformed value is off here and named by problems, which the profile contract and the attach run): this is read inside a lifecycle."""
    return _switch(SEED_FLAG, environ)


def hold_reuse_enabled(environ=None):
    return _switch(HOLD_FLAG, environ)


def ledger_trim_enabled(environ=None):
    return _switch(LEDGER_FLAG, environ)


def any_enabled(environ=None):
    return any(_switch(name, environ) for name in SWITCHES)


def _number(name, environ):
    """The knob's value, or (None, why) when it is set to something that is not a whole number in its range."""
    default, low, high = HOLD_DEFAULTS[name]
    raw = environ.get(name)
    if raw is None:
        return default, None
    if not raw.isascii() or not raw.isdigit() or not low <= int(raw) <= high:
        return None, '%s must be a whole number from %d to %d in plain digits, got %r' % (name, low, high, raw)
    return int(raw), None


def problems(environ=None):
    """[why] for every way the three switches and the hold's three knobs are set wrongly in `environ` (default: the process's), [] when none: a value that is not 0 or 1, a knob outside its
    range, a knob without the hold's switch (it would be ignored), the hold without the extent replay that registers the DRAM predicate, the seed-first lever without engine reuse, the
    lean admission (its reading is what the cold fallback's clearance reads) and the Lever N scheduler (the schedule() wrapper rides on its class)."""
    environ = os.environ if environ is None else environ
    found = []
    for name in SWITCHES:
        if environ.get(name) not in (None, '0', '1'):
            found.append('%s must be 0 or 1, got %r' % (name, environ.get(name)))
    for name in HOLD_KNOBS:
        if name in environ:
            _value, why = _number(name, environ)
            if why:
                found.append(why)
    if not _switch(HOLD_FLAG, environ):
        found += ['%s is set without %s=1: it would be ignored' % (name, HOLD_FLAG) for name in HOLD_KNOBS if name in environ]
    else:
        if environ.get('QWEN_FAST_EXTENT_REPLAY') != '1':
            found.append('%s=1 needs QWEN_FAST_EXTENT_REPLAY=1: the DRAM hold it reuses is registered only there' % HOLD_FLAG)
    if _switch(SEED_FLAG, environ):
        for name in ('QWEN_FAST_PARKED_ENGINES', 'QWEN_FAST_ADMISSION_LEAN', 'QWEN_FAST_LEVER_N'):
            if environ.get(name) != '1':
                found.append('%s=1 needs %s=1' % (SEED_FLAG, name))
    return found


# -- the DRAM hold's reading ----------------------------------------------------------------------------------------------------------------------

class HoldReadings(object):
    """The last allocator reading the DRAM hold took, and the rule that lets the next hold use it (module docstring). Pure host state: `clock` and `signature` are injected."""

    def __init__(self, max_age_ms=HOLD_DEFAULTS[HOLD_AGE_FLAG][0], max_reuses=HOLD_DEFAULTS[HOLD_REUSES_FLAG][0], drift_mb=HOLD_DEFAULTS[HOLD_DRIFT_FLAG][0], clock=time.monotonic):
        self.max_age_s = max_age_ms / 1000.0
        self.max_reuses = max_reuses
        self.drift = drift_mb * MEGABYTE
        self.clock = clock
        self.kept = None
        self.reuses = 0
        self.reads = 0
        self.reused = 0

    def store(self, reading, signature):
        """A real reading: it replaces the kept one and starts its count of reuses again."""
        self.kept = dict(reading, at=self.clock(), signature=signature)
        self.reuses = 0
        self.reads += 1

    def clear(self):
        self.kept = None
        self.reuses = 0

    def lookup(self, signature, short_of):
        """(reading, meta) when the kept reading may stand in for a new one, else None. It may when it is young enough, has served fewer holds than the limit, was taken under the same
        signature (the parked set's slot states and fallback count: an engine built, unparked or rebuilt in between allocates what the drift allowance does not cover) and still clears
        the hold's terms after the discount: `short_of(discounted)` must name no short term (it is serving_prefill_admission.split_short over the hold's need, with the free total and
        the largest block both less (reuses + 1) x drift, the largest block halved first). A trace term that applies needs a trace reading in the kept one (short_of sees None and refuses)."""
        kept = self.kept
        if kept is None:
            return None
        age = self.clock() - kept['at']
        if age > self.max_age_s or age < 0 or self.reuses >= self.max_reuses or kept['signature'] != signature:
            return None
        discount = (self.reuses + 1) * self.drift
        discounted = dict(free=kept['free'] - discount, largest_free=kept['largest_free'] // 2 - discount, trace_largest_free=kept.get('trace_largest_free'))
        if discounted['free'] < 0 or discounted['largest_free'] < 0 or short_of(discounted):
            return None
        self.reuses += 1
        self.reused += 1
        return dict((key, kept.get(key)) for key in ('free', 'largest_free', 'trace_largest_free', 'trace_unread')), dict(read_at=kept['at'], age_ms=age * 1000.0, reuses=self.reuses)


_HOLD = {}


def hold_state(environ=None):
    """The process's HoldReadings, built from the knobs on first use."""
    state = _HOLD.get('state')
    if state is None:
        environ = os.environ if environ is None else environ
        values = [_number(name, environ)[0] for name in HOLD_KNOBS]
        state = _HOLD['state'] = HoldReadings(*values)
    return state


def reset():
    """Forget the process's hold state (a test's, or an attach's)."""
    _HOLD.clear()


def parked_signature(parked):
    """What must be the same between a reading and its reuse: the parked set's fallback, unpark and re-park counts, its off and closed state, and each slot's state and whether its
    single-user capture is released (every one of them an allocation or a release the reading did not see). () without a parked set."""
    if parked is None:
        return ()
    try:
        return (bool(getattr(parked, 'closed', False)), bool(getattr(parked, 'off', False)),
                int(getattr(parked, 'fallbacks', 0)), int(getattr(parked, 'unparks', 0)), int(getattr(parked, 'reparks', 0)),
                tuple((entry.state, bool(parked.single_released(entry))) for entry in parked.slots))
    except Exception as failure:
        return ('unreadable', type(failure).__name__, time.monotonic())


def hold_reading(pool, read, short_of, signature=None, prompt='-', log=_log):
    """(reading, reason, meta) for one DRAM hold decision. `read(pool)` is serving_prefill_admission.dram_reading; `short_of(r)` the hold's split_short over a (discounted) reading; `signature`
    a zero-argument callable (parked_signature of the parked set). meta is None for a real reading, else dict(read_at=, age_ms=, reuses=) of the reuse."""
    state = hold_state()
    sig = () if signature is None else signature()
    hit = state.lookup(sig, short_of)
    if hit is not None:
        reading, meta = hit
        log(HOLD_REUSED_LINE, prompt, meta['age_ms'], meta['reuses'], reading['free'], reading['largest_free'])
        return reading, None, meta
    reading, reason = read(pool)
    if reading is None:
        state.clear()
    else:
        state.store(reading, sig)
    return reading, reason, None


# -- the memory ledger ----------------------------------------------------------------------------------------------------------------------------

def announce(seed_first, log=None):
    """The attach's one line naming the switches and the hold's knobs (INSTALLED_LINE)."""
    hold = hold_state() if hold_reuse_enabled() else None
    log = _log if log is None else log
    log(INSTALLED_LINE, int(bool(seed_first)), int(hold_reuse_enabled()), int(ledger_trim_enabled()), '-' if hold is None else int(hold.max_age_s * 1000),
        '-' if hold is None else hold.max_reuses, '-' if hold is None else hold.drift // MEGABYTE)


def trim_ledger(memory_ledger, log=None):
    """End the process's memory ledger after the attach (QWEN_FAST_ADMISSION_LEDGER_TRIM=1): every later reading (the prefill points, the engine points, the rebind bracket, the first packed round, the
    pair, quad and octo points of a capture) is a no-op, none of them a check (Ledger.before swallows its own errors and no judge reads a bracket per rebind). The attach's own phases are kept."""
    if memory_ledger.active() is None:
        return False
    memory_ledger.end()
    (_log if log is None else log)(LEDGER_TRIM_LINE)
    return True


# -- seed first -----------------------------------------------------------------------------------------------------------------------------------

class SeedFirst(object):
    """The lifecycle's seed-first policy: whether a request's bind may follow its seed (defer), and the registration of the completion the scheduler runs (install).

    `parked`           the attach's serving_parked_engines.ParkedEngineSet (peek() names the slot the next request takes)
    `clears`           clears(request_id, need) -> bool, serving_prefill_admission.reading_clears bound to the attach's reserve: whether the reading this request's own hold took clears `need`
    `fallback_need`    zero-argument callable: the DRAM need of the cold-build fallback (serving_prefill_admission.backstop_need over the reserve)
    `clock`            time.perf_counter (tests inject)
    """

    def __init__(self, parked, clears, fallback_need, log=_log, clock=time.perf_counter):
        self.parked, self.clears, self.fallback_need, self.log, self.clock = parked, clears, fallback_need, log, clock
        self.deferred = 0
        self.eager = {}
        self.lifecycle = None
        self.seeded_at = None

    def refusal(self, request_id, state=None):
        """Why the bind of `request_id` must stay in the sampler, or None when it may follow the seed."""
        parked = self.parked
        if parked is None or getattr(parked, 'closed', False) or getattr(parked, 'off', False):
            return 'no parked engines'
        if getattr(parked, 'fault', None) not in (None, ''):
            return 'an injected parked fault'
        try:
            entry = parked.peek()
        except Exception as failure:
            return 'the placement is unreadable (%s)' % type(failure).__name__
        if entry is None:
            return 'the slot it will take holds no parked engine (a cold build)'
        try:
            clears = self.clears(str(request_id), self.fallback_need())
        except Exception as failure:
            return 'the cold fallback\'s clearance is unreadable (%s)' % type(failure).__name__
        if not clears:
            return 'the admission\'s DRAM reading does not clear the cold fallback with a wide margin'
        return None

    def defer(self, request_id, state=None):
        """Whether the bind of `request_id` follows its seed (the lifecycle then returns the seed and its schedule() wrapper binds). Counts and logs the reason when not."""
        why = self.refusal(request_id, state)
        request_id = str(request_id)
        if why is not None:
            self.eager[why] = self.eager.get(why, 0) + 1
            self.log(SEED_EAGER_LINE, request_id[:48], why)
            return False
        self.deferred += 1
        self.seeded_at = self.clock()
        entry = self.parked.peek()
        self.log(SEED_DEFERRED_LINE, request_id[:48], getattr(entry, 'index', '-'))
        return True

    def built(self, request_id, began):
        done = self.clock()
        self.log(SEED_BUILT_LINE, str(request_id)[:48], (done - began) * 1000.0, (began - (self.seeded_at if self.seeded_at is not None else began)) * 1000.0)

    # -- the scheduler side ---------------------------------------------------------------------------------------------------------------------

    def install(self, lifecycle, config, modules=None, resolve=None):
        """Register the lifecycle's completion under HOLDER_KEY and wrap `schedule` of the scheduler class the engine runs (the one `config` names: serving_request_quarantine.scheduler_class,
        or `resolve(config)`) so that every call completes a pending bind FIRST, before the scheduler's own wrappers (the alternator's clock, the DRAM hold, the gate) see the call. Raises
        ValueError for a class without schedule: the lever would never complete."""
        modules = sys.modules if modules is None else modules
        if resolve is None:
            from serving_request_quarantine import scheduler_class as resolve
        scheduler = resolve(config)
        schedule = getattr(scheduler, 'schedule', None)
        if not callable(schedule):
            raise ValueError('%s=1 needs a scheduler class with schedule(): %r has none' % (SEED_FLAG, scheduler))
        holder = types.ModuleType(HOLDER_KEY)
        holder.complete = lifecycle.complete_deferred
        modules[HOLDER_KEY] = holder
        self.lifecycle, self.holder, self.modules = lifecycle, holder, modules
        if not getattr(schedule, SCHEDULE_WRAPPED, False):
            def wrapped(this, *args, **kwargs):
                complete_pending(modules)
                return schedule(this, *args, **kwargs)

            # update_wrapper carries the inner wrappers' marks (Lever N's _qwen_levern_schedule, the cap's): the prefix graft's install check reads them on the class's
            # schedule, and a bare wrapper hid Lever N from it (S7B-A1 card run 39649281798: 'prefix reuse refused: ... Lever N's schedule wrapper installed on the class').
            functools.update_wrapper(wrapped, schedule)
            setattr(wrapped, SCHEDULE_WRAPPED, True)
            setattr(scheduler, 'schedule', wrapped)
        return self

    def uninstall(self):
        modules = getattr(self, 'modules', None)
        if modules is not None and modules.get(HOLDER_KEY) is getattr(self, 'holder', None):
            del modules[HOLDER_KEY]
        self.lifecycle = None


def complete_pending(modules=None):
    """Run the registered lifecycle's pending bind, if any (the schedule() wrapper's one call). Absent or unregistered: nothing."""
    modules = sys.modules if modules is None else modules
    holder = modules.get(HOLDER_KEY)
    complete = getattr(holder, 'complete', None)
    if callable(complete):
        complete()
