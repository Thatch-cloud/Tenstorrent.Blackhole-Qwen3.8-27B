"""One fast lane beside standard lanes (QWEN_FAST_LANE, GATE ONLY): marking, admission, the round planner and its controller.

The target for coding: ONE fast user at >= 150 tok/s running at the same time as several standard users at >= 75 tok/s each,
on the four cards at TP4. The mechanism is a hybrid frame:

    frame = one PACKED round (every live user, on the M3 block)  +  k SOLO rounds (the fast user alone, on the D0 block)

The fast user rides in every packed round like any other user, and between packed rounds it also gets k solo rounds of its own
(serving_solo_lane). Putting it into a packed round costs a few milliseconds (a padded or full M3 round already pays for four
segments), while a solo round costs a whole round of its own, so the controller raises k only as far as every standard user
stays at its floor. Per-lane rate over a frame = tokens per round x (rounds the lane is in) / (frame time): a standard user is in
one round per frame, the fast user in 1 + k.

What lives here (host only, no device, importable on a laptop):
  request_lane / LaneRefused   the request's mark, SamplingParams.extra_args['qwen_lane'] = 'fast' | 'standard'
  LaneBook                     the worker's admission: exactly ONE fast lane (pool slot 0, reserved), a second fast request is
                               DOWNGRADED to standard (never queued, never refused), standard requests never take slot 0
  LaneController               the frame policy: a tunable ratio k (fixed, or the controller's), a deficit counter for
                               fractional k, the starvation guard (k_max solo rounds between packed rounds, a packed round at
                               least every gap_max ms), the standard floor (S1), F/P/sigma EWMAs from the rounds it sees
  LaneRuntime                  book + controller + the gate the scheduler reads, and the hook's three calls: plan (before the
                               drafts), publish, note_round (after the round)
  the telemetry lines          [LANE-ADMIT] [LANE-ROUND] [LANE-FRAME] (and [LANE-SCHED] from the scheduler)

The scheduler side (serving_fast_lane_scheduler) reads the plan through a module parked at sys.modules[GATE_KEY] - the
_qwen_prefill_gate pattern - and hides every running decode the round does not serve, so vLLM schedules exactly the round's
members and the worker's "scheduled set == prepared set" contract holds unchanged.

PLANNING IS PURE. serving_early_draft drafts the next round INSIDE execute_model and take_draft_token_ids discards and redrafts
when anything moved, so plan() may run twice for one round: it reads the controller's state and changes none. The state moves in
note_round, which runs once per executed round.

Exactness: the lane changes WHICH round a user is in, never what a round computes. Every user's text is a function of its own
prompt alone (the packed block's segments are independent and the solo block is the same program at one segment); the gate's
`lanes` plan asserts it against each user's solo run.
"""

import collections
import math
import os
import sys
import types

LANE_FLAG = 'QWEN_FAST_LANE'
RATIO_FLAG = 'QWEN_FAST_LANE_RATIO'          # 'auto' (default) or a fixed k: solo rounds per packed round, 0 <= k <= k_max
KMAX_FLAG = 'QWEN_FAST_LANE_KMAX'            # the most solo rounds between two packed rounds (default 3)
GAP_FLAG = 'QWEN_FAST_LANE_GAP_MS'           # a packed round at least this often, in ms (default 250)
FLOOR_FLAG = 'QWEN_FAST_LANE_FLOOR'          # the standard lanes' floor, tok/s (default 75)
MARGIN_FLAG = 'QWEN_FAST_LANE_MARGIN'        # the controller aims at floor x (1 + margin) (default 0.05)
RESERVE_FLAG = 'QWEN_FAST_LANE_RESERVE'      # 1 (default): slot 0 is the fast request's alone; 0: a standard request may lend it
EXTRA_ARG = 'qwen_lane'
FAST, STANDARD = 'fast', 'standard'
LANES = (FAST, STANDARD)
GATE_KEY = '_qwen_lane_gate'
FAST_SLOT = 0
FLAG_NAMES = (LANE_FLAG, RATIO_FLAG, KMAX_FLAG, GAP_FLAG, FLOOR_FLAG, MARGIN_FLAG, RESERVE_FLAG)

DEFAULT_KMAX = 3
DEFAULT_GAP_MS = 250.0
DEFAULT_FLOOR = 75.0
DEFAULT_MARGIN = 0.05
# Rounds further apart than this are not one decode cadence (a prefill, a stall): they teach the controller nothing, and the
# time is counted as stall (the steady estimator excludes the same rounds).
STALL_MS = 2000.0
EWMA_ALPHA = 0.2
RATE_EVENTS = 16          # each standard lane's rate is read over its last this many commit events
MIN_RATE_EVENTS = 4
MIN_TAU_ROUNDS = 4        # a standard lane's tokens per round bind the controller after this many rounds
RAMP = 0.25               # k rises by at most this per frame (it halves at once when the floor is missed)

ADMIT_MARKER = '[LANE-ADMIT]'
ROUND_MARKER = '[LANE-ROUND]'
FRAME_MARKER = '[LANE-FRAME]'
SCHED_MARKER = '[LANE-SCHED]'
DESYNC_MARKER = '[LANE-DESYNC]'
ENGAGED_MARKER = '[LANE] engaged'


class LaneRefused(ValueError):
    """A request's lane mark the contract does not admit: this request's own terms (D2 quarantines it), never the engine's."""


def _flag01(name, environ, default='0'):
    value = environ.get(name, default)
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (name, value))
    return value == '1'


def lane_requested(environ=None):
    """QWEN_FAST_LANE=1, strictly: unset or '0' is off, '1' on, anything else a configuration error."""
    return _flag01(LANE_FLAG, os.environ if environ is None else environ)


def _decimal(name, text, low, high, whole=False):
    try:
        value = int(text) if whole else float(text)
    except (TypeError, ValueError):
        raise ValueError('%s must be a %s, got %r' % (name, 'whole number' if whole else 'decimal number', text)) from None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError('%s must be finite, got %r' % (name, text))
    if not low <= value <= high:
        raise ValueError('%s must be within [%s, %s], got %r' % (name, low, high, text))
    return value


class LaneConfig(collections.namedtuple('LaneConfig', 'ratio k_max gap_ms floor margin reserve seats')):
    """The lanes' knobs, read once at attach. `ratio` is None for the controller's k ('auto') or the fixed k."""

    __slots__ = ()

    @classmethod
    def from_environment(cls, environ=None, *, seats=4):
        environ = os.environ if environ is None else environ
        k_max = _decimal(KMAX_FLAG, environ.get(KMAX_FLAG, DEFAULT_KMAX), 1, 8, whole=True)
        text = environ.get(RATIO_FLAG, 'auto')
        ratio = None if text == 'auto' else _decimal(RATIO_FLAG, text, 0.0, float(k_max))
        return cls(ratio=ratio, k_max=k_max,
                   gap_ms=_decimal(GAP_FLAG, environ.get(GAP_FLAG, DEFAULT_GAP_MS), 20.0, 5000.0),
                   floor=_decimal(FLOOR_FLAG, environ.get(FLOOR_FLAG, DEFAULT_FLOOR), 1.0, 1000.0),
                   margin=_decimal(MARGIN_FLAG, environ.get(MARGIN_FLAG, DEFAULT_MARGIN), 0.0, 0.5),
                   reserve=_flag01(RESERVE_FLAG, environ, '1'), seats=seats)


def lane_admission(solo_lane, environ=None, *, seats, log=None):
    """None while QWEN_FAST_LANE is off (nothing else is read). Else the LaneConfig, or ValueError naming every reason the attach
    is refused: the solo lane was not admitted (a solo round IS the fast user's extra round, and the solo admission already holds
    the gate-run, four-card and shape conditions), QWEN_FAST_ANY_REQUEST is not 1 (a bad lane mark ends as one request's
    quarantine, the C2-any D2 consumer), the seats are not four, or a knob is malformed."""
    environ = os.environ if environ is None else environ
    if not lane_requested(environ):
        return None
    problems = []
    if solo_lane is None:
        problems.append('QWEN_FAST_SOLO_LANE is not 1: the fast lane\'s extra rounds run on the one-user block')
    if environ.get('QWEN_FAST_ANY_REQUEST') != '1':
        problems.append('QWEN_FAST_ANY_REQUEST is not 1: a bad lane mark is quarantined by the C2-any consumer')
    if seats != 4:
        problems.append('the lanes are built for four seats (one fast, up to three standard), not %r' % (seats,))
    config = None
    try:
        config = LaneConfig.from_environment(environ, seats=seats)
    except ValueError as failure:
        problems.append(str(failure))
    if problems:
        for problem in problems:
            if log is not None:
                log('[LANE] refused: {}', problem)
        raise ValueError('%s=1 is refused: %s' % (LANE_FLAG, '; '.join(problems)))
    return config


# --- the request's mark ---------------------------------------------------------------------------------------------------

def request_lane(sampling_params):
    """The lane a request asks for: 'fast' or 'standard' (absent means standard). Anything else - an unknown name, a value
    that is not a string, several marks - raises LaneRefused. The mark changes no sampling, so validate_request_sampling is
    what it was; a platform maps an authenticated entitlement to this field and strips any client-supplied value."""
    extra = getattr(sampling_params, 'extra_args', None)
    if extra is None:
        return STANDARD
    if not isinstance(extra, dict):
        raise LaneRefused('extra_args must be a mapping, got %s' % type(extra).__name__)
    if EXTRA_ARG not in extra:
        return STANDARD
    value = extra[EXTRA_ARG]
    if type(value) is not str or value not in LANES:
        raise LaneRefused('%s must be one of %s, got %r' % (EXTRA_ARG, '/'.join(LANES), value))
    return value


def lane_of(request):
    """A scheduler request's asked lane, tolerant: anything unreadable is standard (the worker's admission is the judge)."""
    try:
        return request_lane(getattr(request, 'sampling_params', None))
    except LaneRefused:
        return STANDARD


# --- admission ------------------------------------------------------------------------------------------------------------

class Grant(collections.namedtuple('Grant', 'request_id asked granted slot_order reason alias')):
    __slots__ = ()


class LaneBook:
    """The worker's lane registry: which request holds the ONE fast lane, and each request's granted lane.

    admit(request_id, asked) returns the Grant: a fast request is granted the fast lane, pool slot 0 alone, while nobody holds it;
    a second one is DOWNGRADED to standard (reason 'fast-lane-busy'; the response header the platform adds says so) - it is not
    queued and not refused, so a fast request never waits behind the fast lane. Standard requests take slots 1..seats-1 under
    the reserve (slot 0 is the fast request's alone: a fast arrival never waits for a standard user to finish), or 1..seats-1
    and then 0 when the reserve is off (lend: a fast arrival while a standard user holds slot 0 is downgraded, reason 'slot-busy').
    """

    def __init__(self, config, log=None):
        self.config, self.log = config, log
        self.granted = collections.OrderedDict()      # request id -> lane
        self.fast_id = None
        # A short stable name per request, u1, u2, ... in admission order: the per-round lines name their members by it (a request
        # id is 48 characters, four of them overrun the log line), and the [LANE-ADMIT] line is where a gate maps it to the id.
        self.aliases = {}
        self.serial = 0

    def slot_order(self, lane):
        seats = self.config.seats
        others = tuple(range(1, seats))
        if lane == FAST:
            return (FAST_SLOT,)
        return others if self.config.reserve else others + (FAST_SLOT,)

    def admit(self, request_id, asked, *, slot0_free=True):
        """Grant `asked` (a lane name) to `request_id`. `slot0_free` is the pool's answer for slot 0 (only read when the
        reserve is off and a fast request asks)."""
        if asked not in LANES:
            raise LaneRefused('%s must be one of %s, got %r' % (EXTRA_ARG, '/'.join(LANES), asked))
        if request_id in self.granted:
            raise ValueError('Request %r is already in the lane book' % (request_id,))
        granted, reason = asked, 'ok'
        if asked == FAST:
            if self.fast_id is not None:
                granted, reason = STANDARD, 'fast-lane-busy'
            elif not slot0_free:
                granted, reason = STANDARD, 'slot-busy'
        if granted == FAST:
            self.fast_id = request_id
        self.granted[request_id] = granted
        self.serial += 1
        alias = self.aliases[request_id] = 'u%d' % self.serial
        grant = Grant(request_id, asked, granted, self.slot_order(granted), reason, alias)
        self._log('{} request={} alias={} asked={} granted={} slots={} reason={}', ADMIT_MARKER, str(request_id)[:48], alias,
                  asked, granted, ','.join(map(str, grant.slot_order)), reason)
        return grant

    def release(self, request_id):
        """The request left (finished, aborted, detached). Idempotent; frees the fast lane if it held it."""
        lane = self.granted.pop(request_id, None)
        if lane == FAST and self.fast_id == request_id:
            self.fast_id = None
        if lane is not None:
            self.aliases.pop(request_id, None)
        return lane

    def lane(self, request_id):
        return self.granted.get(request_id)

    def alias(self, request_id):
        return self.aliases.get(request_id) or short(request_id)

    def _log(self, template, *values):
        if self.log is not None:
            self.log(template, *values)


def short(request_id):
    """A request id as the [LANE-*] lines print it: the last 12 characters (the ids differ at their tail)."""
    return str(request_id)[-12:]


# --- the controller ------------------------------------------------------------------------------------------------------

class Ewma:
    __slots__ = ('alpha', 'value', 'count')

    def __init__(self, alpha=EWMA_ALPHA):
        self.alpha, self.value, self.count = alpha, None, 0

    def add(self, sample):
        self.value = sample if self.value is None else self.value + self.alpha * (sample - self.value)
        self.count += 1
        return self.value


class RateWindow:
    """One lane's steady decode rate over its last RATE_EVENTS commit events: tokens committed after the first event over the time
    from the first to the last (the steady estimator's convention: the first round's gap is not a cadence)."""

    def __init__(self, events=RATE_EVENTS):
        self.events = collections.deque(maxlen=events)

    def add(self, now, tokens):
        self.events.append((now, tokens))

    def reset(self):
        self.events.clear()

    def rate(self):
        if len(self.events) < MIN_RATE_EVENTS:
            return None
        span = self.events[-1][0] - self.events[0][0]
        if span <= 0:
            return None
        return sum(tokens for _, tokens in list(self.events)[1:]) / span


class Decision(collections.namedtuple('Decision', 'kind k reason')):
    """kind 'solo' or 'packed'; k the frame's target; reason what bound it (the [LANE-FRAME] `reason`)."""

    __slots__ = ()


def closed_form_k(packed_ms, solo_ms, switch_ms, tau_std, floor):
    """The largest k with the slowest standard lane at `floor` (tok/s): one packed round plus k solo rounds must fit the time a
    standard user needs to write tau_std tokens at the floor, T = 1000 tau_std / floor ms. Switching costs sigma each way, once a
    frame (at most min(k, 1) of them, as k below one runs a solo round only in some frames):

        T - P >= F + 2 sigma :  k = (T - P - 2 sigma) / F        (k >= 1)
        otherwise            :  k = (T - P) / (F + 2 sigma)      (k < 1)

    0 when the packed round alone already exceeds T (the floor is out of reach at any k)."""
    if not all(value is not None and value > 0 for value in (packed_ms, solo_ms, tau_std, floor)):
        return None
    T = 1000.0 * tau_std / floor
    if T <= packed_ms:
        return 0.0
    switch_ms = max(0.0, switch_ms or 0.0)
    if T - packed_ms >= solo_ms + 2 * switch_ms:
        return (T - packed_ms - 2 * switch_ms) / solo_ms
    return (T - packed_ms) / (solo_ms + 2 * switch_ms)


class LaneController:
    """The frame policy. Pure to read (`decide`), moved only by `note_round` - once per executed round.

    A frame is one packed round then solo rounds. The state that decides:
      credit     the deficit counter: each executed packed round adds the frame's k, each solo round takes one; a solo round is
                 due while credit >= 1. Fractional k (0.28 solo rounds per frame) is a solo round every few frames; the credit is
                 capped at k_max so a change of k cannot bank a burst.
      solo_run   solo rounds since the last packed round.

    Guarantees (the design's S1, S2, F1, F2):
      S1  the floor: k rises only while the slowest standard lane's measured rate is at least the floor, and drops at once -
          k halved, credit cleared - when it falls below (fixed ratios skip this: they are the experiment).
      S2  no starvation: never more than k_max solo rounds between two packed rounds, and a packed round at least every gap_ms.
      F1  the fast lane is in every packed round, so it is never slower than a standard lane.
      F2  with no standard user live every round is a solo round; with no fast user live every round is a packed round.
    """

    def __init__(self, config, *, log=None):
        self.config, self.log = config, log
        self.credit = 0.0
        self.solo_run = 0
        self.last_packed_end = None
        self.last_end = None
        self.last_kind = None
        self.k = 0.0 if config.ratio is None else config.ratio
        self.frame = 0
        self.rounds = 0
        self.stall_ms = 0.0
        # The cycle of each kind of round as run (packed_ms, solo_ms: every round), the same over the rounds that did NOT follow a
        # switch (the base), and what a switch costs on top (switch_extra: the rounds that followed one against their base).
        self.packed_ms, self.solo_ms, self.switch_extra = Ewma(), Ewma(), Ewma()
        self.packed_base, self.solo_base = Ewma(), Ewma()
        self.tau_fast, self.tau_std = Ewma(), {}
        self.fast_rate, self.std_rates = RateWindow(), {}
        self.reason = 'start'

    # -- reading

    def floor_target(self):
        return self.config.floor * (1.0 + self.config.margin)

    def std_rate_min(self):
        rates = [rate for rate in (window.rate() for window in self.std_rates.values()) if rate is not None]
        return min(rates) if rates else None

    def tau_std_min(self):
        values = [ewma.value for ewma in self.tau_std.values() if ewma.value is not None and ewma.count >= MIN_TAU_ROUNDS]
        return min(values) if values else None

    def target_k(self):
        """The k the frame aims at: the fixed ratio, or the closed form on the measured EWMAs held below what the measured standard
        rate allows (S1). Returns (k, reason)."""
        config = self.config
        if config.ratio is not None:
            return config.ratio, 'fixed'
        packed, solo = self.packed_ms.value, self.solo_ms.value
        if packed is None:
            return 0.0, 'probe-packed'
        if solo is None:
            return 0.0, 'probe-solo'      # decide() runs one solo round to learn F once packed rounds are known
        tau = self.tau_std_min()
        # The EWMAs are cycles measured in the frame as run - every round after the first of a frame follows a switch, and its
        # cycle already holds sigma - so the closed form takes sigma as zero here (sigma_ms is reported, from the rounds that
        # follow a switch against those that do not). The first frames run at k = 0 and know no switch: the measured floor below
        # (S1) is what catches the few ms that misses.
        closed = closed_form_k(packed, solo, 0.0, tau, self.floor_target())
        if closed is None:
            return 0.0, 'probe-tau'
        if closed <= 0.0:
            return 0.0, 'std-below-floor-at-k0'
        rate = self.std_rate_min()
        if rate is not None and rate < config.floor:
            return min(closed, self.k * 0.5, float(config.k_max)), 'floor-bound'
        ramped = min(closed, float(config.k_max), self.k + RAMP)
        return ramped, ('closed-form' if ramped >= min(closed, float(config.k_max)) - 1e-9 else 'ramp')

    def decide(self, now, *, fast_live, standard_live, solo_possible):
        """The coming round's kind. `solo_possible`: the fast request can run a solo round (D0 admits it). Pure."""
        if not fast_live:
            return Decision('packed', 0.0, 'no-fast')
        if not standard_live:
            return Decision('solo', float(self.config.k_max), 'no-standard') if solo_possible else Decision('packed', 0.0, 'no-solo')
        k, reason = self.target_k()
        if not solo_possible:
            return Decision('packed', k, 'no-solo')
        if self.solo_run >= self.config.k_max:
            return Decision('packed', k, 'kmax')
        if self.last_packed_end is not None and (now - self.last_packed_end) * 1000.0 >= self.config.gap_ms:
            return Decision('packed', k, 'gap-bound')
        if reason == 'probe-solo' and self.packed_ms.count >= 2 and self.solo_run == 0:
            return Decision('solo', k, 'probe-solo')
        if self.credit >= 1.0 - 1e-9:
            return Decision('solo', k, reason)
        return Decision('packed', k, reason)

    # -- moving

    def note_round(self, now, kind, *, committed, fast_ids=(), standard_ids=(), live=None, switch=False):
        """One executed round: `kind`, `committed` {request id: tokens}, the lanes of the members, `now` (seconds, monotonic),
        `switch` (the round ran on the other block than the last one). The round's cycle is the time since the last round ended;
        rounds further apart than STALL_MS are stalls (a prefill, an idle server) and teach nothing. Returns the fields."""
        config = self.config
        cycle = None
        if self.last_end is not None:
            gap = (now - self.last_end) * 1000.0
            if gap <= STALL_MS:
                cycle = gap
            else:
                self.stall_ms += gap
                self.credit, self.solo_run = 0.0, 0
        self.rounds += 1
        self.last_end = now
        for request_id in standard_ids:
            self.std_rates.setdefault(request_id, RateWindow()).add(now, committed.get(request_id, 0))
            self.tau_std.setdefault(request_id, Ewma()).add(max(committed.get(request_id, 0), 0))
        for request_id in fast_ids:
            self.fast_rate.add(now, committed.get(request_id, 0))
            self.tau_fast.add(max(committed.get(request_id, 0), 0))
        if cycle is not None:
            everything, base = (self.packed_ms, self.packed_base) if kind == 'packed' else (self.solo_ms, self.solo_base)
            everything.add(cycle)
            if switch:
                if base.value is not None:
                    self.switch_extra.add(max(cycle - base.value, 0.0))
            else:
                base.add(cycle)
        if kind == 'packed':
            self.last_packed_end = now
            self.solo_run = 0
            self.frame += 1
            k, self.reason = self.target_k()
            if k < self.k and self.reason == 'floor-bound':
                self.credit = 0.0
            self.k = k
            self.credit = min(self.credit + k, float(config.k_max))
        else:
            self.solo_run += 1
            self.credit = max(self.credit - 1.0, 0.0)
        self.last_kind = kind
        return dict(round=self.rounds, kind=kind, cycle_ms=cycle)

    def frame_fields(self):
        return dict(frame=self.frame, k=self.k, F_ms=self.solo_ms.value, P_ms=self.packed_ms.value,
                    sigma_ms=self.switch_extra.value, tau_fast=self.tau_fast.value, tau_std_min=self.tau_std_min(),
                    fast_rate=self.fast_rate.rate(), std_rate_min=self.std_rate_min(), floor=self.config.floor,
                    reason=self.reason, stall_ms=self.stall_ms)

    def forget(self, request_id):
        """A request left: its rate window and tau no longer bind the controller."""
        self.std_rates.pop(request_id, None)
        self.tau_std.pop(request_id, None)

    def reset_fast(self):
        self.fast_rate.reset()


# --- the runtime the hook talks to -----------------------------------------------------------------------------------------

def gate():
    """The shared holder the scheduler side reads, created on first use by whichever side runs first."""
    holder = sys.modules.get(GATE_KEY)
    if holder is None:
        holder = types.ModuleType(GATE_KEY)
        holder.members = None       # frozenset of request ids the coming round serves, or None: no gate this round
        holder.fast_id = None       # the request holding the fast lane
        holder.round = 0
        holder.seats = 0
        holder.reserve = False
        holder.log = None
        holder.aliases = {}
        sys.modules[GATE_KEY] = holder
    return holder


class Plan(collections.namedtuple('Plan', 'kind members k reason')):
    """kind 'solo' or 'packed'; members the request ids the round serves (a frozenset)."""

    __slots__ = ()


def fmt(value, digits=1):
    return '-' if value is None else ('%.' + str(digits) + 'f') % value


class LaneRuntime:
    """Book + controller + gate, and the three calls the worker hook makes for a round."""

    def __init__(self, config, *, log=None, clock=None):
        import time

        self.config = config
        self.log = log
        self.clock = time.perf_counter if clock is None else clock
        self.book = LaneBook(config, log=log)
        self.controller = LaneController(config, log=log)
        self.gate = gate()
        self.gate.members, self.gate.fast_id = None, None
        self.gate.seats, self.gate.reserve, self.gate.log = config.seats, config.reserve, log
        self.gate.aliases = self.book.aliases
        self.planned = None
        self.last_block = None
        self.rounds = 0
        if log is not None:
            log('{} seats={} ratio={} k_max={} gap_ms={} floor={} margin={} reserve={}', ENGAGED_MARKER, config.seats,
                'auto' if config.ratio is None else config.ratio, config.k_max, config.gap_ms, config.floor,
                config.margin, int(config.reserve))

    # -- admission (the bridge factory)

    def admit(self, request_id, sampling_params, *, slot0_free=True):
        grant = self.book.admit(request_id, request_lane(sampling_params), slot0_free=slot0_free)
        self.gate.fast_id = self.book.fast_id
        return grant

    def release(self, request_id):
        lane = self.book.release(request_id)
        self.controller.forget(request_id)
        if lane == FAST:
            self.controller.reset_fast()
        self.gate.fast_id = self.book.fast_id
        if self.planned is not None and request_id in self.planned.members:
            self.planned = None
            self.gate.members = None
        return lane

    # -- planning (before the drafts)

    def plan(self, requests, *, rows_of, solo_rows_of, now=None):
        """The coming round, or None when the lanes do not own it (today's path serves it, nothing hidden).

        `requests` are the live decode requests (objects with .session.request_id / .session.finished); `rows_of(requests)` is
        the packed step's proposal_rows over a set (None: the block cannot serve it as one pass); `solo_rows_of(request)` the
        same for one request on the solo block. Pure: reads the controller, changes nothing.
        """
        now = self.clock() if now is None else now
        live = [request for request in requests if not request.session.finished]
        if not live:
            return None
        lanes = {request.session.request_id: self.book.lane(request.session.request_id) for request in live}
        if any(lane is None for lane in lanes.values()):
            return None        # a request the book does not know (admitted before the lanes were engaged): not ours
        fast = [request for request in live if lanes[request.session.request_id] == FAST]
        standard = [request for request in live if lanes[request.session.request_id] != FAST]
        solo_possible = bool(fast) and solo_rows_of(fast[0]) is not None
        decision = self.controller.decide(now, fast_live=bool(fast), standard_live=bool(standard),
                                          solo_possible=solo_possible)
        if decision.kind == 'solo':
            return Plan('solo', frozenset([fast[0].session.request_id]), decision.k, decision.reason)
        if rows_of(live) is None:
            return None        # a narrow tail, five live, a k-v conflict: the round is not the block's, and not ours
        return Plan('packed', frozenset(lanes), decision.k, decision.reason)

    def publish(self, plan):
        """Hand the plan to the scheduler side: which requests the coming round serves (None: hide nothing)."""
        self.planned = plan
        self.gate.members = None if plan is None else plan.members
        self.gate.round = self.rounds + 1

    # -- the executed round (after it)

    def note_round(self, kind, request_ids, committed, *, now=None, block=None, live=None):
        """Once per executed round: teach the controller, log the round, and the frame after each packed round."""
        now = self.clock() if now is None else now
        switch = self.last_block is not None and block is not None and block != self.last_block
        self.last_block = block if block is not None else self.last_block
        fast_ids = [rid for rid in request_ids if self.book.lane(rid) == FAST]
        standard_ids = [rid for rid in request_ids if self.book.lane(rid) != FAST]
        fields = self.controller.note_round(now, kind, committed=committed, fast_ids=fast_ids, standard_ids=standard_ids,
                                            live=live, switch=switch)
        self.rounds += 1
        self._log('{} round={} kind={} block={} members={} live={} ms={} committed={} switch={}', ROUND_MARKER, self.rounds,
                  kind, block or kind, ','.join(self.book.alias(rid) for rid in request_ids), live if live is not None else len(request_ids),
                  fmt(fields['cycle_ms']), ','.join(str(committed.get(rid, 0)) for rid in request_ids), int(switch))
        if kind == 'packed':
            frame = self.controller.frame_fields()
            self._log('{} frame={} k={} F_ms={} P_ms={} sigma_ms={} tau_fast={} tau_std_min={} fast_rate={} std_rate_min={} '
                      'floor={} reason={}', FRAME_MARKER, frame['frame'], fmt(frame['k'], 2), fmt(frame['F_ms']),
                      fmt(frame['P_ms']), fmt(frame['sigma_ms'], 2), fmt(frame['tau_fast']), fmt(frame['tau_std_min']),
                      fmt(frame['fast_rate']), fmt(frame['std_rate_min']), fmt(frame['floor'], 0), frame['reason'])
        return fields

    def note_other(self, kind, request_ids, committed, *, now=None, live=None):
        """A round neither block served (the sequential step: a narrow tail, survivors). It is logged, teaches the controller no
        round time, and moves its clock so the next cycle is not counted from before it."""
        now = self.clock() if now is None else now
        self.controller.last_end = now
        self.rounds += 1
        self._log('{} round={} kind={} block=- members={} live={} ms=- committed={} switch=0', ROUND_MARKER, self.rounds, kind,
                  ','.join(self.book.alias(rid) for rid in request_ids), live if live is not None else len(request_ids),
                  ','.join(str(committed.get(rid, 0)) for rid in request_ids))

    def _log(self, template, *values):
        if self.log is not None:
            self.log(template, *values)
