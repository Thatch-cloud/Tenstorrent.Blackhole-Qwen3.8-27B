"""Lever N at TP4 (QWEN_FAST_LEVER_N): the pure policy, stdlib only.

Today a prefill is one engine step and every seat that is decoding stops for all of it (a cold 253,920-token prompt
is about 92 s of prefill and about 2 s of engine build, docs/lever-n-tp4-design-2026-10-04.md section 1). Lever N
splits a long prefill into steps that each end on the model's own 2,048-token outer chunk boundary and runs decode
rounds between them. This module holds the two decisions that need no device and no vLLM, so they are unit-testable
to the last token:

1. THE CHUNKING ARITHMETIC (step_budget, final_start, plan). For a prompt of P tokens, CHUNK = 2048,
   F = floor(P / CHUNK) x CHUNK and tail = P - F, the FINAL step starts at

       S_last = F - CHUNK            when tail > 0 and F >= CHUNK     (the last full chunk and the tail)
              = P - CHUNK            when tail == 0 and P >= CHUNK    (the last full chunk)
              = 0                    when P < 2 x CHUNK               (such a prompt is never split)

   so the final step is at most 2 x CHUNK + tail - 1 = 4,095 tokens and always contains the draft window
   [P - 2048, P): the drafter's feature snapshots are taken in the step that builds the engine, as they are
   today (design 3.2, hazard H1). Every non-final step ends on a multiple of 2,048, so every continuation starts
   on one: the exactness argument is op-sequence identity at the model's own chunk boundaries (design 3.4), and a
   step that is not a multiple of 2,048 is REFUSED (a ValueError at flag parse), never rounded.

2. THE ALTERNATION (Alternator). Prefill and decode never share a step on TT (the plugin scheduler runs a step that
   is all prefill or all decode). After a prefill step that paused running decoders, the policy owes them decode
   steps in proportion to the prefill's wall time:

       share mode (QWEN_FAST_LEVERN_PREFILL_SHARE = f in (0, 1]): owed += dt x (1 - f) / f after each prefill step
           that had decoders, capped at KMAX x round time; every decode step pays its own wall time; while anything
           is owed (or fewer than RMIN decode steps have run since the prefill) the next step is a decode step.
           f = 1.0 is back-to-back chunks (today's stall, chunked); f = 0.5 gives the prefill half the wall time.
       static mode (QWEN_FAST_LEVERN_ROUNDS = R): exactly R decode steps after each prefill step that had
           decoders (Lever N M2's r; it overrides f).

   The policy never idles the device: with no decoder running nothing is owed, and a yield always schedules the
   running decodes. Wall time is the scheduler's own: the interval between two schedule() calls of the synchronous
   engine is the previous step's execute + sample + update, so host and transition costs are charged where they
   occur. The Alternator has a clock parameter and no other input, so a test drives it with a list of times.

THE FLAGS (all default off or default value; every malformed value is a ValueError at parse, never a silent default):
    QWEN_FAST_LEVER_N                   0 | 1       the master switch; unset or 0, nothing here runs
    QWEN_FAST_LEVERN_STEP_TOKENS        2048        the step while decoders run (a multiple of 2,048)
    QWEN_FAST_LEVERN_SOLO_STEP_TOKENS   16384       the step with no decoder and nothing waiting (a multiple of 2,048)
    QWEN_FAST_LEVERN_PREFILL_SHARE      0.5         f in (0, 1]
    QWEN_FAST_LEVERN_ROUNDS             unset       static R in 1..64 decode steps per prefill step (overrides f)
    QWEN_FAST_LEVERN_MAX_ROUNDS         8           KMAX, 1..64
    QWEN_FAST_LEVERN_AUDIT              0 | 1       gate only: digests at the end of every prefill (also valid alone, as
                                                    the NON-interleaved control that produces the digests the audited
                                                    interleaved arm is compared with)
    QWEN_FAST_LEVERN_FAULT              unset       gate only negative control: foreign | final-hold
"""

import math
import os
import time
from collections import namedtuple

CHUNK = 2048

FLAG = 'QWEN_FAST_LEVER_N'
STEP_FLAG = 'QWEN_FAST_LEVERN_STEP_TOKENS'
SOLO_FLAG = 'QWEN_FAST_LEVERN_SOLO_STEP_TOKENS'
SHARE_FLAG = 'QWEN_FAST_LEVERN_PREFILL_SHARE'
ROUNDS_FLAG = 'QWEN_FAST_LEVERN_ROUNDS'
MAX_ROUNDS_FLAG = 'QWEN_FAST_LEVERN_MAX_ROUNDS'
AUDIT_FLAG = 'QWEN_FAST_LEVERN_AUDIT'
FAULT_FLAG = 'QWEN_FAST_LEVERN_FAULT'
FAULTS = ('foreign', 'final-hold')
# Every flag of this module: a sibling set while the master switch is off is a configuration error the contract names
# (a typo must not leave an arm silently running the control).
SIBLING_FLAGS = (STEP_FLAG, SOLO_FLAG, SHARE_FLAG, ROUNDS_FLAG, MAX_ROUNDS_FLAG, FAULT_FLAG)
ALL_FLAGS = (FLAG, AUDIT_FLAG) + SIBLING_FLAGS

DEFAULT_STEP = CHUNK
DEFAULT_SOLO = 8 * CHUNK
DEFAULT_SHARE = 0.5
DEFAULT_MAX_ROUNDS = 8
MAX_STATIC_ROUNDS = 64
RMIN = 1
# The decode round time the cap on what is owed is expressed in until a decode step has been timed (the eight-live
# round, about 245 ms on the eight-seat 262k image).
INITIAL_ROUND_MS = 250.0

# The log lines. The scheduler, the model route and the smoke check share these, so a rule never greps a line a
# producer renamed.
INSTALLED_LINE = '[PINDIAG] lever N installed on {}: step={} solo={} share={} rounds={} max_rounds={}'
PLATFORM_LINE = '[PINDIAG] lever N: chunked prefill kept for qwen3_5 (budget={} threshold={})'
STEP_LINE = ('[PINDIAG] lever N step n={} kind={} seats={} req={} start={} tokens={} end={} prompt={} final={} reason={} '
             'prev={}:{}ms owed_ms={} owed_rounds={}')
ROUTE_LINE = ('[PINDIAG] lever N route req={} start={} end={} prompt={} final={} wrote_slot={} ms={:.1f} programs={}->{}')
ROUTE_INSTALLED_LINE = '[PINDIAG] lever N route installed: {}'
ROUTE_WARM_LINE = '[PINDIAG] lever N route warmed before the packed traces: steps={} programs={}->{} ms={:.0f}'
DIGEST_LINE = '[PINDIAG] lever N digest req={} prompt={} slot_sha={} logits_sha={} kv_sha={}'
FINAL_HOLD_LINE = '[PINDIAG] lever N final-step dram hold req={} prompt={} decodes={} short={}'
REFUSED_LINE = '[PINDIAG] lever N REFUSED {}'

Config = namedtuple('Config', 'step solo share rounds max_rounds')


def _flag(environ, name):
    value = environ.get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (name, value))
    return value == '1'


def enabled(environ=None):
    """Whether QWEN_FAST_LEVER_N=1, strictly: unset or '0' is off, '1' is on, anything else is a ValueError."""
    return _flag(os.environ if environ is None else environ, FLAG)


def audit_enabled(environ=None):
    """Whether QWEN_FAST_LEVERN_AUDIT=1 (strictly 0 or 1)."""
    return _flag(os.environ if environ is None else environ, AUDIT_FLAG)


def _whole(environ, name, default, low, high=None, multiple=None):
    value = environ.get(name)
    if value is None:
        return default
    if not value.isascii() or not value.isdigit():
        raise ValueError('%s must be a whole number in plain digits, got %r' % (name, value))
    number = int(value)
    if number < low or (high is not None and number > high):
        raise ValueError('%s must be %s, got %d' % (name, 'at least %d' % low if high is None else
                                                      'from %d to %d' % (low, high), number))
    if multiple is not None and number % multiple:
        raise ValueError('%s must be a multiple of %d (a step that does not end on the model\'s own %d-token chunk '
                         'boundary is not covered by the exactness argument), got %d' % (name, multiple, multiple, number))
    return number


def _share(environ):
    value = environ.get(SHARE_FLAG)
    if value is None:
        return DEFAULT_SHARE
    if not value.isascii() or not value.replace('.', '', 1).isdigit():
        raise ValueError('%s must be a decimal number in (0, 1], got %r' % (SHARE_FLAG, value))
    share = float(value)
    if not math.isfinite(share) or not 0.0 < share <= 1.0:
        raise ValueError('%s must be in (0, 1], got %r' % (SHARE_FLAG, value))
    return share


def config(environ=None):
    """The parsed Config of `environ`: step, solo, share, rounds (None when the share mode runs), max_rounds."""
    environ = os.environ if environ is None else environ
    step = _whole(environ, STEP_FLAG, DEFAULT_STEP, CHUNK, multiple=CHUNK)
    solo = _whole(environ, SOLO_FLAG, DEFAULT_SOLO, CHUNK, multiple=CHUNK)
    if solo < step:
        raise ValueError('%s (%d) must be at least %s (%d)' % (SOLO_FLAG, solo, STEP_FLAG, step))
    rounds = _whole(environ, ROUNDS_FLAG, None, 1, MAX_STATIC_ROUNDS)
    max_rounds = _whole(environ, MAX_ROUNDS_FLAG, DEFAULT_MAX_ROUNDS, 1, MAX_STATIC_ROUNDS)
    fault = environ.get(FAULT_FLAG)
    if fault is not None and fault not in FAULTS:
        raise ValueError('%s must be one of %s, got %r' % (FAULT_FLAG, ', '.join(FAULTS), fault))
    return Config(step, solo, _share(environ), rounds, max_rounds)


def fault(environ=None):
    """The negative control QWEN_FAST_LEVERN_FAULT names, or None."""
    value = (os.environ if environ is None else environ).get(FAULT_FLAG)
    return value if value in FAULTS else None


def config_problems(environ=None):
    """Every reason the flags in `environ` cannot be read, one string each; [] when they all parse."""
    environ = os.environ if environ is None else environ
    problems = []
    try:
        on = enabled(environ)
        audit_enabled(environ)
        config(environ)
    except ValueError as failure:
        problems.append(str(failure))
        on = environ.get(FLAG) == '1'
    siblings = [name for name in SIBLING_FLAGS if environ.get(name) is not None]
    if siblings and not on:
        problems.append('%s set without %s=1: %s' % (', '.join(siblings), FLAG, 'a typo must not leave an arm running '
                                                       'the non-interleaved control'))
    return problems


def final_start(prompt):
    """S_last: where the final step of a `prompt`-token prefill starts (the module docstring). 0 means the prompt is
    never split."""
    if type(prompt) is not int or prompt < 1:
        raise ValueError('A positive integer prompt length is required, got %r' % (prompt,))
    full = prompt // CHUNK * CHUNK
    if full < CHUNK:
        return 0
    return full - CHUNK if prompt - full else prompt - CHUNK


def step_budget(prompt, computed, *, decoding, others_waiting=0, step=DEFAULT_STEP, solo=DEFAULT_SOLO):
    """The tokens one prefill step schedules for a request of `prompt` tokens with `computed` already done.

    The final step (computed >= S_last) takes everything left. Before it, a step is `step` tokens while any decoder
    runs or another prompt waits, and `solo` tokens when nobody is hurt by a longer one (the arrival of a prompt
    then waits at most one solo step); never past S_last. `computed` must be 0 or a multiple of 2,048 below S_last:
    anything else means a chunk started off the model's boundary, which is refused."""
    s_last = final_start(prompt)
    if type(computed) is not int or not 0 <= computed < prompt:
        raise ValueError('computed must be an integer in [0, %d), got %r' % (prompt, computed))
    if computed >= s_last:
        return prompt - computed
    if computed % CHUNK:
        raise ValueError('a prefill step may start only on a %d-token boundary: computed=%d prompt=%d'
                         % (CHUNK, computed, prompt))
    for name, value in (('step', step), ('solo', solo)):
        if type(value) is not int or value < CHUNK or value % CHUNK:
            raise ValueError('%s must be a positive multiple of %d, got %r' % (name, CHUNK, value))
    limit = solo if not decoding and not others_waiting else step
    return min(limit, s_last - computed)


def plan(prompt, *, decoding=True, step=DEFAULT_STEP, solo=DEFAULT_SOLO):
    """The (start, end) of every step of an uninterrupted prefill under a constant `decoding`; the tests' oracle."""
    steps, computed = [], 0
    while computed < prompt:
        budget = step_budget(prompt, computed, decoding=decoding, step=step, solo=solo)
        steps.append((computed, computed + budget))
        computed += budget
    return steps


Decision = namedtuple('Decision', 'kind reason prev_kind prev_ms owed_ms owed_rounds')


class Alternator(object):
    """The time-share (or static-R) alternation between prefill steps and decode steps.

    begin(decodes, pending) is called at the start of every schedule() and returns a Decision whose kind is 'decode'
    (run a decode-only step now: a partial prefill and the waiting queues are hidden) or 'prefill' (let the plugin
    scheduler run its default step). end(kind) is called with what the step turned out to be, so the next begin can
    charge the interval to it. A kind is 'prefill' only for a step that scheduled prefill tokens."""

    def __init__(self, cfg, clock=time.monotonic):
        self.cfg, self.clock = cfg, clock
        self.reset()
        self.round_ms = INITIAL_ROUND_MS

    def reset(self):
        self.last_at = None
        self.last_kind = None
        self.last_decodes = 0
        self.owed_ms = 0.0
        self.owed_rounds = 0
        self.since_prefill = RMIN

    def _account(self, kind, decodes, dt_ms):
        cfg = self.cfg
        if kind == 'prefill':
            if decodes > 0:
                self.since_prefill = 0
                if cfg.rounds is not None:
                    self.owed_rounds = cfg.rounds
                elif cfg.share < 1.0:
                    cap = cfg.max_rounds * self.round_ms
                    self.owed_ms = min(cap, self.owed_ms + dt_ms * (1.0 - cfg.share) / cfg.share)
            return
        self.since_prefill += 1
        # A decode step's wall time is the round time the owed cap is expressed in (a slow first step cannot move it far).
        self.round_ms = 0.8 * self.round_ms + 0.2 * dt_ms if dt_ms > 0 else self.round_ms
        if cfg.rounds is not None:
            self.owed_rounds = max(0, self.owed_rounds - 1)
        else:
            self.owed_ms = max(0.0, self.owed_ms - dt_ms)

    def owes(self):
        cfg = self.cfg
        if cfg.rounds is not None:
            return self.owed_rounds > 0
        if cfg.share >= 1.0:
            return False
        return self.owed_ms > 0.0 or self.since_prefill < RMIN

    def begin(self, decodes, pending):
        now = self.clock()
        prev_kind, prev_ms = self.last_kind, None
        if self.last_at is not None:
            prev_ms = (now - self.last_at) * 1000.0
            if prev_kind is not None:
                self._account(prev_kind, self.last_decodes, prev_ms)
        self.last_at = now
        self.last_decodes = decodes
        if not decodes:
            # Nothing to yield to: nothing is owed, so the device is never idled for a decoder that is not there.
            self.owed_ms, self.owed_rounds, self.since_prefill = 0.0, 0, RMIN
        if not pending:
            # No prefill work in the system: the next prefill starts a fresh account and is never delayed.
            self.last_kind = None
            self.last_at = None
            self.owed_ms, self.owed_rounds, self.since_prefill = 0.0, 0, RMIN
            return Decision('prefill', 'none-pending', prev_kind, prev_ms, 0.0, 0)
        if decodes and self.owes():
            return Decision('decode', 'owed', prev_kind, prev_ms, self.owed_ms, self.owed_rounds)
        return Decision('prefill', 'idle' if not decodes else 'paid', prev_kind, prev_ms, self.owed_ms, self.owed_rounds)

    def end(self, kind):
        if kind not in ('prefill', 'decode'):
            raise ValueError('a step is prefill or decode, got %r' % (kind,))
        self.last_kind = kind
