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

THE MERGED ROUTE (Lever N with prefix reuse, docs/lever-n-prefix-merged-route.md; every flag below is read by merged_config and is
meaningful only beside QWEN_FAST_LEVER_N=1):
    QWEN_FAST_LEVERN_TTFT_TARGET_S      180         the deadline governor's target T* in whole seconds (a client deadline for a large
                                                    prompt, minus margin); 0 turns the governor off
    QWEN_FAST_LEVERN_SHORT_TOKENS       16384       a waiting request whose remaining tokens after its peeked hit are at most this is
                                                    SHORT (admission v2: it may preempt a long prefill at a step boundary)
    QWEN_FAST_LEVERN_PARK               0 | host    host: a short may park a long prefill's scratch to host (admission v2); 0: one prefill
                                                    in flight (v1)
    QWEN_FAST_LEVERN_PARK_SLOTS         1           parked prefills at once, 1..4
    QWEN_FAST_LEVERN_MAX_PARK_S         30          a long parked this many seconds in total admits no further short, 1..3600
    QWEN_FAST_LEVERN_EPOCH_SCOPE        global      route: an intermediate step that wrote no decode slot bumps no fixture epoch
                                                    (verify_prestage's disjoint writer class); global: every step does (today's rule)
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
TTFT_FLAG = 'QWEN_FAST_LEVERN_TTFT_TARGET_S'
SHORT_FLAG = 'QWEN_FAST_LEVERN_SHORT_TOKENS'
PARK_FLAG = 'QWEN_FAST_LEVERN_PARK'
PARK_SLOTS_FLAG = 'QWEN_FAST_LEVERN_PARK_SLOTS'
MAX_PARK_FLAG = 'QWEN_FAST_LEVERN_MAX_PARK_S'
EPOCH_FLAG = 'QWEN_FAST_LEVERN_EPOCH_SCOPE'
PARK_MODES = ('0', 'host')
EPOCH_SCOPES = ('global', 'route')
MERGED_FLAGS = (TTFT_FLAG, SHORT_FLAG, PARK_FLAG, PARK_SLOTS_FLAG, MAX_PARK_FLAG, EPOCH_FLAG)
# Every flag of this module: a sibling set while the master switch is off is a configuration error the contract names
# (a typo must not leave an arm silently running the control).
SIBLING_FLAGS = (STEP_FLAG, SOLO_FLAG, SHARE_FLAG, ROUNDS_FLAG, MAX_ROUNDS_FLAG, FAULT_FLAG) + MERGED_FLAGS
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
# The merged route's step line: STEP_LINE and the share the deadline governor ran the alternation at (c2_smoke_check's STEP regex reads the prefix).
STEP_LINE_MERGED = STEP_LINE + ' f_eff={:.3f}'
# wrote_slot is MEASURED (the number of _write_gdn_slot calls the step made, not a function of the step's position) and window is the
# program-cache entries the drafter-window snapshot compiled inside the step (dflash_prefill_window.window_programs): they cannot be warmed
# (keyed on the prompt's geometry), and the four-card tripwire excludes them the same way (B-A-W).
ROUTE_LINE = ('[PINDIAG] lever N route req={} start={} end={} prompt={} final={} wrote_slot={} ms={:.1f} programs={}->{} window={}')
# The merged route's route line: ROUTE_LINE and where the step's state came from and which checkpoints it took.
ROUTE_LINE_MERGED = ROUTE_LINE + ' source={} captured={}'
ROUTE_INSTALLED_LINE = '[PINDIAG] lever N route installed: {}'
ROUTE_WARM_LINE = '[PINDIAG] lever N route warmed before the packed traces: steps={} programs={}->{} ms={:.0f}'
# tokens_sha keys a digest to its prompt (two prompts of one length admitted in a different order across arms must not be compared
# with each other). kv_sha is KV_SKIPPED for a prompt longer than KV_DIGEST_MAX_PROMPT: the digest reads every cache tensor of the whole
# pool to the host (about a minute or more), so only the single-user rows of levern_equal (the longest is 32,785 tokens) take it.
DIGEST_LINE = '[PINDIAG] lever N digest req={} prompt={} tokens_sha={} slot_sha={} logits_sha={} kv_sha={}'
KV_DIGEST_MAX_PROMPT = 32785
KV_SKIPPED = '0' * 32
FINAL_HOLD_LINE = '[PINDIAG] lever N final-step dram hold req={} prompt={} decodes={} short={}'
REFUSED_LINE = '[PINDIAG] lever N REFUSED {}'
# The merged route's lines (docs/lever-n-prefix-merged-route.md). PARK_LINE: a scratch parked to host or restored from it; QUARANTINE_LINE: a request
# ended before allocation because the state its continuation needs is gone; EPOCH_LINE: the fixture-epoch writer class a step took.
PARK_LINE = '[PINDIAG] lever N park {} req={} at={} ms={:.1f} bytes={} parked_now={}'
QUARANTINE_LINE = '[PINDIAG] lever N quarantine req={}: {}'
KILL_LINE = '[PINDIAG] lever N kill switch {} present: no new prompt is split and the short lane is closed until the engine restarts'
SHORT_LINE = '[PINDIAG] lever N short lane req={} remaining={} parks={} active={}'
OFF_FILE = 'levern.off'
OFF_PATH = '/models/.qwen-c2/levern.off'
OFF_PATH_ENV = 'QWEN_FAST_LEVERN_OFF_PATH'
OFF_POLL_S = 1.0
# What a parked scratch costs on the host (qwen_prefix_registry.CHECKPOINT_NBYTES: 48 GDN layers x both chips' fp32 state and bf16 carry).
PARK_NBYTES = 2 * 48 * (24 * 128 * 128 * 4 + 3 * 5120 * 2)
# What the route and the scheduler share (the lifecycle's prefill gate pattern: a module parked under a fixed sys.modules key, so neither imports
# the other): the scratch's owner and the parked requests, each as {request id: the position its state is at}.
STATE_KEY = '_qwen_levern_state'

Config = namedtuple('Config', 'step solo share rounds max_rounds')


def state_holder(create=True):
    """The shared route state: owner (the (request id, position) whose suspended state the B=1 scratch holds, or None) and parks ({request id:
    position} of the states parked on the host). The route writes it on every change; the scheduler reads it before it allocates anything
    for a continuation, so a request whose state is gone is ended before it takes a block. None when absent and not `create`."""
    import sys
    import types

    holder = sys.modules.get(STATE_KEY)
    if holder is None and create:
        holder = types.ModuleType(STATE_KEY)
        holder.owner = None
        holder.parks = {}
        holder.route = False
        sys.modules[STATE_KEY] = holder
    return holder


def reset_state():
    holder = state_holder()
    holder.owner, holder.parks, holder.route = None, {}, False


def state_has(request_id, position):
    """Whether the route can continue `request_id` from `position`: it owns the scratch there or has a parked state there. True when no route
    has published (nothing to check against)."""
    holder = state_holder(create=False)
    if holder is None or not getattr(holder, 'route', False):
        return True
    owner = holder.owner
    if owner is not None and owner[0] == request_id and owner[1] == position:
        return True
    return holder.parks.get(request_id) == position


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


Merged = namedtuple('Merged', 'ttft_s short_tokens park park_slots max_park_s epoch_scope')
DEFAULT_TTFT_S = 180
DEFAULT_SHORT_TOKENS = 16384
DEFAULT_PARK_SLOTS = 1
DEFAULT_MAX_PARK_S = 30
# A long prefill's hit lives for the whole parked time on the host: 154 MB per parked scratch at 262k (design 3.8), so the slot count is small.
MAX_PARK_SLOTS = 4


def merged_config(environ=None):
    """The merged route's flags (MERGED_FLAGS): the TTFT target, the short class, host parking and its limits, the epoch scope. Every value
    is strict (a ValueError names it); the defaults are what a Lever N profile without them runs (no parking, the governor on)."""
    environ = os.environ if environ is None else environ
    park = environ.get(PARK_FLAG, '0')
    if park not in PARK_MODES:
        raise ValueError('%s must be one of %s, got %r' % (PARK_FLAG, ', '.join(PARK_MODES), park))
    scope = environ.get(EPOCH_FLAG, 'global')
    if scope not in EPOCH_SCOPES:
        raise ValueError('%s must be one of %s, got %r' % (EPOCH_FLAG, ', '.join(EPOCH_SCOPES), scope))
    return Merged(_whole(environ, TTFT_FLAG, DEFAULT_TTFT_S, 0, 3600), _whole(environ, SHORT_FLAG, DEFAULT_SHORT_TOKENS, 0, 262144), park,
                  _whole(environ, PARK_SLOTS_FLAG, DEFAULT_PARK_SLOTS, 1, MAX_PARK_SLOTS),
                  _whole(environ, MAX_PARK_FLAG, DEFAULT_MAX_PARK_S, 1, 3600), scope)


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
        merged_config(environ)
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


def valid_step(prompt, computed, chunk):
    """Whether a prefill step of `chunk` tokens that starts at `computed` is a step of a plan for `prompt` tokens (design R3): a step starts on
    a 2,048-token boundary (0 or a whole number of chunks: a cold start, a prefix hit's Q, or the end of an earlier step), and either takes
    the whole rest of the prompt from a start at or below P - 2,048 (the drafter window [P - 2,048, P) is then this step's own chunks), or
    ends on a boundary at or below S_last in a prompt of at least two chunks (so the final step still contains the window)."""
    for name, value in (('prompt', prompt), ('computed', computed), ('chunk', chunk)):
        if type(value) is not int:
            return False
    if prompt < 1 or not 0 <= computed < prompt or chunk < 1 or computed % CHUNK:
        return False
    end = computed + chunk
    if end > prompt:
        return False
    if end == prompt:
        return computed <= max(prompt - CHUNK, 0)
    return prompt >= 2 * CHUNK and end % CHUNK == 0 and end <= final_start(prompt)


def plan_from(prompt, start, *, decoding=True, step=DEFAULT_STEP, solo=DEFAULT_SOLO):
    """The (start, end) of every step of an uninterrupted prefill that resumes at `start` (a prefix hit's Q: 0 or a boundary at or below
    S_last); plan() is plan_from(prompt, 0). The tests' oracle for a hit's steps."""
    if type(start) is not int or not 0 <= start < prompt or start % CHUNK or start > max(final_start(prompt), 0):
        raise ValueError('a plan resumes on a %d-token boundary at or below S_last=%d, got start=%r for prompt=%r'
                         % (CHUNK, final_start(prompt), start, prompt))
    steps, computed = [], start
    while computed < prompt:
        budget = step_budget(prompt, computed, decoding=decoding, step=step, solo=solo)
        steps.append((computed, computed + budget))
        computed += budget
    return steps


# THE DEADLINE GOVERNOR (design section 11). The timing model docs/lever-n-tp4-timing-model.py fits a chunk of 2,048 tokens at context p
# to 420 ms + 2.5 ms per 1,000 tokens of p (T3/T4), 40 ms of transition per step and 2.5 s for the engine build. The seeds are NOT trusted at
# 254k (never measured there): StepTimes scales them by an EWMA of what the route measures.
SEED_BASE_MS = 420.0
SEED_PER_1K_MS = 2.5
TRANSITION_MS = 40.0
BUILD_MS = 2500.0
SCALE_MIN, SCALE_MAX = 0.25, 4.0


class StepTimes(object):
    """Device ms per 2,048-token chunk at a context, from the seeds corrected by what was measured: predict(context) = scale x
    (SEED_BASE_MS + SEED_PER_1K_MS x context / 1000); observe() moves `scale` by an EWMA of measured over seed."""

    def __init__(self):
        self.scale = 1.0
        self.observed = 0

    @staticmethod
    def seed(context):
        return SEED_BASE_MS + SEED_PER_1K_MS * context / 1000.0

    def predict(self, context):
        return self.scale * self.seed(context)

    def observe(self, start, tokens, ms):
        """One route step [start, start + tokens) that took `ms`: its per-chunk time against the seed at its start."""
        if ms <= 0 or tokens < 1:
            return
        chunks = max(1, -(-tokens // CHUNK))
        ratio = (ms / chunks) / self.seed(start)
        weight = 0.2     # also for the first sample: one odd step (a restore, a park) must not move the scale to the clamp
        self.scale = min(SCALE_MAX, max(SCALE_MIN, (1.0 - weight) * self.scale + weight * ratio))
        self.observed += 1

    def remaining_ms(self, prompt, computed, step=DEFAULT_STEP):
        """Device ms the rest of a prefill of `prompt` tokens still needs from `computed`: every chunk before the final step at its
        context, the final step's chunks (the last full chunk and the tail), and a transition per step of `step` tokens."""
        if prompt <= computed:
            return 0.0
        s_last = max(final_start(prompt), 0)
        start = min(computed, s_last)
        total = sum(self.predict(position) for position in range(start, s_last, CHUNK))
        steps = -(-(s_last - start) // step) if s_last > start else 0
        final_tokens = prompt - start if start >= s_last else prompt - s_last
        total += max(1, -(-final_tokens // CHUNK)) * self.predict(s_last)
        return total + TRANSITION_MS * (steps + 1)


def effective_share(base, target_s, pending, now, build_ms=BUILD_MS):
    """f_eff = clamp(max(f_base, max_j (sum_{i<=j} D_i + B) / (T* - (now - arrival_j))), f_base, 1): the prefill's share of the wall time
    that still lets every pending prefill j (in service order, the in-flight one first) finish by T* seconds after its arrival.
    `pending` is [(arrival_s, remaining_device_ms)]. target 0 turns the governor off; a pending prefill already past T* gets 1.0 (nothing
    is owed to the decoders: the prefill finishes as fast as it can, today's stall, chunked). Never below `base`."""
    if not target_s or base >= 1.0 or not pending:
        return base
    needed, cumulative = base, 0.0
    for arrival, remaining in pending:
        cumulative += remaining
        slack = target_s - (now - arrival)
        if slack <= 0:
            return 1.0
        needed = max(needed, (cumulative + build_ms) / (slack * 1000.0))
        if needed >= 1.0:
            return 1.0
    return min(1.0, max(base, needed))


class OffSwitch(object):
    """The Lever N kill switch (design section 10): a flag file next to the prefix kill switch (OFF_PATH), polled at most once per OFF_POLL_S
    and latched for the life of the process. Once engaged no new prompt is split (a whole-prompt cap) and the short lane is closed; prefills in
    flight finish through the route. `poll()` is True exactly once, on the poll that first sees the file."""

    def __init__(self, path=None, poll_s=OFF_POLL_S, clock=time.monotonic, exists=os.path.exists):
        self.path = (os.environ.get(OFF_PATH_ENV, OFF_PATH) if path is None else path) or None
        self.poll_s, self.clock, self.exists = float(poll_s), clock, exists
        self.engaged = False
        self._last = None

    def poll(self):
        if self.engaged or not self.path:
            return False
        now = self.clock()
        if self._last is not None and now - self._last < self.poll_s:
            return False
        self._last = now
        try:
            present = bool(self.exists(self.path))
        except Exception:
            present = False
        if present:
            self.engaged = True
        return present


Decision = namedtuple('Decision', 'kind reason prev_kind prev_ms owed_ms owed_rounds')


class Alternator(object):
    """The time-share (or static-R) alternation between prefill steps and decode steps.

    begin(decodes, pending) is called at the start of every schedule() and returns a Decision whose kind is 'decode'
    (run a decode-only step now: a partial prefill and the waiting queues are hidden) or 'prefill' (let the plugin
    scheduler run its default step). end(kind) is called with what the step turned out to be, so the next begin can
    charge the interval to it. A kind is 'prefill' only for a step that scheduled prefill tokens."""

    def __init__(self, cfg, clock=time.monotonic, merged=None, wall=time.time):
        self.cfg, self.clock = cfg, clock
        self.merged = merged
        self.wall = wall
        # The deadline governor's input, set by the scheduler before every begin(): [(arrival_s, remaining_device_ms)] in service order.
        self.pending_hint = []
        self.share = cfg.share
        self.reset()
        self.round_ms = INITIAL_ROUND_MS

    def reset(self):
        self.last_at = None
        self.last_kind = None
        self.last_short = False
        self.last_decodes = 0
        self.owed_ms = 0.0
        self.owed_rounds = 0
        self.since_prefill = RMIN

    def _account(self, kind, decodes, dt_ms):
        cfg = self.cfg
        if kind == 'prefill':
            if decodes > 0:
                self.since_prefill = 0
                if self.last_short:
                    # Admission v2: a SHORT prefill (its remaining tokens fit QWEN_FAST_LEVERN_SHORT_TOKENS) is paced at R = 1 whatever the share is:
                    # one decode round after each of its steps, so the decoders' gap stays about one step and the short's own latency is the
                    # cheapest one that still serves them (design section 9). The deadline governor never governs it.
                    if cfg.rounds is not None:
                        self.owed_rounds = max(self.owed_rounds, 1)
                    else:
                        self.owed_ms = min(cfg.max_rounds * self.round_ms, self.owed_ms + self.round_ms)
                elif cfg.rounds is not None:
                    self.owed_rounds = cfg.rounds
                elif self.share < 1.0:
                    cap = cfg.max_rounds * self.round_ms
                    self.owed_ms = min(cap, self.owed_ms + dt_ms * (1.0 - self.share) / self.share)
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
        if self.share >= 1.0:
            return False
        return self.owed_ms > 0.0 or self.since_prefill < RMIN

    def govern(self):
        """Recompute the share from the pending prefills the scheduler handed in (pending_hint); the base share when the governor is off."""
        merged = self.merged
        if merged is None or not merged.ttft_s or self.cfg.rounds is not None:
            self.share = self.cfg.share
        else:
            self.share = effective_share(self.cfg.share, merged.ttft_s, self.pending_hint, self.wall())
        return self.share

    def begin(self, decodes, pending):
        now = self.clock()
        self.govern()
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
            self.last_short = False
            self.owed_ms, self.owed_rounds, self.since_prefill = 0.0, 0, RMIN
            return Decision('prefill', 'none-pending', prev_kind, prev_ms, 0.0, 0)
        if decodes and self.owes():
            return Decision('decode', 'owed', prev_kind, prev_ms, self.owed_ms, self.owed_rounds)
        return Decision('prefill', 'idle' if not decodes else 'paid', prev_kind, prev_ms, self.owed_ms, self.owed_rounds)

    def end(self, kind, short=False):
        if kind not in ('prefill', 'decode'):
            raise ValueError('a step is prefill or decode, got %r' % (kind,))
        self.last_kind = kind
        self.last_short = bool(short) and kind == 'prefill'
