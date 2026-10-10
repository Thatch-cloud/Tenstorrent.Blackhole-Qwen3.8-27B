"""The pure half of the sub-device overlap harnesses (H1 on one card, H2 on four): no ttnn, no torch, no device.

The question both harnesses answer: can the DFlash2 drafter run CONCURRENTLY with the target verify on one Blackhole chip, the target on a
sub-device of 80 cores fed by command queue 0 and the drafter on a sub-device of 30 cores fed by command queue 1 (tt-metal sub-devices, one
sub-device manager, two command queues, one trace per queue)? If the two traces overlap, a verify block and the drafter pass cost
max(V, D) instead of V + D; if the queues serialise, the sub-device split only costs the target its cores.

This module holds everything that can be decided without a card, so that CPU tests pin it:

  plan_split      the two disjoint core rectangles from the chip's reported compute grid (80 / 30 of 110 by default, whole columns)
  build_recipe    the op list of a target-like (about 400 ops) or drafter-like (about 900 ops) trace: matmuls, activations, products and
                  residual adds shaped like a 64-row verify slice, every source defined before it is used
  summarize       the statistics of a list of wall-clock samples
  verdict         the PASS rule: concurrent wall <= PASS_RATIO x max(solo), outputs identical, no hang, no error
  Watchdog        the per-call deadline (a poller thread, plus a faulthandler backstop for a call that holds the GIL)
  parse_args      the harness arguments, validated

Nothing here names a card, a host or a path.
"""

import argparse
import faulthandler
import os
import statistics
import sys
import threading
import time
from contextlib import contextmanager

KIND = 'subdev-h1'
VERDICT_TAG = 'SUBDEV_H1'
PASS_RATIO = 1.10
TARGET_CORES = 80
DRAFTER_CORES = 30
TARGET_OPS = 400
DRAFTER_OPS = 900
TARGET_ROWS = 64
DRAFTER_ROWS = 32
# A concurrent wall this close to the SUM of the solo walls is "serialised": the queues did not overlap at all.
SERIALISED_FRACTION = 0.90
ARMS = ('t_solo', 'd_solo', 'concurrent', 'chained', 'one_queue')
EXIT_CODES = dict(PASS=0, UNTIMED_PASS=0, FAIL=1, NOT_MEASURED=2, HANG=3)


def say(line):
    """print with a flush: stdout is a pipe under `docker run | tee`, and a hang ends the process with os._exit, which flushes nothing."""
    print(line, flush=True)


class PlanError(ValueError):
    """The requested split or recipe cannot be built."""


# ---------------------------------------------------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------------------------------------------------

def plan_split(grid_x, grid_y, target_cores=TARGET_CORES, drafter_cores=DRAFTER_CORES):
    """Two disjoint rectangles of whole columns from a (grid_x, grid_y) compute grid: the target's on the left, the drafter's right of it.

    80 / 30 of the 11 x 10 grid is 8 columns and 3 columns exactly. When the grid is narrower than the request (a chip that reports a
    smaller grid with two command queues) both are scaled down in proportion and the plan says `exact: False`. Returns a dict with
    `target` and `drafter` as (x0, y0, x1, y1) inclusive rectangles, their grids (columns, rows), core counts and the idle cores."""
    grid_x, grid_y = int(grid_x), int(grid_y)
    if grid_x < 2 or grid_y < 1:
        raise PlanError('a compute grid of %d x %d cannot hold two sub-devices' % (grid_x, grid_y))
    if target_cores < 1 or drafter_cores < 1:
        raise PlanError('both sub-devices need at least one core (target %d, drafter %d)' % (target_cores, drafter_cores))
    want_t = max(1, int(round(target_cores / float(grid_y))))
    want_d = max(1, int(round(drafter_cores / float(grid_y))))
    if want_t + want_d > grid_x:
        want_d = max(1, int(round(grid_x * want_d / float(want_t + want_d))))
        want_t = grid_x - want_d
    target = (0, 0, want_t - 1, grid_y - 1)
    drafter = (want_t, 0, want_t + want_d - 1, grid_y - 1)
    return dict(grid=[grid_x, grid_y], target=list(target), drafter=list(drafter),
                target_grid=[want_t, grid_y], drafter_grid=[want_d, grid_y],
                target_cores=want_t * grid_y, drafter_cores=want_d * grid_y,
                idle_cores=grid_x * grid_y - (want_t + want_d) * grid_y,
                exact=(want_t * grid_y == target_cores and want_d * grid_y == drafter_cores))


def rectangles_disjoint(first, second):
    """Whether two inclusive (x0, y0, x1, y1) rectangles share no core."""
    return first[2] < second[0] or second[2] < first[0] or first[3] < second[1] or second[3] < first[1]


# ---------------------------------------------------------------------------------------------------------------------
# The recipes
# ---------------------------------------------------------------------------------------------------------------------

# name -> (rows are the activation's, hidden width H, intermediate width I, attention width Q).
# target: a 64-row verify slice on one chip of a TP4 mesh (hidden 5120, MLP shard 4352, attention projection shard 2560).
# drafter: a small block the quad draft runs 32 rows at a time (launch-bound: many small ops).
PROFILES = dict(
    target=dict(hidden=5120, inter=4352, attn=2560, norm_ops=False),
    drafter=dict(hidden=2560, inter=3072, attn=2048, norm_ops=True),
)

# Weights are N(0, gain^2 / K): GATE_GAIN for the gate, up and qkv projections and OUT_GAIN for the down and output projections that feed the residual, and
# the drafter's scale tensor is uniform in +-SCALE_AMPLITUDE. A residual stack of silu products is unstable (the quadratic term takes over once the hidden state
# is a few units: 0.1 for OUT_GAIN overflows bfloat16 within 900 drafter ops, 0.05 stays under 25), so these keep 400 target and 900 drafter ops finite, 5 times
# below the drafter's overflow (a float32 simulation of the recipe with bfloat16 rounding after every op: the last output has rms about 1 for the target and 2 for
# the drafter, largest value 5 and 16). A harness refuses a run whose solo references are not finite.
WEIGHT_GAIN = 0.5
OUT_GAIN = 0.02
SCALE_AMPLITUDE = 0.02

# One layer pair's ops: (kind, destination, first source, second source). 'h' is the running hidden state: the pair's input is
# h and its output replaces it. MLP: gate and up projections, silu, product, down projection, residual. Attention-like: qkv
# projection, silu, output projection, residual. The drafter profile adds a scale product and a second residual (its norms).
MLP = (('mm', 'g', 'h', 'w_gate'), ('mm', 'u', 'h', 'w_up'), ('silu', 's', 'g', None), ('mul', 'p', 's', 'u'),
       ('mm', 'd', 'p', 'w_down'), ('add', 'h', 'd', 'h'))
ATTN = (('mm', 'q', 'h', 'w_qkv'), ('silu', 'a', 'q', None), ('mm', 'o', 'a', 'w_o'), ('add', 'h', 'o', 'h'))
NORM = (('mul', 'n', 'h', 'scale'), ('add', 'h', 'n', 'h'))


def weight_shapes(profile):
    """{weight name: (K, N)} of a profile, row-major (activation @ weight)."""
    spec = PROFILES[profile]
    hidden, inter, attn = spec['hidden'], spec['inter'], spec['attn']
    return dict(w_gate=(hidden, inter), w_up=(hidden, inter), w_down=(inter, hidden), w_qkv=(hidden, attn), w_o=(attn, hidden))


def layer_pattern(profile):
    """The op tuples of one layer pair of a profile."""
    pattern = list(MLP) + list(ATTN)
    if PROFILES[profile]['norm_ops']:
        pattern += list(NORM)
    return pattern


def build_recipe(profile, ops):
    """The first `ops` ops of `profile`'s layer pattern repeated, as a list of (kind, destination, source a, source b) with unique
    destination names ('o0', 'o1', ...) except the running hidden state, which is renamed per layer so that every op output is a
    distinct tensor (a trace keeps them all alive; nothing is freed during capture).

    Returns a dict: ops (the list), inputs (the names the recipe expects from outside: 'h0' the input activation, the weights and
    'scale'), output (the last op's destination), checkpoints_every.  Every source is defined before it is used."""
    if profile not in PROFILES:
        raise PlanError('unknown profile %r (one of %s)' % (profile, ', '.join(sorted(PROFILES))))
    if ops < 1:
        raise PlanError('a recipe needs at least one op, got %d' % ops)
    pattern = layer_pattern(profile)
    external = set(weight_shapes(profile)) | {'scale'}
    listing, hidden_name, written = [], 'h0', 0
    while len(listing) < ops:
        local = {}
        for kind, dest, first, second in pattern:
            if len(listing) >= ops:
                break
            def source(name):
                if name is None:
                    return None
                if name == 'h':
                    return hidden_name
                if name in external:
                    return name
                return local[name]
            first_name, second_name = source(first), source(second)
            out = 'o%d' % written
            written += 1
            local[dest] = out
            if dest == 'h':
                hidden_name = out
            listing.append((kind, out, first_name, second_name))
    defined = {'h0'} | external
    for kind, dest, first, second in listing:
        for name in (first, second):
            if name is not None and name not in defined:
                raise PlanError('recipe bug: %s reads %s before it is defined' % (dest, name))
        defined.add(dest)
    return dict(ops=listing, inputs=sorted({'h0'} | external), output=listing[-1][1], count=len(listing),
                matmuls=sum(1 for op in listing if op[0] == 'mm'))


def checkpoint_names(recipe, every):
    """The op outputs a harness reads back and compares: every `every`-th, and the last."""
    names = [op[1] for index, op in enumerate(recipe['ops']) if every > 0 and index % every == 0]
    last = recipe['ops'][-1][1]
    if last not in names:
        names.append(last)
    return names


# ---------------------------------------------------------------------------------------------------------------------
# Statistics and the verdict
# ---------------------------------------------------------------------------------------------------------------------

def summarize(samples_ns):
    """{n, median_ms, p10_ms, p90_ms, min_ms, mean_ms} of nanosecond samples (empty: n = 0)."""
    if not samples_ns:
        return dict(n=0)
    ordered = sorted(samples_ns)

    def at(fraction):
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] / 1e6

    return dict(n=len(ordered), median_ms=round(statistics.median(ordered) / 1e6, 4), p10_ms=round(at(0.1), 4),
                p90_ms=round(at(0.9), 4), min_ms=round(ordered[0] / 1e6, 4), mean_ms=round(sum(ordered) / len(ordered) / 1e6, 4))


def _median(arms, name):
    return arms.get(name, {}).get('median_ms')


def ratios(arms):
    """The derived numbers from the arm summaries: ratio = concurrent / max(solo), overlap = how much of the shorter solo wall the
    concurrent run hid (1.0 = all of it), chained_over_sum = chained / (t + d) (the event's overhead), one_queue_over_sum."""
    t, d, c = _median(arms, 't_solo'), _median(arms, 'd_solo'), _median(arms, 'concurrent')
    out = {}
    if t and d:
        out['sum_ms'] = round(t + d, 4)
        out['max_solo_ms'] = round(max(t, d), 4)
        if c:
            out['ratio'] = round(c / max(t, d), 4)
            out['overlap'] = round((t + d - c) / min(t, d), 4)
            out['concurrent_over_sum'] = round(c / (t + d), 4)
            low = arms['concurrent'].get('min_ms')
            tmin, dmin = arms['t_solo'].get('min_ms'), arms['d_solo'].get('min_ms')
            if low and tmin and dmin:
                out['ratio_min'] = round(low / max(tmin, dmin), 4)
        for name in ('chained', 'one_queue'):
            value = _median(arms, name)
            if value:
                out[name + '_over_sum'] = round(value / (t + d), 4)
    return out


def verdict(arms, exactness, timing=True, error=None, pass_ratio=PASS_RATIO):
    """(verdict text, evidence dict). Texts:
      NOT-MEASURED   an error before the arms ran (the device did not open, the manager could not be built)
      FAIL-BYTES     any compared tensor differed from its solo reference (the exactness half of the rule, whatever the timing)
      INCOMPLETE     a required arm has no samples
      UNTIMED-PASS   timing is off (the watcher pass): bytes identical, nothing hung, no ratio is judged
      PASS           concurrent <= pass_ratio x max(solo) and the bytes are identical
      FAIL-SERIALISED   the concurrent wall is within SERIALISED_FRACTION of t + d: the queues did not overlap
      FAIL-PARTIAL   in between: some overlap, more than pass_ratio x the longer solo
    `exactness` = dict(compared=<tensors compared>, mismatched=<tensors that differed>, ...)."""
    evidence = dict(ratios(arms))
    evidence.update(compared=exactness.get('compared', 0), mismatched=exactness.get('mismatched', 0), timing=bool(timing),
                    pass_ratio=pass_ratio)
    if error:
        return 'NOT-MEASURED', evidence
    if exactness.get('mismatched', 0):
        return 'FAIL-BYTES', evidence
    if not exactness.get('compared', 0):
        return 'INCOMPLETE', evidence
    if not timing:
        return 'UNTIMED-PASS', evidence
    if 'ratio' not in evidence:
        return 'INCOMPLETE', evidence
    return classify(evidence['ratio'], evidence['concurrent_over_sum'], pass_ratio), evidence


def classify(ratio, over_sum, pass_ratio=PASS_RATIO):
    """PASS when the concurrent wall is within pass_ratio of the longer solo wall; FAIL-SERIALISED when it is within SERIALISED_FRACTION of the
    sum of the two solo walls (the queues did not overlap); FAIL-PARTIAL in between."""
    if ratio <= pass_ratio:
        return 'PASS'
    if over_sum >= SERIALISED_FRACTION:
        return 'FAIL-SERIALISED'
    return 'FAIL-PARTIAL'


def verdict_h2(arms, exactness, timing=True, error=None, pass_ratio=PASS_RATIO, separate_run=False):
    """(verdict text, evidence) of the four-card collective probe H2. `arms` has t_solo, d_solo, shared (both gathers concurrent on the SAME fabric
    link: num_links 1 on each), optionally shared2 (num_links 2 on each; informational) and separate (the drafter's gather on link 1, which needs the
    link-offset graft; `separate_run` says it was attempted). Texts:
      NOT-MEASURED / FAIL-BYTES / INCOMPLETE / UNTIMED-PASS   as verdict()
      PASS                  the shared-link concurrent wall is within pass_ratio of the longer solo wall: no graft needed
      PASS-SEPARATE-LINKS   the shared link does not overlap but the separate links do: the link-offset graft is required
      FAIL-SERIALISED / FAIL-PARTIAL   neither overlaps (the separate arm's class when it ran, else the shared arm's)"""
    evidence = dict(compared=exactness.get('compared', 0), mismatched=exactness.get('mismatched', 0), timing=bool(timing), pass_ratio=pass_ratio)
    t, d = _median(arms, 't_solo'), _median(arms, 'd_solo')
    if t and d:
        evidence.update(sum_ms=round(t + d, 4), max_solo_ms=round(max(t, d), 4))
        for name in ('shared', 'shared2', 'separate'):
            value = _median(arms, name)
            if value:
                evidence[name + '_ms'] = value
                evidence[name + '_ratio'] = round(value / max(t, d), 4)
                evidence[name + '_over_sum'] = round(value / (t + d), 4)
    if error:
        return 'NOT-MEASURED', evidence
    if exactness.get('mismatched', 0):
        return 'FAIL-BYTES', evidence
    if not exactness.get('compared', 0):
        return 'INCOMPLETE', evidence
    if not timing:
        return 'UNTIMED-PASS', evidence
    if 'shared_ratio' not in evidence:
        return 'INCOMPLETE', evidence
    shared = classify(evidence['shared_ratio'], evidence['shared_over_sum'], pass_ratio)
    if shared == 'PASS':
        return 'PASS', evidence
    if separate_run and 'separate_ratio' in evidence:
        separate = classify(evidence['separate_ratio'], evidence['separate_over_sum'], pass_ratio)
        if separate == 'PASS':
            return 'PASS-SEPARATE-LINKS', evidence
        return separate, evidence
    return shared, evidence


def exit_status(text):
    """The process exit status of a verdict text: 0 PASS, UNTIMED-PASS and PASS-SEPARATE-LINKS, 1 any FAIL or INCOMPLETE, 2 NOT-MEASURED, 3 HANG."""
    if text in ('PASS', 'UNTIMED-PASS', 'PASS-SEPARATE-LINKS'):
        return EXIT_CODES['PASS']
    if text == 'NOT-MEASURED':
        return EXIT_CODES['NOT_MEASURED']
    if text == 'HANG':
        return EXIT_CODES['HANG']
    return EXIT_CODES['FAIL']


def verdict_line(text, evidence, plan=None, manager=None, extra=None):
    """The single machine-readable line: SUBDEV_H1 verdict=... ratio=... conc_ms=... (numbers that are missing print as '-')."""
    def number(key, default='-'):
        value = evidence.get(key)
        return default if value is None else value

    parts = ['%s verdict=%s' % (VERDICT_TAG, text), 'ratio=%s' % number('ratio'), 'conc_ms=%s' % number('concurrent_ms'),
             't_ms=%s' % number('t_solo_ms'), 'd_ms=%s' % number('d_solo_ms'), 'sum_ms=%s' % number('sum_ms'),
             'chained_ms=%s' % number('chained_ms'), 'one_queue_ms=%s' % number('one_queue_ms'),
             'bytes=%s' % ('identical' if not evidence.get('mismatched') and evidence.get('compared') else
                           'DIFFER(%s/%s)' % (evidence.get('mismatched'), evidence.get('compared'))),
             'timing=%d' % int(bool(evidence.get('timing')))]
    if plan:
        parts.append('split=%d/%d' % (plan['target_cores'], plan['drafter_cores']))
        parts.append('grid=%dx%d' % tuple(plan['grid']))
    if manager:
        parts.append('load_ms=%s' % manager.get('load_ms_median', '-'))
        parts.append('clear_ms=%s' % manager.get('clear_ms_median', '-'))
    for key, value in sorted((extra or {}).items()):
        parts.append('%s=%s' % (key, value))
    return ' '.join(parts)


def flatten_evidence(arms, evidence):
    """The evidence dict with each arm's median under '<arm>_ms' so verdict_line can print them."""
    flat = dict(evidence)
    for name in ARMS:
        value = _median(arms, name)
        if value is not None:
            flat[name + '_ms'] = value
    return flat


# ---------------------------------------------------------------------------------------------------------------------
# The watchdog
# ---------------------------------------------------------------------------------------------------------------------

class Watchdog(object):
    """A per-call deadline. `span(label, seconds)` arms it; if the call has not returned in time the poller thread prints
    '<tag> verdict=HANG stage=<label>', calls `on_fire(label)` (the partial report) and exits 3. A blocking ttnn call (a read, a synchronise on
    a hung program) may hold the GIL so the poller never runs; each span therefore also arms a faulthandler backstop, a C thread that
    dumps every stack ('Timeout (h:mm:ss)!') and exits 1 at the budget plus `grace`, which the harness script reads as a hang as well.
    Spans nest: leaving an inner span re-arms the outer one for its remaining time."""

    def __init__(self, tag=VERDICT_TAG, on_fire=None, grace=60.0, backstop=True, exit_fn=None, clock=time.monotonic, stream=None):
        self.tag, self.on_fire, self.grace = tag, on_fire, grace
        self.backstop = bool(backstop)
        self.exit_fn = exit_fn if exit_fn is not None else os._exit
        self.clock, self.stream = clock, stream
        self.stack = []
        self.fired = None
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.poll, name='subdev-watchdog', daemon=True)
            self.thread.start()
        return self

    @contextmanager
    def span(self, label, seconds):
        if not seconds:
            yield
            return
        with self.lock:
            self.stack.append((label, self.clock() + seconds, seconds))
        self._backstop(seconds)
        try:
            yield
        finally:
            with self.lock:
                self.stack.pop()
                outer = self.stack[-1] if self.stack else None
            if outer is None:
                self._cancel()
            else:
                self._backstop(max(1.0, outer[1] - self.clock()))

    def current(self):
        with self.lock:
            return self.stack[-1][0] if self.stack else None

    def _backstop(self, seconds):
        if self.backstop:
            try:
                faulthandler.dump_traceback_later(seconds + self.grace, exit=True, file=self.stream or sys.stdout)
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

    def _cancel(self):
        if self.backstop:
            try:
                faulthandler.cancel_dump_traceback_later()
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

    def check(self):
        """One poll: fire when the innermost armed span is past its deadline. Returns whether it fired."""
        with self.lock:
            top = self.stack[-1] if self.stack else None
        if top is None or self.clock() < top[1]:
            return False
        label, _, seconds = top
        self.fired = label
        stream = self.stream or sys.stdout
        try:
            stream.write('%s verdict=HANG stage=%s after_s=%s\n' % (self.tag, label, seconds))
            stream.flush()
            if self.on_fire is not None:
                self.on_fire(label)
        finally:
            self.exit_fn(3)
        return True

    def poll(self):
        while not self.check():
            time.sleep(1.0)


class Heartbeat(object):
    """The stall watch: a daemon thread that prints '<tag> alive stage=<stage> in_stage_s=<n>' every `period` seconds, so a window driver
    reading the log can tell a slow stage from a stalled one even before the watchdog's budget runs out."""

    def __init__(self, tag, period=30.0, clock=time.monotonic, stream=None, sleep=time.sleep):
        self.tag, self.period, self.clock, self.stream, self.sleep = tag, period, clock, stream, sleep
        self.stage, self.since = 'start', clock()
        self.stopped = False
        self.thread = None

    def set(self, stage):
        self.stage, self.since = stage, self.clock()

    def line(self):
        return '%s alive stage=%s in_stage_s=%d' % (self.tag, self.stage, int(self.clock() - self.since))

    def beat(self):
        stream = self.stream or sys.stdout
        stream.write(self.line() + '\n')
        stream.flush()

    def run(self):
        while not self.stopped:
            self.sleep(self.period)
            if not self.stopped:
                self.beat()

    def start(self):
        if self.thread is None and self.period:
            self.thread = threading.Thread(target=self.run, name='subdev-heartbeat', daemon=True)
            self.thread.start()
        return self

    def stop(self):
        self.stopped = True


# ---------------------------------------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description='Sub-device overlap harness H1: target trace on queue 0 and drafter trace on queue 1, one card.')
    parser.add_argument('--out', required=True, help='the JSON report')
    parser.add_argument('--target-cores', type=int, default=TARGET_CORES)
    parser.add_argument('--drafter-cores', type=int, default=DRAFTER_CORES)
    parser.add_argument('--target-ops', type=int, default=TARGET_OPS)
    parser.add_argument('--drafter-ops', type=int, default=DRAFTER_OPS)
    parser.add_argument('--rows', type=int, default=TARGET_ROWS, help='the target activation rows (a verify slice: 64)')
    parser.add_argument('--drafter-rows', type=int, default=DRAFTER_ROWS)
    parser.add_argument('--rounds', type=int, default=30, help='timed rounds of every arm')
    parser.add_argument('--warmup', type=int, default=5, help='untimed rounds before them')
    parser.add_argument('--check-every', type=int, default=8, help='compare every n-th op output (and the last) byte for byte')
    parser.add_argument('--stress', type=int, default=20, help='extra concurrent replays before the final byte comparison')
    parser.add_argument('--manager-cycles', type=int, default=3, help='load + clear cycles timed before the run')
    parser.add_argument('--pass-ratio', type=float, default=PASS_RATIO)
    parser.add_argument('--no-timing', action='store_true',
                        help='the watcher pass: few rounds, bytes and hangs only (the watcher distorts every wall); verdict UNTIMED-PASS')
    parser.add_argument('--timing', dest='no_timing', action='store_false', help='time the arms even under the watcher (the last of --timing / --no-timing wins)')
    parser.add_argument('--watchdog-s', type=float, default=120.0, help='seconds per device call before exit 3 (0 off)')
    parser.add_argument('--compile-watchdog-s', type=float, default=900.0, help='the same for a first-run compile / capture phase')
    parser.add_argument('--trace-region-bytes', type=int, default=134217728)
    parser.add_argument('--dispatch', choices=('default', 'worker', 'ethernet'), default='default')
    parser.add_argument('--seed', type=int, default=0)
    return parser


def problems_of(options):
    """Why the options cannot run, [] when they can."""
    problems = []
    for name in ('rows', 'drafter_rows'):
        value = getattr(options, name)
        if value < 32 or value > 256 or value % 32:
            problems.append('--%s must be a multiple of 32 in 32..256, got %d' % (name.replace('_', '-'), value))
    for name in ('target_ops', 'drafter_ops'):
        if getattr(options, name) < 4:
            problems.append('--%s must be at least 4, got %d' % (name.replace('_', '-'), getattr(options, name)))
    if options.target_cores < 1 or options.drafter_cores < 1:
        problems.append('both sub-devices need at least one core')
    if options.rounds < 3 or options.warmup < 0 or options.stress < 0 or options.manager_cycles < 0:
        problems.append('--rounds >= 3, --warmup, --stress and --manager-cycles >= 0')
    if options.check_every < 1:
        problems.append('--check-every must be at least 1')
    if options.pass_ratio <= 1.0:
        problems.append('--pass-ratio must exceed 1.0 (a ratio of concurrent to the longer solo wall)')
    if options.watchdog_s < 0 or options.compile_watchdog_s < 0:
        problems.append('the watchdog budgets are not negative')
    return problems


def parse_args(argv=None):
    """(options, problems): the parsed arguments and why they cannot run (an empty list when they can). --no-timing shortens the rounds."""
    options = build_parser().parse_args(argv)
    if options.no_timing:
        options.rounds = min(options.rounds, 4)
        options.warmup = min(options.warmup, 1)
        options.stress = min(options.stress, 6)
    return options, problems_of(options)

