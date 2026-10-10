"""Op-fusion programme, host-gap package WPH, lever 4: the instrument for ONE unprofiled job (host only; nothing here changes a byte the device sees).

Two flags, both default off:

  QWEN_FAST_TP4_HOSTGAP_LOG=1         (the stage 0 flag of tp4/hostgap; this module adds NEW lines to it, no existing line changes) the spans below, and one
                                      `[PACKED-HOSTGAP-ROUND]` line a step.
  QWEN_FAST_TP4_HOSTGAP_PROBE=1       (needs HOSTGAP_LOG) every PROBE_EVERY-th 8-live round (two quads in flight, QWEN_FAST_TP4_HOSTGAP_PROBE_EVERY, default 16) the
                                      coordinator does one TIMED synchronize right after the two quad launches (dflash_packed_proposal_coordinator, next to
                                      ledger.mark('quads1')): the wait is the quads' remaining device time, so 2Q is measured directly instead of inferred. The round it
                                      runs in is a PROBE ROUND: the window then runs after the quads instead of under them, so its time is not a round time.
                                      The probe round is named on a `[PACKED-HOSTGAP-PROBE]` line and by `probe=1` on its round line, and the paired timing readers drop it
                                      (w2ln_timing_compare.timed_rounds, hostgap_read).

WHAT IS MEASURED. The 8-live round is 150 ms with the device idle 16.5-18.6 ms across 7 host turnarounds, and 22% of the rounds are slow in the same places (a quad launch
of 5.8-6.4 ms instead of about 2, a write 2-5 times slower). Wall time alone cannot say why. Each SPAN (a block's pre-stage, the window, the fence, the quad launches, the
collect, a verify's stage and readback, each commit, the early draft, the step) is written with the host's own account of it:
  wall_ms / cpu_ms          this thread's elapsed and CPU time: cpu much less than wall is waiting or being descheduled, cpu about wall is Python work (or GC);
  nvcsw / nivcsw            this thread's voluntary and involuntary context switches (getrusage RUSAGE_THREAD): an involuntary switch is the scheduler taking the CPU away
                            (CI runners, the dispatch threads, the serving container's own), a voluntary one is blocking;
  minflt / majflt           page faults;
  gc_n / gc_ms / gc_max_ms  every garbage collection in the span (not only those over 5 ms, which stage 0 already logs);
  per host call the span made, count / total ms / longest single ms (/ calls over 1 ms): copy (copy_host_to_device_tensor: a copy that BLOCKED shows as a long single one while
  cpu stays low: the command queue was full), up (from_torch), tnb (execute_trace, non-blocking: the enqueue), tb (execute_trace blocking), fence (synchronize_device), read
  (to_torch), d2h (copy_device_to_host_tensor).
The calls are timed by thin wrappers installed on the ttnn namespace once, at the attach, with the flag on (`install`); they call the real function with the caller's own
arguments and return its result, and count nothing else. Off, nothing is installed and nothing here is imported (verify_prestage.hostgap_span imports this module only with
the flag).

A reader of the lines is hostgap_read.py (tables, probe rounds dropped, slow rounds classified by cause).
"""

import contextlib
import gc
import os
import time

try:
    import resource
except ImportError:         # not a POSIX host: no context-switch counts
    resource = None

LOG_FLAG = 'QWEN_FAST_TP4_HOSTGAP_LOG'
PROBE_FLAG = 'QWEN_FAST_TP4_HOSTGAP_PROBE'
PROBE_EVERY_FLAG = 'QWEN_FAST_TP4_HOSTGAP_PROBE_EVERY'
PROBE_EVERY = 16
ENGAGED_MARKER = '[PINDIAG] tp4 hostgap instrument engaged'
PROBE_ENGAGED_MARKER = '[PINDIAG] tp4 hostgap probe engaged'
ROUND_MARKER = '[PACKED-HOSTGAP-ROUND]'
SPAN_MARKER = '[PACKED-HOSTGAP-SPAN]'
PROBE_MARKER = '[PACKED-HOSTGAP-PROBE]'
LONG_CALL_S = 0.001
# attribute on the ttnn namespace -> the short name its figures carry
CALLS = (('copy_host_to_device_tensor', 'copy'), ('from_torch', 'up'), ('execute_trace', 'trace'), ('synchronize_device', 'fence'), ('to_torch', 'read'),
         ('copy_device_to_host_tensor', 'd2h'))
KINDS = ('copy', 'up', 'tnb', 'tb', 'fence', 'read', 'd2h')
WRAPPED_ATTRIBUTE = '_hostgap_wrapped'


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % name)
    return value == '1'


def log_enabled(environ=None):
    return _flag(LOG_FLAG, environ)


def probe_enabled(environ=None):
    """QWEN_FAST_TP4_HOSTGAP_PROBE=1 under QWEN_FAST_TP4_HOSTGAP_LOG=1 (alone it probes nothing)."""
    return _flag(PROBE_FLAG, environ) and log_enabled(environ)


def probe_every(environ=None):
    value = (os.environ if environ is None else environ).get(PROBE_EVERY_FLAG)
    if value is None:
        return PROBE_EVERY
    if not value.isdigit() or int(value) < 1:
        raise ValueError('%s must be a positive integer' % PROBE_EVERY_FLAG)
    return int(value)


def log_line(text):
    import verify_prestage

    verify_prestage.log_line(text)


class Meter:
    """The process's running totals: per call kind [count, wall s, cpu s, calls over 1 ms], garbage collections, the open spans' longest single calls, the step counter and
    the probe's state."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = {kind: [0, 0.0, 0.0, 0] for kind in KINDS}
        self.gc = dict(n=0, ms=0.0, max_ms=0.0, started=0.0, installed=False, callback=None)
        self.frames = []
        self.step = 0
        self.last = None            # the snapshot at the last step entry
        self.probed = False         # a probe ran in the interval that is open
        self.probes = 0
        self.first_enqueue = None   # perf_counter of the first non-blocking trace enqueue since the last mark
        self.installed = {}


METER = Meter()


def reset():
    """A new attach, and the tests: the totals, the open spans and the probe forgotten (the installed wrappers and the collection callback stay until `uninstall`)."""
    installed, callback, was = METER.installed, METER.gc['callback'], METER.gc['installed']
    METER.reset()
    METER.installed = installed
    METER.gc.update(callback=callback, installed=was)


# ---- the clocks -------------------------------------------------------------------------------------------------------------------

def usage():
    """(voluntary, involuntary context switches, minor faults, major faults) of this thread; zeros where the host cannot say."""
    if resource is None:
        return 0, 0, 0, 0
    try:
        used = resource.getrusage(getattr(resource, 'RUSAGE_THREAD', resource.RUSAGE_SELF))
    except (OSError, ValueError):
        return 0, 0, 0, 0
    return used.ru_nvcsw, used.ru_nivcsw, used.ru_minflt, used.ru_majflt


def snapshot():
    """Everything the figures subtract: wall, thread CPU, the thread's usage counters, the collections, and every call kind's totals."""
    return dict(wall=time.perf_counter(), cpu=time.thread_time(), usage=usage(), gc=(METER.gc['n'], METER.gc['ms']),
                calls={kind: list(values) for kind, values in METER.calls.items()})


def difference(start, end):
    """The figures of [start, end] as a flat dict (milliseconds, counts)."""
    out = dict(wall_ms=(end['wall'] - start['wall']) * 1000, cpu_ms=(end['cpu'] - start['cpu']) * 1000)
    for name, (a, b) in zip(('nvcsw', 'nivcsw', 'minflt', 'majflt'), zip(start['usage'], end['usage'])):
        out[name] = b - a
    out['gc_n'] = end['gc'][0] - start['gc'][0]
    out['gc_ms'] = end['gc'][1] - start['gc'][1]
    out['calls'] = {kind: (end['calls'][kind][0] - start['calls'][kind][0], (end['calls'][kind][1] - start['calls'][kind][1]) * 1000,
                           (end['calls'][kind][2] - start['calls'][kind][2]) * 1000, end['calls'][kind][3] - start['calls'][kind][3])
                    for kind in KINDS}
    return out


def figures(found, maxima=None, gc_max=None):
    """`found` (difference) as `key=value` text. The calls a span made follow, `kind_n= kind_ms= kind_max_ms=` for each that was made at all (and `kind_gt1=` for those over 1 ms)."""
    parts = ['wall_ms=%.2f' % found['wall_ms'], 'cpu_ms=%.2f' % found['cpu_ms'], 'nvcsw=%d' % found['nvcsw'], 'nivcsw=%d' % found['nivcsw'],
             'minflt=%d' % found['minflt'], 'majflt=%d' % found['majflt'], 'gc_n=%d' % found['gc_n'], 'gc_ms=%.2f' % found['gc_ms']]
    if gc_max is not None:
        parts.append('gc_max_ms=%.2f' % gc_max)
    for kind in KINDS:
        count, wall_ms, cpu_ms, long_calls = found['calls'][kind]
        if count:
            longest = (maxima or {}).get(kind, 0.0) * 1000
            parts.append('%s_n=%d %s_ms=%.2f %s_cpu_ms=%.2f %s_max_ms=%.2f %s_gt1=%d' % (kind, count, kind, wall_ms, kind, cpu_ms, kind, longest, kind, long_calls))
    return ' '.join(parts)


# ---- the wrappers -----------------------------------------------------------------------------------------------------------------

class Timed:
    """`function` with its wall and CPU time recorded under kind_of(args, kwargs); the result and the exceptions are the function's, and every other attribute (ttnn registers its
    operations as objects with attributes of their own) is the function's too."""

    def __init__(self, function, kind_of):
        self.__dict__[WRAPPED_ATTRIBUTE] = function
        self.__dict__['_kind_of'] = kind_of

    def __getattr__(self, name):
        return getattr(self.__dict__[WRAPPED_ATTRIBUTE], name)

    def __call__(self, *args, **kwargs):
        started, cpu = time.perf_counter(), time.thread_time()
        try:
            return self.__dict__[WRAPPED_ATTRIBUTE](*args, **kwargs)
        finally:
            spent = time.perf_counter() - started
            kind = self.__dict__['_kind_of'](args, kwargs)
            record = METER.calls[kind]
            record[0] += 1
            record[1] += spent
            record[2] += time.thread_time() - cpu
            if spent >= LONG_CALL_S:
                record[3] += 1
            if kind == 'tnb' and METER.first_enqueue is None:
                METER.first_enqueue = started
            for frame in METER.frames:
                if spent > frame['max'].get(kind, 0.0):
                    frame['max'][kind] = spent


def timed(function, kind_of):
    return Timed(function, kind_of)


def trace_kind(args, kwargs):
    blocking = kwargs.get('blocking', args[3] if len(args) > 3 else True)
    return 'tb' if blocking else 'tnb'


def install(operations):
    """Wrap the host-call functions of `operations` (the ttnn namespace) once; returns the names wrapped this time. Idempotent. The garbage-collection callback is installed
    here too (once a process)."""
    done = []
    for attribute, kind in CALLS:
        function = getattr(operations, attribute, None)
        if not callable(function) or isinstance(function, Timed):
            continue
        kind_of = trace_kind if kind == 'trace' else (lambda args, kwargs, kind=kind: kind)
        setattr(operations, attribute, timed(function, kind_of))
        METER.installed[(id(operations), attribute)] = (operations, function)
        done.append(attribute)
    install_gc()
    if done:
        log_line('%s wrapped=%s' % (ENGAGED_MARKER, ','.join(done)))
    return done


def uninstall(operations=None):
    """Put the wrapped functions back (the tests; a process never needs it)."""
    for key, (owner, function) in list(METER.installed.items()):
        if operations is None or owner is operations:
            setattr(owner, key[1], function)
            del METER.installed[key]
    if operations is None:
        uninstall_gc()


def install_gc():
    if METER.gc['installed']:
        return False

    def callback(phase, info):
        try:
            if phase == 'start':
                METER.gc['started'] = time.perf_counter()
                return
            ms = (time.perf_counter() - METER.gc['started']) * 1000
            METER.gc['n'] += 1
            METER.gc['ms'] += ms
            for frame in METER.frames:
                if ms > frame['gc_max']:
                    frame['gc_max'] = ms
        except BaseException:
            pass

    gc.callbacks.append(callback)
    METER.gc.update(installed=True, callback=callback)
    return True


def uninstall_gc():
    callback = METER.gc['callback']
    if callback is not None:
        try:
            gc.callbacks.remove(callback)
        except ValueError:
            pass
    METER.gc.update(installed=False, callback=None)


# ---- spans and the round line -----------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def span(name, **tags):
    """One `[PACKED-HOSTGAP-SPAN] step= name= <tags> <figures>` line for the code inside. Never raises, never changes what the code inside does."""
    frame = dict(max={}, gc_max=0.0, start=snapshot())
    METER.frames.append(frame)
    try:
        yield frame
    finally:
        try:
            for index, open_frame in enumerate(METER.frames):
                if open_frame is frame:
                    del METER.frames[index]
                    break
            found = difference(frame['start'], snapshot())
            extra = ''.join(' %s=%s' % (key, str(value).replace(' ', '_')) for key, value in tags.items())
            log_line('%s step=%d name=%s%s %s' % (SPAN_MARKER, METER.step, name, extra, figures(found, frame['max'], frame['gc_max'])))
        except BaseException:
            pass


def note_step():
    """A step's entry (verify_prestage.entry_line, once a step): the ROUND line of the interval that ended here - every step is the time between two entries - then a new
    interval. The first call only opens one."""
    now = snapshot()
    last = METER.last
    METER.last = now
    METER.step += 1
    probed, METER.probed = METER.probed, False
    if last is None:
        return None
    found = difference(last, now)
    line = '%s step=%d probe=%d %s' % (ROUND_MARKER, METER.step - 1, int(probed), figures(found))
    log_line(line)
    return line


# ---- the probe --------------------------------------------------------------------------------------------------------------------

def probe_due(quads):
    """Whether this round is a probe round: the probe is on, the round has two quads in flight, and it is every PROBE_EVERY-th such round."""
    if quads != 2 or not probe_enabled():
        return False
    METER.probes += 1
    return METER.probes % probe_every() == 0


def sync_probe(fence, round_number, quads, started):
    """The timed synchronize right after the quad launches. `fence` is the coordinator's (operations, mesh) pair and `started` the perf_counter at the start of the launches.
    Writes the probe line; marks the open interval as a probe round. Never raises past the synchronize itself."""
    launched = time.perf_counter()
    first = METER.first_enqueue
    fence[0].synchronize_device(fence[1])
    done = time.perf_counter()
    METER.probed = True
    log_line('%s round=%d step=%d quads=%d every=%d launch_ms=%.2f sync_ms=%.2f total_ms=%.2f first_enqueue_ms=%s twoq_ms=%s' % (
        PROBE_MARKER, round_number, METER.step, quads, probe_every(), (launched - started) * 1000, (done - launched) * 1000, (done - started) * 1000,
        '-' if first is None or first < started else '%.2f' % ((first - started) * 1000),
        '-' if first is None or first < started else '%.2f' % ((done - first) * 1000)))
    return (done - launched) * 1000


def begin_launch():
    """Right before the quad launches: forget the last enqueue stamp so the probe finds this round's first."""
    METER.first_enqueue = None
    return time.perf_counter()
