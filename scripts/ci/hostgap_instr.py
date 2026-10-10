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

THE SCHEDULER'S SIDE (added after the first instrumented run, whose slow spans were descheduled, CPU bound or waiting, none of them in a collection):
  - the ROUND line ends with the container's CPU-bandwidth counters for the interval (`thr_n` periods in which the cgroup's quota ran out, `thr_ms` time it spent throttled, `periods`
    elapsed: cgroup v2 cpu.stat, or v1) so CFS-quota throttling (docker --cpus) is told from plain contention directly; and with the longest single call of every kind in the interval
    (the interval is an open span of its own);
  - every CENSUS_EVERY-th step (QWEN_FAST_TP4_HOSTGAP_CENSUS_EVERY, default 25, 0 = never) one `[PACKED-HOSTGAP-THREADS]` line: the container's cgroup limits (cpu.max, cpuset), the CPUs
    the main thread may run on, the thread count, and the busiest threads of THIS process since the last census (comm, tid, percent of a CPU, involuntary switches, the CPU it last ran on).
    The census step reads about 150 /proc files, so it is named `census=1` on its round line and the reader drops it with the probe steps;
  - at every fence the number of copies and non-blocking trace enqueues made since the previous synchronization point (a fence, a blocking trace or a blocking read) is written on the
    span as `fence_q_copy=` and `fence_q_tnb=`: what the fence waited behind, besides the traces' own device time;
  - every copy, upload or trace enqueue that takes 1 ms or more (the first BLOCKED_LINES_MAX of the process) gets a `[PACKED-HOSTGAP-BLOCKED]` line of its own: which span it was in, how
    long it took against the CPU it used (a thread that sleeps inside the call, not one that was descheduled), how long after the last non-blocking trace enqueue and the last
    synchronization point it started, and how many copies and trace enqueues were already queued behind that point.

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
CENSUS_EVERY_FLAG = 'QWEN_FAST_TP4_HOSTGAP_CENSUS_EVERY'
CENSUS_EVERY = 25
CENSUS_THREADS = 12
THREADS_MARKER = '[PACKED-HOSTGAP-THREADS]'
BLOCKED_MARKER = '[PACKED-HOSTGAP-BLOCKED]'
BLOCKED_LINE_S = 0.001
BLOCKED_LINES_MAX = 400
ENQUEUE_KINDS = ('copy', 'up', 'tnb', 'd2h')
CGROUP_ROOT = '/sys/fs/cgroup'
PROC_ROOT = '/proc'
SYNC_KINDS = ('fence', 'tb', 'read')
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


def census_every(environ=None):
    """Steps between thread censuses (0 = none)."""
    value = (os.environ if environ is None else environ).get(CENSUS_EVERY_FLAG)
    if value is None:
        return CENSUS_EVERY
    if not value.isdigit():
        raise ValueError('%s must be a non-negative integer' % CENSUS_EVERY_FLAG)
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
        self.pending = dict(copy=0, tnb=0)      # copies and non-blocking trace enqueues since the last synchronization point
        self.interval = None        # the open frame of the step interval (its longest single calls)
        self.cgroup = None          # the cgroup counters at the last step entry
        self.census = None          # the per-thread CPU ticks at the last census
        self.census_steps = 0
        self.censused = False       # a census ran in the interval that is open
        self.last_tnb = None        # perf_counter at the end of the last non-blocking trace enqueue
        self.last_sync = None       # perf_counter at the end of the last synchronization point
        self.blocked_lines = 0


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
            ended = time.perf_counter()
            spent = ended - started
            kind = self.__dict__['_kind_of'](args, kwargs)
            cpu_spent = time.thread_time() - cpu
            record = METER.calls[kind]
            record[0] += 1
            record[1] += spent
            record[2] += cpu_spent
            if spent >= LONG_CALL_S:
                record[3] += 1
                if kind in ENQUEUE_KINDS and METER.blocked_lines < BLOCKED_LINES_MAX:
                    blocked(kind, started, ended, cpu_spent)
            if kind == 'tnb':
                METER.last_tnb = ended
                if METER.first_enqueue is None:
                    METER.first_enqueue = started
            if kind in METER.pending:
                METER.pending[kind] += 1
            elif kind in SYNC_KINDS:
                if kind == 'fence':
                    queued = dict(METER.pending)
                    for frame in METER.frames:
                        frame['fence_queue'] = queued
                METER.pending['copy'] = METER.pending['tnb'] = 0
                METER.last_sync = ended
            for frame in METER.frames:
                if spent > frame['max'].get(kind, 0.0):
                    frame['max'][kind] = spent


def blocked(kind, started, ended, cpu_spent):
    """One `[PACKED-HOSTGAP-BLOCKED]` line for a call that handed work to the device and took LONG_CALL_S or more. Never raises."""
    try:
        METER.blocked_lines += 1
        inside = next((frame['name'] for frame in reversed(METER.frames) if frame.get('name')), '-')
        since = lambda mark: '-' if mark is None else '%.2f' % ((started - mark) * 1000)
        log_line('%s step=%d kind=%s ms=%.2f cpu_ms=%.2f in=%s since_tnb_ms=%s since_sync_ms=%s queued_copy=%d queued_tnb=%d' % (
            BLOCKED_MARKER, METER.step, kind, (ended - started) * 1000, cpu_spent * 1000, inside, since(METER.last_tnb), since(METER.last_sync),
            METER.pending['copy'], METER.pending['tnb']))
    except BaseException:
        pass


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
    frame = dict(name=name, max={}, gc_max=0.0, start=snapshot())
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
            queue = frame.get('fence_queue')
            tail = '' if queue is None else ' fence_q_copy=%d fence_q_tnb=%d' % (queue['copy'], queue['tnb'])
            log_line('%s step=%d name=%s%s %s%s' % (SPAN_MARKER, METER.step, name, extra, figures(found, frame['max'], frame['gc_max']), tail))
        except BaseException:
            pass


def read_text(path):
    try:
        with open(path) as handle:
            return handle.read()
    except (OSError, ValueError):
        return None


def cgroup_counters(root=None):
    """(periods, throttled periods, throttled microseconds) of this container's cgroup (v2 cpu.stat, else v1), or None where the host cannot say."""
    root = CGROUP_ROOT if root is None else root
    text = read_text(root + '/cpu.stat')
    if text is None:
        text = read_text(root + '/cpu/cpu.stat') or read_text(root + '/cpu,cpuacct/cpu.stat')
    if text is None:
        return None
    found = {}
    for line in text.splitlines():
        name, _, value = line.partition(' ')
        if value.strip().isdigit():
            found[name] = int(value)
    if 'nr_throttled' not in found:
        return None
    if 'throttled_usec' in found:
        micro = found['throttled_usec']
    else:
        micro = found.get('throttled_time', 0) // 1000                        # v1 counts throttled_time in nanoseconds
    return found.get('nr_periods', 0), found['nr_throttled'], int(micro)


def cgroup_limits(root=None):
    """The container's CPU limits as text: 'quota=<quota>/<period> cpuset=<cpus>' (cgroup v2 cpu.max, else v1 cfs_quota_us and cfs_period_us; quota 'max' is none), '-' where unreadable."""
    root = CGROUP_ROOT if root is None else root
    quota = (read_text(root + '/cpu.max') or '').split()
    if len(quota) == 2:
        limit = '%s/%s' % (quota[0], quota[1])
    else:
        micro = (read_text(root + '/cpu/cpu.cfs_quota_us') or read_text(root + '/cpu,cpuacct/cpu.cfs_quota_us') or '').strip()
        period = (read_text(root + '/cpu/cpu.cfs_period_us') or read_text(root + '/cpu,cpuacct/cpu.cfs_period_us') or '').strip()
        limit = '%s/%s' % (micro, period) if micro and period else '-'
    cpuset = (read_text(root + '/cpuset.cpus.effective') or read_text(root + '/cpuset/cpuset.effective_cpus') or '-').strip() or '-'
    return 'quota=%s cpuset=%s' % (limit, cpuset)


def thread_ticks(proc=None):
    """{tid: (comm, ticks used, involuntary switches, last cpu)} for the threads of this process: /proc/self/task/<tid>/stat and status. Unreadable threads are left out."""
    proc = PROC_ROOT if proc is None else proc
    found = {}
    base = proc + '/self/task'
    try:
        tids = os.listdir(base)
    except OSError:
        return found
    for tid in tids:
        text = read_text('%s/%s/stat' % (base, tid))
        if not text or ')' not in text:
            continue
        head, tail = text.rsplit(')', 1)
        fields = tail.split()
        if len(fields) < 37:
            continue
        comm = head.split('(', 1)[-1]
        involuntary = 0
        status = read_text('%s/%s/status' % (base, tid)) or ''
        for line in status.splitlines():
            if line.startswith('nonvoluntary_ctxt_switches'):
                involuntary = int(line.split()[1])
                break
        found[tid] = (comm.replace(' ', '_'), int(fields[11]) + int(fields[12]), involuntary, int(fields[36]))
    return found


def census(step, proc=None, root=None):
    """The `[PACKED-HOSTGAP-THREADS]` line: the limits, the main thread's allowed CPUs, the thread count and the busiest threads since the last census. Returns the line."""
    now, ticks = time.perf_counter(), thread_ticks(proc)
    before, METER.census = METER.census, (now, ticks)
    allowed = '-'
    status = read_text((PROC_ROOT if proc is None else proc) + '/self/status') or ''
    for line in status.splitlines():
        if line.startswith('Cpus_allowed_list'):
            allowed = line.split(None, 1)[1].strip()
    busy = []
    if before is not None:
        interval = max(now - before[0], 1e-6)
        tick = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') else 100
        for tid, (comm, used, involuntary, cpu) in ticks.items():
            old = before[1].get(tid)
            share = 100.0 * (used - (old[1] if old else 0)) / tick / interval
            busy.append((share, comm, tid, involuntary - (old[2] if old else 0), cpu))
        busy.sort(reverse=True)
    line = '%s step=%d threads=%d allowed=%s %s interval_s=%s busy=%s' % (
        THREADS_MARKER, METER.step, len(ticks), allowed, cgroup_limits(root), '-' if before is None else '%.2f' % (now - before[0]),
        ','.join('%s:%s:%.0f%%:%dinv:cpu%d' % (comm, tid, share, inv, cpu) for share, comm, tid, inv, cpu in busy[:CENSUS_THREADS]) or '-')
    log_line(line)
    return line


def note_step(root=None, proc=None):
    """A step's entry (verify_prestage.entry_line, once a step): the ROUND line of the interval that ended here - every step is the time between two entries - then a new
    interval. The first call only opens one. The line ends with the interval's longest single call of each kind and the container's CPU-bandwidth counters (thr_n throttled
    periods, thr_ms throttled time, periods)."""
    now = snapshot()
    last = METER.last
    METER.last = now
    METER.step += 1
    probed, METER.probed = METER.probed, False
    censused, METER.censused = METER.censused, False
    counters, before = cgroup_counters(root), METER.cgroup
    METER.cgroup = counters
    interval, METER.interval = METER.interval, dict(max={}, gc_max=0.0, start=now)
    METER.frames[:] = [frame for frame in METER.frames if frame is not interval]
    METER.frames.append(METER.interval)
    line = None
    if last is not None:
        found = difference(last, now)
        tail = ''
        if counters is not None and before is not None:
            tail = ' periods=%d thr_n=%d thr_ms=%.1f' % (counters[0] - before[0], counters[1] - before[1], (counters[2] - before[2]) / 1000.0)
        line = '%s step=%d probe=%d census=%d %s%s' % (ROUND_MARKER, METER.step - 1, int(probed), int(censused), figures(
            found, None if interval is None else interval['max'], None if interval is None else interval['gc_max']), tail)
        log_line(line)
    every = census_every()
    if every and METER.step > 1 and (METER.step - 1) % every == 0:
        METER.censused = True
        census(METER.step, proc, root)
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
