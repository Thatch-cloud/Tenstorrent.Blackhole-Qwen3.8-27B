"""The A0 screen's abort rule: a small process, started BEFORE the harness, that kills the screen's container the moment the GPU
host runs short of memory. Python 3.7, stdlib only. The GPU host has unified memory shared with other work and a second large model
can exhaust it; the rule is conservative and does not negotiate.

TRIPS (checked every POLL_SECONDS; the first one kills):
    avail_low          MemAvailable below the floor (default 10 GiB) on TWO consecutive polls
    free_low_loading   MemFree below its floor (default 4 GiB) while the harness says it is loading CUDA (phase file)
    swap_in            the swap-in counter of /proc/vmstat grew since the previous poll
    xid                a new NVRM Xid line in the kernel log
    foreign_container  `docker events` shows a container other than ours starting (another service starting)
    deadline           the wall clock passed the deadline

FAIL CLOSED. The watchdog rewrites a HEARTBEAT file on every poll; the harness (a0_run) stops when it is missing or older than 10 s, so a
dead or hung watchdog never leaves the screen running unguarded. After a trip the kill is VERIFIED (`docker inspect`) and escalated
(`docker kill`, `docker rm -f`, then SIGKILL to the container's last known PID) from a worker thread until the container is gone; the poll
loop is never blocked by a hung docker call and does not exit until the container is confirmed dead. Nothing in the poll loop forks: the
kernel log (/dev/kmsg, or journalctl in a thread) and `docker events` are read by threads; the poll reads /proc/meminfo and /proc/vmstat.
The watchdog hardens itself (oom_score_adj -1000, mlockall) and the screen's container is started with a high OOM score and no swap
(`a0_watchdog.py flags`). `a0_watchdog.py handback` is the hand-back check: it drops the screen's files from the page cache, then
requires MemAvailable AND MemFree floors the operator names.

`kill` is injected (the CLI wires the verified killer of OUR container only). `--mode observe` (the staging day, with the
host still serving) records a trip and the exit code 3 instead of killing anything; `--mode kill` kills. The trip reason is also
written to the trip file, so the harness (a0_run) stops cleanly between turns. Counts and reason codes only: no container name is printed.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

POLL_SECONDS = 0.5
AVAIL_FLOOR_GIB = 10.0
FREE_FLOOR_GIB = 4.0
CONSECUTIVE = 2
XID = re.compile(r'NVRM: Xid')
REASONS = ('avail_low', 'free_low_loading', 'swap_in', 'xid', 'foreign_container', 'deadline')
GIB = float(1 << 20)          # /proc/meminfo is in kB
NL = chr(10)


def parse_meminfo(text):
    """{field: kB} from /proc/meminfo text."""
    out = {}
    for line in text.splitlines():
        name, _, rest = line.partition(':')
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[name.strip()] = int(parts[0])
    return out


def parse_swap_in(text):
    """pswpin of /proc/vmstat text (pages swapped in since boot), or None."""
    for line in text.splitlines():
        if line.startswith('pswpin '):
            return int(line.split()[1])
    return None


def new_xids(text):
    return [line for line in text.splitlines() if XID.search(line)]


def foreign_starts(lines, own_name):
    """The container start events in `docker events --format '{{json .}}'` lines whose container is not `own_name`."""
    count = 0
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('Type', 'container') != 'container' or event.get('status', event.get('Action')) != 'start':
            continue
        name = ((event.get('Actor') or {}).get('Attributes') or {}).get('name')
        if name != own_name:
            count += 1
    return count


class Watchdog(object):
    """One poll at a time over injected readers (so the rule is tested with fixtures, no host)."""

    def __init__(self, read_meminfo, read_vmstat, read_kernel, read_events, own_name, kill, now, deadline, phase=lambda: 'running',
                 mode='kill', avail_floor_gib=AVAIL_FLOOR_GIB, free_floor_gib=FREE_FLOOR_GIB, trip_file=None, heartbeat_file=None):
        self.read_meminfo, self.read_vmstat, self.read_kernel, self.read_events = read_meminfo, read_vmstat, read_kernel, read_events
        self.own_name, self.kill, self.now, self.deadline, self.phase, self.mode = own_name, kill, now, deadline, phase, mode
        self.avail_floor, self.free_floor = avail_floor_gib * GIB, free_floor_gib * GIB
        self.trip_file, self.heartbeat_file = trip_file, heartbeat_file
        self.low_avail = 0
        self.swap_in = None
        self.tripped = None
        self.polls = 0

    def poll(self):
        """Run one poll; the reason it tripped (and killed, in kill mode) or None. A tripped watchdog stays tripped."""
        self.beat()
        if self.tripped:
            return self.tripped
        self.polls += 1
        reason = self._check()
        if reason:
            self.tripped = reason
            if self.trip_file:
                with open(self.trip_file, 'w') as handle:
                    handle.write(reason + '\n')
            if self.mode == 'kill':
                self.kill()
        return reason

    def beat(self):
        """The heartbeat the harness fails closed on: the time of this poll, replaced atomically. A write that fails is not hidden: the
        harness sees the stale file."""
        if self.heartbeat_file:
            try:
                write_heartbeat(self.heartbeat_file, self.now())
            except OSError:
                pass

    def _check(self):
        info = self.read_meminfo()
        avail, free = info.get('MemAvailable', 0), info.get('MemFree', 0)
        if avail < self.avail_floor:
            self.low_avail += 1
        else:
            self.low_avail = 0
        if self.low_avail >= CONSECUTIVE:
            return 'avail_low'
        if self.phase() == 'loading' and free < self.free_floor:
            return 'free_low_loading'
        swapped = self.read_vmstat()
        if swapped is not None:
            if self.swap_in is not None and swapped > self.swap_in:
                self.swap_in = swapped
                return 'swap_in'
            self.swap_in = swapped
        if new_xids(self.read_kernel()):
            return 'xid'
        if foreign_starts(self.read_events(), self.own_name):
            return 'foreign_container'
        if self.now() >= self.deadline:
            return 'deadline'
        return None


# -- the host wiring ------------------------------------------------------------------------------------------------------

def write_heartbeat(path, stamp):
    temporary = path + '.tmp'
    with open(temporary, 'w') as handle:
        handle.write('%r\n' % stamp)
    os.replace(temporary, path)


def read_text(path):
    with open(path, encoding='utf-8', errors='replace') as handle:
        return handle.read()


class EventTail(object):
    """`docker events` for container start events, read by a thread into a list the poll drains."""

    def __init__(self):
        self.lines, self.lock, self.process = [], threading.Lock(), None

    def start(self):
        self.process = subprocess.Popen(['docker', 'events', '--filter', 'type=container', '--filter', 'event=start',
                                         '--format', '{{json .}}'], stdout=subprocess.PIPE, universal_newlines=True)

        def pump():
            for line in self.process.stdout:
                with self.lock:
                    self.lines.append(line)
        threading.Thread(target=pump, daemon=True).start()

    def drain(self):
        with self.lock:
            lines, self.lines = self.lines, []
        return lines

    def stop(self):
        if self.process:
            self.process.terminate()


def parse_kmsg_record(record):
    """The message of one /dev/kmsg record ("priority,sequence,timestamp,flags;message") as text; '' when it has none."""
    text = record.decode('utf-8', 'replace') if isinstance(record, bytes) else record
    head, sep, message = text.partition(';')
    return message.strip() if sep else ''


class KmsgTail(object):
    """/dev/kmsg read by a thread from its END (records already in the buffer are history): one record per read, BlockingIOError when
    none is waiting. `opener(path)` -> fd and `reader(fd)` -> bytes are injected for tests."""

    def __init__(self, path='/dev/kmsg', opener=None, reader=None, interval=0.2):
        self.path, self.interval = path, interval
        self.opener = opener or (lambda target: os.open(target, os.O_RDONLY | os.O_NONBLOCK))
        self.reader = reader or (lambda descriptor: os.read(descriptor, 8192))
        self.lines, self.lock, self.descriptor = [], threading.Lock(), None
        self.stopping = threading.Event()

    def available(self):
        try:
            self.descriptor = self.opener(self.path)
        except OSError:
            return False
        try:
            os.lseek(self.descriptor, 0, os.SEEK_END)
        except (OSError, AttributeError):
            pass
        return True

    def pump(self):
        """Read every record waiting now; returns how many."""
        count = 0
        while True:
            try:
                record = self.reader(self.descriptor)
            except BlockingIOError:
                return count
            except OSError:
                return count
            if not record:
                return count
            message = parse_kmsg_record(record)
            if message:
                with self.lock:
                    self.lines.append(message)
            count += 1

    def start(self):
        def loop():
            while not self.stopping.is_set():
                self.pump()
                self.stopping.wait(self.interval)
        threading.Thread(target=loop, daemon=True).start()

    def read(self):
        with self.lock:
            lines, self.lines = self.lines, []
        return NL.join(lines)

    def stop(self):
        self.stopping.set()


class ThreadedReader(object):
    """Runs a slow reader (journalctl) on a thread every `interval` seconds and hands the accumulated text to the poll, so the poll never forks."""

    def __init__(self, function, interval=2.0):
        self.function, self.interval = function, interval
        self.chunks, self.lock, self.stopping = [], threading.Lock(), threading.Event()

    def start(self):
        def loop():
            while not self.stopping.is_set():
                try:
                    text = self.function()
                except Exception:
                    text = ''
                if text:
                    with self.lock:
                        self.chunks.append(text)
                self.stopping.wait(self.interval)
        threading.Thread(target=loop, daemon=True).start()

    def read(self):
        with self.lock:
            chunks, self.chunks = self.chunks, []
        return NL.join(chunks)

    def stop(self):
        self.stopping.set()


class KernelTail(object):
    """New kernel-log lines since the last read (`journalctl -k --since`); used on a thread when /dev/kmsg cannot be opened."""

    def __init__(self, run=subprocess.run, clock=time.time):
        self.run, self.clock, self.since = run, clock, clock()

    def read(self):
        since = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.since))
        self.since = self.clock()
        try:
            result = self.run(['journalctl', '-k', '--since', since, '--no-pager'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              universal_newlines=True, timeout=10)
            return result.stdout
        except (OSError, subprocess.SubprocessError):
            return ''


def docker_kill(name):
    for command in (['docker', 'kill', name], ['docker', 'rm', '-f', name]):
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        except (OSError, subprocess.SubprocessError):
            pass


def docker_state(name, run=subprocess.run):
    """('running' | 'gone' | 'unknown', pid or None) from `docker inspect`: a refusal that says there is no such container is 'gone'; a
    timeout or an error that says nothing is 'unknown' (never read as gone)."""
    try:
        result = run(['docker', 'inspect', '-f', '{{.State.Running}} {{.State.Pid}}', name], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     universal_newlines=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return 'unknown', None
    if result.returncode != 0:
        return ('gone', None) if 'no such' in (result.stderr or '').lower() else ('unknown', None)
    parts = (result.stdout or '').split()
    if len(parts) != 2:
        return 'unknown', None
    pid = int(parts[1]) if parts[1].isdigit() and int(parts[1]) > 0 else None
    return ('running' if parts[0] == 'true' else 'gone'), pid


class Killer(object):
    """Kill OUR container until it is gone. One attempt = `kill(name)` (docker kill, docker rm -f), then, when it is still there, SIGKILL
    to the last PID seen. `state()` -> 'running' | 'gone' | 'unknown'. Attempts run on a worker thread (a docker call can take its full
    timeout under memory pressure); `dead` is true once a state check says 'gone' (or the container never existed)."""

    def __init__(self, name, kill=docker_kill, state=None, kill_pid=os.kill, pid_alive=None, sigkill=getattr(signal, 'SIGKILL', 9)):
        self.name, self.kill, self.kill_pid, self.sigkill = name, kill, kill_pid, sigkill
        self.state = state or (lambda: docker_state(name))
        self.pid_alive = pid_alive or (lambda pid: os.path.exists('/proc/%d' % pid))
        self.last_pid = None
        self.attempts = 0
        self.escalations = 0
        self.dead = False
        self.thread = None
        self.lock = threading.Lock()

    def note_pid(self):
        """Remember the PID of the running container (called from a thread of its own, while the container is up)."""
        state, pid = self.state()
        if state == 'running' and pid:
            self.last_pid = pid

    def attempt(self):
        self.attempts += 1
        self.kill()
        state, _ = self.state()
        if state == 'gone':
            self.dead = True
            return
        if self.last_pid and self.pid_alive(self.last_pid):         # docker could not do it (or could not answer): the PID can
            self.escalations += 1
            try:
                self.kill_pid(self.last_pid, self.sigkill)
            except OSError:
                pass
        state, _ = self.state()
        self.dead = state == 'gone' or (state == 'unknown' and self.last_pid is not None and not self.pid_alive(self.last_pid))

    def start(self):
        """Begin (or, if the last attempt finished without success, repeat) the kill on a worker thread; returns at once."""
        with self.lock:
            if self.dead or (self.thread is not None and self.thread.is_alive()):
                return
            self.thread = threading.Thread(target=self.attempt, daemon=True)
            self.thread.start()


def watch_pid(killer, stop, interval=2.0):
    """A thread that keeps the killer's PID current while the container runs (the container starts after the watchdog)."""
    def loop():
        while not stop.is_set():
            try:
                killer.note_pid()
            except Exception:
                pass
            stop.wait(interval)
    threading.Thread(target=loop, daemon=True).start()


def harden(write_text=None, lock_memory=None):
    """The watchdog's own protection against the pressure it watches for: oom_score_adj -1000 and mlockall (needs root or the capabilities;
    each result is reported, never assumed). -> {'oom': bool, 'mlock': bool}."""
    def write(path, value):
        with open(path, 'w') as handle:
            handle.write(value)

    def lock():
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.mlockall(3) != 0:                                    # MCL_CURRENT | MCL_FUTURE
            raise OSError(ctypes.get_errno(), 'mlockall')
    out = dict(oom=False, mlock=False)
    try:
        (write_text or write)('/proc/self/oom_score_adj', '-1000')
        out['oom'] = True
    except OSError:
        pass
    try:
        (lock_memory or lock)()
        out['mlock'] = True
    except (OSError, AttributeError, ValueError):
        pass
    return out


def container_flags(memory_gib):
    """The docker flags the screen's container is started with: a hard memory limit, swap equal to it (so none), and the highest OOM score
    so the kernel chooses it before anything else."""
    memory = int(memory_gib)
    if memory < 1:
        raise ValueError('a memory limit of at least 1 GiB is required')
    return ['--memory=%dg' % memory, '--memory-swap=%dg' % memory, '--oom-score-adj=1000']


def drop_file_cache(directory):
    """posix_fadvise(DONTNEED) on every file under `directory` (the page cache of the screen's files); the number of files advised."""
    advise = getattr(os, 'posix_fadvise', None)
    if advise is None:
        return 0
    count = 0
    for base, _, names in os.walk(directory):
        for name in names:
            try:
                descriptor = os.open(os.path.join(base, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                advise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                count += 1
            finally:
                os.close(descriptor)
    return count


def handback_ready(read_meminfo, drop_dirs, drop, avail_floor_gib, free_floor_gib):
    """Drop the screen's files from the page cache, then check the host has what the next tenant needs: MemAvailable AND MemFree (cached
    files count as available but are not free). -> dict(ready, dropped, avail_gib, free_gib)."""
    dropped = sum(drop(directory) for directory in drop_dirs)
    info = read_meminfo()
    avail, free = info.get('MemAvailable', 0) / GIB, info.get('MemFree', 0) / GIB
    return dict(ready=avail >= avail_floor_gib and free >= free_floor_gib, dropped=dropped, avail_gib=round(avail, 1), free_gib=round(free, 1))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--container', required=True, help='the screen container (the only one this process may kill)')
    parser.add_argument('--mode', choices=('kill', 'observe'), default='kill')
    parser.add_argument('--deadline', type=float, required=True, help='epoch seconds after which the screen is killed')
    parser.add_argument('--phase-file', help='holds `loading` while the harness loads CUDA, `running` after')
    parser.add_argument('--trip-file', help='the reason is written here on a trip')
    parser.add_argument('--heartbeat-file', help='rewritten every poll; the harness stops when it is missing or stale (required in kill mode)')
    parser.add_argument('--avail-floor-gib', type=float, default=AVAIL_FLOOR_GIB)
    parser.add_argument('--free-floor-gib', type=float, default=FREE_FLOOR_GIB)
    parser.add_argument('--poll-seconds', type=float, default=POLL_SECONDS)
    return parser


def flags_main(argv, say):
    parser = argparse.ArgumentParser(prog='a0_watchdog.py flags', description='the docker flags the screen container runs with')
    parser.add_argument('--memory-gib', type=int, required=True)
    options = parser.parse_args(argv)
    say(' '.join(container_flags(options.memory_gib)))
    return 0


def handback_main(argv, say, read_meminfo=None, drop=drop_file_cache):
    """Exit 0 and `ready` when the host has what the next tenant needs; exit 1 and `not_ready` (with the two numbers) otherwise."""
    parser = argparse.ArgumentParser(prog='a0_watchdog.py handback', description=handback_ready.__doc__.split('\n')[0])
    parser.add_argument('--drop-dir', action='append', default=[], help='a directory of the screen\'s files to drop from the page cache')
    parser.add_argument('--avail-floor-gib', type=float, required=True)
    parser.add_argument('--free-floor-gib', type=float, required=True)
    options = parser.parse_args(argv)
    reader = read_meminfo or (lambda: parse_meminfo(read_text('/proc/meminfo')))
    result = handback_ready(reader, options.drop_dir, drop, options.avail_floor_gib, options.free_floor_gib)
    say('%s: %d files dropped, MemAvailable %.1f GiB, MemFree %.1f GiB' % ('ready' if result['ready'] else 'not_ready', result['dropped'],
                                                                         result['avail_gib'], result['free_gib']))
    return 0 if result['ready'] else 1


def main(argv=None, say=print, sleep=time.sleep, now=time.time, hardener=harden, make_killer=Killer, make_kernel=None, make_events=EventTail):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'flags':
        return flags_main(argv[1:], say)
    if argv and argv[0] == 'handback':
        return handback_main(argv[1:], say)
    options = build_parser().parse_args(argv)
    if options.mode == 'kill' and not options.heartbeat_file:
        say('refused: kill mode needs --heartbeat-file')
        return 2
    protection = hardener()
    say('hardened: oom=%s mlock=%s' % ('yes' if protection['oom'] else 'no', 'yes' if protection['mlock'] else 'no'))
    events = make_events()
    events.start()
    kernel = make_kernel() if make_kernel else KmsgTail()
    if isinstance(kernel, KmsgTail) and not kernel.available():
        kernel = ThreadedReader(KernelTail().read)
    kernel.start()
    killer = make_killer(options.container)
    stop = threading.Event()
    watch_pid(killer, stop)

    def phase():
        try:
            return read_text(options.phase_file).strip() if options.phase_file else 'running'
        except OSError:
            return 'running'

    dog = Watchdog(lambda: parse_meminfo(read_text('/proc/meminfo')), lambda: parse_swap_in(read_text('/proc/vmstat')), kernel.read,
                   events.drain, options.container, killer.start, now, options.deadline, phase,
                   options.mode, options.avail_floor_gib, options.free_floor_gib, options.trip_file, options.heartbeat_file)
    reported, last_note = False, 0
    try:
        while True:
            reason = dog.poll()
            if reason:
                if not reported:
                    say('trip: %s (%s)' % (reason, options.mode))
                    reported = True
                if options.mode == 'observe':
                    return 3
                killer.start()                                  # repeats an attempt that ended without the container gone
                if killer.dead:
                    say('killed: %d attempts, %d escalations' % (killer.attempts, killer.escalations))
                    return 3
                if now() - last_note >= 30:
                    last_note = now()
                    say('kill not yet confirmed: %d attempts, %d escalations' % (killer.attempts, killer.escalations))
            sleep(options.poll_seconds)
    finally:
        stop.set()
        events.stop()
        kernel.stop()


if __name__ == '__main__':
    sys.exit(main())
