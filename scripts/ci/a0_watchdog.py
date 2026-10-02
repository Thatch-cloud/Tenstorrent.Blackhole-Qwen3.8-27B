"""The A0 screen's abort rule: a small process, started BEFORE the harness, that kills the screen's container the moment the GPU
host starts to starve. Python 3.7, stdlib only. The GPU host has unified memory shared with other work and a second large model
can starve it; the rule is conservative and does not negotiate.

TRIPS (checked every POLL_SECONDS; the first one kills):
    avail_low          MemAvailable below the floor (default 10 GiB) on TWO consecutive polls
    free_low_loading   MemFree below its floor (default 4 GiB) while the harness says it is loading CUDA (phase file)
    swap_in            the swap-in counter of /proc/vmstat grew since the previous poll
    xid                a new NVRM Xid line in the kernel log
    foreign_container  `docker events` shows a container other than ours starting (another service starting)
    deadline           the wall clock passed the deadline

`kill` is injected (the CLI wires `docker kill` then `docker rm -f` of OUR container only). `--mode observe` (the staging day, with the
host still serving) records a trip and the exit code 3 instead of killing anything; `--mode kill` kills. The trip reason is also
written to the trip file, so the harness (a0_run) stops cleanly between turns. Counts and reason codes only: no container name is printed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

POLL_SECONDS = 2.0
AVAIL_FLOOR_GIB = 10.0
FREE_FLOOR_GIB = 4.0
CONSECUTIVE = 2
XID = re.compile(r'NVRM: Xid')
REASONS = ('avail_low', 'free_low_loading', 'swap_in', 'xid', 'foreign_container', 'deadline')
GIB = float(1 << 20)          # /proc/meminfo is in kB


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
                 mode='kill', avail_floor_gib=AVAIL_FLOOR_GIB, free_floor_gib=FREE_FLOOR_GIB, trip_file=None):
        self.read_meminfo, self.read_vmstat, self.read_kernel, self.read_events = read_meminfo, read_vmstat, read_kernel, read_events
        self.own_name, self.kill, self.now, self.deadline, self.phase, self.mode = own_name, kill, now, deadline, phase, mode
        self.avail_floor, self.free_floor = avail_floor_gib * GIB, free_floor_gib * GIB
        self.trip_file = trip_file
        self.low_avail = 0
        self.swap_in = None
        self.tripped = None
        self.polls = 0

    def poll(self):
        """Run one poll; the reason it tripped (and killed, in kill mode) or None. A tripped watchdog stays tripped."""
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


class KernelTail(object):
    """New kernel-log lines since the last read (`journalctl -k --since`, falling back to `dmesg`)."""

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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--container', required=True, help='the screen container (the only one this process may kill)')
    parser.add_argument('--mode', choices=('kill', 'observe'), default='kill')
    parser.add_argument('--deadline', type=float, required=True, help='epoch seconds after which the screen is killed')
    parser.add_argument('--phase-file', help='holds `loading` while the harness loads CUDA, `running` after')
    parser.add_argument('--trip-file', help='the reason is written here on a trip')
    parser.add_argument('--avail-floor-gib', type=float, default=AVAIL_FLOOR_GIB)
    parser.add_argument('--free-floor-gib', type=float, default=FREE_FLOOR_GIB)
    parser.add_argument('--poll-seconds', type=float, default=POLL_SECONDS)
    return parser


def main(argv=None, say=print, sleep=time.sleep, now=time.time):
    options = build_parser().parse_args(argv)
    events, kernel = EventTail(), KernelTail()
    events.start()

    def phase():
        try:
            return read_text(options.phase_file).strip() if options.phase_file else 'running'
        except OSError:
            return 'running'

    dog = Watchdog(lambda: parse_meminfo(read_text('/proc/meminfo')), lambda: parse_swap_in(read_text('/proc/vmstat')), kernel.read,
                   events.drain, options.container, lambda: docker_kill(options.container), now, options.deadline, phase,
                   options.mode, options.avail_floor_gib, options.free_floor_gib, options.trip_file)
    try:
        while True:
            reason = dog.poll()
            if reason:
                say('trip: %s (%s)' % (reason, options.mode))
                return 3
            sleep(options.poll_seconds)
    finally:
        events.stop()


if __name__ == '__main__':
    sys.exit(main())
