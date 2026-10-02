"""a0_watchdog on fixtures: meminfo, vmstat, kernel-log and docker-event samples, the hysteresis, the loading-phase floor, the kill
that fires once, observe mode that never kills, the trip file."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a0_watchdog as wd  # noqa: E402

GIB = 1 << 20


def meminfo(avail_gib, free_gib=50.0):
    return {'MemAvailable': int(avail_gib * GIB), 'MemFree': int(free_gib * GIB)}


class Rig(object):
    """Scripted readers; poll() consumes the next sample of each."""

    def __init__(self, **kwargs):
        self.mem = [meminfo(100)]
        self.swap = [0]
        self.kernel = ['']
        self.events = [[]]
        self.clock = 1000.0
        self.kills = 0
        self.phase_value = 'running'
        self.dog = wd.Watchdog(lambda: self.pop(self.mem), lambda: self.pop(self.swap), lambda: self.pop(self.kernel),
                               lambda: self.pop(self.events), 'ours', self.kill, lambda: self.clock, kwargs.pop('deadline', 5000.0),
                               lambda: self.phase_value, **kwargs)

    def pop(self, queue):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def kill(self):
        self.kills += 1


class ParseTests(unittest.TestCase):
    def test_meminfo(self):
        text = 'MemTotal:       131072000 kB\nMemFree:          4194304 kB\nMemAvailable:   104857600 kB\nSwapTotal: 0 kB\n'
        info = wd.parse_meminfo(text)
        self.assertEqual((info['MemFree'], info['MemAvailable']), (4194304, 104857600))

    def test_vmstat_swap_in(self):
        self.assertEqual(wd.parse_swap_in('nr_free_pages 3\npswpin 17\npswpout 4\n'), 17)
        self.assertIsNone(wd.parse_swap_in('nr_free_pages 3\n'))

    def test_xid_lines(self):
        text = 'kernel: usb 1-1: new device\nkernel: NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus\n'
        self.assertEqual(len(wd.new_xids(text)), 1)
        self.assertEqual(wd.new_xids('nothing here'), [])

    def test_docker_events(self):
        mine = json.dumps(dict(Type='container', status='start', Actor=dict(Attributes=dict(name='ours'))))
        theirs = json.dumps(dict(Type='container', status='start', Actor=dict(Attributes=dict(name='other'))))
        died = json.dumps(dict(Type='container', status='die', Actor=dict(Attributes=dict(name='other'))))
        image = json.dumps(dict(Type='image', status='start', Actor=dict(Attributes=dict(name='x'))))
        self.assertEqual(wd.foreign_starts([mine, died, image, 'not json'], 'ours'), 0)
        self.assertEqual(wd.foreign_starts([mine, theirs, theirs], 'ours'), 2)


class RuleTests(unittest.TestCase):
    def test_a_healthy_host_never_trips(self):
        rig = Rig()
        for _ in range(20):
            self.assertIsNone(rig.dog.poll())
        self.assertEqual(rig.kills, 0)

    def test_avail_low_needs_two_consecutive_polls(self):
        rig = Rig()
        rig.mem = [meminfo(9.9), meminfo(60), meminfo(9.9), meminfo(9.9)]
        self.assertIsNone(rig.dog.poll())          # one dip
        self.assertIsNone(rig.dog.poll())          # recovered: the count resets
        self.assertIsNone(rig.dog.poll())          # a dip again
        self.assertEqual(rig.dog.poll(), 'avail_low')
        self.assertEqual(rig.kills, 1)

    def test_exactly_at_the_floor_is_not_below_it(self):
        rig = Rig()
        rig.mem = [meminfo(10.0)] * 3
        for _ in range(3):
            self.assertIsNone(rig.dog.poll())

    def test_free_floor_applies_only_while_loading(self):
        rig = Rig()
        rig.mem = [meminfo(60, 3.0)]
        self.assertIsNone(rig.dog.poll())
        rig.phase_value = 'loading'
        self.assertEqual(rig.dog.poll(), 'free_low_loading')

    def test_swap_in_needs_a_baseline_then_trips_on_growth(self):
        rig = Rig()
        rig.swap = [100, 100, 101]
        self.assertIsNone(rig.dog.poll())          # baseline
        self.assertIsNone(rig.dog.poll())          # unchanged
        self.assertEqual(rig.dog.poll(), 'swap_in')

    def test_xid(self):
        rig = Rig()
        rig.kernel = ['', 'NVRM: Xid (PCI:0): 31, pid=1']
        self.assertIsNone(rig.dog.poll())
        self.assertEqual(rig.dog.poll(), 'xid')

    def test_foreign_container_start_trips_and_our_own_does_not(self):
        own = json.dumps(dict(Type='container', status='start', Actor=dict(Attributes=dict(name='ours'))))
        other = json.dumps(dict(Type='container', status='start', Actor=dict(Attributes=dict(name='resident'))))
        rig = Rig()
        rig.events = [[own], [other]]
        self.assertIsNone(rig.dog.poll())
        self.assertEqual(rig.dog.poll(), 'foreign_container')

    def test_deadline(self):
        rig = Rig(deadline=1010.0)
        self.assertIsNone(rig.dog.poll())
        rig.clock = 1010.0
        self.assertEqual(rig.dog.poll(), 'deadline')

    def test_the_kill_fires_once_and_a_tripped_watchdog_stays_tripped(self):
        rig = Rig()
        rig.kernel = ['NVRM: Xid 79']
        self.assertEqual(rig.dog.poll(), 'xid')
        self.assertEqual(rig.dog.poll(), 'xid')
        self.assertEqual(rig.dog.polls, 1)
        self.assertEqual(rig.kills, 1)

    def test_observe_mode_records_but_never_kills(self):
        rig = Rig(mode='observe')
        rig.kernel = ['NVRM: Xid 79']
        self.assertEqual(rig.dog.poll(), 'xid')
        self.assertEqual(rig.kills, 0)

    def test_trip_file_holds_the_reason(self):
        root = tempfile.mkdtemp()
        try:
            path = os.path.join(root, 'trip')
            rig = Rig(trip_file=path)
            rig.kernel = ['NVRM: Xid 79']
            rig.dog.poll()
            with open(path) as handle:
                self.assertEqual(handle.read().strip(), 'xid')
        finally:
            shutil.rmtree(root)

    def test_reasons_are_the_documented_set(self):
        self.assertEqual(set(wd.REASONS), set(['avail_low', 'free_low_loading', 'swap_in', 'xid', 'foreign_container', 'deadline']))


class KillTests(unittest.TestCase):
    def test_docker_kill_runs_kill_then_remove_for_the_named_container_only(self):
        calls = []
        original = wd.subprocess.run
        wd.subprocess.run = lambda command, **kwargs: calls.append(command)
        try:
            wd.docker_kill('ours')
        finally:
            wd.subprocess.run = original
        self.assertEqual(calls, [['docker', 'kill', 'ours'], ['docker', 'rm', '-f', 'ours']])

    def test_cli_requires_a_container_and_a_deadline(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            wd.build_parser().parse_args(['--container', 'x'])


if __name__ == '__main__':
    unittest.main()
