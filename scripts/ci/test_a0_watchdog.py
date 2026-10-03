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
NL = chr(10)


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


class HeartbeatTests(unittest.TestCase):
    def test_every_poll_writes_the_time_and_a_tripped_watchdog_keeps_writing(self):
        root = tempfile.mkdtemp()
        try:
            path = os.path.join(root, 'beat')
            rig = Rig(heartbeat_file=path)
            rig.dog.poll()
            with open(path) as handle:
                self.assertEqual(float(handle.read()), 1000.0)
            rig.clock = 1003.0
            rig.dog.poll()
            with open(path) as handle:
                self.assertEqual(float(handle.read()), 1003.0)
            rig.kernel = ['NVRM: Xid 79']
            rig.clock = 1004.0
            self.assertEqual(rig.dog.poll(), 'xid')
            rig.clock = 1006.0
            rig.dog.poll()                                   # tripped: still beating (the trip file, not silence, tells the harness)
            with open(path) as handle:
                self.assertEqual(float(handle.read()), 1006.0)
            self.assertFalse(os.path.exists(path + '.tmp'))
        finally:
            shutil.rmtree(root)

    def test_a_heartbeat_that_cannot_be_written_does_not_stop_the_poll(self):
        rig = Rig(heartbeat_file=os.path.join(tempfile.gettempdir(), 'no-such-dir-a0', 'beat'))
        self.assertIsNone(rig.dog.poll())

    def test_the_harness_guard_reads_what_the_watchdog_writes(self):
        import a0_run
        root = tempfile.mkdtemp()
        try:
            path = os.path.join(root, 'beat')
            rig = Rig(heartbeat_file=path)
            rig.dog.poll()
            guard = a0_run.Guard(None, 10 ** 9, lambda: rig.clock + 2, lambda: 100.0, heartbeat_file=path)
            self.assertIsNone(guard.stop_reason())
            guard.clock = lambda: rig.clock + 60
            self.assertEqual(guard.stop_reason(), 'watchdog_stale')
        finally:
            shutil.rmtree(root)


class KillerTests(unittest.TestCase):
    def killer(self, states, pid_alive_values=(True,), pid=4242, kill=None):
        calls = dict(kill=0, pid=[])
        queue = list(states)

        def state():
            return (queue.pop(0) if len(queue) > 1 else queue[0]), pid

        def kill_fn():
            calls['kill'] += 1
        sent = calls['pid']
        alive = list(pid_alive_values)
        killer = wd.Killer('ours', kill=kill or kill_fn, state=state, kill_pid=lambda target, sig: sent.append((target, sig)),
                           pid_alive=lambda target: alive.pop(0) if len(alive) > 1 else alive[0], sigkill=9)
        killer.last_pid = pid
        return killer, calls

    def test_a_container_gone_after_docker_kill_needs_no_escalation(self):
        killer, calls = self.killer(['gone'])
        killer.attempt()
        self.assertTrue(killer.dead)
        self.assertEqual((calls['kill'], calls['pid']), (1, []))

    def test_a_container_that_survives_docker_kill_gets_a_sigkill_to_its_pid(self):
        killer, calls = self.killer(['running', 'gone'])
        killer.attempt()
        self.assertEqual(calls['pid'], [(4242, 9)])
        self.assertTrue(killer.dead)
        self.assertEqual(killer.escalations, 1)

    def test_still_running_after_the_escalation_is_not_dead_and_is_tried_again(self):
        killer, calls = self.killer(['running'])
        killer.attempt()
        self.assertFalse(killer.dead)
        killer.attempt()
        self.assertEqual(killer.attempts, 2)
        self.assertEqual(len(calls['pid']), 2)

    def test_an_unanswering_docker_is_never_read_as_gone_while_the_pid_lives(self):
        killer, calls = self.killer(['unknown'], pid_alive_values=(True,))
        killer.attempt()
        self.assertFalse(killer.dead)
        self.assertEqual(len(calls['pid']), 1)

    def test_an_unanswering_docker_with_a_dead_pid_counts_as_dead(self):
        killer, calls = self.killer(['unknown'], pid_alive_values=(False,))
        killer.attempt()
        self.assertTrue(killer.dead)

    def test_unknown_with_no_pid_is_not_dead(self):
        killer, calls = self.killer(['unknown'])
        killer.last_pid = None
        killer.attempt()
        self.assertFalse(killer.dead)

    def test_start_runs_on_a_thread_does_not_overlap_and_stops_once_dead(self):
        import threading
        release, entered = threading.Event(), threading.Event()

        def slow():
            entered.set()
            release.wait(5)
        killer, calls = self.killer(['gone'], kill=slow)
        killer.start()
        self.assertTrue(entered.wait(5))
        killer.start()                                       # an attempt is running: no second one
        release.set()
        killer.thread.join(5)
        self.assertEqual(killer.attempts, 1)
        self.assertTrue(killer.dead)
        killer.start()
        self.assertEqual(killer.attempts, 1)

    def test_note_pid_remembers_a_running_container_only(self):
        killer, _ = self.killer(['running'], pid=77)
        killer.last_pid = None
        killer.note_pid()
        self.assertEqual(killer.last_pid, 77)
        other, _ = self.killer(['gone'], pid=88)
        other.last_pid = None
        other.note_pid()
        self.assertIsNone(other.last_pid)

    def test_docker_state_reads_inspect(self):
        class Result(object):
            def __init__(self, code, out='', err=''):
                self.returncode, self.stdout, self.stderr = code, out, err

        def run_with(result):
            def run(command, **kwargs):
                self.last = command
                if isinstance(result, Exception):
                    raise result
                return result
            return run
        self.assertEqual(wd.docker_state('ours', run_with(Result(0, 'true 321'))), ('running', 321))
        self.assertEqual(self.last[:3], ['docker', 'inspect', '-f'])
        self.assertEqual(self.last[-1], 'ours')
        self.assertEqual(wd.docker_state('ours', run_with(Result(0, 'false 0'))), ('gone', None))
        self.assertEqual(wd.docker_state('ours', run_with(Result(1, '', 'Error: No such object: ours'))), ('gone', None))
        self.assertEqual(wd.docker_state('ours', run_with(Result(1, '', 'Cannot connect to the Docker daemon'))), ('unknown', None))
        self.assertEqual(wd.docker_state('ours', run_with(wd.subprocess.TimeoutExpired('docker', 10))), ('unknown', None))
        self.assertEqual(wd.docker_state('ours', run_with(OSError())), ('unknown', None))
        self.assertEqual(wd.docker_state('ours', run_with(Result(0, 'garbage'))), ('unknown', None))


class KernelReaderTests(unittest.TestCase):
    def test_kmsg_records_are_parsed_to_their_message(self):
        self.assertEqual(wd.parse_kmsg_record(b'6,1234,5678,-;NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus'),
                         'NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus')
        self.assertEqual(wd.parse_kmsg_record(b'no semicolon'), '')
        self.assertEqual(wd.parse_kmsg_record('6,1,2,-;text;with;semicolons'), 'text;with;semicolons')

    def test_pump_reads_every_waiting_record_then_stops_on_would_block(self):
        records = [b'6,1,1,-;usb connected', b'3,2,2,-;NVRM: Xid 79', b'']
        queue = list(records)

        def reader(descriptor):
            item = queue.pop(0)
            if item == b'':
                raise BlockingIOError()
            return item
        tail = wd.KmsgTail(opener=lambda path: 7, reader=reader)
        tail.descriptor = 7
        self.assertEqual(tail.pump(), 2)
        text = tail.read()
        self.assertEqual(len(wd.new_xids(text)), 1)
        self.assertEqual(tail.read(), '')                    # drained

    def test_an_unopenable_kmsg_is_reported_so_the_caller_falls_back(self):
        def refuse(path):
            raise PermissionError()
        self.assertFalse(wd.KmsgTail(opener=refuse).available())

    def test_the_threaded_reader_accumulates_and_drains_and_survives_an_error(self):
        import time as clock
        outputs = iter(['one', RuntimeError('x'), 'two'])

        def function():
            item = next(outputs, '')
            if isinstance(item, Exception):
                raise item
            return item
        reader = wd.ThreadedReader(function, interval=0.01)
        reader.start()
        deadline = clock.time() + 3
        seen = ''
        while 'two' not in seen and clock.time() < deadline:
            seen += reader.read()
            clock.sleep(0.01)
        reader.stop()
        self.assertIn('one', seen)
        self.assertIn('two', seen)


class HardeningAndFlagsTests(unittest.TestCase):
    def test_harden_reports_each_protection_on_its_own(self):
        written = []
        self.assertEqual(wd.harden(lambda path, value: written.append((path, value)), lambda: None), dict(oom=True, mlock=True))
        self.assertEqual(written, [('/proc/self/oom_score_adj', '-1000')])

        def denied(*args):
            raise PermissionError()
        self.assertEqual(wd.harden(denied, lambda: None), dict(oom=False, mlock=True))
        self.assertEqual(wd.harden(lambda path, value: None, denied), dict(oom=True, mlock=False))

    def test_container_flags_set_a_limit_no_swap_and_the_highest_oom_score(self):
        self.assertEqual(wd.container_flags(100), ['--memory=100g', '--memory-swap=100g', '--oom-score-adj=1000'])
        with self.assertRaises(ValueError):
            wd.container_flags(0)

    def test_the_flags_subcommand_prints_them(self):
        lines = []
        self.assertEqual(wd.main(['flags', '--memory-gib', '90'], say=lines.append), 0)
        self.assertEqual(lines, ['--memory=90g --memory-swap=90g --oom-score-adj=1000'])


class HandbackTests(unittest.TestCase):
    def test_both_floors_must_hold_after_the_drop(self):
        dropped = []
        drop = lambda directory: dropped.append(directory) or 3
        info = lambda free, avail: (lambda: dict(MemFree=int(free * GIB), MemAvailable=int(avail * GIB)))
        ok = wd.handback_ready(info(105, 115), ['a', 'b'], drop, 112.0, 100.0)
        self.assertTrue(ok['ready'])
        self.assertEqual((ok['dropped'], dropped), (6, ['a', 'b']))
        self.assertFalse(wd.handback_ready(info(60, 115), ['a'], drop, 112.0, 100.0)['ready'])      # page cache counts as available, not free
        self.assertFalse(wd.handback_ready(info(105, 100), ['a'], drop, 112.0, 100.0)['ready'])
        self.assertTrue(wd.handback_ready(info(100, 112), [], drop, 112.0, 100.0)['ready'])          # exactly at the floors

    def test_the_subcommand_needs_both_floors_and_sets_the_exit_code(self):
        lines = []
        reader = lambda: dict(MemFree=int(50 * GIB), MemAvailable=int(120 * GIB))
        self.assertEqual(wd.handback_main(['--avail-floor-gib', '112', '--free-floor-gib', '100'], lines.append, reader, lambda d: 0), 1)
        self.assertTrue(lines[0].startswith('not_ready'))
        self.assertEqual(wd.handback_main(['--avail-floor-gib', '112', '--free-floor-gib', '40'], lines.append, reader, lambda d: 0), 0)
        self.assertTrue(lines[1].startswith('ready'))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            wd.handback_main(['--avail-floor-gib', '112'], lines.append)

    def test_drop_file_cache_counts_files(self):
        root = tempfile.mkdtemp()
        try:
            with open(os.path.join(root, 'x'), 'wb') as handle:
                handle.write(b'x')
            self.assertIn(wd.drop_file_cache(root), (0, 1))
        finally:
            shutil.rmtree(root)


class MainLoopTests(unittest.TestCase):
    """The CLI loop with every host dependency injected: a trip is not over until the container is confirmed gone."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.clock = [1000.0]
        self.sleeps = 0
        self.lines = []
        self.original_read = wd.read_text
        wd.read_text = self.fake_read
        self.mem = [100.0]

    def tearDown(self):
        wd.read_text = self.original_read
        shutil.rmtree(self.root)

    def fake_read(self, path):
        if path == '/proc/meminfo':
            avail = self.mem.pop(0) if len(self.mem) > 1 else self.mem[0]
            return 'MemFree: %d kB%sMemAvailable: %d kB%s' % (int(50 * GIB), NL, int(avail * GIB), NL)
        if path == '/proc/vmstat':
            return 'pswpin 0' + NL
        return self.original_read(path)

    def sleep(self, seconds):
        self.sleeps += 1
        self.clock[0] += 1.0
        if self.sleeps > 50:
            raise AssertionError('the loop did not end')

    def now(self):
        return self.clock[0]

    class Quiet(object):
        def start(self):
            pass

        def stop(self):
            pass

        def drain(self):
            return []

        def read(self):
            return ''

    def run_main(self, argv, killer_dead_after, mode='kill'):
        outer = self

        class FakeKiller(object):
            def __init__(self, name):
                self.name, self.dead, self.attempts, self.escalations, self.calls = name, False, 0, 0, 0

            def start(self):
                self.calls += 1
                if self.calls >= killer_dead_after:
                    self.attempts += 1
                    self.dead = True

            def note_pid(self):
                pass
        outer.killer = None

        def make_killer(name):
            outer.killer = FakeKiller(name)
            return outer.killer
        return wd.main(argv + ['--container', 'ours', '--deadline', '1e12', '--mode', mode, '--poll-seconds', '0'], say=self.lines.append,
                       sleep=self.sleep, now=self.now, hardener=lambda: dict(oom=True, mlock=False), make_killer=make_killer,
                       make_kernel=lambda: self.Quiet(), make_events=lambda: self.Quiet())

    def test_kill_mode_needs_a_heartbeat_file(self):
        self.assertEqual(self.run_main([], 1), 2)
        self.assertEqual(self.lines, ['refused: kill mode needs --heartbeat-file'])

    def test_a_healthy_host_beats_and_a_low_one_trips_and_waits_for_the_kill(self):
        beat = os.path.join(self.root, 'beat')
        self.mem = [100.0, 100.0, 5.0]
        code = self.run_main(['--heartbeat-file', beat, '--trip-file', os.path.join(self.root, 'trip')], killer_dead_after=3)
        self.assertEqual(code, 3)
        self.assertTrue(self.lines[0].startswith('hardened: oom=yes mlock=no'))
        self.assertIn('trip: avail_low (kill)', self.lines)
        self.assertTrue(any(line.startswith('killed: ') for line in self.lines))
        self.assertGreaterEqual(self.killer.calls, 3)                # the trip started it once, the loop kept asking until it was dead
        with open(os.path.join(self.root, 'trip')) as handle:
            self.assertEqual(handle.read().strip(), 'avail_low')

    def test_observe_mode_returns_at_the_first_trip_without_a_kill(self):
        self.mem = [5.0]
        code = self.run_main([], killer_dead_after=99, mode='observe')
        self.assertEqual(code, 3)
        self.assertEqual(self.killer.calls, 0)


if __name__ == '__main__':
    unittest.main()
