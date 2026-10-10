"""Op-fusion programme, host-gap package WPH, lever 4: the host-gap instrument (hostgap_instr, under QWEN_FAST_TP4_HOSTGAP_LOG=1 and the probe flag
QWEN_FAST_TP4_HOSTGAP_PROBE=1; new lines only) and its reader (hostgap_read).

Pinned here:
  - the flags are 0 or 1 (the probe needs the log flag, the period is a positive integer), and with the log flag off nothing is imported, installed or written;
  - the wrappers on the host calls pass their arguments and results through, time wall and thread CPU, count the long calls and the longest, tell a blocking trace from a
    non-blocking one, are installed once and are taken off again;
  - a span writes one line with the host's own account (wall, CPU, context switches, faults, collections, the calls made and the longest of each) and never raises or
    changes what runs inside it; nested spans each have their own longest call;
  - the step line is written once an entry, for the interval that ended, and names a probe round;
  - the probe: every PROBE_EVERY-th two-quad round of the coordinator does ONE timed synchronize right after the launches and writes its line (2Q from the first non-blocking
    enqueue); no other round does, and nothing at all without both flags;
  - the real block under the log flag writes the verify stage, the readback, the window's pre-stages, the fence and the collect spans;
  - the paired timing reader drops a probe round and keeps its neighbours;
  - hostgap_read classifies a slow span by its cause (collections, a blocked copy, descheduled, CPU bound, wait) and drops the probe steps from every table."""

import contextlib
import gc
import io
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dflash_packed_proposal_coordinator as coordinator
import hostgap_instr as instr
import hostgap_read as reader
import packed_verifier
import test_verify_prestage as tvp
import verify_prestage
import w2ln_timing_compare as timing

HERE = Path(__file__).resolve().parent
LOG, PROBE, EVERY = 'QWEN_FAST_TP4_HOSTGAP_LOG', 'QWEN_FAST_TP4_HOSTGAP_PROBE', 'QWEN_FAST_TP4_HOSTGAP_PROBE_EVERY'


class Fake:
    """A ttnn namespace with the calls the wrappers time."""

    def __init__(self):
        self.calls = []

    def copy_host_to_device_tensor(self, host, device, cq_id=None):
        self.calls.append(('copy', host, device))
        return 'copied'

    def from_torch(self, value, **kwargs):
        self.calls.append(('up', value, kwargs))
        return 'uploaded'

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        self.calls.append(('trace', blocking))
        return 'ran'

    def synchronize_device(self, mesh):
        self.calls.append(('fence', mesh))

    def to_torch(self, value, mesh_composer=None):
        self.calls.append(('read', value))
        return 'read'

    def copy_device_to_host_tensor(self, device, host, blocking=True, cq_id=None):
        self.calls.append(('d2h', blocking))


class Clean(unittest.TestCase):
    def setUp(self):
        environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
        patcher = patch.dict(os.environ, environ, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        instr.reset()
        self.addCleanup(instr.reset)
        self.addCleanup(instr.uninstall)
        self.log = []
        for target in (verify_prestage,):
            logger = patch.object(target, 'log_line', side_effect=self.log.append)
            logger.start()
            self.addCleanup(logger.stop)

    def lines(self, marker):
        return [line for line in self.log if line.startswith(marker)]


class FlagTests(Clean):
    def test_the_flags_are_zero_or_one_the_probe_needs_the_log_and_the_period_is_a_positive_integer(self):
        self.assertFalse(instr.log_enabled({}))
        self.assertFalse(instr.probe_enabled({PROBE: '1'}))
        self.assertTrue(instr.probe_enabled({LOG: '1', PROBE: '1'}))
        for name in (LOG, PROBE):
            with self.assertRaises(ValueError):
                instr._flag(name, {name: 'yes'})
        self.assertEqual(instr.probe_every({}), 16)
        self.assertEqual(instr.probe_every({EVERY: '4'}), 4)
        for bad in ('0', '-1', 'x', ''):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                instr.probe_every({EVERY: bad})

    def test_without_the_log_flag_a_span_is_a_null_context_and_the_module_is_not_imported(self):
        sys.modules.pop('hostgap_instr', None)
        try:
            with verify_prestage.hostgap_span('anything', block='A') as entered:
                self.assertIsNone(entered)
            self.assertNotIn('hostgap_instr', sys.modules)
            self.assertEqual(self.log, [])
        finally:
            sys.modules['hostgap_instr'] = instr


class WrapperTests(Clean):
    def test_the_wrappers_pass_arguments_and_results_through_and_count_the_calls(self):
        ops = Fake()
        names = instr.install(ops)
        self.assertEqual(sorted(names), sorted(attribute for attribute, kind in instr.CALLS))
        self.assertEqual(ops.copy_host_to_device_tensor('h', 'd', cq_id=1), 'copied')
        self.assertEqual(ops.from_torch('v', dtype='int32'), 'uploaded')
        self.assertEqual(ops.execute_trace('m', 't', cq_id=0, blocking=False), 'ran')
        self.assertEqual(ops.execute_trace('m', 't', 0, False), 'ran')
        self.assertEqual(ops.execute_trace('m', 't'), 'ran')
        ops.synchronize_device('m')
        self.assertEqual(ops.to_torch('x', mesh_composer='c'), 'read')
        self.assertEqual([call[0] for call in ops.calls], ['copy', 'up', 'trace', 'trace', 'trace', 'fence', 'read'])
        self.assertEqual([ops.calls[2][1], ops.calls[3][1], ops.calls[4][1]], [False, False, True])
        counts = {kind: values[0] for kind, values in instr.METER.calls.items()}
        self.assertEqual(counts, dict(copy=1, up=1, tnb=2, tb=1, fence=1, read=1, d2h=0))
        self.assertIsNotNone(instr.METER.first_enqueue)
        self.assertEqual(self.lines(instr.ENGAGED_MARKER)[0].split('wrapped=')[1].count(','), 5, 'the six calls this namespace has')

    def test_a_wrapped_function_keeps_the_attributes_of_the_original(self):
        class Operation:
            golden_function = staticmethod(lambda: 'golden')
            __name__ = 'to_torch'

            def __call__(self, value, mesh_composer=None):
                return 'read'

        ops = Fake()
        ops.to_torch = Operation()
        instr.install(ops)
        self.assertEqual(ops.to_torch.golden_function(), 'golden')
        self.assertEqual(ops.to_torch('x'), 'read')
        instr.uninstall(ops)
        self.assertIsInstance(ops.to_torch, Operation)

    def test_installing_twice_wraps_once_and_uninstall_puts_the_functions_back(self):
        ops = Fake()
        original = ops.copy_host_to_device_tensor
        first = instr.install(ops)
        self.assertEqual(instr.install(ops), [])
        self.assertTrue(first)
        ops.copy_host_to_device_tensor('h', 'd')
        self.assertEqual(instr.METER.calls['copy'][0], 1)
        instr.uninstall(ops)
        self.assertEqual(ops.copy_host_to_device_tensor, original)
        ops.copy_host_to_device_tensor('h', 'd')
        self.assertEqual(instr.METER.calls['copy'][0], 1)

    def test_a_raising_call_is_counted_and_raises_the_same_error(self):
        ops = Fake()

        def broken(*args, **kwargs):
            raise RuntimeError('queue full')

        ops.synchronize_device = broken
        instr.install(ops)
        with self.assertRaises(RuntimeError) as raised:
            ops.synchronize_device('m')
        self.assertEqual(str(raised.exception), 'queue full')
        self.assertEqual(instr.METER.calls['fence'][0], 1)

    def test_a_long_call_is_counted_and_the_longest_of_each_span_is_its_own(self):
        ops = Fake()
        slow = ops.copy_host_to_device_tensor

        def sleeping(host, device, cq_id=None):
            time.sleep(0.0015)
            return slow(host, device, cq_id)

        ops.copy_host_to_device_tensor = sleeping
        instr.install(ops)
        with instr.span('outer') as outer:
            ops.copy_host_to_device_tensor('h', 'd')
            with instr.span('inner') as inner:
                ops.copy_host_to_device_tensor('h', 'd')
            ops.synchronize_device('m')
        self.assertEqual(instr.METER.calls['copy'][3], 2, 'both copies were over 1 ms')
        self.assertGreaterEqual(inner['max']['copy'], 0.0015)
        self.assertGreaterEqual(outer['max']['copy'], 0.0015)
        self.assertLess(inner['max'].get('fence', 0.0), 0.0015)
        self.assertIn('copy_gt1=1', self.lines(instr.SPAN_MARKER)[0], 'the inner span saw one long copy')
        self.assertIn('copy_gt1=2', self.lines(instr.SPAN_MARKER)[1])


class SpanTests(Clean):
    def test_a_span_writes_the_hosts_own_account(self):
        ops = Fake()
        instr.install(ops)
        with instr.span('prestage', block='A') as frame:
            ops.copy_host_to_device_tensor('h', 'd')
            ops.copy_host_to_device_tensor('h', 'd')
            ops.synchronize_device('m')
            gc.collect()
        lines = self.lines(instr.SPAN_MARKER)
        self.assertEqual(len(lines), 1)
        fields = reader.fields(lines[0][len(instr.SPAN_MARKER):])
        self.assertEqual((fields['name'], fields['block'], fields['copy_n'], fields['fence_n']), ('prestage', 'A', 2.0, 1.0))
        for key in ('wall_ms', 'cpu_ms', 'nvcsw', 'nivcsw', 'minflt', 'majflt', 'gc_n', 'gc_ms', 'gc_max_ms', 'copy_ms', 'copy_cpu_ms', 'copy_max_ms', 'copy_gt1', 'fence_max_ms'):
            self.assertIn(key, fields)
        self.assertGreaterEqual(fields['gc_n'], 1.0, 'the collection inside the span is counted')
        self.assertNotIn('read_n', fields, 'a call the span did not make is not written')

    def test_a_span_never_raises_and_never_changes_what_runs_inside_it(self):
        with self.assertRaises(KeyError):
            with instr.span('x'):
                raise KeyError('inside')
        self.assertEqual(len(self.lines(instr.SPAN_MARKER)), 1)
        with patch.object(instr, 'log_line', side_effect=RuntimeError('logger broke')):
            with instr.span('y'):
                pass
        self.assertEqual(instr.METER.frames, [])

    def test_the_step_line_is_written_for_the_interval_that_ended_and_names_a_probe(self):
        self.assertIsNone(instr.note_step())
        with instr.span('a'):
            pass
        line = instr.note_step()
        self.assertRegex(line, r'^\[PACKED-HOSTGAP-ROUND\] step=1 probe=0 wall_ms=[0-9.]+ cpu_ms=[0-9.]+ nvcsw=\d+ nivcsw=\d+ minflt=\d+ majflt=\d+ gc_n=\d+ gc_ms=[0-9.]+')
        instr.METER.probed = True
        line = instr.note_step()
        self.assertRegex(line, r'^\[PACKED-HOSTGAP-ROUND\] step=2 probe=1 ')
        self.assertFalse(instr.METER.probed)
        self.assertEqual(len(self.lines(instr.ROUND_MARKER)), 2)
        self.assertEqual(self.lines(instr.SPAN_MARKER)[0].split()[1], 'step=1')

    def test_the_usage_counters_are_integers_and_zero_where_the_host_cannot_say(self):
        self.assertTrue(all(isinstance(value, int) for value in instr.usage()))
        with patch.object(instr, 'resource', None):
            self.assertEqual(instr.usage(), (0, 0, 0, 0))

    def test_entry_line_writes_the_step_line_only_under_the_flag(self):
        verify_prestage.note_scratch('entry', dict(admit_ms=1.0, update_states_ms=1.0, storage_ms=1.0, reservation_ms=1.0, refresh_ms=1.0, refresh_writes=0, started=0.0))
        verify_prestage.entry_line(1.0)
        self.assertEqual(self.lines(instr.ROUND_MARKER), [])
        os.environ[LOG] = '1'
        for _ in range(2):
            verify_prestage.note_scratch('entry', dict(admit_ms=1.0, update_states_ms=1.0, storage_ms=1.0, reservation_ms=1.0, refresh_ms=1.0, refresh_writes=0, started=0.0))
            verify_prestage.entry_line(1.0)
        self.assertEqual(len(self.lines(instr.ROUND_MARKER)), 1, 'the first entry only opens an interval')


class ProbeTests(Clean):
    def setUp(self):
        super().setUp()
        os.environ.update({'QWEN_FAST_QUAD_DRAFT': '1', 'QWEN_FAST_QUAD_DRAFT_BLOCKS': '2'})
        self.coordinator = coordinator.PackedProposalCoordinator()
        self.fences = []
        fence = (SimpleNamespace(synchronize_device=lambda mesh: self.fences.append(mesh)), 'mesh')
        self.quads = 2
        quads = self

        class Trace:
            def run_audit(self, number):
                pass

        def prepare_quad_blocks(coord, groups, by_slot, round_number, batched, prepared):
            return [dict(fence=fence, trace=Trace(), devices=[]) for _ in range(quads.quads)], []

        self.patches = [patch.object(coordinator.PackedProposalCoordinator, '_prepare_quad_blocks', prepare_quad_blocks)]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def bridges(self, count=8):
        return [SimpleNamespace(request=SimpleNamespace(session=SimpleNamespace(seed=index)), slot=index) for index in range(count)]

    def rounds(self, count):
        with patch('serving_worker_hook.pipelined_device', lambda bridge: SimpleNamespace(pool_slot=SimpleNamespace(index=bridge.slot))):
            for _ in range(count):
                self.coordinator.prepare(self.bridges())

    def test_every_sixteenth_two_quad_round_does_one_timed_synchronize_after_the_launches(self):
        os.environ.update({LOG: '1', PROBE: '1'})
        self.rounds(33)
        probes = self.lines(instr.PROBE_MARKER)
        self.assertEqual(len(probes), 2)
        self.assertRegex(probes[0], r'^\[PACKED-HOSTGAP-PROBE\] round=16 step=0 quads=2 every=16 launch_ms=[0-9.]+ sync_ms=[0-9.]+ total_ms=[0-9.]+ first_enqueue_ms=- twoq_ms=-$')
        self.assertEqual([line.split()[1] for line in probes], ['round=16', 'round=32'])
        self.assertEqual(len(self.fences), 33 + 2, 'the round fence every round, and the probe\'s synchronize on two')
        self.assertEqual(len(self.lines(instr.ROUND_MARKER)), 0)

    def test_the_period_is_the_flag_and_other_rounds_and_other_widths_are_never_probed(self):
        os.environ.update({LOG: '1', PROBE: '1', EVERY: '3'})
        self.rounds(9)
        self.assertEqual(len(self.lines(instr.PROBE_MARKER)), 3)
        self.quads = 1
        self.rounds(9)
        self.assertEqual(len(self.lines(instr.PROBE_MARKER)), 3, 'a round with one quad in flight is not an eight-live round')

    def test_without_both_flags_the_round_is_todays_and_the_module_is_not_touched(self):
        for environ in ({}, {LOG: '1'}, {PROBE: '1'}):
            with self.subTest(environ=environ):
                self.fences.clear()
                self.log.clear()
                with patch.dict(os.environ, environ):
                    with patch.object(instr, 'probe_due', side_effect=AssertionError('the probe is off')):
                        self.rounds(20)
                self.assertEqual(len(self.fences), 20)
                self.assertEqual(self.lines(instr.PROBE_MARKER), [])

    def test_the_probe_names_its_round_and_the_two_q_is_measured_from_the_first_enqueue(self):
        ops = Fake()
        instr.install(ops)
        instr.METER.step = 7
        started = instr.begin_launch()
        ops.execute_trace('m', 't', blocking=False)
        fence = (SimpleNamespace(synchronize_device=lambda mesh: None), 'mesh')
        wait = instr.sync_probe(fence, 48, 2, started)
        self.assertGreaterEqual(wait, 0.0)
        line = self.lines(instr.PROBE_MARKER)[0]
        fields = reader.fields(line[len(instr.PROBE_MARKER):])
        self.assertEqual((fields['round'], fields['step'], fields['quads']), (48.0, 7.0, 2.0))
        self.assertIsInstance(fields['twoq_ms'], float)
        self.assertTrue(instr.METER.probed)


class RealBlockTests(tvp.PrestageFixture):
    def test_the_block_writes_the_verify_spans_and_the_windows_prestage_span_under_the_log_flag(self):
        os.environ[LOG] = '1'
        instr.reset()
        self.addCleanup(instr.reset)
        self.addCleanup(instr.uninstall)
        block = self.open_block()
        users = tvp.base_users()
        self.round(block, users)
        users = tvp.advanced(users, 2)
        self.window(block, users)
        self.round(block, users)
        names = [re.search(r'name=(\S+)', line).group(1) for line in self.h1a if line.startswith(instr.SPAN_MARKER)]
        self.assertEqual(names.count('verify_stage'), 2)
        self.assertEqual(names.count('prestage'), 1)
        stage = [line for line in self.h1a if line.startswith(instr.SPAN_MARKER) and 'name=verify_stage' in line][0]
        self.assertRegex(stage, r' block=\S+ wall_ms=')
        self.assertTrue(any('copy_n=' in line for line in self.h1a if 'name=prestage' in line), 'the pre-stage span carries the copies it made')

    def test_without_the_flag_no_span_line_is_written(self):
        block = self.open_block()
        users = tvp.base_users()
        self.round(block, users)
        self.window(block, tvp.advanced(users, 2))
        self.assertEqual([line for line in self.h1a if line.startswith('[PACKED-HOSTGAP')], [])


class TimingReaderTests(unittest.TestCase):
    LOG = '\n'.join([
        '2026-10-10 10:00:00.000 | INFO | [PHASE] execute total=64 new=0 cached=8 spec=8',
        ] + ['2026-10-10 10:00:00.100 | INFO | [PACKED] request=r%d position=%d' % (index, 4000 + index) for index in range(8)] + [
        '2026-10-10 10:00:00.150 | INFO | [PHASE] execute total=64 new=0 cached=8 spec=8',
        ] + ['2026-10-10 10:00:00.200 | INFO | [PACKED] request=r%d position=%d' % (index, 4000 + index) for index in range(8)] + [
        '2026-10-10 10:00:00.220 | INFO | [PACKED-HOSTGAP-PROBE] round=16 step=2 quads=2 every=16 launch_ms=1.0 sync_ms=20.0 total_ms=21.0 first_enqueue_ms=- twoq_ms=-',
        '2026-10-10 10:00:00.400 | INFO | [PHASE] execute total=64 new=0 cached=8 spec=8',
        ] + ['2026-10-10 10:00:00.450 | INFO | [PACKED] request=r%d position=%d' % (index, 4000 + index) for index in range(8)] + [
        '2026-10-10 10:00:00.550 | INFO | [PHASE] execute total=64 new=0 cached=8 spec=8',
    ])

    def test_a_probe_round_is_not_a_round_time_and_its_neighbours_are_kept(self):
        rounds, dropped = timing.timed_rounds(self.LOG)
        self.assertEqual([round(item['seconds'], 3) for item in rounds], [0.15, 0.15])
        self.assertEqual(dropped, 0, 'a probe round is not counted as a Lever N drop')
        rounds, dropped = timing.timed_rounds(self.LOG.replace('[PACKED-HOSTGAP-PROBE]', '[PACKED-HOSTGAP-OTHER]'))
        self.assertEqual(len(rounds), 3)


class ReaderTests(unittest.TestCase):
    def span(self, step, wall, cpu, **extra):
        extra.setdefault('nivcsw', 0)
        extra.setdefault('gc_ms', 0.0)
        text = '[PACKED-HOSTGAP-SPAN] step=%d name=launch wall_ms=%.2f cpu_ms=%.2f nvcsw=0 nivcsw=%d minflt=0 majflt=0 gc_n=%d gc_ms=%.2f' % (
            step, wall, cpu, extra['nivcsw'], 1 if extra['gc_ms'] else 0, extra['gc_ms'])
        for key in ('copy_max_ms', 'fence_max_ms'):
            if key in extra:
                text += ' %s=%.2f' % (key, extra[key])
        return '2026-10-10 10:00:00.000 | INFO | ' + text

    def test_each_slow_span_gets_the_cause_the_hosts_own_account_names(self):
        lines = [self.span(step, 2.0, 1.9) for step in range(1, 40)]
        lines += [self.span(100, 6.0, 5.9, gc_ms=3.9)]                        # collections took the excess
        lines += [self.span(101, 6.0, 0.4, copy_max_ms=3.9)]                  # one copy blocked while the CPU idled
        lines += [self.span(102, 6.0, 2.0, nivcsw=3)]                         # the scheduler took the CPU
        lines += [self.span(103, 6.0, 5.8)]                                   # busy
        lines += [self.span(104, 6.0, 2.5)]                                   # waiting outside the timed calls
        report = reader.analyse(reader.parse('\n'.join(lines)))
        slow = report['slow']['launch']
        self.assertEqual(slow['n'], 44)
        self.assertEqual(slow['causes'], {'gc': 1, 'blocked-copy': 1, 'descheduled': 1, 'cpu-bound': 1, 'wait': 1})
        self.assertAlmostEqual(report['slow_step_share'], 5 / 44.0)
        self.assertAlmostEqual(slow['median_ms'], 2.0)

    def test_probe_steps_are_dropped_from_every_table(self):
        lines = [self.span(step, 2.0, 1.9) for step in range(1, 40)] + [self.span(7, 30.0, 1.0)]
        lines += ['[PACKED-HOSTGAP-PROBE] round=16 step=7 quads=2 every=16 launch_ms=1.0 sync_ms=20.0 total_ms=21.0 first_enqueue_ms=1.50 twoq_ms=19.5']
        lines += ['[PACKED-HOSTGAP-ROUND] step=%d probe=%d wall_ms=%.2f cpu_ms=10.00 nvcsw=0 nivcsw=0 minflt=0 majflt=0 gc_n=0 gc_ms=0.00' % (step, int(step == 7), 150.0 if step != 7 else 300.0)
                  for step in range(1, 12)]
        report = reader.analyse(reader.parse('\n'.join(lines)))
        self.assertEqual(report['probe_steps'], [7])
        self.assertEqual(report['spans']['launch']['wall_ms']['n'], 38)
        self.assertEqual(report['rounds']['wall_ms']['n'], 10)
        self.assertEqual(report['rounds']['wall_ms']['p50'], 150.0)
        self.assertEqual((report['probe']['n'], report['probe']['twoq_ms']['p50']), (1, 19.5))

    def test_the_window_lines_say_how_often_the_window_is_the_critical_path(self):
        text = '\n'.join('[PACKED-HOSTGAP-WINDOW] round=%d blocks=2 stage_window_ms=0.00,0.00 prestage_ms=9.00,9.00 prestaged=1,1 window_ms=18.00 fence_wait_ms=%.2f' % (
            number, 0.4 if number % 4 else 9.0) for number in range(1, 41))
        report = reader.analyse(reader.parse(text))
        self.assertEqual(report['window']['n'], 40)
        self.assertAlmostEqual(report['window']['critical_share'], 0.75)

    def test_the_text_report_and_the_command_line(self):
        lines = [self.span(step, 2.0, 1.9) for step in range(1, 30)]
        path = Path(os.environ.get('TMPDIR', '/tmp')) / 'hostgap_read_test.log'
        path.write_text('\n'.join(lines), encoding='utf-8')
        self.addCleanup(path.unlink)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(reader.main([str(path)]), 0)
        self.assertIn('SPANS', output.getvalue())
        self.assertIn('launch', output.getvalue())
        self.assertIn('steps with at least one slow span: 0.0%', output.getvalue())


class ShippingTests(unittest.TestCase):
    def test_the_hooks_import_the_module_only_behind_the_flag(self):
        for name in ('packed_verifier.py', 'dflash_packed_proposal_coordinator.py', 'verify_prestage.py', 'serving_worker_hook.py'):
            text = (HERE / name).read_text(encoding='utf-8')
            self.assertNotRegex(text, r'(?m)^import hostgap_instr', name)
        self.assertIn("if verify_prestage.hostgap_log_enabled():", (HERE / 'packed_verifier.py').read_text(encoding='utf-8'))

    def test_the_markers_are_the_ones_the_reader_reads(self):
        for marker, key in ((instr.ROUND_MARKER, 'rounds'), (instr.SPAN_MARKER, 'spans'), (instr.PROBE_MARKER, 'probes')):
            self.assertEqual(len(reader.parse(marker + ' step=1')[key]), 1)
        self.assertEqual(timing.HOSTGAP_PROBE, instr.PROBE_MARKER)


if __name__ == '__main__':
    unittest.main()
