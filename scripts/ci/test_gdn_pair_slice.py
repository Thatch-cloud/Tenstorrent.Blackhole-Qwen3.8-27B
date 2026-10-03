"""tp4/gluefix: the GDN pair slice (QWEN_FAST_GDN_PAIR_SLICE), the dispatch diagnostic (QWEN_FAST_GDN_DISPATCH_DIAG), their profiles and the window's templates.

The slicing arithmetic and the op sequence are proven over the packed-segments fixture's fakes (no card here): flag off the twin is the
pinned class call for call; flag on, the odd users' half-tile slices come from ONE row-major conversion (to_layout / slice / to_layout, the
three ops ttnn's own Slice runs for a begin that is not tile-aligned) and nothing else about the layer changes. Bit-exactness on the card is
the audited arm G1 (c2-packed-tp4-gate-pairslice), which compares every shared-conversion slice with ttnn's own on every chip.

Run at py 3.11: `py -3.11 -m unittest test_gdn_pair_slice` from scripts/ci.
"""

import json
import os
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import c2_smoke_check  # noqa: E402
import gdn_device_loop_state as pinned_state  # noqa: E402
import gdn_device_loop_state_tp as twin  # noqa: E402
import gdn_pair_slice_tp as pair_slice  # noqa: E402
import test_gdn_packed_segments  # noqa: E402
import tp4_vglue  # noqa: E402
from test_gdn_packed_segments import PackedFixture  # noqa: E402
from test_tp4_vglue_twin import OUTPUT_WIDTH, TwinTests, env, four, moves  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FOLDER = HERE / 'references' / 'tp4-gluefix-jobs'
with open(HERE / 'qwen_c2_profiles.json', encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)['profiles']
NAMES = sorted(PROFILES)
SLICE, GLUE, DIAG, AUDIT = 'QWEN_FAST_GDN_PAIR_SLICE', 'QWEN_FAST_TP4_GDN_GLUE', 'QWEN_FAST_GDN_DISPATCH_DIAG', 'QWEN_FAST_TP4_VGLUE_AUDIT'
CONTROL, TIMED_ARM, AUDITED_ARM, DIAG_ARM = ('c2-packed-tp4-speed-strace', 'c2-packed-tp4-speed-strace-pairslice',
                                             'c2-packed-tp4-gate-pairslice', 'c2-packed-tp4-speed-strace-dispatchdiag')
IMAGE = 'tp4-gluefix-1'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')


class HalfTileTests(unittest.TestCase):
    """The slicing arithmetic: which slices the shared conversion serves."""

    def test_the_odd_users_of_a_four_user_block_are_served(self):
        for user in (1, 3):
            start = user * 16
            self.assertEqual(pair_slice.half_tile_rows((0, start, 0), (1, start + 16, 8240), (1, 64, 8240)), start)

    def test_the_tile_aligned_users_are_left_to_ttnn(self):
        for user in (0, 2):
            start = user * 16
            self.assertIsNone(pair_slice.half_tile_rows((0, start, 0), (1, start + 16, 8240), (1, 64, 8240)))

    def test_every_other_shape_is_left_to_ttnn(self):
        shape = (1, 64, 8240)
        cases = [
            ((0, 16, 0), (1, 48, 8240), shape),          # not 16 rows
            ((0, 16, 0), (1, 32, 4096), shape),          # not the whole width
            ((0, 16, 32), (1, 32, 8240), shape),         # not from column 0
            ((1, 16, 0), (2, 32, 8240), (2, 64, 8240)),  # not from batch 0
            ((0, 8, 0), (1, 24, 8240), shape),           # a quarter-tile start
            ((0, 16, 0), (1, 32, 8240), (1, 48, 8240)),  # a block that is not whole tiles (48 rows)
            ((0, 16), (1, 32), (64, 8240)),              # two axes
            ((0, 48, 0), (1, 64, 8240), (1, 56, 8240)),  # past the end
        ]
        for starts, ends, found in cases:
            with self.subTest(starts=starts, ends=ends, shape=found):
                self.assertIsNone(pair_slice.half_tile_rows(starts, ends, found))

    def test_a_larger_block_serves_every_odd_half_tile(self):
        served = [row for row in range(0, 128, 16)
                  if pair_slice.half_tile_rows((0, row, 0), (1, row + 16, 64), (1, 128, 64)) is not None]
        self.assertEqual(served, [16, 48, 80, 112])


class FlagTests(unittest.TestCase):
    def test_strict_zero_one_and_off_by_default(self):
        with four():
            self.assertFalse(tp4_vglue.pair_slice_enabled({'QWEN_FAST_TP': '4'}))
            self.assertFalse(tp4_vglue.pair_slice_enabled({'QWEN_FAST_TP': '4', SLICE: '0'}))
            self.assertTrue(tp4_vglue.pair_slice_enabled({'QWEN_FAST_TP': '4', SLICE: '1'}))
            self.assertFalse(tp4_vglue.dispatch_diag_enabled({'QWEN_FAST_TP': '4'}))
            self.assertTrue(tp4_vglue.dispatch_diag_enabled({'QWEN_FAST_TP': '4', DIAG: '1'}))
            for name, call in ((SLICE, tp4_vglue.pair_slice_enabled), (DIAG, tp4_vglue.dispatch_diag_enabled)):
                with self.assertRaises(ValueError):
                    call({'QWEN_FAST_TP': '4', name: 'yes'})

    def test_both_are_refused_at_the_pair(self):
        for name, call in ((SLICE, tp4_vglue.pair_slice_enabled), (DIAG, tp4_vglue.dispatch_diag_enabled)):
            for pair_env in ({}, {'QWEN_FAST_TP': '2'}):
                with self.assertRaises(ValueError):
                    call(dict(pair_env, **{name: '1'}))

    def test_the_audit_counts_the_pair_slice_as_a_lever_but_not_the_diagnostic(self):
        base = {'QWEN_FAST_TP': '4', AUDIT: '1'}
        self.assertFalse(tp4_vglue.audit_enabled(base))
        self.assertTrue(tp4_vglue.audit_enabled(dict(base, **{SLICE: '1'})))
        self.assertFalse(tp4_vglue.audit_enabled(dict(base, **{DIAG: '1'})))
        self.assertFalse(tp4_vglue.audit_enabled({'QWEN_FAST_TP': '4', SLICE: '1'}))

    def test_the_pair_slice_is_not_one_of_the_levers_and_keeps_the_marker_unchanged(self):
        self.assertNotIn(SLICE, tp4_vglue.LEVERS)
        self.assertNotIn(DIAG, tp4_vglue.LEVERS)
        with four(), env(**{SLICE: '1'}):
            self.assertEqual(tp4_vglue.engaged_levers(), ())

    def test_wrap_does_nothing_with_both_off_and_when_v2_replaces_the_slices(self):
        lines = []
        with four(), patch('gdn_pair_slice_tp.tp4_vglue.log_line', side_effect=lines.append):
            pair_slice._LOGGED.clear()
            self.assertIsNone(pair_slice.wrap('ops', False, {'QWEN_FAST_TP': '4'}))
            self.assertIsNone(pair_slice.wrap('ops', True, {'QWEN_FAST_TP': '4', SLICE: '1'}))
            self.assertEqual(len(lines), 1)
            self.assertIn(pair_slice.IDLE, lines[0])
            self.assertIsNotNone(pair_slice.wrap('ops', False, {'QWEN_FAST_TP': '4', SLICE: '1'}))
            self.assertIsNotNone(pair_slice.wrap('ops', True, {'QWEN_FAST_TP': '4', DIAG: '1'}))


class TwinFakes(PackedFixture):
    spans = ((0, 16), (16, 32), (32, 48), (48, 64))
    slots = TwinTests.slots
    checkpoints = TwinTests.checkpoints
    build_state = TwinTests.build_state
    segment_runner = TwinTests.segment_runner

    def setUp(self):
        tp4_vglue.take()
        twin.DeviceLoopState.glue_fallbacks = set()
        pair_slice._LOGGED.clear()
        pair_slice._DIAG['calls'] = 0
        self.launches, self.lines = [], []
        patcher = patch('gdn_device_loop_state_tp.tp4_vglue.log_line', side_effect=self.lines.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_decode(self, cls, flags=None, runner_error=None):
        state, operations, layer, active, calls = self.build_state(cls)
        # a tile-layout projection and the ops the shared conversion makes, named so the sequence can be read back
        joined = operations.concat.side_effect

        def concat(parts, **keywords):
            value = joined(parts, **keywords)
            value.layout = 'tile'
            return value

        def to_layout(value, layout, **keywords):
            calls.append(('to_layout', value.name, layout))
            return SimpleNamespace(shape=value.shape, name='%s:%s' % (layout, value.name), layout=layout,
                                   memory_config=lambda: 'l1')

        operations.concat = Mock(side_effect=concat)
        operations.to_layout = Mock(side_effect=to_layout)
        operations.ROW_MAJOR_LAYOUT = 'rm'
        runner = self.segment_runner(calls)

        def run(*args, **keywords):
            if runner_error is not None:
                raise runner_error
            return runner(*args, **keywords)

        def launch(mesh, sources, destinations, tasks, **keywords):
            self.launches.append((tuple(source.name for source in sources), tuple(destination.name for destination in destinations)))

        with env(**(flags or {})), patch('gdn_rows_dma_tp.launch', side_effect=launch), \
                patch('gdn_rows_dma_tp.problem', return_value=None), \
                patch('gdn_device_loop_state.run_user_batched_projected', side_effect=run), \
                patch('gdn_device_loop_state_tp.run_user_batched_projected', side_effect=run), \
                patch('gdn_device_loop_state.restore_prefix'), patch('gdn_device_loop_state.release_owned'), \
                patch('gdn_device_loop_state_tp.release_owned'), \
                patch('gdn_device_loop_state.copy_compact', side_effect=lambda s, d: calls.append(('copy_compact', s[0], d[0]))), \
                patch('gdn_device_loop_state_tp.copy_compact', side_effect=lambda s, d: calls.append(('copy_compact', s[0], d[0]))):
            packed = SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1')
            result = state.decode(packed, self.checkpoints, [0] * 4, segments=self.spans, slots=self.slots, deferred=True)
        return result, calls, operations, state


class PairSliceTwinTests(TwinFakes):
    def test_flag_off_is_the_pinned_class_call_for_call_and_never_proxies(self):
        with four():
            expected, expected_calls, _, _ = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState)
        self.assertEqual(calls, expected_calls)
        self.assertEqual(sorted(result), sorted(expected))
        self.assertIs(state.operations, operations)
        operations.to_layout.assert_not_called()
        self.assertEqual(tp4_vglue.take(), {})
        self.assertEqual(self.lines, [])

    def test_flag_zero_is_off(self):
        with four():
            expected, expected_calls, _, _ = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={SLICE: '0', DIAG: '0'})
        self.assertEqual(calls, expected_calls)

    def test_flag_on_makes_one_row_major_conversion_for_both_odd_users(self):
        with four():
            expected, expected_calls, _, _ = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={SLICE: '1'})
        self.assertEqual([entry for entry in calls if entry[0] == 'to_layout' and entry[2] == 'rm'], [('to_layout', 'projected', 'rm')])
        self.assertEqual([entry for entry in calls if entry[0] == 'to_layout' and entry[2] == 'tile'],
                         [('to_layout', 'rm:projected:16', 'tile'), ('to_layout', 'rm:projected:48', 'tile')])
        # the recurrence launch reads: ttnn's tile slices for users 0 and 2, the converted pieces for 1 and 3
        self.assertEqual([entry for entry in calls if entry[0] == 'batched'],
                         [('batched', ('piece:0', 'tile:rm:projected:16', 'piece:32', 'tile:rm:projected:48'))])
        self.assertEqual([entry for entry in expected_calls if entry[0] == 'batched'],
                         [('batched', ('piece:0', 'piece:16', 'piece:32', 'piece:48'))])
        # the row-major slices are the odd users' rows, every other slice is the pinned one
        slices = [entry for entry in calls if entry[0] == 'slice']
        self.assertEqual(slices, [('slice', 0, 32), ('slice', 32, 64), ('slice', 0, 16), ('slice', 16, 32), ('slice', 32, 48),
                                  ('slice', 48, 64)])
        self.assertEqual(sorted(slices), sorted(entry for entry in expected_calls if entry[0] == 'slice'))
        # every state move, in the pinned order; same keys and the same output block
        self.assertEqual(moves(calls), moves(expected_calls))
        self.assertEqual(sorted(result), sorted(expected))
        self.assertEqual(result['output'].name, expected['output'].name)
        self.assertEqual(result['segments'], expected['segments'])
        self.assertEqual(tp4_vglue.take(), {'gdn_pair_slice': 2})
        self.assertEqual([line for line in self.lines if pair_slice.ENGAGED in line], ['%s site=gdn_slice' % pair_slice.ENGAGED])

    def test_the_shared_conversion_is_freed_and_the_real_operations_come_back(self):
        with four():
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={SLICE: '1'})
        self.assertIs(state.operations, operations)
        frees = [entry[1] for entry in calls if entry[0] == 'free']
        self.assertIn('rm:projected', frees)
        self.assertEqual(frees.count('rm:projected'), 1)
        # the intermediate row-major slices are freed as soon as they are tilized
        self.assertIn('rm:projected:16', frees)
        self.assertIn('rm:projected:48', frees)

    def test_the_shared_conversion_is_freed_after_the_last_slice_not_at_close(self):
        with four():
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={SLICE: '1'})
        names = [(entry[0], entry[1] if entry[0] == 'free' else None) for entry in calls]
        freed = names.index(('free', 'rm:projected'))
        last_slice = max(index for index, entry in enumerate(calls) if entry[0] == 'slice')
        batched = [index for index, entry in enumerate(calls) if entry[0] == 'batched'][0]
        self.assertGreater(freed, last_slice)
        self.assertLess(freed, batched)

    def test_the_pieces_are_owned_like_the_slices(self):
        with four():
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={SLICE: '1'})
        names = [getattr(value, 'name', None) for value in result['owned']]
        for name in ('piece:0', 'tile:rm:projected:16', 'piece:32', 'tile:rm:projected:48'):
            self.assertIn(name, names)

    def test_the_operations_come_back_and_the_conversion_is_freed_when_the_call_raises(self):
        with four(), self.assertRaises(RuntimeError):
            self.run_decode(twin.DeviceLoopState, flags={SLICE: '1'}, runner_error=RuntimeError('launch'))
        # no handle on the state when it raised: the rebuild below proves the proxy never leaks into a second call
        with four():
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={SLICE: '1'})
        self.assertIs(state.operations, operations)

    def test_the_raise_path_frees_the_conversion_and_restores_the_operations(self):
        state, operations, layer, active, calls = self.build_state(twin.DeviceLoopState)
        operations.to_layout = Mock(side_effect=lambda value, layout, **kw: SimpleNamespace(
            shape=value.shape, name='%s:%s' % (layout, value.name), layout=layout, memory_config=lambda: 'l1'))
        operations.ROW_MAJOR_LAYOUT = 'rm'
        joined = operations.concat.side_effect

        def concat(parts, **keywords):
            value = joined(parts, **keywords)
            value.layout = 'tile'
            return value

        operations.concat = Mock(side_effect=concat)
        packed = SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1')
        with four(), env(**{SLICE: '1'}), \
                patch('gdn_device_loop_state.run_user_batched_projected', side_effect=RuntimeError('launch')), \
                patch('gdn_device_loop_state.release_owned'), patch('gdn_device_loop_state_tp.release_owned'), \
                patch('gdn_device_loop_state.copy_compact'), patch('gdn_device_loop_state_tp.copy_compact'):
            with self.assertRaises(RuntimeError):
                state.decode(packed, self.checkpoints, [0] * 4, segments=self.spans, slots=self.slots, deferred=True)
        self.assertIs(state.operations, operations)
        self.assertEqual([entry[1] for entry in calls if entry[0] == 'free'].count('rm:projected'), 1)

    def test_v2_on_leaves_the_slices_to_it_and_says_so(self):
        with four():
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={SLICE: '1', GLUE: '1'})
        operations.to_layout.assert_not_called()
        self.assertEqual(len(self.launches), 2)
        self.assertEqual(tp4_vglue.take(), {'gdn_glue': 1, 'gdn_merge': 1})
        self.assertTrue([line for line in self.lines if pair_slice.IDLE in line])
        self.assertFalse([line for line in self.lines if pair_slice.FALLBACK in line])

    def test_a_non_tile_projection_falls_back_to_ttnn_and_says_so(self):
        state, operations, layer, active, calls = self.build_state(twin.DeviceLoopState)
        call = pair_slice.Call(operations, pair_slice.PairSlice(operations, False), None)
        row_major = SimpleNamespace(shape=(1, 64, 8240), layout='rm', name='x')
        self.assertIsNone(call.pair.serve(row_major, (0, 16, 0), (1, 32, 8240)))
        self.assertEqual(call.pair.served, 0)
        self.assertEqual(len([line for line in self.lines if pair_slice.FALLBACK in line]), 1)
        self.assertIsNone(call.pair.serve(row_major, (0, 16, 0), (1, 32, 8240)))
        self.assertEqual(len([line for line in self.lines if pair_slice.FALLBACK in line]), 1)  # said once

    def test_extra_arguments_are_never_intercepted(self):
        state, operations, layer, active, calls = self.build_state(twin.DeviceLoopState)
        pair = pair_slice.PairSlice(operations, False)
        proxy = pair_slice.Operations(operations, pair, None)
        tile = SimpleNamespace(shape=(1, 64, 8240), layout='tile', name='projected', memory_config=lambda: 'l1')
        operations.TILE_LAYOUT = 'tile'
        operations.to_layout = Mock()
        proxy.slice(tile, (0, 16, 0), (1, 32, 8240), memory_config='dram')
        operations.to_layout.assert_not_called()
        self.assertEqual(pair.served, 0)


class AuditTests(TwinFakes):
    def test_the_audit_holds_both_sides_of_each_odd_user_and_the_entries_are_expected(self):
        with four():
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={SLICE: '1', AUDIT: '1'})
            entries = result['vglue_pair_audit']
            self.assertEqual([entry['label'] for entry in entries], ['pair slice user 1', 'pair slice user 3'])
            self.assertEqual([entry['mine'].name for entry in entries], ['copy:tile:rm:projected:16', 'copy:tile:rm:projected:48'])
            self.assertEqual([entry['served'].name for entry in entries], ['copy:piece:16', 'copy:piece:48'])
            self.assertEqual(tp4_vglue.expected_entries(result, {'QWEN_FAST_TP': '4', SLICE: '1', AUDIT: '1'}),
                             {'pair slice user 1': 1, 'pair slice user 3': 1})
            self.assertEqual(tp4_vglue.missing_entries(result, {'QWEN_FAST_TP': '4', SLICE: '1', AUDIT: '1'}), [])
            self.assertEqual(len(tp4_vglue.audit_held_of(result)), 4)
        # ttnn's own slice of the audit is freed at once: it is never held, and the held copies are outside `owned`
        self.assertIn('piece:16', [entry[1] for entry in calls if entry[0] == 'free'])
        owned = [getattr(value, 'name', None) for value in result['owned']]
        self.assertFalse([name for name in owned if name and name.startswith('copy:')])

    def test_a_layer_without_the_entries_is_reported(self):
        with four():
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={SLICE: '1'})
            self.assertNotIn('vglue_pair_audit', result)
            missing = tp4_vglue.missing_entries(result, {'QWEN_FAST_TP': '4', SLICE: '1', AUDIT: '1'})
        self.assertEqual(len(missing), 2)
        self.assertIn('pair slice user 1', missing[0])

    def test_without_the_audit_flag_nothing_is_cloned(self):
        with four():
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={SLICE: '1'})
        operations.clone.assert_not_called()

    def test_v2_entries_are_unchanged_when_v2_is_on(self):
        with four():
            expected = tp4_vglue.expected_entries({'segment_results': [{}] * 4}, {'QWEN_FAST_TP': '4', GLUE: '1', SLICE: '1'})
        self.assertEqual(sorted(expected), ['merged output', 'piece user 0', 'piece user 1', 'piece user 2', 'piece user 3'])


class DiagnosticTests(TwinFakes):
    def test_the_recorder_reports_the_ops_and_the_host_gap_between_conv_gates_launches(self):
        recorder = pair_slice.Recorder()
        stamps = iter(range(1000, 100000, 5000))
        with patch('gdn_pair_slice_tp.time.perf_counter_ns', side_effect=lambda: next(stamps)):
            for name in ('slice', 'to_layout', 'gdn_decode_conv_gates', 'slice', 'gdn_decode_conv_gates', 'gdn_decode_conv_gates',
                         'concat'):
                recorder.note(name)
        found = recorder.summary()
        self.assertEqual(found, dict(ops=7, conv_gates=3, ops_between=[1, 0], host_gap_us=[10, 5], host_span_us=30))

    def test_the_transformer_launches_are_recorded_and_still_made(self):
        inner = SimpleNamespace(gdn_decode_conv_gates=Mock(return_value='out'), constant=7)
        operations = SimpleNamespace(transformer=inner, slice=Mock(return_value='piece'), L1_MEMORY_CONFIG='l1', Tensor=dict)
        recorder = pair_slice.Recorder()
        proxy = pair_slice.Operations(operations, None, recorder)
        self.assertEqual(proxy.transformer.gdn_decode_conv_gates('a', b=1), 'out')
        inner.gdn_decode_conv_gates.assert_called_once_with('a', b=1)
        self.assertEqual(proxy.transformer.constant, 7)
        self.assertEqual(proxy.slice('t', (0,), (1,)), 'piece')
        self.assertEqual(proxy.L1_MEMORY_CONFIG, 'l1')
        self.assertIs(proxy.Tensor, dict)           # a class is never wrapped
        self.assertEqual([name for name, _ in recorder.events], ['gdn_decode_conv_gates', 'slice'])

    def test_the_diagnostic_changes_no_op_and_logs_the_first_calls_only(self):
        with four():
            expected, expected_calls, _, _ = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={DIAG: '1'})
            self.assertEqual(calls, expected_calls)
            self.assertIs(state.operations, operations)
            lines = [line for line in self.lines if pair_slice.DIAG in line]
            self.assertEqual(len(lines), 1)
            self.assertRegex(lines[0], r'call=1 ops=\d+ conv_gates=0 ops_between_conv_gates=\[\] host_gap_us=\[\] host_span_us=\d+')
            for _ in range(pair_slice.DIAG_LINES + 3):
                self.run_decode(twin.DeviceLoopState, flags={DIAG: '1'})
        self.assertEqual(len([line for line in self.lines if pair_slice.DIAG in line]), pair_slice.DIAG_LINES)

    def test_both_flags_together_do_both(self):
        with four():
            result, calls, operations, state = self.run_decode(twin.DeviceLoopState, flags={SLICE: '1', DIAG: '1'})
        self.assertEqual([entry for entry in calls if entry[0] == 'to_layout' and entry[2] == 'rm'], [('to_layout', 'projected', 'rm')])
        self.assertTrue([line for line in self.lines if pair_slice.DIAG in line])


class ModuleTests(unittest.TestCase):
    def read(self, *parts):
        return (ROOT.joinpath(*parts)).read_text(encoding='utf-8')

    def test_the_module_is_stdlib_and_tp4_vglue_only_and_lf(self):
        text = (HERE / 'gdn_pair_slice_tp.py').read_text(encoding='utf-8')
        self.assertEqual(sorted(set(re.findall(r'^(?:import|from) ([a-z_0-9]+)', text, flags=re.M))), ['time', 'tp4_vglue'])
        self.assertNotIn(b'\r', (HERE / 'gdn_pair_slice_tp.py').read_bytes())

    def test_the_twin_overrides_decode_packed_and_nothing_pinned_changed(self):
        self.assertIn('_decode_packed', vars(twin.DeviceLoopState))
        source = (HERE / 'gdn_device_loop_state.py').read_text()
        self.assertIn('return self._decode_packed(packed, rows, spans, checkpoints, prefixes, slots, defer_publication, deferred)', source)
        self.assertNotIn('pair_slice', source)

    def test_it_ships_in_every_list_the_other_vglue_modules_travel_in(self):
        self.assertIn('gdn_pair_slice_tp.py', tp4_vglue.RUNTIME_FILES)
        self.assertRegex(self.read('docker', 'qwen-c2-overlay.txt'), r'(?m)^scripts/ci/gdn_pair_slice_tp\.py$')
        self.assertIn('scripts/ci/gdn_pair_slice_tp.py', self.read('docker', 'qwen-fast-serving.Dockerfile'))
        self.assertRegex(self.read('.github', 'workflows', 'qwen-fast-serving-image.yml'), r'extent_attention_fold_tp\.py gdn_pair_slice_tp\.py; do')
        self.assertIn('test_gdn_pair_slice', self.read('.github', 'workflows', 'qwen-integration-cpu.yml'))


def env_of(name):
    return PROFILES[name]['env']


def differing(left, right):
    return {key for key in set(left) | set(right) if left.get(key) != right.get(key)}


class ProfileTests(unittest.TestCase):
    def test_the_timed_arm_is_its_control_plus_exactly_the_flag(self):
        self.assertEqual(differing(env_of(TIMED_ARM), env_of(CONTROL)), {SLICE})
        self.assertEqual(env_of(TIMED_ARM)[SLICE], '1')
        self.assertNotIn(GLUE, [key for key, value in env_of(TIMED_ARM).items() if value == '1'])

    def test_the_diagnostic_arm_is_its_control_plus_exactly_the_flag(self):
        self.assertEqual(differing(env_of(DIAG_ARM), env_of(CONTROL)), {DIAG})
        self.assertEqual(env_of(DIAG_ARM)[DIAG], '1')

    def test_the_audited_arm_is_the_gate_plus_the_flag_and_the_audit(self):
        self.assertEqual(differing(env_of(AUDITED_ARM), env_of('c2-packed-tp4-gate')), {SLICE, AUDIT})
        self.assertEqual((env_of(AUDITED_ARM)['QWEN_FAST_VERIFY_T1_AUDIT'], env_of(AUDITED_ARM)['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'))

    def test_only_the_other_keys_are_equal_and_every_arm_is_gate_only(self):
        for name, base in ((TIMED_ARM, CONTROL), (DIAG_ARM, CONTROL), (AUDITED_ARM, 'c2-packed-tp4-gate')):
            with self.subTest(profile=name):
                for key in set(PROFILES[name]) | set(PROFILES[base]):
                    if key not in ('env', 'description'):
                        self.assertEqual(PROFILES[name].get(key), PROFILES[base].get(key), key)
                self.assertIs(PROFILES[name].get('gate_only'), True)
                self.assertTrue(PROFILES[name]['description'].startswith('GATE ONLY'))

    def test_production_and_the_best_recipe_carry_neither_flag(self):
        for name in ('c2-packed-tp4', 'c2-packed-tp4-best', 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-gate'):
            self.assertNotIn(SLICE, env_of(name), name)
            self.assertNotIn(DIAG, env_of(name), name)
        flagged = sorted(name for name, body in PROFILES.items() if SLICE in body['env'] or DIAG in body['env'])
        self.assertEqual(flagged, sorted([TIMED_ARM, AUDITED_ARM, DIAG_ARM]))


class SmokeCheckTests(unittest.TestCase):
    def slice_problems(self, log, profile=TIMED_ARM):
        problems, _ = c2_smoke_check.check('', log, False, env=env_of(profile))
        return [problem for problem in problems if 'pair slice' in problem.lower()]

    def test_an_engaged_line_passes(self):
        self.assertEqual(self.slice_problems('%s site=gdn_slice' % pair_slice.ENGAGED), [])

    def test_no_engaged_line_fails_the_arm_that_asked_for_it(self):
        found = self.slice_problems('')
        self.assertEqual(len(found), 1)
        self.assertIn(SLICE, found[0])

    def test_an_idle_line_alone_does_not_count_as_engaged(self):
        self.assertEqual(len(self.slice_problems('%s reason=none' % pair_slice.IDLE)), 1)

    def test_a_fall_back_line_fails_even_beside_an_engaged_one(self):
        found = self.slice_problems('%s site=gdn_slice\n%s site=gdn_slice reason=x' % (pair_slice.ENGAGED, pair_slice.FALLBACK))
        self.assertEqual(len(found), 1)
        self.assertIn('fell back', found[0])

    def test_the_control_and_the_diagnostic_arm_are_not_asked_for_the_marker(self):
        self.assertEqual(self.slice_problems('', CONTROL), [])
        self.assertEqual(self.slice_problems('', DIAG_ARM), [])

    def audit_problems(self, log, profile=AUDITED_ARM):
        problems, _ = c2_smoke_check.check('', log, False, env=env_of(profile))
        return [problem for problem in problems if 'nothing was compared' in problem]

    def test_the_audited_arm_needs_a_passing_audit_line(self):
        engaged = '%s site=gdn_slice\n' % pair_slice.ENGAGED
        self.assertEqual(len(self.audit_problems(engaged)), 1)
        self.assertEqual(len(self.audit_problems(engaged + '[PINDIAG] tp4 vglue audit mismatch round=1 x')), 1)
        self.assertEqual(self.audit_problems(engaged + '[PINDIAG] tp4 vglue audit 1 exact=True layers=all entries=8'), [])

    def test_the_timed_arm_is_not_asked_for_an_audit_line(self):
        self.assertEqual(self.audit_problems('%s site=gdn_slice' % pair_slice.ENGAGED, TIMED_ARM), [])

    def test_the_diagnostic_arm_needs_a_diagnostic_line(self):
        def found(log, profile=DIAG_ARM):
            problems, _ = c2_smoke_check.check('', log, False, env=env_of(profile))
            return [problem for problem in problems if 'diagnostic line' in problem]
        self.assertEqual(len(found('')), 1)
        self.assertEqual(found('[PINDIAG] tp4 gdn dispatch diag ops=3'), [])
        self.assertEqual(found('', CONTROL), [])

    def test_the_markers_are_the_modules(self):
        self.assertEqual(c2_smoke_check.PAIR_SLICE_ENGAGED, pair_slice.ENGAGED)
        self.assertEqual(c2_smoke_check.PAIR_SLICE_FELL_BACK, pair_slice.FALLBACK)
        self.assertEqual(c2_smoke_check.PAIR_SLICE_FLAG, SLICE)


def order_rows():
    with open(FOLDER / 'ORDER.txt', encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def parsed(name):
    return job.read_job(job.parse_env((FOLDER / (name + '.env')).read_text(encoding='utf-8')), NAMES)


class BindingTests(unittest.TestCase):
    """The twin only runs if tp_addresses binds it: the new levers must bind it, production and the control must not."""

    def bound_state_row(self, name):
        import tp_addresses
        rows, _modules = tp_addresses.bound_twins(env_of(name))
        return any(row[:2] == ('gdn_device_loop_state', 'DeviceLoopState') for row in rows)

    def test_the_new_profiles_bind_the_device_loop_state_twin(self):
        for name in (TIMED_ARM, AUDITED_ARM, DIAG_ARM):
            with self.subTest(name=name):
                self.assertTrue(self.bound_state_row(name))

    def test_production_and_the_control_leave_it_unbound(self):
        for name in ('c2-packed-tp4', CONTROL):
            with self.subTest(name=name):
                self.assertFalse(self.bound_state_row(name))


ORDERED = ('B0-build', 'A0-agentstop-unserve', 'G1-pairslice-exact-audited', 'GT1-control-timed', 'GT2-pairslice-timed',
           'GT3-control-timed', 'GT4-pairslice-timed', 'DD1-dispatchdiag-timed', 'H1-handback-reset-fabric-agentstart')
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')


class WindowTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_in_this_order_with_one_image(self):
        rows = order_rows()
        self.assertTrue(all(len(row) == 4 for row in rows))
        self.assertEqual([row[0] for row in rows], list(ORDERED))
        self.assertEqual(sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env')), sorted(ORDERED))
        modes = {row[0]: row[1] for row in rows}
        for name, mode, image, minutes in rows:
            self.assertIn(mode, ('stop', 'optional'))
            self.assertEqual(image, IMAGE)
            self.assertEqual(parsed(name)['tag'], IMAGE)
            self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= 180)
        for name in ('B0-build', 'A0-agentstop-unserve', 'G1-pairslice-exact-audited', 'H1-handback-reset-fabric-agentstart'):
            self.assertEqual(modes[name], 'stop')

    def test_templates_parse_are_lf_and_name_no_card_host_address_registry_or_digest(self):
        for name in ORDERED:
            text = (FOLDER / (name + '.env')).read_bytes()
            self.assertNotIn(b'\r', text, name)
            self.assertIsNone(BANNED.search(text.decode('utf-8')), name)
            parsed(name)
        order = (FOLDER / 'ORDER.txt').read_bytes()
        self.assertNotIn(b'\r', order)
        self.assertIsNone(BANNED.search(order.decode('utf-8')))

    def test_only_a0_takes_production_down_the_build_opens_no_card_and_the_hand_back_never_deploys(self):
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            self.assertEqual('build' in actions, name == 'B0-build', name)
            self.assertEqual(bool({'agentstop', 'unserve'} & set(actions)), name == 'A0-agentstop-unserve', name)
            self.assertNotIn('push', actions)
            self.assertFalse(re.search(r'(?im)^C2_(PLACE|DEPLOY)', (FOLDER / (name + '.env')).read_text(encoding='utf-8')), name)
        self.assertEqual(parsed('A0-agentstop-unserve')['actions'], 'status agentstop unserve')
        self.assertEqual(parsed('B0-build')['actions'], 'build')
        self.assertEqual(parsed('H1-handback-reset-fabric-agentstart')['actions'].split(), ['status', 'reset', 'fabric', 'agentstart'])
        self.assertEqual(order_rows()[-1][0], 'H1-handback-reset-fabric-agentstart')
        for name in ORDERED:
            outputs = parsed(name)
            if name in ('B0-build', 'A0-agentstop-unserve', 'H1-handback-reset-fabric-agentstart'):
                continue
            actions = outputs['actions'].split()
            self.assertEqual(outputs['cards'], 'quad')
            self.assertEqual(actions[0], 'reset', name)
            self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1, name)

    def test_the_audited_arm_serves_the_audited_profile_and_the_pairs_alternate_control_and_lever(self):
        self.assertEqual(parsed('G1-pairslice-exact-audited')['profile'], AUDITED_ARM)
        self.assertEqual(parsed('G1-pairslice-exact-audited')['tests'], 'warmup,coding,concurrent4_code_equal,concurrent4_code_32k')
        profiles = [parsed(name)['profile'] for name in ORDERED[3:7]]
        self.assertEqual(profiles, [CONTROL, TIMED_ARM, CONTROL, TIMED_ARM])
        self.assertEqual(parsed('DD1-dispatchdiag-timed')['profile'], DIAG_ARM)
        names = [row[0] for row in order_rows()]
        for name in ORDERED[3:7]:
            self.assertLess(names.index('G1-pairslice-exact-audited'), names.index(name))
        for word in ('ABAB', 'PAIRED per round', 'if G1 fails', 'Nothing combined has run on a card', 'FRESH tag'):
            self.assertIn(word, (FOLDER / 'ORDER.txt').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
