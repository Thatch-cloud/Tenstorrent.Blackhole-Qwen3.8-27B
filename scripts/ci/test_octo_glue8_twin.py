"""QWEN_FAST_OCTO_GLUE8, sites 'split' and 'merge' (V2) and the V1 wiring: the twin DeviceLoopState at the octo block's eight users of eight rows, over the packed-segments
fixture's fakes (the same harness as test_tp4_vglue_twin, at eight users).

The contract this holds:
  * flag unset or '0': the octo block runs the SERVED path at the glue sites exactly as before this lever existed - call for call the pinned class's, the mover launches none, the
    tp4_vglue.EIGHT_ROW_SERVED lines and counts are today's, and neither new module is imported (byte-identical off, proven by call-sequence equality);
  * flag on: the per-user slices become one quarter-tile split launch and the concat one merge launch, and nothing else changes: the state moves, their order, the T1 notes, the
    recurrence launch's arguments and the result dictionary's keys are the pinned body's; the counts and markers are the glue8 ones; a launch the kernel cannot carry falls back
    before any state move under the glue8 FALLBACK marker;
  * a sixteen-row M3 block is untouched by the flag, whatever its value.

Run: `python -m unittest test_octo_glue8_twin` from scripts/ci.
"""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gdn_device_loop_state as pinned_state
import gdn_device_loop_state_tp as twin
import gdn_rows_dma8_tp as rows8
import gdn_rows_dma_tp as rows16
import octo_glue8
import test_gdn_packed_segments
import tp4_vglue
from test_gdn_packed_segments import PackedFixture

BLOCK_WIDTH = 8240              # the packed-segments fixture's projection width (the planners are width-generic)
OUTPUT_WIDTH = 3072
OCTO_SPANS = tuple((8 * user, 8 * user + 8) for user in range(8))
M3_SPANS = ((0, 16), (16, 32), (32, 48), (48, 64))


def four():
    """QWEN_FAST_TP=4 without the address seam: the fixture's fakes hold two shards per tensor."""
    return patch.dict(os.environ, {'QWEN_FAST_TP': '4'})


def env(**values):
    return patch.dict(os.environ, values)


def moves(log):
    return [entry for entry in log if entry[0] in ('copy_compact', 'restore', 'save', 'histories')]


class OctoTwinTests(PackedFixture):
    def setUp(self):
        tp4_vglue.take()
        twin.DeviceLoopState.glue_fallbacks = set()
        self.launches, self.lines = [], []
        for patcher in (patch('gdn_device_loop_state_tp.tp4_vglue.log_line', side_effect=self.lines.append),
                        patch.object(octo_glue8, 'log_line', side_effect=self.lines.append), patch.object(octo_glue8, '_LOGGED', set())):
            patcher.start()
            self.addCleanup(patcher.stop)

    def build_state(self, cls, users):
        with patch.object(test_gdn_packed_segments, 'DeviceLoopState', cls):
            state, operations, layer, active, calls = self.build(commit_only=True, users=users, user_batch=True)
        counter = iter(range(100))
        rows = 64 // users

        def empty(shape, **keywords):
            name = 'split:%d' % next(counter) if shape[1] == rows else 'merged'
            return SimpleNamespace(shape=tuple(shape), name=name, dtype='bf16', layout='tile',
                                   memory_config=lambda memory=keywords.get('memory_config'): memory)

        operations.empty = Mock(side_effect=empty)
        whole_tile_join = operations.concat.side_effect

        def card_concat(parts, **keywords):
            joined = whole_tile_join(parts, **keywords)
            if keywords.get('memory_config') is not None:
                placed = keywords['memory_config']
            elif parts[0].shape[1] % 32:
                placed = 'dram'
            else:
                return joined
            joined.memory_config = lambda: placed
            return joined

        operations.concat = Mock(side_effect=card_concat)
        operations.DRAM_MEMORY_CONFIG = 'dram'
        operations.bfloat16, operations.TILE_LAYOUT = 'bf16', 'tile'
        operations.clone = Mock(side_effect=lambda value, memory_config=None: SimpleNamespace(
            shape=value.shape, name='copy:' + value.name, memory_config=lambda: memory_config))
        return state, operations, layer, active, calls

    def segment_runner(self, calls):
        def run(mesh, users, taps, dt_bias, neg_exp_A, norm_w, kernels, operations=None, **options):
            calls.append(('batched', tuple(piece.name for piece, initial, history in users)))
            calls.append(('histories', tuple((initial, tuple(history)) for piece, initial, history in users)))
            return [dict(output=SimpleNamespace(shape=(1, piece.shape[1], OUTPUT_WIDTH), name='out%d' % index, dtype='bf16', layout='tile',
                                                memory_config=lambda: 'l1'),
                         states=SimpleNamespace(shape=(piece.shape[1], 24, 128, 128)), conv_prefixes=[None] * piece.shape[1], owned=[],
                         packed_conv_states=[], packed_checkpoints=True, deferred_conv_publication=True, norm_batch=False,
                         prefix_zero_reuse=False, user_batched=True)
                    for index, (piece, initial, history) in enumerate(users)]
        return run

    def run_decode(self, cls, spans=OCTO_SPANS, flags=None, launch_error=None, patch_eight=True):
        import contextlib

        users = len(spans)
        state, operations, layer, active, calls = self.build_state(cls, users)
        slots = [['%s%d' % (user, index) for index in range(5)] for user in range(users)]
        checkpoints = [['ck%d' % user] for user in range(users)]

        def launch(module):
            def run(mesh, sources, destinations, tasks, **keywords):
                if launch_error is not None:
                    raise launch_error
                self.launches.append((module, tuple(source.name for source in sources), tuple(dest.name for dest in destinations), tasks))
            return run

        def move(source, destination):
            calls.append(('copy_compact', source[0], destination[0]))

        with env(**(flags or {})), contextlib.ExitStack() as stack:
            stack.enter_context(patch('gdn_rows_dma_tp.launch', side_effect=launch('rows16')))
            stack.enter_context(patch('gdn_rows_dma_tp.problem', return_value=None))
            if patch_eight:
                stack.enter_context(patch('gdn_rows_dma8_tp.launch', side_effect=launch('rows8')))
                stack.enter_context(patch('gdn_rows_dma8_tp.problem', return_value=None))
            stack.enter_context(patch('gdn_device_loop_state.run_user_batched_projected', side_effect=self.segment_runner(calls)))
            stack.enter_context(patch('gdn_device_loop_state_tp.run_user_batched_projected', side_effect=self.segment_runner(calls)))
            stack.enter_context(patch('gdn_device_loop_state.restore_prefix'))
            stack.enter_context(patch('gdn_device_loop_state.release_owned'))
            stack.enter_context(patch('gdn_device_loop_state_tp.release_owned'))
            stack.enter_context(patch('gdn_device_loop_state.copy_compact', side_effect=move))
            stack.enter_context(patch('gdn_device_loop_state_tp.copy_compact', side_effect=move))
            packed = SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1')
            result = state.decode(packed, checkpoints, [0] * users, segments=spans, slots=slots, deferred=True)
        return result, calls, operations

    # -- flag off: the octo block is today's

    def test_flag_off_the_octo_block_is_the_pinned_class_call_for_call_and_imports_nothing_new(self):
        for name in ('gdn_rows_dma8_tp', 'octo_glue8'):
            self.addCleanup(sys.modules.__setitem__, name, sys.modules[name])
            sys.modules.pop(name, None)
        with four():
            expected, expected_calls, unused = self.run_decode(pinned_state.DeviceLoopState, patch_eight=False)
            expected_counts = tp4_vglue.take()
            self.lines.clear()
            result, calls, operations = self.run_decode(twin.DeviceLoopState, flags={'QWEN_FAST_TP4_GDN_GLUE': '1'}, patch_eight=False)
        self.assertEqual(calls, expected_calls)
        self.assertEqual(sorted(result), sorted(expected))
        self.assertEqual(self.launches, [], 'no mover launch: the served ops ran')
        operations.empty.assert_not_called()
        self.assertEqual(tp4_vglue.take(), {'gdn_split_eight_row_served': 1, 'gdn_merge_eight_row_served': 1})
        self.assertEqual(expected_counts, {})
        self.assertTrue(any(line.startswith(tp4_vglue.EIGHT_ROW_SERVED) and 'site=split' in line for line in self.lines), self.lines)
        self.assertTrue(any(line.startswith(tp4_vglue.EIGHT_ROW_SERVED) and 'site=merge' in line for line in self.lines), self.lines)
        self.assertFalse(any('glue8' in line or line.startswith(tp4_vglue.FALLBACK) for line in self.lines), self.lines)
        self.assertNotIn('gdn_rows_dma8_tp', sys.modules)
        self.assertNotIn('octo_glue8', sys.modules)

    def test_flag_zero_is_off(self):
        with four():
            expected, expected_calls, unused = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations = self.run_decode(twin.DeviceLoopState, flags={'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_OCTO_GLUE8': '0'})
        self.assertEqual(calls, expected_calls)
        self.assertEqual(self.launches, [])
        self.assertEqual(tp4_vglue.take(), {'gdn_split_eight_row_served': 1, 'gdn_merge_eight_row_served': 1})

    def test_flag_on_without_v2_is_off_for_the_glue_sites(self):
        with four():
            expected, expected_calls, unused = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations = self.run_decode(twin.DeviceLoopState, flags={'QWEN_FAST_OCTO_GLUE8': '1'})
        self.assertEqual(calls, expected_calls)
        self.assertEqual(self.launches, [])
        self.assertEqual(tp4_vglue.take(), {})

    # -- flag on

    def test_flag_on_cuts_the_pieces_and_joins_the_outputs_with_one_quarter_tile_launch_each(self):
        flags = {'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_OCTO_GLUE8': '1'}
        with four():
            expected, expected_calls, unused = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations = self.run_decode(twin.DeviceLoopState, flags=flags)
        split, merge = self.launches
        self.assertEqual(split[:3], ('rows8', ('projected',), tuple('split:%d' % user for user in range(8))))
        self.assertEqual(split[3], rows8.split_pieces(8, BLOCK_WIDTH))
        self.assertEqual(merge[:3], ('rows8', tuple('out%d' % user for user in range(8)), ('merged',)))
        self.assertEqual(merge[3], rows8.merge_outputs(8, OUTPUT_WIDTH))
        self.assertEqual([entry for entry in calls if entry[0] == 'batched'], [('batched', tuple('split:%d' % user for user in range(8)))])
        self.assertEqual([entry for entry in expected_calls if entry[0] == 'batched'],
                         [('batched', tuple('piece:%d' % (8 * user) for user in range(8)))])
        self.assertEqual([entry for entry in calls if entry[0] == 'slice'], [('slice', 0, 32), ('slice', 32, 64)],
                         'no per-user slice of the block remains (only the projection\'s own two 32-row tile slices)')
        self.assertEqual(moves(calls), moves(expected_calls), 'every state move, in the pinned order')
        self.assertEqual(result['output'].name, 'merged')
        self.assertEqual(result['output'].shape, (1, 64, OUTPUT_WIDTH))
        self.assertEqual(sorted(result), sorted(expected))
        self.assertEqual(result['segments'], expected['segments'])
        self.assertEqual(len(result['segment_results']), 8)
        self.assertEqual(tp4_vglue.take(), {'gdn_glue': 1, 'gdn_merge': 1, 'glue8_split': 1, 'glue8_merge': 1})
        self.assertEqual([line for line in self.lines], [octo_glue8.engaged_line('split'), octo_glue8.engaged_line('merge')])

    def test_flag_on_places_and_owns_like_the_served_path(self):
        flags = {'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_OCTO_GLUE8': '1'}
        with four():
            result, calls, operations = self.run_decode(twin.DeviceLoopState, flags=flags)
        names = [getattr(value, 'name', None) for value in result['owned']]
        for name in ['projected'] + ['split:%d' % user for user in range(8)] + ['merged']:
            self.assertIn(name, names)
        self.assertEqual(operations.empty.call_count, 9)
        piece_call = operations.empty.call_args_list[0]
        self.assertEqual(piece_call.args[0], (1, 8, BLOCK_WIDTH))
        self.assertEqual(piece_call.kwargs['memory_config'], 'l1', 'where resident_piece leaves every served piece')
        self.assertEqual(operations.empty.call_args_list[8].args[0], (1, 64, OUTPUT_WIDTH))
        self.assertEqual(operations.empty.call_args_list[8].kwargs['memory_config'], 'dram', 'where the served concat leaves its join')
        self.assertEqual(result['output'].memory_config(), 'dram')

    def test_flag_on_takes_every_t1_carry_mode_the_pinned_body_takes(self):
        modes = ({}, {'QWEN_FAST_VERIFY_T1': '1'}, {'QWEN_FAST_VERIFY_T1': '1', 'QWEN_FAST_VERIFY_T1_SKIP': 'last_carry'},
                 {'QWEN_FAST_VERIFY_T1': '1', 'QWEN_FAST_VERIFY_T1_SKIP': 'direct_carry'})
        for mode in modes:
            with self.subTest(mode=mode), four():
                self.launches.clear()
                expected, expected_calls, unused = self.run_decode(pinned_state.DeviceLoopState, flags=mode)
                result, calls, operations = self.run_decode(twin.DeviceLoopState,
                                                            flags=dict(mode, QWEN_FAST_TP4_GDN_GLUE='1', QWEN_FAST_OCTO_GLUE8='1'))
                self.assertEqual(moves(calls), moves(expected_calls))
                self.assertEqual(len(self.launches), 2)

    def test_a_launch_the_kernel_cannot_carry_falls_back_before_any_state_move_under_the_glue8_marker(self):
        flags = {'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_OCTO_GLUE8': '1'}
        with four():
            expected, expected_calls, unused = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations = self.run_decode(twin.DeviceLoopState, flags=flags, launch_error=rows8.Unsupported('too long'))
        ours = [entry for entry in calls if not (entry[0] == 'free' and str(entry[1]).startswith(('split:', 'merged')))]
        self.assertEqual(ours, expected_calls, 'the pinned path ran, once, from a clean start')
        self.assertEqual([entry for entry in calls if entry not in ours], [('free', 'split:%d' % index) for index in range(8)] + [('free', 'merged')])
        self.assertTrue(any(line.startswith(octo_glue8.FALLBACK) and 'site=split' in line for line in self.lines), self.lines)
        self.assertTrue(any(line.startswith(octo_glue8.FALLBACK) and 'site=merge' in line for line in self.lines), self.lines)
        self.assertFalse(any(line.startswith(tp4_vglue.FALLBACK) or line.startswith(tp4_vglue.EIGHT_ROW_SERVED) for line in self.lines), self.lines)
        self.assertEqual(tp4_vglue.take(), {'glue8_split_fallback': 1, 'glue8_merge_fallback': 1})
        problems = octo_glue8.glue8_problems('\n'.join(self.lines), dict(os.environ, **{'QWEN_FAST_TP': '4', 'QWEN_FAST_OCTO': 'live', 'QWEN_C2_GATE_PROFILE': '1', **flags}))
        self.assertTrue(any('fell back' in problem for problem in problems), problems)

    def test_a_block_that_is_not_the_octo_blocks_is_still_the_ordinary_refusal(self):
        state, operations, layer, active, calls = self.build_state(twin.DeviceLoopState, 8)
        with four(), env(QWEN_FAST_OCTO_GLUE8='1'), patch('gdn_rows_dma8_tp.problem', return_value=None), patch('gdn_rows_dma_tp.problem', return_value=None):
            self.assertIsNone(state.glue_problem(SimpleNamespace(shape=(1, 64, BLOCK_WIDTH)), OCTO_SPANS))
            self.assertIn('not (1, 64, W)', state.glue_problem(SimpleNamespace(shape=(1, 32, BLOCK_WIDTH)), OCTO_SPANS))
            self.assertIn('not contiguous', state.glue_problem(SimpleNamespace(shape=(1, 32, BLOCK_WIDTH)), OCTO_SPANS[:4]))
            self.assertIsNone(state.merge_problem([SimpleNamespace(shape=(1, 8, OUTPUT_WIDTH))] * 8))
            self.assertIn('not equal (1, 16, N)', state.merge_problem([SimpleNamespace(shape=(1, 8, OUTPUT_WIDTH))] * 4))
            self.assertIn('not equal (1, 16, N)', state.merge_problem([SimpleNamespace(shape=(1, 8, OUTPUT_WIDTH))] * 7 + [SimpleNamespace(shape=(1, 16, OUTPUT_WIDTH))]))
        with four():
            self.assertEqual(state.glue_problem(SimpleNamespace(shape=(1, 64, BLOCK_WIDTH)), OCTO_SPANS), twin.EIGHT_ROW_REASON)
            self.assertEqual(state.merge_problem([SimpleNamespace(shape=(1, 8, OUTPUT_WIDTH))] * 8), twin.EIGHT_ROW_REASON)

    # -- the sixteen-row block

    def test_an_m3_block_is_untouched_by_the_flag_whatever_its_value(self):
        base = {'QWEN_FAST_TP4_GDN_GLUE': '1'}
        with four():
            expected, expected_calls, unused = self.run_decode(twin.DeviceLoopState, spans=M3_SPANS, flags=base)
            first = list(self.launches)
            self.launches.clear()
            tp4_vglue.take()
            for value in ('1', '0', '2'):
                with self.subTest(value=value):
                    self.launches.clear()
                    result, calls, operations = self.run_decode(twin.DeviceLoopState, spans=M3_SPANS, flags=dict(base, QWEN_FAST_OCTO_GLUE8=value))
                    self.assertEqual(calls, expected_calls)
                    self.assertEqual([launch[0] for launch in self.launches], ['rows16', 'rows16'])
                    self.assertEqual(self.launches, first)
                    self.assertEqual(tp4_vglue.take(), {'gdn_glue': 1, 'gdn_merge': 1})

    def test_a_bad_value_raises_at_the_octo_block(self):
        with four(), self.assertRaises(ValueError):
            self.run_decode(twin.DeviceLoopState, flags={'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_OCTO_GLUE8': '2'})

    # -- the audit

    def test_the_audit_holds_a_copy_of_each_launch_output_beside_the_served_one_for_eight_users(self):
        flags = {'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_OCTO_GLUE8': '1', 'QWEN_FAST_TP4_VGLUE_AUDIT': '1', 'QWEN_FAST_OCTO': 'live'}
        with four():
            result, calls, operations = self.run_decode(twin.DeviceLoopState, flags=flags)
        entries = tp4_vglue.audit_entries(result)
        self.assertEqual([entry['label'] for entry in entries], ['piece user %d' % user for user in range(8)] + ['merged output'])
        self.assertEqual(len(tp4_vglue.audit_held_of(result)), 18)
        self.assertEqual(entries[-1]['placement'], ('dram', 'dram'))
        with four(), env(**flags):
            self.assertEqual(tp4_vglue.missing_entries(result), [], 'every lever that is on has its entry for every one of the eight users')
            self.assertGreater(tp4_vglue.audit_round.__code__.co_argcount, 0)

    def test_packed_verifier_audits_the_octo_block_when_the_flag_makes_its_sites_native(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parent / 'packed_verifier.py').read_text(encoding='utf-8')
        self.assertIn("served_only=(self.users, self.rows_per_user) == (8, 8) and os.environ.get('QWEN_FAST_OCTO_GLUE8', '0') == '0'", text)
        records = [(0, {'segment_results': [{} for _ in range(8)]})]
        with four(), env(QWEN_FAST_TP4_GDN_GLUE='1', QWEN_FAST_TP4_VGLUE_AUDIT='1'):
            self.assertEqual(tp4_vglue.audit_round(None, records, 1, served_only=True), 0)
            with self.assertRaises(AssertionError):
                tp4_vglue.audit_round(None, records, 1, served_only=False)     # a native site that declined a layer is a failure, not a pass on what is left


class BlockStageWiringTests(unittest.TestCase):
    """DeviceLoopState.run_users at the octo block: which stage gets the layer, and which marker a decline carries."""

    def setUp(self):
        tp4_vglue.take()
        twin.DeviceLoopState.glue_fallbacks = set()
        self.lines = []
        for patcher in (patch('gdn_device_loop_state_tp.tp4_vglue.log_line', side_effect=self.lines.append),
                        patch.object(octo_glue8, 'log_line', side_effect=self.lines.append), patch.object(octo_glue8, '_LOGGED', set())):
            patcher.start()
            self.addCleanup(patcher.stop)

    def state(self):
        state = object.__new__(twin.DeviceLoopState)
        state.gdn = SimpleNamespace(mesh='mesh', tw=dict(conv_taps=['a'] * 4, dt_bias='dt', neg_exp_A='neg', norm_w='norm'))
        state.operations, state.kernels, state.prefix_zero_reuse = 'ops', 'kernels', False
        return state

    def pending(self, users, rows):
        return [(SimpleNamespace(shape=(1, rows, 4120)), 'initial', ()) for unused in range(users)]

    def call(self, flags, users, rows, stage_result):
        seen = {}
        eight_stage = Mock(return_value=stage_result)
        sixteen_stage = Mock(return_value=stage_result)
        with patch('gdn_device_loop_state_tp.run_user_batched_projected', side_effect=lambda *args, **kwargs: seen.update(kwargs) or []), \
                patch('gdn_block_conv8_tp.stage', eight_stage), patch('gdn_block_conv_tp.stage', sixteen_stage), env(**flags):
            self.state().run_users('projected', [], self.pending(users, rows))
            block_stage = seen.get('block_stage')
            if block_stage is not None:
                block_stage(self.pending(users, rows), 'windows')
        return seen, eight_stage, sixteen_stage

    ON = {'QWEN_FAST_TP': '4', 'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_TP4_GDN_BLOCK_CONV': '1', 'QWEN_FAST_VERIFY_T2': '1'}

    def test_flag_off_the_octo_block_gets_the_sixteen_row_stage_which_only_runs_if_windows_engaged(self):
        seen, eight, sixteen = self.call(self.ON, 8, 8, None)
        self.assertIn('block_stage', seen)
        eight.assert_not_called()
        sixteen.assert_called_once()

    def test_flag_on_the_octo_block_gets_the_eight_row_stage_and_counts_it(self):
        found = SimpleNamespace(users=[], owned=[], held=[], entries=[])
        seen, eight, sixteen = self.call(dict(self.ON, QWEN_FAST_OCTO_GLUE8='1'), 8, 8, found)
        eight.assert_called_once()
        sixteen.assert_not_called()
        self.assertEqual(tp4_vglue.take(), {'gdn_block_conv': 1, 'glue8_block_conv': 1})
        self.assertEqual(self.lines, [octo_glue8.engaged_line('block_conv')])

    def test_flag_on_a_declined_stage_is_a_glue8_fallback_not_a_vglue_one(self):
        def decline(*args, note_fallback=None, **kwargs):
            note_fallback('no fit')
            return None

        with patch('gdn_block_conv8_tp.stage', side_effect=decline), patch('gdn_device_loop_state_tp.run_user_batched_projected',
                                                                         side_effect=lambda *args, **kwargs: kwargs['block_stage'](self.pending(8, 8), 'w') and []), \
                env(**dict(self.ON, QWEN_FAST_OCTO_GLUE8='1')):
            self.state().run_users('projected', [], self.pending(8, 8))
        self.assertEqual(self.lines, [octo_glue8.fallback_line('block_conv', 'no fit')])
        self.assertEqual(tp4_vglue.take(), {'glue8_block_conv_fallback': 1})

    def test_flag_on_an_m3_block_still_gets_the_sixteen_row_stage(self):
        seen, eight, sixteen = self.call(dict(self.ON, QWEN_FAST_OCTO_GLUE8='1'), 4, 16, None)
        eight.assert_not_called()
        sixteen.assert_called_once()
        seen, eight, sixteen = self.call(dict(self.ON, QWEN_FAST_OCTO_GLUE8='1'), 8, 16, None)
        eight.assert_not_called()


if __name__ == '__main__':
    unittest.main()
