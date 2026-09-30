"""The twin DeviceLoopState (tp4/vglue V2) against the pinned class, over the packed-segments fixture's fakes.

Flag off, the twin is the pinned class call for call. Flag on, it changes the two places it is written to change - the per-user
slices become one split launch and the concat becomes one merge launch - and nothing else: the state moves, their order, the
T1 notes, the recurrence launch's arguments and the result dictionary's keys are the pinned body's.

Run at py 3.11: `py -3.11 -m unittest test_tp4_vglue_twin` from scripts/ci.
"""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gdn_device_loop_state as pinned_state
import gdn_device_loop_state_tp as twin
import gdn_rows_dma_tp as rows_dma
import test_gdn_packed_segments
import tp4_vglue
from test_gdn_packed_segments import PackedFixture
from test_gdn_tp_twins import four as four_installed, pair

HERE = Path(__file__).resolve().parent
BLOCK_WIDTH = 8240              # the packed-segments fixture's projection width (the planners are width-generic)
OUTPUT_WIDTH = 3072


def four():
    """QWEN_FAST_TP=4 without the address seam: the fixture's fakes hold two shards per tensor."""
    return patch.dict(os.environ, {'QWEN_FAST_TP': '4'})


def env(**values):
    return patch.dict(os.environ, values)


def moves(log):
    return [entry for entry in log if entry[0] in ('copy_compact', 'restore', 'save', 'histories')]


class TwinTests(PackedFixture):
    spans = ((0, 16), (16, 32), (32, 48), (48, 64))
    slots = [['%s%d' % (user, index) for index in range(5)] for user in 'ABCD']
    checkpoints = [['ck' + user] for user in 'ABCD']

    def setUp(self):
        tp4_vglue.take()
        twin.DeviceLoopState.glue_fallbacks = set()
        self.launches, self.lines = [], []
        patcher = patch('gdn_device_loop_state_tp.tp4_vglue.log_line', side_effect=self.lines.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def build_state(self, cls):
        with patch.object(test_gdn_packed_segments, 'DeviceLoopState', cls):
            state, operations, layer, active, calls = self.build(commit_only=True, users=4, user_batch=True)
        counter = iter(range(100))

        def empty(shape, **keywords):
            name = 'split:%d' % next(counter) if shape[1] == 16 else 'merged'
            return SimpleNamespace(shape=tuple(shape), name=name, dtype='bf16', layout='tile',
                                   memory_config=lambda memory=keywords.get('memory_config'): memory)

        operations.empty = Mock(side_effect=empty)
        operations.bfloat16, operations.TILE_LAYOUT = 'bf16', 'tile'
        operations.clone = Mock(side_effect=lambda value, memory_config=None: SimpleNamespace(
            shape=value.shape, name='copy:' + value.name, memory_config=lambda: memory_config))
        return state, operations, layer, active, calls

    def segment_runner(self, calls):
        def run(mesh, users, taps, dt_bias, neg_exp_A, norm_w, kernels, operations=None, **options):
            calls.append(('batched', tuple(piece.name for piece, initial, history in users)))
            calls.append(('histories', tuple((initial, tuple(history)) for piece, initial, history in users)))
            return [dict(output=SimpleNamespace(shape=(1, piece.shape[1], OUTPUT_WIDTH), name='out%d' % index,
                                                dtype='bf16', layout='tile', memory_config=lambda: 'l1'),
                         states=SimpleNamespace(shape=(piece.shape[1], 24, 128, 128)),
                         conv_prefixes=[None] * piece.shape[1], owned=[], packed_conv_states=[],
                         packed_checkpoints=True, deferred_conv_publication=True, norm_batch=False,
                         prefix_zero_reuse=False, user_batched=True)
                    for index, (piece, initial, history) in enumerate(users)]
        return run

    def run_decode(self, cls, flags=None, launch_error=None):
        state, operations, layer, active, calls = self.build_state(cls)

        def launch(mesh, sources, destinations, tasks, **keywords):
            if launch_error is not None:
                raise launch_error
            self.launches.append((tuple(source.name for source in sources), tuple(dest.name for dest in destinations), tasks))

        def move(source, destination):
            calls.append(('copy_compact', source[0], destination[0]))

        with env(**(flags or {})), patch('gdn_rows_dma_tp.launch', side_effect=launch), \
                patch('gdn_rows_dma_tp.problem', return_value=None), \
                patch('gdn_device_loop_state.run_user_batched_projected', side_effect=self.segment_runner(calls)), \
                patch('gdn_device_loop_state_tp.run_user_batched_projected', side_effect=self.segment_runner(calls)), \
                patch('gdn_device_loop_state.restore_prefix'), patch('gdn_device_loop_state.release_owned'), \
                patch('gdn_device_loop_state_tp.release_owned'), \
                patch('gdn_device_loop_state.copy_compact', side_effect=move), \
                patch('gdn_device_loop_state_tp.copy_compact', side_effect=move):
            packed = SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1')
            result = state.decode(packed, self.checkpoints, [0] * 4, segments=self.spans, slots=self.slots, deferred=True)
        return result, calls, operations

    # -- flag off

    def test_flag_off_is_the_pinned_class_call_for_call(self):
        with four():
            expected, expected_calls, unused = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations = self.run_decode(twin.DeviceLoopState)
        self.assertEqual(calls, expected_calls)
        self.assertEqual(sorted(result), sorted(expected))
        self.assertEqual(result['output'].name, expected['output'].name)
        self.assertEqual(self.launches, [])
        operations.empty.assert_not_called()
        self.assertEqual(tp4_vglue.take(), {})

    def test_flag_zero_is_off(self):
        with four():
            expected, expected_calls, unused = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations = self.run_decode(twin.DeviceLoopState, flags={'QWEN_FAST_TP4_GDN_GLUE': '0'})
        self.assertEqual(calls, expected_calls)
        self.assertEqual(self.launches, [])

    def test_the_twin_is_a_subclass_and_the_seam_binds_it(self):
        import importlib

        import tp_addresses
        self.assertTrue(issubclass(twin.DeviceLoopState, pinned_state.DeviceLoopState))
        rows = [row for row in tp_addresses.TWINS if row[0] == 'gdn_device_loop_state']
        self.assertEqual(rows, [('gdn_device_loop_state', 'DeviceLoopState', 'gdn_device_loop_state_tp', 'DeviceLoopState')])
        with four_installed():
            self.assertIs(importlib.import_module('gdn_device_loop_state').DeviceLoopState, twin.DeviceLoopState)

    def test_only_the_two_methods_are_overridden_and_the_pinned_ones_reach_them_through_self(self):
        own = {name for name, value in vars(twin.DeviceLoopState).items() if callable(value) and not name.startswith('__')}
        self.assertTrue({'_recurrence_user_batched', '_finish_packed'} <= own)
        for name in ('decode', '_decode_packed', '_recurrence', 'close'):
            self.assertNotIn(name, own)
        source = (HERE / 'gdn_device_loop_state.py').read_text()
        self.assertIn('self._recurrence_user_batched(projected, spans, slots, entries, owned)', source)
        self.assertIn('self._finish_packed(spans, results, outputs, owned)', source)

    # -- flag on

    def test_flag_on_cuts_the_pieces_and_joins_the_outputs_with_one_launch_each(self):
        with four():
            expected, expected_calls, unused = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations = self.run_decode(twin.DeviceLoopState, flags={'QWEN_FAST_TP4_GDN_GLUE': '1'})
        split, merge = self.launches
        self.assertEqual(split[:2], (('projected',), ('split:0', 'split:1', 'split:2', 'split:3')))
        self.assertEqual(split[2], rows_dma.split_pieces(4, BLOCK_WIDTH))
        self.assertEqual(merge[:2], (('out0', 'out1', 'out2', 'out3'), ('merged',)))
        self.assertEqual(merge[2], rows_dma.merge_outputs(4, OUTPUT_WIDTH))
        # the recurrence launch reads the launch's pieces, where the pinned run's read the slices'
        self.assertEqual([entry for entry in calls if entry[0] == 'batched'],
                         [('batched', ('split:0', 'split:1', 'split:2', 'split:3'))])
        self.assertEqual([entry for entry in expected_calls if entry[0] == 'batched'],
                         [('batched', ('piece:0', 'piece:16', 'piece:32', 'piece:48'))])
        # no per-user slice of the block remains (only the projection's own two 32-row tile slices)
        self.assertEqual([entry for entry in calls if entry[0] == 'slice'], [('slice', 0, 32), ('slice', 32, 64)])
        # every state move, in the pinned order
        self.assertEqual(moves(calls), moves(expected_calls))
        self.assertEqual(result['output'].name, 'merged')
        self.assertEqual(result['output'].shape, (1, 64, OUTPUT_WIDTH))
        self.assertEqual(sorted(result), sorted(expected))
        self.assertEqual(result['segments'], expected['segments'])
        self.assertEqual(len(result['segment_results']), 4)
        self.assertEqual(result['commit_only_gdn'], expected['commit_only_gdn'])
        self.assertEqual(tp4_vglue.take(), {'gdn_glue': 1, 'gdn_merge': 1})

    def test_flag_on_owns_the_pieces_and_the_merged_output_like_the_slices_and_the_concat(self):
        with four():
            result, calls, operations = self.run_decode(twin.DeviceLoopState, flags={'QWEN_FAST_TP4_GDN_GLUE': '1'})
        names = [getattr(value, 'name', None) for value in result['owned']]
        for name in ('projected', 'split:0', 'split:1', 'split:2', 'split:3', 'merged'):
            self.assertIn(name, names)
        self.assertEqual(operations.empty.call_count, 5)
        piece_call = operations.empty.call_args_list[0]
        self.assertEqual(piece_call.args[0], (1, 16, BLOCK_WIDTH))
        self.assertEqual(piece_call.kwargs['memory_config'], 'l1')
        self.assertEqual(operations.empty.call_args_list[4].args[0], (1, 64, OUTPUT_WIDTH))
        self.assertEqual(operations.empty.call_args_list[4].kwargs['memory_config'], 'l1', "the outputs' own placement")

    def test_flag_on_takes_every_t1_carry_mode_the_pinned_body_takes(self):
        modes = ({}, {'QWEN_FAST_VERIFY_T1': '1'},
                 {'QWEN_FAST_VERIFY_T1': '1', 'QWEN_FAST_VERIFY_T1_SKIP': 'last_carry'},
                 {'QWEN_FAST_VERIFY_T1': '1', 'QWEN_FAST_VERIFY_T1_SKIP': 'direct_carry'})
        for mode in modes:
            with self.subTest(mode=mode), four():
                self.launches.clear()
                expected, expected_calls, unused = self.run_decode(pinned_state.DeviceLoopState, flags=mode)
                result, calls, operations = self.run_decode(twin.DeviceLoopState,
                                                            flags=dict(mode, QWEN_FAST_TP4_GDN_GLUE='1'))
                self.assertEqual(moves(calls), moves(expected_calls))
                self.assertEqual(len(self.launches), 2)

    def test_a_block_that_is_not_contiguous_sixteen_row_users_is_a_problem(self):
        state, operations, layer, active, calls = self.build_state(twin.DeviceLoopState)
        with four(), patch('gdn_rows_dma_tp.problem', return_value=None):
            projected = SimpleNamespace(shape=(1, 64, BLOCK_WIDTH))
            for spans in (((0, 16), (16, 32), (32, 48), (56, 72)), ((0, 8), (8, 16)), ((16, 32), (32, 48))):
                self.assertIn('not contiguous', state.glue_problem(projected, spans))
            self.assertIn('not (1, 32, W)', state.glue_problem(projected, ((0, 16), (16, 32))))
            self.assertIsNone(state.glue_problem(projected, ((0, 16), (16, 32), (32, 48), (48, 64))))

    def test_a_launch_the_kernel_cannot_carry_falls_back_before_any_state_move(self):
        with four():
            expected, expected_calls, unused = self.run_decode(pinned_state.DeviceLoopState)
            result, calls, operations = self.run_decode(
                twin.DeviceLoopState, flags={'QWEN_FAST_TP4_GDN_GLUE': '1'}, launch_error=rows_dma.Unsupported('too long'))
        ours = [entry for entry in calls if not (entry[0] == 'free' and str(entry[1]).startswith(('split:', 'merged')))]
        self.assertEqual(ours, expected_calls, 'the pinned path ran, once, from a clean start')
        self.assertEqual([entry for entry in calls if entry not in ours],
                         [('free', 'split:%d' % index) for index in range(4)] + [('free', 'merged')])
        self.assertTrue(any(line.startswith(tp4_vglue.FALLBACK) and 'site=split' in line for line in self.lines))
        self.assertEqual(tp4_vglue.take(), {'gdn_split_fallback': 1, 'gdn_merge_fallback': 1})

    def test_the_flag_at_the_pair_raises(self):
        with pair(), env(QWEN_FAST_TP4_GDN_GLUE='1'):
            with self.assertRaises(ValueError):
                tp4_vglue.enabled(tp4_vglue.GDN_GLUE)

    # -- the audit

    def test_the_audit_holds_a_copy_of_each_launch_output_beside_the_served_one(self):
        with four():
            result, calls, operations = self.run_decode(
                twin.DeviceLoopState, flags={'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_TP4_VGLUE_AUDIT': '1'})
        entries = tp4_vglue.audit_entries(result)
        self.assertEqual([entry['label'] for entry in entries],
                         ['piece user %d' % user for user in range(4)] + ['merged output'])
        held = tp4_vglue.audit_held_of(result)
        self.assertEqual(len(held), 10)
        # held outside `owned`: the layer frees `owned` inside the trace, the audit reads after the replay
        owned_names = [getattr(value, 'name', None) for value in result['owned']]
        self.assertFalse(any(getattr(value, 'name', '').startswith('copy:') for value in result['owned']), owned_names)
        # the served merge (an L1 concat) is freed at once; only DRAM copies are held
        self.assertEqual(entries[-1]['placement'], ('l1', 'l1'), 'the launch block and the served concat, as placed')

    def test_flag_on_without_the_audit_holds_nothing(self):
        with four():
            result, calls, operations = self.run_decode(twin.DeviceLoopState, flags={'QWEN_FAST_TP4_GDN_GLUE': '1'})
        self.assertEqual(tp4_vglue.audit_entries(result), [])
        operations.clone.assert_not_called()


class AuditCompareTests(unittest.TestCase):
    def operations(self, tensors):
        import torch

        return SimpleNamespace(get_device_tensors=lambda value: [value] * 2, to_torch=lambda value: tensors[value])

    def test_bit_patterns_decide_not_float_equality(self):
        import torch

        same = torch.tensor([[1, -32768, 5]], dtype=torch.int16)
        signed = torch.tensor([[1, 0, 5]], dtype=torch.int16)          # -0 against +0: equal as floats, different bits
        entry = dict(label='piece user 1', mine='a', served='b')
        self.assertEqual(tp4_vglue.compare_entry(self.operations({'a': same, 'b': same.clone()}), entry), [])
        found = tp4_vglue.compare_entry(self.operations({'a': same, 'b': signed}), entry)
        self.assertEqual(len(found), 2)
        self.assertIn('1 of 3 elements differ', found[0])

    def test_shape_and_placement_mismatches_are_reported(self):
        import torch

        entry = dict(label='merged output', mine='a', served='b', placement=('l1', 'dram'))
        found = tp4_vglue.compare_entry(self.operations({'a': torch.zeros(1, 4, dtype=torch.int16),
                                                         'b': torch.zeros(1, 8, dtype=torch.int16)}), entry)
        self.assertTrue(found[0].startswith('merged output: placement'))
        self.assertTrue(any('shape' in text for text in found))

    def test_audit_round_reads_the_rotation_and_raises_on_a_difference(self):
        import torch

        good = torch.arange(6, dtype=torch.int16).reshape(1, 6)
        bad = good.clone()
        bad[0, 0] = 9
        tensors = {'m': good, 's': good.clone(), 'x': bad}
        records = [(None, dict(vglue_audit=[dict(label='piece user 0', mine='m', served='s')]), None) for unused in range(48)]
        lines = []
        with patch('tp4_vglue.log_line', side_effect=lines.append):
            self.assertEqual(tp4_vglue.audit_round(self.operations(tensors), records, 1), 48)
            self.assertEqual(tp4_vglue.audit_round(self.operations(tensors), records, 2), 2)
            records[0][1]['vglue_audit'][0]['served'] = 'x'
            with self.assertRaises(AssertionError):
                tp4_vglue.audit_round(self.operations(tensors), records, 1)
        self.assertTrue(lines[0].startswith(tp4_vglue.AUDIT_MARKER) and 'layers=0-47' in lines[0])
        self.assertTrue(lines[-1].startswith(tp4_vglue.AUDIT_MISMATCH))


if __name__ == '__main__':
    unittest.main()
