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
        # the card's join, measured (N1): an explicit memory_config wins; a join of parts that are not whole tiles
        # (16-row users) lands in interleaved DRAM, whatever the parts' placement; whole-tile joins are unmeasured and keep the parts' placement here
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
            self.assertIs(importlib.import_module('gdn_device_loop_state').DeviceLoopState, pinned_state.DeviceLoopState)
        with patch.dict(os.environ, {'QWEN_FAST_TP4_GDN_GLUE': '1'}), four_installed():
            self.assertIs(importlib.import_module('gdn_device_loop_state').DeviceLoopState, twin.DeviceLoopState)

    def test_only_the_two_methods_are_overridden_and_the_pinned_ones_reach_them_through_self(self):
        own = {name for name, value in vars(twin.DeviceLoopState).items() if callable(value) and not name.startswith('__')}
        self.assertTrue({'_recurrence_user_batched', '_finish_packed'} <= own)
        # _decode_packed is wrapped too (tp4/gluefix: the pair slice and the dispatch diagnostic), and calls the pinned one
        self.assertIn('_decode_packed', own)
        for name in ('decode', '_recurrence', 'close'):
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
        self.assertEqual(operations.empty.call_args_list[4].kwargs['memory_config'], 'dram',
                         'where the served concat leaves its join')

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

    def test_the_octo_blocks_eight_row_users_take_the_served_path_under_their_own_marker_not_the_fallback_one(self):
        # Octo-T8: eight users of eight rows are a quarter tile each, which no glue kernel moves yet. The pinned path runs (exact), and the line says so
        # without the marker a gated arm fails on (tp4_vglue.FALLBACK): the octo block's known state.
        state, operations, layer, active, calls = self.build_state(twin.DeviceLoopState)
        octo_spans = tuple((8 * user, 8 * user + 8) for user in range(8))
        with four(), patch('gdn_rows_dma_tp.problem', return_value=None):
            self.assertEqual(state.glue_problem(SimpleNamespace(shape=(1, 64, BLOCK_WIDTH)), octo_spans), twin.EIGHT_ROW_REASON)
            self.assertEqual(state.merge_problem([SimpleNamespace(shape=(1, 8, OUTPUT_WIDTH))] * 8), twin.EIGHT_ROW_REASON)
            # two or four eight-row users are NOT the octo block: the ordinary refusal
            self.assertIn('not contiguous', state.glue_problem(SimpleNamespace(shape=(1, 32, BLOCK_WIDTH)), octo_spans[:4]))
            self.assertIn('not equal (1, 16, N)', state.merge_problem([SimpleNamespace(shape=(1, 8, OUTPUT_WIDTH))] * 4))
            state.note_glue_fallback('split', twin.EIGHT_ROW_REASON)
            state.note_glue_fallback('merge', twin.EIGHT_ROW_REASON)
        self.assertTrue(any(line.startswith(tp4_vglue.EIGHT_ROW_SERVED) and 'site=split' in line for line in self.lines), self.lines)
        self.assertFalse(any(line.startswith(tp4_vglue.FALLBACK) for line in self.lines), self.lines)
        self.assertNotEqual(tp4_vglue.EIGHT_ROW_SERVED, tp4_vglue.FALLBACK)
        self.assertFalse(tp4_vglue.EIGHT_ROW_SERVED.startswith(tp4_vglue.FALLBACK))
        self.assertEqual(tp4_vglue.take(), {'gdn_split_eight_row_served': 1, 'gdn_merge_eight_row_served': 1})

    def test_the_audit_has_nothing_to_read_at_the_octo_blocks_served_path_and_still_fails_a_lever_that_declined_elsewhere(self):
        # QWEN_FAST_TP4_VGLUE_AUDIT demands an entry per lever per user for the audited layers; the octo block's glue sites run the served path, so it holds none, and
        # served_only (packed_verifier passes it for 8 users x 8 rows) says so instead of raising 'nothing compared' on the first octo replay.
        records = [(0, {'segment_results': [{} for _ in range(8)]})]
        with four(), env(QWEN_FAST_TP4_GDN_GLUE='1', QWEN_FAST_TP4_GDN_BLOCK_CONV='1', QWEN_FAST_TP4_VGLUE_AUDIT='1'):
            self.assertEqual(tp4_vglue.audit_round(None, records, 1, served_only=True), 0)
            with self.assertRaises(AssertionError):
                tp4_vglue.audit_round(None, records, 1)           # the same records on an M3 block: a declined lever is a failure
        text = (HERE / 'packed_verifier.py').read_text(encoding='utf-8')
        self.assertIn('served_only=(self.users, self.rows_per_user) == (8, 8)', text)

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
        # the served merge is freed at once; only DRAM copies are held
        self.assertEqual(entries[-1]['placement'], ('dram', 'dram'), 'the launch block and the served concat, as placed')

    def test_the_gdn_glue_outputs_land_where_the_served_path_lands_them(self):
        """The audit requires placement equality (downstream programs compile per input placement): at 4 users and 64 rows the
        GDN glue pieces, the merged block and the output of the twin are placed as the pinned class places its own (the other
        vglue sites are not covered here)."""
        flags = {'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_TP4_VGLUE_AUDIT': '1'}
        with four():
            served, served_calls, served_ops = self.run_decode(pinned_state.DeviceLoopState)
            lever, lever_calls, lever_ops = self.run_decode(twin.DeviceLoopState, flags=flags)
        self.assertEqual(served['output'].memory_config(), 'dram', 'the card joins 16-row outputs into DRAM')
        self.assertEqual(lever['output'].memory_config(), served['output'].memory_config())
        merged = lever_ops.empty.call_args_list[4].kwargs['memory_config']
        self.assertEqual(merged, served['output'].memory_config())
        # the pieces: the served resident_piece ends in L1 (this fixture records no served move, so only the launch side is checked)
        piece_memories = {lever_ops.empty.call_args_list[index].kwargs['memory_config'] for index in range(4)}
        self.assertEqual(piece_memories, {lever_ops.L1_MEMORY_CONFIG})
        entry = tp4_vglue.audit_entries(lever)[-1]
        self.assertEqual(entry['placement'][0], entry['placement'][1])

    def test_the_audit_reports_a_placement_mismatch_of_the_merge(self):
        """A launch block in L1 against a served join that lands in DRAM must show in the entry (the N1 failure), and the
        served join must be made exactly as the pinned path makes it (no memory_config)."""
        with four():
            state, operations, layer, active, calls = self.build_state(twin.DeviceLoopState)
            operations.concat = Mock(side_effect=lambda parts, **keywords: SimpleNamespace(
                shape=(1, 64, 1), memory_config=lambda: keywords.get('memory_config') or 'dram'))
            operations.clone = Mock(side_effect=lambda value, memory_config=None: SimpleNamespace(
                shape=value.shape, memory_config=lambda: memory_config))
            merged = SimpleNamespace(shape=(1, 64, 1), memory_config=lambda: 'l1')
            entry = state.hold_served_merge([SimpleNamespace(shape=(1, 16, 1))] * 4, merged)[0]
        self.assertEqual(entry['placement'], ('l1', 'dram'))
        self.assertNotIn('memory_config', operations.concat.call_args.kwargs)

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

    def records(self, entries_of=None):
        full = lambda: dict(segment_results=(dict(vglue_audit=[dict(label='piece user 0', mine='m', served='s')]),
                                             dict(vglue_audit=[dict(label='piece user 1', mine='m', served='s')])),
                            vglue_merge_audit=[dict(label='merged output', mine='m', served='s')])
        return [(None, full(), None) for unused in range(48)]

    def test_audit_round_reads_the_rotation_and_raises_on_a_difference(self):
        import torch

        good = torch.arange(6, dtype=torch.int16).reshape(1, 6)
        bad = good.clone()
        bad[0, 0] = 9
        tensors = {'m': good, 's': good.clone(), 'x': bad}
        records = self.records()
        lines = []
        with four(), env(QWEN_FAST_TP4_GDN_GLUE='1', QWEN_FAST_TP4_VGLUE_AUDIT='1'),                 patch('tp4_vglue.log_line', side_effect=lines.append):
            self.assertEqual(tp4_vglue.audit_round(self.operations(tensors), records, 1), 48 * 3)
            self.assertEqual(tp4_vglue.audit_round(self.operations(tensors), records, 2), 2 * 3)
            records[0][1]['vglue_merge_audit'][0]['served'] = 'x'
            with self.assertRaises(AssertionError):
                tp4_vglue.audit_round(self.operations(tensors), records, 1)
        self.assertTrue(lines[0].startswith(tp4_vglue.AUDIT_MARKER) and 'layers=0-47' in lines[0])
        self.assertTrue(lines[-1].startswith(tp4_vglue.AUDIT_MISMATCH))

    def test_a_declined_lever_fails_the_audit_instead_of_passing_on_what_is_left(self):
        import torch

        good = torch.arange(6, dtype=torch.int16).reshape(1, 6)
        tensors = {'m': good, 's': good.clone()}
        lines = []
        with four(), env(QWEN_FAST_TP4_GDN_GLUE='1', QWEN_FAST_TP4_VGLUE_AUDIT='1'),                 patch('tp4_vglue.log_line', side_effect=lines.append):
            records = self.records()
            del records[3][1]['vglue_merge_audit']          # the merge declined on layer 3
            with self.assertRaises(AssertionError):
                tp4_vglue.audit_round(self.operations(tensors), records, 1)
            self.assertIn('layer 3 merged output: 0 entries, expected 1', lines[-1])
            records = self.records()
            records[0][1]['segment_results'][1].pop('vglue_audit')   # the split declined for user 1
            with self.assertRaises(AssertionError):
                tp4_vglue.audit_round(self.operations(tensors), records, 1)
            self.assertIn('layer 0 piece user 1: 0 entries', lines[-1])
            records = [(None, dict(segment_results=({}, {})), None)] * 48   # V2 declined everywhere: nothing compared
            with self.assertRaises(AssertionError):
                tp4_vglue.audit_round(self.operations(tensors), records, 1)
        self.assertFalse(any('exact=True' in line for line in lines))

    def test_block_conv_requires_eight_entries_per_user(self):
        import torch

        good = torch.arange(6, dtype=torch.int16).reshape(1, 6)
        tensors = {'m': good, 's': good.clone()}
        block = lambda user, count: [dict(label='block conv user %d %s' % (user, name), mine='m', served='s')
                                     for name in ('conv', 'beta', 'g', 'z', 'window 0', 'window 1', 'window 2', 'window 3')[:count]]
        records = self.records()
        for _, result, _ in records:
            for user, piece in enumerate(result['segment_results']):
                piece['vglue_block_audit'] = block(user, 8)
        flags = dict(QWEN_FAST_TP4_GDN_GLUE='1', QWEN_FAST_TP4_GDN_BLOCK_CONV='1', QWEN_FAST_TP4_VGLUE_AUDIT='1')
        with four(), env(**flags), patch('tp4_vglue.log_line'):
            self.assertEqual(tp4_vglue.audit_round(self.operations(tensors), records, 2), 2 * 19)
            records[0][1]['segment_results'][0]['vglue_block_audit'] = block(0, 5)
            with self.assertRaises(AssertionError):
                tp4_vglue.audit_round(self.operations(tensors), records, 1)

    def test_only_levers_without_gdn_entries_leave_the_gdn_audit_idle(self):
        with four(), env(QWEN_FAST_TP4_ATTN_FOLD='1', QWEN_FAST_TP4_VGLUE_AUDIT='1'):
            self.assertEqual(tp4_vglue.audit_round(self.operations({}), [(None, {}, None)] * 48, 1), 0)


class ReleaseTests(unittest.TestCase):
    """packed_verifier frees what the audit holds (model_batch is held unedited by test_quad_draft, so the verifier does it)."""

    def setUp(self):
        import verify_trace_t1
        import packed_verifier

        self.t1, self.verifier = verify_trace_t1, packed_verifier
        self.freed = []
        self.operations = SimpleNamespace(
            get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda a=id(value) + chip: a) for chip in range(4)],
            deallocate=lambda value: self.freed.append(value))
        self.t1.VALUE_REFERENCES.clear()
        self.addCleanup(self.t1.VALUE_REFERENCES.clear)
        # the pinned release_owned counts two chips; at four cards install() rebinds it to tp_addresses', which counts four
        patcher = patch('packed_verifier.release_owned', side_effect=lambda operations, values: self.freed.extend(values))
        patcher.start()
        self.addCleanup(patcher.stop)

    def fixture(self):
        entry = dict(label='piece user 0', mine='held-mine', served='held-served')
        merged = dict(label='merged output', mine='merged-mine', served='merged-served')
        records = [(None, dict(segment_results=(dict(vglue_audit=[entry]), {}), vglue_merge_audit=[merged]), None)]
        return SimpleNamespace(retained=SimpleNamespace(records=records))

    def test_the_held_tensors_and_the_sampler_reference_are_freed_when_the_audit_is_on(self):
        values = object()
        self.t1.VALUE_REFERENCES[id(values)] = 'reference'
        with four(), env(QWEN_FAST_TP4_SHARD_VALUES='1', QWEN_FAST_TP4_VGLUE_AUDIT='1'):
            self.verifier.release_vglue_audit(self.operations, self.fixture(), values)
        self.assertEqual(sorted(self.freed), ['held-mine', 'held-served', 'merged-mine', 'merged-served', 'reference'])
        self.assertEqual(self.t1.VALUE_REFERENCES, {})

    def test_nothing_is_touched_when_the_audit_is_off_or_the_fixture_is_not_a_model_batch(self):
        with four(), env(QWEN_FAST_TP4_SHARD_VALUES='1'):
            self.verifier.release_vglue_audit(self.operations, self.fixture(), object())
        with four(), env(QWEN_FAST_TP4_SHARD_VALUES='1', QWEN_FAST_TP4_VGLUE_AUDIT='1'):
            self.verifier.release_vglue_audit(self.operations, SimpleNamespace(), None)
        with pair():
            self.verifier.release_vglue_audit(self.operations, self.fixture(), object())
        self.assertEqual(self.freed, [])

    def test_another_traces_reference_is_left_alone(self):
        mine, other = object(), object()
        self.t1.VALUE_REFERENCES[id(other)] = 'other'
        with four(), env(QWEN_FAST_TP4_SHARD_VALUES='1', QWEN_FAST_TP4_VGLUE_AUDIT='1'):
            self.verifier.release_vglue_audit(self.operations, SimpleNamespace(retained=None), mine)
        self.assertEqual(self.freed, [])
        self.assertEqual(self.t1.VALUE_REFERENCES, {id(other): 'other'})

    def test_the_value_audit_compares_every_chip_and_raises_on_a_difference(self):
        import torch

        values = ('ids', 'ignored', 'values')
        maxima = SimpleNamespace(name='max')
        self.t1.VALUE_REFERENCES[id(values[2])] = maxima
        logical = {id(maxima) + chip: torch.tensor([[1.0], [2.0]]) for chip in range(4)}
        operations = SimpleNamespace(get_device_tensors=lambda value: [SimpleNamespace(address=id(value) + chip) for chip in range(4)],
                                     to_torch=lambda part: logical[part.address])
        lines = []
        with patch('packed_verifier.diagnostic', side_effect=lines.append):
            gathered = [torch.tensor([1.0, 2.0])] * 4
            self.verifier.audit_shard_values(operations, values, 2, gathered)
            self.assertTrue(lines[-1].startswith(tp4_vglue.AUDIT_MARKER))
            with self.assertRaises(AssertionError):
                self.verifier.audit_shard_values(operations, values, 2, gathered[:3] + [torch.tensor([1.0, 3.0])])
        self.assertTrue(lines[-1].startswith(tp4_vglue.AUDIT_MISMATCH) and 'chip=3' in lines[-1])
        with patch('packed_verifier.diagnostic', side_effect=lines.append):
            self.t1.VALUE_REFERENCES.clear()
            before = len(lines)
            self.verifier.audit_shard_values(operations, values, 2, gathered)
            self.assertEqual(len(lines), before, 'no reference recorded: nothing to compare')
            with four(), env(QWEN_FAST_TP4_SHARD_VALUES='1', QWEN_FAST_TP4_VGLUE_AUDIT='1'):
                with self.assertRaises(AssertionError):
                    self.verifier.audit_shard_values(operations, values, 2, gathered)
            self.assertIn('the gather fell back', lines[-1])


if __name__ == '__main__':
    unittest.main()
