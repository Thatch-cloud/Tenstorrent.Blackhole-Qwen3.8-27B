"""The batched packed block: convolution per user, ONE recurrence for all of them.

Two things have to hold. First, the convolution half is untouched - one window DMA and
one conv+gates op per user, in user order, reading that user's own history - because
neither can be batched today (docs/gdn-user-batch-launches.md). Second, what replaces
four recurrence/norm pairs is exactly one `gdn_user_batch.execute`, and every user gets
back a result the retained block cannot tell from the per-user one: its own 16-row prefix
states, its own four windows, the same keys.

The wiring tests then hold `DeviceLoopState` to the same per-segment state moves the
per-user path makes, in the same order, with the native buffer left holding the LAST
user - that invariant is what `commit_user` and any following unpacked decode rely on.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gdn_device_loop_state import DeviceLoopState
from gdn_user_batch_conv import run_user_batched_projected, validate_options
from test_gdn_packed_segments import PackedFixture
from test_gdn_user_batch import FakeTensor


ROWS = 16


def fake_operations(calls):
    operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1')
    counter = [7000]

    def make(name, shape):
        counter[0] += 16
        return FakeTensor(name, shape, counter[0])

    operations.get_device_tensors = Mock(side_effect=lambda value: [
        SimpleNamespace(buffer_address=lambda address=value.address + chip: address) for chip in range(2)])
    operations.deallocate = Mock(side_effect=lambda value: calls.append(('free', value.name)))
    operations.to_memory_config = Mock(side_effect=lambda value, memory: value)

    def record_slice(value, start, stop, memory_config=None):
        calls.append(('slice', value.name, start[2], stop[1]))
        return make('z:' + value.name, (1, stop[1] - start[1], stop[2] - start[2]))

    operations.slice = Mock(side_effect=record_slice)

    def conv_gates(projected, windows, taps, a, b, dt_bias, neg_exp_A, batch=None, **kwargs):
        calls.append(('conv_gates', projected.name, batch, tuple(value.name for value in windows)))
        return [make('conv:' + projected.name, (1, batch, 5120)),
                make('beta:' + projected.name, (1, batch, 24)),
                make('gate:' + projected.name, (1, batch, 24))]

    operations.transformer = SimpleNamespace(gdn_decode_conv_gates=Mock(side_effect=conv_gates))
    operations.make = make
    return operations


def user_group(index, operations, rows=ROWS):
    projected = operations.make('projected%d' % index, (1, rows, 8256))
    initial = operations.make('initial%d' % index, (1, 24, 128, 128))
    states = [operations.make('conv%d.%d' % (index, tap), (1, 1, 5120)) for tap in range(4)]
    return (projected, initial, states)


class BatchedConvTests(unittest.TestCase):
    def build(self, users=4, rows=ROWS):
        calls = []
        operations = fake_operations(calls)
        groups = [user_group(index, operations, rows) for index in range(users)]
        taps = [operations.make('tap%d' % tap, (1, 1, 5120)) for tap in range(4)]
        extras = dict(dt_bias=operations.make('dt', (1, 1, 24)),
                      neg_exp_A=operations.make('nega', (1, 1, 24)),
                      norm_w=operations.make('norm_w', (1, 1, 128)))
        return calls, operations, groups, taps, extras

    def windows(self, calls, operations):
        def build(mesh, projected, history):
            calls.append(('windows', projected.name, tuple(value.name for value in history)))
            return [operations.make('window:%s.%d' % (projected.name, slot), (1, projected.shape[1], 5120))
                    for slot in range(4)]
        return build

    def launch(self, users=4, rows=ROWS, execute=None):
        calls, operations, groups, taps, extras = self.build(users, rows)

        def batched(mesh, inputs, kernels, ops, output_memory=None):
            calls.append(('batched', tuple(value[0].name for value in inputs), len(inputs)))
            return [(operations.make('out%d' % index, (1, rows, 3072)),
                     operations.make('prefix%d' % index, (rows, 24, 128, 128)))
                    for index in range(len(inputs))]

        with patch('gdn_conv_windows.build_windows', side_effect=self.windows(calls, operations)), \
                patch('gdn_user_batch.execute', side_effect=execute or batched) as launch:
            results = run_user_batched_projected('mesh', groups, taps, extras['dt_bias'], extras['neg_exp_A'],
                                                 extras['norm_w'], 'kernels', operations)
        return calls, operations, groups, results, launch

    def test_one_windows_and_one_conv_gates_per_user_then_exactly_one_recurrence(self):
        calls, operations, groups, results, launch = self.launch()
        self.assertEqual(len([entry for entry in calls if entry[0] == 'windows']), 4)
        self.assertEqual(len([entry for entry in calls if entry[0] == 'conv_gates']), 4)
        self.assertEqual(launch.call_count, 1, 'four recurrence/norm pairs become one launch')
        self.assertEqual([entry[0] for entry in calls if entry[0] in ('windows', 'conv_gates', 'batched')],
                         ['windows', 'conv_gates'] * 4 + ['batched'])

    def test_each_user_convolves_from_its_own_history_at_its_own_width(self):
        calls, operations, groups, results, launch = self.launch()
        for index, (projected, initial, states) in enumerate(groups):
            self.assertIn(('windows', projected.name, tuple(value.name for value in states)), calls)
        self.assertEqual([entry[2] for entry in calls if entry[0] == 'conv_gates'], [ROWS] * 4)

    def test_the_single_launch_receives_every_users_own_conv_state_and_z(self):
        calls, operations, groups, results, launch = self.launch()
        inputs = launch.call_args.args[1]
        self.assertEqual(len(inputs), 4)
        for index, (user, (projected, initial, states)) in enumerate(zip(inputs, groups)):
            self.assertEqual(len(user), 6)
            self.assertEqual(user[0].name, 'conv:projected%d' % index)
            self.assertEqual(user[1].name, 'beta:projected%d' % index)
            self.assertEqual(user[2].name, 'gate:projected%d' % index)
            self.assertIs(user[3], initial)
            self.assertEqual(user[4].name, 'z:projected%d' % index)
        self.assertEqual(len({id(user[5]) for user in inputs}), 1, 'one shared norm weight')

    def test_every_result_carries_that_users_own_prefix_geometry_and_windows(self):
        calls, operations, groups, results, launch = self.launch()
        self.assertEqual(len(results), 4)
        for index, result in enumerate(results):
            self.assertEqual(result['states'].shape, (ROWS, 24, 128, 128))
            self.assertEqual(result['output'].shape, (1, ROWS, 3072))
            self.assertEqual(len(result['packed_conv_states']), 4)
            self.assertTrue(all(value.shape == (1, ROWS, 5120) for value in result['packed_conv_states']))
            self.assertEqual(result['conv_prefixes'], [None] * ROWS)
            self.assertTrue(result['packed_checkpoints'])
            self.assertTrue(result['deferred_conv_publication'])
            self.assertTrue(result['user_batched'])
            self.assertFalse(result['norm_batch'])
            self.assertEqual(result['available_conv_prefixes'], tuple(range(1, ROWS + 1)))
            self.assertEqual(result['materialized_conv_prefixes'], ())
            self.assertEqual(result['packed_users'], 4)

    def test_each_result_owns_only_its_own_tensors(self):
        calls, operations, groups, results, launch = self.launch()
        for index, result in enumerate(results):
            names = [value.name for value in result['owned']]
            self.assertIn('out%d' % index, names)
            self.assertIn('prefix%d' % index, names)
            self.assertTrue(all(str(index) in name or name == 'norm_w' for name in names), names)
        everything = [value.name for result in results for value in result['owned']]
        self.assertEqual(len(everything), len(set(everything)), 'no tensor is owned twice')

    def test_a_ragged_block_keeps_every_users_own_width(self):
        calls = []
        operations = fake_operations(calls)
        groups = [user_group(index, operations, rows) for index, rows in enumerate((16, 8))]
        taps = [operations.make('tap%d' % tap, (1, 1, 5120)) for tap in range(4)]

        def batched(mesh, inputs, kernels, ops, output_memory=None):
            return [(operations.make('out%d' % index, (1, value[0].shape[1], 3072)),
                     operations.make('prefix%d' % index, (value[0].shape[1], 24, 128, 128)))
                    for index, value in enumerate(inputs)]

        with patch('gdn_conv_windows.build_windows', side_effect=self.windows(calls, operations)), \
                patch('gdn_user_batch.execute', side_effect=batched):
            results = run_user_batched_projected('mesh', groups, taps, operations.make('dt', (1, 1, 24)),
                operations.make('nega', (1, 1, 24)), operations.make('norm_w', (1, 1, 128)), 'kernels', operations)
        self.assertEqual([result['states'].shape[0] for result in results], [16, 8])
        self.assertEqual([entry[2] for entry in calls if entry[0] == 'conv_gates'], [16, 8])

    def test_a_failed_launch_releases_every_tensor_built_for_it(self):
        def explode(mesh, inputs, kernels, ops, output_memory=None):
            raise RuntimeError('device')

        with self.assertRaises(RuntimeError):
            self.launch(execute=explode)

    def test_the_options_this_path_does_not_serve_are_rejected(self):
        validate_options(True, True, True, False)
        for options in ((False, True, True, False), (True, False, True, False), (True, True, False, False)):
            with self.assertRaisesRegex(ValueError, 'DMA windows, packed checkpoints'):
                validate_options(*options)
        for options in ((1, True, True, False), (True, True, True, 'no')):
            with self.assertRaisesRegex(ValueError, 'Explicit bool'):
                validate_options(*options)

    def test_more_users_than_the_grid_holds_is_rejected_before_any_convolution(self):
        calls, operations, groups, taps, extras = self.build(users=4)
        groups = groups + [user_group(4, operations)]
        with patch('gdn_conv_windows.build_windows', side_effect=self.windows(calls, operations)), \
                patch('gdn_user_batch.execute') as launch:
            with self.assertRaisesRegex(ValueError, 'One to 4 packed'):
                run_user_batched_projected('mesh', groups, taps, extras['dt_bias'], extras['neg_exp_A'],
                                           extras['norm_w'], 'kernels', operations)
        launch.assert_not_called()
        self.assertEqual([entry for entry in calls if entry[0] == 'windows'], [])


class WiringTests(PackedFixture):
    """`DeviceLoopState` with the flag on: one launch, same state moves, same invariant."""

    spans = ((0, 16), (16, 32), (32, 48), (48, 64))
    slots = [['%s%d' % (user, index) for index in range(5)] for user in 'ABCD']
    checkpoints = [['ck' + user] for user in 'ABCD']

    def segment_result(self, calls):
        def run(mesh, users, taps, dt_bias, neg_exp_A, norm_w, kernels, operations=None, **options):
            calls.append(('batched', tuple(piece.name for piece, initial, history in users)))
            calls.append(('histories', tuple((initial, tuple(history)) for piece, initial, history in users)))
            return [dict(output=SimpleNamespace(shape=(1, piece.shape[1], 3072), name='out%d' % index),
                         states=SimpleNamespace(shape=(piece.shape[1], 24, 128, 128)),
                         conv_prefixes=[None] * piece.shape[1], owned=[], packed_conv_states=[],
                         packed_checkpoints=True, deferred_conv_publication=True, norm_batch=False,
                         prefix_zero_reuse=False, user_batched=True)
                    for index, (piece, initial, history) in enumerate(users)]
        return run

    def run_batched(self, state, calls, **options):
        packed = SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1')
        with patch('gdn_device_loop_state.run_user_batched_projected', side_effect=self.segment_result(calls)) as run, \
                patch('gdn_device_loop_state.restore_prefix') as restore, \
                patch('gdn_device_loop_state.copy_compact',
                      side_effect=lambda source, destination: calls.append(('copy_compact', source[0], destination[0]))), \
                patch('gdn_device_loop_state.release_owned'):
            result = state.decode(packed, self.checkpoints, [0] * 4, segments=self.spans,
                                  slots=self.slots, deferred=True, **options)
        return result, run, restore

    def test_the_flag_is_off_unless_the_environment_sets_it(self):
        with patch.dict('os.environ', {}, clear=True):
            state, operations, layer, active, calls = self.build(commit_only=True, users=4)
        self.assertFalse(state.user_batch)
        with patch.dict('os.environ', {'QWEN_FAST_GDN_USER_BATCH': '1'}, clear=True):
            state, operations, layer, active, calls = self.build(commit_only=True, users=4)
        self.assertTrue(state.user_batch)

    def test_the_flag_off_still_launches_one_recurrence_per_user(self):
        state, operations, layer, active, calls = self.build(commit_only=True, users=4, user_batch=False)
        with patch('gdn_device_loop_state.run_batched_projected', side_effect=self.recurrence(calls)), \
                patch('gdn_device_loop_state.copy_compact',
                      side_effect=lambda source, destination: calls.append(('copy_compact', source[0], destination[0]))), \
                patch('gdn_device_loop_state.restore_prefix'), patch('gdn_device_loop_state.release_owned'), \
                patch('gdn_device_loop_state.run_user_batched_projected') as batched:
            state.decode(SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1'),
                         self.checkpoints, [0] * 4, segments=self.spans, slots=self.slots, deferred=True)
        batched.assert_not_called()
        self.assertEqual([entry for entry in calls if entry[0] == 'recur'], [('recur', 16)] * 4)

    def test_the_flag_on_replaces_four_recurrences_with_one_launch(self):
        state, operations, layer, active, calls = self.build(commit_only=True, users=4, user_batch=True)
        result, run, restore = self.run_batched(state, calls)
        self.assertEqual(run.call_count, 1)
        self.assertEqual([entry for entry in calls if entry[0] == 'recur'], [])
        self.assertEqual([entry for entry in calls if entry[0] == 'batched'],
                         [('batched', ('piece:0', 'piece:16', 'piece:32', 'piece:48'))])
        # Two, because this fixture's layer carries no M3 graft: one native call per
        # 32-row tile of the 64-row block, joined on the row axis. Never one per user.
        self.assertEqual(layer._project_qkvzab_raw.call_count, 2, 'tile count, never user count')
        self.assertEqual(result['output'].shape, (1, 64, 3072))
        self.assertEqual(result['segments'], self.spans)
        self.assertEqual(len(result['segment_results']), 4)
        self.assertEqual(state.checkpoint_calls, 4)
        self.assertEqual(state.calls, 1)
        restore.assert_not_called()

    def test_every_user_reaches_the_launch_from_its_own_carried_state(self):
        state, operations, layer, active, calls = self.build(commit_only=True, users=4, user_batch=True)
        self.run_batched(state, calls)
        order = [entry for entry in calls if entry[0] in ('copy_compact', 'restore', 'save', 'batched')]
        entries = state.segment_entries
        self.assertEqual(order, [
            ('copy_compact', 'A0', entries[0][0]),
            ('copy_compact', 'B0', entries[1][0]),
            ('copy_compact', 'C0', entries[2][0]),
            ('restore', 'D0'), ('save', entries[3][0]),
            ('batched', ('piece:0', 'piece:16', 'piece:32', 'piece:48')),
        ], 'three compact moves, the last user through the native buffer, then one launch')
        histories = [entry for entry in calls if entry[0] == 'histories'][0][1]
        self.assertEqual([initial for initial, history in histories], [entry[0] for entry in entries])
        self.assertEqual([history for initial, history in histories], [tuple(entry[1:]) for entry in entries])

    def test_the_native_buffer_is_left_holding_the_last_user(self):
        state, operations, layer, active, calls = self.build(commit_only=True, users=4, user_batch=True)
        self.run_batched(state, calls)
        self.assertEqual([entry for entry in calls if entry[0] == 'restore'], [('restore', 'D0')])
        self.assertEqual(active.save.call_count, 1)

    def test_the_batched_path_refuses_anything_but_a_deferred_packed_block(self):
        # No commit_only and no per-user entries: a RETAINED packed block, which decides
        # at decode. The batched path publishes nothing, so it must refuse this.
        state, operations, layer, active, calls = self.build(user_batch=True, defer_conv_publication=True)
        packed = SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1')
        with patch('gdn_device_loop_state.run_user_batched_projected') as run, \
                patch('gdn_device_loop_state.copy_compact'), patch('gdn_device_loop_state.release_owned'), \
                patch('gdn_device_loop_state.restore_prefix'):
            with self.assertRaisesRegex(ValueError, 'deferred packed decode'):
                state.decode(packed, self.checkpoints, [0] * 4, segments=self.spans, slots=self.slots)
        run.assert_not_called()

    def test_the_flag_requires_the_packed_batched_deferred_configuration(self):
        active = SimpleNamespace(direct=True, gdn=SimpleNamespace(B=8, _stable_state=True, rec_state='rec',
                                 conv_states=['c0', 'c1', 'c2', 'c3']),
                                 live=['rec', 'c0', 'c1', 'c2', 'c3'])
        operations = SimpleNamespace(get_device_tensors=lambda value: [
            SimpleNamespace(buffer_address=lambda: 1), SimpleNamespace(buffer_address=lambda: 2)])
        for options in (dict(), dict(batch_conv=True), dict(batch_conv=True, dma_windows=True),
                        dict(batch_conv=True, dma_windows=True, packed_checkpoints=True)):
            with self.assertRaisesRegex(ValueError, 'User-batched GDN requires'):
                DeviceLoopState(active, operations, 'kernels', user_batch=True, **options)
        for value in (1, 'yes', 0):
            with self.assertRaisesRegex(ValueError, 'Explicit bool user-batched'):
                DeviceLoopState(active, operations, 'kernels', user_batch=value)

    def test_a_launch_that_loses_a_user_or_a_users_width_is_caught(self):
        for broken in ('short', 'wide', 'norm', 'flag'):
            state, operations, layer, active, calls = self.build(commit_only=True, users=4, user_batch=True)
            base = self.segment_result(calls)

            def damaged(*args, _base=base, _broken=broken, **kwargs):
                results = _base(*args, **kwargs)
                if _broken == 'short':
                    return results[:3]
                if _broken == 'wide':
                    results[2]['states'] = SimpleNamespace(shape=(32, 24, 128, 128))
                if _broken == 'norm':
                    results[1]['norm_batch'] = True
                if _broken == 'flag':
                    results[0]['user_batched'] = False
                return results

            packed = SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1')
            with patch('gdn_device_loop_state.run_user_batched_projected', side_effect=damaged), \
                    patch('gdn_device_loop_state.copy_compact'), patch('gdn_device_loop_state.release_owned'), \
                    patch('gdn_device_loop_state.restore_prefix'):
                with self.assertRaises(AssertionError):
                    state.decode(packed, self.checkpoints, [0] * 4, segments=self.spans,
                                 slots=self.slots, deferred=True)


if __name__ == '__main__':
    unittest.main()
