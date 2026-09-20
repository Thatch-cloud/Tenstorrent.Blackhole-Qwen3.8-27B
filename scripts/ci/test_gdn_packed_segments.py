"""Packed GDN: one weight pass, one recurrence per user, from that user's state.

The row axis of a verify block is TIME for the GDN. Packed it carries one segment
per user, so the recurrence has to restart from that user's own carried state at
every boundary - otherwise user B continues user A's recurrence, which is wrong
for every row of B and leaves A advanced by the whole block.

What must NOT change is where the weights are read. `_project_qkvzab_raw` runs
once across all rows and `finish_output` runs the single output projection over
the concatenated rows, so packing still costs one pass over the weights. If either
of those ran per segment the pack would buy nothing, which is why both are asserted
here rather than assumed.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gdn_device_loop_state import DeviceLoopState, validate_segments


class PackedFixture(unittest.TestCase):
    """One GDN layer's device-loop state over fakes that record every device call."""

    def build(self, **options):
        calls = []
        operations = SimpleNamespace(L1_MEMORY_CONFIG='l1', DRAM_MEMORY_CONFIG='dram')

        def record_slice(value, start, stop, *args, **kwargs):
            calls.append(('slice', start[1], stop[1]))
            return SimpleNamespace(shape=(1, stop[1] - start[1], 5120), name='piece')

        operations.slice = Mock(side_effect=record_slice)
        operations.concat = Mock(side_effect=lambda parts, **kwargs: SimpleNamespace(
            shape=(1, sum(part.shape[1] for part in parts), 5120), name='joined'))
        # addresses() runs in __init__, before any patch a test could install
        operations.get_device_tensors = Mock(side_effect=lambda tensor: [
            SimpleNamespace(buffer_address=lambda value=str(tensor): hash(value)) for _ in range(2)])
        operations.deallocate = Mock()

        layer = SimpleNamespace(B=8, _stable_state=True, rec_state='rec',
            conv_states=['c0', 'c1', 'c2', 'c3'], mesh='mesh',
            tw={'conv_taps': (), 'dt_bias': None, 'neg_exp_A': None, 'norm_w': None, 'out': None})
        layer._project_qkvzab_raw = Mock(side_effect=lambda packed, rows, memory: (
            calls.append(('project', rows)) or SimpleNamespace(shape=(1, rows, 8240), name='projected')))

        allocations = []

        def allocate():
            allocations.append(['E%d.%d' % (len(allocations), index) for index in range(5)])
            return allocations[-1]

        active = SimpleNamespace(direct=True, gdn=layer, live=['rec', 'c0', 'c1', 'c2', 'c3'])
        active.allocate = Mock(side_effect=allocate)
        active.save = Mock(side_effect=lambda destination: calls.append(('save', destination[0])))
        active.restore = Mock(side_effect=lambda source: calls.append(('restore', source[0])))

        state = DeviceLoopState(active, operations, kernels='kernels', batch_conv=True,
                                dma_windows=True, packed_checkpoints=True, **options)
        return state, operations, layer, active, calls

    def recurrence(self, calls):
        def run(mesh, projected, initial, conv_states, *args, **kwargs):
            calls.append(('recur', projected.shape[1]))
            calls.append(('history', initial, tuple(conv_states)))
            return dict(output=SimpleNamespace(shape=(1, projected.shape[1], 5120), name='out'),
                        states=SimpleNamespace(shape=(projected.shape[1], 24, 128, 128)),
                        conv_prefixes=[None] * projected.shape[1], owned=[],
                        packed_checkpoints=True,
                        deferred_conv_publication=kwargs.get('defer_conv_publication', False),
                        norm_batch=False, prefix_zero_reuse=False)
        return run

    def run_packed(self, state, operations, calls, spans, prefixes, slots, checkpoints, **options):
        with patch('gdn_device_loop_state.run_batched_projected', side_effect=self.recurrence(calls)), \
                patch('gdn_device_loop_state.restore_prefix',
                      side_effect=lambda ops, result, entry, destination, accepted:
                          calls.append(('prefix', destination[0] if isinstance(destination, list) else destination, accepted))), \
                patch('gdn_device_loop_state.copy_compact'), \
                patch('gdn_device_loop_state.release_owned'), \
                patch('gdn_device_loop_state.norm_batch_enabled', return_value=False):
            return state.decode(SimpleNamespace(shape=(1, 32, 5120)), checkpoints, prefixes,
                                segments=spans, slots=slots, **options)


class PackedSegmentTests(PackedFixture):
    def test_one_projection_serves_every_segment(self):
        state, operations, layer, active, calls = self.build()
        slots = [['A%d' % index for index in range(5)], ['B%d' % index for index in range(5)]]
        result = self.run_packed(state, operations, calls, ((0, 16), (16, 32)), [12, 9], slots,
                                 [['ckA'], ['ckB']])
        self.assertEqual(layer._project_qkvzab_raw.call_count, 1,
                         'the input projection is the weight pass and must run once')
        self.assertEqual([entry for entry in calls if entry[0] == 'project'], [('project', 32)])
        self.assertEqual([entry for entry in calls if entry[0] == 'recur'], [('recur', 16), ('recur', 16)])
        self.assertEqual([entry for entry in calls if entry[0] == 'slice'],
                         [('slice', 0, 16), ('slice', 16, 32)])
        self.assertEqual(result['output'].shape, (1, 32, 5120))
        self.assertEqual(result['segments'], ((0, 16), (16, 32)))
        self.assertIs(result['segment_results'], state.segment_results)
        self.assertEqual(len(state.segment_results), 2)

    def test_each_segment_starts_from_its_own_carried_state(self):
        state, operations, layer, active, calls = self.build()
        slots = [['A%d' % index for index in range(5)], ['B%d' % index for index in range(5)]]
        self.run_packed(state, operations, calls, ((0, 16), (16, 32)), [12, 9], slots, [['ckA'], ['ckB']])
        order = [entry for entry in calls if entry[0] in ('restore', 'recur')]
        self.assertEqual(order, [('restore', 'A0'), ('recur', 16), ('restore', 'B0'), ('recur', 16)],
                         'each user is restored immediately before its own recurrence')

    def test_each_user_checkpoints_its_own_prefix_and_advances_only_its_own_rows(self):
        state, operations, layer, active, calls = self.build()
        slots = [['A%d' % index for index in range(5)], ['B%d' % index for index in range(5)]]
        self.run_packed(state, operations, calls, ((0, 16), (16, 32)), [12, 9], slots, [['ckA'], ['ckB']])
        self.assertEqual([entry for entry in calls if entry[0] == 'prefix'],
                         [('prefix', 'ckA', 12), ('prefix', 'A0', 16),
                          ('prefix', 'ckB', 9), ('prefix', 'B0', 16)],
                         'accepted prefix into the checkpoint, own width into the carried state')

    def test_unpacked_decode_is_unchanged(self):
        state, operations, layer, active, calls = self.build()
        with patch('gdn_device_loop_state.run_batched_projected', side_effect=self.recurrence(calls)), \
                patch('gdn_device_loop_state.restore_prefix') as prefix, \
                patch('gdn_device_loop_state.copy_compact'), \
                patch('gdn_device_loop_state.release_owned'), \
                patch('gdn_device_loop_state.norm_batch_enabled', return_value=False):
            result = state.decode(SimpleNamespace(shape=(1, 32, 5120)), ['ck'], 12)
        self.assertEqual([entry for entry in calls if entry[0] == 'recur'], [('recur', 32)])
        operations.slice.assert_not_called()
        operations.concat.assert_not_called()
        self.assertEqual(prefix.call_count, 2, 'checkpoint and end-of-block state')
        self.assertNotIn('segments', result)
        self.assertEqual(state.checkpoint_calls, 1)
        self.assertIsNone(state.segment_results)
        self.assertIsNone(state.segment_entries)


class DeferredPackedTests(PackedFixture):
    """The packed serving step: prefixes are known only after the verify readback.

    So the decode restores each user from its carry, snapshots that block-start state
    into the user's OWN entry, runs the recurrence, and decides nothing. The retained
    block's commit_user later rebuilds any accepted prefix from that entry and that
    user's histories, straight into the carry - which is why the carry must be left
    exactly as decode found it: a prefix of zero then has nothing to write.
    """

    spans = ((0, 16), (16, 32))
    slots = [['A%d' % index for index in range(5)], ['B%d' % index for index in range(5)]]
    checkpoints = [['ckA'], ['ckB']]

    def test_every_user_keeps_its_own_entry_and_nothing_is_decided_at_decode(self):
        state, operations, layer, active, calls = self.build(commit_only=True, users=2)
        result = self.run_packed(state, operations, calls, self.spans, [0, 0], self.slots, self.checkpoints,
                                 deferred=True)
        self.assertEqual(layer._project_qkvzab_raw.call_count, 1, 'still one weight pass')
        first, second = state.segment_entries
        self.assertEqual((state.entry, state.state), ([], []), 'no shared entry, no speculative state')
        self.assertNotEqual(first, second)
        self.assertEqual([entry for entry in calls if entry[0] in ('restore', 'save', 'recur')],
                         [('restore', 'A0'), ('save', first[0]), ('recur', 16),
                          ('restore', 'B0'), ('save', second[0]), ('recur', 16)],
                         'each user: restored from its carry, snapshotted into its own entry, run')
        self.assertEqual([entry for entry in calls if entry[0] == 'history'],
                         [('history', first[0], tuple(first[1:])), ('history', second[0], tuple(second[1:]))],
                         'the recurrence starts from that user\'s own entry, read-only')
        self.assertEqual([entry for entry in calls if entry[0] == 'prefix'], [],
                         'neither the checkpoint nor the carry is written before the readback')
        self.assertEqual([piece['states'].shape[0] for piece in state.segment_results], [16, 16])
        self.assertIs(result['segment_results'], state.segment_results)
        self.assertTrue(all(piece['deferred_conv_publication'] for piece in state.segment_results))
        self.assertTrue(result['commit_only_gdn'])
        self.assertEqual(result['segments'], self.spans)
        self.assertEqual((state.calls, state.checkpoint_calls), (1, 2), 'one deferred decision per user')

    def test_deferred_decode_needs_commit_only_and_one_entry_per_user(self):
        with self.assertRaises(ValueError):
            self.build(users=2)
        for options, deferred in (({}, True), (dict(commit_only=True), False), (dict(commit_only=True), True),
                                  (dict(commit_only=True, users=3), True), (dict(commit_only=True, users=2), 1)):
            state, operations, layer, active, calls = self.build(**options)
            with self.assertRaises(ValueError):
                self.run_packed(state, operations, calls, self.spans, [0, 0], self.slots, self.checkpoints,
                                deferred=deferred)
            layer._project_qkvzab_raw.assert_not_called()
            active.restore.assert_not_called()
        # a single sequence decides at decode; it has nothing to defer
        state, operations, layer, active, calls = self.build()
        with self.assertRaises(ValueError):
            state.decode(SimpleNamespace(shape=(1, 32, 5120)), ['ck'], 12, deferred=True)
        layer._project_qkvzab_raw.assert_not_called()

    def test_close_frees_every_users_entry(self):
        state, operations, layer, active, calls = self.build(commit_only=True, users=2)
        entries = [list(entry) for entry in state.segment_entries]
        with patch('gdn_device_loop_state.release_owned') as release:
            state.close()
        release.assert_called_once_with(operations, [value for entry in entries for value in entry])
        self.assertEqual([list(entry) for entry in state.segment_entries], [[], []])
        self.assertIsNone(state.segment_results)


class SegmentValidationTests(unittest.TestCase):
    def test_spans_must_cover_the_block_in_order(self):
        for spans in (((0, 16),), ((0, 16), (17, 32)), ((16, 32), (0, 16)), ((0, 32), (32, 32))):
            with self.assertRaises(ValueError):
                validate_segments(spans, [None, None], [1, 1], 32, [None, None])

    def test_every_user_needs_a_checkpoint_a_prefix_and_carried_state(self):
        spans = ((0, 16), (16, 32))
        for checkpoint, prefix, slots in ((['a'], [1, 1], [1, 2]), ([None, None], 5, [1, 2]),
                                          ([None, None], [1, 1], None), ([None, None], [1, 1], [1])):
            with self.assertRaises(ValueError):
                validate_segments(spans, checkpoint, prefix, 32, slots)

    def test_an_accepted_prefix_must_fit_its_own_segment(self):
        with self.assertRaises(ValueError):
            validate_segments(((0, 16), (16, 32)), [None, None], [17, 1], 32, [1, 2])

    def test_unpacked_still_requires_an_integer_prefix_and_no_slots(self):
        self.assertEqual(validate_segments(None, None, 12, 32, None), (None, None, None))
        with self.assertRaises(ValueError):
            validate_segments(None, None, 33, 32, None)
        with self.assertRaises(ValueError):
            validate_segments(None, None, 12, 32, ['slot'])


if __name__ == '__main__':
    unittest.main()
