from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gdn_device_loop_state import DeviceLoopState
from test_gdn_packed_segments import PackedFixture


class DeviceLoopStateTests(unittest.TestCase):
    def test_commit_only_allocates_no_speculative_working_state(self):
        adapter, active = self.fixture(commit_only=True)
        self.assertEqual(adapter.state, [])
        active.allocate.assert_called_once_with()

    def test_commit_only_retains_histories_without_speculative_state_publication(self):
        for rows in (2, 4, 8, 16, 32):
            for prefix in range(rows + 1):
                adapter, active = self.fixture()
                adapter.commit_only = adapter.batch_conv = adapter.dma_windows = adapter.packed_checkpoints = True
                with patch('gdn_device_loop_state.copy_compact') as copy, \
                        patch('gdn_device_loop_state.run_batched_projected', return_value=dict(
                            owned=[], deferred_conv_publication=True)) as run, \
                        patch('gdn_device_loop_state.restore_prefix') as restore:
                    result = adapter.decode(SimpleNamespace(shape=(1, rows, 5120)), [], prefix)
                active.save.assert_called_once_with(adapter.entry)
                copy.assert_not_called()
                restore.assert_not_called()
                active.restore.assert_not_called()
                self.assertEqual(run.call_args.args[3], adapter.entry[1:])
                self.assertTrue(result['commit_only_gdn'])

    def test_commit_only_requires_retained_geometry_and_preserves_native_t1(self):
        adapter, active = self.fixture()
        for invalid in (True, 1, '1', None):
            with self.assertRaises(ValueError):
                DeviceLoopState(active, adapter.operations, 'kernels', commit_only=invalid)
        adapter.commit_only = True
        with self.assertRaises(ValueError):
            adapter.decode(SimpleNamespace(shape=(1, 1, 5120)), [], 0)
        active.save.assert_not_called()

    def test_norm_source_override_is_forwarded_without_changing_publication(self):
        adapter, active = self.fixture()
        adapter.batch_conv = adapter.dma_windows = adapter.packed_checkpoints = adapter.norm_batch = True
        adapter.norm_source_root = '/audited/source'
        with patch('gdn_device_loop_state.copy_compact'), \
                patch('gdn_device_loop_state.run_batched_projected', return_value=dict(owned=[], norm_batch=True)) as run, \
                patch('gdn_device_loop_state.restore_prefix'):
            adapter.decode(SimpleNamespace(shape=(1, 8, 5120)), [], 0)
        self.assertEqual(run.call_args.kwargs['norm_source_root'], '/audited/source')
        active.restore.assert_called_once_with(adapter.state)

    def test_deferred_publication_skips_initial_copy_but_preserves_both_restores(self):
        for rows in (1, 2, 4, 8, 16, 32):
            for prefix in (0, rows):
                adapter, active = self.fixture()
                adapter.batch_conv = adapter.dma_windows = adapter.packed_checkpoints = adapter.defer_conv_publication = True
                with patch('gdn_device_loop_state.copy_compact') as copy, \
                        patch('gdn_device_loop_state.run_batched_projected', return_value=
                            dict(owned=[], deferred_conv_publication=rows > 1)) as run, \
                        patch('gdn_device_loop_state.restore_prefix') as restore:
                    adapter.decode(SimpleNamespace(shape=(1, rows, 5120)), [], prefix)
                self.assertEqual(copy.call_count, int(rows == 1))
                history = adapter.entry if rows > 1 else adapter.state
                self.assertEqual(run.call_args.args[3], history[1:])
                self.assertEqual(run.call_args.kwargs.get('defer_conv_publication', False), rows > 1)
                self.assertEqual([call.args[-1] for call in restore.call_args_list], [prefix, rows])
                active.restore.assert_called_once_with(adapter.state)

    def test_deferred_publication_must_engage_before_live_state_is_written(self):
        adapter, active = self.fixture()
        adapter.batch_conv = adapter.dma_windows = adapter.packed_checkpoints = adapter.defer_conv_publication = True
        with patch('gdn_device_loop_state.copy_compact'), patch('gdn_device_loop_state.release_owned'), \
                patch('gdn_device_loop_state.run_batched_projected', return_value=dict(owned=[])), \
                patch('gdn_device_loop_state.restore_prefix') as restore:
            with self.assertRaisesRegex(AssertionError, 'did not engage'):
                adapter.decode(SimpleNamespace(shape=(1, 8, 5120)), [], 0)
        restore.assert_not_called()
        active.restore.assert_not_called()

    def test_prefix_reuse_is_forwarded_and_must_engage(self):
        adapter, active = self.fixture()
        adapter.batch_conv = adapter.dma_windows = adapter.packed_checkpoints = adapter.prefix_zero_reuse = True
        for engaged in (True, False):
            with patch('gdn_device_loop_state.copy_compact'), patch('gdn_device_loop_state.release_owned'), \
                    patch('gdn_device_loop_state.restore_prefix'), patch('gdn_device_loop_state.run_batched_projected',
                        return_value=dict(owned=[], prefix_zero_reuse=engaged)) as run:
                if engaged:
                    adapter.decode(SimpleNamespace(shape=(1, 4, 5120)), [], 2)
                else:
                    with self.assertRaisesRegex(AssertionError, 'did not engage'):
                        adapter.decode(SimpleNamespace(shape=(1, 4, 5120)), [], 2)
                self.assertTrue(run.call_args.kwargs['prefix_zero_reuse'])

    def test_norm_batch_uses_same_publication_and_checks_engagement(self):
        adapter, active = self.fixture()
        adapter.batch_conv = adapter.dma_windows = adapter.packed_checkpoints = adapter.norm_batch = True
        with patch('gdn_device_loop_state.copy_compact'), \
                patch('gdn_device_loop_state.run_batched_projected', return_value=dict(owned=[], norm_batch=True)) as run, \
                patch('gdn_device_loop_state.restore_prefix') as restore:
            adapter.decode(SimpleNamespace(shape=(1, 8, 5120)), [], 3)
        self.assertEqual(run.call_args.kwargs, dict(dma_windows=True, packed_checkpoints=True, norm_batch=True))
        self.assertEqual([call.args[-1] for call in restore.call_args_list], [3, 8])
        active.restore.assert_called_once_with(adapter.state)
        active.restore.reset_mock()
        with patch('gdn_device_loop_state.copy_compact'), \
                patch('gdn_device_loop_state.run_batched_projected', return_value=dict(owned=[])), \
                patch('gdn_device_loop_state.release_owned'):
            with self.assertRaisesRegex(AssertionError, 'did not engage'):
                adapter.decode(SimpleNamespace(shape=(1, 8, 5120)), [], 3)
        active.restore.assert_not_called()

    def test_packed_checkpoints_are_opt_in(self):
        adapter, active = self.fixture()
        self.assertFalse(adapter.packed_checkpoints)
        adapter.batch_conv = adapter.dma_windows = adapter.packed_checkpoints = True
        with patch('gdn_device_loop_state.copy_compact'), \
                patch('gdn_device_loop_state.run_batched_projected', return_value=dict(owned=[])) as run, \
                patch('gdn_device_loop_state.restore_prefix'):
            adapter.decode(SimpleNamespace(shape=(1, 4, 5120)), [], 1)
        self.assertEqual(run.call_args.kwargs, dict(dma_windows=True, packed_checkpoints=True))

    def test_batched_dma_uses_same_selected_and_final_publication(self):
        adapter, active = self.fixture()
        adapter.compact_prologue = adapter.batch_conv = adapter.dma_windows = True
        with patch('gdn_device_loop_state.copy_compact'), \
                patch('gdn_device_loop_state.run_batched_projected', return_value=dict(owned=[])) as run, \
                patch('gdn_device_loop_state.run_projected') as serial, \
                patch('gdn_device_loop_state.restore_prefix') as restore:
            adapter.decode(SimpleNamespace(shape=(1, 4, 5120)), [], 2)
        self.assertEqual(run.call_args.kwargs, dict(conv_checkpoints=(2, 4), hoist_input=True, dma_windows=True))
        serial.assert_not_called()
        self.assertEqual([call.args[-1] for call in restore.call_args_list], [2, 4])
        active.restore.assert_called_once_with(adapter.state)

    def test_compact_prologue_selects_checkpoint_and_final_only(self):
        adapter, active = self.fixture()
        adapter.compact_prologue = True
        for prefix, expected in ((0, (4,)), (2, (2, 4)), (4, (4,))):
            with patch('gdn_device_loop_state.copy_compact'), \
                    patch('gdn_device_loop_state.run_projected', return_value=dict(owned=[])) as run, \
                    patch('gdn_device_loop_state.restore_prefix'):
                adapter.decode(SimpleNamespace(shape=(1, 4, 5120)), [], prefix)
            self.assertEqual(run.call_args.kwargs, dict(conv_checkpoints=expected, hoist_input=True))

    def fixture(self, *, commit_only=False):
        live = [object() for index in range(5)]
        entry = [object() for index in range(5)]
        state = [object() for index in range(5)]
        layer = SimpleNamespace(B=8, _stable_state=True, rec_state=live[0], conv_states=live[1:], mesh='mesh',
            tw=dict(conv_taps=[1, 2, 3, 4], dt_bias='bias', neg_exp_A='decay', norm_w='norm'),
            _project_qkvzab_raw=Mock(return_value=object()))
        active = SimpleNamespace(direct=True, gdn=layer, live=live,
            allocate=Mock(side_effect=[entry, state]), save=Mock(), restore=Mock())
        operations = SimpleNamespace(L1_MEMORY_CONFIG='l1',
            get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda: id(value))] * 2)
        return DeviceLoopState(active, operations, 'kernels', commit_only=commit_only,
            batch_conv=commit_only, dma_windows=commit_only, packed_checkpoints=commit_only), active

    def test_selected_checkpoint_and_final_state_are_distinct_publications(self):
        adapter, active = self.fixture()
        checkpoint = [object() for index in range(5)]
        result = dict(owned=[])
        with patch('gdn_device_loop_state.copy_compact') as copy, \
                patch('gdn_device_loop_state.run_projected', return_value=result), \
                patch('gdn_device_loop_state.restore_prefix') as restore:
            self.assertIs(adapter.decode(SimpleNamespace(shape=(1, 4, 5120)), checkpoint, 2), result)
        active.save.assert_called_once_with(adapter.entry)
        copy.assert_called_once_with(adapter.entry, adapter.state)
        self.assertEqual([call.args[-1] for call in restore.call_args_list], [2, 4])
        self.assertIs(restore.call_args_list[0].args[-2], checkpoint)
        active.restore.assert_called_once_with(adapter.state)
        self.assertEqual((adapter.calls, adapter.checkpoint_calls, adapter.gdn.B), (1, 1, 8))
        self.assertIs(adapter.gdn.rec_state, active.live[0])

    def test_failure_does_not_publish_live_state(self):
        adapter, active = self.fixture()
        with patch('gdn_device_loop_state.copy_compact'), \
                patch('gdn_device_loop_state.run_projected', side_effect=RuntimeError('kernel')), \
                patch('gdn_device_loop_state.release_owned') as release:
            with self.assertRaisesRegex(RuntimeError, 'kernel'):
                adapter.decode(SimpleNamespace(shape=(1, 4, 5120)), [], 2)
        active.restore.assert_not_called()
        release.assert_called_once()
        self.assertEqual(adapter.calls, 0)

    def test_invalid_prefix_and_changed_binding_rejected_before_save(self):
        for prefix in (-1, 5, True):
            adapter, active = self.fixture()
            with self.assertRaises(ValueError):
                adapter.decode(SimpleNamespace(shape=(1, 4, 5120)), [], prefix)
            active.save.assert_not_called()
        adapter, active = self.fixture()
        adapter.gdn.B = 1
        with self.assertRaises(ValueError):
            adapter.decode(SimpleNamespace(shape=(1, 4, 5120)), [], 2)
        active.save.assert_not_called()


class PackedLaunchReductionTests(PackedFixture):
    """T32: overlay-only reduction of GDN launches in the four-user packed decode round.

    Every packed segment but the last moved its carried state into its own entry by
    round-tripping through the native 8-slot active buffer: `active.restore(slot)`
    (one launch, compact source -> native row 0) then, inside `_recurrence`,
    `active.save(entry)` (one launch, native row 0 -> compact destination). Both `slot`
    and `entry` are compact-shaped, and the native buffer never needed to hold that
    value for any reason - nothing reads it between the two calls - so the pair is
    replaced with a single `copy_compact(slot, entry)` launch (compact -> compact,
    directly) for every segment except the last.

    The LAST segment keeps the real `active.restore` + `active.save` round trip,
    because `_decode_packed` documents that the native buffer must end a packed round
    holding the LAST segment's user (an unpacked decode on the same layer afterwards
    would otherwise inherit an arbitrary one). That is the one invariant this overlay
    must not break, so N segments cost N+1 launches here, not N: three single-launch
    segments plus one two-launch segment for four users, saving 3 launches per packed
    round per GDN layer rather than 4 - see gdn_device_loop_state.py's `_decode_packed`
    for why the last segment cannot be reduced further without losing that invariant.
    """

    spans = ((0, 16), (16, 32), (32, 48), (48, 64))
    slots = [['%s%d' % (name, index) for index in range(5)] for name in 'ABCD']
    checkpoints = [['ck%s' % name] for name in 'ABCD']

    def propagating(self, active, calls):
        """Rewire active.restore/active.save and copy_compact to actually move the
        marker values they are given (mirroring the real no-arithmetic device copy),
        so the test can check entries and the native buffer end up holding the right
        values - not just that the right calls were made in the right order."""
        native = [None] * 5

        def restore(source):
            calls.append(('restore', source[0]))
            native[:] = list(source)

        def save(destination):
            calls.append(('save', destination[0]))
            destination[:] = list(native)

        def copy_compact(source, destination):
            calls.append(('copy_compact', source[0], destination[0]))
            destination[:] = list(source)

        active.restore.side_effect = restore
        active.save.side_effect = save
        return native, copy_compact

    def test_four_segment_packed_decode_reduces_launches_and_keeps_every_value(self):
        state, operations, layer, active, calls = self.build(commit_only=True, users=4)
        native, copy_compact = self.propagating(active, calls)
        packed = SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1')
        with patch('gdn_device_loop_state.run_batched_projected', side_effect=self.recurrence(calls)), \
                patch('gdn_device_loop_state.copy_compact', side_effect=copy_compact), \
                patch('gdn_device_loop_state.release_owned'), \
                patch('gdn_device_loop_state.norm_batch_enabled', return_value=False):
            state.decode(packed, self.checkpoints, [0] * 4, segments=self.spans, slots=self.slots, deferred=True)

        # Launch sequence per segment, in order: one copy_compact launch each for the
        # first three segments (A, B, C), then a restore+save pair (two launches) for
        # the last (D) - never the old two-launch restore+save pair for every segment.
        launches = [entry for entry in calls if entry[0] in ('restore', 'save', 'copy_compact')]
        self.assertEqual(launches, [
            ('copy_compact', 'A0', 'E0.0'),
            ('copy_compact', 'B0', 'E1.0'),
            ('copy_compact', 'C0', 'E2.0'),
            ('restore', 'D0'), ('save', 'E3.0'),
        ])
        self.assertEqual(len(launches), 5, 'three single-launch segments plus one '
                         'two-launch segment: 5 launches, not the original 8 (2 per segment)')

        # Every entry ends up holding exactly its own segment's slot values, bit for
        # bit - copy_compact and the restore/save pair are both bit-exact, no-arithmetic
        # moves (gdn_state_copy.py), so the mechanism differs but the values must not.
        for slot, entry in zip(self.slots, state.segment_entries):
            self.assertEqual(list(entry), slot)

        # The native buffer is documented to end a packed round holding the LAST
        # segment's user - D - and nothing else, since the first three segments never
        # touch it at all under the reduction.
        self.assertEqual(native, self.slots[-1])

    def test_single_user_branch_is_unaffected_by_the_packed_launch_reduction(self):
        """`_recurrence`'s new `slot` parameter defaults to None, and the single-user
        (spans-is-None) call site in `decode` never passes one, so this path is exactly
        what it was: entry is populated by reading the native buffer's current row 0
        through `active.save`, never by a `copy_compact` sourced from a packed segment's
        slot - there is no such slot on this path at all."""
        state, operations, layer, active, calls = self.build()
        native, copy_compact = self.propagating(active, calls)
        native[:] = ['native-rec', 'native-c0', 'native-c1', 'native-c2', 'native-c3']
        packed = SimpleNamespace(shape=(1, 32, 5120), name='packed', memory_config=lambda: 'l1')
        with patch('gdn_device_loop_state.run_batched_projected', side_effect=self.recurrence(calls)), \
                patch('gdn_device_loop_state.restore_prefix'), \
                patch('gdn_device_loop_state.copy_compact', side_effect=copy_compact), \
                patch('gdn_device_loop_state.release_owned'), \
                patch('gdn_device_loop_state.norm_batch_enabled', return_value=False):
            state.decode(packed, ['ck'], 12)

        # The very first device touch of the native buffer is a save reading its
        # current row 0 into entry - a copy_compact sourced from a slot never happens,
        # because there is no packed segment here for one to come from. (The logged
        # tag is entry's own pre-call marker id, 'E0.0' - the same convention every
        # other fixture call here uses.)
        first_native_touch = next(entry for entry in calls if entry[0] in ('save', 'copy_compact'))
        self.assertEqual(first_native_touch, ('save', 'E0.0'))
        self.assertEqual(list(state.entry), ['native-rec', 'native-c0', 'native-c1', 'native-c2', 'native-c3'],
                         "entry took the native buffer's row 0 as it stood, unchanged from before this overlay")
