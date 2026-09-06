from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gdn_device_loop_state import DeviceLoopState


class DeviceLoopStateTests(unittest.TestCase):
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

    def fixture(self):
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
        return DeviceLoopState(active, operations, 'kernels'), active

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
