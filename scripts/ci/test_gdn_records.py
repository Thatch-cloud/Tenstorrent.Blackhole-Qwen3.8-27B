from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gdn_records import RetainedGDNBlock


class RecordTests(unittest.TestCase):
    def fixture(self, count=48):
        operations = SimpleNamespace(get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda: id(value))] * 2,
                                     synchronize_device=Mock())
        block = RetainedGDNBlock(4, operations)
        mesh = object()
        for index in range(count):
            live = [object() for slot in range(5)]
            state = SimpleNamespace(gdn=SimpleNamespace(B=8, _stable_state=True, rec_state=live[0], conv_states=live[1:], mesh=mesh),
                native_addresses=[(id(value), id(value)) for value in live], entry=[object()] * 5,
                active=SimpleNamespace(restore=Mock()))
            result = dict(packed_checkpoints=True, states=SimpleNamespace(shape=(4, 24, 128, 128)), owned=[object()])
            block.append(state, result, [object() for slot in range(5)])
        return block

    def test_replay_rearms_only_after_synchronized_commit_and_execution(self):
        block = self.fixture()
        with patch('gdn_records.restore_prefix'):
            for epoch in range(3):
                block.commit(epoch, synchronize=True)
                self.assertTrue(block.replay_ready)
                operation = Mock(return_value=None)
                block.replay(operation)
                operation.assert_called_once_with()
                self.assertIsNone(block.selected_prefix)
                self.assertEqual(block.replay_epoch, epoch + 1)
                with self.assertRaises(ValueError):
                    block.replay(operation)
        self.assertEqual(block.operations.synchronize_device.call_count, 6)

    def test_unsynchronized_or_failed_commit_cannot_replay(self):
        for failure in ('unsynchronized', 'publication', 'synchronization'):
            block = self.fixture()
            if failure == 'synchronization':
                block.operations.synchronize_device.side_effect = RuntimeError('device')
            with patch('gdn_records.restore_prefix', side_effect=RuntimeError('device') if failure == 'publication' else None):
                if failure == 'unsynchronized':
                    block.commit(2)
                else:
                    with self.assertRaises(RuntimeError):
                        block.commit(2, synchronize=True)
            operation = Mock()
            with self.assertRaises(ValueError):
                block.replay(operation)
            operation.assert_not_called()

    def test_failed_or_rebound_replay_is_poisoned(self):
        for failure in ('binding', 'callback', 'synchronization', 'return-value', 'reentrancy'):
            block = self.fixture()
            with patch('gdn_records.restore_prefix'):
                block.commit(2, synchronize=True)
            operation = Mock(return_value=None)
            if failure == 'binding':
                block.records[-1][0].gdn.B = 1
            elif failure == 'callback':
                operation.side_effect = RuntimeError('device')
            elif failure == 'synchronization':
                block.operations.synchronize_device.side_effect = RuntimeError('device')
            elif failure == 'return-value':
                operation.return_value = object()
            else:
                operation.side_effect = lambda: block.replay(lambda: None)
            with self.assertRaises((RuntimeError, ValueError)):
                block.replay(operation)
            with self.assertRaises(ValueError):
                block.replay(lambda: None)
            with self.assertRaises(ValueError):
                block.commit(2)
            self.assertEqual(block.replay_epoch, 0)

    def test_all_layers_commit_once_and_keep_records_until_trace_release(self):
        for prefix in range(5):
            block = self.fixture()
            with patch('gdn_records.restore_prefix') as restore, patch('gdn_records.release_owned') as release:
                block.commit(prefix)
                self.assertEqual(restore.call_count, 48)
                self.assertTrue(all(call.args[-1] == prefix for call in restore.call_args_list))
                for state, result, checkpoint in block.records:
                    state.active.restore.assert_called_once_with(checkpoint)
                release.assert_not_called()
                with self.assertRaises(ValueError):
                    block.commit(prefix)
                block.close()
                block.close()
                release.assert_called_once()

    def test_incomplete_invalid_or_rebound_blocks_do_not_write(self):
        for invalid in ('incomplete', 'prefix', 'binding', 'unstable'):
            block = self.fixture(47 if invalid == 'incomplete' else 48)
            if invalid == 'binding':
                block.records[-1][0].gdn.B = 1
            if invalid == 'unstable':
                block.records[-1][0].gdn._stable_state = False
            with patch('gdn_records.restore_prefix') as restore:
                with self.assertRaises(ValueError):
                    block.commit(True if invalid == 'prefix' else 2)
            restore.assert_not_called()
            self.assertIsNone(block.selected_prefix)

    def test_partial_device_failure_cannot_be_retried_as_a_fresh_commit(self):
        block = self.fixture()
        with patch('gdn_records.restore_prefix', side_effect=RuntimeError('device')):
            with self.assertRaises(RuntimeError):
                block.commit(2)
        with self.assertRaises(ValueError):
            block.commit(2)

    def test_dma_publishes_all_records_in_one_launch(self):
        block = self.fixture()
        mesh = object()
        for state, result, checkpoint in block.records:
            state.gdn.mesh = mesh
            result['packed_conv_states'] = [object() for slot in range(4)]
        with patch('gdn_commit_dma.publish') as publish, patch('gdn_records.restore_prefix') as restore:
            block.commit(2, dma=True)
            publish.assert_called_once()
            self.assertIs(publish.call_args.args[0], mesh)
            self.assertEqual(len(publish.call_args.args[1]), 48)
            self.assertTrue(all(len(layer) == 20 for layer in publish.call_args.args[1]))
            self.assertEqual(publish.call_args.args[2], 2)
            restore.assert_not_called()
            for state, result, checkpoint in block.records:
                state.active.restore.assert_not_called()
        with self.assertRaises(ValueError):
            block.commit(2, dma=True)

    def test_duplicate_or_unpacked_record_rejected(self):
        block = self.fixture(1)
        state, result, checkpoint = block.records[0]
        with self.assertRaises(ValueError):
            block.append(state, result, checkpoint)
        result['packed_checkpoints'] = False
        with self.assertRaises(ValueError):
            block.append(state, result, checkpoint)

    def test_bound_publication_still_requires_native_binding_and_single_decision(self):
        block = self.fixture()
        publication = Mock()
        block.commit(2, dma=True, publication=publication)
        publication.assert_called_once_with(2)
        with self.assertRaises(ValueError):
            block.commit(2, dma=True, publication=publication)
        for invalid in ('binding', 'mode'):
            block = self.fixture()
            if invalid == 'binding':
                block.records[-1][0].gdn.B = 1
            publication = Mock()
            with self.assertRaises(ValueError):
                block.commit(2, dma=invalid != 'mode', publication=publication)
            publication.assert_not_called()
