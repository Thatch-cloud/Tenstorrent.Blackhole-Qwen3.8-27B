import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from attention_replay import ReplayAttentionReader


class ReplayReaderTests(unittest.TestCase):
    def fixture(self, rows=16):
        operations = SimpleNamespace(int32='int32', SDPAProgramConfig=Mock())
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        upload = Mock(side_effect=lambda value, dtype=None: value)
        with patch('attention_replay.prepare', return_value='program'):
            reader = ReplayAttentionReader(operations, mesh, rows, 4352,
                torch.arange(68).reshape(1, 68), upload)
        return reader, upload

    def test_prepared_family_has_zero_masks_and_bounded_parallel_pages(self):
        reader, upload = self.fixture()
        self.assertEqual([len(entry[0]) for entry in reader.metadata], [3, 1])
        self.assertEqual(len(reader.programs), 2)
        self.assertEqual(upload.call_count, 5)
        for bundle, pages, mask, config in reader.metadata:
            self.assertEqual(tuple(pages.shape), (len(bundle), 68))
            self.assertTrue(torch.all(mask == 0))

    def test_out_of_family_stage_rejected_before_any_copy(self):
        reader, _ = self.fixture()
        for start in (4095, 4337, -1):
            with self.assertRaises(ValueError):
                reader.stage(start)
        self.assertEqual(reader.start, 4096)

    def test_close_is_idempotent_and_rejects_replay(self):
        reader, _ = self.fixture()
        with patch('attention_replay.release_owned') as release:
            reader.close()
            reader.close()
            release.assert_called_once()
        self.assertTrue(reader.closed)
        with self.assertRaises(RuntimeError):
            reader.validate(4096)

    def test_partial_staging_failure_poisons_future_replay(self):
        reader, _ = self.fixture()
        reader.operations.ROW_MAJOR_LAYOUT = 'row'
        reader.operations.ReplicateTensorToMesh = Mock()
        reader.operations.from_torch = Mock()
        reader.operations.copy_host_to_device_tensor = Mock(side_effect=RuntimeError('copy failed'))
        with patch('attention_replay.addresses', return_value=(1, 2)):
            with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                reader.stage(4103)
        self.assertTrue(reader.failed)
        with self.assertRaisesRegex(RuntimeError, 'poisoned'):
            reader.validate(4096)

    def test_small_bucket_is_not_silently_reinterpreted(self):
        for rows in (1, 2, 4, True):
            with self.assertRaises(ValueError):
                self.fixture(rows)

    def test_call_refreshes_before_attention_and_preserves_borrowed_buffers(self):
        reader, _ = self.fixture()
        query = SimpleNamespace(shape=(1, 16, 12, 256))
        keys, values, scratch, result = object(), object(), object(), object()
        events = []

        def attention(mesh, operations, query, keys, values, metadata, owned, **kwargs):
            events.append('attention')
            owned.extend((query, keys, values, scratch, result))
            return result

        with patch('attention_replay.addresses', side_effect=lambda operations, value: (id(value), id(value))), patch(
                'attention_replay.refresh_mask', side_effect=lambda *args: events.append('mask')), patch(
                'attention_replay.execute', side_effect=attention), patch('attention_replay.release_owned') as release:
            self.assertIs(reader(query, keys, values, scale=0.0625, memory_config='L1'), result)
            release.assert_called_once_with(reader.operations, [scratch])
        self.assertEqual(events, ['mask', 'mask', 'attention'])
        self.assertEqual((reader.calls, reader.refresh_calls), (1, 2))
