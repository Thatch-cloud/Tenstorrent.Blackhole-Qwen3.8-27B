import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from attention_replay import ReplayAttentionReader


class ReplayReaderTests(unittest.TestCase):
    def fixture(self, rows=16, *, max_group_rows=4):
        operations = SimpleNamespace(int32='int32', SDPAProgramConfig=Mock())
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        upload = Mock(side_effect=lambda value, dtype=None: value)
        with patch('attention_replay.prepare', return_value='program'):
            reader = ReplayAttentionReader(operations, mesh, rows, 4352,
                torch.arange(68).reshape(1, 68), upload, max_group_rows=max_group_rows)
        return reader, upload

    def test_wide_replay_requires_explicit_compact_scratch_and_preserves_defaults(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'compact native scratch'):
                self.fixture(max_group_rows=8)
        for width in (True, 8.0, 16, 3):
            with self.assertRaisesRegex(ValueError, 'explicitly four or eight'):
                self.fixture(max_group_rows=width)
        with patch.dict('os.environ', {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}):
            wide, _ = self.fixture(rows=32, max_group_rows=8)
            control, _ = self.fixture(rows=32)
        self.assertEqual([len(entry[0]) for entry in wide.metadata], [3, 1])
        self.assertEqual([len(entry[0]) for entry in control.metadata], [3, 3, 2])
        self.assertTrue(all(group['rows'] == 8 for entry in wide.metadata for group in entry[0]))
        self.assertEqual([tuple(entry[2].shape) for entry in wide.metadata], [(3, 1, 96, 4352), (1, 1, 96, 4352)])

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

    def test_shared_masks_refresh_once_for_each_sixteen_layer_forward(self):
        reader, _ = self.fixture()
        query = SimpleNamespace(shape=(1, 16, 12, 256))
        keys, values, result = object(), object(), object()
        events = []
        with patch('attention_replay.addresses', side_effect=lambda operations, value: (id(value), id(value))), patch(
                'attention_replay.refresh_mask', side_effect=lambda *args: events.append('mask')), patch(
                'attention_replay.execute', side_effect=lambda *args, **kwargs: events.append('attention') or result), patch(
                'attention_replay.release_owned'):
            for forward in range(2):
                with reader.shared_masks(16):
                    for layer in range(16):
                        self.assertIs(reader(query, keys, values, scale=0.0625, memory_config='L1'), result)
        self.assertEqual(events, (['mask'] * 2 + ['attention'] * 16) * 2)
        self.assertEqual((reader.calls, reader.refresh_calls), (32, 4))
        self.assertIsNone(reader.mask_scope)
        self.assertFalse(reader.failed)

    def test_incomplete_or_failed_shared_forward_poisons_reader(self):
        for expected in (1, 16):
            reader, _ = self.fixture()
            with patch('attention_replay.refresh_mask'):
                with self.assertRaisesRegex(AssertionError, 'exact attention call budget'):
                    with reader.shared_masks(expected):
                        pass
            self.assertTrue(reader.failed)
            self.assertIsNone(reader.mask_scope)
        reader, _ = self.fixture()
        with patch('attention_replay.refresh_mask', side_effect=RuntimeError('mask failed')):
            with self.assertRaisesRegex(RuntimeError, 'mask failed'):
                with reader.shared_masks(16):
                    self.fail('Failed mask refresh must not enter the model forward')
        self.assertTrue(reader.failed)
        self.assertIsNone(reader.mask_scope)

    def test_shared_forward_cannot_stage_close_or_nest(self):
        for operation in ('stage', 'close', 'nest'):
            reader, _ = self.fixture()
            with patch('attention_replay.refresh_mask'), self.assertRaises(RuntimeError):
                with reader.shared_masks(16):
                    if operation == 'stage':
                        reader.stage(4103)
                    elif operation == 'close':
                        reader.close()
                    else:
                        with reader.shared_masks(1):
                            self.fail('Nested mask scope must not enter')
            self.assertTrue(reader.failed)
            self.assertFalse(reader.closed)

    def test_shared_forward_rejects_excess_calls_before_attention(self):
        reader, _ = self.fixture()
        query = SimpleNamespace(shape=(1, 16, 12, 256))
        with patch('attention_replay.addresses', return_value=(1, 2)), patch(
                'attention_replay.refresh_mask'), patch('attention_replay.execute') as execute, patch(
                'attention_replay.release_owned'):
            with self.assertRaisesRegex(AssertionError, 'exceeded'):
                with reader.shared_masks(1):
                    reader(query, object(), object(), scale=0.0625, memory_config='L1')
                    reader(query, object(), object(), scale=0.0625, memory_config='L1')
            execute.assert_called_once()
        self.assertTrue(reader.failed)

    def test_shared_forward_requires_explicit_bounded_budget(self):
        reader, _ = self.fixture()
        for expected in (0, 17, True, 1.0, None):
            with self.assertRaises(ValueError):
                with reader.shared_masks(expected):
                    self.fail('Invalid scope must not enter')
        self.assertFalse(reader.failed)
