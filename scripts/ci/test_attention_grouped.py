import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from attention_grouped import GroupedAttentionReader


class GroupedReaderTests(unittest.TestCase):
    def test_dma_selection_must_be_explicit_boolean(self):
        with self.assertRaises(ValueError):
            GroupedAttentionReader(None, None, 0, 8, None, None, None, None, dma_layout='yes')

    def fixture(self, rows, capacity=16640, parallel=False, max_group_rows=4):
        operations = SimpleNamespace(int32='int32', SDPAProgramConfig=Mock(),
            transformer=SimpleNamespace(paged_scaled_dot_product_attention_decode=Mock()))
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        upload = Mock(side_effect=lambda value, dtype=None: value)
        reader = GroupedAttentionReader(operations, mesh, 16383, rows,
            torch.arange(capacity // 64).reshape(1, -1), list(range(rows)), 'pages', upload,
            dma_layout=parallel, parallel=parallel, max_group_rows=max_group_rows)
        return reader, operations, upload

    def test_small_width_uses_native_without_masks(self):
        for rows in (1, 2, 4):
            reader, operations, upload = self.fixture(rows)
            upload.assert_not_called()
            self.assertFalse(reader.metadata)
        reader, operations, _ = self.fixture(1)
        query = SimpleNamespace(shape=(1, 1, 12, 256))
        reader(query, None, None, page_table_tensor=None, cur_pos_tensor=None)
        self.assertIs(operations.transformer.paged_scaled_dot_product_attention_decode.call_args.args[0], query)

    def test_long_boundary_has_bounded_views_and_groups(self):
        reader, operations, upload = self.fixture(32)
        self.assertEqual([entry[0]['rows'] for entry in reader.metadata], [1, 4, 4, 4, 4, 4, 4, 4, 3])
        self.assertEqual([entry[1].shape[1] for entry in reader.metadata], [256] + [260] * 8)
        self.assertEqual(upload.call_count, 18)
        self.assertEqual(len(reader.owned), 18)

    def test_insufficient_cache_rejected(self):
        with self.assertRaises(ValueError):
            self.fixture(8, capacity=16384)

    def test_parallel_requires_dma_and_explicit_boolean(self):
        for parallel in ('yes', True):
            with self.assertRaises(ValueError):
                GroupedAttentionReader(None, None, 0, 8, None, None, None, None, parallel=parallel)

    def test_parallel_boundary_metadata_preserves_each_query_mask(self):
        reader, _, upload = self.fixture(32, parallel=True)
        self.assertEqual([len(entry[0]) for entry in reader.metadata], [1, 3, 3, 1, 1])
        self.assertEqual(upload.call_count, 10)
        for bundle, pages, mask, _ in reader.metadata:
            self.assertEqual(pages.shape[0], len(bundle))
            self.assertEqual(mask.shape[0], len(bundle))
            for index, group in enumerate(bundle):
                for row in range(group['rows']):
                    position = 16383 + group['offset'] + row
                    selected = mask[index, 0, row * 12 // 2]
                    self.assertTrue(torch.all(selected[:position + 1] == 0))
                    self.assertTrue(torch.all(torch.isneginf(selected[position + 1:])))

    def test_changed_geometry_rejected(self):
        reader, _, _ = self.fixture(8)
        with self.assertRaises(ValueError):
            reader(SimpleNamespace(shape=(1, 16, 12, 256)), None, None, page_table_tensor=None, cur_pos_tensor=None)

    def test_eight_row_parallel_boundary_preserves_masks_and_offsets(self):
        reader, _, upload = self.fixture(32, parallel=True, max_group_rows=8)
        self.assertEqual([len(entry[0]) for entry in reader.metadata], [1, 3, 1])
        self.assertEqual([group['rows'] for entry in reader.metadata for group in entry[0]], [1, 8, 8, 8, 7])
        self.assertEqual(upload.call_count, 6)
        for bundle, pages, mask, _ in reader.metadata:
            for index, group in enumerate(bundle):
                for head in range(group['rows'] * 12):
                    position = 16383 + group['offset'] + (head % (group['rows'] * 6)) // 6
                    self.assertTrue(torch.all(mask[index, 0, head, :position + 1] == 0))
                    self.assertTrue(torch.all(torch.isneginf(mask[index, 0, head, position + 1:])))

    def test_eight_row_width_never_silently_enables_parallel(self):
        for width in (8, 16, True, 4.0):
            with self.assertRaises(ValueError):
                self.fixture(8, max_group_rows=width)

    def test_eight_row_policy_keeps_small_buckets_native(self):
        for rows in (1, 2, 4):
            reader, _, upload = self.fixture(rows, parallel=True, max_group_rows=8)
            upload.assert_not_called()
            self.assertFalse(reader.metadata)
