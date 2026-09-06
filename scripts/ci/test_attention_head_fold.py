import unittest

import torch

from attention_head_fold import causal_mask, chunk_groups, fold_query, unfold_output


class HeadFoldTests(unittest.TestCase):
    def test_chunk_boundaries_split_queries_before_core_assignment_changes(self):
        self.assertEqual(chunk_groups(4095, 32, max_group_rows=32), [dict(offset=0, rows=1, signature=(256, 4096)),
                                                dict(offset=1, rows=31, signature=(256, 4352))])
        self.assertEqual(chunk_groups(16384, 32, max_group_rows=32), [dict(offset=0, rows=32, signature=(256, 16640))])
        self.assertEqual(chunk_groups(31, 2), [dict(offset=0, rows=1, signature=(32, 32)),
                                             dict(offset=1, rows=1, signature=(64, 64))])
        self.assertEqual(chunk_groups(4095, 2, max_chunk_tiles=4)[1]['signature'], (128, 4224))

    def test_bounded_groups_cover_every_token_once(self):
        groups = chunk_groups(4095, 32)
        self.assertEqual([group['rows'] for group in groups], [1, 4, 4, 4, 4, 4, 4, 4, 3])
        self.assertEqual([offset for group in groups for offset in range(group['offset'], group['offset'] + group['rows'])],
                         list(range(32)))

    def test_roundtrip_and_gqa_mapping(self):
        for rows in range(1, 33):
            query = torch.arange(rows * 12 * 256).reshape(1, rows, 12, 256)
            folded = fold_query(query)
            self.assertTrue(torch.equal(unfold_output(folded, rows), query))
            for head in range(12):
                for token in range(rows):
                    mapped = (head // 6) * rows * 6 + token * 6 + head % 6
                    self.assertTrue(torch.equal(folded[0, 0, mapped], query[0, token, head]))

    def test_each_virtual_head_masks_future_tokens(self):
        for rows in (1, 2, 4, 8, 16, 32):
            mask = causal_mask(rows, 63, 128)
            for head in range(rows * 12):
                position = 63 + (head % (rows * 6)) // 6
                self.assertTrue(torch.all(mask[0, 0, head, :position + 1] == 0))
                self.assertTrue(torch.all(torch.isneginf(mask[0, 0, head, position + 1:])))

    def test_invalid_geometry_is_rejected(self):
        for geometry in ((33, 0, 64), (2, -1, 64), (2, 63, 64), (True, 0, 64)):
            with self.assertRaises(ValueError):
                causal_mask(*geometry)
