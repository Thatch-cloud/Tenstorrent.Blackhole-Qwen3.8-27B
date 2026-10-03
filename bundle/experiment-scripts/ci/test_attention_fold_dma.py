import unittest

import torch

from attention_fold_dma import source_row
from attention_head_fold import fold_query


class FoldDMATests(unittest.TestCase):
    def test_every_group_row_matches_native_head_order(self):
        for rows in range(1, 9):
            query = torch.arange(rows * 12).reshape(1, rows, 12, 1).expand(1, rows, 12, 256)
            expected = fold_query(query)[0, 0, :, 0].tolist()
            mapping = [source_row(rows, index) for index in range(rows * 12)]
            self.assertEqual(mapping, expected)
            inverse = [source_row(rows, index, inverse=True) for index in range(rows * 12)]
            self.assertEqual([mapping[index] for index in inverse], list(range(rows * 12)))

    def test_invalid_rows_rejected(self):
        for rows in (0, 9, True, 4.0):
            with self.assertRaises(ValueError):
                source_row(rows, 0)
