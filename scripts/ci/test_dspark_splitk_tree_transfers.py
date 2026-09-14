import unittest

from dspark_splitk_fp32_factory import TREE_BUFFERS, promote_tree_transfers


class TreeTransfersTests(unittest.TestCase):
    def test_uniform_sender_receiver_and_scratch_formats(self):
        self.assertEqual(set(TREE_BUFFERS), {6, 7, 16, 17, 18, 19, 21})
        source = '\n'.join(f'add_cb(CBIndex::c_{index}, {count} * stats_tile_size, stats_df, stats_tile_size, &stats_tile);'
            for index, count in TREE_BUFFERS.items())
        result = promote_tree_transfers(source)
        self.assertEqual(result.count('if (qwen_splitk_fp32)'), 7)
        self.assertEqual(result.count('} else {'), 7)
        for index, count in TREE_BUFFERS.items():
            self.assertIn(f'add_cb(CBIndex::c_{index}, {count} * im_tile_size, im_df, im_tile_size, &im_tile);', result)
            self.assertIn(f'add_cb(CBIndex::c_{index}, {count} * stats_tile_size, stats_df, stats_tile_size, &stats_tile);', result)
        with self.assertRaises(ValueError):
            promote_tree_transfers(source.replace('CBIndex::c_18', 'CBIndex::c_99'))


if __name__ == '__main__':
    unittest.main()
