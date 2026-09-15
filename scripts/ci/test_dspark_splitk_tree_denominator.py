import unittest

from dspark_splitk_tree_denominator import TREE_ARITHMETIC, transform


class TreeDenominatorTests(unittest.TestCase):
    def source(self):
        return '\n'.join((TREE_ARITHMETIC, 'void kernel_main() {',
            'move_block<true>(cb_l_in, cb_prev_sum_2, Sq_chunk_t);',
            'move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);',
            'move_block<true>(cb_prev_sum, cb_out_l, Sq_chunk_t);',
            'mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);'))

    def test_single_pack_preserves_scales_for_numerator(self):
        result = transform(self.source())
        self.assertNotIn(TREE_ARITHMETIC, result)
        self.assertEqual(result.count('pack_tile(0, output);'), 1)
        for name in ('left', 'right'):
            self.assertEqual(result.count('CircularBuffer(' + name + ').pop_front(tiles);'), 1)
            self.assertNotIn('CircularBuffer(' + name + '_scale).pop_front', result)
        self.assertIn('mul_binary_tile(1, 2, 1);', result)
        self.assertEqual(result.count('qwen_splitk_denominator_move('), 3)
        self.assertIn('mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);', result)

    def test_changed_or_repeated_source_rejected(self):
        for source in (self.source() + self.source(), self.source().replace('cb_l_in', 'changed'), ''):
            with self.assertRaises(ValueError):
                transform(source)


if __name__ == '__main__':
    unittest.main()
