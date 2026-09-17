import unittest

from dspark_splitk_output_rounding import transform


class OutputRoundingTests(unittest.TestCase):
    def test_final_copy_rounds_and_preserves_ownership(self):
        source = '#include "api/compute/eltwise_unary/recip.h"\n' + (
            '                move_block<true>(cb_out_accumulate_im, cb_out_final, out_chunk_tiles);')
        result = transform(source)
        self.assertIn('typecast_tile_init<', result)
        self.assertIn('typecast_tile<', result)
        self.assertEqual(result.count('pop_front(out_chunk_tiles)'), 1)
        self.assertEqual(result.count('push_back(1)'), 1)

    def test_rejects_source_drift(self):
        with self.assertRaisesRegex(ValueError, 'final-output'):
            transform('')


if __name__ == '__main__':
    unittest.main()
