import unittest

from dspark_splitk_denominator import MULTIPLY, transform


class DenominatorTests(unittest.TestCase):
    def test_configures_multiply_and_exact_reciprocal_copy(self):
        source = MULTIPLY + '\n                copy_tile_to_dst_init_short(cb_prev_sum);\n                    copy_tile(cb_prev_sum, tile, 0);'
        result = transform(source)
        self.assertIn('reconfig_data_format(cb_prev_sum, cb_exp_max_diff);', result)
        self.assertIn('pack_reconfig_data_format(cb_prev_sum);', result)
        self.assertIn('qwen_splitk_copy_fp32(cb_prev_sum, tile, 0);', result)
        self.assertNotIn('copy_tile_to_dst_init_short', result)
        with self.assertRaises(ValueError):
            transform(source + source)


if __name__ == '__main__':
    unittest.main()
