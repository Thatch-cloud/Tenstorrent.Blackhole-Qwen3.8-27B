import unittest

from dspark_splitk_merge_exp import transform
from dspark_splitk_unfused_correction import REPLACEMENT


class MergeExpTests(unittest.TestCase):
    def test_full_scale_and_unchanged_input_ownership(self):
        result = transform('TREE REDUCTION LOGIC\n' + REPLACEMENT + '\nvoid kernel_main() {}')
        self.assertEqual(result.count('qwen_splitk_merge_exp<scale_fp32>'), 2)
        self.assertIn('mul_unary_tile(0, scale_fp32);', result)
        self.assertIn('exp_tile_init<false>();', result)
        self.assertIn('exp_tile<false>(0);', result)
        self.assertNotIn('scale_fp32 >> 16', result)
        self.assertNotIn('pop_front', result)
        self.assertIn('tile < tiles', result)

    def test_rejects_missing_and_duplicate_calls(self):
        for source in ('void kernel_main() {}', REPLACEMENT * 2 + 'void kernel_main() {}'):
            with self.assertRaises(ValueError):
                transform(source)


if __name__ == '__main__':
    unittest.main()
