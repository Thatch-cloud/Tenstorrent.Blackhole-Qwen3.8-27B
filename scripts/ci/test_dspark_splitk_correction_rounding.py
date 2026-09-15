import unittest

from dspark_splitk_correction_rounding import FORMAT, transform
from dspark_splitk_merge_exp import HELPER


class CorrectionRoundingTests(unittest.TestCase):
    def test_rounds_correction_only_before_pack(self):
        source = 'untouched prefix\n' + HELPER + '\nuntouched suffix'
        result = transform(source)
        self.assertTrue(result.startswith('untouched prefix'))
        self.assertTrue(result.endswith('untouched suffix'))
        self.assertEqual(result.count('typecast_tile' + FORMAT + '(0);'), 1)
        self.assertLess(result.index('exp_tile<false>(0);'), result.index('typecast_tile' + FORMAT + '(0);'))
        self.assertLess(result.index('typecast_tile' + FORMAT + '(0);'), result.index('tile_regs_commit();'))
        self.assertNotIn('pop_front', result)
        with self.assertRaises(ValueError):
            transform(source + source)


if __name__ == '__main__':
    unittest.main()
