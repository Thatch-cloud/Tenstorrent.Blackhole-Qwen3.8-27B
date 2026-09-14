import unittest

from dspark_splitk_precise_exp import BEFORE, transform


class PreciseExpTests(unittest.TestCase):
    def test_preserves_scale_and_full_tile_traversal(self):
        result = transform('#include "api/compute/eltwise_unary/recip.h"\n' + BEFORE)
        self.assertIn('mul_unary_tile(0, scale_fp32);', result)
        self.assertIn('exp_tile_init<false>();', result)
        self.assertIn('exp_tile<false>(0);', result)
        self.assertIn('typecast_tile<static_cast<uint32_t>(DataFormat::Float32)', result)
        self.assertLess(result.index('exp_tile<false>(0);'), result.index('typecast_tile_init<'))
        self.assertIn('row < Sq_chunk_t', result)
        self.assertIn('column < Sk_chunk_t_dynamic', result)
        self.assertEqual(result.count('pop_front(1)'), 1)
        self.assertEqual(result.count('push_back(1)'), 1)

    def test_rejects_source_drift(self):
        with self.assertRaisesRegex(ValueError, 'anchors'):
            transform(BEFORE)


if __name__ == '__main__':
    unittest.main()
