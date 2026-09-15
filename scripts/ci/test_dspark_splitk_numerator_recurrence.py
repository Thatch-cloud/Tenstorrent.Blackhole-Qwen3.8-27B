import unittest

from dspark_splitk_numerator_recurrence import ADD, MULTIPLY, transform


class NumeratorRecurrenceTests(unittest.TestCase):
    def test_single_pack_and_exact_input_lifetimes(self):
        source = '\n'.join((MULTIPLY, ADD, '#include "api/compute/eltwise_binary_sfpu.h"', 'void kernel_main() {'))
        result = transform(source)
        self.assertNotIn(MULTIPLY, result)
        self.assertNotIn(ADD, result)
        self.assertEqual(result.count('pack_tile(0, accumulator);'), 1)
        self.assertEqual(result.count('CircularBuffer(partial).pop_front(tiles);'), 1)
        self.assertEqual(result.count('CircularBuffer(correction).pop_front(rows);'), 1)
        self.assertIn('qwen_splitk_copy_fp32(partial, row * columns + column, 1);', result)
        self.assertLess(result.index('sfpu_mul_bcast_col(0, 1);'), result.index('add_binary_tile(0, 1, 0);'))
        with self.assertRaises(ValueError):
            transform(source + source)


if __name__ == '__main__':
    unittest.main()
