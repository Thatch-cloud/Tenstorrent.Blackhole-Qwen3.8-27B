import unittest

from dspark_splitk_denominator_recurrence import ADD, MOVE, MULTIPLY, transform


class RecurrenceTests(unittest.TestCase):
    def test_scoped_replacement_and_consumption(self):
        source = '\n'.join((MULTIPLY, 'consume_correction_in_numerator();', ADD, MOVE, 'void kernel_main() {'))
        result = transform(source)
        self.assertNotIn(MULTIPLY, result)
        self.assertNotIn(ADD, result)
        self.assertNotIn(MOVE, result)
        self.assertEqual(result.count('CircularBuffer(previous).pop_front(tiles);'), 1)
        self.assertNotIn('CircularBuffer(correction).pop_front', result)
        self.assertIn('mul_binary_tile(0, 1, 0);', result)
        self.assertIn('qwen_splitk_copy_fp32(source, tile, 0);', result)
        self.assertLess(result.index('qwen_splitk_denominator_recurrence(cb_cur_sum'),
            result.index('consume_correction_in_numerator();'))
        with self.assertRaises(ValueError):
            transform(source + source)


if __name__ == '__main__':
    unittest.main()
