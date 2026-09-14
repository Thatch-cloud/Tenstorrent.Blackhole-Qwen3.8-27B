import unittest

from dspark_splitk_accumulator_add import BEFORE, transform


class AccumulatorTests(unittest.TestCase):
    def test_exact_local_replacement_and_ownership(self):
        source = 'void kernel_main() {' + BEFORE + '}'
        result = transform(source)
        self.assertNotIn(BEFORE, result)
        self.assertIn('qwen_splitk_copy_fp32(partial, tile, 1);', result)
        self.assertIn('add_binary_tile(0, 1, 0);', result)
        self.assertEqual(result.count('CircularBuffer(partial).pop_front(tiles);'), 1)
        self.assertEqual(result.count('CircularBuffer(accumulator).push_back(1);'), 1)
        with self.assertRaises(ValueError):
            transform(source + source)


if __name__ == '__main__':
    unittest.main()
