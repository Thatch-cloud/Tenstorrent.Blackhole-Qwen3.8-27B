import unittest

from gdn_conv_prefix_copy import validate_prefix


class PrefixCopyTests(unittest.TestCase):
    def test_every_supported_prefix(self):
        for rows in (1, 2, 4, 8, 16):
            for prefix in range(1, rows + 1):
                self.assertEqual(validate_prefix([(1, rows, 5120)] * 4, [(1, 1, 5120)] * 4, prefix), rows)

    def test_rejects_invalid_shapes_and_prefixes(self):
        for prefix in (0, -1, 5, True, 1.0):
            with self.assertRaises(ValueError):
                validate_prefix([(1, 4, 5120)] * 4, [(1, 1, 5120)] * 4, prefix)
        for source in ([], [(1, 4, 5120)] * 3, [(1, 3, 5120)] * 4, [(1, 4, 5120)] * 3 + [(1, 2, 5120)]):
            with self.assertRaises(ValueError):
                validate_prefix(source, [(1, 1, 5120)] * 4, 1)
        for destination in ([], [(1, 1, 5120)] * 3, [(1, 8, 5120)] * 4):
            with self.assertRaises(ValueError):
                validate_prefix([(1, 4, 5120)] * 4, destination, 1)
