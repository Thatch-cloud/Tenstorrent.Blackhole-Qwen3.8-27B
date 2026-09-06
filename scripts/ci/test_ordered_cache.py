import unittest
from unittest.mock import patch

from ordered_cache import load_kernels, replace_once, validate_shapes


class OrderedCacheTests(unittest.TestCase):
    def test_supported_geometry(self):
        for rows in (1, 2, 4, 8, 16):
            self.assertEqual(validate_shapes((8, 2, 64, 256), (1, rows, 32, 256), (rows,), (rows, 4)), rows)

    def test_rejects_geometry_before_dispatch(self):
        cases = [((8, 1, 64, 256), (1, 2, 32, 256), (2,), (2, 4)),
                 ((8, 2, 64, 256), (1, 3, 32, 256), (3,), (3, 4)),
                 ((8, 2, 64, 256), (1, 2, 2, 256), (2,), (2, 4)),
                 ((8, 2, 64, 256), (1, 2, 32, 256), (1,), (2, 4)),
                 ((8, 2, 64, 256), (1, 2, 32, 256), (2,), (1, 4)),
                 ((8, 2, 64, 256), (1, 2, 32, 256), (2,), (2, 513))]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_shapes(*arguments)

    def test_source_transform_requires_unique_anchor(self):
        self.assertEqual(replace_once('before anchor after', 'anchor', 'new'), 'before new after')
        for source in ('missing', 'anchor anchor'):
            with self.assertRaises(ValueError):
                replace_once(source, 'anchor', 'new')

    def test_rejects_unpinned_native_source(self):
        with patch('ordered_cache.Path.read_bytes', return_value=b'changed'), self.assertRaisesRegex(ValueError, 'Unaudited'):
            load_kernels('/unused')


if __name__ == '__main__':
    unittest.main()
