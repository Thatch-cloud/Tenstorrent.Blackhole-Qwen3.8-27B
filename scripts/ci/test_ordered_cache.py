import unittest
from unittest.mock import patch

from ordered_cache import (WIDE_PAGE_WIDTHS, load_kernels, page_width_admitted, replace_once,
                           validate_shapes)


class OrderedCacheTests(unittest.TestCase):
    def test_supported_geometry(self):
        for rows in (1, 2, 4, 8, 16, 32):
            self.assertEqual(validate_shapes((8, 2, 64, 256), (1, rows, 32, 256), (rows,), (rows, 4)), rows)
            self.assertEqual(validate_shapes((8200, 2, 64, 256), (1, rows, 32, 256), (rows,), (rows, 1024)), rows)

    def test_rejects_geometry_before_dispatch(self):
        cases = [((8, 1, 64, 256), (1, 2, 32, 256), (2,), (2, 4)),
                 ((8, 2, 64, 256), (1, 3, 32, 256), (3,), (3, 4)),
                 ((8, 2, 64, 256), (1, 2, 2, 256), (2,), (2, 4)),
                 ((8, 2, 64, 256), (1, 2, 32, 256), (1,), (2, 4)),
                 ((8, 2, 64, 256), (1, 2, 32, 256), (2,), (1, 4)),
                 ((8, 2, 64, 256), (1, 2, 32, 256), (2,), (2, 16)),
                 ((8, 2, 64, 256), (1, 2, 32, 256), (2,), (2, 1025))]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_shapes(*arguments)

    def test_the_131k_width_is_admitted(self):
        """Run 35787268942 prefilled 131,072 tokens and died on the first decode write at
        2,052 pages. The kernels have no 1,024 limit; 2,052 is admitted as a reviewed width."""
        for rows in (1, 2, 4, 8, 16, 32):
            self.assertEqual(
                validate_shapes((8212, 2, 64, 256), (1, rows, 32, 256), (rows,), (rows, 2052)), rows)

    def test_1025_is_still_refused_by_default(self):
        """The original envelope is unchanged: this is a scoped admission, not a lifted cap."""
        with self.assertRaises(ValueError):
            validate_shapes((8212, 2, 64, 256), (1, 2, 32, 256), (2,), (2, 1025))

    def test_only_reviewed_wide_widths_are_admitted(self):
        """4,096 needs its own hardware check; 4,100 and 4,104 exceed the model's 262,144
        max_position_embeddings and are never valid; nearby widths are not reviewed."""
        for width in (1028, 2048, 2051, 2053, 2056, 4096, 4100, 4104):
            with self.subTest(width=width), self.assertRaises(ValueError):
                validate_shapes((8212, 2, 64, 256), (1, 2, 32, 256), (2,), (2, width))

    def test_a_wide_table_cannot_exceed_the_cache(self):
        """The kernel does not check that page-table entries exist; this shape check is the
        only guard that the cache holds at least as many blocks as the table addresses."""
        with self.assertRaises(ValueError):
            validate_shapes((2051, 2, 64, 256), (1, 1, 32, 256), (1,), (1, 2052))
        self.assertEqual(validate_shapes((2052, 2, 64, 256), (1, 1, 32, 256), (1,), (1, 2052)), 1)

    def test_page_width_admitted(self):
        self.assertEqual(WIDE_PAGE_WIDTHS, frozenset({2052}))
        for width in (1, 4, 516, 1024, 2052):
            self.assertTrue(page_width_admitted(width), width)
        for width in (0, -1, 1025, 2048, 4096, 4104, 2052.0, '2052', None):
            self.assertFalse(page_width_admitted(width), width)

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
