import unittest

from draft_kv_slide import row_source
from draft_kv_slide_direct import segments


class DirectSlideTests(unittest.TestCase):
    def test_all_accepted_lengths_and_face_boundaries(self):
        for history in (1, 15, 16, 17, 31, 32, 33, 2016, 2031, 2047, 2048):
            for prefix in range(1, 33):
                for tile in range(64):
                    rows = []
                    for kind, source, destination, count in segments(history, prefix, tile):
                        self.assertEqual(destination, len(rows))
                        if kind != 'zero':
                            self.assertLessEqual(source % 16 + count, 16)
                            self.assertLessEqual(destination % 16 + count, 16)
                        rows.extend((kind, source + offset if kind != 'zero' else 0) for offset in range(count))
                    self.assertEqual(rows, [row_source(history, prefix, tile * 32 + row) for row in range(32)])

    def test_invalid_tile(self):
        for tile in (-1, 64, True):
            with self.assertRaises(ValueError):
                segments(2048, 16, tile)
