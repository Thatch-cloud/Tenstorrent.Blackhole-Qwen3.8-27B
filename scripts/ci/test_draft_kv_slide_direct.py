import unittest

from draft_kv_slide import row_source
from draft_kv_slide_direct import segments, transfers


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

    def test_every_face_segment_dma_alignment_and_scratch_bounds(self):
        for base in (0, 16, 32, 48):
            for source in range(32):
                for destination in range(32):
                    for count in range(1, min(16 - source % 16, 16 - destination % 16) + 1):
                        for face in (0, 1):
                            hops = transfers(source, destination, count, face, base, 0x594000)
                            for kind, read, write, length in hops:
                                alignment = 64 if kind == 'dram' else 16
                                self.assertEqual(read % alignment, write % alignment)
                                self.assertEqual(length, count * 32)
                                self.assertGreaterEqual(write, base)
                                self.assertLessEqual(write + length, base + 8192)
                            if len(hops) == 2:
                                self.assertEqual(hops[0][2], hops[1][1])
                                self.assertLessEqual(hops[0][2] + count * 32, base + 6144)
                            self.assertGreaterEqual(hops[-1][2], base + 6144)

    def test_simulator_regression_requires_two_hops(self):
        hops = transfers(4, 3, 1, 0, 0x1B900, 0x594E00)
        self.assertEqual(hops[0][1], 0x594E80)
        self.assertEqual(hops[-1][2], 0x1D160)
        self.assertEqual(len(hops), 2)
