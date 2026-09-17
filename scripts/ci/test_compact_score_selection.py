import random
import struct
import unittest

from compact_score_selection import fp32_bits, ordered_key, partitions, select


class CompactSelectionTests(unittest.TestCase):
    def test_random_finite_encodings_and_full_vocabulary(self):
        generator = random.Random(38527)
        bits = []
        while len(bits) < 248320:
            value = generator.getrandbits(32)
            if value & 0x7f800000 != 0x7f800000:
                bits.append(value)
        values = [struct.unpack('<f', struct.pack('<I', value))[0] for value in bits]
        reference = max(range(len(values)), key=lambda token: values[token])
        for workers in (1, 64, 110):
            self.assertEqual(select(bits, workers), reference)

    def test_ties_at_partition_boundaries_and_signed_zero(self):
        for workers in (1, 3, 64, 110):
            scores = [-2.0] * 248320
            ranges = partitions(len(scores), workers)
            for start, end in ranges:
                scores[start] = 1.0
                scores[end - 1] = 1.0
            self.assertEqual(select(fp32_bits(scores), workers), 0)
        self.assertEqual(select(fp32_bits([-1.0, -0.0, 0.0, -0.0]), 3), 1)
        self.assertEqual(ordered_key(0), ordered_key(0x80000000))

    def test_nonfinite_and_invalid_geometry_rejected(self):
        for bits in (0x7f800000, 0xff800000, 0x7fc00000, 0xffffffff, -1, 2 ** 32):
            with self.assertRaises(ValueError):
                ordered_key(bits)
        for vocabulary, workers in ((0, 1), (64, 0), (64, 65), (True, 1)):
            with self.assertRaises(ValueError):
                partitions(vocabulary, workers)
