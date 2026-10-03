import unittest

import torch

from gdn_publication_fixture import expected, host_layer


class PublicationFixtureTests(unittest.TestCase):
    def test_deterministic_distinct_patterns_and_layers(self):
        first = host_layer(0, 0)
        repeated = host_layer(0, 0)
        for left, right in zip(first, repeated, strict=True):
            self.assertTrue(torch.equal(left.view(torch.int16), right.view(torch.int16)))
        self.assertFalse(torch.equal(first[0], host_layer(1, 0)[0]))
        self.assertFalse(torch.equal(first[0], host_layer(0, 1)[0]))
        self.assertFalse(torch.equal(first[0][:1], first[0][1:]))

    def test_every_prefix_keeps_chips_and_inactive_slots_separate(self):
        values = host_layer(0, 0)
        original_native = values[10].clone()
        for prefix in range(17):
            result = expected(values, prefix)
            for chip in range(2):
                selected = values[0][chip] if prefix == 0 else values[5][chip * 16 + prefix - 1]
                self.assertTrue(torch.equal(result[10][chip * 8], selected))
                self.assertTrue(torch.equal(result[15][chip], selected))
                self.assertTrue(torch.equal(result[10][chip * 8 + 1:chip * 8 + 8],
                    values[10][chip * 8 + 1:chip * 8 + 8]))
            for slot in range(1, 5):
                selected = values[slot] if prefix == 0 else values[5 + slot][:, prefix - 1:prefix]
                self.assertTrue(torch.equal(result[15 + slot], selected))
                self.assertTrue(torch.equal(result[10 + slot][:, :1], selected))
                self.assertTrue(torch.equal(result[10 + slot][:, 1:], values[10 + slot][:, 1:]))
        self.assertTrue(torch.equal(values[10], original_native))

    def test_invalid_geometry_or_selection_is_rejected(self):
        values = host_layer(0, 0)
        for prefix in (-1, 17, True):
            with self.assertRaises(ValueError):
                expected(values, prefix)
        with self.assertRaises(ValueError):
            expected(values[:-1], 0)
