import math
import unittest

import torch

from draft_head_preparation import rope_tables
from dspark_rope_tables import DSparkRotary
from test_dspark_intake import configuration


class DSparkRopeTableTests(unittest.TestCase):
    def config(self):
        config = configuration()
        config['max_position_embeddings'] = 262144
        return config

    def test_correction_band_and_both_frequency_extremes(self):
        rotary = DSparkRotary(self.config())
        low, high = rotary.correction_range
        self.assertEqual((low, high), (14, 29))
        self.assertAlmostEqual(rotary.attention_scaling, 1.3465735902799727)
        for index, frequency in enumerate(rotary.inverse_frequency):
            ramp = min(max((index - low) / (high - low), 0), 1)
            expected = (ramp / 32 + 1 - ramp) / (10000000 ** (2 * index / 128))
            self.assertAlmostEqual(float(frequency) / expected, 1., places=6)

    def test_scale_applies_even_at_zero_and_tables_are_not_dflash2(self):
        rotary = DSparkRotary(self.config())
        cosine, sine = rotary.tables(0, 1, dtype=torch.float32)
        self.assertTrue(torch.equal(sine, torch.zeros_like(sine)))
        self.assertTrue(torch.equal(cosine, torch.full_like(cosine, rotary.attention_scaling)))
        for start in (0, 4096, 8191, 8192, 32768, 64504, 262137):
            actual = rotary.tables(start, 7)
            ordinary = rope_tables(start, 7)
            self.assertFalse(torch.equal(actual[0], ordinary[0]))
            self.assertTrue(torch.equal(actual[0][..., :64], actual[0][..., 64:]))

    def test_chunking_and_absolute_query_suffix_are_consistent(self):
        rotary = DSparkRotary(self.config())
        whole = rotary.tables(8189, 11)
        parts = [rotary.tables(8189, 4), rotary.tables(8193, 7)]
        for index in range(2):
            self.assertTrue(torch.equal(whole[index], torch.cat([part[index] for part in parts], dim=2)))
        block = rotary.block_tables(8189, 4)
        for index in range(2):
            self.assertTrue(torch.equal(block['k'][index], whole[index]))
            self.assertTrue(torch.equal(block['q'][index], parts[1][index]))
            self.assertNotEqual(block['q'][index].data_ptr(), block['k'][index].data_ptr())

    def test_selected_positions_match_float32_phase_then_bf16_tables(self):
        rotary = DSparkRotary(self.config())
        positions = torch.tensor([262143, 1, 8192, 0, 8191], dtype=torch.int64)
        cosine, sine = rotary.positions(positions, dtype=torch.float32)
        for row, position in enumerate(positions.tolist()):
            for column in (0, 14, 15, 28, 29, 63):
                angle = float(torch.tensor(position, dtype=torch.float32) * rotary.inverse_frequency[column])
                self.assertAlmostEqual(float(cosine[0, 0, row, column]), math.cos(angle) * rotary.attention_scaling, places=6)
                self.assertAlmostEqual(float(sine[0, 0, row, column]), math.sin(angle) * rotary.attention_scaling, places=6)
        for actual, expected in zip(rotary.positions(positions), (cosine, sine), strict=True):
            self.assertTrue(torch.equal(actual, expected.bfloat16()))

    def test_configuration_snapshot_and_invalid_requests(self):
        config = self.config()
        rotary = DSparkRotary(config)
        config['rope_parameters']['factor'] = 1.
        self.assertGreater(rotary.attention_scaling, 1.)
        for field, value in (('max_position_embeddings', 8192), ('head_dim', 64)):
            invalid = self.config()
            invalid[field] = value
            with self.assertRaises(ValueError):
                DSparkRotary(invalid)
        for start, rows in ((True, 7), (-1, 7), (262143, 2), (0, 0)):
            with self.assertRaises(ValueError):
                rotary.tables(start, rows)
        for positions in (torch.tensor([1.]), torch.tensor([-1]), torch.tensor([262144]),
                torch.empty(0, dtype=torch.int64), torch.tensor([[0]])):
            with self.assertRaises(ValueError):
                rotary.positions(positions)
        with self.assertRaises(ValueError):
            rotary.tables(0, 1, dtype=torch.float64)
        with self.assertRaises(ValueError):
            rotary.block_tables(0, 0)


if __name__ == '__main__':
    unittest.main()
