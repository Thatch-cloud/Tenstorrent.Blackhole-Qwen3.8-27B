from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from dspark_fixed_inputs import fixed_mask, proposal_inputs, validate_fixed_mask


class FixedInputsTests(unittest.TestCase):
    def test_full_history_and_proposal_keys_are_visible_but_the_capacity_gap_is_not(self):
        for position in (4093, 4096, 4111, 4384):
            mask = fixed_mask(position, 4384, 15)
            self.assertEqual(tuple(mask.shape), (1, 1, 32, 4416))
            self.assertTrue(torch.all(mask[..., :15, :position] == 0))
            self.assertTrue(torch.isneginf(mask[..., :15, position:4384]).all())
            self.assertTrue(torch.all(mask[..., :15, 4384:4399] == 0))
            self.assertTrue(torch.isneginf(mask[..., :15, 4399:]).all())
            self.assertEqual(torch.count_nonzero(mask[..., 15:, :] == 0), 17)
            self.assertTrue(torch.all(mask[..., 15:, 4384] == 0))
            validate_fixed_mask(mask, position, 4384, 15)

    def test_exposing_padding_or_hiding_any_oldest_or_final_proposal_key_fails(self):
        for index, value in ((0, float('-inf')), (4383, 0), (4398, float('-inf')), (4399, 0)):
            mask = fixed_mask(4096, 4384, 15)
            mask[..., 0, index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_fixed_mask(mask, 4096, 4384, 15)

    def test_absolute_rotary_positions_do_not_move_to_the_physical_capacity(self):
        cosine = torch.full((1, 1, 15, 128), 7, dtype=torch.bfloat16)
        sine = torch.full_like(cosine, 9)
        rotary = SimpleNamespace(tables=Mock(return_value=(cosine, sine)))
        values = proposal_inputs(71093, 4109, 4384, 15, rotary)
        rotary.tables.assert_called_once_with(4109, 15)
        self.assertEqual(values['identifiers'][0].tolist(), [71093] + [248070] * 14)
        self.assertEqual(values['anchor'].item(), 71093)
        self.assertTrue(torch.equal(values['cosine'][..., :15, :], cosine))
        self.assertTrue(torch.equal(values['sine'][..., :15, :], sine))
        self.assertTrue(torch.all(values['cosine'][..., 15:, :] == 1))
        self.assertTrue(torch.all(values['sine'][..., 15:, :] == 0))
        self.assertEqual(values['live'].dtype, torch.float32)
        self.assertEqual(values['live'].sum(), 15)

    def test_invalid_frontier_capacity_and_unqualified_width_reject(self):
        for position, capacity, proposals in ((True, 4384, 15), (0, 4384, 15), (4385, 4384, 15),
                (4096, 4385, 15), (4096, 8224, 15), (4096, 4384, 31)):
            with self.subTest(position=position, capacity=capacity, proposals=proposals), self.assertRaises(ValueError):
                fixed_mask(position, capacity, proposals)


if __name__ == '__main__':
    unittest.main()
