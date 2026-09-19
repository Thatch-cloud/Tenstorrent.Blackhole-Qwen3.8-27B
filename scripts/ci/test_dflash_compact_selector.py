import unittest

import torch

from draft_selector import select_active_candidates as reference
from dflash_compact_selector import select_active_candidates as candidate


def operands(positions, batch=1, seed=3827):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn((batch, positions, 256), generator=generator).bfloat16()
    choices = torch.stack([torch.randperm(1024, generator=generator)[:16]
        for unused in range(batch * positions)]).reshape(batch, positions, 16)
    unary = torch.randn((batch, positions, 16), generator=generator)
    codes = [torch.randn((1024, 256), generator=generator).bfloat16() for unused in range(2)]
    return [hidden, choices, unary, *codes, torch.arange(batch, dtype=torch.int64)]


class CompactSelectorTests(unittest.TestCase):
    def test_tokens_and_every_score_are_bitwise_equal(self):
        for positions in (1, 7, 8, 15, 16, 31):
            for batch in (1, 2):
                for seed in (3827, 3828):
                    values = operands(positions, batch, seed)
                    before = [value.clone() for value in values]
                    actual, expected = candidate(*values), reference(*values)
                    with self.subTest(positions=positions, batch=batch, seed=seed):
                        for current, control in zip(actual, expected, strict=True):
                            self.assertTrue(torch.equal(current, control))
                        self.assertTrue(all(torch.equal(current, original)
                            for current, original in zip(values, before, strict=True)))

    def test_ties_keep_first_candidate_order_not_token_order(self):
        values = operands(15)
        values[0].zero_()
        values[2].zero_()
        tokens, scores = candidate(*values)
        self.assertTrue(torch.equal(tokens, values[1][..., 0]))
        self.assertTrue(torch.equal(scores, reference(*values)[1]))

    def test_late_invalid_rows_and_overflow_fail_closed(self):
        for corruption in ('nan', 'duplicate', 'bounds', 'overflow'):
            values = operands(15)
            if corruption == 'nan':
                values[0][:, 14, 0] = float('nan')
            elif corruption == 'duplicate':
                values[1][:, 14, 1] = values[1][:, 14, 0]
            elif corruption == 'bounds':
                values[1][:, 14, 0] = 1024
            else:
                values[0] = torch.full_like(values[0], 1e200, dtype=torch.float64)
                values[3] = torch.full_like(values[3], 1e200, dtype=torch.float64)
            for implementation in (reference, candidate):
                with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                    implementation(*values)


if __name__ == '__main__':
    unittest.main()
