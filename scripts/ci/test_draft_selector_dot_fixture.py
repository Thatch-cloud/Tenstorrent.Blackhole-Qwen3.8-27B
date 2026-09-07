import unittest

import torch

from draft_selector_dot_fixture import prepare_selector_dot


class SelectorDotFixtureTests(unittest.TestCase):
    def test_learned_fixture_packing_preserves_all_active_scores(self):
        weights = {'candidate_selector.predecessor_codebook': torch.ones(1, 256, dtype=torch.bfloat16).expand(248320, 256),
                   'candidate_selector.successor_codebook': torch.ones(1, 256, dtype=torch.bfloat16).expand(248320, 256)}
        fixture = prepare_selector_dot(weights)
        self.assertEqual(fixture['left'].shape, (1, 16, 32, 256))
        self.assertEqual(fixture['right'].shape, (1, 16, 32, 256))
        for key in ('left', 'right'):
            self.assertEqual(int(torch.count_nonzero(fixture[key][:, 7:])), 0)
            self.assertEqual(int(torch.count_nonzero(fixture[key][:, :, 16:])), 0)
        actual = fixture['left'].double() @ fixture['right'].double().transpose(-1, -2)
        actual = actual[:, :7, :16, :16] + fixture['unary'].double()[:, :, None, :]
        torch.testing.assert_close(actual, fixture['expected_scores'], atol=1e-12, rtol=1e-12)
        self.assertEqual(fixture['expected_path'].shape, (1, 7))

    def test_wrong_codebooks_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Pinned selector'):
            prepare_selector_dot({'candidate_selector.predecessor_codebook': torch.zeros(2, 256),
                                  'candidate_selector.successor_codebook': torch.zeros(2, 256)})
