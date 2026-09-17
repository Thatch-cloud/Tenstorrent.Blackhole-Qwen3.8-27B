import unittest

import torch

from dspark_inputs import MASK_TOKEN, query_inputs, verifier_inputs


class DSparkInputsTests(unittest.TestCase):
    def test_query_has_anchor_plus_six_masks_and_samples_all_seven_rows(self):
        inputs = query_inputs(1596, 4096)
        self.assertEqual(inputs['identifiers'].tolist(), [[1596] + [MASK_TOKEN] * 6])
        self.assertEqual(inputs['query_positions'].tolist(), [list(range(4096, 4103))])
        self.assertEqual(inputs['verifier_positions'].tolist(), [list(range(4096, 4104))])
        self.assertEqual(inputs['first_sampled_query_row'], 0)

    def test_verifier_prepends_anchor_without_discarding_the_first_proposal(self):
        proposals = torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.int64)
        block = verifier_inputs(1596, proposals)
        self.assertEqual(block.tolist(), [[1596, 10, 11, 12, 13, 14, 15, 16]])
        block.zero_()
        self.assertEqual(proposals.tolist(), [[10, 11, 12, 13, 14, 15, 16]])
        with self.assertRaises(ValueError):
            verifier_inputs(1596, proposals[:, 1:])

    def test_invalid_width_ids_and_no_room_for_full_verification_fail(self):
        for anchor, prefix in ((True, 4096), (-1, 4096), (248320, 4096), (1, 0), (1, 262137), (1, True)):
            with self.assertRaises(ValueError):
                query_inputs(anchor, prefix)
        self.assertEqual(int(query_inputs(248319, 262136)['verifier_positions'][0, -1]), 262143)
        for proposals in (torch.zeros(1, 8, dtype=torch.int64), torch.zeros(1, 7),
                torch.full((1, 7), -1), torch.full((1, 7), 248320)):
            with self.assertRaises(ValueError):
                verifier_inputs(1, proposals)


if __name__ == '__main__':
    unittest.main()
