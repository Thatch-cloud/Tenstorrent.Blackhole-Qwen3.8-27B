import unittest
import unittest.mock

import torch

from dflash_packed_proposal import (DRAFT_FILLER, packed_identifiers, select_packed,
                                    split_selection, user_slices)


class PackedProposalTests(unittest.TestCase):
    def test_identifiers_place_each_anchor_at_its_own_block_row(self):
        packed = packed_identifiers([1234, 5678], block_rows=16)
        self.assertEqual(tuple(packed.shape), (1, 32))
        self.assertEqual(int(packed[0, 0]), 1234)
        self.assertEqual(int(packed[0, 16]), 5678)
        self.assertTrue(all(int(value) == DRAFT_FILLER for value in packed[0, 1:16]))
        self.assertTrue(all(int(value) == DRAFT_FILLER for value in packed[0, 17:32]))

    def test_one_user_reproduces_todays_identifier_row(self):
        """DFlashDevice builds [[seed, 248070 x max_drafts]] with max_drafts = 15."""
        packed = packed_identifiers([99], block_rows=16)
        today = torch.tensor([[99, *([DRAFT_FILLER] * 15)]], dtype=torch.int64)
        self.assertTrue(torch.equal(packed[:, :16], today))

    def test_slices_follow_the_anchor_drop_in_merge_chunk_candidates(self):
        parts = user_slices(2, 16)
        self.assertEqual([(part['drafts'].start, part['drafts'].stop) for part in parts],
                         [(0, 15), (16, 31)])
        self.assertEqual([(part['selector'].start, part['selector'].stop) for part in parts],
                         [(1, 16), (17, 32)])

    def test_split_gives_each_user_its_own_rows(self):
        hidden = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1).expand(1, 32, 256)
        candidates = torch.arange(31, dtype=torch.int64).reshape(1, 31, 1).expand(1, 31, 16)
        parts = split_selection(hidden, candidates, candidates.clone(), 2, 16)
        self.assertEqual([int(part['hidden'][0, 0, 0]) for part in parts], [1, 17])
        self.assertEqual([int(part['candidates'][0, 0, 0]) for part in parts], [0, 16])
        self.assertEqual([int(part['candidates'][0, -1, 0]) for part in parts], [14, 30])
        for part in parts:
            self.assertEqual(part['hidden'].shape[1], 15)
            self.assertEqual(part['candidates'].shape[1], 15)

    def test_selection_runs_once_per_user_with_its_own_anchor(self):
        seen = []

        def fake_select(hidden, candidates, unary, predecessors, successors, anchors):
            seen.append((int(anchors[0]), int(candidates[0, 0, 0])))
            return torch.full((1, 15), int(anchors[0]), dtype=torch.int64), None

        hidden = torch.zeros(1, 32, 256)
        candidates = torch.arange(31, dtype=torch.int64).reshape(1, 31, 1).expand(1, 31, 16)
        parts = split_selection(hidden, candidates, candidates.clone(), 2, 16)
        with unittest.mock.patch('draft_selector.select_active_candidates', fake_select):
            tokens = select_packed(parts, [7, 9], [15, 3], torch.zeros(2, 2), torch.zeros(2, 2))
        self.assertEqual(seen, [(7, 0), (9, 16)], 'each user keeps its own anchor and rows')
        self.assertEqual([len(part) for part in tokens], [15, 3])
        self.assertEqual(tokens[0][0], 7)
        self.assertEqual(tokens[1][0], 9)

    def test_bad_counts_and_shapes_are_refused(self):
        hidden, candidates = torch.zeros(1, 32, 256), torch.zeros(1, 31, 16, dtype=torch.int64)
        parts = split_selection(hidden, candidates, candidates.clone(), 2, 16)
        with self.assertRaises(ValueError):
            select_packed(parts, [1, 2], [16, 1], torch.zeros(2, 2), torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            split_selection(torch.zeros(1, 16, 256), candidates, candidates, 2, 16)
        with self.assertRaises(ValueError):
            packed_identifiers([1, 2, 3], block_rows=16)
        with self.assertRaises(ValueError):
            user_slices(3, 16)


if __name__ == '__main__':
    unittest.main()
