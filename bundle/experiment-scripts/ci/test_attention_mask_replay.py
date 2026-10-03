import unittest

import torch

from attention_head_fold import causal_mask
from attention_head_fold import chunk_groups
from attention_mask_replay import mask_position, source_hashes, validate_ticket


class MaskReplayTests(unittest.TestCase):
    def test_short_context_requires_opt_in_and_fixed_native_chunk_signature(self):
        for capacity in (256, 512, 768):
            first = max(128, capacity - 256)
            for start in range(first, capacity - 7):
                validate_ticket(start, 8, capacity, short_context=True)
                self.assertEqual({group['signature'] for group in chunk_groups(start, 8)}, {(256, capacity)})
                with self.assertRaises(ValueError):
                    validate_ticket(start, 8, capacity)
        for start, rows, capacity in ((127, 8, 256), (249, 8, 256), (255, 8, 512),
                                      (170, 4, 256), (170, 16, 256), (768, 8, 1024)):
            with self.assertRaises(ValueError):
                validate_ticket(start, rows, capacity, short_context=True)
        with self.assertRaises(ValueError):
            validate_ticket(170, 8, 256, short_context=1)

    def test_sources_resolve_from_installed_helper_not_harness_layout(self):
        hashes = source_hashes()
        self.assertEqual(set(hashes), {'attention_mask_replay.py', 'attention_mask_replay.cpp'})
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_ticket_rejects_crossing_or_stale_family(self):
        for start, rows, capacity in ((4095, 8, 4352), (4350, 4, 4352), (4096, 0, 4352),
                                      (True, 4, 4352), (4096, 4, 4300), (0, 4, 256)):
            with self.assertRaises(ValueError):
                validate_ticket(start, rows, capacity)
        for start in (4096, 4103, 4320):
            validate_ticket(start, 32, 4352)
            self.assertEqual({group['signature'] for group in chunk_groups(start, 32)}, {(256, 4352)})

    def test_all_folded_heads_match_static_native_mask(self):
        for rows in range(1, 9):
            for batch in range(3):
                for start in (4096, 4103, 4320):
                    expected = causal_mask(rows, start + 4 + batch * rows, 4352)
                    for head in range(rows * 12):
                        position = mask_position(start, rows, batch, head, offset=4)
                        self.assertTrue(torch.all(expected[0, 0, head, :position + 1] == 0))
                        self.assertTrue(torch.all(torch.isneginf(expected[0, 0, head, position + 1:])))
