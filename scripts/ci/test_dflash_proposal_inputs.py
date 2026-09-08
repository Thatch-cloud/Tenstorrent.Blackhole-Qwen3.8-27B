import unittest

import torch

from dflash_proposal_inputs import proposal_contexts, proposal_inputs
from draft_attention import draft_attention_mask
from draft_head_preparation import rope_tables


class ProposalInputTests(unittest.TestCase):
    def test_all_request_contexts_are_prepared_without_future_token_knowledge(self):
        self.assertEqual(proposal_contexts(170, 513), (256, 512, 1024))
        self.assertEqual(proposal_contexts(2047, 513), (2048,))
        self.assertEqual(proposal_contexts(256, 1), (256, 512))

    def test_mask_and_rope_relocation_preserve_every_visible_token(self):
        for block in (8, 32):
            for position, rows, context in ((170, 170, 256), (178, 178, 256), (256, 256, 256),
                    (257, 257, 512), (3072, 2048, 2048)):
                with self.subTest(block=block, position=position):
                    values = proposal_inputs(248044, position, rows, block, context)
                    mask, rope = values['mask'], values['rope']
                    compact = draft_attention_mask(rows, block_rows=block)
                    restored = torch.cat((mask[..., :rows], mask[..., context:context + block]), dim=-1)
                    self.assertTrue(torch.equal(restored, compact[..., :rows + block]))
                    self.assertTrue(torch.isneginf(mask[..., rows:context]).all())
                    self.assertTrue(torch.isneginf(mask[..., context + block:]).all())
                    self.assertTrue(torch.isfinite(mask).any(-1).all())
                    for actual, expected in zip(rope['q'], rope_tables(position, 32), strict=True):
                        self.assertTrue(torch.equal(actual, expected))
                    for actual, expected in zip(rope['k'], rope_tables(position - rows, rows + block), strict=True):
                        restored = torch.cat((actual[..., :rows, :], actual[..., context:context + block, :]), dim=-2)
                        self.assertTrue(torch.equal(restored, expected))
                    self.assertEqual(values['identifiers'].tolist(), [[248044, *([248070] * (block - 1))]])

    def test_rejects_unbounded_or_uncommitted_geometry(self):
        valid = (17, 170, 170, 8, 256)
        for index, value in ((0, -1), (0, 248320), (1, 169), (2, 0), (2, 257), (3, 16), (4, 128), (4, True)):
            arguments = list(valid)
            arguments[index] = value
            with self.assertRaises(ValueError):
                proposal_inputs(*arguments)
