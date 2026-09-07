from types import SimpleNamespace
import unittest

import torch

from draft_shared_head import candidate_chunks, merge_chunk_candidates, shared_head_candidates


class SharedHeadTests(unittest.TestCase):
    def fixture(self):
        values = -torch.arange(248320, dtype=torch.float32).repeat(8, 1)
        boundaries = [0, 32767, 32768, 65535, 65536, 98303, 98304, 124159,
                      124160, 156927, 156928, 189695, 189696, 222463, 222464, 248319]
        values[:, boundaries] = torch.arange(1, 17, dtype=torch.float32)
        chunks = []
        for chip in range(2):
            for start, stop in candidate_chunks():
                scores, indices = values[:, chip * 124160 + start:chip * 124160 + stop].topk(16, dim=-1)
                chunks.append(dict(chip=chip, start=start, stop=stop, values=scores, indices=indices))
        return values, chunks

    def test_global_top16_and_proposal_rows_preserved(self):
        values, chunks = self.fixture()
        tokens, scores = merge_chunk_candidates(list(reversed(chunks)))
        expected_scores, expected_tokens = values.topk(16, dim=-1)
        self.assertTrue(torch.equal(tokens, expected_tokens[None, 1:8]))
        self.assertTrue(torch.equal(scores, expected_scores[None, 1:8]))

    def test_missing_duplicate_and_padded_indices_rejected(self):
        _, chunks = self.fixture()
        for selected in (chunks[:-1], chunks + [chunks[0]]):
            with self.assertRaises(ValueError):
                merge_chunk_candidates(selected)
        chunks[-1]['indices'][0, 0] = 32767
        with self.assertRaisesRegex(ValueError, 'in-range'):
            merge_chunk_candidates(chunks)

    def test_candidate_ties_are_deterministic_among_received_ids(self):
        _, chunks = self.fixture()
        for chunk in chunks:
            chunk['values'].zero_()
        first = merge_chunk_candidates(chunks)
        second = merge_chunk_candidates(list(reversed(chunks)))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.all(first[0][..., 1:] > first[0][..., :-1]))

    def test_borrowed_head_no_target_norm_or_full_logit_gather(self):
        calls = []
        def tensor(shape):
            return SimpleNamespace(shape=shape, dtype='bf16', layout='tile', memory_config=lambda: 'dram')
        def linear(hidden, weight):
            calls.append(('linear', weight))
            return tensor((1, 1, 8, 124160))
        def slice_tensor(value, start, stop, stride):
            return tensor((1, 1, 8, stop[-1] - start[-1]))
        def pad(value, padding, fill):
            calls.append(('pad', fill))
            return tensor((1, 1, 8, value.shape[-1] + padding[-1][1]))
        def topk(value, **options):
            calls.append(('topk', value.shape[-1], options))
            return tensor((1, 1, 8, 16)), tensor((1, 1, 8, 16))
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
                                     linear=linear, slice=slice_tensor, pad=pad, topk=topk)
        weight = object()
        model = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True, lm_head_weight=weight)
        owned = []
        outputs = shared_head_candidates(operations, model, tensor((1, 1, 8, 5120)), owned)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(owned), 14)
        self.assertIs(calls[0][1], weight)
        self.assertEqual(sum(call[0] == 'pad' for call in calls), 1)
        self.assertEqual([call[1] for call in calls if call[0] == 'topk'], [32768] * 4)
