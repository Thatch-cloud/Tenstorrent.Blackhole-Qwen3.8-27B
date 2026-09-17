from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import dspark_full_attention as attention
from draft_dot import dot_geometry


def reference(query, key, value, mask):
    return torch.nn.functional.scaled_dot_product_attention(query.float(),
        key.float().repeat_interleave(4, dim=1), value.float().repeat_interleave(4, dim=1),
        attn_mask=mask.float(), is_causal=False)


def fixture(context, proposals, pattern=0):
    generator = torch.Generator().manual_seed(383900 + pattern)
    keys = attention.geometry(context, proposals)[-1][1]
    return (torch.randn(1, 16, 32, 128, generator=generator).bfloat16(),
        torch.randn(1, 4, keys, 128, generator=generator).bfloat16(),
        torch.randn(1, 4, keys, 128, generator=generator).bfloat16(), attention.full_mask(context, proposals))


class DSparkFullAttentionTests(unittest.TestCase):
    def test_complete_4k_history_never_exceeds_cached_kernel_geometry(self):
        for proposals in (7, 15):
            self.assertEqual(attention.geometry(4096, proposals), ((0, 2048), (2048, 4096), (4096, 4160)))
            for start, end in attention.geometry(4096, proposals):
                dot_geometry((1, 16, 32, 128), (1, 16, end - start, 128))
                dot_geometry((1, 16, 32, end - start), (1, 16, 128, end - start))

    def test_tail_and_maximum_context_are_fully_covered(self):
        for context in (1, 32, 2033, 2048, 4097, 8192):
            for proposals in (7, 15):
                chunks = attention.geometry(context, proposals)
                self.assertEqual(chunks[0][0], 0)
                self.assertGreaterEqual(chunks[-1][1], context + proposals)
                self.assertLess(chunks[-1][1] - context - proposals, 64)
                self.assertTrue(all(end - start <= 2048 and (end - start) % 64 == 0 for start, end in chunks))
                self.assertTrue(all(left[1] == right[0] for left, right in zip(chunks, chunks[1:])))

    def test_invalid_geometry_rejects(self):
        for context, proposals in ((0, 7), (8193, 7), (True, 7), (4096, 8), (4096, True)):
            with self.subTest(context=context, proposals=proposals), self.assertRaises(ValueError):
                attention.geometry(context, proposals)

    def test_padding_mask_keeps_a_finite_global_anchor_for_inactive_queries(self):
        mask = attention.full_mask(4096, 15)
        attention.validate_mask(mask, 4096, 15)
        self.assertTrue(torch.isneginf(mask[:, :, 15:, :4096]).all())
        self.assertTrue(torch.all(mask[:, :, 15:, 4096] == 0))
        self.assertTrue(torch.all(mask[:, :, :15, :4111] == 0))
        self.assertTrue(torch.isneginf(mask[:, :, :15, 4111:]).all())
        changed = mask.clone()
        changed[:, :, 0, 0] = float('-inf')
        with self.assertRaises(ValueError):
            attention.validate_mask(changed, 4096, 15)

    def execute(self, values, context, proposals):
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram', float32=torch.float32, bfloat16=torch.bfloat16,
            typecast=lambda value, dtype:value.to(dtype),
            slice=lambda value, start, end:value[tuple(slice(first, last) for first, last in zip(start, end))],
            repeat_interleave=lambda value, repeats, dim, **kwargs:torch.repeat_interleave(value, repeats, dim=dim),
            multiply=lambda left, right, **kwargs:left * right,
            add=lambda left, right, **kwargs:left + right,
            subtract=lambda left, right, **kwargs:left - right,
            max=lambda value, dim, keepdim:torch.amax(value, dim=dim, keepdim=keepdim),
            maximum=torch.maximum, exp=lambda value, **kwargs:torch.exp(value),
            reciprocal=torch.reciprocal, transpose=torch.transpose)

        def dot(mesh, left, right, owned, **kwargs):
            dot_geometry(tuple(left.shape), tuple(right.shape))
            output = torch.matmul(left, right.transpose(-1, -2))
            owned.append(output)
            return output

        def total(mesh, value, owned):
            output = value.sum(dim=-1, keepdim=True)
            owned.append(output)
            return output

        with patch.object(attention, 'validate_inputs', return_value=attention.geometry(context, proposals)), \
                patch.object(attention, 'fused_dot', side_effect=dot), patch.object(attention, 'row_sum', side_effect=total):
            return attention.execute(operations, object(), *values, [], context_rows=context, proposals=proposals, mask_validated=True)

    def test_global_softmax_matches_full_context_reference_including_all_masked_local_chunks(self):
        for context, proposals in ((32, 7), (4096, 7), (4096, 15), (4097, 15)):
            with self.subTest(context=context, proposals=proposals):
                values = fixture(context, proposals)
                output = self.execute(values, context, proposals)
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output.float(), reference(*values), rtol=.01, atol=.01)

    def test_oldest_history_and_last_proposal_both_affect_output_but_padding_does_not(self):
        values = fixture(4096, 15)
        original = self.execute(values, 4096, 15)
        for index in (0, 4110):
            changed = [value.clone() for value in values]
            changed[2][:, :, index] += 64
            self.assertFalse(torch.equal(self.execute(changed, 4096, 15)[:, :, :15], original[:, :, :15]))
        changed = [value.clone() for value in values]
        for value in changed[1:3]:
            value[:, :, 4111:] = 31
        self.assertTrue(torch.equal(self.execute(changed, 4096, 15), original))


if __name__ == '__main__':
    unittest.main()
