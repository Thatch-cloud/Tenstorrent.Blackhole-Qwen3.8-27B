"""The causal convolution must restart at every packed segment boundary.

grouped_causal_convolution shifts by one ROW: row r reads row r-1. A packed block
holds several users' rows, so without segment spans the first row of user B reads
the last draft row of user A. That row is B's anchor, so the whole block below it
is corrupted - and nothing in the attention mask can catch it, because the damage
happens before attention.
"""

import unittest

import torch

from draft_convolution import convolution_reference, validate_boundaries


def operands(rows):
    torch.manual_seed(20260921)
    hidden = torch.randn(1, 1, rows, 5120).bfloat16()
    dynamic = [torch.randn(1, 1, rows, 320).bfloat16() for _ in range(2)]
    base = [torch.randn(1, 1, 1, 5120).bfloat16() for _ in range(2)]
    return hidden, dynamic, base


class PackedConvolutionTests(unittest.TestCase):
    def test_each_packed_user_matches_a_standalone_32_row_run_of_its_own_rows(self):
        """Rows must be a legal block width, so compare against a 32-row run whose
        first 16 rows are this user's and whose tail is ignored."""
        hidden, dynamic, base = operands(32)
        packed = convolution_reference(hidden, dynamic, base, boundaries=((0, 16), (16, 32)))
        for index, (start, stop) in enumerate(((0, 16), (16, 32))):
            shifted = torch.zeros_like(hidden)
            shifted[..., :stop - start, :] = hidden[..., start:stop, :]
            moved = [torch.zeros_like(value) for value in dynamic]
            for value, source in zip(moved, dynamic):
                value[..., :stop - start, :] = source[..., start:stop, :]
            alone = convolution_reference(shifted, moved, base)
            self.assertTrue(torch.equal(packed[..., start:stop, :], alone[..., :stop - start, :]),
                            'user %d differs from its standalone convolution' % index)

    def test_without_boundaries_a_packed_block_leaks_across_the_seam(self):
        """The bug this guards against, stated as a measurement."""
        hidden, dynamic, base = operands(32)
        packed = convolution_reference(hidden, dynamic, base, boundaries=((0, 16), (16, 32)))
        continuous = convolution_reference(hidden, dynamic, base)
        self.assertTrue(torch.equal(packed[..., :16, :], continuous[..., :16, :]),
                        'the first user is unaffected either way')
        self.assertFalse(torch.equal(packed[..., 16, :], continuous[..., 16, :]),
                         'the second user anchor is exactly what the seam corrupts')
        self.assertTrue(torch.equal(packed[..., 17:, :], continuous[..., 17:, :]),
                        'a depth-two convolution corrupts only the first row of a segment')

    def test_one_segment_is_todays_behaviour(self):
        hidden, dynamic, base = operands(32)
        self.assertTrue(torch.equal(convolution_reference(hidden, dynamic, base, boundaries=((0, 32),)),
                                    convolution_reference(hidden, dynamic, base)))

    def test_malformed_spans_are_refused(self):
        for spans in (((0, 16),), ((0, 16), (17, 32)), ((16, 32), (0, 16)), ((0, 0), (0, 32)), ()):
            with self.assertRaises(ValueError):
                validate_boundaries(spans, 32)
        self.assertIsNone(validate_boundaries(None, 32))

    def test_the_fused_kernel_now_takes_the_same_spans(self):
        """It used to refuse them. draft_convolution_fused_io.cpp reads a seam
        bitmask as its ninth runtime argument, and test_draft_convolution_fused_seams
        pins that the mask means what the kernel reads."""
        from draft_convolution_fused import seam_mask

        self.assertEqual(seam_mask(((0, 16), (16, 32)), 32), (1 << 0) | (1 << 16))
        self.assertEqual(seam_mask(None, 32), 1)


if __name__ == '__main__':
    unittest.main()
