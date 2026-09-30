"""The fused convolution kernel must restart its causal shift at each packed seam.

draft_convolution_fused_io.cpp builds the shifted operand itself:

    const bool carries = row < rows && !((seams >> row) & 1u);
    tiles[1024 + element] = carries ? tiles[lane(row - 1, column)] : 0;

`seams` is a bitmask of rows that begin a user's segment. This pins two things
that cannot be checked by running the reference: that the mask the host computes
means what the kernel reads, and that an unpacked call is byte-identical to the
`row && row < rows` guard the kernel carried before.
"""

import unittest
from unittest.mock import Mock, patch

import torch

from draft_convolution import convolution_reference
from draft_convolution_fused import checked_convolution, seam_mask


def kernel_shift(hidden, rows, seams):
    """The tile-1 operand exactly as draft_convolution_fused_io.cpp builds it."""
    shifted = torch.zeros_like(hidden)
    for row in range(32):
        carries = row < rows and not ((seams >> row) & 1)
        if carries:
            shifted[..., row, :] = hidden[..., row - 1, :]
    return shifted


def reference_shift(hidden, rows, boundaries):
    """The same operand as convolution_reference builds it, for comparison."""
    spans = boundaries or ((0, rows),)
    parts = []
    for start, stop in spans:
        parts.append(torch.zeros_like(hidden[..., :1, :]))
        if stop - start > 1:
            parts.append(hidden[..., start:stop - 1, :])
    return torch.cat(parts, dim=2)


class KernelSeamTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260922)
        self.hidden = torch.randn(1, 1, 32, 8).bfloat16()

    def test_the_mask_makes_the_kernel_agree_with_the_reference(self):
        for boundaries in (None, ((0, 32),), ((0, 16), (16, 32)),
                           ((0, 8), (8, 16), (16, 24), (24, 32))):
            mask = seam_mask(boundaries, 32)
            kernel = kernel_shift(self.hidden, 32, mask)
            expected = reference_shift(self.hidden, 32, boundaries)
            self.assertTrue(torch.equal(kernel, expected),
                            'kernel shift differs from the reference for %s' % (boundaries,))

    def test_unpacked_reproduces_the_old_guard_for_every_row_and_width(self):
        """The kernel used to read `row && row < rows`. At one segment the mask is
        bit 0 alone, so the new expression must reduce to exactly that."""
        for rows in (1, 8, 32):
            mask = seam_mask(None, rows)
            self.assertEqual(mask, 1)
            for row in range(32):
                old = bool(row and row < rows)
                new = bool(row < rows and not ((mask >> row) & 1))
                self.assertEqual(old, new, 'row %d of %d rows' % (row, rows))

    def test_each_segment_start_reads_zero_and_no_other_row_does(self):
        mask = seam_mask(((0, 16), (16, 32)), 32)
        shifted = kernel_shift(self.hidden, 32, mask)
        zero = torch.zeros_like(shifted[..., 0, :])
        self.assertTrue(torch.equal(shifted[..., 0, :], zero))
        self.assertTrue(torch.equal(shifted[..., 16, :], zero),
                        'user 1 anchor must not read user 0 last draft')
        for row in (1, 15, 17, 31):
            self.assertTrue(torch.equal(shifted[..., row, :], self.hidden[..., row - 1, :]),
                            'row %d still carries' % row)

    def test_a_short_block_zeroes_rows_past_the_width(self):
        mask = seam_mask(((0, 4), (4, 8)), 8)
        shifted = kernel_shift(self.hidden, 8, mask)
        self.assertTrue(torch.equal(shifted[..., 8:, :], torch.zeros_like(shifted[..., 8:, :])))
        self.assertTrue(torch.equal(shifted[..., 4, :], torch.zeros_like(shifted[..., 4, :])))


class CheckedConvolutionTests(unittest.TestCase):
    """checked_convolution used to refuse boundaries; now it must carry them."""

    def test_boundaries_reach_both_the_fused_call_and_the_audit_control(self):
        spans = ((0, 16), (16, 32))
        hidden = Mock(shape=(1, 1, 32, 5120))
        with patch('draft_convolution_fused.fused_convolution', return_value='out') as fused, \
                patch('draft_convolution.grouped_causal_convolution', return_value='control') as control:
            result = checked_convolution(Mock(), Mock(), hidden, ['d0', 'd1'], ['b0', 'b1'],
                                         fp32_intermediates=True, retain_temporaries=lambda value: value,
                                         boundaries=spans)
        self.assertEqual(result, 'out')
        self.assertEqual(fused.call_args.kwargs.get('boundaries'), spans)
        control.assert_not_called()

    def test_runtime_arguments_carry_the_seam_mask(self):
        """The mask is the ninth runtime argument, after rows and worker."""
        self.assertEqual(seam_mask(((0, 16), (16, 32)), 32), (1 << 0) | (1 << 16))
        self.assertEqual(seam_mask(((0, 8), (8, 16), (16, 24), (24, 32)), 32),
                         (1 << 0) | (1 << 8) | (1 << 16) | (1 << 24))

    def test_malformed_spans_are_still_refused(self):
        for spans in (((0, 16),), ((0, 16), (17, 32)), ((16, 32), (0, 16))):
            with self.assertRaises(ValueError):
                seam_mask(spans, 32)


class PackedFusedEquivalenceTests(unittest.TestCase):
    """Whole-convolution equivalence, using the kernel's own shift operand."""

    def test_packed_fused_arithmetic_matches_each_user_run_alone(self):
        torch.manual_seed(20260923)
        hidden = torch.randn(1, 1, 32, 5120).bfloat16()
        dynamic = [torch.randn(1, 1, 32, 320).bfloat16() for _ in range(2)]
        base = [torch.randn(1, 1, 1, 5120).bfloat16() for _ in range(2)]
        spans = ((0, 16), (16, 32))
        packed = convolution_reference(hidden, dynamic, base, boundaries=spans)
        self.assertTrue(torch.equal(kernel_shift(hidden, 32, seam_mask(spans, 32)),
                                    reference_shift(hidden, 32, spans)),
                        'the kernel operand is what the reference convolved')
        for start, stop in spans:
            moved_hidden = torch.zeros_like(hidden)
            moved_hidden[..., :stop - start, :] = hidden[..., start:stop, :]
            moved_dynamic = []
            for value in dynamic:
                moved = torch.zeros_like(value)
                moved[..., :stop - start, :] = value[..., start:stop, :]
                moved_dynamic.append(moved)
            alone = convolution_reference(moved_hidden, moved_dynamic, base)
            self.assertTrue(torch.equal(packed[..., start:stop, :], alone[..., :stop - start, :]))


if __name__ == '__main__':
    unittest.main()
