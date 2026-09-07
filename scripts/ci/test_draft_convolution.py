import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from draft_convolution import convolution_reference, grouped_causal_convolution


class DraftConvolutionTests(unittest.TestCase):
    def test_trace_ownership_defers_sync_and_release(self):
        hidden = SimpleNamespace(shape=(1, 1, 8, 5120), dtype='bf16')
        dynamic = [SimpleNamespace(shape=(1, 1, 8, 320), dtype='bf16') for _ in range(2)]
        bases = [SimpleNamespace(shape=(1, 1, 1, 5120), dtype='bf16') for _ in range(2)]
        operations = SimpleNamespace(bfloat16='bf16', float32='fp32', DRAM_MEMORY_CONFIG='dram',
            synchronize_device=Mock(), deallocate=Mock())
        for name in ('zeros_like', 'slice', 'concat', 'repeat_interleave', 'typecast', 'multiply', 'add'):
            setattr(operations, name, Mock(side_effect=lambda *args, **kwargs: object()))
        owned = []
        with patch('draft_convolution.addresses', side_effect=lambda operations, value: id(value)), \
                patch('draft_convolution.release_owned') as release:
            output = grouped_causal_convolution(operations, SimpleNamespace(shape=[1, 2]), hidden,
                dynamic, bases, fp32_intermediates=True, retain_temporaries=owned.append)
            self.assertIn(output, owned)
            self.assertEqual(len(owned), 38)
            self.assertTrue(all(value is not hidden for value in owned))
            release.assert_not_called()
        operations.synchronize_device.assert_not_called()
        operations.deallocate.assert_not_called()

    def test_causal_shift_and_group_boundary(self):
        hidden = torch.zeros((1, 1, 8, 5120), dtype=torch.bfloat16)
        hidden[..., 0, :] = 1
        dynamic = [torch.zeros((1, 1, 8, 320), dtype=torch.bfloat16) for offset in range(2)]
        base = [torch.zeros((1, 1, 1, 5120), dtype=torch.bfloat16) for offset in range(2)]
        dynamic[1][..., 1, 1] = 2
        actual = convolution_reference(hidden, dynamic, base)
        expected = torch.zeros_like(hidden)
        expected[..., 1, 16:32] = 2
        self.assertTrue(torch.equal(actual, expected))

    def test_one_row_does_not_wrap_last_value_into_first(self):
        hidden = torch.ones((1, 1, 1, 5120), dtype=torch.bfloat16)
        dynamic = [torch.ones((1, 1, 1, 320), dtype=torch.bfloat16) for offset in range(2)]
        base = [torch.ones_like(hidden) for offset in range(2)]
        self.assertTrue(torch.equal(convolution_reference(hidden, dynamic, base), hidden * 2))

    def test_rejects_wrong_group_count_and_nonfinite(self):
        hidden = torch.ones((1, 1, 8, 5120), dtype=torch.bfloat16)
        dynamic = [torch.ones((1, 1, 8, 320), dtype=torch.bfloat16) for offset in range(2)]
        base = [torch.ones((1, 1, 1, 5120), dtype=torch.bfloat16) for offset in range(2)]
        with self.assertRaises(ValueError):
            convolution_reference(hidden, dynamic[:1], base)
        hidden[..., 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            convolution_reference(hidden, dynamic, base)
