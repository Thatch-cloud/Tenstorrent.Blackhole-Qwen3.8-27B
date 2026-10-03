from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dspark_approx_native_attention import execute


class ApproxNativeAttentionTests(unittest.TestCase):
    def test_only_exponential_policy_changes_and_output_is_owned(self):
        output = object()
        operations = SimpleNamespace(WormholeComputeKernelConfig=Mock(),
            SDPAProgramConfig=Mock(), MathFidelity=SimpleNamespace(HiFi4='hifi4'),
            DRAM_MEMORY_CONFIG='dram', transformer=SimpleNamespace(
                scaled_dot_product_attention=Mock(return_value=output)))
        operands = [object(), SimpleNamespace(shape=(1, 4, 4416, 128)), object(), object()]
        owned = []
        with patch('dspark_approx_native_attention.validate_inputs') as validate:
            self.assertIs(execute(operations, object(), *operands, owned,
                context_rows=4384, proposals=15, mask_validated=True), output)
        validate.assert_called_once_with(operations, *operands, 4384, 15, True)
        operations.WormholeComputeKernelConfig.assert_called_once_with(
            math_fidelity='hifi4', math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=False)
        operations.SDPAProgramConfig.assert_called_once_with(
            compute_with_storage_grid_size=(8, 8), q_chunk_size=32,
            k_chunk_size=64, exp_approx_mode=True)
        self.assertEqual(owned, [output])

    def test_invalid_inputs_do_not_dispatch(self):
        operations = Mock()
        with patch('dspark_approx_native_attention.validate_inputs', side_effect=ValueError('layout')):
            with self.assertRaisesRegex(ValueError, 'layout'):
                execute(operations, object(), *[object() for index in range(4)], [],
                    context_rows=4384, proposals=15)
        operations.transformer.scaled_dot_product_attention.assert_not_called()
