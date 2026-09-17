from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dspark_ladder_attention import adapter
from dspark_ladder_geometry import ladder


class LadderAttentionTests(unittest.TestCase):
    def test_every_context_preserves_inputs_and_masks_new_padding(self):
        for fixture in ladder():
            with self.subTest(context=fixture['context']), patch('dspark_ladder_attention.validate_inputs') as validate:
                operations = SimpleNamespace(pad=Mock(side_effect=lambda *args: object()),
                    WormholeComputeKernelConfig=Mock(), SDPAProgramConfig=Mock(),
                    MathFidelity=SimpleNamespace(HiFi4='hifi4'), DRAM_MEMORY_CONFIG='dram',
                    transformer=SimpleNamespace(scaled_dot_product_attention=Mock(return_value='output')))
                key = SimpleNamespace(shape=(1, 4, fixture['storage_keys'], 128))
                query, value, mask, mesh, owned = object(), object(), object(), object(), []
                result = adapter(fixture['context'])(operations, mesh, query, key, value, mask, owned,
                    context_rows=fixture['capacity'], proposals=15, mask_validated=True)
                validate.assert_called_once_with(operations, query, key, value, mask,
                    fixture['capacity'], 15, True)
                calls = operations.pad.call_args_list
                self.assertEqual([call.args[2] for call in calls], [8192., -8192., float('-inf')])
                self.assertEqual(calls[0].args[1][2], (0, fixture['extra_masked_keys']))
                self.assertEqual(calls[2].args[1][3], (0, fixture['extra_masked_keys']))
                self.assertEqual(len(owned), 4)
                self.assertEqual(result, 'output')

    def test_wrong_capacity_rejected_before_tensor_validation(self):
        with patch('dspark_ladder_attention.validate_inputs') as validate:
            with self.assertRaises(ValueError):
                adapter(32768)(None, object(), None, None, None, None, [],
                    context_rows=8192, proposals=15)
            validate.assert_not_called()
