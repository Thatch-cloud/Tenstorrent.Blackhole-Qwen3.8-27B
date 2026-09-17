from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from dspark_norm_precision import normalize
from test_dspark_projection import operations, tensor


class DSparkNormPrecisionTests(unittest.TestCase):
    def test_preserves_wide_native_output_until_explicit_bf16_rounding(self):
        runtime = operations()
        narrowed,gamma = tensor((1,1,32,5120)),tensor((1,1,1,5120))
        widened,rounded = SimpleNamespace(),SimpleNamespace()
        runtime.typecast.side_effect = [widened,rounded]
        runtime.rms_norm.return_value = tensor((1,1,32,5120),'fp32')
        owned = []
        def retain(value):
            owned.append(value)
            return value
        result = normalize(runtime,narrowed,gamma,retain)
        self.assertEqual(runtime.typecast.call_args_list[0].args,(narrowed,'fp32'))
        self.assertEqual(runtime.typecast.call_args_list[1].args,(runtime.rms_norm.return_value,'bf16'))
        self.assertEqual(runtime.rms_norm.call_args.args,(widened,))
        self.assertIsNone(runtime.rms_norm.call_args.kwargs['weight'])
        self.assertEqual(runtime.mul.call_args.args,(rounded,gamma))
        self.assertEqual(len(owned),4)
        self.assertIs(result['unweighted_norm'],rounded)

    def test_rejects_wrong_native_output_precision_and_input_shape(self):
        runtime = operations()
        runtime.rms_norm.return_value = tensor((1,1,32,5120),'bf16')
        with self.assertRaises(ValueError):
            normalize(runtime,tensor((1,1,32,5120)),tensor((1,1,1,5120)),lambda value:value)
        runtime.mul.assert_not_called()
        runtime = operations()
        with self.assertRaises(ValueError):
            normalize(runtime,tensor((1,1,32,2560)),tensor((1,1,1,5120)),lambda value:value)
        runtime.rms_norm.assert_not_called()

    def test_composed_path_selects_fp32_reduction_and_accurate_products(self):
        runtime = operations()
        runtime.multiply = MagicMock(return_value=tensor((1,1,32,5120),'fp32'))
        runtime.sum,runtime.rsqrt = MagicMock(),MagicMock()
        normalize(runtime,tensor((1,1,32,5120)),tensor((1,1,1,5120)),lambda value:value,composed=True)
        runtime.rms_norm.assert_not_called()
        self.assertEqual(runtime.sum.call_args.kwargs['dim'],3)
        self.assertTrue(runtime.sum.call_args.kwargs['keepdim'])
        self.assertNotIn('fast_and_approximate_mode',runtime.sum.call_args.kwargs)
        self.assertIn('compute_kernel_config',runtime.sum.call_args.kwargs)
        self.assertEqual(runtime.multiply.call_count,3)
        self.assertTrue(all(call.kwargs['fast_and_approximate_mode'] is False for call in runtime.multiply.call_args_list))
        self.assertEqual(runtime.multiply.call_args_list[1].args[1],1/5120)
        self.assertEqual(runtime.add.call_args.args[1],1e-6)
        self.assertFalse(runtime.rsqrt.call_args.kwargs['fast_and_approximate_mode'])


if __name__ == '__main__':
    unittest.main()
