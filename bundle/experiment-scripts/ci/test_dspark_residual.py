import unittest

from dspark_residual import add
from test_dspark_projection import operations, tensor


class DSparkResidualTests(unittest.TestCase):
    def test_native_fp32_output_is_rounded_explicitly_to_bf16(self):
        runtime = operations()
        first,second = tensor((1,1,32,5120)),tensor((1,1,32,5120))
        owned = []
        def retain(value):
            owned.append(value)
            return value
        result = add(runtime,first,second,retain)
        self.assertEqual(runtime.add.call_args.args,(first,second))
        self.assertEqual(runtime.add.call_args.kwargs['dtype'],'fp32')
        self.assertEqual(runtime.typecast.call_args.args,(runtime.add.return_value,'bf16'))
        self.assertIs(result,runtime.typecast.return_value)
        self.assertEqual(owned,[runtime.add.return_value,runtime.typecast.return_value])

    def test_incomplete_or_wide_inputs_reject_before_dispatch(self):
        runtime = operations()
        for first in (tensor((1,1,7,5120)),tensor((1,1,32,5120),'fp32')):
            with self.assertRaises(ValueError):
                add(runtime,first,tensor((1,1,32,5120)),lambda value:value)
        runtime.add.assert_not_called()

    def test_widening_both_inputs_selects_fp32_arithmetic_not_just_output_format(self):
        runtime = operations()
        first,second = tensor((1,1,32,5120)),tensor((1,1,32,5120))
        first_wide,second_wide,output = [tensor((1,1,32,5120),'fp32') for unused in range(3)]
        runtime.typecast.side_effect = [first_wide,second_wide,output]
        result = add(runtime,first,second,lambda value:value,widen_inputs=True)
        self.assertEqual(runtime.typecast.call_args_list[0].args,(first,'fp32'))
        self.assertEqual(runtime.typecast.call_args_list[1].args,(second,'fp32'))
        self.assertEqual(runtime.add.call_args.args,(first_wide,second_wide))
        self.assertEqual(runtime.typecast.call_args_list[2].args,(runtime.add.return_value,'bf16'))
        self.assertIs(result,output)


if __name__ == '__main__':
    unittest.main()
