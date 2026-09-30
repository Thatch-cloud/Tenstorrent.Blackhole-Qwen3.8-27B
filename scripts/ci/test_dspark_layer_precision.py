from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dspark_layer_precision import DIMENSIONS, execute


class LayerPrecisionTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(MathFidelity=SimpleNamespace(HiFi4='hifi4', HiFi2='hifi2'),
            float32='fp32', WormholeComputeKernelConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            matmul=Mock(return_value=object()), sum=Mock(return_value=object()))
        config = SimpleNamespace(math_fidelity='hifi4', math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=False)
        return operations, config

    def test_all_seven_projections_and_unchanged_nonmatmul_operations(self):
        operations, config = self.fixture()
        output = object()
        def backend(selected):
            self.assertIs(selected.sum, operations.sum)
            for inputs, outputs in DIMENSIONS:
                selected.matmul(SimpleNamespace(shape=(1, 1, 32, inputs)),
                    SimpleNamespace(shape=(1, 1, inputs, outputs)), dtype='fp32', compute_kernel_config=config)
            return output
        self.assertIs(execute(backend, operations), output)
        self.assertEqual(operations.matmul.call_count, 7)
        self.assertTrue(all(call.kwargs['compute_kernel_config'].math_fidelity == 'hifi2'
            for call in operations.matmul.call_args_list))
        self.assertEqual(config.math_fidelity, 'hifi4')

    def test_missing_out_of_order_and_target_width_rejected(self):
        operations, config = self.fixture()
        with self.assertRaises(ValueError):
            execute(lambda selected: object(), operations)
        for rows, width in ((16, 2048), (32, 8704)):
            def backend(selected):
                return selected.matmul(SimpleNamespace(shape=(1, 1, rows, 5120)),
                    SimpleNamespace(shape=(1, 1, 5120, width)), dtype='fp32', compute_kernel_config=config)
            with self.assertRaises(ValueError):
                execute(backend, operations)
        operations.matmul.assert_not_called()
