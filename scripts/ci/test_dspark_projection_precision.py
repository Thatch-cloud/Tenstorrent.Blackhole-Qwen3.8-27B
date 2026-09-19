from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dspark_projection_precision import ProjectionOperations, linear


class ProjectionPrecisionTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(bfloat16='bf16', float32='fp32', TILE_LAYOUT='tile',
            DRAM_MEMORY_CONFIG='dram', MathFidelity=SimpleNamespace(HiFi4='hifi4', HiFi2='hifi2'),
            WormholeComputeKernelConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            MatmulMultiCoreReuseMultiCast1DProgramConfig=lambda **kwargs: kwargs,
            matmul=Mock(return_value=object()), typecast=Mock(return_value=object()))
        def tensor(shape):
            return SimpleNamespace(shape=shape, dtype='bf16', layout='tile', memory_config=lambda: 'dram')
        return operations, tensor((1, 1, 32, 5120)), tensor((1, 1, 5120, 8704))

    def test_only_matmul_fidelity_changes_and_rounding_is_preserved(self):
        for rounded in (False, True):
            operations, value, weight = self.fixture()
            retained = []
            def retain(tensor):
                retained.append(tensor)
                return tensor
            result = linear(operations, value, weight, retain, rounded=rounded)
            options = operations.matmul.call_args.kwargs
            self.assertEqual(vars(options['compute_kernel_config']), dict(math_fidelity='hifi2',
                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
            self.assertEqual(options['dtype'], 'fp32')
            self.assertEqual(options['memory_config'], 'dram')
            self.assertEqual(options['program_config']['in0_block_w'], 8)
            self.assertEqual(len(retained), 2 if rounded else 1)
            self.assertIs(result, retained[-1])
            self.assertEqual(operations.typecast.call_count, int(rounded))
            self.assertEqual(operations.MathFidelity.HiFi4, 'hifi4')

    def test_wrong_reference_policy_and_target_shapes_rejected(self):
        operations, value, weight = self.fixture()
        proxy = ProjectionOperations(operations)
        with self.assertRaises(ValueError):
            proxy.matmul(value, weight, dtype='fp32', compute_kernel_config=SimpleNamespace())
        value.shape = (1, 1, 16, 5120)
        with self.assertRaises(ValueError):
            linear(operations, value, weight, lambda tensor: tensor)
        operations.matmul.assert_not_called()

    def test_candidate_failure_does_not_modify_shared_operations(self):
        operations, value, weight = self.fixture()
        original = operations.matmul
        original.side_effect = RuntimeError('candidate failed')
        with self.assertRaisesRegex(RuntimeError, 'candidate failed'):
            linear(operations, value, weight, lambda tensor: tensor)
        self.assertIs(operations.matmul, original)
        self.assertEqual(operations.MathFidelity.HiFi4, 'hifi4')
