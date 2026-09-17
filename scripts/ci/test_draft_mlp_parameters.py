from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import torch

from draft_mlp_branch import execute_mlp_branch, prepare_mlp_branch


class DraftMLPParameterTests(unittest.TestCase):
    def test_trace_gate_rejects_simulator_latency(self):
        environment = {name: value for name, value in os.environ.items()
            if name not in ('TT_METAL_SLOW_DISPATCH_MODE', 'TT_METAL_MOCK_CLUSTER_DESC_PATH')}
        environment['TT_METAL_SIMULATOR'] = 'simulator-placeholder'
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('draft-mlp-trace-probe.py')),
            '--fixture', '/missing', '--convolution-fixture', '/missing', '--output', '/missing', '--timing'],
            env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Latency measurements require allocated hardware', result.stderr)

    def test_simulator_timing_is_rejected_before_fixture_loading(self):
        environment = {name: value for name, value in os.environ.items()
            if name not in ('TT_METAL_SLOW_DISPATCH_MODE', 'TT_METAL_MOCK_CLUSTER_DESC_PATH')}
        environment['TT_METAL_SIMULATOR'] = 'simulator-placeholder'
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('draft-mlp-replay-probe.py')),
            '--fixture', '/missing', '--convolution-fixture', '/missing', '--output', '/missing', '--timing'],
            env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Latency measurements require allocated hardware', result.stderr)

    def fixture(self):
        value = SimpleNamespace(shape=(1, 1, 32, 5120), dtype='bf16')
        operations = SimpleNamespace(bfloat16='bf16', float32='fp32', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='row',
            DRAM_MEMORY_CONFIG='dram', MathFidelity=SimpleNamespace(HiFi4='hifi4'))
        for name in ('from_torch', 'ShardTensorToMesh', 'ReplicateTensorToMesh', 'WormholeComputeKernelConfig',
                'MatmulMultiCoreReuseMultiCast1DProgramConfig', 'matmul', 'rms_norm', 'typecast', 'slice', 'add'):
            setattr(operations, name, Mock(side_effect=lambda *args, **kwargs: SimpleNamespace()))
        weights = {f'layers.0.mlp.{name}_proj.weight': object() for name in ('gate', 'up', 'down')}
        convolution = {'layers.0.post_attention_layernorm.weight': torch.ones(5120),
            'layers.0.mlp_conv.kernel_projection.weight': torch.ones(2, 2),
            'layers.0.mlp_conv.base_kernel': torch.ones(2, 2, 5120)}
        return operations, value, weights, convolution

    def test_repeated_blocks_do_not_reupload_or_repack_parameters(self):
        operations, hidden, weights, convolution = self.fixture()
        mesh, collective = object(), object()
        parameters_owned, temporary = [], []
        def retain_parameter(value):
            parameters_owned.append(value)
            return value
        def retain_temporary(value):
            temporary.append(value)
            return value
        shards = tuple(tuple(torch.ones(2, 2) for _ in range(3)) for _ in range(2))
        with patch('draft_mlp_branch.split_mlp_weights', return_value=shards) as split, \
                patch('draft_mlp_branch.grouped_causal_convolution', return_value=object()), \
                patch('draft_mlp_branch.gather_add_projection', return_value=object()), \
                patch('draft_mlp_branch.swiglu_device', return_value=object()):
            parameters = prepare_mlp_branch(operations, mesh, weights, convolution, retain_parameter)
            self.assertEqual(operations.from_torch.call_count, 9)
            self.assertEqual(len(parameters_owned), 9)
            for _ in range(2):
                result = execute_mlp_branch(operations, mesh, collective, hidden, weights, convolution,
                    retain_temporary, parameters=parameters)
                self.assertIs(result['hidden'], hidden)
                self.assertIs(result['shards'], shards)
            split.assert_called_once()
            self.assertEqual(operations.from_torch.call_count, 9)
            self.assertEqual(operations.matmul.call_count, 8)
            expected = [parameters['device_conv'], *parameters['device_projections']] * 2
            for call, weight in zip(operations.matmul.call_args_list, expected, strict=True):
                self.assertIs(call.args[1], weight)
            self.assertTrue(all(all(value is not owned for owned in parameters_owned) for value in temporary))
            for other_mesh, other_weights, other_convolution in (
                    (object(), weights, convolution), (mesh, dict(weights), convolution), (mesh, weights, dict(convolution))):
                with self.assertRaisesRegex(ValueError, 'different mesh or learned layer'):
                    execute_mlp_branch(operations, other_mesh, collective, hidden, other_weights, other_convolution,
                        retain_temporary, parameters=parameters)


if __name__ == '__main__':
    unittest.main()
