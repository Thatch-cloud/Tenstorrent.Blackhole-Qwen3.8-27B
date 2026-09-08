from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from draft_kv_projection import project_key_value


class DraftKVProjectionTests(unittest.TestCase):
    def fixture(self, rows):
        operations = SimpleNamespace(bfloat16='bf16', float32='fp32', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram')
        for name in ('matmul', 'typecast', 'rms_norm', 'MatmulMultiCoreReuseMultiCast1DProgramConfig'):
            setattr(operations, name, Mock(side_effect=lambda *args, **kwargs: object()))
        operations.experimental = SimpleNamespace(rotary_embedding_hf=Mock(return_value=object()))
        def tensor(shape):
            return SimpleNamespace(shape=shape, dtype='bf16', layout='tile', memory_config=lambda: 'dram')
        inputs, query = tensor((1, 1, rows, 5120)), tensor((1, 1, 32, 2048))
        rope = tuple(tensor((1, 1, rows, 128)) for unused in range(2))
        parameters = dict(operations=operations, native_head_layout=True, kernel=object(),
            projections=dict(k=object(), v=object()), head_norms=dict(k=object()))
        return operations, inputs, query, rope, parameters

    def test_full_and_incremental_rows_preserve_projection_precision_and_head_math(self):
        for rows in (32, 256, 288, 2048, 2080):
            operations, inputs, query, rope, parameters = self.fixture(rows)
            owned = []
            def retain(value):
                owned.append(value)
                return value
            heads = dict(q=object(), k=object(), v=object())
            with patch('draft_kv_projection.split_projected_heads', return_value=heads) as split:
                output = project_key_value(operations, inputs, query, rope, retain, parameters=parameters)
            self.assertIs(output['v'], heads['v'])
            self.assertIs(split.call_args.args[1], query)
            self.assertIs(split.call_args.args[-1], retain)
            program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig.call_args.kwargs
            self.assertEqual(program, dict(compute_with_storage_grid_size=(8, 8), in0_block_w=4,
                out_subblock_h=1, out_subblock_w=1, per_core_M=rows // 32, per_core_N=1,
                fuse_batch=True, fused_activation=None, mcast_in0=True))
            self.assertEqual(operations.matmul.call_count, 2)
            for name, call in zip(('k', 'v'), operations.matmul.call_args_list, strict=True):
                self.assertIs(call.args[0], inputs)
                self.assertIs(call.args[1], parameters['projections'][name])
                self.assertIs(call.kwargs['compute_kernel_config'], parameters['kernel'])
                self.assertEqual(call.kwargs['dtype'], 'fp32')
            operations.rms_norm.assert_called_once_with(heads['k'], epsilon=1e-6,
                weight=parameters['head_norms']['k'], compute_kernel_config=parameters['kernel'], memory_config='dram')
            self.assertFalse(operations.experimental.rotary_embedding_hf.call_args.kwargs['is_decode_mode'])
            self.assertEqual(len(owned), 10)

    def test_invalid_projection_geometry_rejected_before_device_operations(self):
        for rows in (0, 7, 31, 33, 2112):
            operations, inputs, query, rope, parameters = self.fixture(rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                project_key_value(operations, inputs, query, rope, lambda value: value, parameters=parameters)
            operations.matmul.assert_not_called()

    def test_rotary_tables_layout_and_parameter_ownership_are_explicit(self):
        for mutation in ('rope', 'dtype', 'layout', 'native', 'runtime', 'query'):
            operations, inputs, query, rope, parameters = self.fixture(32)
            if mutation == 'rope':
                rope = rope[:1]
            elif mutation == 'dtype':
                rope[0].dtype = 'fp32'
            elif mutation == 'layout':
                inputs.layout = 'row-major'
            elif mutation == 'native':
                parameters['native_head_layout'] = False
            elif mutation == 'runtime':
                parameters['operations'] = object()
            else:
                query.shape = (1, 1, 32, 512)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                project_key_value(operations, inputs, query, rope, lambda value: value, parameters=parameters)
            operations.matmul.assert_not_called()

    def test_probe_cannot_open_hardware(self):
        environment = {name: value for name, value in os.environ.items()
            if name not in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH')}
        environment.update(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('draft-kv-projection-probe.py')),
            '--fixture', '/missing', '--output', '/missing'], env=environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Simulator', result.stderr)


if __name__ == '__main__':
    unittest.main()
