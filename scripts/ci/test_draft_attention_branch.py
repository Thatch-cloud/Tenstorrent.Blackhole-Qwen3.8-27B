import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from draft_attention_branch import prepare_attention_branch, execute_attention_branch


class DraftAttentionBranchTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(bfloat16='bf16', float32='fp32', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='row',
            DRAM_MEMORY_CONFIG='dram', MathFidelity=SimpleNamespace(HiFi4='hifi4'))
        for name in ('from_torch', 'ShardTensorToMesh', 'ReplicateTensorToMesh', 'WormholeComputeKernelConfig',
                'MatmulMultiCoreReuseMultiCast1DProgramConfig', 'matmul', 'rms_norm', 'typecast', 'slice', 'add',
                'zeros_like', 'concat', 'reshape', 'transpose'):
            setattr(operations, name, Mock(side_effect=lambda *args, **kwargs: object()))
        operations.experimental = SimpleNamespace(rotary_embedding_hf=Mock(return_value=object()))
        weights = {f'layers.0.self_attn.{name}_proj.weight': torch.ones(2, 2) for name in ('q', 'k', 'v', 'o')}
        weights.update({f'layers.0.self_attn.{name}_norm.weight': torch.ones(128) for name in ('q', 'k')})
        convolution = {'layers.0.input_layernorm.weight': torch.ones(5120),
            'layers.0.attention_conv.kernel_projection.weight': torch.ones(2, 2),
            'layers.0.attention_conv.base_kernel': torch.ones(2, 2, 5120)}
        hidden = SimpleNamespace(shape=(1, 1, 32, 5120), dtype='bf16')
        history = SimpleNamespace(shape=(1, 1, 64, 5120), dtype='bf16')
        mask = SimpleNamespace(shape=(1, 1, 32, 64), dtype='bf16')
        rope = {name: tuple(SimpleNamespace(shape=(1, 1, rows, 128), dtype='bf16') for _ in range(2))
            for name, rows in (('q', 32), ('k', 64))}
        return operations, weights, convolution, hidden, history, mask, rope

    def test_parameters_are_prepared_once_and_execution_retains_temporaries(self):
        operations, weights, convolution, hidden, history, mask, rope = self.fixture()
        mesh, collective = object(), object()
        persistent, transient = [], []
        def retain_parameter(value):
            persistent.append(value)
            return value
        def retain_temporary(value):
            transient.append(value)
            return value
        with patch('draft_attention_branch.grouped_causal_convolution', return_value=object()) as conv, \
                patch('draft_attention_branch.gather_add_projection', return_value=object()) as gather, \
                patch('draft_attention_branch.composed_draft_attention', return_value=object()) as attention:
            parameters = prepare_attention_branch(operations, mesh, weights, convolution, retain_parameter)
            self.assertEqual(len(persistent), 12)
            for _ in range(2):
                execute_attention_branch(operations, mesh, collective, hidden, history, mask, rope,
                    retain_temporary, parameters=parameters, context=31)
            self.assertEqual(operations.from_torch.call_count, 12)
            self.assertEqual(operations.matmul.call_count, 10)
            for call in (*conv.call_args_list, *gather.call_args_list):
                self.assertIs(call.kwargs['retain_temporaries'], retain_temporary)
            for call in attention.call_args_list:
                self.assertIsInstance(call.kwargs['trace_owned'], list)
                self.assertIs(call.kwargs['cache_dot_tiles'], True)
            self.assertTrue(all(all(value is not owned for owned in persistent) for value in transient))

    def test_invalid_context_tables_and_mesh_fail_before_device_operations(self):
        operations, weights, convolution, hidden, history, mask, rope = self.fixture()
        mesh = object()
        parameters = dict(operations=operations, mesh=mesh)
        for context, tables, target_mesh in ((0, rope, mesh), (True, rope, mesh), (2049, rope, mesh),
                (31, {}, mesh), (31, rope, object()), (64, rope, mesh)):
            with self.assertRaises(ValueError):
                execute_attention_branch(operations, target_mesh, object(), hidden, history, mask, tables,
                    lambda value: value, parameters=parameters, context=context)
        operations.matmul.assert_not_called()

    def test_native_branch_replaces_composition_and_retains_output(self):
        operations, weights, convolution, hidden, history, mask, rope = self.fixture()
        mesh, collective, native_output = object(), object(), object()
        transient = []
        def retain(value):
            transient.append(value)
            return value
        with patch('draft_attention_branch.grouped_causal_convolution', return_value=object()), \
                patch('draft_attention_branch.gather_add_projection', return_value=object()), \
                patch('draft_attention_branch.composed_draft_attention') as composed, \
                patch('draft_attention_branch.draft_sdpa', return_value=native_output) as native, \
                patch('native_draft_sdpa.audit_active_kernel', return_value={'audited': True}) as audit, \
                patch.dict(os.environ, TT_METAL_HOME='/audited'):
            parameters = prepare_attention_branch(operations, mesh, weights, convolution, lambda value: value,
                precise_native=True)
            audit.assert_called_once_with('/audited')
            execute_attention_branch(operations, mesh, collective, hidden, history, mask, rope,
                retain, parameters=parameters, context=31)
            native.assert_called_once()
            composed.assert_not_called()
            self.assertTrue(any(value is native_output for value in transient))
            with self.assertRaisesRegex(ValueError, 'placement'):
                execute_attention_branch(operations, mesh, collective, hidden, history, mask, rope,
                    retain, parameters=parameters, context=31, wide_dot_placement=True)

    def test_captured_stack_cannot_be_promoted_directly_to_hardware(self):
        environment = {name: value for name, value in os.environ.items()
            if name not in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH', 'TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('learned-attention-probe.py')),
            '--hardware', '--captured-stack', '--fixture', '/missing', '--output', '/missing'],
            env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Captured stack requires', result.stderr)
