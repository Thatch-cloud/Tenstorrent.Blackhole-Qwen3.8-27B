import unittest
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

from draft_mlp import split_mlp_weights, swiglu_reference, swiglu_device


class DraftMlpTests(unittest.TestCase):
    def test_integrated_hardware_requires_allocation_before_fixture_load(self):
        environment = {name: value for name, value in os.environ.items()
            if name not in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH', 'TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='0')
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('learned-mlp-probe.py')),
            '--hardware', '--fixture', '/not-a-fixture', '--convolution-fixture', '/not-a-fixture',
            '--output', '/not-an-output'], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Explicit allocation and non-simulated hardware required', result.stderr)

    def test_device_sequence_preserves_reference_rounding(self):
        operations = SimpleNamespace(bfloat16=torch.bfloat16, float32=torch.float32, DRAM_MEMORY_CONFIG=object(),
            typecast=lambda value, dtype: value.to(dtype),
            silu=lambda value, **kwargs: torch.nn.functional.silu(value),
            multiply=lambda left, right, **kwargs: left * right)
        gate = torch.tensor([[-3.123, .123, 2.456]])
        up = torch.tensor([[.156, 3.141, -2.718]])
        retained = []

        def retain(value):
            retained.append(value)
            return value

        self.assertTrue(torch.equal(swiglu_device(operations, gate, up, retain), swiglu_reference(gate, up)))
        self.assertEqual(len(retained), 9)

    def test_hidden_and_intermediate_axes_are_not_interchanged(self):
        gate = torch.arange(32).reshape(8, 4).bfloat16()
        up = (gate + 2).bfloat16()
        down = torch.arange(32).reshape(4, 8).bfloat16()
        ranks = split_mlp_weights(gate, up, down)
        self.assertTrue(torch.equal(torch.cat([rank[0].T for rank in ranks]), gate))
        self.assertTrue(torch.equal(torch.cat([rank[1].T for rank in ranks]), up))
        self.assertTrue(torch.equal(torch.cat([rank[2].T for rank in ranks], dim=1), down))
        hidden = torch.ones(2, 4)
        partial = [hidden @ rank[0].float() @ rank[2].float() for rank in ranks]
        self.assertTrue(torch.equal(partial[0] + partial[1], hidden @ gate.float().T @ down.float().T))

    def test_swiglu_rounding_is_explicit(self):
        gate = torch.tensor([[-3.123, .123, 2.456]])
        up = torch.tensor([[.156, 3.141, -2.718]])
        expected = torch.nn.functional.silu(gate.bfloat16().float()).bfloat16() * up.bfloat16()
        self.assertTrue(torch.equal(swiglu_reference(gate, up), expected))

    def test_invalid_geometry_and_dtype_fail_closed(self):
        gate = torch.zeros(8, 4, dtype=torch.bfloat16)
        for up, down in ((gate[:7], gate.T), (gate, gate), (gate.float(), gate.T)):
            with self.assertRaises(ValueError):
                split_mlp_weights(gate, up, down)
