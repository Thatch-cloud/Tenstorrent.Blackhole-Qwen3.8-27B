import unittest

import torch

from draft_mlp import split_mlp_weights, swiglu_reference


class DraftMlpTests(unittest.TestCase):
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
