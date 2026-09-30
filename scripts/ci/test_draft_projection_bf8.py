"""QWEN_FAST_DRAFT_BF8: the draft's projection matrices upload as bfloat8_b, everything else
stays bfloat16, and the lend line reports the dtype the LENT tensors carry.

Projections: per learned layer the attention q/k/v and o and the MLP gate/up/down, plus the
fc feature projection - 5 x 7 + 1 = 36 tensors. Norms, convolution kernel projections and
bases, and the selector projection always stay bf16. Unset (the default) every upload is
bf16 and the lend line is exactly the old one."""

from contextlib import ExitStack
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import dflash_device
from dflash_device import PreparedDraftWeights
from draft_attention_branch import prepare_attention_branch
from draft_mlp_branch import DRAFT_BF8_FLAG, draft_projection_dtype, prepare_mlp_branch
from test_serving_buffer_pool import FakeOperations as PoolOperations

PROJECTIONS = 5 * 7 + 1


class FakeOperations(PoolOperations):
    bfloat8_b = 'bf8'


def attention_weights():
    weights = {'layers.0.self_attn.%s_proj.weight' % name: torch.zeros(8, 8) for name in ('q', 'k', 'v', 'o')}
    weights.update({'layers.0.self_attn.%s_norm.weight' % name: torch.zeros(128) for name in ('q', 'k')})
    return weights


def convolution_weights():
    return {'layers.0.input_layernorm.weight': torch.zeros(5120),
            'layers.0.attention_conv.kernel_projection.weight': torch.zeros(4, 4),
            'layers.0.attention_conv.base_kernel': torch.zeros(2, 2, 5120),
            'layers.0.post_attention_layernorm.weight': torch.zeros(5120),
            'layers.0.mlp_conv.kernel_projection.weight': torch.zeros(4, 4),
            'layers.0.mlp_conv.base_kernel': torch.zeros(2, 2, 5120)}


def mlp_weights():
    return {'layers.0.mlp.gate_proj.weight': torch.zeros(4, 6, dtype=torch.bfloat16),
            'layers.0.mlp.up_proj.weight': torch.zeros(4, 6, dtype=torch.bfloat16),
            'layers.0.mlp.down_proj.weight': torch.zeros(6, 4, dtype=torch.bfloat16)}


def environment(flag):
    clean = {name: value for name, value in os.environ.items() if name != DRAFT_BF8_FLAG}
    if flag is not None:
        clean[DRAFT_BF8_FLAG] = flag
    return patch.dict(os.environ, clean, clear=True)


class ProjectionDtypeTests(unittest.TestCase):
    def test_only_the_value_one_selects_bfloat8(self):
        operations = FakeOperations()
        for value, expected in ((None, 'bf16'), ('0', 'bf16'), ('true', 'bf16'), ('1', 'bf8')):
            with self.subTest(value=value):
                self.assertEqual(draft_projection_dtype(operations, {} if value is None else {DRAFT_BF8_FLAG: value}),
                                 expected)


class BranchUploadTests(unittest.TestCase):
    def mlp(self, flag):
        operations = FakeOperations()
        with environment(flag):
            prepared = prepare_mlp_branch(operations, 'mesh', mlp_weights(), convolution_weights(), lambda value: value)
        return prepared

    def attention(self, flag):
        operations = FakeOperations()
        with environment(flag):
            prepared = prepare_attention_branch(operations, 'mesh', attention_weights(), convolution_weights(),
                                                lambda value: value, native_head_layout=True, block_rows=16)
        return prepared

    def test_unset_every_mlp_upload_is_bfloat16(self):
        prepared = self.mlp(None)
        tensors = [prepared['device_norm'], prepared['device_conv'], *prepared['bases'], *prepared['device_projections']]
        self.assertEqual({tensor.dtype for tensor in tensors}, {'bf16'})

    def test_the_flag_moves_only_the_mlp_projections_to_bfloat8(self):
        prepared = self.mlp('1')
        self.assertEqual([tensor.dtype for tensor in prepared['device_projections']], ['bf8'] * 3)
        self.assertEqual([tensor.dtype for tensor in (prepared['device_norm'], prepared['device_conv'], *prepared['bases'])],
                         ['bf16'] * 6)
        # Sharded and tiled exactly as before; only the dtype moves.
        self.assertEqual({tensor.mapper for tensor in prepared['device_projections']}, {('shard', 'mesh', 0)})
        self.assertEqual({tensor.layout for tensor in prepared['device_projections']}, {'tile'})

    def test_unset_every_attention_upload_is_bfloat16(self):
        prepared = self.attention(None)
        tensors = [prepared['norm'], prepared['convolution'], *prepared['bases'], *prepared['projections'].values(),
                   *prepared['head_norms'].values(), prepared['output_projection']]
        self.assertEqual({tensor.dtype for tensor in tensors}, {'bf16'})

    def test_the_flag_moves_only_the_attention_projections_to_bfloat8(self):
        prepared = self.attention('1')
        self.assertEqual({name: tensor.dtype for name, tensor in prepared['projections'].items()},
                         dict(q='bf8', k='bf8', v='bf8'))
        self.assertEqual(prepared['output_projection'].dtype, 'bf8')
        kept = [prepared['norm'], prepared['convolution'], *prepared['bases'], *prepared['head_norms'].values()]
        self.assertEqual({tensor.dtype for tensor in kept}, {'bf16'})


class SharedWeightDtypeTests(unittest.TestCase):
    def setUp(self):
        self.layers = [(attention_weights(), convolution_weights(), mlp_weights()) for _ in range(5)]
        self.projection = {'fc.weight': torch.zeros(4, 4), 'hidden_norm.weight': torch.zeros(5120)}
        self.selector = {'norm.weight': torch.zeros(5120), 'candidate_selector.hidden_projection.weight': torch.zeros(2, 2),
                         'candidate_selector.predecessor_codebook': torch.zeros(1),
                         'candidate_selector.successor_codebook': torch.zeros(1)}

    def prepare(self, operations, flag):
        with environment(flag), ExitStack() as stack:
            stack.enter_context(patch('dflash_device.projection_shards', return_value=[torch.zeros(2, 4)]))
            return PreparedDraftWeights(operations, 'mesh', self.layers, self.projection, self.selector, block_rows=16)

    def lend_line(self, weights, flag):
        borrower = SimpleNamespace(name='device-A')
        with environment(flag), patch.object(dflash_device, 'pindiag') as log:
            weights.lend(borrower, mesh='mesh', layers=self.layers, projection=self.projection, selector=self.selector,
                         block_rows=16, live_query_qk=False, native_proposal_attention=False)
            weights.release(borrower)
        return log.call_args_list[0].args[0].format(*log.call_args_list[0].args[1:])

    def test_the_flag_moves_the_fc_projection_and_keeps_the_norms_and_selector(self):
        operations = FakeOperations()
        weights = self.prepare(operations, '1')
        self.assertEqual(weights.projection.dtype, 'bf8')
        self.assertEqual([weights.feature_norm.dtype, weights.final_norm.dtype, weights.selector_projection.dtype],
                         ['bf16'] * 3)
        self.assertEqual(weights.projection_dtypes(), {'bf8': PROJECTIONS})
        weights.close()

    def test_unset_the_lend_line_is_exactly_the_old_one(self):
        weights = self.prepare(FakeOperations(), None)
        self.assertEqual(weights.projection_dtypes(), {'bf16': PROJECTIONS})
        self.assertEqual(self.lend_line(weights, None),
                         '[PINDIAG] draft weights lent to device-A (borrowers=1 tensors=%d)' % len(weights.tensors))
        weights.close()

    def test_the_lend_line_reports_the_dtype_the_lent_tensors_carry_not_the_flag(self):
        # Uploaded under the flag, lent with it unset: the line still says bf8.
        weights = self.prepare(FakeOperations(), '1')
        self.assertEqual(self.lend_line(weights, None),
                         '[PINDIAG] draft weights lent to device-A (borrowers=1 tensors=%d) projections dtype=bf8 x%d'
                         % (len(weights.tensors), PROJECTIONS))
        weights.close()
        # Uploaded without it, lent with it set: the line is the old one.
        weights = self.prepare(FakeOperations(), None)
        self.assertEqual(self.lend_line(weights, '1'),
                         '[PINDIAG] draft weights lent to device-A (borrowers=1 tensors=%d)' % len(weights.tensors))
        weights.close()

    def test_a_mixed_set_is_reported_as_mixed(self):
        weights = self.prepare(FakeOperations(), '1')
        weights.projection.dtype = 'bf16'
        self.assertIn('projections dtype=bf16 x1,bf8 x%d' % (PROJECTIONS - 1), self.lend_line(weights, None))
        weights.close()


if __name__ == '__main__':
    unittest.main()
