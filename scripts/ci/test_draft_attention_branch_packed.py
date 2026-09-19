"""The packed multi-user path through execute_attention_branch.

Numerical equivalence is proved on real tensors in test_dflash_batched_mask. What
this pins down is that the DEVICE code assembles that exact layout: each user's
cached rows, then its own 16 rows taken out of the shared live block, then padding
to its aligned span - and that pack=None still walks today's code unchanged.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dflash_batched_mask import key_value_plan, live_key_rope
from draft_attention_branch import execute_attention_branch, prepare_attention_branch


class PackedBranchTests(unittest.TestCase):
    contexts = (2048, 2048)

    def fixture(self, key_rows):
        pieces = {}
        operations = SimpleNamespace(bfloat16='bf16', float32='fp32', TILE_LAYOUT='tile',
            ROW_MAJOR_LAYOUT='row', DRAM_MEMORY_CONFIG='dram',
            MathFidelity=SimpleNamespace(HiFi4='hifi4'))
        for name in ('from_torch', 'ShardTensorToMesh', 'ReplicateTensorToMesh',
                'WormholeComputeKernelConfig', 'MatmulMultiCoreReuseMultiCast1DProgramConfig',
                'matmul', 'rms_norm', 'typecast', 'add', 'zeros_like', 'reshape', 'transpose', 'pad'):
            setattr(operations, name, Mock(side_effect=lambda *args, **kwargs: object()))

        def record_slice(value, start, stop, *args, **kwargs):
            handle = object()
            pieces[id(handle)] = ('slice', id(value), start[2], stop[2])
            return handle

        def record_concat(parts, *args, **kwargs):
            handle = object()
            pieces[id(handle)] = ('concat', [pieces.get(id(part), ('given', id(part))) for part in parts])
            return handle

        operations.slice = Mock(side_effect=record_slice)
        operations.concat = Mock(side_effect=record_concat)
        operations.experimental = SimpleNamespace(rotary_embedding_hf=Mock(return_value=object()))
        weights = {'layers.0.self_attn.%s_proj.weight' % name: torch.ones(2, 2) for name in ('q', 'k', 'v', 'o')}
        weights.update({'layers.0.self_attn.%s_norm.weight' % name: torch.ones(128) for name in ('q', 'k')})
        convolution = {'layers.0.input_layernorm.weight': torch.ones(5120),
            'layers.0.attention_conv.kernel_projection.weight': torch.ones(2, 2),
            'layers.0.attention_conv.base_kernel': torch.ones(2, 2, 5120)}
        tensors = SimpleNamespace(
            hidden=SimpleNamespace(shape=(1, 1, 32, 5120), dtype='bf16'),
            history=SimpleNamespace(shape=(1, 1, key_rows, 5120), dtype='bf16'),
            mask=SimpleNamespace(shape=(1, 1, 32, key_rows), dtype='bf16'))
        rope = {'q': tuple(SimpleNamespace(shape=(1, 1, 32, 128), dtype='bf16') for _ in range(2)),
                'k': tuple(SimpleNamespace(shape=(1, 1, key_rows, 128), dtype='bf16') for _ in range(2)),
                'live_k': tuple(SimpleNamespace(shape=(1, 1, 32, 128), dtype='bf16') for _ in range(2))}
        return operations, weights, convolution, tensors, rope, pieces

    def users(self):
        return [dict(position=4096, history_rows=self.contexts[0]),
                dict(position=9000, history_rows=self.contexts[1])]

    def run_packed(self, operations, weights, convolution, tensors, rope):
        mesh, collective = object(), object()
        caches = [{name: SimpleNamespace(shape=(1, 4, context, 128), dtype='bf16') for name in ('k', 'v')}
                  for context in self.contexts]
        live = {name: object() for name in ('q', 'k', 'v')}
        with patch('draft_attention_branch.grouped_causal_convolution', return_value=object()), \
                patch('draft_attention_branch.gather_add_projection', return_value=object()), \
                patch('draft_attention_branch.concatenate_query_heads', return_value=object()), \
                patch('draft_attention_branch.split_projected_heads',
                      side_effect=lambda *a, **k: {name: object() for name in ('q', 'k', 'v')}), \
                patch('draft_kv_projection.project_key_value', return_value=live) as projection, \
                patch('dflash_t16_native_attention.attention', return_value=object()), \
                patch('dflash_t16_native_scope.require_active', return_value=None):
            parameters = prepare_attention_branch(operations, mesh, weights, convolution, lambda value: value)
            parameters.update(block_rows=16, native_head_layout=True, native_proposal_attention=True)
            execute_attention_branch(operations, mesh, collective, tensors.hidden, None,
                tensors.mask, rope, lambda value: value, parameters=parameters, context=None,
                pack=self.users(), cached_history=caches, native_proposal_mask_validated=True)
        return projection, caches, live

    def test_key_axis_is_assembled_exactly_as_the_plan_says(self):
        plan, spans, key_rows = key_value_plan(list(self.contexts), block_rows=16)
        operations, weights, convolution, tensors, rope, pieces = self.fixture(key_rows)
        self.run_packed(operations, weights, convolution, tensors, rope)
        assembled = [entry for entry in pieces.values()
                     if entry[0] == 'concat' and len(entry[1]) == len(plan)]
        self.assertEqual(len(assembled), 2, 'one assembled key axis and one value axis')
        for entry in assembled:
            self.assertEqual([part[0] for part in entry[1]],
                             ['given', 'slice', 'slice', 'given', 'slice', 'slice'],
                             'cached rows come from the caches, block rows from the live block')
            self.assertEqual([(part[2], part[3]) for part in entry[1] if part[0] == 'slice'],
                             [(0, 16), (0, 16), (16, 32), (0, 16)],
                             'user 0 takes live rows 0-16, user 1 takes 16-32, pads reuse masked rows')

    def test_the_live_block_is_not_padded_and_uses_the_supplied_block_rope(self):
        plan, spans, key_rows = key_value_plan(list(self.contexts), block_rows=16)
        operations, weights, convolution, tensors, rope, pieces = self.fixture(key_rows)
        projection, caches, live = self.run_packed(operations, weights, convolution, tensors, rope)
        self.assertIs(projection.call_args.args[3], rope['live_k'],
                      'the packed path must rotate the block with the caller-supplied table')
        operations.pad.assert_not_called()

    def test_a_missing_cache_or_a_leftover_context_is_refused(self):
        plan, spans, key_rows = key_value_plan(list(self.contexts), block_rows=16)
        for overrides in ({'context': 2048}, {'cached_history': None}):
            operations, weights, convolution, tensors, rope, pieces = self.fixture(key_rows)
            mesh = object()
            parameters = dict(operations=operations, mesh=mesh, block_rows=16,
                              native_head_layout=True, native_proposal_attention=True)
            arguments = dict(context=None, pack=self.users(),
                             cached_history=[{'k': None, 'v': None}, {'k': None, 'v': None}])
            arguments.update(overrides)
            with self.assertRaises(ValueError):
                execute_attention_branch(operations, mesh, object(), tensors.hidden, None,
                    tensors.mask, rope, lambda value: value, parameters=parameters, **arguments)
            operations.matmul.assert_not_called()

    def test_an_unread_packed_history_tensor_is_refused(self):
        """Packed and cached, keys come from the caches and the live block, so a
        history tensor would be 42 MB nobody reads. Passing one is a mistake."""
        plan, spans, key_rows = key_value_plan(list(self.contexts), block_rows=16)
        operations, weights, convolution, tensors, rope, pieces = self.fixture(key_rows)
        mesh = object()
        parameters = dict(operations=operations, mesh=mesh, block_rows=16,
                          native_head_layout=True, native_proposal_attention=True)
        with self.assertRaises(ValueError):
            execute_attention_branch(operations, mesh, object(), tensors.hidden, tensors.history,
                tensors.mask, rope, lambda value: value, parameters=parameters, context=None,
                pack=self.users(),
                cached_history=[{'k': None, 'v': None}, {'k': None, 'v': None}],
                native_proposal_mask_validated=True)

    def test_live_key_rope_shape_matches_what_the_branch_expects(self):
        tables = live_key_rope(self.users(), block_rows=16)
        self.assertEqual([tuple(table.shape) for table in tables], [(1, 1, 32, 128)] * 2)


if __name__ == '__main__':
    unittest.main()
