from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import dspark_cached_layer as cached
import dspark_history as history
from dspark_full_attention import full_mask, geometry
from test_dspark_history import HostOperations, require_tensor


class DSparkCachedLayerTests(unittest.TestCase):
    def run_layer(self, position, proposals):
        operations = HostOperations()
        operations.float32 = torch.float32
        operations.reshape = torch.reshape
        operations.transpose = torch.transpose
        operations.multiply = lambda left, right, **kwargs: left.float() * right.float()
        mesh = SimpleNamespace(shape=(1, 2))
        weights = {name: name for name in cached.SPECIFICATIONS}
        source = torch.arange(32 * 5120).reshape(1, 1, 32, 5120).remainder(113).bfloat16()
        live = torch.zeros(1, 1, 32, 1, dtype=torch.float32)
        live[..., :proposals, :] = 1
        historical = tuple((torch.arange(position).reshape(1, 1, position, 1).remainder(127) + offset)
            .expand(1, 4, position, 128).bfloat16().contiguous() for offset in (0, 20))
        snapshots = tuple(value.clone() for value in historical)
        tables = (torch.ones(1, 1, 32, 128, dtype=torch.bfloat16), torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16))
        mask = full_mask(position, proposals)
        owned, projections = [], []

        def retain(value):
            owned.append(value)
            return value

        def linear(operations, value, weight, retain, *, rounded=True):
            projections.append((weight, tuple(value.shape)))
            width = 2048 if weight == 'self_attn.q_proj.weight' else 512
            if weight == 'self_attn.o_proj.weight':
                return retain(value.float().mean(-1, keepdim=True).expand(1, 1, 32, 5120).contiguous())
            return retain(value[..., :width].clone())

        def rotate(operations, value, cosine, sine, owned, **kwargs):
            result = value.clone()
            owned.append(result)
            return result

        def attend(operations, mesh, query, key, value, passed_mask, owned, **kwargs):
            self.assertIs(passed_mask, mask)
            self.assertEqual(kwargs, dict(context_rows=position, proposals=proposals, mask_validated=True))
            self.assertEqual(tuple(key.shape), (1, 4, geometry(position, proposals)[-1][1], 128))
            for actual, original in zip((key, value), historical, strict=True):
                self.assertTrue(torch.equal(actual[..., :position, :], original))
                new = source[..., :512].reshape(1, 32, 4, 128).transpose(1, 2)[..., :proposals, :]
                self.assertTrue(torch.equal(actual[..., position:position + proposals, :], new))
                self.assertTrue((actual[..., position + proposals:, :] == 0).all())
            result = torch.nn.functional.scaled_dot_product_attention(query.float(), key.float().repeat_interleave(4, 1),
                value.float().repeat_interleave(4, 1), attn_mask=mask.float()).bfloat16()
            owned.append(result)
            return result

        def gather(operations, mesh, collectives, partial, retain):
            self.assertTrue((partial[..., proposals:, :] == 0).all())
            return partial, partial

        def mlp(operations, first, second, noise, weights, retain):
            self.assertIs(noise, source)
            return dict(down_partial=first, attention_residual=noise)

        with ExitStack() as stack:
            stack.enter_context(patch.object(cached, 'require_tensor', side_effect=require_tensor))
            stack.enter_context(patch.object(cached, 'linear', side_effect=linear))
            stack.enter_context(patch.object(cached, 'norm', side_effect=lambda operations, value, gamma, retain: retain(value.clone())))
            stack.enter_context(patch.object(cached, 'rotate', side_effect=rotate))
            stack.enter_context(patch.object(cached, 'attend', side_effect=attend))
            gather_mock = stack.enter_context(patch.object(cached, 'gather_partials', side_effect=gather))
            stack.enter_context(patch.object(cached, 'mlp_partial', side_effect=mlp))
            finish_mock = stack.enter_context(patch.object(cached, 'finish', return_value={'output': source}))
            output = cached.execute(operations, mesh, None, source, historical, weights, tables, mask, live, retain,
                position=position, proposals=proposals, mask_validated=True)
        self.assertIs(output['finish']['output'], source)
        self.assertEqual(gather_mock.call_count, 2)
        self.assertIs(finish_mock.call_args.args[3], source)
        self.assertEqual(projections, [('self_attn.q_proj.weight', (1, 1, 32, 5120)),
            ('self_attn.k_proj.weight', (1, 1, 32, 5120)), ('self_attn.v_proj.weight', (1, 1, 32, 5120)),
            ('self_attn.o_proj.weight', (1, 1, 32, 2048))])
        for actual, original in zip(historical, snapshots, strict=True):
            self.assertTrue(torch.equal(actual, original))
        self.assertTrue(torch.isfinite(output['attention']).all())
        operations.to_torch.assert_not_called()

    def test_cached_history_is_not_reprojected_and_queries_have_no_padding_gap(self):
        for position, proposals in ((32, 7), (4093, 15), (4096, 15), (4109, 15)):
            with self.subTest(position=position, proposals=proposals):
                self.run_layer(position, proposals)

    def test_unvalidated_mask_rejects_before_native_operations(self):
        operations = Mock()
        with self.assertRaises(ValueError):
            cached.execute(operations, SimpleNamespace(shape=(1, 2)), None, None, (), {}, (), None, None,
                lambda value: value, position=4096, proposals=15)
        self.assertEqual(operations.mock_calls, [])

    def test_history_projection_shares_one_feature_projection_across_all_five_layers(self):
        operations = SimpleNamespace(reshape=torch.reshape, transpose=torch.transpose)
        weights = [{name: f'{layer}.{name}' for name in cached.SPECIFICATIONS} for layer in range(5)]
        features = tuple(torch.zeros(1, 1, 32, 2560, dtype=torch.bfloat16) for tap in range(5))
        parameters = {'fc.weight': object(), 'hidden_norm.weight': object()}
        context = torch.randn(1, 1, 32, 5120).bfloat16()
        tables = (object(), object())
        partial, peers = object(), (object(), object())
        projections = []

        def linear(operations, value, weight, retain):
            self.assertIs(value, context)
            projections.append(weight)
            return retain(value[..., :512].clone())

        def rotate(operations, value, cosine, sine, owned, **kwargs):
            self.assertIs(cosine, tables[0])
            self.assertIs(sine, tables[1])
            result = value.clone()
            owned.append(result)
            return result

        with patch.object(history, 'project', return_value={'partial': partial}) as project, \
                patch.object(history, 'gather_partials', return_value=peers) as gather, \
                patch.object(history, 'normalize_partials', return_value={'context': context}) as normalized, \
                patch.object(history, 'linear', side_effect=linear), \
                patch.object(history, 'norm', side_effect=lambda operations, value, gamma, retain: value) as key_norm, \
                patch.object(history, 'rotate', side_effect=rotate):
            result = history.project_block(operations, object(), object(), features, parameters, weights, tables, lambda value: value)
        self.assertEqual(project.call_count, 1)
        self.assertEqual(gather.call_count, 1)
        self.assertEqual(normalized.call_count, 1)
        self.assertEqual(key_norm.call_count, 5)
        self.assertTrue(all('.self_attn.k_norm.weight' in call.args[2] for call in key_norm.call_args_list))
        self.assertEqual(projections, [f'{layer}.self_attn.{name}_proj.weight' for layer in range(5) for name in ('k', 'v')])
        self.assertTrue(all(tuple(value.shape) == (1, 4, 32, 128) for value in history.leaves(result)))


if __name__ == '__main__':
    unittest.main()
