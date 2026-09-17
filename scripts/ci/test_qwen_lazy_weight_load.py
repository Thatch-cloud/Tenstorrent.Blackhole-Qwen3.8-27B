import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qwen_lazy_weight_load import lazy_gate_up_loader, lazy_shard_loader, lazy_weight_load


class LazyWeightTests(unittest.TestCase):
    def original(self, torch_tensor, mesh, dim, memory_config, cache_path, dtype='bfp8'):
        raise AssertionError('Original loader should not run in selected scope')

    def operations(self, load):
        return SimpleNamespace(bfloat8_b='bfp8', TILE_LAYOUT='tile', as_tensor=load,
            ShardTensorToMesh=lambda mesh, dim: (mesh, dim))

    def test_cache_hit_never_materializes_input_and_preserves_options(self):
        cached, untouched = object(), object()
        received = {}
        def load(tensor, **kwargs):
            self.assertIs(tensor, untouched)
            received.update(kwargs)
            return cached
        module = SimpleNamespace(shard_w=self.original)
        original, records = module.shard_w, []
        with patch.dict(os.environ, QWEN_LAZY_WEIGHT_LOAD='1'):
            with lazy_weight_load(module, self.operations(load), None, records):
                result = module.shard_w(untouched, 'mesh', -1, 'dram', 'cache', dtype='bfp4')
        self.assertIs(result, cached)
        self.assertIs(module.shard_w, original)
        self.assertEqual(records[0]['calls'], 1)
        self.assertEqual(records[0]['materializations'], 0)
        self.assertTrue(records[0]['restored'])
        self.assertEqual({key: value for key, value in received.items() if key != 'preprocess'},
            dict(dtype='bfp4', device='mesh', mesh_mapper=('mesh', -1), layout='tile',
                memory_config='dram', cache_file_name='cache'))

    def test_cache_miss_preserves_conversion_order(self):
        actions = []
        class Tensor:
            def to(self, dtype):
                actions.append(('to', dtype))
                return self
            @property
            def T(self):
                actions.append('transpose')
                return self
            def contiguous(self):
                actions.append('contiguous')
                return 'materialized'
        def load(tensor, **kwargs):
            return kwargs['preprocess'](tensor)
        record = dict(calls=0, materializations=0)
        loader = lazy_shard_loader(self.original, self.operations(load),
            SimpleNamespace(bfloat16='bf16'), record)
        self.assertEqual(loader(Tensor(), 'mesh', 0, 'dram', None), 'materialized')
        self.assertEqual(actions, [('to', 'bf16'), 'transpose', 'contiguous'])
        self.assertEqual(record['materializations'], 1)

    def test_unselected_scope_fails(self):
        with patch.dict(os.environ, QWEN_LAZY_WEIGHT_LOAD='0'):
            with self.assertRaises(ValueError):
                with lazy_weight_load(SimpleNamespace(shard_w=self.original), None, None, []):
                    self.fail('Unselected scope entered')

    def test_packed_cache_hit_does_not_touch_either_input(self):
        cached, gate, up = object(), object(), object()
        def load(tensor, **kwargs):
            self.assertIs(tensor, gate)
            self.assertEqual(kwargs['dtype'], 'bfp4')
            self.assertEqual(kwargs['cache_file_name'], 'packed-cache')
            return cached
        operations = self.operations(load)
        operations.bfloat4_b = 'bfp4'
        operations.DRAM_MEMORY_CONFIG = 'dram'
        record = dict(packed_calls=0, packed_materializations=0)
        loader = lazy_gate_up_loader(lambda: None, operations, None, None, record)
        self.assertIs(loader(gate, up, 'mesh', 2, 'packed-cache'), cached)
        self.assertEqual(record, dict(packed_calls=1, packed_materializations=0))

    def test_torch_miss_matches_original_packing_input_exactly(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch environment required for tensor equivalence')
        gate = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 7
        up = -gate - 0.125
        expected = torch.cat([gate.to(torch.bfloat16).T.contiguous(),
            up.to(torch.bfloat16).T.contiguous()], dim=-1)
        def prepare(value, ndev, gate_is_first):
            self.assertEqual(ndev, 2)
            self.assertTrue(gate_is_first)
            self.assertTrue(torch.equal(value, expected))
            self.assertTrue(value.is_contiguous())
            return value
        operations = self.operations(lambda value, **kwargs: kwargs['preprocess'](value))
        operations.bfloat4_b = 'bfp4'
        operations.DRAM_MEMORY_CONFIG = 'dram'
        record = dict(packed_calls=0, packed_materializations=0)
        loader = lazy_gate_up_loader(lambda: None, operations, torch, prepare, record)
        self.assertTrue(torch.equal(loader(gate, up, 'mesh', 2, None), expected))
        self.assertEqual(record['packed_materializations'], 1)


if __name__ == '__main__':
    unittest.main()
