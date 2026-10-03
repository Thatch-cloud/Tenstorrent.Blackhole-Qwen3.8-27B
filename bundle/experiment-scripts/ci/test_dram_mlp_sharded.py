from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dram_mlp_sharded


class Tensor:
    def __init__(self, shape, dtype, memory):
        self.shape, self.dtype, self.memory = shape, dtype, memory

    def memory_config(self):
        return self.memory


class DramMlpShardedTests(unittest.TestCase):
    def test_shared_staging_and_sharded_product(self):
        conversions, projects, products, released, kept = [], [], [], [], []
        source = Tensor((1, 1, 16, 5120), 'bf16', 'l1')
        weights, configs = {}, {}
        for name, (inner, width, unused, dtype, activated) in dram_mlp_sharded.PROJECTIONS.items():
            weights[name] = Tensor((1, 1, inner, width), dtype, 'dram')
            configs[name] = dict(weights='dram', inputs='down-input' if name == 'down' else 'shared-input')
        def convert(value, memory):
            result = Tensor(value.shape, value.dtype, memory)
            conversions.append((value, memory, result))
            return result
        def product(gate, up, memory_config):
            products.append((gate, up, memory_config))
            return Tensor(gate.shape, gate.dtype, memory_config)
        def project(ops, value, weight, config, compute, retain):
            projects.append(value)
            return retain(Tensor((1, 1, 16, weight.shape[-1]), 'bf16', 'native-sharded'))
        operations = SimpleNamespace(bfloat16='bf16', bfloat4_b='bfloat4_b', bfloat8_b='bfloat8_b',
            L1_MEMORY_CONFIG='l1', to_memory_config=convert, mul=product)
        with patch.object(dram_mlp_sharded, 'project', side_effect=project), patch.object(
                dram_mlp_sharded, 'release_owned', side_effect=lambda ops, tensors: released.extend(tensors)):
            output = dram_mlp_sharded.execute(operations, source, weights, configs, None, kept.append)
        self.assertIs(projects[0], projects[1])
        self.assertEqual([memory for value, memory, result in conversions], ['shared-input', 'down-input', 'l1'])
        self.assertEqual(products[0][2], 'native-sharded')
        self.assertEqual(kept, [output])
        self.assertFalse(any(value is source or value is output for value in released))
        self.assertEqual(len({id(value) for value in released}), len(released))
