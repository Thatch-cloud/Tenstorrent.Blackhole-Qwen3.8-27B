from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dram_mlp_down
from test_dram_mlp_sharded import Tensor


class DramMlpDownTests(unittest.TestCase):
    def fixture(self):
        source = Tensor((1, 1, 16, 5120), 'bf16', 'l1')
        weights = {name: Tensor((5120, 8704), 'bf4', 'dram') for name in ('gate', 'up')}
        weights['down'] = object()
        configs = {name: dict(program=name) for name in weights}
        operations = SimpleNamespace(bfloat16='bf16', bfloat4_b='bf4', L1_MEMORY_CONFIG='l1',
            DRAM_MEMORY_CONFIG='dram', linear=Mock(side_effect=[object(), object()]), mul=Mock(return_value=object()))
        return operations, source, weights, configs

    def test_only_down_is_sharded_and_owned_temporaries_are_released(self):
        operations, source, weights, configs = self.fixture()
        scratch, output, released, kept = object(), object(), [], []
        def project(ops, value, weight, config, compute, retain, **keywords):
            self.assertIs(value, operations.mul.return_value)
            self.assertIs(weight, weights['down'])
            self.assertEqual(keywords, dict(preserve_partials=True))
            retain(scratch)
            return retain(output)
        with patch.object(dram_mlp_down, 'project', side_effect=project) as projection, patch.object(
                dram_mlp_down, 'release_owned', side_effect=lambda ops, values: released.extend(values)):
            result = dram_mlp_down.execute(operations, source, weights, configs, None, kept.append)
        self.assertIs(result, output)
        self.assertEqual(projection.call_count, 1)
        self.assertEqual([call.kwargs['program_config'] for call in operations.linear.call_args_list], ['gate', 'up'])
        self.assertEqual(operations.mul.call_args.kwargs, dict(memory_config='l1'))
        self.assertEqual(kept, [output])
        self.assertEqual(len(released), 4)
        self.assertEqual(len({id(value) for value in released}), 4)
        self.assertFalse(any(value is output or value is source for value in released))

    def test_failed_down_releases_native_outputs_hidden_and_scratch(self):
        operations, source, weights, configs = self.fixture()
        scratch, released = object(), []
        def project(ops, value, weight, config, compute, retain, **keywords):
            retain(scratch)
            raise RuntimeError('down failed')
        with patch.object(dram_mlp_down, 'project', side_effect=project), patch.object(
                dram_mlp_down, 'release_owned', side_effect=lambda ops, values: released.extend(values)):
            with self.assertRaisesRegex(RuntimeError, 'down failed'):
                dram_mlp_down.execute(operations, source, weights, configs, None, lambda value: value)
        self.assertEqual(len(released), 4)
        self.assertEqual(len({id(value) for value in released}), 4)

    def test_sharded_gate_rejected_before_any_execution(self):
        operations, source, weights, configs = self.fixture()
        weights['gate'].memory = 'sharded'
        with self.assertRaises(ValueError):
            dram_mlp_down.execute(operations, source, weights, configs, None, None)
        operations.linear.assert_not_called()


if __name__ == '__main__':
    unittest.main()
