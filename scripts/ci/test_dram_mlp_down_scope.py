from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dram_mlp_down_scope as scope
from test_dram_mlp_sharded import Tensor


class DownScopeTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(bfloat4_b='bf4', bfloat8_b='bf8', DRAM_MEMORY_CONFIG='dram',
            L1_MEMORY_CONFIG='l1', MathFidelity=SimpleNamespace(LoFi='lofi'),
            synchronize_device=Mock(), deallocate=Mock())
        operations.to_memory_config = Mock(side_effect=lambda value, memory: value if value.memory == memory
            else Tensor(value.shape, value.dtype, memory))
        layers = []
        for unused in range(64):
            mlp = SimpleNamespace(_mlp_1d_decode=True, _dram_sharded=False,
                compute_kernel_config_decode=SimpleNamespace(math_fidelity='lofi', fp32_dest_acc_en=True, packer_l1_acc=True),
                weights=SimpleNamespace(w1=Tensor((5120, 8704), 'bf4', 'dram'),
                    w3=Tensor((5120, 8704), 'bf4', 'dram'), w2=Tensor((8704, 5120), 'bf8', 'dram')),
                forward=Mock(return_value='native'), args=SimpleNamespace(mlp_w1_decode_1d_progcfg='gate',
                    mlp_w3_decode_1d_progcfg='up', ccl_topology=lambda: 'linear'), device='mesh', tt_ccl='ccl')
            layers.append(SimpleNamespace(feed_forward=mlp))
        return operations, SimpleNamespace(layers=layers, args=SimpleNamespace(num_devices=2), mesh_device='mesh')

    def test_t16_only_all_layers_restore_and_caller_owned_input_survives(self):
        operations, model = self.fixture()
        originals = [layer.feed_forward.forward for layer in model.layers]
        release, collective = Mock(), Mock(return_value='reduced')
        with patch.object(scope, 'configurations', return_value=dict(weights='sharded')), patch.object(
                scope, 'addresses', side_effect=lambda ops, value: id(value)), patch.object(
                scope, 'execute', return_value='partial') as execute, patch.object(scope, 'release_owned', release):
            with scope.scoped_down(operations, model, collective) as audit:
                for layer in model.layers:
                    for rows in (1, 32, 4096):
                        self.assertEqual(layer.feed_forward.forward(Tensor((1, 1, rows, 5120), 'bf16', 'l1')), 'native')
                    self.assertEqual(layer.feed_forward.forward(Tensor((1, 1, 16, 5120), 'bf16', 'l1')), 'reduced')
                self.assertEqual(execute.call_count, 64)
                operations.deallocate.assert_not_called()
                self.assertTrue(all(call.kwargs['cluster_axis'] == 0 and call.kwargs['dim'] == 3
                    for call in collective.call_args_list))
            self.assertTrue(audit['restored'])
            self.assertTrue(audit['native_bindings_unchanged'])
            self.assertEqual(audit['hits'], [1] * 64)
            self.assertEqual(audit['fallbacks'], [3] * 64)
            self.assertTrue(all(layer.feed_forward.forward is original
                for layer, original in zip(model.layers, originals)))
            self.assertEqual(len(release.call_args.args[1]), 64)

    def test_partial_setup_failure_releases_allocations_without_installing(self):
        operations, model = self.fixture()
        originals = [layer.feed_forward.forward for layer in model.layers]
        operations.to_memory_config.side_effect = [object(), RuntimeError('allocation failed')]
        with patch.object(scope, 'configurations', return_value=dict(weights='sharded')), patch.object(
                scope, 'addresses', side_effect=lambda ops, value: id(value)), patch.object(scope, 'release_owned') as release:
            with self.assertRaisesRegex(RuntimeError, 'allocation failed'):
                scope.DownOnlyArm(operations, model, Mock())
            self.assertEqual(len(release.call_args.args[1]), 1)
            self.assertTrue(all(layer.feed_forward.forward is original
                for layer, original in zip(model.layers, originals)))

    def test_forward_failure_releases_only_converted_input_and_restores(self):
        operations, model = self.fixture()
        original = model.layers[0].feed_forward.forward
        value = Tensor((1, 1, 16, 5120), 'bf16', 'width_sharded')
        with patch.object(scope, 'configurations', return_value=dict(weights='sharded')), patch.object(
                scope, 'addresses', side_effect=lambda ops, tensor: id(tensor)), patch.object(
                scope, 'execute', side_effect=RuntimeError('kernel failed')), patch.object(scope, 'release_owned'):
            with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
                with scope.scoped_down(operations, model, Mock()):
                    model.layers[0].feed_forward.forward(value)
        operations.deallocate.assert_called_once()
        self.assertIsNot(operations.deallocate.call_args.args[0], value)
        self.assertIs(model.layers[0].feed_forward.forward, original)


if __name__ == '__main__':
    unittest.main()
