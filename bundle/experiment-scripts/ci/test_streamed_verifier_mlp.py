from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamed_verifier_mlp import StreamedMlpScope
from tensix_stream_mlp import StreamBufferPool


class StreamedVerifierMlpTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', L1_MEMORY_CONFIG='l1',
            MathFidelity=SimpleNamespace(LoFi='lofi'), Topology=SimpleNamespace(Linear='linear'),
            get_device_tensors=lambda value: [value, value], deallocate=Mock(), to_memory_config=Mock())
        mesh = SimpleNamespace(shape=[1, 2])
        collective = SimpleNamespace(get_num_links=Mock(return_value=4))
        def tensor(address):
            return SimpleNamespace(shape=(1, 1, 8, 5120), dtype='bf16', layout='tile', buffer_address=lambda: address)
        source, converted = tensor(1024), tensor(2048)
        operations.to_memory_config.return_value = converted
        workspace = {name: tensor(4096 * (index + 1)) for index, name in enumerate(('gate', 'up', 'hidden', 'partial'))}
        layers = []
        for index in range(64):
            module = SimpleNamespace(device=mesh, num_devices=2, _mlp_1d_decode=True, _dram_sharded=False, tt_ccl=collective,
                args=SimpleNamespace(dim=5120, hidden_dim=17408, decode_grid_w=11, ccl_topology=lambda: 'linear'),
                compute_kernel_config_decode=SimpleNamespace(math_fidelity='lofi', math_approx_mode=True,
                    fp32_dest_acc_en=True, packer_l1_acc=True), weights=SimpleNamespace(w1=object(), w2=object(), w3=object()),
                forward=Mock(return_value=index))
            layers.append(SimpleNamespace(feed_forward=module))
        model = SimpleNamespace(mesh_device=mesh, tt_ccl=collective, layers=layers)
        pool = StreamBufferPool(operations, mesh)
        pool.entries.update({36864: object(), 34816: object()})
        return operations, model, pool, workspace, source, converted

    def test_all_layers_share_workspace_and_restore_existing_instance_methods(self):
        operations, model, pool, workspace, source, converted = self.fixture()
        originals = [layer.feed_forward.forward for layer in model.layers]
        scope = StreamedMlpScope(operations, model, pool, workspace, '/native')
        with patch('streamed_verifier_mlp.prepare_mlp', return_value='prepared') as prepare, \
                patch('streamed_verifier_mlp.execute_mlp', return_value=workspace) as execute, \
                patch('streamed_verifier_mlp.reduce_partial', return_value='reduced') as reduce:
            for unused in range(2):
                with scope.capture():
                    for layer in model.layers:
                        self.assertEqual(layer.feed_forward.forward(source), 'reduced')
                self.assertEqual([layer.feed_forward.forward for layer in model.layers], originals)
                self.assertNotIn('_qwen_streamed_mlp_scope', vars(model))
            self.assertEqual(prepare.call_count, 128)
            self.assertTrue(all(call.args[2] is converted and call.args[4] == workspace and call.args[5] is pool
                for call in prepare.call_args_list))
            self.assertEqual(execute.call_count, 128)
            self.assertEqual(reduce.call_count, 128)
        self.assertEqual(operations.deallocate.call_count, 128)
        self.assertTrue(all(call.args == (converted,) for call in operations.deallocate.call_args_list))
        self.assertEqual(scope.summary()['completed_forwards'], 2)
        self.assertEqual(scope.summary()['pool_buffers'], 2)
        self.assertFalse(scope.summary()['failed'])

    def test_aliased_native_conversion_never_frees_caller_input(self):
        operations, model, pool, workspace, source, unused = self.fixture()
        operations.to_memory_config.return_value = source
        scope = StreamedMlpScope(operations, model, pool, workspace, '/native')
        with patch('streamed_verifier_mlp.prepare_mlp'), patch('streamed_verifier_mlp.execute_mlp', return_value=workspace), \
                patch('streamed_verifier_mlp.reduce_partial'):
            with scope.capture():
                for layer in model.layers:
                    layer.feed_forward.forward(source)
        operations.deallocate.assert_not_called()
        self.assertEqual(scope.summary()['converted_inputs'], 0)

    def test_incomplete_or_failed_forward_restores_methods_and_cannot_resume(self):
        for failure in ('incomplete', 'kernel'):
            operations, model, pool, workspace, source, converted = self.fixture()
            originals = [layer.feed_forward.forward for layer in model.layers]
            scope = StreamedMlpScope(operations, model, pool, workspace, '/native')
            with patch('streamed_verifier_mlp.prepare_mlp'), \
                    patch('streamed_verifier_mlp.execute_mlp', side_effect=RuntimeError('kernel')):
                with self.subTest(failure=failure), self.assertRaises((AssertionError, RuntimeError)):
                    with scope.capture():
                        if failure == 'kernel':
                            model.layers[0].feed_forward.forward(source)
            self.assertEqual([layer.feed_forward.forward for layer in model.layers], originals)
            self.assertNotIn('_qwen_streamed_mlp_scope', vars(model))
            self.assertTrue(scope.failed)
            if failure == 'kernel':
                operations.deallocate.assert_called_once_with(converted)
            with self.assertRaises(RuntimeError), scope.capture():
                self.fail('Failed scope cannot resume')

    def test_wrong_width_order_or_workspace_fails_before_input_conversion(self):
        for failure in ('width', 'order', 'workspace', 'nested'):
            operations, model, pool, workspace, source, unused = self.fixture()
            scope = StreamedMlpScope(operations, model, pool, workspace, '/native')
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                with scope.capture():
                    if failure == 'width':
                        source.shape = (1, 1, 4, 5120)
                    elif failure == 'workspace':
                        workspace['partial'].buffer_address = lambda: 17
                    elif failure == 'nested':
                        with scope.capture():
                            self.fail('Nested ownership must fail')
                    model.layers[1 if failure == 'order' else 0].feed_forward.forward(source)
            operations.to_memory_config.assert_not_called()

    def test_native_mode_and_matched_link_policy_cannot_change_implicitly(self):
        for failure in ('layers', 'duplicate', 'math', 'layout', 'collective', 'links', 'pool'):
            operations, model, pool, workspace, unused_source, unused_converted = self.fixture()
            if failure == 'layers':
                model.layers.pop()
            elif failure == 'duplicate':
                model.layers[1] = model.layers[0]
            elif failure == 'math':
                model.layers[0].feed_forward.compute_kernel_config_decode.math_approx_mode = False
            elif failure == 'layout':
                model.layers[0].feed_forward._dram_sharded = True
            elif failure == 'collective':
                model.layers[0].feed_forward.tt_ccl = object()
            elif failure == 'links':
                model.tt_ccl.get_num_links.return_value = 2
            else:
                pool.entries.pop(34816)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                StreamedMlpScope(operations, model, pool, workspace, '/native')
            operations.to_memory_config.assert_not_called()
            operations.deallocate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
