from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tensix_stream_mlp import StreamBufferPool, execute_from_dram, execute_mlp, prepare_mlp


class TensixStreamMlpTests(unittest.TestCase):
    def test_dram_boundary_is_copied_before_execution_and_preserves_destination(self):
        destination = SimpleNamespace(buffer_address=lambda: 4096)
        source = SimpleNamespace(shape=(1, 1, 8, 5120), dtype='bf16', layout='tile',
            memory_config=lambda: 'dram', buffer_address=lambda: 8192)
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            get_device_tensors=lambda value: [value, value], copy=Mock(return_value=destination))
        prepared = SimpleNamespace(references=[destination], bindings=[(4096, 4096)],
            pool=SimpleNamespace(operations=operations))
        with patch('tensix_stream_mlp.execute_mlp') as execute:
            execute.side_effect = lambda *unused: operations.copy.assert_called_once_with(source, destination)
            execute_from_dram(operations, source, prepared)
            execute.assert_called_once_with(operations, prepared)
            execute.reset_mock()
            operations.copy.return_value = source
            with self.assertRaisesRegex(ValueError, 'replaced'):
                execute_from_dram(operations, source, prepared)
            execute.assert_not_called()
            operations.copy.reset_mock()
            source.buffer_address = lambda: 4096
            with self.assertRaisesRegex(ValueError, 'alias'):
                execute_from_dram(operations, source, prepared)
            operations.copy.assert_not_called()

    def fixture(self, producers=8):
        operations = SimpleNamespace(bfloat16='bf16', bfloat4_b='bf4', bfloat8_b='bf8', TILE_LAYOUT='tile',
            L1_MEMORY_CONFIG='l1', DRAM_MEMORY_CONFIG='dram', CoreCoord=lambda column, row: (column, row),
            CoreRange=lambda begin, end: (begin, end), CoreRangeSet=tuple,
            get_device_tensors=lambda value: [value, value],
            create_global_circular_buffer=Mock(side_effect=lambda *unused: SimpleNamespace(sender_core_type=lambda: 'worker')),
            multiply=Mock())
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        count = 0
        def tensor(shape, dtype='bf16', memory='l1'):
            nonlocal count
            count += 1
            address = 1024 * count
            return SimpleNamespace(shape=shape, dtype=dtype, layout='tile', memory_config=lambda: memory,
                buffer_address=lambda: address)
        source = tensor((1, 1, 8, 5120))
        weights = {name: tensor((1, 1, 8704, 5120) if name == 'down' else (1, 1, 5120, 8704),
            'bf8' if name == 'down' else 'bf4', 'dram') for name in ('gate', 'up', 'down')}
        buffers = {name: tensor((1, 1, 8, 5120 if name == 'partial' else 8704))
            for name in ('gate', 'up', 'hidden', 'partial')}
        pool = StreamBufferPool(operations, mesh, producers)
        operations.multiply.return_value = buffers['hidden']
        return operations, mesh, source, weights, buffers, pool

    def test_pool_is_bounded_and_reused_across_layers_and_bf4_projections(self):
        operations, mesh, unused_source, unused_weights, unused_buffers, pool = self.fixture()
        initial = {}
        for unused_layer in range(64):
            for size in (36864, 36864, 34816):
                value = pool.acquire(mesh, pool.expected_mapping(size), size)
                if size in initial:
                    self.assertIs(value, initial[size])
                initial[size] = value
        self.assertEqual(len(pool.entries), 2)
        self.assertEqual(operations.create_global_circular_buffer.call_count, 2)

    def test_pool_rejects_foreign_mesh_changed_mapping_and_unknown_sizes(self):
        operations, mesh, unused_source, unused_weights, unused_buffers, pool = self.fixture()
        for mutation in ('mesh', 'mapping', 'size'):
            mapping = pool.expected_mapping(36864)
            if mutation == 'mapping':
                mapping = mapping[:-1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                pool.acquire(object() if mutation == 'mesh' else mesh, mapping, 1 if mutation == 'size' else 36864)
        operations.create_global_circular_buffer.assert_not_called()

    def prepared(self, producers=8):
        operations, mesh, source, weights, buffers, pool = self.fixture(producers)
        def projection(pooled, used_mesh, activation, weight, output, geometry, native):
            size = 2 * geometry['page_bytes']
            return SimpleNamespace(name=geometry['projection'], output=output, geometry=geometry,
                gcb=pooled.create_global_circular_buffer(used_mesh, pool.expected_mapping(size), size))
        with patch('tensix_stream_mlp.prepare_projection', side_effect=projection):
            prepared = prepare_mlp(operations, mesh, source, weights, buffers, pool, 'native')
        return operations, prepared

    def test_sixteen_producer_composition_uses_two_matched_fifos(self):
        operations, prepared = self.prepared(16)
        self.assertEqual(prepared.pool.producers, 16)
        self.assertEqual(operations.create_global_circular_buffer.call_count, 2)
        for projection in prepared.projections.values():
            self.assertEqual(projection.geometry['producers'], 16)
            self.assertEqual(len(projection.geometry['mapping']), 16)
        for call in operations.create_global_circular_buffer.call_args_list:
            self.assertEqual(len(call.args[1]), 16)
        with self.assertRaises(AttributeError):
            prepared.pool.producers = 8
        with self.assertRaises(ValueError):
            StreamBufferPool(operations, prepared.pool.mesh, 12)

    def test_composition_uses_native_multiply_and_preallocated_output(self):
        operations, prepared = self.prepared()
        order = []
        def project(unused, value):
            order.append(value.name)
            return value.output
        def multiply(*unused, **arguments):
            order.append('multiply')
            return arguments['output_tensor']
        operations.multiply.side_effect = multiply
        with patch('tensix_stream_mlp.execute_projection', side_effect=project):
            self.assertIs(execute_mlp(operations, prepared), prepared.buffers)
        self.assertEqual(order, ['gate', 'up', 'multiply', 'down'])
        self.assertIs(operations.multiply.call_args.kwargs['output_tensor'], prepared.buffers['hidden'])
        self.assertEqual(operations.create_global_circular_buffer.call_count, 2)

    def test_cross_component_alias_is_rejected_before_any_fifo_allocation(self):
        operations, mesh, source, weights, buffers, pool = self.fixture()
        buffers['up'] = buffers['gate']
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            prepare_mlp(operations, mesh, source, weights, buffers, pool, 'native')
        operations.create_global_circular_buffer.assert_not_called()

    def test_changed_binding_fails_before_dispatch(self):
        operations, prepared = self.prepared()
        prepared.references[0].buffer_address = lambda: 999
        with patch('tensix_stream_mlp.execute_projection') as project, self.assertRaisesRegex(ValueError, 'bindings'):
            execute_mlp(operations, prepared)
        project.assert_not_called()

    def test_replaced_multiply_output_stops_before_down_projection(self):
        operations, prepared = self.prepared()
        operations.multiply.return_value = SimpleNamespace(buffer_address=lambda: 123456)
        with patch('tensix_stream_mlp.execute_projection', side_effect=lambda unused, value: value.output) as project:
            with self.assertRaisesRegex(ValueError, 'replaced'):
                execute_mlp(operations, prepared)
        self.assertEqual(project.call_count, 2)


if __name__ == '__main__':
    unittest.main()
