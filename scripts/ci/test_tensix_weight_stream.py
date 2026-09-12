from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tensix_weight_stream import prepare_stream, stream_geometry


class TensixWeightStreamTests(unittest.TestCase):
    def test_sixteen_producers_only_change_fanout_and_disjoint_sender_coordinates(self):
        for projection, blocks in (('gate', 20), ('up', 20), ('down', 34)):
            original = stream_geometry(projection, blocks)
            expanded = stream_geometry(projection, blocks, 16)
            self.assertEqual({name: value for name, value in expanded.items() if name not in ('producers', 'mapping')},
                {name: value for name, value in original.items() if name != 'mapping'})
            senders = {coordinate for coordinate, unused in expanded['mapping']}
            self.assertEqual(senders, {(column, row) for column in range(8) for row in (8, 9)})
            rows = (expanded['receivers'] + 10) // 11
            activation = {(column, row) for column in range(11) for row in range(rows)}
            self.assertFalse(senders & activation)
            self.assertEqual(len(senders | activation), 104 if projection == 'down' else 93)
            assigned = [index for unused, indices in expanded['mapping'] for index in indices]
            self.assertEqual(sorted(assigned), list(range(expanded['receivers'])))
            self.assertEqual(max(len(indices) for unused, indices in expanded['mapping']), 5)
        for count in (True, 8.0, 0, 12, 24):
            with self.subTest(count=count), self.assertRaises(ValueError):
                stream_geometry('gate', 20, count)

    def test_receivers_and_senders_are_disjoint_complete_worker_sets(self):
        for projection, count in (('gate', 68), ('up', 68), ('down', 80)):
            geometry = stream_geometry(projection, 5)
            receivers = set(geometry['coordinates'])
            senders = {coordinate for coordinate, unused in geometry['mapping']}
            mapped = [index for unused, indices in geometry['mapping'] for index in indices]
            self.assertEqual(len(receivers), count)
            self.assertEqual(len(senders), 8)
            self.assertFalse(receivers & senders)
            self.assertEqual(sorted(mapped), list(range(count)))
            self.assertTrue(all(0 <= column < 11 and 0 <= row < 10 for column, row in receivers | senders))

    def test_native_shapes_compression_and_two_page_budgets(self):
        for projection, blocks, width, receivers, page_bytes, dtype in (
                ('gate', 20, 8704, 68, 18432, 'bfloat4_b'),
                ('up', 20, 8704, 68, 18432, 'bfloat4_b'),
                ('down', 34, 5120, 80, 17408, 'bfloat8_b')):
            geometry = stream_geometry(projection, blocks)
            self.assertTrue(geometry['full_projection'])
            self.assertEqual(geometry['rows'], blocks * 256)
            self.assertEqual(geometry['width'], width)
            self.assertEqual(geometry['dtype'], dtype)
            self.assertEqual(geometry['page_bytes'], page_bytes)
            self.assertEqual(geometry['per_receiver'] * receivers * 32, width)
            self.assertLessEqual(2 * max(len(indices) for unused, indices in geometry['mapping']) * page_bytes, 348160)
            self.assertFalse(stream_geometry(projection, 5)['full_projection'])

    def test_unequal_gate_bank_counts_are_preserved(self):
        self.assertEqual([len(indices) for unused, indices in stream_geometry('gate', 5)['mapping']], [9] * 4 + [8] * 4)
        self.assertEqual([len(indices) for unused, indices in stream_geometry('down', 5)['mapping']], [10] * 8)

    def test_invalid_scope_fails_closed(self):
        for projection, blocks in (('attention', 5), ('gate', 0), ('gate', 21), ('down', 35),
                ('gate', True), ('gate', 5.0), ('gate', '5')):
            with self.subTest(projection=projection, blocks=blocks), self.assertRaises(ValueError):
                stream_geometry(projection, blocks)

    def fixture(self):
        geometry = stream_geometry('gate', 5)
        source = SimpleNamespace(shape=(1, 1, 1280, 8704), dtype='bf4', layout='tile',
            memory_config=lambda: 'dram', buffer_address=lambda: 1024)
        output = SimpleNamespace(shape=(1, 1, 10880, 144), dtype='uint32', layout='row',
            memory_config=lambda: 'dram', buffer_address=lambda: 2048)
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        operations = SimpleNamespace(bfloat4_b='bf4', uint32='uint32', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='row',
            DRAM_MEMORY_CONFIG='dram', get_device_tensors=Mock(side_effect=lambda value: [value, value]),
            create_global_circular_buffer=Mock())
        return operations, mesh, source, output, geometry

    def test_shape_dtype_layout_geometry_and_alias_guards_precede_allocation(self):
        for mutation in ('grid', 'geometry', 'source_shape', 'source_dtype', 'source_layout', 'source_memory',
                'output_shape', 'output_dtype', 'output_layout', 'output_memory', 'shards', 'alias'):
            operations, mesh, source, output, geometry = self.fixture()
            if mutation == 'grid':
                mesh.compute_with_storage_grid_size = lambda: SimpleNamespace(x=12, y=10)
            elif mutation == 'geometry':
                geometry['per_receiver'] = 3
            elif mutation == 'shards':
                operations.get_device_tensors.side_effect = lambda value: [value]
            elif mutation == 'alias':
                output.buffer_address = source.buffer_address
            else:
                tensor_name, field = mutation.split('_')
                value = source if tensor_name == 'source' else output
                if field == 'memory':
                    value.memory_config = lambda: 'l1'
                else:
                    setattr(value, field, (1, 1, 32, 32) if field == 'shape' else 'invalid')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                prepare_stream(operations, mesh, source, output, geometry)
            operations.create_global_circular_buffer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
