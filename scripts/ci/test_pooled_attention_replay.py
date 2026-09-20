"""The pooled replay reader: the pinned ReplayAttentionReader with its per-bundle page
tables lent by the serving pool instead of uploaded - staged at construction, borrowed,
never freed here - and the family and bundle geometry the pool needs to size them."""

import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from attention_replay import ReplayAttentionReader
from pooled_attention_replay import (PooledReplayAttentionReader, bundle_batches, family_capacities, family_start,
                                     validate_storage)


class FamilyTests(unittest.TestCase):
    """What the serving pool needs to size the reader's page tables before any request."""

    def test_family_capacities_are_the_admitted_chunk_families_cut_to_the_page_table(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(family_capacities(), tuple(range(4096, 16640 + 1, 256)))
            self.assertEqual(len(family_capacities()), 50)
            self.assertEqual(family_capacities(page_width=68), (4096, 4352))
            self.assertEqual(family_capacities(page_width=64), (4096,))
            self.assertEqual(family_capacities(page_width=63), ())
            self.assertEqual(family_capacities(page_width=1024), family_capacities())
            self.assertEqual(family_capacities(short_context=True), (256, 512, 768))
        for width in (0, 68.0, True, -1):
            with self.assertRaises(ValueError):
                family_capacities(page_width=width)

    def test_bundle_batches_follow_the_readers_own_grouping_in_every_family(self):
        self.assertEqual(bundle_batches(16, 4352), (3, 1))
        self.assertEqual(bundle_batches(8, 4352), (2,))
        self.assertEqual(bundle_batches(32, 4352), (3, 3, 2))
        self.assertEqual(bundle_batches(32, 4352, max_group_rows=8), (3, 1))
        self.assertEqual(bundle_batches(16, 4352, max_group_rows=8), (2,))
        self.assertEqual(bundle_batches(8, 256, short_context=True), (2,))
        self.assertEqual((family_start(4352), family_start(256, short_context=True)), (4096, 128))
        with patch.dict('os.environ', {}, clear=True):
            for capacity in family_capacities():
                for rows, batches in ((8, (2,)), (16, (3, 1)), (32, (3, 3, 2))):
                    self.assertEqual(bundle_batches(rows, capacity), batches, (rows, capacity))


def pooled_tables(shapes):
    """Pool-lent page tables: shape, dtype, layout and a stable address per chip."""
    return [SimpleNamespace(shape=shape, dtype='int32', layout='row', value=None,
                            address=(0x1000 + 0x100 * index, 0x9000 + 0x100 * index))
            for index, shape in enumerate(shapes)]


def fake_addresses(operations, value):
    return value.address if hasattr(value, 'address') else (id(value), id(value))


class PooledReaderTests(unittest.TestCase):
    """Pooled, the reader stages the request's page table into the lent tables at
    construction, reads them from its metadata, and never uploads or frees them."""

    def operations(self, copies):
        def copy(host, target):
            target.value = host.value.clone()
            copies.append(target)

        return SimpleNamespace(int32='int32', ROW_MAJOR_LAYOUT='row', SDPAProgramConfig=Mock(),
            from_torch=Mock(side_effect=lambda value, **kwargs: SimpleNamespace(value=value.clone(), **kwargs)),
            copy_host_to_device_tensor=Mock(side_effect=copy), synchronize_device=Mock(),
            ReplicateTensorToMesh=lambda mesh: ('replicate', mesh))

    def build(self, storage, operations=None, rows=16, capacity=4352, prepare='program', **options):
        copies = []
        operations = self.operations(copies) if operations is None else operations
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        upload = Mock(side_effect=lambda value, dtype=None: value)
        with patch('attention_replay.prepare', **(dict(return_value=prepare) if not isinstance(prepare, list) else dict(side_effect=prepare))), \
                patch('pooled_attention_replay.addresses', side_effect=fake_addresses):
            reader = PooledReplayAttentionReader(operations, mesh, rows, capacity, torch.arange(68).reshape(1, 68), upload,
                                                 storage=storage, **options)
        return reader, upload, operations, copies

    def test_the_interception_relies_on_the_pinned_readers_upload_order(self):
        """The base asks `upload` for the 1-D positions word, then per bundle the 2-D int32
        page table and the BF16 mask; only the 2-D int32 requests are answered from the pool."""
        source = inspect.getsource(ReplayAttentionReader.__init__)
        lines = ['self.positions = upload(words, operations.int32)',
                 'pages = upload(pages_host[:, :capacity // 64].repeat(len(bundle), 1).contiguous(), operations.int32)',
                 'mask = upload(torch.zeros(len(bundle), 1, count * 12, capacity, dtype=torch.bfloat16))']
        for line in lines:
            self.assertIn(line, source)
        self.assertEqual(sorted(source.index(line) for line in lines), [source.index(line) for line in lines])
        self.assertEqual(source.count('upload('), 3)
        self.assertTrue(issubclass(PooledReplayAttentionReader, ReplayAttentionReader))

    def test_pooled_tables_are_staged_in_place_borrowed_and_neither_uploaded_nor_freed(self):
        storage = pooled_tables([(3, 68), (1, 68)])
        reader, upload, operations, copies = self.build(storage)
        self.assertEqual([entry[1] for entry in reader.metadata], storage)
        self.assertEqual(reader.borrowed, storage)
        self.assertEqual(len(reader.owned), 3, 'the positions word and the two masks')
        self.assertFalse(any(table is owned for table in storage for owned in reader.owned))
        # The positions word and the two masks are still uploaded; the tables are copied into.
        self.assertEqual(upload.call_count, 3)
        self.assertEqual([call.args[1] if len(call.args) > 1 else None for call in upload.call_args_list],
                         ['int32', None, None])
        self.assertEqual(copies, storage)
        for table, batches in zip(storage, (3, 1), strict=True):
            self.assertTrue(torch.equal(table.value, torch.arange(68).reshape(1, 68).repeat(batches, 1)))
        for source in operations.from_torch.call_args_list:
            self.assertEqual((source.kwargs['dtype'], source.kwargs['layout'], source.kwargs['mesh_mapper']),
                             ('int32', 'row', ('replicate', reader.mesh)))
        operations.synchronize_device.assert_called_once()
        self.assertFalse(reader.failed)
        # The base's surface, untouched: the family start, the positions word, the programs.
        self.assertEqual((reader.start, reader.rows, reader.capacity, len(reader.programs)), (4096, 16, 4352, 2))
        self.assertEqual(reader.positions.tolist(), [4096, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual([len(entry[0]) for entry in reader.metadata], [3, 1])
        released = []
        with patch('attention_replay.release_owned', side_effect=lambda operations, tensors: released.extend(tensors)):
            reader.close()
        self.assertEqual(len(released), 3)
        self.assertFalse(any(value is table for value in released for table in storage))
        self.assertTrue(reader.closed)

    def test_the_base_reader_is_unchanged_and_owns_its_tables(self):
        operations = self.operations([])
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        upload = Mock(side_effect=lambda value, dtype=None: value)
        with patch('attention_replay.prepare', return_value='program'):
            reader = ReplayAttentionReader(operations, mesh, 16, 4352, torch.arange(68).reshape(1, 68), upload)
        self.assertEqual(upload.call_count, 5)
        self.assertEqual(len(reader.owned), 5)
        self.assertFalse(hasattr(reader, 'borrowed'))
        operations.copy_host_to_device_tensor.assert_not_called()
        operations.synchronize_device.assert_not_called()
        with self.assertRaises(TypeError):
            ReplayAttentionReader(operations, mesh, 16, 4352, torch.arange(68).reshape(1, 68), upload, storage=[])

    def test_tables_of_another_geometry_or_none_are_refused_before_any_upload(self):
        copies = []
        operations = self.operations(copies)
        cases = {'one bundle short': [(3, 68)], 'one bundle over': [(3, 68), (1, 68), (1, 68)],
                 'bundles swapped': [(1, 68), (3, 68)], 'narrow': [(3, 67), (1, 68)], 'wide': [(3, 68), (1, 69)]}
        wrong = pooled_tables([(3, 68), (1, 68)])
        wrong[0].dtype = 'uint32'
        cases['dtype'] = wrong
        wrong = pooled_tables([(3, 68), (1, 68)])
        wrong[1].layout = 'tile'
        cases['layout'] = wrong
        for name, storage in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Pooled replay page table'):
                self.build(storage if isinstance(storage[0], SimpleNamespace) else pooled_tables(storage), operations)
        with self.assertRaisesRegex(ValueError, 'needs the lent page tables'):
            self.build(None, operations)
        for options in (dict(rows=4), dict(rows=16.0), dict(capacity=4352.0), dict(max_group_rows=6),
                        dict(short_context=1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.build(pooled_tables([(3, 68), (1, 68)]), operations, **options)
        # A ticket outside the family is the base's refusal, still before anything is lent.
        with self.assertRaises(ValueError):
            self.build(pooled_tables([(3, 60), (1, 60)]), operations, capacity=3840)
        self.assertEqual(copies, [])
        operations.from_torch.assert_not_called()
        self.assertIsNone(validate_storage(operations, None, [[1], [2]], 4352))

    def test_a_failed_staging_poisons_frees_the_uploads_and_keeps_the_pool_tables(self):
        storage = pooled_tables([(3, 68), (1, 68)])
        operations = self.operations([])
        operations.copy_host_to_device_tensor = Mock(side_effect=[None, RuntimeError('copy failed')])
        released = []
        with patch('attention_replay.release_owned', side_effect=lambda operations, tensors: released.extend(tensors)), \
                self.assertRaisesRegex(RuntimeError, 'copy failed'):
            self.build(storage, operations)
        self.assertEqual(len(released), 3)
        self.assertFalse(any(value is table for value in released for table in storage))

    def test_a_failure_inside_the_pinned_constructor_frees_its_uploads_and_keeps_the_pool_tables(self):
        """The base's own failure path calls close() with the lent tables still in `owned`."""
        storage = pooled_tables([(3, 68), (1, 68)])
        operations = self.operations([])
        released = []
        with patch('attention_replay.release_owned', side_effect=lambda operations, tensors: released.extend(tensors)), \
                self.assertRaisesRegex(RuntimeError, 'prepare failed'):
            self.build(storage, operations, prepare=['program', RuntimeError('prepare failed')])
        # The positions word and both masks, not the two tables it had been handed.
        self.assertEqual(len(released), 3)
        self.assertFalse(any(value is table for value in released for table in storage))
        operations.copy_host_to_device_tensor.assert_not_called()

    def test_a_table_that_moves_under_the_staging_is_refused(self):
        storage = pooled_tables([(3, 68), (1, 68)])
        operations = self.operations([])

        def move(host, target):
            target.address = (target.address[0] + 1, target.address[1])

        operations.copy_host_to_device_tensor = Mock(side_effect=move)
        with patch('attention_replay.release_owned'), \
                self.assertRaisesRegex(AssertionError, 'replaced a pooled replay page table'):
            self.build(storage, operations)

    def test_the_base_surface_serves_staging_refresh_and_shared_masks(self):
        storage = pooled_tables([(3, 68), (1, 68)])
        reader, upload, operations, copies = self.build(storage)
        events = []
        with patch('attention_replay.refresh_mask', side_effect=lambda *args: events.append('mask')):
            with reader.shared_masks(16):
                reader.calls += 16
        self.assertEqual(events, ['mask', 'mask'])
        self.assertEqual(reader.refresh_calls, 2)
        for start in (4095, 4337):
            with self.assertRaises(ValueError):
                reader.validate(start)
        reader.validate(4103)
        self.assertEqual(reader.start, 4096)
        self.assertFalse(reader.failed)


if __name__ == '__main__':
    unittest.main()
