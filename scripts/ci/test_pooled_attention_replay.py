"""The pooled replay reader: the pinned ReplayAttentionReader with its per-bundle page
tables lent by the serving pool instead of uploaded - staged at construction, borrowed,
never freed here - and the family and bundle geometry the pool needs to size them."""

import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from attention_replay import ReplayAttentionReader
from pooled_attention_replay import (PackedReplayAttentionReader, PooledReplayAttentionReader, bundle_batches,
                                     family_capacities, family_start, validate_segments, validate_storage)


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


class PackedReaderTests(unittest.TestCase):
    """One pooled reader per packed user, each over its own start word and lent tables;
    a 32-row query dispatched a segment at a time and reassembled in row order (M1b)."""

    SEGMENTS = ((0, 16), (16, 32))

    def operations(self, copies, events=None):
        events = [] if events is None else events

        def copy(host, target):
            target.value = host.value.clone()
            copies.append(target)

        def slice_rows(tensor, start, stop, memory_config):
            events.append(('slice', start, stop, memory_config))
            return SimpleNamespace(shape=(1, stop[1] - start[1], 12, 256), name='rows%d' % start[1])

        def concat(parts, dim, memory_config):
            events.append(('concat', [part.name for part in parts], dim, memory_config))
            return SimpleNamespace(name='joined')

        return SimpleNamespace(int32='int32', ROW_MAJOR_LAYOUT='row', DRAM_MEMORY_CONFIG='dram', SDPAProgramConfig=Mock(),
            from_torch=Mock(side_effect=lambda value, **kwargs: SimpleNamespace(value=value.clone(), **kwargs)),
            copy_host_to_device_tensor=Mock(side_effect=copy), synchronize_device=Mock(),
            ReplicateTensorToMesh=lambda mesh: ('replicate', mesh), slice=Mock(side_effect=slice_rows),
            concat=Mock(side_effect=concat), deallocate=Mock(side_effect=lambda value: events.append(('free', value.name))))

    def tables(self, users=2):
        return [pooled_tables([(3, 68), (1, 68)]) for user in range(users)]

    def hosts(self, users=2):
        return [torch.full((1, 68), 7 + 4 * user, dtype=torch.int32) for user in range(users)]

    def build(self, storage='pooled', operations=None, segments=SEGMENTS, capacity=4352, prepare='program', hosts=None, **options):
        copies, events = [], []
        operations = self.operations(copies, events) if operations is None else operations
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        upload = Mock(side_effect=lambda value, dtype=None: SimpleNamespace(value=value.clone(), dtype=dtype, name='upload',
                                                                            shape=tuple(value.shape)))
        storage = self.tables(len(segments)) if storage == 'pooled' else storage
        with patch('attention_replay.prepare', **(dict(return_value=prepare) if not isinstance(prepare, list) else dict(side_effect=prepare))), \
                patch('pooled_attention_replay.addresses', side_effect=fake_addresses):
            reader = PackedReplayAttentionReader(operations, mesh, segments, capacity, self.hosts(len(segments)) if hosts is None else hosts,
                                                 upload, storage=storage, **options)
        return reader, storage, operations, copies, events

    def test_each_user_gets_its_own_pooled_reader_over_its_own_tables_and_start(self):
        reader, storage, operations, copies, events = self.build()
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue(all(type(own) is PooledReplayAttentionReader for own in reader.readers))
        self.assertEqual((reader.rows, reader.capacity, reader.segments), (32, 4352, self.SEGMENTS))
        self.assertEqual([own.rows for own in reader.readers], [16, 16])
        self.assertEqual(reader.starts, (4096, 4096))
        # user u's reader reads user u's lent tables, staged with user u's own page table
        for user, own in enumerate(reader.readers):
            self.assertEqual([entry[1] for entry in own.metadata], storage[user])
            self.assertEqual(own.borrowed, storage[user])
            for table, batches in zip(storage[user], (3, 1), strict=True):
                self.assertTrue(torch.equal(table.value, torch.full((batches, 68), 7 + 4 * user, dtype=torch.int32)))
            self.assertEqual(own.positions.value.tolist(), [4096, 0, 0, 0, 0, 0, 0, 0])
            self.assertIsNot(own.positions, reader.readers[1 - user].positions)
        self.assertEqual(reader.borrowed, [*storage[0], *storage[1]])
        self.assertEqual(reader.metadata, [*reader.readers[0].metadata, *reader.readers[1].metadata])
        self.assertEqual(copies, [*storage[0], *storage[1]])
        self.assertEqual((reader.calls, reader.refresh_calls, reader.failed, reader.closed, reader.audit), (0, 0, False, False, None))

    def test_the_query_is_dispatched_a_segment_at_a_time_and_reassembled_in_row_order(self):
        reader, storage, operations, copies, events = self.build()
        query = SimpleNamespace(shape=(1, 32, 12, 256))
        keys, values = object(), object()
        served = []

        def attention(mesh, ops, rows, keys, values, metadata, owned, **kwargs):
            served.append((rows.name, [entry[1] for entry in metadata], kwargs))
            result = SimpleNamespace(name='out%s' % rows.name[4:])
            owned.append(result)
            return result

        with patch('attention_replay.addresses', side_effect=lambda operations, value: (id(value), id(value))), \
                patch('attention_replay.refresh_mask', side_effect=lambda *args: events.append(('mask',))), \
                patch('attention_replay.execute', side_effect=attention), patch('attention_replay.release_owned'):
            result = reader(query, keys, values, page_table_tensor='ignored', cur_pos_tensor='ignored', scale=0.0625, memory_config='L1')
        self.assertEqual(result.name, 'joined')
        # user 0's rows [0, 16) through user 0's bundles, user 1's rows [16, 32) through user 1's
        self.assertEqual([entry[:2] for entry in served], [('rows0', storage[0]), ('rows16', storage[1])])
        self.assertTrue(all(entry[2] == dict(scale=0.0625, memory_config='L1') for entry in served))
        self.assertEqual([event for event in events if event[0] == 'slice'],
                         [('slice', (0, 0, 0, 0), (1, 16, 12, 256), 'dram'), ('slice', (0, 16, 0, 0), (1, 32, 12, 256), 'dram')])
        self.assertEqual([event for event in events if event[0] == 'concat'], [('concat', ['out0', 'out16'], 1, 'L1')])
        # masks refreshed per reader (two bundles each) before its attention; every slice and per-user output freed
        self.assertEqual(events.count(('mask',)), 4)
        self.assertEqual(sorted(event[1] for event in events if event[0] == 'free'), ['out0', 'out16', 'rows0', 'rows16'])
        self.assertEqual((reader.calls, [own.calls for own in reader.readers], reader.refresh_calls), (1, [1, 1], 4))
        with self.assertRaisesRegex(ValueError, 'query geometry'):
            reader(SimpleNamespace(shape=(1, 16, 12, 256)), keys, values, scale=0.0625, memory_config='L1')

    def test_a_failing_segment_frees_what_was_built_and_poisons_that_reader(self):
        reader, storage, operations, copies, events = self.build()
        query = SimpleNamespace(shape=(1, 32, 12, 256))
        with patch('attention_replay.addresses', side_effect=lambda operations, value: (id(value), id(value))), \
                patch('attention_replay.refresh_mask'), patch('attention_replay.release_owned'), \
                patch('attention_replay.execute', side_effect=[SimpleNamespace(name='out0'), RuntimeError('sdpa failed')]), \
                self.assertRaisesRegex(RuntimeError, 'sdpa failed'):
            reader(query, object(), object(), scale=0.0625, memory_config='L1')
        self.assertEqual(sorted(event[1] for event in events if event[0] == 'free'), ['out0', 'rows0', 'rows16'])
        self.assertEqual([own.failed for own in reader.readers], [False, True])
        self.assertTrue(reader.failed)
        self.assertEqual(reader.calls, 0)
        reader.failed = False
        self.assertEqual([own.failed for own in reader.readers], [False, False])

    def test_validate_stage_refresh_and_shared_masks_reach_every_reader(self):
        reader, storage, operations, copies, events = self.build()
        reader.validate((4100, 4200))
        for starts in ((4095, 4200), (4100, 4337), (4100,), (4100, 4200, 4300)):
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                reader.validate(starts)
        with patch('attention_replay.addresses', side_effect=lambda operations, value: (id(value), id(value))):
            reader.stage((4100, 4200))
        self.assertEqual(reader.starts, (4100, 4200))
        self.assertEqual([own.positions.value.tolist()[0] for own in reader.readers], [4100, 4200])
        with patch('attention_replay.refresh_mask', side_effect=lambda *args: events.append(('mask',))):
            reader.refresh()
            self.assertEqual(events.count(('mask',)), 4)
            with reader.shared_masks(16):
                for own in reader.readers:
                    own.calls += 16
        self.assertEqual(events.count(('mask',)), 8)
        self.assertEqual(reader.refresh_calls, 8)
        self.assertIsNone(reader.readers[0].mask_scope)
        self.assertFalse(reader.failed)

    def test_close_closes_every_reader_keeps_the_pool_tables_and_refuses_further_use(self):
        reader, storage, operations, copies, events = self.build()
        released = []
        with patch('attention_replay.release_owned', side_effect=lambda operations, tensors: released.extend(tensors)):
            reader.close()
            reader.close()
        self.assertTrue(reader.closed and all(own.closed for own in reader.readers))
        self.assertEqual(len(released), 6, 'two positions words and four masks')
        self.assertFalse(any(value is table for value in released for tables in storage for table in tables))
        for operation in (lambda: reader.validate((4100, 4200)), lambda: reader.refresh(),
                          lambda: reader(SimpleNamespace(shape=(1, 32, 12, 256)), None, None, scale=1.0, memory_config='L1')):
            with self.assertRaises(RuntimeError):
                operation()

    def test_a_failure_building_one_reader_closes_the_ones_built_and_frees_no_pool_table(self):
        storage = self.tables()
        released = []
        with patch('attention_replay.release_owned', side_effect=lambda operations, tensors: released.extend(tensors)), \
                self.assertRaisesRegex(RuntimeError, 'prepare failed'):
            self.build(storage, prepare=['program', 'program', RuntimeError('prepare failed')])
        # the first reader's positions word and two masks, the second's positions word and one mask
        self.assertEqual(len(released), 5)
        self.assertFalse(any(value is table for value in released for tables in storage for table in tables))

    def test_geometry_and_storage_are_checked_before_anything_is_built(self):
        copies = []
        operations = self.operations(copies)
        cases = {'one table set short': dict(storage=self.tables(1)), 'one table set over': dict(storage=self.tables(3)),
                 'one host table short': dict(hosts=self.hosts(1)),
                 'segments not from zero': dict(segments=((16, 32), (32, 48))), 'segments with a gap': dict(segments=((0, 16), (20, 36))),
                 'a four-row segment': dict(segments=((0, 4), (4, 8))),
                 'past the block': dict(segments=((0, 32), (32, 64), (64, 96))),
                 'no segments': dict(segments=()), 'short context': dict(short_context=True),
                 'tables of another geometry': dict(storage=[pooled_tables([(3, 68), (1, 68)]), pooled_tables([(2, 68)])])}
        for name, overrides in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.build(operations=operations, **overrides)
        self.assertEqual(copies, [])
        self.assertEqual(validate_segments(((0, 8), (8, 16), (16, 24), (24, 32))), ((0, 8), (8, 16), (16, 24), (24, 32)))
        self.assertEqual(validate_segments([[0, 32]]), ((0, 32),))
        # 64 rows is the widest block (M3): four 16-row users, or two 32-row ones
        self.assertEqual(validate_segments(((0, 16), (16, 32), (32, 48), (48, 64))), ((0, 16), (16, 32), (32, 48), (48, 64)))
        self.assertEqual(validate_segments(((0, 32), (32, 64))), ((0, 32), (32, 64)))
        with self.assertRaises(ValueError):
            validate_segments(((0, 32), (32, 64), (64, 96)))
        with self.assertRaises(ValueError):
            validate_segments(((0, 32), (32, 64)), block_rows_limit=32)

    def test_four_sixteen_row_users_fill_the_sixty_four_row_block_a_segment_at_a_time(self):
        """The M3 block: four pooled 16-row readers, a (1, 64, 12, 256) query dispatched in
        four slices and reassembled in row order."""
        segments = ((0, 16), (16, 32), (32, 48), (48, 64))
        reader, storage, operations, copies, events = self.build(segments=segments)
        self.assertEqual((reader.rows, reader.segments, len(reader.readers)), (64, segments, 4))
        self.assertEqual([own.rows for own in reader.readers], [16] * 4)
        self.assertEqual(reader.starts, (4096,) * 4)
        for user, own in enumerate(reader.readers):
            self.assertEqual(own.borrowed, storage[user])
            for table, batches in zip(storage[user], (3, 1), strict=True):
                self.assertTrue(torch.equal(table.value, torch.full((batches, 68), 7 + 4 * user, dtype=torch.int32)))
        self.assertEqual(len({id(own.positions) for own in reader.readers}), 4)
        served = []

        def attention(mesh, ops, rows, keys, values, metadata, owned, **kwargs):
            served.append((rows.name, [entry[1] for entry in metadata]))
            result = SimpleNamespace(name='out%s' % rows.name[4:])
            owned.append(result)
            return result

        with patch('attention_replay.addresses', side_effect=lambda operations, value: (id(value), id(value))), \
                patch('attention_replay.refresh_mask', side_effect=lambda *args: events.append(('mask',))), \
                patch('attention_replay.execute', side_effect=attention), patch('attention_replay.release_owned'):
            result = reader(SimpleNamespace(shape=(1, 64, 12, 256)), object(), object(), scale=0.0625, memory_config='L1')
        self.assertEqual(result.name, 'joined')
        self.assertEqual(served, [('rows%d' % first, storage[user]) for user, (first, last) in enumerate(segments)])
        self.assertEqual([event for event in events if event[0] == 'slice'],
                         [('slice', (0, first, 0, 0), (1, last, 12, 256), 'dram') for first, last in segments])
        self.assertEqual([event for event in events if event[0] == 'concat'],
                         [('concat', ['out0', 'out16', 'out32', 'out48'], 1, 'L1')])
        self.assertEqual(events.count(('mask',)), 8, 'two bundles per reader, four readers')
        self.assertEqual((reader.calls, [own.calls for own in reader.readers]), (1, [1] * 4))
        reader.validate((4100, 4200, 4150, 4300))
        with self.assertRaises(ValueError):
            reader.validate((4100, 4200, 4150))
        with self.assertRaisesRegex(ValueError, 'query geometry'):
            reader(SimpleNamespace(shape=(1, 32, 12, 256)), None, None, scale=0.0625, memory_config='L1')

    def test_unpooled_segments_take_the_pinned_reader_and_upload_their_own_tables(self):
        reader, storage, operations, copies, events = self.build(storage=None)
        self.assertTrue(all(type(own) is ReplayAttentionReader for own in reader.readers))
        self.assertEqual(reader.borrowed, [])
        self.assertEqual(copies, [])
        for user, own in enumerate(reader.readers):
            self.assertEqual([tuple(entry[1].shape) for entry in own.metadata], [(3, 68), (1, 68)])
            self.assertTrue(all(bool((entry[1].value == 7 + 4 * user).all()) for entry in own.metadata))
            self.assertEqual(len(own.owned), 5)


class SdpaModesTests(unittest.TestCase):
    """QWEN_FAST_SDPA_MODES ('tail', stage 1; 'share', stage 3; 'slice' and 'readahead', stage 4): the per-bundle
    SDPA configs of a built replay reader carry the grafted factory's q_chunk_size sentinel; unset, nothing
    changes."""

    # The files this change must not touch (their bytes are gate evidence), at HEAD d21cd875+.
    PINNED = {
        'attention_replay.py': '4eff1c51fd42bb04adf68fc40bf74a0cca0cd455c3ae2fc50caf720f5281137a',
        'attention_batch.py': '64e2ed10cdb5f38485ac07178300a784a5ea2700f4c5275601abd6188304c9f0',
        'gdn_multitoken_conv.py': '6a39d5dfe9b48500621a4900aee01f1656c578d09e84f27edd03046abf59f859',
        'attention_mask_replay.py': '3e431742e35a2b94b4a02a60fa334a93a44a471eaefcacd25e52fbafdf03361f',
        'attention_mask_replay.cpp': 'e10cae1d6fe97f9b1509ac5ef918f6e7eda8d51bfbd77dcfd9e95662bb838af8',
        'attention_parallel.py': '7bf5ba445100d184f7b9289fed4c20b4a730dbee29ee382cb97b6c2ad17f0e58',
    }

    def setUp(self):
        import pooled_attention_replay
        patcher = patch.object(pooled_attention_replay, '_binary_checked', [])
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def config(**kwargs):
        return SimpleNamespace(**kwargs)

    def reader(self, batches=(3, 1), short_context=False):
        """A built replay reader's surface: metadata entries of (bundle, pages, mask, config)."""
        operations = SimpleNamespace(SDPAProgramConfig=Mock(side_effect=self.config))
        metadata = [([dict(rows=4, offset=4 * index)] * count, SimpleNamespace(name='pages%d' % index),
                     SimpleNamespace(name='mask%d' % index),
                     self.config(compute_with_storage_grid_size=(11, 10), exp_approx_mode=False,
                                 q_chunk_size=0, k_chunk_size=256))
                    for index, count in enumerate(batches)]
        return SimpleNamespace(operations=operations, short_context=short_context, rows=16, capacity=4352,
                               mesh=SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10)),
                               metadata=metadata)

    def test_the_environment_parses_tail_and_share_and_refuses_narrow_by_name(self):
        from pooled_attention_replay import SDPA_MODES_ENV, sdpa_modes
        self.assertEqual(sdpa_modes({}), frozenset())
        self.assertEqual(sdpa_modes({SDPA_MODES_ENV: ''}), frozenset())
        self.assertEqual(sdpa_modes({SDPA_MODES_ENV: ' , '}), frozenset())
        self.assertEqual(sdpa_modes({SDPA_MODES_ENV: 'tail'}), frozenset({'tail'}))
        self.assertEqual(sdpa_modes({SDPA_MODES_ENV: ' tail ,tail'}), frozenset({'tail'}))
        self.assertEqual(sdpa_modes({SDPA_MODES_ENV: 'tail,share'}), frozenset({'tail', 'share'}))
        self.assertEqual(sdpa_modes({SDPA_MODES_ENV: 'share'}), frozenset({'share'}))
        for value, message in (('tail,narrow', 'narrow not in this build'), ('tail,share,narrow', 'stage 1b'),
                               ('narrow', 'stage 1b'), ('tial', 'Unknown'), ('tail,1', 'Unknown'), ('shared', 'Unknown')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                sdpa_modes({SDPA_MODES_ENV: value})
        with patch.dict('os.environ', {SDPA_MODES_ENV: 'tail'}):
            self.assertEqual(sdpa_modes(), frozenset({'tail'}))

    def test_tail_flags_every_bundle_and_share_only_multi_entry_ones(self):
        from pooled_attention_replay import mode_flags
        self.assertEqual([mode_flags({'tail'}, batches) for batches in (3, 1)], [0x1, 0x1])
        self.assertEqual([mode_flags({'tail', 'share'}, batches) for batches in (3, 1)], [0x3, 0x1])
        self.assertEqual(mode_flags(frozenset(), 3), 0)

    def test_tail_rewrites_each_config_with_the_sentinel_and_keeps_every_tensor(self):
        from pooled_attention_replay import SDPA_MODES_MARKER, apply_sdpa_modes
        reader = self.reader()
        before = list(reader.metadata)
        lines = []
        result = apply_sdpa_modes(reader, frozenset({'tail'}), log=lines.append,
                                  binary_check=lambda markers: ('/opt/x/_ttnncpp.so', True))
        self.assertEqual(result, (0x1, 0x1))
        self.assertEqual(reader.sdpa_modes_applied, (0x1, 0x1))
        for old, new in zip(before, reader.metadata, strict=True):
            self.assertIs(new[0], old[0])
            self.assertIs(new[1], old[1], 'the page table (pool-lent or owned) is the same object')
            self.assertIs(new[2], old[2], 'the wide mask is the same object')
            self.assertEqual(vars(new[3]), dict(compute_with_storage_grid_size=(11, 10), exp_approx_mode=False,
                                                q_chunk_size=0x51DEC001, k_chunk_size=256))
            self.assertEqual(vars(new[3]), dict(vars(old[3]), q_chunk_size=0x51DEC001))
        self.assertEqual(lines, [
            SDPA_MODES_MARKER + ' binary /opt/x/_ttnncpp.so carries the [QWEN-SDPA] branch',
            SDPA_MODES_MARKER + " modes=tail rows=16 capacity=4352 bundles=[3, 1] flags=['0x1', '0x1'] mask=wide"])
        # A second apply is a no-op, and the binary is checked once per process.
        self.assertEqual(apply_sdpa_modes(reader, frozenset({'tail'}), log=lines.append,
                                          binary_check=lambda markers: self.fail('checked twice')), (0x1, 0x1))
        self.assertEqual(len(lines), 2)
        other = self.reader(batches=(2,))
        apply_sdpa_modes(other, frozenset({'tail'}), log=lines.append,
                         binary_check=lambda markers: self.fail('checked twice'))
        self.assertEqual(lines[-1], SDPA_MODES_MARKER + " modes=tail rows=16 capacity=4352 bundles=[2] flags=['0x1'] mask=wide")

    def test_no_modes_is_a_no_op_that_reads_nothing(self):
        from pooled_attention_replay import apply_sdpa_modes
        reader = self.reader()
        before = list(reader.metadata)
        self.assertIsNone(apply_sdpa_modes(reader, frozenset(), log=self.fail, binary_check=self.fail))
        self.assertEqual(reader.metadata, before)
        self.assertFalse(hasattr(reader, 'sdpa_modes_applied'))
        reader.operations.SDPAProgramConfig.assert_not_called()

    def test_an_old_binary_short_context_or_a_later_mode_applies_nothing(self):
        from pooled_attention_replay import _binary_checked, apply_sdpa_modes
        reader = self.reader()
        before = list(reader.metadata)
        with self.assertRaisesRegex(RuntimeError, 'lacks the .QWEN-SDPA. factory branch'):
            apply_sdpa_modes(reader, frozenset({'tail'}), log=self.fail,
                             binary_check=lambda markers: ('/old/_ttnncpp.so', False))
        self.assertEqual(_binary_checked, [])
        with self.assertRaisesRegex(ValueError, 'long-context only'):
            apply_sdpa_modes(self.reader(short_context=True), frozenset({'tail'}), log=self.fail,
                             binary_check=lambda markers: ('/x/_ttnncpp.so', True))
        for modes in ({'tail', 'narrow'}, {'share', 'narrow'}, {'wide'}):
            with self.subTest(modes=modes), self.assertRaisesRegex(ValueError, 'not in this build'):
                apply_sdpa_modes(reader, frozenset(modes), log=self.fail,
                                 binary_check=lambda markers: ('/x/_ttnncpp.so', True))
        self.assertEqual(reader.metadata, before)
        self.assertFalse(hasattr(reader, 'sdpa_modes_applied'))
        reader.operations.SDPAProgramConfig.assert_not_called()

    def test_the_config_is_the_pinned_readers_own_construction_but_the_sentinel(self):
        """If attention_replay.py ever builds its config differently, the rewrite must follow."""
        import pooled_attention_replay
        pinned = inspect.getsource(ReplayAttentionReader.__init__)
        self.assertIn('config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),', pinned)
        self.assertIn('exp_approx_mode=False, q_chunk_size=0, k_chunk_size=256)', pinned)
        ours = inspect.getsource(pooled_attention_replay.apply_sdpa_modes)
        self.assertIn('config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),', ours)
        self.assertIn('exp_approx_mode=False, q_chunk_size=QWEN_DECODE_MAGIC | flags, k_chunk_size=256)', ours)

    def test_the_binary_check_reads_the_one_mapped_ttnncpp(self):
        import tempfile
        from pooled_attention_replay import loaded_binary_has_modes, required_binary_markers
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graft, stock = root / 'graft' / '_ttnncpp.so', root / 'stock' / '_ttnncpp.so'
            for path, body in ((graft, b'x' * 4096 + b'[QWEN-SDPA] flags={:#x} B={}' + b'y' * 64),
                               (stock, b'x' * 4096 + b'[QWEN-SDPA] unknown flags {:#x}')):
                path.parent.mkdir()
                path.write_bytes(body)
            maps = root / 'maps'

            def mapped(*paths, markers=None):
                maps.write_text(''.join('7f00-7f10 r-xp 00000000 08:01 1 %s%s' % (path, chr(10)) for path in paths)
                                + '7f20-7f30 r--p 00000000 08:01 2 /usr/lib/libc.so.6' + chr(10))
                if markers is None:
                    return loaded_binary_has_modes(maps=str(maps))
                return loaded_binary_has_modes(markers, maps=str(maps))

            self.assertEqual(mapped(graft, graft), (str(graft), True))
            self.assertEqual(mapped(stock), (str(stock), False))
            # 'share' also needs the stage-3 factory (F9's format literal): K64e has only the first.
            share = required_binary_markers(frozenset({'tail', 'share'}))
            self.assertEqual(share, (b'[QWEN-SDPA] flags=', b'[QWEN-SDPA] KV-share twin bands'))
            self.assertEqual(required_binary_markers(frozenset({'tail'})), (b'[QWEN-SDPA] flags=',))
            self.assertEqual(mapped(graft, markers=share), (str(graft), False))
            stage3 = root / 'stage3' / '_ttnncpp.so'
            stage3.parent.mkdir()
            stage3.write_bytes(b'x' * 4096 + b'[QWEN-SDPA] flags={:#x} B={}' + b'y' * 64
                               + b'[QWEN-SDPA] KV-share twin bands do not fit the {}x{} grid for B={}')
            self.assertEqual(mapped(stage3, markers=share), (str(stage3), True))
            self.assertEqual(mapped(stage3), (str(stage3), True))
            for paths in ((), (graft, stock)):
                with self.subTest(paths=paths), self.assertRaisesRegex(RuntimeError, 'exactly one mapped'):
                    mapped(*paths)

    def test_the_pooled_reader_applies_the_modes_from_the_environment_before_returning(self):
        from pooled_attention_replay import SDPA_MODES_ENV
        lines = []
        with patch.dict('os.environ', {SDPA_MODES_ENV: 'tail'}), \
                patch('pooled_attention_replay._pindiag', side_effect=lines.append), \
                patch('pooled_attention_replay.loaded_binary_has_modes', return_value=('/g/_ttnncpp.so', True)):
            reader, upload, operations, copies = PooledReaderTests.build(PooledReaderTests(), pooled_tables([(3, 68), (1, 68)]))
        self.assertEqual(reader.sdpa_modes_applied, (0x1, 0x1))
        calls = operations.SDPAProgramConfig.call_args_list
        self.assertEqual([call.kwargs['q_chunk_size'] for call in calls], [0, 0, 0x51DEC001, 0x51DEC001])
        self.assertEqual(len(lines), 2)

    def test_unset_the_pooled_and_packed_readers_keep_the_pinned_configs(self):
        with patch.dict('os.environ', {}, clear=True), patch('pooled_attention_replay._pindiag', side_effect=self.fail), \
                patch('pooled_attention_replay.loaded_binary_has_modes', side_effect=self.fail):
            reader, upload, operations, copies = PooledReaderTests.build(PooledReaderTests(), pooled_tables([(3, 68), (1, 68)]))
            packed, storage, packed_operations, packed_copies, events = PackedReaderTests.build(PackedReaderTests(), storage=None)
        self.assertFalse(hasattr(reader, 'sdpa_modes_applied'))
        self.assertEqual([call.kwargs['q_chunk_size'] for call in operations.SDPAProgramConfig.call_args_list], [0, 0])
        self.assertFalse(any(hasattr(own, 'sdpa_modes_applied') for own in packed.readers))
        self.assertEqual([call.kwargs['q_chunk_size'] for call in packed_operations.SDPAProgramConfig.call_args_list], [0] * 4)

    def test_the_packed_block_applies_the_modes_to_every_user_pooled_or_not(self):
        from pooled_attention_replay import SDPA_MODES_ENV
        for storage in ('pooled', None):
            lines = []
            with self.subTest(storage=storage), patch.dict('os.environ', {SDPA_MODES_ENV: 'tail'}), \
                    patch('pooled_attention_replay._binary_checked', []), \
                    patch('pooled_attention_replay._pindiag', side_effect=lines.append), \
                    patch('pooled_attention_replay.loaded_binary_has_modes', return_value=('/g/_ttnncpp.so', True)):
                packed, tables, operations, copies, events = PackedReaderTests.build(
                    PackedReaderTests(), storage=storage, segments=((0, 16), (16, 32), (32, 48), (48, 64)))
                self.assertEqual([own.sdpa_modes_applied for own in packed.readers], [(0x1, 0x1)] * 4)
                sentinels = [call.kwargs['q_chunk_size'] for call in operations.SDPAProgramConfig.call_args_list]
                self.assertEqual(sentinels.count(0x51DEC001), 8, 'two bundles per user, four users, once each')
                self.assertEqual(sum(_is_modes_marker(line) for line in lines), 5, 'one binary line, one per reader')
                if storage == 'pooled':
                    self.assertEqual([entry[1] for entry in packed.metadata], [t for user in tables for t in user])

    def test_a_bad_value_refuses_the_packed_block_before_any_reader_is_built(self):
        from pooled_attention_replay import SDPA_MODES_ENV
        with patch.dict('os.environ', {SDPA_MODES_ENV: 'tail,narrow'}), self.assertRaisesRegex(ValueError, 'stage 1b'):
            PackedReaderTests.build(PackedReaderTests(), storage=None)
        with patch.dict('os.environ', {SDPA_MODES_ENV: 'tail'}), \
                patch('pooled_attention_replay.loaded_binary_has_modes', return_value=('/old/_ttnncpp.so', False)), \
                patch('attention_replay.release_owned') as released, \
                self.assertRaisesRegex(RuntimeError, 'lacks the .QWEN-SDPA. factory branch'):
            PackedReaderTests.build(PackedReaderTests(), storage=None)
        released.assert_called()

    def test_share_flags_only_bundles_of_more_than_one_entry(self):
        """0x2 only where the bundle has twins: 4-row (3, 1) -> 0x3, 0x1; 8-row (2,) -> 0x3."""
        from pooled_attention_replay import SDPA_MODES_MARKER, apply_sdpa_modes, required_binary_markers
        checked = []

        def check(markers):
            checked.append(markers)
            return '/g/_ttnncpp.so', True

        for batches, modes, expected in (((3, 1), {'tail', 'share'}, (0x3, 0x1)), ((2,), {'tail', 'share'}, (0x3,)),
                                         ((2,), {'share'}, (0x2,)), ((1,), {'share'}, (0x0,))):
            with self.subTest(batches=batches, modes=modes):
                reader = self.reader(batches=batches)
                before = list(reader.metadata)
                lines = []
                self.assertEqual(apply_sdpa_modes(reader, frozenset(modes), log=lines.append, binary_check=check), expected)
                for old, new, flags in zip(before, reader.metadata, expected, strict=True):
                    self.assertIs(new[1], old[1], 'the page table is the same object')
                    self.assertIs(new[2], old[2], 'the wide mask is the same object')
                    if flags:
                        self.assertEqual(vars(new[3]), dict(vars(old[3]), q_chunk_size=0x51DEC000 | flags))
                    else:
                        self.assertIs(new[3], old[3], 'a single-entry bundle under share alone keeps the pinned config')
                self.assertEqual(lines[-1], "%s modes=%s rows=16 capacity=4352 bundles=%s flags=%s mask=wide" % (
                    SDPA_MODES_MARKER, ','.join(sorted(modes)), list(batches), ['0x%x' % value for value in expected]))
        # Both mode sets need the same two markers: checked once per process.
        self.assertEqual(checked, [required_binary_markers(frozenset({'share'}))])

    def test_share_on_a_stage_one_binary_is_refused_before_any_config_changes(self):
        from pooled_attention_replay import _binary_checked, apply_sdpa_modes
        reader = self.reader(batches=(2,))
        before = list(reader.metadata)

        def stage1(markers):
            return '/k64e/_ttnncpp.so', markers == (b'[QWEN-SDPA] flags=',)

        with self.assertRaisesRegex(RuntimeError, 'needs the stage-3 .QWEN-SDPA. KV-share factory branch'):
            apply_sdpa_modes(reader, frozenset({'tail', 'share'}), log=self.fail, binary_check=stage1)
        self.assertEqual(reader.metadata, before)
        self.assertFalse(hasattr(reader, 'sdpa_modes_applied'))
        self.assertEqual(_binary_checked, [])
        lines = []
        self.assertEqual(apply_sdpa_modes(reader, frozenset({'tail'}), log=lines.append, binary_check=stage1), (0x1,))
        self.assertEqual(lines[0], '[PINDIAG] sdpa qwen-modes binary /k64e/_ttnncpp.so carries the [QWEN-SDPA] branch')

    def test_the_binary_line_names_the_kv_share_branch(self):
        from pooled_attention_replay import apply_sdpa_modes
        lines = []
        apply_sdpa_modes(self.reader(batches=(2,)), frozenset({'tail', 'share'}), log=lines.append,
                         binary_check=lambda markers: ('/k64f/_ttnncpp.so', True))
        self.assertEqual(lines, [
            '[PINDIAG] sdpa qwen-modes binary /k64f/_ttnncpp.so carries the [QWEN-SDPA] branch with KV share',
            "[PINDIAG] sdpa qwen-modes modes=share,tail rows=16 capacity=4352 bundles=[2] flags=['0x3'] mask=wide"])

    def test_slice_and_readahead_parse_and_readahead_needs_share(self):
        from pooled_attention_replay import SDPA_MODES_ENV, sdpa_modes
        self.assertEqual(sdpa_modes({SDPA_MODES_ENV: 'tail,share,slice'}), frozenset({'tail', 'share', 'slice'}))
        self.assertEqual(sdpa_modes({SDPA_MODES_ENV: 'readahead,share,slice,tail'}),
                         frozenset({'tail', 'share', 'slice', 'readahead'}))
        self.assertEqual(sdpa_modes({SDPA_MODES_ENV: 'slice'}), frozenset({'slice'}))
        for value in ('readahead', 'tail,slice,readahead'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'readahead needs share'):
                sdpa_modes({SDPA_MODES_ENV: value})

    def test_slice_flags_only_groups_the_factory_saves_a_tile_on(self):
        """The stage-4 rule (K1 design section 2): 6-8-row groups 3 -> 2 tiles, 16 rows 6 -> 3; 1-5 rows none."""
        from pooled_attention_replay import mode_flags, q_slice_saves
        self.assertEqual([rows for rows in range(1, 9) if q_slice_saves(rows)], [6, 7, 8])
        self.assertTrue(q_slice_saves(16))
        modes = frozenset({'tail', 'share', 'slice', 'readahead'})
        self.assertEqual(mode_flags(modes, 2, 8), 0xF)
        self.assertEqual(mode_flags(modes - {'readahead'}, 2, 8), 0x7)
        self.assertEqual(mode_flags(modes, 1, 8), 0x5, '0x2 and 0x8 are inert at one entry')
        self.assertEqual(mode_flags(modes, 3, 4), 0xB, 'no saving at 4-row groups: no 0x4')
        self.assertEqual(mode_flags(modes, 1, 4), 0x1)
        self.assertEqual(mode_flags(modes, 2), 0xB, 'no rows: no slice')
        self.assertEqual(mode_flags(frozenset({'tail', 'share'}), 2, 8), 0x3, 'the served modes are unchanged')

    def test_slice_on_four_row_groups_keeps_the_served_flags(self):
        from pooled_attention_replay import apply_sdpa_modes
        lines = []
        reader = self.reader(batches=(3, 1))
        self.assertEqual(apply_sdpa_modes(reader, frozenset({'tail', 'share', 'slice'}), log=lines.append,
                                          binary_check=lambda markers: ('/k64i/_ttnncpp.so', True)), (0x3, 0x1))
        self.assertEqual(lines, [
            '[PINDIAG] sdpa qwen-modes binary /k64i/_ttnncpp.so carries the [QWEN-SDPA] branch with KV share and the '
            'stage-4 q-slice',
            "[PINDIAG] sdpa qwen-modes modes=share,slice,tail rows=16 capacity=4352 bundles=[3, 1] flags=['0x3', '0x1'] "
            "mask=wide"])

    def test_slice_on_a_stage_three_binary_is_refused_before_any_config_changes(self):
        from pooled_attention_replay import QWEN_SDPA_SLICE_MARKER, _binary_checked, apply_sdpa_modes, required_binary_markers
        self.assertEqual(required_binary_markers(frozenset({'tail', 'share', 'slice'})),
                         (b'[QWEN-SDPA] flags=', b'[QWEN-SDPA] KV-share twin bands', b'[QWEN-SDPA] q-slice rows_per_kv='))
        self.assertEqual(required_binary_markers(frozenset({'share', 'readahead'}))[-1], QWEN_SDPA_SLICE_MARKER)
        reader = self.reader(batches=(2,))
        before = list(reader.metadata)

        def stage3(markers):
            return '/k64g/_ttnncpp.so', QWEN_SDPA_SLICE_MARKER not in markers

        for modes in ({'tail', 'share', 'slice'}, {'tail', 'share', 'readahead'}):
            with self.subTest(modes=modes),                     self.assertRaisesRegex(RuntimeError, 'needs the stage-4 .QWEN-SDPA. q-slice factory .K64i onward.'):
                apply_sdpa_modes(reader, frozenset(modes), log=self.fail, binary_check=stage3)
        self.assertEqual(reader.metadata, before)
        self.assertFalse(hasattr(reader, 'sdpa_modes_applied'))
        self.assertEqual(_binary_checked, [])

    def test_the_packed_block_at_eight_row_groups_slices_and_reads_ahead_for_every_user(self):
        """M3NATIVE_REPLAY_GROUP_ROWS=8 + tail,share,slice,readahead: each T16 user is one batch-2 bundle -> 0xF."""
        from pooled_attention_replay import SDPA_MODES_ENV
        lines = []
        with patch.dict('os.environ', {SDPA_MODES_ENV: 'tail,share,slice,readahead', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}),                 patch('pooled_attention_replay._binary_checked', []),                 patch('pooled_attention_replay._pindiag', side_effect=lines.append),                 patch('pooled_attention_replay.loaded_binary_has_modes', return_value=('/g/_ttnncpp.so', True)) as check:
            packed, tables, operations, copies, events = PackedReaderTests.build(
                PackedReaderTests(), storage=None, segments=((0, 16), (16, 32), (32, 48), (48, 64)), max_group_rows=8)
        self.assertEqual([own.sdpa_modes_applied for own in packed.readers], [(0xF,)] * 4)
        sentinels = [call.kwargs['q_chunk_size'] for call in operations.SDPAProgramConfig.call_args_list]
        self.assertEqual(sentinels.count(0x51DEC00F), 4)
        check.assert_called_once_with((b'[QWEN-SDPA] flags=', b'[QWEN-SDPA] KV-share twin bands',
                                       b'[QWEN-SDPA] q-slice rows_per_kv='))
        self.assertEqual(sum(_is_modes_marker(line) for line in lines), 5)
        self.assertIn("modes=readahead,share,slice,tail rows=16 capacity=4352 bundles=[2] flags=['0xf'] mask=wide",
                      lines[-1])

    def test_extent_parses_needs_tail_and_sets_0x20_on_every_bundle(self):
        """K64j (optimisation/ttnn-op/k64j): 'extent' is flag 0x20 and needs 'tail' (the factory refuses 0x20 without
        0x1). It rides on every bundle, whatever its entries or rows, beside the flags the other modes set."""
        from pooled_attention_replay import QWEN_RUNTIME_EXTENT, SDPA_MODE_NAMES, SDPA_MODES_ENV, mode_flags, sdpa_modes
        self.assertEqual(QWEN_RUNTIME_EXTENT, 0x20)
        self.assertEqual(SDPA_MODE_NAMES, ('tail', 'share', 'slice', 'readahead', 'extent'))
        self.assertEqual(sdpa_modes({SDPA_MODES_ENV: 'tail,extent'}), frozenset({'tail', 'extent'}))
        self.assertEqual(sdpa_modes({SDPA_MODES_ENV: 'extent,share,slice,readahead,tail'}),
                         frozenset({'tail', 'share', 'slice', 'readahead', 'extent'}))
        for value in ('extent', 'share,extent', 'slice,share,readahead,extent'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'extent needs tail'):
                sdpa_modes({SDPA_MODES_ENV: value})
        modes = frozenset({'tail', 'extent'})
        self.assertEqual([mode_flags(modes, batches) for batches in (3, 1)], [0x21, 0x21])
        every = frozenset({'tail', 'share', 'slice', 'readahead', 'extent'})
        self.assertEqual(mode_flags(every, 2, 8), 0x2F)
        self.assertEqual(mode_flags(every, 1, 8), 0x25, '0x2 and 0x8 are inert at one entry; 0x20 is not')
        self.assertEqual(mode_flags(every, 3, 4), 0x2B)
        self.assertEqual(mode_flags(every - {'extent'}, 2, 8), 0xF, 'unchanged without extent')

    def test_extent_needs_the_k64j_binary_literal(self):
        from pooled_attention_replay import QWEN_SDPA_EXTENT_MARKER, required_binary_markers
        self.assertEqual(QWEN_SDPA_EXTENT_MARKER, b'[QWEN-SDPA] runtime-extent entries=')
        self.assertEqual(required_binary_markers(frozenset({'tail', 'extent'})),
                         (b'[QWEN-SDPA] flags=', QWEN_SDPA_EXTENT_MARKER))
        self.assertEqual(required_binary_markers(frozenset({'tail', 'share', 'slice', 'extent'})),
                         (b'[QWEN-SDPA] flags=', b'[QWEN-SDPA] KV-share twin bands', b'[QWEN-SDPA] q-slice rows_per_kv=',
                          QWEN_SDPA_EXTENT_MARKER))
        self.assertNotIn(QWEN_SDPA_EXTENT_MARKER, required_binary_markers(frozenset({'tail', 'share', 'slice', 'readahead'})))
        # The factory's own F22 format literal (optimisation/ttnn-op/k64j/apply_factory_k64j.py).
        factory = Path(__file__).resolve().parents[2] / 'optimisation' / 'ttnn-op' / 'k64j' / 'apply_factory_k64j.py'
        self.assertIn("EXTENT_LOG_MARKER = '%s'" % QWEN_SDPA_EXTENT_MARKER.decode(), factory.read_text(encoding='utf-8'))

    def test_extent_is_refused_on_a_reader_without_runtime_extent_before_anything_is_checked(self):
        """The pooled and packed readers stage wide masks and no cur_pos words: the factory would refuse 0x20 at the
        first capture, so the mode is refused here, before the binary check and before any config changes."""
        from pooled_attention_replay import _binary_checked, apply_sdpa_modes
        reader = self.reader(batches=(3, 1))
        before = list(reader.metadata)
        with self.assertRaisesRegex(ValueError, 'extent .flag 0x20, K64j. needs a replay reader that stages one cur_pos word'):
            apply_sdpa_modes(reader, frozenset({'tail', 'extent'}), log=self.fail, binary_check=self.fail)
        self.assertEqual(reader.metadata, before)
        self.assertFalse(hasattr(reader, 'sdpa_modes_applied'))
        self.assertEqual(_binary_checked, [])
        reader.operations.SDPAProgramConfig.assert_not_called()

    def test_extent_on_a_k64i_binary_is_refused_and_on_k64j_sets_0x21(self):
        from pooled_attention_replay import QWEN_SDPA_EXTENT_MARKER, _binary_checked, apply_sdpa_modes
        reader = self.reader(batches=(3, 1))
        reader.runtime_extent = True
        before = list(reader.metadata)

        def k64i(markers):
            return '/k64i/_ttnncpp.so', QWEN_SDPA_EXTENT_MARKER not in markers

        with self.assertRaisesRegex(RuntimeError, 'needs the K64j .QWEN-SDPA. runtime-extent factory; /k64i/_ttnncpp.so lacks it'):
            apply_sdpa_modes(reader, frozenset({'tail', 'extent'}), log=self.fail, binary_check=k64i)
        self.assertEqual(reader.metadata, before)
        self.assertEqual(_binary_checked, [])
        lines = []
        self.assertEqual(apply_sdpa_modes(reader, frozenset({'tail', 'extent'}), log=lines.append,
                                          binary_check=lambda markers: ('/k64j/_ttnncpp.so', True)), (0x21, 0x21))
        for old, new in zip(before, reader.metadata, strict=True):
            self.assertEqual(vars(new[3]), dict(vars(old[3]), q_chunk_size=0x51DEC021))
        self.assertEqual(lines, [
            '[PINDIAG] sdpa qwen-modes binary /k64j/_ttnncpp.so carries the [QWEN-SDPA] branch and the K64j runtime extent',
            "[PINDIAG] sdpa qwen-modes modes=extent,tail rows=16 capacity=4352 bundles=[3, 1] flags=['0x21', '0x21'] mask=narrow"])

    def test_the_packed_block_at_eight_row_groups_shares_every_user(self):
        """M3NATIVE_REPLAY_GROUP_ROWS=8 + tail,share: each T16 user is one batch-2 bundle -> 0x3."""
        from pooled_attention_replay import SDPA_MODES_ENV
        lines = []
        with patch.dict('os.environ', {SDPA_MODES_ENV: 'tail,share', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}), \
                patch('pooled_attention_replay._binary_checked', []), \
                patch('pooled_attention_replay._pindiag', side_effect=lines.append), \
                patch('pooled_attention_replay.loaded_binary_has_modes', return_value=('/g/_ttnncpp.so', True)) as check:
            packed, tables, operations, copies, events = PackedReaderTests.build(
                PackedReaderTests(), storage=None, segments=((0, 16), (16, 32), (32, 48), (48, 64)), max_group_rows=8)
        self.assertEqual([own.sdpa_modes_applied for own in packed.readers], [(0x3,)] * 4)
        sentinels = [call.kwargs['q_chunk_size'] for call in operations.SDPAProgramConfig.call_args_list]
        self.assertEqual(sentinels.count(0x51DEC003), 4)
        check.assert_called_once_with((b'[QWEN-SDPA] flags=', b'[QWEN-SDPA] KV-share twin bands'))
        self.assertEqual(sum(_is_modes_marker(line) for line in lines), 5)
        self.assertIn("modes=share,tail rows=16 capacity=4352 bundles=[2] flags=['0x3'] mask=wide", lines[-1])

    def test_the_pinned_files_keep_their_bytes(self):
        import hashlib
        here = Path(__file__).resolve().parent
        for name, digest in self.PINNED.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((here / name).read_bytes()).hexdigest(), digest)


def _is_modes_marker(line):
    from pooled_attention_replay import SDPA_MODES_MARKER
    return line.startswith(SDPA_MODES_MARKER)


if __name__ == '__main__':
    unittest.main()
