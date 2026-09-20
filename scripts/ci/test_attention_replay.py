import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from attention_replay import ReplayAttentionReader, bundle_batches, family_capacities, validate_storage


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

    def build(self, storage, operations=None, rows=16, capacity=4352):
        copies = []
        operations = self.operations(copies) if operations is None else operations
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        upload = Mock(side_effect=lambda value, dtype=None: value)
        with patch('attention_replay.prepare', return_value='program'), \
                patch('attention_replay.addresses', side_effect=fake_addresses):
            reader = ReplayAttentionReader(operations, mesh, rows, capacity, torch.arange(68).reshape(1, 68), upload,
                                           storage=storage)
        return reader, upload, operations, copies

    def test_pooled_tables_are_staged_in_place_borrowed_and_neither_uploaded_nor_freed(self):
        storage = pooled_tables([(3, 68), (1, 68)])
        reader, upload, operations, copies = self.build(storage)
        self.assertEqual([entry[1] for entry in reader.metadata], storage)
        self.assertEqual(reader.borrowed, storage)
        self.assertFalse(any(table is owned for table in storage for owned in reader.owned))
        # The positions word and the two masks are still uploaded; the tables are copied into.
        self.assertEqual(upload.call_count, 3)
        self.assertEqual(copies, storage)
        for table, batches in zip(storage, (3, 1), strict=True):
            self.assertTrue(torch.equal(table.value, torch.arange(68).reshape(1, 68).repeat(batches, 1)))
        for source, (bundle, pages, mask, config) in zip(operations.from_torch.call_args_list, reader.metadata, strict=True):
            self.assertEqual((source.kwargs['dtype'], source.kwargs['layout'], source.kwargs['mesh_mapper']),
                             ('int32', 'row', ('replicate', reader.mesh)))
        operations.synchronize_device.assert_called_once()
        self.assertFalse(reader.failed)
        released = []
        with patch('attention_replay.release_owned', side_effect=lambda operations, tensors: released.extend(tensors)):
            reader.close()
        self.assertEqual(len(released), 3)
        self.assertFalse(any(value is table for value in released for table in storage))

    def test_an_unpooled_reader_owns_its_tables_as_before(self):
        reader, upload, operations, copies = self.build(None)
        self.assertEqual(reader.borrowed, [])
        self.assertEqual(upload.call_count, 5)
        self.assertEqual(copies, [])
        operations.copy_host_to_device_tensor.assert_not_called()
        operations.synchronize_device.assert_not_called()
        self.assertEqual(len(reader.owned), 5)

    def test_tables_of_another_geometry_are_refused_before_any_copy(self):
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

    def test_a_table_that_moves_under_the_staging_is_refused(self):
        storage = pooled_tables([(3, 68), (1, 68)])
        operations = self.operations([])

        def move(host, target):
            target.address = (target.address[0] + 1, target.address[1])

        operations.copy_host_to_device_tensor = Mock(side_effect=move)
        with patch('attention_replay.release_owned'), self.assertRaisesRegex(AssertionError, 'replaced a pooled replay page table'):
            self.build(storage, operations)


class ReplayReaderTests(unittest.TestCase):
    def test_short_masks_are_fully_initialized_inside_each_replay(self):
        events = []
        operations = SimpleNamespace(int32='int32', SDPAProgramConfig=Mock(),
            full_like=Mock(side_effect=lambda *args, **kwargs: events.append('fill')))
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        with patch('attention_replay.prepare', return_value='program'):
            reader = ReplayAttentionReader(operations, mesh, 8, 512, torch.arange(8).reshape(1, 8),
                lambda value, dtype=None: value, short_context=True)
        with patch('attention_replay.refresh_mask', side_effect=lambda *args: events.append('tail')):
            reader.refresh()
            reader.refresh()
        self.assertEqual(events, ['fill', 'tail', 'fill', 'tail'])
        self.assertEqual(operations.full_like.call_count, 2)
        for call in operations.full_like.call_args_list:
            self.assertIs(call.args[0], reader.metadata[0][2])
            self.assertIs(call.kwargs['optional_tensor'], reader.metadata[0][2])
            self.assertEqual(call.args[1], 0.0)

    def test_short_reader_uses_128_start_and_preserves_fixed_masks(self):
        operations = SimpleNamespace(int32='int32', SDPAProgramConfig=Mock())
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        with patch('attention_replay.prepare', return_value='program') as prepare:
            reader = ReplayAttentionReader(operations, mesh, 8, 256, torch.arange(4).reshape(1, 4),
                lambda value, dtype=None: value, short_context=True)
        self.assertEqual(reader.start, 128)
        self.assertEqual(reader.positions.tolist(), [128, 0, 0, 0, 0, 0, 0, 0])
        prepare.assert_called_once()
        self.assertIs(prepare.call_args.kwargs['short_context'], True)
        self.assertEqual(prepare.call_args.kwargs['batches'], 2)
        reader.validate(170)
        reader.validate(248)
        for start in (127, 249, 256):
            with self.assertRaises(ValueError):
                reader.validate(start)
        with self.assertRaisesRegex(ValueError, 'four-row groups'):
            ReplayAttentionReader(operations, mesh, 8, 256, torch.arange(4).reshape(1, 4),
                lambda value, dtype=None: value, short_context=True, max_group_rows=8)

    def fixture(self, rows=16, *, max_group_rows=4):
        operations = SimpleNamespace(int32='int32', SDPAProgramConfig=Mock())
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        upload = Mock(side_effect=lambda value, dtype=None: value)
        with patch('attention_replay.prepare', return_value='program'):
            reader = ReplayAttentionReader(operations, mesh, rows, 4352,
                torch.arange(68).reshape(1, 68), upload, max_group_rows=max_group_rows)
        return reader, upload

    def test_wide_replay_requires_explicit_compact_scratch_and_preserves_defaults(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'compact native scratch'):
                self.fixture(max_group_rows=8)
        for width in (True, 8.0, 16, 3):
            with self.assertRaisesRegex(ValueError, 'explicitly four or eight'):
                self.fixture(max_group_rows=width)
        with patch.dict('os.environ', {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}):
            wide, _ = self.fixture(rows=32, max_group_rows=8)
            control, _ = self.fixture(rows=32)
        self.assertEqual([len(entry[0]) for entry in wide.metadata], [3, 1])
        self.assertEqual([len(entry[0]) for entry in control.metadata], [3, 3, 2])
        self.assertTrue(all(group['rows'] == 8 for entry in wide.metadata for group in entry[0]))
        self.assertEqual([tuple(entry[2].shape) for entry in wide.metadata], [(3, 1, 96, 4352), (1, 1, 96, 4352)])

    def test_prepared_family_has_zero_masks_and_bounded_parallel_pages(self):
        reader, upload = self.fixture()
        self.assertEqual([len(entry[0]) for entry in reader.metadata], [3, 1])
        self.assertEqual(len(reader.programs), 2)
        self.assertEqual(upload.call_count, 5)
        for bundle, pages, mask, config in reader.metadata:
            self.assertEqual(tuple(pages.shape), (len(bundle), 68))
            self.assertTrue(torch.all(mask == 0))

    def test_out_of_family_stage_rejected_before_any_copy(self):
        reader, _ = self.fixture()
        for start in (4095, 4337, -1):
            with self.assertRaises(ValueError):
                reader.stage(start)
        self.assertEqual(reader.start, 4096)

    def test_close_is_idempotent_and_rejects_replay(self):
        reader, _ = self.fixture()
        with patch('attention_replay.release_owned') as release:
            reader.close()
            reader.close()
            release.assert_called_once()
        self.assertTrue(reader.closed)
        with self.assertRaises(RuntimeError):
            reader.validate(4096)

    def test_partial_staging_failure_poisons_future_replay(self):
        reader, _ = self.fixture()
        reader.operations.ROW_MAJOR_LAYOUT = 'row'
        reader.operations.ReplicateTensorToMesh = Mock()
        reader.operations.from_torch = Mock()
        reader.operations.copy_host_to_device_tensor = Mock(side_effect=RuntimeError('copy failed'))
        with patch('attention_replay.addresses', return_value=(1, 2)):
            with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                reader.stage(4103)
        self.assertTrue(reader.failed)
        with self.assertRaisesRegex(RuntimeError, 'poisoned'):
            reader.validate(4096)

    def test_small_bucket_is_not_silently_reinterpreted(self):
        for rows in (1, 2, 4, True):
            with self.assertRaises(ValueError):
                self.fixture(rows)

    def test_call_refreshes_before_attention_and_preserves_borrowed_buffers(self):
        reader, _ = self.fixture()
        query = SimpleNamespace(shape=(1, 16, 12, 256))
        keys, values, scratch, result = object(), object(), object(), object()
        events = []

        def attention(mesh, operations, query, keys, values, metadata, owned, **kwargs):
            events.append('attention')
            owned.extend((query, keys, values, scratch, result))
            return result

        with patch('attention_replay.addresses', side_effect=lambda operations, value: (id(value), id(value))), patch(
                'attention_replay.refresh_mask', side_effect=lambda *args: events.append('mask')), patch(
                'attention_replay.execute', side_effect=attention), patch('attention_replay.release_owned') as release:
            self.assertIs(reader(query, keys, values, scale=0.0625, memory_config='L1'), result)
            release.assert_called_once_with(reader.operations, [scratch])
        self.assertEqual(events, ['mask', 'mask', 'attention'])
        self.assertEqual((reader.calls, reader.refresh_calls), (1, 2))

    def test_shared_masks_refresh_once_for_each_sixteen_layer_forward(self):
        reader, _ = self.fixture()
        query = SimpleNamespace(shape=(1, 16, 12, 256))
        keys, values, result = object(), object(), object()
        events = []
        with patch('attention_replay.addresses', side_effect=lambda operations, value: (id(value), id(value))), patch(
                'attention_replay.refresh_mask', side_effect=lambda *args: events.append('mask')), patch(
                'attention_replay.execute', side_effect=lambda *args, **kwargs: events.append('attention') or result), patch(
                'attention_replay.release_owned'):
            for forward in range(2):
                with reader.shared_masks(16):
                    for layer in range(16):
                        self.assertIs(reader(query, keys, values, scale=0.0625, memory_config='L1'), result)
        self.assertEqual(events, (['mask'] * 2 + ['attention'] * 16) * 2)
        self.assertEqual((reader.calls, reader.refresh_calls), (32, 4))
        self.assertIsNone(reader.mask_scope)
        self.assertFalse(reader.failed)

    def test_incomplete_or_failed_shared_forward_poisons_reader(self):
        for expected in (1, 16):
            reader, _ = self.fixture()
            with patch('attention_replay.refresh_mask'):
                with self.assertRaisesRegex(AssertionError, 'exact attention call budget'):
                    with reader.shared_masks(expected):
                        pass
            self.assertTrue(reader.failed)
            self.assertIsNone(reader.mask_scope)
        reader, _ = self.fixture()
        with patch('attention_replay.refresh_mask', side_effect=RuntimeError('mask failed')):
            with self.assertRaisesRegex(RuntimeError, 'mask failed'):
                with reader.shared_masks(16):
                    self.fail('Failed mask refresh must not enter the model forward')
        self.assertTrue(reader.failed)
        self.assertIsNone(reader.mask_scope)

    def test_shared_forward_cannot_stage_close_or_nest(self):
        for operation in ('stage', 'close', 'nest'):
            reader, _ = self.fixture()
            with patch('attention_replay.refresh_mask'), self.assertRaises(RuntimeError):
                with reader.shared_masks(16):
                    if operation == 'stage':
                        reader.stage(4103)
                    elif operation == 'close':
                        reader.close()
                    else:
                        with reader.shared_masks(1):
                            self.fail('Nested mask scope must not enter')
            self.assertTrue(reader.failed)
            self.assertFalse(reader.closed)

    def test_shared_forward_rejects_excess_calls_before_attention(self):
        reader, _ = self.fixture()
        query = SimpleNamespace(shape=(1, 16, 12, 256))
        with patch('attention_replay.addresses', return_value=(1, 2)), patch(
                'attention_replay.refresh_mask'), patch('attention_replay.execute') as execute, patch(
                'attention_replay.release_owned'):
            with self.assertRaisesRegex(AssertionError, 'exceeded'):
                with reader.shared_masks(1):
                    reader(query, object(), object(), scale=0.0625, memory_config='L1')
                    reader(query, object(), object(), scale=0.0625, memory_config='L1')
            execute.assert_called_once()
        self.assertTrue(reader.failed)

    def test_shared_forward_requires_explicit_bounded_budget(self):
        reader, _ = self.fixture()
        for expected in (0, 17, True, 1.0, None):
            with self.assertRaises(ValueError):
                with reader.shared_masks(expected):
                    self.fail('Invalid scope must not enter')
        self.assertFalse(reader.failed)
