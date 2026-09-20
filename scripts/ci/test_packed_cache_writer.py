"""The ordered K/V write over the 64-row block: one audited 32-row ordered_cache.update per
tile, each over that tile's own staged positions word and page-table rows, the prepared
K/V sliced on its row axis, every slice freed - and tiles of any other geometry refused
before anything runs."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from packed_cache_writer import TILE_ROWS, SegmentedOrderedCacheWriter, tile, tile_rows, validate_tiles


class FakeOperations:
    int32, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'int32', 'row', 'dram'

    def __init__(self):
        self.events = []

    def slice(self, tensor, start, stop, memory_config):
        self.events.append(('slice', tensor.name, start, stop, memory_config))
        return SimpleNamespace(name='%s[%d:%d]' % (tensor.name, start[1], stop[1]), shape=(1, stop[1] - start[1], 32, 256))

    def to_memory_config(self, tensor, memory_config):
        self.events.append(('interleave', tensor.name, memory_config))
        return SimpleNamespace(name=tensor.name + '.dram', shape=tensor.shape, memory_config=lambda: 'dram')

    def deallocate(self, tensor):
        self.events.append(('free', tensor.name))


def device(shape, dtype='int32', layout='row', name='t'):
    return SimpleNamespace(shape=tuple(shape), dtype=dtype, layout=layout, name=name)


def prepared(rows=64, memory='dram'):
    return SimpleNamespace(shape=(1, rows, 32, 256), name='kv', memory_config=lambda: memory)


def tiles(rows=64, page_width=68):
    return [tile((first, last), device((TILE_ROWS,), name='positions%d' % first),
                 device((TILE_ROWS, page_width), name='pages%d' % first))
            for first, last in tile_rows(rows)]


class TileTests(unittest.TestCase):
    def test_tiles_are_the_thirty_two_row_slices_of_a_block_beyond_one_tile(self):
        self.assertEqual(tile_rows(64), ((0, 32), (32, 64)))
        self.assertEqual(tile_rows(96), ((0, 32), (32, 64), (64, 96)))
        for rows in (32, 16, 48, 0, 64.0, True, -64):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                tile_rows(rows)

    def test_tiles_are_checked_for_order_geometry_and_one_page_width(self):
        operations = FakeOperations()
        self.assertEqual(validate_tiles(operations, tiles())[1:], (64, 68))
        self.assertEqual(validate_tiles(operations, tiles(96, 1024))[1:], (96, 1024))
        cases = {'one tile only': tiles()[:1], 'no tiles': [], 'tiles out of order': tiles()[::-1],
                 'a gap': [tiles()[0], tile((48, 80), device((32,)), device((32, 68)))],
                 'a narrow positions word': [tiles()[0], tile((32, 64), device((16,)), device((32, 68)))],
                 'a tiled positions word': [tiles()[0], tile((32, 64), device((32,), layout='tile'), device((32, 68)))],
                 'bf16 pages': [tiles()[0], tile((32, 64), device((32,)), device((32, 68), dtype='bf16'))],
                 'pages of another width': [tiles()[0], tile((32, 64), device((32,)), device((32, 67)))],
                 'pages of sixteen rows': [tiles()[0], tile((32, 64), device((32,)), device((16, 68)))],
                 'a tile without pages': [tiles()[0], SimpleNamespace(rows=(32, 64), positions=device((32,)))]}
        for name, broken in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_tiles(operations, broken)


class WriterTests(unittest.TestCase):
    def writer(self, rows=64):
        operations = FakeOperations()
        return SegmentedOrderedCacheWriter('mesh', operations, 'kernels', tiles(rows)), operations

    def call(self, writer, packed, rows=64, page_width=68, updates=None):
        with patch('ordered_cache.update', **(dict(side_effect=updates) if updates is not None else {})) as update:
            writer('cache', packed, update_idxs_tensor=device((rows,), name='positions'),
                   page_table=device((rows, page_width), name='pages'))
        return update

    def test_each_tile_runs_the_audited_update_over_its_own_slice_and_metadata(self):
        writer, operations = self.writer()
        self.assertEqual((writer.rows, writer.page_width, writer.calls, len(writer.tiles)), (64, 68, 0, 2))
        update = self.call(writer, prepared())
        self.assertEqual([(call.args[0], call.args[1], call.args[2].name, call.args[3].name, call.args[4].name, call.args[5])
                          for call in update.call_args_list],
                         [('mesh', 'cache', 'kv[0:32]', 'positions0', 'pages0', 'kernels'),
                          ('mesh', 'cache', 'kv[32:64]', 'positions32', 'pages32', 'kernels')])
        self.assertIs(update.call_args_list[0].args[3], writer.tiles[0].positions)
        self.assertIs(update.call_args_list[1].args[4], writer.tiles[1].pages)
        # already interleaved: whole tiles sliced on the row axis, each freed after its update
        self.assertEqual(operations.events, [
            ('slice', 'kv', (0, 0, 0, 0), (1, 32, 32, 256), 'dram'), ('free', 'kv[0:32]'),
            ('slice', 'kv', (0, 32, 0, 0), (1, 64, 32, 256), 'dram'), ('free', 'kv[32:64]')])
        self.assertEqual(writer.calls, 1)

    def test_a_sharded_input_is_interleaved_first_and_freed_last(self):
        writer, operations = self.writer()
        update = self.call(writer, prepared(memory='sharded'))
        self.assertEqual([call.args[2].name for call in update.call_args_list], ['kv.dram[0:32]', 'kv.dram[32:64]'])
        self.assertEqual(operations.events[0], ('interleave', 'kv', 'dram'))
        self.assertEqual(operations.events[-1], ('free', 'kv.dram'))
        self.assertEqual(writer.calls, 1)

    def test_a_failing_update_frees_its_slice_and_counts_no_call(self):
        writer, operations = self.writer()
        with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
            self.call(writer, prepared(memory='sharded'), updates=[None, RuntimeError('kernel failed')])
        self.assertEqual([event for event in operations.events if event[0] == 'free'],
                         [('free', 'kv.dram[0:32]'), ('free', 'kv.dram[32:64]'), ('free', 'kv.dram')])
        self.assertEqual(writer.calls, 0)

    def test_geometry_that_does_not_match_the_tiles_is_refused_before_any_slice(self):
        writer, operations = self.writer()
        cases = {'a 32-row input': dict(packed=prepared(32)), 'a 96-row input': dict(packed=prepared(96)),
                 'a 32-row positions word': dict(rows=32), 'a wider page table': dict(page_width=1024)}
        for name, overrides in cases.items():
            options = dict(packed=prepared(), rows=64, page_width=68)
            options.update(overrides)
            with self.subTest(name=name), patch('ordered_cache.update') as update, self.assertRaises(ValueError):
                writer('cache', options['packed'], update_idxs_tensor=device((options['rows'],)),
                       page_table=device((options['rows'], options['page_width'])))
            update.assert_not_called()
        self.assertEqual((operations.events, writer.calls), ([], 0))

    def test_the_writer_keeps_the_ordered_writers_keyword_contract(self):
        """attention_batch.serial_tail binds the writer as paged_update_cache: keywords only."""
        writer, operations = self.writer()
        with self.assertRaises(TypeError):
            writer('cache', prepared(), 'positions', 'pages')
        self.assertEqual(operations.events, [])


if __name__ == '__main__':
    unittest.main()
