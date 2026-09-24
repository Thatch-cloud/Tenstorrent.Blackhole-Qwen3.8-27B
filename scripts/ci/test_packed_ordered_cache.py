"""Verify-trace T2 cut #2: the packed block's K/V write as one chained launch per cache with one
semaphore chain per user (packed_ordered_cache.py).

What the CPU can hold it to: the chains start and end exactly at the segment boundaries; at one
32-row chain the launch IS ordered_cache.update's (same kernels, compile args, CBs, semaphore and
runtime args, core by core), so the only differences at 64 rows are reader arg 7, the chain args
and the core count - CB16 stays at the served 256 pages and the per-core CB footprint at the
audited 390,160 bytes (width 2052); the kernels are load_kernels' own output; the writer keeps
the ordered writer's call contract, counts its calls, notes each one, and in 32-row mode runs the
served slices and tile metadata. Whether the chains leave the served bytes on silicon is card M's
(optimisation/ttnn-op/verify_t2); the guard that makes it exact is test_verify_trace_t2's.
"""

import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ordered_cache
import packed_ordered_cache as poc
from packed_cache_writer import tile as cache_tile, tile_rows
import verify_trace_t2 as t2
from test_gdn_conv_windows_packed import DescriptorTTNN, mesh

SEGMENTS = ((0, 16), (16, 32), (32, 48), (48, 64))
WIDTH = 2052
PAGES = 2064


def fake_native_root(directory):
    """A native tree whose three kernels carry the anchors load_kernels patches; ordered_cache.HASHES
    is patched to their digests (the real sources are hash-pinned in the image)."""
    sources = dict(
        reader='// reader\nconstexpr auto page_table_args = TensorAccessorArgs<index_tensor_args.next_compile_time_args_offset()>();\n'
               'cb_input.reserve_back(Wt);\n    cb_input.push_back(Wt);\n',
        writer='// writer\n', compute='// compute\n')
    hashes = {}
    for role, relative in ordered_cache.FILES.items():
        path = Path(directory) / ordered_cache.ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sources[role].encode())
        hashes[role] = hashlib.sha256(sources[role].encode()).hexdigest()
    return hashes


def block(ttnn, rows=64, width=WIDTH, pages_total=PAGES, packed_memory='dram'):
    cache = ttnn.tensor('cache', (pages_total, 2, 64, 256), 'dram', 'bf8', 'tile')
    packed = ttnn.tensor('kv', (1, rows, 32, 256), packed_memory, 'bf16', 'tile')
    positions = ttnn.tensor('positions', (rows,), 'dram', 'int32', 'row_major')
    pages = ttnn.tensor('pages', (rows, width), 'dram', 'int32', 'row_major')
    return cache, packed, positions, pages


def tiles(ttnn, width=WIDTH):
    return [cache_tile((first, last), ttnn.tensor('positions%d' % first, (32,), 'dram', 'int32', 'row_major'),
                       ttnn.tensor('pages%d' % first, (32, width), 'dram', 'int32', 'row_major'))
            for first, last in tile_rows(64)]


KERNELS = dict(reader='READER', writer='WRITER', compute='COMPUTE')


class ChainTests(unittest.TestCase):
    def test_one_chain_per_segment(self):
        args = poc.chain_args(64, SEGMENTS)
        self.assertEqual([row for row, (wait, signal, successor) in enumerate(args) if not wait], [0, 16, 32, 48])
        self.assertEqual([row for row, (wait, signal, successor) in enumerate(args) if not signal], [15, 31, 47, 63])
        self.assertEqual([successor for wait, signal, successor in args],
                         [row if row in (15, 31, 47, 63) else row + 1 for row in range(64)])

    def test_one_span_is_the_served_chain(self):
        served = [(int(index > 0), int(index < 31), min(index + 1, 31)) for index in range(32)]
        self.assertEqual(poc.chain_args(32, ((0, 32),)), served)

    def test_the_32_row_mode_chains_each_tiles_users(self):
        self.assertEqual(poc.tile_spans(SEGMENTS, 0, 32), ((0, 16), (16, 32)))
        self.assertEqual(poc.tile_spans(SEGMENTS, 32, 64), ((0, 16), (16, 32)))
        # a span across the tile boundary is split; command-queue order keeps its rows in order
        self.assertEqual(poc.tile_spans(((0, 24), (24, 64)), 0, 32), ((0, 24), (24, 32)))
        self.assertEqual(poc.tile_spans(((0, 24), (24, 64)), 32, 64), ((0, 32),))
        args = poc.chain_args(32, poc.tile_spans(SEGMENTS, 32, 64))
        self.assertEqual([row for row, (wait, signal, successor) in enumerate(args) if not wait], [0, 16])

    def test_spans_that_do_not_tile_the_block_are_refused(self):
        for spans in ((), ((0, 16), (16, 32)), ((0, 16), (17, 64)), ((0, 16), (16, 16), (16, 64)), ((16, 64),)):
            with self.subTest(spans=spans), self.assertRaises(poc.Unsupported):
                poc.chain_args(64, spans)

    def test_the_negative_controls(self):
        args = poc.chain_args(64, SEGMENTS)
        self.assertEqual(poc.apply_negative(args, SEGMENTS, None), (args, list(range(64))))
        self.assertEqual(poc.apply_negative(args, SEGMENTS, 'conflict'), (args, list(range(64))))
        nochain, index = poc.apply_negative(args, SEGMENTS, 'nochain')
        self.assertTrue(all(wait == 0 and signal == 0 for wait, signal, successor in nochain))
        swapped, index = poc.apply_negative(args, SEGMENTS, 'index')
        self.assertEqual(swapped, args, 'the chain is untouched')
        self.assertEqual(index[:3], [1, 0, 2])
        with self.assertRaises(ValueError):
            poc.apply_negative(args, SEGMENTS, 'bogus')
        with self.assertRaises(ValueError):
            poc.apply_negative(poc.chain_args(2, ((0, 1), (1, 2))), ((0, 1), (1, 2)), 'index')

    def test_the_cb_footprint_is_the_audited_launchs(self):
        self.assertEqual(poc.cb_bytes(WIDTH), 390160)
        self.assertEqual(poc.cb_bytes(WIDTH, 512) - poc.cb_bytes(WIDTH), 256 * 1088)
        self.assertEqual(poc.reader_compile_args(64, WIDTH)[7], 256)
        self.assertEqual(poc.reader_compile_args(32, WIDTH)[7], 128)

    def test_shapes(self):
        self.assertEqual(poc.validate_chained((PAGES, 2, 64, 256), (1, 64, 32, 256), (64,), (64, WIDTH), 64), 64)
        cases = [((PAGES, 2, 64, 256), (1, 16, 32, 256), (16,), (16, WIDTH), 16),
                 ((PAGES, 2, 64, 256), (1, 64, 32, 256), (64,), (64, 2051), 64),
                 ((PAGES, 2, 64, 256), (1, 64, 32, 256), (32,), (64, WIDTH), 64),
                 ((1000, 2, 64, 256), (1, 64, 32, 256), (64,), (64, 1024), 64),
                 ((PAGES, 4, 64, 256), (1, 64, 32, 256), (64,), (64, WIDTH), 64)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                poc.validate_chained(*case)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.ttnn = DescriptorTTNN()
        self.mesh = mesh()

    def launch(self, rows=64, spans=SEGMENTS, **options):
        cache, packed, positions, pages = block(self.ttnn, rows)
        poc.update_chained(self.mesh, cache, packed, positions, pages, KERNELS, spans, operations=self.ttnn, **options)
        return (cache, packed, positions, pages), self.ttnn.programs()[-1]

    def test_one_launch_on_an_8x8_rectangle_with_the_served_cbs(self):
        tensors, (name, listed, program) = self.launch()
        self.assertEqual(len(self.ttnn.programs()), 1)
        self.assertEqual(listed, list(tensors))
        chip = program[((0, 0), (0, 0))]
        self.assertEqual([kernel['cores'] for kernel in chip['kernels']], [('set', (('range', (0, 0), (7, 7)),))] * 3)
        self.assertEqual([kernel['source'] for kernel in chip['kernels']], ['READER', 'WRITER', 'COMPUTE'])
        self.assertEqual({kernel['source_type'] for kernel in chip['kernels']}, {'source-code'})
        self.assertEqual([(tuple(f.buffer_index for f in cb.format_descriptors), cb.total_size) for cb in chip['cbs']],
                         [((0,), 16 * 1088), ((1,), 8 * 2048), ((24, 25), 16 * 2048), ((26,), 16 * 2048),
                          ((16,), 256 * 1088), ((2,), 4096), ((3,), WIDTH * 4)])
        self.assertEqual(sum(cb.total_size for cb in chip['cbs']), 390160)
        self.assertEqual(chip['semaphores'][0].id, 0)
        self.assertEqual(chip['semaphores'][0].initial_value, 0)

    def test_every_core_gets_its_row_and_its_chain(self):
        (cache, packed, positions, pages), (name, listed, program) = self.launch()
        for chip in range(2):
            reader, writer, compute = program[((0, chip), (0, chip))]['kernels']
            address = [value.shards[chip].address for value in (cache, packed, positions, pages)]
            self.assertEqual(reader['compile'][:19], poc.reader_compile_args(64, WIDTH))
            self.assertEqual(reader['compile'][19:], [7, 1] * 4)
            self.assertEqual(writer['compile'], poc.writer_compile_args(WIDTH) + [7, 1])
            self.assertEqual(compute['compile'], list(poc.COMPUTE_ARGS))
            chains = poc.chain_args(64, SEGMENTS)
            by_core = {(x, y): args for x, y, args in reader['runtime']}
            writes = {(x, y): args for x, y, args in writer['runtime']}
            self.assertEqual(len(by_core), 64)
            for row in range(64):
                wait, signal, successor = chains[row]
                core = (row % 8, row // 8)
                self.assertEqual(by_core[core], (address[0], 0, address[2], row, address[3], wait, address[1]))
                self.assertEqual(writes[core], (address[0], 0, 0, row, signal, successor % 8 + 1, successor // 8 + 2))
            self.assertTrue(all(args == () for x, y, args in compute['runtime']))

    def test_at_one_32_row_chain_it_is_ordered_cache_update_exactly(self):
        cache, packed, positions, pages = block(self.ttnn, 32)
        with patch.dict(sys.modules, {'ttnn': self.ttnn}):
            ordered_cache.update(self.mesh, cache, packed, positions, pages, KERNELS)
        poc.update_chained(self.mesh, cache, packed, positions, pages, KERNELS, ((0, 32),), operations=self.ttnn)
        served, chained = [program for name, listed, program in self.ttnn.programs()]
        self.assertEqual(sorted(served), sorted(chained))
        for coordinate in served:
            for a, b in zip(served[coordinate]['kernels'], chained[coordinate]['kernels']):
                self.assertEqual({k: v for k, v in a.items() if k != 'cores'}, {k: v for k, v in b.items() if k != 'cores'})
            self.assertEqual([(cb.total_size, [f.buffer_index for f in cb.format_descriptors]) for cb in served[coordinate]['cbs']],
                             [(cb.total_size, [f.buffer_index for f in cb.format_descriptors]) for cb in chained[coordinate]['cbs']])
            cores = lambda program: sorted((x, y) for x, y, args in program[coordinate]['kernels'][0]['runtime'])
            self.assertEqual(cores(served), cores(chained))

    def test_cb16_can_be_sized_for_the_card_m_check(self):
        tensors, (name, listed, program) = self.launch(cb16_pages=512)
        cb16 = [cb for cb in program[((0, 0), (0, 0))]['cbs'] if cb.format_descriptors[0].buffer_index == 16][0]
        self.assertEqual(cb16.total_size, 512 * 1088)
        with self.assertRaises(ValueError):
            self.launch(cb16_pages=8)

    def test_the_writer_index_control_swaps_only_the_writers_rows(self):
        tensors, (name, listed, program) = self.launch(negative='index')
        reader, writer, compute = program[((0, 0), (0, 0))]['kernels']
        writes = {(x, y): args for x, y, args in writer['runtime']}
        reads = {(x, y): args for x, y, args in reader['runtime']}
        self.assertEqual((writes[(0, 0)][3], writes[(1, 0)][3]), (1, 0))
        self.assertEqual((reads[(0, 0)][3], reads[(1, 0)][3]), (0, 1))

    def test_a_grid_under_8x8_is_unsupported_and_nothing_launches(self):
        self.mesh = mesh(grid=(8, 7))
        with self.assertRaises(poc.Unsupported):
            self.launch()
        self.assertEqual(self.ttnn.programs(), [])
        self.launch(32, ((0, 16), (16, 32)))

    def test_the_served_input_checks_hold(self):
        cache, packed, positions, pages = block(self.ttnn)
        for broken in (dict(packed=self.ttnn.tensor('kv', (1, 64, 32, 256), 'l1', 'bf16', 'tile')),
                       dict(cache=self.ttnn.tensor('cache', (PAGES, 2, 64, 256), 'dram', 'bf16', 'tile')),
                       dict(positions=self.ttnn.tensor('positions', (64,), 'dram', 'uint32', 'row_major'))):
            arguments = dict(cache=cache, packed=packed, positions=positions, pages=pages)
            arguments.update(broken)
            with self.subTest(broken=sorted(broken)), self.assertRaises(ValueError):
                poc.update_chained(self.mesh, arguments['cache'], arguments['packed'], arguments['positions'],
                                   arguments['pages'], KERNELS, SEGMENTS, operations=self.ttnn)
        aliased = block(self.ttnn)
        aliased[2].shards = aliased[3].shards
        with self.assertRaisesRegex(ValueError, 'must not alias'):
            poc.update_chained(self.mesh, *aliased, KERNELS, SEGMENTS, operations=self.ttnn)
        self.assertEqual(self.ttnn.programs(), [])

    def test_the_kernels_are_load_kernels_own_output(self):
        with tempfile.TemporaryDirectory() as directory:
            hashes = fake_native_root(directory)
            with patch.dict(ordered_cache.HASHES, hashes):
                kernels = ordered_cache.load_kernels(directory)
        self.assertIn('input_args', kernels['reader'])
        cache, packed, positions, pages = block(self.ttnn)
        poc.update_chained(self.mesh, cache, packed, positions, pages, kernels, SEGMENTS, operations=self.ttnn)
        chip = self.ttnn.programs()[0][2][((0, 0), (0, 0))]
        self.assertEqual([kernel['source'] for kernel in chip['kernels']],
                         [kernels['reader'], kernels['writer'], kernels['compute']])
        self.assertNotIn('packed_ordered_cache.cpp', ' '.join(p.name for p in Path(poc.__file__).parent.glob('packed_ordered_cache*')))


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.ttnn = DescriptorTTNN()
        self.mesh = mesh()
        t2.take()
        self.addCleanup(t2.take)

    def writer(self, launch_rows=64, **overrides):
        cache, packed, positions, pages = block(self.ttnn)
        options = dict(positions=positions, pages=pages, spans=SEGMENTS, tiles=tiles(self.ttnn), launch_rows=launch_rows)
        options.update(overrides)
        return poc.ChainedOrderedCacheWriter(self.mesh, self.ttnn, KERNELS, **options), cache, packed

    def call(self, writer, cache, packed, rows=64, width=WIDTH):
        writer(cache, packed, update_idxs_tensor=SimpleNamespace(shape=(rows,)),
               page_table=SimpleNamespace(shape=(rows, width)))

    def test_one_chained_launch_per_call_counted_and_noted(self):
        writer, cache, packed = self.writer()
        self.call(writer, cache, packed)
        self.call(writer, cache, packed)
        self.assertEqual(writer.calls, 2)
        self.assertEqual(t2.take(), {'kv_chains': 2})
        launches = self.ttnn.programs()
        self.assertEqual(len(launches), 2)
        self.assertEqual(launches[0][1], [cache, packed, writer.positions, writer.pages])
        self.assertEqual([entry for entry in self.ttnn.calls if entry[0] == 'slice'], [])

    def test_the_32_row_mode_runs_the_served_slices_over_the_tile_metadata(self):
        writer, cache, packed = self.writer(launch_rows=32)
        self.call(writer, cache, packed)
        self.assertEqual([entry[2:] for entry in self.ttnn.calls if entry[0] == 'slice'],
                         [((0, 0, 0, 0), (1, 32, 32, 256)), ((0, 32, 0, 0), (1, 64, 32, 256))])
        launches = self.ttnn.programs()
        self.assertEqual([listed[2].name for name, listed, program in launches], ['positions0', 'positions32'])
        reader = launches[0][2][((0, 0), (0, 0))]['kernels'][0]
        self.assertEqual(reader['compile'][7], 128, 'the served 32-row reader binary')
        waits = sorted((args[3], args[5]) for x, y, args in reader['runtime'])
        self.assertEqual([row for row, wait in waits if not wait], [0, 16])
        self.assertEqual({piece.name for piece in self.ttnn.deallocated}, {'kv[0:32]', 'kv[32:64]'})
        self.assertEqual(writer.calls, 1)

    def test_a_sharded_input_is_interleaved_first_and_freed_after(self):
        writer, cache, packed = self.writer()
        packed = self.ttnn.tensor('kv', (1, 64, 32, 256), 'sharded', 'bf16', 'tile')
        self.call(writer, cache, packed)
        self.assertEqual(self.ttnn.calls[0], ('interleave', 'kv'))
        self.assertEqual([value.name for value in self.ttnn.deallocated], ['kv.dram'])

    def test_what_it_cannot_serve_is_unsupported_at_construction(self):
        cases = dict(grid=dict(), width=dict(pages=self.ttnn.tensor('pages', (64, 1500), 'dram', 'int32', 'row_major')),
                     spans=dict(spans=((0, 32), (33, 64))), rows=dict(launch_rows=16),
                     tiles=dict(tiles=tiles(self.ttnn, 1024)),
                     positions=dict(positions=self.ttnn.tensor('positions', (32,), 'dram', 'int32', 'row_major')))
        for name, overrides in cases.items():
            with self.subTest(case=name), self.assertRaises(poc.Unsupported):
                if name == 'grid':
                    self.mesh = mesh(grid=(8, 4))
                self.writer(**overrides)
            self.mesh = mesh()
        self.assertEqual(self.ttnn.programs(), [])

    def test_call_geometry_is_refused_before_any_launch(self):
        writer, cache, packed = self.writer()
        for rows, width, shape in ((32, WIDTH, (1, 64, 32, 256)), (64, 1024, (1, 64, 32, 256)), (64, WIDTH, (1, 32, 32, 256))):
            with self.subTest(rows=rows, width=width, shape=shape), self.assertRaises(ValueError):
                self.call(writer, cache, self.ttnn.tensor('kv', shape, 'dram', 'bf16', 'tile'), rows, width)
        self.assertEqual((self.ttnn.programs(), writer.calls, t2.take()), ([], 0, {}))

    def test_the_keyword_contract_of_the_ordered_writers(self):
        writer, cache, packed = self.writer()
        with self.assertRaises(TypeError):
            writer(cache, packed, 'positions', 'pages')


if __name__ == '__main__':
    unittest.main()
