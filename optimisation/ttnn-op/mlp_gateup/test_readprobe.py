"""CPU tests of the WP4 read-bandwidth probe (readprobe.py, readprobe_reader.cpp): its worker plans at the real shapes, the layout fact it rests on (modelled), the kernel's index
arithmetic transliterated over a model of an interleaved bank memory, and the program descriptors it builds.

Run: python -B -m unittest discover -s optimisation/ttnn-op/mlp_gateup -p 'test_*.py'
"""

from collections import Counter
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
CI = HERE.parent.parent.parent / 'scripts' / 'ci'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CI))
import readprobe as probe  # noqa: E402
import test_tp4_mlp_fused as fused_tests  # noqa: E402

BANKS = 8
SHAPES = (('gate', 160, 136, 'bfp4'), ('down', 136, 160, 'bfp8'))


class PlanTests(unittest.TestCase):
    def test_every_page_is_read_exactly_once_in_every_mode_at_both_real_shapes(self):
        for name, rows, columns, dtype in SHAPES:
            for mode, points in (('bank', probe.BANK_POINTS), ('stock', probe.STOCK_POINTS)):
                for run, chunk in points:
                    with self.subTest(shape=name, mode=mode, run=run, chunk=chunk):
                        plan = probe.worker_plan(mode, run, chunk, rows, columns, BANKS, (13, 10))
                        pages = probe.covered_pages(plan, rows, columns, BANKS)
                        self.assertEqual(sorted(pages), list(range(rows * columns)))
                        self.assertEqual(max(Counter(pages).values()), 1)

    def test_the_worker_counts(self):
        counts = dict(((mode, run), len(probe.worker_plan(mode, run, chunk, 160, 136, BANKS, (13, 10)))) for mode, run, chunk in
                      [('bank', 2, 1), ('bank', 3, 3), ('bank', 4, 4), ('bank', 6, 6), ('stock', 2, 1), ('stock', 3, 1), ('stock', 4, 1)])
        self.assertEqual(counts, {('bank', 2): 72, ('bank', 3): 48, ('bank', 4): 40, ('bank', 6): 24, ('stock', 2): 68, ('stock', 3): 46, ('stock', 4): 34})

    def test_a_bank_worker_owns_the_columns_of_one_bank(self):
        for worker in probe.worker_plan('bank', 3, 3, 160, 136, BANKS, (13, 10)):
            banks = {(column + BANKS * m) % BANKS for column, tiles in worker['requests'] for m in range(tiles)}
            self.assertEqual(len(banks), 1)

    def test_a_stock_worker_owns_neighbouring_columns_one_tile_per_request(self):
        plan = probe.worker_plan('stock', 3, 1, 160, 136, BANKS, (13, 10))
        self.assertEqual(plan[0]['requests'], [(0, 1), (1, 1), (2, 1)])
        self.assertEqual(plan[-1]['requests'], [(135, 1)])
        self.assertTrue(all(tiles == 1 for worker in plan for unused, tiles in worker['requests']))

    def test_the_request_sizes(self):
        plan = probe.worker_plan('bank', 6, 3, 160, 136, BANKS, (13, 10))
        largest, mean, per_row = probe.request_bytes(plan, 576)
        self.assertEqual(largest, 3 * 576)
        self.assertGreater(per_row, 0)
        stock = probe.worker_plan('stock', 3, 1, 160, 136, BANKS, (13, 10))
        self.assertEqual(probe.request_bytes(stock, 576)[:2], (576, 576.0))
        self.assertEqual(probe.request_bytes(stock, 576)[2], 136, 'one request per tile of a row')
        self.assertLess(probe.request_bytes(plan, 576)[2], 136)

    def test_cores_fill_a_rectangle_row_major_and_must_fit_the_device(self):
        plan = probe.worker_plan('bank', 2, 2, 160, 136, BANKS, (13, 10))
        self.assertEqual(plan[0]['core'], (0, 0))
        self.assertEqual(plan[13]['core'], (0, 1))
        self.assertEqual(max(worker['core'][1] for worker in plan), 5)
        with self.assertRaisesRegex(ValueError, 'need'):
            probe.worker_plan('bank', 2, 2, 160, 136, BANKS, (13, 4))
        with self.assertRaisesRegex(ValueError, 'width'):
            probe.worker_plan('bank', 2, 2, 160, 136, BANKS, (13, 10), width=14)

    def test_refusals(self):
        for arguments in (('bank', 3, 4), ('bank', 0, 0), ('bank', 9, 1), ('stock', 3, 3), ('diagonal', 3, 1)):
            with self.assertRaises(ValueError):
                probe.worker_plan(arguments[0], arguments[1], arguments[2], 160, 136, BANKS, (13, 10))
        with self.assertRaisesRegex(ValueError, 'whole blocks'):
            probe.worker_plan('bank', 3, 3, 150, 136, BANKS, (13, 10))

    def test_a_column_count_that_is_not_a_bank_multiple_is_refused_by_the_launch_not_the_plan(self):
        with self.assertRaisesRegex(ValueError, 'multiple of the 8 banks'):
            probe.ReadProbe(None, None, None, None, 'bfp4', 160, 137, BANKS, 'bank', 3, 3, (13, 10))


# ---------------------------------------------------------------------------------------------------------------------------
# The kernel's index arithmetic over a model of an interleaved DRAM tensor.
# ---------------------------------------------------------------------------------------------------------------------------

class BankMemory(object):
    """Interleaved pages: page p lives in bank p % banks at bank offset (p // banks) * page_bytes. Each page's bytes are its id repeated (so a read returns identifiable pages)."""

    def __init__(self, pages, page_bytes, banks=BANKS):
        self.page_bytes, self.banks = page_bytes, banks
        self.memory = [[] for unused in range(banks)]
        for page in range(pages):
            self.memory[page % banks].append(page)

    def noc_addr(self, page):
        return (page % self.banks, (page // self.banks) * self.page_bytes)

    def read(self, address, size):
        bank, offset = address
        first = offset // self.page_bytes
        count = size // self.page_bytes
        return self.memory[bank][first:first + count]


def run_kernel(plan, rows, columns, page_bytes, block_rows, run, banks, write_back):
    """readprobe_reader.cpp's loops for every worker, over BankMemory: returns the pages each worker's landing zone holds at the end of each block, and the destination writes."""
    memory = BankMemory(rows * columns, page_bytes, banks)
    landings, writes = [], []
    for worker in plan:
        requests = worker['requests']
        for block in range(rows // block_rows):
            landing = {}
            for row in range(block_rows):
                k = block * block_rows + row
                offset_tiles = 0
                for column, tiles in requests:
                    got = memory.read(memory.noc_addr(k * columns + column), tiles * page_bytes)
                    for index, page in enumerate(got):
                        landing[row * run + offset_tiles + index] = page
                    offset_tiles += tiles
            landings.append(landing)
            if write_back:
                for row in range(block_rows):
                    k = block * block_rows + row
                    offset_tiles = 0
                    for column, tiles in requests:
                        for tile in range(tiles):
                            writes.append((k * columns + column + banks * tile, landing[row * run + offset_tiles + tile]))
                        offset_tiles += tiles
    return landings, writes


class KernelModelTests(unittest.TestCase):
    def test_a_contiguous_read_returns_the_pages_the_plan_names(self):
        for name, rows, columns, dtype in SHAPES:
            page_bytes = probe.PAGE_BYTES[dtype]
            for mode, points in (('bank', probe.BANK_POINTS), ('stock', probe.STOCK_POINTS)):
                for run, chunk in points:
                    with self.subTest(shape=name, mode=mode, run=run, chunk=chunk):
                        plan = probe.worker_plan(mode, run, chunk, rows, columns, BANKS, (13, 10))
                        for worker in plan[:3] + plan[-2:]:
                            landings, writes = run_kernel([worker], rows, columns, page_bytes, probe.BLOCK_ROWS, run, BANKS, True)
                            self.assertEqual(len(landings), rows // probe.BLOCK_ROWS)
                            for block, landing in enumerate(landings):
                                for row in range(probe.BLOCK_ROWS):
                                    k = block * probe.BLOCK_ROWS + row
                                    wanted = [k * columns + column + BANKS * m for column, tiles in worker['requests'] for m in range(tiles)]
                                    got = [landing[row * run + index] for index in range(worker['tiles'])]
                                    self.assertEqual(got, wanted)
                            self.assertTrue(all(page == value for page, value in writes), 'the write-back puts every tile on its own page')

    def test_the_full_write_back_copies_the_whole_tensor(self):
        rows, columns = 16, 136
        for mode, run, chunk in (('bank', 3, 3), ('bank', 4, 2), ('stock', 3, 1)):
            plan = probe.worker_plan(mode, run, chunk, rows, columns, BANKS, (13, 10))
            landings, writes = run_kernel(plan, rows, columns, 576, probe.BLOCK_ROWS, run, BANKS, True)
            self.assertEqual(sorted(page for page, value in writes), list(range(rows * columns)))
            self.assertTrue(all(page == value for page, value in writes))

    def test_a_chunk_across_a_bank_boundary_would_be_caught(self):
        """The negative control: a request that strode by one page instead of by the bank count reads other banks' pages, and the model sees it."""
        memory = BankMemory(160 * 136, 576)
        got = memory.read(memory.noc_addr(5 * 136 + 8), 3 * 576)
        self.assertEqual(got, [5 * 136 + 8, 5 * 136 + 16, 5 * 136 + 24])
        self.assertNotEqual(got, [5 * 136 + 8, 5 * 136 + 9, 5 * 136 + 10])

    def test_the_kernel_text_uses_the_arithmetic_the_model_does(self):
        source = (HERE / probe.KERNEL).read_text(encoding='utf-8')
        self.assertIn('source.get_noc_addr(k * row_pages + column)', source)
        self.assertIn('landing + (row * run + offset_tiles) * page_bytes', source)
        self.assertIn('tiles * page_bytes', source)
        self.assertIn('k * row_pages + column + banks * tile', source)
        self.assertIn('noc_async_read_barrier();', source)
        self.assertIn('if (write_back)', source)
        names = set(re.findall(r'get_named_compile_time_arg_val\("(\w+)"\)', source))
        self.assertEqual(names, {'rows', 'row_pages', 'page_bytes', 'block_rows', 'run', 'banks', 'write_back'})


# ---------------------------------------------------------------------------------------------------------------------------
# The program descriptors.
# ---------------------------------------------------------------------------------------------------------------------------

def make_probe(mode='bank', run=3, chunk=3, write_back=False, dtype='bfp4', rows=160, columns=136, chips=1):
    ops = fused_tests.FakeOps(chips)
    source = fused_tests.Tensor((1, 1, rows * 32, columns * 32), dtype, 500, chips)
    destination = fused_tests.Tensor((1, 1, rows * 32, columns * 32), dtype, 700, chips)
    return ops, probe.ReadProbe(ops, 'mesh', source, destination, dtype, rows, columns, BANKS, mode, run, chunk, (13, 10), write_back=write_back), source, destination


class DescriptorTests(unittest.TestCase):
    def test_one_kernel_on_the_workers_with_the_names_the_kernel_reads_and_constant_argument_lengths(self):
        ops, op, source, destination = make_probe()
        op()
        tensors, mesh_program = ops.launched[0]
        self.assertEqual(tensors, [source, destination])
        program = next(iter(mesh_program.values()))
        self.assertEqual(len(program.kernels), 1)
        kernel = program.kernels[0]
        self.assertTrue(kernel.kernel_source.endswith(probe.KERNEL))
        values = dict(kernel.named_compile_time_args)
        self.assertEqual((values['rows'], values['row_pages'], values['page_bytes'], values['block_rows'], values['run'], values['banks'], values['write_back']),
                         (160, 136, 576, 8, 3, 8, 0))
        lengths = {len(kernel.runtime_args[worker['core'][0]][worker['core'][1]]) for worker in op.plan}
        self.assertEqual(len(lengths), 1, 'every core has the same argument length')
        first = op.plan[0]
        arguments = kernel.runtime_args[first['core'][0]][first['core'][1]]
        self.assertEqual(arguments[:3], [500, 700, len(first['requests'])])
        self.assertEqual(arguments[3:3 + 2 * len(first['requests'])], [value for pair in first['requests'] for value in pair])

    def test_the_landing_buffer_holds_a_block_and_the_write_back_flag_reaches_the_kernel(self):
        ops, op, source, destination = make_probe(run=4, chunk=2, write_back=True, dtype='bfp8')
        op()
        program = next(iter(ops.launched[0][1].values()))
        cb = program.cbs[0]
        self.assertGreaterEqual(cb.total_size, probe.BLOCK_ROWS * 4 * 1088)
        self.assertEqual(cb.total_size % 2048, 0)
        self.assertEqual(dict(program.kernels[0].named_compile_time_args)['write_back'], 1)
        self.assertEqual(dict(program.kernels[0].named_compile_time_args)['page_bytes'], 1088)

    def test_the_tensor_accessors_come_source_then_destination(self):
        ops, op, source, destination = make_probe()
        op()
        kernel = next(iter(ops.launched[0][1].values())).kernels[0]
        self.assertEqual(kernel.compile_time_args, [1000 + 500, 1000 + 700])
        text = (HERE / probe.KERNEL).read_text(encoding='utf-8')
        self.assertLess(text.index('source_args = TensorAccessorArgs<0>()'), text.index('destination_args'))

    def test_counts_and_request_sizes_are_exposed(self):
        ops, op, source, destination = make_probe(run=3, chunk=3)
        self.assertEqual((len(op.plan), op.largest), (48, 3 * 576))
        ops, op, source, destination = make_probe(mode='stock', run=3, chunk=1)
        self.assertEqual((len(op.plan), op.largest, op.requests_per_row), (46, 576, 136))

    def test_a_dtype_it_does_not_know_is_refused(self):
        with self.assertRaises(ValueError):
            make_probe(dtype='bf16')


if __name__ == '__main__':
    unittest.main()
