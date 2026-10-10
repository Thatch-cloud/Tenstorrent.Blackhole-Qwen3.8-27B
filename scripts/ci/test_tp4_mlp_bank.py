"""WP4 F-D3: the bank-strided fused gate|up launch at the TP4 shape and 64 rows (tp4_mlp_bank). CPU only; the arithmetic is the native kernel's and a card's to prove.

Held here:

  the patch      bank_compute fails closed on the anchors the first fused op uses, applies the SiLU to the gate subblock's `chunk` tiles only and not to the up subblock's, packs both
                 as bf16 to the intermediate buffer, and ends with a product loop that pairs gate tile i with up tile chunk + i, one row tile at a time; the compile-time arguments are
                 two input subblocks of `chunk` columns (the destination capacity bounds chunk at 4);
  the plan       workers, rectangle and ownership for every chunk on both grids: every one of the 136 tile columns owned exactly once, all of a worker's columns in one DRAM bank at
                 consecutive bank offsets, the same cores and requests as the read probe's bank plan (the placement its 402.7 GB/s was measured at), the buffers inside L1;
  the dataflow   the weights kernel transliterated over a reduced problem (4 banks, 20 tile columns, two K blocks, two row tiles) for every chunk: the block layout
                 [K tile][gate run][up run], the one-time zeroed padding of a short worker in both halves of the double-buffered circular buffer, the output pages, and the whole product equal
                 to the served pair of matmuls and the multiply under one accumulation order, bit for bit; negative controls for a swapped gate and up and for a wrong bank stride;
  the launch     the program descriptors over a recording fake: the kernels' named compile-time arguments are the ones the C++ reads, the runtime arguments reach the right cores, one program
                 per chip, generic_op gets [x, w1, w3, output].
"""

import hashlib
from pathlib import Path
import re
import sys
import unittest

import torch

import tp4_mlp_bank as bank
import tp4_mlp_fused as fused
import tp4_mlp_gateup as lever
from test_tp4_mlp_fused import FakeOps, NATIVE, NATIVE_WITH_HEADER, SOURCE, Tensor, T, reference, tile, unittest_patch

HERE = Path(__file__).resolve().parent
PROBE_DIR = HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'mlp_gateup'


class PatchTests(unittest.TestCase):
    def test_the_epilogue_applies_the_silu_to_the_gate_subblock_only_and_packs_bf16(self):
        for chunk in bank.CHUNKS:
            with self.subTest(chunk=chunk):
                result = bank.bank_compute(NATIVE_WITH_HEADER, chunk, 2)
                self.assertIn('static_assert(out_subblock_num_tiles == %d && out_subblock_h == 1 && in1_num_subblocks == 2);' % chunk, result)
                self.assertIn('if (in1_subblock == 0) {\n                                    apply_activation_from_pack<KernelActivation::SILU>(%d);' % chunk, result)
                self.assertEqual(result.count('apply_activation_from_pack'), 1)
                self.assertIn('} else {\n                                    tile_regs_wait();', result)
                self.assertIn('pack_block(start_dst_index, rounded_cb, %d)' % chunk, result)
                self.assertIn('cb_reserve_back(rounded_cb, %d);' % chunk, result)
                self.assertIn('cb_push_back(rounded_cb, %d);' % chunk, result)
                self.assertNotIn('old pack', result, 'the native last-block pack is replaced')
                self.assertIn('partial', result, 'the spill branch is the native one')

    def test_the_product_loop_follows_the_k_loop_and_pairs_gate_i_with_up_chunk_plus_i(self):
        for chunk in bank.CHUNKS:
            for rows in (1, 2):
                with self.subTest(chunk=chunk, rows=rows):
                    result = bank.bank_compute(NATIVE_WITH_HEADER, chunk, rows)
                    self.assertLess(result.index('partial'), result.index('mul_binary_tile_init'))
                    self.assertIn('static_assert(in0_num_subblocks == %d);' % rows, result)
                    self.assertIn('for (uint32_t row = 0; row < %d; ++row) {' % rows, result)
                    self.assertIn('cb_wait_front(rounded_cb, %d);' % (2 * chunk), result)
                    self.assertIn('cb_pop_front(rounded_cb, %d);' % (2 * chunk), result)
                    self.assertIn('for (uint32_t pair = 0; pair < %d; ++pair) {' % chunk, result)
                    self.assertIn('copy_tile(rounded_cb, pair, 0);', result)
                    self.assertIn('copy_tile(rounded_cb, %d + pair, 1);' % chunk, result)
                    self.assertIn(fused.BF16_PRODUCT, result)
                    self.assertNotIn('mul_binary_tile(0, 1, 0);', result)
                    self.assertTrue(result.startswith('#include "api/compute/eltwise_binary_sfpu.h"'))
                    self.assertIn(fused.HPP, result)

    def test_every_wait_on_the_intermediate_buffer_is_the_same_size(self):
        result = bank.bank_compute(NATIVE, 4, 2)
        self.assertEqual(set(re.findall(r'cb_wait_front\(rounded_cb, (\d+)\)', result)), {'8'})
        self.assertEqual(set(re.findall(r'cb_pop_front\(rounded_cb, (\d+)\)', result)), {'8'})

    def test_it_shares_the_first_ops_anchors_and_fails_closed(self):
        self.assertEqual(bank.fused.LAST_BLOCK, '                            if (last_out) {')
        for source in ('no matching native kernel', NATIVE + NATIVE):
            with self.assertRaisesRegex(ValueError, 'anchor'):
                bank.bank_compute(source, 4, 2)
        with self.assertRaises(ValueError):
            bank.bank_compute('                            if (last_out) {', 4, 2)            # no end anchor
        for chunk in (1, 5, 6, 8):
            with self.assertRaisesRegex(ValueError, 'chunk'):
                bank.bank_compute(NATIVE, chunk, 2)
        with self.assertRaises(ValueError):
            bank.bank_compute(NATIVE, 4, 3)

    def test_the_compile_time_arguments_are_two_subblocks_of_chunk_columns(self):
        self.assertEqual(bank.compute_arguments(4, 2, 20), [8, 2, 16, 8, 2, 64, 8, 20, 1, 1, 1, 4, 4, 1, 16, 0, 0, 0])
        self.assertEqual(bank.compute_arguments(3, 2, 20), [8, 2, 16, 8, 2, 48, 6, 20, 1, 1, 1, 3, 3, 1, 12, 0, 0, 0])
        self.assertEqual(bank.compute_arguments(2, 1, 20), [8, 1, 8, 8, 2, 32, 4, 20, 1, 1, 1, 2, 2, 1, 4, 0, 0, 0])
        for chunk in bank.CHUNKS:
            arguments = bank.compute_arguments(chunk, 2, 20)
            (block_w, in0_subblocks, in0_block_tiles, in0_subblock_tiles, in1_subblocks, in1_block_tiles, in1_block_w, blocks, unused1, unused2,
             sub_h, sub_w, sub_tiles, batch, out_block_tiles) = arguments[:15]
            self.assertEqual(in1_block_tiles, sub_w * block_w * in1_subblocks)          # the native kernel's own comment on argument 5
            self.assertEqual(in1_block_w, sub_w * in1_subblocks)                          # ... and on argument 6
            self.assertEqual(sub_tiles, sub_h * sub_w)
            self.assertEqual(in0_block_tiles, sub_h * block_w * in0_subblocks)
            self.assertEqual(out_block_tiles, in0_subblocks * in1_subblocks * sub_tiles)
            self.assertLessEqual(sub_tiles, lever.FP32_SUBBLOCK_CAP, 'the fp32 destination holds four tiles of a half')

    def test_the_chunks_are_the_levers(self):
        self.assertEqual(bank.CHUNKS, lever.BANK_CHUNKS)
        self.assertEqual(bank.DEFAULT_CHUNK, 4)
        self.assertLessEqual(max(bank.CHUNKS), lever.FP32_SUBBLOCK_CAP)


class PlanTests(unittest.TestCase):
    def test_the_workers_of_each_chunk(self):
        self.assertEqual([bank.plan(c, (13, 10))['workers'] for c in bank.CHUNKS], [72, 48, 40])
        self.assertEqual([(bank.plan(c, (13, 10))['cols'], bank.plan(c, (13, 10))['rows']) for c in bank.CHUNKS], [(13, 6), (13, 4), (13, 4)])
        self.assertEqual([(bank.plan(c, (11, 10))['cols'], bank.plan(c, (11, 10))['rows']) for c in bank.CHUNKS], [(11, 7), (11, 5), (11, 4)])

    def test_every_column_is_owned_exactly_once_and_in_one_bank_at_consecutive_offsets_on_both_grids(self):
        for grid in ((13, 10), (11, 10)):
            for chunk in bank.CHUNKS:
                with self.subTest(grid=grid, chunk=chunk):
                    found = bank.plan(chunk, grid)
                    owned = sorted(column for worker in found['mapping'] for column in bank.owned_columns(worker))
                    self.assertEqual(owned, list(range(136)))
                    for worker in found['mapping']:
                        columns = bank.owned_columns(worker)
                        self.assertTrue(1 <= worker.valid <= chunk)
                        self.assertEqual(set(column % 8 for column in columns), {worker.bank})
                        for k in (0, 1, 159):
                            pages = [k * 136 + column for column in columns]
                            self.assertEqual(set(page % 8 for page in pages), {worker.bank}, 'one bank')
                            offsets = [page // 8 for page in pages]
                            self.assertEqual(offsets, list(range(offsets[0], offsets[0] + len(offsets))), 'adjacent in the bank: one request reads them all')
                        self.assertTrue(worker.x < grid[0] and worker.y < grid[1])
                    self.assertEqual(found['cells'], found['cols'] * found['rows'])
                    self.assertGreaterEqual(found['cells'], found['workers'])
                    short = [worker.valid for worker in found['mapping'] if worker.valid < chunk]
                    self.assertEqual(short, [17 % chunk] * 8, 'only the last group of each bank is short (17 columns a bank)')

    def test_every_weight_page_is_read_exactly_once_by_the_workers_requests(self):
        for chunk in bank.CHUNKS:
            with self.subTest(chunk=chunk):
                found = bank.plan(chunk, (13, 10))
                count = {}
                for worker in found['mapping']:
                    for k in range(160):
                        base = k * 136 + worker.first
                        for tile_index in range(worker.valid):          # one request of `valid` consecutive pages of the bank
                            page = base + 8 * tile_index
                            count[page] = count.get(page, 0) + 1
                self.assertEqual(sorted(count), list(range(160 * 136)))
                self.assertEqual(set(count.values()), {1})

    def test_the_request_is_chunk_tiles_of_576_bytes(self):
        self.assertEqual([bank.plan(c, (13, 10))['request_bytes'] for c in bank.CHUNKS], [1152, 1728, 2304])

    @unittest.skipUnless((PROBE_DIR / 'readprobe.py').is_file(), 'the harness directory is not in this tree')
    def test_the_placement_and_requests_are_the_read_probes_bank_plan(self):
        sys.path.insert(0, str(PROBE_DIR))
        try:
            import readprobe
        finally:
            sys.path.remove(str(PROBE_DIR))
        for grid in ((13, 10), (11, 10)):
            for chunk in bank.CHUNKS:
                with self.subTest(grid=grid, chunk=chunk):
                    theirs = readprobe.worker_plan('bank', chunk, chunk, 160, 136, 8, grid)
                    ours = bank.plan(chunk, grid)['mapping']
                    self.assertEqual(len(theirs), len(ours))
                    for probe, worker in zip(theirs, ours):
                        self.assertEqual(probe['core'], (worker.x, worker.y))
                        self.assertEqual(probe['requests'], [(worker.first, worker.valid)])

    def test_the_rectangle_is_the_device_width_by_default_and_a_narrower_one_on_request(self):
        self.assertEqual(bank.plan(4, (13, 10))['mapping'][13][:2], (0, 1), 'worker 13 starts the second row')
        narrow = bank.plan(4, (13, 10), width=8)
        self.assertEqual((narrow['cols'], narrow['rows']), (8, 5))

    def test_grid_and_option_errors(self):
        with self.assertRaisesRegex(ValueError, 'need'):
            bank.plan(2, (13, 4))
        with self.assertRaisesRegex(ValueError, 'width'):
            bank.plan(4, (13, 10), width=14)
        for chunk in (1, 5, 8):
            with self.assertRaisesRegex(ValueError, 'chunk'):
                bank.plan(chunk, (13, 10))
        with self.assertRaisesRegex(ValueError, 'multiple'):
            bank.plan(4, (13, 10), pair_columns=135)
        with self.assertRaisesRegex(ValueError, 'multiple'):
            bank.plan(4, (13, 10), banks=7)

    def test_the_buffers_fit_l1_and_are_the_right_sizes(self):
        for chunk in bank.CHUNKS:
            found = bank.plan(chunk, (13, 10))
            self.assertLessEqual(found['worker_bytes'], bank.L1_BUDGET)
            buffers = found['buffers']
            self.assertEqual(buffers['in0']['bytes'], 2 * 8 * 2 * 2048, 'two blocks of two row tiles of eight K tiles')
            self.assertEqual(buffers['in1']['bytes'], 32 * chunk * 576, 'two blocks of eight rows of a gate run and an up run, bfloat4_b')
            self.assertEqual(buffers['partial']['bytes'], 2 * 2 * chunk * 4096, 'fp32 partials of the whole output block')
            self.assertEqual(buffers['rounded']['bytes'], 2 * 2 * chunk * 2048)
            self.assertEqual(buffers['out']['bytes'], 2 * chunk * 2048)
        self.assertEqual(bank.plan(4, (13, 10))['k_blocks'], 20)

    def test_the_default_bank_count_is_the_device_s_or_eight(self):
        class Mesh(object):
            def dram_grid_size(self):
                return type('G', (), dict(x=8, y=1))()

        self.assertEqual(bank.dram_banks(Mesh()), 8)
        self.assertEqual(bank.dram_banks(object()), 8)
        self.assertEqual(bank.dram_banks('mesh'), 8)


# ---------------------------------------------------------------------------------------------------------------------------
# The dataflow: the weights kernel transliterated over a reduced problem.
# ---------------------------------------------------------------------------------------------------------------------------

BLOCK = 8


def bank_reads(x, w1, w3, plan, row_tiles):
    """The activation and weight circular-buffer contents the kernels leave for each worker, block by block, built by tp4_mlp_bank_weights.cpp's own index arithmetic:
    returns {worker index: [(in0 block tiles, in1 block tiles)]}. The weight buffer is two halves; the padding slots of a short worker are zeroed once before the K loop in both
    halves (the buffer starts as NaN tiles so that a slot nobody writes would be seen), block b lands in half b % 2."""
    chunk, banks, pair_columns, k_blocks = plan['chunk'], plan['banks'], plan['pair_columns'], plan['k_blocks']
    row_stride = k_blocks * BLOCK
    zero = torch.zeros(T, T, dtype=torch.bfloat16)
    garbage = torch.full((T, T), float('nan'), dtype=torch.bfloat16)
    row_slots, block_slots = 2 * chunk, BLOCK * 2 * chunk
    out = {}
    for index, worker in enumerate(plan['mapping']):
        memory = [garbage] * (2 * block_slots)
        if worker.valid < chunk:                           # the one-time zeroing before the K loop
            for slot_row in range(2 * BLOCK):
                for pair in range(worker.valid, chunk):
                    memory[slot_row * row_slots + pair] = zero
                    memory[slot_row * row_slots + pair + chunk] = zero
        blocks = []
        for block in range(k_blocks):
            in0 = [None] * (row_tiles * BLOCK)
            for row in range(row_tiles):
                for t in range(BLOCK):
                    page = row * row_stride + block * BLOCK + t
                    in0[row * BLOCK + t] = tile(x, page // row_stride, page % row_stride)
            half = (block % 2) * block_slots
            for inner in range(BLOCK):
                page = (block * BLOCK + inner) * pair_columns + worker.first       # the first page of the run; `valid` pages follow in the bank
                for i in range(worker.valid):
                    wanted = page + banks * i
                    memory[half + inner * row_slots + i] = tile(w1, wanted // pair_columns, wanted % pair_columns)
                    memory[half + inner * row_slots + chunk + i] = tile(w3, wanted // pair_columns, wanted % pair_columns)
            blocks.append((in0, list(memory[half:half + block_slots])))
        out[index] = blocks
    return out


def compute_bank_worker(blocks, plan, row_tiles):
    """The native compute kernel's loops over one worker's blocks with two input subblocks of `chunk` columns (matmul_block over in0 subblock x in1 subblock, partials added block by
    block in fp32, the last block's gate subblock through SiLU and both subblocks to bf16, then the bf16 product loop): the product tiles in the order they reach the output buffer."""
    chunk = plan['chunk']
    partial = {}
    for in0, in1 in blocks:
        for sub in range(row_tiles):
            for s in range(2):
                dest = [torch.zeros(T, T, dtype=torch.float32) for unused in range(chunk)]
                for inner in range(BLOCK):
                    a = in0[sub * BLOCK + inner].float()
                    for i in range(chunk):
                        dest[i] = dest[i] + a @ in1[inner * 2 * chunk + s * chunk + i].float()
                key = (sub, s)
                partial[key] = dest if key not in partial else [old + new for old, new in zip(partial[key], dest)]
    products = []
    for sub in range(row_tiles):
        gates = [(g * torch.sigmoid(g)).to(torch.bfloat16) for g in partial[(sub, 0)]]
        ups = [u.to(torch.bfloat16) for u in partial[(sub, 1)]]
        for i in range(chunk):
            products.append((gates[i].float() * ups[i].float()).to(torch.bfloat16))
    return products


class DataflowTests(unittest.TestCase):
    ROW_TILES, K_BLOCKS, PAIR_COLUMNS, BANKS = 2, 2, 20, 4

    def problem(self, seed):
        generator = torch.Generator().manual_seed(seed)
        x = (torch.randn(self.ROW_TILES * T, self.K_BLOCKS * BLOCK * T, generator=generator) * 0.4).to(torch.bfloat16)
        w1 = (torch.randn(self.K_BLOCKS * BLOCK * T, self.PAIR_COLUMNS * T, generator=generator) * 0.1).to(torch.bfloat16)
        w3 = (torch.randn(self.K_BLOCKS * BLOCK * T, self.PAIR_COLUMNS * T, generator=generator) * 0.1).to(torch.bfloat16)
        return x, w1, w3

    def make_plan(self, chunk, banks=None):
        return bank.plan(chunk, (13, 10), row_tiles=self.ROW_TILES, k_tiles=self.K_BLOCKS * BLOCK, pair_columns=self.PAIR_COLUMNS,
                         banks=self.BANKS if banks is None else banks)

    def run_plan(self, chunk, swap=False):
        plan = self.make_plan(chunk)
        x, w1, w3 = self.problem(60 + chunk)
        reads = bank_reads(x, w3 if swap else w1, w1 if swap else w3, plan, self.ROW_TILES)
        out = torch.zeros(self.ROW_TILES * T, self.PAIR_COLUMNS * T, dtype=torch.bfloat16)
        written = []
        for index, worker in enumerate(plan['mapping']):
            products = compute_bank_worker(reads[index], plan, self.ROW_TILES)
            # the writer kernel: for row, for pair: wait one tile; a valid pair is written to page row * pair_columns + first + banks * pair
            for row in range(self.ROW_TILES):
                for pair in range(chunk):
                    if pair < worker.valid:
                        page = row * self.PAIR_COLUMNS + worker.first + self.BANKS * pair
                        written.append(page)
                        out[(page // self.PAIR_COLUMNS) * T:(page // self.PAIR_COLUMNS + 1) * T,
                            (page % self.PAIR_COLUMNS) * T:(page % self.PAIR_COLUMNS + 1) * T] = products[row * chunk + pair]
        return plan, (x, w1, w3), out, written

    def test_the_output_is_the_served_pair_and_the_multiply_for_every_chunk(self):
        for chunk in bank.CHUNKS:
            with self.subTest(chunk=chunk):
                plan, (x, w1, w3), out, written = self.run_plan(chunk)
                self.assertEqual(sorted(written), list(range(self.ROW_TILES * self.PAIR_COLUMNS)), 'every output page written exactly once')
                expected = reference(x, w1, w3, self.K_BLOCKS, self.ROW_TILES, self.PAIR_COLUMNS)
                self.assertTrue(torch.equal(out.view(torch.int16), expected.view(torch.int16)))

    def test_a_short_worker_is_zero_padded_in_both_halves_and_writes_only_its_valid_columns(self):
        plan = self.make_plan(3)                      # 5 columns a bank: groups of 3 and 2
        self.assertEqual([worker.valid for worker in plan['mapping']], [3, 2] * 4)
        x, w1, w3 = self.problem(70)
        reads = bank_reads(x, w1, w3, plan, self.ROW_TILES)
        short = plan['mapping'].index(next(worker for worker in plan['mapping'] if worker.valid == 2))
        for block, (in0, in1) in enumerate(reads[short]):
            for inner in range(BLOCK):
                for slot in (2, 3 + 2):                # the third column of the gate run and of the up run
                    self.assertTrue(torch.equal(in1[inner * 6 + slot], torch.zeros(T, T, dtype=torch.bfloat16)), (block, inner, slot))
        for chunk in bank.CHUNKS:
            plan, unused, out, written = self.run_plan(chunk)
            self.assertEqual(len(written), self.ROW_TILES * self.PAIR_COLUMNS)

    def test_a_swapped_gate_and_up_or_a_wrong_bank_stride_is_seen(self):
        plan, (x, w1, w3), good, unused = self.run_plan(4)
        expected = reference(x, w1, w3, self.K_BLOCKS, self.ROW_TILES, self.PAIR_COLUMNS)
        self.assertTrue(torch.equal(good.view(torch.int16), expected.view(torch.int16)))
        unused, unused2, swapped, unused3 = self.run_plan(4, swap=True)
        self.assertFalse(torch.equal(swapped.view(torch.int16), expected.view(torch.int16)), 'the negative control must differ')
        # the same plan read with the wrong stride (a neighbouring-column reader) reads other tiles
        plan = self.make_plan(4)
        real = bank_reads(x, w1, w3, plan, self.ROW_TILES)
        wrong = dict(plan, banks=1)
        skewed = bank_reads(x, w1, w3, wrong, self.ROW_TILES)
        self.assertNotEqual([tile.tolist() for tile in real[0][0][1][:2]], [tile.tolist() for tile in skewed[0][0][1][:2]])

    def test_the_kernels_index_arithmetic_is_the_transliterations(self):
        """The constants the transliteration uses are the ones the C++ computes with."""
        weights = (HERE / 'tp4_mlp_bank_weights.cpp').read_text(encoding='utf-8')
        self.assertIn('constexpr uint32_t row_slots = 2 * chunk;', weights)
        self.assertIn('constexpr uint32_t block_slots = block_tiles * row_slots;', weights)
        self.assertIn('const uint32_t page = (block * block_tiles + inner) * pair_columns + first;', weights)
        self.assertIn('const uint32_t gate_run = destination + inner * row_slots * 576;', weights)
        self.assertIn('noc_async_read(gate.get_noc_addr(page), gate_run, valid * 576);', weights)
        self.assertIn('noc_async_read(up.get_noc_addr(page), gate_run + chunk * 576, valid * 576);', weights)
        self.assertIn('noc_async_write_tile(row * pair_columns + first + banks * pair, output, get_read_ptr(4));', weights)
        self.assertIn('cb_wait_front(4, 1)', weights)
        self.assertIn('cb_reserve_back(1, block_slots)', weights)
        self.assertIn('words[word] = 0;', weights)
        self.assertIn('word < 144', weights)                 # 576 bytes of zero tile

    def test_the_padding_is_zeroed_once_before_the_k_loop_in_both_kernels(self):
        for name, loop in (('tp4_mlp_bank_weights.cpp', 'for (uint32_t block = 0; block < k_blocks; ++block) {'),
                           ('tp4_mlp_fused_weights.cpp', 'for (uint32_t block = 0; block < k_blocks; ++block) {')):
            with self.subTest(kernel=name):
                text = (HERE / name).read_text(encoding='utf-8')
                self.assertEqual(text.count(loop), 1)
                before, inside = text.split(loop)
                body = inside.split('cb_push_back(1,')[0]
                self.assertIn('word < 144', before)
                self.assertNotIn('= 0;', body.replace('block = 0;', '').replace('inner = 0;', '').replace('pair = 0;', ''), 'no zero store inside the K loop')
                self.assertIn('get_write_ptr(1)', before, 'the base of the buffer, before the first reservation')
                self.assertIn('2 * block_tiles', before, 'both halves')

    def test_the_first_fused_kernel_still_reads_only_valid_pairs_in_the_k_loop(self):
        text = (HERE / 'tp4_mlp_fused_weights.cpp').read_text(encoding='utf-8')
        loop = text.split('for (uint32_t block = 0; block < k_blocks; ++block) {')[1]
        self.assertIn('for (uint32_t pair = 0; pair < valid_pairs; ++pair) {', loop)
        self.assertNotIn('else {', loop.split('cb_push_back(1,')[0])


# ---------------------------------------------------------------------------------------------------------------------------
# The launch: program descriptors over a recording fake.
# ---------------------------------------------------------------------------------------------------------------------------

def make_op(ops=None, chunk=4, grid=(13, 10), width=None, approx=True, chips=4, banks=8, **options):
    ops = ops or FakeOps(chips)
    x = Tensor((1, 1, 64, 5120), ops.bfloat16, 100, chips)
    w1, w3 = Tensor((1, 1, 5120, 17408), ops.bfloat4_b, 200, chips), Tensor((1, 1, 5120, 17408), ops.bfloat4_b, 300, chips)
    bank._COMPUTE_CACHE.clear()
    with unittest_patch(fused, 'native_source', lambda root: SOURCE):
        op = bank.BankGateUp(ops, 'mesh', w1, w3, chunk=chunk, grid=grid, width=width, math_approx_mode=approx, banks=banks, **options)
    return ops, op, (x, w1, w3)


class LaunchTests(unittest.TestCase):
    def kernels_of(self, op, x, ops):
        op(x)
        return next(iter(ops.launched[0][1].values())).kernels

    def test_the_named_compile_time_arguments_are_the_ones_the_kernels_read(self):
        ops, op, (x, w1, w3) = make_op()
        input_kernel, writer, compute = self.kernels_of(op, x, ops)
        for name, kernel in ((bank.KERNELS[0], input_kernel), (bank.KERNELS[1], writer)):
            source = (HERE / name).read_text(encoding='utf-8')
            read = set(re.findall(r'get_named_compile_time_arg_val\("(\w+)"\)', source))
            given = set(key for key, value in kernel.named_compile_time_args)
            self.assertEqual(read, given, name)
            self.assertTrue(kernel.kernel_source.endswith(name))
        values = dict(writer.named_compile_time_args)
        self.assertEqual((values['chunk'], values['pair_columns'], values['k_blocks'], values['block_tiles'], values['row_tiles'], values['banks']),
                         (4, 136, 20, 8, 2, 8))
        self.assertEqual(dict(input_kernel.named_compile_time_args)['row_stride'], 160)

    def test_the_activation_kernel_is_the_first_fused_ops_unchanged(self):
        self.assertEqual(bank.KERNELS[0], fused.KERNELS[0])
        self.assertEqual(bank.kernel_sources()[bank.KERNELS[0]], fused.kernel_sources()[fused.KERNELS[0]])

    def test_the_tensor_accessors_come_in_the_order_the_kernel_declares_them(self):
        ops, op, (x, w1, w3) = make_op()
        input_kernel, writer, compute = self.kernels_of(op, x, ops)
        self.assertEqual(input_kernel.compile_time_args, [1100])
        self.assertEqual(writer.compile_time_args, [1200, 1300, 1000 + 9000])
        source = (HERE / 'tp4_mlp_bank_weights.cpp').read_text(encoding='utf-8')
        self.assertEqual([match.group(1) for match in re.finditer(r'constexpr auto (\w+)_args = TensorAccessorArgs', source)], ['gate', 'up', 'output'])

    def test_one_program_per_chip_with_that_chips_addresses(self):
        ops, op, (x, w1, w3) = make_op()
        output = op(x)
        tensors, mesh_program = ops.launched[0]
        self.assertEqual(tensors, [x, w1, w3, output])
        self.assertEqual(len(mesh_program), 4)
        for chip, program in enumerate(mesh_program.values()):
            input_kernel, writer, compute = program.kernels
            self.assertEqual(input_kernel.runtime_args[0][0][0], 100 + chip)
            self.assertEqual(writer.runtime_args[0][0][:3], [200 + chip, 300 + chip, 9000 + chip])

    def test_the_writer_arguments_are_the_workers_first_column_and_valid_count(self):
        for chunk in bank.CHUNKS:
            ops, op, (x, w1, w3) = make_op(chunk=chunk)
            writer = self.kernels_of(op, x, ops)[1]
            for worker in op.plan['mapping']:
                self.assertEqual(writer.runtime_args[worker.x][worker.y][3:], [worker.first, worker.valid])
            self.assertEqual(len(op.plan['mapping']), {2: 72, 3: 48, 4: 40}[chunk])

    def test_the_input_runtime_arguments_cover_the_rectangle_and_name_its_corners(self):
        ops, op, (x, w1, w3) = make_op()
        input_kernel = self.kernels_of(op, x, ops)[0]
        plan = op.plan
        seen = []
        for core_x in range(plan['cols']):
            for core_y in range(plan['rows']):
                arguments = input_kernel.runtime_args[core_x][core_y]
                seen.append(arguments[1])
                self.assertEqual(arguments[1], core_y * plan['cols'] + core_x)
                self.assertEqual(arguments[2:6], [1, 2, plan['cols'], plan['rows'] + 1])
                self.assertEqual(arguments[6:], [plan['workers'], plan['cells']])
        self.assertEqual(sorted(seen), list(range(plan['cells'])))

    def test_the_circular_buffers_are_the_plans(self):
        for chunk in bank.CHUNKS:
            ops, op, (x, w1, w3) = make_op(chunk=chunk)
            op(x)
            program = next(iter(ops.launched[0][1].values()))
            sizes = dict((cb.format_descriptors[0].buffer_index, cb.total_size) for cb in program.cbs)
            self.assertEqual(sizes, dict((item['index'], item['bytes']) for item in op.plan['buffers'].values()))
            self.assertEqual(len(program.semaphores), 2)

    def test_the_compute_kernel_is_the_patched_native_source_with_the_served_math_mode(self):
        for approx in (True, False):
            ops, op, (x, w1, w3) = make_op(approx=approx, chunk=3)
            compute = self.kernels_of(op, x, ops)[2]
            self.assertEqual(compute.source_type, 'code')
            self.assertEqual(compute.kernel_source, bank.bank_compute(SOURCE, 3, 2))
            self.assertEqual(compute.compile_time_args, bank.compute_arguments(3, 2, 20))
            self.assertEqual(compute.config.math_approx_mode, approx)
            self.assertEqual((compute.config.math_fidelity, compute.config.fp32_dest_acc_en), ('lofi', True))
            self.assertEqual(compute.config.unpack_to_dest_mode[5], 'fp32')
            self.assertIn(('PACKER_L1_ACC', '1'), compute.defines)
            self.assertIn(('SFPU_ACTIVATION', '1'), compute.defines)
            self.assertEqual(op.manifest['math_approx_mode'], approx)

    def test_the_output_is_a_fresh_l1_tile_tensor_of_the_products_shape(self):
        ops, op, (x, w1, w3) = make_op()
        output = op(x)
        tensor, device, memory, layout = ops.outputs[0]
        self.assertEqual((tensor.shape, tensor.dtype, device, memory, layout), ((1, 1, 64, 4352), 'bf16', 'mesh', 'L1', 'tile'))
        self.assertIs(output, tensor)

    def test_inputs_are_checked(self):
        ops, op, (x, w1, w3) = make_op()
        with self.assertRaisesRegex(ValueError, 'input must be'):
            op(Tensor((1, 1, 32, 5120), 'bf16', 100))
        op.w3 = Tensor((1, 1, 5120, 17408), 'bf8', 300)
        with self.assertRaisesRegex(ValueError, 'bfloat4_b'):
            op(x)
        self.assertEqual(ops.launched, [])

    def test_constructor_errors(self):
        for bad in (None, 1, 'true'):
            with self.assertRaisesRegex(ValueError, 'math approximation'):
                make_op(approx=bad)
        with self.assertRaises(ValueError):
            make_op(rows=48)
        for chunk in (1, 5, 6):
            with self.assertRaisesRegex(ValueError, 'chunk'):
                make_op(chunk=chunk)
        with self.assertRaisesRegex(ValueError, 'multiple'):
            make_op(banks=7)

    def test_the_bank_count_comes_from_the_device_when_not_given(self):
        class Mesh(object):
            def dram_grid_size(self):
                return type('G', (), dict(x=8, y=1))()

        ops = FakeOps()
        w = Tensor((1, 1, 5120, 17408), 'bf4', 200)
        bank._COMPUTE_CACHE.clear()
        with unittest_patch(fused, 'native_source', lambda root: SOURCE):
            op = bank.BankGateUp(ops, Mesh(), w, w, math_approx_mode=True)
        self.assertEqual(op.banks, 8)

    def test_the_native_source_is_read_once_per_shape_and_hashed(self):
        reads = []
        ops = FakeOps()
        bank._COMPUTE_CACHE.clear()
        w = Tensor((1, 1, 5120, 17408), 'bf4', 200)
        with unittest_patch(fused, 'native_source', lambda root: reads.append(root) or SOURCE):
            first = bank.BankGateUp(ops, 'mesh', w, w, math_approx_mode=True, banks=8)
            second = bank.BankGateUp(ops, 'mesh', w, w, math_approx_mode=True, banks=8)
            bank.BankGateUp(ops, 'mesh', w, w, chunk=3, math_approx_mode=True, banks=8)
        self.assertEqual(len(reads), 2)
        self.assertEqual(first.manifest['native_compute_sha256'], hashlib.sha256(SOURCE.encode()).hexdigest())
        self.assertEqual(first.manifest['bank_compute_sha256'], second.manifest['bank_compute_sha256'])
        self.assertEqual(first.manifest['kernels'], bank.kernel_sources())
        self.assertEqual(set(first.manifest['kernels']), set(bank.KERNELS))

    def test_it_does_not_free_its_input_and_counts_its_launches(self):
        ops, op, (x, w1, w3) = make_op()
        op(x)
        op(x)
        self.assertEqual(op.calls, 2)
        self.assertEqual(len(ops.launched), 2)


class SourceTests(unittest.TestCase):
    def test_the_module_does_not_edit_the_pinned_graft_or_the_first_ops_files_beyond_the_padding_fix(self):
        self.assertIn(b'class FusedProjection', (HERE / 'fused_1d.py').read_bytes())
        text = (HERE / 'tp4_mlp_bank.py').read_text(encoding='utf-8')
        self.assertIn('import tp4_mlp_fused as fused', text)
        self.assertNotIn('fused_1d', text.split('"""')[2], 'the code does not import the TP2 module')


if __name__ == '__main__':
    unittest.main()
