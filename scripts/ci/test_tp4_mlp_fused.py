"""WP4 F-D1: the fused gate|up launch at the TP4 shape and 64 rows (tp4_mlp_fused). CPU only; the arithmetic is the native kernel's and a card's to prove.

Held here:

  the patch      fused_compute equals fused_1d.fused_compute (the TP2 port, pinned) byte for byte at one row tile and the TP2 pairs-per-worker, counts the right
                 pairs at two row tiles and at the new pairs-per-worker, and fails closed when the native kernel loses an anchor;
  the plan       workers, rectangle and pair mapping on both grids: every one of the 136 pairs owned exactly once in ascending order, the buffers inside the
                 L1 budget, the compile-time arguments the TP2 list's with the 64-row terms;
  the dataflow   the two kernels transliterated over a reduced problem (two K blocks, seven pairs, two row tiles, three pairs a worker): the activation block layout,
                 the weight block layout [K tile][g0 u0 g1 u1 ...] with the zero tiles of a short worker, and the output pages are exactly what the native compute
                 kernel reads and what the served pair of matmuls and the multiply produce, bit for bit under one shared accumulation order;
  the launch     the program descriptors one chip at a time: the kernels' named compile-time arguments are the ones the C++ reads, the runtime arguments reach
                 the right cores, the circular buffers have the plan's sizes, one program per chip, generic_op gets [x, w1, w3, output].
"""

import hashlib
from pathlib import Path
import re
import types
import unittest

import torch

import fused_1d
import test_verify_trace_t1_graft as graft
import tp4_mlp_fused as fused
import tp4_mlp_gateup as lever

HERE = Path(__file__).resolve().parent

NATIVE = ('// native kernel\nvoid MAIN {\n' + '                            if (last_out) {\n                                old pack\n'
          + '                            } else {\n                                tile_regs_commit();\n                                partial\n'
          + '                            }\n    // end of the K loop\n}\n')
NATIVE_WITH_HEADER = '#include "bmm_fused_activation.hpp"\n' + NATIVE


class PatchTests(unittest.TestCase):
    def test_it_equals_the_pinned_tp2_port_where_that_supports(self):
        for pairs in (3, 4, 5, 7):
            with self.subTest(pairs=pairs):
                self.assertEqual(fused.fused_compute(NATIVE_WITH_HEADER, pairs, 1), fused_1d.fused_compute(NATIVE_WITH_HEADER, pairs_per_worker=pairs))

    def test_the_epilogue_and_the_product_are_the_pinned_ones(self):
        self.assertEqual(fused.BF16_PRODUCT, fused_1d.BF16_PRODUCT)
        self.assertEqual(fused.LAST_BLOCK, '                            if (last_out) {')
        result = fused.fused_compute(NATIVE_WITH_HEADER, 3, 2)
        self.assertIn('apply_activation_from_pack<KernelActivation::SILU>(1)', result)
        self.assertIn('pack_block(start_dst_index, rounded_cb, 2)', result)
        self.assertIn(fused.BF16_PRODUCT, result)
        self.assertNotIn('mul_binary_tile(0, 1, 0);', result)
        self.assertLess(result.index('partial'), result.index('mul_binary_tile_init'), 'the product loop follows the K loop')
        self.assertTrue(result.startswith('#include "api/compute/eltwise_binary_sfpu.h"'))
        self.assertIn('"ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"', result)

    def test_two_row_tiles_wait_for_and_consume_twice_the_pairs(self):
        for pairs in (2, 3, 7):
            one, two = fused.fused_compute(NATIVE, pairs, 1), fused.fused_compute(NATIVE, pairs, 2)
            self.assertIn('cb_wait_front(rounded_cb, %d);' % (2 * pairs), one)
            self.assertIn('pair < %d;' % pairs, one)
            self.assertIn('cb_wait_front(rounded_cb, %d);' % (4 * pairs), two)
            self.assertIn('pair < %d;' % (2 * pairs), two)

    def test_fails_closed(self):
        for source in ('no matching native kernel', NATIVE + NATIVE):
            with self.assertRaisesRegex(ValueError, 'anchor'):
                fused.fused_compute(source, 3, 2)
        with self.assertRaises(ValueError):
            fused.fused_compute(NATIVE, 6, 2)
        with self.assertRaises(ValueError):
            fused.fused_compute(NATIVE, 3, 3)
        with self.assertRaises(ValueError):
            fused.fused_compute('                            if (last_out) {', 3, 2)       # no end anchor

    def test_the_compile_time_arguments_are_the_tp2_list_with_the_row_terms(self):
        for pairs in (3, 4, 5, 7):
            self.assertEqual(fused.compute_arguments(pairs, 1, 20),
                             [8, 1, 8, 8, pairs, 16 * pairs, 2 * pairs, 20, 1, 1, 1, 2, 2, 1, 2 * pairs, 0, 0, 0])
        two = fused.compute_arguments(3, 2, 20)
        self.assertEqual(two, [8, 2, 16, 8, 3, 48, 6, 20, 1, 1, 1, 2, 2, 1, 12, 0, 0, 0])
        self.assertEqual(len(two), 18)


class PlanTests(unittest.TestCase):
    def test_the_workers_of_each_pairs_per_worker(self):
        self.assertEqual([fused.plan(p, (13, 10))['workers'] for p in lever.PAIRS_PER_WORKER], [68, 46, 34, 28, 20])

    def test_every_pair_is_owned_exactly_once_in_ascending_order_on_both_grids(self):
        for grid in ((13, 10), (11, 10)):
            for pairs in lever.PAIRS_PER_WORKER:
                with self.subTest(grid=grid, pairs=pairs):
                    found = fused.plan(pairs, grid)
                    owned = [pair for x, y, begin, count in found['mapping'] for pair in range(begin, begin + count)]
                    self.assertEqual(owned, list(range(136)))
                    self.assertTrue(all(count >= 1 for unused1, unused2, unused3, count in found['mapping']))
                    self.assertEqual(found['mapping'][-1][3], 136 - (found['workers'] - 1) * pairs)
                    self.assertTrue(all(x < grid[0] and y < grid[1] for x, y, begin, count in found['mapping']))
                    self.assertEqual(found['cells'], found['cols'] * found['rows'])
                    self.assertGreaterEqual(found['cells'], found['workers'])

    def test_the_rectangle_is_the_device_width_by_default(self):
        self.assertEqual((fused.plan(3, (13, 10))['cols'], fused.plan(3, (13, 10))['rows']), (13, 4))
        self.assertEqual((fused.plan(3, (11, 10))['cols'], fused.plan(3, (11, 10))['rows']), (11, 5))
        self.assertEqual((fused.plan(3, (13, 10), width=8)['cols'], fused.plan(3, (13, 10), width=8)['rows']), (8, 6))
        self.assertEqual(fused.plan(3, (13, 10))['mapping'][13][:2], (0, 1), 'worker 13 starts the second row')

    def test_grid_and_option_errors(self):
        with self.assertRaisesRegex(ValueError, 'need'):
            fused.plan(2, (13, 4))
        with self.assertRaisesRegex(ValueError, 'width'):
            fused.plan(3, (13, 10), width=14)
        for pairs in (1, 6, 8):
            with self.assertRaises(ValueError):
                fused.plan(pairs, (13, 10))

    def test_the_buffers_fit_l1_and_are_the_right_sizes(self):
        for pairs in lever.PAIRS_PER_WORKER:
            found = fused.plan(pairs, (13, 10))
            self.assertLessEqual(found['worker_bytes'], fused.L1_BUDGET)
            buffers = found['buffers']
            self.assertEqual(buffers['in0']['bytes'], 2 * 8 * 2 * 2048, 'two blocks of two row tiles of eight K tiles')
            self.assertEqual(buffers['in1']['bytes'], 32 * pairs * 576, 'two blocks of 16 pairs-of-tiles a pair, bfloat4_b')
            self.assertEqual(buffers['partial']['bytes'], 2 * 2 * pairs * 4096, 'fp32 partials of the whole output block')
            self.assertEqual(buffers['rounded']['bytes'], 2 * 2 * pairs * 2048)
            self.assertEqual(buffers['out']['bytes'], 2 * pairs * 2048)
        self.assertEqual(fused.plan(3, (13, 10))['k_blocks'], 20)


# ---------------------------------------------------------------------------------------------------------------------------
# The dataflow: the two kernels transliterated, over a reduced problem.
# ---------------------------------------------------------------------------------------------------------------------------

T = 32


def tile(matrix, row, col):
    return matrix[row * T:(row + 1) * T, col * T:(col + 1) * T]


def kernel_reads(x, w1, w3, plan, row_tiles):
    """The activation and weight circular-buffer contents the two kernels leave for each worker, block by block, built by the kernels' own index arithmetic
    (tp4_mlp_fused_input.cpp / tp4_mlp_fused_weights.cpp): returns {worker: [(in0 block tiles, in1 block tiles)]} with tiles as (label, tensor)."""
    k_blocks, pair_columns, block_tiles = plan['k_blocks'], plan['pair_columns'], 8
    row_stride = k_blocks * block_tiles
    zero = torch.zeros(T, T, dtype=torch.bfloat16)
    out = {}
    for index, (cx, cy, first_pair, valid) in enumerate(plan['mapping']):
        blocks = []
        for block in range(k_blocks):
            # activation: worker 0 reads page row * row_stride + block * block_tiles + tile into slot row * block_tiles + tile
            in0 = [None] * (row_tiles * block_tiles)
            for row in range(row_tiles):
                for t in range(block_tiles):
                    page = row * row_stride + block * block_tiles + t
                    in0[row * block_tiles + t] = tile(x, page // row_stride, page % row_stride)
            # weights: slot (inner * 2p + 2 * pair + which) holds page (block * 8 + inner) * pair_columns + first_pair + pair of w1 (which 0) or w3 (1)
            p = plan['pairs_per_worker']
            in1 = [None] * (block_tiles * 2 * p)
            for inner in range(block_tiles):
                for pair in range(p):
                    gate_slot = inner * 2 * p + 2 * pair
                    if pair < valid:
                        page = (block * block_tiles + inner) * pair_columns + first_pair + pair
                        in1[gate_slot] = tile(w1, page // pair_columns, page % pair_columns)
                        in1[gate_slot + 1] = tile(w3, page // pair_columns, page % pair_columns)
                    else:
                        in1[gate_slot] = in1[gate_slot + 1] = zero
            blocks.append((in0, in1))
        out[index] = blocks
    return out


def compute_worker(blocks, plan, row_tiles):
    """The native compute kernel's loops over one worker's blocks (matmul_block over in0 subblock x in1 subblock, partials added block by block in fp32, the last
    block packed through SiLU and bf16, then the bf16 product loop), returning the product tiles in the order they reach the output circular buffer."""
    p, block_tiles = plan['pairs_per_worker'], 8
    partial = {}
    for in0, in1 in blocks:
        for sub in range(row_tiles):                 # in0 subblocks: one row tile each
            for pair in range(p):                    # in1 subblocks: one gate|up pair each
                dest = torch.zeros(2, T, T, dtype=torch.float32)
                for inner in range(block_tiles):
                    a = in0[sub * block_tiles + inner].float()
                    for which in range(2):
                        dest[which] = dest[which] + a @ in1[inner * 2 * p + 2 * pair + which].float()
                key = (sub, pair)
                partial[key] = dest if key not in partial else partial[key] + dest
    rounded = []
    for sub in range(row_tiles):
        for pair in range(p):
            gate = partial[(sub, pair)][0]
            gate = (gate * torch.sigmoid(gate)).to(torch.bfloat16)
            up = partial[(sub, pair)][1].to(torch.bfloat16)
            rounded.append((gate, up))
    return [(g.float() * u.float()).to(torch.bfloat16) for g, u in rounded]


def reference(x, w1, w3, k_blocks, row_tiles, pair_columns):
    """The served pair of matmuls and their multiply under the same accumulation order: per output tile, K blocks of eight tiles each summed in fp32 and the block
    sums added, SiLU on the gate before its bf16 rounding, a bf16 product."""
    out = torch.zeros(row_tiles * T, pair_columns * T, dtype=torch.bfloat16)
    for row in range(row_tiles):
        for col in range(pair_columns):
            total = [torch.zeros(T, T), torch.zeros(T, T)]
            for block in range(k_blocks):
                dest = [torch.zeros(T, T), torch.zeros(T, T)]
                for inner in range(8):
                    k = block * 8 + inner
                    for which, w in enumerate((w1, w3)):
                        dest[which] = dest[which] + tile(x, row, k).float() @ tile(w, k, col).float()
                total = [total[0] + dest[0], total[1] + dest[1]]
            gate = (total[0] * torch.sigmoid(total[0])).to(torch.bfloat16)
            up = total[1].to(torch.bfloat16)
            out[row * T:(row + 1) * T, col * T:(col + 1) * T] = (gate.float() * up.float()).to(torch.bfloat16)
    return out


class DataflowTests(unittest.TestCase):
    ROW_TILES, K_BLOCKS, PAIR_COLUMNS = 2, 2, 7

    def problem(self, seed):
        generator = torch.Generator().manual_seed(seed)
        x = (torch.randn(self.ROW_TILES * T, self.K_BLOCKS * 8 * T, generator=generator) * 0.4).to(torch.bfloat16)
        w1 = (torch.randn(self.K_BLOCKS * 8 * T, self.PAIR_COLUMNS * T, generator=generator) * 0.1).to(torch.bfloat16)
        w3 = (torch.randn(self.K_BLOCKS * 8 * T, self.PAIR_COLUMNS * T, generator=generator) * 0.1).to(torch.bfloat16)
        return x, w1, w3

    def run_plan(self, pairs):
        plan = fused.plan(pairs, (13, 10), row_tiles=self.ROW_TILES, k_tiles=self.K_BLOCKS * 8, pair_columns=self.PAIR_COLUMNS)
        x, w1, w3 = self.problem(30 + pairs)
        reads = kernel_reads(x, w1, w3, plan, self.ROW_TILES)
        out = torch.zeros(self.ROW_TILES * T, self.PAIR_COLUMNS * T, dtype=torch.bfloat16)
        written = []
        for index, (cx, cy, first_pair, valid) in enumerate(plan['mapping']):
            products = compute_worker(reads[index], plan, self.ROW_TILES)
            # the writer kernel: for row, for pair: wait one tile; a valid pair is written to page row * pair_columns + first_pair + pair
            for row in range(self.ROW_TILES):
                for pair in range(pairs):
                    product = products[row * pairs + pair]
                    if pair < valid:
                        page = row * self.PAIR_COLUMNS + first_pair + pair
                        written.append(page)
                        out[(page // self.PAIR_COLUMNS) * T:(page // self.PAIR_COLUMNS + 1) * T,
                            (page % self.PAIR_COLUMNS) * T:(page % self.PAIR_COLUMNS + 1) * T] = product
        return plan, (x, w1, w3), out, written

    def test_the_output_is_the_served_pair_and_the_multiply_for_every_pairs_per_worker(self):
        for pairs in lever.PAIRS_PER_WORKER:
            with self.subTest(pairs=pairs):
                plan, (x, w1, w3), out, written = self.run_plan(pairs)
                self.assertEqual(sorted(written), list(range(self.ROW_TILES * self.PAIR_COLUMNS)), 'every output page written exactly once')
                expected = reference(x, w1, w3, self.K_BLOCKS, self.ROW_TILES, self.PAIR_COLUMNS)
                self.assertTrue(torch.equal(out.view(torch.int16), expected.view(torch.int16)))

    def test_a_short_worker_is_zero_padded_and_writes_only_its_valid_pairs(self):
        plan, unused, out, written = self.run_plan(3)
        self.assertEqual([count for unused1, unused2, unused3, count in plan['mapping']], [3, 3, 1])
        self.assertEqual(len(written), 14)

    def test_a_swapped_gate_and_up_slot_or_a_wrong_page_stride_is_seen(self):
        plan = fused.plan(3, (13, 10), row_tiles=self.ROW_TILES, k_tiles=16, pair_columns=self.PAIR_COLUMNS)
        x, w1, w3 = self.problem(40)
        reads = kernel_reads(x, w1, w3, plan, self.ROW_TILES)
        expected = reference(x, w1, w3, self.K_BLOCKS, self.ROW_TILES, self.PAIR_COLUMNS)
        reads_swapped = kernel_reads(x, w3, w1, plan, self.ROW_TILES)
        products = compute_worker(reads_swapped[0], plan, self.ROW_TILES)
        self.assertFalse(torch.equal(products[0].view(torch.int16), tile(expected, 0, 0).view(torch.int16)), 'the negative control must differ')
        good = compute_worker(reads[0], plan, self.ROW_TILES)
        self.assertTrue(torch.equal(good[0].view(torch.int16), tile(expected, 0, 0).view(torch.int16)))

    def test_the_kernels_index_arithmetic_is_the_transliterations(self):
        """The constants the transliteration uses are the ones the C++ computes with."""
        weights = (HERE / 'tp4_mlp_fused_weights.cpp').read_text(encoding='utf-8')
        inputs = (HERE / 'tp4_mlp_fused_input.cpp').read_text(encoding='utf-8')
        self.assertIn('(block * block_tiles + inner) * pair_columns + first_pair + pair', weights)
        self.assertIn('(inner * 2 * pairs_per_worker + 2 * pair) * 576', weights)
        self.assertIn('gate_tile + 576', weights)
        self.assertIn('row * pair_columns + first_pair + pair', weights)
        self.assertIn('row * row_stride + block * block_tiles + tile', inputs)
        self.assertIn('destination + (row * block_tiles + tile) * 2048', inputs)
        self.assertIn('for (uint32_t word = 0; word < 144; ++word)', weights)          # 576 bytes of zero tile
        self.assertIn('if (pair < valid_pairs)', weights)
        self.assertIn('cb_wait_front(4, 1)', weights)
        self.assertIn('cb_reserve_back(1, 2 * block_tiles * pairs_per_worker)', weights)


# ---------------------------------------------------------------------------------------------------------------------------
# The launch: program descriptors over a recording fake.
# ---------------------------------------------------------------------------------------------------------------------------

class Rec(object):
    def __init__(self, *args, **options):
        self.args = args
        self.__dict__.update(options)


class RuntimeArgs(dict):
    def __getitem__(self, key):
        return self.setdefault(key, {})


class Coord(object):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        return 'C(%d,%d)' % (self.x, self.y)


class Device(object):
    def worker_core_from_logical_core(self, coord):
        return Coord(coord.x + 1, coord.y + 2)


class Shard(object):
    def __init__(self, address):
        self.address = address

    def device(self):
        return Device()

    def buffer_address(self):
        return self.address


class Tensor(object):
    def __init__(self, shape, dtype, base, chips=4):
        self.shape, self.dtype = shape, dtype
        self.shards = [Shard(base + chip) for chip in range(chips)]


class MeshProgram(dict):
    pass


class FakeOps(object):
    bfloat16, bfloat4_b, float32, TILE_LAYOUT, L1_MEMORY_CONFIG = 'bf16', 'bf4', 'f32', 'tile', 'L1'
    MathFidelity = types.SimpleNamespace(LoFi='lofi')
    UnpackToDestMode = types.SimpleNamespace(Default='default', UnpackToDestFp32='fp32')
    DataMovementProcessor = types.SimpleNamespace(RISCV_0='r0', RISCV_1='r1')
    NOC = types.SimpleNamespace(RISCV_0_default='noc0', RISCV_1_default='noc1')

    def __init__(self, chips=4):
        self.chips = chips
        self.launched = []
        self.outputs = []
        self.KernelDescriptor = type('KernelDescriptor', (Rec,), {'SourceType': types.SimpleNamespace(SOURCE_CODE='code')})

    CoreCoord = Coord
    CoreRange = Rec
    CoreRangeSet = Rec
    CBDescriptor = Rec
    CBFormatDescriptor = Rec
    TileDescriptor = Rec
    Tile = Rec
    DataMovementConfigDescriptor = Rec
    ProgramDescriptor = Rec
    SemaphoreDescriptor = Rec
    MeshCoordinate = Rec
    MeshCoordinateRange = Rec
    RuntimeArgs = RuntimeArgs
    MeshProgramDescriptor = MeshProgram

    def ComputeConfigDescriptor(self, **options):
        return types.SimpleNamespace(unpack_to_dest_mode=[], **options)

    def TensorAccessorArgs(self, tensor):
        return types.SimpleNamespace(get_compile_time_args=lambda: [1000 + tensor.buffer_address()])

    def get_device_tensors(self, tensor):
        return tensor.shards

    def empty(self, shape, dtype, layout, device, memory_config):
        tensor = Tensor(tuple(shape), dtype, 9000, self.chips)
        self.outputs.append((tensor, device, memory_config, layout))
        return tensor

    def generic_op(self, tensors, program):
        self.launched.append((tensors, program))


SOURCE = NATIVE_WITH_HEADER


def make_op(ops=None, pairs=3, grid=(13, 10), width=None, approx=True, chips=4, **options):
    ops = ops or FakeOps(chips)
    x = Tensor((1, 1, 64, 5120), ops.bfloat16, 100, chips)
    w1, w3 = Tensor((1, 1, 5120, 17408), ops.bfloat4_b, 200, chips), Tensor((1, 1, 5120, 17408), ops.bfloat4_b, 300, chips)
    fused._COMPUTE_CACHE.clear()
    with unittest_patch(fused, 'native_source', lambda root: SOURCE):
        op = fused.FusedGateUp(ops, 'mesh', w1, w3, pairs_per_worker=pairs, grid=grid, width=width, math_approx_mode=approx, **options)
    return ops, op, (x, w1, w3)


def unittest_patch(module, name, value):
    from unittest import mock
    return mock.patch.object(module, name, value)


class LaunchTests(unittest.TestCase):
    def test_the_named_compile_time_arguments_are_the_ones_the_kernels_read(self):
        ops, op, (x, w1, w3) = make_op()
        output = op(x)
        program = ops.launched[0][1]
        kernels = {}
        for key, entry in program.items():
            input_kernel, writer, compute = entry.kernels
            kernels = dict(input=input_kernel, writer=writer, compute=compute)
            break
        for name, kernel in (('tp4_mlp_fused_input.cpp', kernels['input']), ('tp4_mlp_fused_weights.cpp', kernels['writer'])):
            source = (HERE / name).read_text(encoding='utf-8')
            read = set(re.findall(r'get_named_compile_time_arg_val\("(\w+)"\)', source))
            given = set(key for key, value in kernel.named_compile_time_args)
            self.assertEqual(read, given, name)
            self.assertTrue(kernel.kernel_source.endswith(name))
        values = dict(kernels['writer'].named_compile_time_args)
        self.assertEqual((values['pairs_per_worker'], values['pair_columns'], values['k_blocks'], values['block_tiles'], values['row_tiles']), (3, 136, 20, 8, 2))
        self.assertEqual(dict(kernels['input'].named_compile_time_args)['row_stride'], 160)

    def test_the_tensor_accessors_come_in_the_order_the_kernel_declares_them(self):
        ops, op, (x, w1, w3) = make_op()
        op(x)
        program = next(iter(ops.launched[0][1].values()))
        input_kernel, writer, compute = program.kernels
        self.assertEqual(input_kernel.compile_time_args, [1100])
        self.assertEqual(writer.compile_time_args, [1200, 1300, 1000 + 9000])
        source = (HERE / 'tp4_mlp_fused_weights.cpp').read_text(encoding='utf-8')
        order = [match.group(1) for match in re.finditer(r'constexpr auto (\w+)_args = TensorAccessorArgs', source)]
        self.assertEqual(order, ['gate', 'up', 'output'])

    def test_one_program_per_chip_with_that_chips_addresses(self):
        ops, op, (x, w1, w3) = make_op()
        output = op(x)
        tensors, mesh_program = ops.launched[0]
        self.assertEqual(tensors, [x, w1, w3, output])
        self.assertEqual(len(mesh_program), 4)
        programs = list(mesh_program.values())
        for chip, program in enumerate(programs):
            input_kernel, writer, compute = program.kernels
            first = input_kernel.runtime_args[0][0]
            self.assertEqual(first[0], 100 + chip, 'the activation address of this chip')
            self.assertEqual(writer.runtime_args[0][0][:3], [200 + chip, 300 + chip, 9000 + chip])

    def test_the_input_runtime_arguments_cover_the_rectangle_and_name_its_corners(self):
        ops, op, (x, w1, w3) = make_op()
        op(x)
        program = next(iter(ops.launched[0][1].values()))
        input_kernel = program.kernels[0]
        plan = op.plan
        seen = []
        for core_x in range(plan['cols']):
            for core_y in range(plan['rows']):
                arguments = input_kernel.runtime_args[core_x][core_y]
                seen.append(arguments[1])
                self.assertEqual(arguments[1], core_y * plan['cols'] + core_x)
                self.assertEqual(arguments[2:6], [1, 2, plan['cols'], plan['rows'] + 1], 'first and last physical corners (the fake adds 1, 2)')
                self.assertEqual(arguments[6:], [plan['workers'], plan['cells']])
        self.assertEqual(sorted(seen), list(range(plan['cells'])))

    def test_the_writer_arguments_are_the_pair_mapping(self):
        ops, op, (x, w1, w3) = make_op(pairs=4)
        op(x)
        writer = next(iter(ops.launched[0][1].values())).kernels[1]
        for core_x, core_y, begin, count in op.plan['mapping']:
            self.assertEqual(writer.runtime_args[core_x][core_y][3:], [begin, count])

    def test_the_circular_buffers_are_the_plans(self):
        ops, op, (x, w1, w3) = make_op(pairs=3)
        op(x)
        program = next(iter(ops.launched[0][1].values()))
        sizes = dict((cb.format_descriptors[0].buffer_index, cb.total_size) for cb in program.cbs)
        buffers = op.plan['buffers']
        self.assertEqual(sizes, dict((item['index'], item['bytes']) for item in buffers.values()))
        self.assertEqual(len(program.semaphores), 2)

    def test_the_compute_kernel_is_the_patched_native_source_with_the_served_math_mode(self):
        for approx in (True, False):
            ops, op, (x, w1, w3) = make_op(approx=approx)
            op(x)
            compute = next(iter(ops.launched[0][1].values())).kernels[2]
            self.assertEqual(compute.source_type, 'code')
            self.assertEqual(compute.kernel_source, fused.fused_compute(SOURCE, 3, 2))
            self.assertEqual(compute.compile_time_args, fused.compute_arguments(3, 2, 20))
            self.assertEqual(compute.config.math_approx_mode, approx)
            self.assertEqual((compute.config.math_fidelity, compute.config.fp32_dest_acc_en), ('lofi', True))
            self.assertEqual(compute.config.unpack_to_dest_mode[5], 'fp32')
            self.assertEqual(len(compute.config.unpack_to_dest_mode), 64)
            self.assertIn(('PACKER_L1_ACC', '1'), compute.defines)
            self.assertEqual(op.manifest['math_approx_mode'], approx)

    def test_the_output_is_a_fresh_l1_tile_tensor_of_the_products_shape(self):
        ops, op, (x, w1, w3) = make_op()
        output = op(x)
        tensor, device, memory, layout = ops.outputs[0]
        self.assertEqual((tensor.shape, tensor.dtype, device, memory, layout), ((1, 1, 64, 4352), 'bf16', 'mesh', 'L1', 'tile'))
        self.assertIs(output, tensor)

    def test_a_single_chip_mesh_gets_one_program(self):
        ops, op, (x, w1, w3) = make_op(chips=1)
        op(x)
        self.assertEqual(len(ops.launched[0][1]), 1)

    def test_inputs_are_checked(self):
        ops, op, (x, w1, w3) = make_op()
        with self.assertRaisesRegex(ValueError, 'input must be'):
            op(Tensor((1, 1, 32, 5120), 'bf16', 100))
        with self.assertRaisesRegex(ValueError, 'input must be'):
            op(Tensor((1, 1, 64, 5120), 'f32', 100))
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
        with self.assertRaises(ValueError):
            make_op(pairs=6)

    def test_the_native_source_is_read_once_per_shape_and_hashed(self):
        reads = []
        ops = FakeOps()
        fused._COMPUTE_CACHE.clear()
        w = Tensor((1, 1, 5120, 17408), 'bf4', 200)
        with unittest_patch(fused, 'native_source', lambda root: reads.append(root) or SOURCE):
            first = fused.FusedGateUp(ops, 'mesh', w, w, math_approx_mode=True)
            second = fused.FusedGateUp(ops, 'mesh', w, w, math_approx_mode=True)
            fused.FusedGateUp(ops, 'mesh', w, w, pairs_per_worker=4, math_approx_mode=True)
        self.assertEqual(len(reads), 2)
        self.assertEqual(first.manifest['native_compute_sha256'], hashlib.sha256(SOURCE.encode()).hexdigest())
        self.assertEqual(first.manifest['fused_compute_sha256'], second.manifest['fused_compute_sha256'])
        self.assertEqual(first.manifest['kernels'], fused.kernel_sources())
        self.assertEqual(set(first.manifest['kernels']), set(fused.KERNELS))

    def test_it_does_not_free_its_input_and_counts_its_launches(self):
        ops, op, (x, w1, w3) = make_op()
        op(x)
        op(x)
        self.assertEqual(op.calls, 2)
        self.assertEqual(len(ops.launched), 2)


class SourceTests(unittest.TestCase):
    def test_the_kernels_hold_no_pair_packed_assumption(self):
        weights = (HERE / 'tp4_mlp_fused_weights.cpp').read_text(encoding='utf-8')
        self.assertNotIn('544', weights, 'the TP2 kernel\'s packed row stride must not survive')
        self.assertIn('constexpr auto up_args', weights)

    def test_the_twin_does_not_edit_the_pinned_original(self):
        original = (HERE / 'fused_1d.py').read_bytes()
        self.assertIn(b'class FusedProjection', original)
        self.assertNotIn(b'tp4_mlp', original)


if __name__ == '__main__':
    unittest.main()
