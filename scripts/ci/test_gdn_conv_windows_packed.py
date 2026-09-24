"""Verify-trace T2 cut #1: the packed GDN conv windows (gdn_conv_windows_packed.py) and its wiring
into the batched packed block (gdn_user_batch_conv.run_user_batched_projected).

What the CPU can hold the op to: the work split covers every (user, page) exactly once; the
trimmed copy map IS the served kernel's token loop (gdn_conv_windows.cpp:12-22) on every byte of
every page, padding included; the common-arg layout the kernel indexes is the one the host writes;
the driver makes one generic_op per call with the served output signature, hits its descriptor
cache from the second call, frees its outputs on any failure and refuses what it does not serve
before any kernel runs. And the wiring: flag off the block runs exactly the calls it ran before
(and never imports the op), flag on one packed launch precedes the per-user conv gates, an
Unsupported input takes the served path and is counted, the audit builds the served windows beside
the packed ones outside `owned`. Whether the kernel moves the bytes that way on silicon is card M's
(optimisation/ttnn-op/verify_t2).

DescriptorTTNN below is a recording fake of the generic_op descriptor API; test_packed_ordered_cache
and the card-M harness tests reuse it.
"""

from itertools import count
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import gdn_conv_windows_packed as vtw
import verify_trace_t2 as t2

HERE = Path(__file__).parent
ON = {'QWEN_FAST_VERIFY_T2': '1'}


# ---------------------------------------------------------------------------------------------
# A recording fake of the descriptor API (generic_op and everything that builds its program).
# ---------------------------------------------------------------------------------------------

class Coord:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return isinstance(other, Coord) and (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return 'Coord(%d, %d)' % (self.x, self.y)


class Record(SimpleNamespace):
    """A descriptor: its constructor keywords as attributes, comparable by value."""


class RuntimeArgs(dict):
    def __getitem__(self, key):
        if key not in self:
            dict.__setitem__(self, key, {})
        return dict.__getitem__(self, key)

    def flattened(self):
        return sorted((x, y, tuple(args)) for x, column in self.items() for y, args in column.items())


class KernelDescriptor(Record):
    class SourceType:
        SOURCE_CODE = 'source-code'

    def __init__(self, **kwargs):
        super().__init__(runtime_args=None, common_runtime_args=None, defines=None, source_type=None)
        for name, value in kwargs.items():
            setattr(self, name, value)


class FakeDevice:
    def worker_core_from_logical_core(self, core):
        return Coord(core.x + 1, core.y + 2)


class FakeShard:
    def __init__(self, address, kind):
        self.address, self.kind = address, kind

    def buffer_address(self):
        return self.address

    def device(self):
        return FakeDevice()


class FakeTensor:
    def __init__(self, name, shape, addresses, memory='dram', dtype='bf16', layout='tile', padded=None):
        self.name, self.shape, self.dtype, self.layout, self._memory = name, tuple(shape), dtype, layout, memory
        self.padded_shape = tuple(padded or shape)
        self.shards = [FakeShard(address, memory) for address in addresses]

    def memory_config(self):
        return self._memory

    def __repr__(self):
        return 'FakeTensor(%s)' % self.name


class DescriptorTTNN:
    bfloat16, bfloat8_b, int32, uint32 = 'bf16', 'bf8', 'int32', 'uint32'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT = 'tile', 'row_major'
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = 'dram', 'l1'
    DataMovementProcessor = SimpleNamespace(RISCV_0='riscv0', RISCV_1='riscv1')
    NOC = SimpleNamespace(RISCV_0_default='noc0', RISCV_1_default='noc1')
    KernelDescriptor = KernelDescriptor
    RuntimeArgs = RuntimeArgs

    def __init__(self, chips=2):
        self.chips = chips
        self.addresses = count(0x100000, 0x1000)
        self.calls, self.deallocated, self.made = [], [], []
        self.fail = None

    def tensor(self, name, shape, memory='dram', dtype='bf16', layout='tile', padded=None):
        return FakeTensor(name, shape, [next(self.addresses) for chip in range(self.chips)], memory, dtype, layout,
                          padded)

    # --- allocation --------------------------------------------------------------------------
    def empty(self, shape, device=None, dtype=None, layout=None, memory_config=None):
        tensor = self.tensor('out%d' % len(self.made), shape, memory_config, dtype, layout)
        self.made.append(tensor)
        return tensor

    def deallocate(self, tensor):
        self.deallocated.append(tensor)

    def get_device_tensors(self, tensor):
        return tensor.shards

    def slice(self, tensor, start, stop, memory_config=None):
        piece = self.tensor('%s[%d:%d]' % (tensor.name, start[1], stop[1]),
                            tuple(b - a for a, b in zip(start, stop)), memory_config, tensor.dtype, tensor.layout)
        self.calls.append(('slice', tensor.name, tuple(start), tuple(stop)))
        return piece

    def to_memory_config(self, tensor, memory_config):
        moved = self.tensor(tensor.name + '.' + memory_config, tensor.shape, memory_config, tensor.dtype, tensor.layout)
        self.calls.append(('interleave', tensor.name))
        return moved

    # --- descriptors -------------------------------------------------------------------------
    def CoreCoord(self, x, y):
        return Coord(x, y)

    def CoreRange(self, start, end):
        return ('range', (start.x, start.y), (end.x, end.y))

    def CoreRangeSet(self, ranges):
        return ('set', tuple(ranges))

    def Tile(self, shape):
        return ('tile', tuple(shape))

    def TileDescriptor(self, tile):
        return ('tile_descriptor', tile)

    def CBFormatDescriptor(self, **kwargs):
        return Record(**kwargs)

    def CBDescriptor(self, **kwargs):
        return Record(**kwargs)

    def SemaphoreDescriptor(self, **kwargs):
        return Record(**kwargs)

    def DataMovementConfigDescriptor(self, **kwargs):
        return Record(kind='data_movement', **kwargs)

    def ComputeConfigDescriptor(self, **kwargs):
        return Record(kind='compute', **kwargs)

    def ProgramDescriptor(self, **kwargs):
        return Record(**kwargs)

    def MeshProgramDescriptor(self):
        return {}

    def MeshCoordinate(self, row, col):
        return (row, col)

    def MeshCoordinateRange(self, start, end):
        return (start, end)

    def TensorAccessorArgs(self, shard):
        return SimpleNamespace(get_compile_time_args=lambda: [7 if shard.kind == 'dram' else 9, 1])

    def generic_op(self, tensors, program):
        if self.fail is not None:
            raise self.fail
        snapshot = {}
        for coordinate, descriptor in program.items():
            snapshot[coordinate] = dict(
                kernels=[dict(source=kernel.kernel_source, source_type=kernel.source_type, cores=kernel.core_ranges,
                              compile=list(kernel.compile_time_args), defines=kernel.defines,
                              runtime=kernel.runtime_args.flattened() if kernel.runtime_args is not None else None,
                              common=list(kernel.common_runtime_args) if kernel.common_runtime_args is not None else None,
                              config=kernel.config)
                         for kernel in descriptor.kernels],
                cbs=descriptor.cbs, semaphores=getattr(descriptor, 'semaphores', None))
        self.calls.append(('generic_op', list(tensors), snapshot))
        return tensors[-1]

    def programs(self):
        return [entry for entry in self.calls if entry[0] == 'generic_op']


def mesh(chips=2, grid=(11, 10)):
    return SimpleNamespace(shape=(1, chips), compute_with_storage_grid_size=lambda: SimpleNamespace(x=grid[0], y=grid[1]))


def users(ttnn, count_=4, rows=16, width=8240, piece_memory='l1', history_memory='dram'):
    return [(ttnn.tensor('piece%d' % user, (1, rows, width), piece_memory),
             [ttnn.tensor('hist%d.%d' % (user, h), (1, 1, 5120), history_memory) for h in range(4)])
            for user in range(count_)]


# ---------------------------------------------------------------------------------------------
# Pure planners.
# ---------------------------------------------------------------------------------------------

class PlanTests(unittest.TestCase):
    def test_four_users_on_a_p150a_grid(self):
        layout = vtw.plan(4, 110)
        self.assertEqual(layout['tasks'], 640)
        self.assertEqual([count_ for start, count_ in layout['ranges']], [6] * 90 + [5] * 20)

    def test_every_user_page_exactly_once_on_any_grid(self):
        for users_ in range(1, 5):
            for cores in (48, 64, 88, 110, 130):
                with self.subTest(users=users_, cores=cores):
                    layout = vtw.plan(users_, cores)
                    seen = [vtw.task(t) for start, count_ in layout['ranges'] for t in range(start, start + count_)]
                    self.assertEqual(sorted(seen), [(u, p) for u in range(users_) for p in range(160)])
                    counts = [count_ for start, count_ in layout['ranges']]
                    self.assertLessEqual(max(counts) - min(counts), 1)
        for bad in ((0, 110), (4, 0), (4.0, 110)):
            with self.assertRaises(ValueError):
                vtw.plan(*bad)

    def test_the_grid_is_row_major(self):
        self.assertEqual(vtw.core_coordinates(11, 10, 13)[-3:], [(10, 0), (0, 1), (1, 1)])
        with self.assertRaises(ValueError):
            vtw.core_coordinates(8, 6, 49)


def served_literal(slot, rows=16):
    """gdn_conv_windows.cpp:12-22 read aloud: per token, the history index, the tile it copies
    from (0 the piece, h + 1 history h), the source row and the destination row."""
    out = []
    for token in range(rows):
        history = token + slot
        source_row = 0 if history < 4 else history - 4
        out.append((token, history, history + 1 if history < 4 else 0, source_row, token))
    return out


def random_pages(seed, pages=160):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-32768, 32767, (pages, 1024), generator=generator, dtype=torch.int32).to(torch.int16)


class CopyMapTests(unittest.TestCase):
    def test_the_row_map_is_the_served_loop(self):
        for slot in range(4):
            with self.subTest(slot=slot):
                self.assertEqual(vtw.served_row_map(slot), served_literal(slot))
                expanded = []
                for kind, source, row, length in vtw.window_copies(slot):
                    for step in range(length // 32):
                        if kind == 'hist':
                            expanded.append((row + step, source, source + 1, 0, row + step))
                        else:
                            expanded.append((row + step, source + step + 4, 0, source + step, row + step))
                self.assertEqual(expanded, served_literal(slot))

    def test_every_offset_and_length_is_a_32_byte_multiple_inside_faces_0_and_1(self):
        for slot in range(4):
            for kind, source, row, length in vtw.window_copies(slot):
                self.assertEqual(length % 32, 0)
                self.assertLessEqual(row * 32 + length, 512)
            self.assertEqual(sum(length for kind, source, row, length in vtw.window_copies(slot)), 512)
        with self.assertRaises(ValueError):
            vtw.window_copies(4)

    def test_the_trimmed_map_equals_the_served_loop_on_every_byte_padding_included(self):
        piece = random_pages(1)
        histories = [random_pages(10 + h) for h in range(4)]
        served = vtw.reference_windows(piece, histories)
        trimmed = vtw.apply_copies(piece, histories)
        for slot in range(4):
            with self.subTest(slot=slot):
                self.assertTrue(torch.equal(served[slot], trimmed[slot]))
                self.assertTrue(bool((served[slot][:, 512:] == 0).all()), 'faces 2-3 are zero')

    def test_window_row_r_of_slot_s_is_timeline_row_r_plus_s(self):
        """Rows as values: history h holds 100 + h in row 0, piece row k holds k + 1 (and 777 in
        every row the windows must never read), so each window row names its source."""
        piece = torch.full((2, 1024), 777, dtype=torch.int16)
        for row in range(16):
            for face in range(2):
                piece[:, face * 256 + row * 16:face * 256 + row * 16 + 16] = row + 1
        histories = []
        for h in range(4):
            page = torch.full((2, 1024), 777, dtype=torch.int16)
            page[:, 0:16] = 100 + h
            page[:, 256:272] = 100 + h
            histories.append(page)
        timeline = [100, 101, 102, 103] + [row + 1 for row in range(16)]
        for slot, window in enumerate(vtw.reference_windows(piece, histories)):
            for row in range(16):
                for face in range(2):
                    values = window[0, face * 256 + row * 16:face * 256 + row * 16 + 16]
                    self.assertTrue(bool((values == timeline[row + slot]).all()), (slot, row, face))
            self.assertFalse(bool((window == 777).any()), 'nothing past row 15 of the piece or row 0 of a history')

    def test_minus_zero_denormals_and_nan_payloads_travel_as_bits(self):
        # -0 (0x8000), a denormal (0x0001), -denormal (0x8001), a NaN with a payload (0x7FC1),
        # a denormal (0x007F), -denormal (0x807F), +0, 5
        piece = random_pages(3, pages=4)
        piece[:, :8] = torch.tensor([-32768, 1, -32767, 32705, 127, -32641, 0, 5], dtype=torch.int16)
        histories = [random_pages(4 + h, pages=4) for h in range(4)]
        histories[0][:, 0:4] = torch.tensor([-32768, 1, 32705, -32641], dtype=torch.int16)
        served = vtw.reference_windows(piece, histories)
        trimmed = vtw.apply_copies(piece, histories)
        for slot in range(4):
            self.assertTrue(torch.equal(served[slot], trimmed[slot]))
        self.assertEqual(served[0][0, 0:4].tolist(), [-32768, 1, 32705, -32641])


class ArgsTests(unittest.TestCase):
    def test_the_common_args_are_user_major_pieces_histories_outputs(self):
        pieces = [10, 20]
        histories = [[11, 12, 13, 14], [21, 22, 23, 24]]
        outputs = [[31, 32, 33, 34], [41, 42, 43, 44]]
        args = vtw.common_args(pieces, histories, outputs)
        self.assertEqual(args, [10, 20, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44])
        for user in range(2):
            self.assertEqual(args[vtw.common_index(2, 'piece', user)], pieces[user])
            for index in range(4):
                self.assertEqual(args[vtw.common_index(2, 'history', user, index)], histories[user][index])
                self.assertEqual(args[vtw.common_index(2, 'output', user, index)], outputs[user][index])
        self.assertEqual(len(vtw.common_args([1] * 4, [[1] * 4] * 4, [[1] * 4] * 4)), 36)
        for bad in (([], [], []), ([1], [[1] * 3], [[1] * 4]), ([1], [[1] * 4], [[1] * 4, [1] * 4])):
            with self.assertRaises(ValueError):
                vtw.common_args(*bad)

    def test_settings_the_matrix_and_the_cb_budget(self):
        self.assertEqual(vtw.resolve_settings(), vtw.DEFAULTS)
        self.assertEqual(vtw.resolve_settings(dict(port=True, copy_noc=True)),
                         dict(port=True, hist_row=False, copy_noc=False, nbuf=1))
        matrix = vtw.settings_matrix()
        self.assertEqual(len(matrix), 9)
        self.assertEqual(len({vtw.settings_name(s) for s in matrix}), 9)
        self.assertIn(vtw.resolve_settings(), [vtw.resolve_settings(s) for s in matrix])
        self.assertEqual(vtw.cb_bytes(dict(port=True)), 12288, 'the port keeps the served 6 x 2048')
        self.assertEqual(max(vtw.cb_bytes(s) for s in matrix), 36864)
        self.assertEqual(vtw.cb_plan(dict(nbuf=1)), {0: 10, 1: 4})
        for bad in (dict(nbuf=3), dict(port=1), dict(tiles=2)):
            with self.assertRaises(ValueError):
                vtw.resolve_settings(bad)

    def test_the_defines_carry_the_source_sha_the_role_and_the_controls(self):
        sha = vtw.source_sha()
        self.assertRegex(sha, '^[0-9a-f]{8}$')
        reader = dict(vtw.kernel_defines(sha, 'VTW_ROLE_READER', None))
        # DEFAULTS (hist_row, NOC copies since card B run vt2-20260924T012053) reach both roles
        self.assertEqual(reader, {'VTW_SRC_SHA': '0x' + sha, 'VTW_ROLE_READER': '1', 'VTW_HIST_ROW': '1',
                                  'VTW_COPY_NOC': '1'})
        writer = dict(vtw.kernel_defines(sha, 'VTW_ROLE_WRITER', dict(copy_noc=True, hist_row=False), 'pad'))
        self.assertEqual(writer, {'VTW_SRC_SHA': '0x' + sha, 'VTW_ROLE_WRITER': '1', 'VTW_COPY_NOC': '1',
                                  'VTW_NEG_PAD': '1'})
        self.assertEqual(dict(vtw.kernel_defines(sha, 'VTW_PORT', dict(port=True))), {'VTW_SRC_SHA': '0x' + sha,
                                                                                    'VTW_PORT': '1'})
        with self.assertRaises(ValueError):
            vtw.kernel_defines(sha, 'VTW_PORT', None)
        with self.assertRaises(ValueError):
            vtw.kernel_defines(sha, 'VTW_ROLE_READER', None, 'bogus')
        self.assertEqual(vtw.roles(None), ('VTW_ROLE_READER', 'VTW_ROLE_WRITER'))
        self.assertEqual(vtw.roles(dict(port=True)), ('VTW_PORT',))


# ---------------------------------------------------------------------------------------------
# The kernel text.
# ---------------------------------------------------------------------------------------------

class KernelTextTests(unittest.TestCase):
    TEXT = (HERE / 'gdn_conv_windows_packed.cpp').read_text(encoding='utf-8')

    def test_the_three_roles_and_every_control_define_exist(self):
        for role in vtw.ROLES:
            self.assertIn('defined(%s)' % role, self.TEXT)
        for define in list(vtw.NEGATIVE_CONTROLS.values()) + ['VTW_HIST_ROW', 'VTW_COPY_NOC']:
            self.assertIn('#ifdef %s' % define, self.TEXT)

    def test_faces_2_3_are_zeroed_once_only_under_rows_16(self):
        writer = self.TEXT[self.TEXT.index('#elif defined(VTW_ROLE_WRITER)'):self.TEXT.index('#elif defined(VTW_PORT)')]
        self.assertIn('static_assert(ROWS == 16', writer)
        self.assertLess(writer.index('static_assert(ROWS == 16'), writer.index('fill_words(scratch_base'))
        self.assertLess(writer.index('fill_words(scratch_base'), writer.index('for (uint32_t t = task_start'))
        port = self.TEXT[self.TEXT.index('#elif defined(VTW_PORT)'):]
        self.assertIn('fill_words(scratch, page, 0)', port, 'the port zeroes the whole scratch per window')
        self.assertLess(port.index('noc_async_write_tile'), port.index('noc_async_write_barrier'))

    def test_the_kernel_indexes_the_common_args_the_host_writes(self):
        self.assertIn('constexpr uint32_t H_BASE = USERS;', self.TEXT)
        self.assertIn('constexpr uint32_t O_BASE = USERS + history * USERS;', self.TEXT)
        self.assertEqual(len(re.findall(r'get_common_arg_val<uint32_t>\(H_BASE \+ history \* user \+ h\)', self.TEXT)), 2)
        self.assertEqual(len(re.findall(r'get_common_arg_val<uint32_t>\(O_BASE \+ slots \* user \+ (s|slot)\)', self.TEXT)), 2)
        self.assertEqual(len(re.findall(r'get_common_arg_val<uint32_t>\(piece_user<USERS>\(user\)\)', self.TEXT)), 2)
        # the compile-time args the host sends: USERS, PAGES, ROWS, NBUF, then the accessors at 5
        self.assertIn('TensorAccessorArgs<5>()', self.TEXT)
        self.assertEqual(vtw.compile_args(4, None, [7, 1, 7, 1, 7, 1])[:5], [4, 160, 16, vtw.DEFAULTS['nbuf'], 640])

    def test_the_writer_invalidates_its_data_cache_before_reading_cb_in_by_words(self):
        """Blackhole's cb_wait_front does not invalidate the RISC's L1 data cache; the word copies
        must not rely on a flush or barrier elsewhere happening to invalidate it."""
        writer = self.TEXT[self.TEXT.index('#elif defined(VTW_ROLE_WRITER)'):self.TEXT.index('#elif defined(VTW_PORT)')]
        wait = writer.index('cb_wait_front(cb_in, tiles_in);')
        guard = writer.index('#ifndef VTW_COPY_NOC', wait)
        invalidate = writer.index('invalidate_l1_cache();', guard)
        self.assertLess(invalidate, writer.index('get_read_ptr(cb_in)'))
        self.assertLess(invalidate, writer.index('build_window(staged'))
        self.assertEqual(writer[guard:invalidate].count('#endif'), 0)

    def test_the_history_row_read_takes_only_the_two_spans_the_windows_use(self):
        reader = self.TEXT[self.TEXT.index('#if defined(VTW_ROLE_READER)'):self.TEXT.index('#elif defined(VTW_ROLE_WRITER)')]
        self.assertIn('noc_async_read(source.get_noc_addr(p, 0), destination, history_span)', reader)
        self.assertIn('noc_async_read(source.get_noc_addr(p, face), destination + face, history_span)', reader)
        self.assertIn('constexpr uint32_t history_span = 64;', self.TEXT)


# ---------------------------------------------------------------------------------------------
# The driver, on the recording fake.
# ---------------------------------------------------------------------------------------------

class DriverTests(unittest.TestCase):
    def setUp(self):
        vtw.clear_cache()
        self.addCleanup(vtw.clear_cache)
        self.ttnn = DescriptorTTNN()
        self.mesh = mesh()

    def build(self, group, **options):
        return vtw.build_windows_packed(self.mesh, group, operations=self.ttnn, **options)

    def test_one_launch_sixteen_outputs_of_the_served_signature(self):
        group = users(self.ttnn)
        outputs = self.build(group)
        self.assertEqual(len(self.ttnn.programs()), 1)
        self.assertEqual([len(own) for own in outputs], [4] * 4)
        for own in outputs:
            for window in own:
                self.assertEqual((window.shape, window.dtype, window.layout, window.memory_config()),
                                 ((1, 16, 5120), 'bf16', 'tile', 'dram'))
        name, tensors, program = self.ttnn.programs()[0]
        self.assertEqual(len(tensors), 4 + 16 + 16)
        self.assertEqual(sorted(program), [((0, 0), (0, 0)), ((0, 1), (0, 1))])
        chip0 = program[((0, 0), (0, 0))]
        self.assertEqual([kernel['config'].processor for kernel in chip0['kernels']], ['riscv1', 'riscv0'])
        self.assertEqual([dict(kernel['defines'])['VTW_ROLE_READER' if index == 0 else 'VTW_ROLE_WRITER']
                          for index, kernel in enumerate(chip0['kernels'])], ['1', '1'])
        self.assertEqual(chip0['kernels'][0]['cores'], ('set', (('range', (0, 0), (10, 9)),)))
        self.assertEqual(chip0['kernels'][0]['compile'][:5], [4, 160, 16, vtw.DEFAULTS['nbuf'], 640])
        self.assertEqual(chip0['kernels'][0]['compile'][5:], [9, 1, 7, 1, 7, 1], 'L1 pieces, DRAM histories and outputs')
        runtime = chip0['kernels'][1]['runtime']
        self.assertEqual(len(runtime), 110)
        self.assertEqual(sorted(args for x, y, args in runtime)[:2], [(0, 6), (6, 6)])
        self.assertEqual(sum(args[1] for x, y, args in runtime), 640)
        self.assertEqual([(cb.format_descriptors[0].buffer_index, cb.total_size) for cb in chip0['cbs']],
                         [(0, 20480), (1, 2048 * vtw.SLOTS * vtw.DEFAULTS['nbuf'])])

    def test_each_chip_gets_its_own_addresses_in_the_kernel_layout(self):
        group = users(self.ttnn)
        outputs = self.build(group)
        program = self.ttnn.programs()[0][2]
        for chip in range(2):
            common = program[((0, chip), (0, chip))]['kernels'][0]['common']
            self.assertEqual(len(common), 36)
            for user, (piece, history) in enumerate(group):
                self.assertEqual(common[vtw.common_index(4, 'piece', user)], piece.shards[chip].address)
                for index in range(4):
                    self.assertEqual(common[vtw.common_index(4, 'history', user, index)], history[index].shards[chip].address)
                    self.assertEqual(common[vtw.common_index(4, 'output', user, index)],
                                     outputs[user][index].shards[chip].address)

    def test_the_second_call_hits_the_descriptor_cache_and_only_the_addresses_change(self):
        first = self.build(users(self.ttnn))
        second = self.build(users(self.ttnn))
        self.assertEqual(vtw.cache_size(), 1)
        one, two = self.ttnn.programs()[0][2], self.ttnn.programs()[1][2]
        for coordinate in one:
            for a, b in zip(one[coordinate]['kernels'], two[coordinate]['kernels']):
                self.assertEqual({k: v for k, v in a.items() if k != 'common'}, {k: v for k, v in b.items() if k != 'common'})
                self.assertNotEqual(a['common'], b['common'])
        self.assertIsNot(first[0][0], second[0][0])
        self.build(users(self.ttnn), settings=dict(port=True))
        self.assertEqual(vtw.cache_size(), 2)

    def test_the_port_runs_alone_on_riscv0_with_the_served_cb(self):
        self.build(users(self.ttnn), settings=dict(port=True))
        chip0 = self.ttnn.programs()[0][2][((0, 0), (0, 0))]
        self.assertEqual(len(chip0['kernels']), 1)
        self.assertEqual(chip0['kernels'][0]['config'].processor, 'riscv0')
        self.assertIn(('VTW_PORT', '1'), chip0['kernels'][0]['defines'])
        self.assertEqual([cb.total_size for cb in chip0['cbs']], [12288])

    def test_a_shared_input_is_listed_once(self):
        group = users(self.ttnn, 2)
        group[1] = (group[1][0], [group[0][1][0]] + group[1][1][1:])
        self.build(group)
        tensors = self.ttnn.programs()[0][1]
        self.assertEqual(len(tensors), len({id(value) for value in tensors}))
        self.assertEqual(len(tensors), 2 + 7 + 8)

    def test_a_failed_launch_frees_every_output(self):
        self.ttnn.fail = RuntimeError('device')
        with self.assertRaisesRegex(RuntimeError, 'device'):
            self.build(users(self.ttnn))
        self.assertEqual(len(self.ttnn.deallocated), 16)
        self.assertEqual({id(value) for value in self.ttnn.deallocated}, {id(value) for value in self.ttnn.made})

    def test_aliases_are_refused_with_the_served_text_and_free_the_outputs(self):
        group = users(self.ttnn, 2)
        piece, history = group[0]
        group[0] = (piece, [history[0], history[0], history[2], history[3]])
        with self.assertRaisesRegex(ValueError, 'Immutable input and mutable windows must not alias'):
            self.build(group)
        self.assertEqual(len(self.ttnn.deallocated), 8)
        self.assertEqual(self.ttnn.programs(), [])

    def test_the_mesh_of_one_chip_is_served(self):
        self.ttnn = DescriptorTTNN(chips=1)
        self.mesh = mesh(1)
        self.build(users(self.ttnn))
        self.assertEqual(sorted(self.ttnn.programs()[0][2]), [((0, 0), (0, 0))])

    def test_everything_it_does_not_serve_is_refused_before_any_device_work(self):
        cases = {
            'five users': users(self.ttnn, 5),
            'eight rows': users(self.ttnn, rows=8),
            'thirty-two rows': users(self.ttnn, rows=32),
            'a bad width': users(self.ttnn, width=8000),
            'three histories': [(p, h[:3]) for p, h in users(self.ttnn)],
            'a row-major piece': [(self.ttnn.tensor('p', (1, 16, 8240), 'l1', layout='row_major'), h)
                                  for p, h in users(self.ttnn, 1)],
            'a sharded history': [(p, [self.ttnn.tensor('h', (1, 1, 5120), 'sharded')] + h[1:])
                                  for p, h in users(self.ttnn, 1)],
            'mixed history placement': [(p, [self.ttnn.tensor('h', (1, 1, 5120), 'l1')] + h[1:])
                                        for p, h in users(self.ttnn, 1)],
        }
        for name, group in cases.items():
            with self.subTest(case=name), self.assertRaises(vtw.Unsupported):
                self.build(group)
        with self.assertRaises(vtw.Unsupported):
            vtw.build_windows_packed(SimpleNamespace(shape=(2, 2), compute_with_storage_grid_size=None),
                                     users(self.ttnn), operations=self.ttnn)
        self.assertEqual((self.ttnn.made, self.ttnn.programs(), vtw.cache_size()), ([], [], 0))

    def test_the_eight_width_is_served_and_keys_its_own_program(self):
        self.build(users(self.ttnn, width=8256))
        self.build(users(self.ttnn, width=8240))
        self.assertEqual(vtw.cache_size(), 2)

    def test_the_negative_controls_are_defines_and_the_user_control_needs_two_users(self):
        for negative, define in vtw.NEGATIVE_CONTROLS.items():
            vtw.clear_cache()
            self.build(users(self.ttnn), negative=negative)
            kernels = self.ttnn.programs()[-1][2][((0, 0), (0, 0))]['kernels']
            self.assertTrue(all((define, '1') in kernel['defines'] for kernel in kernels), negative)
        with self.assertRaises(ValueError):
            self.build(users(self.ttnn, 1), negative='user')


# ---------------------------------------------------------------------------------------------
# The wiring: gdn_user_batch_conv.run_user_batched_projected.
# ---------------------------------------------------------------------------------------------

class WiringTests(unittest.TestCase):
    def setUp(self):
        from test_gdn_user_batch_conv import fake_operations, user_group

        self.calls = []
        self.operations = fake_operations(self.calls)
        self.groups = [user_group(index, self.operations) for index in range(4)]
        self.taps = [self.operations.make('tap%d' % tap, (1, 1, 5120)) for tap in range(4)]
        self.extras = (self.operations.make('dt', (1, 1, 24)), self.operations.make('nega', (1, 1, 24)),
                       self.operations.make('norm_w', (1, 1, 128)))
        t2.take()
        t2._LOGGED.clear()
        self.addCleanup(t2.take)

    def served(self, mesh_, projected, history):
        self.calls.append(('windows', projected.name))
        return [self.operations.make('window:%s.%d' % (projected.name, slot), (1, projected.shape[1], 5120))
                for slot in range(4)]

    def packed(self, mesh_, group, operations=None):
        self.calls.append(('packed', tuple(piece.name for piece, history in group)))
        return [[self.operations.make('packed:%s.%d' % (piece.name, slot), (1, piece.shape[1], 5120))
                 for slot in range(4)] for piece, history in group]

    def run_block(self, environ, packed=None, execute=None):
        import gdn_user_batch_conv

        def batched(mesh_, inputs, kernels, ops, output_memory=None):
            self.calls.append(('batched', len(inputs)))
            return [(self.operations.make('out%d' % index, (1, 16, 3072)),
                     self.operations.make('prefix%d' % index, (16, 24, 128, 128))) for index in range(len(inputs))]

        with patch.dict('os.environ', environ, clear=True), \
                patch('gdn_conv_windows.build_windows', side_effect=self.served), \
                patch('gdn_conv_windows_packed.build_windows_packed', side_effect=packed or self.packed), \
                patch('gdn_user_batch.execute', side_effect=execute or batched):
            return gdn_user_batch_conv.run_user_batched_projected('mesh', self.groups, self.taps, *self.extras,
                                                                  'kernels', self.operations)

    def order(self):
        return [entry[0] for entry in self.calls if entry[0] in ('windows', 'packed', 'conv_gates', 'slice', 'batched')]

    def test_flag_off_the_calls_are_todays_and_the_op_is_never_imported(self):
        sys.modules.pop('gdn_conv_windows_packed', None)
        import gdn_user_batch_conv

        def batched(mesh_, inputs, kernels, ops, output_memory=None):
            return [(self.operations.make('out%d' % index, (1, 16, 3072)),
                     self.operations.make('prefix%d' % index, (16, 24, 128, 128))) for index in range(len(inputs))]

        with patch.dict('os.environ', {}, clear=True), \
                patch('gdn_conv_windows.build_windows', side_effect=self.served), \
                patch('gdn_user_batch.execute', side_effect=batched):
            results = gdn_user_batch_conv.run_user_batched_projected('mesh', self.groups, self.taps, *self.extras,
                                                                     'kernels', self.operations)
        self.assertNotIn('gdn_conv_windows_packed', sys.modules)
        self.assertEqual(self.order(), ['windows', 'conv_gates', 'slice'] * 4)
        self.assertTrue(all('audit_windows' not in result for result in results))
        self.assertEqual(t2.take(), {})

    def test_flag_on_one_packed_launch_then_the_gates_and_z_per_user(self):
        results = self.run_block(ON)
        self.assertEqual(self.order(), ['packed'] + ['conv_gates', 'slice'] * 4 + ['batched'])
        self.assertEqual([entry for entry in self.calls if entry[0] == 'packed'],
                         [('packed', ('projected0', 'projected1', 'projected2', 'projected3'))])
        gates = [entry for entry in self.calls if entry[0] == 'conv_gates']
        for index, entry in enumerate(gates):
            self.assertEqual(entry[3], tuple('packed:projected%d.%d' % (index, slot) for slot in range(4)))
        for index, result in enumerate(results):
            self.assertEqual([w.name for w in result['packed_conv_states']],
                             ['packed:projected%d.%d' % (index, slot) for slot in range(4)])
            owned = [value.name for value in result['owned']]
            self.assertTrue(all(name in owned for name in ('packed:projected%d.%d' % (index, slot) for slot in range(4))))
            self.assertFalse(any('packed:projected%d' % other in name for other in range(4) if other != index
                                 for name in owned))
            self.assertNotIn('audit_windows', result)
        self.assertEqual(t2.take(), {'windows': 1})

    def test_the_packed_op_is_called_with_each_users_piece_and_history(self):
        seen = []

        def packed(mesh_, group, operations=None):
            seen.append(group)
            return self.packed(mesh_, group, operations)

        self.run_block(ON, packed=packed)
        self.assertEqual([(piece.name, [h.name for h in history]) for piece, history in seen[0]],
                         [('projected%d' % u, ['conv%d.%d' % (u, tap) for tap in range(4)]) for u in range(4)])

    def test_unsupported_takes_the_served_path_and_is_counted_once_and_logged_once(self):
        import gdn_conv_windows_packed

        def refuse(mesh_, group, operations=None):
            raise gdn_conv_windows_packed.Unsupported('rows 8 != 16')

        logged = []
        with patch.object(t2, 'log_line', side_effect=logged.append):
            self.run_block(ON, packed=refuse)
            self.run_block(ON, packed=refuse)
        self.assertEqual(self.order().count('windows'), 8)
        self.assertEqual(t2.take(), {'windows_fallback': 2})
        self.assertEqual(logged, ['[PINDIAG] verify t2 fell back site=windows reason=rows 8 != 16'])

    def test_skipping_the_cut_is_todays_path(self):
        self.run_block(dict(ON, QWEN_FAST_VERIFY_T2_SKIP='windows'))
        self.assertEqual(self.order(), ['windows', 'conv_gates', 'slice'] * 4 + ['batched'])

    def test_the_audit_builds_the_served_windows_beside_them_outside_owned(self):
        results = self.run_block(dict(ON, QWEN_FAST_VERIFY_T2_AUDIT='1'))
        self.assertEqual(self.order(), ['packed'] + ['windows', 'conv_gates', 'slice'] * 4 + ['batched'])
        for index, result in enumerate(results):
            self.assertEqual([w.name for w in result['audit_windows']],
                             ['window:projected%d.%d' % (index, slot) for slot in range(4)])
            self.assertFalse(any(value.name.startswith('window:') for value in result['owned']))
            self.assertEqual(t2.audit_windows_of(result), result['audit_windows'])
        self.assertEqual(len(t2.audit_windows_of(dict(segment_results=results))), 16)

    def test_a_failed_launch_frees_the_packed_and_the_audit_windows(self):
        def explode(*args, **kwargs):
            raise RuntimeError('device')

        with self.assertRaisesRegex(RuntimeError, 'device'):
            self.run_block(dict(ON, QWEN_FAST_VERIFY_T2_AUDIT='1'), execute=explode)
        freed = {entry[1] for entry in self.calls if entry[0] == 'free'}
        for user in range(4):
            for slot in range(4):
                self.assertIn('packed:projected%d.%d' % (user, slot), freed)
                self.assertIn('window:projected%d.%d' % (user, slot), freed)

    def test_gdn_records_retains_only_states_and_packed_windows(self):
        """The audit key is ignored by the retained block: its histories are the states and the
        packed windows, so the commit DMA reads the packed windows exactly as the served path's."""
        from gdn_records import block_histories

        results = self.run_block(dict(ON, QWEN_FAST_VERIFY_T2_AUDIT='1'))
        histories = block_histories(dict(results[0], segment_results=tuple(results)))
        self.assertEqual(len(histories), 20)
        self.assertFalse(any(value.name.startswith('window:') for value in histories))


if __name__ == '__main__':
    unittest.main()
