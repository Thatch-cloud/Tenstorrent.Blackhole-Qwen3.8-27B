"""CPU tests for C1e's op (mlp_c1e_pack: the per-layer rebuild of the served packed gate/up weight,
lever_n_m3native_patch section J). No device: the hardware half is the card-B harness,
optimisation/ttnn-op/c1e_gateup (run_card_b.sh), whose own CPU tests sit beside it; the model
graft's are in test_lever_n_m3native_patch.C1eGraftTests.

The page emulator runs mlp_c1e_pack.cpp's loop in Python over the planner's own worker layout,
so a wrong index or a gap in the split shows up as a wrong page. The double-rounding test is the
reason C1e exists: with the SAME fp32 gate and up, rounding them to bf16 before the product (C1,
C1c, C1d) changes a large share of the outputs, so no separate-matmul formulation can be exact.
"""

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import mlp_c1e_pack as pack

HERE = Path(__file__).parent
KERNEL_SOURCE = (HERE / pack.KERNEL).read_text(encoding='utf-8')
CHECK_SOURCE = (HERE / pack.CHECK_KERNEL).read_text(encoding='utf-8')


def interleave(w1, w3):
    """Per-chip tile-pair interleave: [K, N] x 2 -> [K, 2N], column tile 2t = w1 tile t, 2t + 1 = w3 tile t."""
    k, n = w1.shape
    return torch.stack((w1.reshape(k, n // 32, 32), w3.reshape(k, n // 32, 32)), dim=-2).reshape(k, 2 * n)


def emulate_kernel(layout, gate_pages, up_pages, columns, batch=pack.BATCH):
    """mlp_c1e_pack.cpp over every (core, RISC) worker: returns {packed page: bytes} and the write count."""
    packed, writes = {}, 0
    for processor, workers in layout.items():
        for _, (start, count) in workers:
            end = start + count
            for pair in range(start, end, batch):
                n = min(batch, end - pair)
                staged = []
                for i in range(n):
                    staged += [gate_pages[pair + i], up_pages[pair + i]]
                for i in range(n):
                    source = pair + i
                    target = (source // columns) * (2 * columns) + (source % columns) * 2
                    for page, data in ((target, staged[2 * i]), (target + 1, staged[2 * i + 1])):
                        if page in packed:
                            raise AssertionError('page %d written twice' % page)
                        packed[page] = data
                        writes += 1
    return packed, writes


def prepare_model(gate_up, ndev, gate_is_first=True):
    """prepare_for_fused_swiglu(gate_is_first=True) as mlp-sweep.py pins it: its documented inverse is
    packed.reshape(dim, ndev, tiles, 2, 32).permute(0, 3, 1, 2, 4).reshape_as(gate_up)."""
    if gate_is_first is not True:
        raise ValueError('the served packing is gate first')
    dim, width = gate_up.shape
    hidden = width // 2
    return gate_up.reshape(dim, 2, ndev, hidden // ndev // 32, 32).permute(0, 2, 3, 1, 4).reshape(dim, width)


class PlannerTests(unittest.TestCase):
    def test_the_real_tp2_geometry(self):
        for separate, packed in (((5120, 8704), (5120, 17408)), ((1, 1, 5120, 8704), (1, 1, 5120, 17408))):
            self.assertEqual(pack.geometry(separate, packed), (160, 272, 43520))
        self.assertEqual(pack.traffic_bytes((5120, 8704)), (50135040, 50135040))
        for bad in (((5120, 8704), (5120, 8704)), ((5120, 8700), None), ((2, 5120, 8704), None)):
            with self.assertRaises(ValueError):
                pack.geometry(*bad)

    def test_page_map_is_the_inverse_of_packed_weight_check(self):
        # packed_weight_check.cpp: paired_page = (page / columns) * columns * 2 + (page % columns) * 2 + offset
        self.assertIn('(page / columns) * columns * 2 + (page % columns) * 2 + offset', CHECK_SOURCE)
        self.assertIn('(source / columns) * (2 * columns) + (source % columns) * 2', KERNEL_SOURCE)
        for columns, rows in ((1, 3), (3, 2), (272, 4)):
            pairs = columns * rows
            seen = sorted(pack.packed_page(p, columns, g) for p in range(pairs) for g in (0, 1))
            self.assertEqual(seen, list(range(2 * pairs)))
            for p in range(pairs):
                for g in (0, 1):
                    self.assertEqual(pack.separate_page(pack.packed_page(p, columns, g), columns), (g, p))
        with self.assertRaises(ValueError):
            pack.packed_page(0, 4, 2)
        with self.assertRaises(ValueError):
            pack.packed_page(0, 4, True)

    def test_split_covers_every_pair_once_and_balanced(self):
        for pairs, workers in ((43520, 220), (43520, 260), (7, 220), (0, 4), (1000, 3)):
            ranges = pack.split(pairs, workers)
            self.assertEqual(len(ranges), workers)
            covered = [p for start, count in ranges for p in range(start, start + count)]
            self.assertEqual(covered, list(range(pairs)))
            counts = [count for _, count in ranges]
            self.assertLessEqual(max(counts) - min(counts), 1)
        with self.assertRaises(ValueError):
            pack.split(10, 0)

    def test_worker_layout_uses_both_riscs_on_every_core(self):
        layout = pack.worker_layout(11, 10, 43520)
        self.assertEqual(sorted(layout), [0, 1])
        for processor in (0, 1):
            self.assertEqual([core for core, _ in layout[processor]], pack.core_coordinates(11, 10))
        covered = sorted(p for workers in layout.values() for _, (start, count) in workers
                         for p in range(start, start + count))
        self.assertEqual(covered, list(range(43520)))

    def test_staging_fits_and_is_aligned(self):
        size = pack.cb_bytes()
        self.assertEqual(size % pack.CB_PAGE, 0)
        self.assertGreaterEqual(size, 2 * pack.BATCH * pack.PAGE + pack.ALIGN_SLACK)
        self.assertLess(pack.PROCESSORS * size, 96 * 1024)     # two RISCs' staging, well inside L1
        self.assertEqual(pack.PAGE % 64, 0)                     # every staged page stays 64-byte aligned
        self.assertEqual(pack.check_geometry(43520), (64, 43520))
        self.assertEqual(pack.check_geometry(3), (3, 3))

    def test_the_kernel_loop_writes_the_served_packing(self):
        """The emulated kernel over the real worker layout produces packed page (r, 2c + g) = (w1, w3)[g]
        page (r, c) - the element-level interleave the served prepare_for_fused_swiglu makes, per chip."""
        for columns, rows, grid in ((272, 3, (11, 10)), (5, 7, (3, 2)), (2, 1, (11, 10))):
            pairs = columns * rows
            gate = [('g', p) for p in range(pairs)]
            up = [('u', p) for p in range(pairs)]
            for batch in (1, 3, pack.BATCH):
                packed, writes = emulate_kernel(pack.worker_layout(*grid, pairs), gate, up, columns, batch)
                self.assertEqual(writes, 2 * pairs)
                for r in range(rows):
                    for c in range(columns):
                        self.assertEqual(packed[r * 2 * columns + 2 * c], ('g', r * columns + c))
                        self.assertEqual(packed[r * 2 * columns + 2 * c + 1], ('u', r * columns + c))

    def test_the_page_order_is_the_element_interleave(self):
        """TILE-layout page (kt, nt) of a [K, N] tensor holds rows kt*32.., columns nt*32..: so the
        page-level map above is exactly the element-level interleave."""
        k, n = 64, 96
        w1 = torch.arange(k * n, dtype=torch.float32).reshape(k, n)
        w3 = -w1 - 1
        packed = interleave(w1, w3)
        columns = n // 32
        for kt in range(k // 32):
            for nt in range(2 * columns):
                g, pair = pack.separate_page(kt * 2 * columns + nt, columns)
                source = (w1, w3)[g]
                self.assertTrue(torch.equal(packed[kt * 32:(kt + 1) * 32, nt * 32:(nt + 1) * 32],
                                            source[kt * 32:(kt + 1) * 32, (pair % columns) * 32:(pair % columns + 1) * 32]))

    def test_prepare_for_fused_swiglu_chip_shards_are_the_per_chip_interleave(self):
        dim, hidden, ndev = 64, 256, 2
        gate = torch.randn(dim, hidden)
        up = torch.randn(dim, hidden)
        gate_up = torch.cat([gate, up], dim=-1)
        packed = prepare_model(gate_up, ndev)
        restored = packed.reshape(dim, ndev, hidden // ndev // 32, 2, 32).permute(0, 3, 1, 2, 4).reshape_as(gate_up)
        self.assertTrue(torch.equal(restored, gate_up))    # mlp-sweep.py's own inverse check
        local = hidden // ndev
        for chip in range(ndev):
            shard = packed[:, chip * 2 * local:(chip + 1) * 2 * local]
            expected = interleave(gate[:, chip * local:(chip + 1) * local], up[:, chip * local:(chip + 1) * local])
            self.assertTrue(torch.equal(shard, expected))

    def test_source_sha_tracks_the_copy_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / pack.KERNEL).write_bytes(b'one')
            first = pack.source_sha(root)
            (root / pack.KERNEL).write_bytes(b'two')
            self.assertEqual(len(first), 8)
            self.assertNotEqual(pack.source_sha(root), first)


class KernelSourceTests(unittest.TestCase):
    def test_compile_and_common_args_match_the_host(self):
        self.assertIn('#error', KERNEL_SOURCE)       # refuses to build without C1E_SRC_SHA
        for index, name in enumerate(('page_bytes', 'columns', 'batch', 'cb_index')):
            self.assertIn('constexpr uint32_t %s = get_compile_time_arg_val(%d);' % (name, index), KERNEL_SOURCE)
        self.assertIn('TensorAccessorArgs<4>()', KERNEL_SOURCE)
        for index, name in enumerate(('gate', 'up', 'packed')):
            self.assertRegex(KERNEL_SOURCE, r'const auto %s = TensorAccessor\(%s_args, get_common_arg_val<uint32_t>\(%d\), page_bytes\);'
                             % (name, name, index))
        self.assertIn('get_arg_val<uint32_t>(0)', KERNEL_SOURCE)
        self.assertIn('get_arg_val<uint32_t>(1)', KERNEL_SOURCE)
        # Full-page DRAM access only (sub-page transfers silently no-op on this stack).
        self.assertEqual(re.findall(r'noc_async_(?:read|write)\(', KERNEL_SOURCE), [])
        self.assertIn('noc_async_write_barrier();', KERNEL_SOURCE)


# ---------------------------------------------------------------------------------------------
# Descriptor building against a fake ttnn.
# ---------------------------------------------------------------------------------------------

class Config:
    def __init__(self, kind):
        self.kind = kind

    def __eq__(self, other):
        return isinstance(other, Config) and other.kind == self.kind

    def __hash__(self):
        return hash(self.kind)


class Shard:
    def __init__(self, address):
        self.address = address

    def buffer_address(self):
        return self.address


class Tensor:
    def __init__(self, ops, shape, dtype='bf4', kind='dram', chips=2):
        self.shape, self.dtype, self.layout = tuple(shape), dtype, 'tile'
        self._config = Config(kind)
        self.shards = [Shard(address) for address in ops.fresh(chips)]

    def memory_config(self):
        return self._config


class Grid:
    x, y = 11, 10


class Mesh:
    def __init__(self, shape=(1, 2)):
        self.shape = shape

    def compute_with_storage_grid_size(self):
        return Grid()


class Record:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Runtime:
    def __init__(self):
        self.cells = {}

    def __getitem__(self, x):
        runtime = self

        class Column:
            def __setitem__(self, y, values):
                runtime.cells[(x, y)] = list(values)
        return Column()


class Program:
    def __init__(self, kernels, cbs):
        self.snapshot = [dict(source=Path(k.kernel_source).name, common=list(k.common_runtime_args),
                              compile=list(k.compile_time_args), defines=list(getattr(k, 'defines', [])),
                              runtime=dict(k.runtime_args.cells), processor=k.config.processor) for k in kernels]
        self.cbs = cbs


class Ops:
    bfloat4_b, bfloat16, uint32, TILE_LAYOUT = 'bf4', 'bf16', 'u32', 'tile'
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = Config('dram'), Config('l1')

    class DataMovementProcessor:
        RISCV_0, RISCV_1 = 'RISCV_0', 'RISCV_1'

    class NOC:
        RISCV_0_default, RISCV_1_default = 'NOC0', 'NOC1'

    def __init__(self):
        self.next, self.calls = 0x10000, []

    def fresh(self, chips):
        self.next += 0x100000
        return [self.next + chip for chip in range(chips)]

    def empty(self, shape, dtype, layout, device, memory_config):
        return Tensor(self, shape, dtype=dtype, kind=memory_config.kind, chips=len(pack.mesh_coordinates(device.shape)))

    def ShardTensorToMesh(self, mesh, dim):
        return ('shard', dim, len(pack.mesh_coordinates(mesh.shape)))

    def from_torch(self, value, dtype, layout, device, memory_config, mesh_mapper=None):
        _, dim, chips = mesh_mapper
        shape = list(value.shape)
        shape[dim] //= chips
        self.calls.append(('from_torch', tuple(value.shape), dtype, mesh_mapper, bool((value == 0).all())))
        return Tensor(self, shape, dtype=dtype, kind=memory_config.kind, chips=chips)

    def deallocate(self, tensor):
        self.calls.append(('deallocate', tensor))

    def get_device_tensors(self, tensor):
        return tensor.shards

    def TensorAccessorArgs(self, shard):
        class Args:
            def get_compile_time_args(self):
                return [3, 0]
        return Args()

    def CoreCoord(self, x, y):
        return (x, y)

    def CoreRange(self, a, b):
        return (a, b)

    def CoreRangeSet(self, ranges):
        return tuple(ranges)

    def Tile(self, dims):
        return tuple(dims)

    def TileDescriptor(self, tile):
        return tile

    def CBFormatDescriptor(self, **kwargs):
        return Record(**kwargs)

    def CBDescriptor(self, **kwargs):
        return Record(**kwargs)

    def RuntimeArgs(self):
        return Runtime()

    def KernelDescriptor(self, **kwargs):
        kwargs.setdefault('common_runtime_args', [])
        return Record(**kwargs)

    def DataMovementConfigDescriptor(self, **kwargs):
        return Record(**kwargs)

    def ProgramDescriptor(self, kernels, cbs):
        return Program(kernels, cbs)

    def MeshProgramDescriptor(self):
        return {}

    def MeshCoordinate(self, row, col):
        return (row, col)

    def MeshCoordinateRange(self, a, b):
        return (a, b)

    def generic_op(self, tensors, program):
        self.calls.append((list(tensors), {key: value.snapshot for key, value in program.items()}))
        return tensors[-1]


class DescriptorTests(unittest.TestCase):
    def setUp(self):
        pack.clear_cache()
        pack._SCRATCH.clear()
        self.ops, self.mesh = Ops(), Mesh()

    def tearDown(self):
        pack.clear_cache()
        pack._SCRATCH.clear()

    def layer(self):
        return Tensor(self.ops, (5120, 8704)), Tensor(self.ops, (5120, 8704))

    def test_one_scratch_per_mesh_made_like_the_served_weight(self):
        w1, _ = self.layer()
        scratch = pack.allocate_scratch(self.ops, self.mesh, w1)
        self.assertEqual((scratch.shape, scratch.dtype, scratch.memory_config()), ((5120, 17408), 'bf4', Config('dram')))
        # _build_gate_up's path: a host [K, 2N x chips] tensor, bfloat4_b, split on dim -1 across the mesh.
        self.assertEqual(self.ops.calls, [('from_torch', (5120, 34816), 'bf4', ('shard', -1, 2), True)])
        self.assertIs(pack.allocate_scratch(self.ops, self.mesh, self.layer()[0]), scratch)
        self.assertEqual(len(self.ops.calls), 1)
        pack.release_scratch(self.ops, self.mesh)
        self.assertEqual(self.ops.calls[-1], ('deallocate', scratch))
        plain = pack.allocate_scratch(self.ops, self.mesh, w1, served_topology=False)
        self.assertEqual((plain.shape, plain.dtype), ((5120, 17408), 'bf4'))

    def test_the_program_per_chip_and_the_per_call_rewrites(self):
        scratch = pack.allocate_scratch(self.ops, self.mesh, self.layer()[0])
        layers = [self.layer() for _ in range(3)]
        for w1, w3 in layers:
            self.assertIs(pack.pack_gate_up(self.mesh, w1, w3, scratch, operations=self.ops, directory=HERE), scratch)
        self.assertEqual(pack.cache_size(), 1)                  # one descriptor set for every layer
        self.assertEqual(len([call for call in self.ops.calls if call[0] != 'from_torch']), 3)
        calls = [call for call in self.ops.calls if call[0] != 'from_torch']
        for (w1, w3), (tensors, programs) in zip(layers, calls):
            self.assertEqual(tensors, [w1, w3, scratch])
            self.assertEqual(sorted(programs), [((0, 0), (0, 0)), ((0, 1), (0, 1))])
            for chip, key in enumerate(sorted(programs)):
                kernels = programs[key]
                self.assertEqual([k['processor'] for k in kernels], ['RISCV_0', 'RISCV_1'])
                addresses = [t.shards[chip].address for t in (w1, w3, scratch)]
                covered = []
                for processor, kernel in enumerate(kernels):
                    self.assertEqual(kernel['source'], pack.KERNEL)
                    self.assertEqual(kernel['common'], addresses)          # this call's layer, not the first
                    self.assertEqual(kernel['compile'], [pack.PAGE, 272, pack.BATCH, processor] + [3, 0] * 3)
                    self.assertEqual([name for name, _ in kernel['defines']], ['C1E_SRC_SHA'])
                    self.assertEqual(len(kernel['runtime']), 110)
                    covered += [p for start, count in kernel['runtime'].values() for p in range(start, start + count)]
                self.assertEqual(sorted(covered), list(range(43520)))

    def test_refusals(self):
        scratch = pack.allocate_scratch(self.ops, self.mesh, self.layer()[0])
        w1, w3 = self.layer()
        cases = [
            (Tensor(self.ops, (5120, 8704), dtype='bf8'), w3, scratch),
            (w1, Tensor(self.ops, (5120, 8704), kind='l1'), scratch),
            (w1, Tensor(self.ops, (5120, 4352)), scratch),
            (w1, w3, Tensor(self.ops, (5120, 8704))),
            (w1, w3, None),
        ]
        for case in cases:
            with self.assertRaises(pack.Unsupported):
                pack.pack_gate_up(self.mesh, *case, operations=self.ops, directory=HERE)
        aliased = Tensor(self.ops, (5120, 17408))
        aliased.shards = w1.shards
        with self.assertRaisesRegex(pack.Unsupported, 'alias'):
            pack.pack_gate_up(self.mesh, w1, w3, aliased, operations=self.ops, directory=HERE)
        self.assertEqual([call for call in self.ops.calls if call[0] != 'from_torch'], [])

    def test_a_one_chip_device(self):
        mesh = Mesh((1, 1))
        w1, w3 = Tensor(self.ops, (5120, 8704), chips=1), Tensor(self.ops, (5120, 8704), chips=1)
        scratch = pack.allocate_scratch(self.ops, mesh, w1)
        self.assertEqual(self.ops.calls[-1][:2], ('from_torch', (5120, 17408)))
        pack.pack_gate_up(mesh, w1, w3, scratch, operations=self.ops, directory=HERE)
        self.assertEqual(sorted(self.ops.calls[-1][1]), [((0, 0), (0, 0))])


class ScratchTests(unittest.TestCase):
    """One scratch per (mesh, projection shape): the model's first MLP allocates it from args (a shape),
    every later layer - and the forward, from a w1 tensor - gets the same one."""

    def setUp(self):
        pack.clear_cache()
        pack._SCRATCH.clear()
        self.ops, self.mesh = Ops(), Mesh()

    def tearDown(self):
        pack._SCRATCH.clear()

    def test_a_shape_and_a_w1_shard_name_the_same_scratch(self):
        self.assertFalse(pack.has_scratch(self.mesh, (5120, 8704)))
        scratch = pack.allocate_scratch(self.ops, self.mesh, (5120, 8704))
        self.assertTrue(pack.has_scratch(self.mesh, (5120, 8704)))
        self.assertTrue(pack.has_scratch(self.mesh, Tensor(self.ops, (1, 1, 5120, 8704))))
        self.assertIs(pack.allocate_scratch(self.ops, self.mesh, Tensor(self.ops, (5120, 8704))), scratch)
        self.assertEqual(len([call for call in self.ops.calls if call[0] == 'from_torch']), 1)

    def test_another_shape_or_mesh_gets_its_own_and_release_frees_a_whole_mesh(self):
        first = pack.allocate_scratch(self.ops, self.mesh, (5120, 8704))
        other_shape = pack.allocate_scratch(self.ops, self.mesh, (5120, 4352))
        other_mesh = Mesh()
        elsewhere = pack.allocate_scratch(self.ops, other_mesh, (5120, 8704))
        self.assertEqual(len({id(first), id(other_shape), id(elsewhere)}), 3)
        self.assertEqual(other_shape.shape, (5120, 8704))
        pack.release_scratch(self.ops, self.mesh)
        freed = [call[1] for call in self.ops.calls if call[0] == 'deallocate']
        self.assertEqual(sorted(map(id, freed)), sorted([id(first), id(other_shape)]))
        self.assertTrue(pack.has_scratch(other_mesh, (5120, 8704)))
        self.assertFalse(pack.has_scratch(self.mesh, (5120, 8704)))

    def test_a_shape_that_is_not_tile_aligned_is_refused(self):
        with self.assertRaises(ValueError):
            pack.allocate_scratch(self.ops, self.mesh, (5120, 8700))
        self.assertEqual(self.ops.calls, [])


class AuditTests(unittest.TestCase):
    """audit_pairs: packed gate pages vs w1 (offset 0) and up pages vs w3 (offset 1) with
    packed_weight_check.cpp, the result tensors freed, exact only when every chip of both is."""

    def run_audit(self, verdicts):
        ops, mesh = Ops(), Mesh()
        w1, w3 = Tensor(ops, (5120, 8704)), Tensor(ops, (5120, 8704))
        packed = Tensor(ops, (5120, 17408))
        launched = []

        def check_pairs(mesh_, packed_, separate, offset, owned, operations=None, directory=None):
            launched.append((packed_, separate, offset))
            result = ('result', offset)
            owned.append(result)
            return result

        def read_check(operations, result, pages):
            return [dict(chip=chip, pages=pages, mismatched_words=0 if ok else 5, exact=ok)
                    for chip, ok in enumerate(verdicts[result[1]])]

        with mock.patch.object(pack, 'check_pairs', check_pairs), mock.patch.object(pack, 'read_check', read_check):
            report = pack.audit_pairs(ops, mesh, packed, w1, w3)
        self.assertEqual(launched, [(packed, w1, 0), (packed, w3, 1)])
        self.assertEqual([call[1] for call in ops.calls if call[0] == 'deallocate'], [('result', 0), ('result', 1)])
        return report

    def test_exact_needs_every_chip_of_both_projections(self):
        report = self.run_audit({0: (True, True), 1: (True, True)})
        self.assertEqual((report['exact'], report['pages'], report['mismatched_words']), (True, 43520, 0))
        report = self.run_audit({0: (True, True), 1: (True, False)})
        self.assertEqual((report['exact'], report['mismatched_words']), (False, 5))
        self.assertEqual([item['exact'] for item in report['up']], [True, False])

    def test_the_result_tensors_are_freed_when_a_check_raises(self):
        ops, mesh = Ops(), Mesh()
        w1, w3, packed = Tensor(ops, (5120, 8704)), Tensor(ops, (5120, 8704)), Tensor(ops, (5120, 17408))

        def check_pairs(mesh_, packed_, separate, offset, owned, operations=None, directory=None):
            owned.append('output')
            return 'result'

        def read_check(operations, result, pages):
            raise AssertionError('incomplete device comparison coverage')

        with mock.patch.object(pack, 'check_pairs', check_pairs), mock.patch.object(pack, 'read_check', read_check):
            with self.assertRaisesRegex(AssertionError, 'coverage'):
                pack.audit_pairs(ops, mesh, packed, w1, w3)
        self.assertEqual(ops.calls, [('deallocate', 'output')])


class DoubleRoundingTests(unittest.TestCase):
    """Why no separate-matmul formulation (C1, C1c, C1d) can be byte-identical to the served fused op:
    with the SAME fp32 gate and up, bf16(silu(g) * u) rounded once differs from the product of the
    bf16-rounded silu(g) and u in a large share of elements."""

    def test_rounding_first_changes_a_large_share_of_the_products(self):
        generator = torch.Generator().manual_seed(7)
        gate = torch.randn(200000, generator=generator) * 1.43
        up = torch.randn(200000, generator=generator) * 1.43
        silu = torch.nn.functional.silu(gate)
        fused = (silu * up).to(torch.bfloat16).view(torch.int16)                      # one rounding
        separate = (silu.to(torch.bfloat16).float() * up.to(torch.bfloat16).float()).to(torch.bfloat16).view(torch.int16)
        self.assertGreater(int((fused != separate).sum()) / 200000, 0.25)

    def test_the_documented_counterexample(self):
        g = torch.tensor([0xBFB582D0], dtype=torch.int64).to(torch.int32).view(torch.float32)
        u = torch.tensor([0x3FAD03BC], dtype=torch.int64).to(torch.int32).view(torch.float32)
        s = torch.nn.functional.silu(g)
        self.assertEqual(int((s * u).to(torch.bfloat16).view(torch.int16)) & 0xFFFF, 0xBEBF)
        self.assertEqual(int((s.to(torch.bfloat16).float() * u.to(torch.bfloat16).float()).to(torch.bfloat16)
                             .view(torch.int16)) & 0xFFFF, 0xBEC0)
        self.assertIn('0xbfb582d0, up 0x3fad03bc: fused 0xbebf', pack.__doc__)


if __name__ == '__main__':
    unittest.main()
