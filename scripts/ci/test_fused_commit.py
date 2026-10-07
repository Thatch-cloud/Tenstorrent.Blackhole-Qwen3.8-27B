"""Round-fence plan H1b, F-I - the fused commit (fused_commit.py; QWEN_FAST_FUSED_COMMIT, _INPLACE,
_LIVE_BANKS, _AUDIT, every one default off).

What is pinned here, on CPU:
  - the flags, and the arm's and the gate's refusals of a sub-flag without its parent;
  - the slide program: one bank out of place is draft_kv_slide.prepare's own program, field for field,
    over a fake ttnn; in place every bank's 16 workers carry [bank, delta, bank, 2048, prefix, drop,
    rows, worker] and the io list is [bank, delta, bank] per bank, five banks a program;
  - T_proj IS today's per-user op sequence at count 16: DFlashDevice.project_features, DraftKVHistory.
    project_inputs and project_key_value at prefix 16 log the very same calls, shapes and arguments as
    the captured body (minus today's two table uploads, plus the ten delta copies), and the staged
    tables are the bytes today uploads;
  - R2: every fused-commit buffer is allocated before the block's verify capture; T_proj and the slide
    traces are captured after the GDN commit traces (the real packed block over test_packed_verifier's
    fixture);
  - the publication: a guarded fused publication enqueues T_proj and the slide without a fence, marks
    the history stale, commits with no swap (in place) or today's swap (out of place); every refusal
    takes today's installers argument for argument; the window stages the RoPE tables; the audit
    shadows and repairs; F4 binds the pair to the live banks and copy_cache goes;
  - with every flag off, each module H1b touches is its PARENT (653732d8) call for call.
"""

from contextlib import contextmanager
from itertools import count
import difflib
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dflash_device import DFlashDevice
import draft_kv_history
from draft_kv_history import DraftKVHistory
import fused_commit
from fused_commit import DELTA_SHAPE, HEADS, KV_SHAPE
from packed_shapes import m3_shape
from packed_verifier import PackedFeatureTaps
import packed_verifier
import serving_packed_step
import test_packed_verifier as tpv
import verifier_engine
import verify_prestage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# H1b's parent: M-F0's harness commit, after H1a (596226f8) and its gate (89a4cde9).
PARENT = '653732d8'
PAGE_WIDTH = tpv.PAGE_WIDTH


def clean_environment(**flags):
    environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environ.update(flags)
    return patch.dict(os.environ, environ, clear=True)


def parent_module(relative):
    from test_padded_block import parent_module as load

    return load(relative, PARENT)


def logged():
    """Every line the modules log (loguru stubbed), as the server log would hold it."""
    lines = []
    stub = ModuleType('loguru')
    stub.logger = SimpleNamespace(info=lambda template, *values, **named: lines.append(template.format(*values, **named)),
                                  warning=lambda *args, **kwargs: None)
    return lines, patch.dict('sys.modules', {'loguru': stub})


# ------------------------------------------------------------------------------------------------------
# Flags
# ------------------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_every_flag_is_zero_or_one_and_off_by_default(self):
        for reader, flag in ((fused_commit.enabled, fused_commit.FLAG),
                             (fused_commit.inplace_enabled, fused_commit.INPLACE_FLAG),
                             (fused_commit.live_banks_enabled, fused_commit.LIVE_BANKS_FLAG),
                             (fused_commit.audit_enabled, fused_commit.AUDIT_FLAG)):
            with self.subTest(flag=flag):
                self.assertFalse(reader({}))
                for value in ('true', '2', ''):
                    with self.assertRaises(ValueError):
                        reader({fused_commit.FLAG: '1', fused_commit.INPLACE_FLAG: '1', flag: value})

    def test_a_sub_flag_is_inert_without_its_parent(self):
        on = {fused_commit.FLAG: '1', fused_commit.INPLACE_FLAG: '1', fused_commit.LIVE_BANKS_FLAG: '1',
              fused_commit.AUDIT_FLAG: '1'}
        self.assertTrue(all(reader(on) for reader in (fused_commit.enabled, fused_commit.inplace_enabled,
                                                      fused_commit.live_banks_enabled, fused_commit.audit_enabled)))
        alone = dict(on, **{fused_commit.FLAG: '0'})
        self.assertFalse(any(reader(alone) for reader in (fused_commit.inplace_enabled, fused_commit.live_banks_enabled,
                                                          fused_commit.audit_enabled)))
        self.assertFalse(fused_commit.live_banks_enabled(dict(on, **{fused_commit.INPLACE_FLAG: '0'})))

    def test_the_block_imports_the_module_only_under_the_flag(self):
        self.assertFalse(packed_verifier.fused_commit_requested({}))
        self.assertTrue(packed_verifier.fused_commit_requested({'QWEN_FAST_FUSED_COMMIT': '1'}))


# ------------------------------------------------------------------------------------------------------
# The slide program against the served driver
# ------------------------------------------------------------------------------------------------------

class ProgramTTNN:
    """The descriptor surface draft_kv_slide.prepare and fused_commit.slide_program use: every descriptor
    a SimpleNamespace (equal by value), device tensors with two shards."""

    bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'
    DataMovementProcessor = SimpleNamespace(RISCV_0='riscv0')
    NOC = SimpleNamespace(RISCV_0_default='noc0')

    def __init__(self):
        self.generic = []
        self.addresses = count(0x10000, 0x1000)

    def tensor(self, shape, *, tile=(32, 32), dtype='bf16'):
        shards = []
        for chip in range(2):
            address = next(self.addresses)
            shards.append(SimpleNamespace(shape=tuple(shape), dtype=dtype, layout='tile', memory_config=lambda: 'dram',
                                          tile=SimpleNamespace(tile_shape=tile, transpose_of_faces=False,
                                                               transpose_within_face=False),
                                          buffer_address=lambda address=address: address, chip=chip))
        return SimpleNamespace(shape=tuple(shape), shards=shards)

    def get_device_tensors(self, value):
        return value.shards

    def CoreCoord(self, x, y):
        return SimpleNamespace(x=x, y=y)

    def CoreRange(self, start, end):
        return ('range', start.x, start.y, end.x, end.y)

    def CoreRangeSet(self, ranges):
        return ('set', tuple(ranges))

    def CBDescriptor(self, **fields):
        return SimpleNamespace(kind='cb', **fields)

    def CBFormatDescriptor(self, **fields):
        return SimpleNamespace(kind='cb-format', **fields)

    def TileDescriptor(self, tile):
        return ('tile-descriptor', tile)

    def Tile(self, shape):
        return ('tile', shape)

    def MeshProgramDescriptor(self):
        return {}

    def KernelDescriptor(self, **fields):
        return SimpleNamespace(kind='kernel', **fields)

    def DataMovementConfigDescriptor(self, **fields):
        return SimpleNamespace(kind='dm-config', **fields)

    def RuntimeArgs(self):
        from collections import defaultdict

        return defaultdict(dict)

    def TensorAccessorArgs(self, value):
        return SimpleNamespace(get_compile_time_args=lambda: [len(value.shape), value.shape[1], 7])

    def MeshCoordinate(self, row, column):
        return ('coordinate', row, column)

    def MeshCoordinateRange(self, start, end):
        return ('coordinate-range', start, end)

    def ProgramDescriptor(self, **fields):
        return SimpleNamespace(kind='program', **fields)

    def generic_op(self, tensors, program):
        self.generic.append((list(tensors), program))
        return tensors[-1]


def mesh(x=11, y=10):
    return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=x, y=y))


class SlideProgramTests(unittest.TestCase):
    def setUp(self):
        self.ttnn = ProgramTTNN()
        patcher = patch.dict('sys.modules', {'ttnn': self.ttnn})
        patcher.start()
        self.addCleanup(patcher.stop)

    def served(self, grid, active, delta, spare, prefix):
        import draft_kv_slide

        draft_kv_slide.prepare(grid, active, delta, spare, history_rows=2048, prefix=prefix)()
        return self.ttnn.generic.pop()

    def test_one_bank_out_of_place_is_the_served_drivers_program_field_for_field(self):
        grid = mesh()
        active, delta, spare = (self.ttnn.tensor(shape) for shape in (KV_SHAPE, DELTA_SHAPE, KV_SHAPE))
        for prefix in (1, 7, 16):
            with self.subTest(prefix=prefix):
                tensors, served = self.served(grid, active, delta, spare, prefix)
                ours = fused_commit.slide_program(self.ttnn, grid, fused_commit.kernel_path(), [(active, delta, spare)],
                                                  history_rows=2048, prefix=prefix, in_place=False)
                self.assertEqual(tensors, [active, delta, spare])
                self.assertEqual(ours, served)

    def test_in_place_five_banks_is_one_eighty_worker_program_per_chip(self):
        grid = mesh()
        banks = [(self.ttnn.tensor(KV_SHAPE), self.ttnn.tensor(DELTA_SHAPE)) for bank in range(5)]
        program = fused_commit.slide_program(self.ttnn, grid, fused_commit.kernel_path(), banks, history_rows=2048,
                                             prefix=9)
        self.assertEqual(len(program), 2)
        for chip in range(2):
            (kernel,) = program[('coordinate-range', ('coordinate', 0, chip), ('coordinate', 0, chip))].kernels
            self.assertEqual(kernel.kernel_source, str(fused_commit.kernel_path()))
            cores = [(x, y) for x, row in kernel.runtime_args.items() for y in row]
            self.assertEqual(len(cores), 80)
            for index, (bank, delta) in enumerate(banks):
                for worker in range(16):
                    core = index * 16 + worker
                    arguments = kernel.runtime_args[core % 11][core // 11]
                    self.assertEqual(arguments, [bank.shards[chip].buffer_address(), delta.shards[chip].buffer_address(),
                                                 bank.shards[chip].buffer_address(), 2048, 9, 9, 2048, worker])
        self.assertEqual(fused_commit.io_list(banks[:2]), [banks[0][0], banks[0][1], banks[0][0],
                                                           banks[1][0], banks[1][1], banks[1][0]])

    def test_the_builder_refuses_what_the_served_driver_would(self):
        grid = mesh()
        bank = self.ttnn.tensor(KV_SHAPE)
        cases = {
            'aliased delta': [(bank, bank)],
            'short delta': [(bank, self.ttnn.tensor((1, 4, 16, 128)))],
            'transposed tile': [(bank, self.ttnn.tensor(DELTA_SHAPE, tile=(16, 32)))],
            'small grid': [(self.ttnn.tensor(KV_SHAPE), self.ttnn.tensor(DELTA_SHAPE)) for index in range(5)],
        }
        for name, banks in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                fused_commit.slide_program(self.ttnn, mesh(4, 4) if name == 'small grid' else grid,
                                           fused_commit.kernel_path(), banks, history_rows=2048, prefix=4)

    def test_the_kernel_is_the_served_drivers_sibling_and_qualified(self):
        import draft_kv_slide

        self.assertEqual(fused_commit.kernel_path(), Path(draft_kv_slide.__file__).with_suffix('.cpp'))
        self.assertIn(fused_commit.kernel_kind(fused_commit.kernel_path()), ('scalar', 'direct'))
        self.assertEqual(fused_commit.kernel_kind(HERE / 'draft_kv_slide_direct.cpp'), 'direct')
        self.assertIsNone(fused_commit.kernel_kind(HERE / 'fused_commit.py'))
        from draft_kv_slide_gate import QUALIFIED_KERNELS

        self.assertEqual(set(fused_commit.QUALIFIED_KERNEL_SHA256), set(QUALIFIED_KERNELS.values()))


class HostTablesTests(unittest.TestCase):
    def test_the_tables_are_draft_kv_historys_at_count_sixteen(self):
        from draft_head_preparation import rope_tables

        cos, sin = fused_commit.host_tables(131000, 32, 16)
        reference = rope_tables(131000, 32)
        for table, full in zip((cos, sin), reference):
            self.assertTrue(torch.equal(table[..., :16, :], full[..., :16, :]))
            self.assertFalse(table[..., 16:, :].any())
            self.assertEqual((tuple(table.shape), table.dtype), ((1, 1, 32, 128), torch.bfloat16))


# ------------------------------------------------------------------------------------------------------
# The FusedCommit over fakes
# ------------------------------------------------------------------------------------------------------

class FusedOps(tpv.FakeTTNN):
    """test_packed_verifier's fake ttnn plus what the fused commit calls; every call in `log`."""

    MathFidelity = SimpleNamespace(HiFi4='hifi4')

    def __init__(self):
        super().__init__()
        self.log, self.generic = [], []

    def WormholeComputeKernelConfig(self, **fields):
        return ('kernel-config',) + tuple(sorted(fields.items()))

    def slice(self, value, start, end):
        result = self.allocate(tuple(last - first for first, last in zip(start, end)))
        self.log.append(('slice', value, tuple(start), tuple(end)))
        return result

    def pad(self, value, padding, fill):
        result = self.allocate(tuple(size + before + after for size, (before, after) in zip(value.shape, padding)))
        self.log.append(('pad', value, tuple(map(tuple, padding)), fill))
        return result

    def generic_op(self, tensors, program):
        self.generic.append((list(tensors), program))
        self.log.append(('generic_op', program))
        return tensors[-1]

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        super().execute_trace(mesh, trace, cq_id=cq_id, blocking=blocking)
        self.log.append(('execute_trace', trace, blocking))

    def synchronize_device(self, mesh):
        super().synchronize_device(mesh)
        self.log.append(('sync',))

    def copy(self, source, destination):
        super().copy(source, destination)
        self.log.append(('copy', source, destination))

    def copy_host_to_device_tensor(self, host, destination):
        super().copy_host_to_device_tensor(host, destination)
        self.log.append(('host_copy', destination))


def pooled_slot(ops, index):
    kv = [{side: {name: ops.allocate(KV_SHAPE) for name in HEADS} for side in ('active', 'spare')} for layer in range(5)]
    return SimpleNamespace(index=index, lent=False, kv=kv, query=ops.allocate((1, 1, 32, 2048)),
                           verifier=SimpleNamespace(carry=tpv.snapshot_set(ops)))


def shared_weights(ops):
    projection, norm = ops.allocate((10240, 5120)), ops.allocate((1, 1, 160, 32))
    layers = [(dict(name='attention%d' % layer), 'mlp', 'weights', 'convolution') for layer in range(5)]
    return SimpleNamespace(closed=False, tensors=[projection, norm], lend=Mock(), layers=layers, projection=projection,
                           feature_norm=norm)


class FusedFixture(unittest.TestCase):
    """A FusedCommit over a fake four-user block: the seams recorded, the slide programs named."""

    INPLACE = True
    AUDIT = False

    def setUp(self):
        self.ops = FusedOps()
        self.mesh = mesh()
        self.slots = [pooled_slot(self.ops, index) for index in range(4)]
        self.weights = shared_weights(self.ops)
        self.collectives = SimpleNamespace(name='shared-ccl')
        self.taps = tuple(self.ops.allocate((1, 1, 64, 2560)) for tap in range(5))
        self.block = SimpleNamespace(rows_per_user=16, users=4, shape=m3_shape(PAGE_WIDTH), segment_slots=self.slots,
                                     taps=self.taps, rounds=3,
                                     segment_of=lambda engine: engine.segment)
        self.calls = []
        for target, value in (('slide_scope_live', Mock(return_value=True)),
                              ('slide_program', Mock(side_effect=self.program)),
                              ('_project_features', Mock(side_effect=self.project_features)),
                              ('_project_key_value', Mock(side_effect=self.project_key_value))):
            patcher = patch.object(fused_commit, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.lines, loguru = logged()
        loguru.start()
        self.addCleanup(loguru.stop)
        environment = clean_environment()
        environment.start()
        self.addCleanup(environment.stop)
        self.traces = count(1)

    def program(self, ttnn, grid, source, banks, *, history_rows, prefix, in_place=True):
        return SimpleNamespace(prefix=prefix, banks=tuple(bank for bank, delta in banks), history_rows=history_rows,
                               source=source, in_place=in_place)

    def project_features(self, owner, features, count_, row_offset, retain):
        self.calls.append(('project_features', owner, tuple(features), count_, row_offset))
        return retain(self.ops.allocate((1, 1, count_, 5120)))

    def project_key_value(self, operations, inputs, query, tables, retain, *, parameters):
        self.calls.append(('project_key_value', inputs.shape, query, tuple(tables), parameters['name']))
        return dict(q=retain(self.ops.allocate((1, 16, 32, 128))), k=retain(self.ops.allocate(DELTA_SHAPE)),
                    v=retain(self.ops.allocate(DELTA_SHAPE)))

    def capture(self, operations, grid, operation):
        self.ops.log.append(('capture', getattr(operation, '__qualname__', '?')))
        operation()
        return 'trace%d' % next(self.traces), None

    def build(self, **options):
        values = dict(operations=self.ops, mesh=self.mesh, pool=None, shared_weights=self.weights,
                      collectives=self.collectives, inplace=self.INPLACE, audit=self.AUDIT)
        values.update(options)
        return fused_commit.FusedCommit(self.block, **values)

    def built(self):
        fused = self.build()
        fused.capture(self.capture)
        self.ops.log.clear()
        self.calls.clear()
        return fused


class ConstructionTests(FusedFixture):
    def test_every_segment_allocates_two_tables_and_ten_deltas_and_nothing_else(self):
        fused = self.build()
        self.assertEqual(len(fused.allocated()), 4 * 12)
        self.assertEqual(self.ops.device_uploads, fused.allocated())
        for storage in fused.segments:
            self.assertEqual([tuple(table.shape) for table in storage.tables], [(1, 1, 32, 128)] * 2)
            self.assertEqual({tuple(storage.deltas[layer][name].shape) for layer in range(5) for name in HEADS},
                             {DELTA_SHAPE})
        self.assertEqual([storage.row_offset for storage in fused.segments], [0, 16, 32, 48])
        self.assertEqual(fused.parameters, tuple(layer[0] for layer in self.weights.layers))

    def test_host_checks_refuse_before_anything_is_allocated(self):
        cases = dict(collectives=dict(collectives=None), weights=dict(shared_weights=SimpleNamespace(layers=())))
        for name, options in cases.items():
            with self.subTest(name=name), self.assertRaises(fused_commit.Refused):
                self.build(**options)
        self.slots[2].kv = self.slots[2].kv[:4]
        with self.assertRaises(fused_commit.Refused):
            self.build()
        self.slots[2].kv = pooled_slot(self.ops, 2).kv
        fused_commit.slide_scope_live.return_value = False
        with self.assertRaises(fused_commit.Refused):
            self.build()
        fused_commit.slide_scope_live.return_value = True
        with patch.object(fused_commit, 'kernel_path', return_value=HERE / 'fused_commit.py'), \
                self.assertRaises(fused_commit.Refused):
            self.build()
        with self.assertRaises(fused_commit.Refused):
            self.build(mesh=mesh(4, 4))
        self.assertEqual(self.ops.device_uploads, [])

    def test_build_logs_a_refusal_and_serves_today(self):
        lines = []
        with clean_environment(QWEN_FAST_FUSED_COMMIT='1'):
            self.assertIsNone(fused_commit.build(self.block, operations=self.ops, mesh=self.mesh, pool=None,
                                                 shared_weights=self.weights, collectives=None,
                                                 diagnostic=lines.append))
        self.assertTrue(lines[0].startswith(fused_commit.REFUSED_MARKER + ' users=4 reason=no_collectives'))
        with clean_environment():
            self.assertIsNone(fused_commit.build(self.block, operations=self.ops, mesh=self.mesh, pool=None,
                                                 shared_weights=self.weights, collectives=self.collectives,
                                                 diagnostic=lines.append))


class CaptureTests(FusedFixture):
    def test_t_proj_is_the_feature_projection_the_pad_and_five_kv_projections_into_the_deltas(self):
        fused = self.build()
        fused.capture(self.capture)
        storage = fused.segments[2]
        owner_calls = [call for call in self.calls if call[0] == 'project_features']
        self.assertEqual(len(owner_calls), 8, 'per segment: warmed, then captured')
        owner = owner_calls[4][1]
        self.assertIs(owner.projection, self.weights.projection)
        self.assertIs(owner.feature_norm, self.weights.feature_norm)
        self.assertIs(owner.collectives, self.collectives)
        self.assertEqual(owner.kernel, self.ops.WormholeComputeKernelConfig(
            math_fidelity='hifi4', math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
        self.assertEqual([call[3:] for call in owner_calls], [(16, 0), (16, 0), (16, 16), (16, 16), (16, 32), (16, 32),
                                                              (16, 48), (16, 48)])
        self.assertEqual(owner_calls[5][2], self.taps)
        kv = [call for call in self.calls if call[0] == 'project_key_value']
        self.assertEqual(len(kv), 40)
        segment_two = kv[25:30]
        self.assertEqual([call[4] for call in segment_two], ['attention%d' % layer for layer in range(5)])
        for call in segment_two:
            self.assertEqual(call[1], (1, 1, 32, 5120))
            self.assertIs(call[2], self.slots[2].query)
            self.assertEqual(call[3], storage.tables)

    def test_the_body_slices_pads_and_copies_every_head_into_its_delta(self):
        fused = self.build()
        storage = fused.segments[1]
        storage.taps = self.taps
        owned = []
        self.ops.log.clear()
        fused.project(storage, owned)
        names = [entry[0] for entry in self.ops.log]
        self.assertEqual(names, ['slice', 'pad'] + ['copy'] * 10)
        self.assertEqual(self.ops.log[0][2:], ((0, 0, 0, 0), (1, 1, 16, 5120)))
        self.assertEqual(self.ops.log[1][2], ((0, 0), (0, 0), (0, 16), (0, 0)))
        copied = [entry[2] for entry in self.ops.log[2:]]
        self.assertEqual(copied, [storage.deltas[layer][name] for layer in range(5) for name in HEADS])
        protected = set(map(id, [*storage.tables, *self.taps, storage.query]))
        self.assertFalse(any(id(value) in protected for value in owned))

    def test_the_retainer_keeps_protected_buffers_and_refuses_a_partial_alias(self):
        fused = self.build()
        storage = fused.segments[0]
        storage.taps = self.taps
        owned = []
        retain = fused.retainer(storage, owned)
        retain(storage.tables[0])
        retain(self.weights.projection)
        self.assertEqual(owned, [])
        fresh = self.ops.allocate((1, 1, 32, 5120))
        self.assertIs(retain(fresh), fresh)
        self.assertEqual(owned, [fresh])
        alias = self.ops.allocate((3,))
        alias.shards = [tpv.FakeShard(alias, storage.tables[1].shards[0].address), tpv.FakeShard(alias, 1)]
        with self.assertRaises(ValueError):
            retain(alias)

    def test_in_place_sixty_four_slide_traces_follow_four_projection_traces_each_warmed_first(self):
        fused = self.build()
        fused.capture(self.capture)
        captures = [entry[1] for entry in self.ops.log if entry[0] == 'capture']
        self.assertEqual(len(captures), 4 + 64)
        self.assertTrue(all('FusedCommit.capture' in name for name in captures))
        first_capture = next(index for index, entry in enumerate(self.ops.log) if entry[0] == 'capture')
        warm_projections = [entry for entry in self.ops.log[:first_capture] if entry[0] == 'copy']
        self.assertEqual(len(warm_projections), 10, 'the first T_proj ran eagerly before its capture')
        slide_captures = [index for index, entry in enumerate(self.ops.log) if entry[0] == 'capture'][4:]
        warm_slides = [entry for entry in self.ops.log[:slide_captures[0]] if entry[0] == 'generic_op']
        self.assertEqual(len(warm_slides), 4 * 16 * 2, 'every slide program once before any slide capture')
        self.assertEqual(fused.trace_count(), 68)
        for storage in fused.segments:
            self.assertEqual(sorted(storage.slides), list(range(1, 17)))
            for prefix, programs in storage.slide_programs.items():
                self.assertEqual([len(chunk) for chunk, program in programs], [5, 5])
                banks = [bank for chunk, program in programs for bank, delta in chunk]
                self.assertEqual(banks, [storage.slot.kv[layer]['active'][name] for layer in range(5) for name in HEADS])
                deltas = [delta for chunk, program in programs for bank, delta in chunk]
                self.assertEqual(deltas, [storage.deltas[layer][name] for layer in range(5) for name in HEADS])
                self.assertEqual({(program.prefix, program.history_rows, program.in_place) for chunk, program in programs},
                                 {(prefix, 2048, True)})
        tensors, program = self.ops.generic[-1]
        chunk = fused.segments[3].slide_programs[16][1][0]
        self.assertEqual(tensors, fused_commit.io_list(chunk))

    def test_out_of_place_captures_only_the_projections(self):
        fused = self.build(inplace=False)
        fused.capture(self.capture)
        self.assertEqual(sum(1 for entry in self.ops.log if entry[0] == 'capture'), 4)
        self.assertEqual(self.ops.generic, [])
        self.assertEqual(fused.trace_count(), 4)

    def test_close_releases_every_trace_and_buffer_but_never_a_pooled_bank(self):
        fused = self.build()
        fused.capture(self.capture)
        buffers = list(fused.allocated())
        released_before = len(self.ops.released)
        fused.close()
        self.assertEqual(len(self.ops.released) - released_before, 68)
        deallocated = set(map(id, self.ops.deallocated))
        self.assertTrue(all(id(value) in deallocated for value in buffers))
        banks = [value for slot in self.slots for layer in slot.kv for side in layer.values() for value in side.values()]
        self.assertFalse(any(id(bank) in deallocated for bank in banks))
        self.assertFalse(any(id(slot.query) in deallocated for slot in self.slots))
        fused.close()


def fused_drafter(fixture, fused, segment, *, position=5000, history_rows=2048, parity=False):
    """A DFlashDevice and its DraftKVHistory, built without their constructors, on the segment's slot."""
    ops, slot = fixture.ops, fixture.slots[segment]
    cache = object.__new__(DraftKVHistory)
    active = [dict(layer['active']) for layer in slot.kv]
    spare = [dict(layer['spare']) for layer in slot.kv]
    cache.active, cache.spare = (spare, active) if parity else (active, spare)
    cache.operations, cache.mesh, cache.parameters = ops, fixture.mesh, fused.parameters
    cache.position, cache.history_rows, cache.pending, cache.closed = position, history_rows, None, False
    cache.owned, cache.borrowed, cache.checks, cache.projection, cache.query = [], [], [], None, slot.query
    device = object.__new__(DFlashDevice)
    device.operations, device.mesh, device.kv_history = ops, fixture.mesh, cache
    device.position, device.history_rows, device.pending, device.closed = position, 2048, None, False
    device.progress, device.proposal_capture, device.pool_slot = None, object(), slot
    device.projection, device.feature_norm = fixture.weights.projection, fixture.weights.feature_norm
    device.collectives = fixture.collectives
    device.history, device.spare_history, device.published_rows = 'history', 'spare-history', 0
    return device


def taps_for(fixture, segment):
    return PackedFeatureTaps(fixture.taps, row_offset=16 * segment, rows=16)


class PublicationTests(FusedFixture):
    def setUp(self):
        super().setUp()
        self.fused = self.built()
        self.today = Mock(side_effect=lambda device, features, prefix, **options: ('today', prefix, options))

        def prepare_publication(device, features, prefix, **options):
            return self.today(device, features, prefix, **options)

        patcher = patch.object(DFlashDevice, 'prepare_publication', prepare_publication)
        patcher.start()
        self.addCleanup(patcher.stop)

    def publish(self, device, segment, prefix, *, merge_release=False, fused_steady_state=False, position=None):
        restore = fused_commit.install_fused_commit(device, self.fused, segment, merge_release=merge_release,
                                                    fused_steady_state=fused_steady_state)
        try:
            publication = device.prepare_publication(taps_for(self, segment), prefix,
                                                     position=device.position if position is None else position)
            if isinstance(publication, SimpleNamespace):
                device.commit_publication(publication)
            return publication
        finally:
            restore()

    def test_a_fused_publication_enqueues_t_proj_then_the_slide_with_no_fence_and_commits_without_a_swap(self):
        device = fused_drafter(self, self.fused, 1)
        self.fused.stage_tables(1, 5000)
        self.ops.log.clear()
        cache = device.kv_history
        active, spare = cache.active, cache.spare
        publication = self.publish(device, 1, 7)
        storage = self.fused.segments[1]
        self.assertEqual(self.ops.log, [('execute_trace', storage.projection_trace, False),
                                        ('execute_trace', storage.slides[7], False)])
        self.today.assert_not_called()
        self.assertEqual(publication.status, 'committed')
        self.assertEqual((device.position, cache.position, cache.history_rows), (5007, 5007, 2048))
        self.assertIs(cache.active, active)
        self.assertIs(cache.spare, spare)
        self.assertTrue(device.history_stale)
        self.assertIsNone(device.pending)
        self.assertIsNone(cache.pending)
        self.assertNotIn('commit', vars(cache))
        self.assertNotIn('prepare_publication', vars(device))
        self.assertEqual(self.lines[-1], '[PACKED-FUSED] round=3 segment=1 prefix=7 path=fused reason=- tables=window')

    def test_a_late_table_is_staged_before_t_proj(self):
        device = fused_drafter(self, self.fused, 0, position=6000)
        self.publish(device, 0, 3)
        storage = self.fused.segments[0]
        self.assertEqual([entry[0] for entry in self.ops.log], ['host_copy', 'host_copy', 'execute_trace', 'execute_trace'])
        self.assertEqual([entry[1] for entry in self.ops.log[:2]], list(storage.tables))
        for table, expected in zip(storage.tables, fused_commit.host_tables(6000, 32, 16)):
            self.assertTrue(torch.equal(table.value, expected))
        self.assertTrue(self.lines[-1].endswith('tables=late'))
        self.assertEqual(storage.staged_position, 6000)

    def test_every_refusal_takes_todays_path_argument_for_argument(self):
        def refuse(name, device):
            if name == 'ramp':
                device.kv_history.history_rows = 2000
            elif name == 'slot':
                device.pool_slot = self.slots[0]
            elif name == 'progress':
                device.progress = Mock()
            elif name == 'no-capture':
                device.proposal_capture = None
            elif name == 'projection':
                device.kv_history.projection = object()
            elif name == 'weights':
                device.projection = object()
            elif name == 'parameters':
                device.kv_history.parameters = tuple(dict(name='other') for layer in range(5))
            elif name == 'collectives':
                device.collectives = object()
            elif name == 'poisoned':
                self.fused.segments[2].poisoned = device.kv_history
            elif name == 'scope':
                fused_commit.slide_scope_live.return_value = False
            elif name == 'no-kv':
                device.kv_history = None

        for name in ('ramp', 'slot', 'progress', 'no-capture', 'projection', 'weights', 'parameters', 'collectives',
                     'poisoned', 'scope', 'no-kv'):
            with self.subTest(reason=name):
                self.today.reset_mock()
                self.ops.log.clear()
                device = fused_drafter(self, self.fused, 2)
                refuse(name, device)
                result = self.publish(device, 2, 4)
                self.assertEqual(result, ('today', 4, {'position': 5000}))
                self.today.assert_called_once()
                self.assertEqual(self.ops.log, [])
                self.assertTrue(self.lines[-1].endswith('path=today reason=%s tables=-' % name), self.lines[-1])
                self.fused.segments[2].poisoned = None
                fused_commit.slide_scope_live.return_value = True

    def test_features_and_prefix_refusals(self):
        device = fused_drafter(self, self.fused, 2)
        restore = fused_commit.install_fused_commit(device, self.fused, 2, merge_release=False, fused_steady_state=False)
        try:
            self.assertEqual(device.prepare_publication(taps_for(self, 1), 4, position=5000)[0], 'today')
            self.assertTrue(self.lines[-1].endswith('reason=features tables=-'))
            self.assertEqual(device.prepare_publication(taps_for(self, 2), 17, position=5000)[0], 'today')
            self.assertTrue(self.lines[-1].endswith('reason=prefix tables=-'))
            self.assertEqual(device.prepare_publication(taps_for(self, 2), 4, position=4999)[0], 'today')
            self.assertTrue(self.lines[-1].endswith('reason=pending tables=-'))
        finally:
            restore()

    def test_the_refusal_path_is_install_publish_options_composition(self):
        device = fused_drafter(self, self.fused, 3)
        device.kv_history.history_rows = 1900
        seen = []
        original = fused_commit.FusedCommit.refusal

        def spy(fused, drafter, segment, features, prefix, position):
            seen.append(('kv prepare installed', 'prepare' in vars(drafter.kv_history)))
            return original(fused, drafter, segment, features, prefix, position)

        with patch.object(fused_commit.FusedCommit, 'refusal', spy):
            result = self.publish(device, 3, 5, merge_release=True, fused_steady_state=True)
        self.assertEqual(result, ('today', 5, {'position': 5000, 'merge_release': True, 'fused_steady_state': True}))
        self.assertEqual(seen, [('kv prepare installed', True)])
        self.assertNotIn('prepare', vars(device.kv_history))
        self.assertNotIn('prepare_publication', vars(device))

    def test_parity_takes_today_once_and_its_swap_normalises_it(self):
        device = fused_drafter(self, self.fused, 0, parity=True)
        cache = device.kv_history

        def today(drafter, features, prefix, *, position, **options):
            publication = SimpleNamespace(position=position, prefix=prefix, rows=2048, status='prepared')
            cache.pending = publication
            drafter.pending = SimpleNamespace(position=position, prefix=prefix, rows=2048, history='spare-history',
                                              kv=publication, status='prepared')
            return drafter.pending

        self.today.side_effect = today
        self.publish(device, 0, 6)
        self.assertTrue(self.lines[-1].endswith('reason=parity tables=-'))
        self.assertIs(cache.active[0]['k'], self.slots[0].kv[0]['active']['k'], "today's commit swapped it back")
        self.publish(device, 0, 2)
        self.assertTrue(self.lines[-1].startswith('[PACKED-FUSED] round=3 segment=0 prefix=2 path=fused'))
        self.assertEqual(device.position, 5008)

    def test_out_of_place_runs_todays_transports_from_the_deltas_and_keeps_the_swap(self):
        fused = self.build(inplace=False)
        fused.capture(self.capture)
        self.fused = fused
        device = fused_drafter(self, fused, 2)
        cache = device.kv_history
        active, spare = cache.active, cache.spare
        transports = []
        transport = Mock(side_effect=lambda grid, a, d, s, **options: transports.append((a, d, s, options)) or (lambda: None))
        self.ops.log.clear()
        with patch.object(fused_commit, '_transport', return_value=transport):
            self.publish(device, 2, 9)
        storage = fused.segments[2]
        self.assertEqual(self.ops.log[-1], ('execute_trace', storage.projection_trace, False))
        self.assertEqual(transports, [(active[layer][name], storage.deltas[layer][name], spare[layer][name],
                                       dict(history_rows=2048, prefix=9)) for layer in range(5) for name in HEADS])
        self.assertIs(cache.active, spare)
        self.assertIs(cache.spare, active)
        self.assertEqual(cache.position, 5009)

    def test_a_discard_after_the_in_place_slide_poisons_the_segment(self):
        device = fused_drafter(self, self.fused, 1)
        restore = fused_commit.install_fused_commit(device, self.fused, 1, merge_release=False, fused_steady_state=False)
        try:
            publication = device.prepare_publication(taps_for(self, 1), 4, position=5000)
            device.discard_publication(publication)
        finally:
            restore()
        self.assertIs(self.fused.segments[1].poisoned, device.kv_history)
        self.assertTrue(any(line.startswith(fused_commit.DISCARD_MARKER) for line in self.lines))
        self.assertEqual(self.fused.refusal(device, 1, taps_for(self, 1), 4, 5000), 'poisoned')

    def test_a_poison_refuses_only_its_own_cache_never_the_next_request_through_the_segment(self):
        failed = fused_drafter(self, self.fused, 1)
        self.fused.segments[1].poisoned = failed.kv_history
        self.assertEqual(self.fused.refusal(failed, 1, taps_for(self, 1), 4, 5000), 'poisoned')
        # The failed request's device closed and released the slot; the next request acquired it
        # (the pool zeroed the banks) and its own cache rewrote them.
        joined = fused_drafter(self, self.fused, 1)
        self.assertIsNot(joined.kv_history, failed.kv_history)
        self.assertIsNone(self.fused.refusal(joined, 1, taps_for(self, 1), 4, 5000))
        self.publish(joined, 1, 4)
        self.assertTrue(self.lines[-1].startswith('[PACKED-FUSED] round=3 segment=1 prefix=4 path=fused'))
        self.assertIsNone(self.fused.segments[1].poisoned)

    def test_a_failed_slide_enqueue_poisons_and_raises(self):
        device = fused_drafter(self, self.fused, 1)
        storage = self.fused.segments[1]
        storage.slides[4] = 'broken'
        original = self.ops.execute_trace

        def execute(grid, trace, cq_id=0, blocking=True):
            if trace == 'broken':
                raise RuntimeError('enqueue')
            return original(grid, trace, cq_id=cq_id, blocking=blocking)

        self.ops.execute_trace = execute
        restore = fused_commit.install_fused_commit(device, self.fused, 1, merge_release=False, fused_steady_state=False)
        try:
            with self.assertRaises(RuntimeError):
                device.prepare_publication(taps_for(self, 1), 4, position=5000)
        finally:
            restore()
        self.assertIs(storage.poisoned, device.kv_history)
        self.assertIsNone(device.kv_history.pending)

    def test_round_b1_splits_are_added_when_the_sink_is_set(self):
        from dflash_traced_publish import PUBLICATION_SPLITS

        device = fused_drafter(self, self.fused, 0)
        splits = {}
        token = PUBLICATION_SPLITS.set(splits)
        try:
            self.publish(device, 0, 1)
        finally:
            PUBLICATION_SPLITS.reset(token)
        self.assertEqual(set(splits), {'kv_in', 'kv_exec'})

    def test_install_refuses_to_stack(self):
        device = fused_drafter(self, self.fused, 0)
        restore = fused_commit.install_fused_commit(device, self.fused, 0, merge_release=False, fused_steady_state=False)
        with self.assertRaises(ValueError):
            fused_commit.install_fused_commit(device, self.fused, 0, merge_release=False, fused_steady_state=False)
        restore()
        with self.assertRaises(ValueError):
            fused_commit.install_fused_commit(device, self.fused, 0, merge_release=1, fused_steady_state=False)


class WindowTests(FusedFixture):
    def test_the_window_stages_each_live_segment_for_its_next_frontier_once(self):
        fused = self.built()
        requests = [SimpleNamespace(engine=SimpleNamespace(segment=segment),
                                    session=SimpleNamespace(position=4000 + segment, finished=segment == 3))
                    for segment in range(4)]
        fused.stage_window(requests)
        self.assertEqual([storage.staged_position for storage in fused.segments], [4000, 4001, 4002, None])
        self.assertEqual(len(self.ops.host_copies), 6)
        fused.stage_window(requests)
        self.assertEqual(len(self.ops.host_copies), 6, 'already staged')

    def test_a_failing_segment_is_logged_and_the_others_still_stage(self):
        fused = self.built()
        requests = [SimpleNamespace(engine=SimpleNamespace(segment=segment), session=SimpleNamespace(position=10 + segment))
                    for segment in range(2)]
        requests.insert(0, SimpleNamespace(engine=SimpleNamespace(segment=9), session=SimpleNamespace(position=1)))
        fused.stage_window(requests)
        self.assertEqual([storage.staged_position for storage in fused.segments[:2]], [10, 11])
        self.assertTrue(any('window-tables failed' in line for line in self.lines))

    def test_while_waiting_stages_the_tables_before_the_prestage_and_without_either_flag(self):
        order = []
        block = SimpleNamespace(round_fences=False, rounds=1,
                                prestaged=SimpleNamespace(prestage_requests=lambda requests: order.append('prestage')),
                                fused=SimpleNamespace(stage_window=lambda requests: order.append(('tables', requests))))
        verify_prestage.WhileWaiting(block, ['r'])()
        self.assertEqual(order, [('tables', ['r']), 'prestage'])
        order.clear()
        alone = SimpleNamespace(round_fences=False, prestaged=None,
                                fused=SimpleNamespace(stage_window=lambda requests: order.append('tables')))
        waiting = serving_packed_step.PackedStep(alone).while_waiting(['r'])
        self.assertIsInstance(waiting, verify_prestage.WhileWaiting)
        waiting()
        waiting.fenced()
        self.assertEqual(order, ['tables'])
        self.assertIsNone(serving_packed_step.PackedStep(SimpleNamespace(round_fences=False, prestaged=None))
                          .while_waiting(['r']))

    def test_the_hook_asks_for_a_window_under_the_flag(self):
        from serving_worker_hook import window_flags_on

        self.assertTrue(window_flags_on({'QWEN_FAST_FUSED_COMMIT': '1'}))
        self.assertFalse(window_flags_on({'QWEN_FAST_FUSED_COMMIT': '0'}))


class AuditTests(FusedFixture):
    AUDIT = True

    def setUp(self):
        super().setUp()
        self.fused = self.built()

    def fill(self, tensor, value):
        tensor.value = value.clone()

    def reference_rows(self, prefix, seed):
        generator = torch.Generator().manual_seed(seed)
        return [{name: [torch.randn((1, 4, prefix, 128), generator=generator).bfloat16()] * 2 for name in HEADS}
                for layer in range(5)]

    def run_audit(self, device, prefix, reference, *, break_bank=None, break_delta=None):
        storage = self.fused.segments[device.pool_slot.index]
        order = []

        def eager(drafter, features, prefix_, position, *, slide):
            order.append(('reference', slide))
            generator = torch.Generator().manual_seed(1)
            for layer in range(5):
                for name in HEADS:
                    bank = torch.randn(KV_SHAPE, generator=generator).bfloat16()
                    self.fill(drafter.kv_history.spare[layer][name], bank)
                    self.fill(drafter.kv_history.active[layer][name], bank)
                    delta = torch.zeros(DELTA_SHAPE, dtype=torch.bfloat16)
                    delta[..., :prefix_, :] = reference[layer][name][0]
                    self.fill(storage.deltas[layer][name], delta)
            if break_bank is not None:
                layer, name = break_bank
                broken = drafter.kv_history.active[layer][name].value.clone()
                broken[0, 0, 2047, 0] += 1
                self.fill(drafter.kv_history.active[layer][name], broken)
            if break_delta is not None:
                layer, name = break_delta
                broken = storage.deltas[layer][name].value.clone()
                broken[0, 1, 0, 3] += 1
                self.fill(storage.deltas[layer][name], broken)
            return reference

        restore = fused_commit.install_fused_commit(device, self.fused, device.pool_slot.index, merge_release=False,
                                                    fused_steady_state=False)
        try:
            with patch.object(self.fused, 'eager_reference', Mock(side_effect=eager)):
                publication = device.prepare_publication(taps_for(self, device.pool_slot.index), prefix,
                                                         position=device.position)
                device.commit_publication(publication)
        finally:
            restore()
        return order

    def test_the_reference_runs_before_the_fused_launch_and_an_exact_round_logs_zero(self):
        device = fused_drafter(self, self.fused, 1)
        self.ops.log.clear()
        order = self.run_audit(device, 6, self.reference_rows(6, 5))
        self.assertEqual(order, [('reference', True)])
        executes = [entry for entry in self.ops.log if entry[0] in ('execute_trace', 'sync')]
        self.assertEqual([entry[0] for entry in executes], ['execute_trace', 'execute_trace', 'sync'])
        audit = [line for line in self.lines if line.startswith(fused_commit.AUDIT_MARKER)]
        self.assertEqual(audit, ['[PACKED-FUSED-AUDIT] round=3 segment=1 prefix=6 mode=inplace checked=40 mismatches=0'])
        self.assertEqual(self.fused.counts['mismatches'], 0)

    def test_a_bank_mismatch_is_logged_and_repaired_from_the_spare(self):
        device = fused_drafter(self, self.fused, 2)
        self.run_audit(device, 4, self.reference_rows(4, 7), break_bank=(3, 'v'))
        cache = device.kv_history
        self.assertTrue(torch.equal(cache.active[3]['v'].value, cache.spare[3]['v'].value))
        self.assertTrue(any(line.startswith(fused_commit.AUDIT_MISMATCH_MARKER + ' round=3 segment=2 prefix=4 at=bank3v.0')
                            for line in self.lines), self.lines)
        self.assertEqual(self.fused.counts['mismatches'], 2)

    def test_a_delta_mismatch_is_the_e2_claim_failing(self):
        device = fused_drafter(self, self.fused, 0)
        self.run_audit(device, 3, self.reference_rows(3, 9), break_delta=(0, 'k'))
        audit = [line for line in self.lines if line.startswith(fused_commit.AUDIT_MARKER)]
        self.assertTrue(audit[-1].endswith('mismatches=2'))

    def test_out_of_place_audits_the_deltas_and_reruns_today_on_a_mismatch(self):
        fused = self.build(inplace=False)
        fused.capture(self.capture)
        self.fused = fused
        device = fused_drafter(self, fused, 1)
        with patch.object(fused_commit, '_transport', return_value=Mock(return_value=lambda: None)):
            order = self.run_audit(device, 5, self.reference_rows(5, 3), break_delta=(4, 'v'))
        self.assertEqual(order, [('reference', False), ('reference', True)])
        audit = [line for line in self.lines if line.startswith(fused_commit.AUDIT_MARKER)]
        self.assertEqual(audit, ['[PACKED-FUSED-AUDIT] round=3 segment=1 prefix=5 mode=oop checked=20 mismatches=2'])


# ------------------------------------------------------------------------------------------------------
# T_proj IS today's op sequence at count 16
# ------------------------------------------------------------------------------------------------------

class ShapeTensor:
    def __init__(self, ops, shape, dtype):
        self.shape, self.dtype, self.layout = tuple(shape), dtype, 'tile'
        self.shards = [tpv.FakeShard(self, next(ops.addresses)) for chip in range(2)]

    def memory_config(self):
        return 'dram'


class ShapeOps:
    """Every op the feature and K/V projections dispatch, with its output's shape, logged by name,
    argument shapes and plain keyword values - so two runs compare call for call."""

    bfloat16, float32, uint32 = 'bf16', 'f32', 'u32'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'tile', 'row', 'dram'
    Topology = SimpleNamespace(Linear='linear')
    MathFidelity = SimpleNamespace(HiFi4='hifi4')

    def __init__(self):
        self.log, self.addresses = [], count(0x200000)
        self.experimental = SimpleNamespace(all_gather_async=self.all_gather_async,
                                            nlp_create_qkv_heads=self.nlp_create_qkv_heads,
                                            rotary_embedding_hf=self.rotary_embedding_hf)

    def tensor(self, shape, dtype='bf16'):
        return ShapeTensor(self, shape, dtype)

    def describe(self, value):
        if isinstance(value, ShapeTensor):
            return ('tensor', value.shape, value.dtype)
        if isinstance(value, (list, tuple)):
            return tuple(self.describe(item) for item in value)
        if isinstance(value, torch.Tensor):
            return ('host', tuple(value.shape), str(value.dtype), value.float().sum().item())
        if isinstance(value, (int, float, str, type(None))):
            return value
        return type(value).__name__

    def record(self, name, *args, **kwargs):
        self.log.append((name, self.describe(args), tuple(sorted((key, self.describe(value))
                                                                 for key, value in kwargs.items()))))

    def get_device_tensors(self, value):
        return value.shards

    def WormholeComputeKernelConfig(self, **fields):
        return ('kernel-config',) + tuple(sorted(fields.items()))

    def MatmulMultiCoreReuseMultiCast1DProgramConfig(self, **fields):
        return ('program',) + tuple(sorted(fields.items()))

    def ReplicateTensorToMesh(self, grid):
        return 'replicate'

    def slice(self, value, start, end):
        self.record('slice', value, tuple(start), tuple(end))
        return self.tensor(tuple(last - first for first, last in zip(start, end)), value.dtype)

    def pad(self, value, padding, fill):
        self.record('pad', value, tuple(map(tuple, padding)), fill)
        return self.tensor(tuple(size + before + after for size, (before, after) in zip(value.shape, padding)), value.dtype)

    def concat(self, values, dim, **kwargs):
        self.record('concat', list(values), dim, **kwargs)
        shape = list(values[0].shape)
        shape[dim] = sum(value.shape[dim] for value in values)
        return self.tensor(shape, values[0].dtype)

    def matmul(self, left, right, **kwargs):
        self.record('matmul', left, right, **kwargs)
        return self.tensor(left.shape[:-1] + (right.shape[-1],), kwargs.get('dtype', left.dtype))

    def all_gather_async(self, value, **kwargs):
        self.record('all_gather_async', value, dim=kwargs['dim'], num_links=kwargs['num_links'])
        return self.tensor((value.shape[0] * 2,) + value.shape[1:], value.dtype)

    def add(self, left, right, **kwargs):
        self.record('add', left, right, **kwargs)
        return self.tensor(left.shape, kwargs.get('dtype', left.dtype))

    def typecast(self, value, dtype):
        self.record('typecast', value, dtype)
        return self.tensor(value.shape, dtype)

    def rms_norm(self, value, **kwargs):
        self.record('rms_norm', value, **kwargs)
        return self.tensor(value.shape, value.dtype)

    def nlp_create_qkv_heads(self, query, kv, **kwargs):
        self.record('nlp_create_qkv_heads', query, kv, **kwargs)
        rows = kv.shape[2]
        return (self.tensor((1, 16, rows, 128)), self.tensor((1, 4, rows, 128)), self.tensor((1, 4, rows, 128)))

    def rotary_embedding_hf(self, value, cos, sin, **kwargs):
        self.record('rotary_embedding_hf', value, cos, sin, **kwargs)
        return self.tensor(value.shape, value.dtype)

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        self.record('from_torch', value, device is not None, dtype, layout)
        return self.tensor(tuple(value.shape), dtype)

    def copy(self, source, destination):
        self.record('copy', source, destination)

    def deallocate(self, value):
        pass


class OpSequenceTests(unittest.TestCase):
    """T_proj's captured body against today's publication at prefix 16, over one ShapeOps."""

    def fixture(self):
        ops = ShapeOps()
        grid = SimpleNamespace(shape=[1, 2], compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        collectives = SimpleNamespace(get_and_cycle_ag_semaphore_handles=lambda: 'ag',
                                      get_and_cycle_barrier_semaphore_handle=lambda: 'barrier')
        parameters = tuple(dict(operations=ops, native_head_layout=True, kernel='kv-kernel',
                                projections=dict(k=ops.tensor((5120, 512)), v=ops.tensor((5120, 512))),
                                head_norms=dict(k=ops.tensor((1, 1, 4, 32)))) for layer in range(5))
        weights = SimpleNamespace(layers=[(parameter, 'mlp', 'weights', 'conv') for parameter in parameters],
                                  projection=ops.tensor((10240, 5120)), feature_norm=ops.tensor((1, 1, 160, 32)),
                                  tensors=[])
        taps = tuple(ops.tensor((1, 1, 64, 2560)) for tap in range(5))
        slots = [SimpleNamespace(index=index, query=ops.tensor((1, 1, 32, 2048)),
                                 kv=[{side: {name: ops.tensor(KV_SHAPE) for name in HEADS} for side in ('active', 'spare')}
                                     for layer in range(5)]) for index in range(4)]
        block = SimpleNamespace(rows_per_user=16, users=4, shape=m3_shape(PAGE_WIDTH), segment_slots=slots, taps=taps,
                                rounds=0)
        return ops, grid, collectives, weights, parameters, taps, slots, block

    def test_t_proj_logs_todays_publication_at_prefix_sixteen_call_for_call(self):
        ops, grid, collectives, weights, parameters, taps, slots, block = self.fixture()
        with patch.object(fused_commit, 'slide_scope_live', return_value=True), \
                patch('feature_collective.projection_links', return_value=4):
            fused = fused_commit.FusedCommit(block, operations=ops, mesh=grid, pool=None, shared_weights=weights,
                                             collectives=collectives, inplace=True, audit=False)
            segment, position = 2, 131000
            storage = fused.segments[segment]
            storage.taps = taps
            ops.log.clear()
            fused.project(storage, [])
            fused_log = [entry for entry in ops.log if entry[0] != 'copy']
            copies = [entry for entry in ops.log if entry[0] == 'copy']
            ops.log.clear()
            # Today: prepare_publication's projection, then DraftKVHistory.prepare's inputs and
            # projections, at the same segment's taps with prefix 16.
            device = SimpleNamespace(operations=ops, mesh=grid, collectives=collectives, projection=weights.projection,
                                     feature_norm=weights.feature_norm, kernel=fused.kernel)
            retain = lambda value: value
            features = PackedFeatureTaps(taps, row_offset=16 * segment, rows=16)
            projected = DFlashDevice.project_features(device, features, 16, retain=retain)
            cache = SimpleNamespace(operations=ops, mesh=grid)
            cache.upload = lambda value: DraftKVHistory.upload(cache, value)
            inputs, tables = DraftKVHistory.project_inputs(cache, projected, 16, position, retain)
            from draft_kv_projection import project_key_value

            for parameter in parameters:
                project_key_value(ops, inputs, slots[segment].query, tables, retain, parameters=parameter)
            today_log = list(ops.log)
        uploads = [entry for entry in today_log if entry[0] == 'from_torch']
        self.assertEqual(len(uploads), 2)
        self.assertEqual([entry for entry in today_log if entry[0] != 'from_torch'], fused_log)
        self.assertGreater(len(fused_log), 60)
        self.assertEqual(len(copies), 10)
        # The uploads are the bytes the window stages.
        for upload, table in zip(uploads, fused_commit.host_tables(position, 32, 16)):
            self.assertEqual(upload[1][0], ('host', (1, 1, 32, 128), 'torch.bfloat16', table.float().sum().item()))

    def test_the_rows_past_the_prefix_are_the_only_difference_from_a_short_prefix(self):
        """At prefix 5 today slices five rows and pads 27 where T_proj slices sixteen and pads sixteen;
        every op after the first pad has the same shapes - the row-independence the audit proves."""
        ops, grid, collectives, weights, parameters, taps, slots, block = self.fixture()
        device = SimpleNamespace(operations=ops, mesh=grid, collectives=collectives, projection=weights.projection,
                                 feature_norm=weights.feature_norm, kernel='kernel')
        logs = []
        with patch('feature_collective.projection_links', return_value=4):
            for count_ in (5, 16):
                ops.log.clear()
                DFlashDevice.project_features(device, PackedFeatureTaps(taps, row_offset=32, rows=16), count_,
                                              retain=lambda value: value)
                logs.append(list(ops.log))
        short, full = logs
        self.assertEqual([entry[0] for entry in short], [entry[0] for entry in full])
        differing = [index for index, (left, right) in enumerate(zip(short, full)) if left != right]
        self.assertEqual({short[index][0] for index in differing}, {'slice', 'pad'})
        self.assertEqual(short[-1][1][2], (1, 1, 5, 5120))


# ------------------------------------------------------------------------------------------------------
# The real packed block: R2 and the capture order
# ------------------------------------------------------------------------------------------------------

class BlockTests(tpv.FourUserFixture):
    def setUp(self):
        with patch.object(tpv, 'FakeTTNN', FusedOps):
            super().setUp()
        for slot in self.pool.slots:
            extra = pooled_slot(self.ttnn, slot.index)
            slot.kv, slot.query = extra.kv, extra.query
        self.weights = shared_weights(self.ttnn)
        self.collectives = SimpleNamespace(name='shared-ccl')
        self.captures = []

        def capture(operations, grid, operation):
            name = 'commit' if isinstance(operation, Mock) else operation.__qualname__
            self.captures.append((name, len(self.ttnn.device_uploads)))
            return 'trace%d' % next(self.traces), operation()

        fixture = FusedFixture('run')

        def project_features(owner, features, count_, offset, retain):
            return retain(self.ttnn.allocate((1, 1, count_, 5120)))

        def project_key_value(operations, inputs, query, tables, retain, **options):
            return dict(q=retain(self.ttnn.allocate((1, 16, 32, 128))), k=retain(self.ttnn.allocate(DELTA_SHAPE)),
                        v=retain(self.ttnn.allocate(DELTA_SHAPE)))

        for target, value in (('slide_scope_live', Mock(return_value=True)),
                              ('slide_program', Mock(side_effect=fixture.program)),
                              ('_project_features', Mock(side_effect=project_features)),
                              ('_project_key_value', Mock(side_effect=project_key_value))):
            patcher = patch.object(fused_commit, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(packed_verifier, 'capture_operation', Mock(side_effect=capture))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.lines, loguru = logged()
        loguru.start()
        self.addCleanup(loguru.stop)

    def test_r2_every_fused_buffer_predates_the_verify_capture_and_the_traces_follow_the_commits(self):
        with clean_environment(QWEN_FAST_FUSED_COMMIT='1', QWEN_FAST_FUSED_COMMIT_INPLACE='1'):
            block = self.build(collectives=self.collectives)
        self.assertIsNotNone(block.fused)
        names = [name for name, uploads in self.captures]
        verify = names.index('PackedVerifierEngine._capture_traces.<locals>.<lambda>')
        self.assertEqual(verify, 0)
        uploads_at_verify = self.captures[verify][1]
        position = {id(value): index for index, value in enumerate(self.ttnn.device_uploads)}
        self.assertTrue(all(position[id(value)] < uploads_at_verify for value in block.fused.allocated()))
        commits = [index for index, name in enumerate(names) if name == 'commit']
        fused = [index for index, name in enumerate(names) if 'FusedCommit.capture' in name]
        self.assertEqual((len(commits), len(fused)), (64, 68))
        self.assertLess(max(commits), min(fused))
        self.assertTrue(any(line.startswith(fused_commit.ENGAGED_MARKER + ' users=4 rows=16 inplace=1 live_banks=0 '
                                            'audit=0 kernel=') and 'traces=68 layout=2x80' in line for line in self.lines),
                        self.lines)
        self.assertEqual(block.describe()['fused_commit']['traces'], 68)
        released = len(self.ttnn.released)
        block.close()
        self.assertEqual(len(self.ttnn.released) - released, 1 + 64 + 68)
        self.assertIsNone(block.fused)

    def test_a_refused_fused_commit_serves_today_and_says_why(self):
        with clean_environment(QWEN_FAST_FUSED_COMMIT='1'):
            block = self.build()
        self.assertIsNone(block.fused)
        self.assertTrue(any(line.startswith(fused_commit.REFUSED_MARKER) for line in self.lines))
        self.assertEqual(len(self.captures), 1 + 64)

    def test_without_the_flag_nothing_is_built_or_imported(self):
        with clean_environment(), patch.object(fused_commit, 'FusedCommit', Mock(side_effect=AssertionError)):
            block = self.build(collectives=self.collectives)
        self.assertIsNone(block.fused)
        self.assertEqual(len(self.captures), 1 + 64)
        self.assertNotIn('fused_commit', block.describe())

    def test_a_round_through_the_step_publishes_through_the_fused_override(self):
        with clean_environment(QWEN_FAST_FUSED_COMMIT='1', QWEN_FAST_FUSED_COMMIT_INPLACE='1'):
            block = self.build(collectives=self.collectives)
        installed = []

        def install(drafter, fused, segment, **options):
            installed.append((drafter, fused, segment, options))
            return Mock()

        drafter = SimpleNamespace(name='drafter')
        request = SimpleNamespace(session=SimpleNamespace(commit=Mock(return_value=SimpleNamespace(emitted=(1,), state_rows=1)),
                                                          position=10, finished=False, abort=Mock()),
                                  engine=SimpleNamespace(adopt_packed=Mock()),
                                  runtime=SimpleNamespace(drafter=drafter, publish=Mock()), collect_timings=False)
        entry = dict(request=request, ticket=SimpleNamespace(position=9), request_id='r')
        with clean_environment(QWEN_FAST_PIPELINED_PUBLISH='1', QWEN_FAST_TRACED_PUBLISH='1'), \
                patch('fused_commit.install_fused_commit', Mock(side_effect=install)), \
                patch('dflash_traced_publish.install_publish_options', Mock(side_effect=AssertionError)):
            serving_packed_step.commit_entry(entry, block, 2, [1], cancelled=lambda: False, metrics={},
                                             verify_started=0.0, verified=0.0)
        self.assertEqual(installed, [(drafter, block.fused, 2, dict(merge_release=True, fused_steady_state=True))])


# ------------------------------------------------------------------------------------------------------
# F4: the pair reads the live banks
# ------------------------------------------------------------------------------------------------------

class LiveBankTests(unittest.TestCase):
    def devices(self, ops, *, parity=(False, False)):
        from test_dflash_proposal_trace import FakeTensor, pair_devices

        devices = pair_devices(ops, context=2048)
        devices[1].position = 5000
        for device, odd in zip(devices, parity):
            slot = device.pool_slot
            slot.kv = [{side: {name: FakeTensor(torch.zeros(1), 'bf16', 'tile', True, ops.addresses) for name in HEADS}
                        for side in ('active', 'spare')} for layer in range(5)]
            active = [dict(layer['active']) for layer in slot.kv]
            spare = [dict(layer['spare']) for layer in slot.kv]
            device.kv_history.active = spare if odd else active
            device.kv_history.borrowed = [value for layer in slot.kv for side in layer.values() for value in side.values()]
        return devices

    def scenario(self, environ, parity=(False, False)):
        from test_dflash_proposal_trace import RecordingOps

        ops = RecordingOps()
        device_a, device_b = self.devices(ops, parity=parity)
        lines, loguru = logged()
        with clean_environment(**environ), loguru, \
                patch('dflash_packed_proposal.select_device_outputs', return_value=((1, 2, 3), (4, 5))), \
                patch.object(fused_commit, '_LIVE_NOTED', []):
            import dflash_proposal_trace

            trace = dflash_proposal_trace.PreparedPackedDFlashProposal(device_a, device_b)
            trace.prepare_device(11, 22)
            trace.finish('a', 2)
            trace.finish('b', 1)
            bucket = trace.buckets[(2048, 2048)]
            copies = [event for event in ops.events if event[0] == 'copy']
            uploads = [event for event in ops.events if event[0] == 'from_torch' and event[2]]
            trace.close()
        return bucket, copies, uploads, lines, (device_a, device_b)

    ON = dict(QWEN_FAST_FUSED_COMMIT='1', QWEN_FAST_FUSED_COMMIT_INPLACE='1', QWEN_FAST_FUSED_COMMIT_LIVE_BANKS='1')

    def test_the_pair_binds_the_pools_active_banks_and_copies_nothing(self):
        bucket, copies, uploads, lines, devices = self.scenario(self.ON)
        self.assertTrue(bucket.live_banks)
        for cache, device in zip(bucket.cached_history, devices):
            self.assertEqual([[layer[name] for name in HEADS] for layer in cache],
                             [[layer['active'][name] for name in HEADS] for layer in device.pool_slot.kv])
        self.assertEqual(copies, [])
        self.assertTrue(any(line.startswith(fused_commit.LIVE_BANKS_MARKER + ' engaged pair=[0, 1]') for line in lines))
        today = self.scenario({})[2]
        self.assertEqual(len(today) - len(uploads), 2 * 5 * 2, 'no cached_history placeholders')

    def test_a_device_on_the_spare_side_is_copied_into_the_pool_active_bank(self):
        bucket, copies, uploads, lines, devices = self.scenario(self.ON, parity=(False, True))
        self.assertEqual(len(copies), 2 * 10, 'the capture update and the round update, device b only')
        self.assertTrue(any(line.startswith(fused_commit.LIVE_BANKS_NORMALISED + ' pair=[0, 1] normalised=10')
                            for line in lines))

    def test_without_in_place_the_sub_flag_binds_nothing(self):
        bucket = self.scenario(dict(QWEN_FAST_FUSED_COMMIT='1', QWEN_FAST_FUSED_COMMIT_LIVE_BANKS='1'))[0]
        self.assertFalse(getattr(bucket, 'live_banks', False))


# ------------------------------------------------------------------------------------------------------
# The gate and the arm
# ------------------------------------------------------------------------------------------------------

class GateTests(unittest.TestCase):
    ON = {'QWEN_FAST_FUSED_COMMIT': '1', 'QWEN_FAST_FUSED_COMMIT_INPLACE': '1', 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS': '1',
          'QWEN_FAST_FUSED_COMMIT_AUDIT': '1'}

    def log(self, rounds=6, ramp=(1,), unexpected=None, mismatch=None):
        lines = ['[PINDIAG] fused commit engaged users=4 rows=16 inplace=1 live_banks=1 audit=1 kernel=direct '
                 'traces=68 layout=2x80', '[PINDIAG] pair live banks engaged pair=[0, 1] context=(2048, 2048)']
        for number in range(1, rounds + 1):
            for segment in range(4):
                if number in ramp:
                    reason = 'ramp'
                elif number == 2 and segment == 1:
                    reason = 'parity'
                elif (number, segment) == unexpected:
                    reason = 'weights'
                else:
                    reason = None
                path = 'today' if reason else 'fused'
                lines.append('[PACKED-FUSED] round=%d segment=%d prefix=%d path=%s reason=%s tables=%s' % (
                    number, segment, 3 + segment, path, reason or '-', '-' if reason else 'window'))
                if path == 'fused':
                    lines.append('[PACKED-FUSED-AUDIT] round=%d segment=%d prefix=%d mode=inplace checked=40 '
                                 'mismatches=%d' % (number, segment, 3 + segment, 2 if (number, segment) == mismatch else 0))
        return chr(10).join(lines)

    def test_a_clean_arm_passes_and_is_summarised(self):
        import lever_n_m3native_gate as gate

        report = gate.flag_marker_report(self.ON, 4, self.log())
        self.assertEqual(report['missing'], [])
        summary = report['round_fence_h1b']
        self.assertEqual((summary['publications'], summary['fused'], summary['today']), (24, 19, 5))
        self.assertEqual(summary['today_reasons'], {'ramp': 4, 'parity': 1})
        self.assertEqual((summary['rounds'], summary['fused_rounds'], summary['four_fused_rounds']), (6, 4, 4))
        self.assertEqual((summary['audits'], summary['audit_mismatches']), (19, 0))
        self.assertEqual(summary['engaged']['traces'], 68)

    def test_missing_lines_an_unexpected_refusal_a_mismatch_or_a_lone_sub_flag_fail(self):
        import lever_n_m3native_gate as gate

        missing = gate.flag_marker_report(self.ON, 4, '')['missing']
        for marker in (gate.FUSED_ENGAGED_MARKER, gate.FUSED_MARKER, gate.FUSED_AUDIT_MARKER,
                       gate.FUSED_LIVE_BANKS_MARKER):
            self.assertTrue(any(marker in line for line in missing), marker)
        odd = gate.flag_marker_report(self.ON, 4, self.log(unexpected=(4, 2)))['missing']
        self.assertTrue(any("today's path for reasons" in line and 'weights' in line for line in odd), odd)
        bad = gate.flag_marker_report(self.ON, 4, self.log(mismatch=(5, 3)))['missing']
        self.assertTrue(any('no mismatch' in line for line in bad), bad)
        refused = gate.flag_marker_report(self.ON, 4, self.log() + chr(10) +
                                          '[PINDIAG] fused commit refused users=4 reason=x')['missing']
        self.assertTrue(any('refused the fused commit' in line for line in refused))
        lone = gate.flag_marker_report({'QWEN_FAST_FUSED_COMMIT_AUDIT': '1'}, 4, '')['missing']
        self.assertTrue(any('does nothing' in line for line in lone))
        no_inplace = gate.flag_marker_report({'QWEN_FAST_FUSED_COMMIT': '1', 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS': '1'},
                                             4, self.log())['missing']
        self.assertTrue(any('the live bank moves' in line for line in no_inplace))
        discarded = gate.flag_marker_report(self.ON, 4, self.log() + chr(10) + fused_commit.DISCARD_MARKER +
                                            ' segment=2 position=5000 prefix=4')
        self.assertEqual(discarded['round_fence_h1b']['discards'], 1)
        self.assertTrue(any('discarded after the slide' in line and 'segment=2' in line
                            for line in discarded['missing']), discarded['missing'])
        self.assertEqual(gate.flag_marker_report(self.ON, 4, self.log())['round_fence_h1b']['discards'], 0)

    def test_the_gate_reads_the_lines_the_module_writes(self):
        import lever_n_m3native_gate as gate

        self.assertEqual((gate.FUSED_ENGAGED_MARKER, gate.FUSED_REFUSED_MARKER, gate.FUSED_AUDIT_MISMATCH_MARKER,
                          gate.FUSED_DISCARD_MARKER),
                         (fused_commit.ENGAGED_MARKER, fused_commit.REFUSED_MARKER, fused_commit.AUDIT_MISMATCH_MARKER,
                          fused_commit.DISCARD_MARKER))
        self.assertEqual(gate.FUSED_MARKER, fused_commit.MARKER + ' round=')
        self.assertEqual(gate.FUSED_AUDIT_MARKER, fused_commit.AUDIT_MARKER + ' round=')
        self.assertTrue(gate.FUSED_LIVE_BANKS_MARKER.startswith(fused_commit.LIVE_BANKS_MARKER))
        self.assertEqual(gate.FUSED_EXPECTED_REFUSALS, fused_commit.EXPECTED_REFUSALS)


class ArmTests(unittest.TestCase):
    ARM = HERE / 'lever_n_m3native_run_arm.sh'
    START = '# Round-fence plan H1b (fused_commit.py; every flag default off).'

    def text(self):
        return self.ARM.read_text(encoding='utf-8')

    def validate(self, **environ):
        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        text = self.text()
        start = text.index(self.START)
        end = text.index(chr(10) + 'fi' + chr(10), text.index('if [ -n "${M3NATIVE_FUSED_COMMIT:-}" ]; then', start)) + 4
        script = 'set -euo pipefail' + chr(10) + 'users="${USERS_UNDER_TEST}"' + chr(10) + text[start:end] + 'echo VALID' + chr(10)
        try:
            return subprocess.run([bash, '-c', script], capture_output=True, text=True, timeout=60,
                                  env=dict(PATH=os.environ.get('PATH', ''), USERS_UNDER_TEST=environ.pop('users', '4'),
                                           **environ))
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)

    BASE = dict(M3NATIVE_TRACED_PROPOSAL='1', M3NATIVE_PAIR_MASK_REFRESH='1', M3NATIVE_PACKED_PROPOSAL='1')

    def test_the_arm_refuses_what_the_gate_would(self):
        for environ in ({}, dict(self.BASE, M3NATIVE_FUSED_COMMIT='1'),
                        dict(self.BASE, M3NATIVE_FUSED_COMMIT='1', M3NATIVE_FUSED_COMMIT_INPLACE='1',
                             M3NATIVE_FUSED_COMMIT_LIVE_BANKS='1', M3NATIVE_FUSED_COMMIT_AUDIT='1')):
            with self.subTest(accepted=environ):
                result = self.validate(**environ)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('VALID', result.stdout)
        for environ, message in (
                (dict(M3NATIVE_FUSED_COMMIT='yes'), 'must be 1 or unset'),
                (dict(M3NATIVE_FUSED_COMMIT_AUDIT='1'), 'without M3NATIVE_FUSED_COMMIT=1'),
                (dict(self.BASE, M3NATIVE_FUSED_COMMIT='1', M3NATIVE_FUSED_COMMIT_LIVE_BANKS='1'), 'needs M3NATIVE_FUSED_COMMIT_INPLACE=1'),
                (dict(M3NATIVE_FUSED_COMMIT='1', M3NATIVE_PAIR_MASK_REFRESH='1'), 'needs M3NATIVE_TRACED_PROPOSAL=1'),
                (dict(self.BASE, M3NATIVE_FUSED_COMMIT='1', users='1'), 'serves the packed block only'),
                (dict(M3NATIVE_TRACED_PROPOSAL='1', M3NATIVE_PAIR_MASK_REFRESH='1', M3NATIVE_FUSED_COMMIT='1',
                      M3NATIVE_FUSED_COMMIT_INPLACE='1', M3NATIVE_FUSED_COMMIT_LIVE_BANKS='1'), 'needs M3NATIVE_PACKED_PROPOSAL=1')):
            with self.subTest(refused=environ):
                result = self.validate(**environ)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_the_four_switches_cross_before_the_entrypoint(self):
        text = self.text()
        entry = text.index('--entrypoint python3')
        for name in ('FUSED_COMMIT', 'FUSED_COMMIT_INPLACE', 'FUSED_COMMIT_LIVE_BANKS', 'FUSED_COMMIT_AUDIT'):
            line = '${M3NATIVE_%s:+-e QWEN_FAST_%s=1}' % (name, name)
            with self.subTest(name=name):
                self.assertEqual(text.count(line), 1)
                self.assertLess(text.index(line), entry)
        self.assertLess(text.index('${M3NATIVE_ROUND_FENCES:+-e QWEN_FAST_ROUND_FENCES=1}'),
                        text.index('${M3NATIVE_FUSED_COMMIT:+-e QWEN_FAST_FUSED_COMMIT=1}'))


# ------------------------------------------------------------------------------------------------------
# Flag off: the PARENT, call for call
# ------------------------------------------------------------------------------------------------------

class ParentTests(unittest.TestCase):
    """With every flag off, each module H1b touches against its PARENT copy: H1a's and M2's parent
    comparisons re-pointed at PARENT, plus the pair trace, the window, the commit and the attach."""

    def repointed(self, module, cls, method):
        if parent_module('packed_verifier.py') is None:
            self.skipTest('no git history for %s' % PARENT)
        result = unittest.TestResult()
        with patch.object(module, 'PARENT', PARENT):
            getattr(module, cls)(method).run(result)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertEqual(result.skipped, [])

    def test_the_steps_answers_and_rounds_are_the_parents(self):
        import test_verify_prestage

        self.repointed(test_verify_prestage, 'ParentTests', 'test_the_steps_answers_and_rounds_are_the_parents')

    def test_the_real_blocks_round_is_the_parents(self):
        import test_verify_prestage

        self.repointed(test_verify_prestage, 'ParentTests', 'test_the_real_blocks_round_is_the_parents')

    def test_the_verifier_rounds_are_the_parents_call_for_call(self):
        import test_verify_prestage

        self.repointed(test_verify_prestage, 'ParentTests', 'test_the_verifier_rounds_are_the_parents_call_for_call')

    def test_the_packed_blocks_fenced_rounds_are_the_parents(self):
        import test_round_fences

        self.repointed(test_round_fences, 'BlockParentTests', 'test_the_rounds_are_the_parents')

    def test_the_hooks_drafts_and_pass_throughs_are_the_parents(self):
        import test_verify_prestage

        self.repointed(test_verify_prestage, 'PlumbingTests', 'test_flag_off_the_hooks_drafts_are_the_parents_call_for_call')
        self.repointed(test_verify_prestage, 'PlumbingTests', 'test_flag_off_the_hooks_pass_throughs_are_the_parents')

    def test_the_gates_report_is_the_parents(self):
        import test_verify_prestage

        self.repointed(test_verify_prestage, 'GateTests', 'test_flag_off_the_report_is_the_parents')

    def test_the_pair_trace_is_the_parents_call_for_call(self):
        from test_dflash_proposal_trace import RecordingOps, pair_devices, pair_scenario, pinned_module

        parent = pinned_module('dflash_proposal_trace.py', 'dflash_proposal_trace_h1b_parent', commit=PARENT)
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        import dflash_proposal_trace

        for context in (300, 2048):
            for flags in ({}, {'QWEN_FAST_ROUND_B1': '1'}, {'QWEN_FAST_PAIR_MASK_REFRESH': '1'},
                          {'QWEN_FAST_FUSED_COMMIT': '1', 'QWEN_FAST_FUSED_COMMIT_INPLACE': '1'}):
                def devices(ops, context_=context, **options):
                    made = pair_devices(ops, context=context_)
                    made[1].position = max(made[1].position, context_ + 100)
                    return made

                with self.subTest(context=context, flags=flags), clean_environment(**flags), \
                        patch('test_dflash_proposal_trace.pair_devices', devices):
                    _, before = pair_scenario(parent, RecordingOps())
                    _, today = pair_scenario(dflash_proposal_trace, RecordingOps())
                    self.assertGreater(len(before), 50)
                    self.assertEqual(today, before)

    def test_the_window_is_the_parents(self):
        parent = parent_module('verify_prestage.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        for fences in (False, True):
            records = []
            for module in (verify_prestage, parent):
                calls = []
                block = SimpleNamespace(round_fences=fences, rounds=2, fence_token=lambda: calls.append('token') or 7,
                                        note_round_fence=lambda token: calls.append(('armed', token)),
                                        prestaged=SimpleNamespace(prestage_requests=lambda requests: calls.append(
                                            ('prestage', requests)), drop=lambda reason: calls.append(('drop', reason))))
                waiting = module.WhileWaiting(block, ['r'])
                waiting()
                waiting.drop(RuntimeError('x'))
                waiting.fenced()
                records.append(calls)
            self.assertEqual(records[0], records[1])

    def test_the_commit_is_the_parents(self):
        parent = parent_module('serving_packed_step.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)

        def run(module, environ):
            installed = []
            request = SimpleNamespace(session=SimpleNamespace(commit=Mock(return_value=SimpleNamespace(emitted=(1,), state_rows=1)),
                                                              position=10, finished=False, abort=Mock()),
                                      engine=SimpleNamespace(adopt_packed=Mock()),
                                      runtime=SimpleNamespace(drafter=SimpleNamespace(), publish=Mock()),
                                      collect_timings=False)
            entry = dict(request=request, ticket=SimpleNamespace(position=9), request_id='r')
            block = SimpleNamespace(rounds=1)
            with clean_environment(**environ), patch('dflash_traced_publish.install_publish_options',
                                                     Mock(side_effect=lambda *a, **k: installed.append(k) or Mock())):
                output = module.commit_entry(entry, block, 1, [1], cancelled=lambda: False, metrics={},
                                             verify_started=0.0, verified=0.0)
            return output, installed, [(call.args[:3], call.args[3] is request.runtime.publish)
                                       for call in request.session.commit.call_args_list]

        for environ in ({}, {'QWEN_FAST_PIPELINED_PUBLISH': '1'}, {'QWEN_FAST_PIPELINED_PUBLISH': '1',
                                                                   'QWEN_FAST_TRACED_PUBLISH': '1'}):
            with self.subTest(environ=environ):
                self.assertEqual(run(serving_packed_step, environ), run(parent, environ))

    def test_the_attach_adds_only_the_flagged_collectives(self):
        result = subprocess.run(['git', 'show', '%s:scripts/ci/serving_runtime.py' % PARENT], capture_output=True,
                                cwd=str(HERE), timeout=60)
        if result.returncode != 0:
            self.skipTest('no git history for %s' % PARENT)
        before = result.stdout.decode('utf-8').splitlines()
        after = without_trace_census(without_levern(without_diag_trim(without_prefill_scratch(without_any_request(without_sticky(
            without_solo_and_lanes(without_m3_blocks(without_request_warm(without_parked_engines((HERE / 'serving_runtime.py').read_text(encoding='utf-8').splitlines()))))))))))
        changed = [line for line in difflib.unified_diff(before, after, lineterm='', n=0)
                   if line[:1] in '+-' and not line.startswith(('+++', '---'))]
        added = [line[1:].strip() for line in changed if line.startswith('+')]
        removed = [line[1:].strip() for line in changed if line.startswith('-')]
        self.assertEqual(removed, ["if padded_min_users is not None else {}))"])
        self.assertEqual(added[0], "if padded_min_users is not None else {}),")
        self.assertIn("**({'collectives': collectives}", added)
        self.assertIn("if os.environ.get('QWEN_FAST_FUSED_COMMIT') == '1'", added)
        # On four cards (fused commit off) S2 B6's publication warm needs the collectives too.
        self.assertIn("or (os.environ.get('QWEN_FAST_EXTENT_REPLAY') == '1'", added)
        self.assertIn("and os.environ.get('QWEN_FAST_TP', '2') != '2') else {}))", added)
        self.assertTrue(all(line.startswith('#') for line in added[1:]
                            if 'collectives' not in line and 'FUSED' not in line and 'EXTENT_REPLAY' not in line
                            and 'QWEN_FAST_TP' not in line))


def cut_code(lines, first, last, code, replacement=()):
    """cut_once, asserting that the run's statements (its lines less blanks and comments, stripped) are
    exactly `code`, so the cut hides nothing else."""
    starts = [index for index, value in enumerate(lines) if value.strip() == first]
    if len(starts) != 1:
        raise AssertionError('%r is not in serving_runtime.py exactly once' % first)
    ends = [index for index in range(starts[0], len(lines)) if lines[index].strip() == last]
    if not ends:
        raise AssertionError('%r has no %r after it' % (first, last))
    found = tuple(value.strip() for value in lines[starts[0]:ends[0] + 1]
                  if value.strip() and not value.strip().startswith('#'))
    if found != tuple(code):
        raise AssertionError('The run from %r holds more than expected: %r' % (first, found))
    return lines[:starts[0]] + list(replacement) + lines[ends[0] + 1:]


def replace_run(lines, run, replacement):
    """lines with the one contiguous run whose lines equal (stripped) `run` replaced by `replacement` (lines, indentation as given);
    exactly one such run, or AssertionError."""
    stripped = [value.strip() for value in lines]
    starts = [index for index in range(len(lines) - len(run) + 1) if stripped[index:index + len(run)] == list(run)]
    if len(starts) != 1:
        raise AssertionError('%r is not in serving_runtime.py exactly once' % (run,))
    return lines[:starts[0]] + list(replacement) + lines[starts[0] + len(run):]


def without_solo_and_lanes(lines):
    """serving_runtime.py less the lanes window's two default-off hunks, which landed after every parent this test compares with:
    D0, the one-user block (QWEN_FAST_SOLO_LANE) and the fast lane (QWEN_FAST_LANE). Every statement they add or widen is put back to
    the line it was, each found exactly once, so nothing else is hidden."""
    lines = cut_once(lines, '# QWEN_FAST_SOLO_LANE (D0, default off; strictly \'0\' or \'1\'): the one-user 16-row block beside M3, gate only. Refused',
                     "lane_config = serving_fast_lane.lane_admission(solo_lane, seats=policy['scheduler_requests'], log=pindiag)")
    lines = cut_once(lines, 'solo_shape_value = None',
                     'distinct_shapes = distinct_shapes + ((solo_shape_value.users, solo_shape_value.rows_per_user),)')
    lines = cut_once(lines, "# D0: the one-user block, after M3 (a block is built before any request exists, and the pool's slot 0 is",
                     "memory_ledger.record('P6', point='solo', packed_block=solo_block)")
    lines = replace_run(lines, ("extents = [getattr(packed_block, 'extent', False)",
                                'for packed_block in packed_blocks + ([solo_block] if solo_block is not None else [])]'),
                        [' ' * 12 + "extents = [getattr(packed_block, 'extent', False) for packed_block in packed_blocks]"])
    lines = replace_run(lines, ('packed_step = PackedStep(packed_blocks if four_as_two else packed_blocks[0],',
                                "**({'solo': solo_block} if solo_block is not None else {}))"),
                        [' ' * 12 + 'packed_step = PackedStep(packed_blocks if four_as_two else packed_blocks[0])'])
    lines = replace_run(lines, ('else dict(block=packed_blocks[0].describe())),',
                                '**(dict(solo=solo_block.describe()) if solo_block is not None else {}))'),
                        [' ' * 19 + 'else dict(block=packed_blocks[0].describe())))'])
    lines = replace_run(lines, ('packed_any_admission.admit_blocks(packed_blocks + ([solo_block] if solo_block is not None else []),',
                                'log=pindiag)'),
                        [' ' * 12 + 'packed_any_admission.admit_blocks(packed_blocks, log=pindiag)'])
    lines = cut_run(lines, ('lanes = None', 'if lane_config is not None:',
                            'lanes = serving_fast_lane.LaneRuntime(lane_config, log=pindiag)'))
    lines = cut_once(lines, '# QWEN_FAST_LANE: the lane the request is granted decides which pool slots its engine may borrow (the fast request',
                     'grant = lanes.admit(state.req_id, state.sampling_params, slot0_free=not pool.slots[0].lent)')
    lines = replace_run(lines, ('try:', 'if grant is None:',
                                'request = create_request() if experiment is None else experiment.create(create_request)',
                                'else:', 'with pool.slot_order(grant.slot_order):',
                                'request = create_request() if experiment is None else experiment.create(create_request)',
                                'except BaseException:', 'if lanes is not None:', 'lanes.release(state.req_id)', 'raise'),
                        [' ' * 12 + 'request = create_request() if experiment is None else experiment.create(create_request)'])
    lines = replace_run(lines, ('if lanes is not None:', 'lanes.release(state.req_id)', 'request.close(state.req_id)'),
                        [' ' * 16 + 'request.close(state.req_id)'])
    return replace_run(lines, ('cancelled=cancelled, packed_step=packed_step,', "**({'lanes': lanes} if lanes is not None else {}))"),
                       [' ' * 12 + 'cancelled=cancelled, packed_step=packed_step)'])


# The tp4/seats8 hunks of serving_runtime.py against the tp4/next-2 head (QWEN_FAST_M3_BLOCKS, default off), each as (what the file holds
# now, what it held before), in file order: the m3_blocks flag and helpers, the two-phase completion, the eight-request refusal text and the
# block count at every place the attach builds a block. without_m3_blocks puts each back, found exactly once, so nothing else is hidden.
M3_BLOCKS_HUNKS = [('# Eight seats on TWO 64-row M3 blocks (default off): block A over pool slots (0..3), block B over (4..7), each exactly\n'
  "# the qualified 4-user block, run back to back in one step. '1' (or unset) is today's one block at four requests; '2'\n"
  '# is admitted at exactly eight scheduler requests and nowhere else; any other value is refused.\n'
  "M3_BLOCKS_FLAG = 'QWEN_FAST_M3_BLOCKS'\n"
  'M3_BLOCKS_USERS = 8\n'
  "M3_BLOCKS_MARKER = '[PINDIAG] M3 blocks={} over pool slots {} (QWEN_FAST_M3_BLOCKS={}); each block is the qualified 4-user 64-row "
  "block'\n"
  "CAPTURE_PROGRAMS_MARKER = '[PINDIAG] packed blocks capture block={} programs={}->{}'\n"
  "CLOSE_FAILED_MARKER = '[PINDIAG] a sibling block did not close without the fence: {}'\n",
  ''),
 ('def m3_blocks(environ=None):\n'
  '    """QWEN_FAST_M3_BLOCKS, strictly: unset or \'1\' is one M3 block (today\'s), \'2\' is two (eight seats), and anything\n'
  '    else - an empty value included - is a configuration error naming the flag, refused before anything is built."""\n'
  "    value = (os.environ if environ is None else environ).get(M3_BLOCKS_FLAG, '1')\n"
  "    if value not in ('1', '2'):\n"
  "        raise ValueError('%s must be 1 or 2, got %r' % (M3_BLOCKS_FLAG, value))\n"
  '    return int(value)\n'
  '\n'
  '\n'
  'def m3_blocks_for(policy, environ=None):\n'
  '    """How many M3 blocks this attach builds (1 or 2), refusing - ValueError naming QWEN_FAST_M3_BLOCKS - a malformed\n'
  '    value and the value 2 at any scheduler request count but eight: two blocks are the eight-seat shape, and at four\n'
  '    requests the flag would be a second, silent way to ask for something the four-seat attach already is."""\n'
  '    environ = os.environ if environ is None else environ\n'
  '    blocks = m3_blocks(environ)\n'
  '    policy = policy() if callable(policy) else policy\n'
  "    if blocks == 2 and policy['scheduler_requests'] != M3_BLOCKS_USERS:\n"
  "        raise ValueError('%s=2 builds two 4-user M3 blocks and is admitted at exactly %d scheduler requests, not %s'\n"
  "                         % (M3_BLOCKS_FLAG, M3_BLOCKS_USERS, policy['scheduler_requests']))\n"
  '    return blocks\n'
  '\n'
  '\n',
  ''),
 ('    QWEN_FAST_FOUR_AS_TWO=0 and QWEN_FAST_PACKED_STEP=1 - or, under QWEN_FAST_M3_BLOCKS=2, the two M3 blocks\n'
  '    of eight scheduler requests (each block exactly that same 4-user block), and a short description of the\n'
  '    shape actually configured, for a marker (it names the block count only when it is not one). `policy` is\n'
  '    validate_fast_config\'s dict, or a zero-argument callable returning it."""\n',
  '    QWEN_FAST_FOUR_AS_TWO=0 and QWEN_FAST_PACKED_STEP=1 - and a short description of the\n'
  "    shape actually configured, for a marker. `policy` is validate_fast_config's dict, or a\n"
  '    zero-argument callable returning it."""\n'),
 ('    blocks = m3_blocks(environ)\n'
  "    met = users == (4 if blocks == 1 else M3_BLOCKS_USERS) and four_as_two == '0' and packed == '1'\n"
  "    return met, 'users=%s FOUR_AS_TWO=%s PACKED_STEP=%s%s' % (users, four_as_two, packed,\n"
  "                                                             '' if blocks == 1 else ' M3_BLOCKS=%d' % blocks)\n",
  "    met = users == 4 and four_as_two == '0' and packed == '1'\n"
  "    return met, 'users=%s FOUR_AS_TWO=%s PACKED_STEP=%s' % (users, four_as_two, packed)\n"),
 ("                             '(users=4 FOUR_AS_TWO=0 PACKED_STEP=1; users=8 with QWEN_FAST_M3_BLOCKS=2), not ' + shape)\n",
  "                             '(users=4 FOUR_AS_TWO=0 PACKED_STEP=1), not ' + shape)\n"),
 ("                         '(users=4 FOUR_AS_TWO=0 PACKED_STEP=1; users=8 with QWEN_FAST_M3_BLOCKS=2), not ' + shape)\n",
  "                         '(users=4 FOUR_AS_TWO=0 PACKED_STEP=1), not ' + shape)\n"),
 ('def complete_blocks_two_phase(blocks, model=None, log=None, before_captures=None):\n'
  '    """QWEN_FAST_M3_BLOCKS=2 (A1c): finish the construction of blocks built with defer_capture=True. Each block has\n'
  '    already allocated its persistent state (the initial snapshot, checkpoints, taps) and the first phase here runs\n'
  "    every block's warm forward and builds every fixture (extent words, masks, retained storage); only then does any\n"
  '    block capture a trace, and the publication warm and the reseed come after the LAST capture. The rule is\n'
  "    packed_verifier's CONSTRUCTION ORDER: a buffer a block keeps across rounds that is allocated after a capture\n"
  "    lands in the holes that capture freed, and every replay overwrites it. Built one after the other, block B's\n"
  '    fixture inputs, taps, checkpoints and extent words would have been exactly that.\n'
  '\n'
  '    A log line per capture gives the program-cache count before and after each block: block B compiles zero programs\n'
  '    after block A\'s capture (the probe window reads it).\n'
  '\n'
  '    `before_captures` (QWEN_FAST_M3_REQUEST_WARM on at two blocks; request_width_warm): called once, after every block\'s warm_and_fixture and\n'
  '    before the first capture, inside the same guard - a failure closes both blocks without the fence and propagates."""\n'
  '    log = pindiag if log is None else log\n'
  '    try:\n'
  '        for block in blocks:\n'
  '            block.warm_and_fixture()\n'
  '        if before_captures is not None:\n'
  '            before_captures()\n'
  '        for index, block in enumerate(blocks):\n'
  '            before = program_count(model) if model is not None else None\n'
  '            block.capture_traces()\n'
  '            log(CAPTURE_PROGRAMS_MARKER, index, before, program_count(model) if model is not None else None)\n'
  '        for index, block in enumerate(blocks):\n'
  '            if index:\n'
  "                block.warm_publication = False      # block A's warm covered the same plan on the shared program cache\n"
  '            block.finish_construction()\n'
  '    except BaseException:\n'
  '        # The failing block closed itself without the device fence (it may be hung); a sibling still under construction\n'
  '        # would be closed by the attach scope WITH the fence and block there on the same hung device. Close it the same\n'
  '        # way, here, then let the original failure through.\n'
  '        for block in blocks:\n'
  '            try:\n'
  '                block.close(wait=False)\n'
  '            except BaseException as error:\n'
  '                log(CLOSE_FAILED_MARKER, repr(error)[:200])\n'
  '        raise\n'
  '\n'
  '\n',
  ''),
 ('    # QWEN_FAST_M3_BLOCKS (default 1; 2 only at eight scheduler requests): read strictly before anything is built, so a\n'
  '    # malformed value, or 2 at any other request count, is refused here by name.\n'
  '    m3_blocks_count = m3_blocks_for(policy)\n',
  ''),
 ('        # QWEN_FAST_M3_BLOCKS=2: two 64-row M3 blocks over pool slots (0..3) and (4..7), the same multi-block path.\n'
  '        m3_blocks_two = False\n',
  ''),
 ('        if m3_blocks_count == 2 and not packed_requested:\n'
  "            raise ValueError('%s=2 builds packed blocks and needs QWEN_FAST_PACKED_STEP=1, not %s'\n"
  "                             % (M3_BLOCKS_FLAG, os.environ.get('QWEN_FAST_PACKED_STEP', 'unset')))\n",
  ''),
 ('            from packed_shapes import m1_shape, m3_shape as m3_block_shape, serving_shape\n'
  '\n'
  '            if m3_blocks_count == 2:\n'
  "                if os.environ.get('QWEN_FAST_FOUR_AS_TWO', '1') != '0':\n"
  "                    raise ValueError('%s=2 needs QWEN_FAST_FOUR_AS_TWO=0, not %s'\n"
  "                                     % (M3_BLOCKS_FLAG, os.environ.get('QWEN_FAST_FOUR_AS_TWO', 'unset')))\n"
  '                m3_blocks_two = True\n'
  '                packed_shapes = (m3_block_shape(page_width), m3_block_shape(page_width))\n'
  "            elif policy['scheduler_requests'] == 4 and os.environ.get('QWEN_FAST_FOUR_AS_TWO', '1') != '0':\n",
  '            from packed_shapes import m1_shape, serving_shape\n'
  '\n'
  "            if policy['scheduler_requests'] == 4 and os.environ.get('QWEN_FAST_FOUR_AS_TWO', '1') != '0':\n"),
 ("                             '(QWEN_FAST_PACKED_STEP=%s, %d scheduler requests)%s'\n",
  "                             '(QWEN_FAST_PACKED_STEP=%s, %d scheduler requests)'\n"),
 ("                                policy['scheduler_requests'],\n"
  '                                # Eight requests are the two-block shape: say which flag builds it.\n'
  "                                '; eight requests need %s=2 (two 64-row M3 blocks)' % M3_BLOCKS_FLAG\n"
  "                                if policy['scheduler_requests'] == M3_BLOCKS_USERS else ''))\n",
  "                                policy['scheduler_requests']))\n"),
 ('        if four_as_two or m3_blocks_two:\n', '        if four_as_two:\n'),
 ("        if m3_blocks_two:\n            pindiag(M3_BLOCKS_MARKER, 2, '(0, 1, 2, 3) and (4, 5, 6, 7)', 2)\n", ''),
 ("                **({'packed_replicas': {distinct_shapes[0]: len(packed_shapes)}} if four_as_two or m3_blocks_two else {}),\n",
  "                **({'packed_replicas': {distinct_shapes[0]: len(packed_shapes)}} if four_as_two else {}),\n"),
 ("                                                    **({'pool_slots': tuple(range(slot, slot + shape.users))}\n"
  '                                                       if four_as_two or m3_blocks_two else {}),\n'
  '                                                    # QWEN_FAST_M3_BLOCKS=2 (A1c): every block allocates and warms\n'
  '                                                    # first, then every block captures (complete_blocks_two_phase above).\n'
  "                                                    **({'defer_capture': True} if m3_blocks_two else {}),\n",
  "                                                    **({'pool_slots': tuple(range(slot, slot + shape.users))} if four_as_two else "
  '{}),\n'),
 ('                if not m3_blocks_two:\n'
  "                    memory_ledger.record('P6', point='block%d' % len(packed_blocks), packed_block=packed_block)\n",
  "                memory_ledger.record('P6', point='block%d' % len(packed_blocks), packed_block=packed_block)\n"),
 ('            if m3_blocks_two:\n'
  '                if request_widths:\n'
  '                    from request_width_warm import warm_request_widths\n'
  '\n'
  '                    complete_blocks_two_phase(packed_blocks, model, before_captures=lambda: warm_request_widths(\n'
  '                        operations, model, helpers, sampler, page_width, widths=request_widths))\n'
  '                else:\n'
  '                    complete_blocks_two_phase(packed_blocks, model)\n'
  '                for index, packed_block in enumerate(packed_blocks, 1):\n'
  "                    memory_ledger.record('P6', point='block%d' % index, packed_block=packed_block)\n",
  ''),
 ('            # QWEN_FAST_M3_BLOCKS=2: refused unless BOTH blocks report that their captured trace reads every carry in\n'
  "            # place (QWEN_FAST_VERIFY_T1 #3). Every block's commit DMA writes the model's native slot 0, so a block whose\n"
  "            # trace read slot 0 instead of its own carries would be fed another block's state: the attach is refused.\n"
  '            if m3_blocks_two:\n'
  '                refused_blocks = [index for index, packed_block in enumerate(packed_blocks)\n'
  "                                  if getattr(packed_block, 'carries_in_place', False) is not True]\n"
  '                if refused_blocks:\n'
  "                    raise ValueError('%s=2 needs every M3 block to read its carries in place (QWEN_FAST_VERIFY_T1 #3); '\n"
  "                                     'blocks %s do not report carries_in_place' % (M3_BLOCKS_FLAG, refused_blocks))\n",
  ''),
 ('            packed_step = PackedStep(packed_blocks if four_as_two or m3_blocks_two else packed_blocks[0],\n'
  "                                     **({'solo': solo_block} if solo_block is not None else {}),\n"
  "                                     # QWEN_FAST_M3_BLOCKS=2: each block's ticket width is its own (PackedStep.proposal_groups).\n"
  "                                     **({'per_block_widths': True} if m3_blocks_two else {}))\n"
  '            if m3_blocks_two:\n'
  '                # New arrivals fill a block that has exactly one live user first, else the fuller block that is not full\n'
  '                # (ServingBufferPool.place_blocks), so a lone user is rare and a block runs packed whenever it can.\n'
  '                pool.place_blocks(tuple(tuple(range(index * 4, index * 4 + 4)) for index in range(2)))\n',
  '            packed_step = PackedStep(packed_blocks if four_as_two else packed_blocks[0],\n'
  "                                     **({'solo': solo_block} if solo_block is not None else {}))\n"),
 ('                **(dict(blocks=[packed_block.describe() for packed_block in packed_blocks])\n'
  '                   if four_as_two or m3_blocks_two else dict(block=packed_blocks[0].describe())),\n',
  '                **(dict(blocks=[packed_block.describe() for packed_block in packed_blocks]) if four_as_two\n'
  '                   else dict(block=packed_blocks[0].describe())),\n')]


def without_m3_blocks(lines):
    """serving_runtime.py less the eight-seat flag's hunks (M3_BLOCKS_HUNKS), which landed after this parent. Applied first, so the
    lanes window's helper below finds the PackedStep call and the block lists as they were."""
    text = chr(10).join(lines) + chr(10)
    for index, (now, before) in enumerate(M3_BLOCKS_HUNKS):
        if text.count(now) != 1:
            raise AssertionError('M3 blocks hunk %d is not in serving_runtime.py exactly once: %r' % (index, now[:120]))
        text = text.replace(now, before)
    return text.split(chr(10))[:-1]


def without_sticky(lines):
    """serving_runtime.py less phase-1 sticky sessions (QWEN_FAST_STICKY_SESSIONS, default off), which
    landed after this parent: its widened policy import back to C2-any's, the `import time` it added, the
    STICKY_ENGINE_MARKER constant, the prefix-warm note in the attach comment, the flag read and
    capture_factory's `start` branch back to the one return, and the timed engine build back to the bare
    call. Each is cut exactly once and checked statement by statement, so nothing else is hidden."""
    line = ('from serving_fast_policy import STICKY_SESSIONS_FLAG, any_request_enabled, sticky_sessions_enabled, '
            'validate_fast_config')
    if lines.count(line) != 1:
        raise AssertionError('The sticky import %r is not in serving_runtime.py exactly once' % line)
    lines = ['from serving_fast_policy import any_request_enabled, validate_fast_config' if value == line else value
             for value in lines]
    lines = cut_code(lines, 'import time', 'import time', ('import time',))
    lines = cut_code(lines, "# Sticky sessions (QWEN_FAST_STICKY_SESSIONS=1 only): one line per admitted request's "
                     'engine build,', "STICKY_ENGINE_MARKER = '[PINDIAG] sticky engine built req='",
                     ("STICKY_ENGINE_MARKER = '[PINDIAG] sticky engine built req='",))
    lines = cut_code(lines, "# (Under QWEN_PREFIX_REUSE=1 that warmup first runs the prefix-reuse model graft's",
                     "# model's persistent B=1 prefill scratch, and no trace.)", ())
    lines = cut_code(lines, '# Sticky sessions (QWEN_FAST_STICKY_SESSIONS, read once at attach; default off). On, every',
                     'def capture_factory(position, start=0):',
                     ('sticky = sticky_sessions_enabled()', 'def capture_factory(position, start=0):'),
                     ['        def capture_factory(position):'])
    lines = cut_code(lines, 'if not sticky:',
                     'return PrefillWindowCapture(operations, model, position, TARGET_TAPS, start=start, '
                     'prefix_route=True)',
                     ('if not sticky:', 'if start:',
                      "raise ValueError('A prefill resumed at %d needs %s=1' % (start, STICKY_SESSIONS_FLAG))",
                      'return PrefillWindowCapture(operations, model, position, TARGET_TAPS)',
                      'return PrefillWindowCapture(operations, model, position, TARGET_TAPS, start=start, '
                      'prefix_route=True)'),
                     ['            return PrefillWindowCapture(operations, model, position, TARGET_TAPS)'])
    guards = [index for index, value in enumerate(lines) if value.strip() == 'if sticky:']
    if len(guards) != 2 or lines[guards[0] + 1].strip() != 'began = time.perf_counter()':
        raise AssertionError('The sticky engine-build timing is not in serving_runtime.py exactly once')
    lines = lines[:guards[0]] + lines[guards[0] + 2:]
    return cut_code(lines, 'if sticky:', 'len(state.prompt_token_ids))',
                    ('if sticky:',
                     "pindiag(STICKY_ENGINE_MARKER + '{} ms={:.1f} frontier={} prompt={}', str(state.req_id)[:48],",
                     '(time.perf_counter() - began) * 1000.0, state.num_computed_tokens,',
                     'len(state.prompt_token_ids))'))


# The engine-reuse hunks of serving_runtime.py against the tp4/levern-prefix head (QWEN_FAST_PARKED_ENGINES, default off), each as (what the file holds
# now, what it held before), in file order. without_parked_engines puts each back, found exactly once, so nothing else is hidden.
PARKED_ENGINES_HUNKS = [
    ("    return blocks\n\n\n# Engine reuse (serving_parked_engines; default off): one engine per pool slot, parked at attach.\nPARKED_ENGINES_FLAG = 'QWEN_FAST_PARKED_ENGINES'\nPARKED_AUDIT_FLAG = 'QWEN_FAST_PARKED_AUDIT'\nPARKED_DRAFTS_FLAG = 'QWEN_FAST_PARKED_DRAFTS'\n\n",
     '    return blocks\n\n'),
    ("\n        # Engine reuse (QWEN_FAST_PARKED_ENGINES; default off, strictly '0' or '1'): one engine per pool slot, built here on a synthetic request and\n        # parked (serving_parked_engines.ParkedEngineSet) - after the blocks, which refuse to build once a slot is lent, and before the DRAM\n        # admission and the lifecycle. Registered after the blocks, so it closes before them, the weights and the pool, which refuse to close\n        # while a slot or a weight is still lent. R2's replay ledger (QWEN_FAST_PARKED_AUDIT=1, gate only) installs with or without the parked\n        # engines, so the flag-off control arm of the audit twin runs the same check. Off (unset or '0'), nothing is imported or built.\n        parked_engines = None\n        if (os.environ.get(PARKED_ENGINES_FLAG, '0') != '0' or os.environ.get(PARKED_AUDIT_FLAG, '0') != '0'\n                or os.environ.get(PARKED_DRAFTS_FLAG, '0') != '0'):\n            import serving_parked_engines\n\n            replay_ledger = serving_parked_engines.install_replay_ledger(operations)\n            if replay_ledger is not None:\n                scopes.callback(replay_ledger.uninstall)\n            if os.environ.get(PARKED_ENGINES_FLAG, '0') != '0':\n                serving_parked_engines.parked_engines_enabled()   # strictly '1' from here: any other value is refused\n                parked_engines = serving_parked_engines.ParkedEngineSet(operations=operations, model=model,\n                    sampler=sampler, helpers=helpers, pool=pool, weights=weights, fixtures=fixtures,\n                    collectives=collectives, blocks=packed_blocks if packed_shapes else (),\n                    capture_rows=capture_rows if trimmed else None)\n                scopes.callback(parked_engines.close)\n                parked_engines.build()\n                import levern_policy\n\n                if levern_policy.build_ms_mode() == 'learned':\n                    # The deadline governor charges each pending prefill the cost of the slot it will take: a rebind while parked engines remain.\n                    levern_policy.admission_cost().parked_free = parked_engines.parked_count\n\n        # Sticky sessions (QWEN_FAST_STICKY_SESSIONS, read once at attach; default off). On, every\n",
     '\n        # Sticky sessions (QWEN_FAST_STICKY_SESSIONS, read once at attach; default off). On, every\n'),
    ("        sticky = sticky_sessions_enabled()\n        cost_learning = False\n        if parked_engines is not None:\n            import levern_policy\n\n            cost_learning = levern_policy.build_ms_mode() == 'learned'\n        lanes = None\n",
     '        sticky = sticky_sessions_enabled()\n        lanes = None\n'),
    ('                    collectives=collectives, buffer_pool=pool, shared_weights=weights,\n                    **(dict(capture_rows=capture_rows) if trimmed else {}),\n                    **(dict(parked=parked_engines) if parked_engines is not None else {}))\n\n',
     '                    collectives=collectives, buffer_pool=pool, shared_weights=weights,\n                    **(dict(capture_rows=capture_rows) if trimmed else {}))\n\n'),
    ('                grant = lanes.admit(state.req_id, state.sampling_params, slot0_free=not pool.slots[0].lent)\n            learning = parked_engines is not None and cost_learning\n            if sticky or learning:\n                began = time.perf_counter()\n',
     '                grant = lanes.admit(state.req_id, state.sampling_params, slot0_free=not pool.slots[0].lent)\n            if sticky:\n                began = time.perf_counter()\n'),
    ("                raise\n            if learning:\n                # QWEN_FAST_LEVERN_BUILD_MS=learned: what this admission cost, by kind (the governor's per-pending charge).\n                import levern_policy\n\n                levern_policy.observe_admission_cost('rebind' if getattr(request, 'parked_slot', None) is not None else 'build',\n                                                     (time.perf_counter() - began) * 1000.0, log=pindiag)\n            # Engine reuse: a request rebound onto a parked engine did not build one; its engine's ledger walk ran at attach (P7p) and its census\n            # at attach too. The sticky line below stays for every request, with kind= naming which it was (the prefix gate's A8 reads the line).\n            rebound = parked_engines is not None and getattr(request, 'parked_slot', None) is not None\n            if sticky:\n",
     '                raise\n            if sticky:\n'),
    ("                # its tail prefill and the build phase 1 still pays (STICKY_ENGINE_MARKER).\n                pindiag(STICKY_ENGINE_MARKER + '{} ms={:.1f} frontier={} prompt={}' + ('' if parked_engines is None else ' kind={}'),\n                        str(state.req_id)[:48], (time.perf_counter() - began) * 1000.0, state.num_computed_tokens,\n                        len(state.prompt_token_ids), *(() if parked_engines is None else ('rebind' if rebound else 'build',)))\n            # The allocator after this request's engine and its captures: one line per\n",
     "                # its tail prefill and the build phase 1 still pays (STICKY_ENGINE_MARKER).\n                pindiag(STICKY_ENGINE_MARKER + '{} ms={:.1f} frontier={} prompt={}', str(state.req_id)[:48],\n                        (time.perf_counter() - began) * 1000.0, state.num_computed_tokens,\n                        len(state.prompt_token_ids))\n            # The allocator after this request's engine and its captures: one line per\n"),
    ("                pindiag('[PINDIAG] dram after engine {}: {}', str(state.req_id)[:48], dram_line(pool))\n            if not rebound and memory_ledger.admission_diag('engine'):\n                memory_ledger.engine_admitted(str(state.req_id), engine_request=request)\n",
     "                pindiag('[PINDIAG] dram after engine {}: {}', str(state.req_id)[:48], dram_line(pool))\n            if memory_ledger.admission_diag('engine'):\n                memory_ledger.engine_admitted(str(state.req_id), engine_request=request)\n"),
    ('                memory_ledger.engine_admitted(str(state.req_id), engine_request=request)\n            if not rebound:\n                trace_census.census_engine(str(state.req_id), request, operations)\n            try:\n',
     '                memory_ledger.engine_admitted(str(state.req_id), engine_request=request)\n            trace_census.census_engine(str(state.req_id), request, operations)\n            try:\n'),
    ("                    lanes.release(state.req_id)\n                if rebound:\n                    # A failed binding leaves the slot unfit to park; its close unparks it.\n                    request.parked_slot.unfit = 'page binding failed'\n                request.close(state.req_id)\n",
     '                    lanes.release(state.req_id)\n                request.close(state.req_id)\n'),
    ('        if extent_replay_enabled():\n            scopes.callback(register_dram_admission(pool, **({} if parked_engines is None else dict(parked=parked_engines))))\n        capture_factory, bridge_factory = prefill_tripwire(model, capture_factory, bridge_factory)\n',
     '        if extent_replay_enabled():\n            scopes.callback(register_dram_admission(pool))\n        capture_factory, bridge_factory = prefill_tripwire(model, capture_factory, bridge_factory)\n'),
    ("            cancelled=cancelled, packed_step=packed_step,\n            **({'lanes': lanes} if lanes is not None else {}),\n            **({} if parked_engines is None else dict(idle=parked_engines.idle, parked_poll=parked_engines.poll_off)))\n    except BaseException as failure:\n",
     "            cancelled=cancelled, packed_step=packed_step,\n            **({'lanes': lanes} if lanes is not None else {}))\n    except BaseException as failure:\n"),
]


def without_parked_engines(lines):
    text = '\n'.join(lines) + '\n'
    for now, before in PARKED_ENGINES_HUNKS:
        if text.count(now) != 1:
            raise AssertionError('The engine-reuse hunk %r is not in serving_runtime.py exactly once' % now[:60])
        text = text.replace(now, before)
    return text.splitlines()


def without_request_warm(lines):
    """serving_runtime.py less the tp4/warm4 hunks (QWEN_FAST_M3_REQUEST_WARM, default off), which landed after this parent: the flag
    constants, m3_request_warm, the parse line after validate_fast_config and the hook between the prefill warm and the block; each
    found exactly once, so nothing else is hidden."""
    text = '\n'.join(lines)

    def cut(text, start, end, keep_end=True):
        if text.count(start) != 1:
            raise AssertionError('The request-warm hunk starting %r is not in serving_runtime.py exactly once' % start[:50])
        head, rest = text.split(start)
        tail = rest[rest.index(end):] if keep_end else rest[rest.index(end) + len(end):]
        return head + tail

    text = cut(text, '# QWEN_FAST_M3_REQUEST_WARM (default off;', "M3_REQUEST_WARM_FLAG = 'QWEN_FAST_M3_REQUEST_WARM'\n", False)
    text = cut(text, 'def m3_request_warm(blocks, policy', 'def c2_any_without_block(')
    text = cut(text, '    # QWEN_FAST_M3_REQUEST_WARM: the widths to warm', '    # Measurement-only, env-gated admission of one named', True)
    text = cut(text, '        if request_widths and not m3_blocks_two:\n', "        # The device step. By default the sequential one", True)
    return text.split('\n')


def without_levern(lines):
    """serving_runtime.py less Lever N's attach block (tp4/lever-n: the route's install and warm under QWEN_FAST_LEVER_N / QWEN_FAST_LEVERN_AUDIT, never
    imported otherwise), which landed after this parent: the one contiguous block from its flag test to the warm call, found once."""
    start = [index for index, value in enumerate(lines) if value.strip().startswith("if os.environ.get('QWEN_FAST_LEVER_N', '0') != '0'")]
    if len(start) != 1:
        raise AssertionError('Lever N attach block is not in serving_runtime.py exactly once')
    # ... through the merged route's epoch-scope registration (tp4/levern-prefix), the last statement of the same block
    end = next(index for index in range(start[0], len(lines))
               if lines[index].strip() == 'levern_route.engage_epoch_scope(model, lambda tensor: device_addresses(operations, tensor), log=pindiag)')
    return lines[:start[0]] + lines[end + 1:]


def without_trace_census(lines):
    """serving_runtime.py less the sequential-hang diagnostics' hooks (trace_census; each a no-op unless its flag is set), which
    landed after this parent: the import and the four call lines (tp4-serve-4's engine build guard among them), asserted to be
    exactly those, found once each."""
    hooks = ('import trace_census', 'trace_census.note_collectives(collectives, model, sampler)', 'trace_census.engine_begin()',
             'trace_census.census_engine(str(state.req_id), request, operations)',
             'create_request = trace_census.build_guard(create_request, operations, model.mesh_device, state.req_id)')
    found = [value for value in lines if value.strip() in hooks]
    if sorted(value.strip() for value in found) != sorted(hooks):
        raise AssertionError('The trace_census hooks are not in serving_runtime.py exactly once each')
    return [value for value in lines if value.strip() not in hooks]


def without_diag_trim(lines):
    """serving_runtime.py less the admission-diagnostics trim (QWEN_FAST_ADMISSION_DIAG_TRIM, tp4/freeze), which landed
    after this parent: each of its four guarded ledger and log calls put back as the parent had it, the block
    asserted to be exactly the trim's text and found once."""
    sites = (
        (["# QWEN_FAST_ADMISSION_DIAG_TRIM=1 (the traffic twins): the first admission's point only.",
          "if memory_ledger.admission_diag('prefill_before'):",
          "memory_ledger.record('prefill', point='before prompt=%d' % position)"],
         ["memory_ledger.record('prefill', point='before prompt=%d' % position)"]),
        (["if memory_ledger.admission_diag('prefill_after'):",
          "memory_ledger.record('prefill', point='after req=%s' % memory_ledger.short_id(state.req_id),",
          "request=str(state.req_id), model_after_prefill=model)"],
         ["memory_ledger.record('prefill', point='after req=%s' % memory_ledger.short_id(state.req_id),",
          "request=str(state.req_id), model_after_prefill=model)"]),
        (["# Under QWEN_FAST_ADMISSION_DIAG_TRIM=1 the line is off (it reads the allocator of every chip, and",
          "# the first engine's ledger point carries the same reading) and only the first engine is walked.",
          "if not memory_ledger.trim_enabled():",
          "pindiag('[PINDIAG] dram after engine {}: {}', str(state.req_id)[:48], dram_line(pool))",
          "if memory_ledger.admission_diag('engine'):",
          "memory_ledger.engine_admitted(str(state.req_id), engine_request=request)"],
         ["pindiag('[PINDIAG] dram after engine {}: {}', str(state.req_id)[:48], dram_line(pool))",
          "memory_ledger.engine_admitted(str(state.req_id), engine_request=request)"]),
    )
    stripped = [value.strip() for value in lines]
    for new, old in sites:
        starts = [i for i in range(len(lines) - len(new) + 1) if stripped[i:i + len(new)] == new]
        if len(starts) != 1:
            raise AssertionError('The diagnostics-trim hunk %r is not in serving_runtime.py exactly once' % new[0])
        first = starts[0]
        indent = len(lines[first]) - len(lines[first].lstrip())
        restored = [' ' * indent + value if not value.startswith('request=') else
                    ' ' * (indent + len('memory_ledger.record(')) + value for value in old]
        # a continuation line keeps the parent's alignment, one column past the call's opening parenthesis (the diff is exact)
        lines = lines[:first] + restored + lines[first + len(new):]
        stripped = [value.strip() for value in lines]
    return lines


def without_prefill_scratch(lines):
    """serving_runtime.py less the four-card prefill warm hunk (tp4/stack-fix), which landed after this parent: its helper
    functions (program_count, prefill_warm_before_traces, prefill_tripwire) and the two call lines. Asserted to be exactly that,
    found once each."""
    starts = [i for i, value in enumerate(lines) if value.startswith('def prefill_warm_before_traces(')]
    mine = ('def program_count(', 'def prefill_warm_before_traces(', 'def prefill_tripwire(')
    ends = [i for i, value in enumerate(lines) if value.startswith(('def ', '@')) and starts and i > starts[0]
            and not value.startswith(mine)][:1]
    calls = [i for i, value in enumerate(lines)
             if value.strip() in ("prefill_warm_before_traces(runner, model, policy['scheduler_requests'], operations=operations)",
                                  'capture_factory, bridge_factory = prefill_tripwire(model, capture_factory, bridge_factory)')]
    # The helpers sit between the constants they use; the marker constants and program_count/tripwire are inside the span.
    consts = [i for i, value in enumerate(lines) if value.startswith(('WARM_MARKER = ', 'PREFILL_PROGRAMS_MARKER = ',
                                                                     'WARM_SLOT_TOKENS = ', 'WARM_LONG_PROMPTS = '))]
    if len(starts) != 1 or len(ends) != 1 or len(calls) != 2 or ends[0] < starts[0] or len(consts) != 4 or consts[0] > starts[0]:
        raise AssertionError('The prefill-warm hunk is not in serving_runtime.py exactly once')
    drop = set(range(consts[0], ends[0])) | set(calls)
    return [value for i, value in enumerate(lines) if i not in drop]


def without_any_request(lines):
    """serving_runtime.py less the C2-any (QWEN_FAST_ANY_REQUEST, plan S1) attach hunk, which
    landed after this parent: its two widened imports back to theirs, and the one guarded block
    that refuses a replaying capture cap and runs attach_source_check. Asserted to be exactly
    that - found once, comments plus those statements - so the exclusion hides nothing else."""
    imports = {
        'from serving_fast_policy import any_request_enabled, validate_fast_config':
            'from serving_fast_policy import validate_fast_config',
        'from serving_request_factory import attach_source_check, from_prefill, sequential_captures':
            'from serving_request_factory import from_prefill',
    }
    for line, parent in imports.items():
        if lines.count(line) != 1:
            raise AssertionError('The C2-any import %r is not in serving_runtime.py exactly once' % line)
        lines = [parent if value == line else value for value in lines]
    starts = [index for index, value in enumerate(lines)
              if value.strip().startswith('# QWEN_FAST_ANY_REQUEST (C2-any, plan S1; default off)')]
    ends = [index for index, value in enumerate(lines) if value.strip() == 'attach_source_check()']
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise AssertionError('The C2-any attach hunk is not in serving_runtime.py exactly once')
    block = [value.strip() for value in lines[starts[0]:ends[0] + 1]]
    code = [value for value in block if not value.startswith('#')]
    if (code[0] != 'if any_request_enabled():' or code[1] != 'if not sequential_captures(capture_rows):'
            or not code[2].startswith("raise ValueError('QWEN_FAST_ANY_REQUEST=1 needs")
            or not all(value.startswith("'") for value in code[3:-1]) or code[-1] != 'attach_source_check()'):
        raise AssertionError('The C2-any attach hunk holds more than its guard: %r' % (code,))
    return without_s2_memory(without_no_block(lines[:starts[0]] + lines[ends[0] + 1:]))


def without_s2_memory(lines):
    """serving_runtime.py less S2 W6b's attach hunk (QWEN_FAST_EXTENT_REPLAY=1 only), which landed after this
    parent: its one import line and the one guarded registration of the DRAM admission hold, each cut exactly
    once and to its known last line, so nothing else is hidden."""
    line = 'from serving_request_factory import extent_replay_enabled, register_dram_admission'
    if lines.count(line) != 1:
        raise AssertionError('The S2 W6b import %r is not in serving_runtime.py exactly once' % line)
    lines = [value for value in lines if value != line]
    return cut_once(lines, '# S2 W6b (QWEN_FAST_EXTENT_REPLAY=1 only; unset, nothing is registered and the scheduler '
                           'admits exactly', 'scopes.callback(register_dram_admission(pool))')


def cut_once(lines, first, last, replacement=()):
    """lines less the one run from the line equal (stripped) to `first` through the next equal to
    `last`, with `replacement` in its place; exactly one such run, or AssertionError."""
    starts = [index for index, value in enumerate(lines) if value.strip() == first]
    if len(starts) != 1:
        raise AssertionError('%r is not in serving_runtime.py exactly once' % first)
    ends = [index for index in range(starts[0], len(lines)) if lines[index].strip() == last]
    if not ends:
        raise AssertionError('%r has no %r after it' % (first, last))
    return lines[:starts[0]] + list(replacement) + lines[ends[0] + 1:]


def without_no_block(lines):
    """serving_runtime.py less C2-any's no-block changes, which landed after the attach hunk (runs
    36218104858 and 36219636175): C2_ANY_SHAPE and c2_any_without_block, its paragraph and branch in
    register_reader_reason, the sequential-width cap with no block, and its own trim line. Each is
    cut exactly once and to its known last line, so nothing else is hidden."""
    lines = cut_once(lines, "C2_ANY_SHAPE = 'C2-any with no packed block'", "C2_ANY_SHAPE = 'C2-any with no packed block'")
    lines = cut_once(lines, 'def c2_any_without_block(environ=None):',
                     "return environ.get('QWEN_FAST_ANY_REQUEST') == '1' and environ.get('QWEN_FAST_PACKED_STEP', 'unset') != '1'")
    lines = cut_once(lines, 'The same holds for C2-any with no packed block (c2_any_without_block, the c2 profile): its',
                     '')   # the paragraph and the blank line it added before 'The policy is evaluated'
    lines = cut_once(lines, 'if not met and c2_any_without_block(environ):',
                     "return (C2_ANY_SHAPE, 'register-epilogue reader on native w_gate_up')")
    lines = cut_once(lines, '# C2-any with no packed block at all (the c2 profile sets QWEN_FAST_PACKED_STEP=0): its',
                     'capture_rows = M3_SEQUENTIAL_CAPTURE_ROWS')
    lines = cut_once(lines, 'if trimmed and no_block_any_request:', 'elif trimmed:', ['        if trimmed:'])
    lines = without_capture_position(lines)
    # The cuts leave the blank lines around the removed constant and function doubled.
    collapsed = []
    for value in lines:
        if value.strip() == '' and len(collapsed) >= 2 and collapsed[-1].strip() == '' and collapsed[-2].strip() == '':
            continue
        collapsed.append(value)
    return without_packed_any(collapsed)


def cut_guarded(lines, last):
    """lines less the one `if extent_replay:` block that ends in the line equal (stripped) to `last`: that
    guard, its comment lines and `last`, and nothing else; exactly one such block, or AssertionError."""
    ends = [index for index, value in enumerate(lines) if value.strip() == last]
    if len(ends) != 1:
        raise AssertionError('%r is not in serving_runtime.py exactly once' % last)
    start = ends[0] - 1
    while start >= 0 and lines[start].strip().startswith('#'):
        start -= 1
    if start < 0 or lines[start].strip() != 'if extent_replay:':
        raise AssertionError('%r is not guarded by `if extent_replay:` alone' % last)
    return lines[:start] + lines[ends[0] + 1:]


def cut_run(lines, run):
    """lines less the one contiguous run whose lines equal (stripped) `run`, in order; exactly one, or
    AssertionError."""
    stripped = [value.strip() for value in lines]
    starts = [index for index in range(len(lines) - len(run) + 1) if stripped[index:index + len(run)] == list(run)]
    if len(starts) != 1:
        raise AssertionError('%r is not in serving_runtime.py exactly once' % (run,))
    return lines[:starts[0]] + lines[starts[0] + len(run):]


def without_packed_any(lines):
    """serving_runtime.py less S2's W7 attach hunks (s2-design.md W7, after the no-block changes): the override's
    record kept for the admission, the packed-any admission block under QWEN_FAST_EXTENT_REPLAY, and the pool
    and block checks after each is built. Each is cut exactly once and to its known last line. W7's own refusal
    of the flag with no block and its pool keyword are the same hunks W3 landed (merged into W3's form, with
    W7's admission read after W3's strict one), so without_extent_replay cuts those, once."""
    kept = '    binary_record = override_runtime_binary(runtime_root, log=pindiag)'
    if lines.count(kept) != 1:
        raise AssertionError('%r is not in serving_runtime.py exactly once' % kept)
    lines = ['    override_runtime_binary(runtime_root, log=pindiag)' if value == kept else value for value in lines]
    lines = cut_run(lines, ('if extent_replay:', 'import packed_any_admission', '',
                            "packed_any_admission.extent_replay_enabled()   # strictly '1' from here: any other value "
                            'is refused',
                            'packed_any_admission.admit(runtime_root, m3=m3_shape(policy), binary_record=binary_record, '
                            'log=pindiag)'))
    lines = cut_guarded(lines, 'packed_any_admission.admit_pool(pool, log=pindiag)')
    return cut_guarded(lines, 'packed_any_admission.admit_blocks(packed_blocks, log=pindiag)')


def without_capture_position(lines):
    """serving_runtime.py less S2's gate-only capture-position knob (design W3, B1:
    QWEN_FAST_PACKED_CAPTURE_POSITION, G3b), which landed after C2-any: its `re` import, its two
    constants, packed_capture_position, the parse beside the padded admission, the override line
    and the block keyword. Each is cut exactly once and to its known last line, so nothing else is
    hidden."""
    lines = cut_once(lines, 'import re', 'import re')
    lines = cut_once(lines, "CAPTURE_POSITION_FLAG = 'QWEN_FAST_PACKED_CAPTURE_POSITION'",
                     "CAPTURE_POSITION_MARKER = '[PINDIAG] packed capture position override='")
    lines = cut_once(lines, 'def packed_capture_position(environ=None):', 'return int(text)')
    lines = cut_once(lines, '# QWEN_FAST_PACKED_CAPTURE_POSITION (S2 G3b, gate only, default unset): parsed here, before',
                     'capture_position = packed_capture_position()')
    lines = cut_once(lines, 'if capture_position is not None:',
                     "pindiag('{}{} (gate only)', CAPTURE_POSITION_MARKER, capture_position)")
    lines = cut_once(lines, "# S2 G3b's gate-only knob; unset, no keyword at all.",
                     'if capture_position is not None else {}),')
    return without_extent_replay(lines)


def without_extent_replay(lines):
    """serving_runtime.py less S2's QWEN_FAST_EXTENT_REPLAY plumbing (design W2/W3), which landed after the
    capture-position knob: its constant, extent_replay_requested, the read beside the knob's, the no-block
    refusal, the pool keyword and the check that every block took the flag. Each is cut exactly once and to
    its known last line, so nothing else is hidden."""
    lines = without_draft_masks(lines)
    lines = cut_once(lines, "EXTENT_REPLAY_FLAG = 'QWEN_FAST_EXTENT_REPLAY'", "EXTENT_REPLAY_FLAG = 'QWEN_FAST_EXTENT_REPLAY'")
    lines = cut_once(lines, 'def extent_replay_requested(environ=None):', "return value == '1'")
    lines = cut_once(lines, "# S2 C2-packed-any (QWEN_FAST_EXTENT_REPLAY, default off; strictly '0' or '1', read here "
                            'before anything', 'extent_replay = extent_replay_requested()')
    lines = cut_once(lines, '# S2 (QWEN_FAST_EXTENT_REPLAY=1) serves its rounds through the packed block alone - the pool '
                            'lends', "policy['scheduler_requests']))")
    replicas = "**({'packed_replicas': {distinct_shapes[0]: len(packed_shapes)}} if four_as_two else {})"
    lines = cut_once(lines, replicas + ',', "**({'extent_replay': True} if extent_replay else {}))))",
                     [' ' * 16 + replicas + ')))'])
    return cut_once(lines, '# S2: every block must be the extent block under the flag, and none may be without it. The',
                    '% (EXTENT_REPLAY_FLAG, int(extent_replay), extents))')


def without_draft_masks(lines):
    """serving_runtime.py less S2 M0's pooled draft masks (QWEN_FAST_EXTENT_REPLAY=1 only), which landed after the
    extent plumbing: the guarded read of the shapes and the pool keyword after the extent one. Each is cut exactly
    once and to its known last line, so nothing else is hidden."""
    lines = without_draft_outputs(lines)
    lines = cut_once(lines, '# S2: one mask per packed draft (each fixed pair, and the quad), allocated with the pool '
                            'before any trace,', "draft_masks = pooled_draft_mask_shapes(policy['scheduler_requests'], 16)")
    return cut_once(lines, "**({'extent_replay': True} if extent_replay else {}))),",
                    "**({'draft_masks': draft_masks} if draft_masks else {}))",
                    [' ' * 16 + "**({'extent_replay': True} if extent_replay else {}))))"])


def without_draft_outputs(lines):
    """serving_runtime.py less S2 v86's pooled draft outputs (QWEN_FAST_EXTENT_REPLAY=1 only), which landed after the
    pooled masks: the guarded read of the output shapes and the pool keyword after the masks' one. Each is cut
    exactly once and to its known last line, so nothing else is hidden."""
    lines = cut_once(lines, "# S2 v86 (run 36416471352): every traced draft's head outputs - each slot's single-user "
                            'draft, each fixed',
                     "draft_outputs = pooled_draft_output_shapes(policy['scheduler_requests'], 16)")
    return cut_once(lines, "**({'draft_masks': draft_masks} if draft_masks else {}),",
                    "**({'draft_outputs': draft_outputs} if draft_outputs else {}))",
                    [' ' * 12 + "**({'draft_masks': draft_masks} if draft_masks else {}))"])


class ShippingTests(unittest.TestCase):
    def test_the_module_reaches_the_image_in_both_lists(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        for name in ('fused_commit.py', 'packed_verifier.py', 'serving_packed_step.py', 'dflash_proposal_trace.py',
                     'verify_prestage.py', 'serving_worker_hook.py', 'serving_runtime.py', 'dflash_traced_publish.py',
                     'dflash_device.py'):
            with self.subTest(module=name):
                self.assertIn(name, dockerfile_modules(dockerfile_text()))
                self.assertIn(name, context_modules())

    def test_the_suite_runs_in_the_cpu_workflow(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertRegex(workflow, r'python -B -m unittest [^\n]*\btest_fused_commit\b')

    def test_the_pinned_sources_are_untouched(self):
        result = subprocess.run(['git', 'diff', '--name-only', PARENT, '--', 'draft_kv_slide.py', 'draft_kv_slide.cpp',
                                 'draft_kv_history.py', 'draft_kv_projection.py', 'draft_selector.py',
                                 'draft_head_layout.py', 'feature_collective.py', 'feature_projection.py',
                                 'attention_batch.py', 'gdn_multitoken_conv.py', 'draft_kv_slide_scope.py',
                                 'draft_kv_slide_gate.py', 'draft_kv_slide_adapter.py'],
                                capture_output=True, cwd=str(HERE), timeout=60)
        if result.returncode != 0:
            self.skipTest('no git history for %s' % PARENT)
        self.assertEqual(result.stdout.decode().strip(), '')


if __name__ == '__main__':
    unittest.main()
