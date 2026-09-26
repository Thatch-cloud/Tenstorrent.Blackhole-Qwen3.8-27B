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
        verify = names.index('PackedVerifierEngine.__init__.<locals>.<lambda>')
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
        after = without_any_request((HERE / 'serving_runtime.py').read_text(encoding='utf-8').splitlines())
        changed = [line for line in difflib.unified_diff(before, after, lineterm='', n=0)
                   if line[:1] in '+-' and not line.startswith(('+++', '---'))]
        added = [line[1:].strip() for line in changed if line.startswith('+')]
        removed = [line[1:].strip() for line in changed if line.startswith('-')]
        self.assertEqual(removed, ["if padded_min_users is not None else {}))"])
        self.assertEqual(added[0], "if padded_min_users is not None else {}),")
        self.assertIn("**({'collectives': collectives}", added)
        self.assertIn("if os.environ.get('QWEN_FAST_FUSED_COMMIT') == '1' else {}))", added)
        self.assertTrue(all(line.startswith('#') for line in added[1:] if 'collectives' not in line and 'FUSED' not in line))


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
    return without_no_block(lines[:starts[0]] + lines[ends[0] + 1:])


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
    # The cuts leave the blank lines around the removed constant and function doubled.
    collapsed = []
    for value in lines:
        if value.strip() == '' and len(collapsed) >= 2 and collapsed[-1].strip() == '' and collapsed[-2].strip() == '':
            continue
        collapsed.append(value)
    return without_packed_any(collapsed)


def without_packed_any(lines):
    """serving_runtime.py less S2's W7 attach hunks (s2-design.md W7, after the no-block changes): the override's
    record kept for the admission, the packed-any admission block under QWEN_FAST_EXTENT_REPLAY and the DRAM
    statistics check after the pool. Each is cut exactly once and to its known last line."""
    kept = '    binary_record = override_runtime_binary(runtime_root, log=pindiag)'
    if lines.count(kept) != 1:
        raise AssertionError('%r is not in serving_runtime.py exactly once' % kept)
    lines = ['    override_runtime_binary(runtime_root, log=pindiag)' if value == kept else value for value in lines]
    lines = cut_once(lines, "# S2 C2-packed-any (QWEN_FAST_EXTENT_REPLAY, default off; strictly '0' or '1'): packed "
                            'rounds at any',
                     'packed_any_admission.admit(runtime_root, m3=m3_shape(policy), binary_record=binary_record, '
                     'log=pindiag)')
    return cut_once(lines, 'if extent_replay:', 'packed_any_admission.admit_statistics(pool, log=pindiag)')


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
