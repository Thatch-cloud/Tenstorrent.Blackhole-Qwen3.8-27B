"""W3 MR and 2a (tp4_mesh_read): the verify readback as mesh-level reads, and the quad's top-16 outputs as one joined tensor.

Both levers are default off, strict 0/1, TP4 only, and move bytes only. What is held here, on the CPU:

  - the flag grammar (a value but 0 or 1 raises, the audit alone raises, the pair raises) and that with every flag unset nothing is imported
    and every readback, pool spec, pool report and log line is the one it always was;
  - MR: PackedVerifierEngine.shard_predictions reads two mesh-level tensors instead of eight per-chip parts, to the same per-chip host tensors
    (negative zero and NaN payloads included), the same combined ids, the chip-count refusal; the audit logs equal=1 and raises on an injected
    difference; the hostgap line says mr=1 only when on;
  - 2a: the pool spec grows a `joined` key for the quad groups only, allocated with the pool before any capture (after the group's other
    buffers), described only where present; publish_outputs adds the two concats and two copies only for a set that holds joined buffers;
    settle tries the join once and falls back to the chunks on any refusal; the quad readback from the joined tensors selects bit for bit what
    the chunk readback does, in the same chunk-major chip-minor order, with 9 reads (8 + one chip's features) instead of 20, 3 under the mesh
    lever too; the audit reads both ways and raises on a single flipped bit;
  - the smoke rule module (w3_mr_smoke) and the shipping lists.

    py -3.11 -B -m unittest test_tp4_w3_mr      (from scripts/ci)
"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import dflash_packed_proposal_coordinator as coordinator  # noqa: E402
import dflash_proposal_trace  # noqa: E402
import packed_verifier  # noqa: E402
import quad_draft_tp as twin  # noqa: E402
import serving_buffer_pool  # noqa: E402
import tp4_mesh_read  # noqa: E402
import tp_shapes  # noqa: E402
import w3_mr_smoke  # noqa: E402
from serving_buffer_pool import ServingBufferPool, validate_draft_outputs  # noqa: E402
from test_quad_draft_tp4 import HostOps4, four_chip_users  # noqa: E402
from test_serving_buffer_pool import FakeOperations  # noqa: E402
from tp_test_support import four_cards  # noqa: E402

MR, MR_AUDIT = tp4_mesh_read.MESH_READ_FLAG, tp4_mesh_read.MESH_READ_AUDIT_FLAG
JOINED, JOINED_AUDIT = tp4_mesh_read.JOINED_FLAG, tp4_mesh_read.JOINED_AUDIT_FLAG
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'


def environment(**flags):
    """No QWEN_FAST_* flag but the ones given, at four cards."""
    base = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    base.update({'QWEN_FAST_TP': '4'}, **flags)
    return patch.dict(os.environ, base, clear=True)


class FourCardEnvironment(object):
    """environment(...) plus the four-card seam serving_startup installs (tp_test_support.four_cards): the chunk tables, the vocabulary shard
    and the pooled shapes are the four-card ones (two 32,768-column chunks of a 62,080 shard)."""

    def __init__(self, **flags):
        self.flags = flags

    def __enter__(self):
        self.outer, self.inner = environment(**self.flags), four_cards()
        self.outer.__enter__()
        self.inner.__enter__()
        return self

    def __exit__(self, *exc):
        self.inner.__exit__(*exc)
        return self.outer.__exit__(*exc)


def captured_log():
    lines = []
    return lines, patch.object(tp4_mesh_read, 'log_line', lines.append)


# ---------------------------------------------------------------------------------------------
# Fakes: a mesh whose tensors hold one host tensor per chip.
# ---------------------------------------------------------------------------------------------

class MeshOps(HostOps4):
    """test_quad_draft_tp4's per-chip host ops, plus the mesh-level read, the concat and the copy. `log` records every read."""

    def __init__(self, corrupt_mesh=False):
        self.log = []
        self.corrupt_mesh = corrupt_mesh
        self.concats = self.copies = 0
        self.DRAM_MEMORY_CONFIG = 'dram'

    def to_torch(self, tensor, mesh_composer=None):
        if mesh_composer is None:
            self.log.append(('shard',))
            return tensor.values.clone()
        self.log.append(('mesh', mesh_composer.dim))
        parts = [part.clone() for part in tensor.chips]
        if self.corrupt_mesh:
            # one flipped bit in chip 2's part: a read that is not the per-chip read
            raw = parts[2].reshape(-1).view(torch.uint8)
            raw[0] = raw[0] ^ 1
        return torch.cat(parts, dim=mesh_composer.dim)

    def ConcatMeshToTensor(self, mesh, dim):
        return SimpleNamespace(mesh=mesh, dim=dim)

    def concat(self, pieces, dim, memory_config=None):
        self.concats += 1
        return Chips([torch.cat([piece.chips[chip] for piece in pieces], dim=dim) for chip in range(len(pieces[0].chips))],
                     pieces[0].dtype, pieces[0].layout)

    def copy(self, source, destination):
        self.copies += 1
        for target, value in zip(destination.chips, source.chips):
            target.copy_(value)

    def deallocate(self, tensor):
        pass

    def synchronize_device(self, mesh):
        pass

    @property
    def shard_reads(self):
        return sum(1 for entry in self.log if entry[0] == 'shard')

    @property
    def mesh_reads(self):
        return sum(1 for entry in self.log if entry[0] == 'mesh')


class Chips(SimpleNamespace):
    """A mesh tensor: `chips` (one host tensor per chip) with the shape, dtype and layout the join checks read."""

    def __init__(self, chips, dtype='bf16', layout='tile'):
        super().__init__(chips=chips, dtype=dtype, layout=layout)

    @property
    def shape(self):
        return tuple(self.chips[0].shape)


def as_chips(tensor, dtype):
    return Chips(list(tensor.chips), dtype)


def tensors_for(quad):
    """The `four_chip_users` quad's outputs with shape-aware tensors (values bf16, indices uint16, as the head leaves them)."""
    chunks = [dict(start=chunk['start'], stop=chunk['stop'], values=as_chips(chunk['values'], 'bf16'),
                   indices=as_chips(chunk['indices'], 'uint16')) for chunk in quad.chunks]
    return SimpleNamespace(chunks=chunks, projected=Chips(list(quad.projected.chips)))


def quad_outputs(seed, chips=4):
    """(singles, the quad's outputs as shape-aware tensors), built at four cards: two 32,768-column chunks of a 62,080-column shard."""
    with four_cards():
        singles, raw = four_chip_users(torch.Generator().manual_seed(seed), chips=chips)
    return singles, tensors_for(raw)


def joined_from(outputs):
    """The pooled joined tensors the traced pass would leave: the chunks' columns side by side, per chip."""
    return {key: Chips([torch.cat([chunk[key].chips[chip] for chunk in outputs.chunks], dim=3) for chip in range(4)],
                       outputs.chunks[0][key].dtype) for key in ('values', 'indices')}


def with_joined(outputs):
    found = SimpleNamespace(chunks=outputs.chunks, projected=outputs.projected)
    found.joined = joined_from(outputs)
    return found


def same_selection(left, right):
    for mine, theirs in zip(left, right):
        for key in ('hidden', 'candidates', 'unary'):
            if mine[key].dtype != theirs[key].dtype or not tp4_mesh_read.identical(mine[key], theirs[key]):
                return False
    return len(left) == len(right)


# ---------------------------------------------------------------------------------------------
# The flags.
# ---------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_unset_and_zero_are_off(self):
        for flags in ({}, {MR: '0', JOINED: '0'}):
            with environment(**flags):
                self.assertFalse(tp4_mesh_read.mesh_read_enabled())
                self.assertFalse(tp4_mesh_read.joined_enabled())
                self.assertFalse(tp4_mesh_read.mesh_read_audit_enabled())
                self.assertFalse(tp4_mesh_read.joined_audit_enabled())

    def test_one_is_on_and_anything_else_raises(self):
        with environment(**{MR: '1', JOINED: '1'}):
            self.assertTrue(tp4_mesh_read.mesh_read_enabled())
            self.assertTrue(tp4_mesh_read.joined_enabled())
        for name, function in ((MR, tp4_mesh_read.mesh_read_enabled), (JOINED, tp4_mesh_read.joined_enabled),
                               (MR_AUDIT, tp4_mesh_read.mesh_read_audit_enabled), (JOINED_AUDIT, tp4_mesh_read.joined_audit_enabled)):
            for value in ('2', 'true', 'yes', ''):
                with self.subTest(flag=name, value=value), environment(**dict({MR: '1', JOINED: '1'}, **{name: value})):
                    with self.assertRaises(ValueError):
                        function()

    def test_an_audit_without_its_lever_raises(self):
        with environment(**{MR_AUDIT: '1'}), self.assertRaises(ValueError):
            tp4_mesh_read.mesh_read_enabled()
        with environment(**{JOINED_AUDIT: '1'}), self.assertRaises(ValueError):
            tp4_mesh_read.joined_enabled()
        with environment(**{MR: '1', MR_AUDIT: '1'}):
            self.assertTrue(tp4_mesh_read.mesh_read_audit_enabled())
            self.assertFalse(tp4_mesh_read.joined_audit_enabled())

    def test_the_pair_raises(self):
        for name, function in ((MR, tp4_mesh_read.mesh_read_enabled), (JOINED, tp4_mesh_read.joined_enabled)):
            with self.subTest(flag=name), patch.dict(os.environ, {name: '1', 'QWEN_FAST_TP': '2'}):
                with self.assertRaisesRegex(ValueError, 'TP4 lever'):
                    function()

    def test_any_flag_set_sees_the_four(self):
        with environment():
            self.assertFalse(tp4_mesh_read.any_flag_set())
        for name in tp4_mesh_read.FLAGS:
            with environment(**{name: '1'}):
                self.assertTrue(tp4_mesh_read.any_flag_set())


# ---------------------------------------------------------------------------------------------
# MR: the verify readback.
# ---------------------------------------------------------------------------------------------

def bf16_bits(bits):
    return torch.tensor([bit - 65536 if bit >= 32768 else bit for bit in bits], dtype=torch.int16).view(torch.bfloat16)


def shard_engine(operations, ids, values, **extra):
    tensors = {'ids': ids, 'values': values}
    return SimpleNamespace(operations=operations, output=(None, 'ids', 'values', 'reference'), block_rows=3, shard_audit=False,
                           mesh='mesh', rounds=4, **extra)


class VerifyTensor(SimpleNamespace):
    """A sampler output as a mesh tensor: one host tensor per chip, addressed as the engine's output slots do."""


class VerifyOps(MeshOps):
    """MeshOps that answers get_device_tensors for the two output names the engine fake holds."""

    def __init__(self, ids, values, **options):
        super().__init__(**options)
        self.tensors = {'ids': VerifyTensor(chips=ids), 'values': VerifyTensor(chips=values)}

    def get_device_tensors(self, name):
        tensor = self.tensors[name] if isinstance(name, str) else name
        return [SimpleNamespace(values=value) for value in tensor.chips]

    def to_torch(self, tensor, mesh_composer=None):
        if isinstance(tensor, str):
            tensor = self.tensors[tensor]
        return super().to_torch(tensor, mesh_composer)


def sampler_parts():
    ids = [torch.tensor([1, 1, 1, 0], dtype=torch.int32), torch.tensor([2, 2, 2, 0], dtype=torch.int32),
           torch.tensor([3, 3, 3, 0], dtype=torch.int32), torch.tensor([4, 4, 4, 0], dtype=torch.int32)]
    values = [torch.tensor([1.0, 1.0, 5.0, 0.0]).bfloat16(), torch.tensor([2.0, 1.0, 1.0, 0.0]).bfloat16(),
              torch.tensor([1.0, 1.0, 1.0, 0.0]).bfloat16(), torch.tensor([1.0, 3.0, 1.0, 0.0]).bfloat16()]
    return ids, values


class MeshReadTests(unittest.TestCase):
    def setUp(self):
        patcher = environment()
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_engine(self, **extra):
        ids, values = sampler_parts()
        operations = VerifyOps(ids, values, corrupt_mesh=extra.pop('corrupt_mesh', False))
        engine = shard_engine(operations, 'ids', 'values', **extra)
        found = packed_verifier.PackedVerifierEngine.shard_predictions(engine)
        return found, operations, engine

    def test_off_reads_the_eight_parts_one_at_a_time_in_the_order_it_always_did(self):
        # an engine fake with no `mesh_read` attribute at all: test_tp4_shard_merges' shape
        found, operations, engine = self.run_engine()
        self.assertEqual(found, [2 + 62080, 4 + 3 * 62080, 1])
        self.assertEqual(operations.log, [('shard',)] * 8)
        self.assertEqual(engine.readback_split[0] >= 0, True)
        # ... and the same with the attribute present and False
        found2, operations2, _ = self.run_engine(mesh_read=False)
        self.assertEqual((found2, operations2.log), (found, operations.log))

    def test_on_it_reads_two_mesh_tensors_and_serves_the_same_ids(self):
        control, _, _ = self.run_engine()
        found, operations, engine = self.run_engine(mesh_read=True, mesh_read_audit=False)
        self.assertEqual(found, control)
        self.assertEqual(operations.log, [('mesh', 0), ('mesh', 0)], 'one read a tensor, composed along dim 0')
        self.assertEqual(operations.shard_reads, 0)

    def test_the_mesh_read_returns_the_loops_host_tensors_bit_for_bit(self):
        # negative zero, a NaN with a payload, an infinity, a subnormal: bf16 patterns the loop returns as they are
        patterns = [[0x8000, 0x7FC1, 0x7F80, 0x0001], [0x0000, 0xFFC2, 0xFF80, 0x8001], [0x3F80, 0xBF80, 0x7F7F, 0xFF7F], [0x0100, 0x8100, 0x4000, 0xC000]]
        values = [bf16_bits(bits) for bits in patterns]
        ids = [torch.tensor([chip, chip + 10, chip + 20, chip + 30], dtype=torch.int32) for chip in range(4)]
        operations = VerifyOps(ids, values)
        mesh_ids, mesh_values = tp4_mesh_read.read_predictions(operations, 'mesh', 'ids', 'values', 4, 4)
        loop_ids = [operations.to_torch(part).reshape(-1)[:4] for part in operations.get_device_tensors('ids')]
        loop_values = [operations.to_torch(part).reshape(-1)[:4] for part in operations.get_device_tensors('values')]
        for chip in range(4):
            self.assertTrue(tp4_mesh_read.identical(mesh_ids[chip], loop_ids[chip]))
            self.assertTrue(tp4_mesh_read.identical(mesh_values[chip], loop_values[chip]), 'chip %d' % chip)
        # the comparison is a bit comparison: -0 and +0, and two NaN payloads, are different
        self.assertFalse(tp4_mesh_read.identical(bf16_bits([0x8000]), bf16_bits([0x0000])))
        self.assertFalse(tp4_mesh_read.identical(bf16_bits([0x7FC1]), bf16_bits([0x7FC2])))
        self.assertFalse(tp4_mesh_read.identical(torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int64)))
        self.assertFalse(tp4_mesh_read.identical(torch.zeros(2), torch.zeros(3)))

    def test_per_chip_parts_are_not_mixed_up_when_every_chip_differs(self):
        values = [torch.full((1, 1, 1, 64), float(chip + 1)).bfloat16() for chip in range(4)]
        operations = VerifyOps([torch.full((1, 1, 1, 64), chip, dtype=torch.int32) for chip in range(4)], values)
        parts = tp4_mesh_read.chip_parts(operations, 'mesh', 'values', 4, True)
        for chip in range(4):
            self.assertEqual(float(parts[chip].reshape(-1)[0]), chip + 1.0)
            self.assertEqual(tuple(parts[chip].shape), (1, 1, 1, 64))

    def test_two_chips_are_refused_at_four_cards_on_both_routes(self):
        ids, values = sampler_parts()
        for mesh_read in (False, True):
            operations = VerifyOps(ids[:2], values[:2])
            engine = shard_engine(operations, 'ids', 'values', mesh_read=mesh_read, mesh_read_audit=False)
            with self.subTest(mesh_read=mesh_read), self.assertRaisesRegex(AssertionError, 'Four chip-local outputs required'):
                packed_verifier.PackedVerifierEngine.shard_predictions(engine)
            self.assertEqual(operations.log, [], 'refused before any read')

    def test_the_audit_logs_equal_and_raises_on_a_difference(self):
        lines, patcher = captured_log()
        with patcher:
            found, operations, _ = self.run_engine(mesh_read=True, mesh_read_audit=True)
        self.assertEqual(found, [2 + 62080, 4 + 3 * 62080, 1])
        self.assertEqual(lines, ['[PACKED-MESH-READ-AUDIT] round=5 chips=4 equal=1'])
        self.assertEqual(operations.mesh_reads, 2)
        self.assertEqual(operations.shard_reads, 8, 'the audit also runs the loop')
        lines, patcher = captured_log()
        with patcher, self.assertRaisesRegex(AssertionError, 'differs from the per-chip reads'):
            self.run_engine(mesh_read=True, mesh_read_audit=True, corrupt_mesh=True)
        self.assertEqual(len(lines), 1)
        self.assertIn('equal=0', lines[0])
        self.assertIn('ids.2', lines[0])

    def test_the_hostgap_line_says_mr_only_when_on(self):
        seen = []
        for attributes in ({}, {'mesh_read': False}, {'mesh_read': True}):
            engine = SimpleNamespace(prestaged=None, rounds=0, readback_split=(1.0, 2.0), **attributes)
            with patch.object(packed_verifier, 'diagnostic', seen.append):
                packed_verifier.PackedVerifierEngine.note_hostgap_verify(engine, (0, 1), None, 1.0, 2.0, 3.0, 4.0, 5.0)
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0], seen[1])
        self.assertTrue(seen[0].endswith('readback_cpu_ms=0.00'))
        self.assertEqual(seen[2], seen[0] + ' mr=1')

    def test_the_engaged_line_is_logged_only_when_on(self):
        lines, patcher = captured_log()
        with patcher:
            tp4_mesh_read.note_mesh_read_engaged(4)
        self.assertEqual(lines, ['[PINDIAG] tp4 verify mesh read engaged chips=4'])

    def test_off_nothing_is_imported_by_the_readback(self):
        stash = sys.modules.pop('tp4_mesh_read')
        try:
            ids, values = sampler_parts()
            engine = shard_engine(VerifyOps(ids, values), 'ids', 'values')
            packed_verifier.PackedVerifierEngine.shard_predictions(engine)
            engine.mesh_read = False
            packed_verifier.PackedVerifierEngine.shard_predictions(engine)
            self.assertNotIn('tp4_mesh_read', sys.modules)
        finally:
            sys.modules['tp4_mesh_read'] = stash

    def test_the_constructor_reads_the_flags_strictly_and_only_when_set(self):
        source = (HERE / 'packed_verifier.py').read_text(encoding='utf-8')
        self.assertIn("os.environ.get('QWEN_FAST_TP4_VERIFY_MESH_READ', '0') != '0'", source)
        self.assertIn('self.mesh_read = tp4_mesh_read.mesh_read_enabled() and self.shard_argmax', source)
        # a lever set on a verify that does not read back through the shard argmax says so
        self.assertIn('note_mesh_read_fell_back', source)


# ---------------------------------------------------------------------------------------------
# 2a: the pool spec and the pass.
# ---------------------------------------------------------------------------------------------

QUAD_ENV = {'QWEN_FAST_PACKED_PROPOSAL': '1', 'QWEN_FAST_QUAD_DRAFT': '1'}


class PoolSpecTests(unittest.TestCase):
    CHUNKS = ((0, 32768), (32768, 62080))

    def spec(self, rows, joined=None):
        found = dict(chunks=self.CHUNKS, head=(1, 1, rows, 16), projected=(1, 1, rows, 256))
        if joined is not None:
            found['joined'] = joined
        return found

    def test_the_keys_are_still_the_three_and_a_spec_without_joined_validates_to_what_it_was(self):
        self.assertEqual(serving_buffer_pool.DRAFT_OUTPUT_KEYS, ('chunks', 'head', 'projected'))
        checked = validate_draft_outputs({(0, 1, 2, 3): self.spec(64)}, 4)
        self.assertEqual(sorted(checked[(0, 1, 2, 3)]), ['chunks', 'head', 'projected'])

    def test_a_joined_shape_is_validated_like_the_others(self):
        checked = validate_draft_outputs({(0, 1, 2, 3): self.spec(64, (1, 1, 64, 32))}, 4)
        self.assertEqual(checked[(0, 1, 2, 3)]['joined'], (1, 1, 64, 32))
        for bad in ((1, 1, 64), (1, 1, 0, 32), (1, 1, 64, 'x'), 5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_draft_outputs({(0, 1, 2, 3): self.spec(64, bad)}, 4)
        with self.assertRaises(ValueError):
            validate_draft_outputs({(0, 1, 2, 3): dict(self.spec(64), other=(1, 1, 1, 1))}, 4)

    def test_pooled_shapes_are_unchanged_off_and_carry_joined_for_the_quad_groups_only(self):
        with FourCardEnvironment(**QUAD_ENV):
            off = coordinator.pooled_draft_output_shapes(4, 16)
        self.assertFalse(any('joined' in spec for spec in off.values()))
        with FourCardEnvironment(**QUAD_ENV, **{JOINED: '1'}):
            on = coordinator.pooled_draft_output_shapes(4, 16)
        self.assertEqual(sorted(on), sorted(off))
        self.assertEqual(on[(0, 1, 2, 3)]['joined'], (1, 1, 64, 32))
        for group, spec in on.items():
            if group != (0, 1, 2, 3):
                self.assertNotIn('joined', spec, group)
            self.assertEqual({key: value for key, value in spec.items() if key != 'joined'}, off[group])

    def test_the_quad_blocks_groups_carry_joined_too(self):
        with FourCardEnvironment(**QUAD_ENV, QWEN_FAST_QUAD_DRAFT_BLOCKS='2', QWEN_FAST_EXTENT_REPLAY='1', QWEN_FAST_M3_BLOCKS='2',
                                 **{JOINED: '1'}):
            try:
                on = coordinator.pooled_draft_output_shapes(8, 16)
            except ValueError as refusal:
                self.skipTest('the quad blocks refuse this environment here: %s' % refusal)
        self.assertEqual([group for group, spec in sorted(on.items()) if 'joined' in spec], [(0, 1, 2, 3), (4, 5, 6, 7)])

    def test_the_flag_at_the_pair_raises_at_attach(self):
        with patch.dict(os.environ, dict(QUAD_ENV, **{JOINED: '1', 'QWEN_FAST_TP': '2'})), self.assertRaises(ValueError):
            coordinator.pooled_draft_output_shapes(4, 16)


class PoolTests(unittest.TestCase):
    def outputs(self, joined):
        spec = PoolSpecTests().spec
        singles = {(slot,): spec(16) for slot in range(4)}
        return {**singles, (0, 1, 2, 3): spec(64, (1, 1, 64, 32) if joined else None)}

    def test_off_the_pool_is_what_it_was(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=4, draft_outputs=self.outputs(False))
        self.assertTrue(all(getattr(held, 'joined', None) is None for held in pool.draft_outputs.values()))
        report = pool.describe()
        self.assertFalse(any('joined' in entry for entry in report['draft_outputs']))

    def test_joined_buffers_are_allocated_with_the_pool_after_the_groups_other_buffers_for_the_quad_only(self):
        off, on = FakeOperations(), FakeOperations()
        ServingBufferPool(off, 'mesh', users=4, draft_outputs=self.outputs(False))
        pool = ServingBufferPool(on, 'mesh', users=4, draft_outputs=self.outputs(True))
        self.assertEqual(len(on.live), len(off.live) + 2)
        joined = pool.draft_outputs[(0, 1, 2, 3)].joined
        self.assertEqual((joined['values'].shape, joined['values'].dtype, joined['values'].layout), ((1, 1, 64, 32), 'bf16', 'tile'))
        self.assertEqual((joined['indices'].shape, joined['indices'].dtype), ((1, 1, 64, 32), 'uint16'))
        self.assertEqual(joined['indices'].mapper, ('replicate', 'mesh'))
        # right after the group's own chunks and projection (groups are allocated in sorted order, the quad's second), before the next group
        position = [index for index, tensor in enumerate(on.live) if tensor is joined['values']]
        self.assertEqual(len(position), 1)
        first = position[0]
        self.assertIs(on.live[first + 1], joined['indices'])
        self.assertIs(on.live[first - 1], pool.draft_outputs[(0, 1, 2, 3)].projected)
        self.assertEqual([tensor.shape for tensor in on.live[:first] + on.live[first + 2:]], [tensor.shape for tensor in off.live])
        tile = 32 * 32 * 2
        self.assertEqual(pool.draft_output_bytes - ServingBufferPool(FakeOperations(), 'mesh', users=4,
                                                                    draft_outputs=self.outputs(False)).draft_output_bytes, 2 * 2 * tile)
        for group, held in pool.draft_outputs.items():
            if group != (0, 1, 2, 3):
                self.assertIsNone(getattr(held, 'joined', None), group)
        entry = [entry for entry in pool.describe()['draft_outputs'] if entry['group'] == [0, 1, 2, 3]][0]
        self.assertEqual(entry['joined'], [1, 1, 64, 32])
        self.assertEqual(len(entry['joined_addresses']), 2)
        others = [entry for entry in pool.describe()['draft_outputs'] if entry['group'] != [0, 1, 2, 3]]
        self.assertFalse(any('joined' in entry for entry in others))

    def test_a_refused_set_takes_its_joined_buffers_with_it(self):
        class Refusing(FakeOperations):
            def from_torch(self, value, **options):
                if options.get('dtype') == self.uint16:
                    raise RuntimeError('no uint16')
                return super().from_torch(value, **options)

        pool = ServingBufferPool(Refusing(), 'mesh', users=4, draft_outputs=self.outputs(True))
        self.assertEqual(pool.draft_outputs, {})
        self.assertTrue(pool.draft_outputs_refused)


class PassTests(unittest.TestCase):
    """publish_outputs, settle and traced_pass over a quad pass's outputs."""

    def quad(self):
        return quad_outputs(31)[1]

    def pooled(self, outputs, joined=True):
        chunks = [dict(start=chunk['start'], stop=chunk['stop'],
                       values=Chips([torch.zeros_like(part) for part in chunk['values'].chips], 'bf16'),
                       indices=Chips([torch.zeros_like(part) for part in chunk['indices'].chips], 'uint16')) for chunk in outputs.chunks]
        held = SimpleNamespace(projected=Chips([torch.zeros_like(part) for part in outputs.projected.chips]), chunks=chunks)
        if joined:
            held.joined = {key: Chips([torch.zeros(1, 1, 64, 32, dtype=outputs.chunks[0][key].chips[0].dtype) for _ in range(4)],
                                      outputs.chunks[0][key].dtype) for key in ('values', 'indices')}
        return held

    def test_a_set_without_joined_publishes_exactly_as_before(self):
        outputs = self.quad()
        operations = MeshOps()
        fresh = dflash_proposal_trace.publish_outputs(operations, outputs, self.pooled(outputs, joined=False))
        self.assertEqual(operations.copies, 2 * 2 + 1)
        self.assertEqual(operations.concats, 0)
        self.assertFalse(hasattr(fresh, 'joined'))

    def test_a_set_with_joined_adds_two_concats_and_two_copies_and_the_join_is_the_chunks_columns(self):
        outputs = self.quad()
        held = self.pooled(outputs)
        operations = MeshOps()
        fresh = dflash_proposal_trace.publish_outputs(operations, outputs, held)
        self.assertEqual((operations.concats, operations.copies), (2, 2 * 2 + 1 + 2))
        self.assertIs(fresh.joined['values'], held.joined['values'])
        self.assertEqual(len(fresh.joined['temporaries']), 2)
        for key in ('values', 'indices'):
            for chip in range(4):
                for number, chunk in enumerate(outputs.chunks):
                    self.assertTrue(torch.equal(held.joined[key].chips[chip][..., 16 * number:16 * (number + 1)], chunk[key].chips[chip]))

    def test_the_join_allocates_no_persistent_buffer(self):
        outputs = self.quad()
        held = self.pooled(outputs)

        class Counting(MeshOps):
            allocations = 0

            def empty(self, *args, **options):
                Counting.allocations += 1

            from_torch = zeros_like = empty

        operations = Counting()
        tp4_mesh_read.publish_joined(operations, outputs, held)
        self.assertEqual(Counting.allocations, 0)

    def test_settle_engages_once_and_logs(self):
        tp4_mesh_read.reset_state()
        outputs = self.quad()
        held = self.pooled(outputs)
        operations = MeshOps()
        lines, patcher = captured_log()
        with patcher:
            settled = tp4_mesh_read.settle(operations, outputs, held, [0, 1, 2, 3])
        self.assertIs(settled, held)
        self.assertEqual(lines, ['[PINDIAG] tp4 quad outputs joined engaged quads=1 slots=[0,1,2,3] chunks=2 width=32'])
        self.assertEqual(operations.concats, 2)

    def test_settle_falls_back_to_the_chunks_on_a_refused_concat(self):
        outputs = self.quad()
        held = self.pooled(outputs)

        class Refusing(MeshOps):
            def concat(self, pieces, dim, memory_config=None):
                raise RuntimeError('uint16 TILE concat unsupported')

        lines, patcher = captured_log()
        with patcher:
            settled = tp4_mesh_read.settle(Refusing(), outputs, held, [0, 1, 2, 3])
        self.assertIsNot(settled, held)
        self.assertIsNone(getattr(settled, 'joined', None))
        self.assertIs(settled.projected, held.projected)
        self.assertEqual([chunk['values'] for chunk in settled.chunks], [chunk['values'] for chunk in held.chunks])
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('[PINDIAG] tp4 quad outputs joined fell back slots=[0,1,2,3] reason=join refused: RuntimeError'))
        # and the capture then publishes the chunks alone
        fresh = dflash_proposal_trace.publish_outputs(MeshOps(), outputs, settled)
        self.assertFalse(hasattr(fresh, 'joined'))

    def test_settle_without_a_pooled_set_or_with_a_mismatch_falls_back_with_a_reason(self):
        outputs = self.quad()
        lines, patcher = captured_log()
        with patcher:
            self.assertIsNone(tp4_mesh_read.settle(MeshOps(), outputs, None, [0, 1, 2, 3]))
            held = self.pooled(outputs)
            held.joined['values'] = Chips([torch.zeros(1, 1, 64, 16) for _ in range(4)], 'bf16')
            settled = tp4_mesh_read.settle(MeshOps(), outputs, held, [0, 1, 2, 3])
            wrong_dtype = self.pooled(outputs)
            wrong_dtype.joined['indices'] = Chips(wrong_dtype.joined['indices'].chips, 'int32')
            tp4_mesh_read.settle(MeshOps(), outputs, wrong_dtype, [0, 1, 2, 3])
            no_joined = self.pooled(outputs, joined=False)
            tp4_mesh_read.settle(MeshOps(), outputs, no_joined, [0, 1, 2, 3])
        self.assertIsNone(getattr(settled, 'joined', None))
        self.assertEqual(len(lines), 4)
        self.assertIn('QWEN_FAST_EXTENT_REPLAY=1', lines[0])
        self.assertIn('shape', lines[1])
        self.assertIn('dtype', lines[2])
        self.assertIn('no joined buffer', lines[3])
        self.assertTrue(all(line.startswith(tp4_mesh_read.JOINED_FELL_BACK) for line in lines))

    def test_traced_pass_returns_outputs_the_readback_reads_joined(self):
        outputs = self.quad()
        held = self.pooled(outputs)
        operations = MeshOps()
        recorded = dflash_proposal_trace.traced_pass(operations, lambda: outputs, held)
        with environment(**{JOINED: '1'}):
            levers = tp4_mesh_read.quad_levers(recorded)
        self.assertTrue(levers.joined)
        self.assertEqual(dflash_proposal_trace.traced_pass(operations, lambda: outputs, None), outputs)


# ---------------------------------------------------------------------------------------------
# 2a: the quad readback.
# ---------------------------------------------------------------------------------------------

class ReadbackTests(unittest.TestCase):
    def setUp(self):
        self.singles, self.outputs = quad_outputs(41)
        self.device = lambda operations: SimpleNamespace(operations=operations, mesh='mesh')

    def read(self, outputs, **flags):
        operations = MeshOps()
        with environment(**flags), four_cards():
            parts = twin.read_quad_outputs(self.device(operations), outputs)
        return parts, operations

    def test_off_the_reads_are_the_loops_in_the_order_it_always_made_them(self):
        calls = []

        class Logging(MeshOps):
            def get_device_tensors(self, tensor):
                calls.append(('parts', id(tensor)))
                return super().get_device_tensors(tensor)

            def to_torch(self, tensor, mesh_composer=None):
                calls.append(('read',))
                return super().to_torch(tensor, mesh_composer)

        operations = Logging()
        with environment(), four_cards():
            twin.read_quad_outputs(self.device(operations), self.outputs)
        expected = []
        for chunk in self.outputs.chunks:
            expected += [('parts', id(chunk['values'])), ('parts', id(chunk['indices']))]
            expected += [('read',), ('read',)] * 4
        expected += [('parts', id(self.outputs.projected))] + [('read',)] * 4
        self.assertEqual(calls, expected)
        self.assertEqual(operations.shard_reads, 2 * 4 * 2 + 4)

    def test_the_joined_readback_selects_bit_for_bit_what_the_chunk_readback_does_with_nine_reads(self):
        control, loop = self.read(self.outputs)
        joined, operations = self.read(with_joined(self.outputs), **{JOINED: '1'})
        self.assertTrue(same_selection(joined, control))
        self.assertEqual((loop.shard_reads, loop.mesh_reads), (20, 0))
        self.assertEqual((operations.shard_reads, operations.mesh_reads), (4 * 2 + 1, 0), 'values and indices from each chip, one chip of features')

    def test_with_the_mesh_lever_too_it_is_three_reads(self):
        control, _ = self.read(self.outputs)
        both, operations = self.read(with_joined(self.outputs), **{JOINED: '1', MR: '1'})
        self.assertTrue(same_selection(both, control))
        self.assertEqual((operations.shard_reads, operations.mesh_reads), (1, 2))

    def test_the_mesh_lever_alone_reads_the_chunks_in_five_mesh_reads(self):
        control, _ = self.read(self.outputs)
        mesh, operations = self.read(self.outputs, **{MR: '1'})
        self.assertTrue(same_selection(mesh, control))
        self.assertEqual((operations.shard_reads, operations.mesh_reads), (0, 2 * 2 + 1), '2 chunks x (values, indices), and the features')

    def test_the_joined_lever_on_a_build_that_fell_back_reads_the_chunks(self):
        control, _ = self.read(self.outputs)
        found, operations = self.read(self.outputs, **{JOINED: '1'})
        self.assertTrue(same_selection(found, control))
        self.assertEqual(operations.shard_reads, 20)

    def test_the_replicated_feature_guard_still_runs_on_every_chip_when_it_reads_them(self):
        broken = quad_outputs(41)[1]
        broken.projected.chips[3][0, 0, 5, 3] += 1
        with self.assertRaisesRegex(AssertionError, 'Replicated learned selector features differ'):
            self.read(broken)
        with self.assertRaisesRegex(AssertionError, 'Replicated learned selector features differ'):
            self.read(broken, **{MR: '1'})
        # under the joined lever without its audit the guard reads chip 0 alone (the design's 2a); the audit restores all four
        self.read(with_joined(broken), **{JOINED: '1'})
        with self.assertRaisesRegex(AssertionError, 'Replicated learned selector features differ'):
            self.read(with_joined(broken), **{JOINED: '1', JOINED_AUDIT: '1'})

    def test_two_chips_of_a_head_are_refused_at_four_cards_on_every_route(self):
        two = quad_outputs(42, chips=2)[1]
        for flags in ({}, {MR: '1'}):
            with self.subTest(flags=flags), self.assertRaises(AssertionError):
                self.read(two, **flags)

    def test_the_audit_logs_equal_and_counts_rounds(self):
        lines, patcher = captured_log()
        with patcher:
            _, plain = self.read(with_joined(self.outputs), **{JOINED: '1', JOINED_AUDIT: '1'})
            _, meshed = self.read(with_joined(self.outputs), **{JOINED: '1', JOINED_AUDIT: '1', MR: '1'})
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].endswith('chips=4 chunks=2 equal=1'))
        self.assertTrue(lines[1].startswith('[QUAD-JOINED-AUDIT] round='))
        # the joined read (8), the chunk reads the old way (16) and all four chips of features (4)
        self.assertEqual((plain.shard_reads, plain.mesh_reads), (8 + 16 + 4, 0))
        # with the mesh lever: the joined read and the features are mesh reads, the audit's chunk loop stays the loop
        self.assertEqual((meshed.shard_reads, meshed.mesh_reads), (16, 2 + 1))

    def test_the_audit_raises_on_one_flipped_bit_in_one_chip_of_the_joined_tensor(self):
        outputs = with_joined(self.outputs)
        raw = outputs.joined['values'].chips[1].reshape(-1).view(torch.uint8)
        raw[3] = raw[3] ^ 4
        lines, patcher = captured_log()
        with patcher, self.assertRaisesRegex(AssertionError, 'joined read differs from the chunk reads'):
            self.read(outputs, **{JOINED: '1', JOINED_AUDIT: '1'})
        self.assertIn('equal=0', lines[-1])
        self.assertIn('chip=1', lines[-1])
        # and a flipped index
        outputs = with_joined(self.outputs)
        outputs.joined['indices'].chips[3][0, 0, 7, 20] += 1
        with patcher, self.assertRaises(AssertionError):
            self.read(outputs, **{JOINED: '1', JOINED_AUDIT: '1'})

    def test_the_joined_hosts_are_the_chunk_hosts_in_the_same_order(self):
        outputs = with_joined(self.outputs)
        operations = MeshOps()
        mine = tp4_mesh_read.joined_hosts(operations, None, outputs, 4, 64, False)
        theirs = tp4_mesh_read.chunk_hosts(operations, None, outputs.chunks, 4, 64, False)
        self.assertEqual([(chunk['start'], chip) for chunk, chip, _, _ in mine], [(chunk['start'], chip) for chunk, chip, _, _ in theirs])
        for (_, _, value, index), (_, _, other_value, other_index) in zip(mine, theirs):
            self.assertTrue(tp4_mesh_read.identical(value, other_value) and tp4_mesh_read.identical(index, other_index))
            self.assertEqual((value.dtype, index.dtype), (torch.float32, torch.int64))

    def test_a_random_pass_of_many_seeds_selects_the_same(self):
        for seed in range(50, 56):
            outputs = quad_outputs(seed)[1]
            control, _ = self.read(outputs)
            found, _ = self.read(with_joined(outputs), **{JOINED: '1'})
            self.assertTrue(same_selection(found, control), seed)


# ---------------------------------------------------------------------------------------------
# The smoke rule.
# ---------------------------------------------------------------------------------------------

ENGAGED_MR = '[PINDIAG] tp4 verify mesh read engaged chips=4'
ENGAGED_JOINED = '[PINDIAG] tp4 quad outputs joined engaged quads=1 slots=[0,1,2,3] chunks=2 width=32'
GOOD_MR_AUDIT = '[PACKED-MESH-READ-AUDIT] round=1 chips=4 equal=1'
GOOD_JOINED_AUDIT = '[QUAD-JOINED-AUDIT] round=1 chips=4 chunks=2 equal=1'
VERIFY_LINE = '[PACKED-HOSTGAP-VERIFY] block=A round=3 live=4 path=diff reason=- reads_ms=0.20 mr=1'


class SmokeTests(unittest.TestCase):
    def test_a_profile_without_the_flags_must_show_no_lines(self):
        self.assertEqual(w3_mr_smoke.problems({}, 'ordinary log\n'), [])
        self.assertEqual(w3_mr_smoke.problems(None, ''), [])
        for line in (ENGAGED_MR, ENGAGED_JOINED, GOOD_MR_AUDIT, GOOD_JOINED_AUDIT, VERIFY_LINE):
            with self.subTest(line=line):
                self.assertEqual(len(w3_mr_smoke.problems({}, line)), 1)

    def test_mr_needs_its_engaged_line_and_mr_on_the_hostgap_lines(self):
        env = {MR: '1', 'QWEN_FAST_TP4_HOSTGAP_LOG': '1'}
        self.assertEqual(len(w3_mr_smoke.problems(env, '')), 1)
        self.assertEqual(w3_mr_smoke.problems(env, '\n'.join([ENGAGED_MR, VERIFY_LINE])), [])
        problems = w3_mr_smoke.problems(env, '\n'.join([ENGAGED_MR, VERIFY_LINE.replace(' mr=1', '')]))
        self.assertEqual(len(problems), 1)
        self.assertIn('reads_ms', problems[0])
        self.assertEqual(w3_mr_smoke.problems({MR: '1'}, ENGAGED_MR), [], 'no hostgap log, nothing to hold')

    def test_the_mr_audit_must_have_run_and_be_equal(self):
        env = {MR: '1', MR_AUDIT: '1'}
        self.assertEqual(len(w3_mr_smoke.problems(env, ENGAGED_MR)), 1)
        self.assertEqual(w3_mr_smoke.problems(env, '\n'.join([ENGAGED_MR, GOOD_MR_AUDIT])), [])
        bad = '[PACKED-MESH-READ-AUDIT] round=2 chips=4 equal=0 differing=ids.2'
        problems = w3_mr_smoke.problems(env, '\n'.join([ENGAGED_MR, GOOD_MR_AUDIT, bad]))
        self.assertEqual(len(problems), 1)
        self.assertIn('found a difference', problems[0])
        self.assertEqual(len(w3_mr_smoke.problems({MR_AUDIT: '1'}, '')), 1, 'the audit without the lever')

    def test_joined_needs_one_engaged_line_a_quad_block_and_its_audit(self):
        env = {JOINED: '1'}
        self.assertEqual(len(w3_mr_smoke.problems(env, '')), 1)
        self.assertEqual(w3_mr_smoke.problems(env, ENGAGED_JOINED), [])
        two = dict(env, QWEN_FAST_QUAD_DRAFT_BLOCKS='2')
        self.assertEqual(len(w3_mr_smoke.problems(two, ENGAGED_JOINED)), 1)
        self.assertEqual(w3_mr_smoke.problems(two, '\n'.join([ENGAGED_JOINED, ENGAGED_JOINED.replace('0,1,2,3', '4,5,6,7')])), [])
        audited = dict(env, **{JOINED_AUDIT: '1'})
        self.assertEqual(len(w3_mr_smoke.problems(audited, ENGAGED_JOINED)), 1)
        self.assertEqual(w3_mr_smoke.problems(audited, '\n'.join([ENGAGED_JOINED, GOOD_JOINED_AUDIT])), [])
        problems = w3_mr_smoke.problems(audited, '\n'.join([ENGAGED_JOINED, GOOD_JOINED_AUDIT, GOOD_JOINED_AUDIT.replace('equal=1', 'equal=0')]))
        self.assertEqual(len(problems), 1)

    def test_a_fall_back_is_a_problem_on_every_arm(self):
        for line in ('[PINDIAG] tp4 verify mesh read fell back reason=x', '[PINDIAG] tp4 quad outputs joined fell back slots=[0,1,2,3] reason=y'):
            with self.subTest(line=line):
                self.assertTrue(any('fell back' in problem for problem in w3_mr_smoke.problems({}, line)))
                self.assertTrue(any('fell back' in problem for problem in w3_mr_smoke.problems({MR: '1', JOINED: '1'}, line)))


# ---------------------------------------------------------------------------------------------
# Shipping.
# ---------------------------------------------------------------------------------------------

class ShippingTests(unittest.TestCase):
    def test_the_module_is_in_the_overlay_and_the_test_is_in_the_cpu_workflow(self):
        overlay = (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').splitlines()
        self.assertIn('scripts/ci/tp4_mesh_read.py', overlay)
        self.assertNotIn('scripts/ci/w3_mr_smoke.py', overlay, 'host side, read by the smoke check on the runner')
        self.assertIn('test_tp4_w3_mr', CPU_WORKFLOW.read_text(encoding='utf-8'))

    def test_the_modules_are_stdlib_only_at_import_and_lf_only(self):
        for name in ('tp4_mesh_read.py', 'w3_mr_smoke.py'):
            data = (HERE / name).read_bytes()
            self.assertNotIn(b'\r', data, name)
            head = data.decode().split('\ndef ', 1)[0]
            self.assertNotIn('import torch', head, name)
            self.assertNotIn('import ttnn', head, name)

    def test_the_flags_are_in_no_profile_yet(self):
        # the parent adds the gate-only profiles; until then, and in every traffic profile after, none carries a lever flag
        import json

        found = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
        for name, profile in found.items():
            if not profile.get('gate_only'):
                with self.subTest(profile=name):
                    self.assertFalse(set(tp4_mesh_read.FLAGS) & set(profile.get('env', {})))


if __name__ == '__main__':
    unittest.main()
