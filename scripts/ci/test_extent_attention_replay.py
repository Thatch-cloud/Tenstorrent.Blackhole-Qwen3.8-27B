"""S2 extent replay readers (extent_attention_replay.py), on a fake two-chip device.

The design's W1 CPU tests 1-11 (s2-design.md section 4): the helpers and their host mirror of
the pinned mask kernel, the capacity-256 prepare, construction staging, staging, the shared-mask
budget, what every SDPA call carries, the refusals, and that nothing pinned is subclassed or
changed. Plus the seam with serving_buffer_pool's extent storage (W2): the pool lends exactly
what the readers take.
"""

import ast
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import attention_mask_replay
import extent_attention_replay
from attention_head_fold import causal_mask, parallel_groups
from extent_attention_replay import (EXTENT_FLAGS, F22_MARKER, K, LAYOUT, MIN_LIVE_START, ExtentSegmentReader,
                                     PackedExtentReplayReader, accept_limit, admits, extent, extent_values,
                                     mask_tiles, narrow_mask_host, prepare_narrow, replay_mask_host)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ENV = {'QWEN_FAST_SDPA_MODES': 'tail,share,slice', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}
C = 131328           # the served capacity: page width 2052
WIDTH = C // 64
QWEN_DECODE_MAGIC = 0x51DEC000

# attention_mask_replay.py exists in three versions, and test 11 (StructureTests) tells them apart:
#   FROZEN_MASK    the frozen recipe's revision (frozen_recipe_context.REVISION, 8c102b20). The frozen T16 and DSpark
#                  reports in this tree pin it (target-t16-attention-simulator.json, dspark-request-hardware.json).
#   SERVED_MASK    the version every image SERVES. The frozen recipe stages FROZEN_MASK through
#                  frozen_target_replay.adapt_target_mask, which lets validate_ticket also admit the selected
#                  context's capacity. serving_bundle.py packages that staged tree (read-combined), and P8's
#                  /experiment-scripts/ci is the bundle. frozen-evidence/target-replay.json (92c51875,
#                  frozen_combined_gate.REPORTS) pins it, so frozen_combined_runtime.qualify hashes it at every
#                  attach, and c2_overlay's install refuses any other bytes there. The bundle's own manifest
#                  (artifact qwen-fast-serving-bundle-35489235797, serving-bundle.json, tar 0ee04c47) records it for
#                  experiment-scripts/ci and frozen-evidence/target, and the C2 build v51 (run 36255706983) found it
#                  in the image: 6a31981c.
#   CHECKOUT_MASK  this checkout's copy: FROZEN_MASK plus 29b43224's simulator-only context ladder. No image carries
#                  it: neither copy list names the file, and the overlay install would refuse it.
# The three differ only in validate_ticket and one import. prepare, mask_position, execute and source_hashes are the
# same bytes in all three, and so is attention_mask_replay.cpp (e10cae1d).
FROZEN_REVISION = '8c102b20df22329106955b4006bf4d650bb94e40'
FROZEN_MASK = '7841495a15ee090aae7b78edc118ba0de2967bb3ad72b843d53251a091435749'
SERVED_MASK = '6a31981cb9203439b8e6e78e8bd712e078f00335a7f211c732d4ac47cabca3a8'
CHECKOUT_MASK = '3e431742e35a2b94b4a02a60fa334a93a44a471eaefcacd25e52fbafdf03361f'


def in_checkout():
    """True in a git checkout. False in an image, whose /experiment-scripts/ci has no repository above it."""
    return (ROOT / '.git').exists()


def frozen_mask_source():
    """attention_mask_replay.py at FROZEN_REVISION, from git. It raises in a checkout that cannot reach the revision
    (a shallow clone; CI fetches the full history), so no check built on it passes vacuously."""
    import subprocess

    result = subprocess.run(['git', '-C', str(ROOT), 'show',
                             '%s:scripts/ci/attention_mask_replay.py' % FROZEN_REVISION], capture_output=True)
    if result.returncode:
        raise AssertionError('frozen revision %s is unreachable from %s (a shallow clone?): %s'
                             % (FROZEN_REVISION[:8], ROOT, result.stderr.decode('utf-8', 'replace').strip()))
    return result.stdout.decode('utf-8')


def served_mask_source():
    """attention_mask_replay.py as the image serves it. In an image, this is the file beside this test, which is the
    served tree itself. In a checkout, it is rebuilt exactly as frozen_recipe_context stages it: FROZEN_REVISION's
    bytes through frozen_target_replay.adapt_target_mask. The recipe's other adapters leave this file alone, and the
    tests hold the rebuild to SERVED_MASK."""
    if not in_checkout():
        return (HERE / 'attention_mask_replay.py').read_bytes().decode('utf-8')
    from frozen_target_replay import adapt_target_mask
    return adapt_target_mask(frozen_mask_source())


def top_level(source):
    """A module's top level: each function's and class's source by name, plus 'imports' (the set of its import
    statements), 'docstring' and 'statements' (anything else, in order)."""
    tree = ast.parse(source)
    found = dict(imports=frozenset(ast.get_source_segment(source, node) for node in tree.body
                                   if isinstance(node, (ast.Import, ast.ImportFrom))),
                 docstring=ast.get_docstring(tree), statements=[])
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[node.name] = ast.get_source_segment(source, node)
        elif not isinstance(node, (ast.Import, ast.ImportFrom)) and not (index == 0 and found['docstring'] is not None):
            found['statements'].append(ast.get_source_segment(source, node))
    found['statements'] = tuple(found['statements'])
    return found


def function_body(source, name):
    """(argument names, body) of the top-level function `name`: the body's lines as written, without its docstring."""
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    arguments = [argument.arg for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    return arguments, ''.join(source.splitlines(keepends=True)[body[0].lineno - 1:node.end_lineno])


def structure(bundles):
    return [[(group['offset'], group['rows']) for group in bundle] for bundle in bundles]


class FakeShard:
    def __init__(self, address):
        self.address = address

    def buffer_address(self):
        return self.address


class FakeTensor:
    def __init__(self, shape, dtype, layout, memory, shards, value=None, name='', origin=None):
        self.shape, self.dtype, self.layout, self.memory = tuple(shape), dtype, layout, memory
        self.shards, self.value, self.name = shards, value, name
        # What made it: ('slice', source, start, stop), ('concat', parts), ('fold', source, inverse,
        # offset) or ('sdpa', query, options), so a test can walk an output back to its rows.
        self.origin = origin

    @property
    def padded_shape(self):
        return self.shape

    def memory_config(self):
        return self.memory


class FakeDevice:
    """Two chips with independent bump allocators; device tensors hold a host copy of their value, so
    a test reads back what the host staged into them. Slices, concats and SDPA results record their
    inputs (FakeTensor.origin), and `events` orders the SDPA calls against the mask refreshes a test
    records there."""

    int32, uint32, bfloat16 = 'int32', 'uint32', 'bf16'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'row_major', 'tile', 'dram'

    def __init__(self):
        self.next = [0x1000, 0x800000]
        self.live, self.deallocated, self.copies, self.sdpa, self.events = [], [], [], [], []
        self.fences = 0
        self.copy_hook = None
        self.transformer = SimpleNamespace(paged_scaled_dot_product_attention_decode=self.attention)

    def device_tensor(self, shape, dtype=None, layout=None, memory='dram', value=None, name='', origin=None):
        shards = []
        for chip in range(2):
            shards.append(FakeShard(self.next[chip]))
            self.next[chip] += 0x100
        tensor = FakeTensor(shape, dtype, layout, memory, shards, value, name, origin)
        self.live.append(tensor)
        return tensor

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        if device is None:
            return FakeTensor(value.shape, dtype, layout, None, [], value.clone(), 'host')
        # The pool's history and K/V banks are large and never read back here.
        held = value.clone() if value.numel() <= 1 << 20 else None
        return self.device_tensor(value.shape, dtype, layout, memory_config, held, 'upload')

    def copy_host_to_device_tensor(self, source, destination):
        if tuple(source.shape) != tuple(destination.shape):
            raise AssertionError('copy of %r into %r' % (source.shape, destination.shape))
        if self.copy_hook is not None:
            self.copy_hook(source, destination)
        destination.value = source.value.clone()
        self.copies.append(destination)

    def synchronize_device(self, mesh):
        self.fences += 1

    def get_device_tensors(self, tensor):
        return tensor.shards

    def deallocate(self, tensor):
        if any(tensor is value for value in self.deallocated):
            raise AssertionError('Double free')
        self.deallocated.append(tensor)

    def full_like(self, tensor, value, *, optional_tensor):
        pass

    @staticmethod
    def SDPAProgramConfig(**options):
        return SimpleNamespace(**options)

    @staticmethod
    def ReplicateTensorToMesh(mesh):
        return ('replicate', mesh)

    @staticmethod
    def ShardTensorToMesh(mesh, dim):
        return ('shard', mesh, dim)

    def slice(self, tensor, start, stop, memory_config):
        return self.device_tensor(tuple(b - a for a, b in zip(start, stop)), 'bf16', 'tile', memory_config,
                                  name='slice%d' % start[1], origin=('slice', tensor, tuple(start), tuple(stop)))

    def concat(self, parts, dim, memory_config):
        shape = list(parts[0].shape)
        shape[dim] = sum(part.shape[dim] for part in parts)
        return self.device_tensor(shape, 'bf16', 'tile', memory_config, name='concat', origin=('concat', tuple(parts)))

    def attention(self, query, keys, values, **options):
        self.sdpa.append(options)
        self.events.append(('sdpa', options['attn_mask']))
        return self.device_tensor(query.shape, 'bf16', 'tile', options['memory_config'], name='sdpa',
                                  origin=('sdpa', query, options))


MESH = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))


def lend(device, rows=16, width=WIDTH, bundles=None):
    """What serving_buffer_pool lends one segment: per bundle a (B, width) table and a (B,) cur_pos."""
    bundles = LAYOUT(rows, 8) if bundles is None else bundles
    pairs = []
    for bundle in bundles:
        batches = len(bundle)
        pairs.append((device.device_tensor((batches, width), 'int32', 'row_major', 'dram',
                                           torch.zeros(batches, width, dtype=torch.int32), 'table'),
                      device.device_tensor((batches,), 'int32', 'row_major', 'dram',
                                           torch.zeros(batches, dtype=torch.int32), 'cur_pos')))
    return pairs


def host_table(seed, width=WIDTH):
    return (torch.arange(width, dtype=torch.int32) * 3 + seed).reshape(1, width)


def fake_device_layout_dma(device):
    def permute(mesh, source, rows, owned, *, inverse=False, offset=0):
        output = device.device_tensor((1, rows, 12, 256) if inverse else (1, 1, rows * 12, 256), 'bf16', 'tile',
                                      'dram', name='fold', origin=('fold', source, inverse, offset))
        owned.append(output)
        return output
    return permute


def recording_refresh(device, refreshed):
    """attention_mask_replay.execute's stand-in: records each mask refresh, and orders it against the
    SDPA calls in device.events."""
    def execute(positions, mask, program):
        refreshed.append((positions, mask, program))
        device.events.append(('refresh', mask))
    return execute


def expect(condition, what):
    if not condition:
        raise AssertionError(what)


def route(output):
    """Walk a segment reader's output back through the fake device, one entry per chunk in the order
    its concat joined them: (the tensor whose rows were folded in, the group offsets stacked, the
    bundle entry sliced back out, and that SDPA call's page table, cur_pos and mask)."""
    expect(output.origin is not None and output.origin[0] == 'concat', 'not a concat: %r' % (output.origin,))
    routes = []
    for chunk in output.origin[1]:
        kind, selected, inverse, offset = chunk.origin
        expect(kind == 'fold' and inverse, 'chunk is not an inverse fold: %r' % (chunk.origin,))
        kind, result, first, last = selected.origin
        expect(kind == 'slice' and last[1] == first[1] + 1, 'chunk is not one bundle entry: %r' % (selected.origin,))
        kind, stacked, options = result.origin
        expect(kind == 'sdpa', 'entry is not an SDPA result: %r' % (result.origin,))
        kind, folds = stacked.origin
        expect(kind == 'concat' and all(fold.origin[0] == 'fold' and not fold.origin[2] for fold in folds),
               'SDPA query is not a stack of folds: %r' % (stacked.origin,))
        sources = [fold.origin[1] for fold in folds]
        expect(all(source is sources[0] for source in sources), 'one bundle folds two sources')
        routes.append((sources[0], tuple(fold.origin[3] for fold in folds), first[1],
                       options['page_table_tensor'], options['cur_pos_tensor'], options['attn_mask']))
    return routes


class Harness(object):
    """Patches for building readers on the fake device: the environment, the binary check, the mask
    program (prepare_narrow needs ttnn) and the logs."""

    def __init__(self, test, env=ENV, f22=True):
        self.stack = ExitStack()
        self.logs, self.checked = [], []
        self.stack.enter_context(patch.dict(os.environ, env, clear=True))
        self.stack.enter_context(patch('pooled_attention_replay._binary_checked', []))
        self.stack.enter_context(patch('pooled_attention_replay._pindiag', side_effect=self.logs.append))
        self.stack.enter_context(patch('extent_attention_replay._pindiag', side_effect=self.logs.append))

        def binary(markers):
            self.checked.append(markers)
            return '/k64j/_ttnncpp.so' if f22 else '/k64i/_ttnncpp.so', f22 or F22_MARKER not in markers

        self.stack.enter_context(patch('pooled_attention_replay.loaded_binary_has_modes', side_effect=binary))
        self.programs = []

        def program(mesh, positions, mask, *, rows, batches, offset):
            self.programs.append(dict(positions=positions, mask=mask, rows=rows, batches=batches, offset=offset))
            return 'program%d' % len(self.programs)

        self.stack.enter_context(patch('extent_attention_replay.prepare_narrow', side_effect=program))
        test.addCleanup(self.stack.close)


def segment(device, start=4200, rows=16, width=WIDTH, pairs=None, table=None, **options):
    pairs = lend(device, rows, width) if pairs is None else pairs
    table = host_table(5, width) if table is None else table
    options.setdefault('max_group_rows', 8)
    return ExtentSegmentReader(device, MESH, rows, width, table, storage=pairs, start=start, **options), pairs


class HelperTests(unittest.TestCase):
    """Tests 1-5: the geometry the readers and the block share."""

    def test_1_layout_is_the_pinned_grouping_of_every_non_crossing_ticket_at_128_or_more(self):
        for group_rows in (4, 8):
            for rows in (8, 16, 32):
                layout = structure(LAYOUT(rows, group_rows))
                self.assertEqual(layout, structure(parallel_groups(256, rows, max_group_rows=group_rows)))
                # The pinned reader's own capture start, family by family.
                for family in range(512, C + 1, K):
                    first = family - K
                    with self.subTest(rows=rows, group_rows=group_rows, family=family):
                        self.assertEqual(structure(parallel_groups(first, rows, max_group_rows=group_rows)), layout)
                # Any non-crossing start at or above the floor, in the first family and near every edge.
                starts = set(range(MIN_LIVE_START, K - rows + 1))
                for family in range(K, C + 1, K * 16):
                    starts.update(family - K + residue for residue in (0, 1, 7, 127, 128, K - rows)
                                  if family - K + residue >= MIN_LIVE_START)
                for start in sorted(starts):
                    bundles = parallel_groups(start, rows, max_group_rows=group_rows)
                    with self.subTest(rows=rows, group_rows=group_rows, start=start):
                        self.assertEqual(structure(bundles), layout)
                        self.assertEqual({group['signature'] for bundle in bundles for group in bundle},
                                         {(K, extent(start))})
        # The floor: a ticket from 127 has a 128-key first row and bundles otherwise.
        self.assertNotEqual(structure(parallel_groups(127, 16, max_group_rows=8)), structure(LAYOUT(16, 8)))
        self.assertEqual(structure(LAYOUT(16, 8)), [[(0, 8), (8, 8)]])

    def test_2_the_narrow_mask_is_the_last_256_columns_of_the_wide_mask_in_every_family(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        residues = (0, 7, 127, 128, 240, 241, 255)
        families = range(K, C + 1, K)
        self.assertEqual(len(families), 513)
        # The served bundle (two eight-row groups at offset 0) in all 513 families; the other bundle
        # offsets a 32-row segment makes, and a one-group bundle, in every 16th family and the last.
        geometries = [((8, 2, 0), families), ((8, 2, 8), families[::16]), ((8, 3, 0), families[::16]),
                      ((8, 1, 24), families[::16]), ((4, 3, 0), families[::16]), ((8, 1, 0), (C,))]
        for (rows, batches, offset), sweep in geometries:
            head_tiles = (rows * 12 + 31) // 32
            for residue in residues:
                narrow_pages, narrow = mask_tiles(residue, rows, batches, offset, K)
                self.assertEqual(set(narrow.flatten().tolist()) - {0, 0xff80 - (1 << 16)}, set())
                grid = torch.arange(batches * head_tiles * 8)
                self.assertTrue(torch.equal(narrow_pages, grid), 'capacity 256: the narrow tensor, page for page')
                for family in sweep:
                    start = family - K + residue
                    pages, wide = mask_tiles(start, rows, batches, offset, family)
                    row = grid // 8
                    if not torch.equal(wide, narrow) or not torch.equal(
                            pages, row * (family // 32) + family // 32 - 8 + grid % 8):
                        self.fail('narrow != wide tail: rows=%d batches=%d offset=%d family=%d start=%d'
                                  % (rows, batches, offset, family, start))
        # Dense, as the device holds them: the wide mask is +0.0 before its last 256 columns.
        for family in (256, 512, 4352, 16640):
            for residue in residues:
                start = family - K + residue
                wide = replay_mask_host(start, 8, 2, 0, family).view(torch.int16)
                narrow = narrow_mask_host(start & 255, 8, 2, 0).view(torch.int16)
                self.assertEqual(tuple(narrow.shape), (2, 1, 96, 256))
                self.assertTrue(torch.equal(wide[..., family - K:], narrow), (family, residue))
                self.assertFalse(bool(wide[..., :family - K].any()), (family, residue))
                self.assertEqual(set((narrow.int() & 0xffff).flatten().tolist()) - {0, 0xff80}, set())

    def test_2b_the_narrow_mask_is_the_folded_causal_mask_of_each_group(self):
        """An oracle that is not the transliteration: attention_head_fold.causal_mask per group, over
        the family's last 256 keys (crossing rows included, read past E)."""
        for family in (256, 4352, C):
            for residue in (0, 7, 128, 240, 250, 255):
                start = family - K + residue
                narrow = narrow_mask_host(residue, 8, 2, 0)
                for batch in range(2):
                    oracle = causal_mask(8, start + batch * 8, family + K)[0, 0, :, family - K:family]
                    with self.subTest(family=family, residue=residue, batch=batch):
                        self.assertTrue(torch.equal(narrow[batch, 0].view(torch.int16), oracle.view(torch.int16)))

    def test_3_the_word_and_cur_pos_name_one_family(self):
        for start in range(0, C - 16 + 1):
            word, position = extent_values(start)
            family = position + 1
            if not (0 <= word < K and family % K == 0 and start < family <= start + K
                    and start - word + K == family and family == extent(start)):
                self.fail('start %d gives word %d and cur_pos %d' % (start, word, position))
        self.assertEqual(extent_values(C - 16), (240, C - 1))
        for bad in (-1, 1.0, True, None, '5'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                extent_values(bad)

    def test_4_idle_starts_zero_and_thirty_two_read_the_first_family(self):
        self.assertEqual(extent_values(0), (0, 255))
        self.assertEqual(extent_values(32), (32, 255))
        self.assertFalse(admits(0, 16, C) or admits(32, 16, C), 'idle starts are not live tickets')

    def test_5_accept_limit_caps_the_rows_past_the_family(self):
        table = {residue: accept_limit(4096 + residue, 16) for residue in range(240, 256)}
        self.assertEqual(table, {240: 16, 241: 15, 242: 14, 243: 13, 244: 12, 245: 11, 246: 10, 247: 9,
                                 248: 8, 249: 7, 250: 6, 251: 5, 252: 4, 253: 3, 254: 2, 255: 1})
        self.assertEqual([accept_limit(residue, 16) for residue in (0, 128, 239)], [16, 16, 16])
        self.assertEqual(accept_limit(C - 16, 16), 16)
        with self.assertRaises(ValueError):
            accept_limit(250, 0)

    def test_admits_holds_live_tickets_to_the_floor_and_the_table(self):
        self.assertTrue(admits(128, 16, C) and admits(C - 16, 16, C))
        for start in (127, C - 15, -1, 128.0, True, None):
            with self.subTest(start=start):
                self.assertFalse(admits(start, 16, C))

    def test_the_pool_lends_what_the_readers_take(self):
        """serving_buffer_pool may not import this module (the P8 closure: it is a P8-copied module),
        so it keeps its own copy of the layout; this is the pin that keeps the two equal."""
        import serving_buffer_pool
        self.assertEqual(serving_buffer_pool.EXTENT_LAYOUT_START, K)
        self.assertEqual((serving_buffer_pool.EXTENT_GROUP_ROWS, serving_buffer_pool.EXTENT_BUNDLE_ENTRIES),
                         (extent_attention_replay.EXTENT_GROUP_ROWS, extent_attention_replay.EXTENT_BUNDLE_ENTRIES))
        for group_rows in (4, 8):
            for rows in (8, 16, 32):
                self.assertEqual(serving_buffer_pool.extent_bundle_batches(rows, group_rows),
                                 tuple(len(bundle) for bundle in LAYOUT(rows, group_rows)))


def fake_ttnn():
    """Enough of ttnn for the two prepares to build a program descriptor on the host."""
    class KernelDescriptor:
        def __init__(self, **options):
            self.__dict__.update(options)
            self.runtime_args = None

    return SimpleNamespace(
        int32='int32', bfloat16='bf16', ROW_MAJOR_LAYOUT='row_major', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
        CoreCoord=lambda x, y: ('coord', x, y), CoreRange=lambda a, b: ('range', a, b),
        CoreRangeSet=lambda ranges: ('set', tuple(ranges)), Tile=lambda dims: ('tile', tuple(dims)),
        TileDescriptor=lambda tile: ('tile_descriptor', tile),
        CBFormatDescriptor=lambda **options: ('format', tuple(sorted(options.items()))),
        CBDescriptor=lambda **options: ('cb', options['total_size'], options['core_ranges'],
                                        tuple(options['format_descriptors'])),
        get_device_tensors=lambda tensor: tensor.shards, MeshProgramDescriptor=dict, KernelDescriptor=KernelDescriptor,
        TensorAccessorArgs=lambda shard: SimpleNamespace(get_compile_time_args=lambda: ['interleaved', 'dram']),
        DataMovementConfigDescriptor=lambda **options: ('data_movement', tuple(sorted(options.items()))),
        DataMovementProcessor=SimpleNamespace(RISCV_0='riscv0'), NOC=SimpleNamespace(RISCV_0_default='noc0'),
        RuntimeArgs=lambda: defaultdict(dict), MeshCoordinate=lambda a, b: (a, b),
        MeshCoordinateRange=lambda a, b: ('mesh_range', a, b),
        ProgramDescriptor=lambda kernels, cbs: SimpleNamespace(kernels=kernels, cbs=cbs))


def program_view(program):
    return {key: [(kernel.kernel_source, kernel.core_ranges, kernel.compile_time_args, kernel.config,
                   {column: dict(rows) for column, rows in kernel.runtime_args.items()}) for kernel in value.kernels]
            + [value.cbs] for key, value in program.items()}


class PrepareNarrowTests(unittest.TestCase):
    """prepare_narrow is attention_mask_replay.prepare at capacity 256: the same kernel, cores, buffer and
    arguments, but for the capacity argument and the mask width."""

    def tensors(self, width, batches=2, rows=8):
        positions = FakeTensor((8,), 'int32', 'row_major', 'dram', [FakeShard(0x100), FakeShard(0x9100)])
        mask = FakeTensor((batches, 1, rows * 12, width), 'bf16', 'tile', 'dram', [FakeShard(0x200), FakeShard(0x9200)])
        return positions, mask

    def test_the_program_is_the_pinned_one_but_the_capacity_argument(self):
        with patch.dict(sys.modules, {'ttnn': fake_ttnn()}), patch.dict(os.environ, {}, clear=True):
            for rows, batches, offset in ((8, 2, 0), (8, 1, 8), (8, 3, 0), (4, 3, 0)):
                positions, wide = self.tensors(4352, batches, rows)
                pinned = program_view(attention_mask_replay.prepare(MESH, positions, wide, rows=rows, batches=batches,
                                                                    offset=offset, capacity=4352))
                positions, narrow = self.tensors(K, batches, rows)
                ours = program_view(prepare_narrow(MESH, positions, narrow, rows=rows, batches=batches, offset=offset))
                self.assertEqual(set(ours), set(pinned))
                for key in pinned:
                    for kernel, (ours_kernel, pinned_kernel) in enumerate(zip(ours[key][:-1], pinned[key][:-1])):
                        self.assertEqual(ours_kernel[:4], pinned_kernel[:4])
                        self.assertEqual(ours_kernel[0], str(Path(attention_mask_replay.__file__).with_suffix('.cpp')))
                        for column, entries in pinned_kernel[4].items():
                            for row, arguments in entries.items():
                                self.assertEqual(arguments[3], 4352)
                                self.assertEqual(ours_kernel[4][column][row], arguments[:3] + [K] + arguments[4:])
                        self.assertEqual(sum(len(entries) for entries in ours_kernel[4].values()),
                                         batches * ((rows * 12 + 31) // 32) * 8)
                    self.assertEqual(ours[key][-1], pinned[key][-1])

    def test_it_refuses_a_wide_mask_aliases_and_bad_geometry(self):
        with patch.dict(sys.modules, {'ttnn': fake_ttnn()}):
            positions, wide = self.tensors(4352)
            with self.assertRaisesRegex(ValueError, 'narrow attention mask'):
                prepare_narrow(MESH, positions, wide, rows=8, batches=2, offset=0)
            positions, narrow = self.tensors(K)
            narrow.shards = positions.shards
            with self.assertRaisesRegex(ValueError, 'must not alias'):
                prepare_narrow(MESH, positions, narrow, rows=8, batches=2, offset=0)
            positions, narrow = self.tensors(K)
            narrow.memory = 'l1'
            with self.assertRaisesRegex(ValueError, 'Interleaved DRAM'):
                prepare_narrow(MESH, positions, narrow, rows=8, batches=2, offset=0)
            for options in (dict(rows=9, batches=2, offset=0), dict(rows=8, batches=4, offset=0),
                            dict(rows=8, batches=2, offset=24), dict(rows=8.0, batches=2, offset=0)):
                positions, narrow = self.tensors(K)
                with self.subTest(options=options), self.assertRaises(ValueError):
                    prepare_narrow(MESH, positions, narrow, **options)


class SegmentReaderTests(unittest.TestCase):
    """Tests 6-10 on one sixteen-row segment."""

    def setUp(self):
        self.device = FakeDevice()
        self.harness = Harness(self)

    def test_7_construction_stages_the_word_cur_pos_and_table_for_its_start(self):
        for start in (0, 32, 128, 1500, 20000, 60000, C - 16):
            device = FakeDevice()
            reader, pairs = segment(device, start=start)
            word, position = extent_values(start)
            with self.subTest(start=start):
                self.assertEqual(reader.start, start)
                self.assertEqual(reader.positions.value.tolist(), [word, 0, 0, 0, 0, 0, 0, 0])
                ((table, cur_pos),) = pairs
                self.assertEqual(cur_pos.value.tolist(), [position, position])
                self.assertTrue(torch.equal(table.value, host_table(5).repeat(2, 1)))
                # One fenced write of exactly these three, in this order.
                self.assertEqual(device.copies, [reader.positions, cur_pos, table])
                self.assertEqual(device.fences, 1)
                self.assertFalse(reader.failed)
                self.assertEqual((reader.capacity, reader.rows, reader.max_group_rows), (C, 16, 8))
                self.assertEqual(reader.borrowed, [table, cur_pos])
                self.assertEqual(reader.cur_pos, [cur_pos])
        # model_batch.py:611-614 skips the initial stage when the start already matches: sound, the
        # device already holds the start's word, cur_pos and table.
        self.assertEqual(reader.start, C - 16)

    def test_7_a_failed_construction_copy_poisons_closes_and_frees_only_its_own(self):
        states = []
        original = ExtentSegmentReader.close

        def close(reader):
            states.append(reader.failed)
            return original(reader)

        pairs = lend(self.device)

        def fail(source, destination):
            if destination is pairs[0][1]:
                raise RuntimeError('copy failed')

        self.device.copy_hook = fail
        with patch.object(ExtentSegmentReader, 'close', autospec=True, side_effect=close), \
                self.assertRaisesRegex(RuntimeError, 'copy failed'):
            segment(self.device, pairs=pairs)
        self.assertEqual(states, [True])
        freed = self.device.deallocated
        self.assertEqual(sorted(value.name for value in freed), ['upload', 'upload'], 'the word and the mask')
        self.assertFalse(any(value is lent for value in freed for pair in pairs for lent in pair))

    def test_8_staging_keeps_addresses_and_a_failed_copy_poisons(self):
        reader, ((table, cur_pos),) = segment(self.device, start=4200)
        before = [list(value.shards) for value in (reader.positions, table, cur_pos)]
        new = host_table(9)
        reader.stage(70000, table=new)
        self.assertEqual(reader.start, 70000)
        self.assertEqual(reader.positions.value.tolist()[0], 70000 & 255, 'never the absolute word')
        self.assertEqual(cur_pos.value.tolist(), [extent(70000) - 1] * 2)
        self.assertTrue(torch.equal(table.value, new.repeat(2, 1)))
        self.assertEqual([list(value.shards) for value in (reader.positions, table, cur_pos)], before)
        reader.stage(4200)
        self.assertTrue(torch.equal(table.value, new.repeat(2, 1)), 'a stage without a table leaves the lent table')

        def move(source, destination):
            if destination is cur_pos:
                destination.shards = [FakeShard(0x7777), destination.shards[1]]

        self.device.copy_hook = move
        with self.assertRaisesRegex(AssertionError, 'replaced a captured extent buffer'):
            reader.stage(4300)
        self.assertTrue(reader.failed)
        self.assertEqual(reader.start, 4200)
        with self.assertRaisesRegex(RuntimeError, 'poisoned'):
            reader.validate(4200)
        other, pairs = segment(FakeDevice())
        other.operations.copy_hook = lambda source, destination: (_ for _ in ()).throw(RuntimeError('copy failed'))
        with self.assertRaisesRegex(RuntimeError, 'copy failed'):
            other.stage(4300)
        self.assertTrue(other.failed)

    def test_stage_values_are_the_word_then_cur_pos_and_table_per_bundle(self):
        reader, ((table, cur_pos),) = segment(self.device, start=4200)
        values = reader.stage_values(131100, host_table(3))
        self.assertEqual([value[0] for value in values], [reader.positions, cur_pos, table])
        self.assertEqual(values[0][1].tolist(), [131100 & 255] + [0] * 7)
        self.assertEqual(values[1][1].tolist(), [C - 1, C - 1])
        self.assertTrue(torch.equal(values[2][1], host_table(3).repeat(2, 1)))
        self.assertEqual({(dtype, layout) for destination, value, dtype, layout in values}, {('int32', 'row_major')})
        self.assertTrue(all(tuple(destination.shape) == tuple(value.shape) for destination, value, _, _ in values))
        self.assertEqual(len(self.device.copies), 3, 'host only: nothing written')
        for start in (C - 15, -1, 4200.0):
            with self.subTest(start=start), self.assertRaises(ValueError):
                reader.stage_values(start, host_table(3))
        with self.assertRaises(ValueError):
            reader.stage_values(4200, host_table(3, WIDTH - 1))

    def test_a_restage_after_a_served_round_writes_the_word_and_cur_pos_and_leaves_the_table_alone(self):
        """Review defect 1. The block writes each round's tables through stage_values -> write_packed,
        never through reader.stage, so a later stage(start) must not restage a table the reader held
        since capture: that would silently point every attention layer at the capture's pages."""
        reader, ((table, cur_pos),) = segment(self.device, start=4200)
        served = host_table(9)
        for destination, value, dtype, layout in reader.stage_values(70000, served):
            self.device.copy_host_to_device_tensor(self.device.from_torch(value, dtype=dtype, layout=layout),
                                                   destination)
        reader.start = 70000  # stage_packed sets each reader's start after write_packed
        copies, fences = len(self.device.copies), self.device.fences
        for start in (70000, 90000):
            reader.stage(start)
            with self.subTest(start=start):
                self.assertTrue(torch.equal(table.value, served.repeat(2, 1)), 'the capture-time table came back')
                self.assertEqual(cur_pos.value.tolist(), [extent(start) - 1] * 2)
                self.assertEqual(reader.positions.value.tolist(), [start & 255] + [0] * 7)
        self.assertEqual(self.device.copies[copies:], [reader.positions, cur_pos] * 2, 'the word and cur_pos only')
        self.assertEqual(self.device.fences, fences + 2)
        # A table is written only when one is passed, as construction passes its own.
        reader.stage(4200, table=host_table(11))
        self.assertTrue(torch.equal(table.value, host_table(11).repeat(2, 1)))
        self.assertEqual(self.device.copies[-3:], [reader.positions, cur_pos, table])
        self.assertEqual([value[0] for value in reader.stage_values(4300, None)], [reader.positions, cur_pos])
        self.assertFalse(hasattr(reader, 'pages_host'), 'the reader keeps no host table to restage')

    def test_9_a_shared_mask_forward_refreshes_each_mask_once(self):
        reader, pairs = segment(self.device)
        refreshed = []
        query = self.device.device_tensor((1, 16, 12, 256), 'bf16', 'tile')
        keys, values = self.device.device_tensor((1,), 'bf16', 'tile'), self.device.device_tensor((1,), 'bf16', 'tile')
        with patch('attention_mask_replay.execute', side_effect=lambda *arguments: refreshed.append(arguments)), \
                patch('extent_attention_replay.device_layout_dma', side_effect=fake_device_layout_dma(self.device)):
            with reader.shared_masks(16):
                for layer in range(16):
                    reader(query, keys, values, scale=0.0625, memory_config='dram')
            self.assertEqual(reader.refresh_calls, len(reader.metadata))
            self.assertEqual(refreshed, [(reader.positions, reader.metadata[0][2], reader.programs[0])])
            with self.assertRaisesRegex(AssertionError, 'exact attention call budget'):
                with reader.shared_masks(16):
                    reader(query, keys, values, scale=0.0625, memory_config='dram')
        self.assertTrue(reader.failed)

    def test_10_every_sdpa_call_carries_its_cur_pos_the_narrow_mask_and_0x27(self):
        reader, ((table, cur_pos),) = segment(self.device, start=60000)
        query = self.device.device_tensor((1, 16, 12, 256), 'bf16', 'tile')
        keys, values = self.device.device_tensor((1,), 'bf16', 'tile'), self.device.device_tensor((1,), 'bf16', 'tile')
        refreshed = []
        with patch('attention_mask_replay.execute', side_effect=recording_refresh(self.device, refreshed)), \
                patch('extent_attention_replay.device_layout_dma', side_effect=fake_device_layout_dma(self.device)):
            result = reader(query, keys, values, page_table_tensor='ignored', cur_pos_tensor='ignored',
                            scale=0.0625, memory_config='dram')
            (call,) = self.device.sdpa
            mask = reader.metadata[0][2]
            # Review defect 3: outside shared_masks every call first refreshes each bundle's narrow mask,
            # once; a call that skipped it would read a stale or zero mask and see future keys.
            self.assertEqual(refreshed, [(reader.positions, mask, reader.programs[0])])
            self.assertEqual(reader.refresh_calls, len(reader.metadata))
            self.assertEqual(self.device.events, [('refresh', mask), ('sdpa', mask)])
            reader(query, keys, values, scale=0.0625, memory_config='dram')
            self.assertEqual(reader.refresh_calls, 2 * len(reader.metadata))
            self.assertEqual(self.device.events[2:], [('refresh', mask), ('sdpa', mask)])
        # Review defect 2, in the segment: both groups of the query stacked at offsets 0 and 8, and the
        # two bundle entries sliced back out in order, all through this user's table, cur_pos and mask.
        self.assertEqual(route(result), [(query, (0, 8), 0, table, cur_pos, mask), (query, (0, 8), 1, table, cur_pos, mask)])
        self.assertIs(call['cur_pos_tensor'], cur_pos)
        self.assertIs(call['page_table_tensor'], table)
        self.assertIs(call['attn_mask'], mask)
        self.assertEqual((mask.shape, mask.dtype, mask.layout), ((2, 1, 96, 256), 'bf16', 'tile'))
        self.assertEqual(call['program_config'].q_chunk_size, QWEN_DECODE_MAGIC | EXTENT_FLAGS)
        self.assertEqual((call['program_config'].k_chunk_size, call['is_causal'], call['scale']), (256, False, 0.0625))
        self.assertEqual(result.shape, (1, 16, 12, 256))
        self.assertEqual(reader.calls, 2)
        self.assertEqual(reader.sdpa_modes_applied, (EXTENT_FLAGS,))
        # The mask program was prepared on the reader's own word and narrow mask, at the bundle's geometry.
        (program,) = self.harness.programs
        self.assertEqual(program, dict(positions=reader.positions, mask=mask, rows=8, batches=2, offset=0))
        self.assertIn(F22_MARKER, self.harness.checked[0])
        self.assertTrue(any('mask=narrow' in line and "flags=['0x27']" in line for line in self.harness.logs))

    def test_6_refusals(self):
        cases = {
            'no tail': dict(env=dict(ENV, QWEN_FAST_SDPA_MODES='share,slice')),
            'modes unset': dict(env={'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}),
            'tree scratch unset': dict(env={'QWEN_FAST_SDPA_MODES': 'tail,share,slice'}),
            'tree scratch not 1': dict(env=dict(ENV, QWEN_SDPA_TREE_SCRATCH_ROUNDS='2')),
            'G4': dict(options=dict(max_group_rows=4)),
            'G8 as a float': dict(options=dict(max_group_rows=8.0)),
            'an 8-row segment (B1)': dict(options=dict(rows=8)),
            'a 32-row segment (B3 + B1)': dict(options=dict(rows=32)),
            'a width not of whole families': dict(options=dict(width=2050)),
            'a short host table': dict(options=dict(table=host_table(1, WIDTH - 1))),
            'a float start': dict(options=dict(start=4200.0)),
            'a bool start': dict(options=dict(start=True)),
            'a negative start': dict(options=dict(start=-1)),
            'start + rows past C': dict(options=dict(start=C - 15)),
        }
        for name, case in cases.items():
            with self.subTest(name=name):
                device = FakeDevice()
                with ExitStack() as stack:
                    if 'env' in case:
                        stack.enter_context(patch.dict(os.environ, case['env'], clear=True))
                    options = dict(case.get('options', {}))
                    rows, width = options.pop('rows', 16), options.pop('width', WIDTH)
                    pairs = lend(device, 16, width)
                    with self.assertRaises(ValueError):
                        segment(device, rows=rows, width=width, pairs=pairs, **options)
                self.assertEqual(len(device.live), 2, 'refused before anything was allocated')
                self.assertEqual(device.copies, [])

    def test_6_lent_storage_of_the_wrong_geometry_or_aliasing_is_refused_before_anything_is_built(self):
        def mutate(change):
            device = FakeDevice()
            pairs = lend(device)
            return device, change(device, pairs)

        def table(**changes):
            def change(device, pairs):
                for key, value in changes.items():
                    setattr(pairs[0][0], key, value)
                return pairs
            return change

        def positions(**changes):
            def change(device, pairs):
                for key, value in changes.items():
                    setattr(pairs[0][1], key, value)
                return pairs
            return change

        class Padded(FakeTensor):
            @property
            def padded_shape(self):
                return (32,)

        def padded(device, pairs):
            old = pairs[0][1]
            return [(pairs[0][0], Padded(old.shape, old.dtype, old.layout, old.memory, old.shards, old.value))]

        def alias(device, pairs):
            pairs[0][1].shards = [FakeShard(pairs[0][0].shards[0].address), pairs[0][1].shards[1]]
            return pairs

        class Unconfigured(FakeTensor):
            """A lent tensor that cannot state its memory config (review defect 4)."""

            def __getattribute__(self, name):
                if name == 'memory_config':
                    raise AttributeError(name)
                return super().__getattribute__(name)

        def unconfigured(which):
            def change(device, pairs):
                old = pairs[0][which]
                bare = Unconfigured(old.shape, old.dtype, old.layout, old.memory, old.shards, old.value)
                return [tuple(bare if index == which else tensor for index, tensor in enumerate(pairs[0]))]
            return change

        # Never taken for interleaved DRAM: the F20 check reads every tensor's own memory config.
        for name, change in (('table', unconfigured(0)), ('cur_pos', unconfigured(1))):
            with self.subTest(name='%s without memory_config' % name):
                device, pairs = mutate(change)
                allocated = len(device.live)
                with self.assertRaises(AttributeError):
                    ExtentSegmentReader(device, MESH, 16, WIDTH, host_table(1), storage=pairs, max_group_rows=8, start=4200)
                self.assertEqual(len(device.live), allocated)
                self.assertEqual(device.copies, [])
        cases = {'table narrow': table(shape=(2, WIDTH - 4)), 'table one entry': table(shape=(1, WIDTH)),
                 'table uint32': table(dtype='uint32'), 'table tiled': table(layout='tile'), 'table in L1': table(memory='l1'),
                 'cur_pos one entry': positions(shape=(1,)), 'cur_pos 2-D': positions(shape=(2, 1)),
                 'cur_pos uint32': positions(dtype='uint32'), 'cur_pos tiled': positions(layout='tile'),
                 'cur_pos sharded': positions(memory='l1_sharded'), 'cur_pos padded': padded,
                 'cur_pos aliases the table': alias, 'no storage': lambda device, pairs: None,
                 'two pairs for one bundle': lambda device, pairs: pairs + lend(device),
                 'a table alone': lambda device, pairs: [(pairs[0][0],)]}
        for name, change in cases.items():
            with self.subTest(name=name):
                device, pairs = mutate(change)
                allocated = len(device.live)
                with self.assertRaises(ValueError):
                    ExtentSegmentReader(device, MESH, 16, WIDTH, host_table(1), storage=pairs, max_group_rows=8, start=4200)
                self.assertEqual(len(device.live), allocated)
                self.assertEqual(device.copies, [])

    def test_6_a_binary_without_f22_or_modes_without_0x27_close_the_reader(self):
        for name, harness in (('K64i .so', dict(f22=False)), ('tail only', dict(env=dict(ENV, QWEN_FAST_SDPA_MODES='tail'))),
                              ('tail,share', dict(env=dict(ENV, QWEN_FAST_SDPA_MODES='tail,share'))),
                              ('readahead', dict(env=dict(ENV, QWEN_FAST_SDPA_MODES='tail,share,slice,readahead')))):
            with self.subTest(name=name):
                device = FakeDevice()
                pairs = lend(device)
                Harness(self, **harness)
                with self.assertRaises((RuntimeError, ValueError)) as caught:
                    segment(device, pairs=pairs)
                if name == 'K64i .so':
                    self.assertIn('runtime-extent factory', str(caught.exception))
                else:
                    self.assertIn('qualified at 0x27 only', str(caught.exception))
                self.assertEqual(sorted(value.name for value in device.deallocated), ['upload', 'upload'])
                self.assertFalse(any(value is lent for value in device.deallocated for pair in pairs for lent in pair))

    def test_6_an_attention_audit_is_refused(self):
        reader, pairs = segment(self.device)
        reader.audit = None
        with self.assertRaisesRegex(ValueError, 'takes no attention audit'):
            reader.audit = object()
        self.assertIsNone(reader.audit)

    def test_close_frees_the_word_and_masks_and_never_the_pool_storage(self):
        reader, pairs = segment(self.device)
        reader.close()
        reader.close()
        self.assertTrue(reader.closed)
        self.assertEqual(sorted(value.name for value in self.device.deallocated), ['upload', 'upload'])
        self.assertFalse(any(value is lent for value in self.device.deallocated for pair in pairs for lent in pair))
        for operation in (lambda: reader.validate(4200), lambda: reader.refresh(), lambda: reader.stage(4200)):
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                operation()


class PackedReaderTests(unittest.TestCase):
    """The block's adapter: four segments, each at its own family, dispatched a segment at a time."""

    STARTS = (1500, 20000, 60000, C - 16)
    SEGMENTS = ((0, 16), (16, 32), (32, 48), (48, 64))

    def setUp(self):
        self.device = FakeDevice()
        self.harness = Harness(self)

    def build(self, starts=STARTS, storage=None, tables=None, **options):
        storage = [lend(self.device) for segment in self.SEGMENTS] if storage is None else storage
        tables = [host_table(user) for user in range(4)] if tables is None else tables
        options.setdefault('max_group_rows', 8)
        return PackedExtentReplayReader(self.device, MESH, self.SEGMENTS, WIDTH, tables, storage=storage,
                                        starts=starts, **options), storage

    def test_four_users_at_four_families_each_stage_their_own_word_cur_pos_and_table(self):
        reader, storage = self.build()
        self.assertEqual((reader.rows, reader.capacity, len(reader.readers)), (64, C, 4))
        self.assertEqual(reader.starts, self.STARTS)
        for user, (own, start) in enumerate(zip(reader.readers, self.STARTS)):
            ((table, cur_pos),) = storage[user]
            word, position = extent_values(start)
            self.assertEqual(own.positions.value.tolist()[0], word)
            self.assertEqual(cur_pos.value.tolist(), [position] * 2)
            self.assertTrue(torch.equal(table.value, host_table(user).repeat(2, 1)))
        self.assertEqual([own.positions.value.tolist()[0] for own in reader.readers], [220, 32, 96, 240])
        self.assertEqual([storage[user][0][1].value.tolist()[0] for user in range(4)], [1535, 20223, 60159, C - 1])
        self.assertEqual(reader.borrowed, [tensor for pairs in storage for pair in pairs for tensor in pair])
        self.assertEqual(reader.metadata, [entry for own in reader.readers for entry in own.metadata])
        engaged = [line for line in self.harness.logs if line.startswith('[PINDIAG] extent replay engaged')]
        self.assertEqual(engaged, ['[PINDIAG] extent replay engaged segments=4 flags=0x27,0x27,0x27,0x27 '
                                   'mask=narrow capacity=%d' % C])
        self.assertEqual((reader.calls, reader.refresh_calls, reader.failed, reader.closed, reader.audit),
                         (0, 0, False, False, None))

    def test_a_padded_round_puts_idle_users_at_0_and_32_on_the_zero_table(self):
        zero = torch.zeros(1, WIDTH, dtype=torch.int32)
        reader, storage = self.build(starts=(0, 32, 1500, 60000), tables=[zero, zero, host_table(2), host_table(3)])
        for user in (0, 1):
            ((table, cur_pos),) = storage[user]
            self.assertEqual(reader.readers[user].positions.value.tolist()[0], 32 * user)
            self.assertEqual(cur_pos.value.tolist(), [255, 255])
            self.assertFalse(bool(table.value.any()))

    def test_9_and_10_the_query_is_dispatched_a_segment_at_a_time_through_each_users_cur_pos(self):
        reader, storage = self.build()
        refreshed = []
        query = self.device.device_tensor((1, 64, 12, 256), 'bf16', 'tile')
        keys, values = self.device.device_tensor((1,), 'bf16', 'tile'), self.device.device_tensor((1,), 'bf16', 'tile')
        with patch('attention_mask_replay.execute', side_effect=lambda *arguments: refreshed.append(arguments)), \
                patch('extent_attention_replay.device_layout_dma', side_effect=fake_device_layout_dma(self.device)):
            with reader.shared_masks(16):
                for layer in range(16):
                    result = reader(query, keys, values, scale=0.0625, memory_config='dram')
        self.assertEqual(reader.refresh_calls, len(reader.metadata))
        self.assertEqual(len(refreshed), 4)
        self.assertEqual(len(self.device.sdpa), 64)
        for index, call in enumerate(self.device.sdpa):
            ((table, cur_pos),) = storage[index % 4]
            self.assertIs(call['cur_pos_tensor'], cur_pos)
            self.assertIs(call['page_table_tensor'], table)
            self.assertIs(call['attn_mask'], reader.readers[index % 4].metadata[0][2])
            self.assertEqual(call['program_config'].q_chunk_size, QWEN_DECODE_MAGIC | EXTENT_FLAGS)
        self.assertEqual(result.shape, (1, 64, 12, 256))
        self.assertEqual((reader.calls, [own.calls for own in reader.readers]), (16, [16] * 4))
        with self.assertRaisesRegex(ValueError, 'query geometry'):
            reader(self.device.device_tensor((1, 32, 12, 256), 'bf16', 'tile'), keys, values,
                   scale=0.0625, memory_config='dram')

    def test_each_segment_s_rows_go_through_its_own_cur_pos_and_table_and_return_in_row_order(self):
        """Review defect 2 (the pooled reader's pin, test_pooled_attention_replay.py:400-440): user k's
        rows [16k, 16k + 16) are sliced from the block's query, run through user k's own table, cur_pos
        and mask, and come back as the k-th part of the output, after that segment's own refresh."""
        reader, storage = self.build()
        query = self.device.device_tensor((1, 64, 12, 256), 'bf16', 'tile', name='query')
        keys, values = self.device.device_tensor((1,), 'bf16', 'tile'), self.device.device_tensor((1,), 'bf16', 'tile')
        refreshed = []
        with patch('attention_mask_replay.execute', side_effect=recording_refresh(self.device, refreshed)), \
                patch('extent_attention_replay.device_layout_dma', side_effect=fake_device_layout_dma(self.device)):
            result = reader(query, keys, values, scale=0.0625, memory_config='dram')
        sliced = [tensor for tensor in self.device.live
                  if tensor.origin is not None and tensor.origin[0] == 'slice' and tensor.origin[1] is query]
        self.assertEqual([tensor.origin[2:] for tensor in sliced],
                         [((0, first, 0, 0), (1, last, 12, 256)) for first, last in self.SEGMENTS])
        kind, parts = result.origin
        self.assertEqual((kind, len(parts), result.shape), ('concat', 4, (1, 64, 12, 256)))
        masks = [own.metadata[0][2] for own in reader.readers]
        for user, (part, rows) in enumerate(zip(parts, sliced)):
            ((table, cur_pos),) = storage[user]
            with self.subTest(user=user):
                self.assertEqual(route(part), [(rows, (0, 8), 0, table, cur_pos, masks[user]),
                                               (rows, (0, 8), 1, table, cur_pos, masks[user])])
        # Review defect 3, packed: unscoped, each segment refreshes its own mask once, before its SDPA.
        self.assertEqual(self.device.events, [event for mask in masks for event in (('refresh', mask), ('sdpa', mask))])
        self.assertEqual(reader.refresh_calls, len(reader.metadata))

    def test_a_packed_restage_after_a_served_round_leaves_every_user_s_table_alone(self):
        """Review defect 1, packed: stage(starts) writes each segment's word and cur_pos, and never a table."""
        reader, storage = self.build()
        starts = (300, 70000, 90000, 131100)
        served = [host_table(10 + user) for user in range(4)]
        for own, start, table in zip(reader.readers, starts, served):
            for destination, value, dtype, layout in own.stage_values(start, table):
                self.device.copy_host_to_device_tensor(self.device.from_torch(value, dtype=dtype, layout=layout),
                                                       destination)
            own.start = start
        copies = len(self.device.copies)
        reader.stage(starts)
        for user in range(4):
            ((table, cur_pos),) = storage[user]
            with self.subTest(user=user):
                self.assertTrue(torch.equal(table.value, served[user].repeat(2, 1)), 'the capture-time table came back')
                self.assertEqual(cur_pos.value.tolist(), [extent(starts[user]) - 1] * 2)
        self.assertEqual(self.device.copies[copies:],
                         [tensor for own, pairs in zip(reader.readers, storage) for tensor in (own.positions, pairs[0][1])])

    def test_every_start_is_checked_on_the_host_before_any_segment_is_built(self):
        """Review defect 6: a bad start for a later segment used to surface only after the segments
        before it were built and staged into their lent storage."""
        for starts in ((1500, 20000, C - 15, 60000), (1500, 20000, 60000, 4200.0), (1500, -1, 60000, 4200),
                       (1500, 20000, 60000, True)):
            with self.subTest(starts=starts):
                device = FakeDevice()
                storage = [lend(device) for segment in self.SEGMENTS]
                allocated = len(device.live)
                with self.assertRaisesRegex(ValueError, 'integer start'):
                    PackedExtentReplayReader(device, MESH, self.SEGMENTS, WIDTH, [host_table(user) for user in range(4)],
                                             storage=storage, max_group_rows=8, starts=starts)
                self.assertEqual(len(device.live), allocated, 'no segment reader was built')
                self.assertEqual(device.copies, [], 'nothing was staged into any lent storage')

    def test_validate_and_stage_reach_every_segment_and_idle_starts_validate(self):
        reader, storage = self.build()
        reader.validate((0, 32, 4096, C - 16))
        for starts in ((0, 32, 4096, C - 15), (0, 32, 4096), (0, 32, 4096, 5000, 6000), (0, 32, -1, 5)):
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                reader.validate(starts)
        reader.stage((300, 70000, 90000, 131100))
        self.assertEqual(reader.starts, (300, 70000, 90000, 131100))
        self.assertEqual([storage[user][0][1].value.tolist()[0] for user in range(4)],
                         [511, extent(70000) - 1, extent(90000) - 1, C - 1])
        self.assertEqual([own.positions.value.tolist()[0] for own in reader.readers],
                         [300 & 255, 70000 & 255, 90000 & 255, 131100 & 255])
        reader.failed = True
        self.assertTrue(all(own.failed for own in reader.readers))
        reader.failed = False
        self.assertFalse(reader.failed)
        with self.assertRaisesRegex(ValueError, 'takes no attention audit'):
            reader.audit = object()

    def test_a_failure_building_one_segment_closes_the_ones_built_and_frees_no_pool_storage(self):
        storage = [lend(self.device) for segment in self.SEGMENTS]

        def fail(source, destination):
            if destination is storage[2][0][1]:
                raise RuntimeError('copy failed')

        self.device.copy_hook = fail
        with self.assertRaisesRegex(RuntimeError, 'copy failed'):
            self.build(storage=storage)
        self.assertEqual(len(self.device.deallocated), 6, 'three readers built: a word and a mask each')
        self.assertFalse(any(value is lent for value in self.device.deallocated
                             for pairs in storage for pair in pairs for lent in pair))
        self.assertFalse(any(line.startswith('[PINDIAG] extent replay engaged') for line in self.harness.logs))

    def test_storage_is_checked_across_segments_before_any_reader_is_built(self):
        shared = lend(self.device)
        cases = {'a set shared by two segments': [shared, shared, lend(self.device), lend(self.device)],
                 'a set short': [lend(self.device) for index in range(3)],
                 'a set of another width': [lend(self.device), lend(self.device, width=WIDTH - 4), lend(self.device),
                                            lend(self.device)],
                 'no storage': None}
        for name, storage in cases.items():
            with self.subTest(name=name):
                allocated = len(self.device.live)
                with self.assertRaises(ValueError):
                    PackedExtentReplayReader(self.device, MESH, self.SEGMENTS, WIDTH, [host_table(0)] * 4,
                                             storage=storage, max_group_rows=8, starts=self.STARTS)
                self.assertEqual(len(self.device.live), allocated)
        with self.assertRaises(ValueError):
            self.build(starts=self.STARTS[:3])
        self.assertEqual(self.device.copies, [])

    def test_close_closes_every_segment_and_keeps_the_pool_storage(self):
        reader, storage = self.build()
        reader.close()
        reader.close()
        self.assertTrue(reader.closed and all(own.closed for own in reader.readers))
        self.assertEqual(len(self.device.deallocated), 8)
        self.assertFalse(any(value is lent for value in self.device.deallocated
                             for pairs in storage for pair in pairs for lent in pair))
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            reader.validate(self.STARTS)

    def test_the_pool_s_extent_storage_is_what_the_block_reader_takes(self):
        """serving_buffer_pool.PackedExtentStorage, lent to the reader as-is: the W2/W1 seam."""
        from serving_buffer_pool import ServingBufferPool
        device = FakeDevice()
        helpers = [SimpleNamespace(allocate=lambda: [device.device_tensor((1, 1, 32), 'bf16', 'tile')])
                   for layer in range(48)]

        def rope(positions):
            return tuple(device.device_tensor((1, len(positions), 1, 64), 'bf16', 'tile') for side in (0, 1))

        with patch('serving_buffer_pool.pindiag'):
            pool = ServingBufferPool(device, MESH, users=1, helpers=helpers, page_width=WIDTH, bucket_rows=(1,),
                                     rope=rope, packed_shapes=((4, 16),), packed_replay_group_rows=8, extent_replay=True)
        lent = pool.packed_extent(4, 16).take()
        self.device = device
        reader, storage = self.build(storage=lent.segment_storage())
        self.assertEqual(reader.borrowed, list(lent.tensors))
        self.assertEqual([storage[user][0][1].value.tolist() for user in range(4)],
                         [[extent(start) - 1] * 2 for start in self.STARTS])
        reader.close()
        lent.release()
        pool.close()
        self.assertEqual(sum(1 for value in device.deallocated if any(value is lent for lent in lent.tensors)),
                         len(lent.tensors), 'the pool, not the reader, frees its storage')


class StructureTests(unittest.TestCase):
    """Test 11: beside the pinned readers, never through them."""

    # target_t16_attention_gate.SOURCES as the image SERVES them: the bytes frozen_combined_runtime.qualify hashes at
    # every attach, and the files the extent path runs. Each is the frozen recipe's 8c102b20 file, and only
    # attention_mask_replay.py is also adapted (SERVED_MASK, above). The image is what pins these. Build v51
    # (run 36255706983) failed at test_11_the_pinned_sources_keep_their_bytes while this map pinned the checkout's
    # mask module, which no image carries. test_extent_reader_card_b holds the CB2b harness to these bytes, and the
    # harness loads them from the image.
    SOURCES = {
        'attention_replay.py': '4eff1c51fd42bb04adf68fc40bf74a0cca0cd455c3ae2fc50caf720f5281137a',
        'attention_mask_replay.py': SERVED_MASK,
        'attention_mask_replay.cpp': 'e10cae1d6fe97f9b1509ac5ef918f6e7eda8d51bfbd77dcfd9e95662bb838af8',
        'attention_parallel.py': '7bf5ba445100d184f7b9289fed4c20b4a730dbee29ee382cb97b6c2ad17f0e58',
        'attention_fold_dma.py': '5ce9d7d1590be2a9739a01d7604037f9fe70556594396067025bf1b3188151e5',
        'attention_fold_dma.cpp': '066fa6709127dcddbcdc033de9f0e0ad59a2c6756ceba3a99c5b0fd94cf26ab9',
        'target-t16-attention-probe.py': '6f5daa43e1379d8c7b06761f6aa1e046a84ec12f9f3b6e22b1dd232b7cbc25cd',
    }
    # This checkout's copies: the same bytes except the mask module, whose checkout copy no image carries
    # (CHECKOUT_MASK). test_pooled_attention_replay.SdpaModesTests.PINNED guards the shared ones against an edit.
    CHECKOUT = dict(SOURCES, **{'attention_mask_replay.py': CHECKOUT_MASK})

    # prepare_narrow against the SERVED attention_mask_replay.prepare: every difference, and why. Line numbers are the
    # served copy's. The checkout's are five more: prepare :41-84, the family check :48-49. prepare_narrow's docstring
    # cites the checkout's lines; its bytes are what CB2b qualifies (5633fc3a, packed_any_evidence.json), so they stay.
    PREPARE_NARROW_CHANGES = (
        ('(rows, batches, offset, capacity))', '(rows, batches, offset))',
         ':39 capacity is no argument: the kernel runs at K = 256'),
        ('    first = max(128, capacity - 256) if short_context else capacity - 256\n'
         '    validate_ticket(first, offset + rows * batches, capacity, short_context=short_context)\n', '',
         ':43-44 the family check is left out: the pinned validate_ticket refuses capacity 256 (minimum 4096, '
         ':18 and :21-22), and the extent reader validates its own starts'),
        ('(batches, 1, rows * 12, capacity)', '(batches, 1, rows * 12, K)', ':47 the mask is 256 keys wide'),
        ("'Fixed-shape BF16 folded attention mask required'",
         "'Fixed-shape BF16 folded narrow attention mask required'", ':48 its refusal names the narrow mask'),
        ("kernel_source=str(Path(__file__).with_suffix('.cpp')), core_ranges=cores,",
         "kernel_source=str(Path(attention_mask_replay.__file__).with_suffix('.cpp')),\n            core_ranges=cores,",
         ':68 the .cpp beside the pinned module, which is the kernel the served prepare compiles'),
        ('rows, capacity, offset, task]', 'rows, K, offset, task]', ':75 the capacity runtime argument is 256'),
    )

    def test_11_no_class_is_a_pinned_reader(self):
        from attention_replay import ReplayAttentionReader
        from pooled_attention_replay import PooledReplayAttentionReader
        for value in vars(extent_attention_replay).values():
            if isinstance(value, type) and value.__module__ == 'extent_attention_replay':
                with self.subTest(cls=value.__name__):
                    self.assertNotIn(ReplayAttentionReader, value.__mro__)
                    self.assertNotIn(PooledReplayAttentionReader, value.__mro__)

    def test_11_the_pinned_sources_keep_their_bytes(self):
        """In an image, every pinned source must be its served bytes. In a checkout, each must be the checkout's own:
        an edit there changes nothing any image runs, and it breaks this tree's evidence."""
        from target_t16_attention_gate import SOURCES
        self.assertEqual(set(SOURCES), set(self.SOURCES))
        checkout = in_checkout()
        for name, digest in sorted((self.CHECKOUT if checkout else self.SOURCES).items()):
            path = HERE / name
            with self.subTest(name=name):
                if not path.is_file() and not checkout:
                    self.skipTest('%s is not in this tree (an image holds what the frozen recipe names)' % name)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_11_the_image_serves_the_frozen_recipe_s_stage_of_the_mask_module(self):
        """SERVED_MASK is the frozen recipe's stage of FROZEN_MASK, rebuilt here from git, and FROZEN_MASK is what the
        frozen T16 report pins. This runs in a checkout only, because an image has no git."""
        if not in_checkout():
            self.skipTest('an image holds the served copy itself (test_11_the_pinned_sources_keep_their_bytes)')
        self.assertIn("REVISION = '%s'" % FROZEN_REVISION,
                      (HERE / 'frozen_recipe_context.py').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256(frozen_mask_source().encode('utf-8')).hexdigest(), FROZEN_MASK)
        report = json.loads((HERE / 'target-t16-attention-simulator.json').read_text(encoding='utf-8'))
        self.assertEqual([report[field]['attention_mask_replay.py'] for field in ('sources', 'sources_after')],
                         [FROZEN_MASK, FROZEN_MASK])
        self.assertEqual(hashlib.sha256(served_mask_source().encode('utf-8')).hexdigest(), SERVED_MASK)
        self.assertEqual(self.SOURCES['attention_mask_replay.py'], SERVED_MASK)

    def test_11_prepare_narrow_is_the_served_prepare_but_its_documented_changes(self):
        """prepare_narrow is a copy of the prepare the image serves, and PREPARE_NARROW_CHANGES lists every
        difference between the two. In an image the served copy is the file beside this test. In a checkout it is
        rebuilt from git."""
        served = served_mask_source()
        self.assertEqual(hashlib.sha256(served.encode('utf-8')).hexdigest(), SERVED_MASK)
        pinned_arguments, body = function_body(served, 'prepare')
        arguments, ours = function_body((HERE / 'extent_attention_replay.py').read_text(encoding='utf-8'),
                                        'prepare_narrow')
        self.assertEqual(arguments, [name for name in pinned_arguments if name not in ('capacity', 'short_context')])
        for before, after, why in self.PREPARE_NARROW_CHANGES:
            with self.subTest(change=why):
                self.assertEqual(body.count(before), 1, before)
            body = body.replace(before, after, 1)
        self.assertEqual(ours, body)

    def test_11_the_checkout_s_mask_module_differs_from_the_served_one_only_where_the_extent_path_never_runs(self):
        """Every CPU test here runs the checkout's mask module: PrepareNarrowTests compares against its prepare, and
        the readers call its execute. Every image runs the served one. The two must be the same bytes everywhere but
        validate_ticket and one import. The extent module never names validate_ticket, and it uses only execute and
        __file__ (test_11_imports_...). A change in either copy's prepare, execute or mask_position would leave this
        evidence describing code that no image runs."""
        if not in_checkout():
            self.skipTest('an image holds the served copy only')
        served = top_level(served_mask_source())
        ours = top_level((HERE / 'attention_mask_replay.py').read_text(encoding='utf-8'))
        self.assertEqual(set(ours), set(served))
        self.assertEqual({name for name in served if served[name] != ours[name]}, {'validate_ticket', 'imports'})
        self.assertEqual((served['imports'] - ours['imports'], ours['imports'] - served['imports']),
                         ({'from frozen_context_geometry import selected_geometry'}, {'import os'}))
        # test_pooled_attention_replay's checkout guard agrees. It is read, never imported: no image carries that
        # module, and the overlay closure tests (test_c2_overlay_closure, test_c2_image_overlay) refuse any import of
        # it here, lazy or not.
        pooled = ast.parse((HERE / 'test_pooled_attention_replay.py').read_text(encoding='utf-8'))
        shared = next(ast.literal_eval(node.value) for cls in pooled.body
                      if isinstance(cls, ast.ClassDef) and cls.name == 'SdpaModesTests' for node in cls.body
                      if isinstance(node, ast.Assign) and [target.id for target in node.targets] == ['PINNED'])
        self.assertEqual({name: shared[name] for name in set(shared) & set(self.CHECKOUT)},
                         {name: self.CHECKOUT[name] for name in set(shared) & set(self.CHECKOUT)})

    def test_11_imports_only_what_the_design_allows_and_never_the_pinned_validate_or_prepare(self):
        tree = ast.parse((HERE / 'extent_attention_replay.py').read_text(encoding='utf-8'))
        top = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                top.setdefault(node.module, set()).update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top.setdefault(alias.name, set())
        self.assertEqual(top, {
            'contextlib': {'ExitStack', 'contextmanager'}, 'os': set(), 'pathlib': {'Path'},
            'attention_mask_replay': set(), 'attention_fold_dma': {'device_layout_dma'},
            'attention_head_fold': {'parallel_groups'}, 'gdn_multitoken_conv': {'addresses', 'release_owned'},
            'pooled_attention_replay': {'QWEN_SDPA_EXTENT_MARKER', 'apply_sdpa_modes', 'sdpa_modes', 'validate_segments'}})
        lazy = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                and node not in tree.body for alias in node.names}
        self.assertEqual(lazy, {'torch', 'ttnn', 'logger'})
        used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name) and node.value.id == 'attention_mask_replay'}
        self.assertEqual(used, {'execute', '__file__'})
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn('validate_ticket', names)
        self.assertNotIn('ReplayAttentionReader', names)


if __name__ == '__main__':
    unittest.main()
