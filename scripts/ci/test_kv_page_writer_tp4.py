"""F-B1, the page-parallel K/V cache writer (kv_page_writer_tp4, its kernels and its binding in packed_ordered_cache).

What the CPU can hold it to - and what it cannot. NOT here: that bfloat8_b read-modify-write is idempotent on the hardware packer (plan unknown U1), which
is the card's (ordered_writer_tp4_card_test.py, the page64 arm). Here:

  * the unit table and the core layout on an 11x10 and a 13x10 grid; the kernel's key / slot / mask arithmetic against a brute-force reading of the
    positions it is given, for every offset class the stack writes;
  * the kernel source: the constants it hard-codes equal the module's (circular buffers, runtime-word counts, face offsets), and it compiles (g++
    -fsyntax-only against stub headers whose compile-time arguments are the module's), in every role, source and negative control;
  * the launch: one generic_op for K and V, 64 cores in the served 8x8 rectangle, the served compute kernel unchanged with Wt = 2 and one head, runtime
    words core by core, the L1 source's NOC pairs, the ORDERED mode's chain, program identity between the warm forward's mode and the capture's;
  * the binding: flag unset - nothing imported and the chained writer is what it was; flag set - the first call held, the second launches, any
    block / grid / source / record the page writer cannot take falls back to the chained writer with one FELL_BACK line;
  * THE EVIDENCE GATE: the repository's record has no page_writer block, so the lever refuses; a record with a block that matches the live design,
    width and source lets it engage; a stale signature, another width, another source, a missing proof, a changed kernel each refuse;
  * the audit: prep, the served chained writer on the shadow cache, the page launch, check - in that order - and audit_round's reading of the counters.
"""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import kv_page_writer_tp4 as kvpw
import kv_page_writer_tp4_smoke as smoke
import ordered_cache
import packed_ordered_cache as poc
import page_width_tp4
import tp_shapes
import verify_trace_t2 as t2
from packed_cache_writer import tile as cache_tile, tile_rows
from test_gdn_conv_windows_packed import Coord, DescriptorTTNN, FakeTensor, mesh

HERE = Path(__file__).resolve().parent
FOUR = {'QWEN_FAST_TP': '4'}
ON = dict(FOUR, QWEN_FAST_KV_PAGE_WRITER='1')
SEGMENTS = ((0, 16), (16, 32), (32, 48), (48, 64))
WIDTH = 4096
KERNELS = dict(reader='READER', writer='WRITER', compute='COMPUTE')


def four():
    return mock.patch.dict(os.environ, FOUR)


@contextlib.contextmanager
def environment(**values):
    base = {key: value for key, value in os.environ.items() if not key.startswith('QWEN_FAST_KV_PAGE_WRITER') and key != 'QWEN_FAST_TP'}
    base.update(values)
    with mock.patch.dict(os.environ, base, clear=True):
        yield


class ShardConfig(object):
    """What ttnn.MemoryConfig carries for the AttnPrep output: height-sharded L1, one (32, 256) shard per row on an 8x8 grid."""

    def __init__(self, shape=(32, 256), cores=64, layout='HEIGHT_SHARDED', buffer='L1', orientation='ROW_MAJOR'):
        end = Coord(7, cores // 8 - 1) if cores >= 8 else Coord(cores - 1, 0)
        self.memory_layout, self.buffer_type = layout, buffer
        grid = SimpleNamespace(ranges=lambda: [SimpleNamespace(start=Coord(0, 0), end=end)])
        self.shard_spec = SimpleNamespace(grid=grid, shape=shape, orientation=orientation)

    def __eq__(self, other):
        return False

    def __hash__(self):
        return 0


class FakeTTNN(DescriptorTTNN):
    """DescriptorTTNN plus what the page writer and its audit touch: sharded tensors, allocation, to_torch."""
    float32 = 'fp32'

    def __init__(self, chips=1):
        DescriptorTTNN.__init__(self, chips)
        self.uploaded = []

    def sharded(self, name, shape, config=None):
        tensor = self.tensor(name, shape, 'l1', 'bf16', 'tile')
        tensor._memory = config if config is not None else ShardConfig()
        return tensor

    def ReplicateTensorToMesh(self, device):  # noqa: N802
        return None

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        tensor = self.tensor('uploaded%d' % len(self.uploaded), tuple(value.shape), memory_config or 'dram', dtype, layout)
        tensor.host = value
        self.uploaded.append(tensor)
        return tensor

    def to_torch(self, tensor):
        return tensor.host


def caches(ttnn, count=2, blocks=4112):
    return [ttnn.tensor('cache%d' % index, (blocks, 1, 64, 256), 'dram', 'bf8', 'tile') for index in range(count)]


def block(ttnn, width=WIDTH, source='dram', blocks=4112):
    k, v = caches(ttnn, 2, blocks)
    if source == 'dram':
        packed = [ttnn.tensor('k', (1, 64, 32, 256), 'dram', 'bf16', 'tile'), ttnn.tensor('v', (1, 64, 32, 256), 'dram', 'bf16', 'tile')]
    else:
        packed = [ttnn.sharded('k', (1, 64, 32, 256)), ttnn.sharded('v', (1, 64, 32, 256))]
    positions = ttnn.tensor('positions', (64,), 'dram', 'int32', 'row_major')
    pages = ttnn.tensor('pages', (64, width), 'dram', 'int32', 'row_major')
    return dict(caches=[k, v], packed=packed, positions=positions, pages=pages)


def tiles(ttnn, width=WIDTH):
    return [cache_tile((first, last), ttnn.tensor('positions%d' % first, (32,), 'dram', 'int32', 'row_major'),
                       ttnn.tensor('pages%d' % first, (32, width), 'dram', 'int32', 'row_major'))
            for first, last in tile_rows(64)]


def passing_block(wt=kvpw.DEFAULT_WT, **changes):
    entry = dict(status='PASS', scope='full', failures=0, design_signature=kvpw.design_signature(wt), wt=wt, units=kvpw.unit_count(wt),
                 widths=[2052, 4096], sources=['dram', 'l1'], regimes=list(kvpw.PROOF_REGIMES), modes=list(kvpw.PROOF_MODES), caches=['k', 'v'],
                 seeds=[0, 1, 2], counts=dict(checks=100, exact=100),
                 proofs=dict(noop_served=dict(checks=10, exact=10), noop_page=dict(checks=10, exact=10), twice_page=dict(checks=10, exact=10)))
    entry.update(changes)
    return entry


# ---------------------------------------------------------------------------------------------
# The planners.
# ---------------------------------------------------------------------------------------------

class PlanTests(unittest.TestCase):
    def test_the_unit_table_is_the_cross_product_in_core_order(self):
        for wt in kvpw.WIDTHS:
            found = kvpw.units(wt)
            self.assertEqual(len(found), kvpw.unit_count(wt))
            self.assertEqual(kvpw.unit_count(wt), 128 // wt)
            self.assertEqual([unit['index'] for unit in found], list(range(len(found))))
            self.assertEqual(len({(u['cache'], u['group'], u['slot'], u['pair']) for u in found}), len(found))
            for unit in found:
                self.assertEqual(kvpw.unit_index(unit['cache'], unit['group'], unit['slot'], unit['pair'], wt), unit['index'])
                self.assertEqual((unit['first_column'], unit['first_row']), (unit['pair'] * wt, unit['group'] * 16))
        with self.assertRaises(ValueError):
            kvpw.unit_index(0, 4, 0, 0)
        with self.assertRaises(ValueError):
            kvpw.pairs(3)

    def test_every_cache_tile_row_column_pair_is_covered_once_per_slot(self):
        found = kvpw.units(2)
        for cache in range(2):
            for group in range(4):
                for slot in range(2):
                    columns = sorted(tile for unit in found if (unit['cache'], unit['group'], unit['slot']) == (cache, group, slot)
                                     for tile in range(unit['first_column'], unit['first_column'] + 2))
                    self.assertEqual(columns, list(range(8)))

    def test_the_default_layout_is_the_served_8x8_rectangle_on_either_grid(self):
        for grid in ((11, 10), (13, 10), (8, 8)):
            points, width = kvpw.core_layout(2, *grid)
            self.assertEqual((len(points), width, len(set(points))), (64, 8, 64), grid)
            self.assertEqual(points[0], (0, 0))
            self.assertEqual(points[63], (7, 7))
            self.assertEqual(points[9], (1, 1))
        with self.assertRaises(kvpw.Unsupported):
            kvpw.core_layout(2, 8, 7)
        with self.assertRaises(kvpw.Unsupported):
            kvpw.core_layout(2, 7, 10)

    def test_one_tile_units_need_128_cores_which_only_a_13x10_grid_has(self):
        points, width = kvpw.core_layout(1, 13, 10)
        self.assertEqual((len(points), width, max(y for x, y in points)), (128, 13, 9))
        self.assertEqual(len(set(points)), 128)
        with self.assertRaises(kvpw.Unsupported):
            kvpw.core_layout(1, 11, 10)
        for wt in (4, 8):
            points, width = kvpw.core_layout(wt, 11, 10)
            self.assertEqual((len(points), width), (128 // wt, 8))

    def test_the_ordered_chain_runs_group_to_group_within_one_cache_slot_and_columns(self):
        for unit in kvpw.units(2):
            before = kvpw.predecessor(unit, 2)
            if unit['group'] == 0:
                self.assertIsNone(before)
            else:
                other = kvpw.units(2)[before]
                self.assertEqual((other['cache'], other['slot'], other['pair'], other['group']),
                                 (unit['cache'], unit['slot'], unit['pair'], unit['group'] - 1))

    def test_keys_slots_and_masks_against_a_brute_force_reading(self):
        table = list(range(100, 100 + 4200))
        for start in list(range(0, 130)) + [262080, 262096, 262111, 262127]:
            positions = list(range(start, start + 16))
            tile_rows_hit = []
            for position in positions:
                key = (position // 64, (position % 64) // 32)
                if key not in tile_rows_hit:
                    tile_rows_hit.append(key)
            self.assertLessEqual(len(tile_rows_hit), 2)
            self.assertIsNone(kvpw.group_problem(positions))
            for slot in range(2):
                unit = kvpw.resolve_unit(positions, table, slot)
                if slot >= len(tile_rows_hit):
                    self.assertFalse(unit['valid'], (start, slot))
                    continue
                entry, tile_row = tile_rows_hit[slot]
                rows = [row for row, position in enumerate(positions) if (position // 64, (position % 64) // 32) == (entry, tile_row)]
                self.assertEqual((unit['valid'], unit['block'], unit['tile_row'], unit['rows']), (True, table[entry], tile_row, rows), (start, slot))
                self.assertEqual(unit['mask'], sum(1 << (positions[row] % 32) for row in rows))
            self.assertEqual(bool(kvpw.resolve_unit(positions, table, 1)['valid']), len(tile_rows_hit) == 2)

    def test_the_offset_classes_the_stack_writes(self):
        table = list(range(1000, 1000 + 4200))
        # aligned in a tile row, inside it, across the 32-row boundary, across the 64-token page boundary, the window's last 16 positions
        cases = {0: (1, 0xFFFF), 16: (1, 0xFFFF << 16), 17: (2, None), 31: (2, None), 32: (1, 0xFFFF), 48: (1, 0xFFFF << 16),
                 49: (2, None), 63: (2, None)}
        for offset, (slots, mask) in cases.items():
            positions = list(range(offset, offset + 16))
            valid = [kvpw.resolve_unit(positions, table, slot)['valid'] for slot in range(2)]
            self.assertEqual(valid, [True, slots == 2], offset)
            if mask is not None:
                self.assertEqual(kvpw.resolve_unit(positions, table, 0)['mask'], mask & 0xFFFFFFFF, offset)
        last = list(range(262128, 262144))
        unit = kvpw.resolve_unit(last, list(range(4096)), 0)
        self.assertEqual((unit['block'], unit['tile_row'], unit['mask']), (4095, 1, 0xFFFF << 16))
        crossing = list(range(49, 65))
        self.assertEqual([kvpw.resolve_unit(crossing, table, slot)['block'] for slot in range(2)], [table[0], table[1]])
        self.assertEqual([kvpw.resolve_unit(crossing, table, slot)['tile_row'] for slot in range(2)], [1, 0])

    def test_a_repeated_offset_is_one_mask_bit(self):
        positions = [5, 5, 5, 6] + [7] * 12
        unit = kvpw.resolve_unit(positions, list(range(10)), 0)
        self.assertEqual(unit['mask'], (1 << 5) | (1 << 6) | (1 << 7))
        self.assertEqual(unit['rows'], list(range(16)))

    def test_a_group_the_unit_table_cannot_hold_is_named(self):
        self.assertIn('touches 3 tile rows', kvpw.group_problem([0] * 14 + [40, 100]))
        self.assertIn('15 positions', kvpw.group_problem(list(range(15))))
        self.assertIn('non-negative', kvpw.group_problem([-1] + [0] * 15))

    def test_the_face_row_offsets_are_a_32_row_face_ordered_tile(self):
        self.assertEqual([kvpw.face_row_offset(0, 0), kvpw.face_row_offset(0, 1), kvpw.face_row_offset(1, 0)], [0, 512, 32])
        self.assertEqual([kvpw.face_row_offset(16, 0), kvpw.face_row_offset(16, 1), kvpw.face_row_offset(31, 1)], [1024, 1536, 1536 + 15 * 32])
        offsets = {kvpw.face_row_offset(row, half) for row in range(32) for half in range(2)}
        self.assertEqual(len(offsets), 64)
        self.assertEqual(max(offsets) + 32, 2048)

    def test_circular_buffers_are_small_and_the_compute_args_are_the_served_ones_at_one_block(self):
        self.assertEqual(kvpw.compute_args(2), (0, 1, 24, 25, 26, 16, 2, 1))
        served = poc.COMPUTE_ARGS
        self.assertEqual(kvpw.compute_args(8)[:6], served[:6])
        self.assertLess(kvpw.cb_bytes(2), 64 * 1024)
        self.assertLess(kvpw.cb_bytes(2), poc.cb_bytes(2052) // 6)
        table = {indices: (pages, dtype, size) for indices, pages, dtype, size, tiled in kvpw.cb_table(2)}
        self.assertEqual(table[(24, 25)], (4, 'bfloat16', 2048))
        self.assertEqual(table[(0,)], (4, 'bfloat8_b', 1088))
        self.assertEqual(kvpw.scratch_bytes(2) % 64, 0)
        self.assertGreaterEqual(kvpw.scratch_bytes(2), 320 + 16 * 2 * 2 * 64)
        self.assertGreaterEqual(kvpw.scratch_bytes(8), 512 + 2 * 8 * 1088 + 64)


# ---------------------------------------------------------------------------------------------
# The flags.
# ---------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_strict_values(self):
        with environment(**FOUR):
            self.assertFalse(kvpw.enabled({}))
            self.assertFalse(kvpw.enabled({'QWEN_FAST_KV_PAGE_WRITER': '0', 'QWEN_FAST_TP': '4'}))
            self.assertTrue(kvpw.enabled({'QWEN_FAST_KV_PAGE_WRITER': '1', 'QWEN_FAST_TP': '4'}))
            for bad in ('', 'true', '2', 'on'):
                with self.assertRaises(ValueError):
                    kvpw.enabled({'QWEN_FAST_KV_PAGE_WRITER': bad, 'QWEN_FAST_TP': '4'})

    def test_a_pair_process_cannot_turn_it_on(self):
        with self.assertRaisesRegex(ValueError, 'TP4 lever'):
            kvpw.enabled({'QWEN_FAST_KV_PAGE_WRITER': '1'})

    def test_the_audit_needs_the_lever(self):
        with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_KV_PAGE_WRITER=1'):
            kvpw.enabled({'QWEN_FAST_KV_PAGE_WRITER_AUDIT': '1', 'QWEN_FAST_TP': '4'})
        self.assertFalse(kvpw.audit_enabled({'QWEN_FAST_KV_PAGE_WRITER': '1', 'QWEN_FAST_TP': '4'}))
        self.assertTrue(kvpw.audit_enabled({'QWEN_FAST_KV_PAGE_WRITER': '1', 'QWEN_FAST_KV_PAGE_WRITER_AUDIT': '1', 'QWEN_FAST_TP': '4'}))
        self.assertFalse(kvpw.audit_enabled({'QWEN_FAST_TP': '4'}))

    def test_the_unit_width(self):
        self.assertEqual(kvpw.tiles_per_core({}), 2)
        for value in (1, 2, 4, 8):
            self.assertEqual(kvpw.tiles_per_core({'QWEN_FAST_KV_PAGE_WRITER_WT': str(value)}), value)
        with self.assertRaises(ValueError):
            kvpw.tiles_per_core({'QWEN_FAST_KV_PAGE_WRITER_WT': '3'})


# ---------------------------------------------------------------------------------------------
# The kernel source.
# ---------------------------------------------------------------------------------------------

def cpp():
    return (HERE / 'kv_page_writer_tp4.cpp').read_text(encoding='utf-8')


STUB = '''#pragma once
#include <cstdint>
#define FORCE_INLINE inline
#define tt_l1_ptr
#ifndef STUB_ARGS
#error "STUB_ARGS: the compile-time arguments of the role"
#endif
constexpr uint32_t stub_arguments[] = {STUB_ARGS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
constexpr uint32_t get_compile_time_arg_val(uint32_t index) { return stub_arguments[index]; }
template <typename T> T get_arg_val(uint32_t) { return T(); }
template <uint32_t Base> struct TensorAccessorArgs { constexpr uint32_t next_compile_time_args_offset() const { return Base + 1; } };
struct TensorAccessorStub {
    template <typename A> TensorAccessorStub(const A&, uint32_t, uint32_t) {}
    uint64_t get_noc_addr(uint32_t, uint32_t = 0) const { return 0; }
};
#define TensorAccessor TensorAccessorStub
inline uint64_t get_noc_addr(uint32_t, uint32_t, uint32_t) { return 0; }
inline void noc_async_read(uint64_t, uint32_t, uint32_t) {}
inline void noc_async_write(uint32_t, uint64_t, uint32_t) {}
inline void noc_async_read_barrier() {}
inline void noc_async_write_barrier() {}
inline void noc_async_atomic_barrier() {}
inline uint32_t get_semaphore(uint32_t) { return 0; }
inline void noc_semaphore_wait(volatile uint32_t*, uint32_t) {}
inline void noc_semaphore_set(volatile uint32_t*, uint32_t) {}
inline void noc_semaphore_inc(uint64_t, uint32_t) {}
inline void cb_reserve_back(uint32_t, uint32_t) {}
inline void cb_push_back(uint32_t, uint32_t) {}
inline void cb_wait_front(uint32_t, uint32_t) {}
inline void cb_pop_front(uint32_t, uint32_t) {}
inline uint32_t get_write_ptr(uint32_t) { return 0; }
inline uint32_t get_read_ptr(uint32_t) { return 0; }
inline void invalidate_l1_cache() {}
'''


class SourceTests(unittest.TestCase):
    def test_the_constants_the_kernel_hard_codes_are_the_modules(self):
        text = cpp()
        names = dict(cb_cache=kvpw.CB_CACHE, cb_in=kvpw.CB_IN, cb_scratch=kvpw.CB_SCRATCH, cb_out=kvpw.CB_OUT, cb_untilized=kvpw.CB_UNTILIZED,
                     cb_untilized2=kvpw.CB_UNTILIZED2, cb_untilized_in=kvpw.CB_UNTILIZED_IN, cb_meta=kvpw.CB_META, tile_bytes=kvpw.TILE,
                     cache_tile_bytes=kvpw.CACHE_TILE, row_tiles=kvpw.ROW_TILES, face_bytes=512, face_row_bytes=32)
        for name, value in names.items():
            match = re.search(r'constexpr uint32_t %s = (\d+);' % name, text)
            self.assertIsNotNone(match, name)
            self.assertEqual(int(match.group(1)), value, name)
        self.assertIn('((position >> 6) << 1) | ((position >> 5) & 1u)', text)
        self.assertIn('(block * 2u + tile_row) * tiles_per_tile_row + column', text)
        self.assertIn('((row >> 4) * 2u + half) * face_bytes + (row & 15u) * face_row_bytes', text)
        self.assertEqual(kvpw.key_of(262143), (4095 << 1) | 1)

    def test_the_source_is_lf_only_and_names_its_roles(self):
        data = (HERE / 'kv_page_writer_tp4.cpp').read_bytes()
        self.assertNotIn(b'\r', data)
        for role in kvpw.ROLE_DEFINES.values():
            self.assertIn(role, data.decode())
        self.assertEqual(kvpw.source_text(), data.decode())
        self.assertEqual(kvpw.source_sha256(), hashlib.sha256(data).hexdigest())

    def test_the_head_row_spans_are_face_0_and_face_1_at_bytes_0_and_512(self):
        text = cpp()
        self.assertIn('get_noc_addr(source_x, source_y, tile_address + face_bytes)', text)
        self.assertIn('packed.get_noc_addr(page, face_bytes)', text)
        self.assertNotIn('1024', text.replace('1024 mantissa bytes', ''))

    def test_the_semaphore_and_fences_are_the_repos_idioms(self):
        text = cpp()
        self.assertIn('noc_semaphore_wait(arrived, 1)', text)
        self.assertIn('noc_semaphore_inc(get_noc_addr(target_x, target_y, get_semaphore(semaphore_id)), 1)', text)
        self.assertIn('noc_async_atomic_barrier()', text)
        self.assertGreaterEqual(text.count('invalidate_l1_cache()'), 6)

    @unittest.skipUnless(shutil.which('g++'), 'g++ not found')
    def test_every_role_source_and_negative_control_compiles_against_the_stub_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            include = Path(directory) / 'api' / 'dataflow'
            include.mkdir(parents=True)
            (include / 'dataflow_api.h').write_text(STUB)
            arguments = {
                'KVPW_ROLE_READER': [16, 2, 16384, 64, kvpw.READER_WORDS],
                'KVPW_ROLE_WRITER': [2, kvpw.WRITER_WORDS],
                'KVPW_ROLE_AUDIT_PREP': [16, 2, 16384, 64, kvpw.AUDIT_PREP_WORDS],
                'KVPW_ROLE_AUDIT_CHECK': [16, 2, 16384, 64, kvpw.AUDIT_CHECK_WORDS],
            }
            variants = [(role, []) for role in arguments]
            variants += [('KVPW_ROLE_READER', ['KVPW_SRC_L1']), ('KVPW_ROLE_READER', ['KVPW_NEG_DROP']),
                         ('KVPW_ROLE_READER', ['KVPW_NEG_SLOT', 'KVPW_SRC_L1']), ('KVPW_ROLE_AUDIT_PREP', ['KVPW_NEG_SLOT']),
                         ('KVPW_ROLE_AUDIT_CHECK', ['KVPW_NEG_DROP'])]
            for wt in (1, 2, 8):
                variants.append(('KVPW_ROLE_READER', ['WT_%d' % wt]))
            for role, extra in variants:
                words = list(arguments[role])
                for item in extra:
                    if item.startswith('WT_'):
                        words[1] = int(item[3:])
                command = ['g++', '-std=c++17', '-fsyntax-only', '-Wall', '-Wno-unused-variable', '-Wno-unused-but-set-variable',
                           '-Wno-unused-function', '-Wno-unused-parameter', '-I', directory, '-D%s' % role,
                           '-DSTUB_ARGS=%s' % ','.join(str(word) for word in words)]
                command += ['-D%s=1' % item for item in extra if not item.startswith('WT_')]
                command += ['-DKVPW_SRC_SHA=0x12345678', '-x', 'c++', '-include', str(include / 'dataflow_api.h'), str(HERE / 'kv_page_writer_tp4.cpp')]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, '%s %s\n%s' % (role, extra, result.stderr[-2000:]))

    @unittest.skipUnless(shutil.which('g++'), 'g++ not found')
    def test_the_compile_catches_a_runtime_word_count_that_drifted(self):
        with tempfile.TemporaryDirectory() as directory:
            include = Path(directory) / 'api' / 'dataflow'
            include.mkdir(parents=True)
            (include / 'dataflow_api.h').write_text(STUB)
            command = ['g++', '-std=c++17', '-fsyntax-only', '-I', directory, '-DKVPW_ROLE_READER', '-DSTUB_ARGS=16,2,16384,64,39',
                       '-x', 'c++', str(HERE / 'kv_page_writer_tp4.cpp')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('static assertion', result.stderr)


# ---------------------------------------------------------------------------------------------
# The evidence gate.
# ---------------------------------------------------------------------------------------------

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for name in ('ordered_cache.py', 'ordered_writer_evidence_tp4.json'):
            shutil.copyfile(page_width_tp4.HERE / name, self.dir / name)
        self.path = self.dir / 'ordered_writer_evidence_tp4.json'

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, block=None, **overrides):
        """Write the shipped record with `block` as its page_writer (None: no block at all) and the overrides; returns the file's sha256."""
        evidence = json.loads(page_width_tp4.EVIDENCE.read_text())
        evidence.update(overrides)
        if block is not None:
            evidence['page_writer'] = block
        payload = (json.dumps(evidence, indent=1) + '\n').encode()
        self.path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def problems(self, expected, **options):
        return kvpw.evidence_problems(path=self.path, expected=expected, sources_root=self.dir, **options)

    def test_the_shipped_record_has_no_page_block_so_the_lever_refuses(self):
        evidence = json.loads(page_width_tp4.EVIDENCE.read_text())
        self.assertNotIn('page_writer', evidence)
        problems = kvpw.evidence_problems()
        self.assertEqual(problems, ['no page_writer record'])

    def test_a_missing_or_unpinned_record_refuses_before_the_block_is_read(self):
        self.assertIn('does not stand', self.problems('0' * 64)[0])
        self.path.unlink()
        self.assertIn('is missing', self.problems('0' * 64)[0])

    def test_a_matching_block_lets_it_engage_at_the_recorded_widths_and_sources(self):
        digest = self.record(passing_block())
        self.assertEqual(self.problems(digest), [])
        for width in (2052, 4096):
            for source in ('dram', 'l1'):
                self.assertEqual(self.problems(digest, width=width, source=source), [])

    def test_a_width_or_a_source_the_record_does_not_cover_refuses(self):
        digest = self.record(passing_block(widths=[4096], sources=['dram']))
        self.assertTrue(any('widths lack' in p for p in self.problems(digest)))
        self.assertTrue(any('width 2052' in p for p in self.problems(digest, width=2052)))
        digest = self.record(passing_block())
        self.assertEqual(self.problems(digest, width=1024)[0].split(' is not among')[0], 'page-table width 1024')

    def test_a_changed_design_refuses_whatever_the_block_says_about_itself(self):
        digest = self.record(passing_block(design_signature='0' * 64))
        problems = self.problems(digest)
        self.assertTrue(any('qualified design' in p for p in problems), problems)
        with mock.patch.object(kvpw, 'source_sha256', return_value='1' * 64):
            digest = self.record(passing_block())
        problems = self.problems(digest)
        self.assertTrue(any('qualified design' in p for p in problems), problems)
        digest = self.record(passing_block())
        self.assertTrue(any('qualified design' in p for p in self.problems(digest, compute_sha256='2' * 64)))

    def test_each_part_of_the_block_is_required(self):
        mutations = {
            'status': dict(status='FAIL'),
            'scope': dict(scope='reduced'),
            'failures': dict(failures=1),
            'unit width': dict(wt=4),
            'counts': dict(counts=dict(checks=10, exact=9)),
            'no counts': dict(counts={}),
            'one regime': dict(regimes=['exact']),
            'one mode': dict(modes=['eager']),
            'one cache': dict(caches=['k']),
            'no noop for the served writer': dict(proofs=dict(noop_page=dict(checks=1, exact=1), twice_page=dict(checks=1, exact=1))),
            'an inexact noop': dict(proofs=dict(noop_served=dict(checks=2, exact=1), noop_page=dict(checks=1, exact=1), twice_page=dict(checks=1, exact=1))),
        }
        for label, change in mutations.items():
            digest = self.record(passing_block(**change))
            self.assertTrue(self.problems(digest), label)
        digest = self.record(None)
        self.assertEqual(self.problems(digest), ['no page_writer record'])

    def test_a_block_is_not_enough_without_the_e1_record_beneath_it(self):
        digest = self.record(passing_block(), status='FAIL')
        self.assertIn('does not stand', self.problems(digest)[0])
        digest = self.record(passing_block(), status='PASS')
        (self.dir / 'ordered_cache.py').write_bytes((self.dir / 'ordered_cache.py').read_bytes() + b'\n')
        self.assertIn('does not stand', self.problems(digest)[0])

    def test_the_signature_binds_the_kernel_the_pinned_compute_and_the_geometry(self):
        base = kvpw.design_signature(2)
        self.assertEqual(base, kvpw.design_signature(2))
        self.assertNotEqual(base, kvpw.design_signature(4))
        self.assertNotEqual(base, kvpw.design_signature(2, compute_sha256='3' * 64))
        with mock.patch.object(kvpw, 'source_sha256', return_value='4' * 64):
            self.assertNotEqual(base, kvpw.design_signature(2))
        self.assertEqual(kvpw.design(2)['compute_sha256'], ordered_cache.HASHES['compute'])
        self.assertEqual(kvpw.design(2)['compute_args'], [0, 1, 24, 25, 26, 16, 2, 1])


# ---------------------------------------------------------------------------------------------
# The launch.
# ---------------------------------------------------------------------------------------------

class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.ttnn = FakeTTNN(chips=1)
        self.mesh = mesh(chips=1, grid=(13, 10))
        self.env = four()
        self.env.start()
        self.addCleanup(self.env.stop)

    def launch(self, source='dram', ordered=False, wt=2, grid=(13, 10), width=WIDTH, **options):
        tensors = block(self.ttnn, width=width, source=source)
        cores = kvpw.source_mode(self.ttnn, tensors['packed'][0])[1] if source == 'l1' else None
        self.mesh = mesh(chips=1, grid=grid)
        texts = dict(reader='READ', writer='WRITE')
        kvpw.launch(self.ttnn, self.mesh, tensors, KERNELS, texts, wt=wt, source=source, source_cores=cores, ordered=ordered, **options)
        name, listed, program = self.ttnn.programs()[-1]
        return tensors, listed, program[((0, 0), (0, 0))]

    def test_one_generic_op_for_the_layers_k_and_v(self):
        tensors, listed, chip = self.launch()
        self.assertEqual(len(self.ttnn.programs()), 1)
        self.assertEqual(listed, [*tensors['caches'], *tensors['packed'], tensors['positions'], tensors['pages']])
        self.assertEqual([kernel['cores'] for kernel in chip['kernels']], [('set', (('range', (0, 0), (7, 7)),))] * 3)
        self.assertEqual([kernel['source'] for kernel in chip['kernels']], ['READ', 'WRITE', 'COMPUTE'])
        self.assertEqual({kernel['source_type'] for kernel in chip['kernels']}, {'source-code'})
        self.assertEqual(chip['semaphores'][0].id, 0)
        self.assertEqual(chip['semaphores'][0].initial_value, 0)

    def test_the_compute_kernel_is_the_served_one_with_wt_and_one_head(self):
        for wt in (1, 2, 4, 8):
            tensors, listed, chip = self.launch(wt=wt)
            reader, writer, compute = chip['kernels']
            self.assertEqual(compute['compile'], [0, 1, 24, 25, 26, 16, wt, 1])
            self.assertEqual(compute['source'], 'COMPUTE')
            self.assertEqual(compute['config'].fp32_dest_acc_en, False)
            self.assertEqual(len(reader['runtime']), kvpw.unit_count(wt))

    def test_the_circular_buffers_are_the_tables(self):
        tensors, listed, chip = self.launch()
        self.assertEqual([(tuple(f.buffer_index for f in cb.format_descriptors), cb.total_size) for cb in chip['cbs']],
                         [(indices, pages * size) for indices, pages, dtype, size, tiled in kvpw.cb_table(2)])
        self.assertEqual(sum(cb.total_size for cb in chip['cbs']), kvpw.cb_bytes(2))

    def test_compile_arguments_defines_and_word_counts(self):
        tensors, listed, chip = self.launch()
        reader, writer, compute = chip['kernels']
        self.assertEqual(reader['compile'][:5], [16, 2, WIDTH * 4, 64, kvpw.READER_WORDS])
        self.assertEqual(reader['compile'][5:], [7, 1] * 4, 'cache, positions, pages and the DRAM prepared K/V')
        self.assertEqual(writer['compile'], [2, kvpw.WRITER_WORDS, 7, 1])
        reader_define, writer_define = dict(reader['defines']), dict(writer['defines'])
        self.assertEqual((reader_define['KVPW_ROLE_READER'], 'KVPW_SRC_L1' in reader_define), ('1', False))
        self.assertEqual(writer_define['KVPW_ROLE_WRITER'], '1')
        self.assertEqual(reader_define['KVPW_SRC_SHA'], kvpw.sha_define())
        self.assertTrue(all(len(args) == kvpw.READER_WORDS for x, y, args in reader['runtime']))
        self.assertTrue(all(len(args) == kvpw.WRITER_WORDS for x, y, args in writer['runtime']))
        self.assertTrue(all(args == () for x, y, args in compute['runtime']))

    def test_the_l1_source_drops_the_prepared_accessor_and_passes_the_shard_cores(self):
        tensors, listed, chip = self.launch(source='l1')
        reader, writer, compute = chip['kernels']
        self.assertEqual(reader['compile'][5:], [7, 1] * 3)
        self.assertEqual(dict(reader['defines'])['KVPW_SRC_L1'], '1')
        shard_cores = kvpw.shard_cores(tensors['packed'][0].memory_config().shard_spec)
        self.assertEqual(shard_cores[:3], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(shard_cores[63], (7, 7))
        by_core = {(x, y): args for x, y, args in reader['runtime']}
        for unit in kvpw.units(2):
            args = by_core[(unit['index'] % 8, unit['index'] // 8)]
            for row in range(16):
                x, y = shard_cores[unit['first_row'] + row]
                self.assertEqual(tuple(args[8 + 2 * row:10 + 2 * row]), (x + 1, y + 2), 'the FakeDevice offsets logical cores by (1, 2)')

    def test_runtime_words_core_by_core(self):
        tensors, listed, chip = self.launch()
        reader, writer, compute = chip['kernels']
        address = lambda value: value.shards[0].address
        by_reader = {(x, y): args for x, y, args in reader['runtime']}
        by_writer = {(x, y): args for x, y, args in writer['runtime']}
        for unit in kvpw.units(2):
            core = (unit['index'] % 8, unit['index'] // 8)
            cache, packed = unit['cache'], unit['cache']
            self.assertEqual(by_reader[core][:8], (address(tensors['caches'][cache]), address(tensors['positions']), address(tensors['pages']),
                                                   address(tensors['packed'][packed]), unit['first_row'], unit['slot'], unit['first_column'], 0))
            self.assertEqual(by_reader[core][8:], (0,) * 32, 'DRAM source: no NOC pairs')
            self.assertEqual(by_writer[core], (address(tensors['caches'][cache]), unit['first_column'], 0, 0, 0, 0, 0, 0))

    def test_ordered_mode_chains_group_to_group_and_changes_only_runtime_words(self):
        tensors, listed, plain = self.launch(ordered=False)
        tensors2, listed2, ordered = self.launch(ordered=True)
        for a, b in zip(plain['kernels'], ordered['kernels']):
            self.assertEqual({k: v for k, v in a.items() if k not in ('runtime',)}, {k: v for k, v in b.items() if k not in ('runtime',)})
        reader, writer, compute = ordered['kernels']
        by_reader = {(x, y): args for x, y, args in reader['runtime']}
        by_writer = {(x, y): args for x, y, args in writer['runtime']}
        for unit in kvpw.units(2):
            core = (unit['index'] % 8, unit['index'] // 8)
            self.assertEqual(by_reader[core][7], int(unit['group'] > 0), unit)
            self.assertEqual(by_writer[core][2], int(unit['group'] < 3), unit)
            if unit['group'] < 3:
                next_unit = kvpw.units(2)[kvpw.unit_index(unit['cache'], unit['group'] + 1, unit['slot'], unit['pair'], 2)]
                self.assertEqual(tuple(by_writer[core][3:5]), (next_unit['index'] % 8 + 1, next_unit['index'] // 8 + 2))
            else:
                self.assertEqual(tuple(by_writer[core][3:5]), (0, 0))

    def test_the_two_modes_are_one_program_for_the_jit_and_the_program_cache(self):
        a = self.launch(ordered=False)[2]['kernels']
        b = self.launch(ordered=True)[2]['kernels']
        for x, y in zip(a, b):
            self.assertEqual((x['source'], x['compile'], x['defines'], x['cores']), (y['source'], y['compile'], y['defines'], y['cores']))

    def test_the_grid_is_read_from_the_device_and_8x8_is_enough(self):
        for grid in ((11, 10), (13, 10), (8, 8)):
            tensors, listed, chip = self.launch(grid=grid)
            self.assertEqual(chip['kernels'][0]['cores'], ('set', (('range', (0, 0), (7, 7)),)))
        with self.assertRaises(kvpw.Unsupported):
            self.launch(grid=(8, 7))
        tensors, listed, chip = self.launch(wt=1, grid=(13, 10))
        self.assertEqual(len(chip['kernels'][0]['runtime']), 128)
        with self.assertRaises(kvpw.Unsupported):
            self.launch(wt=1, grid=(11, 10))

    def test_negative_controls_are_defines_only(self):
        a = self.launch()[2]['kernels']
        b = self.launch(negative='drop')[2]['kernels']
        self.assertIn(('KVPW_NEG_DROP', '1'), b[0]['defines'])
        self.assertNotIn(('KVPW_NEG_DROP', '1'), a[0]['defines'])
        self.assertEqual([(x, y, len(args)) for x, y, args in a[0]['runtime']], [(x, y, len(args)) for x, y, args in b[0]['runtime']])
        with self.assertRaises(ValueError):
            self.launch(negative='bogus')

    def test_aliased_buffers_are_refused_before_anything_launches(self):
        tensors = block(self.ttnn)
        tensors['positions'].shards = tensors['pages'].shards
        with self.assertRaisesRegex(ValueError, 'must not alias'):
            kvpw.launch(self.ttnn, self.mesh, tensors, KERNELS, dict(reader='R', writer='W'))
        self.assertEqual(self.ttnn.programs(), [])

    def test_a_four_chip_mesh_gets_four_programs(self):
        ttnn = FakeTTNN(chips=4)
        tensors = block(ttnn)
        kvpw.launch(ttnn, mesh(chips=4, grid=(13, 10)), tensors, KERNELS, dict(reader='R', writer='W'))
        name, listed, program = ttnn.programs()[0]
        self.assertEqual(sorted(program), [((0, c), (0, c)) for c in range(4)])
        words = {chip: program[((0, chip), (0, chip))]['kernels'][0]['runtime'][0][2][0] for chip in range(4)}
        self.assertEqual(len(set(words.values())), 4, 'each chip its own cache address')

    def test_the_tensor_checks(self):
        tensors = block(self.ttnn)
        kvpw.validate_tensors(self.ttnn, tensors['caches'], tensors['packed'], tensors['positions'], tensors['pages'], WIDTH)
        bad = {
            'cache dtype': lambda t: setattr(t['caches'][0], 'dtype', 'bf16'),
            'two heads': lambda t: setattr(t['caches'][0], 'shape', (4112, 2, 64, 256)),
            'prepared shape': lambda t: setattr(t['packed'][1], 'shape', (1, 32, 32, 256)),
            'positions dtype': lambda t: setattr(t['positions'], 'dtype', 'uint32'),
            'table memory': lambda t: setattr(t['pages'], '_memory', 'l1'),
            'fewer blocks than the table is wide': lambda t: setattr(t['caches'][1], 'shape', (2000, 1, 64, 256)),
        }
        for label, mutate in bad.items():
            tensors = block(self.ttnn)
            mutate(tensors)
            with self.subTest(label), self.assertRaises(ValueError):
                kvpw.validate_tensors(self.ttnn, tensors['caches'], tensors['packed'], tensors['positions'], tensors['pages'], WIDTH)

    def test_the_source_mode_reads_the_placement(self):
        self.assertEqual(kvpw.source_mode(self.ttnn, self.ttnn.tensor('k', (1, 64, 32, 256), 'dram'))[0], 'dram')
        mode, cores = kvpw.source_mode(self.ttnn, self.ttnn.sharded('k', (1, 64, 32, 256)))
        self.assertEqual((mode, len(cores)), ('l1', 64))
        for config in (ShardConfig(shape=(32, 128)), ShardConfig(cores=32), ShardConfig(layout='WIDTH_SHARDED'), ShardConfig(buffer='DRAM'),
                       ShardConfig(orientation='COL_MAJOR')):
            with self.subTest(config=vars(config)), self.assertRaises(kvpw.Unsupported):
                kvpw.source_mode(self.ttnn, self.ttnn.sharded('k', (1, 64, 32, 256), config))
        with self.assertRaises(kvpw.Unsupported):
            kvpw.source_mode(self.ttnn, self.ttnn.tensor('k', (1, 64, 32, 256), 'l1'))


# ---------------------------------------------------------------------------------------------
# The writer and its binding.
# ---------------------------------------------------------------------------------------------

class WriterTests(unittest.TestCase):
    def setUp(self):
        self.ttnn = FakeTTNN(chips=1)
        self.mesh = mesh(chips=1, grid=(13, 10))
        self.lines = []
        self.patches = [mock.patch.dict(os.environ, ON), mock.patch.object(kvpw, 'log_line', self.lines.append),
                        mock.patch.object(kvpw, 'evidence_problems', return_value=[]), mock.patch.object(kvpw, '_NOTED', set())]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        t2.take()
        self.addCleanup(t2.take)

    def chained(self, spans=SEGMENTS, launch_rows=64, source='dram', width=WIDTH, **options):
        tensors = block(self.ttnn, width=width, source=source)
        arguments = dict(positions=tensors['positions'], pages=tensors['pages'], spans=spans, tiles=tiles(self.ttnn, width), launch_rows=launch_rows)
        arguments.update(options)
        writer = poc.ChainedOrderedCacheWriter(self.mesh, self.ttnn, KERNELS, **arguments)
        return writer, tensors

    def call(self, writer, cache, packed):
        writer(cache, packed, update_idxs_tensor=SimpleNamespace(shape=(64,)), page_table=SimpleNamespace(shape=(64, WIDTH)))

    def pair(self, writer, tensors):
        self.call(writer, tensors['caches'][0], tensors['packed'][0])
        self.call(writer, tensors['caches'][1], tensors['packed'][1])

    def test_the_first_call_is_held_the_second_launches_both_and_the_calls_are_counted_as_ever(self):
        writer, tensors = self.chained()
        self.assertIsNotNone(writer.page)
        self.call(writer, tensors['caches'][0], tensors['packed'][0])
        self.assertEqual(self.ttnn.programs(), [])
        self.assertFalse(writer.page.idle())
        self.call(writer, tensors['caches'][1], tensors['packed'][1])
        self.assertEqual(len(self.ttnn.programs()), 1)
        self.assertTrue(writer.page.idle())
        self.assertEqual((writer.calls, writer.page.launches), (2, 1))
        self.assertEqual(t2.take(), {'kv_chains': 2})
        self.assertEqual([call for call in self.ttnn.calls if call[0] != 'generic_op'], [], 'no copy to interleaved DRAM')
        self.assertTrue(any(line.startswith(kvpw.ENGAGED) and 'ordered=0' in line and 'source=dram' in line for line in self.lines), self.lines)

    def test_the_l1_source_launches_without_the_interleaving_copy(self):
        writer, tensors = self.chained(source='l1')
        self.pair(writer, tensors)
        self.assertEqual(len(self.ttnn.programs()), 1)
        self.assertEqual([call for call in self.ttnn.calls if call[0] == 'interleave'], [])
        self.assertTrue(any(' source=l1 ' in line for line in self.lines), self.lines)

    def test_many_layers_many_pairs(self):
        writer, tensors = self.chained()
        for unused in range(5):
            self.pair(writer, tensors)
        self.assertEqual((len(self.ttnn.programs()), writer.calls), (5, 10))

    def test_two_k_calls_in_a_row_raise(self):
        writer, tensors = self.chained()
        self.call(writer, tensors['caches'][0], tensors['packed'][0])
        with self.assertRaisesRegex(ValueError, 'K then V'):
            self.call(writer, tensors['caches'][0], tensors['packed'][1])

    def test_the_warm_forwards_single_span_is_the_ordered_mode(self):
        writer, tensors = self.chained(spans=((0, 64),))
        self.assertTrue(writer.page.ordered)
        self.pair(writer, tensors)
        reader = self.ttnn.programs()[0][2][((0, 0), (0, 0))]['kernels'][0]
        self.assertTrue(any(args[7] == 1 for x, y, args in reader['runtime']))
        self.assertTrue(any(' ordered=1 ' in line for line in self.lines))

    def test_blocks_the_page_writer_cannot_take_fall_back_to_the_chained_writer_with_one_line(self):
        cases = {
            'launch rows 32': dict(launch_rows=32),
            'other spans': dict(spans=((0, 24), (24, 64))),
            'three users': dict(spans=((0, 16), (16, 32), (32, 64))),
        }
        for label, options in cases.items():
            self.lines[:] = []
            kvpw._NOTED.clear()
            writer, tensors = self.chained(**options)
            self.assertIsNone(writer.page, label)
            self.assertEqual(len([line for line in self.lines if kvpw.FELL_BACK in line]), 1, label)
            before = len([call for call in self.ttnn.calls if call[0] == 'generic_op'])
            self.pair(writer, tensors)
            self.assertGreater(len([call for call in self.ttnn.calls if call[0] == 'generic_op']), before, label)

    def test_a_grid_under_8x8_falls_back(self):
        tensors = block(self.ttnn)
        small = mesh(chips=1, grid=(8, 7))
        self.assertIsNone(kvpw.attach(small, self.ttnn, KERNELS, positions=tensors['positions'], pages=tensors['pages'], spans=SEGMENTS,
                                      launch_rows=64))
        self.assertTrue(any(kvpw.FELL_BACK in line and 'this grid is 8x7' in line for line in self.lines), self.lines)
        self.assertIsNotNone(kvpw.attach(mesh(chips=1, grid=(11, 10)), self.ttnn, KERNELS, positions=tensors['positions'],
                                         pages=tensors['pages'], spans=SEGMENTS, launch_rows=64))

    def test_a_source_the_kernel_cannot_read_falls_back_at_the_first_call_before_anything_is_held(self):
        writer, tensors = self.chained()
        tensors['packed'][0]._memory = ShardConfig(shape=(32, 128))
        self.call(writer, tensors['caches'][0], tensors['packed'][0])
        self.assertTrue(writer.page.dead)
        self.assertEqual(len(self.ttnn.programs()), 1, 'the chained writer served that call')
        self.call(writer, tensors['caches'][1], tensors['packed'][1])
        self.assertEqual(len(self.ttnn.programs()), 2)
        self.assertEqual(len([line for line in self.lines if kvpw.FELL_BACK in line]), 1)

    def test_a_missing_record_falls_back_and_says_why_once(self):
        with mock.patch.object(kvpw, 'evidence_problems', return_value=['no page_writer record']):
            writer, tensors = self.chained()
            other, tensors2 = self.chained()
        self.assertIsNone(writer.page)
        self.assertIsNone(other.page)
        fell = [line for line in self.lines if kvpw.FELL_BACK in line]
        self.assertEqual(len(fell), 1)
        self.assertIn('no page_writer record', fell[0])
        self.pair(writer, tensors)
        self.assertEqual(len([call for call in self.ttnn.calls if call[0] == 'generic_op']), 2, 'the two chained launches')

    def test_the_record_is_asked_about_this_width_and_this_source(self):
        asked = []
        with mock.patch.object(kvpw, 'evidence_problems', side_effect=lambda *args, **kwargs: asked.append((args, kwargs)) or []):
            writer, tensors = self.chained(source='l1')
            self.pair(writer, tensors)
        self.assertEqual(asked[0][0], (2, WIDTH))
        self.assertEqual(asked[-1][0], (2, WIDTH, 'l1'))

    def test_unset_and_zero_leave_the_chained_writer_exactly_as_it_was(self):
        for value in (None, '0'):
            environment_ = dict(FOUR)
            if value is not None:
                environment_['QWEN_FAST_KV_PAGE_WRITER'] = value
            with environment(**environment_):
                writer, tensors = self.chained()
                self.assertIsNone(writer.page)
                self.pair(writer, tensors)
        programs = [call for call in self.ttnn.calls if call[0] == 'generic_op']
        self.assertEqual(len(programs), 4)
        self.assertEqual(self.lines, [])
        self.assertEqual(t2.take(), {'kv_chains': 4})

    def test_with_the_flag_unset_the_module_is_never_imported(self):
        code = ("import os, sys\nsys.path.insert(0, %r)\nfrom types import SimpleNamespace\n"
                "import packed_ordered_cache as poc\nimport test_packed_ordered_cache as t\n"
                "from test_gdn_conv_windows_packed import DescriptorTTNN, mesh\nttnn = DescriptorTTNN()\n"
                "cache, packed, positions, pages = t.block(ttnn)\n"
                "writer = poc.ChainedOrderedCacheWriter(mesh(), ttnn, t.KERNELS, positions=positions, pages=pages, spans=t.SEGMENTS,"
                " tiles=t.tiles(ttnn), launch_rows=64)\n"
                "writer(cache, packed, update_idxs_tensor=SimpleNamespace(shape=(64,)), page_table=SimpleNamespace(shape=(64, t.WIDTH)))\n"
                "print(writer.page is None, 'kv_page_writer_tp4' in sys.modules)\n") % str(HERE)
        env = {key: value for key, value in os.environ.items() if not key.startswith('QWEN_FAST_KV_PAGE') and key != 'QWEN_FAST_TP'}
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, env=env, cwd=str(HERE))
        self.assertEqual(result.stdout.strip().splitlines()[-1:], ['True False'], result.stderr[-1200:])

    def test_a_misspelled_flag_value_raises_instead_of_falling_back(self):
        with environment(QWEN_FAST_TP='4', QWEN_FAST_KV_PAGE_WRITER='yes'):
            with self.assertRaises(ValueError):
                self.chained()

    def test_the_flag_at_the_pair_raises(self):
        with environment(QWEN_FAST_KV_PAGE_WRITER='1'):
            with self.assertRaisesRegex(ValueError, 'TP4 lever'):
                self.chained(width=2052)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.ttnn = FakeTTNN(chips=1)
        self.mesh = mesh(chips=1, grid=(13, 10))
        self.lines = []
        self.patches = [mock.patch.dict(os.environ, dict(ON, QWEN_FAST_KV_PAGE_WRITER_AUDIT='1')), mock.patch.object(kvpw, 'log_line', self.lines.append),
                        mock.patch.object(kvpw, 'evidence_problems', return_value=[]), mock.patch.object(kvpw, '_NOTED', set()),
                        mock.patch.object(kvpw, '_REGISTRY', [])]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_prep_then_the_served_writer_on_the_shadow_then_the_page_launch_then_check(self):
        tensors = block(self.ttnn)
        writer = poc.ChainedOrderedCacheWriter(self.mesh, self.ttnn, KERNELS, positions=tensors['positions'], pages=tensors['pages'],
                                               spans=SEGMENTS, tiles=tiles(self.ttnn), launch_rows=64)
        self.assertIsNotNone(writer.page.audit)
        self.assertEqual(kvpw._REGISTRY, [writer.page])
        writer(tensors['caches'][0], tensors['packed'][0], update_idxs_tensor=SimpleNamespace(shape=(64,)), page_table=SimpleNamespace(shape=(64, WIDTH)))
        writer(tensors['caches'][1], tensors['packed'][1], update_idxs_tensor=SimpleNamespace(shape=(64,)), page_table=SimpleNamespace(shape=(64, WIDTH)))
        programs = [call for call in self.ttnn.calls if call[0] == 'generic_op']
        self.assertEqual(len(programs), 5, 'prep, two served launches on the shadow, the page launch, check')
        sources = [[kernel['source'] for kernel in list(call[2].values())[0]['kernels']] for call in programs]
        self.assertEqual(sources[0], [kvpw.source_text()])
        self.assertEqual(sources[1], ['READER', 'WRITER', 'COMPUTE'])
        self.assertEqual(sources[2], ['READER', 'WRITER', 'COMPUTE'])
        self.assertEqual(sources[3], [kvpw.source_text(), kvpw.source_text(), 'COMPUTE'])
        self.assertEqual(sources[4], [kvpw.source_text()])
        audit = writer.page.audit
        served = [call[1] for call in programs[1:3]]
        self.assertEqual([tuple(listed[0].shape) for listed in served], [(kvpw.AUDIT_BLOCKS, 1, 64, 256)] * 2)
        self.assertEqual([listed[2] for listed in served], [audit.positions] * 2, 'the shadow positions')
        self.assertEqual([listed[3] for listed in served], [audit.table] * 2, 'the identity table')
        prep_defines = dict(list(programs[0][2].values())[0]['kernels'][0]['defines'])
        check_defines = dict(list(programs[4][2].values())[0]['kernels'][0]['defines'])
        self.assertIn('KVPW_ROLE_AUDIT_PREP', prep_defines)
        self.assertIn('KVPW_ROLE_AUDIT_CHECK', check_defines)

    def test_the_warm_forwards_ordered_writer_carries_no_audit_and_is_not_registered(self):
        tensors = block(self.ttnn)
        writer = poc.ChainedOrderedCacheWriter(self.mesh, self.ttnn, KERNELS, positions=tensors['positions'], pages=tensors['pages'],
                                               spans=((0, 64),), tiles=tiles(self.ttnn), launch_rows=64)
        self.assertTrue(writer.page.ordered)
        self.assertIsNone(writer.page.audit)
        self.assertEqual(kvpw._REGISTRY, [])
        for cache, packed in zip(tensors['caches'], tensors['packed']):
            writer(cache, packed, update_idxs_tensor=SimpleNamespace(shape=(64,)), page_table=SimpleNamespace(shape=(64, WIDTH)))
        self.assertEqual(len([call for call in self.ttnn.calls if call[0] == 'generic_op']), 1, 'the page launch alone')

    def test_prep_and_check_runtime_words(self):
        tensors = block(self.ttnn)
        writer = poc.ChainedOrderedCacheWriter(self.mesh, self.ttnn, KERNELS, positions=tensors['positions'], pages=tensors['pages'],
                                               spans=SEGMENTS, tiles=tiles(self.ttnn), launch_rows=64)
        audit = writer.page.audit
        for role, words in (('audit_prep', kvpw.AUDIT_PREP_WORDS), ('audit_check', kvpw.AUDIT_CHECK_WORDS)):
            extra = dict(shadow_positions=audit.positions) if role == 'audit_prep' else dict(counters=audit.counters)
            program = kvpw.build_audit_program(self.ttnn, self.mesh, role, dict(caches=tensors['caches'], shadows=audit.shadows,
                                               positions=tensors['positions'], pages=tensors['pages'], **extra), dict(audit_prep='P', audit_check='C'))
            runtime = {(x, y): args for x, y, args in program[((0, 0), (0, 0))].kernels[0].runtime_args.flattened()}
            self.assertEqual(len(runtime), 64)
            self.assertTrue(all(len(args) == words for args in runtime.values()))
        prep = kvpw.build_audit_program(self.ttnn, self.mesh, 'audit_prep', dict(caches=tensors['caches'], shadows=audit.shadows,
                                        positions=tensors['positions'], pages=tensors['pages'], shadow_positions=audit.positions), dict(audit_prep='P'))
        runtime = {(x, y): args for x, y, args in prep[((0, 0), (0, 0))].kernels[0].runtime_args.flattened()}
        writers = [core for core, args in runtime.items() if args[8]]
        self.assertEqual(len(writers), 4, 'one shadow-positions writer per group: cache K, slot 0, first column pair')
        self.assertEqual(sorted(args[9] for core, args in runtime.items() if args[8]), [0, 1, 2, 3])

    def test_audit_round_reads_the_counters(self):
        tensors = block(self.ttnn)
        writer = poc.ChainedOrderedCacheWriter(self.mesh, self.ttnn, KERNELS, positions=tensors['positions'], pages=tensors['pages'],
                                               spans=SEGMENTS, tiles=tiles(self.ttnn), launch_rows=64)
        counters = writer.page.audit.counters
        units = kvpw.unit_count(2)

        def records(mismatched=None):
            rows = [[0, 1 if unit % 2 == 0 else 0, 7, 0] + [0] * 12 for unit in range(units)]
            if mismatched is not None:
                rows[mismatched][0] = 5
            return SimpleNamespace(tolist=lambda: rows)

        with mock.patch.object(self.ttnn, 'to_torch', lambda shard: records()):
            self.assertEqual(kvpw.audit_round(self.ttnn, 0), units // 2)
        self.assertTrue(any(line.startswith(kvpw.AUDIT_MARKER) and 'exact=True' in line and 'writers=1' in line for line in self.lines), self.lines)
        with mock.patch.object(self.ttnn, 'to_torch', lambda shard: records(mismatched=4)):
            with self.assertRaises(AssertionError):
                kvpw.audit_round(self.ttnn, 1)
        self.assertTrue(any(line.startswith(kvpw.AUDIT_MISMATCH) and '5 words differ' in line for line in self.lines))
        with mock.patch.object(self.ttnn, 'to_torch', lambda shard: SimpleNamespace(tolist=lambda: [[0] * 16 for unit in range(units)])):
            with self.assertRaises(AssertionError):
                kvpw.audit_round(self.ttnn, 2)


# ---------------------------------------------------------------------------------------------
# The smoke rules.
# ---------------------------------------------------------------------------------------------

class SmokeTests(unittest.TestCase):
    ENGAGED = [kvpw.ENGAGED + ' cores=64 wt=2 source=l1 ordered=1 width=4096 audit=0 design=0123456789abcdef',
               kvpw.ENGAGED + ' cores=64 wt=2 source=l1 ordered=0 width=4096 audit=0 design=0123456789abcdef']

    def test_the_markers_are_the_modules(self):
        self.assertEqual((smoke.ENGAGED, smoke.FELL_BACK, smoke.AUDIT, smoke.MISMATCH), (kvpw.ENGAGED, kvpw.FELL_BACK, kvpw.AUDIT_MARKER, kvpw.AUDIT_MISMATCH))
        self.assertEqual((smoke.FLAG, smoke.AUDIT_FLAG), (kvpw.FLAG, kvpw.AUDIT_FLAG))
        self.assertTrue(all(marker.startswith('[PINDIAG] tp4 kv page writer ') for marker in (kvpw.ENGAGED, kvpw.FELL_BACK, kvpw.AUDIT_MARKER)))

    def test_rules(self):
        text = '\n'.join(self.ENGAGED)
        on = {smoke.FLAG: '1'}
        self.assertEqual(smoke.problems(on, text), [])
        self.assertEqual(smoke.problems(None, 'nothing'), [])
        self.assertTrue(smoke.problems(on, 'nothing'))
        self.assertTrue(smoke.problems(on, self.ENGAGED[0]), 'the warm forward alone is not the captured forward')
        self.assertTrue(smoke.problems(on, text.replace('cores=64', 'cores=0')))
        self.assertTrue(smoke.problems(on, text + '\n' + kvpw.FELL_BACK + ' reason=x'))
        self.assertTrue(smoke.problems({}, text))
        both = dict(on, **{smoke.AUDIT_FLAG: '1'})
        self.assertTrue(smoke.problems(both, text))
        passing = text + '\n' + kvpw.AUDIT_MARKER + ' 3 exact=True writers=16 units=512'
        self.assertEqual(smoke.problems(both, passing), [])
        self.assertTrue(smoke.problems(both, passing + '\n' + kvpw.AUDIT_MISMATCH + ' round=1 x'))


if __name__ == '__main__':
    unittest.main()
