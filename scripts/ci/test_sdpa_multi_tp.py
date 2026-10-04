"""QWEN_FAST_TP4_SDPA=multi (sdpa_multi_tp): ONE SDPA launch per layer for every user of a packed block, byte for byte the per-user launches.

What is tested, all on CPU:
  - the shape census: the factory's core allocation as arithmetic (16 cores per entry for 1..6 users at the served 110-core grid,
    15 and 13 at 7 and 8, refused), the factory lines the argument quotes (read out of the fixture), every refusal;
  - the launches' index maps: the fold-in, the fold-out, the mask and the page-table gather, EMULATED from the runtime arguments the
    planners write and compared with the served composition (two 8-token groups per user, the pinned narrow masks, the lent tables);
  - the reader twin: flag off is the served path (nothing allocated, nothing logged, the pinned context manager), flag on makes ONE
    launch per layer with the multi program, one refresh per forward booked as model_batch counts it, the call budget, the failure
    path, the refusals at construction, and the audit (both paths run, the compare counters decide);
  - the smoke rule, the two gate-only profiles, the shipment (three copy lists, the CPU allowlist) and the kernels' text.

Run at py 3.11: `py -3.11 -B -m unittest test_sdpa_multi_tp` from scripts/ci.
"""

from contextlib import ExitStack
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import attention_block_fold_tp as fold
import c2_smoke_check
import extent_attention_fold_tp as folded
import extent_attention_replay as pinned
import extent_attention_replay_tp as quad
import pooled_attention_replay
import sdpa_long_tp
import sdpa_multi_tp as multi
from test_extent_attention_replay import WIDTH, host_table
from test_extent_attention_replay_tp import FourChipDevice, Harness, MESH, four, lend
from test_tp4_vglue_attention import COLUMNS, HEAD, from_pages, kernel_task, to_pages

import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FLAG, AUDIT = sdpa_long_tp.FLAG, multi.AUDIT_FLAG
SEGMENTS = ((0, 16), (16, 32), (32, 48), (48, 64))
GRID = (11, 10)
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
CONTROL, TWIN, AUDITED = ('c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-best-sdpamulti',
                          'c2-packed-tp4-8x262k-best-sdpamulti-audit')


def env(**values):
    return patch.dict(os.environ, values)


# --- the shape census -----------------------------------------------------------------------------------------------------

class CensusTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_factory_gives_sixteen_cores_per_entry_for_one_to_six_users_and_fewer_beyond(self):
        found = {users: multi.factory_cores(users, 110)['cores_per_entry'] for users in range(1, 9)}
        self.assertEqual(found, {1: 16, 2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 15, 8: 13})

    def test_the_census_names_the_refusals_and_only_them(self):
        table = multi.census(GRID)
        self.assertEqual([users for users, value in table.items() if value == 16], [1, 2, 3, 4, 5, 6])
        for users in (7, 8):
            self.assertIn('the factory gives %d cores per entry' % (15 if users == 7 else 13), table[users])
            self.assertIn('B=%d' % users, table[users])

    def test_active_cores_rounds_and_scratch_slots_of_the_served_shapes(self):
        for users in range(1, 7):
            found = multi.factory_cores(users, 110)
            self.assertEqual((found['active_cores'], found['rounds'], found['scratch_slots']), (16 * users, 4, 4))
        # 16 cores: ceil(log2(16)) = 4 rounds, the compact scratch's four slots: the factory line's scratch_slots=4
        self.assertEqual(multi.factory_cores(4, 110, scratch_rounds_env=False)['scratch_slots'], 15)

    def test_the_factory_arithmetic_for_other_grids(self):
        # the served program on an 8x10 grid (80 cores): five entries still get 16, six get 13
        self.assertEqual(multi.factory_cores(5, 80)['cores_per_entry'], 16)
        self.assertEqual(multi.factory_cores(6, 80)['cores_per_entry'], 13)
        self.assertIsNone(multi.plan_problem(5, (8, 10)))
        self.assertIn('13 cores per entry', multi.plan_problem(6, (8, 10)))
        self.assertIn('8 cores per entry', multi.plan_problem(2, (4, 4)))
        self.assertIsNone(multi.plan_problem(1, (4, 4)))

    def test_two_kv_heads_and_other_caps_are_the_factorys_own(self):
        # the model of the allocation holds beyond the served shape: it is the factory's integer arithmetic
        self.assertEqual(multi.factory_cores(2, 110, max_cores=55)['cores_per_entry'], 55)
        two = multi.factory_cores(2, 110, max_cores=16, kv_heads=2)
        self.assertEqual((two['cores_per_head'], two['cores_per_entry']), (16, 32))
        self.assertEqual(multi.factory_cores(1, 4)['cores_per_entry'], 4)
        self.assertEqual(multi.factory_cores(1, 1)['rounds'], 0)

    def test_the_plan_for_four_users(self):
        found = multi.plan(4, GRID)
        self.assertEqual(tuple(found), (4, 16, 64, 4, 4, 3, 0x21))
        self.assertEqual((found.flags, multi.entry_rows(), multi.entry_tiles()), (0x21, 96, 3))

    def test_every_refusal(self):
        for users in (0, -1, 9, True, 1.0, None):
            with self.subTest(users=users):
                self.assertIsNotNone(multi.plan_problem(users, GRID))
        for users in (7, 8):
            with self.assertRaisesRegex(ValueError, 'QWEN_FAST_TP4_SDPA=multi: the factory gives'):
                multi.plan(users, GRID)
        self.assertIn('runtime-argument slots', multi.plan_problem(9, GRID))
        self.assertIn('cores for', multi.plan_problem(3, (1, 2)))

    def test_the_block_shapes_the_launch_is_written_for(self):
        problem = multi.shape_problem
        self.assertIsNone(problem(SEGMENTS, [16] * 4, [1] * 4, GRID))
        self.assertIsNone(problem(SEGMENTS[:1], [16], [1], GRID))
        self.assertIn('T16 user', problem(((0, 8), (8, 24)), [8, 16], [1, 1], GRID))
        self.assertIn('T16 user', problem(((0, 32),), [32], [2], GRID))
        self.assertIn('one bundle', problem(SEGMENTS[:1], [16], [2], GRID))
        self.assertIn('tile the block', problem(((16, 32), (0, 16)), [16, 16], [1, 1], GRID))
        self.assertIn('tile the block', problem(((0, 16), (32, 48)), [16, 16], [1, 1], GRID))
        self.assertIn('no segments', problem((), [], [], GRID))
        seven = tuple((16 * user, 16 * user + 16) for user in range(7))
        self.assertIn('15 cores per entry', problem(seven, [16] * 7, [1] * 7, GRID))
        self.assertIn('one row count', problem(SEGMENTS, [16] * 3, [1] * 4, GRID))

    def test_the_factory_lines_the_argument_quotes_are_in_the_fixtures(self):
        self.assertEqual(multi.citation_problems(ROOT / 'optimisation' / 'ttnn-op'), [])
        self.assertGreaterEqual(len(multi.FACTORY_CITATIONS), 11)

    def test_a_drifted_citation_is_reported(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            files = {}
            for name, line, text in multi.FACTORY_CITATIONS:
                lines = files.setdefault(name, ['x'] * 1000)
                lines[line - 1] = text
            for name, lines in files.items():
                (Path(directory) / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
            self.assertEqual(multi.citation_problems(directory), [])
            (Path(directory) / multi.FACTORY_CITATIONS[1][0]).write_text('nothing here\n', encoding='utf-8')
            self.assertTrue(multi.citation_problems(directory))
        self.assertTrue(multi.citation_problems(HERE / 'references'))

    def test_the_exactness_lines_quote_the_factory(self):
        text = ' '.join(multi.exactness())
        for word in ('16 * B', '198-200', 'get_workload_for_core', 'rt_args_common', ':240', ':398', ':785', 'refused'):
            self.assertIn(word, text)

    def test_the_constants_match_the_modules_that_own_them(self):
        self.assertEqual(multi.QWEN_DECODE_MAGIC, pooled_attention_replay.QWEN_DECODE_MAGIC)
        self.assertEqual(multi.FLAGS, pooled_attention_replay.QWEN_MASK_TAIL | pooled_attention_replay.QWEN_RUNTIME_EXTENT)
        self.assertEqual(multi.ENGAGED, sdpa_long_tp.ENGAGED)
        self.assertEqual(multi.ENTRY_CORES, sdpa_long_tp.KV_CORES_PER_ENTRY)
        self.assertEqual(multi.CHUNK, pinned.K)
        self.assertEqual(multi.ROWS, 16)


# --- the launches' index maps, emulated -------------------------------------------------------------------------------------

def row_local_sdpa(user, stacked):
    """A stand-in for the SDPA whose output row depends on its input row and its USER only (never on the entry layout), as the real
    op's does: rows never interact."""
    return stacked * 1.5 + user * 1000.0


def served_block(query, segments):
    """The served composition on logical tensors: per user two 8-token groups stacked on dim 1 (token-major rows), the SDPA, the groups
    back, then the users joined."""
    users = []
    for user, (first, last) in enumerate(segments):
        chunk = query[:, first:last]
        groups = [chunk[:, offset:offset + 8].reshape(1, 1, 8 * HEAD, 256) for offset in (0, 8)]
        result = row_local_sdpa(user, torch.cat(groups, dim=1))
        users.append(torch.cat([result[:, index:index + 1].reshape(1, 8, HEAD, 256) for index in range(2)], dim=1))
    return torch.cat(users, dim=1)


def run_plan(plan, sources, destinations, buffers, inverse, cores=110):
    flat = fold.flat_tasks(plan, sources, destinations)
    for arguments in fold.runtime_arguments(inverse, fold.distribute(flat, cores)):
        count = arguments[1]
        for index in range(count):
            source, destination, rows, source_base, destination_base, task = arguments[2 + index * 6:8 + index * 6]
            page, tile = kernel_task(bool(inverse), rows, buffers[source].__getitem__, source_base, destination_base, task)
            buffers[destination][page] = tile


class FoldMapTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_fold_in_then_the_stub_then_fold_out_is_the_served_composition_byte_for_byte(self):
        for segments in (SEGMENTS, SEGMENTS[:1], SEGMENTS[:2], SEGMENTS[:3]):
            users = len(segments)
            total = segments[-1][1]
            query = torch.arange(total * HEAD * 256, dtype=torch.float32).reshape(1, total, HEAD, 256) + 0.5
            expected = served_block(query, segments)
            buffers = {1: to_pages(query), 2: {}}
            run_plan(multi.forward_plan(segments), [1] * users, [2] * users, buffers, False)
            stacked, clean = from_pages(buffers[2], users, 96)
            self.assertTrue(clean, 'the stacked padding is zero')
            for user, (first, last) in enumerate(segments):
                self.assertTrue(torch.equal(stacked[0, user], query[0, first:last].reshape(96, 256)), 'user %d token-major' % user)
            result = torch.stack([row_local_sdpa(user, stacked[:, user]) for user in range(users)], dim=1)
            buffers[3], buffers[4] = to_pages(result), {}
            run_plan(multi.inverse_plan(segments), [3] * users, [4] * users, buffers, True)
            built, clean = from_pages(buffers[4], total, HEAD)
            self.assertTrue(clean)
            self.assertTrue(torch.equal(built, expected), segments)
            self.assertEqual(sorted(buffers[4]), list(range(total * COLUMNS)), 'every output page is written once')
            self.assertEqual(sorted(buffers[2]), list(range(users * 3 * COLUMNS)))

    def test_task_counts_and_core_distribution(self):
        self.assertEqual([len(part) for part in multi.forward_plan(SEGMENTS)], [24] * 4)
        self.assertEqual([len(part) for part in multi.inverse_plan(SEGMENTS)], [128] * 4)
        self.assertEqual(multi.mask_tasks(4), list(range(96)))
        self.assertEqual(len(multi.mask_tasks(8)), 192)
        per_core = multi.distribute(multi.mask_tasks(8), 110)
        self.assertEqual(len(per_core), 110)
        self.assertEqual(sorted(task for core in per_core for task in core), list(range(192)))
        self.assertEqual(len(multi.distribute([1, 2], 110)), 2)
        with self.assertRaises(ValueError):
            multi.distribute([], 110)

    def test_the_block_fold_kernel_is_unchanged_and_serves_sixteen_rows(self):
        text = (HERE / 'attention_block_fold_tp.cpp').read_text(encoding='utf-8')
        self.assertIn('(rows > 4 ? rows : 4) * 2048', text)           # the staging area grows with the rows: 16 tiles in, then the output
        self.assertIn('source_tiles = inverse ? (rows * QWEN_FOLD_HEAD_ROWS + 31) / 32 : rows', text)


def emulate_mask(arguments, positions_words, capacity=256):
    """sdpa_multi_mask_tp.cpp's loops over the runtime args the planner writes: {page: 32x32 int16 tile in (row, column) order}."""
    pages = {}
    for core in arguments:
        rows, given = core[1], core[2]
        head_tiles = (rows * HEAD + 31) // 32
        for index in range(given):
            task = core[11 + index]
            batch, head_tile, column_tile = task // (head_tiles * 8), (task // 8) % head_tiles, task % 8
            start = positions_words[core[3 + batch]]
            tile = torch.zeros(32, 32, dtype=torch.int16)
            for row in range(32):
                head = head_tile * 32 + row
                position = start + head // 6
                for column in range(32):
                    masked = head >= rows * HEAD or column_tile * 32 + column > position
                    tile[row, column] = -128 if masked else 0
            pages[task] = tile
    return pages


def dense_mask(pages, users):
    out = torch.zeros(users, 96, 256, dtype=torch.int16)
    for task, tile in pages.items():
        batch, head_tile, column_tile = task // 24, (task // 8) % 3, task % 8
        out[batch, head_tile * 32:(head_tile + 1) * 32, column_tile * 32:(column_tile + 1) * 32] = tile
    return out


class MaskTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_the_host_mask_is_the_token_major_causal_mask_of_each_user(self):
        words = [0, 7, 120, 240]
        mask = multi.mask_host(words)
        self.assertEqual(tuple(mask.shape), (4, 1, 96, 256))
        for user, word in enumerate(words):
            for token in range(16):
                for head in range(6):
                    row = mask[user, 0, token * 6 + head]
                    self.assertTrue(bool((row[:word + token + 1] == 0).all()) and bool((row[word + token + 1:] == -128).all()),
                                    (user, token))

    def test_every_row_is_the_pinned_narrow_masks_row_of_the_same_token(self):
        # the served entry b of a user's bundle masks tokens b * 8 .. b * 8 + 7 with word + b * 8 + token-in-group: the same
        # position the one 16-token entry gives that token. Compared bit for bit with the pinned kernel's host transliteration.
        for word in (0, 1, 7, 127, 128, 200, 239, 240):
            served = quad.narrow_mask_host(word, 8, 2, 0).view(torch.int16)
            built = multi.mask_host([word])
            for token in range(16):
                entry, inside = divmod(token, 8)
                for head in range(6):
                    self.assertTrue(torch.equal(built[0, 0, token * 6 + head], served[entry, 0, inside * 6 + head]), (word, token, head))

    def test_the_kernel_emulation_writes_the_host_mask_on_every_page(self):
        words = [3, 130, 241 - 1, 17]
        positions = {0x100 + 0x40 * user: word for user, word in enumerate(words)}
        for cores in (110, 8, 96):
            tasks = multi.mask_tasks(4)
            per_core = multi.distribute(tasks, cores)
            capacity = max(len(core) for core in per_core)
            arguments = multi.mask_arguments(0x900, sorted(positions), per_core, capacity)
            self.assertTrue(all(len(core) == 11 + capacity for core in arguments))
            pages = emulate_mask(arguments, positions)
            self.assertEqual(sorted(pages), tasks)
            self.assertTrue(torch.equal(dense_mask(pages, 4), multi.mask_host(words)[:, 0]), cores)

    def test_the_kernels_tile_body_is_the_pinned_siblings(self):
        left = (HERE / 'attention_mask_replay_tp.cpp').read_text(encoding='utf-8')
        right = (HERE / multi.KERNEL_MASK).read_text(encoding='utf-8')
        for fragment in ('(row / 16) * 512 + (column / 16) * 256 + (row % 16) * 16 + column % 16', '0xff80 : 0',
                         'noc_async_read(get_noc_addr(0, positions), staged, 32)', 'staged + 2048', 'noc_async_write_tile('):
            self.assertIn(fragment, left)
            self.assertIn(fragment, right)
        self.assertIn('QWEN_FOLD_HEAD_ROWS', right)
        self.assertIn('const uint32_t position = start + head / 6;', right)
        self.assertNotIn('batch * rows', right, 'one entry per user: the served group offset is gone')

    def test_arguments_refuse_more_than_eight_users_and_pad_short_lists(self):
        with self.assertRaises(ValueError):
            multi.mask_arguments(1, list(range(9)), [[0]], 1)
        arguments = multi.mask_arguments(0x900, [0x10, 0x20], [[0, 5], [1]], 2)
        self.assertEqual(arguments[0], [0x900, 16, 2, 0x10, 0x20, 0, 0, 0, 0, 0, 0, 0, 5])
        self.assertEqual(arguments[1], [0x900, 16, 1, 0x10, 0x20, 0, 0, 0, 0, 0, 0, 1, 0])


def emulate_gather(arguments, tables, positions, table_page_bytes=64, position_in_bytes=32, position_out_bytes=32):
    """sdpa_multi_gather_tp.cpp: tables {address: {page: list of ints}}, positions {address: {page: list of ints}}."""
    for core in arguments:
        task, users, table_out, position_out = core[:4]
        if task < users:
            tables[table_out][task] = list(tables[core[4 + 2 * task]][0])
        else:
            page = [0] * (position_out_bytes // 4)
            for user in range(users):
                page[user] = positions[core[5 + 2 * user]][0][0]
            positions[position_out][0] = page


class GatherTests(unittest.TestCase):
    def test_the_gather_copies_every_users_row_and_assembles_cur_pos(self):
        users = 4
        table_addresses = [0x1000 + 0x100 * user for user in range(users)]
        position_addresses = [0x2000 + 0x100 * user for user in range(users)]
        tables = {address: {0: [user * 1000 + page for page in range(8)], 1: [user * 1000 + page for page in range(8)]}
                  for user, address in enumerate(table_addresses)}
        tables[0x3000] = {}
        positions = {address: {0: [4000 + 256 * user + 255, 4000 + 256 * user + 255]} for user, address in enumerate(position_addresses)}
        positions[0x4000] = {}
        arguments = [multi.gather_arguments(task, users, 0x3000, 0x4000, table_addresses, position_addresses) for task in range(users + 1)]
        self.assertTrue(all(len(core) == 20 for core in arguments))
        emulate_gather(arguments, tables, positions)
        self.assertEqual([tables[0x3000][user] for user in range(users)],
                         [[user * 1000 + page for page in range(8)] for user in range(users)])
        self.assertEqual(positions[0x4000][0][:4], [4255 + 256 * user for user in range(users)])
        self.assertEqual(positions[0x4000][0][4:], [0] * 4)

    def test_arguments_refuse_a_mismatched_or_oversized_set(self):
        with self.assertRaises(ValueError):
            multi.gather_arguments(0, 2, 1, 2, [1], [2, 3])
        with self.assertRaises(ValueError):
            multi.gather_arguments(0, 9, 1, 2, list(range(9)), list(range(9)))

    def test_the_kernel_reads_the_argument_layout_the_planner_writes(self):
        text = (HERE / multi.KERNEL_GATHER).read_text(encoding='utf-8')
        for fragment in ('get_arg_val<uint32_t>(4 + 2 * task)', 'get_arg_val<uint32_t>(2)', 'get_arg_val<uint32_t>(3)',
                         'get_arg_val<uint32_t>(5 + 2 * user)', 'noc_async_write(staged, get_noc_addr(task, destination), TABLE_BYTES)'):
            self.assertIn(fragment, text)


class AuditMapTests(unittest.TestCase):
    def test_the_ranges_cover_the_tiles_exactly_once(self):
        for tiles in (512, 8, 100, 1):
            ranges = multi.audit_ranges(tiles)
            self.assertEqual(len(ranges), multi.AUDIT_CORES)
            covered = [tile for first, count in ranges for tile in range(first, first + count)]
            self.assertEqual(covered, list(range(tiles)))
        self.assertEqual(multi.audit_ranges(512)[0], (0, 8))
        self.assertEqual(multi.audit_ranges(512)[63], (504, 8))

    def counters(self, layers, differing=None, live=8, compared=8 * 512):
        out = torch.zeros(multi.SLOTS * multi.AUDIT_CORES, multi.COUNTER_WORDS, dtype=torch.int32)
        for slot in range(layers):
            for core in range(multi.AUDIT_CORES):
                page = out[slot * multi.AUDIT_CORES + core]
                page[1], page[2] = live, compared
                if differing and (slot, core) in differing:
                    page[0] = differing[(slot, core)]
        return out

    def test_a_clean_run_is_exact(self):
        differing, live, compared, reasons = multi.audit_verdict(self.counters(16), 16)
        self.assertEqual((differing, reasons), (0, []))
        self.assertEqual(compared, 16 * 64 * 8 * 512)

    def test_one_differing_word_anywhere_fails_and_names_the_layer(self):
        differing, live, compared, reasons = multi.audit_verdict(self.counters(16, {(5, 17): 1}), 16)
        self.assertEqual(differing, 1)
        self.assertEqual(reasons, ['layer 5: 1 of %d words differ' % (64 * 8 * 512)])

    def test_two_all_zero_outputs_prove_nothing(self):
        _, _, _, reasons = multi.audit_verdict(self.counters(16, live=0), 16)
        self.assertEqual(len(reasons), 16)
        self.assertIn('nothing was compared', reasons[0])

    def test_no_layer_run_fails(self):
        self.assertEqual(multi.audit_verdict(self.counters(0), 0)[3], ['no attention layer ran the multi launch'])

    def test_a_layer_that_never_ran_is_not_required(self):
        self.assertEqual(multi.audit_verdict(self.counters(2), 2)[3], [])

    def test_the_kernel_counts_what_the_verdict_reads(self):
        text = (HERE / multi.KERNEL_AUDIT).read_text(encoding='utf-8')
        for fragment in ('out[0] = differing;', 'out[1] = live;', 'out[2] = compared;', 'if (x != b[word]) { differing++; }',
                         'if (x != 0) { live++; }', 'noc_async_read_tile(tile, left, staged);',
                         'noc_async_read_tile(tile, right, staged + 2048);'):
            self.assertIn(fragment, text)
        self.assertIn('uint32_t', text)
        self.assertNotIn('float', text, 'a bit compare, not a numeric one: -0 and +0 differ')


# --- the launch builders on a fake ttnn -------------------------------------------------------------------------------------

class FakeBuffer(object):
    def __init__(self, address, layout, page):
        self.address, self.layout, self.page = address, layout, page

    def buffer_address(self):
        return self.address

    def buffer_aligned_page_size(self):
        return self.page


class FakeBuilt(object):
    def __init__(self, shape, base, page=64, layout='rm', chips=4):
        self.shape, self.base, self.page, self.layout, self.chips = tuple(shape), base, page, layout, chips
        self.dtype, self.memory = 'bf16', 'dram'

    def memory_config(self):
        return self.memory


def builder_ttnn(record):
    def generic_op(tensors, program):
        record.append((list(tensors), program))

    def shards(tensor):
        return [FakeBuffer(tensor.base + chip * 0x10000000, tensor.layout, tensor.page) for chip in range(tensor.chips)]

    return SimpleNamespace(
        bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram', get_device_tensors=shards, generic_op=generic_op,
        CoreCoord=lambda x, y: (x, y), CoreRange=lambda a, b: (a, b), CoreRangeSet=lambda ranges: list(ranges),
        CBDescriptor=SimpleNamespace, CBFormatDescriptor=SimpleNamespace, TileDescriptor=lambda value: value, Tile=list,
        MeshProgramDescriptor=dict, ProgramDescriptor=SimpleNamespace, MeshCoordinate=lambda *v: v,
        MeshCoordinateRange=lambda *v: v, KernelDescriptor=SimpleNamespace,
        DataMovementConfigDescriptor=SimpleNamespace, DataMovementProcessor=SimpleNamespace(RISCV_0='r0'),
        NOC=SimpleNamespace(RISCV_0_default=0),
        TensorAccessorArgs=lambda shard: SimpleNamespace(get_compile_time_args=lambda: ['accessor', shard.layout]),
        RuntimeArgs=lambda: __import__('collections').defaultdict(dict))


def kernels_of(program):
    return {key: description.kernels[0] for key, description in program.items()}


def core_lists(kernel):
    return [arguments for column in kernel.runtime_args.values() for arguments in column.values()]


class BuilderTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ttnn = builder_ttnn([])

    def test_the_mask_program(self):
        positions = [FakeBuilt((8,), 0x1000 + 0x100 * user, 32) for user in range(4)]
        mask = FakeBuilt((4, 1, 96, 256), 0x5000, 2048, 'tile')
        with patch.dict(sys.modules, {'ttnn': self.ttnn}):
            program, io = multi.build_mask_program(self.ttnn, MESH, positions, mask, 4)
        self.assertEqual(sorted(program), [((0, chip), (0, chip)) for chip in range(4)])
        self.assertEqual([tensor for tensor in io], [*positions, mask])
        for chip, kernel in enumerate(kernels_of(program).values()):
            self.assertEqual(kernel.kernel_source, str(HERE / multi.KERNEL_MASK))
            self.assertEqual(dict(kernel.defines), {'QWEN_FOLD_HEAD_ROWS': '6'})
            self.assertEqual(kernel.compile_time_args[-1], 1, 'one task per core: 96 tasks on 96 of 110 cores')
            arguments = core_lists(kernel)
            self.assertEqual(len(arguments), 96)
            self.assertEqual(sorted(task for core in arguments for task in core[11:11 + core[2]]), list(range(96)))
            for core in arguments:
                self.assertEqual(core[0], 0x5000 + chip * 0x10000000)
                self.assertEqual(core[3:7], [0x1000 + 0x100 * user + chip * 0x10000000 for user in range(4)])

    def test_the_mask_program_with_more_tasks_than_cores_carries_a_capacity(self):
        positions = [FakeBuilt((8,), 0x1000 + 0x100 * user, 32) for user in range(6)]
        mask = FakeBuilt((6, 1, 96, 256), 0x5000, 2048, 'tile')
        with patch.dict(sys.modules, {'ttnn': self.ttnn}):
            program, io = multi.build_mask_program(self.ttnn, MESH, positions, mask, 6)
        kernel = list(kernels_of(program).values())[0]
        self.assertEqual(kernel.compile_time_args[-1], 2)
        self.assertEqual(sorted(task for core in core_lists(kernel) for task in core[11:11 + core[2]]), list(range(144)))

    def test_the_mask_program_refuses_aliasing_and_mixed_layouts(self):
        positions = [FakeBuilt((8,), 0x1000 + 0x100 * user, 32) for user in range(2)]
        with patch.dict(sys.modules, {'ttnn': self.ttnn}):
            with self.assertRaisesRegex(ValueError, 'must not alias'):
                multi.build_mask_program(self.ttnn, MESH, positions, FakeBuilt((2, 1, 96, 256), 0x1000, 2048, 'tile'), 2)
            positions[1].layout = 'other'
            with self.assertRaisesRegex(ValueError, 'accessor layout'):
                multi.build_mask_program(self.ttnn, MESH, positions, FakeBuilt((2, 1, 96, 256), 0x5000, 2048, 'tile'), 2)

    def test_the_gather_program(self):
        users = 4
        lent_tables = [FakeBuilt((2, WIDTH), 0x1000 + 0x100 * user, 8256) for user in range(users)]
        lent_positions = [FakeBuilt((2,), 0x2000 + 0x100 * user, 64) for user in range(users)]
        table, position = FakeBuilt((users, WIDTH), 0x3000, 8256), FakeBuilt((users,), 0x4000, 64)
        with patch.dict(sys.modules, {'ttnn': self.ttnn}):
            program, io = multi.build_gather_program(self.ttnn, MESH, lent_tables, lent_positions, table, position)
        self.assertEqual(io, [*lent_tables, *lent_positions, table, position])
        for chip, kernel in enumerate(kernels_of(program).values()):
            self.assertEqual(kernel.kernel_source, str(HERE / multi.KERNEL_GATHER))
            self.assertEqual(kernel.compile_time_args[-3:], [8256, 64, 64])
            arguments = core_lists(kernel)
            self.assertEqual(sorted(core[0] for core in arguments), list(range(users + 1)))
            for core in arguments:
                self.assertEqual(core[1:4], [users, 0x3000 + chip * 0x10000000, 0x4000 + chip * 0x10000000])
                self.assertEqual(core[4:12], [value for user in range(users) for value in (
                    0x1000 + 0x100 * user + chip * 0x10000000, 0x2000 + 0x100 * user + chip * 0x10000000)])
        for description in program.values():
            self.assertGreaterEqual(description.cbs[0].total_size, 8256 + 256)
            self.assertEqual(description.cbs[0].total_size % 2048, 0)

    def test_the_gather_refuses_a_table_page_that_is_not_the_stacked_ones(self):
        lent_tables = [FakeBuilt((2, WIDTH), 0x1000, 8256), FakeBuilt((2, WIDTH), 0x1100, 64)]
        lent_positions = [FakeBuilt((2,), 0x2000, 64), FakeBuilt((2,), 0x2100, 64)]
        with patch.dict(sys.modules, {'ttnn': self.ttnn}):
            with self.assertRaisesRegex(ValueError, 'page size'):
                multi.build_gather_program(self.ttnn, MESH, lent_tables, lent_positions, FakeBuilt((2, WIDTH), 0x3000, 8256),
                                           FakeBuilt((2,), 0x4000, 64))

    def test_a_runtime_that_cannot_report_page_sizes_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, 'aligned page size'):
            multi.aligned_page_bytes(SimpleNamespace(buffer_address=lambda: 1))

    def test_the_audit_program(self):
        served, built = FakeBuilt((1, 64, 6, 256), 0x6000, 2048, 'tile'), FakeBuilt((1, 64, 6, 256), 0x7000, 2048, 'tile')
        counters = FakeBuilt((16 * 64, 16), 0x8000, 64)
        with patch.dict(sys.modules, {'ttnn': self.ttnn}):
            program, io = multi.build_audit_program(self.ttnn, MESH, served, built, counters, 5)
        self.assertEqual(io, [served, built, counters])
        for chip, kernel in enumerate(kernels_of(program).values()):
            self.assertEqual(kernel.compile_time_args[-1], 64)
            # One accessor layout per accessor the kernel declares (served, multi, counters), then the page bytes (v444: a single
            # shared layout for served + multi shifted the counter args past the end, 'Index out of range').
            accessors = (HERE / multi.KERNEL_AUDIT).read_text(encoding='utf-8').count('TensorAccessorArgs<')
            self.assertEqual(accessors, 3)
            self.assertEqual(kernel.compile_time_args[:-1].count('accessor'), accessors)
            arguments = core_lists(kernel)
            self.assertEqual(len(arguments), 64)
            self.assertEqual(sorted(core[3] for core in arguments), list(range(5 * 64, 6 * 64)))
            self.assertEqual(sum(core[5] for core in arguments), 512)
            for core in arguments:
                self.assertEqual(core[:3], [0x6000 + chip * 0x10000000, 0x7000 + chip * 0x10000000, 0x8000 + chip * 0x10000000])

    def test_every_launch_covers_four_chips(self):
        positions = [FakeBuilt((8,), 0x1000, 32, chips=2)]
        with patch.dict(sys.modules, {'ttnn': self.ttnn}):
            with self.assertRaisesRegex(ValueError, 'chip-local buffers required'):
                multi.build_mask_program(self.ttnn, MESH, positions, FakeBuilt((1, 1, 96, 256), 0x5000, 2048, 'tile', chips=2), 1)


# --- the reader twin ---------------------------------------------------------------------------------------------------------

class MultiDevice(FourChipDevice):
    """FourChipDevice with what the multi path calls: empty, generic_op, to_torch, and the page reports."""

    bfloat16 = 'bf16'

    def __init__(self):
        FourChipDevice.__init__(self)
        self.generic = []
        self.counters_value = None

    def empty(self, shape, dtype=None, layout=None, device=None, memory_config=None):
        return self.device_tensor(shape, dtype, layout, memory_config, name='empty')

    def generic_op(self, tensors, program):
        self.generic.append((list(tensors), program))
        self.events.append(('generic', program))

    def to_torch(self, shard):
        return self.counters_value


def block(device, segments=SEGMENTS, starts=(4200, 9000, 4300, 5000)):
    lent = [lend(device, last - first) for first, last in segments]
    return folded.PackedExtentReplayReader(device, MESH, segments, WIDTH, [host_table(i + 1) for i in range(len(segments))],
                                           storage=lent, max_group_rows=8, starts=starts[:len(segments)])


class Rig(object):
    """The patches a multi reader needs on the fake device: the three builders (named programs), the fold launches and the pinned
    refresh, with the lists a test reads back."""

    def __init__(self, test, device):
        self.device, self.logs, self.refreshed, self.folds, self.audits = device, [], [], [], []
        self.stack = ExitStack()
        test.addCleanup(self.stack.close)
        self.stack.enter_context(patch('sdpa_multi_tp._log', side_effect=self.logs.append))
        self.stack.enter_context(patch('sdpa_long_tp._log', side_effect=self.logs.append))
        self.stack.enter_context(patch('sdpa_multi_tp.build_mask_program', side_effect=lambda ttnn, mesh, positions, mask, users: (
            ('mask-program', users), [*positions, mask])))
        self.stack.enter_context(patch('sdpa_multi_tp.build_gather_program', side_effect=lambda ttnn, mesh, tables, cur, table, position: (
            ('gather-program', len(tables)), [*tables, *cur, table, position])))
        self.stack.enter_context(patch('sdpa_multi_tp.build_audit_program', side_effect=self.audit_program))
        self.stack.enter_context(patch('attention_block_fold_tp._launch', side_effect=self.launch))
        self.stack.enter_context(patch('extent_attention_replay_tp.attention_mask_replay.execute',
                                       side_effect=lambda *a: self.refreshed.append(a)))
        self.stack.enter_context(patch('extent_attention_replay_tp.device_layout_dma', side_effect=self.served_fold))
        self.stack.enter_context(patch('attention_block_fold_tp.fold_in', side_effect=self.block_fold_in))
        self.stack.enter_context(patch('attention_block_fold_tp.fold_out', side_effect=self.block_fold_out))

    def audit_program(self, ttnn, mesh, served, built, counters, slot):
        self.audits.append((served, built, counters, slot))
        return ('audit-program', slot), [served, built, counters]

    def launch(self, mesh, sources, destinations, plan, *, inverse, tag):
        self.folds.append(dict(sources=sources, destinations=destinations, plan=plan, inverse=inverse, tag=tag))
        self.device.events.append(('fold', tag))

    def served_fold(self, mesh, source, rows, owned, *, inverse=False, offset=0):
        output = self.device.device_tensor((1, rows, 6, 256) if inverse else (1, 1, rows * 6, 256), 'bf16', 'tile', 'dram', name='fold')
        owned.append(output)
        return output

    def block_fold_in(self, mesh, query, chunks, owned):
        stacked = [self.device.device_tensor((1, chunk.batches, chunk.rows * 6, 256), 'bf16', 'tile', 'dram', name='stack')
                   for chunk in chunks]
        owned.extend(stacked)
        return stacked

    def block_fold_out(self, mesh, results, chunks, memory_config, owned):
        total = chunks[-1].first + chunks[-1].groups[-1][0] + chunks[-1].groups[-1][1]
        output = self.device.device_tensor((1, total, 6, 256), 'bf16', 'tile', memory_config, name='block-out')
        owned.append(output)
        return output


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.device = MultiDevice()
        self.harness = Harness(self)
        self.rig = Rig(self, self.device)

    def build(self, **flags):
        with env(**flags):
            return block(self.device)

    def query(self):
        return (self.device.device_tensor((1, 64, 6, 256), 'bf16', 'tile', 'dram', name='query'),
                self.device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys'))

    def call(self, reader, **kwargs):
        query, keys = self.query()
        return reader(query, keys, keys, scale=0.0625, memory_config='dram', **kwargs)

    # -- flag off -----------------------------------------------------------------------------------------------------

    def test_flag_off_allocates_nothing_logs_nothing_and_is_the_served_path(self):
        before = len(self.device.live)
        reader = self.build()
        served_only = len(self.device.live) - before
        self.assertIsNone(reader.multi)
        self.assertIsNone(folded.PackedExtentReplayReader.multi)
        self.assertEqual([line for line in self.rig.logs if multi.ENGAGED in line or multi.CALL_MARKER in line], [])
        with patch.object(folded.PackedExtentReplayReader, 'served_call', return_value='served') as served:
            self.assertEqual(self.call(reader), 'served')
        self.assertEqual(served.call_count, 1)
        self.assertEqual(reader.sdpa_audit_round(1), 0)
        self.assertEqual(self.device.generic, [])
        # the shared-mask scope is the pinned reader's own context manager
        scope = reader.shared_masks(16)
        self.assertEqual(type(scope).__qualname__, quad.PackedExtentReplayReader.shared_masks(reader, 16).__class__.__qualname__)
        self.assertGreater(served_only, 0)

    def test_flag_off_calls_make_four_sdpa_launches_and_book_the_pinned_bookkeeping(self):
        reader = self.build()
        self.call(reader)
        self.assertEqual(len(self.device.sdpa), 4)
        self.assertEqual((reader.calls, [r.calls for r in reader.readers]), (1, [1, 1, 1, 1]))
        self.assertEqual(len(self.rig.refreshed), 4, 'one mask program per user bundle, as served')

    # -- flag on ------------------------------------------------------------------------------------------------------

    def test_attach_allocates_the_stacked_buffers_before_any_call_and_logs_the_engaged_marker(self):
        reader = self.build(**{FLAG: 'multi'})
        block_ = reader.multi
        self.assertIsNotNone(block_)
        self.assertEqual(tuple(block_.table.shape), (4, WIDTH))
        self.assertEqual(tuple(block_.cur_pos.shape), (4,))
        self.assertEqual(tuple(block_.mask.shape), (4, 1, 96, 256))
        self.assertEqual((block_.table.dtype, block_.cur_pos.dtype, block_.mask.dtype), ('int32', 'int32', 'bf16'))
        self.assertEqual((block_.table.layout, block_.cur_pos.layout, block_.mask.layout), ('row_major', 'row_major', 'tile'))
        self.assertEqual(vars(block_.config), dict(compute_with_storage_grid_size=(11, 10), exp_approx_mode=False,
                                                   q_chunk_size=0x51DEC021, k_chunk_size=256))
        engaged = [line for line in self.rig.logs if line.startswith(multi.ENGAGED)]
        self.assertEqual(engaged, ['%s config=multi grid=11x10 entries=4 users=4 flags=0x21 cores_per_entry=16 active_cores=64 audit=0'
                                   % multi.ENGAGED])
        self.assertEqual([program for tensors, program in self.device.generic], [('gather-program', 4), ('mask-program', 4)],
                         'one eager refresh at attach: a kernel that does not build fails the attach')
        self.assertEqual(self.device.generic[0][0][-2:], [block_.table, block_.cur_pos])
        self.assertEqual(len({id(tensor) for tensor in block_.owned}), len(block_.owned))

    def test_one_call_is_one_launch_of_the_multi_program_with_the_stacked_inputs(self):
        reader = self.build(**{FLAG: 'multi'})
        block_ = reader.multi
        self.device.generic.clear()
        output = self.call(reader)
        self.assertEqual(len(self.device.sdpa), 1)
        (options,) = self.device.sdpa
        self.assertIs(options['page_table_tensor'], block_.table)
        self.assertIs(options['cur_pos_tensor'], block_.cur_pos)
        self.assertIs(options['attn_mask'], block_.mask)
        self.assertIs(options['program_config'], block_.config)
        self.assertEqual((options['is_causal'], options['scale'], options['memory_config']), (False, 0.0625, 'dram'))
        sdpa_query = next(tensor for tensor in self.device.live if tensor.name == 'sdpa')
        self.assertEqual(tuple(sdpa_query.shape), (1, 4, 96, 256))
        self.assertEqual(tuple(output.shape), (1, 64, 6, 256))
        self.assertEqual([fold_['tag'] for fold_ in self.rig.folds], ['multi fold-in', 'multi fold-out'])
        fold_in, fold_out = self.rig.folds
        self.assertFalse(fold_in['inverse'])
        self.assertTrue(fold_out['inverse'])
        self.assertEqual(fold_in['plan'], multi.forward_plan(SEGMENTS))
        self.assertEqual(fold_out['plan'], multi.inverse_plan(SEGMENTS))
        self.assertIs(fold_out['destinations'][0], output)
        self.assertEqual([program for tensors, program in self.device.generic], [('gather-program', 4), ('mask-program', 4)],
                         'no scope: this call refreshed once')
        self.assertEqual(self.rig.refreshed, [], 'the per-user mask programs never run')
        self.assertTrue(any(multi.CALL_MARKER in line for line in self.rig.logs))
        self.assertEqual(len([line for line in self.rig.logs if multi.CALL_MARKER in line]), 1)

    def test_sixteen_layers_in_one_forward_refresh_once_and_book_what_model_batch_counts(self):
        reader = self.build(**{FLAG: 'multi'})
        self.device.generic.clear()
        before = reader.refresh_calls
        with reader.shared_masks(16):
            for layer in range(16):
                self.call(reader)
        self.assertEqual(reader.refresh_calls - before, len(reader.metadata), 'model_batch: one refresh per bundle per forward')
        self.assertEqual([program for tensors, program in self.device.generic], [('gather-program', 4), ('mask-program', 4)])
        self.assertEqual(len(self.device.sdpa), 16)
        self.assertEqual((reader.calls, [r.calls for r in reader.readers]), (16, [16] * 4))
        self.assertEqual(reader.multi.calls, 16)
        self.assertFalse(reader.failed)
        self.assertEqual(self.rig.refreshed, [])

    def test_the_call_budget_is_enforced_both_ways(self):
        reader = self.build(**{FLAG: 'multi'})
        with self.assertRaisesRegex(AssertionError, 'exact attention call budget'):
            with reader.shared_masks(16):
                for layer in range(15):
                    self.call(reader)
        self.assertTrue(reader.failed)
        reader = self.build(**{FLAG: 'multi'})
        with self.assertRaisesRegex(AssertionError, 'exceeded its attention call budget'):
            with reader.shared_masks(2):
                for layer in range(3):
                    self.call(reader)
        self.assertTrue(reader.failed)

    def test_scopes_do_not_nest_and_the_budget_is_bounded(self):
        reader = self.build(**{FLAG: 'multi'})
        with reader.shared_masks(1):
            with self.assertRaisesRegex(RuntimeError, 'cannot nest'):
                with reader.shared_masks(1):
                    pass
            self.call(reader)
        for budget in (0, 17, '16'):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, 'one-to-16'):
                with reader.shared_masks(budget):
                    pass

    def test_a_query_the_launch_is_not_written_for_raises_and_poisons(self):
        reader = self.build(**{FLAG: 'multi'})
        keys = self.device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys')
        with self.assertRaisesRegex(ValueError, 'Packed extent query geometry|cannot serve this call'):
            reader(self.device.device_tensor((1, 32, 6, 256), 'bf16', 'tile', 'dram'), keys, keys, scale=1.0, memory_config='dram')
        with self.assertRaisesRegex(ValueError, 'cannot serve this call: query is not bf16 TILE'):
            reader(self.device.device_tensor((1, 64, 6, 256), 'int32', 'tile', 'dram'), keys, keys, scale=1.0, memory_config='dram')
        with self.assertRaisesRegex(ValueError, 'output memory config'):
            reader(self.device.device_tensor((1, 64, 6, 256), 'bf16', 'tile', 'dram'), keys, keys, scale=1.0, memory_config='sharded')
        self.assertEqual(self.device.sdpa, [])

    def test_a_failure_poisons_every_segment_reader_and_releases_what_the_call_made(self):
        reader = self.build(**{FLAG: 'multi'})
        released = []
        with patch('tp_addresses.release_owned', side_effect=lambda operations, values: released.extend(values)):
            with patch.object(self.device, 'attention', side_effect=RuntimeError('boom')):
                self.device.transformer = SimpleNamespace(paged_scaled_dot_product_attention_decode=self.device.attention)
                with self.assertRaisesRegex(RuntimeError, 'boom'):
                    self.call(reader)
        self.assertTrue(all(segment.failed for segment in reader.readers))
        self.assertTrue(any(value.name == 'empty' for value in released), 'the stacked query is released')

    def test_the_output_is_not_released_and_the_intermediates_are(self):
        reader = self.build(**{FLAG: 'multi'})
        released = []
        with patch('tp_addresses.release_owned', side_effect=lambda operations, values: released.extend(values)):
            output = self.call(reader)
        names = sorted(value.name for value in released)
        self.assertEqual(names, ['empty', 'sdpa'], 'the stacked query and the SDPA result')
        self.assertFalse(any(value is output for value in released))

    def test_close_releases_the_multi_buffers_once(self):
        reader = self.build(**{FLAG: 'multi'})
        owned = list(reader.multi.owned)
        released = []
        with patch('tp_addresses.release_owned', side_effect=lambda operations, values: released.extend(values)):
            reader.close()
            reader.close()
        self.assertTrue(all(any(value is tensor for value in released) for tensor in owned))
        self.assertTrue(reader.multi.closed)

    def test_the_bookkeeping_matches_the_served_reader_after_a_forward(self):
        served = self.build()
        with served.shared_masks(16):
            for layer in range(16):
                self.call(served)
        reader = self.build(**{FLAG: 'multi'})
        with reader.shared_masks(16):
            for layer in range(16):
                self.call(reader)
        self.assertEqual((reader.calls, [r.calls for r in reader.readers], reader.refresh_calls, reader.failed),
                         (served.calls, [r.calls for r in served.readers], served.refresh_calls, served.failed))

    # -- refusals -----------------------------------------------------------------------------------------------------

    def test_a_user_the_pinned_reader_does_not_qualify_never_reaches_the_multi_path(self):
        # the pinned reader builds T16 users in one bundle of two eight-row groups and nothing else (a T32 user is three groups and
        # one; a T8 user one group), and at most 64 rows: so a block is one to four T16 users and the multi path's own shape check
        # (CensusTests) is a second wall. The refusal closes the readers it had built.
        closed = []
        original = folded.PackedExtentReplayReader.close
        with patch.object(folded.PackedExtentReplayReader, 'close', lambda self: (closed.append(1), original(self))[1]):
            with env(**{FLAG: 'multi'}):
                lent = [lend(self.device, 32)]
                with self.assertRaisesRegex(ValueError, 'qualified at G8B2 only'):
                    folded.PackedExtentReplayReader(self.device, MESH, ((0, 32),), WIDTH, [host_table(1)], storage=lent,
                                                    max_group_rows=8, starts=(4200,))
        self.assertTrue(closed)

    def test_more_than_four_users_are_refused_by_the_pinned_block_before_the_multi_path_sees_them(self):
        segments = tuple((16 * user, 16 * user + 16) for user in range(5))
        with env(**{FLAG: 'multi'}):
            with self.assertRaisesRegex(ValueError, 'at most a 64-row block'):
                block(self.device, segments, starts=(4200, 9000, 4300, 5000, 6000))

    def test_a_multi_attach_that_fails_on_the_shape_closes_the_reader(self):
        # a block the pinned reader accepts but the factory would give 15 cores per entry: the multi path refuses by name.
        closed = []
        original = folded.PackedExtentReplayReader.close
        with patch.object(folded.PackedExtentReplayReader, 'close', lambda self: (closed.append(1), original(self))[1]):
            with env(**{FLAG: 'multi'}), patch('sdpa_multi_tp.MAX_USERS', 8),                     patch('sdpa_multi_tp.plan_problem', side_effect=lambda users, grid: 'the factory gives 15 cores per entry at B=7'):
                with self.assertRaisesRegex(ValueError, 'QWEN_FAST_TP4_SDPA=multi: the factory gives 15 cores per entry at B=7'):
                    block(self.device)
        self.assertTrue(closed)

    def test_a_kernel_that_does_not_run_at_attach_fails_the_attach_and_releases_what_was_allocated(self):
        released = []
        with patch('tp_addresses.release_owned', side_effect=lambda operations, values: released.extend(values)):
            with patch.object(self.device, 'generic_op', side_effect=RuntimeError('kernel did not build')):
                with self.assertRaisesRegex(RuntimeError, 'kernel did not build'):
                    self.build(**{FLAG: 'multi'})
        self.assertTrue(any(value.name == 'upload' and tuple(value.shape) == (4, WIDTH) for value in released),
                        'the stacked table is released with the rest')

    def test_the_extent_audit_needs_the_multi_audit(self):
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_EXTENT_AUDIT=1 reads back'):
            self.build(**{FLAG: 'multi', multi.EXTENT_AUDIT_FLAG: '1'})
        self.build(**{FLAG: 'multi', multi.EXTENT_AUDIT_FLAG: '1', AUDIT: '1'})

    def test_the_audit_flag_without_multi_is_refused(self):
        for value in ('served', 'grid8x4'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'audits the multi launch'):
                self.build(**{FLAG: value, AUDIT: '1'})
        with self.assertRaisesRegex(ValueError, 'audits the multi launch: it needs QWEN_FAST_TP4_SDPA=multi'):
            self.build(**{AUDIT: '1'})
        with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
            self.build(**{FLAG: 'multi', AUDIT: 'yes'})

    def test_the_audit_flag_off_values_change_nothing(self):
        for value in ('', '0'):
            reader = self.build(**{FLAG: 'multi', AUDIT: value})
            self.assertFalse(reader.multi.audit)
        reader = self.build(**{AUDIT: '0'})
        self.assertIsNone(reader.multi)

    # -- the audit ----------------------------------------------------------------------------------------------------

    def test_under_the_audit_both_paths_run_and_the_compare_is_launched_per_layer(self):
        reader = self.build(**{FLAG: 'multi', AUDIT: '1'})
        self.assertTrue(reader.multi.audit)
        self.assertEqual(tuple(reader.multi.counters.shape), (16 * 64, 16))
        self.assertIn('audit=1', [line for line in self.rig.logs if line.startswith(multi.ENGAGED)][0])
        self.device.sdpa.clear()
        before = reader.refresh_calls
        with reader.shared_masks(16):
            for layer in range(16):
                output = self.call(reader)
        self.assertEqual(len(self.device.sdpa), 16 * 5, 'four served launches and one multi launch per layer')
        self.assertEqual(reader.refresh_calls - before, len(reader.metadata), 'booked once, by the served path')
        self.assertEqual(len(self.rig.audits), 16)
        self.assertEqual([entry[3] for entry in self.rig.audits], list(range(16)), 'one counter slot per layer')
        for served_output, built, counters, slot in self.rig.audits:
            self.assertIs(counters, reader.multi.counters)
            self.assertEqual(tuple(served_output.shape), (1, 64, 6, 256))
            self.assertEqual(tuple(built.shape), (1, 64, 6, 256))
            self.assertIsNot(served_output, built)
        self.assertEqual((reader.calls, [r.calls for r in reader.readers]), (16, [16] * 4), 'booked once: by the served path')
        self.assertEqual(len(self.rig.refreshed), 4, 'the per-user masks are refreshed once for the served launches')
        self.assertIs(self.rig.audits[-1][1], output, 'the multi launch\'s output is the one returned')

    def test_the_audits_served_outputs_are_released_and_the_multi_output_is_not(self):
        reader = self.build(**{FLAG: 'multi', AUDIT: '1'})
        released = []
        with patch('tp_addresses.release_owned', side_effect=lambda operations, values: released.extend(values)):
            output = self.call(reader)
        served_output = self.rig.audits[0][0]
        self.assertTrue(any(value is served_output for value in released))
        self.assertFalse(any(value is output for value in released))

    def counters(self, layers, **options):
        return AuditMapTests.counters(None, layers, **options)

    def test_an_audit_round_logs_exact_when_every_layer_is_equal_and_live(self):
        reader = self.build(**{FLAG: 'multi', AUDIT: '1'})
        with reader.shared_masks(16):
            for layer in range(16):
                self.call(reader)
        self.device.counters_value = self.counters(16)
        self.assertGreater(reader.sdpa_audit_round(1), 0)
        lines = [line for line in self.rig.logs if line.startswith(multi.AUDIT_MARKER)]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r'^\[PINDIAG\] tp4 sdpa audit 1 exact=True users=4 layers=16 chips=4 words=\d+ live=\d+$')
        self.assertNotIn(multi.AUDIT_MISMATCH, lines[0])
        reader.sdpa_audit_round(2)
        self.assertEqual(len([line for line in self.rig.logs if line.startswith(multi.AUDIT_MARKER)]), 2)

    def test_a_differing_word_logs_the_mismatch_and_raises(self):
        reader = self.build(**{FLAG: 'multi', AUDIT: '1'})
        with reader.shared_masks(16):
            for layer in range(16):
                self.call(reader)
        self.device.counters_value = self.counters(16, differing={(3, 9): 2})
        with self.assertRaisesRegex(AssertionError, 'tp4 sdpa audit MISMATCH round=7 users=4'):
            reader.sdpa_audit_round(7)
        self.assertTrue(any(multi.AUDIT_MISMATCH in line and 'layer 3' in line for line in self.rig.logs))
        self.assertFalse(any('exact=True' in line for line in self.rig.logs))

    def test_an_audit_that_compared_nothing_fails(self):
        reader = self.build(**{FLAG: 'multi', AUDIT: '1'})
        with self.assertRaisesRegex(AssertionError, 'no attention layer ran the multi launch'):
            self.device.counters_value = self.counters(0)
            reader.sdpa_audit_round(1)

    def test_the_audit_round_is_inert_without_the_audit(self):
        reader = self.build(**{FLAG: 'multi'})
        self.assertEqual(reader.sdpa_audit_round(1), 0)
        self.assertEqual([line for line in self.rig.logs if multi.AUDIT_MARKER in line], [])

    def test_packed_verifier_calls_the_audit_hook_after_the_replay(self):
        text = (HERE / 'packed_verifier.py').read_text(encoding='utf-8')
        self.assertIn("getattr(getattr(self.fixture, 'replay_reader', None), 'sdpa_audit_round', None)", text)
        self.assertLess(text.index('tp4_vglue.audit_round(self.operations'), text.index('sdpa_audit(self.rounds + 1)'))
        self.assertLess(text.index('sdpa_audit(self.rounds + 1)'), text.index('predictions = [host[slice(*segment_rows'))


# --- the flag, the binding, the smoke rule ----------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_multi_is_servable_and_the_others_are_as_before(self):
        self.assertTrue(sdpa_long_tp.CONFIGS['multi']['servable'])
        self.assertEqual(sdpa_long_tp.selected({'QWEN_FAST_TP': '4', FLAG: 'multi'}), 'multi')
        self.assertIn('multi', sdpa_long_tp.servable_names())
        for name in ('rowsplit', 'ra'):
            self.assertFalse(sdpa_long_tp.CONFIGS[name]['servable'])

    def test_multi_is_refused_at_the_pair(self):
        with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_TP=4'):
            sdpa_long_tp.selected({FLAG: 'multi'})

    def test_the_plan_for_multi_is_the_mesh_grid(self):
        self.assertEqual(sdpa_long_tp.plan('multi', (11, 10)), (11, 10))

    def test_the_audit_flag(self):
        self.assertFalse(multi.audit_enabled({}))
        self.assertFalse(multi.audit_enabled({AUDIT: '0'}))
        self.assertFalse(multi.audit_enabled({AUDIT: ' '}))
        self.assertTrue(multi.audit_enabled({AUDIT: '1'}))
        with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
            multi.audit_enabled({AUDIT: 'true'})

    def test_the_flag_off_selection_is_unchanged_by_the_audit_module(self):
        self.assertIsNone(sdpa_long_tp.selected({'QWEN_FAST_TP': '4'}))
        self.assertIsNone(sdpa_long_tp.selected({'QWEN_FAST_TP': '4', AUDIT: '0'}))
        with self.assertRaisesRegex(ValueError, 'audits the multi launch'):
            sdpa_long_tp.selected({'QWEN_FAST_TP': '4', AUDIT: '1'})

    def test_the_marker_names_the_launch(self):
        with four():
            line = multi.marker(multi.plan(4, GRID), GRID, True)
        self.assertEqual(line, '%s config=multi grid=11x10 entries=4 users=4 flags=0x21 cores_per_entry=16 active_cores=64 audit=1'
                         % sdpa_long_tp.ENGAGED)


class SmokeTests(unittest.TestCase):
    ENGAGED = '[PINDIAG] tp4 sdpa engaged config=multi grid=11x10 entries=4 users=4 flags=0x21 cores_per_entry=16 active_cores=64 audit=%d'
    CALL = '[PINDIAG] tp4 sdpa multi call users=4 flags=0x21 cores_per_entry=16 audit=%d'
    PASS = '[PINDIAG] tp4 sdpa audit %d exact=True users=4 layers=16 chips=4 words=1 live=1'

    def text(self, audit, passes=2, extra=()):
        lines = [self.ENGAGED % audit, self.CALL % audit] + [self.PASS % (index + 1) for index in range(passes)] + list(extra)
        return '\n'.join(['noise'] + lines + ['more noise'])

    def test_a_timed_multi_profile_needs_the_engaged_and_the_call_lines(self):
        env_ = {FLAG: 'multi'}
        self.assertEqual(c2_smoke_check.sdpa_multi_problems(env_, self.text(0, passes=0)), [])
        quiet = c2_smoke_check.sdpa_multi_problems(env_, 'nothing')
        self.assertEqual(len(quiet), 2)
        self.assertIn('config=multi flags=0x21', quiet[0])
        self.assertIn('built and never called', quiet[1])
        served = '[PINDIAG] tp4 sdpa engaged config=served grid=11x10 entries=8'
        self.assertEqual(len(c2_smoke_check.sdpa_multi_problems(env_, served + '\n' + self.CALL % 0)), 1)
        self.assertEqual(len(c2_smoke_check.sdpa_multi_problems(env_, self.ENGAGED % 0)), 1, 'no call line')

    def test_the_audited_profile_needs_a_passing_audit_line_and_no_other(self):
        env_ = {FLAG: 'multi', AUDIT: '1'}
        self.assertEqual(c2_smoke_check.sdpa_multi_problems(env_, self.text(1)), [])
        none = c2_smoke_check.sdpa_multi_problems(env_, self.text(1, passes=0))
        self.assertEqual(len(none), 1)
        self.assertIn('nothing was compared', none[0])
        mismatch = '[PINDIAG] tp4 sdpa audit MISMATCH round=3 users=4 differing=2 chip 0 layer 3: 2 of 262144 words differ'
        found = c2_smoke_check.sdpa_multi_problems(env_, self.text(1, extra=[mismatch]))
        self.assertEqual(len(found), 1)
        self.assertIn('audit found a difference', found[0])
        self.assertIn('layer 3', found[0])
        false = '[PINDIAG] tp4 sdpa audit 9 exact=False users=4'
        self.assertEqual(len(c2_smoke_check.sdpa_multi_problems(env_, self.text(1, extra=[false]))), 1)
        only_mismatch = '\n'.join([self.ENGAGED % 1, self.CALL % 1, mismatch])
        self.assertEqual(len(c2_smoke_check.sdpa_multi_problems(env_, only_mismatch)), 2, 'a mismatch line is not a passing line')

    def test_other_profiles_are_not_judged(self):
        for env_ in ({}, None, {FLAG: 'served'}, {FLAG: 'grid8x4'}, {FLAG: 'off'}):
            self.assertEqual(c2_smoke_check.sdpa_multi_problems(env_, 'nothing'), [], env_)

    def test_the_check_is_wired_into_the_smoke_problems(self):
        text = (HERE / 'c2_smoke_check.py').read_text(encoding='utf-8')
        self.assertIn('problems += sdpa_multi_problems(env, container_text)', text)
        self.assertEqual(c2_smoke_check.SDPA_LONG_ENGAGED, sdpa_long_tp.ENGAGED)
        self.assertEqual((c2_smoke_check.SDPA_MULTI_CALL, c2_smoke_check.SDPA_AUDIT_LINE, c2_smoke_check.SDPA_AUDIT_MISMATCH),
                         (multi.CALL_MARKER, multi.AUDIT_MARKER, multi.AUDIT_MISMATCH))
        self.assertEqual(c2_smoke_check.SDPA_AUDIT_FLAG, multi.AUDIT_FLAG)

    def test_the_engaged_line_the_module_logs_passes_the_rule(self):
        line = multi.ENGAGED + ' config=multi grid=11x10 entries=4 users=4 flags=0x%x cores_per_entry=16 active_cores=64 audit=1' % multi.FLAGS
        text = '\n'.join([line, self.CALL % 1, self.PASS % 1])
        self.assertEqual(c2_smoke_check.sdpa_multi_problems({FLAG: 'multi', AUDIT: '1'}, text), [])


# --- the profiles ---------------------------------------------------------------------------------------------------------

class ProfileTests(unittest.TestCase):
    def test_the_timed_twin_is_the_control_plus_exactly_the_flag(self):
        control, twin = PROFILES[CONTROL], PROFILES[TWIN]
        self.assertEqual(set(twin['env']) - set(control['env']), {FLAG})
        self.assertEqual({key: value for key, value in twin['env'].items() if key != FLAG}, control['env'])
        self.assertEqual(twin['env'][FLAG], 'multi')
        for key in set(control) | set(twin):
            if key not in ('env', 'description'):
                self.assertEqual(control.get(key), twin.get(key), key)

    def test_the_audited_twin_is_the_timed_twin_plus_exactly_the_audit_flag(self):
        twin, audited = PROFILES[TWIN], PROFILES[AUDITED]
        self.assertEqual(set(audited['env']) - set(twin['env']), {AUDIT})
        self.assertEqual({key: value for key, value in audited['env'].items() if key != AUDIT}, twin['env'])
        self.assertEqual(audited['env'][AUDIT], '1')
        for key in set(twin) | set(audited):
            if key not in ('env', 'description'):
                self.assertEqual(twin.get(key), audited.get(key), key)

    def test_both_are_gate_only_eight_seat_262k_profiles_with_the_waiver_and_unqualified_text(self):
        for name in (TWIN, AUDITED):
            body = PROFILES[name]
            self.assertIs(body['gate_only'], True, name)
            self.assertEqual(body['env']['QWEN_C2_GATE_PROFILE'], '1')
            self.assertEqual(body['env']['QWEN_FAST_262K_EVIDENCE_WAIVER'], '1')
            self.assertEqual(body['env']['QWEN_FAST_MAX_POSITION'], '262144')
            self.assertEqual(body['engine']['max-num-seqs'], 8)
            self.assertEqual(body['env']['QWEN_FAST_M3_BLOCKS'], '2')
            self.assertIn('Not for traffic', body['description'])
            self.assertIn('UNVERIFIED on hardware', body['description'])
            self.assertEqual(body['env']['QWEN_FAST_VERIFY_T1_AUDIT'], '0')
            self.assertNotIn(multi.EXTENT_AUDIT_FLAG, body['env'])

    def test_the_audited_twin_is_never_a_timed_arm(self):
        self.assertIn('never a timed arm', PROFILES[AUDITED]['description'])
        self.assertNotIn('never a timed arm', PROFILES[TWIN]['description'])

    def test_no_other_profile_carries_the_audit_flag_and_the_default_is_untouched(self):
        for name, body in PROFILES.items():
            if name != AUDITED:
                self.assertNotIn(AUDIT, body.get('env', {}), name)
        self.assertEqual(json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['default'], 'c2-packed-tp4')

    def test_the_profile_describes_the_exactness_and_the_two_block_launch(self):
        text = PROFILES[TWIN]['description']
        for word in ('ONE launch per layer', '0x21', 'U=4', '15 and 13', 'byte-identical', 'docs/sdpa-multi.md', 'paired control'):
            self.assertIn(word, text)
        self.assertTrue((ROOT / 'docs' / 'sdpa-multi.md').is_file())


# --- the shipment -----------------------------------------------------------------------------------------------------------

class ShipmentTests(unittest.TestCase):
    def read(self, path):
        return (ROOT / path).read_text(encoding='utf-8')

    def test_every_runtime_file_is_in_the_three_copy_lists(self):
        dockerfile = self.read('docker/qwen-fast-serving.Dockerfile')
        workflow = self.read('.github/workflows/qwen-fast-serving-image.yml')
        overlay = self.read('docker/qwen-c2-overlay.txt').splitlines()
        self.assertEqual(sdpa_long_tp.RUNTIME_FILES[1:], multi.RUNTIME_FILES)
        for name in multi.RUNTIME_FILES:
            self.assertTrue((HERE / name).is_file(), name)
            self.assertIn('scripts/ci/%s ' % name, dockerfile)
            self.assertRegex(workflow, r'for name in [^\n]*\b%s\b' % re.escape(name))
            self.assertIn('scripts/ci/%s' % name, overlay)

    def test_the_cpu_allowlist_names_the_tests(self):
        cpu = self.read('.github/workflows/qwen-integration-cpu.yml')
        for module in ('test_sdpa_multi_tp', 'test_tp4_sdpa_multi_window'):
            self.assertRegex(cpu, r'python -B -m unittest [^\n]*\b%s\b' % module)

    def test_the_module_is_stdlib_at_import_and_py37_clean(self):
        text = (HERE / 'sdpa_multi_tp.py').read_text(encoding='utf-8')
        self.assertNotRegex(text, r'(?m)^import (torch|ttnn|numpy)')
        self.assertNotRegex(text, r'(?m)^(import|from) sdpa_long_tp')
        self.assertNotIn(':=', text)
        self.assertNotIn('strict=True', text)

    def test_the_new_files_are_lf(self):
        for name in (*multi.RUNTIME_FILES, 'test_sdpa_multi_tp.py', 'test_tp4_sdpa_multi_window.py'):
            path = HERE / name
            if path.is_file():
                self.assertNotIn(b'\r', path.read_bytes(), name)
        self.assertNotIn(b'\r', (ROOT / 'docs' / 'sdpa-multi.md').read_bytes())

    def test_the_reader_twin_still_subclasses_the_pinned_reader(self):
        self.assertEqual(folded.PackedExtentReplayReader.__mro__[1].__module__, 'extent_attention_replay_tp')
        self.assertNotIn('multi', quad.PackedExtentReplayReader.__dict__)


if __name__ == '__main__':
    unittest.main()
