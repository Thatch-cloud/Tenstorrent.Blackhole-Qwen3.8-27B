"""QWEN_FAST_OCTO_ATTN_BUNDLE (octo_attn_bundle): the octo block's attention in ceil(8 / N) launches per layer, byte for byte the eight single-entry launches.

What is tested, all on CPU:
  - the flags: strict values, the admission questions, the one environment read a flag-off process makes;
  - the shape: the launch split (4 + 4 at the recommended 4), that every launch keeps 16 cores per entry (the factory's allocation as arithmetic), the fold chunks, every refusal;
  - the launches' index maps EMULATED from the runtime arguments the planners write and compared with the served composition: the fold-in and fold-out of N consecutive users'
    eight tokens, the mask of N users' own positions words (bit for bit the pinned narrow mask of the single launch for every start-word class), the page-table gather;
  - the builder (the rows-parameterised mask launch) on a fake ttnn;
  - the reader twin on the four-chip fake device with the octo block: flag off is the served path (the module is not even imported), flag on makes ceil(8 / N) SDPA launches
    per layer with the stacked inputs, one refresh per forward booked as model_batch counts it, the call budget, the failure path, the refusals, the audit;
  - the smoke rule, the shipment and the doc.

What this cannot show, and what is left to the card: that K64j's 0x21 program at B > 1 entries of EIGHT rows equals the single launches (the audit, job OA1), and the time it saves.

Run at py 3.11: `py -3.11 -B -m unittest test_octo_attn_bundle` from scripts/ci.
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
import extent_attention_fold_tp as folded
import extent_attention_octo_tp as octo
import extent_attention_replay as pinned
import extent_attention_replay_tp as quad
import octo_attn_bundle as bundle
import sdpa_multi_tp as multi
import test_sdpa_multi_tp as multitests
from test_extent_attention_replay import MESH, WIDTH, host_table
from test_extent_attention_replay_tp import FourChipDevice, Harness, four, lend
from test_tp4_vglue_attention import HEAD, COLUMNS, from_pages, run_launch, to_pages

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FLAG, AUDIT = bundle.FLAG, bundle.AUDIT_FLAG
SEGMENTS = tuple((8 * user, 8 * user + 8) for user in range(8))
M3_SEGMENTS = ((0, 16), (16, 32), (32, 48), (48, 64))
GRID = (11, 10)
OCTO_ON = {'QWEN_FAST_OCTO': 'alternate'}


def env(**values):
    return patch.dict(os.environ, values)


def flags(size='4', audit=None, **more):
    values = dict(OCTO_ON, **{FLAG: size})
    if audit is not None:
        values[AUDIT] = audit
    values.update(more)
    return values


# --- the flags ------------------------------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_off_is_unset_empty_or_zero(self):
        for value in (None, '', '0', ' 0 ', ' '):
            source = {} if value is None else {FLAG: value}
            self.assertIsNone(bundle.per_launch(source), value)
            self.assertFalse(bundle.requested(source), value)
        self.assertIsNone(bundle.selected({}))
        self.assertEqual(bundle.admission_problems({}), [])

    def test_a_value_is_the_users_per_launch_two_to_six_and_nothing_else(self):
        for size in range(2, 7):
            self.assertEqual(bundle.per_launch({FLAG: str(size)}), size)
            self.assertTrue(bundle.requested({FLAG: str(size)}))
        for value in ('1', '7', '8', '04', '4.0', '-3', 'x', 'on', 'true', '2,4'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'users per launch'):
                bundle.per_launch({FLAG: value})

    def test_the_audit_flag_is_strict(self):
        self.assertFalse(bundle.audit_enabled({}))
        for value in ('', '0'):
            self.assertFalse(bundle.audit_enabled({AUDIT: value}))
        self.assertTrue(bundle.audit_enabled({AUDIT: '1'}))
        for value in ('2', 'yes', 'true'):
            with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                bundle.audit_enabled({AUDIT: value})
        self.assertTrue(bundle.requested({AUDIT: '1'}))

    def test_the_twins_environment_read_is_this_modules(self):
        for source in ({}, {FLAG: '0'}, {FLAG: '4'}, {AUDIT: '1'}, {AUDIT: '0'}, {FLAG: '', AUDIT: ''}):
            with self.subTest(source=source), patch.dict(os.environ, source, clear=True):
                self.assertEqual(folded.octo_flags_requested(), bundle.requested(source))
        self.assertEqual(folded.OCTO_ATTN_BUNDLE_FLAGS, (FLAG, AUDIT))

    def test_the_admission_questions(self):
        good = dict(flags('4'), QWEN_FAST_TP='4')
        self.assertEqual(bundle.admission_problems(good), [])
        self.assertEqual(bundle.selected(good), 4)
        self.assertEqual(bundle.admission_problems(dict(good, QWEN_FAST_OCTO='live')), [])
        self.assertIn('needs QWEN_FAST_OCTO=live or alternate', bundle.admission_problems({FLAG: '4', 'QWEN_FAST_TP': '4'})[0])
        self.assertIn('needs QWEN_FAST_OCTO=live or alternate', bundle.admission_problems(dict(good, QWEN_FAST_OCTO='off'))[0])
        self.assertIn('needs QWEN_FAST_TP=4', ' '.join(bundle.admission_problems(dict(flags('4')))))
        self.assertIn('two ways to bundle', ' '.join(bundle.admission_problems(dict(good, QWEN_FAST_TP4_SDPA='multi'))))
        self.assertEqual(bundle.admission_problems(dict(good, QWEN_FAST_TP4_SDPA='off')), [])
        self.assertIn('needs %s' % FLAG, bundle.admission_problems(dict(OCTO_ON, QWEN_FAST_TP='4', **{AUDIT: '1'}))[0])
        self.assertEqual(bundle.admission_problems({FLAG: '9'})[0][:len(FLAG)], FLAG)
        with self.assertRaises(ValueError):
            bundle.selected(dict(good, QWEN_FAST_OCTO='off'))
        self.assertIn('unequal sizes [3, 3, 2]', ' '.join(bundle.admission_problems(dict(good, **{FLAG: '3'}))))
        for size in ('2', '4', '5', '6'):
            self.assertEqual(bundle.admission_problems(dict(good, **{FLAG: size})), [], size)


# --- the shape --------------------------------------------------------------------------------------------------------------------

class ShapeTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_launch_split(self):
        self.assertEqual(bundle.launch_groups(8, 4), ((0, 1, 2, 3), (4, 5, 6, 7)))
        self.assertEqual(bundle.launch_groups(8, 2), ((0, 1), (2, 3), (4, 5), (6, 7)))
        self.assertEqual(bundle.launch_groups(8, 3), ((0, 1, 2), (3, 4, 5), (6, 7)))
        self.assertEqual(bundle.launch_groups(8, 5), ((0, 1, 2, 3), (4, 5, 6, 7)))
        self.assertEqual(bundle.launch_groups(8, 6), ((0, 1, 2, 3), (4, 5, 6, 7)))
        self.assertEqual(bundle.launch_groups(7, 6), ((0, 1, 2, 3), (4, 5, 6)))
        self.assertEqual(bundle.launch_groups(8, 8), ((0, 1, 2, 3, 4, 5, 6, 7),))
        for size in range(1, 9):
            groups = bundle.launch_groups(8, size)
            self.assertEqual([user for group in groups for user in group], list(range(8)))
            self.assertEqual(len(groups), -(-8 // size))
            self.assertLessEqual(max(map(len, groups)), size)
            self.assertLessEqual(max(map(len, groups)) - min(map(len, groups)), 1, 'near-equal')
        for bad in ((0, 4), (8, 0), ('8', 4), (8, 4.0)):
            with self.assertRaises(ValueError):
                bundle.launch_groups(*bad)

    def test_every_launch_the_flag_allows_keeps_sixteen_cores_per_entry(self):
        for size in range(bundle.MIN_PER_LAUNCH, bundle.MAX_PER_LAUNCH + 1):
            for group in bundle.launch_groups(8, size):
                found = multi.factory_cores(len(group), GRID[0] * GRID[1])
                self.assertEqual(found['cores_per_entry'], multi.ENTRY_CORES, (size, group))
                self.assertIsNone(multi.plan_problem(len(group), GRID), (size, group))
        # one user more is the factory's 15 cores, and a group of 8 is its 13: the reason 8 users are two launches
        self.assertIsNotNone(multi.plan_problem(7, GRID))
        self.assertIsNotNone(multi.plan_problem(8, GRID))
        self.assertEqual(bundle.MAX_PER_LAUNCH, max(users for users in range(1, 9) if multi.plan_problem(users, GRID) is None))

    def test_the_chunks_tile_the_block_in_ascending_order_one_group_per_user(self):
        for size in (2, 3, 4, 6):
            groups = bundle.launch_groups(8, size)
            chunks = bundle.chunks_of(groups)
            self.assertEqual([chunk.batches for chunk in chunks], [len(group) for group in groups])
            self.assertEqual([chunk.first for chunk in chunks], [8 * group[0] for group in groups])
            self.assertTrue(all(chunk.rows == 8 for chunk in chunks))
            self.assertEqual([token for chunk in chunks for token in chunk.tokens()], list(range(64)))
            self.assertEqual([chunk.groups for chunk in chunks],
                             [tuple((8 * index, 8) for index in range(len(group))) for group in groups])

    def test_the_entry_is_the_single_launchs_entry(self):
        self.assertEqual((bundle.entry_rows(), bundle.entry_tiles()), (48, 2))
        self.assertEqual(bundle.FLAGS, octo.OCTO_FLAGS)
        self.assertEqual(bundle.FLAGS, 0x21)
        self.assertEqual(bundle.ROWS, octo.OCTO_ROWS)
        self.assertEqual(bundle.USERS, 8)

    def test_every_refusal(self):
        groups = bundle.launch_groups(8, 4)
        self.assertIsNone(bundle.shape_problem(SEGMENTS, [8] * 8, [1] * 8, groups, GRID))
        self.assertIn('at least two users', bundle.shape_problem(SEGMENTS[:1], [8], [1], ((0,),), GRID))
        self.assertIn('one row count and one bundle count', bundle.shape_problem(SEGMENTS, [8] * 7, [1] * 8, groups, GRID))
        self.assertIn('T8 octo user', bundle.shape_problem(M3_SEGMENTS, [16] * 4, [1] * 4, ((0, 1), (2, 3)), GRID))
        self.assertIn('one bundle', bundle.shape_problem(SEGMENTS, [8] * 8, [1] * 7 + [2], groups, GRID))
        self.assertIn('do not tile', bundle.shape_problem(SEGMENTS[::-1], [8] * 8, [1] * 8, groups, GRID))
        self.assertIn('do not cover the users in order', bundle.shape_problem(SEGMENTS, [8] * 8, [1] * 8, ((0, 1, 2, 3), (5, 4, 6, 7)), GRID))
        self.assertIn('do not cover the users in order', bundle.shape_problem(SEGMENTS, [8] * 8, [1] * 8, ((0, 1, 2, 3),), GRID))
        refused = bundle.shape_problem(SEGMENTS, [8] * 8, [1] * 8, ((0, 1, 2, 3, 4, 5, 6), (7,)), GRID)
        self.assertIn('15 cores per entry', refused)
        self.assertIn('every user would run a different partition and tree', refused)
        self.assertIn('13 cores per entry', bundle.shape_problem(SEGMENTS, [8] * 8, [1] * 8, (tuple(range(8)),), GRID))
        self.assertIn('unequal sizes [3, 3, 2]', bundle.shape_problem(SEGMENTS, [8] * 8, [1] * 8, bundle.launch_groups(8, 3), GRID))
        self.assertIn('never run over mixed shapes', bundle.unequal_reason(bundle.launch_groups(8, 3)))

    def test_the_exactness_lines_quote_the_factory_and_the_citations_are_in_the_fixtures(self):
        text = '\n'.join(bundle.exactness())
        for fragment in ('16 * B', ':198-200', 'rt_args_common :35-93', ':240', 'PNHt = 2', ':398, :407', ':785', 'QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT'):
            self.assertIn(fragment, text)
        self.assertEqual(multi.citation_problems(ROOT / 'optimisation' / 'ttnn-op'), [])

    def test_the_constants_match_the_modules_that_own_them(self):
        self.assertEqual(bundle.MARKER, '[OCTO-ATTN-BUNDLE]')
        self.assertNotIn('[OCTO]', bundle.MARKER, 'a flag-off octo arm\'s "no [OCTO] line" rule never sees this lever')
        self.assertEqual(bundle.FLAGS, multi.FLAGS)
        self.assertEqual(multi.TILE_COLUMNS, 8)
        self.assertEqual(bundle.RUNTIME_FILES, ('octo_attn_bundle.py',))


# --- the launches' index maps, emulated -------------------------------------------------------------------------------------------

def entry_stub(user, stacked):
    """A stand-in for the SDPA whose output row depends on its input row and its USER only, never on the entry layout or its slot in a launch, as the real op's does."""
    return stacked * 1.5 + user * 1000.0


def served_octo(query, stub):
    """The served composition at eight users of one group: per user its eight tokens as a (1, 1, 48, 256) entry, the SDPA, the entry back, the users joined."""
    users = []
    for user, (first, last) in enumerate(SEGMENTS):
        entry = query[:, first:last].reshape(1, 1, 8 * HEAD, 256)
        result = entry_stub(user, entry)
        users.append(result.reshape(1, 8, HEAD, 256))
    return torch.cat(users, dim=1)


class FoldMapTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_bundled_fold_in_sdpa_stub_fold_out_is_the_served_composition_byte_for_byte(self):
        query = torch.arange(64 * HEAD * 256, dtype=torch.float32).reshape(1, 64, HEAD, 256) + 0.5
        expected = served_octo(query, entry_stub)
        for size in (2, 3, 4, 6):
            groups = bundle.launch_groups(8, size)
            chunks = bundle.chunks_of(groups)
            stacked_addresses = [100 + index for index in range(len(chunks))]
            result_addresses = [200 + index for index in range(len(chunks))]
            buffers = {1: to_pages(query), 300: {}}
            for address in stacked_addresses:
                buffers[address] = {}
            flat = fold.flat_tasks(fold.forward_tasks(chunks), [1] * len(chunks), stacked_addresses)
            run_launch(fold.runtime_arguments(False, fold.distribute(flat, 110)), buffers)
            for index, (group, chunk) in enumerate(zip(groups, chunks)):
                stack, clean = from_pages(buffers[stacked_addresses[index]], len(group), 8 * HEAD)
                self.assertTrue(clean, 'the stacked padding is zero')
                for slot, user in enumerate(group):
                    served_entry = query[:, 8 * user:8 * user + 8].reshape(1, 1, 8 * HEAD, 256)
                    self.assertTrue(torch.equal(stack[:, slot:slot + 1], served_entry), 'size %d user %d is the single launch\'s entry' % (size, user))
                result = torch.cat([entry_stub(user, stack[:, slot:slot + 1]) for slot, user in enumerate(group)], dim=1)
                buffers[result_addresses[index]] = to_pages(result)
            flat = fold.flat_tasks(fold.inverse_tasks(chunks), result_addresses, [300] * len(chunks))
            run_launch(fold.runtime_arguments(True, fold.distribute(flat, 110)), buffers)
            built, clean = from_pages(buffers[300], 64, HEAD)
            self.assertTrue(clean)
            self.assertTrue(torch.equal(built, expected), size)
            self.assertEqual(sorted(buffers[300]), list(range(64 * COLUMNS)), 'every output page is written exactly once')
            self.assertEqual(sum(len(part) for part in fold.forward_tasks(chunks)), 8 * 2 * COLUMNS, 'the same 128 fold-in pages as eight single launches')
            self.assertEqual(sum(len(part) for part in fold.inverse_tasks(chunks)), 64 * COLUMNS, 'the same 512 fold-out pages')

    def test_the_fold_functions_the_bundle_calls_take_any_number_of_groups_per_chunk(self):
        # fold_in / fold_out do not call fold.problem(): the (1, 2) bundle bound is V3a's own gate, and the planners are indexed by the group's batch slot.
        chunks = bundle.chunks_of(bundle.launch_groups(8, 4))
        self.assertIsNotNone(fold.problem(SimpleNamespace(shape=(1, 64, HEAD, 256), dtype='bf16', layout='tile', memory_config=lambda: 'dram'), chunks, 'dram',
                                          SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram')),
                             'V3a itself still refuses bundles of four: this lever does not widen it')


def emulate_dense(pages, users, rows=8):
    tiles = (rows * HEAD + 31) // 32
    out = torch.zeros(users, tiles * 32, 256, dtype=torch.int16)
    for task, tile in pages.items():
        batch, head_tile, column_tile = task // (tiles * 8), (task // 8) % tiles, task % 8
        out[batch, head_tile * 32:(head_tile + 1) * 32, column_tile * 32:(column_tile + 1) * 32] = tile
    return out


class MaskTests(unittest.TestCase):
    WORDS = (0, 1, 7, 127, 128, 200, 239, 240, 247, 248, 255)

    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_each_entry_is_the_pinned_narrow_mask_of_the_single_launch_bit_for_bit(self):
        for word in self.WORDS:
            single = quad.narrow_mask_host(word, 8, 1, 0).view(torch.int16)
            built = bundle.mask_host([word])
            self.assertEqual(tuple(built.shape), (1, 1, 48, 256))
            self.assertTrue(torch.equal(built[0], single[0]), word)
        words = [3, 130, 247, 17, 0, 255, 128, 64]
        mask = bundle.mask_host(words)
        self.assertEqual(tuple(mask.shape), (8, 1, 48, 256))
        for user, word in enumerate(words):
            self.assertTrue(torch.equal(mask[user], quad.narrow_mask_host(word, 8, 1, 0).view(torch.int16)[0]), user)

    def test_the_kernel_emulation_writes_the_host_mask_on_every_page(self):
        words = [3, 130, 247, 17, 0, 255]
        positions = {0x100 + 0x40 * user: word for user, word in enumerate(words)}
        for cores in (110, 8, 96):
            tasks = bundle.mask_tasks(len(words))
            self.assertEqual(tasks, list(range(len(words) * 2 * 8)))
            per_core = multi.distribute(tasks, cores)
            capacity = max(len(core) for core in per_core)
            arguments = bundle.mask_arguments(0x900, sorted(positions), per_core, capacity)
            self.assertTrue(all(len(core) == 11 + capacity for core in arguments))
            self.assertTrue(all(core[1] == 8 for core in arguments), 'the entry\'s rows is a runtime argument')
            pages = multitests.emulate_mask(arguments, positions)
            self.assertEqual(sorted(pages), tasks)
            dense = emulate_dense(pages, len(words))
            self.assertTrue(torch.equal(dense[:, :48], bundle.mask_host(words)[:, 0]), cores)
            self.assertTrue(bool((dense[:, 48:] == -128).all()), 'the padded head rows are fully masked, as the single launch\'s are')

    def test_the_arguments_refuse_more_than_eight_users_and_pad_short_lists(self):
        with self.assertRaises(ValueError):
            bundle.mask_arguments(1, list(range(9)), [[0]], 1)
        arguments = bundle.mask_arguments(0x900, [0x10, 0x20], [[0, 5], [1]], 2)
        self.assertEqual(arguments[0], [0x900, 8, 2, 0x10, 0x20, 0, 0, 0, 0, 0, 0, 0, 5])
        self.assertEqual(arguments[1], [0x900, 8, 1, 0x10, 0x20, 0, 0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(bundle.mask_arguments(0x900, [0x10], [[0]], 1, rows=16)[0][1], 16)

    def test_the_kernel_takes_the_rows_as_a_runtime_argument_and_the_position_from_the_entrys_own_word(self):
        text = (HERE / multi.KERNEL_MASK).read_text(encoding='utf-8')
        self.assertIn('const uint32_t rows = get_arg_val<uint32_t>(1);', text)
        self.assertIn('const uint32_t head_tiles = (rows * QWEN_FOLD_HEAD_ROWS + 31) / 32;', text)
        self.assertIn('const uint32_t position = start + head / 6;', text)
        self.assertIn('head >= rows * QWEN_FOLD_HEAD_ROWS', text)


class GatherTests(unittest.TestCase):
    def test_the_gather_of_a_bundles_users_assembles_their_rows_and_cur_pos(self):
        for users in (2, 3, 4, 6):
            table_addresses = [0x1000 + 0x100 * user for user in range(users)]
            position_addresses = [0x2000 + 0x100 * user for user in range(users)]
            # an octo user's lent table has ONE row (a bundle of one group)
            tables = {address: {0: [user * 1000 + page for page in range(8)]} for user, address in enumerate(table_addresses)}
            tables[0x3000] = {}
            positions = {address: {0: [4000 + 256 * user + 255]} for user, address in enumerate(position_addresses)}
            positions[0x4000] = {}
            arguments = [multi.gather_arguments(task, users, 0x3000, 0x4000, table_addresses, position_addresses) for task in range(users + 1)]
            multitests.emulate_gather(arguments, tables, positions)
            self.assertEqual([tables[0x3000][user] for user in range(users)], [[user * 1000 + page for page in range(8)] for user in range(users)])
            self.assertEqual(positions[0x4000][0][:users], [4255 + 256 * user for user in range(users)])
            self.assertEqual(positions[0x4000][0][users:], [0] * (len(positions[0x4000][0]) - users))


# --- the launch builder on a fake ttnn -------------------------------------------------------------------------------------------

class BuilderTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ttnn = multitests.builder_ttnn([])

    def test_the_mask_program_at_eight_rows(self):
        positions = [multitests.FakeBuilt((8,), 0x1000 + 0x100 * user, 32) for user in range(4)]
        mask = multitests.FakeBuilt((4, 1, 48, 256), 0x5000, 2048, 'tile')
        with patch.dict(sys.modules, {'ttnn': self.ttnn}):
            program, io = bundle.build_mask_program(self.ttnn, MESH, positions, mask, 4)
        self.assertEqual(sorted(program), [((0, chip), (0, chip)) for chip in range(4)])
        self.assertEqual(list(io), [*positions, mask])
        for chip, kernel in enumerate(multitests.kernels_of(program).values()):
            self.assertEqual(kernel.kernel_source, str(HERE / multi.KERNEL_MASK))
            self.assertEqual(dict(kernel.defines), {'QWEN_FOLD_HEAD_ROWS': '6'})
            arguments = multitests.core_lists(kernel)
            self.assertEqual(len(arguments), 64, 'two row tiles x eight columns x four users on 64 of 110 cores')
            self.assertEqual(kernel.compile_time_args[-1], 1)
            self.assertEqual(sorted(task for core in arguments for task in core[11:11 + core[2]]), list(range(64)))
            for core in arguments:
                self.assertEqual(core[0], 0x5000 + chip * 0x10000000)
                self.assertEqual(core[1], 8)
                self.assertEqual(core[3:7], [0x1000 + 0x100 * user + chip * 0x10000000 for user in range(4)])

    def test_six_users_need_two_tasks_per_core_and_the_refusals_are_the_multi_builders(self):
        positions = [multitests.FakeBuilt((8,), 0x1000 + 0x100 * user, 32) for user in range(6)]
        mask = multitests.FakeBuilt((6, 1, 48, 256), 0x5000, 2048, 'tile')
        with patch.dict(sys.modules, {'ttnn': self.ttnn}):
            program, io = bundle.build_mask_program(self.ttnn, MESH, positions, mask, 6)
            kernel = list(multitests.kernels_of(program).values())[0]
            self.assertEqual(sorted(task for core in multitests.core_lists(kernel) for task in core[11:11 + core[2]]), list(range(96)))
            with self.assertRaisesRegex(ValueError, 'must not alias'):
                bundle.build_mask_program(self.ttnn, MESH, positions[:2], multitests.FakeBuilt((2, 1, 48, 256), 0x1000, 2048, 'tile'), 2)
            positions[1].layout = 'other'
            with self.assertRaisesRegex(ValueError, 'accessor layout'):
                bundle.build_mask_program(self.ttnn, MESH, positions[:2], multitests.FakeBuilt((2, 1, 48, 256), 0x5000, 2048, 'tile'), 2)


# --- the reader twin on the four-chip fake device ---------------------------------------------------------------------------------

class OctoRig(object):
    """The patches an octo bundle needs on the fake device: the mask and gather builders (named programs), the audit builder, the block fold launches, the pinned refresh and
    the served per-user fold, with the lists a test reads back."""

    def __init__(self, test, device):
        self.device, self.logs, self.refreshed, self.audits, self.fold_ins, self.fold_outs = device, [], [], [], [], []
        self.stack = ExitStack()
        test.addCleanup(self.stack.close)
        self.stack.enter_context(patch('octo_attn_bundle._log', side_effect=self.logs.append))
        self.stack.enter_context(patch('sdpa_multi_tp._log', side_effect=self.logs.append))
        self.stack.enter_context(patch('sdpa_long_tp._log', side_effect=self.logs.append))
        self.stack.enter_context(patch('octo_attn_bundle.build_mask_program', side_effect=lambda ttnn, mesh, positions, mask, users: (
            ('mask-program', users, tuple(mask.shape)), [*positions, mask])))
        self.stack.enter_context(patch('sdpa_multi_tp.build_gather_program', side_effect=lambda ttnn, mesh, tables, cur, table, position: (
            ('gather-program', len(tables), tuple(table.shape)), [*tables, *cur, table, position])))
        self.stack.enter_context(patch('sdpa_multi_tp.build_audit_program', side_effect=self.audit_program))
        self.stack.enter_context(patch('attention_block_fold_tp.fold_in', side_effect=self.block_fold_in))
        self.stack.enter_context(patch('attention_block_fold_tp.fold_out', side_effect=self.block_fold_out))
        self.stack.enter_context(patch('extent_attention_replay_tp.attention_mask_replay.execute', side_effect=lambda *a: self.refreshed.append(a)))
        self.stack.enter_context(patch('extent_attention_replay_tp.device_layout_dma', side_effect=self.served_fold))

    def audit_program(self, ttnn, mesh, served, built, counters, slot):
        self.audits.append((served, built, counters, slot))
        return ('audit-program', slot), [served, built, counters]

    def served_fold(self, mesh, source, rows, owned, *, inverse=False, offset=0):
        output = self.device.device_tensor((1, rows, 6, 256) if inverse else (1, 1, rows * 6, 256), 'bf16', 'tile', 'dram', name='fold')
        owned.append(output)
        return output

    def block_fold_in(self, mesh, query, chunks, owned):
        self.fold_ins.append(list(chunks))
        stacked = [self.device.device_tensor((1, chunk.batches, chunk.rows * 6, 256), 'bf16', 'tile', 'dram', name='stack') for chunk in chunks]
        owned.extend(stacked)
        return stacked

    def block_fold_out(self, mesh, results, chunks, memory_config, owned):
        self.fold_outs.append((list(results), list(chunks)))
        total = chunks[-1].first + chunks[-1].groups[-1][0] + chunks[-1].groups[-1][1]
        output = self.device.device_tensor((1, total, 6, 256), 'bf16', 'tile', memory_config, name='block-out')
        owned.append(output)
        return output


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.device = multitests.MultiDevice()
        self.harness = Harness(self)
        self.rig = OctoRig(self, self.device)

    def block(self, segments=SEGMENTS, rows=8, starts=None):
        users = len(segments)
        starts = starts or tuple(4200 + 100 * user for user in range(users))
        lent = [lend(self.device, rows, WIDTH) for user in range(users)]
        reader = folded.PackedExtentReplayReader(self.device, MESH, segments, WIDTH, [host_table(user + 1, WIDTH) for user in range(users)], storage=lent,
                                                 max_group_rows=8, starts=starts)
        return reader, lent

    def build(self, **values):
        with env(**values):
            return self.block()[0]

    def query(self):
        return (self.device.device_tensor((1, 64, 6, 256), 'bf16', 'tile', 'dram', name='query'),
                self.device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys'))

    def call(self, reader, **kwargs):
        query, keys = self.query()
        return reader(query, keys, keys, scale=0.0625, memory_config='dram', **kwargs)

    def lines(self, marker):
        return [line for line in self.rig.logs if line.startswith(marker)]

    # -- flag off -----------------------------------------------------------------------------------------------------

    def test_flag_off_the_module_is_never_imported_and_nothing_is_allocated_or_logged(self):
        sys.modules.pop('octo_attn_bundle', None)
        try:
            before = len(self.device.live)
            reader = self.build()
            self.assertNotIn('octo_attn_bundle', sys.modules, 'the twin asked its one environment question and went no further')
            self.assertIsNone(reader.multi)
            self.assertIsNone(folded.PackedExtentReplayReader.multi)
            self.assertEqual(self.rig.logs, [])
            self.assertEqual(self.device.generic, [])
            self.assertGreater(len(self.device.live) - before, 0)
        finally:
            sys.modules['octo_attn_bundle'] = bundle

    def test_flag_off_values_change_nothing(self):
        for value in ('', '0'):
            reader = self.build(**dict(OCTO_ON, **{FLAG: value, AUDIT: value}))
            self.assertIsNone(reader.multi)
        self.assertEqual(self.rig.logs, [])

    def test_flag_off_a_call_makes_eight_single_launches_and_books_the_pinned_bookkeeping(self):
        reader = self.build()
        with patch.object(folded.PackedExtentReplayReader, 'served_call', wraps=reader.served_call) as served:
            self.call(reader)
        self.assertEqual(served.call_count, 1)
        self.assertEqual(len(self.device.sdpa), 8)
        self.assertEqual((reader.calls, [r.calls for r in reader.readers]), (1, [1] * 8))
        self.assertEqual(len(self.rig.refreshed), 8, 'one mask program per user bundle, as served')

    def test_the_m3_block_is_untouched_with_the_flag_on(self):
        with env(**flags('4')):
            lent = [lend(self.device, 16, WIDTH) for user in range(4)]
            reader = folded.PackedExtentReplayReader(self.device, MESH, M3_SEGMENTS, WIDTH, [host_table(user + 1, WIDTH) for user in range(4)], storage=lent,
                                                     max_group_rows=8, starts=(4200,) * 4)
        self.assertIsNone(reader.multi)
        self.assertEqual({type(r) for r in reader.readers}, {quad.ExtentSegmentReader})
        self.assertEqual(self.lines(bundle.MARKER), [])

    # -- flag on ------------------------------------------------------------------------------------------------------

    def test_attach_allocates_each_launchs_stacked_buffers_before_any_call_and_logs_the_engaged_marker(self):
        reader = self.build(**flags('4'))
        block = reader.multi
        self.assertIsInstance(block, bundle.OctoBundle)
        self.assertEqual(block.groups, ((0, 1, 2, 3), (4, 5, 6, 7)))
        self.assertEqual(len(block.launches), 2)
        for launch in block.launches:
            self.assertEqual(tuple(launch['table'].shape), (4, WIDTH))
            self.assertEqual(tuple(launch['cur_pos'].shape), (4,))
            self.assertEqual(tuple(launch['mask'].shape), (4, 1, 48, 256))
            self.assertEqual((launch['table'].dtype, launch['cur_pos'].dtype, launch['mask'].dtype), ('int32', 'int32', 'bf16'))
            self.assertEqual((launch['table'].layout, launch['cur_pos'].layout, launch['mask'].layout), ('row_major', 'row_major', 'tile'))
        self.assertEqual(vars(block.config), dict(compute_with_storage_grid_size=(11, 10), exp_approx_mode=False, q_chunk_size=0x51DEC021, k_chunk_size=256))
        self.assertEqual(self.lines(bundle.ENGAGED),
                         ['%s users=8 launches=2 per_launch=4,4 flags=0x21 cores_per_entry=16 active_cores=64,64 grid=11x10 audit=0' % bundle.ENGAGED])
        unqualified = self.lines(bundle.UNQUALIFIED)
        self.assertEqual(len(unqualified), 1)
        self.assertIn('B > 1 entries of eight rows', unqualified[0])
        self.assertEqual([program for tensors, program in self.device.generic],
                         [('gather-program', 4, (4, WIDTH)), ('mask-program', 4, (4, 1, 48, 256))] * 2,
                         'one eager refresh at attach per launch: a kernel that does not build fails the attach')
        self.assertEqual(len({id(tensor) for tensor in block.owned}), len(block.owned))
        self.assertIs(reader.multi, block)

    def test_the_launch_split_follows_the_flags_value(self):
        for value, split in (('2', '2,2,2,2'), ('4', '4,4'), ('5', '4,4'), ('6', '4,4')):
            reader = self.build(**flags(value))
            self.assertIn('per_launch=%s ' % split, self.lines(bundle.ENGAGED)[-1])
            self.assertEqual(','.join(str(len(launch['users'])) for launch in reader.multi.launches), split)
        with self.assertRaisesRegex(ValueError, 'unequal sizes'):
            self.build(**flags('3'))

    def test_one_call_is_ceil_eight_over_n_launches_with_the_stacked_inputs_and_one_fold_pair(self):
        reader = self.build(**flags('4'))
        block = reader.multi
        self.device.generic.clear()
        output = self.call(reader)
        self.assertEqual(len(self.device.sdpa), 2)
        for options, launch in zip(self.device.sdpa, block.launches):
            self.assertIs(options['page_table_tensor'], launch['table'])
            self.assertIs(options['cur_pos_tensor'], launch['cur_pos'])
            self.assertIs(options['attn_mask'], launch['mask'])
            self.assertIs(options['program_config'], block.config)
            self.assertEqual((options['is_causal'], options['scale'], options['memory_config']), (False, 0.0625, 'dram'))
        stacks = [tensor for tensor in self.device.live if tensor.name == 'stack']
        self.assertEqual([tuple(tensor.shape) for tensor in stacks], [(1, 4, 48, 256)] * 2)
        self.assertEqual(tuple(output.shape), (1, 64, 6, 256))
        (chunks_in,), (fold_out,) = self.rig.fold_ins, self.rig.fold_outs
        self.assertEqual([(chunk.first, chunk.groups) for chunk in chunks_in], [(0, ((0, 8), (8, 8), (16, 8), (24, 8))), (32, ((0, 8), (8, 8), (16, 8), (24, 8)))])
        self.assertEqual(len(fold_out[0]), 2)
        self.assertEqual([chunk.first for chunk in fold_out[1]], [0, 32])
        self.assertEqual([program for tensors, program in self.device.generic],
                         [('gather-program', 4, (4, WIDTH)), ('mask-program', 4, (4, 1, 48, 256))] * 2, 'no scope: this call refreshed once')
        self.assertEqual(self.rig.refreshed, [], 'the per-user mask programs never run')
        self.assertEqual(len(self.lines(bundle.CALL_MARKER)), 1)
        self.assertEqual(self.lines(bundle.CALL_MARKER)[0], '%s users=8 launches=2 flags=0x21 cores_per_entry=16 audit=0' % bundle.CALL_MARKER)

    def test_sixteen_layers_in_one_forward_refresh_once_and_book_what_model_batch_counts(self):
        reader = self.build(**flags('4'))
        self.device.generic.clear()
        before = reader.refresh_calls
        with reader.shared_masks(16):
            for layer in range(16):
                self.call(reader)
        self.assertEqual(reader.refresh_calls - before, len(reader.metadata), 'model_batch: one refresh per bundle per forward (8)')
        self.assertEqual(len(reader.metadata), 8)
        self.assertEqual(len(self.device.generic), 4, 'gather and mask per launch, once for the forward')
        self.assertEqual(len(self.device.sdpa), 32, 'two launches a layer')
        self.assertEqual((reader.calls, [r.calls for r in reader.readers]), (16, [16] * 8))
        self.assertEqual(reader.multi.calls, 16)
        self.assertFalse(reader.failed)
        self.assertEqual(self.rig.refreshed, [])

    def test_the_bookkeeping_matches_the_served_reader_after_a_forward(self):
        served = self.build()
        with served.shared_masks(16):
            for layer in range(16):
                self.call(served)
        reader = self.build(**flags('4'))
        with reader.shared_masks(16):
            for layer in range(16):
                self.call(reader)
        self.assertEqual((reader.calls, [r.calls for r in reader.readers], reader.refresh_calls, reader.failed),
                         (served.calls, [r.calls for r in served.readers], served.refresh_calls, served.failed))

    def test_a_segment_rebound_after_attach_fails_the_scope_loudly_and_poisons(self):
        reader = self.build(**flags('4'))
        self.assertIsNone(reader.multi.rebound())
        self.assertIsNone(reader.rebound_reason())
        with reader.shared_masks(1):
            self.call(reader)
        segment = reader.readers[5]
        original = segment.cur_pos
        segment.cur_pos = [self.device.device_tensor((1,), 'int32', 'row_major', 'dram', name='other')]
        self.assertIn('cur_pos', reader.multi.rebound())
        self.assertIn('cur_pos', reader.rebound_reason())
        with self.assertRaisesRegex(RuntimeError, 'rebound after attach'):
            with reader.shared_masks(1):
                pass
        self.assertTrue(all(r.failed for r in reader.readers))
        segment.cur_pos = original

    def test_the_call_budget_is_enforced_both_ways_and_scopes_do_not_nest(self):
        reader = self.build(**flags('4'))
        with self.assertRaisesRegex(AssertionError, 'exact attention call budget'):
            with reader.shared_masks(16):
                for layer in range(15):
                    self.call(reader)
        self.assertTrue(reader.failed)
        reader = self.build(**flags('4'))
        with self.assertRaisesRegex(AssertionError, 'exceeded its attention call budget'):
            with reader.shared_masks(2):
                for layer in range(3):
                    self.call(reader)
        reader = self.build(**flags('4'))
        with reader.shared_masks(1):
            with self.assertRaisesRegex(RuntimeError, 'cannot nest'):
                with reader.shared_masks(1):
                    pass
            self.call(reader)
        for budget in (0, 17, '16'):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, 'one-to-16'):
                with reader.shared_masks(budget):
                    pass

    def test_a_query_the_launches_are_not_written_for_raises_and_poisons(self):
        reader = self.build(**flags('4'))
        keys = self.device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys')
        with self.assertRaisesRegex(ValueError, 'cannot serve this call'):
            reader(self.device.device_tensor((1, 32, 6, 256), 'bf16', 'tile', 'dram'), keys, keys, scale=1.0, memory_config='dram')
        with self.assertRaisesRegex(ValueError, 'cannot serve this call: query is not bf16 TILE'):
            reader(self.device.device_tensor((1, 64, 6, 256), 'int32', 'tile', 'dram'), keys, keys, scale=1.0, memory_config='dram')
        with self.assertRaisesRegex(ValueError, 'output memory config'):
            reader(self.device.device_tensor((1, 64, 6, 256), 'bf16', 'tile', 'dram'), keys, keys, scale=1.0, memory_config='sharded')
        self.assertEqual(self.device.sdpa, [])

    def test_a_failure_poisons_every_segment_reader_and_releases_what_the_call_made(self):
        reader = self.build(**flags('4'))
        released = []
        with patch('tp_addresses.release_owned', side_effect=lambda operations, values: released.extend(values)):
            with patch.object(self.device, 'attention', side_effect=RuntimeError('boom')):
                self.device.transformer = SimpleNamespace(paged_scaled_dot_product_attention_decode=self.device.attention)
                with self.assertRaisesRegex(RuntimeError, 'boom'):
                    self.call(reader)
        self.assertTrue(all(segment.failed for segment in reader.readers))
        self.assertTrue(any(value.name == 'stack' for value in released), 'the stacked queries are released')

    def test_the_output_is_not_released_and_the_intermediates_are(self):
        reader = self.build(**flags('4'))
        released = []
        with patch('tp_addresses.release_owned', side_effect=lambda operations, values: released.extend(values)):
            output = self.call(reader)
        self.assertEqual(sorted(value.name for value in released), ['sdpa', 'sdpa', 'stack', 'stack'], 'the two stacked queries and the two SDPA results')
        self.assertFalse(any(value is output for value in released))

    def test_close_releases_the_bundles_buffers_once(self):
        reader = self.build(**flags('4'))
        owned = list(reader.multi.owned)
        released = []
        with patch('tp_addresses.release_owned', side_effect=lambda operations, values: released.extend(values)):
            reader.close()
            reader.close()
        self.assertTrue(all(any(value is tensor for value in released) for tensor in owned))
        self.assertTrue(reader.multi.closed)

    # -- refusals -----------------------------------------------------------------------------------------------------

    def test_a_flag_the_process_cannot_serve_refuses_the_attach_by_name_and_closes_the_readers(self):
        closed = []
        original = folded.PackedExtentReplayReader.close
        with patch.object(folded.PackedExtentReplayReader, 'close', lambda self: (closed.append(1), original(self))[1]):
            with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_OCTO=live or alternate'):
                self.build(**{FLAG: '4'})
            with self.assertRaisesRegex(ValueError, 'users per launch'):
                self.build(**flags('9'))
        self.assertTrue(closed)

    def test_the_audit_flag_without_the_bundle_flag_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'it needs %s' % FLAG):
            self.build(**dict(OCTO_ON, **{AUDIT: '1'}))
        with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
            self.build(**flags('4', audit='yes'))

    def test_the_sdpa_flag_beside_the_bundle_is_refused(self):
        # the process-level question is admission_problems' (tested above); at the reader the SDPA flag's own attach, which runs first, refuses the eight-row block too
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_TP4_SDPA=multi'):
            self.build(**flags('4', QWEN_FAST_TP4_SDPA='multi'))
        self.assertIn('two ways to bundle', ' '.join(bundle.admission_problems(dict(flags('4'), QWEN_FAST_TP='4', QWEN_FAST_TP4_SDPA='multi'))))

    def test_a_multi_launch_already_attached_is_refused(self):
        with env(**flags('4')):
            lent = [lend(self.device, 8, WIDTH) for user in range(8)]
            with patch.object(folded.PackedExtentReplayReader, 'multi', SimpleNamespace(close=lambda: None)):
                with self.assertRaisesRegex(ValueError, 'already runs a multi launch'):
                    folded.PackedExtentReplayReader(self.device, MESH, SEGMENTS, WIDTH, [host_table(user + 1, WIDTH) for user in range(8)], storage=lent,
                                                    max_group_rows=8, starts=(4200,) * 8)

    def test_a_group_the_factory_would_give_other_than_sixteen_cores_is_refused_at_attach(self):
        closed = []
        original = folded.PackedExtentReplayReader.close
        with patch.object(folded.PackedExtentReplayReader, 'close', lambda self: (closed.append(1), original(self))[1]):
            # the admission would already refuse the uneven split; the attach's own wall is the factory's cores per entry
            with patch('octo_attn_bundle.launch_groups', side_effect=lambda users, size: (tuple(range(7)), (7,))), \
                    patch('octo_attn_bundle.admission_problems', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'QWEN_FAST_OCTO_ATTN_BUNDLE: .*15 cores per entry'):
                    self.build(**flags('4'))
        self.assertTrue(closed)

    def test_a_reader_not_at_flags_0x21_is_refused(self):
        reader = self.build()
        reader.readers[3].sdpa_modes_applied = (0x23,)
        with self.assertRaisesRegex(ValueError, 'qualified at flags 0x21 only, the octo reader runs \\[\'0x23\'\\]'):
            bundle.apply(reader, dict(flags('4'), QWEN_FAST_TP='4'))
        self.assertIsNone(reader.multi)
        reader.readers[3].sdpa_modes_applied = (0x21,)
        self.assertIsNotNone(bundle.apply(reader, dict(flags('4'), QWEN_FAST_TP='4')))

    def test_apply_is_a_no_op_for_the_m3_block_and_for_the_flag_off_process(self):
        reader = self.build()
        self.assertIsNone(bundle.apply(reader, {}))
        self.assertIsNone(bundle.apply(reader, {FLAG: '0'}))
        self.assertIsNone(reader.multi)
        with env(**flags('4')):
            lent = [lend(self.device, 16, WIDTH) for user in range(4)]
            m3 = folded.PackedExtentReplayReader(self.device, MESH, M3_SEGMENTS, WIDTH, [host_table(user + 1, WIDTH) for user in range(4)], storage=lent,
                                                 max_group_rows=8, starts=(4200,) * 4)
        self.assertFalse(bundle.reader_is_octo(m3))
        self.assertTrue(bundle.reader_is_octo(reader))
        self.assertIsNone(bundle.apply(m3, dict(flags('4'), QWEN_FAST_TP='4')))

    def test_the_extent_audit_needs_the_bundle_audit(self):
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_EXTENT_AUDIT=1 reads back'):
            self.build(**flags('4', QWEN_FAST_EXTENT_AUDIT='1'))
        self.build(**flags('4', audit='1', QWEN_FAST_EXTENT_AUDIT='1'))

    def test_a_kernel_that_does_not_run_at_attach_fails_the_attach_and_releases_what_was_allocated(self):
        released = []
        with patch('tp_addresses.release_owned', side_effect=lambda operations, values: released.extend(values)):
            with patch.object(self.device, 'generic_op', side_effect=RuntimeError('kernel did not build')):
                with self.assertRaisesRegex(RuntimeError, 'kernel did not build'):
                    self.build(**flags('4'))
        self.assertTrue(any(value.name == 'upload' and tuple(value.shape) == (4, WIDTH) for value in released), 'the stacked table is released with the rest')
        self.assertEqual(self.lines(bundle.ENGAGED), [])

    # -- the audit ----------------------------------------------------------------------------------------------------

    def test_under_the_audit_both_paths_run_and_the_compare_is_launched_per_layer(self):
        reader = self.build(**flags('4', audit='1'))
        self.assertTrue(reader.multi.audit)
        self.assertEqual(tuple(reader.multi.counters.shape), (16 * 64, 16))
        self.assertIn('audit=1', self.lines(bundle.ENGAGED)[0])
        self.device.sdpa.clear()
        before = reader.refresh_calls
        with reader.shared_masks(16):
            for layer in range(16):
                output = self.call(reader)
        self.assertEqual(len(self.device.sdpa), 16 * (8 + 2), 'eight single launches and two bundled ones per layer')
        self.assertEqual(reader.refresh_calls - before, len(reader.metadata), 'booked once, by the served path')
        self.assertEqual(len(self.rig.audits), 16)
        self.assertEqual([entry[3] for entry in self.rig.audits], list(range(16)), 'one counter slot per layer')
        for served_output, built, counters, slot in self.rig.audits:
            self.assertIs(counters, reader.multi.counters)
            self.assertEqual((tuple(served_output.shape), tuple(built.shape)), ((1, 64, 6, 256), (1, 64, 6, 256)))
            self.assertIsNot(served_output, built)
        self.assertEqual((reader.calls, [r.calls for r in reader.readers]), (16, [16] * 8), 'booked once: by the served path')
        self.assertEqual(len(self.rig.refreshed), 8, 'the per-user masks are refreshed once for the single launches')
        self.assertIs(self.rig.audits[-1][1], output, 'the bundled launches\' output is the one returned')

    def test_the_audits_served_outputs_are_released_and_the_bundled_output_is_not(self):
        reader = self.build(**flags('4', audit='1'))
        released = []
        with patch('tp_addresses.release_owned', side_effect=lambda operations, values: released.extend(values)):
            output = self.call(reader)
        self.assertTrue(any(value is self.rig.audits[0][0] for value in released))
        self.assertFalse(any(value is output for value in released))

    def test_an_audit_round_logs_exact_when_every_layer_is_equal_and_live(self):
        reader = self.build(**flags('4', audit='1'))
        with reader.shared_masks(16):
            for layer in range(16):
                self.call(reader)
        self.device.counters_value = multitests.AuditMapTests.counters(None, 16)
        self.assertGreater(reader.sdpa_audit_round(1), 0)
        lines = self.lines(bundle.AUDIT_MARKER)
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r'^\[OCTO-ATTN-BUNDLE\] audit 1 exact=True users=8 launches=2 layers=16 chips=4 words=\d+ live=\d+$')
        reader.sdpa_audit_round(2)
        self.assertEqual(len(self.lines(bundle.AUDIT_MARKER)), 2)

    def test_a_differing_word_logs_the_mismatch_and_raises(self):
        reader = self.build(**flags('4', audit='1'))
        with reader.shared_masks(16):
            for layer in range(16):
                self.call(reader)
        self.device.counters_value = multitests.AuditMapTests.counters(None, 16, differing={(3, 9): 2})
        with self.assertRaisesRegex(AssertionError, r'\[OCTO-ATTN-BUNDLE\] audit MISMATCH round=7 users=8 launches=2'):
            reader.sdpa_audit_round(7)
        self.assertTrue(any(bundle.AUDIT_MISMATCH in line and 'layer 3' in line for line in self.rig.logs))
        self.assertFalse(any('exact=True' in line for line in self.rig.logs))

    def test_an_audit_that_compared_nothing_fails(self):
        reader = self.build(**flags('4', audit='1'))
        self.device.counters_value = multitests.AuditMapTests.counters(None, 0)
        with self.assertRaisesRegex(AssertionError, 'no attention layer ran the multi launch'):
            reader.sdpa_audit_round(1)

    def test_the_audit_round_is_inert_without_the_audit(self):
        reader = self.build(**flags('4'))
        self.assertEqual(reader.sdpa_audit_round(1), 0)
        self.assertEqual(self.lines(bundle.AUDIT_MARKER), [])


# --- the smoke rule ------------------------------------------------------------------------------------------------------------------

class SmokeTests(unittest.TestCase):
    ENGAGED = '%s users=8 launches=2 per_launch=4,4 flags=0x21 cores_per_entry=16 active_cores=64,64 grid=11x10 audit=0' % bundle.ENGAGED
    CALL = '%s users=8 launches=2 flags=0x21 cores_per_entry=16 audit=0' % bundle.CALL_MARKER
    UNQUALIFIED = '%s: %s' % (bundle.UNQUALIFIED, bundle.UNQUALIFIED_TEXT)
    AUDIT = '%s 3 exact=True users=8 launches=2 layers=16 chips=4 words=100 live=50' % bundle.AUDIT_MARKER
    ON = dict(flags('4'), QWEN_FAST_TP='4')

    def log(self, *lines):
        return '\n'.join('(EngineCore pid=67) 2026-10-10 02:47:34.407 | INFO | x:y:1 - ' + line for line in lines)

    def test_the_logs_the_module_writes_pass_the_rule(self):
        device = multitests.MultiDevice()
        harness = Harness(self)
        rig = OctoRig(self, device)
        lent = [lend(device, 8, WIDTH) for user in range(8)]
        with env(**flags('4', audit='1')):
            reader = folded.PackedExtentReplayReader(device, MESH, SEGMENTS, WIDTH, [host_table(user + 1, WIDTH) for user in range(8)], storage=lent,
                                                     max_group_rows=8, starts=(4200,) * 8)
        keys = device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys')
        with reader.shared_masks(16):
            for layer in range(16):
                reader(device.device_tensor((1, 64, 6, 256), 'bf16', 'tile', 'dram'), keys, keys, scale=1.0, memory_config='dram')
        device.counters_value = multitests.AuditMapTests.counters(None, 16)
        reader.sdpa_audit_round(1)
        text = self.log(*rig.logs)
        self.assertEqual(bundle.bundle_problems(text, dict(flags('4', audit='1'))), [])
        self.assertTrue(bundle.bundle_problems(text, flags('4')), 'an audit line in a profile without the audit flag')

    def test_a_good_log_passes_and_each_missing_piece_is_named(self):
        text = self.log(self.ENGAGED, self.UNQUALIFIED, self.CALL)
        self.assertEqual(bundle.bundle_problems(text, self.ON), [])
        self.assertIn('never attached', ' '.join(bundle.bundle_problems(self.log(self.UNQUALIFIED, self.CALL), self.ON)))
        self.assertIn('never called', ' '.join(bundle.bundle_problems(self.log(self.ENGAGED, self.UNQUALIFIED), self.ON)))
        self.assertIn('UNQUALIFIED', ' '.join(bundle.bundle_problems(self.log(self.ENGAGED, self.CALL), self.ON)))
        self.assertIn('attached more than once', ' '.join(bundle.bundle_problems(self.log(self.ENGAGED, self.ENGAGED, self.UNQUALIFIED, self.CALL), self.ON)))

    def test_the_engaged_line_must_say_the_split_the_flags_value_gives(self):
        for value, split, launches in (('4', '4,4', 2), ('6', '4,4', 2), ('2', '2,2,2,2', 4)):
            line = '%s users=8 launches=%d per_launch=%s flags=0x21 cores_per_entry=16 active_cores=64,64 grid=11x10 audit=0' % (bundle.ENGAGED, launches, split)
            self.assertEqual(bundle.bundle_problems(self.log(line, self.UNQUALIFIED, self.CALL), dict(flags(value), QWEN_FAST_TP='4')), [], value)
            self.assertEqual(bool(bundle.bundle_problems(self.log(line, self.UNQUALIFIED, self.CALL), self.ON)), value == '2', value)
        wrong = self.ENGAGED.replace('flags=0x21', 'flags=0x23').replace('cores_per_entry=16', 'cores_per_entry=15')
        problems = bundle.bundle_problems(self.log(wrong, self.UNQUALIFIED, self.CALL), self.ON)
        self.assertEqual(len(problems), 2)
        self.assertIn('flags=0x21', problems[0])

    def test_the_audit_arm_needs_a_passing_audit_line_and_no_mismatch(self):
        on = dict(flags('4', audit='1'))
        base = self.ENGAGED.replace('audit=0', 'audit=1')
        text = self.log(base, self.UNQUALIFIED, self.CALL.replace('audit=0', 'audit=1'), self.AUDIT)
        self.assertEqual(bundle.bundle_problems(text, on), [])
        self.assertIn('nothing was compared', ' '.join(bundle.bundle_problems(self.log(base, self.UNQUALIFIED, self.CALL.replace('audit=0', 'audit=1')), on)))
        mismatch = '%s round=7 users=8 launches=2 differing=2 layer 3: 2 of 9 words differ' % bundle.AUDIT_MISMATCH
        problems = bundle.bundle_problems(self.log(base, self.UNQUALIFIED, self.CALL.replace('audit=0', 'audit=1'), self.AUDIT, mismatch), on)
        self.assertTrue(any('found a difference' in problem for problem in problems))
        failed = self.AUDIT.replace('exact=True', 'exact=False')
        self.assertTrue(any('found a difference' in problem for problem in bundle.bundle_problems(self.log(base, self.UNQUALIFIED, self.CALL, failed), on)))

    def test_a_profile_without_the_flag_must_log_no_marker_line(self):
        self.assertEqual(bundle.bundle_problems(self.log('[OCTO] round=1 shape=octo'), {}), [])
        self.assertEqual(bundle.bundle_problems('', {}), [])
        problems = bundle.bundle_problems(self.log(self.ENGAGED), {})
        self.assertEqual(len(problems), 1)
        self.assertIn('does not set %s' % FLAG, problems[0])
        self.assertEqual(bundle.bundle_problems(self.log(self.ENGAGED), {FLAG: '0'})[0][:len(bundle.MARKER)], bundle.MARKER)

    def test_malformed_and_unpaired_flags_are_reported(self):
        self.assertIn('malformed', bundle.bundle_problems('', {FLAG: '9'})[0])
        self.assertIn('without %s' % FLAG, bundle.bundle_problems('', {AUDIT: '1'})[0])


# --- the shipment ---------------------------------------------------------------------------------------------------------------

class ShipmentTests(unittest.TestCase):
    def read(self, path):
        return (ROOT / path).read_text(encoding='utf-8')

    def test_the_reused_files_exist_and_ship_in_the_overlay(self):
        overlay = self.read('docker/qwen-c2-overlay.txt').splitlines()
        for name in bundle.REUSED_FILES:
            self.assertTrue((HERE / name).is_file(), name)
        for name in multi.RUNTIME_FILES:
            self.assertIn('scripts/ci/%s' % name, overlay, name)
        self.assertEqual(set(bundle.REUSED_FILES) - {'attention_block_fold_tp.py', 'attention_block_fold_tp.cpp'}, set(multi.RUNTIME_FILES))
        self.assertIn('attention_block_fold_tp.py', self.read('docker/qwen-fast-serving.Dockerfile'))

    def test_the_module_is_stdlib_at_import_and_py37_clean(self):
        text = (HERE / 'octo_attn_bundle.py').read_text(encoding='utf-8')
        self.assertNotRegex(text, r'(?m)^import (torch|ttnn|numpy)')
        self.assertNotRegex(text, r'(?m)^(import|from) (sdpa_long_tp|serving_octo|c2_smoke_check)')
        self.assertNotIn(':=', text)
        self.assertNotIn('strict=True', text)
        self.assertNotRegex(text, r'\bf"|\bf\'')

    def test_the_twin_imports_the_module_only_after_asking_the_flags(self):
        text = (HERE / 'extent_attention_fold_tp.py').read_text(encoding='utf-8')
        self.assertNotRegex(text, r'(?m)^(import|from) octo_attn_bundle')
        self.assertLess(text.index('if octo_flags_requested():'), text.index('import octo_attn_bundle'))
        self.assertLess(text.index('sdpa_long_tp.apply(self)'), text.index('octo_attn_bundle.apply(self)'))

    def test_the_new_files_are_lf(self):
        for name in ('octo_attn_bundle.py', 'test_octo_attn_bundle.py', 'extent_attention_fold_tp.py'):
            self.assertNotIn(b'\r', (HERE / name).read_bytes(), name)
        doc = ROOT / 'docs' / 'tp4-octo-attn-bundle.md'
        if doc.is_file():
            self.assertNotIn(b'\r', doc.read_bytes())

    def test_the_pinned_reader_twin_is_untouched_and_still_the_base(self):
        import hashlib
        import json

        evidence = json.loads((HERE / 'packed_any_evidence_tp4.json').read_text())
        digest = hashlib.sha256((HERE / 'extent_attention_replay_tp.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        self.assertEqual(digest, evidence['sources']['extent_attention_replay_tp.py'])
        self.assertEqual(folded.PackedExtentReplayReader.__mro__[1].__module__, 'extent_attention_replay_tp')
        self.assertNotIn('multi', quad.PackedExtentReplayReader.__dict__)


if __name__ == '__main__':
    unittest.main()
