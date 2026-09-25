"""Verify-trace T2: the flag frame, the #2 host guard, the #1 audit, and the wiring into the packed
block (packed_verifier), the step (serving_packed_step) and the fixture (model_batch).

  flag       QWEN_FAST_VERIFY_T2 exactly '1'; the skip list and the kv-rows knob are read only
             while it is on and refuse what they do not name; flag off, nothing new is imported,
             the warm fixture is today's and no line is logged.
  guard      kv_tile_rows is the ordered kernel's own addressing (pt[pos / 64], (pos % 64) / 32);
             kv_conflict names the first (page, tile row) two users would write; the served
             all-page-0 placeholders conflict, which is why the warm forward runs one chain.
  audit      every GDN layer on round 1, then two per round in rotation; every user, slot and
             chip, as int16 bits.
  block      the served placeholders, the warm fixture on one chain and the captured one on
             per-user chains under #2; the marker counts only the captured forward;
             stage_packed refuses shared tile rows before any copy; the audit runs every round.
  step       proposal_rows drafts a round with shared tile rows at the engines' own widths, so
             the exact sequential step serves it (with the real trimmed widths 1, 2, 4); a
             conflict first seen at the step refuses the round and writes nothing.
  fixture    cache_writer builds the chained writer only when asked, falls back on Unsupported
             and counts it; the warm fixture's single span; the audit windows are released with
             the records, and flag off nothing audit-related runs.
"""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import packed_verifier
import verify_trace_t2 as t2
from test_packed_verifier import FakeModelBatch, FourUserFixture, entry, request

HERE = Path(__file__).parent
ON = {'QWEN_FAST_VERIFY_T2': '1'}


class FlagTests(unittest.TestCase):
    def test_the_flag_is_on_only_for_exactly_one(self):
        for environ, expected in (({}, False), ({'QWEN_FAST_VERIFY_T2': '0'}, False),
                                  ({'QWEN_FAST_VERIFY_T2': 'true'}, False), (ON, True)):
            with patch.dict('os.environ', environ, clear=True):
                self.assertIs(t2.enabled(), expected)

    def test_the_audit_needs_the_flag(self):
        for environ, expected in (({'QWEN_FAST_VERIFY_T2_AUDIT': '1'}, False),
                                  (dict(ON, QWEN_FAST_VERIFY_T2_AUDIT='1'), True), (ON, False)):
            with patch.dict('os.environ', environ, clear=True):
                self.assertIs(t2.audit_enabled(), expected)

    def test_the_skip_list_is_read_only_while_the_flag_is_on_and_names_only_cuts(self):
        with patch.dict('os.environ', dict(ON, QWEN_FAST_VERIFY_T2_SKIP=' windows '), clear=True):
            self.assertEqual(t2.skipped(), {'windows'})
            self.assertEqual([name for name in t2.CUTS if t2.cut(name)], ['kv_chains'])
        with patch.dict('os.environ', ON, clear=True):
            self.assertTrue(all(t2.cut(name) for name in t2.CUTS))
        with patch.dict('os.environ', {'QWEN_FAST_VERIFY_T2_SKIP': 'bogus'}, clear=True):
            self.assertEqual(t2.skipped(), frozenset())
            self.assertFalse(any(t2.cut(name) for name in t2.CUTS))
        with patch.dict('os.environ', dict(ON, QWEN_FAST_VERIFY_T2_SKIP='windows,chains'), clear=True):
            with self.assertRaisesRegex(ValueError, 'names no cut: chains'):
                t2.skipped()
        with self.assertRaises(ValueError):
            t2.cut('window')

    def test_the_kv_rows_knob(self):
        for environ, expected in (({}, 64), ({'QWEN_FAST_VERIFY_T2_KV_ROWS': '64'}, 64),
                                  ({'QWEN_FAST_VERIFY_T2_KV_ROWS': '32'}, 32)):
            with patch.dict('os.environ', environ, clear=True):
                self.assertEqual(t2.kv_rows(), expected)
        for value in ('16', '32 ', '', 'sixty-four'):
            with patch.dict('os.environ', {'QWEN_FAST_VERIFY_T2_KV_ROWS': value}, clear=True), \
                    self.assertRaises(ValueError):
                t2.kv_rows()

    def test_counts_are_taken_once_and_the_marker_keeps_its_field_order(self):
        t2.take()
        t2.note('windows')
        t2.note('windows', 47)
        self.assertEqual(t2.take(), {'windows': 48})
        self.assertEqual(t2.take(), {})
        self.assertEqual(t2.engaged_line('packed_verify', windows=48, windows_fallback=0, kv_chains=32, kv_fallback=0,
                                         kv_rows=64, warm_chain='single', audit=0),
                         '[PINDIAG] verify t2 engaged site=packed_verify windows=48 windows_fallback=0 kv_chains=32 '
                         'kv_fallback=0 kv_rows=64 warm_chain=single audit=0')

    def test_a_fallback_is_logged_once_per_site(self):
        t2._LOGGED.clear()
        logged = []
        with patch.object(t2, 'log_line', side_effect=logged.append):
            self.assertTrue(t2.fell_back('windows', 'a'))
            self.assertFalse(t2.fell_back('windows', 'b'))
            self.assertTrue(t2.fell_back('kv_chains', 'c'))
        self.assertEqual(logged, ['[PINDIAG] verify t2 fell back site=windows reason=a',
                                  '[PINDIAG] verify t2 fell back site=kv_chains reason=c'])
        t2._LOGGED.clear()

    def test_the_module_imports_only_the_standard_library_at_import(self):
        tree = ast.parse((HERE / 'verify_trace_t2.py').read_text(encoding='utf-8'))
        imported = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
        imported += [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
        self.assertEqual(imported, ['os'])


class ShippingTests(unittest.TestCase):
    def test_every_runtime_file_is_in_both_image_copy_lists(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        docker, context = dockerfile_modules(dockerfile_text()), context_modules()
        for name in t2.RUNTIME_FILES:
            with self.subTest(file=name):
                self.assertTrue((HERE / name).is_file())
                self.assertIn(name, docker)
                self.assertIn(name, context)

    def test_every_runtime_file_and_t2_module_is_lf(self):
        """The image ships what git normalises, but card M mounts the checkout's bytes: a CRLF
        file there would record a sha256 that is not the shipped file's."""
        names = list(t2.RUNTIME_FILES) + ['test_verify_trace_t2.py', 'test_gdn_conv_windows_packed.py',
                                          'test_packed_ordered_cache.py', 'gdn_user_batch_conv.py', 'model_batch.py',
                                          'packed_verifier.py', 'serving_packed_step.py', 'lever_n_m3native_gate.py']
        harness = HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'verify_t2'
        paths = [HERE / name for name in names] + sorted(harness.glob('*.py')) + sorted(harness.glob('*.sh')) \
            + sorted(harness.glob('*.cpp'))
        for path in paths:
            with self.subTest(file=path.name):
                self.assertNotIn(b'\r', path.read_bytes())

    def test_the_table_is_the_union_of_the_ops_own_tables(self):
        import gdn_conv_windows_packed

        self.assertLessEqual(set(gdn_conv_windows_packed.RUNTIME_FILES), set(t2.RUNTIME_FILES))
        self.assertIn('packed_ordered_cache.py', t2.RUNTIME_FILES)
        self.assertIn('verify_trace_t2.py', t2.RUNTIME_FILES)

    def test_every_importer_reaches_the_image(self):
        from test_serving_image_copy_closure import copied_modules, dockerfile_text

        shipped = copied_modules(dockerfile_text())
        for name in ('gdn_user_batch_conv.py', 'model_batch.py', 'packed_verifier.py', 'serving_packed_step.py',
                     'ordered_cache.py', 'packed_cache_writer.py', 'verify_trace_t1.py', 'gdn_user_batch.py'):
            with self.subTest(module=name):
                self.assertIn(name, shipped)
                source = (HERE / name).read_text(encoding='utf-8')
                if name in ('gdn_user_batch_conv.py', 'model_batch.py', 'packed_verifier.py', 'serving_packed_step.py'):
                    self.assertIn('verify_trace_t2', source)

    def test_no_pinned_or_frozen_source_is_edited(self):
        """gdn_conv_windows.*, gdn_multitoken_conv.py and ordered_cache.py are pinned or frozen
        evidence (the serving attach path hashes the first three); tp_common.py is never touched.
        None of them knows T2: every T2 branch lives in a caller."""
        for name in ('gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'gdn_multitoken_conv.py', 'ordered_cache.py',
                     'attention_batch.py', 'packed_cache_writer.py'):
            with self.subTest(module=name):
                source = (HERE / name).read_text(encoding='utf-8')
                self.assertNotIn('verify_trace_t2', source)
                self.assertNotIn('VERIFY_T2', source)


# ---------------------------------------------------------------------------------------------
# #2: the host guard.
# ---------------------------------------------------------------------------------------------

class GuardTests(unittest.TestCase):
    TABLE = list(range(100, 100 + 2052))

    def test_the_kernels_own_addressing(self):
        table = [7, 11, 13]
        self.assertEqual(t2.kv_tile_rows([0, 31], table), {(7, 0)})
        self.assertEqual(t2.kv_tile_rows([31, 32], table), {(7, 0), (7, 1)})
        self.assertEqual(t2.kv_tile_rows([63, 64], table), {(7, 1), (11, 0)})
        self.assertEqual(t2.kv_tile_rows(range(120, 136), table), {(11, 1), (13, 0)})

    def test_the_last_table_entries_at_131k(self):
        positions = range(131072 - 8, 131072 + 8)   # entries 2047 and 2048
        self.assertEqual(t2.kv_tile_rows(positions, self.TABLE), {(100 + 2047, 1), (100 + 2048, 0)})
        self.assertEqual(t2.kv_tile_rows([131327], self.TABLE), {(100 + 2051, 1)})

    def test_disjoint_users_and_one_page_at_different_tile_rows_pass(self):
        users = [(range(4100, 4116), [5] * 80), (range(4200, 4216), [6] * 80),
                 (range(4150, 4166), [7] * 80), (range(4300, 4316), [8] * 80)]
        self.assertIsNone(t2.kv_conflict(users))
        # users 0 and 1 on one physical page, tile rows 0 and 1 of it
        shared_page = [(range(64, 80), [0, 9]), (range(96, 112), [0, 9])]
        self.assertIsNone(t2.kv_conflict(shared_page))

    def test_a_shared_tile_row_names_the_users_the_page_and_the_row(self):
        users = [(range(4100, 4116), [5] * 80), (range(4200, 4216), [6] * 80), (range(4106, 4122), [5] * 80)]
        self.assertEqual(t2.kv_conflict(users), dict(users=(0, 2), page=5, tile_row=0))
        self.assertEqual(t2.kv_conflict_reason(t2.kv_conflict(users)),
                         'verify t2 kv tile rows shared: users 0,2 page 5 tile row 0')

    def test_the_served_placeholders_conflict_so_the_warm_forward_runs_one_chain(self):
        capture = 32768
        served = [(range(capture, capture + 16), torch.zeros(2052, dtype=torch.int32)) for user in range(4)]
        self.assertEqual(t2.kv_conflict(served), dict(users=(0, 1), page=0, tile_row=0))
        # one chain over the whole block is one "user": nothing to race
        self.assertIsNone(t2.kv_conflict([(range(capture, capture + 64), torch.zeros(2052, dtype=torch.int32))]))

    def test_block_users_reads_the_staged_rows_segment_by_segment(self):
        positions = torch.cat([torch.arange(100 * u, 100 * u + 16, dtype=torch.int32) for u in range(4)])
        pages = torch.cat([torch.full((16, 8), u + 3, dtype=torch.int32) for u in range(4)])
        users = t2.block_users(positions, pages, 16, 4)
        self.assertEqual([user_positions[0] for user_positions, table in users], [0, 100, 200, 300])
        self.assertEqual([int(table[0]) for user_positions, table in users], [3, 4, 5, 6])


# ---------------------------------------------------------------------------------------------
# #1: the audit.
# ---------------------------------------------------------------------------------------------

class Shard:
    def __init__(self, value):
        self.value = value


def chip_windows(values):
    return SimpleNamespace(shards=[Shard(value) for value in values])


AUDIT_OPERATIONS = SimpleNamespace(get_device_tensors=lambda tensor: tensor.shards, to_torch=lambda shard: shard.value)


def record(users=4, differ=None):
    """One retained layer record with packed and audit windows per user; `differ` = (user, slot,
    chip) flips one bit there (the -0 of a +0)."""
    pieces = []
    for user in range(users):
        packed, served = [], []
        for slot in range(4):
            values = [torch.full((16, 64), float(user * 4 + slot), dtype=torch.bfloat16) for chip in range(2)]
            copies = [value.clone() for value in values]
            if differ == (user, slot, 1):
                copies[1][3, 5] = -0.0 if float(values[1][3, 5]) == 0.0 else -values[1][3, 5]
            packed.append(chip_windows(values))
            served.append(chip_windows(copies))
        pieces.append(dict(packed_conv_states=packed, audit_windows=served, states=None))
    return (None, dict(segment_results=tuple(pieces)), None)


class AuditTests(unittest.TestCase):
    def setUp(self):
        t2._AUDIT.update(rounds=0)

    def test_round_one_compares_every_layer_then_two_rotate(self):
        self.assertEqual(t2.audit_layers(1), tuple(range(48)))
        self.assertEqual([t2.audit_layers(n) for n in (2, 3, 25, 26, 50)], [(0, 1), (2, 3), (46, 47), (0, 1), (0, 1)])
        with self.assertRaises(ValueError):
            t2.audit_layers(0)

    def test_a_short_arm_still_compares_every_layer_and_rotates_through_them_again(self):
        # a 4 x 32k 256-token arm runs about 38 rounds (6.8 tokens per user per round); a
        # fully-accepted one 16
        for rounds in (38, 16, 2):
            with self.subTest(rounds=rounds):
                seen = {layer for number in range(1, rounds + 1) for layer in t2.audit_layers(number)}
                self.assertEqual(seen, set(range(48)))
        again = {layer for number in range(2, 26) for layer in t2.audit_layers(number)}
        self.assertEqual(again, set(range(48)), 'every layer again within 24 rounds after the first')

    def test_an_exact_round_logs_its_line(self):
        records = [record() for layer in range(48)]
        logged = []
        with patch.object(t2, 'log_line', side_effect=logged.append):
            self.assertEqual(t2.audit_round(AUDIT_OPERATIONS, records, 1), 16 * 48)
            self.assertEqual(t2.audit_round(AUDIT_OPERATIONS, records, 27), 32)
        self.assertEqual(logged, ['[PINDIAG] verify t2 audit 1 exact=True layers=0-47 windows=768',
                                  '[PINDIAG] verify t2 audit 2 exact=True layers=2,3 windows=32'])

    def test_a_signed_zero_is_a_mismatch_and_raises(self):
        records = [record() for layer in range(48)]
        records[4] = record(differ=(0, 2, 1))
        for number, layers in ((1, 'layers=0-47'), (4, 'layers=4,5')):
            logged = []
            with self.subTest(round=number), patch.object(t2, 'log_line', side_effect=logged.append), \
                    self.assertRaises(AssertionError):
                t2.audit_round(AUDIT_OPERATIONS, records, number)
            self.assertTrue(logged[0].startswith('[PINDIAG] verify t2 audit mismatch round=%d %s layer 4 user 0 slot 2 '
                                                 'chip 1: 1' % (number, layers)), logged)
        # a round that does not reach layer 4 passes
        with patch.object(t2, 'log_line'):
            self.assertEqual(t2.audit_round(AUDIT_OPERATIONS, records, 2), 32)

    def test_missing_audit_windows_are_a_mismatch(self):
        broken = record()
        broken[1]['segment_results'][3].pop('audit_windows')
        with patch.object(t2, 'log_line'), self.assertRaisesRegex(AssertionError, 'user 3 holds 4 packed and 0 served'):
            t2.audit_round(AUDIT_OPERATIONS, [broken] * 48, 1)
        empty = (None, dict(segment_results=()), None)
        with patch.object(t2, 'log_line'), self.assertRaisesRegex(AssertionError, 'layer 0 user 0 holds 0 packed and 0 served'):
            t2.audit_round(AUDIT_OPERATIONS, [empty] + [record()] * 47, 1)


# ---------------------------------------------------------------------------------------------
# The packed block.
# ---------------------------------------------------------------------------------------------

class ChainAwareModelBatch(FakeModelBatch):
    """The fake fixture with what model_batch exposes under T2: kv_chains decided from the flag
    (as chained_writer_options does), and a forward that engages #1 per GDN layer and #2 per
    K/V write the way the real ones note them."""

    engage = True
    audit_values = None

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        chained = t2.cut('kv_chains')
        self.kv_chains, self.kv_fallback, self.kv_rows = chained, 0, (t2.kv_rows() if chained else 0)
        self.kv_single_chain = options.get('kv_single_chain', False)

    def forward(self, *, sharded_logits):
        fresh = self.retained is not None and not self.retained.records
        result = FakeModelBatch.forward(self, sharded_logits=sharded_logits)
        if type(self).engage and t2.cut('windows'):
            t2.note('windows', 48)
        if type(self).engage and self.kv_chains:
            t2.note('kv_chains', 32)
        if fresh and t2.cut('windows'):
            ttnn = type(self).ttnn
            for layer, (state, result_, carries) in enumerate(self.retained.records):
                for user, piece in enumerate(result_['segment_results']):
                    windows, served = [], []
                    for slot in range(4):
                        value = torch.full((16, 64), float(layer + user + slot), dtype=torch.bfloat16)
                        windows.append(ttnn.allocate((1, 16, 5120), 'bf16', 'tile', value))
                        other = value.clone() if type(self).audit_values is None else type(self).audit_values(layer, user, slot, value)
                        served.append(ttnn.allocate((1, 16, 5120), 'bf16', 'tile', other))
                    piece['packed_conv_states'] = windows
                    if t2.audit_enabled():
                        piece['audit_windows'] = served
        return result


class PackedBlockTests(FourUserFixture):
    def setUp(self):
        super().setUp()
        ChainAwareModelBatch.engage, ChainAwareModelBatch.audit_values = True, None
        patcher = patch.object(packed_verifier, 'ModelBatch', ChainAwareModelBatch)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.lines = []
        patcher = patch.object(packed_verifier, 'diagnostic', Mock(side_effect=self.lines.append))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.logged = []
        patcher = patch.object(t2, 'log_line', Mock(side_effect=self.logged.append))
        patcher.start()
        self.addCleanup(patcher.stop)
        t2.take()
        t2._AUDIT.update(rounds=0)
        self.addCleanup(t2.take)

    def build_with(self, environ):
        with patch.dict('os.environ', environ, clear=True):
            return self.build()

    def verify(self, block, environ, entries=None):
        with patch.dict('os.environ', environ, clear=True):
            return block.verify(entries if entries is not None else self.four(order=(0, 1, 2, 3)))

    def t2_lines(self):
        return [line for line in self.lines if 'verify t2' in line]

    def test_flag_off_the_placeholders_are_page_0_and_nothing_is_said(self):
        block = self.build_with({})
        for fixture in ChainAwareModelBatch.instances:
            self.assertTrue(all(bool((user['pages'] == 0).all()) for user in fixture.pack))
            self.assertFalse(fixture.kv_chains)
            self.assertNotIn('kv_single_chain', fixture.options, 'flag off: the fixtures are built as today')
        self.assertFalse(block.kv_chains)
        self.assertEqual(self.t2_lines(), [])
        self.verify(block, {})
        self.assertEqual(self.logged, [])

    def test_flag_on_the_placeholders_stay_page_0_and_only_the_warm_forward_is_one_chain(self):
        block = self.build_with(ON)
        warm, captured = ChainAwareModelBatch.instances
        for fixture in (warm, captured):
            self.assertTrue(all(bool((user['pages'] == 0).all()) for user in fixture.pack),
                            'the served placeholders: no page but vLLM\'s null block is ever written')
        self.assertIs(warm.options.get('kv_single_chain'), True)
        self.assertNotIn('kv_single_chain', captured.options, 'the capture bakes one chain per user')
        self.assertIs(block.fixture, captured)
        self.assertTrue(block.kv_chains)
        self.assertEqual(self.t2_lines(), ['[PINDIAG] verify t2 engaged site=packed_verify windows=48 windows_fallback=0 '
                                           'kv_chains=32 kv_fallback=0 kv_rows=64 warm_chain=single audit=0'])

    def test_the_marker_counts_only_the_captured_forward_and_reports_the_knob(self):
        ChainAwareModelBatch.engage = False
        self.build_with(dict(ON, QWEN_FAST_VERIFY_T2_KV_ROWS='32', QWEN_FAST_VERIFY_T2_SKIP='windows'))
        self.assertEqual(self.t2_lines(), ['[PINDIAG] verify t2 engaged site=packed_verify windows=0 windows_fallback=0 '
                                           'kv_chains=0 kv_fallback=0 kv_rows=32 warm_chain=single audit=0'])

    def test_skipping_kv_chains_builds_the_warm_fixture_as_today(self):
        block = self.build_with(dict(ON, QWEN_FAST_VERIFY_T2_SKIP='kv_chains'))
        self.assertFalse(block.kv_chains)
        for fixture in ChainAwareModelBatch.instances:
            self.assertTrue(all(bool((user['pages'] == 0).all()) for user in fixture.pack))
            self.assertNotIn('kv_single_chain', fixture.options)
        self.assertEqual(self.t2_lines(), ['[PINDIAG] verify t2 engaged site=packed_verify windows=48 windows_fallback=0 '
                                           'kv_chains=0 kv_fallback=0 kv_rows=0 warm_chain=none audit=0'])

    def test_shared_tile_rows_are_refused_before_any_staging_copy(self):
        block = self.build_with(ON)
        owners = [request(name, self.pool.slots[index], position, page)
                  for index, (name, position, page) in enumerate(zip('ABCD', (4100, 4200, 4106, 4300), (7, 11, 7, 17)))]
        entries = [entry(owner, range(16)) for owner in owners]
        copies, executed = len(self.ttnn.host_copies), len(self.ttnn.executed)
        with self.assertRaisesRegex(ValueError, 'disjoint cache tile rows: verify t2 kv tile rows shared: users 0,2 '
                                                'page 7 tile row 0'):
            self.verify(block, ON, entries)
        self.assertEqual((len(self.ttnn.host_copies), len(self.ttnn.executed)), (copies, executed))
        self.assertEqual(block.phase, 'failed')
        self.assertEqual(self.logged, ['[PINDIAG] verify t2 kv shared site=stage_packed verify t2 kv tile rows shared: '
                                       'users 0,2 page 7 tile row 0'])

    def test_without_the_chained_writer_the_same_round_is_staged(self):
        """kv_chains skipped: the served segmented writer serialises every row, so shared tile
        rows are the served path's own business and the backstop stays out of it."""
        block = self.build_with(dict(ON, QWEN_FAST_VERIFY_T2_SKIP='kv_chains'))
        owners = [request(name, self.pool.slots[index], position, page)
                  for index, (name, position, page) in enumerate(zip('ABCD', (4100, 4200, 4106, 4300), (7, 11, 7, 17)))]
        self.verify(block, dict(ON, QWEN_FAST_VERIFY_T2_SKIP='kv_chains'), [entry(owner, range(16)) for owner in owners])
        self.assertEqual(block.phase, 'verified')

    def test_the_audit_compares_every_layer_then_two_per_round_and_logs_the_kv_rows(self):
        audit = dict(ON, QWEN_FAST_VERIFY_T2_AUDIT='1')
        block = self.build_with(audit)
        self.assertIn('audit=1', self.t2_lines()[0])
        for round_number, prefixes in ((1, (1, 2, 3, 4)), (2, (0, 0, 0, 0))):
            predictions, metrics = self.verify(block, audit)
            for segment, prefix in zip(metrics['segments'], prefixes):
                block.commit_user(segment, prefix)
        self.assertEqual(self.logged, [
            '[PINDIAG] verify t2 audit kv_rows_per_user=1,1,2,1',
            '[PINDIAG] verify t2 audit 1 exact=True layers=0-47 windows=768',
            '[PINDIAG] verify t2 audit kv_rows_per_user=1,1,2,1',
            '[PINDIAG] verify t2 audit 2 exact=True layers=0,1 windows=32'])

    def test_an_audit_mismatch_fails_the_round(self):
        audit = dict(ON, QWEN_FAST_VERIFY_T2_AUDIT='1')

        def flip(layer, user, slot, value):
            other = value.clone()
            if (layer, user, slot) == (0, 2, 3):
                other[0, 0] = -other[0, 0] if float(other[0, 0]) else -0.0
            return other

        ChainAwareModelBatch.audit_values = staticmethod(flip)
        block = self.build_with(audit)
        entries = self.four(order=(0, 1, 2, 3))
        with self.assertRaises(AssertionError):
            self.verify(block, audit, entries)
        self.assertEqual(block.phase, 'failed')
        self.assertTrue(any(line.startswith('[PINDIAG] verify t2 audit mismatch round=1 layers=0-47 layer 0 user 2 '
                                            'slot 3') for line in self.logged), self.logged)
        for item in entries:
            item['request'].session.fail_verification.assert_called_once()

    def test_the_audit_waits_for_the_windows_cut(self):
        audit = dict(ON, QWEN_FAST_VERIFY_T2_AUDIT='1', QWEN_FAST_VERIFY_T2_SKIP='windows')
        block = self.build_with(audit)
        self.assertIn('audit=0', self.t2_lines()[0])
        self.verify(block, audit)
        self.assertEqual(self.logged, ['[PINDIAG] verify t2 audit kv_rows_per_user=1,1,2,1'])


# ---------------------------------------------------------------------------------------------
# The step.
# ---------------------------------------------------------------------------------------------

class StepGuardTests(unittest.TestCase):
    """The per-round guard where the round is decided. Beside the 64-row block - the only one that
    chains - every per-request engine captures only the sequential widths (1, 2, 4)
    (packed_shapes.M3_SEQUENTIAL_CAPTURE_ROWS), so the guard must act BEFORE the round is drafted
    at the block's 16 rows: proposal_rows answers None and the round is drafted at the engines'
    own widths, which the exact sequential step serves. A conflict first seen at the step (the
    backstop) finds 16-row tickets nothing else captured: the round is refused and nothing is
    written."""

    TRIMMED = (1, 2, 4)

    def setUp(self):
        import verifier_engine
        from test_serving_packed_step import FakeBlock

        verifier_engine.note_prefill()
        self.block = FakeBlock(users=4)
        self.stepped = []
        t2._LOGGED.clear()
        self.addCleanup(t2._LOGGED.clear)

    def owners(self, positions=(100, 3000, 700, 4000), pages=(7, 11, 13, 17), widths=TRIMMED):
        """A, B, C, D in segments 0-3 with their frontiers and page tables, nothing drafted yet."""
        from test_serving_packed_step import FakeRequest

        made = []
        for segment, (name, position, page) in enumerate(zip('ABCD', positions, pages)):
            owner = FakeRequest(name, position, self.stepped)
            owner.engine.widths = tuple(widths)
            owner.engine.pages = torch.full((1, 68), page, dtype=torch.int32)
            self.block.bind(owner.engine, segment)
            made.append(owner)
        return made

    def draft(self, owners, rows):
        from test_serving_packed_step import entry as step_entry

        for segment, owner in enumerate(owners):
            owner.propose(self.block.predictions_for(segment), 3, rows)
        return [step_entry(owners[index]) for index in (2, 0, 3, 1)]

    def test_flag_off_ineligible_and_proposal_rows_are_todays(self):
        from serving_packed_step import ineligible, proposal_rows

        owners = self.owners(pages=(7, 7, 7, 7), positions=(100, 100, 100, 100))
        with patch.object(t2, 'log_line') as logged:
            self.assertEqual(proposal_rows(self.block, owners), 16)
            self.assertIsNone(ineligible(self.draft(owners, 16), self.block))
        logged.assert_not_called()

    def test_disjoint_tile_rows_keep_the_block_round(self):
        from serving_packed_step import ineligible, proposal_rows

        self.block.kv_chains = True
        # users 0 and 2 on one physical page at different tile rows: served
        owners = self.owners(positions=(100, 3000, 130, 4000), pages=(7, 11, 7, 17))
        with patch.object(t2, 'log_line') as logged:
            self.assertEqual(proposal_rows(self.block, owners), 16)
            self.assertIsNone(ineligible(self.draft(owners, 16), self.block))
        logged.assert_not_called()

    def test_a_conflict_before_drafting_drafts_the_round_for_the_exact_sequential_step(self):
        from serving_packed_step import packed_device_step, proposal_rows

        self.block.kv_chains = True
        owners = self.owners(positions=(100, 3000, 110, 4000), pages=(7, 11, 7, 17))
        logged = []
        with patch.object(t2, 'log_line', side_effect=logged.append):
            self.assertIsNone(proposal_rows(self.block, owners))
            self.assertIsNone(proposal_rows(self.block, owners), 'every tick, logged once')
            # drafted at the engines' own width (serving_worker_hook: packed_rows None)
            entries = self.draft(owners, 4)
            outputs = packed_device_step(entries, cancelled=lambda: False, block=self.block)
        self.assertEqual(logged, ['[PINDIAG] verify t2 kv shared site=proposal_rows verify t2 kv tile rows shared: '
                                  'users 0,2 page 7 tile row 1'])
        self.assertEqual(self.block.calls, [], 'the block was never touched')
        self.assertEqual([item[0] for item in self.stepped], ['C', 'A', 'D', 'B'])
        self.assertEqual([output.request_id for output in outputs], ['C', 'A', 'D', 'B'])
        self.assertTrue(all(owner.session.phase == 'pending' for owner in owners), 'no session failed')

    def test_a_conflict_first_seen_at_the_step_refuses_the_round_and_writes_nothing(self):
        from serving_packed_step import ineligible, packed_device_step

        self.block.kv_chains = True
        owners = self.owners()
        entries = self.draft(owners, 16)
        # the tables change between the drafting and the step (the backstop's case)
        owners[2].engine.pages = torch.full((1, 68), 7, dtype=torch.int32)
        owners[2].session.pending.position = owners[2].engine.position = owners[2].session.position = 110
        logged = []
        with patch.object(t2, 'log_line', side_effect=logged.append), patch('sys.stdout'):
            reason = ineligible(entries, self.block)
            outputs = packed_device_step(entries, cancelled=lambda: False, block=self.block)
        self.assertEqual(reason, 'verify t2 kv tile rows shared: users 0,2 page 7 tile row 1')
        self.assertEqual(logged, ['[PINDIAG] verify t2 kv shared site=ineligible ' + reason] * 2)
        self.assertEqual(self.block.calls, [], 'the block was never touched')
        self.assertEqual(self.stepped, [], 'nothing captured the 16-row tickets: no sequential step either')
        self.assertTrue(all(output.finished and output.cancelled for output in outputs))
        self.assertTrue(all(owner.session.phase == 'failed' for owner in owners))

    def test_an_unmapped_position_or_a_missing_table_fails_closed_at_both_sites(self):
        from serving_packed_step import ineligible, proposal_rows

        self.block.kv_chains = True
        owners = self.owners(positions=(100, 3000, 68 * 64 - 8, 4000))     # C's rows run past its table
        with patch.object(t2, 'log_line') as logged:
            self.assertIsNone(proposal_rows(self.block, owners))
            self.assertTrue(ineligible(self.draft(owners, 16), self.block).startswith('verify t2 kv tile rows unmapped'))
        self.assertTrue(all('kv tile rows unmapped' in call.args[0] for call in logged.call_args_list))
        t2._LOGGED.clear()
        self.stepped.clear()
        self.block = type(self.block)(users=4)
        self.block.kv_chains = True
        owners = self.owners()
        del owners[1].engine.pages
        with patch.object(t2, 'log_line'):
            self.assertIsNone(proposal_rows(self.block, owners))
            self.assertTrue(ineligible(self.draft(owners, 16), self.block).startswith('verify t2 kv tile rows unmapped'))

    def test_the_hook_drafts_at_the_engines_width_when_the_policy_says_none(self):
        """serving_worker_hook: a round the policy answers None for discards any stale block-width
        ticket and redrafts at the engine's own width - the width the trimmed engines capture."""
        from serving_packed_step import PackedStep, unservable
        from serving_worker_hook import discard_stale_ticket

        self.block.kv_chains = True
        owners = self.owners(positions=(100, 3000, 110, 4000), pages=(7, 11, 7, 17))
        entries = self.draft(owners, 16)                         # drafted before the conflict was seen
        with patch.object(t2, 'log_line'):
            packed_rows = PackedStep(self.block).proposal_rows(owners)
        self.assertIsNone(packed_rows)
        for owner in owners:
            discard_stale_ticket(owner, packed_rows)
        self.assertTrue(all(owner.session.pending is None for owner in owners))
        entries = self.draft(owners, 4)
        self.assertEqual(unservable(entries), [])


# ---------------------------------------------------------------------------------------------
# The fixture.
# ---------------------------------------------------------------------------------------------

class FixtureTests(unittest.TestCase):
    def setUp(self):
        t2.take()
        t2._LOGGED.clear()
        self.addCleanup(t2.take)

    def tiles(self):
        from packed_cache_writer import tile
        from test_gdn_conv_windows_packed import DescriptorTTNN

        ttnn = DescriptorTTNN()
        return ttnn, [tile((first, last), ttnn.tensor('p%d' % first, (32,), 'dram', 'int32', 'row_major'),
                           ttnn.tensor('t%d' % first, (32, 68), 'dram', 'int32', 'row_major'))
                      for first, last in ((0, 32), (32, 64))]

    def test_the_options_exist_only_for_a_packed_wide_block_with_the_cut(self):
        from model_batch import chained_writer_options

        pack = dict(segments=((0, 16), (16, 32), (32, 48), (48, 64)))
        with patch.dict('os.environ', {}, clear=True):
            self.assertIsNone(chained_writer_options(pack, ['tile'], 'positions', 'pages'))
        with patch.dict('os.environ', ON, clear=True):
            self.assertEqual(chained_writer_options(pack, ['tile'], 'positions', 'pages'),
                             dict(positions='positions', pages='pages', spans=pack['segments'], tiles=['tile'],
                                  launch_rows=64))
            self.assertIsNone(chained_writer_options(None, ['tile'], 'positions', 'pages'))
            self.assertIsNone(chained_writer_options(pack, [], 'positions', 'pages'))
        with patch.dict('os.environ', dict(ON, QWEN_FAST_VERIFY_T2_KV_ROWS='32'), clear=True):
            self.assertEqual(chained_writer_options(pack, ['tile'], 'p', 'q')['launch_rows'], 32)
        with patch.dict('os.environ', dict(ON, QWEN_FAST_VERIFY_T2_SKIP='kv_chains'), clear=True):
            self.assertIsNone(chained_writer_options(pack, ['tile'], 'positions', 'pages'))

    def test_the_writer_is_chained_only_when_asked_and_falls_back_counted(self):
        from model_batch import cache_writer, chained_writer_summary
        from packed_cache_writer import SegmentedOrderedCacheWriter
        from packed_ordered_cache import ChainedOrderedCacheWriter
        from test_gdn_conv_windows_packed import mesh

        ttnn, tiles = self.tiles()
        chained = dict(positions=ttnn.tensor('positions', (64,), 'dram', 'int32', 'row_major'),
                       pages=ttnn.tensor('pages', (64, 68), 'dram', 'int32', 'row_major'),
                       spans=((0, 16), (16, 32), (32, 48), (48, 64)), tiles=tiles, launch_rows=64)
        serial = Mock()
        self.assertIsInstance(cache_writer(ttnn, mesh(), 'k', ordered_cache=True, cache_tiles=tiles, serial=serial),
                              SegmentedOrderedCacheWriter)
        writer = cache_writer(ttnn, mesh(), 'k', ordered_cache=True, cache_tiles=tiles, serial=serial, chained=chained)
        self.assertIsInstance(writer, ChainedOrderedCacheWriter)
        self.assertEqual(t2.take(), {})
        logged = []
        with patch.object(t2, 'log_line', side_effect=logged.append):
            small = [cache_writer(ttnn, mesh(grid=(8, 4)), 'k', ordered_cache=True, cache_tiles=tiles, serial=serial,
                                  chained=chained) for layer in range(16)]
        self.assertTrue(all(isinstance(value, SegmentedOrderedCacheWriter) for value in small))
        self.assertEqual(t2.take(), {'kv_fallback': 16})
        self.assertEqual(len(logged), 1)
        self.assertTrue(logged[0].startswith('[PINDIAG] verify t2 fell back site=kv_chains reason=a 64-row chained'))
        serial.assert_not_called()
        self.assertEqual(chained_writer_summary([writer] * 16, chained), (True, 0, 64))
        self.assertEqual(chained_writer_summary(small, chained), (False, 16, 0))
        self.assertEqual(chained_writer_summary(small, None), (False, 0, 0))

    def closable(self, audit):
        from model_batch import ModelBatch

        fixture = ModelBatch.__new__(ModelBatch)
        served = [SimpleNamespace(name='w%d' % index) for index in range(8)]
        records = [(None, dict(segment_results=(dict(audit_windows=served[:4]), dict(audit_windows=served[4:]))), None),
                   (None, dict(segment_results=(dict(), dict())), None)]
        fixture.retained = SimpleNamespace(records=records, close=Mock())
        fixture.working_states, fixture.grouped_readers, fixture.buffers, fixture.borrowed = [], [], [], []
        fixture.operations = SimpleNamespace(deallocate=Mock(),
                                             get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda v=value: id(v))] * 2)
        if audit is not None:
            fixture.verify_t2_audit = audit
        return fixture, served

    def test_close_releases_the_audit_windows_the_records_hold(self):
        fixture, served = self.closable(True)
        fixture.close()
        self.assertEqual([call.args[0] for call in fixture.operations.deallocate.call_args_list], served)
        fixture.retained.close.assert_called_once()

    def test_without_the_audit_close_never_looks_for_audit_windows(self):
        for audit in (False, None):
            with self.subTest(audit=audit), patch.object(t2, 'audit_windows_of') as looked:
                fixture, served = self.closable(audit)
                fixture.close()
                looked.assert_not_called()
                fixture.operations.deallocate.assert_not_called()
                fixture.retained.close.assert_called_once()

    def test_the_audit_is_read_once_at_construction_and_only_under_the_flag(self):
        source = (HERE / 'model_batch.py').read_text(encoding='utf-8')
        self.assertEqual(source.count('self.verify_t2_audit = verify_trace_t2.audit_enabled()'), 1)
        self.assertEqual(source.count('verify_trace_t2.audit_windows_of('), 2)
        for site in ('                    if self.verify_t2_audit:' + chr(10) + '                        audit = verify_trace_t2.audit_windows_of(result)',
                     "            if getattr(self, 'verify_t2_audit', False):" + chr(10) + '                audit = ['):
            self.assertIn(site, source)

    def test_the_warm_fixture_gets_one_span_over_the_block(self):
        from model_batch import chained_writer_options

        pack = dict(segments=((0, 16), (16, 32), (32, 48), (48, 64)))
        with patch.dict('os.environ', ON, clear=True):
            self.assertEqual(chained_writer_options(pack, ['tile'], 'p', 'q', single_chain=True)['spans'], ((0, 64),))
            self.assertEqual(chained_writer_options(pack, ['tile'], 'p', 'q')['spans'], pack['segments'])
        with patch.dict('os.environ', {}, clear=True):
            self.assertIsNone(chained_writer_options(pack, ['tile'], 'p', 'q', single_chain=True))
        with patch.dict('os.environ', dict(ON, QWEN_FAST_VERIFY_T2_SKIP='kv_chains'), clear=True):
            self.assertIsNone(chained_writer_options(pack, ['tile'], 'p', 'q', single_chain=True))

    def test_the_single_span_chain_is_one_chain_in_row_order(self):
        import packed_ordered_cache as poc

        chains = poc.chain_args(64, ((0, 64),))
        self.assertEqual(chains[0], (0, 1, 1))
        self.assertEqual(chains[63], (1, 0, 63))
        self.assertTrue(all(chains[row] == (1, 1, row + 1) for row in range(1, 63)))
        # the 32-row mode splits it into the served structure: one chain per tile
        self.assertEqual([poc.tile_spans(((0, 64),), first, last) for first, last in ((0, 32), (32, 64))],
                         [((0, 32),), ((0, 32),)])


if __name__ == '__main__':
    unittest.main()
