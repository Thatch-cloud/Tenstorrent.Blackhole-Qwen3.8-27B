"""Op-fusion programme, host-gap package WPH, lever 1: the pre-stage writes only the destinations whose bytes differ (QWEN_FAST_PRESTAGE_DIFF, audit twin
QWEN_FAST_PRESTAGE_DIFF_AUDIT; prestage_diff.py, hooked in verify_prestage.BlockPrestage; default off, host only).

The block is test_verify_prestage's four-user block (and test_tp4_hostgap's pair of real blocks) over the fake device model, which computes every replay's outputs
from what is staged, so a wrong staged byte is a wrong prediction too. Pinned here:

  - the flags are 0 or 1, the audit needs the lever, and with the flag off nothing of the lever exists: no state on the block, the window's lines are today's;
  - same_bits: dtype, shape and every bit (+0 is not -0, a NaN payload is its own), integers by value;
  - over random users, page tables, idle sets and T2 on or off, the device after every window and every verify holds exactly what the full stage writes, the
    predictions are the flag-off run's, and the destinations a window writes are exactly the ones whose value is not bit-equal to the one the device held (an
    independent oracle over the previous round's values);
  - a page append rewrites that user's page buffers and the block-wide tables, and the round after it rewrites none of them;
  - every other writer of the fixture's inputs kills the resident (the next window is the full one, with its reason), a pre-stage that raises forgets it, a
    destination list that is not the resident's takes the full pre-stage, and a full verify stage records nothing;
  - the audit twin reads every pre-staged destination back from every chip, finds a destination a trace changed that the resident did not know, repairs the round, and
    latches the lever off; the existing full audit (QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT) finds the same difference at the verify, independently;
  - it composes with KEYED (round_host), with the lean writer, and with two real blocks under the per-block epochs: each block's resident survives the other's round,
    and an external writer kills both;
  - the smoke rule and the shipping lists."""

import os
from pathlib import Path
import random
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import packed_verifier
import prestage_diff
import round_host
import test_tp4_hostgap as thg
import test_verify_prestage as tvp
import verify_prestage
import write_packed_lean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NAMES = tvp.NAMES
DIFF, AUDIT, LEAN, KEYED = ('QWEN_FAST_PRESTAGE_DIFF', 'QWEN_FAST_PRESTAGE_DIFF_AUDIT', 'QWEN_FAST_WRITE_PACKED_LEAN', 'QWEN_FAST_TP4_ROUND_HOST_KEYED')


def clean():
    prestage_diff.reset()
    write_packed_lean.reset()
    round_host.reset_counters()
    verify_prestage._MODE.update(first=False, blocks=False)
    verify_prestage._LOCAL.clear()
    verify_prestage._ADDRESSES.clear()


class FlagTests(unittest.TestCase):
    def test_the_flags_are_zero_or_one_and_the_audit_needs_the_lever(self):
        self.assertFalse(prestage_diff.enabled({}))
        self.assertTrue(prestage_diff.enabled({DIFF: '1'}))
        self.assertFalse(prestage_diff.audit_enabled({AUDIT: '1'}))
        self.assertTrue(prestage_diff.audit_enabled({DIFF: '1', AUDIT: '1'}))
        for name in (DIFF, AUDIT):
            for value in ('yes', '2', ''):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    prestage_diff._flag(name, {name: value})

    def test_a_misset_flag_fails_at_the_attach_and_an_unset_or_zero_one_imports_nothing(self):
        clean()
        block = SimpleNamespace(users=4)
        with patch.dict(os.environ, {DIFF: 'yes'}), self.assertRaises(ValueError):
            verify_prestage.BlockPrestage(block, audit=False)
        with patch.dict(os.environ, {DIFF: '0', AUDIT: '1'}):
            self.assertIsNone(verify_prestage.BlockPrestage(block, audit=False).diff)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(DIFF, None)
            os.environ.pop(AUDIT, None)
            made = verify_prestage.BlockPrestage(block, audit=False)
        self.assertEqual((made.diff, made.lean, made.resident), (None, None, None))


class SameBitsTests(unittest.TestCase):
    def test_floating_point_values_are_compared_by_their_bits(self):
        plus, minus = torch.tensor([0.0, 1.0], dtype=torch.bfloat16), torch.tensor([-0.0, 1.0], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(plus, minus), 'torch.equal calls them equal')
        self.assertFalse(prestage_diff.same_bits(plus, minus))
        self.assertTrue(prestage_diff.same_bits(plus, plus.clone()))
        nan_a = torch.tensor([0x7FC1], dtype=torch.int16).view(torch.bfloat16)
        nan_b = torch.tensor([0x7FC2], dtype=torch.int16).view(torch.bfloat16)
        self.assertFalse(prestage_diff.same_bits(nan_a, nan_b))
        self.assertTrue(prestage_diff.same_bits(nan_a, nan_a.clone()))
        for dtype in (torch.float32, torch.float16, torch.float64):
            value = torch.tensor([1.5, -0.0], dtype=dtype)
            self.assertTrue(prestage_diff.same_bits(value, value.clone()))
            self.assertFalse(prestage_diff.same_bits(value, torch.tensor([1.5, 0.0], dtype=dtype)))

    def test_integers_dtype_and_shape(self):
        value = torch.arange(6, dtype=torch.int32).reshape(2, 3)
        self.assertTrue(prestage_diff.same_bits(value, value.clone()))
        self.assertFalse(prestage_diff.same_bits(value, value.reshape(3, 2)))
        self.assertFalse(prestage_diff.same_bits(value, value.to(torch.int64)))
        other = value.clone()
        other[1, 2] += 1
        self.assertFalse(prestage_diff.same_bits(value, other))
        self.assertTrue(prestage_diff.same_bits(torch.zeros(8, dtype=torch.int32)[2:5], torch.zeros(3, dtype=torch.int32)))


class DiffFixture(tvp.PrestageFixture):
    """The four-user block with the lever on (and whatever else a test asks for), every line captured."""

    def setUp(self):
        super().setUp()
        clean()
        self.addCleanup(clean)
        for name in (DIFF, AUDIT, LEAN, KEYED, 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT', 'QWEN_FAST_TP4_ROUND_HOST_AUDIT'):
            os.environ.pop(name, None)
        round_host.refresh()
        self.addCleanup(round_host.refresh, {})

    def open_diff(self, *, audit=False, lean=False, keyed=False, full_audit=False, padded=None, kv_chains=False, **flags):
        os.environ[DIFF] = '1'
        if audit:
            os.environ[AUDIT] = '1'
        if lean:
            os.environ[LEAN] = '1'
        if keyed:
            os.environ[KEYED] = '1'
        if full_audit:
            os.environ['QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT'] = '1'
        os.environ.update(flags)
        round_host.refresh(dict(os.environ))
        return self.open_block(padded=padded, kv_chains=kv_chains)

    def diff_lines(self, marker):
        return [line for line in self.h1a if line.startswith(marker + ' ')]

    def diff_paths(self):
        return [tuple(re.search(r'path=(\w+) reason=(\S+) total=(\d+) written=(\d+)', line).groups()) for line in self.diff_lines(prestage_diff.ROUND_MARKER)]

    def expected_changed(self, block, previous, users):
        """The independent oracle: the pre-staged destinations whose value is not bit-equal to the previous round's, from packed_values over the two rounds' users."""
        old, _ = packed_verifier.packed_values(block.operations, block.model, block.fixture, block.shape, previous, guard=False)
        new, _ = packed_verifier.packed_values(block.operations, block.model, block.fixture, block.shape, users, guard=False)
        tokens = block.fixture.tokens
        return [new[index][0] for index in range(len(new)) if new[index][0] is not tokens
                and not prestage_diff.same_bits(old[index][1], new[index][1])]


class EquivalenceTests(DiffFixture, tvp.EquivalenceTests):
    """The existing random schedules, with the lever on: the device state is the full stage's after every round and the predictions are the flag-off run's."""

    # the inherited tests are tvp's own (flag off); the lever's are below
    test_random_rounds_stage_exactly_what_stage_packed_writes = None

    def run_arm(self, plan, padded, kv_chains, **flags):
        block = self.open_diff(padded=2 if padded else None, kv_chains=kv_chains, **flags)
        served, previous = [], None
        for number, spec in enumerate(plan):
            if number:
                finished = [name for name in NAMES if name not in spec['live']]
                if previous is not None:
                    expected = self.expected_changed(block, previous, self.staged_users(block, self.entries(spec['window'], spec['token_base'])))
                else:
                    expected = None
                before = len(self.ttnn.host_copies)
                self.window(block, spec['window'], finished=finished)
                copied = [destination for host, destination in self.ttnn.host_copies[before:]]
                if expected is not None and self.diff_paths() and self.diff_paths()[-1][0] == 'diff':
                    self.assertEqual([id(value) for value in copied], [id(value) for value in expected], 'round %d: the window wrote exactly the changed' % number)
            predictions, metrics, staged = self.round(block, spec['users'], spec['token_base'])
            self.assert_device_holds(block, staged)
            previous = staged
            served.append((predictions, metrics['segments']))
        block.close()
        self.block = None
        return served

    def test_random_rounds_stage_exactly_what_stage_packed_writes_with_the_lever_on(self):
        for seed in range(8):
            padded, kv_chains = seed % 2 == 1, seed % 4 >= 2
            with self.subTest(seed=seed, padded=padded, kv_chains=kv_chains):
                plan = self.schedule(random.Random(seed), rounds=6, padded=padded)
                self.h1a.clear()
                clean()
                on = self.run_arm(plan, padded, kv_chains)
                diff_paths = [path for path, reason, total, written in self.diff_paths()]
                off = self.run_schedule(plan, False, padded, kv_chains)
                self.assertEqual(on, off)
                self.assertEqual(diff_paths[0], 'full')
                self.assertEqual(set(diff_paths[1:]), {'diff'}, 'after the first round every window is a diff: %s' % diff_paths)
                for path, reason, total, written in self.diff_paths():
                    self.assertLessEqual(int(written), int(total))
                self.assertTrue(any(int(written) < int(total) for path, reason, total, written in self.diff_paths() if path == 'diff'))


class DiffTests(DiffFixture):
    def test_off_the_block_holds_nothing_and_the_window_line_is_todays(self):
        block = self.open_block()
        self.assertEqual((block.prestaged.diff, block.prestaged.lean, block.prestaged.resident), (None, None, None))
        users = tvp.base_users()
        self.round(block, users)
        self.window(block, tvp.advanced(users, 2))
        self.round(block, tvp.advanced(users, 2))
        self.assertIsNone(block.prestaged.resident)
        self.assertEqual(self.diff_lines(prestage_diff.ROUND_MARKER), [])
        self.assertRegex(self.marked(verify_prestage.WINDOW_MARKER)[0], r'^\[PACKED-PRESTAGE-WINDOW\] round=2 buffers=[0-9]+ ms=[0-9.]+ live=4$')
        self.assertNotIn(prestage_diff.ENGAGED_MARKER, ' '.join(self.h1a))

    def test_the_first_window_after_a_full_round_is_full_then_the_windows_write_the_changed_and_the_verify_writes_the_tokens(self):
        block = self.open_diff()
        self.assertTrue(any(line.startswith(prestage_diff.ENGAGED_MARKER) for line in self.h1a))
        users = tvp.base_users()
        self.round(block, users)
        self.assertIsNone(block.prestaged.resident, 'a full stage records nothing')
        users = tvp.advanced(users, 2)
        full = self.written(lambda: self.window(block, users))
        self.assertEqual(self.diff_paths()[-1][:2], ('full', 'no-resident'))
        self.assertEqual(len(full), int(self.diff_paths()[-1][2]))
        written = self.written(lambda: self.round(block, users))
        self.assertEqual(written, [block.fixture.tokens])
        self.assertIsNotNone(block.prestaged.resident)
        nxt = tvp.advanced(users, 3)
        expected = self.expected_changed(block, self.staged_users(block, self.entries(users)), self.staged_users(block, self.entries(nxt)))
        written = self.written(lambda: self.window(block, nxt))
        self.assertEqual(self.diff_paths()[-1][:2], ('diff', '-'))
        self.assertEqual([id(value) for value in written], [id(value) for value in expected])
        self.assertLess(len(written), len(full))
        fixture = block.fixture
        self.assertNotIn(id(fixture.pages), [id(value) for value in written], 'no page changed: no page table is rewritten')
        self.assertTrue(all(id(row) not in {id(value) for value in written} for row in fixture.row_pages))
        self.assertIn(id(fixture.positions), [id(value) for value in written])
        self.assertIn(id(fixture.cos), [id(value) for value in written])
        self.assertEqual(self.written(lambda: self.round(block, nxt)), [fixture.tokens])
        self.assert_device_holds(block, self.staged_users(block, self.entries(nxt)))

    def test_a_page_append_rewrites_that_users_page_buffers_once_and_the_round_after_it_none(self):
        block = self.open_diff()
        users = tvp.base_users()
        self.round(block, users)
        users = tvp.advanced(users, 5)
        self.window(block, users)
        self.round(block, users)
        appended = dict(tvp.advanced(users, 3))
        position, pages = appended['C']
        pages = pages.clone()
        pages[0, (position + 15) // 64 + 1] = 99
        appended['C'] = (position, pages)
        fixture = block.fixture
        written = self.written(lambda: self.window(block, appended))
        ids = {id(value) for value in written}
        segment = 2
        rows = range(16 * segment, 16 * segment + 16)
        tile = next(tile for tile in fixture.cache_tiles if tile.rows[0] <= rows[0] < tile.rows[1])
        reader = fixture.replay_reader.readers[segment]
        for destination in (fixture.pages, *[fixture.row_pages[row] for row in rows], tile.pages, *[entry[1] for entry in reader.metadata]):
            self.assertIn(id(destination), ids)
        other_rows = [row for row in range(64) if row not in rows]
        self.assertTrue(all(id(fixture.row_pages[row]) not in ids for row in other_rows), 'the other users page tables stay')
        self.round(block, appended)
        again = tvp.advanced(appended, 2)
        written = self.written(lambda: self.window(block, again))
        self.assertTrue({id(fixture.pages)} .isdisjoint({id(value) for value in written}))
        self.assertTrue(all(id(fixture.row_pages[row]) not in {id(value) for value in written} for row in range(64)))
        self.assertEqual(self.diff_paths()[-1][:2], ('diff', '-'))

    def test_every_other_writer_kills_the_resident_and_the_next_window_is_the_full_one(self):
        for reason, writer in (('stage_packed', lambda block: verify_prestage.bump_fixture(block.fixture, 'stage_packed')),
                               ('prefill-chunk', lambda block: verify_prestage.bump('prefill-chunk')),
                               ('admission', lambda block: verify_prestage.bump('admission')),
                               ('detach', lambda block: verify_prestage.bump('detach'))):
            with self.subTest(reason=reason):
                clean()
                self.h1a.clear()
                block = self.open_diff()
                users = tvp.base_users()
                self.round(block, users)
                users = tvp.advanced(users, 2)
                self.window(block, users)
                self.round(block, users)
                self.assertIsNotNone(block.prestaged.resident)
                writer(block)
                nxt = tvp.advanced(users, 2)
                total = self.written(lambda: self.window(block, nxt))
                self.assertEqual(self.diff_paths()[-1][:2], ('full', 'epoch:%s' % reason), self.diff_paths())
                self.assertEqual(len(total), int(self.diff_paths()[-1][2]))
                self.assertEqual(self.written(lambda: self.round(block, nxt)), [block.fixture.tokens])
                self.assertEqual(self.paths()[-1], 'diff')
                after = tvp.advanced(nxt, 2)
                self.window(block, after)
                self.assertEqual(self.diff_paths()[-1][:2], ('diff', '-'), 'the lever is back the round after')

    def test_a_snapshot_that_was_not_used_still_leaves_the_resident_what_the_device_holds(self):
        block = self.open_diff()
        users = tvp.base_users()
        self.round(block, users)
        self.window(block, tvp.advanced(users, 2))
        self.window(block, tvp.advanced(users, 4))        # a second window with no verify between: the device holds the first
        self.assertEqual([item[:2] for item in self.diff_paths()], [('full', 'no-resident'), ('diff', '-')])
        nxt = tvp.advanced(users, 4)
        self.round(block, nxt)
        self.assert_device_holds(block, self.staged_users(block, self.entries(nxt)))

    def test_a_pre_stage_that_raises_forgets_the_resident_and_the_next_window_is_full(self):
        block = self.open_diff()
        users = tvp.base_users()
        self.round(block, users)
        users = tvp.advanced(users, 2)
        self.window(block, users)
        self.round(block, users)
        self.assertIsNotNone(block.prestaged.resident)
        with patch.object(packed_verifier, 'packed_values', side_effect=ValueError('boom')):
            self.window(block, tvp.advanced(users, 1))
        self.assertIsNone(block.prestaged.resident)
        self.assertIsNone(block.prestaged.snapshot)
        self.round(block, tvp.advanced(users, 1))
        self.assertEqual(self.paths()[-1], 'full')
        self.window(block, tvp.advanced(users, 3))
        self.assertEqual(self.diff_paths()[-1][0], 'full')

    def test_a_destination_list_that_is_not_the_residents_takes_the_full_pre_stage(self):
        values = [(object(), torch.zeros(2), 'int32', 'row_major') for _ in range(4)]
        resident = prestage_diff.Resident(0, 0, [value[0] for value in values], [value[1:] for value in values], 1)
        prestage_diff.reset()
        self.assertEqual(prestage_diff.plan(resident, None, values, [0, 1, 2, 3]), ([], 'diff', '-'))
        replaced = list(values)
        replaced[2] = (object(),) + values[2][1:]
        self.assertEqual(prestage_diff.plan(resident, None, replaced, [0, 1, 2, 3]), ([0, 1, 2, 3], 'full', 'destinations'))
        self.assertEqual(prestage_diff.plan(resident, None, values[:3], [0, 1, 2]), ([0, 1, 2], 'full', 'destinations'))
        changed = list(values)
        changed[1] = (values[1][0], torch.ones(2), 'int32', 'row_major')
        retyped = list(values)
        retyped[3] = (values[3][0], values[3][1], 'uint32', 'row_major')
        self.assertEqual(prestage_diff.plan(resident, None, changed, [0, 1, 2, 3])[0], [1])
        self.assertEqual(prestage_diff.plan(resident, None, retyped, [0, 1, 2, 3])[0], [3])
        self.assertEqual(prestage_diff.plan(None, 'no-resident', values, [0, 1]), ([0, 1], 'full', 'no-resident'))

    def test_the_lever_own_failure_is_the_full_pre_stage_and_a_latch(self):
        block = self.open_diff()
        users = tvp.base_users()
        self.round(block, users)
        users = tvp.advanced(users, 2)
        self.window(block, users)
        self.round(block, users)
        with patch.object(prestage_diff, 'same_bits', side_effect=RuntimeError('x')):
            written = self.written(lambda: self.window(block, tvp.advanced(users, 3)))
        self.assertEqual(self.diff_paths()[-1][:2], ('full', 'plan-failed'))
        self.assertEqual(len(written), int(self.diff_paths()[-1][2]))
        self.assertEqual(prestage_diff.latched(), 'plan:RuntimeError')
        self.assertTrue(any(line.startswith(prestage_diff.FELL_BACK_MARKER) for line in self.h1a))
        self.round(block, tvp.advanced(users, 3))
        self.window(block, tvp.advanced(users, 5))
        self.assertEqual(self.diff_paths()[-1][:2], ('full', 'latched:plan:RuntimeError'))

    def test_a_sign_of_zero_in_a_rotary_table_is_a_difference(self):
        block = self.open_diff()
        users = tvp.base_users()
        self.round(block, users)
        users = tvp.advanced(users, 2)
        self.window(block, users)
        self.round(block, users)
        resident = block.prestaged.resident
        fixture = block.fixture
        index = next(position for position, destination in enumerate(resident.destinations) if destination is fixture.cos)
        kept = resident.values[index]
        zero = torch.zeros_like(kept[0])
        resident.values[index] = (zero, kept[1], kept[2])
        values, readers = packed_verifier.packed_values(block.operations, block.model, fixture, block.shape,
                                                        self.staged_users(block, self.entries(users)), guard=False)
        current = list(values)
        current[index] = (current[index][0], (-zero).clone(), current[index][2], current[index][3])
        indices = [position for position, value in enumerate(current) if value[0] is not fixture.tokens]
        changed, path, reason = prestage_diff.plan(resident, None, current, indices)
        self.assertIn(index, changed, '-0 against +0 is written: the bytes may differ')
        self.assertEqual(path, 'diff')


class AuditTests(DiffFixture):
    def run_two_rounds(self, block):
        users = tvp.base_users()
        self.round(block, users)
        users = tvp.advanced(users, 2)
        self.window(block, users)
        self.round(block, users)
        return users

    def test_the_audit_reads_every_pre_staged_destination_from_every_chip_and_passes(self):
        block = self.open_diff(audit=True)
        users = self.run_two_rounds(block)
        self.window(block, tvp.advanced(users, 3))
        self.round(block, tvp.advanced(users, 3))
        audits = self.diff_lines(prestage_diff.AUDIT_MARKER)
        self.assertEqual(len(audits), 2)
        chips = len(self.ttnn.get_device_tensors(block.fixture.tokens))
        total = int(self.diff_paths()[-1][2])
        for line in audits:
            self.assertRegex(line, r'^\[PINDIAG\] tp4 prestage diff audit block=- round=\d+ checked=\d+ skipped=\d+ mismatches=0 exact=True$')
        checked, skipped = (int(value) for value in re.search(r'checked=(\d+) skipped=(\d+)', audits[-1]).groups())
        self.assertEqual(checked, total * chips)
        self.assertEqual(skipped, total - int(self.diff_paths()[-1][3]))
        self.assertGreater(skipped, 0)
        self.assertIsNone(prestage_diff.latched())

    def test_the_audit_flag_alone_audits_nothing(self):
        os.environ[AUDIT] = '1'
        block = self.open_block()
        self.run_two_rounds(block)
        self.assertEqual(self.diff_lines(prestage_diff.AUDIT_MARKER), [])
        self.assertIsNone(block.prestaged.diff)

    def test_a_destination_a_trace_changed_is_found_repaired_and_latches_the_lever(self):
        block = self.open_diff(audit=True)
        users = self.run_two_rounds(block)
        fixture = block.fixture
        # a trace (or a writer the epoch missed) changed a page table the resident believes is right: the next window would skip it
        fixture.row_pages[5].value = fixture.row_pages[5].value + 1
        nxt = tvp.advanced(users, 3)
        self.window(block, nxt)
        mismatches = self.diff_lines(prestage_diff.MISMATCH_MARKER)
        self.assertEqual(len(mismatches), 1)
        self.assertRegex(mismatches[0], r'mismatches=\d+ at=\S+ exact=False$')
        self.assertEqual(prestage_diff.latched(), 'audit_mismatch')
        # repaired before the trace (the pre-staged destinations hold the full values; the readers' starts move at the verify)
        values, _ = packed_verifier.packed_values(block.operations, block.model, fixture, block.shape, self.staged_users(block, self.entries(nxt)), guard=False)
        wrong = [index for index, (destination, value, dtype, layout) in enumerate(values)
                 if destination is not fixture.tokens and not torch.equal(destination.value, value)]
        self.assertEqual(wrong, [])
        self.round(block, nxt)
        later = tvp.advanced(nxt, 2)
        self.window(block, later)
        self.assertEqual(self.diff_paths()[-1][:2], ('full', 'latched:audit_mismatch'))

    def test_without_the_audit_the_existing_full_audit_finds_it_at_the_verify(self):
        block = self.open_diff(full_audit=True)
        users = self.run_two_rounds(block)
        block.fixture.row_pages[5].value = block.fixture.row_pages[5].value + 1
        nxt = tvp.advanced(users, 3)
        self.window(block, nxt)
        self.round(block, nxt)
        lines = self.marked(verify_prestage.FULL_AUDIT_MARKER)
        self.assertTrue(any('path=diff' in line and 'mismatches=0' not in line for line in lines), lines)


class CompositionTests(DiffFixture):
    def run_rounds(self, block, steps=5):
        users, served = tvp.base_users(), []
        for number in range(steps):
            if number:
                self.window(block, users)
            predictions, metrics, staged = self.round(block, users, 3 * number)
            self.assert_device_holds(block, staged)
            served.append((predictions, metrics['segments']))
            users = tvp.advanced(users, 2 + number)
        return served

    def reference(self, steps=5):
        clean()
        block = self.open_block(prestage=False)
        users, served = tvp.base_users(), []
        for number in range(steps):
            served.append(self.round(block, users, 3 * number)[:2][0:1][0])
            users = tvp.advanced(users, 2 + number)
        block.close()
        self.block = None
        return served

    def test_with_keyed_the_resident_after_a_keyed_write_is_the_snapshot_and_the_device_is_the_full_stage(self):
        flag_off = [self.reference()]
        block = self.open_diff(keyed=True, full_audit=True)
        served = self.run_rounds(block)
        self.assertEqual([item[0] for item in served], flag_off[0])
        self.assertGreater(round_host.COUNTS['keyed'], 0, 'the key stood on the diff rounds')
        self.assertEqual(self.paths(), ['full'] + ['diff'] * 4)
        self.assertEqual([item[0] for item in self.diff_paths()], ['full'] + ['diff'] * 3)
        audits = self.marked(verify_prestage.FULL_AUDIT_MARKER)
        self.assertTrue(all('mismatches=0' in line for line in audits), audits)

    def test_with_the_lean_writer_the_same_bytes_and_the_same_predictions(self):
        flag_off = self.reference()
        block = self.open_diff(lean=True, audit=True)
        served = self.run_rounds(block)
        self.assertEqual([item[0] for item in served], flag_off)
        self.assertEqual(self.diff_paths()[0][0], 'full')
        self.assertEqual({item[0] for item in self.diff_paths()[1:]}, {'diff'})
        self.assertTrue(all(' exact=True' in line for line in self.diff_lines(prestage_diff.AUDIT_MARKER)))

    def test_with_everything_on_and_a_padded_block(self):
        clean()
        block = self.open_diff(lean=True, keyed=True, audit=True, padded=2)
        users = tvp.base_users()
        self.round(block, users)
        for step in range(1, 4):
            users = tvp.advanced(users, 3)
            live = {name: users[name] for name in 'ACD'} if step == 2 else users
            self.window(block, live, finished=[name for name in NAMES if name not in live])
            predictions, metrics, staged = self.round(block, live, step)
            self.assert_device_holds(block, staged)
        self.assertEqual(self.diff_lines(prestage_diff.MISMATCH_MARKER), [])


class TwoBlocks(thg.TwoRealBlockTests):
    """Two real blocks in blocks mode with the lever: each block's resident survives the other block's round."""

    # the parent class's own tests are not re-run here
    test_both_real_blocks_take_the_diff_path_after_the_first_round_and_predict_what_the_control_predicts = None
    test_audited_both_blocks_read_back_every_destination_with_zero_mismatches = None

    def setUp(self):
        super().setUp()
        clean()
        self.addCleanup(clean)
        for name in (DIFF, AUDIT, LEAN, KEYED):
            os.environ.pop(name, None)

    def run_pair(self, plan, **flags):
        base = {'QWEN_FAST_PRESTAGE': '1', 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE': '1', 'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS': '1', DIFF: '1'}
        base.update(flags)
        blocks = self.open_pair(**base)
        self.assertEqual(verify_prestage.engage_two_block(blocks, dict(os.environ)), 'blocks', self.h1a)
        return blocks

    def block_paths(self):
        found = {}
        for line in self.h1a:
            if line.startswith(prestage_diff.ROUND_MARKER + ' '):
                label, path, reason = re.search(r'block=(\S+) round=\d+ path=(\w+) reason=(\S+)', line).groups()
                found.setdefault(label, []).append((path, reason))
        return found

    def test_each_blocks_resident_survives_the_other_blocks_round(self):
        plan = self.rounds(steps=4)
        expected = self.reference(plan)
        blocks = self.run_pair(plan, QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT='1', **{AUDIT: '1'})
        served = self.drive(blocks, plan, window=True)
        self.assertEqual(served, expected)
        found = self.block_paths()
        self.assertEqual(sorted(found), ['A', 'B'])
        for label in 'AB':
            self.assertEqual(found[label][0], ('full', 'no-resident'))
            self.assertEqual({item for item in found[label][1:]}, {('diff', '-')}, 'block %s: %s' % (label, found[label]))
        fulls = self.marked(verify_prestage.FULL_AUDIT_MARKER)
        self.assertTrue(all('mismatches=0' in line for line in fulls))
        self.assertEqual(self.lines_of(prestage_diff.MISMATCH_MARKER), [])
        self.assertTrue(all(' exact=True' in line for line in self.lines_of(prestage_diff.AUDIT_MARKER)))

    def lines_of(self, marker):
        return [line for line in self.h1a if line.startswith(marker + ' ')]

    def test_an_external_writer_kills_both_residents(self):
        plan = self.rounds(steps=4)
        blocks = self.run_pair(plan)
        self.drive(blocks, plan[:2], window=True)
        verify_prestage.bump('prefill-chunk')
        waiting = [verify_prestage.WhileWaiting(block, self.requests_of(index, plan[2][0][index])) for index, block in enumerate(blocks)]
        for item in waiting:
            verify_prestage.log_line  # noqa: B018 (the lines are captured by the fixture's patch)
            item()
        found = self.block_paths()
        for label in 'AB':
            self.assertEqual(found[label][-1], ('full', 'epoch:prefill-chunk'), found)


class ShippingTests(unittest.TestCase):
    def test_the_module_and_its_markers_are_the_ones_the_manifest_names(self):
        import json

        manifest = json.loads((HERE / 'fusion-wp' / 'WPH.json').read_text(encoding='utf-8'))
        lever = next(item for item in manifest['levers'] if item['id'] == 'prestage-diff')
        self.assertEqual((lever['flag'], lever['audit_flag']), (prestage_diff.FLAG, prestage_diff.AUDIT_FLAG))
        self.assertEqual(lever['marker'], prestage_diff.ENGAGED_MARKER[len('[PINDIAG] '):-len(' engaged')])
        paths = [item['path'] if isinstance(item, dict) else item for item in manifest['image_files']]
        self.assertIn('scripts/ci/prestage_diff.py', paths)

    def test_the_modules_are_imported_only_with_their_flags(self):
        text = (HERE / 'verify_prestage.py').read_text(encoding='utf-8')
        self.assertIn("os.environ.get(PRESTAGE_DIFF_FLAG, '0') != '0'", text)
        self.assertIn("os.environ.get(WRITE_LEAN_FLAG, '0') != '0'", text)
        self.assertNotRegex(text, r'(?m)^import (prestage_diff|write_packed_lean)')


if __name__ == '__main__':
    unittest.main()
