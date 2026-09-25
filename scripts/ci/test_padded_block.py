"""Variable-user packed rounds M2 (QWEN_FAST_PADDED_BLOCK=1): the 64-row block serves two or three
live users as one pass, the missing segments idle on page 0 (packed_verifier.py, VARIABLE-USER
ROUNDS; serving_packed_step.py; serving_runtime.padded_block_admission).

The block is test_packed_verifier's four-user fixture (the real stage_packed and the real per-user
replay readers over a fake ttnn); the step's requests are test_serving_packed_step's, bound to the
real block by the carry they borrowed. R1 (run v188, image P1) measured on hardware what these
fakes cannot: live rows of padded patterns {0,1}, {1,3}, {0,1,2} bit-identical to all-live, idle
carries intact, no page-0 hit, 140.2 ms per replay. Here: that the serving path stages, commits and
refuses exactly as that measurement assumed, that the gate and the report read the lines the
modules actually log, and - against the parent commit (PARENT), module by module - that with the
flag off every path is the one that ran before."""

import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import packed_verifier
from packed_verifier import padded_block_min_users, page_zero_index
import padded_probe
from serving_fast_request import CommittedOutput
import serving_packed_step
from serving_packed_step import ineligible, kv_guard, packed_device_step, proposal_rows
import test_packed_verifier as tpv
import test_padded_probe as tpp
from test_serving_packed_step import FakeBlock, FakeRequest, entry as step_entry
import verifier_engine
import verify_trace_t2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# M2's parent: M1 (ea0813e2) plus two gate-only commits; every module M2 touches is M1's there.
PARENT = 'a01b5f3d'
PAGE_WIDTH = tpv.PAGE_WIDTH
ADMITTED = packed_verifier.PADDED_ADMITTED_MARKER


def parent_module(relative, commit=PARENT):
    """scripts/ci/<relative> at `commit`, loaded beside today's modules (its own imports resolve to
    today's siblings), with its __file__ where today's is; None without git history."""
    try:
        result = subprocess.run(['git', 'show', '%s:scripts/ci/%s' % (commit, relative)], capture_output=True,
                                cwd=str(HERE), timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    module = ModuleType(relative[:-3] + '_parent')
    module.__file__ = str(HERE / relative)
    exec(compile(result.stdout.decode('utf-8'), '%s@%s' % (relative, commit), 'exec'), module.__dict__)
    return module


def clean_environment():
    return patch.dict(os.environ, {name: value for name, value in os.environ.items()
                                   if not name.startswith('QWEN_FAST_')}, clear=True)


class PaddedFixture(tpv.FourUserFixture):
    """The M3 block with every padded-path line captured: the block's (packed_verifier.diagnostic)
    and the step's (serving_packed_step.padded_log, through verify_trace_t2's log_line)."""

    def setUp(self):
        super().setUp()
        environment = clean_environment()
        environment.start()
        self.addCleanup(environment.stop)
        self.lines = []
        for patcher in (patch.object(packed_verifier, 'diagnostic', side_effect=self.lines.append),
                        patch.object(verify_trace_t2, 'log_line', side_effect=self.lines.append)):
            patcher.start()
            self.addCleanup(patcher.stop)
        verify_trace_t2._LOGGED.clear()
        self.addCleanup(verify_trace_t2._LOGGED.clear)

    def padded(self, minimum=2, **options):
        return self.build(padded_min_users=minimum, **options)

    def entries(self, names, rows=16):
        """Entries for the named users (A..D admitted through slots 0..3), in the order named."""
        owners = {name: tpv.request(name, self.pool.slots[index], self.POSITIONS[index], self.PAGES[index])
                  for index, name in enumerate('ABCD')}
        return [tpv.entry(owners[name], range(self.TOKENS['ABCD'.index(name)], self.TOKENS['ABCD'.index(name)] + rows))
                for name in names]

    def marked(self, marker):
        return [line for line in self.lines if line.startswith(marker)]


class FlagTests(unittest.TestCase):
    def test_the_flag_and_its_minimum(self):
        self.assertIsNone(padded_block_min_users({}))
        self.assertIsNone(padded_block_min_users({'QWEN_FAST_PADDED_BLOCK': '0'}))
        self.assertIsNone(padded_block_min_users({'QWEN_FAST_PADDED_BLOCK_MIN_USERS': 'junk'}), 'unread while off')
        self.assertEqual(padded_block_min_users({'QWEN_FAST_PADDED_BLOCK': '1'}), 2)
        self.assertEqual(padded_block_min_users({'QWEN_FAST_PADDED_BLOCK': '1', 'QWEN_FAST_PADDED_BLOCK_MIN_USERS': '3'}), 3)
        for environ in ({'QWEN_FAST_PADDED_BLOCK': 'yes'}, {'QWEN_FAST_PADDED_BLOCK': ''},
                        {'QWEN_FAST_PADDED_BLOCK': '1', 'QWEN_FAST_PADDED_BLOCK_MIN_USERS': 'two'},
                        {'QWEN_FAST_PADDED_BLOCK': '1', 'QWEN_FAST_PADDED_BLOCK_MIN_USERS': '02'},
                        {'QWEN_FAST_PADDED_BLOCK': '1', 'QWEN_FAST_PADDED_BLOCK_MIN_USERS': '-2'}):
            with self.subTest(environ=environ), self.assertRaises(ValueError):
                padded_block_min_users(environ)

    def test_page_zero_index_reads_only_the_used_range(self):
        table = torch.full((1, PAGE_WIDTH), 7, dtype=torch.int32)
        self.assertIsNone(page_zero_index(table, 4100, 16))
        table[0, 65] = 0
        self.assertIsNone(page_zero_index(table, 4100, 16), '(4100 + 16 + 63) // 64 = 65 pages: index 65 is unused')
        table[0, 64] = 0
        self.assertEqual(page_zero_index(table, 4100, 16), 64)
        self.assertEqual(page_zero_index([[3, 0, 5]], 0, 16), None)
        self.assertEqual(page_zero_index([[3, 0, 5]], 64, 16), 1)
        with self.assertRaisesRegex(ValueError, 'cannot map'):
            page_zero_index(torch.full((1, 4), 7, dtype=torch.int32), 4100, 16)


class ConstructionTests(PaddedFixture):
    def test_the_minimum_is_refused_before_anything_is_allocated(self):
        for minimum in (0, 1, 4, 5, '2', 2.0, True):
            with self.subTest(minimum=minimum), self.assertRaisesRegex(ValueError, 'padded_min_users'):
                self.build(padded_min_users=minimum)
        for helper in self.helpers:
            helper.allocate.assert_not_called()
        self.assertEqual((tpv.FakeModelBatch.instances, self.prepared), ([], []))

    def test_two_and_three_are_admitted_and_said_once_at_attach(self):
        block = self.padded(2)
        self.assertEqual([block.pads(count) for count in range(6)], [False, False, True, True, False, False])
        self.assertEqual(self.marked(ADMITTED),
                         ['[PINDIAG] packed padded block admitted min_users=2 users=4 max_idle=2 carries_in_place=0'])
        self.assertEqual(block.describe()['padded'], dict(min_users=2, max_idle=2, carries_in_place=False, rounds=0))
        block.close()
        three = self.padded(3)
        self.assertEqual([three.pads(count) for count in range(5)], [False, False, False, True, False])

    def test_without_it_the_block_pads_nothing_and_says_nothing(self):
        block = self.build()
        self.assertIsNone(block.padded_min_users)
        self.assertFalse(any(block.pads(count) for count in range(6)))
        self.assertEqual(self.lines, [])
        self.assertNotIn('padded', block.describe())

    def test_carries_in_place_means_both_halves_of_cut_3_in_every_gdn_layer(self):
        block = self.padded()
        for counts, expected in ((dict(direct_carry=48, last_carry=48), True), (dict(direct_carry=48, last_carry=0), False),
                                 (dict(direct_carry=47, last_carry=47), False), ({}, False)):
            with self.subTest(counts=counts):
                block.note_verify_t1(counts)
                self.assertIs(block.carries_in_place, expected)

    def test_the_two_user_block_takes_one_idle_segment_at_most(self):
        class TwoUsers(tpv.BlockFixture):
            def runTest(self):
                pass

        fixture = TwoUsers()
        fixture.setUp()
        try:
            with patch.object(packed_verifier, 'diagnostic'):
                with self.assertRaisesRegex(ValueError, r'in \[1, 2\)'):
                    fixture.build(padded_min_users=2)
                self.assertTrue(fixture.build(padded_min_users=1).pads(1))
        finally:
            fixture.doCleanups()


class SegmentsTests(PaddedFixture):
    def test_two_three_and_four_entries_each_take_their_own_segment(self):
        block = self.padded()
        for names, expected in (('CAD', (2, 0, 3)), ('BD', (1, 3)), ('CADB', (2, 0, 3, 1))):
            with self.subTest(names=names):
                self.assertEqual(block.segments(self.entries(names)), expected)

    def test_one_entry_or_one_slot_twice_is_refused(self):
        block = self.padded()
        with self.assertRaisesRegex(ValueError, 'serves 2 to 4 users; 1 entries given'):
            block.segments(self.entries('A'))
        twice = self.entries('AB') + [tpv.entry(tpv.request('E', self.pool.slots[0], 4100, 3), range(16))]
        with self.assertRaisesRegex(ValueError, 'same pool slot'):
            block.segments(twice)
        with self.assertRaisesRegex(ValueError, 'serves 2 to 4 users; 5 entries given'):
            block.segments(self.entries('ABCDA'))

    def test_without_the_flag_every_count_but_four_is_refused_as_before(self):
        block = self.build()
        for names in ('CAD', 'BD', 'A'):
            with self.subTest(names=names), \
                    self.assertRaisesRegex(ValueError, 'serves exactly 4 users; %d entries given' % len(names)):
                block.segments(self.entries(names))


class PaddedRoundTests(PaddedFixture):
    def test_three_live_users_are_served_by_the_one_trace_and_the_idle_segment_commits_nothing(self):
        block = self.padded()
        entries = self.entries('CAD')               # slots 2, 0, 3: segment 1 is idle
        executed, copies = len(self.ttnn.executed), len(self.ttnn.host_copies)
        predictions, metrics = block.verify(entries)
        self.assertEqual(metrics['segments'], (2, 0, 3))
        self.assertEqual((metrics['live'], metrics['idle'], metrics['users']), (3, [1], 4))
        self.assertEqual(predictions, [list(range(1032, 1048)), list(range(1000, 1016)), list(range(1048, 1064))])
        fixture = block.fixture
        for user in (0, 2, 3):
            rows = slice(16 * user, 16 * user + 16)
            self.assertEqual(fixture.tokens.value[rows, 0].tolist(), list(range(self.TOKENS[user], self.TOKENS[user] + 16)))
            self.assertEqual(fixture.positions.value[rows].tolist(),
                             list(range(self.POSITIONS[user], self.POSITIONS[user] + 16)))
            self.assertTrue(bool((fixture.pages.value[rows] == self.PAGES[user]).all()))
        # the idle segment: tokens 1 from the family start, every row on page 0
        first = block.replay_capacity - 256
        self.assertEqual(fixture.tokens.value[16:32, 0].tolist(), [1] * 16)
        self.assertEqual(fixture.positions.value[16:32].tolist(), list(range(first, first + 16)))
        self.assertFalse(bool(fixture.pages.value[16:32].any()))
        self.assertEqual([own.positions.value.tolist()[0] for own in fixture.replay_reader.readers],
                         [4100, first, 4150, 4300])
        # every captured buffer restaged, as in a four-user round, and the one trace once
        self.assertEqual((metrics['staged_buffers'], len(self.ttnn.host_copies) - copies), (149, 149))
        self.assertEqual(self.ttnn.executed[executed:], ['trace1'])
        # the idle segment decided at prefix 0 at once: no trace, no fence
        retained = fixture.retained
        self.assertEqual(retained.commits, [(1, 0)])
        self.assertFalse(retained.commit_user.call_args.kwargs['synchronize'])
        self.assertEqual((block.phase, block.pending_segments, block.idle_segments),
                         ('verified', {0, 2, 3}, frozenset({1})))
        self.assertEqual(self.marked(packed_verifier.PADDED_ROUND_MARKER),
                         ['[PINDIAG] packed padded round live=3 round=1 segments=0,2,3 idle=1 padded=1'])
        # the live commits in entries order; the last live one is the round's fence
        for segment, prefix in zip(metrics['segments'], (9, 16, 4)):
            block.commit_user(segment, prefix)
        self.assertEqual(retained.commits, [(1, 0), (2, 9), (0, 16), (3, 4)])
        self.assertEqual([call.kwargs['synchronize'] for call in retained.commit_user.call_args_list],
                         [False, False, False, True])
        self.assertEqual(self.ttnn.executed[executed + 1:], [block.commits[2][9], block.commits[0][16], block.commits[3][4]])
        self.assertEqual((block.phase, block.rounds, block.padded_rounds), ('idle', 1, 1))

    def test_two_live_users_leave_two_idle_segments_on_their_own_tile_rows_and_the_next_round_replays(self):
        block = self.padded()
        predictions, metrics = block.verify(self.entries('DA'))       # segments 3, 0: 1 and 2 idle
        self.assertEqual((metrics['segments'], metrics['idle']), ((3, 0), [1, 2]))
        first = block.replay_capacity - 256
        positions = block.fixture.positions.value
        self.assertEqual(positions[16:32].tolist(), list(range(first, first + 16)))
        self.assertEqual(positions[32:48].tolist(), list(range(first + 32, first + 48)))
        retained = block.fixture.retained
        self.assertEqual(retained.commits, [(1, 0), (2, 0)])
        for segment, prefix in zip(metrics['segments'], (5, 0)):
            block.commit_user(segment, prefix)
        self.assertEqual([call.kwargs['synchronize'] for call in retained.commit_user.call_args_list],
                         [False, False, False, True], 'A at prefix 0 is still the fence')
        self.assertEqual(block.phase, 'idle')
        # all four live again: the same trace, through the retained block's replay
        predictions, metrics = block.verify(self.entries('ABCD'))
        retained.replay.assert_called_once()
        self.assertEqual((metrics['live'], metrics['idle'], block.idle_segments), (4, [], frozenset()))
        self.assertEqual(len(self.marked(packed_verifier.PADDED_ROUND_MARKER)), 1, 'an all-live round is not padded')

    def test_an_idle_segment_never_commits_a_prefix(self):
        block = self.padded()
        block.verify(self.entries('CAD'))
        executed = len(self.ttnn.executed)
        with self.assertRaisesRegex(ValueError, 'Idle segment 1 of a padded round commits nothing'):
            block.commit_user(1, 3)
        self.assertEqual(self.ttnn.executed[executed:], [])
        self.assertEqual(self.marked(packed_verifier.PADDED_IDLE_COMMIT_MARKER),
                         ['[PINDIAG] packed padded idle commit refused segment=1 prefix=3 round=1'])
        self.assertEqual(block.phase, 'verified', 'the round goes on')
        for segment in (2, 0, 3):
            block.commit_user(segment, 1)
        self.assertEqual(block.phase, 'idle')

    def test_the_audit_line_closes_with_live_and_idle_and_every_parser_still_reads_it(self):
        import acceptance_report
        from lever_n_m3native_profile_report import PACKED_PHASE_RE

        block = self.padded()
        with patch.dict(os.environ, {'QWEN_FAST_PACKED_AUDIT': '1'}):
            predictions, metrics = block.verify(self.entries('CAD'))
            for segment in metrics['segments']:
                block.commit_user(segment, 1)
            block.verify(self.entries('ABCD'))
        phases = self.marked('[PACKED-PHASE]')
        self.assertTrue(phases[0].endswith(' live=3 idle=1') and phases[1].endswith(' live=4 idle=-'), phases)
        for line in phases:
            self.assertIsNotNone(PACKED_PHASE_RE.search(line))
            self.assertIsNotNone(acceptance_report.PACKED_PHASE_LINE.search(line))
        self.assertEqual([acceptance_report.PACKED_PHASE_LIVE.search(line).groups()[2:] for line in phases],
                         [('3', '1'), ('4', '-')])

    def test_without_the_flag_the_audit_line_and_the_metrics_are_todays(self):
        block = self.build()
        with patch.dict(os.environ, {'QWEN_FAST_PACKED_AUDIT': '1'}):
            predictions, metrics = block.verify(self.entries('CADB'))
        self.assertNotIn('live', metrics)
        self.assertNotIn('idle', metrics)
        (line,) = self.marked('[PACKED-PHASE]')
        self.assertRegex(line, r'readback_ms=[0-9.]+$')

    def test_the_probe_does_not_probe_a_padded_round(self):
        with patch.dict(os.environ, {'QWEN_FAST_PADDED_PROBE': '1'}):
            block = self.padded()
        padded_probe._STATE.update(rounds=0, hits=0)
        with patch.object(padded_probe, 'log_line', side_effect=self.lines.append):
            for _ in range(2):
                predictions, metrics = block.verify(self.entries('CADB'))
                for segment in metrics['segments']:
                    block.commit_user(segment, 2)
            executed = len(self.ttnn.executed)
            predictions, metrics = block.verify(self.entries('CAD'))
        probe = [line for line in self.lines if line.startswith(padded_probe.MARKER + ' round=')]
        self.assertEqual(probe, ['[PINDIAG] padded probe round=3 live=0,2,3 exact=refused trace_ms=- idle_carry_intact=- '
                                 'idle=1 differ=- reason=a_padded_round'])
        self.assertEqual(self.ttnn.executed[executed:], ['trace1'], 'no pattern replay')
        self.assertEqual(padded_probe._STATE, dict(rounds=3, hits=0), 'the idle segment is no page-0 hit')


class SeparabilityTests(tpp.ProbeFixture):
    """test_padded_probe's device model: a replay computes every output row from what is staged.
    Separable, the live rows of a padded round are the all-live round's; coupled, they are not -
    which is what R1's G-pad probe measured on hardware for the real trace."""

    def setUp(self):
        super().setUp()
        patcher = patch.object(packed_verifier, 'diagnostic')
        patcher.start()
        self.addCleanup(patcher.stop)

    def serve(self, block, names):
        entries = [item for item in self.four(order=(0, 1, 2, 3)) if item['request_id'] in names]
        predictions, metrics = block.verify(entries)
        for segment in metrics['segments']:
            block.commit_user(segment, 2)
        return dict(zip((item['request_id'] for item in entries), predictions))

    def test_live_rows_of_every_padded_pattern_are_the_all_live_rows(self):
        block = self.build(padded_min_users=2)
        self.model_hook = tpp.DeviceModel(self, block)
        everyone = self.serve(block, 'ABCD')
        carries = [[value.value.clone() for snapshot in carry for value in snapshot] for carry in block.carries]
        for live in ('AB', 'BD', 'ABC', 'ACD'):
            with self.subTest(live=live):
                self.assertEqual(self.serve(block, live), {name: everyone[name] for name in live})
        self.assertTrue(all(torch.equal(mine, theirs) for before, carry in zip(carries, block.carries)
                            for mine, theirs in zip(before, [value.value for snapshot in carry for value in snapshot])))
        # a trace coupling rows would show in the live rows
        self.model_hook.cross_row = True
        everyone = self.serve(block, 'ABCD')
        self.assertNotEqual(self.serve(block, 'AB'), {name: everyone[name] for name in 'AB'})


class RefusalTests(PaddedFixture):
    @staticmethod
    def table(page, width=PAGE_WIDTH):
        return torch.full((1, width), page, dtype=torch.int32)

    def test_the_rules(self):
        block = self.padded()
        live = [(None, 4100, self.table(7)), None, (None, 4150, self.table(13)), (None, 4300, self.table(17))]
        self.assertIsNone(block.padded_refusal(live))
        self.assertIsNone(block.padded_refusal([live[0], None, None, live[3]]))
        self.assertEqual(block.padded_refusal([live[0], None, None, None]), 'padded round of 1 live users outside [2, 4)')
        self.assertEqual(block.padded_refusal([live[0], live[0], live[2], live[3]]),
                         'padded round of 4 live users outside [2, 4)')
        self.assertEqual(block.padded_refusal(live[:3]), 'padded round of 2 live users outside [2, 4)')
        zero = self.table(7)
        zero[0, 64] = 0
        self.assertEqual(block.padded_refusal([(None, 4100, zero), None, live[2], live[3]]),
                         'page0 in a live table: segment 0 position 4100 page_index 64')
        for broken in (self.table(7, width=10), None, 'table'):
            with self.subTest(table=broken):
                self.assertTrue(block.padded_refusal([(None, 4100, broken), None, live[2], live[3]])
                                .startswith('page0 unmapped: segment 0 position 4100'))
        # the idle slot rule
        self.pool.slots[1].lent = True
        self.assertEqual(block.padded_refusal(live), 'idle segment 1: pool slot 1 is lent and the verify trace moves '
                                                     'carries (QWEN_FAST_VERIFY_T1 #3 not in every layer)')
        block.carries_in_place = True
        self.assertIsNone(block.padded_refusal(live))

    def test_page_zero_in_a_live_table_is_refused_before_anything_is_staged(self):
        block = self.padded()
        entries = self.entries('CAD')
        entries[1]['request'].engine.pages[0, 64] = 0              # A at 4100 reads pages [0, 65)
        copies, executed = len(self.ttnn.host_copies), len(self.ttnn.executed)
        with self.assertRaisesRegex(ValueError, 'page0 in a live table: segment 0 position 4100 page_index 64'):
            block.verify(entries)
        self.assertEqual((block.phase, len(self.ttnn.host_copies), len(self.ttnn.executed)), ('idle', copies, executed))
        self.assertEqual(self.marked(packed_verifier.PADDED_PAGE0_MARKER),
                         ['[PINDIAG] packed padded page0 site=verify page0 in a live table: segment 0 position 4100 '
                          'page_index 64'])
        for item in entries:
            item['request'].session.fail_verification.assert_not_called()
        # past the used range, and in an all-live round (no idle writer), page 0 is no hazard
        entries = self.entries('CAD')
        entries[1]['request'].engine.pages[0, 65:] = 0
        predictions, metrics = block.verify(entries)
        for segment in metrics['segments']:
            block.commit_user(segment, 0)
        entries = self.entries('CADB')
        entries[1]['request'].engine.pages[0, 64] = 0
        block.verify(entries)
        self.assertEqual(len(self.marked(packed_verifier.PADDED_PAGE0_MARKER)), 1)

    def test_a_lent_idle_slot_is_refused_unless_the_trace_reads_every_carry_in_place(self):
        block = self.padded()
        self.pool.slots[1].lent = True
        with self.assertRaisesRegex(ValueError, 'idle segment 1: pool slot 1 is lent'):
            block.verify(self.entries('CAD'))
        self.assertEqual(len(self.marked(packed_verifier.PADDED_REFUSED_MARKER + ' site=verify idle segment 1')), 1)
        self.assertEqual(block.phase, 'idle')
        block.carries_in_place = True
        predictions, metrics = block.verify(self.entries('CAD'))
        self.assertEqual(metrics['idle'], [1])


class StepFixture(PaddedFixture):
    """The step over the real padded block, with test_serving_packed_step's requests: sessions,
    engines and runtimes with the state machines of the real ones, each engine trimmed to the
    sequential widths (1, 2, 4) as beside the real 64-row block."""

    def setUp(self):
        super().setUp()
        self.ids.value = torch.arange(0, 64, dtype=torch.int32)
        self.stepped = []

    def owners(self, names='ABCD'):
        owners = {}
        for index, name in enumerate('ABCD'):
            owner = FakeRequest(name, self.POSITIONS[index], self.stepped, carry=self.pool.slots[index].verifier.carry)
            owner.engine.pages = torch.full((1, PAGE_WIDTH), self.PAGES[index], dtype=torch.int32)
            owner.engine.widths = (1, 2, 4)
            owners[name] = owner
        return [owners[name] for name in names]

    def draft(self, owners, rows=16, accepts=None):
        for owner in owners:
            index = 'ABCD'.index(owner.session.request_id)
            owner.propose(list(range(16 * index, 16 * index + 16)), accept=(accepts or {}).get(owner.session.request_id, 3),
                          rows=rows)
        return [step_entry(owner) for owner in owners]


class ProposalRowsTests(StepFixture):
    def test_two_to_four_live_requests_draft_at_the_blocks_width(self):
        block = self.padded()
        a, b, c, d = self.owners()
        for live, expected in (([a, b, c, d], 16), ([c, a, d], 16), ([b, d], 16), ([a], None)):
            with self.subTest(live=[owner.session.request_id for owner in live]):
                self.assertEqual(proposal_rows(block, live), expected)
        d.session.finished = True
        self.assertEqual(proposal_rows(block, [a, b, c, d]), 16, 'a finished request is not live')
        # the per-request checks stand: a block round of budget, the family, a bound engine
        c.session.emitted = [1] * 241
        self.assertIsNone(proposal_rows(block, [a, b, c]))
        c.session.emitted = []
        a.session.position = 4340
        self.assertIsNone(proposal_rows(block, [a, b, c]))
        a.session.position = 4100
        stranger = FakeRequest('E', 4100, self.stepped, carry=tpv.snapshot_set(self.ttnn))
        self.assertIsNone(proposal_rows(block, [a, b, stranger]))
        self.assertEqual(self.lines[1:], [], 'nothing padded-specific refused')

    def test_min_users_three_drafts_two_live_requests_narrow(self):
        block = self.padded(3)
        a, b, c, d = self.owners()
        self.assertEqual(proposal_rows(block, [a, b, c]), 16)
        self.assertIsNone(proposal_rows(block, [a, b]))

    def test_without_the_flag_only_four_live_requests_draft_at_the_blocks_width(self):
        block = self.build()
        a, b, c, d = self.owners()
        self.assertEqual(proposal_rows(block, [a, b, c, d]), 16)
        for live in ([a, b, c], [b, d], [a]):
            self.assertIsNone(proposal_rows(block, live))
        self.assertEqual(self.lines, [])

    def test_page_zero_in_a_live_table_drafts_the_round_narrow_and_says_so_once(self):
        block = self.padded()
        a, b, c, d = self.owners()
        a.engine.pages[0, 64] = 0
        self.assertIsNone(proposal_rows(block, [a, b, c]))
        self.assertIsNone(proposal_rows(block, [a, b, c]), 'every tick')
        self.assertEqual(self.marked(packed_verifier.PADDED_PAGE0_MARKER),
                         ['[PINDIAG] packed padded page0 site=proposal_rows page0 in a live table: segment 0 '
                          'position 4100 page_index 64'])
        self.assertEqual(proposal_rows(block, [a, b, c, d]), 16, 'all four live: nothing writes page 0')

    def test_a_lent_idle_slot_drafts_narrow_unless_the_trace_reads_every_carry_in_place(self):
        block = self.padded()
        a, b, c, d = self.owners()
        self.pool.slots[3].lent = True
        self.assertIsNone(proposal_rows(block, [a, b, c]))
        self.assertEqual(len(self.marked(packed_verifier.PADDED_REFUSED_MARKER + ' site=proposal_rows idle segment 3')), 1)
        block.carries_in_place = True
        self.assertEqual(proposal_rows(block, [a, b, c]), 16)


class KvGuardTests(StepFixture):
    """QWEN_FAST_VERIFY_T2 #2 beside a padded round: the idle segments' page-0 tile rows are written
    in the same chain as the live ones, so the guard checks them - where the parent's guard returned
    None for any round with an empty segment (the regression)."""

    def test_the_idle_segments_are_filled_before_the_tile_rows_are_checked(self):
        block = self.padded()
        block.fixture.kv_chains = True
        a, b, c, d = self.owners()
        self.assertIsNone(kv_guard([(a.engine, 4100), (b.engine, 4200), (c.engine, 4150)], block))
        # C writing its rows at the family start through page 0 shares tile row 0 of page 0 with
        # the idle segment 3 (the j = 0 idle segment starts there)
        c.engine.pages[0, 64] = 0
        owners = [(a.engine, 4100), (b.engine, 4200), (c.engine, 4096)]
        self.assertEqual(kv_guard(owners, block), 'verify t2 kv tile rows shared: users 2,3 page 0 tile row 0')
        parent = parent_module('serving_packed_step.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        self.assertIsNone(parent.kv_guard(owners, block), 'the parent skipped the round unchecked')

    def test_two_owners_through_one_segment_is_a_reason_in_a_padded_round(self):
        block = self.padded()
        block.fixture.kv_chains = True
        a, b, c, d = self.owners()
        twin = FakeRequest('E', 4100, self.stepped, carry=self.pool.slots[0].verifier.carry)
        twin.engine.pages = a.engine.pages
        self.assertEqual(kv_guard([(a.engine, 4100), (twin.engine, 4100), (b.engine, 4200)], block),
                         'verify t2 kv padded round with two owners through one segment')

    def test_a_padded_round_with_disjoint_rows_keeps_the_chained_write(self):
        block = self.padded()
        block.fixture.kv_chains = True
        a, b, c, d = self.owners()
        self.assertEqual(proposal_rows(block, [a, b, c]), 16)
        entries = self.draft([c, a, b])
        self.assertIsNone(ineligible(entries, block))
        self.assertEqual([line for line in self.lines if verify_trace_t2.KV_SHARED in line], [])


class StepTests(StepFixture):
    def test_a_padded_round_is_one_pass_for_the_live_requests(self):
        block = self.padded()
        a, b, c, d = self.owners()
        self.assertEqual(proposal_rows(block, [a, b, c]), 16)
        entries = self.draft([c, a, b], accepts=dict(A=15, B=9, C=0))
        executed = len(self.ttnn.executed)
        outputs = packed_device_step(entries, cancelled=lambda: False, block=block)
        self.assertEqual(self.stepped, [], 'not the sequential step')
        self.assertEqual(outputs, [CommittedOutput('C', (32,), 4151, False),
                                   CommittedOutput('A', tuple(range(0, 16)), 4116, False),
                                   CommittedOutput('B', tuple(range(16, 26)), 4210, False)])
        retained = block.fixture.retained
        self.assertEqual(retained.commits, [(3, 0), (2, 1), (0, 16), (1, 10)])
        self.assertEqual([call.kwargs['synchronize'] for call in retained.commit_user.call_args_list],
                         [False, False, False, True])
        self.assertEqual(self.ttnn.executed[executed:], ['trace1', block.commits[2][1], block.commits[0][16],
                                                         block.commits[1][10]])
        self.assertEqual(block.fixture.tokens.value[48:, 0].tolist(), [1] * 16, "D's segment staged idle")
        for owner, segment in ((a, 0), (b, 1), (c, 2)):
            self.assertEqual([seg for ticket, unused, seg in owner.engine.adopted], [segment])
            self.assertEqual(owner.session.phase, 'idle')
        self.assertEqual((block.phase, block.rounds, block.padded_rounds), ('idle', 1, 1))
        self.assertIsNone(verifier_engine._resident)

    def test_the_step_backstop_refuses_a_page_zero_round_drafted_before_it_was_seen(self):
        block = self.padded()
        a, b, c, d = self.owners()
        entries = self.draft([c, a, b])
        a.engine.pages[0, 64] = 0
        with patch.dict(sys.modules, loguru=None), patch('sys.stdout', new_callable=io.StringIO) as out:
            reason = ineligible(entries, block)
            outputs = packed_device_step(entries, cancelled=lambda: False, block=block)
        self.assertEqual(reason, 'page0 in a live table: segment 0 position 4100 page_index 64')
        self.assertEqual(self.marked(packed_verifier.PADDED_PAGE0_MARKER),
                         ['[PINDIAG] packed padded page0 site=ineligible ' + reason] * 2)
        self.assertIn('A round the block cannot serve (page0 in a live table', out.getvalue())
        self.assertTrue(all(output.finished and output.cancelled for output in outputs))
        self.assertEqual(block.fixture.retained.commits, [], 'nothing was staged or committed')

    def test_a_padded_round_whose_publication_fails_leaves_no_segment_undecided(self):
        block = self.padded()
        a, b, c, d = self.owners()
        b.runtime.fail = RuntimeError('publication failed')
        entries = self.draft([c, a, b], accepts=dict(A=15, B=9, C=0))
        with self.assertRaisesRegex(RuntimeError, 'publication failed'):
            packed_device_step(entries, cancelled=lambda: False, block=block)
        retained = block.fixture.retained
        self.assertEqual(retained.commits, [(3, 0), (2, 1), (0, 16), (1, 0)], 'fail_round released B at prefix 0')
        self.assertTrue(retained.commit_user.call_args.kwargs['synchronize'])
        self.assertEqual((block.phase, block.pending_segments), ('idle', set()))
        self.assertEqual(b.session.phase, 'failed')
        # and the next round is served
        for owner in (a, c):
            owner.session.pending, owner.session.phase = None, 'idle'
        entries = self.draft([a, c])
        self.assertEqual([output.request_id for output in packed_device_step(entries, cancelled=lambda: False, block=block)],
                         ['A', 'C'])

    def test_a_round_it_did_not_pad_says_why_and_whether_it_could_have(self):
        block = self.padded()
        a, b, c, d = self.owners()
        entries = self.draft([c, a, b], rows=4)                    # drafted narrow
        packed_device_step(entries, cancelled=lambda: False, block=block)
        self.assertEqual([item[0] for item in self.stepped], ['C', 'A', 'B'])
        self.assertEqual(self.marked(packed_verifier.PADDED_SKIPPED_MARKER),
                         ['[PINDIAG] packed padded skipped live=3 eligible=1 reason=request=C_rows=4_rows_per_user=16'])
        # a user under a block round of budget, by its session or by vLLM's count: not eligible
        for narrow in ('session', 'bridge'):
            with self.subTest(narrow=narrow):
                self.lines.clear()
                a, b, c, d = self.owners()
                entries = self.draft([c, a, b], rows=4)
                if narrow == 'session':
                    a.session.emitted = [1] * 241
                else:
                    entries[1]['bridge'] = SimpleNamespace(state=SimpleNamespace(
                        sampling_params=SimpleNamespace(max_tokens=256), output_token_ids=[0] * 241))
                packed_device_step(entries, cancelled=lambda: False, block=block)
                self.assertEqual([line.split(' reason=')[0] for line in self.marked(packed_verifier.PADDED_SKIPPED_MARKER)],
                                 ['[PINDIAG] packed padded skipped live=3 eligible=0'])
        # one live user and four narrow ones are no padded count: nothing said
        self.lines.clear()
        packed_device_step(self.draft(self.owners('A'), rows=4), cancelled=lambda: False, block=block)
        packed_device_step(self.draft(self.owners('BCDA'), rows=4), cancelled=lambda: False, block=block)
        self.assertEqual(self.lines, [])

    def test_without_the_flag_a_three_survivor_round_is_todays_sequential_step_and_says_nothing(self):
        block = self.build()
        a, b, c, d = self.owners()
        packed_device_step(self.draft([c, a, b], rows=4), cancelled=lambda: False, block=block)
        self.assertEqual([item[0] for item in self.stepped], ['C', 'A', 'B'])
        self.assertEqual(self.lines, [])


class GateTests(StepFixture):
    """lever_n_m3native_gate over the lines the block and the step actually log."""

    ENV = {'QWEN_FAST_PADDED_BLOCK': '1'}

    def run_lines(self):
        """One admission, one padded round, one eligible skip and one ineligible one, as logged."""
        block = self.padded()
        packed_device_step(self.draft(self.owners('CAB')), cancelled=lambda: False, block=block)
        packed_device_step(self.draft(self.owners('CAB'), rows=4), cancelled=lambda: False, block=block)
        narrow = self.owners('CAB')
        narrow[1].session.emitted = [1] * 241
        packed_device_step(self.draft(narrow, rows=4), cancelled=lambda: False, block=block)
        self.assertEqual([item[0] for item in self.stepped], ['C', 'A', 'B'] * 2)
        return list(self.lines)

    def report(self, lines, environ=None):
        from lever_n_m3native_gate import flag_marker_report
        return flag_marker_report(dict(self.ENV if environ is None else environ), 4, chr(10).join(lines))

    def test_the_gate_reads_the_names_the_modules_use(self):
        import lever_n_m3native_gate as gate
        import serving_runtime

        for name in ('PADDED_BLOCK_FLAG', 'PADDED_MIN_USERS_FLAG', 'PADDED_MIN_USERS_DEFAULT', 'PADDED_ADMITTED_MARKER',
                     'PADDED_ROUND_MARKER', 'PADDED_SKIPPED_MARKER', 'PADDED_REFUSED_MARKER', 'PADDED_PAGE0_MARKER',
                     'PADDED_IDLE_COMMIT_MARKER'):
            with self.subTest(name=name):
                self.assertEqual(getattr(gate, name), getattr(packed_verifier, name))
        self.assertEqual(serving_runtime.PADDED_BLOCK_FLAG, packed_verifier.PADDED_BLOCK_FLAG)
        self.assertTrue(gate.REFUSED_ROUND_MARKER in open(HERE / 'serving_packed_step.py', encoding='utf-8').read())

    def test_the_logged_lines_pass_and_are_summarised(self):
        lines = self.run_lines()
        good = lines + [lines[[index for index, line in enumerate(lines)
                               if line.startswith(packed_verifier.PADDED_ROUND_MARKER)][0]]] * 3
        report = self.report(good)
        self.assertEqual(report['missing'], [])
        summary = report['padded_block']
        self.assertEqual(summary['admitted'], dict(min_users=2, users=4, max_idle=2, carries_in_place=False))
        self.assertEqual((summary['padded_rounds'], summary['padded_by_live'], summary['skipped_eligible'],
                          summary['skipped_ineligible'], summary['eligible_rounds'], summary['engagement']),
                         (4, {'3': 4}, 1, 1, 5, 0.8))
        self.assertEqual(summary['skip_reasons'], {'eligible:request': 1, 'ineligible:request': 1})

    def test_under_the_engagement_floor_the_run_fails(self):
        report = self.report(self.run_lines())
        self.assertEqual(report['missing'], ['QWEN_FAST_PADDED_BLOCK: padded rounds 1 of 2 eligible (0.50), under 0.80'])

    @staticmethod
    def served(lines):
        """The admission and the padded round of run_lines: the least a passing M2 arm carries."""
        return [line for line in lines if line.startswith((ADMITTED, packed_verifier.PADDED_ROUND_MARKER))]

    def test_the_admission_is_required_and_its_minimum_must_be_the_one_asked_for(self):
        lines = self.run_lines()
        admitted = [line for line in lines if line.startswith(ADMITTED)]
        rest = [line for line in lines if not line.startswith(ADMITTED)]
        self.assertIn('QWEN_FAST_PADDED_BLOCK: %s' % ADMITTED, self.report(rest)['missing'])
        missing = self.report(self.served(lines), dict(self.ENV, QWEN_FAST_PADDED_BLOCK_MIN_USERS='3'))['missing']
        self.assertEqual(missing, ['QWEN_FAST_PADDED_BLOCK_MIN_USERS: the block admitted min_users=2, not the 3 asked for'])
        self.assertEqual(self.report(admitted, {'QWEN_FAST_PADDED_BLOCK_MIN_USERS': '2'})['missing'],
                         ['QWEN_FAST_PADDED_BLOCK_MIN_USERS=2 without QWEN_FAST_PADDED_BLOCK=1 pads nothing'])
        self.assertNotIn('padded_block', self.report(admitted, {}))

    def test_an_arm_that_never_padded_a_round_fails(self):
        # Built and admitted, but no round served padded (nor any skip logged): the arm has not tested
        # M2, whatever else it logged.
        lines = self.run_lines()
        admitted = [line for line in lines if line.startswith(ADMITTED)]
        self.assertEqual(self.report(admitted)['missing'],
                         ['QWEN_FAST_PADDED_BLOCK: %s' % packed_verifier.PADDED_ROUND_MARKER])
        self.assertEqual(len(self.served(lines)), 2)
        self.assertEqual(self.report(self.served(lines))['missing'], [])

    def test_a_page_zero_line_an_idle_commit_a_backstop_or_a_refused_round_fails(self):
        admitted = self.served(self.run_lines())
        cases = {
            'page0': '[PINDIAG] packed padded page0 site=proposal_rows page0 in a live table: segment 0 position 4100 '
                     'page_index 64',
            'idle commit': '[PINDIAG] packed padded idle commit refused segment=1 prefix=3 round=7',
            'backstop': '[PINDIAG] packed padded refused site=ineligible idle segment 1: pool slot 1 is lent',
            'refused round': '[PACKED] A round the block cannot serve (x) holds tickets no request engine captured (y)'}
        for name, line in cases.items():
            with self.subTest(case=name):
                missing = self.report(admitted + [line])['missing']
                self.assertEqual(len(missing), 1, missing)
                self.assertTrue(missing[0].startswith('QWEN_FAST_PADDED_BLOCK: '), missing)
        # the idle slot rule refusing a round before it is drafted is a skip, not a failure
        self.assertEqual(self.report(admitted + ['[PINDIAG] packed padded refused site=proposal_rows idle segment 1: '
                                                 'pool slot 1 is lent'])['missing'], [])


class AcceptanceReportTests(unittest.TestCase):
    LINES = ['2026-09-24 03:18:37.311 | INFO | [PACKED-PHASE] round=30 users=4 bind_ms=1.00 input_ms=2.00 '
             'trace_ms=140.20 sync_ms=0.10 readback_ms=3.00 live=4 idle=-',
             '[PACKED-PHASE] round=31 users=4 bind_ms=1.00 input_ms=2.00 trace_ms=140.20 sync_ms=0.10 readback_ms=3.00 '
             'live=3 idle=1',
             '[PINDIAG] packed padded round live=3 round=31 segments=0,2,3 idle=1 padded=1',
             '[PACKED-PHASE] round=32 users=4 bind_ms=1.00 input_ms=2.00 trace_ms=140.20 sync_ms=0.10 readback_ms=3.00 '
             'live=2 idle=1,2',
             '[PINDIAG] packed padded round live=2 round=32 segments=0,3 idle=1,2 padded=2',
             '[PINDIAG] packed padded skipped live=2 eligible=0 reason=request=A_rows=4_rows_per_user=16',
             '[PINDIAG] packed padded skipped live=3 eligible=1 reason=request=C_rows=4_rows_per_user=16']

    def test_the_padded_rounds_by_live_count_and_their_engagement(self):
        import acceptance_report as ar

        self.assertEqual(ar.padded_rounds(chr(10).join(self.LINES)),
                         dict(packed_by_live={'4': 1, '3': 1, '2': 1}, padded={'3': 1, '2': 1}, eligible_skipped=1,
                              ineligible_skipped=1, engagement=0.6667))
        report = ar.report(chr(10).join(self.LINES), [])
        self.assertEqual(report['rounds']['packed_rounds'], 3)
        self.assertEqual(report['rounds']['padded'], {'3': 1, '2': 1})
        self.assertTrue(report['summary_line'].endswith(' | padded 3:1 2:1 of 3 eligible (0.67)'), report['summary_line'])

    def test_a_run_without_the_flag_keeps_its_report(self):
        import acceptance_report as ar
        from test_acceptance_report import fixture

        log, streams = fixture('v185')
        self.assertIsNone(ar.padded_rounds(log))
        report = ar.report(log, streams)
        self.assertFalse({'packed_by_live', 'padded', 'engagement'} & set(report['rounds']))
        self.assertNotIn(' | padded', report['summary_line'])


class ParentTests(unittest.TestCase):
    """With the flag off, each module M2 touches against its PARENT copy: the same calls, the same
    answers, the same report (skipped without git history)."""

    def parent(self, name):
        module = parent_module(name)
        if module is None:
            self.skipTest('no git history for %s' % PARENT)
        return module

    def test_the_acceptance_report_of_a_flag_off_run_is_the_parents(self):
        from test_acceptance_report import fixture
        import acceptance_report

        parent = self.parent('acceptance_report.py')
        for tag in ('v155', 'v185'):
            with self.subTest(tag=tag):
                log, streams = fixture(tag)
                self.assertEqual(acceptance_report.report(log, streams), parent.report(log, streams))

    def test_the_gates_marker_report_is_the_parents(self):
        import lever_n_m3native_gate as gate
        from test_acceptance_report import fixture

        parent = self.parent('lever_n_m3native_gate.py')
        log, streams = fixture('v185')
        for environ in ({}, {'QWEN_FAST_PAIR_MASK_REFRESH': '1', 'QWEN_FAST_PADDED_PROBE': '1'},
                        {'QWEN_FAST_VERIFY_T2': '1', 'QWEN_FAST_GDN_USER_BATCH': '1'}):
            with self.subTest(environ=environ):
                self.assertEqual(gate.flag_marker_report(environ, 4, log), parent.flag_marker_report(environ, 4, log))

    def step_scenarios(self, module):
        """The step's answers and rounds over test_serving_packed_step's fake block, flag off."""
        verifier_engine.note_prefill()
        stepped, results = [], []
        block = FakeBlock(users=4)

        def owners(widths=(1, 2, 4, 8, 16)):
            made = []
            for segment, name in enumerate('ABCD'):
                request = FakeRequest(name, 4100 + 50 * segment, stepped)
                request.engine.widths = widths
                request.engine.pages = torch.full((1, PAGE_WIDTH), 7 + segment, dtype=torch.int32)
                block.bind(request.engine, segment)
                made.append(request)
            return made

        requests = owners()
        answers = [module.proposal_rows(block, live) for live in (requests, requests[:3], requests[:2], requests[:1])]
        requests[3].session.finished = True
        answers.append(module.proposal_rows(block, requests))
        results.append(answers)
        requests = owners()
        for segment, request in enumerate(requests):
            request.propose(block.predictions_for(segment), accept=4 * segment)
        entries = [step_entry(requests[index]) for index in (2, 0, 3, 1)]
        results.append((module.packed_device_step(entries, cancelled=lambda: False, block=block), list(block.calls)))
        for rows in (4, 16):
            requests = owners(widths=(1, 2, 4))
            for segment, request in enumerate(requests[:3]):
                request.propose(block.predictions_for(segment), accept=1, rows=rows)
            stepped.clear()
            with patch.dict(sys.modules, loguru=None), patch('sys.stdout', new_callable=io.StringIO) as out:
                outputs = module.packed_device_step([step_entry(request) for request in requests[:3]],
                                                    cancelled=lambda: False, block=block)
            results.append((outputs, list(stepped), out.getvalue(), [request.session.phase for request in requests[:3]]))
        block.kv_chains = True
        requests = owners()
        requests[2].session.position = requests[2].engine.position = 4110
        requests[2].engine.pages = requests[0].engine.pages
        with patch.object(verify_trace_t2, 'log_line') as logged:
            verify_trace_t2._LOGGED.clear()
            results.append((module.proposal_rows(block, requests), [call.args for call in logged.call_args_list]))
            verify_trace_t2._LOGGED.clear()
        return results

    def test_the_steps_answers_and_rounds_are_the_parents(self):
        parent = self.parent('serving_packed_step.py')
        with clean_environment():
            self.assertEqual(self.step_scenarios(serving_packed_step), self.step_scenarios(parent))

    def test_the_real_blocks_round_is_the_parents(self):
        """The verifier and the step together over the real block, flag off: the same traces, the same
        commits, the same outputs as the parent's two modules."""
        parent_verifier = self.parent('packed_verifier.py')
        parent_step = self.parent('serving_packed_step.py')

        class Fixture(tpv.FourUserFixture):
            def runTest(self):
                pass

        def scenario(verifier_module, step_module):
            fixture = Fixture()
            fixture.setUp()
            try:
                fixture.ids.value = torch.arange(0, 64, dtype=torch.int32)
                with patch.object(verifier_module, 'ModelBatch', tpv.FakeModelBatch), \
                        patch.object(verifier_module, 'PreparedTargetFeatures', tpv.FakeFeatures), \
                        patch.object(verifier_module, 'prepare', packed_verifier.prepare), \
                        patch.object(verifier_module, 'capture_operation', packed_verifier.capture_operation), \
                        patch.object(verifier_module, 'sample_rows', packed_verifier.sample_rows):
                    block = verifier_module.PackedVerifierEngine(
                        fixture.ttnn, fixture.model, fixture.helpers, 'sampler', pool=fixture.pool,
                        shared_weights=fixture.weights, shape=fixture.shape(), feature_taps=tpv.TAPS)
                    stepped, results = [], []
                    owners = []
                    for index, name in enumerate('ABCD'):
                        owner = FakeRequest(name, fixture.POSITIONS[index], stepped,
                                            carry=fixture.pool.slots[index].verifier.carry)
                        owner.engine.pages = torch.full((1, PAGE_WIDTH), fixture.PAGES[index], dtype=torch.int32)
                        owner.engine.widths = (1, 2, 4)
                        owners.append(owner)
                    for rows, live in ((16, 'ABCD'), (4, 'CAB'), (16, 'DCBA'), (4, 'BD')):
                        chosen = [owners['ABCD'.index(name)] for name in live]
                        for owner in chosen:
                            # the fake sequential step leaves its ticket pending: a fresh one each round
                            owner.session.pending, owner.session.phase = None, 'idle'
                            index = 'ABCD'.index(owner.session.request_id)
                            owner.propose(list(range(16 * index, 16 * index + 16)), accept=index * 3, rows=rows)
                        outputs = step_module.packed_device_step([step_entry(owner) for owner in chosen],
                                                                 cancelled=lambda: False, block=block)
                        results.append((outputs, list(stepped)))
                    results.append((list(fixture.ttnn.executed), list(block.fixture.retained.commits), block.rounds,
                                    block.phase, [len(fixture.ttnn.host_copies)]))
                    return results
            finally:
                fixture.doCleanups()

        with clean_environment():
            self.assertEqual(scenario(packed_verifier, serving_packed_step), scenario(parent_verifier, parent_step))

    def test_the_verifier_rounds_are_the_parents_call_for_call(self):
        parent = self.parent('packed_verifier.py')

        class Rounds(tpp.ProbeFixture):
            run_rounds = tpp.FlagOffTests.run_rounds

            def runTest(self):
                pass

        def recorded(module):
            fixture = Rounds()
            fixture.setUp()
            try:
                return fixture.run_rounds(module)
            finally:
                fixture.doCleanups()

        today, before = recorded(packed_verifier), recorded(parent)
        self.assertGreater(len(before['copies']), 400)
        self.assertEqual(today, before)


class RuntimeAdmissionTests(unittest.TestCase):
    """serving_runtime.padded_block_admission: the 64-row M3 block only, as QWEN_FAST_SINGLE_GATEUP."""

    M3 = {'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}

    def admission(self, environ, requests=4):
        import serving_runtime
        return serving_runtime.padded_block_admission(dict(scheduler_requests=requests), environ)

    def test_off_it_reads_nothing(self):
        import serving_runtime
        from unittest.mock import Mock

        policy = Mock(return_value=dict(scheduler_requests=4))
        for environ in ({}, {'QWEN_FAST_PADDED_BLOCK': '0'}, dict(self.M3, QWEN_FAST_PADDED_BLOCK_MIN_USERS='9')):
            self.assertIsNone(serving_runtime.padded_block_admission(policy, environ))
        policy.assert_not_called()

    def test_the_sixty_four_row_block_takes_the_minimum_asked_for(self):
        self.assertEqual(self.admission(dict(self.M3, QWEN_FAST_PADDED_BLOCK='1')), 2)
        self.assertEqual(self.admission(dict(self.M3, QWEN_FAST_PADDED_BLOCK='1', QWEN_FAST_PADDED_BLOCK_MIN_USERS='3')), 3)

    def test_every_other_shape_or_a_bad_value_is_refused(self):
        on = dict(self.M3, QWEN_FAST_PADDED_BLOCK='1')
        for environ, requests in ((on, 2), (on, 1), (dict(on, QWEN_FAST_FOUR_AS_TWO='1'), 4),
                                  ({'QWEN_FAST_PADDED_BLOCK': '1', 'QWEN_FAST_PACKED_STEP': '1'}, 4),
                                  ({'QWEN_FAST_PADDED_BLOCK': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}, 4)):
            with self.subTest(environ=environ, requests=requests), \
                    self.assertRaisesRegex(ValueError, 'QWEN_FAST_PADDED_BLOCK=1 is admitted only at the 64-row M3 block'):
                self.admission(environ, requests)
        for environ in (dict(self.M3, QWEN_FAST_PADDED_BLOCK='true'),
                        dict(on, QWEN_FAST_PADDED_BLOCK_MIN_USERS='x')):
            with self.subTest(environ=environ), self.assertRaises(ValueError):
                self.admission(environ)


class ArmTests(unittest.TestCase):
    """M3NATIVE_PADDED_BLOCK and _MIN_USERS cross as QWEN_FAST_PADDED_BLOCK(_MIN_USERS) right after the
    probe's line; the arm refuses what serving_runtime would refuse at attach, before the docker run."""

    LINES = ('${M3NATIVE_PADDED_BLOCK:+-e QWEN_FAST_PADDED_BLOCK=1}',
             '${M3NATIVE_PADDED_BLOCK_MIN_USERS:+-e QWEN_FAST_PADDED_BLOCK_MIN_USERS=$M3NATIVE_PADDED_BLOCK_MIN_USERS}')

    @staticmethod
    def text():
        from test_m3native_arm_env import arm_text
        return arm_text()

    def bash(self, script, **environ):
        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        try:
            return subprocess.run([bash, '-c', script], env=dict(PATH=os.environ.get('PATH', ''), **environ),
                                  capture_output=True, text=True, timeout=60)
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)

    def test_the_two_switches_cross_after_the_probe_before_the_entrypoint(self):
        text = self.text()
        lines = text.split(chr(10))
        start = next(number for number, line in enumerate(lines) if self.LINES[0] in line)
        self.assertEqual(lines[start - 1].strip(), '${M3NATIVE_PADDED_PROBE:+-e QWEN_FAST_PADDED_PROBE=1} ' + chr(92))
        for offset, expected in enumerate(self.LINES):
            with self.subTest(line=expected):
                self.assertEqual(text.count(expected), 1)
                self.assertEqual(lines[start + offset].strip(), expected + ' ' + chr(92), 'nothing else on the line')
                self.assertLess(text.index(expected), text.index('--entrypoint python3'))

    def test_unset_nothing_crosses(self):
        script = 'printf "%s|" ' + ' '.join(self.LINES) + chr(10)
        self.assertEqual(self.bash(script).stdout.strip('|'), '')
        self.assertEqual(self.bash(script, M3NATIVE_PADDED_BLOCK='1', M3NATIVE_PADDED_BLOCK_MIN_USERS='3').stdout,
                         '-e|QWEN_FAST_PADDED_BLOCK=1|-e|QWEN_FAST_PADDED_BLOCK_MIN_USERS=3|')

    def test_the_arm_refuses_what_the_attach_would(self):
        lines = self.text().split(chr(10))
        start = next(number for number, line in enumerate(lines) if line.startswith('# M3NATIVE_PADDED_BLOCK=1 (variable-user'))
        end = next(number for number in range(start, len(lines)) if lines[number] == 'fi')
        script = chr(10).join(['users="${M3NATIVE_USERS:-4}"'] + lines[start:end + 1] + ['echo passed'])
        cases = [(dict(), 0), (dict(M3NATIVE_PADDED_BLOCK='1'), 0),
                 (dict(M3NATIVE_PADDED_BLOCK='1', M3NATIVE_PADDED_BLOCK_MIN_USERS='3'), 0),
                 (dict(M3NATIVE_PADDED_BLOCK='yes'), 1),
                 (dict(M3NATIVE_PADDED_BLOCK='1', M3NATIVE_PADDED_BLOCK_MIN_USERS='1'), 1),
                 (dict(M3NATIVE_PADDED_BLOCK='1', M3NATIVE_PADDED_BLOCK_MIN_USERS='4'), 1),
                 (dict(M3NATIVE_PADDED_BLOCK_MIN_USERS='2'), 1),
                 (dict(M3NATIVE_PADDED_BLOCK='1', M3NATIVE_USERS='1'), 1),
                 (dict(M3NATIVE_PADDED_BLOCK='1', M3NATIVE_SEQUENTIAL_USERS='4'), 1)]
        for environ, code in cases:
            with self.subTest(environ=environ):
                result = self.bash(script, **environ)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual('passed' in result.stdout, code == 0)
        self.assertIn('min_users 3', self.bash(script, M3NATIVE_PADDED_BLOCK='1', M3NATIVE_PADDED_BLOCK_MIN_USERS='3').stdout)


class ShippingTests(unittest.TestCase):
    RUNTIME = ('packed_verifier.py', 'serving_packed_step.py', 'serving_runtime.py', 'padded_probe.py', 'gdn_user_batch.py')

    def test_every_module_m2_changes_is_in_both_image_copy_lists(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        docker, context = dockerfile_modules(dockerfile_text()), context_modules()
        for name in self.RUNTIME:
            with self.subTest(module=name):
                self.assertIn(name, docker)
                self.assertIn(name, context)

    def test_the_suite_runs_in_the_cpu_workflow(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertRegex(workflow, r'python -B -m unittest [^\n]*\btest_padded_block\b')

    def test_the_flag_names_are_read_where_the_arm_sends_them(self):
        for module in ('packed_verifier.py', 'serving_runtime.py', 'lever_n_m3native_gate.py'):
            with self.subTest(module=module):
                self.assertIn("'QWEN_FAST_PADDED_BLOCK'", (HERE / module).read_text(encoding='utf-8'))
        for module in ('packed_verifier.py', 'lever_n_m3native_gate.py'):
            self.assertIn("'QWEN_FAST_PADDED_BLOCK_MIN_USERS'", (HERE / module).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
