"""QWEN_FAST_DRAFT_SINGLES_AUDIT: every batched draft (a packed pair, the four-user quad) against its members' own single-user
drafts, in the same round (draft_singles_audit.py, and its three hooks in the packed proposal coordinator).

  - the flag: unset or empty is off, 'all' or a positive N (the first N rounds that ran a batched group), anything else raises;
  - the readers: the single's parts (features, candidates, scores) are exactly what DFlashDevice.select_proposal hands the
    selector, checked against the served method itself at four cards; a batched trace's rows for a user are the rows the single
    reads;
  - one round: singles prepared with the round's seeds behind the batched pass, every raw output read once after the fence, the
    comparison after the selection - equal rounds log one line per group, and each kind of difference names its stage
    (raw values, indices and features per chip and user; the parts; the tokens); a single that cannot be prepared, a read that
    fails and a trace with nothing pending are the audit's verdict and never the round's; the singles' pending replays are
    always discarded;
  - the coordinator: with the flag off nothing is called; on, the audit starts after the batched traces are enqueued and before
    the fence, reads right after it and before the selection, is judged after it; the first N rounds only; the singles are kept
    (nothing is released) while the flag is on; a failing selection closes the audit.

    py -3.11 -B -m unittest test_draft_singles_audit      (from scripts/ci)
"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import draft_singles_audit as audit  # noqa: E402
import test_quad_draft as base  # noqa: E402
from test_quad_draft_tp4 import HostOps4, MESH4, four_chip_users, single_outputs  # noqa: E402
from tp_test_support import four_cards  # noqa: E402

FLAG = 'QWEN_FAST_DRAFT_SINGLES_AUDIT'


def clean_environment(**flags):
    environment = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environment.update(flags)
    return patch.dict(os.environ, environment, clear=True)


class FlagTests(unittest.TestCase):
    def test_unset_and_empty_are_off_all_and_a_count_are_on_anything_else_raises(self):
        self.assertIsNone(audit.audit_rounds({}))
        self.assertIsNone(audit.audit_rounds({FLAG: ''}))
        self.assertIsNone(audit.audit_rounds({FLAG: '0'}))
        self.assertFalse(audit.enabled({}))
        self.assertEqual(audit.audit_rounds({FLAG: 'all'}), 'all')
        self.assertEqual(audit.audit_rounds({FLAG: '3'}), 3)
        self.assertTrue(audit.enabled({FLAG: '1'}))
        for value in ('-1', '03', 'x', '1.5', ' 2', 'ALL'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.audit_rounds({FLAG: value})

    def test_the_first_n_batched_rounds_are_selected(self):
        self.assertEqual([audit.selected(n, {FLAG: '2'}) for n in (1, 2, 3)], [True, True, False])
        self.assertTrue(audit.selected(99, {FLAG: 'all'}))
        self.assertFalse(audit.selected(1, {}))


class ReaderTests(unittest.TestCase):
    def test_the_singles_parts_are_what_the_served_select_proposal_hands_the_selector(self):
        import dflash_device

        generator = torch.Generator().manual_seed(31)
        with four_cards():
            singles, _ = four_chip_users(generator)
            outputs = single_outputs(singles[2])
            operations = HostOps4()
            seen = {}

            def selector(hidden, candidates, unary, predecessors, successors, anchors):
                seen.update(hidden=hidden, candidates=candidates, unary=unary, anchors=anchors)
                return torch.zeros(1, 15, dtype=torch.int64), None

            fake = SimpleNamespace(operations=operations, block_rows=16, max_drafts=15, predecessors='p', successors='s',
                                   proposal_calls=0, position=0, history_rows=2048)
            with patch('dflash_device.select_active_candidates', side_effect=selector):
                dflash_device.DFlashDevice.select_proposal(fake, outputs, 77, 15)
            mine = audit.read_parts(audit.user_rows(audit.raw_outputs(operations, outputs), 0))
        for key, name in (('hidden', 'hidden'), ('candidates', 'candidates'), ('unary', 'unary')):
            with self.subTest(part=key):
                self.assertTrue(audit.same_bits(mine[key], seen[name]))
        self.assertEqual(tuple(mine['hidden'].shape), (1, 15, 256))
        self.assertEqual(tuple(mine['candidates'].shape), (1, 15, 16))

    def test_a_batched_traces_rows_for_a_user_are_the_rows_its_single_reads(self):
        generator = torch.Generator().manual_seed(32)
        with four_cards():
            singles, quad = four_chip_users(generator)
            operations = HostOps4()
            batched = audit.raw_outputs(operations, quad)
            for user, single in enumerate(singles):
                mine = audit.user_rows(batched, 16 * user)
                theirs = audit.user_rows(audit.raw_outputs(operations, single_outputs(single)), 0)
                self.assertEqual(len(mine['chunks']), 2)
                for left, right in zip(mine['chunks'], theirs['chunks']):
                    for key in ('values', 'indices'):
                        for a, b in zip(left[key], right[key]):
                            self.assertTrue(audit.same_bits(a.contiguous(), b.contiguous()), (user, key))
                for a, b in zip(mine['projected'], theirs['projected']):
                    self.assertTrue(audit.same_bits(a.contiguous(), b.contiguous()), user)

    def test_replicated_features_that_differ_are_refused(self):
        generator = torch.Generator().manual_seed(33)
        with four_cards():
            singles, _ = four_chip_users(generator)
            raw = audit.user_rows(audit.raw_outputs(HostOps4(), single_outputs(singles[0])), 0)
            raw['projected'][2][3, 3] += 1
            with self.assertRaises(AssertionError):
                audit.read_parts(raw)


# ---------------------------------------------------------------------------------------------
# One audited round on fakes.
# ---------------------------------------------------------------------------------------------

class Round:
    """Four users' singles and their batched traces (a quad, or two pairs) on fakes, built from the same rows."""

    def __init__(self, kind='quad', seed=41):
        generator = torch.Generator().manual_seed(seed)
        self.singles, self.quad_outputs = four_chip_users(generator)
        self.operations = HostOps4()
        self.seeds = (100, 101, 102, 103)
        self.expected = [tuple(range(user, user + 15)) for user in range(4)]
        self.captures, self.devices, self.by_slot = [], [], {}
        for user in range(4):
            capture = SimpleNamespace(_pending=None, discard_pending=Mock())
            capture.prepare_device = Mock(side_effect=lambda seed, capture=capture, user=user: self.prepare(capture, user, seed))
            device = SimpleNamespace(operations=self.operations, proposal_capture=capture, predecessors='p', successors='s',
                                     select_proposal=Mock(side_effect=lambda outputs, seed, count, user=user: self.expected[user]))
            self.captures.append(capture)
            self.devices.append(device)
            self.by_slot[user] = dict(device=device, seed=self.seeds[user])
        self.kind = kind
        if kind == 'quad':
            bucket = SimpleNamespace(outputs=self.quad_outputs, tokens=tuple(self.expected))
            trace = SimpleNamespace(devices=list(self.devices), _pending=(self.seeds, bucket, []))
            self.batched = [([0, 1, 2, 3], trace)]
            self.buckets = [bucket]
        else:
            self.batched, self.buckets = [], []
            for index in range(2):
                users = (2 * index, 2 * index + 1)
                chunks = [dict(start=chunk['start'], stop=chunk['stop'],
                               values=SimpleNamespace(chips=[torch.cat([self.singles[user]['chunks'][number]['values'][chip]
                                                                        for user in users], dim=2) for chip in range(4)]),
                               indices=SimpleNamespace(chips=[torch.cat([self.singles[user]['chunks'][number]['indices'][chip]
                                                                         for user in users], dim=2) for chip in range(4)]))
                          for number, chunk in enumerate(self.singles[users[0]]['chunks'])]
                projected = torch.cat([self.singles[user]['projected'][:, :, :16] for user in users], dim=2)
                bucket = SimpleNamespace(outputs=SimpleNamespace(chunks=chunks, projected=SimpleNamespace(
                    chips=[projected.clone() for _ in range(4)])), tokens=tuple(self.expected[user] for user in users))
                trace = SimpleNamespace(device_a=self.devices[users[0]], device_b=self.devices[users[1]],
                                        _pending=(self.seeds[users[0]], self.seeds[users[1]], bucket, []))
                self.batched.append((list(users), trace))
                self.buckets.append(bucket)

    def prepare(self, capture, user, seed):
        capture._pending = (seed, SimpleNamespace(outputs=single_outputs(self.singles[user])), [])
        return True

    def run(self, **options):
        lines, loguru = base.logged()
        with loguru:
            state = audit.start(self.batched, self.by_slot, 7, lambda device: None)
            audit.read(state)
            results = audit.finish(state)
        return state, results, lines


class RoundTests(unittest.TestCase):
    def setUp(self):
        cards = four_cards()
        cards.__enter__()
        self.addCleanup(cards.__exit__, None, None, None)

    def test_an_equal_quad_round_logs_one_line_and_compares_every_raw_read_part_and_token(self):
        fixture = Round('quad')
        state, results, lines = fixture.run()
        # per user: 2 chunks x 2 keys x 4 chips + 4 feature reads, then 3 parts and the tokens
        self.assertEqual(results, [([0, 1, 2, 3], True, 'all', 4 * (16 + 4) + 4 * 4)])
        self.assertEqual(lines, ['[DRAFT-SINGLES-AUDIT] round=7 group=[0, 1, 2, 3] equal=1 stage=all checks=96'])
        for capture, seed in zip(fixture.captures, fixture.seeds):
            capture.prepare_device.assert_called_once_with(seed)
            capture.discard_pending.assert_called_once_with()
        self.assertEqual(state.singles, [], 'every single is released from the audit')

    def test_an_equal_round_of_two_pairs_logs_one_line_per_pair(self):
        fixture = Round('pairs')
        state, results, lines = fixture.run()
        self.assertEqual([(labels, equal, stage) for labels, equal, stage, _ in results],
                         [([0, 1], True, 'all'), ([2, 3], True, 'all')])
        self.assertEqual(lines, ['[DRAFT-SINGLES-AUDIT] round=7 group=[0, 1] equal=1 stage=all checks=48',
                                 '[DRAFT-SINGLES-AUDIT] round=7 group=[2, 3] equal=1 stage=all checks=48'])

    def mutate_and_run(self, kind, mutate):
        fixture = Round(kind)
        mutate(fixture)
        return fixture.run()

    def test_each_kind_of_difference_names_its_stage_and_user(self):
        cases = (
            ('quad', 'raw:values:chunk1:chip2:u2',
             lambda f: f.quad_outputs.chunks[1]['values'].chips[2].__setitem__((0, 0, 35, 4), 99.0)),
            ('quad', 'raw:indices:chunk0:chip3:u1',
             lambda f: f.quad_outputs.chunks[0]['indices'].chips[3].__setitem__((0, 0, 20, 0), 12345)),
            ('quad', 'raw:projected:chip0:u3',
             lambda f: [chip.__setitem__((0, 0, 50, 7), 3.0) for chip in f.quad_outputs.projected.chips]),
            ('pairs', 'raw:values:chunk0:chip0:u3',
             lambda f: f.buckets[1].outputs.chunks[0]['values'].chips[0].__setitem__((0, 0, 20, 1), -5.0)),
            ('quad', 'tokens:u2', lambda f: setattr(f.buckets[0], 'tokens', (f.expected[0], f.expected[1], (9,) * 15,
                                                                            f.expected[3]))),
            ('pairs', 'tokens:u1', lambda f: setattr(f.buckets[0], 'tokens', (f.expected[0], (0,) * 15))),
        )
        for kind, stage, mutate in cases:
            with self.subTest(stage=stage):
                _, results, lines = self.mutate_and_run(kind, mutate)
                failing = [(labels, found) for labels, equal, found, _ in results if not equal]
                self.assertEqual([found for _, found in failing], [stage])
                self.assertEqual(len([line for line in lines if 'equal=0' in line]), 1)
                self.assertIn('stage=%s ' % stage, [line for line in lines if 'equal=0' in line][0] + ' ')

    def test_a_difference_the_raw_reads_do_not_show_is_named_by_the_parts(self):
        fixture = Round('quad')
        real = audit._batched_parts

        def altered(group):
            parts = real(group)
            parts[3] = dict(parts[3], candidates=parts[3]['candidates'] + 1)
            return parts

        with patch.object(audit, '_batched_parts', side_effect=altered):
            _, results, _ = fixture.run()
        self.assertEqual([(equal, stage) for _, equal, stage, _ in results], [(False, 'candidates:u3')])

    def test_the_first_difference_stops_the_group_and_the_other_group_is_still_judged(self):
        fixture = Round('pairs')
        fixture.buckets[0].outputs.chunks[1]['indices'].chips[2][0, 0, 3, 3] = 777
        _, results, _ = fixture.run()
        self.assertEqual([(labels, equal) for labels, equal, _, _ in results], [([0, 1], False), ([2, 3], True)])

    def test_a_single_that_cannot_be_prepared_is_the_audits_verdict_and_prepared_singles_are_discarded(self):
        fixture = Round('quad')
        fixture.captures[2].prepare_device = Mock(return_value=False)
        state, results, lines = fixture.run()
        self.assertEqual(lines, ['[DRAFT-SINGLES-AUDIT] round=7 group=[] equal=0 stage=singles-unavailable:ValueError '
                                 'checks=0'])
        for user in (0, 1):
            fixture.captures[user].discard_pending.assert_called_once_with()
        fixture.captures[3].prepare_device.assert_not_called()

    def test_a_read_that_fails_and_a_trace_with_nothing_pending_are_the_audits_verdict(self):
        fixture = Round('quad')
        fixture.batched[0][1]._pending = None
        _, results, lines = fixture.run()
        self.assertEqual([(equal, stage) for _, equal, stage, _ in results], [(False, 'read-error:ValueError')])
        for capture in fixture.captures:
            capture.discard_pending.assert_called_once_with()
        fixture = Round('quad')
        fixture.devices[1].select_proposal = Mock(side_effect=RuntimeError('read failed'))
        _, results, _ = fixture.run()
        self.assertEqual([(equal, stage) for _, equal, stage, _ in results], [(False, 'read-error:RuntimeError')])
        fixture = Round('quad')
        fixture.captures[1]._pending = None
        fixture.captures[1].prepare_device = Mock(return_value=True)
        _, results, _ = fixture.run()
        self.assertEqual([(equal, stage) for _, equal, stage, _ in results], [(False, 'read-error:ValueError')])

    def test_a_batched_round_that_selected_nothing_is_the_audits_verdict(self):
        fixture = Round('quad')
        fixture.buckets[0].tokens = None
        _, results, _ = fixture.run()
        self.assertEqual([(equal, stage) for _, equal, stage, _ in results], [(False, 'error:ValueError')])

    def test_the_view_a_batched_pair_installs_is_looked_through_to_the_single(self):
        fixture = Round('quad')
        original = fixture.devices[0].proposal_capture
        fixture.devices[0].proposal_capture = SimpleNamespace(_trace=object(), _which=0, _original=original)
        self.assertIs(audit.single_capture(fixture.devices[0]), original)
        fixture.devices[0].proposal_capture = SimpleNamespace(_trace=object(), _which=0, _original=None)
        self.assertIsNone(audit.single_capture(fixture.devices[0]))
        self.assertIs(audit.single_capture(fixture.devices[1]), fixture.captures[1])

    def test_ensure_single_runs_for_every_member_before_its_capture_is_prepared(self):
        fixture = Round('quad')
        order = []
        for user, capture in enumerate(fixture.captures):
            capture.prepare_device = Mock(side_effect=lambda seed, user=user, capture=capture: order.append(('prepare', user))
                                          or fixture.prepare(capture, user, seed))
        with base.logged()[1]:
            state = audit.start(fixture.batched, fixture.by_slot, 1, lambda device: order.append(
                ('ensure', fixture.devices.index(device))))
        self.assertEqual(order, [item for user in range(4) for item in (('ensure', user), ('prepare', user))])
        state.close()

    def test_the_members_of_a_trace_are_its_devices_in_row_order(self):
        fixture = Round('quad')
        devices, rows = audit.trace_members([0, 1, 2, 3], fixture.batched[0][1])
        self.assertEqual((devices, rows), (fixture.devices, [0, 16, 32, 48]))
        pairs = Round('pairs')
        devices, rows = audit.trace_members([2, 3], pairs.batched[1][1])
        self.assertEqual((devices, rows), (pairs.devices[2:], [0, 16]))
        with self.assertRaises(ValueError):
            audit.trace_members([0, 1, 2], fixture.batched[0][1])


# ---------------------------------------------------------------------------------------------
# The coordinator's hooks.
# ---------------------------------------------------------------------------------------------

LOG = base.LOG


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        flags = dict(QWEN_FAST_QUAD_DRAFT='1', QWEN_FAST_PACKED_PROPOSAL='1', QWEN_FAST_PAIR_ROW_EXACT='1',
                     QWEN_FAST_ROUND_B1='1', QWEN_FAST_FUSED_COMMIT_LIVE_BANKS='1', QWEN_FAST_PACKED_AUDIT='1')
        environment = clean_environment(**flags)
        environment.start()
        self.addCleanup(environment.stop)
        del LOG[:]
        base.FakeQuadTrace.instances, base.FakeQuadTrace.failures = [], []
        self.Pair = base.pair_trace_class()
        self.Pair.instances = []
        from test_dflash_packed_proposal_coordinator import FakeSingleUserCapture

        for target, value in (('quad_draft.PreparedQuadDFlashProposal', base.FakeQuadTrace),
                              ('dflash_proposal_trace.PreparedPackedDFlashProposal', self.Pair),
                              ('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture),
                              ('dflash_packed_proposal.select_packed_batched', Mock(side_effect=base.selected_tokens))):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: LOG.append(('sync',))))
        self.mesh = object()
        self.state = SimpleNamespace(close=Mock(side_effect=lambda: LOG.append(('close',))))
        for name, effect in (('start', lambda *args: LOG.append(('audit-start', len(args[0]))) or self.state),
                             ('read', lambda state: LOG.append(('audit-read',)) or state),
                             ('finish', lambda state: LOG.append(('audit-finish',)) or [])):
            patcher = patch.object(audit, name, side_effect=effect)
            patcher.start()
            self.addCleanup(patcher.stop)

    def coordinator(self):
        import dflash_packed_proposal_coordinator

        return dflash_packed_proposal_coordinator.PackedProposalCoordinator()

    def prepare(self, coordinator, bridges, **options):
        lines, loguru = base.logged()
        with loguru:
            prepared = coordinator.prepare(bridges, **options)
        return prepared, lines

    def test_off_nothing_is_called_and_the_singles_are_released_as_before(self):
        coordinator, bridges = self.coordinator(), base.quad_bridges(self.operations, self.mesh)
        self.prepare(coordinator, bridges)
        self.assertNotIn(('audit-start', 1), LOG)
        self.assertEqual([entry[0] for entry in LOG if entry[0].startswith('audit-')], [])
        self.assertTrue(all(bridge.request.runtime.drafter._packed_capture_released for bridge in bridges))

    def test_the_audit_starts_behind_the_batched_pass_before_the_fence_reads_after_it_and_is_judged_after_the_selection(self):
        os.environ[FLAG] = 'all'
        coordinator, bridges = self.coordinator(), base.quad_bridges(self.operations, self.mesh)
        self.prepare(coordinator, bridges)
        self.assertEqual(LOG, [('quad', (100, 101, 102, 103)), ('audit-start', 1), ('sync',), ('audit-read',),
                               ('collect', 'quad'), ('audit', 1), ('audit-finish',)])
        self.assertFalse(any(getattr(bridge.request.runtime.drafter, '_packed_capture_released', False) for bridge in bridges),
                         'the singles the audit replays are kept')
        del LOG[:]
        os.environ[FLAG] = 'all'
        self.Pair.instances = []
        coordinator, bridges = self.coordinator(), base.quad_bridges(self.operations, self.mesh)
        os.environ['QWEN_FAST_QUAD_DRAFT'] = '0'
        self.prepare(coordinator, bridges)
        self.assertEqual([entry for entry in LOG if entry[0] in ('audit-start', 'audit-read', 'audit-finish', 'sync')],
                         [('audit-start', 2), ('sync',), ('audit-read',), ('audit-finish',)], 'two pair groups, one audit')

    def test_only_the_first_n_batched_rounds_are_audited_and_rounds_without_a_group_are_not_counted(self):
        os.environ[FLAG] = '2'
        coordinator, bridges = self.coordinator(), base.quad_bridges(self.operations, self.mesh)
        self.prepare(coordinator, bridges[:1])
        self.assertEqual(coordinator.singles_audit_candidates, 0, 'one user: no batched group')
        for _ in range(3):
            self.prepare(coordinator, bridges)
        self.assertEqual(coordinator.singles_audit_candidates, 3)
        self.assertEqual([entry for entry in LOG if entry[0] == 'audit-start'], [('audit-start', 1)] * 2)

    def test_a_failing_selection_closes_the_audit_and_the_round_still_fails(self):
        os.environ[FLAG] = 'all'
        coordinator, bridges = self.coordinator(), base.quad_bridges(self.operations, self.mesh)
        with patch('dflash_packed_proposal_coordinator.select_round', side_effect=RuntimeError('select failed')), \
                self.assertRaises(RuntimeError):
            self.prepare(coordinator, bridges)
        self.assertIn(('close',), LOG)
        self.assertNotIn(('audit-finish',), LOG)

    def test_a_bad_flag_value_fails_the_round_and_a_round_without_batched_traces_reads_nothing(self):
        os.environ[FLAG] = 'yes'
        coordinator, bridges = self.coordinator(), base.quad_bridges(self.operations, self.mesh)
        with patch.object(audit, 'selected', wraps=lambda n: audit.audit_rounds() and True):
            with self.assertRaises(ValueError):
                self.prepare(coordinator, bridges)
        os.environ[FLAG] = 'all'
        del LOG[:]
        os.environ['QWEN_FAST_ROUND_B1'] = '0'
        coordinator = self.coordinator()
        self.prepare(coordinator, base.quad_bridges(self.operations, self.mesh))
        self.assertEqual([entry for entry in LOG if entry[0].startswith('audit-')], [], 'no batched selection, no audit')


if __name__ == '__main__':
    unittest.main()
