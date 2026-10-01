"""S2 W4: the step and the boundary cap (design s2-design.md section 4, W4; section 2.4).

A packed round on the extent block verifies every ticket at its own family E = (start // 256 + 1)
* 256; a ticket whose rows cross E has rows that read all of [0, E) and never their own key, so its
user commits at most accept_limit = min(16, E - start) rows. The SESSION cap is the control
(GreedySession.commit max_rows, passed by serving_packed_step.commit_entry through commit_limit);
the block's refusal in commit_user is the backstop, and this module shows it unreachable:
  - SessionCapTests: GreedySession.commit with max_rows - exactly the cap commits when every proposal
    is accepted, EOS and a mismatch inside the cap are unchanged, a bad cap is refused;
  - CommitEntryTests: commit_entry passes the block's limit (and the gate-only forced cap), and a
    block without accept_limit is called exactly as before, audit line included;
  - RealBlockStepTests: proposal_rows, ineligible and the padded skip line on the REAL extent block
    (test_packed_extent_block's fixture) - mixed families admitted, 127 refused and narrowed;
  - PropertyTests: for EVERY start 0..131312 and random predictions, what the session publishes is
    within accept_limit, checked by the block's own backstop; then whole rounds through
    run_verified_block on the real block at the residues that matter, and the mutation control
    (no session cap) that the backstop does fire.
"""

from pathlib import Path
import os
import random
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding/harness'))

from greedy_session import BlockTicket, GreedySession  # noqa: E402
from acceptance_report import PACKED_LINE  # noqa: E402
from extent_attention_replay import accept_limit, extent  # noqa: E402
from packed_verifier import PackedVerifierEngine  # noqa: E402
import serving_packed_step  # noqa: E402
from serving_packed_step import (AUDIT_CAP, AUDIT_LINE, BUDGET_CAP_MARKER, FORCE_CAP_MARKER, PackedStep, admitted,  # noqa: E402
                                 commit_entry, commit_limit, forced_cap, ineligible, packed_device_step,
                                 proposal_rows, run_verified_block)
import test_packed_extent_block as blocks  # noqa: E402
import test_serving_packed_step as steps  # noqa: E402

PROMPT = (0, 1, 2) * 12
C = blocks.C


def neural(request_id, history, count):
    """A feature drafter standing in for DFlash: `count` proposals continuing the history."""
    return tuple((history[-1] + 1 + index) % 100 for index in range(count))


def session(request_id='request', budget=4096, eos_ids=(), position=None):
    live = GreedySession(request_id, PROMPT, 0, vocab_size=100, max_new_tokens=budget, eos_ids=eos_ids,
                         neural={'dflash2': neural}, lookup_enabled=False)
    if position is not None:
        live.position = position
    return live


def draft(live, rows=16):
    return live.propose(live.request_id, max_rows=rows, selected='dflash2')


def extent_limit(position):
    """The extent block's accept_limit for a 16-row ticket (packed_verifier.accept_limit)."""
    return accept_limit(position, 16)


class CappedBlock(steps.FakeBlock):
    """The step's fake block with the extent block's two S2 methods."""

    extent = True

    def admits(self, position):
        return type(position) is int and 128 <= position and position + 16 <= C

    def accept_limit(self, position):
        return extent_limit(position)


def request_over(live, block, segment, carry=None):
    engine = steps.FakeEngine(live, carry)
    request = SimpleNamespace(session=live, engine=engine, runtime=steps.FakeRuntime(engine), collect_timings=False,
                              closed=False, busy=False, cancelled=False)
    if carry is None:
        block.bind(engine, segment)
    return request


def clean_environment(test, **values):
    stack = patch.dict(os.environ, values)
    stack.start()
    test.addCleanup(stack.stop)
    for name in ('QWEN_FAST_GATE_FORCE_CAP', 'QWEN_FAST_PACKED_AUDIT'):
        if name not in values:
            os.environ.pop(name, None)


class SessionCapTests(unittest.TestCase):
    def test_with_every_proposal_accepted_exactly_the_cap_commits(self):
        for cap in (1, 2, 6, 15, 16):
            live = session()
            ticket = draft(live)
            self.assertEqual(ticket.tokens, tuple(range(16)))
            predictions = (*ticket.tokens[1:], 55)
            published = []
            decision = live.commit('request', ticket, predictions, published.append, max_rows=cap)
            with self.subTest(cap=cap):
                self.assertEqual(published, [cap])
                self.assertEqual(decision.state_rows, cap)
                expected = ticket.tokens[1:cap + 1] if cap < 16 else (*ticket.tokens[1:], 55)
                self.assertEqual(decision.emitted, expected)
                self.assertEqual(decision.accepted, cap - 1)
                self.assertEqual(live.position, len(PROMPT) + cap)
                self.assertEqual((live.phase, live.pending), ('idle', None))
                # the capped rows count as drafted proposals, like rejected ones
                self.assertEqual(live.committed_block_proposals, 15)

    def test_eos_inside_the_cap_finishes_as_it_always_did(self):
        live = session(eos_ids=(3,))
        ticket = draft(live)
        decision = live.commit('request', ticket, (*ticket.tokens[1:], 55), lambda prefix: None, max_rows=6)
        self.assertEqual((decision.emitted, decision.state_rows, decision.finished), ((1, 2, 3), 3, True))
        self.assertTrue(live.finished)

    def test_a_mismatch_before_the_cap_is_the_uncapped_decision(self):
        for cap in (4, 6, 16, None):
            live = session()
            ticket = draft(live)
            predictions = (1, 2, 99, *ticket.tokens[4:], 55)
            published = []
            options = {} if cap is None else dict(max_rows=cap)
            decision = live.commit('request', ticket, predictions, published.append, **options)
            with self.subTest(cap=cap):
                self.assertEqual((decision.emitted, decision.state_rows, published), ((1, 2, 99), 3, [3]))

    def test_the_full_width_cap_is_the_uncapped_call(self):
        results = []
        for options in ({}, dict(max_rows=16)):
            live = session()
            ticket = draft(live)
            decision = live.commit('request', ticket, (*ticket.tokens[1:9], 42, *ticket.tokens[10:], 55),
                                   lambda prefix: None, **options)
            results.append((decision, live.position, tuple(live.emitted)))
        self.assertEqual(results[0], results[1])

    def test_a_bad_cap_is_refused_with_nothing_published_and_the_ticket_still_live(self):
        for cap in (0, 17, -1, '6', 6.0, True):
            live = session()
            ticket = draft(live)
            publish = Mock()
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, 'A commit cap of 1 to 16 rows is required'):
                live.commit('request', ticket, (*ticket.tokens[1:], 55), publish, max_rows=cap)
            publish.assert_not_called()
            self.assertEqual((live.phase, live.pending), ('pending', ticket))


class CommitEntryTests(unittest.TestCase):
    def setUp(self):
        clean_environment(self)

    def commit(self, block, position, predictions=None):
        live = session(position=position)
        request = request_over(live, block, 0)
        ticket = draft(live)
        block.phase, block.pending_segments = 'verified', {0}
        entry = dict(request_id='request', request=request, ticket=ticket)
        predictions = list((*ticket.tokens[1:], 55)) if predictions is None else predictions
        output = commit_entry(entry, block, 0, predictions, cancelled=lambda: False, metrics={}, verify_started=0.0,
                              verified=0.0)
        return output, [call for call in block.calls if call[0] == 'commit']

    def test_commit_entry_passes_the_blocks_accept_limit(self):
        for residue, limit in ((0, 16), (240, 16), (241, 15), (250, 6), (255, 1)):
            position = 20224 - 256 + residue
            output, commits = self.commit(CappedBlock(), position)
            with self.subTest(residue=residue):
                self.assertEqual(commits, [('commit', 0, limit)])
                self.assertEqual(len(output.token_ids), limit)
                self.assertEqual(output.position, position + limit)

    def test_the_gate_only_forced_cap_caps_every_block_lower_still(self):
        clean_environment(self, QWEN_FAST_GATE_FORCE_CAP='8')
        self.assertEqual(self.commit(CappedBlock(), 20224 - 256 + 10)[1], [('commit', 0, 8)])
        self.assertEqual(self.commit(CappedBlock(), 20224 - 256 + 250)[1], [('commit', 0, 6)])
        self.assertEqual(self.commit(steps.FakeBlock(), 5000)[1], [('commit', 0, 8)], 'a family block too')

    def test_a_block_without_accept_limit_is_called_exactly_as_before(self):
        """steps.FakeSession.commit takes no max_rows: a keyword would raise TypeError."""
        block, stepped = steps.FakeBlock(), []
        request = steps.FakeRequest('A', 4100, stepped)
        block.bind(request.engine, 0)
        request.propose(block.predictions_for(0), 15)
        block.phase, block.pending_segments = 'verified', {0}
        self.assertIsNone(commit_limit(block, request.session.pending))
        entry = dict(request_id='A', request=request, ticket=request.session.pending)
        output = commit_entry(entry, block, 0, block.predictions_for(0), cancelled=lambda: False, metrics={},
                              verify_started=0.0, verified=0.0)
        self.assertEqual(len(output.token_ids), 16)

    def test_the_audit_line_gains_cap_only_when_the_commit_had_one(self):
        clean_environment(self, QWEN_FAST_PACKED_AUDIT='1')
        logged = []
        with patch.object(serving_packed_step, 'audit_log', side_effect=lambda message, **values: logged.append(
                (message, values))):
            self.commit(steps.FakeBlock(), 5000)
            self.commit(CappedBlock(), 20224 - 256 + 250)
        (plain, plain_values), (capped, capped_values) = [item for item in logged if item[0].startswith('[PACKED] ')]
        self.assertEqual(plain, AUDIT_LINE)
        self.assertNotIn('cap', plain_values)
        self.assertEqual(capped, AUDIT_LINE + AUDIT_CAP)
        self.assertEqual(capped_values['cap'], 6)
        text = capped.format(**capped_values)
        self.assertTrue(text.endswith(' cap=6'))
        self.assertEqual(PACKED_LINE.search(text).group(4), '6', "the gates' parser reads the capped line")
        self.assertEqual(plain.format(**plain_values), AUDIT_LINE.format(**plain_values))

    def test_commit_limit_is_the_blocks_limit_under_the_forced_cap(self):
        ticket = SimpleNamespace(position=20218, tokens=tuple(range(16)))
        self.assertIsNone(commit_limit(SimpleNamespace(), ticket))
        self.assertEqual(commit_limit(CappedBlock(), ticket), 6)
        self.assertEqual(commit_limit(SimpleNamespace(accept_limit=lambda position: None), ticket), None)
        with patch.dict(os.environ, {'QWEN_FAST_GATE_FORCE_CAP': '4'}):
            self.assertEqual(commit_limit(CappedBlock(), ticket), 4)
            self.assertEqual(commit_limit(SimpleNamespace(), ticket), 4)

    def test_the_forced_cap_is_read_strictly_and_refused_at_attach(self):
        self.assertIsNone(forced_cap({}))
        self.assertEqual([forced_cap({'QWEN_FAST_GATE_FORCE_CAP': value}) for value in ('1', '8', '16', '32')],
                         [1, 8, 16, 32])
        for value in ('0', '33', '08', '8.0', '', ' 8', 'eight', '-1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'QWEN_FAST_GATE_FORCE_CAP must be'):
                forced_cap({'QWEN_FAST_GATE_FORCE_CAP': value})
        with patch.dict(os.environ, {'QWEN_FAST_GATE_FORCE_CAP': 'x'}), self.assertRaises(ValueError):
            PackedStep(steps.FakeBlock())
        # The fake keeps audit_log's real signature (one message, keyword values): a looser fake let a
        # three-positional call through here, and the engine died at attach on the rig (v62).
        logged = []

        def audit_log(message, **values):
            logged.append(message.format(**values))
        with patch.dict(os.environ, {'QWEN_FAST_GATE_FORCE_CAP': '8'}), \
                patch.object(serving_packed_step, 'audit_log', side_effect=audit_log):
            PackedStep(steps.FakeBlock())
        self.assertEqual(logged, [FORCE_CAP_MARKER + '8 (gate only)'])
        logged.clear()
        with patch.object(serving_packed_step, 'audit_log', side_effect=audit_log):
            PackedStep(steps.FakeBlock())
        self.assertEqual(logged, [], 'unset, nothing is logged')

    def test_admitted_prefers_the_blocks_own_admits(self):
        block = steps.FakeBlock(users=4)
        self.assertTrue(admitted(block, 5), 'a block with neither: no check, as before')
        block.replay_capacity = 4352
        self.assertEqual([admitted(block, position) for position in (4095, 4096, 4336, 4337)], [False, True, True, False])
        capped = CappedBlock()
        capped.replay_capacity = 4352
        self.assertEqual([admitted(capped, position) for position in (127, 128, 20000, C - 16, C - 15)],
                         [False, True, True, True, False])


class RealBlockStepTests(blocks.ExtentFixture):
    """proposal_rows, ineligible and the padded skip line against the REAL extent block."""

    def requests(self, block, positions, widths=(1, 2, 4)):
        made, stepped = [], []
        for segment, position in enumerate(positions):
            if position is None:
                continue
            request = steps.FakeRequest('R%d' % segment, position, stepped,
                                        carry=self.pool.slots[segment].verifier.carry)
            request.engine.widths = widths
            request.engine.pages = torch.full((1, blocks.WIDTH), 7 + segment, dtype=torch.int32)
            made.append(request)
        return made, stepped

    def test_proposal_rows_admits_mixed_families_and_refuses_the_floor_and_the_ceiling(self):
        block = self.build()
        requests, stepped = self.requests(block, (1500, 20000, 60000, 131312))
        self.assertEqual(proposal_rows(block, requests), 16)
        for index, position in ((0, 127), (3, C - 15)):
            moved, stepped = self.requests(block, (1500, 20000, 60000, 131312))
            moved[index].session.position = position
            with self.subTest(position=position):
                self.assertIsNone(proposal_rows(block, moved))
        # budget: exactly 16 left still drafts the block; 15 does not (unchanged)
        requests[2].session.emitted = [1] * 240
        self.assertEqual(proposal_rows(block, requests), 16)
        requests[2].session.emitted = [1] * 241
        self.assertIsNone(proposal_rows(block, requests))

    def test_a_ticket_outside_the_extent_path_is_ineligible_and_narrowed_for_the_sequential_step(self):
        block = self.build()
        requests, stepped = self.requests(block, (127, 20000, 60000, 131312))
        entries = []
        for request in requests:
            request.propose(list(range(1, 16)), 15)
            entries.append(steps.entry(request))
        self.assertEqual(ineligible(entries, block), 'request=R0 position=127 outside the extent path')
        entries[0]['ticket'] = requests[0].session.pending
        with patch.object(serving_packed_step, 'step_log'):
            outputs = packed_device_step(entries, cancelled=lambda: False, block=block)
        self.assertEqual([output.request_id for output in outputs], ['R0', 'R1', 'R2', 'R3'])
        self.assertEqual(block.rounds, 0, 'the block never verified the round')
        self.assertEqual([request.rows_stepped for request in requests], [[4]] * 4, 'each narrowed to its widest capture')
        # at the floor the round is the block's
        requests, stepped = self.requests(block, (128, 20000, 60000, 131312))
        entries = []
        for request in requests:
            request.propose(list(range(1, 16)), 15)
            entries.append(steps.entry(request))
        self.assertIsNone(ineligible(entries, block))

    def test_the_padded_skip_line_asks_the_blocks_admits(self):
        block = self.build(padded_min_users=2)
        requests, stepped = self.requests(block, (127, 20000, 60000, None))
        entries = []
        for request in requests:
            request.propose(list(range(1, 16)), 15)
            entries.append(steps.entry(request))
        lines = []
        with patch('verify_trace_t2.log_line', side_effect=lines.append), patch.object(serving_packed_step, 'step_log'):
            packed_device_step(entries, cancelled=lambda: False, block=block)
        skipped = [line for line in lines if line.startswith('[PINDIAG] packed padded skipped')]
        self.assertEqual(len(skipped), 1)
        self.assertIn('live=3 eligible=0 reason=request=R0_position=127_outside_the_extent_path', skipped[0])


class PropertyTests(blocks.ExtentFixture):
    def backstop(self, start):
        """The block's own backstop (PackedVerifierEngine.check_extent_cap) for a segment whose round
        started at `start`, on a stand-in holding only what it reads."""
        stub = SimpleNamespace(extent=True, rows_per_user=16, round_starts={0: start}, idle_segments=frozenset(),
                               extent_counts=dict(cap_refused=0), rounds=1)
        stub.accept_limit = lambda position: PackedVerifierEngine.accept_limit(stub, position)
        return lambda prefix: PackedVerifierEngine.check_extent_cap(stub, 0, prefix)

    def test_every_start_commits_within_its_accept_limit_so_the_backstop_is_unreachable(self):
        """For every start 0..131312 and random predictions: the session, capped by commit_limit exactly
        as commit_entry caps it, publishes a prefix the block's backstop admits."""
        rng = random.Random(20260927)
        block = CappedBlock()
        fired = 0
        for start in range(0, C - 16 + 1):
            if start % 512 == 0:
                # a fresh session every 512 starts keeps the drafter's lookup history short (its commit
                # grows with it); what is under test is commit's cap, not the history
                live = session(budget=10 ** 9)
            first = rng.randrange(100)
            tokens = tuple((first + index) % 100 for index in range(16))
            # half the tickets agree on all fifteen proposals (the case the cap bites), the rest anywhere
            agree = 15 if rng.random() < 0.5 else rng.randrange(16)
            predictions = [tokens[index + 1] if index < agree else (tokens[index + 1] + 1) % 100
                           for index in range(15)] + [(first + 37) % 100]
            live.epoch += 1
            ticket = BlockTicket('request', live.epoch, start, tokens, 'dflash2', 0)
            live.pending, live.phase, live.position = ticket, 'pending', start
            limit = commit_limit(block, ticket)
            decision = live.commit('request', ticket, predictions, self.backstop(start), max_rows=limit)
            if not (decision.state_rows <= limit == min(16, extent(start) - start)):
                self.fail('start %d: %d rows committed against a limit of %d' % (start, decision.state_rows, limit))
            fired += decision.state_rows == limit < 16
        self.assertGreater(fired, 1000, 'the cap was exercised at the family ends')

    ROUNDS = ((0, 1, 7, 127), (128, 239, 240, 241), (250, 255, 200, 16))
    FAMILIES = (20, 80, 234, 400)

    def run_round(self, block, residues):
        """One round through run_verified_block: four real sessions at their residues, every proposal the
        target agrees with, so each user commits exactly its limit."""
        requests, entries, ids = [], [], torch.zeros(64, dtype=torch.int32)
        for segment, (family, residue) in enumerate(zip(self.FAMILIES, residues)):
            live = session('R%d' % segment, position=256 * family + residue)
            request = request_over(live, None, segment, carry=self.pool.slots[segment].verifier.carry)
            request.engine.pages = torch.full((1, blocks.WIDTH), 7 + segment, dtype=torch.int32)
            ticket = draft(live)
            ids[16 * segment:16 * segment + 16] = torch.tensor((*ticket.tokens[1:], 55), dtype=torch.int32)
            requests.append(request)
            entries.append(dict(request_id='R%d' % segment, request=request, ticket=ticket))
        self.ids.value = ids
        outputs = run_verified_block(entries, cancelled=lambda: False, block=block)
        return requests, outputs

    def test_whole_rounds_on_the_real_block_commit_each_users_limit_and_never_the_backstop(self):
        block = self.build()
        for residues in self.ROUNDS:
            marked = len(block.fixture.retained.commits)
            requests, outputs = self.run_round(block, residues)
            limits = [min(16, 256 - residue) for residue in residues]
            with self.subTest(residues=residues):
                self.assertEqual(block.fixture.retained.commits[marked:], list(enumerate(limits)))
                self.assertEqual([len(output.token_ids) for output in outputs], limits)
                self.assertEqual(block.phase, 'idle')
        self.assertEqual(block.extent_counts['cap_refused'], 0)
        self.assertFalse(any(line.startswith('[PINDIAG] packed extent cap refused') for line in self.lines))
        self.assertEqual(block.extent_counts['cap_events'], 3, '241, 250 and 255')

    def test_without_the_session_cap_the_blocks_backstop_fires(self):
        """The mutation control: were commit_entry to drop the cap, a crossing ticket's user would publish
        rows past E and the block would refuse it - so the property above is what keeps it unreachable."""
        block = self.build()
        with patch.object(serving_packed_step, 'commit_limit', return_value=None), \
                self.assertRaisesRegex(ValueError, 'commits at most 6 rows; prefix 16 refused'):
            self.run_round(block, (250, 0, 0, 0))
        self.assertEqual(block.extent_counts['cap_refused'], 1)
        self.assertIn('[PINDIAG] packed extent cap refused round=1 segment=0 start=5370 prefix=16 limit=6', self.lines)
        self.assertEqual(block.phase, 'idle', 'fail_round released every segment at prefix 0')



class BudgetCapTests(blocks.ExtentFixture):
    """QWEN_FAST_BUDGET_CAP on the real extent block: a user near the end of its budget stays in the
    packed round and commits at most its remaining tokens, composed by min with the block's accept_limit;
    the others commit whole."""

    FAMILIES = PropertyTests.FAMILIES

    def setUp(self):
        super().setUp()
        self.env(QWEN_FAST_BUDGET_CAP='1')
        for name in ('QWEN_FAST_GATE_FORCE_CAP', 'QWEN_FAST_PACKED_AUDIT'):
            os.environ.pop(name, None)

    def round(self, block, residues, remaining, bridges=None):
        """One round through packed_device_step: user u has remaining[u] tokens left (its session's own
        budget; `bridges` may give vLLM's own), every proposal agreed, no sequential step."""
        requests, entries, ids = [], [], torch.zeros(64, dtype=torch.int32)
        for segment, (family, residue, left) in enumerate(zip(self.FAMILIES, residues, remaining)):
            live = session('R%d' % segment, budget=left + 1, position=256 * family + residue)
            request = request_over(live, None, segment, carry=self.pool.slots[segment].verifier.carry)
            request.engine.pages = torch.full((1, blocks.WIDTH), 7 + segment, dtype=torch.int32)
            ticket = live.propose(live.request_id, max_rows=16, selected='dflash2', full_width=True)
            ids[16 * segment:16 * segment + 16] = torch.tensor((*ticket.tokens[1:], 55), dtype=torch.int32)
            requests.append(request)
            entry = dict(request_id='R%d' % segment, request=request, ticket=ticket)
            if bridges is not None and bridges[segment] is not None:
                entry['bridge'] = bridges[segment]
            entries.append(entry)
        self.ids.value = ids
        with patch.object(serving_packed_step, 'sequential_packed_step',
                          side_effect=AssertionError('the round must stay packed')):
            outputs = packed_device_step(entries, cancelled=lambda: False, block=block)
        return requests, outputs

    def test_a_user_with_five_left_commits_five_in_a_packed_round_and_the_others_commit_whole(self):
        block = self.build()
        marked = len(block.fixture.retained.commits)
        requests, outputs = self.round(block, (0, 0, 0, 0), (5, 200, 200, 200))
        self.assertEqual(block.fixture.retained.commits[marked:], [(0, 5), (1, 16), (2, 16), (3, 16)])
        self.assertEqual([len(output.token_ids) for output in outputs], [5, 16, 16, 16])
        self.assertEqual([output.finished for output in outputs], [True, False, False, False])
        self.assertEqual(block.phase, 'idle')

    def test_the_budget_composes_with_the_blocks_accept_limit_by_min(self):
        block = self.build()
        for left, limit in ((4, 4), (6, 6), (9, 6), (200, 6)):
            marked = len(block.fixture.retained.commits)
            requests, outputs = self.round(block, (250, 0, 0, 0), (left, 200, 200, 200))
            with self.subTest(left=left):
                self.assertEqual(block.fixture.retained.commits[marked:][0], (0, limit))
                self.assertEqual(len(outputs[0].token_ids), limit)
        self.assertEqual(block.extent_counts['cap_refused'], 0)

    def test_the_emission_is_the_smaller_of_the_budget_the_family_end_and_the_block(self):
        block = self.build()
        for residue in (0, 100, 240, 241, 250, 255):
            for left in (1, 2, 5, 6, 15, 16, 17):
                requests, outputs = self.round(block, (residue, 0, 0, 0), (left, 200, 200, 200))
                with self.subTest(residue=residue, left=left):
                    self.assertEqual(len(outputs[0].token_ids), min(left, 256 - residue, 16))
        self.assertEqual(block.extent_counts['cap_refused'], 0)

    def test_vllms_smaller_budget_caps_at_vllms_and_one_line_is_logged_per_capped_commit(self):
        self.env(QWEN_FAST_PACKED_AUDIT='1')
        block = self.build()

        def bridge(max_tokens, produced):
            state = SimpleNamespace(sampling_params=SimpleNamespace(max_tokens=max_tokens),
                                    output_token_ids=[0] * produced)
            return SimpleNamespace(state=state)

        bridges = [bridge(10, 5), None, None, None]  # user 0: vLLM owes 5, its own session 9
        logged = []
        with patch.object(serving_packed_step, 'audit_log', side_effect=lambda message, **values: logged.append(
                message.format(**values))):
            requests, outputs = self.round(block, (0, 0, 0, 0), (9, 200, 200, 200), bridges=bridges)
        self.assertEqual([len(output.token_ids) for output in outputs], [5, 16, 16, 16])
        cap = [line for line in logged if line.startswith(BUDGET_CAP_MARKER)]
        self.assertEqual(len(cap), 1)
        self.assertRegex(cap[0], r'^\[PINDIAG\] packed budget cap request=R0 segment=0 position=\d+ remaining=5 limit=5$')
        packed = [line for line in logged if line.startswith('[PACKED] ')]
        self.assertEqual(len(packed), 4)
        self.assertTrue(packed[0].endswith(' cap=5'))
        self.assertTrue(all(line.endswith(' cap=16') for line in packed[1:]), 'the extent block already logs its own cap')

    def test_a_budget_under_one_raises_and_the_round_fails_closed(self):
        ticket = SimpleNamespace(position=5000, tokens=tuple(range(16)))
        for budget in (0, -1):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, 'budget of at least one token'):
                commit_limit(CappedBlock(), ticket, budget)
        block = self.build()
        state = SimpleNamespace(sampling_params=SimpleNamespace(max_tokens=5), output_token_ids=[0] * 5)
        bridges = [SimpleNamespace(state=state), None, None, None]
        with self.assertRaisesRegex(ValueError, 'budget of at least one token'):
            self.round(block, (0, 0, 0, 0), (9, 200, 200, 200), bridges=bridges)
        self.assertEqual(block.phase, 'idle', 'fail_round released every segment at prefix 0')

    def test_a_budget_that_does_not_bite_leaves_the_limit_as_it_was(self):
        ticket = SimpleNamespace(position=20218, tokens=tuple(range(16)))
        self.assertIsNone(commit_limit(steps.FakeBlock(), ticket, 16))
        self.assertIsNone(commit_limit(steps.FakeBlock(), ticket, 40))
        self.assertEqual(commit_limit(steps.FakeBlock(), ticket, 5), 5)
        self.assertEqual(commit_limit(CappedBlock(), ticket, 40), 6)
        self.assertEqual(commit_limit(CappedBlock(), ticket, 4), 4)
        self.assertEqual(commit_limit(CappedBlock(), ticket), 6)
        with patch.dict(os.environ, {'QWEN_FAST_GATE_FORCE_CAP': '3'}):
            self.assertEqual(commit_limit(CappedBlock(), ticket, 4), 3)
            self.assertEqual(commit_limit(CappedBlock(), ticket, 2), 2)

    def test_flag_off_the_step_passes_no_budget(self):
        os.environ.pop('QWEN_FAST_BUDGET_CAP')
        block = self.build()
        with patch.object(serving_packed_step, 'commit_limit', wraps=serving_packed_step.commit_limit) as limit:
            self.round(block, (0, 0, 0, 0), (200, 200, 200, 200))
        self.assertEqual([call.args[2] for call in limit.call_args_list], [None] * 4)

    def test_the_capped_emission_is_the_uncapped_prefix_for_every_accepted_and_remaining(self):
        rng = random.Random(20261001)
        block = CappedBlock()
        for _ in range(600):
            start = rng.randrange(128, C - 16)
            left = rng.randrange(1, 40)
            first = rng.randrange(100)
            tokens = tuple((first + index) % 100 for index in range(16))
            agree = rng.randrange(16)
            predictions = [tokens[index + 1] if index < agree else (tokens[index + 1] + 1) % 100
                           for index in range(15)] + [(first + 37) % 100]
            results = []
            for budget in (left + 1, 10 ** 6):
                live = session(budget=budget)
                live.epoch += 1
                ticket = BlockTicket('request', live.epoch, start, tokens, 'dflash2', 0)
                live.pending, live.phase, live.position = ticket, 'pending', start
                limit = commit_limit(block, ticket, left if budget == left + 1 else None)
                results.append(live.commit('request', ticket, predictions, lambda prefix: None, max_rows=limit))
            capped, uncapped = results
            self.assertLessEqual(len(capped.emitted), min(left, extent(start) - start))
            self.assertEqual(capped.emitted, uncapped.emitted[:len(capped.emitted)])


if __name__ == '__main__':
    unittest.main()
