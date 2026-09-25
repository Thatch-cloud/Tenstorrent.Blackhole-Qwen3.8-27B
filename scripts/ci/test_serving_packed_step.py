"""The packed step: one verify pass for every user, each user's own commit in the
scheduler's order, and every round the block cannot serve handed to the sequential step whole.

Fakes in the shape of what the step drives: a block with the contract of
packed_verifier.PackedVerifierEngine (segments bound to engines by their carry, predictions
in entries order, commits one segment at a time), and requests whose session, engine and
runtime keep the state machines of greedy_session.GreedySession,
VerifierEngine.adopt_packed/publish and DFlashRequestRuntime.publish. Which commit the
block fences is the block's own rule (packed_verifier.commit_user fences the one that
empties its pending segments) and is asserted on the REAL block, through
test_packed_verifier's fixture, at the end."""

import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from serving_fast_request import CommittedOutput
import serving_packed_step
from serving_packed_step import PackedStep, audit_log, describe, packed_device_step, proposal_rows
from serving_sequential_step import describe as describe_sequential
from test_packed_verifier import PAGE_WIDTH, BlockFixture, FourUserFixture
import verifier_engine

ROWS = 16
TAPS = ('tap0', 'tap1', 'tap2', 'tap3', 'tap4')


class FakeBlock:
    """PackedVerifierEngine as the step sees it. Segments are bound to engines (by carry on
    the real block), verify returns predictions and segments in entries order, and every
    commit is recorded in the order it arrived."""

    def __init__(self, users=2, rows=ROWS):
        self.shape = SimpleNamespace(users=users, rows_per_user=rows, block_rows=users * rows)
        self.rows_per_user = rows
        self.bound = {}
        self.phase, self.pending_segments, self.rounds = 'idle', set(), 0
        self.calls = []

    def bind(self, engine, segment):
        self.bound[id(engine)] = segment

    def segment_of(self, engine):
        if id(engine) not in self.bound:
            raise ValueError('The request engine borrows no carry this block restores')
        return self.bound[id(engine)]

    def predictions_for(self, segment):
        """The target's ids for segment u's rows."""
        return [1000 + self.rows_per_user * segment + row for row in range(self.rows_per_user)]

    def verify(self, entries):
        if self.phase != 'idle':
            raise ValueError('An idle packed block is required')
        entries = list(entries)
        segments = tuple(self.segment_of(entry['request'].engine) for entry in entries)
        for entry in entries:
            entry['request'].session.check_ticket(entry['request_id'], entry['ticket'])
        self.calls.append(('verify', [entry['request_id'] for entry in entries]))
        self.phase, self.pending_segments, self.rounds = 'verified', set(segments), self.rounds + 1
        return [self.predictions_for(segment) for segment in segments], dict(segments=segments, users=self.shape.users)

    def check_segment(self, segment):
        if self.phase != 'verified' or segment not in self.pending_segments:
            raise ValueError('Segment %r has no verified, uncommitted block' % (segment,))

    def features(self, segment):
        self.check_segment(segment)
        return TAPS

    def commit_user(self, segment, prefix):
        self.check_segment(segment)
        self.calls.append(('commit', segment, prefix))
        self.pending_segments.discard(segment)
        if not self.pending_segments:
            self.phase = 'idle'

    def describe(self):
        return dict(name='packed-fake')


class FakeSession:
    """GreedySession's ticket, commit and abort, with greedy_verify.select_prefix's decision."""

    def __init__(self, request_id, position):
        self.request_id, self.position, self.seed = request_id, position, 5
        self.phase, self.pending, self.finished = 'idle', None, False
        self.emitted, self.aborted, self.epoch = [], 0, 0
        self.max_new_tokens = 256

    def propose(self, tokens):
        if self.phase != 'idle' or self.pending is not None:
            raise ValueError('An unfinished idle request is required')
        self.epoch += 1
        self.pending = SimpleNamespace(request_id=self.request_id, epoch=self.epoch, position=self.position,
                                       tokens=tuple(tokens))
        self.phase = 'pending'
        return self.pending

    def check_ticket(self, request_id, ticket):
        if request_id != self.request_id or self.phase != 'pending' or ticket is not self.pending:
            raise ValueError('The current live block ticket is required')

    def commit(self, request_id, ticket, predictions, publish):
        self.check_ticket(request_id, ticket)
        accepted = 0
        for proposed, predicted in zip(ticket.tokens[1:], predictions):
            if proposed != predicted:
                break
            accepted += 1
        decision = SimpleNamespace(emitted=(*ticket.tokens[1:accepted + 1], predictions[accepted]),
                                   accepted=accepted, state_rows=accepted + 1, finished=False)
        self.phase = 'committing'
        try:
            if publish(decision.state_rows) is not None:
                raise RuntimeError('Publication must return None')
        except BaseException:
            self.phase = 'failed'
            raise
        self.position += decision.state_rows
        self.emitted.extend(decision.emitted)
        self.pending, self.phase = None, 'idle'
        return decision

    def abort(self, request_id, ticket, restore):
        self.check_ticket(request_id, ticket)
        self.phase = 'committing'
        try:
            if restore(0) is not None:
                raise RuntimeError('Abort must return None')
        except BaseException:
            self.phase = 'failed'
            raise
        self.aborted += 1
        self.pending, self.phase = None, 'idle'

    def fail_verification(self, request_id, ticket):
        self.check_ticket(request_id, ticket)
        self.phase = 'failed'


class FakeEngine:
    """VerifierEngine.adopt_packed and the packed branch of its publish."""

    def __init__(self, session, carry=None, widths=(1, 2, 4, 8, 16)):
        self.session, self.position = session, session.position
        self.phase, self.pending, self.packed = 'idle', None, None
        self.adopted = []
        # the engine's own captures (VerifierEngine.widths / serves): the full T16 set, or
        # the sequential widths beside the four-user block
        self.widths = tuple(widths)
        if carry is not None:
            # borrowed from a pool slot, the way VerifierEngine.allocate_carry borrows it
            self.carry = [list(snapshot) for snapshot in carry]

    def serves(self, ticket):
        return len(ticket.tokens) in self.widths

    def adopt_packed(self, ticket, block, segment):
        self.session.check_ticket(self.session.request_id, ticket)
        if self.phase != 'idle' or self.pending is not None or ticket.position != self.position:
            raise ValueError('Only an idle engine at the ticket frontier can adopt a packed verification')
        if getattr(block, 'rows_per_user', None) != len(ticket.tokens):
            raise ValueError("A packed block segment holding exactly this ticket's rows is required")
        self.adopted.append((ticket, block, segment))
        self.packed, self.phase, self.pending = (block, segment), 'verified', ticket

    def verified_features_for_publication(self, ticket):
        if self.phase != 'verified' or self.pending is not ticket or self.session.phase != 'committing':
            raise ValueError('Feature publication requires the current committing verifier ticket')
        block, segment = self.packed
        return block.features(segment)

    def publish(self, prefix):
        ticket = self.pending
        if (self.phase != 'verified' or ticket is None or self.session.pending is not ticket
                or self.session.phase != 'committing' or not 0 <= prefix <= len(ticket.tokens)):
            raise ValueError('Publication requires the live verified ticket during its owner decision')
        self.phase = 'committing'
        try:
            block, segment = self.packed
            block.commit_user(segment, prefix)
            verifier_engine._resident = None
            self.position += prefix
            self.phase, self.pending, self.packed = 'idle', None, None
        except BaseException:
            self.phase = 'failed'
            raise


class FakeRuntime:
    """DFlashRequestRuntime.publish: the features first, from the block at this user's
    segment, then the target publication; a failed history publication fails the engine."""

    def __init__(self, engine):
        self.engine, self.features, self.published, self.fail = engine, None, [], None

    def publish(self, prefix):
        engine = self.engine
        ticket = engine.session.pending
        if engine.phase != 'verified' or engine.pending is not ticket or engine.session.phase != 'committing':
            raise ValueError('Feature publication requires the current verified target transaction')
        try:
            if prefix:
                self.features = engine.verified_features_for_publication(ticket)
            if self.fail is not None:
                raise self.fail
            engine.publish(prefix)
            self.published.append(prefix)
        except BaseException:
            engine.phase = 'failed'
            raise


class FakeRequest:
    def __init__(self, request_id, position, stepped, carry=None):
        self.session = FakeSession(request_id, position)
        self.engine = FakeEngine(self.session, carry)
        self.runtime = FakeRuntime(self.engine)
        self.closed = self.cancelled = self.busy = False
        self.collect_timings = False
        self.stepped = stepped

    def propose(self, predictions, accept, rows=ROWS):
        """A ticket whose first `accept` proposals the target will agree with."""
        return self.session.propose((self.session.seed, *predictions[:accept], *([0] * (rows - 1 - accept))))

    def step(self, request_id, *, cancelled):
        """What the sequential fallback drives: this request's own step."""
        flag = cancelled()
        self.stepped.append((request_id, flag))
        return CommittedOutput(request_id, () if flag else (7,), self.session.position, flag, flag)


def entry(request):
    return dict(request_id=request.session.request_id, request=request, ticket=request.session.pending)


def answers(*flags):
    """A cancellation callback answering these in turn, then the last one forever."""
    remaining = list(flags)

    def cancelled():
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    return cancelled


class ProposalRowsTests(unittest.TestCase):
    """The ticket width of the coming round, decided before drafting over every live request:
    the block's rows when the block will serve the round as one pass, else None."""

    def setUp(self):
        verifier_engine.note_prefill()
        self.block = FakeBlock(users=4)
        self.stepped = []

    def requests(self, names='ABCD', bind=True):
        made = []
        for segment, name in enumerate(names):
            request = FakeRequest(name, 4100 + segment * 50, self.stepped)
            if bind:
                self.block.bind(request.engine, segment)
            made.append(request)
        return made

    def test_the_blocks_rows_when_exactly_its_users_are_live_each_with_a_block_left_and_each_bound(self):
        requests = self.requests()
        self.assertEqual(proposal_rows(self.block, requests), 16)
        self.assertEqual(proposal_rows(self.block, list(reversed(requests))), 16, 'in any order')
        # exactly 16 tokens left still holds a block; 15 do not
        requests[2].session.emitted = [1] * 240
        self.assertEqual(proposal_rows(self.block, requests), 16)
        requests[2].session.emitted = [1] * 241
        self.assertIsNone(proposal_rows(self.block, requests))

    def test_survivors_finished_users_foreign_engines_and_a_frontier_outside_the_family_make_the_round_sequential(self):
        requests = self.requests()
        for live in (requests[:3], requests[:1], requests[1:]):
            with self.subTest(live=[request.session.request_id for request in live]):
                self.assertIsNone(proposal_rows(self.block, live))
        # a finished request is not live: three left of four is sequential, four of five packed
        requests[3].session.finished = True
        self.assertIsNone(proposal_rows(self.block, requests))
        fifth = FakeRequest('E', 4300, self.stepped)
        self.block.bind(fifth.engine, 3)
        self.assertEqual(proposal_rows(self.block, [*requests, fifth]), 16)
        requests[3].session.finished = False
        self.assertIsNone(proposal_rows(self.block, [*requests, fifth]), 'five live is no block')
        # an engine the block was not captured against
        foreign = self.requests(bind=False)
        self.assertIsNone(proposal_rows(self.block, foreign))
        # the block's native chunk family bounds every frontier: a block that would leave it
        self.block.replay_capacity = 4352
        requests = self.requests()
        self.assertEqual(proposal_rows(self.block, requests), 16)
        requests[0].session.position = 4340
        self.assertIsNone(proposal_rows(self.block, requests))
        requests[0].session.position = 4336
        self.assertEqual(proposal_rows(self.block, requests), 16)
        self.assertEqual(self.stepped, [])


class PackedStepTests(unittest.TestCase):
    def setUp(self):
        verifier_engine.note_prefill()
        self.block = FakeBlock()
        self.stepped = []

    def request(self, request_id, segment, position, accept, rows=ROWS, bind=True):
        request = FakeRequest(request_id, position, self.stepped)
        if bind:
            self.block.bind(request.engine, segment)
        request.propose(self.block.predictions_for(segment), accept, rows)
        return request

    def two(self):
        """A in pool slot 0 (segment 0), B in slot 1 (segment 1), presented B first
        (probe 35436807668 saw the scheduler present the pair as ['B', 'A'])."""
        first = self.request('A', 0, 100, accept=15)
        second = self.request('B', 1, 3000, accept=9)
        return [entry(second), entry(first)]

    def step(self, entries, cancelled=None):
        return packed_device_step(entries, cancelled=cancelled or (lambda: False), block=self.block)

    def test_one_verify_pass_serves_every_entry_and_each_commits_its_own_segment_in_order(self):
        entries = self.two()
        second, first = entries[0]['request'], entries[1]['request']
        verifier_engine._resident = object()
        outputs = self.step(entries)
        # one verify over both, then B's commit (segment 1) and A's (segment 0), in the
        # scheduler's order
        self.assertEqual(self.block.calls, [('verify', ['B', 'A']), ('commit', 1, 10), ('commit', 0, 16)])
        self.assertEqual(self.stepped, [], 'no request went through the sequential step')
        self.assertEqual([output.request_id for output in outputs], ['B', 'A'])
        self.assertEqual(outputs[0], CommittedOutput('B', tuple(self.block.predictions_for(1)[:10]), 3010, False))
        self.assertEqual(outputs[1], CommittedOutput('A', tuple(self.block.predictions_for(0)[:16]), 116, False))
        # adopt_packed took metrics['segments'][i]: B, entry 0, was segment 1 and A, entry 1, segment 0
        self.assertEqual(second.engine.adopted, [(entries[0]['ticket'], self.block, 1)])
        self.assertEqual(first.engine.adopted, [(entries[1]['ticket'], self.block, 0)])
        # each publication took its features from the block at its own segment
        self.assertEqual((second.runtime.features, second.runtime.published), (TAPS, [10]))
        self.assertEqual((first.runtime.features, first.runtime.published), (TAPS, [16]))
        for request in (first, second):
            self.assertEqual((request.session.phase, request.session.pending, request.engine.phase), ('idle', None, 'idle'))
            self.assertEqual(request.engine.position, request.session.position)
            self.assertFalse(request.busy or request.cancelled)
        self.assertEqual((self.block.phase, self.block.pending_segments), ('idle', set()))
        self.assertIsNone(verifier_engine._resident, 'slot 0 is nobody\'s after a packed round')
        # the next round, presented the other way: A's commit first
        first.propose(self.block.predictions_for(0), accept=2)
        second.propose(self.block.predictions_for(1), accept=15)
        outputs = self.step([entry(first), entry(second)])
        self.assertEqual(self.block.calls[3:], [('verify', ['A', 'B']), ('commit', 0, 3), ('commit', 1, 16)])
        self.assertEqual([(output.request_id, output.position) for output in outputs], [('A', 119), ('B', 3026)])
        self.assertEqual(self.block.rounds, 2)

    def test_the_memory_ledger_reads_p12_once_after_the_first_rounds_commits(self):
        # QWEN_FAST_MEMORY_LEDGER=1: P12 after the first packed round's verify AND every
        # commit have returned (outside any capture), and never again.
        import memory_ledger

        entries = self.two()
        ledger = memory_ledger.MemoryLedger(None, None)
        seen = []

        def phase(name, point=None, **walked):
            seen.append((name, point, list(self.block.calls), walked))

        with patch.object(memory_ledger, '_active', ledger), patch.object(ledger, 'phase', side_effect=phase):
            self.step(entries)
            first, second = entries[1]['request'], entries[0]['request']
            first.propose(self.block.predictions_for(0), accept=2)
            second.propose(self.block.predictions_for(1), accept=15)
            self.step([entry(first), entry(second)])
        self.assertEqual(len(seen), 1)
        name, point, calls, walked = seen[0]
        self.assertEqual((name, point), ('P12', 'first_packed_round'))
        self.assertEqual(calls, [('verify', ['B', 'A']), ('commit', 1, 10), ('commit', 0, 16)])
        self.assertIs(walked['packed_block'], self.block)
        self.assertEqual(len(walked['round_requests']), 2)

    def test_without_a_ledger_the_round_reads_nothing(self):
        import memory_ledger

        self.assertIsNone(memory_ledger.active())
        with patch.object(memory_ledger.MemoryLedger, 'phase', side_effect=AssertionError('no ledger')):
            self.step(self.two())

    def test_a_round_that_fails_its_verify_records_no_p12(self):
        import memory_ledger

        entries = self.two()
        self.block.verify = lambda entries: (_ for _ in ()).throw(RuntimeError('device failure'))
        ledger = memory_ledger.MemoryLedger(None, None)
        with patch.object(memory_ledger, '_active', ledger), patch.object(ledger, 'phase') as phase, \
                self.assertRaisesRegex(RuntimeError, 'device failure'):
            self.step(entries)
        phase.assert_not_called()
        self.assertFalse(ledger.first_round_recorded)

    def test_a_round_the_block_cannot_serve_goes_to_the_sequential_step_whole(self):
        cases = {}
        cases['a narrow ticket'] = [entry(self.request('B', 1, 3000, accept=7, rows=8)), entry(self.request('A', 0, 100, accept=15))]
        cases['one survivor'] = [entry(self.request('A', 0, 100, accept=15))]
        cases['more entries than the block serves'] = [entry(self.request(name, index, 100, accept=15))
                                                       for index, name in enumerate('ABC')]
        cases['an engine the block was not captured against'] = [entry(self.request('B', 1, 3000, accept=9)),
                                                                 entry(self.request('C', 0, 100, accept=15, bind=False))]
        for name, entries in cases.items():
            with self.subTest(name=name):
                self.stepped.clear()
                outputs = self.step(entries)
                self.assertEqual(self.stepped, [(item['request_id'], False) for item in entries])
                self.assertEqual([output.request_id for output in outputs], [item['request_id'] for item in entries])
                self.assertEqual(self.block.calls, [], 'the block was not touched')

    def test_a_round_drafted_for_the_block_that_the_block_cannot_serve_degrades_without_raising_past_the_step(self):
        """Beside the 64-row block the engines capture only (1, 2, 4): a 16-row ticket the
        block does not serve (its entries changed after drafting) has no capture anywhere and
        cannot be re-proposed. serving_worker_hook.discard_stale_ticket keeps this mix from
        ever forming in normal steady-state or transition operation; reached anyway (a
        scheduler race, not a hardware fault), the round degrades - failed exactly as a
        refused verify would fail it, naming the reason and the tickets, but returned rather
        than raised, so a race costs this round for this request, not the engine for every
        other live user."""
        request = FakeRequest('A', 100, self.stepped)
        request.engine.widths = (1, 2, 4)
        self.block.bind(request.engine, 0)
        request.propose(self.block.predictions_for(0), accept=15)
        with patch.dict(sys.modules, loguru=None), patch('sys.stdout', new_callable=io.StringIO) as output:
            outputs = self.step([entry(request)])
        self.assertEqual(outputs, [CommittedOutput('A', (), 100, True, True)])
        self.assertIn('cannot serve (entries=1 block_users=2) holds tickets no request '
                      'engine captured (request=A rows=16)', output.getvalue())
        self.assertEqual((self.stepped, self.block.calls), ([], []))
        self.assertEqual(request.session.phase, 'failed', 'the round is failed, as a refused verify would fail it')
        self.assertEqual(request.engine.adopted, [])
        # a narrow ticket the trimmed engine captured goes to the sequential step as before
        survivor = FakeRequest('B', 3000, self.stepped)
        survivor.engine.widths = (1, 2, 4)
        self.block.bind(survivor.engine, 1)
        survivor.propose(self.block.predictions_for(1), accept=3, rows=4)
        outputs = self.step([entry(survivor)])
        self.assertEqual((self.stepped, [output.request_id for output in outputs]), ([('B', False)], ['B']))
        # an engine without the serves contract (an untrimmed one) is never refused here
        plain = FakeRequest('C', 100, self.stepped)
        del plain.engine.widths
        plain.engine.serves = None
        self.block.bind(plain.engine, 0)
        plain.propose(self.block.predictions_for(0), accept=15)
        self.stepped.clear()
        self.step([entry(plain)])
        self.assertEqual(self.stepped, [('C', False)])

    def test_the_packed_step_object_binds_the_block_and_carries_the_proposal_policy(self):
        step = PackedStep(self.block)
        self.assertIs(step.block, self.block)
        entries = self.two()
        outputs = step(entries, cancelled=lambda: False)
        self.assertEqual([output.request_id for output in outputs], ['B', 'A'])
        self.assertEqual(len(self.block.calls), 3, 'one verify and two commits through the block')
        requests = [FakeRequest(name, 100, self.stepped) for name in 'AB']
        for segment, request in enumerate(requests):
            self.block.bind(request.engine, segment)
        self.assertEqual(step.proposal_rows(requests), 16)
        self.assertIsNone(step.proposal_rows(requests[:1]))
        with self.assertRaises(ValueError):
            PackedStep(None)

    def test_a_cancellation_before_the_verify_is_answered_by_each_requests_own_step(self):
        entries = self.two()
        outputs = self.step(entries, cancelled=lambda: True)
        self.assertEqual(self.stepped, [('B', True), ('A', True)])
        self.assertTrue(all(output.cancelled for output in outputs))
        self.assertEqual(self.block.calls, [])
        for item in entries:
            self.assertEqual(item['request'].engine.adopted, [])

    def test_a_cancellation_after_the_verify_aborts_every_user_through_the_block(self):
        entries = self.two()
        outputs = self.step(entries, cancelled=answers(False, True))
        # prefix 0 for both, in order
        self.assertEqual(self.block.calls, [('verify', ['B', 'A']), ('commit', 1, 0), ('commit', 0, 0)])
        self.assertEqual(outputs, [CommittedOutput('B', (), 3000, True, True), CommittedOutput('A', (), 100, True, True)])
        for item in entries:
            request = item['request']
            self.assertEqual((request.session.aborted, request.session.phase, request.session.pending), (1, 'idle', None))
            self.assertTrue(request.cancelled)
            self.assertFalse(request.busy)
            self.assertEqual(request.runtime.features, None, 'an abort publishes no features')
        self.assertEqual((self.block.phase, self.block.pending_segments), ('idle', set()))
        self.assertEqual(self.stepped, [])

    def test_a_cancellation_between_two_commits_aborts_only_the_users_still_pending(self):
        entries = self.two()
        outputs = self.step(entries, cancelled=answers(False, False, True))
        self.assertEqual(self.block.calls, [('verify', ['B', 'A']), ('commit', 1, 10), ('commit', 0, 0)])
        self.assertEqual(outputs, [CommittedOutput('B', tuple(self.block.predictions_for(1)[:10]), 3010, False),
                                   CommittedOutput('A', (), 100, True, True)])
        self.assertEqual([item['request'].cancelled for item in entries], [False, True])
        self.assertEqual(self.block.phase, 'idle')

    def test_a_failed_publication_fails_the_round_and_leaves_no_segment_undecided(self):
        # B, first in the scheduler's order, fails its history publication before the
        # block commits: B's segment is still pending and A was never adopted
        entries = self.two()
        second, first = entries[0]['request'], entries[1]['request']
        second.runtime.fail = RuntimeError('history publication failure')
        verifier_engine._resident = object()
        with self.assertRaisesRegex(RuntimeError, 'history publication failure'):
            self.step(entries)
        self.assertEqual((second.session.phase, second.engine.phase), ('failed', 'failed'))
        self.assertEqual((first.session.phase, first.engine.phase, first.engine.adopted), ('failed', 'idle', []))
        # both segments released at prefix 0, so the block can serve or close; no trace ran
        self.assertEqual(self.block.calls, [('verify', ['B', 'A']), ('commit', 0, 0), ('commit', 1, 0)])
        self.assertEqual((self.block.phase, self.block.pending_segments), ('idle', set()))
        self.assertFalse(first.busy or second.busy)
        self.assertIsNone(verifier_engine._resident)
        # A, second in order, fails after B committed: only A's segment is left to release
        self.block = FakeBlock()
        self.stepped = []
        entries = self.two()
        entries[1]['request'].runtime.fail = RuntimeError('history publication failure')
        with self.assertRaises(RuntimeError):
            self.step(entries)
        self.assertEqual(self.block.calls, [('verify', ['B', 'A']), ('commit', 1, 10), ('commit', 0, 0)])
        self.assertEqual(entries[0]['request'].session.position, 3010)
        self.assertEqual(self.block.phase, 'idle')

    def test_a_failure_in_the_verify_itself_touches_no_user(self):
        entries = self.two()

        def failing(entries):
            raise RuntimeError('device failure')

        self.block.verify = failing
        with self.assertRaisesRegex(RuntimeError, 'device failure'):
            self.step(entries)
        for item in entries:
            request = item['request']
            # the real block fails every session itself; here the tickets are still pending, so the step does
            self.assertEqual((request.session.phase, request.engine.adopted, request.busy), ('failed', [], False))
        self.assertEqual(self.block.calls, [])

    def test_refusals_come_before_any_device_work(self):
        cases = {}
        stranger = self.two()
        stranger[0]['ticket'] = SimpleNamespace(request_id='someone-else', position=3000, tokens=stranger[0]['ticket'].tokens)
        cases['a ticket for another request'] = stranger
        replaced = self.two()
        replaced[0]['ticket'] = SimpleNamespace(request_id='B', position=3000, tokens=replaced[0]['ticket'].tokens)
        cases['a ticket that is not the pending one'] = replaced
        for flag in ('busy', 'closed', 'cancelled'):
            entries = self.two()
            setattr(entries[1]['request'], flag, True)
            cases['a %s request' % flag] = entries
        for name, entries in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.step(entries)
        with self.assertRaises(ValueError):
            self.step([])
        with self.assertRaises(ValueError):
            packed_device_step(self.two(), cancelled=None, block=self.block)
        with self.assertRaises(ValueError):
            packed_device_step(self.two(), cancelled=lambda: False, block=None)
        self.assertEqual((self.block.calls, self.stepped), ([], []))

    def test_the_audit_line_names_the_segment_the_request_and_what_it_accepted(self):
        def lines(entries, enabled, cancelled=None):
            with patch.dict('os.environ', {'QWEN_FAST_PACKED_AUDIT': '1' if enabled else '0'}), \
                    patch('serving_packed_step.audit_log') as log:
                self.step(entries, cancelled=cancelled)
            # This test is about AUDIT_LINE (one '[PACKED] ' line per user) specifically;
            # the round's own '[PACKED-COMMIT-HOST]' line is covered separately, below.
            return [item.args[0].format(**item.kwargs) for item in log.call_args_list
                    if item.args[0].startswith('[PACKED] ')]

        self.assertEqual(lines(self.two(), False), [])
        self.block = FakeBlock()
        self.assertEqual(lines(self.two(), True), [
            '[PACKED] request=B segment=1 position=3000 prefix=10 emitted=10 '
            'predictions=[1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023]',
            '[PACKED] request=A segment=0 position=100 prefix=16 emitted=16 '
            'predictions=[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]'])
        # an aborted user logs prefix 0; ids are cut to 48 characters so a line stays short
        self.block = FakeBlock()
        long_id = 'B' * 100
        entries = [entry(self.request(long_id, 1, 32768, accept=9)), entry(self.request('A', 0, 100, accept=15))]
        seen = lines(entries, True, cancelled=answers(False, True))
        self.assertEqual(seen[0], '[PACKED] request=%s segment=1 position=32768 prefix=0 emitted=0 '
                                  'predictions=[1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023]' % ('B' * 48))
        self.assertTrue(all(len(line) < 200 for line in seen), max(map(len, seen)))

    def test_audit_log_prints_when_loguru_is_absent(self):
        with patch.dict(sys.modules, loguru=None), patch('sys.stdout', new_callable=io.StringIO) as output:
            audit_log('[PACKED] request={request} segment={segment}', request='A', segment=0)
        self.assertEqual(output.getvalue().strip(), '[PACKED] request=A segment=0')

    def test_the_commit_host_line_reports_one_line_per_round_not_per_user(self):
        """commit_entry's own adopt/session/other split, plus the block's per-segment
        commit_block_ms (packed_verifier.PackedVerifierEngine), as one
        '[PACKED-COMMIT-HOST]' line covering the whole round - unlike AUDIT_LINE's
        '[PACKED]', which is one line per user."""
        def lines(entries, enabled):
            with patch.dict('os.environ', {'QWEN_FAST_PACKED_AUDIT': '1' if enabled else '0'}), \
                    patch('serving_packed_step.audit_log') as log:
                self.step(entries)
            return [item.args[0].format(**item.kwargs) for item in log.call_args_list]

        self.assertEqual(lines(self.two(), False), [], 'silent without the flag, like every other audit line')
        self.block = FakeBlock()
        seen = lines(self.two(), True)
        host_lines = [line for line in seen if line.startswith('[PACKED-COMMIT-HOST]')]
        publish_lines = [line for line in seen if line.startswith('[PACKED-PUBLISH]')]
        packed_lines = [line for line in seen if line.startswith('[PACKED] ')]
        self.assertEqual(len(packed_lines), 2, 'one [PACKED] line per user, as before')
        self.assertEqual(len(host_lines), 1, 'exactly one [PACKED-COMMIT-HOST] line for the whole round')
        self.assertEqual(len(publish_lines), 1, 'exactly one [PACKED-PUBLISH] line for the whole round')
        # FakeBlock carries no commit_block_ms (the real PackedVerifierEngine does): every
        # entry's block_ms falls back to 0.00, and the round is FakeBlock.rounds (1, after
        # one verify())
        self.assertRegex(host_lines[0],
            r'^\[PACKED-COMMIT-HOST\] round=1 adopt_ms=\[[\d.]+,[\d.]+\] block_ms=\[0\.00,0\.00\] '
            r'session_ms=\[[\d.]+,[\d.]+\] other_ms=\[[\d.]+,[\d.]+\]$')
        # FakeRuntime.publish() (test_serving_packed_step.py's own fake) never calls
        # publication_stage at all, so every stage falls back to 0.00 for both users -
        # the line still reports every PUBLISH_STAGES name, unconditionally.
        self.assertEqual(publish_lines[0], '[PACKED-PUBLISH] round=1 stages={features: [0.00,0.00], '
            'prepare_history: [0.00,0.00], publish_target: [0.00,0.00], commit_history: [0.00,0.00]}')
        # both round-level lines come after both users' own [PACKED] lines: the round's
        # last commit produces [PACKED-COMMIT-HOST] then [PACKED-PUBLISH], in that order.
        self.assertEqual(seen[-2:], [host_lines[0], publish_lines[0]])

    def test_the_commit_host_lines_block_ms_follows_each_entrys_own_segment(self):
        """Predictions, segments and now block_ms are all indexed by SEGMENT, never by an
        entry's position in the round (serving_packed_step's own long-standing rule,
        docs/packed-device-step-plan-2026-09-20.md; probe 35436807668)."""
        self.block = FakeBlock()
        self.block.commit_block_ms = [1.5, 2.5]  # segment 0 (A): 1.50 ms, segment 1 (B): 2.50 ms
        entries = self.two()  # presented as [B (segment 1), A (segment 0)]
        with patch.dict('os.environ', {'QWEN_FAST_PACKED_AUDIT': '1'}), patch('serving_packed_step.audit_log') as log:
            self.step(entries)
        (host_call,) = [item for item in log.call_args_list if item.args[0].startswith('[PACKED-COMMIT-HOST]')]
        self.assertEqual(host_call.kwargs['block'], '2.50,1.50', 'B (segment 1) first, then A (segment 0)')

    def test_timings_are_recorded_for_a_request_that_collects_them(self):
        entries = self.two()
        second = entries[0]['request']
        second.collect_timings, second.timings = True, []
        second.prepared_timing, second.last_commit_time = (1.0, 1.5), None
        self.step(entries)
        (timing,) = second.timings
        self.assertEqual((timing['position'], timing['rows'], timing['committed']), (3000, 16, 10))
        self.assertEqual((timing['verifier']['packed'], timing['verifier']['segment'], timing['verifier']['segments']),
                         (True, 1, (1, 0)))
        self.assertEqual(set(timing), {'position', 'rows', 'committed', 'draft_ms', 'verify_host_ms', 'commit_host_ms',
                                       'cycle_ms', 'outside_phases_ms', 'verifier'})
        self.assertAlmostEqual(timing['draft_ms'], 500.0)
        self.assertIsNone(second.prepared_timing)
        self.assertIsNotNone(second.last_commit_time)
        self.assertFalse(hasattr(entries[1]['request'], 'timings'), 'a request not collecting records nothing')

    def test_it_reports_one_weight_pass_for_every_user_and_what_it_falls_back_to(self):
        cost = describe()
        self.assertIs(cost['batched'], True)
        self.assertEqual(cost['weight_passes_per_round'], 'one for all users')
        self.assertEqual(cost['fallback'], describe_sequential())
        with patch.dict('os.environ', {'QWEN_FAST_PACKED_AUDIT': '0'}):
            self.assertIs(serving_packed_step.audit_enabled(), False)
        with patch.dict('os.environ', {'QWEN_FAST_PACKED_AUDIT': '1'}):
            self.assertIs(serving_packed_step.audit_enabled(), True)


class InstrumentedFakeRuntime:
    """Mirrors DFlashRequestRuntime.publish()'s actual stage structure - each of
    'features', 'prepare_history', 'publish_target', 'commit_history' wrapped in its
    own self.publication_stage(name, prefix) call, and a real self.drafter - unlike
    FakeRuntime above, which models neither. Used only by the tests below that check
    the audit instrumentation and the pipelined-publish shim actually reach something
    that calls both."""

    def __init__(self, engine, drafter):
        self.engine, self.drafter = engine, drafter
        self.published = []

    def publication_stage(self, name, prefix):
        from contextlib import nullcontext
        return nullcontext()

    def publish(self, prefix):
        engine = self.engine
        ticket = engine.session.pending
        if engine.phase != 'verified' or engine.pending is not ticket or engine.session.phase != 'committing':
            raise ValueError('Feature publication requires the current verified target transaction')
        publication = None
        try:
            if prefix:
                with self.publication_stage('features', prefix):
                    features = engine.verified_features_for_publication(ticket)
                with self.publication_stage('prepare_history', prefix):
                    publication = self.drafter.prepare_publication(features, prefix, position=engine.position)
            with self.publication_stage('publish_target', prefix):
                engine.publish(prefix)
            if publication is not None:
                with self.publication_stage('commit_history', prefix):
                    self.drafter.commit_publication(publication)
            self.published.append(prefix)
        except BaseException:
            engine.phase = 'failed'
            raise


class FakeDrafter:
    """dflash_device.DFlashDevice.prepare_publication/commit_publication as
    InstrumentedFakeRuntime's publish() calls them - records merge_release and
    fused_steady_state so the pipelined-publish/traced-publish wiring can be checked
    without any device machinery."""

    def __init__(self):
        self.prepare_calls = []
        self.commit_calls = []

    def prepare_publication(self, features, prefix, *, position, merge_release=False, fused_steady_state=False):
        self.prepare_calls.append((features, prefix, position, merge_release, fused_steady_state))
        return SimpleNamespace(status='prepared')

    def commit_publication(self, publication):
        self.commit_calls.append(publication)
        publication.status = 'committed'


class PublishInstrumentationTests(unittest.TestCase):
    """The [PACKED-PUBLISH] stage line (QWEN_FAST_PACKED_AUDIT) and the
    QWEN_FAST_PIPELINED_PUBLISH merge_release wiring, against InstrumentedFakeRuntime -
    FakeRuntime above models neither, so PackedStepTests' own tests cannot exercise
    either path (test_the_commit_host_line_reports_one_line_per_round_not_per_user
    above already covers the case where the runtime does not model publication_stage
    at all)."""

    def setUp(self):
        verifier_engine.note_prefill()
        self.block = FakeBlock()
        self.stepped = []

    def request(self, request_id, segment, position, accept, rows=ROWS):
        request = FakeRequest(request_id, position, self.stepped)
        drafter = FakeDrafter()
        request.runtime = InstrumentedFakeRuntime(request.engine, drafter)
        self.block.bind(request.engine, segment)
        request.propose(self.block.predictions_for(segment), accept, rows)
        return request

    def two(self):
        first = self.request('A', 0, 100, accept=15)
        second = self.request('B', 1, 3000, accept=9)
        return [entry(second), entry(first)]

    def step(self, entries, cancelled=None):
        return packed_device_step(entries, cancelled=cancelled or (lambda: False), block=self.block)

    def test_publish_line_is_silent_without_the_audit_flag(self):
        entries = self.two()
        with patch.dict('os.environ', {'QWEN_FAST_PACKED_AUDIT': '0'}), patch('serving_packed_step.audit_log') as log:
            self.step(entries)
        self.assertFalse(any(call.args[0].startswith('[PACKED-PUBLISH]') for call in log.call_args_list))

    def test_publish_line_reports_every_stage_for_every_user_in_entries_order(self):
        entries = self.two()  # presented [B (segment 1), A (segment 0)]
        with patch.dict('os.environ', {'QWEN_FAST_PACKED_AUDIT': '1'}), patch('serving_packed_step.audit_log') as log:
            self.step(entries)
        (publish_call,) = [item for item in log.call_args_list if item.args[0].startswith('[PACKED-PUBLISH]')]
        stages = publish_call.kwargs['stages']
        for name in ('features:', 'prepare_history:', 'publish_target:', 'commit_history:'):
            self.assertIn(name, stages)
        # both users accepted a nonzero prefix, so every stage ran for both - two
        # comma-separated ms values inside each stage's own brackets.
        import re
        for match in re.finditer(r': \[([^\]]*)\]', stages):
            self.assertEqual(len(match.group(1).split(',')), 2)

    def test_a_zero_prefix_entry_reports_zero_for_features_and_prepare_history(self):
        """publish() skips 'features'/'prepare_history' entirely when prefix is 0
        (dflash_request_runtime.py:78) - reached on cancellation, session.abort's own
        restore(0) call. The line still reports a slot for that user, as 0.00, never
        a gap."""
        entries = self.two()
        # False first (passes packed_device_step's own pre-verify check), True after
        # (both commits cancelled) - answers()'s own established pattern for this,
        # PackedStepTests.test_a_cancellation_after_the_verify_aborts_every_user_through_the_block.
        with patch.dict('os.environ', {'QWEN_FAST_PACKED_AUDIT': '1'}), patch('serving_packed_step.audit_log') as log:
            self.step(entries, cancelled=answers(False, True))
        (publish_call,) = [item for item in log.call_args_list if item.args[0].startswith('[PACKED-PUBLISH]')]
        self.assertIn('features: [0.00,0.00]', publish_call.kwargs['stages'])
        self.assertIn('prepare_history: [0.00,0.00]', publish_call.kwargs['stages'])

    def split_round(self, *, b1, audit):
        """One round whose drafters record what the M0a sink holds while they publish."""
        import os

        from dflash_traced_publish import PUBLICATION_SPLITS, add_split

        entries = self.two()
        sinks = []
        for item in entries:
            drafter = item['request'].runtime.drafter
            publish = drafter.prepare_publication

            def recording(features, prefix, *, position, publish=publish, **options):
                sink = PUBLICATION_SPLITS.get()
                sinks.append(sink)
                if sink is not None:
                    add_split(sink, 'proj', 0.001 * len(sinks))
                    add_split(sink, 'kv_exec', 0.0005)
                return publish(features, prefix, position=position, **options)
            drafter.prepare_publication = recording
        environ = {'QWEN_FAST_PACKED_AUDIT': '1' if audit else '0'}
        if b1:
            environ['QWEN_FAST_ROUND_B1'] = '1'
        with patch.dict('os.environ', environ), patch('serving_packed_step.audit_log') as log:
            if not b1:
                os.environ.pop('QWEN_FAST_ROUND_B1', None)
            self.step(entries)
        self.assertIsNone(PUBLICATION_SPLITS.get(), 'the sink never outlives one commit')
        lines = [item for item in log.call_args_list if item.args[0].startswith('[PACKED-PUBLISH-SPLIT]')]
        return sinks, lines

    def test_round_b1_splits_one_line_per_entry_from_each_drafter_sink(self):
        """QWEN_FAST_ROUND_B1 (M0a) with the audit: each user's publication sees its own sink,
        and the round logs one split line per entry in entries order, every split named."""
        from dflash_traced_publish import PUBLICATION_SPLIT_NAMES

        sinks, lines = self.split_round(b1=True, audit=True)
        self.assertEqual(len(sinks), 2)
        self.assertIsNot(sinks[0], sinks[1])
        self.assertEqual([line.kwargs['entry'] for line in lines], [0, 1])
        for number, line in enumerate(lines, start=1):
            splits = line.kwargs['splits']
            self.assertEqual([part.split('=')[0] for part in splits.split(' ')], list(PUBLICATION_SPLIT_NAMES))
            self.assertIn('proj=%.2f' % number, splits)
            self.assertIn('kv_exec=0.50', splits)
            self.assertIn('hist=0.00', splits)
            self.assertLess(len(line.args[0].format(**line.kwargs)), 180, 'inside the log capture budget')

    def test_round_b1_splits_need_both_the_flag_and_the_audit(self):
        for b1, audit in ((False, True), (True, False), (False, False)):
            with self.subTest(b1=b1, audit=audit):
                sinks, lines = self.split_round(b1=b1, audit=audit)
                self.assertEqual(sinks, [None, None])
                self.assertEqual(lines, [])

    def test_pipelined_and_traced_publish_off_by_default_leaves_both_options_false(self):
        import os

        entries = self.two()
        self.assertNotIn('QWEN_FAST_PIPELINED_PUBLISH', os.environ)
        self.assertNotIn('QWEN_FAST_TRACED_PUBLISH', os.environ)
        self.step(entries)
        for e in entries:
            drafter = e['request'].runtime.drafter
            self.assertTrue(drafter.prepare_calls)
            self.assertTrue(all(call[3] is False and call[4] is False for call in drafter.prepare_calls))
            self.assertNotIn('prepare_publication', drafter.__dict__, 'no installer ever ran')

    def test_pipelined_publish_flag_alone_installs_merge_release_only(self):
        entries = self.two()
        with patch.dict('os.environ', {'QWEN_FAST_PIPELINED_PUBLISH': '1'}):
            self.step(entries)
        for e in entries:
            drafter = e['request'].runtime.drafter
            self.assertTrue(drafter.prepare_calls)
            self.assertTrue(all(call[3] is True and call[4] is False for call in drafter.prepare_calls))
            self.assertNotIn('prepare_publication', drafter.__dict__, 'restored after the commit')

    def test_traced_publish_flag_alone_installs_fused_steady_state_only(self):
        entries = self.two()
        with patch.dict('os.environ', {'QWEN_FAST_TRACED_PUBLISH': '1'}):
            self.step(entries)
        for e in entries:
            drafter = e['request'].runtime.drafter
            self.assertTrue(drafter.prepare_calls)
            self.assertTrue(all(call[3] is False and call[4] is True for call in drafter.prepare_calls))
            self.assertNotIn('prepare_publication', drafter.__dict__, 'restored after the commit')

    def test_both_publish_flags_together_install_both_options_on_one_shim(self):
        """Neither flag's effect is silently dropped by the other overwriting
        drafter.prepare_publication after it - dflash_traced_publish.
        install_publish_options composes both into ONE installer, never two stacked
        ones."""
        entries = self.two()
        with patch.dict('os.environ', {'QWEN_FAST_PIPELINED_PUBLISH': '1', 'QWEN_FAST_TRACED_PUBLISH': '1'}):
            self.step(entries)
        for e in entries:
            drafter = e['request'].runtime.drafter
            self.assertTrue(drafter.prepare_calls)
            self.assertTrue(all(call[3] is True and call[4] is True for call in drafter.prepare_calls))

    def test_pipelined_publish_flag_is_inert_for_a_runtime_with_no_drafter(self):
        """FakeRuntime (PackedStepTests' own fixture) has no .drafter at all - the
        flag must not try to install anything on it. Two plain FakeRequests (not this
        class's own InstrumentedFakeRuntime-bearing request()), matching FakeBlock's
        default two users."""
        request_a = FakeRequest('A', 100, self.stepped)
        request_b = FakeRequest('B', 3000, self.stepped)
        self.block.bind(request_a.engine, 0)
        self.block.bind(request_b.engine, 1)
        request_a.propose(self.block.predictions_for(0), 15)
        request_b.propose(self.block.predictions_for(1), 9)
        with patch.dict('os.environ', {'QWEN_FAST_PIPELINED_PUBLISH': '1'}):
            self.step([entry(request_b), entry(request_a)])
        self.assertEqual(request_a.runtime.published, [16])
        self.assertEqual(request_b.runtime.published, [10])


class FourUserStepTests(unittest.TestCase):
    """The step over the M3 block: four entries in the scheduler's order, one verify, four
    commits; and rounds the four-user block cannot serve - one to three survivors after
    partners finished - handed to the sequential step whole."""

    def setUp(self):
        verifier_engine.note_prefill()
        self.block = FakeBlock(users=4)
        self.stepped = []

    def request(self, request_id, segment, position, accept, rows=ROWS):
        request = FakeRequest(request_id, position, self.stepped)
        self.block.bind(request.engine, segment)
        request.propose(self.block.predictions_for(segment), accept, rows)
        return request

    def four(self):
        """A..D in pool slots 0..3, presented C, A, D, B."""
        owners = [self.request('A', 0, 100, accept=15), self.request('B', 1, 3000, accept=9),
                  self.request('C', 2, 700, accept=0), self.request('D', 3, 5000, accept=12)]
        return [entry(owners[index]) for index in (2, 0, 3, 1)]

    def step(self, entries, cancelled=None):
        return packed_device_step(entries, cancelled=cancelled or (lambda: False), block=self.block)

    def test_one_verify_serves_four_entries_and_each_commits_its_own_segment_in_order(self):
        entries = self.four()
        outputs = self.step(entries)
        self.assertEqual(self.block.calls, [('verify', ['C', 'A', 'D', 'B']), ('commit', 2, 1), ('commit', 0, 16),
                                            ('commit', 3, 13), ('commit', 1, 10)])
        self.assertEqual(self.stepped, [])
        self.assertEqual([output.request_id for output in outputs], ['C', 'A', 'D', 'B'])
        self.assertEqual(outputs, [CommittedOutput('C', tuple(self.block.predictions_for(2)[:1]), 701, False),
                                   CommittedOutput('A', tuple(self.block.predictions_for(0)[:16]), 116, False),
                                   CommittedOutput('D', tuple(self.block.predictions_for(3)[:13]), 5013, False),
                                   CommittedOutput('B', tuple(self.block.predictions_for(1)[:10]), 3010, False)])
        for item, segment in zip(entries, (2, 0, 3, 1), strict=True):
            request = item['request']
            self.assertEqual(request.engine.adopted, [(item['ticket'], self.block, segment)])
            self.assertEqual((request.session.phase, request.engine.phase, request.busy), ('idle', 'idle', False))
            self.assertEqual(request.engine.position, request.session.position)
        # C accepted nothing: its publication still ran, at prefix 1 (the target's own token)
        self.assertEqual(entries[0]['request'].runtime.published, [1])
        self.assertEqual((self.block.phase, self.block.pending_segments, self.block.rounds), ('idle', set(), 1))
        self.assertIsNone(verifier_engine._resident)

    def test_survivors_of_a_four_user_block_take_the_sequential_step_whole(self):
        entries = self.four()
        # two of four finished: the two survivors, in the scheduler's order, one step each
        for survivors in (entries[:2], entries[1:], entries[:1], entries[:3]):
            self.stepped.clear()
            outputs = self.step(survivors)
            self.assertEqual(self.stepped, [(item['request_id'], False) for item in survivors])
            self.assertEqual([output.request_id for output in outputs], [item['request_id'] for item in survivors])
            self.assertEqual(self.block.calls, [], 'the four-user block was not touched')
        # five entries are no block either
        fifth = self.request('E', 0, 9000, accept=3)
        self.stepped.clear()
        self.step([*entries, entry(fifth)])
        self.assertEqual([item[0] for item in self.stepped], ['C', 'A', 'D', 'B', 'E'])
        self.assertEqual(self.block.calls, [])

    def test_a_mixed_round_from_the_three_to_four_live_transition_degrades_without_raising(self):
        """The exact shape of run 35535533720: at the 3->4-live transition, D keeps a stale
        narrower ticket - drafted while fewer than four were live - while A, B and C draft
        fresh at the block's width once proposal_rows sees all four bound and live. Beside
        the real 64-row block every engine is trimmed to its sequential captures (1, 2, 4)
        (packed_shapes.sequential_capture_rows), so D's own ticket is servable standalone but
        A, B and C's 16-row tickets - captured only on the block - are not.
        serving_worker_hook.discard_stale_ticket keeps this mix from ever forming in normal
        steady-state or transition operation; reached anyway, the round degrades every
        request rather than raising past the step and crashing the engine for every live
        user."""
        owners = [self.request('A', 0, 100, accept=15), self.request('B', 1, 3000, accept=9),
                  self.request('C', 2, 700, accept=0), self.request('D', 3, 5000, accept=12)]
        for owner in owners:
            owner.engine.widths = (1, 2, 4)
        # D's ticket is stale: drafted narrow, before it shared the others' width
        owners[3].session.pending, owners[3].session.phase = None, 'idle'
        owners[3].propose(self.block.predictions_for(3), accept=3, rows=4)
        entries = [entry(owners[index]) for index in (2, 0, 3, 1)]
        with patch.dict(sys.modules, loguru=None), patch('sys.stdout', new_callable=io.StringIO) as output:
            outputs = self.step(entries)
        self.assertEqual(outputs, [CommittedOutput('C', (), 700, True, True), CommittedOutput('A', (), 100, True, True),
                                   CommittedOutput('D', (), 5000, True, True), CommittedOutput('B', (), 3000, True, True)])
        self.assertIn('request=D rows=4 rows_per_user=16', output.getvalue())
        self.assertIn('request=C rows=16', output.getvalue())
        self.assertEqual(self.block.calls, [], 'the block was never touched')
        for owner in owners:
            self.assertEqual(owner.session.phase, 'failed')
            self.assertEqual(owner.engine.adopted, [])

    def test_discarding_the_stale_ticket_lets_the_transition_round_reach_the_block(self):
        """The other half of the previous test: `serving_worker_hook.discard_stale_ticket`
        applied to D's stale narrow ticket before the round, exactly as the worker hook
        applies it every tick, replaces it with a fresh one at the block's width - and the
        round it drafts into is uniform again and reaches block.verify. The transition
        makes progress within this one extra draft, not never."""
        from serving_worker_hook import discard_stale_ticket

        owners = [self.request('A', 0, 100, accept=15), self.request('B', 1, 3000, accept=9),
                  self.request('C', 2, 700, accept=0), self.request('D', 3, 5000, accept=12)]
        for owner in owners:
            owner.engine.widths = (1, 2, 4)
        owners[3].session.pending, owners[3].session.phase = None, 'idle'
        owners[3].propose(self.block.predictions_for(3), accept=3, rows=4)
        discard_stale_ticket(owners[3], 16)
        self.assertIsNone(owners[3].session.pending, 'a stale ticket mismatching the round is discarded')
        owners[3].propose(self.block.predictions_for(3), accept=12, rows=16)
        entries = [entry(owners[index]) for index in (2, 0, 3, 1)]
        outputs = self.step(entries)
        self.assertEqual(self.block.calls[0], ('verify', ['C', 'A', 'D', 'B']))
        self.assertEqual([output.request_id for output in outputs], ['C', 'A', 'D', 'B'])
        for owner in owners:
            self.assertEqual(owner.session.phase, 'idle')

    def test_survivors_holding_stale_block_width_tickets_after_a_partner_finishes_are_refused_not_served(self):
        """The exact shape of run 35564623068, the opposite direction from the previous two
        tests: entries drop BELOW the block's own user count when one of four finishes, but
        a survivor's PENDING ticket is still the block's own 16-row width - drafted while
        all four were live, never discarded because a round the policy answers None for
        (too few live requests for a full group) was not treated as one that could hold a
        stale ticket. Beside the real 64-row block every engine is trimmed to its
        sequential captures (1, 2, 4) (packed_shapes.sequential_capture_rows), so a 16-row
        ticket has no capture to fall back to: `ineligible` refuses the round on entry count
        alone and `unservable` finds every ticket unservable, so `packed_device_step` calls
        `refuse_round` - failing every survivor's session, not just the finished partner's.
        The next `serving_vllm_contract.prepared_ticket` call for any of them then raises,
        because their session.phase is 'failed', not 'pending' - the engine crash the bug
        report traced. This is the state `serving_worker_hook._drafts` must never produce."""
        owners = [self.request('A', 0, 100, accept=15), self.request('B', 1, 3000, accept=9),
                  self.request('C', 2, 700, accept=0), self.request('D', 3, 5000, accept=12)]
        for owner in owners:
            owner.engine.widths = (1, 2, 4)
        for live in (owners[:3], owners[:2], owners[:1]):
            with self.subTest(live=[owner.session.request_id for owner in live]):
                for owner in live:
                    # A stale ticket the coming round has no shared width for: minted
                    # fresh here at the block's width, standing in for one drafted
                    # before the partner finished and never discarded.
                    owner.session.pending, owner.session.phase = None, 'idle'
                    owner.propose(self.block.predictions_for(owners.index(owner)), accept=0, rows=16)
                entries = [entry(owner) for owner in live]
                with patch.dict(sys.modules, loguru=None), patch('sys.stdout', new_callable=io.StringIO) as captured:
                    outputs = self.step(entries)
                self.assertEqual(outputs, [CommittedOutput(owner.session.request_id, (), owner.session.position,
                                                            True, True) for owner in live])
                self.assertIn('block_users=4', captured.getvalue())
                self.assertEqual(self.block.calls, [], 'the block was never touched')
                for owner in live:
                    self.assertEqual(owner.session.phase, 'failed')
                    self.assertEqual(owner.engine.adopted, [])

    def test_survivors_of_a_trimmed_block_reach_the_sequential_step_once_discard_gives_them_native_tickets(self):
        """The fix for the previous test: `serving_worker_hook.discard_stale_ticket`, called
        for every live request whenever a packed policy is configured - even a round it
        answers None for - drops the survivor's stale block-width ticket, exactly as it
        drops a mismatched narrower one for the three-to-four transition above. Redrafted at
        each engine's own trimmed capture, `unservable` finds every ticket servable
        standalone and `packed_device_step` hands the round to `sequential_packed_step`
        instead of `refuse_round`: degrade, not crash - all the way down, three of four
        survivors, then two, then the last."""
        from serving_worker_hook import discard_stale_ticket

        owners = [self.request('A', 0, 100, accept=15), self.request('B', 1, 3000, accept=9),
                  self.request('C', 2, 700, accept=0), self.request('D', 3, 5000, accept=12)]
        for owner in owners:
            owner.engine.widths = (1, 2, 4)
        for live in (owners[:3], owners[:2], owners[:1]):
            with self.subTest(live=[owner.session.request_id for owner in live]):
                for owner in live:
                    # discard_stale_ticket(owner, None) is exactly what the widened
                    # _drafts guard now runs for every live request when proposal_rows
                    # answers None - regardless of what width was pending before.
                    discard_stale_ticket(owner, None)
                    self.assertIsNone(owner.session.pending)
                    owner.propose(self.block.predictions_for(owners.index(owner)), accept=1, rows=4)
                self.stepped.clear()
                entries = [entry(owner) for owner in live]
                outputs = self.step(entries)
                self.assertEqual(self.stepped, [(owner.session.request_id, False) for owner in live])
                self.assertEqual([output.request_id for output in outputs],
                                 [owner.session.request_id for owner in live])
                self.assertEqual(self.block.calls, [], 'the four-user block was not touched')
                for owner in live:
                    # sequential_packed_step never touches session state itself (that is
                    # FastRequest.step's own job, covered elsewhere) - what matters here
                    # is that refuse_round's fail_round never ran, so the session is not
                    # left 'failed' for the next _drafts call to trip over.
                    self.assertNotEqual(owner.session.phase, 'failed')

    def test_a_cancellation_between_commits_aborts_the_users_still_pending(self):
        entries = self.four()
        outputs = self.step(entries, cancelled=answers(False, False, False, True))
        self.assertEqual(self.block.calls, [('verify', ['C', 'A', 'D', 'B']), ('commit', 2, 1), ('commit', 0, 16),
                                            ('commit', 3, 0), ('commit', 1, 0)])
        self.assertEqual([(output.request_id, output.cancelled) for output in outputs],
                         [('C', False), ('A', False), ('D', True), ('B', True)])
        self.assertEqual((self.block.phase, self.block.pending_segments), ('idle', set()))


class TwoBlockProposalRowsTests(unittest.TestCase):
    """proposal_rows over SEVERAL blocks (QWEN_FAST_FOUR_AS_TWO's pair of 32-row blocks for
    four users): the shared width when every live entry's OWN matched block is itself fully
    live (with budget and family beside) - exactly as the single m3 block required all
    four, since two disjoint two-user blocks covering the same four users make "every
    entry's own block is complete" and "all four are live" the same condition. Else None:
    a live survivor of an incomplete block (its partner already finished) takes every live
    request back to drafting at its own captured width - including another block's still-
    intact pair - rather than a forced 16 the trimmed survivor's engine could never capture
    (packed_shapes.M3_SEQUENTIAL_CAPTURE_ROWS beside two 32-row blocks, serving_runtime.py).
    `packed_device_step`'s own per-block partition still degrades a MIXED round gracefully
    when one reaches it anyway (a stale-ticket race, TwoBlockStepTests below) - this is the
    policy that keeps such a round from forming in normal steady-state operation."""

    def setUp(self):
        verifier_engine.note_prefill()
        self.block_a = FakeBlock(users=2)
        self.block_b = FakeBlock(users=2)
        self.stepped = []

    def requests(self, block, names, position=4100):
        made = []
        for segment, name in enumerate(names):
            request = FakeRequest(name, position + segment * 50, self.stepped)
            block.bind(request.engine, segment)
            made.append(request)
        return made

    def test_the_shared_width_when_both_blocks_full_pairs_are_live(self):
        a, b = self.requests(self.block_a, 'AB')
        c, d = self.requests(self.block_b, 'CD')
        self.assertEqual(proposal_rows([self.block_a, self.block_b], [a, b, c, d]), 16)
        self.assertEqual(proposal_rows([self.block_a, self.block_b], [c, d, a, b]), 16, 'in any order')

    def test_a_finished_partner_leaves_its_block_incomplete_and_answers_none_for_the_whole_round(self):
        a, b = self.requests(self.block_a, 'AB')
        c, d = self.requests(self.block_b, 'CD')
        d.session.finished = True
        self.assertIsNone(proposal_rows([self.block_a, self.block_b], [a, b, c, d]),
                          "block A's own pair is intact, but block B is not, and two blocks over four users "
                          "means that is the same as not all four being live")
        # d simply absent from the round is the same answer
        self.assertIsNone(proposal_rows([self.block_a, self.block_b], [a, b, c]))

    def test_a_request_bound_to_no_configured_block_makes_the_whole_round_none(self):
        a, b = self.requests(self.block_a, 'AB')
        foreign = FakeRequest('X', 9000, self.stepped)
        self.assertIsNone(proposal_rows([self.block_a, self.block_b], [a, b, foreign]))

    def test_budget_and_family_are_checked_per_request_against_its_own_matched_block(self):
        a, b = self.requests(self.block_a, 'AB')
        c, d = self.requests(self.block_b, 'CD')
        c.session.emitted = [1] * 241
        self.assertIsNone(proposal_rows([self.block_a, self.block_b], [a, b, c, d]),
                          'fewer than sixteen tokens left for C')
        c.session.emitted = []
        self.block_b.replay_capacity = 4352
        d.session.position = 4340
        self.assertIsNone(proposal_rows([self.block_a, self.block_b], [a, b, c, d]),
                          "D's frontier leaves block B's native chunk family")
        d.session.position = 4336
        self.assertEqual(proposal_rows([self.block_a, self.block_b], [a, b, c, d]), 16)

    def test_the_bare_single_block_form_is_unaffected(self):
        a, b = self.requests(self.block_a, 'AB')
        self.assertEqual(proposal_rows(self.block_a, [a, b]), 16, 'a lone block still takes the bare, non-list form')
        self.assertIsNone(proposal_rows(self.block_a, [a]))


class TwoBlockStepTests(unittest.TestCase):
    """QWEN_FAST_FOUR_AS_TWO: two independent 32-row blocks instead of one 64-row block.
    Entries are partitioned by the block their engine is bound to; each block runs its own
    verify+commit round - exactly as a lone block's round always has - in the blocks'
    CONFIGURED order (block A's whole round, then block B's); a block whose pair is not
    fully live falls to the sequential step for its own live member(s) while another
    block, still complete, runs packed in the same round."""

    def setUp(self):
        verifier_engine.note_prefill()
        self.block_a = FakeBlock(users=2, rows=ROWS)
        self.block_b = FakeBlock(users=2, rows=ROWS)
        self.stepped = []

    def request(self, block, request_id, segment, position, accept, rows=ROWS, bind=True):
        request = FakeRequest(request_id, position, self.stepped)
        if bind:
            block.bind(request.engine, segment)
        request.propose(block.predictions_for(segment), accept, rows)
        return request

    def four(self):
        """A, B bound to block A (segments 0, 1); C, D to block B (segments 0, 1);
        presented interleaved - a scheduler need not group entries by block."""
        a = self.request(self.block_a, 'A', 0, 100, accept=15)
        b = self.request(self.block_a, 'B', 1, 3000, accept=9)
        c = self.request(self.block_b, 'C', 0, 700, accept=0)
        d = self.request(self.block_b, 'D', 1, 5000, accept=12)
        return [entry(c), entry(a), entry(d), entry(b)]

    def step(self, entries, cancelled=None):
        return PackedStep([self.block_a, self.block_b])(entries, cancelled=cancelled or (lambda: False))

    def test_partitions_entries_by_block_and_runs_each_blocks_round_in_configured_order(self):
        entries = self.four()
        outputs = self.step(entries)
        # block A's whole round (its own entries' relative order: A then B) before block B's
        self.assertEqual(self.block_a.calls, [('verify', ['A', 'B']), ('commit', 0, 16), ('commit', 1, 10)])
        self.assertEqual(self.block_b.calls, [('verify', ['C', 'D']), ('commit', 0, 1), ('commit', 1, 13)])
        self.assertEqual(self.stepped, [], 'no request went through the sequential step')
        # the scheduler's own order, not the blocks' processing order
        self.assertEqual([output.request_id for output in outputs], ['C', 'A', 'D', 'B'])
        self.assertEqual(outputs, [
            CommittedOutput('C', tuple(self.block_b.predictions_for(0)[:1]), 701, False),
            CommittedOutput('A', tuple(self.block_a.predictions_for(0)[:16]), 116, False),
            CommittedOutput('D', tuple(self.block_b.predictions_for(1)[:13]), 5013, False),
            CommittedOutput('B', tuple(self.block_a.predictions_for(1)[:10]), 3010, False)])
        for block in (self.block_a, self.block_b):
            self.assertEqual((block.phase, block.pending_segments, block.rounds), ('idle', set(), 1))
        self.assertIsNone(verifier_engine._resident)

    def test_a_block_whose_pair_is_not_fully_live_falls_to_sequential_while_the_other_stays_packed(self):
        a = self.request(self.block_a, 'A', 0, 100, accept=15)
        b = self.request(self.block_a, 'B', 1, 3000, accept=9)
        # D already finished and is not part of this round: block B's group is only C
        c = self.request(self.block_b, 'C', 0, 700, accept=3)
        outputs = self.step([entry(c), entry(a), entry(b)])
        self.assertEqual(self.block_a.calls, [('verify', ['A', 'B']), ('commit', 0, 16), ('commit', 1, 10)])
        self.assertEqual(self.block_b.calls, [], 'block B was never touched: only one of its two users is present')
        self.assertEqual(self.stepped, [('C', False)], "C's own step, since block B cannot serve it alone")
        self.assertEqual([output.request_id for output in outputs], ['C', 'A', 'B'])
        self.assertEqual(outputs[0], CommittedOutput('C', (7,), 700, False))
        self.assertEqual(outputs[1], CommittedOutput('A', tuple(self.block_a.predictions_for(0)[:16]), 116, False))
        self.assertEqual(outputs[2], CommittedOutput('B', tuple(self.block_a.predictions_for(1)[:10]), 3010, False))

    def test_both_blocks_incomplete_pairs_go_entirely_to_the_sequential_step(self):
        a = self.request(self.block_a, 'A', 0, 100, accept=15)
        c = self.request(self.block_b, 'C', 0, 700, accept=3)
        outputs = self.step([entry(c), entry(a)])
        self.assertEqual((self.block_a.calls, self.block_b.calls), ([], []))
        self.assertEqual(self.stepped, [('C', False), ('A', False)])
        self.assertEqual([output.request_id for output in outputs], ['C', 'A'])

    def test_an_entry_bound_to_neither_configured_block_joins_the_sequential_batch(self):
        a = self.request(self.block_a, 'A', 0, 100, accept=15)
        b = self.request(self.block_a, 'B', 1, 3000, accept=9)
        foreign = self.request(self.block_b, 'X', 0, 100, accept=3, bind=False)
        outputs = self.step([entry(foreign), entry(a), entry(b)])
        self.assertEqual(self.block_a.calls, [('verify', ['A', 'B']), ('commit', 0, 16), ('commit', 1, 10)])
        self.assertEqual(self.block_b.calls, [])
        self.assertEqual(self.stepped, [('X', False)])
        self.assertEqual([output.request_id for output in outputs], ['X', 'A', 'B'])

    def test_a_cancellation_before_the_verify_is_answered_by_each_requests_own_step(self):
        entries = self.four()
        outputs = self.step(entries, cancelled=lambda: True)
        self.assertEqual(self.stepped, [('C', True), ('A', True), ('D', True), ('B', True)])
        self.assertTrue(all(output.cancelled for output in outputs))
        self.assertEqual((self.block_a.calls, self.block_b.calls), ([], []))

    def test_a_round_a_blocks_own_group_cannot_serve_degrades_without_raising_past_the_step(self):
        """A stale 16-row ticket for block A, whose partner is missing from this round, has
        no capture anywhere once its engine is trimmed to (1, 2, 4): the degrade-not-crash
        refusal (serving_packed_step.refuse_round) applies to block A's own group, while
        block B's intact pair still runs packed in the same round."""
        a = self.request(self.block_a, 'A', 0, 100, accept=15)
        a.engine.widths = (1, 2, 4)
        c = self.request(self.block_b, 'C', 0, 700, accept=0)
        d = self.request(self.block_b, 'D', 1, 5000, accept=12)
        with patch.dict(sys.modules, loguru=None), patch('sys.stdout', new_callable=io.StringIO) as output:
            outputs = self.step([entry(c), entry(a), entry(d)])
        self.assertEqual(self.block_a.calls, [], 'block A never ran: its own group could not serve the ticket')
        self.assertEqual(self.block_b.calls, [('verify', ['C', 'D']), ('commit', 0, 1), ('commit', 1, 13)])
        self.assertIn('cannot serve (entries=1 block_users=2) holds tickets no request '
                      'engine captured (request=A rows=16)', output.getvalue())
        self.assertEqual([output.request_id for output in outputs], ['C', 'A', 'D'])
        self.assertEqual(outputs[1], CommittedOutput('A', (), 100, True, True))
        self.assertEqual(a.session.phase, 'failed')
        self.assertEqual(a.engine.adopted, [])
        self.assertEqual(self.stepped, [], 'the degraded group is refused, not sent to the sequential step')

    def test_single_block_packed_step_still_runs_the_unchanged_packed_device_step(self):
        """A PackedStep built over ONE block runs `packed_device_step` directly, never the
        multi-block partition path - the single-block behavior stays exactly what it was."""
        block = FakeBlock(users=2)
        step = PackedStep(block)
        self.assertIs(step.block, block)
        a, b = FakeRequest('A', 100, self.stepped), FakeRequest('B', 3000, self.stepped)
        block.bind(a.engine, 0)
        block.bind(b.engine, 1)
        a.propose(block.predictions_for(0), accept=15)
        b.propose(block.predictions_for(1), accept=9)
        with patch('serving_packed_step.packed_device_rounds') as rounds:
            outputs = step([entry(a), entry(b)], cancelled=lambda: False)
        rounds.assert_not_called()
        self.assertEqual([output.request_id for output in outputs], ['A', 'B'])
        self.assertEqual(block.calls, [('verify', ['A', 'B']), ('commit', 0, 16), ('commit', 1, 10)])

    def test_the_packed_step_object_carries_several_blocks_and_their_shared_proposal_policy(self):
        step = PackedStep([self.block_a, self.block_b])
        self.assertEqual(step.blocks, (self.block_a, self.block_b))
        self.assertIsNone(step.block, 'no single block owns a multi-block round')
        requests = [FakeRequest(name, 100 + index * 50, self.stepped) for index, name in enumerate('ABCD')]
        for block, pair in ((self.block_a, requests[:2]), (self.block_b, requests[2:])):
            for segment, request in enumerate(pair):
                block.bind(request.engine, segment)
        self.assertEqual(step.proposal_rows(requests), 16)
        self.assertIsNone(step.proposal_rows(requests[:3]))
        with self.assertRaises(ValueError):
            PackedStep([])
        with self.assertRaises(ValueError):
            PackedStep(None)


class RealBlockTests(BlockFixture):
    """The step over the real PackedVerifierEngine (test_packed_verifier's fakes underneath:
    a fake ttnn, a fake ModelBatch retaining one record per layer, the commit DMA recorded).
    What only the real block can show: the segment each entry is served by is the pool slot
    its engine's carry came from, and the fence lands on the last commit of the round."""

    def setUp(self):
        super().setUp()
        # The fixture's readback ids run from 1000, past its 100-token vocabulary. Here the
        # tickets built from them go through the real block's staging validator, so the
        # readback gives segment u the ids 16u + row instead.
        self.ids.value = torch.arange(0, 32, dtype=torch.int32)

    def requests(self):
        """A admitted through pool slot 0 with page 7, B through slot 1 with page 11,
        presented B first."""
        stepped = []
        # Both inside the block's native chunk family [4096, 4352), as the serving pin keeps them.
        first = FakeRequest('A', 4100, stepped, carry=self.pool.slots[0].verifier.carry)
        second = FakeRequest('B', 4200, stepped, carry=self.pool.slots[1].verifier.carry)
        first.engine.pages = torch.full((1, PAGE_WIDTH), 7, dtype=torch.int32)
        second.engine.pages = torch.full((1, PAGE_WIDTH), 11, dtype=torch.int32)
        first.propose(list(range(0, 16)), accept=15)
        second.propose(list(range(16, 32)), accept=9)
        return first, second, stepped

    def test_the_real_block_serves_each_entry_from_its_slots_segment_and_fences_the_last_commit(self):
        block = self.build()
        first, second, stepped = self.requests()
        entries = [entry(second), entry(first)]
        executed = len(self.ttnn.executed)
        outputs = packed_device_step(entries, cancelled=lambda: False, block=block)
        self.assertEqual(stepped, [])
        self.assertEqual(outputs, [CommittedOutput('B', tuple(range(16, 26)), 4210, False),
                                   CommittedOutput('A', tuple(range(0, 16)), 4116, False)])
        # B's ticket and pages were staged into segment 1's rows, A's into segment 0's -
        # and into segment 1's and segment 0's own replay reader (its start word and tables)
        fixture = block.fixture
        self.assertEqual(fixture.tokens.value[16:32, 0].tolist(), [5, *range(16, 25), 0, 0, 0, 0, 0, 0])
        self.assertEqual(fixture.tokens.value[:16, 0].tolist(), [5, *range(0, 15)])
        self.assertTrue(bool((fixture.pages.value[16:] == 11).all()) and bool((fixture.pages.value[:16] == 7).all()))
        self.assertEqual(fixture.positions.value.tolist(), [*range(4100, 4116), *range(4200, 4216)])
        self.assertEqual(fixture.replay_reader.starts, (4100, 4200))
        self.assertEqual([bool((entry[1].value == page).all()) for own, page in zip(fixture.replay_reader.readers, (7, 11))
                          for entry in own.metadata], [True] * 4)
        # B, entry 0, was adopted at segment 1 (its carry is slot 1's); A, entry 1, at segment 0
        self.assertEqual([(ticket.request_id, segment) for ticket, unused, segment in second.engine.adopted], [('B', 1)])
        self.assertEqual([(ticket.request_id, segment) for ticket, unused, segment in first.engine.adopted], [('A', 0)])
        # the one verify trace, then B's prefix-10 trace and A's prefix-16 trace, in entries order
        self.assertEqual(self.ttnn.executed[executed:], ['trace1', block.commits[1][10], block.commits[0][16]])
        # the block's own fence rule, seen through the retained block it commits to: the
        # first commit of the round is not fenced, the last one is
        retained = block.fixture.retained
        self.assertEqual(retained.commits, [(1, 10), (0, 16)])
        self.assertEqual([call.kwargs['synchronize'] for call in retained.commit_user.call_args_list], [False, True])
        self.assertEqual((block.phase, block.pending_segments, block.rounds), ('idle', set(), 1))
        self.assertIsNone(verifier_engine._resident)
        # a cancellation after the verify: prefix 0 for both, the last one still the fence
        first.propose(list(range(0, 16)), accept=15)
        second.propose(list(range(16, 32)), accept=9)
        outputs = packed_device_step([entry(first), entry(second)], cancelled=answers(False, True), block=block)
        self.assertEqual([(output.request_id, output.cancelled) for output in outputs], [('A', True), ('B', True)])
        self.assertEqual(retained.commits[2:], [(0, 0), (1, 0)])
        self.assertEqual([call.kwargs['synchronize'] for call in retained.commit_user.call_args_list[2:]], [False, True])
        self.assertEqual(self.ttnn.executed[executed + 3:], ['trace1'], 'prefix 0 runs no trace')
        self.assertEqual(block.phase, 'idle')

    def test_the_real_block_hands_a_narrow_ticket_round_to_the_sequential_step(self):
        block = self.build()
        first, second, stepped = self.requests()
        first.session.pending, first.session.phase = None, 'idle'
        first.propose(list(range(0, 8)), accept=7, rows=8)
        executed = len(self.ttnn.executed)
        outputs = packed_device_step([entry(second), entry(first)], cancelled=lambda: False, block=block)
        self.assertEqual(stepped, [('B', False), ('A', False)])
        self.assertEqual([output.request_id for output in outputs], ['B', 'A'])
        self.assertEqual((self.ttnn.executed[executed:], block.rounds, block.phase), ([], 0, 'idle'))


class FourUserRealBlockTests(FourUserFixture):
    """The step over the real M3 block: four requests admitted through pool slots 0..3,
    presented C, A, D, B - each served from its slot's segment, its rows staged into that
    segment, its reader and cache tile carrying its pages, the last commit the fence."""

    def setUp(self):
        super().setUp()
        self.ids.value = torch.arange(0, 64, dtype=torch.int32)

    def requests(self):
        stepped = []
        owners = []
        for index, name in enumerate('ABCD'):
            owner = FakeRequest(name, self.POSITIONS[index], stepped, carry=self.pool.slots[index].verifier.carry)
            owner.engine.pages = torch.full((1, PAGE_WIDTH), self.PAGES[index], dtype=torch.int32)
            owners.append(owner)
        return owners, stepped

    def test_the_real_block_serves_four_entries_from_their_slots_segments(self):
        block = self.build()
        owners, stepped = self.requests()
        accepts = (15, 9, 0, 12)
        for index, owner in enumerate(owners):
            owner.propose(list(range(16 * index, 16 * index + 16)), accept=accepts[index])
        entries = [entry(owners[index]) for index in (2, 0, 3, 1)]
        executed = len(self.ttnn.executed)
        outputs = packed_device_step(entries, cancelled=lambda: False, block=block)
        self.assertEqual(stepped, [])
        self.assertEqual(outputs, [CommittedOutput('C', (32,), 4151, False),
                                   CommittedOutput('A', tuple(range(0, 16)), 4116, False),
                                   CommittedOutput('D', tuple(range(48, 61)), 4313, False),
                                   CommittedOutput('B', tuple(range(16, 26)), 4210, False)])
        fixture = block.fixture
        for index, owner in enumerate(owners):
            rows = slice(16 * index, 16 * index + 16)
            self.assertEqual(fixture.tokens.value[rows, 0].tolist()[0], 5, 'the seed token leads every segment')
            self.assertTrue(bool((fixture.pages.value[rows] == self.PAGES[index]).all()))
            self.assertEqual(fixture.positions.value[rows].tolist(), list(range(self.POSITIONS[index], self.POSITIONS[index] + 16)))
            self.assertEqual([(ticket.request_id, segment) for ticket, unused, segment in owner.engine.adopted],
                             [(owner.session.request_id, index)])
        self.assertEqual(fixture.replay_reader.starts, self.POSITIONS)
        self.assertEqual([bool((entry[1].value == page).all()) for own, page in zip(fixture.replay_reader.readers, self.PAGES)
                          for entry in own.metadata], [True] * 8)
        self.assertEqual([tile.pages.value[:, 0].tolist() for tile in fixture.cache_tiles],
                         [[7] * 16 + [11] * 16, [13] * 16 + [17] * 16])
        # the one verify trace, then C's prefix-1, A's prefix-16, D's prefix-13 and B's prefix-10 traces
        self.assertEqual(self.ttnn.executed[executed:], ['trace1', block.commits[2][1], block.commits[0][16],
                                                         block.commits[3][13], block.commits[1][10]])
        retained = fixture.retained
        self.assertEqual(retained.commits, [(2, 1), (0, 16), (3, 13), (1, 10)])
        self.assertEqual([call.kwargs['synchronize'] for call in retained.commit_user.call_args_list],
                         [False, False, False, True])
        self.assertEqual((block.phase, block.pending_segments, block.rounds), ('idle', set(), 1))
        self.assertIsNone(verifier_engine._resident)
        # two of four finished: the survivors' round goes to the sequential step, the block untouched
        for owner in owners[:2]:
            owner.propose(list(range(16)), accept=3)
        outputs = packed_device_step([entry(owners[1]), entry(owners[0])], cancelled=lambda: False, block=block)
        self.assertEqual(stepped, [('B', False), ('A', False)])
        self.assertEqual([output.request_id for output in outputs], ['B', 'A'])
        self.assertEqual((len(self.ttnn.executed) - executed, block.rounds), (5, 1))


if __name__ == '__main__':
    unittest.main()
