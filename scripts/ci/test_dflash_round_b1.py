"""QWEN_FAST_ROUND_B1 - build 1 of the round host-phase cuts - proved on the host.

Every cut in build 1 is token-exact by construction: the same device work per user, or
host bookkeeping only, or dropping work whose result nothing reads. What is pinned here:

  C1  one batched FP64 selection returns bit-identical tokens AND scores to the per-user
      selections it replaces, on real-shaped random inputs, with exact ties, shared ids
      and any thread count - and end to end through the pair trace and the coordinator;
  C2  the set-indexed retain() keeps, refuses, queues and releases exactly what the list
      scans did, on the same protected lists, and a whole draft-cache life is identical;
  C7  the skipped feature-history write is the only change, only when nothing on the path
      can read the history, and every reader of a stale history raises;
  C8  the pair update stops only the uploads the packed trace never reads, and the live
      rows sliced from one table build equal live_key_rope's bit for bit;
  M0a the publication splits are timing only - the same transport calls in the same order;
and that with the flag unset each path is the one that ran before. The B1 copies of flag-off
functions (the timed slide path, the B1 publication, the C1 readback) are pinned call for call
to the functions they copy, so an edit to one cannot silently miss the other.

QWEN_FAST_ROUND_B1_AUDIT, the hardware half's shadow check (a wrong draft token only lowers
acceptance, so the final text cannot see one): each audited cut re-done the flag-off way
beside the served result, a mismatch logged and raised, the running counts logged per round.

Per-round [PACKED] prefixes are not deterministic run to run on hardware (v126 vs v127),
so hardware exactness is the gate's final text against the references; this file is the
bit-level half of the proof.
"""

import os
import re
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import dflash_packed_proposal
import test_dflash_device_publish as publish_fixtures
import test_dflash_packed_proposal_coordinator as coordinator_fixtures
import test_dflash_packed_proposal_trace as trace_fixtures
import test_dflash_traced_publish as traced_fixtures
import test_draft_attention_branch_packed as branch_fixtures
from dflash_packed_proposal import select_packed, select_packed_batched, split_selection
from draft_selector import select_active_candidates


HERE = Path(__file__).parent
FLAG = 'QWEN_FAST_ROUND_B1'
VOCABULARY = 4096
RANK = 256


AUDIT_FLAG = 'QWEN_FAST_ROUND_B1_AUDIT'
# Non-empty inside audited(on=True): round_b1() then keeps the audit on for whatever it wraps.
_AUDITING = []


@contextmanager
def round_b1(on):
    """The flag exactly on or exactly absent, whatever the caller's environment holds - and
    its audit on only inside audited(), never from the caller's environment."""
    with patch.dict(os.environ):
        os.environ.pop(FLAG, None)
        os.environ.pop(AUDIT_FLAG, None)
        if on:
            os.environ[FLAG] = '1'
        if _AUDITING:
            os.environ[AUDIT_FLAG] = '1'
        yield


def fresh_audit_counts():
    return dict(rounds=0, **dict.fromkeys(dflash_packed_proposal.ROUND_B1_AUDIT_COUNTS, 0))


@contextmanager
def audited(on=True):
    """QWEN_FAST_ROUND_B1 on, the audit flag exactly on or absent, and fresh audit counts."""
    counts = fresh_audit_counts()
    if on:
        _AUDITING.append(True)
    try:
        with round_b1(True), patch.object(dflash_packed_proposal, '_ROUND_B1_AUDIT', counts):
            yield counts
    finally:
        if on:
            _AUDITING.pop()


@contextmanager
def printed_lines():
    """What the B1 log helpers print with loguru absent, as a list of strings."""
    lines = []
    with patch.dict('sys.modules', {'loguru': None}), \
            patch('builtins.print', side_effect=lambda message, **kwargs: lines.append(message)):
        yield lines


_ISOLATION = []


def setUpModule():
    """The B1 marker and the audit counts are once-per-process module state. The CI job runs
    many test modules in one process, so this module works on its own copies of both and puts
    the originals back after, and no later module sees a marker these tests already logged."""
    for name, value in (('_ROUND_B1_NOTED', []), ('_ROUND_B1_AUDIT', fresh_audit_counts())):
        patcher = patch.object(dflash_packed_proposal, name, value)
        patcher.start()
        _ISOLATION.append(patcher)


def tearDownModule():
    while _ISOLATION:
        _ISOLATION.pop().stop()


def bits(value):
    return value.contiguous().view(torch.int64 if value.dtype == torch.float64 else torch.int16)


def codebooks(generator, vocabulary=VOCABULARY):
    """Predecessor/successor codebooks as the lent weights hold them: bf16 values widened
    to FP64 (dflash_device.PreparedDraftWeights)."""
    return tuple(torch.randn((vocabulary, RANK), generator=generator).bfloat16().double() for _ in range(2))


def pair_parts(generator, *, pool=VOCABULARY, hidden=None, unary=None, candidates=None):
    """One pair's selector parts exactly as read_device_outputs hands them over: a (1, 32,
    256) bf16 selector projection and (1, 31, 16) merged candidates/scores, split by user."""
    projected = torch.randn((1, 32, RANK), generator=generator).bfloat16() if hidden is None else hidden
    if candidates is None:
        candidates = torch.stack([torch.randperm(pool, generator=generator)[:16] for _ in range(31)])[None]
    if unary is None:
        unary = torch.randn((1, 31, 16), generator=generator).bfloat16().float()
    return split_selection(projected, candidates, unary, 2, 16)


def per_user(parts, seeds, predecessors, successors):
    """Today's path: dflash_proposal_trace.PreparedPackedDFlashProposal.finish selects each
    pair through select_packed, one selector call per user."""
    tokens = []
    for index in range(0, len(parts), 2):
        tokens.extend(select_packed(parts[index:index + 2], seeds[index:index + 2], (15, 15), predecessors, successors))
    return tuple(tokens)


def call_sequence(function, *, skip=(), drop_first=False):
    """Every call `function` makes, in evaluation order, as its source text: a sub-call before
    the call it feeds, and `f(...)()` as f's call then '<invoke>' - as is a call of a local
    named `operation` (the timed slide path's split of that same call). Calls whose text
    starts with one of `skip` (timing and B1 bookkeeping) are left out. The docstring is
    ignored, and `drop_first` drops the first statement (a flag-off function's B1 dispatch)."""
    import ast
    import inspect
    import textwrap

    definition = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    body = list(definition.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if drop_first:
        body = body[1:]
    calls = []

    def visit(node):
        for child in ast.iter_child_nodes(node):
            visit(child)
        if not isinstance(node, ast.Call):
            return
        if isinstance(node.func, ast.Call) or (isinstance(node.func, ast.Name) and node.func.id == 'operation'):
            calls.append('<invoke>')
        elif not ast.unparse(node).startswith(tuple(skip)):
            calls.append(ast.unparse(node))
    for statement in body:
        visit(statement)
    return calls


class BatchedSelectionTests(unittest.TestCase):
    """C1: select_packed_batched against select_packed, per user."""

    def assert_scores_equal(self, parts, seeds, predecessors, successors):
        batched_tokens, batched_scores = select_active_candidates(
            torch.cat([part['hidden'] for part in parts]), torch.cat([part['candidates'] for part in parts]),
            torch.cat([part['unary'] for part in parts]), predecessors, successors,
            torch.tensor(seeds, dtype=torch.int64))
        ties = 0
        for index, (part, seed) in enumerate(zip(parts, seeds)):
            tokens, scores = select_active_candidates(part['hidden'], part['candidates'], part['unary'],
                predecessors, successors, torch.tensor([seed], dtype=torch.int64))
            self.assertTrue(torch.equal(tokens[0], batched_tokens[index]), 'user %d tokens' % index)
            self.assertTrue(torch.equal(bits(scores[0]), bits(batched_scores[index])), 'user %d FP64 scores' % index)
            ties += int(((scores[0] == scores[0].max(dim=-1, keepdim=True).values).sum(dim=-1) > 1).sum())
        return ties

    def test_random_rounds_match_the_per_user_selection_bit_for_bit(self):
        for seed in range(40):
            with self.subTest(seed=seed):
                generator = torch.Generator().manual_seed(seed)
                predecessors, successors = codebooks(generator)
                parts = [*pair_parts(generator), *pair_parts(generator)]
                seeds = [int(value) for value in torch.randint(0, VOCABULARY, (4,), generator=generator)]
                self.assertEqual(select_packed_batched(parts, seeds, (15,) * 4, predecessors, successors),
                                 per_user(parts, seeds, predecessors, successors))
                self.assert_scores_equal(parts, seeds, predecessors, successors)

    def test_exact_ties_resolve_to_the_same_candidate(self):
        """Two kinds of exact tie. Users 0-1: a zero selector row makes every edge exactly
        zero, so the score is the unary alone, and the unary repeats its maximum. Users 2-3:
        three candidates per position share one successor row and one (dominant) unary, so
        their scores are the same FP64 value. argmax keeps the first either way."""
        generator = torch.Generator().manual_seed(7)
        predecessors, successors = codebooks(generator)
        unary = torch.randn((1, 31, 16), generator=generator).bfloat16().float()
        unary[..., [0, 5, 9]] = 4.0
        zero_pair = pair_parts(generator, hidden=torch.zeros((1, 32, RANK), dtype=torch.bfloat16), unary=unary)
        candidates = torch.stack([torch.randperm(VOCABULARY, generator=generator)[:16] for _ in range(31)])[None]
        tied = unary.clone()
        tied[..., [2, 6, 11]] = 900.0
        for row in range(31):
            first = candidates[0, row, 2]
            for column in (6, 11):
                successors[candidates[0, row, column]] = successors[first]
        tie_pair = pair_parts(generator, candidates=candidates, unary=tied)
        parts = [*zero_pair, *tie_pair]
        seeds = [1, 2, 3, 4]
        self.assertEqual(select_packed_batched(parts, seeds, (15,) * 4, predecessors, successors),
                         per_user(parts, seeds, predecessors, successors))
        self.assertGreater(self.assert_scores_equal(parts, seeds, predecessors, successors), 0,
                           'the fixture must actually produce exact ties')

    def test_users_sharing_every_id_and_anchor(self):
        """torch.unique folds shared ids into one row of the union; the rows it gathers hold
        the same values, so sharing changes nothing."""
        generator = torch.Generator().manual_seed(11)
        predecessors, successors = codebooks(generator)
        candidates = torch.stack([torch.randperm(64, generator=generator)[:16] for _ in range(31)])[None]
        parts = [*pair_parts(generator, candidates=candidates), *pair_parts(generator, candidates=candidates)]
        seeds = [5, 5, 5, 5]
        self.assertEqual(select_packed_batched(parts, seeds, (15,) * 4, predecessors, successors),
                         per_user(parts, seeds, predecessors, successors))
        self.assert_scores_equal(parts, seeds, predecessors, successors)

    def test_the_thread_count_changes_nothing(self):
        generator = torch.Generator().manual_seed(13)
        predecessors, successors = codebooks(generator)
        parts = [*pair_parts(generator), *pair_parts(generator)]
        seeds = [10, 20, 30, 40]
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            expected = per_user(parts, seeds, predecessors, successors)
            for count in (1, 2, 4, 8):
                torch.set_num_threads(count)
                self.assertEqual(select_packed_batched(parts, seeds, (15,) * 4, predecessors, successors), expected)
        finally:
            torch.set_num_threads(threads)

    def test_counts_slice_each_user_like_select_packed(self):
        generator = torch.Generator().manual_seed(17)
        predecessors, successors = codebooks(generator)
        parts = [*pair_parts(generator), *pair_parts(generator)]
        seeds = [1, 2, 3, 4]
        counts = (1, 15, 7, 3)
        batched = select_packed_batched(parts, seeds, counts, predecessors, successors)
        self.assertEqual([len(tokens) for tokens in batched], list(counts))
        full = per_user(parts, seeds, predecessors, successors)
        self.assertEqual(batched, tuple(tokens[:count] for tokens, count in zip(full, counts)))

    def test_bad_counts_and_arity_are_refused_as_select_packed_refuses_them(self):
        generator = torch.Generator().manual_seed(19)
        predecessors, successors = codebooks(generator)
        parts = list(pair_parts(generator))
        for counts in ((0, 15), (15, 16), (15, 15.0)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                select_packed_batched(parts, (1, 2), counts, predecessors, successors)
        for arguments in (((1,), (15, 15)), ((1, 2), (15,))):
            with self.assertRaises(ValueError):
                select_packed_batched(parts, *arguments, predecessors, successors)
        with self.assertRaises(ValueError):
            select_packed_batched([], (), (), predecessors, successors)

    def test_a_non_finite_operand_fails_the_whole_batch(self):
        """The one behavioural difference, and the reason it is safe: an operand that fails
        one user's selection today fails the round one call later; batched it fails at once."""
        generator = torch.Generator().manual_seed(23)
        predecessors, successors = codebooks(generator)
        parts = [*pair_parts(generator), *pair_parts(generator)]
        parts[2]['unary'][0, 3, 4] = float('nan')
        with self.assertRaises(ValueError):
            per_user(parts, [1, 2, 3, 4], predecessors, successors)
        with self.assertRaises(ValueError):
            select_packed_batched(parts, [1, 2, 3, 4], (15,) * 4, predecessors, successors)


def device_outputs(generator, vocabulary=VOCABULARY):
    """One pair's head outputs as the fake runtime reads them back: four vocabulary chunks
    per chip, top-16 values and local indices per row, and the replicated selector
    projection. Chip 0's first chunk dominates, so every merged candidate is a token below
    `vocabulary` and small codebooks serve the real merge."""
    from draft_shared_head import candidate_chunks

    chunks = []
    for index, (start, stop) in enumerate(candidate_chunks()):
        values, indices = [], []
        for chip in range(2):
            winning = index == 0 and chip == 0
            score = torch.rand((32, 16), generator=generator) + (10.0 if winning else 0.0)
            values.append(score.sort(dim=-1, descending=True).values.bfloat16())
            indices.append(torch.stack([torch.randperm(vocabulary if winning else stop - start, generator=generator)[:16]
                                        for _ in range(32)]).int())
        chunks.append(dict(start=start, stop=stop, values=values, indices=indices))
    projected = torch.randn((1, 1, 32, RANK), generator=generator).bfloat16()
    return SimpleNamespace(chunks=chunks, projected=[projected, projected.clone()])


def reading_operations(operations=None):
    operations = operations or SimpleNamespace()
    operations.get_device_tensors = lambda value: value
    operations.to_torch = lambda value: value
    return operations


class ReadDeviceOutputsTests(unittest.TestCase):
    def test_select_device_outputs_is_select_packed_over_read_device_outputs(self):
        """What lets C1 read without selecting: select_device_outputs hands select_packed
        exactly read_device_outputs' parts, the seeds, the counts and the device's own
        codebooks."""
        generator = torch.Generator().manual_seed(29)
        predecessors, successors = codebooks(generator)
        device = SimpleNamespace(operations=reading_operations(), predecessors=predecessors, successors=successors)
        outputs = device_outputs(generator)
        expected = dflash_packed_proposal.read_device_outputs(device, outputs, 2, 16)
        with patch('dflash_packed_proposal.select_packed', return_value='chosen') as select:
            self.assertEqual(dflash_packed_proposal.select_device_outputs(device, outputs, (3, 4), (15, 15), 2, 16), 'chosen')
        parts, seeds, counts, given_predecessors, given_successors = select.call_args.args
        self.assertEqual((seeds, counts), ((3, 4), (15, 15)))
        self.assertIs(given_predecessors, predecessors)
        self.assertIs(given_successors, successors)
        for part, reference in zip(parts, expected, strict=True):
            for name in ('hidden', 'candidates', 'unary'):
                self.assertTrue(torch.equal(part[name], reference[name]))

    def test_the_copy_is_select_device_outputs_up_to_its_selection(self):
        """select_device_outputs keeps the text that ran before B1 (it never goes through the
        copy), and read_device_outputs makes the same calls up to, not including, its final
        select_packed - so an edit to either that the other misses fails here."""
        import inspect

        from dflash_packed_proposal import read_device_outputs, select_device_outputs

        self.assertNotIn('read_device_outputs', inspect.getsource(select_device_outputs))
        flag_off = call_sequence(select_device_outputs)
        self.assertTrue(flag_off[-1].startswith('select_packed(split_selection('))
        self.assertEqual(call_sequence(read_device_outputs), flag_off[:-1])


def build_pair(seed, *, predecessors, successors, generator, transients=None):
    """A real PreparedPackedDFlashProposal over the trace test's fake devices, prepared for
    (seed, seed + 1), whose pending outputs are real-shaped head readbacks. `transients`
    is what the pair update's retain scope hands back as its owned list."""
    from dflash_proposal_trace import PreparedPackedDFlashProposal

    operations = reading_operations(trace_fixtures.fake_operations())
    mesh = trace_fixtures.FakeMesh()
    device_a = trace_fixtures.fake_device(operations, mesh, position=4096, history_rows=300, name='a')
    device_b = trace_fixtures.fake_device(operations, mesh, position=1200, history_rows=300, name='b')
    device_a.predecessors, device_a.successors = predecessors, successors
    if transients is not None:
        device_a.temporaries = lambda protected: (list(transients), lambda value: value)
    trace = PreparedPackedDFlashProposal(device_a, device_b)
    trace.prepare_device(seed, seed + 1)
    trace._pending[2].outputs = device_outputs(generator)
    return trace


@contextmanager
def fake_trace_runtime(released=None):
    with patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)), \
            patch('dflash_proposal_trace.release_owned',
                  side_effect=(lambda operations, values: released.append(list(values))) if released is not None else None):
        yield


class PairCollectAdoptTests(unittest.TestCase):
    """C1 through the real pair trace: collect() + one batched selection + adopt() hands
    phase B's finish() the tokens finish() would have selected itself."""

    def rounds(self, batched):
        generator = torch.Generator().manual_seed(31)
        predecessors, successors = codebooks(generator)
        with fake_trace_runtime(), round_b1(batched):
            pairs = [build_pair(seed, predecessors=predecessors, successors=successors,
                                generator=torch.Generator().manual_seed(100 + seed)) for seed in (40, 50)]
            if batched:
                from dflash_packed_proposal_coordinator import select_round

                with patch('dflash_packed_proposal.select_packed_batched',
                           side_effect=select_packed_batched) as select:
                    select_round([([0, 1], pairs[0]), ([2, 3], pairs[1])], [], 1)
                self.assertEqual(select.call_count, 1, 'one selector call for both pairs')
                self.assertEqual(len(select.call_args.args[0]), 4)
            return [trace.finish(which, 15) for trace in pairs for which in ('a', 'b')]

    def test_batched_round_tokens_equal_finish_selected_tokens(self):
        self.assertEqual(self.rounds(True), self.rounds(False))

    def test_collect_releases_once_and_a_later_discard_releases_nothing_twice(self):
        generator = torch.Generator().manual_seed(37)
        predecessors, successors = codebooks(generator)
        released = []
        transients = [object(), object()]
        with fake_trace_runtime(released):
            trace = build_pair(60, predecessors=predecessors, successors=successors, generator=generator,
                               transients=transients)
            released.clear()
            self.assertEqual(trace._pending[3], transients)
            result = trace.collect()
            self.assertEqual(released, [transients], 'the pending transients, released as finish() releases them')
            self.assertEqual((result['seeds'], result['counts']), ((60, 61), (15, 15)))
            self.assertEqual(len(result['parts']), 2)
            with self.assertRaises(ValueError):
                trace.adopt([(1,)])
            trace.discard_pending()
        self.assertEqual(released[1:], [[]], 'nothing released a second time')

    def test_adopt_and_collect_refuse_out_of_turn(self):
        generator = torch.Generator().manual_seed(41)
        predecessors, successors = codebooks(generator)
        with fake_trace_runtime():
            trace = build_pair(70, predecessors=predecessors, successors=successors, generator=generator)
            trace.collect()
            trace.adopt([(1, 2), (3, 4)])
            with self.assertRaises(ValueError):
                trace.adopt([(1, 2), (3, 4)])
            with self.assertRaises(ValueError):
                trace.collect()
            self.assertEqual(trace.finish('b', 1), (3,))
            self.assertEqual(trace.finish('a', 2), (1, 2))
            with self.assertRaises(ValueError):
                trace.collect()
            with self.assertRaises(ValueError):
                trace.adopt([(1,), (2,)])

    def test_the_audit_reselects_through_the_flag_off_readback(self):
        """audit_selection is select_device_outputs over the pair's own outputs - the tokens
        finish() would have selected itself - and a batched selection that differs (anchors
        rotated between users) is caught."""
        from dflash_packed_proposal_coordinator import select_round

        generator = torch.Generator().manual_seed(53)
        predecessors, successors = codebooks(generator)

        def pairs():
            return [build_pair(seed, predecessors=predecessors, successors=successors,
                               generator=torch.Generator().manual_seed(200 + seed)) for seed in (40, 50)]
        with fake_trace_runtime():
            with audited(False):
                expected = [trace.finish(which, 15) for trace in pairs() for which in ('a', 'b')]
            with audited() as counts, printed_lines():
                served = pairs()
                select_round([([0, 1], served[0]), ([2, 3], served[1])], [], 1)
                self.assertEqual([trace.finish(which, 15) for trace in served for which in ('a', 'b')], expected)
                self.assertEqual((counts['select'], counts['rounds']), (4, 1))

                def rotated(parts, seeds, counts, predecessors, successors):
                    seeds = list(seeds)
                    return select_packed_batched(parts, seeds[1:] + seeds[:1], counts, predecessors, successors)
                broken = pairs()
                with patch('dflash_packed_proposal.select_packed_batched', side_effect=rotated), \
                        self.assertRaisesRegex(AssertionError, 'cut=C1'):
                    select_round([([0, 1], broken[0]), ([2, 3], broken[1])], [], 2)

    def test_moved_inputs_still_raise_after_releasing(self):
        generator = torch.Generator().manual_seed(43)
        predecessors, successors = codebooks(generator)
        released = []
        with fake_trace_runtime(released):
            trace = build_pair(80, predecessors=predecessors, successors=successors, generator=generator)
            trace._pending[2].addresses[0] = ('moved',)
            released.clear()
            with self.assertRaisesRegex(AssertionError, 'addresses moved'):
                trace.collect()
        self.assertEqual(len(released), 1)


class CollectingTrace(coordinator_fixtures.FakeTrace):
    """The coordinator test's FakeTrace plus collect()/adopt(), logging into `events`."""
    events = []

    def __init__(self, device_a, device_b):
        super().__init__(device_a, device_b)
        self.adopted = None
        self.collect_calls = 0
        self.collect_failure = None
        self.audit_calls = 0
        self.audit_reference = None

    def collect(self):
        self.collect_calls += 1
        CollectingTrace.events.append(('collect', self))
        if self.collect_failure is not None:
            raise self.collect_failure
        seed_a, seed_b = self.prepared[-1]
        return dict(parts=[('part', seed_a), ('part', seed_b)], seeds=(seed_a, seed_b), counts=(15, 15))

    def adopt(self, tokens):
        self.adopted = tuple(tokens)

    def audit_selection(self):
        self.audit_calls += 1
        return self.adopted if self.audit_reference is None else self.audit_reference


class RecordingSingleUserCapture(coordinator_fixtures.FakeSingleUserCapture):
    """The coordinator test's single-user capture, counting discard_pending() calls."""

    def __init__(self, device, *, max_new_tokens):
        super().__init__(device, max_new_tokens=max_new_tokens)
        self.discard_calls = 0

    def discard_pending(self):
        self.discard_calls += 1


class CoordinatorBatchedSelectTests(unittest.TestCase):
    """C1 in PackedProposalCoordinator.prepare: after the round's one fence, every prepared
    pair is collected and selected in one call, and the tokens go back to each trace."""

    def setUp(self):
        coordinator_fixtures.FakeTrace.instances = []
        CollectingTrace.events = []
        patcher = patch('dflash_proposal_trace.PreparedPackedDFlashProposal', CollectingTrace)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: CollectingTrace.events.append(('fence',))))
        self.mesh = object()
        self.books = (object(), object())

    def bridges(self, count=4):
        out = []
        for index in range(count):
            device = coordinator_fixtures.make_device(self.operations, self.mesh, slot=index)
            device.predecessors, device.successors = self.books
            out.append(coordinator_fixtures.make_bridge('r%d' % index, device, seed=100 + index))
        return out

    def test_four_users_are_selected_in_one_call_after_the_fence(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        tokens = ((1,), (2,), (3,), (4,))
        with round_b1(True), patch('dflash_packed_proposal.select_packed_batched', return_value=tokens) as select:
            PackedProposalCoordinator().prepare(self.bridges())
        select.assert_called_once()
        parts, seeds, counts, predecessors, successors = select.call_args.args
        self.assertEqual(parts, [('part', 100), ('part', 101), ('part', 102), ('part', 103)])
        self.assertEqual((seeds, counts), ([100, 101, 102, 103], [15, 15, 15, 15]))
        self.assertIs(predecessors, self.books[0])
        self.assertIs(successors, self.books[1])
        first, second = coordinator_fixtures.FakeTrace.instances
        self.assertEqual((first.adopted, second.adopted), (tokens[:2], tokens[2:]))
        self.assertEqual([event[0] for event in CollectingTrace.events], ['fence', 'collect', 'collect'],
                         'one fence, then the readbacks')

    def test_with_the_flag_off_nothing_is_collected_or_adopted(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        with round_b1(False), patch('dflash_packed_proposal.select_packed_batched') as select:
            PackedProposalCoordinator().prepare(self.bridges())
        select.assert_not_called()
        for trace in coordinator_fixtures.FakeTrace.instances:
            self.assertEqual((trace.collect_calls, trace.adopted), (0, None))

    def test_pairs_lending_different_codebooks_are_selected_separately(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges()
        bridges[2].request.runtime.drafter.predecessors = object()
        with round_b1(True), patch('dflash_packed_proposal.select_packed_batched',
                                   side_effect=[((1,), (2,)), ((3,), (4,))]) as select:
            PackedProposalCoordinator().prepare(bridges)
        self.assertEqual(select.call_count, 2)
        first, second = coordinator_fixtures.FakeTrace.instances
        self.assertEqual((first.adopted, second.adopted), (((1,), (2,)), ((3,), (4,))))

    def test_a_failed_collect_drops_every_prepared_pair_and_raises(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        coordinator = PackedProposalCoordinator()
        bridges = self.bridges()
        with round_b1(True), patch('dflash_packed_proposal.select_packed_batched', return_value=((1,), (2,), (3,), (4,))):
            coordinator.prepare(bridges)
            for trace in coordinator_fixtures.FakeTrace.instances:
                trace.collect_failure = AssertionError('Prepared packed proposal input addresses moved')
            with self.assertRaisesRegex(AssertionError, 'addresses moved'):
                coordinator.prepare(bridges)
        for trace in coordinator_fixtures.FakeTrace.instances:
            self.assertEqual(trace.discard_calls, 1, 'each shared trace discarded once')

    def test_a_failed_selection_also_drops_a_single_user_pending_under_a_pair_view(self):
        """Slot 1 finished: slot 0 is prepared on its own while still wearing the view of pair
        (0, 1), so its pending lives in the view's single-user capture, rebuilt for this round.
        When pair (2, 3)'s readback then fails, that capture's pending is discarded too, not
        only the pair traces'."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, _PackedCaptureView

        coordinator = PackedProposalCoordinator()
        bridges = self.bridges()
        with round_b1(True), patch('dflash_proposal_trace.PreparedDFlashProposal', RecordingSingleUserCapture), \
                patch('dflash_packed_proposal.select_packed_batched', return_value=((1,), (2,), (3,), (4,))):
            coordinator.prepare(bridges)
            first, second = coordinator_fixtures.FakeTrace.instances
            second.collect_failure = AssertionError('Prepared packed proposal input addresses moved')
            with self.assertRaisesRegex(AssertionError, 'addresses moved'):
                coordinator.prepare([bridges[0], bridges[2], bridges[3]])
        view = bridges[0].request.runtime.drafter.proposal_capture
        self.assertIsInstance(view, _PackedCaptureView)
        self.assertIsInstance(view._original, RecordingSingleUserCapture, 'rebuilt for the lone round')
        self.assertEqual(view._original.discard_calls, 1)
        self.assertEqual((first.discard_calls, second.discard_calls), (1, 1), 'each trace once')

    def test_the_audit_reselects_every_pair_and_logs_the_round(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        tokens = ((1,), (2,), (3,), (4,))
        with audited() as counts, printed_lines() as lines, \
                patch('dflash_packed_proposal.select_packed_batched', return_value=tokens):
            coordinator = PackedProposalCoordinator()
            bridges = self.bridges()
            coordinator.prepare(bridges)
            coordinator.prepare(bridges)
        for trace in coordinator_fixtures.FakeTrace.instances:
            self.assertEqual(trace.audit_calls, 2)
        self.assertEqual(counts['select'], 8)
        audit = [line for line in lines if line.startswith(dflash_packed_proposal.ROUND_B1_AUDIT_MARKER)]
        self.assertEqual(audit, ['[PINDIAG] round b1 audit 1 exact=True select=4 rope=0 retain=0 borrowed=0 release=0',
                                 '[PINDIAG] round b1 audit 2 exact=True select=8 rope=0 retain=0 borrowed=0 release=0'])

    def test_an_audit_mismatch_is_logged_raised_and_drops_the_round(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        with audited() as counts, printed_lines() as lines, \
                patch('dflash_packed_proposal.select_packed_batched', return_value=((1,), (2,), (3,), (4,))):
            coordinator = PackedProposalCoordinator()
            bridges = self.bridges()
            coordinator.prepare(bridges)
            first, second = coordinator_fixtures.FakeTrace.instances
            second.audit_reference = ((3,), (5,))
            with self.assertRaisesRegex(AssertionError, 'round b1 audit mismatch cut=C1'):
                coordinator.prepare(bridges)
        mismatches = [line for line in lines if line.startswith(dflash_packed_proposal.ROUND_B1_AUDIT_MISMATCH)]
        self.assertEqual(len(mismatches), 1)
        self.assertIn('cut=C1', mismatches[0])
        self.assertEqual((first.discard_calls, second.discard_calls), (1, 1))
        self.assertEqual(counts['rounds'], 1, 'no round line after the mismatch')

    def test_without_the_audit_flag_nothing_is_reselected(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        with audited(False) as counts, printed_lines() as lines, \
                patch('dflash_packed_proposal.select_packed_batched', return_value=((1,), (2,), (3,), (4,))):
            PackedProposalCoordinator().prepare(self.bridges())
        self.assertEqual([trace.audit_calls for trace in coordinator_fixtures.FakeTrace.instances], [0, 0])
        self.assertEqual(counts, fresh_audit_counts())
        self.assertFalse(any(dflash_packed_proposal.ROUND_B1_AUDIT_MARKER in line for line in lines))

    def test_the_audit_flag_without_b1_does_nothing(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        counts = fresh_audit_counts()
        with round_b1(False), patch.dict(os.environ, {AUDIT_FLAG: '1'}), \
                patch.object(dflash_packed_proposal, '_ROUND_B1_AUDIT', counts), \
                patch('dflash_packed_proposal.select_packed_batched') as select:
            PackedProposalCoordinator().prepare(self.bridges())
        select.assert_not_called()
        self.assertEqual([trace.audit_calls for trace in coordinator_fixtures.FakeTrace.instances], [0, 0])
        self.assertEqual(counts, fresh_audit_counts())

    def test_the_marker_is_logged_once_per_process(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        with patch.object(dflash_packed_proposal, '_ROUND_B1_NOTED', []), \
                patch('builtins.print') as printed, patch.dict('sys.modules', {'loguru': None}), round_b1(True), \
                patch('dflash_packed_proposal.select_packed_batched', return_value=((1,), (2,), (3,), (4,))):
            coordinator = PackedProposalCoordinator()
            bridges = self.bridges()
            coordinator.prepare(bridges)
            coordinator.prepare(bridges)
        lines = [call.args[0] for call in printed.call_args_list if dflash_packed_proposal.ROUND_B1_MARKER in call.args[0]]
        self.assertEqual(lines, ['[PINDIAG] round b1 engaged site=batched-select cuts=C1,C2,C7,C8,M0a'])


class FakeShard:
    def __init__(self, address):
        self.address = address

    def buffer_address(self):
        return self.address


class FakeTensor:
    reads = 0

    def __init__(self, first, second):
        self.shards = (FakeShard(first), FakeShard(second))


def counting_operations():
    def shards(tensor):
        FakeTensor.reads += 1
        return tensor.shards
    return SimpleNamespace(get_device_tensors=shards)


def identity_scenario(generator):
    """A protected list and probes covering every retain() outcome: the same object, another
    object with the same identity, a chip-0-only alias, a chip-1-only alias, and a fresh
    tensor - over ~110 protected identities, the served device's size."""
    addresses = torch.randperm(100000, generator=generator).tolist()
    take = iter(addresses).__next__
    protected = [FakeTensor(take(), take()) for _ in range(90)]
    protected.append(protected[3])
    borrowed = [FakeTensor(take(), take()) for _ in range(20)]
    everything = protected + borrowed
    probes = []
    for _ in range(300):
        kind = int(torch.randint(0, 5, (1,), generator=generator))
        other = everything[int(torch.randint(0, len(everything), (1,), generator=generator))]
        first, second = (shard.address for shard in other.shards)
        if kind == 0:
            probes.append(other)
        elif kind == 1:
            probes.append(FakeTensor(first, second))
        elif kind == 2:
            probes.append(FakeTensor(first, take()))
        elif kind == 3:
            probes.append(FakeTensor(take(), second))
        else:
            probes.append(FakeTensor(take(), take()))
    return protected, borrowed, probes


def run_retain(retain, probes):
    outcomes = []
    for probe in probes:
        try:
            outcomes.append(('kept', retain(probe) is probe))
        except ValueError as error:
            outcomes.append(('refused', str(error)))
    return outcomes


class DeviceTemporariesTests(unittest.TestCase):
    """C2 for DFlashDevice.temporaries: the set path and the scan path on the same lists."""

    def both(self, protected, borrowed, probes):
        from dflash_device import DFlashDevice

        results = []
        for on in (False, True):
            device = SimpleNamespace(operations=counting_operations(), borrowed=list(borrowed))
            with round_b1(on):
                owned, retain = DFlashDevice.temporaries(device, protected)
                outcomes = run_retain(retain, probes)
            results.append((outcomes, [id(value) for value in owned]))
        return results

    def test_set_lookups_decide_exactly_as_the_scans_do(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                protected, borrowed, probes = identity_scenario(torch.Generator().manual_seed(seed))
                scanned, indexed = self.both(protected, borrowed, probes)
                self.assertEqual(indexed, scanned)
                kinds = {outcome[0] for outcome in scanned[0]}
                self.assertEqual(kinds, {'kept', 'refused'}, 'the scenario must exercise both outcomes')
                self.assertTrue(scanned[1], 'and queue temporaries')

    def test_the_borrowed_addresses_are_read_once_per_borrowed_set(self):
        from dflash_device import DFlashDevice

        protected, borrowed, _ = identity_scenario(torch.Generator().manual_seed(3))
        device = SimpleNamespace(operations=counting_operations(), borrowed=list(borrowed))
        with round_b1(True):
            DFlashDevice.temporaries(device, protected)
            FakeTensor.reads = 0
            DFlashDevice.temporaries(device, protected)
            self.assertEqual(FakeTensor.reads, len(protected), 'only the protected list is re-read')
            self.assertEqual([id(value) for value in device._round_b1_borrowed[0]], [id(value) for value in borrowed],
                             'the cache holds the borrowed tensors themselves')
            device.borrowed[5] = FakeTensor(10 ** 7, 10 ** 7 + 1)
            FakeTensor.reads = 0
            owned, retain = DFlashDevice.temporaries(device, protected)
            self.assertEqual(FakeTensor.reads, len(protected) + len(borrowed), 'a changed set is re-read')
            with self.assertRaises(ValueError):
                retain(FakeTensor(10 ** 7, 5))
            device.borrowed.clear()
            owned, retain = DFlashDevice.temporaries(device, protected)
            fresh = FakeTensor(10 ** 7, 10 ** 7 + 1)
            self.assertIs(retain(fresh), fresh)
            self.assertEqual(owned, [fresh], 'a returned weight is no longer protected')

    def test_the_audit_checks_every_decision_and_changes_none(self):
        from dflash_device import DFlashDevice

        protected, borrowed, probes = identity_scenario(torch.Generator().manual_seed(21))
        with round_b1(False):
            device = SimpleNamespace(operations=counting_operations(), borrowed=list(borrowed))
            owned, retain = DFlashDevice.temporaries(device, protected)
            reference = (run_retain(retain, probes), [id(value) for value in owned])
        device = SimpleNamespace(operations=counting_operations(), borrowed=list(borrowed))
        with audited() as counts:
            DFlashDevice.temporaries(device, protected)
            owned, retain = DFlashDevice.temporaries(device, protected)
            self.assertEqual((run_retain(retain, probes), [id(value) for value in owned]), reference)
        self.assertEqual(counts['retain'], len(probes))
        self.assertEqual(counts['borrowed'], 1, 'the second scope reused the cache and re-read it')

    def test_the_audit_catches_a_stale_borrowed_cache(self):
        """A lent tensor that moved while lent - which C2's cache assumes cannot happen - is
        caught at the next scope; without the audit the stale address would be used."""
        from dflash_device import DFlashDevice

        protected, borrowed, _ = identity_scenario(torch.Generator().manual_seed(23))
        for on in (False, True):
            with self.subTest(audit=on):
                device = SimpleNamespace(operations=counting_operations(), borrowed=list(borrowed))
                with audited(on), printed_lines() as lines:
                    DFlashDevice.temporaries(device, protected)
                    moved = borrowed[4].shards[0].address
                    borrowed[4].shards[0].address = 10 ** 8
                    try:
                        if on:
                            with self.assertRaisesRegex(AssertionError, 'cut=C2 cached borrowed addresses are stale'):
                                DFlashDevice.temporaries(device, protected)
                            self.assertTrue(any('cut=C2' in line for line in lines))
                        else:
                            DFlashDevice.temporaries(device, protected)
                    finally:
                        borrowed[4].shards[0].address = moved

    def test_the_audit_catches_a_wrong_decision(self):
        """audited_retain reads each decision off what retain() did: one that keeps a partial
        alias, or queues a protected identity, disagrees with the scan."""
        from dflash_packed_proposal import audited_retain

        identify = lambda value: value
        protected_ids = [(1, 2), (3, 4)]
        for decision, probe in (('queue', (1, 9)), ('queue', (1, 2)), ('keep', (5, 6))):
            with self.subTest(decision=decision, probe=probe):
                owned = []

                def wrong(value):
                    if decision == 'queue':
                        owned.append(value)
                    return value
                with audited(), printed_lines(), self.assertRaisesRegex(AssertionError, 'cut=C2 retain'):
                    audited_retain(wrong, owned, identify, protected_ids)(probe)

    def test_the_flag_off_path_keeps_no_cache(self):
        from dflash_device import DFlashDevice

        protected, borrowed, _ = identity_scenario(torch.Generator().manual_seed(5))
        device = SimpleNamespace(operations=counting_operations(), borrowed=list(borrowed))
        with round_b1(False):
            DFlashDevice.temporaries(device, protected)
        self.assertFalse(hasattr(device, '_round_b1_borrowed'))


class CacheTemporariesTests(unittest.TestCase):
    """C2 for DraftKVHistory.temporaries: kept, refused, queued and released alike, including
    a value the scope adds to the cache's own owned list before it exits."""

    def run_scope(self, on, protected, borrowed, probes, owned_before, adopted):
        from draft_kv_history import DraftKVHistory

        released = []
        cache = SimpleNamespace(operations=counting_operations(), owned=list(owned_before),
                                projection=SimpleNamespace(owned=owned_before[:2]), borrowed=list(borrowed))
        with round_b1(on), patch('draft_kv_history.release_owned',
                                 side_effect=lambda operations, values: released.append([id(value) for value in values])):
            with DraftKVHistory.temporaries(cache, protected) as retain:
                outcomes = run_retain(retain, probes)
                cache.owned.extend(adopted)
        return outcomes, released

    def test_set_lookups_and_the_exit_filter_match_the_scans(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                generator = torch.Generator().manual_seed(seed)
                protected, borrowed, probes = identity_scenario(generator)
                owned_before = protected[:6]
                protected = protected[6:]
                fresh = [probe for probe in probes if not any(probe is value for value in protected + borrowed + owned_before)]
                adopted = fresh[:3]
                scanned = self.run_scope(False, protected, borrowed, probes, owned_before, adopted)
                indexed = self.run_scope(True, protected, borrowed, probes, owned_before, adopted)
                self.assertEqual(indexed, scanned)
                self.assertTrue(scanned[1][0], 'the scope released temporaries')

    def test_the_audit_checks_the_scope_and_changes_nothing(self):
        generator = torch.Generator().manual_seed(27)
        protected, borrowed, probes = identity_scenario(generator)
        owned_before, protected = protected[:6], protected[6:]
        fresh = [probe for probe in probes if not any(probe is value for value in protected + borrowed + owned_before)]
        reference = self.run_scope(False, protected, borrowed, probes, owned_before, fresh[:3])
        with audited() as counts:
            scoped = self.run_scope(True, protected, borrowed, probes, owned_before, fresh[:3])
        self.assertEqual(scoped, reference)
        self.assertEqual((counts['retain'], counts['release']), (len(probes), 1))

    def test_the_audit_catches_a_release_list_that_differs(self):
        from dflash_packed_proposal import audit_release

        first, second = object(), object()
        with audited(), printed_lines():
            audit_release([first, second], [first, second])
            for released in ([first], [second, first], [first, object()]):
                with self.subTest(released=len(released)), self.assertRaisesRegex(AssertionError, 'cut=C2 scope'):
                    audit_release([first, second], released)

    def test_an_exception_inside_the_scope_still_releases_alike(self):
        from draft_kv_history import DraftKVHistory

        protected, borrowed, probes = identity_scenario(torch.Generator().manual_seed(9))
        results = []
        for on in (False, True):
            released = []
            cache = SimpleNamespace(operations=counting_operations(), owned=[], projection=None, borrowed=list(borrowed))
            with round_b1(on), patch('draft_kv_history.release_owned',
                                     side_effect=lambda operations, values: released.append([id(value) for value in values])):
                with self.assertRaises(RuntimeError):
                    with DraftKVHistory.temporaries(cache, protected) as retain:
                        for probe in probes:
                            try:
                                retain(probe)
                            except ValueError:
                                pass
                        raise RuntimeError('publication failed')
            results.append(released)
        self.assertEqual(results[1], results[0])

    def test_a_whole_draft_cache_life_is_identical(self):
        """Construction, 32 prepare/commit rounds through the steady state, a discard and an
        audit, on real tensors (test_draft_kv_history's own fixture): the same banks bit for
        bit and the same deallocations, with and without the flag."""
        import test_draft_kv_history as history_fixtures

        lives = []
        for on in (False, True):
            case = history_fixtures.DraftKVHistoryTests('test_lent_banks_are_protected_from_the_temporaries')
            storage = history_fixtures.pooled_storage(layers=2)
            features = case.features(2048, 1)
            with round_b1(on), case.fixture(features, 4093, storage=storage) as (cache, operations):
                for prefix in range(1, 33):
                    candidate = case.features(32, cache.position)
                    publication = cache.prepare(candidate, prefix, position=cache.position)
                    if prefix == 5:
                        cache.discard(publication)
                        publication = cache.prepare(candidate, prefix, position=cache.position)
                    cache.commit(publication)
                    features = torch.cat((features, candidate[..., :prefix, :]), dim=2)[..., -2048:, :]
                cache.audit(features)
                banks = [bits(value).clone() for value in history_fixtures.bank_tensors(storage)]
                freed = [tuple(value.shape) for value in history_fixtures.freed(operations)]
            lives.append((banks, freed))
        for mine, theirs in zip(lives[1][0], lives[0][0], strict=True):
            self.assertTrue(torch.equal(mine, theirs))
        self.assertEqual(lives[1][1], lives[0][1])


def served_device(operations, *, kv_history, progress=None, capture=True, history_rows=2048):
    device = publish_fixtures.build_device(operations, kv_history=kv_history)
    device.history_rows = history_rows
    device.progress = progress
    device.proposal_capture = object() if capture else None
    return device


def fake_kv_history():
    return SimpleNamespace(prepare=Mock(side_effect=lambda projected, prefix, position: SimpleNamespace(
        projected=projected, prefix=prefix, position=position)), discard=Mock(), commit=Mock(), audit=Mock())


def recorded_publication(on, *, fused=True, merge_release=False, **device_options):
    """One prepare_publication on the device-publish test's mock runtime, returning the
    ordered operation names, the calls that touched either history buffer, and the device."""
    operations = publish_fixtures.fake_operations()
    order = Mock()
    for name in ('slice', 'pad', 'matmul', 'typecast', 'rms_norm', 'concat', 'copy', 'synchronize_device'):
        order.attach_mock(getattr(operations, name), name)
    kv_history = device_options.pop('kv_history', fake_kv_history())
    device = served_device(operations, kv_history=kv_history, **device_options)
    stack = publish_fixtures.patched(operations)
    with round_b1(on), stack[0], stack[1], stack[2], stack[3]:
        pending = device.prepare_publication([publish_fixtures.make_feature_tap() for _ in range(5)], 3,
            position=100, merge_release=merge_release, fused_steady_state=fused)
    names = [call[0] for call in order.mock_calls]
    buffers = (device.history, device.spare_history)
    touching = [call[0] for call in order.mock_calls
                if any(argument is buffer for argument in (*call[1], *call[2].values()) for buffer in buffers)]
    return names, touching, device, pending


class HistoryWriteTests(unittest.TestCase):
    """C7: the fused steady-state history write is skipped only where nothing can read it."""

    def test_the_served_state_drops_exactly_the_three_history_operations(self):
        for merge_release in (False, True):
            with self.subTest(merge_release=merge_release):
                off_names, off_touching, off_device, off_pending = recorded_publication(False, merge_release=merge_release)
                on_names, on_touching, on_device, on_pending = recorded_publication(True, merge_release=merge_release)
                self.assertEqual(off_touching, ['slice', 'copy'], 'today: the history slice and the copy into the spare')
                self.assertEqual(on_touching, [], 'no operation reads or writes either history buffer')
                self.assertEqual(off_names.count('concat'), 1, 'one projection chunk: the only concat is the history one')
                index = off_names.index('concat') - 1
                expected = off_names[:index] + off_names[index + 3:]
                self.assertEqual(off_names[index:index + 3], ['slice', 'concat', 'copy'])
                self.assertEqual(on_names, expected, 'every other operation, in the same order')
                self.assertTrue(on_device.history_stale)
                self.assertFalse(getattr(off_device, 'history_stale', False))
                self.assertIs(on_pending.history, on_device.spare_history, 'commit still swaps the same pair')
                self.assertEqual((on_pending.rows, on_pending.prefix), (off_pending.rows, off_pending.prefix))
                self.assertEqual(on_device.kv_history.prepare.call_args.args[1:], (3,))
                self.assertEqual(on_device.kv_history.prepare.call_args.kwargs, dict(position=100))

    def test_every_state_that_might_read_the_history_keeps_the_write(self):
        states = dict(audit=dict(progress=lambda *a, **k: None), no_cache=dict(kv_history=None),
                      eager=dict(capture=False), ramp=dict(history_rows=50), unfused=dict(fused=False))
        for label, options in states.items():
            with self.subTest(state=label):
                off = recorded_publication(False, **dict(options))
                on = recorded_publication(True, **dict(options))
                self.assertEqual(on[0], off[0])
                self.assertEqual(on[1], off[1])
                self.assertFalse(getattr(on[2], 'history_stale', False))

    def test_once_stale_the_history_stays_stale_across_commits(self):
        operations = publish_fixtures.fake_operations()
        device = served_device(operations, kv_history=fake_kv_history())
        device.published_rows = 0
        stack = publish_fixtures.patched(operations)
        from dflash_device import DFlashDevice

        device.commit_publication = lambda publication: DFlashDevice.commit_publication(device, publication)
        with round_b1(True), stack[0], stack[1], stack[2], stack[3]:
            for _ in range(3):
                publication = device.prepare_publication([publish_fixtures.make_feature_tap() for _ in range(5)], 2,
                    position=device.position, fused_steady_state=True)
                device.commit_publication(publication)
                self.assertTrue(device.history_stale)
        operations.copy.assert_not_called()

    def test_the_readers_of_a_stale_history_raise(self):
        from dflash_device import DFlashDevice
        from dflash_proposal_trace import PreparedDFlashProposal

        stale = SimpleNamespace(closed=False, pending=None, proposal_capture=None, max_drafts=15, history_stale=True)
        with self.assertRaisesRegex(ValueError, 'stale'):
            DFlashDevice.propose(stale, 7, 3)
        publication = SimpleNamespace(status='prepared', position=100, kv=object(), history=object(), rows=2048, prefix=2)
        auditing = SimpleNamespace(closed=False, pending=publication, position=100, kv_history=fake_kv_history(),
                                   history=object(), progress=lambda *a, **k: None, published_rows=0, history_stale=True)
        with self.assertRaisesRegex(ValueError, 'stale'):
            DFlashDevice.commit_publication(auditing, publication)
        auditing.kv_history.audit.assert_not_called()
        capture, bucket, device = self.single_user_capture(kv_history=None, history_stale=True)
        with patch('dflash_proposal_trace.release_owned'), self.assertRaisesRegex(ValueError, 'stale'):
            capture.update(bucket, 5)
        device.operations.slice.assert_not_called()

    def single_user_capture(self, *, kv_history, history_stale):
        from dflash_proposal_trace import PreparedDFlashProposal

        operations = trace_fixtures.fake_operations()
        device = SimpleNamespace(operations=operations, mesh=object(), position=4096, history_rows=2048, block_rows=16,
            progress=None, history=object(), spare_history=object(), history_stale=history_stale,
            live_query_qk=False, native_proposal_attention=False, owned=[],
            temporaries=lambda protected: ([], lambda value: value))
        capture = PreparedDFlashProposal.__new__(PreparedDFlashProposal)
        capture.device, capture.operations, capture.mesh = device, operations, device.mesh
        capture.owned, capture.kv_history = [], kv_history
        placeholder = lambda: SimpleNamespace(dtype='bf16', layout='tile')
        bucket = SimpleNamespace(context=2048, identifiers=placeholder(), mask=placeholder(),
            rope={name: (placeholder(), placeholder()) for name in ('q', 'k')}, history=placeholder(),
            cached_history=None if kv_history is None else [{name: placeholder() for name in ('k', 'v')}
                                                            for _ in kv_history.active])
        bucket.inputs, bucket.addresses = [], []
        return capture, bucket, device

    def test_the_served_single_user_proposal_never_hands_the_history_to_an_operation(self):
        """The fallback single-user trace on the served path (kv_history set, no audit)
        copies the K/V banks and nothing else: device.history is in no operation's
        arguments, stale or not - which is why skipping its write changes no proposal."""
        kv_history = SimpleNamespace(pending=None, position=4096, history_rows=2048, owned=[], borrowed=[],
                                     active=[{'k': object(), 'v': object()} for _ in range(5)])
        for stale in (False, True):
            with self.subTest(stale=stale):
                capture, bucket, device = self.single_user_capture(kv_history=kv_history, history_stale=stale)
                with patch('dflash_proposal_trace.release_owned'), \
                        patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)):
                    capture.update(bucket, 5)
                operations = device.operations
                calls = [*operations.slice.call_args_list, *operations.copy.call_args_list,
                         *operations.copy_host_to_device_tensor.call_args_list]
                self.assertEqual(operations.slice.call_count, 10, 'the five layers\' k and v banks')
                for call in calls:
                    self.assertFalse(any(argument is device.history for argument in (*call.args, *call.kwargs.values())))


class PairRopeTests(unittest.TestCase):
    """C8: only the dead rope['k'] uploads go, and the live rows are the same bits."""

    def test_live_rows_from_one_table_build_equal_live_key_rope_bit_for_bit(self):
        from dflash_batched_mask import live_key_rope, live_key_rope_from, packed_rope_tables

        generator = torch.Generator().manual_seed(47)
        geometries = [[(4096, 2048), (9000, 2048)], [(4093, 2048), (262000, 2048)], [(300, 300), (1200, 300)],
                      [(2048, 2048)], [(700, 256)], [(5000, 1024), (6000, 512)]]
        for _ in range(20):
            geometries.append([(int(torch.randint(2048, 262000, (1,), generator=generator)), 2048) for _ in range(2)])
        for geometry in geometries:
            users = [dict(position=position, history_rows=rows) for position, rows in geometry]
            with self.subTest(users=geometry):
                tables = packed_rope_tables(users, 16)
                for mine, reference in zip(live_key_rope_from(tables['k'], users, 16), live_key_rope(users, 16), strict=True):
                    self.assertTrue(torch.equal(bits(mine), bits(reference)))
        with self.assertRaises(ValueError):
            live_key_rope_from(packed_rope_tables([dict(position=4096, history_rows=2048)], 16)['k'],
                               [dict(position=4096, history_rows=1024)], 16)

    def uploads(self, on):
        """Every host-to-device copy one pair update makes: (destination, host tensor)."""
        with fake_trace_runtime():
            operations = trace_fixtures.fake_operations()
            operations.from_torch.side_effect = lambda value, *a, dtype=None, layout=None, **k: SimpleNamespace(
                dtype=dtype, layout=layout, value=value)
            mesh = trace_fixtures.FakeMesh()
            device_a = trace_fixtures.fake_device(operations, mesh, position=4096, history_rows=300, name='a')
            device_b = trace_fixtures.fake_device(operations, mesh, position=1200, history_rows=300, name='b')
            from dflash_proposal_trace import PreparedPackedDFlashProposal

            trace = PreparedPackedDFlashProposal(device_a, device_b)
            with round_b1(False):
                trace.prepare_device(11, 22)
            bucket = trace._pending[2]
            trace.discard_pending()
            operations.copy_host_to_device_tensor.reset_mock()
            device_a.position, device_b.position = 4103, 1215
            with round_b1(on):
                trace._update(bucket, 33, 44)
        return bucket, [(call.args[1], call.args[0].value) for call in operations.copy_host_to_device_tensor.call_args_list]

    def test_the_update_skips_only_the_key_table_uploads(self):
        def labelled(bucket, uploads):
            names = {id(bucket.identifiers): 'identifiers', id(bucket.rope['q'][0]): 'q0', id(bucket.rope['q'][1]): 'q1',
                     id(bucket.rope['k'][0]): 'k0', id(bucket.rope['k'][1]): 'k1',
                     id(bucket.rope['live_k'][0]): 'live0', id(bucket.rope['live_k'][1]): 'live1'}
            return [(names[id(destination)], value) for destination, value in uploads]

        before = labelled(*self.uploads(False))
        after = labelled(*self.uploads(True))
        self.assertEqual([name for name, _ in before], ['identifiers', 'q0', 'q1', 'k0', 'k1', 'live0', 'live1'])
        self.assertEqual([name for name, _ in after], ['identifiers', 'q0', 'q1', 'live0', 'live1'])
        kept = [(name, value) for name, value in before if name not in ('k0', 'k1')]
        for (name, mine), (_, reference) in zip(after, kept, strict=True):
            with self.subTest(upload=name):
                self.assertEqual(mine.dtype, reference.dtype)
                self.assertTrue(torch.equal(mine, reference) if mine.dtype == torch.int64
                                else torch.equal(bits(mine), bits(reference)))

    def test_the_audit_rebuilds_the_live_rows_and_catches_a_shift(self):
        with audited() as counts, printed_lines():
            bucket, audited_uploads = self.uploads(True)
            self.assertEqual(counts['rope'], 1)
        _, plain_uploads = self.uploads(True)
        self.assertEqual(len(audited_uploads), len(plain_uploads))
        for (_, mine), (_, reference) in zip(audited_uploads, plain_uploads):
            self.assertTrue(torch.equal(bits(mine), bits(reference)) if mine.dtype != torch.int64 else torch.equal(mine, reference))
        import dflash_batched_mask

        original = dflash_batched_mask.live_key_rope_from

        def shifted(tables, users, block_rows=16, **options):
            return tuple(torch.roll(table, 1, dims=2) for table in original(tables, users, block_rows, **options))
        with audited(), printed_lines() as lines, patch('dflash_batched_mask.live_key_rope_from', side_effect=shifted):
            with self.assertRaisesRegex(AssertionError, 'cut=C8'):
                self.uploads(True)
        self.assertTrue(any('round b1 audit mismatch cut=C8' in line for line in lines))

    def test_the_packed_branch_never_hands_the_key_table_to_an_operation(self):
        """What makes C8 exact: on the packed cached path draft_attention_branch only checks
        rope['k']'s shape. Walk every recorded call of the branch - ttnn operations, the
        rotary, the K/V projection, the native attention and the convolutions - and find
        neither table among the arguments, while rope['live_k'] and rope['q'] are there."""
        from dflash_batched_mask import key_value_plan

        case = branch_fixtures.PackedBranchTests('test_key_axis_is_assembled_exactly_as_the_plan_says')
        _, _, key_rows = key_value_plan(list(case.contexts), block_rows=16)
        operations, weights, convolution, tensors, rope, _ = case.fixture(key_rows)
        mocks = case.run_packed(operations, weights, convolution, tensors, rope)
        calls = [call for value in vars(operations).values() if isinstance(value, Mock) for call in value.call_args_list]
        calls.extend(operations.experimental.rotary_embedding_hf.call_args_list)
        for mock in mocks:
            if isinstance(mock, Mock):
                calls.extend(mock.call_args_list)

        def walk(value):
            yield value
            if isinstance(value, (list, tuple)):
                for item in value:
                    yield from walk(item)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)

        seen = [item for call in calls for item in walk((call.args, call.kwargs))]
        self.assertFalse(any(item is table for item in seen for table in rope['k']))
        self.assertTrue(any(item is rope['live_k'] for item in seen))
        self.assertTrue(any(item is rope['q'][0] for item in seen))


class PublicationSplitTests(traced_fixtures.FusedKVHistoryFixture, unittest.TestCase):
    """M0a: the splits are perf_counter reads around the same calls."""

    def slide_life(self, on, sink):
        import draft_kv_history
        import draft_kv_slide
        from dflash_traced_publish import PUBLICATION_SPLITS, install_fused_kv_history

        calls = []

        def transport(mesh, active, delta, spare, *, history_rows, prefix):
            operation = traced_fixtures.slide_style_transport(mesh, active, delta, spare,
                                                              history_rows=history_rows, prefix=prefix)
            roles = {id(bank[name]): (layer, side, name) for layer in range(5) for side, banks in
                     (('active', cache.active), ('spare', cache.spare)) for bank in [banks[layer]] for name in ('k', 'v')}
            calls.append(('build', roles.get(id(active)), roles.get(id(spare)), history_rows, prefix))

            def execute():
                calls.append(('execute',))
                return operation()
            return execute

        position = 4093
        features = self.features(2048, position)
        with round_b1(on), self.fixture(features, position, layers=5) as (cache, unused), \
                patch.object(draft_kv_slide, 'prepare', transport):
            marker, _ = traced_fixtures.build_recognizable_slide_candidate(transport)
            with patch.object(draft_kv_history.DraftKVHistory, 'prepare', marker):
                restore = install_fused_kv_history(cache)
            token = PUBLICATION_SPLITS.set(sink) if sink is not None else None
            try:
                for prefix in (1, 7, 15, 32):
                    cache.commit(cache.prepare(self.features(32, cache.position), prefix, position=cache.position))
            finally:
                if token is not None:
                    PUBLICATION_SPLITS.reset(token)
                restore()
            banks = [bits(pair[name]).clone() for pair in cache.active for name in ('k', 'v')]
        return calls, banks

    def test_the_timed_slide_path_makes_the_same_transport_calls_and_banks(self):
        reference_calls, reference_banks = self.slide_life(False, None)
        sink = {}
        calls, banks = self.slide_life(True, sink)
        self.assertEqual(calls, reference_calls)
        self.assertEqual(len([call for call in calls if call[0] == 'build']), 40, 'ten transports per publication')
        for mine, theirs in zip(banks, reference_banks, strict=True):
            self.assertTrue(torch.equal(mine, theirs))
        self.assertEqual(set(sink), {'kv_in', 'kv_proj', 'kv_build', 'kv_exec', 'kv_sync', 'kv_rel'})
        self.assertTrue(all(value >= 0.0 for value in sink.values()))

    def test_without_a_sink_the_flag_takes_the_untimed_path(self):
        reference_calls, reference_banks = self.slide_life(False, None)
        with patch('dflash_traced_publish._fused_kv_history_prepare_via_slide_timed') as timed:
            calls, banks = self.slide_life(True, None)
        timed.assert_not_called()
        self.assertEqual(calls, reference_calls)

    def test_prepare_publication_adds_its_phases_into_the_sink(self):
        from dflash_traced_publish import PUBLICATION_SPLITS

        sink = {}
        token = PUBLICATION_SPLITS.set(sink)
        try:
            recorded_publication(True)
        finally:
            PUBLICATION_SPLITS.reset(token)
        self.assertEqual(set(sink), {'proj', 'hist', 'kv', 'sync', 'rel'})
        names, touching, _, _ = recorded_publication(True)
        self.assertIsNone(PUBLICATION_SPLITS.get())


class CopyDriftTests(unittest.TestCase):
    """The B1 paths that copy a flag-off function make that function's calls, in its order,
    with its arguments, apart from a closed list of timing and B1 additions (review 0,
    finding 3: on the rig QWEN_FAST_PACKED_AUDIT is always set, so every B1 run serves the
    timed slide copy). An edit to one side that misses the other fails here."""

    def test_the_timed_slide_path_is_the_slide_path(self):
        from dflash_traced_publish import _fused_kv_history_prepare_via_slide, _fused_kv_history_prepare_via_slide_timed

        timing = ('clock()', 'add_split(')
        flag_off = call_sequence(_fused_kv_history_prepare_via_slide, drop_first=True)
        self.assertIn('<invoke>', flag_off)
        self.assertEqual(call_sequence(_fused_kv_history_prepare_via_slide_timed, skip=timing), flag_off)

    def test_the_b1_publication_is_the_publication_but_for_c7(self):
        from dflash_device import DFlashDevice

        additions = ('clock()', 'add_split(', 'note_round_b1(', 'PUBLICATION_SPLITS.get()',
                     "getattr(self, 'progress', None)", "getattr(self, 'proposal_capture', None)")
        flag_off = call_sequence(DFlashDevice.prepare_publication, drop_first=True)
        self.assertEqual(call_sequence(DFlashDevice._prepare_publication_round_b1, skip=additions), flag_off)

    def test_the_flag_off_functions_start_with_their_b1_dispatch(self):
        """What drop_first drops is the B1 dispatch and nothing else."""
        import ast
        import inspect
        import textwrap

        from dflash_device import DFlashDevice
        from dflash_traced_publish import _fused_kv_history_prepare_via_slide

        for function in (DFlashDevice.prepare_publication, _fused_kv_history_prepare_via_slide):
            with self.subTest(function=function.__name__):
                body = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0].body
                if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    body = body[1:]
                first = body[0]
                self.assertIsInstance(first, ast.If)
                self.assertTrue(ast.unparse(first.test).startswith("os.environ.get('QWEN_FAST_ROUND_B1') == '1'"))
                self.assertEqual([type(node) for node in first.body], [ast.Return])
                self.assertEqual(first.orelse, [])

    def test_the_drift_pin_sees_a_missed_edit(self):
        """call_sequence itself: one extra fence in a copy is a difference."""
        def original(operations, mesh):
            operations.copy(1, 2)
            operations.synchronize_device(mesh)

        def copied(operations, mesh):
            operations.copy(1, 2)
            operations.synchronize_device(mesh)
            operations.synchronize_device(mesh)
        self.assertNotEqual(call_sequence(copied), call_sequence(original))


class ShippingTests(unittest.TestCase):
    """Only an image build reaches the rig, and a module ships only if BOTH the Dockerfile
    and the image workflow copy it - otherwise the frozen bundle's version runs. Every
    module build 1 changes must be in both, and none of the pinned or frozen ones may be
    among the changed."""

    CHANGED = ('dflash_packed_proposal.py', 'dflash_packed_proposal_coordinator.py', 'dflash_proposal_trace.py',
               'dflash_batched_mask.py', 'dflash_device.py', 'draft_kv_history.py', 'dflash_traced_publish.py',
               'serving_packed_step.py')
    # Carries B1 code but is not an image module: the arm bind-mounts the checkout's gate.
    GATE = ('lever_n_m3native_gate.py',)
    # Ship from the frozen bundle, in neither copy list: an edit to one never reaches the rig.
    FROZEN = ('draft_selector.py', 'draft_kv_projection.py', 'draft_head_layout.py', 'feature_collective.py',
              'draft_kv_slide.py')
    # draft_kv_slide_gate's own pins.
    SLIDE_PINNED = ('draft_kv_slide.py', 'draft_kv_slide.cpp')
    # DraftKVHistory.prepare as draft_kv_slide_adapter.build_prepare matches it (the text before B1).
    PREPARE_SHA256 = 'fbc2586fa1b5a5ae89ff44ca4ab0e079f357a6cb7e19e2913ff45def81905409'

    @staticmethod
    def served_pins():
        """dflash_t16_native_attention_gate.SOURCES, the sources re-hashed at serve time
        (dflash_t16_native_scope.validate_record), read without importing the gate."""
        import ast

        tree = ast.parse((HERE / 'dflash_t16_native_attention_gate.py').read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(target, 'id', None) == 'SOURCES' for target in node.targets):
                return tuple(ast.literal_eval(node.value))
        raise AssertionError('dflash_t16_native_attention_gate.SOURCES not found')

    def test_no_pinned_or_frozen_module_is_among_the_changed(self):
        pinned = set(self.served_pins()) | set(self.FROZEN) | set(self.SLIDE_PINNED)
        self.assertTrue({'gdn_multitoken_conv.py', 'feature_projection.py', 'attention_batch.py'} <= pinned)
        self.assertEqual(pinned & set(self.CHANGED), set())
        for name in sorted(pinned):
            path = HERE / name
            if path.exists():
                with self.subTest(module=name):
                    self.assertIsNone(re.search('ROUND_B1|round_b1', path.read_text(encoding='utf-8', errors='replace')))

    def test_every_module_carrying_b1_code_is_listed(self):
        """So CHANGED cannot go stale: a module that gains B1 code must be listed here, and so
        checked against the pins and both copy lists."""
        carrying = {path.name for path in HERE.glob('*.py') if not path.name.startswith('test_')
                    and re.search('ROUND_B1|round_b1', path.read_text(encoding='utf-8', errors='replace'))}
        self.assertEqual(carrying, set(self.CHANGED) | set(self.GATE))

    def test_the_slide_adapter_still_matches_the_qualified_prepare_text(self):
        import draft_kv_history
        import draft_kv_slide_adapter

        source = (HERE / 'draft_kv_history.py').read_text(encoding='utf-8')
        _, record = draft_kv_slide_adapter.build_prepare(source, vars(draft_kv_history), lambda *args, **kwargs: None)
        self.assertEqual(record['original_prepare_sha256'], self.PREPARE_SHA256,
                         'DraftKVHistory.prepare must stay the text the slide adapter was qualified against')

    def test_every_changed_module_is_in_both_copy_lists(self):
        root = HERE.parent.parent
        dockerfile = (root / 'docker' / 'qwen-fast-serving.Dockerfile').read_text(encoding='utf-8')
        workflow = (root / '.github' / 'workflows' / 'qwen-fast-serving-image.yml').read_text(encoding='utf-8')
        copied = set(re.findall(r'scripts/ci/([A-Za-z0-9_]+\.py)', ' '.join(
            line for line in dockerfile.splitlines() if line.startswith('COPY '))))
        looped = set()
        for names in re.findall(r'for name in ([^;]+); do', workflow):
            looped.update(names.split())
        for name in self.CHANGED:
            with self.subTest(module=name):
                self.assertIn(name, copied)
                self.assertIn(name, looped)

    def test_the_flag_gates_every_changed_module(self):
        """dflash_batched_mask.py only gains live_key_rope_from, which nothing but the pair
        update's flagged branch calls; every other changed module reads the flag itself."""
        for name in self.CHANGED:
            source = (HERE / name).read_text(encoding='utf-8')
            if name == 'dflash_batched_mask.py':
                continue
            with self.subTest(module=name):
                self.assertTrue("os.environ.get('QWEN_FAST_ROUND_B1') == '1'" in source or 'round_b1_enabled()' in source
                                or 'ROUND_B1_FLAG' in source)
        callers = sorted(path.name for path in HERE.glob('*.py')
                         if not path.name.startswith('test_') and 'live_key_rope_from' in path.read_text(encoding='utf-8'))
        self.assertEqual(callers, ['dflash_batched_mask.py', 'dflash_proposal_trace.py'])


if __name__ == '__main__':
    unittest.main()
