"""tree_verify: the two-candidate accept/commit logic (docs/tp4-tree-verify.md), on the real GreedySession and a fake target.

What is held:
  - one candidate is today's round, call for call (the session ends every round in the state plain commit leaves it in);
  - exactness: whatever candidates a round is offered (right, wrong, shared first rows, shorter, repeated), the tokens a request emits are the target's own greedy
    continuation, round after round, to the budget and across an end-of-text row;
  - the choice: the winner emits the most tokens of any candidate, ties go to the primary, and `copy_back` is exactly the rounds a branch won;
  - the state routing of the sibling-segment design on a toy block: segments with their own recurrent state and K/V cells, the winner's state committed at its
    accepted prefix, its K/V rows copied back to the request's own cells when a branch won; after the run the request's state and K/V equal a plain decode's;
  - the planner gives branches only the block's free segments.
"""
import random
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'speculative-decoding/harness'))

import tree_verify as tree  # noqa: E402
from greedy_session import BlockTicket, GreedySession  # noqa: E402
from greedy_verify import select_prefix  # noqa: E402

VOCAB = 1000
EOS = 999


def make_stream(total, seed, eos_at=None):
    rng = random.Random(seed)
    out = []
    while len(out) < total:
        if len(out) > 40 and rng.random() < 0.4:
            start = rng.randrange(0, len(out) - 20)
            out.extend(out[start:start + rng.randrange(6, 20)])
        else:
            out.extend(rng.randrange(900) for _ in range(rng.randrange(2, 8)))
    out = out[:total]
    if eos_at is not None:
        out[eos_at] = EOS
    return out


class Target:
    """A target whose greedy continuation is the fixed `stream`: the row i of a chain at position p predicts stream[p + 1 + i]."""

    def __init__(self, stream):
        self.stream = stream

    def predictions(self, position, rows):
        return tuple(self.stream[position + 1 + i] for i in range(rows))


def chain(target, position, width, rng, quality):
    """A proposal chain of `width - 1` rows: each row the right token with probability `quality`, else a random one."""
    return tuple(target.stream[position + 1 + i] if rng.random() < quality else rng.randrange(900) for i in range(width - 1))


def new_session(stream, n, budget, eos_ids=()):
    prompt = tuple(stream[:n])
    session = GreedySession('r', prompt, stream[n], vocab_size=VOCAB, max_new_tokens=budget, eos_ids=eos_ids, neural={'dflash2': _no_drafter},
                            lookup_enabled=False)
    return session


def _no_drafter(request_id, history, count):
    """HybridDraft's neural slot: the tests stage the pending ticket themselves, so the drafter is never asked for rows."""
    return ()


def stage(session, tokens, source='dflash2'):
    """Make `tokens` (seed first) the session's pending ticket, as propose() would have."""
    session.epoch += 1
    ticket = BlockTicket(session.request_id, session.epoch, session.position, tuple(tokens), source, 0)
    session.pending, session.phase = ticket, 'pending'
    return ticket


def run_session(stream, n, budget, offers, *, eos_ids=(), width=16, tree_mode=True, rng=None):
    """Drive a session to its end. `offers(session, target, rng)` -> the candidate chains' proposals for the round (list, primary first). Returns (session,
    commits, log): the tree commits, and per round (candidates, winner, emitted)."""
    target = Target(stream)
    session = new_session(stream, n, budget, eos_ids)
    commits, log = [], []
    while not session.finished:
        chains = offers(session, target, rng)
        primary = stage(session, (session.seed,) + tuple(chains[0]))
        candidates = tree.build_candidates(primary, [('lookup', proposals) for proposals in chains[1:]])
        predictions = [target.predictions(session.position, len(candidate.tokens)) for candidate in candidates]
        published = []
        commit = tree.commit_tree(session, 'r', candidates, predictions, lambda branch: (lambda rows: published.append((branch, rows))))
        commits.append(commit)
        log.append((candidates, commit.branch, commit.decision.emitted, published))
    return session, commits, log


def random_offers(width=16, extra=2, quality=(0.5, 0.8)):
    def offers(session, target, rng):
        remaining = session.max_new_tokens - len(session.emitted)
        rows = max(1, min(width, remaining))
        chains = [chain(target, session.position, rows, rng, rng.uniform(*quality))]
        for _ in range(rng.randrange(0, extra + 1)):
            shorter = rng.randrange(2, rows + 1) if rows > 1 else rows
            chains.append(chain(target, session.position, shorter, rng, rng.uniform(0.2, 0.95)))
        return chains
    return offers


class SelectTests(unittest.TestCase):
    seed = 7

    def candidates(self, *chains):
        return tuple(tree.Candidate(i, 'x', (self.seed,) + tuple(chain_)) for i, chain_ in enumerate(chains))

    def select(self, candidates, predictions, **kwargs):
        return tree.select_branch(candidates, predictions, vocab_size=VOCAB, **kwargs)

    def test_the_longest_emission_wins_and_ties_go_to_the_primary(self):
        cands = self.candidates((1, 2, 3), (1, 2, 9), (1, 2, 3))
        # the target says 1, 2, 3, 4 after the seed: the primary accepts 3 (emits 4), the branch 1 accepts 2 (emits 3), the branch 2 is the primary's chain again
        predictions = [(1, 2, 3, 4)] * 3
        decision = self.select(cands, predictions)
        self.assertEqual((decision.branch, decision.copy_back, decision.emitted_by_branch), (0, False, (4, 3, 4)))
        self.assertEqual(decision.decision.emitted, (1, 2, 3, 4))
        # the branch wins only by strictly more
        cands = self.candidates((1, 9, 9), (1, 2, 3))
        decision = self.select(cands, [(1, 2, 3, 4)] * 2)
        self.assertEqual((decision.branch, decision.copy_back, decision.emitted_by_branch), (1, True, (2, 4)))
        self.assertEqual(decision.decision.emitted, (1, 2, 3, 4))
        self.assertEqual(decision.decision.state_rows, 4)
        equal = self.select(self.candidates((1, 2, 9), (1, 2, 8)), [(1, 2, 3, 4)] * 2)
        self.assertEqual((equal.branch, equal.copy_back), (0, False))

    def test_the_emissions_of_all_candidates_agree_on_their_common_prefix(self):
        stream = make_stream(400, 3)
        target = Target(stream)
        rng = random.Random(1)
        for position in range(50, 300, 7):
            chains = [chain(target, position, 16, rng, rng.uniform(0.3, 0.9)) for _ in range(3)]
            cands = self.candidates(*chains)
            cands = tuple(tree.Candidate(i, 'x', (stream[position],) + tuple(c)) for i, c in enumerate(chains))
            decisions = [tree.decide(c, target.predictions(position, 16), vocab_size=VOCAB, eos_ids=(), remaining=1 << 30, max_proposals=15) for c in cands]
            shortest = min(len(d.emitted) for d in decisions)
            self.assertEqual(len(set(d.emitted[:shortest] for d in decisions)), 1)
            for d in decisions:
                self.assertEqual(d.emitted, tuple(stream[position + 1:position + 1 + len(d.emitted)]))

    def test_a_single_candidate_is_select_prefix(self):
        cands = self.candidates((1, 2, 9, 4))
        decision = self.select(cands, [(1, 2, 3, 4, 5)])
        self.assertEqual(decision.decision, select_prefix((1, 2, 9, 4), (1, 2, 3, 4, 5), vocab_size=VOCAB))
        self.assertEqual((decision.branch, decision.copy_back), (0, False))

    def test_the_budget_and_the_commit_cap_cut_each_chain_as_commit_cuts_a_ticket(self):
        cands = self.candidates((1, 2, 3, 4), (1, 2, 3, 9, 9, 9))
        predictions = [(1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6, 7)]
        # 5 rows against a budget of 3 left: committed at most 3 tokens; the 7-row chain too
        decision = self.select(cands, predictions, remaining=3)
        self.assertEqual(decision.emitted_by_branch, (3, 3))
        self.assertEqual((decision.branch, decision.decision.emitted), (0, (1, 2, 3)))
        capped = self.select(cands, predictions, max_rows=2)
        self.assertEqual(capped.emitted_by_branch, (2, 2))
        # a chain no wider than the budget left is not cut by it
        self.assertEqual(self.select(cands, predictions, remaining=7).emitted_by_branch, (5, 4))

    def test_end_of_text_inside_an_accepted_row_ends_that_chain_without_a_correction(self):
        cands = self.candidates((1, EOS, 5), (1, 2, 3))
        predictions = [(1, EOS, 5, 6), (1, 2, 3, 4)]
        decision = self.select(cands, predictions, eos_ids=(EOS,))
        self.assertEqual(decision.emitted_by_branch, (2, 4))
        self.assertEqual(decision.branch, 1)
        self.assertFalse(decision.decision.finished)

    def test_refusals(self):
        good = self.candidates((1, 2, 3))
        for call in (lambda: self.select((), ()),
                     lambda: self.select(good, []),
                     lambda: self.select(good, [(1, 2)]),
                     lambda: self.select(good + (tree.Candidate(1, 'x', (8, 1, 2, 3)),), [(1, 2, 3, 4)] * 2),
                     lambda: self.select((tree.Candidate(1, 'x', (7, 1)),), [(1, 2)])):
            with self.assertRaises(ValueError):
                call()


class BuildTests(unittest.TestCase):
    def ticket(self, *tokens):
        return SimpleNamespace(tokens=tokens, source='dflash2')

    def test_alternatives_become_candidates_on_the_primarys_seed_and_duplicates_are_dropped(self):
        built = tree.build_candidates(self.ticket(7, 1, 2, 3), [('lookup', (1, 2, 3)), ('lookup', (1, 9, 9)), ('lookup', (4,)), ('lookup', ())])
        self.assertEqual([c.tokens for c in built], [(7, 1, 2, 3), (7, 1, 9, 9), (7, 4)])
        self.assertEqual([c.index for c in built], [0, 1, 2])
        self.assertEqual([c.source for c in built], ['dflash2', 'lookup', 'lookup'])

    def test_too_many_or_too_wide(self):
        with self.assertRaises(ValueError):
            tree.build_candidates(self.ticket(7, 1, 2), [('l', (3, 4)), ('l', (5, 6)), ('l', (6, 7)), ('l', (8, 9))])
        with self.assertRaises(ValueError):
            tree.build_candidates(self.ticket(7, 1, 2), [('l', (3, 4, 5))])
        with self.assertRaises(ValueError):
            tree.build_candidates(self.ticket(), [])
        self.assertEqual(len(tree.build_candidates(self.ticket(7, 1, 2), [('l', (3, 4)), ('l', (5, 6)), ('l', (6, 7))])), 4)


class SessionTests(unittest.TestCase):
    def test_one_candidate_leaves_the_session_where_plain_commit_does(self):
        for seed in range(6):
            stream = make_stream(500, seed)
            plain_rng, tree_rng = random.Random(seed), random.Random(seed)
            plain = new_session(stream, 50, 200)
            target = Target(stream)
            offers = random_offers(extra=0, quality=(0.6, 0.6))
            while not plain.finished:
                ticket = stage(plain, (plain.seed,) + tuple(offers(plain, target, plain_rng)[0]))
                plain.commit('r', ticket, target.predictions(plain.position, len(ticket.tokens)), lambda state_rows: None)
            tree_session, commits, _ = run_session(stream, 50, 200, random_offers(extra=0, quality=(0.6, 0.6)), rng=tree_rng)
            with self.subTest(seed=seed):
                self.assertEqual(tree_session.emitted, plain.emitted)
                self.assertEqual((tree_session.position, tree_session.seed, tree_session.epoch), (plain.position, plain.seed, plain.epoch))
                self.assertEqual((tree_session.committed_blocks, tree_session.accepted_proposals), (plain.committed_blocks, plain.accepted_proposals))
                self.assertFalse(any(commit.copy_back for commit in commits))

    def test_the_emitted_tokens_are_the_targets_greedy_continuation_whatever_is_offered(self):
        branch_wins = 0
        for seed in range(40):
            stream = make_stream(700, seed)
            rng = random.Random(1000 + seed)
            session, commits, log = run_session(stream, 60, 260, random_offers(width=rng.choice((4, 8, 16)), extra=3), rng=rng)
            with self.subTest(seed=seed):
                self.assertEqual(session.emitted, stream[60:60 + 260])
                self.assertEqual(session.position, 60 + len(session.emitted) - 1, 'the position is the last emitted token\'s: the seed is not consumed yet')
            for (candidates, winner, emitted, published), commit in zip(log, commits):
                self.assertEqual(commit.emitted_by_branch[winner], max(commit.emitted_by_branch))
                self.assertEqual(winner, commit.emitted_by_branch.index(max(commit.emitted_by_branch)))
                self.assertEqual(commit.copy_back, winner != 0)
                self.assertEqual(len(published), 1, 'one publication per round, by the winner\'s segment')
                self.assertEqual(published[0][0], winner)
                self.assertEqual(published[0][1], len(emitted))
                branch_wins += commit.copy_back
        self.assertGreater(branch_wins, 20, 'the offers were generated to make branches win sometimes')

    def test_a_tree_round_commits_at_least_what_every_single_chain_would(self):
        for seed in range(15):
            stream = make_stream(600, seed)
            rng = random.Random(seed)
            target = Target(stream)
            session = new_session(stream, 80, 400)
            while not session.finished:
                rows = max(1, min(16, session.max_new_tokens - len(session.emitted)))
                chains = [chain(target, session.position, rows, rng, rng.uniform(0.3, 0.9)) for _ in range(3)]
                alone = []
                for proposals in chains:
                    candidate = tree.Candidate(0, 'x', (session.seed,) + proposals)
                    alone.append(len(tree.decide(candidate, target.predictions(session.position, rows), vocab_size=VOCAB, eos_ids=(),
                                                 remaining=session.max_new_tokens - len(session.emitted), max_proposals=15).emitted))
                primary = stage(session, (session.seed,) + chains[0])
                candidates = tree.build_candidates(primary, [('lookup', c) for c in chains[1:]])
                predictions = [target.predictions(session.position, len(c.tokens)) for c in candidates]
                before = len(session.emitted)
                tree.commit_tree(session, 'r', candidates, predictions, lambda branch: (lambda state_rows: None))
                self.assertGreaterEqual(len(session.emitted) - before, max(alone[0], max(alone[1:]) if len(candidates) > 1 else 0))

    def test_end_of_text_ends_the_request_through_a_branch(self):
        stream = make_stream(300, 11, eos_at=90)
        rng = random.Random(4)
        session, commits, _ = run_session(stream, 50, 200, random_offers(extra=2, quality=(0.9, 1.0)), eos_ids=(EOS,), rng=rng)
        self.assertTrue(session.finished)
        self.assertEqual(session.emitted, stream[50:91])
        self.assertEqual(session.emitted[-1], EOS)

    def test_refusals_leave_the_session_untouched(self):
        stream = make_stream(300, 2)
        target = Target(stream)
        session = new_session(stream, 50, 100)
        primary = stage(session, (session.seed,) + tuple(stream[51:66]))
        candidates = tree.build_candidates(primary, [('lookup', tuple(stream[51:60]))])
        predictions = [target.predictions(50, len(c.tokens)) for c in candidates]
        with self.assertRaises(ValueError):
            tree.commit_tree(session, 'r', candidates, predictions, None)
        with self.assertRaises(ValueError):
            tree.commit_tree(session, 'r', candidates[::-1], predictions[::-1], lambda branch: (lambda rows: None))
        with self.assertRaises(ValueError):
            tree.commit_tree(session, 'r', candidates, predictions[:1], lambda branch: (lambda rows: None))
        other = SimpleNamespace(tokens=(1, 2, 3), source='x')
        with self.assertRaises(ValueError):
            tree.commit_tree(session, 'r', tree.build_candidates(other), [(1, 2, 3)], lambda branch: (lambda rows: None))
        self.assertIs(session.pending, primary)
        self.assertEqual((session.phase, session.position), ('pending', 50))
        commit = tree.commit_tree(session, 'r', candidates, predictions, lambda branch: (lambda rows: None))
        self.assertEqual(session.position, 50 + len(commit.decision.emitted))


class BlockSim:
    """A toy packed block: per segment a recurrent state (a hash chain over the rows consumed), K/V cells (position -> a hash of the token and everything before it)
    and checkpoints (the state after each prefix of the round's rows). A request owns segment 0 of its group; its branches own the group's others. The round forks
    the request's state into every segment, writes each segment's K/V into cells of its own, and commits only the winner: its state at the accepted prefix into the
    request's, and, if the winner is a branch, the K/V rows of that prefix into the request's own cells."""

    def __init__(self, prompt):
        self.state = 0
        self.kv = {}
        for position, token in enumerate(prompt):
            self.kv[position] = self.cell(self.state, token)
            self.state = self.step(self.state, token)
        self.copies = 0

    @staticmethod
    def step(state, token):
        return (state * 1000003 + token + 1) % (1 << 61)

    @staticmethod
    def cell(state, token):
        return ('kv', state, token)

    def verify(self, position, candidates):
        """Per candidate: (checkpoints[k] = state after k rows, cells = {position: cell}) in its own segment."""
        out = []
        for candidate in candidates:
            state, checkpoints, cells = self.state, [self.state], {}
            for offset, token in enumerate(candidate.tokens):
                cells[position + offset] = self.cell(state, token)
                state = self.step(state, token)
                checkpoints.append(state)
            out.append((checkpoints, cells))
        return out

    def commit(self, position, results, branch, state_rows):
        checkpoints, cells = results[branch]
        self.state = checkpoints[state_rows]
        if branch != 0:
            self.copies += 1          # the K/V rows of the accepted prefix move from the branch's cells to the request's
        for offset in range(state_rows):
            self.kv[position + offset] = cells[position + offset]


class StateRoutingTests(unittest.TestCase):
    def test_after_every_round_the_requests_state_and_kv_equal_a_plain_decodes(self):
        for seed in range(12):
            stream = make_stream(700, seed)
            n = 60
            rng = random.Random(seed)
            target = Target(stream)
            block = BlockSim(stream[:n])
            session = new_session(stream, n, 260)
            offers = random_offers(extra=3)
            copies_expected = 0
            while not session.finished:
                chains = offers(session, target, rng)
                primary = stage(session, (session.seed,) + tuple(chains[0]))
                candidates = tree.build_candidates(primary, [('lookup', c) for c in chains[1:]])
                predictions = [target.predictions(session.position, len(c.tokens)) for c in candidates]
                position = session.position
                results = block.verify(position, candidates)
                commit = tree.commit_tree(session, 'r', candidates, predictions, lambda branch: (lambda rows: block.commit(position, results, branch, rows)))
                copies_expected += commit.copy_back
                # the plain decode of the same emitted tokens: state and cells after consuming prompt + emitted[:-1] (the last emitted token is the seed, not yet consumed)
                reference = BlockSim(stream[:n + len(session.emitted) - 1])
                self.assertEqual(block.state, reference.state, (seed, session.position))
                self.assertEqual(block.kv, reference.kv, (seed, session.position))
            self.assertEqual(block.copies, copies_expected)

    def test_the_copy_back_is_needed_and_what_it_moves_is_the_accepted_prefix(self):
        # a deliberately wrong routing (the winner's state without its K/V rows) is caught by the same comparison
        stream = make_stream(400, 3)
        n, rng, target = 60, random.Random(3), Target(stream)
        block, session = BlockSim(stream[:n]), new_session(stream, n, 150)
        broken = False
        while not session.finished:
            chains = random_offers(extra=3)(session, target, rng)
            primary = stage(session, (session.seed,) + tuple(chains[0]))
            candidates = tree.build_candidates(primary, [('lookup', c) for c in chains[1:]])
            predictions = [target.predictions(session.position, len(c.tokens)) for c in candidates]
            position = session.position
            results = block.verify(position, candidates)
            holder = {}

            def publish_for(branch, results=results, position=position):
                def publish(rows):
                    checkpoints, cells = results[branch]
                    block.state = checkpoints[rows]
                    holder['branch'] = branch
                    if branch == 0:
                        for offset in range(rows):
                            block.kv[position + offset] = cells[position + offset]
                return publish
            tree.commit_tree(session, 'r', candidates, predictions, publish_for)
            if holder['branch'] != 0:
                broken = True
        reference = BlockSim(stream[:n + len(session.emitted) - 1])
        self.assertTrue(broken)
        self.assertEqual(block.state, reference.state, 'the recurrent state is right either way')
        self.assertNotEqual(block.kv, reference.kv, 'without the copy-back the winner\'s K/V rows are missing from the request\'s cells')


class PlanTests(unittest.TestCase):
    def test_a_full_block_plans_nothing(self):
        self.assertEqual(tree.plan_branches(4, (0, 1, 2, 3), {0: 1, 1: 1}), {})
        self.assertEqual(tree.plan_branches(8, tuple(range(8)), {0: 3}), {})

    def test_a_lone_user_gets_every_free_segment_up_to_the_candidate_limit(self):
        self.assertEqual(tree.plan_branches(4, (0,), {0: 3}), {0: [1, 2, 3]})
        self.assertEqual(tree.plan_branches(4, (0,), {0: 1}), {0: [1]})
        self.assertEqual(tree.plan_branches(8, (0,), {0: 9}), {0: [1, 2, 3]}, 'at most MAX_CANDIDATES chains per request')
        self.assertEqual(tree.plan_branches(4, (2,), {2: 0}), {})

    def test_free_segments_go_round_robin_in_priority_order(self):
        plan = tree.plan_branches(4, (0, 1), {1: 2, 0: 2})
        self.assertEqual(plan, {1: [2], 0: [3]})
        plan = tree.plan_branches(4, (0, 1, 3), {3: 1, 0: 1})
        self.assertEqual(plan, {3: [2]})
        plan = tree.plan_branches(8, (0, 1, 2), {2: 3, 0: 3, 1: 3})
        self.assertEqual(sorted(len(v) for v in plan.values()), [1, 2, 2])
        used = [segment for branches in plan.values() for segment in branches]
        self.assertEqual(len(used), len(set(used)), 'a segment carries one branch')
        self.assertFalse(set(used) & {0, 1, 2})

    def test_bad_input(self):
        for call in (lambda: tree.plan_branches(4, (0, 0), {}), lambda: tree.plan_branches(4, (4,), {})):
            with self.assertRaises(ValueError):
                call()


if __name__ == '__main__':
    unittest.main()
