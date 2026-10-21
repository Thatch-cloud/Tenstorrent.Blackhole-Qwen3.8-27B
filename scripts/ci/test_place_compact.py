"""tp4/place-1: compact placement (QWEN_FAST_TP4_PLACE_COMPACT, serving_buffer_pool.place_blocks / placement_slot) and its Markov model (place_model). CPU only.

Pinned here:
  - the flag is 0 or 1, strict; unset it is off, and with it off the placement is the rule of before the flag for EVERY set of lent slots (the reference below is that rule, copied);
  - on, the block is the fullest open block (the first on a tie), and the two rules differ ONLY where one block holds exactly one live user and the other two or three;
  - the pool refuses the flag at any pool size but eight, before it allocates, and writes the engaged line once;
  - the engine-reuse set asks the same function over its serving slots (R5), so a compact arrival takes the same slot there as in the pool;
  - passes per round, from arrival and leave sequences replayed through the real placement: hand-built sequences, and a seeded process in which both rules see the SAME turns;
  - place_model's rules equal the real function, its mixture reproduces the production decode-time share, and its numbers keep the order the document states.
"""

import heapq
import itertools
import os
from pathlib import Path
import random
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import place_model  # noqa: E402
import serving_buffer_pool  # noqa: E402
from serving_buffer_pool import PLACE_COMPACT_FLAG, ServingBufferPool, place_compact_enabled  # noqa: E402

BLOCKS = ((0, 1, 2, 3), (4, 5, 6, 7))


def before_the_flag(lent):
    """The placement of the tree before this lever, copied: the slot index, or None."""
    counts = [sum(1 for index in block if index in lent) for block in BLOCKS]
    open_blocks = [position for position, block in enumerate(BLOCKS) if counts[position] < len(block)]
    if not open_blocks:
        return None
    lone = [position for position in open_blocks if counts[position] == 1]
    chosen = lone[0] if lone else max(open_blocks, key=lambda position: (counts[position], -position))
    return next(index for index in BLOCKS[chosen] if index not in lent)


def pool_with(lent=(), compact=None, environ=None, log=None):
    """The pool's own placement over eight slots: place_blocks with the flag named (compact) or read from `environ`; its engaged line goes to `log` (a Mock) when given."""
    pool = SimpleNamespace(slots=[SimpleNamespace(index=index, lent=index in lent) for index in range(8)])
    pool.placement_slot = lambda live=None: ServingBufferPool.placement_slot(pool, live)
    with patch.dict(os.environ, environ or {}, clear=False), patch.object(serving_buffer_pool, 'pindiag', Mock() if log is None else log):
        ServingBufferPool.place_blocks(pool, BLOCKS, **({} if compact is None else {'compact': compact}))
    return pool


def placed(lent, compact):
    slot = pool_with(lent, compact).placement_slot()
    return None if slot is None else slot.index


def block_of(slot):
    return slot // 4


def passes(live):
    """The verify passes of a round with these live slots: one octo pass from six live, else one per block that holds a live user."""
    if len(live) >= 6:
        return 1
    return len({block_of(slot) for slot in live})


class Seats:
    """Eight seats placed by the real rule; `arrive` returns the slot, `leave` frees one."""

    def __init__(self, compact):
        self.pool = pool_with((), compact)
        self.live = set()

    def arrive(self):
        slot = self.pool.placement_slot()
        self.pool.slots[slot.index].lent = True
        self.live.add(slot.index)
        return slot.index

    def leave(self, slot):
        self.pool.slots[slot].lent = False
        self.live.remove(slot)

    def passes(self):
        return passes(self.live)

    def counts(self):
        return (sum(1 for slot in self.live if slot < 4), sum(1 for slot in self.live if slot >= 4))


class FlagTests(unittest.TestCase):
    def test_the_flag_is_strict(self):
        self.assertFalse(place_compact_enabled({}))
        self.assertFalse(place_compact_enabled({PLACE_COMPACT_FLAG: '0'}))
        self.assertTrue(place_compact_enabled({PLACE_COMPACT_FLAG: '1'}))
        for bad in ('', '2', 'true', 'on', ' 1', '01'):
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, PLACE_COMPACT_FLAG):
                place_compact_enabled({PLACE_COMPACT_FLAG: bad})

    def test_place_blocks_reads_the_flag_when_it_is_not_named(self):
        engaged = Mock()
        self.assertFalse(pool_with(log=engaged).__dict__['_compact'])
        self.assertFalse(pool_with(environ={PLACE_COMPACT_FLAG: '0'}, log=engaged).__dict__['_compact'])
        engaged.assert_not_called()
        self.assertTrue(pool_with(environ={PLACE_COMPACT_FLAG: '1'}, log=engaged).__dict__['_compact'])
        with self.assertRaisesRegex(ValueError, PLACE_COMPACT_FLAG):
            pool_with(environ={PLACE_COMPACT_FLAG: 'yes'}, log=engaged)
        self.assertEqual(engaged.call_count, 1, 'the engaged line is written once, when the flag is on, and never when it is off')
        self.assertEqual(engaged.call_args.args[0], serving_buffer_pool.PLACE_COMPACT_MARKER)
        self.assertEqual(engaged.call_args.args[1], [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertTrue(pool_with(compact=True).__dict__['_compact'])
        self.assertFalse(pool_with(compact=False, environ={PLACE_COMPACT_FLAG: '1'}).__dict__['_compact'], 'a named value wins over the environment')
        with self.assertRaisesRegex(ValueError, 'explicit bool'):
            pool_with(compact=1)

    def test_the_default_pool_has_no_placement_and_no_compact(self):
        self.assertFalse(hasattr(ServingBufferPool, '_blocks'))
        self.assertFalse(hasattr(ServingBufferPool, '_compact'))

    def test_the_pool_refuses_the_flag_at_any_size_but_eight_before_it_allocates(self):
        operations = Mock()
        with patch.dict(sys.modules, {'torch': Mock()}), patch.dict(os.environ, {PLACE_COMPACT_FLAG: '1'}):
            for users in (1, 2, 4, 7):
                with self.subTest(users=users), self.assertRaisesRegex(ValueError, 'exactly 8 pool slots'):
                    ServingBufferPool(operations, 'mesh', users=users)
        self.assertEqual(operations.mock_calls, [], 'nothing was asked of the device')

    def test_a_malformed_flag_is_refused_by_the_pool_too(self):
        with patch.dict(sys.modules, {'torch': Mock()}), patch.dict(os.environ, {PLACE_COMPACT_FLAG: 'x'}):
            with self.assertRaisesRegex(ValueError, PLACE_COMPACT_FLAG):
                ServingBufferPool(Mock(), 'mesh', users=8)


class RuleTests(unittest.TestCase):
    def test_off_is_the_placement_of_before_the_flag_for_every_set_of_lent_slots(self):
        for mask in range(256):
            lent = {index for index in range(8) if mask >> index & 1}
            for compact in (None, False):
                with self.subTest(lent=sorted(lent), compact=compact):
                    pool = pool_with(lent, compact, environ={} if compact is None else None)
                    slot = pool.placement_slot()
                    self.assertEqual(None if slot is None else slot.index, before_the_flag(lent))

    def test_the_rules_differ_only_where_one_block_holds_one_user_and_the_other_two_or_three(self):
        differing = 0
        for mask in range(256):
            lent = {index for index in range(8) if mask >> index & 1}
            counts = (sum(1 for index in lent if index < 4), sum(1 for index in lent if index >= 4))
            today, compact = placed(lent, False), placed(lent, True)
            if (1 in counts) and (set(counts) - {1}) and max(counts) in (2, 3):
                differing += 1
                fuller = counts.index(max(counts))
                self.assertEqual(block_of(compact), fuller, (sorted(lent), 'the fuller block'))
                self.assertEqual(block_of(today), counts.index(1), (sorted(lent), 'the lone user\'s block'))
                self.assertIn(compact, BLOCKS[fuller])
                self.assertNotIn(compact, lent)
            else:
                self.assertEqual(compact, today, sorted(lent))
        self.assertGreater(differing, 0)

    def test_compact_is_the_fullest_open_block_and_the_first_on_a_tie_and_the_lowest_free_slot(self):
        self.assertEqual(placed((), True), 0)
        self.assertEqual(placed((0,), True), 1)
        self.assertEqual(placed((4,), True), 5, 'B holds one and A none: the fuller')
        self.assertEqual(placed((0, 4), True), 1, 'one and one: the first block')
        self.assertEqual(placed((0, 1, 4), True), 2, 'A two, B one: A (today\'s rule joins B)')
        self.assertEqual(placed((0, 1, 2, 4), True), 3, 'A three, B one: A')
        self.assertEqual(placed((0, 1, 2, 3, 4), True), 5, 'A full: B')
        self.assertEqual(placed((1, 2, 4), True), 0, 'the lowest free slot of the block')
        self.assertIsNone(placed(range(8), True))

    def test_the_engine_reuse_set_asks_the_same_function_over_its_serving_slots(self):
        # R5: a parked slot is lent and not live. Every slot is lent (eight parked engines), slots 0, 1 and 4 serve: the count of lent slots calls both blocks full, the set's predicate
        # says A holds two and B one. Compact joins the fuller (A, slot 2); today's rule joins the lone user (B, slot 5).
        serving = {0, 1, 4}
        every = set(range(8))
        self.assertEqual(pool_with(every, True).placement_slot(live=lambda index: index in serving).index, 2)
        self.assertEqual(pool_with(every, False).placement_slot(live=lambda index: index in serving).index, 5)
        self.assertIsNone(pool_with(every, True).placement_slot(), 'without the predicate every block is full')

    def test_acquire_places_by_the_compact_rule_and_a_slot_order_still_wins(self):
        pool = SimpleNamespace(
            slots=[SimpleNamespace(index=index, lent=False, owner=None, zeroed=(), verifier=None, addresses=((0, 0),) * 3, kv=(), verify=lambda: None) for index in range(8)],
            closed=False, users=8, operations=SimpleNamespace(full_like=Mock()))
        pool.placement_slot = lambda: ServingBufferPool.placement_slot(pool)
        with patch.object(serving_buffer_pool, 'pindiag'):
            ServingBufferPool.place_blocks(pool, BLOCKS, compact=True)
            order = [ServingBufferPool.acquire(pool, owner=owner).index for owner in range(5)]
            self.assertEqual(order, [0, 1, 2, 3, 4])
            for index in (1, 2):                                   # A holds slots 0 and 3, B holds 4
                pool.slots[index].lent = False
            self.assertEqual(ServingBufferPool.acquire(pool, owner='x').index, 1, 'A (two) is fuller than B (one); today\'s rule gives 5')
            pool._slot_order = (6,)
            self.assertEqual(ServingBufferPool.acquire(pool, owner='y').index, 6)


class PassesPerRoundTests(unittest.TestCase):
    def trajectory(self, compact, script):
        """[(slot, (live in A, live in B), verify passes)] after each step of a script: 'a' an arrival, ('l', n) the leaving of the n-th oldest arrival still live."""
        seats, order, result = Seats(compact), [], []
        for step in script:
            if step == 'a':
                slot = seats.arrive()
                order.append(slot)
            else:
                slot = order.pop(step[1])
                seats.leave(slot)
            result.append((slot, seats.counts(), seats.passes()))
        return result

    def test_a_departure_at_five_live_then_an_arrival_then_the_lone_user_leaves(self):
        # five arrive (4 + 1); the first of block A leaves (3 + 1: four live in two blocks, both rules); one arrives; then the user of block B (the fifth arrival, now index 3) leaves
        script = ['a'] * 5 + [('l', 0), 'a', ('l', 3)]
        today, compact = self.trajectory(False, script), self.trajectory(True, script)
        self.assertEqual([row[1:] for row in today[:5]], [((1, 0), 1), ((2, 0), 1), ((3, 0), 1), ((4, 0), 1), ((4, 1), 2)])
        self.assertEqual(today[:5], compact[:5], 'arrivals alone: the same slots under both rules')
        self.assertEqual(today[5][1:], ((3, 1), 2))
        self.assertEqual(compact[5][1:], ((3, 1), 2), 'the departure is nobody\'s choice')
        self.assertEqual(today[6], (5, (3, 2), 2), 'today: the arrival joins the lone user of block B')
        self.assertEqual(compact[6], (0, (4, 1), 2), 'compact: the arrival refills block A')
        self.assertEqual(today[7][1:], ((3, 1), 2), 'today: four live, still two passes')
        self.assertEqual(compact[7][1:], ((4, 0), 1), 'compact: four live in one block, one pass')

    def test_a_fuller_block_a_does_not_give_its_arrival_to_the_lone_user_of_b(self):
        # five arrive; the two oldest (slots 0 and 1) leave: A holds two, B one. The arrival: today B (2 + 2), compact A (3 + 1). Four live, two passes either way ...
        script = ['a'] * 5 + [('l', 0), ('l', 0), 'a']
        today, compact = self.trajectory(False, script), self.trajectory(True, script)
        self.assertEqual(today[-1], (5, (2, 2), 2))
        self.assertEqual(compact[-1], (0, (3, 1), 2))
        # ... until the user of block B (the third oldest, slot 4) leaves: three live in two blocks today, in one block under compact
        script += [('l', 2)]
        today, compact = self.trajectory(False, script), self.trajectory(True, script)
        self.assertEqual(today[-1][1:], ((2, 1), 2))
        self.assertEqual(compact[-1][1:], ((3, 0), 1))

    def test_arrivals_alone_fill_block_a_first_under_both_rules(self):
        for compact in (False, True):
            rows = self.trajectory(compact, ['a'] * 8)
            self.assertEqual([row[0] for row in rows], list(range(8)))
            # five live need two M3 passes whatever the placement; six or more run the octo pass
            self.assertEqual([row[2] for row in rows], [1, 1, 1, 1, 2, 1, 1, 1])

    def test_a_seeded_process_with_the_same_turns_for_both_rules(self):
        """Turns arrive at rate 2 per mean turn, last Exp(1), and are admitted when a seat is free. Both rules see the SAME turns, so the live count at every instant is the same and only the
        split of the live users over the blocks differs. Time at two passes per live count: compact never more, and strictly less in total."""
        rng = random.Random(20261021)
        horizon, rate = 4000.0, 2.0
        turns, now = [], 0.0
        while now < horizon:
            now += rng.expovariate(rate)
            turns.append((now, rng.expovariate(1.0)))

        def run(compact):
            seats, ends, slot_of = Seats(compact), [], {}
            time_at, two_at, last = {}, {}, 0.0
            arrivals = iter(enumerate(turns))
            pending = next(arrivals, None)
            while pending is not None or ends:
                next_arrival = pending[1][0] if pending else float('inf')
                next_leave = ends[0][0] if ends else float('inf')
                moment = min(next_arrival, next_leave)
                live = len(seats.live)
                if live:
                    time_at[live] = time_at.get(live, 0.0) + (moment - last)
                    if seats.passes() == 2:
                        two_at[live] = two_at.get(live, 0.0) + (moment - last)
                last = moment
                if next_leave <= next_arrival:
                    _, number = heapq.heappop(ends)
                    seats.leave(slot_of.pop(number))
                else:
                    number, (start, length) = pending
                    pending = next(arrivals, None)
                    if len(seats.live) < 8:
                        slot_of[number] = seats.arrive()
                        heapq.heappush(ends, (start + length, number))
            return time_at, two_at

        today_time, today_two = run(False)
        compact_time, compact_two = run(True)
        self.assertEqual(today_time.keys(), compact_time.keys())
        for live in today_time:
            self.assertAlmostEqual(today_time[live], compact_time[live], places=6, msg='the live count does not depend on the placement')
        for live in (2, 3, 4):
            with self.subTest(live=live):
                self.assertLessEqual(compact_two.get(live, 0.0), today_two.get(live, 0.0) + 1e-9)
        self.assertEqual(today_two.get(5), today_time.get(5), 'five live always run two passes')
        self.assertEqual(compact_two.get(5), compact_time.get(5))
        self.assertLess(sum(compact_two.get(live, 0.0) for live in (2, 3, 4)), 0.95 * sum(today_two.get(live, 0.0) for live in (2, 3, 4)))


class ModelTests(unittest.TestCase):
    def test_the_models_rules_are_the_real_placement(self):
        for a, b in itertools.product(range(5), repeat=2):
            lent = set(range(a)) | set(range(4, 4 + b))
            for compact, rule in ((False, place_model.lone_first), (True, place_model.fullest_first)):
                with self.subTest(a=a, b=b, compact=compact):
                    slot = placed(lent, compact)
                    self.assertEqual(None if slot is None else block_of(slot), rule(a, b))

    def test_the_stationary_distribution_is_a_distribution_and_balances(self):
        for rule in (place_model.lone_first, place_model.fullest_first):
            for rho in (0.5, 2.0, 6.0):
                chain = place_model.stationary(rule, rho)
                self.assertAlmostEqual(sum(chain.values()), 1.0, places=9)
                self.assertTrue(all(value >= -1e-12 for value in chain.values()))
                # the live count does not depend on the rule: the eight-seat loss system
                by_live = [sum(share for (a, b), share in chain.items() if a + b == live) for live in range(9)]
                for mine, theirs in zip(by_live, place_model.truncated_poisson(rho)):
                    self.assertAlmostEqual(mine, theirs, places=9)

    def test_the_mixture_reproduces_the_production_decode_time_share(self):
        weights = place_model.fit_mixture()
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        for mine, theirs in zip(place_model.conditional_share(weights), place_model.OBSERVED_SHARE):
            self.assertLess(abs(mine - theirs), 0.01)

    def test_the_numbers_keep_the_order_the_document_states(self):
        weights = place_model.fit_mixture()
        today, fuller = place_model.mixed_states(place_model.lone_first, weights), place_model.mixed_states(place_model.fullest_first, weights)
        for name in place_model.WALL_ORDER:
            walls = place_model.WALLS[name]
            with self.subTest(walls=name):
                base = place_model.throughput(today, walls)
                gain = place_model.throughput(fuller, walls) / base - 1.0
                bound = place_model.ideal_throughput(weights, walls) / base - 1.0
                octo = place_model.throughput(today, walls, True) / base - 1.0
                self.assertTrue(0.008 < gain < 0.02, gain)                 # the placement alone: about one percent
                self.assertTrue(0.06 < bound < 0.11, bound)                # a free migration: seven to nine
                self.assertTrue(0.04 < octo < 0.08, octo)                  # the octo pass for a split round: five to seven, no migration
                self.assertGreater(bound, gain * 4)
                self.assertGreater(place_model.throughput(fuller, walls, True), place_model.throughput(today, walls, True))
                rounds_today = place_model.two_pass_shares(today, walls)[1]
                rounds_fuller = place_model.two_pass_shares(fuller, walls)[1]
                for live in (2, 3, 4):
                    self.assertLess(rounds_fuller[live], rounds_today[live])
                self.assertAlmostEqual(rounds_today[5], 1.0, places=9)
                self.assertAlmostEqual(rounds_fuller[5], 1.0, places=9)

    def test_no_sampled_stationary_policy_beats_fullest_first(self):
        # the exhaustive search (place_model.py --exhaustive) finds fullest_first the best of all 4,096; a seeded sample of 150 here
        rng = random.Random(7)
        walls = place_model.WALLS['64k']
        best = place_model.throughput(place_model.stationary(place_model.fullest_first, 2.0), walls)
        choice_states = [(a, b) for a in range(4) for b in range(4) if a != b]
        for _ in range(150):
            table = dict((state, rng.randrange(2)) for state in choice_states)

            def rule(a, b, table=table):
                if a == 4 and b == 4:
                    return None
                if a == 4:
                    return 1
                if b == 4:
                    return 0
                return 0 if a == b else table[(a, b)]

            self.assertLessEqual(place_model.throughput(place_model.stationary(rule, 2.0), walls), best + 1e-12)

    def test_the_event_simulation_agrees_with_the_chain_and_the_sign_survives_other_turns(self):
        walls = place_model.WALLS['64k']
        for rule in (place_model.lone_first, place_model.fullest_first):
            chain, events = place_model.stationary(rule, 2.0), place_model.simulate(rule, 2.0, horizon=60000.0)
            self.assertAlmostEqual(sum(events.values()), 1.0, places=9)
            for state in place_model.STATES:
                self.assertLess(abs(chain[state] - events[state]), 0.01, (rule.__name__, state))
        for sigma in (None, 1.0, 1.5):
            for burst in (1, 2):
                with self.subTest(sigma=sigma, burst=burst):
                    today = place_model.simulate(place_model.lone_first, 2.0, sigma, burst)
                    fuller = place_model.simulate(place_model.fullest_first, 2.0, sigma, burst)
                    gain = place_model.throughput(fuller, walls) / place_model.throughput(today, walls) - 1.0
                    self.assertTrue(0.003 < gain < 0.025, gain)
        self.assertIn('log-normal sigma 1.5', '\n'.join(place_model.sensitivity_lines()))

    def test_the_report_prints(self):
        text = place_model.tables()
        for needle in ('fullest_first (QWEN_FAST_TP4_PLACE_COMPACT=1)', 'ideal_compaction', 'octo_on_split', 'production (measured)'):
            self.assertIn(needle, text)


if __name__ == '__main__':
    unittest.main()
