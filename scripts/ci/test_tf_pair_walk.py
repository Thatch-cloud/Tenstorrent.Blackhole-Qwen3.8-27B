"""tf_pair_walk on fake drafters: oracle -> full rounds, null -> 1, off-by-one, the 256-edge caps, the forced schedule,
the look-ahead guard. No model, no text: token ids are small integers."""
import os
import random
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tf_pair_walk as walk  # noqa: E402
from extent_attention_replay import accept_limit  # noqa: E402


def make_turn(prompt_len=300, answer_len=200, seed=1, think=50, tool_at=None):
    rng = random.Random(seed)
    return dict(prompt_ids=[rng.randrange(1, 1000) for _ in range(prompt_len)],
                output_ids=[rng.randrange(1, 1000) for _ in range(answer_len)], think_tokens=think, tool_at=tool_at)


class Oracle(object):
    """Proposes the logged answer's next tokens (the best possible drafter)."""
    def __init__(self, turn):
        self.prompt, self.answer = len(turn['prompt_ids']), turn['output_ids']

    def propose(self, sequence, start, count):
        index = start - self.prompt
        return [self.answer[index + 1 + at] if index + 1 + at < len(self.answer) else -1 for at in range(count)]


class Null(object):
    def propose(self, sequence, start, count):
        return [-1] * count


class RightFor(Oracle):
    """Right for the first `good` proposals, wrong after."""
    def __init__(self, turn, good):
        Oracle.__init__(self, turn)
        self.good = good

    def propose(self, sequence, start, count):
        proposed = Oracle.propose(self, sequence, start, count)
        return [token if at < self.good else -2 for at, token in enumerate(proposed)]


class Peeker(object):
    """Reads the token after the anchor: the view must refuse."""
    def propose(self, sequence, start, count):
        return [sequence[start + 1]] * count


class RuleTests(unittest.TestCase):
    def test_the_restated_extent_rule_equals_the_serving_rule(self):
        import extent_attention_replay as serving
        for start in list(range(0, 3000)) + [4095, 4096, 65535, 65536, 122773, 262143]:
            self.assertEqual(walk.extent(start), serving.extent(start))
            for rows in (1, 2, 8, 15, 16):
                self.assertEqual(walk.accept_limit(start, rows), serving.accept_limit(start, rows))


class WalkTests(unittest.TestCase):
    def test_oracle_commits_a_full_block_where_the_extent_allows(self):
        turn = make_turn(prompt_len=300, answer_len=400)
        rounds = walk.walk_free(turn, Oracle(turn), 15)
        for entry in rounds:
            self.assertEqual(entry['accepted'], min(15, entry['available']))
            self.assertEqual(entry['committed'], min(16, entry['cap']))
            self.assertEqual(entry['uncapped'], min(16, entry['available'] + 1))
        self.assertTrue(any(entry['cap'] == 16 and entry['committed'] == 16 for entry in rounds))

    def test_null_drafter_commits_one_token_per_round(self):
        turn = make_turn()
        rounds = walk.walk_free(turn, Null(), 15)
        self.assertEqual(walk.tau(rounds), 1.0)
        # every token but the seed and the terminal round is a counted round
        self.assertEqual(len(rounds), len(turn['output_ids']) - 2)

    def test_off_by_one_drafter(self):
        turn = make_turn(answer_len=500)
        rounds = walk.walk_free(turn, RightFor(turn, 3), 15)
        for entry in rounds:
            self.assertEqual(entry['accepted'], min(3, entry['available']))
        self.assertGreater(walk.tau(rounds), 1.0)
        self.assertLess(walk.tau(rounds), 4.0 + 1e-9)

    def test_seed_and_terminal_round_are_not_counted(self):
        turn = make_turn(answer_len=40)
        rounds = walk.walk_free(turn, Oracle(turn), 15)
        self.assertEqual(rounds[0]['offset'], 1)                       # the first counted round starts after the seed
        self.assertTrue(all(not entry['terminal'] for entry in rounds))
        covered = sum(entry['committed'] for entry in rounds) + 1       # + the seed
        self.assertLess(covered, 40)                                    # the terminal round's tokens are left out

    def test_caps_at_every_256_edge(self):
        for start in (255, 256, 257, 4096 - 3, 4095, 4096):
            turn = make_turn(prompt_len=start, answer_len=60)
            rounds = walk.walk_free(turn, Oracle(turn), 15)
            first = rounds[0]
            self.assertEqual(first['start'], start)
            self.assertEqual(first['cap'], min(accept_limit(start, 16), 59))
            self.assertEqual(first['committed'], first['cap'])
            self.assertEqual(first['uncapped'], 16)

    def test_cap_one_at_the_last_row_of_an_extent(self):
        turn = make_turn(prompt_len=255, answer_len=30)
        first = walk.walk_free(turn, Oracle(turn), 15)[0]
        self.assertEqual((first['cap'], first['committed']), (1, 1))

    def test_t8_uses_eight_rows(self):
        turn = make_turn(prompt_len=300, answer_len=100)
        rounds = walk.walk_free(turn, Oracle(turn), 7)
        self.assertTrue(all(entry['rows'] == 8 for entry in rounds))
        self.assertEqual(max(entry['committed'] for entry in rounds), 8)

    def test_look_ahead_is_refused(self):
        turn = make_turn()
        with self.assertRaises(IndexError):
            walk.walk_free(turn, Peeker(), 15)

    def test_wrong_proposal_count_refused(self):
        class Short(object):
            def propose(self, sequence, start, count):
                return [1] * (count - 1)
        with self.assertRaises(ValueError):
            walk.walk_free(make_turn(), Short(), 15)

    def test_regions_follow_the_offset(self):
        turn = make_turn(answer_len=300, think=100, tool_at=250)
        rounds = walk.walk_free(turn, Oracle(turn), 15)
        for entry in rounds:
            expected = 'tool' if entry['offset'] >= 250 else ('reasoning' if entry['offset'] < 100 else 'content')
            self.assertEqual(entry['region'], expected)
        self.assertEqual(set(entry['region'] for entry in rounds), set(['reasoning', 'content', 'tool']))

    def test_forced_schedule_reproduces_the_served_counts(self):
        turn = make_turn(answer_len=300)
        free = walk.walk_free(turn, RightFor(turn, 5), 15)
        # the served schedule IS the free walk of the same drafter plus its terminal round
        logged = [entry['committed'] for entry in free]
        logged.append(299 - sum(logged))
        forced = walk.walk_forced(turn, RightFor(turn, 5), 15, logged)
        self.assertEqual(len(forced), len(free))
        self.assertTrue(all(entry['exact'] for entry in forced))
        exact, count, committed, served = walk.schedule_agreement(forced)
        self.assertEqual((exact, committed), (count, served))

    def test_forced_schedule_flags_a_disagreeing_drafter(self):
        turn = make_turn(answer_len=300)
        free = walk.walk_free(turn, Oracle(turn), 15)
        logged = [entry['committed'] for entry in free]
        logged.append(299 - sum(logged))
        forced = walk.walk_forced(turn, Null(), 15, logged)
        self.assertTrue(any(not entry['exact'] for entry in forced))
        self.assertTrue(all(entry['committed'] == 1 for entry in forced))

    def test_forced_schedule_that_misses_the_end_is_refused(self):
        turn = make_turn(answer_len=50)
        with self.assertRaises(ValueError):
            walk.walk_forced(turn, Null(), 15, [1, 1, 1])
        with self.assertRaises(ValueError):
            walk.walk_forced(turn, Null(), 15, [1] * 60)

    def test_position_acceptance_is_a_prefix_survival_curve(self):
        turn = make_turn(answer_len=600)
        rounds = walk.walk_free(turn, RightFor(turn, 4), 15)
        curve = walk.position_acceptance(rounds, 15)
        for position, (accepted, available) in enumerate(curve, 1):
            if position <= 4:
                self.assertEqual(accepted, available)
            else:
                self.assertEqual(accepted, 0)
        self.assertTrue(all(curve[at][1] >= curve[at + 1][1] for at in range(14)))

    def test_records_hold_counts_only(self):
        turn = make_turn()
        for entry in walk.walk_free(turn, Oracle(turn), 15):
            for value in entry.values():
                self.assertIsInstance(value, (int, bool, str))

    def test_uncapped_never_below_served(self):
        turn = make_turn(prompt_len=250, answer_len=500)
        for entry in walk.walk_free(turn, RightFor(turn, 9), 15):
            self.assertGreaterEqual(entry['uncapped'], entry['committed'])


if __name__ == '__main__':
    unittest.main()
