from dataclasses import replace
import random
import unittest

from history_append_plan import BankAppendPlanner


class AppendPlanTests(unittest.TestCase):
    def test_commits_discards_and_tile_boundaries_match_complete_bank(self):
        for initial in (0, 31, 32, 4093, 65535, 65536):
            capacity = (initial + 4096 + 31) // 32 * 32
            planner = BankAppendPlanner(initial, capacity)
            active = list(range(1, initial + 1)) + [0] * (capacity - initial)
            spare = active.copy()
            randomizer = random.Random(initial)
            for ordinal in range(80):
                prefix = randomizer.randint(1, 32)
                committed = randomizer.choice((True, False))
                plan = planner.prepare(prefix)
                self.assertLessEqual(plan.end_row - plan.first_row, 96)
                before = active.copy()
                delta = [100000 + ordinal * 32 + row for row in range(prefix)]
                expected = active[:planner.position] + delta + [0] * (capacity - planner.position - prefix)
                for row in range(plan.first_row, plan.end_row):
                    spare[row] = (active[row] if row < plan.position else
                        delta[row - plan.position] if row < plan.position + prefix else 0)
                self.assertEqual(spare, expected)
                self.assertEqual(active, before)
                planner.resolve(plan, commit=committed)
                if committed:
                    active, spare = spare, active
                self.assertEqual(active, expected if committed else before)
                with self.assertRaises(ValueError):
                    planner.resolve(plan, commit=True)

    def test_failed_or_foreign_transaction_cannot_advance_frontier(self):
        planner = BankAppendPlanner(65536, 66560)
        plan = planner.prepare(16)
        with self.assertRaises(ValueError):
            planner.prepare(1)
        with self.assertRaises(ValueError):
            planner.resolve(replace(plan), commit=True)
        self.assertEqual(planner.position, 65536)
        planner.resolve(plan, commit=False)
        self.assertEqual(planner.position, 65536)

    def test_capacity_and_prefix_guards(self):
        with self.assertRaises(ValueError):
            BankAppendPlanner(32, 33)
        planner = BankAppendPlanner(31, 32)
        for prefix in (0, 2, True, 33):
            with self.assertRaises(ValueError):
                planner.prepare(prefix)
        plan = planner.prepare(1)
        planner.resolve(plan, commit=True)
        self.assertEqual(planner.position, 32)


if __name__ == '__main__':
    unittest.main()
