from dataclasses import replace
import unittest

from history_append_dma import validate_plan
from history_append_plan import BankAppendPlanner


class AppendDmaTests(unittest.TestCase):
    def test_valid_boundary_plan_and_malformed_ranges(self):
        plan = BankAppendPlanner(65535, 66560).prepare(32)
        validate_plan(plan, 66560)
        for changed in (replace(plan, first_row=1), replace(plan, end_row=65536),
                replace(plan, prefix=33), replace(plan, first_row=0), replace(plan, prefix=True)):
            with self.assertRaises(ValueError):
                validate_plan(changed, 66560)


if __name__ == '__main__':
    unittest.main()
