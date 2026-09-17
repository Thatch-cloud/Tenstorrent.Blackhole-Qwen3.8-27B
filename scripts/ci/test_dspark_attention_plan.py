"""T16 family-routing prerequisite; does not authorize device execution."""

import unittest

from attention_request_plan import capture_plan
from attention_mask_replay import validate_ticket


class DSparkAttentionPlanTests(unittest.TestCase):
    def test_every_t16_request_position_has_a_safe_route(self):
        for start in (4095, 4096, 4337, 4351, 4352):
            budget = 257
            plan = capture_plan(start, 65536, 32, budget, max_verify_rows=16)
            self.assertTrue(all(capture.rows <= 16 for capture in plan.captures))
            for position in range(start, start + budget):
                remaining = start + budget - position
                maximum = plan.max_rows(position, remaining)
                self.assertIn(maximum, (1, 2, 4, 8, 16))
                for rows in (1, 2, 4, 8, 16):
                    if rows > maximum:
                        continue
                    capture = plan.select(position, rows, remaining)
                    self.assertLessEqual(position + rows, plan.stop)
                    if rows < 8:
                        self.assertIsNone(capture.capacity)
                    else:
                        validate_ticket(position, rows, capture.capacity)

    def test_boundary_uses_native_small_width_then_next_family(self):
        plan = capture_plan(4096, 65536, 32, 257, max_verify_rows=16)
        self.assertEqual(plan.max_rows(4351, 2), 2)
        self.assertIsNone(plan.select(4351, 2, 2).capacity)
        with self.assertRaises(ValueError):
            plan.select(4351, 16, 2)
        self.assertEqual(plan.select(4336, 16, 17).capacity, 4352)


if __name__ == '__main__':
    unittest.main()
