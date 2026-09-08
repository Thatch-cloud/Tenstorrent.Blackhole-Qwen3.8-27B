import unittest

from attention_request_plan import capture_plan
from attention_mask_replay import validate_ticket


class RequestPlanTests(unittest.TestCase):
    def test_short_coding_budget_has_three_families_and_native_boundary_fallback(self):
        plan = capture_plan(170, 65536, 32, 512, max_verify_rows=8, short_context=True)
        self.assertEqual([capture.key for capture in plan.captures],
            [(1, None), (2, None), (4, None), (8, 256), (8, 512), (8, 768)])
        self.assertEqual(plan.max_rows(254, 428), 4)
        for position in range(170, 682):
            remaining = 682 - position
            maximum = plan.max_rows(position, remaining)
            self.assertLessEqual(maximum, 8)
            for rows in (1, 2, 4, 8):
                if rows > maximum:
                    continue
                capture = plan.select(position, rows, remaining)
                if capture.capacity is not None:
                    validate_ticket(position, rows, capture.capacity, short_context=True)
        for position, budget, limit in ((127, 32, 8), (170, 512, 32), (170, 599, 8)):
            with self.assertRaises(ValueError):
                capture_plan(position, 65536, 32, budget, max_verify_rows=limit, short_context=True)

    def test_cross_boundary_request_prepares_both_families_up_front(self):
        plan = capture_plan(4078, 65536, 32, 128)
        self.assertEqual([capture.key for capture in plan.captures],
            [(1, None), (2, None), (4, None), (8, 4096), (16, 4096), (8, 4352), (16, 4352), (32, 4352)])
        self.assertEqual(plan.max_rows(4078, 128), 16)
        self.assertEqual(plan.max_rows(4095, 111), 4)
        self.assertIsNone(plan.select(4095, 4, 111).capacity)
        self.assertEqual(plan.select(4096, 32, 110).position, 4096)
        with self.assertRaises(ValueError):
            plan.select(4095, 8, 111)

    def test_every_possible_ticket_position_has_a_safe_prepared_route(self):
        for initial in (3840, 4063, 4078, 4095, 4096, 16363, 16383, 16384):
            for budget in (1, 2, 4, 7, 8, 17, 32, 128):
                plan = capture_plan(initial, 65536, 32, budget)
                for position in range(initial, initial + budget):
                    remaining = initial + budget - position
                    maximum = plan.max_rows(position, remaining)
                    for rows in (1, 2, 4, 8, 16, 32):
                        if rows <= maximum:
                            capture = plan.select(position, rows, remaining)
                            if capture.capacity is not None:
                                validate_ticket(position, rows, capture.capacity)
                            self.assertLessEqual(position + rows, plan.stop)

    def test_rejects_changed_budget_or_unsupported_ticket_width(self):
        plan = capture_plan(4096, 65536, 32, 128)
        for position, rows, remaining in ((4095, 1, 128), (4224, 1, 1), (4096, 3, 128),
                                          (4096, True, 128), (4096, 1, 129)):
            with self.assertRaises(ValueError):
                plan.select(position, rows, remaining)

    def test_unqualified_long_context_does_not_silently_enable_masks(self):
        with self.assertRaises(ValueError):
            capture_plan(16640, 65536, 32, 128)
