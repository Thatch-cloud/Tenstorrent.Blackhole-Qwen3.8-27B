import unittest

from matched_context_geometry import CONTEXTS
from matched_target_geometry import geometry, validate_ticket
from target_t16_64k_geometry import geometry as previous_geometry


class MatchedTargetGeometryTests(unittest.TestCase):
    def test_preserves_qualified_64k_component_geometry(self):
        self.assertEqual(geometry(65536), previous_geometry(hardware=True))

    def test_all_contexts_cover_frontier_and_last_complete_block(self):
        for context in CONTEXTS:
            plan = geometry(context)
            for start in plan['starts']:
                validate_ticket(context, start, 16, plan['capacity'])
            self.assertEqual(plan['starts'][2] + 16, plan['capacity'])

    def test_rejects_truncation_overflow_and_short_context(self):
        for context in CONTEXTS:
            for start, rows, capacity, short in (
                    (context - 1, 16, context + 256, False),
                    (context + 241, 16, context + 256, False),
                    (context, 17, context + 256, False),
                    (context, 16, context + 128, False),
                    (context, 16, context + 256, True)):
                with self.assertRaises(ValueError):
                    validate_ticket(context, start, rows, capacity, short_context=short)


if __name__ == '__main__':
    unittest.main()
