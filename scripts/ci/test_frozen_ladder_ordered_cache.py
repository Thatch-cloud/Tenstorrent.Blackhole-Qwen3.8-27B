import unittest

from frozen_context_geometry import CONTEXTS, geometry
from frozen_ladder_ordered_cache import page_geometry, validate_shapes
import ordered_cache


class LadderOrderedCacheTests(unittest.TestCase):
    def test_each_context_uses_its_allocated_page_count(self):
        for context in CONTEXTS:
            specification = geometry(context)
            count = specification['target_page_count']
            for rows in (1, 2, 4, 8, 16, 32):
                args = ((count + 8, 2, 64, 256), (1, rows, 32, 256), (rows,), (rows, count))
                self.assertEqual(validate_shapes(*args, context=context), rows)
                with self.assertRaises(ValueError):
                    validate_shapes(*args[:3], (rows, count + 1), context=context)

    def test_default_guard_admits_only_the_reviewed_width_and_scope_restores_on_failure(self):
        """ordered_cache's default guard admits page width 2,052 (a 131,328 window) since the
        131k work - reviewed against the kernels and qualified on hardware by the independent
        writer probe (docs/four-streams-131k-feasibility-2026-09-23.md). It still refuses its
        neighbours, and this scope must still hand back the guard it replaced, even when the
        body raises."""
        original = ordered_cache.validate_shapes
        args = ((2060, 2, 64, 256), (1, 16, 32, 256), (16,), (16, 2052))
        self.assertEqual(original(*args), 16)
        with self.assertRaises(ValueError):
            original((2060, 2, 64, 256), (1, 16, 32, 256), (16,), (16, 2056))
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with page_geometry(131072) as evidence:
                self.assertEqual(ordered_cache.validate_shapes(*args), 16)
                raise RuntimeError('fixture')
        self.assertIs(ordered_cache.validate_shapes, original)
        self.assertTrue(evidence['restored'])
        self.assertEqual(evidence['calls'], 1)

    def test_malformed_or_undersized_storage_rejected(self):
        args = ((2060, 2, 64, 256), (1, 16, 32, 256), (16,), (16, 2052))
        for index, changed in ((0, (1024, 2, 64, 256)), (1, (1, 3, 32, 256)),
                (2, (8,)), (3, (8, 2052)), (3, (16, True))):
            values = list(args)
            values[index] = changed
            with self.assertRaises(ValueError):
                validate_shapes(*values, context=131072)


if __name__ == '__main__':
    unittest.main()
