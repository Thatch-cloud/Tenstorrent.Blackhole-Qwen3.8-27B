import unittest

import native_draft_sdpa
from dspark_ladder_scalar_reciprocal import scalar_reciprocal, BEFORE, BODY
from frozen_reciprocal_isolation import isolate, isolated_reciprocal


class ReciprocalIsolationTests(unittest.TestCase):
    def test_only_preprocessor_boundary_changes(self):
        original = native_draft_sdpa.replacements
        with scalar_reciprocal():
            before = native_draft_sdpa.replacements()
            with isolated_reciprocal():
                after = native_draft_sdpa.replacements()
            self.assertEqual(native_draft_sdpa.replacements(), before)
        self.assertIs(native_draft_sdpa.replacements, original)
        self.assertEqual(before['sdpa.cpp'], after['sdpa.cpp'])
        changed = [(old, new) for old, new in zip(before['compute_common.hpp'], after['compute_common.hpp']) if old != new]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0][0], (BEFORE, BEFORE + BODY))
        self.assertEqual(changed[0][1], (BEFORE, BEFORE + '\n#if defined(QWEN_DRAFT_EXP_APPROX)\n' + BODY + '#endif\n'))

    def test_missing_scalar_candidate_rejected(self):
        with self.assertRaises(ValueError):
            isolate(native_draft_sdpa.replacements())


if __name__ == '__main__':
    unittest.main()
