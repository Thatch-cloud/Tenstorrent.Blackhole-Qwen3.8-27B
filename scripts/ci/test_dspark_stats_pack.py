"""Scoped correction-path transformation checks; no device qualification."""

import unittest

import native_draft_sdpa
from dspark_stats_pack import BEFORE, AFTER, SELECTOR_ASSERT, scoped_stats_pack


class StatsPackTests(unittest.TestCase):
    def test_scope_preserves_original_and_restores_on_failure(self):
        original = native_draft_sdpa.replacements
        baseline = original()
        with self.assertRaisesRegex(RuntimeError, 'scope exit'):
            with scoped_stats_pack():
                changed = native_draft_sdpa.replacements()
                self.assertEqual(changed['sdpa.cpp'][0][0], baseline['sdpa.cpp'][0][0])
                self.assertEqual(changed['sdpa.cpp'][0][1], baseline['sdpa.cpp'][0][1].replace(
                    '#include "compute_common.hpp"', SELECTOR_ASSERT + '#include "compute_common.hpp"'))
                self.assertEqual(changed['compute_common.hpp'],
                    baseline['compute_common.hpp'] + ((BEFORE, AFTER),))
                self.assertNotIn('recip_tile_first_column<false>', str(changed))
                raise RuntimeError('scope exit')
        self.assertIs(native_draft_sdpa.replacements, original)
        self.assertEqual(original(), baseline)


if __name__ == '__main__':
    unittest.main()
