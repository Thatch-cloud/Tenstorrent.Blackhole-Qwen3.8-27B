import unittest

from dspark_ladder_geometry import geometry as previous_geometry
from matched_context_geometry import CONTEXTS, geometry, ladder, require_reference


class MatchedContextTests(unittest.TestCase):
    def test_preserves_actual_64k_geometry(self):
        reference = previous_geometry(65536)
        planned = require_reference(reference)
        self.assertEqual(planned['capacity'], 66560)
        self.assertEqual(planned['storage_keys'], 66624)
        self.assertEqual(planned['native_keys'], 67584)
        self.assertEqual(planned['key_chunks_per_worker'], 33)
        reference['native_keys'] -= 256
        with self.assertRaises(ValueError):
            require_reference(reference)

    def test_whole_ladder_preserves_history_and_workload(self):
        rows = ladder()
        self.assertEqual(tuple(row['context'] for row in rows), CONTEXTS)
        for row in rows:
            with self.subTest(context=row['context']):
                self.assertEqual(row['target_page_count'] * 64, row['capacity'])
                self.assertGreaterEqual(row['storage_keys'], row['capacity'] + 15)
                self.assertEqual(row['native_keys'], row['storage_keys'] + row['extra_masked_keys'])
                self.assertEqual(row['native_keys'], row['key_chunks_per_worker'] * 8 * 256)
                self.assertEqual((row['output_budget'], row['streams'], row['verifier_rows']), (256, 1, 16))
                self.assertEqual(row['draft_history_bank_bytes_per_chip'], row['capacity'] * 20480)
                self.assertFalse(row['runtime_admitted'])
                self.assertFalse(row['hardware_fit_qualified'])
                self.assertFalse(row['performance_qualified'])

    def test_no_implicit_or_fractional_contexts(self):
        for context in (True, 65536.0, '65536', 0, 32767, 524288):
            with self.subTest(context=context), self.assertRaises(ValueError):
                geometry(context)


if __name__ == '__main__':
    unittest.main()
