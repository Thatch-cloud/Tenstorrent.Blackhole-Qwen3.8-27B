import unittest

import dspark_ladder_normalization
from dspark_normalization_direct_stage import END, normalization_stage_scope, transform


class NormalizationStageTests(unittest.TestCase):
    def test_preserves_normalization_math_and_buffer_lifetime(self):
        original = dspark_ladder_normalization.HELPER
        changed = transform(original)
        self.assertEqual(changed[changed.index(END):], original[original.index(END):])
        self.assertNotIn('index < 1024', changed)
        self.assertIn('qwen_stage_score_tile(reciprocal_cb, scratch_cb, false);', changed)
        with normalization_stage_scope():
            self.assertEqual(dspark_ladder_normalization.HELPER, changed)
        self.assertEqual(dspark_ladder_normalization.HELPER, original)

    def test_rejects_changed_or_already_transformed_input(self):
        for source in ('', transform(dspark_ladder_normalization.HELPER)):
            with self.assertRaises(ValueError):
                transform(source)


if __name__ == '__main__':
    unittest.main()
