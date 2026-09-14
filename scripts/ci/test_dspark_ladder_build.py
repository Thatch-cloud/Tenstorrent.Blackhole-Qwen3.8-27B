import os
from pathlib import Path
import unittest
from unittest.mock import patch

import dspark_fp32_build as baseline
from dspark_ladder_build import factory_scope, main, validate_manifest
from dspark_ladder_factory import geometry_predicate


class LadderBuildTests(unittest.TestCase):
    def test_scope_restores_original_builder_and_rejects_disabled_variant(self):
        original = baseline.transform, baseline.REPLACEMENT
        with factory_scope():
            self.assertIn(geometry_predicate('Skt', 'Sk_chunk_t'), baseline.REPLACEMENT)
            with self.assertRaises(ValueError):
                baseline.transform(b'unused', enabled=False)
        self.assertEqual((baseline.transform, baseline.REPLACEMENT), original)

    def test_wrong_case_rejected_before_build(self):
        with patch.dict(os.environ, {'QWEN_SIM_CASE': 'unrelated'}), patch.object(baseline, 'main') as build:
            with self.assertRaises(ValueError):
                main()
            build.assert_not_called()

    def test_old_build_manifest_cannot_qualify_ladder(self):
        with patch('dspark_ladder_build.BASELINE_VALIDATE', return_value={'passed': True}):
            with self.assertRaisesRegex(ValueError, 'provenance'):
                validate_manifest('/unused', '/unused')

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
    def test_transformed_source_matches_validator_replacement(self):
        original = (Path(os.environ['TT_NATIVE_TEST_ROOT']) / baseline.SOURCE).read_bytes()
        with factory_scope():
            candidate = baseline.transform(original, enabled=True)
            self.assertEqual(candidate.count(baseline.REPLACEMENT.encode()), 1)
            self.assertEqual(baseline.restore_factory_source(candidate, baseline.REPLACEMENT), original)
