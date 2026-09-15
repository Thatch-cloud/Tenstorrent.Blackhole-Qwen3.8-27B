import unittest
from unittest.mock import patch

import matched_combined_build as candidate


class MatchedCombinedBuildTests(unittest.TestCase):
    def test_components_do_not_grant_timing_or_request_admission(self):
        with patch.object(candidate, 'qualify_simulator', return_value=dict(
                factory_source_before='before', factory_source_after='after', report_sha256='sim')), \
                patch.object(candidate, 'qualify_draft', return_value=dict(
                    factory_sha256='after', report_sha256='draft', kernel={'source_active': 'kernel'})), \
                patch.object(candidate, 'qualify_target', return_value={'component_qualified': True}):
            result = candidate.admission('fixture')
        self.assertTrue(result['component_qualified'])
        for field in ('full_request_qualified', 'performance_qualified', 'serving_qualified'):
            self.assertFalse(result[field])

    def test_mixed_factories_rejected(self):
        with patch.object(candidate, 'qualify_simulator', return_value={'factory_source_after': 'first'}), \
                patch.object(candidate, 'qualify_draft', return_value={'factory_sha256': 'second'}), \
                patch.object(candidate, 'qualify_target', return_value={}):
            with self.assertRaises(ValueError):
                candidate.admission('fixture')

    def test_scope_restores_baseline(self):
        before = candidate.baseline.admission
        with patch.dict(candidate.os.environ, QWEN_MATCHED_COMBINED='1'):
            with candidate.identity_scope():
                self.assertIs(candidate.baseline.admission, candidate.admission)
        self.assertIs(candidate.baseline.admission, before)


if __name__ == '__main__':
    unittest.main()
