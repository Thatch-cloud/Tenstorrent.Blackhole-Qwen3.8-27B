import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import dspark_splitk_combined_build as candidate


class CombinedBuildTests(unittest.TestCase):
    def test_prefill_and_decode_share_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / candidate.SOURCE
            source.parent.mkdir(parents=True)
            source.write_bytes(b'native')
            evidence = dict(factory_source_before=hashlib.sha256(b'native').hexdigest(),
                factory_source_after=hashlib.sha256(b'combined').hexdigest(),
                report_sha256='hardware', simulator_report_sha256='simulator')
            prefill = Mock(return_value=dict(capacity=66560, factory_sha256='prefill'))
            with patch.object(candidate, 'admission', return_value=evidence), \
                    patch.object(candidate, 'transform', return_value='combined'), \
                    patch.object(candidate, 'BUILDERS', ()):
                result = candidate.prepare(directory, directory, 'report', prepare_prefill=prefill)
                self.assertEqual(source.read_bytes(), b'combined')
                self.assertEqual(result['factory_sha256'], 'prefill')
                self.assertEqual(result['splitk_factory']['source_after'], evidence['factory_source_after'])
                prefill.assert_called_once_with(directory, directory, 'report')
                build = dict(factory_inputs=result)
                with patch.object(candidate.dspark_64k_build, 'validate_build', return_value=build):
                    validated = candidate.validate_combined(directory, directory, 'build')
                    self.assertFalse(validated['full_request_qualified'])
                    source.write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, 'same combined build'):
                        candidate.validate_combined(directory, directory, 'build')
                with self.assertRaisesRegex(ValueError, 'Pristine'):
                    candidate.prepare(directory, directory, 'report', prepare_prefill=prefill)
                self.assertEqual(prefill.call_count, 1)

    def test_no_implicit_serving_selection(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Explicit allocated'):
                candidate.require_selected()


if __name__ == '__main__':
    unittest.main()
