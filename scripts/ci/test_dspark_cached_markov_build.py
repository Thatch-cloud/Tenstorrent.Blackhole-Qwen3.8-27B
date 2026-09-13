import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dspark_cached_markov_build import SOURCE, admitted_reference, completed, prepare


class CachedMarkovBuildTests(unittest.TestCase):
    def test_reference_requires_original_source_and_matching_admission(self):
        from markov_sparse_fp32 import SOURCE_SHA256
        with self.assertRaisesRegex(ValueError, 'Original pinned'):
            admitted_reference('/missing', '/missing', {}, {})
        with patch('dspark_cached_markov_build.qualify', return_value={'accepted': True}):
            with self.assertRaisesRegex(ValueError, 'Accepted simulator'):
                admitted_reference('/missing', '/missing', {SOURCE: SOURCE_SHA256},
                    {'factory_inputs': {'admission': {}, 'source_before': SOURCE_SHA256}})

    def test_disabled_does_not_require_sources(self):
        self.assertIsNone(prepare('/missing', '/missing', enabled=False))
        with self.assertRaises(ValueError):
            prepare('/missing', '/missing', enabled='1')

    def test_completion_checks_both_binary_paths_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / SOURCE
            source.parent.mkdir(parents=True)
            source.write_bytes(b'factory')
            inputs = dict(factory_sha256=hashlib.sha256(b'factory').hexdigest())
            checksum = hashlib.sha256(b'binary').hexdigest()
            for name in ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'):
                binary = root / name
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b'binary')
            self.assertTrue(completed(root, inputs, checksum, import_passed=True)['passed'])
            with self.assertRaises(ValueError):
                completed(root, inputs, checksum, import_passed=False)
            binary.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'binary'):
                completed(root, inputs, checksum, import_passed=True)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'factory'):
                completed(root, inputs, checksum, import_passed=True)
