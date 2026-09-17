import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import dspark_8k_build as candidate
from dspark_runtime_cache import cache_key


class BuildTests(unittest.TestCase):
    def test_disabled_path_never_reads_or_changes_factory(self):
        with patch.object(candidate, 'qualify') as qualify:
            self.assertIsNone(candidate.prepare('missing', 'missing', enabled=False))
            qualify.assert_not_called()
        with self.assertRaises(ValueError):
            candidate.prepare('missing', 'missing', enabled=1)

    def test_qualified_transform_and_completed_binary_evidence(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / candidate.SOURCE
            path.parent.mkdir(parents=True)
            path.write_bytes(b'original')
            with patch.object(candidate, 'qualify', return_value={'report_sha256': 'retained'}), \
                    patch.object(candidate, 'transform', return_value=b'changed') as transform:
                inputs = candidate.prepare(directory, directory, enabled=True)
                transform.assert_called_once_with(b'original')
            self.assertEqual(path.read_bytes(), b'changed')
            checksum = hashlib.sha256(b'binary').hexdigest()
            for name in ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'):
                binary = Path(directory) / name
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(b'binary')
            result = candidate.completed(directory, inputs, checksum, import_passed=True)
            self.assertTrue(result['passed'])
            with self.assertRaises(ValueError):
                candidate.completed(directory, inputs, checksum, import_passed=False)
            binary.write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                candidate.completed(directory, inputs, checksum, import_passed=True)

    def test_factory_provenance_changes_cache_key(self):
        base = dict(image='same', builders={'builder': 'same'})
        previous = cache_key(base)
        changed = cache_key(dict(base, draft_8k_factory={'factory_sha256': 'qualified'}))
        self.assertNotEqual(previous, changed)
        self.assertNotEqual(changed, cache_key(dict(base, draft_8k_factory={'factory_sha256': 'different'})))


if __name__ == '__main__':
    unittest.main()
