import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import dspark_8k_admission as candidate


class AdmissionTests(unittest.TestCase):
    def test_context_and_headroom_are_exact(self):
        candidate.validate_request(8192, 256)
        for context, output in ((4096, 256), (8192, 128), (8192, 257), (True, 256), (8192, True)):
            with self.assertRaises(ValueError):
                candidate.validate_request(context, output)

    def test_admission_restores_after_exception_and_rejects_nesting(self):
        with TemporaryDirectory() as directory:
            binaries = {}
            for name in ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'):
                path = Path(directory) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture binary')
                binaries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            build = dict(factory_sha256='fixture', import_passed=True, passed=True, binaries=binaries)
            options = dict(context=8192, output_tokens=256, factory_root=directory, build_evidence=build)
            self.assertEqual(candidate.history_limit(), 8192)
            with patch.object(candidate, 'qualify', return_value={'component_only': True}), \
                    patch.object(candidate, 'verify_factory', return_value='fixture'):
                with self.assertRaisesRegex(RuntimeError, 'request failed'):
                    with candidate.admitted_request(directory, **options):
                        self.assertEqual(candidate.history_limit(), 8448)
                        with self.assertRaises(ValueError):
                            with candidate.admitted_request(directory, **options):
                                self.fail('Nested admission entered')
                        raise RuntimeError('request failed')
                self.assertEqual(candidate.history_limit(), 8192)
                build['passed'] = False
                with self.assertRaises(ValueError):
                    with candidate.admitted_request(directory, **options):
                        self.fail('Failed build entered admission')
                self.assertEqual(candidate.history_limit(), 8192)

    def test_unmodified_factory_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / candidate.SOURCE
            path.parent.mkdir(parents=True)
            path.write_bytes(candidate.ANCHOR.encode())
            with self.assertRaises(ValueError):
                candidate.verify_factory(directory)


if __name__ == '__main__':
    unittest.main()
