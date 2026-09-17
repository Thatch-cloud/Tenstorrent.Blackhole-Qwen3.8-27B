import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from dspark_64k_build import completed, prepare, validate_build, verify_factory
from dspark_fp32_intermediates import SOURCE


@unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT') and os.environ.get('QWEN_64K_REPORT'),
    'Pinned native source and retained hardware report required')
class BuildTests(unittest.TestCase):
    def test_real_factory_transform_and_fixture_binary_tamper_rejection(self):
        scripts = Path(__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / SOURCE
            source.parent.mkdir(parents=True)
            source.write_bytes((Path(os.environ['TT_NATIVE_TEST_ROOT']) / SOURCE).read_bytes())
            inputs = prepare(root, scripts, os.environ['QWEN_64K_REPORT'])
            self.assertEqual(verify_factory(root), inputs['factory_sha256'])
            checksum = hashlib.sha256(b'unit-test binary fixture, not hardware evidence').hexdigest()
            for name in ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'unit-test binary fixture, not hardware evidence')
            evidence = completed(root, inputs, checksum, import_passed=True)
            report = root / 'test-build.json'
            report.write_text(json.dumps(evidence))
            self.assertEqual(validate_build(root, report, scripts), evidence)
            (root / 'build_Release/lib/_ttnncpp.so').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'binary differs'):
                validate_build(root, report, scripts)
