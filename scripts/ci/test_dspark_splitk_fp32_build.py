import json
from pathlib import Path
import tempfile
import unittest

from dspark_hardware_gate import digest
from dspark_splitk_fp32_build import BUILDERS, validate
from dspark_splitk_fp32_factory import SOURCE


class FactoryAdmissionTests(unittest.TestCase):
    def test_rejects_mutated_source_and_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / SOURCE
            source.parent.mkdir(parents=True)
            source.write_text('candidate source')
            names = ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True)
                path.write_bytes(b'candidate binary')
            report = dict(passed=True, source_after=digest(source),
                builders={name: digest(Path(__file__).parent / name) for name in BUILDERS},
                binaries={name: digest(root / name) for name in names})
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(report))
            self.assertTrue(validate(root, manifest)['passed'])
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'factory rebuild'):
                validate(root, manifest)
            source.write_text('candidate source')
            (root / names[0]).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'binary changed'):
                validate(root, manifest)


if __name__ == '__main__':
    unittest.main()
