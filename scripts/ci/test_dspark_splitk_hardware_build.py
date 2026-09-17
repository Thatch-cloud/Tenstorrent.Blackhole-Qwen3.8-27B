import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dspark_splitk_hardware_build as build
from dspark_hardware_gate import digest


class HardwareBuildTests(unittest.TestCase):
    def test_rejects_changed_source_or_loaded_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factory = root / build.SOURCE
            factory.parent.mkdir(parents=True)
            factory.write_bytes(b'qualified factory')
            transformer = root / 'ttnn/cpp/ttnn/operations/transformer'
            registration = transformer / 'sources.cmake'
            registration.write_bytes(b'registrations')
            implementations = {}
            for name in build.implementation_sources():
                source = transformer / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b'implementation')
                implementations[name] = digest(source)
            for name in build.BINARIES:
                binary = root / name
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(b'rebuilt library')
            admission = dict(report_sha256='retained', factory_source_after=digest(factory))
            report = dict(passed=True, import_passed=True, backend='hardware', image=build.IMAGE,
                builders={'builder': 'hash'}, simulator_report_sha256='retained',
                registration=dict(source_after=digest(registration), implementation_sources=implementations),
                source_after=digest(factory), binaries={name: digest(root / name) for name in build.BINARIES})
            output = root / 'build.json'
            output.write_text(json.dumps(report))
            with patch.object(build, 'qualify', return_value=admission), \
                    patch.object(build, 'fingerprints', return_value={'builder': 'hash'}):
                self.assertTrue(build.validate_build(root, directory, output, 'retained.json')['passed'])
                registration.write_bytes(b'changed registrations')
                with self.assertRaisesRegex(ValueError, 'registration changed'):
                    build.validate_build(root, directory, output, 'retained.json')
                registration.write_bytes(b'registrations')
                source.write_bytes(b'changed implementation')
                with self.assertRaisesRegex(ValueError, 'implementation changed'):
                    build.validate_build(root, directory, output, 'retained.json')
                source.write_bytes(b'implementation')
                (root / build.BINARIES[0]).write_bytes(b'wrong library')
                with self.assertRaisesRegex(ValueError, 'binary changed'):
                    build.validate_build(root, directory, output, 'retained.json')
                factory.write_bytes(b'wrong factory')
                with self.assertRaisesRegex(ValueError, 'factory build'):
                    build.validate_build(root, directory, output, 'retained.json')


if __name__ == '__main__':
    unittest.main()
