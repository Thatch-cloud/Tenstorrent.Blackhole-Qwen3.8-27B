import hashlib
from pathlib import Path
import tempfile
import unittest

from dspark_splitk_sim_gate import qualify, verify_sources


class SimulatorGateTests(unittest.TestCase):
    def test_rejects_fabricated_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'report.json'
            report.write_text('{"passed": true}')
            with self.assertRaisesRegex(ValueError, 'Pinned successful'):
                qualify(directory, report)

    def test_rejects_changed_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'kernel.py'
            source.write_bytes(b'qualified')
            manifest = {'kernel.py': hashlib.sha256(source.read_bytes()).hexdigest()}
            verify_sources(directory, manifest)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                verify_sources(directory, manifest)
            with self.assertRaisesRegex(ValueError, 'Nonempty'):
                verify_sources(directory, {})


if __name__ == '__main__':
    unittest.main()
