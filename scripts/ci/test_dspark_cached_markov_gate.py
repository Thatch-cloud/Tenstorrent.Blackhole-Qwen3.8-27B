import tempfile
import unittest
from pathlib import Path

from dspark_cached_markov_gate import REPORTS, qualify


class CachedMarkovGateTests(unittest.TestCase):
    def test_missing_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                qualify(directory, directory)

    def test_fabricated_success_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in REPORTS:
                (Path(directory) / (name + '.json')).write_text('{"passed": true}')
            with self.assertRaisesRegex(ValueError, 'Exact accepted'):
                qualify(directory, directory)
