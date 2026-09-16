import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import frozen_combined_gate as gate


class CombinedGateTests(unittest.TestCase):
    def test_reports_are_content_pinned(self):
        with TemporaryDirectory() as directory:
            payload = json.dumps({'passed': True}).encode()
            path = Path(directory) / 'fixture.json'
            path.write_bytes(payload)
            with patch.object(gate, 'REPORTS', {'fixture.json': hashlib.sha256(payload).hexdigest()}):
                self.assertEqual(gate.load_reports(directory), {'fixture.json': {'passed': True}})
                path.write_bytes(payload + b' ')
                with self.assertRaisesRegex(ValueError, 'Pinned component'):
                    gate.load_reports(directory)

    def test_other_contexts_cannot_borrow_32k_evidence(self):
        for context in (True, '32768', 8192, 16384, 65536, 131072, 262144):
            with self.assertRaisesRegex(ValueError, 'Only 32768'):
                gate.qualify('.', draft_sources='.', target_sources='.', context=context)


if __name__ == '__main__':
    unittest.main()
