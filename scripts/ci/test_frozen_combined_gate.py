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
        for context in (True, '32768', 8192, 16384, 131072, 262144):
            with self.assertRaisesRegex(ValueError, 'Only 32768'):
                gate.qualify('.', draft_sources='.', target_sources='.', context=context)

    def test_65536_is_recognized_but_refuses_as_unqualified(self):
        # 65536 is a real staged combined-runtime context (frozen_recipe_context.py
        # COMBINED_RUNTIME_CONTEXTS), unlike the values above - so it must not fall into
        # the generic "Only 32768" rejection reserved for unsupported contexts. It has to
        # refuse for a different, more specific reason: no qualified evidence yet.
        with self.assertRaisesRegex(ValueError, 'no qualified combined-runtime evidence'):
            gate.qualify('.', draft_sources='.', target_sources='.', context=65536)

    def test_65536_placeholder_is_explicit_not_missing(self):
        self.assertIn(65536, gate.CONTEXT_REPORTS)
        self.assertIsNone(gate.CONTEXT_REPORTS[65536])
        self.assertEqual(gate.CONTEXT_REPORTS[32768], gate.REPORTS)

    def test_qualifying_65536_once_evidence_exists_reuses_32768_machinery(self):
        # Once real evidence lands, filling in CONTEXT_REPORTS[65536] is the only change
        # needed - qualify() takes the pins from the table, not a hardcoded 32768 shape.
        # Prove the 65536 pins (not the 32768 ones) are what gets consulted: give it a
        # wrong-content file under the expected name and check it is caught by hash, not
        # skipped or matched against unrelated 32768 report names.
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'only.json'
            path.write_bytes(b'{}')
            with patch.object(gate, 'CONTEXT_REPORTS', {32768: gate.REPORTS, 65536: {'only.json': 'deadbeef'}}):
                with self.assertRaisesRegex(ValueError, 'Pinned component report required: only.json'):
                    gate.qualify(directory, draft_sources='.', target_sources='.', context=65536)


if __name__ == '__main__':
    unittest.main()
