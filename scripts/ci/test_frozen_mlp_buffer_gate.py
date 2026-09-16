import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from frozen_mlp_buffer_gate import qualify, REPORT_SHA256
from frozen_mlp_buffer_trial import transform
from frozen_recipe_context import REVISION


class BufferGateTests(unittest.TestCase):
    def test_unknown_report_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / 'report.json'
            report.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Exact retained'):
                qualify(temporary, report)

    @unittest.skipUnless(os.environ.get('QWEN_BUFFER_REPORT'), 'Retained simulator artifact required')
    def test_retained_artifact_against_reconstructed_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ('fused_1d.py', 'fused_1d_input.cpp', 'fused_1d_weights.cpp',
                    'fusion_trace.py', 'fused-t16-target-simulator.json', 'fused-t16-target-simulator.exit-status'):
                payload = subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/{name}'])
                (directory / name).write_bytes(payload)
            candidate = directory / 'frozen_mlp_buffer_candidate.py'
            candidate.write_bytes(transform((directory / 'fused_1d.py').read_text()).encode())
            evidence = qualify(directory, os.environ['QWEN_BUFFER_REPORT'])
            self.assertEqual(evidence['report_sha256'], REPORT_SHA256)
            self.assertFalse(evidence['performance_qualified'])
            candidate.write_bytes(candidate.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'Projection source differs'):
                qualify(directory, os.environ['QWEN_BUFFER_REPORT'])


if __name__ == '__main__':
    unittest.main()
