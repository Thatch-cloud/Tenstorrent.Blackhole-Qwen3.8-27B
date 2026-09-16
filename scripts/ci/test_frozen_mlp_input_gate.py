import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from frozen_mlp_input_gate import qualify, validate_report, REPORT_SHA256
from frozen_mlp_input_prefetch import projection, reader
from frozen_recipe_context import REVISION


class InputGateTests(unittest.TestCase):
    def test_unknown_report_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / 'report.json'
            report.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Exact retained'):
                qualify(temporary, report)

    @unittest.skipUnless(os.environ.get('QWEN_INPUT_REPORT'), 'Retained simulator report required')
    def test_retained_report_and_candidate_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ('fused_1d.py', 'fused_1d_input.cpp', 'fused_1d_weights.cpp',
                    'fusion_trace.py', 'fused-t16-target-simulator.json', 'fused-t16-target-simulator.exit-status'):
                (directory / name).write_bytes(subprocess.check_output(['git', 'show',
                    f'{REVISION}:scripts/ci/{name}']))
            candidate = directory / 'frozen-mlp-input-candidate'
            candidate.mkdir()
            (candidate / 'fused_1d.py').write_bytes(projection((directory / 'fused_1d.py').read_text()).encode())
            (candidate / 'fused_1d_input.cpp').write_bytes(reader((directory / 'fused_1d_input.cpp').read_text()).encode())
            (candidate / 'fused_1d_weights.cpp').write_bytes((directory / 'fused_1d_weights.cpp').read_bytes())
            evidence = qualify(directory, os.environ['QWEN_INPUT_REPORT'])
            self.assertEqual(evidence['report_sha256'], REPORT_SHA256)
            self.assertFalse(evidence['performance_qualified'])
            report = json.loads(Path(os.environ['QWEN_INPUT_REPORT']).read_bytes())
            for field in ('checks', 'weight_checks', 'trace_replays'):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_report(dict(report, **{field: []}))
            original = (candidate / 'fused_1d_input.cpp').read_bytes()
            (candidate / 'fused_1d_input.cpp').write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'Projection source differs'):
                qualify(directory, os.environ['QWEN_INPUT_REPORT'])


if __name__ == '__main__':
    unittest.main()
