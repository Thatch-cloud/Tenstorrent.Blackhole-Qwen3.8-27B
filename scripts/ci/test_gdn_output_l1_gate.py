import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gdn_output_l1_gate import LOCAL_SOURCES, NATIVE_SOURCE, qualify


class OutputGateTests(unittest.TestCase):
    def test_exact_matrix_and_source_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {f'/experiment-scripts/ci/{name}': root / name for name in LOCAL_SOURCES}
            paths['/opt/tt-metal/' + NATIVE_SOURCE] = root / NATIVE_SOURCE
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('source')
            sources = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
            report = dict(passed=True, closed_cleanly=True, scope_restored=True, backend='simulator',
                hardware_qualified=False, timing_qualified=False, sources=sources, sources_after=sources,
                checks=[dict(label=label, chip=chip, exact=True)
                    for label in ('eager_0', 'eager_1', 'eager_2', 'replay_1', 'replay_2', 'replay_0')
                    for chip in (0, 1)])
            report_path = root / 'report.json'
            report_path.write_text(json.dumps(report))
            with patch('gdn_output_l1_gate.REPORT_SHA256', hashlib.sha256(report_path.read_bytes()).hexdigest()):
                self.assertFalse(qualify(report_path, root, root)['hardware_qualified'])
                (root / NATIVE_SOURCE).write_text('changed')
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    qualify(report_path, root, root)
                (root / NATIVE_SOURCE).write_text('source')
            report['checks'].pop()
            report_path.write_text(json.dumps(report))
            with patch('gdn_output_l1_gate.REPORT_SHA256', hashlib.sha256(report_path.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, 'Complete stable'):
                    qualify(report_path, root, root)

    def test_unknown_report_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'report.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Exact retained'):
                qualify(path, temporary, temporary)
