import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dspark_splitk_hardware_gate import qualify, validate_matrix


class HardwareGateTests(unittest.TestCase):
    def test_unpinned_report_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'report.json'
            report.write_text('{"passed": true}')
            with self.assertRaisesRegex(ValueError, 'Exact retained'):
                qualify(directory, report, report)

    @unittest.skipUnless(os.environ.get('QWEN_SPLITK_HARDWARE_REPORT'), 'Retained hardware artifact required')
    def test_retained_report_and_corruptions(self):
        directory = Path(__file__).parent
        report_path = Path(os.environ['QWEN_SPLITK_HARDWARE_REPORT'])
        simulator = Path(os.environ['QWEN_SPLITK_SIMULATOR_REPORT'])
        evidence = qualify(directory, report_path, simulator)
        self.assertTrue(evidence['component_qualified'])
        for name in ('runtime_admitted', 'full_request_qualified', 'performance_qualified', 'serving_qualified'):
            self.assertFalse(evidence[name])
        original = json.loads(report_path.read_text())
        for section in ('eager_checks', 'replay_checks', 'input_checks', 'layout_checks', 'fixture_controls', 'stale_controls'):
            incomplete = copy.deepcopy(original)
            incomplete[section].pop()
            with self.subTest(section=section), self.assertRaises(ValueError):
                validate_matrix(incomplete)
        for field, value in (('failed_elements', 1), ('replay_exact', False), ('numerical_close', False)):
            changed = copy.deepcopy(original)
            changed['replay_checks'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_matrix(changed)
        original_read = Path.read_bytes

        def modified(path):
            data = original_read(path)
            return data + b'\n' if path.name == 'dspark_splitk_tree_denominator.py' else data

        with patch.object(Path, 'read_bytes', modified):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                qualify(directory, report_path, simulator)


if __name__ == '__main__':
    unittest.main()
