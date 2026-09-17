import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dspark_proposal_gate as gate
from dspark_hardware_gate import digest


class ProposalGateTests(unittest.TestCase):
    def report(self, root):
        source = root / 'dspark_fixed_inputs.py'
        source.write_text('fixed source')
        hashes = {source.name: digest(source)}
        report = dict(passed=True, closed_cleanly=True, backend='simulator', stage='complete',
            sources=hashes, sources_after=hashes, native_sources={'kernel': 'unchanged'},
            native_sources_after={'kernel': 'unchanged'}, positions=[4096, 4109], capacity=4384, proposal_rows=15,
            chunks=[[0, 2048], [2048, 4096], [4096, 4416]], numerical_tolerances=dict(rtol=.01, atol=.01))
        for name, count in gate.COUNTS.items():
            flag = 'exact' if name == 'input_checks' else 'detected' if name in ('fixture_controls', 'stale_controls') else 'passed'
            report[name] = [{flag: True} for index in range(count)]
        return report

    def qualify(self, root, report):
        path = root / gate.REPORT
        path.write_text(json.dumps(report))
        path.with_suffix('.exit-status').write_text('0\n')
        with patch.object(gate, 'SHA256', digest(path)):
            return gate.qualify(root)

    def test_unqualified_gate_blocks_before_reading_or_opening_hardware(self):
        with patch.object(gate, 'SHA256', None), patch.object(gate, 'digest') as checksum:
            with self.assertRaisesRegex(ValueError, 'not yet been independently qualified'):
                gate.qualify('missing')
            checksum.assert_not_called()

    def test_complete_pinned_matrix_passes_but_incomplete_or_changed_layouts_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self.report(root)
            self.assertEqual(set(self.qualify(root, baseline)), {gate.REPORT})
            for key, value in (('closed_cleanly', False), ('passed', False), ('positions', [4096, 4097]),
                    ('capacity', 4096), ('proposal_rows', 7), ('numerical_tolerances', dict(rtol=.02, atol=.02)),
                    ('native_sources_after', {'kernel': 'changed'}), ('replay_checks', [])):
                report = copy.deepcopy(baseline)
                report[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.qualify(root, report)
            report = copy.deepcopy(baseline)
            report['fixture_controls'][0]['detected'] = False
            with self.assertRaisesRegex(ValueError, 'Every fixed-capacity'):
                self.qualify(root, report)
            (root / 'dspark_fixed_inputs.py').write_text('modified source')
            with self.assertRaisesRegex(ValueError, 'Changed simulated'):
                self.qualify(root, baseline)


if __name__ == '__main__':
    unittest.main()
