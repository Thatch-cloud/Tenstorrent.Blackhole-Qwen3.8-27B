import json
from pathlib import Path
import unittest
from unittest.mock import patch

import splitk_workers_gate as candidate


class WorkersGateTests(unittest.TestCase):
    def fixture(self):
        control = dict(diagnostic_override={'key_chunk_size': 256}, splitk_factory={'builders': {'factory': 'hash'}})
        report = dict(passed=True, closed_cleanly=True, backend='simulator', capacity=384,
            positions=[128, 369], proposal_rows=15, candidate='splitk-maxima-worker-limit16',
            numerical_tolerances=dict(rtol=.01, atol=.01), splitk_execution_calls=3,
            diagnostic_override=dict(key_chunk_size=256, max_cores_per_head=16, local_maximum_storage='float32'),
            worker_calls=[dict(key_chunk_size=256, requested_worker_limit=8, selected_worker_limit=16,
                stripe_keys=False, fp32_dest_acc=True)] * 3,
            sources={'source': 'hash'}, sources_after={'source': 'hash'},
            native_sources={'native': 'hash'}, native_sources_after={'native': 'hash'},
            candidate_sources={'worker': 'hash'},
            target_attention=dict(passed=True, closed=True, sources={'target': 'hash'}, sources_after={'target': 'hash'}),
            splitk_factory=dict(passed=True, device_access=False, source_after=candidate.FACTORY_SHA256,
                builders={'factory': 'hash'}))
        for field, count, flag in (('input_checks', 48, 'exact'), ('layout_checks', 16, 'passed'),
                ('fixture_controls', 8, 'detected'), ('stale_controls', 2, 'detected')):
            report[field] = [{flag: True} for ordinal in range(count)]
        for mode, cases in (('eager', (0, 1)), ('replay', (1, 0))):
            report[mode + '_checks'] = [dict(ordinal=ordinal, case=case, chip=chip,
                passed=True, failed_elements=0, numerical_close=True, replay_exact=True)
                for ordinal, case in enumerate(cases) for chip in range(2)]
        return control, report

    def qualify(self, control, report):
        with patch.object(candidate, 'qualify_control'), \
                patch.object(candidate, 'digest', return_value=candidate.REPORT_SHA256), \
                patch.object(candidate, 'verify_sources'), \
                patch.object(Path, 'read_bytes', side_effect=[json.dumps(control).encode(), json.dumps(report).encode()]):
            return candidate.qualify('.', 'workers.json')

    def test_simulator_pass_does_not_grant_hardware_or_timing(self):
        result = self.qualify(*self.fixture())
        self.assertTrue(result['simulator_qualified'])
        for field in ('hardware_qualified', 'full_request_qualified', 'performance_qualified', 'serving_qualified'):
            self.assertFalse(result[field])

    def test_missing_negative_control_or_changed_worker_call_rejected(self):
        for field in ('fixture_controls', 'worker_calls', 'replay_checks'):
            control, report = self.fixture()
            report[field].pop()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.qualify(control, report)

    def test_unintended_arithmetic_change_rejected(self):
        control, report = self.fixture()
        report['diagnostic_override']['key_chunk_size'] = 128
        with self.assertRaises(ValueError):
            self.qualify(control, report)


if __name__ == '__main__':
    unittest.main()
