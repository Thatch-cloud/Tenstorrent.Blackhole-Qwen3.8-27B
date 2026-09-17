from copy import deepcopy
import unittest

from gdn_wait_clock import ZONES
from gdn_wait_clock_report import MODES, validate


def fixture():
    report = dict(passed=True, closed_cleanly=True, norm_unchanged=True, state_math_unchanged=True,
        shared_qk_preparation=True, wait_clock=True, missing_execution_rejected=True,
        missing_execution_rejected_after_replay=True, stage='complete', backend='simulator', rows=16,
        sources={'kernel': 'abc'}, sources_after={'kernel': 'abc'}, hardware_qualified=False,
        timing_qualified=False, committed_tg=None,
        generated_kernels=[dict(control_sha256='1' * 64, candidate_sha256='2' * 64, token=8)])
    for field, operands in (('checks', 3), ('immutable_checks', 6)):
        report[field] = [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in MODES for operand in range(operands) for chip in (0, 1)]
    report['wait_clock_samples'] = [dict(label=mode, poisoned_before_execution=True,
        samples=[dict(chip=chip, processor=processor, worker=0, token=8, zone=name,
            start_cycle=100 * index, end_cycle=100 * index + 20, duration_cycles=20)
            for chip in range(2) for processor in range(3) for index, (name, _) in enumerate(ZONES)])
        for mode in MODES]
    return report


class WaitReportTests(unittest.TestCase):
    def test_complete_matrix_is_not_performance_or_source_admission(self):
        result = validate(fixture())
        self.assertEqual(result['samples'], 168)
        self.assertTrue(result['source_admission_required'])
        self.assertFalse(result['performance_qualified'])
        self.assertIsNone(result['committed_tg'])

    def test_incomplete_numeric_or_sample_matrix_rejected(self):
        for field in ('checks', 'immutable_checks', 'wait_clock_samples'):
            report = fixture()
            report[field].pop()
            with self.assertRaises(ValueError):
                validate(report)
        for field, value in (('token', 9), ('chip', 1), ('duration_cycles', 21),
                ('start_cycle', -1), ('end_cycle', 100_000_001)):
            report = fixture()
            report['wait_clock_samples'][0]['samples'][0][field] = value
            with self.assertRaises(ValueError):
                validate(report)

    def test_missing_poison_source_drift_or_speed_claim_rejected(self):
        for field, value in (('missing_execution_rejected', False),
                ('missing_execution_rejected_after_replay', False), ('sources_after', {}),
                ('hardware_qualified', True), ('committed_tg', 200), ('backend', 'hardware')):
            report = fixture()
            report[field] = value
            with self.assertRaises(ValueError):
                validate(report)
        report = fixture()
        report['wait_clock_samples'][1] = deepcopy(report['wait_clock_samples'][0])
        with self.assertRaises(ValueError):
            validate(report)


if __name__ == '__main__':
    unittest.main()
