import copy
from pathlib import Path
import tempfile
import unittest

from gdn_shared_qk_t32_gate import qualify, validate_report


def fixture():
    report = dict(passed=True, closed_cleanly=True, stage='complete', backend='simulator',
        rows=32, hardware_qualified=False, timing_qualified=False, norm_unchanged=True,
        state_math_unchanged=True, shared_qk_preparation=True, norm_bridge_scatter=True,
        norm_direct_read_bytes=8192, sources={'fixture': 'unchanged'}, sources_after={'fixture': 'unchanged'})
    for field, operands in (('checks', 3), ('immutable_checks', 6)):
        report[field] = [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
            for operand in range(operands) for chip in (0, 1)]
    report['stale_controls'] = [dict(seed=seed, chip=chip, detected=True)
        for seed in (1, 2, 0) for chip in (0, 1)]
    return report


class T32RecurrenceAdmissionTests(unittest.TestCase):
    def test_complete_matrix(self):
        validate_report(fixture())

    def test_rejects_partial_stale_changed_or_wrong_width_evidence(self):
        for key, value in (('rows', 16), ('rows', 32.0), ('passed', 1),
                ('hardware_qualified', True), ('sources_after', {}), ('error', 'failure')):
            report = fixture()
            report[key] = value
            with self.assertRaises(ValueError):
                validate_report(report)
        for field in ('checks', 'immutable_checks', 'stale_controls'):
            report = copy.deepcopy(fixture())
            report[field].pop()
            with self.assertRaises(ValueError):
                validate_report(report)

    def test_unretained_report_cannot_admit_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / 'gdn-shared-recurrence.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Exact retained T32'):
                qualify(temporary, temporary, temporary)


if __name__ == '__main__':
    unittest.main()
