import copy
import unittest

from gdn_dual_state_gate import KERNEL, validate_report


def fixture():
    report = dict(passed=True, closed_cleanly=True, norm_unchanged=True,
        state_math_unchanged=True, shared_qk_preparation=True, dual_state_copy=True,
        stage='complete', backend='simulator', rows=16, sources={'source': 'a' * 64},
        sources_after={'source': 'a' * 64}, hardware_qualified=False, timing_qualified=False,
        generated_kernels=[dict(KERNEL)])
    for field, operands in (('checks', 3), ('immutable_checks', 6)):
        report[field] = [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
            for operand in range(operands) for chip in (0, 1)]
    return report


class DualStateGateTests(unittest.TestCase):
    def test_complete_component_evidence(self):
        validate_report(fixture())

    def test_missing_changed_or_promoted_evidence_rejected(self):
        for mutate in (lambda report: report['checks'].pop(),
                lambda report: report['immutable_checks'].pop(),
                lambda report: report['generated_kernels'][0].update(feedback_cb=18),
                lambda report: report['sources_after'].clear(),
                lambda report: report.update(hardware_qualified=True),
                lambda report: report.update(numerical_failures=[{}])):
            report = copy.deepcopy(fixture())
            mutate(report)
            with self.assertRaises(ValueError):
                validate_report(report)
