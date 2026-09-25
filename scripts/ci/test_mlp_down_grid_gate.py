import copy
import unittest

from mlp_down_grid_gate import validate_report


def fixture():
    labels = ('eager_0', 'eager_1', 'eager_2', 'replay_1', 'replay_2', 'replay_0')
    return dict(passed=True, closed_cleanly=True, backend='simulator', stage='complete',
        hardware_qualified=False, timing_qualified=False, projection='mlp_down', rows=16,
        k=8704, n=5120, grids=dict(control=[11, 3], candidate=[11, 8]),
        output_placement='L1', weights='bfloat8_b', sources={'source': 'a' * 64}, sources_after={'source': 'a' * 64},
        checks=[dict(label=label, chip=chip, exact=True) for label in labels for chip in (0, 1)],
        immutable_checks=[dict(label=label, operand='activation', chip=chip, exact=True)
            for label in labels for chip in (0, 1)] + [dict(label='complete', operand='weight', chip=chip, exact=True)
            for chip in (0, 1)], negative_controls=[dict(label=label, poisoned=True) for label in labels[3:]])


class DownGateTests(unittest.TestCase):
    def test_complete_matrix(self):
        validate_report(fixture())

    def test_partial_forged_or_other_projection_rejected(self):
        for mutate in (lambda report: report['checks'].pop(), lambda report: report['immutable_checks'].pop(),
                lambda report: report['negative_controls'].pop(), lambda report: report['sources_after'].clear(),
                lambda report: report.update(projection='gdn_output'), lambda report: report.update(k=3072),
                lambda report: report.update(output_placement='DRAM'), lambda report: report.update(timing_qualified=True)):
            report = copy.deepcopy(fixture())
            mutate(report)
            with self.assertRaises(ValueError):
                validate_report(report)
