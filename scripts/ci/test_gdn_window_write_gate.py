import unittest

from gdn_window_write_gate import SOURCES, validate


def fixture():
    modes = [('eager', 0), ('replay', 0), ('replay', 1), ('replay', 2)]
    return dict(passed=True, closed_cleanly=True, backend='simulator', stage='complete',
        hardware_qualified=False, timing_qualified=False,
        sources=dict.fromkeys(SOURCES, 'a' * 64), sources_after=dict.fromkeys(SOURCES, 'a' * 64),
        checks=[dict(seed=seed, mode=mode, arm=arm, slot=slot, chip=chip, exact=True)
            for mode, seed in modes for arm in ('control', 'overlap') for slot in range(4) for chip in range(2)],
        immutable_checks=[dict(seed=seed, mode=mode, operand=operand, chip=chip, exact=True)
            for mode, seed in modes for operand in range(9) for chip in range(2)])


class WindowGateTests(unittest.TestCase):
    def test_complete_matrix(self):
        validate(fixture())

    def test_incomplete_changed_or_non_simulator_evidence_rejected(self):
        for mutate in (lambda report: report['checks'].pop(),
                lambda report: report['immutable_checks'].pop(),
                lambda report: report['checks'][0].update(exact=False),
                lambda report: report.update(closed_cleanly=False),
                lambda report: report.update(backend='hardware'),
                lambda report: report['sources_after'].update(extra='changed')):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate(report)
