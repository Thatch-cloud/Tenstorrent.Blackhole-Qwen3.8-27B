import copy
import hashlib
from pathlib import Path
import unittest

from compact_score_report import validate


class ReportTests(unittest.TestCase):
    def fixture(self):
        directory = Path(__file__).parent
        names = ('compact_score_device.py', 'compact_score_io.cpp', 'compact_score_compute.cpp',
                 'compact_score_reduce.cpp', 'dspark_score_layout.py', 'dspark_score_layout_io.cpp',
                 'dspark_score_layout_compute.cpp', 'attention_batch.py', 'gdn_multitoken_conv.py')
        sources = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}
        sources['gdn-output-grid-probe.py'] = hashlib.sha256((directory / 'compact-score-probe.py').read_bytes()).hexdigest()
        checks = [dict(pattern=pattern, step=step, mode=mode, chip=chip, exact=True)
                  for mode, patterns in (('eager', (0,)), ('replay', (1, 2, 3, 0)))
                  for pattern in patterns for step in (0, 14) for chip in (0, 1)]
        return dict(passed=True, closed_cleanly=True, backend='simulator', vocabulary=248320,
                    stage='complete', performance_qualified=False, sources=sources,
                    sources_after=dict(sources), checks=checks,
                    immutable_checks=[dict(record, operand=operand) for record in checks for operand in (0, 1)])

    def test_complete_matrix_and_fail_closed_mutations(self):
        fixture = self.fixture()
        self.assertFalse(validate(fixture, Path(__file__).parent)['hardware_qualified'])
        mutations = (
            lambda report: report.update(closed_cleanly=False),
            lambda report: report.update(vocabulary=64),
            lambda report: report.update(performance_qualified=True),
            lambda report: report['checks'].pop(),
            lambda report: report['checks'].__setitem__(0, report['checks'][1]),
            lambda report: report['immutable_checks'][0].update(exact=False),
            lambda report: report['sources_after'].update(extra='0' * 64),
            lambda report: (report['sources'].update({'compact_score_reduce.cpp': '0' * 64}),
                            report['sources_after'].update({'compact_score_reduce.cpp': '0' * 64})),
        )
        for mutation in mutations:
            report = copy.deepcopy(fixture)
            mutation(report)
            with self.assertRaises(ValueError):
                validate(report, Path(__file__).parent)
