import copy
import unittest

from mlp_clock_report import validate
from mlp_clock_samples import ZONES


def fixture():
    samples = []
    for role, workers in (('input', (0, 1)), ('weights', (0,))):
        for chip in range(2):
            for worker in workers:
                for index in ((0, 4) if role == 'input' and worker == 1 else (0, 1, 2, 3)):
                    samples.append(dict(role=role, chip=chip, worker=worker, zone=ZONES[role][index][1],
                        start_cycle=100 * index, end_cycle=100 * index + 20, duration_cycles=20))
    return dict(passed=True, backend='simulator', math_approx_mode=True,
        missing_execution_rejected=True, committed_tg=None, timings=[],
        checks=[dict(rows=16, chip=chip, exact=True) for chip in range(2)],
        weight_checks=[dict(projection=projection, chip=chip, exact=True, source_exact=True,
            pages=43520, workers=64, mismatched_words=0) for projection in ('gate', 'up') for chip in range(2)],
        trace_replays=[dict(rows=16, passed=True, timings=[],
            checks=[dict(arm=arm, repetition=repetition, pattern=pattern, chip=chip, exact=True)
                for repetition, pattern in enumerate((0, 1, 0)) for arm in ('control', 'fused') for chip in range(2)],
            negative_controls=[dict(arm=arm, chip=chip, stale_input_detected=True)
                for arm in ('control', 'fused') for chip in range(2)])],
        clock_samples=[dict(label=label, poisoned_before_execution=True, samples=copy.deepcopy(samples))
            for label in ('eager', 'replay-0', 'replay-0', 'replay-1', 'replay-2')])


class ClockReportTests(unittest.TestCase):
    def test_complete_diagnostic_does_not_claim_throughput(self):
        result = validate(fixture(), backend='simulator')
        self.assertEqual(result['samples'], 100)
        self.assertIsNone(result['committed_tg'])
        self.assertTrue(result['diagnostic_only'])

    def test_missing_corrupt_duplicate_or_unqualified_records_rejected(self):
        mutations = (
            lambda report: report.update(committed_tg=200),
            lambda report: report.update(missing_execution_rejected=False),
            lambda report: report['checks'].pop(),
            lambda report: report['weight_checks'][0].update(pages=1),
            lambda report: report['trace_replays'][0]['checks'].pop(),
            lambda report: report['trace_replays'][0]['negative_controls'].pop(),
            lambda report: report['clock_samples'].pop(),
            lambda report: report['clock_samples'][0]['samples'].pop(),
            lambda report: report['clock_samples'][0]['samples'][1].update(zone=ZONES['input'][0][1]),
            lambda report: report['clock_samples'][0]['samples'][0].update(duration_cycles=10),
            lambda report: report['clock_samples'][0]['samples'][1].update(start_cycle=0, end_cycle=20),
            lambda report: report['clock_samples'][0].update(poisoned_before_execution=False),
        )
        for mutate in mutations:
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate(report, backend='simulator')
        with self.assertRaises(ValueError):
            validate(fixture(), backend='hardware')
