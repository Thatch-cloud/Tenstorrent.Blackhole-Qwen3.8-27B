from copy import deepcopy
import unittest

from mlp_compute_clock import ZONES
from mlp_compute_clock_report import validate
from test_mlp_clock_report import fixture as reader_fixture


def fixture():
    report = reader_fixture()
    old = report.pop('clock_samples')
    samples = [dict(chip=chip, processor=processor, worker=0, zone=name, block=block, subblock=subblock,
        start_cycle=100 * index, end_cycle=100 * index + 20, duration_cycles=20)
        for chip in range(2) for processor in range(3)
        for index, (name, predicate, block, subblock) in enumerate(ZONES)]
    report['compute_clock_samples'] = [dict(label=record['label'], poisoned_before_execution=True,
        samples=deepcopy(samples)) for record in old]
    return report


class ComputeReportTests(unittest.TestCase):
    def test_complete_capture(self):
        result = validate(fixture(), backend='simulator')
        self.assertEqual(result['samples'], 180)
        self.assertIsNone(result['committed_tg'])
        self.assertFalse(result['active_utilization_measured'])

    def test_reject_missing_numerical_or_processor_evidence(self):
        for mutate in (
                lambda report: report['checks'].pop(),
                lambda report: report['trace_replays'][0]['negative_controls'].pop(),
                lambda report: report['compute_clock_samples'].pop(),
                lambda report: report['compute_clock_samples'][0]['samples'].pop(),
                lambda report: report['compute_clock_samples'][0]['samples'][0].update(processor=1),
                lambda report: report['compute_clock_samples'][0]['samples'][0].update(block=11),
                lambda report: report['compute_clock_samples'][0]['samples'][0].update(duration_cycles=0),
                lambda report: report['compute_clock_samples'][0].update(poisoned_before_execution=False),
                lambda report: report.update(committed_tg=200)):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate(report, backend='simulator')
