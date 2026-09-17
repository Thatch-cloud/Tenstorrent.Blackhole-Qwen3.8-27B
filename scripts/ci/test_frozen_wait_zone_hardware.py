import copy
import json
import os
from pathlib import Path
import unittest

from frozen_wait_zone_hardware import validate_numerics, adapt_hardware_probe, qualify
from frozen_mlp_buffer_trial import adapt_probe


class HardwareMarkerTests(unittest.TestCase):
    def fixture(self):
        return dict(passed=True, backend='hardware', math_approx_mode=True, timings=[],
            checks=[dict(rows=16, chip=chip, exact=True) for chip in (0, 1)],
            trace_replays=[dict(rows=16, passed=True,
                checks=[dict(arm=arm, repetition=index, pattern=pattern, chip=chip, exact=True)
                    for index, pattern in enumerate((0, 1, 0)) for arm in ('control', 'fused') for chip in (0, 1)],
                negative_controls=[dict(arm=arm, chip=chip, stale_input_detected=True)
                    for arm in ('control', 'fused') for chip in (0, 1)])],
            weight_checks=[dict(projection=projection, chip=chip, pages=43520,
                mismatched_words=0, exact=True, source_exact=True)
                for projection in ('gate', 'up') for chip in (0, 1)])

    def test_full_matrix(self):
        validate_numerics(self.fixture(), 'hardware')

    def test_incomplete_or_timed_report_rejected(self):
        for mutation in (
                lambda report: report['checks'].pop(),
                lambda report: report['trace_replays'][0]['checks'].pop(),
                lambda report: report['trace_replays'][0]['negative_controls'].pop(),
                lambda report: report['weight_checks'].pop(),
                lambda report: report['weight_checks'][0].update(mismatched_words=1),
                lambda report: report['timings'].append(1),
                lambda report: report.update(passed=False)):
            report = copy.deepcopy(self.fixture())
            mutation(report)
            with self.assertRaises(ValueError):
                validate_numerics(report, 'hardware')

    def test_hardware_probe_keeps_environment_gate(self):
        source = Path(__file__).with_name('fused-batch-probe.py').read_text()
        candidate = adapt_hardware_probe(adapt_probe(source))
        self.assertIn('require_projection_environment(os.environ, options.hardware)', candidate)
        self.assertIn('if not options.hardware or options.timing or not (', candidate)
        self.assertIn('options.trace_replay and options.device_weight_check', candidate)
        with self.assertRaises(ValueError):
            adapt_hardware_probe(candidate)

    @unittest.skipUnless(os.environ.get('QWEN_MARKER_EVIDENCE'), 'retained artifact optional')
    def test_actual_simulator_matrix(self):
        evidence = Path(os.environ['QWEN_MARKER_EVIDENCE'])
        report = json.loads((evidence / 'fused-batch.json').read_text())
        validate_numerics(report, 'simulator')
        with self.assertRaises(ValueError):
            qualify(Path(__file__).parent, evidence)


if __name__ == '__main__':
    unittest.main()
