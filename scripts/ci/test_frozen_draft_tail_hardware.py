from pathlib import Path
import unittest

from frozen_draft_tail_hardware import adapt, validate_hardware, SIMULATOR_SHA256


class DraftTailHardwareTests(unittest.TestCase):
    def fixture(self):
        from test_frozen_draft_tail_gate import DraftTailGateTests
        report = DraftTailGateTests().fixture()
        for check in report['checks']:
            check['position'] += 32960
        report['checks'].extend(dict(position=position, proposals=proposals, chip=chip,
            name=name, seed=2, exact=True) for position in (33024, 33056) for proposals in (7, 15)
            for chip in (0, 1) for name in ('reference_timed', 'candidate_timed'))
        report.update(backend='hardware', simulator_report_sha256=SIMULATOR_SHA256,
            component_timings=[dict(position=position, proposals=proposals, repeat=repeat, arm=arm,
                elapsed_ms=2.0 if arm == 'reference' else 1.0)
                for position in (33024, 33056) for proposals in (7, 15) for repeat in range(4)
                for arm in ('reference', 'candidate', 'candidate', 'reference')])
        return report

    def test_complete_hardware_matrix_without_tg_claim(self):
        result = validate_hardware(self.fixture())
        self.assertIsNone(result['committed_tg'])
        self.assertEqual(len(result['cases']), 4)
        self.assertEqual(result['cases'][0]['median_ms'], dict(reference=2.0, candidate=1.0))

    def test_missing_checks_and_bad_timing_rejected(self):
        for mutate in (lambda report: report['checks'].pop(),
                lambda report: report['checks'][0].update(exact=False),
                lambda report: report['component_timings'].reverse(),
                lambda report: report['component_timings'][0].update(elapsed_ms=float('nan'))):
            report = self.fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate_hardware(report)

    def test_full_geometry_and_simulator_admission_preserve_replay_matrix(self):
        source = Path(__file__).with_name('draft-tail-probe.py').read_text()
        result = adapt(source)
        self.assertIn('for position in (33024, 33056):', result)
        self.assertIn('for proposals in (7, 15):', result)
        self.assertIn('for seed in (1, 0, 2):', result)
        self.assertIn("qualify(Path(__file__).parent, '/experiment/results/simulator'", result)
        self.assertIn("for name in ('reference', 'candidate', 'candidate', 'reference')", result)
        self.assertIn("len(checks) == 128 and len(timings) == 64", result)
        self.assertIn('performance_qualified=False, model_integrated=False', result)
        self.assertIn("or os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT')", result)
        self.assertIn("compare(device_history, history, 'history_unchanged'", result)
        self.assertIn("compare(device_queries, queries, 'queries_unchanged'", result)
        with self.assertRaises(ValueError):
            adapt(result)
