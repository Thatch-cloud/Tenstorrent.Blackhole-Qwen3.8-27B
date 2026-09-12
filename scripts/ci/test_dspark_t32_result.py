import copy
import importlib.util
from pathlib import Path
import unittest

from dspark_t32_result import validate


def complete_report():
    report = dict(passed=True, closed_cleanly=True, backend='simulator', stage='complete',
        proposals=31, vocabulary=64, native_arithmetic_reference=True,
        target_integrated=False, eligible_for_hardware=False,
        sources={'probe': 'sha'}, sources_after={'probe': 'sha'},
        native_sources={'runtime': 'sha'}, native_sources_after={'runtime': 'sha'})
    report['eager_checks'] = [dict(pattern=pattern, chip=chip, step=step,
        token_exact=True, full_vocabulary_exact=True)
        for pattern in range(3) for chip in range(2) for step in range(31)]
    report['replay_checks'] = [dict(repetition=repetition, pattern=pattern, chip=chip, step=step,
        token_and_scores_exact=True, bindings_stable=True)
        for repetition, pattern in enumerate((0, 1, 2, 0)) for chip in range(2) for step in range(31)]
    report['input_checks'] = [dict(phase=phase, ordinal=ordinal, tensor=tensor, chip=chip, exact=True)
        for phase, count in (('eager', 3), ('replay', 4)) for ordinal in range(count)
        for tensor in range(2) for chip in range(2)]
    report['weight_checks'] = [dict(phase=phase, tensor=tensor, chip=chip, exact=True)
        for phase in ('before', 'after') for tensor in range(2) for chip in range(2)]
    report['stale_controls'] = [dict(chip=chip, missing_update_detected=True) for chip in range(2)]
    return report


class T32ResultTests(unittest.TestCase):
    def test_learned_coverage_requires_explicit_manifest_and_policy(self):
        from dspark_markov_fixture import expected_manifest

        report = complete_report()
        report.update(vocabulary=248320, fixture=expected_manifest())
        report['eager_checks'] = [entry for entry in report['eager_checks'] if entry['pattern'] < 2]
        report['replay_checks'] = [dict(entry, pattern=0 if entry['repetition'] == 2 else entry['pattern'])
            for entry in report['replay_checks'] if entry['repetition'] < 3]
        report['input_checks'] = [entry for entry in report['input_checks']
            if entry['ordinal'] < (2 if entry['phase'] == 'eager' else 3)]
        self.assertEqual(validate(report, learned=True)['eager_queries'], 124)
        with self.assertRaises(ValueError):
            validate(report)
        report['fixture'] = {}
        with self.assertRaises(ValueError):
            validate(report, learned=True)

    def test_complete_coverage_is_synthetic_only(self):
        result = validate(complete_report())
        self.assertEqual(result['eager_queries'], 186)
        self.assertEqual(result['replay_queries'], 248)
        self.assertFalse(result['target_integrated'])

    def test_missing_duplicate_and_failed_records_reject(self):
        for field in ('eager_checks', 'replay_checks', 'input_checks', 'weight_checks', 'stale_controls'):
            for mutation in ('missing', 'duplicate', 'failed'):
                report = complete_report()
                if mutation == 'missing':
                    report[field].pop()
                elif mutation == 'duplicate':
                    report[field][-1] = copy.deepcopy(report[field][0])
                else:
                    for key, value in report[field][-1].items():
                        if value is True:
                            report[field][-1][key] = False
                with self.subTest(field=field, mutation=mutation), self.assertRaises(ValueError):
                    validate(report)

    def test_runtime_drift_and_premature_pass_reject(self):
        for field, value in (('closed_cleanly', False), ('stage', 'trace_capture'),
                ('native_sources_after', {}), ('sources_after', {}), ('proposals', 15),
                ('eligible_for_hardware', True), ('native_arithmetic_reference', False)):
            report = complete_report()
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(report)

    def test_probe_source_hashing_runs_before_ci(self):
        path = Path(__file__).with_name('dspark-t32-markov-probe.py')
        spec = importlib.util.spec_from_file_location('t32_probe_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        hashes = module.source_hashes()
        self.assertIn(path.name, hashes)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))


if __name__ == '__main__':
    unittest.main()
