import copy
import unittest

from dspark_markov_gate import qualify_native
from dspark_native_reference import POLICY
from test_dspark_markov_gate import check, report_fixture


def native_fixture(learned=True):
    report = report_fixture(learned)
    report.update(accuracy_policy=POLICY, native_arithmetic_reference=True, fp32_replacement_qualified=False,
        fp32_diagnostic_tolerances=dict(rtol=1e-4, atol=1e-4))
    for entry in report['eager_checks']:
        step, pattern = entry['step'], entry['pattern']
        anchors = (1596, 248319) if learned else (2, 63, 0)
        previous = step - 1 if step else anchors[pattern]
        entry.pop('full_vocabulary_close')
        entry.update(full_vocabulary_exact=True, previous=previous, fp32_previous=previous,
            same_input_fp32=dict(max_abs=.0025, mismatched=42, full_vocabulary_close=False, token=step),
            fp32_trajectory=dict(max_abs=.0025, mismatched=42, full_vocabulary_close=False, token=step))
    return report


def check_native(report, exit_status='0'):
    return qualify_native(report, sources={'probe': 'sha'}, native={'binary': 'sha'}, exit_status=exit_status)


class DSparkNativeGateTests(unittest.TestCase):
    def test_complete_matrix_is_native_only_despite_fp32_score_failures(self):
        for learned, checks in ((False, 80), (True, 100)):
            result = check_native(native_fixture(learned))
            self.assertEqual(result['checks'], checks)
            self.assertGreater(result['fp32_score_checks_failed'], 0)
            self.assertFalse(result['fp32_replacement_qualified'])
            self.assertFalse(result['eligible_for_hardware'])

    def test_policies_cannot_be_silently_interchanged(self):
        with self.assertRaises(ValueError):
            check(native_fixture())
        with self.assertRaises(ValueError):
            check_native(report_fixture(True))
        report = native_fixture()
        report['accuracy_policy'] = report_fixture(True)['accuracy_policy']
        with self.assertRaises(ValueError):
            check(report)

    def test_every_coordinate_is_required_without_duplicates(self):
        for name in ('eager_checks', 'replay_checks', 'input_checks', 'weight_checks', 'stale_controls'):
            for duplicate in (False, True):
                report = native_fixture()
                if duplicate:
                    report[name].append(copy.deepcopy(report[name][0]))
                else:
                    report[name].pop()
                with self.assertRaises(ValueError):
                    check_native(report)

    def test_lifecycle_source_and_policy_drift_reject(self):
        for name, value in (('passed', False), ('closed_cleanly', False), ('sources_after', {}),
                ('native_sources_after', {}), ('backend', 'hardware'), ('stage', 'capture'),
                ('eligible_for_hardware', True), ('fixture', None), ('fp32_replacement_qualified', True),
                ('native_arithmetic_reference', False), ('fp32_diagnostic_tolerances', dict(rtol=.01, atol=.01))):
            report = native_fixture()
            report[name] = value
            with self.assertRaises(ValueError):
                check_native(report)
        with self.assertRaises(ValueError):
            check_native(native_fixture(), '124')

    def test_native_tolerance_and_incomplete_diagnostics_reject(self):
        for field, value in (('max_abs', 1e-8), ('full_vocabulary_exact', False), ('previous', True),
                ('previous', 2), ('fp32_previous', 2), ('same_input_fp32', {}), ('fp32_trajectory', None)):
            report = native_fixture()
            report['eager_checks'][0][field] = value
            with self.assertRaises(ValueError):
                check_native(report)
        for field, value in (('max_abs', float('nan')), ('max_abs', -1), ('mismatched', 248321),
                ('mismatched', True), ('mismatched', 0), ('full_vocabulary_close', True), ('token', -1)):
            report = native_fixture()
            report['eager_checks'][0]['fp32_trajectory'][field] = value
            with self.assertRaises(ValueError):
                check_native(report)

    def test_divergent_fp32_trajectory_must_use_its_own_previous_token(self):
        report = native_fixture()
        for entry in report['eager_checks']:
            step = entry['step']
            entry['fp32_trajectory']['token'] = step + 10
            if step:
                entry['fp32_previous'] = step + 9
            else:
                entry['same_input_fp32']['token'] = 10
        result = check_native(report)
        self.assertEqual(result['fp32_token_checks_differed'], 28)
        report['eager_checks'][2]['fp32_previous'] = report['eager_checks'][2]['previous']
        with self.assertRaises(ValueError):
            check_native(report)


if __name__ == '__main__':
    unittest.main()
