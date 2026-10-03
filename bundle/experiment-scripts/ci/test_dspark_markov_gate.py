import copy
import unittest

from dspark_markov_fixture import expected_manifest
from dspark_markov_gate import qualify


def report_fixture(learned):
    steps, patterns, vocabulary = (7, 2, 248320) if learned else (3, 3, 64)
    replay = [*range(patterns), 0]
    return dict(passed=True, closed_cleanly=True, stage='complete', backend='simulator',
        target_integrated=False, eligible_for_hardware=False, sources={'probe': 'sha'}, sources_after={'probe': 'sha'},
        native_sources={'binary': 'sha'}, native_sources_after={'binary': 'sha'}, vocabulary=vocabulary, proposals=steps,
        fixture=expected_manifest() if learned else None, accuracy_policy='FP32 bias and sum; no SGLang BF16 bitwise claim',
        eager_checks=[dict(pattern=pattern, step=step, chip=chip, token_exact=True, full_vocabulary_close=True,
            token=step, max_abs=0.) for pattern in range(patterns) for step in range(steps) for chip in range(2)],
        replay_checks=[dict(repetition=repetition, pattern=pattern, step=step, chip=chip,
            token_and_scores_exact=True, bindings_stable=True) for repetition, pattern in enumerate(replay)
            for step in range(steps) for chip in range(2)],
        input_checks=[dict(phase=phase, ordinal=ordinal, tensor=tensor, chip=chip, exact=True)
            for phase, count in (('eager', patterns), ('replay', len(replay)))
            for ordinal in range(count) for tensor in range(2) for chip in range(2)],
        weight_checks=[dict(phase=phase, tensor=tensor, chip=chip, exact=True)
            for phase in ('before', 'after') for tensor in range(2) for chip in range(2)],
        stale_controls=[dict(chip=chip, missing_update_detected=True) for chip in range(2)])


def check(report, exit_status='0\n'):
    return qualify(report, sources={'probe': 'sha'}, native={'binary': 'sha'}, exit_status=exit_status)


class DSparkMarkovGateTests(unittest.TestCase):
    def test_complete_toy_and_learned_gates_remain_outside_hardware(self):
        for learned, count in ((False, 80), (True, 100)):
            result = check(report_fixture(learned))
            self.assertEqual(result['checks'], count)
            self.assertEqual(result['learned_matrices'], learned)
            self.assertFalse(result['eligible_for_hardware'])
            self.assertFalse(result['target_integrated'])

    def test_every_coordinate_is_required_without_duplicates(self):
        for name in ('eager_checks', 'replay_checks', 'input_checks', 'weight_checks', 'stale_controls'):
            for duplicate in (False, True):
                report = report_fixture(True)
                if duplicate:
                    report[name].append(copy.deepcopy(report[name][0]))
                else:
                    report[name].pop()
                with self.assertRaises(ValueError):
                    check(report)

    def test_source_drift_and_incomplete_teardown_or_exit_fail(self):
        for name, value in (('passed', False), ('closed_cleanly', False), ('stage', 'capture'),
                ('sources_after', {}), ('native_sources_after', {}), ('backend', 'hardware'),
                ('eligible_for_hardware', True), ('error', 'failure')):
            report = report_fixture(True)
            report[name] = value
            with self.assertRaises(ValueError):
                check(report)
        with self.assertRaises(ValueError):
            check(report_fixture(True), '124')

    def test_failed_checks_and_invalid_numbers_fail(self):
        for field, value in (('token_exact', False), ('full_vocabulary_close', False), ('token', -1),
                ('token', 248320), ('max_abs', float('nan')), ('max_abs', float('inf')), ('max_abs', -1.)):
            report = report_fixture(True)
            report['eager_checks'][0][field] = value
            with self.assertRaises(ValueError):
                check(report)

    def test_replicated_proposal_disagreement_fails(self):
        report = report_fixture(True)
        report['eager_checks'][0]['token'] = 4
        with self.assertRaises(ValueError):
            check(report)

    def test_boolean_coordinates_are_not_integer_audits(self):
        report = report_fixture(True)
        report['eager_checks'][0]['chip'] = False
        with self.assertRaises(ValueError):
            check(report)

    def test_fixture_or_geometry_drift_is_not_a_wider_qualification(self):
        for key, value in (('fixture', None), ('proposals', 15), ('vocabulary', 124160),
                ('accuracy_policy', 'exact BF16'), ('proposals', True)):
            report = report_fixture(True)
            report[key] = value
            with self.assertRaises(ValueError):
                check(report)


if __name__ == '__main__':
    unittest.main()
