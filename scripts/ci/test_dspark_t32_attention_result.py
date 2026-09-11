import copy
import unittest

from dspark_t32_attention_result import NAMES, validate
from t32_ci_runtime import PINS


class ResultTests(unittest.TestCase):
    def setUp(self):
        sources = {'probe.py': 'a' * 64}
        resource = dict(bounded=True, boot_id='test-boot', cgroup='/test', cpu_period=100000,
                        cpu_quota=1600000, limits={'memory.max': 64 * 1024**3, 'memory.swap.max': 0},
                        events=dict(oom=0, oom_kill=0))
        self.report = dict(passed=True, closed_cleanly=True, stage='complete', backend='simulator',
            capacity=4384, proposal_rows=31, positions=[4096, 4109], key_chunk_size=64,
            numerical_tolerances=dict(rtol=.01, atol=.01), target_integrated=False, committed_tg=None,
            sources=sources, sources_after=dict(sources), native_sources=dict(PINS), native_sources_after=dict(PINS),
            resources_before=resource, resources_after=copy.deepcopy(resource),
            eager_checks=[], replay_checks=[], input_checks=[], layout_checks=[],
            fixture_controls=[dict(name=name, case=int(name == 'frontier_update'), chip=chip, detected=True)
                for name in ('oldest', 'last_proposal', 'gap_poison', 'frontier_update') for chip in range(2)],
            stale_controls=[dict(chip=chip, detected=True) for chip in range(2)])
        for mode, cases in (('eager', (0, 1)), ('replay', (1, 0))):
            for ordinal, case in enumerate(cases):
                for chip in range(2):
                    checksum = str(case + chip * 2) * 64
                    coordinate = dict(ordinal=ordinal, case=case, chip=chip)
                    self.report[mode + '_checks'].append(dict(coordinate, passed=True,
                        numerical_close=True, failed_elements=0, max_abs=.001,
                        sha256=checksum, expected_sha256='f' * 64, replay_exact=mode == 'replay'))
                    self.report['input_checks'].extend(dict(coordinate, mode=mode, name=name, exact=True)
                                                       for name in NAMES)
                    self.report['layout_checks'].extend(dict(coordinate, mode=mode, name=name, passed=True,
                        sha256=checksum, expected_sha256=checksum) for name in ('key', 'value'))

    def test_complete_report_not_full_request_admission(self):
        result = validate(self.report)
        self.assertEqual(result['input_checks'], 48)
        self.assertFalse(result['full_request_qualified'])
        self.assertFalse(result['immutable_source_identity_verified'])

    def test_duplicate_and_missing_coverage(self):
        for field in ('eager_checks', 'replay_checks', 'input_checks', 'layout_checks',
                      'fixture_controls', 'stale_controls'):
            for duplicate in (False, True):
                report = copy.deepcopy(self.report)
                report[field].pop()
                if duplicate:
                    report[field].append(dict(report[field][0]))
                with self.subTest(field=field, duplicate=duplicate), self.assertRaises(ValueError):
                    validate(report)

    def test_replay_checksum_not_just_pass_boolean(self):
        self.report['replay_checks'][0]['sha256'] = 'e' * 64
        with self.assertRaises(ValueError):
            validate(self.report)

    def test_32_key_result_requires_explicit_policy(self):
        self.report['key_chunk_size'] = 32
        with self.assertRaises(ValueError):
            validate(self.report)
        self.assertFalse(validate(self.report, key_chunk_size=32)['full_request_qualified'])

    def test_nonfinite_error_rejected(self):
        self.report['eager_checks'][0]['max_abs'] = float('nan')
        with self.assertRaises(ValueError):
            validate(self.report)

    def test_runtime_and_resource_drift(self):
        for mutation in ('binary', 'quota', 'oom'):
            report = copy.deepcopy(self.report)
            if mutation == 'binary':
                name = next(iter(PINS))
                report['native_sources'][name] = report['native_sources_after'][name] = '0' * 64
            elif mutation == 'quota':
                report['resources_after']['cpu_quota'] = 800000
            else:
                report['resources_after']['events']['oom'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate(report)


if __name__ == '__main__':
    unittest.main()
