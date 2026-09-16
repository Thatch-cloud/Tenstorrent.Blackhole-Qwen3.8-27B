from copy import deepcopy
import unittest

from frozen_gdn_norm_gate import validate_report


class NormGateTests(unittest.TestCase):
    def fixture(self):
        report = dict(passed=True, closed_cleanly=True, norm_unchanged=True,
            state_math_unchanged=True, shared_qk_preparation=True, norm_bridge_prefetch=True,
            backend='simulator', rows=16, norm_staging_bytes=8192, sources={}, sources_after={})
        for field, operands in (('checks', 3), ('immutable_checks', 6)):
            report[field] = [dict(mode=mode, operand=operand, chip=chip, exact=True)
                for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
                for operand in range(operands) for chip in (0, 1)]
        return report

    def test_complete_matrix(self):
        validate_report(self.fixture())

    def test_missing_or_failed_check_rejected(self):
        for field in ('checks', 'immutable_checks'):
            for mutation in ('missing', 'failed', 'duplicate'):
                report = self.fixture()
                if mutation == 'missing':
                    report[field].pop()
                elif mutation == 'failed':
                    report[field][0]['exact'] = False
                else:
                    report[field][-1] = deepcopy(report[field][0])
                with self.subTest(field=field, mutation=mutation), self.assertRaises(ValueError):
                    validate_report(report)

    def test_wrong_variant_or_sources_rejected(self):
        for field, value in (('norm_bridge_prefetch', False), ('rows', 32),
                ('norm_staging_bytes', 4096), ('sources_after', {'changed': 'hash'})):
            report = self.fixture()
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_report(report)


if __name__ == '__main__':
    unittest.main()
