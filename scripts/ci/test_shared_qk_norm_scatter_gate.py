from copy import deepcopy
import unittest

from shared_qk_norm_scatter_gate import validate_report


class ScatterGateTests(unittest.TestCase):
    def fixture(self):
        report = dict(passed=True, closed_cleanly=True, stage='complete', backend='simulator',
            rows=16, norm_direct_read_bytes=8192, norm_unchanged=True,
            state_math_unchanged=True, shared_qk_preparation=True, norm_bridge_scatter=True,
            hardware_qualified=False, timing_qualified=False,
            sources={'source': 'hash'}, sources_after={'source': 'hash'})
        for field, operands in (('checks', 3), ('immutable_checks', 6)):
            report[field] = [dict(mode=mode, operand=operand, chip=chip, exact=True)
                for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
                for operand in range(operands) for chip in (0, 1)]
        return report

    def test_matrix(self):
        validate_report(self.fixture())
        for field in ('checks', 'immutable_checks'):
            for mutation in ('missing', 'duplicate', 'failed'):
                report = self.fixture()
                if mutation == 'missing':
                    report[field].pop()
                elif mutation == 'duplicate':
                    report[field][-1] = deepcopy(report[field][0])
                else:
                    report[field][0]['exact'] = False
                with self.subTest(field=field, mutation=mutation), self.assertRaises(ValueError):
                    validate_report(report)

    def test_incomplete_or_wrong_variant(self):
        for field, value in (('norm_bridge_scatter', False), ('rows', 32),
                ('stage', 'capture'), ('norm_direct_read_bytes', 4096),
                ('sources', {}), ('sources_after', {'different': 'hash'}),
                ('numerical_failures', ['failure']), ('timing_qualified', True)):
            report = self.fixture()
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_report(report)


if __name__ == '__main__':
    unittest.main()
