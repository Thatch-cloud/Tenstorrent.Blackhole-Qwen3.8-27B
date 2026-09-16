from copy import deepcopy
import unittest

from matched_score_request import validate_score_execution


class MatchedScoreTests(unittest.TestCase):
    def test_requires_actual_execution_in_both_clean_requests(self):
        records = [dict(restored=True, calls=2), dict(restored=True, calls=2)]
        report = dict(passed=True, full_request_passed=True, closed_cleanly=True,
            request_checks=[dict(score_64k_reintegration=record) for record in records])
        validate_score_execution(report, records)
        for field in ('passed', 'full_request_passed', 'closed_cleanly'):
            with self.assertRaises(ValueError):
                validate_score_execution(dict(report, **{field: False}), records)
        altered = deepcopy(records)
        altered[1]['calls'] = 0
        with self.assertRaises(ValueError):
            validate_score_execution(report, altered)
        with self.assertRaises(ValueError):
            validate_score_execution(report, records[:1])


if __name__ == '__main__':
    unittest.main()
