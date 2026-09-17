import copy
import unittest
from unittest.mock import patch

import winning_verifier_profile as profile


class WinningProfileTests(unittest.TestCase):
    def report(self):
        return dict(pp=100, committed_tg=200, request_output_limit=64, request_checks=[dict(arm='publication', length=4096,
            verifier_profile={'records': [1]}, incremental_history={'enabled': True},
            gdn_norm_prefetch={'enabled': True}, draft_tail={'enabled': True},
            fused_t16_mlp={'restored': True})])

    def test_profile_cannot_publish_throughput(self):
        report = self.report()
        with patch('frozen_ladder_requests.validate_audit') as audit:
            profile.finish_profile(report)
        audit.assert_called_once_with(report['request_checks'][0])
        self.assertIsNone(report['pp'])
        self.assertIsNone(report['committed_tg'])
        self.assertFalse(report['performance_qualified'])
        self.assertFalse(report['eligible_for_serving'])

    def test_missing_optimisation_or_wrong_context_is_rejected(self):
        for key, value in (('draft_tail', {}), ('incremental_history', {}), ('gdn_norm_prefetch', {}),
                ('fused_t16_mlp', {}), ('verifier_profile', {}), ('length', 32768)):
            report = copy.deepcopy(self.report())
            report['request_checks'][0][key] = value
            with patch('frozen_ladder_requests.validate_audit'), self.assertRaises(ValueError):
                profile.finish_profile(report)

    def test_no_request_or_failed_full_audit_is_rejected(self):
        with self.assertRaises(ValueError):
            profile.finish_profile({'request_checks': []})
        with patch('frozen_ladder_requests.validate_audit', side_effect=ValueError('audit failed')), \
                self.assertRaisesRegex(ValueError, 'audit failed'):
            profile.finish_profile(self.report())

    def test_unexpected_source_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'source set'):
            profile.adapt({'kernel.cpp': 'unchanged'})

    def test_profile_budget_is_explicit(self):
        report = self.report()
        report['request_output_limit'] = 256
        with patch('frozen_ladder_requests.validate_audit'), self.assertRaises(ValueError):
            profile.finish_profile(report)


if __name__ == '__main__':
    unittest.main()
