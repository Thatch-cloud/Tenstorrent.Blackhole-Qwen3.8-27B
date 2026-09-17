import copy
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from frozen_context_geometry import CONTEXTS
from frozen_ladder_requests import finish
import test_frozen_request_phases


class LadderRequestTests(unittest.TestCase):
    def fixture(self, context):
        audit = test_frozen_request_phases.RequestPhaseTests().requests()[1]
        audit.update(length=context, prompt_tokens=[1] * context)
        audit['blocks'][0]['position'] = context
        audit['gdn_verify_checks'][0]['position'] = context
        for check in audit['dspark']['proposal_checks']:
            check['position'] = context
        return dict(request_checks=[audit, dict(copy.deepcopy(audit), instrumented_timing=False),
            dict(copy.deepcopy(audit), instrumented_timing=False)])

    def test_all_contexts_require_fresh_audit_and_route_validation(self):
        for context in CONTEXTS:
            report = self.fixture(context)
            route = Mock()
            summary = Mock(return_value=dict(ctx=context, pp=1, committed_tg=2))
            with patch.dict(os.environ, QWEN_DSPARK_REQUEST_CONTEXT=str(context)), patch.dict(
                    sys.modules, gdn_shared_qk_variants=SimpleNamespace(validate_route=route)):
                finish(report, summary)
            self.assertEqual(route.call_count, 3)
            summary.assert_called_once_with(report['request_checks'])
            self.assertTrue(report['fresh_context_audit'])

    def test_missing_feature_audit_and_changed_proposals_rejected(self):
        for field, value in (('gdn_verify_checks', []), ('blocks', []), ('instrumented_timing', False),
                ('length', 8192), ('arm', 'control')):
            report = self.fixture(32768)
            report['request_checks'][0][field] = value
            with patch.dict(os.environ, QWEN_DSPARK_REQUEST_CONTEXT='32768'), patch.dict(
                    sys.modules, gdn_shared_qk_variants=SimpleNamespace(validate_route=lambda *args: None)):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    finish(report, Mock())


if __name__ == '__main__':
    unittest.main()
