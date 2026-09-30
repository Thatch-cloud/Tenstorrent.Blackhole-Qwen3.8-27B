import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dspark_intake import TAPS
from t32_timing import authorize, validate, validate_timed


class TimingTests(unittest.TestCase):
    def fixture(self):
        checks = [dict(position=4096, tensors=6, exact=True)] * 2
        features = [dict(position=position, rows=1, tap=tap, chip=chip, exact=True)
            for position in (4096, 4097) for tap in TAPS for chip in (0, 1)]
        request = dict(exact=True, state_exact=True, inactive_exact=True, instrumented_timing=True,
            length=4096, prompt_tokens=[1] * 4096, emitted=[1, 2, 3], max_new_tokens=3,
            committed_tokens_per_second=None,
            blocks=[dict(position=4096, rows=32, committed=1), dict(position=4097, rows=1, committed=1)],
            gdn_verify_checks=[dict(position=4096, rows=32, unchanged=True)],
            dspark=dict(proposals=31, verifier_rows=32, t32_lifecycle=dict(hardware_audit_experiment=True),
                feature_checks=features, proposal_checks=copy.deepcopy(checks)),
            fused_t32_mlp=dict(rows=32, hits=[1] * 64, restored=True, native_bindings_unchanged=True))
        return request, checks

    def test_full_audit_required_before_timing_and_output_must_remain_identical(self):
        request, checks = self.fixture()
        state = SimpleNamespace(timing_reference=None, record=dict(native_proposal_checks=checks))
        timed = dict(request, instrumented_timing=False, committed_tokens_per_second=123.0)
        with patch('t32_score_hardware.require_active', return_value=state):
            with self.assertRaises(ValueError):
                validate_timed(timed)
            authorize(request)
            validate_timed(timed)
            with self.assertRaises(ValueError):
                authorize(request)
            with self.assertRaises(ValueError):
                validate_timed(dict(timed, emitted=[1, 2, 4]))
            with self.assertRaises(ValueError):
                validate_timed(dict(timed, state_exact=False))

    def test_missing_native_replay_feature_and_commit_evidence_rejected(self):
        request, checks = self.fixture()
        validate(request, checks)
        with self.assertRaises(ValueError):
            validate(request, checks[:-1])
        for field in ('proposal_checks', 'feature_checks'):
            changed = copy.deepcopy(request)
            changed['dspark'][field].pop()
            with self.assertRaises(ValueError):
                validate(changed, checks)
        changed = copy.deepcopy(request)
        changed['gdn_verify_checks'] = []
        with self.assertRaises(ValueError):
            validate(changed, checks)
        changed = copy.deepcopy(request)
        changed['fused_t32_mlp']['hits'][0] = 0
        with self.assertRaises(ValueError):
            validate(changed, checks)
