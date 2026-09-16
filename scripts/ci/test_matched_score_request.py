from copy import deepcopy
from contextlib import contextmanager
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dspark_64k_mlp_timed as mlp
from dspark_64k_score_timed import measurement_scope as score_scope
from dspark_score_pair import paired_scope

from matched_score_request import validate_pair_execution, validate_score_execution


class MatchedScoreTests(unittest.TestCase):
    def test_pair_composes_score_capture_with_mlp_and_restores_hooks(self):
        active = dict(mlp=False, score=False)
        captures, records, pairs = [], [], []
        owner = self

        class Device:
            def prepare_trace(self, anchor, *, audit=False):
                owner.assertTrue(active['mlp'])
                captures.append(dict(active))

        class ScoreArm:
            def __init__(self, device, *, hardware_audit):
                self.restored = False

            @contextmanager
            def install(self):
                active['score'] = True
                try:
                    yield
                finally:
                    active['score'] = False
                    self.restored = True

            def summary(self):
                return dict(restored=self.restored, calls=1)

        class MlpArm:
            def __init__(self, operations, model, collective):
                self.audit = dict(unit_fixture=True)

            @contextmanager
            def install(self):
                active['mlp'] = True
                try:
                    yield
                finally:
                    active['mlp'] = False

        def measure(*, prompt, operations, model, predecessor, successor,
                audit_features, max_new_tokens, proposal_trace, captured_publication,
                target_attention_t16, commit_only_gdn, native_attention):
            Device().prepare_trace(object())
            return dict(prompt_tokens=prompt, committed_decode_tokens=135,
                decode_ms=3000, prefill_ms=1000, exact=True,
                state_exact=True, inactive_exact=True)

        module = SimpleNamespace(measure_dspark_request=measure)
        hardware_audit = Mock(return_value=dict(unit_fixture=True))
        original_prepare = Device.prepare_trace
        environment = dict(QWEN_64K_SCORE_TIMED='1', QWEN_64K_SHARED_QK_TIMED='1',
            QWEN_DSPARK_SFPU_TIMED='1', QWEN_DSPARK_SFPU_REQUEST_SCREEN='0',
            QWEN_64K_MLP_TIMED='1', QWEN_64K_MLP_AUDIT='0')
        arguments = dict(prompt=[0] * 65536, operations=object(),
            model=SimpleNamespace(mesh_device=object()), predecessor=object(), successor=object(),
            audit_features=False, max_new_tokens=256, proposal_trace=True,
            captured_publication=True, target_attention_t16=True,
            commit_only_gdn=True, native_attention=True)

        def candidate(isolated):
            return score_scope(isolated, Device, ScoreArm, hardware_audit, records)

        with patch.dict(os.environ, environment), patch.object(mlp.splitk, 'require_timed'), \
                patch.object(mlp, 'validate_result') as validate:
            with paired_scope(module, candidate, pairs, lambda record: None):
                with mlp.measurement_scope(module, MlpArm, object()):
                    control = module.measure_dspark_request(**arguments)
                    fused = module.measure_dspark_request(**arguments)
            self.assertEqual(validate.call_count, 2)
        hardware_audit.assert_called_once()
        self.assertEqual(captures, [dict(mlp=True, score=False), dict(mlp=True, score=True)])
        self.assertNotIn('score_64k_reintegration', control)
        self.assertEqual(fused['score_64k_reintegration'], records[0])
        self.assertTrue(records[0]['restored'])
        self.assertEqual(active, dict(mlp=False, score=False))
        self.assertIs(Device.prepare_trace, original_prepare)
        self.assertIs(module.measure_dspark_request, measure)
        self.assertIs(pairs[0]['request'], control)
        self.assertIs(pairs[1]['request'], fused)

    def test_pair_requires_distinct_actual_paths(self):
        record = dict(restored=True, calls=2)
        requests = [{}, dict(score_64k_reintegration=record)]
        pairs = [dict(summary=dict(arm=arm), request=request)
            for arm, request in zip(('control', 'score_layout'), requests)]
        report = dict(passed=True, full_request_passed=True, closed_cleanly=True, request_checks=requests)
        validate_pair_execution(report, [record], pairs)
        requests[0]['score_64k_reintegration'] = record
        with self.assertRaises(ValueError):
            validate_pair_execution(report, [record], pairs)

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
