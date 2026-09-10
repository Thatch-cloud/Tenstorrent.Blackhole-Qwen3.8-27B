from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from dspark_request_experiment import PREREQUISITES, request_preflight, summarize


class DSparkRequestExperimentTests(unittest.TestCase):
    def requests(self):
        return [dict(instrumented_timing=audit, exact=True, state_exact=True, inactive_exact=True,
            prompt_tokens=[1, 2] * 2048, emitted=[3, 4, 5], length=4096, committed_decode_tokens=2,
            decode_ms=milliseconds, prefill_ms=1000, proposed=15, accepted=1, prefill_setup_decode_ms=2500 + milliseconds,
            feature_setup_ms=500, engine_setup_ms=1000) for audit, milliseconds in ((True, 9000), (False, 100), (False, 300))]

    def test_complete_pooled_tg_retains_slow_samples_and_excludes_instrumentation(self):
        result = summarize(self.requests())
        self.assertEqual(result['committed_tg'], 10)
        self.assertEqual(result['pp'], 4096)
        self.assertEqual(result['mean_setup_inclusive_ms'], 2700)
        self.assertEqual(result['acceptance'], 2 / 30)
        self.assertEqual(result['committed_tokens'], 4)
        self.assertFalse(result['proposal_trace'] or result['held_out_coding_quality'] or result['serving_qualified'])

    def test_missing_audit_changed_tokens_failed_state_and_nonfinite_times_reject(self):
        for field, value in (('instrumented_timing', True), ('exact', False), ('state_exact', False),
                ('inactive_exact', False), ('emitted', [3, 4, 6]), ('prompt_tokens', [9, 2]),
                ('decode_ms', float('nan')), ('prefill_ms', -1), ('feature_setup_ms', float('inf'))):
            records = self.requests()
            records[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(records)
        with self.assertRaises(ValueError):
            summarize(self.requests()[1:])

    def test_real_audited_component_artifacts_and_complete_request_source_closure_pass(self):
        result = request_preflight(Path(__file__).parent)
        self.assertEqual(result['request_prerequisites'], PREREQUISITES)
        for name in ('full_dspark_request.py', 'dspark_device.py', 'dspark_history.py', 'verifier_engine.py',
                'model_batch.py', 'gdn_multitoken.py', 'gdn_commit_dma.py', 'force_argmax.py',
                '../../speculative-decoding/harness/greedy_session.py'):
            self.assertIn(name, result['sources'])

    def test_modified_component_artifact_cannot_be_promoted(self):
        with patch('dspark_request_experiment.digest', return_value='0' * 64), self.assertRaisesRegex(ValueError, 'audited'):
            request_preflight(Path(__file__).parent)

    def test_explicit_request_ci_route_is_not_a_serving_or_runner_access_change(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load((root / '.github/workflows/qwen-experiments.yml').read_text())
        dispatch = workflow.get('on', workflow.get(True))['workflow_dispatch']
        self.assertIn('dspark-request', dispatch['inputs']['suite']['options'])
        self.assertEqual(dispatch['inputs']['suite']['default'], 'inventory')
        steps = [step for job in workflow['jobs'].values() for step in job.get('steps', [])
            if step.get('run', '').endswith('bash scripts/ci/run-dspark-hardware.sh')]
        self.assertEqual(len(steps), 1)
        self.assertIn("inputs.suite == 'dspark-request'", steps[0]['if'])
        self.assertIn("&& 'request'", steps[0]['env']['QWEN_DSPARK_MODE'])
        runner = (root / 'scripts/ci/run-dspark-hardware.sh').read_text()
        suite = (root / 'scripts/ci/dspark-hardware-suite.sh').read_text()
        self.assertIn('docker cp speculative-decoding "$test_id:/speculative-decoding"', runner)
        self.assertIn('request_options=(--request)', suite)
        self.assertIn('report_name=dspark-request-hardware', suite)
        self.assertIn('Failed to discover available ethernet links', suite)


if __name__ == '__main__':
    unittest.main()
