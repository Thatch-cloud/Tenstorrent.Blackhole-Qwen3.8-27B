from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlp_compute_clock_experiment as experiment
from mlp_compute_clock_combined_stage import adapt


class ClockExperimentTests(unittest.TestCase):
    def test_one_audited_request_preserves_existing_route_and_releases_after_scope(self):
        events = []
        model = SimpleNamespace(mesh_device=object())
        report = {'streams': 1}
        options = dict(prompt=[1] * 4096, captured_publication=True)
        required = {name: True for name in ('audit_features', 'fused_t16_mlp', 'gdn_shared_qk',
            'captured_publication', 'target_attention_t16', 'commit_only_gdn', 'proposal_trace', 'native_attention')}
        def original_measure(*args, **kwargs):
            self.assertEqual(kwargs, required)
            events.append('measure')
            return dict(instrumented_timing=True, committed_tokens_per_second=None)
        full = SimpleNamespace(measure_dspark_request=original_measure)
        def audit(value):
            events.append('audit')
            self.assertTrue(value['instrumented_timing'])
        original_finish = lambda *args: self.fail('Timed summary must not run')
        ladder = SimpleNamespace(finish=original_finish, validate_audit=audit)
        def run(*args, **kwargs):
            self.assertEqual(kwargs, options)
            result = full.measure_dspark_request(**required)
            result['arm'] = 'publication'
            report['request_checks'] = [result]
            ladder.finish(report, None)
        class Bank:
            def __init__(self, target, captures, candidate):
                self.assertions = len(captures)
                self.active = False
            @contextmanager
            def install(self, *args):
                self.active = True
                events.append('install')
                try:
                    yield self
                finally:
                    self.active = False
                    events.append('restore')
            def assert_releasable(self):
                if self.active:
                    raise ValueError('scope active')
                events.append('releasable')
            def summary(self):
                self.assert_releasable()
                return dict(layers=self.assertions, committed_tg=None)
        def capture(operations, mesh, owned):
            value = object()
            owned.append(value)
            return value
        modules = dict(dspark_request_experiment=SimpleNamespace(run_loaded_requests=run),
            frozen_ladder_requests=ladder, full_dspark_request=full, fused_t16_scope=SimpleNamespace(),
            verifier_engine=SimpleNamespace(VerifierEngine=object),
            gdn_multitoken_conv=SimpleNamespace(release_owned=lambda operations, owned: events.append(('release', len(owned)))))
        environment = {name: '1' for name in ('QWEN_MLP_COMPUTE_CLOCK_COMBINED', 'QWEN_FROZEN_COMBINED_RUNTIME',
            'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS')}
        with patch.dict('sys.modules', modules), patch.dict('os.environ', environment, clear=True), \
                patch.object(experiment, 'candidate_class', return_value=object()), \
                patch.object(experiment, 'ComputeClockCapture', capture), patch.object(experiment, 'CombinedComputeClockCapture', Bank):
            experiment.run_loaded_requests(None, None, model, None, None, None, None, None, None, None, None,
                None, report, lambda stage: events.append(stage), **options)
        self.assertEqual(report['mlp_compute_clock_combined']['layers'], 64)
        self.assertIsNone(report['committed_tg'])
        self.assertIsNone(report['pp'])
        self.assertEqual(events.count('measure'), 1)
        self.assertEqual(events[-1], ('release', 64))
        self.assertIs(full.measure_dspark_request, original_measure)
        self.assertIs(ladder.finish, original_finish)
        self.assertEqual(report['mlp_compute_clock_sources'], report['mlp_compute_clock_sources_after'])

    def test_staging_only_replaces_request_schedule_and_entrypoint(self):
        sources = {
            'dspark_request_experiment.py': "def run():\n    schedule = (('publication', True), ('publication', False), ('publication', False))\n",
            'dspark-target-hardware.py': 'def main():\n    if True:\n        if True:\n            from dspark_request_experiment import run_loaded_requests\n',
            'run-dspark-hardware.sh': 'docker create \\\n    -e "QWEN_DSPARK_MODE=$mode"\n',
        }
        result = adapt(sources)
        self.assertIn("schedule = (('publication', True),)", result['dspark_request_experiment.py'])
        self.assertIn('from mlp_compute_clock_experiment import run_loaded_requests', result['dspark-target-hardware.py'])
        self.assertIn('QWEN_MLP_COMPUTE_CLOCK_COMBINED', result['run-dspark-hardware.sh'])
        with self.assertRaises(ValueError):
            adapt(result)
