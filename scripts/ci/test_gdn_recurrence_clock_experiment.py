from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_recurrence_clock_experiment as experiment


class RecurrenceExperimentTests(unittest.TestCase):
    def test_single_audit_keeps_winning_options_and_releases_after_close(self):
        events = []
        model = SimpleNamespace(mesh_device=object())
        report = dict(streams=1)
        options = dict(prompt=[1] * 4096, captured_publication=True)
        required = {name: True for name in ('audit_features', 'fused_t16_mlp', 'gdn_shared_qk',
            'captured_publication', 'target_attention_t16', 'commit_only_gdn', 'proposal_trace',
            'native_attention', 'score_layout')}
        def measure(*args, **kwargs):
            self.assertEqual(kwargs, required)
            events.append('measure')
            return dict(instrumented_timing=True, committed_tokens_per_second=None,
                gdn_norm_prefetch=dict(enabled=True, builds=96))
        full = SimpleNamespace(measure_dspark_request=measure)
        def audit(request):
            self.assertTrue(request['instrumented_timing'])
            events.append('audit')
        ladder = SimpleNamespace(finish=lambda *args: self.fail('Timed finish must not run'), validate_audit=audit)
        def run(*args, **kwargs):
            self.assertEqual(options, kwargs)
            request = full.measure_dspark_request(**required)
            request['arm'] = 'publication'
            report['request_checks'] = [request]
            ladder.finish(report, None)
        class Bank:
            def __init__(self, operations, model, captures, candidate, admission):
                self.active = False
                self.builds = [None] * 96
                if len(captures) != 48:
                    raise ValueError('All GDN layers required')
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
                    raise ValueError('still active')
            def summary(self):
                self.assert_releasable()
                return dict(layers=48, committed_tg=None)
        def capture(operations, mesh, owned):
            value = object()
            owned.append(value)
            return value
        def release(operations, owned):
            self.assertEqual(len(owned), 48)
            self.assertIn('restore', events)
            events.append('release')
        modules = dict(dspark_request_experiment=SimpleNamespace(run_loaded_requests=run),
            full_dspark_request=full, frozen_ladder_requests=ladder,
            gdn_shared_qk_pipeline=SimpleNamespace(), gdn_recurrence_clock_pipeline=SimpleNamespace(),
            verifier_engine=SimpleNamespace(VerifierEngine=object),
            gdn_multitoken_conv=SimpleNamespace(release_owned=release))
        environment = {name: '1' for name in ('QWEN_GDN_RECURRENCE_CLOCK_COMBINED',
            'QWEN_FROZEN_COMBINED_RUNTIME', 'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS')}
        environment['TT_METAL_HOME'] = 'runtime'
        with patch.dict('sys.modules', modules), patch.dict('os.environ', environment, clear=True), \
                patch.object(experiment, 'qualify', return_value={}), \
                patch.object(experiment, 'FILES', ()), \
                patch.object(experiment, 'RecurrenceClockCapture', capture), \
                patch.object(experiment, 'CombinedRecurrenceCapture', Bank):
            experiment.run_loaded_requests(None, None, model, None, None, None, None, None,
                None, None, None, None, report, lambda message: events.append('allocate'), **options)
        self.assertEqual(events, ['allocate', 'install', 'measure', 'restore', 'audit', 'release'])
        self.assertIs(full.measure_dspark_request, measure)
        self.assertIsNone(report['committed_tg'])


if __name__ == '__main__':
    unittest.main()
