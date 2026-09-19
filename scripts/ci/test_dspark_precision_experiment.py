from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dspark_precision_experiment as experiment
from dspark_precision_comparison import SCHEDULE
from dspark_request_experiment import summarize
from test_dspark_precision_comparison import fixture


class PrecisionExperimentTests(unittest.TestCase):
    def exercise(self, *, fail_candidate=False):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in experiment.SOURCES:
                (directory / name).write_text('{}' if name.endswith('.json') else 'source')
            baseline = object()
            prepared = SimpleNamespace(TracedDSparkDevice=baseline)
            devices, calls = [], []

            class Candidate:
                def __init__(self):
                    self.closed = False
                    self.prepared = SimpleNamespace(closed=False)
                    self.precision_layer_calls = 5
                    devices.append(self)

                def close(self):
                    self.prepared.closed = True
                    self.closed = True

            requests = fixture()

            def measure(*args, **kwargs):
                precision, audit = SCHEDULE[len(calls)]
                calls.append((precision, audit))
                if precision == 'hifi2':
                    self.assertIsNot(prepared.TracedDSparkDevice, baseline)
                    device = prepared.TracedDSparkDevice()
                    if fail_candidate:
                        raise RuntimeError('proposal failure')
                    device.close()
                else:
                    self.assertIs(prepared.TracedDSparkDevice, baseline)
                return requests[len(calls) - 1]

            full = SimpleNamespace(measure_dspark_request=measure)
            original_finish = lambda *args: self.fail('Original single-arm finish ran')
            ladder = SimpleNamespace(finish=original_finish, validate_audit=lambda value: None)
            report = dict(streams=1)

            def run(*args, **kwargs):
                report['request_checks'] = []
                for precision, audit in SCHEDULE:
                    flags = {name: True for name in ('gdn_shared_qk', 'fused_t16_mlp', 'captured_publication',
                        'target_attention_t16', 'commit_only_gdn', 'proposal_trace', 'native_attention', 'score_layout')}
                    report['request_checks'].append(full.measure_dspark_request(audit_features=audit, **flags))
                    self.assertIs(prepared.TracedDSparkDevice, baseline)
                ladder.finish(report, summarize)

            modules = dict(dspark_request_experiment=SimpleNamespace(run_loaded_requests=run),
                frozen_ladder_requests=ladder, full_dspark_request=full, dspark_prepared_proposal=prepared,
                dspark_native_cached_layer=SimpleNamespace(execute=lambda: None),
                gdn_shared_qk_variants=SimpleNamespace(validate_route=lambda value, arm: None))
            environment = {name: '1' for name in ('QWEN_DSPARK_PROJECTION_HIFI2',
                'QWEN_FROZEN_COMBINED_RUNTIME', 'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS')}
            with patch.dict('sys.modules', modules), patch.dict('os.environ', environment, clear=True), \
                    patch.object(experiment, '__file__', str(directory / 'dspark_precision_experiment.py')), \
                    patch.object(experiment, 'qualify', return_value=dict(component_execution_passed=True)), \
                    patch.object(experiment, 'device_type', return_value=Candidate):
                def invoke():
                    experiment.run_loaded_requests(None, None, None, None, None, None, None, None,
                        None, None, None, None, report, lambda stage: None,
                        prompt=[1] * 4096, captured_publication=True)
                if fail_candidate:
                    with self.assertRaisesRegex(RuntimeError, 'proposal failure'):
                        invoke()
                else:
                    invoke()
            self.assertIs(prepared.TracedDSparkDevice, baseline)
            self.assertIs(full.measure_dspark_request, measure)
            self.assertIs(ladder.finish, original_finish)
            self.assertTrue(all(device.closed and device.prepared.closed for device in devices))
            self.assertEqual(report['drafter_precision_sources'], report['drafter_precision_sources_after'])
            if not fail_candidate:
                self.assertEqual(calls, list(SCHEDULE))
                self.assertEqual(report['drafter_precision_comparison']['arms']['hifi2']['committed_tg'], 120)
                self.assertEqual(len(devices), 3)

    def test_combined_abba_uses_candidate_only_in_its_own_arm(self):
        self.exercise()

    def test_candidate_failure_closes_owned_device_and_restores_routes(self):
        self.exercise(fail_candidate=True)
