from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cumulative_t16_experiment as experiment


class CumulativeExperimentTests(unittest.TestCase):
    def exercise(self, missing_calls=False, fail_request=False, with_down=False, missing_down=False):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in experiment.FILES:
                (directory / name).write_text('fixture')
            active, restored = [], []

            @contextmanager
            def scope(direct, compact, source_directory, *, down_admission=None):
                audit = dict(direct=dict(hits=96, restored=False),
                             compact=dict(calls=0 if missing_calls else 2, steps=30, restored=False))
                self.assertEqual(down_admission, {'qualified': True} if with_down else None)
                if with_down:
                    audit['down'] = dict(hits=[0 if missing_down else 2] * 64, restored=False)
                active.append(True)
                try:
                    yield audit
                finally:
                    active.pop()
                    for value in audit.values():
                        value['restored'] = True
                    restored.append(True)

            report = {}
            original_scope = experiment.direct_experiment.scoped_direct_windows

            def run(*arguments, **options):
                requests = report.setdefault('request_checks', [])
                for enabled, audited in experiment.direct_experiment.SCHEDULE:
                    if enabled:
                        with experiment.direct_experiment.scoped_direct_windows({}, directory):
                            self.assertTrue(active)
                            if fail_request:
                                raise RuntimeError('request failed')
                    self.assertFalse(active)
                    requests.append(dict(gdn_direct_window=dict(direct=enabled), instrumented_timing=audited,
                        gdn_shared_qk=dict(loads=[{}] * 96), score_layout=dict(calls=2),
                        fused_t16_mlp=dict(hits=[2] * 64)))
                report['gdn_direct_window_comparison'] = dict(improvement_screen_passed=True,
                    pairs=[dict(unchanged_tg=120, direct_tg=130)] * 2)

            with patch.dict('os.environ', {'QWEN_CUMULATIVE_T16': '1', 'QWEN_COMPACT_SCORE_HARDWARE': '1',
                    'QWEN_CUMULATIVE_MLP_DOWN': '1' if with_down else '0'}), \
                    patch.object(experiment, '__file__', str(directory / 'cumulative_t16_experiment.py')), \
                    patch.object(experiment, 'qualify', return_value={}), \
                    patch.object(experiment, 'qualify_down', return_value={'qualified': True}), \
                    patch.object(experiment, 'scoped_cumulative_t16', scope), \
                    patch.object(experiment.direct_experiment, 'run_loaded_requests', run):
                try:
                    experiment.run_loaded_requests(*([None] * 12), report, lambda stage: None)
                finally:
                    self.assertFalse(active)
                    self.assertIs(experiment.direct_experiment.scoped_direct_windows, original_scope)
            self.assertEqual(restored, [True] * 3)
            return report

    def test_same_three_candidates_enable_both_components(self):
        report = self.exercise()
        self.assertTrue(report['cumulative_t16'])
        self.assertEqual([request['compact_score']['compact'] for request in report['request_checks']],
                         [False, True, False, True, True, False])
        self.assertEqual(report['cumulative_sources'], report['cumulative_sources_after'])

    def test_missing_compact_execution_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Both cumulative components'):
            self.exercise(missing_calls=True)

    def test_three_component_stack_keeps_native_controls(self):
        report = self.exercise(with_down=True)
        self.assertEqual(report['cumulative_components'], ['direct_windows', 'compact_scores', 'wider_mlp_down'])
        for request in report['request_checks']:
            enabled = request['compact_score']['compact']
            self.assertIs(request['mlp_down_grid']['wider_down'], enabled)
            self.assertEqual(request['mlp_down_grid']['hits'], [2 if enabled else 0] * 64)

    def test_missing_down_execution_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Every fused MLP layer'):
            self.exercise(with_down=True, missing_down=True)

    def test_request_failure_restores_driver(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.exercise(fail_request=True)
