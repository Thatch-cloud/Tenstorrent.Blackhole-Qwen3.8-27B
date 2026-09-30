from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cumulative_t16_experiment as experiment
import cumulative_register_runtime as runtime
import cumulative_norm_runtime as normalization
from cumulative_norm_validation import HISTORY_SHA256
from test_cumulative_fusion_validation import fixture, BASELINE_SHA256, REGISTER_SHA256


class RegisterDriverTests(unittest.TestCase):
    def exercise(self, *, with_norm=False, fail=False):
        active = []
        original_build, original_gate = Mock(return_value=(1, 2, 3)), Mock()
        shared = SimpleNamespace(build=original_build)
        gate = SimpleNamespace(qualify=original_gate)
        prefetch, scatter = Mock(return_value=(1, 2, 3)), Mock(return_value=(1, 2, 3))

        @contextmanager
        def register(*args, **kwargs):
            audit = dict(report_sha256=REGISTER_SHA256, constructions=64, calls=128, restored=False)
            active.append(True)
            try:
                yield audit
            finally:
                active.pop()
                audit['restored'] = True

        @contextmanager
        def combined(*args, **kwargs):
            audit = dict(direct=dict(hits=96, restored=False),
                compact=dict(calls=2, steps=30, restored=False), down=dict(hits=[2] * 64, restored=False))
            try:
                yield audit
            finally:
                for component in audit.values():
                    component['restored'] = True

        def measure(**options):
            if fail and active:
                raise RuntimeError('combined request failed')
            result = fixture(bool(active))
            del result['register_epilogue']
            result.update(gdn_shared_qk=dict(loads=[{}] * 96), score_layout=dict(calls=2))
            if with_norm:
                for layer in range(96):
                    shared.build(None, None, [])
                result['gdn_shared_qk']['admission'] = gate.qualify()
                result['incremental_history'] = dict(enabled=True, report_sha256=HISTORY_SHA256)
            return result

        full = SimpleNamespace(measure_dspark_request=measure)
        fusion = SimpleNamespace(REPORT_SHA256=BASELINE_SHA256)

        def route(request, arm):
            self.assertEqual(request['fused_t16_mlp']['passed_simulator'], fusion.REPORT_SHA256)
            if with_norm:
                self.assertEqual(request['gdn_shared_qk']['admission']['report_sha256'], variants.REPORT_SHA256)

        variants = SimpleNamespace(validate_route=route, REPORT_SHA256='original-shared-admission')
        report = {}
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in experiment.FILES + experiment.REGISTER_PAYLOADS:
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')

            def run(*args, **kwargs):
                measure_request = full.measure_dspark_request
                requests = report.setdefault('request_checks', [])
                for enabled, audited in experiment.direct_experiment.SCHEDULE:
                    with experiment.direct_experiment.scoped_direct_windows({}, directory) if enabled else nullcontext():
                        result = measure_request(gdn_shared_qk=True)
                    result.update(gdn_direct_window=dict(direct=enabled), instrumented_timing=audited)
                    variants.validate_route(result, 'publication')
                    requests.append(result)
                report['gdn_direct_window_comparison'] = dict(improvement_screen_passed=True,
                    pairs=[dict(unchanged_tg=120, direct_tg=130)] * 2)

            with patch.dict('sys.modules', dict(full_dspark_request=full,
                    gdn_shared_qk_variants=variants, dspark_fusion_variants=fusion,
                    gdn_shared_qk_scope=shared, gdn_shared_qk_gate=gate)), \
                    patch.dict('os.environ', dict(QWEN_CUMULATIVE_T16='1', QWEN_COMPACT_SCORE_HARDWARE='1',
                        QWEN_CUMULATIVE_REGISTER='1', QWEN_CUMULATIVE_NORM='1' if with_norm else '0',
                        QWEN_CUMULATIVE_MLP_DOWN='1', TT_METAL_HOME='native')), \
                    patch.object(normalization, 'qualify_prefetch', return_value=dict(report_sha256=normalization.PREFETCH_SHA256)), \
                    patch.object(normalization, 'qualify_scatter', return_value=dict(report_sha256=normalization.SCATTER_SHA256)), \
                    patch.object(normalization, 'prefetch_build', prefetch), \
                    patch.object(normalization, 'scatter_build', scatter), \
                    patch.object(experiment, '__file__', str(directory / 'cumulative_t16_experiment.py')), \
                    patch.object(experiment, 'qualify', return_value={}), \
                    patch.object(experiment, 'qualify_down', return_value={}), \
                    patch.object(experiment, 'scoped_cumulative_t16', combined), \
                    patch.object(runtime, 'scoped_register_epilogue', register), \
                    patch.object(experiment.direct_experiment, 'run_loaded_requests', run):
                try:
                    with normalization.normalization_scope(directory) if with_norm else nullcontext():
                        experiment.run_loaded_requests(*([None] * 12), report, lambda stage: None)
                finally:
                    self.assertIs(full.measure_dspark_request, measure)
                    self.assertIs(variants.validate_route, route)
                    self.assertIs(shared.build, original_build)
                    self.assertIs(gate.qualify, original_gate)
                    self.assertEqual(variants.REPORT_SHA256, 'original-shared-admission')
                    self.assertEqual(fusion.REPORT_SHA256, BASELINE_SHA256)
                    self.assertFalse(normalization._active.get())
                    self.assertIsNone(normalization._selected.get())
                    self.assertFalse(active)
        self.assertEqual(report['cumulative_components'],
            ['direct_windows', 'compact_scores', 'wider_mlp_down']
            + (['norm_scatter'] if with_norm else []) + ['register_epilogue'])
        self.assertEqual([value['register_epilogue']['register_resident'] for value in report['request_checks']],
            [False, True, False, True, True, False])
        self.assertIs(full.measure_dspark_request, measure)
        self.assertIs(variants.validate_route, route)
        self.assertFalse(active)
        if with_norm:
            self.assertEqual(prefetch.call_count, 288)
            self.assertEqual(scatter.call_count, 288)
            self.assertEqual([value['norm_reader']['policy'] for value in report['request_checks']],
                ['prefetch', 'scatter', 'prefetch', 'scatter', 'scatter', 'prefetch'])

    def test_candidate_identity_exists_before_combined_finish_validation(self):
        self.exercise()

    def test_all_five_routes_preserve_both_admissions_and_controls(self):
        self.exercise(with_norm=True)

    def test_failed_five_component_request_restores_both_wrappers(self):
        with self.assertRaisesRegex(RuntimeError, 'combined request failed'):
            self.exercise(with_norm=True, fail=True)
