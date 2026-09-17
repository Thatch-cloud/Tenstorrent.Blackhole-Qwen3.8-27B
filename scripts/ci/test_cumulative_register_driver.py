from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cumulative_t16_experiment as experiment
import cumulative_register_runtime as runtime
from test_cumulative_fusion_validation import fixture, BASELINE_SHA256, REGISTER_SHA256


class RegisterDriverTests(unittest.TestCase):
    def test_candidate_identity_exists_before_combined_finish_validation(self):
        active = []

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

        def measure():
            result = fixture(bool(active))
            del result['register_epilogue']
            result.update(gdn_shared_qk=dict(loads=[{}] * 96), score_layout=dict(calls=2))
            return result

        full = SimpleNamespace(measure_dspark_request=measure)
        fusion = SimpleNamespace(REPORT_SHA256=BASELINE_SHA256)

        def route(request, arm):
            self.assertEqual(request['fused_t16_mlp']['passed_simulator'], fusion.REPORT_SHA256)

        variants = SimpleNamespace(validate_route=route)
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
                        result = measure_request()
                    result.update(gdn_direct_window=dict(direct=enabled), instrumented_timing=audited)
                    variants.validate_route(result, 'publication')
                    requests.append(result)
                report['gdn_direct_window_comparison'] = dict(improvement_screen_passed=True,
                    pairs=[dict(unchanged_tg=120, direct_tg=130)] * 2)

            with patch.dict('sys.modules', dict(full_dspark_request=full,
                    gdn_shared_qk_variants=variants, dspark_fusion_variants=fusion)), \
                    patch.dict('os.environ', dict(QWEN_CUMULATIVE_T16='1', QWEN_COMPACT_SCORE_HARDWARE='1',
                        QWEN_CUMULATIVE_REGISTER='1', QWEN_CUMULATIVE_NORM='0', QWEN_CUMULATIVE_MLP_DOWN='1')), \
                    patch.object(experiment, '__file__', str(directory / 'cumulative_t16_experiment.py')), \
                    patch.object(experiment, 'qualify', return_value={}), \
                    patch.object(experiment, 'qualify_down', return_value={}), \
                    patch.object(experiment, 'scoped_cumulative_t16', combined), \
                    patch.object(runtime, 'scoped_register_epilogue', register), \
                    patch.object(experiment.direct_experiment, 'run_loaded_requests', run):
                experiment.run_loaded_requests(*([None] * 12), report, lambda stage: None)
        self.assertEqual(report['cumulative_components'],
            ['direct_windows', 'compact_scores', 'wider_mlp_down', 'register_epilogue'])
        self.assertEqual([value['register_epilogue']['register_resident'] for value in report['request_checks']],
            [False, True, False, True, True, False])
        self.assertIs(full.measure_dspark_request, measure)
        self.assertIs(variants.validate_route, route)
        self.assertFalse(active)
