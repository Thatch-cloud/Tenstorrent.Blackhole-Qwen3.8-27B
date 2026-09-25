from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import shared_qk_norm_comparison_scope as scope
from shared_qk_norm_comparison import SCHEDULE, summarize
from shared_qk_norm_stage import adapt
from dspark_request_experiment import summarize as summarize_requests
from test_mlp_read_order_comparison import fixture as read_fixture


def fixture():
    result = read_fixture()
    for value in result:
        value['norm_reader'] = dict(scatter=value.pop('weight_read_order')['staggered'])
    return result


class NormComparisonTests(unittest.TestCase):
    def test_same_cycle_accounting_and_reject_changed_acceptance(self):
        result = summarize(fixture(), summarize_requests, lambda value: None, lambda value, arm: None)
        self.assertEqual(result['arms']['prefetched']['committed_tg'], 100)
        self.assertEqual(result['arms']['scatter']['committed_tg'], 120)
        self.assertTrue(result['improvement_screen_passed'])
        self.assertFalse(result['performance_promoted'])
        for field, changed in (('emitted', [999]), ('decode_ms', 1), ('state_exact', False)):
            requests = fixture()
            requests[3][field] = changed
            with self.assertRaises(ValueError):
                summarize(requests, summarize_requests, lambda value: None, lambda value, arm: None)

    def test_scope_preserves_incremental_route_and_restores_builds(self):
        for fail in (False, True):
            calls, routes, builds = [], [], []
            original_build = object()
            original_qualify = object()
            shared = SimpleNamespace(build=original_build)
            gate = SimpleNamespace(qualify=original_qualify)
            variants = SimpleNamespace(REPORT_SHA256='original')

            def route(value, arm):
                self.assertEqual(arm, 'publication')
                self.assertTrue(value['incremental_history']['enabled'])
                expected = scope.SCATTER_SHA256 if value['norm_reader']['scatter'] else scope.PREFETCH_SHA256
                self.assertEqual(variants.REPORT_SHA256, expected)
                self.assertEqual(value['admission']['report_sha256'], expected)
                routes.append(expected)

            variants.validate_route = route

            def measure(**kwargs):
                index = len(calls)
                enabled, audit = SCHEDULE[index]
                calls.append(index)
                if fail:
                    raise RuntimeError('injected request failure')
                self.assertEqual(kwargs['audit_features'], audit)
                admission = gate.qualify()
                for layer in range(48):
                    self.assertEqual(shared.build(layer), [1, 2, 3])
                value = fixture()[index]
                value.update(admission=admission, incremental_history=dict(enabled=True))
                return value

            full = SimpleNamespace(measure_dspark_request=measure)

            @contextmanager
            def incremental(directory):
                yield

            modules = dict(full_dspark_request=full, gdn_shared_qk_scope=shared,
                gdn_shared_qk_gate=gate, gdn_shared_qk_variants=variants)
            environment = dict(QWEN_SHARED_QK_NORM_COMPARISON='1', QWEN_DSPARK_REQUEST_CONTEXT='4096',
                TT_METAL_HOME='pinned')
            def selected(enabled):
                def build(layer):
                    builds.append(enabled)
                    return [1, 2, 3]
                return build

            with patch.dict('sys.modules', modules), patch.dict('os.environ', environment, clear=True), \
                    patch.object(scope, 'incremental_scope', incremental), \
                    patch.object(scope, 'qualify_prefetch', return_value=dict(report_sha256=scope.PREFETCH_SHA256)), \
                    patch.object(scope, 'qualify_scatter', return_value=dict(report_sha256=scope.SCATTER_SHA256)), \
                    patch.object(scope, 'prefetch_build', selected(False)), \
                    patch.object(scope, 'scatter_build', selected(True)):
                def run():
                    with scope.runtime_scope('directory'):
                        for enabled, audit in SCHEDULE:
                            options = {name: True for name in ('gdn_shared_qk', 'fused_t16_mlp',
                                'captured_publication', 'target_attention_t16', 'commit_only_gdn',
                                'proposal_trace', 'native_attention', 'score_layout')}
                            value = full.measure_dspark_request(audit_features=audit, **options)
                            self.assertIs(shared.build, original_build)
                            self.assertIs(gate.qualify, original_qualify)
                            variants.validate_route(value, 'publication')
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'injected request failure'):
                        run()
                else:
                    run()
                    self.assertEqual(builds, [enabled for enabled, audit in SCHEDULE for layer in range(48)])
                    self.assertEqual(len(routes), 6)
            self.assertIs(shared.build, original_build)
            self.assertIs(gate.qualify, original_qualify)
            self.assertIs(full.measure_dspark_request, measure)
            self.assertIs(variants.validate_route, route)
            self.assertEqual(variants.REPORT_SHA256, 'original')

    def test_stage_replaces_norm_scope_without_disabling_incremental_history(self):
        sources = {
            'dspark_request_experiment.py': "def run():\n    schedule = (('publication', True), ('publication', False), ('publication', False))\n    return schedule\n",
            'dspark-target-hardware.py': 'def main():\n    if True:\n        if True:\n            from dspark_request_experiment import run_loaded_requests\n',
            'run-dspark-hardware.sh': 'docker create \\\n    -e "QWEN_DSPARK_MODE=$mode"\n',
            'frozen_draft_tail_scope.py': 'from frozen_gdn_norm_scope import runtime_scope as norm_scope\n',
        }
        result = adapt(sources)
        self.assertIn('from shared_qk_norm_comparison_scope import runtime_scope', result['frozen_draft_tail_scope.py'])
        self.assertIn('from shared_qk_norm_experiment import run_loaded_requests', result['dspark-target-hardware.py'])
        self.assertIn('QWEN_SHARED_QK_NORM_COMPARISON', result['run-dspark-hardware.sh'])
        namespace = {}
        exec(result['dspark_request_experiment.py'], namespace)
        self.assertEqual(namespace['run'](), tuple(('publication', audit) for enabled, audit in SCHEDULE))
        with self.assertRaises(ValueError):
            adapt(result)
