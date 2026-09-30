from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlp_k_block_experiment as experiment
from mlp_k_block_comparison import SCHEDULE
from mlp_k_block_gate import REPORT_SHA256
from mlp_k_block_stage import adapt
from dspark_request_experiment import summarize
from test_mlp_k_block_comparison import fixture


class ReadOrderExperimentTests(unittest.TestCase):
    def test_matched_runtime_routes_restore_between_all_six_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidate = directory / 'mlp-k-block-candidate'
            candidate.mkdir()
            (candidate / 'fused_1d.py').write_text('FusedProjection = object()\n')
            for name in ('fused_1d_input.cpp', 'fused_1d_weights.cpp'):
                (candidate / name).write_text('source\n')
            for name in ('mlp_k_block_gate.py', 'mlp_k_block_comparison.py',
                    'mlp_k_block_experiment.py', 'mlp_k_block.py'):
                (directory / name).write_text('source\n')
            baseline = object()
            original_qualify = lambda: {'report_sha256': 'baseline'}
            fusion = SimpleNamespace(FusedProjection=baseline, qualify_simulator=original_qualify)
            variants = SimpleNamespace(REPORT_SHA256='baseline')
            requests, calls, routed = fixture(), [], []
            def measure(*args, **kwargs):
                index = len(calls)
                enabled, audit = SCHEDULE[index]
                self.assertIs(fusion.FusedProjection is not baseline, enabled)
                self.assertEqual(fusion.qualify_simulator()['report_sha256'], REPORT_SHA256 if enabled else 'baseline')
                self.assertIs(kwargs['audit_features'], audit)
                calls.append((enabled, audit))
                return requests[index]
            full = SimpleNamespace(measure_dspark_request=measure)
            def original_route(value, arm):
                enabled = value['mlp_k_block']['k32']
                self.assertEqual(variants.REPORT_SHA256, REPORT_SHA256 if enabled else 'baseline')
                routed.append(enabled)
            shared = SimpleNamespace(validate_route=original_route)
            def audit(value):
                shared.validate_route(value, 'publication')
            original_finish = lambda *args: self.fail('Original single-arm finish should not run')
            ladder = SimpleNamespace(validate_audit=audit, finish=original_finish)
            report = dict(streams=1)
            def run(*args, **kwargs):
                report['request_checks'] = []
                for enabled, audited in SCHEDULE:
                    options = {name: True for name in ('gdn_shared_qk', 'fused_t16_mlp', 'captured_publication',
                        'target_attention_t16', 'commit_only_gdn', 'proposal_trace', 'native_attention', 'score_layout')}
                    result = full.measure_dspark_request(audit_features=audited, **options)
                    self.assertIs(fusion.FusedProjection, baseline)
                    self.assertIs(fusion.qualify_simulator, original_qualify)
                    if audited:
                        ladder.validate_audit(result)
                    report['request_checks'].append(result)
                ladder.finish(report, summarize)
            modules = dict(dspark_request_experiment=SimpleNamespace(run_loaded_requests=run),
                frozen_ladder_requests=ladder, full_dspark_request=full, fused_t16_scope=fusion,
                dspark_fusion_variants=variants, gdn_shared_qk_variants=shared)
            environment = {name: '1' for name in ('QWEN_MLP_K_BLOCK', 'QWEN_FROZEN_COMBINED_RUNTIME',
                'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS')}
            with patch.dict('sys.modules', modules), patch.dict('os.environ', environment, clear=True), \
                    patch.object(experiment, '__file__', str(directory / 'mlp_k_block_experiment.py')), \
                    patch.object(experiment, 'qualify', return_value=dict(report_sha256=REPORT_SHA256)):
                experiment.run_loaded_requests(None, None, None, None, None, None, None, None, None, None,
                    None, None, report, lambda stage: None, prompt=[1] * 4096, captured_publication=True)
            self.assertEqual(calls, list(SCHEDULE))
            self.assertEqual(report['k_block_comparison']['arms']['k32']['committed_tg'], 120)
            self.assertEqual(report['k_block_sources'], report['k_block_sources_after'])
            self.assertIs(shared.validate_route, original_route)
            self.assertIs(full.measure_dspark_request, measure)
            self.assertIs(ladder.finish, original_finish)
            self.assertEqual(variants.REPORT_SHA256, 'baseline')

    def test_stage_schedule_keeps_winning_publication_policy_in_every_arm(self):
        result = adapt({
            'dspark_request_experiment.py': "def run():\n    schedule = (('publication', True), ('publication', False), ('publication', False))\n    return schedule\n",
            'dspark-target-hardware.py': 'def main():\n    if True:\n        if True:\n            from dspark_request_experiment import run_loaded_requests\n',
            'run-dspark-hardware.sh': 'docker create \\\n    -e "QWEN_DSPARK_MODE=$mode"\n',
        })
        namespace = {}
        exec(result['dspark_request_experiment.py'], namespace)
        self.assertEqual(namespace['run'](), tuple(('publication', audited) for enabled, audited in SCHEDULE))
        self.assertIn('QWEN_MLP_K_BLOCK', result['run-dspark-hardware.sh'])
        with self.assertRaises(ValueError):
            adapt(result)
