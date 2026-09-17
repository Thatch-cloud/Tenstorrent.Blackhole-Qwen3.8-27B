from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_window_write_experiment as experiment
from dspark_request_experiment import summarize
from test_gdn_window_write_comparison import fixture


class DownGridExperimentTests(unittest.TestCase):
    def test_all_six_requests_keep_norm_prefetch_and_restore_routes(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in experiment.FILES:
                (directory / name).parent.mkdir(parents=True, exist_ok=True)
                (directory / name).write_text('source')
            requests, calls, active = fixture(), [], []
            @contextmanager
            def scope(evidence, directory):
                audit = dict(hits=96, restored=False)
                active.append(True)
                try:
                    yield audit
                finally:
                    active.pop()
                    audit['restored'] = True
            def measure(*args, **kwargs):
                enabled, audited = experiment.SCHEDULE[len(calls)]
                self.assertEqual(bool(active), enabled)
                calls.append((enabled, audited))
                result = requests[len(calls) - 1]
                result['gdn_norm_prefetch'] = dict(enabled=True, builds=96)
                result['fused_t16_mlp'] = dict(hits=[2] * 64)
                result['gdn_shared_qk'] = dict(loads=[{}] * 96)
                return result
            full = SimpleNamespace(measure_dspark_request=measure)
            original_finish = lambda *args: self.fail('Single-arm finish ran')
            ladder = SimpleNamespace(finish=original_finish, validate_audit=lambda request: None)
            report = dict(streams=1)
            def run(*args, **kwargs):
                report['request_checks'] = []
                for enabled, audited in experiment.SCHEDULE:
                    flags = {name: True for name in ('gdn_shared_qk', 'fused_t16_mlp', 'captured_publication',
                        'target_attention_t16', 'commit_only_gdn', 'proposal_trace', 'native_attention', 'score_layout')}
                    report['request_checks'].append(full.measure_dspark_request(audit_features=audited, **flags))
                    self.assertFalse(active)
                ladder.finish(report, summarize)
            modules = dict(dspark_request_experiment=SimpleNamespace(run_loaded_requests=run),
                frozen_ladder_requests=ladder, full_dspark_request=full,
                gdn_shared_qk_variants=SimpleNamespace(validate_route=lambda request, arm: None))
            environment = {name: '1' for name in ('QWEN_GDN_WINDOW_WRITE', 'QWEN_FROZEN_COMBINED_RUNTIME',
                'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS')}
            environment['TT_METAL_HOME'] = 'runtime'
            with patch.dict('sys.modules', modules), patch.dict('os.environ', environment, clear=True), \
                    patch.object(experiment, '__file__', str(directory / 'gdn_window_write_experiment.py')), \
                    patch.object(experiment, 'qualify', return_value={}), patch.object(experiment, 'scoped_window_writes', scope):
                experiment.run_loaded_requests(None, None, None, None, None, None, None, None, None, None,
                    None, None, report, lambda stage: None, prompt=[1] * 4096, captured_publication=True)
            self.assertEqual(calls, list(experiment.SCHEDULE))
            self.assertEqual(report['gdn_window_write_comparison']['arms']['overlap']['committed_tg'], 120)
            self.assertEqual(report['gdn_window_write_sources'], report['gdn_window_write_sources_after'])
            self.assertIs(full.measure_dspark_request, measure)
            self.assertIs(ladder.finish, original_finish)
