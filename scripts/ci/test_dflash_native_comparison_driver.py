from contextlib import ExitStack, nullcontext
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dflash_native_comparison_experiment import run_loaded_requests
from dflash_native_comparison_report import SCHEDULE
from sampling_link_policy import SOURCES


class NativeDriverTests(unittest.TestCase):
    def exercise(self, fail_audit=False):
        calls = []
        callbacks = dict(prefill=object(), decode=object(), live_digest=object(),
            kv_digest=object(), inactive_digest=object(), eos_ids=[1])
        sampler = SimpleNamespace(tt_sampling=SimpleNamespace(num_argmax_gather_links=4))
        model = SimpleNamespace(mesh_device=object(), layers=[], _paged_kv_caches=[])
        report = dict(streams=1, sampling_link_sources=dict(SOURCES))
        environment = {name: '1' for name in ('QWEN_DFLASH_NATIVE_COMPARISON', 'QWEN_CARDS_ALLOCATED',
            'QWEN_HARDWARE_TESTS', 'QWEN_CUMULATIVE_NORM', 'QWEN_CUMULATIVE_REGISTER', 'QWEN_CUMULATIVE_MLP_DOWN')}
        environment.update(TT_METAL_HOME='.', TT_METAL_SIMULATOR='', TT_METAL_DEVICE_PROFILER='')

        def measure(*args, **kwargs):
            native = 'native_attention_evidence' in kwargs
            self.assertIs(kwargs['prefill'], callbacks['prefill'])
            self.assertEqual(kwargs['max_new_tokens'], 256)
            calls.append((native, kwargs['audit_features']))
            return dict(emitted=[20, 1], eos_ids=[1], native_proposal_attention=native)

        def audit(requests, **kwargs):
            if fail_audit and requests[0]['native_proposal_attention']:
                raise ValueError('candidate target audit failed')

        replacements = {
            'full_dflash_request.load_dflash_fixtures': lambda *args: (),
            'full_dflash_request.summarize_dflash_requests': audit,
            'dflash_combined_request.measure_combined_dflash': measure,
            'dflash_t16_native_scope.admit': lambda *args: {},
            'drafter_request_environment.prepare': lambda *args: (sampler, [], callbacks),
            'dspark_request_experiment.warm_native_control': lambda *args: None,
            'dspark_request_experiment.cache_formats': lambda *args: {},
            'sampling_link_policy.sampler_links': lambda *args: nullcontext(),
            'sampling_link_policy.audit': lambda *args: dict(SOURCES),
            'cumulative_register_runtime.runtime_scope': lambda *args, **kwargs: nullcontext(),
            'cumulative_norm_runtime.require_active': lambda: None,
            'dflash_native_comparison_experiment.summarize': lambda *args: {'checked': True},
        }
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment))
            for name, implementation in replacements.items():
                stack.enter_context(patch(name, implementation))
            try:
                run_loaded_requests(None, None, model, None, None, None, None, None,
                    None, None, None, None, report, lambda *args: None,
                    prompt=[1] * 4096, context={}, max_new_tokens=None)
            finally:
                self.assertEqual(calls, list(SCHEDULE[:2] if fail_audit else SCHEDULE))
                self.assertEqual(len(report['request_checks']), len(calls))
                self.assertEqual(report['drafter_comparison_sources'], report['drafter_comparison_sources_after'])

    def test_all_six_requests_share_target_and_explicit_candidate_evidence(self):
        self.exercise()

    def test_failed_target_audit_prevents_timing_and_retains_result(self):
        with self.assertRaisesRegex(ValueError, 'target audit'):
            self.exercise(fail_audit=True)


if __name__ == '__main__':
    unittest.main()
