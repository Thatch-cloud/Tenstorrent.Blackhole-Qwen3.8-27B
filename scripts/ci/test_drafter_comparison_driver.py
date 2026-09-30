from contextlib import ExitStack, contextmanager, nullcontext
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from drafter_comparison_experiment import run_loaded_requests
from drafter_comparison_report import SCHEDULE
from sampling_link_policy import SOURCES


class ComparisonDriverTests(unittest.TestCase):
    def exercise(self, fail=False):
        calls, active = [], []

        @contextmanager
        def tracked():
            active.append(True)
            try:
                yield
            finally:
                active.pop()

        def measure(drafter, *args, **kwargs):
            self.assertEqual(kwargs['max_new_tokens'], 256)
            self.assertIs(kwargs['prefill'], callbacks['prefill'])
            calls.append((drafter, kwargs['audit_features']))
            if drafter == 'dspark':
                self.assertTrue(active)
                for key in ('gdn_shared_qk', 'target_attention_t16', 'captured_publication', 'score_layout'):
                    self.assertIs(kwargs[key], True)
            elif fail:
                raise RuntimeError('DFlash2 failed')
            return dict(emitted=[20, 1], eos_ids=[1])

        callbacks = dict(prefill=object(), decode=object(), live_digest=object(),
                         kv_digest=object(), inactive_digest=object(), eos_ids=[1])
        sampler = SimpleNamespace(tt_sampling=SimpleNamespace(num_argmax_gather_links=4))
        model = SimpleNamespace(mesh_device=object(), layers=[], _paged_kv_caches=[])
        report = dict(streams=1, sampling_link_sources=dict(SOURCES))
        environment = {name: '1' for name in ('QWEN_DRAFTER_COMPARISON', 'QWEN_CARDS_ALLOCATED',
            'QWEN_HARDWARE_TESTS', 'QWEN_CUMULATIVE_NORM', 'QWEN_CUMULATIVE_REGISTER', 'QWEN_CUMULATIVE_MLP_DOWN')}
        environment.update(TT_METAL_HOME='.', TT_METAL_SIMULATOR='', TT_METAL_DEVICE_PROFILER='')
        audit = dict(direct=dict(hits=48), compact=dict(calls=1), down=dict(hits=[1] * 64))
        replacements = {
            'full_dflash_request.load_dflash_fixtures': lambda *args: (),
            'full_dspark_request.measure_dspark_request': lambda *args, **kwargs: measure('dspark', *args, **kwargs),
            'dflash_combined_request.measure_combined_dflash': lambda *args, **kwargs: measure('dflash2', *args, **kwargs),
            'drafter_request_environment.prepare': lambda *args: (sampler, [], callbacks),
            'dspark_request_experiment.warm_native_control': lambda *args: None,
            'dspark_request_experiment.cache_formats': lambda *args: {},
            'sampling_link_policy.sampler_links': lambda *args: nullcontext(),
            'sampling_link_policy.audit': lambda *args: dict(SOURCES),
            'native_draft_sdpa.precise_draft_kernel': lambda *args: nullcontext({}),
            'cumulative_t16_scope.scoped_cumulative_t16': lambda *args, **kwargs: nullcontext(audit),
            'cumulative_t16_scope.validate_request': lambda *args: None,
            'cumulative_register_runtime.runtime_scope': lambda *args, **kwargs: nullcontext(tracked),
            'cumulative_norm_runtime.require_active': lambda: None,
            'cumulative_norm_runtime.select_norm': lambda *args: nullcontext(),
            'cumulative_norm_validation.validate_norm_history': lambda *args: None,
            'frozen_ladder_requests.validate_audit': lambda *args: None,
            'gdn_shared_qk_variants.validate_route': lambda *args: None,
            'compact_score_gate.qualify': lambda *args: {},
            'gdn_direct_window_gate.qualify': lambda *args: {},
            'mlp_down_grid_gate.qualify': lambda *args: {},
            'dspark_score_layout_hardware_gate.qualify': lambda *args: {},
            'dspark_score_layout_hardware_audit.audit': lambda *args: {},
            'full_dflash_request.summarize_dflash_requests': lambda *args, **kwargs: {},
            'drafter_comparison_experiment.summarize': lambda *args: {'checked': True},
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
                self.assertFalse(active)
        self.assertEqual(calls, list(SCHEDULE))
        self.assertEqual(len(report['request_checks']), 6)
        self.assertTrue(all(request['ended_with_eos'] and request['sampler_num_links'] == 4
                            and request['fabric_sources'] == SOURCES for request in report['request_checks']))
        self.assertEqual(report['drafter_comparison_sources'], report['drafter_comparison_sources_after'])

    def test_all_six_complete_requests_share_target_callbacks(self):
        self.exercise()

    def test_failed_drafter_does_not_enter_timed_requests(self):
        with self.assertRaisesRegex(RuntimeError, 'DFlash2 failed'):
            self.exercise(fail=True)
