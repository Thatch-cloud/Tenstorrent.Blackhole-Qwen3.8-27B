import ast
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import serving_fast_policy
from serving_plugin_patch import patch_capacity, patch_platform, patch_worker
from test_serving_fast_policy import FastPolicyTests


SOURCE = '''def get_tt_max_batch_size(vllm_config: "VllmConfig") -> int:
    """Native capacity."""
    return int(vllm_config.scheduler_config.max_num_seqs)

def unrelated():
    return 42
'''


class PluginPatchTests(unittest.TestCase):
    def test_platform_retains_default_rejection_and_requires_valid_fast_profile(self):
        source = '''class Platform:
    @classmethod
    def configure(cls, vllm_config):
        assert not vllm_config.speculative_config, "unsupported"
        return "configured"
'''
        patched = patch_platform(source)
        namespace = {}
        with patch.dict(sys.modules, {'vllm_tt_plugin.qwen_fast_policy': serving_fast_policy}):
            exec(patched, namespace)
            config = FastPolicyTests().fixture()
            self.assertEqual(namespace['Platform'].configure(config), 'configured')
            config.additional_config = {}
            with self.assertRaises(AssertionError):
                namespace['Platform'].configure(config)
            config.speculative_config = None
            self.assertEqual(namespace['Platform'].configure(config), 'configured')
            config = FastPolicyTests().fixture()
            config.scheduler_config.max_num_seqs = 2
            with self.assertRaises(ValueError):
                namespace['Platform'].configure(config)
        with self.assertRaises(ValueError):
            patch_platform(patched)

    def test_worker_routes_only_explicit_fast_opt_in_and_closes_before_baseline(self):
        source = '''class Worker:
    def compile_or_warm_up_model(self):
        return "baseline"
    def __del__(self):
        self.baseline_closed = True
'''
        patched = patch_worker(source)
        namespace = {}
        startup = SimpleNamespace(warmup=Mock(return_value='fast'), stop=Mock())
        with patch.dict(sys.modules, {'vllm_tt_plugin.qwen_fast_policy': serving_fast_policy,
                'serving_startup': startup}):
            exec(patched, namespace)
            worker = namespace['Worker']()
            worker.vllm_config = FastPolicyTests().fixture()
            self.assertEqual(worker.compile_or_warm_up_model(), 'fast')
            startup.warmup.assert_called_once_with(worker)
            worker.vllm_config.additional_config = None
            self.assertEqual(worker.compile_or_warm_up_model(), 'baseline')
            worker._qwen_fast_resources = object()
            startup.stop.side_effect = lambda value: self.assertFalse(hasattr(value, 'baseline_closed'))
            worker.__del__()
            self.assertTrue(worker.baseline_closed)
            worker._qwen_fast_resources = None
        with self.assertRaises(ValueError):
            patch_worker(patched)

    def check_source(self, source):
        result = patch_capacity(source)
        tree = ast.parse(result)
        function = next(node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'get_tt_max_batch_size')
        namespace = {}
        package = ModuleType('vllm_tt_plugin')
        with patch.dict(sys.modules, {'vllm_tt_plugin': package,
                'vllm_tt_plugin.qwen_fast_policy': serving_fast_policy}):
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<patched-capacity>', 'exec'), namespace)
            config = FastPolicyTests().fixture()
            self.assertEqual(namespace['get_tt_max_batch_size'](config), 8)
            self.assertEqual(config.scheduler_config.max_num_seqs, 1)
            config.additional_config = {}
            self.assertEqual(namespace['get_tt_max_batch_size'](config), 1)
        with self.assertRaises(ValueError):
            patch_capacity(result)
        return result

    def test_capacity_hook_only(self):
        result = self.check_source(SOURCE)
        self.assertIn('def unrelated():\n    return 42', result)

    def test_changed_upstream_contract_rejected(self):
        for source in (SOURCE.replace('max_num_seqs)', 'max_num_seqs + 1)'),
                SOURCE.replace('get_tt_max_batch_size', 'renamed'), SOURCE + SOURCE):
            with self.assertRaises(ValueError):
                patch_capacity(source)

    @unittest.skipUnless(os.environ.get('QWEN_PLUGIN_SOURCE'), 'Pinned source available in installed-vLLM CI')
    def test_actual_pinned_plugin_source(self):
        package = Path(os.environ['QWEN_PLUGIN_SOURCE']) / 'src/vllm_tt_plugin'
        source = package / 'config.py'
        self.check_source(source.read_text(encoding='utf-8'))
        patch_platform((package / 'platform.py').read_text(encoding='utf-8'))
        patch_worker((package / 'worker.py').read_text(encoding='utf-8'))
