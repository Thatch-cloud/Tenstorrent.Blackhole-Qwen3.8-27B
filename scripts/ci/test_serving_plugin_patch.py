import ast
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import serving_fast_policy
from serving_plugin_patch import patch_capacity
from test_serving_fast_policy import FastPolicyTests


SOURCE = '''def get_tt_max_batch_size(vllm_config: "VllmConfig") -> int:
    """Native capacity."""
    return int(vllm_config.scheduler_config.max_num_seqs)

def unrelated():
    return 42
'''


class PluginPatchTests(unittest.TestCase):
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
        source = Path(os.environ['QWEN_PLUGIN_SOURCE']) / 'src/vllm_tt_plugin/config.py'
        self.check_source(source.read_text(encoding='utf-8'))
