import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('attention_layer_gate', Path(__file__).with_name('attention-batch.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class AttentionLayerCLITests(unittest.TestCase):
    def test_tree_width_requires_explicit_parallel_path(self):
        with patch.dict(os.environ, {'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'}, clear=True), patch(
                'sys.argv', ['attention-batch.py', '--tree-parallel']):
            with self.assertRaises(SystemExit):
                gate.main()

    def test_tree_width_requires_native_scratch_before_runtime_import(self):
        arguments = ['attention-batch.py', '--timing', '--ordered-cache', '--grouped', '--dma-layout',
                     '--parallel-groups', '--tree-parallel']
        with patch.dict(os.environ, {'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'}, clear=True), patch(
                'sys.argv', arguments):
            with self.assertRaisesRegex(RuntimeError, 'process-fixed compact native scratch'):
                gate.main()
