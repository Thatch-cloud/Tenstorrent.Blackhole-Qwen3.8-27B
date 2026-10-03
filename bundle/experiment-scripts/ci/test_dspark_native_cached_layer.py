import ast
from pathlib import Path
import unittest

import dspark_cached_layer
import dspark_native_cached_layer
import dspark_native_full_attention


class NativeCachedLayerTests(unittest.TestCase):
    def test_candidate_changes_only_attention_backend(self):
        directory = Path(__file__).parent
        implementations = []
        for name in ('dspark_cached_layer.py', 'dspark_native_cached_layer.py'):
            module = ast.parse((directory / name).read_text())
            implementations.append([ast.dump(node, include_attributes=False) for node in module.body
                if isinstance(node, ast.FunctionDef)])
        self.assertEqual(implementations[0], implementations[1])
        self.assertIs(dspark_native_cached_layer.attend, dspark_native_full_attention.execute)
        self.assertIsNot(dspark_native_cached_layer.attend, dspark_cached_layer.attend)
