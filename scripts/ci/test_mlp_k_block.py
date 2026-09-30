import subprocess
import unittest

from frozen_recipe_context import REVISION
from mlp_k_block import CHANGES, transform


class KBlockTests(unittest.TestCase):
    def test_exact_reversible_edits_and_reject_double_application(self):
        for name, changes in CHANGES.items():
            original = subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/{name}'], text=True)
            candidate = transform(name, original)
            with self.assertRaises(ValueError):
                transform(name, candidate)
            restored = candidate
            for before, after in reversed(changes):
                self.assertEqual(restored.count(after), 1)
                restored = restored.replace(after, before)
            self.assertEqual(restored, original)

    def test_weight_and_input_traversal_preserved(self):
        for worker in range(91):
            traversals = []
            for width in (8, 32):
                pages = []
                for block in range(160 // width):
                    for inner in range(width):
                        for column in range(6):
                            pages.append((block * width + inner) * 544 + worker * 6 + column
                                         if worker * 6 + column < 544 else None)
                traversals.append(pages)
                self.assertEqual([block * width + inner for block in range(160 // width)
                                  for inner in range(width)], list(range(160)))
            self.assertEqual(*traversals)
            self.assertEqual(traversals[0].count(None), 320 if worker == 90 else 0)

    def test_native_control_and_compute_source_unchanged(self):
        import ast
        original = subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/fused_1d.py'], text=True)
        trees = [ast.parse(source) for source in (original, transform('fused_1d.py', original))]
        for name in ('native_gate_up_control', 'fused_compute'):
            bodies = [ast.dump(next(node for node in tree.body if getattr(node, 'name', None) == name))
                      for tree in trees]
            self.assertEqual(*bodies)
