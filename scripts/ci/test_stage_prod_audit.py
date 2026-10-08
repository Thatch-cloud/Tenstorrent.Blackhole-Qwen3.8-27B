"""stage_prod_audit (optimisation/ttnn-op/kv_region_read): the production model.py with the narrowed audit, on the CPU.

Run at py 3.11: `py -3.11 -B -m unittest test_stage_prod_audit` from scripts/ci."""

from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'optimisation' / 'ttnn-op' / 'kv_region_read'))

import qwen_prefix_model_patch as patcher  # noqa: E402
import stage_prod_audit as stage  # noqa: E402

ORIGINAL = HERE / 'fixtures' / 'qwen36_model.py'


class StageTests(unittest.TestCase):
    def test_it_writes_the_pinned_graft_and_its_sha_and_rewrites_a_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'MANIFEST.sha256').write_text('x  ./old\n')
            (Path(directory) / 'qwen_kv_read.so').write_bytes(b'so')
            self.assertEqual(stage.main(['--original', str(ORIGINAL), '--out', directory]), 0)
            staged = (Path(directory) / 'prod-audit' / 'model.py').read_bytes()
            self.assertEqual(stage.sha(staged), patcher.PATCHED_SHA256['model.py'])
            self.assertEqual((Path(directory) / 'prod-audit' / 'model.py.sha256').read_text(), '%s  model.py\n' % stage.sha(staged))
            manifest = (Path(directory) / 'MANIFEST.sha256').read_text()
            self.assertIn('./prod-audit/model.py', manifest)
            self.assertIn('./qwen_kv_read.so', manifest)
            self.assertNotIn('./old', manifest)

    def test_a_drifted_original_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            drifted = Path(directory) / 'model.py'
            drifted.write_bytes(ORIGINAL.read_bytes() + b'\n# drift\n')
            with self.assertRaises(SystemExit) as raised:
                stage.main(['--original', str(drifted), '--out', directory])
            self.assertIn('not the pinned', str(raised.exception))

    def test_the_method_diff_names_what_changed(self):
        diff = stage.method_diff('class A:\n    def f(self):\n        return 1\n    def g(self):\n        return 2\n',
                                 'class A:\n    def f(self):\n        return 1\n    def g(self):\n        return 3\n    def h(self):\n        return 4\n')
        self.assertEqual(diff, dict(changed=['A.g'], added=['A.h'], removed=[]))

    def test_exactly_one_source_is_required(self):
        with self.assertRaises(SystemExit):
            stage.main(['--out', 'x'])


if __name__ == '__main__':
    unittest.main()
