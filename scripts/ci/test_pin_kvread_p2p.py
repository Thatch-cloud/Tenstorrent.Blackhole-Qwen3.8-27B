"""pin_kvread.build_problems for version 3 of the qwen_kv_read extension (the P2P restore's device ops and the kernels shipped beside the .so).

Run from scripts/ci:  python3 -B -m unittest test_pin_kvread_p2p"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import pin_kvread  # noqa: E402

SOURCE_DIR = ROOT / 'optimisation' / 'ttnn-op' / 'kv_region_read'
CPP = SOURCE_DIR / 'qwen_kv_read.cpp'


def fake_build(directory, op_names=None, files=None):
    """A build directory as build_kv_read.sh leaves it: the .so (its bytes carry the op names), the manifest, the source and the shipped files."""
    names = list(pin_kvread.OP_NAMES) + list(pin_kvread.P2P_OP_NAMES) if op_names is None else op_names
    (directory / pin_kvread.EXTENSION).write_bytes(b'\x7fELF' + b'\x00'.join(names))
    (directory / pin_kvread.MANIFEST).write_text('manifest\n')
    shutil.copy(str(CPP), str(directory / pin_kvread.SOURCE))
    for name in pin_kvread.P2P_FILES if files is None else files:
        shutil.copy(str(SOURCE_DIR / name), str(directory / name))


class Version3BuildTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix='pin_kvread_'))
        self.addCleanup(shutil.rmtree, str(self.directory), True)

    def test_this_checkouts_source_is_version_3(self):
        self.assertIn(pin_kvread.P2P_OP_NAMES[0], CPP.read_bytes())
        for name in pin_kvread.P2P_OP_NAMES:
            self.assertIn(name, CPP.read_bytes())
        for name in pin_kvread.P2P_FILES:
            self.assertTrue((SOURCE_DIR / name).is_file(), name)

    def test_a_complete_version_3_build_is_accepted(self):
        fake_build(self.directory)
        self.assertEqual(pin_kvread.build_problems(self.directory, CPP), [])

    def test_a_version_2_binary_is_refused(self):
        fake_build(self.directory, op_names=list(pin_kvread.OP_NAMES))
        problems = pin_kvread.build_problems(self.directory, CPP)
        self.assertEqual(len(problems), 1)
        self.assertIn('qwen_p2p_staging_alloc', problems[0])
        self.assertIn('not version 3', problems[0])

    def test_a_missing_or_changed_kernel_is_refused(self):
        fake_build(self.directory, files=['qwen_p2p_plan.hpp', 'qwen_p2p_checksum.cpp'])
        problems = pin_kvread.build_problems(self.directory, CPP)
        self.assertEqual(len(problems), 1)
        self.assertIn('qwen_p2p_scatter.cpp', problems[0])
        (self.directory / 'qwen_p2p_scatter.cpp').write_text('// another kernel\n')
        problems = pin_kvread.build_problems(self.directory, CPP)
        self.assertEqual(len(problems), 1)
        self.assertIn('not this checkout', problems[0])

    def test_a_checkout_that_predates_version_3_still_pins_version_2(self):
        old_source = self.directory / 'old' / 'qwen_kv_read.cpp'
        old_source.parent.mkdir()
        old_source.write_text('// version 2 source: no P2P ops\n')
        build = self.directory / 'build'
        build.mkdir()
        (build / pin_kvread.EXTENSION).write_bytes(b'\x00'.join(pin_kvread.OP_NAMES))
        (build / pin_kvread.MANIFEST).write_text('manifest\n')
        shutil.copy(str(old_source), str(build / pin_kvread.SOURCE))
        self.assertEqual(pin_kvread.build_problems(build, old_source), [])


if __name__ == '__main__':
    unittest.main()
