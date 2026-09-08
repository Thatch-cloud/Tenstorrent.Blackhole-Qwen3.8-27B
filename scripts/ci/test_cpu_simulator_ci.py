import os
from pathlib import Path
import shutil
import subprocess
import unittest


class CpuSimulatorCiTests(unittest.TestCase):
    def test_container_has_no_physical_device_permissions(self):
        source = Path(__file__).with_name('run-simulator.sh').read_text()
        for prohibited in ('--device', '--privileged', '--cap-add', 'src=/dev', 'src=/home,dst='):
            self.assertNotIn(prohibited, source)
        for required in ('--network none', '--cap-drop ALL', '--memory 64g', '--cpus 16', 'sha256sum -c -'):
            self.assertIn(required, source)

    def test_full_simulator_path_cannot_fall_back_to_hardware(self):
        source = Path(__file__).with_name('simulator-suite.sh').read_text()
        self.assertIn('test ! -e /dev/tenstorrent', source)
        self.assertIn('unset QWEN_HARDWARE_TESTS QWEN_CARDS_ALLOCATED', source)
        self.assertIn('export TT_METAL_SIMULATOR=/tmp/ttsim/libttsim_bh_x2.so', source)
        self.assertIn('--stack-layers 5', source)
        self.assertIn('--captured-stack', source)
        self.assertNotIn('--hardware', source)

    @unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Linux CI shell required')
    def test_missing_opt_ins_fail_before_docker(self):
        script = Path(__file__).with_name('run-simulator.sh')
        for enabled, stack in (('0', '0'), ('0', '1'), ('1', '0')):
            result = subprocess.run(['bash', str(script)], env={**os.environ,
                'QWEN_SIM_ONLY': enabled, 'QWEN_LEARNED_STACK': stack}, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b'')
            self.assertEqual(result.stderr, b'')


if __name__ == '__main__':
    unittest.main()
