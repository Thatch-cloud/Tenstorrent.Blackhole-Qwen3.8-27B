import os
from pathlib import Path
import shutil
import subprocess
import unittest


class CpuSimulatorCiTests(unittest.TestCase):
    def test_dedicated_fusion_workflow_is_serialized_and_cpu_only(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/qwen-ttsim.yml').read_text()
        self.assertIn('group: qwen-two-p150a-exclusive', workflow)
        self.assertIn('timeout-minutes: 180', workflow)
        self.assertIn('bash scripts/ci/run-simulator.sh', workflow)
        suite = Path(__file__).with_name('simulator-suite.sh').read_text()
        self.assertIn('--device-weight-check --trace-replay --trace-t16', suite)
        self.assertIn('compatibility.patched_bytes', suite)
        self.assertIn('fused-batch.exit-status', suite)

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

    def test_shortlist_gate_uses_same_exclusive_cpu_container(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/qwen-experiments.yml').read_text()
        self.assertIn('group: qwen-two-p150a-exclusive', workflow)
        self.assertIn("inputs.learned_stack && 'stack' || 'shortlist'", workflow)
        self.assertIn('bash scripts/ci/run-simulator.sh', workflow)
        suite = Path(__file__).with_name('simulator-suite.sh').read_text()
        self.assertIn('for width in 32768 65536', suite)
        self.assertIn('draft-shortlist-probe.py', suite)
        self.assertIn('test "${QWEN_SIM_CASE:-stack}" = stack', suite)

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
