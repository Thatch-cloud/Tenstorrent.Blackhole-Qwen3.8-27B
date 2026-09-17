from pathlib import Path
import shutil
import subprocess
import unittest


class LadderHardwareCiTests(unittest.TestCase):
    def test_registration_and_bounded_probe(self):
        directory = Path(__file__).parent
        suite = (directory / 'ladder-hardware-suite.sh').read_text()
        runner = (directory / 'run-ladder-hardware.sh').read_text()
        workflow = (directory.parents[1] / '.github/workflows/qwen-experiments.yml').read_text()
        self.assertIn("if: inputs.suite == 'dspark-ladder-attention-hardware'", workflow)
        self.assertIn('timeout -k 30 2400 bash scripts/ci/run-ladder-hardware.sh', workflow)
        self.assertLess(suite.index('device-owners.py'), suite.index('dspark_ladder_build.py'))
        self.assertLess(suite.index('hardware-correctness.py'), suite.index('dspark_ladder_build.py'))
        self.assertIn('timeout -k 15 300 python3', suite)
        self.assertIn('--hardware --output', suite)
        self.assertIn('--device /dev/tenstorrent/0 --device /dev/tenstorrent/2', runner)
        self.assertIn('p150_x2_mesh_graph_descriptor.textproto', runner)
        self.assertIn('type=bind,src=$results,dst=/experiment/results', runner)
        self.assertNotIn('QWEN_SIM_SHARED_BDF=1', runner)
        self.assertNotIn('reset-cards', runner)
        self.assertNotIn('--privileged', runner)

    @unittest.skipUnless(shutil.which('bash'), 'Bash required')
    def test_scripts_parse_and_reject_unallocated_launch(self):
        directory = Path(__file__).parent
        for name in ('run-ladder-hardware.sh', 'ladder-hardware-suite.sh'):
            subprocess.run(['bash', '-n', str(directory / name)], check=True, timeout=10)
            result = subprocess.run(['bash', str(directory / name)], env={'QWEN_CARDS_ALLOCATED': '0'},
                capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
