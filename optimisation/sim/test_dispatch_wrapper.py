import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Linux shell regression test')
class DispatchWrapperTests(unittest.TestCase):
    def run_edit_race(self, source, probe='dispatch-probe'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            simulator = root / 'simulator'
            simulator.mkdir()
            for name in ('libttsim_bh_x2.so', 'blackhole_P300_both_mmio.yaml'):
                (simulator / name).touch()
            descriptor = root / 'tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml'
            descriptor.parent.mkdir(parents=True)
            descriptor.touch()
            executable = root / 'venv/bin/python'
            executable.parent.mkdir(parents=True)
            executable.write_text('#!/bin/bash\nprintf ready > "$SIM_ROOT/started"\n'
                'while [ ! -f "$SIM_ROOT/release" ]; do sleep 0.01; done\nexit 0\n')
            executable.chmod(0o755)
            wrapper = root / 'run-dispatch-probe.sh'
            wrapper.write_text(source)
            environment = {**os.environ, 'SIM_ROOT': str(root), 'QWEN_SIM_PACKER_ZERO_GRAFT': '0',
                'QWEN_SIM_SHARED_BDF': '0', 'QWEN_SIM_DISPATCH_PROBE': probe, 'KERNEL_TIMEOUT': '10'}
            process = subprocess.Popen(['bash', str(wrapper)], env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic() + 5
                while not (root / 'started').exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not (root / 'started').exists():
                    (root / 'release').touch()
                    output, unused_error = process.communicate(timeout=5)
                    self.fail(f'Fake kernel did not start: {output}')
                terminal = source.index("printf '%s\\n'")
                wrapper.write_text(source[:terminal] + 'invalid shell text "\n')
                (root / 'release').touch()
                output, unused_error = process.communicate(timeout=10)
                statuses = [path.read_text().strip() for path in (root / 'results').glob('*.exit-status')]
                return process.returncode, output, statuses
            finally:
                (root / 'release').touch()
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)

    def test_terminal_block_survives_edits_to_a_running_wrapper(self):
        source = Path(__file__).with_name('run-dispatch-probe.sh').read_text()
        code, output, statuses = self.run_edit_race(source)
        self.assertEqual(code, 0, output)
        self.assertEqual(statuses, ['0'])

    def test_old_unparsed_tail_reproduces_the_recorded_failure(self):
        source = Path(__file__).with_name('run-dispatch-probe.sh').read_text()
        source = source.replace('{\nREPORT=', 'REPORT=').replace('\n}\n', '\n')
        code, output, statuses = self.run_edit_race(source)
        self.assertNotEqual(code, 0)
        self.assertIn('unexpected EOF', output)
        self.assertEqual(statuses, [])

    def test_dspark_rotary_route_preserves_terminal_status(self):
        source = Path(__file__).with_name('run-dispatch-probe.sh').read_text()
        code, output, statuses = self.run_edit_race(source, 'dspark-rotary-probe')
        self.assertEqual(code, 0, output)
        self.assertIn('dspark-rotary-probe.json', output)
        self.assertEqual(statuses, ['0'])

    def test_dspark_projection_route_preserves_terminal_status(self):
        source = Path(__file__).with_name('run-dispatch-probe.sh').read_text()
        for probe in ('dspark-projection-probe','dspark-norm-precision-probe','dspark-layer-probe'):
            with self.subTest(probe=probe):
                code, output, statuses = self.run_edit_race(source, probe)
                self.assertEqual(code, 0, output)
                self.assertIn(probe+'.json', output)
                self.assertEqual(statuses, ['0'])


if __name__ == '__main__':
    unittest.main()
