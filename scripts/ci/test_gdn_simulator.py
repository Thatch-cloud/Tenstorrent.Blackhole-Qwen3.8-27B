import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('gdn_simulator', Path(__file__).resolve().parents[2] / 'optimisation/sim/gdn-multitoken.py')
simulator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simulator)


class SimulatorGuardTests(unittest.TestCase):
    def test_no_hardware_fallback(self):
        for environment in ({}, {'TT_METAL_SIMULATOR': '/missing', 'TT_METAL_SLOW_DISPATCH_MODE': '1'}):
            with self.assertRaises(RuntimeError):
                simulator.require_simulator(environment)

    def test_library_and_slow_dispatch_required(self):
        with tempfile.NamedTemporaryFile() as library:
            environment = {'TT_METAL_SIMULATOR': library.name}
            with self.assertRaises(RuntimeError):
                simulator.require_simulator(environment)
            environment['TT_METAL_SLOW_DISPATCH_MODE'] = '1'
            simulator.require_simulator(environment)
            for flag in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'):
                with self.assertRaises(RuntimeError):
                    simulator.require_simulator(dict(environment, **{flag: '1'}))


if __name__ == '__main__':
    unittest.main()
