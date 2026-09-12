import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('device_readback', Path(__file__).with_name('device-readback.py'))
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


class HealthTests(unittest.TestCase):
    def test_explicit_allocation_required(self):
        for environment in ({}, {'QWEN_HARDWARE_TESTS': '1'}, {'QWEN_CARDS_ALLOCATED': '1'}):
            with self.assertRaises(RuntimeError):
                health.require_hardware(environment)
        health.require_hardware({'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'})

    def test_simulator_and_slow_dispatch_rejected(self):
        for name in ('TT_METAL_SIMULATOR', 'TT_METAL_SLOW_DISPATCH_MODE'):
            with self.assertRaises(RuntimeError):
                health.require_hardware({'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1', name: '1'})

    def test_both_shards_required(self):
        for shards in ([], [object()], [object()] * 3):
            with self.assertRaises(AssertionError):
                health.check_shards(shards)
        health.check_shards([object(), object()])


if __name__ == '__main__':
    unittest.main()
