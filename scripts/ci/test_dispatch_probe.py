import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


spec = importlib.util.spec_from_file_location('dispatch_probe', Path(__file__).with_name('dispatch-probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class DispatchProbeTests(unittest.TestCase):
    def test_hardware_requires_both_allocation_flags(self):
        for environment in ({}, {'QWEN_HARDWARE_TESTS': '1'}, {'QWEN_CARDS_ALLOCATED': '1'}):
            with self.assertRaises(RuntimeError):
                probe.backend(environment)
        self.assertEqual(probe.backend({'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'}), 'hardware')

    def test_simulator_does_not_claim_silicon_and_rejects_slow_dispatch(self):
        environment = {'TT_METAL_SIMULATOR': '/opt/ttsim/simulator/libttsim_bh_x2.so'}
        self.assertEqual(probe.backend(environment), 'ttsim-fast-dispatch')
        with self.assertRaisesRegex(RuntimeError, 'fast-dispatch'):
            probe.backend(dict(environment, TT_METAL_SLOW_DISPATCH_MODE='1'))

    def test_explicit_type_leaves_axis_resolution_to_runtime(self):
        operations = SimpleNamespace(DispatchCoreType=SimpleNamespace(ETH='eth', WORKER='worker'), DispatchCoreConfig=Mock())
        for selection, expected in (('ethernet', 'eth'), ('worker', 'worker')):
            probe.dispatch_config(operations, selection)
            operations.DispatchCoreConfig.assert_called_with(expected)
        for selection in ('auto', None, True):
            with self.assertRaises(ValueError):
                probe.dispatch_config(operations, selection)
