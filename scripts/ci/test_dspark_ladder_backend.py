import unittest

from dspark_ladder_backend import require_backend


class LadderBackendTests(unittest.TestCase):
    def test_hardware_requires_allocation_and_rejects_simulator_state(self):
        environment = dict(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1',
            QWEN_LADDER_BACKEND='hardware', QWEN_LADDER_CONTEXT='65536', QWEN_PROJECTION_LINKS='4')
        self.assertEqual(require_backend(environment, hardware=True, device_present=True), 'hardware')
        for name in environment:
            altered = dict(environment)
            altered.pop(name)
            with self.subTest(missing=name), self.assertRaises((RuntimeError, ValueError)):
                require_backend(altered, hardware=True, device_present=True)
        for name in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH', 'QWEN_SIM_ONLY',
                'QWEN_SIM_CASE', 'QWEN_SIM_SHARED_BDF', 'QWEN_SIM_BOUNDED_MEMORY', 'QWEN_LADDER_SCORE_SMOKE'):
            with self.subTest(contaminated=name), self.assertRaises((RuntimeError, ValueError)):
                require_backend(dict(environment, **{name: '1'}), hardware=True, device_present=True)
        with self.assertRaises(ValueError):
            require_backend(environment, hardware=True, device_present=False)

    def test_simulator_cannot_admit_physical_cards(self):
        environment = dict(TT_METAL_SIMULATOR='/lib/sim.so', QWEN_SIM_SHARED_BDF='1',
            QWEN_SIM_BOUNDED_MEMORY='1')
        self.assertEqual(require_backend(environment, hardware=False, device_present=False), 'simulator')
        with self.assertRaises(ValueError):
            require_backend(environment, hardware=False, device_present=True)
        with self.assertRaises(ValueError):
            require_backend(dict(environment, QWEN_CARDS_ALLOCATED='1'), hardware=False, device_present=False)
