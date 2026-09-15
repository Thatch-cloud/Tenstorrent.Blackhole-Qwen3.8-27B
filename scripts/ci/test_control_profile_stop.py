from types import SimpleNamespace
import unittest

from control_profile_stop import ControlProfileComplete, stop_after_control


class ControlStopTests(unittest.TestCase):
    def test_complete_control_retained_and_binding_restored(self):
        result = dict(exact=True, state_exact=True, inactive_exact=True, length=65536,
            blocks=[{}], committed_decode_tokens=135)
        original = lambda: result
        module, requests = SimpleNamespace(measure_dspark_request=original), []
        with self.assertRaises(ControlProfileComplete):
            with stop_after_control(module, requests):
                module.measure_dspark_request()
        self.assertEqual(requests, [result])
        self.assertIs(module.measure_dspark_request, original)

    def test_inexact_control_is_not_a_completed_diagnostic(self):
        module = SimpleNamespace(measure_dspark_request=lambda: dict(exact=False))
        requests = []
        with self.assertRaises(ValueError):
            with stop_after_control(module, requests):
                module.measure_dspark_request()
        self.assertFalse(requests)


if __name__ == '__main__':
    unittest.main()
