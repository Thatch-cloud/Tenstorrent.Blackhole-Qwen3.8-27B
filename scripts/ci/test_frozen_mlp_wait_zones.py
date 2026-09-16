from pathlib import Path
import unittest

from frozen_mlp_wait_zones import instrument, remove_scopes, scoped_statement, ZONES, PROFILER_ENV, profile_trace_source


class WaitZoneTests(unittest.TestCase):
    def test_drain_does_not_replace_replays_or_checks(self):
        source = Path(__file__).with_name('fusion_trace.py').read_text()
        candidate = profile_trace_source(source)
        compile(candidate, 'fusion_trace.py', 'exec')
        self.assertEqual(candidate.count('operations.ReadDeviceProfiler(mesh)'), 4)
        self.assertEqual(candidate.count('operations.execute_trace('), source.count('operations.execute_trace('))
        self.assertIn('operations.ReadDeviceProfiler(mesh)\n        for trace in traces.values():', candidate)
        self.assertEqual(candidate.count('raise AssertionError'), source.count('raise AssertionError'))

    def test_trace_identity_tracking_required(self):
        self.assertEqual(dict(entry.split('=') for entry in PROFILER_ENV.split()),
            dict(TT_METAL_DEVICE_PROFILER='1', TT_METAL_PROFILER_TRACE_TRACKING='1'))

    def test_original_sources_round_trip(self):
        for role in ZONES:
            with self.subTest(role=role):
                source = Path(__file__).with_name(f'fused_1d_{role}.cpp').read_text()
                candidate = instrument(source, role)
                self.assertEqual(remove_scopes(candidate, role), source)
                self.assertEqual(candidate.count('DeviceZoneScopedN('), len(ZONES[role]))
                self.assertIn('block == 10', candidate)
                with self.assertRaises(ValueError):
                    instrument(candidate, role)

    def test_each_branch_executes_same_single_statement(self):
        for entries in ZONES.values():
            for statement, name, predicate in entries:
                wrapped = scoped_statement(statement, name, predicate)
                self.assertEqual(wrapped.count(statement), 2)
                self.assertEqual(wrapped.count(' else {'), 1)
                self.assertEqual(wrapped.count('DeviceZoneScopedN'), 1)
                self.assertIn(f'if ({predicate})', wrapped)

    def test_source_drift_rejected(self):
        source = Path(__file__).with_name('fused_1d_input.cpp').read_text()
        with self.assertRaises(ValueError):
            instrument(source.replace('cb_reserve_back(0, 8);', 'cb_reserve_back(0, 16);'), 'input')
        with self.assertRaises(ValueError):
            instrument(source + '\ncb_reserve_back(0, 8);\n', 'input')

    def test_unknown_role_rejected(self):
        with self.assertRaises(ValueError):
            instrument('', 'compute')


if __name__ == '__main__':
    unittest.main()
