import unittest

from verifier_trace_profile import profile_replays


class VerifierTraceProfileTests(unittest.TestCase):
    def run_profile(self, events, **overrides):
        options = dict(rows=8, length=4095, traces=dict(serial=1, control=2, batch=3),
            restore=lambda: events.append('restore'), synchronize=lambda: events.append('sync'),
            execute=lambda trace: events.append(('execute', trace)),
            validate=lambda arm: events.append(('validate', arm)),
            dump=lambda: events.append('dump'), signpost=lambda label: events.append(label))
        options.update(overrides)
        return profile_replays(**options)

    def test_exact_existing_traces_and_boundary_order(self):
        events = []
        report = self.run_profile(events)
        self.assertEqual(len(report['records']), 9)
        self.assertEqual([record['trace_id'] for record in report['records']], [1, 2, 3] * 3)
        for index, record in enumerate(report['records']):
            self.assertEqual(events[index * 9:(index + 1) * 9], [
                'restore', 'sync', 'dump', record['label'] + '_begin',
                ('execute', record['trace_id']), 'sync', record['label'] + '_end',
                'dump', ('validate', record['arm'])])

    def test_invalid_scope_has_no_side_effects(self):
        for overrides in (dict(rows=16), dict(length=31), dict(traces=dict(batch=1))):
            events = []
            with self.assertRaises(ValueError):
                self.run_profile(events, **overrides)
            self.assertEqual(events, [])

    def test_failed_replay_preserves_end_marker_and_propagates(self):
        events = []
        def fail(trace):
            raise RuntimeError('replay failed')
        with self.assertRaisesRegex(RuntimeError, 'replay failed'):
            self.run_profile(events, execute=fail)
        self.assertEqual(events[-2:], ['qwen_verifier_t8_ctx4095_serial_0_end', 'dump'])

    def test_failed_validation_stops_before_next_replay(self):
        events = []
        def fail(arm):
            raise AssertionError('state mismatch')
        with self.assertRaisesRegex(AssertionError, 'state mismatch'):
            self.run_profile(events, validate=fail)
        self.assertEqual(sum(event == 'restore' for event in events), 1)
