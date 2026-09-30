import unittest
from types import SimpleNamespace

from dflash_host_attribution import measure_proposals


class Capture:
    def __init__(self):
        self.closed = False
        self.elapsed = 0
        self.events = []
        self.device = SimpleNamespace(progress=None, position=4096, select_proposal=self.select)

    def clock(self):
        return self.elapsed

    def update(self, seed):
        self.events.append(('update', seed))
        self.elapsed += 0.002

    def select(self, seed, count):
        self.events.append(('select', seed, count))
        self.elapsed += 0.003
        return (seed,) * count

    def propose(self, seed, count):
        self.update(seed)
        self.events.append(('replay',))
        self.elapsed += 0.010
        return self.device.select_proposal(seed, count)


class HostAttributionTests(unittest.TestCase):
    def test_intervals_outputs_order_and_restoration(self):
        capture = Capture()
        original = capture.propose
        with measure_proposals(capture, clock=capture.clock) as records:
            self.assertEqual(capture.propose(7, 2), (7, 7))
        self.assertEqual(capture.propose, original)
        self.assertNotIn('propose', capture.__dict__)
        self.assertNotIn('_host_attribution_active', capture.__dict__)
        self.assertEqual(capture.events, [('update', 7), ('replay',), ('select', 7, 2)])
        self.assertEqual(len(records), 1)
        for name, expected in dict(prepare_ms=2, readback_merge_select_ms=3,
                replay_and_bookkeeping_ms=10, proposal_ms=15).items():
            self.assertAlmostEqual(records[0][name], expected)
        self.assertFalse(records[0]['device_kernel_timing'])

    def test_bound_does_not_stop_generation(self):
        capture = Capture()
        with measure_proposals(capture, limit=1, clock=capture.clock) as records:
            capture.propose(7, 1)
            self.assertEqual(capture.propose(8, 1), (8,))
        self.assertEqual(len(records), 1)
        self.assertEqual(len(capture.events), 6)

    def test_exception_restores_owned_methods(self):
        capture = Capture()
        def fail(seed, count):
            raise RuntimeError('selection failed')
        capture.device.select_proposal = fail
        with self.assertRaisesRegex(RuntimeError, 'selection failed'):
            with measure_proposals(capture, clock=capture.clock) as records:
                capture.propose(7, 1)
        self.assertEqual(records, [])
        self.assertIs(capture.device.select_proposal, fail)
        self.assertNotIn('propose', capture.__dict__)

    def test_rejects_audit_and_nested_scopes(self):
        capture = Capture()
        capture.device.progress = object()
        with self.assertRaises(ValueError):
            with measure_proposals(capture):
                self.fail('audit accepted')
        capture.device.progress = None
        with measure_proposals(capture):
            with self.assertRaises(ValueError):
                with measure_proposals(capture):
                    self.fail('nested scope accepted')

    def test_rejects_changed_call_coverage(self):
        capture = Capture()
        capture.propose = lambda seed, count: (seed,)
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            with measure_proposals(capture, clock=capture.clock):
                capture.propose(7, 1)

    def test_rejects_nonfinite_clock(self):
        capture = Capture()
        with self.assertRaisesRegex(ValueError, 'Monotonic'):
            with measure_proposals(capture, clock=lambda: float('nan')) as records:
                capture.propose(7, 1)
        self.assertEqual(records, [])

    def test_rejects_unbounded_or_invalid_limits(self):
        for limit in (0, 65, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                with measure_proposals(Capture(), limit=limit):
                    self.fail('invalid limit accepted')
