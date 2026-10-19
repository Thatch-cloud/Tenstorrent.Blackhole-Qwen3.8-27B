"""The two fusion packages that wrap begin_trace_capture / end_trace_capture (draft_fusion_tp for WP6's audits, grid13_tp for M1's) must not re-wrap each other.

Card run 38129663805 (the audited attach of the combined candidate): from round ~81 every mid-serving draft build (a new pair, the octo draft's rebuild) fell back with
'RecursionError: maximum recursion depth exceeded' and the octo draft was disabled for the process. Both modules' track() called on EVERY audit call and each looked only for its own marker
on the TOP wrapper; the other's wrapper being on top, each wrapped again, so the chain grew by one layer per alternation (a Lever N cycle alternates prefill chunks, whose agmm audits call
grid13's, with draft passes, whose audits call WP6's) until the next capture's begin_trace_capture recursed through it. Neither package alone alternates, so neither package's tests saw it."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import draft_fusion_tp as wp6  # noqa: E402
import grid13_tp as grid13  # noqa: E402


class Device(object):
    """A ttnn stand-in with the two capture calls, counting the real ones."""

    def __init__(self):
        self.began = self.ended = 0
        self.begin_trace_capture = self._begin
        self.end_trace_capture = self._end

    def _begin(self, mesh, cq_id=0):
        self.began += 1
        return self.began

    def _end(self, mesh, trace, cq_id=0):
        self.ended += 1


def depth(function):
    """The number of tracker wrappers in the chain from `function` down to the first function that is not a tracker's (each wrapper lists itself and the trackers beneath it)."""
    return len(getattr(function, 'capture_trackers', ()))


class TrackerTests(unittest.TestCase):
    def setUp(self):
        wp6.reset()
        grid13.reset()

    def test_alternating_audit_calls_wrap_once_each_and_never_again(self):
        device = Device()
        raw_begin, raw_end = device.begin_trace_capture, device.end_trace_capture
        for unused in range(5000):
            grid13.track(device)
            wp6.track(device)
        top = device.begin_trace_capture
        self.assertEqual(sorted(top.capture_trackers), ['grid13', 'wp6'], 'two wrappers, one per package, in whichever order they came')
        self.assertEqual(sorted(device.end_trace_capture.capture_trackers), ['grid13', 'wp6'])
        self.assertEqual(depth(top), 2)
        self.assertEqual(depth(device.end_trace_capture), 2)
        # the capture calls go through (a chain thousands deep raised RecursionError here)
        trace = device.begin_trace_capture(object(), cq_id=0)
        self.assertEqual((trace, device.began), (1, 1))
        device.end_trace_capture(object(), trace, cq_id=0)
        self.assertEqual(device.ended, 1)
        self.assertIsNot(top, raw_begin)
        self.assertIsNot(device.end_trace_capture, raw_end)

    def test_either_order_and_either_package_first_gives_the_same_two_wrappers(self):
        for first, second in ((grid13, wp6), (wp6, grid13)):
            with self.subTest(first=first.__name__):
                wp6.reset()
                grid13.reset()
                device = Device()
                for unused in range(2000):
                    first.track(device)
                    second.track(device)
                    first.in_capture(device)
                    second.in_capture(device)
                self.assertEqual(depth(device.begin_trace_capture), 2)
                self.assertEqual(sorted(device.begin_trace_capture.capture_trackers), ['grid13', 'wp6'])

    def test_both_packages_know_a_capture_is_open_through_the_one_chain(self):
        device = Device()
        for unused in range(10):
            grid13.track(device)
            wp6.track(device)
        self.assertFalse(grid13.in_capture())
        self.assertFalse(wp6.in_capture())
        trace = device.begin_trace_capture(object(), cq_id=0)
        self.assertTrue(grid13.in_capture())
        self.assertTrue(wp6.in_capture())
        device.end_trace_capture(object(), trace, cq_id=0)
        self.assertFalse(grid13.in_capture())
        self.assertFalse(wp6.in_capture())

    def test_a_wrapper_that_is_not_ours_on_top_costs_one_more_layer_not_a_growing_chain(self):
        """The parked-engine replay ledger puts its own begin_trace_capture on top while it is installed (and restores the saved one when it is removed)."""
        device = Device()
        grid13.track(device)
        wp6.track(device)
        saved = device.begin_trace_capture
        calls = []

        def ledger(*args, **kwargs):
            calls.append('ledger')
            return saved(*args, **kwargs)

        device.begin_trace_capture = ledger
        for unused in range(2000):
            grid13.track(device)
            wp6.track(device)
        # each package wrapped once more above the ledger, then both found themselves on top: bounded, not 2000 layers
        self.assertEqual(depth(device.begin_trace_capture), 2)
        device.begin_trace_capture(object(), cq_id=0)
        self.assertEqual(calls, ['ledger'])
        device.begin_trace_capture = saved
        grid13.track(device)
        wp6.track(device)
        self.assertEqual(device.begin_trace_capture, saved, 'both found themselves beneath the restored function')

    def test_a_device_without_the_capture_calls_is_left_alone(self):
        bare = SimpleNamespace()
        grid13.track(bare)
        wp6.track(bare)
        self.assertFalse(hasattr(bare, 'begin_trace_capture'))

    def test_the_old_marker_attributes_are_still_set(self):
        device = Device()
        grid13.track(device)
        wp6.track(device)
        top = device.begin_trace_capture
        self.assertTrue(getattr(top, 'tracked_by_wp6', False) or getattr(top, 'tracked_by_grid13', False))


if __name__ == '__main__':
    unittest.main()
