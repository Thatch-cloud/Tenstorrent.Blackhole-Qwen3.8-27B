"""Lever N at TP4: the prefill capture's side (dflash_prefill_window.PrefillWindowCapture.wrap_slots reads what the route reports).

A batched prefill call that is not the last step of a split prompt writes no decode slot, and levern_route says so through a model attribute. The capture
reads and clears it after every batched call (segment_wrote_slot), and the lifecycle's displacement after a continuation is keyed on it. A model that
never sets it (every profile without QWEN_FAST_LEVER_N) gives True and the stock behaviour. The model and capture doubles are
test_dflash_prefill_window's."""

import unittest
from unittest.mock import patch

import dflash_prefill_window as window
import levern_route as route
import test_dflash_prefill_window as base
from dflash_prefill_window import PrefillWindowCapture


def helper():
    return base.PrefillWindowTests('test_absolute_frontier_and_window_are_distinct')


class CaptureReportTests(unittest.TestCase):
    def rig(self, wrote):
        """A 6144-token prompt in three engine steps; `wrote[i]` is what the route reports after step i (None: nothing reported)."""
        owner = helper()
        operations, (model, calls) = owner.operations(), owner.plugin_model()
        capture = PrefillWindowCapture(operations, model, 6144, (1, 3))
        for name in ('prefill_paged_slots', 'prefill_paged_slots_range'):
            original = getattr(model, name)

            def reporting(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                value = wrote[len(calls) - 1]
                if value is not None:
                    vars(model)[route.WROTE_ATTR] = value
                return result

            setattr(model, name, reporting)
        return owner, capture, model

    def run_steps(self, owner, capture, model, slots=(1, 1, 1)):
        seen = []
        with owner.storage_addresses(), patch('dflash_prefill_window._log'):
            for index, slot in enumerate(slots):
                with capture.segment():
                    self.assertIsNone(capture.segment_wrote_slot, 'reset when the segment opens')
                    owner.plugin_chunk(model, index, slot)
                seen.append(capture.segment_wrote_slot)
        return seen

    def test_the_attribute_name_is_pinned_to_the_route_s(self):
        self.assertEqual(window.LEVERN_WROTE_ATTR, route.WROTE_ATTR)
        self.assertIn(route.ENTRY, window.BATCHED_PREFILL_ENTRIES)

    def test_a_model_that_reports_nothing_gives_the_stock_true(self):
        owner, capture, model = self.rig([None, None, None])
        self.assertEqual(self.run_steps(owner, capture, model), [True, True, True])
        capture.close()

    def test_an_intermediate_step_reports_false_and_the_last_true(self):
        owner, capture, model = self.rig([False, False, True])
        self.assertEqual(self.run_steps(owner, capture, model), [False, False, True])
        self.assertNotIn(route.WROTE_ATTR, vars(model), 'the capture consumes the report')
        self.assertTrue(capture.complete)
        capture.close()

    def test_a_stale_report_cannot_leak_into_the_next_call(self):
        owner, capture, model = self.rig([None, None, None])
        vars(model)[route.WROTE_ATTR] = False                 # left by an earlier prompt's route call
        self.assertEqual(self.run_steps(owner, capture, model), [True, True, True])
        capture.close()

    def test_the_lifecycle_keeps_the_resident_engine_after_a_step_that_wrote_nothing(self):
        from serving_lifecycle import FastServingLifecycle
        from types import SimpleNamespace

        owner, capture, model = self.rig([False, False, True])
        lifecycle = SimpleNamespace(capture=capture, request_id='req')
        displaced = []
        with owner.storage_addresses(), patch('dflash_prefill_window._log'), \
                patch('serving_lifecycle.note_prefill', side_effect=lambda: displaced.append(capture.segments)):
            for index in range(3):
                with capture.segment():
                    owner.plugin_chunk(model, index, 0)       # every step names slot 0, the working row
                if index:
                    FastServingLifecycle._displace_after_continuation(lifecycle)
        self.assertEqual(displaced, [3], 'only the step that wrote slot 0 displaces')
        capture.close()

    def test_without_the_report_the_same_sequence_displaces_after_each_slot_zero_step(self):
        from serving_lifecycle import FastServingLifecycle
        from types import SimpleNamespace

        owner, capture, model = self.rig([None, None, None])
        lifecycle = SimpleNamespace(capture=capture, request_id='req')
        displaced = []
        with owner.storage_addresses(), patch('dflash_prefill_window._log'), \
                patch('serving_lifecycle.note_prefill', side_effect=lambda: displaced.append(capture.segments)):
            for index in range(3):
                with capture.segment():
                    owner.plugin_chunk(model, index, 0)
                if index:
                    FastServingLifecycle._displace_after_continuation(lifecycle)
        self.assertEqual(displaced, [2, 3])
        capture.close()


if __name__ == '__main__':
    unittest.main()
