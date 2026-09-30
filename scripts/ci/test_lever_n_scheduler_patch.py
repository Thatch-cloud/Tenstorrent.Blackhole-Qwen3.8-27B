"""The M2 scheduler patcher must edit the right method, or fail loudly.

These fixtures are trimmed but line-for-line faithful, in the anchors this module
touches, to the real sources CPU probe run 35535280599 dumped from
/opt/qwen-fast-plugin/src/vllm_tt_plugin (scheduler.py sha256=a1bd6257d3a14c90,
lane_scheduler.py sha256=f8e19e1907c05b24). Each also carries a decoy with the
same ambiguity risk lever_n_model_patch's tests guard against: a second method
with a similarly-shaped body, so a text-scoped patch could silently edit the
wrong one if the AST scoping ever regressed to a whole-file replace.
"""

import ast
import unittest

from lever_n_scheduler_patch import (function_span, patch_negotiate_forced_mode,
                                     patch_one_in_flight, replace_once)

# Trimmed stand-in for lane_scheduler.py. _decoy_negotiate_forced_mode carries the
# same two-line body as the real method under a different name, so a patch that
# matched by text alone (not by AST method-name scoping) could hit it instead.
LANE_SCHEDULER = '''from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode

logger = init_tt_logger(__name__)


class TTLaneCoordinator:
    """Single-process multi-lane scheduler for TT gathered-batch execution."""

    def _local_prefill_intent(self, sched):
        has_waiting = bool(sched.waiting) or bool(sched.skipped_waiting)
        has_running = bool(sched.running)
        has_partial_prefill = any(r.is_prefill_chunk for r in sched.running)
        has_capacity = len(sched.running) < self._per_lane_max
        return int(
            has_partial_prefill or (has_waiting and ((not has_running) or has_capacity))
        )

    def _negotiate_forced_mode(self):
        """Pick the single mode (prefill- or decode-only) all lanes will run."""
        intent = max(self._local_prefill_intent(sched) for sched in self.lanes)
        return TTSchedulingMode.from_prefill_intent(intent)

    def _schedule_all_lanes(self, forced_mode):
        lane_outputs = []
        for sched in self.lanes:
            sched.set_forced_mode(forced_mode)
            lane_outputs.append(sched.schedule())
        return lane_outputs

    def schedule(self, throttle_prefills=False):
        forced_mode = self._negotiate_forced_mode()
        lane_outputs = self._schedule_all_lanes(forced_mode)
        return lane_outputs


class _DecoyCoordinator:
    def _decoy_negotiate_forced_mode(self):
        intent = max(self._local_prefill_intent(sched) for sched in self.lanes)
        return TTSchedulingMode.from_prefill_intent(intent)
'''

# Trimmed stand-in for scheduler.py. decoy_prefill_only carries the same
# try/finally shape as the real method under a different name.
SCHEDULER = '''from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue


class TTScheduler:
    """Scheduler for the TT (Tenstorrent) platform."""

    def decoy_prefill_only(self):
        saved_max = self.max_num_running_reqs
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
        return result

    def _schedule_prefill_only(self):
        """Schedule prefill work: waiting requests and partial continuations."""
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_max = self.max_num_running_reqs
        self.running = partial_prefills
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
        return result

    def _schedule_decode_only(self):
        """Schedule only running decode requests."""
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_waiting = self.waiting
        saved_skipped = getattr(self, "skipped_waiting", None)
        self.waiting = create_request_queue(self.policy)
        if saved_skipped is not None:
            self.skipped_waiting = create_request_queue(self.policy)
        if partial_prefills:
            self.running = [r for r in self.running if not r.is_prefill_chunk]
        try:
            result = super().schedule()
        finally:
            if self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            if saved_skipped is not None:
                if self.skipped_waiting:
                    saved_skipped.prepend_requests(self.skipped_waiting)
                self.skipped_waiting = saved_skipped
            self.waiting = saved_waiting
            if partial_prefills:
                self.running.extend(partial_prefills)
        return result
'''

# The real source imports create_request_queue at module scope but does not use
# the literal method body decoy_prefill_only has; SCHEDULER above already
# satisfies patch_one_in_flight's own "must import create_request_queue" check.


class ScopingTests(unittest.TestCase):
    def test_span_isolates_negotiate_forced_mode(self):
        start, end = function_span(LANE_SCHEDULER, '_negotiate_forced_mode')
        region = ''.join(LANE_SCHEDULER.splitlines(keepends=True)[start:end])
        self.assertIn('Pick the single mode', region)
        self.assertNotIn('_schedule_all_lanes', region)

    def test_decoy_negotiate_method_is_untouched(self):
        out = patch_negotiate_forced_mode(LANE_SCHEDULER)
        # The real method gained the alternation branch...
        self.assertEqual(out.count('_m2_alternation_step'), 2)
        # ...but the decoy, under a different name in a different class, did not.
        decoy_start = out.index('class _DecoyCoordinator')
        self.assertNotIn('_m2_alternation_step', out[decoy_start:])
        self.assertIn(
            'intent = max(self._local_prefill_intent(sched) for sched in self.lanes)\n'
            '        return TTSchedulingMode.from_prefill_intent(intent)',
            out[decoy_start:],
        )

    def test_span_isolates_schedule_prefill_only(self):
        start, end = function_span(SCHEDULER, '_schedule_prefill_only')
        region = ''.join(SCHEDULER.splitlines(keepends=True)[start:end])
        self.assertIn('waiting requests and partial continuations', region)
        self.assertNotIn('_schedule_decode_only', region)

    def test_decoy_prefill_method_is_untouched(self):
        out = patch_one_in_flight(SCHEDULER)
        self.assertEqual(out.count('create_request_queue(self.policy)'), 4)  # 2 real + ...
        decoy_start = out.index('def decoy_prefill_only')
        decoy_end = out.index('def _schedule_prefill_only')
        decoy_region = out[decoy_start:decoy_end]
        self.assertNotIn('create_request_queue', decoy_region)
        self.assertIn(
            '        finally:\n'
            '            self.running.extend(pure_decodes)\n'
            '            self.max_num_running_reqs = saved_max\n'
            '        return result',
            decoy_region,
        )

    def test_replace_once_refuses_an_ambiguous_region(self):
        lines = SCHEDULER.splitlines(keepends=True)
        with self.assertRaises(ValueError):
            replace_once(lines, (0, len(lines)), 'saved_max = self.max_num_running_reqs',
                        'x', 'ambiguous')

    def test_missing_method_is_an_error(self):
        with self.assertRaises(ValueError):
            function_span(LANE_SCHEDULER, 'not_a_method')


class AlternationTests(unittest.TestCase):
    def test_output_is_valid_python(self):
        out = patch_negotiate_forced_mode(LANE_SCHEDULER)
        ast.parse(out)

    def test_alternation_gated_on_partial_prefill_and_running_decode(self):
        out = patch_negotiate_forced_mode(LANE_SCHEDULER)
        for probe in ('has_partial_prefill = any(', 'has_running_decode = any(',
                      'if has_partial_prefill and has_running_decode:',
                      'r.is_prefill_chunk for r in sched.running', 'DECODE_ONLY'):
            self.assertIn(probe, out, probe)

    def test_local_prefill_intent_is_not_touched(self):
        """Change 1a is scoped to the negotiation, not the per-lane vote it reads."""
        out = patch_negotiate_forced_mode(LANE_SCHEDULER)
        start, _ = function_span(out, '_local_prefill_intent')
        _, end = function_span(LANE_SCHEDULER, '_local_prefill_intent')
        before = LANE_SCHEDULER.splitlines(keepends=True)[start:end]
        after = out.splitlines(keepends=True)[start:end]
        self.assertEqual(before, after)

    def test_schedule_all_lanes_and_schedule_are_not_touched(self):
        """The PREFILL_ONLY -> DECODE_ONLY fallback and its bookkeeping carry live
        in schedule()/_schedule_all_lanes, not in the negotiation; M2 must not
        move or duplicate that logic."""
        out = patch_negotiate_forced_mode(LANE_SCHEDULER)
        for name in ('_schedule_all_lanes', 'schedule'):
            before_span = function_span(LANE_SCHEDULER, name)
            after_span = function_span(out, name)
            before = ''.join(LANE_SCHEDULER.splitlines(keepends=True)[slice(*before_span)])
            after = ''.join(out.splitlines(keepends=True)[slice(*after_span)])
            self.assertEqual(before, after, name)

    def test_patching_twice_raises(self):
        out = patch_negotiate_forced_mode(LANE_SCHEDULER)
        with self.assertRaises(ValueError):
            patch_negotiate_forced_mode(out)

    def test_period_is_one_in_two(self):
        """The chosen ratio: every other qualifying step forces decode-only."""
        out = patch_negotiate_forced_mode(LANE_SCHEDULER)
        self.assertIn('step % 2 == 1', out)

    def test_logs_a_positive_control_marker(self):
        """The gate greps server logs for this to confirm alternation actually
        fired, the same role [M1] prefill path markers play for M1."""
        out = patch_negotiate_forced_mode(LANE_SCHEDULER)
        self.assertIn('[M2] alternation:', out)


class OneInFlightTests(unittest.TestCase):
    def test_output_is_valid_python(self):
        out = patch_one_in_flight(SCHEDULER)
        ast.parse(out)

    def test_hides_waiting_and_skipped_waiting_only_when_partial_prefills(self):
        out = patch_one_in_flight(SCHEDULER)
        for probe in ('if partial_prefills:\n            self.waiting = create_request_queue(self.policy)',
                      'saved_skipped = getattr(self, "skipped_waiting", None)',
                      'saved_waiting.prepend_requests(self.waiting)',
                      'saved_skipped.prepend_requests(self.skipped_waiting)'):
            self.assertIn(probe, out, probe)

    def test_uses_the_request_queue_factory_not_a_raw_list_or_none(self):
        """self.waiting is a RequestQueue; swapping in [] or None would break the
        base scheduler's waiting-loop calls on it (e.g. peek_request)."""
        out = patch_one_in_flight(SCHEDULER)
        start, end = function_span(out, '_schedule_prefill_only')
        region = ''.join(out.splitlines(keepends=True)[start:end])
        self.assertNotIn('self.waiting = []', region)
        self.assertNotIn('self.waiting = None', region)
        self.assertEqual(region.count('create_request_queue(self.policy)'), 2)

    def test_decode_only_hiding_idiom_is_untouched(self):
        """_schedule_decode_only is the template this copies from; it must be
        byte-identical after patching _schedule_prefill_only."""
        out = patch_one_in_flight(SCHEDULER)
        before_span = function_span(SCHEDULER, '_schedule_decode_only')
        after_span = function_span(out, '_schedule_decode_only')
        before = ''.join(SCHEDULER.splitlines(keepends=True)[slice(*before_span)])
        after = ''.join(out.splitlines(keepends=True)[slice(*after_span)])
        self.assertEqual(before, after)

    def test_missing_import_is_rejected(self):
        stripped = SCHEDULER.replace(
            'from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue\n',
            '')
        with self.assertRaises(ValueError):
            patch_one_in_flight(stripped)

    def test_patching_twice_raises(self):
        out = patch_one_in_flight(SCHEDULER)
        with self.assertRaises(ValueError):
            patch_one_in_flight(out)


if __name__ == '__main__':
    unittest.main()
