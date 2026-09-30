"""M2 item 1 in the seat that actually runs: TTScheduler.schedule, executed not matched.

Run 35707860782 (v65) mounted the alternation graft on TTLaneCoordinator and it never
fired. Delivery was not the problem - the graft artifact's lane_scheduler.py differs from
its .orig by exactly the policy lines, and the sibling one-in-flight graft in the same
mounted file set printed its marker 54 times in that run. The class was never built:
platform.check_and_update_config selects TTLaneCoordinator only when
uses_tt_lane_coordinator() is true, and the run logged data_parallel_size=1 and loaded
vllm_tt_plugin.scheduler.TTScheduler.

So with no coordinator, set_forced_mode is never called, _forced_mode stays DEFAULT, and
the prefer-prefill rule in TTScheduler.schedule's default branch is the stall: decode
gets a step only when prefill schedules ZERO tokens, so a decoding user freezes for the
whole of another user's prefill. That is the 79.4 s, 31% of user-facing wall.

Same policy as section 3.3, same env names, same R, same gate, same marker - only the
seat changed. These tests DRIVE schedule() and record which sub-scheduler each step
called. _schedule_prefill_only / _schedule_decode_only are stubbed because they are the
machinery under the decision; the decision itself - the graft - runs for real.

test_lever_n_alternation.py still covers patch_lane_scheduler, which stays correct for a
lane-mode deployment. The two are mutually exclusive: in lane mode _forced_mode is set
every step, so this default branch is unreachable.
"""

import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from lever_n_model_patch import (MARKER_ALTERNATE, patch_scheduler_alternation,
                                 patch_scheduler_full)

FIXTURE = Path(__file__).parent / 'fixtures' / 'plugin_scheduler.py'


class Out(object):
    """Stands in for SchedulerOutput; only the token count is read by schedule()."""

    def __init__(self, tokens=1):
        self.total_num_scheduled_tokens = tokens


class Base(object):
    """Stands in for AsyncScheduler. schedule() is the no-pending-prefill path."""

    def schedule(self):
        return Out(1)


class Mode(object):
    """Stands in for TTSchedulingMode. The fixture is a reduced dump that uses the
    enum without defining it; the real scheduler.py defines exactly these three
    members at lines 17-28. Only identity is ever tested, so sentinels suffice."""

    DEFAULT = 'DEFAULT'
    PREFILL_ONLY = 'PREFILL_ONLY'
    DECODE_ONLY = 'DECODE_ONLY'


def stubs():
    """Everything the reduced fixture reads from its module globals."""
    return dict(
        AsyncScheduler=Base, SchedulerOutput=Out, Request=object,
        TTSchedulingMode=Mode, RequestQueue=list,
        create_request_queue=lambda policy: [], cast=lambda t, v: v,
        logger=types.SimpleNamespace(info=lambda *a, **k: None))


def request(is_chunk):
    return type('R', (), {'is_prefill_chunk': is_chunk})()


def build(running, waiting=(), prefill_tokens=1):
    module = types.ModuleType('patched_scheduler')
    module.__dict__.update(stubs())
    exec(compile(patch_scheduler_full(FIXTURE.read_text(encoding='utf-8')),
                 'patched_scheduler', 'exec'), module.__dict__)

    class Harness(module.TTScheduler):
        def __init__(self):
            self.running = list(running)
            self.waiting = list(waiting)
            self.skipped_waiting = []
            self.policy = None
            self._forced_mode = Mode.DEFAULT
            self.calls = []

        def _schedule_prefill_only(self):
            self.calls.append('PREFILL')
            return Out(prefill_tokens)

        def _schedule_decode_only(self):
            self.calls.append('DECODE')
            return Out(1)

        def _finalize_scheduler_output(self, result):
            return result

    return Harness()


def steps(running, waiting=(), n=6, environ=None, prefill_tokens=1):
    built = build(running, waiting, prefill_tokens)
    with patch.dict(os.environ, environ or {}, clear=True):
        for _ in range(n):
            built.schedule()
    return built.calls


class AlternationInTTScheduler(unittest.TestCase):
    def test_a_partial_beside_a_decode_alternates(self):
        """The stall case, and the whole point. Shipped behaviour is PREFILL every step."""
        self.assertEqual(steps([request(True), request(False)]),
                         ['PREFILL', 'DECODE', 'PREFILL', 'DECODE', 'PREFILL', 'DECODE'])

    def test_the_shipped_policy_really_does_starve_decode(self):
        """The negative control: unpatched, the same inputs never yield a decode step.

        Without this, an 'alternates' assertion proves nothing about what changed.
        """
        module = types.ModuleType('stock_scheduler')
        module.__dict__.update(stubs())
        exec(compile(FIXTURE.read_text(encoding='utf-8'), 'stock_scheduler', 'exec'),
             module.__dict__)

        class Harness(module.TTScheduler):
            def __init__(self):
                self.running = [request(True), request(False)]
                self.waiting = []
                self.skipped_waiting = []
                self._forced_mode = Mode.DEFAULT
                self.calls = []

            def _schedule_prefill_only(self):
                self.calls.append('PREFILL')
                return Out(1)

            def _schedule_decode_only(self):
                self.calls.append('DECODE')
                return Out(1)

            def _finalize_scheduler_output(self, result):
                return result

        built = Harness()
        for _ in range(6):
            built.schedule()
        self.assertEqual(built.calls, ['PREFILL'] * 6)

    def test_a_partial_with_nothing_decoding_still_takes_every_step(self):
        """Nothing to yield TO. Alternating here would just idle the device."""
        self.assertEqual(steps([request(True)]), ['PREFILL'] * 6)

    def test_a_waiting_prompt_does_not_trigger_a_yield(self):
        """has_pending_prefill is true for a fresh prompt too, but yielding there
        delays admission and worsens the TTFT staircase. Section 3.3's condition is a
        PARTIAL prefill, and that is deliberately what this tests."""
        self.assertEqual(steps([request(False)], waiting=[request(False)]),
                         ['PREFILL'] * 6)

    def test_r_sets_how_many_decode_steps_follow_a_chunk(self):
        """TT_DECODE_STEPS_PER_PREFILL_CHUNK, as section 3.3 names it."""
        self.assertEqual(
            steps([request(True), request(False)], n=9,
                  environ={'TT_DECODE_STEPS_PER_PREFILL_CHUNK': '2'}),
            ['PREFILL', 'DECODE', 'DECODE'] * 3)

    def test_the_gate_restores_stock_behaviour(self):
        """TT_PREFILL_DECODE_INTERLEAVE=0 must be indistinguishable from unpatched."""
        self.assertEqual(
            steps([request(True), request(False)],
                  environ={'TT_PREFILL_DECODE_INTERLEAVE': '0'}),
            ['PREFILL'] * 6)

    def test_a_malformed_r_falls_back_to_one_rather_than_raising(self):
        """A scheduler that raises on a bad env var takes the engine down mid-serve."""
        self.assertEqual(
            steps([request(True), request(False)], n=4,
                  environ={'TT_DECODE_STEPS_PER_PREFILL_CHUNK': 'two'}),
            ['PREFILL', 'DECODE', 'PREFILL', 'DECODE'])

    def test_the_zero_token_fallback_still_works(self):
        """The shipped escape hatch - prefill cannot move, so decode runs - must
        survive the graft. Here every prefill schedules nothing."""
        self.assertEqual(
            steps([request(True), request(False)], n=4, prefill_tokens=0),
            ['PREFILL', 'DECODE', 'DECODE', 'PREFILL', 'DECODE', 'DECODE'])

    def test_decodes_alone_never_reach_the_branch(self):
        self.assertEqual(steps([request(False)]), [])

    def test_the_marker_is_emitted_from_inside_schedule(self):
        body = patch_scheduler_alternation(FIXTURE.read_text(encoding='utf-8'))
        self.assertIn(MARKER_ALTERNATE, body)
        self.assertIn('f"' + MARKER_ALTERNATE, body, 'f-string, not brace-format')

    def test_patching_twice_raises(self):
        once = patch_scheduler_alternation(FIXTURE.read_text(encoding='utf-8'))
        with self.assertRaisesRegex(ValueError, 'already carries'):
            patch_scheduler_alternation(once)

    def test_a_source_without_the_anchor_raises(self):
        with self.assertRaisesRegex(ValueError, 'anchor matched 0 times'):
            patch_scheduler_alternation('class TTScheduler:' + chr(10)
                                        + '    def schedule(self):' + chr(10)
                                        + '        return 1' + chr(10))

    def test_the_graft_table_applies_BOTH_scheduler_patches(self):
        """One function per file in the table, so mapping only patch_scheduler is how
        the alternation would silently go missing - which is this run's whole lesson."""
        both = patch_scheduler_full(FIXTURE.read_text(encoding='utf-8'))
        self.assertIn(MARKER_ALTERNATE, both)
        self.assertIn('[PINDIAG] m2 one-in-flight:', both)


if __name__ == '__main__':
    unittest.main()
