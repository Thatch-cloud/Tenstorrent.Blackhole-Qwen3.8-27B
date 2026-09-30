"""M2 item 1: the alternation policy, executed rather than string-matched.

fixtures/plugin_lane_scheduler.py is _local_prefill_intent and _negotiate_forced_mode
as cpu-probe 35665853903 dumped them from
/opt/qwen-fast-plugin/src/vllm_tt_plugin/lane_scheduler.py (sha f8e19e1907c05b24).

The policy is NOT invented here. docs/lever-N-prefill-decode-interleave.md section 3.3
gives it as pseudo-code and the graft is that pseudo-code with the plugin's names:

    intent = max(lane_intent)
    if intent == PREFILL and interleave and any_running_decode() and any_partial_prefill():
        if self._prefill_streak >= 1 and self._decode_credit < R:
            self._decode_credit += 1;  return DECODE_ONLY
        self._decode_credit = 0; self._prefill_streak += 1;  return PREFILL_ONLY
    self._prefill_streak = 1 if intent == PREFILL else 0;  self._decode_credit = 0
    return from_prefill_intent(intent)

Why it matters: the shipped _local_prefill_intent votes prefill whenever
has_partial_prefill, and _negotiate_forced_mode takes the max across lanes - its own
docstring says "if any lane wants to prefill, the whole step is prefill-only". So one
chunked prompt owns every step until it finishes and a decoding user freezes for all of
it. That is the 79.4 s decode stall, 31% of user-facing wall.

Every test here builds the patched class and runs steps through it. The previous
one-in-flight policy shipped wrong precisely because its tests checked arithmetic and
never drove the class (run 35690327326).
"""

import ast
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from lever_n_model_patch import MARKER_ALTERNATE, patch_lane_scheduler

FIXTURE = Path(__file__).parent / 'fixtures' / 'plugin_lane_scheduler.py'


class Mode(object):
    """Stands in for TTSchedulingMode: 1 means prefill-only, 0 decode-only."""

    @staticmethod
    def from_prefill_intent(intent):
        return 'PREFILL' if intent else 'DECODE'


def request(is_chunk):
    return type('R', (), {'is_prefill_chunk': is_chunk})()


def lane(running, waiting=(), skipped=()):
    return type('S', (), {'running': list(running), 'waiting': list(waiting),
                          'skipped_waiting': list(skipped)})()


def scheduler(lanes, per_lane_max=4):
    module = types.ModuleType('patched_lane_scheduler')
    module.__dict__['TTSchedulingMode'] = Mode
    module.__dict__['TTScheduler'] = object
    exec(compile(patch_lane_scheduler(FIXTURE.read_text(encoding='utf-8')),
                 'patched_lane_scheduler', 'exec'), module.__dict__)
    built = module.LaneScheduler()
    built.lanes = lanes
    built._per_lane_max = per_lane_max
    return built


def modes(lanes, steps=6, environ=None):
    built = scheduler(lanes)
    with patch.dict(os.environ, environ or {}, clear=True):
        return [built._negotiate_forced_mode() for _ in range(steps)]


class AlternationTests(unittest.TestCase):
    def test_a_partial_beside_a_decode_alternates(self):
        """The stall case, and the whole point. Shipped behaviour is all PREFILL."""
        self.assertEqual(modes([lane([request(True), request(False)])]),
                         ['PREFILL', 'DECODE', 'PREFILL', 'DECODE', 'PREFILL', 'DECODE'])

    def test_a_partial_with_nothing_decoding_still_takes_every_step(self):
        """Nothing to yield TO. Alternating here would just idle the device."""
        self.assertEqual(modes([lane([request(True)])]), ['PREFILL'] * 6)

    def test_decodes_alone_are_untouched(self):
        self.assertEqual(modes([lane([request(False)])]), ['DECODE'] * 6)

    def test_r_sets_how_many_decode_steps_follow_a_chunk(self):
        """TT_DECODE_STEPS_PER_PREFILL_CHUNK, as section 3.3 names it."""
        self.assertEqual(
            modes([lane([request(True), request(False)])], steps=9,
                  environ={'TT_DECODE_STEPS_PER_PREFILL_CHUNK': '2'}),
            ['PREFILL', 'DECODE', 'DECODE'] * 3)

    def test_the_gate_restores_stock_behaviour(self):
        """TT_PREFILL_DECODE_INTERLEAVE=0 must be indistinguishable from unpatched."""
        self.assertEqual(
            modes([lane([request(True), request(False)])],
                  environ={'TT_PREFILL_DECODE_INTERLEAVE': '0'}),
            ['PREFILL'] * 6)

    def test_a_malformed_r_falls_back_to_one_rather_than_raising(self):
        """A scheduler that raises on a bad env var takes the engine down mid-serve."""
        self.assertEqual(
            modes([lane([request(True), request(False)])], steps=4,
                  environ={'TT_DECODE_STEPS_PER_PREFILL_CHUNK': 'two'}),
            ['PREFILL', 'DECODE', 'PREFILL', 'DECODE'])

    def test_the_decode_lane_can_be_a_different_lane(self):
        """any_running_decode and any_partial_prefill are across lanes, not per lane."""
        self.assertEqual(modes([lane([request(True)]), lane([request(False)])], steps=4),
                         ['PREFILL', 'DECODE', 'PREFILL', 'DECODE'])

    def test_the_marker_is_emitted_from_inside_the_method(self):
        body = patch_lane_scheduler(FIXTURE.read_text(encoding='utf-8'))
        self.assertIn(MARKER_ALTERNATE, body)
        self.assertIn('f"' + MARKER_ALTERNATE, body, 'f-string, not brace-format')

    def test_local_prefill_intent_is_untouched(self):
        """The edit belongs in _negotiate_forced_mode. Changing the intent function
        would alter what every lane votes, not merely how the votes are combined."""
        before = FIXTURE.read_text(encoding='utf-8')
        after = patch_lane_scheduler(before)

        def body(text, name):
            lines = text.splitlines(keepends=True)
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return ''.join(lines[node.lineno - 1:node.end_lineno])
            raise AssertionError(name)

        self.assertEqual(body(before, '_local_prefill_intent'),
                         body(after, '_local_prefill_intent'))

    def test_patching_twice_raises(self):
        once = patch_lane_scheduler(FIXTURE.read_text(encoding='utf-8'))
        with self.assertRaisesRegex(ValueError, 'already carries'):
            patch_lane_scheduler(once)

    def test_a_source_without_the_anchor_raises(self):
        with self.assertRaises(ValueError):
            patch_lane_scheduler('class L:' + chr(10)
                                 + '    def _negotiate_forced_mode(self):' + chr(10)
                                 + '        return 1' + chr(10))


if __name__ == '__main__':
    unittest.main()
