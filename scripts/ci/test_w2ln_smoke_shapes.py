"""The combined window's smoke shapes (c2_serving_smoke): cold2_254k, run for real against test_levern_smoke's fake server, and the rule that no shape
sends more concurrent requests than the eight seats of the profile it runs on.

A request past max-num-seqs waits for a seat, and its time to first token then measures somebody else's token budget: the first design of cold2_254k
(seven decoders and two cold arrivals, nine requests on eight seats) would have read as a stall.
"""

import ast
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check as check  # noqa: E402
from test_levern_smoke import SMOKE, SmokeRuns  # noqa: E402

SEATS = 8


def constants():
    """The smoke's top-level simple assignments (numbers, names, sums, dicts of them), evaluated in order, without importing the script (it runs on import)."""
    namespace = {}
    for node in ast.parse(SMOKE.read_text(encoding='utf-8')).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if any(isinstance(inner, (ast.Call, ast.Attribute, ast.Subscript, ast.Lambda, ast.ListComp)) for inner in ast.walk(node.value)):
            continue
        try:
            exec(compile(ast.Module([node], []), str(SMOKE), 'exec'), namespace)
        except NameError:
            continue
    return namespace


class SeatCountTests(unittest.TestCase):
    def test_every_declared_shape_fits_the_eight_seats(self):
        found = constants()
        shapes = found['SHAPE_SEATS']
        self.assertIn('cold2_254k', shapes)
        for name, seats in shapes.items():
            with self.subTest(name=name):
                self.assertLessEqual(seats, SEATS)

    def test_cold2_is_six_decoders_and_two_arrivals(self):
        found = constants()
        self.assertEqual((found['COLD2_DECODERS'], found['COLD2_ARRIVALS']), (6, 2))
        self.assertEqual(found['COLD2_DECODERS'] + found['COLD2_ARRIVALS'], SEATS)
        self.assertGreaterEqual(found['COLD2_BUDGET'], found['STALL_SHORT_BUDGET'], 'the decoders outlast both prefills as the stall shape\'s do')

    def test_the_shape_is_registered_as_an_opt_in_test(self):
        source = SMOKE.read_text(encoding='utf-8')
        self.assertIn("if ONLY and 'cold2_254k' in ONLY:", source)


class ColdTwoTests(SmokeRuns):
    def test_two_simultaneous_cold_arrivals_beside_six_decoders(self):
        results, chats, _completions, _tokenized, _ = self.ran('cold2_254k')
        entry = results['cold2_254k']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'users'})
        self.assertEqual(entry['fit']['targets'], [253920, 253920])
        self.assertEqual(len(chats), 8, 'six decoders and two arrivals: never a ninth request')
        decoders = [chat for chat in chats if chat['body'].get('ignore_eos')]
        self.assertEqual(len(decoders), 6)
        self.assertEqual({chat['body']['max_tokens'] for chat in decoders}, {6000})
        self.assertEqual(len(entry['arrivals']), 2)
        self.assertEqual(len(entry['seat_windows']), 6)
        self.assertEqual(len(entry['seat_gaps']), 6)
        for key in ('arrival_started_at', 'longest_gap_s', 'window'):
            self.assertIn(key, entry)
        for arrival in entry['arrivals']:
            self.assertIsNotNone(arrival['ttft_s'])
        self.assertEqual(check.smoke_problems(results), [])


class ColdTwoCheckTests(unittest.TestCase):
    def good(self):
        stream = dict(tokens=40, text='fine words here', finish='length')
        return dict(users=[dict(stream) for _ in range(8)], arrivals=[dict(ttft_s=90.0, error=None), dict(ttft_s=110.0, error=None)],
                    seat_gaps=[dict(seat=index, longest_gap_s=4.0, error=None) for index in range(6)])

    def test_a_clean_result_has_no_problem_and_no_test_has_none(self):
        self.assertEqual(check.cold2_problems(self.good()), [])
        self.assertEqual(check.cold2_problems(None), [])
        self.assertEqual(check.smoke_problems({'cold2_254k': self.good()}), [])

    def test_a_failed_arrival_an_errored_test_a_stalled_decoder_and_missing_gaps_are_each_a_problem(self):
        broken = self.good()
        broken['arrivals'][1] = dict(ttft_s=None, error='timed out')
        self.assertEqual(len(check.cold2_problems(broken)), 2, 'the error and the missing time to first token')
        self.assertTrue(check.smoke_problems({'cold2_254k': broken}))
        self.assertTrue(check.smoke_problems({'cold2_254k': dict(error='boom')}))
        stalled = self.good()
        stalled['users'][2] = dict(error='read timed out')
        self.assertTrue(any('user 2' in text for text in check.cold2_problems(stalled)))
        quiet = self.good()
        quiet['seat_gaps'][3] = dict(seat=3, longest_gap_s=None, error=None)
        self.assertTrue(any('seat 3' in text for text in check.cold2_problems(quiet)))
        self.assertTrue(check.cold2_problems(dict(self.good(), seat_gaps=[])))
        one = self.good()
        one['arrivals'] = one['arrivals'][:1]
        self.assertTrue(check.cold2_problems(one))
        empty = self.good()
        empty['users'][0] = dict(tokens=0, finish='length')
        self.assertTrue(check.cold2_problems(empty))


class MultiStampTests(unittest.TestCase):
    """An arm that runs the multi-user SDPA launch (G16 0x21) is outside the 262k evidence: the waiver is gone from the window profiles, so the attach's own UNQUALIFIED
    line must stamp every summary the smoke check and the two gates write (a result on a multi profile must never read as qualified 262k evidence)."""
    LINE = '2026-10-07 12:00:00.000 | INFO | x [PINDIAG] tp4 sdpa multi UNQUALIFIED capacity=262144 (G16 flags 0x21 is outside the 262k evidence: gate only, never traffic)'
    WAIVER = '[PINDIAG] 262k evidence WAIVED (gate-only)'

    def test_the_stamp_names_what_the_log_shows(self):
        self.assertEqual(check.unqualified_stamp(''), '')
        self.assertEqual(check.unqualified_stamp(self.LINE), check.MULTI_STAMP)
        self.assertEqual(check.unqualified_stamp(self.WAIVER), check.WAIVER_STAMP)
        self.assertEqual(check.unqualified_stamp(self.LINE + chr(10) + self.WAIVER), check.WAIVER_STAMP + ' + ' + check.MULTI_STAMP)
        self.assertIn('multi', check.MULTI_STAMP)

    def test_the_smoke_check_carries_it_in_its_facts(self):
        problems, facts = check.check('', self.LINE, False, env={'QWEN_FAST_TP4_SDPA': 'multi'})
        self.assertEqual(facts.get('unqualified'), check.MULTI_STAMP)
        problems, facts = check.check('', '', False, env={})
        self.assertNotIn('unqualified', facts)

    def test_the_serving_gate_stamps_its_summary_and_the_prefix_gate_its_arm_and_summary(self):
        import c2_serving_gate as serving
        from types import SimpleNamespace
        self.assertEqual(serving.unqualified_stamp(SimpleNamespace(waived=False, multi_unqualified=True)), check.MULTI_STAMP)
        self.assertEqual(serving.unqualified_stamp(SimpleNamespace(waived=True, multi_unqualified=True)), check.WAIVER_STAMP + ' + ' + check.MULTI_STAMP)
        source = (HERE / 'c2_serving_gate.py').read_text(encoding='utf-8')
        self.assertIn("if log_text and c2_smoke_check.SDPA_MULTI_UNQUALIFIED in log_text:", source)
        import tempfile
        from unittest import mock
        import test_c2_prefix_gate as prefix_fixture
        extra = self.LINE

        def lines(log):
            return list(log.engine.lines) + [extra]

        with tempfile.TemporaryDirectory() as results, mock.patch.object(prefix_fixture.fakes.FakeLog, 'lines', lines):
            _, _, runner = prefix_fixture.run_plan('bringup', results)
        self.assertTrue(runner.multi_unqualified)
        self.assertFalse(runner.waived)
        self.assertEqual(serving.unqualified_stamp(runner), check.MULTI_STAMP)
        self.assertTrue(all(result.get('unqualified') == check.MULTI_STAMP for result in runner.arms.values()))


if __name__ == '__main__':
    unittest.main()
