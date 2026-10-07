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


if __name__ == '__main__':
    unittest.main()
