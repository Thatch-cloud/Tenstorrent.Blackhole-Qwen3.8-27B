"""w2ln_timing_compare: the round-time pairing of the combined window's timed jobs, on synthetic logs."""

import json
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import w2ln_timing_compare as t  # noqa: E402

ORIGIN = datetime(2026, 10, 7, 12, 0, 0)


def stamp(seconds):
    return (ORIGIN + timedelta(seconds=seconds)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def execute(at, new=0, live=8):
    return '%s | INFO | x [PHASE] execute total=%d new=%d cached=%d spec=0' % (stamp(at), new + live, new, live)


def packed(position, users=8):
    return ['[PACKED] request=r%d segment=%d position=%d prefix=3 emitted=3 predictions=[1]' % (index, index, position) for index in range(users)]


def log(rounds, prefill_at=(), live=8):
    """rounds: [(position, seconds)] back-to-back eight-live decode rounds; prefill_at: round indexes that carry a Lever N prefill step line."""
    lines, clock = [], 0.0
    for index, (position, seconds) in enumerate(rounds):
        lines.append(execute(clock, live=live))
        lines += packed(position, live)
        if index in prefill_at:
            lines.append('[PINDIAG] lever N step n=1 kind=prefill seats=7 req=a start=0 tokens=2048 end=2048 prompt=9000 final=0 reason=x prev=decode:1ms')
        clock += seconds
    lines.append(execute(clock, live=live))
    return '\n'.join(lines)


def steady(count, seconds, start=30000, step=40):
    return [(start + index * step, seconds) for index in range(count)]


class RoundTests(unittest.TestCase):
    def test_a_clean_log_yields_its_rounds_and_the_last_has_no_successor(self):
        rounds, dropped = t.timed_rounds(log(steady(10, 0.2)))
        self.assertEqual(len(rounds), 10)
        self.assertEqual(dropped, 0)
        self.assertAlmostEqual(rounds[0]['seconds'], 0.2, places=3)
        self.assertEqual(rounds[0]['mean_position'], 30000)

    def test_rounds_beside_a_lever_n_prefill_step_are_dropped_with_the_one_after(self):
        rounds, dropped = t.timed_rounds(log(steady(10, 0.2), prefill_at=(4,)))
        self.assertEqual((len(rounds), dropped), (8, 2))

    def test_only_fully_packed_rounds_of_the_asked_live_count_count(self):
        text = log(steady(5, 0.2), live=7)
        self.assertEqual(t.timed_rounds(text, 8)[0], [])
        self.assertEqual(len(t.timed_rounds(text, 7)[0]), 5)
        partial = log(steady(5, 0.2)).replace('[PACKED] request=r7 segment=7', '[X] r7')
        self.assertEqual(t.timed_rounds(partial)[0], [])


class PairTests(unittest.TestCase):
    def test_a_faster_arm_reads_a_negative_delta_matched_by_context_bucket_not_by_position(self):
        a = log(steady(300, 0.200, start=30000))
        b = log(steady(300, 0.190, start=30011, step=37))       # other exact positions, the same buckets
        result = t.compare(a, b)
        self.assertEqual(result['verdict'], 'MEASURED')
        self.assertGreaterEqual(result['matched'], 200)
        self.assertAlmostEqual(result['delta_ms'], -10.0, places=0)
        self.assertEqual(result['a_median_ms'], 200.0)
        self.assertLess(result['ratio_b_over_a'], 1.0)

    def test_too_few_matched_rounds_is_void_never_a_delta(self):
        result = t.compare(log(steady(50, 0.2)), log(steady(50, 0.19)))
        self.assertEqual(result['verdict'], 'VOID')
        self.assertIn('pre-registered', result['reason'])
        self.assertNotIn('delta_ms', result)

    def test_arms_that_never_share_a_context_bucket_are_void(self):
        result = t.compare(log(steady(300, 0.2, start=30000)), log(steady(300, 0.2, start=120000)))
        self.assertEqual((result['verdict'], result['matched']), ('VOID', 0))

    def test_a_lever_n_arm_loses_the_rounds_beside_its_prefill_steps_and_keeps_the_rest(self):
        b = log(steady(300, 0.2), prefill_at=set(range(0, 300, 10)))
        result = t.compare(log(steady(300, 0.2)), b)
        self.assertEqual(result['dropped_for_prefill']['b'], 60)
        self.assertEqual(result['rounds_b'], 240)

    def test_the_floor_is_the_spread_of_the_a_medians(self):
        floor = t.drift_floor_ms([log(steady(20, 0.200)), log(steady(20, 0.204)), log(steady(20, 0.198))])
        self.assertEqual(floor, 6.0)
        self.assertIsNone(t.drift_floor_ms([log(steady(20, 0.2))]))


class JudgeTests(unittest.TestCase):
    def pair(self, delta, verdict='MEASURED'):
        return dict(verdict=verdict, delta_ms=delta)

    def test_go_needs_all_three_pairs_negative_and_a_gain_past_the_floor_and_one_percent(self):
        self.assertEqual(t.judge([self.pair(-10), self.pair(-9), self.pair(-12)], 3.0, 200.0)['verdict'], 'GO')
        small = t.judge([self.pair(-1.5)] * 3, 1.0, 200.0)
        self.assertEqual(small['verdict'], 'INCONCLUSIVE', 'a gain under 1% of the round is inside the noise')
        floor = t.judge([self.pair(-5)] * 3, 6.0, 200.0)
        self.assertEqual(floor['verdict'], 'INCONCLUSIVE', 'a gain under the A-to-A drift is inside the noise')
        self.assertEqual(t.judge([self.pair(-10), self.pair(-9), self.pair(2)], 3.0, 200.0)['verdict'], 'INCONCLUSIVE', 'one slower pair is no GO and not yet a NO-GO')

    def test_no_go_needs_two_positive_pairs_of_the_three(self):
        self.assertEqual(t.judge([self.pair(4), self.pair(5), self.pair(-1)], 1.0, 200.0)['verdict'], 'NO-GO')

    def test_a_void_pair_is_never_a_vote(self):
        pairs = [self.pair(-10), self.pair(-10), self.pair(0, 'VOID')]
        self.assertEqual(t.judge(pairs, 1.0, 200.0)['verdict'], 'INCONCLUSIVE')
        self.assertEqual(t.judge(pairs[:2] + [self.pair(-10)], 1.0, 200.0)['verdict'], 'GO')
        self.assertEqual(t.judge([self.pair(4), self.pair(5), self.pair(0, 'VOID')], 1.0, 200.0)['verdict'], 'NO-GO', 'two measured slower pairs already decide')

    def test_load_above_the_maximum_voids_and_no_samples_cannot_be_called_clean(self):
        self.assertEqual(t.load_void('[LOAD] 1 1.5\n[LOAD] 61 2.5', 4.0), (False, None))
        self.assertEqual(t.load_void('[LOAD] 1 1.5\n[LOAD] 61 9.5', 4.0)[0], True)
        self.assertIsNone(t.load_void('nothing', 4.0)[0])


class TextTests(unittest.TestCase):
    def smoke(self, hashes):
        return {'concurrent8_code_32k': {'users': [{'content_sha256': h, 'reasoning_sha256': 'r'} for h in hashes]}, 'warmup': {'text': 'x'}}

    def test_equal_hashes_have_no_mismatch_and_a_difference_names_the_user(self):
        self.assertEqual(t.texts_equal(self.smoke('abc'), self.smoke('abc')), [])
        problems = t.texts_equal(self.smoke('abc'), self.smoke('abd'))
        self.assertEqual(len(problems), 1)
        self.assertIn('user 2', problems[0])
        self.assertTrue(t.texts_equal(self.smoke('ab'), self.smoke('abc')))

    def test_the_smoke_json_line_is_read(self):
        self.assertEqual(t.smoke_of('x\nSMOKE_JSON %s\n' % json.dumps({'a': 1})), {'a': 1})


class CliTests(unittest.TestCase):
    def test_the_pair_command_prints_one_json_line(self):
        import tempfile
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / 'a.log', Path(folder) / 'b.log'
            a.write_text(log(steady(300, 0.2)), encoding='utf-8')
            b.write_text(log(steady(300, 0.19)) + '\n[LOAD] 1 1.0', encoding='utf-8')
            out = io.StringIO()
            with redirect_stdout(out):
                code = t.main(['pair', str(a), str(b), '--max-load', '4.0'])
            result = json.loads(out.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(result['verdict'], 'VOID', 'A logged no load samples: the pair cannot be called clean')


if __name__ == '__main__':
    unittest.main()
