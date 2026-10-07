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
        self.assertGreaterEqual(result['matched'], 100)
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
        self.assertEqual(t.load_void('[LOAD] 1 5.5\n[LOAD] 61 9.5\n[LOAD] 121 1.0', 4.0)[0], True)
        self.assertEqual(t.load_void('[LOAD] 1 1.5\n[LOAD] 61 9.5\n[LOAD] 121 1.0', 4.0), (False, None), 'one spike is no void: the median is the condition')
        self.assertIsNone(t.load_void('nothing', 4.0)[0])

    def test_the_load_maximum_is_calibrated_from_a_serving_baseline_not_the_idle_host(self):
        serving = '\n'.join('[LOAD] %d %.1f' % (60 * index, value) for index, value in enumerate((3.0, 6.0, 5.5, 5.8, 6.1, 5.9)))
        self.assertEqual(t.load_limit(serving, 3.0), 8.85)
        self.assertIsNone(t.load_limit('nothing'))
        self.assertEqual(t.load_gap_void('[LOAD] 1 5.0', '[LOAD] 1 6.5', 2.0), (False, None))
        self.assertEqual(t.load_gap_void('[LOAD] 1 5.0', '[LOAD] 1 9.0', 2.0)[0], True)
        self.assertIsNone(t.load_gap_void('[LOAD] 1 5.0', '', 2.0)[0])


SHAPES = {'steady': 4500, '32k': 33000, '128k': 120500, 'skew': 66500}


def shaped(seconds_by_shape, count=150):
    """One timed job's log: the four shapes' eight-live rounds, back to back, each at its own round time."""
    rounds = []
    for name in ('steady', '32k', '128k', 'skew'):
        rounds += steady(count, seconds_by_shape[name], start=SHAPES[name], step=3)
    return log(rounds)


def uniform(seconds):
    return dict.fromkeys(SHAPES, seconds)


DRIFT = (0.0, 0.001, -0.001)


class WindowTests(unittest.TestCase):
    def test_the_windows_do_not_overlap_and_hold_their_shapes_mean_positions(self):
        spans = sorted(t.WINDOWS.values())
        for first, second in zip(spans, spans[1:]):
            self.assertLessEqual(first[1], second[0])
        for name, position in SHAPES.items():
            low, high = t.WINDOWS[name]
            self.assertTrue(low <= position < high, name)
        # the skewed eight's mean position: two 253,920 users and six 4,096 ones, plus the 800 tokens they decode
        self.assertTrue(t.WINDOWS['skew'][0] <= (2 * 253920 + 6 * 4096) / 8 + 800 < t.WINDOWS['skew'][1])

    def test_a_window_keeps_only_its_shapes_rounds_and_a_pooled_read_would_have_mixed_them(self):
        a = shaped(uniform(0.2))
        b = shaped(dict(uniform(0.2), **{'32k': 0.18, '128k': 0.21}))
        self.assertEqual(t.compare(a, b, window='32k')['delta_ms'], -20.0)
        self.assertEqual(t.compare(a, b, window='128k')['delta_ms'], 10.0)
        self.assertEqual(t.compare(a, b, window='steady')['delta_ms'], 0.0)
        pooled = t.compare(a, b)
        self.assertLess(pooled['delta_ms'], 0, 'pooled, the 32k gain outweighs the 128k loss: exactly what the per-length read exists to prevent')

    def test_a_loss_at_128k_reads_no_go_there_though_the_32k_gain_carries_the_pool(self):
        a_logs = [shaped(uniform(0.2 + drift)) for drift in DRIFT]
        b_logs = [shaped(dict(uniform(0.2 + drift), **{'32k': 0.18 + drift, '128k': 0.21 + drift})) for drift in DRIFT]
        read = t.read_lengths(a_logs, b_logs)
        self.assertEqual(read['32k']['w2']['verdict'], 'GO')
        self.assertEqual(read['128k']['w2']['verdict'], 'NO-GO')
        self.assertEqual(read['overall']['verdict'], 'NO-GO')
        self.assertEqual(read['overall']['because'], ['128k'])
        self.assertEqual(read['32k']['floor_ms'], 2.0, 'the floor is per length: the spread of the A medians inside the window')

    def test_a_skew_loss_beyond_the_a_to_a_floor_on_two_of_three_pairs_is_a_kill(self):
        a_logs = [shaped(uniform(0.2 + drift)) for drift in DRIFT]
        gain = dict(uniform(0.19), skew=0.2 + 0.005)
        read = t.read_lengths(a_logs, [shaped(dict((key, value + drift) for key, value in gain.items())) for drift in DRIFT])
        self.assertEqual(read['skew']['w2']['verdict'], 'NO-GO')
        self.assertEqual(read['overall']['verdict'], 'NO-GO')
        small = dict(gain, skew=0.2 + 0.0015)
        read = t.read_lengths(a_logs, [shaped(dict((key, value + drift) for key, value in small.items())) for drift in DRIFT])
        self.assertNotEqual(read['skew']['w2']['verdict'], 'NO-GO', 'a loss inside the floor is no kill')

    def test_go_at_32k_and_128k_with_no_skew_kill_is_go_overall(self):
        a_logs = [shaped(uniform(0.2 + drift)) for drift in DRIFT]
        b_logs = [shaped(dict(uniform(0.19 + drift), skew=0.2 + drift)) for drift in DRIFT]
        self.assertEqual(t.read_lengths(a_logs, b_logs)['overall']['verdict'], 'GO')

    def test_a_window_with_too_few_rounds_is_void_there_alone(self):
        a = shaped(uniform(0.2), count=300)
        thin = log(steady(300, 0.19, start=SHAPES['32k'], step=3) + steady(20, 0.19, start=SHAPES['128k'], step=3))
        self.assertEqual(t.compare(a, thin, window='32k')['verdict'], 'MEASURED')
        self.assertEqual(t.compare(a, thin, window='128k')['verdict'], 'VOID')

    def test_lever_n_cost_is_inside_the_floor_or_outside_or_unmeasured_never_silently_inside(self):
        b_logs = [shaped(uniform(0.2)), shaped(uniform(0.2))]
        self.assertEqual(t.lever_n_cost(b_logs, [shaped(uniform(0.203)), shaped(uniform(0.204))], 5.0, '32k')['verdict'], 'INSIDE')
        self.assertEqual(t.lever_n_cost(b_logs, [shaped(uniform(0.210)), shaped(uniform(0.209))], 5.0, '32k')['verdict'], 'OUTSIDE')
        self.assertEqual(t.lever_n_cost(b_logs, [shaped(uniform(0.203)), shaped(uniform(0.204), count=20)], 5.0, '32k')['verdict'], 'UNMEASURED')

    def test_a_lever_n_arm_whose_early_users_finished_before_the_last_was_admitted_leaves_few_eight_live_rounds(self):
        # admission one prefill step at a time: one live, two live ... seven live first, eight-live only at the end
        text, clock = [], 0.0
        for live in range(1, 8):
            for _ in range(60):
                text.append(execute(clock, live=live))
                text += packed(33000, live)
                clock += 0.15
        for position, seconds in steady(40, 0.2, start=33000):
            text.append(execute(clock, live=8))
            text += packed(position, 8)
            clock += seconds
        text.append(execute(clock, live=8))
        body = '\n'.join(text)
        found, _ = t.timed_rounds(body)
        self.assertEqual(len(found), 40, 'only the eight-live rounds count')
        result = t.compare(shaped(uniform(0.2)), body, window='32k')
        self.assertEqual(result['verdict'], 'VOID', 'a VOID is never a vote and never an "inside the floor"')

    def test_the_judge_pair_floor_and_load_limit_commands_read_files(self):
        import tempfile
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for name, text in (('a1', shaped(uniform(0.2))), ('a2', shaped(uniform(0.201))), ('a3', shaped(uniform(0.199))),
                               ('b1', shaped(dict(uniform(0.19), **{'128k': 0.21}))), ('b2', shaped(dict(uniform(0.191), **{'128k': 0.211}))),
                               ('b3', shaped(dict(uniform(0.189), **{'128k': 0.209}))), ('c1', shaped(uniform(0.19))), ('c2', shaped(uniform(0.191)))):
                paths[name] = str(Path(folder) / (name + '.log'))
                Path(paths[name]).write_text(text, encoding='utf-8')

            def run(*argv):
                out = io.StringIO()
                with redirect_stdout(out):
                    code = t.main(list(argv))
                return code, json.loads(out.getvalue())

            code, read = run('judge', '--a', paths['a1'], paths['a2'], paths['a3'], '--b', paths['b1'], paths['b2'], paths['b3'], '--c', paths['c1'], paths['c2'])
            self.assertEqual(code, 0)
            self.assertEqual(read['128k']['w2']['verdict'], 'NO-GO')
            self.assertEqual(read['32k']['w2']['verdict'], 'GO')
            self.assertEqual(read['overall']['because'], ['128k'])
            self.assertIn('lever_n', read['32k'])
            self.assertEqual(run('pair', paths['a1'], paths['b1'], '--window', '128k')[1]['delta_ms'], 10.0)
            self.assertEqual(run('floor', paths['a1'], paths['a2'], paths['a3'], '--window', '32k')[1]['floor_ms'], 2.0)
            loadfile = Path(folder) / 'load.log'
            loadfile.write_text('[LOAD] 1 5.0\n[LOAD] 61 6.0\n[LOAD] 121 7.0', encoding='utf-8')
            self.assertEqual(run('load-limit', str(loadfile), '--margin', '3')[1]['max_load'], 9.0)

    def test_load_applies_per_pair_in_the_judge_read(self):
        a_logs = [shaped(uniform(0.2)) for _ in range(3)]
        b_logs = [shaped(uniform(0.19)) for _ in range(3)]
        quiet, noisy = '[LOAD] 1 5.0\n[LOAD] 61 5.5', '[LOAD] 1 11.0\n[LOAD] 61 11.5'
        loads = dict(A=[quiet] * 3, B=[quiet, quiet, noisy], C=[])
        read = t.read_lengths(a_logs, b_logs, (), loads, 8.0)
        self.assertEqual([pair['verdict'] for pair in read['32k']['pairs']], ['MEASURED', 'MEASURED', 'VOID'])
        self.assertEqual(read['32k']['w2']['verdict'], 'INCONCLUSIVE', 'a VOID pair is no vote: fewer than three measured pairs is never GO')


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
