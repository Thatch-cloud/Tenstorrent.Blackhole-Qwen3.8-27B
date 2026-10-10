"""The prefill ladder smoke tests (c2_serving_smoke: prefill_ladder_solo, prefill_ladder_busy, prefill_few_decoders), run for real against test_levern_smoke's fake server,
their smoke check (c2_smoke_check.prefill_ladder_problems) and the report (prefill_ladder_report).

The ladder is a measurement: one request at a time at four prompt sizes, the same four beside seven decoders, and one or two decoders beside a long cold arrival. These
tests hold its SHAPE (the requests, their order, the seats, the fields); the times are the card's.
"""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check as check  # noqa: E402
import prefill_ladder_report as report  # noqa: E402
from test_levern_smoke import Fake, SmokeRuns  # noqa: E402
from test_w2ln_smoke_shapes import constants  # noqa: E402

SEATS = 8
RUNGS = ['4096', '32768', '131072', '253920']


def arrivals(chats):
    """The ladder's rung requests of a fake's chat log: the ones that are not decoders (no ignore_eos)."""
    return [chat for chat in chats if not chat['body'].get('ignore_eos')]


def max_overlap(intervals):
    """The most intervals (start, end) open at one instant."""
    events = sorted([(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals], key=lambda item: (item[0], item[1]))
    open_now = best = 0
    for _, step in events:
        open_now += step
        best = max(best, open_now)
    return best


class SoloTests(SmokeRuns):
    def test_four_rungs_one_request_at_a_time_ascending_with_nothing_beside_them(self):
        results, chats, _completions, _tokenized, _ = self.ran('prefill_ladder_solo')
        entry = results['prefill_ladder_solo']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'rungs'})
        self.assertEqual(list(entry['rungs']), RUNGS, 'ascending, in order')
        self.assertEqual(entry['lengths'], [4096, 32768, 131072, 253920])
        self.assertEqual(len(chats), 4, 'one request per rung and nothing else: no decoder')
        self.assertEqual({chat['body']['max_tokens'] for chat in chats}, {16})
        self.assertFalse(any(chat['body'].get('ignore_eos') for chat in chats))
        self.assertEqual(sorted(chat['body']['max_tokens'] for chat in chats), [16] * 4)
        # the rungs were sent in ascending order of size
        sizes = [len(json.dumps(chat['body'])) for chat in sorted(chats, key=lambda chat: chat['seq'])]
        self.assertEqual(sizes, sorted(sizes))
        spans = [(row['started_at'], row['ended_at']) for row in entry['rungs'].values()]
        self.assertEqual(max_overlap(spans), 1, 'never two at once')
        self.assertEqual([span[0] >= previous[1] for previous, span in zip(spans, spans[1:])], [True] * 3, 'each rung starts after the one before ended')
        self.assertEqual(entry['fit']['targets'], [253920, 131072, 32768, 4096] * 2)

    def test_every_rung_has_the_recorded_fields(self):
        results, *_ = self.ran('prefill_ladder_solo')
        for key, row in results['prefill_ladder_solo']['rungs'].items():
            with self.subTest(rung=key):
                for field in ('prompt_tokens', 'ttft_s', 'prefill_tok_s', 'tokens', 'finish', 'wall_s', 'text', 'started_at', 'first_at', 'ended_at'):
                    self.assertIn(field, row)
                # (the fake answers a small prompt within the stamps' millisecond: a zero time to first token has no rate, which is recorded as None, never divided)
                self.assertGreaterEqual(row['ttft_s'], 0)
                if row['ttft_s'] > 0:
                    self.assertAlmostEqual(row['prefill_tok_s'], row['prompt_tokens'] / row['ttft_s'], delta=0.1 + 0.001 * row['prefill_tok_s'])
                else:
                    self.assertIsNone(row['prefill_tok_s'])
                self.assertEqual(row['tokens'], 16)
                self.assertNotIn('error', row)
        self.assertEqual(check.smoke_problems(results), [])

    def test_the_four_prompts_are_four_different_prompts(self):
        _results, chats, *_ = self.ran('prefill_ladder_solo')
        digests = {hashlib.sha256(json.dumps(chat['body']['messages']).encode()).hexdigest() for chat in chats}
        self.assertEqual(len(digests), 4)


class BusyTests(SmokeRuns):
    def test_each_rung_arrives_beside_seven_decoders_and_never_makes_a_ninth_request(self):
        results, chats, *_ = self.ran('prefill_ladder_busy')
        entry = results['prefill_ladder_busy']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'rungs'})
        self.assertEqual(list(entry['rungs']), RUNGS)
        self.assertEqual(entry['decoders'], 7)
        self.assertEqual(entry['budgets'], [600, 1500, 4000, 6000])
        self.assertEqual(len(chats), 4 * 8, 'seven decoders and the rung, four times')
        decoders = [chat for chat in chats if chat['body'].get('ignore_eos')]
        self.assertEqual(len(decoders), 28)
        self.assertEqual(sorted(chat['body']['max_tokens'] for chat in decoders), sorted([600] * 7 + [1500] * 7 + [4000] * 7 + [6000] * 7))
        self.assertEqual(len(arrivals(chats)), 4)
        self.assertEqual({chat['body']['max_tokens'] for chat in arrivals(chats)}, {16})
        spans = []
        for row in entry['rungs'].values():
            self.assertEqual(len(row['decoders']), 7)
            spans += [(row['started_at'], row['ended_at'])] + [(user['started_at'], user['ended_at']) for user in row['decoders']]
        self.assertLessEqual(max_overlap(spans), SEATS, 'at most the eight seats of the profile at any instant')

    def test_every_rung_records_the_decoders_gap_window_and_live_count(self):
        results, *_ = self.ran('prefill_ladder_busy')
        for key, row in results['prefill_ladder_busy']['rungs'].items():
            with self.subTest(rung=key):
                for field in ('prompt_tokens', 'ttft_s', 'prefill_tok_s', 'decoders', 'decoders_live_at_first_token', 'longest_gap_s', 'window', 'busy_over_solo',
                              'arrival_started_at'):
                    self.assertIn(field, row)
                self.assertEqual(row['window']['seats'], 7)
                for seat, user in enumerate(row['decoders']):
                    self.assertEqual(user['seat'], seat)
                    for field in ('tokens', 'longest_gap_s', 'chunks', 'est_tok_s', 'ended_at'):
                        self.assertIn(field, user)
                self.assertIsNone(row['busy_over_solo'], 'the solo ladder did not run in this process')
        long_rung = results['prefill_ladder_busy']['rungs']['253920']
        self.assertGreaterEqual(long_rung['decoders_live_at_first_token'], 6, 'the long rung ran beside its decoders')
        self.assertEqual(check.smoke_problems(results), [])

    def test_the_busy_over_solo_ratio_is_filled_in_when_both_ran_in_one_smoke_run(self):
        with Fake() as fake:
            results, _stdout = self.run_smoke(fake, ('prefill_ladder_solo', 'prefill_ladder_busy'))
            chats = list(fake.chats)
        solo, busy = results['prefill_ladder_solo']['rungs'], results['prefill_ladder_busy']['rungs']
        for key in RUNGS:
            if solo[key]['ttft_s'] > 0:
                self.assertAlmostEqual(busy[key]['busy_over_solo'], busy[key]['ttft_s'] / solo[key]['ttft_s'], places=2)
            else:
                self.assertIsNone(busy[key]['busy_over_solo'], 'no ratio over a zero solo time')
        self.assertGreater(solo['253920']['ttft_s'], 0.3, 'the fake holds a long prompt 0.4 s')
        self.assertGreater(busy['253920']['busy_over_solo'], 0)
        eight = arrivals(chats)
        self.assertEqual(len(eight), 8)
        self.assertEqual(len({hashlib.sha256(json.dumps(chat['body']['messages']).encode()).hexdigest() for chat in eight}), 8,
                         'eight different prompts: a repeated rung never finds its blocks in a prefix cache')
        self.assertEqual(check.smoke_problems(results), [])


class FewDecodersTests(SmokeRuns):
    def test_one_decoder_and_120k_then_two_decoders_and_254k(self):
        results, chats, *_ = self.ran('prefill_few_decoders')
        entry = results['prefill_few_decoders']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'shapes'})
        shapes = entry['shapes']
        self.assertEqual([(shape['shape_decoders'], shape['target_tokens']) for shape in shapes], [(1, 120000), (2, 253920)])
        self.assertEqual([len(shape['decoders']) for shape in shapes], [1, 2])
        self.assertEqual(len(chats), 2 + 3, 'one decoder and its arrival, two decoders and theirs')
        self.assertEqual(sorted(chat['body']['max_tokens'] for chat in chats if chat['body'].get('ignore_eos')), [4000, 6000, 6000])
        spans = []
        for shape in shapes:
            spans += [(shape['started_at'], shape['ended_at'])] + [(user['started_at'], user['ended_at']) for user in shape['decoders']]
        self.assertLessEqual(max_overlap(spans), 3)
        for shape in shapes:
            for field in ('prompt_tokens', 'ttft_s', 'prefill_tok_s', 'longest_gap_s', 'window', 'decoders_live_at_first_token'):
                self.assertIn(field, shape)
            self.assertTrue(all(isinstance(user['tokens'], int) for user in shape['decoders']), 'the decoders\' tokens are recorded')
        self.assertEqual(check.smoke_problems(results), [])


class ShapeAndDispatchTests(unittest.TestCase):
    def test_the_seat_counts_are_declared_and_fit_the_eight_seats(self):
        found = constants()
        seats = found['SHAPE_SEATS']
        self.assertEqual((seats['prefill_ladder_solo'], seats['prefill_ladder_busy'], seats['prefill_few_decoders']), (1, 8, 3))
        self.assertLessEqual(max(seats[name] for name in check.PREFILL_LADDER_TESTS), SEATS)
        self.assertEqual(found['LADDER_DECODERS'] + 1, seats['prefill_ladder_busy'])
        self.assertEqual(found['LADDER_TOKENS'], (4096, 32768, 131072, 253920))
        self.assertEqual(len(found['LADDER_BUSY_BUDGETS']), len(found['LADDER_TOKENS']))
        self.assertEqual(found['FEWDEC_SHAPES'], ((1, 120000, 4000), (2, 253920, 6000)))
        self.assertLessEqual(max(shape[0] for shape in found['FEWDEC_SHAPES']) + 1, seats['prefill_few_decoders'])

    def test_the_three_tests_are_opt_in_and_run_solo_first(self):
        source = (HERE / 'c2_serving_smoke.py').read_text(encoding='utf-8')
        self.assertIn("for _ladder_name in ('prefill_ladder_solo', 'prefill_ladder_busy', 'prefill_few_decoders'):", source)
        self.assertIn('if ONLY and _ladder_name in ONLY:', source)


def stream_row(**more):
    row = dict(prompt_tokens=4096, ttft_s=3.5, prefill_tok_s=1170.3, tokens=16, finish='length', text='fine words here', wall_s=4.0)
    row.update(more)
    return row


def decoders(count=7):
    return [dict(seat=seat, tokens=900, finish='length', text='fine words here', longest_gap_s=0.4, chunks=30, est_tok_s=20.0, ended_at=2.0)
            for seat in range(count)]


def good():
    rungs = dict((key, stream_row(prompt_tokens=int(key))) for key in RUNGS)
    busy = dict((key, stream_row(prompt_tokens=int(key), decoders=decoders(), decoders_live_at_first_token=7, longest_gap_s=0.5, busy_over_solo=1.4)) for key in RUNGS)
    few = [stream_row(prompt_tokens=120000, decoders=decoders(1), shape_decoders=1, target_tokens=120000),
           stream_row(prompt_tokens=253920, decoders=decoders(2), shape_decoders=2, target_tokens=253920)]
    return {'prefill_ladder_solo': dict(rungs=rungs), 'prefill_ladder_busy': dict(rungs=busy), 'prefill_few_decoders': dict(shapes=few)}


class CheckTests(unittest.TestCase):
    def test_a_clean_run_and_a_run_without_the_tests_have_no_problem(self):
        self.assertEqual(check.prefill_ladder_problems(good()), [])
        self.assertEqual(check.smoke_problems(good()), [])
        self.assertEqual(check.prefill_ladder_problems({}), [])
        self.assertEqual(check.prefill_ladder_problems({'warmup': dict(value=200)}), [])

    def test_a_whole_test_that_failed_or_recorded_nothing_is_a_problem(self):
        self.assertEqual(check.prefill_ladder_problems({'prefill_ladder_solo': dict(error='boom')}), ['prefill_ladder_solo: boom'])
        self.assertTrue(any('no rung or shape' in text for text in check.prefill_ladder_problems({'prefill_ladder_solo': dict(rungs={})})))
        self.assertTrue(any('no rung or shape' in text for text in check.prefill_ladder_problems({'prefill_few_decoders': dict(shapes=[])})))
        self.assertTrue(check.prefill_ladder_problems({'prefill_ladder_busy': 7}))

    def test_a_rung_with_an_error_no_ttft_no_prompt_count_or_no_tokens_is_a_problem(self):
        broken = good()
        broken['prefill_ladder_solo']['rungs']['32768'] = dict(error='timed out', wall_s=3600.0)
        self.assertTrue(any('rung 32768: timed out' in text for text in check.prefill_ladder_problems(broken)))
        broken = good()
        broken['prefill_ladder_solo']['rungs']['4096']['ttft_s'] = None
        self.assertTrue(any('rung 4096: no time to first token' in text for text in check.prefill_ladder_problems(broken)))
        broken = good()
        broken['prefill_ladder_solo']['rungs']['4096']['prompt_tokens'] = None
        self.assertTrue(any('rung 4096: the server reported no prompt token count' in text for text in check.prefill_ladder_problems(broken)))
        broken = good()
        broken['prefill_ladder_solo']['rungs']['131072']['tokens'] = 0
        self.assertTrue(any('rung 131072: no tokens' in text for text in check.prefill_ladder_problems(broken)))

    def test_a_decoder_that_failed_or_a_busy_rung_without_decoders_is_a_problem(self):
        broken = good()
        broken['prefill_ladder_busy']['rungs']['4096']['decoders'][3]['error'] = 'read timed out'
        self.assertTrue(any('prefill_ladder_busy rung 4096 decoder 3: read timed out' in text for text in check.prefill_ladder_problems(broken)))
        broken = good()
        broken['prefill_ladder_busy']['rungs']['4096']['decoders'] = []
        self.assertTrue(any('no decoder was recorded' in text for text in check.prefill_ladder_problems(broken)))
        broken = good()
        broken['prefill_few_decoders']['shapes'][1]['decoders'][0]['tokens'] = 0
        self.assertTrue(any('shape 1 decoder 0: no tokens' in text for text in check.prefill_ladder_problems(broken)))

    def test_decoders_that_ended_early_and_slow_times_are_recorded_not_gated(self):
        slow = good()
        slow['prefill_ladder_busy']['rungs']['4096']['decoders_live_at_first_token'] = 0
        slow['prefill_ladder_busy']['rungs']['253920']['ttft_s'] = 9999.0
        self.assertEqual(check.prefill_ladder_problems(slow), [])


class ReportTests(unittest.TestCase):
    def results(self, factor=1.0):
        data = good()
        for key in RUNGS:
            data['prefill_ladder_solo']['rungs'][key]['ttft_s'] = 10.0 * factor
            data['prefill_ladder_busy']['rungs'][key]['ttft_s'] = 18.0 * factor
            data['prefill_ladder_busy']['rungs'][key]['busy_over_solo'] = 1.8
        data['prefill_few_decoders']['shapes'][0]['ttft_s'] = 60.0 * factor
        data['prefill_few_decoders']['shapes'][1]['ttft_s'] = 120.0 * factor
        return data

    def test_the_ladder_renders_solo_beside_busy_with_the_ratio_and_the_shapes(self):
        text = report.render(self.results())
        self.assertIn('| 253,920 |', text)
        self.assertIn('| 4,096 |', text)
        self.assertIn('1.800', text)
        self.assertIn('1 decoder(s), arrival of about 120,000', text)
        self.assertIn('2 decoder(s), arrival of about 253,920', text)
        self.assertNotIn('B over A', text)

    def test_the_second_arm_adds_the_b_over_a_column(self):
        text = report.render(self.results(), self.results(0.5))
        self.assertIn('B over A', text)
        self.assertIn('| solo 4,096 | 10.00 | 5.00 | 0.500 |', text)
        self.assertIn('| busy (7 decoders) 253,920 | 18.00 | 9.00 | 0.500 |', text)
        self.assertIn('| 2 decoder(s), 253,920 | 120.00 | 60.00 | 0.500 |', text)

    def test_the_pair_table_says_whether_the_arrival_text_is_byte_identical(self):
        one, two = self.results(), self.results(0.5)
        for data in (one, two):
            for shape in data['prefill_few_decoders']['shapes']:
                shape['content_sha256'] = 'a' * 64
        text = report.render(one, two)
        self.assertIn('| arrival text |', text)
        self.assertEqual(text.count('| same |'), 2)
        two['prefill_few_decoders']['shapes'][1]['content_sha256'] = 'b' * 64
        text = report.render(one, two)
        self.assertEqual((text.count('| same |'), text.count('| DIFFERS |')), (1, 1))
        self.assertEqual(report.same_text({}, {'content_sha256': 'a'}), '-')

    def test_an_errored_rung_and_a_missing_test_render_without_failing(self):
        data = self.results()
        data['prefill_ladder_solo']['rungs']['131072'] = dict(error='timed out', wall_s=3600.0)
        text = report.render(data)
        self.assertIn('ERROR: timed out', text)
        self.assertIn('No prefill ladder test', report.render({}))
        solo_only = {'prefill_ladder_solo': data['prefill_ladder_solo']}
        self.assertIn('| 4,096 |', report.render(solo_only))

    def test_the_command_reads_a_results_file_or_a_log_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            plain = Path(folder) / 'results.json'
            plain.write_text(json.dumps(self.results()), encoding='utf-8')
            log = Path(folder) / 'smoke.log'
            log.write_text('noise\nSMOKE_JSON ' + json.dumps(self.results(0.5)) + '\n', encoding='utf-8')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(report.main([str(plain)]), 0)
                self.assertEqual(report.main([str(plain), '--second', str(log)]), 0)
            self.assertIn('B over A', out.getvalue())
            empty = Path(folder) / 'empty.json'
            empty.write_text('{"warmup": {}}', encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(report.main([str(empty)]), 2)
                self.assertEqual(report.main([str(Path(folder) / 'missing.json')]), 2)

    def test_the_report_script_names_no_host_path_or_registry(self):
        source = (HERE / 'prefill_ladder_report.py').read_text(encoding='utf-8')
        for needle in ('/home/', 'zot.', '192.168.', '172.16.', '10.10.'):
            self.assertNotIn(needle, source)


if __name__ == '__main__':
    unittest.main()
