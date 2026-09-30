"""tp_agreement: the collect and compare arithmetic and the smoke hook, on the CPU with an injected server."""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tp_agreement as agree  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def run(name, tokens, logprob=-0.5, finish='length', text=None, available=True):
    return dict(name=name, text=text if text is not None else ' '.join(tokens), tokens=tokens if available else [],
                logprobs=[logprob] * len(tokens) if available else [], finish=finish, logprobs_available=available)


def words(count, offset=0):
    return ['token_id:%d' % (index * 7 + offset) for index in range(count)]


class CompareTests(unittest.TestCase):
    def pair(self, ref_tokens, cand_tokens, cand_logprob=-0.5):
        ref = dict(label='tp2', prompts=[run('a', ref_tokens)])
        cand = dict(label='tp4', prompts=[run('a', cand_tokens, cand_logprob)])
        return agree.compare(ref, cand)

    def test_identical_runs_agree_completely(self):
        report = self.pair(words(100), words(100))
        row = report['prompts'][0]
        self.assertTrue(row['identical'])
        self.assertEqual(row['common_prefix'], 100)
        self.assertAlmostEqual(row['perplexity_ratio'], 1.0)
        self.assertEqual((report['verdict'], report['reasons']), ('AGREE', []))

    def test_a_late_flip_still_agrees_and_an_early_one_asks_for_review(self):
        late = words(100)[:60] + words(40, 1)
        self.assertEqual(self.pair(words(100), late)['verdict'], 'AGREE')
        self.assertEqual(self.pair(words(100), late)['prompts'][0]['common_prefix'], 60)
        early = words(100)[:5] + words(95, 1)
        report = self.pair(words(100), early)
        self.assertEqual(report['verdict'], 'REVIEW')
        self.assertIn('median common prefix 5 tokens < 24', report['reasons'][0])

    def test_a_less_sure_candidate_is_flagged_by_the_perplexity_ratio(self):
        report = self.pair(words(100), words(100), cand_logprob=-0.6)
        self.assertAlmostEqual(report['max_perplexity_ratio'], math.exp(0.1), places=6)
        self.assertEqual(report['verdict'], 'AGREE' if math.exp(0.1) <= agree.MAX_PPL_RATIO else 'REVIEW')
        worse = self.pair(words(100), words(100), cand_logprob=-0.8)
        self.assertEqual(worse['verdict'], 'REVIEW')
        self.assertIn('perplexity ratio', worse['reasons'][0])

    def test_an_empty_or_looping_completion_is_incoherent(self):
        looping = ['token_id:1', 'token_id:2'] * 60
        report = self.pair(words(100), looping)
        self.assertFalse(report['prompts'][0]['coherent'])
        self.assertIn('repeated loop', report['prompts'][0]['incoherent_because'])
        report = agree.compare(dict(label='x', prompts=[run('a', words(50))]),
                               dict(label='y', prompts=[run('a', [], text='')]))
        self.assertEqual(report['verdict'], 'REVIEW')

    def test_a_short_answer_that_stopped_is_coherent(self):
        ok, _ = agree.coherent(run('a', words(5), finish='stop'))
        self.assertTrue(ok)
        ok, why = agree.coherent(run('a', words(5)))
        self.assertFalse(ok)
        self.assertIn('no stop', why)

    def test_without_logprobs_it_compares_text_and_says_so(self):
        ref = dict(label='tp2', prompts=[run('a', words(50), text='the quick brown fox ' * 20)])
        cand = dict(label='tp4', prompts=[run('a', words(50), text='the quick brown fox ' * 20, available=False)])
        report = agree.compare(ref, cand)
        self.assertEqual(report['prompts'][0]['unit'], 'characters')
        self.assertEqual(report['verdict'], 'REVIEW')
        self.assertTrue(any('no token-level comparison' in reason for reason in report['reasons']))

    def test_a_missing_prompt_is_reported(self):
        report = agree.compare(dict(label='a', prompts=[run('a', words(50)), run('b', words(50))]),
                               dict(label='b', prompts=[run('a', words(50))]))
        self.assertTrue(report['prompts'][1]['missing'])
        self.assertIn('1 prompts missing', report['reasons'][0])


class CollectTests(unittest.TestCase):
    def server(self, refuse_logprobs=False):
        calls = []

        def send(base, path, body, timeout=0):
            calls.append(body)
            if refuse_logprobs and body.get('logprobs'):
                return 400, 'logprobs unsupported'
            choice = dict(message=dict(content='hello world ' * 20), finish_reason='length')
            if body.get('logprobs'):
                choice['logprobs'] = dict(content=[dict(token='token_id:%d' % i, logprob=-0.25) for i in range(40)])
            return 200, dict(choices=[choice], usage=dict(completion_tokens=40))

        return send, calls

    def test_the_prompts_are_this_checkouts_own_files_and_greedy(self):
        prompts = agree.read_prompts(ROOT)
        self.assertGreaterEqual(len(prompts), 3, 'the repository holds its own real text')
        send, calls = self.server()
        result = agree.collect('http://x', 'm', ROOT, 'tp4', send=send, log=lambda line: None)
        self.assertEqual(len(result['prompts']), len(prompts))
        for body in calls:
            self.assertEqual(body['temperature'], 0)
            self.assertEqual(body['max_tokens'], agree.MAX_TOKENS)
            self.assertTrue(body['return_tokens_as_token_ids'])
        self.assertTrue(result['prompts'][0]['logprobs_available'])
        self.assertEqual(result['prompts'][0]['logprobs'][0], -0.25)

    def test_a_server_that_refuses_logprobs_is_asked_again_without(self):
        send, calls = self.server(refuse_logprobs=True)
        result = agree.collect('http://x', 'm', ROOT, 'tp4', send=send, log=lambda line: None)
        first = result['prompts'][0]
        self.assertFalse(first['logprobs_available'])
        self.assertTrue(first['text'])
        self.assertIsNone(first.get('error'))
        self.assertEqual([bool(body.get('logprobs')) for body in calls[:2]], [True, False])

    def test_main_writes_the_file_and_compare_reads_two(self):
        send, _ = self.server()
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for label in ('tp2', 'tp4'):
                path = os.path.join(directory, label + '.json')
                self.assertEqual(agree.main(['collect', 'http://x', 'm', path, '--label', label, '--root', ROOT],
                                            send=send, log=lambda line: None), 0)
                outputs.append(path)
            report = os.path.join(directory, 'agreement.json')
            lines = []
            self.assertEqual(agree.main(['compare'] + outputs + ['--report', report], log=lines.append), 0)
            with open(report) as handle:
                self.assertEqual(json.load(handle)['verdict'], 'AGREE')
            self.assertIn('verdict=AGREE', lines[-1])

    def test_a_failed_request_fails_collect(self):
        def send(base, path, body, timeout=0):
            return 500, 'boom'

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'out.json')
            self.assertEqual(agree.main(['collect', 'http://x', 'm', path, '--root', ROOT], send=send,
                                        log=lambda line: None), 1)


class SmokeHookTests(unittest.TestCase):
    def test_the_smoke_runs_agreement_only_when_named(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn("if ONLY and 'agreement' in ONLY:", text)
        self.assertIn("record('agreement'", text)


if __name__ == '__main__':
    unittest.main()
