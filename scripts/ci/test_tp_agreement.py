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


def run(name, tokens, logprob=-0.5, finish='length', text=None, available=True, second=-3.0, device=None):
    """A collect() entry: each position's candidates are the chosen token and one alternative `second` nats down
    (one byte per token, so character n is token n)."""
    text = text if text is not None else ' '.join(tokens)
    return dict(name=name, text=text, tokens=tokens if available else [],
                logprobs=[logprob] * len(tokens) if available else [], finish=finish, logprobs_available=available,
                tops=[[[token, logprob], ['alt:' + token, logprob + second]] for token in tokens] if available else [],
                piece_bytes=[1] * len(tokens) if available else [], prompt_sha256='sha-' + name,
                device=dict(text=text) if device is None else device)


def words(count, offset=0):
    return ['token_id:%d' % (index * 7 + offset) for index in range(count)]


class CompareTests(unittest.TestCase):
    def pair(self, ref_tokens, cand_tokens, cand_logprob=-0.5):
        ref = dict(label='tp2', prompts=[run('a', ref_tokens)])
        cand = dict(label='tp4', prompts=[run('a', cand_tokens, cand_logprob)])
        return agree.compare(ref, cand)

    def flipped(self, at, ref_second=-0.2, cand_second=-0.2):
        """A candidate that parts from the reference at token `at`, each side scoring the other's token as a
        near tie by default; the parted tokens are candidates in each other's lists."""
        ref_tokens = words(100)
        cand_tokens = ref_tokens[:at] + words(100 - at, 1)
        ref, cand = run('a', ref_tokens), run('a', cand_tokens)
        ref['tops'][at] = [[ref_tokens[at], -0.5], [cand_tokens[at], -0.5 + ref_second]]
        cand['tops'][at] = [[cand_tokens[at], -0.5], [ref_tokens[at], -0.5 + cand_second]]
        return agree.compare(dict(label='tp2', prompts=[ref]), dict(label='tp4', prompts=[cand]))

    def test_identical_runs_agree_completely(self):
        report = self.pair(words(100), words(100))
        row = report['prompts'][0]
        self.assertTrue(row['identical'])
        self.assertEqual(row['common_prefix'], 100)
        self.assertIsNone(row['divergence'])
        self.assertAlmostEqual(row['perplexity_ratio'], 1.0)
        self.assertEqual((report['verdict'], report['reasons']), ('AGREE', []))

    def test_a_late_flip_at_a_near_tie_agrees_and_an_early_one_asks_for_review(self):
        report = self.flipped(60)
        self.assertEqual((report['verdict'], report['reasons']), ('AGREE', []))
        self.assertEqual(report['prompts'][0]['common_prefix'], 60)
        self.assertTrue(report['prompts'][0]['divergence']['ok'])
        early = self.flipped(5)
        self.assertEqual(early['verdict'], 'REVIEW')
        self.assertIn('median common prefix 5 tokens < 24', early['reasons'][0])

    def test_a_flip_that_is_not_a_near_tie_is_a_content_failure_however_late(self):
        report = self.flipped(60, ref_second=-4.0, cand_second=-4.0)
        self.assertEqual(report['verdict'], 'REVIEW')
        self.assertTrue(any('nats apart' in reason for reason in report['reasons']), report['reasons'])
        self.assertFalse(report['prompts'][0]['divergence']['ok'])

    def test_a_flip_to_a_token_outside_the_others_top_k_is_refused(self):
        ref_tokens = words(100)
        cand_tokens = ref_tokens[:60] + words(40, 1)
        report = agree.compare(dict(label='a', prompts=[run('a', ref_tokens)]),
                               dict(label='b', prompts=[run('a', cand_tokens)]))
        self.assertEqual(report['verdict'], 'REVIEW')
        self.assertTrue(any('outside the other' in reason for reason in report['reasons']), report['reasons'])

    def test_runs_that_asked_different_prompts_are_not_comparable(self):
        ref = dict(label='a', prompts=[run('a', words(100))])
        cand = dict(label='b', prompts=[dict(run('a', words(100)), prompt_sha256='another')])
        report = agree.compare(ref, cand)
        self.assertEqual(report['verdict'], 'NOT_COMPARABLE')
        self.assertIn('did not ask the same prompts', report['reasons'][0])
        old = dict(run('a', words(100)))
        del old['prompt_sha256']
        self.assertEqual(agree.compare(dict(label='a', prompts=[old]), cand)['verdict'], 'NOT_COMPARABLE')
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, body in (('ref', ref), ('cand', cand)):
                paths.append(os.path.join(directory, name + '.json'))
                with open(paths[-1], 'w') as handle:
                    json.dump(body, handle)
            self.assertEqual(agree.main(['compare'] + paths, log=lambda line: None), 2)

    def test_a_less_sure_candidate_is_flagged_by_the_perplexity_ratio(self):
        report = self.pair(words(100), words(100), cand_logprob=-0.6)
        self.assertAlmostEqual(report['max_perplexity_ratio'], math.exp(0.1), places=6)
        self.assertEqual(report['verdict'], 'AGREE' if math.exp(0.1) <= agree.MAX_PPL_RATIO else 'REVIEW')
        worse = self.pair(words(100), words(100), cand_logprob=-0.8)
        self.assertEqual(worse['verdict'], 'REVIEW')
        self.assertIn('perplexity ratio', worse['reasons'][0])

    def test_a_candidate_more_sure_than_the_reference_is_flagged_too(self):
        # a confidently wrong model scores its own text LOWER: the ratio is judged on both sides
        better = self.pair(words(100), words(100), cand_logprob=-0.2)
        self.assertLess(better['max_perplexity_ratio'], 1.0)
        self.assertEqual(better['verdict'], 'REVIEW')
        self.assertIn('perplexity ratio outside', better['reasons'][0])

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


class DeviceSamplerTests(unittest.TestCase):
    def entry(self, device_text, gap):
        tokens = ['token_id:%d' % i for i in range(20)]
        entry = run('a', tokens, text='abcdefghijklmnopqrst', device=dict(text=device_text))
        entry['tops'][10] = [[tokens[10], -0.4], ['token_id:99', -0.4 - gap]]
        return entry

    def test_identical_text_is_ok(self):
        check = agree.device_check(self.entry('abcdefghijklmnopqrst', 1.0))
        self.assertTrue(check['ok'])
        self.assertTrue(check['identical'])

    def test_a_difference_at_a_bfloat16_tie_is_ok_and_at_a_sure_token_is_not(self):
        tie = agree.device_check(self.entry('abcdefghijXlmnopqrst', 0.1))
        self.assertTrue(tie['ok'])
        self.assertIn('near-tie at character 10', tie['why'])
        sure = agree.device_check(self.entry('abcdefghijXlmnopqrst', 3.0))
        self.assertFalse(sure['ok'])
        self.assertIn('host was sure', sure['why'])

    def test_a_difference_the_host_cannot_place_is_not_ok(self):
        entry = self.entry('abcdefghijXlmnopqrst', 0.1)
        entry['piece_bytes'] = []
        self.assertFalse(agree.device_check(entry)['ok'])
        entry = self.entry('abcdefghijXlmnopqrst', 0.1)
        del entry['device']
        self.assertIn('no device-sampled answer', agree.device_check(entry)['why'])
        entry = self.entry('abcdefghijklmnopqrst', 0.1)
        entry['device']['error'] = 'HTTP 500'
        self.assertFalse(agree.device_check(entry)['ok'])

    def test_compare_reports_the_candidates_device_sampler(self):
        ref = dict(label='a', prompts=[run('a', words(100))])
        bad = run('a', words(100), device=dict(text='a totally different answer'))
        report = agree.compare(ref, dict(label='b', prompts=[bad]))
        self.assertEqual(report['verdict'], 'REVIEW')
        self.assertTrue(any('candidate device sampler' in reason for reason in report['reasons']), report['reasons'])


class SelfCheckTests(unittest.TestCase):
    def test_a_sound_run_passes(self):
        ok, reasons = agree.self_check(dict(prompts=[run('a', words(100)), run('b', words(60))]))
        self.assertEqual((ok, reasons), (True, []))

    def test_what_the_smoke_fails_on(self):
        cases = {
            'incoherent': dict(prompts=[run('a', ['token_id:1', 'token_id:2'] * 60)]),
            'errored': dict(prompts=[dict(run('a', words(100)), error='HTTP 500')]),
            'device sampler': dict(prompts=[run('a', words(100), device=dict(text='other'))]),
            'perplexity': dict(prompts=[run('a', words(100), logprob=-3.0)]),
            'no prompt': dict(prompts=[]),
        }
        for word, result in cases.items():
            ok, reasons = agree.self_check(result)
            self.assertFalse(ok, word)
            self.assertTrue(any(word in reason for reason in reasons), (word, reasons))


class CollectTests(unittest.TestCase):
    def server(self, refuse_logprobs=False):
        calls = []

        def send(base, path, body, timeout=0):
            calls.append(body)
            if refuse_logprobs and body.get('logprobs'):
                return 400, 'logprobs unsupported'
            choice = dict(message=dict(content='hello world ' * 20), finish_reason='length')
            if body.get('logprobs'):
                choice['logprobs'] = dict(content=[dict(token='token_id:%d' % i, logprob=-0.25, bytes=[104] * 6,
                                                        top_logprobs=[dict(token='token_id:%d' % i, logprob=-0.25),
                                                                      dict(token='token_id:9', logprob=-2.0)])
                                                   for i in range(40)])
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
            if body.get('logprobs'):
                self.assertTrue(body['return_tokens_as_token_ids'])
                self.assertEqual(body['top_logprobs'], agree.TOP_K)
        self.assertTrue(result['prompts'][0]['logprobs_available'])
        self.assertEqual(result['prompts'][0]['logprobs'][0], -0.25)
        self.assertEqual(result['prompts'][0]['tops'][0][1], ['token_id:9', -2.0])
        self.assertEqual(result['prompts'][0]['piece_bytes'][0], 6)

    def test_every_prompt_is_asked_twice_once_with_logprobs_and_once_without(self):
        # at four devices any logprobs request is host-sampled: the request without them is the device sampler's
        send, calls = self.server()
        result = agree.collect('http://x', 'm', ROOT, 'tp4', send=send, log=lambda line: None)
        self.assertEqual(len(calls), 2 * len(result['prompts']))
        self.assertEqual([bool(body.get('logprobs')) for body in calls[:4]], [True, False, True, False])
        for entry in result['prompts']:
            self.assertEqual(len(entry['prompt_sha256']), 64)
            self.assertEqual(entry['device']['text'], entry['text'])
            self.assertTrue(agree.device_check(entry)['identical'])

    def test_a_server_that_refuses_logprobs_is_asked_again_without(self):
        send, calls = self.server(refuse_logprobs=True)
        result = agree.collect('http://x', 'm', ROOT, 'tp4', send=send, log=lambda line: None)
        first = result['prompts'][0]
        self.assertFalse(first['logprobs_available'])
        self.assertTrue(first['text'])
        self.assertIsNone(first.get('error'))
        self.assertEqual([bool(body.get('logprobs')) for body in calls[:3]], [True, False, False])

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
    def source(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            return handle.read()

    def test_the_smoke_runs_agreement_only_when_named(self):
        text = self.source()
        self.assertIn("if ONLY and 'agreement' in ONLY:", text)
        self.assertIn("record('agreement'", text)

    def test_the_smoke_fails_its_step_on_a_failed_agreement_check(self):
        text = self.source()
        self.assertIn('tp_agreement.self_check(out)', text)
        self.assertIn('sys.exit(1)', text.split("print('SMOKE_JSON '")[1])


if __name__ == '__main__':
    unittest.main()
