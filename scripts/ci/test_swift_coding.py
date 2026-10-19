"""The Swift evaluation window's coding set (references/swift/coding-problems.json), the smoke test that generates it (c2_serving_smoke.coding_set) and the CPU-side report
(swift_coding_report.py), held on CPU.

Every problem is held to its own reference solution (the tests pass) and to a stub (they fail); the smoke test is run for real against a fake OpenAI server that answers each
problem from its reference (or with a stub), and the report scores what the smoke saved: pass and fail, the thinking and answer tokens, the pairwise table.
"""

import contextlib
import http.server
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import swift_coding_report as report  # noqa: E402

SMOKE = HERE / 'c2_serving_smoke.py'
DOCUMENT, PROBLEMS = report.load_problems()
STUB = 'def {entry}(*args, **kwargs):\n    return None\n'


class ProblemTests(unittest.TestCase):
    def test_twenty_distinct_problems_with_the_fields_the_smoke_and_the_report_read(self):
        self.assertEqual(len(DOCUMENT['problems']), 20)
        self.assertEqual(len(PROBLEMS), 20)
        for item in DOCUMENT['problems']:
            with self.subTest(problem=item['id']):
                self.assertEqual(sorted(item), ['entry_point', 'id', 'prompt', 'reference', 'tests'])
                self.assertIn('`%s(' % item['entry_point'], item['prompt'], 'the prompt names the function')
                self.assertRegex(item['id'], r'^p[0-9]{2}-[a-z0-9-]+$')
                self.assertIn('def %s' % item['entry_point'], item['reference'])
        self.assertEqual(len(set(item['entry_point'] for item in DOCUMENT['problems'])), 20)
        self.assertEqual(DOCUMENT['instructions'] % 'f', DOCUMENT['instructions'].replace('%s', 'f'))
        self.assertGreaterEqual(DOCUMENT['max_tokens'], 8192)

    def test_nothing_internal_is_in_the_file(self):
        text = (HERE / 'references' / 'swift' / 'coding-problems.json').read_text(encoding='utf-8')
        for needle in ('/home/', 'zot.', '192.168.', '172.16.', '10.10.', 'thatch', 'password', 'token=', '@'):
            self.assertNotIn(needle, text)

    def test_every_reference_passes_its_tests_and_a_stub_fails_them(self):
        for item in DOCUMENT['problems']:
            with self.subTest(problem=item['id']):
                status, detail = report.run_code(item['reference'], item['tests'], 60)
                self.assertEqual((status, detail), ('pass', ''))
                status, _ = report.run_code(STUB.format(entry=item['entry_point']), item['tests'], 60)
                self.assertNotEqual(status, 'pass', 'the tests accept a function that returns None')

    def test_a_slow_answer_is_a_timeout_and_a_wrong_name_an_error(self):
        item = DOCUMENT['problems'][0]
        slow = 'import time\ndef %s(intervals):\n    time.sleep(60)\n' % item['entry_point']
        self.assertEqual(report.run_code(slow, item['tests'], 3)[0], 'timeout')
        status, detail = report.run_code('def other_name(x):\n    return 0\n', item['tests'], 20)
        self.assertEqual(status, 'error')
        self.assertIn('NameError', detail)


class ExtractionTests(unittest.TestCase):
    def test_the_last_block_that_defines_the_entry_point_wins(self):
        content = 'Here:\n```python\ndef f(x):\n    return 1\n```\nand better:\n```python\ndef f(x):\n    return 2\n```\nusage:\n```python\nprint(f(1))\n```\n'
        self.assertEqual(report.extract_code(content, 'f'), 'def f(x):\n    return 2\n')

    def test_fallbacks_and_none(self):
        self.assertEqual(report.extract_code('```\nx = 1\n```', 'f'), 'x = 1\n')
        self.assertEqual(report.extract_code('```text\nnot code\n```\n```python\ny = 2\n```', 'f'), 'y = 2\n')
        self.assertIsNone(report.extract_code('no fences here', 'f'))
        self.assertIsNone(report.extract_code('', 'f'))
        self.assertEqual(report.extract_code('```py\r\ndef f():\r\n    pass\r\n```', 'f'), 'def f():\r\n    pass\r\n')


class TokenTests(unittest.TestCase):
    def test_the_servers_counts_are_used_when_present_and_an_estimate_is_labelled(self):
        self.assertEqual(report.token_counts(dict(completion_tokens=100, reasoning_tokens=70, content_tokens=30)),
                         dict(reasoning_tokens=70, answer_tokens=30, total_tokens=100, token_source='server'))
        estimate = report.token_counts(dict(completion_tokens=100, reasoning='a' * 300, content='b' * 100))
        self.assertEqual((estimate['reasoning_tokens'], estimate['answer_tokens'], estimate['token_source']), (75, 25, 'estimated'))
        self.assertEqual(report.token_counts(dict())['token_source'], 'none')

    def test_the_summary_and_the_pairwise_table(self):
        def run(passes, tokens):
            rows = [dict(id='p%02d-x' % n, status='pass' if ok else 'fail', finish='stop', reasoning_tokens=t // 2, answer_tokens=t - t // 2, total_tokens=t, token_source='server')
                    for n, (ok, t) in enumerate(zip(passes, tokens), 1)]
            return dict(label='x', rows=rows)

        base, other = run([True, True, False, True], [1000, 2000, 3000, 4000]), run([True, False, True, True], [500, 900, 1500, 2000])
        summary = report.summarize(base)
        self.assertEqual((summary['passed'], summary['problems'], summary['mean_total_tokens']), (3, 4, 2500.0))
        compared = report.compare(base, other)
        self.assertEqual((compared['only_first'], compared['only_second'], compared['solved_by_both']), (['p02-x'], ['p03-x'], 2))
        self.assertAlmostEqual(compared['token_ratio_second_over_first_on_both'], (500 + 2000) / float(1000 + 4000))


class FakeServer(object):
    """Answers the chat requests of the coding set: problem k gets its reference in a fenced block (every even k) or a stub (every odd k), after some "reasoning"."""

    def __init__(self):
        fake = self
        self.requests, self.lock = [], threading.Lock()

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('content-length') or 0)))
                if self.path == '/tokenize':
                    text = body['prompt']
                    data = json.dumps(dict(count=len(text) // 4 + 1)).encode()
                    self.send_response(200)
                    self.send_header('content-type', 'application/json')
                    self.send_header('content-length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                prompt = body['messages'][0]['content']
                with fake.lock:
                    fake.requests.append(body)
                index = next(n for n, item in enumerate(DOCUMENT['problems']) if prompt.startswith(item['prompt']))
                item = DOCUMENT['problems'][index]
                code = item['reference'] if index % 2 == 0 else STUB.format(entry=item['entry_point'])
                reasoning = 'Let me think about %s. ' % item['entry_point'] * (3 + index)
                content = 'Here is the solution.\n```python\n' + code + '```\n'
                self.send_response(200)
                self.send_header('content-type', 'text/event-stream')
                self.end_headers()
                for key, text in (('reasoning_content', reasoning), ('content', content)):
                    for start in range(0, len(text), 40):
                        self.wfile.write(('data: %s\n\n' % json.dumps(dict(choices=[dict(delta={key: text[start:start + 40]})]))).encode())
                final = dict(choices=[dict(delta={}, finish_reason='stop')], usage=dict(prompt_tokens=50, completion_tokens=(len(reasoning) + len(content)) // 4))
                self.wfile.write(('data: %s\n\ndata: [DONE]\n\n' % json.dumps(final)).encode())
                self.wfile.flush()

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *failure):
        self.server.shutdown()
        self.server.server_close()
        return False

    @property
    def base(self):
        return 'http://127.0.0.1:%d' % self.server.server_address[1]


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name) / 'coding-set.json'
        with FakeServer() as fake:
            environment = dict(os.environ, CODING_SET_OUT=str(out), CODING_SET_MAX_TOKENS='777', CODING_SET_WORKERS='4', AGREEMENT_LABEL='the-profile')
            done = subprocess.run([sys.executable, '-B', str(SMOKE), fake.base, 'org/served-name', 'coding_set'], capture_output=True, text=True, timeout=300, env=environment, cwd=str(HERE))
            cls.requests = list(fake.requests)
        cls.done, cls.out = done, out
        cls.results = json.loads([line for line in done.stdout.splitlines() if line.startswith('SMOKE_JSON ')][-1][len('SMOKE_JSON '):])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_it_ran_clean_and_recorded_only_the_coding_set(self):
        self.assertEqual(self.done.returncode, 0, self.done.stdout[-800:] + self.done.stderr[-800:])
        self.assertEqual(sorted(self.results), ['coding_set'])
        entry = self.results['coding_set']
        self.assertEqual((entry['problems'], entry['errors'], entry['finish_length'], entry['empty_content'], entry['server_token_counts']), (20, 0, 0, 0, 20))

    def test_one_request_per_problem_with_the_prompt_the_instructions_and_the_configured_budget(self):
        self.assertEqual(len(self.requests), 20)
        for body in self.requests:
            self.assertEqual(body['model'], 'org/served-name')
            self.assertEqual(body['max_tokens'], 777)
            self.assertTrue(body['stream'])
            self.assertEqual(len(body['messages']), 1)
            self.assertNotIn('temperature', body, 'greedy is the contract\'s to force, not this client\'s to ask')
        prompts = sorted(body['messages'][0]['content'] for body in self.requests)
        wanted = sorted(item['prompt'] + '\n\n' + (DOCUMENT['instructions'] % item['entry_point']) for item in DOCUMENT['problems'])
        self.assertEqual(prompts, wanted)
        for body in self.requests:
            for item in DOCUMENT['problems']:
                self.assertNotIn(item['reference'], body['messages'][0]['content'], 'the reference solution is never sent')
                self.assertNotIn(item['tests'], body['messages'][0]['content'], 'the tests are never sent')

    def test_the_saved_file_holds_every_answer_whole_with_the_servers_token_counts(self):
        saved = json.loads(self.out.read_text(encoding='utf-8'))
        self.assertEqual((saved['label'], saved['model'], saved['max_tokens'], saved['version']), ('the-profile', 'org/served-name', 777, 1))
        self.assertEqual([item['id'] for item in saved['results']], [item['id'] for item in DOCUMENT['problems']], 'in the problems\' order')
        for item, row in zip(DOCUMENT['problems'], saved['results']):
            self.assertTrue(row['reasoning'].startswith('Let me think about %s' % item['entry_point']))
            self.assertIn('```python', row['content'])
            self.assertEqual(row['finish'], 'stop')
            self.assertEqual(row['reasoning_tokens'], len(row['reasoning']) // 4 + 1)
            self.assertEqual(row['content_tokens'], len(row['content']) // 4 + 1)

    def test_the_report_scores_it_and_compares_two_runs(self):
        out = io.StringIO()
        report_path = Path(self.tmp.name) / 'report.json'
        with contextlib.redirect_stdout(out):
            code = report.main(['--run', 'base=%s' % self.out, '--run', 'swift=%s' % self.out, '--json-out', str(report_path), '--timeout', '60'])
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn('SWIFT_CODING base: passed 10/20', text)
        self.assertIn('(token counts: server)', text)
        self.assertIn('SWIFT_CODING base vs swift: solved by both 10, only base: -, only swift: -', text)
        written = json.loads(report_path.read_text())
        statuses = dict((row['id'], row['status']) for row in written['runs']['base']['rows'])
        for index, item in enumerate(DOCUMENT['problems']):
            if index % 2 == 0:
                self.assertEqual(statuses[item['id']], 'pass', item['id'])
            else:
                self.assertIn(statuses[item['id']], ('fail', 'error'), item['id'])

    def test_a_file_with_an_unknown_problem_is_refused(self):
        saved = json.loads(self.out.read_text(encoding='utf-8'))
        saved['results'][0]['id'] = 'p99-unknown'
        path = Path(self.tmp.name) / 'bad.json'
        path.write_text(json.dumps(saved))
        with self.assertRaises(ValueError):
            report.score_run(str(path), PROBLEMS)


class SmokeCheckTests(unittest.TestCase):
    def test_a_coding_set_that_errored_is_a_fault_and_a_clean_one_is_not(self):
        import c2_smoke_check as check
        self.assertEqual(check.smoke_problems(dict(coding_set=dict(problems=20, errors=0))), [])
        self.assertEqual(check.smoke_problems(dict(coding_set=dict(problems=20, errors=3))), ['coding_set: 3 of 20 requests errored'])
        self.assertEqual(check.smoke_problems(dict(coding_set=dict(error='boom'))), ['coding_set: boom'])


class TauCompareTests(unittest.TestCase):
    @staticmethod
    def summary(pooled, low, high, a1, buckets, positions, complete=True, image='tp4-swift-1'):
        return dict(complete=complete, image_tag=image, arms=dict(A1=dict(tau=a1, tau_per_turn=dict(p10=a1 - 1.0, p50=a1))),
                    k_inputs=dict(pooled_tau=pooled, ci95=dict(low=low, high=high), per_turn=dict(p10=pooled - 1.0, p50=pooled, mean=pooled), buckets=dict((k, dict(tau=v)) for k, v in buckets.items())),
                    positions=dict(lab=positions))

    def compare_runs(self, base, other):
        import swift_tau_compare as compare
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, value in (('base', base), ('other', other)):
                path = Path(tmp) / (name + '.json')
                path.write_text(json.dumps(value))
                paths.append(str(path))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = compare.main(['--base', paths[0], '--other', paths[1], '--json-out', str(Path(tmp) / 'out.json')])
            return code, out.getvalue(), json.loads((Path(tmp) / 'out.json').read_text())

    def test_overlapping_intervals_are_indistinguishable_and_disjoint_ones_are_not(self):
        base = self.summary(5.0, 4.8, 5.2, 5.1, {'4k': 6.0}, [0.9, 0.8, 0.7])
        code, text, written = self.compare_runs(base, self.summary(4.9, 4.7, 5.1, 5.0, {'4k': 5.8}, [0.9, 0.79, 0.69]))
        self.assertEqual((code, written['verdict']), (0, 'INDISTINGUISHABLE'))
        self.assertIn('SWIFT_TAU pooled', text)
        self.assertIn('SWIFT_TAU bucket_4k', text)
        self.assertIn('SWIFT_TAU position_3', text)
        self.assertEqual(self.compare_runs(base, self.summary(4.0, 3.8, 4.4, 4.0, {'4k': 4.0}, [0.8]))[2]['verdict'], 'LOWER')
        self.assertEqual(self.compare_runs(base, self.summary(5.9, 5.5, 6.3, 6.0, {'4k': 7.0}, [0.95]))[2]['verdict'], 'HIGHER')

    def test_an_incomplete_run_exits_2_and_a_different_image_is_warned_about(self):
        base = self.summary(5.0, 4.8, 5.2, 5.1, {}, [])
        code, text, _ = self.compare_runs(base, self.summary(5.0, 4.8, 5.2, 5.1, {}, [], complete=False, image='other-image'))
        self.assertEqual(code, 2)
        self.assertIn('different images', text)

    def test_a_missing_interval_is_not_established(self):
        import swift_tau_compare as compare
        self.assertEqual(compare.verdict(dict(), dict()), 'NOT_ESTABLISHED')


if __name__ == '__main__':
    unittest.main()
