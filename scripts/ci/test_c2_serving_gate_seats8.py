"""The gate and the smoke at n = the profile's seats (tp4/seats8: QWEN_FAST_M3_BLOCKS=2, eight seats on two 64-row blocks).

c2_serving_gate: the memory plan's users, the churn plan's users (at least CHURN_MIN_REPLACEMENTS replacements over the
seats), live_n_of (the rounds with every seat live and packed), memory_s2_checks and the flag-phase filter at n live.
acceptance_report: live_rate(live=n) and the per-block trace split are held in test_acceptance_report. c2_serving_smoke:
the eight-user opt-in tests and the factorial shapes' eight-user versions. At four seats every value here is what it was."""

import json
import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import acceptance_report  # noqa: E402
import c2_serving_gate as driver  # noqa: E402
import test_c2_serving_gate as base  # noqa: E402

PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))
EIGHT = 'c2-packed-tp4-8'
FOUR = 'c2-packed-tp4'
STAMP = '(EngineCore pid=1) 2026-10-02 03:18:%06.3f | INFO     | serving_worker_hook:_execute:229 - '


def parse(arm):
    return base.parse_harness(arm[1])


class ChurnDefaultsTests(unittest.TestCase):
    def test_four_seats_keep_the_twelve_user_set(self):
        for seats in (1, 2, 3, 4):
            self.assertEqual(driver.churn_defaults(seats), (driver.CHURN_LENGTHS, driver.CHURN_MAX_TOKENS))
        self.assertEqual(len(driver.CHURN_LENGTHS), 12)

    def test_eight_seats_churn_sixteen_users_for_eight_replacements(self):
        lengths, budgets = driver.churn_defaults(8)
        self.assertEqual((len(lengths), len(budgets)), (16, 16))
        self.assertEqual(len(lengths) - 8, driver.CHURN_MIN_REPLACEMENTS)
        self.assertEqual(driver.churn_defaults(5), driver.churn_defaults(8))
        self.assertLess(lengths[-1], 2048, 'one short user last: the one-bucket ladder under churn')
        self.assertTrue(all(100000 <= length <= 123136 for length in lengths[:-1]))
        self.assertGreater(budgets[-1] + lengths[-1], 2048, 'the short user decodes past 2048 of history')
        for length in lengths:
            self.assertIn(length, driver.WARM_LENGTHS, 'every churn length is one the warm plan compiled')

    def test_more_seats_still_leave_the_minimum_replacements(self):
        for seats in (9, 12, 16):
            lengths, budgets = driver.churn_defaults(seats)
            self.assertGreaterEqual(len(lengths) - seats, driver.CHURN_MIN_REPLACEMENTS)
            self.assertEqual(len(lengths), len(budgets))
            self.assertLess(lengths[-1], 2048)

    def test_a_bad_seat_count_is_refused(self):
        for bad in (0, -1, 4.0, '8', None, True and 0):
            with self.subTest(seats=bad), self.assertRaises(ValueError):
                driver.churn_defaults(bad)


class EightSeatPlanTests(unittest.TestCase):
    def test_the_eight_seat_profile_reads_eight_seats(self):
        self.assertEqual(driver.profile_seats(PROFILES, EIGHT), 8)
        self.assertEqual(driver.profile_seats(PROFILES, FOUR), 4)

    def test_churn_at_eight_seats_is_sixteen_users_eight_replacements_and_no_note(self):
        notes = []
        churn, = driver.plan_arms('churn', EIGHT, PROFILES, notes=notes)
        options = parse(churn)
        self.assertEqual((options.users, options.alive_check, notes), (16, 8, []))
        self.assertGreaterEqual(options.users - options.alive_check, 8, 'M11: at least eight replacements')
        self.assertEqual(options.events['max_tokens'], dict(enumerate(driver.CHURN_MAX_TOKENS_8)))
        self.assertEqual(options.prompt_lengths, list(driver.CHURN_LENGTHS_8))
        self.assertEqual(sorted(options.events['ignore_eos']), list(range(16)))

    def test_churn_at_four_seats_is_unchanged(self):
        notes = []
        churn, = driver.plan_arms('churn', FOUR, PROFILES, notes=notes)
        options = parse(churn)
        self.assertEqual((options.users, options.alive_check, notes), (12, 4, []))
        self.assertEqual(options.events['max_tokens'], dict(enumerate(driver.CHURN_MAX_TOKENS)))
        self.assertEqual(options.prompt_lengths, list(driver.CHURN_LENGTHS))

    def test_churn_with_too_few_users_for_eight_seats_notes_the_shortfall(self):
        notes = []
        driver.plan_arms('churn', EIGHT, PROFILES, lengths=[110000] * 12, notes=notes)
        self.assertIn('fewer than the 8 M11 asks for', ' '.join(notes))
        self.assertIn('12 users over 8 seats is 4 replacements', ' '.join(notes))

    def test_memory_at_eight_seats_runs_eight_users_and_at_four_seats_four(self):
        concurrent, short = driver.plan_arms('memory', EIGHT, PROFILES)
        for arm in (concurrent, short):
            options = parse(arm)
            self.assertEqual(options.users, 8)
            self.assertEqual(len(options.prompt_lengths), 8)
        arms = driver.plan_arms('memory', FOUR, PROFILES)
        for arm in arms:
            self.assertEqual(parse(arm).users, 4)

    def test_the_matrix_and_staggered_plans_take_eight_lengths(self):
        lengths = [4096, 8192, 16384, 24576, 32768, 49152, 60000, 120000]
        concurrent, solo = driver.plan_arms('matrix', EIGHT, PROFILES, lengths=lengths, max_tokens=512)
        self.assertEqual(parse(concurrent).users, 8)
        self.assertEqual(parse(solo).sequential_users, 8)
        arms = driver.plan_arms('staggered', EIGHT, PROFILES, lengths=lengths, max_tokens=4096)
        self.assertEqual(parse(arms[0]).users, 8)

    def test_every_eight_seat_plan_fits_one_gate_step(self):
        step = base.WorkflowTests.budget_literals()['step']
        for plan in ('churn', 'memory'):
            arms = {plan: driver.plan_arms(plan, EIGHT, PROFILES)}
            self.assertLessEqual(driver.worst_case_seconds([plan], arms), step, plan)


class LiveNTests(unittest.TestCase):
    def test_four_reads_the_four_live_record_as_before(self):
        record = dict(live=4, median_round_ms=150.0)
        self.assertEqual(driver.live_n_of({'c2_gate_live4': record}, 4), record)
        self.assertEqual(driver.live_n_of({'s2': {'live4': record}}), record)
        self.assertIsNone(driver.live_n_of({'c2_gate_live4': dict(error='x')}, 4))
        self.assertIsNone(driver.live_n_of(None, 4))
        self.assertEqual(driver.live4_of({'c2_gate_live4': record}), record)

    def test_eight_reads_the_hosts_eight_live_record_and_never_the_four_live_one(self):
        four, eight = dict(live=4, median_round_ms=150.0), dict(live=8, median_round_ms=270.0)
        report = {'c2_gate_live4': four, driver.LIVE_N_KEY: eight}
        self.assertEqual(driver.live_n_of(report, 8), eight)
        self.assertEqual(driver.live_n_of(report, 4), four)
        self.assertIsNone(driver.live_n_of({'c2_gate_live4': four}, 8))
        self.assertIsNone(driver.live_n_of({driver.LIVE_N_KEY: four}, 8), 'a record of another live count is not read')
        self.assertIsNone(driver.live_n_of({driver.LIVE_N_KEY: dict(error='boom')}, 8))
        self.assertIsNone(driver.live_n_of({driver.LIVE_N_KEY: None}, 8))

    def test_the_host_reads_the_log_at_the_asked_live_count(self):
        step = lambda at, live, emitted: (
            [STAMP % at + '[PHASE] execute total=%d new=0 cached=%d spec=%d finished=[] preempted=[]' % (
                live * 16, live, live)]
            + ['[PACKED] request=r%d segment=%d position=100 prefix=%d emitted=%d' % (i, i, value, value)
               for i, value in enumerate(emitted)])
        lines = []
        for index in range(5):
            lines += step(10.0 + 0.25 * index, 8, [5] * 8)
        lines += step(12.0, 4, [5] * 4) + step(12.2, 4, [5] * 4) + step(12.4, 1, [5])
        text = '\n'.join(lines)
        eight = driver.host_live_rate(text, 8)
        self.assertEqual((eight['live'], eight['rounds']), (8, 5))
        self.assertEqual(driver.host_live_rate(text)['live'], 4)
        self.assertEqual(driver.host_live_rate(text, 4)['rounds'], 2)
        self.assertIn('error', driver.host_live_rate(None, 8))

    def test_the_runner_stores_the_seats_reading_beside_the_four_live_one(self):
        source = Path(driver.__file__).read_text(encoding='utf-8')
        self.assertIn("report[LIVE_N_KEY] = host_live_rate(log_text, seats)", source)
        self.assertIn("if seats != 4:", source)


def memory_report(hold_decodes=(), engines=8):
    return {'s2': {'dram_hold': {'hold_decodes': list(hold_decodes), 'lines': ['hold']},
                   'before': {'floor_gb': 1.0, 'contiguous_floor_gb': 0.5, 'by_op': {'engine': 1}},
                   'extent_replay': True, 'request_buffers': {}},
            'dram': {'engines': engines, 'min_free_gb': 4.6, 'min_largest_free_mb': 900}}


class MemoryAtEightSeatsTests(unittest.TestCase):
    def test_a_hold_with_a_seat_free_is_a_failed_fit_below_eight_decodes(self):
        problems, shortfalls = driver.memory_s2_checks('memory', memory_report([7]), seats=8)
        self.assertEqual(len(problems), 1)
        self.assertIn('with a seat free (decodes [7] of 8 seats)', problems[0])
        self.assertEqual(shortfalls, [])

    def test_a_hold_while_every_one_of_eight_seats_decodes_is_the_ninth_user_waiting(self):
        problems, shortfalls = driver.memory_s2_checks('memory', memory_report([8]), seats=8)
        self.assertEqual((problems, shortfalls), ([], []))

    def test_at_four_seats_decodes_four_to_seven_are_no_failure_and_five_to_seven_are_not_judged_a_fit_failure(self):
        problems, _ = driver.memory_s2_checks('memory', memory_report([4]), seats=4)
        self.assertEqual(problems, [])
        problems, _ = driver.memory_s2_checks('memory', memory_report([3]), seats=4)
        self.assertEqual(len(problems), 1)

    def test_the_memory_verdict_wants_an_engine_line_per_seat(self):
        report = {'dram': {'engines': 4, 'min_free_gb': 4.0, 'min_largest_free_mb': 900}, 'streams': []}
        short = driver.memory_verdict(report, users=8)
        self.assertEqual(short['verdict'], 'FAIL')
        self.assertIn('4 dram-after-engine lines for 8 users', ' '.join(short['problems']))
        full = driver.memory_verdict(dict(report, dram=dict(report['dram'], engines=8)), users=8)
        self.assertNotIn('dram-after-engine lines', ' '.join(full['problems']))


class RunMemoryWantsTheSeatsTests(unittest.TestCase):
    """run_memory judges the engine lines against the streams the ARM asks for (--users: the profile's seats by default, fewer for M9a), not the four-user default."""

    def run_it(self, seats, engines):
        from unittest import mock
        report = {'dram': {'engines': engines, 'min_free_gb': 4.0, 'min_largest_free_mb': 900}, 'streams': []}
        runner = mock.Mock()
        runner.seats_for.return_value = seats
        runner.s2_for.return_value = False
        runner.arms = {}
        runner.profile = 'p'
        with mock.patch.object(driver, 'run_arm', return_value=report), mock.patch.object(driver, 'arm_problems', return_value=[]):
            return driver.run_memory('memory', runner, [('memory', ['--users', str(seats)], 100)])

    def test_four_engine_lines_do_not_pass_an_eight_seat_profile(self):
        result = self.run_it(8, 4)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('4 dram-after-engine lines for 8 users', ' '.join(result['problems']))

    def test_eight_engine_lines_pass_it_and_four_pass_a_four_seat_profile(self):
        self.assertEqual(self.run_it(8, 8)['verdict'], 'PASS')
        self.assertEqual(self.run_it(4, 4)['verdict'], 'PASS')


class FlagPhaseAtEightTests(unittest.TestCase):
    @staticmethod
    def step(at, live, blocks, packed_users=None, traces=(58.0, 60.0), new=0, window=None):
        packed_users = live if packed_users is None else packed_users
        lines = [STAMP % at + '[PHASE] execute total=%d new=%d cached=%d spec=%d finished=[] preempted=[]' % (
            live * 16, new, live, live)]
        lines += ['[PACKED] request=r%d segment=%d position=%d prefix=5 emitted=5' % (i, i, 1000 + i)
                  for i in range(packed_users)]
        for block in range(blocks):
            lines.append('[PHASE] packed_verify r%d end 70.0 ms' % block)
            lines.append('[PACKED-PHASE] round=%d users=4 bind_ms=0.10 input_ms=1.00 trace_ms=%.2f sync_ms=0.20 '
                         'readback_ms=0.30' % (1 + int(at), traces[block]))
            lines.append('[PACKED-FENCES] round=%d diff_ms=0.50 write_ms=0.40' % (1 + int(at)))
        for user in range(live):
            lines.append('[PHASE] packed_commit r%d end 1.0 ms' % user)
        window = blocks == 1 if window is None else window     # two packed blocks in a round pre-stage nothing (prestage=False)
        for block in range(blocks):
            lines.append('[PACKED-GDN-AFTER-PAIRS] round=%d commits=4 site=window enqueue_ms=1.5 segments=a' % (1 + int(at)))
            if window:
                lines.append('[PACKED-PRESTAGE-WINDOW] round=%d buffers=2 ms=6.50' % (1 + int(at)))
        return lines

    def test_the_default_is_four_live_and_one_block(self):
        lines = self.step(10.0, 4, 1, traces=(58.6,)) + self.step(10.2, 4, 1, traces=(58.6,))
        rounds = driver.flag_phase_rounds(chr(10).join(lines))
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[0]['missing'], [])
        self.assertEqual(rounds[0]['phases']['replay'], round(58.6 + 0.2, 3))

    def test_eight_live_counts_two_blocks_of_lines_and_sums_their_phases(self):
        lines = (self.step(10.0, 8, 2) + self.step(10.25, 8, 2) + self.step(10.5, 8, 2) + self.step(10.75, 1, 1, traces=(58.0,)))
        text = chr(10).join(lines)
        self.assertEqual(driver.flag_phase_rounds(text), [], 'at four live none of these rounds is read')
        rounds = driver.flag_phase_rounds(text, live=8)
        self.assertEqual(len(rounds), 3)
        first = rounds[0]
        self.assertEqual(first['missing'], [])
        self.assertEqual(first['round_ms'], 250.0)
        self.assertEqual(first['phases']['replay'], round(58.0 + 60.0 + 0.4, 3), 'both blocks\' trace and sync')
        self.assertEqual(first['phases']['readback'], round(0.6, 3))
        self.assertEqual(first['phases']['staging'], round(2 * 0.9, 3))
        self.assertEqual(first['phases']['commit'], round(8 * 1.0 + 2 * 1.5, 3))
        self.assertEqual(first['phases']['window'], 0.0, 'two packed blocks pre-stage nothing: no window line is due')
        self.assertEqual(len(first['fingerprint']), 8)

    def test_an_eight_live_round_with_one_block_of_lines_is_reported_missing_not_summed(self):
        lines = self.step(10.0, 8, 1, traces=(58.0,)) + self.step(10.25, 1, 1, traces=(58.0,))
        rounds = driver.flag_phase_rounds(chr(10).join(lines), live=8)
        self.assertEqual(len(rounds), 1)
        self.assertIsNone(rounds[0]['phases'])
        self.assertTrue(rounds[0]['missing'])

    def test_five_live_with_four_packed_lines_is_not_a_packed_round(self):
        lines = self.step(10.0, 5, 1, packed_users=4, traces=(58.0,)) + self.step(10.2, 1, 1, traces=(58.0,))
        self.assertEqual(driver.flag_phase_rounds(chr(10).join(lines), live=5), [])





# ---- the smoke itself, run against a fake OpenAI server -------------------------------------------------------------------------
import http.server  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

SMOKE = HERE / 'c2_serving_smoke.py'
MODEL = 'Qwen/Qwen3.8-27B'


class Fake(object):
    """A tiny OpenAI-style server: records every request in arrival order (seq 0, 1, ...), streams deltas, and can hold
    requests: `gather` makes each request wait until that many have arrived in its group (the group is every request since the
    last time all of them finished), `hold` is a function (request, chunk_number) -> bool naming a chunk after which the stream
    waits for `release`."""

    def __init__(self, gather=1, cap=50, hold=None, on_arrival=None, fail=None):
        self.gather, self.cap, self.hold, self.on_arrival, self.fail = gather, cap, hold, on_arrival, fail
        self.requests, self.sent, self.lock = [], {}, threading.Lock()
        self.release = threading.Event()
        self.arrived = threading.Condition(self.lock)
        self.group_arrivals = 0
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get('content-length') or 0)
                body = json.loads(self.rfile.read(length))
                with fake.lock:
                    seq = len(fake.requests)
                    fake.requests.append(dict(seq=seq, body=body, at=time.time(), sent_at_arrival=dict(fake.sent)))
                    fake.sent[seq] = 0
                    fake.group_arrivals += 1
                    fake.arrived.notify_all()
                if fake.on_arrival:
                    fake.on_arrival(fake, seq, body)
                with fake.lock:
                    fake.arrived.wait_for(lambda: fake.group_arrivals >= fake.gather or fake.gather <= 1, timeout=30)
                if fake.fail and fake.fail(seq):
                    self.send_error(500)
                    return
                tokens = int(body.get('max_tokens') or 1)
                if not body.get('ignore_eos'):
                    tokens = min(tokens, fake.cap)
                if not body.get('stream'):
                    payload = json.dumps(dict(choices=[dict(message=dict(content='ok'), finish_reason='length')],
                                              usage=dict(completion_tokens=tokens, prompt_tokens=50))).encode()
                    self.send_response(200)
                    self.send_header('content-type', 'application/json')
                    self.send_header('content-length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header('content-type', 'text/event-stream')
                self.end_headers()
                for number in range(1, tokens + 1):
                    chunk = dict(choices=[dict(delta=dict(content='t%d ' % number))])
                    self.wfile.write(('data: %s\n\n' % json.dumps(chunk)).encode())
                    self.wfile.flush()
                    with fake.lock:
                        fake.sent[seq] = number
                        fake.arrived.notify_all()
                    if fake.hold and fake.hold(fake.requests[seq]['body'], number):
                        fake.release.wait(30)
                final = dict(choices=[dict(delta={}, finish_reason='length')], usage=dict(completion_tokens=tokens,
                                                                                         prompt_tokens=100))
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

    def prompts(self):
        return [request['body']['messages'][0]['content'] for request in self.requests]


class SmokeCase(unittest.TestCase):
    """Runs c2_serving_smoke.py for real (ONLY = the named tests) against a Fake and reads its SMOKE_JSON line."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / 'corpus' / 'pkg'
        root.mkdir(parents=True)
        # 1.6 MB of distinct python-looking text: eight disjoint 32k-token (117k character) windows.
        lines = ['def function_%d(argument):\n    return argument + %d\n' % (index, index) for index in range(40000)]
        (root / 'module.py').write_text(''.join(lines), encoding='utf-8')
        cls.code_root = str(root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_smoke(self, fake, tests, timeout=240):
        environment = dict(os.environ, SMOKE_CODE_ROOT=self.code_root)
        done = subprocess.run([sys.executable, '-B', str(SMOKE), fake.base, MODEL, ','.join(tests)], capture_output=True,
                              text=True, timeout=timeout, env=environment, cwd=str(HERE))
        self.assertEqual(done.returncode, 0, done.stdout[-1500:] + done.stderr[-1500:])
        line = [text for text in done.stdout.splitlines() if text.startswith('SMOKE_JSON ')][-1]
        return json.loads(line[len('SMOKE_JSON '):])


class EightUserSmokeTests(SmokeCase):
    TESTS = ('concurrent8_code_equal', 'concurrent8_code_32k', 'concurrent8_steady', 'replay_concurrent8',
             'concurrent8_drain')

    def smoke(self):
        """One run of all five tests against a fresh fake that gathers eight at a time; cached on the class."""
        cls = type(self)
        if getattr(cls, 'ran', None) is None:
            fake = Fake(gather=8)
            with fake:
                cls.ran = (self.run_smoke(fake, self.TESTS), list(fake.requests))
        return cls.ran

    def requests_of(self, index):
        return self.smoke()[1][8 * index:8 * index + 8]

    def test_the_tests_run_in_source_order_eight_requests_each(self):
        results, requests = self.smoke()
        self.assertEqual(len(requests), 40)
        for name in self.TESTS:
            self.assertIn(name, results, name)
            self.assertNotIn('error', results[name], results[name])
        # source order: steady (after the like-for-like part), then the coding tests, replay, drain
        order = [request['body'].get('stream') for request in requests]
        self.assertEqual(order[:8], [True] * 8)

    def test_concurrent8_code_equal_is_eight_4k_code_prompts_800_out(self):
        results, _requests = self.smoke()
        users = results['concurrent8_code_equal']['users']
        self.assertEqual(len(users), 8)
        for user in users:
            self.assertEqual((user['finish'], user['tokens'] > 0), ('length', True))
        self.assertEqual(results['concurrent8_code_equal']['corpus']['files'], 1)
        bodies = [request['body'] for request in self.all_for('concurrent8_code_equal')]
        self.assertEqual({body['max_tokens'] for body in bodies}, {800})
        prompts = [body['messages'][0]['content'] for body in bodies]
        self.assertEqual(len(set(prompts)), 8, 'eight different windows of the corpus')
        for prompt in prompts:
            self.assertTrue(prompt.startswith('<repository_context>'))
            self.assertLess(abs(len(prompt) - 4096 * 3.6), 1500)

    def all_for(self, name):
        """The eight requests a test made, by its place in the smoke's source order."""
        order = ['concurrent8_steady', 'concurrent8_code_equal', 'concurrent8_code_32k', 'replay_concurrent8',
                 'concurrent8_drain']
        return self.requests_of(order.index(name))

    def test_concurrent8_code_32k_prompts_are_about_32k_tokens(self):
        bodies = [request['body'] for request in self.all_for('concurrent8_code_32k')]
        self.assertEqual({body['max_tokens'] for body in bodies}, {800})
        for body in bodies:
            self.assertLess(abs(len(body['messages'][0]['content']) - 32768 * 3.6), 2000)
        self.assertEqual(len({body['messages'][0]['content'] for body in bodies}), 8)

    def test_concurrent8_steady_is_eight_distinct_stretches_the_first_four_are_the_four_user_shapes(self):
        eight = [request['body']['messages'][0]['content'] for request in self.all_for('concurrent8_steady')]
        self.assertEqual(len(set(eight)), 8)
        with Fake(gather=4) as fake:
            self.run_smoke(fake, ('concurrent4_steady',))
            four = fake.prompts()
        self.assertEqual(len(four), 4)
        self.assertTrue(set(four) <= set(eight), 'the four-user shape prompts are four of the eight')

    def test_replay_concurrent8_is_eight_non_streamed_300_token_requests_with_replay4s_first_four(self):
        bodies = [request['body'] for request in self.all_for('replay_concurrent8')]
        self.assertEqual({body.get('stream') for body in bodies}, {None})
        self.assertEqual({body['max_tokens'] for body in bodies}, {300})
        prompts = sorted(body['messages'][0]['content'] for body in bodies)
        self.assertEqual(prompts, sorted('Write a unit test for a %s parser in Python.' % name
                                         for name in ('CSV', 'JSON', 'INI', 'TOML', 'YAML', 'XML', 'TSV', 'HTML')))
        with Fake(gather=4) as fake:
            results = self.run_smoke(fake, ('replay_concurrent4',))
            four = sorted(request['body']['messages'][0]['content'] for request in fake.requests)
        self.assertEqual(len(four), 4)
        self.assertTrue(set(four) <= set(prompts), 'replay_concurrent4 is unchanged: its four are the first four of eight')
        self.assertEqual(len(results['replay_concurrent4']['users']), 4)

    def test_concurrent8_drain_budgets_200_to_1600_all_ignore_eos(self):
        bodies = [request['body'] for request in self.all_for('concurrent8_drain')]
        self.assertEqual(sorted(body['max_tokens'] for body in bodies), [200, 400, 600, 800, 1000, 1200, 1400, 1600])
        self.assertEqual({body.get('ignore_eos') for body in bodies}, {True})
        results, _ = self.smoke()
        users = results['concurrent8_drain']['users']
        self.assertEqual(sorted(user['tokens'] for user in users), [200, 400, 600, 800, 1000, 1200, 1400, 1600],
                         'every user ends at its own budget')
        self.assertEqual(results['concurrent8_drain']['budgets'], [200, 400, 600, 800, 1000, 1200, 1400, 1600])

    def test_the_four_user_tests_do_not_send_ignore_eos(self):
        with Fake(gather=4) as fake:
            self.run_smoke(fake, ('concurrent4_steady',))
        self.assertEqual({request['body'].get('ignore_eos') for request in fake.requests}, {None})
        self.assertEqual({request['body']['max_tokens'] for request in fake.requests}, {800})


class SplitSmokeTests(SmokeCase):
    """concurrent5_split: users 0-3 first, user 4 alone once all four stream, the sixth after user 4's 60 chunks."""

    def test_the_arrivals_are_released_by_the_progress_of_the_one_before(self):
        release_after = {}

        def hold(body, number):
            # The long users hold at chunk 10, the fifth at chunk 400: both wait for the sixth arrival, so every earlier user
            # is still streaming when it comes.
            return (body['max_tokens'] == 1600 and number == 10) or (body['max_tokens'] == 1200 and number == 400)

        def on_arrival(fake, seq, body):
            if seq == 5:
                fake.release.set()

        with Fake(gather=1, hold=hold, on_arrival=on_arrival) as fake:
            results = self.run_smoke(fake, ('concurrent5_split',))
        requests = fake.requests
        self.assertEqual(len(requests), 6)
        self.assertEqual([request['body']['max_tokens'] for request in requests[:4]], [1600] * 4)
        self.assertEqual([request['body']['max_tokens'] for request in requests[4:]], [1200, 400])
        self.assertEqual({request['body'].get('ignore_eos') for request in requests}, {True})
        # user 4 arrives only after each of users 0-3 has streamed a token
        for user in range(4):
            self.assertGreaterEqual(requests[4]['sent_at_arrival'].get(user, 0), 1, 'user %d was streaming' % user)
        # the sixth arrives only after user 4 streamed SPLIT_CHUNKS chunks, and every earlier user was still mid-stream
        self.assertGreaterEqual(requests[5]['sent_at_arrival'][4], 60)
        for user in range(4):
            self.assertLess(requests[5]['sent_at_arrival'][user], 1600, 'user %d had not finished' % user)
        entry = results['concurrent5_split']
        self.assertNotIn('error', entry, entry)
        self.assertEqual(entry['user4_chunks_at_sixth_arrival'], 60)
        self.assertEqual(entry['budgets'], [1600, 1600, 1600, 1600, 1200, 400])
        self.assertEqual([user['tokens'] for user in entry['users']], [1600, 1600, 1600, 1600, 1200, 400])
        self.assertGreaterEqual(entry['user4_chunks_at_sixth_arrival'], 50, 'at least 50 rounds with block B narrowed')

    def test_a_dying_fifth_user_does_not_deadlock_the_sixth(self):
        def on_arrival(fake, seq, body):
            if seq == 4:
                fake.release.set()

        with Fake(gather=1, on_arrival=on_arrival, fail=lambda seq: seq == 4) as fake:
            results = self.run_smoke(fake, ('concurrent5_split',))
        self.assertEqual(len(fake.requests), 6)
        entry = results['concurrent5_split']
        self.assertIn('error', entry['users'][4])
        self.assertIsNone(entry['user4_chunks_at_sixth_arrival'])
        self.assertEqual(entry['users'][5]['tokens'], 400)


class SmokeCheckKnowsTheEightUserTests(unittest.TestCase):
    NEW = ('concurrent8_code_equal', 'concurrent8_code_32k', 'concurrent5_split', 'concurrent8_drain', 'concurrent8_steady',
           'replay_concurrent8')

    def test_the_stop_conditions_judge_the_new_tests(self):
        import c2_smoke_check as check
        for name in self.NEW:
            with self.subTest(test=name):
                self.assertIn(name, check.CORE)
        for name in ('concurrent8_code_equal', 'concurrent8_code_32k', 'concurrent5_split', 'concurrent8_drain', 'concurrent8_steady'):
            self.assertIn(name, check.CONCURRENT_TESTS)
            self.assertIn(name, check.TEXT_TESTS)
        self.assertIn('replay_concurrent8', check.REPLAY_TESTS)

    def test_an_error_entry_and_a_dead_user_fail_a_new_test(self):
        import c2_smoke_check as check
        self.assertEqual(check.smoke_problems({'concurrent8_steady': dict(error='ReadTimeout()')}),
                         ['concurrent8_steady: ReadTimeout()'])
        good = dict(tokens=300, finish='length', text='def f(x):\n    return x + 1\n' * 10, completion_tokens=300)
        dead = dict(error='boom')
        problems = check.smoke_problems({'concurrent8_drain': dict(users=[good] * 7 + [dead])})
        self.assertTrue(any('concurrent8_drain user 7' in text for text in problems), problems)
        replay = [dict(status=200, tokens=300, finish='length')] * 7 + [dict(status=500, tokens=0, finish='x')]
        problems = check.smoke_problems({'replay_concurrent8': dict(users=replay)})
        self.assertEqual(problems, ['replay_concurrent8 user 7: status 500'])
        self.assertEqual(check.smoke_problems({'replay_concurrent8': dict(users=[dict(status=200, tokens=3, finish='stop')] * 8)}), [])

    def test_the_four_user_replay_is_judged_as_before(self):
        import c2_smoke_check as check
        problems = check.smoke_problems({'replay_concurrent4': dict(users=[dict(status=200, tokens=0, finish='x')])})
        self.assertEqual(problems, ['replay_concurrent4 user 0: no tokens'])


class SmokeCheckRequestWarmRule(unittest.TestCase):
    """QWEN_FAST_M3_REQUEST_WARM=1: the warm line precedes block 0's capture and the first engine build's programs delta is small."""

    WARM = '[PINDIAG] request widths warmed before the packed traces: rows=(1, 2, 4) programs=700->938 ms=900'
    CAPTURE = '[PINDIAG] packed blocks capture block=0 programs=938->938'
    ENGINE = '[PINDIAG] four-card engine programs=%d->%d req=r1'

    def log(self, *lines):
        return '\n'.join(lines)

    def problems(self, text, flag='1'):
        import c2_smoke_check as check
        return check.check('', text, False, env={'QWEN_FAST_TP': '4', 'QWEN_FAST_M3_REQUEST_WARM': flag})[0]

    def only_warm(self, text, flag='1'):
        return [p for p in self.problems(text, flag) if 'request warm' in p or 'request widths' in p or 'engine build' in p
                or 'block 0' in p]

    def test_a_warm_before_the_capture_and_a_small_first_engine_delta_pass(self):
        text = self.log(self.WARM, self.CAPTURE, self.ENGINE % (938, 944), self.ENGINE % (944, 948))
        self.assertEqual(self.only_warm(text), [])

    def test_the_first_engine_delta_is_recorded_not_bounded(self):
        import c2_smoke_check as check
        self.assertFalse(hasattr(check, 'FIRST_ENGINE_PROGRAMS_MAX'))
        for build in ((700, 716), (768, 1006)):
            text = self.log(self.WARM, self.CAPTURE, self.ENGINE % build)
            self.assertEqual(self.only_warm(text), [])
            facts = check.request_warm_problems(text)[1]
            self.assertEqual(facts['first_engine_programs'], build[1] - build[0])
            self.assertEqual(facts['request_warm_programs'], 238)

    def test_a_warm_that_compiled_nothing_fails(self):
        empty = self.WARM.replace('700->938', '938->938')
        failed = self.only_warm(self.log(empty, self.CAPTURE, self.ENGINE % (938, 944)))
        self.assertEqual(len(failed), 1)
        self.assertIn('compiled 0 programs', failed[0])
        none = self.WARM.replace('700->938', 'None->None')
        self.assertEqual(self.only_warm(self.log(none, self.CAPTURE)), [])

    def test_later_engine_builds_are_not_bounded(self):
        text = self.log(self.WARM, self.CAPTURE, self.ENGINE % (938, 944), self.ENGINE % (944, 1100))
        self.assertEqual(self.only_warm(text), [])

    def test_a_missing_or_late_warm_fails(self):
        self.assertTrue(any('never ran' in p for p in self.only_warm(self.log(self.CAPTURE, self.ENGINE % (1, 3)))))
        late = self.only_warm(self.log(self.CAPTURE, self.WARM, self.ENGINE % (1, 3)))
        self.assertEqual(len(late), 1)
        self.assertIn('after block 0 captured', late[0])

    def test_flag_off_the_rule_is_not_applied(self):
        text = self.log(self.CAPTURE, self.ENGINE % (768, 1006))
        self.assertEqual(self.only_warm(text, flag='0'), [])
        import c2_smoke_check as check
        self.assertEqual(check.check('', text, False, env={'QWEN_FAST_TP': '4'})[1].get('first_engine_programs'), None)


if __name__ == '__main__':
    unittest.main()
