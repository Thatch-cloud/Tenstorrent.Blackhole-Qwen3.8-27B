"""The NP9 long-answer window pack (references/fusion-jobs/NP9, make_np9_jobs.py) and the smoke tests it runs (c2_serving_smoke long_answer_*, c2_smoke_check.long_answer_problems), held on CPU.

The pack's templates are read by the job reader as the workflow reads them, on the one placeholder image, the approved ship-4 profile and Swift's pinned revision; the smoke tests run for real against test_levern_smoke's fake
OpenAI server with a short answer (LONG_ANSWER_TOKENS), and the smoke check's rules are held with the faults they exist for: a server whose output ceiling clamped the request, a run whose container log never
installed the 32,768-token ceiling, and a long answer that does not begin with the short one."""

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check as check  # noqa: E402
import make_np9_jobs as pack  # noqa: E402
import serving_kv_reservation as reservation  # noqa: E402
import test_w2ln_smoke_shapes as shapes  # noqa: E402
from test_levern_smoke import SMOKE, SmokeRuns  # noqa: E402

FOLDER = HERE / 'references' / 'fusion-jobs' / 'NP9'
PROFILES = HERE / 'qwen_c2_profiles.json'
CATALOG = json.loads((HERE / 'qwen_c2_checkpoints.json').read_text(encoding='utf-8'))
SWIFT = CATALOG['checkpoints'][0]
SHIP = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s4'
CEILING_LINE = '[QWEN-C2] request contract installed: greedy, max_tokens <= 32768, prompt <= 253920, eos [248044, 248046]'


def values(name):
    found = {}
    for line in (FOLDER / ('%s.env' % name)).read_text(encoding='utf-8').splitlines():
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            found[key] = value
    return found


def templates():
    return sorted(path.stem for path in FOLDER.glob('*.env'))


class PackTests(unittest.TestCase):
    def test_the_checked_in_pack_is_what_the_generator_writes(self):
        self.assertEqual(pack.main(['--check']), 0)
        self.assertEqual(templates(), ['B0-build-window', 'N1-long-answer-solo', 'N2-long-answer8-short', 'N3-long-answer8-deep', 'X0-status-rescan-reset', 'Z-reset'])

    def test_every_template_passes_the_job_reader(self):
        for name in templates():
            with self.subTest(template=name):
                done = subprocess.run([sys.executable, '-s', '-B', str(HERE / 'c2_serving_job.py'), str(FOLDER / ('%s.env' % name)), str(PROFILES)], capture_output=True, text=True,
                                      timeout=120, cwd=str(HERE))
                self.assertEqual(done.returncode, 0, done.stdout[-800:] + done.stderr[-800:])

    def test_one_placeholder_image_the_approved_profile_and_swifts_pinned_revision(self):
        profile = json.loads(PROFILES.read_text(encoding='utf-8'))['profiles'][SHIP]
        self.assertEqual(pack.SHIP, SHIP)
        self.assertTrue(profile['owner_traffic_waiver']['decision'].startswith('APPROVED'))
        for name in templates():
            found = values(name)
            with self.subTest(template=name):
                self.assertEqual(found['C2_IMAGE_TAG'], 'tp4-np9-1')
                self.assertEqual(found['C2_CARDS'], 'quad')
                if name.startswith('N'):
                    self.assertEqual((found['C2_PROFILE'], found['C2_MODEL_REPO'], found['C2_MODEL_REVISION']), (SHIP, SWIFT['repo'], SWIFT['revision']))
                    self.assertEqual(found['C2_ACTIONS'], 'reset smoke')
                    self.assertIn('concurrent8_steady', found['C2_SMOKE_TESTS'].split(','), 'the profile asks for the eight-seat checks')
                    self.assertNotIn('C2_SMOKE_PARTIAL', found)
                else:
                    self.assertNotIn('C2_MODEL_REPO', found)
        build = values('B0-build-window')
        self.assertEqual((build['C2_ACTIONS'], build['C2_BAKE_DEFAULT_PROFILE'], build['C2_BAKE_GRAFT']), ('build', SHIP, 'K64j-OQ'))

    def test_the_tests_the_jobs_name_exist_and_are_opt_in(self):
        source = SMOKE.read_text(encoding='utf-8')
        named = set()
        for name in ('N1-long-answer-solo', 'N2-long-answer8-short', 'N3-long-answer8-deep'):
            named |= set(values(name)['C2_SMOKE_TESTS'].split(','))
        long_tests = named - {'warmup', 'concurrent8_steady'}
        self.assertEqual(long_tests, set(check.LONG_ANSWER_TESTS) | {check.LONG_ANSWER_PREFIX_TEST})
        found = shapes.constants()
        self.assertEqual(set(found['LONG_ANSWER_SHAPES']), set(check.LONG_ANSWER_TESTS))
        self.assertIn('for _long_name in LONG_ANSWER_SHAPES:', source)
        self.assertIn('if ONLY and _long_name in ONLY:', source)
        self.assertIn("if ONLY and 'long_answer_prefix' in ONLY:", source)
        self.assertEqual((found['LONG_ANSWER_TOKENS'], found['LONG_ANSWER_PREFIX_TOKENS']), (30000, 16384))
        for name, (seats, _tokens) in found['LONG_ANSWER_SHAPES'].items():
            self.assertEqual(found['SHAPE_SEATS'][name], seats)
            self.assertLessEqual(seats, shapes.SEATS)
        self.assertEqual(found['SHAPE_SEATS']['long_answer_prefix'], 1)

    def test_the_order_lists_every_template_once_and_every_need_resolves(self):
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8').splitlines()
        rows = [line.split() for line in order if line and not line.startswith('#')]
        self.assertEqual([row[0] for row in rows], [row['name'] for row in pack.rows()])
        self.assertEqual(sorted(row[0] for row in rows), templates())
        self.assertEqual({row[1] for row in rows}, {'stop', 'soft'})
        self.assertEqual({row[2] for row in rows}, {'tp4-np9-1'})
        short = {row[0].split('-')[0] for row in rows}
        for line in order:
            if line.startswith('# NEEDS '):
                left, right = line[len('# NEEDS '):].split(' <- ')
                self.assertTrue(set(left.split()) <= short and set(right.split()) <= short, line)

    def test_the_numbers_the_text_quotes_are_the_reservation_modules(self):
        text = ' '.join(' '.join((FOLDER / ('%s.env' % name)).read_text(encoding='utf-8').split()) for name in templates())
        blocks = reservation.request_blocks
        self.assertEqual(blocks(200000, 30000), 3596)
        self.assertEqual(blocks(4096, 30000), 535)
        self.assertEqual(8 * blocks(4096, 30000), 4280)
        self.assertEqual(blocks(96000, 30000), 1971)
        self.assertEqual(8 * blocks(96000, 30000), 15768)
        self.assertLessEqual(15768, 16499)
        for quoted in ('3,596', '535 blocks', '4,280 of 16,499', '1,971', '15,768 of the 16,499', '101,872-token prompts', 'up to 99,104'):
            self.assertIn(quoted, text, quoted)
        eight = lambda prompt, max_tokens: 8 * blocks(prompt, max_tokens) <= 16499
        self.assertTrue(eight(101872, 30000) and not eight(101873 + 63, 30000))
        self.assertTrue(eight(99104, 32768) and not eight(99104 + 64, 32768))

    def test_no_host_address_registry_path_or_token_is_in_a_template(self):
        for path in list(FOLDER.glob('*')):
            text = path.read_text(encoding='utf-8')
            with self.subTest(file=path.name):
                self.assertIsNone(re.search(r'\b\d{1,3}(\.\d{1,3}){3}\b', text))
                for word in ('/home/', 'zot.', '.local', 'sha256:', 'ghp_', 'github_pat', 'password', '@'):
                    self.assertNotIn(word, text, word)


class SmokeShapeTests(SmokeRuns):
    ENV = dict(LONG_ANSWER_TOKENS='700', LONG_ANSWER_PREFIX_TOKENS='300')

    def test_one_seat_at_4k_asks_for_the_long_answer_with_ignore_eos(self):
        results, chats, _completions, _tokenized, _ = self.ran('long_answer_solo_4k', extra_env=self.ENV)
        entry = results['long_answer_solo_4k']
        self.assertNotIn('error', entry, entry)
        self.assertEqual([(chat['body']['max_tokens'], chat['body']['ignore_eos']) for chat in chats], [(700, True)])
        self.assertEqual((entry['asked'], len(entry['users']), entry['users'][0]['tokens'], entry['users'][0]['finish']), (700, 1, 700, 'length'))
        self.assertEqual(check.smoke_problems(results), [])

    def test_one_seat_at_200k_is_a_fitted_prompt(self):
        results, chats, _completions, _tokenized, _ = self.ran('long_answer_solo_200k', extra_env=self.ENV)
        entry = results['long_answer_solo_200k']
        self.assertNotIn('error', entry, entry)
        self.assertEqual(entry['fit']['targets'], [200000])
        self.assertEqual(len(chats), 1)
        self.assertEqual(check.smoke_problems(results), [])

    def test_eight_seats_at_4k_start_together_with_the_same_budget(self):
        results, chats, _completions, _tokenized, _ = self.ran('long_answer8_4k', extra_env=self.ENV)
        entry = results['long_answer8_4k']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'users'})
        self.assertEqual(len(chats), 8, 'eight requests: never a ninth')
        self.assertEqual({chat['body']['max_tokens'] for chat in chats}, {700})
        self.assertTrue(all(chat['body']['ignore_eos'] for chat in chats))
        self.assertEqual([user['tokens'] for user in entry['users']], [700] * 8)
        self.assertEqual(check.smoke_problems(results), [])

    def test_eight_seats_at_96k_are_fitted_prompts(self):
        results, chats, _completions, _tokenized, _ = self.ran('long_answer8_96k', extra_env=self.ENV)
        entry = results['long_answer8_96k']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'users'})
        self.assertEqual(entry['fit']['targets'], [96000] * 8)
        self.assertEqual(len(chats), 8)
        self.assertEqual(check.smoke_problems(results), [])

    def test_the_prefix_test_asks_the_old_ceiling_then_the_long_answer_and_they_agree(self):
        results, chats, _completions, _tokenized, _ = self.ran('long_answer_prefix', extra_env=self.ENV)
        entry = results['long_answer_prefix']
        self.assertNotIn('error', entry, entry)
        self.assertEqual([chat['body']['max_tokens'] for chat in chats], [300, 700])
        self.assertTrue(all(chat['body']['ignore_eos'] for chat in chats))
        self.assertEqual(chats[0]['body']['messages'], chats[1]['body']['messages'], 'the same prompt twice')
        self.assertEqual(entry['asked'], [300, 700])
        self.assertTrue(entry['prefix_equal'], entry)
        self.assertGreater(entry['compared_chars'], 0)
        self.assertIsNone(entry['first_difference'])
        self.assertEqual(check.smoke_problems(results), [])


class CheckRuleTests(unittest.TestCase):
    @staticmethod
    def user(tokens=30000, finish='length', **more):
        return dict(tokens=tokens, finish=finish, text='fine words here', **more)

    def good(self):
        return {'long_answer8_4k': dict(asked=30000, users=[self.user() for _ in range(8)]),
                'long_answer_solo_4k': dict(asked=30000, users=[self.user()]),
                'long_answer_prefix': dict(asked=[16384, 30000], short=dict(tokens=16384, finish='length'), long=dict(tokens=30000, finish='length'), prefix_equal=True,
                                           first_difference=None, compared_chars=40000)}

    def test_a_good_run_is_clean_and_a_run_without_the_tests_is_not_judged(self):
        self.assertEqual(check.long_answer_problems(self.good()), [])
        self.assertEqual(check.long_answer_problems(self.good(), 'noise\n' + CEILING_LINE + '\nnoise'), [])
        self.assertEqual(check.long_answer_problems({}, 'a log with no ceiling line'), [], 'no long-answer test ran: the ceiling line is not asked')

    def test_a_server_whose_ceiling_clamped_the_request_is_a_fault(self):
        results = self.good()
        results['long_answer_solo_4k']['users'] = [self.user(tokens=16384)]
        problems = check.long_answer_problems(results)
        self.assertEqual(len(problems), 1)
        self.assertIn('16384 tokens, asked 30000', problems[0])
        self.assertIn('output ceiling', problems[0])
        results = self.good()
        results['long_answer8_4k']['users'][3] = self.user(finish='stop')
        self.assertTrue(any('not "length"' in problem for problem in check.long_answer_problems(results)))
        results = self.good()
        results['long_answer8_4k']['users'][5] = dict(error='timed out')
        self.assertTrue(any('timed out' in problem for problem in check.long_answer_problems(results)))
        results = self.good()
        results['long_answer8_4k']['users'] = []
        self.assertTrue(check.long_answer_problems(results))

    def test_a_container_log_without_the_32768_ceiling_is_a_fault(self):
        problems = check.long_answer_problems(self.good(), 'the profile says: request contract installed: greedy, max_tokens <= 16384, prompt <= 253920')
        self.assertEqual(len(problems), 1)
        self.assertIn('max_tokens <= 32768', problems[0])
        self.assertIn('output budget did not reach the request contract', problems[0])

    def test_the_prefix_rule_holds_both_answers_and_the_agreement(self):
        results = self.good()
        results['long_answer_prefix']['prefix_equal'] = False
        results['long_answer_prefix']['first_difference'] = 1234
        problems = check.long_answer_problems(results)
        self.assertEqual(len(problems), 1)
        self.assertIn('first difference at character 1234', problems[0])
        results = self.good()
        results['long_answer_prefix']['long']['tokens'] = 16384
        self.assertTrue(any('long answer' in problem and '16384 tokens' in problem for problem in check.long_answer_problems(results)))
        self.assertTrue(check.long_answer_problems(dict(long_answer_prefix=dict(error='boom'))))

    def test_smoke_problems_calls_the_rule_with_the_container_log(self):
        results = self.good()
        self.assertEqual([problem for problem in check.smoke_problems(results, CEILING_LINE) if problem.startswith('long')], [])
        quiet = check.smoke_problems(results, 'some other container log')
        self.assertTrue(any('max_tokens <= 32768' in problem for problem in quiet), quiet)


if __name__ == '__main__':
    unittest.main()
