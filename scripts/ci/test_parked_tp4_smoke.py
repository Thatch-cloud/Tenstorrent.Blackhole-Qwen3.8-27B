"""Engine reuse: the smoke tests it adds to c2_serving_smoke.py, run for real against the fake OpenAI server test_levern_smoke defines.

parked_equal (the exact-length ladder through the page, chunk and history-window boundaries), parked_budgets (one prompt at budgets 1, 2, 3, 4, 5, 16 and
256: a cold engine captures only the widths its budget holds), parked_churn and parked_churn_long (sixteen users on the eight seats, waves of mixed lengths
and budgets, every stream's hashes compared between the arms), parked_abort_reuse (a client drop mid-decode, then a full request, a one-token request and an
ignore_eos one on the slot it freed) and parked_turns (eight seats, closed loops of coding turns: ER5's pairing). The fake answers /tokenize with its own
tokenizer (three characters a token) and the chat and completions endpoints with deterministic text."""

import ast
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check as check  # noqa: E402
from test_levern_smoke import SMOKE, Fake, SmokeRuns, function_source  # noqa: E402

FAST = {'SMOKE_PARKED_ABORT_AFTER_S': '0.05', 'SMOKE_PARKED_THINK_S': '0.01', 'SMOKE_PARKED_STAGGER_S': '0.01'}


class LadderTests(SmokeRuns):
    def test_the_ladder_is_sent_at_exactly_its_token_counts_as_ids(self):
        results, chats, completions, _tokenized, _ = self.ran('parked_equal')
        entry = results['parked_equal']
        self.assertNotIn('error', entry, entry)
        lengths = [63, 64, 65, 127, 128, 129, 2047, 2048, 2049, 4096, 32768]
        self.assertEqual(entry['lengths'], lengths)
        self.assertEqual([len(body['prompt']) for body in completions], lengths)
        self.assertEqual({body['max_tokens'] for body in completions}, {256})
        self.assertEqual(chats, [])
        for length in lengths:
            row = entry['prompts'][str(length)]
            self.assertEqual((row['prompt_tokens'], row['prompt_tokens_sent']), (length, length))
        self.assertEqual(check.smoke_problems(results), [])

    def test_the_budget_ladder_is_one_prompt_at_every_budget(self):
        results, _chats, completions, _tokenized, _ = self.ran('parked_budgets')
        entry = results['parked_budgets']
        self.assertNotIn('error', entry, entry)
        self.assertEqual([body['max_tokens'] for body in completions], [1, 2, 3, 4, 5, 16, 256])
        self.assertEqual(len({tuple(body['prompt']) for body in completions}), 1, 'one prompt: only the budget changes')
        self.assertEqual(len(completions[0]['prompt']), 4097)
        self.assertEqual(check.smoke_problems(results), [])

    def test_a_budget_row_that_overshoots_is_a_problem(self):
        results, *_ = self.ran('parked_budgets')
        broken = json.loads(json.dumps(results))
        broken['parked_budgets']['prompts']['2']['tokens'] = 3
        self.assertTrue(any('past the budget' in text for text in check.parked_smoke_problems(broken)))


class ChurnTests(SmokeRuns):
    def test_sixteen_users_arrive_per_wave_with_the_scripts_budgets_and_ignore_eos(self):
        results, chats, _completions, _tokenized, _ = self.ran('parked_churn')
        entry = results['parked_churn']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'users'})
        self.assertEqual((entry['waves'], entry['admissions'], len(entry['users'])), (4, 64, 64))
        self.assertEqual(len(chats), 64)
        self.assertTrue(all(chat['body']['ignore_eos'] for chat in chats))
        self.assertTrue({chat['body']['max_tokens'] for chat in chats} <= {16, 64, 128, 256})
        self.assertTrue(all('delta_stamps' not in user for user in entry['users']))
        self.assertEqual(check.smoke_problems(results), [])

    def test_the_script_is_deterministic_and_mixed(self):
        namespace = {'PARKED_CHURN_LENGTHS': (60, 255, 1536, 2047, 2048, 2049, 4096, 4096, 8192, 8192, 16384, 30000),
                     'PARKED_CHURN_BUDGETS': (16, 64, 128, 256)}
        exec(function_source('churn_script'), namespace)
        first, again = namespace['churn_script'](2, 16), namespace['churn_script'](2, 16)
        self.assertEqual(first, again)
        self.assertNotEqual(first, namespace['churn_script'](3, 16))
        self.assertGreater(len(set(first[0])), 4)
        self.assertGreater(len(set(first[1])), 2)
        source = SMOKE.read_text(encoding='utf-8')
        for name in ('PARKED_CHURN_LENGTHS', 'PARKED_CHURN_BUDGETS'):
            declared = [node for node in ast.parse(source).body if isinstance(node, ast.Assign) and node.targets[0].id == name][0]
            self.assertEqual(tuple(ast.literal_eval(declared.value)), namespace[name])

    def test_the_long_variant_is_at_least_two_hundred_admissions(self):
        source = SMOKE.read_text(encoding='utf-8')
        self.assertIn("return parked_churn_run('parked_churn_long', 14)", source)
        self.assertGreaterEqual(14 * 16, 200)


class LifecycleTests(SmokeRuns):
    def test_a_dropped_request_is_followed_by_a_full_a_one_token_and_an_ignore_eos_request(self):
        results, chats, *_ = self.ran('parked_abort_reuse', extra_env=FAST)
        entry = results['parked_abort_reuse']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'users'})
        self.assertEqual((entry['rounds'], len(entry['users'])), (6, 18))
        budgets = [chat['body']['max_tokens'] for chat in chats]
        self.assertEqual(sorted(set(budgets)), [1, 64, 128, 600])
        self.assertEqual([user['tokens'] for user in entry['users']][1::3], [1] * 6)
        self.assertEqual(check.smoke_problems(results), [])

    def test_the_turns_are_closed_loops_of_eight_seats_with_the_latency_numbers(self):
        results, chats, *_ = self.ran('parked_turns', extra_env=FAST)
        entry = results['parked_turns']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'users'})
        self.assertEqual((entry['seats'], entry['turns'], len(entry['users']), len(chats)), (8, 5, 40, 40))
        for user in entry['users']:
            self.assertIn('longest_gap_s', user)
            self.assertIn('ttft', user)
            self.assertNotIn('delta_stamps', user)
        self.assertGreater(entry['tokens_per_s'], 0)
        self.assertEqual(entry['completion_tokens'], sum(user['tokens'] for user in entry['users']))
        self.assertEqual(check.smoke_problems(results), [])


class ListedTests(unittest.TestCase):
    def test_every_parked_test_the_smoke_defines_is_one_the_check_judges_and_the_smoke_records(self):
        source = SMOKE.read_text(encoding='utf-8')
        for name in check.PARKED_TESTS:
            self.assertIn('def %s(' % name, source)
            self.assertIn("'%s'" % name, source)
        self.assertEqual(set(check.PARKED_ROW_TESTS) | set(check.PARKED_USER_TESTS), set(check.PARKED_TESTS))


if __name__ == '__main__':
    unittest.main()
