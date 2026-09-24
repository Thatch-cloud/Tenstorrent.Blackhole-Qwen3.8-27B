"""real_text_compare: a concurrent real-text arm against a sequential one, offline, stdlib only."""

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import real_text_compare as rtc  # noqa: E402
from lever_n_m3native_gate import BEGIN, END  # noqa: E402

SHAS = ['%064x' % (index + 1) for index in range(4)]
TEXTS = ['def test_one():\n    assert f(1) == 2\n' * 20, 'Issue 1: the loop ' * 30,
         'File a.py does X. ' * 25, '```python\ndef g(x):\n    return x\n```\n' * 15]


CONFIGURATION = dict(QWEN_FAST_SDPA_PF='1', QWEN_FAST_PACKED_PROPOSAL='1')


def report(texts=TEXTS, finish=('length',) * 4, shas=SHAS, users=4, sequential=0, rates=(28.0, 27.0, 29.0, 26.0),
           means=(7.1, 7.4, 6.9, 7.8), completion=(256,) * 4, max_tokens=256, configuration=CONFIGURATION):
    return dict(
        users=users, sequential_users=sequential, context=33024, gate_passed=True, max_tokens=max_tokens,
        qwen_configuration=dict(configuration) if configuration is not None else None,
        streams=[dict(text=text, finish_reason=reason, completion_tokens=count)
                 for text, reason, count in zip(texts, finish, completion)],
        real_text=dict(users=[dict(user=i, prompt_sha256=sha, prompt_tokens=32768) for i, sha in enumerate(shas)]),
        comparisons=[dict(user=i, prompt_sha256=sha, max_tokens=max_tokens) for i, sha in enumerate(shas)],
        acceptance=dict(summary_line='[ACCEPT] synthetic %d' % users, users=[
            dict(user=i, full_draft=dict(rounds=30, mean_emitted=m, p_emitted_gt_8=0.3, p_emitted_gt_11=0.1,
                                         p_emitted_eq_16=0.02, max_emitted=16), all=dict(mean_emitted=m - 0.5))
            for i, m in enumerate(means)]),
        decode_rate=dict(users=[dict(user=i, steady_tok_s=r, all_active_tok_s=r - 4, median_gap_tok_s=r - 2)
                                for i, r in enumerate(rates)]))


class ExtractTests(unittest.TestCase):
    def test_plain_json_gate_stdout_and_github_logs(self):
        body = json.dumps(report(), indent=2)
        self.assertEqual(rtc.extract_report(body)['users'], 4)
        stdout = '[REALTEXT] building\n%s\n%s\n%s\n<<<M3NATIVE_GATE_LOG_BEGIN>>>\nnoise\n' % (BEGIN, body, END)
        self.assertEqual(rtc.extract_report(stdout)['context'], 33024)
        stamped = '\n'.join('2026-09-24T01:02:03.4567890Z ' + line for line in stdout.split('\n'))
        self.assertEqual(rtc.extract_report(stamped)['streams'][1]['text'], TEXTS[1])
        gh = '\n'.join('arm\tRun the native M3 gate\t2026-09-24T01:02:03.4567890Z ' + line for line in stdout.split('\n'))
        self.assertEqual(rtc.extract_report(gh)['streams'][3]['text'], TEXTS[3])
        with self.assertRaises(ValueError):
            rtc.extract_report('no report here\n')


class CompareTests(unittest.TestCase):
    def test_identical_arms_are_exact_with_rates_and_ratios(self):
        result = rtc.compare(report(), report(users=1, sequential=4, rates=(40.0, 40.0, 40.0, 40.0)))
        self.assertTrue(result['exact'])
        self.assertEqual([u['verdict'] for u in result['users']], ['exact'] * 4)
        self.assertEqual(result['users'][0]['rate_ratio_steady'], 0.7, 'the ratio is of the steady rates')
        self.assertEqual(result['users'][0]['rate_ratio_all_active'], 0.667)
        self.assertEqual(result['users'][0]['acceptance_concurrent']['mean'], 7.1)
        self.assertIsNone(result['users'][0]['first_divergence'])
        self.assertIs(result['same_configuration'], True)
        rendered = rtc.render(result)
        self.assertIn('EXACT', rendered)
        self.assertIn('configurations: identical QWEN_* flags', rendered)
        self.assertNotIn('different configurations', rendered)

    def test_a_prefix_at_the_same_budget_is_a_divergence_near_the_end(self):
        """Both arms spent 256 tokens: 'foo' against 'fo' + 'obar' is a strict prefix, but the token
        sequences differ, so it is not exact (the review's case)."""
        texts, single_texts = list(TEXTS), list(TEXTS)
        texts[0], single_texts[0] = TEXTS[0] + 'foo', TEXTS[0] + 'foobar'
        result = rtc.compare(report(texts=texts), report(texts=single_texts, users=1, sequential=4))
        self.assertEqual(result['users'][0]['verdict'], 'diverged')
        self.assertTrue(result['users'][0]['identical_prefix'])
        self.assertFalse(result['exact'])

    def test_the_same_text_in_a_different_token_count_is_not_exact(self):
        result = rtc.compare(report(completion=(256, 255, 256, 256)), report(users=1, sequential=4))
        self.assertEqual(result['users'][1]['verdict'], 'token-mismatch')
        self.assertFalse(result['exact'])

    def test_different_flag_sets_are_named_and_the_columns_labelled(self):
        single = report(users=1, sequential=4, configuration=dict(QWEN_FAST_SDPA_PF='1', QWEN_FAST_DRAFT_BF8='0'))
        result = rtc.compare(report(configuration=dict(CONFIGURATION, QWEN_FAST_DRAFT_BF8='1')), single)
        self.assertEqual(result['configuration_diff'], dict(QWEN_FAST_DRAFT_BF8=['1', '0'],
                                                             QWEN_FAST_PACKED_PROPOSAL=['1', None]))
        self.assertIs(result['same_configuration'], False)
        rendered = rtc.render(result)
        self.assertIn('configurations DIFFER', rendered)
        self.assertIn('  QWEN_FAST_DRAFT_BF8=1 | 0', rendered)
        self.assertIn('concurrent (different configurations) [ACCEPT]', rendered)
        unknown = rtc.compare(report(configuration=None), single)
        self.assertIsNone(unknown['same_configuration'])
        self.assertIn('not recorded in both reports', rtc.render(unknown))

    def test_a_divergence_reports_its_first_character(self):
        texts = list(TEXTS)
        texts[2] = TEXTS[2][:40] + 'Z' + TEXTS[2][41:]
        result = rtc.compare(report(texts=texts), report(users=1, sequential=4))
        self.assertFalse(result['exact'])
        self.assertEqual(result['users'][2]['verdict'], 'diverged')
        self.assertEqual(result['users'][2]['first_divergence'], 40)
        self.assertIn('NOT EXACT', rtc.render(result))

    def test_a_stream_cut_by_eos_in_one_arm_only_is_a_finding(self):
        texts = list(TEXTS)
        texts[1] = TEXTS[1][:100]
        finish = ('length', 'stop', 'length', 'length')
        result = rtc.compare(report(texts=texts, finish=finish), report(users=1, sequential=4))
        self.assertEqual(result['users'][1]['verdict'], 'eos-mismatch')
        self.assertEqual(result['users'][1]['first_divergence'], 100)
        self.assertFalse(result['exact'])

    def test_a_prefix_passes_only_when_the_budgets_differ_and_it_hit_its_own(self):
        texts = list(TEXTS)
        texts[0] = TEXTS[0][:200]
        shorter = report(texts=texts, completion=(128, 256, 256, 256), max_tokens=128)
        result = rtc.compare(shorter, report(users=1, sequential=4))
        self.assertEqual(result['users'][0]['verdict'], 'prefix')
        self.assertTrue(result['users'][0]['partial'])
        self.assertEqual((result['users'][0]['max_tokens_concurrent'], result['users'][0]['max_tokens_single']), (128, 256))
        self.assertEqual([u['verdict'] for u in result['users']][1:], ['exact'] * 3)
        same_budget = rtc.compare(report(texts=texts, completion=(128, 256, 256, 256)), report(users=1, sequential=4))
        self.assertEqual(same_budget['users'][0]['verdict'], 'diverged')
        unknown = report(texts=texts, completion=(128, 256, 256, 256), max_tokens=None)
        self.assertEqual(rtc.compare(unknown, report(users=1, sequential=4))['users'][0]['verdict'], 'diverged',
                         'budgets that are not recorded are not known to differ')

    def test_same_text_different_finish_and_errors_and_prompt_mismatch(self):
        finish = ('stop', 'length', 'length', 'length')
        self.assertEqual(rtc.compare(report(finish=finish), report())['users'][0]['verdict'], 'finish-mismatch')
        broken = report()
        broken['streams'][3]['error'] = 'TimeoutError: timed out'
        self.assertEqual(rtc.compare(broken, report())['users'][3]['verdict'], 'error')
        shas = list(SHAS)
        shas[2] = 'f' * 64
        result = rtc.compare(report(shas=shas), report())
        self.assertEqual(result['users'][2]['verdict'], 'prompt-mismatch')
        self.assertFalse(result['exact'])

    def test_main_reads_files_and_exits_by_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            concurrent, single, out = (Path(directory, name) for name in ('c.log', 's.json', 'out.json'))
            concurrent.write_text('%s\n%s\n%s\n' % (BEGIN, json.dumps(report(), indent=2), END), encoding='utf-8')
            single.write_text(json.dumps(report(users=1, sequential=4)), encoding='utf-8')
            with redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(rtc.main([str(concurrent), str(single), '--json', str(out)]), 0)
            self.assertTrue(json.loads(out.read_text(encoding='utf-8'))['exact'])
            self.assertIn('[ACCEPT] synthetic 4', printed.getvalue())
            texts = list(TEXTS)
            texts[0] = 'different'
            single.write_text(json.dumps(report(texts=texts, users=1, sequential=4)), encoding='utf-8')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(rtc.main([str(concurrent), str(single)]), 1)
            single.write_text('nothing', encoding='utf-8')
            with redirect_stdout(io.StringIO()), open(os.devnull, 'w') as sink:
                stderr, sys.stderr = sys.stderr, sink
                try:
                    self.assertEqual(rtc.main([str(concurrent), str(single)]), 2)
                finally:
                    sys.stderr = stderr


if __name__ == '__main__':
    unittest.main()
