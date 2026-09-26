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


def diverged(index=0, at=5):
    texts = list(TEXTS)
    texts[index] = texts[index][:at] + 'X' + texts[index][at + 1:]
    return texts


class PolicyTests(unittest.TestCase):
    """The exactness policy (c2-serve-for-real-plan 2.2 item 5): a solo reference on the same image and
    arithmetic, full answers, one re-run of both arms on a first divergence."""

    def solo(self, **kwargs):
        return report(users=1, sequential=4, **kwargs)

    def test_identical_arms_pass_without_a_rerun(self):
        result = rtc.exactness_policy(report(), self.solo())
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual([u['verdict'] for u in result['users']], ['IDENTICAL'] * 4)
        self.assertIsNone(result['reproducible'])

    def test_a_first_divergence_asks_for_the_rerun(self):
        result = rtc.exactness_policy(report(texts=diverged(1)), self.solo())
        self.assertEqual(result['verdict'], 'RERUN')
        self.assertEqual(result['users'][1]['verdict'], 'RERUN')
        self.assertEqual(result['users'][1]['first_divergence'], 5)

    def test_a_divergence_that_reproduces_fails(self):
        result = rtc.exactness_policy(report(texts=diverged(1)), self.solo(),
                                      rerun=(report(texts=diverged(1)), self.solo()))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['users'][1], dict(user=1, verdict='DIVERGED', first='DIVERGED', rerun='DIVERGED',
                                                  first_divergence=5, rerun_divergence=5, reason=None))
        self.assertEqual(result['reproducible'], dict(concurrent=True, single=True))

    def test_a_divergence_that_does_not_reproduce_is_unstable_not_a_pass(self):
        result = rtc.exactness_policy(report(texts=diverged(2)), self.solo(), rerun=(report(), self.solo()))
        self.assertEqual(result['verdict'], 'UNSTABLE')
        self.assertEqual(result['users'][2]['verdict'], 'UNSTABLE')
        self.assertEqual(result['reproducible'], dict(concurrent=False, single=True))
        # A user identical first and divergent in the re-run is nondeterministic too.
        late = rtc.exactness_policy(report(texts=diverged(2)), self.solo(),
                                    rerun=(report(texts=diverged(3)), self.solo()))
        self.assertEqual([u['verdict'] for u in late['users']], ['IDENTICAL', 'IDENTICAL', 'UNSTABLE', 'UNSTABLE'])

    def test_an_arithmetic_difference_makes_a_divergence_not_comparable(self):
        other = dict(CONFIGURATION, QWEN_FAST_SINGLE_GATEUP='1')
        result = rtc.exactness_policy(report(texts=diverged(0), configuration=other), self.solo())
        self.assertEqual(result['verdict'], 'NOT_COMPARABLE')
        self.assertIn('QWEN_FAST_SINGLE_GATEUP', result['users'][0]['reason'])
        # Identical text under different arithmetic is still identical.
        self.assertEqual(rtc.exactness_policy(report(configuration=other), self.solo())['verdict'], 'PASS')

    def test_logging_audit_and_serving_flags_are_not_arithmetic(self):
        noisy = dict(CONFIGURATION, QWEN_FAST_PHASE_LOG='1', QWEN_FAST_VERIFY_T2_AUDIT='1', QWEN_C2_PROFILE='c2',
                     QWEN_FAST_FAULTHANDLER='0', QWEN_C2_SERVING='1')
        self.assertEqual(rtc.arithmetic_diff(report(configuration=noisy), self.solo()), {})
        self.assertEqual(sorted(rtc.configuration_diff(report(configuration=noisy), self.solo())),
                         ['QWEN_C2_PROFILE', 'QWEN_C2_SERVING', 'QWEN_FAST_FAULTHANDLER', 'QWEN_FAST_PHASE_LOG',
                          'QWEN_FAST_VERIFY_T2_AUDIT'])
        result = rtc.exactness_policy(report(texts=diverged(0), configuration=noisy), self.solo())
        self.assertEqual(result['verdict'], 'RERUN', 'a divergence under neutral flags is still a divergence')
        self.assertIsNone(rtc.arithmetic_diff(report(configuration=None), self.solo()))

    def test_errors_fail_and_prompt_or_budget_differences_are_not_comparable(self):
        streams = report()
        streams['streams'][3] = dict(error='HTTP 400')
        self.assertEqual(rtc.exactness_policy(streams, self.solo())['verdict'], 'FAIL')
        shas = list(SHAS)
        shas[1] = 'e' * 64
        self.assertEqual(rtc.exactness_policy(report(shas=shas), self.solo())['users'][1]['verdict'], 'NOT_COMPARABLE')
        short = [text[:40] for text in TEXTS]
        budget = rtc.exactness_policy(report(texts=short, completion=(64,) * 4, max_tokens=64), self.solo())
        self.assertEqual(budget['verdict'], 'NOT_COMPARABLE', 'a prefix: the arms ran different budgets')
        self.assertEqual(rtc.exactness_policy(report(users=0, texts=()), self.solo(texts=()))['verdict'], 'FAIL')

    def test_full_answers_are_compared_not_prefixes(self):
        """The same budget ending in an earlier EOS on one side only is a divergence (eos-mismatch)."""
        texts = list(TEXTS)
        texts[0] = TEXTS[0][:100]
        result = rtc.exactness_policy(report(texts=texts, finish=('stop', 'length', 'length', 'length'),
                                             completion=(40, 256, 256, 256)), self.solo())
        self.assertEqual(result['users'][0]['verdict'], 'RERUN')

    def test_main_policy_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory, name) for name in ('c.json', 's.json', 'c2.json', 's2.json', 'out.json')]
            paths[0].write_text(json.dumps(report(texts=diverged(1))), encoding='utf-8')
            for path in paths[1:4]:
                path.write_text(json.dumps(self.solo()), encoding='utf-8')
            with redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(rtc.main([str(paths[0]), str(paths[1]), '--policy']), 5)
                self.assertEqual(rtc.main([str(paths[0]), str(paths[1]), '--policy', '--rerun', str(paths[2]),
                                           str(paths[3]), '--json', str(paths[4])]), 4)
                self.assertEqual(rtc.main([str(paths[2]), str(paths[1]), '--policy']), 0)
            self.assertIn('POLICY UNSTABLE', printed.getvalue())
            self.assertEqual(json.loads(paths[4].read_text(encoding='utf-8'))['verdict'], 'UNSTABLE')
            with redirect_stdout(io.StringIO()), open(os.devnull, 'w') as sink:
                stderr, sys.stderr = sys.stderr, sink
                try:
                    with self.assertRaises(SystemExit):
                        rtc.main([str(paths[0]), str(paths[1]), '--rerun', str(paths[2]), str(paths[3])])
                    with self.assertRaises(SystemExit):
                        rtc.main([str(paths[0])])
                finally:
                    sys.stderr = stderr


class LifecycleTests(unittest.TestCase):
    """lifecycle_pass: what each event leaves must agree with the solo text as far as it goes."""

    def solo(self, finish=('stop', 'length', 'length', 'length')):
        return report(users=1, sequential=4, max_tokens=1024, finish=finish)

    def event(self, streams, budgets=(1024,) * 4, ignore_eos=(False,) * 4):
        base = report(max_tokens=1024)
        base['streams'] = streams
        base['comparisons'] = [dict(user=i, prompt_sha256=SHAS[i], max_tokens=budgets[i], ignore_eos=ignore_eos[i])
                               for i in range(4)]
        return base

    def full(self, index, finish='length'):
        return dict(text=TEXTS[index], finish_reason=finish, completion_tokens=256)

    def test_drops_budgets_and_ignore_eos_are_consistent_prefixes(self):
        streams = [dict(text=TEXTS[0][:17], dropped='after 3 chunks'),                    # dropped mid-answer
                   dict(text='', dropped='no byte within 2 s'),                            # cancelled in prefill
                   dict(text=TEXTS[2][:3], finish_reason='length', completion_tokens=1),   # max_tokens=1
                   dict(text=TEXTS[3] + ' and on past EOS', finish_reason='length', completion_tokens=1024)]
        result = rtc.lifecycle_pass(self.event(streams, budgets=(1024, 1024, 1, 1024),
                                               ignore_eos=(False, False, False, True)),
                                    self.solo(finish=('stop', 'length', 'length', 'stop')))
        self.assertEqual([u['verdict'] for u in result['users']], ['IDENTICAL'] * 4)
        self.assertEqual([u['detail'] for u in result['users']],
                         ['dropped after 3 chunks', 'dropped no byte within 2 s', 'its own budget 1', 'ignore_eos past the solo EOS'])

    def test_a_drop_or_a_cut_that_disagrees_diverges(self):
        streams = [dict(text='Xdef', dropped='after 1 chunks'), self.full(1), self.full(2),
                   dict(text=TEXTS[3][:5] + '!', finish_reason='length', completion_tokens=1)]
        result = rtc.lifecycle_pass(self.event(streams, budgets=(1024, 1024, 1024, 1)), self.solo())
        self.assertEqual([u['verdict'] for u in result['users']], ['DIVERGED', 'IDENTICAL', 'IDENTICAL', 'DIVERGED'])
        policy = rtc.exactness_policy(self.event(streams, budgets=(1024, 1024, 1024, 1)), self.solo(),
                                      pass_function=rtc.lifecycle_pass)
        self.assertEqual(policy['verdict'], 'RERUN')

    def test_a_survivor_is_compared_in_full_and_an_error_is_an_error(self):
        streams = [self.full(0, 'stop'), dict(text=TEXTS[1][:-1] + '?', finish_reason='length', completion_tokens=256),
                   dict(error='HTTP 500'), self.full(3)]
        result = rtc.lifecycle_pass(self.event(streams), self.solo())
        self.assertEqual([u['verdict'] for u in result['users']], ['IDENTICAL', 'DIVERGED', 'ERROR', 'IDENTICAL'])
        # ignore_eos when the solo run never reached EOS: the two must be identical.
        streams = [self.full(0, 'stop'), self.full(1), self.full(2), dict(text=TEXTS[3] + 'x', finish_reason='length',
                                                                          completion_tokens=256)]
        result = rtc.lifecycle_pass(self.event(streams, ignore_eos=(False, False, False, True)), self.solo())
        self.assertEqual(result['users'][3]['verdict'], 'DIVERGED')


    def test_an_ignore_eos_the_engine_ignored_is_a_divergence(self):
        """Review finding 5: the engine stopping at EOS anyway left a stream equal to the solo text, which
        read IDENTICAL; it must run past EOS and spend its whole budget."""
        solo = self.solo(finish=('stop', 'length', 'length', 'stop'))
        ignored = [self.full(0, 'stop'), self.full(1), self.full(2), dict(text=TEXTS[3], finish_reason='stop',
                                                                          completion_tokens=256)]
        result = rtc.lifecycle_pass(self.event(ignored, ignore_eos=(False, False, False, True)), solo)
        self.assertEqual((result['users'][3]['verdict'], result['users'][3]['detail']),
                         ('DIVERGED', 'ignore_eos ignored: stopped at the solo EOS'))
        for stream in (dict(text=TEXTS[3] + ' more', finish_reason='stop', completion_tokens=300),    # a later EOS
                       dict(text=TEXTS[3] + ' more', finish_reason='length', completion_tokens=300),  # short of 1024
                       dict(text='X' + TEXTS[3], finish_reason='length', completion_tokens=1024)):    # not an extension
            with self.subTest(stream=stream):
                streams = ignored[:3] + [stream]
                result = rtc.lifecycle_pass(self.event(streams, ignore_eos=(False, False, False, True)), solo)
                self.assertEqual(result['users'][3]['verdict'], 'DIVERGED')

    def test_a_budget_cut_user_must_spend_its_own_budget(self):
        """Review finding 5: max_tokens=1 checked the prefix and finish only, not the one token."""
        solo = self.solo()
        for stream, verdict in ((dict(text=TEXTS[3][:3], finish_reason='length', completion_tokens=1), 'IDENTICAL'),
                                (dict(text=TEXTS[3][:3], finish_reason='length', completion_tokens=2), 'DIVERGED'),
                                (dict(text=TEXTS[3][:3], finish_reason='abort', completion_tokens=1), 'DIVERGED')):
            with self.subTest(stream=stream):
                streams = [self.full(0, 'stop'), self.full(1), self.full(2), stream]
                result = rtc.lifecycle_pass(self.event(streams, budgets=(1024, 1024, 1024, 1)), solo)
                self.assertEqual(result['users'][3]['verdict'], verdict)
        # Where the solo run reached EOS inside the cut, the cut user's answer is the solo answer itself.
        stopped = self.solo(finish=('stop', 'length', 'length', 'stop'))
        streams = [self.full(0, 'stop'), self.full(1), self.full(2), dict(text=TEXTS[3], finish_reason='stop',
                                                                          completion_tokens=256)]
        result = rtc.lifecycle_pass(self.event(streams, budgets=(1024, 1024, 1024, 300)), stopped)
        self.assertEqual(result['users'][3]['verdict'], 'IDENTICAL')


class ConfigurationTests(unittest.TestCase):
    """Review findings 4 and 13: an unset flag and its default are one arithmetic; the extended scope."""

    def test_an_unset_flag_reads_as_its_default(self):
        served = report(configuration=dict(CONFIGURATION, QWEN_FAST_OUTPUT_BUDGET='256'))
        self.assertEqual(rtc.arithmetic_diff(served, report()), {})
        self.assertEqual(rtc.configuration_diff(served, report()), {'QWEN_FAST_OUTPUT_BUDGET': ['256', None]},
                         'still shown, not judged')
        raised = report(configuration=dict(CONFIGURATION, QWEN_FAST_OUTPUT_BUDGET='16384'))
        self.assertEqual(rtc.arithmetic_diff(raised, report()), {'QWEN_FAST_OUTPUT_BUDGET': ['16384', None]})
        self.assertEqual(rtc.EFFECTIVE_DEFAULTS['QWEN_FAST_OUTPUT_BUDGET'],
                         str(__import__('serving_fast_policy')._output_budget(None)))

    def test_two_reports_are_compared_on_the_names_both_recorded(self):
        extended = dict(CONFIGURATION, QWEN35_GDN_STATE_BF16='1', TT_METAL_CACHE='/k1', OMP_NUM_THREADS='8')
        new = report(configuration=extended)
        new['configuration_scope'] = 'qwen-tt'
        self.assertEqual(rtc.configuration_diff(new, report()), {}, 'a QWEN_-only report never recorded the rest')
        other = report(configuration=dict(extended, QWEN35_GDN_STATE_BF16='0', TT_METAL_CACHE='/k2',
                                          OMP_NUM_THREADS='16'))
        other['configuration_scope'] = 'qwen-tt'
        self.assertEqual(rtc.arithmetic_diff(new, other), {'OMP_NUM_THREADS': ['8', '16'],
                                                          'QWEN35_GDN_STATE_BF16': ['1', '0']},
                         'a cache path is not arithmetic; the GDN state dtype and the thread count are')

    def test_a_divergence_reproduces_only_where_it_first_appeared(self):
        """Review finding 13: two divergences at different characters are nondeterminism, not a repeat."""
        solo = report(users=1, sequential=4)
        moved = rtc.exactness_policy(report(texts=diverged(1, at=5)), solo, rerun=(report(texts=diverged(1, at=9)), solo))
        self.assertEqual((moved['verdict'], moved['users'][1]['verdict']), ('UNSTABLE', 'UNSTABLE'))
        self.assertIn('different characters (5, 9)', moved['users'][1]['reason'])
        same = rtc.exactness_policy(report(texts=diverged(1, at=5)), solo, rerun=(report(texts=diverged(1, at=5)), solo))
        self.assertEqual(same['verdict'], 'FAIL')

V235 = Path(__file__).resolve().parent / 'references' / 'c2-serving' / 'v235-real-text-4x131072.json'


class ReferenceTests(unittest.TestCase):
    """The bring-up: a served arm against the tracked v235 texts."""

    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(V235.read_text(encoding='utf-8'))

    def served(self, texts=None, configuration=None, shas=None):
        streams = self.reference['streams']
        texts = texts or [s['text'] for s in streams]
        served = report(texts=texts, shas=shas or [s['prompt_sha256'] for s in streams],
                        completion=[s['completion_tokens'] for s in streams],
                        finish=[s['finish_reason'] for s in streams],
                        configuration=configuration if configuration is not None else dict(
                            self.reference['qwen_configuration'], QWEN_C2_SERVING='1', QWEN_C2_PROFILE='exact'))
        return served

    def test_the_tracked_reference_is_v235s_four_complete_streams(self):
        self.assertEqual(self.reference['source']['run_id'], 36087022223)
        self.assertEqual((self.reference['users'], self.reference['prompt_tokens'], self.reference['max_tokens']),
                         (4, 131072, 256))
        self.assertEqual(len(self.reference['streams']), 4)
        for stream in self.reference['streams']:
            self.assertEqual((stream['finish_reason'], stream['completion_tokens'], stream['prompt_tokens']),
                             ('length', 256, 131072))
            self.assertEqual(stream['text_sha256'], __import__('hashlib').sha256(
                stream['text'].encode('utf-8')).hexdigest())
        self.assertEqual([s['prompt_sha256'] for s in self.reference['streams']],
                         [u['prompt_sha256'] for u in self.reference['real_text']['users']])

    def test_an_identical_bring_up(self):
        result = rtc.reference_verdicts(self.served(), self.reference)
        self.assertEqual(result['verdict'], 'IDENTICAL')
        self.assertEqual([u['verdict'] for u in result['users']], ['IDENTICAL'] * 4)
        self.assertEqual(result['arithmetic_diff'], {}, 'the serving flags are not arithmetic')
        self.assertIn('BRING-UP IDENTICAL', rtc.render_reference(result))

    def test_a_divergent_user_is_named_with_its_character(self):
        texts = [s['text'] for s in self.reference['streams']]
        texts[2] = texts[2][:300] + '#' + texts[2][301:]
        result = rtc.reference_verdicts(self.served(texts=texts), self.reference)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['users'][2]['verdict'], 'DIVERGED')
        self.assertEqual(result['users'][2]['first_divergence'], 300)
        self.assertIn('user 2: DIVERGED at character 300', rtc.render_reference(result))

    def test_another_prompt_or_arithmetic_is_not_comparable(self):
        shas = [s['prompt_sha256'] for s in self.reference['streams']]
        shas[0] = '0' * 64
        result = rtc.reference_verdicts(self.served(shas=shas), self.reference)
        self.assertEqual((result['verdict'], result['users'][0]['verdict']), ('NOT_COMPARABLE', 'NOT_COMPARABLE'))
        texts = [s['text'] for s in self.reference['streams']]
        texts[1] = 'x' + texts[1][1:]
        configuration = dict(self.reference['qwen_configuration'], QWEN_FAST_SDPA_BF8='1')
        result = rtc.reference_verdicts(self.served(texts=texts, configuration=configuration), self.reference)
        self.assertEqual(result['users'][1]['verdict'], 'NOT_COMPARABLE')
        self.assertIn('QWEN_FAST_SDPA_BF8', result['users'][1]['reason'])

    def test_main_against_the_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'served.json')
            path.write_text(json.dumps(self.served()), encoding='utf-8')
            with redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(rtc.main([str(path), '--against-reference', str(V235)]), 0)
            self.assertIn('user 3: IDENTICAL', printed.getvalue())


if __name__ == '__main__':
    unittest.main()
