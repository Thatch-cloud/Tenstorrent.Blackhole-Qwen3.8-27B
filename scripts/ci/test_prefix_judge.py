"""prefix_judge: the oracle, the cold/hit comparison, the re-run policy and marker resolution, on CPU.

The oracle is held to the design's own worked cases (2.0.1, 2.0.2, 2.0.4; P0b's shared-prefix
measurements): a chain hits at floor2048 of the previous prompt, 2047/2048/2049, the num_tokens-1
cap, the gap capture across one tenant's conversations, an early divergence falling back, the
salt, the kill switch, reset, and the store's LRU. Then the comparison, the re-run policy (two
agreeing cold runs make any hit divergence a FAIL), the batching rule for concurrent hits (their
batch-matched control, pair_summary's tolerance, G1 v48's own concurrent pairs replayed from its
records), the preemption allowance by admissions, a resumed request's rows, restores against grants, captures
against plans, the program cache across every hit and the first capture, the digests, and vLLM's
raw hit read from its counters."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prefix_judge as pj  # noqa: E402
import prefix_markers as pm  # noqa: E402


def seq(start, length):
    return list(range(start, start + length))


class AuditAbsenceTests(unittest.TestCase):
    def test_two_audit_rows_without_digests_are_not_an_agreement(self):
        bare = dict(tag='x', markers=dict(audit=dict(kv_range=None, kv_sha=None, slot_sha=None)))
        other = dict(tag='y', markers=dict(audit=dict(kv_range=None, kv_sha=None, slot_sha=None)))
        (severity, text), = pj.audit_problems(bare, other)
        self.assertEqual(severity, 'NOT_EXERCISED')
        self.assertIn('kv_range, kv_sha, slot_sha', text)
        full = dict(kv_range='0:10', kv_sha='aa', slot_sha='bb')
        self.assertEqual(pj.audit_problems(dict(tag='x', markers=dict(audit=full)),
                                              dict(tag='y', markers=dict(audit=dict(full)))), [])


class OracleTests(unittest.TestCase):
    def test_prefix_digests_name_prefixes(self):
        a = pj.prefix_digests(seq(0, 200))
        b = pj.prefix_digests(seq(0, 150) + [7] * 50)
        self.assertEqual(len(a), 3)
        self.assertEqual(a[:2], b[:2])
        self.assertNotEqual(a[2], b[2])

    def test_a_chain_hits_at_the_previous_prompts_boundary(self):
        oracle = pj.Oracle()
        first = seq(0, 5000)
        self.assertEqual(oracle.admit('s', first), dict(h=0, q=0, plan=[4096], published=4096))
        second = first + seq(90000, 3000)
        result = oracle.admit('s', second)
        self.assertEqual((result['h'], result['q'], result['plan']), (4096, 4096, [6144]))
        third = second + seq(95000, 100)
        result = oracle.admit('s', third)
        self.assertEqual((result['h'], result['q'], result['plan']), (6144, 6144, []), 'a tail-only hit plans nothing')

    def test_the_answer_past_the_cap_is_never_served(self):
        """Turn N+1 contains turn N's prompt and answer verbatim, but only floor2048(P_N) was published."""
        oracle = pj.Oracle()
        prompt = seq(0, 3000)
        oracle.admit('s', prompt)
        result = oracle.admit('s', prompt + seq(50000, 900) + seq(60000, 500))
        self.assertEqual((result['h'], result['q']), (2048, 2048))

    def test_the_boundaries_2047_2048_2049(self):
        for length, want in ((2047, 0), (2048, 2048), (2049, 2048)):
            oracle = pj.Oracle()
            first = oracle.admit('s', seq(0, length))
            self.assertEqual(first['plan'], [2048] if length >= 2048 else [], length)
            second = oracle.admit('s', seq(0, length) + seq(70000, 300))
            self.assertEqual(second['q'], want, length)

    def test_a_fully_cached_prompt_is_capped_at_num_tokens_minus_one(self):
        """P0b C3: at an exact multiple of 2048 the hit is P - 64, and Q falls to the newest
        checkpoint at or below it - none here, so a miss that captures the gap at 2048."""
        oracle = pj.Oracle()
        oracle.admit('s', seq(0, 4096))
        again = oracle.admit('s', seq(0, 4096))
        self.assertEqual((again['h'], again['q'], again['plan']), (4032, 0, [2048, 4096]))
        third = oracle.admit('s', seq(0, 4096))
        self.assertEqual((third['h'], third['q']), (4032, 2048), 'the gap capture is the older checkpoint')

    def test_salts_are_tenants_and_no_salt_is_nothing(self):
        oracle = pj.Oracle()
        oracle.admit('a', seq(0, 5000))
        self.assertEqual(oracle.admit('b', seq(0, 5000) + [1])['q'], 0)
        self.assertEqual(oracle.admit(None, seq(0, 5000) + [1]), dict(h=0, q=0, plan=[], published=0))
        self.assertEqual(oracle.admit('a', seq(0, 5000) + [1])['q'], 4096)

    def test_one_tenants_shared_system_block_is_captured_at_the_gap(self):
        """P0b B1: siblings share a ~6k block. The first publishes; the second misses (no checkpoint
        inside the shared part) and captures floor2048(h); the third hits it."""
        oracle = pj.Oracle()
        shared = seq(0, 6100)
        results = [oracle.admit('t', shared + seq(100000 * (index + 1), 3000)) for index in range(3)]
        self.assertEqual([r['q'] for r in results], [0, 0, 4096])
        self.assertEqual(results[1]['plan'], [4096, 8192])

    def test_an_early_divergence_falls_back_to_an_older_checkpoint(self):
        oracle = pj.Oracle()
        turn1 = seq(0, 2600)
        turn2 = turn1 + seq(10000, 2000)
        turn3 = turn2 + seq(20000, 5000)
        for prompt in (turn1, turn2, turn3):
            oracle.admit('c', prompt)
        early = turn1 + [5] + turn2[2601:] + seq(30000, 1000)
        self.assertEqual(oracle.admit('c', early)['q'], 2048)

    def test_the_kill_switch_latches_and_reset_forgets(self):
        oracle = pj.Oracle()
        oracle.admit('s', seq(0, 5000))
        oracle.kill()
        self.assertEqual(oracle.admit('s', seq(0, 6000))['q'], 0)
        self.assertEqual(oracle.admit('s', seq(0, 7000))['q'], 0)
        other = pj.Oracle()
        other.admit('s', seq(0, 5000))
        other.reset_prefix_cache()
        self.assertEqual(other.admit('s', seq(0, 6000))['q'], 0)

    def test_the_store_is_an_lru_touched_by_grants(self):
        oracle = pj.Oracle(capacity=2)
        a = seq(0, 3000)
        oracle.admit('a', a)
        oracle.admit('b', seq(10000, 3000))
        oracle.admit('a', a + seq(500, 10))           # a grant touches a's checkpoint
        oracle.admit('c', seq(20000, 3000))           # pushes b out, not a
        self.assertEqual(oracle.admit('a', a + seq(600, 10))['q'], 2048)
        self.assertEqual(oracle.admit('b', seq(10000, 3000) + [1])['q'], 0)
        self.assertEqual(pj.Oracle(capacity=0).admit('x', seq(0, 5000))['plan'], [4096])

    def test_the_store_size_is_the_registrys(self):
        self.assertEqual(pj.store_entries(8.0), 55)
        self.assertEqual(pj.store_entries(0.5), 3)


def record(tag, tokens=(1, 2, 3), finish='stop', ok=True, prompt='p', role='hit', **extra):
    base = dict(tag=tag, token_ids=list(tokens), finish=finish, ok=ok, prompt_sha=prompt, prompt_tokens=100,
                role=role, content='c', reasoning='r', tool_calls=[], completion_tokens=len(tokens))
    base.update(extra)
    return base




class CompareTests(unittest.TestCase):
    def test_identical_and_diverged(self):
        self.assertEqual(pj.compare(record('c'), record('h'))['verdict'], 'IDENTICAL')
        result = pj.compare(record('c'), record('h', tokens=(1, 9, 3)))
        self.assertEqual((result['verdict'], result['token']), ('DIVERGED', 1))
        self.assertEqual(pj.compare(record('c'), record('h', tokens=(1, 2)))['token'], 2, 'a strict prefix diverges')
        self.assertEqual(pj.compare(record('c'), record('h', finish='length'))['verdict'], 'DIVERGED')

    def test_prompts_and_failures(self):
        self.assertEqual(pj.compare(record('c'), record('h', prompt='q'))['verdict'], 'NOT_COMPARABLE')
        self.assertEqual(pj.compare(record('c', ok=False, error='x'), record('h'))['verdict'], 'ERROR')
        self.assertEqual(pj.compare(record('c'), record('h', aborted='closed'))['verdict'], 'ERROR')
        self.assertEqual(pj.compare(None, record('h'))['verdict'], 'ERROR')

    def test_text_is_compared_when_no_token_ids_came_back(self):
        a, b = record('c', token_ids=None), record('h', token_ids=None)
        self.assertEqual(pj.compare(a, b)['verdict'], 'IDENTICAL')
        b['tool_calls'] = [dict(function=dict(name='read', arguments='{}'))]
        self.assertEqual(pj.compare(a, b)['verdict'], 'DIVERGED')

    def test_the_rerun_policy(self):
        """PLAN 2.2 item 5's flip policy: two agreeing cold runs make any hit divergence a FAIL; the
        second hit (at the same Q) only says whether it reproduces. Cold runs that disagree are
        UNSTABLE."""
        same, other = record('c'), record('h', tokens=(1, 9, 3))
        self.assertEqual(pj.pair_verdict(same, record('h'))['verdict'], 'IDENTICAL')
        self.assertEqual(pj.pair_verdict(same, other)['verdict'], 'RERUN')
        reproduced = pj.pair_verdict(same, other, record('c2'), record('h2', tokens=(1, 9, 3)))
        self.assertEqual(reproduced['verdict'], 'DIVERGED')
        self.assertIn('diverged again at token 1 (the same place)', reproduced['reason'])
        elsewhere = pj.pair_verdict(same, other, record('c2'), record('h2', tokens=(1, 2, 7)))
        self.assertEqual(elsewhere['verdict'], 'DIVERGED', 'a deterministic bug moves when the re-run differs')
        self.assertIn('diverged again at token 2', elsewhere['reason'])
        healed = pj.pair_verdict(same, other, record('c2'), record('h2'))
        self.assertEqual(healed['verdict'], 'DIVERGED', 'one divergence from agreeing colds is inexact')
        self.assertIn('did not reproduce', healed['reason'])
        no_second_hit = pj.pair_verdict(same, other, record('c2'))
        self.assertEqual(no_second_hit['verdict'], 'DIVERGED')
        cold_flip = pj.pair_verdict(same, other, record('c2', tokens=(4, 4, 4)), record('h2'))
        self.assertEqual(cold_flip['verdict'], 'UNSTABLE')
        self.assertIn('not deterministic', cold_flip['reason'])
        failed_hit = pj.pair_verdict(same, other, record('c2'), record('h2', ok=False, error='boom'))
        self.assertEqual(failed_hit['verdict'], 'DIVERGED')
        failed_cold = pj.pair_verdict(same, other, record('c2', ok=False, error='boom'), record('h2'))
        self.assertEqual(failed_cold['verdict'], 'ERROR')

    def test_the_reviewers_two_unstable_cases_are_now_diverged(self):
        """Colds agree, the hit diverges twice at different places, or once: both were UNSTABLE."""
        cold, cold2 = record('c', tokens=(1, 2, 3, 4)), record('c2', tokens=(1, 2, 3, 4))
        hit, hit2 = record('h', tokens=(1, 9, 3, 4)), record('h2', tokens=(1, 2, 9, 4))
        self.assertEqual(pj.pair_verdict(cold, hit, cold2, hit2)['verdict'], 'DIVERGED')
        self.assertEqual(pj.pair_verdict(cold, hit, cold2, record('h2', tokens=(1, 2, 3, 4)))['verdict'], 'DIVERGED')

    def test_the_batching_rule(self):
        """Prefix reuse must not add divergence beyond what batching alone adds (G1 v48): a concurrent
        hit is judged against its batch-matched cold control, not only its solo cold twin."""
        cold, hit = record('c'), record('h', tokens=(1, 9, 3))
        solo = pj.concurrent_verdict(cold, record('h'))
        self.assertEqual((solo['verdict'], solo['basis']), ('IDENTICAL', 'solo'))
        self.assertEqual(pj.concurrent_verdict(cold, hit)['verdict'], 'RERUN', 'no second solo cold run')
        self.assertEqual(pj.concurrent_verdict(cold, hit, record('c2'))['verdict'], 'RERUN', 'no batch control ran')
        matched = pj.concurrent_verdict(cold, hit, record('c2'), record('b', tokens=(1, 9, 3)))
        self.assertEqual((matched['verdict'], matched['basis']), ('IDENTICAL', 'batch'))
        self.assertIn('equals its batch-matched cold control', matched['reason'])
        reuse = pj.concurrent_verdict(cold, hit, record('c2'), record('b'))
        self.assertEqual((reuse['verdict'], reuse['basis']), ('DIVERGED', 'reuse'), 'the control is the solo run')
        self.assertIn('prefix reuse introduced the divergence', reuse['reason'])
        batching = pj.concurrent_verdict(cold, hit, record('c2'), record('b', tokens=(1, 2, 8)))
        self.assertEqual((batching['verdict'], batching['basis']), ('NOT_COMPARABLE', 'batching'))
        self.assertIn('this batch moves the baseline itself', batching['reason'])
        self.assertEqual(pj.concurrent_verdict(cold, hit, record('c2', tokens=(5,)), record('b'))['verdict'], 'UNSTABLE')
        failed = pj.concurrent_verdict(cold, hit, record('c2'), record('b', ok=False, error='HTTP 500'))
        self.assertEqual(failed['verdict'], 'ERROR')
        self.assertIn('batch-matched cold control failed', failed['reason'])
        self.assertEqual(pj.concurrent_verdict(cold, hit, record('c2', ok=False, error='x'), record('b'))['verdict'],
                         'ERROR')
        other = pj.concurrent_verdict(cold, hit, record('c2'), record('b', prompt='q'))
        self.assertEqual((other['verdict'], other['basis']), ('NOT_COMPARABLE', 'prompts'))
        self.assertEqual(pj.concurrent_verdict(cold, record('h', prompt='q'))['basis'], 'prompts')

    def test_settle_excuses_only_a_preempted_request(self):
        index = dict(h=dict(markers=dict(admissions=2)), c=dict(markers=dict(admissions=1)),
                     h1=dict(markers=dict(admissions=1)))
        pair = pj.settle(dict(verdict='DIVERGED', hit='h', cold='c', detail='first differing output token 3'), index)
        self.assertEqual(pair['verdict'], 'NOT_COMPARABLE')
        self.assertIn('h (2 admissions)', pair['detail'])
        pair = pj.settle(dict(verdict='DIVERGED', hit='h1', cold='c', detail='x'), index)
        self.assertEqual(pair['verdict'], 'DIVERGED')
        self.assertEqual(pj.settle(dict(verdict='IDENTICAL', hit='h', cold='c'), index)['verdict'], 'IDENTICAL')
        control = pj.settle(dict(verdict='NOT_COMPARABLE', hit='h1', cold='c', batch='h', detail='batch'), index)
        self.assertEqual(control['verdict'], 'NOT_COMPARABLE')
        self.assertIn('h (2 admissions)', control['detail'], 'a preempted batch control is named')


def pair(case, verdict, kind='sequential', basis=None):
    return dict(case=case, verdict=verdict, kind=kind, basis=basis, cold='c', hit='h')


class PairSummaryTests(unittest.TestCase):
    """pair_summary: the batching rule over an arm's pairs, and the line every arm prints."""

    def test_not_comparable_beside_an_identical_solo_pair_of_its_family_is_tolerated(self):
        pairs = [pair('life-first', 'IDENTICAL'), pair('arrivals', 'IDENTICAL', 'concurrent', 'batch'),
                 pair('arrivals', 'NOT_COMPARABLE', 'concurrent', 'batching'), pair('arrivals:solo', 'IDENTICAL'),
                 pair('tiny', 'NOT_COMPARABLE', 'concurrent', 'preemption'), pair('tiny:solo', 'IDENTICAL')]
        summary = pj.pair_summary(pairs)
        self.assertTrue(summary['tolerated'], summary['untolerated'])
        self.assertEqual((summary['identical'], summary['identical_by_batch'], summary['not_comparable_batching'],
                          summary['not_comparable_preempted'], summary['failed']), (4, 1, 1, 1, 0))
        self.assertEqual(summary['line'], 'pairs: 4 identical (1 by the batch-matched control), 2 '
                                          'not-comparable-by-batching (1 preempted), 0 failed; solo 3 of 3 identical; of 6')
        self.assertEqual(summary['tolerance'], 'not comparable tolerated (2): every solo pair is IDENTICAL and arrivals, '
                                               'tiny each have an IDENTICAL solo pair')
        self.assertEqual(pj.family('arrivals:solo'), 'arrivals')

    def test_what_is_never_tolerated(self):
        cases = dict(
            no_solo_in_family=([pair('life-first', 'IDENTICAL'), pair('arrivals', 'NOT_COMPARABLE', 'concurrent',
                                                                         'batching')],
                               'family arrivals has 1 not comparable and no IDENTICAL solo pair'),
            a_solo_pair_unstable=([pair('arrivals', 'NOT_COMPARABLE', 'concurrent', 'batching'),
                                   pair('arrivals:solo', 'IDENTICAL'), pair('store', 'UNSTABLE')],
                                  '1 of 2 solo pairs are not IDENTICAL'),
            a_solo_pair_preempted=([pair('arrivals', 'NOT_COMPARABLE', 'concurrent', 'batching'),
                                    pair('arrivals:solo', 'IDENTICAL'), pair('chain', 'NOT_COMPARABLE', basis='preemption')],
                                   '1 of 2 solo pairs are not IDENTICAL'),
            prompts=([pair('arrivals', 'NOT_COMPARABLE', 'concurrent', 'prompts'), pair('arrivals:solo', 'IDENTICAL')],
                     'for a reason other than batching (prompts)'))
        for name, (pairs, reason) in cases.items():
            with self.subTest(name=name):
                summary = pj.pair_summary(pairs)
                self.assertFalse(summary['tolerated'])
                self.assertTrue(any(reason in text for text in summary['untolerated']), summary['untolerated'])
                self.assertTrue(summary['tolerance'].startswith('not comparable NOT tolerated'))

    def test_without_not_comparable_there_is_nothing_to_tolerate(self):
        summary = pj.pair_summary([pair('chain', 'IDENTICAL'), pair('chain', 'DIVERGED'), pair('x', 'ERROR'),
                                   pair('y', 'RERUN')])
        self.assertEqual((summary['tolerated'], summary['tolerance']), (False, None))
        self.assertEqual(summary['line'], 'pairs: 1 identical (0 by the batch-matched control), 0 '
                                          'not-comparable-by-batching, 2 failed; solo 1 of 4 identical; of 4, '
                                          '1 without their re-run')

    def test_settle_names_preemption_as_the_basis(self):
        index = dict(h=dict(markers=dict(admissions=2)), c=dict(markers=dict(admissions=1)))
        settled = pj.settle(dict(verdict='DIVERGED', basis='reuse', hit='h', cold='c', detail='x', kind='concurrent',
                                 case='tiny'), index)
        self.assertEqual((settled['verdict'], settled['basis']), ('NOT_COMPARABLE', 'preemption'))
        self.assertEqual(pj.pair_summary([settled, pair('tiny:solo', 'IDENTICAL')])['not_comparable_preempted'], 1)


# G1 v48 (run 36251045616, image g1-c654916, lifecycle plan): every concurrent pair of lifecycle-evict and
# lifecycle-tiny as its records.jsonl logged it - each record's (token sequence, finish, admissions, tag) and
# the group's distinct output token sequences, cut just past the group's last divergence, which keeps every
# pairwise first divergence and every identity the judge reads. v48 judged six of them NOT_COMPARABLE.
V48_CONCURRENT = (
    dict(arm='lifecycle-evict', case='arrivals', conv='life-0', v48='NOT_COMPARABLE',
         records=dict(cold=(0, 'tool_calls', 1, 'pfx-lifecycle-evict-0014-cold'),
                      cold2=(0, 'tool_calls', 1, 'pfx-lifecycle-evict-0022-cold'),
                      hit=(1, 'tool_calls', 1, 'pfx-lifecycle-evict-0009-hit'),
                      batch=(1, 'tool_calls', 1, 'pfx-lifecycle-evict-0028-cold-batch')),
         sequences=(
             '760 2468 314 279 26156 5224 4816 310 381 7132 36412 1892 424 5686 1040 279 2468 369',
             '760 2468 314 279 26156 5224 4816 310 381 7132 36412 1892 424 5686 1040 279 2468 314',
         )),
    dict(arm='lifecycle-evict', case='arrivals', conv='life-1', v48='NOT_COMPARABLE',
         records=dict(cold=(0, 'tool_calls', 1, 'pfx-lifecycle-evict-0016-cold'),
                      cold2=(0, 'tool_calls', 1, 'pfx-lifecycle-evict-0024-cold'),
                      hit=(1, 'tool_calls', 1, 'pfx-lifecycle-evict-0010-hit'),
                      batch=(2, 'tool_calls', 1, 'pfx-lifecycle-evict-0030-cold-batch')),
         sequences=(
             '81404 1892 279 2468 314 1510 4466 471 77 63 369 8755 2086 2144 1056 1092 279 1156 3162 290 13 561 '
             '999 383 12977 4816 310 381 2086 494 1092 279 1156 369 3221 506 13 13428 11 279 2468 369 8755 4965 '
             '220 18 16 4006 20 20 314 264 2086 999 30 2844 579 15804 13 6558 728 1716 1495 13 271 50821 11 279 '
             '2468 4774 279 2614 25 198 71093 198 262 220 18 16 2672 33341 283 498 8217 29933 28 13927 11 51623 '
             '47980 53897 11 2923 28 9239 11 15911 28 28693 11 3817 68672 3497 11 11857 68672 3497 8 198 71093 271 '
             '1919 3070 914 2353 279 1156 579 24147 13 85152 11 3655 1892 6970 279 2468 2597 57739 506 279 6941 '
             '318 1719 1118 220 18 15 4965 998 3799 974 45690 561',
             '81404 1892 279 2468 314 1510 4466 471 77 63 369 8755 2086 2144 1056 1092 353 3481 13 1049 4816 1040 '
             '279 999 2144 369 1602 12234 67247 13 13428 11 279 2468 4774 4965 220 18 16 4006 20 20 11 321 279 '
             '2144 369 883 1510 265 1315 7561 1510 1307 1217 1098 7561 1510 1689 1386 63 1076 1061 3070 914 2353 '
             '279 1510 1200 5437 537 63 709 353 5312 6575 13 271 77264 11 3655 13 32645 11 279 2468 369 15804 13 '
             '561 1118 1510 851 63 1562 8280 4965 220 22 4006 19 16 8222 279 1510 1200 5437 537 63 709 13 4543 '
             '1510 4466 471 77 63 369 8755 4965 220 18 16 4006 20 20 440 2086 2144 13 1061 369 1546 14457 13 271 '
             '13784 11 6970 279 2468 369 1602 57739 466',
             '81404 1892 279 2468 314 1510 4466 471 77 63 369 8755 2086 2144 1056 1092 353 3481 13 1049 4816 1040 '
             '279 999 2144 369 1602 12234 67247 13 13428 11 279 2468 4774 4965 220 18 16 4006 20 20 11 321 279 '
             '2144 369 883 1510 265 1315 7561 1510 1307 1217 1098 7561 1510 1689 1386 63 1076 1061 3070 914 2353 '
             '279 1510 1200 5437 537 63 709 353 5312 6575 13 271 77264 11 3655 13 32645 11 279 2468 369 15804 13 '
             '561 1118 1510 851 63 1562 8280 4965 220 22 4006 19 16 8222 279 1510 1200 5437 537 63 709 13 4543 '
             '1510 4466 471 77 63 369 8755 4965 220 18 16 4006 20 20 440 2086 2144 13 1061 369 1546 14457 13 271 '
             '13784 11 6970 279 2468 369 1602 57739 303',
         )),
    dict(arm='lifecycle-evict', case='arrivals', conv='same-step-0', v48='IDENTICAL',
         records=dict(cold=(0, 'tool_calls', 1, 'pfx-lifecycle-evict-0018-cold'),
                      hit=(0, 'tool_calls', 1, 'pfx-lifecycle-evict-0011-hit')),
         sequences=(
             '9764',
         )),
    dict(arm='lifecycle-evict', case='arrivals', conv='same-step-1', v48='NOT_COMPARABLE',
         records=dict(cold=(0, 'tool_calls', 1, 'pfx-lifecycle-evict-0020-cold'),
                      cold2=(0, 'tool_calls', 1, 'pfx-lifecycle-evict-0026-cold'),
                      hit=(1, 'tool_calls', 1, 'pfx-lifecycle-evict-0012-hit'),
                      batch=(1, 'tool_calls', 1, 'pfx-lifecycle-evict-0034-cold-batch')),
         sequences=(
             '760 1156 369 9859 728 310 1301 1510 19226 2805 72 3082 765 33717 9546 1323 6971 7561 10033 1092 1510 '
             '8987 63 1503 321 1332 424 579 2512 494 11 321 1179 28647 264',
             '760 1156 369 9859 728 310 1301 1510 19226 2805 72 3082 765 33717 9546 1323 6971 7561 10033 1092 1510 '
             '8987 63 1503 321 1332 424 579 2512 494 11 321 1179 28647 799',
         )),
    dict(arm='lifecycle-tiny', case='tiny-grant', conv='tiny-grant', v48='IDENTICAL',
         records=dict(cold=(0, 'tool_calls', 1, 'pfx-lifecycle-tiny-0006-cold'),
                      hit=(0, 'tool_calls', 1, 'pfx-lifecycle-tiny-0004-hit')),
         sequences=(
             '760',
         )),
    dict(arm='lifecycle-tiny', case='tiny', conv='tiny-0', v48='NOT_COMPARABLE',
         records=dict(cold=(0, 'tool_calls', 1, 'pfx-lifecycle-tiny-0014-cold'),
                      cold2=(0, 'tool_calls', 1, 'pfx-lifecycle-tiny-0020-cold'),
                      hit=(1, 'tool_calls', 2, 'pfx-lifecycle-tiny-0010-hit'),
                      batch=(1, 'tool_calls', 2, 'pfx-lifecycle-tiny-0026-cold-batch')),
         sequences=(
             '760 1156 369 9859 728 310 1301 1510 19226 2805 72 3082 765 33717 9546 1323 6971 63 321 10033 1092 '
             '1510 11900 63 1503 321 1332 424 579 2512 494 11 321 1179 310 28647 264 4821 1228 364 424 13 271 760 '
             '3555',
             '760 1156 369 9859 728 310 1301 1510 19226 2805 72 3082 765 33717 9546 1323 6971 63 321 10033 1092 '
             '1510 11900 63 1503 321 1332 424 579 2512 494 11 321 1179 310 28647 264 4821 1228 364 424 13 271 760 '
             '2193',
         )),
    dict(arm='lifecycle-tiny', case='tiny', conv='tiny-1', v48='NOT_COMPARABLE',
         records=dict(cold=(0, 'tool_calls', 1, 'pfx-lifecycle-tiny-0016-cold'),
                      cold2=(0, 'tool_calls', 1, 'pfx-lifecycle-tiny-0022-cold'),
                      hit=(1, 'tool_calls', 1, 'pfx-lifecycle-tiny-0011-hit'),
                      batch=(1, 'tool_calls', 1, 'pfx-lifecycle-tiny-0028-cold-batch')),
         sequences=(
             '760 1156 369 9859 728 310 10033 821 3019 303 2250 16916 3387 1518 3154 866 4203 13 561 1156 682 1048 '
             '3766 1010 5604 1970 66320 11 694 279 4880 3274 369 310 89109 1510 1877 1611 306 5830 85763 27625 '
             '12678 82 33030 18 17 25018 25092 7663 62 4111 7402 8685 63 303 1510 19226 2805 72 12333 788 51976 '
             '85023 6971 27653 271 5170 11 1042 728 1301 279 2100 999 310 3418 1092 353 2688 14131 440 13 198 '
             '248069 271 248058 198 27 1628 86779 29 198 27 15704 57242 2551 29 198 17674 33901 33902 16240 268 '
             '45787 7725 27887 29430 27325 16451 18 13 23 12 17 22 33 38068 2805 72 12333 788 51976 85023 6971 198 '
             '510 15704 29 198 510 1628 29 198 248059 248046 198 248045 846 248046 198 248045 248046 198 248045 '
             '74455 198 248068 198 760 999 369 4147 30 6558 728 1716 13 198 248069 271 248058 198 27 1628 21402 '
             '956 29 198 27 15704 28 5454 29 198 4577 471 4120 593 4955 33901 33902 16240 268 45787 7725 27887 '
             '29430 27325 16451 18 13 23 12 17 22 33 38068 2805 72 12333 788 51976 85023 6971 976 25700 471 75 593 '
             '4955 33901 33902 16240 268 45787 7725 27887 29430 27325 16451 18 13 23 12 17 22 33 38068 2805 72 '
             '12333 788 51976 85023 6971 198 510 15704 29 198 27 15704 28 4532 29 198 3840 2100 999 1331 321',
             '760 1156 369 9859 728 310 10033 821 3019 303 2250 16916 3387 1518 3154 866 4203 13 561 1156 682 1048 '
             '3766 1010 5604 1970 66320 11 694 279 4880 3274 369 310 89109 1510 1877 1611 306 5830 85763 27625 '
             '12678 82 33030 18 17 25018 25092 7663 62 4111 7402 8685 63 303 1510 19226 2805 72 12333 788 51976 '
             '85023 6971 27653 271 5170 11 1042 728 1301 279 2100 999 310 3418 1092 353 2688 14131 440 13 198 '
             '248069 271 248058 198 27 1628 86779 29 198 27 15704 57242 2551 29 198 17674 33901 33902 16240 268 '
             '45787 7725 27887 29430 27325 16451 18 13 23 12 17 22 33 38068 2805 72 12333 788 51976 85023 6971 198 '
             '510 15704 29 198 510 1628 29 198 248059 248046 198 248045 846 248046 198 248045 248046 198 248045 '
             '74455 198 248068 198 760 999 369 4147 30 6558 728 1716 13 198 248069 271 248058 198 27 1628 21402 '
             '956 29 198 27 15704 28 5454 29 198 4577 471 4120 593 4955 33901 33902 16240 268 45787 7725 27887 '
             '29430 27325 16451 18 13 23 12 17 22 33 38068 2805 72 12333 788 51976 85023 6971 976 25700 471 75 593 '
             '4955 33901 33902 16240 268 45787 7725 27887 29430 27325 16451 18 13 23 12 17 22 33 38068 2805 72 '
             '12333 788 51976 85023 6971 198 510 15704 29 198 27 15704 28 4532 29 198 3840 2100 999 1331 198',
         )),
    dict(arm='lifecycle-tiny', case='tiny', conv='tiny-2', v48='NOT_COMPARABLE',
         records=dict(cold=(0, 'tool_calls', 1, 'pfx-lifecycle-tiny-0018-cold'),
                      cold2=(0, 'tool_calls', 1, 'pfx-lifecycle-tiny-0024-cold'),
                      hit=(1, 'tool_calls', 1, 'pfx-lifecycle-tiny-0012-hit'),
                      batch=(1, 'tool_calls', 1, 'pfx-lifecycle-tiny-0030-cold-batch')),
         sequences=(
             '760 1156 369 9859 728 310 10033 821',
             '760 1156 369 9859 728 310 10033 279',
         )),
)


def v48_records(group):
    out = {}
    for role, (index, finish, admissions, tag) in group['records'].items():
        tokens = [int(token) for token in group['sequences'][index].split()]
        out[role] = dict(tag=tag, token_ids=tokens, finish=finish, ok=True, prompt_sha='v48-%s' % group['conv'],
                         prompt_tokens=1000, markers=dict(admissions=admissions))
    return out


class V48ReplayTests(unittest.TestCase):
    """The batching rule on G1 v48's own concurrent pairs, from its records.jsonl."""

    # (arm, conv) -> (verdict, basis) under the batching rule; v48's verdict is in the fixture.
    EXPECTED = {('lifecycle-evict', 'life-0'): ('IDENTICAL', 'batch'),
                ('lifecycle-evict', 'life-1'): ('NOT_COMPARABLE', 'batching'),
                ('lifecycle-evict', 'same-step-0'): ('IDENTICAL', 'solo'),
                ('lifecycle-evict', 'same-step-1'): ('IDENTICAL', 'batch'),
                ('lifecycle-tiny', 'tiny-grant'): ('IDENTICAL', 'solo'),
                ('lifecycle-tiny', 'tiny-0'): ('IDENTICAL', 'batch'),
                ('lifecycle-tiny', 'tiny-1'): ('IDENTICAL', 'batch'),
                ('lifecycle-tiny', 'tiny-2'): ('IDENTICAL', 'batch')}

    def verdicts(self):
        out = {}
        for group in V48_CONCURRENT:
            records = v48_records(group)
            result = pj.concurrent_verdict(records['cold'], records['hit'], records.get('cold2'), records.get('batch'))
            out[(group['arm'], group['conv'])] = (group, records, result)
        return out

    def test_each_v48_concurrent_pair_under_the_batching_rule(self):
        verdicts = self.verdicts()
        self.assertEqual(set(verdicts), set(self.EXPECTED))
        for key, (group, records, result) in verdicts.items():
            with self.subTest(pair=key):
                self.assertEqual((result['verdict'], result.get('basis')), self.EXPECTED[key], result.get('reason'))
        self.assertEqual(sum(1 for group, _, _ in verdicts.values() if group['v48'] == 'NOT_COMPARABLE'), 6)
        # life-1: the control moved from the solo run at token 16, and from the hit at 144.
        _, _, life1 = verdicts[('lifecycle-evict', 'life-1')]
        self.assertEqual((life1['batch']['token'], life1['batch_hit']['token'], life1['first']['token']), (16, 144, 16))
        # tiny-0: the hit and its control were both preempted and resumed, and still match.
        _, records, tiny0 = verdicts[('lifecycle-tiny', 'tiny-0')]
        self.assertEqual((records['hit']['markers']['admissions'], records['batch']['markers']['admissions']), (2, 2))
        self.assertEqual(pj.settle(dict(tiny0, hit='h', cold='c'), {})['verdict'], 'IDENTICAL')

    def test_the_v48_arms_pass_the_rule_with_a_solo_anchor_and_not_without(self):
        """lifecycle-tiny's four concurrent pairs are all IDENTICAL now; lifecycle-evict keeps life-1 not
        comparable, tolerated only beside an IDENTICAL solo pair of the arrivals family (the anchor
        prefix_replay now runs) and with every solo pair IDENTICAL (v48's kill-switch pairs were ERRORs:
        HTTP 400 at the API edge, the sizing fault fixed apart)."""
        verdicts = self.verdicts()
        arms = {}
        for (arm, conv), (group, _, result) in verdicts.items():
            arms.setdefault(arm, []).append(dict(case=group['case'], conv=conv, kind='concurrent', cold='c', hit='h',
                                                 verdict=result['verdict'], basis=result.get('basis')))
        tiny = pj.pair_summary(arms['lifecycle-tiny'])
        self.assertEqual((tiny['identical'], tiny['identical_by_batch'], tiny['tolerance']), (4, 3, None))
        solo = [pair(case, 'IDENTICAL') for case in ('life-first', 'life-first', 'abort-waiting-retry',
                                                      'abort-prefill-retry', 'after-flood', 'after-reset', 'kill-before',
                                                      'before-reload', 'after-reload', 'after-reload-2')]
        evict = arms['lifecycle-evict'] + solo
        self.assertFalse(pj.pair_summary(evict)['tolerated'], 'no IDENTICAL solo pair in the arrivals family')
        anchored = pj.pair_summary(evict + [pair('arrivals:solo', 'IDENTICAL')])
        self.assertTrue(anchored['tolerated'], anchored['untolerated'])
        self.assertIn('1 not-comparable-by-batching, 0 failed', anchored['line'])
        v48_kill = pj.pair_summary(evict + [pair('arrivals:solo', 'IDENTICAL'), pair('kill-on', 'ERROR'),
                                            pair('kill-latched', 'ERROR')])
        self.assertFalse(v48_kill['tolerated'])
        self.assertIn('2 failed', v48_kill['line'])


class ResolveTests(unittest.TestCase):
    def scanned(self):
        return dict(grants=[dict(tag='t-hit', h=4160, q=4096, plan=[8192], index=3)],
                    rows=[dict(tag='t-hit', q=4096, l=9000, index=4, programs=10),
                          dict(tag=None, q=0, l=500, index=12, programs=10),
                          dict(tag=None, q=0, l=700, index=30), dict(tag=None, q=0, l=700, index=31)],
                    audits=[dict(tag='t-hit', kv_range='0:9000', kv_sha='a', slot_sha='b')],
                    capture_skipped=[dict(tag='t-hit', pos='8192', reason='MemoryError()')])

    def test_by_tag_then_by_window_and_length(self):
        records = [dict(tag='t-hit', prompt_tokens=9000, log_window=[0, 5]),
                   dict(tag='t-cold', prompt_tokens=500, log_window=[10, 20]),
                   dict(tag='t-two', prompt_tokens=700, log_window=[25, 40]),
                   dict(tag='t-none', prompt_tokens=800, log_window=[41, 50])]
        pj.resolve(records, self.scanned())
        hit, cold, two, none = [r['markers'] for r in records]
        self.assertEqual((hit['q'], hit['grant']['h'], hit['audit']['kv_sha'], hit['matched']), (4096, 4160, 'a', 'tag'))
        self.assertEqual((hit['admissions'], len(hit['skipped'])), (1, 1))
        self.assertEqual((cold['q'], cold['matched']), (0, 'window'))
        self.assertIsNone(two['q'])
        self.assertIn('ambiguous', two['matched'])
        self.assertEqual((none['rows'], none['grant'], none['admissions']), ([], None, 0))

    def test_a_preempted_request_is_judged_on_its_first_admission(self):
        """The reviewer's case: a resumed request re-prefills prompt + output, so its second row's L
        is larger; the first row is the admission, the count names the preemption."""
        lines = [
            '[PINDIAG] prefix: grant req=chatcmpl-pfx-lifecycle-tiny-0007-hit-1a2b3c4d h=24576 Q=24576 plan=[]',
            '[PREFIX] req=chatcmpl-pfx-lifecycle-tiny-0007-hit-1a2b3c4d Q=24576 L=26700 path=traced restored_ms=1 '
            'captured=[] capture_ms=0 programs=5',
            '[PINDIAG] prefix: grant req=chatcmpl-pfx-lifecycle-tiny-0007-hit-1a2b3c4d h=24576 Q=24576 plan=[]',
            '[PREFIX] req=chatcmpl-pfx-lifecycle-tiny-0007-hit-1a2b3c4d Q=24576 L=27900 path=traced restored_ms=1 '
            'captured=[] capture_ms=0 programs=5',
        ]
        rec = dict(tag='pfx-lifecycle-tiny-0007-hit', ok=True, role='hit', prompt_tokens=26700, completion_tokens=2048,
                   expected=dict(h=24576, q=24576, plan=[]), log_window=None)
        pj.resolve([rec], pm.scan(lines))
        self.assertEqual((rec['markers']['admissions'], rec['markers']['q'], rec['markers']['l']), (2, 24576, 26700))
        self.assertEqual(rec['markers']['grant']['index'], 0, 'the first admission\'s grant')
        problems = pj.reuse_problems(rec, sequential=False)
        self.assertEqual([severity for severity, _ in problems], ['NOTE'])
        self.assertIn('preempted and re-admitted at L=27900', problems[0][1])
        rec['completion_tokens'] = 100
        self.assertIn('FAIL', [severity for severity, _ in pj.reuse_problems(rec)], 'L past prompt + output')

    def test_a_readmission_with_no_grant_line_keeps_the_first_grant_apart(self):
        scanned = dict(grants=[dict(tag='t', h=4096, q=4096, plan=[], index=1)],
                       rows=[dict(tag='t', q=4096, l=5000, index=2), dict(tag='t', q=0, l=5100, index=9)])
        rec = dict(tag='t', ok=True, prompt_tokens=5000, completion_tokens=200, role='hit')
        pj.resolve([rec], scanned)
        self.assertEqual(rec['markers']['grant']['q'], 4096)
        self.assertEqual([s for s, _ in pj.reuse_problems(rec)], ['NOTE'])


def resolved(tag, role='hit', q=None, l=10000, grant=None, expected=None, ok=True, rows=True, prompt_tokens=10000,
             captured=(), skipped=()):
    row = dict(q=q, l=l, tag=tag, captured=list(captured))
    markers = dict(rows=[row] if rows else [], row=row if rows else None, q=q if rows else None, l=l if rows else None,
                   grant=grant, grants=[grant] if grant else [], skipped=list(skipped), matched='tag',
                   admissions=1 if rows else 0)
    return dict(tag=tag, role=role, ok=ok, markers=markers, expected=expected, prompt_tokens=prompt_tokens)


class ReuseTests(unittest.TestCase):
    def severities(self, record, sequential=True):
        return [severity for severity, _ in pj.reuse_problems(record, sequential)]

    def test_a_hit_the_oracle_expected(self):
        self.assertEqual(self.severities(resolved('h', q=4096, grant=dict(h=4160, q=4096, plan=[]),
                                                  expected=dict(h=4160, q=4096, plan=[]))), [])

    def test_missing_and_inconsistent_rows(self):
        self.assertEqual(self.severities(resolved('h', rows=False)), ['FAIL'])
        self.assertEqual(self.severities(resolved('h', q=0, l=99)), ['FAIL'])
        self.assertEqual(self.severities(resolved('h', q=2048, grant=dict(h=4160, q=4096),
                                                  expected=dict(h=4160, q=4096))), ['FAIL', 'LOST'])
        self.assertEqual(self.severities(resolved('h', q=1000, expected=dict(h=4160, q=4096))), ['FAIL', 'FAIL', 'LOST'])
        self.assertEqual(self.severities(resolved('h', ok=False, rows=False)), [], 'a failed request is judged elsewhere')

    def test_a_restore_without_a_grant_fails(self):
        texts = [t for _, t in pj.reuse_problems(resolved('h', q=2048, expected=dict(h=2048, q=2048)))]
        self.assertTrue(any('restore without a grant' in t for t in texts), texts)

    def test_captures_against_the_plan(self):
        grant = dict(h=0, q=0, plan=[4096])
        self.assertEqual(self.severities(resolved('c', role='cold', q=0, grant=grant, captured=[4096],
                                                  expected=dict(h=0, q=0, plan=[4096]))), [])
        self.assertEqual(self.severities(resolved('c', role='cold', q=0, grant=grant, captured=[],
                                                  expected=dict(h=0, q=0, plan=[4096]))), ['FAIL'])
        self.assertEqual(self.severities(resolved('c', role='cold', q=0, grant=grant, captured=[],
                                                  skipped=[dict(reason='MemoryError()')],
                                                  expected=dict(h=0, q=0, plan=[4096]))), ['NOTE'])
        self.assertEqual(self.severities(resolved('u', role='unsalted', q=0, captured=[2048])), ['FAIL'],
                         'a capture nobody planned')
        wrong_plan = resolved('c', role='cold', q=0, grant=dict(h=0, q=0, plan=[2048]), captured=[2048],
                              expected=dict(h=0, q=0, plan=[4096]))
        self.assertEqual(self.severities(wrong_plan), ['FAIL'])
        self.assertEqual(self.severities(wrong_plan, sequential=False), ['NOTE'])

    def test_tenancy(self):
        self.assertEqual(self.severities(resolved('u', role='unsalted', q=0, grant=dict(h=0, q=0))), ['FAIL'])
        self.assertEqual(self.severities(resolved('u', role='unsalted', q=2048)), ['FAIL', 'FAIL'])
        self.assertEqual(self.severities(resolved('u', role='unsalted', q=0)), [])
        for role in ('cold', 'capture', 'cold-batch'):
            self.assertIn('FAIL', self.severities(resolved('c', role=role, q=2048, grant=dict(h=2048, q=2048),
                                                           expected=dict(h=0, q=0))), role)

    def test_above_and_below_the_oracle(self):
        above = resolved('h', q=8192, grant=dict(h=8256, q=8192), expected=dict(h=4160, q=4096))
        self.assertEqual(self.severities(above), ['FAIL', 'FAIL'])
        self.assertEqual(self.severities(above, sequential=False), ['NOTE', 'NOTE'])
        below = resolved('h', q=0, grant=dict(h=0, q=0), expected=dict(h=4160, q=4096))
        self.assertEqual(self.severities(below), ['LOST', 'LOST'])


def row(index, tag, q=0, l=100, programs=10, captured=(), before=None):
    return dict(index=index, tag=tag, q=q, l=l, programs=programs, captured=list(captured), programs_before=before)


class ProgramCacheTests(unittest.TestCase):
    def test_every_hit_after_its_cold_twin_is_measured(self):
        rows = [row(0, 'c1', l=5000, programs=10), row(1, 'h1', q=4096, l=5000, programs=10),
                row(2, 'c2', l=9000, programs=12), row(3, 'h2', q=8192, l=9000, programs=13)]
        pairs = [dict(cold='c1', hit='h1'), dict(cold='c2', hit='h2')]
        problems, detail, missing = pj.program_cache_problems(rows, pairs, first_capture=False)
        self.assertEqual(detail['hits_measured'], 2)
        self.assertEqual(len(problems), 1)
        self.assertIn('the hit h2', problems[0], 'the second hit, not only the first')

    def test_without_programs_before_only_an_adjacent_hit_is_measured(self):
        rows = [row(0, 'c1', l=5000), row(1, 'x', l=7000, programs=11), row(2, 'h1', q=4096, l=5000, programs=11)]
        problems, detail, _ = pj.program_cache_problems(rows, [dict(cold='c1', hit='h1')], first_capture=False)
        self.assertEqual(detail['hits_measured'], 0)
        self.assertIn('not measured', problems[0])
        self.assertEqual(pj.program_cache_problems(rows, [dict(cold='c1', hit='h1')], first_capture=False,
                                                   require_hit=False)[0], [])

    def test_a_rows_own_count_ignores_what_decode_compiled_between_rows(self):
        """The cold twin's decode steps may compile between its row and the hit's: with
        programs_before the hit is judged on what its own row compiled."""
        rows = [row(0, 'c1', l=5000, programs=10, before=9), row(1, 'h1', q=4096, l=5000, programs=12, before=12)]
        problems, detail, _ = pj.program_cache_problems(rows, [dict(cold='c1', hit='h1')], first_capture=False)
        self.assertEqual((problems, detail['hits_measured']), ([], 1))
        rows[1]['programs'] = 13
        problems, _, _ = pj.program_cache_problems(rows, [dict(cold='c1', hit='h1')], first_capture=False)
        self.assertIn('compiled 1 programs (the row itself)', problems[0])
        rows = [row(0, 'c1', l=5000, before=9), row(1, 'x', l=7000, before=10), row(2, 'h1', q=4096, l=5000, before=10)]
        self.assertEqual(pj.program_cache_problems(rows, [dict(cold='c1', hit='h1')], first_capture=False)[1]
                         ['hits_measured'], 1, 'not adjacent, still measured')

    def test_the_first_capture_against_the_same_prompt_uncaptured(self):
        rows = [row(0, 'u1', l=2600, programs=10), row(1, 'k1', l=2600, programs=10, captured=[2048]),
                row(2, 'c', l=5000, programs=10, captured=[4096]), row(3, 'h', q=2048, l=5000, programs=10,
                                                                        captured=[4096])]
        problems, detail, missing = pj.program_cache_problems(rows, [dict(cold='c', hit='h')])
        self.assertEqual((problems, missing, detail['capture_compiled']), ([], [], 0))
        rows[1]['programs'] = 11
        problems, _, _ = pj.program_cache_problems(rows, [dict(cold='c', hit='h')])
        self.assertTrue(any('the first capture (k1) compiled 1' in p for p in problems), problems)
        rows[0]['l'] = 2500
        _, _, missing = pj.program_cache_problems(rows, [dict(cold='c', hit='h')])
        self.assertIn('not measured', missing[0])
        self.assertIn('no [PREFIX] row captured', pj.program_cache_problems([row(0, 'a')], [])[2][0])

    def test_a_missing_programs_field(self):
        rows = [row(0, 'c1', programs=None), row(1, 'h1', q=4096, programs=10)]
        problems, _, _ = pj.program_cache_problems(rows, [dict(cold='c1', hit='h1')], first_capture=False)
        self.assertIn('not measured', problems[0])


class DigestTests(unittest.TestCase):
    def pair(self, a, b):
        return (dict(tag='c', markers=dict(row=a, audit=a)), dict(tag='h', markers=dict(row=b, audit=b)))

    def test_slot_and_logits_digests(self):
        same = dict(slot_sha='s', logits_sha='l')
        self.assertEqual(pj.digest_problems(*self.pair(same, dict(same))), [])
        self.assertEqual([s for s, _ in pj.digest_problems(*self.pair(same, dict(same, slot_sha='x')))], ['FAIL'])
        self.assertEqual([s for s, _ in pj.digest_problems(*self.pair(same, dict(same, logits_sha='x')))], ['FAIL'])
        self.assertEqual([s for s, _ in pj.digest_problems(*self.pair(same, {}))], ['NOT_EXERCISED'] * 2)

    def test_audit_digests(self):
        same = dict(kv_range='0:900', kv_sha='k', slot_sha='s')
        self.assertEqual(pj.audit_problems(*self.pair(same, dict(same))), [])
        self.assertEqual([s for s, _ in pj.audit_problems(*self.pair(same, dict(same, kv_sha='x')))], ['FAIL'])
        self.assertEqual([s for s, _ in pj.audit_problems(*self.pair(same, dict(same, slot_sha='x')))], ['FAIL'])
        self.assertEqual([s for s, _ in pj.audit_problems(*self.pair(same, dict(same, kv_range='400:900')))],
                         ['NOT_EXERCISED'])
        self.assertEqual([s for s, _ in pj.audit_problems(*self.pair(same, None))], ['NOT_EXERCISED'])


class CounterTests(unittest.TestCase):
    def counted(self, hits, queries, length=1000):
        return dict(prompt_tokens=length, counters=dict(
            before={'vllm:prefix_cache_hits': 100.0, 'vllm:prefix_cache_queries': 5000.0},
            after={'vllm:prefix_cache_hits': 100.0 + hits, 'vllm:prefix_cache_queries': 5000.0 + queries}))

    def test_the_raw_hit_per_admission_attempt(self):
        self.assertEqual(pj.raw_hit_per_attempt(self.counted(0, 1000)), (0.0, 1))
        self.assertEqual(pj.raw_hit_per_attempt(self.counted(4096, 2000)), (2048.0, 2), 'counted once per attempt')
        self.assertEqual(pj.raw_hit_per_attempt(dict(prompt_tokens=5)), (None, None))
        self.assertIsNone(pj.counter_delta(dict(counters=dict(before={}, after={})), 'vllm:prefix_cache_hits'))

    def test_what_a_salt_has_published(self):
        oracle = pj.Oracle()
        prompt = seq(0, 5000)
        self.assertEqual(oracle.published_tokens('s', prompt), 0)
        oracle.admit('s', prompt)
        self.assertEqual(oracle.published_tokens('s', prompt + seq(90000, 10)), 4096)
        self.assertEqual(oracle.published_tokens('other', prompt), 0)
        self.assertEqual(oracle.published_tokens(None, prompt), 0)
        oracle.kill()
        oracle.admit('s', prompt + seq(90000, 5000))
        self.assertEqual(oracle.published_tokens('s', prompt + seq(90000, 5000)), 4096, 'nothing under the kill switch')


class WorstTests(unittest.TestCase):
    def test_order(self):
        self.assertEqual(pj.worst(['PASS', 'UNSTABLE', 'NOT_EXERCISED']), 'UNSTABLE')
        self.assertEqual(pj.worst(['PASS', 'INFRA', 'FAIL']), 'FAIL')
        self.assertEqual(pj.worst(['PASS']), 'PASS')
        self.assertEqual(pj.worst([]), 'FAIL')


if __name__ == '__main__':
    unittest.main()
