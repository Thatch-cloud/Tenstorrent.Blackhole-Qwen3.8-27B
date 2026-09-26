"""prefix_judge: the oracle, the cold/hit comparison, the re-run policy and marker resolution, on CPU.

The oracle is held to the design's own worked cases (2.0.1, 2.0.2, 2.0.4; P0b's shared-prefix
measurements): a chain hits at floor2048 of the previous prompt, 2047/2048/2049, the num_tokens-1
cap, the gap capture across one tenant's conversations, an early divergence falling back, the
salt, the kill switch, reset, and the store's LRU. Then the comparison, the re-run policy (two
agreeing cold runs make any hit divergence a FAIL), concurrent hits and their batch control, the
preemption allowance by admissions, a resumed request's rows, restores against grants, captures
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

    def test_a_concurrent_hit_needs_the_batch_control_before_it_fails(self):
        cold, hit = record('c'), record('h', tokens=(1, 9, 3))
        self.assertEqual(pj.concurrent_verdict(cold, record('h'))['verdict'], 'IDENTICAL')
        self.assertEqual(pj.concurrent_verdict(cold, hit)['verdict'], 'RERUN')
        self.assertEqual(pj.concurrent_verdict(cold, hit, record('c2'))['verdict'], 'RERUN', 'no batch control ran')
        batch_same = pj.concurrent_verdict(cold, hit, record('c2'), record('b'))
        self.assertEqual(batch_same['verdict'], 'DIVERGED')
        batch_differs = pj.concurrent_verdict(cold, hit, record('c2'), record('b', tokens=(1, 2, 8)))
        self.assertEqual(batch_differs['verdict'], 'NOT_COMPARABLE')
        self.assertIn('batch shape changes the bytes', batch_differs['reason'])
        self.assertEqual(pj.concurrent_verdict(cold, hit, record('c2', tokens=(5,)), record('b'))['verdict'], 'UNSTABLE')

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
