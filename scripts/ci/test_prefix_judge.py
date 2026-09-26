"""prefix_judge: the oracle, the cold/hit comparison, the re-run policy and marker resolution, on CPU.

The oracle is held to the design's own worked cases (2.0.1, 2.0.2, 2.0.4; P0b's shared-prefix
measurements): a chain hits at floor2048 of the previous prompt, 2047/2048/2049, the num_tokens-1
cap, the gap capture across one tenant's conversations, an early divergence falling back, the
salt, the kill switch, reset, and the store's LRU."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prefix_judge as pj  # noqa: E402


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
        same, other = record('c'), record('h', tokens=(1, 9, 3))
        self.assertEqual(pj.pair_verdict(same, record('h'))['verdict'], 'IDENTICAL')
        self.assertEqual(pj.pair_verdict(same, other)['verdict'], 'RERUN')
        reproduced = pj.pair_verdict(same, other, record('c2'), record('h2', tokens=(1, 9, 3)))
        self.assertEqual(reproduced['verdict'], 'DIVERGED')
        same_place = pj.pair_verdict(same, other, record('c2'), record('h2', tokens=(1, 8, 3)))
        self.assertEqual(same_place['verdict'], 'DIVERGED', 'the same first differing token')
        elsewhere = pj.pair_verdict(same, other, record('c2'), record('h2', tokens=(1, 2, 7)))
        self.assertEqual(elsewhere['verdict'], 'UNSTABLE')
        healed = pj.pair_verdict(same, other, record('c2'), record('h2'))
        self.assertEqual(healed['verdict'], 'UNSTABLE')
        cold_flip = pj.pair_verdict(same, other, record('c2', tokens=(4, 4, 4)), record('h2'))
        self.assertEqual(cold_flip['verdict'], 'UNSTABLE')
        self.assertIn('not deterministic', cold_flip['reason'])
        failed = pj.pair_verdict(same, other, record('c2'), record('h2', ok=False, error='boom'))
        self.assertEqual(failed['verdict'], 'ERROR')


class ResolveTests(unittest.TestCase):
    def scanned(self):
        return dict(grants=[dict(tag='t-hit', h=4160, q=4096, plan=[8192], index=3)],
                    rows=[dict(tag='t-hit', q=4096, l=9000, index=4, programs=10),
                          dict(tag=None, q=0, l=500, index=12, programs=10),
                          dict(tag=None, q=0, l=700, index=30), dict(tag=None, q=0, l=700, index=31)],
                    audits=[dict(tag='t-hit', kv_range='0:9000', kv_sha='a', slot_sha='b')])

    def test_by_tag_then_by_window_and_length(self):
        records = [dict(tag='t-hit', prompt_tokens=9000, log_window=[0, 5]),
                   dict(tag='t-cold', prompt_tokens=500, log_window=[10, 20]),
                   dict(tag='t-two', prompt_tokens=700, log_window=[25, 40]),
                   dict(tag='t-none', prompt_tokens=800, log_window=[41, 50])]
        pj.resolve(records, self.scanned())
        hit, cold, two, none = [r['markers'] for r in records]
        self.assertEqual((hit['q'], hit['grant']['h'], hit['audit']['kv_sha'], hit['matched']), (4096, 4160, 'a', 'tag'))
        self.assertEqual((cold['q'], cold['matched']), (0, 'window'))
        self.assertIsNone(two['q'])
        self.assertIn('ambiguous', two['matched'])
        self.assertEqual((none['rows'], none['grant']), ([], None))


def resolved(tag, role='hit', q=None, l=100, grant=None, expected=None, ok=True, rows=True, prompt_tokens=100):
    markers = dict(rows=[dict(q=q, l=l, tag=tag)] if rows else [], q=q if rows else None, l=l if rows else None,
                   grant=grant, matched='tag')
    return dict(tag=tag, role=role, ok=ok, markers=markers, expected=expected, prompt_tokens=prompt_tokens)


class ReuseTests(unittest.TestCase):
    def severities(self, record, sequential=True):
        return [severity for severity, _ in pj.reuse_problems(record, sequential)]

    def test_a_hit_the_oracle_expected(self):
        self.assertEqual(self.severities(resolved('h', q=4096, grant=dict(h=4160, q=4096),
                                                  expected=dict(h=4160, q=4096))), [])

    def test_missing_and_inconsistent_rows(self):
        self.assertEqual(self.severities(resolved('h', rows=False)), ['FAIL'])
        self.assertEqual(self.severities(resolved('h', q=0, l=99)), ['FAIL'])
        self.assertEqual(self.severities(resolved('h', q=2048, grant=dict(h=4160, q=4096),
                                                  expected=dict(h=4160, q=4096))), ['FAIL', 'LOST'])
        self.assertEqual(self.severities(resolved('h', q=1000, expected=dict(h=4160, q=4096))), ['FAIL', 'LOST'])
        self.assertEqual(self.severities(resolved('h', ok=False, rows=False)), [], 'a failed request is judged elsewhere')

    def test_tenancy(self):
        self.assertEqual(self.severities(resolved('u', role='unsalted', q=0, grant=dict(h=0, q=0))), ['FAIL'])
        self.assertEqual(self.severities(resolved('u', role='unsalted', q=2048)), ['FAIL'])
        self.assertEqual(self.severities(resolved('u', role='unsalted', q=0)), [])
        self.assertEqual(self.severities(resolved('c', role='cold', q=2048, expected=dict(h=0, q=0))),
                         ['FAIL', 'FAIL'])

    def test_above_and_below_the_oracle(self):
        above = resolved('h', q=8192, grant=dict(h=8256, q=8192), expected=dict(h=4160, q=4096))
        self.assertEqual(self.severities(above), ['FAIL', 'FAIL'])
        self.assertEqual(self.severities(above, sequential=False), ['NOTE', 'NOTE'])
        below = resolved('h', q=0, grant=dict(h=0, q=0), expected=dict(h=4160, q=4096))
        self.assertEqual(self.severities(below), ['LOST', 'LOST'])


class ProgramCacheTests(unittest.TestCase):
    def records(self, programs):
        rows = [dict(index=i, q=q, tag='r%d' % i, programs=p) for i, (q, p) in enumerate(programs)]
        return [dict(markers=dict(rows=[row])) for row in rows]

    def test_unchanged_grew_missing_and_no_hit(self):
        problems, detail = pj.program_cache_problems(self.records([(0, 10), (0, 10), (4096, 10), (8192, 12)]))
        self.assertEqual((problems, detail['before'], detail['after']), ([], 10, 10))
        problems, _ = pj.program_cache_problems(self.records([(0, 10), (4096, 11)]))
        self.assertIn('grew from 10 to 11', problems[0])
        problems, _ = pj.program_cache_problems(self.records([(0, None), (4096, 11)]))
        self.assertIn('no programs= field', problems[0])
        problems, _ = pj.program_cache_problems(self.records([(0, 10)]))
        self.assertIn('no hit', problems[0])


class AuditTests(unittest.TestCase):
    def pair(self, a, b):
        return dict(tag='c', markers=dict(audit=a)), dict(tag='h', markers=dict(audit=b))

    def test_digests(self):
        same = dict(kv_range='0:900', kv_sha='k', slot_sha='s')
        self.assertEqual(pj.audit_problems(*self.pair(same, dict(same))), [])
        self.assertEqual([s for s, _ in pj.audit_problems(*self.pair(same, dict(same, kv_sha='x')))], ['FAIL'])
        self.assertEqual([s for s, _ in pj.audit_problems(*self.pair(same, dict(same, slot_sha='x')))], ['FAIL'])
        self.assertEqual([s for s, _ in pj.audit_problems(*self.pair(same, dict(same, kv_range='400:900')))],
                         ['NOT_EXERCISED'])
        self.assertEqual([s for s, _ in pj.audit_problems(*self.pair(same, None))], ['NOT_EXERCISED'])


class WorstTests(unittest.TestCase):
    def test_order(self):
        self.assertEqual(pj.worst(['PASS', 'UNSTABLE', 'NOT_EXERCISED']), 'UNSTABLE')
        self.assertEqual(pj.worst(['PASS', 'INFRA', 'FAIL']), 'FAIL')
        self.assertEqual(pj.worst(['PASS']), 'PASS')
        self.assertEqual(pj.worst([]), 'FAIL')


if __name__ == '__main__':
    unittest.main()
