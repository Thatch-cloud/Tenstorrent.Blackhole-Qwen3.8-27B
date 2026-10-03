"""prefix_agent_turns: the agent-turn replay's per-turn rows, its reuse rule, and the transcript comparison across arms."""

import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prefix_agent_turns as turns  # noqa: E402


def record(conv, turn, ids, q=None, prompt=1000, ttft=1.0, wall=3.0, out=None, ok=True, sha=None, role='agent', sent=0.0,
           finish='stop', build=None):
    markers = {} if q is None else dict(q=q, l=prompt, sticky_builds=[dict(ms=build)] if build else [])
    return dict(tag='%s-%s' % (conv, turn), case='agents-8', conv=conv, turn=turn, role=role, ok=ok, continuation=turn > 0,
                prompt_tokens=prompt, prompt_sha=sha or ('sha-%s-%s' % (conv, turn)), token_ids=list(ids), finish=finish,
                completion_tokens=len(ids) if out is None else out, ttft_s=ttft, wall_s=wall, sent_s=sent, markers=markers,
                content='', reasoning=None)


class RowTests(unittest.TestCase):
    def test_a_row_has_reused_new_ttft_and_decode_rate(self):
        rows = turns.turn_rows([record('a', 1, range(101), q=4096, prompt=6000, ttft=2.0, wall=7.0, build=2400.0)])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row['reused'], row['new'], row['prompt_tokens']), (4096, 1904, 6000))
        self.assertEqual(row['decode_tps'], 20.0)         # 100 tokens after the first, in 5 s
        self.assertEqual(row['build_ms'], 2400.0)
        line = turns.render_row('agent-turns-prefix', row)
        for word in ('reused=4096', 'new=1904', 'ttft=2.0', 'decode=20.0', 'build=2400.0'):
            self.assertIn(word, line)

    def test_a_control_turn_has_no_reuse_marker_and_a_failed_turn_says_so(self):
        rows = turns.turn_rows([record('a', 0, [1, 2, 3]), record('a', 1, [], ok=False, sent=1.0)])
        self.assertEqual((rows[0]['reused'], rows[0]['new']), (None, None))
        self.assertIn('FAILED', turns.render_row('x', rows[1]))

    def test_only_agent_records_count_and_a_single_token_answer_has_no_rate(self):
        rows = turns.turn_rows([record('a', 0, [1], out=1), record('a', 0, [1, 2], role='cold')])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]['decode_tps'])


class ReuseTests(unittest.TestCase):
    def test_every_continuing_agent_reusing_passes_and_a_cold_agent_is_not_exercised(self):
        good = [record('a', 0, [1]), record('a', 1, [1], q=2048), record('b', 0, [1]), record('b', 1, [1], q=4096)]
        problems, missing, lines = turns.reuse_findings(good)
        self.assertEqual((problems, missing), ([], []))
        self.assertIn('2 of 2', lines[0])
        bad = good[:3] + [record('b', 1, [1], q=0)]
        problems, missing, _ = turns.reuse_findings(bad)
        self.assertEqual(problems, [])
        self.assertEqual(len(missing), 1)
        self.assertIn('b', missing[0])

    def test_no_continuation_and_no_served_turn_are_not_exercised(self):
        self.assertTrue(turns.reuse_findings([])[1])
        self.assertIn('no agent continued', turns.reuse_findings([record('a', 0, [1])])[1][0])


class CompareTests(unittest.TestCase):
    def arm(self, scale=1.0, changes=None):
        out = []
        for conv in ('a', 'b'):
            for turn in range(3):
                ids = [conv == 'a', turn, 7]
                out.append(record(conv, turn, [int(x) for x in ids], prompt=1000 + turn, ttft=scale * (1 + turn)))
        for key, ids in (changes or {}).items():
            for r in out:
                if (r['conv'], r['turn']) == key:
                    r['token_ids'] = ids
        return out

    def test_identical_transcripts_pass_and_the_ttft_ratio_is_paired(self):
        result = turns.compare(self.arm(), self.arm(scale=2.0), 'control', 'prefix')
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(result['counts']['IDENTICAL'], 6)
        self.assertTrue(any('median 0.5' in line for line in result['lines']), result['lines'])

    def test_a_divergence_is_reported_once_at_its_first_turn_and_later_turns_are_not_comparable(self):
        other = self.arm(changes={('a', 1): [9, 9]})
        for r in other:
            if r['conv'] == 'a' and r['turn'] == 2:
                r['prompt_sha'] = 'different'
        result = turns.compare(self.arm(), other, 'control', 'prefix')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['counts']['DIVERGED'], 1)
        self.assertEqual(result['counts']['NOT_COMPARABLE'], 1)
        self.assertEqual(len(result['problems']), 1, 'the downstream turn is a consequence, not a second finding')
        self.assertIn('a turn 1 diverged', result['problems'][0])

    def test_a_not_comparable_turn_without_an_earlier_divergence_is_a_problem(self):
        other = self.arm()
        other[1]['prompt_sha'] = 'x'
        result = turns.compare(self.arm(), other)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('no earlier turn of it diverged', result['problems'][0])

    def test_a_turn_only_one_arm_ran_and_a_failed_turn_fail_the_comparison(self):
        short = self.arm()[:-1]
        result = turns.compare(self.arm(), short, 'control', 'prefix')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('only control ran' in problem for problem in result['problems']))
        failed = self.arm()
        failed[0]['ok'] = False
        self.assertEqual(turns.compare(self.arm(), failed)['verdict'], 'FAIL')

    def test_nothing_in_common_fails(self):
        self.assertEqual(turns.compare([], [])['verdict'], 'FAIL')

    def test_the_offline_comparator_reads_results_directories_and_exits_by_the_verdict(self):
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for name, records in (('A', self.arm()), ('B', self.arm(scale=0.5)), ('C', self.arm(changes={('b', 2): [5]}))):
                folder = os.path.join(root, name, 'agent-turns-' + name)
                os.makedirs(folder)
                with open(os.path.join(folder, 'records.jsonl'), 'w', encoding='utf-8') as handle:
                    handle.write(''.join(json.dumps(r) + '\n' for r in records))
                paths.append(os.path.join(root, name))
            lines = []
            self.assertEqual(turns.main([paths[0], paths[1], '--label-a', 'control', '--label-b', 'prefix'], lines.append), 0)
            self.assertIn('agent turns control vs prefix: PASS', lines[-1])
            lines = []
            self.assertEqual(turns.main([paths[0], paths[2]], lines.append), 1)
            self.assertIn('FAIL', lines[-1])
            self.assertEqual(turns.main([paths[0], os.path.join(root, 'missing')], lambda text: None), 2)


if __name__ == '__main__':
    unittest.main()
