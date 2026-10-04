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
           finish='stop', build=None, want=None, chunks=None):
    markers = {} if q is None else dict(q=q, l=prompt, sticky_builds=[dict(ms=build)] if build else [])
    extra = {} if want is None else dict(expected=dict(q=want, h=want))
    if chunks is not None:
        extra['chunk_times'] = list(chunks)
    return dict(extra, **dict(tag='%s-%s' % (conv, turn), case='agents-8', conv=conv, turn=turn, role=role, ok=ok, continuation=turn > 0,
                prompt_tokens=prompt, prompt_sha=sha or ('sha-%s-%s' % (conv, turn)), token_ids=list(ids), finish=finish,
                completion_tokens=len(ids) if out is None else out, ttft_s=ttft, wall_s=wall, sent_s=sent, markers=markers,
                content='', reasoning=None))


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
        good = [record('a', 0, [1]), record('a', 1, [1], q=2048, want=2048), record('b', 0, [1]),
                record('b', 1, [1], q=4096, want=4096)]
        problems, missing, lines = turns.reuse_findings(good)
        self.assertEqual((problems, missing), ([], []))
        self.assertIn('2 of 2', lines[0])
        self.assertTrue(any('judged against the oracle: 2 of 2' in line for line in lines), lines)
        bad = good[:3] + [record('b', 1, [1], q=0, want=4096)]
        problems, missing, _ = turns.reuse_findings(bad)
        self.assertEqual(len(problems), 1, 'a miss the oracle did not expect is a problem now')
        self.assertTrue(any('b' in entry for entry in missing), 'and the conversation that never reused is still reported')

    def test_a_later_cold_continuation_fails_when_the_oracle_says_it_reuses_and_nothing_was_evicted(self):
        rows = [record('a', 0, [1]), record('a', 1, [1], q=4096, want=4096), record('a', 2, [1], q=0, want=6144)]
        problems, missing, lines = turns.reuse_findings(rows)
        self.assertEqual(missing, [], 'the conversation did reuse once: the old rule was satisfied')
        self.assertEqual(len(problems), 1)
        self.assertIn('a turn 2 restored Q=0 where the oracle says 6144', problems[0])
        self.assertIn('no eviction is on record', problems[0])

    def test_an_eviction_on_record_excuses_a_shortfall_but_never_an_excess(self):
        rows = [record('a', 0, [1]), record('a', 1, [1], q=2048, want=4096)]
        problems, _, lines = turns.reuse_findings(rows, dict(evicted_coupled=3))
        self.assertEqual(problems, [])
        self.assertTrue(any('1 short of it, excused by 3 evictions' in line for line in lines), lines)
        problems, _, _ = turns.reuse_findings(rows, dict(evicted_lru=0, evicted_coupled=0))
        self.assertEqual(len(problems), 1)
        excess = [record('a', 0, [1]), record('a', 1, [1], q=8192, want=4096)]
        problems, _, _ = turns.reuse_findings(excess, dict(evicted_coupled=9))
        self.assertIn('more than the oracle', problems[0])

    def test_without_an_oracle_expectation_nothing_is_judged_and_that_is_not_exercised(self):
        rows = [record('a', 0, [1]), record('a', 1, [1], q=2048)]
        problems, missing, lines = turns.reuse_findings(rows)
        self.assertEqual(problems, [])
        self.assertTrue(any('no continuation could be judged against the oracle' in entry for entry in missing), missing)
        self.assertTrue(any('carry no oracle expectation' in line for line in lines))

    def test_a_turn_the_oracle_expects_to_reuse_with_no_row_is_not_exercised(self):
        rows = [record('a', 0, [1]), record('a', 1, [1], q=2048, want=2048), record('a', 2, [1], want=4096)]
        problems, missing, _ = turns.reuse_findings(rows)
        self.assertEqual(problems, [])
        self.assertTrue(any('a turn 2: the oracle says Q=4096 but the turn has no [PREFIX] row' in entry for entry in missing))

    def test_no_continuation_and_no_served_turn_are_not_exercised(self):
        self.assertTrue(turns.reuse_findings([])[1])
        self.assertIn('no agent continued', turns.reuse_findings([record('a', 0, [1])])[1][0])


class StallTests(unittest.TestCase):
    """The per-arrival stall: the other seats' longest chunk gap while a turn prefills (sent .. first token)."""

    def test_the_others_longest_gap_inside_the_prefill_window_is_read_per_arrival(self):
        # seat b streams every 0.1 s from t=0 and then stops for 4 s while seat a's turn prefills from t=5 to t=11
        steady = [round(0.1 * i, 3) for i in range(50)]                   # chunks at 0 .. 4.9 s
        resumed = [9.0 + 0.1 * i for i in range(30)]                       # the next chunk comes at 9.0 s, a 4.1 s gap
        victim = record('b', 1, range(80), q=1, sent=0.0, ttft=0.2, wall=12.0, chunks=steady + resumed)
        arrival = record('a', 1, range(5), q=2048, prompt=6000, sent=5.0, ttft=6.0, wall=8.0, chunks=[6.0, 6.1])
        rows = turns.stall_rows([victim, arrival])
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]['conv'], rows[0]['others'], rows[0]['new']), ('a', 1, 3952))
        self.assertAlmostEqual(rows[0]['worst_gap_s'], 4.1, places=2)
        self.assertIn('worst_gap=4.1', turns.render_stall('arm', rows[0]))
        self.assertIn('max at a turn 1', turns.stall_summary('x', rows))

    def test_a_seat_that_was_not_streaming_across_the_window_is_not_counted(self):
        done = record('b', 1, range(10), sent=0.0, ttft=0.2, wall=1.0, chunks=[0.2, 0.3, 0.4])
        later = record('c', 1, range(10), sent=20.0, ttft=0.2, wall=1.0, chunks=[0.2, 0.3])
        arrival = record('a', 1, range(5), sent=5.0, ttft=6.0, wall=8.0, chunks=[6.0, 6.1])
        self.assertEqual(turns.stall_rows([done, later, arrival]), [])
        self.assertIn('no turn prefilled', turns.stall_summary('x', []))

    def test_records_without_chunk_times_give_no_rows_and_compare_says_so(self):
        arm = [record('a', 0, [1, 2]), record('b', 0, [1, 2])]
        self.assertEqual(turns.stall_rows(arm), [])
        result = turns.compare(arm, arm, 'control', 'prefix')
        self.assertTrue(any('control stall: no turn prefilled' in line for line in result['lines']), result['lines'])


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

    def test_an_answered_turn_without_output_token_ids_is_an_error_not_a_text_comparison(self):
        other = self.arm()
        other[0]['token_ids'] = None
        result = turns.compare(self.arm(), other, 'control', 'prefix')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['counts']['ERROR'], 1)
        self.assertIn('per token', result['problems'][0])

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
