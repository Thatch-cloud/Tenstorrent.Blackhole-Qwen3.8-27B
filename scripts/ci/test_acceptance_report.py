"""acceptance_report: tokens emitted per round, per user, from a finished gate run's server log.

The fixtures are real excerpts (fixtures/acceptance/): v155 (run 35926504810, 4 x 131k packed,
synthetic prompt) and v149 (run 35921558856, four single-stream 32k requests one after another),
reduced to the lines the report reads. The rest are hand-built detail streams, the shape the
gate's real-text arms record."""

import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acceptance_report as ar  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'acceptance'


def fixture(tag):
    log = (FIXTURES / ('%s-server-excerpt.log' % tag)).read_text(encoding='utf-8')
    streams = json.loads((FIXTURES / ('%s-streams.json' % tag)).read_text(encoding='utf-8'))['streams']
    return log, streams


class PackedRunTests(unittest.TestCase):
    """v155: four users, 33 packed rounds, the record of the last two requests after SIGTERM."""

    @classmethod
    def setUpClass(cls):
        cls.log, cls.streams = fixture('v155')
        cls.report = ar.report(cls.log, cls.streams)

    def test_every_phases_record_is_read_including_those_after_the_shutdown(self):
        records, unparsed = ar.phase_records(self.log)
        self.assertEqual(unparsed, 0)
        self.assertEqual([len(r['blocks']) for r in records], [38, 39, 47, 47])
        self.assertEqual([sum(b['committed'] for b in r['blocks']) for r in records], [255] * 4)
        self.assertEqual([sum(1 for b in r['blocks'] if b['packed']) for r in records], [33] * 4)
        self.assertEqual(self.report['sources']['phases_after_shutdown'], 2)
        self.assertIsNotNone(self.report['sources']['shutdown_line'])

    def test_the_packed_audit_agrees_with_the_phases_records(self):
        audit = ar.packed_audit(self.log)
        self.assertEqual(sorted(len(v) for v in audit.values()), [33] * 4)
        self.assertTrue(all(line['prefix'] == line['emitted'] for lines in audit.values() for line in lines))
        self.assertEqual(self.report['agreement']['packed_audit_vs_phases'], '4/4')

    def test_the_synthetic_prompt_never_emits_more_than_ten(self):
        full = self.report['overall']['full_draft']
        self.assertEqual(full['max_emitted'], 10)
        self.assertEqual((full['p_emitted_gt_11'], full['p_emitted_eq_16']), (0.0, 0.0))
        self.assertEqual(full['rounds'], 132, 'the 33 packed rounds x 4 users are the full-draft rounds')
        self.assertEqual(sum(full['histogram'].values()), full['rounds'])
        positions = self.report['overall']['per_position_full_draft']
        self.assertEqual(len(positions), 15)
        self.assertEqual(positions[9:], [0.0] * 6)
        self.assertEqual(self.report['packed']['full_draft']['rounds'], 132)

    def test_vllms_interval_is_parsed_and_consistent(self):
        intervals = ar.spec_decoding(self.log)
        self.assertEqual(len(intervals), 1)
        self.assertEqual((intervals[0]['accepted'], intervals[0]['drafted'], intervals[0]['drafts']), (402, 1020, 68))
        self.assertEqual(intervals[0]['per_position'][:3], [0.941, 0.941, 0.926])
        self.assertIs(self.report['agreement']['vllm_consistent_with_phases'], True)
        self.assertEqual(self.report['vllm']['per_position_accepted'][0], 64)

    def test_active_users_per_decode_step(self):
        rounds = self.report['rounds']
        self.assertEqual(rounds['decode_steps_by_active_users'], {'4': 38, '3': 1, '2': 8})
        self.assertEqual((rounds['prefill_steps'], rounds['packed_rounds']), (3, 33))

    def test_without_detail_streams_only_unique_chunk_counts_attribute(self):
        users = {u['user']: u for u in self.report['users']}
        self.assertEqual(users[2]['attributed_by'], 'chunk-count')
        self.assertEqual(users[3]['attributed_by'], 'chunk-count')
        self.assertIsNone(users[0]['attributed_by'], 'two 47-block records, two 48-chunk streams: ambiguous')
        self.assertEqual(users[2]['stream_agreement'], 'chunks=blocks+1')
        self.assertTrue(users[2]['completion_is_seed_plus_rounds'])
        self.assertEqual(self.report['agreement']['users_attributed'], '2/4')

    def test_the_summary_line(self):
        line = self.report['summary_line']
        self.assertTrue(line.startswith('[ACCEPT] full-draft rounds=132 mean=6.88 '), line)
        self.assertIn('max=10', line)
        self.assertIn('active-user steps 4:38 3:1 2:8', line)
        self.assertIn('agree audit=4/4 vllm=True', line)
        json.dumps(self.report)

    def test_the_median_gap_rate_uses_the_existing_convention(self):
        rates = ar.decode_rates(self.streams)
        user = rates['users'][0]
        expected = (256 / 48) / (sorted(self.streams[0]['gaps_ms'])[23] / 1000.0)
        self.assertAlmostEqual(user['median_gap_tok_s'], expected, places=1)
        self.assertIsNone(rates['window'], 'no chunk times without detail streams')
        self.assertTrue(ar.rate_line(rates).startswith('[RATE] per-user tok/s steady '))
        self.assertIn('| median-gap ', ar.rate_line(rates))

    def test_the_steady_rate_is_the_packed_rounds_without_stalls(self):
        """v155's median-gap rate reads 20-25 tok/s because the 4-row tail rounds dilute tokens per
        chunk; the packed rounds alone, minus the first (it waits for the other prefills) and the two
        ~1 s capture stalls, give the programme's ~28 tok/s by its own mean/median convention."""
        users = {u['user']: u for u in self.report['users']}
        for user in (2, 3):
            with self.subTest(user=user):
                steady = users[user]['steady']
                self.assertEqual(steady['source'], 'gaps_ms')
                self.assertEqual(steady['rounds'] + steady['stalls_excluded'], 32, '33 packed rounds minus the first')
                self.assertEqual(steady['stalls_excluded'], 2)
                self.assertTrue(250.0 < steady['median_round_ms'] < 270.0, steady)
                self.assertTrue(25.0 < steady['steady_tok_s'] < 28.0, steady)
                self.assertTrue(27.5 < steady['mean_over_median_tok_s'] < 29.5, steady)
        self.assertIsNone(users[0]['steady'], 'an unattributed user has no blocks to join')
        rates = ar.decode_rates(self.streams, acceptance=self.report)
        self.assertEqual(rates['users'][2]['steady_tok_s'], users[2]['steady']['steady_tok_s'])
        self.assertGreater(rates['users'][2]['steady_tok_s'], rates['users'][2]['median_gap_tok_s'])
        self.assertIsNone(rates['users'][0]['steady_tok_s'])


class SequentialRunTests(unittest.TestCase):
    """v149: four lone 32k streams; vLLM printed five intervals and never the last."""

    @classmethod
    def setUpClass(cls):
        cls.log, cls.streams = fixture('v149')
        cls.report = ar.report(cls.log, cls.streams, sequential=True)

    def test_vllms_first_interval_is_the_first_users_first_six_blocks(self):
        interval = ar.spec_decoding(self.log)[0]
        first = ar.phase_records(self.log)[0][0]['blocks'][:6]
        self.assertEqual([b['committed'] for b in first], [11, 10, 5, 6, 1, 10])
        self.assertEqual(interval['accepted'], sum(b['committed'] - 1 for b in first))
        self.assertEqual(interval['drafted'], sum(b['rows'] - 1 for b in first))
        self.assertEqual(interval['drafts'], 6)

    def test_every_interval_recovers_its_draft_count_exactly(self):
        intervals = ar.spec_decoding(self.log)
        self.assertEqual([(i['drafts'], i['drafts_recovered']) for i in intervals],
                         [(6, 'exact'), (30, 'exact'), (35, 'exact'), (37, 'exact'), (5, 'exact')])
        self.assertEqual(self.report['vllm']['drafts_recovered'], {'exact': 5})
        # 442 drafted over 30 drafts: one of them proposed 7, so D / 15 would not even be an integer.
        self.assertEqual(intervals[1]['drafted'], 442)

    def test_records_are_attributed_by_order_in_a_sequential_run(self):
        self.assertEqual([u['attributed_by'] for u in self.report['users']], ['order'] * 4)
        self.assertEqual(self.report['agreement']['users_attributed'], '4/4')
        unordered = ar.report(self.log, self.streams)
        self.assertEqual(unordered['agreement']['users_attributed'], '1/4', 'what the chunk counts alone give')
        wrong = ar.report(self.log, self.streams, [32768, 32768, 1, 32768], sequential=True)
        self.assertEqual([u['attributed_by'] for u in wrong['users']][:2], ['order', 'order'])
        self.assertNotEqual(wrong['users'][2]['attributed_by'], 'order', 'a length that disagrees is not taken')

    def test_the_offline_command_reads_a_gate_report_and_its_log(self):
        """Default-mode arms carry no acceptance report (the gate keeps their output as it was), so
        the same report is run offline from the artifacts."""
        import io
        import tempfile
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory, 'm3native-gate.json')
            gate.write_text(json.dumps(dict(sequential_users=4, streams=self.streams)), encoding='utf-8')
            out = Path(directory, 'out.json')
            with redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(ar.main([str(FIXTURES / 'v149-server-excerpt.log'), str(gate), '--json', str(out)]), 0)
            written = json.loads(out.read_text(encoding='utf-8'))
        self.assertTrue(printed.getvalue().startswith('[ACCEPT] full-draft rounds='))
        self.assertIn('[RATE] per-user tok/s steady ', printed.getvalue())
        self.assertEqual(written['acceptance']['agreement']['users_attributed'], '4/4', 'sequential from the report')

    def test_the_steady_rate_is_the_16_row_rounds_without_capture_stalls(self):
        """v149's first user carries seven capture stalls of 1-1.5 s; without them every user
        decodes at ~64-65 tok/s, where its own first-to-last window read 17.6 tok/s."""
        steady = [u['steady'] for u in self.report['users']]
        self.assertEqual([s['stalls_excluded'] for s in steady], [7, 1, 0, 0])
        for entry in steady:
            self.assertTrue(63.0 < entry['steady_tok_s'] < 66.0, entry)
            self.assertTrue(100.0 < entry['median_round_ms'] < 120.0, entry)
        rates = ar.decode_rates(self.streams, concurrent=False, acceptance=self.report)
        self.assertTrue(64.0 < rates['mean_steady_tok_s'] < 65.5, rates['mean_steady_tok_s'])

    def test_single_stream_logs_no_packed_audit_and_one_active_user(self):
        self.assertEqual(self.report['sources']['packed_audit_requests'], 0)
        self.assertIsNone(self.report['agreement']['packed_audit_vs_phases'])
        self.assertEqual(self.report['rounds']['decode_steps_by_active_users'], {'1': 146})
        self.assertIsNone(self.report['packed'])
        self.assertEqual(self.report['sources']['spec_decoding_intervals'], 5)
        self.assertLess(self.report['vllm']['coverage_of_drafting_blocks'], 1.0)

    def test_full_draft_and_tail_rounds(self):
        overall = self.report['overall']
        self.assertEqual(overall['full_draft']['max_emitted'], 11)
        self.assertEqual(overall['all']['rounds'], 146)
        self.assertEqual(overall['all']['tokens'], 4 * 255)


class VariableUserRunTests(unittest.TestCase):
    """v185 (run 35950414318, 4 x 131k real text, image A7): the baseline the variable-user arms
    (R1, R2) are read against. Its decode wall is mostly NOT the packed round: 26 packed rounds
    hold 17% of it; three live users on their 4-row engines hold 52%."""

    @classmethod
    def setUpClass(cls):
        cls.log, cls.streams = fixture('v185')
        cls.report = ar.report(cls.log, cls.streams)
        cls.rates = ar.decode_rates(cls.streams, acceptance=cls.report)

    def test_rounds_split_by_live_count_and_packing(self):
        split = self.report['rounds_by_live']
        self.assertTrue(split['aligned'], split['alignment'])
        self.assertEqual(split['decode_steps'], 134)
        groups = [(g['live'], g['packed'], g['rounds'], g['timed_rounds'], g['median_round_ms'],
                   g['tokens_per_user_per_round'], g['rows_per_user']) for g in split['groups']]
        self.assertEqual(groups, [(4, True, 26, 26, 246.5, 5.327, 16.0), (4, False, 3, 3, 521.0, 2.0, 3.583),
                                  (3, False, 55, 55, 376.0, 1.867, 3.988), (2, False, 38, 38, 242.5, 1.171, 3.934),
                                  (1, False, 12, 11, 122.0, 3.75, 3.75)])
        by = {(g['live'], g['packed']): g for g in split['groups']}
        self.assertEqual(by[(3, False)]['per_user_tok_s'], 4.96, 'the plan: 5.0 tok/s per user at three live')
        self.assertAlmostEqual(sum(g['wall_share'] for g in split['groups'] if not g['packed']), 0.8266, places=3)
        self.assertEqual(sum(g['tokens'] for g in split['groups']), 4 * 255, 'every decode token, seeds apart')
        self.assertEqual(self.report['rounds']['decode_steps_by_active_users'], {'4': 29, '3': 55, '2': 38, '1': 12})
        self.assertIn(' | by live 4p:26x246ms/5.33 4s:3x521ms/2.00 3s:55x376ms/1.87 2s:38x242ms/1.17 '
                      '1s:12x122ms/3.75', self.report['summary_line'])
        json.dumps(self.report)

    def test_completion_rates_and_the_aggregate(self):
        completion = self.rates['completion']
        self.assertEqual(completion['decode_start'], 'the last seed chunk')
        self.assertEqual([u['completion_tok_s'] for u in completion['users']], [6.65, 6.39, 8.74, 30.29])
        self.assertEqual([u['finish_s'] for u in completion['users']], [38.4904, 40.0622, 29.276, 8.4516])
        self.assertEqual((completion['tokens'], completion['decode_wall_s']), (1024, 40.0622))
        self.assertEqual((self.rates['mean_completion_tok_s'], self.rates['aggregate_tok_s']), (13.02, 25.56))
        self.assertEqual([u['completion_tok_s'] for u in self.rates['users']], [6.65, 6.39, 8.74, 30.29])
        # the steady rate reads the packed round alone
        self.assertGreater(self.rates['mean_steady_tok_s'], 20.0)
        self.assertIn('| completion 6.7 6.4 8.7 30.3 (mean 13.0) aggregate 25.6 over a 40.06 s decode wall',
                      ar.rate_line(self.rates))

    def test_a_sequential_reference_is_timed_by_live_count_but_not_split(self):
        log, streams = fixture('v149')
        split = ar.report(log, streams, sequential=True)['rounds_by_live']
        self.assertFalse(split['aligned'])
        self.assertTrue(split['alignment'].startswith('step 1: 1 live over 16 rows, the records hold 4 blocks'))
        self.assertEqual([(g['live'], g['packed'], g['rounds'], g['tokens']) for g in split['groups']],
                         [(1, None, 146, None)])
        # the excerpt holds no prefill lines: only the last step has no successor to time it by
        self.assertEqual(split['groups'][0]['timed_rounds'], 145)
        self.assertTrue(100.0 < split['groups'][0]['median_round_ms'] < 120.0)

    def test_a_step_the_records_do_not_explain_withholds_the_token_split(self):
        stamp = '(EngineCore pid=1) 2026-09-24 03:18:%06.3f | INFO     | serving_worker_hook:_execute:229 - '
        execute = stamp + '[PHASE] execute total=%d new=%d cached=%d spec=%d finished=[] preempted=[]'
        records = [phases_line([block(10, 16, 5, packed=True), block(15, 4, 2)]),
                   phases_line([block(10, 16, 7, packed=True), block(17, 4, 4)])]
        good = [execute % (1.0, 32, 0, 2, 2), execute % (1.25, 8, 0, 2, 2), execute % (2.0, 2048, 1, 0, 0)]
        split = ar.round_split('\n'.join(good + records), ar.phase_records('\n'.join(records))[0])
        self.assertTrue(split['aligned'])
        # the second step is followed by a prefill (new=1), which in a sequential reference would
        # also hold the client's wait for its next request: counted, never timed
        self.assertEqual([(g['live'], g['packed'], g['timed_rounds'], g['median_round_ms'], g['tokens']) for g in split['groups']],
                         [(2, True, 1, 250.0, 12), (2, False, 0, None, 6)])
        bad = [execute % (1.0, 32, 0, 2, 2), execute % (1.25, 9, 0, 2, 2)]
        split = ar.round_split('\n'.join(bad), ar.phase_records('\n'.join(records))[0])
        self.assertEqual((split['aligned'], split['alignment']),
                         (False, 'step 2: 2 live over 9 rows, the records hold 2 blocks over 8 rows'))
        self.assertIsNone(ar.round_split('\n'.join(records), ar.phase_records('\n'.join(records))[0]))

    def test_completion_rates_need_every_streams_chunk_times(self):
        a = dict(started_s=0.0, chunk_s=[1.0, 2.0, 3.0], completion_tokens=9)
        b = dict(started_s=0.5, chunk_s=[2.0, 3.0, 4.5], completion_tokens=12)
        concurrent = ar.completion_rates([a, b])
        self.assertEqual(concurrent['decode_wall_s'], 2.5, 'from b\'s seed at 2.5 s to its last chunk at 5.0 s')
        self.assertEqual([u['completion_tok_s'] for u in concurrent['users']], [18.0, 4.8])
        self.assertEqual(concurrent['aggregate_tok_s'], 8.4)
        alone = ar.completion_rates([a, b], concurrent=False)
        self.assertEqual([u['completion_tok_s'] for u in alone['users']], [4.5, 4.8])
        self.assertIsNone(alone['aggregate_tok_s'])
        self.assertIsNone(ar.completion_rates([a, dict(b, chunk_s=None)]))
        self.assertIsNone(ar.completion_rates([]))


def detail_stream(request_id, committed, start, round_s=0.25, finish='length'):
    """A detail stream as stream_once(detail=True) records it: the seed chunk, then one per round."""
    tokens = [1] + list(committed)
    return dict(request_id=request_id, started_s=start, chunk_s=[1.0 + round_s * i for i in range(len(tokens))],
                chunk_tokens=tokens, tokens=len(tokens), completion_tokens=sum(tokens), finish_reason=finish,
                gaps_ms=[round_s * 1000.0] * (len(tokens) - 1), ttft_s=1.0)


def phases_line(blocks):
    return '(EngineCore pid=57) ' + json.dumps(dict(stage='fast_serving_phases', blocks=blocks, cancelled=False,
                                                    finished=True))


def block(position, rows, committed, packed=False):
    entry = dict(position=position, rows=rows, committed=committed)
    if packed:
        entry['verifier'] = dict(users=4, packed=True, segment=0)
    return entry


class DetailStreamTests(unittest.TestCase):
    def test_real_text_records_attribute_by_prompt_length_and_streams_agree(self):
        lengths = [32710, 32731, 32704, 32768]
        committed = [[16, 12, 9, 3], [5, 16, 16, 2], [9, 9, 9, 1], [4, 4, 4, 4]]
        order = [2, 0, 3, 1]      # close order
        lines = [phases_line([block(lengths[u] + sum(committed[u][:i]), 16, c, packed=True)
                              for i, c in enumerate(committed[u])]) for u in order]
        streams = [detail_stream('cmpl-%d' % u, committed[u], 100.0 + u) for u in range(4)]
        audit = ['[PACKED] request=cmpl-%d-0-ab segment=0 position=1 prefix=%d emitted=%d predictions=[1]' % (u, c, c)
                 for u in range(4) for c in committed[u]]
        report = ar.report('\n'.join(lines + audit), streams, lengths)
        users = report['users']
        self.assertEqual([u['attributed_by'] for u in users], ['position'] * 4)
        self.assertEqual([u['first_position'] for u in users], lengths)
        self.assertEqual([u['stream_agreement'] for u in users], ['exact'] * 4)
        self.assertEqual([u['packed_audit']['agrees_with_phases'] for u in users], [True] * 4)
        self.assertEqual(users[1]['full_draft']['max_emitted'], 16)
        self.assertEqual(users[1]['full_draft']['p_emitted_eq_16'], round(2 / 3, 4), 'the terminal block is left out')
        self.assertEqual(report['agreement']['stream_vs_phases'], {'exact': 4})

    def test_equal_prompt_lengths_fall_back_to_the_streams(self):
        committed = [[6, 7, 8], [8, 7, 6]]
        lines = [phases_line([block(100, 16, c) for c in committed[u]]) for u in (1, 0)]
        streams = [detail_stream('a', committed[0], 0.0), detail_stream('b', committed[1], 0.0)]
        report = ar.report('\n'.join(lines), streams, [100, 100])
        self.assertEqual([u['attributed_by'] for u in report['users']], ['stream', 'stream'])
        self.assertEqual(report['records'][0]['user'], 1)

    def test_coalesced_chunks_still_agree_and_a_wrong_split_does_not(self):
        blocks = [block(0, 16, c) for c in (5, 6, 7)]
        stream = detail_stream('x', [5, 6, 7], 0.0)
        self.assertEqual(ar.stream_agreement(stream, blocks), 'exact')
        stream['chunk_tokens'] = [1, 11, 7]
        self.assertEqual(ar.stream_agreement(stream, blocks), 'coalesced')
        stream['chunk_tokens'] = [6, 6, 7]
        self.assertEqual(ar.stream_agreement(stream, blocks), 'coalesced', 'seed coalesced with round 1')
        stream['chunk_tokens'] = [1, 4, 7, 7]
        self.assertEqual(ar.stream_agreement(stream, blocks), 'disagree')
        stream['chunk_tokens'] = [1, None, 7]
        self.assertIsNone(ar.stream_agreement(dict(stream, tokens=None), blocks))

    def test_per_position_conventions(self):
        blocks = [dict(block(0, 16, 16), terminal=False), dict(block(0, 16, 1), terminal=False),
                  dict(block(0, 4, 4), terminal=False)]
        conditional = ar.per_position(blocks)
        self.assertEqual(conditional[:4], [round(2 / 3, 3), round(2 / 3, 3), round(2 / 3, 3), 0.5])
        self.assertEqual(conditional[14], 0.5)
        vllm = ar.per_position(blocks, convention='vllm')
        self.assertEqual(vllm[3], round(1 / 3, 3))

    def test_an_eos_terminal_block_is_not_an_acceptance_sample(self):
        record = dict(blocks=[block(0, 16, 9), block(9, 16, 2)])
        marked = ar.acceptance_blocks(record)
        section = ar.section(marked)
        self.assertEqual(section['full_draft']['rounds'], 1)
        self.assertEqual(section['all']['rounds'], 2)

    def test_all_active_window_rates_leave_the_seed_chunk_out(self):
        """b prefills last: its seed lands at 2.0 s and its first round only at 4.0 s. Timed from its
        seed, b's window would hold that wait and read 4.5 tok/s instead of 6."""
        a = dict(started_s=0.0, chunk_s=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], chunk_tokens=[1, 8, 8, 8, 8, 8, 8],
                 tokens=7, completion_tokens=49, gaps_ms=[500.0] + [1000.0] * 5)
        b = dict(started_s=0.0, chunk_s=[2.0, 4.0, 5.0, 6.0], chunk_tokens=[1, 6, 6, 6], tokens=4, completion_tokens=19,
                 gaps_ms=[2000.0, 1000.0, 1000.0])
        rates = ar.decode_rates([a, b])
        self.assertEqual(rates['window'], dict(seconds=2.0, empty=False), 'from b\'s first ROUND, not its seed')
        self.assertEqual(rates['users'][0]['all_active_tok_s'], 8.0)
        self.assertEqual(rates['users'][0]['all_active_chunks'], 3)
        self.assertEqual(rates['users'][1]['all_active_tok_s'], 6.0)
        alone = ar.decode_rates([a, b], concurrent=False)
        self.assertEqual(alone['users'][0]['all_active_tok_s'], 8.0)
        self.assertEqual(alone['users'][1]['all_active_seconds'], 2.0)
        self.assertEqual(alone['users'][1]['all_active_tok_s'], 6.0)
        self.assertEqual(rates['users'][0]['median_gap_tok_s'], 7.0)
        self.assertIsNone(rates['mean_steady_tok_s'], 'no acceptance report, no steady rate')

    def test_steady_rate_from_detail_streams_joins_chunks_to_blocks(self):
        committed = [9, 12, 8, 10, 11, 9, 4]
        stream = detail_stream('cmpl-0', committed, 0.0, round_s=0.25)
        for index in range(3, len(stream['chunk_s'])):
            stream['chunk_s'][index] += 2.0     # a 2 s capture stall before the chunk of round index 2
        records, _ = ar.phase_records(phases_line([block(32768 + i, 16, c, packed=True) for i, c in enumerate(committed)]))
        blocks = ar.acceptance_blocks(records[0])
        steady = ar.steady_rate(stream, blocks, concurrent=True)
        self.assertEqual(steady['source'], 'chunk_s')
        # Rounds 2..6 minus the terminal one (4) and the stalled one: 12, 10, 11, 9 over 4 x 0.25 s.
        self.assertEqual((steady['rounds'], steady['stalls_excluded'], steady['tokens']), (4, 1, 42))
        self.assertEqual(steady['steady_tok_s'], 42.0)
        self.assertIsNone(ar.steady_rate(stream, blocks, concurrent=False)['steady_tok_s'], 'no 16-row sequential rounds')
        stream['chunk_tokens'][2] = None
        self.assertIsNone(ar.steady_rate(stream, blocks, concurrent=True), 'chunks that cannot be tied to rounds')

    def test_equal_real_text_lengths_attribute_by_the_packed_audit(self):
        """Every real-text prompt is the arm's request context, so position says nothing; the audit's
        request id names the stream and its emitted sequence names the record."""
        committed = [[16, 12, 9, 3], [5, 16, 16, 2], [9, 9, 9, 1], [4, 4, 4, 4]]
        lines = [phases_line([block(32768 + sum(committed[u][:i]), 16, c, packed=True)
                              for i, c in enumerate(committed[u])]) for u in (3, 1, 0, 2)]
        streams = [dict(request_id='cmpl-%d' % u, tokens=5, completion_tokens=sum(committed[u]) + 1) for u in range(4)]
        audit = ['[PACKED] request=cmpl-%d-0-ab segment=0 position=1 prefix=%d emitted=%d predictions=[1]' % (u, c, c)
                 for u in range(4) for c in committed[u]]
        report = ar.report('\n'.join(lines + audit), streams, [32768] * 4)
        self.assertEqual([u['attributed_by'] for u in report['users']], ['audit'] * 4)
        self.assertEqual([r['user'] for r in report['records']], [3, 1, 0, 2])

    def test_a_chunk_count_two_records_share_attributes_neither(self):
        lines = [phases_line([block(0, 16, c) for c in committed]) for committed in ([5, 6], [7, 8], [9])]
        streams = [dict(tokens=3), dict(tokens=9), dict(tokens=2)]
        report = ar.report('\n'.join(lines), streams)
        self.assertEqual([u['attributed_by'] for u in report['users']], [None, None, 'chunk-count'])

    def test_draft_recovery_edge_cases(self):
        self.assertEqual(ar.recover_drafts('1.00', 0, 45, ['0.000'] * 15), (3, 'no-acceptance'))
        self.assertEqual(ar.recover_drafts('nan', 0, 0, []), (0, 'no-drafts'))
        self.assertEqual(ar.recover_drafts('7.17', 37, 90, '0.833 0.833 0.833 0.833 0.667 0.500 0.500 0.500 0.500 '
                                           '0.167 0.000 0.000 0.000 0.000 0.000'.split()), (6, 'exact'))
        # A zero-acceptance interval now weighs in: the pooled mean is not biased up by dropping it.
        line = ('SpecDecoding metrics: Mean acceptance length: %s, Accepted throughput: 0.00 tokens/s, Drafted '
                'throughput: 0.00 tokens/s, Accepted: %d tokens, Drafted: %d tokens, Per-position acceptance rate: '
                '%s, Avg Draft acceptance rate: 0.0%%')
        text = '\n'.join([line % ('7.17', 37, 90, ', '.join('0.833 0.833 0.833 0.833 0.667 0.500 0.500 0.500 0.500 '
                                                          '0.167 0.000 0.000 0.000 0.000 0.000'.split())),
                          line % ('1.00', 0, 90, ', '.join(['0.000'] * 15))])
        totals = ar.spec_totals(ar.spec_decoding(text))
        self.assertEqual((totals['drafts'], totals['accepted'], totals['drafted']), (12, 37, 180))
        self.assertEqual(totals['mean_acceptance_length'], round(1 + 37 / 12, 3))

    def test_garbage_and_empty_logs_do_not_raise(self):
        empty = ar.report('', [None, {}])
        self.assertEqual(empty['sources']['phases_records'], 0)
        self.assertIsNone(empty['vllm'])
        self.assertTrue(empty['summary_line'].startswith('[ACCEPT] '))
        broken = ar.report('x {"stage": "fast_serving_phases", "blocks": [ {broken\n', [])
        self.assertEqual(broken['sources']['phases_unparsed'], 1)
        self.assertIsNone(ar.decode_rates([None])['window'])


if __name__ == '__main__':
    unittest.main()
