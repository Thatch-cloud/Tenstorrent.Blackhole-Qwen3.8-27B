"""Production telemetry and the Tier 2 comparator (prod_telemetry, prod_telemetry_compare). Stdlib only, no device.

The first class reads the producers' own constants and sources, so a renamed log line breaks CI here and not silently in production.
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

import levern_policy
import prod_telemetry as pt
import prod_telemetry_compare as cmp
import w2ln_timing_compare as w2ln

HERE = Path(__file__).resolve().parent
PREFIX = 'acme_serving_'    # a stand-in for the platform's metric prefix: the tooling takes it as an argument and ships no platform name
LOG_GLOB_NAME = 'engine_*.log'


def loguru(moment, text, prefix='(EngineCore pid=66) '):
    return '%s%s | INFO     | serving_worker_hook:_execute:329 - %s' % (prefix, moment.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], text)


def execute(moment, total, new, cached):
    return loguru(moment, '[PHASE] execute total=%d new=%d cached=%d spec=%d finished=[] preempted=[]' % (total, new, cached, 4 if cached else 0))


def packed(moment, request, segment, position):
    return loguru(moment, '[PACKED] request=%s segment=%d position=%d prefix=4 emitted=4 predictions=[1144, 4087, 1156, 579, 328, 814, 20139, 79091] cap=16'
                  % (request, segment, position))


def stats(running, waiting):
    return ('(APIServer pid=1) INFO 10-04 17:04:44 [loggers.py:273] Engine 000: Avg prompt throughput: 1055.3 tokens/s, Avg generation throughput: '
            '0.3 tokens/s, Running: %d reqs, Waiting: %d reqs, GPU KV cache usage: 0.8%%, Prefix cache hit rate: 0.0%%' % (running, waiting))


def access(status, path='/v1/chat/completions'):
    return '(APIServer pid=1) INFO:     172.17.0.1:40978 - "POST %s HTTP/1.1" %d OK' % (path, status)


T0 = datetime(2026, 10, 9, 12, 0, 0)


def decode_rounds(count, live=8, start=T0, round_s=0.25, position=33000, prefill_between=None):
    """Log lines of `count` decode rounds, each with one audit line per live user (a prefill step after round `prefill_between`, if asked)."""
    lines, moment = [], start
    for index in range(count):
        lines.append(execute(moment, live * 8, 0, live))
        for user in range(live):
            lines.append(packed(moment + timedelta(milliseconds=2), 'cmpl-%d-%d' % (index, user), user, position + index + user))
        moment += timedelta(seconds=round_s)
        if prefill_between is not None and index == prefill_between:
            lines.append(execute(moment, 2048, 1, 0))
            moment += timedelta(seconds=0.4)
    lines.append(execute(moment, live * 8, 0, live))
    return lines


class ProducerContractTests(unittest.TestCase):
    """The lines the rules read still have the text the readers expect (formatted by the producers' own constants where they are importable)."""

    def test_the_lever_n_step_line_is_read_back(self):
        text = levern_policy.STEP_LINE.format(7, 'prefill', 3, 'cmpl-abc', 0, 2048, 2048, 4096, 0, 'plan', 'decode', '12.5', '250', 1)
        final = levern_policy.STEP_LINE.format(8, 'prefill', 3, 'cmpl-abc', 2048, 2048, 4096, 4096, 1, 'plan', 'decode', '12.5', '250', 1)
        extractor = pt.Extractor()
        self.assertEqual(extractor.feed(loguru(T0, text)), [])
        records = extractor.feed(loguru(T0 + timedelta(seconds=30), final))
        self.assertEqual([kind for kind, _ in records], ['request'])
        record = records[0][1]
        self.assertEqual((record['prompt'], record['steps'], record['first_to_final_s']), (4096, 2, 30.0))
        self.assertEqual(record['req'], pt.request_hash('cmpl-abc'))

    def test_the_merged_route_step_line_is_read_back_too(self):
        text = levern_policy.STEP_LINE_MERGED.format(9, 'prefill', 2, 'cmpl-m', 0, 2048, 2048, 2049, 0, 'plan', '-', '-', '0', 0, 0.5)
        extractor = pt.Extractor()
        extractor.feed(loguru(T0, text))
        self.assertIn('cmpl-m', extractor.requests)

    def test_the_quarantine_kill_and_refused_lines_are_read_back(self):
        extractor = pt.Extractor()
        quarantine = extractor.feed(loguru(T0, levern_policy.QUARANTINE_LINE.format('cmpl-q', 'state lost')))
        self.assertEqual(quarantine[0][1]['code'], 'levern_quarantine')
        self.assertEqual(quarantine[0][1]['req'], pt.request_hash('cmpl-q'))
        kill = extractor.feed(loguru(T0, levern_policy.KILL_LINE.format('/models/.qwen-c2/levern.off')))
        self.assertEqual(kill[0][1]['code'], 'levern_kill')
        refused = extractor.feed(loguru(T0, levern_policy.REFUSED_LINE.format('cap computed from Q=0')))
        self.assertEqual(refused[0][1]['code'], 'levern_refused')
        self.assertEqual(extractor.counts['levern_quarantine'], 1)

    def test_the_installed_lines_count(self):
        extractor = pt.Extractor()
        extractor.feed(loguru(T0, levern_policy.INSTALLED_LINE.format('TTScheduler', 'x', 'y', 'z', 1, 2)))
        extractor.feed(loguru(T0, levern_policy.ROUTE_INSTALLED_LINE.format('ok')))
        self.assertEqual((extractor.counts['levern_installed'], extractor.counts['levern_route']), (1, 1))

    def test_the_kill_switch_paths_match_the_producers(self):
        self.assertEqual(levern_policy.OFF_PATH, '/models/.qwen-c2/levern.off')
        registry = (HERE / 'qwen_prefix_registry.py').read_text(encoding='utf-8')
        self.assertIn("KILL_SWITCH_PATH = '/models/.qwen-c2/prefix-reuse.off'", registry)
        wrapper = (HERE / 'prod_telemetry_rig_check.sh').read_text(encoding='utf-8')
        self.assertIn('/models/.qwen-c2', wrapper)
        for name in ('prod_telemetry_rig_check.sh', 'prod_telemetry_rig_capture.sh'):   # the log glob is the caller's, never a default of the tool
            self.assertIn('--log-glob is required', (HERE / name).read_text(encoding='utf-8'))

    def test_the_engine_line_formats_are_still_the_ones_the_reader_parses(self):
        hook = (HERE / 'serving_worker_hook.py').read_text(encoding='utf-8')
        self.assertIn("'[PHASE] execute total={} new={} cached={} spec={} finished={} preempted={}'", hook)
        step = (HERE / 'serving_packed_step.py').read_text(encoding='utf-8')
        self.assertIn("'[PACKED] request={request} segment={segment} position={position} '", step)
        self.assertIn("'prefix={prefix} emitted={emitted} predictions={predictions}'", step)
        verifier = (HERE / 'packed_verifier.py').read_text(encoding='utf-8')
        self.assertIn("'[PACKED-PHASE] round=%d users=%d bind_ms=%.2f input_ms=%.2f trace_ms=%.2f '", verifier)
        scheduler = (HERE / 'serving_lifecycle.py').read_text(encoding='utf-8')
        self.assertIn('[PINDIAG] request quarantined', scheduler)

    def test_the_prefix_counters_the_rules_read_exist(self):
        import qwen_prefix_metrics as metrics
        for key in ('attempts', 'admissions', 'grants', 'grant_tokens', 'session_denied', 'killed_denied'):
            self.assertIn(key, metrics.COUNTERS)
        self.assertEqual(metrics.metric_name('grants')[0], 'qwen_prefix_grants')
        rows = metrics.metric_rows([], 0, 0)
        self.assertIn('qwen_prefix_kill_switch_file', [row[1] for row in rows])
        for name in ('qwen_prefix_attempts_total', 'qwen_prefix_admissions_total', 'qwen_prefix_grants_total', 'qwen_prefix_grant_tokens_total'):
            self.assertIn(name.replace('_total', ''), [metrics.metric_name(k)[0] for k in ('attempts', 'admissions', 'grants', 'grant_tokens')])
        self.assertTrue(set(pt.RAW_REQUIRED) >= {'qwen_prefix_attempts_total', 'qwen_prefix_grants_total'})


class ExtractorTests(unittest.TestCase):
    def rounds(self, lines):
        extractor = pt.Extractor()
        out = []
        for line in lines:
            out += [record for kind, record in extractor.feed(line) if kind == 'round']
        return out, extractor

    def test_eight_live_rounds_are_timed_and_placed(self):
        out, extractor = self.rounds(decode_rounds(10))
        self.assertEqual(len(out), 10)
        self.assertTrue(all(r['live'] == 8 and r['packed'] and abs(r['ms'] - 250.0) < 1.0 for r in out))
        self.assertAlmostEqual(out[0]['mean_pos'], 33000 + 3.5, places=1)
        self.assertEqual(extractor.counts['rounds_packed'], 10)

    def test_a_round_followed_by_a_prefill_is_not_a_round_and_the_next_one_is_flagged(self):
        out, _ = self.rounds(decode_rounds(6, prefill_between=2))
        self.assertEqual(len(out), 5)                       # the round before the prefill has no decode step after it
        self.assertEqual(sum(1 for r in out if r['after_prefill']), 1)   # the first round after the prefill is flagged, the rest are clean

    def test_the_step_after_a_prefill_is_flagged_after_prefill(self):
        lines = [execute(T0, 64, 0, 8), execute(T0 + timedelta(seconds=0.25), 2048, 1, 0), execute(T0 + timedelta(seconds=0.65), 64, 0, 8),
                 execute(T0 + timedelta(seconds=0.9), 64, 0, 8)]
        out, _ = self.rounds(lines)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]['after_prefill'])

    def test_a_long_gap_is_idle_not_a_round(self):
        lines = [execute(T0, 64, 0, 8), execute(T0 + timedelta(seconds=60), 64, 0, 8)]
        out, extractor = self.rounds(lines)
        self.assertEqual(out, [])
        self.assertEqual(extractor.counts['gaps'], 1)

    def test_a_round_without_a_full_audit_has_no_mean_position(self):
        lines = [execute(T0, 64, 0, 8), packed(T0, 'a', 0, 100), execute(T0 + timedelta(seconds=0.25), 64, 0, 8)]
        out, _ = self.rounds(lines)
        self.assertFalse(out[0]['packed'])
        self.assertIsNone(out[0]['mean_pos'])

    def test_events_carry_the_log_day_and_a_hashed_id(self):
        extractor = pt.Extractor()
        extractor.feed(execute(T0, 64, 0, 8))
        events = extractor.feed(access(503))
        self.assertEqual(events[0][1]['code'], 'http5xx')
        self.assertEqual(events[0][1]['day'], '2026-10-09')
        self.assertNotIn('172.17', json.dumps(events))

    def test_no_event_for_a_200(self):
        self.assertEqual(pt.Extractor().feed(access(200)), [])

    def kept(self, lines):
        out, _ = self.rounds(lines)
        return [(r['mean_pos'], round(r['ms'])) for r in out if r['packed'] and not r['after_prefill'] and r['live'] == 8]

    def test_the_rounds_kept_are_the_ones_the_w2ln_reader_keeps(self):
        """decode, decode, a Lever N chunk step, decode x3: the round after the chunk is dropped by both readers."""
        step = levern_policy.STEP_LINE.format(1, 'prefill', 3, 'cmpl-x', 0, 2048, 2048, 4096, 0, 'plan', 'decode', '12.5', '250', 1)
        lines, moment = [], T0
        for index in range(6):
            lines.append(execute(moment, 64, 0, 8))
            for user in range(8):
                lines.append(packed(moment + timedelta(milliseconds=2), 'cmpl-%d-%d' % (index, user), user, 40000 + 100 * index + user))
            if index == 2:
                lines.append(loguru(moment + timedelta(milliseconds=5), step))
            moment += timedelta(milliseconds=300 if index == 3 else 250)
        lines.append(execute(moment, 64, 0, 8))
        theirs, _ = w2ln.timed_rounds('\n'.join(lines))
        expected = [(round(r['mean_position'], 1), round(r['seconds'] * 1000)) for r in theirs]
        self.assertEqual(len(expected), 4)
        self.assertEqual(self.kept(lines), expected)

    def test_an_engine_prefill_step_also_drops_the_round_after_it(self):
        out, _ = self.rounds(decode_rounds(6, prefill_between=2))
        self.assertEqual(sum(1 for r in out if not r['after_prefill']), 4)

    def test_events_hold_no_raw_ids_or_free_text(self):
        extractor = pt.Extractor()
        events = extractor.feed(loguru(T0, "[PINDIAG] request quarantined: 'chatcmpl-SECRETID-123' ends as FINISHED_ABORTED, engine kept (3 decoding): "
                                       'prompt text the user wrote'))
        events += extractor.feed(loguru(T0, levern_policy.QUARANTINE_LINE.format('chatcmpl-SECRETID-456', 'the user prompt says hello')))
        text = json.dumps(events)
        for banned in ('SECRETID', 'user wrote', 'user prompt', 'hello'):
            self.assertNotIn(banned, text)
        self.assertEqual([record['req'] for _, record in events], [pt.request_hash('chatcmpl-SECRETID-123'), pt.request_hash('chatcmpl-SECRETID-456')])
        self.assertIn('decoding=3', events[0][1]['detail'])

    def test_scrub_hashes_quoted_and_bare_ids_and_addresses(self):
        text = pt.scrub("boom on 'chatcmpl-abcdef123' and cmpl-9z8y7x6w5v from 10.1.2.3:80 ids=[1, 2, 3]")
        for banned in ('abcdef123', '9z8y7x6w5v', '10.1.2.3', '[1, 2, 3]'):
            self.assertNotIn(banned, text)

    def test_an_event_before_any_timestamp_takes_the_utc_day_not_unknown(self):
        events = pt.Extractor().feed(access(503))
        self.assertRegex(events[0][1]['day'], r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$')

    def test_installed_and_follow_events_carry_the_log_time_and_the_image(self):
        extractor = pt.Extractor()
        events = extractor.feed(loguru(T0, levern_policy.INSTALLED_LINE.format('TTScheduler', 1, 2, 3, 4, 5)))
        self.assertEqual((events[0][1]['code'], events[0][1]['day']), ('levern_installed', '2026-10-09'))
        self.assertTrue(events[0][1]['t'].startswith('12:00:00'))
        marker = extractor.feed(pt.FOLLOW_MARKER + ' /x/y/engine_7.log image=sha256:0123456789ab container=abc123')
        self.assertEqual(marker[0][1]['code'], 'follow')
        self.assertIn('engine_7.log image=sha256:0123456789ab', marker[0][1]['detail'])
        self.assertNotIn('/x/y', marker[0][1]['detail'])


class SanitiseTests(unittest.TestCase):
    def test_token_ids_and_addresses_are_removed(self):
        line = packed(T0, 'cmpl-1', 0, 5) + ' via 172.17.0.1:40978'
        clean = pt.sanitise_line(line)
        self.assertNotIn('20139', clean)
        self.assertNotIn('172.17', clean)
        self.assertIn('predictions=<8 ids dropped>', clean)
        self.assertIn('position=5', clean)

    def test_any_integer_list_is_reduced(self):
        self.assertEqual(pt.sanitise_line('x tokens=[1, 2, 3] y'), 'x tokens=<3 ids dropped> y')
        self.assertEqual(pt.sanitise_line('x group=[7] y'), 'x group=[7] y')
        self.assertEqual(pt.sanitise_line('x predictions=[5] y'), 'x predictions=<1 ids dropped> y')

    def test_unknown_lines_are_dropped(self):
        lines = ['random user text here', packed(T0, 'a', 0, 1), 'prompt: tell me a secret', execute(T0, 64, 0, 8)]
        out = list(pt.sanitise_stream(lines))
        self.assertEqual(len(out), 2)
        self.assertTrue(all('secret' not in line for line in out))

    def test_records_reach_the_disk_within_the_status_interval_not_only_at_the_end(self):
        directory = tempfile.mkdtemp()
        try:
            seen = []

            def lines():
                for index, line in enumerate(decode_rounds(5)):
                    if index == 30:
                        seen.append(sum(os.path.getsize(os.path.join(directory, n)) for n in os.listdir(directory) if n.startswith('rounds-')))
                    yield line

            clock = iter(range(0, 10 ** 6, 40))
            with redirect_stdout(io.StringIO()):
                pt.run(lines(), directory, clock=lambda: float(next(clock)))
            self.assertGreater(seen[0], 0)
        finally:
            shutil.rmtree(directory)

    def test_every_round_carries_the_host_load(self):
        directory = tempfile.mkdtemp()
        try:
            clock = iter(range(1000, 100000))
            with redirect_stdout(io.StringIO()):
                pt.run(decode_rounds(5), directory, clock=lambda: float(next(clock)), load_fn=lambda: 2.5, load_every=1.0)
            records = [json.loads(line) for name in os.listdir(directory) if name.startswith('rounds-')
                       for line in Path(directory, name).read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records), 5)
            self.assertTrue(all(r['load1'] == 2.5 for r in records))
            self.assertLess(len(json.dumps(records[0], sort_keys=True, separators=(',', ':'))), 200)   # the per-round record size the runbook quotes
        finally:
            shutil.rmtree(directory)

    def test_capture_files_hold_no_ids_prompts_or_raw_request_ids(self):
        directory = tempfile.mkdtemp()
        try:
            lines = decode_rounds(5) + [levern_policy.QUARANTINE_LINE.format('cmpl-0-3', 'gone at 172.17.0.1'),
                                        "[PINDIAG] request quarantined: 'chatcmpl-SECRETID-123' ends as FINISHED_ABORTED, engine kept (2 decoding): the user text"]
            with redirect_stdout(io.StringIO()):
                pt.run(lines, directory, clock=lambda: 1000.0)
            text = ''.join(Path(directory, name).read_text(encoding='utf-8') for name in os.listdir(directory))
            self.assertIn('"k":"round"', text)
            self.assertNotIn('SECRETID', text)
            self.assertNotIn('the user text', text)
            self.assertFalse([name for name in os.listdir(directory) if 'unknown' in name])
            self.assertNotIn('predictions', text)
            self.assertNotIn('20139', text)
            self.assertNotIn('cmpl-0-3', text)
            self.assertNotIn('172.17', text)
        finally:
            shutil.rmtree(directory)


class WatcherTests(unittest.TestCase):
    def watcher(self, **kwargs):
        return pt.Watcher(start=0.0, **kwargs)

    def live(self, w, count, until):
        """The metrics poll reading `count` live requests every 10 s up to `until`."""
        for at in range(0, int(until) + 1, 10):
            w.set_live(count, float(at))

    def test_a_hang_fires_once_after_300_s_with_a_request_live(self):
        w = self.watcher()
        w.feed(execute(T0, 64, 0, 2), 11.0)
        self.live(w, 2, 312)
        self.assertEqual(w.tick(300.0), [])
        fired = w.tick(312.0)
        self.assertEqual([t['code'] for t in fired], ['T0-HANG'])
        self.assertEqual(fired[0]['live'], 2)
        self.assertEqual(w.tick(313.0), [])
        w.feed(execute(T0, 64, 0, 2), 500.0)
        self.live(w, 2, 900)
        self.assertEqual([t['code'] for t in w.tick(900.0)], ['T0-HANG'])

    def test_no_hang_when_nothing_is_running(self):
        w = self.watcher()
        self.live(w, 0, 5000)
        self.assertEqual(w.tick(5000.0), [])

    def test_the_hang_clock_starts_with_the_watcher_before_any_execute_line(self):
        w = self.watcher()
        self.live(w, 1, 310)
        self.assertEqual([t['code'] for t in w.tick(310.0)], ['T0-HANG'])

    def test_the_last_stats_line_decides_between_idle_and_hung(self):
        """vLLM prints a final stats line (Running: 0 when the work is done) and then nothing, so a silent engine is idle or hung by that last line."""
        idle = self.watcher()
        idle.feed(execute(T0, 64, 0, 1), 10.0)
        idle.feed(stats(1, 0), 11.0)
        idle.feed(stats(0, 0), 21.0)
        self.assertEqual(idle.tick(5000.0), [])
        hung = self.watcher()
        hung.feed(execute(T0, 64, 0, 1), 10.0)
        hung.feed(stats(1, 0), 11.0)
        self.assertEqual([t['code'] for t in hung.tick(5000.0)], ['T0-HANG'])

    def test_a_follow_marker_forgets_the_old_stats_line(self):
        w = self.watcher()
        w.feed(stats(4, 0), 5.0)
        w.feed(pt.FOLLOW_MARKER + ' /x/engine_2.log', 100.0)
        self.assertEqual(w.tick(350.0), [])
        self.assertEqual(w.tick(5000.0), [])

    def test_a_fresh_metrics_zero_beats_a_stale_busy_stats_line(self):
        w = self.watcher()
        w.feed(stats(3, 0), 5.0)
        w.set_live(0, 300.0)
        self.assertEqual(w.tick(320.0), [])

    def test_the_stats_line_counts_when_there_is_no_metrics_gauge(self):
        w = self.watcher(hang_s=10.0)
        w.feed(stats(2, 0), 20.0)
        self.assertEqual([t['code'] for t in w.tick(25.0)], ['T0-HANG'])

    def test_the_api_going_quiet_for_two_minutes_after_answering_is_tier_0(self):
        w = self.watcher()
        w.set_api(True, 10.0)
        w.set_api(False, 20.0)
        w.set_api(False, 30.0)
        self.assertEqual(w.tick(100.0), [])
        fired = w.tick(145.0)
        self.assertEqual([t['code'] for t in fired], ['T0-API-UNREACHABLE'])
        self.assertEqual(w.tick(200.0), [])
        w.set_api(True, 210.0)
        w.set_api(False, 220.0)
        self.assertEqual([t['code'] for t in w.tick(345.0)], ['T0-API-UNREACHABLE'])

    def test_an_api_that_never_answered_does_not_alarm(self):
        w = self.watcher()
        w.set_api(False, 0.0)
        self.assertEqual(w.tick(1000.0), [])

    def test_a_metrics_reading_that_ages_out_is_no_reading(self):
        w = self.watcher()
        w.set_live(5, 0.0)
        self.assertEqual(w.tick(400.0), [])

    def test_live_requests_from_a_metrics_text(self):
        raw = ('# HELP x\nvllm:num_requests_running{engine="0",model_name="m"} 3.0\nvllm:num_requests_waiting{engine="0",model_name="m"} 2.0\n'
               'vllm:prompt_tokens_total{engine="0"} 99\n')
        self.assertEqual(pt.parse_live(raw), 5)
        platform = PREFIX + 'running_requests{a="b"} 1\n' + PREFIX + 'queue_depth{a="b"} 4\n'
        self.assertEqual(pt.parse_live(platform, PREFIX), 5)
        self.assertIsNone(pt.parse_live(platform))      # no prefix given: the platform's names mean nothing
        self.assertIsNone(pt.parse_live('vllm:prompt_tokens_total 1\n'))
        self.assertEqual(pt.parse_metrics(raw + 'vllm:generation_tokens_total{engine="0"} 77\n'), (5, 77.0))
        self.assertEqual(pt.parse_metrics(platform + PREFIX + 'generation_tokens_total 9\n', PREFIX), (5, 9.0))

    def test_three_5xx_in_an_hour_fire_once_and_two_do_not(self):
        w = self.watcher()
        self.assertEqual(w.feed(access(500), 100.0) + w.feed(access(502), 200.0), [])
        fired = w.feed(access(503), 300.0)
        self.assertEqual([t['code'] for t in fired], ['T0-SERVER-ERRORS'])
        self.assertEqual(w.feed(access(500), 400.0), [])
        w.tick(300.0 + 3700.0 + 400.0)
        self.assertTrue(w.errors_armed)

    def test_5xx_older_than_an_hour_do_not_count(self):
        w = self.watcher()
        w.feed(access(500), 0.0)
        w.feed(access(500), 10.0)
        self.assertEqual(w.feed(access(500), 3700.0), [])

    def test_4xx_and_non_api_paths_do_not_count(self):
        w = self.watcher()
        for index in range(5):
            self.assertEqual(w.feed(access(404), 10.0 * index), [])
            self.assertEqual(w.feed(access(500, '/health'), 10.0 * index), [])

    def test_a_quarantine_line_fires_tier_1(self):
        fired = self.watcher().feed(loguru(T0, levern_policy.QUARANTINE_LINE.format('cmpl-z', 'lost')), 5.0)
        self.assertEqual([t['code'] for t in fired], ['T1-LEVERN-QUARANTINE'])
        self.assertIn('levern.off', fired[0]['action'])

    def test_an_engine_death_line_fires_tier_0(self):
        fired = self.watcher().feed('(EngineCore pid=66) ERROR vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue', 5.0)
        self.assertEqual([t['code'] for t in fired], ['T0-ENGINE-DEATH'])

    def step(self, req, n, start, tokens, prompt, final, at):
        end = start + tokens
        return loguru(at, levern_policy.STEP_LINE.format(n, 'prefill', 3, req, start, tokens, end, prompt, final, 'plan', 'decode', '12.5', '250', 1))

    def test_a_long_prompt_finished_after_240_s_fires_and_a_short_one_does_not(self):
        w = self.watcher()
        w.feed(self.step('long', 1, 0, 2048, 200000, 0, T0), 1.0)
        fired = w.feed(self.step('long', 2, 2048, 2048, 200000, 1, T0 + timedelta(seconds=250)), 251.0)
        self.assertEqual([t['code'] for t in fired], ['T1-LEVERN-LONG-TTFT'])
        self.assertEqual(fired[0]['state'], 'finished')
        s = self.watcher()
        s.feed(self.step('short', 1, 0, 2048, 20000, 0, T0), 1.0)
        self.assertEqual(s.feed(self.step('short', 2, 2048, 2048, 20000, 1, T0 + timedelta(seconds=300)), 301.0), [])

    def keep_stepping(self, w, until, req='long', prompt=200000):
        """A Lever N prefill that logs a chunk step every 10 s from t=1 to `until` (wall seconds)."""
        for index, at in enumerate(range(1, int(until) + 1, 10)):
            w.feed(self.step(req, index + 1, 2048 * index, 2048, prompt, 0, T0 + timedelta(seconds=at)), float(at))

    def test_a_long_prompt_still_running_after_240_s_fires_from_the_clock(self):
        w = self.watcher()
        self.keep_stepping(w, 245)
        self.assertEqual(w.tick(200.0), [])
        fired = w.tick(250.0)
        self.assertEqual([t['code'] for t in fired], ['T1-LEVERN-LONG-TTFT'])
        self.assertEqual(fired[0]['state'], 'running')
        self.assertEqual(w.tick(251.0), [])

    def test_an_aborted_long_prompt_never_fires(self):
        """One prefill step of a 200k prompt, then the client cancels: no final step ever comes, and the old reader fired at 260 s."""
        for ending in ('abort', 'finished', 'quarantine', 'silence'):
            w = self.watcher()
            w.feed(self.step('cmpl-long-request', 1, 0, 2048, 200000, 0, T0), 1.0)
            if ending == 'abort':
                w.feed('(APIServer pid=1) INFO 10-09 12:00:20 [async_llm.py:300] Aborted request chatcmpl-long-request.', 20.0)
            elif ending == 'finished':
                w.feed(loguru(T0, "[PHASE] execute total=64 new=0 cached=8 spec=0 finished=['cmpl-long-request-0-ab12'] preempted=[]"), 20.0)
            elif ending == 'quarantine':
                w.feed(loguru(T0, levern_policy.QUARANTINE_LINE.format('cmpl-long-request', 'x')), 20.0)
            self.assertEqual(w.tick(260.0), [], ending)
            self.assertEqual(w.tick(2000.0), [], ending)

    def test_a_prefill_that_stopped_stepping_is_forgotten_not_fired(self):
        w = self.watcher()
        self.keep_stepping(w, 100)
        self.assertEqual(w.tick(150.0), [])          # its last step was 59 s ago: still running, and only 149 s old
        self.assertEqual(w.tick(300.0), [])          # 209 s since the last step line: not running
        self.assertEqual(w.tick(5000.0), [])
        self.assertEqual(len(w.extractor.inflight()), 0)

    def test_the_ids_in_a_finished_list_are_matched_both_ways(self):
        extractor = pt.Extractor()
        extractor.feed(self.step('cmpl-aaaa-bbbb', 1, 0, 2048, 200000, 0, T0))
        self.assertEqual(len(extractor.inflight()), 1)
        extractor.feed(loguru(T0, "[PHASE] execute total=64 new=0 cached=8 spec=0 finished=['cmpl-aaaa-bbbb-0-ff00'] preempted=[]"))
        self.assertEqual(len(extractor.inflight()), 0)

    def test_tokens_flowing_with_a_silent_phase_log_is_a_stalled_log_not_a_hang(self):
        w = self.watcher()
        w.feed(execute(T0, 64, 0, 2), 10.0)
        for at in range(0, 400, 10):
            w.set_live(2, float(at))
            w.set_tokens(1000.0 + at * 5, float(at))
        fired = w.tick(320.0)
        self.assertEqual([t['code'] for t in fired], ['T0-LOG-STALLED'])
        self.assertNotIn('rollback', fired[0]['action'].lower().replace('not a rollback', ''))
        self.assertEqual(w.tick(330.0), [])
        # the tokens stop too: now it is a hang
        for at in range(400, 520, 10):
            w.set_live(2, float(at))
            w.set_tokens(5000.0, float(at))
        self.assertEqual([t['code'] for t in w.tick(520.0)], ['T0-HANG'])

    def test_a_stats_line_with_generation_throughput_also_means_the_engine_works(self):
        w = self.watcher()
        w.feed(execute(T0, 64, 0, 1), 10.0)
        w.feed(stats(1, 0).replace('generation throughput: 0.3', 'generation throughput: 40.0'), 305.0)
        self.assertEqual([t['code'] for t in w.tick(320.0)], ['T0-LOG-STALLED'])

    def test_tokens_that_do_not_move_leave_the_hang_a_hang(self):
        w = self.watcher()
        w.feed(execute(T0, 64, 0, 2), 10.0)
        for at in range(0, 320, 10):
            w.set_live(2, float(at))
            w.set_tokens(777.0, float(at))
        self.assertEqual([t['code'] for t in w.tick(320.0)], ['T0-HANG'])

    def test_a_long_prompt_under_240_s_does_not_fire(self):
        w = self.watcher()
        w.feed(self.step('long', 1, 0, 2048, 200000, 0, T0), 1.0)
        self.assertEqual(w.feed(self.step('long', 2, 2048, 2048, 200000, 1, T0 + timedelta(seconds=100)), 101.0), [])

    def test_a_silent_pipe_still_raises_the_hang_from_the_idle_tick(self):
        import time

        def stream():
            yield stats(1, 0)
            time.sleep(0.6)

        out = io.StringIO()
        with redirect_stdout(out):
            pt.run(stream(), None, idle_tick=0.05, hang_s=0.2)
        self.assertIn('T0-HANG', out.getvalue())   # the stats line was fresh when the clock ran out (no metrics poll: the fallback)

    def serve_metrics(self, text):
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = text.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def test_the_metrics_poll_makes_a_silent_busy_engine_a_hang_and_a_silent_idle_one_not(self):
        import time

        def silent():
            yield execute(T0, 64, 0, 2)
            time.sleep(0.8)

        for text, expected in (('vllm:num_requests_running 2\nvllm:num_requests_waiting 0\n', True),
                               ('vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n', False)):
            server = self.serve_metrics(text)
            try:
                out = io.StringIO()
                with redirect_stdout(out):
                    pt.run(silent(), None, idle_tick=0.05, hang_s=0.3, metrics_url='http://127.0.0.1:%d/metrics' % server.server_port, metrics_every=0.05)
                self.assertEqual('T0-HANG' in out.getvalue(), expected, out.getvalue())
            finally:
                server.shutdown()
                server.server_close()

    def test_run_prints_triggers_and_writes_them(self):
        directory = tempfile.mkdtemp()
        try:
            clock = iter(range(1000, 100000))
            lines = [stats(1, 0), execute(T0, 64, 0, 1), access(500), access(500), access(500)]
            out = io.StringIO()
            with redirect_stdout(out):
                pt.run(lines, directory, clock=lambda: float(next(clock)))
            self.assertIn('T0-SERVER-ERRORS', out.getvalue())
            self.assertIn('T0-SERVER-ERRORS', Path(directory, 'triggers.jsonl').read_text(encoding='utf-8'))
            self.assertTrue(Path(directory, 'capture-state.json').is_file())
        finally:
            shutil.rmtree(directory)


class CheckTests(unittest.TestCase):
    def good(self):
        return decode_rounds(20) + [stats(8, 0)]

    def test_a_complete_window_passes(self):
        report = pt.check_log(self.good())
        self.assertEqual((report['verdict'], report['absent']), ('PASS', []))

    def test_a_window_without_audit_lines_fails(self):
        lines = [line for line in self.good() if '[PACKED]' not in line]
        report = pt.check_log(lines)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertIn('packed_audit', report['absent'])
        self.assertIn('rounds_packed', report['absent'])

    def test_the_stats_line_is_informational(self):
        report = pt.check_log([line for line in self.good() if 'Running:' not in line], live_gauges=True)    # the gauges carry the hang rule's 'live' reading
        self.assertEqual(report['absent'], [])
        row = [r for r in report['signals'] if r['signal'] == 'stats_line'][0]
        self.assertEqual((row['present'], row['required']), (False, False))

    def test_an_idle_window_is_inconclusive(self):
        report = pt.check_log([stats(0, 0)])
        self.assertEqual(report['verdict'], 'IDLE')

    def test_lever_n_lines_are_required_only_when_expected(self):
        self.assertEqual(pt.check_log(self.good())['verdict'], 'PASS')
        report = pt.check_log(self.good(), expect_levern=True)
        self.assertEqual(sorted(report['absent']), ['levern_installed', 'levern_steps'])
        lines = self.good() + [loguru(T0, levern_policy.INSTALLED_LINE.format('TTScheduler', 1, 2, 3, 4, 5)),
                               loguru(T0, levern_policy.STEP_LINE.format(1, 'prefill', 1, 'r', 0, 2048, 2048, 4096, 0, 'plan', '-', '-', '0', 0))]
        self.assertEqual(pt.check_log(lines, expect_levern=True)['verdict'], 'PASS')

    def test_the_report_carries_counts_and_no_log_text(self):
        text = json.dumps(pt.check_log(self.good()))
        self.assertNotIn('cmpl-', text)
        self.assertNotIn('20139', text)

    def test_metrics_names_raw_and_platform(self):
        raw = '\n'.join('%s{a="b"} 1' % name for name in pt.RAW_REQUIRED + pt.RAW_INFO)
        self.assertEqual(pt.check_metrics(raw)['absent'], [])
        self.assertEqual(pt.check_metrics(raw)['family'], 'raw')
        self.assertTrue(pt.check_metrics(raw)['live_gauges'])
        required, info = pt.platform_names(PREFIX)
        platform = '\n'.join('%s 1' % name for name in required[:-1])
        result = pt.check_metrics(platform, 'auto', PREFIX)
        self.assertEqual(result['family'], 'platform')
        self.assertEqual(result['absent'], [required[-1]])
        self.assertEqual(result['info_absent'], list(info))
        self.assertFalse(result['live_gauges'])
        self.assertEqual(pt.check_metrics(platform)['family'], 'raw')     # without the prefix the platform's names are not recognised

    def test_the_gauges_are_informational_and_a_stats_line_satisfies_the_hang_source(self):
        """The live TT scrape shape (no running/waiting gauges, no kill-switch gauge) on a healthy engine is a PASS."""
        directory = tempfile.mkdtemp()
        try:
            scrape = Path(directory, 'scrape.prom')
            names = [name for name in pt.RAW_REQUIRED] + ['vllm:prefix_cache_hits_total', 'qwen_prefix_registry_entries']
            scrape.write_text(''.join('# TYPE %s counter\n%s{engine="0"} 1\n' % (n, n) for n in names), encoding='utf-8')
            log = Path(directory, 'engine.log')
            log.write_text('\n'.join(self.good()) + '\n', encoding='utf-8')
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(pt.main(['check', str(log), '--metrics', str(scrape)]), 0, out.getvalue())
            report = json.loads(out.getvalue())
            self.assertEqual(report['verdict'], 'PASS')
            self.assertEqual(report['metrics']['absent'], [])
            self.assertIn('vllm:num_requests_running', report['metrics']['info_absent'])
            source = [r for r in report['signals'] if r['signal'] == 'hang_live_source'][0]
            self.assertEqual(source['source'], 'stats-line')
        finally:
            shutil.rmtree(directory)

    def test_no_gauge_and_no_stats_line_is_a_failure_of_the_hang_source(self):
        report = pt.check_log([line for line in self.good() if 'Running:' not in line], live_gauges=False)
        self.assertEqual((report['verdict'], report['absent']), ('FAIL', ['hang_live_source']))
        self.assertEqual(pt.check_log([line for line in self.good() if 'Running:' not in line], live_gauges=True)['verdict'], 'PASS')

    def test_a_boot_line_counted_over_the_whole_file_satisfies_expect_levern(self):
        tail = self.good() + [loguru(T0, levern_policy.STEP_LINE.format(1, 'prefill', 1, 'r', 0, 2048, 2048, 4096, 0, 'plan', '-', '-', '0', 0))]
        self.assertEqual(pt.check_log(tail, expect_levern=True)['absent'], ['levern_installed'])
        self.assertEqual(pt.check_log(tail, expect_levern=True, boot={'levern_installed': 1})['verdict'], 'PASS')

    def test_an_unreachable_metrics_url_is_exit_4_not_a_traceback(self):
        directory = tempfile.mkdtemp()
        try:
            log = Path(directory, 'engine.log')
            log.write_text('\n'.join(self.good()) + '\n', encoding='utf-8')
            err = io.StringIO()
            from contextlib import redirect_stderr
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                self.assertEqual(pt.main(['check', str(log), '--metrics-url', 'http://127.0.0.1:1/metrics']), 4)
            self.assertIn('did not answer', err.getvalue())
        finally:
            shutil.rmtree(directory)

    def test_the_cli_exit_codes(self):
        directory = tempfile.mkdtemp()
        try:
            good = Path(directory, 'good.log')
            good.write_text('\n'.join(self.good()) + '\n', encoding='utf-8')
            idle = Path(directory, 'idle.log')
            idle.write_text(stats(0, 0) + '\n', encoding='utf-8')
            bad = Path(directory, 'bad.log')
            bad.write_text('\n'.join(line for line in self.good() if '[PACKED] request=' not in line) + '\n', encoding='utf-8')
            for path, code in ((good, 0), (bad, 2), (idle, 3)):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(pt.main(['check', str(path)]), code, path.name)
        finally:
            shutil.rmtree(directory)


def round_records(count, ms, position=33000.0, live=8, start=T0, spread=0.0, step_s=0.4, load=1.0):
    """Capture-style round dicts, one every `step_s` seconds, positions cycling over 3 buckets, round times ms (+/- spread, alternating), host load `load`."""
    out = []
    for index in range(count):
        out.append(dict(at=start + timedelta(seconds=index * step_s), live=live, ms=ms + (spread if index % 2 else -spread),
                        mean_pos=position + (index % 3) * 1100.0, load=load))
    return out


def redirect_stderr_to_null():
    from contextlib import redirect_stderr
    return redirect_stderr(io.StringIO())


class ComparatorTests(unittest.TestCase):
    cutover = datetime(2026, 10, 9, 0, 0, 0)

    def two_sides(self, base_ms, cand_ms, count=300, base_days=3, **kwargs):
        base = []
        for day in range(base_days):
            base += round_records(count // base_days, base_ms, start=self.cutover - timedelta(days=day + 1) + timedelta(hours=1), **kwargs)
        cand = round_records(count, cand_ms, start=self.cutover + timedelta(hours=1), **kwargs)
        return base, cand

    def test_equal_sides_are_inside(self):
        base, cand = self.two_sides(250.0, 250.0)
        result = cmp.read_stratum(base, cand, '32k', 8)
        self.assertEqual(result['verdict'], 'INSIDE')
        self.assertEqual(result['delta_pct'], 0.0)

    def test_ten_percent_worse_is_a_rollback(self):
        base, cand = self.two_sides(250.0, 275.0)
        result = cmp.read_stratum(base, cand, '32k', 8)
        self.assertEqual(result['verdict'], 'WORSE')
        self.assertAlmostEqual(result['delta_pct'], 10.0, places=0)

    def test_two_percent_worse_is_inside_the_three_percent_margin(self):
        base, cand = self.two_sides(250.0, 255.0)
        self.assertEqual(cmp.read_stratum(base, cand, '32k', 8)['verdict'], 'INSIDE')

    def test_faster_is_better(self):
        base, cand = self.two_sides(250.0, 225.0)
        self.assertEqual(cmp.read_stratum(base, cand, '32k', 8)['verdict'], 'BETTER')

    def test_under_100_matched_rounds_is_void_never_a_rollback(self):
        base, cand = self.two_sides(250.0, 400.0, count=150)
        result = cmp.read_stratum(base, cand, '32k', 8, min_matched=100)
        self.assertEqual(cmp.read_stratum(base[:60], cand[:60], '32k', 8)['verdict'], 'VOID')
        self.assertIn(result['verdict'], ('WORSE', 'VOID'))
        void = cmp.read_stratum(base[:30], cand, '32k', 8)
        self.assertEqual(void['verdict'], 'VOID')
        self.assertIn('under the pre-registered 100', void['reason'])

    def test_the_other_window_and_live_counts_do_not_leak_in(self):
        base, cand = self.two_sides(250.0, 400.0)
        self.assertEqual(cmp.read_stratum(base, cand, '128k', 8)['verdict'], 'VOID')
        self.assertEqual(cmp.read_stratum(base, cand, '32k', 7)['verdict'], 'VOID')

    def test_the_drift_floor_widens_the_threshold(self):
        base = []
        for day, shift in enumerate((0.0, 20.0, 10.0)):
            base += round_records(200, 250.0 + shift, start=self.cutover - timedelta(days=day + 1) + timedelta(hours=1))
        cand = round_records(300, 250.0 + 10.0 + 12.5, start=self.cutover + timedelta(hours=1))   # +5 % over the pooled median, inside the floor
        result = cmp.read_stratum(base, cand, '32k', 8)
        self.assertGreater(result['floor_pct'], 5.0)
        self.assertEqual(result['verdict'], 'INSIDE')
        self.assertTrue(result['floor_basis'].startswith('3 baseline days'))
        worse = round_records(300, 250.0 + 10.0 + 60.0, start=self.cutover + timedelta(hours=1))
        self.assertEqual(cmp.read_stratum(base, worse, '32k', 8)['verdict'], 'WORSE')

    def test_one_baseline_day_applies_the_margin_alone(self):
        base, cand = self.two_sides(250.0, 250.0, base_days=1)
        self.assertIn('margin alone', cmp.read_stratum(base, cand, '32k', 8)['floor_basis'])

    def test_rounds_run_hot_are_dropped_and_do_not_make_a_false_rollback(self):
        """A busy CI week: the candidate's slow rounds ran at load 9 against a baseline median of 1; they are over the limit (1 + 3) and not read."""
        base, cand = self.two_sides(250.0, 250.0)
        hot = round_records(300, 400.0, start=self.cutover + timedelta(hours=3), load=9.0)
        result = cmp.read_stratum(base, cand + hot, '32k', 8)
        self.assertEqual(result['verdict'], 'INSIDE')
        self.assertEqual(result['rounds_over_load_limit']['candidate'], 300)
        self.assertEqual(result['load_limit'], 4.0)

    def test_two_sides_under_different_loads_are_void_never_a_rollback(self):
        base, _ = self.two_sides(250.0, 250.0)
        cand = round_records(300, 400.0, start=self.cutover + timedelta(hours=1), load=3.5)     # under the limit (4) but 2.5 above the baseline median
        result = cmp.read_stratum(base, cand, '32k', 8)
        self.assertEqual(result['verdict'], 'VOID')
        self.assertIn('differ by 2.50', result['reason'])
        near = round_records(300, 400.0, start=self.cutover + timedelta(hours=1), load=2.9)     # a gap of 1.9: matched, and 60 % slower: a rollback
        self.assertEqual(cmp.read_stratum(base, near, '32k', 8)['verdict'], 'WORSE')

    def test_rounds_without_a_load_sample_are_void(self):
        base, cand = self.two_sides(250.0, 400.0)
        for item in cand:
            item['load'] = None
        result = cmp.read_stratum(base, cand, '32k', 8)
        self.assertEqual(result['verdict'], 'VOID')
        self.assertIn('no host load samples on the candidate side', result['reason'])

    def test_the_cross_boot_floor_widens_the_threshold_for_its_window_only(self):
        base, cand = self.two_sides(250.0, 270.0)      # +8 %
        self.assertEqual(cmp.read_stratum(base, cand, '32k', 8)['verdict'], 'WORSE')
        result = cmp.read_stratum(base, cand, '32k', 8, floor_w={'32k': 0.10})
        self.assertEqual((result['verdict'], result['threshold_pct'], result['floor_w_pct']), ('INSIDE', 10.0, 10.0))
        self.assertEqual(cmp.parse_floor_w('32k=0.04,128k=0.06'), {'32k': 0.04, '128k': 0.06})
        self.assertIn('not given', cmp.read_stratum(base, cand, '32k', 8)['floor_w_basis'])

    def test_no_request_file_is_ttft_unmeasured_not_a_pass(self):
        base, cand = self.two_sides(250.0, 250.0)
        report = cmp.judge(base + cand, None, self.cutover, '72h', windows=('32k',), live_counts=(8,))
        self.assertEqual(report['rounds_verdict'], 'PASS')
        self.assertEqual(report['ttft']['verdict'], 'VOID')
        self.assertIn('TTFT not read', report['ttft']['reason'])
        self.assertEqual(report['verdict'], 'UNMEASURED')       # one half measured nothing: not named a pass
        self.assertEqual(report['unmeasured'], ['ttft'])

    def test_the_prompt_length_is_the_total_with_the_cached_tokens(self):
        directory = tempfile.mkdtemp()
        try:
            path = Path(directory, 'r.csv')
            path.write_text('occurred_at,prompt_tokens,cached_tokens,ttft_ms,status\n2026-10-07 21:00:01+00,2000,198000,2500,200\n'
                            '2026-10-07 21:00:02+00,2000,0,900,200\n', encoding='utf-8')
            self.assertEqual([r['prompt'] for r in cmp.load_requests([str(path)])], [200000, 2000])
            path.write_text('occurred_at,total_prompt_tokens,prompt_tokens,cached_tokens,ttft_ms\n2026-10-07 21:00:01+00,5000,1,1,2500\n', encoding='utf-8')
            self.assertEqual(cmp.load_requests([str(path)])[0]['prompt'], 5000)
        finally:
            shutil.rmtree(directory)

    def test_the_cutover_is_required_and_the_first_event_default_is_opt_in(self):
        directory = tempfile.mkdtemp()
        try:
            self.assertIsNone(cmp.first_event([directory], 'levern_installed'))
            Path(directory, 'events-2026-10-09.jsonl').write_text(
                json.dumps(dict(k='event', code='follow', day='2026-10-09', t='11:00:00.000')) + '\n' +
                json.dumps(dict(k='event', code='levern_installed', day='2026-10-09', t='12:30:00.0000')) + '\n' +
                json.dumps(dict(k='event', code='levern_installed', day='2026-10-10', t='01:00:00.0000')) + '\n', encoding='utf-8')
            self.assertEqual(cmp.first_event([directory], 'levern_installed'), datetime(2026, 10, 9, 12, 30, 0))
            with redirect_stdout(io.StringIO()), redirect_stderr_to_null():
                self.assertEqual(cmp.main(['--dir', directory, '--stage', '72h']), 64)                 # no --cutover: refused, not guessed
                self.assertEqual(cmp.main(['--dir', directory, '--stage', '72h', '--cutover-from-first-event']), 3)   # a replay: found, no rounds
        finally:
            shutil.rmtree(directory)
        empty = tempfile.mkdtemp()
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr_to_null():
                self.assertEqual(cmp.main(['--dir', empty, '--cutover-from-first-event']), 64)
        finally:
            shutil.rmtree(empty)

    def follow_event(self, at, image):
        return json.dumps(dict(k='event', code='follow', day=at.strftime('%Y-%m-%d'), t=at.strftime('%H:%M:%S.000'),
                               detail='engine_1.log image=%s container=cafe' % image))

    def test_an_image_on_both_sides_of_the_cutover_refuses_the_comparison(self):
        """A capture started after the cutover boot: its 'baseline' days were captured under the cutover image. The cutover time given is wrong."""
        directory = tempfile.mkdtemp()
        try:
            base, cand = self.two_sides(250.0, 250.0)
            Path(directory, 'events-2026-10-08.jsonl').write_text(self.follow_event(self.cutover - timedelta(days=4), 'sha256:newimage') + '\n', encoding='utf-8')
            timeline = cmp.image_timeline([directory])
            report = cmp.judge(base + cand, None, self.cutover, '72h', windows=('32k',), live_counts=(8,), timeline=timeline)
            self.assertEqual(report['verdict'], 'CUTOVER-INCONSISTENT')
            self.assertEqual(report['images']['images_on_both_sides'], ['sha256:newimage'])
            self.assertIn('BOTH sides', report['reason'])
        finally:
            shutil.rmtree(directory)

    def test_distinct_images_either_side_are_reported_and_the_read_goes_ahead(self):
        directory = tempfile.mkdtemp()
        try:
            base, cand = self.two_sides(250.0, 250.0)
            Path(directory, 'events-2026-10-08.jsonl').write_text(
                self.follow_event(self.cutover - timedelta(days=4), 'sha256:oldimage') + '\n' + self.follow_event(self.cutover, 'sha256:newimage') + '\n',
                encoding='utf-8')
            report = cmp.judge(base + cand, None, self.cutover, '72h', windows=('32k',), live_counts=(8,), timeline=cmp.image_timeline([directory]))
            self.assertEqual(report['images']['baseline_images'], ['sha256:oldimage'])
            self.assertEqual(report['images']['candidate_images'], ['sha256:newimage'])
            self.assertEqual(report['rounds_verdict'], 'PASS')
        finally:
            shutil.rmtree(directory)

    def test_a_rounds_half_with_every_stratum_void_is_unmeasured_whatever_ttft_says(self):
        base, cand = self.two_sides(250.0, 250.0, count=30)         # too few rounds: every stratum is VOID
        requests = [dict(at=self.cutover - timedelta(hours=1 + i % 50), prompt=1000, ttft_s=1.0) for i in range(600)]
        requests += [dict(at=self.cutover + timedelta(hours=1 + i % 50), prompt=1000, ttft_s=1.0) for i in range(600)]
        report = cmp.judge(base + cand, requests, self.cutover, '72h', windows=('32k',), live_counts=(8,))
        self.assertEqual(report['rounds_verdict'], 'INSUFFICIENT')
        self.assertEqual(report['ttft']['verdict'], 'INSIDE')
        self.assertEqual((report['verdict'], report['unmeasured']), ('UNMEASURED', ['round time']))
        directory = tempfile.mkdtemp()
        try:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cmp.main(['--dir', directory, '--cutover', '2026-10-09 00:00:00']), 3)    # nothing in either half: INSUFFICIENT
        finally:
            shutil.rmtree(directory)

    def test_a_measured_loss_in_one_half_is_a_rollback_even_when_the_other_is_unmeasured(self):
        base, cand = self.two_sides(250.0, 400.0)
        report = cmp.judge(base + cand, None, self.cutover, '72h', windows=('32k',), live_counts=(8,))
        self.assertEqual(report['verdict'], 'ROLLBACK')

    def test_overall_verdicts(self):
        self.assertEqual(cmp.overall([dict(verdict='INSIDE'), dict(verdict='WORSE')]), 'ROLLBACK')
        self.assertEqual(cmp.overall([dict(verdict='INSIDE'), dict(verdict='BETTER')]), 'PASS')
        self.assertEqual(cmp.overall([dict(verdict='INSIDE'), dict(verdict='VOID')]), 'PASS-PARTIAL')
        self.assertEqual(cmp.overall([dict(verdict='VOID')]), 'INSUFFICIENT')

    def test_judge_reads_the_stage_window_and_the_baseline_window(self):
        base, cand = self.two_sides(250.0, 275.0, count=300)
        late = round_records(300, 900.0, start=self.cutover + timedelta(hours=100))   # after the 72 h stage: not read at 72h
        report = cmp.judge(base + cand + late, None, self.cutover, '72h', windows=('32k',), live_counts=(8,))
        self.assertEqual(report['verdict'], 'ROLLBACK')
        self.assertEqual(report['round_strata'][0]['candidate_rounds'], 300)
        final = cmp.judge(base + cand + late, None, self.cutover, 'final', windows=('32k',), live_counts=(8,))
        self.assertEqual(final['round_strata'][0]['candidate_rounds'], 600)

    def test_judge_with_no_rounds_is_insufficient(self):
        self.assertEqual(cmp.judge([], None, self.cutover, '72h')['verdict'], 'INSUFFICIENT')

    def requests(self, count, ttft, start, prompt=8000, jitter=0.2):
        return [dict(at=start + timedelta(seconds=index * 60), prompt=prompt, ttft_s=ttft * (1.0 + jitter * ((index % 10) / 10.0 - 0.45)))
                for index in range(count)]

    def test_ttft_p90_inside_and_worse(self):
        base = []
        for day in range(3):
            base += self.requests(200, 4.0, self.cutover - timedelta(days=day + 1))
        inside = cmp.read_ttft(base, self.requests(600, 4.0 * 1.15, self.cutover))
        self.assertEqual(inside['verdict'], 'INSIDE')
        worse = cmp.read_ttft(base, self.requests(600, 4.0 * 1.30, self.cutover))
        self.assertEqual(worse['verdict'], 'WORSE')
        self.assertEqual(worse['threshold_pct'], 20.0)

    def test_ttft_ignores_long_prompts_and_needs_500_a_side(self):
        base = self.requests(600, 4.0, self.cutover - timedelta(days=1)) + self.requests(600, 400.0, self.cutover - timedelta(days=1), prompt=200000)
        cand = self.requests(600, 4.0, self.cutover) + self.requests(600, 900.0, self.cutover, prompt=200000)
        self.assertEqual(cmp.read_ttft(base, cand)['verdict'], 'INSIDE')
        self.assertEqual(cmp.read_ttft(base[:400], cand)['verdict'], 'VOID')

    def test_ttft_day_spread_widens_the_threshold(self):
        base = self.requests(300, 4.0, self.cutover - timedelta(days=1)) + self.requests(300, 6.0, self.cutover - timedelta(days=2))
        result = cmp.read_ttft(base, self.requests(600, 5.0 * 1.5, self.cutover))
        self.assertGreater(result['threshold_pct'], 20.0)
        self.assertAlmostEqual(result['threshold_pct'], 2.0 * result['day_spread_pct'], delta=0.05)

    def test_times_with_zones_and_postgres_style_stamps(self):
        self.assertEqual(cmp.parse_time('2026-10-07 21:00:01.123456+00'), datetime(2026, 10, 7, 21, 0, 1, 123456))
        self.assertEqual(cmp.parse_time('2026-10-07T21:00:01Z'), datetime(2026, 10, 7, 21, 0, 1))
        self.assertEqual(cmp.parse_time('2026-10-08 09:00:01+12:00'), datetime(2026, 10, 7, 21, 0, 1))
        self.assertEqual(cmp.parse_time('2026-10-07 21:00:01.5'), datetime(2026, 10, 7, 21, 0, 1, 500000))

    def test_request_files_csv_and_jsonl(self):
        directory = tempfile.mkdtemp()
        try:
            csv_path = Path(directory, 'r.csv')
            csv_path.write_text('occurred_at,prompt_tokens,ttft_ms,status\n2026-10-07 21:00:01+00,1000,2500,200\n2026-10-07 21:00:02+00,1000,,200\n'
                                '2026-10-07 21:00:03+00,1000,900,500\n', encoding='utf-8')
            rows = cmp.load_requests([str(csv_path)])
            self.assertEqual([(r['prompt'], r['ttft_s']) for r in rows], [(1000, 2.5)])
            json_path = Path(directory, 'r.jsonl')
            json_path.write_text(json.dumps(dict(at='2026-10-07 21:00:01', prompt_tokens=5, ttft_s=1.5)) + '\n', encoding='utf-8')
            self.assertEqual(cmp.load_requests([str(json_path)])[0]['ttft_s'], 1.5)
        finally:
            shutil.rmtree(directory)

    def test_end_to_end_capture_then_judge(self):
        """capture writes the files, the comparator reads them back: a +10 % candidate is a rollback, an equal one passes."""
        directory = tempfile.mkdtemp()
        try:
            def run_side(start, round_s):
                lines = []
                for block in range(4):
                    lines += decode_rounds(120, start=start + timedelta(minutes=block * 5), round_s=round_s, position=30000 + block * 1000)
                with redirect_stdout(io.StringIO()):
                    pt.run(lines, directory, watch=False, clock=lambda: 1.0, load_fn=lambda: 1.0)
            run_side(datetime(2026, 10, 8, 12, 0, 0), 0.250)
            run_side(datetime(2026, 10, 9, 12, 0, 0), 0.275)
            with redirect_stdout(io.StringIO()) as out:
                code = cmp.main(['--dir', directory, '--cutover', '2026-10-09 00:00:00', '--stage', '72h', '--windows', '32k', '--live', '8'])
            report = json.loads(out.getvalue())
            self.assertEqual(code, 1, report)
            self.assertEqual(report['verdict'], 'ROLLBACK')
            self.assertEqual(report['round_strata'][0]['verdict'], 'WORSE')
        finally:
            shutil.rmtree(directory)

    def test_end_to_end_a_busy_candidate_week_is_void_not_a_rollback(self):
        """The same slow candidate captured at a much higher host load: the stratum is void, so no image rollback is called for."""
        directory = tempfile.mkdtemp()
        try:
            def run_side(start, round_s, load):
                lines = []
                for block in range(4):
                    lines += decode_rounds(120, start=start + timedelta(minutes=block * 5), round_s=round_s, position=30000 + block * 1000)
                with redirect_stdout(io.StringIO()):
                    pt.run(lines, directory, watch=False, clock=lambda: 1.0, load_fn=lambda: load)
            run_side(datetime(2026, 10, 8, 12, 0, 0), 0.250, 1.0)
            run_side(datetime(2026, 10, 9, 12, 0, 0), 0.275, 7.0)
            with redirect_stdout(io.StringIO()) as out:
                code = cmp.main(['--dir', directory, '--cutover', '2026-10-09 00:00:00', '--stage', '72h', '--windows', '32k', '--live', '8'])
            report = json.loads(out.getvalue())
            self.assertEqual(code, 3, report)
            self.assertEqual(report['round_strata'][0]['verdict'], 'VOID')
        finally:
            shutil.rmtree(directory)


class DeathRuleTests(unittest.TestCase):
    """The engine-death rule must not fire on a recoverable warning that carries a ttnn error text."""

    def watcher(self, **kwargs):
        return pt.Watcher(start=0.0, **kwargs)

    def prefix_warning(self, error='TT_THROW @ /tt-metal/tt_metal/impl/buffers/buffer.cpp:123: tt::exception'):
        """The prefix patch's own format for a recoverable capture failure (qwen_prefix_model_patch.py: logger.warning(f"[PREFIX] capture not stored req=... pos=...: {error!r}"))."""
        return loguru(T0, "[PREFIX] capture not stored req=cmpl-1 pos=4096: RuntimeError(%r)" % error).replace('| INFO     |', '| WARNING  |')

    def live(self, w, count, until):
        for at in range(0, int(until) + 1, 10):
            w.set_live(count, float(at))

    def test_a_recoverable_prefix_capture_failure_is_not_a_death_even_with_a_request_live(self):
        w = self.watcher()
        self.live(w, 3, 400)
        w.feed(execute(T0, 24, 0, 3), 1.0)
        self.assertEqual(w.feed(self.prefix_warning(), 5.0), [])
        for now in (10.0, 60.0, 120.0, 200.0):
            w.feed(execute(T0 + timedelta(seconds=now), 24, 0, 3), now)
            self.assertEqual(w.tick(now), [])
        self.assertEqual(w.extractor.counts['engine_death'], 0)
        self.assertEqual(w.extractor.counts['native_error'], 0)

    def test_a_prefix_or_pindiag_line_is_never_a_death_whatever_its_level(self):
        for line in (loguru(T0, '[PREFIX] capture skipped: TT_FATAL @ x'), loguru(T0, '[PINDIAG] prefix: capture skipped req=r pos=1: TT_THROW @ y'),
                     '(EngineCore pid=66) WARNING 10-04 17:04:44 [x.py:1] TT_THROW @ somewhere recoverable'):
            self.assertIsNone(pt.death_marker(line), line)

    def test_a_native_error_text_is_a_death_only_when_confirmed_by_silence_with_a_request_live(self):
        w = self.watcher()
        self.live(w, 2, 400)
        w.feed(execute(T0, 16, 0, 2), 1.0)
        self.assertEqual(w.feed('(EngineCore pid=66) ERROR TT_THROW @ /tt-metal/x.cpp:1: device timeout', 5.0), [])    # a suspicion, not a trigger
        self.assertEqual(w.tick(20.0), [])
        fired = w.tick(60.0)                                         # 55 s on, no [PHASE] execute since, a request live
        self.assertEqual([t['code'] for t in fired], ['T0-ENGINE-DEATH'])
        self.assertEqual(fired[0]['detail'], 'TT_THROW')
        self.assertEqual(fired[0]['confirmed'], 'no-step')

    def test_a_native_error_text_followed_by_steps_is_dropped(self):
        w = self.watcher()
        self.live(w, 2, 400)
        w.feed('(EngineCore pid=66) ERROR TT_FATAL @ /tt-metal/x.cpp:1: recovered', 5.0)
        w.feed(execute(T0, 16, 0, 2), 20.0)
        self.assertEqual(w.tick(80.0), [])
        self.assertIsNone(w.native)

    def test_a_native_error_text_with_the_api_gone_is_confirmed_at_once(self):
        w = self.watcher()
        w.set_api(True, 0.0)
        w.feed('(EngineCore pid=66) ERROR TT_THROW @ /tt-metal/x.cpp:1: device', 5.0)
        w.set_api(False, 6.0)
        fired = w.tick(7.0)
        self.assertEqual([(t['code'], t['confirmed']) for t in fired], [('T0-ENGINE-DEATH', 'api-down')])

    def test_a_native_error_on_an_idle_engine_is_not_a_death(self):
        w = self.watcher()
        self.live(w, 0, 400)
        w.feed('(EngineCore pid=66) ERROR TT_THROW @ /tt-metal/x.cpp:1: while idle', 5.0)
        self.assertEqual(w.tick(100.0), [])
        self.assertEqual(w.extractor.counts['native_unconfirmed'], 1)

    def test_engine_level_markers_fire_at_once(self):
        for line in ('(EngineCore pid=66) ERROR vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue',
                     'ERROR Engine core proc EngineCore_0 died unexpectedly, shutting down client.', 'Fatal Python error: Aborted',
                     'Segmentation fault (core dumped)', '(EngineCore pid=66) Segmentation fault', 'Aborted (core dumped)'):
            fired = self.watcher().feed(line, 5.0)
            self.assertEqual([t['code'] for t in fired], ['T0-ENGINE-DEATH'], line)

    def test_a_trigger_and_an_event_carry_the_marker_name_not_the_line(self):
        directory = tempfile.mkdtemp()
        try:
            secret = 'EngineDeadError while serving the user prompt: my secret plan 172.17.0.1'
            with redirect_stdout(io.StringIO()) as out:
                pt.run([secret], directory, clock=lambda: 1000.0)
            text = out.getvalue() + ''.join(Path(directory, name).read_text(encoding='utf-8') for name in os.listdir(directory))
            self.assertIn('EngineDeadError', text)
            self.assertNotIn('secret plan', text)
            self.assertNotIn('172.17', text)
        finally:
            shutil.rmtree(directory)


class ChunkedPrefillTests(unittest.TestCase):
    def test_a_chunk_continuation_step_between_decode_steps_is_not_a_decode_step_on_either_image(self):
        """vLLM's chunk-continuation step logs total=<chunk> new=0 cached=1: it is prefill work, so the rounds beside it are dropped on both sides."""
        chunk = execute(T0 + timedelta(seconds=0.9), 8192, 0, 1)
        ex = pt.Extractor()
        records = []
        for line in decode_rounds(2, live=8) + [chunk] + decode_rounds(2, live=8, start=T0 + timedelta(seconds=1.0)):
            records += ex.feed(line)
        rounds = [r for k, r in records if k == 'round']
        # without the chunk rule the chunk step counts as a decode step and a 0.4 s "round" is formed from the step before it
        self.assertEqual([r['after_prefill'] for r in rounds], [False, False, True, False])
        self.assertEqual(ex.counts['decode'], 6)


    def test_the_decode_step_rows_rule(self):
        ex = pt.Extractor()
        ex.feed(execute(T0, 8 * 16, 0, 8))              # 16 rows a user: the widest draft step is still a decode step
        ex.feed(execute(T0 + timedelta(seconds=0.3), 8 * 17, 0, 8))      # more rows than a draft step carries: a chunk
        self.assertEqual(ex.counts['decode'], 1)


class SanitiseKeepAllTests(unittest.TestCase):
    def test_a_single_id_list_in_vllms_request_log_shape_is_reduced(self):
        self.assertEqual(pt.sanitise_line('prompt_token_ids: [5]'), 'prompt_token_ids=<1 ids dropped>')
        self.assertEqual(pt.sanitise_line("x prompt_token_ids: [1, 2, 3], y"), 'x prompt_token_ids=<3 ids dropped>, y')
        self.assertEqual(pt.sanitise_line('x group=[7] y'), 'x group=[7] y')

    def test_keep_all_drops_every_line_that_can_carry_prompt_text(self):
        lines = ["INFO Received request cmpl-1: prompt: 'my private text', params: SamplingParams(), prompt_token_ids: [5, 6, 7], lora_request: None",
                 'prompt_token_ids: [9]', execute(T0, 64, 0, 8), stats(1, 0), 'plain line']
        out = list(pt.sanitise_stream(lines, keep_all=True))
        self.assertEqual(len(out), 3)
        self.assertTrue(all('private' not in line and '[9]' not in line for line in out))


class NotificationTests(unittest.TestCase):
    """The capture publishes a heartbeat and a counter per trigger code (a node_exporter textfile): the log-only rules must reach the alert rules."""

    def run_capture(self, lines, **kwargs):
        directory = tempfile.mkdtemp()
        prom = os.path.join(directory, 'tt.prom')
        try:
            with redirect_stdout(io.StringIO()):
                pt.run(lines, directory, clock=lambda: 1000.0, prom_file=prom, **kwargs)
            return Path(prom).read_text(encoding='utf-8'), directory
        except Exception:
            shutil.rmtree(directory)
            raise

    def sample(self, text, name, label=''):
        for line in text.splitlines():
            if line.startswith(name + label + ' ') or (not label and line.startswith(name + ' ')):
                return float(line.rsplit(' ', 1)[1])
        raise AssertionError('no %s%s in\n%s' % (name, label, text))

    def test_the_heartbeat_and_a_zero_counter_per_code_exist_from_the_start(self):
        text, directory = self.run_capture(decode_rounds(2))
        try:
            self.assertGreater(self.sample(text, 'tt_prod_telemetry_heartbeat_timestamp_seconds'), 1.0e9)
            for code in pt.TRIGGER_CODES:
                self.assertEqual(self.sample(text, 'tt_prod_telemetry_trigger_total', '{code="%s"}' % code), 0.0)
            self.assertEqual(self.sample(text, 'tt_prod_telemetry_maintenance'), 0.0)
        finally:
            shutil.rmtree(directory)

    def test_a_quarantine_line_moves_its_counter_before_the_file_is_next_written(self):
        text, directory = self.run_capture([loguru(T0, levern_policy.QUARANTINE_LINE.format('cmpl-z', 'lost'))])
        try:
            self.assertEqual(self.sample(text, 'tt_prod_telemetry_trigger_total', '{code="T1-LEVERN-QUARANTINE"}'), 1.0)
            self.assertEqual(self.sample(text, 'tt_prod_telemetry_trigger_total', '{code="T0-ENGINE-DEATH"}'), 0.0)
        finally:
            shutil.rmtree(directory)

    def test_a_maintenance_marker_suppresses_tier_0_but_not_tier_1(self):
        directory = tempfile.mkdtemp()
        try:
            marker = os.path.join(directory, 'maintenance')
            Path(marker).write_text('window\n', encoding='utf-8')
            prom = os.path.join(directory, 'tt.prom')
            lines = ['(EngineCore pid=66) ERROR vllm.v1.engine.exceptions.EngineDeadError: x', loguru(T0, levern_policy.QUARANTINE_LINE.format('cmpl-z', 'lost'))]
            with redirect_stdout(io.StringIO()) as out:
                pt.run(lines, directory, clock=lambda: 1000.0, prom_file=prom, maintenance_file=marker)
            text = Path(prom).read_text(encoding='utf-8')
            self.assertEqual(self.sample(text, 'tt_prod_telemetry_trigger_total', '{code="T0-ENGINE-DEATH"}'), 0.0)
            self.assertEqual(self.sample(text, 'tt_prod_telemetry_trigger_suppressed_total', '{code="T0-ENGINE-DEATH"}'), 1.0)
            self.assertEqual(self.sample(text, 'tt_prod_telemetry_trigger_total', '{code="T1-LEVERN-QUARANTINE"}'), 1.0)
            self.assertEqual(self.sample(text, 'tt_prod_telemetry_maintenance'), 1.0)
            triggers = [json.loads(line) for line in Path(directory, 'triggers.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([t.get('suppressed') for t in triggers], ['maintenance', None])
        finally:
            shutil.rmtree(directory)

    def test_a_forgotten_marker_expires(self):
        directory = tempfile.mkdtemp()
        try:
            marker = os.path.join(directory, 'maintenance')
            Path(marker).write_text('x', encoding='utf-8')
            self.assertTrue(pt.maintenance_active(marker))
            old = os.path.getmtime(marker) - pt.MAINT_MAX_S - 60
            os.utime(marker, (old, old))
            self.assertFalse(pt.maintenance_active(marker))
            self.assertFalse(pt.maintenance_active(os.path.join(directory, 'absent')))
            self.assertFalse(pt.maintenance_active(None))
        finally:
            shutil.rmtree(directory)

    def test_the_poller_follows_the_url_file(self):
        directory = tempfile.mkdtemp()
        try:
            urlfile = os.path.join(directory, 'metrics.url')
            self.assertEqual(pt.current_url('http://127.0.0.1:8000/metrics', urlfile), 'http://127.0.0.1:8000/metrics')
            Path(urlfile).write_text('http://127.0.0.1:8123/metrics\n', encoding='utf-8')
            self.assertEqual(pt.current_url('http://127.0.0.1:8000/metrics', urlfile), 'http://127.0.0.1:8123/metrics')
            Path(urlfile).write_text('', encoding='utf-8')
            self.assertEqual(pt.current_url('http://127.0.0.1:8000/metrics', urlfile), 'http://127.0.0.1:8000/metrics')
            self.assertEqual(pt.current_url('u', None), 'u')
        finally:
            shutil.rmtree(directory)


FAKE_DOCKER = r"""#!/bin/sh
# a stand-in for docker: `exec` runs the command on this machine (the "container" is a directory of log files), `inspect` and `port` answer from files
sub="$1"; shift
d="$FAKE_DIR"
case "$sub" in
  inspect)
    fmt="$2"
    [ -f "$d/gone" ] && { echo "Error: No such object" >&2; exit 1; }
    case "$fmt" in
      *State.Status*) echo "status=running running=true restarts=0 started=x log_driver=none health=${FAKE_HEALTH:-healthy}" ;;
      *.Image*) cat "$d/image" ;;
      *.Id*) cat "$d/cid" ;;
    esac ;;
  port) echo "0.0.0.0:$FAKE_PORT" ;;
  exec)
    shift
    [ -f "$d/gone" ] && exit 1
    case "$1" in
      df) if [ -f "$d/df" ]; then cat "$d/df"; else printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\ntmpfs 524288 1000 523288 1%% /tmp\n'; fi; exit 0 ;;
      tail) echo started >> "$d/tails.log" ;;
    esac
    exec "$@" ;;
  logs) cat "$d/dockerlogs" 2>/dev/null ;;
esac
"""

FAKE_CURL = """#!/bin/sh
echo 200
"""


class WrapperTests(unittest.TestCase):
    """The two shell wrappers, run for real against a fake docker whose container is a directory (the first execution of the real code must not be on production)."""

    @classmethod
    def setUpClass(cls):
        if shutil.which('sh') is None or os.name == 'nt':
            raise unittest.SkipTest('needs a POSIX sh and tail -F (run in CI or WSL)')

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.bin = os.path.join(self.dir, 'bin')
        os.makedirs(self.bin)
        os.makedirs(os.path.join(self.dir, 'tmp'))
        for name, text in (('docker', FAKE_DOCKER), ('curl', FAKE_CURL)):
            path = os.path.join(self.bin, name)
            with open(path, 'w', newline='\n') as handle:
                handle.write(text)
            os.chmod(path, 0o755)
        self.write('image', 'sha256:0123456789ab\n')
        self.write('cid', 'cafe0123456789\n')
        self.server = None
        self.procs = []
        self.set_metrics('vllm:prompt_tokens_total 1\n')

    def tearDown(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name, text):
        with open(os.path.join(self.dir, name), 'w', newline='\n') as handle:
            handle.write(text)

    def set_metrics(self, text):
        import http.server
        import threading
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = text.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def env(self, **extra):
        env = dict(os.environ)
        env.update(PATH=self.bin + os.pathsep + env['PATH'], FAKE_DIR=self.dir, FAKE_PORT=str(self.server.server_port), PROD_TELEMETRY_PYBIN=sys.executable,
                   PROD_TELEMETRY_PY=str(HERE / 'prod_telemetry.py'), PROD_TELEMETRY_RECHECK_S='1', PROD_TELEMETRY_RETRY_S='1')
        env.update(extra)
        return env

    def logfile(self, name, lines):
        path = os.path.join(self.dir, 'tmp', name)
        with open(path, 'a', newline='\n') as handle:
            handle.write('\n'.join(lines) + '\n')
        return path

    @property
    def glob(self):
        return os.path.join(self.dir, 'tmp', LOG_GLOB_NAME)

    def check(self, *args, **extra):
        return subprocess.run(['sh', str(HERE / 'prod_telemetry_rig_check.sh'), 'c1', '--log-glob', self.glob, '--growth-s', '0', *args], env=self.env(**extra),
                              capture_output=True, text=True, timeout=120)

    def good_log(self):
        return decode_rounds(20) + [stats(8, 0)]

    def test_check_passes_on_a_healthy_engine_whose_scrape_has_no_gauges(self):
        self.set_metrics(''.join('%s{engine="0"} 1\n' % name for name in pt.RAW_REQUIRED))
        self.logfile('engine_1.log', self.good_log())
        done = self.check()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn('"verdict": "PASS"', done.stdout)
        self.assertIn('log_fs_total_kb=524288', done.stdout)
        self.assertNotIn('cmpl-', done.stdout)

    def test_check_finds_the_boot_line_that_scrolled_out_of_the_tail(self):
        self.set_metrics(''.join('%s{engine="0"} 1\n' % name for name in pt.RAW_REQUIRED))
        installed = loguru(T0, levern_policy.INSTALLED_LINE.format('TTScheduler', 1, 2, 3, 4, 5))
        step = loguru(T0, levern_policy.STEP_LINE.format(1, 'prefill', 1, 'r', 0, 2048, 2048, 4096, 0, 'plan', '-', '-', '0', 0))
        self.logfile('engine_1.log', [installed] + decode_rounds(60) + [step, stats(8, 0)])
        self.assertEqual(self.check('--expect-levern', '--lines', '40').returncode, 0)
        self.logfile('engine_2.log', decode_rounds(60) + [step, stats(8, 0)])
        os.utime(os.path.join(self.dir, 'tmp', 'engine_2.log'), (time_ahead(), time_ahead()))
        self.assertEqual(self.check('--expect-levern', '--lines', '40').returncode, 2)      # the newest generation has no installed line at all

    def test_check_exit_codes_for_a_down_container_a_dead_metrics_url_a_full_log_filesystem_and_no_glob(self):
        self.set_metrics('x 1\n')
        self.logfile('engine_1.log', self.good_log())
        self.write('gone', '')
        self.assertEqual(self.check().returncode, 4)
        os.remove(os.path.join(self.dir, 'gone'))
        self.assertEqual(self.check(FAKE_PORT='1').returncode, 4)       # /metrics refuses the connection
        self.set_metrics(''.join('%s{engine="0"} 1\n' % name for name in pt.RAW_REQUIRED))
        self.write('df', 'Filesystem 1024-blocks Used Available Capacity Mounted on\ntmpfs 524288 500000 24288 96% /tmp\n')
        done = self.check()
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn('FAIL: the log filesystem is nearly full', done.stdout)
        done = subprocess.run(['sh', str(HERE / 'prod_telemetry_rig_check.sh'), 'c1'], env=self.env(), capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 64)

    def test_check_reads_the_port_docker_published(self):
        self.set_metrics(''.join('%s{engine="0"} 1\n' % name for name in pt.RAW_REQUIRED))
        self.logfile('engine_1.log', self.good_log())
        done = self.check()
        self.assertIn('host port %d' % self.server.server_port, done.stdout)

    def start_capture(self, out, **extra):
        return subprocess.Popen(['sh', str(HERE / 'prod_telemetry_rig_capture.sh'), 'c1', out, '--log-glob', self.glob],
                                env=self.env(**extra), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)

    def wait_for(self, predicate, seconds=20.0):
        import time
        end = time.time() + seconds
        while time.time() < end:
            if predicate():
                return True
            time.sleep(0.1)
        return False

    def tails_started(self):
        path = os.path.join(self.dir, 'tails.log')
        return len(open(path).read().split()) if os.path.exists(path) else 0

    def test_capture_follows_across_a_container_replacement_and_a_log_rollover_and_stops_clean(self):
        import signal
        import time
        out = os.path.join(self.dir, 'out')
        first = self.logfile('engine_1.log', [stats(0, 0)])
        proc = self.start_capture(out)
        self.procs.append(proc)
        self.assertTrue(self.wait_for(lambda: self.tails_started() >= 1), 'the follower never attached')
        time.sleep(0.5)
        self.logfile('engine_1.log', decode_rounds(5, start=T0))
        # a second capture into the same directory is refused
        second = subprocess.run(['sh', str(HERE / 'prod_telemetry_rig_capture.sh'), 'c1', out, '--log-glob', self.glob], env=self.env(),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(second.returncode, 75, second.stderr)
        time.sleep(1.0)
        # the container is replaced: the old tail dies, the name is gone for a moment, a new container (new image, new log file) comes up
        self.write('gone', '')
        subprocess.run(['pkill', '-f', 'tail -n 0 -F ' + first], check=False)
        self.write('image', 'sha256:fedcba987654' + chr(10))
        self.write('cid', 'beef0123456789' + chr(10))
        second_file = self.logfile('engine_2.log', [stats(0, 0)])
        os.utime(second_file, (time_ahead(), time_ahead()))
        time.sleep(2.5)
        os.remove(os.path.join(self.dir, 'gone'))
        self.assertTrue(self.wait_for(lambda: self.tails_started() >= 2), 'no re-attach after the container was replaced')
        time.sleep(0.5)
        self.logfile('engine_2.log', decode_rounds(5, start=T0 + timedelta(hours=1)))
        # the engine rolls to a new log generation inside the same container
        third_file = self.logfile('engine_3.log', [stats(0, 0)])
        os.utime(third_file, (time_ahead() + 5, time_ahead() + 5))
        self.assertTrue(self.wait_for(lambda: self.tails_started() >= 3), 'no re-attach after the log rolled over')
        time.sleep(0.5)
        self.logfile('engine_3.log', decode_rounds(5, start=T0 + timedelta(hours=2)))
        time.sleep(1.5)
        pid = int(open(os.path.join(out, 'capture.pid')).read())
        os.kill(pid, signal.SIGTERM)                                  # the wrapper alone, not its group: it must stop what it started
        proc.wait(timeout=30)
        self.assertEqual(proc.returncode, 143)
        self.assertFalse(os.path.exists(os.path.join(out, 'capture.pid')))
        rounds = [json.loads(line) for name in sorted(os.listdir(out)) if name.startswith('rounds-') for line in open(os.path.join(out, name)).read().splitlines()]
        events = [json.loads(line) for name in sorted(os.listdir(out)) if name.startswith('events-') for line in open(os.path.join(out, name)).read().splitlines()]
        self.assertEqual(len(rounds), 15)
        self.assertTrue(all(isinstance(r['load1'], float) for r in rounds))
        follows = [e['detail'] for e in events if e['code'] == 'follow']
        self.assertEqual(len(follows), 3, follows)
        self.assertEqual(sum('engine_1.log image=sha256:0123456789ab' in f for f in follows), 1, follows)
        self.assertEqual(sum('engine_2.log image=sha256:fedcba987654' in f for f in follows), 1, follows)
        self.assertEqual(sum('engine_3.log image=sha256:fedcba987654' in f for f in follows), 1, follows)
        text = ''.join(open(os.path.join(out, name)).read() for name in os.listdir(out) if name.endswith('.jsonl'))
        self.assertNotIn('20139', text)
        self.assertNotIn('cmpl-', text)
        self.assertEqual(subprocess.run(['pgrep', '-f', 'tail -n 0 -F ' + self.dir], capture_output=True).returncode, 1, 'a tail was left running')

    def test_capture_follows_the_published_port_when_a_redeploy_moves_it(self):
        import time
        out = os.path.join(self.dir, 'out')
        self.logfile('engine_1.log', [stats(0, 0)])
        proc = self.start_capture(out)
        self.procs.append(proc)
        url_file = os.path.join(out, 'metrics.url')
        self.assertTrue(self.wait_for(lambda: os.path.exists(url_file)), 'the follower never published the metrics URL')
        self.assertIn(':%d/' % self.server.server_port, open(url_file).read())
        proc.terminate()
        proc.wait(timeout=30)
        # --port fixes the port: the follower publishes nothing
        fixed = os.path.join(self.dir, 'out2')
        proc = subprocess.Popen(['sh', str(HERE / 'prod_telemetry_rig_capture.sh'), 'c1', fixed, '--log-glob', self.glob, '--port', '9'], env=self.env(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        self.procs.append(proc)
        self.assertTrue(self.wait_for(lambda: self.tails_started() >= 2), 'the second capture never attached')
        time.sleep(0.5)
        self.assertFalse(os.path.exists(os.path.join(fixed, 'metrics.url')))

    def test_capture_writes_the_heartbeat_file_it_is_given(self):
        import signal
        out = os.path.join(self.dir, 'out')
        prom = os.path.join(self.dir, 'tt.prom')
        self.logfile('engine_1.log', [stats(0, 0)])
        proc = subprocess.Popen(['sh', str(HERE / 'prod_telemetry_rig_capture.sh'), 'c1', out, '--log-glob', self.glob, '--prom-file', prom], env=self.env(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        self.procs.append(proc)
        self.assertTrue(self.wait_for(lambda: os.path.exists(prom)), 'no heartbeat file')
        self.assertIn('tt_prod_telemetry_heartbeat_timestamp_seconds', open(prom).read())
        os.kill(int(open(os.path.join(out, 'capture.pid')).read()), signal.SIGTERM)
        proc.wait(timeout=30)

    def test_capture_needs_a_log_glob(self):
        done = subprocess.run(['sh', str(HERE / 'prod_telemetry_rig_capture.sh'), 'c1', os.path.join(self.dir, 'out')], env=self.env(), capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(done.returncode, 64)


def time_ahead():
    import time
    return time.time() + 100


class HygieneTests(unittest.TestCase):
    """The tooling is committed to a public repo: no host, address, serial, platform path or credential in it."""

    FILES = ('prod_telemetry.py', 'prod_telemetry_compare.py', 'prod_telemetry_rig_check.sh', 'prod_telemetry_rig_capture.sh',
             'test_prod_telemetry.py')
    # built from parts so that this file does not contain what it forbids
    BANNED = tuple(first + second for first, second in (('black', 'hole-'), ('that', 'ch'), ('zo', 't.'), ('spa', 'rk-'), ('pl', 'ink'), ('pass', 'word'),
                                                        ('tok', 'en='), ('node', '-agent'), ('node', ' agent'), ('p1', '50a'), ('non', 'e` logging')))

    def test_no_addresses_serials_or_internal_names(self):
        for name in self.FILES:
            text = (HERE / name).read_text(encoding='utf-8')
            for found in re.findall(r'\b[0-9]{1,3}(?:\.[0-9]{1,3}){3}\b', text):
                self.assertIn(found, ('127.0.0.1', '172.17.0.1', '10.1.2.3', '0.0.0.0'), '%s: %s' % (name, found))
            for banned in self.BANNED:
                self.assertNotIn(banned, text.lower(), '%s: %s' % (name, banned))

    def test_the_files_are_lf_only(self):
        for name in self.FILES:
            self.assertNotIn(b'\r', (HERE / name).read_bytes(), name)

    def test_the_script_is_posix_sh_and_parses(self):
        if shutil.which('sh') is None:
            self.skipTest('no sh')
        for name in ('prod_telemetry_rig_check.sh', 'prod_telemetry_rig_capture.sh'):
            subprocess.check_call(['sh', '-n', str(HERE / name)])


if __name__ == '__main__':
    unittest.main()
