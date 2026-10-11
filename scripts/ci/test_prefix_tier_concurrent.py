"""The tier-concurrent plans (tp4/prefix-tiers; docs/prefix-store-hygiene.md "Concurrent restore and spill"): seven seats decode while an eighth session returns from host RAM
and while a long fresh prompt displaces blocks into the tier. The measurement (prefix_concurrent: windows, round times, the longest wait, the budget, the text comparison),
its judge (c2_prefix_gate.concurrent_findings), the plan wiring (the control profile with no tier, the audited and the timed arm), the scenario end to end on the tier fake of
test_prefix_tier_gate (real sleeps of tens of milliseconds, a stall injected into the restore), and the offline comparator (tier_concurrent_compare).
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import make_prefix_tier_profiles as tier_twin  # noqa: E402
import prefix_concurrent as conc  # noqa: E402
import prefix_replay as replay  # noqa: E402
import test_c2_prefix_gate as gate_tests  # noqa: E402
import test_prefix_replay as fakes  # noqa: E402
import test_prefix_tier_gate as tiers  # noqa: E402
import tier_concurrent_compare as compare  # noqa: E402

FAKE = tiers.FAKE
REAL_PROFILES = HERE / 'qwen_c2_profiles.json'


# -- synthetic records ---------------------------------------------------------------------------------------

def steady(start=0.0, end=400.0, step=0.1, gaps=()):
    """Chunk offsets from a send at `start`: one every `step` seconds until `end`, with the intervals of `gaps` ((at, length) pairs: no chunk from `at` for `length`)."""
    out, now = [], start
    while now <= end:
        if not any(at < now < at + length for at, length in gaps):
            out.append(round(now - start, 4))
        now = round(now + step, 6)
    return out


def seat(index, offsets, sent=0.0, tokens=2000):
    return dict(role='seat', case=conc.SEAT_CASE, conv='%s-%d' % (conc.SEAT_CASE, index), sent_s=sent, chunk_times=offsets, token_ids=list(range(tokens)), ok=False,
                intended_abort=True, aborted='closed on signal', tag='pfx-a-%04d-seat' % index)


def request(case, tag, sent, ttft, wall, ids=(1, 2, 3), role='hit', q=0, window=None, **extra):
    return dict(role=role, case=case, conv=case, sent_s=sent, ttft_s=ttft, wall_s=wall, token_ids=list(ids), finish='stop', ok=True, tag=tag, markers=dict(q=q),
                log_window=window, **extra)


def arm_records(seats=3, gaps=(), tokens=2000, ids=(1, 2, 3)):
    records = [seat(index, steady(gaps=gaps), tokens=tokens) for index in range(seats)]
    records.append(request(conc.RETURN_CASE, 'ret', 100.0, 3.0, 8.0, ids=ids, q=29000, window=[0, 5]))
    records.append(request(conc.SPILL_CASE, 'spill', 150.0, 20.0, 20.5, ids=(9,), role='flood', window=[10, 20]))
    records.append(request(conc.COLD_CASE, 'cold', 200.0, 25.0, 30.0, ids=ids, role='cold', window=[30, 40]))
    return records


def event(seats=3, **values):
    base = dict(seats=seats, control_s=60.0, settle_s=10.0, built_ok=True, seats_running=True, return_tag='ret', return_ok=True, return_ttft_s=3.0, spill_tag='spill',
                spill_ok=True, cold_tag='cold', cold_ttft_s=25.0, return_tokens=29000, spill_tokens=32000, seat_budget=20000)
    base.update(values)
    return base


def scanned(restore=True, spill=True):
    return dict(tier_restores=[dict(tag='ret', blocks=450, bytes=3.5e9, ms=210.0, index=2)] if restore else [],
                tier_spills=[dict(index=15, bytes=1e9, ms=90.0), dict(index=16, bytes=1e9, ms=95.0)] if spill else [])


def arm(timed=True, tier=True, **env):
    return dict(arm='tier-concurrent-timed' if timed else 'tier-concurrent', scenario='tier_concurrent_timed' if timed else 'tier_concurrent', served_env=dict(env),
                tier=dict(gib=32.0, audit=not timed, preconverted=True) if tier else None)


# -- (1) the measurement ---------------------------------------------------------------------------------------

class MeasurementTests(unittest.TestCase):
    def test_percentile_interpolates_and_has_no_value_for_nothing(self):
        self.assertIsNone(conc.percentile([], 0.5))
        self.assertEqual(conc.percentile([7], 0.99), 7)
        self.assertEqual(conc.percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(conc.percentile(list(range(101)), 0.99), 99.0)

    def test_a_window_counts_the_seats_that_decoded_across_all_of_it_and_the_longest_wait_of_any(self):
        seats = [seat(0, steady(gaps=[(40.0, 5.0)])), seat(1, steady()), seat(2, steady(end=50.0))]       # seat 2 finished its budget at 50 s
        stats = conc.window_stats(seats, 30.0, 60.0)
        self.assertEqual(stats['seats'], 2, 'the seat that ended inside the window is not a full-house reading')
        self.assertEqual(stats['max_gap_s'], 5.0, 'the 5 s wait (40.0 to 45.0), whichever seat saw it')
        self.assertEqual(stats['p50_s'], 0.1)
        self.assertEqual(stats['p99_s'], 0.1, 'one 5 s wait among a thousand rounds is the max, not the p99')
        self.assertGreater(stats['mean_s'], 0.1)
        self.assertGreater(stats['intervals'], 500)
        late = conc.window_stats(seats, 44.0, 55.0)
        self.assertEqual(late['seats'], 2)
        self.assertEqual(late['max_gap_s'], 5.0, 'a wait that began before the window and ended inside it is the window\'s')
        self.assertEqual(conc.window_stats([], 0, 10)['seats'], 0)

    def test_a_wait_over_the_window_edge_is_in_the_longest_wait_but_not_in_the_percentiles(self):
        stats = conc.window_stats([seat(0, steady(gaps=[(58.0, 4.0)]))], 50.0, 60.0)
        self.assertEqual(stats['max_gap_s'], 4.0)
        self.assertEqual(stats['p99_s'], 0.1, 'the interval that crosses the edge is not a round of the window')

    def test_the_windows_are_read_from_the_records(self):
        windows = conc.windows_from_records(arm_records(), 60.0)
        self.assertEqual(windows, dict(control=[40.0, 100.0], restore=[100.0, 103.0], restore_decode=[103.0, 108.0], spill=[150.0, 170.0], cold=[200.0, 225.0]))
        self.assertEqual(conc.windows_from_records([], 60.0), {})
        only = [r for r in arm_records() if r['case'] != conc.SPILL_CASE and r['case'] != conc.COLD_CASE]
        self.assertEqual(sorted(conc.windows_from_records(only)), ['control', 'restore', 'restore_decode'])
        failed = [dict(r, ttft_s=None) if r['case'] == conc.RETURN_CASE else r for r in arm_records()]
        self.assertEqual(sorted(conc.windows_from_records(failed)), ['cold', 'control', 'spill'], 'a request that streamed nothing has no restore window')

    def test_the_control_window_defaults_and_follows_the_event(self):
        self.assertEqual(conc.windows_from_records(arm_records())['control'], [40.0, 100.0])
        self.assertEqual(conc.windows_from_records(arm_records(), 20)['control'], [80.0, 100.0])
        self.assertEqual(conc.windows_from_records([dict(r, sent_s=5.0) if r['case'] == conc.RETURN_CASE else r for r in arm_records()], 60)['control'][0], 0.0)

    def test_the_regression_is_a_percentage_of_the_control_windows_p99(self):
        stats = dict(control=dict(p99_s=0.2), restore=dict(p99_s=0.25), spill=dict(p99_s=0.2), cold=dict(p99_s=None))
        self.assertEqual(conc.regression(stats, 'restore'), 25.0)
        self.assertEqual(conc.regression(stats, 'spill'), 0.0)
        self.assertIsNone(conc.regression(stats, 'cold'))
        self.assertIsNone(conc.regression(stats, 'absent'))
        self.assertIsNone(conc.regression(dict(restore=dict(p99_s=1.0)), 'restore'), 'no control window, no percentage')

    def test_the_budget_is_the_decode_gap_floor_of_the_profile_else_eight_seconds(self):
        self.assertEqual(conc.gap_budget({}), 8.0)
        self.assertEqual(conc.gap_budget({'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S': '12'}), 12.0)
        self.assertEqual(conc.gap_budget({'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S': '0'}), 8.0, 'a profile with the floor off is still held to 8 s')
        self.assertEqual(conc.gap_budget({'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S': 'x'}), 8.0)
        self.assertEqual(conc.gap_budget(None), 8.0)
        stats = dict(control=dict(max_gap_s=0.3), restore=dict(max_gap_s=8.0), spill=dict(max_gap_s=8.01), cold=dict(max_gap_s=None))
        problems = conc.budget_problems(stats)
        self.assertEqual(len(problems), 1, 'exactly 8.0 is within the floor; over it is not')
        self.assertIn('window spill', problems[0])
        self.assertEqual(conc.budget_problems(stats, 0.2).__len__(), 3)

    def test_render_prints_every_window_with_its_regression_against_the_control_window(self):
        stats = conc.stats_of(arm_records(), 60.0)
        text = '\n'.join(conc.render('timed', stats))
        for name in conc.WINDOWS:
            self.assertIn('window %s:' % name, text)
        self.assertIn('round p99 vs control', text)

    def test_seats_are_compared_over_the_tokens_both_arms_streamed(self):
        a = [seat(0, [], tokens=3000), seat(1, [], tokens=3000), seat(2, [], tokens=3000)]
        b = [seat(0, [], tokens=2500), seat(1, [], tokens=3000), seat(2, [], tokens=3000)]
        b[1]['token_ids'][1500] = -1
        b[2]['token_ids'] = b[2]['token_ids'][:400]
        rows = dict((row['conv'], row) for row in conc.compare_seats(a, b))
        self.assertEqual(rows['%s-0' % conc.SEAT_CASE]['verdict'], 'IDENTICAL')
        self.assertEqual(rows['%s-0' % conc.SEAT_CASE]['tokens'], 2500, 'the common length of arms aborted at different counts')
        self.assertEqual((rows['%s-1' % conc.SEAT_CASE]['verdict'], rows['%s-1' % conc.SEAT_CASE]['at']), ('DIVERGED', 1500))
        self.assertEqual(rows['%s-2' % conc.SEAT_CASE]['verdict'], 'SHORT', 'under the minimum nothing is shown')
        self.assertEqual(conc.compare_seats(a, []), [])

    def test_the_loader_reads_a_file_a_directory_and_an_arm_directory(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(tmp), True)
        text = replay.records_jsonl(arm_records())
        (tmp / 'direct').mkdir()
        (tmp / 'direct' / 'records.jsonl').write_text(text, encoding='utf-8')
        (tmp / 'nested' / 'tier-concurrent-timed').mkdir(parents=True)
        (tmp / 'nested' / 'tier-concurrent-timed' / 'records.jsonl').write_text(text, encoding='utf-8')
        for path in (tmp / 'direct' / 'records.jsonl', tmp / 'direct', tmp / 'nested'):
            self.assertEqual(len(conc.load_records(str(path))), len(arm_records()), path)
        with self.assertRaises(IOError):
            conc.load_records(str(tmp / 'nowhere'))
        (tmp / 'empty').mkdir()
        with self.assertRaises(IOError):
            conc.load_records(str(tmp / 'empty'))


# -- (2) the judge ---------------------------------------------------------------------------------------------

class JudgeTests(unittest.TestCase):
    def judge(self, arm_=None, records=None, event_=None, scanned_=None, events=None):
        events = events if events is not None else {'tier-concurrent': event_ if event_ is not None else event()}
        return gate.concurrent_findings(arm_ or arm(), records if records is not None else arm_records(), events, scanned_ or scanned())

    def test_a_clean_timed_arm_passes_and_states_every_number(self):
        problems, missing, lines = self.judge()
        self.assertEqual((problems, missing), ([], []), lines)
        text = '\n'.join(lines)
        for needle in ('window restore:', 'window spill:', 'round p99 vs control +0.0%', 'stall budget 8.0 s', 'within', 'restore: 450 blocks 3.50 GB in 210.0 ms',
                       'spill: the long prompt\'s displaced blocks: 2.00 GB', 'tier_concurrent_compare.py'):
            self.assertIn(needle, text)

    def test_a_seat_that_waited_over_the_floor_fails_the_timed_arm_in_every_window_it_waited(self):
        # a wait is the window's when it overlaps it: a 9 s wait from 101 s lies across the restore (100-103) and the returning session's decode (103-108)
        for at, windows in ((101.0, ['restore', 'restore_decode']), (155.0, ['spill']), (70.0, ['control'])):
            with self.subTest(at=at):
                problems, missing, lines = self.judge(records=arm_records(gaps=[(at, 9.0)]))
                self.assertEqual([text.split(':')[0] for text in problems], ['window %s' % name for name in windows], (problems, lines))
                self.assertTrue(all('a seat waited 9.000 s between two tokens, over the 8.0 s decode gap floor' in text for text in problems), problems)

    def test_the_audited_arm_prints_the_stall_and_does_not_judge_it(self):
        problems, missing, lines = self.judge(arm_=arm(timed=False), records=arm_records(gaps=[(101.0, 9.0)]))
        self.assertEqual(problems, [])
        self.assertTrue(any(line.startswith('(not judged: audited arm) window restore') for line in lines), lines)
        self.assertTrue(any('printed only on this audited arm (over)' in line for line in lines), lines)

    def test_the_profiles_own_floor_is_the_budget(self):
        records = arm_records(gaps=[(101.0, 3.0)])
        self.assertEqual(self.judge(records=records)[0], [])
        problems, _, _ = self.judge(arm_=arm(QWEN_FAST_LEVERN_MAX_DECODE_GAP_S='2'), records=records)
        self.assertTrue(problems and 'over the 2.0 s decode gap floor' in problems[0], problems)

    def test_a_window_without_every_seat_is_not_exercised_not_a_pass(self):
        records = arm_records()
        records[2] = seat(2, steady(end=120.0))        # the third seat finished its budget before the spill window
        _, missing, _ = self.judge(records=records)
        self.assertTrue(any(text.startswith('window spill: 2 of 3 seats decoded across it') for text in missing), missing)
        self.assertFalse(any(text.startswith('window control') for text in missing))
        _, missing, lines = self.judge(records=records[:-1] + [dict(records[-1])])
        self.assertTrue(any('window cold: only 2 of 3 seats were still decoding' in line for line in lines), lines)
        self.assertFalse(any('window cold' in text for text in missing), 'the cold twin\'s window is a note when the seats are gone')

    def test_seats_that_left_no_chunk_times_are_reported(self):
        records = [dict(r, chunk_times=[]) if r['role'] == 'seat' else r for r in arm_records()]
        _, missing, _ = self.judge(records=records)
        self.assertTrue(any('only 0 of 3 seats left chunk times' in text for text in missing), missing)

    def test_the_scenario_that_never_measured_is_said_so_in_one_line(self):
        self.assertTrue(any('no tier-concurrent event' in text for text in self.judge(events={})[1]))
        _, missing, _ = self.judge(event_=event(skipped=True))
        self.assertTrue(any('cannot hold the returning session' in text for text in missing), missing)
        _, missing, _ = self.judge(event_=event(built_ok=False))
        self.assertTrue(any('could not be built' in text for text in missing), missing)
        _, missing, _ = self.judge(event_=event(seats_running=False))
        self.assertTrue(any('not every seat had streamed a first token' in text for text in missing), missing)

    def test_a_tier_arm_needs_the_restore_and_the_spill_on_record(self):
        _, missing, _ = self.judge(scanned_=scanned(restore=False))
        self.assertTrue(any('no "host tier restore" line for the returning turn ret (Q=29000)' in text for text in missing), missing)
        _, missing, _ = self.judge(scanned_=scanned(spill=False))
        self.assertTrue(any('displaced nothing to host RAM' in text for text in missing), missing)
        records = [dict(r, ok=False, error='boom') if r['case'] == conc.RETURN_CASE else r for r in arm_records()]
        _, missing, _ = self.judge(records=records, event_=event(return_ok=False))
        self.assertTrue(any('the returning turn failed (boom)' in text for text in missing), missing)

    def test_the_control_arm_needs_no_tier_lines_and_says_what_its_windows_are(self):
        problems, missing, lines = self.judge(arm_=arm(tier=False), scanned_=scanned(restore=False, spill=False))
        self.assertEqual((problems, missing), ([], []), lines)
        self.assertTrue(any('tier OFF (the control profile)' in line for line in lines), lines)
        self.assertTrue(any('no tier on this profile: the restore window is a cold re-prefill' in line for line in lines), lines)

    def test_the_continuations_rule_of_the_timed_arms_is_not_asked_of_the_concurrent_ones(self):
        # the control's returning turn is a cold re-prefill by design: the generic "no continuation restored Q > 0" rule would make the control NOT_EXERCISED
        source = Path(gate.__file__).read_text(encoding='utf-8')
        self.assertIn("arm['scenario'] in TIMED_SCENARIOS and arm['prefix'] and arm['scenario'] not in CONCURRENT_SCENARIOS", source)


# -- (3) the plans ---------------------------------------------------------------------------------------------

class PlanTests(unittest.TestCase):
    def test_the_two_plans_are_registered_in_the_job_parser_and_the_gate(self):
        for plan, scenario in (('tier-concurrent', 'tier_concurrent'), ('tier-concurrent-timed', 'tier_concurrent_timed')):
            self.assertIn(plan, job.PREFIX_PLANS)
            self.assertIn(plan, gate.PLANS)
            self.assertEqual([entry[:3] for entry in gate.PLAN_ARMS[plan]], [(plan, scenario, 'prefix')])
            self.assertIn(scenario, replay.SCENARIOS)
            self.assertIn(scenario, gate.TIER_SCENARIOS)
            self.assertIn(plan, gate.TIER_OPTIONAL_ARMS)
        self.assertIn('tier_concurrent_timed', gate.TIMED_SCENARIOS)
        self.assertNotIn('tier_concurrent', gate.TIMED_SCENARIOS)
        self.assertEqual(gate.CONCURRENT_SCENARIOS, ('tier_concurrent', 'tier_concurrent_timed'))
        self.assertIs(replay.SCENARIOS['tier_concurrent'], replay.SCENARIOS['tier_concurrent_timed'])
        self.assertNotIn('tiers', [plan for plan, _ in job.PREFIX_ARM_PLANS], 'the concurrent plans are plans of their own, not arms of the tiers plan')

    def test_the_tiers_plan_still_has_its_three_arms(self):
        self.assertEqual([entry[0] for entry in gate.PLAN_ARMS['tiers']], ['tier-attach', 'tier-returning', 'tier-timed'])

    def test_the_arm_runs_on_a_tier_profile_and_on_its_control_with_no_tier(self):
        tier_arms = gate.plan_arms('tier-concurrent-timed', FAKE, 'none', tiers.tier_profiles())
        control_arms = gate.plan_arms('tier-concurrent-timed', FAKE, 'none', gate_tests.sticky_profiles())
        self.assertEqual([a['arm'] for a in tier_arms], ['tier-concurrent-timed'])
        self.assertEqual([a['arm'] for a in control_arms], ['tier-concurrent-timed'])
        self.assertTrue(tier_arms[0]['tier'])
        self.assertIsNone(control_arms[0]['tier'])
        self.assertEqual(tier_arms[0]['timeout'], 7200)
        self.assertIsNone(tier_arms[0]['derived'], 'no derived profile: the timed arm is the profile itself, with the audits it names')
        self.assertFalse(gate.wants_digests(tier_arms[0]), 'a timed arm carries no digests')
        audited = gate.plan_arms('tier-concurrent', FAKE, 'none', tiers.tier_profiles())[0]
        self.assertTrue(gate.wants_digests(audited))
        self.assertEqual(gate.not_applicable('tier-concurrent', FAKE, gate_tests.sticky_profiles()), [])
        self.assertEqual(gate.not_applicable('tier-concurrent', FAKE, tiers.tier_profiles()), [])

    def test_a_tier_profile_without_the_preconverted_checkpoints_is_still_refused_for_it(self):
        half = tiers.tier_profiles()
        half['profiles'][FAKE]['env'].pop('QWEN_PREFIX_CKPT_PRECONVERTED')
        self.assertEqual(gate.plan_arms('tier-concurrent', FAKE, 'none', half), [])
        self.assertTrue(all('QWEN_PREFIX_CKPT_PRECONVERTED=1' in why for _, why in gate.not_applicable('tier-concurrent', FAKE, half)))

    def test_the_image_check_is_asked_of_the_tier_arm_only(self):
        tier_arms = gate.plan_arms('tier-concurrent-timed', FAKE, 'none', tiers.tier_profiles())
        control_arms = gate.plan_arms('tier-concurrent-timed', FAKE, 'none', gate_tests.sticky_profiles())
        bad = dict(gate_tests.GOOD_ANCHOR, raw_ops=False)
        self.assertTrue(gate.tier_image_problems(tier_arms, bad))
        self.assertEqual(gate.tier_image_problems(control_arms, bad), [], 'the control runs the production extension: no raw ops asked')

    def test_main_plans_the_real_ship_candidate_its_audit_twin_and_the_control(self):
        results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, results, True)
        rows = {}
        for profile, plan in ((tier_twin.FX_PARENT, 'tier-concurrent-timed'), (tier_twin.SHIP, 'tier-concurrent-timed'), (tier_twin.SHIP_AUDIT, 'tier-concurrent')):
            lines = []
            code = gate.main(['--image', 'img', '--profiles', str(REAL_PROFILES), '--results', results, '--profile', profile, '--plan', plan, '--baseline', 'none',
                              '--dry-run', '--cards', 'quad'], devices=['/a', '/b', '/c', '/d'], log=lines.append)
            self.assertEqual(code, 0, (profile, lines))
            rows[profile] = [json.loads(line) for line in lines if line.startswith('{') and '"docker"' in line]
            self.assertEqual([row['arm'] for row in rows[profile]], [plan], profile)
            self.assertEqual(rows[profile][0]['served'], profile, 'the profile itself serves: no derived profile')
        self.assertNotIn('QWEN_PREFIX_DIGESTS=1', rows[tier_twin.FX_PARENT][0]['docker'])
        self.assertNotIn('QWEN_PREFIX_DIGESTS=1', rows[tier_twin.SHIP][0]['docker'], 'a timed arm carries no digests')
        self.assertIn('QWEN_PREFIX_DIGESTS=1', rows[tier_twin.SHIP_AUDIT][0]['docker'])
        self.assertIn('QWEN_C2_GATE=1', rows[tier_twin.SHIP_AUDIT][0]['docker'], 'the audit twin is gate only')
        self.assertEqual(rows[tier_twin.SHIP][0]['timeout'], 7200)

    def test_the_timed_arm_on_the_ship_candidate_reads_a_tier_with_its_audits_off(self):
        document = json.loads(REAL_PROFILES.read_text(encoding='utf-8'))
        arm_ = gate.plan_arms('tier-concurrent-timed', tier_twin.SHIP, 'none', document)[0]
        self.assertEqual((arm_['tier']['audit'], arm_['tier']['preconverted_audit'], arm_['tier']['verify']), (False, False, 'sample'))
        audited = gate.plan_arms('tier-concurrent', tier_twin.SHIP_AUDIT, 'none', document)[0]
        self.assertEqual((audited['tier']['audit'], audited['tier']['preconverted_audit'], audited['tier']['verify']), (True, True, 'all'))
        control = gate.plan_arms('tier-concurrent-timed', tier_twin.FX_PARENT, 'none', document)[0]
        self.assertIsNone(control['tier'])
        self.assertEqual(control['seats'], 8, 'seven seats and the returning session')


# -- (4) the scenario, end to end on the fakes ---------------------------------------------------------------------

class SlowRestoreEngine(tiers.TierFakeEngine):
    """The tier fake whose engine thread stays in a restored prefill for STALL seconds: every seat waits that long, as they would behind a slow restore."""

    STALL = 0.6

    def prefill(self, request, q):
        super().prefill(request, q)
        if q and request.tag.endswith('-hit'):
            time.sleep(self.STALL)


PATCHES = dict(TIER_CONC_SETTLE_S=0.05, TIER_CONC_CONTROL_S=0.6, TIER_CONC_BETWEEN_S=0.1, TIER_CONC_SEAT_BUDGET=24000, TIER_CONC_SEAT_MIN_BUDGET=6000,
               TIER_CONC_RETURN_MAX_TOKENS=3000)


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results, True)
        for name, value in PATCHES.items():
            patcher = mock.patch.object(replay, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(gate, 'SETTLE_SECONDS', 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_plan(self, plan, document, engine_class=None, **faults):
        harness = gate_tests.Harness(engine_class or tiers.TierFakeEngine, document=document, **faults)
        runner = harness.runner(self.results)
        runner.sleep = lambda seconds: time.sleep(min(seconds, 1.0))
        arms = gate.plan_arms(plan, FAKE, 'none', document)
        return gate.run_plan(plan, arms, runner, gate_tests.GOOD_ANCHOR), harness

    def records(self, plan):
        with open(os.path.join(self.results, plan, 'records.jsonl'), encoding='utf-8') as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_the_timed_tier_arm_passes_with_every_window_measured_and_the_restore_on_record(self):
        result, harness = self.run_plan('tier-concurrent-timed', tiers.tier_profiles())
        arm_result = result['arms']['tier-concurrent-timed']
        self.assertEqual(arm_result['verdict'], 'PASS', (arm_result['problems'], arm_result['not_exercised'], arm_result['lines'][-14:]))
        text = '\n'.join(arm_result['lines'])
        for needle in ('window control:', 'window restore:', 'window restore_decode:', 'window spill:', 'window cold:', 'restore: ', 'spill: the long prompt'):
            self.assertIn(needle, text)
        self.assertTrue(harness.engines[-1].check_kv())
        records = self.records('tier-concurrent-timed')
        cases = [r['case'] for r in records]
        for case in (conc.RETURN_CASE, conc.SPILL_CASE, conc.COLD_CASE, 'tier-conc-build'):
            self.assertIn(case, cases)
        seats = [r for r in records if r['role'] == 'seat']
        self.assertEqual(len(seats), 3, 'one seat fewer than the profile\'s seats: the returning session is the other')
        self.assertTrue(all(r.get('intended_abort') for r in seats), 'the scenario stops its seats; they do not fail')
        order = sorted((r for r in records if r['case'] in (conc.RETURN_CASE, conc.SPILL_CASE, conc.COLD_CASE)), key=lambda r: r['sent_s'])
        self.assertEqual([r['case'] for r in order], [conc.RETURN_CASE, conc.SPILL_CASE, conc.COLD_CASE])
        self.assertTrue(all(r['sent_s'] >= min(s['sent_s'] for s in seats) + replay.TIER_CONC_SETTLE_S + replay.TIER_CONC_CONTROL_S - 0.2 for r in order[:1]),
                        'the returning turn went out after the control window')

    def test_the_restored_answer_equals_its_cold_twin_and_the_event_carries_the_windows(self):
        document = tiers.tier_profiles()
        harness = gate_tests.Harness(tiers.TierFakeEngine, document=document)
        runner = harness.runner(self.results)
        runner.sleep = lambda seconds: time.sleep(min(seconds, 1.0))
        arms = gate.plan_arms('tier-concurrent', FAKE, 'none', document)
        driver, result = runner.run(arms[0])
        self.assertEqual(result['verdict'], 'PASS', (result['problems'], result['not_exercised'], result['lines'][-12:]))
        self.assertEqual([(p['case'], p['verdict']) for p in driver.pairs], [('tier-conc', 'IDENTICAL')])
        event_ = driver.events['tier-concurrent']
        self.assertEqual(sorted(event_['windows']), ['cold', 'control', 'restore', 'restore_decode', 'spill'])
        self.assertTrue(event_['return_ok'] and event_['spill_ok'] and event_['seats_running'])
        self.assertEqual(event_['seats'], 3)
        self.assertNotIn('log_window', event_, 'an event with a log window is a quiet window to the failure scan')
        self.assertGreaterEqual(len(event_['flood_sizes']), 2)

    def test_the_control_profile_with_no_tier_runs_the_same_scenario_and_passes(self):
        result, _ = self.run_plan('tier-concurrent-timed', gate_tests.sticky_profiles(), engine_class=fakes.FakeEngine)
        arm_result = result['arms']['tier-concurrent-timed']
        self.assertEqual(arm_result['verdict'], 'PASS', (arm_result['problems'], arm_result['not_exercised'], arm_result['lines'][-12:]))
        self.assertTrue(any('tier OFF (the control profile)' in line for line in arm_result['lines']), arm_result['lines'])

    def test_a_slow_restore_over_the_floor_fails_the_timed_arm_and_only_prints_on_the_audited_one(self):
        document = tiers.tier_profiles(QWEN_FAST_LEVERN_MAX_DECODE_GAP_S='0.3')
        result, _ = self.run_plan('tier-concurrent-timed', document, engine_class=SlowRestoreEngine)
        timed = result['arms']['tier-concurrent-timed']
        self.assertEqual(timed['verdict'], 'FAIL', timed['lines'][-10:])
        self.assertTrue(any('a seat waited' in text and 'over the 0.3 s decode gap floor' in text for text in timed['problems']), timed['problems'])
        shutil.rmtree(self.results, True)
        self.results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results, True)
        result, _ = self.run_plan('tier-concurrent', document, engine_class=SlowRestoreEngine)
        audited = result['arms']['tier-concurrent']
        self.assertNotEqual(audited['verdict'], 'FAIL', (audited['problems'], audited['lines'][-10:]))
        self.assertTrue(any(line.startswith('(not judged: audited arm)') for line in audited['lines']), audited['lines'])

    def test_a_tier_that_never_spills_is_not_exercised_in_the_concurrent_arm_too(self):
        result, _ = self.run_plan('tier-concurrent-timed', tiers.tier_profiles(), **{FAKE: dict(tier_never_spills=True)})
        arm_result = result['arms']['tier-concurrent-timed']
        self.assertEqual(arm_result['verdict'], 'NOT_EXERCISED', (arm_result['problems'], arm_result['lines'][-8:]))
        self.assertTrue(any('host tier restore' in text or 'tier_spill_blocks' in text or 'spilled no block' in text for text in arm_result['not_exercised']), arm_result['not_exercised'])

    def test_the_seats_text_is_the_same_with_the_tier_and_without_it(self):
        self.run_plan('tier-concurrent-timed', tiers.tier_profiles())
        tier_records = self.records('tier-concurrent-timed')
        shutil.rmtree(self.results, True)
        self.results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results, True)
        self.run_plan('tier-concurrent-timed', gate_tests.sticky_profiles(), engine_class=fakes.FakeEngine)
        control_records = self.records('tier-concurrent-timed')
        outcome = compare.compare_pair(control_records, tier_records, 'control', 'tier', minimum=200)
        self.assertEqual(outcome['problems'], [], outcome['lines'])
        self.assertEqual(outcome['verdict'], 'PASS')
        self.assertEqual([row['verdict'] for row in outcome['seats']], ['IDENTICAL'] * 3)
        self.assertEqual([row['case'] for row in outcome['whole']], [conc.RETURN_CASE, conc.SPILL_CASE, conc.COLD_CASE])


# -- (5) the comparator ------------------------------------------------------------------------------------------

class CompareTests(unittest.TestCase):
    def pair(self, b_records=None, **options):
        return compare.compare_pair(arm_records(), b_records if b_records is not None else arm_records(), 'control', 'tier', minimum=1000, **options)

    def test_two_equal_arms_pass_and_print_the_side_by_side_stall(self):
        outcome = self.pair()
        self.assertEqual((outcome['verdict'], outcome['problems']), ('PASS', []), outcome['lines'])
        text = '\n'.join(outcome['lines'])
        self.assertIn('seat text control vs tier: 3 seats compared, 3 identical, 0 diverged, 0 short', text)
        self.assertIn('tier-conc-return control vs tier: IDENTICAL', text)
        self.assertIn('window restore: round p50/p99 0.1/0.1 s (control) vs 0.1/0.1 s (tier)', text)
        self.assertIn('round p99 tier +0.0% against control', text)

    def test_a_seat_that_diverges_or_exists_in_one_arm_only_fails(self):
        other = arm_records()
        other[1]['token_ids'][1200] = -5
        outcome = self.pair(other)
        self.assertEqual(outcome['verdict'], 'FAIL')
        self.assertTrue(any('diverged at output token 1200' in text for text in outcome['problems']), outcome['problems'])
        fewer = arm_records(seats=2)
        outcome = self.pair(fewer)
        self.assertTrue(any('seats only control ran' in text for text in outcome['problems']), outcome['problems'])

    def test_a_returning_answer_that_differs_is_a_failure_and_a_missing_one_is_named(self):
        other = [dict(r, token_ids=[1, 2, 4]) if r['case'] == conc.RETURN_CASE else r for r in arm_records()]
        outcome = self.pair(other)
        self.assertTrue(any(text.startswith('%s DIVERGED' % conc.RETURN_CASE) for text in outcome['problems']), outcome['problems'])
        missing = [r for r in arm_records() if r['case'] != conc.COLD_CASE]
        outcome = self.pair(missing)
        self.assertTrue(any(text.startswith('%s MISSING' % conc.COLD_CASE) for text in outcome['problems']), outcome['problems'])

    def test_a_stall_over_the_budget_in_either_arm_is_a_problem(self):
        outcome = self.pair(arm_records(gaps=[(101.0, 9.0)]))
        self.assertTrue(any(text.startswith('tier window restore: a seat waited') for text in outcome['problems']), outcome['problems'])
        outcome = compare.compare_pair(arm_records(gaps=[(155.0, 9.0)]), arm_records(), 'control', 'tier', minimum=1000)
        self.assertTrue(any(text.startswith('control window spill: a seat waited') for text in outcome['problems']), outcome['problems'])

    def test_the_command_line_compares_one_pair_or_the_abab_with_its_floor(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(tmp), True)
        paths = []
        for name in ('a', 'b', 'c', 'd'):
            (tmp / name).mkdir()
            (tmp / name / 'records.jsonl').write_text(replay.records_jsonl(arm_records()), encoding='utf-8')
            paths.append(str(tmp / name))
        out = []
        self.assertEqual(compare.main(paths[:2], out=out.append), 0, out)
        self.assertIn('tier concurrent control vs tier: PASS', out)
        out = []
        self.assertEqual(compare.main(paths, out=out.append), 0, out)
        self.assertIn('tier concurrent control-repeat vs tier-repeat: PASS', out)
        self.assertIn('tier concurrent floor control vs control-repeat: PASS', out)
        self.assertTrue(any(line.startswith('FLOOR (the same control profile twice') for line in out), out)
        out = []
        self.assertEqual(compare.main(paths[:3], out=out.append), 2, 'three paths are not a pair or an ABAB')
        out = []
        self.assertEqual(compare.main([paths[0], str(tmp / 'nowhere')], out=out.append), 2)
        self.assertTrue(out and out[0].startswith('refused:'), out)
        diverged = arm_records()
        diverged[0]['token_ids'][1500] = -1
        (tmp / 'b' / 'records.jsonl').write_text(replay.records_jsonl(diverged), encoding='utf-8')
        out = []
        self.assertEqual(compare.main(paths[:2], out=out.append), 1, out)

    def test_the_comparator_is_stdlib_python37_and_takes_no_environment(self):
        source = (HERE / 'tier_concurrent_compare.py').read_text(encoding='utf-8')
        self.assertNotIn('os.environ', source)
        self.assertNotIn(':=', source)
        for forbidden in ('subprocess', 'socket', 'urllib', 'requests'):
            self.assertNotIn('import %s' % forbidden, source)


# -- (6) the job pack ----------------------------------------------------------------------------------------------

PACK = HERE / 'references' / 'fusion-jobs' / 'tier'


class PackTests(unittest.TestCase):
    """references/fusion-jobs/tier: the templates, the order and what each one runs. (Each template also exits zero through the reader in test_tp4_fusion_jobs, which holds
    every package folder to the public-safety rules and to the folder's one image.)"""

    IMAGE = 'tp4-fusion-tier-1'

    def order(self):
        rows = [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line and not line.startswith('#')]
        return rows

    def env(self, name):
        return job.parse_env((PACK / ('%s.env' % name)).read_text(encoding='utf-8'))

    def test_the_order_lists_exactly_the_templates_in_a_sensible_sequence(self):
        rows = self.order()
        names = [row[0] for row in rows]
        self.assertEqual(sorted(names), sorted(path.stem for path in PACK.glob('*.env')))
        self.assertEqual([name.split('-')[0] for name in names], ['B0', 'X0', 'TCX0', 'TCX1', 'TC1', 'TQ1', 'TC2', 'TC3', 'TC4', 'TC5', 'Z'])
        for name, mode, image, minutes in rows:
            self.assertIn(mode, ('stop', 'soft'), name)
            self.assertEqual(image, self.IMAGE, name)
            self.assertGreater(int(minutes), 0, name)
            self.assertEqual(self.env(name)['C2_IMAGE_TAG'], self.IMAGE, name)
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        needs = [line for line in text.splitlines() if line.startswith('# NEEDS ')]
        short = set(name.split('-')[0] for name in names)
        for line in needs:
            left, right = line[len('# NEEDS '):].split(' <- ')
            self.assertTrue(set(left.split() + right.split()) <= short, line)
        self.assertIn('# NEEDS TC3 <- TC1 TC2', needs, 'the tier is timed only after its audited arm and the control passed')
        self.assertIn('# NEEDS TQ1 <- TC1', needs)

    def test_every_template_runs_what_its_name_says(self):
        expect = {
            'TCX0-exact-control': ('smoke', None, tier_twin.FX_PARENT), 'TCX1-exact-tier': ('smoke', None, tier_twin.SHIP),
            'TC1-tier-concurrent-audited': ('prefix', 'tier-concurrent', tier_twin.SHIP_AUDIT),
            'TQ1-tier-quota-isolation': ('prefix', 'tier-quota', tier_twin.SHIP_AUDIT),
            'TC2-tier-concurrent-control-timed': ('prefix', 'tier-concurrent-timed', tier_twin.FX_PARENT),
            'TC3-tier-concurrent-tier-timed': ('prefix', 'tier-concurrent-timed', tier_twin.SHIP),
            'TC4-tier-concurrent-control-repeat': ('prefix', 'tier-concurrent-timed', tier_twin.FX_PARENT),
            'TC5-tier-concurrent-tier-repeat': ('prefix', 'tier-concurrent-timed', tier_twin.SHIP)}
        for name, (action, plan, profile) in expect.items():
            env = self.env(name)
            with self.subTest(name):
                self.assertEqual(env['C2_ACTIONS'], 'reset %s' % action)
                if plan is None:
                    self.assertEqual(env['C2_PROFILE'], profile)
                    self.assertEqual(env['C2_SMOKE_TESTS'], 'warmup,parked_equal,parked_budgets,levern_equal,levern_equal_busy')
                else:
                    self.assertEqual((env['C2_PREFIX_PLAN'], env['C2_PREFIX_PROFILE'], env['C2_PREFIX_BASELINE']), (plan, profile, 'none'))
                    self.assertIn(plan, job.PREFIX_PLANS)
                self.assertEqual(env['C2_CARDS'], 'quad')
        for name in ('B0-tier-build', 'X0-status-rescan-reset', 'Z-reset'):
            self.assertNotIn('C2_BAKE_DEFAULT_PROFILE', self.env(name), 'a PENDING profile cannot be baked: the image is for gates')

    def test_the_abab_alternates_control_and_tier_and_the_audit_twin_is_the_only_audited_arm(self):
        profiles_ = [self.env(name)['C2_PREFIX_PROFILE'] for name in ('TC2-tier-concurrent-control-timed', 'TC3-tier-concurrent-tier-timed',
                                                                      'TC4-tier-concurrent-control-repeat', 'TC5-tier-concurrent-tier-repeat')]
        self.assertEqual(profiles_, [tier_twin.FX_PARENT, tier_twin.SHIP, tier_twin.FX_PARENT, tier_twin.SHIP])
        document = json.loads(REAL_PROFILES.read_text(encoding='utf-8'))['profiles']
        self.assertIsNone(tiers_of(document[tier_twin.FX_PARENT]))
        self.assertIsNotNone(tiers_of(document[tier_twin.SHIP]))
        self.assertTrue(document[tier_twin.SHIP_AUDIT].get('gate_only'))
        self.assertIsNot(document[tier_twin.SHIP].get('gate_only'), True, 'the candidate is a traffic profile')
        self.assertIn('PENDING', document[tier_twin.SHIP]['owner_traffic_waiver']['decision'])

    def test_the_order_names_the_profiles_the_waiver_and_the_question(self):
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        for needle in (tier_twin.SHIP, tier_twin.FX_PARENT, tier_twin.SHIP_AUDIT, tier_twin.SHIP_WAIVER_ID, 'tier_concurrent_compare.py', 'ESTIMATE (not measured)',
                       '8 s', 'Lever N decode gap floor'):
            self.assertIn(needle, text)
        for name in ('TC1-tier-concurrent-audited', 'TC3-tier-concurrent-tier-timed', 'TQ1-tier-quota-isolation'):
            body = (PACK / ('%s.env' % name)).read_text(encoding='utf-8')
            self.assertIn('Duration (estimate', body)
        compare_text = (PACK / 'TC2-tier-concurrent-control-timed.env').read_text(encoding='utf-8')
        self.assertIn('tier_concurrent_compare.py <TC2 results> <TC3 results> <TC4 results> <TC5 results>', compare_text)
        self.assertIn('parked_compare.py', (PACK / 'TCX1-exact-tier.env').read_text(encoding='utf-8'))
        quota_text = (PACK / 'TQ1-tier-quota-isolation.env').read_text(encoding='utf-8')
        self.assertIn('tier_concurrent_compare.py --quota <results>/quota-control <results>/quota-isolated', quota_text)
        self.assertIn('DEVICE-RESIDENT', quota_text)

    def test_every_prefix_template_plans_in_the_gate_as_a_dry_run(self):
        results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, results, True)
        for name in ('TC1-tier-concurrent-audited', 'TQ1-tier-quota-isolation', 'TC2-tier-concurrent-control-timed', 'TC3-tier-concurrent-tier-timed'):
            env = self.env(name)
            lines = []
            code = gate.main(['--image', 'img', '--profiles', str(REAL_PROFILES), '--results', results, '--profile', env['C2_PREFIX_PROFILE'], '--plan', env['C2_PREFIX_PLAN'],
                              '--baseline', env['C2_PREFIX_BASELINE'], '--dry-run', '--cards', 'quad'], devices=['/a', '/b', '/c', '/d'], log=lines.append)
            self.assertEqual(code, 0, (name, lines))


def tiers_of(profile):
    return gate.tier_profile(profile)


if __name__ == '__main__':
    unittest.main()
