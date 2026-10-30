"""The prefix gate's NVMe plans (tp4/kv-nvme-2, docs/tp4-kv-nvme.md section 12): nvme-exact, nvme-timed, nvme-concurrent, nvme-concurrent-whole and nvme-restart; c2_prefix_gate,
prefix_replay, prefix_markers and c2_serving_job on the CPU.

* the wiring: every plan is registered in the job parser and the gate as one arm of its name, the derived kinds carry exactly what they name (and may take a name OUT of a profile),
  the scenarios are registered and classified (timed or judged), the arms apply to an NVMe profile and the optional ones to a host-tier profile without it, an NVMe profile needs a
  drive handed over, and the restart drill is refused on a force-format profile (its second boot would wipe the store);
* the markers: the stop line and the encryption line, the trouble words;
* the judge (nvme_findings) with fabricated logs and counters in its PASS, NOT_EXERCISED and FAIL shapes, for every scenario;
* the restart scenario with a fake engine and fake restart callbacks: the order of the calls, the events it records, the persistence wait and its timeout;
* the plans over test_prefix_tier_gate's fake engine: a profile without the NVMe tier runs the timed plan as the control, one that asks for the tier and never logs it is NOT_EXERCISED.

Run at py 3.11: `py -3.11 -B -m unittest test_nvme_gate` from scripts/ci."""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c2_cache_drive as cache  # noqa: E402
import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import prefix_markers as pm  # noqa: E402
import prefix_replay as replay  # noqa: E402
import test_c2_prefix_gate as gate_tests  # noqa: E402
import test_prefix_replay as fakes  # noqa: E402
import test_prefix_tier_gate as tier_gate  # noqa: E402

FAKE = tier_gate.FAKE
PLANS = {
    'nvme-exact': ('nvme_exact', None),
    'nvme-timed': ('nvme_timed', None),
    'nvme-concurrent': ('nvme_concurrent', None),
    'nvme-concurrent-whole': ('nvme_concurrent', 'whole'),
    'nvme-restart': ('nvme_restart', 'nvmeplain'),
}
DEVICE = '/dev/c2-cache-drive'
NVME_ENV = {'QWEN_PREFIX_NVME': '1', 'QWEN_PREFIX_NVME_DEVICE': DEVICE, 'QWEN_PREFIX_NVME_RETENTION_ACK': '1', 'QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION': '1',
            'QWEN_PREFIX_NVME_AUDIT': '1'}
SLICE_ENV = {'QWEN_PREFIX_TIER_RESTORE_SLICE': '512', 'QWEN_PREFIX_TIER_RESTORE_SLICE_MS': '200'}


def nvme_profiles(**env):
    """test_prefix_tier_gate's fixtures with the NVMe tier's names (and the sliced restore) on c2-packed-prefix."""
    return tier_gate.tier_profiles(**dict(dict(NVME_ENV, **SLICE_ENV), **env))


# -- (1) the wiring -------------------------------------------------------------------------------------------

class WiringTests(unittest.TestCase):
    def test_every_plan_is_registered_in_the_job_parser_and_the_gate_as_one_arm_of_the_same_name(self):
        for plan, (scenario, kind) in PLANS.items():
            with self.subTest(plan=plan):
                self.assertIn(plan, job.PREFIX_PLANS)
                self.assertIn(plan, gate.PLANS)
                self.assertEqual([entry[0] for entry in gate.PLAN_ARMS[plan]], [plan])
                entry = gate.PLAN_ARMS[plan][0]
                self.assertEqual((entry[1], entry[2], entry[3]), (scenario, 'prefix', kind))
                self.assertIn(scenario, replay.SCENARIOS)
                self.assertNotIn(plan, [arm for arm, _ in job.PREFIX_ARM_PLANS], 'no arm has a plan of its own twice')

    def test_the_job_parser_takes_the_plans_and_refuses_an_arm_beside_itself(self):
        values = dict(C2_IMAGE_TAG='tp4-nvme1', C2_ACTIONS='prefix', C2_PREFIX_PLAN='nvme-exact,nvme-restart', C2_PREFIX_PROFILE='general-prefix')
        out = job.read_prefix(values, ['general-prefix', 'general'], True)
        self.assertEqual(out['prefix_plan'], 'nvme-exact,nvme-restart')
        with self.assertRaises(job.JobError):
            job.read_prefix(dict(values, C2_PREFIX_PLAN='nvme-exact,nvme-exact'), ['general-prefix', 'general'], True)

    def test_the_judged_scenarios_are_judged_and_the_timed_ones_are_timed(self):
        for scenario in ('nvme_timed', 'nvme_concurrent'):
            self.assertIn(scenario, gate.TIMED_SCENARIOS)
        for scenario in ('nvme_exact', 'nvme_restart'):
            self.assertNotIn(scenario, gate.TIMED_SCENARIOS)
        self.assertEqual(set(gate.NVME_SCENARIOS), {'nvme_exact', 'nvme_timed', 'nvme_concurrent', 'nvme_restart'})
        for scenario in gate.NVME_SCENARIOS:
            self.assertIn(scenario, gate.TIER_SCENARIOS, 'they ride the host tier: the arm carries the tier\'s settings')
        for scenario in ('nvme_exact', 'nvme_timed', 'nvme_concurrent'):
            self.assertIn(scenario, gate.TIER_RESTORE_SCENARIOS)
        self.assertNotIn('nvme_restart', gate.TIER_RESTORE_SCENARIOS, 'its registry restarts: the counters are the last boot\'s')
        self.assertEqual(gate.CONCURRENT_SCENARIOS, ('tier_concurrent', 'tier_concurrent_timed'), 'the host tier\'s own set is unchanged')
        self.assertIn('nvme_concurrent', gate.CONCURRENT_JUDGED)

    def test_the_scenarios_are_the_tier_cycles_and_the_restart_drill(self):
        self.assertIs(replay.SCENARIOS['nvme_exact'], replay.scenario_tier_upload)
        self.assertIs(replay.SCENARIOS['nvme_timed'], replay.scenario_tier_timed)
        self.assertIs(replay.SCENARIOS['nvme_concurrent'], replay.scenario_tier_concurrent)
        self.assertIs(replay.SCENARIOS['nvme_restart'], replay.scenario_nvme_restart)

    def test_the_kinds_carry_exactly_what_they_name(self):
        self.assertEqual(gate.DERIVED['whole']['env'], {'QWEN_PREFIX_TIER_RESTORE_SLICE': None, 'QWEN_PREFIX_TIER_RESTORE_SLICE_MS': None})
        self.assertEqual(gate.DERIVED['nvmeplain']['env'], {'QWEN_PREFIX_NVME_AUDIT': None})

    def test_a_derived_kind_with_none_takes_the_name_out_of_the_profile_and_leaves_the_rest(self):
        document = nvme_profiles()
        before = copy.deepcopy(document['profiles'][FAKE]['env'])
        name, derived = gate.derive(document, FAKE, 'whole')
        env = derived['profiles'][name]['env']
        self.assertNotIn('QWEN_PREFIX_TIER_RESTORE_SLICE', env)
        self.assertNotIn('QWEN_PREFIX_TIER_RESTORE_SLICE_MS', env)
        self.assertEqual({key: value for key, value in env.items()}, {key: value for key, value in before.items() if key not in SLICE_ENV})
        self.assertEqual(document['profiles'][FAKE]['env'], before, 'the image\'s own profile is not touched')
        name, derived = gate.derive(document, FAKE, 'nvmeplain')
        env = derived['profiles'][name]['env']
        self.assertNotIn('QWEN_PREFIX_NVME_AUDIT', env)
        self.assertEqual(env['QWEN_PREFIX_NVME'], '1')
        # a name the profile never had is simply not there (no KeyError)
        name, derived = gate.derive(nvme_profiles(), FAKE, 'whole')
        gate.derive(tier_gate.tier_profiles(), FAKE, 'whole')

    def test_the_arms_apply_to_an_nvme_profile_and_the_optional_ones_also_to_a_host_tier_profile(self):
        host_only = tier_gate.tier_profiles()
        with_nvme = nvme_profiles()
        for plan in PLANS:
            with self.subTest(plan=plan):
                self.assertEqual([arm['arm'] for arm in gate.plan_arms(plan, FAKE, 'none', with_nvme)], [plan])
                arms = gate.plan_arms(plan, FAKE, 'none', host_only)
                if plan in gate.NVME_OPTIONAL_ARMS:
                    self.assertEqual([arm['arm'] for arm in arms], [plan], 'the host-only control of the ABAB pair')
                    self.assertIsNone(arms[0]['nvme'])
                else:
                    self.assertEqual(arms, [])
                    self.assertTrue(all('does not name QWEN_PREFIX_NVME=1' in why for _, why in gate.not_applicable(plan, FAKE, host_only)), plan)
        self.assertEqual(set(gate.NVME_OPTIONAL_ARMS), {'nvme-timed', 'nvme-concurrent', 'nvme-concurrent-whole'})

    def test_a_profile_without_the_host_tier_has_none_of_them(self):
        document = gate_tests.sticky_profiles()
        for plan in PLANS:
            self.assertEqual(gate.plan_arms(plan, FAKE, 'none', document), [], plan)
            self.assertTrue(all('names no QWEN_PREFIX_HOST_TIER_GIB' in why for _, why in gate.not_applicable(plan, FAKE, document)), plan)

    def test_an_arm_on_an_nvme_profile_carries_the_tiers_settings(self):
        arm = gate.plan_arms('nvme-exact', FAKE, 'none', nvme_profiles())[0]
        self.assertEqual(arm['nvme'], dict(device=DEVICE, audit=True, format=False, verify='sample', require_sed=True, spill='through'))
        self.assertTrue(arm['tier'])
        other = gate.plan_arms('nvme-exact', FAKE, 'none', nvme_profiles(QWEN_PREFIX_NVME_REQUIRE_SED='0', QWEN_PREFIX_NVME_FORMAT='1', QWEN_PREFIX_NVME_VERIFY='all'))[0]
        self.assertEqual((other['nvme']['require_sed'], other['nvme']['format'], other['nvme']['verify']), (False, True, 'all'))
        self.assertIsNone(gate.nvme_profile(dict(env={'QWEN_PREFIX_NVME': '0'})))
        self.assertIsNone(gate.nvme_profile(dict(env={})))

    def test_the_restart_drill_is_refused_on_a_force_format_profile(self):
        document = nvme_profiles(QWEN_PREFIX_NVME_FORMAT='1')
        with self.assertRaises(gate.PlanError) as caught:
            gate.plan_arms('nvme-restart', FAKE, 'none', document)
        self.assertIn('would wipe the store', str(caught.exception))
        self.assertEqual(len(gate.plan_arms('nvme-exact', FAKE, 'none', document)), 1, 'the first gate is the one that formats; the other arms do not restart')

    def test_the_restart_arm_turns_the_tiers_write_audit_off_in_its_derived_profile_and_the_whole_arm_turns_the_slice_off(self):
        document = nvme_profiles()
        arm = gate.plan_arms('nvme-restart', FAKE, 'none', document)[0]
        env = arm['derived']['profiles'][arm['served']]['env']
        self.assertNotIn('QWEN_PREFIX_NVME_AUDIT', env)
        self.assertEqual(arm['served'], FAKE + '+nvmeplain')
        arm = gate.plan_arms('nvme-concurrent-whole', FAKE, 'none', document)[0]
        env = arm['derived']['profiles'][arm['served']]['env']
        self.assertTrue(set(SLICE_ENV).isdisjoint(env))
        self.assertEqual(gate.plan_arms('nvme-concurrent', FAKE, 'none', document)[0]['served'], FAKE, 'the sliced arm serves the profile as it is')


class MainTests(unittest.TestCase):
    def setUp(self):
        self.results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results, True)
        self.profiles = os.path.join(self.results, 'profiles.json')
        self.write(nvme_profiles())

    def write(self, document):
        document['profiles'][FAKE]['mesh_device'] = 'P150x4'          # the drive is for the four-card job: the fixture's profile opens the (1, 4) mesh
        with open(self.profiles, 'w', encoding='utf-8') as handle:
            json.dump(document, handle)

    def main(self, *extra):
        lines = []
        code = gate.main(['--image', 'img', '--profiles', self.profiles, '--results', os.path.join(self.results, 'out'), '--profile', FAKE, '--baseline', 'none', '--cards', 'quad']
                         + list(extra), devices=['/a', '/b', '/c', '/d'], log=lines.append, anchor=gate_tests.GOOD_ANCHOR)
        return code, lines

    def test_an_arm_on_an_nvme_profile_is_refused_up_front_without_a_drive(self):
        code, lines = self.main('--plan', 'nvme-exact,nvme-timed', '--dry-run')
        self.assertEqual(code, 2)
        self.assertIn('no cache drive was given', lines[-1])
        self.assertIn('nvme-exact, nvme-timed are served', lines[-1])

    def test_the_dry_run_with_a_drive_shows_the_placeholder_and_the_derived_arms(self):
        code, lines = self.main('--plan', 'nvme-restart,nvme-concurrent-whole', '--nvme-device', 'status', '--p2pmem', 'auto', '--dry-run')
        self.assertEqual(code, 0, lines)
        rows = [json.loads(line) for line in lines if line.startswith('{') and '"docker"' in line]
        self.assertEqual([row['arm'] for row in rows], ['nvme-restart', 'nvme-concurrent-whole'])
        for row in rows:
            self.assertIn('<the cache drive and p2pmem arguments, resolved per arm at run time>', row['docker'])
            self.assertTrue(any(token.endswith('profiles.json') or 'profiles.json' in token for token in row['docker']), 'the derived profile is mounted')

    def test_a_host_tier_profile_runs_the_optional_plan_without_a_drive(self):
        self.write(tier_gate.tier_profiles())
        code, lines = self.main('--plan', 'nvme-timed', '--dry-run')
        self.assertEqual(code, 0, lines)
        code, lines = self.main('--plan', 'nvme-exact', '--dry-run')
        self.assertEqual(code, 2)
        self.assertIn('has no arm that applies', lines[-1])


# -- (2) the markers ------------------------------------------------------------------------------------------

class MarkerTests(unittest.TestCase):
    def test_the_stop_and_encryption_lines_are_read(self):
        lines = ['2026-10-30T01:00:00.1Z [PINDIAG] prefix: nvme tier at-rest encryption verified: opal locking_enabled=1 locked=0 media_encryption=1 source=status-file',
                 '2026-10-30T01:00:01.1Z [PINDIAG] prefix: nvme tier stopped drained=1 snapshot=0 latched=0 ms=412']
        scanned = pm.scan(lines)
        self.assertEqual([(e['source'], e['media_encryption']) for e in scanned['nvme_atrest']], [('status-file', 1)])
        self.assertEqual([(e['drained'], e['snapshot'], e['latched'], e['ms']) for e in scanned['nvme_stops']], [(1, 0, 0, 412)])
        self.assertEqual(scanned['nvme_trouble'], [], 'a normal stop is not trouble')

    def test_the_sink_and_pack_failures_are_trouble(self):
        for text in ('commit sink failed: ValueError: x', 'spill sink failed: OSError: y', 'checkpoint not packed pos=2048: ValueError: z', 'writer error: OSError: w'):
            scanned = pm.scan(['2026-10-30T01:00:00.1Z [PINDIAG] prefix: nvme tier ' + text])
            self.assertEqual(len(scanned['nvme_trouble']), 1, text)

    def test_the_lines_the_tier_writes_are_the_ones_the_scanner_reads(self):
        import qwen_kv_nvme as tier
        import inspect

        source = inspect.getsource(tier)
        for fragment in ("'nvme tier stopped drained=%d snapshot=%d latched=%d ms=%.0f'", "'nvme tier at-rest encryption verified: opal locking_enabled=1 locked=0 media_encryption=1 source=%s'",
                         "'nvme tier restore req=%s blocks=%d bytes=%d from=%d to=%d backend=%s verify=%s'"):
            self.assertIn(fragment, source)


# -- (3) the judge --------------------------------------------------------------------------------------------

def arm_of(scenario='nvme_exact', **nvme):
    settings = dict(device=DEVICE, audit=True, format=False, verify='all', require_sed=True, spill='through')
    settings.update(nvme)
    return dict(arm=scenario.replace('_', '-'), scenario=scenario, nvme=settings, tier=dict(gib=20.0, preconverted=True, audit=False))


ON = [dict(index=3, backend='host', ttl_s=86400, verify='all', audit=1, spill='through', queue_bytes=1 << 31, capacity_bytes=1 << 39, drain_s=4, at_rest='opal')]
OPEN = [dict(index=2, entries=0, kv=0, ckpt=0, recovered_records=0, fp_drops=0, verify='all', audit=1, at_rest='opal', fingerprint='ab' * 8)]
ATREST = [dict(index=1, source='status-file')]
STATS = dict(nvme_spill_blocks=9000, nvme_restore_requests=3, nvme_restore_blocks=2000, nvme_ckpt_restored=3, nvme_audit_reads=9000, nvme_on=1, nvme_at_rest=1)


def scanned_of(restores=(), **more):
    scanned = dict(nvme_on=ON, nvme_open=OPEN, nvme_atrest=ATREST, nvme_trouble=[], nvme_stops=[], nvme_restores=list(restores), tier_restores=[])
    scanned.update(more)
    return scanned


def timed_events(sizes=(64000, 128000, 254000)):
    return dict(('tier-timed-%d' % size, dict(return_ok=True, return_tag='r%d' % size)) for size in sizes)


def restore(tag, blocks=1000, **more):
    return dict(tag=tag, blocks=blocks, bytes=blocks * 2228224, ms=500.0, **dict({'from': 0, 'to': blocks * 64}, **more))


class JudgeTests(unittest.TestCase):
    def judge(self, arm=None, events=None, scanned=None, stats=STATS):
        arm = arm or arm_of()
        events = timed_events() if events is None else events
        scanned = scanned or scanned_of([restore('r%d' % size) for size in (64000, 128000, 254000)])
        return gate.nvme_findings(arm, [], events, scanned, stats)

    def test_a_clean_exactness_arm_passes_and_prints_each_restore_from_the_drive(self):
        problems, missing, lines = self.judge()
        self.assertEqual((problems, missing), ([], []))
        text = '\n'.join(lines)
        self.assertIn('nvme tier: {"backend": "host"', text)
        self.assertIn('nvme open: entries=0', text)
        self.assertIn('at_rest=opal', text)
        for size in (64000, 128000, 254000):
            self.assertIn('tier-timed-%d: the returning turn restored 1000 blocks' % size, text)
        self.assertIn('GB/s', text)

    def test_a_returning_turn_with_no_restore_line_is_not_exercised(self):
        _, missing, _ = self.judge(scanned=scanned_of([restore('r64000'), restore('r254000')]))
        self.assertEqual(len(missing), 1)
        self.assertIn('tier-timed-128000: no "nvme tier restore" line for the returning turn r128000', missing[0])

    def test_trouble_lines_fail_the_arm(self):
        trouble = [dict(index=7, line='[PINDIAG] prefix: nvme tier DEVICE LOST: the controller is dead: latched off')]
        problems, _, _ = self.judge(scanned=scanned_of([restore('r64000')], nvme_trouble=trouble))
        self.assertEqual(len(problems), 1)
        self.assertIn('DEVICE LOST', problems[0])

    def test_a_tier_that_never_logged_its_install_or_open_line_is_not_exercised(self):
        problems, missing, _ = self.judge(scanned=scanned_of(nvme_on=[], nvme_open=[]))
        self.assertEqual(problems, [])
        self.assertTrue(any('no "nvme tier on" line' in text for text in missing), missing)
        self.assertTrue(any('no "nvme tier open" line' in text for text in missing), missing)

    def test_the_encryption_check_must_have_passed_unless_the_profile_turns_it_off(self):
        scanned = scanned_of([restore('r64000'), restore('r128000'), restore('r254000')], nvme_open=[dict(OPEN[0], at_rest='unchecked')], nvme_atrest=[])
        problems, _, _ = self.judge(scanned=scanned)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any('at_rest=unchecked' in text for text in problems))
        self.assertTrue(any('no "at-rest encryption verified" line' in text for text in problems))
        problems, _, _ = self.judge(arm=arm_of(require_sed=False), scanned=scanned)
        self.assertEqual(problems, [], 'a gate\'s own REQUIRE_SED=0 asks nothing')

    def test_every_failure_counter_fails_the_arm(self):
        for name, _why in gate.NVME_BAD_COUNTERS:
            with self.subTest(counter=name):
                problems, _, _ = self.judge(stats=dict(STATS, **{name: 1}))
                self.assertEqual(len(problems), 1, problems)
                self.assertIn('%s=1' % name, problems[0])
        self.assertEqual(gate.nvme_stat_problems(STATS, 'x'), [])
        names = [name for name, _ in gate.NVME_BAD_COUNTERS]
        import qwen_kv_nvme as tier
        self.assertTrue(set(names) <= set(tier.COUNTER_NAMES), sorted(set(names) - set(tier.COUNTER_NAMES)))

    def test_the_counters_that_prove_the_drive_was_used_are_asked_of_the_cycle_scenarios(self):
        _, missing, _ = self.judge(stats=dict(STATS, nvme_restore_blocks=0, nvme_ckpt_restored=0))
        self.assertTrue(any('restored no block' in text for text in missing), missing)
        self.assertTrue(any('no GDN checkpoint was read back' in text for text in missing), missing)
        _, missing, _ = self.judge(stats=dict(STATS, nvme_audit_reads=0))
        self.assertTrue(any('nvme_audit_reads is 0' in text for text in missing), missing)
        _, missing, _ = self.judge(arm=arm_of(audit=False), stats=dict(STATS, nvme_audit_reads=0))
        self.assertEqual(missing, [])

    def test_the_idle_copy_cost_is_printed_and_judged_on_the_timed_arms(self):
        stats = dict(STATS, nvme_through_copy_steps=100, nvme_through_copy_ms=1500.0, nvme_through_copied_blocks=3200)
        problems, _, lines = gate.nvme_findings(arm_of('nvme_timed'), [], timed_events(), scanned_of([restore('r%d' % size) for size in (64000, 128000, 254000)]), stats)
        self.assertEqual(problems, [])
        self.assertTrue(any('idle copy (write-through): 3200 blocks in 100 steps, 15.0 ms a step' in line for line in lines), lines)
        slow = dict(stats, nvme_through_copy_ms=2500.0)
        problems, _, _ = gate.nvme_findings(arm_of('nvme_timed'), [], timed_events(), scanned_of([restore('r%d' % size) for size in (64000, 128000, 254000)]), slow)
        self.assertTrue(any('25.0 ms a scheduler step' in text and 'N6' in text for text in problems), problems)
        problems, _, _ = gate.nvme_findings(arm_of('nvme_exact'), [], timed_events(), scanned_of([restore('r%d' % size) for size in (64000, 128000, 254000)]), slow)
        self.assertEqual(problems, [], 'the audited arm prints it: the audit is its cost')

    def test_no_stats_export_is_not_exercised(self):
        problems, missing, _ = self.judge(stats=None)
        self.assertEqual(problems, [])
        self.assertTrue(any('no registry stats export' in text for text in missing), missing)

    def test_the_concurrent_arm_looks_for_its_returning_turn(self):
        arm = arm_of('nvme_concurrent', audit=False)
        events = {'tier-concurrent': dict(return_ok=True, return_tag='conc-back')}
        problems, missing, lines = gate.nvme_findings(arm, [], events, scanned_of([restore('conc-back', 3000)]), dict(STATS, nvme_ckpt_restored=0))
        self.assertEqual((problems, missing), ([], []), 'a checkpoint restore is not asked of the concurrent arm')
        self.assertIn('tier-concurrent: the returning turn restored 3000 blocks', '\n'.join(lines))
        _, missing, _ = gate.nvme_findings(arm, [], events, scanned_of([]), STATS)
        self.assertTrue(any('tier-concurrent: no "nvme tier restore" line' in text for text in missing), missing)

    def test_an_arm_without_the_tier_is_not_judged_for_it(self):
        arm = arm_of()
        arm['nvme'] = None
        self.assertIsNone(arm.get('nvme'))      # (judge_arm calls nvme_findings only for an arm that carries the settings)


class RestartJudgeTests(unittest.TestCase):
    def event(self, graceful=True, **more):
        base = dict(nominal=64000, size=64000, graceful=graceful, build_ok=True, build_tag='b', persisted=True if graceful else None, persist_waited_s=40.0, persist_stored_blocks=980,
                    seconds=480.0, stop_s=6.1, exit_code=0 if graceful else 137, return_tag='r-%d' % graceful, return_ok=True)
        base.update(more)
        return base

    def events(self, **more):
        return {'nvme-stop-64000': self.event(True, **more), 'nvme-kill-128000': self.event(False)}

    def scanned(self, boots=3, restores=('r-1',), stops=None, **more):
        found = dict(nvme_on=[dict(ON[0], index=i) for i in range(boots)], nvme_open=[dict(OPEN[0], index=i, entries=900 * i) for i in range(boots)],
                     nvme_stops=[dict(index=5, drained=1, snapshot=1, latched=0, ms=900)] if stops is None else stops)
        found.update(more)
        return scanned_of([restore(tag, 900) for tag in restores], **found)

    def rows(self, events=None, scanned=None):
        return gate.nvme_restart_rows(arm_of('nvme_restart'), self.events() if events is None else events, scanned or self.scanned())

    def test_a_clean_pair_of_drills_passes_and_prints_the_stop_and_the_recovery(self):
        problems, missing, lines = self.rows()
        self.assertEqual((problems, missing), ([], []))
        text = '\n'.join(lines)
        self.assertIn('nvme-stop-64000: stopped in 6.1 s with exit code 0 (grace 10 s)', text)
        self.assertIn('nvme-kill-128000: SIGKILL (exit code 137)', text)
        self.assertIn('nvme tier stopped: drained=1 snapshot=1 latched=0 in 900 ms', text)
        self.assertIn('nvme-stop-64000: the returning turn restored 900 blocks', text)
        self.assertIn('a cold turn: nothing of it survived the kill, which is allowed', text)
        self.assertIn('nvme tier open (boot)', text)

    def test_a_graceful_stop_that_was_sigkilled_fails_naming_the_grace(self):
        problems, _, _ = self.rows(events=self.events(exit_code=137, stop_s=10.2))
        self.assertEqual(len(problems), 1)
        self.assertIn('SIGKILLed after the 10 s grace', problems[0])
        self.assertIn('QWEN_PREFIX_NVME_DRAIN_S', problems[0])
        problems, _, _ = self.rows(events=self.events(exit_code=None, stop_s=16.0))
        self.assertTrue(any('docker stop took 16.0 s' in text for text in problems), problems)

    def test_the_session_the_drive_held_must_come_back_from_it_after_a_graceful_stop(self):
        problems, _, _ = self.rows(scanned=self.scanned(restores=()))
        self.assertEqual(len(problems), 1)
        self.assertIn('was not restored from it', problems[0])

    def test_a_drive_that_did_not_hold_the_session_before_the_stop_is_not_exercised_not_failed(self):
        events = self.events(persisted=False, persist_stored_blocks=12)
        problems, missing, _ = self.rows(events=events, scanned=self.scanned(restores=()))
        self.assertEqual(problems, [])
        self.assertTrue(any('did not hold the session within 40.0 s' in text for text in missing), missing)

    def test_the_tier_must_come_back_at_every_boot(self):
        problems, _, _ = self.rows(scanned=self.scanned(boots=2))
        self.assertTrue(any('came back 1 time(s) after 2 restart(s)' in text for text in problems), problems)

    def test_a_missing_or_latched_stop_line_fails(self):
        problems, _, _ = self.rows(scanned=self.scanned(stops=[]))
        self.assertTrue(any('no "nvme tier stopped" line' in text for text in problems), problems)
        problems, _, _ = self.rows(scanned=self.scanned(stops=[dict(index=5, drained=0, snapshot=0, latched=1, ms=9000)]))
        self.assertTrue(any('latched off at its stop' in text for text in problems), problems)
        _, _, lines = self.rows(scanned=self.scanned(stops=[dict(index=5, drained=0, snapshot=0, latched=0, ms=9000)]))
        self.assertTrue(any(line.startswith('note: the drain did not finish') for line in lines))

    def test_a_returning_turn_that_failed_after_a_restart_fails(self):
        problems, _, _ = self.rows(events=dict(self.events(), **{'nvme-kill-128000': self.event(False, return_ok=False)}))
        self.assertTrue(any('the returning turn' in text and 'failed after the restart' in text for text in problems), problems)

    def test_a_fingerprint_that_changed_across_a_restart_of_the_same_image_fails(self):
        opened = [dict(OPEN[0], index=i, fp_drops=5 if i == 1 else 0) for i in range(3)]
        problems, _, _ = self.rows(scanned=self.scanned(nvme_open=opened))
        self.assertTrue(any('dropped 5 entries of another fingerprint' in text for text in problems), problems)

    def test_no_drill_is_not_exercised(self):
        problems, missing, _ = gate.nvme_restart_rows(arm_of('nvme_restart'), {'nvme-restart': dict(skipped=True, reason='no restart callbacks')}, self.scanned())
        self.assertEqual(problems, [])
        self.assertIn('no restart callbacks', missing[0])

    def test_the_restart_arm_in_the_judge_adds_its_rows(self):
        arm = arm_of('nvme_restart', audit=False)
        problems, missing, lines = gate.nvme_findings(arm, [], self.events(), self.scanned(), dict(STATS, nvme_restore_requests=0))
        self.assertEqual((problems, missing), ([], []), 'the counters are the last boot\'s: nothing is asked of them but the failure counters')
        self.assertIn('nvme-stop-64000: stopped in 6.1 s', '\n'.join(lines))


# -- (4) the scenario -----------------------------------------------------------------------------------------

class FakeClock(object):
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ScenarioTests(unittest.TestCase):
    def driver(self):
        document = nvme_profiles()
        engine = tier_gate.TierFakeEngine(profile=document['profiles'][FAKE], name=FAKE)
        driver = fakes.sticky_driver(engine, 'nvme-restart')
        clock = FakeClock()
        driver.clock, driver.sleep = clock, clock.sleep
        return driver, engine, clock

    def run_scenario(self, counters, sizes=(8000, 8000), **callbacks):
        driver, engine, clock = self.driver()
        calls = []

        def restart(grace):
            calls.append(('restart', grace))
            return dict(seconds=300.0, stop_s=5.5, exit_code=0, log_window=[1, 2])

        def kill():
            calls.append(('kill',))
            return dict(seconds=290.0, stop_s=0.2, exit_code=137, log_window=[3, 4])

        feed = iter(counters)
        last = [None]

        def stats(_driver):
            try:
                last[0] = next(feed)
            except StopIteration:
                pass
            return dict(last[0]) if last[0] else {}

        with mock.patch.object(replay, 'read_stats', stats):
            replay.scenario_nvme_restart(driver, sizes=sizes, restart=callbacks.get('restart', restart), kill=callbacks.get('kill', kill))
        return driver, calls, clock

    @staticmethod
    def rising(blocks=1000, steps=60):
        """Counters that show the writer storing the session: nothing, then the blocks, queue empty."""
        out = [dict(nvme_spill_blocks=0, nvme_queue_bytes=0, nvme_through_backlog_blocks=0, nvme_entries=0)]
        out += [dict(nvme_spill_blocks=blocks * 2, nvme_queue_bytes=0, nvme_through_backlog_blocks=0, nvme_entries=10)] * steps
        return out

    def test_without_the_callbacks_it_records_that_it_could_not_run(self):
        driver, _, _ = self.driver()
        replay.scenario_nvme_restart(driver)
        self.assertEqual(driver.events['nvme-restart'], dict(skipped=True, reason='no restart callbacks'))

    def test_the_graceful_drill_waits_for_the_drive_then_stops_with_the_platforms_grace_and_the_hard_one_kills_at_once(self):
        driver, calls, clock = self.run_scenario(self.rising())
        self.assertEqual(calls, [('restart', replay.NVME_STOP_GRACE_S), ('kill',)])
        stop, kill = driver.events['nvme-stop-8000'], driver.events['nvme-kill-8000']
        self.assertTrue(stop['graceful'] and stop['persisted'] and stop['build_ok'] and stop['return_ok'])
        self.assertEqual((stop['stop_s'], stop['exit_code']), (5.5, 0))
        self.assertFalse(kill['graceful'])
        self.assertNotIn('persisted', kill, 'the hard drill does not wait: it kills while the copy is under way')
        self.assertEqual(kill['exit_code'], 137)
        self.assertGreaterEqual(stop['persist_waited_s'], replay.NVME_PERSIST_POLL_S * 2)
        roles = [record['role'] for record in driver.records]
        self.assertIn('flood', roles, 'the kicks that keep the scheduler stepping')
        self.assertEqual(len([pair for pair in driver.pairs if pair['verdict'] == 'IDENTICAL']), 2, [(p['case'], p['verdict']) for p in driver.pairs])

    def test_a_drive_that_never_holds_the_session_times_out_and_the_drill_still_runs_marked_not_persisted(self):
        stuck = [dict(nvme_spill_blocks=0, nvme_queue_bytes=0, nvme_through_backlog_blocks=0, nvme_entries=0)] * 400
        driver, calls, clock = self.run_scenario(stuck)
        stop = driver.events['nvme-stop-8000']
        self.assertFalse(stop['persisted'])
        self.assertGreaterEqual(stop['persist_waited_s'], replay.NVME_PERSIST_TIMEOUT_S)
        self.assertEqual(calls[0], ('restart', replay.NVME_STOP_GRACE_S))

    def test_a_queue_that_has_not_drained_is_not_persisted(self):
        busy = [dict(nvme_spill_blocks=5000, nvme_queue_bytes=1 << 20, nvme_through_backlog_blocks=0, nvme_entries=1)] * 400
        driver, _, _ = self.run_scenario(busy)
        self.assertFalse(driver.events['nvme-stop-8000']['persisted'])

    def test_the_oracle_is_reset_by_each_restart(self):
        driver, engine, clock = self.driver()
        seen = []
        driver.oracle = replay.judge.Oracle(driver.oracle.capacity, sticky=driver.oracle.sticky, undrop=driver.oracle.undrop, carry=driver.oracle.carry)
        first = driver.oracle

        def restart(grace):
            seen.append(driver.oracle)
            return dict(seconds=1.0, stop_s=1.0, exit_code=0, log_window=[0, 0])

        with mock.patch.object(replay, 'read_stats', lambda d: dict(self.rising()[1])):
            replay.scenario_nvme_restart(driver, sizes=(8000,), restart=restart, kill=lambda: dict(seconds=1.0, stop_s=0.1, exit_code=137, log_window=[0, 0]))
        self.assertIs(seen[0], first)
        self.assertIsNot(driver.oracle, first)

    def test_the_container_gets_a_kill_and_an_exit_code_reader(self):
        calls = []

        def run(arguments, stdout=None, stderr=None, timeout=None):
            calls.append(arguments)

            class Result(object):
                returncode = 0
                stdout = b'137\n'

            return Result()

        container = replay.Container('c', run=run)
        container.kill()
        self.assertEqual(calls[-1], ['docker', 'kill', '--signal', 'KILL', 'c'])
        self.assertEqual(container.exit_code(), 137)
        self.assertEqual(calls[-1], ['docker', 'inspect', '-f', '{{.State.ExitCode}}', 'c'])


class RunnerRestartTests(unittest.TestCase):
    """Runner.restart with a grace and with a hard kill."""

    class Container(object):
        def __init__(self):
            self.calls = []

        def stop(self, *args):
            self.calls.append(('stop',) + args)

        def kill(self):
            self.calls.append(('kill',))

        def start(self):
            self.calls.append(('start',))

        def exit_code(self):
            return 143

    class Follower(object):
        def mark(self):
            return 5

        def last_time(self):
            return 't'

        def stop(self):
            pass

        def start(self, since=None):
            pass

    def runner(self):
        harness = gate_tests.Harness()
        runner = harness.runner(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, runner.results, True)
        runner.wait_ready = lambda client, container, seconds: None
        return runner

    def test_the_default_restart_is_what_it_was_and_the_grace_and_kill_are_new(self):
        runner = self.runner()
        container = self.Container()
        info = runner.restart(container, None, self.Follower())
        self.assertEqual(container.calls, [('stop',), ('start',)], 'the lifecycle drill\'s call is unchanged')
        self.assertEqual((info['exit_code'], sorted(info)), (143, ['exit_code', 'log_window', 'seconds', 'stop_s']))
        container = self.Container()
        runner.restart(container, None, self.Follower(), grace=10)
        self.assertEqual(container.calls, [('stop', 10), ('start',)])
        container = self.Container()
        runner.restart(container, None, self.Follower(), hard=True)
        self.assertEqual(container.calls, [('kill',), ('start',)])

    def test_a_container_without_an_exit_code_reader_gives_none(self):
        runner = self.runner()

        class Plain(self.Container):
            exit_code = None

        self.assertIsNone(runner.restart(Plain(), None, self.Follower())['exit_code'])


# -- (5) the plans over the fake engine -----------------------------------------------------------------------

def run_nvme(plan, results, document, **faults):
    harness = gate_tests.Harness(tier_gate.TierFakeEngine, document=document, **faults)
    runner = harness.runner(results)
    arms = gate.plan_arms(plan, FAKE, 'none', document)
    return gate.run_plan(plan, arms, runner, gate_tests.GOOD_ANCHOR), harness, runner


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results, True)

    def test_the_timed_plan_runs_as_the_host_only_control_on_a_profile_without_the_nvme_tier(self):
        result, _, _ = run_nvme('nvme-timed', self.results, tier_gate.tier_profiles())
        arm = result['arms']['nvme-timed']
        self.assertEqual(arm['verdict'], 'PASS', (arm['problems'], arm['not_exercised'], arm['lines'][-8:]))
        self.assertTrue(any(line.startswith('tier-timed-') for line in arm['lines']))

    def test_an_arm_that_asks_for_the_tier_and_never_sees_it_installed_does_not_pass(self):
        result, _, _ = run_nvme('nvme-timed', self.results, nvme_profiles())
        arm = result['arms']['nvme-timed']
        self.assertEqual(arm['verdict'], 'FAIL', (arm['problems'], arm['not_exercised']))
        self.assertTrue(any('QWEN_PREFIX_NVME=1 is set and no engaged line ([PINDIAG] prefix: nvme tier on) was logged' in text for text in arm['problems']), arm['problems'])
        self.assertTrue(any('no "nvme tier on" line' in text for text in arm['not_exercised']), arm['not_exercised'])
        self.assertTrue(any('no "nvme tier open" line' in text for text in arm['not_exercised']), arm['not_exercised'])

    def test_a_dead_drive_makes_the_arm_infra_and_starts_nothing(self):
        document = nvme_profiles()
        harness = gate_tests.Harness(tier_gate.TierFakeEngine, document=document)
        runner = harness.runner(self.results)
        runner.cache_drive = ('status', ())
        arms = gate.plan_arms('nvme-timed', FAKE, 'none', document)
        with mock.patch.object(cache, 'plan', side_effect=cache.CacheDriveError('the cache drive\'s controller state reads \'dead\', not live')):
            result = gate.run_plan('nvme-timed', arms, runner, gate_tests.GOOD_ANCHOR)
        self.assertEqual(result['verdict'], 'INFRA')
        self.assertIn('cannot be used', result['arms']['nvme-timed']['reason'])
        self.assertEqual([call for call in harness.calls if call[:3] == ['docker', 'run', '-d']], [], 'no container was started on a dead drive')

    def test_a_live_drive_reaches_docker_run_and_the_recorded_argv_is_redacted(self):
        document = nvme_profiles()
        harness = gate_tests.Harness(tier_gate.TierFakeEngine, document=document)
        runner = harness.runner(self.results)
        runner.cache_drive = ('status', ('0000:41:00.0',))
        arms = gate.plan_arms('nvme-timed', FAKE, 'none', document)
        planned = dict(arguments=['--device', '/dev/nvme3n1:/dev/c2-cache-drive:rwm', '--mount', 'type=bind,src=/sys/bus/pci/devices/0000:41:00.0/p2pmem/allocate,dst=/sys/bus/pci/devices/0000:41:00.0/p2pmem/allocate',
                                  '--security-opt', 'apparmor=unconfined', '-e', 'QWEN_PREFIX_NVME_DEVICE=/dev/c2-cache-drive', '-e', 'QWEN_P2PMEM_DEVICES=0000:41:00.0'],
                       lines=['R1 cache drive: controller state=live controllers=1 namespace_bytes=1000204886016 (resolved from the by-id link at run time)'], env={})
        seen = []
        with mock.patch.object(cache, 'plan', side_effect=lambda *spec: seen.append(spec) or planned):
            gate.run_plan('nvme-timed', arms, runner, gate_tests.GOOD_ANCHOR)
        self.assertEqual(seen, [('status', ('0000:41:00.0',))], 'resolved for the arm, from the specs')
        run = [call for call in harness.calls if call[:3] == ['docker', 'run', '-d']][0]
        self.assertIn('apparmor=unconfined', run)
        self.assertIn('QWEN_PREFIX_NVME_DEVICE=/dev/c2-cache-drive', run)
        self.assertLess(run.index('apparmor=unconfined'), run.index('--entrypoint'))
        with open(os.path.join(self.results, 'nvme-timed', 'docker-run.json'), encoding='utf-8') as handle:
            recorded = handle.read()
        self.assertNotIn('0000:41:00.0', recorded)
        self.assertNotIn('nvme3n1', recorded)
        self.assertIn('<drive>:/dev/c2-cache-drive:rwm', recorded)


if __name__ == '__main__':
    unittest.main()
