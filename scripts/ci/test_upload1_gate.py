"""The prefix gate's upload plans (tp4/upload-1, docs/tp4-upload1.md): tier-upload (audited, ~64k / 128k / 254k), tier-upload-timed and its control, the thread and call-size
sweeps; c2_prefix_gate, prefix_replay, prefix_markers, qwen_tier_upload and the real scheduler graft on the CPU.

* the wiring: every plan and arm is registered in the job parser and the gate, the derived kinds carry exactly the levers they name, the timed pair differs by the levers alone,
  the s5 profiles already carry the engine-start zero fill, and the derived profile of every arm on the real s5 profiles passes the contract's boot checks;
* the scenario at its sizes, and the dry-run on the real profiles;
* the judge (upload_findings) with fabricated logs and counters in its PASS, NOT_EXERCISED and FAIL shapes;
* the plans end to end over test_prefix_tier_gate's fake engine with the REAL graft and the threaded tier IO over the model graft's own adapter on a numpy device, so the whole
  path (derived env, wrapper, counters, markers, judge) runs.

Run at py 3.11: `py -3.11 -B -m unittest test_upload1_gate` from scripts/ci."""

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

import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import prefix_markers as pm  # noqa: E402
import prefix_replay as replay  # noqa: E402
import qwen_prefix_registry as registry_module  # noqa: E402
import qwen_tier_upload as upload  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import test_c2_prefix_gate as gate_tests  # noqa: E402
import test_prefix_replay as fakes  # noqa: E402
import test_prefix_tier_gate as tier_gate  # noqa: E402
import test_qwen_tier_upload as tier_upload_tests  # noqa: E402

REAL_PROFILES = os.path.join(HERE, 'qwen_c2_profiles.json')
SHIP = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s5'
AUDIT = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-s5-audit'
FAKE = tier_gate.FAKE
IO, CHUNK, VERIFY = upload.ENV_IO_THREADS, upload.ENV_IO_CHUNK, upload.ENV_VERIFY_THREADS
ZEROS, ZEROS_AUDIT = 'QWEN_FAST_DEVICE_ZEROS', 'QWEN_FAST_DEVICE_ZEROS_AUDIT'

# plan -> (the derived kind, the scenario, which real profile the plan is run on)
PLANS = {
    'tier-upload': ('upload1', 'tier_upload', AUDIT),
    'tier-upload-timed': ('uptime', 'tier_upload_timed', SHIP),
    'tier-upload-timed-control': ('tiertime', 'tier_upload_timed', SHIP),
    'tier-upload-timed-8': ('uptime8', 'tier_upload_timed', SHIP),
    'tier-upload-timed-c128': ('uptimec128', 'tier_upload_timed', SHIP),
    'tier-upload-timed-c2048': ('uptimec2048', 'tier_upload_timed', SHIP),
}


def real_profiles():
    with open(REAL_PROFILES, encoding='utf-8') as handle:
        return json.load(handle)


# -- (1) the wiring -------------------------------------------------------------------------------------------

class WiringTests(unittest.TestCase):
    def test_every_plan_is_registered_in_the_job_parser_and_the_gate_as_one_arm_of_the_same_name(self):
        for plan, (kind, scenario, _profile) in PLANS.items():
            with self.subTest(plan=plan):
                self.assertIn(plan, job.PREFIX_PLANS)
                self.assertIn(plan, gate.PLANS)
                self.assertEqual([entry[0] for entry in gate.PLAN_ARMS[plan]], [plan])
                entry = gate.PLAN_ARMS[plan][0]
                self.assertEqual((entry[1], entry[3]), (scenario, kind if kind != 'tiertime' else 'tiertime'))
                self.assertIn(scenario, replay.SCENARIOS)
                self.assertIn(plan, [name for name in gate.PLANS])

    def test_the_audited_scenario_is_judged_and_the_timed_one_is_timed(self):
        self.assertIn('tier_upload_timed', gate.TIMED_SCENARIOS)
        self.assertNotIn('tier_upload', gate.TIMED_SCENARIOS)
        for scenario in ('tier_upload', 'tier_upload_timed'):
            self.assertIn(scenario, gate.TIER_SCENARIOS)
            self.assertIn(scenario, gate.TIER_RESTORE_SCENARIOS, 'a session must be seen to leave the pool and come back')

    def test_the_kinds_carry_exactly_what_they_name(self):
        tiertime = gate.DERIVED['tiertime']['env']
        self.assertEqual(gate.DERIVED['upload1']['env'], {IO: '4', VERIFY: '4'})
        self.assertEqual(gate.DERIVED['uptime']['env'], dict(tiertime, **{IO: '4', VERIFY: '4'}))
        self.assertEqual(gate.DERIVED['uptime8']['env'], dict(tiertime, **{IO: '8', VERIFY: '8'}))
        self.assertEqual(gate.DERIVED['uptimec128']['env'], dict(tiertime, **{IO: '4', VERIFY: '4', CHUNK: '128'}))
        self.assertEqual(gate.DERIVED['uptimec2048']['env'], dict(tiertime, **{IO: '4', VERIFY: '4', CHUNK: '2048'}))
        for kind in ('uptime', 'uptime8', 'uptimec128', 'uptimec2048'):
            self.assertEqual(gate.DERIVED[kind]['env']['QWEN_PREFIX_HOST_TIER_AUDIT'], '0', 'unaudited')
        self.assertEqual(set(gate.UPLOAD_KINDS), {'upload1', 'uptime', 'uptime8', 'uptimec128', 'uptimec2048'})
        self.assertNotIn('upzeros', gate.DERIVED, 'the engine-start zero fill is in the s5 profiles: nothing to derive')

    def test_the_s5_profiles_already_carry_the_engine_start_levers_so_no_arm_derives_them(self):
        document = real_profiles()['profiles']
        for name in (SHIP, AUDIT):
            self.assertEqual(document[name]['env'].get(ZEROS), '1', name)
        self.assertEqual(document[AUDIT]['env'].get(ZEROS_AUDIT), '1', 'the audited twin reads the zero fill back at every engine start')
        self.assertNotIn(ZEROS_AUDIT, document[SHIP]['env'])
        for kind in gate.UPLOAD_KINDS:
            self.assertFalse({ZEROS, ZEROS_AUDIT} & set(gate.DERIVED[kind]['env']))

    def test_the_timed_pair_differs_by_the_levers_and_nothing_else(self):
        control, treated = gate.DERIVED['tiertime']['env'], gate.DERIVED['uptime']['env']
        self.assertEqual({name: value for name, value in treated.items() if control.get(name) != value}, {IO: '4', VERIFY: '4'})
        self.assertEqual(set(control) - set(treated), set())

    def test_the_names_are_the_modules(self):
        self.assertEqual((gate.UPLOAD_IO_FLAG, gate.UPLOAD_CHUNK_FLAG, gate.UPLOAD_VERIFY_FLAG), (IO, CHUNK, VERIFY))
        self.assertEqual({IO, CHUNK, VERIFY}, set(upload.NAMES))
        self.assertEqual(ZEROS, 'QWEN_FAST_DEVICE_ZEROS')

    def test_the_scenario_runs_the_tier_timed_cycle_at_the_upload_sizes_under_the_tier_timed_event_names(self):
        self.assertEqual(replay.TIER_UPLOAD_SIZES, (64000, 128000, 254000))
        document = tier_gate.tier_profiles()
        engine = tier_gate.TierFakeEngine(profile=document['profiles'][FAKE], name=FAKE)
        driver = fakes.sticky_driver(engine, 'tier-upload')
        replay.scenario_tier_upload(driver, pool_tokens=engine.num_blocks * replay.BLOCK if hasattr(replay, 'BLOCK') else engine.num_blocks * 64)
        names = sorted(name for name in driver.events if name.startswith('tier-timed-') and name[len('tier-timed-'):].isdigit())
        self.assertEqual(names, ['tier-timed-128000', 'tier-timed-254000', 'tier-timed-64000'])
        self.assertTrue(all(pair['verdict'] == 'IDENTICAL' for pair in driver.pairs), [(p['case'], p['verdict']) for p in driver.pairs])
        self.assertEqual(len(driver.pairs), 3)

    def test_the_sizes_are_a_parameter(self):
        engine = tier_gate.TierFakeEngine(profile=tier_gate.tier_profiles()['profiles'][FAKE], name=FAKE)
        driver = fakes.sticky_driver(engine, 'tier-upload')
        replay.scenario_tier_upload(driver, pool_tokens=engine.num_blocks * 64, sizes=(32000,))
        self.assertEqual([name for name in driver.events if name.startswith('tier-timed-') and name[len('tier-timed-'):].isdigit()], ['tier-timed-32000'])

    def test_a_profile_without_the_tier_has_none_of_these_arms_to_run(self):
        document = gate_tests.sticky_profiles()
        for plan in PLANS:
            self.assertEqual(gate.plan_arms(plan, FAKE, 'none', document), [], plan)
            self.assertTrue(all('names no QWEN_PREFIX_HOST_TIER_GIB' in why for _, why in gate.not_applicable(plan, FAKE, document)), plan)

    def test_the_derived_profile_of_every_arm_on_the_real_s5_profiles_passes_the_boot_checks(self):
        from test_octo3_contract import boot_problems

        document = real_profiles()
        for plan, (kind, scenario, profile) in PLANS.items():
            with self.subTest(plan=plan):
                arms = gate.plan_arms(plan, profile, 'none', document)
                self.assertEqual([arm['arm'] for arm in arms], [plan])
                arm = arms[0]
                served = arm['derived']['profiles'][arm['served']] if arm['derived'] else document['profiles'][arm['served']]
                self.assertEqual(boot_problems(copy.deepcopy(served)) + contract.prefix_policy_problems(served), [])
                self.assertTrue(arm['tier'], 'the arm reads the profile\'s tier settings')
                self.assertEqual(arm['kind'], kind)
                env = served['env']
                wanted = gate.DERIVED[kind]['env'] if kind != 'tiertime' else {}
                self.assertEqual({name: env.get(name) for name in (IO, VERIFY, CHUNK) if name in env}, {name: wanted[name] for name in (IO, VERIFY, CHUNK) if name in wanted})

    def test_the_dry_run_on_the_real_profiles_plans_the_arm_with_the_levers_in_its_derived_profile(self):
        for plan, (kind, scenario, profile) in PLANS.items():
            with self.subTest(plan=plan):
                results = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, results, True)
                lines = []
                code = gate.main(['--image', 'img', '--profiles', REAL_PROFILES, '--results', results, '--profile', profile, '--plan', plan, '--baseline', 'none', '--dry-run',
                                  '--cards', 'quad'], devices=['/a', '/b', '/c', '/d'], log=lines.append)
                self.assertEqual(code, 0, lines)
                rows = [json.loads(line) for line in lines if line.startswith('{') and '"docker"' in line]
                self.assertEqual([row['arm'] for row in rows], [plan])
                self.assertEqual(rows[0]['served'], '%s+%s' % (profile, kind))
                self.assertTrue(any('profiles.json' in token for token in rows[0]['docker']), 'the derived profile is mounted')
                arm = gate.plan_arms(plan, profile, 'none', real_profiles())[0]
                env = arm['derived']['profiles'][rows[0]['served']]['env']
                for name, value in gate.DERIVED[kind]['env'].items():
                    self.assertEqual(env.get(name), value, (plan, name))

    def test_a_traffic_profile_derived_for_the_timed_arms_keeps_its_waiver_which_is_what_admits_the_levers(self):
        document = real_profiles()
        arm = gate.plan_arms('tier-upload-timed', SHIP, 'none', document)[0]
        served = arm['derived']['profiles'][arm['served']]
        self.assertIn(contract.TRAFFIC_WAIVER, served)
        self.assertNotIn('gate_only', served)
        self.assertEqual(contract.prefix_policy_problems(served), [])
        stripped = copy.deepcopy(served)
        stripped[contract.TRAFFIC_WAIVER]['levers'].pop('QWEN_PREFIX_HOST_TIER_GIB')
        self.assertTrue(any('upload levers' in problem for problem in contract.prefix_policy_problems(stripped)))


# -- (2) the markers ------------------------------------------------------------------------------------------

class MarkerTests(unittest.TestCase):
    LINES = ['2026-10-21T01:00:00.000000001Z (EngineCore pid=9) INFO [PINDIAG] prefix: host tier upload on io_threads=4 io_chunk_blocks=128 verify_threads=4 io_wrapped=1',
             '2026-10-21T01:00:01.000000001Z (EngineCore pid=9) INFO [PINDIAG] prefix: host tier upload fell back: the tier IO is not the threaded one (io_threads=4)']

    def test_both_lines_are_read(self):
        scanned = pm.scan(self.LINES)
        self.assertEqual([(e['io_threads'], e['io_chunk_blocks'], e['verify_threads'], e['io_wrapped']) for e in scanned['tier_upload_on']], [(4, 128, 4, 1)])
        self.assertEqual(len(scanned['tier_upload_trouble']), 1)
        self.assertIn('fell back', scanned['tier_upload_trouble'][0]['line'])
        self.assertEqual(pm.scan([])['tier_upload_on'], [])

    def test_the_lines_are_the_ones_the_graft_writes(self):
        self.assertEqual(pm.TIER_UPLOAD_ON, upload.MARKER + ' on ')
        self.assertTrue(upload.ON_LINE.startswith(pm.TIER_UPLOAD_ON.rstrip()))


# -- (3) the judge -------------------------------------------------------------------------------------------

def arm_of(**env):
    return dict(arm='tier-upload-timed', scenario='tier_upload_timed', kind='uptime', served_env=dict(env), tier=dict(gib=48.0))


STATS = dict(tier_io_write_blocks=3841, tier_io_write_ms=420.0, tier_io_write_device_ms=380.0, tier_io_write_stage_ms=600.0,
             tier_io_read_blocks=3841, tier_io_read_ms=500.0, tier_io_read_device_ms=450.0, tier_io_read_stage_ms=300.0,
             tier_digest_checks=240, tier_verify_batches=8, tier_verify_blocks=240, tier_verify_ms=40.0)


class JudgeTests(unittest.TestCase):
    ON = [dict(index=3, io_threads=4, io_chunk_blocks=0, verify_threads=4, io_wrapped=1)]

    def judge(self, scanned=None, stats=STATS, **env):
        scanned = dict(tier_upload_on=self.ON, tier_upload_trouble=[]) if scanned is None else scanned
        return gate.upload_findings(arm_of(**(env or {IO: '4', VERIFY: '4'})), scanned, stats)

    def test_a_clean_arm_passes_and_prints_the_overlap(self):
        problems, missing, lines = self.judge()
        self.assertEqual((problems, missing), ([], []))
        text = '\n'.join(lines)
        self.assertIn('upload levers: {"io_threads": 4', text)
        self.assertIn('threaded restore writes: 3841 blocks in 420.0 ms wall; 380.0 ms inside device calls, 600.0 ms of staging work on the workers (serial would be 980.0 ms)', text)
        self.assertIn('threaded spill reads: 3841 blocks in 500.0 ms wall', text)
        self.assertIn('parallel digest checks: 8 batches, 240 blocks, 40.0 ms wall', text)

    def test_a_missing_install_line_is_not_exercised(self):
        problems, missing, _ = self.judge(scanned=dict(tier_upload_on=[], tier_upload_trouble=[]))
        self.assertEqual(problems, [])
        self.assertTrue(any('no "host tier upload on" line' in text for text in missing), missing)

    def test_a_fell_back_line_fails(self):
        problems, _, _ = self.judge(scanned=dict(tier_upload_on=self.ON, tier_upload_trouble=[dict(index=9, line='[PINDIAG] prefix: host tier upload fell back: x')]))
        self.assertEqual(len(problems), 1)
        self.assertIn('fell back', problems[0])

    def test_a_restore_that_did_not_go_through_the_threaded_io_is_not_exercised(self):
        _, missing, _ = self.judge(stats=dict(STATS, tier_io_write_blocks=0))
        self.assertTrue(any('tier_io_write_blocks is 0' in text and 'restore' in text for text in missing), missing)
        _, missing, _ = self.judge(stats=dict(STATS, tier_io_read_blocks=0))
        self.assertTrue(any('tier_io_read_blocks is 0' in text and 'spill' in text for text in missing), missing)

    def test_digest_checks_that_never_reached_the_verifier_are_not_exercised(self):
        _, missing, _ = self.judge(stats=dict(STATS, tier_verify_batches=0))
        self.assertTrue(any('did not run on the verifier' in text for text in missing), missing)
        _, missing, _ = self.judge(stats=dict(STATS, tier_verify_batches=0, tier_digest_checks=0))
        self.assertEqual(missing, [], 'a sampled restore that checked nothing owes no batch')

    def test_a_verify_only_arm_asks_nothing_of_the_io_counters(self):
        _, missing, _ = self.judge(stats=dict(STATS, tier_io_write_blocks=0, tier_io_read_blocks=0), **{VERIFY: '4'})
        self.assertEqual(missing, [])

    def test_no_stats_export_is_not_this_judges_to_report(self):
        problems, missing, _ = self.judge(stats=None)
        self.assertEqual((problems, missing), ([], []))


# -- (4) the plans end to end -------------------------------------------------------------------------------------

class UploadFakeEngine(tier_gate.TierFakeEngine):
    """TierFakeEngine with the upload levers: the registry reads the profile's lever names, and the tier IO is the model graft's adapter over a numpy device, so the registry wraps
    it in the threaded IO and the restore, the spill and the audit's read-back go through it."""

    def make_registry(self):
        registry = super().make_registry()
        env = self.profile['env']
        environ = {name: env[name] for name in upload.NAMES if name in env}
        if not environ:
            return registry
        base = dict(tier_gate.tier_environ(self.BLOCKS, int(self.store_gib * (1 << 30)), QWEN_PREFIX_HOST_TIER_AUDIT=env.get('QWEN_PREFIX_HOST_TIER_AUDIT', '0'),
                                           QWEN_PREFIX_HOST_TIER_VERIFY=env.get('QWEN_PREFIX_HOST_TIER_VERIFY', 'sample'), QWEN_PREFIX_HOST_TIER_OFF_PATH=self.tier_off_path),
                    QWEN_PREFIX_STORE_GIB=str(self.store_gib))
        base.update(environ)
        return tier_gate.graft.PrefixRegistry(environ=base)

    def attach_extras(self):
        tier_upload_tests.AdapterKvIO.pool_blocks = self.num_blocks + 16
        with mock.patch.object(tier_gate, 'FakeKvIO', tier_upload_tests.AdapterKvIO):
            super().attach_extras()


def upload_profiles(**env):
    document = tier_gate.tier_profiles()
    document['profiles'][FAKE]['env'].update(env)
    return document


def run_plan(plan, results, document=None, **faults):
    document = document or upload_profiles()
    harness = gate_tests.Harness(UploadFakeEngine, document=document, **faults)
    runner = harness.runner(results)
    arms = gate.plan_arms(plan, FAKE, 'none', document)
    return gate.run_plan(plan, arms, runner, gate_tests.GOOD_ANCHOR), harness, runner


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results, True)

    def test_the_audited_plan_restores_through_the_threaded_io_and_passes(self):
        result, harness, _ = run_plan('tier-upload', self.results)
        arm = result['arms']['tier-upload']
        self.assertEqual(arm['verdict'], 'PASS', (arm['problems'], arm['not_exercised'], arm['lines'][-14:]))
        text = '\n'.join(arm['lines'])
        self.assertIn('upload levers: ', text)
        self.assertIn('threaded restore writes: ', text)
        self.assertIn('threaded spill reads: ', text)
        self.assertIn('parallel digest checks: ', text)
        self.assertEqual(len([line for line in arm['lines'] if line.startswith('tier-timed-') and 'restore ' in line]), 3, arm['lines'])
        for size in ('64000', '128000', '254000'):
            self.assertTrue(any(line.startswith('tier-timed-' + size) for line in arm['lines']), size)
        for engine in harness.engines:
            self.assertTrue(engine.check_kv(), 'every cached block, restored ones included, holds the bytes its hash stands for')
            self.assertIsInstance(engine.registry.tier_io, upload.ThreadedTierIO)
            self.assertGreater(engine.registry.stats['tier_io_write_blocks'], 0)

    def test_the_timed_pair_runs_and_the_control_has_no_upload_machinery(self):
        timed, harness, _ = run_plan('tier-upload-timed', self.results)
        self.assertEqual(timed['arms']['tier-upload-timed']['verdict'], 'PASS', (timed['arms']['tier-upload-timed']['problems'], timed['arms']['tier-upload-timed']['not_exercised']))
        self.assertTrue(all(isinstance(engine.registry.tier_io, upload.ThreadedTierIO) for engine in harness.engines))
        control_results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, control_results, True)
        control, harness, _ = run_plan('tier-upload-timed-control', control_results)
        arm = control['arms']['tier-upload-timed-control']
        self.assertEqual(arm['verdict'], 'PASS', (arm['problems'], arm['not_exercised']))
        for engine in harness.engines:
            self.assertNotIsInstance(engine.registry.tier_io, upload.ThreadedTierIO)
            self.assertFalse(set(registry_module.UPLOAD_STAT_NAMES) & set(engine.registry.stats))
        self.assertFalse(any('threaded restore writes' in line for line in arm['lines']), 'the control prints no overlap')

    def test_the_sweeps_pass_and_name_their_call_size(self):
        for plan, chunk in (('tier-upload-timed-c128', 128), ('tier-upload-timed-8', 0)):
            with self.subTest(plan=plan):
                results = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, results, True)
                result, harness, _ = run_plan(plan, results)
                arm = result['arms'][plan]
                self.assertEqual(arm['verdict'], 'PASS', (arm['problems'], arm['not_exercised']))
                io = harness.engines[-1].registry.tier_io
                self.assertEqual(io.chunk_blocks, chunk)
                self.assertEqual(io.threads, 8 if plan.endswith('-8') else 4)

    def test_the_derived_profile_of_the_audited_arm_is_kept_in_the_results(self):
        run_plan('tier-upload', self.results)
        with open(os.path.join(self.results, 'tier-upload', 'profiles.json'), encoding='utf-8') as handle:
            derived = json.load(handle)
        env = derived['profiles'][FAKE + '+upload1']['env']
        self.assertEqual({name: env[name] for name in (IO, VERIFY)}, {IO: '4', VERIFY: '4'})


if __name__ == '__main__':
    unittest.main()
