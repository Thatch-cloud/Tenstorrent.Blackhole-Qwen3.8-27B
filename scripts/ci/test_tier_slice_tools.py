"""The sliced restore's host-side pieces (docs/prefix-store-hygiene.md "Sliced restore"): the stall breakdown of a log, the cost model, the gate's plans and judge for the flag, the contract's rules,
the TS job pack, and the whole tier-concurrent plan end to end over the tier fake with the flag on (the returning request held between slices while the seats decode).
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import prefix_concurrent as conc  # noqa: E402
import prefix_markers as markers  # noqa: E402
import prefix_replay as replay  # noqa: E402
import qwen_prefix_registry as registry_module  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import test_c2_prefix_gate as gate_tests  # noqa: E402
import test_prefix_tier_gate as tiers  # noqa: E402
import tier_slice_model as model  # noqa: E402
import tier_stall_breakdown as breakdown  # noqa: E402

FAKE = tiers.FAKE
REAL_PROFILES = HERE / 'qwen_c2_profiles.json'
S4_AUDIT = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-s4-audit'
S4_TRAFFIC = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s4'
SLICE = registry_module.ENV_TIER_RESTORE_SLICE
SLICE_MS = registry_module.ENV_TIER_RESTORE_SLICE_MS


# -- the breakdown ---------------------------------------------------------------------------------------------------

def stamp(seconds):
    whole = int(seconds)
    return '2026-10-11T11:05:%02d.%03d000000Z' % (whole, int((seconds - whole) * 1000))


REQUEST = 'chatcmpl-pfx-tier-concurrent-0036-hit-a8fe2ebe'


def synthetic_log():
    """The shape of TC1's log (run 38133584894) around the returning request, to the millisecond of its phases."""
    rows = [
        (20.000, '(EngineCore pid=67) 2026-10-11 11:05:20.000 | INFO     | packed_verifier:diagnostic:224 - [PACKED-PHASE] round=146 users=8 bind_ms=0.03'),
        (20.179, '(EngineCore pid=67) INFO | serving_prefill_admission:_log:364 - [PINDIAG] kv reservation fit request=%s reserved=3970' % REQUEST),
        (27.100, '(EngineCore pid=67) [PINDIAG] prefix: host tier spill blocks=512 bytes=1140850688 ms=887.7 held=15539634176'),
        (30.000, '(EngineCore pid=67) [PINDIAG] prefix: host tier restore req=%s blocks=3905 bytes=8701214720 ms=6321.0 from=0 to=249920' % REQUEST),
        (32.142, '(EngineCore pid=67) dflash_device:pindiag:46 - [PINDIAG] lever N route req=%s start=249856 end=253442 prompt=253442 final=1 wrote_slot=1 ms=1950.4 programs=1410->1414' % REQUEST),
        (34.233, '(EngineCore pid=67) dflash_device:pindiag:46 - [PINDIAG] sticky engine built req=%s ms=1837.5 frontier=249856 prompt=253442 kind=rebind' % REQUEST),
        (34.848, '(EngineCore pid=67) packed_verifier:diagnostic:224 - [PACKED-PHASE] round=147 users=8 bind_ms=0.03'),
        (42.926, '(EngineCore pid=67) packed_verifier:diagnostic:224 - [PACKED-PHASE] round=148 users=8 bind_ms=0.03'),
        (43.500, '(EngineCore pid=67) packed_verifier:diagnostic:224 - [PACKED-PHASE] round=149 users=8 bind_ms=0.03'),
    ]
    return ['%s %s' % (stamp(when), text) for when, text in rows]


class BreakdownTests(unittest.TestCase):
    def test_the_phases_of_the_stall_are_read_from_the_log(self):
        result = breakdown.timeline(synthetic_log())
        self.assertEqual(result['request'], REQUEST)
        phases = dict(result['phases'])
        self.assertAlmostEqual(phases['last round before -> admission'], 0.179, places=2)
        self.assertAlmostEqual(phases['admission -> restore begins (prelims + digests)'], 30.0 - 6.321 - 20.179, places=2)
        self.assertAlmostEqual(phases['restore work (allocate, spill, write, audit)'], 6.321, places=3)
        self.assertAlmostEqual(phases['restore ends -> tail prefill done'], 2.142, places=2)
        self.assertAlmostEqual(phases['tail prefill done -> engine built (slot adoption, tap window, rebind)'], 2.091, places=2)
        self.assertAlmostEqual(phases['engine built -> first packed round'], 0.615, places=2)
        self.assertAlmostEqual(result['freeze'], 14.848, places=2)
        self.assertEqual([(round(length, 1), first, second) for length, first, second in result['gaps']], [(14.8, 146, 147), (8.1, 147, 148)])

    def test_it_renders_the_timeline_and_the_totals(self):
        lines = breakdown.render(breakdown.timeline(synthetic_log()))
        text = '\n'.join(lines)
        for needle in ('returning request ' + REQUEST, 'restore ends: 3905 blocks, 8.70 GB', 'sticky engine built (1838 ms, kind=rebind)', 'host tier spill of 512 displaced blocks',
                       "the seats' freeze (last packed round before to first after): 14.84", 'a gap of 8.07', 'between packed rounds 147 and 148'):
            self.assertIn(needle, text)

    def test_a_sliced_restore_line_names_its_slices(self):
        rows = synthetic_log()
        rows[3] = '%s (EngineCore pid=67) [PINDIAG] prefix: host tier restore req=%s blocks=3905 bytes=8701214720 ms=1100.0 from=0 to=249920 slices=9 max_slice_ms=210.5 held_steps=8' % (
            stamp(30.0), REQUEST)
        text = '\n'.join(breakdown.render(breakdown.timeline(rows)))
        self.assertIn('9 slices (max 210.5 ms), held 8 steps', text)

    def test_a_log_without_a_restore_is_said_so_and_a_missing_file_refused(self):
        out = []
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        path = os.path.join(tmp, 'server.log')
        with open(path, 'w') as handle:
            handle.write('nothing here\n')
        self.assertEqual(breakdown.main([path], out=out.append), 1)
        self.assertIn('no "host tier restore" line', out[0])
        self.assertEqual(breakdown.main([os.path.join(tmp, 'absent')], out=out.append), 2)
        with open(path, 'w') as handle:
            handle.write('\n'.join(synthetic_log()) + '\n')
        out[:] = []
        self.assertEqual(breakdown.main([path, '--request', 'a8fe2ebe', '--gap', '5'], out=out.append), 0)
        self.assertEqual(len([line for line in out if line.startswith('a gap of')]), 2)


# -- the model -------------------------------------------------------------------------------------------------------

class ModelTests(unittest.TestCase):
    def test_the_slices_are_a_probe_then_the_budget_over_the_cost_capped(self):
        self.assertEqual(model.sizes(100, 2.0, 20.0, 64), [8, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2])
        self.assertEqual(model.sizes(40, 0.01, 1000.0, 16), [8, 16, 16])
        self.assertEqual(model.sizes(5, 1.0, 100.0, 16), [5])
        self.assertEqual(sum(model.sizes(3905, 0.3, 200.0, 512)), 3905)

    def test_the_model_reproduces_what_was_measured(self):
        whole = model.row(3905, model.RATES['audited'], 200.0, 512)
        self.assertAlmostEqual(whole['whole_stall_s'], 10.4, delta=0.6, msg='TC1: digests 3.63 s, restore 6.32 s, a round 0.49 s')
        quiet = model.row(3841, model.RATES['unaudited'], 200.0, 512)
        self.assertAlmostEqual(quiet['whole_stall_s'] - model.RATES['unaudited']['round_ms'] / 1000.0, 0.961 + 0.142, delta=0.03, msg='T3: 961 ms and the spill of 512 blocks')

    def test_slicing_cuts_the_restore_attributable_wait_to_a_slice_and_costs_the_returning_request_rounds(self):
        for blocks in (1985, 3905):
            for slice_ms in (100.0, 200.0, 400.0):
                r = model.row(blocks, model.RATES['unaudited'], slice_ms, 512)
                self.assertLess(r['seat_wait_s'], r['whole_stall_s'])
                self.assertLessEqual(r['seat_wait_s'], (slice_ms + model.RATES['unaudited']['round_ms'] + 1.0) / 1000.0 + 0.001)
                self.assertAlmostEqual(r['ttft_extra_s'], (r['slices'] - 1) * 0.150, places=6)
        self.assertEqual(model.row(3905, model.RATES['unaudited'], 200.0, 512)['slices'], 12)

    def test_the_table_and_the_cpu_timing_print(self):
        out = []
        self.assertEqual(model.main(['--cpu'], out=out.append), 0)
        text = '\n'.join(out)
        self.assertIn('unaudited:', text)
        self.assertIn('audited:', text)
        self.assertIn('the returning request\'s tail prefill and engine build', text)
        self.assertIn('sha256', text)
        digest, compare = model.cpu_cost(repeat=1)
        self.assertGreater(digest, 0)
        self.assertGreater(compare, 0)


# -- the gate: derived env, plans, judge, contract, markers ---------------------------------------------------------------

class GateTests(unittest.TestCase):
    def test_the_plans_are_registered_and_need_the_tier(self):
        for plan, scenario in (('tier-concurrent-sliced', 'tier_concurrent'), ('tier-concurrent-timed-sliced', 'tier_concurrent_timed')):
            self.assertIn(plan, job.PREFIX_PLANS)
            self.assertIn(plan, gate.PLANS)
            self.assertEqual([entry[:4] for entry in gate.PLAN_ARMS[plan]], [(plan, scenario, 'prefix', 'slice')])
            self.assertNotIn(plan, gate.TIER_OPTIONAL_ARMS, 'a sliced restore with no tier is inert: the arm is not applicable there')
        document = json.loads(REAL_PROFILES.read_text(encoding='utf-8'))
        self.assertEqual(gate.plan_arms('tier-concurrent-timed-sliced', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic', 'none', document), [])
        self.assertTrue(gate.not_applicable('tier-concurrent-timed-sliced', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic', document))

    def test_the_arm_is_the_ship_4_profile_plus_the_two_names_and_the_contract_accepts_it(self):
        document = json.loads(REAL_PROFILES.read_text(encoding='utf-8'))
        for profile, plan in ((S4_TRAFFIC, 'tier-concurrent-timed-sliced'), (S4_AUDIT, 'tier-concurrent-sliced')):
            arm = gate.plan_arms(plan, profile, 'none', document)[0]
            derived = arm['derived']['profiles'][arm['served']]
            self.assertEqual((derived['env'][SLICE], derived['env'][SLICE_MS]), (str(gate.SLICE_BLOCKS), str(gate.SLICE_MS)))
            self.assertEqual(contract.prefix_policy_problems(derived), [], profile)
            self.assertEqual((arm['kind'], bool(arm['tier'])), ('slice', True))
            base = document['profiles'][profile]
            self.assertNotIn(SLICE, base['env'], 'the profile itself names nothing: the derivation adds it')
        self.assertEqual(registry_module.restore_slice_config(gate.slice_env({})), (gate.SLICE_BLOCKS, float(gate.SLICE_MS)), 'the gate\'s numbers pass the engine\'s strict parser')

    def test_the_contract_rules_for_the_two_names(self):
        base = json.loads(REAL_PROFILES.read_text(encoding='utf-8'))['profiles'][S4_AUDIT]

        def problems(**env):
            profile = json.loads(json.dumps(base))
            profile['env'].update(env)
            return contract.prefix_policy_problems(profile)

        self.assertEqual(problems(**{SLICE: '512', SLICE_MS: '200'}), [])
        for env, needle in (({SLICE: '1e3'}, 'digits only'), ({SLICE: '70000'}, 'digits only'), ({SLICE_MS: '50'}, 'inert'), ({SLICE: '64', SLICE_MS: 'inf'}, 'milliseconds'),
                            ({SLICE: '64', SLICE_MS: '0'}, 'milliseconds')):
            with self.subTest(env=env):
                found = problems(**env)
                self.assertTrue(found and needle in ' '.join(found), found)
        traffic = json.loads(json.dumps(json.loads(REAL_PROFILES.read_text(encoding='utf-8'))['profiles'][S4_TRAFFIC]))
        traffic['env'][SLICE] = '512'
        self.assertEqual(contract.prefix_policy_problems(traffic), [], 'the candidate\'s waiver names the tier: the sliced restore rides on it')
        bare = json.loads(json.dumps(base))
        for name in list(bare['env']):
            if name.startswith('QWEN_PREFIX_HOST_TIER') or name.startswith('QWEN_PREFIX_CKPT'):
                del bare['env'][name]
        bare['env'][SLICE] = '512'
        self.assertTrue(any('inert' in text for text in contract.prefix_policy_problems(bare)))
        profile = dict(base, name=S4_AUDIT)
        inherited = {name: '7' for name in contract.PREFIX_SLICE_NAMES}
        self.assertFalse(set(contract.apply_environment(json.loads(json.dumps(profile)), dict(inherited))) & set(contract.PREFIX_SLICE_NAMES))

    def test_the_marker_scan_and_the_registry_agree_on_the_lines(self):
        lines = ['[PINDIAG] prefix: host tier restore slice on blocks=512 ms=200',
                 '[PINDIAG] prefix: host tier restore slice req=chatcmpl-x-hit-ab blocks=64 done=64 of 3905 ms=31.2']
        scan = markers.scan(lines)
        self.assertEqual((scan['tier_slice_on'][0]['blocks'], scan['tier_slice_on'][0]['ms']), (512, 200))
        self.assertEqual(scan['tier_slices'][0]['tag'], 'x-hit-ab')

    def findings(self, restore, on=True, trouble=()):
        event = dict(return_tag='ret', return_ok=True)
        scanned = dict(tier_slice_on=[dict(blocks=512, ms=200)] if on else [], tier_slice_trouble=[dict(index=1, line=text) for text in trouble],
                       tier_restores=[dict(restore, tag='ret')] if restore is not None else [])
        return gate.slice_findings(dict(arm='a', kind='slice'), event, scanned)

    def test_the_judge_wants_the_install_line_the_slices_and_no_slice_over_a_second(self):
        good = dict(blocks=3905, slices=9, max_slice_ms=210.5, held_steps=8, ms=1100.0)
        problems, missing, lines = self.findings(good)
        self.assertEqual((problems, missing), ([], []))
        self.assertTrue(any('3905 blocks in 9 slices, the longest slice 210.5 ms, the request held for 8 steps' in line for line in lines), lines)
        self.assertTrue(any('slice 1001' in text or 'over the 1000 ms a slice may take' in text for text in self.findings(dict(good, max_slice_ms=1001.0))[0]))
        self.assertTrue(any('no "host tier restore slice on" line' in text for text in self.findings(good, on=False)[1]))
        self.assertTrue(any('carries no slices= field' in text for text in self.findings(dict(blocks=3905, ms=900.0))[1]))
        self.assertTrue(any('held_steps=0' in text for text in self.findings(dict(good, held_steps=0))[1]))
        problems, _, _ = self.findings(good, trouble=['[PINDIAG] prefix: host tier restore slice fell back: no one-fresh-prefill admission wrapper'])
        self.assertTrue(problems and 'fell back' in problems[0])
        problems, _, _ = self.findings(good, trouble=['[PINDIAG] prefix: host tier restore slice aborted req=a after 3 of 9 blocks: the request ended'])
        self.assertEqual(problems, [], 'a job that ended with its request is not a defect')
        problems, _, _ = self.findings(good, trouble=['[PINDIAG] prefix: host tier restore slice aborted req=a after 3 of 9 blocks: a slice failed'])
        self.assertTrue(problems)


# -- the whole plan over the tier fake -----------------------------------------------------------------------------------

class SlicedFakeEngine(tiers.TierFakeEngine):
    """The tier fake with the sliced restore: the registry reads the profile's slice names, the install line is logged, and a step that has seats running asks the graft whether
    to hold the head candidate (what serving_prefill_admission does), a held step admitting nothing."""

    held_steps = 0

    def make_registry(self):
        environ = dict(QWEN_PREFIX_STORE_GIB=str(self.store_gib))
        env = self.profile['env']
        environ.update(tiers.tier_environ(self.BLOCKS, int(self.store_gib * (1 << 30)), QWEN_PREFIX_HOST_TIER_AUDIT=env.get('QWEN_PREFIX_HOST_TIER_AUDIT', '0'),
                                          QWEN_PREFIX_HOST_TIER_VERIFY=env.get('QWEN_PREFIX_HOST_TIER_VERIFY', 'sample'), QWEN_PREFIX_HOST_TIER_OFF_PATH=self.tier_off_path))
        environ.update((name, str(value)) for name, value in env.items() if name in contract.PREFIX_SLICE_NAMES)
        return tiers.graft.PrefixRegistry(environ=environ)

    def attach_extras(self):
        super().attach_extras()
        hooks = self.graft.tier_hooks
        if hooks is not None and hooks.slice_blocks:
            hooks.hold_ready = True
            self.say('(EngineCore pid=9) INFO | scheduler:install:1 - [PINDIAG] prefix: host tier restore slice on blocks=%d ms=%.0f' % (hooks.slice_blocks, hooks.slice_ms))

    def admit(self):
        self.graft.step_no += 1
        if self.prefix and self.running and self.waiting and self.graft.restore_hold(self.waiting[0]):
            SlicedFakeEngine.held_steps += 1
            saved, self.waiting = self.waiting, deque()
            try:
                return super().admit()
            finally:
                self.waiting = saved
        return super().admit()


class EndToEndTests(unittest.TestCase):
    PATCHES = dict(TIER_CONC_SETTLE_S=0.05, TIER_CONC_CONTROL_S=0.6, TIER_CONC_BETWEEN_S=0.1, TIER_CONC_SEAT_BUDGET=24000, TIER_CONC_SEAT_MIN_BUDGET=6000,
                   TIER_CONC_RETURN_MAX_TOKENS=3000)

    def setUp(self):
        self.results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results, True)
        for name, value in self.PATCHES.items():
            patcher = mock.patch.object(replay, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, value in (('SETTLE_SECONDS', 0.0), ('SLICE_BLOCKS', 64), ('SLICE_MS', 50)):
            patcher = mock.patch.object(gate, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        SlicedFakeEngine.held_steps = 0

    def run_plan(self, plan, engine_class=SlicedFakeEngine):
        import time

        document = tiers.tier_profiles()
        harness = gate_tests.Harness(engine_class, document=document)
        runner = harness.runner(self.results)
        runner.sleep = lambda seconds: time.sleep(min(seconds, 1.0))
        arms = gate.plan_arms(plan, FAKE, 'none', document)
        return gate.run_plan(plan, arms, runner, gate_tests.GOOD_ANCHOR), harness

    def test_the_sliced_timed_arm_passes_and_says_how_the_restore_went(self):
        result, harness = self.run_plan('tier-concurrent-timed-sliced')
        arm = result['arms']['tier-concurrent-timed-sliced']
        self.assertEqual(arm['verdict'], 'PASS', (arm['problems'], arm['not_exercised'], arm['lines'][-14:]))
        text = '\n'.join(arm['lines'])
        self.assertIn('sliced restore: cap 64 blocks, budget 50 ms', text)
        self.assertRegex(text, r"the returning turn's restore: \d+ blocks in \d+ slices, the longest slice [\d.]+ ms, the request held for \d+ steps")
        self.assertGreater(SlicedFakeEngine.held_steps, 1, 'the seats decoded while the returning turn was held')
        stats = harness.engines[-1].registry.stats
        self.assertGreater(stats['tier_slice_steps'], 1)
        self.assertEqual(stats['tier_slice_aborted'], 0)
        self.assertTrue(harness.engines[-1].check_kv())

    def test_the_restored_answer_is_the_same_with_the_flag_on_and_off(self):
        result, _ = self.run_plan('tier-concurrent-timed-sliced')
        with open(os.path.join(self.results, 'tier-concurrent-timed-sliced', 'records.jsonl'), encoding='utf-8') as handle:
            sliced = [json.loads(line) for line in handle if line.strip()]
        shutil.rmtree(self.results, True)
        self.results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results, True)
        result, _ = self.run_plan('tier-concurrent-timed')
        with open(os.path.join(self.results, 'tier-concurrent-timed', 'records.jsonl'), encoding='utf-8') as handle:
            whole = [json.loads(line) for line in handle if line.strip()]
        import tier_concurrent_compare as compare

        outcome = compare.compare_pair(whole, sliced, 'whole', 'sliced', minimum=200)
        self.assertEqual(outcome['problems'], [], outcome['lines'])
        self.assertEqual([row['case'] for row in outcome['whole']], [conc.RETURN_CASE, conc.SPILL_CASE, conc.COLD_CASE])
        self.assertTrue(all(row['verdict'] == 'IDENTICAL' for row in outcome['whole']))

    def test_an_engine_without_the_wrapper_falls_back_and_the_judge_says_so(self):
        class Bare(SlicedFakeEngine):
            def attach_extras(self):
                tiers.TierFakeEngine.attach_extras(self)
                self.say('(EngineCore pid=9) INFO | scheduler:install:1 - [PINDIAG] prefix: host tier restore slice fell back: no one-fresh-prefill admission wrapper on this scheduler')

        result, _ = self.run_plan('tier-concurrent-timed-sliced', engine_class=Bare)
        arm = result['arms']['tier-concurrent-timed-sliced']
        self.assertNotEqual(arm['verdict'], 'PASS')
        self.assertTrue(any('fell back' in text for text in arm['problems']), arm['problems'])


# -- the TS job pack ------------------------------------------------------------------------------------------------------

PACK = HERE / 'references' / 'fusion-jobs' / 'tier-slice'


class PackTests(unittest.TestCase):
    IMAGE = 'tp4-ship-4'

    def env(self, name):
        return job.parse_env((PACK / ('%s.env' % name)).read_text(encoding='utf-8'))

    def test_the_order_lists_the_templates_and_each_runs_what_its_name_says(self):
        rows = [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line and not line.startswith('#')]
        names = [row[0] for row in rows]
        self.assertEqual(sorted(names), sorted(path.stem for path in PACK.glob('*.env')))
        self.assertEqual([name.split('-')[0] for name in names], ['B0', 'X0', 'TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'Z'])
        for name, mode, image, minutes in rows:
            self.assertEqual((image, mode in ('stop', 'soft'), int(minutes) > 0), (self.IMAGE, True, True), name)
            self.assertEqual(self.env(name)['C2_IMAGE_TAG'], self.IMAGE)
        self.assertEqual([row[0].split('-')[0] for row in rows if row[1] == 'stop'], ['B0', 'X0', 'TS1'])
        total = sum(int(row[3]) for row in rows)
        self.assertIn('%d minutes (%d h %d min) in all' % (total, total // 60, total % 60), (PACK / 'ORDER.txt').read_text(encoding='utf-8'))
        expect = {'TS1-tier-slice-audited': ('tier-concurrent-sliced', S4_AUDIT), 'TS2-tier-slice-control-timed': ('tier-concurrent-timed', S4_TRAFFIC),
                  'TS3-tier-slice-sliced-timed': ('tier-concurrent-timed-sliced', S4_TRAFFIC), 'TS4-tier-slice-control-repeat': ('tier-concurrent-timed', S4_TRAFFIC),
                  'TS5-tier-slice-sliced-repeat': ('tier-concurrent-timed-sliced', S4_TRAFFIC)}
        for name, (plan, profile) in expect.items():
            env = self.env(name)
            self.assertEqual((env['C2_ACTIONS'], env['C2_PREFIX_PLAN'], env['C2_PREFIX_PROFILE'], env['C2_PREFIX_BASELINE']), ('reset prefix', plan, profile, 'none'), name)
            self.assertIn(plan, job.PREFIX_PLANS)
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        needs = [line for line in text.splitlines() if line.startswith('# NEEDS ')]
        self.assertIn('# NEEDS TS3 <- TS1 TS2', needs)
        for needle in ('tier_concurrent_compare.py', 'tier_stall_breakdown.py', 'ESTIMATE', 'QWEN_PREFIX_TIER_RESTORE_SLICE', 'ship-4 candidate'):
            self.assertIn(needle, text)

    def test_every_prefix_template_plans_in_the_gate_as_a_dry_run(self):
        results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, results, True)
        for path in sorted(PACK.glob('*.env')):
            env = job.parse_env(path.read_text(encoding='utf-8'))
            if 'C2_PREFIX_PLAN' not in env:
                continue            # the build, the status and the reset run no prefix plan
            lines = []
            code = gate.main(['--image', 'img', '--profiles', str(REAL_PROFILES), '--results', results, '--profile', env['C2_PREFIX_PROFILE'], '--plan', env['C2_PREFIX_PLAN'],
                              '--baseline', env['C2_PREFIX_BASELINE'], '--dry-run', '--cards', 'quad'], devices=['/a', '/b', '/c', '/d'], log=lines.append)
            self.assertEqual(code, 0, (path.name, lines))


if __name__ == '__main__':
    unittest.main()
