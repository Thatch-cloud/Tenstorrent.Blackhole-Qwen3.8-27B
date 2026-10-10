"""The prefix gate's tiers plan (tier-attach, tier-returning, tier-timed; c2_prefix_gate, prefix_replay, prefix_markers) on the CPU.

Three layers. (1) The wiring: the plan and its arms are registered, a profile without the tier or the preconverted checkpoints is refused, the image check, the anchor probe's
raw-ops reading, the dry-run's docker argv on the real tier twin, the marker lines parsed. (2) The judge, with fabricated logs, counters and records: every finding the
plan reads (a failure counter, a [PREFIX-AUDIT-CKPT] line with a 0, a session the tier lost with room to spare, a fresh-salt request restored from the tier, the kill switch's
latch, a restore on record) in its PASS, NOT_EXERCISED and FAIL shapes. (3) The scenarios end to end on TierFakeEngine: test_prefix_replay's engine with the REAL scheduler
graft, a pool that speaks vLLM's block-pool API, and an in-memory device whose blocks hold a pure function of their hash, so the whole plan runs - sessions, a flood the size of the
pool, the returning turns restored from the tier, the tier's kill switch - and every cached block is checked to hold the bytes its hash stands for.

Run at py 3.11: `py -3.11 -B -m unittest test_prefix_tier_gate` from scripts/ci."""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import make_prefix_tier_profiles as tier_twin  # noqa: E402
import prefix_judge as judge  # noqa: E402
import prefix_markers as pm  # noqa: E402
import prefix_replay as replay  # noqa: E402
import qwen_prefix_registry as prefix_registry  # noqa: E402
import qwen_prefix_scheduler_patch as graft  # noqa: E402
import test_c2_prefix_gate as gate_tests  # noqa: E402
import test_prefix_replay as fakes  # noqa: E402
from test_qwen_prefix_scheduler_patch import FakeKvIO, kv_content  # noqa: E402
from test_qwen_prefix_tiers import tier_environ  # noqa: E402

BLOCK = judge.BLOCK
TIER = tier_twin.TWIN
FAKE = 'c2-packed-prefix'
TIER_ENV = dict(tier_twin.ENV)
REAL_PROFILES = os.path.join(HERE, 'qwen_c2_profiles.json')


def read(name):
    with open(os.path.join(HERE, name), encoding='utf-8') as handle:
        return handle.read()


# -- the fakes ---------------------------------------------------------------------------------------------

class TierPool(fakes.FakePool):
    """The fake block pool with the part of vLLM's BlockPool API the tier's restore calls (get_num_free_blocks, get_new_blocks, cache_full_blocks, free_blocks)."""

    def get_num_free_blocks(self):
        return len(self.free)

    def get_new_blocks(self, count):
        got = self.allocate(count)
        if got is None:
            raise ValueError('cannot get %d free blocks' % count)
        return got

    def cache_full_blocks(self, request, blocks, num_cached, num_full, block_size, group):
        hashes = request.block_hashes
        for index in range(num_cached, num_full):
            block = blocks[index]
            if block.block_hash is None and hashes[index] not in self.cached:
                block.block_hash = hashes[index]
                self.cached[hashes[index]] = block

    def free_blocks(self, blocks):
        for block in blocks:
            block.ref -= 1
            if block.ref == 0:
                self.free[block.block_id] = block


class TierFakeEngine(fakes.FakeEngine):
    """test_prefix_replay's engine with the host KV tier: the real graft's TierHooks over a TierPool, an in-memory device (FakeKvIO: block id -> bytes) that holds, for every
    block the model wrote, kv_content of the block's hash - written at the row's prefill, AFTER the scheduler step that spilled whatever the step evicted, as on a card - and the
    log lines the model graft and the scheduler graft print. Faults: tier_never_spills (the hooks never hold an eviction), tier_no_preconverted (no 'checkpoint preconverted' line),
    tier_bad_ckpt_audit (a [PREFIX-AUDIT-CKPT] line with conversion_equal=0), tier_no_io_line."""

    FAULTS = fakes.FakeEngine.FAULTS + ('tier_never_spills', 'tier_no_preconverted', 'tier_bad_ckpt_audit', 'tier_no_io_line')
    BLOCKS = 40000        # payloads the tier holds: more than any pool of the tests, so only a fault empties it

    def __init__(self, *args, **kwargs):
        self.tier_off_path = os.path.join(tempfile.gettempdir(), 'pfx-tier-off-%d-%d' % (os.getpid(), id(self)))
        super().__init__(*args, **kwargs)

    def tier_configured(self):
        try:
            return float((self.profile.get('env') or {}).get('QWEN_PREFIX_HOST_TIER_GIB') or 0) > 0
        except ValueError:
            return False

    def make_pool(self, count):
        return TierPool(count)

    def make_registry(self):
        environ = dict(QWEN_PREFIX_STORE_GIB=str(self.store_gib))
        if self.tier_configured():
            env = self.profile['env']
            environ.update(tier_environ(self.BLOCKS, int(self.store_gib * (1 << 30)),
                                        QWEN_PREFIX_HOST_TIER_AUDIT=env.get('QWEN_PREFIX_HOST_TIER_AUDIT', '0'),
                                        QWEN_PREFIX_HOST_TIER_VERIFY=env.get('QWEN_PREFIX_HOST_TIER_VERIFY', 'sample'),
                                        QWEN_PREFIX_HOST_TIER_OFF_PATH=self.tier_off_path))
        return graft.PrefixRegistry(environ=environ)

    def attach_extras(self):
        self.io = FakeKvIO()
        if self.registry.tier is None:
            return
        self.registry.attach_tier_io(self.io)
        self.graft.make_block_hash = lambda block_hash, group: block_hash
        self.graft.tier_hooks = graft.TierHooks(self.graft)
        self.graft.tier_hooks.kill_switch.poll_s = 0.0
        if 'tier_never_spills' in self.faults:
            self.graft.tier_hooks.on_evict = lambda block, keys: False
        if 'tier_no_io_line' not in self.faults:
            self.say('(EngineCore pid=9) INFO | models.demos.blackhole.qwen36.tt.model:_qwen_prefix_tier_attach:1006 - [PINDIAG] prefix: host tier IO attached '
                     'block_bytes=%d tensors=32 chips=4 slice_bytes=4 fingerprint=%s' % (self.io.block_bytes, self.io.fingerprint))

    def tier_switch(self, on):
        if on:
            with open(self.tier_off_path, 'w') as handle:
                handle.write(replay.KILL_SWITCH_OWNER + '\n')
        elif os.path.exists(self.tier_off_path):
            os.remove(self.tier_off_path)

    def admit(self):
        new = super().admit()
        if self.graft.tier_hooks is not None:
            self.graft.tier_hooks.flush()        # the end of schedule(): what the step evicted is read out before the model writes
        return new

    def prefill(self, request, q):
        super().prefill(request, q)
        if self.registry.tier is None:
            return
        owned = self.single.req_to_blocks.get(request.request_id) or []
        for block in owned[q // BLOCK:]:
            if block.block_hash is not None:
                self.io.device[block.block_id] = kv_content(block.block_hash)
        env = self.profile.get('env') or {}
        if str(env.get('QWEN_PREFIX_CKPT_PRECONVERTED')) == '1' and request.admissions == 1 and 'tier_no_preconverted' not in self.faults:
            for position in request.captured:
                self.say('(EngineCore pid=9) INFO | model:_qwen_prefix_capture:790 - [PINDIAG] prefix: checkpoint preconverted req=%s pos=%d tensors=96 bytes=106954752'
                         % (request.request_id, position))
        if q and str(env.get('QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT')) == '1':
            self.say('(EngineCore pid=9) INFO | model:_qwen_prefix_audit_preconverted:905 - [PREFIX-AUDIT-CKPT] tensors=96 conversion_equal=%d readback_equal=1 differing=[]'
                     % (0 if 'tier_bad_ckpt_audit' in self.faults else 1))

    def check_kv(self):
        """Every cached block holds the bytes its hash stands for, and so does every record of the tier."""
        for block in self.pool.blocks:
            if block.block_hash is not None:
                assert self.io.device.get(block.block_id) == kv_content(block.block_hash), 'cached block %d holds the wrong bytes' % block.block_id
        if self.registry.tier is not None:
            for key, record in self.registry.tier.records.items():
                assert bytes(record.payload) == kv_content(key), 'a tier record holds the wrong bytes'
        return True


def tier_profiles(**env):
    """The gate fixtures' sticky fast-path profiles with the tier's names on c2-packed-prefix (the fake reads the booleans; its tier is sized in payloads)."""
    document = gate_tests.sticky_profiles()
    document['profiles'][FAKE]['env'].update(TIER_ENV)
    document['profiles'][FAKE]['env'].update(env)
    return document


def run_tiers(plan, results, engine_class=TierFakeEngine, document=None, **faults):
    document = document or tier_profiles()
    harness = gate_tests.Harness(engine_class, document=document, **faults)
    runner = harness.runner(results)
    arms = gate.plan_arms(plan, FAKE, 'none', document)
    return gate.run_plan(plan, arms, runner, gate_tests.GOOD_ANCHOR), harness, runner


# -- (1) the wiring ------------------------------------------------------------------------------------------

class WiringTests(unittest.TestCase):
    def test_the_plan_and_its_three_arms_are_registered_in_the_job_parser_and_the_gate(self):
        self.assertIn('tiers', job.PREFIX_PLANS)
        arms = [arm for arm, plan in job.PREFIX_ARM_PLANS if plan == 'tiers']
        self.assertEqual(arms, ['tier-attach', 'tier-returning', 'tier-timed'])
        for name in ['tiers'] + arms:
            self.assertIn(name, gate.PLANS)
        self.assertEqual([entry[0] for entry in gate.PLAN_ARMS['tiers']], arms)
        for arm in arms:
            self.assertEqual([entry[0] for entry in gate.PLAN_ARMS[arm]], [arm], 'a single-arm plan runs just it')
        for entry in gate.PLAN_ARMS['tiers']:
            self.assertIn(entry[1], replay.SCENARIOS)

    def test_only_the_timed_arm_is_a_timed_scenario_and_it_derives_the_audit_free_profile(self):
        by_arm = dict((entry[0], entry) for entry in gate.PLAN_ARMS['tiers'])
        self.assertEqual(by_arm['tier-timed'][3], 'tiertime')
        self.assertIsNone(by_arm['tier-attach'][3])
        self.assertIsNone(by_arm['tier-returning'][3])
        self.assertIn('tier_timed', gate.TIMED_SCENARIOS)
        self.assertNotIn('tier_returning', gate.TIMED_SCENARIOS)
        self.assertTrue(by_arm['tier-attach'][5], 'the attach chain is strict like the bring-up')
        self.assertEqual(gate.DERIVED['tiertime']['env']['QWEN_PREFIX_HOST_TIER_AUDIT'], '0')

    def test_a_profile_that_names_no_tier_or_no_preconverted_checkpoints_has_no_tier_arm_to_run(self):
        document = gate_tests.sticky_profiles()
        self.assertEqual(gate.plan_arms('tiers', FAKE, 'none', document), [])
        skipped = gate.not_applicable('tiers', FAKE, document)
        self.assertEqual([arm for arm, _ in skipped], ['tier-attach', 'tier-returning', 'tier-timed'])
        self.assertTrue(all('names no QWEN_PREFIX_HOST_TIER_GIB' in why for _, why in skipped))
        half = tier_profiles()
        half['profiles'][FAKE]['env'].pop('QWEN_PREFIX_CKPT_PRECONVERTED')
        self.assertEqual(gate.plan_arms('tier-returning', FAKE, 'none', half), [])
        self.assertTrue(all('QWEN_PREFIX_CKPT_PRECONVERTED=1' in why for _, why in gate.not_applicable('tier-returning', FAKE, half)))

    def test_the_arms_read_the_profiles_tier_settings(self):
        arms = gate.plan_arms('tiers', FAKE, 'none', tier_profiles())
        self.assertEqual([arm['arm'] for arm in arms], ['tier-attach', 'tier-returning', 'tier-timed'])
        for arm in arms[:2]:
            self.assertEqual((arm['tier']['audit'], arm['tier']['preconverted_audit'], arm['tier']['verify']), (True, True, 'all'))
            self.assertIsNone(arm['derived'])
        timed = arms[2]
        self.assertEqual((timed['tier']['audit'], timed['tier']['preconverted_audit'], timed['tier']['verify']), (False, False, 'sample'))
        self.assertTrue(timed['derived'])
        self.assertEqual(timed['served'], FAKE + '+tiertime')
        self.assertTrue(timed['tier']['preconverted'])
        self.assertFalse(gate.wants_digests(timed))
        self.assertTrue(gate.wants_digests(arms[0]) and gate.wants_digests(arms[1]))
        self.assertEqual(arms[1]['timeout'], 9000)

    def test_the_tier_arms_need_no_baseline_and_a_baseline_arm_never_appears(self):
        arms = gate.plan_arms('tiers', FAKE, 'none', tier_profiles())
        self.assertTrue(all(arm['prefix'] for arm in arms))

    def test_the_image_check_names_the_arms_and_what_to_build(self):
        arms = gate.plan_arms('tiers', FAKE, 'none', tier_profiles())
        good = dict(gate_tests.GOOD_ANCHOR, raw_ops=True)
        self.assertEqual(gate.tier_image_problems(arms, good), [])
        for anchor, needle in ((dict(good, raw_ops=False), 'qwen_read_blocks_raw'), (dict(good, raw_ops=None), 'qwen_read_blocks_raw'),
                               (dict(good, stage_mismatched=['model.py']), 'predates the preconverted checkpoints'), (dict(error='boom'), 'could not read the image')):
            with self.subTest(needle=needle):
                problems = gate.tier_image_problems(arms, anchor)
                self.assertTrue(problems and needle in problems[0], problems)
                self.assertIn('tier-attach', problems[0])
        self.assertEqual(gate.tier_image_problems([arm for arm in gate.plan_arms('exactness-eager', 'c2-packed-prefix', 'none', gate_tests.sticky_profiles())], {}), [])

    def test_the_anchor_probe_reports_the_raw_ops_beside_the_region_read(self):
        script = gate.anchor_script()
        for name in ('qwen_read_blocks_raw', 'qwen_write_blocks_raw', 'qwen_block_bytes'):
            self.assertIn(name, script)
        self.assertEqual(gate.parse_anchor('==region\nregion 1 raw 1\n')['raw_ops'], True)
        self.assertEqual(gate.parse_anchor('==region\nregion 1 raw 0\n')['raw_ops'], False)
        old = gate.parse_anchor('==region\nregion 1\n')
        self.assertEqual((old['region_read'], old['raw_ops']), (True, None), 'an image that predates the raw ops prints the old two-word line')

    def test_main_refuses_an_image_without_the_raw_ops_and_runs_nothing(self):
        results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, results, True)
        lines = []
        code = gate.main(['--image', 'img', '--profiles', REAL_PROFILES, '--results', results, '--profile', TIER, '--plan', 'tier-attach', '--baseline', 'none',
                          '--cards', 'quad'], devices=['/a', '/b', '/c', '/d'], log=lines.append, anchor=dict(gate_tests.GOOD_ANCHOR, raw_ops=False),
                         runner_factory=lambda *a, **k: self.fail('a runner was built'))
        self.assertEqual(code, 2)
        self.assertTrue(any(line.startswith('refused:') and 'qwen_read_blocks_raw' in line for line in lines), lines)

    def test_the_dry_run_on_the_real_tier_twin_plans_three_arms_with_the_gate_switch_and_the_timed_one_derived(self):
        results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, results, True)
        lines = []
        code = gate.main(['--image', 'img', '--profiles', REAL_PROFILES, '--results', results, '--profile', TIER, '--plan', 'tiers', '--baseline', 'none',
                          '--dry-run', '--cards', 'quad'], devices=['/a', '/b', '/c', '/d'], log=lines.append)
        self.assertEqual(code, 0, lines)
        rows = [json.loads(line) for line in lines if line.startswith('{') and '"docker"' in line]
        self.assertEqual([row['arm'] for row in rows], ['tier-attach', 'tier-returning', 'tier-timed'])
        for row in rows:
            self.assertIn('QWEN_C2_GATE=1', row['docker'], 'a gate-only profile boots only with the gate switch')
            self.assertIn('QWEN_PREFIX_STATS_S=0', row['docker'])
        self.assertIn('QWEN_PREFIX_DIGESTS=1', rows[0]['docker'])
        self.assertIn('QWEN_PREFIX_DIGESTS=1', rows[1]['docker'])
        self.assertNotIn('QWEN_PREFIX_DIGESTS=1', rows[2]['docker'], 'the timed arm times the restore, not the digests')
        self.assertEqual(rows[2]['served'], TIER + '+tiertime')
        self.assertEqual(rows[0]['served'], TIER)
        self.assertTrue(any('profiles.json' in token for token in rows[2]['docker']), 'the derived profile is mounted')
        self.assertEqual(rows[1]['timeout'], 9000)

    def test_the_tier_plan_on_a_profile_without_a_tier_is_refused_by_main(self):
        results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, results, True)
        lines = []
        code = gate.main(['--image', 'img', '--profiles', REAL_PROFILES, '--results', results, '--profile', tier_twin.PARENT, '--plan', 'tier-timed', '--baseline', 'none',
                          '--dry-run', '--cards', 'quad'], devices=['/a', '/b', '/c', '/d'], log=lines.append)
        self.assertEqual(code, 2)
        self.assertTrue(any('names no QWEN_PREFIX_HOST_TIER_GIB' in line for line in lines), lines)


class SequentialSdpaTests(unittest.TestCase):
    """T1 (run 38082865675): three IDENTICAL pairs, a preconverted restore audited equal, and a FAIL on 'QWEN_FAST_TP4_SDPA_AUDIT=1 is set and no passing audit
    line was logged' - the arm sends one request at a time, so no packed round of two or more users ran and the multi launch had nothing to compare."""

    ENV = {'QWEN_FAST_TP4_SDPA': 'multi', 'QWEN_FAST_TP4_SDPA_AUDIT': '1'}
    LOG = '\n'.join(['[PINDIAG] tp4 sdpa engaged config=multi grid=11x10 entries=4 users=4 flags=0x21 cores_per_entry=16 active_cores=64 audit=1',
                     '[PINDIAG] tp4 sdpa multi call users=4 flags=0x21 cores_per_entry=16 audit=1', 'UNQUALIFIED'])
    NOT_COMPARED = 'nothing was compared'

    def report(self, by_live):
        return dict(problems=[], rounds=dict(count=sum(by_live.values()), by_live=by_live), extent_audit=dict(lines=0, mismatches=0))

    def findings(self, scenario, by_live, log=None):
        arm = dict(arm=scenario.replace('_', '-'), scenario=scenario, served_env=dict(self.ENV), env=(), sticky=False)
        with mock.patch.object(gate.harness, 's2_report', return_value=self.report(by_live)):
            problems, _, lines, _ = gate.s2_findings(arm, self.LOG if log is None else log, dict(), [])
        return problems, lines

    def test_the_optional_rule_needs_a_sequential_scenario_and_no_multi_user_round(self):
        for scenario in gate.SEQUENTIAL_SCENARIOS:
            self.assertTrue(gate.sdpa_audit_optional(dict(scenario=scenario), self.report({})), scenario)
            self.assertTrue(gate.sdpa_audit_optional(dict(scenario=scenario), self.report({'1': 12})), 'rounds of one live user are not a multi-user round')
            self.assertFalse(gate.sdpa_audit_optional(dict(scenario=scenario), self.report({'1': 4, '2': 1})), 'one round of two users and the rule is asked in full')
            self.assertFalse(gate.sdpa_audit_optional(dict(scenario=scenario), self.report({2: 1})), 'integer keys too')
            self.assertFalse(gate.sdpa_audit_optional(dict(scenario=scenario), self.report({'many': 1})), 'a key it cannot read keeps the rule')
        for scenario in ('exactness_shared', 'lifecycle_evict', 'bringup_prefix', 'tier_timed', 'agent_turns', 'levern_hit', 'exactness_audit'):
            self.assertFalse(gate.sdpa_audit_optional(dict(scenario=scenario), self.report({})), '%s is never waived, whatever it saw' % scenario)
        self.assertTrue(set(gate.SEQUENTIAL_SCENARIOS) <= set(gate.TIER_SCENARIOS))

    def test_t1_and_t2_no_longer_fail_on_the_audit_they_could_not_have_logged(self):
        for scenario in gate.SEQUENTIAL_SCENARIOS:
            problems, lines = self.findings(scenario, {})
            self.assertEqual([p for p in problems if self.NOT_COMPARED in p], [], scenario)
            self.assertTrue(any(line.startswith('multi SDPA audit: not asked of this sequential arm') for line in lines), lines)

    def test_the_rule_is_asked_in_full_wherever_the_multi_launch_ran_or_could_have(self):
        for scenario, by_live in (('tier_returning', {'2': 1}), ('tier_attach', {'1': 3, '4': 2}), ('exactness_shared', {}), ('lifecycle_evict', {}),
                                  ('bringup_prefix', {'1': 9})):
            problems, lines = self.findings(scenario, by_live)
            self.assertTrue([p for p in problems if self.NOT_COMPARED in p], (scenario, by_live, problems))
            self.assertFalse(any('not asked of this sequential arm' in line for line in lines), (scenario, lines))

    def test_a_sequential_arm_is_still_judged_on_what_the_log_says(self):
        mismatch = '[PINDIAG] tp4 sdpa audit MISMATCH round=3 users=4 differing=2 chip 0 layer 3: 2 of 262144 words differ'
        problems, _ = self.findings('tier_attach', {}, log=self.LOG + '\n' + mismatch)
        self.assertTrue(any('audit found a difference' in p for p in problems), problems)
        problems, _ = self.findings('tier_returning', {}, log='')
        self.assertTrue(any('config=multi flags=0x21' in p for p in problems), 'no engaged line: the multi path never attached')
        self.assertTrue(any('built and never called' in p for p in problems))
        problems, lines = self.findings('tier_attach', {}, log=self.LOG + '\n[PINDIAG] tp4 sdpa audit 1 exact=True users=4 layers=16 chips=4 words=1 live=2')
        self.assertEqual([p for p in problems if self.NOT_COMPARED in p], [])

    def test_a_profile_without_the_audit_says_nothing_about_it(self):
        arm = dict(arm='tier-attach', scenario='tier_attach', served_env={'QWEN_FAST_TP4_SDPA': 'multi'}, env=(), sticky=False)
        with mock.patch.object(gate.harness, 's2_report', return_value=self.report({})):
            _, _, lines, _ = gate.s2_findings(arm, self.LOG, dict(), [])
        self.assertFalse(any('multi SDPA audit' in line for line in lines))


class MarkerTests(unittest.TestCase):
    LINES = [
        '2026-10-10T01:00:00.000000001Z (EngineCore pid=9) INFO [PINDIAG] prefix: host tier on gib=32.00 kv_gib=24.00 block_bytes=2228224 verify=all audit=1 '
        'spill_max_blocks=512 min_tokens=8192 slack=1 off_path=/tmp/qwen-prefix-host-tier.off',
        '2026-10-10T01:00:01.000000001Z (EngineCore pid=9) INFO [PINDIAG] prefix: host tier IO attached block_bytes=2228224 tensors=32 chips=4 slice_bytes=17408 fingerprint=kvtier1:32x4x17408',
        '2026-10-10T01:00:02.000000001Z [PINDIAG] prefix: host tier spill blocks=512 bytes=1140850688 ms=412.5 held=512',
        '2026-10-10T01:00:03.000000001Z [PINDIAG] prefix: host tier restore req=chatcmpl-pfx-tier-returning-0044-hit-1a2b3c4d blocks=900 bytes=2005401600 ms=388.2 from=0 to=57600',
        '2026-10-10T01:00:04.000000001Z [PINDIAG] prefix: host tier latched off: kill switch /tmp/qwen-prefix-host-tier.off present',
        '[PINDIAG] prefix: host tier restore failed req=x (1 so far): RuntimeError: device write refused',
        '[PREFIX-AUDIT-CKPT] tensors=96 conversion_equal=1 readback_equal=0 differing=[3:readback]',
        '[PINDIAG] prefix: checkpoint preconverted req=chatcmpl-a-1 pos=2048 tensors=96 bytes=106954752',
        '[PINDIAG] prefix: checkpoint preconverted req=chatcmpl-a-1 pos=4096 tensors=96 bytes=106954752',
    ]

    def test_every_tier_line_is_read_into_its_list(self):
        scanned = pm.scan(self.LINES)
        self.assertEqual(scanned['tier_on'][0]['gib'], 32.0)
        self.assertEqual((scanned['tier_on'][0]['verify'], scanned['tier_on'][0]['audit'], scanned['tier_on'][0]['block_bytes']), ('all', 1, 2228224))
        self.assertEqual((scanned['tier_io'][0]['tensors'], scanned['tier_io'][0]['chips'], scanned['tier_io'][0]['slice_bytes']), (32, 4, 17408))
        self.assertEqual((scanned['tier_spills'][0]['blocks'], scanned['tier_spills'][0]['bytes'], scanned['tier_spills'][0]['ms']), (512, 1140850688, 412.5))
        restore = scanned['tier_restores'][0]
        self.assertEqual((restore['tag'], restore['blocks'], restore['bytes'], restore['ms'], restore['from'], restore['to']),
                         ('pfx-tier-returning-0044-hit', 900, 2005401600, 388.2, 0, 57600))
        self.assertEqual(len(scanned['tier_trouble']), 2)
        self.assertIn('latched off', scanned['tier_trouble'][0]['line'])
        self.assertEqual((scanned['ckpt_audits'][0]['conversion_equal'], scanned['ckpt_audits'][0]['readback_equal']), (1, 0))
        self.assertEqual(scanned['ckpt_preconverted'], 2)

    def test_an_ordinary_log_has_none_of_them(self):
        scanned = pm.scan(['[PINDIAG] prefix: stats {"grants": 1}', '[PREFIX] row=1 req=a path=traced registry=present grant=none Q=0 L=10'])
        for key in ('tier_on', 'tier_io', 'tier_spills', 'tier_restores', 'tier_trouble', 'ckpt_audits'):
            self.assertEqual(scanned[key], [], key)
        self.assertEqual(scanned['ckpt_preconverted'], 0)

    def test_the_lines_the_scheduler_graft_prints_are_the_ones_the_markers_read(self):
        """The format strings in the graft (the real source) carry the key=value names the scan reads."""
        source = read('qwen_prefix_scheduler_patch.py')
        for text in ('host tier on gib=%.2f kv_gib=%.2f block_bytes=%d verify=%s audit=%d', 'host tier spill blocks=%d bytes=%d ms=%.1f held=%d',
                     'host tier restore req=%s blocks=%d bytes=%d ms=%.1f from=%d to=%d'):
            self.assertIn(text, source)
        model = read('qwen_prefix_model_patch.py')
        for text in ('[PINDIAG] prefix: host tier IO attached block_bytes=', '[PREFIX-AUDIT-CKPT] tensors=', '[PINDIAG] prefix: checkpoint preconverted req='):
            self.assertIn(text, model)
        self.assertIn('host tier latched off', read('qwen_prefix_registry.py'))


# -- (2) the judge -----------------------------------------------------------------------------------------------

def arm_of(scenario, audited=True):
    return dict(arm=scenario.replace('_', '-'), scenario=scenario, tier=dict(gib=32.0, audit=audited, preconverted=True, preconverted_audit=audited, verify='all' if audited else 'sample'))


def good_scan(**extra):
    scanned = dict(tier_on=[dict(index=1, gib=32.0, kv_gib=24.0, block_bytes=2228224, verify='all', audit=1, spill_max_blocks=512, min_tokens=8192, slack=1)],
                   tier_io=[dict(index=0)], tier_spills=[], tier_restores=[], tier_trouble=[], ckpt_audits=[dict(index=5, tensors=96, conversion_equal=1, readback_equal=1)],
                   ckpt_preconverted=4)
    scanned.update(extra)
    return scanned


GOOD_STATS = dict(pins=0, commit_mismatch=0, tier_spill_blocks=3000, tier_restore_requests=3, tier_restore_blocks=2700, tier_audit_reads=2700, tier_digest_checks=2700,
                  tier_latched=0, tier_bytes=1, tier_entries=1)


def record(tag, role, case, q=0, expected=0, ok=True, window=(0, 100), **extra):
    values = dict(tag=tag, role=role, case=case, ok=ok, markers=dict(q=q), expected=dict(q=expected), prompt_tokens=50000, log_window=list(window))
    values.update(extra)
    return values


def returning_records(q=(51200, 51200, 51200), expected=(51200, 51200, 51200)):
    return [record('pfx-r-%d-hit' % index, 'hit', 'tier-after-flood', q=got, expected=want) for index, (got, want) in enumerate(zip(q, expected))]


def returning_events(**extra):
    events = {'tier-flood': dict(pool_tokens=1000, floods=4), 'stats-before-tier-kill': dict(stats=dict(GOOD_STATS)),
              'tier-kill': dict(written=True, removed=True, latched_tag='pfx-latched-hit')}
    events.update(extra)
    return events


def restores_of(records, blocks=900):
    return [dict(index=10 + number, tag=r['tag'], blocks=blocks, bytes=blocks * 2228224, ms=400.0, **{'from': 0, 'to': 57600}) for number, r in enumerate(records)]


class JudgeTests(unittest.TestCase):
    def judge_returning(self, records=None, stats=None, events=None, scanned=None, **scan_extra):
        records = records if records is not None else returning_records()
        scanned = scanned or good_scan(tier_restores=restores_of(records), tier_trouble=[dict(index=99, line='[PINDIAG] prefix: host tier latched off: kill switch x present')],
                                       **scan_extra)
        stats = dict(GOOD_STATS, tier_latched=1) if stats is None else stats
        return gate.tier_findings(arm_of('tier_returning'), records + [record('pfx-latched-hit', 'hit', 'tier-latched', ok=True)], events or returning_events(),
                                  scanned, stats)

    def test_the_good_returning_arm_has_nothing_to_say(self):
        problems, missing, lines = self.judge_returning()
        self.assertEqual((problems, missing), ([], []))
        self.assertTrue(any(line.startswith('after the flood') for line in lines))
        self.assertTrue(any(line.startswith('tier kill switch') for line in lines))

    def test_each_failure_counter_is_a_fail_naming_the_counter(self):
        for name in ('tier_spill_failures', 'tier_restore_failures', 'tier_restore_refused_digest', 'tier_digest_failures', 'tier_audit_mismatches'):
            with self.subTest(name=name):
                before = dict(GOOD_STATS, **{name: 1})
                problems, _, _ = self.judge_returning(events=returning_events(**{'stats-before-tier-kill': dict(stats=before)}))
                self.assertTrue(any(name in problem and 'before the tier kill switch' in problem for problem in problems), problems)
                problems, _, _ = self.judge_returning(stats=dict(GOOD_STATS, tier_latched=1, **{name: 2}))
                self.assertTrue(any(name in problem and 'at the end' in problem for problem in problems), problems)

    def test_the_tier_latching_before_the_drill_is_a_fail_but_after_it_is_the_drill(self):
        before = dict(GOOD_STATS, tier_latched=1)
        problems, _, _ = self.judge_returning(events=returning_events(**{'stats-before-tier-kill': dict(stats=before)}))
        self.assertEqual(problems, [], 'tier_latched is not a failure counter: the drill and its own line are what judge it')
        problems, _, _ = self.judge_returning(stats=dict(GOOD_STATS, tier_latched=0))
        self.assertTrue(any('tier_latched=0' in problem for problem in problems), 'the kill switch was written and the tier did not latch')
        problems, _, _ = self.judge_returning(scanned=good_scan(tier_restores=restores_of(returning_records()), tier_trouble=[]))
        self.assertTrue(any('no "host tier latched off" line' in problem for problem in problems))
        problems, _, _ = self.judge_returning(scanned=good_scan(tier_restores=restores_of(returning_records()),
                                                                tier_trouble=[dict(index=9, line='[PINDIAG] prefix: host tier restore failed req=a (1 so far): x')]))
        self.assertTrue(any('host tier restore failed' in problem for problem in problems))

    def test_a_flood_that_left_the_sessions_on_the_device_is_not_exercised(self):
        records = returning_records()
        problems, missing, _ = self.judge_returning(records=records, scanned=good_scan(tier_restores=[],
                                                    tier_trouble=[dict(index=99, line='host tier latched off: kill switch')]),
                                                    events=returning_events(**{'stats-before-tier-kill': dict(stats=dict(GOOD_STATS, tier_restore_requests=0, tier_restore_blocks=0))}))
        self.assertEqual(problems, [])
        self.assertTrue(any('no returning turn was restored from the tier' in text for text in missing), missing)
        self.assertTrue(any('tier_restore_requests' in text or 'no request was restored' in text for text in missing), missing)

    def test_a_session_the_tier_lost_with_room_to_spare_is_a_fail_and_with_a_drop_on_record_a_note(self):
        records = returning_records(q=(51200, 28672, 51200))
        problems, _, lines = self.judge_returning(records=records)
        self.assertTrue(any('pfx-r-1-hit' in problem and 'no tier loss on record' in problem for problem in problems), problems)
        short = dict(GOOD_STATS, tier_evicted=40)
        problems, _, lines = self.judge_returning(records=records, events=returning_events(**{'stats-before-tier-kill': dict(stats=short)}))
        self.assertEqual(problems, [])
        self.assertTrue(any('short of room' in line and 'tier_evicted' in line for line in lines), lines)

    def test_the_checkpoint_stores_own_evictions_explain_a_lost_session_too(self):
        records = returning_records(q=(51200, 28672, 51200))
        problems, _, lines = self.judge_returning(records=records, events=returning_events(**{'stats-before-tier-kill': dict(stats=dict(GOOD_STATS, evicted_lru=3))}))
        self.assertEqual(problems, [])
        self.assertTrue(any('evicted_lru' in line for line in lines), lines)

    def test_a_latch_that_is_not_the_drills_is_a_fail_in_every_arm(self):
        problems, _, _ = self.judge_returning(scanned=good_scan(tier_restores=restores_of(returning_records()),
                                                                tier_trouble=[dict(index=9, line='[PINDIAG] prefix: host tier latched off: three restores in a row failed')]))
        self.assertTrue(any('three restores in a row failed' in problem for problem in problems), problems)
        for scenario in ('tier_attach', 'tier_timed'):
            problems, _, _ = gate.tier_findings(arm_of(scenario), [], {}, good_scan(), dict(GOOD_STATS, tier_latched=1))
            self.assertTrue(any('tier_latched=1: the tier latched itself off' in problem for problem in problems), (scenario, problems))

    def test_a_fresh_salt_request_restored_from_the_tier_is_another_tenants_pages(self):
        records = returning_records() + [record('pfx-cold-1', 'cold', 'tier-after-flood')]
        scanned = good_scan(tier_restores=restores_of(records), tier_trouble=[dict(index=99, line='host tier latched off: kill switch')])
        problems, _, _ = gate.tier_findings(arm_of('tier_returning'), records + [record('pfx-latched-hit', 'hit', 'tier-latched')], returning_events(), scanned,
                                            dict(GOOD_STATS, tier_latched=1))
        self.assertTrue(any('pfx-cold-1' in problem and "another tenant's pages" in problem for problem in problems), problems)

    def test_pins_and_commit_mismatches_after_the_arm_are_fails(self):
        problems, _, _ = self.judge_returning(stats=dict(GOOD_STATS, tier_latched=1, pins=2, commit_mismatch=1))
        self.assertTrue(any('2 checkpoint pins' in problem for problem in problems), problems)
        self.assertTrue(any('1 grants refused at commit' in problem for problem in problems), problems)

    def test_the_kill_switch_that_could_not_be_written_is_not_exercised_and_one_left_behind_is_a_fail(self):
        _, missing, _ = self.judge_returning(events=returning_events(**{'tier-kill': dict(written=False)}))
        self.assertTrue(any('could not be written' in text for text in missing))
        problems, _, _ = self.judge_returning(events=returning_events(**{'tier-kill': dict(written=True, removed=False, latched_tag='pfx-latched-hit')}))
        self.assertTrue(any('could not be removed' in problem for problem in problems))

    def test_the_audit_instruments_must_have_run_when_the_profile_names_them(self):
        stats = dict(GOOD_STATS, tier_audit_reads=0, tier_digest_checks=0, tier_latched=1)
        _, missing, _ = self.judge_returning(stats=stats, events=returning_events(**{'stats-before-tier-kill': dict(stats=dict(GOOD_STATS, tier_audit_reads=0, tier_digest_checks=0))}))
        self.assertTrue(any('read back and compared' in text for text in missing), missing)
        self.assertTrue(any('digest was checked' in text for text in missing), missing)
        problems, missing, _ = gate.tier_findings(arm_of('tier_returning', audited=False), returning_records() + [record('pfx-latched-hit', 'hit', 'tier-latched')],
                                                  returning_events(**{'stats-before-tier-kill': dict(stats=dict(GOOD_STATS, tier_audit_reads=0, tier_digest_checks=0))}),
                                                  good_scan(tier_restores=restores_of(returning_records()), tier_trouble=[dict(index=99, line='host tier latched off: kill switch x present')]),
                                                  dict(GOOD_STATS, tier_latched=1, tier_audit_reads=0, tier_digest_checks=0))
        self.assertEqual((problems, missing), ([], []), 'an unaudited (timed-profile) arm does not ask for the audit counters')

    def test_the_preconverted_checkpoints_and_their_audit_lines(self):
        records = returning_records() + [record('pfx-latched-hit', 'hit', 'tier-latched')]
        base = dict(tier_restores=restores_of(returning_records()), tier_trouble=[dict(index=99, line='host tier latched off: kill switch x present')])
        args = (arm_of('tier_returning'), records, returning_events())
        stats = dict(GOOD_STATS, tier_latched=1)
        problems, missing, _ = gate.tier_findings(*args, good_scan(ckpt_preconverted=0, **base), stats)
        self.assertTrue(any('no checkpoint was stored preconverted' in text for text in missing))
        problems, missing, _ = gate.tier_findings(*args, good_scan(ckpt_audits=[], **base), stats)
        self.assertTrue(any('no [PREFIX-AUDIT-CKPT] line' in text for text in missing))
        for bad in (dict(conversion_equal=0, readback_equal=1), dict(conversion_equal=1, readback_equal=0)):
            problems, missing, _ = gate.tier_findings(*args, good_scan(ckpt_audits=[dict(index=5, tensors=96, differing=['3:readback'], **bad)], **base), stats)
            self.assertTrue(any('[PREFIX-AUDIT-CKPT] line 5' in problem for problem in problems), problems)

    def test_the_tier_must_have_attached(self):
        records = returning_records() + [record('pfx-latched-hit', 'hit', 'tier-latched')]
        base = dict(tier_restores=restores_of(returning_records()), tier_trouble=[dict(index=99, line='host tier latched off: kill switch x present')])
        _, missing, _ = gate.tier_findings(arm_of('tier_returning'), records, returning_events(), good_scan(tier_io=[], tier_on=[], **base), dict(GOOD_STATS, tier_latched=1))
        self.assertTrue(any('IO attached' in text for text in missing) and any('"host tier on"' in text for text in missing), missing)

    def test_no_stats_export_is_not_exercised_not_a_pass(self):
        problems, missing, _ = gate.tier_findings(arm_of('tier_attach'), [], {}, good_scan(), None)
        self.assertTrue(any('no registry stats export' in text for text in missing))

    def test_the_attach_arm_asks_only_for_the_attach_and_the_audit_lines(self):
        problems, missing, lines = gate.tier_findings(arm_of('tier_attach'), [], {}, good_scan(), dict(GOOD_STATS, tier_spill_blocks=0, tier_restore_requests=0, tier_restore_blocks=0))
        self.assertEqual((problems, missing), ([], []), 'a chain with no eviction spills and restores nothing: that is not the attach arm\'s to ask')

    def timed_event(self, nominal, **extra):
        values = dict(nominal=nominal, size=nominal, build_tag='b%d' % nominal, build_prompt_tokens=nominal, build_ttft_s=11.5, resident_tag='r%d' % nominal,
                      resident_prompt_tokens=nominal + 1700, resident_ttft_s=2.5, return_tag='t%d' % nominal, return_prompt_tokens=nominal + 3400, return_ttft_s=6.2,
                      return_ok=True)
        values.update(extra)
        return values

    def test_the_timed_arm_reports_every_size_and_needs_a_restore_at_each(self):
        sizes = (32000, 128000, 254000)
        events = dict(('tier-timed-%d' % n, self.timed_event(n)) for n in sizes)
        records = [record('t%d' % n, 'hit', 'tier-timed-return', q=n, expected=n, window=(100 * index, 100 * index + 50)) for index, n in enumerate(sizes)]
        restores = [dict(index=100 * index + 10, tag='t%d' % n, blocks=n // 64, bytes=(n // 64) * 2228224, ms=900.0 * (index + 1), **{'from': 0, 'to': n}) for index, n in enumerate(sizes)]
        spills = [dict(index=100 * index + 20, blocks=n // 64, bytes=(n // 64) * 2228224, ms=800.0, held=1) for index, n in enumerate(sizes)]
        scanned = good_scan(tier_restores=restores, tier_spills=spills, ckpt_audits=[])
        arm = arm_of('tier_timed', audited=False)
        problems, missing, lines = gate.tier_findings(arm, records, events, scanned, dict(GOOD_STATS))
        self.assertEqual((problems, missing), ([], []))
        timed = [line for line in lines if line.startswith('tier-timed-')]
        self.assertEqual([line.split(':')[0] for line in timed], ['tier-timed-32000', 'tier-timed-128000', 'tier-timed-254000'], 'by size, not as strings')
        self.assertIn('cold prefill TTFT 11.5 s', timed[0])
        self.assertIn('device-resident hit TTFT 2.5 s', timed[0])
        self.assertIn('returning hit TTFT 6.2 s', timed[0])
        self.assertIn('restore 500 blocks 1.11 GB in 900.0 ms', timed[0])
        self.assertIn('displaced blocks spilled in the same request: 1.11 GB in 800.0 ms', timed[0])
        # a size with no restore line, and a returning turn that failed
        _, missing, _ = gate.tier_findings(arm, records, events, good_scan(tier_restores=restores[:2], tier_spills=spills, ckpt_audits=[]), dict(GOOD_STATS))
        self.assertTrue(any('tier-timed-254000' in text and 'no "host tier restore" line' in text for text in missing), missing)
        events['tier-timed-32000'] = self.timed_event(32000, return_ok=False)
        _, missing, _ = gate.tier_findings(arm, records, events, scanned, dict(GOOD_STATS))
        self.assertTrue(any('tier-timed-32000: the returning turn failed' in text for text in missing), missing)
        _, missing, _ = gate.tier_findings(arm, [], {}, scanned, dict(GOOD_STATS))
        self.assertTrue(any('no timed session was built' in text for text in missing))


# -- (3) the scenarios on the fake ----------------------------------------------------------------------------------

class TierEngineTests(unittest.TestCase):
    """The fake itself: what the end-to-end runs rest on."""

    def test_an_eviction_spills_and_the_return_restores_and_every_cached_block_holds_its_hash(self):
        engine = TierFakeEngine(profile=tier_profiles()['profiles'][FAKE], name=FAKE)
        driver = fakes.sticky_driver(engine, 'tier-returning')
        replay.scenario_tier_returning(driver, pool_tokens=engine.num_blocks * BLOCK)
        self.assertTrue(engine.check_kv())
        stats = engine.registry.snapshot()
        self.assertGreater(stats['tier_spill_blocks'], 0)
        self.assertGreater(stats['tier_restore_blocks'], 0)
        self.assertEqual([stats[name] for name in ('tier_spill_failures', 'tier_restore_failures', 'tier_restore_refused_digest', 'tier_digest_failures', 'tier_audit_mismatches')],
                         [0] * 5)
        self.assertEqual(stats['tier_latched'], 1, 'the scenario ends with the tier\'s kill switch')
        self.assertTrue(all(pair['verdict'] == 'IDENTICAL' for pair in driver.pairs), [(p['case'], p['verdict']) for p in driver.pairs])

    def test_without_the_tier_the_same_scenario_loses_the_sessions(self):
        profile = tier_profiles()['profiles'][FAKE]
        engine = fakes.FakeEngine(profile=profile, name=FAKE)
        driver = fakes.sticky_driver(engine, 'tier-returning')
        engine.tier_switch = lambda on: None        # the plain fake has no tier file: the drill is a no-op for it
        replay.scenario_tier_returning(driver, pool_tokens=engine.num_blocks * BLOCK)
        records = fakes.resolved(driver, engine)
        after = [r for r in records if r.get('case') == 'tier-after-flood' and r['role'] == 'hit']
        lost = [r for r in after if (r['markers'].get('q') or 0) < (r.get('expected') or {}).get('q', 0)]
        self.assertTrue(lost, 'the flood evicted a session: the negative control that the tier arms are different')


class EndToEndTests(unittest.TestCase):
    """c2_prefix_gate.run_plan over TierFakeEngine: containers, derived profiles, the log, the stats file, the judge."""

    def setUp(self):
        self.results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results, True)

    def arm(self, result, name):
        return result['arms'][name]

    def test_the_whole_tiers_plan_passes_and_every_cached_block_is_right(self):
        result, harness, _ = run_tiers('tiers', self.results)
        verdicts = dict((name, arm['verdict']) for name, arm in result['arms'].items())
        detail = dict((name, arm['lines'][-12:]) for name, arm in result['arms'].items())
        self.assertEqual(verdicts, {'tier-attach': 'PASS', 'tier-returning': 'PASS', 'tier-timed': 'PASS'}, detail)
        self.assertEqual(result['verdict'], 'PASS')
        for engine in harness.engines:
            self.assertTrue(engine.check_kv())
        returning = self.arm(result, 'tier-returning')
        self.assertTrue(any(line.startswith('after the flood') for line in returning['lines']))
        timed = self.arm(result, 'tier-timed')
        self.assertEqual(len([line for line in timed['lines'] if line.startswith('tier-timed-') and 'restore ' in line]), 3, timed['lines'])

    def test_a_tier_that_never_spills_is_not_exercised_not_a_pass(self):
        result, _, _ = run_tiers('tier-returning', self.results, **{FAKE: dict(tier_never_spills=True)})
        arm = self.arm(result, 'tier-returning')
        self.assertEqual(arm['verdict'], 'NOT_EXERCISED', arm['lines'][-8:])
        self.assertTrue(any('no returning turn was restored from the tier' in text for text in arm['not_exercised']), arm['not_exercised'])

    def test_a_missing_io_line_a_missing_preconverted_line_and_a_bad_audit_line(self):
        for fault, expect, verdict in (('tier_no_io_line', 'IO attached', 'NOT_EXERCISED'), ('tier_no_preconverted', 'stored preconverted', 'NOT_EXERCISED'),
                                       ('tier_bad_ckpt_audit', 'conversion_equal=0', 'FAIL')):
            with self.subTest(fault=fault):
                results = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, results, True)
                result, _, _ = run_tiers('tier-attach', results, **{FAKE: {fault: True}})
                arm = self.arm(result, 'tier-attach')
                self.assertEqual(arm['verdict'], verdict, arm['lines'][-8:])
                self.assertTrue(any(expect in text for text in arm['problems'] + arm['not_exercised']), (arm['problems'], arm['not_exercised']))

    def test_the_results_directory_keeps_the_arms_logs_and_the_derived_profile_of_the_timed_arm(self):
        run_tiers('tier-timed', self.results)
        arm_dir = os.path.join(self.results, 'tier-timed')
        self.assertTrue(os.path.isfile(os.path.join(arm_dir, 'server-final.log')))
        with open(os.path.join(arm_dir, 'profiles.json'), encoding='utf-8') as handle:
            derived = json.load(handle)
        env = derived['profiles'][FAKE + '+tiertime']['env']
        self.assertEqual((env['QWEN_PREFIX_HOST_TIER_AUDIT'], env['QWEN_PREFIX_HOST_TIER_VERIFY']), ('0', 'sample'))


class BenchTests(unittest.TestCase):
    def test_the_cpu_bench_runs_on_the_adapter_in_the_model_graft_at_the_production_geometry(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest('numpy is not installed')
        import bench_prefix_tier
        lines = []
        result = bench_prefix_tier.run(4, out=lines.append)
        self.assertEqual(result['block_bytes'], 32 * 4 * 17408, 'the geometry docs/prefix-store-hygiene.md plans with')
        text = '\n'.join(lines)
        for phrase in ('adapter read, into slab (slots resident)', 'adapter write from slab views', 'store put 4 blocks', 'digest worker finished', 'verify (sha256 + compare)'):
            self.assertIn(phrase, text)
        self.assertEqual(bench_prefix_tier.main(['--blocks', '2']), 0)


if __name__ == '__main__':
    unittest.main()
