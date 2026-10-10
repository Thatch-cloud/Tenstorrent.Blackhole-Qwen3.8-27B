"""The tier-quota plan (tp4/prefix-tiers; docs/prefix-store-hygiene.md "Tenant quotas"): tenant A floods the device pool while tenant B's session must still be a
device-resident hit when it returns. The scenario (prefix_replay.scenario_tier_quota), its judge (c2_prefix_gate.quota_findings), the plan wiring (quota-control and
quota-isolated, the derived quota env), the contract's rules for the quota names, and the whole thing end to end on the tier fake of test_prefix_tier_gate with its pool
made vLLM-shaped (a free queue, ref counts) and the real DeviceQuota attached to it.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import make_prefix_tier_profiles as tier_twin  # noqa: E402
import prefix_markers as markers  # noqa: E402
import prefix_replay as replay  # noqa: E402
import qwen_prefix_registry as registry_module  # noqa: E402
import qwen_prefix_scheduler_patch as patch_module  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import test_c2_prefix_gate as gate_tests  # noqa: E402
import test_prefix_replay as fakes  # noqa: E402
import test_prefix_tier_gate as tiers  # noqa: E402
from test_qwen_prefix_quota import ShimQueue  # noqa: E402

FAKE = tiers.FAKE
REAL_PROFILES = HERE / 'qwen_c2_profiles.json'
AUDIT_TWIN = tier_twin.SHIP_AUDIT


# -- the fake: a vLLM-shaped pool under the tier fake, with the real DeviceQuota on it ------------------------------------------------

class QuotaBlock(fakes.FakeBlock):
    """The fake block with vLLM's free-queue links and its name for the reference count."""
    __slots__ = ('prev_free_block', 'next_free_block')

    def __init__(self, block_id):
        super().__init__(block_id)
        self.prev_free_block = self.next_free_block = None

    @property
    def ref_cnt(self):
        return self.ref


class FreeView(object):
    def __init__(self, queue):
        self.queue = queue

    def __len__(self):
        return self.queue.num_free_blocks


class QuotaPool(tiers.TierPool):
    """TierPool over vLLM's free queue (the linked list the quota walks): allocation pops its head, a free appends cached blocks and prepends the others, touch removes."""

    def __init__(self, count):
        super().__init__(count)
        self.blocks = [QuotaBlock(index) for index in range(count)]
        self.queue = ShimQueue(list(self.blocks))
        self.free_block_queue = self.queue

    @property
    def free(self):
        return FreeView(self.queue)

    @free.setter
    def free(self, value):          # FakePool.__init__ assigns it once
        pass

    def touch(self, blocks):
        for block in blocks:
            if block.ref == 0:
                self.queue.remove(block)
            block.ref += 1

    def get_new_blocks(self, count):
        if count > self.queue.num_free_blocks:
            raise ValueError('cannot get %d free blocks' % count)
        taken = self.queue.popleft_n(count)
        for block in taken:
            self.evict(block)
            block.ref = 1
        return taken

    def allocate(self, count):
        if count > self.queue.num_free_blocks:
            return None
        return self.get_new_blocks(count)

    def free_blocks(self, ordered_blocks):
        cached, plain = [], []
        for block in ordered_blocks:
            block.ref -= 1
            if block.ref == 0:
                (plain if block.block_hash is None else cached).append(block)
        self.queue.prepend_n(plain)
        self.queue.append_n(cached)

    def release(self, blocks):
        self.free_blocks(list(reversed(blocks)))


class QuotaFakeEngine(tiers.TierFakeEngine):
    """The tier fake over QuotaPool, with the real DeviceQuota attached exactly as install() does it (DeviceQuota.attach)."""

    def make_pool(self, count):
        return QuotaPool(count)

    def attach_extras(self):
        super().attach_extras()
        config = self.registry.quota
        if config is not None and config.device_on:
            quota = patch_module.DeviceQuota(self.graft, config)
            self.graft.quota = quota
            self.registry.quota_gauge = quota.gauge
            quota.attach(self.manager)
            self.say('(EngineCore pid=9) INFO | scheduler:install:1 - [PINDIAG] prefix: tenant quota on device_cap_tokens=%d device_floor_tokens=%d host_cap_bytes=%d '
                     'host_floor_bytes=%d device=1 host=%d' % (config.device_cap_blocks * 64, config.device_floor_blocks * 64, config.host_cap_bytes,
                                                                config.host_floor_bytes, int(config.host_on)))

    def make_registry(self):
        """TierFakeEngine's registry, with the profile's quota names in its environment (the fake builds the environment by hand)."""
        environ = dict(QWEN_PREFIX_STORE_GIB=str(self.store_gib))
        if self.tier_configured():
            env = self.profile['env']
            environ.update(tiers.tier_environ(self.BLOCKS, int(self.store_gib * (1 << 30)),
                                              QWEN_PREFIX_HOST_TIER_AUDIT=env.get('QWEN_PREFIX_HOST_TIER_AUDIT', '0'),
                                              QWEN_PREFIX_HOST_TIER_VERIFY=env.get('QWEN_PREFIX_HOST_TIER_VERIFY', 'sample'),
                                              QWEN_PREFIX_HOST_TIER_OFF_PATH=self.tier_off_path))
        environ.update((name, str(value)) for name, value in self.profile['env'].items() if name in registry_module.QUOTA_ENV_NAMES)
        return tiers.graft.PrefixRegistry(environ=environ)


class BlindQuotaEngine(QuotaFakeEngine):
    """A defect: the quota is configured and never attached (the pool is plain vLLM order)."""

    def attach_extras(self):
        tiers.TierFakeEngine.attach_extras(self)
        config = self.registry.quota
        if config is not None:
            self.say('(EngineCore pid=9) INFO | scheduler:install:1 - [PINDIAG] prefix: tenant quota on device_cap_tokens=%d device_floor_tokens=%d host_cap_bytes=0 '
                     'host_floor_bytes=0 device=1 host=0' % (config.device_cap_blocks * 64, config.device_floor_blocks * 64))


# The quota numbers of the fake (its pool is four 40,960-token contexts: 2,560 blocks): a floor above B's session (about 460 blocks), a cap well under the pool.
FAKE_QUOTA = dict(QUOTA_DEVICE_CAP_TOKENS=65536, QUOTA_DEVICE_FLOOR_TOKENS=32768, QUOTA_HOST_CAP_GIB=0.00004, QUOTA_HOST_FLOOR_GIB=0.00001)


def quota_environ_for_fake():
    """The fake's tier is sized in 32-byte payloads (40,000 of them), so the host quota is tiny fractions of a GiB."""
    return FAKE_QUOTA


# -- synthetic material for the judge -------------------------------------------------------------------------------------------

def record(case, tag, q, expected, ok=True, role='hit', ttft=1.0):
    return dict(case=case, tag=tag, role=role, ok=ok, ttft_s=ttft, markers=dict(q=q), expected=dict(q=expected), conv=case, finish='stop')


def event(**values):
    base = dict(b_size=56000, a_size=32000, b_return_tag='b-ret', b_return_ok=True, a_return_tag='a-ret', a_return_ok=True, b_return_ttft_s=1.0, a_return_ttft_s=2.0)
    base.update(values)
    return base


def quota_on(host=1):
    return dict(index=3, time=None, device_cap_tokens=262144, device_floor_tokens=65536, host_cap_bytes=12 << 30 if host else 0, host_floor_bytes=4 << 30 if host else 0,
                device=1, host=host)


STATS = dict(quota_dev_demoted_blocks=900, quota_dev_evictions_own=800, quota_dev_evictions_global=0, quota_dev_floor_protected=40, quota_dev_floor_overrides=0,
             quota_dev_tenants=3, quota_dev_at_cap=1, quota_host_evictions_own=300, quota_host_evictions_global=0, quota_host_floor_protected=10, quota_host_tenants=3,
             quota_host_at_cap=1)


def arm(isolated=True, tier=True):
    return dict(arm='quota-isolated' if isolated else 'quota-control', scenario='tier_quota', kind='quota' if isolated else None,
                tier=dict(gib=32.0, audit=True, preconverted=True) if tier else None)


class JudgeTests(unittest.TestCase):
    def judge(self, arm_=None, records=None, event_=None, on=None, restores=(), stats=None):
        records = records if records is not None else [record('quota-b-return', 'b-ret', 54000, 54000), record('quota-a-return', 'a-ret', 30000, 30000)]
        scanned = dict(quota_on=[quota_on()] if on is None else on, tier_restores=[dict(tag=tag) for tag in restores])
        return gate.quota_findings(arm_ or arm(), records, {'tier-quota': event_ if event_ is not None else event()}, scanned, dict(STATS) if stats is None else stats)

    def test_the_isolated_arm_passes_when_b_is_resident_and_the_cap_bit(self):
        problems, missing, lines = self.judge()
        self.assertEqual((problems, missing), ([], []), lines)
        text = '\n'.join(lines)
        for needle in ('tenant B returned: Q=54000 (the oracle\'s 54000), device-resident', 'quota: {"device_cap_tokens": 262144', 'device quota: 900 blocks demoted',
                       'host quota: evicted 300 records by the cap'):
            self.assertIn(needle, text)

    def test_a_b_that_was_restored_from_host_ram_or_lost_blocks_fails_the_isolated_arm(self):
        problems, _, lines = self.judge(restores=['b-ret'])
        self.assertTrue(problems and 'not a device-resident hit' in problems[0] and 'restored from host RAM' in problems[0], problems)
        short = [record('quota-b-return', 'b-ret', 20480, 54000), record('quota-a-return', 'a-ret', 30000, 30000)]
        problems, _, _ = self.judge(records=short)
        self.assertTrue(problems and 'tenant A\'s flood reached B\'s blocks' in problems[0], problems)

    def test_overridden_floors_make_the_arm_not_exercised_not_failed(self):
        short = [record('quota-b-return', 'b-ret', 0, 54000), record('quota-a-return', 'a-ret', 30000, 30000)]
        problems, missing, _ = self.judge(records=short, stats=dict(STATS, quota_dev_floor_overrides=12))
        self.assertEqual(problems, [])
        self.assertTrue(any('overridden 12 times' in text for text in missing), missing)

    def test_the_control_arm_must_lose_b_or_it_proves_nothing(self):
        lost = [record('quota-b-return', 'b-ret', 0, 54000), record('quota-a-return', 'a-ret', 0, 30000)]
        problems, missing, lines = self.judge(arm_=arm(isolated=False), records=lost, on=[], stats={})
        self.assertEqual((problems, missing), ([], []), lines)
        problems, missing, _ = self.judge(arm_=arm(isolated=False), on=[], stats={})
        self.assertTrue(any('control arm kept tenant B\'s session in the pool' in text for text in missing), missing)
        restored = self.judge(arm_=arm(isolated=False), on=[], stats={}, restores=['b-ret'])
        self.assertEqual(restored[1], [], 'restored from host RAM is not resident: the flood reached B')

    def test_a_control_that_installed_a_quota_is_a_problem(self):
        lost = [record('quota-b-return', 'b-ret', 0, 54000), record('quota-a-return', 'a-ret', 0, 30000)]
        problems, _, _ = self.judge(arm_=arm(isolated=False), records=lost, on=[quota_on()], stats={})
        self.assertTrue(any('control arm installed a tenant quota' in text for text in problems), problems)

    def test_the_install_line_and_the_counters_are_asked_of_the_isolated_arm(self):
        _, missing, _ = self.judge(on=[])
        self.assertTrue(any('no "tenant quota on" line' in text for text in missing), missing)
        _, missing, _ = self.judge(stats=dict(STATS, quota_dev_evictions_own=0))
        self.assertTrue(any('the cap never bit' in text for text in missing), missing)
        _, missing, _ = self.judge(stats=dict(STATS, quota_host_evictions_own=0))
        self.assertTrue(any('the host cap never bit' in text for text in missing), missing)
        _, missing, _ = self.judge(on=[quota_on(host=0)], stats=dict(STATS, quota_host_evictions_own=0))
        self.assertEqual(missing, [], 'no host quota configured: none asked')

    def test_a_failed_return_or_a_scenario_that_never_ran_is_named(self):
        _, missing, _ = self.judge(event_=event(b_return_ok=False))
        self.assertTrue(any('tenant B\'s returning turn failed' in text for text in missing), missing)
        self.assertTrue(any('no tier-quota event' in text for text in gate.quota_findings(arm(), [], {}, dict(quota_on=[]), {})[1]))
        self.assertTrue(any('cannot hold the two sessions' in text for text in self.judge(event_=event(skipped=True))[1]))

    def test_no_tenant_is_named_in_what_the_judge_prints(self):
        _, _, lines = self.judge()
        self.assertFalse(any('salt' in line.lower() for line in lines))


# -- the plan wiring -----------------------------------------------------------------------------------------------------------

class PlanTests(unittest.TestCase):
    def test_the_plan_and_its_two_arms_are_registered(self):
        self.assertIn('tier-quota', job.PREFIX_PLANS)
        self.assertEqual([arm_ for arm_, plan in job.PREFIX_ARM_PLANS if plan == 'tier-quota'], ['quota-control', 'quota-isolated'])
        for name in ('tier-quota', 'quota-control', 'quota-isolated'):
            self.assertIn(name, gate.PLANS)
        self.assertEqual([entry[0] for entry in gate.PLAN_ARMS['tier-quota']], ['quota-control', 'quota-isolated'])
        self.assertEqual([entry[3] for entry in gate.PLAN_ARMS['tier-quota']], [None, 'quota'])
        for name in ('quota-control', 'quota-isolated'):
            self.assertEqual([entry[0] for entry in gate.PLAN_ARMS[name]], [name])
        self.assertIn('tier_quota', replay.SCENARIOS)
        self.assertIn('tier_quota', gate.TIER_SCENARIOS)
        self.assertEqual(gate.QUOTA_SCENARIOS, ('tier_quota',))
        self.assertNotIn('tier_quota', gate.TIMED_SCENARIOS)

    def test_the_isolated_arm_is_the_profile_with_the_quota_env_and_the_host_numbers_only_with_a_tier(self):
        document = json.loads(REAL_PROFILES.read_text(encoding='utf-8'))
        arms = gate.plan_arms('tier-quota', AUDIT_TWIN, 'none', document)
        self.assertEqual([a['arm'] for a in arms], ['quota-control', 'quota-isolated'])
        control, isolated = arms
        self.assertIsNone(control['derived'])
        env = isolated['derived']['profiles'][isolated['served']]['env']
        self.assertEqual((env[registry_module.ENV_QUOTA_DEVICE_CAP], env[registry_module.ENV_QUOTA_DEVICE_FLOOR]), ('262144', '65536'))
        self.assertEqual((env[registry_module.ENV_QUOTA_HOST_CAP], env[registry_module.ENV_QUOTA_HOST_FLOOR]), ('12', '4'))
        base = document['profiles'][AUDIT_TWIN]
        self.assertFalse(set(registry_module.QUOTA_ENV_NAMES) & set(base['env']), 'the audit twin itself names no quota')
        self.assertEqual(gate.quota_env({'env': {}}), {registry_module.ENV_QUOTA_DEVICE_CAP: '262144', registry_module.ENV_QUOTA_DEVICE_FLOOR: '65536'}, 'no tier: no host quota')
        self.assertEqual(contract.prefix_policy_problems(dict(base, env=dict(base['env'], **env))), [], 'the contract accepts the derived env on the gate-only twin')
        self.assertTrue(gate.quota_env({'env': {}}) and registry_module.quota_config(gate.quota_env({'env': {}})).device_on)
        config = registry_module.quota_config(env, tier_kv_bytes=24 << 30)
        self.assertTrue(config.device_on and config.host_on, 'the gate\'s numbers pass the engine\'s strict parser')

    def test_the_quota_arm_is_audited_so_it_needs_the_audit_twin(self):
        document = json.loads(REAL_PROFILES.read_text(encoding='utf-8'))
        with self.assertRaises(gate.PlanError):
            gate.plan_arms('tier-quota', tier_twin.SHIP, 'none', document)

    def test_the_image_check_is_asked_of_a_tier_arm_and_the_dry_run_plans_both(self):
        results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, results, True)
        lines = []
        code = gate.main(['--image', 'img', '--profiles', str(REAL_PROFILES), '--results', results, '--profile', AUDIT_TWIN, '--plan', 'tier-quota', '--baseline', 'none',
                          '--dry-run', '--cards', 'quad'], devices=['/a', '/b', '/c', '/d'], log=lines.append)
        self.assertEqual(code, 0, lines)
        rows = [json.loads(line) for line in lines if line.startswith('{') and '"docker"' in line]
        self.assertEqual([row['arm'] for row in rows], ['quota-control', 'quota-isolated'])
        self.assertEqual(rows[1]['served'], AUDIT_TWIN + '+quota')
        self.assertTrue(any('profiles.json' in token for token in rows[1]['docker']), 'the derived profile is mounted')
        self.assertIn('QWEN_PREFIX_DIGESTS=1', rows[0]['docker'])
        self.assertEqual(rows[0]['timeout'], 9000)

    def test_the_marker_scan_reads_the_install_line(self):
        scan = markers.scan(['[PINDIAG] prefix: tenant quota on device_cap_tokens=262144 device_floor_tokens=65536 host_cap_bytes=0 host_floor_bytes=0 device=1 host=0'])
        self.assertEqual([(e['device_cap_tokens'], e['device_floor_tokens'], e['device'], e['host']) for e in scan['quota_on']], [(262144, 65536, 1, 0)])
        self.assertEqual(markers.scan(['[PINDIAG] prefix: host tier on gib=1'])['quota_on'], [])


class CompareTests(unittest.TestCase):
    """tier_concurrent_compare --quota: the two arms' returning answers, whole."""

    def arm_records(self, b_ids=(1, 2, 3), a_ids=(4, 5)):
        return [dict(record('quota-b-return', 'b', 5, 5), token_ids=list(b_ids), sent_s=10.0), dict(record('quota-b-return-cold', 'bc', 0, 0, role='cold'), token_ids=[9], sent_s=11.0),
                dict(record('quota-a-return', 'a', 5, 5), token_ids=list(a_ids), sent_s=12.0)]

    def test_identical_answers_pass_and_a_difference_fails(self):
        import tier_concurrent_compare as compare

        same = compare.compare_quota(self.arm_records(), self.arm_records(), 'control', 'isolated')
        self.assertEqual((same['verdict'], same['problems']), ('PASS', []), same['lines'])
        self.assertEqual([row['case'] for row in same['whole']], ['quota-b-return', 'quota-a-return'])
        other = compare.compare_quota(self.arm_records(), self.arm_records(b_ids=(1, 2, 4)))
        self.assertEqual(other['verdict'], 'FAIL')
        self.assertTrue(any(text.startswith('quota-b-return DIVERGED') for text in other['problems']), other['problems'])
        self.assertEqual(compare.compare_quota([], [])['verdict'], 'FAIL')
        half = compare.compare_quota(self.arm_records(), self.arm_records()[:2])
        self.assertTrue(any('quota-a-return' in text for text in half['problems']), half['problems'])

    def test_the_command_line_reads_two_result_directories(self):
        import tier_concurrent_compare as compare

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(tmp), True)
        for name, ids in (('quota-control', (1, 2, 3)), ('quota-isolated', (1, 2, 3))):
            (tmp / name).mkdir()
            (tmp / name / 'records.jsonl').write_text(replay.records_jsonl(self.arm_records(b_ids=ids)), encoding='utf-8')
        out = []
        self.assertEqual(compare.main(['--quota', str(tmp / 'quota-control'), str(tmp / 'quota-isolated')], out=out.append), 0, out)
        self.assertIn('tier quota control vs isolated: PASS', out)
        nested = tmp / 'job'
        (nested / 'quota-isolated').mkdir(parents=True)
        (nested / 'quota-isolated' / 'records.jsonl').write_text(replay.records_jsonl(self.arm_records()), encoding='utf-8')
        out = []
        self.assertEqual(compare.main(['--quota', str(tmp / 'quota-control'), str(nested)], out=out.append), 0, out)
        out = []
        self.assertEqual(compare.main(['--quota', str(tmp / 'quota-control')], out=out.append), 2)
        (tmp / 'quota-isolated' / 'records.jsonl').write_text(replay.records_jsonl(self.arm_records(a_ids=(7,))), encoding='utf-8')
        out = []
        self.assertEqual(compare.main(['--quota', str(tmp / 'quota-control'), str(tmp / 'quota-isolated')], out=out.append), 1, out)


class ContractTests(unittest.TestCase):
    BASE = {'env': {'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_STICKY_SESSIONS': '1'}, 'engine': {'enable-prefix-caching': True}, 'gate_only': True}

    def problems(self, **env):
        profile = json.loads(json.dumps(self.BASE))
        profile['env'].update(env)
        return contract.prefix_policy_problems(profile)

    def test_the_names_are_the_registrys(self):
        self.assertEqual(set(contract.PREFIX_QUOTA_NAMES), set(registry_module.QUOTA_ENV_NAMES))

    def test_a_good_set_is_clean_and_each_bad_value_is_named(self):
        self.assertEqual(self.problems(**{registry_module.ENV_QUOTA_DEVICE_CAP: '262144', registry_module.ENV_QUOTA_DEVICE_FLOOR: '65536'}), [])
        cases = [({registry_module.ENV_QUOTA_DEVICE_CAP: '1e3'}, 'whole number of 64-token blocks'), ({registry_module.ENV_QUOTA_DEVICE_CAP: '100'}, '64-token'),
                 ({registry_module.ENV_QUOTA_DEVICE_CAP: '1024'}, 'below one 2048-token chunk'),
                 ({registry_module.ENV_QUOTA_DEVICE_CAP: '4096', registry_module.ENV_QUOTA_DEVICE_FLOOR: '8192'}, 'exceeds'),
                 ({registry_module.ENV_QUOTA_HOST_CAP: 'inf'}, 'finite'), ({registry_module.ENV_QUOTA_HOST_CAP: '1'}, 'inert')]
        for env, needle in cases:
            with self.subTest(env=env):
                found = self.problems(**env)
                self.assertTrue(found and needle in ' '.join(found), found)

    def test_a_traffic_profile_may_not_name_a_quota_and_a_profile_without_reuse_is_inert(self):
        profile = json.loads(json.dumps(self.BASE))
        profile['gate_only'] = False
        profile['env'][registry_module.ENV_QUOTA_DEVICE_CAP] = '262144'
        found = contract.prefix_policy_problems(profile)
        self.assertTrue(any('no qualification record exists for the per-tenant quotas' in text for text in found), found)
        inert = {'env': {registry_module.ENV_QUOTA_DEVICE_CAP: '262144'}, 'engine': {}, 'gate_only': True}
        self.assertTrue(any('inert' in text for text in contract.prefix_policy_problems(inert)))

    def test_an_inherited_quota_is_dropped_under_a_profile_that_names_none_and_kept_under_one_that_does(self):
        profile = dict(json.loads(REAL_PROFILES.read_text(encoding='utf-8'))['profiles'][AUDIT_TWIN], name=AUDIT_TWIN)
        inherited = {name: '262144' for name in contract.PREFIX_QUOTA_NAMES}
        self.assertFalse(set(contract.apply_environment(json.loads(json.dumps(profile)), dict(inherited))) & set(contract.PREFIX_QUOTA_NAMES))
        named = json.loads(json.dumps(profile))
        named['env'][registry_module.ENV_QUOTA_DEVICE_CAP] = '131072'
        environ = contract.apply_environment(named, dict(inherited))
        self.assertEqual(environ[registry_module.ENV_QUOTA_DEVICE_CAP], '131072')
        self.assertNotIn(registry_module.ENV_QUOTA_DEVICE_FLOOR, environ)

# -- end to end ------------------------------------------------------------------------------------------------------------------

def quota_document(**env):
    document = tiers.tier_profiles(**env)
    return document


class EndToEndTests(unittest.TestCase):
    """The tier-quota plan over QuotaFakeEngine: the control loses tenant B to the flood, the isolated arm (the real DeviceQuota on a vLLM-shaped queue) keeps it."""

    def setUp(self):
        self.results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results, True)
        for name, value in FAKE_QUOTA.items():
            patcher = mock.patch.object(gate, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(gate, 'SETTLE_SECONDS', 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_plan(self, plan, engine_class=QuotaFakeEngine, **faults):
        document = quota_document()
        harness = gate_tests.Harness(engine_class, document=document, **faults)
        runner = harness.runner(self.results)
        arms = gate.plan_arms(plan, FAKE, 'none', document)
        return gate.run_plan(plan, arms, runner, gate_tests.GOOD_ANCHOR), harness

    def test_the_control_loses_b_and_the_isolated_arm_keeps_it(self):
        result, harness = self.run_plan('tier-quota')
        control, isolated = result['arms']['quota-control'], result['arms']['quota-isolated']
        detail = dict(control=(control['problems'], control['not_exercised'], control['lines'][-10:]), isolated=(isolated['problems'], isolated['not_exercised'], isolated['lines'][-10:]))
        self.assertEqual((control['verdict'], isolated['verdict']), ('PASS', 'PASS'), detail)
        self.assertEqual(result['verdict'], 'PASS')
        text = '\n'.join(isolated['lines'])
        self.assertIn('tenant B returned: Q=', text)
        self.assertIn('device-resident', text)
        self.assertIn('device quota:', text)
        self.assertTrue(any('tenant B returned' in line and 'device-resident' not in line for line in control['lines']), control['lines'])
        for engine in harness.engines:
            self.assertTrue(engine.check_kv())
        quota_engine = harness.engines[-1]
        self.assertGreater(quota_engine.registry.stats['quota_dev_evictions_own'], 0)

    def test_the_floor_alone_keeps_b_in_the_pool_and_is_counted(self):
        with mock.patch.object(gate, 'QUOTA_DEVICE_CAP_TOKENS', 0):
            result, harness = self.run_plan('quota-isolated')
        isolated = result['arms']['quota-isolated']
        self.assertEqual(isolated['verdict'], 'PASS', (isolated['problems'], isolated['not_exercised'], isolated['lines'][-8:]))
        stats = harness.engines[-1].registry.stats
        self.assertGreater(stats['quota_dev_floor_protected'], 0, 'the flood walked into B\'s blocks and they were set aside')
        self.assertEqual(stats['quota_dev_evictions_own'], 0, 'no cap: nothing was demoted')
        self.assertEqual(stats['quota_dev_floor_overrides'], 0)

    def test_every_pair_of_both_arms_is_identical(self):
        result, _ = self.run_plan('tier-quota')
        for name, arm_result in result['arms'].items():
            self.assertTrue(any(line.startswith('pairs: ') and ' 0 failed' in line for line in arm_result['lines']), (name, arm_result['lines'][:6]))

    def test_a_quota_that_is_configured_and_never_attached_fails_the_isolated_arm(self):
        result, _ = self.run_plan('quota-isolated', engine_class=BlindQuotaEngine)
        isolated = result['arms']['quota-isolated']
        self.assertEqual(isolated['verdict'], 'FAIL', (isolated['problems'], isolated['not_exercised'], isolated['lines'][-8:]))
        self.assertTrue(any('not a device-resident hit' in text for text in isolated['problems']), isolated['problems'])

    def test_the_isolated_arm_writes_a_derived_profile_with_the_quota_env(self):
        self.run_plan('quota-isolated')
        with open(os.path.join(self.results, 'quota-isolated', 'profiles.json'), encoding='utf-8') as handle:
            derived = json.load(handle)
        env = derived['profiles'][FAKE + '+quota']['env']
        self.assertEqual(env[registry_module.ENV_QUOTA_DEVICE_CAP], str(FAKE_QUOTA['QUOTA_DEVICE_CAP_TOKENS']))


if __name__ == '__main__':
    unittest.main()
