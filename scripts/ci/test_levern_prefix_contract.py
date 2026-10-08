"""Lever N with prefix reuse: the two gate-only profiles and the contract's merged-route rules (serving_c2_contract.levern_problems, merged_route_problems;
docs/lever-n-prefix-merged-route.md sections 4 (R1, R4) and 12.3).

  c2-packed-tp4-8x262k-ship-prefix-levern        = the production profile (c2-packed-tp4-8x262k-ship-prefix) plus exactly the Lever N flags, gate only
  c2-packed-tp4-8x262k-ship-prefix-levern-audit  = its audited twin (c2-packed-tp4-8x262k-ship-prefix-audit) plus the same flags, the Lever N digest
                                                   instrument and the prefix route's digests, gate only

What is held: each profile is its twin plus exactly its flags and nothing else (the engine argv unchanged); both pass the contract; the production
profile and its audit twin are untouched byte for byte and name no Lever N flag; every requirement of the merged route is refused by name (one lever
without the other, an image whose route lacks a source, a merged flag that would do nothing, the route epoch scope without its pre-stage); the argv
and environment of the unmerged profiles are what they were; the merged profile arms the platform hook."""

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import levern_policy  # noqa: E402
import levern_route  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
from test_qwen_prefix_image import boot_api_server  # noqa: E402

PROFILES = HERE / 'qwen_c2_profiles.json'
MERGE_BASE = '0cab164f'
SHIP, SHIP_AUDIT = 'c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-audit'
PLAIN, AUDITED = 'c2-packed-tp4-8x262k-ship-prefix-levern', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit'
BASE = {PLAIN: SHIP, AUDITED: SHIP_AUDIT}
FLAGS = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_STEP_TOKENS': '2048', 'QWEN_FAST_LEVERN_SOLO_STEP_TOKENS': '16384',
         'QWEN_FAST_LEVERN_PREFILL_SHARE': '0.5', 'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180', 'QWEN_FAST_LEVERN_SHORT_TOKENS': '16384',
         'QWEN_FAST_LEVERN_PARK': 'host', 'QWEN_FAST_LEVERN_PARK_SLOTS': '1', 'QWEN_FAST_LEVERN_MAX_PARK_S': '30',
         'QWEN_FAST_LEVERN_EPOCH_SCOPE': 'route', 'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S': '8'}
EXTRA = {PLAIN: FLAGS, AUDITED: dict(FLAGS, QWEN_FAST_LEVERN_AUDIT='1', QWEN_PREFIX_DIGESTS='1')}


def profiles():
    return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']


def named(name):
    return dict(copy.deepcopy(profiles()[name]), name=name)


def flat(profile):
    out = copy.deepcopy(profile)
    out.pop('description')
    out.pop('name', None)
    return out


class ProfileTests(unittest.TestCase):
    def test_each_profile_is_its_twin_plus_exactly_its_flags(self):
        found = profiles()
        for name, base in BASE.items():
            with self.subTest(name=name):
                expected = flat(found[base])
                expected['env'].update(EXTRA[name])
                expected['gate_only'] = True
                self.assertEqual(flat(found[name]), expected)

    def test_the_engine_argv_is_the_production_argv_byte_for_byte(self):
        for name, base in BASE.items():
            with self.subTest(name=name):
                self.assertEqual(contract.engine_arguments(named(name), '/snapshot'), contract.engine_arguments(named(base), '/snapshot'))
                self.assertIs(profiles()[name]['engine']['enable-chunked-prefill'], True)
                self.assertEqual(profiles()[name]['engine']['max-num-batched-tokens'], profiles()[name]['engine']['max-model-len'])

    def test_both_are_gate_only_with_no_waiver_and_the_production_profile_is_not(self):
        found = profiles()
        for name in (PLAIN, AUDITED):
            self.assertIs(found[name]['gate_only'], True)
            self.assertNotIn('QWEN_FAST_262K_EVIDENCE_WAIVER', found[name]['env'])
            self.assertNotIn('QWEN_C2_GATE_PROFILE', found[name]['env'])
        self.assertNotIn('gate_only', found[SHIP])
        self.assertEqual(json.loads(PROFILES.read_text(encoding='utf-8'))['default'], 'c2-packed-tp4')

    def test_both_pass_every_contract_check_that_names_them(self):
        for name in (PLAIN, AUDITED):
            with self.subTest(name=name):
                profile = named(name)
                self.assertEqual(contract.levern_problems(profile), [])
                self.assertEqual(contract.prefix_reuse_problems(profile), [])
                self.assertEqual(contract.merged_route_problems(profile, {key: str(value) for key, value in profile['env'].items()}), [])
                self.assertTrue(contract.levern_on(profile))
                self.assertTrue(contract.prefix_reuse(profile) and contract.sticky_sessions(profile))

    def test_the_flags_parse_to_what_the_description_says(self):
        env = profiles()[PLAIN]['env']
        cfg, merged = levern_policy.config(env), levern_policy.merged_config(env)
        self.assertEqual((cfg.step, cfg.solo, cfg.share, cfg.rounds), (2048, 16384, 0.5, None))
        self.assertEqual(merged, levern_policy.Merged(180, 16384, 'host', 1, 30, 'route', 8))
        self.assertEqual(levern_policy.config_problems(env), [])

    def test_the_production_profiles_are_untouched_and_name_no_lever_n_flag(self):
        found = profiles()
        for name in (SHIP, SHIP_AUDIT):
            self.assertEqual([key for key in found[name]['env'] if key in levern_policy.ALL_FLAGS], [])
            self.assertFalse(contract.levern_on(named(name)))
            self.assertEqual(contract.levern_problems(named(name)), [])
        for name in (SHIP, SHIP_AUDIT):
            try:
                old = subprocess.run(['git', 'show', '%s:scripts/ci/qwen_c2_profiles.json' % MERGE_BASE], cwd=str(HERE), capture_output=True, check=True)
            except (OSError, subprocess.CalledProcessError):
                self.skipTest('no git history for %s' % MERGE_BASE)
            self.assertEqual(json.loads(old.stdout.decode('utf-8'))['profiles'][name], found[name], '%s changed since the merge' % name)

    def test_no_other_profile_gained_a_merged_flag(self):
        for name, profile in profiles().items():
            if name in (PLAIN, AUDITED) or name.startswith(PLAIN):      # the window's Lever N carriers are held by test_tp4_w2ln_profiles
                continue
            with self.subTest(name=name):
                self.assertEqual([key for key in profile['env'] if key in levern_policy.MERGED_FLAGS], [])


TRAFFIC = 'c2-packed-tp4-8x262k-ship-prefix-levern-traffic'


class TrafficProfileTests(unittest.TestCase):
    """The Lever N + prefix reuse traffic profile (short-window plan, stage 1): the gate arm's engine environment with no gate marker."""

    def test_it_is_the_gate_arm_without_gate_only_and_nothing_else(self):
        found = profiles()
        self.assertNotIn('gate_only', found[TRAFFIC])
        self.assertEqual({key for key in set(found[TRAFFIC]) | set(found[PLAIN]) if found[TRAFFIC].get(key) != found[PLAIN].get(key)}, {'description', 'gate_only'})
        self.assertFalse(found[TRAFFIC]['description'].startswith('GATE ONLY'))

    def test_it_carries_no_gate_instrument_no_waiver_and_no_gate_marker(self):
        env = profiles()[TRAFFIC]['env']
        for key in ('QWEN_FAST_LEVERN_AUDIT', 'QWEN_FAST_LEVERN_FAULT', 'QWEN_PREFIX_DIGESTS', 'QWEN_FAST_262K_EVIDENCE_WAIVER', 'QWEN_C2_GATE_PROFILE',
                    'QWEN_FAST_TP4_HOSTGAP_LOG'):
            self.assertNotIn(key, env)
        self.assertEqual(env['QWEN_FAST_LEVER_N'], '1')
        self.assertEqual((env['QWEN_PREFIX_REUSE'], env['QWEN_FAST_STICKY_SESSIONS']), ('1', '1'))

    def test_the_contract_accepts_it_and_it_boots_without_the_gate_switch(self):
        profile = named(TRAFFIC)
        self.assertTrue(contract.levern_on(profile))
        self.assertEqual(contract.levern_problems(profile), [])
        self.assertEqual(contract.gate_problems(profile, {}), [])

    def test_it_is_a_baked_default_candidate(self):
        import c2_serving_job as job

        values = {'C2_ACTIONS': 'build', 'C2_CARDS': 'quad', 'C2_IMAGE_TAG': 'tp4-serve-11', 'C2_BAKE_DEFAULT_PROFILE': TRAFFIC}
        self.assertEqual(job.read_job(values, sorted(profiles()), root=str(HERE.parent.parent))['bake_default_profile'], TRAFFIC)
        values['C2_BAKE_DEFAULT_PROFILE'] = PLAIN
        with self.assertRaises(job.JobError):
            job.read_job(values, sorted(profiles()), root=str(HERE.parent.parent))


class PinTests(unittest.TestCase):
    def test_the_contract_names_the_routes_sources(self):
        self.assertEqual(contract.MERGED_SOURCES, levern_route.SOURCES)

    def test_every_lever_n_flag_is_a_contract_flag(self):
        self.assertEqual(sorted(contract.LEVERN_ENV_FLAGS), sorted(levern_policy.ALL_FLAGS))
        for name in levern_policy.MERGED_FLAGS:
            self.assertIn(name, contract.LEVERN_ENV_FLAGS)

    def test_the_route_and_the_capture_ledger_name_the_same_attribute(self):
        import dflash_prefill_window as window

        self.assertEqual(levern_route.MERGED_ATTR, window.LEVERN_MERGED_ATTR)


class RefusalTests(unittest.TestCase):
    def mutated(self, name=PLAIN, env=None, engine=None, drop_env=(), **top):
        profile = copy.deepcopy(named(name))
        profile['env'].update(env or {})
        for key in drop_env:
            profile['env'].pop(key, None)
        profile['engine'].update(engine or {})
        profile.update(top)
        return profile

    def refused(self, profile, needle):
        problems = contract.levern_problems(profile)
        self.assertTrue(any(needle in problem for problem in problems), (needle, problems))
        return problems

    def test_prefix_reuse_and_sticky_sessions_are_both_on_or_both_off(self):
        for dropped in ('QWEN_PREFIX_REUSE', 'QWEN_FAST_STICKY_SESSIONS'):
            with self.subTest(dropped=dropped):
                self.refused(self.mutated(drop_env=(dropped,)), 'both on or both off')
        # a Lever N profile with neither is the stage-1 shape and still passes
        stage_one = self.mutated(SHIP_AUDIT.replace('ship-prefix-audit', 'ship-prefix-levern-audit'),
                                 drop_env=('QWEN_PREFIX_REUSE', 'QWEN_FAST_STICKY_SESSIONS', 'QWEN_FAST_LEVERN_PARK', 'QWEN_FAST_LEVERN_EPOCH_SCOPE'))
        self.assertEqual([problem for problem in contract.merged_route_problems(stage_one, {k: str(v) for k, v in stage_one['env'].items()})
                          if 'both on or both off' in problem], [])

    def test_the_merged_route_needs_everything_prefix_reuse_needs(self):
        cases = (
            ('prefix caching off', dict(engine={'enable-prefix-caching': False, 'no-enable-prefix-caching': True}), 'enable-prefix-caching'),
            ('hash algo', dict(engine={'prefix-caching-hash-algo': 'xxhash'}), 'prefix-caching-hash-algo'),
            ('block size', dict(engine={'block-size': 128}), 'block-size'),
            ('speculative', dict(engine={'speculative-config': None}), None),
            ('async scheduling', dict(engine={'no-async-scheduling': False}), 'no-async-scheduling'),
        )
        for label, kwargs, needle in cases:
            if needle is None:
                continue
            with self.subTest(label):
                self.refused(self.mutated(**kwargs), needle)

    def test_the_budget_refusal_says_only_the_lever_n_cap_may_split(self):
        problems = self.refused(self.mutated(engine={'max-num-batched-tokens': 131072}), 'must equal max-model-len')
        problems = contract.prefix_reuse_problems(self.mutated(engine={'max-num-batched-tokens': 131072}))
        self.assertTrue(any('only the Lever N cap may' in problem for problem in problems), problems)

    def test_an_image_whose_route_lacks_a_source_cannot_boot_the_merged_profile(self):
        for lacking in ('PARKED', 'CHECKPOINT'):
            with self.subTest(lacking=lacking):
                sources = tuple(source for source in levern_route.SOURCES if source != lacking)
                with mock.patch.object(levern_route, 'SOURCES', sources):
                    problems = self.refused(self.mutated(), lacking)
                self.assertTrue(any('stage-1 image' in problem for problem in problems))
        # and a route with no SOURCES at all is a stage-1 image
        with mock.patch.object(levern_route, 'SOURCES', ()):
            self.refused(self.mutated(), 'COLD')
        saved = levern_route.SOURCES
        try:
            del levern_route.SOURCES
            self.refused(self.mutated(), 'COLD')
        finally:
            levern_route.SOURCES = saved

    def test_a_merged_flag_that_would_do_nothing_is_refused(self):
        stage_one = dict(drop_env=('QWEN_PREFIX_REUSE', 'QWEN_FAST_STICKY_SESSIONS'))
        self.refused(self.mutated(**stage_one), 'QWEN_FAST_LEVERN_PARK=host needs the merged route')
        self.refused(self.mutated(**stage_one), 'QWEN_FAST_LEVERN_EPOCH_SCOPE=route needs the merged route')
        # the route epoch scope without the pre-stage it is checked against
        self.refused(self.mutated(drop_env=('QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE',)), 'EPOCH_SCOPE=route needs QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE=1')
        self.refused(self.mutated(drop_env=('QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS',)), 'EPOCH_SCOPE=route needs QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS=1')
        self.assertEqual(contract.levern_problems(self.mutated(env={'QWEN_FAST_LEVERN_EPOCH_SCOPE': 'global', 'QWEN_FAST_LEVERN_PARK': '0'},
                                                               drop_env=('QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE',))), [])

    def test_a_bad_merged_flag_value_is_refused_by_name(self):
        for name, value in (('QWEN_FAST_LEVERN_PARK', 'device'), ('QWEN_FAST_LEVERN_EPOCH_SCOPE', 'local'), ('QWEN_FAST_LEVERN_TTFT_TARGET_S', '-1'),
                            ('QWEN_FAST_LEVERN_SHORT_TOKENS', 'x'), ('QWEN_FAST_LEVERN_PARK_SLOTS', '0'), ('QWEN_FAST_LEVERN_MAX_PARK_S', '0')):
            with self.subTest(name=name):
                self.refused(self.mutated(env={name: value}), name)

    def test_the_stage_one_requirements_still_hold_for_the_merged_profile(self):
        # the merged route on a traffic profile is allowed (the short-window plan's stage 1); the stage-1 shape and the gate instruments are not
        self.assertEqual(contract.levern_problems(self.mutated(gate_only=False)), [])
        self.refused(self.mutated(gate_only=False, env={'QWEN_FAST_LEVERN_AUDIT': '1'}), 'gate-only')
        self.refused(self.mutated(gate_only=False, env={'QWEN_FAST_LEVERN_FAULT': 'foreign'}), 'gate-only')
        self.refused(self.mutated(gate_only=False, env={'QWEN_PREFIX_REUSE': '0', 'QWEN_FAST_STICKY_SESSIONS': '0', 'QWEN_FAST_LEVERN_PARK': '0',
                                                        'QWEN_FAST_LEVERN_EPOCH_SCOPE': 'global'}), 'gate-only')
        self.refused(self.mutated(env={'QWEN_FAST_KV_RESERVATION': '0'}), 'KV_RESERVATION')
        self.refused(self.mutated(env={'QWEN_FAST_ANY_REQUEST': '0'}), 'QWEN_FAST_ANY_REQUEST=1')
        self.refused(self.mutated(engine={'no-enable-chunked-prefill': True}), 'enable-chunked-prefill')
        self.refused(self.mutated(env={'QWEN_FAST_LANE': '1'}), 'QWEN_FAST_LANE')

    def test_the_merged_flags_without_the_master_switch_are_a_typo(self):
        profile = self.mutated(SHIP, env={'QWEN_FAST_LEVERN_PARK': 'host'})
        self.assertTrue(any('QWEN_FAST_LEVER_N=1' in problem for problem in contract.levern_problems(profile)))

    def test_the_audit_switch_needs_a_gate_only_profile_as_before(self):
        self.refused(self.mutated(AUDITED, gate_only=False), 'gate-only')


class EnvironmentTests(unittest.TestCase):
    def test_the_unmerged_profiles_launch_with_none_of_the_flags_whatever_the_process_inherited(self):
        inherited = {name: '1' for name in contract.LEVERN_ENV_FLAGS}
        for name in (SHIP, SHIP_AUDIT):
            environ = contract.apply_environment(named(name), dict(inherited, KEEP='me'))
            self.assertEqual(environ['KEEP'], 'me')
            self.assertEqual([key for key in environ if key in contract.LEVERN_ENV_FLAGS], [])

    def test_the_merged_profile_sets_exactly_its_flags(self):
        inherited = {name: '1' for name in contract.LEVERN_ENV_FLAGS}
        environ = contract.apply_environment(named(PLAIN), dict(inherited))
        self.assertEqual({key: environ[key] for key in contract.LEVERN_ENV_FLAGS if key in environ}, FLAGS)
        self.assertNotIn('QWEN_FAST_LEVERN_AUDIT', environ)
        self.assertNotIn('QWEN_FAST_LEVERN_FAULT', environ)
        audited = contract.apply_environment(named(AUDITED), dict(inherited))
        self.assertEqual(audited['QWEN_FAST_LEVERN_AUDIT'], '1')

    def test_the_production_environment_and_argv_are_the_merge_bases_with_the_flags_off(self):
        production = contract.apply_environment(named(SHIP), {})
        twin = contract.apply_environment(named(PLAIN), {})
        self.assertEqual({key: value for key, value in twin.items() if key not in FLAGS}, production)


class BootTests(unittest.TestCase):
    GATE = {contract.GATE_SWITCH: '1'}

    def test_the_merged_profile_boots_as_a_gate_and_arms_the_platform_hook(self):
        for name in (PLAIN, AUDITED):
            with self.subTest(name=name):
                profile, environ, logged, hooks, launched = boot_api_server(name, self.GATE)
                self.assertEqual(profile['name'], name)
                self.assertIn(contract.LEVERN_PLATFORM_MODULE, sorted(hook.name for hook in hooks))
                self.assertTrue(any('Lever N armed' in line for line in logged), logged)
                self.assertIn('--enable-chunked-prefill', launched)
                self.assertEqual(environ['QWEN_FAST_LEVER_N'], '1')
                self.assertEqual(environ['QWEN_PREFIX_REUSE'], '1')
                self.assertEqual(environ['QWEN_FAST_STICKY_SESSIONS'], '1')

    def test_without_the_gate_switch_it_does_not_boot(self):
        for name in (PLAIN, AUDITED):
            refused, *_ = boot_api_server(name)
            self.assertIsInstance(refused, ValueError)
            self.assertIn('is gate only', str(refused))

    def test_a_merged_profile_that_breaks_a_rule_does_not_boot(self):
        import tempfile

        broken = json.loads(PROFILES.read_text(encoding='utf-8'))
        broken['profiles'][PLAIN]['env'].pop('QWEN_FAST_STICKY_SESSIONS')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profiles.json'
            path.write_text(json.dumps(broken), encoding='utf-8')
            with mock.patch.object(sys.modules['test_qwen_prefix_image'], 'PROFILES', path):
                refused, environ, _, hooks, _ = boot_api_server(PLAIN, self.GATE)
        self.assertIsInstance(refused, ValueError)
        self.assertIn('QWEN_FAST_STICKY_SESSIONS', str(refused))
        self.assertNotIn('QWEN_FAST_LEVER_N', environ)
        self.assertEqual(hooks, [])


if __name__ == '__main__':
    unittest.main()
