"""The host KV tier's audited gate twin (make_prefix_tier_profiles, docs/prefix-store-hygiene.md): it is the production profile plus exactly the tier's and the preconverted
checkpoints' names, gate only, passing the prefix contract; the checked-in file is what the generator makes; no other profile names the features; an inherited process
environment never turns either on under a profile that does not name it; and the job pack's tier arms run on it.

Run at py 3.11: `py -3.11 -B -m unittest test_prefix_tier_profiles` from scripts/ci."""

import copy
import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job  # noqa: E402
import make_fusion_profiles  # noqa: E402
import make_kvread_profiles  # noqa: E402
import make_octo2_profiles  # noqa: E402
import make_octo_profiles  # noqa: E402
import make_parked_profiles  # noqa: E402
import make_prefix_tier_profiles as generator  # noqa: E402
import make_round_host_profiles  # noqa: E402
import make_ship4_profiles as ship4  # noqa: E402
import make_np1_profiles as np1  # noqa: E402
import make_s5_profiles as s5  # noqa: E402
import make_probe_profiles as probe_twins  # noqa: E402
import make_xe_profiles as xe  # noqa: E402
import make_pf2_profiles as pf2  # noqa: E402
import make_sched_profiles as sched  # noqa: E402
import make_upload_profiles  # noqa: E402
import make_w2_kill_profiles as kill_generator  # noqa: E402
import prefix_replay as replay  # noqa: E402
import profile_twins  # noqa: E402
import qwen_prefix_registry as registry  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES_TEXT = (HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8')
DATA = json.loads(PROFILES_TEXT)
PROFILES = DATA['profiles']
TWIN = PROFILES[generator.TWIN]
PARENT = PROFILES[generator.PARENT]
NAMES = contract.PREFIX_TIER_ENV_NAMES


class GeneratorTests(unittest.TestCase):
    def test_the_checked_in_twin_is_what_the_generator_makes(self):
        self.assertEqual(generator.render(generator.generate(json.loads(PROFILES_TEXT))), PROFILES_TEXT)

    def test_the_generator_is_idempotent_and_leaves_every_other_profile_alone(self):
        once = generator.generate(copy.deepcopy(DATA))
        self.assertEqual(generator.generate(once), once)
        others = [name for name in DATA['profiles'] if name != generator.TWIN]
        self.assertEqual([name for name in once['profiles'] if name != generator.TWIN], others)
        for name in others:
            self.assertEqual(once['profiles'][name], DATA['profiles'][name], name)

    def test_the_profiles_sit_after_the_kill_drill_twin_and_the_upload_family_which_stay_where_their_generators_put_them(self):
        names = list(PROFILES)
        self.assertEqual(names.index(kill_generator.TWIN), names.index(generator.PARENT) + 1)
        upload = [name for name in names if name.startswith(generator.PARENT + '-')]
        self.assertTrue(upload, 'the engine-start upload family')
        self.assertEqual(names.index(upload[0]), names.index(kill_generator.TWIN) + 1, 'make_upload_profiles inserts right after the kill twin')
        self.assertEqual(names[names.index(upload[-1]) + 1:names.index(upload[-1]) + 4], [generator.TWIN, generator.SHIP, generator.SHIP_AUDIT])
        self.assertEqual(names[-len(make_kvread_profiles.twin_names()):], list(make_kvread_profiles.twin_names()), 'the end of the file is the region-read twins\'')
        fusion = [name for name in names if name.startswith(make_fusion_profiles.NAMESPACE)]
        self.assertGreater(names.index(fusion[0]), names.index(generator.SHIP_AUDIT), 'the op-fusion block is generated after this one')

    def test_every_generator_agrees_with_the_one_file(self):
        """The order above is the one every generator\'s own --check reproduces: this file, and the ones it neighbours, are stable under each of them."""
        for module in (generator, kill_generator, make_kvread_profiles, make_parked_profiles, make_octo_profiles, make_octo2_profiles, make_round_host_profiles,
                       make_upload_profiles, make_fusion_profiles):
            with self.subTest(generator=module.__name__):
                self.assertEqual(module.main(['--check']), 0)

    def test_the_profiles_are_registered_with_the_census_exemptions(self):
        self.assertEqual(generator.twin_names(), (generator.TWIN, generator.SHIP_AUDIT))
        self.assertEqual(generator.ship_names(), (generator.SHIP,))
        for name in generator.twin_names():
            self.assertIn(name, profile_twins.twin_names())
        self.assertNotIn(generator.SHIP, profile_twins.twin_names(), 'a traffic profile is not a gate-only twin')
        self.assertIn(generator.SHIP, profile_twins.generated_names())

    def test_a_parent_that_cannot_be_twinned_is_refused_by_name(self):
        cases = (('gate only', lambda p: p.update(gate_only=True), 'is gate_only'),
                 ('no prefix reuse', lambda p: p['env'].pop('QWEN_PREFIX_REUSE'), 'does not run prefix reuse'),
                 ('already tiered', lambda p: p['env'].update({'QWEN_PREFIX_HOST_TIER_GIB': '8'}), 'already names QWEN_PREFIX_HOST_TIER_GIB'),
                 ('store too big', lambda p: p['env'].update({'QWEN_PREFIX_STORE_GIB': '40'}), 'not under the tier'))
        for label, edit, needle in cases:
            with self.subTest(label):
                broken = copy.deepcopy(DATA)
                edit(broken['profiles'][generator.PARENT])
                with self.assertRaises(ValueError) as caught:
                    generator.generate(broken)
                self.assertIn(needle, str(caught.exception))
        broken = copy.deepcopy(DATA)
        del broken['profiles'][generator.PARENT]
        with self.assertRaises(ValueError):
            generator.generate(broken)


class TwinTests(unittest.TestCase):
    def test_the_twin_is_its_parent_plus_exactly_the_tier_names(self):
        mine, theirs = copy.deepcopy(TWIN), copy.deepcopy(PARENT)
        for name, value in generator.ENV.items():
            self.assertEqual(mine['env'].pop(name), value, name)
        self.assertIn(generator.PARENT, mine.pop('description'))
        theirs.pop('description')
        theirs.pop(generator.WAIVER_FIELD)
        mine.pop('gate_only')
        self.assertEqual(mine, theirs, 'everything but the names, the description, the waiver and gate_only is the parent\'s')
        self.assertIs(TWIN['gate_only'], True)
        self.assertNotIn(generator.WAIVER_FIELD, TWIN)

    def test_the_parent_names_none_of_the_features(self):
        self.assertFalse([name for name in NAMES if name in PARENT['env']])
        self.assertNotIn('gate_only', PARENT)

    def test_the_owner_defaults_are_the_numbers_and_the_store_keeps_its_share(self):
        env = TWIN['env']
        self.assertEqual(env['QWEN_PREFIX_HOST_TIER_GIB'], '32')
        self.assertLess(float(env['QWEN_PREFIX_STORE_GIB']), 32.0)
        config = registry.tier_config(dict((key, str(value)) for key, value in env.items()), int(float(env['QWEN_PREFIX_STORE_GIB']) * (1 << 30)))
        self.assertEqual(config.total, 32 << 30)
        self.assertEqual(config.kv_bytes, (32 - int(float(env['QWEN_PREFIX_STORE_GIB']))) << 30)
        self.assertTrue(config.audit)
        self.assertEqual(config.verify, 'all')

    def test_every_instrument_is_on_and_the_kill_switch_file_is_the_gates(self):
        env = TWIN['env']
        for name in ('QWEN_PREFIX_HOST_TIER_AUDIT', 'QWEN_PREFIX_CKPT_PRECONVERTED', 'QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT'):
            self.assertEqual(env[name], '1', name)
        self.assertEqual(env['QWEN_PREFIX_HOST_TIER_VERIFY'], 'all')
        self.assertEqual(env['QWEN_PREFIX_HOST_TIER_OFF_PATH'], replay.TIER_DRILL_OFF_PATH, 'the file the scenario writes is the file the engine polls')
        self.assertTrue(env['QWEN_PREFIX_HOST_TIER_OFF_PATH'].startswith('/tmp/'), 'a scratch path inside the container, never the hub mount')

    def test_the_description_says_what_it_is_and_that_it_is_unqualified(self):
        text = TWIN['description']
        self.assertTrue(text.startswith('GATE ONLY, UNQUALIFIED'))
        for phrase in (generator.PARENT, 'QWEN_PREFIX_HOST_TIER_GIB=32', 'version 2', 'make_prefix_tier_profiles.py', 'regenerate, never hand-merge'):
            self.assertIn(phrase, text)

    def test_the_twin_passes_the_prefix_contract(self):
        twin = dict(copy.deepcopy(TWIN), name=generator.TWIN)
        self.assertEqual(contract.prefix_policy_problems(twin), [])
        self.assertEqual(contract.prefix_reuse_problems(twin), [])

    def test_the_w2_parent_gets_the_sdpa_audit_the_gates_extent_audit_needs_and_nothing_else_of_w2(self):
        import c2_serving_gate
        self.assertEqual(PARENT['env']['QWEN_FAST_TP4_SDPA'], 'multi')
        self.assertNotIn('QWEN_FAST_TP4_SDPA_AUDIT', PARENT['env'])
        self.assertEqual(TWIN['env']['QWEN_FAST_TP4_SDPA_AUDIT'], '1')
        self.assertIsNone(c2_serving_gate.multi_audit_refusal(DATA, generator.TWIN), 'an audited arm adds the extent audit; the twin carries the SDPA audit it needs')
        self.assertIsNotNone(c2_serving_gate.multi_audit_refusal(DATA, generator.PARENT), 'the parent without it would be refused: the reason for the twin\'s one extra name')
        self.assertEqual(sorted(name for name in TWIN['env'] if name not in PARENT['env']), sorted(generator.ENV))

    def test_a_parent_without_the_multi_launch_is_refused(self):
        broken = copy.deepcopy(DATA)
        broken['profiles'][generator.PARENT]['env']['QWEN_FAST_TP4_SDPA'] = 'off'
        with self.assertRaises(ValueError) as caught:
            generator.generate(broken)
        self.assertIn('multi-user SDPA launch', str(caught.exception))

    def test_the_gate_derives_its_timed_arm_from_it_with_the_instruments_off(self):
        profiles = dict(DATA)
        name, derived = gate.derive(profiles, generator.TWIN, 'tiertime')
        env = derived['profiles'][name]['env']
        self.assertEqual(env['QWEN_PREFIX_HOST_TIER_GIB'], '32')
        for flag in ('QWEN_PREFIX_HOST_TIER_AUDIT', 'QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT'):
            self.assertEqual(env[flag], '0')
        self.assertEqual(env['QWEN_PREFIX_HOST_TIER_VERIFY'], 'sample')
        self.assertEqual(env['QWEN_FAST_TP4_SDPA_AUDIT'], '0', 'a timed arm carries no extent audit, so it needs no SDPA audit and does not pay for it')
        self.assertEqual(env['QWEN_PREFIX_CKPT_PRECONVERTED'], '1', 'the timed arm still restores preconverted checkpoints')
        self.assertEqual(contract.prefix_policy_problems(dict(derived['profiles'][name], name=name)), [])


class ShipCandidateTests(unittest.TestCase):
    """-tierfx-traffic (the op-fusion ship profile plus the tier, under its own PENDING owner traffic waiver) and -tierfx-audit (its gate-only audit twin)."""

    SHIP = PROFILES[generator.SHIP]
    AUDIT = PROFILES[generator.SHIP_AUDIT]
    FX = PROFILES[generator.FX_PARENT]
    FX_AUDIT = PROFILES[generator.FX_AUDIT_PARENT]

    def named(self, profile):
        return dict(copy.deepcopy(profile), name='probe')

    def test_the_ship_candidate_is_the_fusion_ship_profile_plus_exactly_the_tier_env(self):
        mine, theirs = copy.deepcopy(self.SHIP), copy.deepcopy(self.FX)
        for name, value in generator.SHIP_ENV.items():
            self.assertEqual(mine['env'].pop(name), value, name)
        self.assertIn(generator.FX_PARENT, mine.pop('description'))
        theirs.pop('description')
        waiver, parent_waiver = mine.pop(generator.WAIVER_FIELD), theirs.pop(generator.WAIVER_FIELD)
        self.assertEqual(mine, theirs, 'everything but the tier names, the description and the waiver is the fusion ship profile\'s')
        self.assertNotIn('gate_only', self.SHIP, 'a traffic profile')
        self.assertEqual(generator.SHIP_ENV['QWEN_PREFIX_HOST_TIER_VERIFY'], 'sample')
        self.assertEqual((generator.SHIP_ENV['QWEN_PREFIX_HOST_TIER_AUDIT'], generator.SHIP_ENV['QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT']), ('0', '0'))
        for gate_instrument in ('QWEN_PREFIX_HOST_TIER_OFF_PATH', 'QWEN_FAST_TP4_SDPA_AUDIT'):
            self.assertNotIn(gate_instrument, self.SHIP['env'])
        self.assertEqual(waiver['levers'], dict(parent_waiver['levers'], **contract.TIER_WAIVABLE_LEVERS))
        self.assertNotEqual(waiver['id'], parent_waiver['id'])

    def test_its_waiver_is_well_formed_names_the_tier_and_is_pending(self):
        waiver = self.SHIP[contract.TRAFFIC_WAIVER]
        profile = self.named(self.SHIP)
        self.assertEqual(contract.traffic_waiver_problems(profile), [])
        self.assertEqual(sorted(waiver), sorted(contract.TRAFFIC_WAIVER_FIELDS))
        self.assertTrue(waiver['decision'].startswith('PENDING'), waiver['decision'])
        self.assertTrue(contract.traffic_waiver_pending(profile))
        self.assertEqual(contract.waived_levers(profile), waiver['levers'])
        self.assertIn('QWEN_PREFIX_HOST_TIER_GIB', contract.waived_levers(profile))
        self.assertEqual(waiver['decision'], generator.SHIP_DECISION)
        self.assertGreaterEqual(len(waiver['evidence']), 5)
        for run in ('38082194508', '38082317009', '38082865675', '38083229148', '38084230415'):
            self.assertTrue(any(run in item for item in waiver['evidence']), run)
        text = json.dumps([self.SHIP['description'], waiver])
        for word in make_fusion_profiles.SHIP_FORBIDDEN:
            self.assertNotIn(word, text)

    def test_a_pending_ship_candidate_boots_but_no_build_bakes_it(self):
        self.assertTrue(c2_serving_job.bake_waiver_pending(self.SHIP))
        with self.assertRaises(c2_serving_job.JobError) as caught:
            c2_serving_job.read_bake({'C2_BAKE_DEFAULT_PROFILE': generator.SHIP}, 'build', 'quad')
        self.assertIn('not APPROVED', str(caught.exception))

    def test_the_contract_accepts_the_tier_on_this_traffic_profile_and_only_under_the_waiver(self):
        profile = self.named(self.SHIP)
        self.assertEqual(contract.prefix_policy_problems(profile), [])
        self.assertEqual(contract.prefix_reuse_problems(profile), [])
        # the waiver is what lets it: without the tier's levers named, the same env is refused
        for levers in (dict((k, v) for k, v in self.SHIP[contract.TRAFFIC_WAIVER]['levers'].items() if k != 'QWEN_PREFIX_HOST_TIER_GIB'),
                       dict((k, v) for k, v in self.SHIP[contract.TRAFFIC_WAIVER]['levers'].items() if k != 'QWEN_PREFIX_CKPT_PRECONVERTED')):
            broken = self.named(self.SHIP)
            broken[contract.TRAFFIC_WAIVER]['levers'] = levers
            problems = contract.prefix_policy_problems(broken)
            self.assertTrue(any('only a gate-only profile may name them' in problem for problem in problems), problems)
        stale = self.named(self.SHIP)
        stale['env'].pop('QWEN_PREFIX_CKPT_PRECONVERTED')
        self.assertTrue(contract.traffic_waiver_problems(stale), 'a lever the waiver names and the profile does not set is a stale waiver')
        bare = self.named(self.FX)
        bare['env'].update(generator.SHIP_ENV)
        self.assertTrue(any('only a gate-only profile may name them' in problem for problem in contract.prefix_policy_problems(bare)), 'the same env with the fusion waiver is refused')

    def test_the_waiver_never_covers_the_audits_the_kill_path_or_another_value(self):
        for name, value in (('QWEN_PREFIX_HOST_TIER_AUDIT', '1'), ('QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT', '1'), ('QWEN_PREFIX_HOST_TIER_OFF_PATH', '/tmp/off')):
            probe = self.named(self.SHIP)
            probe['env'][name] = value
            problems = contract.prefix_policy_problems(probe)
            self.assertTrue(any(name in problem and 'gate-only' in problem for problem in problems), (name, problems))
        wrong = self.named(self.SHIP)
        wrong['env']['QWEN_PREFIX_HOST_TIER_GIB'] = '16'
        self.assertTrue(any('only a gate-only profile may name them' in problem or 'waiver' in problem for problem in contract.prefix_policy_problems(wrong) +
                            contract.traffic_waiver_problems(wrong)), 'a different size is not what the waiver names')
        flagged = self.named(self.SHIP)
        flagged[contract.TRAFFIC_WAIVER]['levers']['QWEN_PREFIX_HOST_TIER_AUDIT'] = '1'
        self.assertTrue(any('is not waivable' in problem for problem in contract.traffic_waiver_problems(flagged)))
        other = self.named(self.SHIP)
        other[contract.TRAFFIC_WAIVER]['levers']['QWEN_PREFIX_HOST_TIER_GIB'] = '16'
        self.assertTrue(any('waivable only at 32' in problem for problem in contract.traffic_waiver_problems(other)))

    def test_the_audit_twin_is_the_fusion_audit_arm_plus_the_tier_instruments_and_gate_only(self):
        mine, theirs = copy.deepcopy(self.AUDIT), copy.deepcopy(self.FX_AUDIT)
        for name, value in generator.ENV.items():
            self.assertEqual(mine['env'].pop(name), value, name)
        self.assertIn(generator.FX_AUDIT_PARENT, mine.pop('description'))
        theirs.pop('description')
        self.assertEqual(mine, theirs)
        self.assertIs(self.AUDIT['gate_only'], True)
        self.assertNotIn(contract.TRAFFIC_WAIVER, self.AUDIT)
        self.assertEqual(contract.prefix_policy_problems(self.named(self.AUDIT)), [])
        self.assertEqual(self.AUDIT['env']['QWEN_FAST_TP4_SDPA_AUDIT'], '1', 'the extent audit the gate adds needs the multi launch\'s own audit beside W2')
        self.assertTrue(self.AUDIT['description'].startswith('GATE ONLY'))

    def test_the_audit_twin_carries_every_fusion_lever_of_the_ship_candidate(self):
        ship_levers = {k: v for k, v in self.SHIP['env'].items() if k not in generator.SHIP_ENV}
        for key, value in ship_levers.items():
            self.assertEqual(self.AUDIT['env'].get(key), value, key)

    def test_both_gates_see_the_candidate_as_a_tier_profile(self):
        document = {'profiles': PROFILES}
        arms = gate.plan_arms('tier-concurrent-timed', generator.SHIP, 'none', document)
        self.assertEqual([arm['arm'] for arm in arms], ['tier-concurrent-timed'])
        self.assertTrue(arms[0]['tier'] and arms[0]['tier']['preconverted'] and not arms[0]['tier']['audit'])
        audited = gate.plan_arms('tier-concurrent', generator.SHIP_AUDIT, 'none', document)
        self.assertTrue(audited[0]['tier']['audit'] and audited[0]['tier']['preconverted_audit'])
        control = gate.plan_arms('tier-concurrent-timed', generator.FX_PARENT, 'none', document)
        self.assertIsNone(control[0]['tier'], 'the fusion ship profile is the control: no tier')


class ContractTests(unittest.TestCase):
    def traffic(self, **env):
        profile = copy.deepcopy(PARENT)
        profile['env'].update(env)
        return dict(profile, name=generator.PARENT)

    def test_a_traffic_profile_cannot_name_either_feature(self):
        for name, value in (('QWEN_PREFIX_HOST_TIER_GIB', '32'), ('QWEN_PREFIX_CKPT_PRECONVERTED', '1'), ('QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT', '1'),
                            ('QWEN_PREFIX_HOST_TIER_AUDIT', '1'), ('QWEN_PREFIX_HOST_TIER_OFF_PATH', '/tmp/off')):
            with self.subTest(name=name):
                problems = contract.prefix_policy_problems(self.traffic(**{name: value}))
                self.assertTrue(any(name in problem and 'gate-only' in problem for problem in problems), problems)

    def test_the_same_names_are_fine_on_the_gate_only_twin(self):
        self.assertEqual(contract.prefix_policy_problems(dict(copy.deepcopy(TWIN), name=generator.TWIN)), [])

    def test_a_tier_that_leaves_nothing_for_pages_is_refused(self):
        twin = dict(copy.deepcopy(TWIN), name=generator.TWIN)
        twin['env']['QWEN_PREFIX_HOST_TIER_GIB'] = '8'
        self.assertTrue(any('must exceed QWEN_PREFIX_STORE_GIB' in problem for problem in contract.prefix_policy_problems(twin)))

    def test_no_other_profile_names_either_feature(self):
        for name, profile in PROFILES.items():
            if name not in generator.NAMES and name not in ship4.owned_names() and name not in np1.twin_names() and name not in s5.owned_names() and name not in probe_twins.twin_names() and name not in xe.twin_names() and name not in pf2.twin_names() and name not in sched.twin_names():       # (tp4/ship-4's combined candidate and its audited twin carry the tier: test_ship4_profiles; so do tp4/np1-fastlane's copies of them; so do tp4/exact-end's copies of them; so do tp4/prefill-2's twins of them; so do tp4/sched-1's copies of them)
                self.assertFalse([flag for flag in NAMES if flag in profile['env']], name)

    def test_an_inherited_process_environment_never_turns_either_on_under_a_profile_that_does_not_name_it(self):
        inherited = dict((name, '1' if name.endswith(('PRECONVERTED', 'AUDIT')) else '32') for name in NAMES)
        inherited['KEEP'] = 'me'
        environ = contract.apply_environment(dict(copy.deepcopy(PARENT), name=generator.PARENT), dict(inherited))
        self.assertEqual(environ['KEEP'], 'me')
        self.assertFalse([name for name in NAMES if name in environ], environ)
        environ = contract.apply_environment(dict(copy.deepcopy(TWIN), name=generator.TWIN), dict(inherited, QWEN_PREFIX_HOST_TIER_GIB='999'))
        self.assertEqual(environ['QWEN_PREFIX_HOST_TIER_GIB'], '32', 'the profile wins over the process')
        self.assertEqual(sorted(name for name in NAMES if name in environ), sorted(name for name in NAMES if name in TWIN['env']))

    def test_every_name_the_registry_and_the_model_read_is_one_the_contract_knows(self):
        read = {registry.ENV_TIER_GIB, registry.ENV_TIER_AUDIT, registry.ENV_TIER_SPILL_MAX, registry.ENV_TIER_MIN_TOKENS, registry.ENV_TIER_MIN_AVAILABLE,
                registry.ENV_TIER_VERIFY, registry.ENV_TIER_OFF_PATH, 'QWEN_PREFIX_CKPT_PRECONVERTED', 'QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT',
                # (the complete-spill mode, tp4/sched-1: test_sched_spill)
                registry.ENV_TIER_SPILL_MODE, registry.ENV_TIER_SPILL_HARD_MAX, registry.ENV_TIER_SPILL_BATCH, registry.ENV_TIER_SPILL_WARN_MS}
        self.assertEqual(read, set(NAMES))


class HygieneTests(unittest.TestCase):
    def test_the_generator_names_no_host_address_or_home_path(self):
        pattern = re.compile(r'(/home/|/Users/|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.local\b|zot\.|sha256:[0-9a-f]{16}|ghp_|token=)')
        for path in (HERE / 'make_prefix_tier_profiles.py', HERE / 'pin_kvread.py'):
            with self.subTest(path=path.name):
                self.assertIsNone(pattern.search(path.read_text(encoding='utf-8')))

    def test_the_generator_is_a_host_only_tool_the_image_does_not_carry(self):
        for name in ('make_prefix_tier_profiles.py', 'pin_kvread.py'):
            for path in (HERE.parent.parent / 'docker' / 'qwen-c2-serving.Dockerfile', HERE / 'build-c2-serving-image.sh'):
                self.assertNotIn(name, path.read_text(encoding='utf-8'), (name, path.name))


if __name__ == '__main__':
    unittest.main()
