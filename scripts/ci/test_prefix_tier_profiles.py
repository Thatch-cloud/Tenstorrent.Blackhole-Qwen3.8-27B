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
import make_prefix_tier_profiles as generator  # noqa: E402
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

    def test_the_twin_sits_after_the_kill_drill_twin_which_stays_right_after_the_parent(self):
        names = list(PROFILES)
        self.assertEqual(names.index(kill_generator.TWIN), names.index(generator.PARENT) + 1)
        self.assertEqual(names.index(generator.TWIN), names.index(kill_generator.TWIN) + 1)
        self.assertNotEqual(names[-1], generator.TWIN, 'the end of the file is the region-read twins\'')

    def test_the_twin_is_registered_with_the_census_exemptions(self):
        self.assertIn(generator.TWIN, profile_twins.twin_names())
        self.assertEqual(generator.twin_names(), (generator.TWIN,))

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
            if name != generator.TWIN:
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
                registry.ENV_TIER_VERIFY, registry.ENV_TIER_OFF_PATH, 'QWEN_PREFIX_CKPT_PRECONVERTED', 'QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT'}
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
