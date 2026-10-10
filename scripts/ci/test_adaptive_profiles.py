"""The adaptive Lever N governor's gate-only profile (make_adaptive_profiles, docs/lever-n-adaptive-governor.md): the twin is the production profile plus exactly the
four adaptive names and nothing else, gate only and without the owner waiver, passing the Lever N contract; the checked-in file is what the generator makes; the
production profile and every other profile name no adaptive flag; the flags never reach a traffic profile or an inherited process environment; the card jobs of the
paired timing (references/tp4-prefill-jobs) name the twin and the production profile."""

import copy
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import levern_policy  # noqa: E402
import make_adaptive_profiles as generator  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES_TEXT = (HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8')
PROFILES = json.loads(PROFILES_TEXT)['profiles']
FLAGS = tuple(generator.ENV)


class GeneratorTests(unittest.TestCase):
    def test_the_checked_in_twin_is_what_the_generator_makes(self):
        self.assertEqual(generator.render(generator.generate(json.loads(PROFILES_TEXT))), PROFILES_TEXT)

    def test_the_generator_is_idempotent_and_leaves_every_other_profile_alone(self):
        data = json.loads(PROFILES_TEXT)
        once = generator.generate(data)
        self.assertEqual(generator.generate(once), once)
        twins = set(generator.twin_names())
        self.assertEqual([name for name in once['profiles'] if name not in twins], [name for name in data['profiles'] if name not in twins])
        for name, profile in data['profiles'].items():
            if name not in twins:
                self.assertEqual(once['profiles'][name], profile, name)

    def test_the_twin_closes_the_file_before_the_region_read_twins_and_the_other_generators_still_agree(self):
        import make_kvread_profiles
        import make_w2_kill_profiles

        names = list(PROFILES)
        kvread = list(make_kvread_profiles.twin_names())
        self.assertEqual(names[-len(kvread):], kvread)
        self.assertEqual(names[-len(kvread) - 1], generator.TWIN)
        for other in (make_kvread_profiles, make_w2_kill_profiles):
            self.assertEqual(other.render(other.generate(json.loads(PROFILES_TEXT))), PROFILES_TEXT, other.__name__)

    def test_the_twin_name_is_registered_with_the_census_exemptions(self):
        self.assertIn(generator.TWIN, profile_twins.twin_names())
        self.assertIn(generator.TWIN, PROFILES)

    def test_a_parent_that_cannot_be_twinned_is_refused_by_name(self):
        data = json.loads(PROFILES_TEXT)
        cases = (('gate only', lambda p: p.update(gate_only=True), 'is gate_only'),
                 ('already adaptive', lambda p: p['env'].update({'QWEN_FAST_LEVERN_ADAPTIVE': '1'}), 'already names QWEN_FAST_LEVERN_ADAPTIVE'),
                 ('no merged route', lambda p: p['env'].pop('QWEN_PREFIX_REUSE'), 'merged route'),
                 ('static rounds', lambda p: p['env'].update({'QWEN_FAST_LEVERN_ROUNDS': '2'}), 'static rounds'),
                 ('floor off', lambda p: p['env'].update({'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S': '0'}), 'floor off'))
        for label, edit, needle in cases:
            with self.subTest(label):
                broken = copy.deepcopy(data)
                edit(broken['profiles'][generator.PARENT])
                with self.assertRaises(ValueError) as caught:
                    generator.generate(broken)
                self.assertIn(needle, str(caught.exception))
        broken = copy.deepcopy(data)
        del broken['profiles'][generator.PARENT]
        with self.assertRaises(ValueError):
            generator.generate(broken)


class TwinTests(unittest.TestCase):
    def test_the_twin_is_the_production_profile_plus_exactly_the_four_names(self):
        mine, parent = copy.deepcopy(PROFILES[generator.TWIN]), copy.deepcopy(PROFILES[generator.PARENT])
        for name, value in generator.ENV.items():
            self.assertEqual(mine['env'].pop(name), value)
        self.assertIn(generator.PARENT, mine.pop('description'))
        parent.pop('description')
        self.assertIn('owner_traffic_waiver', parent)
        parent.pop('owner_traffic_waiver')
        self.assertIs(parent.pop('parser_rechunk'), True)
        self.assertIs(mine.pop('parser_rechunk'), True)
        self.assertIs(mine.pop('gate_only'), True)
        self.assertNotIn('gate_only', parent)
        self.assertEqual(mine, parent, 'everything but the four names, the waiver, gate_only and the description is the parent\'s')

    def test_the_flags_are_the_policys_defaults_written_out_and_the_parent_names_none(self):
        parsed = levern_policy.adaptive_config({key: str(value) for key, value in PROFILES[generator.TWIN]['env'].items()})
        self.assertEqual(parsed, levern_policy.Adaptive(2, 0.9, 120))
        self.assertEqual(parsed, levern_policy.adaptive_config({'QWEN_FAST_LEVER_N': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_LEVERN_ADAPTIVE': '1'}))
        self.assertFalse([flag for flag in FLAGS if flag in PROFILES[generator.PARENT]['env']])

    def test_the_description_says_what_the_arm_is(self):
        text = PROFILES[generator.TWIN]['description']
        self.assertTrue(text.startswith('GATE ONLY, UNQUALIFIED'))
        for word in (generator.PARENT, 'QWEN_FAST_LEVERN_ADAPTIVE=1', '0.9', '120 s', 'step for step', 'decode-gap floor', 'never hand-merge'):
            self.assertIn(word, text)

    def test_the_twin_passes_the_lever_n_contract_and_its_flags_parse(self):
        profile = dict(PROFILES[generator.TWIN], name=generator.TWIN)
        self.assertEqual(contract.levern_problems(profile), [])
        env = {key: str(value) for key, value in profile['env'].items()}
        self.assertEqual(levern_policy.config_problems(env), [])
        merged = levern_policy.merged_config(env)
        self.assertEqual((merged.ttft_s, merged.max_gap_s), (180, 8), 'the floor the policy requires is on, and the profile\'s own target is the one the policy lowers')

    def test_no_other_profile_names_a_flag_and_the_production_profile_is_untouched(self):
        for name, profile in PROFILES.items():
            if name != generator.TWIN:
                self.assertFalse([flag for flag in FLAGS if flag in profile['env']], name)
        self.assertIn('owner_traffic_waiver', PROFILES[generator.PARENT])
        self.assertNotIn('gate_only', PROFILES[generator.PARENT])

    def test_every_adaptive_flag_is_a_contract_flag(self):
        self.assertEqual(sorted(contract.LEVERN_ADAPTIVE_FLAGS), sorted(levern_policy.ADAPTIVE_FLAGS))
        for name in levern_policy.ADAPTIVE_FLAGS:
            self.assertIn(name, contract.LEVERN_ENV_FLAGS)


class RefusalTests(unittest.TestCase):
    def refused(self, profile, needle):
        problems = contract.levern_problems(profile)
        self.assertTrue(any(needle in problem for problem in problems), (needle, problems))

    def test_a_traffic_profile_cannot_carry_any_adaptive_flag(self):
        for name in FLAGS:
            with self.subTest(name=name):
                broken = copy.deepcopy(PROFILES[generator.PARENT])
                broken['env'][name] = generator.ENV[name]
                self.refused(dict(broken, name='x'), 'gate instrument')
        traffic = dict(copy.deepcopy(PROFILES[generator.TWIN]), name='x', gate_only=False)
        self.refused(traffic, 'gate instrument')

    def test_a_policy_that_cannot_work_beside_its_profile_is_refused(self):
        cases = (({'QWEN_FAST_LEVERN_ROUNDS': '1'}, 'static rounds'), ({'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S': '0'}, 'REQUIRES the decode-gap floor'),
                 ({'QWEN_FAST_LEVERN_ADAPTIVE_SHARE': '1'}, 'QWEN_FAST_LEVERN_ADAPTIVE_SHARE'))
        for env, needle in cases:
            with self.subTest(needle=needle):
                broken = copy.deepcopy(PROFILES[generator.TWIN])
                broken['env'].update(env)
                self.refused(dict(broken, name='x'), needle)

    def test_a_knob_without_the_switch_is_a_typo_the_contract_names(self):
        broken = copy.deepcopy(PROFILES[generator.TWIN])
        broken['env'].pop('QWEN_FAST_LEVERN_ADAPTIVE')
        self.refused(dict(broken, name='x'), 'QWEN_FAST_LEVERN_ADAPTIVE=1')

    def test_an_inherited_process_environment_never_reaches_a_profile_that_does_not_name_the_flags(self):
        inherited = {name: generator.ENV[name] for name in FLAGS}
        inherited['KEEP'] = 'me'
        environ = contract.apply_environment(dict(PROFILES[generator.PARENT], name=generator.PARENT), dict(inherited))
        self.assertEqual(environ['KEEP'], 'me')
        self.assertFalse([name for name in FLAGS if name in environ])
        environ = contract.apply_environment(dict(PROFILES[generator.TWIN], name=generator.TWIN), dict(inherited))
        self.assertEqual({name: environ[name] for name in FLAGS}, generator.ENV)


if __name__ == '__main__':
    unittest.main()
