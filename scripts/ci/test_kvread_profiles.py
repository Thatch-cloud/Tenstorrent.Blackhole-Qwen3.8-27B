"""The region-read audit twins (make_kvread_profiles, docs/prefix-audit-cost.md): each is its audited Lever N parent plus exactly QWEN_FAST_LEVERN_KV_READ and
nothing else, gate only, passing the Lever N contract; the checked-in file is what the generator makes; no other profile names the knob; the knob never reaches a
traffic profile or an inherited process environment."""

import copy
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import levern_policy  # noqa: E402
import make_kvread_profiles as generator  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES_TEXT = (HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8')
PROFILES = json.loads(PROFILES_TEXT)['profiles']
KNOBS = (generator.READ, generator.CROSS_STEPS)


class GeneratorTests(unittest.TestCase):
    def test_the_checked_in_twins_are_what_the_generator_makes(self):
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

    def test_the_twins_are_appended_at_the_end_in_spec_order(self):
        names = list(PROFILES)
        self.assertEqual(names[-len(generator.twin_names()):], list(generator.twin_names()))

    def test_the_twin_names_are_registered_with_the_census_exemptions(self):
        for name in generator.twin_names():
            self.assertIn(name, profile_twins.twin_names())
            self.assertIn(name, PROFILES)

    def test_no_name_starts_where_another_generator_anchors_its_block(self):
        # make_octo_profiles inserts its twins after the last name that starts with '<levern>-parked', make_parked_profiles regenerates every such name
        prefix = 'c2-packed-tp4-8x262k-ship-prefix-levern-parked'
        for name in generator.twin_names():
            self.assertFalse(name.startswith(prefix), name)

    def test_a_parent_that_cannot_be_twinned_is_refused_by_name(self):
        data = json.loads(PROFILES_TEXT)
        parent = generator.TWINS[0][0]
        cases = (('not gate_only', lambda p: p.pop('gate_only'), 'not gate_only'),
                 ('no audit', lambda p: p['env'].pop(generator.AUDIT), 'does not carry QWEN_FAST_LEVERN_AUDIT=1'),
                 ('already read', lambda p: p['env'].update({generator.READ: 'cross'}), 'already names QWEN_FAST_LEVERN_KV_READ'))
        for label, edit, needle in cases:
            with self.subTest(label):
                broken = copy.deepcopy(data)
                edit(broken['profiles'][parent])
                with self.assertRaises(ValueError) as caught:
                    generator.generate(broken)
                self.assertIn(needle, str(caught.exception))
        with self.assertRaises(ValueError):
            broken = copy.deepcopy(data)
            del broken['profiles'][parent]
            generator.generate(broken)


class TwinTests(unittest.TestCase):
    def test_each_twin_is_its_parent_plus_exactly_the_knob(self):
        for twin, parent, mode in [(twin, parent, mode) for parent, twin, mode in generator.TWINS]:
            with self.subTest(twin=twin):
                mine, theirs = copy.deepcopy(PROFILES[twin]), copy.deepcopy(PROFILES[parent])
                self.assertEqual(mine['env'].pop(generator.READ), mode)
                self.assertIn(parent, mine.pop('description'))
                theirs.pop('description')
                self.assertEqual(mine, theirs, 'everything but the knob and the description is the parent\'s')
                self.assertIs(PROFILES[twin]['gate_only'], True)

    def test_the_cross_twin_is_the_only_one_that_cross_checks_and_every_other_reads_the_region(self):
        modes = {twin: PROFILES[twin]['env'][generator.READ] for twin in generator.twin_names()}
        self.assertEqual(sorted(name for name, mode in modes.items() if mode == 'cross'), [name for name in modes if name.endswith('-kvx')])
        self.assertEqual(sorted(name for name, mode in modes.items() if mode == 'region'), sorted(name for name in modes if not name.endswith('-kvx')))
        for twin in generator.twin_names():
            self.assertNotIn(generator.CROSS_STEPS, PROFILES[twin]['env'], 'one cross-checked digest is the default and enough')

    def test_the_description_says_what_the_arm_is_and_that_it_is_unqualified(self):
        for twin in generator.twin_names():
            text = PROFILES[twin]['description']
            self.assertTrue(text.startswith('GATE ONLY, UNQUALIFIED'), twin)
            self.assertIn(generator.READ, text)
            if PROFILES[twin]['env'][generator.READ] == 'cross':
                self.assertIn('QUALIFICATION', text)
                self.assertIn('mismatched=', text)
            else:
                self.assertIn('falls back to the whole-cache read', text)

    def test_every_twin_passes_the_lever_n_contract_and_its_flags_parse(self):
        for twin in generator.twin_names():
            with self.subTest(twin=twin):
                self.assertEqual(contract.levern_problems(dict(PROFILES[twin], name=twin)), [])
                env = {key: str(value) for key, value in PROFILES[twin]['env'].items()}
                self.assertEqual(levern_policy.config_problems(env), [])
                self.assertEqual(levern_policy.kv_read_mode(env), PROFILES[twin]['env'][generator.READ])

    def test_no_other_profile_names_the_knob_and_the_parents_are_unchanged(self):
        twins = set(generator.twin_names())
        for name, profile in PROFILES.items():
            if name not in twins:
                self.assertFalse([knob for knob in KNOBS if knob in profile['env']], name)
        for parent in {parent for parent, twin, mode in generator.TWINS}:
            self.assertEqual(levern_policy.kv_read_mode(PROFILES[parent]['env']), 'full', parent)

    def test_a_traffic_profile_cannot_carry_the_knob(self):
        for twin in generator.twin_names():
            broken = dict(copy.deepcopy(PROFILES[twin]), name=twin, gate_only=False)
            problems = contract.levern_problems(broken)
            self.assertTrue(any('QWEN_FAST_LEVERN' in problem and 'gate' in problem for problem in problems), (twin, problems))

    def test_an_inherited_process_environment_never_reaches_a_profile_that_does_not_name_the_knob(self):
        inherited = {generator.READ: 'cross', generator.CROSS_STEPS: '5', 'KEEP': 'me'}
        parent = generator.TWINS[0][0]
        environ = contract.apply_environment(dict(PROFILES[parent], name=parent), dict(inherited))
        self.assertEqual(environ['KEEP'], 'me')
        self.assertFalse([knob for knob in KNOBS if knob in environ])
        twin = generator.TWINS[0][1]
        environ = contract.apply_environment(dict(PROFILES[twin], name=twin), dict(inherited))
        self.assertEqual(environ[generator.READ], 'region')
        self.assertNotIn(generator.CROSS_STEPS, environ)


if __name__ == '__main__':
    unittest.main()
