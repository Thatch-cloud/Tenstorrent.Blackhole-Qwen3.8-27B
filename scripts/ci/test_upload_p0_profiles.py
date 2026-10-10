"""The engine-start upload twins (make_upload_profiles, docs/tp4-fabric-upload.md P0 and P1a): each is the production profile plus exactly its switches and nothing else,
gate only; the checked-in file is what the generator makes; no other profile names a switch; the audit flags never reach a traffic profile."""

import copy
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_fusion_profiles as fusion  # noqa: E402
import make_upload_profiles as generator  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES_TEXT = (HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8')
PROFILES = json.loads(PROFILES_TEXT)['profiles']
PARENT = generator.PARENT


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

    def test_the_twins_follow_the_kill_drill_twin_in_spec_order_and_leave_the_last_block_to_the_kvread_twins(self):
        names = list(PROFILES)
        start = names.index(generator.KILL_TWIN) + 1
        self.assertEqual(names[start:start + len(generator.twin_names())], list(generator.twin_names()))
        self.assertEqual(names[names.index(PARENT) + 1], generator.KILL_TWIN, 'make_w2_kill_profiles keeps its twin right after the parent')
        self.assertNotIn(names[-1], generator.twin_names(), 'the file\'s end is make_kvread_profiles\'')
        without = json.loads(PROFILES_TEXT)
        del without['profiles'][generator.KILL_TWIN]
        names = list(generator.generate(without)['profiles'])
        self.assertEqual(names[names.index(PARENT) + 1:names.index(PARENT) + 1 + len(generator.twin_names())], list(generator.twin_names()))

    def test_the_twin_names_are_registered_with_the_census_exemptions(self):
        for name in generator.twin_names():
            self.assertIn(name, profile_twins.twin_names())
            self.assertIn(name, PROFILES)
        self.assertEqual(len(profile_twins.twin_names()), len(set(profile_twins.twin_names())))

    def test_no_name_starts_where_another_generator_anchors_its_block(self):
        for name in generator.twin_names():
            self.assertFalse(name.startswith('c2-packed-tp4-8x262k-ship-prefix-levern-parked'), name)

    def test_a_parent_that_cannot_be_twinned_is_refused_by_name(self):
        data = json.loads(PROFILES_TEXT)
        cases = (('already names a lever', lambda p: p['env'].update({generator.ZEROS: '1'}), 'already names QWEN_FAST_DEVICE_ZEROS'),
                 ('is gate only', lambda p: p.update(gate_only=True), 'is gate_only'),
                 ('not four cards', lambda p: p.update(mesh_device=None), 'does not open the four-card mesh'),
                 ('not tp4', lambda p: p['env'].update({'QWEN_FAST_TP': '2'}), 'is not a TP4 profile'))
        for label, edit, needle in cases:
            with self.subTest(label):
                broken = copy.deepcopy(data)
                edit(broken['profiles'][PARENT])
                with self.assertRaises(ValueError) as caught:
                    generator.generate(broken)
                self.assertIn(needle, str(caught.exception))
        with self.assertRaises(ValueError):
            broken = copy.deepcopy(data)
            del broken['profiles'][PARENT]
            generator.generate(broken)


class TwinTests(unittest.TestCase):
    def test_each_twin_is_the_parent_plus_exactly_its_switches_and_gate_only(self):
        self.assertEqual(len(generator.TWINS), 6)
        for suffix, env in generator.TWINS:
            twin = generator.twin_name(suffix)
            with self.subTest(twin=twin):
                mine, theirs = copy.deepcopy(PROFILES[twin]), copy.deepcopy(PROFILES[PARENT])
                for name, value in env.items():
                    self.assertEqual(mine['env'].pop(name), value)
                self.assertIn(PARENT, mine.pop('description'))
                theirs.pop('description')
                self.assertIs(mine.pop('gate_only'), True)
                self.assertIn(generator.WAIVER_FIELD, theirs)
                theirs.pop(generator.WAIVER_FIELD)
                self.assertEqual(mine, theirs, 'everything but the switches, the description, gate_only and the owner traffic waiver is the parent\'s')

    def test_the_audited_twins_carry_the_audit_of_the_levers_they_switch_on(self):
        by_suffix = dict(generator.TWINS)
        self.assertEqual(by_suffix['dzero'], {generator.ZEROS: '1'})
        self.assertEqual(by_suffix['dzero-audit'], {generator.ZEROS: '1', generator.ZEROS_AUDIT: '1'})
        self.assertEqual(by_suffix['lazy-audit'], {generator.LAZY: '1', generator.LAZY_AUDIT: '1'})
        self.assertEqual(by_suffix['upload'], {generator.ZEROS: '1', generator.LAZY: '1'})
        self.assertEqual(set(by_suffix['upload-audit']), {generator.ZEROS, generator.ZEROS_AUDIT, generator.LAZY, generator.LAZY_AUDIT})

    def test_the_description_says_what_the_arm_is_and_that_it_is_unqualified(self):
        for suffix, env in generator.TWINS:
            text = PROFILES[generator.twin_name(suffix)]['description']
            self.assertTrue(text.startswith('GATE ONLY, UNQUALIFIED'), suffix)
            for name in env:
                self.assertIn(name, text)
            self.assertIn(PARENT, text)
            self.assertNotIn('QWEN_FAST_DEVICE_ZEROS_AUDIT' if generator.ZEROS_AUDIT not in env else 'never-present', text)

    def test_every_twin_passes_the_upload_contract_and_the_audits_need_a_gate_profile(self):
        for suffix, env in generator.TWINS:
            twin = generator.twin_name(suffix)
            with self.subTest(twin=twin):
                profile = dict(PROFILES[twin], name=twin)
                self.assertEqual(contract.upload_problems(profile), [])
                traffic = dict(copy.deepcopy(profile), gate_only=False)
                audited = generator.ZEROS_AUDIT in env or generator.LAZY_AUDIT in env
                self.assertEqual(bool(contract.upload_problems(traffic)), audited, 'only the audit flags are a gate profile\'s own')

    def test_the_parent_and_every_other_profile_carry_no_switch(self):
        twins = set(generator.twin_names())
        combined = {fusion.NAMESPACE + 'all', fusion.NAMESPACE + 'all-audit'} | set(fusion.ship_names())            # (the ship candidate is the timed combined arm as a traffic profile)
        combined |= {name for name in PROFILES if name.startswith(fusion.NAMESPACE + 'wph-')}                       # (WPH's composition twins are the combined arm plus their own flags)
        for name, profile in PROFILES.items():
            if name not in twins and name not in combined:
                self.assertFalse([knob for knob in generator.KNOBS if knob in profile['env']], name)
        for name in sorted(combined):
            self.assertEqual(sorted(knob for knob in generator.KNOBS if knob in PROFILES[name]['env'] and not knob.endswith('_AUDIT')), sorted([generator.ZEROS, generator.LAZY]), name)
        self.assertEqual(contract.upload_problems(dict(PROFILES[PARENT], name=PARENT)), [])

    def test_an_inherited_process_environment_never_reaches_the_parent_and_the_twins_keep_their_own(self):
        inherited = {name: '1' for name in generator.KNOBS}
        inherited['KEEP'] = 'me'
        environ = contract.apply_environment(dict(PROFILES[PARENT], name=PARENT), dict(inherited))
        self.assertEqual(environ['KEEP'], 'me')
        self.assertFalse([name for name in generator.KNOBS if name in environ])
        twin = generator.twin_name('dzero')
        environ = contract.apply_environment(dict(PROFILES[twin], name=twin), dict(inherited))
        self.assertEqual(environ[generator.ZEROS], '1')
        self.assertFalse([name for name in generator.KNOBS if name != generator.ZEROS and name in environ])

    def test_the_knobs_are_the_contracts(self):
        self.assertEqual(set(generator.KNOBS), set(contract.UPLOAD_NAMES))


if __name__ == '__main__':
    unittest.main()
