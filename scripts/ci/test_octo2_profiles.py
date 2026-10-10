"""The tp4/octo-2 lever twins (make_octo2_profiles): each is its parent (the octo-T8 timed arm) plus exactly the documented env; gate only; no other profile names a lever flag; both
generators agree on the one profiles file; and each twin is admitted over the C2 image's own ENV (the draft's and the bundle's reasons join the octo admission)."""

import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_octo2_profiles as twins  # noqa: E402
import make_octo_profiles as octo_twins  # noqa: E402
import serving_octo as octo  # noqa: E402
from test_c2_packed_tp4_profiles import profiles  # noqa: E402
from test_octo_profiles import container_env  # noqa: E402
import test_octo as host_tests  # noqa: E402

PROFILES_TEXT = (HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8')
NAMES = twins.twin_names()
M3 = (True, 'users=8 FOUR_AS_TWO=0 PACKED_STEP=1 M3_BLOCKS=2')
EXPECTED = {
    twins.BASE + '-octo-draft': {twins.DRAFT: '1'},
    twins.BASE + '-octo-bundle': {twins.BUNDLE: '4'},
    twins.BASE + '-octo-bundle-audit': {twins.BUNDLE: '4', twins.BUNDLE_AUDIT: '1'},
    twins.BASE + '-octo-glue8': {twins.GLUE8: '1'},
    twins.BASE + '-octo-glue8-audit': {twins.GLUE8: '1'},
    twins.BASE + '-octo-levers': {twins.DRAFT: '1', twins.BUNDLE: '4', twins.GLUE8: '1'},
}


class GeneratorTests(unittest.TestCase):
    def test_the_checked_in_twins_are_what_the_generator_makes(self):
        self.assertEqual(twins.render(twins.generate(json.loads(PROFILES_TEXT))), PROFILES_TEXT)

    def test_the_octo_generator_still_agrees_on_the_same_file(self):
        self.assertEqual(octo_twins.render(octo_twins.generate(json.loads(PROFILES_TEXT))), PROFILES_TEXT)

    def test_the_twins_follow_the_octo_block_and_nothing_else_moved(self):
        names = list(json.loads(PROFILES_TEXT)['profiles'])
        last = max(index for index, name in enumerate(names) if name in octo_twins.twin_names())
        self.assertEqual(names[last + 1:last + 1 + len(NAMES)], list(NAMES))

    def test_the_generator_is_idempotent_and_refuses_a_moved_parent(self):
        data = json.loads(PROFILES_TEXT)
        self.assertEqual(twins.generate(twins.generate(data)), twins.generate(data))
        broken = json.loads(PROFILES_TEXT)
        broken['profiles'][twins.PARENT]['env'][twins.DRAFT] = '1'
        with self.assertRaises(ValueError) as raised:
            twins.generate(broken)
        self.assertIn('never carries the lever it is twinned with', str(raised.exception))
        broken = json.loads(PROFILES_TEXT)
        broken['profiles'][twins.PARENT]['env']['QWEN_FAST_OCTO'] = 'live'
        with self.assertRaises(ValueError):
            twins.generate(broken)


class TwinTests(unittest.TestCase):
    def test_the_names_and_the_env_additions(self):
        self.assertEqual({name: env for name, parent, env, why in twins.specs()}, EXPECTED)
        self.assertEqual(set(NAMES), set(EXPECTED))

    def test_each_twin_is_its_parent_plus_exactly_its_env_and_gate_only(self):
        for name, added in EXPECTED.items():
            with self.subTest(profile=name):
                parent = profiles()[twins.AUDIT_PARENT if name.endswith('-glue8-audit') else twins.PARENT]
                profile = profiles()[name]
                self.assertEqual(profile['env'], dict(parent['env'], **added))
                self.assertEqual({key: value for key, value in profile.items() if key not in ('env', 'description')},
                                 {key: value for key, value in parent.items() if key not in ('env', 'description')})
                self.assertIs(profile['gate_only'], True)
                self.assertEqual(profile['env']['QWEN_C2_GATE_PROFILE'], '1')
                self.assertEqual(profile['env']['QWEN_FAST_OCTO'], 'alternate')
                self.assertIn('GATE ONLY, UNQUALIFIED', profile['description'])
                self.assertIn('The parent is %s ' % (twins.AUDIT_PARENT if name.endswith('-glue8-audit') else twins.PARENT), profile['description'])

    def test_no_other_profile_names_a_lever_flag(self):
        for name, profile in profiles().items():
            if name in EXPECTED:
                continue
            with self.subTest(profile=name):
                self.assertFalse(set(twins.FLAGS) & set(profile.get('env') or {}))

    def test_the_audited_glue_twin_carries_the_vglue_audit_and_no_draft_beside_the_singles_audit(self):
        env = profiles()[twins.BASE + '-octo-glue8-audit']['env']
        self.assertEqual((env['QWEN_FAST_TP4_VGLUE_AUDIT'], env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1', '1'))
        self.assertEqual(env['QWEN_FAST_DRAFT_SINGLES_AUDIT'], 'all')
        self.assertNotIn(twins.DRAFT, env, 'the octo draft is refused beside the singles audit')
        for name in NAMES:
            if twins.DRAFT in profiles()[name]['env']:
                self.assertEqual(profiles()[name]['env'].get('QWEN_FAST_DRAFT_SINGLES_AUDIT'), None, name)

    def test_the_memory_plan_is_the_parents(self):
        parent = profiles()[twins.PARENT]
        for name in NAMES:
            engine = profiles()[name]['engine']
            self.assertEqual(engine['num-gpu-blocks-override'], parent['engine']['num-gpu-blocks-override'])
            self.assertEqual(engine['additional-config']['tt']['trace_region_size'], parent['engine']['additional-config']['tt']['trace_region_size'])
            self.assertEqual(profiles()[name]['env']['QWEN36_MAX_TOKENS_ALL_USERS'], parent['env']['QWEN36_MAX_TOKENS_ALL_USERS'])


class AdmissionOverTheImageTests(unittest.TestCase):
    """Each twin over the C2 image's own ENV: the octo admission, with the levers' reasons folded in, admits it."""

    def test_the_twins_are_admitted_over_the_images_env(self):
        for name in NAMES:
            with self.subTest(profile=name):
                if name.endswith(('-glue8', '-levers')):
                    try:
                        import octo_glue8  # noqa: F401
                    except ImportError:
                        self.skipTest('Lever 2 is not in this tree yet')
                log = host_tests.Lines()
                record = octo.octo_admission(M3, container_env(name), log=log)
                self.assertEqual((record['mode'], record['min_live']), ('alternate', 6))
                flags = profiles()[name]['env']
                self.assertEqual(bool(record.get('draft')), flags.get(twins.DRAFT) == '1')
                self.assertEqual(len([line for line in log.lines if line.startswith(octo.UNQUALIFIED_MARKER)]), len(octo.UNQUALIFIED_ITEMS))
                self.assertFalse([line for line in log.lines if line.startswith(octo.REFUSED_MARKER)])

    def test_the_judge_reads_each_twin_as_an_alternate_octo_profile(self):
        import octo_judge

        for name in NAMES:
            env = profiles()[name]['env']
            self.assertEqual(octo_judge.mode(env), 'alternate')
            self.assertEqual(octo_judge.min_live(env), 6)


if __name__ == '__main__':
    unittest.main()
