"""The four-card S2 profiles (c2-packed-tp4, c2-packed-tp4-gate) and the pair's profiles they sit beside.

The pair's fast-path profiles are held byte for byte (a digest of each profile's canonical JSON, taken at the commit
before the four-card profiles existed): adding a width must not move what the pair serves. The four-card profiles are
their pair twins' engine and limits with exactly the documented differences - the mesh, the ring descriptor, QWEN_FAST_TP=4
and the fast path's four-card environment, FABRIC_1D - and pass the contract's mesh rules and the admission's environment
check when laid over the image's ENV."""

import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission as admission  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

ROOT = HERE.parent.parent
PROFILES = HERE / 'qwen_c2_profiles.json'
DOCKERFILE = ROOT / 'docker' / 'qwen-c2-serving.Dockerfile'
M3 = (True, 'users=4 FOUR_AS_TWO=0 PACKED_STEP=1')

# sha256 of json.dumps(profile, sort_keys=True, separators=(',', ':')) for the pair's fast-path profiles at fd5ea503.
PAIR_DIGESTS = {
    'exact': 'cb8b837b64cf6be5a81559b482e21cbe2316e40d1d8e3394e2ec0635c0d1cd69',
    'c2': '41e2e2feb2e0e82b8a13f3f4948558f3e267d3454b47524f964a95655412e0e9',
    'c2-gate': 'b283978e048bbbcbfb9772d9d5b8e536b5d5e2e8c5484deb69a3ba7b3c34733d',
    'c2-packed': '5b2a6fe3ca26edb1ec819f36175b62f113a6de98896323e3f6f217e260b649e0',
    'c2-packed-gate': '9e07f12ffed7e83dc8bd98f493b8ee1ec7903795f6992258b8bc5a604f4892a0',
    'c2-packed-prefix': '4a3021630eeaae7e8aaec41206cf8277598715195afcda3c072f8b5b733ee655',
    'c2-packed-prefix-gate': '3fac0901d8a7ccbf0b74332ad7b9dd5558622803315859cf9458f086ac65e5c1',
}
FOUR_ENV = {'QWEN_FAST_TP': '4', 'QWEN_FAST_SDPA_MODES': 'tail,share', 'QWEN_PROJECTION_LINKS': '2',
            'QWEN_GDN_PREFILL_MMRS': '0', 'QWEN_FAST_GDN_PREFILL_CONV_AUDIT': '4', 'QWEN_FAST_TP_KV_SLIDE': '1'}
# The nine image flags the four-card profiles turn off: their code carries the pair's chip or head literals in text-patched,
# kernel or two-chip form (quad_draft, fused_commit, the draft K/V slide and the traced publish's K/V fusion, the MLP block
# stream, the GDN direct-window and shared-QK experiments) that the four-card port has not reached.
OFF_ENV = {'QWEN_FAST_QUAD_DRAFT': '0', 'QWEN_FAST_FUSED_COMMIT': '0', 'QWEN_FAST_FUSED_COMMIT_INPLACE': '0',
           'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS': '0', 'QWEN_DRAFT_KV_SLIDE_EXPERIMENT': '0',
           'QWEN_FAST_TRACED_PUBLISH': '0', 'QWEN_MLP_BLOCK_STREAM_EXPERIMENT': '0', 'QWEN_GDN_DIRECT_WINDOW': '0',
           'QWEN_GDN_SHARED_QK_EXPERIMENT': '0'}
# QWEN_C2_GATE_PROFILE is the admission waiver's marker: only the gate-only profile's own env carries it.
AUDIT_ENV = {'QWEN_FAST_VERIFY_T1_AUDIT': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1', 'QWEN_C2_GATE_PROFILE': '1'}
# The traffic profile serves with the verify audits on (the gate's configuration): with them off it hung twice at the
# packed-to-sequential tail of the first four-user answers (SR v163, SS v164). Only the gate profile takes the waiver.
VERIFY_AUDITS = {'QWEN_FAST_VERIFY_T1_AUDIT': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1'}


SPEED = 'QWEN_FAST_TP_KV_SLIDE'
# The speed window's profiles (tp4/speed): each is c2-packed-tp4-gate plus exactly these env differences and a description.
SPEED_PROFILES = {
    'c2-packed-tp4-gate-noslide': {SPEED: '0'},
    'c2-packed-tp4-speed': {'QWEN_FAST_VERIFY_T1_AUDIT': '0', 'QWEN_FAST_VERIFY_T2_AUDIT': '0'},
    'c2-packed-tp4-speed-noslide': {'QWEN_FAST_VERIFY_T1_AUDIT': '0', 'QWEN_FAST_VERIFY_T2_AUDIT': '0', SPEED: '0'},
}


# The batched-draft window's profiles (tp4/draft): each is its speed or gate base plus exactly these env differences.
QUAD, SINGLES = 'QWEN_FAST_QUAD_DRAFT', 'QWEN_FAST_DRAFT_SINGLES_AUDIT'
DRAFT_PROFILES = {
    'c2-packed-tp4-speed-quad': ('c2-packed-tp4-speed', {QUAD: '1'}),
    'c2-packed-tp4-speed-pairs': ('c2-packed-tp4-speed', {QUAD: '0'}),
    'c2-packed-tp4-gate-quad': ('c2-packed-tp4-gate', {QUAD: '1', SINGLES: 'all'}),
    'c2-packed-tp4-gate-pairs': ('c2-packed-tp4-gate', {QUAD: '0', SINGLES: 'all'}),
}


def profiles():
    return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']


def digest(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def image_env():
    """The C2 image's own ENV (the first ENV of the Dockerfile's final stage), as a dict."""
    text = DOCKERFILE.read_text(encoding='utf-8')
    block = text[text.index('ENV QWEN_ATTN_PREP=1'):]
    block = block[:block.index('\n\n')] if '\n\n' in block else block
    lines = []
    for line in block.splitlines():
        lines.append(line.rstrip('\\').strip())
        if not line.rstrip().endswith('\\'):
            break
    return dict(re.findall(r'(QWEN_[A-Z0-9_]+)=(\S+)', ' '.join(lines)))


class PairUntouchedTests(unittest.TestCase):
    def test_the_pairs_fast_path_profiles_are_byte_for_byte_what_they_were(self):
        found = profiles()
        for name, expected in PAIR_DIGESTS.items():
            self.assertEqual(digest(found[name]), expected, '%s changed: a four-card change must not move the pair' % name)

    def test_the_pairs_profiles_name_no_mesh_and_set_no_width(self):
        for name in PAIR_DIGESTS:
            profile = profiles()[name]
            self.assertNotIn('mesh_device', profile, name)
            self.assertNotIn('QWEN_FAST_TP', profile['env'], name)
            self.assertEqual(profile['mesh_graph_descriptor'], contract.PAIR_DESCRIPTOR, name)
            self.assertEqual(contract.mesh_problems(dict(profile, name=name)), [], name)


class FourCardProfileTests(unittest.TestCase):
    def pair(self, name):
        return profiles()[name]

    def test_each_four_card_profile_is_its_pair_twin_with_the_documented_differences(self):
        for four, twin, extra in (('c2-packed-tp4', 'c2-packed', VERIFY_AUDITS), ('c2-packed-tp4-gate', 'c2-packed-gate', AUDIT_ENV)):
            mine, theirs = profiles()[four], self.pair(twin)
            with self.subTest(profile=four):
                self.assertEqual(mine['env'], dict(theirs['env'], **dict(FOUR_ENV, **dict(OFF_ENV, **extra))))
                engine = json.loads(json.dumps(mine['engine']))
                self.assertEqual(engine['additional-config']['tt'].pop('fabric_config'), 'FABRIC_1D')
                self.assertEqual(engine, theirs['engine'], 'the engine block is the twin\'s: 131,328 x 4 seats, 8,208 blocks')
                for key in set(mine) | set(theirs):
                    if key not in ('description', 'env', 'engine', 'mesh_device', 'mesh_graph_descriptor', 'gate_only', 'name'):
                        self.assertEqual(mine.get(key), theirs.get(key), key)
                self.assertEqual(mine['mesh_device'], 'P150x4')
                self.assertEqual(mine['mesh_graph_descriptor'], contract.RING_DESCRIPTOR)
                self.assertIs(mine.get('gate_only'), True if four.endswith('-gate') else None)
                self.assertNotIn('sample_on_device_mode', contract.tt_config(mine))
                self.assertTrue(mine['description'].startswith('GATE ONLY') == four.endswith('-gate'))

    def test_the_contract_accepts_the_fast_path_on_four_cards_only_under_the_width(self):
        for name in ('c2-packed-tp4', 'c2-packed-tp4-gate'):
            profile = dict(profiles()[name], name=name)
            self.assertEqual(contract.mesh_problems(profile), [], name)
            self.assertTrue(contract.ring_mesh(profile), name)
            environ = contract.apply_environment(profile, {'MESH_DEVICE': 'P300', 'TT_MESH_GRAPH_DESC_PATH': 'p300'})
            self.assertEqual((environ['MESH_DEVICE'], environ['TT_MESH_GRAPH_DESC_PATH'], environ['QWEN_FAST_TP']),
                             ('P150x4', contract.RING_DESCRIPTOR, '4'), name)
        base = json.loads(json.dumps(profiles()['c2-packed-tp4']))
        no_width = json.loads(json.dumps(base))
        del no_width['env']['QWEN_FAST_TP']
        self.assertTrue(any('fast path' in problem for problem in contract.mesh_problems(no_width)))
        sampler = json.loads(json.dumps(base))
        sampler['engine']['additional-config']['tt']['sample_on_device_mode'] = 'decode_only'
        self.assertTrue(any('sample twice' in problem for problem in contract.mesh_problems(sampler)))
        pair_with_width = json.loads(json.dumps(profiles()['c2-packed']))
        pair_with_width['env']['QWEN_FAST_TP'] = '4'
        self.assertTrue(any('needs mesh_device P150x4' in problem for problem in contract.mesh_problems(pair_with_width)))
        general_with_width = json.loads(json.dumps(profiles()['general-tp4']))
        general_with_width['env']['QWEN_FAST_TP'] = '4'
        self.assertTrue(any('qwen_fast_t16' in problem for problem in contract.mesh_problems(general_with_width)))

    def test_the_gate_profile_boots_only_in_a_gate_and_the_other_never_takes_the_waiver(self):
        self.assertIs(profiles()['c2-packed-tp4-gate']['gate_only'], True)
        self.assertNotIn('gate_only', profiles()['c2-packed-tp4'])

    def test_the_admission_accepts_each_profile_laid_over_the_image_environment(self):
        image = image_env()
        self.assertEqual(image['QWEN_FAST_SDPA_MODES'], 'tail,share,slice', 'the image ENV names the pair\'s modes')
        for name in ('c2-packed-tp4', 'c2-packed-tp4-gate'):
            environ = dict(image, **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])
        # and the pair's profile still passes the pair's check over the same image ENV
        environ = dict(image, **profiles()['c2-packed']['env'])
        self.assertEqual(admission.check_environment(environ, M3), [])

    def test_both_profiles_verify_with_the_audits_and_only_the_gate_takes_the_waiver(self):
        for name, waiver in (('c2-packed-tp4-gate', True), ('c2-packed-tp4', False)):
            env = profiles()[name]['env']
            self.assertEqual({key: env.get(key) for key in VERIFY_AUDITS}, VERIFY_AUDITS, name)
            self.assertEqual('QWEN_C2_GATE_PROFILE' in env, waiver, name)

    def test_the_switched_off_flags_are_the_images_own_and_are_off(self):
        image = image_env()
        for name in OFF_ENV:
            self.assertIn(name, image, name)
            self.assertEqual(image[name], '1', 'the image turns it on: %s' % name)
            for profile in ('c2-packed-tp4', 'c2-packed-tp4-gate'):
                self.assertEqual(profiles()[profile]['env'][name], '0', (profile, name))

    def test_the_four_card_environment_names_no_value_the_pair_does(self):
        for name in ('c2-packed-tp4', 'c2-packed-tp4-gate'):
            env = profiles()[name]['env']
            self.assertEqual(env['QWEN_PROJECTION_LINKS'], '2', 'two trained links per edge (the image names the pair\'s 4)')
            self.assertEqual(env['QWEN_FAST_SDPA_MODES'], 'tail,share')
            self.assertEqual(env['QWEN_GDN_PREFILL_MMRS'], '0')


class SpeedProfileTests(unittest.TestCase):
    def test_the_slide_flag_is_the_four_card_profiles_and_no_pairs(self):
        found = profiles()
        for name in ('c2-packed-tp4', 'c2-packed-tp4-gate', 'c2-packed-tp4-gate-ring', 'c2-packed-tp4-gate-bf16'):
            self.assertEqual(found[name]['env'][SPEED], '1', name)
        for name in PAIR_DIGESTS:
            self.assertNotIn(SPEED, found[name]['env'], name)
        self.assertNotIn(SPEED, image_env(), 'the image leaves it unset: the eager chain is the default')

    def test_each_speed_profile_is_the_gate_with_only_its_documented_difference(self):
        found = profiles()
        gate = found['c2-packed-tp4-gate']
        for name, difference in SPEED_PROFILES.items():
            mine = found[name]
            with self.subTest(profile=name):
                self.assertEqual(mine['env'], dict(gate['env'], **difference))
                self.assertEqual(mine['engine'], gate['engine'])
                for key in set(mine) | set(gate):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), gate.get(key), key)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                self.assertIs(mine['gate_only'], True)
                self.assertEqual(mine['env']['QWEN_C2_GATE_PROFILE'], '1', 'the admission waiver: no evidence section is filled yet')
                self.assertEqual(contract.mesh_problems(dict(mine, name=name)), [], name)

    def test_the_timed_arms_have_both_verify_audits_off_and_the_audited_arms_on(self):
        found = profiles()
        for name, audited in (('c2-packed-tp4-gate', True), ('c2-packed-tp4-gate-noslide', True),
                              ('c2-packed-tp4-speed', False), ('c2-packed-tp4-speed-noslide', False)):
            env = found[name]['env']
            for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
                self.assertEqual(env[key], '1' if audited else '0', (name, key))

    def test_the_speed_pair_differs_only_in_the_slide(self):
        found = profiles()
        on, off = found['c2-packed-tp4-speed']['env'], found['c2-packed-tp4-speed-noslide']['env']
        self.assertEqual({key for key in on if on[key] != off.get(key)}, {SPEED})
        self.assertEqual((on[SPEED], off[SPEED]), ('1', '0'))
        on, off = found['c2-packed-tp4-gate']['env'], found['c2-packed-tp4-gate-noslide']['env']
        self.assertEqual({key for key in on if on[key] != off.get(key)}, {SPEED})

    def test_the_admission_accepts_each_speed_profile_over_the_image_environment(self):
        image = image_env()
        for name in SPEED_PROFILES:
            environ = dict(image, **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])


class DraftProfileTests(unittest.TestCase):
    def test_each_draft_profile_is_its_base_with_only_its_documented_difference(self):
        found = profiles()
        for name, (base_name, difference) in DRAFT_PROFILES.items():
            mine, base = found[name], found[base_name]
            with self.subTest(profile=name):
                self.assertEqual(mine['env'], dict(base['env'], **difference))
                self.assertEqual(mine['engine'], base['engine'])
                for key in set(mine) | set(base):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), base.get(key), key)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                self.assertIs(mine['gate_only'], True)
                self.assertEqual(mine['env']['QWEN_C2_GATE_PROFILE'], '1')
                self.assertEqual(contract.mesh_problems(dict(mine, name=name)), [], name)
                self.assertEqual(mine['env']['QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'], '0',
                                 'the four-card quad copies the active banks: the live-banks flag stays off')

    def test_the_flag_pair_differs_in_the_quad_flag_alone(self):
        found = profiles()
        for on, off in (('c2-packed-tp4-speed-quad', 'c2-packed-tp4-speed-pairs'),
                        ('c2-packed-tp4-gate-quad', 'c2-packed-tp4-gate-pairs')):
            left, right = found[on]['env'], found[off]['env']
            self.assertEqual({key for key in set(left) | set(right) if left.get(key) != right.get(key)}, {QUAD}, on)
            self.assertEqual((left[QUAD], right[QUAD]), ('1', '0'))
        self.assertEqual(found['c2-packed-tp4-speed-pairs']['env'], found['c2-packed-tp4-speed']['env'],
                         'the -pairs profile is the speed profile, stated as the pair of -quad')

    def test_the_quad_is_the_only_profile_family_with_the_flag_on_and_the_pairs_flags_are_the_images(self):
        found = profiles()
        on = sorted(name for name, profile in found.items() if profile['env'].get(QUAD) == '1'
                    and profile['env'].get('QWEN_FAST_TP') == '4')
        self.assertEqual(on, ['c2-packed-tp4-gate-quad', 'c2-packed-tp4-speed-quad'])
        image = image_env()
        for name in ('QWEN_FAST_PACKED_PROPOSAL', 'QWEN_FAST_PAIR_ROW_EXACT', 'QWEN_FAST_ROUND_B1', 'QWEN_FAST_PACKED_AUDIT'):
            self.assertEqual(image[name], '1', 'the batched draft needs the image\'s own %s' % name)
        for name in DRAFT_PROFILES:
            self.assertNotIn('QWEN_FAST_PACKED_PROPOSAL', found[name]['env'], 'the image sets it; a profile never turns it off')
        for name in PAIR_DIGESTS:
            self.assertNotIn(SINGLES, found[name]['env'])
        self.assertNotIn(SINGLES, image, 'off in the image: only the audited draft profiles ask for it')

    def test_the_audit_is_on_the_audited_profiles_only_and_the_timed_ones_have_no_audit(self):
        found = profiles()
        for name, audited in (('c2-packed-tp4-gate-quad', True), ('c2-packed-tp4-gate-pairs', True),
                              ('c2-packed-tp4-speed-quad', False), ('c2-packed-tp4-speed-pairs', False)):
            env = found[name]['env']
            self.assertEqual(env.get(SINGLES), 'all' if audited else None, name)
            for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
                self.assertEqual(env[key], '1' if audited else '0', (name, key))

    def test_the_admission_accepts_each_draft_profile_over_the_image_environment(self):
        image = image_env()
        for name in DRAFT_PROFILES:
            environ = dict(image, **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])


if __name__ == '__main__':
    unittest.main()
