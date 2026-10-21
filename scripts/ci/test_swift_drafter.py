"""The Swift-1.5 drafter fine-tune v3 (the long-context fine-tune of the served drafter): its local manifest, committed config, tau-lab arm and job lines.

CPU only. The weights are not in the repository: every check here reads pins, never bytes (the bytes are checked at the image build by
drafter_stage.py and at every attach by drafter_fixtures).
"""

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import c2_tau_lab as lab  # noqa: E402
import drafter_manifest as manifests  # noqa: E402
import drafter_pair_report as pair_report  # noqa: E402

NAME = 'sw1-ft3-d35114a8'
WEIGHTS_SHA256 = 'd35114a88046edd9f4a16cfe0ad62138dd880b6c19b3f2b2780b91b5b5e06e5d'
ARM = 'sw1ft3-bf8'
SWIFT_REPO = 'ukisai/Swift-1.5-Qwen3.8-27b'
SWIFT_REVISION = 'b4c84d42903a8646b25857eb2827288d92ed85a4'
PROFILES = HERE / 'qwen_c2_profiles.json'
PROFILE_NAMES = sorted(json.loads(PROFILES.read_text(encoding='utf-8'))['profiles'])
SHIP_PROFILE = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s7'

# The job lines of the card window (the operator copies them over .github/c2-serving-job.env, ONE TAG AT A TIME): the build of the candidate image, the
# control tau run on the s7 ship image and the paired candidate tau run. The tags are the operator's; C2_TAULAB_PAIR_CONTROL is the control run's id.
BUILD = 'C2_CARDS=quad\nC2_ACTIONS=build\nC2_IMAGE_TAG=tp4-s7-dr3\nC2_BAKE_DEFAULT_PROFILE=%s\nC2_BAKE_GRAFT=K64j-OQ\nC2_DRAFTER_MANIFEST=%s\n' % (SHIP_PROFILE, NAME)
SWIFT = 'C2_MODEL_REPO=%s\nC2_MODEL_REVISION=%s\n' % (SWIFT_REPO, SWIFT_REVISION)
TAU_CONTROL = ('C2_CARDS=quad\nC2_ACTIONS=reset taulab\nC2_IMAGE_TAG=tp4-s7s\n' + SWIFT + 'C2_TAULAB_PROFILE=c2-packed-tp4\nC2_TAULAB_DRAFTER_ARM=control\n'
               'C2_TAULAB_ARMS=A1 A2\nC2_TAULAB_DEADLINE=70\nC2_TAULAB_IN_FLIGHT=8\n')
TAU_CANDIDATE = ('C2_CARDS=quad\nC2_ACTIONS=reset taulab\nC2_IMAGE_TAG=tp4-s7-dr3\n' + SWIFT + 'C2_TAULAB_PROFILE=c2-packed-tp4\nC2_TAULAB_DRAFTER_ARM=%s\n'
                 'C2_TAULAB_ARMS=A1 A2\nC2_TAULAB_PAIR_CONTROL=9999999999\nC2_TAULAB_DEADLINE=70\nC2_TAULAB_IN_FLIGHT=8\n' % ARM)


class ManifestTests(unittest.TestCase):
    def test_the_manifest_is_committed(self):
        self.assertIn(NAME, manifests.names())

    def test_it_is_a_local_checkpoint_in_the_default_layout(self):
        manifest = manifests.load(NAME)
        default = manifests.load(manifests.DEFAULT)
        self.assertEqual(manifest['source'], 'local')
        self.assertTrue(manifest['model'].startswith('local/'))
        self.assertEqual(manifest['weights_sha256'], WEIGHTS_SHA256)
        self.assertEqual(manifest['revision'], manifest['weights_sha256'][:40])
        self.assertTrue(NAME.endswith(manifest['weights_sha256'][:8]))
        self.assertEqual(manifest['header_sha256'], default['header_sha256'])
        self.assertEqual(manifest['checkpoint_bytes'], default['checkpoint_bytes'])
        self.assertEqual(set(manifest['tensors']), set(default['tensors']))
        for tensor, pin in manifest['tensors'].items():
            self.assertEqual(pin['shape'], default['tensors'][tensor]['shape'], tensor)
        # a fine-tune moves the trained tensors; the frozen ones (the recipe freezes fc and hidden_norm) are the served drafter's own
        self.assertEqual(manifest['tensors']['fc.weight'], default['tensors']['fc.weight'])
        self.assertEqual(manifest['tensors']['hidden_norm.weight'], default['tensors']['hidden_norm.weight'])
        self.assertEqual(manifest['fc_first32_sha256'], default['fc_first32_sha256'])
        moved = [tensor for tensor in manifest['tensors'] if manifest['tensors'][tensor]['sha256'] != default['tensors'][tensor]['sha256']]
        self.assertGreater(len(moved), 60, 'a fine-tune that left most tensors alone is not one')
        self.assertEqual(manifest['trained_block_size'], 16)

    def test_the_committed_config_is_the_served_one_with_block_16(self):
        manifest = manifests.load(NAME)
        path = HERE / 'references' / 'drafter-configs' / (NAME + '.config.json')
        self.assertEqual(manifests.sha256_file(path), manifest['config_sha256'])
        ours = json.loads(path.read_text(encoding='utf-8'))
        theirs = json.loads((HERE / 'references' / 'drafter-configs' / 'dedf8df6.config.json').read_text(encoding='utf-8'))
        self.assertEqual(ours['dflash_config']['block_size'], 16)
        ours['dflash_config']['block_size'] = theirs['dflash_config']['block_size']
        self.assertEqual(ours, theirs)

    def test_the_build_script_copies_the_committed_config_of_a_local_drafter(self):
        text = (HERE / 'build-c2-serving-image.sh').read_text(encoding='utf-8')
        self.assertIn('drafter_source', text)
        self.assertIn('references/drafter-configs/$drafter.config.json', text)
        self.assertLess(text.index('drafter-configs/$drafter.config.json'), text.index('resolve/$revision/config.json'))
        dockerfile = (HERE.parent.parent / 'docker' / 'qwen-c2-serving.Dockerfile').read_text(encoding='utf-8')
        self.assertIn('install -d -m 0755 /experiment-scripts/ci/references/drafter-configs', dockerfile)

    def test_the_overlay_carries_the_manifest_and_the_config(self):
        overlay = (HERE.parent.parent / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').splitlines()
        self.assertIn('scripts/ci/references/drafter-manifests/%s.json' % NAME, overlay)
        self.assertIn('scripts/ci/references/drafter-configs/%s.config.json' % NAME, overlay)

    def test_a_local_manifest_is_refused_when_it_is_not_one(self):
        good = json.loads(json.dumps(manifests.load(NAME)))
        manifests.validate(good, NAME)
        for key, value in (('source', 'elsewhere'), ('model', 'owner/name'), ('revision', '0' * 40), ('base', None)):
            bad = json.loads(json.dumps(good))
            bad[key] = value
            if value is None:
                del bad[key]
            with self.assertRaises(ValueError, msg=key):
                manifests.validate(bad, NAME)

    def test_no_private_detail_in_what_this_branch_commits(self):
        paths = [HERE / 'references' / 'drafter-manifests' / (NAME + '.json'), HERE / 'references' / 'drafter-configs' / (NAME + '.config.json')]
        for path in paths:
            text = path.read_text(encoding='utf-8').lower()
            for word in ('/home/', '192.168', '10.10.', '172.16', 'thatch', 'password', 'token ', 'zot.', 'ghcr'):
                self.assertNotIn(word, text, '%s: %s' % (path.name, word))


class ArmTests(unittest.TestCase):
    def test_the_arm_is_wired_everywhere_a_drafter_arm_is(self):
        self.assertIn(ARM, job.TAULAB_DRAFTER_ARMS)
        self.assertIn(ARM, pair_report.CANDIDATE_ARMS)
        self.assertNotIn(ARM, pair_report.SAME_IMAGE_ARMS)
        self.assertEqual(lab.DRAFTER_ARMS[ARM]['manifest'], NAME)
        self.assertEqual(lab.DRAFTER_ARMS[ARM]['env'], (('QWEN_DRAFTER_MANIFEST', NAME),))

    def test_the_derived_profile_adds_only_the_manifest_flag(self):
        profiles = json.loads(PROFILES.read_text(encoding='utf-8'))
        name, document = lab.derive_profile(profiles, 'c2-packed-tp4', ARM)
        self.assertEqual(lab.arithmetic_diff(profiles, 'c2-packed-tp4', document, ARM), [])
        added = dict((k, v) for k, v in document['profiles'][name]['env'].items() if profiles['profiles']['c2-packed-tp4']['env'].get(k) != v)
        self.assertEqual(added, dict(QWEN_FAST_PACKED_AUDIT='1', QWEN_FAST_PHASE_TIMING='1', QWEN_DRAFTER_MANIFEST=NAME))

    def test_the_log_marker_is_the_manifests(self):
        derived = 'c2-packed-tp4+taulab+' + ARM
        good = ['[QWEN-C2] profile %s' % derived, '[DRAFTER_MANIFEST] %s in force: local/%s at %s, 81 tensors verified' % (
            NAME, NAME, manifests.load(NAME)['revision'][:12]), '[QWEN-C2] checkpoint %s@%s key=k' % (SWIFT_REPO, SWIFT_REVISION)]
        self.assertEqual([p for p in lab.launched_problems(good[:2], derived, ARM) if 'checkpoint' not in p], [])
        self.assertTrue(lab.launched_problems(good[:1], derived, ARM))


class JobLineTests(unittest.TestCase):
    def read(self, text):
        return job.read_job(job.parse_env(text), PROFILE_NAMES, meshes=job.profile_meshes())

    def test_the_build_makes_the_candidate_image_from_the_ship_profile(self):
        outputs = self.read(BUILD)
        self.assertEqual((outputs['cards'], outputs['tag'], outputs['drafter_manifest'], outputs['bake_graft']), ('quad', 'tp4-s7-dr3', NAME, 'K64j-OQ'))
        self.assertEqual(outputs['bake_default_profile'], SHIP_PROFILE)
        self.assertIn(SHIP_PROFILE, PROFILE_NAMES)
        self.assertNotIn('push', job.parse_env(BUILD)['C2_ACTIONS'].split())

    def test_the_control_tau_run_reads_and_the_candidate_needs_its_control(self):
        outputs = self.read(TAU_CONTROL)
        self.assertEqual((outputs['taulab_arms'], outputs['taulab_profile']), ('A1 A2', 'c2-packed-tp4'))
        data = job.parse_env(TAU_CANDIDATE)
        self.assertEqual(self.read(TAU_CANDIDATE)['taulab_arms'], 'A1 A2')
        self.assertEqual((data['C2_TAULAB_DRAFTER_ARM'], data['C2_IMAGE_TAG']), (ARM, 'tp4-s7-dr3'))
        # a candidate arm is judged against its control run: without the key the reader refuses it
        without = dict((k, v) for k, v in data.items() if k != 'C2_TAULAB_PAIR_CONTROL')
        with self.assertRaises(job.JobError):
            job.read_job(without, PROFILE_NAMES, meshes=job.profile_meshes())

    def test_the_pair_runs_one_profile_and_the_swift_checkpoint(self):
        control, candidate = job.parse_env(TAU_CONTROL), job.parse_env(TAU_CANDIDATE)
        for key in ('C2_TAULAB_PROFILE', 'C2_MODEL_REPO', 'C2_MODEL_REVISION', 'C2_TAULAB_ARMS', 'C2_TAULAB_DEADLINE', 'C2_TAULAB_IN_FLIGHT'):
            self.assertEqual(control[key], candidate[key], key)
        self.assertEqual((control['C2_MODEL_REPO'], control['C2_MODEL_REVISION']), (SWIFT_REPO, SWIFT_REVISION))
        self.assertNotEqual(control['C2_IMAGE_TAG'], candidate['C2_IMAGE_TAG'])


if __name__ == '__main__':
    unittest.main()
