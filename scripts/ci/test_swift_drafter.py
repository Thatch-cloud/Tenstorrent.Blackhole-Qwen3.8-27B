"""The Swift-1.5 drafter fine-tune (tp4/swift-drafter): its manifest, config, checkpoint pins, tau-lab arm and card pack.

CPU only. The weights are not in the repository: every check here reads pins, never bytes (the bytes are checked at the image build by
drafter_stage.py and at every attach by drafter_fixtures / drafter_checkpoint).
"""

import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import c2_tau_lab as lab  # noqa: E402
import drafter_checkpoint as checkpoints  # noqa: E402
import drafter_manifest as manifests  # noqa: E402
import drafter_pair_report as pair_report  # noqa: E402

PACK = HERE / 'references' / 'fusion-jobs' / 'DR'
PROFILES = HERE / 'qwen_c2_profiles.json'
ARM = 'sw1ft-bf8'
SWIFT_REPO = 'ukisai/Swift-1.5-Qwen3.8-27b'
SWIFT_REVISION = 'b4c84d42903a8646b25857eb2827288d92ed85a4'
CONTROL_IMAGE = 'tp4-swift-drafter-0'
CANDIDATE_IMAGE = 'tp4-swift-drafter-1'
PROFILE_NAMES = sorted(json.loads(PROFILES.read_text(encoding='utf-8'))['profiles'])


def candidate_name():
    """The manifest the pack builds: B1's C2_DRAFTER_MANIFEST (older sw1-ft manifests may stay committed beside it)."""
    return job.parse_env((PACK / 'B1-build-sw1ft.env').read_text(encoding='utf-8'))['C2_DRAFTER_MANIFEST']


NAME = candidate_name()


def rows():
    return [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


def values(name):
    return job.parse_env((PACK / (name + '.env')).read_text(encoding='utf-8'))


class ManifestTests(unittest.TestCase):
    def test_the_packs_manifest_is_committed_and_the_only_one_the_arm_runs(self):
        self.assertIn(NAME, manifests.names())
        self.assertTrue(NAME.startswith('sw1-ft'))

    def test_it_is_a_local_checkpoint_in_the_default_layout(self):
        manifest = manifests.load(NAME)
        default = manifests.load(manifests.DEFAULT)
        self.assertEqual(manifest['source'], 'local')
        self.assertTrue(manifest['model'].startswith('local/'))
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
        paths = [HERE / 'references' / 'drafter-manifests' / (NAME + '.json'), HERE / 'references' / 'drafter-configs' / (NAME + '.config.json'),
                 HERE / 'drafter_checkpoints.json'] + sorted(PACK.glob('*'))
        for path in paths:
            text = path.read_text(encoding='utf-8').lower()
            for word in ('/home/', '192.168', '10.10.', '172.16', 'thatch', 'password', 'token ', 'zot.', 'ghcr'):
                self.assertNotIn(word, text, '%s: %s' % (path.name, word))


class CheckpointTableTests(unittest.TestCase):
    def test_the_table_holds_a_valid_pinned_entry(self):
        document = checkpoints.load()
        self.assertEqual(checkpoints.table_problems(document), [])
        entry = document['checkpoints'][self.entry_id(document)]
        manifest = manifests.load(NAME)
        self.assertEqual(entry['revision'], manifest['revision'])
        self.assertEqual(entry['config_sha256'], manifest['config_sha256'])
        self.assertEqual(entry['geometry'], document['checkpoints'][document['default']]['geometry'])
        self.assertEqual(sorted(entry['manifests']), ['attention/manifest.json', 'convolution/manifest.json', 'layer-1/manifest.json',
                                                      'layer-2/manifest.json', 'layer-3/manifest.json', 'layer-4/manifest.json',
                                                      'mlp/manifest.json', 'projection/manifest.json', 'selector/manifest.json'])

    def entry_id(self, document):
        found = [name for name in document['checkpoints'] if name != document['default'] and NAME in name]
        self.assertEqual(len(found), 1, sorted(document['checkpoints']))
        return found[0]


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


class PackTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates(self):
        listed = sorted(row[0] + '.env' for row in rows())
        self.assertEqual(listed, sorted(name for name in os.listdir(str(PACK)) if name.endswith('.env')))

    def test_every_row_names_a_mode_an_image_and_minutes(self):
        for row in rows():
            self.assertIn(row[1], ('stop', 'soft'), row)
            self.assertIn(row[2], (CONTROL_IMAGE, CANDIDATE_IMAGE), row)
            self.assertTrue(row[3].isdigit(), row)

    def pushable(self, name):
        data = values(name)
        if data.get('C2_TAULAB_DRAFTER_ARM') not in (None, '', 'control'):
            # a candidate arm is judged against its control run: without the key the reader refuses it, with the placeholder it passes
            without = dict((k, v) for k, v in data.items() if k != 'C2_TAULAB_PAIR_CONTROL')
            with self.assertRaises(job.JobError, msg='%s must be refused without a control run' % name):
                job.read_job(without, PROFILE_NAMES, meshes=job.profile_meshes())
        return data

    def test_every_template_parses_with_the_job_reader(self):
        for row in rows():
            outputs = job.read_job(self.pushable(row[0]), PROFILE_NAMES, meshes=job.profile_meshes())
            self.assertEqual(outputs['cards'], 'quad', row[0])

    def test_the_two_tau_jobs_are_a_paired_control_and_candidate_on_swift(self):
        control, candidate = values('T1-tau-swift-control'), values('T2-tau-swift-sw1ft')
        for data in (control, candidate):
            self.assertEqual((data['C2_MODEL_REPO'], data['C2_MODEL_REVISION']), (SWIFT_REPO, SWIFT_REVISION))
            self.assertEqual(data['C2_TAULAB_PROFILE'], 'c2-packed-tp4')
            self.assertEqual(data['C2_TAULAB_ARMS'].split(), ['A1', 'A2'])
        self.assertEqual((control['C2_IMAGE_TAG'], control['C2_TAULAB_DRAFTER_ARM']), (CONTROL_IMAGE, 'control'))
        self.assertEqual((candidate['C2_IMAGE_TAG'], candidate['C2_TAULAB_DRAFTER_ARM']), (CANDIDATE_IMAGE, ARM))
        self.assertEqual(candidate['C2_TAULAB_PAIR_CONTROL'], '9999999999', 'the placeholder the header tells the operator to replace')
        self.assertNotIn('C2_TAULAB_PAIR_CONTROL', control)

    def test_the_builds_make_the_two_images_from_one_commit(self):
        control, candidate = values('B0-build-control'), values('B1-build-sw1ft')
        self.assertEqual((control['C2_ACTIONS'], control['C2_IMAGE_TAG']), ('build', CONTROL_IMAGE))
        self.assertEqual((candidate['C2_ACTIONS'], candidate['C2_IMAGE_TAG'], candidate['C2_DRAFTER_MANIFEST']), ('build', CANDIDATE_IMAGE, NAME))
        self.assertNotIn('C2_DRAFTER_MANIFEST', control)
        self.assertNotIn('push', candidate['C2_ACTIONS'].split())
        for key in ('C2_BAKE_DEFAULT_PROFILE', 'C2_BAKE_GRAFT'):
            self.assertEqual(control[key], candidate[key])

    def test_the_timed_pair_alternates_the_two_images_on_swift_with_one_profile(self):
        order = ['C1-swift-control-a', 'L1-swift-sw1ft-a', 'C2-swift-control-b', 'L2-swift-sw1ft-b']
        images = [values(name)['C2_IMAGE_TAG'] for name in order]
        self.assertEqual(images, [CONTROL_IMAGE, CANDIDATE_IMAGE, CONTROL_IMAGE, CANDIDATE_IMAGE])
        self.assertEqual(len(set(values(name)['C2_PROFILE'] for name in order)), 1)
        for name in order:
            self.assertEqual(values(name)['C2_MODEL_REPO'], SWIFT_REPO)
            self.assertEqual(values(name)['C2_SMOKE_TESTS'], 'warmup,concurrent8_steady,concurrent8_code_32k_long')
        self.assertEqual([row[0] for row in rows() if row[0] in order], order)

    def test_needs_lines_name_earlier_jobs(self):
        names = [row[0] for row in rows()]
        for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines():
            if line.startswith('# NEEDS '):
                left, right = line[len('# NEEDS '):].split(' <- ')
                for target in left.split():
                    for need in right.split():
                        self.assertIn(target, names)
                        self.assertIn(need, names)
                        self.assertLess(names.index(need), names.index(target), line)

    def test_the_header_minutes_match_the_rows(self):
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        total = sum(int(row[3]) for row in rows())
        self.assertIn('Time (estimate, minutes): %d in all' % total, text)


if __name__ == '__main__':
    unittest.main()
