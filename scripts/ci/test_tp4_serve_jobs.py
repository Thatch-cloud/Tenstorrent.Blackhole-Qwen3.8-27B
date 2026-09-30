"""The four-card serving job templates (references/tp4-serve-jobs): every one parses under the checkout's own job
parser once its placeholders are filled, runs the actions its header says, and stays on the four-card profile.
Stdlib only."""

import glob
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(HERE, 'references', 'tp4-serve-jobs')
C1 = 'abc1234'
TS7 = '0123abc'
ACTIONS = {
    'S1-build-tp4-g1': 'status build',
    'S2a-reset-probe-prefix-quad': 'status reset probe prefix',
    'S2b-replay-quad': 'status replay',
    'S3-unserve-quad': 'status platform unserve reset',
    'S4-reserve-reset-quad': 'status reset',
    'V0-status-quad': 'status',
    'V1-verify-platform-quad': 'status platform',
}


def read(name):
    with open(os.path.join(TEMPLATES, name + '.env'), encoding='utf-8') as handle:
        text = handle.read()
    text = text.replace('tp4-g1-C1', 'tp4-g1-' + C1).replace(':TS7', ':' + TS7)
    return text, job.read_job(job.parse_env(text), job.profile_names())


class TemplateTests(unittest.TestCase):
    def test_every_template_is_listed_here_and_parses(self):
        names = sorted(os.path.basename(path)[:-len('.env')] for path in glob.glob(os.path.join(TEMPLATES, '*.env')))
        self.assertEqual(names, sorted(ACTIONS))
        for name, actions in ACTIONS.items():
            with self.subTest(template=name):
                text, outputs = read(name)
                self.assertEqual(outputs['actions'], actions)
                self.assertEqual(outputs['cards'], 'quad')
                self.assertEqual(outputs['tag'], 'tp4-g1-' + C1)
                self.assertEqual(outputs['profile'], 'general-prefix-tp4')
                self.assertNotIn('C1', job.parse_env(text).get('C2_IMAGE_TAG'))

    def test_the_replay_boots_the_image_default_and_expects_it(self):
        _, outputs = read('S2b-replay-quad')
        self.assertEqual(outputs['replay_profile'], '', 'the agent forwards no profile: the replay boots the default')
        self.assertEqual(outputs['replay_expect_profile'], job.profile_default())
        self.assertEqual(outputs['replay_expect_profile'], 'general-prefix-tp4')
        self.assertEqual(outputs['platform_image'], 'zot.thatch.local:5000/thatch-serving-tt:' + TS7)

    def test_the_prefix_gate_compares_the_tp4_profile_with_its_tp4_baseline(self):
        _, outputs = read('S2a-reset-probe-prefix-quad')
        self.assertEqual((outputs['prefix_profile'], outputs['prefix_baseline']), ('general-prefix-tp4', 'general-tp4'))

    def test_the_unserve_and_the_reserve_reset_run_no_serving_step(self):
        for name in ('S3-unserve-quad', 'S4-reserve-reset-quad', 'V0-status-quad', 'V1-verify-platform-quad'):
            _, outputs = read(name)
            self.assertFalse(set(outputs['actions'].split()) & set(('smoke', 'gate', 'prefix', 'replay', 'build', 'push')), name)

    def test_the_public_repo_carries_no_digest_host_or_serial(self):
        for path in glob.glob(os.path.join(TEMPLATES, '*')):
            with open(path, encoding='utf-8') as handle:
                text = handle.read()
            with self.subTest(file=os.path.basename(path)):
                self.assertIsNone(re.search(r'sha256:[0-9a-f]{8}', text))
                self.assertIsNone(re.search(r'blackhole-[0-9A-F]{8}', text))
                self.assertNotIn('\r', text)

    def test_the_image_default_is_the_four_card_profile(self):
        self.assertEqual(job.profile_default(), 'general-prefix-tp4')
        self.assertEqual(job.profile_meshes()[job.profile_default()], job.TP4_MESH_DEVICE)


if __name__ == '__main__':
    unittest.main()
