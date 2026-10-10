"""The tp4/next-2 integrated window pack (references/tp4-next2-1-jobs) on the CPU: every template parses with the job parser (D-T5 once its control run id placeholder is a number), the ORDER rows match the files and
the one image tag, the dependency lines name real jobs, each job is the source pack's job re-pointed at the integrated image (same profile and tests as its source), and the same-image control D-T1 comes before D-T5.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_next2_1_jobs` from scripts/ci."""

import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402

PACK = HERE / 'references' / 'tp4-next2-1-jobs'
REFERENCES = HERE / 'references'
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
TAG = 'tp4-next2-1'
PLACEHOLDER = '@CONTROL_RUN@'
SOURCES = {
    'X0-status-rescan-reset': 'tp4-w2-kill-jobs', 'K-W2a-control': 'tp4-w2-kill-jobs', 'K-W2b-kill-switch': 'tp4-w2-kill-jobs', 'Z-reset': 'tp4-w2-kill-jobs',
    'Q1-kvread-qualify': 'tp4-kvread-next-jobs', 'Q2-kvread-cross-levern-audit': 'tp4-kvread-next-jobs', 'Q3-kvread-cross-prefix-audit': 'tp4-kvread-next-jobs',
    'A0R-control-audit-region': 'tp4-kvread-next-jobs', 'A1R-parked-audit-region': 'tp4-kvread-next-jobs',
    'D-T1-control': 'tp4-drafter-jobs', 'D-T5-lookup': 'tp4-drafter-jobs',
}


def text(name, folder=PACK):
    return (folder / (name + '.env')).read_text(encoding='utf-8')


def parsed(name, folder=PACK, fill=True):
    body = text(name, folder)
    if fill:
        body = body.replace(PLACEHOLDER, '1234567890')
    return job.read_job(job.parse_env(body), sorted(PROFILES), root=ROOT)


def order():
    return [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


def needs():
    rows = []
    for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines():
        match = re.match(r'# NEEDS ([A-Za-z0-9 -]+) <- ([A-Za-z0-9 -]+)$', line)
        if match:
            rows.append((match.group(1).split(), match.group(2).split()))
    return rows


class PackTests(unittest.TestCase):
    def test_every_template_has_an_order_row_one_tag_and_parses_on_the_quad(self):
        names = sorted(path.name[:-4] for path in PACK.glob('*.env'))
        self.assertEqual(names, sorted(row[0] for row in order()))
        for row in order():
            with self.subTest(row=row[0]):
                self.assertEqual(len(row), 4)
                self.assertIn(row[1], ('stop', 'soft'))
                self.assertEqual(row[2], TAG)
                self.assertTrue(row[3].isdigit())
                outputs = parsed(row[0])
                self.assertEqual((outputs['cards'], outputs['tag']), ('quad', TAG))

    def test_the_dependency_lines_name_real_jobs_in_order(self):
        names = [row[0] for row in order()]
        short = {re.match(r'D-T\d+|K-W2[ab]|[A-Z]\d*R?', name).group(0): name for name in names}
        for left, right in needs():
            for key in left + right:
                self.assertIn(key, short, key)
            for key in left:
                for need in right:
                    self.assertLess(names.index(short[need]), names.index(short[key]), '%s needs %s, which must come first' % (key, need))
        graph = {key: set(right) for left, right in needs() for key in left}
        self.assertEqual(graph['D-T5'], {'D-T1'})
        self.assertEqual(graph['K-W2b'], {'K-W2a'})
        self.assertEqual(graph['A0R'], {'Q2'})

    def test_b0_builds_the_window_image_with_the_ship_profile_baked(self):
        outputs = parsed('B0-build')
        self.assertEqual((outputs['actions'], outputs['profile'], outputs['bake_default_profile']),
                         ('build', 'c2-packed-tp4', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'))
        self.assertEqual(order()[0][:2], ['B0-build', 'stop'])

    def test_each_job_is_its_source_job_on_the_integrated_image(self):
        for name, pack in SOURCES.items():
            with self.subTest(name=name):
                source = text(name, REFERENCES / pack)
                body = text(name)
                values = job.parse_env(body)
                self.assertEqual(values['C2_IMAGE_TAG'], TAG)
                old = job.parse_env(source)
                for key, value in old.items():
                    if key not in ('C2_IMAGE_TAG', 'C2_TAULAB_PAIR_CONTROL'):
                        self.assertEqual(values.get(key), value, key)
                self.assertEqual(set(values) - set(old), {'C2_TAULAB_PAIR_CONTROL'} if name == 'D-T5-lookup' else set())

    def test_d_t5_waits_for_a_same_image_control_and_refuses_the_placeholder(self):
        values = job.parse_env(text('D-T5-lookup'))
        self.assertEqual(values['C2_TAULAB_PAIR_CONTROL'], PLACEHOLDER)
        with self.assertRaises(job.JobError):
            parsed('D-T5-lookup', fill=False)
        self.assertEqual(parsed('D-T5-lookup')['taulab_pair_control'], '1234567890')
        names = [row[0] for row in order()]
        self.assertLess(names.index('D-T1-control'), names.index('D-T5-lookup'))
        control = parsed('D-T1-control')
        self.assertEqual((control['taulab_drafter_arm'], control['tag']), ('control', TAG))
        self.assertEqual(parsed('D-T5-lookup')['taulab_drafter_arm'], 'lookup')
        self.assertIn(PLACEHOLDER, (PACK / 'ORDER.txt').read_text(encoding='utf-8'))

    def test_no_template_names_a_host_an_address_or_a_home_path(self):
        pattern = re.compile(r'(/home/|/Users/|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.local\b|zot\.|sha256:[0-9a-f]{16}|ghp_|token=)')
        for path in sorted(PACK.iterdir()):
            with self.subTest(path=path.name):
                self.assertIsNone(pattern.search(path.read_text(encoding='utf-8')))


if __name__ == '__main__':
    unittest.main()
