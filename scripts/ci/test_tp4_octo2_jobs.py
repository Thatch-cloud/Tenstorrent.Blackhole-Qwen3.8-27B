"""The tp4/octo-2 pack (scripts/ci/references/tp4-octo2-jobs): one image, one window for the three octo levers behind the octo-T8 timed arm.

Holds: ORDER.txt lists exactly the pack's templates, each once, in the asked order (the draft first, then the glue levers, then the bundle, then all three, then the repeat pair), with one image
tag, positive minutes and a total that matches its header; every template parses through the job reader (which holds the card-tooling rule: a four-card smoke of a profile that asks for the
eight-seat levers names concurrent8_steady); B0 builds the image with the profile production runs as its baked default; each job selects the profile its lever's generator makes; the control and
the arms run the same four tests (the audited jobs stay at nine requests); the repeat pair is an ABAB of the first; the NEEDS lines name known jobs that run earlier; the READ rules name
the tools that exist; and the pack names no host, address, registry or home path (the repo is public).
"""

import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import make_octo2_profiles as twins  # noqa: E402
import profile_twins  # noqa: E402

FOLDER = HERE / 'references' / 'tp4-octo2-jobs'
PROFILES = HERE / 'qwen_c2_profiles.json'
IMAGE = 'tp4-octo2-1'
SHIP = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'
TESTS = 'warmup,concurrent8_steady,concurrent8_code_equal,concurrent8_code_32k'

EXPECTED = (
    ('B0-build', 'stop', 40), ('X0-status-rescan-reset', 'stop', 20),
    ('OC1-octo-control-timed', 'soft', 55), ('OD1-octo-draft-timed', 'soft', 60),
    ('OGA-octo-glue8-audit-attach-lean', 'soft', 90), ('OG1-octo-glue8-timed', 'soft', 60),
    ('OBA-octo-bundle-audit', 'soft', 45), ('OB1-octo-bundle-timed', 'soft', 60),
    ('OL1-octo-levers-timed', 'soft', 60),
    ('OC2-octo-control-repeat', 'soft', 55), ('OD2-octo-draft-repeat', 'soft', 60), ('Z-reset', 'soft', 10),
)
TOTAL = 615
PROFILE_OF = {
    'OC1-octo-control-timed': twins.PARENT, 'OC2-octo-control-repeat': twins.PARENT,
    'OD1-octo-draft-timed': twins.BASE + '-octo-draft', 'OD2-octo-draft-repeat': twins.BASE + '-octo-draft',
    'OGA-octo-glue8-audit-attach-lean': twins.BASE + '-octo-glue8-audit', 'OG1-octo-glue8-timed': twins.BASE + '-octo-glue8',
    'OBA-octo-bundle-audit': twins.BASE + '-octo-bundle-audit', 'OB1-octo-bundle-timed': twins.BASE + '-octo-bundle',
    'OL1-octo-levers-timed': twins.BASE + '-octo-levers',
}


def order_text():
    return (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')


def order_lines():
    return [line.split() for line in order_text().splitlines() if line.strip() and not line.startswith('#')]


def values(name):
    return job.parse_env((FOLDER / (name + '.env')).read_text(encoding='utf-8'))


def short(name):
    return name.split('-', 1)[0]


class OrderTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates_in_the_asked_order(self):
        lines = order_lines()
        self.assertEqual([line[0] for line in lines], [name for name, _mode, _minutes in EXPECTED])
        self.assertEqual(sorted(path.stem for path in FOLDER.glob('*.env')), sorted(name for name, _mode, _minutes in EXPECTED))
        for (name, mode, image, minutes), (_expected, expected_mode, expected_minutes) in zip(lines, EXPECTED):
            self.assertEqual((mode, image, int(minutes)), (expected_mode, IMAGE, expected_minutes), name)

    def test_the_order_is_the_priority(self):
        names = [short(line[0]) for line in order_lines()]
        self.assertEqual(names[:2], ['B0', 'X0'])
        self.assertEqual(names[2:4], ['OC1', 'OD1'], 'the control, then Lever 1 (the first priority)')
        self.assertEqual(names[4:6], ['OGA', 'OG1'], 'then Lever 2: the audited attach first')
        self.assertEqual(names[6:8], ['OBA', 'OB1'], 'then Lever 3: the audited bundle first')
        self.assertEqual(names[8], 'OL1')
        self.assertEqual(names[9:11], ['OC2', 'OD2'], 'the repeat pair is the ABAB of the first')
        self.assertEqual(names[-1], 'Z')

    def test_the_total_in_the_header_is_the_sum(self):
        total = sum(int(line[3]) for line in order_lines())
        self.assertEqual(total, TOTAL)
        self.assertIn('%d minutes (%d h %d min)' % (total, total // 60, total % 60), order_text())
        self.assertIn('%d minutes without the repeat pair' % (total - 115), order_text())

    def test_the_needs_lines_name_known_jobs_that_run_earlier(self):
        position = dict((short(line[0]), index) for index, line in enumerate(order_lines()))
        needs = re.findall(r'^# NEEDS (.+?) <- (.+)$', order_text(), re.M)
        self.assertGreater(len(needs), 6)
        waiting = set()
        for left, right in needs:
            for name in left.split():
                self.assertIn(name, position, name)
                waiting.add(name)
                for needed in right.split():
                    self.assertIn(needed, position, needed)
                    self.assertLess(position[needed], position[name], '%s needs %s, which runs later' % (name, needed))
        for name in ('X0', 'OC1', 'OD1', 'OGA', 'OG1', 'OBA', 'OB1', 'OL1', 'OC2', 'OD2'):
            self.assertIn(name, waiting, name)

    def test_only_the_build_and_the_card_state_stop_the_window(self):
        self.assertEqual([line[0] for line in order_lines() if line[1] == 'stop'], ['B0-build', 'X0-status-rescan-reset'])

    def test_the_read_rules_name_tools_that_exist(self):
        for tool in ('octo2_report.py', 'octo_compare.py', 'octo_judge.py', 'c2_smoke_check'):
            self.assertIn(tool.split('.')[0], order_text() + ''.join(path.read_text(encoding='utf-8') for path in FOLDER.glob('*.env')), tool)
        for tool in ('octo2_report.py', 'octo_compare.py'):
            self.assertTrue((HERE / tool).exists(), tool)


class JobTests(unittest.TestCase):
    def test_every_template_passes_the_job_reader(self):
        data = json.loads(PROFILES.read_text(encoding='utf-8'))
        profiles = job.profile_names(str(PROFILES))
        envs = job.profile_envs(str(PROFILES))
        for name, _mode, _minutes in EXPECTED:
            with self.subTest(name):
                outputs = job.read_job(values(name), profiles, envs=envs)
                self.assertTrue(outputs['actions'])
        self.assertIn(SHIP, data['profiles'])

    def test_b0_builds_one_image_and_bakes_the_profile_production_runs(self):
        b0 = values('B0-build')
        self.assertEqual((b0['C2_ACTIONS'], b0['C2_PROFILE'], b0['C2_IMAGE_TAG']), ('build', 'c2-packed-tp4', IMAGE))
        self.assertEqual(b0['C2_BAKE_DEFAULT_PROFILE'], SHIP)
        window = job.parse_env((HERE / 'references' / 'tp4-window-next' / 'B0-build.env').read_text(encoding='utf-8'))
        self.assertEqual(b0['C2_BAKE_DEFAULT_PROFILE'], window['C2_BAKE_DEFAULT_PROFILE'])
        self.assertEqual([name for name, *_ in EXPECTED if values(name).get('C2_ACTIONS') == 'build'], ['B0-build'])

    def test_every_job_serves_the_one_image(self):
        for name, _mode, _minutes in EXPECTED:
            self.assertEqual(values(name).get('C2_IMAGE_TAG'), IMAGE, name)

    def test_each_job_selects_the_profile_its_generator_makes(self):
        data = json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']
        for name, profile in PROFILE_OF.items():
            with self.subTest(name):
                self.assertEqual(values(name)['C2_PROFILE'], profile)
                self.assertIn(profile, data)
        generated = set(twins.twin_names()) | {twins.PARENT}
        self.assertTrue(set(twins.twin_names()) <= set(profile_twins.twin_names()))
        self.assertTrue(set(PROFILE_OF.values()) <= generated)
        # every twin the generator makes has its job
        self.assertEqual(set(PROFILE_OF.values()), generated)

    def test_the_timed_jobs_run_the_four_tests_and_the_audited_ones_stay_at_nine_requests(self):
        for name, profile in PROFILE_OF.items():
            audited = name.startswith(('OGA', 'OBA'))
            self.assertEqual(values(name)['C2_SMOKE_TESTS'], 'warmup,concurrent8_steady' if audited else TESTS, name)
            self.assertIn('concurrent8_steady', values(name)['C2_SMOKE_TESTS'].split(','))
        self.assertIn('8.4 minutes', (FOLDER / 'OGA-octo-glue8-audit-attach-lean.env').read_text(encoding='utf-8'))

    def test_the_repeat_pair_is_an_abab_of_the_first(self):
        self.assertEqual(values('OC1-octo-control-timed')['C2_PROFILE'], values('OC2-octo-control-repeat')['C2_PROFILE'])
        self.assertEqual(values('OD1-octo-draft-timed')['C2_PROFILE'], values('OD2-octo-draft-repeat')['C2_PROFILE'])
        self.assertNotEqual(values('OC1-octo-control-timed')['C2_PROFILE'], values('OD1-octo-draft-timed')['C2_PROFILE'])
        for control, arm in (('OC1-octo-control-timed', 'OD1-octo-draft-timed'), ('OC2-octo-control-repeat', 'OD2-octo-draft-repeat')):
            self.assertEqual(values(control)['C2_SMOKE_TESTS'], values(arm)['C2_SMOKE_TESTS'])

    def test_no_job_starts_the_node_agent_or_pushes_a_tag(self):
        for name, _mode, _minutes in EXPECTED:
            actions = values(name)['C2_ACTIONS'].split()
            self.assertFalse({'agent', 'start', 'serve', 'deploy'} & set(actions), name)


class PublicTests(unittest.TestCase):
    def test_the_pack_names_no_host_address_registry_or_home_path(self):
        pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|/home/|/Users/|[A-Za-z]:[\\/]|\.local\b|\.lan\b|\bssh\b|ghcr\.io|docker\.io|sha256:[0-9a-f]{12}|spark-|thatch@')
        for path in sorted(FOLDER.iterdir()):
            text = path.read_text(encoding='utf-8')
            with self.subTest(path.name):
                self.assertIsNone(pattern.search(text), pattern.search(text) and pattern.search(text).group(0))
                self.assertNotIn('\r', text)


if __name__ == '__main__':
    unittest.main()
