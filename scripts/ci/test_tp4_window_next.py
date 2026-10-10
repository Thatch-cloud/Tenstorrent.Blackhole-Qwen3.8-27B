"""The combined tp4/window-next pack (scripts/ci/references/tp4-window-next): one image, one window for octo-T8, the round-host levers and the prefix gate rerun.

Holds: ORDER.txt lists exactly the pack's templates, each once, in the asked order, with one image tag, positive minutes and a total that matches its header;
every template parses through the job reader (which holds the card-tooling rule: a four-card smoke of a profile that asks for the eight-seat levers names
concurrent8_steady); B0 builds the image with the profile production runs as its baked default (the ship profile, whose owner waiver is approved); the jobs
select profiles that exist, and the octo and round-host twins only ever on the profiles their generators make; the audited octo job stays at nine requests; the
NEEDS lines name known jobs that run earlier; and the pack names no host, address, registry or home path (the repo is public).
"""

import json
import os
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import make_octo_profiles  # noqa: E402
import make_round_host_profiles  # noqa: E402
import profile_twins  # noqa: E402

FOLDER = HERE / 'references' / 'tp4-window-next'
PROFILES = HERE / 'qwen_c2_profiles.json'
IMAGE = 'tp4-next-1'
SHIP = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'

# name -> (mode, minutes), in the order the window runs them
EXPECTED = (
    ('B0-build', 'stop', 40), ('X0-status-rescan-reset', 'stop', 20),
    ('Q1w-k64j-g8b1-watcher', 'soft', 10), ('Q1a-k64j-g8b1-cb1', 'soft', 10), ('Q1b-k64j-g8b1-cb2a', 'soft', 15),
    ('Q2w-gdn-eight-users-watcher', 'soft', 10), ('Q2-gdn-eight-users', 'soft', 15),
    ('O1-octo-audit-attach-lean', 'soft', 90), ('O2-octo-control-timed', 'soft', 55), ('O3-octo-alternate-timed', 'soft', 60),
    ('H1-roundhost-audit-attach-lean', 'soft', 50), ('H2-roundhost-A1-control', 'soft', 55), ('H3-roundhost-B1-arm', 'soft', 55),
    ('H4-roundhost-A2-control', 'soft', 55), ('H5-roundhost-B2-arm', 'soft', 55),
    ('G2-prefix-hit', 'soft', 35), ('O4-octo-alternate-repeat', 'soft', 60), ('Z-reset', 'soft', 10),
)
TOTAL = 700


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

    def test_the_order_is_the_asked_one(self):
        names = [short(line[0]) for line in order_lines()]
        # one build, the shape killer first of the card jobs that matter, the octo block, the round-host block, then the gate rerun
        self.assertEqual(names[:2], ['B0', 'X0'])
        self.assertEqual(names[2:7], ['Q1w', 'Q1a', 'Q1b', 'Q2w', 'Q2'])
        self.assertEqual(names[7:10], ['O1', 'O2', 'O3'])
        self.assertEqual(names[10:15], ['H1', 'H2', 'H3', 'H4', 'H5'])
        self.assertEqual(names[15], 'G2')
        self.assertEqual(names[-1], 'Z')

    def test_the_total_in_the_header_is_the_sum(self):
        total = sum(int(line[3]) for line in order_lines())
        self.assertEqual(total, TOTAL)
        self.assertIn('%d minutes (%d h %d min)' % (total, total // 60, total % 60), order_text())
        self.assertIn('%d minutes without O4' % (total - 60), order_text())

    def test_the_needs_lines_name_known_jobs_that_run_earlier(self):
        position = dict((short(line[0]), index) for index, line in enumerate(order_lines()))
        needs = re.findall(r'^# NEEDS (.+?) <- (.+)$', order_text(), re.M)
        self.assertGreater(len(needs), 10)
        waiting = set()
        for left, right in needs:
            for name in left.split():
                self.assertIn(name, position, name)
                waiting.add(name)
                for needed in right.split():
                    self.assertIn(needed, position, needed)
                    self.assertLess(position[needed], position[name], '%s needs %s, which runs later' % (name, needed))
        # a lever's follow-ups wait for its kill signal; only the build and the card state gate the rest
        for name in ('Q1a', 'Q1b', 'Q2', 'O1', 'O2', 'O3', 'O4', 'H1', 'H2', 'H3', 'H4', 'H5', 'G2'):
            self.assertIn(name, waiting, name)

    def test_only_the_build_and_the_card_state_stop_the_window(self):
        self.assertEqual([line[0] for line in order_lines() if line[1] == 'stop'], ['B0-build', 'X0-status-rescan-reset'])


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
        ship = values_of_pack('tp4-ship-ln-w2-er-jobs', 'B0-build')
        self.assertEqual(b0['C2_BAKE_DEFAULT_PROFILE'], ship['C2_BAKE_DEFAULT_PROFILE'])
        self.assertEqual([name for name, *_ in EXPECTED if values(name).get('C2_ACTIONS') == 'build'], ['B0-build'])

    def test_every_job_serves_the_one_image(self):
        for name, _mode, _minutes in EXPECTED:
            self.assertEqual(values(name).get('C2_IMAGE_TAG'), IMAGE, name)

    def test_the_profiles_the_jobs_select_exist_and_the_twins_are_the_generators(self):
        data = json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']
        selected = {}
        for name, _mode, _minutes in EXPECTED:
            items = values(name)
            for key in ('C2_PROFILE', 'C2_PREFIX_PROFILE'):
                if items.get(key):
                    selected[name] = items[key]
                    self.assertIn(items[key], data, name)
        octo, round_host = set(make_octo_profiles.twin_names()), set(make_round_host_profiles.twin_names())
        self.assertTrue(octo <= set(profile_twins.twin_names()) and round_host <= set(profile_twins.twin_names()))
        for name in ('O1-octo-audit-attach-lean', 'O3-octo-alternate-timed', 'O4-octo-alternate-repeat'):
            self.assertIn(selected[name], octo, name)
        for name in ('H1-roundhost-audit-attach-lean', 'H3-roundhost-B1-arm', 'H5-roundhost-B2-arm'):
            self.assertIn(selected[name], round_host, name)
        for name in ('H2-roundhost-A1-control', 'H4-roundhost-A2-control'):
            self.assertIn(selected[name], round_host, name)
        # the O2 control is the flag-off parent of the octo twins, and G2 gates the profile production runs
        self.assertEqual(selected['O2-octo-control-timed'], make_octo_profiles.PARENT)
        self.assertEqual(selected['G2-prefix-hit'], SHIP)
        # no job runs the two levers together: no selected profile carries both
        for name, profile in selected.items():
            env = data[profile].get('env') or {}
            self.assertFalse(any(key.startswith('QWEN_FAST_TP4_ROUND_HOST_') for key in env) and 'QWEN_FAST_OCTO' in env, name)

    def test_the_audited_octo_attach_stays_at_nine_requests(self):
        o1 = values('O1-octo-audit-attach-lean')
        self.assertEqual(o1['C2_SMOKE_TESTS'], 'warmup,concurrent8_steady')
        self.assertIn('8.4 minutes', (FOLDER / 'O1-octo-audit-attach-lean.env').read_text(encoding='utf-8'))

    def test_the_round_host_block_is_an_abab_of_one_test_list(self):
        tests = [values(name)['C2_SMOKE_TESTS'] for name in ('H2-roundhost-A1-control', 'H3-roundhost-B1-arm', 'H4-roundhost-A2-control', 'H5-roundhost-B2-arm')]
        self.assertEqual(len(set(tests)), 1)
        profiles = [values(name)['C2_PROFILE'] for name in ('H2-roundhost-A1-control', 'H3-roundhost-B1-arm', 'H4-roundhost-A2-control', 'H5-roundhost-B2-arm')]
        self.assertEqual(profiles[0], profiles[2])
        self.assertEqual(profiles[1], profiles[3])
        self.assertNotEqual(profiles[0], profiles[1])
        for needed in ('concurrent8_steady', 'concurrent8_code_32k', 'concurrent8_code_equal'):
            self.assertIn(needed, tests[0].split(','))

    def test_the_q_jobs_are_one_card_evidence_jobs_and_the_gate_reruns_the_levern_plan(self):
        for name in ('Q1w-k64j-g8b1-watcher', 'Q1a-k64j-g8b1-cb1', 'Q1b-k64j-g8b1-cb2a', 'Q2w-gdn-eight-users-watcher', 'Q2-gdn-eight-users'):
            items = values(name)
            self.assertEqual((items['C2_CARDS'], items['C2_ACTIONS']), ('pair', 'cardm'), name)
            self.assertIn('K64J_HARNESS=', items['C2_CARDM_ENV'], name)
        self.assertIn('--octo', values('Q1w-k64j-g8b1-watcher')['C2_CARDM_ARGS'])
        gate = values('G2-prefix-hit')
        self.assertEqual((gate['C2_ACTIONS'], gate['C2_PREFIX_PLAN'], gate['C2_PREFIX_BASELINE']), ('reset prefix', 'levern-hit', 'none'))


class PublicTests(unittest.TestCase):
    def test_the_pack_names_no_host_address_registry_or_home_path(self):
        pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|/home/|/Users/|[A-Za-z]:[\\/]|\.local\b|\.lan\b|\bssh\b|ghcr\.io|docker\.io|sha256:[0-9a-f]{12}|spark-|thatch@')
        for path in sorted(FOLDER.iterdir()):
            text = path.read_text(encoding='utf-8')
            with self.subTest(path.name):
                self.assertIsNone(pattern.search(text), pattern.search(text) and pattern.search(text).group(0))
                self.assertNotIn('\r', text)


def values_of_pack(pack, name):
    return job.parse_env((HERE / 'references' / pack / (name + '.env')).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
