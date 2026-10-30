"""The tp4/stream-1 card jobs (scripts/ci/references/tp4-stream-1-jobs): templates, order, the tau pairs that read the lookup on the production profile and checkpoint.

The window tau-tests and speed-reads the prompt lookup on the ship-5 profile with the production checkpoint (engine reuse, octo, the lone-user round) in the tau lab, as pairs of a control arm
and the lookup arm over the same turns. The cards are under development: no job stops or starts the node agent, hands back or deploys. The templates are public, so they name no rig, card,
address, registry or digest."""

import json
import os
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import c2_tau_lab as lab  # noqa: E402

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'tp4-stream-1-jobs'
with open(HERE / 'qwen_c2_profiles.json', encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)['profiles']
NAMES = sorted(PROFILES)
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@|[A-Za-z]:[\\/]Users')
IMAGE = 'tp4-stream-1'
S5 = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s5'
SWIFT = ('ukisai/Swift-1.5-Qwen3.8-27b', 'b4c84d42903a8646b25857eb2827288d92ed85a4')
BUILD, STATUS, RESET = 'B0-build-s5', 'X0-status-rescan-reset', 'Z-reset'
TAU = ('TL0-tau-control-4seat', 'TL1-tau-lookup-4seat', 'TL2-tau-control-lone', 'TL3-tau-lookup-lone')
ORDERED = (BUILD, STATUS) + TAU + (RESET,)
ARM_OF = {TAU[0]: 'control', TAU[1]: 'lookup', TAU[2]: 'control', TAU[3]: 'lookup'}
IN_FLIGHT = {TAU[0]: '8', TAU[1]: '8', TAU[2]: '1', TAU[3]: '1'}
FORBIDDEN_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'push', 'replay', 'prefix', 'platform', 'smoke'}


def read_order():
    text = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
    return [line.split() for line in text.splitlines() if line.strip() and not line.startswith('#')]


def text_of(name):
    return (FOLDER / (name + '.env')).read_text(encoding='utf-8')


def parsed(name, pair='1'):
    text = text_of(name)
    if ARM_OF.get(name) == 'lookup' and pair:
        text += 'C2_TAULAB_PAIR_CONTROL=%s\n' % pair
    return job.read_job(job.parse_env(text), NAMES)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        self.assertEqual([row[0] for row in rows], list(ORDERED))
        self.assertEqual(sorted(row[0] for row in rows), sorted(path.stem for path in FOLDER.glob('*.env')))

    def test_modes_images_and_minutes(self):
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertEqual(mode, 'stop' if name in (BUILD, STATUS) else 'soft')
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 5 <= int(minutes) <= 60, minutes)
        text = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        total = sum(int(row[3]) for row in read_order())
        self.assertIn('%d minutes in all' % total, text)
        self.assertIn('%d of cards' % (total - int(read_order()[0][3])), text)

    def test_the_needs_lines_name_templates_and_form_one_chain(self):
        text = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        needs = [line[len('# NEEDS '):].split(' <- ') for line in text.splitlines() if line.startswith('# NEEDS ')]
        chain = [(left.split(), right.split()) for left, right in needs]
        for left, right in chain:
            for name in left + right:
                self.assertIn(name, ORDERED)
        self.assertEqual([left[0] for left, _ in chain], list(ORDERED[1:-1]))
        for index, (left, right) in enumerate(chain):
            self.assertEqual(right, [ORDERED[index]], 'each job needs the one before it')

    def test_the_arc_pause_names_the_timed_jobs(self):
        text = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        self.assertIn('# ARC-ZERO ' + ' '.join(name.split('-')[0] for name in TAU), text)


class TemplateTests(unittest.TestCase):
    def test_each_template_parses_through_the_job_parser_on_four_cards(self):
        for name in ORDERED:
            with self.subTest(job=name):
                self.assertEqual(parsed(name)['cards'], 'quad')

    def test_no_job_touches_the_agent_hands_back_deploys_or_smokes(self):
        for name in ORDERED:
            actions = set(parsed(name)['actions'].split())
            self.assertFalse(FORBIDDEN_ACTIONS & actions, name)
        self.assertEqual(parsed(BUILD)['actions'], 'build')
        self.assertEqual(parsed(STATUS)['actions'], 'status rescan reset')
        self.assertEqual(parsed(RESET)['actions'], 'status reset')
        for name in TAU:
            self.assertEqual(parsed(name)['actions'], 'reset taulab')

    def test_the_build_bakes_the_ship_5_default_with_the_graft(self):
        values = job.parse_env(text_of(BUILD))
        self.assertEqual(values['C2_BAKE_DEFAULT_PROFILE'], S5)
        self.assertEqual(values['C2_BAKE_GRAFT'], 'K64j-OQ')
        self.assertIsNone(PROFILES[S5].get('gate_only'))
        self.assertTrue(PROFILES[S5]['owner_traffic_waiver']['decision'].startswith('APPROVED'))

    def test_the_tau_jobs_run_the_production_profile_and_checkpoint(self):
        for name in TAU:
            with self.subTest(job=name):
                outputs = parsed(name)
                self.assertEqual(outputs['taulab_profile'], S5)
                self.assertEqual(outputs['taulab_drafter_arm'], ARM_OF[name])
                self.assertEqual(outputs['taulab_arms'], 'A2', 'the own-session set, thinking on; no A3 (its reference is an older stack\'s)')
                self.assertEqual(outputs['taulab_in_flight'], IN_FLIGHT[name])
                self.assertEqual((outputs['model_repo'], outputs['model_revision']), SWIFT)
        self.assertEqual(parsed(TAU[0])['taulab_pair_control'], '')
        self.assertEqual(parsed(TAU[1])['taulab_pair_control'], '1')

    def test_the_deadlines_fit_the_minutes_the_order_gives_each_job(self):
        minutes = dict((row[0], int(row[3])) for row in read_order())
        for name in TAU:
            deadline = int(parsed(name)['taulab_deadline'])
            self.assertLessEqual(deadline + 8, minutes[name], '%s: the deadline counts from the container start; the reset and the stop take the rest' % name)
            self.assertGreaterEqual(deadline, 30)

    def test_a_candidate_arm_is_refused_without_its_control_run(self):
        for name in (TAU[1], TAU[3]):
            with self.subTest(job=name):
                with self.assertRaisesRegex(job.JobError, 'C2_TAULAB_PAIR_CONTROL'):
                    parsed(name, pair='')
                self.assertIn('C2_TAULAB_PAIR_CONTROL', text_of(name))

    def test_the_lookup_arm_is_the_one_the_lab_defines(self):
        self.assertEqual(dict(lab.DRAFTER_ARMS['lookup']['env']),
                         {'QWEN_FAST_LOOKUP_DRAFT': 'n3m12', 'QWEN_FAST_LOOKUP_SHADOW': '1'})
        self.assertEqual(lab.DRAFTER_ARMS['lookup']['manifest'], lab.DRAFTER_ARMS['control']['manifest'])
        # the lab derives the arm on the production profile (it did not on a profile with engine reuse before tp4/stream-1)
        name, document = lab.derive_profile({'profiles': PROFILES}, S5, 'lookup')
        env = document['profiles'][name]['env']
        self.assertEqual(env['QWEN_FAST_PARKED_ENGINES'], '1')
        self.assertEqual(env['QWEN_FAST_LOOKUP_DRAFT'], 'n3m12')
        self.assertEqual(lab.arithmetic_diff({'profiles': PROFILES}, S5, document, 'lookup'), [])

    def test_every_template_shares_the_image_and_the_header(self):
        for name in ORDERED:
            text = text_of(name)
            self.assertIn('\nC2_IMAGE_TAG=%s\n' % IMAGE, text, name)
            self.assertEqual(sorted(set(re.findall(r'tp4-[a-z]+-\d+', text))), [IMAGE], name)
            self.assertIn('ONE TAG AT A TIME', text)
            self.assertIn('never cancel a running card job', text)

    def test_the_pack_is_public(self):
        for path in FOLDER.iterdir():
            text = path.read_text(encoding='utf-8').replace(SWIFT[1], '<the checkpoint revision>')      # the public model revision the s5 window's templates carry too
            self.assertIsNone(BANNED.search(text), (path.name, BANNED.search(text)))


class WorkflowTests(unittest.TestCase):
    def test_the_new_suites_run_in_the_cpu_workflow(self):
        workflow = (HERE.parent.parent / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        for name in ('test_tp4_stream_1_jobs', 'test_lookup_reuse', 'test_tree_verify', 'test_tree_gain_report'):
            self.assertIn(name, workflow)


if __name__ == '__main__':
    unittest.main()
