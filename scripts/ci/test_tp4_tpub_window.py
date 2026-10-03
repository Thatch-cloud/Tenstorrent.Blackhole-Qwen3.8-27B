"""The tp4/tpub window: its job templates (scripts/ci/references/tp4-tpub-jobs), their order, the two profiles they add, the design document and the CI allowlist.

The window times the traced publication of the sequential (lone-user) step at four cards: the per-step GDN carry save and restore as captured traces. A0 takes production down
(the first production job of a window is `agentstop unserve`), S1 is the audited smoke, S2a..S2c three audits-off runs of the hang shapes on the timed arm, S3a..S3d the paired timing
ABAB against c2-packed-tp4-best-strace, S4 the hand-back (reset, fabric re-measure, node agent started; the deploy is the owner's and is in no template). The templates are public, so they name
no rig, card, address, registry or digest."""

import json
import os
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FOLDER = HERE / 'references' / 'tp4-tpub-jobs'
with open(HERE / 'qwen_c2_profiles.json', encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)['profiles']
NAMES = sorted(PROFILES)
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@|[A-Za-z]:[\\/]Users')
IMAGE = 'tp4-tpub-1'
CONTROL, ARM, AUDITED, PRODUCTION = ('c2-packed-tp4-best-strace', 'c2-packed-tp4-best-strace-tpub', 'c2-packed-tp4-best-gate-tpub',
                                     'c2-packed-tp4')
FLAG, AUDIT = 'QWEN_FAST_TP4_TRACED_PUBLISH', 'QWEN_FAST_TP4_TRACED_PUBLISH_AUDIT'
BUILD, STOP_DOWN, SMOKE, HAND_BACK = 'B0-build', 'A0-agentstop', 'S1-audited-smoke', 'S4-handback-reset'
HANG_RUNS = tuple('S2%s-hang-shapes-tpub-strace' % letter for letter in 'abc')
TIMED = ('S3a-best-timed', 'S3b-tpub-timed', 'S3c-best-timed', 'S3d-tpub-timed')
STOP_JOBS = (BUILD, STOP_DOWN, SMOKE) + HANG_RUNS
ORDERED = STOP_JOBS + TIMED + (HAND_BACK,)
PROFILE_OF = {BUILD: PRODUCTION, STOP_DOWN: PRODUCTION, SMOKE: AUDITED, **{name: ARM for name in HANG_RUNS},
              'S3a-best-timed': CONTROL, 'S3b-tpub-timed': ARM, 'S3c-best-timed': CONTROL, 'S3d-tpub-timed': ARM, HAND_BACK: PRODUCTION}
HANG_SHAPES = ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_v164order', 'concurrent4_steady', 'replay_concurrent4']
TIMING_TESTS = ['warmup', 'coding', 'long_real_text', 'concurrent4_code_equal']
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')


def read_order():
    text = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
    return [line.split() for line in text.splitlines() if line.strip() and not line.startswith('#')]


def text_of(name):
    return (FOLDER / (name + '.env')).read_text(encoding='utf-8')


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), NAMES)


def env_of(name):
    return PROFILES[name]['env']


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns_and_the_order_names_no_other(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        self.assertEqual(sorted(row[0] for row in rows), sorted(path.stem for path in FOLDER.glob('*.env')))
        self.assertEqual([row[0] for row in rows], list(ORDERED))

    def test_the_modes_images_and_minutes(self):
        modes = {row[0]: row[1] for row in read_order()}
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertIn(mode, ('stop', 'optional'))
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 15 <= int(minutes) <= 180, minutes)
        for name in STOP_JOBS + (HAND_BACK,):
            self.assertEqual(modes[name], 'stop', name)
        for name in TIMED:
            self.assertEqual(modes[name], 'optional', name)

    def test_the_build_runs_first_opens_no_card_and_a0_is_the_only_job_that_takes_production_down(self):
        self.assertEqual([row[0] for row in read_order()[:2]], [BUILD, STOP_DOWN])
        self.assertEqual(parsed(STOP_DOWN)['actions'], 'status agentstop unserve')
        for name in ORDERED:
            self.assertEqual('build' in parsed(name)['actions'].split(), name == BUILD, name)
            if name != STOP_DOWN:
                self.assertFalse({'agentstop', 'unserve'} & set(parsed(name)['actions'].split()), name)
        outputs = parsed(BUILD)
        self.assertEqual((outputs['actions'], outputs['cards'], outputs['tag']), ('build', 'quad', IMAGE))
        for word in ('BEFORE A0', 'FRESH', 'already exists', 'PLACEHOLDER', 'production still serves'):
            self.assertIn(word, text_of(BUILD))
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('B0 RUNS FIRST, BEFORE A0', 'FRESH tag', 'PRODUCTION IS LIVE ON THE CARDS', 'FIRST production job is A0', 'agentstop'):
            self.assertIn(word, order)

    def test_the_hand_back_is_last_runs_even_after_a_stop_and_says_the_owner_deploys(self):
        self.assertEqual(read_order()[-1][0], HAND_BACK)
        outputs = parsed(HAND_BACK)
        self.assertEqual((outputs['actions'], outputs['cards']), ('status reset fabric agentstart', 'quad'))
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('runs even when a stop job halts the window', 'the all-four reset', 'fabric re-measure', 'agentstart', 'OWNER runs /deploy',
                     'never include the deploy'):
            self.assertIn(word, order)
        text = text_of(HAND_BACK)
        for word in ("TODAY'S PRODUCTION", 'NEVER place a gate arm', 'OWNER', '/deploy', 'never part of this job file', 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE=1'):
            self.assertIn(word, text)
        for name in ORDERED:
            self.assertFalse({'push', 'replay'} & set(parsed(name)['actions'].split()), 'no job pushes, places or replays anything: ' + name)

    def test_the_order_names_the_precondition_the_pairing_and_the_scope(self):
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('NOTHING IN THIS WINDOW HAS RUN ON A CARD', 'THREE audits-off runs', 'three consecutive completions', 'PAIRED per step', 'ABAB',
                     'QWEN_FAST_TP4_TRACED_PUBLISH', 'QWEN_FAST_TRACED_PUBLISH', 'OFF in every profile'):
            self.assertIn(word, order)
        names = [row[0] for row in read_order()]
        for exact in STOP_JOBS:
            for timed in TIMED:
                self.assertLess(names.index(exact), names.index(timed))


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_is_lf_and_names_no_card_host_address_registry_digest_or_path(self):
        for name in ORDERED:
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['cards'], 'quad')
                self.assertNotIn(b'\r', (FOLDER / (name + '.env')).read_bytes(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
        self.assertIsNone(BANNED.search((FOLDER / 'ORDER.txt').read_text(encoding='utf-8')))

    def test_every_device_job_resets_the_cards_first_and_opens_them_in_one_step(self):
        for name in ORDERED[2:-1]:
            actions = parsed(name)['actions'].split()
            with self.subTest(template=name):
                self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertEqual(actions, ['reset', 'smoke'])

    def test_the_profiles_are_the_plans_and_only_the_production_ones_are_traffic_profiles(self):
        for name, profile in PROFILE_OF.items():
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['profile'], profile)
                self.assertIn(profile, PROFILES)
                self.assertEqual(parsed(name)['gate_plan'], 'bringup')
        for name in (BUILD, STOP_DOWN, HAND_BACK):
            self.assertNotIn('gate_only', PROFILES[parsed(name)['profile']], name)
        for name in ORDERED[2:-1]:
            self.assertIs(PROFILES[parsed(name)['profile']].get('gate_only'), True, name)

    def test_the_hang_runs_are_three_templates_on_the_hang_shapes_audits_off(self):
        self.assertEqual(len(HANG_RUNS), 3)
        for name in HANG_RUNS:
            self.assertEqual(parsed(name)['tests'].split(','), HANG_SHAPES, name)
            env = env_of(parsed(name)['profile'])
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'), name)
            self.assertEqual(env['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'], '1', name)
            self.assertEqual(env['QWEN_FAST_BUDGET_CAP'], '1', name)
            self.assertNotIn(AUDIT, env, name)
            self.assertIn('THREE CONSECUTIVE COMPLETIONS', text_of(name))

    def test_the_timing_is_an_abab_of_the_control_and_the_arm_on_the_same_tests_with_lone_user_shapes(self):
        names = [row[0] for row in read_order()]
        self.assertEqual(names[names.index(TIMED[0]):][:4], list(TIMED), 'A B A B, back to back')
        self.assertEqual([parsed(name)['profile'] for name in TIMED], [CONTROL, ARM, CONTROL, ARM])
        for name in TIMED:
            self.assertEqual(parsed(name)['tests'].split(','), TIMING_TESTS, name)
            self.assertIn('speed_window_compare.py', text_of(name))
            self.assertIn('PAIRED', text_of(name))
        self.assertIn('coding', TIMING_TESTS, 'a lone user is the sequential step the lever speeds up')

    def test_the_audited_smoke_runs_the_lone_user_shapes_and_the_steady_mix(self):
        tests = parsed(SMOKE)['tests'].split(',')
        for name in ('warmup', 'coding', 'long_real_text', 'concurrent4_steady', 'steady_resend'):
            self.assertIn(name, tests)
        self.assertEqual(parsed(SMOKE)['profile'], AUDITED)
        for word in ('[TPUB] carry traces engaged', '[TPUB-AUDIT] op=save checked=960 mismatches=0', 'declined'):
            self.assertIn(word, text_of(SMOKE))

    def test_every_named_smoke_test_is_one_the_smoke_runs_in_the_order_the_template_lists_them(self):
        smoke = (HERE / 'c2_serving_smoke.py').read_text(encoding='utf-8')
        executed = re.findall(r"^\s*record\('([a-z0-9_]+)'", smoke, re.M)
        for name in ORDERED[2:-1]:
            listed = parsed(name)['tests'].split(',')
            with self.subTest(template=name):
                for test in listed:
                    self.assertIn("'%s'" % test, smoke)
                self.assertEqual([test for test in executed if test in listed], listed)


class ProfileTests(unittest.TestCase):
    def test_the_arm_is_the_timed_best_plus_the_one_flag_and_the_pair_differs_in_it_alone(self):
        control, arm = PROFILES[CONTROL], PROFILES[ARM]
        self.assertEqual({key for key in set(control['env']) | set(arm['env']) if control['env'].get(key) != arm['env'].get(key)}, {FLAG})
        self.assertEqual((control['env'].get(FLAG), arm['env'][FLAG]), (None, '1'))
        for key in set(control) | set(arm):
            if key not in ('env', 'description'):
                self.assertEqual(control.get(key), arm.get(key), key)
        self.assertIs(arm['gate_only'], True)
        self.assertTrue(arm['description'].startswith('GATE ONLY'))
        self.assertEqual(arm['env']['QWEN_C2_GATE_PROFILE'], '1')

    def test_the_audited_arm_is_the_audited_best_plus_the_flag_and_its_audit(self):
        base, arm = PROFILES['c2-packed-tp4-best-gate'], PROFILES[AUDITED]
        self.assertEqual({key for key in set(base['env']) | set(arm['env']) if base['env'].get(key) != arm['env'].get(key)}, {FLAG, AUDIT})
        self.assertEqual((arm['env'][FLAG], arm['env'][AUDIT]), ('1', '1'))
        self.assertIn('NOT QUALIFIED', arm['description'])

    def test_the_arm_keeps_the_slide_the_hang_fix_the_caps_the_fused_commit_and_the_quad(self):
        env = env_of(ARM)
        self.assertEqual(env['QWEN_FAST_TP_KV_SLIDE'], '1')
        self.assertEqual(env['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'], '1')
        self.assertEqual((env['QWEN_FAST_BUDGET_CAP'], env['QWEN_FAST_SEQ_DEADLINE_S']), ('1', '120'))
        self.assertEqual((env['QWEN_FAST_FUSED_COMMIT'], env['QWEN_FAST_QUAD_DRAFT']), ('1', '1'))

    def test_production_the_image_and_every_other_profile_are_untouched(self):
        for name, profile in PROFILES.items():
            if name not in (ARM, AUDITED):
                self.assertNotIn(FLAG, profile['env'], name)
                self.assertNotIn(AUDIT, profile['env'], name)
        self.assertEqual(env_of(PRODUCTION)['QWEN_FAST_FUSED_COMMIT'], '0')
        self.assertEqual(env_of(PRODUCTION)['QWEN_FAST_TRACED_PUBLISH'], '0')

    def test_the_eager_history_cut_stays_off_in_every_profile_that_names_it(self):
        for name, profile in PROFILES.items():
            if 'QWEN_FAST_TRACED_PUBLISH' in profile['env']:
                self.assertEqual(profile['env']['QWEN_FAST_TRACED_PUBLISH'], '0', name)


class DocumentAndAllowlistTests(unittest.TestCase):
    def test_the_document_is_public_safe_lf_and_names_the_hazards_and_the_open_work(self):
        text = (ROOT / 'docs' / 'tp4-tpub.md').read_text(encoding='utf-8')
        self.assertNotIn('\r', text)
        self.assertIsNone(BANNED.search(text))
        for word in ('QWEN_FAST_TP4_TRACED_PUBLISH', 'QWEN_FAST_TP4_TRACED_PUBLISH_AUDIT', 'capture', 'identity warm', 'allocate_carry', 'declined',
                     'trace region', 'c2-packed-tp4-best-strace-tpub', 'c2-packed-tp4-best-gate-tpub', 'publish_target', 'prepare_history', 'serving_solo_lane',
                     'Needs a card', 'eight seats'):
            self.assertIn(word.lower(), text.lower(), word)

    def test_the_new_tests_are_allowlisted_in_the_cpu_workflow(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        for module in ('test_tp4_tpub', 'test_tp4_tpub_window'):
            self.assertRegex(workflow, r'\b%s\b' % module)


if __name__ == '__main__':
    unittest.main()
