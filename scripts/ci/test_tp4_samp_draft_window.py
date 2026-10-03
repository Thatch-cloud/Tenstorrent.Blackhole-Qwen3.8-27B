"""The tp4/samp-draft window: its job pack (scripts/ci/references/tp4-samp-draft-jobs), its order and the profiles each job serves.

Levers: S1 the sampler's shard argmax kernels (c2-packed-tp4-best-samp, audited arm c2-packed-tp4-best-gate-samp) and D2 the drafter's conv I/O and
head copies (c2-packed-tp4-best-d2, audited arm c2-packed-tp4-best-gate-d2). The pack builds, stops production, audits each lever on a smoke, then times
each against c2-packed-tp4-best-strace in an ABAB. It has no agentstart and no hand-back. The templates are public: no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import c2_smoke_check  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-samp-draft-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)['profiles']
NAMES = sorted(PROFILES)
IMAGE = 'tp4-samp-1'
CONTROL, SAMP, D2 = 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-samp', 'c2-packed-tp4-best-d2'
SAMP_GATE, D2_GATE = 'c2-packed-tp4-best-gate-samp', 'c2-packed-tp4-best-gate-d2'
BUILD, FIRST = 'B0-build', 'A0-agentstop-unserve'
SAMP_AUDIT, D2_AUDIT = 'S1-sampler-audited-smoke', 'S2-drafter-audited-smoke'
SAMP_PAIR = ('TS1-control-timed', 'TS2-sampler-timed', 'TS3-control-timed', 'TS4-sampler-timed')
D2_PAIR = ('TD1-control-timed', 'TD2-drafter-timed', 'TD3-control-timed', 'TD4-drafter-timed')
ORDERED = (BUILD, FIRST, SAMP_AUDIT) + SAMP_PAIR + (D2_AUDIT,) + D2_PAIR
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
TIMING_TESTS = ['warmup', 'coding', 'concurrent4_code_equal', 'concurrent4_code_32k']
AUDIT_TESTS = ['warmup', 'coding', 'concurrent4', 'concurrent4_code_equal']


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def order_text():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return handle.read()


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), NAMES)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns_and_the_order_names_no_other(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [row[0] for row in rows]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(ordered, list(ORDERED))

    def test_the_modes_images_and_minutes(self):
        modes = {}
        for name, mode, image, minutes in read_order():
            modes[name] = mode
            with self.subTest(job=name):
                self.assertIn(mode, ('pre', 'stop', 'optional'))
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= 180, minutes)
        self.assertEqual(modes[BUILD], 'pre')
        self.assertEqual(modes[FIRST], 'stop')
        for name in (SAMP_AUDIT, D2_AUDIT) + SAMP_PAIR + D2_PAIR:
            self.assertEqual(modes[name], 'optional', name)

    def test_the_build_runs_first_production_is_taken_down_second_and_nothing_else_does_it(self):
        self.assertEqual([row[0] for row in read_order()[:2]], [BUILD, FIRST])
        actions = parsed(FIRST)['actions'].split()
        self.assertEqual(actions, ['agentstop', 'unserve'])
        for name in ORDERED:
            if name != FIRST:
                self.assertFalse({'agentstop', 'unserve'} & set(parsed(name)['actions'].split()), name)
        for word in ('PRODUCTION IS LIVE ON THE CARDS', 'A0 (agentstop, unserve) is the first job'):
            self.assertIn(word, order_text())

    def test_the_pack_has_no_agentstart_no_hand_back_no_deploy_and_no_publish(self):
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            for step in ('agentstart', 'push', 'platform', 'probe', 'prefix', 'replay', 'fabric'):
                self.assertNotIn(step, actions, (name, step))
            self.assertFalse(re.search(r'(?im)^C2_(PLACE|DEPLOY)', text_of(name)), name)
        self.assertFalse([row for row in read_order() if row[0].startswith('H')])
        for word in ('ENDS WITHOUT A HAND-BACK', 'no agentstart', 'no deploy'):
            self.assertIn(word, order_text())

    def test_the_order_audits_each_lever_before_its_pair_and_names_the_skip_rules_the_abab_and_the_tau_report(self):
        names = [row[0] for row in read_order()]
        for name in SAMP_PAIR:
            self.assertLess(names.index(SAMP_AUDIT), names.index(name))
            self.assertGreater(names.index(name), names.index(FIRST))
        for name in D2_PAIR:
            self.assertLess(names.index(D2_AUDIT), names.index(name))
        for word in ('ABAB', 'NOTHING COMBINED HAS RUN ON A CARD', 'if S1 fails or hangs, skip TS1-TS4', 'if S2 fails or hangs, skip TD1-TD4',
                     'PAIRED per round', 'tp4_samp_draft_report.py', 'speed_window_compare.py', '--strict-concurrent-prefixes', 'TAU REPORT'):
            self.assertIn(word, order_text())


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_is_lf_and_names_no_card_host_address_registry_or_digest(self):
        for name in ORDERED:
            with self.subTest(template=name):
                parsed(name)
                with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                    self.assertNotIn(b'\r', handle.read(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
        self.assertNotIn('\r', order_text())
        self.assertIsNone(BANNED.search(order_text()))

    def test_the_image_tag_is_the_one_fresh_literal_in_every_template_and_no_placeholder_is_left(self):
        for name in ORDERED:
            self.assertRegex(text_of(name), r'(?m)^C2_IMAGE_TAG=%s$' % IMAGE, name)
        self.assertNotIn('PLACEHOLDER', order_text())

    def test_every_four_card_device_job_resets_first_and_opens_the_cards_in_one_step(self):
        for name in ORDERED:
            outputs = parsed(name)
            if name in (BUILD, FIRST):
                continue
            actions = outputs['actions'].split()
            with self.subTest(template=name):
                self.assertEqual(outputs['cards'], 'quad')
                self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertEqual(actions[0], 'reset', 'links train only at board init: a four-card job follows a reset')

    def test_only_the_first_job_builds_the_image_and_it_opens_no_card(self):
        for name in ORDERED:
            self.assertEqual('build' in parsed(name)['actions'].split(), name == BUILD, name)
        outputs = parsed(BUILD)
        self.assertEqual((outputs['actions'], outputs['cards'], outputs['tag']), ('build', 'quad', IMAGE))
        self.assertFalse(set(outputs['actions'].split()) & set(DEVICE_STEPS + ('reset', 'agentstop', 'unserve', 'agentstart', 'push')))
        for word in ('BEFORE A0', 'FRESH', 'already exists', 'production still serves'):
            self.assertIn(word, text_of(BUILD))

    def test_the_audited_smokes_serve_the_audited_arms_and_run_the_concurrent_solo_comparison(self):
        for name, profile in ((SAMP_AUDIT, SAMP_GATE), (D2_AUDIT, D2_GATE)):
            outputs = parsed(name)
            with self.subTest(job=name):
                self.assertEqual((outputs['actions'], outputs['profile']), ('reset smoke', profile))
                self.assertEqual(outputs['tests'].split(','), AUDIT_TESTS)
                env = PROFILES[profile]['env']
                self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'))
                self.assertIs(PROFILES[profile].get('gate_only'), True)
                self.assertIn('SKIP the lever\'s timing pair', text_of(name))
        self.assertIn('shard argmax audit N exact=True', text_of(SAMP_AUDIT))
        self.assertIn('draft conv audit exact=True', text_of(D2_AUDIT))
        self.assertIn('draft heads audit exact=True', text_of(D2_AUDIT))

    def test_the_timing_jobs_of_each_pair_run_the_same_coding_tests_4x4k_and_4x32k(self):
        for name in SAMP_PAIR + D2_PAIR:
            outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual(outputs['tests'].split(','), TIMING_TESTS)
                self.assertEqual(outputs['actions'], 'reset smoke')
                self.assertEqual(outputs['gate_plan'], 'bringup')

    def test_each_pair_is_a_b_a_b_of_best_strace_against_its_lever_twin(self):
        self.assertEqual([parsed(name)['profile'] for name in SAMP_PAIR], [CONTROL, SAMP, CONTROL, SAMP])
        self.assertEqual([parsed(name)['profile'] for name in D2_PAIR], [CONTROL, D2, CONTROL, D2])
        for name in SAMP_PAIR + D2_PAIR:
            self.assertIs(PROFILES[parsed(name)['profile']].get('gate_only'), True, name)

    def test_the_lever_twins_differ_from_the_control_in_their_flags_alone_and_no_pair_mixes_levers(self):
        differing = lambda left, right: {key for key in set(left) | set(right) if left.get(key) != right.get(key)}  # noqa: E731
        self.assertEqual(differing(PROFILES[SAMP]['env'], PROFILES[CONTROL]['env']), {'QWEN_FAST_TP4_SHARD_ARGMAX'})
        self.assertEqual(differing(PROFILES[D2]['env'], PROFILES[CONTROL]['env']),
                         {'QWEN_FAST_TP4_DRAFT_CONV', 'QWEN_FAST_TP4_DRAFT_HEADS'})

    def test_the_timed_templates_say_what_the_smoke_check_fails_and_how_to_read_the_arm(self):
        for name in SAMP_PAIR + D2_PAIR:
            text = text_of(name)
            for word in ('PAIRED per round', 'ABAB', 'fall-back line', 'hang fix A'):
                self.assertIn(word, text, (name, word))
        for name in D2_PAIR:
            self.assertIn('tp4_samp_draft_report.py', text_of(name))
            self.assertIn('--strict-concurrent-prefixes', text_of(name))

    def test_the_smoke_check_knows_every_lever_flag_a_job_of_this_pack_serves(self):
        known = {flag for flag, _, _, _ in c2_smoke_check.SAMPDRAFT_LEVERS} | {flag for flag, _, _ in c2_smoke_check.SAMPDRAFT_AUDITS}
        for name in ORDERED:
            profile = parsed(name)['profile']
            if profile in PROFILES:
                for flag in PROFILES[profile]['env']:
                    if flag.startswith(('QWEN_FAST_TP4_SHARD_ARGMAX', 'QWEN_FAST_TP4_DRAFT_CONV', 'QWEN_FAST_TP4_DRAFT_HEADS')):
                        self.assertIn(flag, known, (name, flag))


class ReportCommandTests(unittest.TestCase):
    def test_the_report_commands_in_the_templates_name_a_script_that_exists_with_the_flags_it_has(self):
        import tp4_samp_draft_report
        for name in D2_PAIR[:1]:
            text = text_of(name)
            self.assertIn('--control', text)
            self.assertIn('--lever', text)
        self.assertTrue(os.path.isfile(os.path.join(HERE, 'tp4_samp_draft_report.py')))
        self.assertEqual(tp4_samp_draft_report.MIN_ROUNDS, 2000)
        self.assertEqual(tp4_samp_draft_report.TOLERANCE, 0.01)
        self.assertEqual(tp4_samp_draft_report.POSITION_POINTS, 0.02)


if __name__ == '__main__':
    unittest.main()
