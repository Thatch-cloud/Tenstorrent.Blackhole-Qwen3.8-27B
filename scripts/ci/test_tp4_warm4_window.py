"""The tp4/warm4 window: its job templates (scripts/ci/references/tp4-warm4-jobs) and their order.

The window asks whether the four-seat server, with the pinned sampler OUT of the verify trace and the request widths warmed before the
single block's capture (QWEN_FAST_M3_REQUEST_WARM), hangs on the S8-0 shapes, is exact, and is faster than production's recipe. The templates
are public, so they name no rig, card, address, registry or digest.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-warm4-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
IMAGE = 'tp4-warm4a'
FLAG = 'QWEN_FAST_M3_REQUEST_WARM'
BUILD, FIRST, HANDBACK = 'B0-build', 'A0-agentstop-unserve', 'H1-handback-reset-fabric-agentstart'
W1 = 'W1-warm4-diag'
W_SERIES = (W1, 'W1b-warm4-diag', 'W1c-warm4-diag', 'W1d-warm4-diag', 'W1e-warm4-diag')
CONTROL, PARITY, EXACT, SERVED, OLDTAIL = 'W2-nowarm-control', 'W3-even-parity', 'G1-warm4-exact-audited', 'G2-warm4-served-matrix', 'W4-oldtail'
TIMED = ('TA1-control-timed', 'TB1-warm4-timed', 'TA2-control-timed', 'TB2-warm4-timed')
ORDERED = (BUILD, FIRST, W1, CONTROL, PARITY) + W_SERIES[1:] + (EXACT, SERVED) + TIMED + (OLDTAIL, HANDBACK)
SMOKE = ['warmup', 'coding', 'concurrent4', 'concurrent4_steady', 'steady_resend', 'replay_concurrent4', 'concurrent4_code_equal']
TIMING = ['warmup', 'coding', 'concurrent4_code_equal', 'concurrent4_code_32k']
PRODUCTION, A_ARM, B_ARM = 'c2-packed-tp4', 'c2-packed-tp4-speed-strace', 'c2-packed-tp4-speed-warm4'


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


def values(name):
    return job.parse_env(text_of(name))


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
                self.assertIn(mode, ('stop', 'optional', 'manual'))
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= 180, minutes)
        for name in (BUILD, FIRST, HANDBACK, EXACT) + W_SERIES:
            self.assertEqual(modes[name], 'stop', name)
        for name in (CONTROL, PARITY, SERVED) + TIMED:
            self.assertEqual(modes[name], 'optional', name)
        self.assertEqual(modes[OLDTAIL], 'manual')

    def test_the_build_runs_first_while_production_serves_and_opens_no_card(self):
        self.assertEqual([row[0] for row in read_order()[:2]], [BUILD, FIRST])
        for name in ORDERED:
            self.assertEqual('build' in parsed(name)['actions'].split(), name == BUILD, name)
        outputs = parsed(BUILD)
        self.assertEqual((outputs['actions'], outputs['cards'], outputs['tag'], outputs['profile']), ('build', 'quad', IMAGE, PRODUCTION))
        for word in ('BEFORE A0', 'FRESH', 'already exists', 'PLACEHOLDER', 'production still serves'):
            self.assertIn(word, text_of(BUILD))
        self.assertIn('FRESH tag', order_text())

    def test_the_first_job_after_the_build_takes_production_down_and_nothing_else_does(self):
        self.assertEqual(parsed(FIRST)['actions'], 'status agentstop unserve')
        for name in ORDERED:
            if name != FIRST:
                self.assertFalse({'agentstop', 'unserve'} & set(parsed(name)['actions'].split()), name)
        for word in ('PRODUCTION IS LIVE ON THE CARDS', 'A0 (agentstop, unserve) is the first job'):
            self.assertIn(word, order_text())

    def test_the_hand_back_is_last_resets_remeasures_the_fabric_restarts_the_agent_and_never_deploys(self):
        self.assertEqual(read_order()[-1][0], HANDBACK)
        self.assertEqual(parsed(HANDBACK)['actions'].split(), ['status', 'reset', 'fabric', 'agentstart'])
        for word in ("TODAY'S PRODUCTION RECIPE", 'NEVER place a gate arm', 'AN OWNER /deploy IS STILL NEEDED', 'fabric re-measure',
                     'stops before agentstart', 'c2-packed-tp4', 'audits off', 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE=1'):
            self.assertIn(word, text_of(HANDBACK))
        for word in ('runs LAST and even when any earlier job failed or hung', 'An owner /deploy is still needed', 'never includes a deploy',
                     'PROMOTION IS OUT OF SCOPE', 'PAIRED per round', 'ABAB', 'RUN AFTER the current next-3 window has handed back'):
            self.assertIn(word, order_text())
        for name in ORDERED:
            self.assertNotIn('push', parsed(name)['actions'].split())
            self.assertFalse(re.search(r'(?im)^C2_(PLACE|DEPLOY)', text_of(name)), name)

    def test_the_stop_and_skip_rules_are_written_down(self):
        for word in ('B0 or A0 failure stops the window', 'A stall restarts the count of five only on a new image', 'A G1 failure skips the timing',
                     'blocks promotion', 'H1 always runs', 'rshard arm', 'request-drafter warm'):
            self.assertIn(word, order_text())


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_and_is_lf_and_names_no_card_host_address_registry_digest_or_placeholder(self):
        for name in ORDERED:
            with self.subTest(template=name):
                parsed(name)
                with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                    self.assertNotIn(b'\r', handle.read(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
        self.assertNotIn('\r', order_text())
        self.assertIsNone(BANNED.search(order_text()))

    def test_every_four_card_job_resets_first_and_opens_the_cards_in_one_step(self):
        for name in ORDERED:
            outputs = parsed(name)
            if name in (BUILD, FIRST, HANDBACK):
                continue
            actions = outputs['actions'].split()
            with self.subTest(template=name):
                self.assertEqual(outputs['cards'], 'quad')
                self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertEqual(actions[0], 'reset', 'links train only at board init: a four-card job follows a reset')

    def test_every_profile_a_template_names_exists_and_every_non_production_one_is_gate_only(self):
        for name in ORDERED:
            profile = parsed(name)['profile']
            if name in (FIRST, HANDBACK):
                continue
            with self.subTest(template=name):
                self.assertIn(profile, PROFILES['profiles'])
                if name != BUILD:
                    self.assertIs(PROFILES['profiles'][profile].get('gate_only'), True)

    def test_the_w_series_is_five_identical_runs_of_the_warm_diag_profile_on_the_smoke_list(self):
        self.assertEqual(len(W_SERIES), 5)
        for name in W_SERIES:
            outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual((outputs['actions'], outputs['profile']), ('reset smoke', 'c2-packed-tp4-warm4-diag'))
                self.assertEqual(outputs['tests'].split(','), SMOKE)
        configs = {tuple(sorted(values(name).items())) for name in W_SERIES}
        self.assertEqual(len(configs), 1, 'the five repeats differ only in their comments')
        for word in ('five consecutive', 'W pass', 'request_warm_programs', 'before the', 'S8-0'):
            self.assertIn(word, text_of(W1))

    def test_the_control_and_the_parity_arm_run_the_same_tests_on_their_profiles(self):
        self.assertEqual((parsed(CONTROL)['profile'], parsed(PARITY)['profile']),
                         ('c2-packed-tp4-warm4-control', 'c2-packed-tp4-warm4-even-diag'))
        for name in (CONTROL, PARITY):
            self.assertEqual(parsed(name)['tests'].split(','), SMOKE)
            self.assertEqual(parsed(name)['actions'], 'reset smoke')
        self.assertEqual(PROFILES['profiles']['c2-packed-tp4-warm4-control']['env'][FLAG], '0')
        self.assertEqual(PROFILES['profiles']['c2-packed-tp4-warm4-even-diag']['env'][FLAG], 'even')
        for word in ('It must hang', 'verifier_engine.py:291', 'Forcing argmax', 'Two non-hanging controls'):
            self.assertIn(word, text_of(CONTROL))
        for word in ('seq=1 ccl before', 'void', 'cures by parity shift'):
            self.assertIn(word, text_of(PARITY))

    def test_the_audited_arm_and_the_served_arm_run_the_s3a_matrix(self):
        for name, profile in ((EXACT, 'c2-packed-tp4-warm4-gate'), (SERVED, B_ARM)):
            outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual((outputs['actions'], outputs['profile'], outputs['gate_plan']), ('reset gate', profile, 'matrix'))
                self.assertEqual((outputs['gate_lengths'], outputs['gate_max_tokens']), ('4096,16384,32768,60000', '512'))
        env = PROFILES['profiles']['c2-packed-tp4-warm4-gate']['env']
        self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'))
        for word in ('exactness', 'POLICY PASS', 'not hang safety'):
            self.assertIn(word, text_of(EXACT))

    def test_the_timing_jobs_are_an_abab_of_the_control_against_the_warm_twin_on_the_same_coding_tests(self):
        self.assertEqual([parsed(name)['profile'] for name in TIMED], [A_ARM, B_ARM, A_ARM, B_ARM])
        for name in TIMED:
            outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual(outputs['tests'].split(','), TIMING)
                self.assertEqual(outputs['actions'], 'reset smoke')
                self.assertEqual(outputs['gate_plan'], 'bringup')
        for word in ('PAIRED per round', 'speed_window_compare.py', '--strict-concurrent-prefixes', 'no gain', '2 ms'):
            self.assertIn(word, text_of(TIMED[1]))

    def test_the_old_tail_job_is_manual_and_names_the_st_list(self):
        outputs = parsed(OLDTAIL)
        self.assertEqual(outputs['profile'], 'c2-packed-tp4-warm4-diag-oldtail')
        self.assertEqual(outputs['tests'].split(','), ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_v164order',
                                                       'concurrent4_steady', 'replay_concurrent4'])


if __name__ == '__main__':
    unittest.main()
