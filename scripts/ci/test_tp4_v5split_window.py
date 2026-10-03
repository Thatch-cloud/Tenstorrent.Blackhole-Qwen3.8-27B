"""The tp4/v5split window: the recurrence value split's gate-only profiles, its job templates (scripts/ci/references/tp4-v5split-jobs) and their order.

Lever (docs/tp4-recurrence-split.md): V5, the K5-A recurrence launch with each head's value columns split over two cores
(QWEN_FAST_GDN_SPLIT_V=2, gdn_seq_block_split), refused until the single-card byte gate has qualified its generated sources. Every new behaviour
is off in production's profile; each timing twin is its control plus exactly the flag; the audited arm is its control plus the flag and the K5-A
audit; the window opens with 'agentstop unserve', runs the byte gate on card M alone, and hands back with a reset, a fabric re-measure and
agentstart (never a deploy). The templates are public, so they name no rig, card, address, registry or digest.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import c2_smoke_check  # noqa: E402
import gdn_seq_block_split as split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-v5split-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
FLAG = 'QWEN_FAST_GDN_SPLIT_V'
AUDIT_FLAG = 'QWEN_FAST_GDN_SEQ_BLOCK_AUDIT'
BUILD1, FIRST, WATCHER, FULL, BUILD2, AUDITED = ('V0-build', 'A0-agentstop-unserve', 'V1a-cardm-watcher', 'V1b-cardm-full',
                                                 'V2-build-qualified', 'V3-quad-audited-smoke')
HANG = tuple('V4%s-hang-shapes-v5' % letter for letter in 'abcde')
TIMED = ('T1-control-timed', 'T2-v5-timed', 'T3-control-timed', 'T4-v5-timed')
HANDBACK = 'H1-handback-reset-fabric-agentstart'
ORDERED = (BUILD1, FIRST, WATCHER, FULL, BUILD2, AUDITED) + HANG + TIMED + (HANDBACK,)
IMAGE_ONE, IMAGE_TWO = 'tp4-v5split-1', 'tp4-v5split-2'
CONTROL_BEST, CONTROL_PRODUCTION, CONTROL_AUDITED = 'c2-packed-tp4-best', 'c2-packed-tp4-speed-strace', 'c2-packed-tp4-best-gate'
TWIN_BEST, TWIN_PRODUCTION, TWIN_AUDITED = ('c2-packed-tp4-best-v5', 'c2-packed-tp4-speed-strace-v5', 'c2-packed-tp4-best-v5-gate')
NEW = (TWIN_BEST, TWIN_PRODUCTION, TWIN_AUDITED)
PRODUCTION = 'c2-packed-tp4'
TIMING_TESTS = ['warmup', 'coding', 'concurrent4_code_equal', 'concurrent4_code_32k']


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


def body(name):
    return PROFILES['profiles'][name]


def env_of(name):
    return body(name)['env']


def differing(left, right):
    return {key for key in set(left) | set(right) if left.get(key) != right.get(key)}


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns_and_the_order_names_no_other(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(sorted(row[0] for row in rows), on_disk)
        self.assertEqual([row[0] for row in rows], list(ORDERED))

    def test_the_modes_images_and_minutes(self):
        modes = {}
        for name, mode, image, minutes in read_order():
            modes[name] = mode
            with self.subTest(job=name):
                self.assertIn(mode, ('stop', 'optional'))
                self.assertEqual(image, parsed(name)['tag'])
                self.assertIn(image, (IMAGE_ONE, IMAGE_TWO))
                self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= 180, minutes)
        for name in (BUILD1, FIRST, WATCHER, FULL, BUILD2, AUDITED, HANDBACK) + HANG:
            self.assertEqual(modes[name], 'stop', name)
        for name in TIMED:
            self.assertEqual(modes[name], 'optional', name)

    def test_the_two_images_split_at_the_qualified_commit(self):
        for name in (BUILD1, FIRST, WATCHER, FULL):
            self.assertEqual(parsed(name)['tag'], IMAGE_ONE, name)
        for name in (BUILD2, AUDITED) + HANG + TIMED + (HANDBACK,):
            self.assertEqual(parsed(name)['tag'], IMAGE_TWO, name)
        for word in ('tp4-v5split-1', 'tp4-v5split-2', 'PLACEHOLDER', 'a reused tag is skipped by the build step'):
            self.assertIn(word.lower(), order_text().lower(), word)

    def test_the_first_job_takes_production_down_and_nothing_else_does(self):
        self.assertEqual([row[0] for row in read_order()[:2]], [BUILD1, FIRST], 'the build runs first, production still serving')
        actions = parsed(FIRST)['actions'].split()
        self.assertEqual(actions, ['status', 'agentstop', 'unserve'])
        for name in ORDERED:
            if name != FIRST:
                self.assertFalse({'agentstop', 'unserve'} & set(parsed(name)['actions'].split()), name)
        for word in ('PRODUCTION IS LIVE ON THE CARDS', 'A0 is the first job'):
            self.assertIn(word, order_text())

    def test_the_hand_back_is_last_resets_remeasures_the_fabric_restarts_the_agent_and_never_deploys(self):
        self.assertEqual(read_order()[-1][0], HANDBACK)
        self.assertEqual(parsed(HANDBACK)['actions'].split(), ['status', 'reset', 'fabric', 'agentstart'])
        self.assertEqual(parsed(HANDBACK)['cards'], 'quad')
        for word in ("TODAY'S PRODUCTION RECIPE", 'NEVER place a gate arm', 'AN OWNER /deploy IS STILL NEEDED', 'fabric re-measure',
                     'stops before agentstart', 'c2-packed-tp4', 'audits off', 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE=1'):
            self.assertIn(word, text_of(HANDBACK))
        for word in ('runs LAST and even when any earlier job failed or hung', 'An owner /deploy is still needed', 'never'):
            self.assertIn(word, order_text())
        for name in ORDERED:
            self.assertNotIn('push', parsed(name)['actions'].split(), 'no :latest retag, no publish')
            self.assertFalse(re.search(r'(?im)^C2_(PLACE|DEPLOY)', text_of(name)), name)

    def test_the_order_runs_the_byte_gate_then_the_qualified_build_then_the_four_card_arms(self):
        names = [row[0] for row in read_order()]
        self.assertEqual(names[:6], [BUILD1, FIRST, WATCHER, FULL, BUILD2, AUDITED])
        self.assertEqual(names[6:11], list(HANG))
        self.assertEqual(names[11:15], list(TIMED))
        for word in ('ABAB', 'PAIRED per round', 'If V3 fails, skip V4 and T1-T4', 'restarts at zero', 'five consecutive hang-shape',
                     'QUALIFIED', 'full-scope PASS', 'sha256 triple'):
            self.assertIn(word, order_text())


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_and_is_lf_and_names_no_card_host_address_registry_digest_or_placeholder(self):
        for name in ORDERED:
            with self.subTest(job=name):
                text = text_of(name)
                self.assertNotIn('\r', text)
                self.assertIsNone(BANNED.search(text))
                parsed(name)
        self.assertIsNone(BANNED.search(order_text()))

    def test_every_four_card_job_resets_first_and_opens_the_cards_in_one_step(self):
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            if parsed(name)['cards'] == 'quad' and {'smoke', 'gate'} & set(actions):
                with self.subTest(job=name):
                    self.assertEqual(actions[0], 'reset')
                    self.assertEqual(len([a for a in actions if a in ('smoke', 'gate')]), 1)

    def test_only_the_two_build_jobs_build_and_they_open_no_card(self):
        builders = [name for name in ORDERED if 'build' in parsed(name)['actions'].split()]
        self.assertEqual(builders, [BUILD1, BUILD2])
        for name in builders:
            self.assertEqual(parsed(name)['actions'].split(), ['build'])

    def test_the_byte_gate_jobs_are_one_card_cardm_jobs_on_the_v5split_harness(self):
        for name in (WATCHER, FULL):
            with self.subTest(job=name):
                entry = parsed(name)
                self.assertEqual((entry['cards'], entry['actions'].split()), ('pair', ['cardm']))
                self.assertEqual(entry['cardm_harness'], 'optimisation/ttnn-op/v5split/run_card_m.sh')
                self.assertTrue(os.path.isfile(os.path.join(ROOT, entry['cardm_harness'])))
                self.assertIn('IMAGE_TAG=' + entry['tag'], entry['cardm_env'])
        self.assertIn('WATCHER=1', parsed(WATCHER)['cardm_env'])
        self.assertNotIn('WATCHER=1', parsed(FULL)['cardm_env'])
        self.assertEqual(parsed(FULL)['cardm_args'], '')
        # the watcher pass is a reduced scope: it can never read PASS, and says so
        self.assertIn('--sections', parsed(WATCHER)['cardm_args'])
        for word in ('reduced scope never reads PASS', 'FULL scope', 'v5_triple', 'proceed', 'image-build', 'write-bound'):
            self.assertIn(word.lower(), (text_of(WATCHER) + text_of(FULL)).lower(), word)

    def test_the_audited_arm_is_the_audited_best_plus_the_flag_and_the_audit(self):
        entry = parsed(AUDITED)
        self.assertEqual((entry['cards'], entry['profile']), ('quad', TWIN_AUDITED))
        self.assertEqual(entry['actions'].split(), ['reset', 'smoke'])
        for word in ('mismatches=0', 'engaged line', '0, 23 and 47', 'never timed'):
            self.assertIn(word, text_of(AUDITED))

    def test_the_hang_shape_runs_are_five_identical_audits_off_runs_on_the_production_recipe_twin(self):
        bodies = set()
        for name in HANG:
            entry = parsed(name)
            self.assertEqual((entry['cards'], entry['profile']), ('quad', TWIN_PRODUCTION))
            self.assertEqual(entry['actions'].split(), ['reset', 'smoke'])
            bodies.add(''.join(line for line in text_of(name).splitlines(True) if not line.startswith('#')))
        self.assertEqual(len(bodies), 1)
        self.assertIn('FIVE CONSECUTIVE COMPLETIONS', text_of(HANG[0]))

    def test_the_timing_jobs_run_the_same_coding_tests_and_each_pair_is_a_b_a_b_of_the_control_against_its_twin(self):
        for name in TIMED:
            self.assertEqual(parsed(name)['tests'].split(','), TIMING_TESTS, name)
        self.assertEqual([parsed(name)['profile'] for name in TIMED],
                         [CONTROL_BEST, TWIN_BEST, CONTROL_BEST, TWIN_BEST])
        for word in ('PAIRED per round', 'never against the other pair'):
            self.assertIn(word, text_of(TIMED[0]))


class ProfileTests(unittest.TestCase):
    def test_the_timed_twin_is_the_combined_best_plus_exactly_the_flag(self):
        self.assertEqual(differing(env_of(TWIN_BEST), env_of(CONTROL_BEST)), {FLAG})
        self.assertEqual(env_of(TWIN_BEST)[FLAG], '2')
        self.assertEqual({k: v for k, v in body(TWIN_BEST).items() if k not in ('description', 'env')},
                         {k: v for k, v in body(CONTROL_BEST).items() if k not in ('description', 'env')})

    def test_the_production_recipe_twin_is_speed_strace_plus_exactly_the_flag(self):
        self.assertEqual(differing(env_of(TWIN_PRODUCTION), env_of(CONTROL_PRODUCTION)), {FLAG})
        self.assertEqual({k: v for k, v in body(TWIN_PRODUCTION).items() if k not in ('description', 'env')},
                         {k: v for k, v in body(CONTROL_PRODUCTION).items() if k not in ('description', 'env')})

    def test_the_audited_twin_is_the_audited_best_plus_the_flag_and_the_k5a_audit_on_three_layers(self):
        self.assertEqual(differing(env_of(TWIN_AUDITED), env_of(CONTROL_AUDITED)), {FLAG, AUDIT_FLAG})
        self.assertEqual(env_of(TWIN_AUDITED)[AUDIT_FLAG], '0,23,47')
        self.assertEqual(env_of(TWIN_AUDITED)[FLAG], '2')
        self.assertEqual(env_of(TWIN_AUDITED)['QWEN_FAST_VERIFY_T1_AUDIT'], '1')

    def test_the_timing_twins_carry_the_audits_off_recipe_and_the_caps_of_production(self):
        for name in (TWIN_BEST, TWIN_PRODUCTION):
            env = env_of(name)
            self.assertEqual(env['QWEN_FAST_VERIFY_T1_AUDIT'], '0', name)
            self.assertEqual(env['QWEN_FAST_VERIFY_T2_AUDIT'], '0', name)
            self.assertNotIn(AUDIT_FLAG, env, name)
            self.assertEqual(env['QWEN_FAST_TP'], '4')

    def test_production_carries_none_of_it_and_no_other_profile_carries_the_flag(self):
        self.assertNotIn(FLAG, env_of(PRODUCTION))
        self.assertFalse(body(PRODUCTION).get('gate_only'))
        carriers = sorted(name for name in NAMES if FLAG in env_of(name))
        self.assertEqual(carriers, sorted(NEW))

    def test_the_new_profiles_are_gate_only_and_never_traffic(self):
        for name in NEW:
            self.assertTrue(body(name).get('gate_only'), name)
            self.assertIn('GATE ONLY', body(name)['description'])
            self.assertEqual(body(name)['mesh_device'], 'P150x4')
        for name in NEW:
            self.assertIn('qualified', body(name)['description'], name)

    def test_the_image_dockerfile_sets_k5a_on_so_the_flag_needs_nothing_from_the_profile(self):
        with open(os.path.join(ROOT, 'docker', 'qwen-c2-serving.Dockerfile'), encoding='utf-8') as handle:
            dockerfile = handle.read()
        self.assertIn('QWEN_FAST_GDN_SEQ_BLOCK=1 QWEN_FAST_GDN_USER_BATCH=1', dockerfile)
        for name in NEW:
            self.assertNotIn('QWEN_FAST_GDN_SEQ_BLOCK', env_of(name) if name != TWIN_AUDITED else
                             {k: v for k, v in env_of(name).items() if k != AUDIT_FLAG})


class SmokeCheckTests(unittest.TestCase):
    """c2_smoke_check: an arm that asked for the split and logged no engaged line ran K5-A, and its timing is not the lever's."""

    def split_problems(self, log, profile=TWIN_BEST):
        problems, _ = c2_smoke_check.check('', log, False, env=env_of(profile))
        return [problem for problem in problems if 'split recurrence' in problem]

    def test_an_engaged_line_passes(self):
        self.assertEqual(self.split_problems(split.engaged_line(4, True)), [])

    def test_no_engaged_line_fails_every_arm_that_asked_for_the_lever(self):
        for profile in NEW:
            found = self.split_problems('', profile)
            self.assertEqual(len(found), 1, profile)
            self.assertIn(FLAG, found[0])

    def test_the_controls_are_not_asked_for_the_marker(self):
        for profile in (CONTROL_BEST, CONTROL_PRODUCTION, CONTROL_AUDITED, PRODUCTION):
            self.assertEqual(self.split_problems('', profile), [], profile)

    def test_the_markers_are_the_module(self):
        self.assertEqual(c2_smoke_check.GDN_SPLIT_FLAG, split.FLAG)
        self.assertTrue(split.engaged_line(1, False).startswith(c2_smoke_check.GDN_SPLIT_ENGAGED))
        self.assertTrue(split.engaged_line(3, True).startswith(split.MARKER))


class LeverIsOffByDefaultTests(unittest.TestCase):
    def test_the_k5a_launch_is_what_runs_unless_the_flag_is_two(self):
        for value in (None, '', '0', '1'):
            environ = {'QWEN_FAST_TP': '4'}
            if value is not None:
                environ[FLAG] = value
            self.assertEqual(split.factor(environ), 1)
        self.assertEqual(split.QUALIFIED, {}, 'nothing may serve the split before the single-card byte gate has qualified it')


if __name__ == '__main__':
    unittest.main()
