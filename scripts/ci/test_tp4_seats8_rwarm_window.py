"""The tp4-seats8 request-warm probe: its job templates (scripts/ci/references/tp4-seats8-rwarm-jobs), their order and what each asks of the tree.

The eight-seat attach hung at the rows=2 eager warm-up of the first request-engine build after block B's first round (S8-1, S8-3a). The fix is
QWEN_FAST_M3_REQUEST_WARM=1 (request_width_warm). The probe: R1 on the diag twin with the flag (five consecutive completions with R1b to R1e),
R2 the control without it (must still hang), R3 the request-shard discriminator (only if R1 hangs at the same fence), then the probe tail of
tp4-seats8-jobs re-pointed at this image. No job here serves, unserves or hands production back: the window driver does that separately.
The templates are public, so they name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import test_c2_serving_gate as base  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-seats8-rwarm-jobs')
SIBLING = os.path.join(HERE, 'references', 'tp4-seats8-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
IMAGE = 'tp4-seats8-rwarm'
DIAG, NOWARM, RSHARD = ('c2-packed-tp4-8-diag-strace', 'c2-packed-tp4-8-diag-strace-nowarm', 'c2-packed-tp4-8-diag-strace-rshard')
REPEATS = ('R1-rwarm', 'R1b-rwarm', 'R1c-rwarm', 'R1d-rwarm', 'R1e-rwarm')
TAIL = ('S8-1-seats8-attach-audited', 'S8-6-memory8', 'S8-7a-seats4-timed', 'S8-7b-seats8-timed')
ORDERED = ('R1-rwarm', 'R2-nowarm-control', 'R3-rshard') + REPEATS[1:] + TAIL
OPTIONAL = ('R2-nowarm-control', 'S8-7a-seats4-timed', 'S8-7b-seats8-timed')
MANUAL = ('R3-rshard',)
PROFILE_OF = {**{name: DIAG for name in REPEATS}, 'R2-nowarm-control': NOWARM, 'R3-rshard': RSHARD,
              'S8-1-seats8-attach-audited': 'c2-packed-tp4-8-gate', 'S8-6-memory8': 'c2-packed-tp4-8',
              'S8-7a-seats4-timed': 'c2-packed-tp4-speed-strace', 'S8-7b-seats8-timed': 'c2-packed-tp4-8-time-gate'}
R_TESTS = ['warmup', 'concurrent8_steady', 'steady_resend', 'replay_concurrent8', 'concurrent8_drain']


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def order_text():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return handle.read()


def text_of(name, folder=FOLDER):
    with open(os.path.join(folder, name + '.env'), encoding='utf-8') as handle:
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
        step_minutes = base.WorkflowTests.budget_literals()['step_minutes']
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertEqual(mode, 'manual' if name in MANUAL else 'optional' if name in OPTIONAL else 'stop')
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= step_minutes, minutes)
        minutes = {row[0]: int(row[3]) for row in read_order()}
        self.assertEqual((minutes['R1-rwarm'], minutes['R2-nowarm-control'], minutes['R3-rshard']), (40, 35, 40))

    def test_r1_is_first_the_control_and_the_discriminator_follow_and_the_repeats_come_before_the_probe_tail(self):
        ordered = [row[0] for row in read_order()]
        self.assertEqual(ordered[:3], ['R1-rwarm', 'R2-nowarm-control', 'R3-rshard'])
        self.assertEqual(ordered[3:7], list(REPEATS[1:]))
        self.assertEqual(ordered[7:], list(TAIL))
        self.assertEqual(len(REPEATS), 5, 'five in a row')

    def test_there_is_no_handback_and_no_unserve_and_the_order_says_the_driver_handles_production(self):
        text = order_text()
        for word in ('NO HANDBACK AND NO UNSERVE JOB', 'window driver', 'FIVE CONSECUTIVE', 'CONTROL', 'KILL SIGNALS', 'STOP RULES',
                     'QWEN_FAST_M3_REQUEST_WARM', 'RECORDED, not bounded', 'request widths warmed before the packed traces',
                     'packed blocks capture block=0'):
            self.assertIn(word, text)
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            self.assertNotIn('unserve', actions, name)
            self.assertNotIn('build', actions, name)
            self.assertFalse([a for a in actions if a in ('serve', 'place', 'handback')], name)
        self.assertFalse([name for name in ORDERED if 'handback' in name.lower()])


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_and_is_lf_and_names_no_card_host_address_registry_digest(self):
        for name in ORDERED:
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['cards'], 'quad')
                with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                    self.assertNotIn(b'\r', handle.read(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
        with open(os.path.join(FOLDER, 'ORDER.txt'), 'rb') as handle:
            data = handle.read()
        self.assertNotIn(b'\r', data)
        self.assertIsNone(BANNED.search(data.decode('utf-8')))

    def test_every_job_resets_the_cards_first_and_opens_them_in_one_step(self):
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            with self.subTest(template=name):
                self.assertLessEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertIn('reset', actions)
                self.assertEqual(actions[0], 'reset')

    def test_the_profiles_are_the_plans_and_every_one_exists(self):
        for name in ORDERED:
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['profile'], PROFILE_OF[name])
                self.assertIn(PROFILE_OF[name], NAMES)

    def test_every_diag_arm_is_gate_only_and_the_flag_is_on_except_in_the_control(self):
        for name in REPEATS + ('R2-nowarm-control', 'R3-rshard'):
            profile = PROFILES['profiles'][parsed(name)['profile']]
            with self.subTest(template=name):
                self.assertIs(profile['gate_only'], True)
                self.assertEqual(profile['env']['QWEN_FAST_M3_REQUEST_WARM'], '0' if name == 'R2-nowarm-control' else '1')
                self.assertEqual(profile['env']['QWEN_FAST_STALL_BUILD_S'], '150')
        for name in TAIL:
            env = PROFILES['profiles'][parsed(name)['profile']]['env']
            with self.subTest(template=name):
                if 'c2-packed-tp4-8' in PROFILE_OF[name]:
                    self.assertEqual(env['QWEN_FAST_M3_REQUEST_WARM'], '1')
                else:
                    self.assertNotIn('QWEN_FAST_M3_REQUEST_WARM', env)


class SmokeTests(unittest.TestCase):
    @staticmethod
    def executed():
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            return re.findall(r"^\s*record\('([a-z0-9_]+)'", handle.read(), re.M)

    def test_the_r_runs_carry_the_five_hang_shapes_in_the_judges_order(self):
        for name in REPEATS + ('R2-nowarm-control', 'R3-rshard'):
            self.assertEqual(parsed(name)['tests'].split(','), R_TESTS, name)
            self.assertEqual(parsed(name)['actions'], 'reset smoke', name)

    def test_the_tail_lists_are_the_sibling_windows_unchanged(self):
        for name in TAIL:
            with self.subTest(template=name):
                mine, theirs = parsed(name), job.read_job(job.parse_env(text_of(name, SIBLING)), NAMES)
                for key in ('profile', 'actions', 'tests', 'gate_plan', 'gate_lengths', 'gate_max_tokens', 'cards'):
                    self.assertEqual(mine[key], theirs[key], key)
                self.assertEqual(theirs['tag'], 'tp4-seats8')
                self.assertEqual(mine['tag'], IMAGE)

    def test_every_named_test_is_one_the_smoke_knows_and_runs_in_the_order_the_template_lists_them(self):
        executed = self.executed()
        for name in ORDERED:
            listed = parsed(name)['tests'].split(',') if parsed(name)['tests'] else []
            for test in listed:
                self.assertIn(test, executed, '%s: %s' % (name, test))
            self.assertEqual([test for test in executed if test in listed], listed, name)

    def test_the_read_outs_name_the_rule_the_smoke_check_enforces(self):
        import c2_smoke_check as check
        for name in REPEATS:
            text = text_of(name)
            for word in ('request widths warmed before the packed traces', 'packed blocks capture block=0', 'RECORDED, not bounded',
                         'steady_resend', 'five consecutive completions', 'QWEN_FAST_STALL_BUILD_S=150', 'QWEN_FAST_M3_REQUEST_WARM=1'):
                self.assertIn(word, text, name)
        self.assertFalse(hasattr(check, 'FIRST_ENGINE_PROGRAMS_MAX'))
        control = text_of('R2-nowarm-control')
        for word in ('MUST still hang', 'EXPECTED result', 'QWEN_FAST_M3_REQUEST_WARM=0'):
            self.assertIn(word, control)
        rshard = text_of('R3-rshard')
        for word in ('ONLY if R1 hangs at the same fence', 'QWEN_FAST_REQUEST_SHARD_ARGMAX=1', 'QWEN_FAST_REQUEST_SHARD_AUDIT=1',
                     'kept the pinned sampler'):
            self.assertIn(word, rshard)
        with open(os.path.join(HERE, 'verifier_engine_tp.py'), encoding='utf-8') as handle:
            self.assertIn('request shard argmax kept the pinned sampler', handle.read())

    def test_every_tail_template_carries_this_images_note_and_no_stale_tag(self):
        for name in TAIL:
            text = text_of(name)
            with self.subTest(template=name):
                self.assertIn('ON THIS IMAGE (tp4-seats8-rwarm)', text)
                self.assertEqual(len(re.findall(r'^C2_IMAGE_TAG=', text, re.M)), 1)
                self.assertNotIn('C2_IMAGE_TAG=tp4-seats8' + chr(10), text)


class ImageTests(unittest.TestCase):
    def test_the_new_module_ships_in_the_image_and_its_tests_run_in_ci(self):
        root = os.path.dirname(os.path.dirname(HERE))
        with open(os.path.join(root, 'docker', 'qwen-c2-overlay.txt'), encoding='utf-8') as handle:
            self.assertIn('scripts/ci/request_width_warm.py', handle.read().splitlines())
        with open(os.path.join(root, '.github', 'workflows', 'qwen-integration-cpu.yml'), encoding='utf-8') as handle:
            workflow = handle.read()
        for module in ('test_request_width_warm', 'test_tp4_seats8_rwarm_window'):
            self.assertIn(module, workflow)


if __name__ == '__main__':
    unittest.main()
