"""The tp4/next-3-scope window: its job templates (scripts/ci/references/tp4-next3-scope-jobs), their order, the one profile they add, and the two design documents.

The window times lever P1 (drafter K/V publication as device traces: the fused commit at four cards) on the production recipe. S0 takes production down (the first job
of a window is `agentstop unserve`), S1 smokes the production profile, S2 and S3 are the audited smokes of the fused commit, S4a..S4e five audits-off runs of the hang shapes on the timed
arm, S5a..S5d the paired timing ABAB, S6 the hand-back (reset, fabric re-measure, node agent started; the deploy is the owner's and is in no template). The templates are public, so they name no rig,
card, address, registry or digest."""

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
FOLDER = HERE / 'references' / 'tp4-next3-scope-jobs'
with open(HERE / 'qwen_c2_profiles.json', encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)['profiles']
NAMES = sorted(PROFILES)
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@|[A-Za-z]:[\\/]Users')
IMAGE = 'tp4-next-3s'
CONTROL, ARM, PRODUCTION = 'c2-packed-tp4-speed-strace', 'c2-packed-tp4-speed-strace-fcommit', 'c2-packed-tp4'
FUSED = ('QWEN_FAST_FUSED_COMMIT', 'QWEN_FAST_FUSED_COMMIT_INPLACE', 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS')
HANG_RUNS = tuple('S4%s-hang-shapes-fcommit-strace' % letter for letter in 'abcde')
TIMED = ('S5a-speed-timed', 'S5b-fcommit-timed', 'S5c-speed-timed', 'S5d-fcommit-timed')
STOP_JOBS = ('S0-unserve', 'S1-production-smoke', 'S2-fcommit-audited-smoke', 'S3-fcommit-live-audited-smoke') + HANG_RUNS
ORDERED = STOP_JOBS + TIMED + ('S6-handback-reset',)
PROFILE_OF = {'S0-unserve': PRODUCTION, 'S1-production-smoke': PRODUCTION, 'S2-fcommit-audited-smoke': 'c2-packed-tp4-gate-fcommit',
              'S3-fcommit-live-audited-smoke': 'c2-packed-tp4-gate-fcommit-live', **{name: ARM for name in HANG_RUNS},
              'S5a-speed-timed': CONTROL, 'S5b-fcommit-timed': ARM, 'S5c-speed-timed': CONTROL, 'S5d-fcommit-timed': ARM,
              'S6-handback-reset': PRODUCTION}
HANG_SHAPES = ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_v164order', 'concurrent4_steady', 'replay_concurrent4']
TIMING_TESTS = ['warmup', 'coding', 'concurrent4_code_equal', 'concurrent4_code_32k']
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
        on_disk = sorted(path.stem for path in FOLDER.glob('*.env'))
        ordered = [row[0] for row in rows]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(ordered, list(ORDERED))

    def test_the_modes_images_and_minutes(self):
        modes = {row[0]: row[1] for row in read_order()}
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertIn(mode, ('stop', 'optional'))
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 15 <= int(minutes) <= 180, minutes)
        for name in STOP_JOBS + ('S6-handback-reset',):
            self.assertEqual(modes[name], 'stop', name)
        for name in TIMED:
            self.assertEqual(modes[name], 'optional', name)

    def test_the_first_job_is_agentstop_unserve_and_nothing_else_takes_production_down(self):
        self.assertEqual(read_order()[0][0], 'S0-unserve')
        self.assertEqual(parsed('S0-unserve')['actions'], 'status agentstop unserve')
        for name in ORDERED[1:]:
            actions = parsed(name)['actions'].split()
            self.assertNotIn('agentstop', actions, name)
            self.assertNotIn('unserve', actions, name)
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('PRODUCTION IS LIVE ON THE CARDS', 'FIRST job is S0', 'agentstop'):
            self.assertIn(word, order)

    def test_the_hand_back_is_last_runs_even_after_a_stop_and_says_the_owner_deploys(self):
        self.assertEqual(read_order()[-1][0], 'S6-handback-reset')
        outputs = parsed('S6-handback-reset')
        self.assertEqual(outputs['actions'], 'status reset fabric agentstart')
        self.assertEqual(outputs['cards'], 'quad')
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('runs even when a stop job halts the window', 'the all-four reset', 'fabric re-measure', 'agentstart', 'OWNER runs /deploy',
                     'never include the deploy'):
            self.assertIn(word, order)
        text = text_of('S6-handback-reset')
        for word in ("TODAY'S PRODUCTION", 'NEVER place a gate arm', 'OWNER', '/deploy', 'never part of this job file', 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE=1'):
            self.assertIn(word, text)
        for name in ORDERED:
            self.assertNotIn('push', parsed(name)['actions'].split(), 'no job pushes or places anything')
            self.assertNotIn('replay', parsed(name)['actions'].split(), name)

    def test_the_order_names_the_precondition_the_pairing_and_the_scope(self):
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('NOTHING IN THIS WINDOW HAS RUN ON A CARD', 'FIVE audits-off runs', 'five consecutive completions', 'PAIRED per round', 'ABAB',
                     'V5', 'NO job here', 'QWEN_FAST_TRACED_PUBLISH', 'OFF in every profile', 'one-card two-head in-place slide proof'):
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
        for name in ORDERED[1:-1]:
            actions = parsed(name)['actions'].split()
            with self.subTest(template=name):
                self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertLess(actions.index('reset'), actions.index([step for step in actions if step in DEVICE_STEPS][0]))
                self.assertEqual(actions, ['reset', 'smoke'])

    def test_the_profiles_are_the_plans_and_only_the_production_ones_are_traffic_profiles(self):
        for name, profile in PROFILE_OF.items():
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['profile'], profile)
                self.assertIn(profile, PROFILES)
                self.assertEqual(parsed(name)['gate_plan'], 'bringup')
        for name in ('S0-unserve', 'S1-production-smoke', 'S6-handback-reset'):
            self.assertNotIn('gate_only', PROFILES[parsed(name)['profile']], name)
        for name in ORDERED[2:-1]:
            self.assertIs(PROFILES[parsed(name)['profile']].get('gate_only'), True, name)

    def test_the_hang_runs_are_five_templates_on_the_hang_shapes_audits_off(self):
        self.assertEqual(len(HANG_RUNS), 5)
        for name in HANG_RUNS:
            self.assertEqual(parsed(name)['tests'].split(','), HANG_SHAPES, name)
            env = env_of(parsed(name)['profile'])
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'), name)
            self.assertEqual(env['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'], '1', name)
            self.assertEqual(env['QWEN_FAST_BUDGET_CAP'], '1', name)
            self.assertIn('FIVE CONSECUTIVE COMPLETIONS', text_of(name))

    def test_the_timing_is_an_abab_of_the_control_and_the_arm_on_the_same_tests(self):
        names = [row[0] for row in read_order()]
        self.assertEqual(names[names.index(TIMED[0]):][:4], list(TIMED), 'A B A B, back to back')
        self.assertEqual([parsed(name)['profile'] for name in TIMED], [CONTROL, ARM, CONTROL, ARM])
        for name in TIMED:
            self.assertEqual(parsed(name)['tests'].split(','), TIMING_TESTS, name)
            self.assertIn('speed_window_compare.py', text_of(name))
            self.assertIn('PAIRED', text_of(name))

    def test_the_audited_arms_are_the_ones_the_fused_commit_family_defines(self):
        for name in ('c2-packed-tp4-gate-fcommit', 'c2-packed-tp4-gate-fcommit-live'):
            env = env_of(name)
            self.assertEqual((env['QWEN_FAST_FUSED_COMMIT'], env['QWEN_FAST_FUSED_COMMIT_AUDIT']), ('1', '1'), name)
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'), name)

    def test_every_named_smoke_test_is_one_the_smoke_runs_in_the_order_the_template_lists_them(self):
        smoke = (HERE / 'c2_serving_smoke.py').read_text(encoding='utf-8')
        executed = re.findall(r"^\s*record\('([a-z0-9_]+)'", smoke, re.M)
        for name in ORDERED[1:-1]:
            listed = parsed(name)['tests'].split(',')
            with self.subTest(template=name):
                for test in listed:
                    self.assertIn("'%s'" % test, smoke)
                self.assertEqual([test for test in executed if test in listed], listed)


class ProfileTests(unittest.TestCase):
    def test_the_arm_is_the_production_recipe_plus_the_three_fused_commit_flags_and_nothing_else(self):
        control, arm = PROFILES[CONTROL], PROFILES[ARM]
        self.assertEqual({key for key in set(control['env']) | set(arm['env']) if control['env'].get(key) != arm['env'].get(key)}, set(FUSED))
        for flag in FUSED:
            self.assertEqual((control['env'][flag], arm['env'][flag]), ('0', '1'), flag)
        for key in set(control) | set(arm):
            if key not in ('env', 'description'):
                self.assertEqual(control.get(key), arm.get(key), key)
        self.assertIs(arm['gate_only'], True)
        self.assertTrue(arm['description'].startswith('GATE ONLY'))
        self.assertIn('NOT QUALIFIED', arm['description'])
        self.assertEqual(arm['env']['QWEN_C2_GATE_PROFILE'], '1')
        # production is the control less the gate marker (test_c2_packed_tp4_profiles): so the arm is production plus the three flags
        prod = dict(PROFILES[PRODUCTION]['env'], QWEN_C2_GATE_PROFILE='1')
        self.assertEqual({key for key in set(prod) | set(arm['env']) if prod.get(key) != arm['env'].get(key)}, set(FUSED))

    def test_the_arm_keeps_the_slide_the_hang_fix_the_caps_and_the_pair_draft(self):
        env = env_of(ARM)
        self.assertEqual(env['QWEN_FAST_TP_KV_SLIDE'], '1', 'the fused commit is the four-card slide scope')
        self.assertEqual(env['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'], '1')
        self.assertEqual((env['QWEN_FAST_BUDGET_CAP'], env['QWEN_FAST_SEQ_DEADLINE_S']), ('1', '120'))
        self.assertEqual(env['QWEN_FAST_QUAD_DRAFT'], '0')
        self.assertNotIn('QWEN_FAST_FUSED_COMMIT_AUDIT', env, 'a timed arm carries no audit')

    def test_only_the_arm_changed_in_the_profile_file_and_production_is_untouched(self):
        env = env_of(PRODUCTION)
        for flag in FUSED:
            self.assertEqual(env[flag], '0', flag)
        self.assertEqual(env['QWEN_FAST_TRACED_PUBLISH'], '0')

    def test_traced_publish_stays_off_in_every_profile_that_names_it(self):
        named = [name for name, profile in PROFILES.items() if 'QWEN_FAST_TRACED_PUBLISH' in profile['env']]
        self.assertGreater(len(named), 10)
        for name in named:
            self.assertEqual(env_of(name)['QWEN_FAST_TRACED_PUBLISH'], '0', name)


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.publish = (ROOT / 'docs' / 'tp4-traced-publish.md').read_text(encoding='utf-8')
        self.split = (ROOT / 'docs' / 'tp4-recurrence-split.md').read_text(encoding='utf-8')

    def test_the_documents_are_public_safe_and_lf(self):
        for text in (self.publish, self.split):
            self.assertNotIn('\r', text)
            self.assertIsNone(BANNED.search(text))

    def test_the_traced_publish_document_answers_what_the_flag_is_and_names_the_hazards(self):
        for word in ('captures nothing', 'fused commit', 'tp_slide_live', 'bank_shape', 'unrecognized_prepare', 'C7', 'allocated after a capture',
                     'request-engine build', 'trace region', 'G3', 'speed-strace-fcommit', 'OFF in every profile'.lower()):
            self.assertIn(word.lower(), self.publish.lower(), word)

    def test_the_recurrence_document_stops_at_the_design_and_lists_the_files(self):
        for word in ('Design only', 'not a host-side change', 'static_assert', 'gdn_seq_block_split.py', 'gdn_seq_block_split_reader.cpp',
                     'gdn_user_batch_conv.py', 'Do not edit', '630,784', '96 cores', 'owner', 'helper'):
            self.assertIn(word, self.split, word)

    def test_the_files_the_recurrence_document_says_not_to_edit_exist(self):
        for name in ('gdn_seq_block.py', 'gdn_seq_block_reader.cpp', 'gdn_seq_block_writer.cpp', 'gdn_seq_block_compute.cpp', 'gdn_user_batch.py',
                     'gdn_multitoken.py', 'gdn_vsplit.py', 'gdn_user_batch_conv.py', 'tp_shapes.py', 'gdn_user_batch_tp.py'):
            self.assertTrue((HERE / name).is_file(), name)

    def test_the_split_placement_the_document_relies_on_exists_today(self):
        """The 96-core placement is core_shares(workers=24 per user), already supported by the four-card batch module (no kernel needed for that part)."""
        previous = os.environ.get('QWEN_FAST_TP')
        os.environ['QWEN_FAST_TP'] = '4'
        try:
            import gdn_user_batch_tp as batch

            shares = batch.core_shares(11, 10, 4, workers=24)
            points = [point for share in shares for point in share]
            self.assertEqual((len(points), len(set(points))), (96, 96))
            self.assertLess(max(point[0] for point in points), 11)
            for share in shares:
                for first, second in zip(share[0::2], share[1::2]):
                    self.assertEqual((first[0], first[1] + 1), second, 'a head pair is two adjacent cores of one grid column')
            self.assertEqual(len([p for share in batch.core_shares(11, 10, 4) for p in share]), 48)
        finally:
            if previous is None:
                os.environ.pop('QWEN_FAST_TP', None)
            else:
                os.environ['QWEN_FAST_TP'] = previous

    def test_the_circular_buffer_budget_in_the_document_is_the_planned_one(self):
        import gdn_seq_block

        self.assertEqual(gdn_seq_block.cb_bytes(), 630784)
        allocatable = 1572864 - 111488 - 24576
        self.assertEqual(allocatable, 1436800)
        self.assertEqual(allocatable - gdn_seq_block.cb_bytes(), 806016)
        for number in ('1,572,864', '111,488', '24,576', '1,436,800', '806,016'):
            self.assertIn(number, self.split)


if __name__ == '__main__':
    unittest.main()
