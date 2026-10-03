"""The tp4-seats8-262k windows: their job templates (scripts/ci/references/tp4-262k8-jobs), their order and what each one reads.

Stage 1 is the card evidence on card M alone (E0, E1w, E1, E2w, E2a, E2b, E2xw, E2c on the I1 base image) and stage 2 the 262k window on the quad (SB9, B9, M9, L9a, L9b,
C9, T9, hand-back, on one image tp4-262k8-1 built from the commit that records stage 1). The order is B9 -> M9 -> L9a -> L9b -> T9 -> C9 -> T9e: MEMORY before the ladders, because
the pooled KV cache's size is a measurement. The templates are public, so they name no rig, card, host, address, registry or digest, and the only placeholders are the
one-card harnesses' (the served image, the K64j graft directory and its binary's digest)."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-262k8-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
PLACEHOLDERS = {'@SERVED_IMAGE@': 'tt-vllm:four-card-test', '@K64J_GRAFT_DIR@': '/opt/graft-K64j',
                '@K64J_TTNNCPP_SHA256@': 'ab' * 32}
BASE, IMAGE = 'tp4-serve-8', 'tp4-262k8-1'
AUDITS_ON = ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT')
PRODUCTION = 'c2-packed-tp4'
EIGHT, EIGHT_GATE, EIGHT_TIME, EIGHT_DIAG, FOUR_GATE = ('c2-packed-tp4-8x262k', 'c2-packed-tp4-8x262k-gate', 'c2-packed-tp4-8x262k-time-gate',
                                                        'c2-packed-tp4-8x262k-diag-strace', 'c2-packed-tp4-262k-gate')
TIME_131K = 'c2-packed-tp4-8-time-gate'
STAGE1 = ('A0e-agent-stop', 'E0-quad-reset', 'E1w-ordered-writer-watcher', 'E1-ordered-writer', 'E2w-card-watcher', 'E2a-cb1', 'E2b-cb2a',
          'E2xw-reader-watcher', 'E2c-cb2b', 'H1e-status-reset', 'A9e-agent-start')
STAGE2 = ('X0-status-reset', 'SB9-build', 'B9a-flag-off-smoke', 'B9b-262k-attach', 'M9a-cold262k', 'M9b-pool8', 'L9a-ladder4', 'L9b-ladder8',
          'T9a-131k-timed', 'T9b-262k-timed', 'T9c-131k-timed', 'T9d-262k-timed', 'C9-churn16', 'T9e-stall8-cold262k', 'Z-reset')
OPTIONAL = ('C9-churn16', 'T9a-131k-timed', 'T9b-262k-timed', 'T9c-131k-timed', 'T9d-262k-timed', 'T9e-stall8-cold262k')
ACTIONS = {'A0e-agent-stop': 'agentstop unserve', 'E0-quad-reset': 'reset', 'E1w-ordered-writer-watcher': 'cardm', 'E1-ordered-writer': 'cardm',
           'E2w-card-watcher': 'cardm', 'E2a-cb1': 'cardm', 'E2b-cb2a': 'cardm', 'E2xw-reader-watcher': 'cardm', 'E2c-cb2b': 'cardm',
           'H1e-status-reset': 'status reset', 'A9e-agent-start': 'agentstart', 'SB9-build': 'status build', 'X0-status-reset': 'status reset',
           'B9a-flag-off-smoke': 'status unserve reset smoke', 'B9b-262k-attach': 'reset smoke', 'M9a-cold262k': 'reset gate',
           'M9b-pool8': 'reset gate', 'L9a-ladder4': 'reset gate', 'L9b-ladder8': 'reset gate', 'C9-churn16': 'reset gate',
           'T9a-131k-timed': 'reset smoke', 'T9b-262k-timed': 'reset smoke', 'T9c-131k-timed': 'reset smoke', 'T9d-262k-timed': 'reset smoke',
           'T9e-stall8-cold262k': 'reset smoke', 'Z-reset': 'status reset'}
PROFILE_OF = {'B9a-flag-off-smoke': PRODUCTION, 'B9b-262k-attach': EIGHT_GATE, 'M9a-cold262k': EIGHT_TIME, 'M9b-pool8': EIGHT_TIME, 'L9a-ladder4': FOUR_GATE,
              'L9b-ladder8': EIGHT_GATE, 'C9-churn16': EIGHT_GATE, 'T9a-131k-timed': TIME_131K, 'T9b-262k-timed': EIGHT_TIME,
              'T9c-131k-timed': TIME_131K, 'T9d-262k-timed': EIGHT_TIME, 'T9e-stall8-cold262k': EIGHT_TIME}
SMOKE_B9A = ['warmup', 'coding', 'concurrent4_steady', 'steady_resend', 'replay_concurrent4', 'concurrent4_code_equal']
SMOKE_B9B = ['warmup', 'coding', 'concurrent8_code_equal', 'concurrent8_code', 'concurrent5_split', 'concurrent8_drain']
SMOKE_TIMED = ['warmup', 'coding', 'concurrent8_code', 'concurrent8_code_32k', 'concurrent8_code_128k']


def profiles():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)


def rows():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def order_text():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return handle.read()


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    text = text_of(name)
    for placeholder, value in PLACEHOLDERS.items():
        text = text.replace(placeholder, value)
    return job.read_job(job.parse_env(text), sorted(profiles()['profiles']), root=ROOT)


def values(name):
    return dict(line.split('=', 1) for line in text_of(name).splitlines() if line.strip() and not line.startswith('#'))


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns(self):
        found = rows()
        self.assertTrue(all(len(row) == 4 for row in found))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(sorted(row[0] for row in found), on_disk)
        self.assertEqual([row[0] for row in found], list(STAGE2 + STAGE1))

    def test_the_stages_run_on_their_images_and_the_modes_and_minutes(self):
        for name, mode, image, minutes in rows():
            with self.subTest(job=name):
                self.assertEqual(image, BASE if name in STAGE1 else IMAGE)
                self.assertEqual(parsed(name)['tag'], image)
                self.assertTrue(minutes.isdigit() and 5 <= int(minutes) <= 360, minutes)
                self.assertEqual(mode, 'deferred' if name in STAGE1 else 'pre' if name == 'SB9-build' else 'optional' if name in OPTIONAL else 'stop')

    def test_the_minutes_fit_the_workflows_step_and_job_budgets(self):
        literals = base_budgets()
        for name, mode, image, minutes in rows():
            with self.subTest(job=name):
                self.assertLessEqual(int(minutes), literals['step_minutes'], 'one job is one workflow step')
                self.assertLessEqual(int(minutes), literals['job_minutes'])

    def test_the_quad_window_is_x0_sb9_b9_m9_l9_t9_z_with_memory_before_the_ladders_and_the_deferred_jobs_after(self):
        names = [row[0] for row in rows()]
        self.assertEqual(names[:len(STAGE2)], list(STAGE2))
        self.assertEqual(names[len(STAGE2):], list(STAGE1))
        for earlier, later in (('X0-status-reset', 'SB9-build'), ('SB9-build', 'B9a-flag-off-smoke'), ('B9a-flag-off-smoke', 'B9b-262k-attach'),
                               ('B9b-262k-attach', 'M9a-cold262k'), ('M9a-cold262k', 'M9b-pool8'), ('M9b-pool8', 'L9a-ladder4'),
                               ('L9a-ladder4', 'L9b-ladder8'), ('L9b-ladder8', 'T9a-131k-timed'), ('T9d-262k-timed', 'Z-reset'),
                               ('Z-reset', 'A0e-agent-stop')):
            self.assertLess(names.index(earlier), names.index(later), (earlier, later))
        self.assertEqual(names[len(STAGE2) - 1], 'Z-reset')
        self.assertIn('B9 -> M9 -> L9a -> L9b -> T9 -> C9 (optional) -> T9e (optional)', order_text())

    def test_the_timing_pairs_alternate_131k_and_262k_abab(self):
        names = [row[0] for row in rows() if row[0].startswith('T9') and row[0] != 'T9e-stall8-cold262k']
        self.assertEqual([parsed(name)['profile'] for name in names], [TIME_131K, EIGHT_TIME, TIME_131K, EIGHT_TIME])
        self.assertEqual({parsed(name)['tests'] for name in names}, {','.join(SMOKE_TIMED)})

    def test_the_quad_window_has_no_agentstop_no_agentstart_and_no_hand_back_and_the_build_is_card_free(self):
        for name in STAGE2:
            with self.subTest(job=name):
                actions = parsed(name)['actions'].split()
                self.assertNotIn('agentstop', actions)
                self.assertNotIn('agentstart', actions)
        for gone in ('A0-agent-stop', 'A9-agent-start', 'H9-handback-reset'):
            self.assertFalse(os.path.exists(os.path.join(FOLDER, gone + '.env')), gone)
        self.assertEqual(parsed('X0-status-reset')['actions'], 'status reset')
        self.assertEqual(parsed('Z-reset')['actions'], 'status reset')
        self.assertEqual(parsed('SB9-build')['actions'], 'status build')
        self.assertNotIn('reset', parsed('SB9-build')['actions'].split())
        for name in STAGE1:
            self.assertIn('DEFERRED', text_of(name), name)

    def test_the_order_names_what_decides_and_what_is_never_placed(self):
        text = order_text()
        for words in ('KILL SIGNAL', 'WHAT THE WINDOW DECIDES', 'Nothing is moved by it', 'TODAY\'S PRODUCTION RECIPE', 'never a gate',
                      'never a 262k placement by an environment variable', 'PAIRED PER ROUND', '1.03', 'NOT_EXERCISED', 'tp4-262k8-2',
                      '--capacity 262144', 'scripts/ci/__pycache__ is tracked', '4.05 GB', 'QWEN_FAST_262K_EVIDENCE_WAIVER', '262k evidence WAIVED (gate-only)', 'NO agentstop, NO agentstart and NO hand-back', 'UNQUALIFIED', 'a SEPARATE build', 'No other window\'s driver may be alive'):
            self.assertIn(words, text)


def base_budgets():
    import test_c2_serving_gate as base

    return base.WorkflowTests.budget_literals()


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_is_lf_and_names_no_card_host_address_registry_or_digest(self):
        for name in STAGE2 + STAGE1:
            with self.subTest(template=name):
                parsed(name)
                with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                    self.assertNotIn(b'\r', handle.read(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)

    def test_only_the_one_card_harnesses_name_placeholders_and_they_are_the_three(self):
        for name in STAGE2 + STAGE1:
            body = chr(10).join(line for line in text_of(name).splitlines() if not line.startswith('#'))
            found = set(re.findall(r'@[A-Z0-9_]+@', body))
            if name in ('E1w-ordered-writer-watcher', 'E1-ordered-writer', 'E2xw-reader-watcher', 'E2c-cb2b'):
                self.assertEqual(found, set(PLACEHOLDERS), name)
            elif name in ('E2w-card-watcher', 'E2a-cb1', 'E2b-cb2a'):
                self.assertEqual(found, {'@K64J_GRAFT_DIR@', '@K64J_TTNNCPP_SHA256@'}, name)
            else:
                self.assertEqual(found, set(), name)

    def test_the_actions_are_the_plans_and_every_card_job_resets_first(self):
        for name, actions in ACTIONS.items():
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['actions'], actions)
        for name in STAGE2 + ('E0-quad-reset',):
            actions = parsed(name)['actions'].split()
            if set(actions) & set(('smoke', 'gate', 'replay')):
                self.assertIn('reset', actions, 'links train only at board init: a four-card job follows a reset')
                self.assertEqual(parsed(name)['cards'], 'quad')

    def test_stage_one_is_one_card_work_on_card_m_and_the_rest_is_quad(self):
        for name in STAGE1:
            expected = 'pair' if name.startswith('E') and name != 'E0-quad-reset' else 'quad'
            self.assertEqual(parsed(name)['cards'], expected, name)
        for name in STAGE2:
            self.assertEqual(parsed(name)['cards'], 'quad', name)

    def test_each_serving_job_names_its_profile_and_every_262k_profile_exists_with_its_flags(self):
        known = profiles()['profiles']
        for name, profile in PROFILE_OF.items():
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['profile'], profile)
                self.assertIn(profile, known)
        for profile in (EIGHT, EIGHT_GATE, EIGHT_TIME, EIGHT_DIAG):
            self.assertEqual(known[profile]['engine']['max-model-len'], 262144)
            self.assertEqual(known[profile]['env']['QWEN_FAST_KV_RESERVATION'], '1')
        self.assertEqual(known[FOUR_GATE]['engine']['num-gpu-blocks-override'], 4 * 4096)

    def test_no_template_bakes_a_default(self):
        for name in STAGE2 + STAGE1:
            self.assertEqual(parsed(name)['bake_default_profile'], '', name)
            self.assertNotIn('C2_BAKE_DEFAULT_PROFILE', chr(10).join(line for line in text_of(name).splitlines() if not line.startswith('#')))

    def test_the_smoke_jobs_name_the_tests_and_the_deep_and_stall_arms_are_real(self):
        self.assertEqual(parsed('B9a-flag-off-smoke')['tests'].split(','), SMOKE_B9A)
        self.assertEqual(parsed('B9b-262k-attach')['tests'].split(','), SMOKE_B9B)
        self.assertEqual(parsed('T9e-stall8-cold262k')['tests'].split(','), ['warmup', 'stall8_cold262k'])
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        for test in set(SMOKE_B9A + SMOKE_B9B + SMOKE_TIMED + ['stall8_cold262k']):
            self.assertIn("'%s'" % test, smoke, test + ' is not a smoke test')
        self.assertEqual(parsed('B9a-flag-off-smoke')['tests'].split(','), SMOKE_B9A, 'S8-0\'s list')

    def test_the_gate_jobs_carry_the_ladders_the_churn_and_the_memory_keys(self):
        self.assertEqual(parsed('L9a-ladder4')['gate_plan'], 'matrix')
        self.assertEqual(parsed('L9a-ladder4')['gate_lengths'], ','.join(str(item) for item in gate.LADDER4_262K))
        self.assertEqual(parsed('L9b-ladder8')['gate_lengths'], ','.join(str(item) for item in gate.LADDER8_262K))
        for name in ('L9a-ladder4', 'L9b-ladder8'):
            self.assertEqual(parsed(name)['gate_max_tokens'], '256')
            self.assertEqual(values(name)['C2_GATE_JIT'], 'record')
        self.assertEqual(parsed('C9-churn16')['gate_plan'], 'churn')
        self.assertEqual(parsed('C9-churn16')['gate_lengths'], ','.join(str(item) for item in gate.CHURN_LENGTHS_262K))
        self.assertEqual(parsed('C9-churn16')['gate_max_tokens'], '1024')
        self.assertEqual(len(parsed('C9-churn16')['gate_lengths'].split(',')), 16)
        self.assertEqual((parsed('M9a-cold262k')['gate_plan'], parsed('M9a-cold262k')['gate_memory_prompt'], parsed('M9a-cold262k')['gate_memory_users']),
                         ('memory', '253920', '1'))
        self.assertEqual((parsed('M9b-pool8')['gate_plan'], parsed('M9b-pool8')['gate_memory_prompt'], parsed('M9b-pool8')['gate_memory_users']),
                         ('memory', '157000', ''))

    def test_every_gate_job_plans_on_its_profile_without_a_plan_error(self):
        known = profiles()
        for name in ('L9a-ladder4', 'L9b-ladder8', 'C9-churn16', 'M9a-cold262k', 'M9b-pool8'):
            outputs = parsed(name)
            lengths = [int(part) for part in outputs['gate_lengths'].split(',')] if outputs['gate_lengths'] else None
            with self.subTest(job=name):
                arms = gate.plan_arms(outputs['gate_plan'], outputs['profile'], known, lengths=lengths, max_tokens=int(outputs['gate_max_tokens']),
                                      memory_prompt=int(outputs['gate_memory_prompt']) if outputs['gate_memory_prompt'] else None,
                                      memory_users=int(outputs['gate_memory_users']) if outputs['gate_memory_users'] else None)
                self.assertTrue(arms)
                budget = base_budgets()['step']
                self.assertLessEqual(gate.worst_case_seconds([outputs['gate_plan']], {outputs['gate_plan']: arms}), budget, name)

    def test_the_memory_jobs_arithmetic_is_the_plans(self):
        import serving_kv_reservation as kv

        self.assertEqual(8 * kv.request_blocks(157000, 16384), 21688)
        self.assertEqual(sum(kv.request_blocks(length, 1024) for length in gate.CHURN_LENGTHS_262K[:6]), 21384)
        self.assertEqual(sum(kv.request_blocks(length, 256) for length in gate.LADDER8_262K), 15135)

    def test_z_is_last_resets_and_places_nothing(self):
        text = text_of('Z-reset')
        self.assertIn('places NOTHING', text)
        self.assertIn('no hand-back', text)
        self.assertIn('a SEPARATE build that bakes it', text)
        self.assertIn('A9e', order_text())

    def test_the_quad_jobs_run_gate_only_profiles_that_carry_the_waiver_and_the_audits_where_correctness_is_read(self):
        known = profiles()['profiles']
        for name in STAGE2:
            profile = parsed(name)['profile']
            if profile in (EIGHT, ''):
                self.fail('%s runs a traffic or no profile' % name)
        for name in ('B9b-262k-attach', 'L9a-ladder4', 'L9b-ladder8', 'C9-churn16'):
            env = known[parsed(name)['profile']]['env']
            self.assertIs(known[parsed(name)['profile']]['gate_only'], True)
            self.assertEqual({key: env.get(key) for key in AUDITS_ON}, {key: '1' for key in AUDITS_ON}, name)
            self.assertEqual(env['QWEN_FAST_GDN_PREFILL_CONV_AUDIT'], '4')
            self.assertEqual(env['QWEN_FAST_262K_EVIDENCE_WAIVER'], '1')
        for name in ('M9a-cold262k', 'M9b-pool8', 'T9b-262k-timed', 'T9d-262k-timed', 'T9e-stall8-cold262k'):
            env = known[parsed(name)['profile']]['env']
            self.assertIs(known[parsed(name)['profile']]['gate_only'], True)
            self.assertEqual(env['QWEN_FAST_262K_EVIDENCE_WAIVER'], '1')
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))
        self.assertNotIn('QWEN_FAST_262K_EVIDENCE_WAIVER', known[EIGHT]['env'])

    def test_the_build_is_card_free_and_says_what_to_read(self):
        text = text_of('SB9-build')
        for words in ('no card opened', 'No bake', 'provenance report', 'serving_kv_reservation', 'Never run it during a timing job'):
            self.assertIn(words, text)

    def test_e1_and_e2_name_their_recorders_and_kill_signals(self):
        self.assertIn('record_ordered_writer_evidence_tp4.py', text_of('E1-ordered-writer'))
        self.assertIn('--capacity 262144', text_of('E2a-cb1'))
        self.assertIn('blocks 262k', text_of('E2b-cb2a'))
        for name in ('E1-ordered-writer', 'E2a-cb1', 'E2b-cb2a', 'E2c-cb2b'):
            self.assertIn('blocks 262k', text_of(name))
        self.assertIn('262144', values('E2c-cb2b')['C2_CARDM_ARGS'])
        self.assertIn('--r2-named 256,2304,16640,65792,131328,262144', values('E2c-cb2b')['C2_CARDM_ARGS'])
        self.assertIn('--extents 2304,16896,33024,65792,98560,131328,196864,262144', values('E2a-cb1')['C2_CARDM_ARGS'])
        self.assertIn('--cb2-extents 2304,4352,16640,65792,131328,262144', values('E2b-cb2a')['C2_CARDM_ARGS'])
        self.assertIn('--widths 2052,4096 --writers chained64,tiles32 --seeds 0,1,2', values('E1-ordered-writer')['C2_CARDM_ARGS'])

    def test_the_cardm_jobs_pass_the_servers_checks_and_the_runner_dry_runs_them(self):
        for name in STAGE1:
            if parsed(name)['actions'] != 'cardm':
                continue
            with self.subTest(job=name):
                outputs = parsed(name)
                self.assertEqual(outputs['cardm_harness'], 'optimisation/ttnn-op/k64j/run_card_b.sh')
                env = dict(pair.split('=', 1) for pair in outputs['cardm_env'].split())
                self.assertIn(env['K64J_HARNESS'], ('card', 'extent_reader', 'ordered_writer'))
                self.assertIn('EXPECT_TTNNCPP_SHA256', env)


if __name__ == '__main__':
    unittest.main()
