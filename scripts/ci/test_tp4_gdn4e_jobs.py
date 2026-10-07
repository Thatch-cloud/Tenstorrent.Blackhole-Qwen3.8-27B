"""The exact-GDN-memory card-window pack (scripts/ci/references/tp4-gdn4e-jobs): every template parses with c2_serving_job.py against the real
profiles, the order file is consistent (one image, four columns, the dependencies name real jobs), the arithmetic the READ rules quote is the
module's, and the public templates name no rig, card, host, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import gdn_shared_history as shared  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-gdn4e-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
IMAGE = 'tp4-gdn4e-1'
BASE = 'c2-packed-tp4-8x262k-ship-prefix-4e'
CONTROL, ARM, PLAIN, GROW = BASE + '-control-audit', BASE + '-audit', BASE, BASE + '-grow'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
ORDERED = ('X0-status-rescan-reset', 'B0-build', 'S0c-control-attach-smoke', 'S0-audited-attach-smoke', 'H1-hang-shapes-4e', 'H2-hang-shapes-4e',
           'T1-control-timed', 'T2-4e-timed', 'T3-control-timed', 'T4-4e-timed', 'E1-ladder8-exactness', 'A5-admission5-grow',
           'A5c-admission5-control', 'Z-reset')
TIMED = ('T1-control-timed', 'T2-4e-timed', 'T3-control-timed', 'T4-4e-timed')
SOFT = TIMED + ('A5c-admission5-control', 'E1-ladder8-exactness', 'Z-reset')
SHIP = 'c2-packed-tp4-8x262k-ship-prefix'
SMOKE = 'warmup,coding,concurrent8_steady,concurrent8_code_32k,concurrent8_code_equal'


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def order_text():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return handle.read()


def read_order():
    return [line.split() for line in order_text().splitlines() if line.strip() and not line.startswith('#')]


class Pack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(PROFILES_PATH, encoding='utf-8') as handle:
            cls.data = json.load(handle)
        cls.profiles = sorted(cls.data['profiles'])
        cls.meshes = job.profile_meshes(PROFILES_PATH)
        cls.root = os.path.dirname(os.path.dirname(HERE))

    def parsed(self, name):
        return job.read_job(job.parse_env(text_of(name)), self.profiles, root=self.root, meshes=self.meshes)

    def test_every_template_is_in_the_order_once_with_four_columns_and_the_one_image(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        self.assertEqual([row[0] for row in rows], list(ORDERED))
        self.assertEqual(sorted(row[0] for row in rows), sorted(n[:-4] for n in os.listdir(FOLDER) if n.endswith('.env')))
        for name, mode, image, minutes in rows:
            self.assertEqual(image, IMAGE)
            self.assertTrue(minutes.isdigit() and 4 <= int(minutes) <= 180, minutes)
            want = 'pre' if name == ORDERED[0] else 'soft' if name in SOFT else 'stop'
            self.assertEqual(mode, want, name)

    def test_the_total_in_the_order_header_is_the_sum_of_its_lines(self):
        total = sum(int(row[3]) for row in read_order())
        self.assertEqual(total, 505)
        self.assertIn('4+22+40+40+15+15+35+35+35+35+45+90+90+4 = 505 min = 8.4 h', order_text())
        self.assertEqual(round(total / 60, 1), 8.4)
        rows = read_order()
        hard = sum(int(row[3]) for row in rows if row[1] in ('stop', 'pre'))
        self.assertEqual(hard, 226)
        self.assertEqual(hard - 22, 204)
        self.assertIn('226 min = 3.8 h, 204 min = 3.4 h without B0', order_text())
        self.assertEqual(sum(int(row[3]) for row in rows if row[1] == 'soft'), 279)
        self.assertEqual((round(226 / 60, 1), round(204 / 60, 1)), (3.8, 3.4))

    def test_the_minutes_cite_the_measured_times_and_budget_two_attaches_for_the_two_arm_plans(self):
        with open(os.path.join(HERE, 'references', 'tp4-ship-262k-prefix-jobs', 'ORDER.txt'), encoding='utf-8') as handle:
            measured = handle.read()
        self.assertIn('status+rescan+reset 4, image build 22', measured)
        rows = {row[0]: int(row[3]) for row in read_order()}
        self.assertEqual((rows['X0-status-rescan-reset'], rows['B0-build']), (4, 22))
        for name in ('A5-admission5-grow', 'A5c-admission5-control'):
            arms = [arm[0] for arm in self.plan_arms(name)]
            self.assertEqual(arms, ['memory-concurrent', 'memory-short'], name)
            self.assertGreaterEqual(rows[name], len(arms) * 18 + 45, 'one attach per arm and the five cold prefills')
        for name in TIMED:
            self.assertGreaterEqual(rows[name], 18 + 15, name)

    def test_the_dependencies_name_real_jobs(self):
        names = set(ORDERED)
        needs = [line for line in order_text().splitlines() if line.startswith('# NEEDS ') and '<-' in line]
        self.assertEqual(len(needs), 4)
        for line in needs:
            left, right = line[len('# NEEDS '):].split('<-')
            for word in left.split() + right.split():
                self.assertTrue(any(name == word or name.startswith(word + '-') for name in names), (line, word))

    def test_the_kill_rules_name_this_pack_not_the_window(self):
        order = order_text()
        self.assertIn('a failure ends THIS pack (Z still runs)', order)
        self.assertIn('only job whose failure may end the whole window', order)
        for name in ORDERED:
            text = text_of(name)
            self.assertNotRegex(text, r'(stops|ends) the window', name)
            if name != 'X0-status-rescan-reset' and 'Stop rule' in text:
                self.assertIn('ends THIS pack (Z still runs', text, name)
        self.assertIn('WHOLE combined window', text_of('X0-status-rescan-reset'))

    def test_the_stop_line_that_can_fail_on_the_growth_runs_after_the_exactness_ladder_and_the_contrast_needs_it(self):
        names = [row[0] for row in read_order()]
        self.assertLess(names.index('E1-ladder8-exactness'), names.index('A5-admission5-grow'))
        self.assertLess(names.index('A5-admission5-grow'), names.index('A5c-admission5-control'))
        self.assertIn('# NEEDS A5c <- A5', order_text())
        self.assertIn('cutting A5c forfeits the growth proposal', text_of('A5c-admission5-control'))
        self.assertIn('forfeits the growth proposal', order_text())

    def plan_arms(self, name):
        import c2_serving_gate as gate

        out = self.parsed(name)
        lengths = [int(part) for part in out['gate_lengths'].split(',')] if out['gate_lengths'] else None
        tokens = int(out['gate_max_tokens']) if out['gate_max_tokens'] else job.DEFAULT_MAX_TOKENS
        prompt = int(out['gate_memory_prompt']) if out['gate_memory_prompt'] else None
        users = int(out['gate_memory_users']) if out['gate_memory_users'] else None
        return gate.plan_arms(out['gate_plan'], out['profile'], self.data, lengths, tokens, prompt, memory_users=users)

    def test_every_gate_template_passes_the_gates_own_plan_validation(self):
        gates = [name for name in ORDERED if 'gate' in self.parsed(name)['actions'].split()]
        self.assertEqual(gates, ['E1-ladder8-exactness', 'A5-admission5-grow', 'A5c-admission5-control'])
        for name in gates:
            with self.subTest(job=name):
                self.assertTrue(self.plan_arms(name))

    def test_the_ladder_has_four_rungs_above_131072_and_none_over_what_the_profile_admits(self):
        import serving_c2_contract as contract

        out = self.parsed('E1-ladder8-exactness')
        lengths = [int(part) for part in out['gate_lengths'].split(',')]
        self.assertEqual(len(lengths), 8)
        self.assertEqual(len([length for length in lengths if length > 131072]), 4)
        room = contract.request_limits(dict(self.data['profiles'][ARM], name=ARM))['max_prompt_tokens']
        self.assertEqual(room, 253920)
        self.assertLessEqual(max(lengths), room)
        self.assertIn('four rungs above 131,072', text_of('E1-ladder8-exactness'))

    def test_the_timing_arms_are_paired_abab_and_differ_by_the_flag_alone_at_run_time(self):
        import test_gdn_shared_history as flags

        profiles = [self.parsed(name)['profile'] for name in TIMED]
        self.assertEqual(profiles, [SHIP, PLAIN, SHIP, PLAIN])
        for name in TIMED:
            out = self.parsed(name)
            self.assertEqual(out['tests'], 'warmup,coding,concurrent8_steady,concurrent8_code_equal')
            self.assertNotIn('gate', out['actions'].split())
            text = text_of(name)
            self.assertIn('PAIRED per round', text)
            self.assertIn('1.5 ms', text)
        arm, control = flags.effective_environment(PLAIN), flags.effective_environment(SHIP)
        self.assertEqual({key for key in {*arm, *control} if arm.get(key) != control.get(key)}, {shared.FLAG})
        self.assertIn('1.5 ms', order_text())
        self.assertIn('memory-only', order_text())

    def test_the_read_rules_expect_the_flush_the_image_schedule_produces(self):
        import early_draft

        order = order_text()
        self.assertIn('site=shared-history', order)
        for name in ('S0-audited-attach-smoke', 'H1-hang-shapes-4e', 'H2-hang-shapes-4e'):
            text = text_of(name)
            self.assertNotIn("NO 'flush' line", text, name)
            self.assertNotIn('no refused or flush line', text, name)
            self.assertIn('shared-history', text, name)
        self.assertIn(shared.FLUSH_SITE, early_draft.IN_STEP_SITES)
        self.assertNotIn('costs 0 ms', order)
        self.assertNotIn('no profile in this pack does that', order)

    def test_every_env_parses_with_the_job_reader(self):
        for name in ORDERED:
            with self.subTest(job=name):
                out = self.parsed(name)
                self.assertEqual(out['tag'], IMAGE)
                self.assertEqual(out['cards'], 'quad')

    def test_no_agent_actions_and_only_the_build_builds(self):
        for name in ORDERED:
            actions = self.parsed(name)['actions'].split()
            self.assertFalse({'agentstop', 'agentstart', 'platform', 'unserve', 'push', 'replay'} & set(actions), name)
            self.assertEqual('build' in actions, name == 'B0-build', name)
            if name not in ('X0-status-rescan-reset', 'B0-build', 'Z-reset'):
                self.assertIn('reset', actions, 'each job follows its own all-four reset: ' + name)

    def test_the_four_profiles_exist_gate_only_and_differ_as_the_read_rules_say(self):
        profiles = self.data['profiles']
        for name in (CONTROL, ARM, PLAIN, GROW):
            self.assertTrue(profiles[name].get('gate_only'), name)
            self.assertEqual(profiles[name]['mesh_device'], 'P150x4')
            self.assertEqual(profiles[name]['engine']['max-num-seqs'], 8)
        self.assertNotIn(shared.FLAG, profiles[CONTROL]['env'])
        for name in (ARM, PLAIN, GROW):
            self.assertEqual(profiles[name]['env'][shared.FLAG], '1')
        self.assertEqual(profiles[GROW]['env'][shared.GROW_FLAG], '1')
        for name in (CONTROL, ARM, PLAIN):
            self.assertNotIn(shared.GROW_FLAG, profiles[name]['env'])
            self.assertEqual(profiles[name]['engine']['num-gpu-blocks-override'], 19968)
        self.assertEqual(profiles[GROW]['engine']['num-gpu-blocks-override'], 22144)

    def test_the_attach_and_hang_jobs_use_the_profiles_their_comments_name(self):
        self.assertEqual(self.parsed('S0c-control-attach-smoke')['profile'], CONTROL)
        self.assertEqual(self.parsed('S0-audited-attach-smoke')['profile'], ARM)
        for name in ('S0c-control-attach-smoke', 'S0-audited-attach-smoke'):
            self.assertEqual(self.parsed(name)['tests'], SMOKE)
        for name in ('H1-hang-shapes-4e', 'H2-hang-shapes-4e'):
            out = self.parsed(name)
            self.assertEqual(out['profile'], PLAIN)
            self.assertIn('concurrent8_drain', out['tests'])
            self.assertIn('concurrent8_code_equal', out['tests'])
        self.assertEqual(self.parsed('E1-ladder8-exactness')['profile'], ARM)

    def test_the_admission_jobs_ask_for_five_full_windows_and_the_numbers_in_the_comments_are_the_modules(self):
        for name, profile in (('A5-admission5-grow', GROW), ('A5c-admission5-control', CONTROL)):
            out = self.parsed(name)
            self.assertEqual((out['profile'], out['gate_plan'], out['gate_memory_prompt'], out['gate_memory_users']),
                             (profile, 'memory', '253920', '5'), name)
        text = text_of('A5-admission5-grow')
        self.assertIn('5 x 4,097 = 20,485', text)
        self.assertEqual(5 * shared.FULL_WINDOW_BLOCKS, 20485)
        self.assertIn('22,144', text)
        self.assertEqual(shared.grown_pool_edge(), 22144)
        self.assertIn('1416704', text)
        self.assertEqual(shared.pool_tokens_for(22144), 1416704)
        self.assertIn('running=16388 and reserved=4097', text_of('A5c-admission5-control'))
        self.assertEqual(4 * shared.FULL_WINDOW_BLOCKS, 16388)
        s0 = text_of('S0-audited-attach-smoke')
        self.assertIn('freed_per_chip=%d kv_blocks_gained=%d' % (shared.freed_bytes(), shared.kv_blocks_gained()), s0)
        self.assertIn('tensors=%d tensor_bytes=%d' % (shared.TENSORS, 6 * 2 ** 20), s0)
        self.assertIn('1,207,959,552', s0)

    def test_templates_are_lf_and_name_nothing_private(self):
        for name in os.listdir(FOLDER):
            with open(os.path.join(FOLDER, name), 'rb') as handle:
                raw = handle.read()
            self.assertNotIn(b'\r', raw, name)
            self.assertIsNone(BANNED.search(raw.decode('utf-8')), name)


if __name__ == '__main__':
    unittest.main()
