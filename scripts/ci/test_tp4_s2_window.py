"""The four-card S2 hardware window: its job templates (scripts/ci/references/tp4-s2-jobs), their order and the three profiles
its arms serve (general-tp4-ring-mmrs, c2-packed-tp4-gate-ring, c2-packed-tp4-gate-bf16).

The templates are public, so they name no rig, card, address, registry or digest; the two values a one-card harness needs
(the graft directory and the graft binary's digest) are @...@ placeholders the driver of the window fills in. Every template
parses with c2_serving_job, opens the cards in ONE step, agrees with its card set, and the gate plans they name are ones the
gate accepts on the profile they name."""

import json
import make_octo_profiles
import make_parked_profiles
import profile_twins
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-s2-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
PLACEHOLDERS = {'@K64J_GRAFT_DIR@': '/graft', '@K64J_TTNNCPP_SHA256@': '0' * 64}
# What one job may open the cards with: a job carries at most one of these (an all-four reset before it is not a run).
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.')
GATE_JOBS = ('S3a-s2-matrix', 'S3b-g1-tp4-reference', 'O2-s2-bf16-drafter', 'O3-s2-ring-topology')


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        rows = [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]
    return [(name, mode) for name, mode in rows]


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name, fill=True):
    text = text_of(name)
    if fill:
        for placeholder, value in PLACEHOLDERS.items():
            text = text.replace(placeholder, value)
    return job.parse_env(text), job.read_job(job.parse_env(text), NAMES)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_and_the_order_names_no_other(self):
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [name for name, _ in read_order()]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(len(set(ordered)), len(ordered))

    def test_the_stages_run_build_then_one_card_then_four_cards_then_the_optional_arms(self):
        order = read_order()
        self.assertEqual([name.split('-')[0] for name, _ in order],
                         ['S0', 'S1a', 'S1b', 'S1c', 'S1d', 'S2', 'S3a', 'S3b', 'S4', 'O1', 'O2', 'O3'])
        self.assertEqual(set(mode for _, mode in order), {'stop', 'soft', 'optional'})
        self.assertEqual([name for name, mode in order if mode == 'soft'], ['S3b-g1-tp4-reference'])
        self.assertEqual([mode for _, mode in order[-3:]], ['optional'] * 3)
        self.assertNotIn('optional', [mode for _, mode in order[:-3]])

    def test_the_build_runs_no_card_and_the_one_card_harnesses_run_before_any_four_card_run(self):
        seen_quad_run = False
        for name, _ in read_order():
            values, outputs = parsed(name)
            actions = outputs['actions'].split()
            if name.startswith('S0'):
                self.assertEqual(set(actions) & set(DEVICE_STEPS), set(), name)
                self.assertEqual(actions, ['status', 'reset', 'build'], 'M1: every one-card job follows a fresh all-four reset')
                self.assertEqual(outputs['cards'], 'quad', 'the four-card set is what the status step reads')
            if outputs['cards'] == 'quad' and set(actions) & set(DEVICE_STEPS):
                seen_quad_run = True
            if 'cardm' in actions and name.startswith('S1'):
                self.assertFalse(seen_quad_run, '%s: a one-card harness after a four-card run' % name)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_with_and_without_the_placeholders_filled(self):
        for name, _ in read_order():
            with self.subTest(template=name):
                parsed(name)
                parsed(name, fill=False)

    def test_the_templates_name_no_card_host_address_registry_or_digest(self):
        for name, _ in read_order():
            self.assertIsNone(BANNED.search(text_of(name)), name)

    def test_the_only_placeholders_are_the_graft_directory_and_digest_and_only_cardm_uses_them(self):
        for name, _ in read_order():
            found = set(re.findall(r'@[A-Z0-9_]+@', text_of(name)))
            values, outputs = parsed(name)
            if 'cardm' in outputs['actions'].split():
                self.assertEqual(found, set(PLACEHOLDERS), name)
            else:
                self.assertEqual(found, set(), name)

    def test_one_image_tag_for_the_whole_window(self):
        tags = set(parsed(name)[1]['tag'] for name, _ in read_order())
        self.assertEqual(tags, {'tp4-s2-1'})

    def test_a_job_opens_the_cards_in_one_step_and_a_four_card_run_resets_first(self):
        for name, _ in read_order():
            values, outputs = parsed(name)
            actions = outputs['actions'].split()
            with self.subTest(template=name):
                self.assertLessEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                if outputs['cards'] == 'quad' and set(actions) & set(DEVICE_STEPS):
                    self.assertEqual(actions[0], 'reset', 'links train only at board init: a four-card run follows a reset')
                if 'cardm' in actions:
                    self.assertEqual(outputs['cards'], 'pair')
                    self.assertNotIn('reset', actions, 'a pair-only reset leaves the links to the other boards untrained')

    def test_the_card_harnesses_are_the_four_card_ports_two_and_every_first_launch_is_under_the_watcher(self):
        harness = {}
        for name, _ in read_order():
            _, outputs = parsed(name)
            if outputs['cardm_harness']:
                env = dict(pair.split('=', 1) for pair in outputs['cardm_env'].split())
                harness[name] = (env['K64J_HARNESS'], env.get('WATCHER') == '1', env.get('TP4_WIDTH'))
                self.assertEqual(outputs['cardm_harness'], 'optimisation/ttnn-op/k64j/run_card_b.sh', name)
        self.assertEqual(harness, {'S1a-nkv1-watcher': ('nkv1_spike', True, None),
                                   'S1b-gdn-parity': ('gdn_tp4', True, '2'),
                                   'S1c-gdn-tp4-watcher': ('gdn_tp4', True, '4'),
                                   'S1d-gdn-tp4': ('gdn_tp4', False, '4'),
                                   'O1-nkv1-full': ('nkv1_spike', False, None)})
        order = [name for name, _ in read_order()]
        self.assertLess(order.index('S1c-gdn-tp4-watcher'), order.index('S1d-gdn-tp4'))
        self.assertLess(order.index('S1a-nkv1-watcher'), order.index('O1-nkv1-full'))

    def test_the_four_card_gate_jobs_serve_four_card_profiles_on_the_same_prompts(self):
        for name in GATE_JOBS:
            values, outputs = parsed(name)
            self.assertEqual(outputs['gate_plan'], 'matrix', name)
            self.assertEqual(PROFILES['profiles'][outputs['profile']]['mesh_device'], 'P150x4', name)
            if gate.s2_profile(PROFILES, outputs['profile']):
                # the kernel cache holds no four-card entry and no warm plan exists at four cards: growth is recorded
                self.assertEqual(outputs['gate_jit'], 'record', name)
                self.assertEqual(outputs['gate_audits'], '', 'the prestage and fused-commit audits are not ported to four cards')
        self.assertEqual(set(parsed(name)[1]['gate_lengths'] for name in GATE_JOBS), {'4096,16384,32768,60000'})

    def test_the_gate_plans_are_accepted_on_their_profiles_and_fit_the_step(self):
        for name in GATE_JOBS:
            values, outputs = parsed(name)
            lengths = [int(part) for part in outputs['gate_lengths'].split(',')]
            arms = gate.plan_arms('matrix', outputs['profile'], PROFILES, lengths, int(outputs['gate_max_tokens']), None, [], {})
            self.assertEqual([arm[0] for arm in arms], ['matrix-concurrent', 'matrix-solo'], name)
            self.assertEqual(len(lengths), 4, 'four users, four seats: the concurrent arm is a packed-4 round')
            # the workflow's gate step gives the plans 380 min less 10; both arms to their limits and one re-run must fit
            worst = 2 * sum(arm[2] + gate.ARM_OVERHEAD_SECONDS for arm in arms)
            self.assertLessEqual(worst, 380 * 60 - 600, name)

    def test_a_gate_only_profile_is_booted_with_the_gate_switch_in_the_dry_run_argv_of_every_gate_job(self):
        # B1: the contract refuses a gate_only profile without QWEN_C2_GATE=1, and only the smoke set it: the gate's argv must too.
        for name in GATE_JOBS:
            values, outputs = parsed(name)
            lines = []
            code = gate.main(['--image', 'img', '--profile', outputs['profile'], '--plan', 'matrix', '--cards', 'quad',
                              '--dry-run', '--results', os.path.join(HERE, 'no-results'), '--profiles',
                              os.path.join(HERE, 'qwen_c2_profiles.json'), '--lengths', outputs['gate_lengths'],
                              '--max-tokens', outputs['gate_max_tokens'], '--jit', outputs['gate_jit'] or 'auto'],
                             log=lines.append)
            self.assertEqual(code, 0, name)
            arms = [json.loads(line)['docker'] for line in lines[1:]]
            self.assertTrue(arms, name)
            wanted = PROFILES['profiles'][outputs['profile']].get('gate_only') is True
            for argv in arms:
                self.assertEqual('QWEN_C2_GATE=1' in argv, wanted, name)
                if wanted:
                    self.assertEqual(argv[argv.index('QWEN_C2_GATE=1') - 1], '-e', name)
        self.assertTrue(PROFILES['profiles']['c2-packed-tp4-gate']['gate_only'])

    def test_a_profile_that_is_not_gate_only_gets_the_agents_argv_unchanged(self):
        plain = gate.agent_shape('img', 'n', 'general-tp4', ['/d'])
        self.assertNotIn('QWEN_C2_GATE=1', plain)
        self.assertEqual(plain, gate.agent_shape('img', 'n', 'general-tp4', ['/d'], gate_only=False))
        self.assertIn('QWEN_C2_GATE=1', gate.agent_shape('img', 'n', 'c2-packed-tp4-gate', ['/d'], gate_only=True))

    def test_the_smoke_jobs_name_tests_the_smoke_knows_and_the_bench_fits_the_profile(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        for name in ('S2-quad-attach-smoke', 'S4-g1-ring-mmrs'):
            values, outputs = parsed(name)
            for test in outputs['tests'].split(','):
                self.assertIn("'%s'" % test, smoke, (name, test))
        _, outputs = parsed('S4-g1-ring-mmrs')
        engine = PROFILES['profiles'][outputs['profile']]['engine']
        for shape in outputs['bench_shapes'].split(','):
            streams, prompt = (int(part) for part in shape.split('x'))
            self.assertLessEqual(streams, engine['max-num-seqs'], shape)
            self.assertLess(prompt + 256, engine['max-model-len'] + 1, shape)
        # S2 runs no agreement test: the fast path has no device sampler and no log-probabilities to compare
        self.assertNotIn('agreement', parsed('S2-quad-attach-smoke')[1]['tests'])


class WindowProfileTests(unittest.TestCase):
    """Each arm's profile is its base with exactly the documented difference."""

    def test_each_window_profile_is_its_base_with_the_documented_difference(self):
        profiles = PROFILES['profiles']
        cases = (('general-tp4-ring-mmrs', 'general-tp4-mmrs', {}, 'FABRIC_1D_RING'),
                 ('c2-packed-tp4-gate-ring', 'c2-packed-tp4-gate', {'QWEN_FAST_CCL_TOPOLOGY': 'ring'}, 'FABRIC_1D_RING'),
                 ('c2-packed-tp4-gate-bf16', 'c2-packed-tp4-gate', {'QWEN_FAST_DRAFT_BF8': '0'}, None))
        for name, base, env_extra, fabric in cases:
            mine, theirs = json.loads(json.dumps(profiles[name])), json.loads(json.dumps(profiles[base]))
            with self.subTest(profile=name):
                self.assertEqual(mine['env'], dict(theirs['env'], **env_extra))
                if fabric:
                    self.assertEqual(mine['engine']['additional-config']['tt'].pop('fabric_config'), fabric)
                    theirs['engine']['additional-config']['tt'].pop('fabric_config')
                self.assertEqual(mine['engine'], theirs['engine'])
                for key in set(mine) | set(theirs):
                    if key not in ('description', 'env', 'engine'):
                        self.assertEqual(mine.get(key), theirs.get(key), key)
                self.assertIs(mine.get('gate_only'), True, 'an arm of the window is never traffic')
                self.assertTrue(mine['description'].startswith('GATE ONLY'))

    def test_the_ring_fabric_is_asked_for_by_name_and_only_by_the_ring_arms(self):
        for name, body in PROFILES['profiles'].items():
            fabric = body['engine']['additional-config']['tt'].get('fabric_config')
            self.assertEqual(fabric == 'FABRIC_1D_RING', name in ('general-tp4-ring-mmrs', 'c2-packed-tp4-gate-ring', 'c2-packed-tp4-speed-strace-ring'), name)

    def test_the_topology_switch_and_the_drafter_dtype_are_set_by_their_arms_only(self):
        for name, body in PROFILES['profiles'].items():
            env = body['env']
            self.assertEqual(env.get('QWEN_FAST_CCL_TOPOLOGY'), 'ring' if name in ('c2-packed-tp4-gate-ring', 'c2-packed-tp4-speed-strace-ring', 'c2-packed-tp4-8x262k-w1', 'c2-packed-tp4-8x262k-w1-lite', 'c2-packed-tp4-8x262k-w1-audit', 'c2-packed-tp4-8x262k-w2', 'c2-packed-tp4-8x262k-w2-audit', 'c2-packed-tp4-8x262k-w2-nof1', 'c2-packed-tp4-8x262k-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern', 'c2-packed-tp4-8x262k-ship-prefix-levern-traffic', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit', 'c2-packed-tp4-8x262k-ship-prefix-audit-digests', 'c2-packed-tp4-8x262k-ship-prefix-dckdefault', 'c2-packed-tp4-8x262k-ship-prefix-w2', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-sdpa', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-f1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-ln', 'c2-packed-tp4-8x262k-ship-prefix-pool', 'c2-packed-tp4-8x262k-ship-prefix-dbf16', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-w1') + profile_twins.twin_names() else None, name)
            self.assertEqual(env.get('QWEN_FAST_DRAFT_BF8'), '0' if name == 'c2-packed-tp4-gate-bf16' else None, name)


if __name__ == '__main__':
    unittest.main()
