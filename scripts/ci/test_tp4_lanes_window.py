"""The four-card LANES hardware window: its job templates (scripts/ci/references/tp4-lanes-jobs), their order and the plans they run.

The templates are public, so they name no rig, card, address, registry or digest. Every template parses with c2_serving_job, opens the
cards in ONE step, follows an all-four reset, and the gate plans they name are ones the gate accepts on the profiles they name and fit
the workflow's step; the durations the templates quote for the plans' worst cases are the gate's own arithmetic, so a plan that changes
cannot leave a stale estimate behind. Together the jobs measure what the window is for: D0 alone (tokens per round and tok/s), one fast
lane beside one to three standard lanes, and exactness."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_lanes_plans as plans  # noqa: E402
import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-lanes-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.')
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
GATE_JOBS = {'L1-lanes-exact': 'lanes-exact', 'L2-round-timing': 'round-timing', 'L3-lanes-timing': 'lanes-timing',
             'O1-lanes-timing-more': 'lanes-timing-more'}
STEP_MINUTES = 380


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        rows = [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]
    return [(name, mode) for name, mode in rows]


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    values = job.parse_env(text_of(name))
    return values, job.read_job(values, NAMES)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_and_the_order_names_no_other(self):
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [name for name, _ in read_order()]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(len(set(ordered)), len(ordered))

    def test_build_then_exactness_then_the_two_timing_jobs_then_the_optional_one(self):
        order = read_order()
        self.assertEqual([name.split('-')[0] for name, _ in order], ['L0', 'L1', 'L2', 'L3', 'O1'])
        self.assertEqual([mode for _, mode in order], ['stop', 'stop', 'soft', 'soft', 'optional'])

    def test_the_build_runs_no_card_and_every_device_job_follows_an_all_four_reset(self):
        for name, _ in read_order():
            values, outputs = parsed(name)
            actions = outputs['actions'].split()
            with self.subTest(template=name):
                self.assertEqual(outputs['cards'], 'quad')
                self.assertLessEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                if name.startswith('L0'):
                    self.assertEqual(actions, ['status', 'reset', 'build'])
                    self.assertEqual(set(actions) & set(DEVICE_STEPS), set())
                else:
                    self.assertEqual(actions, ['reset', 'gate'], 'links train only at board init: a four-card run follows a reset')


class TemplateTests(unittest.TestCase):
    def test_every_template_parses(self):
        for name, _ in read_order():
            with self.subTest(template=name):
                parsed(name)

    def test_the_templates_name_no_card_host_address_registry_or_digest_and_have_no_placeholder(self):
        for name, _ in read_order():
            self.assertIsNone(BANNED.search(text_of(name)), name)
            self.assertEqual(re.findall(r'@[A-Z0-9_]+@', text_of(name)), [], name)

    def test_one_image_tag_for_the_whole_window_and_it_is_not_the_s2_windows(self):
        tags = set(parsed(name)[1]['tag'] for name, _ in read_order())
        self.assertEqual(tags, {'tp4-lanes-1'})

    def test_every_template_says_what_it_is_how_to_run_it_and_how_long_it_takes(self):
        for name, _ in read_order():
            text = text_of(name)
            with self.subTest(template=name):
                self.assertIn('A committed TEMPLATE of .github/c2-serving-job.env', text)
                self.assertIn('on a throwaway commit of tp4/lanes', text)
                self.assertIn('Duration (estimate', text)
                self.assertIn('C2_IMAGE_TAG', text)

    def test_the_order_file_quotes_a_duration_for_every_job(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            text = handle.read()
        for name, _ in read_order():
            self.assertRegex(text, r'#   %s +\S' % name.split('-')[0], name)
        self.assertIn('Required jobs L0-L3', text)


class GateJobTests(unittest.TestCase):
    def test_the_gate_jobs_name_the_lanes_plans_on_a_four_card_s2_profile_and_record_the_kernel_cache(self):
        for name, plan in GATE_JOBS.items():
            values, outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual(outputs['gate_plan'], plan)
                self.assertIn(plan, job.LANES_GATE_PLANS)
                self.assertEqual(PROFILES['profiles'][outputs['profile']]['mesh_device'], 'P150x4')
                self.assertTrue(gate.s2_profile(PROFILES, outputs['profile']))
                self.assertEqual(outputs['gate_jit'], 'record', 'the lane profiles compile on their first arms')
                self.assertEqual(outputs['gate_lengths'], '', 'each plan carries its own prompts')

    def test_the_plans_are_accepted_on_the_profile_the_job_names_and_fit_the_step(self):
        for name, plan in GATE_JOBS.items():
            values, outputs = parsed(name)
            arms = gate.plan_arms(plan, outputs['profile'], PROFILES, None, int(outputs['gate_max_tokens']), None, [], {})
            worst = gate.worst_case_seconds([plan], {plan: arms})
            with self.subTest(template=name):
                self.assertTrue(arms)
                # the workflow's gate step gives the plans 380 min less 10
                self.assertLessEqual(worst, (STEP_MINUTES - 10) * 60)
                quoted = re.findall(r'worst case (?:\(every arm to its \d+-minute limit, every re-run\) is )?(\d+) min', text_of(name))
                self.assertTrue(quoted, 'the template quotes the plan\'s worst case')
                self.assertEqual(int(quoted[-1]), worst // 60, name)

    def test_the_jobs_measure_what_the_window_is_for(self):
        arms = {plan: [spec[0] for spec in gate.plan_arms(plan, 'c2-packed-tp4-gate', PROFILES, None, 4096, None, [], {})]
                for plan in job.LANES_GATE_PLANS}
        # D0 alone: tokens per round and tok/s on real coding text, against the engines it replaces, at two contexts
        self.assertEqual([name for name in arms['round-timing'] if 'lone' in name], ['base-lone-4k', 'd0-lone-4k', 'd0-lone-32k'])
        # the packed-round times at every live count, in one boot per context
        self.assertEqual([name for name in arms['round-timing'] if name.startswith('padded')], ['padded-4k', 'padded-32k'])
        # one fast lane beside one to three standard lanes
        standard = sorted(set(int(name.split('-n')[1][0]) for plan in ('lanes-timing', 'lanes-timing-more') for name in arms[plan]))
        self.assertEqual(standard, [1, 2, 3])
        # exactness of D0 and of the lanes, against the per-request engines
        self.assertEqual(arms['lanes-exact'], ['ref-solo', 'd0-solo', 'lanes-mixed'])

    def test_every_profile_a_plan_serves_is_gate_only_four_card_and_unqualified_for_traffic(self):
        used = set()
        for plan in job.LANES_GATE_PLANS:
            for spec in gate.plan_arms(plan, 'c2-packed-tp4-gate', PROFILES, None, 4096, None, [], {}):
                used.add(spec.profile)
        self.assertEqual(used, set(plans.EXPECT))
        for name in used:
            entry = PROFILES['profiles'][name]
            self.assertTrue(entry['gate_only'], name)
            self.assertEqual(entry['mesh_device'], 'P150x4', name)

    def test_a_gate_only_profile_is_booted_with_the_gate_switch_in_the_dry_run_argv_of_every_gate_job(self):
        for name, plan in GATE_JOBS.items():
            values, outputs = parsed(name)
            lines = []
            code = gate.main(['--image', 'img', '--profile', outputs['profile'], '--plan', plan, '--cards', 'quad', '--dry-run',
                              '--results', os.path.join(HERE, 'no-results'), '--profiles',
                              os.path.join(HERE, 'qwen_c2_profiles.json'), '--jit', outputs['gate_jit'] or 'auto'], log=lines.append)
            self.assertEqual(code, 0, name)
            arms = [json.loads(line) for line in lines[1:]]
            self.assertTrue(arms, name)
            for entry in arms:
                argv = entry['docker']
                self.assertIn('QWEN_C2_GATE=1', argv, (name, entry['arm']))
                self.assertEqual(argv[argv.index('QWEN_C2_GATE=1') - 1], '-e')
                self.assertTrue(any(item.startswith('QWEN_C2_PROFILE=c2-packed-tp4-') for item in argv), entry['arm'])
                self.assertEqual('--user-lane' in argv, entry['arm'].startswith('lanes-'), entry['arm'])
                self.assertEqual(any(item.startswith('QWEN_FAST_LANE_SCHEDULE=') for item in argv),
                                 entry['arm'].startswith('lanes-'), entry['arm'])


if __name__ == '__main__':
    unittest.main()
