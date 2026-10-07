"""The baseline extension window's job pack (references/tp4-baseline-ext-jobs).

A second TT outage block chained after the baseline window, on today's production ENGINE only. The ORDER (names, value order), the tag block v601-v615, the minutes, the boxes (from the gates' own
timeouts, or twice the measured worst for a smoke job), the hand-back and the clock rule text are pinned; every template parses with the job parser; the node agent is touched only by the first and the
last job; SR10 carries the thin-layer placeholder and nothing private; no file names a rig, card, address, registry, digest or credential. Also pins the hub-directory default of the kill-switch step.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_tp4_baseline_window as base  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-baseline-ext-jobs')
PRODUCTION = 'tp4-serve-10'
BANNED = base.BANNED
HAND_BACK = ('ZR-reset-all-four', 'LM-links-remeasure', 'TICK-topology-wait', 'Z-handback', 'DEPLOY-and-engine-load')
CORE = ('A0X0-agentstop-unserve-rescan-reset', 'H1-hang-shape-ship-prefix', 'H2-hang-shape-ship-prefix', 'SR10-platform-replay', 'C16-CTL-churn16', 'E1-CTL-exactness-eager', 'G0-turns-A-production-bytes', 'T0-timed-production-bytes')
EXPECTED = CORE + HAND_BACK
FIRST_TAG, RESERVE = 601, (611, 612)
ALLOWLISTED_UNUSED = base.ALLOWLISTED_UNUSED
GATE_TIMEOUTS = {'C16-CTL-churn16': 153, 'E1-CTL-exactness-eager': 153, 'G0-turns-A-production-bytes': 153}   # the gates' own timeouts (computed)
GATE_BOXES = {'C16-CTL-churn16': 70, 'E1-CTL-exactness-eager': 153, 'G0-turns-A-production-bytes': 40}     # the admission worst case: twice the measured worst, E1 its real timeout
SMOKE_BOXES = {'H1-hang-shape-ship-prefix': 35, 'H2-hang-shape-ship-prefix': 35, 'T0-timed-production-bytes': 65, 'SR10-platform-replay': 120}
MEASURED_WORST = {'H1-hang-shape-ship-prefix': 16.1, 'H2-hang-shape-ship-prefix': 16.1, 'T0-timed-production-bytes': 31, 'C16-CTL-churn16': 35, 'G0-turns-A-production-bytes': 20}


def read_text(name):
    with open(os.path.join(FOLDER, name), encoding='utf-8', newline='') as handle:
        return handle.read()


def raw(name):
    return job.parse_env(read_text(name + '.env'))


def order():
    return [line.split() for line in read_text('ORDER.txt').splitlines() if line.strip() and not line.startswith('#')]


def row(name):
    return next(line for line in order() if line[0] == name)


def templates():
    return sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))


def minutes(*names):
    return sum(int(row(name)[3]) for name in names)


def box_minutes(name):
    previous = base.FOLDER
    base.FOLDER = FOLDER
    try:
        return base.box_minutes(name)
    finally:
        base.FOLDER = previous


class OrderTests(unittest.TestCase):
    def test_the_order_lists_exactly_the_expected_jobs_in_the_value_order(self):
        self.assertEqual([line[0] for line in order()], list(EXPECTED))

    def test_every_line_has_six_columns_and_every_template_is_in_the_order_once(self):
        for line in order():
            self.assertEqual(len(line), 6, line)
            self.assertIn(line[1], ('stop', 'soft', 'hand', 'drv'), line)
        self.assertEqual(sorted(line[0] for line in order() if line[1] != 'drv'), templates())
        self.assertEqual(len(set(line[0] for line in order())), len(order()))

    def test_classes_and_the_image_column(self):
        self.assertEqual(row(CORE[0])[1], 'stop')
        for name in CORE[1:]:
            self.assertEqual(row(name)[1], 'soft', name)
        for line in order():
            if line[1] == 'drv':
                self.assertEqual((line[2], line[4], line[5]), ('-', '-', '-'), line)
                self.assertFalse(os.path.exists(os.path.join(FOLDER, line[0] + '.env')))
            else:
                self.assertEqual(line[2], PRODUCTION, line)

    def test_the_value_order_puts_the_hang_shapes_first_and_t0_last_after_the_gate_jobs(self):
        names = [line[0] for line in order()]
        self.assertEqual(names[:8], list(CORE))
        self.assertEqual(names[-5:], list(HAND_BACK))
        self.assertEqual(len(names), 13)
        self.assertLess(names.index('G0-turns-A-production-bytes'), names.index('T0-timed-production-bytes'))
        self.assertLess(names.index('E1-CTL-exactness-eager'), names.index('T0-timed-production-bytes'))
        self.assertEqual(names[7], 'T0-timed-production-bytes')

    def test_the_v5_block_is_not_part_of_the_extension(self):
        for name in ('R4-reset-all-four', 'V1a-cardm-watcher', 'V1b-cardm-full'):
            self.assertNotIn(name, templates())
            self.assertNotIn(name, read_text('ORDER.txt').replace('(R4 V1a V1b)', ''))


class NumbersTests(unittest.TestCase):
    def test_the_hand_back_is_95_minutes(self):
        self.assertEqual([int(row(name)[3]) for name in HAND_BACK], [15, 15, 18, 32, 15])
        self.assertEqual(minutes(*HAND_BACK), 95)
        self.assertIn('THE HAND-BACK IS 95 MIN', read_text('ORDER.txt'))

    def test_the_core_numbers_in_the_header_are_the_sums(self):
        core = minutes(*CORE)
        text = read_text('ORDER.txt')
        self.assertEqual(core, 257)
        self.assertIn('= %d min = %.1f h of jobs and %d min = %.1f h with the hand-back' % (core, core / 60.0, core + 95, (core + 95) / 60.0), text)

    def test_the_measured_basis_estimates(self):
        for name in ('H1-hang-shape-ship-prefix', 'H2-hang-shape-ship-prefix'):
            self.assertTrue(11.5 <= int(row(name)[3]) <= 18, name)
        self.assertTrue(60 <= int(row('E1-CTL-exactness-eager')[3]) <= 153)
        self.assertTrue(20 <= int(row('C16-CTL-churn16')[3]) <= 35)
        self.assertTrue(25 <= int(row('T0-timed-production-bytes')[3]) <= 31)
        self.assertTrue(16 <= int(row('G0-turns-A-production-bytes')[3]) <= 20)

    def test_every_box_is_computed_or_twice_the_measured_worst(self):
        for name, expected in GATE_TIMEOUTS.items():
            self.assertEqual(box_minutes(name), expected, name)
        self.assertEqual(box_minutes('C16-CTL-churn16'), -(-(9000 + 180) // 60))
        for line in order():
            if line[1] == 'drv':
                continue
            name = line[0]
            if name in GATE_BOXES:
                self.assertEqual(line[5], str(GATE_BOXES[name]), name)
                self.assertLessEqual(GATE_BOXES[name], GATE_TIMEOUTS[name], name)
                if name in MEASURED_WORST:
                    self.assertGreaterEqual(int(line[5]), 2 * MEASURED_WORST[name] - 1, name)
            elif name in SMOKE_BOXES:
                self.assertEqual(line[5], str(SMOKE_BOXES[name]), name)
                if name in MEASURED_WORST:
                    self.assertGreaterEqual(int(line[5]), 2 * MEASURED_WORST[name] - 1, name)
            else:
                self.assertEqual(line[5], '-', name)
            if line[5] != '-':
                self.assertLessEqual(int(line[3]), int(line[5]), '%s: an estimate above its own box' % name)

    def test_the_clock_rule_and_target_are_stated(self):
        text = read_text('ORDER.txt')
        for phrase in ('WALL-CLOCK rule against READY FOR /deploy', '19:45Z', 'HARD CAP (20:00Z', 'NOW + its ESTIMATE + HA <= the READY TARGET', 'NOW + its BOX + HA <= the HARD CAP', 'ZR + LM + TICK = 48', 'START GATE', 'Nothing is ever cancelled', 'the 95 is for reporting only'):
            self.assertIn(phrase, text.replace('The 95 is', 'the 95 is'))
        self.assertEqual(minutes('ZR-reset-all-four', 'LM-links-remeasure', 'TICK-topology-wait'), 48)

    def test_the_read_rules_carry_the_sr10_caveat(self):
        for text in (read_text('ORDER.txt'), read_text('README.md')):
            self.assertIn('2026-09-26', text)
            self.assertIn('pair-era argv', text)


class TagTests(unittest.TestCase):
    def test_tags_are_v601_upward_one_a_job_allowlisted_with_two_reserve(self):
        tagged = [line for line in order() if line[4] != '-']
        self.assertEqual([line[4] for line in tagged], ['v%d' % (FIRST_TAG + index) for index in range(len(tagged))])
        self.assertEqual(len(tagged), 10)
        for line in tagged:
            self.assertIn(int(line[4][1:]), ALLOWLISTED_UNUSED, line)
        used = set(int(line[4][1:]) for line in tagged)
        self.assertTrue(used.isdisjoint(RESERVE))
        self.assertTrue(set(RESERVE) <= ALLOWLISTED_UNUSED)
        self.assertEqual(max(used | set(RESERVE)), 612)
        self.assertIn('v601 to v615', read_text('ORDER.txt'))
        self.assertIn('reserve v611-v615', read_text('ORDER.txt'))

    def test_the_block_does_not_touch_the_first_windows_tags_or_the_gap_another_plan_may_use(self):
        for line in order():
            if line[4] != '-':
                self.assertFalse(538 <= int(line[4][1:]) <= 600, line)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_with_the_job_parser_and_names_the_production_image(self):
        for name in templates():
            entry = dict(raw(name))
            if entry.get('C2_PLATFORM_IMAGE') == '@THIN_LAYER_IMAGE@':
                entry['C2_PLATFORM_IMAGE'] = 'registry.example/serving-layer@digest'
            self.assertTrue(job.read_job(entry, sorted(base.profiles()['profiles']), root=ROOT), name)
            self.assertEqual(raw(name)['C2_IMAGE_TAG'], PRODUCTION, name)

    def test_the_agent_is_stopped_only_by_the_first_job_and_started_only_by_the_last(self):
        for name in templates():
            actions = raw(name)['C2_ACTIONS'].split()
            if name == CORE[0]:
                self.assertEqual(actions, ['agentstop', 'unserve', 'rescan', 'reset'])
                self.assertNotIn('status', actions)
            elif name == 'Z-handback':
                self.assertEqual(actions, ['status', 'agentstart'])
            else:
                self.assertFalse(set(actions) & {'agentstop', 'agentstart', 'unserve', 'push', 'build', 'rmi', 'platform'}, name)

    def test_only_sr10_replays_and_only_with_the_placeholder_and_the_production_profile(self):
        for name in templates():
            entry = raw(name)
            if name == 'SR10-platform-replay':
                self.assertEqual(entry['C2_ACTIONS'].split(), ['reset', 'replay'])
                self.assertEqual(entry['C2_PLATFORM_IMAGE'], '@THIN_LAYER_IMAGE@')
                # the placeholder appears once, in the value: the driver's sed must not write the image reference into a comment of the pushed commit
                self.assertEqual(read_text(name + '.env').count('@THIN_LAYER_IMAGE@'), 1)
                self.assertEqual(entry['C2_REPLAY_PROFILE'], base.PLAIN)
                self.assertEqual(entry['C2_REPLAY_BUDGET_SMOKE'], '1')
            else:
                self.assertFalse(set(entry) & {'C2_PLATFORM_IMAGE', 'C2_REPLAY_PROFILE', 'C2_BAKE_DEFAULT_PROFILE', 'C2_DRAFTER_CANDIDATES', 'C2_RMI_TAGS'}, name)

    def test_the_production_profile_family_and_no_window_lever(self):
        table = base.profiles()['profiles']
        for name in templates():
            entry = raw(name)
            served = entry.get('C2_PREFIX_PROFILE') or entry.get('C2_PROFILE') or entry.get('C2_REPLAY_PROFILE')
            if served is None:
                continue
            self.assertIn(served, (base.PLAIN, base.AUDIT), name)
            self.assertEqual(table[served]['engine']['max-num-seqs'], 8)
            self.assertEqual(table[served]['engine']['max-model-len'], 262144)
            for key in table[served]['env']:
                self.assertFalse(re.search(r'LEVERN|TP4_SDPA|CONV_GATES_SPREAD|DRAFTER_BF16|DRAFTER_CHECKPOINT', key), '%s names %s' % (name, key))

    def test_the_shapes_of_the_jobs(self):
        h1, h2 = raw('H1-hang-shape-ship-prefix'), raw('H2-hang-shape-ship-prefix')
        self.assertEqual(h1, h2)
        self.assertEqual(h1['C2_PROFILE'], base.PLAIN)
        self.assertEqual(h1['C2_SMOKE_TESTS'].split(','), ['warmup', 'concurrent4_steady', 'concurrent8_steady', 'steady_resend', 'replay_concurrent4', 'replay_concurrent8', 'concurrent8_code_equal', 'concurrent5_split', 'concurrent8_drain'])
        self.assertNotIn('C2_GATE_SALT', h1)
        c16 = raw('C16-CTL-churn16')
        self.assertEqual((c16['C2_GATE_PLAN'], c16['C2_PROFILE'], c16['C2_GATE_SALT']), ('churn', base.AUDIT, 'fresh'))
        self.assertEqual(len(c16['C2_GATE_LENGTHS'].split(',')), 16)
        e1 = raw('E1-CTL-exactness-eager')
        self.assertEqual((e1['C2_PREFIX_PLAN'], e1['C2_PREFIX_PROFILE'], e1['C2_PREFIX_BASELINE']), ('exactness-eager', base.AUDIT, 'none'))
        g0 = raw('G0-turns-A-production-bytes')
        self.assertEqual((g0['C2_PREFIX_PLAN'], g0['C2_PREFIX_PROFILE'], g0['C2_PREFIX_AGENTS']), ('agent-turns-prefix', base.PLAIN, '8'))
        t0 = raw('T0-timed-production-bytes')
        self.assertEqual(t0['C2_PROFILE'], base.PLAIN)
        self.assertIn('concurrent8_steady', t0['C2_SMOKE_TESTS'].split(','))

    def test_the_resets_are_all_four_and_the_hand_back_orders_reset_before_start(self):
        for name in ('ZR-reset-all-four', CORE[0]):
            self.assertEqual(raw(name)['C2_CARDS'], 'quad')
            self.assertIn('reset', raw(name)['C2_ACTIONS'].split())
        self.assertEqual(raw('ZR-reset-all-four')['C2_ACTIONS'].split(), ['status', 'rescan', 'reset'])
        self.assertIn('node agent after start: active', read_text('Z-handback.env'))
        names = [line[0] for line in order()]
        self.assertLess(names.index('ZR-reset-all-four'), names.index('LM-links-remeasure'))
        self.assertLess(names.index('LM-links-remeasure'), names.index('TICK-topology-wait'))
        self.assertLess(names.index('TICK-topology-wait'), names.index('Z-handback'))

    def test_every_template_says_production_engine(self):
        for name in templates():
            self.assertIn('production ENGINE', read_text(name + '.env'), name)
        self.assertIn('production\nENGINE', read_text('README.md'))


class NeedsTests(unittest.TestCase):
    def test_needs_graph(self):
        edges = {}
        for line in read_text('ORDER.txt').splitlines():
            match = re.match(r'# NEEDS (.+?) <- (.+)$', line)
            if match:
                for name in match.group(1).split():
                    self.assertNotIn(name, edges)
                    edges[name] = match.group(2).split()
        self.assertEqual(sorted(edges), sorted(['H1', 'H2', 'SR10', 'C16-CTL', 'E1-CTL', 'G0', 'T0']))
        for name in edges:
            self.assertEqual(edges[name], ['A0X0'])


class WorkflowTests(unittest.TestCase):
    def test_the_kill_switch_step_defaults_to_the_hub_the_arms_mount(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml'), encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('hub="${C2_HUB_DIR:-/home/thatch/hf-cache/hub}"', text)
        self.assertNotIn('$HOME/hf-cache/hub', text)


class HygieneTests(unittest.TestCase):
    def test_no_private_string_in_any_file_of_the_pack(self):
        for name in sorted(os.listdir(FOLDER)):
            text = read_text(name)
            hit = BANNED.search(text)
            self.assertIsNone(hit, '%s: %s' % (name, hit.group(0) if hit else ''))
            self.assertNotIn('\r', text, '%s must have LF endings' % name)
            self.assertTrue(text.endswith('\n'), name)

    def test_the_release_credential_is_never_in_a_script(self):
        text = read_text('ORDER.txt') + read_text('README.md')
        self.assertIn('admin key', text)
        self.assertIn('never in a script', text)

    def test_the_test_is_allowlisted_in_the_cpu_workflow(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml'), encoding='utf-8') as handle:
            self.assertIn('test_tp4_baseline_ext_window', handle.read())


if __name__ == '__main__':
    unittest.main()
