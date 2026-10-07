"""The baseline development window's job pack (references/tp4-baseline-jobs).

A short TT outage on today's production only: the production base image and the production profile family, no image build, no window lever. Every template parses with the job parser, names the production base image,
touches the node agent only in A0 (stop) and Z (start), and the ORDER.txt lines match the files; the minutes, the boxes (computed from the gates' own timeouts), the outage numbers in the ORDER header, the tag map
(lowest free block, one tag a job, a reserve) and the NEEDS graph are pinned; no template or order line names a rig, card, address, registry, digest or credential; the device profile asks for op-support 20000 on a profile
the plan accepts at eight seats; the card-M jobs name the V5 harness with the one-card boot-hook fix.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as prefix_gate  # noqa: E402
import c2_serving_gate as serving_gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import ops_profile_plan  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-baseline-jobs')
PRODUCTION = 'tp4-serve-10'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent/by|home/|zot\.|release_' + 'first|admin ' + 'token|password|plink')
R = 'c2-packed-tp4-8x262k-'
PLAIN, AUDIT = R + 'ship-prefix', R + 'ship-prefix-audit'
HARD_CAP = 540
TARGET_MAX = 480
HANDBACK_MINUTES = 95
CARDM_SETUP_MINUTES = 5
HANDBACK = ('ZR-reset-all-four', 'LM-links-remeasure', 'TICK-topology-wait', 'Z-handback', 'DEPLOY-and-engine-load')
V5_BLOCK = ('R4-reset-all-four', 'V1a-cardm-watcher', 'V1b-cardm-full')
PROFILE_JOB = 'PROF-CTL-ops-profile-8x4k'
EXPECTED = ('A0-agentstop-unserve', 'X0-status-rescan-reset', 'P1a-CTL-exactness-shared', 'P1b-CTL-lifecycle-evict', 'L8-CTL-ladder8-past-131k', 'C16-CTL-churn16', PROFILE_JOB) + V5_BLOCK + HANDBACK
FIRST_TAG, RESERVE = 538, range(550, 558)
ALLOWLISTED_UNUSED = set(range(538, 569)) | set(range(582, 586)) | set(range(589, 593)) | set(range(595, 600)) | set(range(601, 621))
STEP_CAP_MINUTES = 380
SMOKE_STEP_MINUTES = 210


def profiles():
    with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
        return json.load(handle)


def read_text(name):
    with open(os.path.join(FOLDER, name), encoding='utf-8', newline='') as handle:
        return handle.read()


def raw(name):
    return job.parse_env(read_text(name + '.env'))


def parsed(name):
    return job.read_job(raw(name), sorted(profiles()['profiles']), root=ROOT)


def order():
    return [line.split() for line in read_text('ORDER.txt').splitlines() if line.strip() and not line.startswith('#')]


def row(name):
    return next(line for line in order() if line[0] == name)


def templates():
    return sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))


def box_minutes(name):
    """The job's docker-timeout limit from the gates' own tables, or None for a short step."""
    entry = raw(name)
    actions = entry.get('C2_ACTIONS', '').split()
    overhead = prefix_gate.ARM_OVERHEAD_SECONDS
    if 'prefix' in actions:
        seconds = 0
        for plan in entry['C2_PREFIX_PLAN'].split(','):
            for arm in prefix_gate.plan_arms(plan.strip(), entry['C2_PREFIX_PROFILE'], None, profiles()):
                seconds += arm['timeout'] + overhead
        return min(STEP_CAP_MINUTES, -(-seconds // 60))
    if 'gate' in actions:
        lengths = [int(item) for item in entry['C2_GATE_LENGTHS'].split(',')] if entry.get('C2_GATE_LENGTHS') else None
        seconds = 0
        for plan in entry['C2_GATE_PLAN'].split(','):
            for arm in serving_gate.plan_arms(plan.strip(), entry['C2_PROFILE'], profiles(), lengths=lengths, max_tokens=int(entry.get('C2_GATE_MAX_TOKENS') or 4096)):
                seconds += arm[2] + overhead
        return min(STEP_CAP_MINUTES, -(-seconds // 60))
    if 'cardm' in actions:
        with open(os.path.join(ROOT, *entry['C2_CARDM_HARNESS'].split('/')), encoding='utf-8') as handle:
            harness = -(-int(re.search(r'^timeout_s=(\d+)$', handle.read(), re.M).group(1)) // 60)
        # the workflow kills the cardm step at its own timeout-minutes, which is shorter than the harness's
        with open(os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml'), encoding='utf-8') as handle:
            step = re.search(r'name: Run a qualification harness on card M.*?timeout-minutes: (\d+)', handle.read(), re.S)
        return min(harness, int(step.group(1))) + CARDM_SETUP_MINUTES
    return None


def minutes(*names):
    return sum(int(row(name)[3]) for name in names)


class OrderTests(unittest.TestCase):
    def test_the_order_lists_exactly_the_expected_jobs_in_the_value_order(self):
        self.assertEqual([line[0] for line in order()], list(EXPECTED))

    def test_every_line_has_six_columns_and_every_template_is_in_the_order_once(self):
        for line in order():
            self.assertEqual(len(line), 6, line)
            self.assertIn(line[1], ('stop', 'soft', 'opt', 'hand', 'drv'), line)
        steps = [line[0] for line in order() if line[1] != 'drv']
        self.assertEqual(sorted(steps), templates())
        self.assertEqual(len(set(line[0] for line in order())), len(order()))

    def test_driver_steps_have_no_template_no_image_and_no_tag(self):
        for line in order():
            if line[1] == 'drv':
                self.assertEqual((line[2], line[4], line[5]), ('-', '-', '-'), line)
                self.assertFalse(os.path.exists(os.path.join(FOLDER, line[0] + '.env')))

    def test_the_value_order_puts_the_prefix_verdicts_first_and_the_v5_block_after_the_profile(self):
        names = [line[0] for line in order()]
        self.assertLess(names.index('P1a-CTL-exactness-shared'), names.index('P1b-CTL-lifecycle-evict'))
        self.assertLess(names.index('P1b-CTL-lifecycle-evict'), names.index('L8-CTL-ladder8-past-131k'))
        self.assertLess(names.index('C16-CTL-churn16'), names.index(PROFILE_JOB))
        self.assertLess(names.index(PROFILE_JOB), names.index('R4-reset-all-four'))
        self.assertEqual([line[0] for line in order() if line[1] == 'opt'], list(V5_BLOCK))
        self.assertEqual([line[0] for line in order()][-5:], list(HANDBACK))
        self.assertEqual(row('A0-agentstop-unserve')[1], 'stop')
        self.assertEqual(row('X0-status-rescan-reset')[1], 'stop')

    def test_every_image_column_is_the_production_base_image(self):
        for line in order():
            if line[1] != 'drv':
                self.assertEqual(line[2], PRODUCTION, line)


class NumbersTests(unittest.TestCase):
    def test_the_hand_back_is_95_minutes_and_the_planned_window_is_the_core(self):
        handback = minutes(*HANDBACK)
        self.assertEqual(handback, HANDBACK_MINUTES)
        self.assertEqual([int(row(name)[3]) for name in HANDBACK], [15, 15, 18, 32, 15])
        jobs = ('A0-agentstop-unserve', 'X0-status-rescan-reset', 'P1a-CTL-exactness-shared', 'P1b-CTL-lifecycle-evict', 'L8-CTL-ladder8-past-131k', 'C16-CTL-churn16')
        core = minutes(*jobs) + handback
        profile, v5 = int(row(PROFILE_JOB)[3]), minutes(*V5_BLOCK)
        text = read_text('ORDER.txt')
        self.assertEqual((core, v5), (440, 135))
        self.assertLessEqual(core, TARGET_MAX)
        self.assertGreater(core + profile, HARD_CAP, 'the profile does not fit beside the core: it must stay clock-gated')
        self.assertGreater(core + v5, HARD_CAP, 'the V5 block does not fit beside the core: it must stay clock-gated')
        self.assertIn('PLANNED = CORE = %d min = %.1f h' % (core, core / 60.0), text)
        self.assertIn('CORE + PROF-CTL = %d min = %.1f h' % (core + profile, (core + profile) / 60.0), text)
        self.assertIn('CORE + the V5 block (R4 V1a V1b, %d min) = %d min = %.1f h' % (v5, core + v5, (core + v5) / 60.0), text)
        instead_of_c16 = core - int(row('C16-CTL-churn16')[3]) + profile
        v5_instead = core - int(row('L8-CTL-ladder8-past-131k')[3]) - int(row('C16-CTL-churn16')[3]) + v5
        self.assertLessEqual(instead_of_c16, HARD_CAP)
        self.assertLessEqual(v5_instead, HARD_CAP)
        self.assertIn('SKIP_JOBS=C16-CTL-churn16) = %d min = %.1f h, inside the cap by %d minutes' % (instead_of_c16, instead_of_c16 / 60.0, HARD_CAP - instead_of_c16), text)
        self.assertIn('L8-CTL-ladder8-past-131k C16-CTL-churn16 PROF-CTL-ops-profile-8x4k") = %d min = %.1f h' % (v5_instead, v5_instead / 60.0), text)
        self.assertIn('TARGET_MIN=540', text)

    def test_the_stop_early_points_are_the_running_sums(self):
        text = read_text('ORDER.txt')
        after_p1 = minutes('A0-agentstop-unserve', 'X0-status-rescan-reset', 'P1a-CTL-exactness-shared', 'P1b-CTL-lifecycle-evict')
        after_c16 = after_p1 + minutes('L8-CTL-ladder8-past-131k', 'C16-CTL-churn16')
        after_prof = after_c16 + minutes(PROFILE_JOB)
        self.assertIn('after P1b-CTL (the prefix verdicts: %d min = %.1f h' % (after_p1, after_p1 / 60.0), text)
        self.assertIn('after C16-CTL (core, %d min = %.1f h)' % (after_c16, after_c16 / 60.0), text)
        self.assertIn('after PROF-CTL (%d min = %.1f h)' % (after_prof, after_prof / 60.0), text)
        self.assertLessEqual(after_prof, TARGET_MAX)

    def test_the_clock_rule_lets_every_core_job_start_on_time_and_gates_the_rest_by_box(self):
        text = read_text('ORDER.txt')
        self.assertIn('HB the hand-back (95)', text)
        self.assertIn('E + its estimate + HB <= 480', text)
        self.assertIn('E + its BOX + HB <= 540', text)
        clock = 0
        for line in order():
            if line[1] in ('opt', 'hand', 'drv') or line[0] == PROFILE_JOB:
                continue
            self.assertLessEqual(clock + int(line[3]) + HANDBACK_MINUTES, TARGET_MAX, line[0])
            clock += int(line[3])
        # the profile after the core: neither the estimate rule nor the box rule lets it start
        profile = row(PROFILE_JOB)
        self.assertGreater(clock + int(profile[3]) + HANDBACK_MINUTES, HARD_CAP)
        self.assertGreater(clock + int(profile[5]) + HANDBACK_MINUTES, HARD_CAP)
        self.assertIn('E <= 255 (target) or E <= 269 (box)', text)
        self.assertEqual(TARGET_MAX - HANDBACK_MINUTES - int(profile[3]), 255)
        self.assertEqual(HARD_CAP - HANDBACK_MINUTES - int(profile[5]), 269)
        # the V5 block must fit whole, by estimate 135 and by box R4 + V1a + V1b
        box = int(row('R4-reset-all-four')[3]) + int(row('V1a-cardm-watcher')[5]) + int(row('V1b-cardm-full')[5])
        self.assertEqual(box, 15 + 145 + 145)
        self.assertIn('135 min by estimate, 295 by box', text)

    def test_every_box_is_computed_from_the_gates_own_timeouts(self):
        self.assertEqual(box_minutes('P1a-CTL-exactness-shared'), -(-(10800 + 180) // 60))
        self.assertEqual(box_minutes('P1b-CTL-lifecycle-evict'), -(-(9000 + 180) // 60))
        self.assertEqual(box_minutes('L8-CTL-ladder8-past-131k'), -(-(2 * (5400 + 180)) // 60))
        self.assertEqual(box_minutes('C16-CTL-churn16'), -(-(9000 + 180) // 60))
        self.assertEqual(box_minutes('V1a-cardm-watcher'), 145)
        self.assertEqual(box_minutes('V1b-cardm-full'), 145)
        for line in order():
            if line[1] == 'drv':
                continue
            box = box_minutes(line[0])
            self.assertEqual(line[5], '-' if box is None else str(box), line[0])
            if box is not None:
                self.assertLessEqual(int(line[3]), box, '%s: an estimate above its own box' % line[0])


class TagTests(unittest.TestCase):
    def test_tags_are_the_lowest_free_block_in_order_one_a_job_and_allowlisted(self):
        tagged = [line for line in order() if line[4] != '-']
        self.assertEqual([line[4] for line in tagged], ['v%d' % (FIRST_TAG + index) for index in range(len(tagged))])
        self.assertEqual(len(tagged), 12)
        self.assertEqual(len(set(line[4] for line in tagged)), len(tagged))
        for line in tagged:
            self.assertIn(int(line[4][1:]), ALLOWLISTED_UNUSED, line)
        used = set(int(line[4][1:]) for line in tagged)
        self.assertTrue(used.isdisjoint(RESERVE))
        self.assertTrue(set(RESERVE) <= ALLOWLISTED_UNUSED)
        self.assertIn('the reserve v550-v557', read_text('ORDER.txt'))


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_with_the_job_parser_and_is_a_quad_or_card_m_job(self):
        for name in templates():
            result = parsed(name)
            self.assertTrue(result, name)
            self.assertEqual(raw(name)['C2_IMAGE_TAG'], PRODUCTION, name)
            self.assertIn(raw(name)['C2_CARDS'], ('quad', 'pair'), name)

    def test_the_node_agent_is_touched_only_by_the_first_and_the_last_job(self):
        for name in templates():
            actions = set(raw(name)['C2_ACTIONS'].split())
            if name == 'A0-agentstop-unserve':
                self.assertEqual(actions, {'agentstop', 'unserve'})
            elif name == 'Z-handback':
                self.assertEqual(actions, {'status', 'agentstart'})
            else:
                self.assertFalse(actions & {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'push', 'build', 'rmi'}, name)

    def test_no_job_builds_publishes_places_a_model_or_changes_the_baked_default(self):
        for name in templates():
            entry = raw(name)
            self.assertFalse(set(entry) & {'C2_BAKE_DEFAULT_PROFILE', 'C2_DRAFTER_CANDIDATES', 'C2_PLATFORM_IMAGE', 'C2_REPLAY_PROFILE', 'C2_RMI_TAGS'}, name)

    def test_the_control_jobs_serve_the_production_family_and_carry_no_window_lever(self):
        table = profiles()['profiles']
        for name in ('P1a-CTL-exactness-shared', 'P1b-CTL-lifecycle-evict', 'L8-CTL-ladder8-past-131k', 'C16-CTL-churn16', PROFILE_JOB):
            entry = raw(name)
            served = entry.get('C2_PREFIX_PROFILE') or entry['C2_PROFILE']
            self.assertIn(served, (PLAIN, AUDIT), name)
            self.assertEqual(table[served]['engine']['max-num-seqs'], 8)
            self.assertEqual(table[served]['engine']['max-model-len'], 262144)
            env = table[served]['env']
            for key in env:
                self.assertFalse(re.search(r'LEVERN|TP4_SDPA|CONV_GATES_SPREAD|DRAFTER_BF16|DRAFTER_CHECKPOINT', key), '%s names %s' % (name, key))
        self.assertEqual(raw('P1a-CTL-exactness-shared')['C2_PREFIX_PLAN'], 'exactness-shared')
        self.assertEqual(raw('P1b-CTL-lifecycle-evict')['C2_PREFIX_PLAN'], 'lifecycle-evict')
        self.assertEqual(raw('P1a-CTL-exactness-shared')['C2_PREFIX_BASELINE'], 'none')

    def test_the_gate_controls_match_the_ship_templates_they_adapt(self):
        l8, c16 = raw('L8-CTL-ladder8-past-131k'), raw('C16-CTL-churn16')
        self.assertEqual((l8['C2_GATE_PLAN'], c16['C2_GATE_PLAN']), ('matrix', 'churn'))
        self.assertEqual(l8['C2_GATE_LENGTHS'], '4096,32768,140000,253920,16384,65536,200000,253000')
        self.assertEqual(len(c16['C2_GATE_LENGTHS'].split(',')), 16)
        for entry in (l8, c16):
            self.assertEqual((entry['C2_GATE_AUDITS'], entry['C2_GATE_SALT'], entry['C2_GATE_JIT']), ('extent', 'fresh', 'record'))
            self.assertEqual(entry['C2_ACTIONS'], 'reset gate')

    def test_the_prefix_gate_keeps_its_raised_exactness_shared_box(self):
        arms = prefix_gate.plan_arms('exactness-shared', AUDIT, None, profiles())
        self.assertEqual([arm['timeout'] for arm in arms], [10800])

    def test_the_device_profile_is_the_production_profile_at_eight_seats_at_op_support_20000(self):
        entry = raw(PROFILE_JOB)
        self.assertEqual(entry['C2_PROFILE'], PLAIN)
        self.assertEqual(entry['C2_GATE_PLAN'], 'ops-twin,ops-trace')
        self.assertEqual(ops_profile_plan.OP_SUPPORT, 20000)
        self.assertEqual(ops_profile_plan.OP_SUPPORT_LIMIT, 20000)
        self.assertEqual(ops_profile_plan.plan_users(profiles(), PLAIN), 8)
        ops_profile_plan.check_timed_profile(profiles(), PLAIN)
        ops_profile_plan.check_profiles(profiles())
        self.assertIn('--op-support-count', ops_profile_plan.tracy_args())
        self.assertEqual(ops_profile_plan.tracy_args()[ops_profile_plan.tracy_args().index('--op-support-count') + 1], '20000')
        self.assertNotIn('200000', read_text(PROFILE_JOB + '.env').replace('200000 segfaulted', '').replace('200000 segfaults', ''))
        arms = serving_gate.plan_arms('ops-trace', PLAIN, profiles())
        self.assertEqual(len(arms), 1)

    def test_the_card_m_jobs_name_the_v5_harness_with_the_one_card_fix(self):
        for name in ('V1a-cardm-watcher', 'V1b-cardm-full'):
            entry = raw(name)
            self.assertEqual(entry['C2_CARDS'], 'pair')
            self.assertEqual(entry['C2_ACTIONS'], 'cardm')
            self.assertEqual(entry['C2_CARDM_HARNESS'], 'optimisation/ttnn-op/v5split/run_card_m.sh')
            self.assertIn('IMAGE_TAG=' + PRODUCTION, entry['C2_CARDM_ENV'].split())
        self.assertIn('WATCHER=1', raw('V1a-cardm-watcher')['C2_CARDM_ENV'].split())
        self.assertNotIn('WATCHER=1', raw('V1b-cardm-full').get('C2_CARDM_ENV', '').split())
        self.assertNotIn('C2_CARDM_ARGS', raw('V1b-cardm-full'), 'V1b is the FULL scope: no reduced arguments')
        with open(os.path.join(ROOT, 'optimisation', 'ttnn-op', 'v5split', 'run_card_m.sh'), encoding='utf-8') as handle:
            harness = handle.read()
        self.assertIn('-e QWEN_C2_SERVING=0', harness)
        self.assertTrue(os.path.isfile(os.path.join(ROOT, 'optimisation', 'ttnn-op', 'v5split', 'gdn_v5_card_m.py')))

    def test_the_hand_back_reset_rescans_and_z_is_read_by_its_agent_line(self):
        self.assertEqual(raw('ZR-reset-all-four')['C2_ACTIONS'].split(), ['status', 'rescan', 'reset'])
        self.assertEqual(raw('X0-status-rescan-reset')['C2_ACTIONS'].split(), ['status', 'rescan', 'reset'])
        self.assertIn("node agent after start: active", read_text('Z-handback.env'))
        self.assertIn('WAITS UP TO 30 MIN', read_text('Z-handback.env'))
        text = read_text('ORDER.txt')
        for phrase in ('Z READS PASS when its log has', 'READY FOR /deploy', 'ZR passed', 'NO-VERDICT RERUN', 'scope=reduced', 'bytes=0', 'reduced scope: ...'):
            self.assertIn(phrase, text)

    def test_the_profile_template_says_it_profiles_the_unsalted_path(self):
        self.assertNotIn('C2_GATE_SALT', raw(PROFILE_JOB))
        self.assertIn('UNSALTED', read_text(PROFILE_JOB + '.env'))

    def test_the_pack_says_production_engine_not_production_image(self):
        for name in templates():
            self.assertIn('production ENGINE', read_text(name + '.env'), name)
        self.assertIn('production ENGINE bytes, not the production image', read_text('README.md'))

    def test_the_resets_are_all_four_and_the_hand_back_orders_reset_before_start(self):
        for name in ('R4-reset-all-four', 'ZR-reset-all-four', 'X0-status-rescan-reset'):
            self.assertEqual(raw(name)['C2_CARDS'], 'quad')
            self.assertIn('reset', raw(name)['C2_ACTIONS'].split())
        names = [line[0] for line in order()]
        self.assertLess(names.index('ZR-reset-all-four'), names.index('LM-links-remeasure'))
        self.assertLess(names.index('LM-links-remeasure'), names.index('TICK-topology-wait'))
        self.assertLess(names.index('TICK-topology-wait'), names.index('Z-handback'))


class NeedsTests(unittest.TestCase):
    def test_the_needs_graph_names_real_jobs_and_has_no_cycle(self):
        short = dict(('-'.join(line[0].split('-')[:2]) if line[0].split('-')[1] == 'CTL' else line[0].split('-')[0], line[0]) for line in order())
        edges = {}
        for line in read_text('ORDER.txt').splitlines():
            match = re.match(r'# NEEDS (.+?) <- (.+)$', line)
            if match:
                for name in match.group(1).split():
                    self.assertIn(name, short, name)
                    self.assertNotIn(name, edges)
                    edges[name] = match.group(2).split()
                    for dependency in edges[name]:
                        self.assertIn(dependency, short, dependency)
        self.assertEqual(sorted(edges), sorted(['X0', 'P1a-CTL', 'P1b-CTL', 'L8-CTL', 'C16-CTL', 'PROF-CTL', 'R4', 'V1a', 'V1b']))

        def depth(name, seen=()):
            self.assertNotIn(name, seen)
            return 1 + max([depth(dep, seen + (name,)) for dep in edges.get(name, [])] or [0])
        self.assertEqual(depth('V1b'), 5)
        self.assertEqual(edges['V1b'], ['V1a'])
        self.assertEqual(edges['V1a'], ['R4'])


class HygieneTests(unittest.TestCase):
    def test_no_private_string_in_any_file_of_the_pack(self):
        for name in sorted(os.listdir(FOLDER)):
            text = read_text(name)
            hit = BANNED.search(text)
            self.assertIsNone(hit, '%s: %s' % (name, hit.group(0) if hit else ''))
            self.assertNotIn('\r', text, '%s must have LF endings' % name)
            self.assertTrue(text.endswith('\n'), name)

    def test_the_order_never_asks_a_script_to_hold_the_release_credential(self):
        text = read_text('ORDER.txt') + read_text('README.md')
        self.assertIn("admin key", text)
        self.assertIn('never in a script', text)

    def test_the_test_is_allowlisted_in_the_cpu_workflow(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml'), encoding='utf-8') as handle:
            self.assertIn('test_tp4_baseline_window', handle.read())


if __name__ == '__main__':
    unittest.main()
