"""The third card session pack of 2026-10-08/09 (references/tp4-session-0809-jobs): functional tests with CI running, NO timed block and NO ARC gate, plus HL-LN-DIAG, the failing Lever N hang shape with debug logs.

Every template parses with the job parser; the ORDER.txt lines match the files; the minutes, boxes, tags, NEEDS graph, directives and the clock rule are pinned; the admission walk (the private driver's `decide`,
restated here) says which jobs the clock admits at the central estimates, for a late start and when every job runs long; no profile a job names carries a digest, an audit or a gate-only marker; C2_SMOKE_ENV
allows diagnostic log switches only; the tags are allowlisted and distinct; no template or order line names a rig, card, address, registry, digest or credential. Whether a tag is still unpushed on the remote
is the driver's own launch check (git ls-remote), not a CPU test.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_session_0809` from scripts/ci.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as prefix_gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import test_tp4_stage1_windows as stage1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PACK = os.path.join(HERE, 'references', 'tp4-session-0809-jobs')
PREV_PACK = os.path.join(HERE, 'references', 'tp4-session-0808b-jobs')
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml')
SERVING_WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
WINDOW = 'tp4-serve-11'

CAP, TARGET, RESERVE_MIN = 170, 160, 40     # a start at 19:50Z: READY 22:30Z is minute 160, 22:40Z (the soft deadline) minute 170, 23:00Z (the hard cap) minute 190
HARD_CAP_MINUTE = 190
READY_AFTER_LAST_JOB = 20                   # ZR 8, LM 6, TICK 6
START_UTC = 19 * 60 + 50

# name -> (class, image, estimate, tag, box); the order is the ORDER's
JOBS = [
    ('A0X0-agentstop-unserve-rescan-reset', 'stop', WINDOW, 4, 'v589', None),
    ('HL-LN-DIAG-hang-shapes-levern-debug', 'soft', WINDOW, 28, 'v590', 45),
    ('SM-streams-agent-shapes', 'soft', WINDOW, 14, 'v591', 25),
    ('S1-stall-control-A', 'opt', WINDOW, 27, 'v592', 40),
    ('S2-stall-levern-B', 'opt', WINDOW, 27, 'v595', 40),
    ('SR10-platform-replay', 'soft', WINDOW, 25, 'v596', 60),
    ('GG1-turns-hit-control-A', 'opt', WINDOW, 28, 'v597', 45),
    ('GG2-turns-hit-levern-B', 'opt', WINDOW, 28, 'v598', 45),
    ('HL-LN-DIAG2-hang-shapes-levern-debug', 'soft', WINDOW, 28, 'v599', 45),
    ('DA1-deep128k-control-A', 'opt', WINDOW, 18, 'v601', 30),
    ('DL1-deep128k-levern-L', 'opt', WINDOW, 18, 'v602', 30),
    ('S0b2-late-reattach-smoke', 'soft', WINDOW, 11, 'v603', 25),
    ('ZR-reset-all-four', 'hand', WINDOW, 8, 'v585', None),
    ('LM-links-remeasure', 'drv', '-', 6, None, None),
    ('TICK-topology-wait', 'drv', '-', 6, None, None),
    ('Z-handback', 'hand', WINDOW, 12, 'v612', None),
    ('DEPLOY-and-engine-load', 'drv', '-', 8, None, None),
]
BY_NAME = dict((entry[0], entry) for entry in JOBS)
RESERVE_TAGS = ['v613', 'v614', 'v615']
NEEDS = {
    'HL-LN-DIAG': ['A0X0'], 'SM': ['A0X0'], 'S1': ['A0X0'], 'S2': ['A0X0'], 'SR10': ['A0X0'], 'GG1': ['A0X0'], 'GG2': ['A0X0'], 'DA1': ['A0X0'], 'DL1': ['A0X0'], 'S0b2': ['A0X0'],
}
# the launch config lists these two (block separators): SR10 needs the thin layer, DIAG2 is a second diagnostic boot the operator un-skips on purpose
OWNER_SKIPPED = ('SR10-platform-replay', 'HL-LN-DIAG2-hang-shapes-levern-debug')
# tags other sessions pushed or planned tonight (git ls-remote 2026-10-08): s0808 v558-v561 v604 v605, s0808b v569-v584 v586-v588 v593 v594 v600 v616 v617
PUSHED_TONIGHT = set(range(558, 562)) | set(range(569, 585)) | set([586, 587, 588, 593, 594, 600, 604, 605, 616, 617])
TIMED_WORDS = ('T0s', 'TA1', 'TA2', 'TA3', 'TL1', 'TL2')
DEBUG_SWITCHES = {'QWEN_FAST_SEQ_STAGE_LOG': '1', 'VLLM_LOGGING_LEVEL': 'DEBUG', 'QWEN_PREFIX_STATS_S': '5'}


def read_text(name, directory=PACK):
    with open(os.path.join(directory, name), encoding='utf-8', newline='') as handle:
        return handle.read()


def order_rows():
    return [line.split() for line in read_text('ORDER.txt').splitlines() if line.strip() and not line.startswith('#')]


def directive(name):
    return [line.split(None, 3)[3].strip() if len(line.split(None, 3)) > 3 else '' for line in read_text('ORDER.txt').splitlines() if line.startswith('# WINDOW ' + name + ' ')]


def raw(name):
    return job.parse_env(read_text(name + '.env'))


def parsed(name):
    return job.read_job(job.parse_env(read_text(name + '.env')), sorted(stage1.profiles()['profiles']), root=ROOT)


def token_names(token):
    names = []
    for alternative in token.split('|'):
        exact = [entry[0] for entry in JOBS if entry[0] == alternative]
        names += exact or [entry[0] for entry in JOBS if entry[0].startswith(alternative + '-')]
    return names


def needs_key(name):
    keys = [key for key in NEEDS if name == key or name.startswith(key + '-')]
    return max(keys, key=len) if keys else None


def planned_jobs():
    return [entry for entry in JOBS if entry[1] not in ('drv', 'hand', 'pre')]


def walk(failed=(), durations=None, cap=CAP, reserve=RESERVE_MIN, skip=OWNER_SKIPPED):
    """The driver's clock rule over the ORDER (`decide`): -> (ran, skipped {name: why}, E at the end)."""
    state, ran, skipped, clock, halted = {}, [], {}, 0, False
    planned = planned_jobs()
    for index, (name, cls, _image, estimate, _tag, box) in enumerate(planned):
        if name in skip:
            skipped[name] = 'OWNER'
            continue
        if halted:
            skipped[name] = 'HALT'
            continue
        need_ok = True
        for item in NEEDS.get(needs_key(name), []):
            if not any(state.get(member) == 'PASS' for member in token_names(item)):
                need_ok = False
        if not need_ok:
            skipped[name] = 'NEEDS'
            continue
        use_box = box if box is not None else estimate
        if cls == 'opt' and index > 0 and planned[index - 1][1] == 'opt' and skipped.get(planned[index - 1][0]) in ('TIME', 'NEEDS', 'BLOCK'):
            skipped[name] = 'BLOCK'
            continue
        if cls == 'opt' and (index == 0 or planned[index - 1][1] != 'opt'):
            use_box = 0
            for later in planned[index:]:
                if later[1] != 'opt':
                    break
                use_box += later[5]
        if clock + use_box + reserve > cap:
            skipped[name] = 'TIME'
            continue
        clock += (durations or {}).get(name, estimate)
        ran.append(name)
        state[name] = 'FAIL' if any(name.startswith(item) for item in failed) else 'PASS'
        if state[name] == 'FAIL' and cls == 'stop':
            halted = True
    return ran, skipped, clock


class PackFiles(unittest.TestCase):
    def test_every_template_parses_and_the_order_matches_the_files(self):
        templates = sorted(name[:-4] for name in os.listdir(PACK) if name.endswith('.env'))
        rows = order_rows()
        self.assertEqual([row[0] for row in rows], [entry[0] for entry in JOBS])
        self.assertEqual(templates, sorted(row[0] for row in rows if row[1] != 'drv'))
        for name in templates:
            with self.subTest(name=name):
                parsed(name)

    def test_each_line_is_class_image_minutes_tag_box_as_pinned(self):
        for row, (name, cls, image, estimate, tag, box) in zip(order_rows(), JOBS):
            with self.subTest(name=name):
                self.assertEqual(row[1], cls)
                self.assertEqual(row[2], image)
                self.assertEqual(int(row[3]), estimate)
                self.assertEqual(row[4], tag or '-')
                self.assertEqual(row[5], str(box) if box else '-')

    def test_the_image_and_box_of_a_template_are_the_orders(self):
        for name, cls, image, estimate, _tag, box in JOBS:
            if cls == 'drv':
                continue
            with self.subTest(name=name):
                self.assertEqual(raw(name)['C2_IMAGE_TAG'], image)
                if cls == 'hand' or name.startswith('A0X0'):
                    self.assertNotIn('C2_BOX_MINUTES', raw(name))
                    continue
                self.assertEqual(raw(name).get('C2_BOX_MINUTES'), str(box))
                self.assertGreaterEqual(box, estimate)
                self.assertLessEqual(box, 3 * estimate)

    def test_the_templates_shared_with_the_previous_pack_keep_its_job_bytes_apart_from_comments_and_boxes(self):
        def body(text):
            return [line for line in text.splitlines() if line and not line.startswith('#') and not line.startswith('C2_BOX_MINUTES=')]
        for name in [entry[0] for entry in JOBS if entry[1] != 'drv' and os.path.exists(os.path.join(PREV_PACK, entry[0] + '.env'))]:
            with self.subTest(name=name):
                self.assertEqual(body(read_text(name + '.env')), body(read_text(name + '.env', PREV_PACK)))

    def test_the_hand_back_templates_hold_the_end_sequence_contracts(self):
        self.assertIn('reset', raw('ZR-reset-all-four')['C2_ACTIONS'].split())
        z = raw('Z-handback')['C2_ACTIONS'].split()
        self.assertIn('agentstart', z)
        self.assertNotIn('reset', z)
        self.assertEqual(raw('A0X0-agentstop-unserve-rescan-reset')['C2_ACTIONS'].split(), ['agentstop', 'unserve', 'rescan', 'reset'])
        self.assertEqual(directive('END'), ['handback'])

    def test_no_template_or_order_line_names_a_rig_card_address_registry_digest_or_credential(self):
        for name in [entry[0] for entry in JOBS if entry[1] != 'drv'] + ['ORDER', 'README']:
            filename = name + ('.txt' if name == 'ORDER' else '.md' if name == 'README' else '.env')
            with self.subTest(name=name):
                self.assertIsNone(stage1.BANNED.search(read_text(filename)), name)

    def test_templates_are_lf_and_files_end_with_a_newline(self):
        for filename in sorted(os.listdir(PACK)):
            with self.subTest(filename=filename):
                text = read_text(filename)
                self.assertNotIn('\r', text)
                self.assertTrue(text.endswith('\n'))


class CiRunningNoTimedBlock(unittest.TestCase):
    """The owner's rule: CI keeps running, so nothing here is timed and nothing waits for the ARC runners to be scaled down."""

    def test_there_is_no_timed_job_no_arc_directive_and_the_readme_says_the_timings_are_noisy(self):
        names = [entry[0] for entry in JOBS]
        for word in TIMED_WORDS:
            self.assertFalse([name for name in names if name.startswith(word + '-')], word)
            self.assertFalse([name for name in os.listdir(PACK) if name.startswith(word + '-')], word)
        order = read_text('ORDER.txt')
        self.assertFalse([line for line in order.splitlines() if line.startswith('# ARC-ZERO')])
        self.assertNotIn('ARC_ZERO_ACK', ' '.join(read_text(name) for name in os.listdir(PACK) if name.endswith('.env')))
        readme = read_text('README.md')
        for needle in ('CI left running', 'noisy', 'no timed block', 'no ARC gate'):
            self.assertIn(needle.lower(), readme.lower(), needle)

    def test_the_clock_matches_the_owners_times_for_a_start_at_1950z(self):
        self.assertEqual(START_UTC + TARGET, 22 * 60 + 30)
        self.assertEqual(START_UTC + HARD_CAP_MINUTE, 23 * 60)
        self.assertEqual(START_UTC + CAP, 22 * 60 + 40)
        self.assertEqual(directive('CAP'), [str(CAP)])
        self.assertEqual(directive('TARGET'), [str(TARGET)])
        self.assertEqual(directive('RESERVE-MIN'), [str(RESERVE_MIN)])
        self.assertEqual(directive('RESERVE'), [' '.join(RESERVE_TAGS)])
        self.assertLessEqual(CAP - RESERVE_MIN + READY_AFTER_LAST_JOB, TARGET)
        self.assertGreaterEqual(RESERVE_MIN, 8 + 6 + 6 + 12)


class NoDigestNoAudit(unittest.TestCase):
    def test_every_profile_a_job_names_is_a_traffic_or_production_profile_without_digests_or_audits(self):
        document = stage1.profiles()['profiles']
        seen = set()
        for entry in JOBS:
            if entry[1] == 'drv':
                continue
            for key in ('C2_PROFILE', 'C2_REPLAY_PROFILE', 'C2_PREFIX_PROFILE'):
                profile = raw(entry[0]).get(key)
                if not profile:
                    continue
                seen.add(profile)
                with self.subTest(job=entry[0], profile=profile):
                    self.assertIn(profile, (stage1.R + 'ship-prefix', stage1.R + 'ship-prefix-levern-traffic'))
                    self.assertFalse(document[profile].get('gate_only'))
                    for name, value in document[profile]['env'].items():
                        if name == 'QWEN_FAST_GDN_PREFILL_CONV_AUDIT':
                            continue
                        if 'DIGEST' in name or 'AUDIT' in name:
                            self.assertIn(str(value), ('', '0'), name)
        self.assertEqual(seen, set([stage1.R + 'ship-prefix', stage1.R + 'ship-prefix-levern-traffic']))

    def test_the_prefix_gate_arms_are_timed_scenarios_without_an_audited_derived_profile(self):
        for name in ('GG1-turns-hit-control-A', 'GG2-turns-hit-levern-B'):
            entry = raw(name)
            self.assertEqual(entry['C2_ACTIONS'].split(), ['reset', 'prefix'])
            self.assertNotIn('C2_PREFIX_KVREAD_MOUNT', entry)
            for plan in entry['C2_PREFIX_PLAN'].split(','):
                arms = prefix_gate.plan_arms(plan.strip(), entry['C2_PREFIX_PROFILE'], None, stage1.profiles())
                self.assertTrue(arms, plan)
                for arm in arms:
                    with self.subTest(job=name, arm=arm['arm']):
                        self.assertNotIn(arm['derived'], prefix_gate.AUDIT_KINDS)
                        self.assertNotIn(arm['arm'], prefix_gate.AUDIT_COST_ARMS)

    def test_the_smoke_tests_exist_and_no_job_runs_the_gate_or_fabric_actions(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        for entry in JOBS:
            if entry[1] == 'drv':
                continue
            for test in [t for t in raw(entry[0]).get('C2_SMOKE_TESTS', '').split(',') if t]:
                with self.subTest(job=entry[0], test=test):
                    self.assertIn("'%s'" % test, smoke, test)
            self.assertNotIn('gate', raw(entry[0]).get('C2_ACTIONS', '').split(), entry[0])
            self.assertNotIn('fabric', raw(entry[0]).get('C2_ACTIONS', '').split(), entry[0])


class DiagnosticRun(unittest.TestCase):
    """HL-LN-DIAG: the failing hang shape of s0808b's HL-LN, the same image, profile and tests, plus the debug log switches."""

    def test_the_diag_jobs_are_hl_ln_with_the_debug_switches_and_nothing_else_changed(self):
        previous = job.parse_env(read_text('HL-LN-hang-shapes-levern.env', PREV_PACK))
        for name in ('HL-LN-DIAG-hang-shapes-levern-debug', 'HL-LN-DIAG2-hang-shapes-levern-debug'):
            entry = raw(name)
            with self.subTest(name=name):
                for key in ('C2_CARDS', 'C2_ACTIONS', 'C2_IMAGE_TAG', 'C2_PROFILE', 'C2_SMOKE_TESTS'):
                    self.assertEqual(entry[key], previous[key], key)
                self.assertEqual(dict(pair.split('=') for pair in entry['C2_SMOKE_ENV'].split()), DEBUG_SWITCHES)
                self.assertEqual(parsed(name)['smoke_env'], entry['C2_SMOKE_ENV'])
        self.assertEqual(previous['C2_PROFILE'], stage1.R + 'ship-prefix-levern-traffic')

    def test_only_the_diag_jobs_set_a_smoke_env(self):
        for entry in JOBS:
            if entry[1] != 'drv':
                self.assertEqual('C2_SMOKE_ENV' in raw(entry[0]), 'DIAG' in entry[0], entry[0])

    def test_the_diag_header_names_every_switch_and_the_file_that_reads_it(self):
        text = read_text('HL-LN-DIAG-hang-shapes-levern-debug.env')
        for needle in ('QWEN_FAST_SEQ_STAGE_LOG', 'trace_census.py', 'VLLM_LOGGING_LEVEL=DEBUG', 'QWEN_PREFIX_STATS_S', 'qwen_prefix_scheduler_patch.py', 'run 37773733305', 'alternation did not'):
            self.assertIn(needle, text, needle)


class SmokeEnvKey(unittest.TestCase):
    def smoke(self, **extra):
        values = {'C2_ACTIONS': 'reset smoke', 'C2_CARDS': 'quad', 'C2_IMAGE_TAG': 'tp4-serve-11', 'C2_PROFILE': stage1.R + 'ship-prefix-levern-traffic'}
        values.update(extra)
        return job.read_job(values, sorted(stage1.profiles()['profiles']), root=ROOT)

    def test_allowed_switches_pass_through_and_the_default_is_empty(self):
        self.assertEqual(self.smoke()['smoke_env'], '')
        self.assertEqual(self.smoke(C2_SMOKE_ENV='QWEN_FAST_SEQ_STAGE_LOG=1 VLLM_LOGGING_LEVEL=INFO QWEN_PREFIX_STATS_S=30')['smoke_env'],
                         'QWEN_FAST_SEQ_STAGE_LOG=1 VLLM_LOGGING_LEVEL=INFO QWEN_PREFIX_STATS_S=30')

    def test_anything_else_is_refused(self):
        for bad in ('QWEN_PREFIX_DIGESTS=1', 'QWEN_PREFIX_AUDIT=1', 'QWEN_FAST_LEVER_N=0', 'QWEN_FAST_LEVERN_AUDIT=1', 'QWEN_FAST_TRACE_CENSUS=1', 'QWEN_FAST_SEQ_STAGE_LOG=0', 'QWEN_FAST_SEQ_STAGE_LOG',
                    'VLLM_LOGGING_LEVEL=TRACE', 'QWEN_PREFIX_STATS_S=0', 'QWEN_FAST_SEQ_STAGE_LOG=1 QWEN_FAST_SEQ_STAGE_LOG=1', 'PATH=/x'):
            with self.subTest(bad=bad):
                with self.assertRaises(job.JobError):
                    self.smoke(C2_SMOKE_ENV=bad)

    def test_it_needs_the_quad_smoke(self):
        with self.assertRaises(job.JobError):
            self.smoke(C2_ACTIONS='reset', C2_SMOKE_ENV='QWEN_FAST_SEQ_STAGE_LOG=1')

    def test_the_workflow_hands_each_pair_to_the_quad_smoke_container_as_an_env_flag(self):
        with open(SERVING_WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('SMOKE_ENV: ${{ steps.job.outputs.smoke_env }}', text)
        self.assertIn('smoke_env_args+=(-e "$pair")', text)
        self.assertIn('${smoke_env_args[@]+"${smoke_env_args[@]}"}', text)
        self.assertEqual(text.count('smoke_env_args=()'), 1)


class Tags(unittest.TestCase):
    def test_every_job_has_a_distinct_allowlisted_tag_never_pushed_tonight(self):
        tags = [entry[4] for entry in JOBS if entry[4]]
        self.assertEqual(len(tags), len(set(tags)))
        self.assertEqual(len(tags), 14)
        numbers = [int(tag[1:]) for tag in tags]
        self.assertTrue(all(number in stage1.ALLOWLISTED_UNUSED for number in numbers))
        self.assertFalse(set(numbers) & stage1.BASELINE_TAGS)
        self.assertFalse(set(numbers) & PUSHED_TONIGHT)

    def test_the_reserve_is_allowlisted_unused_and_apart_from_the_jobs(self):
        used = set(entry[4] for entry in JOBS if entry[4])
        self.assertFalse(used & set(RESERVE_TAGS))
        self.assertTrue(all(int(tag[1:]) in stage1.ALLOWLISTED_UNUSED for tag in RESERVE_TAGS))
        self.assertFalse(set(int(tag[1:]) for tag in RESERVE_TAGS) & PUSHED_TONIGHT)
        self.assertLessEqual(max(int(tag[1:]) for tag in used | set(RESERVE_TAGS)), 620)


class Directives(unittest.TestCase):
    def test_the_needs_graph_is_pinned(self):
        found = {}
        for line in read_text('ORDER.txt').splitlines():
            match = re.match(r'# NEEDS (.+) <- (.+)$', line)
            if match:
                for item in match.group(1).split():
                    found[item] = match.group(2).split()
        expected = dict(NEEDS)
        expected['HL-LN-DIAG2'] = ['A0X0', 'HL-LN-DIAG']
        self.assertEqual(found, expected)

    def test_the_only_stop_class_job_is_the_attach_and_the_diag_is_soft(self):
        self.assertEqual([entry[0] for entry in JOBS if entry[1] == 'stop'], ['A0X0-agentstop-unserve-rescan-reset'])
        self.assertEqual(BY_NAME['HL-LN-DIAG-hang-shapes-levern-debug'][1], 'soft')

    def test_the_opt_pairs_are_separated_by_soft_rows_so_each_pair_is_its_own_block(self):
        blocks, run = [], []
        for entry in planned_jobs():
            if entry[1] == 'opt':
                run.append(entry[0].split('-')[0])
            elif run:
                blocks.append(run)
                run = []
        if run:
            blocks.append(run)
        self.assertEqual(blocks, [['S1', 'S2'], ['GG1', 'GG2'], ['DA1', 'DL1']])
        self.assertEqual(directive('END'), ['handback'])


class Schedule(unittest.TestCase):
    def test_the_planned_jobs_total_at_the_central_estimates(self):
        self.assertEqual(sum(entry[3] for entry in planned_jobs() if entry[0] not in OWNER_SKIPPED), 4 + 28 + 14 + 54 + 56 + 36 + 11)

    def test_at_a_start_at_1950z_the_attach_diag_client_shapes_stall_pair_and_late_reattach_run_and_the_two_pairs_are_refused_whole(self):
        ran, skipped, clock = walk()
        self.assertEqual(ran, ['A0X0-agentstop-unserve-rescan-reset', 'HL-LN-DIAG-hang-shapes-levern-debug', 'SM-streams-agent-shapes', 'S1-stall-control-A', 'S2-stall-levern-B', 'S0b2-late-reattach-smoke'])
        self.assertEqual(clock, 4 + 28 + 14 + 27 + 27 + 11)
        self.assertEqual(skipped['GG1-turns-hit-control-A'], 'TIME')
        self.assertEqual(skipped['GG2-turns-hit-levern-B'], 'BLOCK')
        self.assertEqual(skipped['DA1-deep128k-control-A'], 'TIME')
        self.assertEqual(skipped['DL1-deep128k-levern-L'], 'BLOCK')
        self.assertEqual(set(name for name, why in skipped.items() if why == 'OWNER'), set(OWNER_SKIPPED))
        self.assertLessEqual(clock + READY_AFTER_LAST_JOB, TARGET)

    def test_a_pair_is_admitted_when_the_earlier_jobs_finish_well_inside_their_estimates(self):
        fast = {'HL-LN-DIAG-hang-shapes-levern-debug': 10, 'SM-streams-agent-shapes': 5, 'S1-stall-control-A': 8, 'S2-stall-levern-B': 8}
        ran, skipped, clock = walk(durations=fast)
        self.assertIn('GG1-turns-hit-control-A', ran)
        self.assertIn('GG2-turns-hit-levern-B', ran)
        self.assertLessEqual(clock + READY_AFTER_LAST_JOB, TARGET)

    def test_a_late_start_shrinks_the_work_first_and_never_the_attach(self):
        ran, _skipped, _ = walk(cap=CAP - 60)
        self.assertEqual(ran[:3], ['A0X0-agentstop-unserve-rescan-reset', 'HL-LN-DIAG-hang-shapes-levern-debug', 'SM-streams-agent-shapes'])
        self.assertNotIn('S1-stall-control-A', ran)
        ran, _skipped, _ = walk(cap=CAP - 80)
        self.assertEqual(ran[:2], ['A0X0-agentstop-unserve-rescan-reset', 'HL-LN-DIAG-hang-shapes-levern-debug'])

    def test_a_failed_diag_does_not_stop_the_window_and_a_failed_attach_does(self):
        ran, skipped, _ = walk(failed=('HL-LN-DIAG',))
        self.assertIn('SM-streams-agent-shapes', ran)
        ran, skipped, _ = walk(failed=('A0X0',))
        self.assertEqual(ran, ['A0X0-agentstop-unserve-rescan-reset'])

    def test_with_the_second_diag_boot_unskipped_it_is_refused_for_time_at_the_central_estimates_and_changes_nothing_before_it(self):
        ran, skipped, _ = walk(skip=('SR10-platform-replay',))
        self.assertEqual(skipped['HL-LN-DIAG2-hang-shapes-levern-debug'], 'TIME')
        self.assertEqual(ran, walk()[0])

    def test_every_admitted_job_in_the_walk_would_also_fit_when_it_runs_to_its_box(self):
        ran, _skipped, _clock = walk()
        clock = 0
        for name in ran:
            entry = BY_NAME[name]
            box = entry[5] if entry[5] else entry[3]
            self.assertLessEqual(clock + box + RESERVE_MIN, CAP, name)
            clock += entry[3]


class Allowlist(unittest.TestCase):
    def test_the_cpu_workflow_names_this_test(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('test_tp4_session_0809', text)


if __name__ == '__main__':
    unittest.main()
