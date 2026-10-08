"""The chained session pack of 2026-10-08 (references/tp4-session-0808-jobs): W-1 (Lever N exactness) and W-2 (robustness, then the cutover or the hand-back) in ONE outage, after the audit op's qualification.

Every template parses with the job parser; the ORDER.txt lines match the files; the minutes, boxes, tags, NEEDS graph, directives and the clock rule are pinned; the admission walk (the private driver's
`decide`, restated here) says which jobs the clock refuses at the central estimates, with the audit op qualified and with it failed; the judge line names jobs of the ORDER; no template or order line names a
rig, card, address, registry, digest or credential.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_session_0808` from scripts/ci.
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
import test_tp4_stage1_windows as stage1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PACK = os.path.join(HERE, 'references', 'tp4-session-0808-jobs')
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml')
PRODUCTION, WINDOW = 'tp4-serve-10', 'tp4-serve-11'
FIXED = 'tp4-serve-11b'   # serve-11 plus the region-read Shape fix (KVQ v560): the image of every job that runs the prefix gate's anchor probe, which holds the served model.py to PATCHED_SHA256

CAP, TARGET, RESERVE_MIN, GATE_RESET_ALLOWANCE = 750, 720, 60, 30
HARD_CAP_MINUTE = 780      # 20:30Z with the start at 07:30Z
READY_AFTER_LAST_JOB = 20  # ZR 8, LM 6, TICK 6
CUTOVER_EXTRA = 30         # the driver's thin-layer check (20) and the publish (10)

# name -> (class, image, estimate, tag, box); the order is the ORDER's
JOBS = [
    ('B0-build-serve-11', 'pre', WINDOW, 25, 'v558', None),
    ('A0X0-agentstop-unserve-rescan-reset', 'stop', WINDOW, 4, 'v559', None),
    ('A1-LN-levern-audited-attach', 'stop', WINDOW, 80, 'v561', 165),
    ('S0-CTL-control-attach-smoke', 'stop', WINDOW, 57, 'v562', 100),
    ('S0b-baked-default-smoke', 'stop', WINDOW, 11, 'v563', 20),
    ('L8-LN-ladder8-past-131k', 'soft', WINDOW, 33, 'v564', 368),
    ('C16-LN-churn16', 'soft', WINDOW, 28, 'v565', 153),
    ('KVQ2-qualify-region-read', 'soft', FIXED, 20, 'v606', 30),
    ('P1ab-LN-exactness-shared-lifecycle-evict', 'soft', FIXED, 120, 'v566', 240),
    ('P1-CTLR-prod-bytes-exactness-shared-lifecycle-evict', 'soft', PRODUCTION, 125, 'v567', 240),
    ('E1-LN-exactness-eager', 'soft', FIXED, 100, 'v568', 153),
    ('HL-LN-hang-shapes-levern', 'stop', WINDOW, 28, 'v582', 55),
    ('HL-LN2-hang-shapes-levern', 'stop', WINDOW, 28, 'v583', 55),
    ('HL-LN3-hang-shapes-levern', 'stop', WINDOW, 28, 'v584', 55),
    ('HF-LN-levern-faults', 'soft', FIXED, 45, 'v585', 90),
    ('S1-stall-control-A', 'opt', WINDOW, 27, 'v589', 50),
    ('S2-stall-levern-B', 'opt', WINDOW, 27, 'v590', 50),
    ('P1b-CTL-lifecycle-evict-production-bytes', 'soft', PRODUCTION, 75, 'v591', 153),
    ('GG1-turns-hit-control-A', 'opt', FIXED, 28, 'v592', 55),
    ('GG2-turns-hit-levern-B', 'opt', FIXED, 28, 'v595', 55),
    ('SR10-platform-replay', 'stop', WINDOW, 25, 'v596', 60),
    ('T0s-timed-production-bytes', 'opt', PRODUCTION, 10, 'v597', 20),
    ('TA1-timed-control-A', 'opt', WINDOW, 10, 'v598', 20),
    ('TL1-timed-levern-L', 'opt', WINDOW, 10, 'v599', 20),
    ('TA2-timed-control-A', 'opt', WINDOW, 10, 'v601', 20),
    ('TL2-timed-levern-L', 'opt', WINDOW, 10, 'v602', 20),
    ('TA3-timed-control-A', 'opt', WINDOW, 10, 'v603', 20),
    ('ZR-reset-all-four', 'hand', WINDOW, 8, 'v604', None),
    ('LM-links-remeasure', 'drv', '-', 6, None, None),
    ('TICK-topology-wait', 'drv', '-', 6, None, None),
    ('Z-handback', 'hand', WINDOW, 12, 'v605', None),
    ('DEPLOY-and-engine-load', 'drv', '-', 8, None, None),
    ('CUT-release-and-live-checks', 'drv', '-', 13, None, None),
]
BY_NAME = dict((entry[0], entry) for entry in JOBS)
RESERVE_TAGS = ['v%d' % n for n in range(607, 612)]   # v606 went to KVQ2 (the KVQ re-run)
NEEDS = {
    'KVQ2': ['A0X0'], 'A1-LN': ['A0X0'], 'S0-CTL': ['A0X0'], 'S0b': ['A0X0'],
    'P1ab-LN': ['S0-CTL', 'KVQ2'], 'E1-LN': ['S0-CTL', 'KVQ2'], 'P1-CTLR': ['KVQ2'],
    'L8-LN': ['S0b'], 'C16-LN': ['S0b'], 'HL-LN': ['S0b'], 'HF-LN': ['S0b'], 'HL-LN2': ['HL-LN'], 'HL-LN3': ['HL-LN2'],
    'T0s': ['S0b', 'HL-LN3', 'HF-LN'], 'TA1': ['S0b', 'HL-LN3', 'HF-LN'], 'TL1': ['S0b', 'HL-LN3', 'HF-LN'], 'TA2': ['S0b', 'HL-LN3', 'HF-LN'],
    'TL2': ['S0b', 'HL-LN3', 'HF-LN'], 'TA3': ['S0b', 'HL-LN3', 'HF-LN'], 'S1': ['S0b', 'HL-LN3', 'HF-LN'], 'S2': ['S0b', 'HL-LN3', 'HF-LN'],
    'GG1': ['S0b', 'HL-LN3', 'HF-LN'], 'GG2': ['S0b', 'HL-LN3', 'HF-LN'], 'SR10': ['S0b'],
}
CUTOVER_NEEDS = ('KVQ2 A1-LN S0-CTL P1ab-LN E1-LN S0b L8-LN C16-LN HL-LN HL-LN2 HL-LN3 HF-LN T0s TA1 TL1 TA2 TL2 TA3 S1 S2 GG1 GG2 SR10').split()
WAIVABLE = ['P1-CTLR|P1b-CTL']
FREE = stage1.FREE        # the allowlisted tags never pushed, after the baseline window's v538-v557
UNUSED_AFTER = ['v%d' % n for n in range(612, 621)]


def read_text(name):
    with open(os.path.join(PACK, name), encoding='utf-8', newline='') as handle:
        return handle.read()


def order_rows():
    return [line.split() for line in read_text('ORDER.txt').splitlines() if line.strip() and not line.startswith('#')]


def directive(name):
    return [line.split(None, 3)[3].strip() if len(line.split(None, 3)) > 3 else '' for line in read_text('ORDER.txt').splitlines() if line.startswith('# WINDOW ' + name + ' ')]


def raw(name):
    return job.parse_env(read_text(name + '.env'))


def parsed(name):
    text = read_text(name + '.env').replace(stage1.THIN, 'thin-layer-placeholder-image')
    return job.read_job(job.parse_env(text), sorted(stage1.profiles()['profiles']), root=ROOT)


def token_names(token):
    """The job names a driver token means: the exact name, else every job whose name starts with token + '-'."""
    names = []
    for alternative in token.split('|'):
        exact = [entry[0] for entry in JOBS if entry[0] == alternative]
        names += exact or [entry[0] for entry in JOBS if entry[0].startswith(alternative + '-')]
    return names


def needs_key(name):
    """The NEEDS key of a job: the longest key that is the name or a prefix of it up to a '-'."""
    keys = [key for key in NEEDS if name == key or name.startswith(key + '-')]
    return max(keys, key=len) if keys else None


def has_gate(name):
    return 'gate' in raw(name).get('C2_ACTIONS', '').split()


def walk(failed=(), durations=None, cap=CAP, reserve=RESERVE_MIN):
    """The driver's clock rule over the ORDER (`decide`): -> (ran, skipped {name: why}, E at the end). `failed` names jobs that run and FAIL (soft ones; a stop-class
    failure halts), `durations` maps a name to the minutes it takes (default: the estimate)."""
    state, ran, skipped, clock, halted = {}, [], {}, 0, False
    planned = [entry for entry in JOBS if entry[1] not in ('drv', 'hand', 'pre')]
    for index, (name, cls, _image, estimate, _tag, box) in enumerate(planned):
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
        need = estimate
        if cls == 'opt' and index > 0 and planned[index - 1][1] == 'opt' and skipped.get(planned[index - 1][0]) in ('TIME', 'NEEDS', 'BLOCK'):
            skipped[name] = 'BLOCK'   # a block member follows its first: the driver (as amended for this session) skips the whole block
            continue
        if cls == 'opt' and (index == 0 or planned[index - 1][1] != 'opt'):
            need, use_box = 0, 0
            for later in planned[index:]:
                if later[1] != 'opt':
                    break
                need += later[3]
                use_box += later[5]
        allowance = GATE_RESET_ALLOWANCE if has_gate(name) else 0
        if clock + use_box + allowance + reserve > cap:
            skipped[name] = 'TIME'
            continue
        took = (durations or {}).get(name, estimate)
        clock += took
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
        with_template = sorted(row[0] for row in rows if row[1] != 'drv')
        self.assertEqual(templates, with_template)
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

    def test_the_image_of_a_template_is_the_orders_unless_the_order_says_otherwise(self):
        for name, cls, image, _estimate, _tag, _box in JOBS:
            if cls == 'drv':
                continue
            with self.subTest(name=name):
                self.assertEqual(raw(name)['C2_IMAGE_TAG'], image)

    def test_every_template_box_equals_the_order_box_and_short_steps_have_none(self):
        for name, cls, _image, estimate, _tag, box in JOBS:
            if cls == 'drv':
                continue
            if cls in ('hand', 'pre') or name.startswith('A0X0'):
                self.assertNotIn('C2_BOX_MINUTES', raw(name), name)
                continue
            self.assertEqual(raw(name).get('C2_BOX_MINUTES'), str(box), name)
            self.assertGreaterEqual(box, estimate, name)

    def test_the_boxes_of_gate_and_prefix_jobs_are_the_gates_own_worst_cases_or_clipped_under_them(self):
        for name in ('L8-LN-ladder8-past-131k', 'C16-LN-churn16', 'P1b-CTL-lifecycle-evict-production-bytes', 'E1-LN-exactness-eager'):
            entry = raw(name)
            actions = entry['C2_ACTIONS'].split()
            if 'gate' in actions:
                worst = -(-stage1.gate_worst_seconds(entry) // 60)
            else:
                arms = dict((plan.strip(), prefix_gate.plan_arms(plan.strip(), entry['C2_PREFIX_PROFILE'], None, stage1.profiles()))
                            for plan in entry['C2_PREFIX_PLAN'].split(','))
                worst = min(stage1.STEP_CAP_MINUTES, -(-prefix_gate.worst_case_seconds(arms) // 60))
            self.assertIn(BY_NAME[name][5] - worst, (0, 1), name)   # the pack rounds C16-LN's 152 up to 153
        for name in ('P1ab-LN-exactness-shared-lifecycle-evict', 'P1-CTLR-prod-bytes-exactness-shared-lifecycle-evict'):
            entry = raw(name)
            arms = dict((plan.strip(), prefix_gate.plan_arms(plan.strip(), entry['C2_PREFIX_PROFILE'], None, stage1.profiles()))
                        for plan in entry['C2_PREFIX_PLAN'].split(','))
            self.assertLess(BY_NAME[name][5], -(-prefix_gate.worst_case_seconds(arms) // 60), name)

    def test_the_kvq_job_is_the_region_read_probe_inside_the_fabric_steps_ceiling(self):
        entry = raw('KVQ2-qualify-region-read')
        self.assertEqual((entry['C2_ACTIONS'], entry['C2_FABRIC_PROBE'], entry['C2_CARDS']), ('reset fabric', 'kvread', 'quad'))
        self.assertEqual(parsed('KVQ2-qualify-region-read')['fabric_probe'], 'kvread')
        self.assertEqual(parsed('KVQ2-qualify-region-read')['box_minutes'], '30')
        self.assertLessEqual(30, job.STEP_MINUTES['fabric'])

    def test_the_hand_back_templates_hold_the_end_sequence_contracts(self):
        self.assertIn('reset', raw('ZR-reset-all-four')['C2_ACTIONS'].split())
        z = raw('Z-handback')['C2_ACTIONS'].split()
        self.assertIn('agentstart', z)
        self.assertNotIn('reset', z)
        self.assertNotIn('rescan', z)
        self.assertNotIn('status', raw('A0X0-agentstop-unserve-rescan-reset')['C2_ACTIONS'].split())

    def test_the_controls_are_the_prefix_gate_on_the_production_audited_profile_and_the_production_base(self):
        for name, plan, mount in (('P1-CTLR-prod-bytes-exactness-shared-lifecycle-evict', 'exactness-shared,lifecycle-evict', '1'), ('P1b-CTL-lifecycle-evict-production-bytes', 'lifecycle-evict', None)):
            entry = raw(name)
            self.assertEqual((entry['C2_PREFIX_PLAN'], entry['C2_PREFIX_PROFILE'], entry['C2_IMAGE_TAG']), (plan, stage1.R + 'ship-prefix-audit', PRODUCTION))
            self.assertEqual(entry['C2_PREFIX_BASELINE'], 'none')
            self.assertEqual(entry.get('C2_PREFIX_KVREAD_MOUNT'), mount)

    def test_no_template_or_order_line_names_a_rig_card_address_registry_digest_or_credential(self):
        for name in [entry[0] for entry in JOBS if entry[1] != 'drv'] + ['ORDER']:
            filename = name + ('.txt' if name == 'ORDER' else '.env')
            with self.subTest(name=name):
                self.assertIsNone(stage1.BANNED.search(read_text(filename)), name)

    def test_templates_are_lf_and_files_end_with_a_newline(self):
        for filename in sorted(os.listdir(PACK)):
            with self.subTest(filename=filename):
                text = read_text(filename)
                self.assertNotIn('\r', text)
                self.assertTrue(text.endswith('\n'))


class Tags(unittest.TestCase):
    def test_every_job_has_a_distinct_allowlisted_free_tag_in_the_free_sets_order(self):
        tags = [entry[4] for entry in JOBS if entry[4]]
        self.assertEqual(len(tags), len(set(tags)))
        self.assertEqual(len(tags), 29)
        numbers = [int(tag[1:]) for tag in tags]
        self.assertEqual(sorted(numbers), sorted([n for n in FREE[:29] if n != 560] + [606]))   # KVQ2 took v606 from the reserve; v560 was KVQ's, run and failed
        self.assertTrue(all(number in stage1.ALLOWLISTED_UNUSED for number in numbers))
        self.assertFalse(set(numbers) & stage1.BASELINE_TAGS)

    def test_the_reserve_follows_and_nine_allowlisted_tags_stay_unused(self):
        self.assertEqual(['v%d' % n for n in FREE[30:35]], RESERVE_TAGS)
        self.assertEqual(['v%d' % n for n in FREE[35:]], UNUSED_AFTER)
        self.assertEqual(directive('RESERVE'), [' '.join(RESERVE_TAGS)])

    def test_this_pack_takes_the_tags_of_the_two_window_packs_so_they_must_never_run_beside_it(self):
        mine = set(entry[4] for entry in JOBS if entry[4])
        theirs = set(row[4] for pack in ('w1', 'w2') for row in stage1.order(pack) if row[4] != '-')
        self.assertTrue(mine & theirs)
        self.assertIn('NEVER run either beside this one', read_text('ORDER.txt'))


class Directives(unittest.TestCase):
    def test_the_directives_are_pinned(self):
        self.assertEqual(directive('END'), ['cutover'])
        self.assertEqual(directive('CAP'), [str(CAP)])
        self.assertEqual(directive('TARGET'), [str(TARGET)])
        self.assertEqual(directive('RESERVE-MIN'), [str(RESERVE_MIN)])
        self.assertEqual(directive('CUTOVER-NEEDS')[0].split(), CUTOVER_NEEDS)
        self.assertEqual(directive('CUTOVER-WAIVABLE')[0].split(), WAIVABLE)

    def test_the_clock_matches_the_owners_times(self):
        # start 07:30Z: READY 19:30Z is minute 720, the hard cap 20:30Z minute 780
        self.assertEqual(TARGET, 12 * 60)
        self.assertEqual(HARD_CAP_MINUTE, 13 * 60)
        # the last admitted job ends by CAP - RESERVE_MIN; READY follows its ZR, LM and TICK; a cutover adds the thin-layer check and the publish
        last_end = CAP - RESERVE_MIN
        self.assertLessEqual(last_end + READY_AFTER_LAST_JOB, TARGET)
        self.assertLessEqual(last_end + READY_AFTER_LAST_JOB + CUTOVER_EXTRA, HARD_CAP_MINUTE)

    def test_the_needs_graph_is_pinned(self):
        found = {}
        for line in read_text('ORDER.txt').splitlines():
            match = re.match(r'# NEEDS (.+) <- (.+)$', line)
            if match:
                for item in match.group(1).split():
                    found[item] = match.group(2).split()
        self.assertEqual(found, NEEDS)

    def test_every_needs_and_cutover_token_names_a_job_and_every_planned_job_is_covered(self):
        for token in list(NEEDS) + [item for items in NEEDS.values() for item in items] + CUTOVER_NEEDS + WAIVABLE:
            self.assertTrue(token_names(token), token)
        # HL-LN means the first hang run only: the driver's prefix rule is name + '-'
        self.assertEqual(token_names('HL-LN'), ['HL-LN-hang-shapes-levern'])
        planned = set(entry[0] for entry in JOBS if entry[1] not in ('drv', 'hand', 'pre'))
        covered = set(name for token in CUTOVER_NEEDS + WAIVABLE + ['A0X0'] for name in token_names(token))
        self.assertEqual(planned - covered, set())

    def test_the_audited_prefix_arms_need_the_qualification(self):
        for name in ('P1ab-LN', 'E1-LN', 'P1-CTLR'):
            self.assertIn('KVQ2', NEEDS[name])
        self.assertNotIn('KVQ2', NEEDS['A1-LN'])
        self.assertNotIn('KVQ2', NEEDS['P1b-CTL'] if 'P1b-CTL' in NEEDS else [])

    def test_the_judge_line_names_jobs_of_the_order_and_a_script_that_exists(self):
        line = directive('CUTOVER-JUDGE')[0]
        script = line.split()[0]
        self.assertTrue(os.path.isfile(os.path.join(ROOT, script)), script)
        for job_name in re.findall(r'\{DIR:([^}]+)\}', line):
            self.assertTrue(token_names(job_name), job_name)
        self.assertEqual(sorted(re.findall(r'\{DIR:([^}]+)\}', line)), sorted(['T0s', 'TA1', 'TA2', 'TA3', 'TL1', 'TL2', 'S0-CTL']))


class Schedule(unittest.TestCase):
    def test_the_planned_jobs_total_at_the_central_estimates(self):
        planned = [entry for entry in JOBS if entry[1] not in ('drv', 'hand', 'pre')]
        self.assertEqual(sum(entry[3] for entry in planned), 977)
        self.assertIn('977 minutes', read_text('ORDER.txt'))

    def test_at_the_central_estimates_the_cores_and_the_controls_run_and_the_tail_is_refused_for_time(self):
        ran, skipped, clock = walk()
        self.assertEqual(clock, 662)
        refused = ['HF-LN-levern-faults', 'S1-stall-control-A', 'S2-stall-levern-B', 'P1b-CTL-lifecycle-evict-production-bytes', 'GG1-turns-hit-control-A', 'GG2-turns-hit-levern-B',
                   'SR10-platform-replay', 'T0s-timed-production-bytes', 'TA1-timed-control-A', 'TL1-timed-levern-L', 'TA2-timed-control-A', 'TL2-timed-levern-L',
                   'TA3-timed-control-A']
        self.assertEqual(ran, [entry[0] for entry in JOBS if entry[1] not in ('drv', 'hand', 'pre') and entry[0] not in refused])
        self.assertEqual(sorted(skipped), sorted(refused))
        # HF-LN is refused for time and every timed, stall and agent-turn job NEEDS it (the W-2 pack's own NEEDS), so the tail is skipped for its dependency
        self.assertEqual(skipped['HF-LN-levern-faults'], 'TIME')
        self.assertEqual(set(skipped[name] for name in refused if name.startswith(('S1', 'S2', 'GG', 'T0s', 'TA', 'TL'))), {'NEEDS'})
        self.assertLessEqual(clock + READY_AFTER_LAST_JOB, TARGET)

    def test_the_cutover_is_not_reachable_at_the_central_estimates_and_the_ready_line_lands_before_the_target(self):
        ran, skipped, clock = walk()
        needed = set(name for token in CUTOVER_NEEDS for name in token_names(token))
        self.assertTrue(needed & set(skipped))
        self.assertEqual(7 * 60 + 30 + clock + READY_AFTER_LAST_JOB, 18 * 60 + 52)   # 18:52Z

    def test_a_failed_qualification_skips_the_three_audited_arms_and_the_freed_time_goes_to_the_tail(self):
        ran, skipped, clock = walk(failed=('KVQ2',))
        for name in ('P1ab-LN-exactness-shared-lifecycle-evict', 'P1-CTLR-prod-bytes-exactness-shared-lifecycle-evict', 'E1-LN-exactness-eager'):
            self.assertEqual(skipped[name], 'NEEDS')
        for name in ('S1-stall-control-A', 'S2-stall-levern-B', 'P1b-CTL-lifecycle-evict-production-bytes', 'GG1-turns-hit-control-A', 'GG2-turns-hit-levern-B'):
            self.assertIn(name, ran)
        self.assertLessEqual(clock + READY_AFTER_LAST_JOB, TARGET)
        self.assertEqual(skipped.get('P1ab-LN-exactness-shared-lifecycle-evict'), 'NEEDS')

    def test_a_stop_class_failure_halts_everything_after_it(self):
        ran, skipped, _ = walk(failed=('S0-CTL',))
        self.assertEqual(ran[-1], 'S0-CTL-control-attach-smoke')
        self.assertEqual(set(skipped.values()), {'HALT'})

    def test_when_every_job_runs_to_one_and_a_half_times_its_estimate_the_clock_still_ends_before_the_target(self):
        pessimistic = dict((entry[0], min(entry[5] or entry[3], int(entry[3] * 1.5))) for entry in JOBS if entry[1] not in ('drv', 'hand', 'pre'))
        ran, skipped, clock = walk(durations=pessimistic)
        self.assertLessEqual(clock + READY_AFTER_LAST_JOB, TARGET)
        self.assertIn('A1-LN-levern-audited-attach', ran)
        self.assertIn('S0-CTL-control-attach-smoke', ran)

    def test_every_admitted_job_in_the_walk_would_also_fit_when_it_runs_to_its_box(self):
        ran, _skipped, _clock = walk()
        clock = 0
        for name in ran:
            entry = BY_NAME[name]
            box = entry[5] if entry[5] else entry[3]
            allowance = GATE_RESET_ALLOWANCE if has_gate(name) else 0
            self.assertLessEqual(clock + box + allowance + RESERVE_MIN, CAP, name)
            clock += entry[3]


class Allowlist(unittest.TestCase):
    def test_the_cpu_workflow_names_this_test_and_the_new_probe_and_judge_tests(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        for module in ('test_tp4_session_0808', 'test_session_cutover_judge', 'test_tp4_kv_read_probe'):
            self.assertIn(module, text, module)

    def test_the_readme_names_the_owners_decisions_and_the_ready_clock(self):
        text = read_text('README.md')
        for needle in ('07:30Z', '19:30Z', '20:30Z', 'WAIVE P1CTL', 'CUTOVER_OWNER_GO', 'KVQ2', 'fallback'):
            self.assertIn(needle, text, needle)


if __name__ == '__main__':
    unittest.main()
