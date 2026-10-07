"""The two stage-1 window packs of the short-window plan (docs/tp4-short-windows.md): references/tp4-w1-levern-jobs (W-1, Lever N exactness) and
references/tp4-w2-levern-jobs (W-2, Lever N robustness, then the cutover).

Every template parses with the job parser; the ORDER.txt lines match the files; the minutes, the hand-back, the boxes (computed from the gates' own timeouts for the
long plans, pinned from the measured ranges for the smoke jobs), the admit-on-box rule, the tag map (the free allowlisted set after the baseline window's v538-v557, one
tag a job, a reserve, how many more tags Stage 2 needs) and the NEEDS graph are pinned; the Lever N traffic profile is the profile the window image bakes in; no template
or order line names a rig, card, address, registry, digest or credential.
"""

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as prefix_gate  # noqa: E402
import c2_serving_gate as serving_gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PACKS = {'w1': os.path.join(HERE, 'references', 'tp4-w1-levern-jobs'), 'w2': os.path.join(HERE, 'references', 'tp4-w2-levern-jobs')}
PRODUCTION, WINDOW = 'tp4-serve-10', 'tp4-serve-11'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent/by|home/|zot\.|release_' + 'first|admin ' + 'token|password|plink')
R = 'c2-packed-tp4-8x262k-'
PLAIN, TRAFFIC = R + 'ship-prefix', R + 'ship-prefix-levern-traffic'
LN_AUDIT, LN_AUDIT_NOLNA, DIGESTS = R + 'ship-prefix-levern-audit', R + 'ship-prefix-levern-audit-nolna', R + 'ship-prefix-audit-digests'
HARD_CAP = 540
THIN = 'local:thin-layer'
# The tags the baseline window holds (v538-v549 planned, v550-v557 reserve): never used here.
BASELINE_TAGS = set(range(538, 558))
# The allowlisted tags never pushed, in order: the baseline window's block, then the free set this pack draws from.
FREE = list(range(558, 569)) + list(range(582, 586)) + list(range(589, 593)) + list(range(595, 600)) + list(range(601, 621))
ALLOWLISTED_UNUSED = set(range(538, 569)) | set(range(582, 586)) | set(range(589, 593)) | set(range(595, 600)) | set(range(601, 621))
STEP_CAP_MINUTES, SMOKE_STEP_MINUTES, REPLAY_STEP_MINUTES = 380, 210, 180
HANDBACK = ('ZR-reset-all-four', 'LM-links-remeasure', 'TICK-topology-wait', 'Z-handback')
W1 = ('B0-build-serve-11', 'A0X0-agentstop-unserve-rescan-reset', 'A1-LN-levern-audited-attach', 'S0-CTL-control-attach-smoke', 'P1ab-LN-exactness-shared-lifecycle-evict',
      'E1-LN-exactness-eager') + HANDBACK + ('DEPLOY-and-engine-load',)
W2 = ('A0X0-agentstop-unserve-rescan-reset', 'S0b-baked-default-smoke', 'L8-LN-ladder8-past-131k', 'HL-LN-hang-shapes-levern', 'HL-LN2-hang-shapes-levern', 'HL-LN3-hang-shapes-levern',
      'HF-LN-levern-faults', 'C16-LN-churn16', 'T0s-timed-production-bytes', 'TA1-timed-control-A', 'TL1-timed-levern-L', 'TA2-timed-control-A',
      'TL2-timed-levern-L', 'TA3-timed-control-A', 'S1-stall-control-A', 'S2-stall-levern-B', 'GG1-turns-hit-control-A', 'GG2-turns-hit-levern-B', 'SR10-platform-replay') \
    + HANDBACK + ('CUT-release-and-live-checks',)
EXPECTED = {'w1': W1, 'w2': W2}
# Stage 2 (W-3 to W-6) jobs that carry a tag, from the plan: W-3 14, W-4 11, W-5 14, W-6 20.
STAGE2_TAGS = 14 + 11 + 14 + 20
# The smoke boxes: 1.5 times the high end of the measured or estimated range, up to 5 minutes.
SMOKE_BOXES = {'A1-LN-levern-audited-attach': 165, 'S0-CTL-control-attach-smoke': 100, 'S0b-baked-default-smoke': 20, 'HL-LN-hang-shapes-levern': 55, 'HL-LN2-hang-shapes-levern': 55,
               'HL-LN3-hang-shapes-levern': 55, 'T0s-timed-production-bytes': 20, 'TA1-timed-control-A': 20, 'TL1-timed-levern-L': 20, 'TA2-timed-control-A': 20,
               'TL2-timed-levern-L': 20, 'TA3-timed-control-A': 20, 'S1-stall-control-A': 50, 'S2-stall-levern-B': 50, 'SR10-platform-replay': 60}
CLIPPED_BOXES = {'HF-LN-levern-faults': 90, 'GG1-turns-hit-control-A': 55, 'GG2-turns-hit-levern-B': 55}


def profiles():
    with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
        return json.load(handle)


def read_text(pack, name):
    with open(os.path.join(PACKS[pack], name), encoding='utf-8', newline='') as handle:
        return handle.read()


def raw(pack, name):
    return job.parse_env(read_text(pack, name + '.env'))


def parsed(pack, name):
    text = read_text(pack, name + '.env').replace(THIN, 'thin-layer-placeholder-image')
    return job.read_job(job.parse_env(text), sorted(profiles()['profiles']), root=ROOT)


def order(pack):
    return [line.split() for line in read_text(pack, 'ORDER.txt').splitlines() if line.strip() and not line.startswith('#')]


def row(pack, name):
    return next(line for line in order(pack) if line[0] == name)


def templates(pack):
    return sorted(name[:-4] for name in os.listdir(PACKS[pack]) if name.endswith('.env'))


def minutes(pack, *names):
    return sum(int(row(pack, name)[3]) for name in names)


def table_box(pack, name):
    """The worst case of a gate or prefix job in minutes, from the gates' OWN helpers (the serving gate's re-runs and 120 s overhead, the prefix gate's 180 s), or None."""
    entry = raw(pack, name)
    actions = entry.get('C2_ACTIONS', '').split()
    if 'prefix' in actions:
        arms_of = dict((plan.strip(), prefix_gate.plan_arms(plan.strip(), entry['C2_PREFIX_PROFILE'], None, profiles())) for plan in entry['C2_PREFIX_PLAN'].split(','))
        seconds = prefix_gate.worst_case_seconds(arms_of)
        return min(STEP_CAP_MINUTES, -(-seconds // 60))
    if 'gate' in actions:
        return -(-gate_worst_seconds(entry) // 60)
    return None


def gate_worst_seconds(entry):
    lengths = [int(item) for item in entry['C2_GATE_LENGTHS'].split(',')] if entry.get('C2_GATE_LENGTHS') else None
    plans = [plan.strip() for plan in entry['C2_GATE_PLAN'].split(',')]
    arms_of = dict((plan, serving_gate.plan_arms(plan, entry['C2_PROFILE'], profiles(), lengths=lengths, max_tokens=int(entry.get('C2_GATE_MAX_TOKENS') or 4096))) for plan in plans)
    return serving_gate.worst_case_seconds(plans, arms_of)


HB_ADMIT = {'w1': 60, 'w2': 75}   # the hand-back's UPPER BOUND each window admits with (the central figures are 40 and 45)
GATE_RESET_ALLOWANCE = 30            # a serving-gate job's reset step is outside the box the gate refuses against


def admit_cost(pack, line):
    box = int(line[5]) if line[5] != '-' else int(line[3])
    if 'gate' in raw(pack, line[0]).get('C2_ACTIONS', '').split():
        box += GATE_RESET_ALLOWANCE
    return box


def admit(pack, hand_back=None, skip=()):
    """The admit-on-box rule walked at the estimates: -> the planned jobs the rule refuses (none when every box fits the cap)."""
    hand_back = HB_ADMIT[pack] if hand_back is None else hand_back
    refused, clock = [], 0
    for line in order(pack):
        if line[1] in ('drv', 'hand', 'pre') or line[0] in skip:
            continue
        if clock + admit_cost(pack, line) + hand_back > HARD_CAP:
            refused.append(line[0])
            continue
        clock += int(line[3])
    return refused


class OrderTests(unittest.TestCase):
    def test_each_order_lists_exactly_the_expected_jobs_in_value_order(self):
        for pack, expected in EXPECTED.items():
            with self.subTest(pack=pack):
                self.assertEqual([line[0] for line in order(pack)], list(expected))

    def test_every_line_has_six_columns_and_every_template_is_in_the_order_once(self):
        for pack in PACKS:
            with self.subTest(pack=pack):
                for line in order(pack):
                    self.assertEqual(len(line), 6, line)
                    self.assertIn(line[1], ('stop', 'soft', 'pre', 'hand', 'drv'), line)
                steps = [line[0] for line in order(pack) if line[1] != 'drv']
                self.assertEqual(sorted(steps), templates(pack))
                self.assertEqual(len(set(line[0] for line in order(pack))), len(order(pack)))

    def test_driver_steps_have_no_template_no_image_and_no_tag(self):
        for pack in PACKS:
            for line in order(pack):
                if line[1] == 'drv':
                    self.assertEqual((line[2], line[4], line[5]), ('-', '-', '-'), line)
                    self.assertFalse(os.path.exists(os.path.join(PACKS[pack], line[0] + '.env')))

    def test_every_image_column_is_the_window_image_but_the_production_bytes_job(self):
        for pack in PACKS:
            for line in order(pack):
                if line[1] == 'drv':
                    continue
                self.assertEqual(line[2], PRODUCTION if line[0] == 'T0s-timed-production-bytes' else WINDOW, line)
                self.assertEqual(raw(pack, line[0])['C2_IMAGE_TAG'], line[2], line[0])

    def test_the_first_job_of_each_window_is_a0x0_with_agentstop_unserve_rescan_reset_and_no_status(self):
        for pack, first in (('w1', 'A0X0-agentstop-unserve-rescan-reset'), ('w2', 'A0X0-agentstop-unserve-rescan-reset')):
            line = [item for item in order(pack) if item[1] != 'pre'][0]
            self.assertEqual(line[0], first)
            self.assertEqual(line[1], 'stop')
            actions = raw(pack, first)['C2_ACTIONS'].split()
            self.assertEqual(actions, ['agentstop', 'unserve', 'rescan', 'reset'])
            self.assertNotIn('status', actions)
            self.assertEqual(actions, [action for action in job.ACTIONS if action in actions], 'the workflow runs them in this fixed order')

    def test_the_hand_back_orders_reset_before_links_before_topology_before_start(self):
        for pack in PACKS:
            names = [line[0] for line in order(pack)]
            self.assertEqual([names.index(name) for name in HANDBACK], sorted(names.index(name) for name in HANDBACK))
            self.assertEqual(raw(pack, 'ZR-reset-all-four')['C2_ACTIONS'].split(), ['status', 'rescan', 'reset'])
            self.assertEqual(raw(pack, 'Z-handback')['C2_ACTIONS'].split(), ['status', 'agentstart'])
            self.assertIn('node agent after start: active', read_text(pack, 'Z-handback.env'))
            self.assertIn('up to 30 minutes', read_text(pack, 'Z-handback.env'))


class NumbersTests(unittest.TestCase):
    def test_w1_is_471_minutes_with_a_40_minute_hand_back(self):
        self.assertEqual(minutes('w1', *HANDBACK, 'DEPLOY-and-engine-load'), 40)
        core = minutes('w1', 'A0X0-agentstop-unserve-rescan-reset', 'A1-LN-levern-audited-attach', 'S0-CTL-control-attach-smoke', 'P1ab-LN-exactness-shared-lifecycle-evict',
                       'E1-LN-exactness-eager')
        self.assertEqual((core, [int(row('w1', n)[3]) for n in ('A0X0-agentstop-unserve-rescan-reset', 'A1-LN-levern-audited-attach', 'S0-CTL-control-attach-smoke',
                                                                'P1ab-LN-exactness-shared-lifecycle-evict', 'E1-LN-exactness-eager')]), (431, [4, 80, 57, 190, 100]))
        text = read_text('w1', 'ORDER.txt')
        self.assertIn('PLANNED = 471 min = 7.9 h', text)
        self.assertIn('HAND-BACK (40 min, always)', text)
        self.assertLessEqual(471, HARD_CAP)

    def test_w2_is_445_minutes_with_a_45_minute_cutover_and_a_40_minute_hand_back(self):
        jobs = [name for name in W2 if name not in HANDBACK + ('CUT-release-and-live-checks',)]
        self.assertEqual(minutes('w2', *jobs), 400)
        cutover = minutes('w2', *HANDBACK, 'CUT-release-and-live-checks')
        self.assertEqual(cutover, 45)
        self.assertEqual(cutover - int(row('w2', 'CUT-release-and-live-checks')[3]) + 8, 40, 'the hand-back swaps the last step for the 8-minute engine load')
        text = read_text('w2', 'ORDER.txt')
        self.assertIn('PLANNED = 445 min = 7.4 h', text)
        self.assertIn('plus the CUTOVER 45 min', text)
        self.assertIn('HAND-BACK instead of the cutover is 40 min', text)

    def test_the_synthesis_block_sums_hold(self):
        w2 = lambda *names: minutes('w2', *names)   # noqa: E731
        self.assertEqual(w2('HL-LN-hang-shapes-levern', 'HL-LN2-hang-shapes-levern', 'HL-LN3-hang-shapes-levern'), 84)
        self.assertEqual(w2('T0s-timed-production-bytes', 'TA1-timed-control-A', 'TL1-timed-levern-L', 'TA2-timed-control-A', 'TL2-timed-levern-L', 'TA3-timed-control-A'), 60)
        self.assertEqual(w2('S1-stall-control-A', 'S2-stall-levern-B'), 54)
        self.assertEqual(w2('GG1-turns-hit-control-A', 'GG2-turns-hit-levern-B'), 56)
        self.assertEqual(w2('T0s-timed-production-bytes', 'TA1-timed-control-A', 'TL1-timed-levern-L', 'TA2-timed-control-A', 'TL2-timed-levern-L', 'TA3-timed-control-A',
                            'S1-stall-control-A', 'S2-stall-levern-B', 'GG1-turns-hit-control-A', 'GG2-turns-hit-levern-B'), 170, 'the ARC runners are at zero for 170 minutes')
        self.assertIn('T0s to GG2: 170 minutes', read_text('w2', 'ORDER.txt'))

    def test_the_admit_rule_uses_the_hand_backs_upper_bound_and_both_orders_say_so(self):
        self.assertEqual(HB_ADMIT, {'w1': 60, 'w2': 75})
        self.assertIn("HB the hand-back's UPPER BOUND (W-1: 60 min;", read_text('w1', 'ORDER.txt'))
        self.assertIn("HB the hand-back's UPPER BOUND (W-2: 75 min,", read_text('w2', 'ORDER.txt'))
        self.assertNotIn('HB the hand-back below', read_text('w2', 'ORDER.txt'))
        self.assertIn('E + box + HB <= 540', read_text('w1', 'README.md'))
        self.assertIn('E + box + HB <= 540', read_text('w2', 'README.md'))

    def test_w2_is_admitted_whole_at_the_estimates_and_w1_carries_e1_over(self):
        self.assertEqual(admit('w2'), [])
        # W-1 at HB 60: P1ab-LN fits with 3 minutes to spare, E1-LN does not (331 > 327): the carry-over the ORDER names
        self.assertEqual(admit('w1'), ['E1-LN-exactness-eager'])
        self.assertEqual(admit('w1', 40), [])
        text = read_text('w1', 'ORDER.txt')
        for phrase in ('P1ab-LN if E <= 144', 'E1-LN if E <= 327', 'CARRY-OVER', 'THE CUTOVER MOVES TO A THIRD WINDOW', 'v610 for the first, v611 for a second'):
            self.assertIn(phrase, text)
        self.assertEqual(HARD_CAP - 336 - HB_ADMIT['w1'], 144)
        self.assertEqual(HARD_CAP - 153 - HB_ADMIT['w1'], 327)

    def test_a_carried_over_job_leaves_w2_without_its_tail_so_the_cutover_slides(self):
        # E1-LN (100) opens W-2 after A0X0: the tail of the window no longer fits the cap at the estimates
        clock, skipped = 100, []
        for line in order('w2'):
            if line[1] in ('drv', 'hand', 'pre'):
                continue
            if clock + admit_cost('w2', line) + HB_ADMIT['w2'] > HARD_CAP:
                skipped.append(line[0])
                continue
            clock += int(line[3])
        self.assertIn('SR10-platform-replay', skipped, 'the carry-over must displace the tail, and the ORDER says the cutover slides')

    def test_l8_runs_right_after_s0b_because_its_box_does_not_fit_later(self):
        names = [line[0] for line in order('w2')]
        self.assertEqual(names.index('L8-LN-ladder8-past-131k'), names.index('S0b-baked-default-smoke') + 1)
        l8 = row('w2', 'L8-LN-ladder8-past-131k')
        later = sum(int(row('w2', n)[3]) for n in W2[:W2.index('HF-LN-levern-faults') + 1] if n != l8[0])
        self.assertGreater(later + int(l8[5]) + GATE_RESET_ALLOWANCE + HB_ADMIT['w2'], HARD_CAP, 'at the old place (after HF-LN) the job would be skipped')
        self.assertLessEqual(15 + int(l8[5]) + GATE_RESET_ALLOWANCE + HB_ADMIT['w2'], HARD_CAP)

    def test_the_fixed_workflow_boxes_would_have_skipped_s2_and_gg1(self):
        """Why the box is a template key: the smoke step's 210 minutes and the G+GH docker timeouts (306) do not fit the cap late in W-2."""
        clock = 0
        skipped = []
        for line in order('w2'):
            if line[1] in ('drv', 'hand', 'pre'):
                continue
            box = int(line[3])
            if line[0].startswith('S2') or raw('w2', line[0]).get('C2_ACTIONS', '').split()[-1:] == ['smoke']:
                box = SMOKE_STEP_MINUTES
            if line[0].startswith('GG'):
                box = 306
            if clock + box + 45 > HARD_CAP:
                skipped.append(line[0])
                continue
            clock += int(line[3])
        self.assertIn('S2-stall-levern-B', skipped)
        self.assertIn('GG1-turns-hit-control-A', skipped)

    def test_w1_keeps_e1_only_if_its_box_fits(self):
        clock = minutes('w1', 'A0X0-agentstop-unserve-rescan-reset', 'A1-LN-levern-audited-attach', 'S0-CTL-control-attach-smoke', 'P1ab-LN-exactness-shared-lifecycle-evict')
        self.assertEqual(clock + int(row('w1', 'E1-LN-exactness-eager')[5]) + 40, 524, 'with the central hand-back it fits')
        self.assertGreater(clock + int(row('w1', 'E1-LN-exactness-eager')[5]) + HB_ADMIT['w1'], HARD_CAP, 'with the upper-bound hand-back it does not')
        # one hour over at the P1ab box and E1 no longer fits: it moves to the head of W-2
        self.assertGreater(clock + 60 + int(row('w1', 'E1-LN-exactness-eager')[5]) + 40, HARD_CAP)
        self.assertIn('E1-LN opens W-2 when its box does not fit here', read_text('w1', 'ORDER.txt'))
        self.assertIn('ADMIT-ON-BOX', read_text('w1', 'ORDER.txt'))


class BoxTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.scratch = tempfile.mkdtemp()
        self.addCleanup(__import__('shutil').rmtree, self.scratch, True)

    def test_every_order_box_is_its_templates_box_and_at_least_its_estimate(self):
        for pack in PACKS:
            for line in order(pack):
                if line[1] == 'drv':
                    self.assertEqual(line[5], '-', line)
                    continue
                if line[1] in ('pre', 'hand') or raw(pack, line[0]).get('C2_ACTIONS', '').split()[0] in ('agentstop', 'status'):
                    self.assertEqual(line[5], '-', line)
                    self.assertEqual(parsed(pack, line[0])['box_minutes'], '', line)
                    continue
                self.assertEqual(line[5], parsed(pack, line[0])['box_minutes'], line[0])
                self.assertGreaterEqual(int(line[5]), int(line[3]), '%s: a box under its own estimate' % line[0])

    def test_the_long_plans_boxes_are_the_gates_own_worst_cases(self):
        """Computed by the gates' own helpers (re-runs and per-arm overhead included), never by a copy of their tables."""
        self.assertEqual(table_box('w1', 'P1ab-LN-exactness-shared-lifecycle-evict'), 336)
        self.assertEqual(table_box('w1', 'E1-LN-exactness-eager'), 153)
        self.assertEqual(table_box('w2', 'L8-LN-ladder8-past-131k'), 368)
        self.assertEqual(table_box('w2', 'C16-LN-churn16'), 152)
        for pack, name in (('w1', 'P1ab-LN-exactness-shared-lifecycle-evict'), ('w1', 'E1-LN-exactness-eager'), ('w2', 'L8-LN-ladder8-past-131k')):
            self.assertEqual(int(row(pack, name)[5]), table_box(pack, name), name)
        self.assertGreaterEqual(int(row('w2', 'C16-LN-churn16')[5]), table_box('w2', 'C16-LN-churn16'))
        arms = prefix_gate.plan_arms('exactness-shared', LN_AUDIT_NOLNA, None, profiles()) + prefix_gate.plan_arms('lifecycle-evict', LN_AUDIT_NOLNA, None, profiles())
        self.assertEqual([arm['timeout'] for arm in arms], [10800, 9000])
        for name in ('L8-LN-ladder8-past-131k', 'C16-LN-churn16'):
            self.assertLessEqual(int(row('w2', name)[5]), STEP_CAP_MINUTES)

    def test_the_serving_gate_accepts_its_box_and_refuses_one_minute_less_in_its_own_dry_run(self):
        """L8 and C16 are serving-gate jobs: --budget-seconds refuses a plan whose worst case (every re-run) exceeds the box. Run the real driver, dry, with the template's own arguments."""
        for name in ('L8-LN-ladder8-past-131k', 'C16-LN-churn16'):
            entry = parsed('w2', name)
            box = int(row('w2', name)[5])
            for budget, expected in ((box * 60, 0), ((table_box('w2', name) - 1) * 60, 2)):
                lines = []
                argv = ['--image', 'img', '--profile', entry['profile'], '--plan', entry['gate_plan'], '--lengths', entry['gate_lengths'], '--max-tokens', str(entry['gate_max_tokens']),
                        '--budget-seconds', str(budget), '--results', os.path.join(self.scratch, name), '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json'), '--cards', 'quad', '--dry-run']
                if entry.get('gate_jit'):
                    argv += ['--jit', entry['gate_jit']]
                if entry.get('gate_audits'):
                    argv += ['--audits', entry['gate_audits']]
                if entry.get('gate_salt'):
                    argv += ['--salt', entry['gate_salt']]
                code = serving_gate.main(argv, devices=['/a', '/b', '/c', '/d'], log=lines.append)
                self.assertEqual(code, expected, '%s at %d s: %s' % (name, budget, lines[-3:]))
                if expected == 2:
                    self.assertTrue(any(line.startswith('C2_BOX verdict=REFUSED step=gate') for line in lines), lines)

    def test_every_gate_and_prefix_box_is_at_least_its_worst_case_or_a_clipped_prefix_box(self):
        for pack in PACKS:
            for line in order(pack):
                if line[5] == '-' or line[0] in CLIPPED_BOXES:
                    continue
                worst = table_box(pack, line[0])
                if worst is not None:
                    self.assertGreaterEqual(int(line[5]), worst, line[0])

    def test_the_clipped_prefix_jobs_boxes_are_under_their_worst_case_and_the_smoke_boxes_are_pinned(self):
        for name, box in CLIPPED_BOXES.items():
            self.assertEqual(int(row('w2', name)[5]), box, name)
            self.assertLess(box, table_box('w2', name), '%s: the box clips the arms (the prefix gate takes --box-seconds)' % name)
            self.assertGreaterEqual(box, 2 * 5, name)
        for name, box in SMOKE_BOXES.items():
            self.assertEqual(int(row('w2', name)[5]) if name in W2 else int(row('w1', name)[5]), box, name)
        self.assertEqual(table_box('w2', 'GG1-turns-hit-control-A'), 306)
        self.assertEqual(table_box('w2', 'HF-LN-levern-faults'), 153)

    def test_no_box_is_above_its_steps_own_timeout(self):
        ceilings = {'smoke': SMOKE_STEP_MINUTES, 'gate': STEP_CAP_MINUTES, 'prefix': STEP_CAP_MINUTES, 'replay': REPLAY_STEP_MINUTES}
        for pack in PACKS:
            for line in order(pack):
                if line[5] == '-':
                    continue
                actions = raw(pack, line[0])['C2_ACTIONS'].split()
                self.assertLessEqual(int(line[5]), max(ceilings[action] for action in actions if action in ceilings), line[0])


class TagTests(unittest.TestCase):
    def test_tags_are_the_free_allowlisted_set_in_order_one_a_job_and_disjoint(self):
        w1 = [line for line in order('w1') if line[4] != '-']
        w2 = [line for line in order('w2') if line[4] != '-']
        self.assertEqual((len(w1), len(w2)), (8, 21))
        tags = [int(line[4][1:]) for line in w1 + w2]
        self.assertEqual(tags, FREE[:8] + FREE[12:33])
        self.assertEqual(len(set(tags)), len(tags))
        self.assertTrue(set(tags).isdisjoint(BASELINE_TAGS), 'v538-v557 belong to the baseline window')
        for tag in tags:
            self.assertIn(tag, ALLOWLISTED_UNUSED)
        self.assertEqual(FREE, sorted(ALLOWLISTED_UNUSED - BASELINE_TAGS))
        self.assertEqual(len(FREE), 44)

    def test_the_reserves_and_the_unused_tags_are_named_and_counted(self):
        text = read_text('w1', 'ORDER.txt')
        self.assertIn('RESERVE is v566, v567, v568, v582', text)
        self.assertIn('RESERVE is v610, v611, v612, v613', read_text('w2', 'ORDER.txt'))
        self.assertEqual(FREE[33:37], [610, 611, 612, 613])
        self.assertEqual(len(FREE) - 37, 7)
        self.assertEqual(FREE[37], 614)
        self.assertIn('7 tags stay unused (v614-v620)', text)
        self.assertIn('v614-v620 stay unused', read_text('w2', 'ORDER.txt'))

    def test_stage_two_needs_the_tags_the_plan_counts(self):
        text = read_text('w1', 'ORDER.txt')
        self.assertEqual(STAGE2_TAGS, 59)
        left = len(FREE) - 37
        self.assertIn('STAGE 2 (W-3 to W-6, research) needs %d tags' % STAGE2_TAGS, text)
        reserve = 4 * 4   # the same reserve of four a window as Stage 1, for W-3 to W-6
        self.assertIn('at least %d MORE tags must be allowlisted before Stage 2 (%d with the same reserve of four a window' % (STAGE2_TAGS - left, STAGE2_TAGS + reserve - left), text)
        self.assertEqual((STAGE2_TAGS - left, STAGE2_TAGS + reserve - left), (52, 68))


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_with_the_job_parser_and_is_a_quad_job(self):
        for pack in PACKS:
            for name in templates(pack):
                with self.subTest(pack=pack, name=name):
                    self.assertTrue(parsed(pack, name))
                    self.assertEqual(raw(pack, name)['C2_CARDS'], 'quad')

    def test_the_node_agent_is_touched_only_by_the_first_and_the_last_job(self):
        for pack in PACKS:
            for name in templates(pack):
                actions = set(raw(pack, name)['C2_ACTIONS'].split())
                if name.startswith('A0X0'):
                    self.assertEqual(actions, {'agentstop', 'unserve', 'rescan', 'reset'})
                elif name == 'Z-handback':
                    self.assertEqual(actions, {'status', 'agentstart'})
                else:
                    self.assertFalse(actions & {'agentstop', 'agentstart', 'unserve', 'platform', 'push', 'rmi'}, name)

    def test_only_b0_builds_and_it_bakes_the_traffic_profile_into_the_window_image(self):
        for pack in PACKS:
            for name in templates(pack):
                entry = raw(pack, name)
                if name == 'B0-build-serve-11':
                    self.assertEqual((entry['C2_ACTIONS'], entry['C2_IMAGE_TAG'], entry['C2_BAKE_DEFAULT_PROFILE']), ('build', WINDOW, TRAFFIC))
                    self.assertEqual(parsed(pack, name)['bake_default_profile'], TRAFFIC)
                else:
                    self.assertNotIn('build', entry['C2_ACTIONS'].split(), name)
                    self.assertFalse(set(entry) & {'C2_BAKE_DEFAULT_PROFILE', 'C2_DRAFTER_CANDIDATES', 'C2_RMI_TAGS'}, name)

    def test_the_traffic_profile_serves_the_jobs_that_read_the_shipped_bytes_and_the_gate_twins_the_audited_ones(self):
        table = profiles()['profiles']
        self.assertNotIn('gate_only', table[TRAFFIC])
        self.assertIs(table[LN_AUDIT]['gate_only'], True)
        self.assertIs(table[LN_AUDIT_NOLNA]['gate_only'], True)
        wanted = {('w1', 'A1-LN-levern-audited-attach'): LN_AUDIT, ('w1', 'S0-CTL-control-attach-smoke'): DIGESTS,
                  ('w1', 'P1ab-LN-exactness-shared-lifecycle-evict'): LN_AUDIT_NOLNA, ('w1', 'E1-LN-exactness-eager'): LN_AUDIT_NOLNA,
                  ('w2', 'S0b-baked-default-smoke'): TRAFFIC, ('w2', 'HL-LN-hang-shapes-levern'): TRAFFIC, ('w2', 'HL-LN2-hang-shapes-levern'): TRAFFIC,
                  ('w2', 'HL-LN3-hang-shapes-levern'): TRAFFIC, ('w2', 'HF-LN-levern-faults'): LN_AUDIT, ('w2', 'L8-LN-ladder8-past-131k'): LN_AUDIT_NOLNA,
                  ('w2', 'C16-LN-churn16'): LN_AUDIT_NOLNA, ('w2', 'T0s-timed-production-bytes'): PLAIN, ('w2', 'TA1-timed-control-A'): PLAIN, ('w2', 'TA2-timed-control-A'): PLAIN,
                  ('w2', 'TA3-timed-control-A'): PLAIN, ('w2', 'TL1-timed-levern-L'): TRAFFIC, ('w2', 'TL2-timed-levern-L'): TRAFFIC, ('w2', 'S1-stall-control-A'): PLAIN,
                  ('w2', 'S2-stall-levern-B'): TRAFFIC, ('w2', 'GG1-turns-hit-control-A'): PLAIN, ('w2', 'GG2-turns-hit-levern-B'): TRAFFIC, ('w2', 'SR10-platform-replay'): TRAFFIC}
        for (pack, name), profile in wanted.items():
            entry = raw(pack, name)
            served = entry.get('C2_REPLAY_PROFILE') or entry.get('C2_PREFIX_PROFILE') or entry['C2_PROFILE']
            self.assertEqual(served, profile, name)
            self.assertEqual(table[served]['engine']['max-num-seqs'], 8, name)
            self.assertEqual(table[served]['engine']['max-model-len'], 262144, name)
            self.assertEqual(entry.get('C2_PREFIX_BASELINE', 'none'), 'none', name)

    def test_the_contract_boots_the_traffic_profile_and_refuses_it_with_a_gate_instrument(self):
        profile = dict(profiles()['profiles'][TRAFFIC], name=TRAFFIC)
        self.assertEqual(contract.levern_problems(profile), [])
        self.assertEqual(contract.gate_problems(profile, {}), [])

    def test_the_swap_targets_of_a1_ln_are_committed_profiles(self):
        table = profiles()['profiles']
        for suffix in ('-lean', '-pool', '-epochglobal'):
            self.assertIn(LN_AUDIT + suffix, table)
            self.assertIs(table[LN_AUDIT + suffix]['gate_only'], True)
            self.assertIn(LN_AUDIT + suffix, read_text('w1', 'ORDER.txt'))

    def test_every_smoke_test_exists_in_the_smoke_script_and_every_plan_applies_to_its_profile(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            source = handle.read()
        for pack in PACKS:
            for name in templates(pack):
                entry = raw(pack, name)
                for test in [item for item in entry.get('C2_SMOKE_TESTS', '').split(',') if item]:
                    self.assertTrue(("'%s'" % test) in source or ('def %s(' % test) in source, '%s names %s' % (name, test))
                for plan in [item for item in entry.get('C2_PREFIX_PLAN', '').split(',') if item]:
                    self.assertTrue(prefix_gate.plan_arms(plan, entry['C2_PREFIX_PROFILE'], None, profiles()), '%s: %s' % (name, plan))

    def test_the_timed_jobs_run_the_same_two_shapes_and_the_stall_jobs_the_same_four(self):
        timed = ('T0s-timed-production-bytes', 'TA1-timed-control-A', 'TL1-timed-levern-L', 'TA2-timed-control-A', 'TL2-timed-levern-L', 'TA3-timed-control-A')
        self.assertEqual(len(set(raw('w2', name)['C2_SMOKE_TESTS'] for name in timed)), 1)
        self.assertEqual(raw('w2', timed[0])['C2_SMOKE_TESTS'], 'warmup,coding,concurrent8_steady,concurrent8_code_equal,concurrent8_code_32k')
        self.assertNotIn('concurrent8_code_128k', raw('w2', timed[0])['C2_SMOKE_TESTS'])
        self.assertNotIn('concurrent8_skew', raw('w2', timed[0])['C2_SMOKE_TESTS'])
        self.assertEqual([name for name in timed if raw('w2', name)['C2_PROFILE'] == TRAFFIC], ['TL1-timed-levern-L', 'TL2-timed-levern-L'])
        self.assertEqual([order_line[2] for order_line in order('w2') if order_line[0] in timed], [PRODUCTION] + [WINDOW] * 5)
        self.assertEqual(raw('w2', 'S1-stall-control-A')['C2_SMOKE_TESTS'], raw('w2', 'S2-stall-levern-B')['C2_SMOKE_TESTS'])
        self.assertEqual(raw('w2', 'S1-stall-control-A')['C2_SMOKE_TESTS'], 'warmup,stall8_cold128k,stall8_cold262k,cold2_254k')
        self.assertEqual(raw('w2', 'GG1-turns-hit-control-A')['C2_PREFIX_PLAN'], 'agent-turns-prefix,levern-hit')
        self.assertEqual(raw('w2', 'GG1-turns-hit-control-A')['C2_PREFIX_AGENTS'], '8')

    def test_the_three_hang_jobs_are_three_templates_of_one_shape(self):
        names = ('HL-LN-hang-shapes-levern', 'HL-LN2-hang-shapes-levern', 'HL-LN3-hang-shapes-levern')
        self.assertEqual(len(set(tuple(sorted(raw('w2', name).items())) for name in names)), 1)

    def test_p1ab_is_one_prefix_job_of_two_plans_and_the_w1_attach_set_carries_the_levern_shapes(self):
        entry = raw('w1', 'P1ab-LN-exactness-shared-lifecycle-evict')
        self.assertEqual(entry['C2_PREFIX_PLAN'], 'exactness-shared,lifecycle-evict')
        self.assertEqual(entry['C2_ACTIONS'], 'reset prefix')
        for name in ('A1-LN-levern-audited-attach', 'S0-CTL-control-attach-smoke'):
            tests = raw('w1', name)['C2_SMOKE_TESTS'].split(',')
            for needed in ('concurrent8_skew', 'levern_equal', 'levern_equal_busy', 'levern_equal_long', 'concurrent8_steady', 'concurrent8_code_32k', 'concurrent8_code_equal'):
                self.assertIn(needed, tests, name)

    def test_sr10_carries_the_thin_layer_placeholder_and_replays_the_traffic_profile(self):
        entry = raw('w2', 'SR10-platform-replay')
        self.assertEqual((entry['C2_ACTIONS'], entry['C2_PLATFORM_IMAGE'], entry['C2_REPLAY_PROFILE']), ('reset replay', THIN, TRAFFIC))
        self.assertNotIn('C2_REPLAY_BUDGET_SMOKE', entry)
        self.assertIn('PLATFORM_REPLAY passed=True', read_text('w2', 'SR10-platform-replay.env'))

    def test_the_resets_are_all_four_and_each_job_resets_before_it_serves(self):
        for pack in PACKS:
            for name in templates(pack):
                actions = raw(pack, name)['C2_ACTIONS'].split()
                if {'smoke', 'gate', 'prefix', 'replay'} & set(actions):
                    self.assertEqual(actions[0], 'reset', name)
                    self.assertEqual(raw(pack, name)['C2_CARDS'], 'quad')



class NeedsTests(unittest.TestCase):
    def needs(self, pack):
        edges = {}
        for line in read_text(pack, 'ORDER.txt').splitlines():
            match = re.match(r'# NEEDS (.+?) <- (.+)$', line)
            if match:
                for name in match.group(1).split():
                    self.assertNotIn(name, edges)
                    edges[name] = match.group(2).split()
        return None, edges

    def test_the_needs_graph_names_real_jobs_and_has_no_cycle(self):
        for pack in PACKS:
            _, edges = self.needs(pack)
            full = [line[0] for line in order(pack)]

            def real(name):
                return any(item == name or item.startswith(name + '-') for item in full)
            for name, dependencies in edges.items():
                self.assertTrue(real(name), (pack, name))
                for dependency in dependencies:
                    self.assertTrue(real(dependency), (pack, dependency))

            def depth(name, seen=()):
                self.assertNotIn(name, seen)
                return 1 + max([depth(dep, seen + (name,)) for dep in edges.get(name, [])] or [0])
            for name in edges:
                depth(name)

    def test_w1_and_w2_dependencies(self):
        _, w1 = self.needs('w1')
        self.assertEqual(w1['A1-LN'], ['A0X0'])
        self.assertEqual(w1['S0-CTL'], ['A0X0'])
        self.assertEqual(w1['P1ab-LN'], ['S0-CTL'])
        _, w2 = self.needs('w2')
        self.assertEqual(w2['HL-LN2'], ['HL-LN'])
        self.assertEqual(w2['HL-LN3'], ['HL-LN2'])
        self.assertEqual(w2['SR10'], ['S0b'])
        for name in ('TA1', 'TL1', 'S1', 'S2', 'GG1', 'GG2'):
            self.assertEqual(w2[name], ['S0b', 'HL-LN3', 'HF-LN'])


class RuleTests(unittest.TestCase):
    def test_the_rollback_rule_is_in_the_w2_order_with_its_three_tiers(self):
        text = read_text('w2', 'ORDER.txt')
        for phrase in ('POST-CUTOVER ROLLBACK RULE (owner-approved 2026-10-07', 'Tier 0 (image rollback', 'one engine death or container restart', 'no [PHASE] execute line for 300 s with a request live',
                       'server errors', '>= 3 in an hour', 'max(1%, baseline + 3 sigma) over 24 h', 'Tier 1 (kill switch, no restart, no key)', 'levern.off when any request over 128k has a TTFT above 240 s',
                       'any Lever N quarantine line', 'prefix-reuse.off when the prefix grant ratio drops more than 25% below its baseline over 24 h', 'Tier 2 (statistical',
                       'at 72 h and finally at 7 days', 'at least 100 load-matched rounds a side', 'max(3%, floor)', 'at least 500 requests', 'max(20%, twice the day-to-day spread)'):
            self.assertIn(phrase, text)

    def test_the_cutover_rules_name_every_gate_and_the_owner(self):
        text = read_text('w2', 'ORDER.txt')
        for phrase in ('CUTOVER RULES', 'S0b, HL-LN, HL-LN2, HL-LN3 and SR10 PASS', 'HF-LN, L8-LN and C16-LN PASS', 'INSIDE the A-to-A floor at steady AND 32k', 'T0s inside the floor of TA1',
                       "owner's admin key (never in a script)", 'a P1a-CTL TIMEBOX blocks every cutover', 'pre-registered X4', 'gateway-salted two-turn conversation shows reused tokens > 0',
                       'a ninth refused', 'the second G pair'):
            self.assertIn(phrase, text)

    def test_g_np6_is_descoped_and_v579_is_the_named_reference(self):
        text = read_text('w1', 'ORDER.txt')
        self.assertIn('descoped G-NP6', text)
        self.assertIn('v579 (run 37240205944)', text)
        self.assertIn('v579 (run 37240205944', read_text('w1', 'S0-CTL-control-attach-smoke.env'))

    def test_the_timed_read_uses_the_comparators_pair_and_floor_per_length(self):
        text = read_text('w2', 'TA1-timed-control-A.env')
        for phrase in ('w2ln_timing_compare.py', 'floor TA1 TA2 TA3', 'pair TAn TLn', '100 matched eight-live rounds', '--max-load', '--load-gap 2', 'ZERO'):
            self.assertIn(phrase, text)
        with open(os.path.join(HERE, 'w2ln_timing_compare.py'), encoding='utf-8') as handle:
            source = handle.read()
        for window in ('steady', '32k'):
            self.assertIn("'%s'" % window, source)
        self.assertIn("add_parser('floor'", source)
        self.assertIn("add_parser('pair'", source)


def workflow_text():
    with open(os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml'), encoding='utf-8') as handle:
        return handle.read()


def step(name_prefix):
    """The text of one workflow step (from its name line to the next step's)."""
    text = workflow_text()
    start = text.index('      - name: ' + name_prefix)
    end = text.find('\n      - name: ', start + 10)
    return text[start:end if end > 0 else len(text)]


def source(name):
    with open(os.path.join(HERE, name), encoding='utf-8') as handle:
        return handle.read()


class WorkflowTests(unittest.TestCase):
    HUB = '/home/thatch/hf-cache/hub'

    def test_the_kill_switch_check_reads_the_hub_the_arms_mount_and_covers_the_replay(self):
        text = step('Kill-switch files must be absent')
        self.assertIn('hub="${C2_HUB_DIR:-%s}"' % self.HUB, text)
        self.assertNotIn('$HOME/', text)
        self.assertIn('%s:/models' % self.HUB, step('Smoke on the four-card set'))
        self.assertIn('flag=%s/.qwen-c2/prefix-reuse.off' % self.HUB, step('Prefix-reuse gates in the agent'))
        condition = text.split('shell: bash')[0]
        for action in ('gate', 'prefix', 'smoke', 'replay'):
            self.assertIn("contains(steps.job.outputs.actions, '%s')" % action, condition, action)

    def test_the_replay_step_removes_its_container_on_every_exit_and_gives_the_interrupt_a_long_grace(self):
        text = step('Replay the node agent')
        self.assertIn("trap 'docker rm -f qwen-c2-platform >/dev/null 2>&1 || true' EXIT", text)
        self.assertIn("default='qwen-c2-platform'", source('c2_platform_replay.py'))
        self.assertIn('timeout --signal=INT -k 300', text)
        self.assertNotIn('-k 120', text)
        self.assertIn('C2_JOB_STARTED', text)
        self.assertIn('C2_BOX verdict=TIMEBOX step=replay', text)
        self.assertIn('local:thin-layer', text)

    def test_the_reset_removes_job_owned_containers_before_it_checks_the_cards(self):
        text = step('Reset all four cards')
        removal, check = text.index('qwen-c2-platform|qwen-c2-smoke|qwen-c2-prefix-'), text.index('card_set_unheld')
        self.assertLess(removal, check)

    def test_every_box_counts_from_the_jobs_start_and_prints_a_verdict_line(self):
        smoke, prefix = step('Smoke on the four-card set'), step('Prefix-reuse gates in the agent')
        self.assertIn('box_end=$(( ${C2_JOB_STARTED:?} + BOX_MINUTES * 60 ))', smoke)
        self.assertIn('C2_BOX verdict=TIMEBOX step=smoke phase=ready', smoke)
        self.assertIn('C2_BOX verdict=TIMEBOX step=smoke phase=client', smoke)
        self.assertIn('BOX_MINUTES * 60 - ($(date +%s) - C2_JOB_STARTED)', prefix)
        self.assertIn('--box-seconds "$box_left"', prefix)
        self.assertIn('C2_BOX verdict=REFUSED step=gate', source('c2_serving_gate.py'))
        self.assertIn('C2_BOX verdict=TIMEBOX step=prefix', source('c2_prefix_gate.py'))

    @unittest.skipUnless(shutil.which('bash'), 'bash is needed to execute the readiness loop')
    def test_the_readiness_wait_is_a_deadline_not_a_count_of_polls(self):
        """Execute the smoke step's own loop with a fake docker and a curl that stalls: it must end at the deadline (here 3 s, not 4800 s)."""
        text = step('Smoke on the four-card set')
        start = text.index('          started=$(date +%s); ready=0')
        end = text.index('          echo "ready=$ready after')
        loop = text[start:end].replace('started + 4800', 'started + 3')
        self.assertIn('started + 3', loop)
        with tempfile.TemporaryDirectory() as folder:
            for name, body in (('docker', 'echo true'), ('curl', 'sleep 1; exit 22')):
                path = os.path.join(folder, name)
                with open(path, 'w', newline='\n') as handle:
                    handle.write('#!/bin/sh\n' + body + '\n')
                os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
            where = folder.replace(os.sep, '/')
            script = 'name=x; results=%s; BOX_MINUTES=; box_end=\n%s\necho "ready=$ready"\n' % (where, loop)
            began = time.time()
            done = subprocess.run(['bash', '-c', script], env=dict(os.environ, PATH=where + os.pathsep + os.environ['PATH']), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=120)
            took = time.time() - began
        self.assertIn('ready=0', done.stdout, done.stderr)
        self.assertLess(took, 60, 'a count-based loop would run 2400 polls')


class HygieneTests(unittest.TestCase):
    def test_no_private_string_in_any_file_of_either_pack_and_every_file_is_lf(self):
        for pack, folder in PACKS.items():
            for name in sorted(os.listdir(folder)):
                text = read_text(pack, name)
                hit = BANNED.search(text)
                self.assertIsNone(hit, '%s/%s: %s' % (pack, name, hit.group(0) if hit else ''))
                self.assertNotIn('\r', text, '%s/%s must have LF endings' % (pack, name))
                self.assertTrue(text.endswith('\n'), name)

    def test_no_script_holds_the_release_credential(self):
        for pack in PACKS:
            text = read_text(pack, 'ORDER.txt') + read_text(pack, 'README.md')
            self.assertIn("admin key", text)
            self.assertIn('never in a script', text)

    def test_the_readme_names_the_tags_and_the_templates_say_what_they_are(self):
        for pack in PACKS:
            readme = read_text(pack, 'README.md')
            if pack == 'w1':
                self.assertIn('Do not modify the runner group', readme)
            for name in templates(pack):
                self.assertTrue(read_text(pack, name + '.env').startswith('#'), name)
        self.assertIn('v558-v565', read_text('w1', 'README.md'))

    def test_the_test_is_allowlisted_in_the_cpu_workflow(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml'), encoding='utf-8') as handle:
            self.assertIn('test_tp4_stage1_windows', handle.read())

    def test_the_public_doc_exists_and_quotes_the_numbers(self):
        with open(os.path.join(ROOT, 'docs', 'tp4-short-windows.md'), encoding='utf-8') as handle:
            doc = handle.read()
        for phrase in ('471 min = 7.9 h', '445 min = 7.4 h', 'c2-packed-tp4-8x262k-ship-prefix-levern-traffic', 'C2_BOX_MINUTES', '10 s', '2 s'):
            self.assertIn(phrase, doc)
        self.assertIsNone(BANNED.search(doc))


if __name__ == '__main__':
    unittest.main()
