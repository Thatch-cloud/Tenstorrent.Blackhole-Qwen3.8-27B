"""The second chained session pack of 2026-10-08 (references/tp4-session-0808b-jobs): performance and robustness on the traffic and production profiles, NO digest and NO audit.

Every template parses with the job parser; the ORDER.txt lines match the files; the minutes, boxes, tags, NEEDS graph, directives and the clock rule are pinned; the admission walk (the private driver's `decide`,
restated here) says which jobs the clock admits at the central estimates, for a late start and when every job runs long; no profile a job names carries a digest, an audit or a gate-only marker, and no prefix-gate arm
a job runs is an audited or an audit-cost arm; the tags are allowlisted, distinct and none is a tag of the first session's audited or hand-back jobs; no template or order line names a rig, card, address, registry, digest
or credential. Whether a tag is still unpushed on the remote is the driver's own launch check (git ls-remote), not a CPU test.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_session_0808b` from scripts/ci.
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
PACK = os.path.join(HERE, 'references', 'tp4-session-0808b-jobs')
FIRST_PACK = os.path.join(HERE, 'references', 'tp4-session-0808-jobs')
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml')
PRODUCTION, WINDOW = 'tp4-serve-10', 'tp4-serve-11'

CAP, TARGET, RESERVE_MIN = 495, 465, 60     # a start at 11:45Z: READY 19:30Z is minute 465, 20:00Z (the soft deadline) minute 495
HARD_CAP_MINUTE = 525                       # 20:30Z
READY_AFTER_LAST_JOB = 20                   # ZR 8, LM 6, TICK 6
START_UTC = 11 * 60 + 45

# name -> (class, image, estimate, tag, box); the order is the ORDER's
JOBS = [
    ('A0X0-agentstop-unserve-rescan-reset', 'stop', WINDOW, 4, 'v582', None),
    ('S0b-baked-default-smoke', 'stop', WINDOW, 11, 'v583', 20),
    ('HL-LN-hang-shapes-levern', 'stop', WINDOW, 28, 'v584', 55),
    ('HL-LN2-hang-shapes-levern', 'stop', WINDOW, 28, 'v585', 55),
    ('HL-LN3-hang-shapes-levern', 'stop', WINDOW, 28, 'v589', 55),
    ('T0s-timed-production-bytes', 'opt', PRODUCTION, 10, 'v590', 20),
    ('TA1-timed-control-A', 'opt', WINDOW, 10, 'v591', 20),
    ('TL1-timed-levern-L', 'opt', WINDOW, 10, 'v592', 20),
    ('TA2-timed-control-A', 'opt', WINDOW, 10, 'v595', 20),
    ('TL2-timed-levern-L', 'opt', WINDOW, 10, 'v596', 20),
    ('TA3-timed-control-A', 'opt', WINDOW, 10, 'v597', 20),
    ('SM-streams-agent-shapes', 'soft', WINDOW, 14, 'v598', 30),
    ('S1-stall-control-A', 'opt', WINDOW, 27, 'v599', 50),
    ('S2-stall-levern-B', 'opt', WINDOW, 27, 'v601', 50),
    ('SR10-platform-replay', 'soft', WINDOW, 25, 'v602', 60),
    ('GG1-turns-hit-control-A', 'opt', WINDOW, 28, 'v603', 55),
    ('GG2-turns-hit-levern-B', 'opt', WINDOW, 28, 'v612', 55),
    ('S0b2-late-reattach-smoke', 'soft', WINDOW, 11, 'v613', 20),
    ('DA1-deep128k-control-A', 'opt', WINDOW, 18, 'v614', 40),
    ('DL1-deep128k-levern-L', 'opt', WINDOW, 18, 'v615', 40),
    ('ZR-reset-all-four', 'hand', WINDOW, 8, 'v616', None),
    ('LM-links-remeasure', 'drv', '-', 6, None, None),
    ('TICK-topology-wait', 'drv', '-', 6, None, None),
    ('Z-handback', 'hand', WINDOW, 12, 'v617', None),
    ('DEPLOY-and-engine-load', 'drv', '-', 8, None, None),
]
BY_NAME = dict((entry[0], entry) for entry in JOBS)
RESERVE_TAGS = ['v618', 'v619', 'v620']
NEEDS = {
    'S0b': ['A0X0'], 'HL-LN': ['S0b'], 'HL-LN2': ['HL-LN'], 'HL-LN3': ['HL-LN2'],
    'SM': ['S0b'], 'SR10': ['S0b'], 'S0b2': ['S0b'],
    'T0s': ['S0b', 'HL-LN3'], 'TA1': ['S0b', 'HL-LN3'], 'TL1': ['S0b', 'HL-LN3'], 'TA2': ['S0b', 'HL-LN3'], 'TL2': ['S0b', 'HL-LN3'], 'TA3': ['S0b', 'HL-LN3'],
    'S1': ['S0b', 'HL-LN3'], 'S2': ['S0b', 'HL-LN3'], 'GG1': ['S0b', 'HL-LN3'], 'GG2': ['S0b', 'HL-LN3'], 'DA1': ['S0b', 'HL-LN3'], 'DL1': ['S0b', 'HL-LN3'],
}
# the launch config lists SR10 until the thin layer is staged on the runner (the replay fails after its reset without it)
OWNER_SKIPPED = ('SR10-platform-replay',)
FIRST_SESSION_TAGS = set(range(558, 569)) | set(range(604, 612))   # its attach/qualify/audited jobs and its hand-back plus reserve
TIMED = ('T0s', 'TA1', 'TL1', 'TA2', 'TL2', 'TA3', 'S1', 'S2', 'GG1', 'GG2', 'DA1', 'DL1')
# the dropped audited arms of the first pack: none may come back as a file
DROPPED = ('KVQ', 'A1-LN', 'S0-CTL', 'L8-LN', 'C16-LN', 'P1ab-LN', 'P1-CTLR', 'P1b-CTL', 'E1-LN', 'HF-LN')


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
    text = read_text(name + '.env')
    return job.read_job(job.parse_env(text), sorted(stage1.profiles()['profiles']), root=ROOT)


def token_names(token):
    """The job names a driver token means: the exact name, else every job whose name starts with token + '-'."""
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
    """The driver's clock rule over the ORDER (`decide`): -> (ran, skipped {name: why}, E at the end). `failed` names jobs that run and FAIL (a stop-class one halts),
    `durations` maps a name to the minutes it takes (default: the estimate). An opt block is admitted whole on the sum of its boxes; the class of the row before a block is read
    whether or not that row runs (an owner-skipped row still separates two blocks)."""
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

    def test_the_image_of_a_template_is_the_orders(self):
        for name, cls, image, _estimate, _tag, _box in JOBS:
            if cls == 'drv':
                continue
            with self.subTest(name=name):
                self.assertEqual(raw(name)['C2_IMAGE_TAG'], image)

    def test_every_template_box_equals_the_order_box_and_short_steps_have_none(self):
        for name, cls, _image, estimate, _tag, box in JOBS:
            if cls == 'drv':
                continue
            if cls == 'hand' or name.startswith('A0X0'):
                self.assertNotIn('C2_BOX_MINUTES', raw(name), name)
                continue
            self.assertEqual(raw(name).get('C2_BOX_MINUTES'), str(box), name)
            self.assertGreaterEqual(box, estimate, name)
            self.assertLessEqual(box, 3 * estimate, name)

    def test_the_templates_the_new_jobs_share_with_the_first_pack_are_its_bytes_apart_from_the_comment_header(self):
        def body(text):
            return [line for line in text.splitlines() if line and not line.startswith('#')]
        for name in [entry[0] for entry in JOBS if entry[1] != 'drv' and os.path.exists(os.path.join(FIRST_PACK, entry[0] + '.env'))]:
            with self.subTest(name=name):
                self.assertEqual(body(read_text(name + '.env')), body(read_text(name + '.env', FIRST_PACK)))

    def test_the_hand_back_templates_hold_the_end_sequence_contracts(self):
        self.assertIn('reset', raw('ZR-reset-all-four')['C2_ACTIONS'].split())
        z = raw('Z-handback')['C2_ACTIONS'].split()
        self.assertIn('agentstart', z)
        self.assertNotIn('reset', z)
        self.assertNotIn('rescan', z)
        a0x0 = raw('A0X0-agentstop-unserve-rescan-reset')['C2_ACTIONS'].split()
        self.assertEqual(a0x0, ['agentstop', 'unserve', 'rescan', 'reset'])

    def test_the_audited_arms_of_the_first_pack_are_not_in_this_one(self):
        names = [entry[0] for entry in JOBS]
        for dropped in DROPPED:
            self.assertFalse([name for name in names if name == dropped or name.startswith(dropped + '-')], dropped)
            self.assertFalse([name for name in os.listdir(PACK) if name.startswith(dropped + '-')], dropped)
        self.assertEqual(directive('END'), ['handback'])
        self.assertEqual(directive('CUTOVER-NEEDS'), [])

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


class NoDigestNoAudit(unittest.TestCase):
    """The reason the pack exists: nothing it runs pays the whole-pool host digest."""

    def profile_names(self, name):
        entry = raw(name)
        return [entry[key] for key in ('C2_PROFILE', 'C2_REPLAY_PROFILE', 'C2_PREFIX_PROFILE') if entry.get(key)]

    def test_every_profile_a_job_names_is_a_traffic_or_production_profile_without_digests_or_audits(self):
        document = stage1.profiles()['profiles']
        allowed_audit_words = ('QWEN_FAST_GDN_PREFILL_CONV_AUDIT',)   # the production profile's own prefill conv check
        seen = set()
        for entry in JOBS:
            if entry[1] == 'drv':
                continue
            for profile in self.profile_names(entry[0]):
                seen.add(profile)
                with self.subTest(job=entry[0], profile=profile):
                    self.assertIn(profile, (stage1.R + 'ship-prefix', stage1.R + 'ship-prefix-levern-traffic'))
                    self.assertNotIn('gate_only', [key for key, value in document[profile].items() if value])
                    env = document[profile]['env']
                    for key, value in env.items():
                        if key in allowed_audit_words:
                            continue
                        if 'DIGEST' in key or 'AUDIT' in key:
                            self.assertIn(str(value), ('', '0'), key)
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
                        self.assertIn(arm['scenario'], prefix_gate.TIMED_SCENARIOS)

    def test_the_smoke_tests_exist_and_no_job_runs_the_audit_instruments(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        for entry in JOBS:
            if entry[1] == 'drv' or not raw(entry[0]).get('C2_SMOKE_TESTS'):
                continue
            for test in raw(entry[0])['C2_SMOKE_TESTS'].split(','):
                with self.subTest(job=entry[0], test=test):
                    self.assertIn("'%s'" % test, smoke, test)
        for entry in JOBS:
            if entry[1] == 'drv':
                continue
            self.assertNotIn('gate', raw(entry[0]).get('C2_ACTIONS', '').split(), entry[0])
            self.assertNotIn('fabric', raw(entry[0]).get('C2_ACTIONS', '').split(), entry[0])

    def test_the_new_profiles_bake_into_the_window_image_and_the_control_is_the_production_profile(self):
        self.assertEqual(raw('TA1-timed-control-A')['C2_PROFILE'], stage1.R + 'ship-prefix')
        self.assertEqual(raw('TL1-timed-levern-L')['C2_PROFILE'], stage1.R + 'ship-prefix-levern-traffic')
        self.assertEqual(raw('T0s-timed-production-bytes')['C2_IMAGE_TAG'], PRODUCTION)
        self.assertEqual(raw('DA1-deep128k-control-A')['C2_SMOKE_TESTS'], raw('DL1-deep128k-levern-L')['C2_SMOKE_TESTS'])


class Tags(unittest.TestCase):
    def test_every_job_has_a_distinct_allowlisted_tag_that_is_not_the_first_sessions(self):
        tags = [entry[4] for entry in JOBS if entry[4]]
        self.assertEqual(len(tags), len(set(tags)))
        self.assertEqual(len(tags), 22)
        numbers = [int(tag[1:]) for tag in tags]
        self.assertEqual(numbers, sorted(numbers))
        self.assertTrue(all(number in stage1.ALLOWLISTED_UNUSED for number in numbers))
        self.assertFalse(set(numbers) & stage1.BASELINE_TAGS)
        self.assertFalse(set(numbers) & FIRST_SESSION_TAGS)

    def test_the_reserve_follows_and_is_allowlisted_and_unused(self):
        self.assertEqual(directive('RESERVE'), [' '.join(RESERVE_TAGS)])
        used = set(entry[4] for entry in JOBS if entry[4])
        self.assertFalse(used & set(RESERVE_TAGS))
        self.assertTrue(all(int(tag[1:]) in stage1.ALLOWLISTED_UNUSED for tag in RESERVE_TAGS))
        self.assertEqual(max(int(tag[1:]) for tag in used | set(RESERVE_TAGS)), 620)

    def test_the_tags_are_the_first_sessions_tail_tags_and_the_unused_top_of_the_allowlist(self):
        numbers = [int(entry[4][1:]) for entry in JOBS if entry[4]]
        self.assertEqual(numbers, [582, 583, 584, 585, 589, 590, 591, 592, 595, 596, 597, 598, 599, 601, 602, 603, 612, 613, 614, 615, 616, 617])
        first = set(int(row[4][1:]) for row in [line.split() for line in read_text('ORDER.txt', FIRST_PACK).splitlines() if line.strip() and not line.startswith('#')] if row[4] != '-')
        # every tag of this pack below 612 is a tag the first pack's tail names (never pushed unless that session ran its tail); 612 and above are in neither pack's job list
        self.assertTrue(set(n for n in numbers if n < 612) <= first)
        self.assertFalse(set(n for n in numbers if n >= 612) & first)
        self.assertIn('NEVER run this pack beside tp4-session-0808', read_text('ORDER.txt'))


class Directives(unittest.TestCase):
    def test_the_directives_are_pinned(self):
        self.assertEqual(directive('END'), ['handback'])
        self.assertEqual(directive('CAP'), [str(CAP)])
        self.assertEqual(directive('TARGET'), [str(TARGET)])
        self.assertEqual(directive('RESERVE-MIN'), [str(RESERVE_MIN)])
        self.assertIn('# ARC-ZERO T0s DL1', read_text('ORDER.txt').splitlines())

    def test_the_clock_matches_the_owners_times_for_a_start_at_1145z(self):
        self.assertEqual(START_UTC + TARGET, 19 * 60 + 30)
        self.assertEqual(START_UTC + HARD_CAP_MINUTE, 20 * 60 + 30)
        self.assertEqual(START_UTC + CAP, 20 * 60)
        last_end = CAP - RESERVE_MIN
        self.assertLessEqual(last_end + READY_AFTER_LAST_JOB, TARGET)

    def test_the_arc_zero_range_covers_every_timed_job_in_one_stretch(self):
        names = [entry[0] for entry in JOBS]
        first = next(i for i, name in enumerate(names) if name.startswith('T0s'))
        last = next(i for i, name in enumerate(names) if name.startswith('DL1'))
        for token in TIMED:
            index = next(i for i, name in enumerate(names) if name.startswith(token + '-'))
            self.assertTrue(first <= index <= last, token)
        # nothing before the block is read against CI load
        for name in names[:first]:
            self.assertFalse(name.startswith(TIMED), name)

    def test_the_needs_graph_is_pinned(self):
        found = {}
        for line in read_text('ORDER.txt').splitlines():
            match = re.match(r'# NEEDS (.+) <- (.+)$', line)
            if match:
                for item in match.group(1).split():
                    found[item] = match.group(2).split()
        self.assertEqual(found, NEEDS)

    def test_every_needs_token_names_a_job_and_the_stop_class_is_what_the_header_says(self):
        for token in list(NEEDS) + [item for items in NEEDS.values() for item in items]:
            self.assertTrue(token_names(token), token)
        self.assertEqual(token_names('HL-LN'), ['HL-LN-hang-shapes-levern'])
        self.assertEqual(token_names('S0b'), ['S0b-baked-default-smoke'])
        self.assertEqual([entry[0] for entry in JOBS if entry[1] == 'stop'], ['A0X0-agentstop-unserve-rescan-reset', 'S0b-baked-default-smoke', 'HL-LN-hang-shapes-levern', 'HL-LN2-hang-shapes-levern', 'HL-LN3-hang-shapes-levern'])

    def test_the_opt_blocks_are_separated_by_soft_rows_so_each_pair_is_its_own_block(self):
        classes = [entry[1] for entry in planned_jobs()]
        blocks, run = [], []
        for entry in planned_jobs():
            if entry[1] == 'opt':
                run.append(entry[0].split('-')[0])
            elif run:
                blocks.append(run)
                run = []
        if run:
            blocks.append(run)
        self.assertEqual(blocks, [['T0s', 'TA1', 'TL1', 'TA2', 'TL2', 'TA3'], ['S1', 'S2'], ['GG1', 'GG2'], ['DA1', 'DL1']])
        self.assertEqual(classes.count('opt'), 12)


class Schedule(unittest.TestCase):
    def test_the_planned_jobs_total_at_the_central_estimates(self):
        planned = planned_jobs()
        self.assertEqual(sum(entry[3] for entry in planned), 355)
        self.assertEqual(sum(entry[3] for entry in planned if entry[0] not in OWNER_SKIPPED), 330)

    def test_at_a_start_at_1145z_every_job_but_the_owner_skipped_replay_runs_and_ready_lands_two_hours_early(self):
        ran, skipped, clock = walk()
        self.assertEqual(clock, 330)
        self.assertEqual(skipped, {'SR10-platform-replay': 'OWNER'})
        self.assertEqual(ran, [entry[0] for entry in planned_jobs() if entry[0] not in OWNER_SKIPPED])
        self.assertEqual(START_UTC + clock + READY_AFTER_LAST_JOB, 17 * 60 + 35)    # 17:35Z
        self.assertLessEqual(clock + READY_AFTER_LAST_JOB, TARGET)

    def test_the_thin_layer_staged_adds_the_replay_inside_the_target(self):
        ran, skipped, clock = walk(skip=())
        self.assertIn('SR10-platform-replay', ran)
        self.assertEqual(skipped, {})
        self.assertEqual(clock, 355)
        self.assertLessEqual(clock + READY_AFTER_LAST_JOB, TARGET)

    def test_a_late_start_refuses_the_tail_first_and_keeps_the_performance_block(self):
        # the wrapper shrinks CAP by the minutes of delay (the soft deadline stays 20:00Z). The blocks are refused whole, last first: the deep pair from 62 minutes late, the agent-turn pair
        # from about 2 h, the stall pair from about 3 h; the hang runs and the timed block stay until the start is more than 3 h late.
        ran, skipped, _ = walk(cap=CAP - 61)
        self.assertEqual(skipped, {'SR10-platform-replay': 'OWNER'})
        ran, skipped, _ = walk(cap=CAP - 90)
        self.assertEqual(sorted(skipped), ['DA1-deep128k-control-A', 'DL1-deep128k-levern-L', 'SR10-platform-replay'])
        self.assertEqual((skipped['DA1-deep128k-control-A'], skipped['DL1-deep128k-levern-L']), ('TIME', 'BLOCK'))
        ran, skipped, _ = walk(cap=CAP - 120)
        self.assertEqual(sorted(name for name, why in skipped.items() if why == 'TIME'), ['DA1-deep128k-control-A', 'GG1-turns-hit-control-A'])
        ran, skipped, _ = walk(cap=CAP - 180)
        self.assertIn('TA3-timed-control-A', ran)
        self.assertEqual(sorted(name for name, why in skipped.items() if why == 'TIME'), ['DA1-deep128k-control-A', 'GG1-turns-hit-control-A', 'S1-stall-control-A'])
        self.assertEqual(skipped['S2-stall-levern-B'], 'BLOCK')

    def test_a_stop_class_failure_halts_everything_after_it(self):
        for failed, last in (('S0b', 'S0b-baked-default-smoke'), ('HL-LN2', 'HL-LN2-hang-shapes-levern')):
            ran, skipped, _ = walk(failed=(failed,))
            self.assertEqual(ran[-1], last)
            self.assertEqual(set(skipped.values()), {'HALT', 'OWNER'})

    def test_a_soft_or_opt_failure_does_not_stop_the_window(self):
        ran, skipped, _ = walk(failed=('SM', 'S1', 'S0b2'))
        self.assertIn('DL1-deep128k-levern-L', ran)
        self.assertIn('GG2-turns-hit-levern-B', ran)

    def test_when_every_job_runs_to_one_and_a_half_times_its_estimate_the_clock_still_ends_before_the_target(self):
        pessimistic = dict((entry[0], min(entry[5] or entry[3], int(entry[3] * 1.5))) for entry in planned_jobs())
        ran, skipped, clock = walk(durations=pessimistic)
        self.assertLessEqual(clock + READY_AFTER_LAST_JOB, TARGET)
        # the hang runs, the timed block, the stall pair and the deep pair still run; the agent-turn pair is refused whole for its boxes
        self.assertEqual(sorted(skipped), ['GG1-turns-hit-control-A', 'GG2-turns-hit-levern-B', 'SR10-platform-replay'])
        for name in ('S0b-baked-default-smoke', 'HL-LN3-hang-shapes-levern', 'TA3-timed-control-A', 'S2-stall-levern-B', 'DL1-deep128k-levern-L'):
            self.assertIn(name, ran)

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
        self.assertIn('test_tp4_session_0808b', text)

    def test_the_readme_names_the_clock_the_arc_rule_and_the_dropped_arms(self):
        text = read_text('README.md')
        for needle in ('19:30Z', '20:30Z', 'ARC_ZERO_ACK', 'SKIP_JOBS', 'no digest', 'P1ab-LN', 'hand-back'):
            self.assertIn(needle, text, needle)


if __name__ == '__main__':
    unittest.main()
