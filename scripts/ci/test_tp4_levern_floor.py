"""The decode-gap floor window pack of 2026-10-09 (references/tp4-levern-floor-jobs): one attach, ONE fresh-boot Lever N hang-shape smoke on tp4-serve-12b, the hand-back.

Every template parses with the job parser; the ORDER.txt lines match the files; the minutes, boxes, tags, NEEDS graph, directives and the clock rule are pinned; the admission walk (the private driver's `decide`,
restated here) says which jobs the clock admits at the central estimates, for a late start and when the smoke runs to its box; the profile the smoke names bakes in the 8 s decode-gap floor and carries no digest, no audit
and no gate-only marker; the hang-shape smoke is the HL-LN template of the first session pack apart from the image; the tags are allowlisted, distinct and none was pushed by an earlier session; no template or order
line names a rig, card, address, registry, digest or credential. Whether a tag is still unpushed on the remote is the driver's own launch check (git ls-remote), not a CPU test.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_levern_floor` from scripts/ci.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import test_tp4_stage1_windows as stage1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PACK = os.path.join(HERE, 'references', 'tp4-levern-floor-jobs')
FIRST_PACK = os.path.join(HERE, 'references', 'tp4-session-0808-jobs')
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml')
WINDOW = 'tp4-serve-12b'
TRAFFIC = stage1.R + 'ship-prefix-levern-traffic'

CAP, TARGET, RESERVE_MIN, HARD_CAP_MINUTE = 100, 70, 40, 120
READY_AFTER_LAST_JOB = 20                   # ZR 8, LM 6, TICK 6

# name -> (class, image, estimate, tag, box); the order is the ORDER's
JOBS = [
    ('A0X0-agentstop-unserve-rescan-reset', 'stop', WINDOW, 4, 'v597', None),
    ('HLF-hang-shapes-levern-floor', 'soft', WINDOW, 28, 'v598', 55),
    ('ZR-reset-all-four', 'hand', WINDOW, 8, 'v599', None),
    ('LM-links-remeasure', 'drv', '-', 6, None, None),
    ('TICK-topology-wait', 'drv', '-', 6, None, None),
    ('Z-handback', 'hand', WINDOW, 12, 'v601', None),
    ('DEPLOY-and-engine-load', 'drv', '-', 8, None, None),
]
RESERVE_TAGS = ['v602', 'v603', 'v606']
# tags an earlier session pushed (the first session's attach/qualify/audited jobs, the 0808b jobs that ran, both hand-backs, the manual re-run): none may come back
PUSHED_EARLIER = set(range(558, 562)) | {582, 583, 584, 585, 589, 590, 591, 604, 605, 616, 617, 619, 620}
EXPECT_FLAG = 'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S'


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


def walk(failed=(), durations=None, cap=CAP, reserve=RESERVE_MIN):
    """The driver's clock rule over the ORDER (`decide`): -> (ran, skipped {name: why}, E at the end)."""
    state, ran, skipped, clock, halted = {}, [], {}, 0, False
    for name, cls, _image, estimate, _tag, box in [entry for entry in JOBS if entry[1] not in ('drv', 'hand')]:
        if halted:
            skipped[name] = 'HALT'
            continue
        if name.startswith('HLF') and state.get('A0X0-agentstop-unserve-rescan-reset') != 'PASS':
            skipped[name] = 'NEEDS'
            continue
        if clock + (box if box is not None else estimate) + reserve > cap:
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
                if box is None:
                    self.assertNotIn('C2_BOX_MINUTES', raw(name))
                else:
                    self.assertEqual(raw(name)['C2_BOX_MINUTES'], str(box))
                    self.assertGreaterEqual(box, estimate)

    def test_the_hand_back_and_attach_templates_hold_the_end_sequence_contracts(self):
        self.assertEqual(raw('A0X0-agentstop-unserve-rescan-reset')['C2_ACTIONS'].split(), ['agentstop', 'unserve', 'rescan', 'reset'])
        self.assertEqual(raw('ZR-reset-all-four')['C2_ACTIONS'].split(), ['status', 'rescan', 'reset'])
        self.assertEqual(raw('Z-handback')['C2_ACTIONS'].split(), ['status', 'agentstart'])

    def test_the_hand_back_and_attach_templates_are_the_session_0808b_bytes_apart_from_the_image_and_header(self):
        def body(text):
            return [line.replace(WINDOW, 'IMAGE') for line in text.splitlines() if line and not line.startswith('#')]
        for name in ('A0X0-agentstop-unserve-rescan-reset', 'ZR-reset-all-four', 'Z-handback'):
            with self.subTest(name=name):
                first = [line.replace('tp4-serve-11', 'IMAGE') for line in read_text(name + '.env', FIRST_PACK).splitlines() if line and not line.startswith('#')] \
                    if os.path.exists(os.path.join(FIRST_PACK, name + '.env')) else None
                if first is not None:
                    self.assertEqual(body(read_text(name + '.env')), first)

    def test_the_hang_shape_smoke_is_the_first_packs_hl_ln_apart_from_the_image_and_the_class(self):
        def body(text):
            return [line.replace('tp4-serve-11', 'IMAGE').replace(WINDOW, 'IMAGE') for line in text.splitlines() if line and not line.startswith('#')]
        self.assertEqual(body(read_text('HLF-hang-shapes-levern-floor.env')), body(read_text('HL-LN-hang-shapes-levern.env', FIRST_PACK)))

    def test_no_template_or_order_line_names_a_rig_card_address_registry_digest_or_credential(self):
        for filename in sorted(os.listdir(PACK)):
            with self.subTest(filename=filename):
                self.assertIsNone(stage1.BANNED.search(read_text(filename)), filename)

    def test_templates_are_lf_and_files_end_with_a_newline(self):
        for filename in sorted(os.listdir(PACK)):
            with self.subTest(filename=filename):
                text = read_text(filename)
                self.assertNotIn('\r', text)
                self.assertTrue(text.endswith('\n'))

    def test_no_pack_file_is_a_stray(self):
        self.assertEqual(sorted(os.listdir(PACK)), sorted([entry[0] + '.env' for entry in JOBS if entry[1] != 'drv'] + ['ORDER.txt', 'README.md']))


class TheFloorIsUnderTest(unittest.TestCase):
    def test_the_smoke_names_the_traffic_profile_and_that_profile_bakes_in_the_8_s_floor(self):
        self.assertEqual(raw('HLF-hang-shapes-levern-floor')['C2_PROFILE'], TRAFFIC)
        document = stage1.profiles()['profiles']
        env = document[TRAFFIC]['env']
        self.assertEqual(env[EXPECT_FLAG], '8')
        self.assertEqual(env['QWEN_FAST_LEVERN_TTFT_TARGET_S'], '180')

    def test_the_profile_carries_no_digest_no_audit_and_no_gate_only_marker(self):
        document = stage1.profiles()['profiles']
        self.assertNotIn('gate_only', [key for key, value in document[TRAFFIC].items() if value])
        for key, value in document[TRAFFIC]['env'].items():
            if key == 'QWEN_FAST_GDN_PREFILL_CONV_AUDIT':    # the production profile's own prefill conv check
                continue
            if 'DIGEST' in key or 'AUDIT' in key:
                self.assertIn(str(value), ('', '0'), key)

    def test_the_smoke_tests_exist_and_the_skew_and_the_alternation_tests_are_named(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        tests = raw('HLF-hang-shapes-levern-floor')['C2_SMOKE_TESTS'].split(',')
        for test in tests:
            with self.subTest(test=test):
                self.assertIn("'%s'" % test, smoke, test)
        for needed in ('levern_arrival_during_prefill', 'concurrent8_skew', 'concurrent8_steady', 'concurrent8_drain'):
            self.assertIn(needed, tests)

    def test_no_job_runs_the_gate_or_the_fabric_probe(self):
        for name, cls, *_ in JOBS:
            if cls == 'drv':
                continue
            self.assertFalse({'gate', 'fabric', 'prefix', 'replay'} & set(raw(name).get('C2_ACTIONS', '').split()), name)

    def test_the_smoke_check_reads_the_governor_line_and_the_gate_asserts_the_floor(self):
        with open(os.path.join(HERE, 'levern_policy.py'), encoding='utf-8') as handle:
            self.assertIn("'[PINDIAG] lever N governor: ttft={} gap_floor={}'", handle.read())
        with open(os.path.join(HERE, 'c2_smoke_check.py'), encoding='utf-8') as handle:
            self.assertIn('gap_floor', handle.read())


class Tags(unittest.TestCase):
    def test_every_job_has_a_distinct_allowlisted_unused_tag_no_earlier_session_pushed(self):
        tags = [entry[4] for entry in JOBS if entry[4]]
        self.assertEqual(tags, ['v597', 'v598', 'v599', 'v601'])
        numbers = [int(tag[1:]) for tag in tags]
        self.assertEqual(len(numbers), len(set(numbers)))
        self.assertTrue(all(number in stage1.ALLOWLISTED_UNUSED for number in numbers))
        self.assertFalse(set(numbers) & stage1.BASELINE_TAGS)
        self.assertFalse(set(numbers) & PUSHED_EARLIER)

    def test_the_reserve_follows_and_is_allowlisted_unused_and_disjoint_from_the_jobs(self):
        self.assertEqual(directive('RESERVE'), [' '.join(RESERVE_TAGS)])
        used = set(entry[4] for entry in JOBS if entry[4])
        self.assertFalse(used & set(RESERVE_TAGS))
        numbers = [int(tag[1:]) for tag in RESERVE_TAGS]
        self.assertTrue(all(number in stage1.ALLOWLISTED_UNUSED for number in numbers))
        self.assertFalse(set(numbers) & PUSHED_EARLIER)


class Directives(unittest.TestCase):
    def test_the_directives_are_pinned(self):
        self.assertEqual(directive('END'), ['handback'])
        self.assertEqual(directive('CAP'), [str(CAP)])
        self.assertEqual(directive('TARGET'), [str(TARGET)])
        self.assertEqual(directive('RESERVE-MIN'), [str(RESERVE_MIN)])
        self.assertEqual(directive('CUTOVER-NEEDS'), [])
        self.assertNotIn('ARC-ZERO', read_text('ORDER.txt').replace('no ARC-ZERO range', ''))

    def test_the_clock_fits_the_smoke_box_and_ready_lands_inside_the_target(self):
        a0x0, hlf = JOBS[0], JOBS[1]
        self.assertLessEqual(a0x0[3] + hlf[5] + RESERVE_MIN, CAP)       # 4 + 55 + 40 <= 100: the smoke is admitted
        self.assertGreater(a0x0[3] + hlf[5] + 15 + RESERVE_MIN, CAP)    # a late start of 15 minutes refuses it (the wrapper shrinks CAP)
        self.assertLessEqual(sum(entry[3] for entry in JOBS[:2]) + READY_AFTER_LAST_JOB, TARGET)
        self.assertLess(CAP, HARD_CAP_MINUTE)
        self.assertEqual(sum(entry[3] for entry in JOBS if entry[1] in ('hand', 'drv')), RESERVE_MIN)

    def test_the_needs_graph_is_pinned(self):
        found = {}
        for line in read_text('ORDER.txt').splitlines():
            match = re.match(r'# NEEDS (.+) <- (.+)$', line)
            if match:
                for item in match.group(1).split():
                    found[item] = match.group(2).split()
        self.assertEqual(found, {'HLF': ['A0X0']})
        self.assertEqual([entry[0] for entry in JOBS if entry[1] == 'stop'], ['A0X0-agentstop-unserve-rescan-reset'])
        self.assertEqual([entry[0] for entry in JOBS if entry[1] == 'soft'], ['HLF-hang-shapes-levern-floor'])


class Schedule(unittest.TestCase):
    def test_at_the_central_estimates_both_card_jobs_run_and_the_window_is_ready_in_time(self):
        ran, skipped, clock = walk()
        self.assertEqual(ran, ['A0X0-agentstop-unserve-rescan-reset', 'HLF-hang-shapes-levern-floor'])
        self.assertEqual(skipped, {})
        self.assertEqual(clock + READY_AFTER_LAST_JOB + 12, 64)     # plus Z, up to the operator's release step
        self.assertLessEqual(clock + READY_AFTER_LAST_JOB, TARGET)

    def test_the_smoke_running_to_its_box_still_ends_before_the_cap_with_the_hand_back(self):
        ran, skipped, clock = walk(durations={'HLF-hang-shapes-levern-floor': 55})
        self.assertEqual(skipped, {})
        self.assertLessEqual(clock + RESERVE_MIN, CAP)

    def test_a_late_start_refuses_the_smoke_and_goes_straight_to_the_hand_back(self):
        ran, skipped, _ = walk(cap=CAP - 2)
        self.assertEqual(skipped, {'HLF-hang-shapes-levern-floor': 'TIME'})
        self.assertEqual(ran, ['A0X0-agentstop-unserve-rescan-reset'])

    def test_a_failed_attach_halts_everything_after_it_and_a_failed_smoke_does_not_strand_the_hand_back(self):
        ran, skipped, _ = walk(failed=('A0X0',))
        self.assertEqual(ran, ['A0X0-agentstop-unserve-rescan-reset'])
        self.assertEqual(skipped, {'HLF-hang-shapes-levern-floor': 'HALT'})
        ran, skipped, _ = walk(failed=('HLF',))
        self.assertEqual(skipped, {})
        self.assertEqual([entry[1] for entry in JOBS if entry[0].startswith('HLF')], ['soft'])


class Allowlist(unittest.TestCase):
    def test_the_cpu_workflow_names_this_test(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            self.assertIn('test_tp4_levern_floor', handle.read())

    def test_the_readme_names_the_clock_the_read_rules_and_the_missing_control(self):
        text = read_text('README.md')
        for needle in ('gap_floor=8', 'levern_alternation_problems', '238 s', 'tp4-serve-10', 'hand-back', 'v597'):
            self.assertIn(needle, text, needle)


if __name__ == '__main__':
    unittest.main()
