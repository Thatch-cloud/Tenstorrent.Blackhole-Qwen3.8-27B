"""The tp4/fusion-2 window pack (scripts/ci/references/fusion-jobs/F2): the two prefill levers (WPP oneq, M1 grid13) on the ship stack, one image, one window.

The pack is written by hand (the generator writes the per-lever and combined packs only), so this module is what keeps it honest: ORDER.txt lists exactly the folder's templates, each once, on the
image B0 builds, with a total that matches its header; B0 builds from the K64j-OQ graft; every NEEDS line names earlier jobs; the ARC-ZERO range starts at the audited attach and ends at the last
timed job; every job runs the profile its name says, and every lever profile is the control plus exactly the flags of the lever(s) and nothing else; the combined twins carry both levers and
the ship profile is not touched; every template parses through the job reader with rc 0."""

import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import make_fusion_profiles as fusion  # noqa: E402

PACK = HERE / 'references' / 'fusion-jobs' / 'F2'
PROFILES = HERE / 'qwen_c2_profiles.json'
READER = HERE / 'c2_serving_job.py'
IMAGE = 'tp4-fusion-6'
NS = fusion.NAMESPACE
ALL = NS + 'all'
BOTH = NS + 'all-oneq-grid13'
BOTH_AUDIT = NS + 'all-oneq-grid13-audit'
GRID = NS + 'all-grid13'
ONEQ_FLAGS = {'QWEN_FAST_SDPA_PF_ONEQ': '1'}
GRID_FLAGS = {'QWEN_FAST_GRID13': '1'}

# job -> (mode, profile, smoke tests, telemetry)
EXPECTED = {
    'B0-build-oq': ('stop', 'c2-packed-tp4', '', ''),
    'X0-status-rescan-reset': ('stop', '', '', ''),
    'A1-oneq-grid13-audited-attach': ('stop', BOTH_AUDIT, 'warmup,prefill_ladder_solo,concurrent8_steady', '1'),
    'PC1-prefill-control-a': ('soft', ALL, 'warmup,prefill_ladder_solo', '1'),
    'PL1-prefill-oneq-grid13-a': ('soft', BOTH, 'warmup,prefill_ladder_solo', '1'),
    'PC2-prefill-control-b': ('soft', ALL, 'warmup,prefill_ladder_solo', '1'),
    'PL2-prefill-oneq-grid13-b': ('soft', BOTH, 'warmup,prefill_ladder_solo', '1'),
    'DC1-decode-control-a': ('soft', ALL, 'warmup,concurrent8_steady', '1'),
    'DL1-decode-oneq-grid13-a': ('soft', BOTH, 'warmup,concurrent8_steady', '1'),
    'DC2-decode-control-b': ('soft', ALL, 'warmup,concurrent8_steady', '1'),
    'DL2-decode-oneq-grid13-b': ('soft', BOTH, 'warmup,concurrent8_steady', '1'),
    'GC1-m1-prefill-control-a': ('soft', ALL, 'warmup,prefill_ladder_solo', '1'),
    'GL1-m1-prefill-grid13-a': ('soft', GRID, 'warmup,prefill_ladder_solo', '1'),
    'GC2-m1-prefill-control-b': ('soft', ALL, 'warmup,prefill_ladder_solo', '1'),
    'GL2-m1-prefill-grid13-b': ('soft', GRID, 'warmup,prefill_ladder_solo', '1'),
    'Z-reset': ('soft', '', '', ''),
}


def order_lines():
    return [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


def directives(prefix):
    return [line[len(prefix):].split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.startswith(prefix)]


def resolve(token, names):
    """The window driver's job token: the full name or a prefix that ends at a '-'."""
    found = [name for name in names if name == token or name.startswith(token + '-')]
    return found


def values(name):
    return job.parse_env((PACK / (name + '.env')).read_text(encoding='utf-8'))


def profiles():
    return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']


class OrderTests(unittest.TestCase):
    def test_the_order_lists_exactly_the_folders_templates_each_once_on_the_one_image(self):
        lines = order_lines()
        names = [line[0] for line in lines]
        self.assertEqual(sorted(names), sorted(path.stem for path in PACK.glob('*.env')))
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names, list(EXPECTED), 'the order of the window')
        for name, mode, image, minutes in lines:
            self.assertEqual((mode, image, int(minutes) > 0), (EXPECTED[name][0], IMAGE, True), name)

    def test_the_total_in_the_header_is_the_sum_of_the_lines(self):
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        total = sum(int(line[3]) for line in order_lines())
        found = re.search(r'Z 10: (\d+) minutes \((\d+) h (\d+) min\) in all, (\d+) minutes of cards', text)
        self.assertIsNotNone(found)
        self.assertEqual((int(found.group(1)), int(found.group(2)) * 60 + int(found.group(3)), int(found.group(4))), (total, total, total - 40))

    def test_only_the_build_the_status_and_the_audited_attach_stop_the_window(self):
        self.assertEqual([line[0] for line in order_lines() if line[1] == 'stop'], ['B0-build-oq', 'X0-status-rescan-reset', 'A1-oneq-grid13-audited-attach'])

    def test_every_needs_line_names_earlier_jobs_and_every_job_but_b0_is_needed_to_follow_something(self):
        names = [line[0] for line in order_lines()]
        needs = directives('# NEEDS ')
        waiting = set()
        for tokens in needs:
            arrow = tokens.index('<-')
            left, right = tokens[:arrow], tokens[arrow + 1:]
            self.assertTrue(left and right, tokens)
            for token in left + right:
                self.assertEqual(len(resolve(token, names)), 1, token)
            first = min(names.index(resolve(token, names)[0]) for token in left)
            for token in right:
                self.assertLess(names.index(resolve(token, names)[0]), first, '%s must run before %s' % (token, left))
            waiting |= set(resolve(token, names)[0] for token in left)
        self.assertEqual(sorted(waiting), sorted(name for name in names if name not in ('B0-build-oq', 'Z-reset')), 'every job but B0 and Z has a NEEDS line')

    def test_the_lever_legs_wait_for_the_audited_attach_and_their_control_and_the_repeat_legs_for_the_leg_before(self):
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        for block in 'PDG':
            self.assertIn('# NEEDS %sL1 <- A1 %sC1\n' % (block, block), text)
            self.assertIn('# NEEDS %sC1 <- X0\n' % block, text)
            self.assertIn('# NEEDS %sC2 <- %sL1\n' % (block, block), text)
            self.assertIn('# NEEDS %sL2 <- %sC2\n' % (block, block), text)

    def test_the_arc_range_runs_from_the_audited_attach_to_the_last_timed_job_and_the_markers_are_in_the_text(self):
        names = [line[0] for line in order_lines()]
        ranges = directives('# ARC-ZERO ')
        self.assertEqual(ranges, [['A1', 'GL2']])
        first, last = (names.index(resolve(token, names)[0]) for token in ranges[0])
        self.assertEqual(names[first], 'A1-oneq-grid13-audited-attach')
        self.assertEqual(names[last], 'GL2-m1-prefill-grid13-b')
        timed = [index for index, name in enumerate(names) if name[0] in 'PDG']
        self.assertTrue(all(first <= index <= last for index in timed))
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        self.assertIn('ARC PAUSE / RESTORE', text)
        for name in names:
            if name[0] in 'PDG':
                self.assertIn('ARC runner sets are at ZERO', (PACK / (name + '.env')).read_text(encoding='utf-8'), name)

    def test_the_read_rules_and_the_estimate_labels_are_in_the_order(self):
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('READ RULES', 'prefill_ladder_report.py', 'w2ln_timing_compare.py', 'NO LOSS', 'NO-GO', 'bisect with M1/M1A and WPP/OQ-A1', 'K64j-OQ'):
            self.assertIn(word, text)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_through_the_reader_with_rc_0(self):
        for path in sorted(PACK.glob('*.env')):
            with self.subTest(path.name):
                result = subprocess.run([sys.executable, '-s', str(READER), str(path), str(PROFILES)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(HERE.parent.parent))
                self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', 'replace'))

    def test_each_job_runs_the_profile_tests_and_telemetry_its_name_says(self):
        for name, (_mode, profile, tests, telemetry) in EXPECTED.items():
            with self.subTest(name):
                found = values(name)
                self.assertEqual(found['C2_IMAGE_TAG'], IMAGE)
                self.assertEqual(found.get('C2_PROFILE', ''), profile)
                self.assertEqual(found.get('C2_SMOKE_TESTS', ''), tests)
                self.assertEqual(found.get('C2_TELEMETRY', ''), telemetry)

    def test_b0_builds_the_ship_default_on_the_oneq_graft_and_no_other_job_bakes(self):
        b0 = values('B0-build-oq')
        self.assertEqual((b0['C2_ACTIONS'], b0['C2_BAKE_DEFAULT_PROFILE'], b0['C2_BAKE_GRAFT']), ('build', fusion.PARENT, 'K64j-OQ'))
        for name in EXPECTED:
            if name != 'B0-build-oq':
                self.assertNotIn('C2_BAKE_GRAFT', values(name), name)
                self.assertNotIn('C2_BAKE_DEFAULT_PROFILE', values(name), name)

    def test_a_ladder_only_job_states_why_it_omits_the_steady_window(self):
        for name, (_mode, _profile, tests, _telemetry) in EXPECTED.items():
            if tests == 'warmup,prefill_ladder_solo':
                self.assertIn('solo prefill ladder', values(name)['C2_SMOKE_PARTIAL'], name)
            else:
                self.assertNotIn('C2_SMOKE_PARTIAL', values(name), name)

    def test_the_abab_alternates_control_and_lever_in_every_block(self):
        for block in 'PDG':
            legs = [values(name)['C2_PROFILE'] for name in EXPECTED if name[0] == block and name[1] in 'CL']
            self.assertEqual(len(legs), 4)
            self.assertEqual(legs[0], legs[2])
            self.assertEqual(legs[1], legs[3])
            self.assertEqual(legs[0], ALL)
            self.assertNotEqual(legs[0], legs[1])

    def test_the_boot_boxes_cover_the_audited_attach_and_the_ladders(self):
        self.assertEqual(int(values('A1-oneq-grid13-audited-attach')['C2_BOX_MINUTES']), 100)
        for name in EXPECTED:
            if name[0] in 'PG':
                self.assertEqual(int(values(name)['C2_BOX_MINUTES']), 60, name)


class ProfileTests(unittest.TestCase):
    def test_each_lever_profile_is_the_control_plus_exactly_the_levers_flags_and_gate_only(self):
        found = profiles()
        control, audit_control = found[ALL], found[NS + 'all-audit']
        for name, flags, parent in ((BOTH, dict(ONEQ_FLAGS, **GRID_FLAGS), control), (GRID, GRID_FLAGS, control),
                                    (BOTH_AUDIT, dict(ONEQ_FLAGS, QWEN_FAST_SDPA_PF_ONEQ_AUDIT='1', QWEN_FAST_GRID13='1', QWEN_FAST_GRID13_AUDIT='1'), audit_control)):
            with self.subTest(name):
                profile = found[name]
                self.assertEqual({key: value for key, value in profile['env'].items() if key not in flags}, parent['env'])
                self.assertEqual({key: profile['env'][key] for key in flags}, flags)
                self.assertIs(profile['gate_only'], True)
                self.assertNotIn('owner_traffic_waiver', profile)

    def test_the_control_is_the_combined_arm_of_the_ship_stack_with_no_prefill_lever(self):
        env = profiles()[ALL]['env']
        for flag in ('QWEN_FAST_SDPA_PF_ONEQ', 'QWEN_FAST_SDPA_PF_ONEQ_AUDIT', 'QWEN_FAST_GRID13', 'QWEN_FAST_GRID13_AUDIT'):
            self.assertNotIn(flag, env)

    def test_the_ship_profile_and_its_waiver_are_not_touched_by_the_prefill_levers(self):
        found = profiles()
        ship = found[NS + 'traffic']
        for flag in ('QWEN_FAST_SDPA_PF_ONEQ', 'QWEN_FAST_SDPA_PF_ONEQ_AUDIT', 'QWEN_FAST_GRID13', 'QWEN_FAST_GRID13_AUDIT'):
            self.assertNotIn(flag, ship['env'])
        self.assertTrue(ship['owner_traffic_waiver']['decision'].startswith(('APPROVED', 'PENDING')))
        self.assertNotIn('gate_only', ship)
        self.assertEqual(ship['env'], found[ALL]['env'], 'the ship profile is the production profile plus exactly the combined arm\'s env')


if __name__ == '__main__':
    unittest.main()
