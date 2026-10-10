"""The tp4/prefill window pack (references/tp4-prefill-jobs) on the CPU: every template parses with the job parser, the ORDER rows match the files and the two image tags, the dependency lines resolve as the window
driver resolves them (the exact name, else the first name that starts with the token and a dash), the ladder, the governor ABAB and the prefill profile name the tests and plans that exist, the ABAB alternates the production
profile with its generated twin on one image, the profiled job comes last before the reset, and nothing in the pack or its docs names a host, an address, a registry, a digest or a home path.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_prefill_jobs` from scripts/ci."""

import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import make_adaptive_profiles as adaptive  # noqa: E402
import ops_profile_plan as ops  # noqa: E402

PACK = HERE / 'references' / 'tp4-prefill-jobs'
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
PRODUCTION = adaptive.PARENT
TWIN = adaptive.TWIN
BASE_TAG, GOVERNOR_TAG = 'tp4-next2-1', 'tp4-prefill-1'
ORDER_NAMES = ['B0-build', 'X0-status-rescan-reset', 'PL1-prefill-ladder', 'AD-A1-production-timed', 'AD-B1-adaptive-timed', 'AD-A2-production-timed',
               'AD-B2-adaptive-timed', 'PP1-prefill-profile', 'Z-reset']


def text(name):
    return (PACK / (name + '.env')).read_text(encoding='utf-8')


def parsed(name):
    return job.read_job(job.parse_env(text(name)), sorted(PROFILES), root=ROOT)


def values(name):
    return job.parse_env(text(name))


def order():
    return [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


def resolve(token, names):
    """The window driver's resolve(): the exact name, else the first job whose name starts with the token and a dash."""
    if token in names:
        return token
    return next((name for name in names if name.startswith(token + '-')), None)


def needs():
    rows = []
    for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines():
        match = re.match(r'# NEEDS ([A-Za-z0-9 -]+) <- ([A-Za-z0-9 -]+)$', line)
        if match:
            rows.append((match.group(1).split(), match.group(2).split()))
    return rows


class PackTests(unittest.TestCase):
    def test_every_template_has_an_order_row_in_order_one_of_two_tags_and_parses_on_the_quad(self):
        self.assertEqual(sorted(path.name[:-4] for path in PACK.glob('*.env')), sorted(ORDER_NAMES))
        self.assertEqual([row[0] for row in order()], ORDER_NAMES)
        for row in order():
            with self.subTest(row=row[0]):
                self.assertEqual(len(row), 4)
                self.assertIn(row[1], ('stop', 'soft'))
                self.assertTrue(row[3].isdigit())
                outputs = parsed(row[0])
                self.assertEqual((outputs['cards'], outputs['tag']), ('quad', row[2]))
                self.assertIn(row[2], (BASE_TAG, GOVERNOR_TAG))

    def test_the_governor_image_is_the_ab_arms_and_its_build_the_others_run_on_the_integrated_image(self):
        for row in order():
            expected = GOVERNOR_TAG if row[0].startswith(('AD-', 'B0')) else BASE_TAG
            self.assertEqual(row[2], expected, row[0])
        build = parsed('B0-build')
        self.assertEqual(build['actions'], 'build')
        self.assertEqual(build['bake_default_profile'], PRODUCTION)
        self.assertFalse(PROFILES[PRODUCTION].get('gate_only'))

    def test_only_the_build_and_the_status_job_stop_the_window_and_the_build_comes_first(self):
        self.assertEqual([row[0] for row in order() if row[1] == 'stop'], ['B0-build', 'X0-status-rescan-reset'])
        self.assertEqual([row[0] for row in order()][:2], ['B0-build', 'X0-status-rescan-reset'])

    def test_the_dependency_lines_resolve_as_the_driver_resolves_them_and_only_to_earlier_jobs(self):
        names = [row[0] for row in order()]
        graph = {}
        for left, right in needs():
            for token in left + right:
                self.assertIsNotNone(resolve(token, names), token)
            for token in left:
                for need in right:
                    self.assertLess(names.index(resolve(need, names)), names.index(resolve(token, names)), '%s needs %s, which must come first' % (token, need))
                graph.setdefault(resolve(token, names), set()).update(resolve(need, names) for need in right)
        self.assertEqual(graph['X0-status-rescan-reset'], {'B0-build'})
        self.assertEqual(graph['AD-B1-adaptive-timed'], {'X0-status-rescan-reset', 'B0-build', 'AD-A1-production-timed'})
        self.assertEqual(graph['AD-A2-production-timed'], {'X0-status-rescan-reset', 'B0-build', 'AD-B1-adaptive-timed'})
        self.assertEqual(graph['AD-B2-adaptive-timed'], {'X0-status-rescan-reset', 'B0-build', 'AD-A2-production-timed'})
        self.assertEqual(graph['PL1-prefill-ladder'], {'X0-status-rescan-reset'})
        self.assertEqual(graph['PP1-prefill-profile'], {'X0-status-rescan-reset'})

    def test_the_minutes_add_up_as_the_order_says(self):
        rows = {row[0]: int(row[3]) for row in order()}
        cards = sum(minutes for name, minutes in rows.items() if name != 'B0-build')
        self.assertEqual((cards, cards + rows['B0-build']), (340, 380))
        header = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        self.assertIn('340 minutes (5 h 40 min) of cards', header)
        self.assertIn('380 with the build', header)

    def test_the_profiled_job_runs_after_every_timed_one_and_before_the_final_reset(self):
        names = [row[0] for row in order()]
        for name in names:
            if name.startswith(('PL1', 'AD-')):
                self.assertLess(names.index(name), names.index('PP1-prefill-profile'), name)
        self.assertEqual(names[-2:], ['PP1-prefill-profile', 'Z-reset'])


class LadderTests(unittest.TestCase):
    def test_pl1_runs_the_solo_and_the_busy_ladder_on_the_production_profile(self):
        outputs = parsed('PL1-prefill-ladder')
        self.assertEqual((outputs['profile'], outputs['actions']), (PRODUCTION, 'reset smoke'))
        self.assertEqual(outputs['tests'].split(','), ['warmup', 'prefill_ladder_solo', 'prefill_ladder_busy'])
        self.assertIn('C2_SMOKE_PARTIAL', values('PL1-prefill-ladder'))
        smoke = (HERE / 'c2_serving_smoke.py').read_text(encoding='utf-8')
        for name in outputs['tests'].split(',')[1:]:
            self.assertIn('def %s():' % name, smoke)
            self.assertIn("'%s'" % name, smoke)
        body = text('PL1-prefill-ladder')
        for phrase in ('ONE request at a time', 'SEVEN decoders', 'prefill_ladder_report.py', 'ZERO', 'Recorded, never gated'):
            self.assertIn(phrase, body)

    def test_the_ladder_sizes_are_the_four_the_brief_asks_for(self):
        smoke = (HERE / 'c2_serving_smoke.py').read_text(encoding='utf-8')
        self.assertIn('LADDER_TOKENS = (4096, 32768, 131072, 253920)', smoke)


class GovernorTests(unittest.TestCase):
    ARMS = [('AD-A1-production-timed', PRODUCTION), ('AD-B1-adaptive-timed', TWIN), ('AD-A2-production-timed', PRODUCTION), ('AD-B2-adaptive-timed', TWIN)]

    def test_the_four_arms_alternate_the_production_profile_and_its_twin_on_one_image(self):
        self.assertEqual([name for name in ORDER_NAMES if name.startswith('AD-')], [name for name, _ in self.ARMS])
        for name, profile in self.ARMS:
            with self.subTest(name=name):
                outputs = parsed(name)
                self.assertEqual((outputs['profile'], outputs['tag'], outputs['actions']), (profile, GOVERNOR_TAG, 'reset smoke'))
        self.assertTrue(PROFILES[TWIN]['gate_only'])
        self.assertFalse(PROFILES[PRODUCTION].get('gate_only'))

    def test_the_arms_run_the_same_tests_and_the_same_box(self):
        reference = values('AD-A1-production-timed')
        for name, _ in self.ARMS:
            found = values(name)
            for key in ('C2_ACTIONS', 'C2_IMAGE_TAG', 'C2_SMOKE_TESTS', 'C2_SMOKE_PARTIAL', 'C2_BOX_MINUTES', 'C2_CARDS'):
                self.assertEqual(found.get(key), reference.get(key), (name, key))
        self.assertEqual(reference['C2_SMOKE_TESTS'].split(','), ['warmup', 'concurrent8_skew', 'concurrent8_code_32k', 'prefill_few_decoders'])
        smoke = (HERE / 'c2_serving_smoke.py').read_text(encoding='utf-8')
        for name in ('concurrent8_skew', 'concurrent8_code_32k', 'prefill_few_decoders'):
            self.assertIn('def %s():' % name, smoke)

    def test_the_headers_state_the_read_and_no_go_rules(self):
        for name, profile in self.ARMS:
            body = text(name)
            for phrase in ('ABAB', 'NO-GO', 'levern_adaptive_problems', 'prefill_ladder_report.py', 'w2ln_timing_compare.py', 'ESTIMATED', 'ZERO', profile, GOVERNOR_TAG):
                self.assertIn(phrase, body, (name, phrase))

    def test_the_twin_is_the_generated_one_and_names_the_four_flags_only_on_b(self):
        for name, profile in self.ARMS:
            env = PROFILES[profile]['env']
            self.assertEqual(bool([flag for flag in adaptive.ENV if flag in env]), profile == TWIN, name)


class ProfileJobTests(unittest.TestCase):
    def test_pp1_names_the_prefill_pair_on_the_production_profile_and_fits_its_box(self):
        outputs = parsed('PP1-prefill-profile')
        self.assertEqual((outputs['profile'], outputs['actions'], outputs['gate_plan'], outputs['tag']), (PRODUCTION, 'status reset gate', 'ops-prefill-twin,ops-prefill-trace', BASE_TAG))
        worst = sum(ops.ARM_SECONDS[plan] for plan in outputs['gate_plan'].split(',')) / 60.0
        self.assertLessEqual(worst + 15, int(outputs['box_minutes']), 'the gate refuses a plan list whose worst case does not fit the box')
        self.assertEqual(ops.OP_SUPPORT, 20000)

    def test_pp1_says_what_the_ops_rules_say(self):
        body = text('PP1-prefill-profile')
        for phrase in ('NEVER CANCEL', 'root', 'op-support 20000', 'QWEN_PREFILL_PROFILE_FLUSH=1', '131,072', '64 prefill chunks', 'prefill-profile-report.md', 'DATA, not a gate', 'NOT TIMED'):
            self.assertIn(phrase, body)

    def test_pp1_plans_exist_in_the_gate_and_run_on_the_production_profile(self):
        import c2_serving_gate as gate

        document = {'profiles': PROFILES}
        for plan in ('ops-prefill-twin', 'ops-prefill-trace'):
            arms = gate.plan_arms(plan, PRODUCTION, document, None, 4096, None, [], None, None)
            self.assertEqual(len(arms), 1)
            self.assertEqual(arms[0].extra['ops']['users'], 1)


class HygieneTests(unittest.TestCase):
    FORBIDDEN = (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), re.compile(r'/home/'), re.compile(r'zot\.'), re.compile(r'[0-9a-f]{64}'), re.compile(r'\.local\b'),
                 re.compile(r'newspark|spark-[0-9a-f]{4}|thatch-control|tt-windows', re.I), re.compile(r'(?i)(password|token)\s*='))

    def test_nothing_in_the_pack_or_its_docs_names_a_host_an_address_a_registry_a_digest_or_a_home_path(self):
        paths = sorted(PACK.iterdir()) + [ROOT / 'docs' / 'lever-n-adaptive-governor.md', ROOT / 'docs' / 'tp4-profile.md', HERE / 'make_adaptive_profiles.py',
                                          HERE / 'prefill_ladder_report.py', HERE / 'tp4_prefill_profile_report.py']
        for path in paths:
            body = path.read_text(encoding='utf-8')
            for pattern in self.FORBIDDEN:
                self.assertIsNone(pattern.search(body), '%s names %s' % (path.name, pattern.pattern))


if __name__ == '__main__':
    unittest.main()
