"""The region-read qualification pack (references/tp4-kvread-next-jobs, docs/prefix-audit-cost.md) on the CPU: every template parses with the job parser, the ORDER rows match the files, the dependency lines name
real jobs, each audited arm runs the profile that carries the knob it qualifies, the A0R/A1R pair is the engine-reuse A0/A1 pair on region twins with the same tests, and no template names a host, a path or an address.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_kvread_next_jobs` from scripts/ci."""

import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import make_kvread_profiles as generator  # noqa: E402

PACK = HERE / 'references' / 'tp4-kvread-next-jobs'
ENGINE_REUSE = HERE / 'references' / 'tp4-engine-reuse-jobs'
WINDOW = HERE / 'references' / 'tp4-window-next'
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']


def parsed(name):
    return job.read_job(job.parse_env((PACK / (name + '.env')).read_text(encoding='utf-8')), sorted(PROFILES), root=ROOT)


def order():
    return [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


def needs():
    rows = []
    for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines():
        match = re.match(r'# NEEDS ([A-Za-z0-9 ]+) <- ([A-Za-z0-9 ]+)$', line)
        if match:
            rows.append((match.group(1).split(), match.group(2).split()))
    return rows


class PackTests(unittest.TestCase):
    def test_every_template_has_an_order_row_and_parses_on_the_quad(self):
        names = sorted(path.name[:-4] for path in PACK.glob('*.env'))
        self.assertEqual(names, sorted(row[0] for row in order()))
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(parsed(name)['cards'], 'quad')

    def test_the_rows_have_a_class_one_tag_and_a_whole_number_of_minutes(self):
        window_tag = re.search(r'C2_IMAGE_TAG=(\S+)', (WINDOW / 'H1-roundhost-audit-attach-lean.env').read_text(encoding='utf-8')).group(1)
        for row in order():
            with self.subTest(row=row[0]):
                self.assertEqual(len(row), 4)
                self.assertIn(row[1], ('stop', 'soft'))
                self.assertEqual(row[2], window_tag, "the window pack's placeholder tag")
                self.assertTrue(row[3].isdigit())
                self.assertEqual(parsed(row[0])['tag'], row[2])

    def test_the_dependency_lines_name_real_jobs_and_the_audited_jobs_wait_for_the_qualification(self):
        names = {row[0] for row in order()}
        short = {name.split('-')[0]: name for name in names}
        for left, right in needs():
            for key in left + right:
                self.assertIn(key, short, key)
        graph = {job_: set(right) for left, right in needs() for job_ in left}
        self.assertEqual(graph['Q2'], {'Q1'})
        self.assertEqual(graph['Q3'], {'Q1'})
        self.assertEqual(graph['A0R'], {'Q2'})
        self.assertEqual(graph['A1R'], {'A0R'})

    def test_q1_is_the_kvread_fabric_probe_and_the_stop_class(self):
        outputs = parsed('Q1-kvread-qualify')
        self.assertEqual((outputs['actions'], outputs['fabric_probe'], outputs['box_minutes']), ('reset fabric', 'kvread', '30'))
        self.assertEqual({row[0]: row[1] for row in order()}['Q1-kvread-qualify'], 'stop')

    def test_q2_runs_the_cross_twin_with_the_exact_length_prompts(self):
        outputs = parsed('Q2-kvread-cross-levern-audit')
        self.assertEqual(outputs['profile'], 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-r2-kvx')
        self.assertEqual(PROFILES[outputs['profile']]['env']['QWEN_FAST_LEVERN_KV_READ'], 'cross')
        self.assertEqual(outputs['tests'].split(',')[:2], ['warmup', 'levern_equal'], 'the digests of the exact-length prompts, after the warmup')

    def test_q3_is_the_prefix_gates_read_qualify_arm_on_an_audited_prefix_profile(self):
        outputs = parsed('Q3-kvread-cross-prefix-audit')
        self.assertEqual((outputs['actions'], outputs['prefix_plan'], outputs['prefix_baseline']), ('reset prefix', 'read-qualify', 'none'))
        self.assertEqual(gate.PLAN_ARMS['read-qualify'][0][3], 'auditcross')
        self.assertEqual(gate.DERIVED['auditcross']['env']['QWEN_PREFIX_AUDIT_READ'], 'cross')
        self.assertIn(outputs['prefix_profile'], PROFILES)
        self.assertEqual(PROFILES[outputs['prefix_profile']]['env']['QWEN_PREFIX_REUSE'], '1')
        self.assertGreaterEqual(int(outputs['box_minutes']) * 60, gate.PLAN_ARMS['read-qualify'][0][4], "the box covers the arm's own docker timeout")

    def test_the_pair_is_the_engine_reuse_pair_on_region_twins_with_the_same_tests(self):
        def field(path, key):
            return re.search(r'^%s=(.*)$' % key, path.read_text(encoding='utf-8'), re.M).group(1)

        for new, old, twin_of in (('A0R-control-audit-region', 'A0-control-audit-attach', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-r2'),
                                  ('A1R-parked-audit-region', 'A1-parked-audit-attach', 'c2-packed-tp4-8x262k-ship-prefix-levern-parked-audit')):
            with self.subTest(new=new):
                outputs = parsed(new)
                self.assertEqual(outputs['tests'], field(ENGINE_REUSE / (old + '.env'), 'C2_SMOKE_TESTS'), 'same tests as the job it replaces')
                self.assertEqual(outputs['actions'], parsed_old(old)['actions'])
                twin = PROFILES[outputs['profile']]
                self.assertEqual(twin['env']['QWEN_FAST_LEVERN_KV_READ'], 'region')
                parent = dict(PROFILES[twin_of]['env'])
                self.assertEqual({key: value for key, value in twin['env'].items() if key != 'QWEN_FAST_LEVERN_KV_READ'}, parent, 'the old job\'s profile plus the knob')
                self.assertIn(outputs['profile'], generator.twin_names())

    def test_every_audited_template_names_a_twin_of_the_generator(self):
        for name in ('Q2-kvread-cross-levern-audit', 'A0R-control-audit-region', 'A1R-parked-audit-region'):
            self.assertIn(parsed(name)['profile'], generator.twin_names(), name)

    def test_the_order_says_why_and_what_the_default_is(self):
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        for phrase in ('QWEN_FAST_LEVERN_KV_READ', 'full (unset: today\'s read)', 'levern_kv_read_problems', 'NO-GO', 'NOT QUALIFIED', '8.4 minutes', 'make_kvread_profiles.py',
                       'NEEDS A1R <- A0R'):
            self.assertIn(phrase, text)

    def test_no_template_names_a_host_an_address_or_a_home_path(self):
        pattern = re.compile(r'(/home/|/Users/|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.local\b|zot\.|sha256:[0-9a-f]{16}|ghp_|token=)')
        for path in sorted(PACK.iterdir()) + [HERE / 'make_kvread_profiles.py']:
            with self.subTest(path=path.name):
                self.assertIsNone(pattern.search(path.read_text(encoding='utf-8')))


def parsed_old(name):
    return job.read_job(job.parse_env((ENGINE_REUSE / (name + '.env')).read_text(encoding='utf-8')), sorted(PROFILES), root=ROOT)


if __name__ == '__main__':
    unittest.main()
