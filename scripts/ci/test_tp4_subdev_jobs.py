"""The sub-device overlap pack (references/tp4-subdev-jobs) on the CPU: every template parses with the job parser (and through its command line, rc 0), the ORDER rows match the files, the one image tag and
the minutes, the dependency lines name real jobs in order, the two H1 jobs are the card-M harness (watcher pass first, timed pass second), the two H2 jobs are the four-card probe (watcher, then timed) and RUN
LAST, before the all-board reset, and no template names a host, an address, a registry or a home path.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_subdev_jobs` from scripts/ci."""

import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402

PACK = HERE / 'references' / 'tp4-subdev-jobs'
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
TAG = 'tp4-next2-1'
HARNESS = 'optimisation/ttnn-op/subdev_h/run_card_m.sh'
H1 = ('H1a-subdev-watcher', 'H1b-subdev-timed')
H2 = ('H2a-subdev-quad-watcher', 'H2b-subdev-quad-timed')
ORDER = H1 + ('X0-status-rescan-reset',) + H2 + ('Z-reset',)


def text(name):
    return (PACK / (name + '.env')).read_text(encoding='utf-8')


def parsed(name):
    return job.read_job(job.parse_env(text(name)), sorted(PROFILES), root=ROOT)


def order():
    return [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


def needs():
    rows = []
    for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines():
        match = re.match(r'# NEEDS ([A-Za-z0-9 -]+) <- ([A-Za-z0-9 -]+)$', line)
        if match:
            rows.append((match.group(1).split(), match.group(2).split()))
    return rows


class PackTests(unittest.TestCase):
    def test_the_order_rows_are_the_templates_in_the_designed_order_with_one_tag(self):
        names = [row[0] for row in order()]
        self.assertEqual(names, list(ORDER))
        self.assertEqual(sorted(names), sorted(path.name[:-4] for path in PACK.glob('*.env')))
        for row in order():
            with self.subTest(row=row[0]):
                self.assertEqual(len(row), 4)
                self.assertIn(row[1], ('stop', 'soft'))
                self.assertEqual(row[2], TAG)
                self.assertTrue(row[3].isdigit())
        self.assertEqual({row[0]: row[1] for row in order()}['X0-status-rescan-reset'], 'stop')

    def test_every_template_parses_and_its_command_line_exits_zero(self):
        for name in ORDER:
            with self.subTest(name=name):
                outputs = parsed(name)
                self.assertEqual(outputs['tag'], TAG)
                result = subprocess.run([sys.executable, '-s', str(HERE / 'c2_serving_job.py'), str(PACK / (name + '.env'))], capture_output=True,
                                        text=True, cwd=str(ROOT))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_the_h1_jobs_are_the_card_m_harness_watcher_pass_first_then_timed(self):
        first, second = parsed(H1[0]), parsed(H1[1])
        for outputs in (first, second):
            self.assertEqual((outputs['cards'], outputs['actions'], outputs['cardm_harness']), ('pair', 'cardm', HARNESS))
            self.assertEqual(outputs['box_minutes'], '')
        self.assertEqual(first['cardm_env'].split(), ['IMAGE_TAG=' + TAG, 'WATCHER=1'])
        self.assertEqual(second['cardm_env'].split(), ['IMAGE_TAG=' + TAG])
        self.assertEqual((first['cardm_args'], second['cardm_args']), ('', ''))

    def test_the_h2_jobs_are_the_quad_probe_watcher_pass_first_then_timed_with_a_box_inside_the_fabric_step(self):
        first, second = parsed(H2[0]), parsed(H2[1])
        for outputs in (first, second):
            self.assertEqual((outputs['cards'], outputs['actions'], outputs['fabric']), ('quad', 'reset fabric', 'FABRIC_1D'))
            self.assertLessEqual(int(outputs['box_minutes']), job.STEP_MINUTES['fabric'])
        self.assertEqual((first['fabric_probe'], second['fabric_probe']), ('subdev-watch', 'subdev'))

    def test_h2_runs_last_and_the_all_board_reset_ends_the_pack(self):
        names = [row[0] for row in order()]
        for quad in H2:
            for earlier in H1:
                self.assertLess(names.index(earlier), names.index(quad))
        self.assertEqual(names[-1], 'Z-reset')
        self.assertEqual(names[-3:-1], list(H2))
        self.assertEqual(parsed('Z-reset')['actions'], 'status reset')
        self.assertEqual(parsed('X0-status-rescan-reset')['actions'], 'status rescan reset')
        self.assertEqual(names.index('X0-status-rescan-reset'), len(H1))

    def test_the_dependency_lines_name_real_jobs_in_order(self):
        names = [row[0] for row in order()]
        short = {re.match(r'[A-Z]\d*[ab]?', name).group(0): name for name in names}
        for left, right in needs():
            for key in left + right:
                self.assertIn(key, short, key)
            for key in left:
                for need in right:
                    self.assertLess(names.index(short[need]), names.index(short[key]), '%s needs %s, which must come first' % (key, need))
        graph = {key: set(right) for left, right in needs() for key in left}
        self.assertEqual(graph['H1b'], {'H1a'})
        self.assertEqual(graph['H2a'], {'X0', 'H1b'})
        self.assertEqual(graph['H2b'], {'H2a'})

    def test_the_stated_minutes_add_up(self):
        rows = order()
        total = sum(int(row[3]) for row in rows)
        body = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        cumulative = [int(n) for n in re.findall(r'\[(\d+)\]', body)]
        running, expected = 0, []
        for row in rows:
            running += int(row[3])
            expected.append(running)
        self.assertEqual(cumulative, expected)
        self.assertIn('%d minutes' % total, body)

    def test_each_template_documents_what_to_read(self):
        for name in H1 + H2:
            with self.subTest(name=name):
                self.assertIn('READ:', text(name))
                self.assertIn('SUBDEV_H', text(name))
        self.assertIn('UNTIMED-PASS', text(H1[0]))
        self.assertIn('PASS = concurrent wall <= 1.1 x max(solo walls)', text(H1[1]))
        self.assertIn('ag_link_offset.patch', text(H2[1]))
        self.assertIn('ALL-BOARD RESET', text(H2[0]))

    def test_the_templates_are_unsuperseded_and_name_no_host_address_registry_digest_or_home(self):
        pattern = re.compile(r'(/home/|/Users/|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.local\b|zot\.|sha256:[0-9a-f]{16}|ghp_|token=|blackhole-[A-Za-z0-9]{8,})')
        for path in sorted(PACK.iterdir()):
            with self.subTest(path=path.name):
                body = path.read_text(encoding='utf-8')
                self.assertIsNone(pattern.search(body))
                self.assertNotIn('C2_SUPERSEDED_BY', body)


class RegistrationTests(unittest.TestCase):
    def test_the_job_parser_and_the_workflow_know_the_subdev_probes(self):
        self.assertIn('subdev', job.FABRIC_PROBES)
        self.assertIn('subdev-watch', job.FABRIC_PROBES)
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        self.assertIn('subdev) script=tp4_subdev_probe.py; report=subdev-probe.json ;;', workflow)
        self.assertIn('subdev-watch) script=tp4_subdev_probe.py; report=subdev-probe.json; probe_args=(--watcher) ;;', workflow)
        self.assertIn('${probe_args[@]+"${probe_args[@]}"}', workflow)
        self.assertIn('probe_args=()', workflow)
        self.assertIn('SUBDEV_H2', workflow)
        self.assertTrue((HERE / 'tp4_subdev_probe.py').is_file())

    def test_a_probe_name_the_parser_does_not_know_is_refused(self):
        values = job.parse_env(text(H2[1]).replace('C2_FABRIC_PROBE=subdev', 'C2_FABRIC_PROBE=subdev-links'))
        with self.assertRaises(job.JobError):
            job.read_job(values, sorted(PROFILES), root=ROOT)

    def test_the_probe_needs_the_fabric_action_and_the_quad(self):
        values = job.parse_env(text(H2[1]).replace('C2_CARDS=quad', 'C2_CARDS=pair'))
        with self.assertRaises(job.JobError):
            job.read_job(values, sorted(PROFILES), root=ROOT)
        values = job.parse_env(text(H2[1]).replace('C2_ACTIONS=reset fabric', 'C2_ACTIONS=reset'))
        with self.assertRaises(job.JobError):
            job.read_job(values, sorted(PROFILES), root=ROOT)


if __name__ == '__main__':
    unittest.main()
