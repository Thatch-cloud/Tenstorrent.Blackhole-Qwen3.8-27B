"""The separate-links pack (references/tp4-subdev-links-jobs) on the CPU: every template parses with the job parser (and through its command line, rc 0), the ORDER rows match
the files, the one image tag and the minutes, the dependency lines name real jobs in order, the two build jobs are cardm runs of build_k64j_lo.sh (the dry run first, the real one
second, and nothing else in the pack builds), the decisive H2d is the subdev-links probe whose header registers the port rule with the thresholds the code applies, and no
template names a host, an address, a registry or a home path.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_subdev_links_jobs` from scripts/ci."""

import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'optimisation' / 'ttnn-op' / 'subdev_h'))

import c2_serving_job as job  # noqa: E402
import subdev_plan as plan  # noqa: E402

PACK = HERE / 'references' / 'tp4-subdev-links-jobs'
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
TAG = 'tp4-next2-1'
BUILD = 'optimisation/ttnn-op/subdev_h/build_k64j_lo.sh'
ORDER = ('L0-build-k64j-lo-dry', 'L1-build-k64j-lo', 'X0-status-rescan-reset', 'H2d-subdev-quad-links', 'Z-reset')


def text(name):
    return (PACK / (name + '.env')).read_text(encoding='utf-8')


def parsed(name):
    return job.read_job(job.parse_env(text(name)), sorted(PROFILES), root=ROOT)


def order():
    return [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


class PackTests(unittest.TestCase):
    def test_the_order_rows_are_the_templates_in_the_designed_order_with_one_tag(self):
        names = [row[0] for row in order()]
        self.assertEqual(names, list(ORDER))
        self.assertEqual(sorted(names), sorted(path.name[:-4] for path in PACK.glob('*.env')))
        for row in order():
            self.assertEqual((len(row), row[2], row[3].isdigit(), row[1] in ('stop', 'soft')), (4, TAG, True, True), row)
        self.assertEqual({row[0]: row[1] for row in order()}['L1-build-k64j-lo'], 'stop', 'the next jobs need the graft')

    def test_every_template_parses_and_its_command_line_exits_zero(self):
        for name in ORDER:
            with self.subTest(name=name):
                self.assertEqual(parsed(name)['tag'], TAG)
                result = subprocess.run([sys.executable, '-s', str(HERE / 'c2_serving_job.py'), str(PACK / (name + '.env'))], capture_output=True, text=True,
                                        cwd=str(ROOT))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_the_build_jobs_are_cardm_runs_of_the_graft_build_dry_first(self):
        dry, real = parsed(ORDER[0]), parsed(ORDER[1])
        for outputs in (dry, real):
            self.assertEqual((outputs['cards'], outputs['actions'], outputs['cardm_harness'], outputs['cardm_args']), ('pair', 'cardm', BUILD, ''))
        self.assertEqual((dry['cardm_env'], real['cardm_env']), ('LO_BUILD_DRY_RUN=1', ''))
        for name in ORDER[2:]:
            self.assertNotEqual(parsed(name)['actions'], 'cardm')

    def test_h2d_is_the_separate_links_probe_with_the_box_inside_the_fabric_step(self):
        outputs = parsed('H2d-subdev-quad-links')
        self.assertEqual((outputs['cards'], outputs['actions'], outputs['fabric_probe'], outputs['fabric']), ('quad', 'reset fabric', 'subdev-links', 'FABRIC_1D'))
        self.assertLessEqual(int(outputs['box_minutes']), job.STEP_MINUTES['fabric'])

    def test_the_rule_in_h2ds_header_is_the_one_the_code_applies(self):
        header = text('H2d-subdev-quad-links')
        self.assertIn('link_cost = t_solo / t_solo2 <= %.2f' % plan.LINK_COST_GO, header)
        self.assertIn('%.2f < link_cost <= %.2f' % (plan.LINK_COST_GO, plan.LINK_COST_MAX), header)
        self.assertIn('link_cost > %.2f' % plan.LINK_COST_MAX, header)
        self.assertIn('0.042', header)
        self.assertIn('0.113', header)
        for needle in ('PASS-SEPARATE-LINKS', 'port=', 'NaN', 'a graft mounted is not a graft executed', 'QWEN_AG_LINK_OFFSET_SD1'):
            self.assertIn(needle, header)

    def test_the_dependency_lines_name_real_jobs_in_order(self):
        names = [row[0] for row in order()]
        short = {re.match(r'[A-Z]\d*d?', name).group(0): name for name in names}
        rows = []
        for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines():
            match = re.match(r'# NEEDS ([A-Za-z0-9 -]+) <- ([A-Za-z0-9 -]+)$', line)
            if match:
                rows.append((match.group(1).split(), match.group(2).split()))
        graph = {key: set(right) for left, right in rows for key in left}
        self.assertEqual(graph, {'L1': {'L0'}, 'X0': {'L1'}, 'H2d': {'X0'}})
        for left, right in rows:
            for key in left + right:
                self.assertIn(key, short, key)
            for key in left:
                for need in right:
                    self.assertLess(names.index(short[need]), names.index(short[key]))

    def test_the_stated_minutes_add_up(self):
        rows = order()
        body = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        running, expected = 0, []
        for row in rows:
            running += int(row[3])
            expected.append(running)
        self.assertEqual([int(n) for n in re.findall(r'\[(\d+)\]', body)], expected)
        self.assertIn('%d minutes' % running, body)

    def test_no_template_names_a_host_an_address_a_registry_a_digest_or_a_home_path(self):
        pattern = re.compile(r'(/home/|/Users/|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.local\b|zot\.|sha256:[0-9a-f]{16}|ghp_|token=|blackhole-[A-Za-z0-9]{8,}|[0-9a-f]{40,})')
        for path in sorted(PACK.iterdir()):
            with self.subTest(path=path.name):
                body = path.read_text(encoding='utf-8')
                self.assertIsNone(pattern.search(body))
                self.assertNotIn('C2_SUPERSEDED_BY', body)


class RegistrationTests(unittest.TestCase):
    def test_the_parser_and_the_workflow_mount_and_check_the_graft(self):
        self.assertIn('subdev-links', job.FABRIC_PROBES)
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        self.assertIn('subdev-links) script=tp4_subdev_probe.py; report=subdev-probe.json; probe_args=(--arms solo,link_cost,separate,chained,one_queue --link-offset 1) ;;', workflow)
        for needle in ('lomount=()', '"$HOME/opgraft-K64j-LO"', 'sha256sum -c --quiet MANIFEST.sha256', '/opt/tt-metal/build_Release/ttnn/_ttnncpp.so:ro',
                       '/opt/tt-metal/build_Release/lib/_ttnncpp.so:ro', 'QWEN_AG_LINK_GRAFT_SHA256=$lo_sha', '${lomount[@]+"${lomount[@]}"}'):
            self.assertIn(needle, workflow)
        self.assertNotIn('opgraft-K64j-LO:', workflow, 'only the binary is mounted, never the whole graft directory')
        # the graft build never writes the base: nothing in the workflow's mount block names ~/opgraft-K64j
        block = workflow[workflow.index('lomount=()'):workflow.index('timeout -k 30 "$limit"')]
        self.assertNotRegex(block, r'opgraft-K64j(?!-LO)')

    def test_the_probe_arms_the_workflow_passes_are_known_to_the_harness(self):
        import subdev_h2
        # the decisive separate arm right after the arms that cannot hang, before chained and one_queue (the first run of this job hung in one_queue, which
        # was a shared-link arm, and never reached separate); no arm of the job puts two streams on one link
        order = ['solo', 'link_cost', 'separate', 'chained', 'one_queue']
        self.assertEqual(subdev_h2.parse_groups('solo,link_cost,chained,one_queue,separate'), order)
        self.assertEqual(subdev_h2.parse_groups('solo,link_cost,separate,chained,one_queue'), order)
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        case = [line for line in workflow.splitlines() if line.strip().startswith('subdev-links) script=')][0]
        arms = case.split('--arms ')[1].split(' ')[0].split(',')
        self.assertEqual(arms, order)
        self.assertFalse({'shared', 'shared2'} & set(arms))
        header = (PACK / 'H2d-subdev-quad-links.env').read_text(encoding='utf-8')
        self.assertLess(header.index('THE DECISIVE ARM, separate'), header.index('then chained'))
        self.assertIn('before_hang', header)


if __name__ == '__main__':
    unittest.main()
