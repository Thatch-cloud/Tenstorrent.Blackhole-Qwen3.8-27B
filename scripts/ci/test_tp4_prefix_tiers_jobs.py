"""The tp4/prefix-tiers job pack (references/tp4-prefix-tiers-jobs, docs/prefix-store-hygiene.md) on the CPU: every template parses with the job parser (rc 0, the way the workflow reads
it), the ORDER rows match the files, the dependency lines name real jobs in the order the cheap kill signals come first, each prefix job runs the plan and the profile it is
about (the tier twin, no baseline, a box that covers the arm), the build job bakes nothing, no template names a host, a path or an address, and pin_kvread (the one command that moves the
extension's pin before B0) rewrites every file that carries it and refuses a build that is not this checkout's version 2.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_prefix_tiers_jobs` from scripts/ci."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import make_prefix_tier_profiles as generator  # noqa: E402
import pin_kvread  # noqa: E402

PACK = HERE / 'references' / 'tp4-prefix-tiers-jobs'
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
TAG = 'tp4-prefix-tiers-1'
TIER = generator.TWIN


def parsed(name):
    return job.read_job(job.parse_env((PACK / (name + '.env')).read_text(encoding='utf-8')), job.profile_names(job.PROFILES), envs=job.profile_envs(job.PROFILES))


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
    def test_every_template_has_an_order_row_and_the_job_parser_exits_zero_on_it(self):
        names = sorted(path.name[:-4] for path in PACK.glob('*.env'))
        self.assertEqual(names, sorted(row[0] for row in order()))
        for name in names:
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, '-s', '-B', str(HERE / 'c2_serving_job.py'), str(PACK / (name + '.env'))], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, cwd=str(ROOT), timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', 'replace'))
                self.assertEqual(parsed(name)['cards'], 'quad')

    def test_the_rows_have_the_one_tag_a_class_and_a_whole_number_of_minutes(self):
        for row in order():
            with self.subTest(row=row[0]):
                self.assertEqual(len(row), 4)
                self.assertIn(row[1], ('stop', 'soft'))
                self.assertEqual(row[2], TAG)
                self.assertTrue(row[3].isdigit())
                self.assertEqual(parsed(row[0])['tag'], TAG)

    def test_the_cheap_kill_signals_come_first_and_the_dependencies_chain_the_pack(self):
        names = [row[0] for row in order()]
        self.assertEqual([name.split('-')[0] for name in names], ['B0', 'X0', 'Q1', 'A1', 'T1', 'T2', 'T3', 'Z'])
        short = {name.split('-')[0]: name for name in names}
        for left, right in needs():
            for key in left + right:
                self.assertIn(key, short, key)
        graph = {job_: set(right) for left, right in needs() for job_ in left}
        self.assertEqual(graph, {'X0': {'B0'}, 'Q1': {'X0'}, 'A1': {'Q1'}, 'T1': {'A1'}, 'T2': {'T1'}, 'T3': {'T2'}})
        classes = {row[0].split('-')[0]: row[1] for row in order()}
        for key in ('B0', 'X0', 'Q1', 'A1', 'T1', 'T2'):
            self.assertEqual(classes[key], 'stop', key)
        self.assertEqual((classes['T3'], classes['Z']), ('soft', 'soft'), 'the timed arm and the reset do not stop the window')

    def test_b0_builds_a_fresh_tag_and_bakes_no_default(self):
        outputs = parsed('B0-build')
        self.assertEqual((outputs['actions'], outputs['bake_default_profile'], outputs['profile']), ('build', '', 'c2-packed-tp4'))

    def test_b0_names_no_path_and_the_build_reads_the_version_2_directory_by_default(self):
        values = [line for line in (PACK / 'B0-build.env').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]
        self.assertEqual(values, ['C2_CARDS=quad', 'C2_ACTIONS=build', 'C2_IMAGE_TAG=%s' % TAG, 'C2_PROFILE=c2-packed-tp4'])
        self.assertFalse([line for line in values if '/' in line], 'no path in the job env')
        script = (HERE / 'build-c2-serving-image.sh').read_text(encoding='utf-8')
        self.assertIn('kvread=${C2_KVREAD_DIR:-/home/thatch/opgraft-KVR2}', script)
        self.assertIn('kvread_name=opgraft-KVR', script)
        sha = pin_kvread.current_pin()
        for name in pin_kvread.PIN_SITES:
            self.assertIn(sha, (ROOT / name).read_text(encoding='utf-8'), name)

    def test_x0_and_z_never_start_stop_or_hand_back_the_agent(self):
        for name in ('X0-status-rescan-reset', 'Z-reset'):
            actions = parsed(name)['actions'].split()
            self.assertEqual(set(actions) - {'status', 'rescan', 'reset'}, set(), name)
            self.assertNotIn('agentstart', actions)
        self.assertEqual(parsed('Z-reset')['actions'], 'status reset')

    def test_q1_is_the_kvread_probe_with_the_probes_own_box(self):
        outputs = parsed('Q1-kvread-v2-raw-probe')
        self.assertEqual((outputs['actions'], outputs['fabric_probe'], outputs['box_minutes']), ('reset fabric', 'kvread', '30'))
        text = (PACK / 'Q1-kvread-v2-raw-probe.env').read_text(encoding='utf-8')
        for phrase in ('KV_READ_PROBE raw=ok', 'raw=absent', 'NOT QUALIFIED', 'qwen_read_blocks_raw'):
            self.assertIn(phrase, text)

    def test_a1_smokes_the_tier_twin_on_the_production_tests(self):
        outputs = parsed('A1-tier-audited-attach-smoke')
        self.assertEqual((outputs['actions'], outputs['profile']), ('reset smoke', TIER))
        self.assertEqual(outputs['tests'].split(','), ['warmup', 'concurrent8_steady', 'concurrent8_code_equal', 'coding'])
        self.assertIs(PROFILES[outputs['profile']]['gate_only'], True)

    def test_each_prefix_job_runs_one_arm_on_the_tier_twin_with_no_baseline_and_a_box_over_the_arm(self):
        for name, plan, arm in (('T1-tier-attach', 'tier-attach', 'tier-attach'), ('T2-tier-returning', 'tier-returning', 'tier-returning'),
                                ('T3-tier-timed-restore', 'tier-timed', 'tier-timed')):
            with self.subTest(name=name):
                outputs = parsed(name)
                self.assertEqual((outputs['actions'], outputs['prefix_plan'], outputs['prefix_profile'], outputs['prefix_baseline']),
                                 ('reset prefix', plan, TIER, 'none'))
                self.assertEqual(outputs['prefix_kvread_mount'], '', 'the image bakes the extension; the mount would shadow nothing and replace model.py')
                arms = gate.plan_arms(plan, TIER, 'none', {'profiles': PROFILES})
                self.assertEqual([entry['arm'] for entry in arms], [arm])
                seconds = arms[0]['timeout'] + gate.ARM_OVERHEAD_SECONDS
                self.assertGreaterEqual(int(outputs['box_minutes']) * 60, arms[0]['timeout'], 'the box covers the arm\'s own docker timeout')
                self.assertLessEqual(int(outputs['box_minutes']) * 60, 380 * 60, "inside the prefix step's own timeout")
                row = {row[0]: row for row in order()}[name]
                self.assertGreaterEqual(int(row[3]) * 60, arms[0]['timeout'] // 2, 'the row\'s estimate is not absurdly under the arm')
                self.assertGreater(seconds, 0)

    def test_the_timed_job_derives_the_audit_free_profile_and_the_others_run_the_twin_itself(self):
        arms = {plan: gate.plan_arms(plan, TIER, 'none', {'profiles': PROFILES})[0] for plan in ('tier-attach', 'tier-returning', 'tier-timed')}
        self.assertEqual(arms['tier-attach']['served'], TIER)
        self.assertEqual(arms['tier-returning']['served'], TIER)
        self.assertEqual(arms['tier-timed']['served'], TIER + '+tiertime')
        self.assertTrue(arms['tier-attach']['tier']['audit'] and arms['tier-returning']['tier']['audit'])
        self.assertFalse(arms['tier-timed']['tier']['audit'])

    def test_the_headers_say_what_is_read_and_the_order_says_what_is_not_done_and_the_extension_is_unqualified(self):
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        for phrase in ('NEEDS A1 <- Q1', 'NO-GO', 'Q1 was the first card job to run them', 'pin_kvread.py', 'build_kv_read.sh', 'opgraft-KVR2', 'tier_image_problems', 'UNQUALIFIED',
                       'NO agentstart', 'NOT_EXERCISED', 'docs/prefix-store-hygiene.md'):
            self.assertIn(phrase, text)
        for name in ('T1-tier-attach', 'T2-tier-returning', 'T3-tier-timed-restore'):
            body = (PACK / (name + '.env')).read_text(encoding='utf-8')
            self.assertIn('READ', body, name)
            self.assertIn('Duration (estimate)', body, name)

    def test_the_flag_names_in_the_headers_are_real_names(self):
        known = set(generator.ENV) | {'QWEN_PREFIX_STORE_GIB', 'QWEN_PREFIX_REUSE'}
        for path in PACK.iterdir():
            for name in set(re.findall(r'QWEN_[A-Z0-9_]+', path.read_text(encoding='utf-8'))):
                with self.subTest(path=path.name, name=name):
                    self.assertTrue(name in known or any(name in profile['env'] for profile in PROFILES.values()), name)

    def test_no_template_names_a_host_an_address_or_a_home_path(self):
        pattern = re.compile(r'(/home/|/Users/|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.local\b|zot\.|sha256:[0-9a-f]{16}|ghp_|token=)')
        for path in sorted(PACK.iterdir()):
            with self.subTest(path=path.name):
                self.assertIsNone(pattern.search(path.read_text(encoding='utf-8')))


class PinTests(unittest.TestCase):
    """scripts/ci/pin_kvread.py on a scratch tree."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root), True)
        self.old = '1' * 64
        self.cpp = self.root / 'src.cpp'
        self.cpp.write_bytes(b'// extension source version 2\n')
        self.build = self.root / 'build'
        self.build.mkdir()
        (self.build / 'qwen_kv_read.so').write_bytes(b'\x7fELF qwen_read_blocks qwen_read_blocks_raw qwen_write_blocks_raw qwen_block_bytes')
        (self.build / 'qwen_kv_read.cpp').write_bytes(self.cpp.read_bytes())
        (self.build / 'MANIFEST.sha256').write_text('x\n')
        self.new = hashlib.sha256((self.build / 'qwen_kv_read.so').read_bytes()).hexdigest()
        self.tree = self.root / 'tree'
        (self.tree / 'docker').mkdir(parents=True)
        (self.tree / 'docker' / 'Dockerfile').write_text('RUN test "$x" = %s; \\\n' % self.old)
        (self.tree / 'build.sh').write_text('kvread_sha=%s\n' % self.old)
        (self.tree / 'prov.py').write_text("KVREAD_SHA256 = '%s'\n" % self.old)
        (self.tree / 'unrelated.txt').write_text('nothing here\n')
        (self.tree / 'history.md').write_text('the version 1 build was %s\n' % self.old)
        self.out = []

    SITES = ('docker/Dockerfile', 'build.sh', 'prov.py')

    def run_main(self, *extra, **kwargs):
        return pin_kvread.main(['--dir', str(self.build)] + list(extra), root=self.tree, out=self.out.append, pin=self.old, cpp=self.cpp,
                               sites=kwargs.pop('sites', self.SITES), **kwargs)

    def test_a_dry_run_names_the_files_and_changes_nothing(self):
        self.assertEqual(self.run_main(), 0)
        self.assertIn('3 files', self.out[0])
        self.assertIn('(dry run', self.out[-1])
        self.assertNotIn('history.md', self.out[0])
        self.assertIn(self.old, (self.tree / 'build.sh').read_text())

    def test_write_moves_the_enforcing_sites_and_leaves_the_files_that_quote_the_old_hash_as_history(self):
        self.assertEqual(self.run_main('--write'), 0, self.out)
        for name in self.SITES:
            text = (self.tree / name).read_text()
            self.assertIn(self.new, text, name)
            self.assertNotIn(self.old, text, name)
        self.assertEqual((self.tree / 'unrelated.txt').read_text(), 'nothing here\n')
        self.assertIn(self.old, (self.tree / 'history.md').read_text(), 'a file that describes the version 1 build stays true')
        self.assertTrue(any('left alone' in line and 'history.md' in line for line in self.out), self.out)
        self.assertIn('3 files', self.out[0])
        self.assertNotIn('history.md', self.out[0])
        self.out[:] = []
        self.assertEqual(self.run_main('--write'), 1, 'with the stale pin, a second run finds the sites already moved and says so instead of passing')
        self.assertTrue(any('should carry the old pin' in line for line in self.out), self.out)

    def test_a_pin_site_that_no_longer_carries_the_pin_is_refused_not_skipped(self):
        (self.tree / 'prov.py').write_text('KVREAD_SHA256 = None\n')
        self.assertEqual(self.run_main('--write'), 1)
        self.assertTrue(any('prov.py should carry the old pin' in line for line in self.out), self.out)
        self.assertIn(self.old, (self.tree / 'build.sh').read_text(), 'nothing is rewritten when a site is missing')

    def test_a_build_that_is_already_the_pin_moves_nothing(self):
        code = pin_kvread.main(['--dir', str(self.build), '--write'], root=self.tree, out=self.out.append, pin=self.new, cpp=self.cpp)
        self.assertEqual(code, 0)
        self.assertIn('already the pinned one', self.out[-1])
        self.assertIn(self.old, (self.tree / 'build.sh').read_text())

    def test_a_build_of_another_source_a_version_one_binary_and_an_unfinished_build_are_refused(self):
        (self.build / 'qwen_kv_read.cpp').write_bytes(b'// another source\n')
        self.assertEqual(self.run_main('--write'), 1)
        self.assertTrue(any("is not this checkout's" in line for line in self.out), self.out)
        self.assertIn(self.old, (self.tree / 'build.sh').read_text(), 'a refused build moves nothing')
        (self.build / 'qwen_kv_read.cpp').write_bytes(self.cpp.read_bytes())
        (self.build / 'qwen_kv_read.so').write_bytes(b'\x7fELF qwen_read_blocks qwen_block_bytes')
        self.out[:] = []
        self.assertEqual(self.run_main('--write'), 1)
        self.assertTrue(any('not version 2' in line and 'qwen_read_blocks_raw' in line for line in self.out), self.out)
        (self.build / 'qwen_kv_read.so').write_bytes(b'\x7fELF qwen_read_blocks qwen_read_blocks_raw qwen_write_blocks_raw qwen_block_bytes')
        (self.build / 'MANIFEST.sha256').unlink()
        self.out[:] = []
        self.assertEqual(self.run_main('--write'), 1)
        self.assertTrue(any('MANIFEST.sha256' in line for line in self.out), self.out)
        self.out[:] = []
        shutil.rmtree(str(self.build))
        self.assertEqual(self.run_main('--write'), 1)
        self.assertTrue(any('build_kv_read.sh' in line for line in self.out), self.out)

    def test_the_real_pin_is_one_value_in_every_file_that_carries_it_so_the_tool_finds_them_all(self):
        old = pin_kvread.current_pin()
        files = pin_kvread.tracked_files(ROOT, old)
        names = {str(path.relative_to(ROOT)) for path in files}
        for expected in pin_kvread.PIN_SITES:
            self.assertIn(expected, names)


if __name__ == '__main__':
    unittest.main()
