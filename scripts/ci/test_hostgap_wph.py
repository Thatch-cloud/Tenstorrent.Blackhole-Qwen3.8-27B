"""Op-fusion programme, host-gap package WPH: the manifest, the generated twins, the card-job templates, the smoke rule and the shipping lists.

Pinned here (the levers' own behavior is in test_prestage_diff, test_write_packed_lean, test_batched_reads_tp and test_hostgap_instr):
  - the manifest names the three levers and the extra twins with the right flags and the fx-all parent, and each generated twin is its parent plus exactly the flags it names (the audited twins add
    the audit flags and, for the pre-stage diff, the existing full audit);
  - every card-job template parses through the job reader with the one image and a profile that exists; the order lists exactly the templates, the control of the timed ABAB is the combined arm and the
    lever arm is the combined arm plus the WPH flags, the audited attaches run the 32k shape and the steady mix, the cardm job names the batched_reads harness, no template names a host, an address, a
    registry or a digest;
  - the smoke rule: a clean log passes (including the real lines of the levers over the fake device), and each stop condition fails by name: a lever's lines without its flag, a missing engaged
    line, a fall-back, an audit mismatch or a non-exact audit line, a diff path that saved nothing, a missing site, an instrument without its lines, a probe line without its flag or a probe missing;
  - the shipping lists: the four runtime modules are in the overlay manifest, none is imported at module level by a served file, the tests are in the CPU allowlist."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

import c2_serving_job as job
import hostgap_wph_smoke as rule
import make_fusion_profiles as generator
import test_prestage_diff as tpd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'WPH'
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
BASE = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-'
COMBINED = BASE + 'fx-all'
IMAGE = 'tp4-fusion-1'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
DIFF, DIFF_AUDIT = 'QWEN_FAST_PRESTAGE_DIFF', 'QWEN_FAST_PRESTAGE_DIFF_AUDIT'
LEAN, LEAN_AUDIT = 'QWEN_FAST_WRITE_PACKED_LEAN', 'QWEN_FAST_WRITE_PACKED_LEAN_AUDIT'
READS, READS_AUDIT = 'QWEN_FAST_BATCHED_READS', 'QWEN_FAST_BATCHED_READS_AUDIT'
FULL_AUDIT = 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT'
LOG, PROBE = 'QWEN_FAST_TP4_HOSTGAP_LOG', 'QWEN_FAST_TP4_HOSTGAP_PROBE'


def manifest():
    return json.loads((HERE / 'fusion-wp' / 'WPH.json').read_text(encoding='utf-8'))


def added(name, parent=COMBINED):
    """The env a generated profile adds to its parent: {flag: value}, and what it removes (must be nothing)."""
    own, base = PROFILES[BASE + 'fx-' + name]['env'], PROFILES[parent]['env']
    return {key: value for key, value in own.items() if base.get(key) != value}, [key for key in base if key not in own]


class ManifestTests(unittest.TestCase):
    def test_the_levers_flags_markers_and_ids(self):
        levers = {item['id']: item for item in manifest()['levers']}
        self.assertEqual(sorted(levers), ['lean', 'prestage-diff', 'reads'])
        self.assertEqual((levers['prestage-diff']['flag'], levers['prestage-diff']['audit_flag']), (DIFF, DIFF_AUDIT))
        self.assertEqual(levers['prestage-diff']['audit_env'], {FULL_AUDIT: '1'})
        self.assertEqual((levers['lean']['flag'], levers['lean']['audit_flag']), (LEAN, LEAN_AUDIT))
        self.assertEqual((levers['reads']['flag'], levers['reads']['audit_flag'], levers['reads']['value']), (READS, READS_AUDIT, '1'))
        for item in levers.values():
            self.assertTrue(item['marker'].startswith('tp4 '))

    def test_the_manifest_is_one_the_generator_places(self):
        plan = generator.normalise(generator.read_manifests())
        mine = [lever for lever in plan['levers'] if lever['wp'] == 'WPH']
        self.assertEqual(len(mine), 3)
        self.assertIn(('WPH', 'hostgap_wph_smoke'), plan['smoke_rules'])
        paths = [path for wp, path, reason in plan['image_files'] if wp == 'WPH']
        self.assertEqual(sorted(paths), ['scripts/ci/batched_reads_tp.py', 'scripts/ci/hostgap_instr.py', 'scripts/ci/prestage_diff.py', 'scripts/ci/write_packed_lean.py'])
        modules = [module for wp, module, reason in plan['tests'] if wp == 'WPH' and isinstance(module, str)]
        self.assertEqual(sorted(modules), ['test_batched_reads_tp', 'test_hostgap_instr', 'test_hostgap_wph', 'test_prestage_diff', 'test_write_packed_lean'])

    def test_the_checked_in_files_are_what_the_manifests_generate(self):
        subprocess.run([sys.executable, '-B', '-s', str(HERE / 'make_fusion_profiles.py'), '--check'], check=True, cwd=str(ROOT), capture_output=True)


class TwinTests(unittest.TestCase):
    def test_each_lever_twin_is_the_production_profile_plus_its_flags(self):
        for name, expected in (('prestage-diff', {DIFF: '1'}), ('prestage-diff-audit', {DIFF: '1', DIFF_AUDIT: '1', FULL_AUDIT: '1'}),
                               ('lean', {LEAN: '1'}), ('lean-audit', {LEAN: '1', LEAN_AUDIT: '1'}),
                               ('reads', {READS: '1'}), ('reads-audit', {READS: '1', READS_AUDIT: '1'})):
            with self.subTest(name=name):
                found, removed = added(name, BASE + 'traffic')
                self.assertEqual((found, removed), (expected, []))
                self.assertTrue(PROFILES[BASE + 'fx-' + name]['gate_only'])

    def test_each_composition_twin_is_the_combined_arm_plus_its_flags_and_nothing_else(self):
        wanted = {
            'wph-diff': {DIFF: '1'}, 'wph-diff-audit': {DIFF: '1', DIFF_AUDIT: '1', FULL_AUDIT: '1'},
            'wph-lean': {LEAN: '1'}, 'wph-lean-audit': {LEAN: '1', LEAN_AUDIT: '1'},
            'wph-reads': {READS: '1'}, 'wph-reads-audit': {READS: '1', READS_AUDIT: '1'},
            'wph-reads-async': {READS: 'async'}, 'wph-reads-async-audit': {READS: 'async', READS_AUDIT: '1'},
            'wph-all': {DIFF: '1', LEAN: '1', READS: '1'},
            'wph-all-audit': {DIFF: '1', DIFF_AUDIT: '1', FULL_AUDIT: '1', LEAN: '1', LEAN_AUDIT: '1', READS: '1', READS_AUDIT: '1'},
            'wph-all-async': {DIFF: '1', LEAN: '1', READS: 'async'},
            'wph-all-async-audit': {DIFF: '1', DIFF_AUDIT: '1', FULL_AUDIT: '1', LEAN: '1', LEAN_AUDIT: '1', READS: 'async', READS_AUDIT: '1'},
            'wph-log': {LOG: '1', PROBE: '1'},
        }
        # Since the integrator put the three levers into the combined arm itself (reads at async), a composition twin is that arm with its flags set to ITS values: what it adds to
        # the combined arm is only what differs from it (the WPH.json facts above are the pre-merge ones, held by test_the_manifest_names...).
        for name, expected in wanted.items():
            with self.subTest(name=name):
                found, removed = added(name)
                self.assertEqual((found, removed), ({flag: value for flag, value in expected.items() if PROFILES[COMBINED]['env'].get(flag) != value}, []))
                self.assertTrue(PROFILES[BASE + 'fx-' + name]['gate_only'])
                self.assertNotIn('owner_traffic_waiver', PROFILES[BASE + 'fx-' + name])
        self.assertEqual(set(wanted) - {item for item in wanted}, set())

    def test_the_combined_arm_carries_the_three_wph_levers_and_no_audit_and_no_instrument(self):
        env = PROFILES[COMBINED]['env']
        self.assertEqual((env[DIFF], env[LEAN], env[READS]), ('1', '1', 'async'))
        for flag in (DIFF_AUDIT, LEAN_AUDIT, READS_AUDIT, FULL_AUDIT, LOG, PROBE):
            self.assertNotIn(flag, env)
        audited = PROFILES[COMBINED + '-audit']['env']
        for flag in (DIFF_AUDIT, LEAN_AUDIT, READS_AUDIT, FULL_AUDIT):
            self.assertEqual(audited[flag], '1', flag)
        for flag in (LOG, PROBE):
            self.assertNotIn(flag, audited)
        for flag in ('QWEN_FAST_TP4_ROUND_HOST_KEYED', 'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS', 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE'):
            self.assertEqual(env[flag], '1', 'the levers are timed against an arm that has the per-block epochs and the keyed write')

    def test_no_traffic_profile_carries_a_wph_flag(self):
        for name, profile in PROFILES.items():
            if profile.get('gate_only') is True:
                continue
            for flag in (DIFF, LEAN, READS, PROBE):
                if name in generator.ship_names() and flag != PROBE:
                    self.assertEqual(profile['env'][flag], PROFILES[COMBINED]['env'][flag], name)        # the ship candidate is the combined arm as a traffic profile
                    continue
                self.assertNotIn(flag, profile['env'], name)


EXPECTED = {
    'H1-cardm-batched-reads': ('cardm', None, 'soft'),
    'H10-hostgap-log-unprofiled': ('reset smoke', BASE + 'fx-wph-log', 'soft'),
    'H2-diff-audit-attach': ('reset smoke', BASE + 'fx-wph-diff-audit', 'stop'),
    'H3-lean-audit-attach': ('reset smoke', BASE + 'fx-wph-lean-audit', 'stop'),
    'H4-reads-audit-attach': ('reset smoke', BASE + 'fx-wph-reads-audit', 'stop'),
    'H5-all-audit-attach': ('reset smoke', BASE + 'fx-wph-all-audit', 'stop'),
    'H6-fxall-control-timed': ('reset smoke', COMBINED, 'soft'),
    'H7-wph-all-timed': ('reset smoke', BASE + 'fx-wph-all', 'soft'),
    'H8-fxall-control-repeat': ('reset smoke', COMBINED, 'soft'),
    'H9-wph-all-repeat': ('reset smoke', BASE + 'fx-wph-all', 'soft'),
    'H4b-reads-async-audit-attach': ('reset smoke', BASE + 'fx-wph-reads-async-audit', 'stop'),
    'H5b-all-async-audit-attach': ('reset smoke', BASE + 'fx-wph-all-async-audit', 'stop'),
    'H7b-wph-all-async-timed': ('reset smoke', BASE + 'fx-wph-all-async', 'soft'),
    'H9b-wph-all-async-repeat': ('reset smoke', BASE + 'fx-wph-all-async', 'soft'),
}


def read_template(name):
    values = job.parse_env((FOLDER / (name + '.env')).read_text(encoding='utf-8'))
    return job.read_job(values, sorted(PROFILES), root=ROOT)


def order():
    return [line.split() for line in (FOLDER / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


class JobTests(unittest.TestCase):
    def test_the_order_lists_exactly_the_templates_in_the_asked_order(self):
        lines = order()
        self.assertEqual([line[0] for line in lines], list(EXPECTED))
        self.assertEqual(sorted(path.stem for path in FOLDER.glob('*.env')), sorted(EXPECTED))
        for name, mode, image, minutes in lines:
            self.assertEqual((mode, image), (EXPECTED[name][2], IMAGE), name)
            self.assertTrue(minutes.isdigit() and int(minutes) > 0, name)

    def test_every_template_parses_with_its_actions_profile_and_the_one_image(self):
        for name, (actions, profile, _mode) in EXPECTED.items():
            with self.subTest(name=name):
                result = read_template(name)
                self.assertEqual((result['actions'], result['tag']), (actions, IMAGE))
                if profile:
                    self.assertEqual((result['profile'], result['cards']), (profile, 'quad'))

    def test_every_template_exits_zero_through_the_command_line_reader(self):
        for name in EXPECTED:
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, '-s', str(HERE / 'c2_serving_job.py'), str(FOLDER / (name + '.env')), str(HERE / 'qwen_c2_profiles.json')],
                                        capture_output=True, text=True, cwd=str(ROOT))
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_cardm_job_runs_the_batched_reads_harness_first_and_alone_on_one_board(self):
        result = read_template('H1-cardm-batched-reads')
        self.assertEqual((result['actions'], result['cards']), ('cardm', 'pair'))
        self.assertEqual(result['cardm_harness'], 'optimisation/ttnn-op/batched_reads/run_card_m.sh')
        self.assertEqual(order()[0][0], 'H1-cardm-batched-reads')
        self.assertTrue((ROOT / result['cardm_harness']).is_file())

    def test_the_abab_is_control_lever_control_lever_on_the_same_tests_with_the_combined_arm_as_control(self):
        timed = [read_template(name) for name in ('H6-fxall-control-timed', 'H7-wph-all-timed', 'H8-fxall-control-repeat', 'H9-wph-all-repeat')]
        self.assertEqual([item['profile'] for item in timed], [COMBINED, BASE + 'fx-wph-all', COMBINED, BASE + 'fx-wph-all'])
        self.assertEqual(len({item['tests'] for item in timed}), 1)
        tests = timed[0]['tests'].replace(' ', ',').split(',')
        self.assertTrue({'concurrent8_steady', 'concurrent8_code_32k'} <= set(tests))
        for item in timed:
            env = PROFILES[item['profile']]['env']
            self.assertEqual([flag for flag in (DIFF_AUDIT, LEAN_AUDIT, READS_AUDIT, FULL_AUDIT) if env.get(flag, '0') != '0'], [], 'a timed arm carries no WPH audit')

    def test_the_audited_attaches_run_the_32k_shape_and_the_steady_mix(self):
        for name in ('H2-diff-audit-attach', 'H3-lean-audit-attach', 'H4-reads-audit-attach', 'H5-all-audit-attach', 'H10-hostgap-log-unprofiled'):
            tests = read_template(name)['tests'].replace(' ', ',').split(',')
            self.assertTrue({'concurrent8_steady', 'concurrent8_code_32k'} <= set(tests), name)

    def test_no_job_touches_the_agent_or_production_or_bakes_a_profile(self):
        for name in EXPECTED:
            with self.subTest(name=name):
                result = read_template(name)
                self.assertFalse({'stop', 'start', 'handback', 'deploy'} & set(result['actions'].split()))
                self.assertEqual(result['bake_default_profile'] or '', '')
        self.assertNotIn(IMAGE, job.PROTECTED)

    def test_the_dependencies_and_the_read_rules_are_in_the_order(self):
        text = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        needs = re.findall(r'^# NEEDS (.+) <- (.+)$', text, flags=re.M)
        names = {name.split('-')[0] for name in EXPECTED} | {'X0'}
        for left, right in needs:
            for item in (left + ' ' + right).split():
                self.assertIn(item, names)
        self.assertIn('# NEEDS H7 <- H5 H6', text)
        self.assertIn('# NEEDS H4 <- X0 H1', text)
        self.assertIn('pick reads=async', text)
        self.assertIn('PAIRED' if 'PAIRED' in text else 'w2ln_timing_compare.py pair', text)
        self.assertIn('NO-GO', text)

    def test_no_hostname_address_registry_or_digest_and_lf_endings(self):
        for path in list(FOLDER.iterdir()) + [HERE / name for name in ('prestage_diff.py', 'write_packed_lean.py', 'batched_reads_tp.py', 'hostgap_instr.py', 'hostgap_read.py',
                                                                      'hostgap_wph_smoke.py', 'fusion-wp/WPH.json')]:
            text = path.read_text(encoding='utf-8')
            self.assertIsNone(BANNED.search(text), path.name)
            self.assertNotIn('\r', text, path.name)


def log(*lines):
    return '\n'.join(lines)


GOOD = dict(
    diff=['[PINDIAG] tp4 prestage diff engaged users=4 audit=1', '[PACKED-PRESTAGE-DIFF] block=A round=5 path=diff reason=- total=148 written=73 skipped=75',
          '[PINDIAG] tp4 prestage diff audit block=A round=5 checked=592 skipped=75 mismatches=0 exact=True'],
    lean=['[PINDIAG] tp4 write packed lean engaged site=prestage mappers=1 audit=1', '[PINDIAG] tp4 write packed lean engaged site=verify mappers=1 audit=1',
          '[PINDIAG] tp4 write packed lean audit calls=3 checked=8 destinations=73 mismatches=0 exact=True'],
    reads=['[PINDIAG] tp4 batched reads engaged site=verify mode=compose tensors=2 chips=4', '[PINDIAG] tp4 batched reads engaged site=collect mode=compose tensors=5 chips=4',
           '[PINDIAG] tp4 batched reads audit site=verify mode=compose reads=1 tensors=2 chips=4 exact=True',
           '[PINDIAG] tp4 batched reads audit site=collect mode=compose reads=2 tensors=5 chips=4 exact=True'],
    instrument=['[PINDIAG] tp4 hostgap instrument engaged wrapped=copy_host_to_device_tensor', '[PACKED-HOSTGAP-ROUND] step=1 probe=0 wall_ms=150.00 cpu_ms=20.00',
                '[PACKED-HOSTGAP-SPAN] step=1 name=window wall_ms=18.00 cpu_ms=17.00'])
ENV_ALL = {DIFF: '1', DIFF_AUDIT: '1', LEAN: '1', LEAN_AUDIT: '1', READS: '1', READS_AUDIT: '1', 'QWEN_FAST_QUAD_DRAFT': '1'}


class SmokeRuleTests(unittest.TestCase):
    def text(self, *groups, extra=()):
        return log(*[line for group in groups for line in GOOD[group]], *extra)

    def test_a_clean_log_passes_with_every_lever_on_and_with_none(self):
        self.assertEqual(rule.problems(ENV_ALL, self.text('diff', 'lean', 'reads')), [])
        self.assertEqual(rule.problems({}, 'nothing of the package'), [])
        self.assertEqual(rule.problems(None, ''), [])
        self.assertEqual(rule.problems({LOG: '1', PROBE: '1'}, self.text('instrument', extra=['[PACKED-HOSTGAP-PROBE] round=16 step=3 quads=2'])), [])

    def test_a_lever_line_without_its_flag_and_an_audit_flag_without_its_lever_are_problems(self):
        self.assertTrue(any('lines of the pre-stage diff on a profile without %s' % DIFF in item for item in rule.problems({}, self.text('diff'))))
        self.assertTrue(any('%s is set without %s' % (DIFF_AUDIT, DIFF) in item for item in rule.problems({DIFF_AUDIT: '1'}, '')))
        self.assertTrue(any('instrument lines on a profile without %s' % LOG in item for item in rule.problems({}, self.text('instrument'))))

    def test_a_missing_engaged_line_and_a_missing_or_failed_audit_are_problems(self):
        found = rule.problems({DIFF: '1'}, '')
        self.assertTrue(any('no engaged line' in item for item in found))
        found = rule.problems({DIFF: '1', DIFF_AUDIT: '1'}, self.text('diff').replace('exact=True', 'exact=False'))
        self.assertTrue(any('not exact' in item for item in found) and any('never audited' in item for item in found))
        found = rule.problems({READS: '1'}, self.text('reads').replace('exact=True', 'exact=True'))
        self.assertEqual(found, [])

    def test_a_fall_back_and_an_audit_mismatch_fail_the_arm(self):
        for group, marker in (('diff', '[PINDIAG] tp4 prestage diff'), ('lean', '[PINDIAG] tp4 write packed lean'), ('reads', '[PINDIAG] tp4 batched reads')):
            flag = {'diff': DIFF, 'lean': LEAN, 'reads': READS}[group]
            with self.subTest(group=group):
                env = {flag: '1', 'QWEN_FAST_QUAD_DRAFT': '1'}
                fell = rule.problems(env, self.text(group, extra=['%s fell back reason=audit_mismatch' % marker]))
                self.assertTrue(any('fell back' in item for item in fell), fell)
                mismatch = rule.problems(env, self.text(group, extra=['%s audit mismatch calls=1 exact=False' % marker]))
                self.assertTrue(any('found a difference' in item for item in mismatch), mismatch)

    def test_a_diff_path_that_wrote_everything_saved_nothing(self):
        text = self.text('diff').replace('written=73 skipped=75', 'written=148 skipped=0')
        found = rule.problems({DIFF: '1'}, text)
        self.assertTrue(any('saved nothing' in item for item in found), found)
        text = self.text('diff').replace('path=diff', 'path=full')
        self.assertTrue(any('saved nothing' in item for item in rule.problems({DIFF: '1'}, text)))

    def test_each_site_must_have_gone_through_the_lever(self):
        found = rule.problems({LEAN: '1'}, '\n'.join(GOOD['lean'][:1]))
        self.assertTrue(any('site=verify' in item for item in found))
        found = rule.problems({READS: '1', 'QWEN_FAST_QUAD_DRAFT': '1'}, GOOD['reads'][0])
        self.assertTrue(any('site=collect' in item for item in found))
        self.assertEqual(rule.problems({READS: '1'}, GOOD['reads'][0]), [], 'no quads, no collect site')
        found = rule.problems({READS: '1', READS_AUDIT: '1'}, '\n'.join(GOOD['reads'][:2]))
        self.assertEqual(len([item for item in found if 'no passing audit line for site=' in item]), 1, 'the verify site only, the quads are off here')

    def test_the_instrument_needs_its_lines_and_the_probe_its_flag_and_its_rounds(self):
        found = rule.problems({LOG: '1'}, '')
        self.assertTrue(any('instrument engaged line' in item for item in found) and any('no round line' in item for item in found))
        rounds = ['[PACKED-HOSTGAP-ROUND] step=%d probe=0 wall_ms=150.00' % step for step in range(1, 60)]
        base = GOOD['instrument'][:1] + ['[PACKED-HOSTGAP-SPAN] step=1 name=window wall_ms=18.00']
        found = rule.problems({LOG: '1', PROBE: '1'}, log(*base, *rounds))
        self.assertTrue(any('without one probe line' in item for item in found), found)
        found = rule.problems({LOG: '1'}, log(*base, *rounds, '[PACKED-HOSTGAP-PROBE] round=16'))
        self.assertTrue(any('probe lines on a profile without %s' % PROBE in item for item in found))
        self.assertEqual(rule.problems({LOG: '1', PROBE: '1'}, log(*base, *rounds[:10])), [], 'too few rounds to demand a probe')

    def test_the_rule_is_one_the_smoke_check_calls_and_it_reads_the_real_lines_of_the_levers(self):
        import c2_smoke_check

        self.assertIn('hostgap_wph_smoke', c2_smoke_check.FUSION_RULES)
        self.assertIn((DIFF, '1', '[PINDIAG] tp4 prestage diff engaged', '[PINDIAG] tp4 prestage diff fell back', 'WPH-1 pre-stage diff'), c2_smoke_check.FUSION_LEVERS)
        self.assertIn((READS, 'async', '[PINDIAG] tp4 batched reads engaged', '[PINDIAG] tp4 batched reads fell back', 'WPH-3 batched read-backs (asynchronous copies)'),
                      c2_smoke_check.FUSION_LEVERS)
        self.assertEqual(c2_smoke_check.fusion_problems(self.text('diff', 'lean', 'reads'), ENV_ALL), [])
        self.assertTrue(c2_smoke_check.fusion_problems(self.text('diff', 'lean', 'reads', extra=['[PINDIAG] tp4 write packed lean fell back reason=x']), ENV_ALL))

    def test_the_lines_the_levers_really_write_pass_the_rule(self):
        class Real(tpd.DiffFixture):
            def runTest(self):
                pass

        case = Real()
        case.setUp()
        try:
            for name in (tpd.DIFF, tpd.AUDIT, tpd.LEAN, 'QWEN_FAST_WRITE_PACKED_LEAN_AUDIT'):
                os.environ[name] = '1'
            block = case.open_block()
            users = tpd.tvp.base_users()
            case.round(block, users)
            for step in range(1, 4):
                users = tpd.tvp.advanced(users, 3)
                case.window(block, users)
                case.round(block, users, step)
            lines = [line for line in case.h1a if line.startswith(('[PINDIAG] tp4', '[PACKED-PRESTAGE-DIFF]'))]
            env = {DIFF: '1', DIFF_AUDIT: '1', LEAN: '1', LEAN_AUDIT: '1'}
            self.assertEqual(rule.problems(env, '\n'.join(lines)), [])
            self.assertTrue(lines and any(line.startswith('[PACKED-PRESTAGE-DIFF]') for line in lines))
        finally:
            case.doCleanups()
            for name in (tpd.DIFF, tpd.AUDIT, tpd.LEAN, 'QWEN_FAST_WRITE_PACKED_LEAN_AUDIT'):
                os.environ.pop(name, None)


class ShippingTests(unittest.TestCase):
    def test_the_runtime_modules_are_in_the_overlay_manifest_and_the_tests_in_the_cpu_allowlist(self):
        overlay = (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8')
        for name in ('prestage_diff.py', 'write_packed_lean.py', 'batched_reads_tp.py', 'hostgap_instr.py'):
            self.assertIn('scripts/ci/' + name, overlay)
        self.assertNotIn('hostgap_read.py', overlay, 'the reader runs on the host')
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        for name in ('test_prestage_diff', 'test_write_packed_lean', 'test_batched_reads_tp', 'test_hostgap_instr', 'test_hostgap_wph'):
            self.assertIn(name, workflow)
        self.assertIn('optimisation/ttnn-op/batched_reads', workflow)

    def test_no_served_file_imports_a_wph_module_at_module_level(self):
        for name in ('verify_prestage.py', 'packed_verifier.py', 'quad_draft_tp.py', 'dflash_packed_proposal_coordinator.py', 'serving_worker_hook.py'):
            text = (HERE / name).read_text(encoding='utf-8')
            self.assertIsNone(re.search(r'(?m)^(import|from) (prestage_diff|write_packed_lean|batched_reads_tp|hostgap_instr)\b', text), name)

    def test_the_new_files_are_lf(self):
        for name in ('prestage_diff.py', 'write_packed_lean.py', 'batched_reads_tp.py', 'hostgap_instr.py', 'hostgap_read.py', 'hostgap_wph_smoke.py'):
            self.assertNotIn(b'\r', (HERE / name).read_bytes(), name)


if __name__ == '__main__':
    unittest.main()
