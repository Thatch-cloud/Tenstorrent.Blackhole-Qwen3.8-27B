"""The NVMe cold tier's gate-only twins of the ship-7 candidate (make_nvme_profiles, tp4/kv-nvme-2, docs/tp4-kv-nvme.md section 12), held on the CPU.

  -nvme-s7         the TIMED arm:   the candidate (`-fx-traffic-s7`) + the tier, its owner waiver id replaced by the gate marker, gate_only;
  -nvme-s7-audit   the AUDITED arm: the ship-7 audited twin (`-fx-s7-audit`) + the tier + QWEN_PREFIX_NVME_AUDIT=1;
  -nvme-s7-fmt     the FIRST gate:  the audited arm + QWEN_PREFIX_NVME_FORMAT=1 (the gate-only force-format).

Held here: the checked-in file is what the generator makes (and every other generator's --check still holds beside it); each twin is its parent plus exactly the tier's names, the tier's
host share cut to the checkpoint store's GiB + 4, and nothing else; the retention acknowledgement and the gate-only override are in, REQUIRE_SED is never named (the default, 1, stands),
the format override is the first gate's alone and the audit the audited arms'; every twin passes the contract's boot checks and the tier's own flag parser, the same names on the traffic
parent are refused; no other profile names the tier; the host tier of an arm cannot hold the biggest session; nothing private is in a twin.

Stdlib only (no torch: the contract's boot checks are called directly). Run: `python3 -B -m unittest test_nvme_profiles` from scripts/ci."""

import copy
import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check  # noqa: E402
import make_fusion_profiles as fusion  # noqa: E402
import make_nvme_profiles as generator  # noqa: E402
import make_s7_profiles as s7  # noqa: E402
import profile_twins  # noqa: E402
import qwen_kv_nvme  # noqa: E402
import qwen_prefix_registry as registry  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
PROFILES_TEXT = PROFILES_PATH.read_text(encoding='utf-8')
PROFILES = json.loads(PROFILES_TEXT)['profiles']
SHIP, AUDIT_PARENT = generator.SHIP, generator.AUDIT_PARENT
TIMED, AUDIT, FORMAT = generator.TIMED, generator.AUDIT, generator.FORMAT
PARENT_OF = {TIMED: SHIP, AUDIT: AUDIT_PARENT, FORMAT: AUDIT_PARENT}
GIB = 1 << 30
BIGGEST_SESSION_BYTES = 8.56e9              # a 254k session's KV (docs/tp4-kv-nvme.md section 12)


def boot_problems(profile, environ=None):
    """Every boot check of serving_c2_contract.boot, in its order, for a profile (what test_octo3_contract.boot_problems and test_s7_profiles run, without their torch import)."""
    if environ is None:
        environ = {'QWEN_C2_SERVING': '1'}
        if profile.get('gate_only') is True:
            environ['QWEN_C2_GATE'] = '1'            # what the gate workflow adds to a gate-only profile's container
    contract.apply_environment(profile, environ)
    found = []
    for check in (lambda: contract.gate_problems(profile, environ) + contract.waiver_problems(profile, environ) + contract.traffic_waiver_problems(profile),
                  lambda: contract.prefix_reuse_problems(profile), lambda: contract.levern_problems(profile), lambda: contract.multi_problems(profile),
                  lambda: contract.octo_problems(profile, environ), lambda: contract.w2_problems(profile, environ), lambda: contract.upload_problems(profile),
                  lambda: contract.drafter_problems(profile), lambda: contract.parked_problems(profile, environ), lambda: contract.mesh_problems(profile)):
        found += check()
    return (found + contract.prefix_policy_problems(profile) + contract.exact_resume_problems(profile) + contract.sched_problems(profile) + contract.sched2_problems(profile)
            + contract.ksplit_problems(profile))


def named(name):
    return dict(copy.deepcopy(PROFILES[name]), name=name)            # (a copy: the tests below edit the env)


class GeneratorTests(unittest.TestCase):
    def test_the_checked_in_twins_are_what_the_generator_makes(self):
        self.assertEqual(generator.render(generator.generate(json.loads(PROFILES_TEXT))), PROFILES_TEXT)

    def test_the_generator_is_idempotent_and_leaves_every_other_profile_alone(self):
        data = json.loads(PROFILES_TEXT)
        once = generator.generate(data)
        self.assertEqual(generator.generate(once), once)
        twins = set(generator.twin_names())
        self.assertEqual([name for name in once['profiles'] if name not in twins], [name for name in data['profiles'] if name not in twins])
        for name, profile in data['profiles'].items():
            if name not in twins:
                self.assertEqual(once['profiles'][name], profile, name)

    def test_the_twins_follow_the_ship_7_audited_twin_in_spec_order_and_leave_the_files_last_block_alone(self):
        names = list(PROFILES)
        start = names.index(s7.AUDIT) + 1
        self.assertEqual(names[start:start + len(generator.NAMES)], list(generator.NAMES))
        self.assertEqual(generator.NAMES, (TIMED, AUDIT, FORMAT))
        self.assertNotIn(names[-1], generator.NAMES, 'the file\'s end is make_kvread_profiles\'')
        without = json.loads(PROFILES_TEXT)
        for name in generator.NAMES:
            del without['profiles'][name]
        regenerated = list(generator.generate(without)['profiles'])
        self.assertEqual(regenerated, names)

    def test_every_other_generator_of_the_chain_still_holds_beside_the_twins(self):
        data = json.loads(PROFILES_TEXT)
        self.assertEqual(s7.render(s7.generate(data)), PROFILES_TEXT)
        self.assertEqual(generator.render(generator.generate(s7.generate(data))), PROFILES_TEXT)

    def test_the_names_are_registered_with_the_census_exemptions_and_fit(self):
        for name in generator.NAMES:
            self.assertIn(name, profile_twins.twin_names())
            self.assertIn(name, PROFILES)
            self.assertLessEqual(len(name), generator.MAX_NAME)
        self.assertEqual(len(profile_twins.twin_names()), len(set(profile_twins.twin_names())))
        self.assertEqual(generator.twin_names(), generator.NAMES)

    def test_the_names_are_not_under_the_fusion_namespace_whose_write_drops_what_it_does_not_own(self):
        for name in generator.NAMES:
            self.assertFalse(name.startswith(fusion.NAMESPACE), name)
        self.assertEqual(fusion.generate_profiles(json.loads(PROFILES_TEXT), fusion.normalise(fusion.read_manifests()))['profiles'][TIMED], PROFILES[TIMED])

    def test_the_command_line_checks_and_writes(self):
        import contextlib
        import io
        import tempfile

        data = json.loads(PROFILES_TEXT)
        for name in generator.NAMES:
            del data['profiles'][name]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profiles.json'
            path.write_text(generator.render(data), encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(generator.main(['--check', '--path', str(path)]), 1)
            self.assertIn('make_nvme_profiles.py', stderr.getvalue())
            self.assertEqual(generator.main(['--write', '--path', str(path)]), 0)
            self.assertEqual(path.read_text(encoding='utf-8'), PROFILES_TEXT)
            self.assertEqual(generator.main(['--check', '--path', str(path)]), 0)
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    generator.main(['--path', str(path)])

    def test_a_parent_that_cannot_be_twinned_is_refused_by_name(self):
        data = json.loads(PROFILES_TEXT)
        cases = (('ship has no waiver', lambda p: p[SHIP].pop(generator.WAIVER_FIELD), 'is not a traffic profile with an owner traffic waiver'),
                 ('ship is gate only', lambda p: p[SHIP].update(gate_only=True), 'is not a traffic profile with an owner traffic waiver'),
                 ('ship lost its waiver id', lambda p: p[SHIP]['env'].pop(generator.WAIVER_ENV), 'does not carry its waiver id'),
                 ('audit parent is not gate only', lambda p: p[AUDIT_PARENT].pop('gate_only'), 'is not the gate-only audited twin of ship 7'),
                 ('audit parent has no marker', lambda p: p[AUDIT_PARENT]['env'].pop('QWEN_C2_GATE_PROFILE'), 'is not the gate-only audited twin of ship 7'),
                 ('ship already names the tier', lambda p: p[SHIP]['env'].update({'QWEN_PREFIX_NVME': '1'}), 'already names QWEN_PREFIX_NVME'),
                 ('audit parent already names a knob', lambda p: p[AUDIT_PARENT]['env'].update({'QWEN_PREFIX_NVME_TTL_S': '60'}), 'already names QWEN_PREFIX_NVME_TTL_S'),
                 ('not four cards', lambda p: p[SHIP].update(mesh_device=None), 'is not a four-card TP4 profile'),
                 ('not tp4', lambda p: p[AUDIT_PARENT]['env'].update({'QWEN_FAST_TP': '2'}), 'is not a four-card TP4 profile'),
                 ('no store budget', lambda p: p[SHIP]['env'].pop('QWEN_PREFIX_STORE_GIB'), 'names no numeric'),
                 ('no room for a cut', lambda p: p[AUDIT_PARENT]['env'].update({'QWEN_PREFIX_HOST_TIER_GIB': '18'}), 'would not be a cut'))
        for label, edit, needle in cases:
            with self.subTest(label):
                broken = copy.deepcopy(data)
                edit(broken['profiles'])
                with self.assertRaises(ValueError) as caught:
                    generator.generate(broken)
                self.assertIn(needle, str(caught.exception))
        for gone in (SHIP, AUDIT_PARENT):
            with self.subTest(gone=gone), self.assertRaises(ValueError):
                broken = copy.deepcopy(data)
                del broken['profiles'][gone]
                generator.generate(broken)


class TwinTests(unittest.TestCase):
    def test_the_timed_twin_is_the_candidate_with_the_gate_marker_for_its_waiver_id_plus_the_tier_and_nothing_else(self):
        mine, theirs = copy.deepcopy(PROFILES[TIMED]), copy.deepcopy(PROFILES[SHIP])
        waiver = theirs.pop(generator.WAIVER_FIELD)
        self.assertEqual(theirs['env'].pop(generator.WAIVER_ENV), waiver['id'])
        self.assertEqual(mine['env'].pop('QWEN_C2_GATE_PROFILE'), '1')
        self.assertNotIn(generator.WAIVER_ENV, mine['env'])
        self.assertNotIn(generator.WAIVER_FIELD, mine)
        self.assertEqual(mine['env'].pop(generator.TIER_GIB), '20')
        self.assertEqual(theirs['env'].pop(generator.TIER_GIB), '48')
        for name, value in generator.NVME_ENV.items():
            self.assertEqual(mine['env'].pop(name), value, name)
        self.assertTrue(mine.pop('description').startswith('GATE ONLY, UNQUALIFIED'))
        theirs.pop('description')
        self.assertIs(mine.pop('gate_only'), True)
        self.assertNotIn('gate_only', theirs)
        self.assertEqual(mine, theirs, 'everything else - the engine, the memory plan, every other flag, the ledger trim - is the candidate\'s')
        self.assertEqual(PROFILES[TIMED]['env']['QWEN_FAST_ADMISSION_LEDGER_TRIM'], '1')

    def test_the_audited_and_the_format_twins_are_the_audited_parent_plus_the_tier(self):
        for name, extra in ((AUDIT, generator.AUDIT_ENV), (FORMAT, dict(generator.AUDIT_ENV, **generator.FORMAT_ENV))):
            with self.subTest(twin=name):
                mine, theirs = copy.deepcopy(PROFILES[name]), copy.deepcopy(PROFILES[AUDIT_PARENT])
                self.assertEqual(mine['env'].pop(generator.TIER_GIB), '20')
                self.assertEqual(theirs['env'].pop(generator.TIER_GIB), '48')
                for flag, value in dict(generator.NVME_ENV, **extra).items():
                    self.assertEqual(mine['env'].pop(flag), value, flag)
                self.assertTrue(mine.pop('description').startswith('GATE ONLY, UNQUALIFIED'))
                theirs.pop('description')
                self.assertEqual(mine, theirs, 'every audit of the shipped stack stays: the engine, the memory plan and every other flag are the parent\'s')
        self.assertEqual(PROFILES[FORMAT]['env'], dict(PROFILES[AUDIT]['env'], QWEN_PREFIX_NVME_FORMAT='1'), 'the first gate is the audited arm plus the force-format')

    def test_every_twin_is_gate_only_without_a_waiver_and_in_the_house_style(self):
        for name in generator.NAMES:
            with self.subTest(twin=name):
                profile = PROFILES[name]
                self.assertIs(profile.get('gate_only'), True)
                self.assertNotIn(generator.WAIVER_FIELD, profile)
                self.assertNotIn(generator.WAIVER_ENV, profile['env'])
                self.assertEqual(profile['env']['QWEN_C2_GATE_PROFILE'], '1', 'the profile\'s own marker: serving_octo admits the octo shape in a gate run only')
                text = profile['description']
                self.assertTrue(text.startswith('GATE ONLY, UNQUALIFIED'))
                self.assertIn(PARENT_OF[name], text)
                self.assertTrue(text.endswith('Generated by scripts/ci/make_nvme_profiles.py: regenerate, never hand-merge.'))
                for flag in generator.NVME_ENV:
                    self.assertIn(flag, text)
                self.assertIn('QWEN_PREFIX_NVME_REQUIRE_SED is unset', text)

    def test_the_tier_names_each_twin_carries(self):
        for name in generator.NAMES:
            env = PROFILES[name]['env']
            with self.subTest(twin=name):
                self.assertEqual({flag: env.get(flag) for flag in generator.NVME_ENV}, generator.NVME_ENV)
                self.assertEqual(env['QWEN_PREFIX_NVME_RETENTION_ACK'], '1', 'the retention acknowledgement: the contract and the engine refuse the tier without it')
                self.assertEqual(env['QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION'], '1', 'the gate-only override: the rig may carry zero-retention signals')
                self.assertEqual(env['QWEN_PREFIX_NVME_DEVICE'], generator.IN_CONTAINER_DEVICE)
        self.assertEqual(generator.NVME_ENV['QWEN_PREFIX_NVME_DEVICE'], '/dev/c2-cache-drive')

    def test_require_sed_is_never_named_and_the_format_and_the_audit_belong_to_their_arms(self):
        for name in generator.NAMES:
            env = PROFILES[name]['env']
            with self.subTest(twin=name):
                self.assertNotIn(qwen_kv_nvme.ENV_REQUIRE_SED, env, 'the default (1) stands: the drive is Opal-set and unlocked, with the status file')
                self.assertNotIn(qwen_kv_nvme.ENV_SED_STATUS, env)
                for flag in (qwen_kv_nvme.ENV_TTL_S, qwen_kv_nvme.ENV_SPILL, qwen_kv_nvme.ENV_THREADS, qwen_kv_nvme.ENV_THROUGH_BLOCKS, qwen_kv_nvme.ENV_THROUGH_IDLE_S,
                             qwen_kv_nvme.ENV_DRAIN_S, qwen_kv_nvme.ENV_DISCARD, qwen_kv_nvme.ENV_VERIFY, qwen_kv_nvme.ENV_COMPAT, qwen_kv_nvme.ENV_CMD_DIR):
                    self.assertNotIn(flag, env, 'defaults')
        self.assertEqual([name for name in generator.NAMES if qwen_kv_nvme.ENV_FORMAT in PROFILES[name]['env']], [FORMAT])
        self.assertEqual([name for name in generator.NAMES if qwen_kv_nvme.ENV_AUDIT in PROFILES[name]['env']], [AUDIT, FORMAT])
        self.assertEqual(PROFILES[FORMAT]['env'][qwen_kv_nvme.ENV_FORMAT], '1')
        self.assertEqual(PROFILES[AUDIT]['env'][qwen_kv_nvme.ENV_AUDIT], '1')

    def test_the_tiers_own_parser_reads_each_twin(self):
        for name in generator.NAMES:
            with self.subTest(twin=name):
                env = {key: str(value) for key, value in PROFILES[name]['env'].items()}
                plan, problems = qwen_kv_nvme.parse(env)
                self.assertEqual(problems, [])
                self.assertEqual(qwen_kv_nvme.flag_problems(env), [])
                self.assertEqual(plan.device, generator.IN_CONTAINER_DEVICE)
                self.assertTrue(plan.require_sed, 'REQUIRE_SED at its default')
                self.assertTrue(plan.allow_zero_retention)
                self.assertEqual(plan.format, 'force' if name == FORMAT else 'auto')
                self.assertEqual(plan.audit, name in (AUDIT, FORMAT))
                self.assertEqual(plan.spill, 'through')
                self.assertEqual(plan.restore, 'host')
                self.assertTrue(qwen_kv_nvme.PATH.fullmatch(plan.device))

    def test_the_device_path_is_the_runners(self):
        try:
            import c2_cache_drive
        except ImportError:
            self.skipTest('no c2_cache_drive module in this tree')
        self.assertEqual(generator.IN_CONTAINER_DEVICE, c2_cache_drive.IN_CONTAINER_DEVICE)

    def test_the_names_are_the_tiers_and_the_contracts(self):
        self.assertEqual(set(generator.NVME_ENV), {qwen_kv_nvme.ENV_SWITCH, qwen_kv_nvme.ENV_DEVICE, qwen_kv_nvme.ENV_RETENTION_ACK, qwen_kv_nvme.ENV_ALLOW_ZERO_RETENTION})
        self.assertEqual(generator.AUDIT_ENV, {qwen_kv_nvme.ENV_AUDIT: '1'})
        self.assertEqual(generator.FORMAT_ENV, {qwen_kv_nvme.ENV_FORMAT: '1'})
        self.assertEqual(generator.REQUIRE_SED, qwen_kv_nvme.ENV_REQUIRE_SED)
        self.assertEqual(set(qwen_kv_nvme.NAMES), set(contract.PREFIX_NVME_NAMES))
        self.assertEqual(generator.TIER_GIB, registry.ENV_TIER_GIB)
        self.assertEqual(generator.STORE_GIB, registry.ENV_STORE_GIB)
        self.assertEqual(generator.WAIVER_ENV, contract.OCTO_WAIVER_ENV)
        self.assertEqual(set(generator.MARKER_ENV), {contract.GATE_PROFILE_MARKER})


class HostShareTests(unittest.TestCase):
    def test_the_host_tier_holds_four_gib_of_kv_pages_so_the_biggest_session_has_to_come_from_the_drive(self):
        for name in generator.NAMES:
            env = PROFILES[name]['env']
            with self.subTest(twin=name):
                self.assertEqual(float(env[generator.TIER_GIB]), float(env[generator.STORE_GIB]) + generator.KV_SHARE_GIB)
                config = registry.tier_config(dict(env), registry.store_budget_bytes(dict(env)))
                self.assertEqual(config.kv_bytes, generator.KV_SHARE_GIB * GIB)
                self.assertLess(config.kv_bytes, BIGGEST_SESSION_BYTES)

    def test_the_parents_keep_their_production_share(self):
        for parent in (SHIP, AUDIT_PARENT):
            env = PROFILES[parent]['env']
            self.assertEqual((env[generator.TIER_GIB], env[generator.STORE_GIB]), ('48', '16'))

    def test_the_share_is_computed_from_the_parent(self):
        profile = {'env': {generator.STORE_GIB: '12.5'}}
        self.assertEqual(generator.tier_share(profile), '16.5')
        profile = {'env': {generator.STORE_GIB: '16'}}
        self.assertEqual(generator.tier_share(profile), '20')


class ContractTests(unittest.TestCase):
    def test_every_twin_passes_every_boot_check_and_the_tiers_policy_rules(self):
        for name in generator.NAMES:
            with self.subTest(twin=name):
                self.assertEqual(boot_problems(named(name)), [])
                self.assertEqual(contract.prefix_policy_problems(named(name)), [])
        self.assertEqual(boot_problems(named(SHIP)), [], 'the candidate itself is untouched')
        self.assertEqual(boot_problems(named(AUDIT_PARENT)), [])

    def test_the_candidates_waiver_id_on_a_gate_only_twin_is_refused(self):
        twin = named(TIMED)
        twin['env'][generator.WAIVER_ENV] = PROFILES[SHIP][generator.WAIVER_FIELD]['id']
        self.assertTrue(any('traffic profile' in problem for problem in contract.octo_problems(twin)))

    def test_the_same_names_on_the_traffic_candidate_are_refused(self):
        traffic = named(SHIP)
        traffic['env'].update(generator.NVME_ENV)
        traffic['env'][generator.TIER_GIB] = '20'
        problems = contract.prefix_policy_problems(traffic)
        self.assertTrue(any('QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION' in problem and 'gate' in problem for problem in problems), problems)
        self.assertTrue(any('QWEN_PREFIX_NVME=1' in problem and contract.TRAFFIC_WAIVER in problem for problem in problems), problems)
        for flag, value in (('QWEN_PREFIX_NVME_FORMAT', '1'), ('QWEN_PREFIX_NVME_AUDIT', '1')):
            traffic = named(SHIP)
            traffic['env'].update(generator.NVME_ENV)
            traffic['env'][flag] = value
            self.assertTrue(any(flag in problem for problem in contract.prefix_policy_problems(traffic)), flag)

    def test_the_tier_without_the_acknowledgement_or_the_host_tier_is_refused(self):
        twin = named(AUDIT)
        del twin['env']['QWEN_PREFIX_NVME_RETENTION_ACK']
        self.assertTrue(contract.prefix_policy_problems(twin))
        twin = named(AUDIT)
        del twin['env'][generator.TIER_GIB]
        self.assertTrue(any(generator.TIER_GIB in problem for problem in contract.prefix_policy_problems(twin)))

    def test_an_inherited_environment_never_reaches_the_parents_and_the_twins_keep_their_own(self):
        inherited = {name: '1' for name in contract.PREFIX_NVME_NAMES}
        inherited['KEEP'] = 'me'
        for parent in (SHIP, AUDIT_PARENT):
            environ = contract.apply_environment(named(parent), dict(inherited, QWEN_C2_SERVING='1'))
            self.assertEqual(environ['KEEP'], 'me')
            self.assertFalse([name for name in contract.PREFIX_NVME_NAMES if name in environ], parent)
        environ = contract.apply_environment(named(FORMAT), dict(inherited, QWEN_C2_SERVING='1'))
        self.assertEqual(environ[qwen_kv_nvme.ENV_FORMAT], '1')
        self.assertEqual(environ[qwen_kv_nvme.ENV_DEVICE], generator.IN_CONTAINER_DEVICE)
        environ = contract.apply_environment(named(TIMED), dict(inherited, QWEN_C2_SERVING='1'))
        self.assertNotIn(qwen_kv_nvme.ENV_FORMAT, environ, 'the timed arm never formats, whatever the process inherited')
        self.assertNotIn(qwen_kv_nvme.ENV_AUDIT, environ)

    def test_the_smoke_check_wants_the_tiers_install_line_from_each_twin(self):
        marker = '[PINDIAG] prefix: nvme tier on'
        for name in generator.NAMES:
            env = PROFILES[name]['env']
            with self.subTest(twin=name):
                silent = [problem for problem in c2_smoke_check.fusion_problems('', env) if 'QWEN_PREFIX_NVME=1' in problem]
                self.assertEqual(len(silent), 1, silent)
                heard = [problem for problem in c2_smoke_check.fusion_problems(marker + ' backend=host\n', env) if 'QWEN_PREFIX_NVME=1' in problem]
                self.assertEqual(heard, [])
                fell = c2_smoke_check.fusion_problems('[PINDIAG] prefix: nvme tier UNAVAILABLE, cannot open\n', env)
                self.assertTrue(any('NVMe cold tier' in problem for problem in fell))

    def test_the_kv_compat_table_classifies_every_name_of_the_twins(self):
        import test_qwen_kv_compat as compat

        names = set()
        for name in generator.NAMES:
            names |= set(PROFILES[name]['env'])
        self.assertEqual(compat.unclassified(names), [])


class CensusTests(unittest.TestCase):
    def test_no_other_profile_names_the_tier(self):
        twins = set(generator.NAMES)
        for name, profile in PROFILES.items():
            if name not in twins:
                self.assertEqual([flag for flag in qwen_kv_nvme.NAMES if flag in profile['env']], [], name)

    def test_the_twins_are_gate_only_so_no_traffic_profile_carries_the_tier(self):
        for name, profile in PROFILES.items():
            if any(flag in profile['env'] for flag in qwen_kv_nvme.NAMES):
                self.assertIs(profile.get('gate_only'), True, name)

    def test_the_tiers_names_and_the_new_descriptions_hold_nothing_private(self):
        patterns = (r'/dev/disk/by-id', r'\bnvme\d', r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d', r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', r'zot\.', r'sha256:', r'samsung', r'\.local\b',
                    r'/home/', r'/run/thatch', r'serial')
        for name in generator.NAMES:
            added = json.dumps(dict((key, value) for key, value in PROFILES[name]['env'].items() if key.startswith('QWEN_PREFIX_NVME')), sort_keys=True)
            for pattern in patterns:
                self.assertIsNone(re.search(pattern, added + PROFILES[name]['description'], re.I), (name, pattern))


if __name__ == '__main__':
    unittest.main()
