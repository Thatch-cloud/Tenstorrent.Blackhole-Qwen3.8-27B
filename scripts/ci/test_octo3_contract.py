"""tp4/octo-3: the profile contract's octo rules (serving_c2_contract.octo_problems), the waiver-run admission of serving_octo and the scrub of inherited octo names.

A gate-only profile carries the octo flags behind its gate marker. A TRAFFIC profile carries the four levers only under its owner traffic waiver, each named at its exact value, with the profile's
own QWEN_FAST_OCTO_WAIVER equal to the waiver's id (the one name by which a non-gate run is admitted); the gate instruments (the bundle's audit, the audit sampling, the lone-user padded round)
are never waivable; an inherited value is removed."""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_fusion_profiles as fusion  # noqa: E402
import make_octo3_profiles as octo3  # noqa: E402
import octo_judge  # noqa: E402
import octo_markers  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_octo as octo  # noqa: E402
from test_c2_packed_tp4_profiles import image_env  # noqa: E402
import test_octo as host_tests  # noqa: E402

PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
SHIPS = octo3.ship_names()
SHIP = SHIPS[0]
M3 = (True, 'users=8 FOUR_AS_TWO=0 PACKED_STEP=1 M3_BLOCKS=2')


def traffic(name=None):
    return dict(copy.deepcopy(PROFILES[name or SHIP]), name=name or SHIP)


def boot_problems(profile, environ=None):
    """Every boot check of serving_c2_contract.boot, in its order, for a profile (what test_fusion_ship's ContractTests run), plus the octo one."""
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
    return found


class NamesTests(unittest.TestCase):
    def test_the_contract_names_are_the_modules(self):
        self.assertEqual(contract.OCTO_SWITCH, octo.OCTO_FLAG)
        self.assertEqual(contract.OCTO_WAIVER_ENV, octo.WAIVER_ENV)
        self.assertEqual(contract.OCTO_WAIVER_ENV, octo_judge.WAIVER_FLAG)
        self.assertEqual(set(contract.OCTO_LEVERS), {octo.OCTO_FLAG, octo.DRAFT_FLAG, 'QWEN_FAST_OCTO_ATTN_BUNDLE', octo.GLUE8_FLAG})
        self.assertIn(octo.MIN_LIVE_FLAG, contract.OCTO_KNOBS)
        self.assertIn(octo.SOLO_PACKED_FLAG, contract.OCTO_GATE_ONLY)
        self.assertIn('QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT', contract.OCTO_GATE_ONLY)
        import audit_sampling

        self.assertEqual(set(audit_sampling.NAMES) & set(contract.OCTO_GATE_ONLY), set(audit_sampling.NAMES))

    def test_the_waivable_octo_levers_are_exactly_the_four_levers_and_the_min_live_knob_at_their_values(self):
        octo_part = {key: value for key, value in contract.WAIVABLE_LEVERS.items() if key.startswith('QWEN_FAST_OCTO')}
        self.assertEqual(octo_part, {'QWEN_FAST_OCTO': 'live', 'QWEN_FAST_OCTO_DRAFT': '1', 'QWEN_FAST_OCTO_ATTN_BUNDLE': '4', 'QWEN_FAST_OCTO_GLUE8': '1', 'QWEN_FAST_OCTO_MIN_LIVE': '6'})
        self.assertNotIn('alternate', octo_part.values())


class GateOnlyTests(unittest.TestCase):
    def test_every_gate_only_octo_twin_passes_every_boot_check(self):
        for name in octo3.twin_names():
            with self.subTest(profile=name):
                self.assertEqual(boot_problems(dict(PROFILES[name], name=name)), [])

    def test_a_gate_only_profile_has_no_waiver_id(self):
        profile = dict(copy.deepcopy(PROFILES[octo3.twin_names()[0]]), name='x')
        profile['env'][contract.OCTO_WAIVER_ENV] = 'whatever-2026'
        self.assertTrue(any('traffic profile' in problem for problem in contract.octo_problems(profile)))


class TrafficTests(unittest.TestCase):
    def test_the_ship_candidates_pass_every_boot_check_pending(self):
        for name in SHIPS:
            with self.subTest(profile=name):
                profile = traffic(name)
                self.assertEqual(boot_problems(profile), [])
                self.assertTrue(contract.traffic_waiver_pending(profile))
                self.assertEqual(profile['env'][contract.OCTO_WAIVER_ENV], profile[contract.TRAFFIC_WAIVER]['id'])
                for flag in contract.OCTO_LEVERS + ('QWEN_FAST_OCTO_MIN_LIVE',):
                    self.assertEqual(contract.waived_levers(profile)[flag], profile['env'][flag], flag)

    def test_an_octo_lever_without_the_waiver_is_refused_by_name(self):
        for flag in contract.OCTO_LEVERS:
            profile = traffic()
            del profile[contract.TRAFFIC_WAIVER]['levers'][flag]
            with self.subTest(flag=flag):
                text = ' '.join(contract.octo_problems(profile))
                self.assertIn('%s=' % flag, text)
                self.assertIn('needs a gate-only profile', text)

    def test_a_waiver_naming_a_lever_the_profile_does_not_set_is_stale(self):
        profile = traffic()
        del profile['env']['QWEN_FAST_OCTO_GLUE8']
        self.assertTrue(any('QWEN_FAST_OCTO_GLUE8' in problem for problem in contract.traffic_waiver_problems(profile)))

    def test_the_waiver_id_must_be_the_profiles_own(self):
        for value in (None, 'another-waiver-2026', ''):
            profile = traffic()
            if value is None:
                del profile['env'][contract.OCTO_WAIVER_ENV]
            else:
                profile['env'][contract.OCTO_WAIVER_ENV] = value
            with self.subTest(value=value):
                self.assertTrue(any(contract.OCTO_WAIVER_ENV in problem for problem in contract.octo_problems(profile)))

    def test_a_waiver_id_without_the_shape_is_refused(self):
        profile = dict(copy.deepcopy(PROFILES[fusion.NAMESPACE + 'traffic']), name='x')
        profile['env'][contract.OCTO_WAIVER_ENV] = 'octo3-fx-262k-2026-10-11'
        self.assertTrue(any('nothing under it' in problem for problem in contract.octo_problems(profile)))

    def test_the_gate_instruments_are_never_waivable_on_traffic(self):
        for name, value in (('QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT', '1'), ('QWEN_FAST_AUDIT_EVERY', '16'), ('QWEN_FAST_AUDIT_FIRST', '1'), ('QWEN_FAST_SOLO_PACKED', '1')):
            profile = traffic()
            profile['env'][name] = value
            with self.subTest(name=name):
                self.assertTrue(any('gate instrument' in problem and name in problem for problem in contract.octo_problems(profile)))

    def test_alternate_is_not_waivable(self):
        profile = traffic()
        profile['env']['QWEN_FAST_OCTO'] = 'alternate'
        self.assertTrue(contract.octo_problems(profile) or contract.traffic_waiver_problems(profile))
        self.assertTrue(contract.traffic_waiver_problems(profile))

    def test_the_min_live_knob_is_waivable_only_at_six(self):
        profile = traffic()
        profile['env']['QWEN_FAST_OCTO_MIN_LIVE'] = '7'
        self.assertTrue(contract.traffic_waiver_problems(profile) or contract.octo_problems(profile))

    def test_a_profile_without_any_octo_name_is_untouched(self):
        for name in (contract.__name__ and fusion.NAMESPACE + 'traffic', fusion.PARENT):
            self.assertEqual(contract.octo_problems(traffic(name)), [], name)

    def test_the_ship_candidate_is_the_fusion_ship_plus_exactly_the_octo_env_and_the_memory_plan(self):
        base = PROFILES[fusion.NAMESPACE + 'traffic']
        for name in SHIPS:
            profile = PROFILES[name]
            added = {key: value for key, value in profile['env'].items() if base['env'].get(key) != value}
            expected = {'QWEN_FAST_OCTO', 'QWEN_FAST_OCTO_MIN_LIVE', 'QWEN_FAST_OCTO_DRAFT', 'QWEN_FAST_OCTO_ATTN_BUNDLE', 'QWEN_FAST_OCTO_GLUE8', contract.OCTO_WAIVER_ENV,
                        'QWEN_FAST_TAP_WINDOW_ALIGNED', 'QWEN36_MAX_TOKENS_ALL_USERS'}
            if name.endswith('oneq'):
                expected.add('QWEN_FAST_SDPA_PF_ONEQ')
            with self.subTest(profile=name):
                self.assertEqual(set(added), expected)
                self.assertNotIn('gate_only', profile)
                for key in profile['env']:
                    self.assertFalse(key.endswith('_AUDIT') and profile['env'][key] == '1' and key not in base['env'], key)
                self.assertNotIn('QWEN_C2_GATE_PROFILE', profile['env'])
                self.assertEqual(profile['engine']['num-gpu-blocks-override'], 16500)
                self.assertEqual(profile['engine']['additional-config']['tt']['trace_region_size'], 640 << 20)


class ScrubTests(unittest.TestCase):
    def test_an_inherited_octo_name_is_removed_unless_the_profile_names_it(self):
        inherited = {name: 'live' for name in contract.OCTO_NAMES}
        inherited['QWEN_FAST_AUDIT_EVERY'] = '4'
        environ = dict(inherited)
        contract.apply_environment(dict(PROFILES[fusion.NAMESPACE + 'traffic'], name='x'), environ)
        self.assertEqual({name for name in environ if name in contract.OCTO_NAMES}, set())
        environ = dict(inherited)
        contract.apply_environment(traffic(), environ)
        self.assertEqual({name for name in environ if name in contract.OCTO_NAMES}, {name for name in PROFILES[SHIP]['env'] if name in contract.OCTO_NAMES})
        self.assertEqual(environ[contract.OCTO_WAIVER_ENV], PROFILES[SHIP]['env'][contract.OCTO_WAIVER_ENV])


class WaiverRunTests(unittest.TestCase):
    """serving_octo admits a non-gate run only through the profile's waiver id; the admitted line says which."""

    def environ(self, name=None):
        environ = dict(image_env())
        contract.apply_environment(traffic(name), environ)
        return environ

    def test_a_waived_traffic_profile_is_admitted_and_the_line_names_the_waiver(self):
        environ = self.environ()
        self.assertFalse(octo.gate_run(environ))
        self.assertTrue(octo.waiver_run(environ))
        log = host_tests.Lines()
        with host_tests.built():
            record = octo.octo_admission(M3, environ, log=log)
        self.assertEqual((record['mode'], record['min_live']), ('live', 6))
        admitted = [line for line in log.lines if line.startswith(octo.ADMITTED_MARKER)]
        self.assertEqual(len(admitted), 1)
        self.assertIn('(owner traffic waiver %s)' % PROFILES[SHIP]['env'][contract.OCTO_WAIVER_ENV], admitted[0])
        parsed = octo_markers.admissions('\n'.join(admitted))
        self.assertEqual(parsed[0]['waiver'], PROFILES[SHIP]['env'][contract.OCTO_WAIVER_ENV])
        self.assertEqual(len([line for line in log.lines if line.startswith(octo.UNQUALIFIED_MARKER)]), len(octo.UNQUALIFIED_ITEMS), 'the card questions are still said, one line each')

    def test_without_the_waiver_id_a_traffic_run_is_refused(self):
        environ = self.environ()
        del environ[contract.OCTO_WAIVER_ENV]
        with host_tests.built(), self.assertRaisesRegex(ValueError, 'gate run of a gate-only'):
            octo.octo_admission(M3, environ, log=host_tests.Lines())

    def test_a_gate_only_profile_is_admitted_through_its_marker_and_says_gate_only(self):
        name = octo3.twin_names()[0]
        environ = dict(image_env())
        contract.apply_environment(dict(PROFILES[name], name=name), environ)
        environ['QWEN_C2_GATE'] = '1'
        self.assertTrue(octo.gate_run(environ))
        self.assertFalse(octo.waiver_run(environ))
        log = host_tests.Lines()
        with host_tests.built():
            octo.octo_admission(M3, environ, log=log)
        admitted = [line for line in log.lines if line.startswith(octo.ADMITTED_MARKER)]
        self.assertTrue(admitted[0].endswith('(gate only)'), admitted)
        self.assertNotIn('waiver', octo_markers.admissions(admitted[0])[0])

    def test_the_gate_marker_wins_over_a_stray_waiver_id(self):
        environ = dict(image_env())
        name = octo3.twin_names()[0]
        contract.apply_environment(dict(PROFILES[name], name=name), environ)
        environ[contract.OCTO_WAIVER_ENV] = 'stray-waiver-2026'
        environ['QWEN_C2_GATE'] = '1'
        self.assertFalse(octo.waiver_run(environ), 'a gate-only profile carries no waiver id (the contract refuses one); the marker admits it')
        self.assertTrue(octo.gate_run(environ))

    def test_the_judge_reads_the_waiver_line_against_the_profile(self):
        environ = self.environ()
        log = host_tests.Lines()
        with host_tests.built():
            octo.octo_admission(M3, environ, log=log)
        text = '\n'.join(log.lines)
        found = octo_markers.scan(text)
        self.assertEqual(found['admitted'][0]['waiver'], PROFILES[SHIP]['env'][contract.OCTO_WAIVER_ENV])
        problems = []
        octo_judge.octo_problems(PROFILES[SHIP]['env'], 'live', found, {}, text)
        problems = [item for item in octo_judge.octo_problems(PROFILES[SHIP]['env'], 'live', found, {}, text) if 'waiver' in item]
        self.assertEqual(problems, [])
        wrong = dict(PROFILES[SHIP]['env'], **{contract.OCTO_WAIVER_ENV: 'another-waiver-2026'})
        self.assertTrue([item for item in octo_judge.octo_problems(wrong, 'live', found, {}, text) if 'waiver' in item])
        gate_env = {key: value for key, value in PROFILES[SHIP]['env'].items() if key != contract.OCTO_WAIVER_ENV}
        self.assertTrue([item for item in octo_judge.octo_problems(gate_env, 'live', found, {}, text) if 'carries no waiver id' in item])


if __name__ == '__main__':
    unittest.main()
