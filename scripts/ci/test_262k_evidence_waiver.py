"""The 262k evidence waiver: QWEN_FAST_262K_EVIDENCE_WAIVER=1, gate only.

The quad-first 262k window runs before the one-card evidence records exist (ordered_writer_evidence_tp4.json and
packed_any_evidence_tp4_262144.json are PENDING skeletons). The waiver lets page width 4,096 and the admission at capacity 262,144 proceed
without them ONLY in a gate run of a gate-only profile (QWEN_C2_GATE=1 and the profile's own QWEN_C2_GATE_PROFILE=1), refuses the flag by
name anywhere else, logs '262k evidence WAIVED (gate-only): ...' once per process, and leaves every path byte-identical when it is unset."""

import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission as admission  # noqa: E402
import page_width_tp4 as pw  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import test_packed_any_admission as pair_tests  # noqa: E402
import test_packed_any_admission_tp4 as quad_tests  # noqa: E402

FLAG = 'QWEN_FAST_262K_EVIDENCE_WAIVER'
M3 = pair_tests.M3
WIDE = dict(quad_tests.GOOD_ENV, QWEN_FAST_MAX_POSITION='262144')
TRAFFIC_ENV = dict(WIDE, **{FLAG: '1'})                                  # a traffic profile's process: no gate switch, no marker
GATE_SWITCH_ONLY = dict(WIDE, QWEN_C2_GATE='1', **{FLAG: '1'})          # QWEN_C2_GATE=1 alone cannot tell a gate profile from a traffic one
GATE_ENV = dict(WIDE, QWEN_C2_GATE='1', QWEN_C2_GATE_PROFILE='1')
WAIVED_ENV = dict(GATE_ENV, **{FLAG: '1'})
E1_PENDING = (False, ['status PENDING, not PASS'])
PROFILES = HERE / 'qwen_c2_profiles.json'


class Lines(list):
    def __call__(self, template, *values):
        self.append(template.format(*values))


def profiles():
    return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']


class Fresh(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(admission._STATE, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        saved = list(pw._WAIVER_LOGGED)
        pw._WAIVER_LOGGED[:] = []
        self.addCleanup(lambda: pw._WAIVER_LOGGED.__setitem__(slice(None), saved))
        self.lines = Lines()
        self.runtime = dict(binaries={'build_Release/lib/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256,
                                      'build_Release/ttnn/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256})

    def admit(self, environ, m3=M3, runtime=None):
        """The real 262k and E1 reads: both are the shipped PENDING skeletons."""
        with mock.patch.object(admission, 'check_runtime', side_effect=lambda root, binaries: runtime or self.runtime), \
                mock.patch.dict(os.environ, environ, clear=True):
            return admission.admit('/opt/tt-metal', m3=m3, environ=dict(environ), log=self.lines)


class SwitchTests(unittest.TestCase):
    def test_unset_zero_and_empty_are_off(self):
        for environ in ({}, {FLAG: ''}, {FLAG: '0'}, dict(TRAFFIC_ENV, **{FLAG: '0'})):
            self.assertIs(pw.waiver_active(environ), False, environ)

    def test_on_only_in_a_gate_run_of_a_gate_only_profile(self):
        self.assertIs(pw.waiver_active(WAIVED_ENV), True)

    def test_refused_by_name_without_the_gate_switch_or_without_the_profile_marker(self):
        for environ in (TRAFFIC_ENV, GATE_SWITCH_ONLY, dict(WIDE, QWEN_C2_GATE_PROFILE='1', **{FLAG: '1'}),
                        dict(WAIVED_ENV, QWEN_C2_GATE='0')):
            with self.subTest(environ=sorted(k for k in environ if k.startswith('QWEN_C2') or k == FLAG)):
                with self.assertRaises(pw.WaiverRefused) as caught:
                    pw.waiver_active(environ)
                self.assertIn(FLAG, str(caught.exception))
                self.assertIn('gate-only profile', str(caught.exception))

    def test_any_value_but_zero_or_one_is_refused(self):
        for value in ('2', 'true', 'yes', ' 1'):
            with self.assertRaises(pw.WaiverRefused):
                pw.waiver_active(dict(WAIVED_ENV, **{FLAG: value}))


class PageWidthTests(Fresh):
    def test_flag_off_width_4096_is_exactly_as_before(self):
        for environ in (WIDE, GATE_ENV, dict(WIDE, **{FLAG: '0'})):
            self.assertFalse(pw.admitted(4096, environ), environ)
        for width in (2052, 1024, 4100, 4104, 0, None, '4096'):
            for environ in (WIDE, GATE_ENV, WAIVED_ENV):
                self.assertEqual(pw.admitted(width, environ), pw.admitted(width, dict(environ, **{FLAG: '0'})), (width, environ))
        self.assertEqual(pw._WAIVER_LOGGED, [])

    def test_waived_width_4096_is_admitted_in_a_gate_run_and_logs_once_loudly(self):
        with mock.patch.object(pw, '_log') as log:
            self.assertTrue(pw.admitted(4096, WAIVED_ENV))
            self.assertTrue(pw.admitted(4096, WAIVED_ENV))
        self.assertEqual(log.call_count, 1)
        text = log.call_args[0][0].format(*log.call_args[0][1:])
        self.assertTrue(text.startswith('262k evidence WAIVED (gate-only): '), text)
        self.assertIn('UNQUALIFIED', text)

    def test_the_waiver_never_admits_another_width_or_the_pair(self):
        for width in (4100, 4104, 4092, 8192, 0, '4096'):
            self.assertFalse(pw.admitted(width, WAIVED_ENV), width)
        pair = dict(WAIVED_ENV, QWEN_FAST_TP='2')
        self.assertFalse(pw.admitted(4096, pair))

    def test_a_traffic_process_with_the_flag_is_refused_not_silently_admitted(self):
        for environ in (TRAFFIC_ENV, GATE_SWITCH_ONLY):
            with self.assertRaises(pw.WaiverRefused):
                pw.admitted(4096, environ)

    def test_a_passing_record_admits_without_logging_a_waiver(self):
        with mock.patch.object(pw, 'evidence_state', return_value=(True, [])), mock.patch.object(pw, '_log') as log:
            self.assertTrue(pw.admitted(4096, WAIVED_ENV))
        log.assert_not_called()


class AdmitTests(Fresh):
    def test_flag_off_a_262k_gate_run_is_refused_exactly_as_before(self):
        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.admit(GATE_ENV)
        self.assertTrue(any('capacity 262144' in problem for problem in caught.exception.problems), caught.exception.problems)
        self.assertFalse([line for line in self.lines if 'WAIVED' in line or 'UNQUALIFIED' in line], self.lines)
        self.assertFalse(admission.admitted())

    def test_waived_a_262k_gate_run_attaches_unqualified_with_one_loud_line(self):
        record = self.admit(WAIVED_ENV)
        self.assertTrue(admission.admitted())
        self.assertEqual(record['capacity'], 262144)
        self.assertIsNone(record['evidence'])
        self.assertTrue(record['waived'])
        waived = [line for line in self.lines if line.startswith('262k evidence WAIVED (gate-only): ')]
        self.assertEqual(len(waived), 1, self.lines)
        passed = [line for line in self.lines if 'passed UNQUALIFIED' in line]
        self.assertEqual(len(passed), 1, self.lines)
        self.assertIn('(262k waiver, gate only)', passed[0])
        self.assertIn('capacity=262144', passed[0])
        self.assertFalse([line for line in self.lines if line.startswith(admission.MARKER + ' passed:')], 'never the qualified line')
        self.assertEqual(len([line for line in self.lines if admission.UNQUALIFIED_MARKER in line]), len(record['waived']))

    def test_the_log_is_once_per_process_across_the_page_width_and_the_admission(self):
        with mock.patch.object(pw, '_log', side_effect=lambda template, *values: self.lines(template, *values)):
            self.assertTrue(pw.admitted(4096, WAIVED_ENV))
        self.admit(WAIVED_ENV)
        self.assertEqual(len([line for line in self.lines if '262k evidence WAIVED (gate-only)' in line]), 1, self.lines)

    def test_a_traffic_process_with_the_flag_is_refused_by_name_at_attach(self):
        for environ in (TRAFFIC_ENV, GATE_SWITCH_ONLY):
            admission._STATE.clear()
            with self.assertRaises(admission.AdmissionRefused) as caught:
                self.admit(environ)
            self.assertTrue(any(FLAG in problem for problem in caught.exception.problems), caught.exception.problems)
            self.assertFalse([line for line in self.lines if 'WAIVED' in line])
            self.assertFalse(admission.admitted())

    def test_the_flag_is_refused_at_131k_too(self):
        environ = dict(quad_tests.GOOD_ENV, **{FLAG: '1'})
        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.admit(environ)
        self.assertTrue(any(FLAG in problem for problem in caught.exception.problems))

    def test_every_other_condition_still_refuses_under_the_waiver(self):
        wrong = dict(binaries={'build_Release/lib/_ttnncpp.so': 'f' * 64})
        with mock.patch.object(admission, 'check_runtime', side_effect=admission.AdmissionRefused('runtime', ['runtime: wrong binary'])), \
                mock.patch.dict(os.environ, WAIVED_ENV, clear=True):
            with self.assertRaises(admission.AdmissionRefused) as caught:
                admission.admit('/opt/tt-metal', m3=M3, environ=dict(WAIVED_ENV), log=self.lines)
        self.assertIn('runtime: wrong binary', caught.exception.problems)
        self.assertFalse([line for line in self.lines if 'passed' in line])
        self.assertIsNotNone(wrong)
        for broken in (dict(WAIVED_ENV, QWEN_FAST_MAX_POSITION='200000'), dict(WAIVED_ENV, QWEN_FAST_EXTENT_REPLAY='0')):
            admission._STATE.clear()
            with self.assertRaises(admission.AdmissionRefused):
                self.admit(broken)

    def test_flag_off_the_131k_attach_is_identical_with_the_flag_zero_or_unset(self):
        lines = []
        for environ in (quad_tests.GOOD_ENV, dict(quad_tests.GOOD_ENV, **{FLAG: '0'}), dict(quad_tests.GOOD_ENV, **{FLAG: ''})):
            admission._STATE.clear()
            self.lines = Lines()
            with mock.patch.object(admission, 'check_runtime', side_effect=lambda root, binaries: self.runtime), \
                    mock.patch.dict(os.environ, environ, clear=True):
                try:
                    record = admission.admit('/opt/tt-metal', m3=M3, environ=dict(environ), log=self.lines)
                except admission.AdmissionRefused as refusal:
                    record = list(refusal.problems)
            lines.append((record, list(self.lines)))
        self.assertEqual(lines[0], lines[1])
        self.assertEqual(lines[0], lines[2])

    def test_the_pool_is_still_held_to_the_admitted_capacity_under_the_waiver(self):
        self.admit(WAIVED_ENV)
        statistics = [dict(chip=0, largest_free=4_000_000_000)] * 4

        def pool(width):
            return mock.Mock(extent_replay=True, page_width=width, dram_statistics=mock.Mock(return_value=statistics))

        self.assertIs(admission.admit_pool(pool(4096), log=Lines()), statistics)
        with self.assertRaises(admission.AdmissionRefused):
            admission.admit_pool(pool(2052), log=Lines())


class GuardTests(Fresh):
    def test_tp_guard_under_the_waiver_returns_the_problems_and_logs_once(self):
        with mock.patch.object(admission, 'check_evidence', return_value={}):
            problems = admission.tp_guard(WAIVED_ENV, log=self.lines)
        self.assertTrue(problems)
        self.assertEqual(len([line for line in self.lines if '262k evidence WAIVED (gate-only)' in line]), 1, self.lines)

    def test_tp_guard_without_the_flag_still_refuses_and_with_it_in_traffic_refuses_by_name(self):
        with mock.patch.object(admission, 'check_evidence', return_value={}):
            with self.assertRaises(admission.AdmissionRefused):
                admission.tp_guard(GATE_ENV, log=self.lines)
            with self.assertRaises(admission.AdmissionRefused) as caught:
                admission.tp_guard(TRAFFIC_ENV, log=self.lines)
        self.assertIn(FLAG, str(caught.exception))


class ProfileContractTests(unittest.TestCase):
    def carrying(self):
        return sorted(name for name, entry in profiles().items() if FLAG in (entry.get('env') or {}))

    def test_exactly_the_gate_only_262k_profiles_carry_the_flag_and_never_a_traffic_profile(self):
        self.assertEqual(self.carrying(), ['c2-packed-tp4-262k-gate', 'c2-packed-tp4-8x262k-best', 'c2-packed-tp4-8x262k-best-audit',
                                           'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-diag-strace',
                                           'c2-packed-tp4-8x262k-gate', 'c2-packed-tp4-8x262k-time-gate'])  # the best arms: tp4/262k8
        self.assertNotIn(FLAG, profiles()['c2-packed-tp4-8x262k-ship']['env'])
        for name, entry in profiles().items():
            if entry.get('gate_only') is not True:
                self.assertNotIn(FLAG, entry.get('env') or {}, name)
        self.assertNotIn(FLAG, profiles()['c2-packed-tp4-8x262k']['env'])
        for name in self.carrying():
            entry = profiles()[name]
            self.assertIs(entry['gate_only'], True, name)
            self.assertEqual(entry['env'][FLAG], '1')
            self.assertEqual(entry['env']['QWEN_C2_GATE_PROFILE'], '1', name)
            self.assertEqual(entry['env']['QWEN_FAST_MAX_POSITION'], '262144', name)

    def test_the_contract_accepts_each_carrier_in_a_gate_run_and_refuses_it_outside_one(self):
        for name in self.carrying():
            entry = dict(profiles()[name], name=name)
            self.assertEqual(contract.waiver_problems(entry, {'QWEN_C2_GATE': '1'}), [], name)
            self.assertTrue(contract.waiver_problems(entry, {}), name)

    def test_a_traffic_profile_cannot_set_the_flag_the_contract_refuses_every_way(self):
        traffic = dict(copy.deepcopy(profiles()['c2-packed-tp4-8x262k']), name='c2-packed-tp4-8x262k')
        self.assertEqual(contract.waiver_problems(traffic, {'QWEN_C2_GATE': '1'}), [])
        traffic['env'][FLAG] = '1'
        problems = contract.waiver_problems(traffic, {'QWEN_C2_GATE': '1'})
        self.assertTrue(any('is not gate only' in problem for problem in problems), problems)
        self.assertTrue(any('marker' in problem for problem in problems), problems)
        self.assertTrue(contract.waiver_problems(traffic, {}))
        # the process's own environment is read too (docker -e on a traffic profile)
        plain = dict(copy.deepcopy(profiles()['c2-packed-tp4-8x262k']), name='c2-packed-tp4-8x262k')
        self.assertTrue(contract.waiver_problems(plain, {FLAG: '1', 'QWEN_C2_GATE': '1'}))
        # a gate-only profile without the marker is refused too
        carrier = dict(copy.deepcopy(profiles()['c2-packed-tp4-8x262k-gate']), name='x')
        del carrier['env']['QWEN_C2_GATE_PROFILE']
        self.assertTrue(any('marker' in problem for problem in contract.waiver_problems(carrier, {'QWEN_C2_GATE': '1'})))
        carrier['env']['QWEN_C2_GATE_PROFILE'] = '1'
        carrier['env'][FLAG] = '2'
        self.assertTrue(any('must be 1 or unset' in problem for problem in contract.waiver_problems(carrier, {'QWEN_C2_GATE': '1'})))

    def test_flag_unset_the_contract_finds_nothing_in_any_profile(self):
        for name, entry in profiles().items():
            if FLAG not in (entry.get('env') or {}):
                self.assertEqual(contract.waiver_problems(dict(entry, name=name), {}), [], name)

    def test_the_two_modules_name_the_same_variables(self):
        self.assertEqual(contract.EVIDENCE_WAIVER, pw.WAIVER_ENV)
        self.assertEqual(contract.GATE_SWITCH, pw.WAIVER_GATE_ENV)
        self.assertEqual(contract.GATE_PROFILE_MARKER, pw.WAIVER_GATE_PROFILE_ENV)
        self.assertEqual(admission.GATE_ENV, pw.WAIVER_GATE_ENV)
        self.assertEqual(admission.GATE_PROFILE_ENV, pw.WAIVER_GATE_PROFILE_ENV)

    def test_the_gate_profiles_that_read_correctness_have_every_audit_on_and_the_timed_ones_none(self):
        found = profiles()
        for name in ('c2-packed-tp4-8x262k-gate', 'c2-packed-tp4-262k-gate'):
            env = found[name]['env']
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT'],
                              env['QWEN_FAST_GDN_PREFILL_CONV_AUDIT'], env['QWEN_FAST_EXTENT_REPLAY']), ('1', '1', '4', '1'), name)
            self.assertNotIn('QWEN_FAST_EXTENT_AUDIT', env, 'a gate-only knob: the gate arms add it, never a profile')
        for name in ('c2-packed-tp4-8x262k-time-gate', 'c2-packed-tp4-8x262k-diag-strace', 'c2-packed-tp4-8x262k'):
            env = found[name]['env']
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'), name)


if __name__ == '__main__':
    unittest.main()
