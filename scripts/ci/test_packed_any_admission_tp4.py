"""packed_any_admission at four cards (QWEN_FAST_TP=4): the same admission with the four-card numbers and its own record.

K64j at one KV head serves 0x23 (no slice), so the modes are tail and share exactly, CB1 must hold a G8B2 0x23 combo,
the reader whose sha256 CB2b records is extent_attention_replay_tp.py, and CB2b's chip view is 1of4. The record is
packed_any_evidence_tp4.json at its own pin - a SKELETON until the card windows record the sections, so the attach is
refused except in a gate run of a gate-only profile (QWEN_C2_GATE=1 and the profile's own QWEN_C2_GATE_PROFILE=1), where each missing piece is logged as UNQUALIFIED.
The pair's admission is untouched (test_packed_any_admission, unchanged)."""

import copy
import hashlib
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
import test_packed_any_admission as pair_tests  # noqa: E402

M3 = pair_tests.M3
FOUR = {'QWEN_FAST_TP': '4', 'QWEN_FAST_SDPA_MODES': 'tail,share'}
GOOD_ENV = dict(pair_tests.GOOD_ENV, **FOUR)
GATE = dict(GOOD_ENV, QWEN_C2_GATE='1', QWEN_C2_GATE_PROFILE='1')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def skeleton():
    return json.loads(admission.EVIDENCE_TP4.read_text(encoding='utf-8'))


def qualifying():
    """The four-card record filled in as the card windows would: the pair's sections re-recorded at four cards."""
    evidence = skeleton()
    pair = pair_tests.checked_in()
    reader = 'extent_attention_replay_tp.py'
    evidence['sources'] = {reader: sha((HERE / reader).read_bytes())}
    evidence['sections']['CB1'] = copy.deepcopy(pair['sections']['CB1'])
    evidence['sections']['CB2a'] = copy.deepcopy(pair['sections']['CB2a'])
    cb2b = pair_tests.cb2b_pass(evidence)
    cb2b['chips'] = '1of4'
    evidence['sections']['CB2b'] = cb2b
    return evidence


class Lines(list):
    def __call__(self, template, *values):
        self.append(template.format(*values))


class EnvironmentTests(unittest.TestCase):
    def test_the_pairs_environment_is_unchanged(self):
        self.assertEqual(admission.check_environment(dict(pair_tests.GOOD_ENV), M3), [])

    def test_four_cards_serve_tail_and_share_only(self):
        self.assertEqual(admission.check_environment(dict(GOOD_ENV), M3), [])
        slice_ = admission.check_environment(dict(GOOD_ENV, QWEN_FAST_SDPA_MODES='tail,share,slice'), M3)
        self.assertEqual(len(slice_), 1)
        self.assertIn('names slice', slice_[0])
        self.assertIn('0x23', slice_[0])
        missing = admission.check_environment(dict(GOOD_ENV, QWEN_FAST_SDPA_MODES='tail'), M3)
        self.assertEqual(len(missing), 1)
        self.assertIn('lacks share', missing[0])
        self.assertIn('0x23 (tail, share and extent)', missing[0])

    def test_the_pair_still_refuses_a_missing_slice(self):
        problems = admission.check_environment(dict(pair_tests.GOOD_ENV, QWEN_FAST_SDPA_MODES='tail,share'), M3)
        self.assertEqual(len(problems), 1)
        self.assertIn('lacks slice', problems[0])
        self.assertIn('0x27', problems[0])


class RecordTests(unittest.TestCase):
    def test_the_skeleton_is_the_pinned_file_and_lf(self):
        data = admission.EVIDENCE_TP4.read_bytes()
        self.assertEqual(sha(data), admission.EVIDENCE_TP4_SHA256,
                         'packed_any_evidence_tp4.json changed: re-pin EVIDENCE_TP4_SHA256 in the same commit')
        self.assertNotIn(b'\r', data)

    def test_the_skeleton_qualifies_nothing_and_names_every_missing_piece(self):
        problems = admission.evidence_problems(skeleton(), HERE, tp=4)
        self.assertEqual(len(problems), 4, problems)
        self.assertIn('sources: no sha256 recorded for extent_attention_replay_tp.py', problems)
        for section in ('CB1', 'CB2a', 'CB2b'):
            self.assertTrue(any(problem.startswith(section + ': status PENDING') for problem in problems), section)
        with self.assertRaises(admission.AdmissionRefused) as refused:
            admission.check_evidence(tp=4)
        self.assertEqual(len(refused.exception.problems), 4)

    def test_the_skeleton_names_the_same_k64j_binary_and_kernels(self):
        evidence = skeleton()
        self.assertEqual(evidence['binary']['ttnncpp_sha256'], admission.K64J_TTNNCPP_SHA256)
        self.assertEqual(evidence['kernels'], admission.K64J_KERNELS)

    def test_a_record_of_the_four_card_sections_qualifies(self):
        self.assertEqual(admission.evidence_problems(qualifying(), HERE, tp=4), [])

    def test_each_four_card_condition_refuses_alone(self):
        cases = {
            'the pair chip view': lambda e: e['sections']['CB2b'].update(chips='1of2'),
            'a stale reader sha': lambda e: e['sources'].update({'extent_attention_replay_tp.py': '0' * 64}),
            'the pinned reader recorded instead': lambda e: e.update(sources={
                'extent_attention_replay.py': sha((HERE / 'extent_attention_replay.py').read_bytes())}),
            'CB1 without the 0x23 combo': lambda e: e['sections']['CB1'].update(combos=[
                {'geometry': 'G8B2', 'flags': ['0x21', '0x27']}]),
            'CB2b sources on the pair reader': lambda e: e['sections']['CB2b'].update(sources={
                'extent_attention_replay.py': e['sources'].get('extent_attention_replay_tp.py', '')}),
        }
        for name, mutate in cases.items():
            evidence = qualifying()
            mutate(evidence)
            with self.subTest(name):
                self.assertNotEqual(admission.evidence_problems(evidence, HERE, tp=4), [], name)

    def test_the_pairs_record_does_not_qualify_four_cards_and_the_reverse(self):
        self.assertNotEqual(admission.evidence_problems(pair_tests.checked_in(), HERE, tp=4), [])
        self.assertNotEqual(admission.evidence_problems(qualifying(), HERE, tp=2), [])

    def test_the_width_selects_the_file_and_the_pin(self):
        self.assertEqual((admission.evidence_path(2), admission.evidence_pin(2)),
                         (admission.EVIDENCE, admission.EVIDENCE_SHA256))
        self.assertEqual((admission.evidence_path(4), admission.evidence_pin(4)),
                         (admission.EVIDENCE_TP4, admission.EVIDENCE_TP4_SHA256))
        self.assertEqual(admission.width({}), 2)
        self.assertEqual(admission.width({'QWEN_FAST_TP': '4'}), 4)
        with self.assertRaises(ValueError):
            admission.width({'QWEN_FAST_TP': '3'})


class AdmitTests(unittest.TestCase):
    def setUp(self):
        self.state = mock.patch.dict(admission._STATE, clear=True)
        self.state.start()
        self.addCleanup(self.state.stop)
        self.lines = Lines()
        self.runtime = dict(binaries={'build_Release/lib/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256,
                                      'build_Release/ttnn/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256})

    def admit(self, environ, evidence=None, runtime=None, m3=M3):
        with mock.patch.object(admission, 'check_runtime', side_effect=runtime or (lambda root, binaries: self.runtime)):
            with mock.patch.dict(os.environ, environ, clear=True):
                if evidence is None:
                    return admission.admit('/opt/tt-metal', m3=m3, environ=dict(environ), log=self.lines)
                with mock.patch.object(admission, 'check_evidence', side_effect=evidence):
                    return admission.admit('/opt/tt-metal', m3=m3, environ=dict(environ), log=self.lines)

    def test_a_qualifying_four_card_record_is_admitted_as_qualified(self):
        record = self.admit(GOOD_ENV, evidence=lambda path, tp=None: qualifying())
        self.assertNotIn('unqualified', record)
        self.assertTrue(self.lines[-1].startswith('[PINDIAG] packed-any admission passed: K64j'))
        self.assertIn(admission.EVIDENCE_TP4_SHA256[:16], self.lines[-1])

    def test_the_skeleton_refuses_the_attach_outside_a_gate_run(self):
        with self.assertRaises(admission.AdmissionRefused) as refused:
            self.admit(GOOD_ENV)
        self.assertFalse(admission.admitted())
        self.assertEqual(len(refused.exception.problems), 4)
        self.assertTrue(all('PENDING' in problem or 'no sha256' in problem for problem in refused.exception.problems))

    def test_a_gate_run_proceeds_unqualified_with_every_piece_on_the_record(self):
        record = self.admit(GATE)
        self.assertEqual(len(record['unqualified']), 4)
        self.assertIsNone(record['evidence'])
        self.assertTrue(admission.admitted())
        unqualified = [line for line in self.lines if admission.UNQUALIFIED_MARKER in line]
        self.assertEqual(len(unqualified), 4)
        self.assertTrue(all(len(line) < pair_tests.LOG_LINE_LIMIT for line in self.lines), self.lines)
        self.assertTrue(self.lines[-1].startswith('[PINDIAG] packed-any admission passed UNQUALIFIED'))

    def test_a_gate_run_still_refuses_everything_that_is_not_the_evidence(self):
        for name, environ, runtime, m3 in (
                ('modes with the slice', dict(GATE, QWEN_FAST_SDPA_MODES='tail,share,slice'), None, M3),
                ('the shape', GATE, None, (False, 'users=2')),
                ('a binary that is not K64j', GATE, lambda root, binaries: (_ for _ in ()).throw(
                    admission.AdmissionRefused('runtime', ['runtime: not K64j'])), M3),
                ('the runtime binary env', dict(GATE, **{admission.RUNTIME_BINARY_ENV: 'f' * 64}), None, M3)):
            with self.subTest(name), mock.patch.dict(admission._STATE, clear=True):
                with self.assertRaises(admission.AdmissionRefused):
                    self.admit(environ, runtime=runtime, m3=m3)
                self.assertFalse(admission.admitted())

    def test_the_gate_switch_never_waives_the_pairs_evidence(self):
        pair_gate = dict(pair_tests.GOOD_ENV, QWEN_C2_GATE='1')
        bad = pair_tests.checked_in()
        bad['sections']['CB2b']['status'] = 'PENDING'
        with self.assertRaises(admission.AdmissionRefused):
            self.admit(pair_gate, evidence=lambda path, tp=None: (_ for _ in ()).throw(
                admission.AdmissionRefused('evidence', ['evidence: CB2b: status PENDING'])))
        self.assertFalse(admission.unqualified_allowed(pair_gate))
        self.assertTrue(admission.unqualified_allowed(GATE))
        self.assertFalse(admission.unqualified_allowed(GOOD_ENV))
        # a traffic profile run as a gate (the workflow sets QWEN_C2_GATE=1 for every gate boot) is not a gate-only profile
        self.assertFalse(admission.unqualified_allowed(dict(GOOD_ENV, QWEN_C2_GATE='1')))


class GuardTests(unittest.TestCase):
    def test_the_pair_needs_nothing(self):
        self.assertIsNone(admission.tp_guard({}))
        self.assertIsNone(admission.tp_guard({'QWEN_C2_GATE': '1'}))

    def test_four_cards_need_the_record_or_a_gate_run(self):
        with self.assertRaises(admission.AdmissionRefused):
            admission.tp_guard(dict(FOUR))
        lines = Lines()
        problems = admission.tp_guard(dict(FOUR, QWEN_C2_GATE='1', QWEN_C2_GATE_PROFILE='1'), log=lines)
        self.assertEqual(len(problems), 4)
        self.assertEqual(len(lines), 4)
        with mock.patch.object(admission, 'check_evidence', return_value=qualifying()):
            self.assertEqual(admission.tp_guard(dict(FOUR)), [])

    def test_the_attach_source_check_runs_the_guard_at_four_cards_only(self):
        import serving_request_factory as factory

        good = lambda directory: {'report_sha256': 'x'}
        with mock.patch.dict(factory._ATTACH_QUALIFICATION, clear=True):
            with mock.patch.dict(os.environ, {}, clear=True):
                factory.attach_source_check(Path('/pair'), qualify=good, log=lambda *a: None)
            with mock.patch.dict(os.environ, dict(FOUR), clear=True):
                with self.assertRaises(admission.AdmissionRefused):
                    factory.attach_source_check(Path('/four'), qualify=good, log=lambda *a: None)
                self.assertNotIn(str(Path('/four')), factory._ATTACH_QUALIFICATION)
            with mock.patch.dict(os.environ, dict(FOUR, QWEN_C2_GATE='1', QWEN_C2_GATE_PROFILE='1'), clear=True):
                factory.attach_source_check(Path('/four'), qualify=good, log=Lines())


if __name__ == '__main__':
    unittest.main()
