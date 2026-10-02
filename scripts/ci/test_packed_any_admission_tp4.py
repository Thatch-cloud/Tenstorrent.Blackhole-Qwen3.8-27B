"""packed_any_admission at four cards (QWEN_FAST_TP=4): the same admission with the four-card numbers and its own record.

K64j at one KV head serves 0x23 (no slice), so the modes are tail and share exactly, CB1 must hold a G8B2 0x23 combo,
the reader whose sha256 CB2b records is extent_attention_replay_tp.py, and CB2b's chip view is 1of4. The record is
packed_any_evidence_tp4.json at its own pin - a SKELETON until the card windows record the sections, so the attach is
refused except in a gate run of a gate-only profile (QWEN_C2_GATE=1 and the profile's own QWEN_C2_GATE_PROFILE=1), where each missing piece is logged as UNQUALIFIED.
The pair's admission is untouched (test_packed_any_admission, unchanged)."""

import contextlib
import copy
import hashlib
import json
import os
import tempfile
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


# The record as the four-card port first committed it (every section PENDING, no reader sha): the committed record is
# now filled in from the card windows, so the skeleton's behaviour is pinned on this fixed copy instead.
SKELETON_TP4_TEXT = '{\n "schema": "qwen-c2-packed-any-evidence/1",\n "what": "The four-card (QWEN_FAST_TP=4) record of packed_any_admission: the same three sections as packed_any_evidence.json, to be recorded from a one-card window (CB1-TP4, CB2a-TP4 and CB2b-TP4 with the four-card ChipView, 1of4) and a four-card window (the extent audit on chips 0-3). SKELETON: every section is PENDING, so packed_any_admission refuses the attach, except for a gate-only profile (QWEN_C2_GATE=1), which logs each missing piece as UNQUALIFIED and proceeds. Recording a section changes this file and EVIDENCE_TP4_SHA256 in the same commit.",\n "provenance": "Skeleton written with the four-card S2 port (docs: the S2 TP4 plan, S2T-09). Nothing here has run on a card.",\n "binary": {\n  "ttnncpp_sha256": "152951c1c0de5c9dfad2d62c295393a43b2ecf353965c55c709da7e539b975b7",\n  "note": "the same K64j binary and kernels as the pair\'s record: K64j takes the head count and geometry as arguments; whether it is exact at one KV head (0x23) is what CB1-TP4 decides"\n },\n "kernels": {\n  "dataflow/reader_decode_qwen.cpp": "adb6091878ba3f0a0805846ff56f05352610d7fe779b5ae96320c437c095db49",\n  "dataflow/reader_decode_qwen_slice.cpp": "518d8096e3cceb160eaef8ab4f0ae976ccbffd3904d31176b7f9d02828c37f8a",\n  "compute/sdpa_flash_decode_qwen.cpp": "409a1aafc3ffaaca2c6afba0e999525b7d141491f0a52e5583efa70447ee5c0e",\n  "dataflow/writer_decode_qwen_slice.cpp": "642c36f809be0f1ad1664deb405dc310dabaa32d5628710320a118eb37a6cc7a"\n },\n "sources": {},\n "sections": {\n  "CB1": {\n   "status": "PENDING",\n   "what": "K64j K1/K3 at NKV = 1: the runtime extent (0x20) at cur_pos E - 1 equals the compile-time call at capacity E, G8B2 flags 0x21, 0x23 (and G16B1 if the spike prefers it)"\n  },\n  "CB2a": {\n   "status": "PENDING",\n   "what": "K2 (bitwise), X7 and Z at one KV head with the extent flag"\n  },\n  "CB2b": {\n   "status": "PENDING",\n   "what": "The extent reader twin (extent_attention_replay_tp.py) at full scope with the 1of4 chip view, C = 131328"\n  }\n }\n}\n'


def skeleton():
    return json.loads(SKELETON_TP4_TEXT)


@contextlib.contextmanager
def skeleton_on_disk():
    """The admission reading the skeleton (at its own pin) where it reads the committed record."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'packed_any_evidence_tp4.json'
        path.write_bytes(SKELETON_TP4_TEXT.encode('utf-8'))
        with mock.patch.object(admission, 'EVIDENCE_TP4', path), \
                mock.patch.object(admission, 'EVIDENCE_TP4_SHA256', sha(path.read_bytes())):
            yield


def qualifying():
    """The four-card record filled in as the recorder writes it from one-KV-head runs: the pair's section shapes with
    kv_heads 1, the one-head combos (no q-slice), CB2b through the 1of4 view with the reader serving 0x23."""
    evidence = skeleton()
    pair = pair_tests.checked_in()
    reader = 'extent_attention_replay_tp.py'
    evidence['sources'] = {reader: sha((HERE / reader).read_bytes())}
    cb1 = copy.deepcopy(pair['sections']['CB1'])
    cb1.update(kv_heads=1, card='M', combos=[{'geometry': 'G4B3', 'flags': ['0x21', '0x23']},
                                             {'geometry': 'G8B2', 'flags': ['0x21', '0x23']}])
    evidence['sections']['CB1'] = cb1
    evidence['sections']['CB2a'] = dict(copy.deepcopy(pair['sections']['CB2a']), kv_heads=1)
    cb2b = pair_tests.cb2b_pass(evidence)
    cb2b.update(chips='1of4', served=dict(flags='0x23', rows=8, batch=2, segments=4))
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

    def test_the_committed_record_qualifies_four_cards(self):
        record = json.loads(admission.EVIDENCE_TP4.read_text(encoding='utf-8'))
        self.assertEqual(admission.evidence_problems(record, HERE, tp=4), [])
        self.assertEqual([record['sections'][name]['status'] for name in admission.SECTIONS], ['PASS'] * 3)
        self.assertEqual(admission.check_evidence(tp=4)['sections']['CB2b']['chips'], '1of4')

    def test_the_skeleton_qualifies_nothing_and_names_every_missing_piece(self):
        problems = admission.evidence_problems(skeleton(), HERE, tp=4)
        self.assertEqual(len(problems), 4, problems)
        self.assertIn('sources: no sha256 recorded for extent_attention_replay_tp.py', problems)
        for section in ('CB1', 'CB2a', 'CB2b'):
            self.assertTrue(any(problem.startswith(section + ': status PENDING') for problem in problems), section)
        with skeleton_on_disk(), self.assertRaises(admission.AdmissionRefused) as refused:
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

    def test_the_pairs_two_head_sections_copied_in_never_qualify_four_cards(self):
        # K64j is one binary at both widths and 0x23 is legal at two KV heads, so only the head count tells the pair's
        # CB1 (card B, G8B2 0x21/0x23/0x27/0x2F) and CB2a (K2 at two heads) from the one-head runs four cards need.
        pair = pair_tests.checked_in()['sections']
        cases = {
            'the pair CB1 verbatim': lambda e: e['sections'].update(CB1=copy.deepcopy(pair['CB1'])),
            'the pair CB2a verbatim': lambda e: e['sections'].update(CB2a=copy.deepcopy(pair['CB2a'])),
            'CB1 at two heads': lambda e: e['sections']['CB1'].update(kv_heads=2),
            'CB1 without kv_heads': lambda e: e['sections']['CB1'].pop('kv_heads'),
            'CB2a kv_heads as text': lambda e: e['sections']['CB2a'].update(kv_heads='1'),
            'CB1 holding a q-slice combo': lambda e: e['sections']['CB1']['combos'][1]['flags'].append('0x27'),
            'CB2b served with the slice': lambda e: e['sections']['CB2b'].update(served=dict(flags='0x27')),
            'CB2b without its served flags': lambda e: e['sections']['CB2b'].pop('served'),
        }
        for name, mutate in cases.items():
            evidence = qualifying()
            mutate(evidence)
            with self.subTest(name):
                problems = admission.evidence_problems(evidence, HERE, tp=4)
                self.assertNotEqual(problems, [], name)
        self.assertEqual(admission.evidence_problems(qualifying(), HERE, tp=4), [])

    def test_the_one_head_rule_is_four_cards_only(self):
        # the pair's own record keeps qualifying the pair: no kv_heads word, the 0x27 combos and served flags are its own
        self.assertEqual(admission.evidence_problems(pair_tests.passing(), HERE, tp=2), [])

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

    def test_the_committed_record_admits_the_traffic_profile(self):
        record = self.admit(GOOD_ENV)
        self.assertNotIn('unqualified', record)
        self.assertTrue(admission.admitted())
        self.assertTrue(self.lines[-1].startswith('[PINDIAG] packed-any admission passed: K64j'))

    def test_the_skeleton_refuses_the_attach_outside_a_gate_run(self):
        with skeleton_on_disk(), self.assertRaises(admission.AdmissionRefused) as refused:
            self.admit(GOOD_ENV)
        self.assertFalse(admission.admitted())
        self.assertEqual(len(refused.exception.problems), 4)
        self.assertTrue(all('PENDING' in problem or 'no sha256' in problem for problem in refused.exception.problems))

    def test_a_gate_run_proceeds_unqualified_with_every_piece_on_the_record(self):
        with skeleton_on_disk():
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

    # --- QWEN_FAST_M3_BLOCKS=2: eight seats on two M3 blocks (serving_runtime.m3_shape) ------------------------------------

    @staticmethod
    def blocks_m3(users, blocks=None, **environ):
        """serving_runtime.m3_shape's (met, description) over the attach's environment, as the real predicate says it."""
        import serving_runtime

        env = {'QWEN_FAST_FOUR_AS_TWO': '0', 'QWEN_FAST_PACKED_STEP': '1', **environ}
        if blocks is not None:
            env['QWEN_FAST_M3_BLOCKS'] = str(blocks)
        return serving_runtime.m3_shape(dict(scheduler_requests=users), env)

    def test_two_m3_blocks_are_admitted_on_the_committed_evidence_and_the_record_names_them(self):
        record = self.admit(dict(GOOD_ENV, QWEN_FAST_M3_BLOCKS='2'), m3=self.blocks_m3(8, 2))
        self.assertNotIn('unqualified', record)
        self.assertEqual(record['blocks'], 2)
        self.assertEqual(record['shape'], 'users=8 FOUR_AS_TWO=0 PACKED_STEP=1 M3_BLOCKS=2')
        self.assertTrue(self.lines[-1].startswith('[PINDIAG] packed-any admission passed: K64j'))
        self.assertTrue(self.lines[-1].endswith(' blocks=2'), self.lines[-1])
        # the evidence is the committed four-card record at its own, unchanged pin: each block is the qualified geometry
        self.assertIn(admission.EVIDENCE_TP4_SHA256[:16], self.lines[-1])
        self.assertEqual(sha(admission.EVIDENCE_TP4.read_bytes()), admission.EVIDENCE_TP4_SHA256)

    def test_a_one_block_attach_records_and_logs_exactly_what_it_always_did(self):
        for environ, m3 in ((GOOD_ENV, M3), (dict(GOOD_ENV, QWEN_FAST_M3_BLOCKS='1'), self.blocks_m3(4, 1))):
            with self.subTest(environ=environ.get('QWEN_FAST_M3_BLOCKS')), mock.patch.dict(admission._STATE, clear=True):
                self.lines.clear()
                record = self.admit(environ, m3=m3)
                self.assertNotIn('blocks', record)
                self.assertFalse(self.lines[-1].endswith('blocks=2'))
                self.assertEqual(record['shape'], 'users=4 FOUR_AS_TWO=0 PACKED_STEP=1')

    def test_a_gate_run_names_the_blocks_on_its_unqualified_line_too(self):
        with skeleton_on_disk():
            record = self.admit(dict(GATE, QWEN_FAST_M3_BLOCKS='2'), m3=self.blocks_m3(8, 2))
        self.assertEqual(record['blocks'], 2)
        self.assertTrue(self.lines[-1].startswith('[PINDIAG] packed-any admission passed UNQUALIFIED'))
        self.assertTrue(self.lines[-1].endswith(' blocks=2'), self.lines[-1])

    def test_every_other_shape_at_eight_users_is_refused(self):
        for name, environ, m3 in (
                ('eight users without the flag', GOOD_ENV, self.blocks_m3(8)),
                ('eight users with one block asked', dict(GOOD_ENV, QWEN_FAST_M3_BLOCKS='1'), self.blocks_m3(8, 1)),
                ('two blocks at four users', dict(GOOD_ENV, QWEN_FAST_M3_BLOCKS='2'), self.blocks_m3(4, 2)),
                ('two blocks with the 32-row pair switch on', dict(GOOD_ENV, QWEN_FAST_M3_BLOCKS='2'),
                 self.blocks_m3(8, 2, QWEN_FAST_FOUR_AS_TWO='1')),
                ('two blocks without the packed step', dict(GOOD_ENV, QWEN_FAST_M3_BLOCKS='2'),
                 self.blocks_m3(8, 2, QWEN_FAST_PACKED_STEP='0'))):
            with self.subTest(name), mock.patch.dict(admission._STATE, clear=True):
                with self.assertRaises(admission.AdmissionRefused) as refused:
                    self.admit(environ, m3=m3)
                self.assertTrue(any('64-row M3 block' in problem for problem in refused.exception.problems),
                                refused.exception.problems)
                self.assertFalse(admission.admitted())

    def test_a_malformed_block_count_is_refused_naming_the_flag(self):
        for value in ('', '0', '3', 'two'):
            with self.subTest(value=value), mock.patch.dict(admission._STATE, clear=True):
                with self.assertRaises(admission.AdmissionRefused) as refused:
                    self.admit(dict(GOOD_ENV, QWEN_FAST_M3_BLOCKS=value), m3=M3)
                self.assertTrue(any('QWEN_FAST_M3_BLOCKS must be 1 or 2' in problem
                                    for problem in refused.exception.problems), refused.exception.problems)

    def test_the_environment_check_reads_the_block_count_strictly(self):
        self.assertEqual(admission.m3_blocks({}), 1)
        self.assertEqual(admission.m3_blocks({'QWEN_FAST_M3_BLOCKS': '2'}), 2)
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_M3_BLOCKS must be 1 or 2'):
            admission.m3_blocks({'QWEN_FAST_M3_BLOCKS': '4'})
        self.assertEqual(admission.check_environment(dict(GOOD_ENV, QWEN_FAST_M3_BLOCKS='2'), self.blocks_m3(8, 2)), [])

    def test_the_block_count_the_admission_reads_is_the_one_the_attach_reads(self):
        import serving_runtime

        for value in (None, '1', '2'):
            environ = {} if value is None else {'QWEN_FAST_M3_BLOCKS': value}
            self.assertEqual(admission.m3_blocks(environ), serving_runtime.m3_blocks(environ))
        self.assertEqual(admission.M3_BLOCKS_ENV, serving_runtime.M3_BLOCKS_FLAG)


class GuardTests(unittest.TestCase):
    def test_the_pair_needs_nothing(self):
        self.assertIsNone(admission.tp_guard({}))
        self.assertIsNone(admission.tp_guard({'QWEN_C2_GATE': '1'}))

    def test_four_cards_need_the_record_or_a_gate_run(self):
        self.assertEqual(admission.tp_guard(dict(FOUR)), [], 'the committed record qualifies four cards')
        with skeleton_on_disk(), self.assertRaises(admission.AdmissionRefused):
            admission.tp_guard(dict(FOUR))
        lines = Lines()
        with skeleton_on_disk():
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
            with skeleton_on_disk(), mock.patch.dict(os.environ, dict(FOUR), clear=True):
                with self.assertRaises(admission.AdmissionRefused):
                    factory.attach_source_check(Path('/four'), qualify=good, log=lambda *a: None)
                self.assertNotIn(str(Path('/four')), factory._ATTACH_QUALIFICATION)
            with skeleton_on_disk(), mock.patch.dict(os.environ, dict(FOUR, QWEN_C2_GATE='1', QWEN_C2_GATE_PROFILE='1'), clear=True):
                factory.attach_source_check(Path('/four'), qualify=good, log=Lines())


if __name__ == '__main__':
    unittest.main()
