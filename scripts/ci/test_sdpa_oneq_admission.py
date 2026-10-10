"""packed_any_admission with the K64j-OQ binary (tp4/prefill-sdpa): the admission serves on K64j (as always) or on K64j-OQ, named by
QWEN_FAST_RUNTIME_BINARY_SHA256; every evidence record was made on K64j, so a K64j-OQ boot borrows K64j's record in a GATE boot only
(QWEN_C2_GATE=1) and a traffic boot needs a record of its own binary. Holds: the K64j path unchanged, the runtime check of K64j-OQ (its extra
literals, no K64j bytes under its name), the evidence rule on the real four-card records at both capacities, the passed and equivalence lines, the
refusals by name, and the recorder's --binary (record_packed_any_evidence_tp4.py).

Run from scripts/ci:  python -B -m unittest test_sdpa_oneq_admission
"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission as admission  # noqa: E402
import record_packed_any_evidence_tp4 as recorder  # noqa: E402
from test_packed_any_admission import FakeRuntime, Lines, M3, sha  # noqa: E402

K64J = admission.K64J_TTNNCPP_SHA256
OQ = admission.K64J_OQ_TTNNCPP_SHA256
ENV = admission.RUNTIME_BINARY_ENV
GATE = {admission.GATE_ENV: '1'}
TP4_ENV = {
    admission.FLAG: '1',
    'QWEN_FAST_ANY_REQUEST': '1',
    'QWEN_FAST_REPLAY_GROUP_ROWS': '8',
    'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1',
    'QWEN_FAST_SDPA_MODES': 'tail,share',
    'QWEN_FAST_TP': '4',
}


def real_record(capacity=admission.CAPACITY_131K):
    return json.loads(admission.evidence_path(4, capacity).read_text(encoding='utf-8'))


class ConstantsTests(unittest.TestCase):
    def test_the_two_served_binaries_and_the_reviewed_borrowing(self):
        self.assertEqual(OQ, '2b81e28f017ccf0ab50028fbae5eb31dd61cfd8a3159233a1ed785712d024a57')
        self.assertEqual(admission.SERVED_BINARIES, {K64J: 'K64j', OQ: 'K64j-OQ'})
        self.assertEqual(admission.EVIDENCE_BORROWS, {OQ: K64J})
        self.assertNotEqual(OQ, K64J)

    def test_the_oneq_literals_are_the_patch_s_two_strings(self):
        sys.path.insert(0, str(HERE.parents[1] / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_oneq'))
        import apply_factory_ps as ps

        wanted = {ps.FATAL_TEXT.split(':')[0].encode(), ps.LOG_TEXT.split(' q_chunks=')[0].encode() + b' q_chunks='}
        self.assertEqual(set(admission.ONEQ_BINARY_LITERALS), wanted)

    def test_served_binary_names_one_of_two_and_defaults_to_k64j(self):
        self.assertEqual(admission.served_binary({}), K64J)
        self.assertEqual(admission.served_binary({ENV: K64J}), K64J)
        self.assertEqual(admission.served_binary({ENV: OQ}), OQ)
        self.assertEqual(admission.served_binary({ENV: OQ.upper()}), OQ)
        self.assertEqual(admission.served_binary({ENV: 'f' * 64}), K64J)
        self.assertEqual((admission.binary_label(K64J), admission.binary_label(OQ)), ('K64j', 'K64j-OQ'))

    def test_equivalence_is_one_pair_in_a_gate_boot_only(self):
        self.assertTrue(admission.equivalent(K64J, OQ, GATE))
        self.assertFalse(admission.equivalent(K64J, OQ, {}))
        self.assertFalse(admission.equivalent(K64J, OQ, {admission.GATE_ENV: '0'}))
        self.assertFalse(admission.equivalent(OQ, K64J, GATE), 'K64j never borrows from K64j-OQ')
        self.assertFalse(admission.equivalent(K64J, K64J, GATE))
        self.assertFalse(admission.equivalent('f' * 64, OQ, GATE))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='oneq-runtime-'))
        self.addCleanup(__import__('shutil').rmtree, str(self.tmp), True)

    def oq_runtime(self, literals):
        runtime = FakeRuntime(self.tmp, literals)
        runtime.patches.append(mock.patch.object(admission, 'K64J_OQ_TTNNCPP_SHA256', sha(runtime.binary)))
        runtime.patches.append(mock.patch.object(admission, 'SERVED_BINARIES', {K64J: 'K64j', sha(runtime.binary): 'K64j-OQ'}))
        return runtime

    def test_the_k64j_call_is_the_two_argument_call_it_always_was(self):
        with FakeRuntime(self.tmp) as runtime:
            record = admission.check_runtime(self.tmp)
            self.assertEqual(set(record['binaries'].values()), {sha(runtime.binary)})
            self.assertEqual(record['literals'], [literal.decode() for literal in admission.BINARY_LITERALS])

    def test_k64j_oq_passes_with_its_extra_literals_and_names_them_in_the_record(self):
        with self.oq_runtime(admission.BINARY_LITERALS + admission.ONEQ_BINARY_LITERALS) as runtime:
            record = admission.check_runtime(self.tmp, served=sha(runtime.binary))
            self.assertEqual(set(record['binaries'].values()), {sha(runtime.binary)})
            self.assertEqual(record['literals'], [literal.decode() for literal in admission.BINARY_LITERALS + admission.ONEQ_BINARY_LITERALS])

    def test_k64j_oq_without_the_oneq_literals_is_refused_by_name(self):
        with self.oq_runtime(admission.BINARY_LITERALS) as runtime:
            with self.assertRaises(admission.AdmissionRefused) as caught:
                admission.check_runtime(self.tmp, served=sha(runtime.binary))
        self.assertIn("lacks '[QWEN-SDPA-PF] oneq needs one q chunk per core', '[QWEN-SDPA-PF] oneq=1 q_chunks='", str(caught.exception))

    def test_the_k64j_bytes_are_not_k64j_oq(self):
        with FakeRuntime(self.tmp):
            with self.assertRaises(admission.AdmissionRefused) as caught:
                admission.check_runtime(self.tmp, served=OQ)
        self.assertIn("not K64j-OQ's %s" % OQ[:16], str(caught.exception))
        self.assertIn('lacks', str(caught.exception))


class EvidenceTests(unittest.TestCase):
    def problems(self, record, environ, capacity=admission.CAPACITY_131K):
        return admission.evidence_problems(record, tp=4, capacity=capacity, environ=environ)

    def test_k64j_is_unchanged_by_everything_above(self):
        for capacity in admission.CAPACITIES:
            self.assertEqual(self.problems(real_record(capacity), {}, capacity), [], capacity)
            self.assertEqual(self.problems(real_record(capacity), {ENV: K64J}, capacity), [], capacity)
            self.assertEqual(self.problems(real_record(capacity), dict(GATE, **{ENV: K64J}), capacity), [], capacity)

    def test_k64j_oq_borrows_k64j_s_records_in_a_gate_boot(self):
        for capacity in admission.CAPACITIES:
            self.assertEqual(self.problems(real_record(capacity), dict(GATE, **{ENV: OQ}), capacity), [], capacity)

    def test_a_traffic_boot_on_k64j_oq_needs_a_record_of_its_own(self):
        for capacity in admission.CAPACITIES:
            found = self.problems(real_record(capacity), {ENV: OQ}, capacity)
            self.assertEqual(len(found), 1, found)
            self.assertIn('binary: the evidence qualified %s, not K64j-OQ %s' % (K64J[:16], OQ[:16]), found[0])
            self.assertIn('re-record it on K64j-OQ before traffic', found[0])

    def test_a_record_made_on_k64j_oq_qualifies_it_strictly_and_k64j_no_longer(self):
        for capacity in admission.CAPACITIES:
            record = real_record(capacity)
            record['binary']['ttnncpp_sha256'] = OQ
            self.assertEqual(self.problems(record, {ENV: OQ}, capacity), [], 'traffic boot, its own record')
            self.assertEqual(self.problems(record, dict(GATE, **{ENV: OQ}), capacity), [])
            found = self.problems(record, {}, capacity)
            self.assertEqual(len(found), 1)
            self.assertIn('not K64j %s' % K64J[:16], found[0])
            self.assertNotIn('re-record', found[0], 'K64j never borrows')

    def test_an_unknown_record_binary_is_refused_in_every_boot(self):
        record = real_record()
        record['binary']['ttnncpp_sha256'] = 'f' * 64
        for environ in ({}, {ENV: OQ}, dict(GATE, **{ENV: OQ}), dict(GATE, **{ENV: K64J})):
            found = self.problems(record, environ)
            self.assertEqual(len(found), 1, (environ, found))
            self.assertIn('ffffffffffffffff', found[0])

    def test_the_kernels_stay_k64j_s_for_k64j_oq(self):
        record = real_record()
        record['kernels'] = dict(record['kernels'], **{'dataflow/reader_decode_qwen.cpp': '0' * 64})
        found = self.problems(record, dict(GATE, **{ENV: OQ}))
        self.assertEqual(len(found), 1)
        self.assertIn("other kernel bytes than K64j's four", found[0])

    def test_check_evidence_and_the_262k_window_take_the_boot_s_environment(self):
        # the real files at their pins, through check_evidence / check_window_262k / tp_guard
        self.assertIsInstance(admission.check_evidence(tp=4, environ=dict(GATE, **{ENV: OQ})), dict)
        with self.assertRaises(admission.AdmissionRefused) as caught:
            admission.check_evidence(tp=4, environ={ENV: OQ})
        self.assertIn('not K64j-OQ', str(caught.exception))
        self.assertIsInstance(admission.check_window_262k(writer_state=(True, []), environ=dict(GATE, **{ENV: OQ})), dict)
        with self.assertRaises(admission.AdmissionRefused) as caught:
            admission.check_window_262k(writer_state=(True, []), environ={ENV: OQ})
        self.assertIn('capacity 262144: evidence: binary: the evidence qualified', str(caught.exception))

    def test_tp_guard_follows_the_same_rule(self):
        base = dict(TP4_ENV, **{ENV: OQ})
        self.assertEqual(admission.tp_guard(dict(base, **GATE), log=Lines()), [])
        with self.assertRaises(admission.AdmissionRefused):
            admission.tp_guard(base, log=Lines())


class AdmitTests(unittest.TestCase):
    def setUp(self):
        self.state = mock.patch.dict(admission._STATE, clear=True)
        self.state.start()
        self.addCleanup(self.state.stop)
        self.lines = Lines()

    def runtime(self, sha_):
        return dict(binaries={'build_Release/lib/_ttnncpp.so': sha_, 'build_Release/ttnn/_ttnncpp.so': sha_}, kernels={}, literals=[])

    def admit(self, environ, served, runtime_check=None, binary_record=None, max_position=None):
        environ = dict(TP4_ENV, **environ)
        if max_position:
            environ['QWEN_FAST_MAX_POSITION'] = str(max_position)
        seen = {}

        def check_runtime(root, binaries=None, served_=None):
            seen['served'] = served_
            return self.runtime(served_ or K64J)

        with mock.patch.object(admission, 'check_runtime', side_effect=runtime_check or check_runtime):
            record = admission.admit('/opt/tt-metal', m3=M3, binary_record=binary_record, environ=environ, log=self.lines)
        return record, seen

    def test_a_gate_boot_on_k64j_oq_passes_on_k64j_s_record_and_says_so(self):
        record, seen = self.admit(dict(GATE, **{ENV: OQ}), OQ)
        self.assertEqual(seen['served'], OQ, 'K64j-OQ names itself as the third argument')
        self.assertEqual(record['binary_equivalence'], dict(recorded=K64J, served=OQ))
        self.assertEqual(len(self.lines), 2)
        self.assertIn('evidence recorded on K64j %s stands for K64j-OQ %s by the reviewed equivalence (gate boot only; traffic needs a record on K64j-OQ)'
                      % (K64J[:16], OQ[:16]), self.lines[0])
        self.assertTrue(self.lines[1].startswith('[PINDIAG] packed-any admission passed: K64j-OQ %s x2; ' % OQ[:16]), self.lines[1])
        for line in self.lines:
            self.assertLess(len(line), 250, 'the log capture truncates long lines')

    def test_a_traffic_boot_on_k64j_oq_is_refused_with_the_record_named(self):
        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.admit({ENV: OQ}, OQ)
        self.assertIn('evidence: binary: the evidence qualified %s, not K64j-OQ %s' % (K64J[:16], OQ[:16]), str(caught.exception))
        self.assertFalse(admission.admitted())

    def test_the_262k_window_follows_the_same_rule(self):
        with mock.patch('page_width_tp4.evidence_state', return_value=(True, [])):
            record, _seen = self.admit(dict(GATE, **{ENV: OQ}), OQ, max_position=262144)
            self.assertEqual(record['binary_equivalence']['served'], OQ)
            self.assertTrue(self.lines[-1].endswith(' capacity=262144'), self.lines[-1])
            self.setUp()
            with self.assertRaises(admission.AdmissionRefused) as caught:
                self.admit({ENV: OQ}, OQ, max_position=262144)
        self.assertIn('capacity 262144: evidence: binary: the evidence qualified %s, not K64j-OQ %s' % (K64J[:16], OQ[:16]), str(caught.exception))

    def test_k64j_is_the_same_two_argument_call_with_the_same_line(self):
        record, seen = self.admit({ENV: K64J}, K64J)
        self.assertIsNone(seen['served'], 'check_runtime(root, binaries): the call every earlier test mocks')
        self.assertNotIn('binary_equivalence', record)
        self.assertEqual(len(self.lines), 1)
        self.assertTrue(self.lines[0].startswith('[PINDIAG] packed-any admission passed: K64j %s x2; ' % K64J[:16]), self.lines[0])

    def test_a_sha_that_is_neither_is_refused_by_name_and_the_override_must_match(self):
        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.admit({ENV: 'a' * 64}, None)
        self.assertIn("QWEN_FAST_RUNTIME_BINARY_SHA256=aaaaaaaaaaaaaaaa, not K64j's %s or K64j-OQ's %s" % (K64J[:16], OQ[:16]), str(caught.exception))
        self.setUp()
        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.admit(dict(GATE, **{ENV: OQ}), OQ, binary_record=dict(override=K64J, binaries={}))
        self.assertIn('the runtime binary override admitted %s, not K64j-OQ' % K64J[:16], str(caught.exception))
        self.setUp()
        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.admit(dict(GATE, **{ENV: K64J}), K64J, binary_record=dict(override=OQ, binaries={}))
        self.assertIn('the runtime binary override admitted %s, not K64j' % OQ[:16], str(caught.exception))


class RecorderTests(unittest.TestCase):
    def test_the_recorder_knows_both_binaries_and_defaults_to_k64j(self):
        self.assertEqual(recorder.binary_choice('k64j'), K64J)
        self.assertEqual(recorder.binary_choice('k64j-oq'), OQ)
        with self.assertRaises(SystemExit):
            recorder.binary_choice('k64i')
        args = recorder.parse_args(['--cb1', 'x.json'])
        self.assertEqual(args.binary, 'k64j')
        self.assertEqual(recorder.parse_args(['--cb1', 'x.json', '--binary', 'k64j-oq']).binary, 'k64j-oq')

    def test_binary_problems_follow_the_chosen_binary(self):
        problems = []
        recorder.binary_problems('CB1', dict(binary=dict(sha256=OQ), kernels=dict(found=dict(admission.K64J_KERNELS))), problems, served=OQ)
        self.assertEqual(problems, [])
        recorder.binary_problems('CB1', dict(binary=dict(sha256=K64J), kernels=dict(found=dict(admission.K64J_KERNELS))), problems, served=OQ)
        self.assertEqual(len(problems), 1)
        self.assertIn('the loaded binary is %s, not K64j-OQ %s' % (K64J[:16], OQ[:16]), problems[0])
        problems = []
        recorder.binary_problems('CB1', dict(binary=dict(sha256=K64J), kernels=dict(found=dict(admission.K64J_KERNELS))), problems)
        self.assertEqual(problems, [])

    def test_record_binds_the_chosen_binary(self):
        current = real_record()
        evidence = recorder.record(copy.deepcopy(current), {}, None, admission.CAPACITY_131K, binary=OQ)
        self.assertEqual(evidence['binary']['ttnncpp_sha256'], OQ)
        self.assertIn('K64j-OQ', evidence['binary']['note'])
        again = recorder.record(copy.deepcopy(current), {}, None, admission.CAPACITY_131K)
        self.assertEqual(again['binary'], current['binary'], 'without --binary the record is what it was')


if __name__ == '__main__':
    unittest.main()
