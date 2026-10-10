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


class InImageRuntimeLogicTests(unittest.TestCase):
    """The in-image test of the image build (test_packed_any_admission.InImageRuntimeTests.test_the_image_runtime_is_k64j_as_qualified), run here against a
    fake runtime root for both build-arg values. The build's test steps run BEFORE the Dockerfile's ENV, so the image's declaration is the GRAFT_SHA build arg
    (exposed to every RUN as an environment variable); a running container has QWEN_FAST_RUNTIME_BINARY_SHA256 instead. Run build 38081569132 failed here: the
    test required K64j whatever was baked."""

    def setUp(self):
        import test_packed_any_admission as module
        self.module = module
        self.tmp = Path(tempfile.mkdtemp(prefix='oneq-inimage-'))
        self.addCleanup(__import__('shutil').rmtree, str(self.tmp), True)

    def run_in_image_test(self, runtime, environ):
        """Call the test method body directly (the class is skipped outside an image) with check_runtime pointed at the fake root and `environ` as the process
        environment."""
        real = admission.check_runtime
        case = self.module.InImageRuntimeTests('test_the_image_runtime_is_k64j_as_qualified')
        with runtime, mock.patch.dict('os.environ', environ, clear=True), \
                mock.patch.object(admission, 'check_runtime', lambda root, **kwargs: real(self.tmp, **kwargs)):
            case.test_the_image_runtime_is_k64j_as_qualified()

    def k64j(self):
        runtime = FakeRuntime(self.tmp)
        runtime.patches.append(mock.patch.object(admission, 'SERVED_BINARIES', {sha(runtime.binary): 'K64j', OQ: 'K64j-OQ'}))
        return runtime

    def oq(self):
        runtime = FakeRuntime(self.tmp, admission.BINARY_LITERALS + admission.ONEQ_BINARY_LITERALS)
        runtime.patches.append(mock.patch.object(admission, 'K64J_TTNNCPP_SHA256', K64J))           # K64j stays the real K64j: these bytes are K64j-OQ's, not K64j's
        runtime.patches.append(mock.patch.object(admission, 'K64J_OQ_TTNNCPP_SHA256', sha(runtime.binary)))
        runtime.patches.append(mock.patch.object(admission, 'SERVED_BINARIES', {K64J: 'K64j', sha(runtime.binary): 'K64j-OQ'}))
        return runtime

    def test_the_k64j_image_passes_with_its_build_arg_with_the_runtime_pin_or_with_neither(self):
        pin = sha(self.k64j().binary)
        for environ in ({'GRAFT_SHA': pin}, {admission.RUNTIME_BINARY_ENV: pin}, {admission.RUNTIME_BINARY_ENV: pin, 'GRAFT_SHA': pin}, {}):
            with self.subTest(environ=sorted(environ)):
                self.run_in_image_test(self.k64j(), environ)

    def test_the_k64j_oq_image_passes_with_its_build_arg_or_its_runtime_pin(self):
        pin = sha(self.oq().binary)
        for environ in ({'GRAFT_SHA': pin}, {admission.RUNTIME_BINARY_ENV: pin}, {admission.RUNTIME_BINARY_ENV: pin, 'GRAFT_SHA': pin}):
            with self.subTest(environ=sorted(environ)):
                self.run_in_image_test(self.oq(), environ)

    def test_a_baked_k64j_oq_does_not_pass_as_k64j_and_the_other_way_round(self):
        with self.assertRaises(admission.AdmissionRefused) as caught:           # K64j declared (the default build arg), K64j-OQ bytes baked
            self.run_in_image_test(self.oq(), {'GRAFT_SHA': K64J})
        self.assertIn("not K64j's", str(caught.exception))
        plain = sha(self.k64j().binary)
        with self.assertRaises(admission.AdmissionRefused) as caught:           # K64j-OQ declared, K64j bytes baked (the build arg without the graft)
            self.run_in_image_test(self.k64j_declaring_oq(plain), {'GRAFT_SHA': 'e' * 64})
        self.assertIn('K64j-OQ', str(caught.exception))

    def k64j_declaring_oq(self, plain):
        runtime = FakeRuntime(self.tmp)
        runtime.patches.append(mock.patch.object(admission, 'K64J_OQ_TTNNCPP_SHA256', 'e' * 64))
        runtime.patches.append(mock.patch.object(admission, 'SERVED_BINARIES', {plain: 'K64j', 'e' * 64: 'K64j-OQ'}))
        return runtime

    def test_any_other_pin_is_refused_by_name_and_two_pins_that_disagree_are_refused(self):
        for environ in ({'GRAFT_SHA': 'ab' * 32}, {'GRAFT_SHA': 'not a sha'}, {admission.RUNTIME_BINARY_ENV: 'cf54d716669be6b71f1d627e74892c90f562495dc9500589408a72b4ddccf4a4'}):
            with self.subTest(environ=environ), self.assertRaises(admission.AdmissionRefused) as caught:
                self.run_in_image_test(self.k64j(), environ)
            self.assertIn('names neither K64j', str(caught.exception))
        with self.assertRaises(admission.AdmissionRefused) as caught:
            admission.image_binary({admission.RUNTIME_BINARY_ENV: OQ, 'GRAFT_SHA': K64J})
        self.assertIn('the image pins disagree', str(caught.exception))

    def test_image_binary_reads_the_runtime_pin_and_the_build_arg_and_defaults_to_k64j(self):
        self.assertEqual(admission.image_binary({}), K64J)
        self.assertEqual(admission.image_binary({'GRAFT_SHA': OQ}), OQ)
        self.assertEqual(admission.image_binary({'GRAFT_SHA': OQ.upper()}), OQ)
        self.assertEqual(admission.image_binary({admission.RUNTIME_BINARY_ENV: K64J}), K64J)
        self.assertEqual(admission.image_binary({admission.RUNTIME_BINARY_ENV: OQ, 'GRAFT_SHA': OQ}), OQ)
        self.assertEqual(admission.image_binary({'GRAFT_SHA': ''}), K64J)

    def test_the_dockerfile_exposes_the_build_arg_before_every_test_step_that_reads_it(self):
        """The in-image test steps are RUN steps before the ENV instruction: GRAFT_SHA must be declared above them (an ARG is an environment variable of the RUN
        steps that follow its declaration) and the runtime-pin ENV must come after, which is why the test reads the build arg."""
        dockerfile = (HERE.parents[1] / 'docker' / 'qwen-c2-serving.Dockerfile').read_text(encoding='utf-8')
        arg = dockerfile.index('\nARG GRAFT_SHA=')
        for step in ('python3 -B -m unittest $tests', 'qwen_prefix_stage.py apply'):
            self.assertLess(arg, dockerfile.index(step), step)
        self.assertLess(dockerfile.index('python3 -B -m unittest $tests'), dockerfile.index('QWEN_FAST_RUNTIME_BINARY_SHA256=${GRAFT_SHA}'))
        self.assertEqual(dockerfile.count('\nFROM '), 1, 'one stage: the ARG stays in scope for every step below it')


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


class BakeGraftJobTests(unittest.TestCase):
    """C2_BAKE_GRAFT (c2_serving_job.py): the build job's switch to the K64j-OQ graft. Empty is K64j; the one other value is K64j-OQ; a build-time key."""

    ROOT = HERE.parents[1]
    PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
    BUILD = {'C2_CARDS': 'quad', 'C2_ACTIONS': 'build', 'C2_IMAGE_TAG': 'tp4-fusion-1', 'C2_PROFILE': 'c2-packed-tp4'}

    def read(self, **extra):
        import c2_serving_job as job
        return job.read_job(dict(self.BUILD, **extra), sorted(self.PROFILES), root=str(self.ROOT), envs={name: p.get('env') or {} for name, p in self.PROFILES.items()})

    def test_unset_and_empty_render_empty_so_the_default_build_is_unchanged(self):
        self.assertEqual(self.read()['bake_graft'], '')
        self.assertEqual(self.read(C2_BAKE_GRAFT='')['bake_graft'], '')

    def test_k64j_oq_is_the_one_accepted_name(self):
        self.assertEqual(self.read(C2_BAKE_GRAFT='K64j-OQ')['bake_graft'], 'K64j-OQ')
        import c2_serving_job as job
        self.assertEqual(job.BAKE_GRAFTS, ('K64j-OQ',))
        for bad in ('K64j', 'k64j-oq', 'K64j-OQ ', 'opgraft-K64j-OQ', 'K64i'):
            with self.subTest(name=bad), self.assertRaisesRegex(job.JobError, 'C2_BAKE_GRAFT must be empty'):
                self.read(C2_BAKE_GRAFT=bad)

    def test_it_is_a_build_time_key(self):
        import c2_serving_job as job
        with self.assertRaisesRegex(job.JobError, 'C2_BAKE_GRAFT is baked at build time'):
            self.read(C2_ACTIONS='reset smoke', C2_SMOKE_TESTS='warmup,concurrent8_steady', C2_BAKE_GRAFT='K64j-OQ', C2_PROFILE='c2-packed-tp4')

    def test_the_workflow_hands_the_output_to_the_build_script(self):
        workflow = (HERE.parents[1] / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        self.assertIn('C2_BAKE_GRAFT: ${{ steps.job.outputs.bake_graft }}', workflow)
        script = (HERE / 'build-c2-serving-image.sh').read_text(encoding='utf-8')
        self.assertIn('${C2_BAKE_GRAFT:-}', script)

    def test_the_oneq_templates_name_the_key_where_it_belongs(self):
        import c2_serving_job as job
        folder = HERE / 'references' / 'fusion-jobs' / 'WPP'
        text = (folder / 'OQ-B0-build-oq.env').read_text(encoding='utf-8')
        values = job.parse_env(text)
        self.assertEqual(values['C2_BAKE_GRAFT'], 'K64j-OQ')
        self.assertEqual(values['C2_IMAGE_TAG'], 'tp4-fusion-1')
        self.assertEqual(values['C2_BAKE_DEFAULT_PROFILE'], 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic')
        for path in sorted(folder.glob('OQ-*.env')):
            if path.name != 'OQ-B0-build-oq.env':
                self.assertNotIn('C2_BAKE_GRAFT', job.parse_env(path.read_text(encoding='utf-8')), path.name)


class BuildScriptTests(unittest.TestCase):
    """build-c2-serving-image.sh: the graft case block, run in bash on its own, and the pins it shares with the admission and G1."""

    SCRIPT = (HERE / 'build-c2-serving-image.sh').read_text(encoding='utf-8')

    def block(self):
        start = self.SCRIPT.index('case "${C2_BAKE_GRAFT:-}" in')
        end = self.SCRIPT.index('esac', start) + len('esac')
        return self.SCRIPT[start:end]

    def run_block(self, bake):
        import subprocess
        program = ('HOME=/h; graft=/home/thatch/opgraft-K64j; graft_name=opgraft-K64j; graft_sha=%s; %s; '
                   'printf "%%s\\n%%s\\n%%s\\n" "$graft" "$graft_name" "$graft_sha"' % (K64J, self.block()))
        env = {'PATH': '/usr/bin:/bin'}
        if bake is not None:
            env['C2_BAKE_GRAFT'] = bake
        return subprocess.run(['bash', '-c', program], capture_output=True, text=True, env=env)

    def test_the_default_is_k64j_byte_for_byte(self):
        done = self.run_block(None)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.split(), ['/home/thatch/opgraft-K64j', 'opgraft-K64j', K64J])
        self.assertEqual(self.run_block('').stdout.split(), done.stdout.split())

    def test_k64j_oq_swaps_the_directory_the_name_and_the_pin_together(self):
        done = self.run_block('K64j-OQ')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.split(), ['/h/opgraft-K64j-OQ', 'opgraft-K64j-OQ', OQ])

    def test_anything_else_stops_the_build_before_it_starts(self):
        for bad in ('K64j', 'k64j-oq', 'K64i'):
            done = self.run_block(bad)
            self.assertEqual(done.returncode, 2, bad)
            self.assertIn('neither empty (K64j) nor K64j-OQ', done.stderr)

    def test_the_pins_are_the_admission_and_g1_constants(self):
        import c2_image_provenance as provenance
        self.assertEqual(OQ, provenance.OQ_GRAFT_TTNNCPP_SHA256)
        self.assertEqual(K64J, provenance.ENVIRONMENT_SUCCESSIONS['QWEN_FAST_RUNTIME_BINARY_SHA256'][1])
        self.assertIn('graft_sha=%s' % K64J, self.SCRIPT)
        self.assertIn('graft_sha=%s' % OQ, self.SCRIPT)
        self.assertIn('--build-arg "GRAFT_NAME=$graft_name" --build-arg "GRAFT_SHA=$graft_sha"', self.SCRIPT)
        dockerfile = (HERE.parents[1] / 'docker' / 'qwen-c2-serving.Dockerfile').read_text(encoding='utf-8')
        self.assertIn('ARG GRAFT_SHA=%s' % K64J, dockerfile)
        self.assertIn('QWEN_FAST_RUNTIME_BINARY_SHA256=${GRAFT_SHA}', dockerfile)
        self.assertEqual(provenance.dockerfile_env(dockerfile)['QWEN_FAST_RUNTIME_BINARY_SHA256'], K64J)

    def test_the_k64j_oq_graft_directory_is_the_one_the_build_script_stages(self):
        build = (HERE.parents[1] / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_oneq' / 'build_k64j_oq.sh').read_text(encoding='utf-8')
        self.assertIn('opgraft-K64j-OQ', build)
        self.assertIn('opgraft-K64j-OQ', self.SCRIPT)

    def test_the_script_is_valid_bash(self):
        import subprocess
        self.assertEqual(subprocess.run(['bash', '-n', str(HERE / 'build-c2-serving-image.sh')], capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
