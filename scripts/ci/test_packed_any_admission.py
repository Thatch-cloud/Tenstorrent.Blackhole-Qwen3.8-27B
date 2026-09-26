"""packed_any_admission (S2 design W7): the c2-packed attach refuses unless the shape, the modes, the K64j
runtime and the pinned hardware evidence all hold, and the extent readers refuse construction under the
flag unless it passed. Host only: fake runtime roots and evidence files; InImageRuntimeTests runs inside
the C2 image build (docker/qwen-c2-serving.Dockerfile runs every overlaid test module) against the image's
own /opt/tt-metal."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission as admission  # noqa: E402

ROOT = HERE.parent.parent
BUILD_K64J = ROOT / 'optimisation' / 'ttnn-op' / 'k64j' / 'build_k64j.sh'
IN_IMAGE_RECORD = Path('/opt/qwen-c2/overlay-install.json')
M3 = (True, 'users=4 FOUR_AS_TWO=0 PACKED_STEP=1')
GOOD_ENV = {
    admission.FLAG: '1',
    'QWEN_FAST_ANY_REQUEST': '1',
    'QWEN_FAST_REPLAY_GROUP_ROWS': '8',
    'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1',
    'QWEN_FAST_SDPA_MODES': 'tail,share,slice',
    admission.RUNTIME_BINARY_ENV: admission.K64J_TTNNCPP_SHA256,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Lines(list):
    """A log(template, *values) that keeps the formatted lines."""

    def __call__(self, template, *values):
        self.append(template.format(*values))


class FlagTests(unittest.TestCase):
    def setUp(self):
        self.state = mock.patch.dict(admission._STATE, clear=True)
        self.state.start()
        self.addCleanup(self.state.stop)

    def test_the_flag_is_strictly_zero_or_one(self):
        self.assertFalse(admission.extent_replay_enabled({}))
        self.assertFalse(admission.extent_replay_enabled({admission.FLAG: '0'}))
        self.assertTrue(admission.extent_replay_enabled({admission.FLAG: '1'}))
        for value in ('yes', 'true', 'on', '01', ' 1', ''):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                admission.extent_replay_enabled({admission.FLAG: value})

    def test_require_admitted_holds_only_under_the_flag(self):
        admission.require_admitted('x', {})
        admission.require_admitted('x', {admission.FLAG: '0'})
        with self.assertRaisesRegex(admission.AdmissionRefused, 'x under QWEN_FAST_EXTENT_REPLAY=1 needs'):
            admission.require_admitted('x', {admission.FLAG: '1'})
        admission._STATE['record'] = {}
        self.assertTrue(admission.admitted())
        admission.require_admitted('x', {admission.FLAG: '1'})

    def test_an_extent_reader_refuses_construction_under_the_flag_before_anything(self):
        """Design W7: no serving process builds an extent reader its attach did not admit. The refusal is the
        constructor's first statement: nothing is allocated or even validated before it."""
        import extent_attention_replay

        operations, mesh = mock.Mock(), mock.Mock()
        with mock.patch.dict(os.environ, {admission.FLAG: '1'}), \
                self.assertRaisesRegex(admission.AdmissionRefused, 'ExtentSegmentReader under'):
            extent_attention_replay.ExtentSegmentReader(operations, mesh, 16, 2052, None, storage=None,
                                                        max_group_rows=8, start=128)
        self.assertEqual((operations.mock_calls, mesh.mock_calls), ([], []))
        # With the flag off (the card harnesses, the CPU suite) the guard is silent and the next refusal speaks.
        with mock.patch.dict(os.environ, {admission.FLAG: '0'}), \
                self.assertRaisesRegex(ValueError, 'explicit T8/T16/T32 segment'):
            extent_attention_replay.ExtentSegmentReader(operations, mesh, 7, 2052, None, storage=None,
                                                        max_group_rows=8, start=128)

    def test_the_packed_reader_builds_its_segments_through_the_guarded_constructor(self):
        import ast

        tree = ast.parse((HERE / 'extent_attention_replay.py').read_text(encoding='utf-8'))
        packed = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == 'PackedExtentReplayReader')
        calls = {node.func.id for node in ast.walk(packed) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)}
        self.assertIn('ExtentSegmentReader', calls)
        segment = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ExtentSegmentReader')
        init = next(node for node in segment.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        first = [node for node in init.body if not (isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import))][0]
        self.assertEqual(ast.unparse(first), "require_admitted('ExtentSegmentReader')")


class EnvironmentCheckTests(unittest.TestCase):
    def test_the_c2_packed_environment_passes(self):
        self.assertEqual(admission.check_environment(GOOD_ENV, M3), [])

    def test_every_refusal(self):
        cases = (
            ((False, 'users=4 FOUR_AS_TWO=unset PACKED_STEP=0'), {}, "the 64-row M3 block's"),
            (M3, {'QWEN_FAST_ANY_REQUEST': None}, 'QWEN_FAST_ANY_REQUEST=(unset), not 1'),
            (M3, {'QWEN_FAST_REPLAY_GROUP_ROWS': '4'}, 'QWEN_FAST_REPLAY_GROUP_ROWS=4, not 8'),
            (M3, {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': None}, "QWEN_SDPA_TREE_SCRATCH_ROUNDS=(unset), not 1: the pinned"),
            (M3, {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '0'}, 'QWEN_SDPA_TREE_SCRATCH_ROUNDS=0, not 1'),
            (M3, {'QWEN_FAST_SDPA_MODES': 'share'}, 'lacks tail'),
            (M3, {'QWEN_FAST_SDPA_MODES': None}, 'lacks tail'),
            (M3, {'QWEN_FAST_SDPA_MODES': 'tail,share,slice,extent'}, 'names extent'),
            (M3, {'QWEN_FAST_SDPA_MODES': 'tail,bogus'}, 'Unknown QWEN_FAST_SDPA_MODES entries: bogus'),
        )
        for m3, change, words in cases:
            environ = dict(GOOD_ENV)
            for name, value in change.items():
                if value is None:
                    environ.pop(name)
                else:
                    environ[name] = value
            with self.subTest(change=change, m3=m3):
                problems = admission.check_environment(environ, m3)
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(words, problems[0])


class FakeRuntime(object):
    """A runtime root with K64j-shaped files, and the module constants patched to its bytes."""

    def __init__(self, directory, literals=admission.BINARY_LITERALS):
        from dflash_combined_sim_runtime import BINARIES
        import sdpa_tree_scratch

        self.root = Path(directory)
        self.binary = b'\x7fELF\x00' + b'\x00'.join(literals) + b'\x00tail'
        for name in BINARIES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.binary)
        self.kernels = {}
        for name in admission.K64J_KERNELS:
            path = self.root / admission.KERNEL_ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('// K64j %s\n' % name)
            self.kernels[name] = sha(path.read_bytes())
        self.tree = {}
        for name in sdpa_tree_scratch.HASHES:
            path = self.root / sdpa_tree_scratch.ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('// %s\n' % name)
            self.tree[name] = sha(path.read_bytes())
        self.patches = [mock.patch.object(admission, 'K64J_TTNNCPP_SHA256', sha(self.binary)),
                        mock.patch.object(admission, 'K64J_KERNELS', dict(self.kernels)),
                        mock.patch.object(sdpa_tree_scratch, 'HASHES', dict(self.tree)),
                        mock.patch.object(sdpa_tree_scratch, 'PATCHED_FACTORY_SHA256',
                                          self.tree['sdpa_decode_program_factory.cpp'])]

    def __enter__(self):
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self.patches):
            patch.stop()


class RuntimeCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='packed-any-runtime-'))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)

    def test_a_k64j_runtime_passes(self):
        from dflash_combined_sim_runtime import BINARIES

        with FakeRuntime(self.tmp) as runtime:
            record = admission.check_runtime(self.tmp)
            self.assertEqual(record['binaries'], dict.fromkeys(BINARIES, sha(runtime.binary)))
            self.assertEqual(record['kernels'], runtime.kernels)
            self.assertEqual(set(record['tree_scratch']), set(runtime.tree))

    def test_a_binary_that_is_not_k64j_is_refused(self):
        from dflash_combined_sim_runtime import BINARIES

        with FakeRuntime(self.tmp):
            (self.tmp / BINARIES[1]).write_bytes(b'\x7fELF K64i ' + b'\x00'.join(admission.BINARY_LITERALS[:3]))
            with self.assertRaises(admission.AdmissionRefused) as caught:
                admission.check_runtime(self.tmp)
        text = str(caught.exception)
        self.assertIn(BINARIES[1], text)
        self.assertIn("not K64j's", text)
        self.assertIn("lacks '[QWEN-SDPA] runtime-extent entries=', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS'", text)
        self.assertNotIn(BINARIES[0] + ' is', text)

    def test_a_binary_without_the_f22_literal_is_refused_even_at_the_right_sha(self):
        literals = tuple(literal for literal in admission.BINARY_LITERALS if b'runtime-extent' not in literal)
        with FakeRuntime(self.tmp, literals=literals):
            with self.assertRaisesRegex(admission.AdmissionRefused, "lacks '.QWEN-SDPA. runtime-extent entries='"):
                admission.check_runtime(self.tmp)

    def test_a_missing_binary_and_a_drifted_kernel_are_both_named(self):
        from dflash_combined_sim_runtime import BINARIES

        with FakeRuntime(self.tmp):
            (self.tmp / BINARIES[0]).unlink()
            (self.tmp / admission.KERNEL_ROOT / 'dataflow/reader_decode_qwen_slice.cpp').write_text('// K64i\n')
            with self.assertRaises(admission.AdmissionRefused) as caught:
                admission.check_runtime(self.tmp)
        self.assertIn('%s is missing' % (self.tmp / BINARIES[0]).as_posix(), str(caught.exception))
        self.assertIn('reader_decode_qwen_slice.cpp is', str(caught.exception))
        self.assertIn('not the K64j kernel', str(caught.exception))

    def test_an_unpatched_tree_scratch_factory_is_refused(self):
        import sdpa_tree_scratch

        with FakeRuntime(self.tmp):
            (self.tmp / sdpa_tree_scratch.ROOT / 'sdpa_decode_program_factory.cpp').write_text('// 05708e6d\n')
            with self.assertRaisesRegex(admission.AdmissionRefused, r'sdpa_tree_scratch.audit\(patched=True\)'):
                admission.check_runtime(self.tmp)

    def test_the_override_record_s_hashes_are_used_instead_of_rehashing(self):
        """runtime_binary_override.install hashed both binaries this process; admit hands its record over."""
        from dflash_combined_sim_runtime import BINARIES

        with FakeRuntime(self.tmp) as runtime, \
                mock.patch.object(admission, 'sha256_file', wraps=admission.sha256_file) as hashed:
            admission.check_runtime(self.tmp, binaries=dict.fromkeys(BINARIES, sha(runtime.binary)))
        hashed_paths = {Path(call.args[0]).name for call in hashed.call_args_list}
        self.assertNotIn('_ttnncpp.so', hashed_paths)
        self.assertTrue(hashed_paths)

    def test_the_kernel_pins_are_build_k64j_s(self):
        if not BUILD_K64J.is_file():
            self.skipTest('the K64j build script is not in this tree (an image carries only scripts/ci)')
        text = BUILD_K64J.read_text(encoding='utf-8')
        recorded = dict(line.split('=', 1) for line in text.splitlines()
                        if line.startswith(('K64J_READER_QWEN=', 'K64J_READER_SLICE=', 'K64J_COMPUTE_QWEN=',
                                            'K64J_WRITER_SLICE=')))
        self.assertEqual(admission.K64J_KERNELS, {
            'dataflow/reader_decode_qwen.cpp': recorded['K64J_READER_QWEN'],
            'dataflow/reader_decode_qwen_slice.cpp': recorded['K64J_READER_SLICE'],
            'compute/sdpa_flash_decode_qwen.cpp': recorded['K64J_COMPUTE_QWEN'],
            'dataflow/writer_decode_qwen_slice.cpp': recorded['K64J_WRITER_SLICE']})
        self.assertIn('K64I_TTNNCPP=%s' % admission.K64I_TTNNCPP_SHA256, text)
        for literal in admission.BINARY_LITERALS[1:4]:
            self.assertIn(literal.decode(), text)

    def test_the_binary_literals_are_the_readers_markers(self):
        import pooled_attention_replay

        markers = pooled_attention_replay.required_binary_markers(frozenset({'tail', 'share', 'slice', 'extent'}))
        self.assertEqual(set(markers) | {b'QWEN_SDPA_TREE_SCRATCH_ROUNDS'}, set(admission.BINARY_LITERALS))


def checked_in():
    return json.loads(admission.EVIDENCE.read_text(encoding='utf-8'))


def passing(evidence=None):
    """The checked-in evidence with CB2b recorded as a full pass on the live reader."""
    evidence = copy.deepcopy(evidence or checked_in())
    evidence['sections']['CB2b'] = dict(status='PASS', run=1, card='M', failures=0,
                                        counts={name: [4, 4] for name in admission.CB2B_COUNTS},
                                        sources=dict(evidence['sources']))
    return evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='packed-any-evidence-'))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)

    def write(self, evidence, name='evidence.json'):
        path = self.tmp / name
        path.write_bytes((json.dumps(evidence, indent=1) + '\n').encode('utf-8'))
        return path, sha(path.read_bytes())

    def check(self, evidence):
        path, digest = self.write(evidence)
        return admission.check_evidence(path, expected_sha256=digest)

    def test_the_checked_in_evidence_is_the_pinned_file(self):
        self.assertEqual(sha(admission.EVIDENCE.read_bytes()), admission.EVIDENCE_SHA256,
                         'packed_any_evidence.json changed: re-pin EVIDENCE_SHA256 in the same commit')
        self.assertNotIn(b'\r\n', admission.EVIDENCE.read_bytes())

    def test_the_checked_in_evidence_lacks_only_cb2b(self):
        """Today: CB1 and CB2a PASS, CB2b not run, so every c2-packed attach is refused. When CB2b's record
        lands this test becomes evidence_problems == [] (and the attach can pass)."""
        self.assertEqual(admission.evidence_problems(checked_in()), ['CB2b: status PENDING, not PASS'])
        with self.assertRaisesRegex(admission.AdmissionRefused, 'CB2b: status PENDING, not PASS'):
            admission.check_evidence()

    def test_the_checked_in_evidence_records_what_the_design_asks(self):
        evidence = checked_in()
        self.assertEqual(evidence['binary']['ttnncpp_sha256'], admission.K64J_TTNNCPP_SHA256)
        self.assertEqual(evidence['kernels'], admission.K64J_KERNELS)
        self.assertEqual(set(evidence['sources']), set(admission.QUALIFIED_SOURCES))
        cb1, cb2a = evidence['sections']['CB1'], evidence['sections']['CB2a']
        self.assertEqual((cb1['run'], cb1['seeds'], cb1['failures']), (36223820488, [0, 1, 2, 3, 4], 0))
        self.assertTrue(set(admission.K1_EXTENTS) <= set(cb1['extents']))
        self.assertIn({'geometry': 'G8B2', 'flags': ['0x21', '0x23', '0x27', '0x2f']}, cb1['combos'])
        self.assertEqual((cb2a['run'], cb2a['k2']['verdict'], cb2a['k2']['tickets'], cb2a['x7'], cb2a['z']['passed']),
                         (36239779235, 'PASS', [1980, 1980], [500, 500], [900, 900]))
        self.assertEqual(cb2a['z']['families'], list(range(256, 4096, 256)))

    def test_a_full_record_passes(self):
        evidence = self.check(passing())
        self.assertEqual(evidence['sections']['CB2b']['status'], 'PASS')

    def test_evidence_mutations_are_refused(self):
        def missing_seed(evidence):
            evidence['sections']['CB1']['seeds'] = [0, 1, 2, 3]

        def wrong_binary(evidence):
            evidence['binary']['ttnncpp_sha256'] = admission.K64I_TTNNCPP_SHA256

        def cut_section(evidence):
            del evidence['sections']['CB2a']

        def source_drift(evidence):
            evidence['sources']['extent_attention_replay.py'] = 'e' * 64
            evidence['sections']['CB2b']['sources']['extent_attention_replay.py'] = 'e' * 64

        def k2_failed(evidence):
            evidence['sections']['CB2a']['k2']['verdict'] = 'FAIL'

        def k2_reduced(evidence):
            evidence['sections']['CB2a']['k2']['verdict'] = 'REDUCED-PASS'
            evidence['sections']['CB2a']['k2']['tickets'] = [38, 38]

        def z_short(evidence):
            evidence['sections']['CB2a']['z']['families'] = [256, 512, 768, 1024]

        def cb2b_on_other_bytes(evidence):
            evidence['sections']['CB2b']['sources']['extent_attention_replay.py'] = 'f' * 64

        def cb2b_partial(evidence):
            evidence['sections']['CB2b']['counts']['R2'] = [99, 100]

        def no_g8b2(evidence):
            evidence['sections']['CB1']['combos'] = [{'geometry': 'G4B3', 'flags': ['0x21']}]

        def cb1_failures(evidence):
            evidence['sections']['CB1']['failures'] = 1

        def unknown_section(evidence):
            evidence['sections']['CB9'] = dict(status='PASS')

        def other_kernels(evidence):
            evidence['kernels']['dataflow/reader_decode_qwen.cpp'] = '0' * 64

        def wrong_schema(evidence):
            evidence['schema'] = 'something/else'

        for mutate, words in ((missing_seed, 'CB1: seeds [0, 1, 2, 3] do not cover'),
                              (wrong_binary, 'binary: the evidence qualified cf54d716'),
                              (cut_section, 'CB2a: the section is missing'),
                              (source_drift, 'sources: extent_attention_replay.py is'),
                              (k2_failed, 'CB2a: K2 verdict FAIL, not PASS'),
                              (k2_reduced, 'CB2a: K2 verdict REDUCED-PASS'),
                              (z_short, 'CB2a: Z families'),
                              (cb2b_on_other_bytes, 'CB2b: ran extent_attention_replay.py at ffff'),
                              (cb2b_partial, 'CB2b: R2 [99, 100] is not a full pass'),
                              (no_g8b2, 'CB1: no G8B2 0x27 combo'),
                              (cb1_failures, 'CB1: failures 1, not 0'),
                              (unknown_section, 'CB9: not a section'),
                              (other_kernels, 'kernels: the evidence names other kernel bytes'),
                              (wrong_schema, 'is not a qwen-c2-packed-any-evidence/1 record')):
            evidence = passing()
            mutate(evidence)
            with self.subTest(mutation=mutate.__name__), self.assertRaises(admission.AdmissionRefused) as caught:
                self.check(evidence)
            self.assertIn(words, str(caught.exception))

    def test_an_edited_file_without_a_new_pin_is_refused(self):
        path, digest = self.write(passing())
        path.write_bytes(path.read_bytes().replace(b'"card": "M"', b'"card": "B"', 1))
        with self.assertRaisesRegex(admission.AdmissionRefused, 'not the reviewed'):
            admission.check_evidence(path, expected_sha256=digest)
        with self.assertRaisesRegex(admission.AdmissionRefused, 'is missing'):
            admission.check_evidence(self.tmp / 'absent.json', expected_sha256=digest)
        path.write_bytes(b'{not json')
        with self.assertRaisesRegex(admission.AdmissionRefused, 'does not parse'):
            admission.check_evidence(path, expected_sha256=sha(b'{not json'))

    def test_the_live_reader_is_what_the_sources_are_read_from(self):
        live = sha((HERE / 'extent_attention_replay.py').read_bytes())
        self.assertEqual(checked_in()['sources']['extent_attention_replay.py'], live,
                         'extent_attention_replay.py changed: CB2b must qualify the new bytes and the evidence be re-pinned')
        other = self.tmp / 'tree'
        other.mkdir()
        (other / 'extent_attention_replay.py').write_text('# another reader\n')
        problems = admission.evidence_problems(passing(), sources_root=other)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('re-run CB2b on these bytes', problems[0])


class AdmitTests(unittest.TestCase):
    def setUp(self):
        self.state = mock.patch.dict(admission._STATE, clear=True)
        self.state.start()
        self.addCleanup(self.state.stop)
        self.lines = Lines()
        self.runtime = dict(binaries={'build_Release/lib/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256,
                                      'build_Release/ttnn/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256})
        self.evidence = passing()

    def admit(self, environ=GOOD_ENV, m3=M3, binary_record=None, runtime=None, evidence=None):
        with mock.patch.object(admission, 'check_runtime', side_effect=runtime or (lambda root, binaries: self.runtime)), \
                mock.patch.object(admission, 'check_evidence', side_effect=evidence or (lambda path: self.evidence)):
            return admission.admit('/opt/tt-metal', m3=m3, binary_record=binary_record, environ=dict(environ),
                                   log=self.lines)

    def test_a_qualified_attach_is_admitted_once_per_process(self):
        record = self.admit()
        self.assertTrue(admission.admitted())
        self.assertEqual((record['runtime'], record['evidence']), (self.runtime, self.evidence))
        self.assertEqual(len(self.lines), 1)
        self.assertTrue(self.lines[0].startswith('[PINDIAG] packed-any admission passed: K64j 152951c1c0de5c9d x2; '
                                                 'kernels 409a1aaf,adb60918,518d8096,642c36f8; evidence '))
        self.assertIn('CB1 36223820488 CB2a 36239779235 CB2b 1', self.lines[0])
        self.assertLess(len(self.lines[0]), 250, 'the log capture truncates long lines')
        again = self.admit(runtime=AssertionError('checked twice'), evidence=AssertionError('checked twice'))
        self.assertIs(again, record)

    def test_the_override_record_s_hashes_reach_the_runtime_check(self):
        seen = {}
        record = dict(override=admission.K64J_TTNNCPP_SHA256, binaries={'a': admission.K64J_TTNNCPP_SHA256})

        def runtime(root, binaries):
            seen.update(root=root, binaries=binaries)
            return self.runtime

        self.admit(binary_record=record, runtime=runtime)
        self.assertEqual(seen, dict(root='/opt/tt-metal', binaries=record['binaries']))

    def test_every_problem_is_named_and_logged_one_per_line_and_nothing_is_cached(self):
        environ = dict(GOOD_ENV, QWEN_FAST_REPLAY_GROUP_ROWS='4')
        environ.pop(admission.RUNTIME_BINARY_ENV)

        def runtime(root, binaries):
            raise admission.AdmissionRefused('the runtime is not K64j as qualified: x')

        def evidence(path):
            raise admission.AdmissionRefused('the evidence does not qualify the extent path: CB2b: status PENDING, not PASS')

        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.admit(environ=environ, m3=(False, 'users=4 FOUR_AS_TWO=unset PACKED_STEP=0'), runtime=runtime,
                       evidence=evidence)
        text = str(caught.exception)
        for words in ("not users=4 FOUR_AS_TWO=unset PACKED_STEP=0", 'QWEN_FAST_REPLAY_GROUP_ROWS=4, not 8',
                      'QWEN_FAST_RUNTIME_BINARY_SHA256=(unset)', 'the runtime is not K64j', 'CB2b: status PENDING'):
            self.assertIn(words, text)
        self.assertEqual(len(self.lines), 5)
        self.assertTrue(all(line.startswith('[PINDIAG] packed-any admission refused (%d/5): ' % index)
                            for index, line in enumerate(self.lines, 1)))
        self.assertFalse(admission.admitted())

    def test_an_override_that_is_not_k64j_is_refused(self):
        with self.assertRaisesRegex(admission.AdmissionRefused, 'the runtime binary override admitted cf54d716'):
            self.admit(binary_record=dict(override=admission.K64I_TTNNCPP_SHA256, binaries={}))
        with self.assertRaisesRegex(admission.AdmissionRefused, 'QWEN_FAST_RUNTIME_BINARY_SHA256=cf54d716'):
            self.admit(environ=dict(GOOD_ENV, QWEN_FAST_RUNTIME_BINARY_SHA256=admission.K64I_TTNNCPP_SHA256))

    def test_the_flag_itself_is_required(self):
        with self.assertRaisesRegex(admission.AdmissionRefused, 'QWEN_FAST_EXTENT_REPLAY is not 1'):
            self.admit(environ=dict(GOOD_ENV, QWEN_FAST_EXTENT_REPLAY='0'))

    def test_today_the_real_evidence_refuses_the_attach(self):
        """With a qualified runtime and environment, the checked-in evidence alone refuses: CB2b has not run."""
        with mock.patch.object(admission, 'check_runtime', return_value=self.runtime), \
                self.assertRaisesRegex(admission.AdmissionRefused, 'CB2b: status PENDING, not PASS'):
            admission.admit('/opt/tt-metal', m3=M3, environ=dict(GOOD_ENV), log=self.lines)


class StatisticsTests(unittest.TestCase):
    def test_readable_statistics_are_returned_and_logged(self):
        lines = Lines()
        statistics = [dict(chip=0, largest_free=1_500_000_000), dict(chip=1, largest_free=1_400_000_000)]
        pool = mock.Mock(dram_statistics=mock.Mock(return_value=statistics))
        self.assertIs(admission.admit_statistics(pool, log=lines), statistics)
        self.assertEqual(lines, ['[PINDIAG] packed-any admission DRAM statistics readable: '
                                 'largest_free=1500.0MB,1400.0MB'])

    def test_unreadable_statistics_refuse_the_attach(self):
        for statistics, words in ((dict(unavailable='AttributeError: no memory view'), 'no memory view'),
                                  (dict(), 'no statistics'),
                                  ([], 'no per-chip largest free block'),
                                  ([dict(chip=0)], 'no per-chip largest free block'),
                                  ([dict(chip=0, largest_free=1.5)], 'no per-chip largest free block')):
            lines = Lines()
            pool = mock.Mock(dram_statistics=mock.Mock(return_value=statistics))
            with self.subTest(statistics=statistics), self.assertRaisesRegex(admission.AdmissionRefused, words):
                admission.admit_statistics(pool, log=lines)
            self.assertTrue(lines[0].startswith('[PINDIAG] packed-any admission refused: DRAM statistics unavailable'))


@unittest.skipUnless(IN_IMAGE_RECORD.is_file(), 'runs inside the C2 image build, after the graft and the overlay are '
                                                'installed (docker/qwen-c2-serving.Dockerfile)')
class InImageRuntimeTests(unittest.TestCase):
    """The image the build just made carries the runtime the evidence qualified: K64j at both binary paths
    with its literals, the four K64j kernels, the patched tree-scratch sources; and the evidence file at its
    pin, naming the reader bytes the overlay installed. CB2b's status is not checked here: the image may be
    built before CB2b runs, and then every c2-packed attach refuses."""

    def test_the_image_runtime_is_k64j_as_qualified(self):
        record = admission.check_runtime('/opt/tt-metal')
        self.assertEqual(set(record['binaries'].values()), {admission.K64J_TTNNCPP_SHA256})
        self.assertEqual(record['kernels'], admission.K64J_KERNELS)

    def test_the_image_evidence_is_the_pinned_file_and_names_the_installed_reader(self):
        self.assertEqual(sha(admission.EVIDENCE.read_bytes()), admission.EVIDENCE_SHA256)
        live = sha((admission.HERE / 'extent_attention_replay.py').read_bytes())
        self.assertEqual(checked_in()['sources']['extent_attention_replay.py'], live)


if __name__ == '__main__':
    unittest.main()
