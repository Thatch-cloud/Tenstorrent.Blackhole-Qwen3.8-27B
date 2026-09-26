"""packed_any_admission (S2 design W7): the c2-packed attach refuses unless the shape, the modes, the K64j
runtime and the pinned hardware evidence all hold, the pool it builds holds the extent storage and the
block is the extent block; and the pool refuses to build that storage under the flag unless the
admission passed. Host only: fake runtime roots and evidence files; InImageRuntimeTests runs inside the
C2 image build (docker/qwen-c2-serving.Dockerfile runs every overlaid test module) against the image's
own /opt/tt-metal."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission as admission  # noqa: E402

ROOT = HERE.parent.parent
BUILD_K64J = ROOT / 'optimisation' / 'ttnn-op' / 'k64j' / 'build_k64j.sh'
CARD_B = ROOT / 'optimisation' / 'ttnn-op' / 'k64j' / 'k64j_card_b.py'
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
# The server log's capture cuts lines at about 250 characters (dflash_device.AUDIT_SWITCH's note).
LOG_LINE_LIMIT = 250


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

    def test_the_guard_is_the_pool_s_and_the_qualified_reader_carries_none(self):
        """Review W7 #4: the evidence pins extent_attention_replay.py's bytes, so the admission guard is not in
        it - CB2b dispatched from any S2 branch qualifies the reader this image serves - but in the one source of
        the storage every extent reader is built over: the pool's constructor, first under extent_replay, before
        any allocation (test_serving_buffer_pool runs it)."""
        reader = (HERE / 'extent_attention_replay.py').read_text(encoding='utf-8')
        self.assertNotIn('packed_any_admission', reader)
        self.assertNotIn('require_admitted', reader)
        tree = ast.parse((HERE / 'serving_buffer_pool.py').read_text(encoding='utf-8'))
        pool = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ServingBufferPool')
        init = next(node for node in pool.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        guards = [node for node in init.body if isinstance(node, ast.If) and ast.unparse(node.test) == 'extent_replay']
        self.assertTrue(guards, 'no `if extent_replay:` at the top level of ServingBufferPool.__init__')
        first = guards[0]
        self.assertEqual([ast.unparse(node) for node in first.body],
                         ['from packed_any_admission import require_admitted',
                          "require_admitted('ServingBufferPool extent storage')"])
        allocations = [node for node in init.body if isinstance(node, ast.Try)]
        self.assertLess(init.body.index(first), init.body.index(allocations[0]), 'the guard runs before the allocations')


class EnvironmentCheckTests(unittest.TestCase):
    def test_the_c2_packed_environment_passes(self):
        self.assertEqual(admission.check_environment(GOOD_ENV, M3), [])
        self.assertEqual(admission.check_environment(dict(GOOD_ENV, QWEN_FAST_SDPA_MODES='slice, tail ,share'), M3), [])

    def test_every_refusal(self):
        cases = (
            ((False, 'users=4 FOUR_AS_TWO=unset PACKED_STEP=0'), {}, "the 64-row M3 block's"),
            (M3, {'QWEN_FAST_ANY_REQUEST': None}, 'QWEN_FAST_ANY_REQUEST=(unset), not 1'),
            (M3, {'QWEN_FAST_REPLAY_GROUP_ROWS': '4'}, 'QWEN_FAST_REPLAY_GROUP_ROWS=4, not 8'),
            (M3, {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': None}, "QWEN_SDPA_TREE_SCRATCH_ROUNDS=(unset), not 1: the pinned"),
            (M3, {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '0'}, 'QWEN_SDPA_TREE_SCRATCH_ROUNDS=0, not 1'),
            (M3, {'QWEN_FAST_SDPA_MODES': 'share'}, 'QWEN_FAST_SDPA_MODES=share lacks slice,tail'),
            (M3, {'QWEN_FAST_SDPA_MODES': None}, 'QWEN_FAST_SDPA_MODES= lacks share,slice,tail'),
            # Review W7 #7: tail alone, or without slice, is not the qualified 0x27 ...
            (M3, {'QWEN_FAST_SDPA_MODES': 'tail'}, 'QWEN_FAST_SDPA_MODES=tail lacks share,slice'),
            (M3, {'QWEN_FAST_SDPA_MODES': 'tail,share'}, 'QWEN_FAST_SDPA_MODES=tail,share lacks slice'),
            # ... nor is readahead (0x2F), which the reader would refuse only after the attach built everything.
            (M3, {'QWEN_FAST_SDPA_MODES': 'tail,share,slice,readahead'}, 'names readahead: CB2a and CB2b qualified 0x27'),
            (M3, {'QWEN_FAST_SDPA_MODES': 'tail,share,slice,extent'}, 'names extent'),
            (M3, {'QWEN_FAST_SDPA_MODES': 'tail,share,slice,bogus'}, 'Unknown QWEN_FAST_SDPA_MODES entries: bogus'),
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
        # One entry per problem, each naming its file relative to the root (review W7 #8).
        self.assertEqual(len(caught.exception.problems), 2, caught.exception.problems)
        self.assertTrue(all(problem.startswith('runtime: %s ' % BINARIES[1]) for problem in caught.exception.problems))

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
        self.assertEqual(caught.exception.problems[0], 'runtime: %s is missing' % BINARIES[0])
        self.assertIn('runtime: kernel dataflow/reader_decode_qwen_slice.cpp is', caught.exception.problems[1])
        self.assertIn('not the K64j kernel', caught.exception.problems[1])
        self.assertIn(self.tmp.as_posix(), str(caught.exception), 'the exception names the root once')

    def test_an_unpatched_tree_scratch_factory_is_refused(self):
        import sdpa_tree_scratch

        with FakeRuntime(self.tmp):
            (self.tmp / sdpa_tree_scratch.ROOT / 'sdpa_decode_program_factory.cpp').write_text('// 05708e6d\n')
            with self.assertRaisesRegex(admission.AdmissionRefused, r'sdpa_tree_scratch.audit\(patched=True\)'):
                admission.check_runtime(self.tmp)
            (self.tmp / sdpa_tree_scratch.ROOT / 'sdpa_decode_program_factory.cpp').unlink()
            with self.assertRaises(admission.AdmissionRefused) as caught:
                admission.check_runtime(self.tmp)
        self.assertEqual(caught.exception.problems, ['runtime: sdpa_tree_scratch.audit(patched=True): '
                                                     'FileNotFoundError sdpa_decode_program_factory.cpp'])

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


# More than 50 distinct families, the five named ones among them (W10b R2's plan at 56 by default).
R2_FAMILIES = sorted(set(admission.CB2B_R2_NAMED) | set(range(512, 512 + 256 * 51, 256)))


def cb2b_pass(evidence):
    """A full-scope CB2b record as W10b's harness reports it, on the evidence's reader."""
    return dict(status='PASS', run=1, card='M', failures=0, scope='full', chips='1of2', capacity=131328,
                seeds=[0, 1, 2], variants=['normal', 'peaky'],
                r1_geometries={name: [0, 7, 32, 127, 128, 240, 255] for name in admission.CB2B_R1_GEOMETRIES},
                r2_families=list(R2_FAMILIES), idle_starts=[0, 32],
                counts={name: [4, 4] for name in admission.CB2B_COUNTS}, sources=dict(evidence['sources']))


def passing(evidence=None):
    """The checked-in evidence with CB2b recorded as a full pass on the live reader."""
    evidence = copy.deepcopy(evidence or checked_in())
    evidence['sections']['CB2b'] = cb2b_pass(evidence)
    return evidence


def set_path(evidence, path, value):
    """Set (or, with value DELETE, remove) evidence[path[0]][path[1]]..., at each path of a list of them."""
    if isinstance(path, list):
        for each in path:
            set_path(evidence, each, value)
        return
    target = evidence
    for key in path[:-1]:
        target = target[key]
    if value is DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value


DELETE = object()
SECTIONS = ('sections',)


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

    def assert_real_evidence_decides(self):
        """The checked-in evidence (whatever admission.EVIDENCE and EVIDENCE_SHA256 name) lacks at most CB2b:
        while CB2b is not PASS its status is the one problem and check_evidence refuses; once it is, nothing
        is missing and check_evidence returns the record."""
        evidence = json.loads(admission.EVIDENCE.read_text(encoding='utf-8'))
        status = evidence['sections']['CB2b']['status']
        if status == 'PASS':
            self.assertEqual(admission.evidence_problems(evidence), [])
            self.assertEqual(admission.check_evidence(admission.EVIDENCE)['sections']['CB2b']['status'], 'PASS')
        else:
            self.assertEqual(admission.evidence_problems(evidence), ['CB2b: status %s, not PASS' % status])
            with self.assertRaises(admission.AdmissionRefused) as caught:
                admission.check_evidence(admission.EVIDENCE)
            self.assertEqual(caught.exception.problems, ['evidence: CB2b: status %s, not PASS' % status])

    def test_the_checked_in_evidence_lacks_at_most_cb2b(self):
        """Today: CB1 and CB2a PASS, CB2b not run, so every c2-packed attach is refused. When CB2b's record lands
        (and EVIDENCE_SHA256 is re-pinned) the same assertion admits it - see the next test."""
        self.assert_real_evidence_decides()

    def test_recording_cb2b_needs_only_the_record_and_the_pin(self):
        """Review W7 #5: the tests that read the checked-in evidence follow its CB2b status, so landing CB2b is
        the record plus the pin, and nothing else changes: the same assertion over a scratch copy with CB2b PASS
        and the pin moved."""
        path, digest = self.write(passing(), 'packed_any_evidence.json')
        with mock.patch.object(admission, 'EVIDENCE', path), mock.patch.object(admission, 'EVIDENCE_SHA256', digest):
            self.assert_real_evidence_decides()
            # check_evidence's defaults were bound at import, so the landed file is read explicitly above;
            # with the default path it is the checked-in one.
            self.assertEqual(admission.check_evidence(path)['sections']['CB2b']['run'], 1)

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
        # The floors are the design's sets as the card harness counts them, and the record meets them exactly.
        self.assertEqual((admission.X7_FLOOR, admission.Z_FLOOR, admission.CB2A_K2_TICKETS), (500, 900, 1980))
        # Until CB2b lands, its template names the keys the admission reads (review W7 #1: W10b's S, not
        # 'construction'); once it has, the record itself is checked (test_the_checked_in_evidence_lacks_at_most_cb2b).
        cb2b = evidence['sections']['CB2b']
        if cb2b['status'] != 'PASS':
            required = cb2b['required']
            self.assertEqual(set(required['counts']), set(admission.CB2B_COUNTS))
            for key in ('scope', 'chips', 'capacity', 'seeds', 'variants', 'r1_geometries', 'r2_families',
                        'idle_starts', 'counts', 'sources', 'run', 'failures', 'status'):
                self.assertIn(key, required)

    def test_the_cb2a_sets_are_the_card_harness_s(self):
        """X7's and Z's floors and K2's and X7's sets are k64j_card_b's (parsed, not imported: it puts the probe
        directories on sys.path)."""
        if not CARD_B.is_file():
            self.skipTest('the card harness is not in this tree (an image carries only scripts/ci)')
        tree = ast.parse(CARD_B.read_text(encoding='utf-8'))
        names = {}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ('CB2_EXTENTS', 'CB2_STARTS', 'IDLE_STARTS', 'Z_STARTS',
                                               'K2_DESIGN_SEEDS', 'K2_DESIGN_VARIANTS')):
                names[node.targets[0].id] = eval(compile(ast.Expression(node.value), str(CARD_B), 'eval'),
                                                 {'__builtins__': {}}, dict(names))
        self.assertEqual(names['CB2_EXTENTS'], admission.CB2_EXTENTS)
        self.assertEqual(names['CB2_STARTS'], admission.CB2_STARTS)
        self.assertEqual(names['Z_STARTS'], admission.Z_STARTS)
        self.assertEqual(names['K2_DESIGN_SEEDS'], admission.SEEDS)
        self.assertEqual(names['K2_DESIGN_VARIANTS'], admission.CB2A_VARIANTS)
        self.assertEqual(tuple(names['IDLE_STARTS']), admission.CB2B_IDLE_STARTS)

    def test_a_full_record_passes(self):
        evidence = self.check(passing())
        self.assertEqual(evidence['sections']['CB2b']['status'], 'PASS')
        self.assertEqual(admission.evidence_problems(passing()), [])

    # Review W7 #1: one mutation per check, each refused with exactly its own problem - so a check that is
    # removed or loosened leaves its mutation admitted and the test fails.
    MUTATIONS = (
        # (name, path, value, the one problem)
        ('wrong schema', ('schema',), 'something/else', 'the evidence is not a qwen-c2-packed-any-evidence/1 record'),
        ('wrong binary', ('binary', 'ttnncpp_sha256'), admission.K64I_TTNNCPP_SHA256,
         'binary: the evidence qualified cf54d716669be6b7, not K64j 152951c1c0de5c9d'),
        ('other kernels', ('kernels', 'dataflow/reader_decode_qwen.cpp'), '0' * 64,
         "kernels: the evidence names other kernel bytes than K64j's four"),
        ('no source sha', [('sources', 'extent_attention_replay.py'),
                           SECTIONS + ('CB2b', 'sources', 'extent_attention_replay.py')], DELETE,
         'sources: no sha256 recorded for extent_attention_replay.py'),
        ('unknown section', SECTIONS + ('CB9',), dict(status='PASS'), 'CB9: not a section this admission knows'),
        # CB1
        ('CB1 cut', SECTIONS + ('CB1',), DELETE, 'CB1: the section is missing'),
        ('CB1 pending', SECTIONS + ('CB1', 'status'), 'PENDING', 'CB1: status PENDING, not PASS'),
        ('CB1 run id', SECTIONS + ('CB1', 'run'), DELETE, 'CB1: no run id'),
        ('CB1 run id zero', SECTIONS + ('CB1', 'run'), 0, 'CB1: no run id'),
        ('CB1 failures', SECTIONS + ('CB1', 'failures'), 1, 'CB1: failures 1, not 0'),
        ('CB1 seeds', SECTIONS + ('CB1', 'seeds'), [0, 1, 2, 3], 'CB1: seeds lack [4]'),
        ('CB1 extents', SECTIONS + ('CB1', 'extents'), [2304, 16896, 33024, 65792, 131328],
         "CB1: extents lack K1's [98560]"),
        ('CB1 no G8B2 0x27', SECTIONS + ('CB1', 'combos'), [{'geometry': 'G4B3', 'flags': ['0x21', '0x27']},
                                                           {'geometry': 'G8B2', 'flags': ['0x21', '0x23']}],
         'CB1: no G8B2 0x27 combo, the one the extent reader serves'),
        ('CB1 extent partial', SECTIONS + ('CB1', 'counts', 'extent'), [2099, 2100], 'CB1: extent [2099, 2100] is not a full pass'),
        ('CB1 mixed empty', SECTIONS + ('CB1', 'counts', 'mixed'), [0, 0], 'CB1: mixed [0, 0] is not a full pass'),
        ('CB1 share_slot0 cut', SECTIONS + ('CB1', 'counts', 'share_slot0'), DELETE,
         'CB1: share_slot0 None is not a full pass'),
        ('CB1 trace partial', SECTIONS + ('CB1', 'counts', 'trace'), [143, 144], 'CB1: trace [143, 144] is not a full pass'),
        ('CB1 skip malformed', SECTIONS + ('CB1', 'counts', 'skip'), [24, 23], 'CB1: skip [24, 23] is not a full pass'),
        # CB2a
        ('CB2a cut', SECTIONS + ('CB2a',), DELETE, 'CB2a: the section is missing'),
        ('CB2a run id', SECTIONS + ('CB2a', 'run'), '36239779235', 'CB2a: no run id'),
        ('CB2a failures', SECTIONS + ('CB2a', 'failures'), None, 'CB2a: failures None, not 0'),
        ('CB2a seeds', SECTIONS + ('CB2a', 'seeds'), [0, 1, 2], 'CB2a: seeds lack [3, 4]'),
        ('CB2a variants', SECTIONS + ('CB2a', 'variants'), ['normal'], "CB2a: variants lack ['peaky']"),
        ('K2 failed', SECTIONS + ('CB2a', 'k2', 'verdict'), 'FAIL', 'CB2a: K2 verdict FAIL, not PASS (a failed or '
         'coverage-reduced K2 leaves the exactness policy to the user, design 6.1 D-c)'),
        ('K2 reduced', SECTIONS + ('CB2a', 'k2', 'verdict'), 'REDUCED-PASS', 'CB2a: K2 verdict REDUCED-PASS, not PASS '
         '(a failed or coverage-reduced K2 leaves the exactness policy to the user, design 6.1 D-c)'),
        ('K2 tickets under the floor', SECTIONS + ('CB2a', 'k2', 'tickets'), [38, 38],
         'CB2a: K2 tickets [38, 38], not a full pass of the 1980 the design asks'),
        ('K2 tickets partial', SECTIONS + ('CB2a', 'k2', 'tickets'), [1979, 1980],
         'CB2a: K2 tickets [1979, 1980], not a full pass of the 1980 the design asks'),
        ('K2 rows', SECTIONS + ('CB2a', 'k2', 'rows'), [29729, 29730],
         'CB2a: K2 rows [29729, 29730] are not all bitwise equal'),
        ('X7 trivial', SECTIONS + ('CB2a', 'x7'), [1, 1], 'CB2a: X7 [1, 1], not a full pass of the 500 the design asks'),
        ('X7 reduced', SECTIONS + ('CB2a', 'x7'), [12, 12], 'CB2a: X7 [12, 12], not a full pass of the 500 the design asks'),
        ('X7 partial', SECTIONS + ('CB2a', 'x7'), [499, 500], 'CB2a: X7 [499, 500], not a full pass of the 500 the design asks'),
        ('Z trivial', SECTIONS + ('CB2a', 'z', 'passed'), [1, 1], 'CB2a: Z [1, 1], not a full pass of the 900 the design asks'),
        ('Z reduced', SECTIONS + ('CB2a', 'z', 'passed'), [48, 48], 'CB2a: Z [48, 48], not a full pass of the 900 the design asks'),
        ('Z partial', SECTIONS + ('CB2a', 'z', 'passed'), [899, 900], 'CB2a: Z [899, 900], not a full pass of the 900 the design asks'),
        ('Z families', SECTIONS + ('CB2a', 'z', 'families'), [256, 512, 768, 1024],
         'CB2a: Z families lack [1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840] of the 15 below 4096'),
        # CB2b
        ('CB2b pending', SECTIONS + ('CB2b', 'status'), 'PENDING', 'CB2b: status PENDING, not PASS'),
        ('CB2b run id', SECTIONS + ('CB2b', 'run'), 'the run id', 'CB2b: no run id'),
        ('CB2b failures', SECTIONS + ('CB2b', 'failures'), 2, 'CB2b: failures 2, not 0'),
        ('CB2b reduced scope', SECTIONS + ('CB2b', 'scope'), 'reduced', "CB2b: scope reduced, not full (a reduced run "
         "- the watcher pass, one seed - is never CB2b's evidence)"),
        ('CB2b no chips', SECTIONS + ('CB2b', 'chips'), DELETE, "CB2b: chips None, not the harness's 1of2"),
        ('CB2b capacity', SECTIONS + ('CB2b', 'capacity'), 16640, 'CB2b: capacity 16640, not the served C = 131328'),
        ('CB2b seeds', SECTIONS + ('CB2b', 'seeds'), [0], 'CB2b: seeds lack [1, 2]'),
        ('CB2b variants', SECTIONS + ('CB2b', 'variants'), ['peaky'], "CB2b: variants lack ['normal']"),
        ('CB2b R1 geometry', SECTIONS + ('CB2b', 'r1_geometries', 'G4B1'), DELETE,
         'CB2b: R1 ran no G4B1 (the design asks G8B2, G4B3 and G4B1)'),
        ('CB2b R1 geometries not a map', SECTIONS + ('CB2b', 'r1_geometries'), ['G8B2', 'G4B3', 'G4B1'],
         'CB2b: R1 ran no G8B2,G4B3,G4B1 (the design asks G8B2, G4B3 and G4B1)'),
        ('CB2b R1 words', SECTIONS + ('CB2b', 'r1_geometries', 'G4B3'), [0, 7, 127, 255],
         'CB2b: R1 G4B3 lacks words [240]'),
        ('CB2b R2 fifty families', SECTIONS + ('CB2b', 'r2_families'),
         list(admission.CB2B_R2_NAMED) + [family for family in R2_FAMILIES if family not in admission.CB2B_R2_NAMED][:45],
         'CB2b: R2 replayed 50 families, not more than 50'),
        ('CB2b R2 named', SECTIONS + ('CB2b', 'r2_families'), [family for family in R2_FAMILIES if family != 65792]
         + [20480], 'CB2b: R2 families lack the named [65792]'),
        ('CB2b R2 duplicate', SECTIONS + ('CB2b', 'r2_families'), R2_FAMILIES + [256],
         'CB2b: r2_families is not a list of distinct 256-key families up to 131328'),
        ('CB2b R2 off family', SECTIONS + ('CB2b', 'r2_families'), R2_FAMILIES + [1000],
         'CB2b: r2_families is not a list of distinct 256-key families up to 131328'),
        ('CB2b R2 count only', SECTIONS + ('CB2b', 'r2_families'), 56,
         'CB2b: r2_families is not a list of distinct 256-key families up to 131328'),
        ('CB2b idle starts', SECTIONS + ('CB2b', 'idle_starts'), [0], 'CB2b: idle_starts lack [32]'),
        ('CB2b R1 partial', SECTIONS + ('CB2b', 'counts', 'R1'), [3, 4], 'CB2b: R1 [3, 4] is not a full pass'),
        ('CB2b S cut', SECTIONS + ('CB2b', 'counts', 'S'), DELETE, 'CB2b: S None is not a full pass'),
        ('CB2b R2 partial', SECTIONS + ('CB2b', 'counts', 'R2'), [99, 100], 'CB2b: R2 [99, 100] is not a full pass'),
        ('CB2b R4 empty', SECTIONS + ('CB2b', 'counts', 'R4'), [0, 0], 'CB2b: R4 [0, 0] is not a full pass'),
        ('CB2b liveness dead', SECTIONS + ('CB2b', 'counts', 'liveness'), [5, 6],
         'CB2b: liveness [5, 6] is not a full pass'),
        ('CB2b on other bytes', SECTIONS + ('CB2b', 'sources', 'extent_attention_replay.py'), 'f' * 64,
         'CB2b: ran extent_attention_replay.py at ffffffffffffffff, not the qualified 5633fc3a21a40036'),
    )

    def test_every_evidence_check_refuses_its_mutation_alone(self):
        for name, path, value, problem in self.MUTATIONS:
            evidence = passing()
            set_path(evidence, path, value)
            with self.subTest(mutation=name):
                self.assertEqual(admission.evidence_problems(evidence), [problem])

    def test_a_source_drift_is_named_once(self):
        evidence = passing()
        evidence['sources']['extent_attention_replay.py'] = 'e' * 64
        evidence['sections']['CB2b']['sources']['extent_attention_replay.py'] = 'e' * 64
        problems = admission.evidence_problems(evidence)
        self.assertEqual(len(problems), 1, problems)
        self.assertTrue(problems[0].startswith('sources: extent_attention_replay.py is 5633fc3a21a40036, but the '
                                               'evidence qualified eeeeeeeeeeeeeeee (re-run CB2b on these bytes)'))

    def test_the_watcher_pass_transcribed_as_pass_is_refused(self):
        """Review W7 #1's scenario: W10b's watcher pass prints 'K64J_READER verdict=PASS scope=reduced' and is
        never CB2b's evidence; transcribed as {status: PASS, counts: R1/R2/R4/construction [n, n]} it is refused,
        and every coverage it lacks is named."""
        evidence = checked_in()
        evidence['sections']['CB2b'] = dict(status='PASS', run=36240000000, failures=0,
                                            counts={name: [12, 12] for name in ('R1', 'R2', 'R4', 'construction')},
                                            sources=dict(evidence['sources']))
        problems = admission.evidence_problems(evidence)
        for words in ('CB2b: scope None, not full', "CB2b: chips None, not the harness's 1of2",
                      'CB2b: capacity None', 'CB2b: seeds lack [0, 1, 2]', "CB2b: variants lack ['normal', 'peaky']",
                      'CB2b: R1 ran no G8B2,G4B3,G4B1', 'CB2b: r2_families is not a list',
                      'CB2b: idle_starts lack [0, 32]', 'CB2b: S None is not a full pass',
                      'CB2b: liveness None is not a full pass'):
            self.assertTrue(any(problem.startswith(words) for problem in problems), (words, problems))
        with self.assertRaises(admission.AdmissionRefused):
            self.check(evidence)
        reduced = passing()
        reduced['sections']['CB2b'].update(scope='reduced', seeds=[0], r2_families=R2_FAMILIES[:8])
        self.assertEqual(admission.evidence_problems(reduced)[:3], [
            "CB2b: scope reduced, not full (a reduced run - the watcher pass, one seed - is never CB2b's evidence)",
            'CB2b: seeds lack [1, 2]', 'CB2b: R2 replayed 8 families, not more than 50'])

    def test_check_evidence_carries_one_problem_per_entry(self):
        evidence = passing()
        evidence['sections']['CB2a']['x7'] = [1, 1]
        evidence['sections']['CB2b']['scope'] = 'reduced'
        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.check(evidence)
        self.assertEqual(caught.exception.problems, [
            'evidence: CB2a: X7 [1, 1], not a full pass of the 500 the design asks',
            "evidence: CB2b: scope reduced, not full (a reduced run - the watcher pass, one seed - is never CB2b's "
            'evidence)'])

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
        self.assertLess(len(self.lines[0]), LOG_LINE_LIMIT, 'the log capture truncates long lines')
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
            raise admission.AdmissionRefused('the runtime is not K64j as qualified: x; y',
                                             ['runtime: x', 'runtime: y'])

        def evidence(path):
            raise admission.AdmissionRefused('the evidence does not qualify the extent path: CB2b: status PENDING, not PASS',
                                             ['evidence: CB2b: status PENDING, not PASS'])

        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.admit(environ=environ, m3=(False, 'users=4 FOUR_AS_TWO=unset PACKED_STEP=0'), runtime=runtime,
                       evidence=evidence)
        text = str(caught.exception)
        for words in ("not users=4 FOUR_AS_TWO=unset PACKED_STEP=0", 'QWEN_FAST_REPLAY_GROUP_ROWS=4, not 8',
                      'QWEN_FAST_RUNTIME_BINARY_SHA256=(unset)', 'runtime: x | runtime: y', 'CB2b: status PENDING'):
            self.assertIn(words, text)
        self.assertEqual(len(self.lines), 6)
        self.assertEqual(caught.exception.problems[3:], ['runtime: x', 'runtime: y',
                                                         'evidence: CB2b: status PENDING, not PASS'])
        for index, (line, problem) in enumerate(zip(self.lines, caught.exception.problems), 1):
            self.assertEqual(line, '[PINDIAG] packed-any admission refused (%d/6): %s' % (index, problem))
        self.assertFalse(admission.admitted())

    def test_a_k64i_image_logs_each_problem_on_a_short_line(self):
        """Review W7 #8: on a K64i image (the wrong binary at both paths, without F22, K64i's kernels, the tree
        scratch unpatched) with the real evidence and a wrong environment, every sub-problem of the runtime and
        the evidence is its own line, and every line fits the log capture."""
        from dflash_combined_sim_runtime import BINARIES
        import sdpa_tree_scratch

        root = Path(tempfile.mkdtemp(prefix='packed-any-k64i-image-' + 'x' * 60))
        self.addCleanup(shutil.rmtree, str(root), True)
        k64i = b'\x7fELF K64i ' + b'\x00'.join(admission.BINARY_LITERALS[:3])
        for name in BINARIES:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(k64i)
        for name in admission.K64J_KERNELS:
            (root / admission.KERNEL_ROOT / name).parent.mkdir(parents=True, exist_ok=True)
            (root / admission.KERNEL_ROOT / name).write_text('// K64i %s\n' % name)
        for name in sdpa_tree_scratch.HASHES:
            (root / sdpa_tree_scratch.ROOT / name).parent.mkdir(parents=True, exist_ok=True)
            (root / sdpa_tree_scratch.ROOT / name).write_text('// %s\n' % name)
        environ = dict(GOOD_ENV, QWEN_FAST_SDPA_MODES='tail,share,slice,readahead',
                       QWEN_FAST_RUNTIME_BINARY_SHA256=admission.K64I_TTNNCPP_SHA256)
        with self.assertRaises(admission.AdmissionRefused) as caught:
            admission.admit(root, m3=M3, binary_record=dict(override=admission.K64I_TTNNCPP_SHA256, binaries={}),
                            environ=environ, log=self.lines)
        problems = caught.exception.problems
        # modes, the pin, the override; two per binary; four kernels; the tree scratch; and the evidence (CB2b while
        # it is not recorded as PASS).
        pending = checked_in()['sections']['CB2b']['status'] != 'PASS'
        self.assertEqual(len(problems), 3 + 2 * len(BINARIES) + 4 + 1 + pending, problems)
        self.assertEqual(len(self.lines), len(problems))
        self.assertTrue(all(line.startswith('[PINDIAG] packed-any admission refused (') for line in self.lines))
        self.assertEqual(sum(problem.startswith('runtime: ') for problem in problems), 2 * len(BINARIES) + 4 + 1)
        for line in self.lines:
            self.assertLess(len(line), LOG_LINE_LIMIT, line)
        self.assertGreater(len(str(caught.exception)), LOG_LINE_LIMIT, 'the joined message is what one line was')

    def test_an_override_that_is_not_k64j_is_refused(self):
        with self.assertRaisesRegex(admission.AdmissionRefused, 'the runtime binary override admitted cf54d716'):
            self.admit(binary_record=dict(override=admission.K64I_TTNNCPP_SHA256, binaries={}))
        with self.assertRaisesRegex(admission.AdmissionRefused, 'QWEN_FAST_RUNTIME_BINARY_SHA256=cf54d716'):
            self.admit(environ=dict(GOOD_ENV, QWEN_FAST_RUNTIME_BINARY_SHA256=admission.K64I_TTNNCPP_SHA256))

    def test_the_flag_itself_is_required(self):
        with self.assertRaisesRegex(admission.AdmissionRefused, 'QWEN_FAST_EXTENT_REPLAY is not 1'):
            self.admit(environ=dict(GOOD_ENV, QWEN_FAST_EXTENT_REPLAY='0'))

    def assert_the_real_evidence_decides_the_attach(self, evidence_path):
        """With a qualified runtime and environment, the evidence alone decides: refused naming CB2b while it is
        not PASS, admitted once it is."""
        status = json.loads(Path(evidence_path).read_text(encoding='utf-8'))['sections']['CB2b']['status']
        with mock.patch.object(admission, 'check_runtime', return_value=self.runtime):
            if status == 'PASS':
                record = admission.admit('/opt/tt-metal', m3=M3, environ=dict(GOOD_ENV), log=self.lines,
                                         evidence=evidence_path)
                self.assertEqual(record['evidence']['sections']['CB2b']['status'], 'PASS')
                self.assertTrue(admission.admitted())
            else:
                with self.assertRaises(admission.AdmissionRefused) as caught:
                    admission.admit('/opt/tt-metal', m3=M3, environ=dict(GOOD_ENV), log=self.lines,
                                    evidence=evidence_path)
                self.assertEqual(caught.exception.problems, ['evidence: CB2b: status %s, not PASS' % status])
                self.assertFalse(admission.admitted())

    def test_the_real_evidence_decides_the_attach(self):
        self.assert_the_real_evidence_decides_the_attach(admission.EVIDENCE)

    def test_the_real_evidence_decides_the_attach_after_cb2b_lands_too(self):
        """Review W7 #5: the same test over a scratch copy with CB2b recorded as PASS and the pin moved."""
        tmp = Path(tempfile.mkdtemp(prefix='packed-any-landed-'))
        self.addCleanup(shutil.rmtree, str(tmp), True)
        path = tmp / 'packed_any_evidence.json'
        path.write_bytes((json.dumps(passing(), indent=1) + '\n').encode('utf-8'))
        with mock.patch.object(admission, 'EVIDENCE_SHA256', sha(path.read_bytes())):
            self.assert_the_real_evidence_decides_the_attach(path)
        self.assertTrue(admission.admitted())


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


class PoolTests(unittest.TestCase):
    """Review W7 #2: after the pool, the attach requires the extent storage the S2 block keys on."""

    STATISTICS = [dict(chip=0, largest_free=4_000_000_000), dict(chip=1, largest_free=4_000_000_000)]

    def test_the_extent_pool_with_readable_statistics_is_admitted(self):
        lines = Lines()
        pool = mock.Mock(extent_replay=True, dram_statistics=mock.Mock(return_value=self.STATISTICS))
        self.assertIs(admission.admit_pool(pool, log=lines), self.STATISTICS)
        self.assertEqual(lines, ['[PINDIAG] packed-any admission DRAM statistics readable: '
                                 'largest_free=4000.0MB,4000.0MB'])

    def test_a_pool_without_the_extent_storage_is_refused_before_its_statistics_are_read(self):
        for pool in (SimpleNamespace(dram_statistics=mock.Mock(return_value=self.STATISTICS)),
                     SimpleNamespace(extent_replay=False, dram_statistics=mock.Mock(return_value=self.STATISTICS)),
                     SimpleNamespace(extent_replay=1, dram_statistics=mock.Mock(return_value=self.STATISTICS))):
            lines = Lines()
            with self.subTest(extent_replay=getattr(pool, 'extent_replay', None)), \
                    self.assertRaises(admission.AdmissionRefused) as caught:
                admission.admit_pool(pool, log=lines)
            self.assertEqual(caught.exception.problems, [
                'the pool holds no extent storage (extent_replay %r): the block would serve the per-family path, not '
                'the admitted one' % (getattr(pool, 'extent_replay', None),)])
            self.assertEqual(lines, ['[PINDIAG] packed-any admission refused (1/1): ' + caught.exception.problems[0]])
            pool.dram_statistics.assert_not_called()

    def test_the_extent_pool_with_unreadable_statistics_is_refused(self):
        pool = SimpleNamespace(extent_replay=True, dram_statistics=mock.Mock(return_value=dict(unavailable='no view')))
        with self.assertRaisesRegex(admission.AdmissionRefused, 'needs readable DRAM statistics'):
            admission.admit_pool(pool, log=Lines())


def extent_block(segments=4, extent=True, runtime_extent=True):
    readers = [SimpleNamespace(runtime_extent=runtime_extent) for _ in range(segments)]
    return SimpleNamespace(extent=extent, fixture=SimpleNamespace(replay_reader=SimpleNamespace(readers=readers)))


class BlockTests(unittest.TestCase):
    """Review W7 #2: after the block, the attach requires the executed path to be the admitted one."""

    def test_the_extent_block_is_admitted(self):
        lines = Lines()
        self.assertEqual(admission.admit_blocks([extent_block()], log=lines), [4])
        self.assertEqual(lines, ['[PINDIAG] packed-any admission extent block engaged: blocks=1 segments=4'])

    def test_every_other_block_is_refused(self):
        family_readers = extent_block()
        family_readers.fixture.replay_reader.readers[2] = SimpleNamespace()
        cases = (
            ([], ['no packed block was built: the extent path serves its rounds through the block']),
            ([extent_block(extent=False)], ['block 0 is not the extent block (extent False)']),
            ([SimpleNamespace(fixture=extent_block().fixture)], ['block 0 is not the extent block (extent None)']),
            ([extent_block(runtime_extent=False)],
             ['block 0 segments [0, 1, 2, 3] do not report runtime_extent: not the extent readers']),
            ([family_readers], ['block 0 segments [2] do not report runtime_extent: not the extent readers']),
            ([SimpleNamespace(extent=True)],
             ['block 0 has no segment readers to check (fixture.replay_reader.readers)']),
            ([extent_block(), extent_block(extent=False, runtime_extent=None)],
             ['block 1 is not the extent block (extent False)',
              'block 1 segments [0, 1, 2, 3] do not report runtime_extent: not the extent readers']),
        )
        for blocks, problems in cases:
            lines = Lines()
            with self.subTest(problems=problems), self.assertRaises(admission.AdmissionRefused) as caught:
                admission.admit_blocks(blocks, log=lines)
            self.assertEqual(caught.exception.problems, problems)
            self.assertEqual(len(lines), len(problems))


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
