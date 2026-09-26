"""CPU checks for the K64j card-B harness (k64j_card_b.py, run_card_b.sh); no device, no ttnn.

  - the helpers: the runnable (shape, flags) combos, the bundle and share positions, the cur_pos words (E - 1, slot 0
    under share, -1 kept), the mixed and share-slot families, the trace's reference sample, the narrow masks against
    the causal writer's generate_mask (test_k64j_probe's mirror), the F4 / F22 log pairing, the cb_bytes extras, the
    decision and the verdict line, the arguments;
  - the contract: the flag, the unknown-flag control, the F20 / F22 literals and the known flags are
    apply_factory_k64j's; the K64j kernel shas are make_k64j_kernels.OUTPUTS and the committed kernels; the runner's
    recorded shas, image and marker are the harness's and the probe runner's; the F22 format the factory prints parses;
  - the runner (needs bash): LF, bash -n, the canonical qual_card block followed by qual_card_select, the launch right
    after the recheck; the dry run on card B with the arm's graft mounts and the harness files; the watcher pass;
    EXPECT_TTNNCPP_SHA256 required; the graft refusals (manifest, binary sha, the F22 literal, a kernel); the serving
    pair refused without ALLOW_SERVING_CARD=1; card M (card B is reserved) with QUAL_CARD=<card M's board id>
    ALLOW_SERVING_CARD=1: the dry run, and the real launch path on a fake rig (scripts/ci/test_qual_card.FakeRig's
    stubs spliced in after the block) - it launches on card M's node with the per-call watchdog, and refuses as on
    card B: a host holder, a --privileged container, any container that can reach any Tenstorrent device (a serving
    target), no override; a hang (the watchdog's exit 3) prints the pair's reset hint;
  - the whole flow on a fake ttnn (test_k64j_probe.FakeTtnn with the K64j factory's semantics: 0x20 reads the words
    at run time, slot 0 under share, -1 skips, the trace re-reads them, the F20 refusals, the F4 and F22 lines):
    PASS end to end once, and each broken variant on the section that must catch it - a program that ignores the
    word (X), share entries on their own slot (M, K), a skip that disturbs a live entry (K), a trace that keeps the
    captured word (T), a word clamped below a family boundary (L: a dead control), a binary that never logs F22 or
    accepts 0x11 (N), a deadline and the SIGTERM handler. CB2a's sections (K2, X7, Z) and the fake's CB2a variants are
    test_k64j_cb2a's.

    py -3.11 -B -m unittest test_k64j_card_b      (from this directory; the bash runner tests need Git Bash on Windows)
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
ROOT = HERE.parents[2]
PROBE_DIR = OPS / 'k64j_probe'
for _path in (str(HERE), str(PROBE_DIR), str(OPS / 'sdpa_decode_qwen'), str(ROOT / 'scripts' / 'ci')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import apply_factory_k64j as factory  # noqa: E402
import k64j_card_b as card_b  # noqa: E402
import make_k64j_kernels as kernels  # noqa: E402
import probe_k64j_card_b as probe  # noqa: E402
import split_model as model  # noqa: E402
import test_k64j_probe as probe_tests  # noqa: E402 - the fake ttnn, the fixtures, the mask mirror, bash
import test_qual_card as qual_tests  # noqa: E402 - scripts/ci: the fake rig (board tree, readlink, docker, fuser)

card = probe.card
RUNNER = HERE / 'run_card_b.sh'
PROBE_RUNNER = PROBE_DIR / 'run_card_b.sh'
QUAL_CARD = ROOT / 'scripts' / 'ci' / 'qual_card.sh'
ARM = ROOT / 'scripts' / 'ci' / 'lever_n_m3native_run_arm.sh'
CARD_B = probe_tests.CARD_B
CARD_M = probe_tests.CARD_M
CARD_A = 'blackhole-3707293C249A5E67'
NL = chr(10)
BASH = probe_tests.BASH
SCRUB = probe_tests.SCRUB + ('K64J_CARD_DRY_RUN',)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return Path(path).read_text(encoding='utf-8')


# ---------------------------------------------------------------------------------------------
# The helpers.
# ---------------------------------------------------------------------------------------------

class HelperTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_the_runnable_combos(self):
        self.assertEqual(card_b.default_combos(), [('G4B3', 0x21), ('G4B3', 0x23), ('G8B2', 0x21), ('G8B2', 0x23),
                                                   ('G8B2', 0x27), ('G8B2', 0x2F)])
        for shape, flags in (('G4B3', 0x27), ('G4B3', 0x2F), ('G4B1', 0x23), ('G8B2', 0x29), ('G8B2', 0x20),
                             ('G8B2', 0x07), ('G8B2', 0x31), ('G9B2', 0x21)):
            self.assertFalse(card_b.valid_combo(shape, flags), (shape, flags))
        self.assertEqual(card_b.parse_combos('G4B3:0x21, G8B2:0x27,G4B3:0x21'), [('G4B3', 0x21), ('G8B2', 0x27)])
        for bad in ('G4B3:0x27', 'G4B3', 'G4B3:zz', 'G8B2:0x11'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                card_b.parse_combos(bad)
        self.assertEqual(card_b.reference_flags(0x2F), 0xF)
        self.assertEqual(card_b.combo_name('G8B2', 0x27), 'G8B2:0x27')

    def test_the_positions_and_the_words(self):
        self.assertEqual(card_b.bundle_positions(2304, 7, 3, 4), [2055, 2059, 2063])
        self.assertEqual(card_b.bundle_positions(2304, 255, 2, 8), [2303, 2303])       # never past the family
        for extent in (512, 2304, 131328):
            for start in probe.STARTS:
                for rows, batch in ((4, 3), (8, 2)):
                    positions = card_b.bundle_positions(extent, start, batch, rows)
                    self.assertTrue(all(model.extent(p) == extent for p in positions))
        self.assertEqual(card_b.share_positions(2055, 3, 4), [2055, 2059, 2063])
        self.assertEqual(card_b.words_for([2055, 300, 131327], False), [2303, 511, 131327])
        self.assertEqual(card_b.words_for([2055, 300, 131327], True), [2303, 2303, 2303])     # slot 0 everywhere
        self.assertEqual(card_b.words_for([-1, 300], False), [-1, 511])
        # Every word is the compile-time call's cur_pos at its family: the split is the call's at capacity E.
        for position in (0, 255, 256, 2055, 131327):
            word = card_b.words_for([position], False)[0]
            self.assertEqual(word, model.extent_cur_pos(position))
            self.assertTrue(model.same_split(word, model.extent(position), 16))

    def test_the_mixed_and_share_families(self):
        self.assertEqual(card_b.mixed_extents(probe.EXTENTS, 3), [2304, 33024, 131328])
        self.assertEqual(card_b.mixed_extents((512, 2304, 4352), 3), [512, 2304, 4352])
        self.assertEqual(card_b.mixed_extents((2304,), 3), [2304] * 3)
        base, others = card_b.share_slot_extents(probe.EXTENTS, probe.CAPACITY, 3)
        self.assertEqual((base, others), (33024, [2304, 16896]))
        self.assertLess(base, probe.CAPACITY)                  # poison past slot 0's family
        base, others = card_b.share_slot_extents((512, 2304, 4352), 4352, 3)
        self.assertEqual((base, others), (2304, [512, 4352]))  # one word below, one above
        self.assertEqual(card_b.share_slot_extents((512, 2304, 4352), 4352, 2), (2304, [512]))

    def test_the_trace_reference_sample(self):
        self.assertEqual(card_b.reference_sample(64, 8), {0, 9, 18, 27, 36, 45, 54, 63})
        self.assertEqual(card_b.reference_sample(4, 2), {0, 3})
        self.assertEqual(card_b.reference_sample(4, 9), {0, 1, 2, 3})
        self.assertEqual((card_b.reference_sample(4, 0), card_b.reference_sample(5, 1)), (set(), {0}))
        plan = card_b.trace_plan(probe.CAPACITY, 64, 2, 8, 0, True)
        self.assertTrue(all(len({model.extent(p) for p in values}) == 1 for values in plan))    # share: one family
        self.assertGreater(probe.distinct_families([values[:1] for values in plan]), 50)
        self.assertGreater(probe.distinct_families(card_b.trace_plan(probe.CAPACITY, 64, 3, 4, 0, False)), 50)

    def test_the_narrow_masks_are_the_causal_writers(self):
        torch = self.torch
        for extent in (512, 2304):
            for start in probe.STARTS:
                positions = card_b.bundle_positions(extent, start, 3, 4)
                mask = probe.position_mask(torch, positions, [extent] * 3, 4, 256)
                self.assertEqual(tuple(mask.shape), (3, 1, 48, 256))
                for slot, position in enumerate(positions):
                    self.assertEqual([bool(value) for value in torch.isinf(mask[slot, 0, 0])],
                                     probe_tests.generate_mask_columns(position))

    def test_the_f4_and_f22_lines_pair(self):
        f4 = ('Op | INFO | [QWEN-SDPA] flags=%s B=%d PNHt=2 St=4104 mask_width_t=8 kv_share=%s scratch_slots=4 '
              'cb_bytes=%d' + NL)
        f22 = ('Op | INFO | [QWEN-SDPA] runtime-extent entries=%d kv_share=%s q_slice=%s '
               'writer=writer_decode_qwen_slice.cpp cur_pos_stick_bytes=64' + NL)
        good = (f4 % ('0x1', 3, 'false', 796736) + f4 % ('0x21', 3, 'false', 796864) + f22 % (3, 'false', 'false')
                + f4 % ('0x27', 2, 'true', 796864) + 'x [QWEN-SDPA] q-slice rows_per_kv=48 ...' + NL
                + f22 % (2, 'true', 'true'))
        events = card_b.extent_lines(good)
        self.assertEqual([kind for kind, _fields in events], ['F4', 'F4', 'F22', 'F4', 'F22'])
        self.assertEqual(card_b.extent_line_problems(events), [])
        self.assertEqual(card_b.cb_extras(events), [
            dict(flags='0x21', B=3, PNHt=2, cb_bytes=796864, k64i_cb_bytes=796736, extra=128),
            dict(flags='0x27', B=2, PNHt=2, cb_bytes=796864, k64i_cb_bytes=796736, extra=128)])
        for broken, needle in (
                (f4 % ('0x21', 3, 'false', 1), '0 F22 lines, expected one'),
                (f4 % ('0x21', 3, 'false', 1) + f22 % (3, 'false', 'false') * 2, '2 F22 lines'),
                (f4 % ('0x23', 3, 'true', 1) + f22 % (3, 'false', 'false'), "'kv_share': ('false', 'true')"),
                (f4 % ('0x27', 2, 'true', 1) + f22 % (2, 'true', 'false'), "'q_slice': ('false', 'true')"),
                (f4 % ('0x21', 3, 'false', 1) + f22 % (2, 'false', 'false'), "'entries': (2, 3)"),
                (f22 % (3, 'false', 'false') + f4 % ('0x1', 3, 'false', 1), 'before any F4 line'),
                (f4 % ('0x1', 3, 'false', 1) + f22 % (3, 'false', 'false'), 'no 0x20')):
            with self.subTest(needle=needle):
                problems = card_b.extent_line_problems(card_b.extent_lines(broken))
                self.assertTrue(any(needle in problem for problem in problems), problems)

    def entry(self, kind, differing=0, decisive=True):
        return probe.comparison('X', kind, 'x', differing, decisive)

    def test_the_decision(self):
        live = [dict(label='a', live=True)]
        good = dict(comparisons=[self.entry('extent_vs_reference'), self.entry('refusal')], liveness=live, failures=[])
        self.assertEqual(card_b.decide(good)['verdict'], 'PASS')
        fail = dict(good, comparisons=good['comparisons'] + [self.entry('refusal', 1)])
        self.assertEqual((card_b.decide(fail)['verdict'], card_b.decide(fail)['decisive_differing']), ('FAIL', 1))
        self.assertEqual(card_b.decide(dict(fail, failures=['x']))['verdict'], 'NO-DECISION')
        self.assertEqual(card_b.decide(dict(good, liveness=[dict(label='b', live=False)]))['verdict'], 'NO-DECISION')
        self.assertEqual(card_b.decide(dict(good, error='boom'))['verdict'], 'NO-DECISION')
        self.assertEqual(card_b.decide(dict(comparisons=[], liveness=[], failures=[]))['verdict'], 'NO-DECISION')
        cut = card_b.decide(dict(good, deadline=dict(seconds=9, skipped=['T/seed0', 'timing/seed0'])))
        self.assertEqual(cut['verdict'], 'NO-DECISION')
        self.assertEqual(card_b.decide(dict(good, deadline=dict(seconds=9, skipped=['L/seed0', 'timing/seed0'])))
                         ['verdict'], 'PASS')              # L is liveness, timing is recorded

    def test_the_verdict_line(self):
        report = dict(comparisons=[self.entry('extent_vs_reference'), self.entry('extent_vs_reference', 2),
                                   self.entry('share_slot0'), self.entry('refusal')],
                      liveness=[dict(label='a', live=True)], failures=[], trace_families_distinct=51,
                      skip_written='unwritten', sections_failed=['T/seed0'], deadline=dict(seconds=5, skipped=['L/seed0']))
        report['decision'] = card_b.decide(report)
        self.assertTrue(card_b.verdict_line(report).startswith(
            'K64J_CARD verdict=FAIL extent=1/2 mixed=none share_slot0=1/1 trace=none fence=none skip=none refusals=1/1 '
            'live=1/1 skipped_rows=unwritten families=51 k2=none k2_verdict=not_run x7=none z=none k4=not_run '
            'sections_failed=T/seed0 deadline_skipped=1 first_differing=["x"]'), card_b.verdict_line(report))

    def test_the_arguments_and_the_run_order(self):
        args = card_b.parse_args(['--out', 'x.json'])
        self.assertEqual((args.capacity, args.extents, args.starts, args.seeds, args.variants, args.sections),
                         (131328, list(probe.EXTENTS), list(probe.STARTS), [0, 1, 2], ['normal'],
                          list(card_b.DEFAULT_SECTIONS)))
        self.assertEqual((args.combos, args.trace_combos, args.trace_families, args.trace_references),
                         (card_b.default_combos(), [('G4B3', 0x21), ('G8B2', 0x27)], 64, 8))
        self.assertEqual(args.expect_binary_sha256, '')
        self.assertEqual([card_b.run_tag(seed, name) for seed, name in card_b.section_runs(args)],
                         ['N/seed0', 'X/seed0', 'M/seed0', 'K/seed0', 'L/seed0', 'T/seed0', 'timing/seed0',
                          'X/seed1', 'M/seed1', 'K/seed1', 'X/seed2', 'M/seed2', 'K/seed2'])
        for bad in (['--extents', '2300'], ['--extents', '2304,2304'], ['--starts', '256'], ['--variants', 'zeroq'],
                    ['--combos', 'G4B3:0x27'], ['--combos', ''], ['--trace-combos', 'G8B2:0x29'], ['--sections', 'S'],
                    ['--expect-binary-sha256', 'abc'], ['--capacity', '1000'], ['--deadline-s', '-1'],
                    ['--trace-references', '-1']):
            with self.subTest(bad=bad), self.assertRaises(SystemExit), mock.patch('sys.stderr'):
                card_b.parse_args(['--out', 'x.json'] + bad)


# ---------------------------------------------------------------------------------------------
# The contract.
# ---------------------------------------------------------------------------------------------

class ContractTests(unittest.TestCase):
    def test_the_constants_are_the_factorys(self):
        self.assertEqual((card_b.EXTENT, card_b.UNKNOWN_CONTROL, card_b.KNOWN_FLAGS),
                         (factory.FLAG_EXTENT, factory.UNKNOWN_FLAG_CONTROL, factory.KNOWN_FLAGS))
        self.assertEqual((card_b.TAIL, card_b.SHARE, card_b.SLICE, card_b.READAHEAD),
                         (factory.FLAG_TAIL, factory.FLAG_SHARE, factory.FLAG_SLICE, factory.FLAG_READAHEAD))
        for name in ('EXTENT_LOG_MARKER', 'EXTENT_MASK_REFUSAL', 'EXTENT_CUR_POS_REFUSAL', 'EXTENT_LAYOUT_REFUSAL',
                     'CUR_POS_REFUSAL'):
            self.assertEqual(getattr(card_b, name), getattr(factory, name), name)
            self.assertIn(getattr(card_b, name), factory.K64J_MARKERS)
        self.assertIn(card_b.UNKNOWN_NEEDLE, factory.F19_MASK_NEW + factory.slice_factory.F15_NEW)
        self.assertEqual(card_b.EXTENT_BINARY_MARKER, factory.EXTENT_LOG_MARKER.encode())
        self.assertEqual(card_b.SLICE_BINARY_MARKER.decode(), factory.slice_factory.SLICE_LOG_MARKER)
        self.assertEqual(card_b.SCRATCH_BINARY_MARKER, b'QWEN_SDPA_TREE_SCRATCH_ROUNDS')
        self.assertEqual(card_b.MAGIC, 0x51DEC000)

    def test_the_f22_line_the_factory_prints_parses(self):
        literal = re.search(r'"(\[QWEN-SDPA\] runtime-extent [^"]*)"', factory.F22).group(1)
        line = literal.format(3, 'true', 'false', 64)
        match = card_b.EXTENT_LINE.search('Op | INFO | ' + line)
        self.assertIsNotNone(match, line)
        self.assertEqual(match.groups(), ('3', 'true', 'false', 'writer_decode_qwen_slice.cpp', '64'))
        self.assertIn('B, qwen_kv_share, qwen_q_slice, cur_pos_stick_size);', factory.F22)
        # The F4 line is K64i's format, so card.FACTORY_LINE reads the 0x20 programs too.
        f4 = re.search(r'"(\[QWEN-SDPA\] flags=[^"]*)"', factory.qwen.F4).group(1)
        self.assertEqual(len(card.factory_lines(f4.format(0x27, 2, 2, 4104, 8, 'true', 4, 796864))), 1)

    def test_the_kernels_are_k64js(self):
        self.assertEqual(card_b.K64J_KERNELS, kernels.OUTPUTS)
        for name, digest in card_b.K64J_KERNELS.items():
            self.assertEqual(sha((HERE / 'kernels' / name).read_bytes()), digest, name)
        self.assertIs(card_b.STOCK_KERNELS, probe.STOCK_KERNELS)
        self.assertEqual(sorted(probe_tests.STOCK_FIXTURES), sorted(probe.STOCK_KERNELS))

    def test_the_runner_records_the_harness_shas_and_the_probes_image(self):
        runner = read(RUNNER)
        names = {'READER_QWEN': 'dataflow/reader_decode_qwen.cpp', 'READER_SLICE': 'dataflow/reader_decode_qwen_slice.cpp',
                 'COMPUTE_QWEN': 'compute/sdpa_flash_decode_qwen.cpp',
                 'WRITER_SLICE': 'dataflow/writer_decode_qwen_slice.cpp'}
        for variable, name in names.items():
            self.assertEqual(re.findall(r'^%s=([0-9a-f]{64})$' % variable, runner, flags=re.M), [kernels.OUTPUTS[name]])
            self.assertIn('"%s:$%s"' % (name, variable), runner)
        for variable, name in (('READER_ALL', 'dataflow/reader_decode_all.cpp'), ('WRITER_ALL', 'dataflow/writer_decode_all.cpp'),
                               ('COMPUTE_ALL', 'compute/sdpa_flash_decode.cpp'),
                               ('DATAFLOW_COMMON', 'dataflow/dataflow_common.hpp'), ('RT_ARGS_COMMON', 'rt_args_common.hpp')):
            self.assertEqual(re.findall(r'^%s=([0-9a-f]{64})$' % variable, runner, flags=re.M), [probe.STOCK_KERNELS[name]])
        image = re.search(r'^IMAGE=\$\{IMAGE:-(sha256:[0-9a-f]{64})\}$', read(PROBE_RUNNER), flags=re.M).group(1)
        self.assertIn('IMAGE=${IMAGE:-%s}' % image, runner)
        self.assertIn("EXTENT_MARKER='%s'" % factory.EXTENT_LOG_MARKER, runner)
        self.assertIn('G=${KOPGRAFT64:-$HOME/opgraft-K64j}', runner)
        self.assertIn('timeout_s=5400 ', runner)
        self.assertIn('-e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', runner)
        self.assertIn('exec python3 -B /bench/k64j_card_b.py "$@"', runner)

    def test_the_runner_is_lf_and_embeds_the_canonical_board_selection(self):
        data = RUNNER.read_bytes()
        self.assertNotIn(b'\r', data)
        text = data.decode('utf-8')
        library = read(QUAL_CARD)
        start = text.index('# >>> qual_card.sh')
        end = text.index(NL, text.index('# <<< qual_card.sh')) + 1
        self.assertEqual(text[start:end], library)
        self.assertEqual(text.count('# >>> qual_card.sh'), 1)
        self.assertTrue(text[end:].startswith('qual_card_select' + NL))
        code = text[end:]
        launches = [m.start() for m in re.finditer(r'^timeout -k 30 "\$timeout_s" "\$\{argv\[@\]\}"', code, flags=re.M)]
        rechecks = [m.start() for m in re.finditer(r'^qual_card_recheck\b', code, flags=re.M)]
        holders = [m.start() for m in re.finditer(r'^\s*qual_refuse_holders$', code, flags=re.M)]
        self.assertEqual((len(launches), len(rechecks)), (1, 1))
        self.assertTrue(holders and holders[-1] < rechecks[0] < launches[0])
        self.assertNotIn('docker run', code[rechecks[0]:launches[0]])
        outside = [line for line in (text[:start] + code).splitlines() if not line.lstrip().startswith('#')]
        for board in (CARD_M, CARD_A):
            self.assertNotIn(board, NL.join(outside))
        self.assertEqual([line for line in outside if re.search(r'/dev/tenstorrent/[0-9]|tt-smi -r [0-9]', line)], [])

    @unittest.skipUnless(BASH, 'bash not found')
    def test_the_runner_parses(self):
        result = subprocess.run([BASH, '-n', RUNNER.as_posix()], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_k4_is_listed_as_not_run(self):
        self.assertEqual(sorted(card_b.NOT_RUN), ['K4'])
        self.assertIn('k2=none k2_verdict=not_run x7=none z=none k4=not_run', card_b.verdict_line(dict(
            comparisons=[], liveness=[], decision=dict(verdict='NO-DECISION', reasons=[], first_differing=[]))))


# ---------------------------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------------------------

BINARY = (b'fake K64j _ttnncpp.so ' + card_b.EXTENT_BINARY_MARKER + b' ' + card_b.SLICE_BINARY_MARKER + b' '
          + card_b.SCRATCH_BINARY_MARKER)


def make_graft(root, binary=BINARY):
    graft = Path(root) / 'graft'
    kernel_dir = graft / 'sdpa_decode' / 'device' / 'kernels'
    for name in ('attn_prep', 'nlp_concat_heads_decode', 'sdpa'):
        (graft / name).mkdir(parents=True)
        (graft / name / 'placeholder.txt').write_bytes(name.encode())
    (graft / '_ttnn.so').write_bytes(b'fake _ttnn.so')
    (graft / '_ttnncpp.so').write_bytes(binary)
    sources = [(name, HERE / 'kernels' / name) for name in kernels.KERNELS] + list(probe_tests.STOCK_FIXTURES.items())
    for name, source in sources:
        (kernel_dir / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, kernel_dir / name)
    probe_tests.write_manifest(graft)
    return graft


@unittest.skipUnless(BASH, 'bash not found')
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_runner(self, **env):
        environ = {key: value for key, value in os.environ.items() if key not in SCRUB}
        environ.update(HOME=self.dir.as_posix(), RESULTS=(self.dir / 'results').as_posix(), K64J_CARD_DRY_RUN='1')
        environ.update(env)
        return subprocess.run([BASH, RUNNER.as_posix()], env=environ, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=120)

    def argv(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')]
        self.assertEqual(len(lines), 1, result.stdout)
        return shlex.split(lines[0][len('### argv: '):])

    def mounts(self, argv):
        out = []
        for index, word in enumerate(argv):
            if word == '--mount':
                out.append(dict(part.split('=', 1) if '=' in part else (part, True)
                                for part in argv[index + 1].split(',')))
        return out

    def arm_graft_mounts(self):
        arm = read(ARM)
        pairs = set(re.findall(r'-v \$KOPGRAFT64/([A-Za-z0-9_.]+):(/opt/[^:" ]+):ro', arm))
        pairs.add(('sdpa', re.search(r'^\s*sdpa_pf_target=(\S+)$', arm, flags=re.M).group(1)))
        self.assertEqual(len(pairs), 7)
        return pairs

    def refused(self, result, needle):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)
        self.assertNotIn('### argv: ', result.stdout)

    def test_the_dry_run_launches_on_card_b_with_the_arms_graft_mounts(self):
        graft = make_graft(self.dir)
        result = self.run_runner(KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(BINARY))
        argv = self.argv(result)
        self.assertEqual(result.stderr, '')
        self.assertIn('### graft %s: _ttnncpp.so %s with the F22 literal, 4 K64j qwen kernels, 5 stock decode sources, '
                      'manifest verified' % (graft.as_posix(), sha(BINARY)[:16]), result.stdout)
        self.assertEqual(argv[:3], ['docker', 'run', '--rm'])
        self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_B)
        self.assertEqual(argv.count('--device'), 1)
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-k64j-card-card-b')
        mounts = self.mounts(argv)
        prefix = graft.as_posix() + '/'
        grafted = {(m['src'][len(prefix):], m['dst']) for m in mounts if m['src'].startswith(prefix)}
        self.assertEqual(grafted, self.arm_graft_mounts())
        self.assertTrue(all(m.get('readonly') for m in mounts if m['src'].startswith(prefix)))
        bench = {m['dst']: m['src'] for m in mounts if m['dst'].startswith('/bench/')}
        self.assertEqual(sorted(bench), ['/bench/k64j_card_b.py', '/bench/probe_k1_card_b.py',
                                         '/bench/probe_k64j_card_b.py', '/bench/split_model.py',
                                         '/bench/test_sdpa_decode_qwen_card_m.py'])
        for dst, src in bench.items():
            self.assertTrue(src.endswith(dst[len('/bench/'):]), src)
        env = [argv[i + 1] for i, word in enumerate(argv) if word == '-e']
        for value in ('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', 'TT_METAL_CACHE=/kcache', 'TT_METAL_HOME=/opt/tt-metal'):
            self.assertIn(value, env)
        self.assertNotIn('TT_METAL_WATCHER=5', env)
        entry = argv.index('--entrypoint')
        self.assertEqual(argv[entry + 1:entry + 4], ['sh', 'sha256:57cb699489436842d7e7bdd5ab917d509b4492fc95f6f083ef7d2865b488c2ef', '-c'])
        inner = argv[entry + 4]
        self.assertIn('exec python3 -B /bench/k64j_card_b.py "$@"', inner)
        for name in list(kernels.KERNELS) + list(probe.STOCK_KERNELS):
            self.assertIn('/sdpa_decode/device/kernels/' + name, inner)
        tail = argv[entry + 5:]
        self.assertEqual(tail[0], 'card')
        args = card_b.parse_args(tail[1:])
        self.assertEqual((args.expect_binary_sha256, args.watchdog, args.no_timing, args.extents, args.combos,
                          args.deadline_s), (sha(BINARY), 300.0, False, list(probe.EXTENTS), card_b.default_combos(),
                                             5400 - card_b.DEADLINE_MARGIN_S))
        self.assertRegex(args.out.as_posix(), r'^/results/card-[0-9]{8}T[0-9]{6}\.json$')

    def test_the_watcher_pass(self):
        graft = make_graft(self.dir)
        result = self.run_runner(KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(BINARY), WATCHER='1')
        argv = self.argv(result)
        env = [argv[i + 1] for i, word in enumerate(argv) if word == '-e']
        self.assertIn('TT_METAL_WATCHER=5', env)
        self.assertTrue(any(m['dst'] == '/opt/tt-metal/generated/watcher' for m in self.mounts(argv)))
        args = card_b.parse_args(argv[argv.index('card') + 1:])
        self.assertEqual((args.extents, args.seeds, args.combos, args.trace_combos, args.trace_families,
                          args.trace_references, args.no_timing, args.watchdog, args.deadline_s),
                         ([2304, 33024], [0], [('G4B3', 0x21), ('G4B3', 0x23), ('G8B2', 0x27)],
                          [('G4B3', 0x21), ('G8B2', 0x27)], 8, 2, True, 120.0, 2100.0))
        self.assertIn('WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog 120 s, container timeout 2700 s', result.stdout)
        argv = self.argv(self.run_runner(KOPGRAFT64=graft.as_posix(), WATCHER='1',
                                         CARD_B_ARGS='--sections N,X --seeds 0,1'))
        args = card_b.parse_args(argv[argv.index('card') + 1:])
        self.assertEqual((args.sections, args.seeds, args.extents), (['N', 'X'], [0, 1], [2304, 33024]))  # last wins

    def test_a_missing_graft_is_reported_in_a_dry_run(self):
        result = self.run_runner(KOPGRAFT64=(self.dir / 'nowhere').as_posix())
        self.argv(result)
        self.assertIn('does not exist here; the graft was not checked', result.stdout)

    def test_a_real_run_needs_the_expected_sha_and_the_card(self):
        graft = make_graft(self.dir)
        self.refused(self.run_runner(KOPGRAFT64=graft.as_posix(), K64J_CARD_DRY_RUN='0'),
                     'refusing: EXPECT_TTNNCPP_SHA256 is required')
        if Path('/dev/tenstorrent/by-id', CARD_B).exists():
            self.skipTest('card B is present on this host: the harness would run')
        result = self.run_runner(KOPGRAFT64=graft.as_posix(), K64J_CARD_DRY_RUN='0', EXPECT_TTNNCPP_SHA256=sha(BINARY))
        self.refused(result, 'refusing: %s (card B, the qualification card) has no device node here' % CARD_B)
        self.assertFalse((self.dir / 'results').exists())

    def test_the_graft_checks_refuse_anything_but_k64j(self):
        graft = make_graft(self.dir)
        expect = dict(KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(BINARY))
        self.refused(self.run_runner(KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256='f' * 64),
                     'refusing: %s/_ttnncpp.so is %s, not ffffffffffffffff' % (graft.as_posix(), sha(BINARY)[:16]))
        reader = graft / 'sdpa_decode' / 'device' / 'kernels' / 'dataflow' / 'reader_decode_qwen_slice.cpp'
        reader.write_bytes(reader.read_bytes() + b'// K64i\n')
        self.refused(self.run_runner(**expect), 'refusing: %s/MANIFEST.sha256 does not verify' % graft.as_posix())
        probe_tests.write_manifest(graft)
        self.refused(self.run_runner(**expect), 'refusing: %s is %s, not the K64j %s'
                     % (reader.as_posix(), sha(reader.read_bytes()), kernels.OUTPUTS[kernels.READER_SLICE]))
        shutil.copyfile(OPS / 'sdpa_decode_slice' / 'reader_decode_qwen_slice.cpp', reader)     # a K64i graft
        probe_tests.write_manifest(graft)
        self.refused(self.run_runner(**expect), 'not the K64j %s' % kernels.OUTPUTS[kernels.READER_SLICE])
        # A K64i binary: no F22 literal.
        old = make_graft(self.dir / 'k64i', binary=b'fake K64i _ttnncpp.so ' + card_b.SLICE_BINARY_MARKER)
        self.refused(self.run_runner(KOPGRAFT64=old.as_posix(), EXPECT_TTNNCPP_SHA256=sha(
            b'fake K64i _ttnncpp.so ' + card_b.SLICE_BINARY_MARKER)),
            "lacks '[QWEN-SDPA] runtime-extent entries=' (not a K64j binary)")
        shutil.rmtree(graft / 'sdpa')
        self.refused(self.run_runner(**expect), 'refusing: %s/sdpa missing' % graft.as_posix())

    def test_the_serving_pair_is_refused(self):
        self.refused(self.run_runner(QUAL_CARD=CARD_M), 'refusing: QUAL_CARD=%s is card M' % CARD_M)

    def test_card_m_is_selected_by_its_board_id_and_the_override(self):
        """Card B is reserved: QUAL_CARD=<card M's board id> ALLOW_SERVING_CARD=1 selects card M (a dry run: the
        launch argv; CardMRunTests runs the real path on a fake rig)."""
        graft = make_graft(self.dir)
        result = self.run_runner(KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(BINARY), QUAL_CARD=CARD_M,
                                 ALLOW_SERVING_CARD='1', WATCHER='1', CARD_B_ARGS='--sections K2,X7,Z')
        argv = self.argv(result)
        self.assertIn('WARNING: ALLOW_SERVING_CARD=1: this run is on %s, card M, half of the serving pair' % CARD_M,
                      result.stderr)
        self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_M)
        self.assertEqual((argv.count('--device'), argv[argv.index('--name') + 1]), (1, 'qwen-k64j-card-card-m'))
        self.assertIn('### k64j-card ', result.stdout)
        self.assertIn(' card=%s (card-m) ' % CARD_M, result.stdout)
        args = card_b.parse_args(argv[argv.index('card') + 1:])
        self.assertEqual((args.sections, args.watchdog, args.expect_binary_sha256),
                         (['K2', 'X7', 'Z'], 120.0, sha(BINARY)))

    def test_the_runner_echoes_the_verdict_line_not_the_summary_line(self):
        lines = [line for line in read(RUNNER).splitlines() if line.startswith('echo ') and 'K64J_CARD' in line]
        self.assertEqual(len(lines), 1, lines)
        log = self.dir / 'card.log'
        log.write_bytes(b'x\nK64J_CARD verdict=PASS extent=10/10\nSDPA_K64J_CARD passed=True\n')
        result = subprocess.run([BASH, '-c', 'set -o pipefail; log=%s; %s' % (shlex.quote(log.as_posix()), lines[0])],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.stdout, '### K64J_CARD verdict=PASS extent=10/10' + NL, result.stderr)


@unittest.skipUnless(BASH, 'bash not found')
class CardMRunTests(unittest.TestCase):
    """Card B is reserved for other work, so the harness runs on card M through this runner: QUAL_CARD=<card M's
    board id> ALLOW_SERVING_CARD=1. These run the runner's REAL path (not the dry run) on a fake rig: the runner and
    the harness files it mounts laid out as in the checkout, with test_qual_card.FakeRig's stubs (the board tree,
    readlink, device numbers, sysfs) plus docker, fuser, sudo, id and timeout stubs spliced in right after the
    embedded qual_card block, so every refusal and the launch run as on the rig."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.rig = qual_tests.FakeRig(self.dir / 'rig')
        self.graft = make_graft(self.dir)
        tree = self.dir / 'ops'
        for source in (HERE / 'k64j_card_b.py', PROBE_DIR / 'probe_k64j_card_b.py', PROBE_DIR / 'split_model.py',
                       OPS / 'sdpa_decode_qwen' / 'test_sdpa_decode_qwen_card_m.py',
                       OPS / 'sdpa_decode_qwen' / 'probe_k1_card_b.py'):
            (tree / source.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, tree / source.parent.name / source.name)
        self.runner = tree / 'k64j' / 'run_card_b.sh'
        self.launched = self.rig.dir / 'docker-run.argv'

    def tearDown(self):
        self.tmp.cleanup()

    def run_on(self, **env):
        text = read(RUNNER)
        end = text.index(NL, text.index('# <<< qual_card.sh')) + 1
        stubs = self.rig.stubs() + [
            'docker() { case $1 in ps) echo %s ;; inspect) cat "$FAKE_DIR/container-$2" ;; image) return 0 ;; '
            'run) printf "%%q " "$@" > "$FAKE_DIR/docker-run.argv"; echo "K64J_CARD verdict=PASS"; '
            'return "${FAKE_RUN_STATUS:-0}" ;; rm) return 0 ;; esac; }' % ' '.join(self.rig.containers),
            'fuser() { echo "$*" >> "$FAKE_DIR/fuser.log"; local n=${@: -1}; case " ${FAKE_HELD:-} " in *" $n "*) '
            'echo "$n: thatch 4242 F.... python3" >&2; return 0 ;; esac; return 1; }',
            'sudo() { return 1; }',
            'id() { echo 1000; }',
            'timeout() { while [ $# -gt 0 ]; do case $1 in -k) shift 2 ;; [0-9]*) shift; break ;; *) break ;; esac; '
            'done; "$@"; }',
        ]
        self.runner.write_bytes((text[:end] + NL.join(stubs) + NL + text[end:]).encode('utf-8'))
        environ = {key: value for key, value in os.environ.items() if key not in SCRUB}
        environ.update(HOME=self.dir.as_posix(), RESULTS=(self.dir / 'results').as_posix(), K64J_CARD_DRY_RUN='0',
                       KOPGRAFT64=self.graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(BINARY), QUAL_CARD=CARD_M,
                       ALLOW_SERVING_CARD='1')
        environ.update(env)
        return subprocess.run([BASH, self.runner.as_posix()], env=environ, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=120)

    def launch_argv(self):
        return shlex.split(self.launched.read_text(encoding='utf-8'))

    def refused(self, result, needle):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)
        self.assertFalse(self.launched.exists(), 'launched despite: ' + needle)

    def test_card_m_launches_on_its_node_with_the_watchdog(self):
        node = self.rig.node(CARD_M)
        result = self.run_on()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('WARNING: ALLOW_SERVING_CARD=1: this run is on %s, card M, half of the serving pair' % CARD_M,
                      result.stderr)
        for line in ('### target card: %s (card M, half of the serving pair) -> %s' % (CARD_M, node),
                     '### containers: none can reach %s (%s)' % (node, CARD_M),
                     '### device holders on %s: none' % node, '### %s is still %s' % (CARD_M, node),
                     '### K64J_CARD verdict=PASS'):
            self.assertIn(line, result.stdout)
        self.assertIn('-v ' + node, self.rig.fuser_calls())
        argv = self.launch_argv()
        self.assertEqual(argv[:2], ['run', '--rm'])
        self.assertEqual((argv.count('--device'), argv[argv.index('--device') + 1]), (1, node))
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-k64j-card-card-m')
        args = card_b.parse_args(argv[argv.index('card') + 1:])
        self.assertEqual((args.watchdog, args.deadline_s, args.expect_binary_sha256), (300.0, 4800.0, sha(BINARY)))
        self.assertTrue(list((self.dir / 'results').glob('card-*.log')))
        # The watcher pass with CB2a's sections: the 120 s per-call watchdog and TT_METAL_WATCHER=5.
        self.launched.unlink()
        result = self.run_on(WATCHER='1', CARD_B_ARGS='--sections K2,X7,Z')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.launch_argv()
        self.assertEqual(argv[argv.index('--device') + 1], node)
        self.assertIn('TT_METAL_WATCHER=5', [argv[i + 1] for i, word in enumerate(argv) if word == '-e'])
        args = card_b.parse_args(argv[argv.index('card') + 1:])
        self.assertEqual((args.sections, args.watchdog, args.deadline_s), (['K2', 'X7', 'Z'], 120.0, 2100.0))

    def test_card_m_needs_the_override(self):
        self.refused(self.run_on(ALLOW_SERVING_CARD='0'), 'refusing: QUAL_CARD=%s is card M' % CARD_M)

    def test_a_host_holder_of_card_m_refuses(self):
        node = self.rig.node(CARD_M)
        self.refused(self.run_on(FAKE_HELD=node), 'refusing: host processes hold %s' % node)
        self.assertEqual(self.rig.fuser_calls(), ['-v ' + node] * 5)

    def test_a_privileged_container_refuses(self):
        self.rig.container('privileged', privileged=True)
        self.refused(self.run_on(), 'refusing: container /privileged is --privileged')

    def test_any_container_on_any_card_refuses_a_card_m_run(self):
        """Card M is a serving card: a container that can reach ANY Tenstorrent device blocks the launch - card M's
        own, card A's, and the card-B agent's (card B only). One on no device does not."""
        for name, card in (('on-card-m', CARD_M), ('on-card-a', CARD_A), ('card-b-agent', CARD_B)):
            with self.subTest(card=card):
                rig = self.rig
                self.rig = qual_tests.FakeRig(self.dir / ('rig-' + name))
                self.launched = self.rig.dir / 'docker-run.argv'
                self.rig.container(name, devices=(self.rig.node(card),))
                self.refused(self.run_on(), 'refusing: container /%s has %s among its devs'
                             % (name, self.rig.node(card)))
                self.rig = rig
                self.launched = self.rig.dir / 'docker-run.argv'
        self.rig.container('no-device', mounts=('/home/thatch/hf-cache',))
        result = self.run_on()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.launched.exists())

    def test_a_hang_on_card_m_resets_the_pair_together(self):
        result = self.run_on(FAKE_RUN_STATUS='3')                   # the harness watchdog's exit
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertIn('HANG SUSPECTED (exit 3)', result.stderr)
        self.assertIn('HANG RECOVERY for %s (card M, half of the serving pair)' % CARD_M, result.stderr)
        self.assertIn('then reset card M and' + NL + '  card A TOGETHER, in one call:', result.stderr)
        self.assertIn('~/.local/bin/tt-smi -r "$m" "$a"', result.stderr)


# ---------------------------------------------------------------------------------------------
# The device flow on a fake ttnn.
# ---------------------------------------------------------------------------------------------

class FakeExtentTtnn(probe_tests.FakeTtnn):
    """test_k64j_probe.FakeTtnn with the K64j factory: flag 0x20 is known (0x10 is not); F20's refusals in the
    factory's order; a 0x20 program reads each entry's word from the cur_pos tensor when it runs (at replay inside a
    trace), slot 0's under share, skips -1, and splits, reads and masks as the non-causal tail call at capacity
    word + 1; each new 0x20 program logs its F4 line and then its F22 line.

    broken: 'ignore_word' (a 0x20 program runs at capacity - 1), 'own_slot' (share entries read their own word),
    'skip_bleeds' (a skipped entry zeroes the next one), 'stale_trace' (a replay keeps the captured words), 'clamp'
    (a word on a family boundary is taken one lower: the liveness control goes dead), 'no_f22', 'accept_0x11';
    for CB2a (test_k64j_cb2a): 'tail_at_capacity' (a tail call reads its mask at the call's capacity, [C - 256, C),
    not at the mask's own last chunk: a wide mask read there, and a narrow one past its width, reading 0.0),
    'narrow_batch_offset' (a sliced call with a narrow mask reads entry 0's mask rows for every entry: the mask
    batch stride lost, the K64i reader_decode_qwen_slice.cpp:338-341 class, shared by 0x27 and 0x7-narrow), and
    'stale_writer' (a 0x20 writer on the compile-time cur_pos: the call hangs wherever split_model.stale_writer_hangs
    says so - the fake lets the harness watchdog's budget pass and fires it).

    The call takes the table as the 4th positional argument or page_table_tensor= (the served calls' spelling) and
    is_causal defaults to True (the native decode call does not pass it); with record_calls, each call's keyword
    names are kept in self.recorded."""

    L1_MEMORY_CONFIG = 'l1'

    def __init__(self, torch, *args, record_calls=False, **kwargs):
        super().__init__(torch, *args, **kwargs)
        self.recorded = [] if record_calls else None

    def attend(self, q, keys, values, table_row, cur_pos, chunk_tiles, scale, cores, causal, mask_row, tail,
               capacity):
        """FakeTtnn.attend with its two products and its sums in float64: a row's bytes then do not depend on how
        many rows share the call (CPU BLAS blocks a 12-row and a 96-row product differently), so K2's B = 1 native
        rows and the G8B2 subject's folded rows agree exactly where the device's per-row arithmetic would. The
        split, the tree, the masks and every bf16 rounding are FakeTtnn's."""
        torch = self.torch

        def bf(value):
            return value.to(torch.bfloat16).float()

        def merge(own, other):
            m = torch.maximum(own[0], other[0])
            a = torch.where(torch.isfinite(own[0]), torch.exp(own[0] - m), torch.zeros_like(m))
            b = torch.where(torch.isfinite(other[0]), torch.exp(other[0] - m), torch.zeros_like(m))
            return m, bf(own[1] * a + other[1] * b), bf(own[2] * a[:, None] + other[2] * b[:, None])

        visible_to = split_pos = cur_pos
        if causal and 'overread' in self.broken:
            visible_to = split_pos = min(capacity - 1, cur_pos + 256)
        if causal and 'split_capacity' in self.broken:
            split_pos = capacity - 1
        plan = model.split(split_pos, cores, chunk_tiles, 8)
        chunk, chunks = plan['chunk'], plan['num_chunks']
        extent = chunks * chunk
        pages = table_row[:-(-extent // card.PAGE)].long()
        positions = torch.arange(extent)
        rows = q.shape[0]
        per_kv = rows // card.KV_HEADS
        out = torch.empty(rows, card.HEAD_DIM)
        for kv in range(card.KV_HEADS):
            lo, hi = kv * per_kv, (kv + 1) * per_kv
            k = keys[pages, kv].reshape(-1, card.HEAD_DIM)[:extent].double()
            v = values[pages, kv].reshape(-1, card.HEAD_DIM)[:extent].double()
            scores = bf((q[lo:hi].double() @ k.T) * scale)
            if causal:
                scores[:, positions > visible_to] = float('-inf')
            elif mask_row is not None:
                bias = mask_row[lo:hi].float()
                if tail:
                    scores[:, extent - 256:] = scores[:, extent - 256:] + bias[:, -256:]
                else:
                    scores = scores + bias[:, :extent]
            blocks = scores.view(hi - lo, chunks, chunk)
            m = blocks.max(dim=2).values
            finite = torch.isfinite(m)
            p = torch.where(finite[..., None], torch.exp(blocks - torch.where(finite, m, torch.zeros_like(m))[..., None]),
                            torch.zeros_like(blocks))
            l = bf(p.double().sum(dim=2))
            o = bf(torch.einsum('rnc,ncd->rnd', p.double(), v.view(chunks, chunk, card.HEAD_DIM)))
            partial = {}
            for core, (first, last) in enumerate(plan['ranges']):
                if first == last:
                    continue
                own = (m[:, first], l[:, first], o[:, first])
                for index in range(first + 1, last):
                    own = merge(own, (m[:, index], l[:, index], o[:, index]))
                partial[core] = own

            def reduce(core):
                own = partial[core]
                for child in model.tree_params(core, cores)['children']:
                    if child is not None and child < chunks and child in partial:
                        own = merge(own, reduce(child))
                return own

            result = reduce(0)
            out[lo:hi] = result[2] / result[1][:, None]
        return out

    def mask_seen(self, mask, flags, batches, capacity):
        """The mask as a broken device reads it (the fake's attend adds the LAST 256 columns of what this returns)."""
        torch = self.torch
        width = mask.shape[3]
        if 'narrow_batch_offset' in self.broken and flags & card_b.SLICE and width == 256 and batches > 1:
            mask = mask[0:1].expand(batches, -1, -1, -1).clone()
        if 'tail_at_capacity' in self.broken and flags & card_b.TAIL:
            read = torch.zeros(mask.shape[:3] + (256,), dtype=mask.dtype)
            first = capacity - 256
            if first + 256 <= width:
                read = mask[..., first:first + 256].clone()
            mask = mask.clone()
            mask[..., -256:] = read
        return mask

    def hang(self):
        """A call that never returns: the harness watchdog's budget passes and it fires (check() prints WATCHDOG,
        runs on_fire and calls its exit, which the tests make raise)."""
        watchdog = probe.WATCHDOG
        with watchdog.lock:
            label = watchdog.label
            watchdog.deadline = 0.0
        if label is None:
            raise RuntimeError('FakeExtentTtnn: a stale writer hung outside a watchdog op (--watchdog 0)')
        watchdog.check()
        raise AssertionError('the watchdog did not fire')

    def paged_scaled_dot_product_attention_decode(self, query, k, v, *positional, **options):
        if self.recorded is not None:
            self.recorded.append(dict(positional=len(positional), options=sorted(options),
                                      program_config=dict(options.get('program_config') or {}),
                                      memory_config=options.get('memory_config'), is_causal=options.get('is_causal'),
                                      rows=query.shape[2], batches=query.shape[1]))
        if len(positional) > 1 or (positional and 'page_table_tensor' in options):
            raise TypeError('one page table, positional or page_table_tensor=')
        pages = positional[0] if positional else options.pop('page_table_tensor')
        is_causal = options.pop('is_causal', True)
        scale, program_config, memory_config = options.pop('scale'), options.pop('program_config'), options.pop(
            'memory_config')
        attn_mask, cur_pos_tensor = options.pop('attn_mask', None), options.pop('cur_pos_tensor', None)
        if options:
            raise TypeError('unexpected keyword arguments %r' % sorted(options))
        sentinel = program_config['q_chunk_size']
        qwen = (sentinel & 0xFFFFFF00) == card.MAGIC
        flags = sentinel & 0xFF if qwen else 0
        if qwen:
            known = card_b.KNOWN_FLAGS | (card_b.UNKNOWN_CONTROL if 'accept_0x11' in self.broken else 0)
            if flags & ~known:
                raise RuntimeError('TT_FATAL: [QWEN-SDPA] unknown flags %#x' % flags)
            extent = bool(flags & card_b.EXTENT)
            if is_causal or (cur_pos_tensor is not None and not extent):
                raise RuntimeError('TT_FATAL: ' + card_b.CUR_POS_REFUSAL)
            if extent:
                width = None if attn_mask is None else attn_mask.shape[3]
                if not (flags & card_b.TAIL and width == 256):
                    raise RuntimeError('TT_FATAL: %s 8-tile mask, got flags %#x and mask width %s'
                                       % (card_b.EXTENT_MASK_REFUSAL, flags, width))
                if cur_pos_tensor is None:
                    raise RuntimeError('TT_FATAL: ' + card_b.EXTENT_CUR_POS_REFUSAL)
                if cur_pos_tensor.shape[-1] != query.shape[1]:
                    raise RuntimeError('TT_FATAL: %s%d entries' % (card_b.EXTENT_LAYOUT_REFUSAL, query.shape[1]))
                return self.extent_call(query, k, v, pages, scale, program_config['k_chunk_size'], flags, attn_mask,
                                        cur_pos_tensor)
            if flags & card_b.UNKNOWN_CONTROL:
                program_config = dict(program_config, q_chunk_size=sentinel & ~card_b.UNKNOWN_CONTROL)
        substitute = None
        if qwen and attn_mask is not None and {'tail_at_capacity', 'narrow_batch_offset'} & self.broken:
            # The broken read of an eager (or captured: FakeTtnn reads it at call time) non-extent qwen call.
            capacity = self.read(pages).shape[1] * card.PAGE
            seen = self.mask_seen(self.read(attn_mask), flags, query.shape[1], capacity)
            attn_mask = substitute = self.from_torch(seen, dtype=self.bfloat16, layout=self.TILE_LAYOUT, device=self)
        try:
            return super().paged_scaled_dot_product_attention_decode(
                query, k, v, pages, is_causal=is_causal, scale=scale, program_config=program_config,
                memory_config=memory_config, attn_mask=attn_mask, cur_pos_tensor=cur_pos_tensor)
        finally:
            if substitute is not None:
                self.deallocate(substitute)

    def extent_call(self, query, k, v, pages, scale, k_chunk, flags, attn_mask, cur_pos_tensor):
        torch = self.torch
        batches, rows = query.shape[1], query.shape[2]
        table = self.read(pages)
        capacity = table.shape[1] * card.PAGE
        share = bool(flags & card_b.SHARE) and batches > 1
        key = (flags, batches, capacity // 32, 8)
        if not self.silent and key not in self.programs:
            self.programs.add(key)
            os.write(1, ('Op | INFO | [QWEN-SDPA] flags=0x%x B=%d PNHt=%d St=%d mask_width_t=8 kv_share=%s '
                         'scratch_slots=4 cb_bytes=7\n' % (flags, batches, -(-rows // 32), capacity // 32,
                                                           'true' if share else 'false')).encode())
            if 'no_f22' not in self.broken:
                os.write(1, ('Op | INFO | [QWEN-SDPA] runtime-extent entries=%d kv_share=%s q_slice=%s '
                             'writer=writer_decode_qwen_slice.cpp cur_pos_stick_bytes=64\n'
                             % (batches, 'true' if share else 'false',
                                'true' if flags & card_b.SLICE else 'false')).encode())
        self.calls += 1
        q = self.read(query).float()
        keys, values = self.read_float(k), self.read_float(v)
        cores = model.cores_per_head(batches)
        snapshot = self.read(cur_pos_tensor).clone()
        output = self.allocate((1, batches, rows, card.HEAD_DIM), 'bf16')
        if output.address not in self.memory or tuple(self.memory[output.address].shape) != output.shape:
            self.memory[output.address] = torch.zeros(output.shape, dtype=torch.bfloat16)

        def write(replay=False):
            data = self.memory[output.address].clone()
            current = snapshot if (replay and 'stale_trace' in self.broken) else self.read(cur_pos_tensor)
            words = [int(value) for value in current.tolist()]
            mask = self.mask_seen(self.read(attn_mask), flags, batches, capacity)   # re-read at every replay
            skipped = []
            for entry in range(batches):
                word = words[entry if (not share or 'own_slot' in self.broken) else 0]
                if word == -1:
                    skipped.append(entry)
                    continue
                if 'stale_writer' in self.broken and model.stale_writer_hangs(word, capacity, cores):
                    self.hang()
                if 'ignore_word' in self.broken:
                    word = capacity - 1
                if 'clamp' in self.broken and word % 256 == 0:
                    word -= 1
                result = self.attend(q[0, entry], keys, values, table[0 if share else entry], word, k_chunk // 32,
                                     scale, cores, False, mask[entry, 0], True, capacity)
                data[0, entry] = result.to(torch.bfloat16)
            if 'skip_bleeds' in self.broken:
                for entry in skipped:
                    if entry + 1 < batches and entry + 1 not in skipped:
                        data[0, entry + 1] = 0
            self.memory[output.address] = data

        if self.capturing is not None:
            self.traces[self.capturing].append(write)
        else:
            write()
        self.now += 20e-6 + 1e-9 * capacity if self.seconds_per_call is None else self.seconds_per_call
        return output


class DryRunTests(unittest.TestCase):
    """The harness end to end on the fake: a 4,352-key table (17 chunks, more than the 16 cores per head), extents
    512 / 2,304 / 4,352. One FULL run covers every section and combo; each broken variant runs only the section
    that must catch it, on the fewest combos that show it (the CPU job's 10-minute budget)."""

    BASE = ['--capacity', '4352', '--extents', '512,2304,4352', '--starts', '7,255', '--seeds', '0',
            '--trace-families', '4', '--trace-references', '2', '--no-timing']

    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        graft = make_graft(self.dir)
        self.kernels = graft / 'sdpa_decode' / 'device' / 'kernels'
        self.binary = graft / '_ttnncpp.so'

    def tearDown(self):
        self.tmp.cleanup()

    def run_card(self, fake, extra=(), name='card', scratch='1', expect=None):
        out = self.dir / ('%s.json' % name)
        fake.report_path = out
        markers = dict(flags=True, share=True, stage1=False)
        argv = ['--out', str(out), '--kernel-root', str(self.kernels),
                '--expect-binary-sha256', sha(self.binary.read_bytes()) if expect is None else expect]
        environ = {card.SCRATCH_ENV: scratch} if scratch is not None else {}
        with mock.patch.dict(sys.modules, {'ttnn': fake}), \
                mock.patch.object(card, 'loaded_binary', return_value=(str(self.binary), markers)), \
                mock.patch.dict(os.environ, environ), \
                mock.patch.object(probe, 'clock', fake.clock), \
                mock.patch.object(card, 'WATCHDOG', card.WATCHDOG), mock.patch.object(probe, 'WATCHDOG', probe.WATCHDOG), \
                mock.patch.object(probe.k1, 'WATCHDOG', probe.k1.WATCHDOG), \
                mock.patch.object(probe, 'DEADLINE', probe.DEADLINE), \
                mock.patch.object(probe, 'print', create=True), mock.patch.object(card_b, 'print', create=True), \
                mock.patch('sys.stdout'):
            if scratch is None:
                os.environ.pop(card.SCRATCH_ENV, None)
            status = card_b.main(argv + self.BASE + list(extra))
        return status, json.loads(out.read_text())

    def kinds(self, report):
        return {kind: (row['equal'], row['runs']) for kind, row in report['tally'].items()}

    def test_pass_end_to_end(self):
        fake = FakeExtentTtnn(self.torch)
        status, report = self.run_card(fake)
        self.assertEqual((report.get('error'), report['failures'], report['warnings']), (None, [], []))
        self.assertEqual((status, report['passed'], report['decision']['verdict']), (0, True, 'PASS'))
        self.assertTrue(report['binary']['k64j'])
        self.assertEqual(report['sections_done'], ['N/seed0', 'X/seed0', 'M/seed0', 'K/seed0', 'L/seed0', 'T/seed0'])
        kinds = self.kinds(report)
        self.assertEqual(kinds['refusal'], (7, 7))
        self.assertTrue(all(outcome['refused'] and outcome['matched'] for outcome in report['refusals'].values()),
                        report['refusals'])
        # X: 14 entries per (extent, start) over the six combos, 3 extents x 2 starts.
        self.assertEqual(kinds['extent_vs_reference'], (84, 84))
        self.assertEqual(kinds['mixed_vs_reference'], (6, 6))
        self.assertEqual(kinds['share_slot0'], (4, 4))                    # G4B3 0x23, G8B2 0x23 / 0x27 / 0x2F
        self.assertEqual(kinds['skip_live'], (4, 4))
        self.assertEqual(kinds['share_skip_slot0_wins'], (1, 1))
        self.assertEqual((report['skip_all_call'], report['share_skip_slot0'], report['skip_idle_call']),
                         ('returned', 'returned', 'returned'))
        self.assertEqual(report['skip_written'], 'unwritten')
        self.assertEqual(kinds['trace_vs_eager'], (8, 8))                 # 4 replays x 2 trace combos
        self.assertEqual(kinds['trace_vs_reference'], (4, 4))             # 2 sampled replays x 2
        self.assertEqual(kinds['trace_skip_live'], (4, 4))                # the non-share combo only
        self.assertEqual(kinds['trace_fence_vs_clean'], (12, 12))
        self.assertEqual(report['fence_extents'], {'G4B3:0x21': [512, 2304, 512], 'G8B2:0x27': [512, 512]})
        # Liveness: L (per entry of each combo at 512 and 2,304) and the two fences.
        self.assertTrue(report['liveness'] and all(entry['live'] for entry in report['liveness']))
        self.assertEqual(len(report['liveness']), 2 * (3 + 3 + 2 + 2 + 2 + 2) + 3 + 2)
        # Every 0x20 program logged F4 then F22; the references logged F4 only.
        self.assertEqual(len(report['extent_lines']), len({(key[0], key[1], key[2]) for key in map(tuple, report[
            'requested_programs']) if key[0] & 0x20}))
        self.assertIn([0x2F, 2, 4352 // 32, 8], report['requested_programs'])
        self.assertIn([0x7, 2, 2304 // 32, 8], report['requested_programs'])
        self.assertEqual(report['not_run'], card_b.NOT_RUN)
        self.assertTrue(report['verdict_line'].startswith(
            'K64J_CARD verdict=PASS extent=84/84 mixed=6/6 share_slot0=5/5 trace=12/12 fence=12/12 skip=8/8 '
            'refusals=7/7 live=33/33 skipped_rows=unwritten'), report['verdict_line'])
        self.assertEqual((fake.closed, fake.live_traces_at_close), (True, 0))

    def test_a_program_that_ignores_the_word_fails(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'ignore_word'}),
                                       ['--sections', 'X', '--combos', 'G4B3:0x21', '--starts', '7'])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertEqual(self.kinds(report)['extent_vs_reference'], (3, 9))       # only E = C is right

    def test_share_entries_on_their_own_slot_fail(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'own_slot'}),
                                       ['--sections', 'M,K', '--combos', 'G4B3:0x21,G4B3:0x23', '--starts', '7'])
        self.assertEqual((report['failures'], report['decision']['verdict']), ([], 'FAIL'))
        self.assertEqual(self.kinds(report)['share_slot0'], (0, 1))
        self.assertEqual(self.kinds(report)['share_skip_slot0_wins'], (0, 1))
        self.assertEqual(self.kinds(report)['mixed_vs_reference'], (3, 3))        # no share: unaffected

    def test_a_skip_that_disturbs_a_live_entry_fails(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'skip_bleeds'}),
                                       ['--sections', 'K', '--combos', 'G4B3:0x21'])
        self.assertEqual(report['decision']['verdict'], 'FAIL')
        self.assertLess(self.kinds(report)['skip_live'][0], 4)

    def test_a_trace_that_keeps_the_captured_word_fails(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'stale_trace'}),
                                       ['--sections', 'T', '--trace-combos', 'G4B3:0x21'])
        self.assertEqual(report['decision']['verdict'], 'FAIL')
        self.assertEqual(self.kinds(report)['trace_vs_eager'], (1, 4))            # only the capture's own words
        # The fence replays stay inside the captured families (the same words, only the mask moves): equal.
        self.assertEqual(self.kinds(report)['trace_fence_vs_clean'], (6, 6))

    def test_a_dead_liveness_control_decides_nothing(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'clamp'}),
                                       ['--sections', 'L', '--combos', 'G8B2:0x27'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))
        self.assertIn('liveness controls did not move', report['decision']['reasons'][0])

    def test_a_binary_that_never_logs_f22_was_not_executed(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'no_f22'}),
                                       ['--sections', 'X', '--combos', 'G4B3:0x21', '--extents', '2304',
                                        '--starts', '7'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))
        self.assertEqual(report['failures'], ['extent log: flags=0x21 B=3 St=136: 0 F22 lines, expected one'])
        status, report = self.run_card(FakeExtentTtnn(self.torch, silent=True),
                                       ['--sections', 'X', '--combos', 'G4B3:0x21', '--extents', '2304', '--starts', '7'],
                                       name='silent')
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertEqual(len(report['failures']), 2)                              # the 0x21 and the 0x1 program
        self.assertTrue(all(failure.startswith('factory log [X]: ') and 'graft mounted, not executed' in failure
                            for failure in report['failures']), report['failures'])

    def test_a_binary_that_accepts_the_unknown_flag_control_fails(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'accept_0x11'}), ['--sections', 'N'])
        self.assertEqual((report['failures'], report['decision']['verdict']), ([], 'FAIL'))
        self.assertEqual(self.kinds(report)['refusal'], (6, 7))
        self.assertFalse(report['refusals']['unknown flag 0x10 (0x11)']['refused'])

    def test_the_binary_the_kernels_and_the_scratch_are_checked_first(self):
        fake = FakeExtentTtnn(self.torch)
        status, report = self.run_card(fake, name='sha', expect='f' * 64)
        self.assertIn('not the expected ffffffffffffffff', report['failures'][0])
        self.assertEqual((status, report['comparisons'], fake.calls), (1, [], 0))
        status, report = self.run_card(FakeExtentTtnn(self.torch), name='noexpect', expect='')
        self.assertIn('no --expect-binary-sha256', report['failures'][0])
        self.binary.write_bytes(b'K64i ' + card_b.SLICE_BINARY_MARKER + b' ' + card_b.SCRATCH_BINARY_MARKER)
        status, report = self.run_card(FakeExtentTtnn(self.torch), name='k64i')
        self.assertIn('the loaded _ttnncpp.so is not K64j', report['failures'][0])
        self.binary.write_bytes(BINARY)
        kernel = self.kernels / 'dataflow' / 'writer_decode_qwen_slice.cpp'
        kernel.write_bytes(kernel.read_bytes() + b'// K64i\n')
        status, report = self.run_card(FakeExtentTtnn(self.torch), name='kernel')
        self.assertIn('not the k64j %s' % kernels.OUTPUTS[kernels.WRITER_SLICE][:16], report['failures'][0])
        shutil.copyfile(HERE / 'kernels' / kernels.WRITER_SLICE, kernel)
        status, report = self.run_card(FakeExtentTtnn(self.torch), name='scratch', scratch=None)
        self.assertIn('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is required', report['failures'][0])
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')

    def test_the_deadline_stops_cleanly_and_lists_the_rest(self):
        fake = FakeExtentTtnn(self.torch, seconds_per_call=1.0)
        status, report = self.run_card(fake, ['--sections', 'N,X,K', '--combos', 'G4B3:0x21', '--extents', '2304',
                                              '--starts', '7', '--deadline-s', '3'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))
        self.assertEqual(report['sections_done'], ['N/seed0', 'X/seed0'])
        self.assertEqual(report['deadline']['skipped'], ['K/seed0'])
        self.assertTrue(fake.closed)

    def test_the_timing_is_recorded(self):
        base, self.BASE = self.BASE, [arg for arg in self.BASE if arg != '--no-timing']
        try:
            status, report = self.run_card(FakeExtentTtnn(self.torch), ['--sections', 'N', '--extents', '2304,4352',
                                                                         '--warmup', '0', '--iters', '2', '--rounds', '1'])
        finally:
            self.BASE = base
        self.assertEqual((status, report['sections_done']), (0, ['N/seed0', 'timing/seed0']))
        self.assertEqual([row['name'] for row in report['timing']['rows']],
                         ['runtime E2304', 'compile E2304', 'runtime E4352', 'compile E4352', 'skip0', 'skip3'])
        self.assertEqual(sorted(report['timing']['runtime_over_compile']), ['E2304', 'E4352'])
        self.assertEqual(sorted(report['timing']['skip_us']), ['skip0', 'skip3'])

    def test_the_sigterm_handler_writes_a_no_decision_report(self):
        written = []
        report = dict(comparisons=[probe.comparison('X', 'extent_vs_reference', 'x', 0, True)], liveness=[],
                      failures=[])
        handler = card_b.term_handler(report, written.append)
        with self.assertRaises(probe.Terminated):
            handler(15, None)
        self.assertEqual(written[0]['in_progress'], 'terminated')
        self.assertEqual(written[0]['decision']['verdict'], 'NO-DECISION')
        self.assertTrue(written[0]['verdict_line'].startswith('K64J_CARD verdict=NO-DECISION'))
        handler(15, None)                                   # a second signal is ignored
        self.assertEqual(len(written), 1)


if __name__ == '__main__':
    unittest.main()
