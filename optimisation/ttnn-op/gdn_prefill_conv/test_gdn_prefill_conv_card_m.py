"""CPU tests for the card-M harness of gdn_prefill_conv_exact: its case matrix, data kinds, byte
comparison and verdict helpers, the section drivers (equality, chain, negative controls, the
denormal-setting selection) against a torch-only fake Bench, and run_card_m.sh (the qualification card, the
pinned image, the op mounted file by file from the one table). The device half runs on the rig only."""

import contextlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'scripts' / 'ci'))

import torch  # noqa: E402

import gdn_prefill_conv_card_m as card  # noqa: E402
import gdn_prefill_conv_exact as pcx  # noqa: E402

SCRIPT = HERE / 'run_card_m.sh'


def script_text():
    return SCRIPT.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))


class MatrixTests(unittest.TestCase):
    def test_every_valid_len_carry_and_t_is_covered(self):
        cases = card.matrix()
        for T in card.TS:
            for vl in card.valid_lens_for(T):
                for carry in card.CARRIES:
                    with self.subTest(T=T, vl=vl, carry=carry):
                        self.assertIn(dict(T=T, valid_len=vl, carry=carry, data='randn', seed=0), cases)
        self.assertEqual(card.valid_lens_for(32), [1, 2, 3, 31, 32, None])
        self.assertEqual(card.valid_lens_for(2048)[-3:], [2047, 2048, None])

    def test_every_data_kind_and_seed_at_2048(self):
        cases = card.matrix()
        for data in card.DATA:
            for seed in card.SEEDS:
                self.assertTrue(any(c['data'] == data and c['seed'] == seed and c['T'] == 2048 for c in cases), (data, seed))
        names = [card.case_name(c) for c in cases]
        self.assertEqual(len(names), len(set(names)))

    def test_quick_is_a_subset(self):
        quick, full = card.matrix(quick=True), card.matrix()
        self.assertLess(len(quick), len(full))
        self.assertTrue(all(case in full for case in quick))
        self.assertTrue(any(c['T'] == 2048 and c['valid_len'] == 1288 for c in quick))


class DataTests(unittest.TestCase):
    def bits(self, kind):
        return card.make_x(torch, kind, 64, 96, 0).view(torch.int16).int() & 0xFFFF

    def test_shapes_and_dtype(self):
        for kind in card.DATA:
            x = card.make_x(torch, kind, 64, 96, 1)
            self.assertEqual((tuple(x.shape), x.dtype), ((64, 96), torch.bfloat16), kind)
            self.assertTrue(torch.isfinite(x.float()).all(), kind)

    def test_signed_zeros_denormals_and_near_max_are_really_there(self):
        zeros = self.bits('zeros')
        self.assertTrue(((zeros == 0x8000).sum() > 0) and ((zeros == 0).sum() > 0))
        denormal = self.bits('denormal')
        self.assertGreater(int((((denormal & 0x7F80) == 0) & ((denormal & 0x7F) != 0)).sum()), 100)
        self.assertGreater(float(card.make_x(torch, 'nearmax', 64, 96, 0).float().abs().min()), 1e38)
        carry = card.make_carry(torch, 'special', 96, 0).view(torch.int16).int() & 0xFFFF
        self.assertTrue((carry == 0x8000).any())
        self.assertTrue((((carry & 0x7F80) == 0) & ((carry & 0x7F) != 0)).any())
        self.assertIsNone(card.make_carry(torch, 'none', 96, 0))
        self.assertTrue(torch.equal(card.make_carry(torch, 'zeros', 96, 0), torch.zeros(3, 96, dtype=torch.bfloat16)))

    def test_deterministic_per_seed(self):
        self.assertTrue(torch.equal(card.make_x(torch, 'ties', 32, 32, 4), card.make_x(torch, 'ties', 32, 32, 4)))
        self.assertFalse(torch.equal(card.make_x(torch, 'randn', 32, 32, 4), card.make_x(torch, 'randn', 32, 32, 5)))


class CompareTests(unittest.TestCase):
    def test_minus_zero_is_a_difference_and_ulps_are_counted(self):
        a = torch.tensor([1.0, 0.0, 2.0], dtype=torch.bfloat16)
        b = torch.tensor([1.0, -0.0, 2.0], dtype=torch.bfloat16)
        result = card.compare(torch, a, b)
        self.assertEqual((result['exact'], result['differing']), (False, 1))
        self.assertEqual(result['max_ulp'], 0)      # +-0 are 0 ulps apart, yet a byte difference
        self.assertTrue(card.compare(torch, a, a.clone())['exact'])
        c = torch.tensor([1.0, 0.0, 2.015625], dtype=torch.bfloat16)   # the next bf16 after 2.0
        self.assertEqual(card.compare(torch, a, c)['max_ulp'], 1)
        self.assertFalse(card.compare(torch, a, a[:2])['exact'])

    def test_shifted_reference(self):
        x = torch.arange(10 * 2, dtype=torch.float32).reshape(10, 2).to(torch.bfloat16)
        carry = torch.full((3, 2), -1.0, dtype=torch.bfloat16)
        shifted = card.shifted_reference(torch, x, carry)
        self.assertTrue(torch.equal(shifted[:3], carry))
        self.assertTrue(torch.equal(shifted[3:], x[:7]))
        self.assertTrue(torch.equal(card.shifted_reference(torch, x, None)[:3], torch.zeros(3, 2, dtype=torch.bfloat16)))

    def test_denormal_settings_per_state_path(self):
        self.assertEqual(card.denormal_settings(pcx, None),
                         [(0, None), (0, False), (0, True), (1, None), (1, False), (1, True),
                          (2, None), (2, False), (2, True)])
        self.assertEqual(card.denormal_settings(pcx, 7),
                         [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True)])
        with mock.patch.object(pcx, 'STATIC_CANON_DEFAULT', True):
            self.assertEqual(card.module_setting(pcx, None), (pcx.FLUSH_X_DENORM_DEFAULT, pcx.CANON_DENORM_DEFAULT))
        with mock.patch.object(pcx, 'STATIC_CANON_DEFAULT', False):
            self.assertEqual(card.module_setting(pcx, None), (pcx.FLUSH_X_DENORM_DEFAULT, None))
            self.assertEqual(card.module_setting(pcx, 7), (pcx.FLUSH_X_DENORM_DEFAULT, pcx.CANON_DENORM_DEFAULT))
        self.assertEqual(card.setting_variant((1, None)), dict(flush_x_denorm=1, static_canon=False))
        self.assertEqual(card.setting_variant((2, True)), dict(flush_x_denorm=2, canon_denorm=True, static_canon=True))
        self.assertEqual((card.state_path(None), card.state_path(2048)), ('static', 'one_hot'))
        self.assertIn('zeros', card.DENORMAL_CASES)     # -0 data runs the search too

    def test_the_defaults_are_what_card_m_measured(self):
        """pcx-20260923T063828: both state paths canonicalise (-0 and denormals -> +0); x itself is
        not flushed (q/k/v exact with flush 0)."""
        self.assertEqual((pcx.FLUSH_X_DENORM_DEFAULT, pcx.CANON_DENORM_DEFAULT, pcx.STATIC_CANON_DEFAULT),
                         (0, True, True))

    def test_settings_choice(self):
        self.assertEqual(card.settings_choice([], (0, False))['verdict'], 'untested')
        both = [[0, False], [0, True]]
        self.assertEqual(card.settings_choice([both, [[0, False]]], (0, False)),
                         dict(verdict='keep', consistent=[[0, False]]))
        self.assertEqual(card.settings_choice([both, [[0, True]]], (0, False)),
                         dict(verdict='change', consistent=[[0, True]]))
        self.assertEqual(card.settings_choice([[[0, True]], [[0, False]]], (0, False))['verdict'], 'inconsistent')
        self.assertEqual(card.settings_choice([[]], (0, None))['verdict'], 'inconsistent')

    def test_recommend_is_one_flush_and_one_canon_for_both_paths(self):
        raw_static = dict(verdict='change', consistent=[[1, None]])
        one_hot = dict(verdict='change', consistent=[[0, True], [1, False], [1, True]])
        self.assertEqual(card.recommend(raw_static, one_hot), [[1, False, False], [1, True, False]])
        untested = dict(verdict='untested', consistent=[])
        self.assertEqual(card.recommend(untested, dict(verdict='keep', consistent=[[2, False]])),
                         [[2, False, False], [2, False, True]])
        self.assertEqual(len(card.recommend(untested, untested)), 12)
        self.assertEqual(card.recommend(raw_static, dict(verdict='keep', consistent=[[0, False]])), [])
        # The card-M finding: both paths canonicalise with denormals.
        canonical = dict(verdict='keep', consistent=[[0, True]])
        self.assertEqual(card.recommend(canonical, canonical), [[0, True, True]])
        # A static path that canonicalises -0 only cannot share CANON_DENORM with a one-hot that flushes.
        self.assertEqual(card.recommend(dict(verdict='change', consistent=[[0, False]]), canonical), [])

    def test_verdict_needs_cases_and_no_failures(self):
        self.assertFalse(card.verdict(dict(failures=[], cases_run=0)))
        self.assertFalse(card.verdict(dict(failures=['x'], cases_run=3)))
        self.assertTrue(card.verdict(dict(failures=[], cases_run=3)))

    def test_args(self):
        args = card.parse_args(['--out', 'x.json', '--ts', '32,64', '--sections', 'equality,chain'])
        self.assertEqual((args.ts, args.sections, args.op_dir.as_posix()), ([32, 64], ['equality', 'chain'], '/bench/pcx'))
        with self.assertRaises(SystemExit):
            card.parse_args(['--out', 'x.json', '--sections', 'bogus'])
        with self.assertRaises(SystemExit):
            card.parse_args(['--out', 'x.json', '--ts', '48'])

    def test_the_fir_pin_and_the_header_list(self):
        self.assertEqual(card.FIR_SHA256[:8], 'fac29122')      # the 0648ca9a image's FIR
        self.assertTrue(card.SHA_FILES[0].endswith('ttnn_gated_deltanet.py'))
        self.assertEqual((card.C, card.KD), (5120, 1024))

    def test_no_device_import_at_module_level(self):
        source = (HERE / 'gdn_prefill_conv_card_m.py').read_text(encoding='utf-8')
        head = source[:source.index('# Device part: the qualification card only.')]
        self.assertIsNone(re.search(r'^(import|from) (ttnn|torch)', head, re.M))


# ---------------------------------------------------------------------------------------------
# The section drivers against a torch-only Bench: an identity "conv" (q/k/v are x's columns, new_state
# x_padded[vl:vl+3]) is enough to see every byte the harness compares. The FIR side can flush
# denormals in its round trip and canonicalise its one-hot state; the candidate honours the op's
# flush_x_denorm / canon_denorm / negative arguments and can shift one output by a row.
# ---------------------------------------------------------------------------------------------

NAMES = ('q', 'k', 'v', 'new_state')


def flush_bits(tensor, mode):
    if not mode:
        return tensor
    word = tensor.contiguous().view(torch.int16).int() & 0xFFFF
    denormal = ((word & 0x7F80) == 0) & ((word & 0x7F) != 0)
    out = torch.where(denormal, (word & 0x8000) if mode == 2 else torch.zeros_like(word), word)
    return torch.where(out >= 0x8000, out - 0x10000, out).to(torch.int16).view(torch.bfloat16)


def canonical_state(state, denorm):
    """-0 -> +0 and, with denorm, every exponent-zero value -> +0 (the reader's canonical_half)."""
    word = state.contiguous().view(torch.int16).int() & 0xFFFF
    zero = word == 0x8000
    if denorm:
        zero = zero | ((word & 0x7F80) == 0)
    return torch.where(zero, torch.zeros_like(word), word).to(torch.int16).view(torch.bfloat16)


class FakeBench(card.Bench):
    """The FIR side defaults to what card M measured (pcx-20260923T063828): x unflushed, and the
    state canonicalised with denormals on both paths (fir_static False: a raw static state)."""

    def __init__(self, shift=None, negative_changes=True, fir_flush=0, fir_canon=True, fir_static=True):
        super().__init__(types.SimpleNamespace(L1_MEMORY_CONFIG='L1', DRAM_MEMORY_CONFIG='DRAM'), torch, pcx,
                         None, None)
        self.shift, self.negative_changes = shift, negative_changes
        self.fir_flush, self.fir_canon, self.fir_static = fir_flush, fir_canon, fir_static
        self.variants = []

    def upload(self, host, memory_config=None):
        return host.clone()

    def read(self, tensor):
        return tensor

    def inputs(self, x, carry, taps):
        return x.clone(), None if carry is None else carry.clone(), [taps[j:j + 1].clone() for j in range(card.K)]

    def release(self, *tensors):
        pass

    @staticmethod
    def outputs(qkv, carry, valid_len, flush, canon, static):
        T, width = qkv.shape
        pad = carry if carry is not None else torch.zeros(3, width, dtype=qkv.dtype)
        padded = flush_bits(torch.cat([pad, qkv], 0), flush)
        state = padded[(T if valid_len is None else valid_len):][:3].clone()
        if valid_len is not None or static:
            state = canonical_state(state, canon)
        x = padded[3:]
        return [x[:, :card.KD].clone(), x[:, card.KD:2 * card.KD].clone(), x[:, 2 * card.KD:].clone(), state]

    def reference(self, qkv, carry, taps, valid_len):
        return tuple(self.outputs(qkv, carry, valid_len, self.fir_flush, self.fir_canon, self.fir_static))

    def candidate(self, qkv, carry, taps, valid_len, **variant):
        self.variants.append(dict(variant))
        flush = variant.get('flush_x_denorm', pcx.FLUSH_X_DENORM_DEFAULT)
        canon = variant.get('canon_denorm', pcx.CANON_DENORM_DEFAULT)
        static = variant.get('static_canon', pcx.STATIC_CANON_DEFAULT)
        outs = self.outputs(qkv, carry, valid_len, flush, canon, static)
        if variant.get('negative') is not None:
            if self.negative_changes:
                outs[0] = torch.roll(outs[0], 1, 0)
            return tuple(outs)
        if self.shift is not None:
            index = NAMES.index(self.shift)
            outs[index] = torch.roll(outs[index], 1, 0)
        return tuple(outs)


def section(name, bench, cases=None):
    report = dict(failures=[], cases=[], negative={}, chain=[], timing={}, denormal_settings={})
    args = types.SimpleNamespace(ts=[32], seeds=[0], quick=False, mirror=False)
    patch = mock.patch.object(card, 'matrix', lambda ts, seeds, quick=False: cases) if cases is not None \
        else contextlib.nullcontext()
    with patch, contextlib.redirect_stdout(io.StringIO()):
        if name == 'equality':
            card.equality_cases(bench, args, report)
        else:
            report['cases_run'] = 1
            dict(chain=card.chain, negative=card.negative_controls)[name](bench, args, report)
    return card.verdict(report), report


class DriverTests(unittest.TestCase):
    def test_an_exact_candidate_passes_equality_and_chain(self):
        for name in ('equality', 'chain'):
            with self.subTest(section=name):
                passed, report = section(name, FakeBench())
                self.assertTrue(passed, report['failures'])
        passed, report = section('equality', FakeBench())
        self.assertEqual(report['cases_run'], len(card.matrix([32], [0])))
        self.assertEqual(report['denormal_choice']['one_hot']['verdict'], 'keep')

    def test_a_one_row_shift_of_any_output_fails_equality_and_chain(self):
        for name in ('equality', 'chain'):
            for output in NAMES:
                with self.subTest(section=name, output=output):
                    passed, report = section(name, FakeBench(shift=output))
                    self.assertFalse(passed)
                    self.assertTrue(report['failures'])
                    if name == 'chain':
                        self.assertTrue(all(not step['exact'] for entry in report['chain'] for step in entry['steps']))

    def test_every_plain_case_that_differs_is_its_own_failure(self):
        cases = [dict(T=64, valid_len=vl, carry=carry, data='randn', seed=0)
                 for vl in (None, 33) for carry in ('none', 'randn')]
        for output in NAMES:
            with self.subTest(output=output):
                passed, report = section('equality', FakeBench(shift=output), cases)
                self.assertFalse(passed)
                self.assertEqual(len(report['failures']), len(cases))
                self.assertTrue(all(output in failure for failure in report['failures']))
                self.assertEqual(report['denormal_choice']['static']['verdict'], 'untested')

    def test_negative_controls_must_change_the_output(self):
        passed, report = section('negative', FakeBench())
        self.assertTrue(passed, report['failures'])
        self.assertTrue(all(entry['differs'] for entry in report['negative'].values()))
        self.assertEqual(sorted(report['negative']), sorted(card.NEGATIVES))
        passed, report = section('negative', FakeBench(negative_changes=False))
        self.assertFalse(passed)
        self.assertEqual(len(report['failures']), len(card.NEGATIVES))

    def denormal_cases(self):
        return [dict(T=64, valid_len=vl, carry=carry, data='denormal', seed=0)
                for vl in (None, 33, 1) for carry in ('randn', 'special')]

    def test_every_denormal_case_tries_every_setting_on_both_paths(self):
        bench = FakeBench()
        passed, report = section('equality', bench, self.denormal_cases())
        self.assertTrue(passed, report['failures'])
        self.assertEqual(len(report['denormal_settings']['static']), 2)
        self.assertEqual(len(report['denormal_settings']['one_hot']), 4)
        tried = [v for v in bench.variants if 'flush_x_denorm' in v]
        self.assertEqual(len(tried), 2 * 8 + 4 * 5)          # 9 static and 6 one-hot settings, less the default
        self.assertEqual(report['denormal_choice']['static']['verdict'], 'keep')
        self.assertEqual(report['denormal_choice']['one_hot']['verdict'], 'keep')

    def test_a_fir_that_flushes_x_is_detected_and_the_flush_named(self):
        for mode in (1, 2):
            with self.subTest(fir_flush=mode):
                passed, report = section('equality', FakeBench(fir_flush=mode), self.denormal_cases())
                self.assertFalse(passed)
                choice = report['denormal_choice']
                self.assertEqual((choice['static']['verdict'], choice['static']['module_default'],
                                  choice['static']['cases']), ('change', [0, True], 2))
                self.assertTrue(choice['static']['consistent'])
                self.assertTrue(all(flush == mode for flush, _ in choice['static']['consistent']))
                self.assertEqual(choice['one_hot']['verdict'], 'change')
                self.assertTrue(choice['recommend'])
                self.assertTrue(all(flush == mode for flush, _, _ in choice['recommend']))
                self.assertTrue(any('static path: the module default [0, True]' in f for f in report['failures']))
                self.assertFalse(any('no flush / canonicalisation setting matches' in f for f in report['failures']))
                # With the named setting as the module default, the same FIR passes.
                flush, canon, static = choice['recommend'][0]
                with mock.patch.object(pcx, 'FLUSH_X_DENORM_DEFAULT', flush), \
                        mock.patch.object(pcx, 'CANON_DENORM_DEFAULT', canon), \
                        mock.patch.object(pcx, 'STATIC_CANON_DEFAULT', static):
                    passed, report = section('equality', FakeBench(fir_flush=mode), self.denormal_cases())
                self.assertTrue(passed, report['failures'])

    def test_a_state_that_keeps_denormals_is_the_canon_choice(self):
        passed, report = section('equality', FakeBench(fir_canon=False), self.denormal_cases())
        self.assertFalse(passed)
        self.assertEqual(report['denormal_choice']['one_hot']['verdict'], 'change')
        self.assertIn([0, False], report['denormal_choice']['one_hot']['consistent'])
        self.assertIn([0, False, True], report['denormal_choice']['recommend'])

    def test_a_raw_static_state_is_the_static_canon_choice(self):
        passed, report = section('equality', FakeBench(fir_static=False), self.denormal_cases())
        self.assertFalse(passed)
        choice = report['denormal_choice']
        self.assertEqual(choice['static']['verdict'], 'change')
        self.assertIn([0, None], choice['static']['consistent'])
        self.assertEqual(choice['one_hot']['verdict'], 'keep')
        self.assertEqual(choice['recommend'], [[0, True, False]])
        self.assertTrue(any('STATIC_CANON_DEFAULT' in f for f in report['failures']))

    def test_minus_zero_data_on_the_static_path_selects_a_setting(self):
        """The case pcx-20260923T063828 could not settle: data=zeros at valid_len None put -0 in the
        static state and the FIR gave +0. It now runs the search and the defaults match it."""
        cases = [dict(T=64, valid_len=None, carry=carry, data='zeros', seed=0) for carry in ('randn', 'special')]
        passed, report = section('equality', FakeBench(), cases)
        self.assertTrue(passed, report['failures'])
        self.assertEqual(report['denormal_choice']['static']['verdict'], 'keep')
        with mock.patch.object(pcx, 'STATIC_CANON_DEFAULT', False):
            passed, report = section('equality', FakeBench(), cases)
        self.assertFalse(passed)
        self.assertEqual(report['denormal_choice']['static']['module_default'], [0, None])
        self.assertIn([0, True], report['denormal_choice']['static']['consistent'])

    def test_a_shifted_denormal_case_matches_no_setting_and_fails_by_name(self):
        passed, report = section('equality', FakeBench(shift='new_state'), self.denormal_cases())
        self.assertFalse(passed)
        self.assertEqual(sum('no flush / canonicalisation setting matches' in f for f in report['failures']), 6)


class RunScriptTests(unittest.TestCase):
    def test_the_qualification_card_the_pinned_image_and_the_refusal(self):
        text = script_text()
        self.assertNotIn('CARD_M=', text)
        self.assertIn('QUAL_CARD_B=blackhole-F36F768B9A5CAFA0', text)          # the embedded qual_card.sh block
        self.assertIn('R=${RESULTS:-$HOME/kwork64/pcx/$QUAL_TAG}', text)
        self.assertIn('IMAGE=${IMAGE:-sha256:0648ca9ad663acc72e7d8ea59d9cde0f9218b583ad74a60d58ff2f31bddd6fae}', text)
        self.assertIn('--device "$node"', text)
        self.assertEqual(text.count('--device '), 1)
        self.assertIn('\nqual_card_resolve\nnode=$QUAL_NODE\nqual_refuse_holders\n', text)
        self.assertLess(text.index('\nqual_refuse_holders\n'), text.index('docker run --rm'))
        self.assertIn('qual_reset_hint >&2', text[text.index('docker run --rm'):])
        self.assertNotIn('tt-smi -r', [line.strip() for line in text.splitlines() if not line.lstrip().startswith(('#', 'echo', '"'))])

    def test_the_op_is_mounted_file_by_file_from_the_table(self):
        text = script_text()
        self.assertIn('p.PREFILL_CONV_FILES', text)
        self.assertIn('OM+=(--mount "type=bind,src=$REPO/scripts/ci/$file,dst=/bench/pcx/$file,readonly")', text)
        self.assertIn('"${OM[@]}"', text)
        self.assertNotRegex(text, r'dst=/bench/pcx[,"]')
        self.assertNotRegex(text, r'dst=/experiment-scripts')
        self.assertIn('-e TT_METAL_CACHE=/kcache', text)

    def test_the_watcher_pass(self):
        text = script_text()
        self.assertIn('-e TT_METAL_WATCHER=5', text)
        block = text[text.index('if [ "${WATCHER:-}" = "1" ]; then'):]
        block = block[:block.index(chr(10) + 'fi' + chr(10))]
        self.assertIn('timeout_s=900', block)
        self.assertIn('--quick --no-timing', block)

    def test_bash_parses_it_and_the_table_snippet_yields_the_four_files(self):
        bash = shutil.which('bash')
        if bash is None or shutil.which('python3') is None:
            self.skipTest('no bash / python3')
        result = subprocess.run([bash, '-n', str(SCRIPT)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = script_text()
        line = [l for l in text.splitlines() if l.startswith('mapfile -t op_files')][0]
        with tempfile.TemporaryDirectory() as directory:
            ci = Path(directory, 'scripts', 'ci')
            ci.mkdir(parents=True)
            for name in ('lever_n_m3native_patch.py', 'gdn_prefill_conv_exact.py'):
                shutil.copy(ROOT / 'scripts' / 'ci' / name, ci / name)
            script = ('set -euo pipefail' + chr(10) + 'REPO=' + Path(directory).as_posix() + chr(10) + line + chr(10)
                      + 'printf "RESULT|%s" "${op_files[*]}"' + chr(10))
            result = subprocess.run([bash, '-c', script], capture_output=True, text=True, timeout=120, cwd=directory,
                                    env=dict(os.environ))
        if result.returncode != 0 and ('No module named' in result.stderr or 'No such file' in result.stderr):
            self.skipTest('python3 here cannot see the temp tree: %s' % result.stderr.strip())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split('RESULT|')[-1].split(), sorted([
            'gdn_prefill_conv_exact.py', 'gdn_prefill_conv_exact_compute.cpp',
            'gdn_prefill_conv_exact_reader.cpp', 'gdn_prefill_conv_exact_writer.cpp']))


if __name__ == '__main__':
    unittest.main()
