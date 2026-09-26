"""CPU checks for the K64j sources (the runtime extent, flag 0x20); no device, no ttnn, no torch.

  - the generators: the committed K64j kernels are K64i's plus exactly R10 / C3 / W4 (they invert, hash to the
    recorded OUTPUTS, regenerate from the committed K64i kernels and from the sources dump), and the factory is
    the audited 3e0a69af base (vendored in fixtures/) through stages 3 and 4 plus exactly F19-F22: it hashes to
    K64J_FACTORY from the base, stage 3 or stage 4 and inverts to stage 4, 3, 1 and the base. Every K64j anchor
    is K64i's own text. Only qwen-named kernels are edited;
  - the flag mask admits 0x20 and still refuses 0x10 (the card tests' unknown-flag control) and every other bit;
  - F20's requirements, as text: under 0x20 the tail flag, the narrow one-chunk mask, an interleaved int32
    row-major cur_pos tensor of B entries, no causal flag; K64i's refusal literal byte for byte with its condition
    relaxed to 'no cur_pos tensor unless 0x20'; F21's compile-time-arg offsets are the kernels' (reader +4 / +9,
    compute 33, writer +3 / +4) and F21 selects the slice writer for every 0x20 program; F22's literal;
  - the kernels: static_assert(!runtime_extent || mask_tail); the cur_pos read under `is_causal ||
    runtime_extent` in all three; both CB copies pushed before the skip test, and the UINT32_MAX return before
    any Q, K or V read and before the KV-share handshake; under share every entry uses slot 0; the writer never
    runs generate_mask under runtime_extent; each cur_pos block is the stock block plus exactly those changes;
  - the reductions: at runtime_extent = 0 every K64j kernel is its K64i base as code; at q_slice = 0 the slice
    writer (W1-W4) IS the stock writer_decode_all.cpp as code (runtime_extent = 0), and under 0x20 the stock
    writer taking the causal block's read and never generating a mask; W3 is write_partial_tiles_to_memory with
    the L1 rebase only, and the one statement it drops is dead;
  - compiled with a host g++ (CI's ubuntu runner; locally set QWEN_PF_GXX, or run this module in WSL): W3 at
    q_tile_start 0 writes exactly what the stock function writes; the three K64j cur_pos blocks run in an
    emulated L1 against the vendored rt_args_common.hpp (every kernel of an entry agrees on its word, slot 0 under
    share, the skip, no consumer waits on a copy never pushed, the split at E - 1 is the compile-time call's at
    E); the factory's own qwen blocks run in a stub create_descriptor (refusals, compile-time args, kernel
    selection, the F4 / F18 / F22 lines, cb_bytes growing by two sticks);
  - the stub syntax compile of the K64j readers and writer at every buildable 0x20 flag set, and the
    static_asserts that must fire (needs any g++ plus the sources dump and the prefill tree; skips otherwise).

    py -3.11 -B -m unittest test_k64j_sources          (from this directory)
    python3 -B -m unittest test_k64j_sources           (WSL or Linux, for the compiled checks)
"""

import difflib
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
ROOT = HERE.parents[2]
QWEN = OPS / 'sdpa_decode_qwen'
SLICE = OPS / 'sdpa_decode_slice'
PROBE = OPS / 'k64j_probe'
for _path in (str(HERE), str(QWEN), str(SLICE), str(PROBE), str(SLICE / 'stubcheck')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import apply_factory_k64j as factory  # noqa: E402
import apply_factory_qwen as qwen_factory  # noqa: E402
import apply_factory_slice as slice_factory  # noqa: E402
import make_k64j_kernels as kernels  # noqa: E402
import make_qwen_kernels as qwen_kernels  # noqa: E402
import make_slice_kernels as slice_kernels  # noqa: E402
import split_model as model  # noqa: E402

NL = chr(10)
FIXTURES = HERE / 'fixtures'
FACTORY_BASE = FIXTURES / 'sdpa_decode_program_factory.3e0a69af.cpp'
PROBE_FIXTURES = PROBE / 'fixtures'
STOCK = {
    'reader': PROBE_FIXTURES / 'reader_decode_all.49a05926.cpp',
    'writer': PROBE_FIXTURES / 'writer_decode_all.734c90c0.cpp',
    'compute': PROBE_FIXTURES / 'sdpa_flash_decode.d24769bd.cpp',
    'common': PROBE_FIXTURES / 'dataflow_common.e4623a22.hpp',
    'rt_args': PROBE_FIXTURES / 'rt_args_common.1b52c60d.hpp',
}
KERNEL_DIR = HERE / 'kernels'
JOB = Path('C:/Users/liamb/.claude/jobs/8376c877/tmp')
DUMP = Path(os.environ.get('QWEN_SDPA_SOURCES_DUMP', JOB / 'sdpa-decode-sources.txt'))
PREFILL_SRC = Path(os.environ.get('QWEN_SDPA_PREFILL_SRC', JOB / 'probe-v25' / 'src' / 'device'))
ARM = ROOT / 'scripts' / 'ci' / 'lever_n_m3native_run_arm.sh'
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'
SPLIT_START = '    auto Sk_chunk_t_dynamic'
COMPUTE_SPLIT = '    // Get dynamic chunk size'
UINT32_MAX = 0xFFFFFFFF


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return Path(path).read_text(encoding='utf-8')


def k64j(name):
    return read(KERNEL_DIR / name)


def k64i(name):
    return read(kernels.COMMITTED[name])


_FACTORY_TEXT = []


def factory_text():
    """The K64j factory, generated from the vendored base (cached)."""
    if not _FACTORY_TEXT:
        _FACTORY_TEXT.append(factory.patch(FACTORY_BASE.read_bytes()).decode('utf-8'))
    return _FACTORY_TEXT[0]


def cur_pos_block(text, end):
    start = text.index('    // Get cur_pos')
    return text[start:text.index(end, start)]


def edit_new(label):
    for name, _old, new in factory.K64J_EDITS:
        if name == label:
            return new
    raise KeyError(label)


# ---------------------------------------------------------------------------------------------
# The kernel generator.
# ---------------------------------------------------------------------------------------------

class KernelGeneratorTests(unittest.TestCase):
    def test_the_committed_kernels_are_k64is_plus_exactly_the_edits(self):
        self.assertEqual(sorted(p.relative_to(KERNEL_DIR).as_posix() for p in KERNEL_DIR.rglob('*') if p.is_file()),
                         sorted(kernels.KERNELS))
        for name in kernels.KERNELS:
            with self.subTest(kernel=name):
                built = (KERNEL_DIR / name).read_bytes()
                self.assertEqual(sha(built), kernels.OUTPUTS[name])
                self.assertNotIn(b'\r', built)
                reverted = kernels.revert_edits(name, built)
                self.assertEqual(sha(reverted), kernels.BASES[name])
                self.assertEqual(reverted, kernels.COMMITTED[name].read_bytes())

    def test_the_bases_are_the_k64i_generators_recorded_outputs(self):
        self.assertEqual(kernels.BASES, {
            'dataflow/reader_decode_qwen.cpp': '280a847fae833891dffff1057d67b999288a183386cd058e3e1614755ce3499b',
            'dataflow/reader_decode_qwen_slice.cpp': '0f5a019ccc06ca603bb4ed44c77cc66f9cb3e5bd35193810eb127f13c9f5631f',
            'compute/sdpa_flash_decode_qwen.cpp': '8776fcc7420c6f27a9c7ae06c54c391225a00ce78322c5397970d74a5063ca8a',
            'dataflow/writer_decode_qwen_slice.cpp': 'ac6cf815c34df85a9d39593d95f28eb2da0cb5b37c232633e65bf3ed116925f4'})
        for name, path in kernels.COMMITTED.items():
            self.assertEqual(sha(path.read_bytes()), kernels.BASES[name], name)

    def test_the_edits_apply_to_the_committed_k64i_kernels_and_refuse_anything_else(self):
        for name in kernels.KERNELS:
            with self.subTest(kernel=name):
                base = kernels.COMMITTED[name].read_bytes()
                self.assertEqual(kernels.apply_edits(name, base), (KERNEL_DIR / name).read_bytes())
                with self.assertRaisesRegex(ValueError, 'base is'):
                    kernels.apply_edits(name, base + b'// drifted\n')
                with self.assertRaisesRegex(ValueError, 'base is'):
                    kernels.apply_edits(name, (KERNEL_DIR / name).read_bytes())    # never applied twice

    def test_every_anchor_occurs_once_in_its_base_and_every_edit_once_in_its_output(self):
        for name, edits in kernels.EDITS.items():
            base, built = k64i(name), k64j(name)
            for label, old, new in edits:
                with self.subTest(kernel=name, edit=label):
                    self.assertEqual(base.count(old), 1)
                    self.assertEqual(built.count(new), 1)

    def test_main_checks_the_committed_kernels_and_catches_a_drifted_copy(self):
        self.assertEqual(kernels.main(['--check', str(KERNEL_DIR)]), 0)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(kernels.main(['--out', directory]), 0)
            for name in kernels.KERNELS:
                self.assertEqual(Path(directory, name).read_bytes(), (KERNEL_DIR / name).read_bytes())
            Path(directory, kernels.WRITER_SLICE).write_bytes(b'x')
            self.assertEqual(kernels.main(['--check', directory]), 1)

    def test_the_kernels_regenerate_from_the_stock_originals(self):
        if not DUMP.is_file():
            self.skipTest('no sources dump at %s' % DUMP)
        self.assertEqual(kernels.main(['--dump', str(DUMP), '--check', str(KERNEL_DIR)]), 0)
        text = DUMP.read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for key, dump_path in (('reader', qwen_kernels.DUMP_PATHS[qwen_kernels.READER_NAME]),
                                   ('writer', slice_kernels.DUMP_WRITER),
                                   ('compute', qwen_kernels.DUMP_PATHS[qwen_kernels.COMPUTE_NAME])):
                paths[key] = Path(directory, key + '.cpp')
                paths[key].write_bytes(qwen_kernels.from_dump(text, dump_path))
            self.assertEqual(kernels.main(['--reader', str(paths['reader']), '--writer', str(paths['writer']),
                                           '--compute', str(paths['compute']), '--check', str(KERNEL_DIR)]), 0)

    def test_the_stock_originals_route_equals_the_vendored_stock_kernels(self):
        """The rig route (make_k64j_kernels --reader/--writer/--compute on ttbuild's originals) from the vendored
        stock kernels, which are the served tt-metal 9f9cd4fd sources, gives the committed kernels."""
        bases = kernels.bases_from(reader=STOCK['reader'], writer=STOCK['writer'], compute=STOCK['compute'])
        for name in kernels.KERNELS:
            self.assertEqual(kernels.apply_edits(name, bases[name]), (KERNEL_DIR / name).read_bytes(), name)

    def test_only_qwen_named_kernels_are_edited(self):
        """K64j never edits a stock kernel or a shared header: the exact profile builds them too, and the arm's
        kernel-cache key hashes only the graft's *qwen*.cpp kernels."""
        for name in kernels.KERNELS:
            self.assertIn('qwen', Path(name).name)
            self.assertTrue(Path(name).name.endswith('.cpp'))
        for stock in ('reader_decode_all.cpp', 'writer_decode_all.cpp', 'sdpa_flash_decode.cpp', 'dataflow_common.hpp',
                      'rt_args_common.hpp'):
            self.assertFalse([name for name in kernels.KERNELS if name.endswith('/' + stock)])
        arm = read(ARM)
        self.assertIn("find . -type f -name '*qwen*.cpp' ! -path ./dataflow/reader_decode_qwen.cpp", arm)
        self.assertIn('cat "$sdpa_kernels/dataflow/reader_decode_qwen.cpp"', arm)
        self.assertIn('"$sdpa_kernels/compute/sdpa_flash_decode_qwen.cpp"', arm)

    def test_the_cache_key_changes_with_every_k64j_kernel(self):
        """The arm's key: the stage-3 pair's bytes, then '<sha> ./<path>' per further *qwen*.cpp in byte order
        (lever_n_m3native_run_arm.sh). All four kernels differ from K64i's, so K64j's key is never K64i's."""
        def key(source):
            others = ''.join('%s ./%s\n' % (sha(source(name).encode()), name)
                             for name in sorted((kernels.READER_SLICE, kernels.WRITER_SLICE)))
            return sha(source(kernels.READER_QWEN).encode() + source(kernels.COMPUTE_QWEN).encode()
                       + others.rstrip(NL).encode())[:12]
        self.assertNotEqual(key(k64j), key(k64i))
        for name in kernels.KERNELS:
            self.assertNotEqual(k64j(name), k64i(name))


# ---------------------------------------------------------------------------------------------
# The factory generator.
# ---------------------------------------------------------------------------------------------

class FactoryGeneratorTests(unittest.TestCase):
    def test_the_fixture_is_the_audited_base(self):
        data = FACTORY_BASE.read_bytes()
        self.assertEqual(sha(data), factory.BASE_FACTORY)
        self.assertEqual(factory.BASE_FACTORY, '3e0a69af9563ae8899db1363286d6dc6cdc1744e0154887dc91a630b16084a4a')
        self.assertTrue(data.startswith(b'// SPDX-FileCopyrightText: \xc2\xa9 2024 Tenstorrent USA, Inc.'))
        self.assertIn(b'SPDX-License-Identifier: Apache-2.0', data[:200])
        self.assertNotIn(b'\r', data)

    def test_every_k64j_anchor_is_k64is_own_text(self):
        """Each anchor or replaced text is a K64i generator's output text, so the edit list is checkable without
        the base: F19 after F13's tag, F19's mask is F15's, F19's declaration after F15's, F20 replaces F2's
        refusal, F21 follows F16's reader suffix, replaces F16's writer suffix, follows F7's compute arg and
        replaces F17's writer selection, F22 follows F18."""
        cases = (('F19', slice_factory.F13), ('F19 mask', slice_factory.F15_NEW), ('F19 decl', slice_factory.F15_DECL),
                 ('F20', qwen_factory.F2), ('F21 reader', slice_factory.F16R), ('F21 writer', slice_factory.F16W),
                 ('F21 compute', qwen_factory.F7), ('F21 kernel', slice_factory.F17W_NEW), ('F22', slice_factory.F18))
        edits = {label: (old, new) for label, old, new in factory.K64J_EDITS}
        self.assertEqual(sorted(edits), sorted(label for label, _source in cases))
        for label, source in cases:
            with self.subTest(edit=label):
                self.assertIn(edits[label][0], source)
        self.assertEqual(factory.F21_WRITER_OLD + '', slice_factory.F16W[len(slice_factory.F16W_ANCHOR):])

    def test_anything_but_the_base_stage_three_or_stage_four_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'unexpected factory'):
            factory.patch(b'not a factory')
        with self.assertRaisesRegex(ValueError, 'unexpected factory'):
            factory.patch(factory_text().encode())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'f.cpp')
            path.write_bytes(b'nope')
            self.assertEqual(factory.main([str(path), '--out', str(Path(directory, 'o.cpp'))]), 1)

    def test_the_base_and_stages_three_and_four_patch_to_the_recorded_bytes_and_invert(self):
        base = FACTORY_BASE.read_bytes()
        stage3 = qwen_factory.patch(base, 3)
        stage4 = slice_factory.patch(base)
        self.assertEqual(sha(stage3), factory.STAGE3_FACTORY)
        self.assertEqual(sha(stage4), factory.STAGE4_FACTORY)
        for source in (base, stage3, stage4):
            built = factory.patch(source)
            self.assertEqual(sha(built), factory.K64J_FACTORY)
            self.assertEqual(factory.unpatch(built), stage4)
            self.assertEqual(factory.unpatch(built, to_stage=3), stage3)
            self.assertEqual(sha(factory.unpatch(built, to_stage=1)), qwen_factory.QWEN_FACTORY)
            self.assertEqual(factory.unpatch(built, to_stage=0), base)

    def test_main_writes_the_recorded_factory_keeps_a_backup_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory, 'k64j.cpp')
            self.assertEqual(factory.main([str(FACTORY_BASE), '--out', str(out)]), 0)
            self.assertEqual(sha(out.read_bytes()), factory.K64J_FACTORY)
            self.assertEqual(factory.main([str(out), '--out', str(Path(directory, 'again.cpp'))]), 0)
            self.assertEqual(Path(directory, 'again.cpp').read_bytes(), out.read_bytes())
            in_place = Path(directory, 'sdpa_decode_program_factory.cpp')
            in_place.write_bytes(slice_factory.patch(FACTORY_BASE.read_bytes()))
            self.assertEqual(factory.main([str(in_place)]), 0)
            self.assertEqual(sha(in_place.read_bytes()), factory.K64J_FACTORY)
            backup = Path(directory, 'sdpa_decode_program_factory.cpp.orig-' + factory.STAGE4_FACTORY[:8])
            self.assertEqual(sha(backup.read_bytes()), factory.STAGE4_FACTORY)

    def test_the_markers_and_every_edit_once(self):
        text = factory_text()
        for label, _old, new in factory.K64J_EDITS:
            self.assertEqual(text.count(new), 1, label)
        for marker in factory.K64J_MARKERS:
            self.assertIn(marker, text)
        for marker in factory.K64J_ABSENT:
            self.assertNotIn(marker, text)
        self.assertNotIn(qwen_factory.STAGE1_SHARE_REFUSAL, text)
        self.assertNotIn(chr(13), text)


# ---------------------------------------------------------------------------------------------
# The flag.
# ---------------------------------------------------------------------------------------------

def constants(text):
    return {name: int(value, 16) for name, value in re.findall(r'constexpr (?:uint32_t|std::size_t) (kQwen\w+) = (0x[0-9A-Fa-f]+)u;', text)}


class FlagTests(unittest.TestCase):
    def test_the_mask_admits_0x20_and_still_refuses_0x10(self):
        text = factory_text()
        values = constants(text)
        mask = re.search(r'TT_FATAL\(\(qwen_flags & ~\(([^)]*)\)\) == 0,\s*"\[QWEN-SDPA\] unknown flags \{:#x\}"', text)
        self.assertIsNotNone(mask)
        admitted = 0
        for name in (part.strip() for part in mask.group(1).split('|')):
            admitted |= values[name]
        self.assertEqual(admitted, factory.KNOWN_FLAGS)
        self.assertEqual(admitted, 0x2F)
        refused = lambda flags: (flags & ~admitted) != 0  # noqa: E731 - the factory's own test
        self.assertFalse(refused(0x20))
        self.assertFalse(refused(0x21) or refused(0x23) or refused(0x27) or refused(0x2F))
        self.assertTrue(refused(0x10))
        self.assertTrue(refused(0x11))
        self.assertTrue(refused(0x31))
        for bit in (0x40, 0x80):
            self.assertTrue(refused(bit))
        self.assertEqual(text.count('"[QWEN-SDPA] unknown flags {:#x}"'), 1)

    def test_0x20_is_k64js_alone_and_0x10_no_ones(self):
        text = factory_text()
        values = constants(text)
        self.assertEqual(values['kQwenRuntimeExtent'], 0x20)
        self.assertEqual(factory.FLAG_EXTENT, 0x20)
        earlier = constants(read(QWEN / 'apply_factory_qwen.py')) | constants(read(SLICE / 'apply_factory_slice.py'))
        flag_values = {value for name, value in earlier.items() if name not in ('kQwenMagic', 'kQwenMagicMask', 'kQwenSliceAbiTag')}
        self.assertEqual(sorted(flag_values), [0x1, 0x2, 0x4, 0x8])
        self.assertNotIn(0x10, set(values.values()))
        self.assertEqual(factory.UNKNOWN_FLAG_CONTROL, 0x10)
        card = read(QWEN / 'test_sdpa_decode_qwen_card_m.py')
        self.assertIn('UNKNOWN_FLAG = MAGIC | 0x10', card)
        self.assertIn("('unknown flag 0x10', 8, 2, 0x11,", read(SLICE / 'sdpa_decode_slice_card_b.py'))


# ---------------------------------------------------------------------------------------------
# F20-F22 as text.
# ---------------------------------------------------------------------------------------------

def preconditions(text):
    start = text.index('    // ========== [QWEN-SDPA] preconditions ==========')
    return text[start:text.index('    // ========== Tree Reduction Setup ==========', start)]


class FactoryTextTests(unittest.TestCase):
    def test_f20_relaxes_k64is_refusal_to_no_cur_pos_tensor_unless_0x20(self):
        text = preconditions(factory_text())
        self.assertIn('TT_FATAL(!is_causal && (!use_cur_pos_tensor || qwen_runtime_extent) && sliding_window_size == 0,', text)
        self.assertNotIn('TT_FATAL(!is_causal && !use_cur_pos_tensor && sliding_window_size == 0,', text)
        # K64i's literal byte for byte: every QWEN string of K64i stays in the binary (build step 6), and the probe's
        # causal control still finds its needle.
        self.assertEqual(text.count('"%s"' % factory.CUR_POS_REFUSAL), 1)
        self.assertIn('"%s"' % factory.CUR_POS_REFUSAL, qwen_factory.F2)
        needle = re.search(r"^CAUSAL_NEEDLE = '([^']*)'", read(PROBE / 'probe_k64j_card_b.py'), flags=re.M).group(1)
        self.assertIn(needle, factory.CUR_POS_REFUSAL)
        # Causal stays refused for every qwen mode, 0x20 included: !is_causal is not relaxed.
        self.assertEqual(text.count('!is_causal'), 1)

    def test_f20_requires_tail_the_narrow_mask_and_an_interleaved_int32_row_major_word_per_entry(self):
        text = preconditions(factory_text())
        block = text[text.index('        if (qwen_runtime_extent) {'):]
        block = block[:block.index('\n        }\n') + len('\n        }\n')]
        conditions = re.findall(r'TT_FATAL\((.*?),\s*"(\[QWEN-SDPA\][^"]*)"', block, flags=re.S)
        self.assertEqual([' '.join(condition.split()) for condition, _literal in conditions], [
            'qwen_mask_tail && use_attention_mask && qwen_mask_width_t == Sk_chunk_t',
            'use_cur_pos_tensor && !is_cur_pos_tensor_sharded',
            'cur_pos_tensor->dtype() == DataType::INT32 && cur_pos_tensor->layout() == Layout::ROW_MAJOR && '
            'cur_pos_tensor->padded_shape()[-1] == B'])
        self.assertEqual([literal.split(' {}')[0].split('{}')[0] for _condition, literal in conditions],
                         [factory.EXTENT_MASK_REFUSAL, factory.EXTENT_CUR_POS_REFUSAL, factory.EXTENT_LAYOUT_REFUSAL])
        # The tensor is dereferenced only after its presence is checked; the block is inside `if (qwen_mode) {`.
        self.assertLess(block.index('use_cur_pos_tensor && !is_cur_pos_tensor_sharded'), block.index('cur_pos_tensor->'))
        self.assertLess(text.index('    if (qwen_mode) {'), text.index('        if (qwen_runtime_extent) {'))
        # The narrow mask is Sk_chunk_t wide: the tail check already admits it ("full width or one chunk").
        self.assertIn('(qwen_mask_width_t == St || qwen_mask_width_t == Sk_chunk_t)', text)
        # qwen_runtime_extent is declared before the checks, from the flag alone (share is not required).
        self.assertIn('const bool qwen_runtime_extent = (qwen_flags & kQwenRuntimeExtent) != 0;', text)
        self.assertLess(text.index('const bool qwen_runtime_extent'), text.index('    if (qwen_mode) {'))

    def test_f21_hands_the_word_to_every_kernel_at_the_offsets_they_read(self):
        text = factory_text()
        # Reader: F6's four, then (under 0x4 or 0x8) F16's five, then runtime_extent, pushed in every qwen program.
        f6 = re.findall(r'reader_compile_time_args_common\.push_back\(([^;]+)\);', qwen_factory.F6[len(qwen_factory.F6_ANCHOR):])
        f16 = re.findall(r'reader_compile_time_args_common\.push_back\(([^;]+)\);', slice_factory.F16R[len(slice_factory.F16R_ANCHOR):])
        f21 = re.findall(r'reader_compile_time_args_common\.push_back\(([^;]+)\);', edit_new('F21 reader')[len(factory.F21_READER_ANCHOR):])
        self.assertEqual((len(f6), len(f16), f21), (4, 5, ['static_cast<uint32_t>(qwen_runtime_extent)']))
        self.assertIn('    if (qwen_mode) {', edit_new('F21 reader'))
        self.assertEqual(kernels.READER_EXTENT_OFFSET, {kernels.READER_QWEN: len(f6), kernels.READER_SLICE: len(f6) + len(f16)})
        self.assertEqual(factory.READER_EXTENT_OFFSET, {'reader_decode_qwen.cpp': 4, 'reader_decode_qwen_slice.cpp': 9})
        for name, offset in kernels.READER_EXTENT_OFFSET.items():
            self.assertEqual(k64j(name).count('get_compile_time_arg_val(qwen_cta + %d) == 1;' % offset), 1, name)
            self.assertIn('constexpr bool runtime_extent = get_compile_time_arg_val(qwen_cta + %d) == 1;' % offset, k64j(name))
        # The reader suffix pushed in file order: F6, F16 (conditional), F21.
        order = [text.index(marker) for marker in ('reader_compile_time_args_common.push_back(kv_ready_semaphore_id);',
                                                   'reader_compile_time_args_common.push_back(kQwenSliceAbiTag);',
                                                   'reader_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_runtime_extent));')]
        self.assertEqual(order, sorted(order))
        # Compute: CTA 32 (mask_tail) then 33.
        compute = re.findall(r'compute_compile_time_args_common\.push_back\(([^;]+)\);\s*// index (\d+)', text)
        self.assertEqual(compute, [('static_cast<uint32_t>(qwen_mask_tail)', '32'),
                                   ('static_cast<uint32_t>(qwen_runtime_extent)', '33')])
        self.assertIn('constexpr bool runtime_extent = get_compile_time_arg_val(%d) == 1;' % factory.COMPUTE_EXTENT_CTA,
                      k64j(kernels.COMPUTE_QWEN))
        # Writer: K64i's three, then q_slice and runtime_extent, for every 0x4 OR 0x20 program.
        writer = re.findall(r'writer_compile_time_args_common\.push_back\(([^;]+)\);\s*// \+(\d)', edit_new('F21 writer'))
        self.assertEqual(writer, [('qwen_pnht_full', '0'), ('qwen_rows_per_kv', '1'), ('kQwenSliceAbiTag', '2'),
                                  ('static_cast<uint32_t>(qwen_q_slice)', '3'),
                                  ('static_cast<uint32_t>(qwen_runtime_extent)', '4')])
        self.assertEqual(factory.WRITER_SUFFIX, ('pnht_full', 'rows_per_kv', 'tag', 'q_slice', 'runtime_extent'))
        self.assertIn('    if (qwen_q_slice || qwen_runtime_extent) {', edit_new('F21 writer'))
        writer_kernel = k64j(kernels.WRITER_SLICE)
        for index, name in enumerate(factory.WRITER_SUFFIX):
            if name != 'tag':
                self.assertRegex(writer_kernel, r'constexpr \w+ %s = get_compile_time_arg_val\(qwen_slice_cta \+ %d\)' % (name, index))
        self.assertIn('static_assert(get_compile_time_arg_val(qwen_slice_cta + 2) == 0x51CE', writer_kernel)

    def test_f21_selects_the_slice_writer_for_every_0x20_program(self):
        text = factory_text()
        self.assertIn('(qwen_q_slice || qwen_runtime_extent) ? "dataflow/writer_decode_qwen_slice.cpp"', text)
        self.assertNotIn('qwen_q_slice ? "dataflow/writer_decode_qwen_slice.cpp"', text)
        # The readers and compute kernel are selected as K64i selects them: every qwen program runs qwen kernels.
        self.assertIn(slice_factory.F17R_NEW, text)
        self.assertIn(qwen_factory.F8C_NEW, text)
        # c_8 and c_15 exist whenever a cur_pos tensor is passed (the base factory, unchanged), so F4's cb_bytes
        # (the sum over every CB) grows by two sticks for every 0x20 program.
        self.assertIn('    if (use_cur_pos_tensor) {', text)
        cbs = text[text.index('    if (use_cur_pos_tensor) {'):text.index('    if (is_paged_attention) {', text.index('    if (use_cur_pos_tensor) {'))]
        self.assertIn('CBIndex::c_8,\n            cur_pos_stick_size,', cbs)
        self.assertIn('add_cb(CBIndex::c_15, cur_pos_stick_size, cur_pos_df, cur_pos_stick_size);', cbs)
        self.assertIn('for (const auto& cb : desc.cbs) {', text)

    def test_f22_logs_one_line_per_0x20_program_with_its_binary_literal(self):
        new = edit_new('F22')
        self.assertIn('    if (qwen_runtime_extent) {', new)
        self.assertEqual(new.count('log_info('), 1)
        literal = re.search(r'"(\[QWEN-SDPA\] runtime-extent [^"]*)"', new).group(1)
        self.assertTrue(literal.startswith(factory.EXTENT_LOG_MARKER))
        self.assertEqual(factory.EXTENT_LOG_MARKER, '[QWEN-SDPA] runtime-extent entries=')
        self.assertEqual(literal.format(3, 'true', 'false', 64), '[QWEN-SDPA] runtime-extent entries=3 kv_share=true q_slice=false '
                         'writer=writer_decode_qwen_slice.cpp cur_pos_stick_bytes=64')
        # No K64i format literal changed: the F4 'flags=' line is apply_factory_qwen's, F18's is the stage-4 one.
        self.assertEqual(factory_text().count('[QWEN-SDPA] flags={:#x}'), 1)
        self.assertNotIn('flags=', ''.join(new for _label, _old, new in factory.K64J_EDITS))


# ---------------------------------------------------------------------------------------------
# R10, C3 and W4 as text.
# ---------------------------------------------------------------------------------------------

READS = {
    kernels.READER_QWEN: 'cur_pos = index_ptr[cur_batch / q_heads_parallel_factor];',
    kernels.READER_SLICE: 'cur_pos = index_ptr[cur_batch / q_heads_parallel_factor];',
    kernels.WRITER_SLICE: 'cur_pos = index_ptr[(uint32_t)(cur_batch / q_heads_parallel_factor)];',
    kernels.COMPUTE_QWEN: 'cur_pos = read_tile_value(cb_cur_pos, 0, cur_batch / q_heads_parallel_factor);',
}
ENDS = {kernels.READER_QWEN: SPLIT_START, kernels.READER_SLICE: SPLIT_START, kernels.WRITER_SLICE: SPLIT_START,
        kernels.COMPUTE_QWEN: COMPUTE_SPLIT}
STOCK_ROLE = {kernels.READER_QWEN: 'reader', kernels.READER_SLICE: 'reader', kernels.WRITER_SLICE: 'writer',
              kernels.COMPUTE_QWEN: 'compute'}


class KernelTextTests(unittest.TestCase):
    def test_r10_asserts_the_tail_mask_and_reads_its_offset(self):
        for name in (kernels.READER_QWEN, kernels.READER_SLICE):
            with self.subTest(kernel=name):
                text = k64j(name)
                self.assertEqual(text.count('static_assert(!runtime_extent || mask_tail, '), 1)
                self.assertIn('static_assert(!runtime_extent || mask_width_t == Sk_chunk_t, ', text)
                self.assertIn('static_assert(!runtime_extent || !is_causal, ', text)
                self.assertIn('static_assert(!runtime_extent || !is_cur_pos_tensor_sharded, ', text)
                # runtime_extent is read after every other suffix arg the kernel reads.
                offsets = [int(v) for v in re.findall(r'get_compile_time_arg_val\(qwen_cta \+ (\d+)\)', text)]
                self.assertEqual(max(offsets), kernels.READER_EXTENT_OFFSET[name])
                self.assertEqual(sorted(set(offsets)), list(range(kernels.READER_EXTENT_OFFSET[name] + 1)))
        self.assertIn('static_assert(!runtime_extent || (mask_tail && !is_causal), ', k64j(kernels.COMPUTE_QWEN))
        self.assertIn('static_assert(!runtime_extent || !is_causal, ', k64j(kernels.WRITER_SLICE))

    def test_every_kernel_reads_the_word_under_is_causal_or_runtime_extent(self):
        for name, read_line in READS.items():
            with self.subTest(kernel=name):
                block = cur_pos_block(k64j(name), ENDS[name])
                self.assertEqual(block.count('if constexpr (is_causal || runtime_extent) {'), 1)
                self.assertNotIn('if constexpr (is_causal) {', block)
                self.assertLess(block.index('if constexpr (is_causal || runtime_extent) {'), block.index(read_line))
                self.assertLess(block.index(read_line), block.index('if (cur_pos == UINT32_MAX) {'))
                self.assertIn('return;', block[block.index('if (cur_pos == UINT32_MAX) {'):])
                self.assertIn('constexpr uint32_t cur_pos_base = St * 32 - 1;', block)
                # The block padding conversion stays causal-only (qwen mode refuses block padding anyway).
                self.assertIn('if constexpr (has_block_padding && is_causal) {', block)
            # Nothing else in the kernel reads the word or branches on runtime_extent at run time.
            self.assertEqual(k64j(name).count(read_line), 1)

    def test_the_skip_returns_before_any_q_k_or_v_read_and_before_the_share_handshake(self):
        for name in (kernels.READER_QWEN, kernels.READER_SLICE):
            with self.subTest(kernel=name):
                text = k64j(name)
                start = text.index('    // Get cur_pos')
                skip = text.index('if (cur_pos == UINT32_MAX) {', start)
                skip_return = text.index('return;', skip)
                for first in ('read_q<', 'read_k<', 'read_v<', 'read_page_table_for_batch(', 'read_mask_chunk<',
                              'kv_ready.wait(', 'Semaphore<>(kv_ready_semaphore_id).up(', 'kv_valid.wait(', 'kv_valid.set(',
                              'noc.async_write_multicast(', 'read_kv_mask_chunks<'):
                    if first in text:
                        self.assertLess(skip_return, text.index(first), first)
                # Both CB copies are pushed before the skip test, so the writer and compute always find their word.
                self.assertLess(text.index('cb_writer.push_back(1);', start), skip)
                self.assertLess(text.index('cb_compute.push_back(1);', start), skip)
                # The only earlier return is the idle core's (q_addr == 0), before any CB traffic.
                self.assertEqual(text[:start].count('return;'), 1)
                self.assertIn('if (q_addr == 0) {\n        return;', text[:start])

    def test_under_share_every_entry_uses_slot_0(self):
        for name in (kernels.READER_QWEN, kernels.READER_SLICE):
            with self.subTest(kernel=name):
                block = cur_pos_block(k64j(name), SPLIT_START)
                share = block[block.index('if constexpr (runtime_extent && kv_share) {'):block.index('cb_writer.push_back(1);')]
                self.assertIn('const uint32_t slot0 = writer_words[0];', share)
                self.assertIn('writer_words[cur_batch / q_heads_parallel_factor] = slot0;', share)
                self.assertIn('compute_words[cur_batch / q_heads_parallel_factor] = slot0;', share)
                self.assertIn('reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr)', share)
                self.assertIn('reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_compute_wr_ptr)', share)
                # After both L1 copies landed (the L1 -> L1 read and its barrier), before either copy is pushed.
                self.assertLess(block.index('noc.async_read_barrier();', block.index('UnicastEndpoint pos_src;')),
                                block.index('if constexpr (runtime_extent && kv_share) {'))
                # The reader's own word is then slot 0's too: it reads its slot of the rewritten c_8 copy.
                self.assertLess(block.index('if constexpr (runtime_extent && kv_share) {'), block.index(READS[name]))
        # The stock path (causal, no extent) keeps each entry's own slot: the share block is compiled out.
        self.assertNotIn('slot0', cur_pos_block(read(STOCK['reader']), SPLIT_START))

    def test_the_writer_never_runs_generate_mask_under_runtime_extent(self):
        text = k64j(kernels.WRITER_SLICE)
        calls = [m.start() for m in re.finditer(r'generate_mask<', text)]
        self.assertEqual(len(calls), 1)
        opener = text.rindex('if constexpr (', 0, calls[0])
        condition = text[opener:text.index('{', opener)]
        self.assertEqual(condition.strip(), 'if constexpr (is_causal && !runtime_extent)')
        self.assertNotIn('}', text[opener:calls[0]])                 # the call is inside that branch
        self.assertIn('static_assert(!runtime_extent || !is_causal, ', text)
        # The reader supplies the tail mask instead (narrow, offset 0, final chunk only).
        self.assertIn('static_assert(!runtime_extent || mask_width_t == Sk_chunk_t, ', k64j(kernels.READER_QWEN))

    def test_c3_keeps_the_causal_only_mask_and_the_runtime_tail_predicate(self):
        text = k64j(kernels.COMPUTE_QWEN)
        self.assertIn('const bool apply_mask_at_last_chunk = do_reduce && is_causal;', text)
        self.assertIn('if (!mask_tail || k_chunk == k_num_chunks - 1) {', text)
        self.assertEqual(text.count('runtime_extent'), 3)   # the CTA, its assert, the read condition
        self.assertEqual(k64i(kernels.COMPUTE_QWEN).count('runtime_extent'), 0)

    def test_each_cur_pos_block_is_the_stock_block_plus_exactly_the_k64j_changes(self):
        """The block every K64j kernel runs is the vendored stock kernel's block with the generator's in-block edits
        applied, and nothing else: the read condition, and in the readers the share slot-0 rewrite."""
        in_block = {kernels.READER_QWEN: (('R10 read', kernels.R10_READ_OLD, kernels.R10_READ_NEW),
                                          ('R10 share', kernels.R10_SHARE_OLD, kernels.R10_SHARE_NEW)),
                    kernels.READER_SLICE: (('R10 read', kernels.R10_READ_OLD, kernels.R10_READ_NEW),
                                           ('R10 share', kernels.R10_SHARE_OLD, kernels.R10_SHARE_NEW)),
                    kernels.WRITER_SLICE: (('W4 read', kernels.W4_READ_OLD, kernels.W4_READ_NEW),),
                    kernels.COMPUTE_QWEN: (('C3 read', kernels.C3_READ_OLD, kernels.C3_READ_NEW),)}
        outside = {kernels.READER_QWEN: ['R10 cta'], kernels.READER_SLICE: ['R10 cta'],
                   kernels.WRITER_SLICE: ['W4 cta', 'W4 start', 'W4 mask'], kernels.COMPUTE_QWEN: ['C3 cta']}
        for name, edits in in_block.items():
            with self.subTest(kernel=name):
                block = cur_pos_block(read(STOCK[STOCK_ROLE[name]]), ENDS[name])
                for label, old, new in edits:
                    self.assertEqual(block.count(old), 1, label)
                    block = block.replace(old, new)
                self.assertEqual(block, cur_pos_block(k64j(name), ENDS[name]))
                # The edits outside the block are compile-time only (CTA reads, static_asserts, comments) or, in the
                # writer, W4's q_tile_start and generate_mask condition (the reductions below).
                labels = [edit[0] for edit in edits]
                self.assertEqual([label for label, _old, _new in kernels.EDITS[name] if label not in labels], outside[name])


# ---------------------------------------------------------------------------------------------
# The reductions: what each kernel IS at a given compile-time value.
# ---------------------------------------------------------------------------------------------

def strip_comments(text):
    out, i, n, quote = [], 0, len(text), None
    while i < n:
        c = text[i]
        if quote:
            out.append(c)
            if c == chr(92) and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c == '"':
            quote = c
            out.append(c)
            i += 1
            continue
        if text.startswith('//', i):
            end = text.find(NL, i)
            i = n if end < 0 else end
            continue
        if text.startswith('/*', i):
            end = text.find('*/', i)
            i = n if end < 0 else end + 2
            continue
        out.append(c)
        i += 1
    return ''.join(out)


def code_lines(text):
    """The kernel as code: no comments, no blank lines, no static_assert (compile-time only), stripped."""
    lines_out, pending = [], None
    for line in strip_comments(text).split(NL):
        stripped = line.strip()
        if pending is not None:
            pending += stripped
            if pending.count('(') == pending.count(')') and pending.endswith(';'):
                pending = None
            continue
        if not stripped:
            continue
        if stripped.startswith('static_assert('):
            if not (stripped.count('(') == stripped.count(')') and stripped.endswith(';')):
                pending = stripped
            continue
        lines_out.append(stripped)
    return lines_out


def partial(condition, values):
    """A C++ condition of identifiers, !, && and || under the known values: True, False or the residual text."""
    def term(word):
        negated = word.startswith('!')
        name = word[1:] if negated else word
        if name in values:
            return (not values[name]) if negated else bool(values[name])
        return word
    ors = []
    for disjunct in condition.split('||'):
        ands = [term(word.strip()) for word in disjunct.split('&&')]
        if any(value is False for value in ands):
            continue
        residual = [value for value in ands if value is not True]
        if not residual:
            return True
        ors.append(' && '.join(residual))
    return ' || '.join(ors) if ors else False


def fold(lines, values, counts=None):
    """Specialise code lines at compile-time values: the named CTA reads go, `if constexpr (cond) {` becomes the
    residual condition, `if constexpr (true) {`, or (false) the whole block goes; `q_slice ? a : b` becomes a or b."""
    out, skip_depth, index = [], None, 0
    for line in lines:
        index += 1
        if skip_depth is not None:
            skip_depth += line.count('{') - line.count('}')
            if skip_depth <= 0:
                skip_depth = None
            continue
        match = re.match(r'constexpr bool (\w+) = get_compile_time_arg_val\([^)]*\) == 1;$', line)
        if match and match.group(1) in values:
            continue
        match = re.match(r'if constexpr \((.*)\) \{$', line)
        if match and any(re.search(r'\b%s\b' % name, match.group(1)) for name in values):
            result = partial(match.group(1), values)
            if counts is not None:
                counts[match.group(1)] = counts.get(match.group(1), 0) + 1
            if result is False:
                skip_depth = 1
                continue
            line = 'if constexpr (%s) {' % ('true' if result is True else result)
        for name, value in values.items():
            line = re.sub(r'\b%s \? (.+?) : (.+?);$' % name, lambda m: (m.group(1) if value else m.group(2)) + ';', line)
        out.append(line)
    return out


class ReductionTests(unittest.TestCase):
    def test_at_runtime_extent_0_every_k64j_kernel_is_its_k64i_base_as_code(self):
        for name in kernels.KERNELS:
            with self.subTest(kernel=name):
                counts = {}
                # K64i selects the slice writer only under 0x4, so its K64i program is q_slice 1.
                values = {'runtime_extent': False, 'q_slice': True} if name == kernels.WRITER_SLICE else {'runtime_extent': False}
                reduced = fold(code_lines(k64j(name)), values, counts)
                self.assertEqual(reduced, code_lines(k64i(name)))
                self.assertTrue(counts)

    def slice_writer_at_q_slice_0(self, runtime_extent):
        """The K64j slice writer at q_slice = 0, reduced step by step; each step checks the premise it rests on."""
        text = k64j(kernels.WRITER_SLICE)
        lines = fold(code_lines(text), {'q_slice': False, 'runtime_extent': runtime_extent})
        # W1/W4's compile-time block: the suffix offset and the slice CTAs feed only static_asserts, W1's
        # q_tile_start (0 here) and W2's stride. They go once their uses are folded below.
        self.assertIn('const uint32_t q_tile_start = 0;', lines)
        lines.remove('const uint32_t q_tile_start = 0;')
        # W2: at q_slice 0 PNHt == pnht_full (W4's assert), and out_chunk_tiles is PNHt * vDHt, so the entry's
        # first output tile is the stock writer's.
        self.assertIn('static_assert(q_slice ? PNHt < pnht_full : PNHt == pnht_full, ', text)
        self.assertIn('constexpr uint32_t out_chunk_tiles = PNHt * vDHt;', lines)
        index = lines.index('uint32_t out_tile_id = cur_batch * pnht_full * vDHt;')
        self.assertLess(lines.index('constexpr uint32_t out_chunk_tiles = PNHt * vDHt;'), index)
        lines[index] = 'uint32_t out_tile_id = cur_batch * out_chunk_tiles;'
        # W3 at q_tile_start 0 is write_partial_tiles_to_memory (the line diff and the compiled comparison below),
        # less its trailing out_tile_id increment, which is dead: out_tile_id is re-declared per head and never read
        # after the call. Replace the call and drop the now unused definition.
        call = [i for i, line in enumerate(lines) if line.startswith('barrier_count = write_partial_tiles_sliced<')]
        self.assertEqual(len(call), 1)
        self.assertEqual(lines[call[0] + 1], 'out_tile_id, out_writer, barrier_count, cur_head, num_heads_to_write, out_chunk_tiles, q_tile_start);')
        lines[call[0]] = 'barrier_count = write_partial_tiles_to_memory<cb_out, ELEMENT_SIZE, barrier_threshold, PNHt>('
        lines[call[0] + 1] = 'out_tile_id, out_writer, barrier_count, cur_head, num_heads_to_write, out_chunk_tiles);'
        start = lines.index('template <uint32_t cb_out, uint32_t ELEMENT_SIZE, uint32_t barrier_threshold, uint32_t PNHt, typename WriterType>')
        end = lines.index('void kernel_main() {')
        self.assertEqual(lines[start + 1], 'uint32_t write_partial_tiles_sliced(')
        del lines[start:end]
        for name in ('pnht_full', 'rows_per_kv', 'qwen_slice_cta'):
            uses = [line for line in lines if re.search(r'\b%s\b' % name, line)]
            self.assertTrue(all(line.startswith('constexpr uint32_t %s = ' % name) for line in uses), (name, uses))
            lines = [line for line in lines if line not in uses]
        self.assertFalse([line for line in lines if 'q_tile_start' in line or 'write_partial_tiles_sliced' in line])
        return lines

    def test_the_slice_writer_is_the_stock_writer_at_q_slice_0(self):
        """README (F21): the slice writer's W1-W3 reduce to the stock writer at q_slice = 0. Without 0x20 the
        reduced kernel IS writer_decode_all.cpp (734c90c0) as code."""
        stock = code_lines(read(STOCK['writer']))
        self.assertEqual(sha(STOCK['writer'].read_bytes()), slice_kernels.WRITER_BASE)
        self.assertEqual(self.slice_writer_at_q_slice_0(False), stock)

    def test_under_0x20_the_writer_is_the_stock_writer_on_the_causal_read_without_generate_mask(self):
        """Every non-slice 0x20 program runs the slice writer at q_slice 0, runtime_extent 1: the stock writer
        whose cur_pos block is taken (the causal read of c_8) and whose generate_mask block is gone."""
        stock = code_lines(read(STOCK['writer']))
        branches = [i for i, line in enumerate(stock) if line == 'if constexpr (is_causal) {']
        self.assertEqual(len(branches), 2)
        read_branch, mask_branch = branches
        self.assertEqual(stock[read_branch + 1], 'if (cur_pos_arg != UINT32_MAX) {')
        self.assertEqual(stock[mask_branch + 1:mask_branch + 3],
                         ['generate_mask<cb_mask_in, PNHt>(k_num_chunks, Sk_chunk_t_dynamic, cur_pos);', '}'])
        expected = list(stock)
        expected[read_branch] = 'if constexpr (true) {'
        del expected[mask_branch:mask_branch + 3]
        self.assertEqual(self.slice_writer_at_q_slice_0(True), expected)

    def test_w3_is_write_partial_tiles_to_memory_with_the_l1_rebase_only(self):
        def body(text, name):
            start = text.index('uint32_t %s(' % name)
            end = text.index('\n}\n', start)
            return [line.split('//')[0].strip() for line in text[start:end].split(NL) if line.split('//')[0].strip()]
        stock = body(read(STOCK['common']), 'write_partial_tiles_to_memory')
        ours = body(k64j(kernels.WRITER_SLICE), 'write_partial_tiles_sliced')
        diff = [line for line in difflib.ndiff(stock, ours) if line[:1] in '+-']
        self.assertEqual(sorted(diff), sorted([
            '- uint32_t write_partial_tiles_to_memory(', '+ uint32_t write_partial_tiles_sliced(',
            '- uint32_t out_chunk_tiles) {', '+ uint32_t out_chunk_tiles,', '+ uint32_t q_tile_start) {',
            '- uint32_t tile_index = head_tile * num_hidden_tiles + hidden_tile;',
            '+ const uint32_t l1_tile_index = (head_tile - q_tile_start) * num_hidden_tiles + hidden_tile;',
            '+ const uint32_t dram_tile_id = out_tile_id + head_tile * num_hidden_tiles + hidden_tile;',
            '- uint32_t l1_read_addr_head = l1_base_addr + tile_index * tile_bytes + in_tile_offset;',
            '+ uint32_t l1_read_addr_head = l1_base_addr + l1_tile_index * tile_bytes + in_tile_offset;',
            '- const uint32_t dram_tile_id = out_tile_id + tile_index;', '- out_tile_id += out_chunk_tiles;']), diff)

    def test_the_dropped_out_tile_id_increment_is_dead_in_the_writer(self):
        """write_partial_tiles_to_memory advances out_tile_id after writing; the writer declares out_tile_id per
        head-loop iteration and reads it after the call only in the other (MQA) constexpr branch."""
        lines = code_lines(read(STOCK['writer']))
        declare = lines.index('uint32_t out_tile_id = cur_batch * out_chunk_tiles;')
        call = next(i for i, line in enumerate(lines) if line.startswith('barrier_count = write_partial_tiles_to_memory<'))
        self.assertLess(declare, call)
        after = lines[call + 2:]
        mqa = next(i for i, line in enumerate(after) if line.startswith('barrier_count = write_tiles_to_memory<'))
        self.assertFalse([line for line in after[:mqa] if 'out_tile_id' in line])
        self.assertFalse([line for line in after[mqa + 2:] if 'out_tile_id' in line])
        self.assertIn('ASSERT(num_heads_per_core == 1);', lines)


# ---------------------------------------------------------------------------------------------
# Compiled checks (host g++).
# ---------------------------------------------------------------------------------------------

def find_host_gxx():
    for candidate in (os.environ.get('QWEN_PF_GXX'), shutil.which('g++'), shutil.which('clang++')):
        if candidate and Path(candidate).is_file() and 'arm-none-eabi' not in candidate:
            return candidate
    return None


def compile_and_run(gxx, source, extra_headers=None, timeout=300):
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        (work / 'tt-metalium').mkdir()
        (work / 'tt-metalium' / 'constants.hpp').write_text(
            '#pragma once\n#include <cstdint>\nnamespace tt { namespace constants {\n'
            'constexpr uint32_t TILE_HEIGHT = 32;\nconstexpr uint32_t TILE_WIDTH = 32;\n} }\n')
        for name, text in (extra_headers or {}).items():
            (work / name).write_text(text, encoding='utf-8', newline=NL)
        (work / 'main.cpp').write_text(source, encoding='utf-8', newline=NL)
        binary = work / 'main.exe'
        built = subprocess.run([gxx, '-std=c++20', '-O1', '-I', str(work), str(work / 'main.cpp'), '-o', str(binary)],
                               capture_output=True, text=True, timeout=timeout)
        if built.returncode:
            raise AssertionError('compile failed:\n' + built.stderr[-4000:])
        run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=timeout)
        if run.returncode:
            raise AssertionError('run failed (%d):\n%s' % (run.returncode, run.stderr[-2000:]))
        return run.stdout


def function_text(text, template_line, name):
    start = text.index(template_line + NL + 'uint32_t %s(' % name)
    return text[start:text.index('\n}\n', start) + 3]


W3_PRELUDE = r'''
#include <cstdint>
#include <cstdio>
#include <vector>
struct Write { uint32_t l1; uint32_t page; uint32_t offset; uint32_t bytes; };
static std::vector<Write> WRITES;
static int FLUSHES;
template <typename T> struct CoreLocalMem { uint32_t addr; explicit CoreLocalMem(uint32_t a) : addr(a) {} };
struct NoArgs {};
struct PageOffset { uint32_t page_id; uint32_t offset_bytes; };
struct Writer {};
struct Noc {
    void async_write(CoreLocalMem<uint32_t> src, const Writer&, uint32_t bytes, NoArgs, PageOffset dst) {
        WRITES.push_back({src.addr, dst.page_id, dst.offset_bytes, bytes});
    }
    void async_writes_flushed() { ++FLUSHES; }
};
constexpr uint32_t get_tile_size(uint32_t) { return 2048; }
struct CircularBuffer {
    explicit CircularBuffer(uint32_t) {}
    uint32_t get_read_ptr() const { return 65536; }
};
'''

W3_MAIN = r'''
template <uint32_t PNHt>
static int compare(uint32_t vdht, uint32_t kv_heads, uint32_t per_kv, uint32_t batch) {
    int cases = 0;
    for (uint32_t head = 0; head < kv_heads; ++head) {
        const uint32_t tiles = PNHt * vdht;
        uint32_t stock_tile = batch * tiles, ours_tile = batch * tiles, stock_barrier = 3, ours_barrier = 3;
        WRITES.clear(); FLUSHES = 0;
        uint32_t stock_ret = write_partial_tiles_to_memory<16, 2, 5, PNHt>(stock_tile, Writer{}, stock_barrier, head, per_kv, tiles);
        std::vector<Write> stock = WRITES; int stock_flushes = FLUSHES;
        WRITES.clear(); FLUSHES = 0;
        uint32_t ours_ret = write_partial_tiles_sliced<16, 2, 5, PNHt>(ours_tile, Writer{}, ours_barrier, head, per_kv, tiles, 0);
        bool same = stock.size() == WRITES.size() && stock_ret == ours_ret && stock_barrier == ours_barrier && stock_flushes == FLUSHES;
        for (size_t i = 0; same && i < stock.size(); ++i) {
            same = stock[i].l1 == WRITES[i].l1 && stock[i].page == WRITES[i].page && stock[i].offset == WRITES[i].offset &&
                   stock[i].bytes == WRITES[i].bytes;
        }
        std::printf("C %u %u %u %u %u %d %zu %u %u\n", PNHt, vdht, kv_heads, per_kv, head, same ? 1 : 0, stock.size(),
                    stock_tile - ours_tile, tiles);
        ++cases;
    }
    return cases;
}
int main() {
    const uint32_t vdhts[] = {1, 4, 8};
    for (uint32_t vdht : vdhts) {
        for (uint32_t kv_heads = 1; kv_heads <= 8; kv_heads *= 2) {
            for (uint32_t per_kv = 1; per_kv <= 48; per_kv += 1) {
                const uint32_t rows = kv_heads * per_kv;
                if (rows <= 32) compare<1>(vdht, kv_heads, per_kv, 2);
                if (rows <= 64) compare<2>(vdht, kv_heads, per_kv, 1);
                if (rows <= 96) compare<3>(vdht, kv_heads, per_kv, 3);
                if (rows <= 192) compare<6>(vdht, kv_heads, per_kv, 0);
            }
        }
    }
    return 0;
}
'''

TEMPLATE_LINE = 'template <uint32_t cb_out, uint32_t ELEMENT_SIZE, uint32_t barrier_threshold, uint32_t PNHt, typename WriterType>'

CURPOS_PRELUDE = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <tuple>
#include "rt_args_common.hpp"

#define tt_l1_ptr
struct Hang { uint32_t cb; };
static uint8_t L1[4096];
static uint32_t DRAM_PAGE[64];
static uint32_t CB_BASE[32];
static int PUSHED[32], POPPED[32], RESERVED[32], DRAM_READS, L1_COPIES;
static void reset_core() {
    std::memset(L1, 0xA5, sizeof L1);
    for (int i = 0; i < 32; ++i) { PUSHED[i] = POPPED[i] = RESERVED[i] = 0; CB_BASE[i] = 128u * (uint32_t)i; }
    DRAM_READS = L1_COPIES = 0;
}
static volatile uint32_t* l1_words(uint32_t addr) { return reinterpret_cast<volatile uint32_t*>(L1 + addr); }
template <typename T> struct CoreLocalMem { uint32_t addr; explicit CoreLocalMem(uint32_t a) : addr(a) {} };
struct PageArgs { uint32_t page_id; };
struct UnicastArgs { uint32_t noc_x; uint32_t noc_y; uint32_t addr; };
struct NoArgs {};
struct PosReader {};
struct UnicastEndpoint {};
static PosReader TensorAccessor(int, uint32_t) { return PosReader{}; }
static uint8_t my_x[2] = {1, 1};
static uint8_t my_y[2] = {2, 2};
struct Noc {
    uint8_t get_noc_id() const { return 0; }
    void async_read(const PosReader&, CoreLocalMem<uint32_t> dst, uint32_t bytes, PageArgs src, NoArgs) {
        if (src.page_id != 0) throw Hang{99};
        std::memcpy(L1 + dst.addr, DRAM_PAGE, bytes);
        ++DRAM_READS;
    }
    void async_read(const UnicastEndpoint&, CoreLocalMem<uint32_t> dst, uint32_t bytes, UnicastArgs src, NoArgs) {
        std::memcpy(L1 + dst.addr, L1 + src.addr, bytes);
        ++L1_COPIES;
    }
    void async_read_barrier() {}
};
struct CircularBuffer {
    uint32_t id;
    explicit CircularBuffer(uint32_t index) : id(index) {}
    void reserve_back(int n) { RESERVED[id] += n; }
    uint32_t get_write_ptr() const { return CB_BASE[id]; }
    uint32_t get_read_ptr() const { return CB_BASE[id]; }
    void push_back(int n) { PUSHED[id] += n; }
    void wait_front(int n) { if (PUSHED[id] - POPPED[id] < n) throw Hang{id}; }
    void pop_front(int n) { POPPED[id] += n; }
};
static uint32_t read_tile_value(uint32_t cb, uint32_t tile, uint32_t index) { return l1_words(CB_BASE[cb] + tile * 2048u)[index]; }
struct Outcome { bool skipped; uint32_t cur_pos; };
constexpr uint32_t CAPACITY_ST = 4104;   // 131,328 keys
'''

CURPOS_CONSTANTS = r'''
    constexpr uint32_t St = CAPACITY_ST;
    constexpr uint32_t q_heads_parallel_factor = 1;
    constexpr bool is_cur_pos_tensor_sharded = false;
    constexpr uint32_t index_stick_size_B = 64;
    constexpr uint32_t original_block_size = 64;
    constexpr bool has_block_padding = false;
    constexpr uint32_t cb_writer_cur_pos = 8;
    constexpr uint32_t cb_compute_cur_pos = 15;
    constexpr uint32_t cb_cur_pos = CB_CUR_POS;
    const int pos_args = 0;
    const uint32_t pos_addr = 0;
    Noc noc;
    (void)noc; (void)pos_args; (void)pos_addr; (void)cb_writer_cur_pos; (void)cb_compute_cur_pos; (void)cb_cur_pos;
    (void)index_stick_size_B; (void)original_block_size;
'''


def block_function(function, text, end, cb_cur_pos):
    block = cur_pos_block(text, end)
    block = block.replace('reinterpret_cast<volatile tt_l1_ptr uint32_t*>(', 'l1_words(')
    if block.count('return;') != 1:
        raise AssertionError('%s: expected one return in the cur_pos block' % function)
    block = block.replace('return;', 'return Outcome{true, cur_pos};')
    return ('template <bool is_causal, bool runtime_extent, bool kv_share>\n'
            'Outcome %s(uint32_t cur_pos_arg, uint32_t cur_batch) {\n%s%s    return Outcome{false, cur_pos};\n}\n'
            % (function, CURPOS_CONSTANTS.replace('CB_CUR_POS', str(cb_cur_pos)), block))


# (label, is_causal, runtime_extent, kv_share, cur_pos_arg or None for the tensor, words): every combination a
# qwen or legacy program can build (the factory refuses causal qwen programs; K64i's non-causal ones pass no
# tensor), including the skips.
CURPOS_SCENARIOS = (
    ('extent', 0, 1, 0, None, (2303, 16895, 131327)),
    ('extent-skip-1', 0, 1, 0, None, (2303, UINT32_MAX, 98559)),
    ('extent-skip-all', 0, 1, 0, None, (UINT32_MAX, UINT32_MAX, UINT32_MAX)),
    ('extent-share', 0, 1, 1, None, (33023, 2303, UINT32_MAX)),
    ('extent-share-skip0', 0, 1, 1, None, (UINT32_MAX, 2303, 65791)),
    ('noncausal', 0, 0, 0, None, (5, 6, 7)),
    ('noncausal-share', 0, 0, 1, None, (5, 6, 7)),
    ('causal-tensor', 1, 0, 0, None, (100, UINT32_MAX, 131000)),
    ('causal-vector', 1, 0, 0, 4242, (1, 2, 3)),
)


def curpos_source():
    kernel_text = {name: k64j(name) for name in kernels.KERNELS}
    parts = [CURPOS_PRELUDE,
             block_function('reader_qwen', kernel_text[kernels.READER_QWEN], SPLIT_START, 8),
             block_function('reader_slice', kernel_text[kernels.READER_SLICE], SPLIT_START, 8),
             block_function('writer_slice', kernel_text[kernels.WRITER_SLICE], SPLIT_START, 8),
             block_function('compute_qwen', kernel_text[kernels.COMPUTE_QWEN], COMPUTE_SPLIT, 15)]
    main = ['int main() {']
    for label, causal, extent, share, arg, words in CURPOS_SCENARIOS:
        flags = '%s, %s, %s' % ('true' if causal else 'false', 'true' if extent else 'false', 'true' if share else 'false')
        cur_pos_arg = 'UINT32_MAX' if arg is None else '%du' % arg
        for reader in ('reader_qwen', 'reader_slice'):
            for entry in range(len(words)):
                main.append('    {')
                main.append('        reset_core();')
                main.append('        const uint32_t words[] = {%s};' % ', '.join('%du' % w for w in words))
                main.append('        std::memset(DRAM_PAGE, 0, sizeof DRAM_PAGE); std::memcpy(DRAM_PAGE, words, sizeof words);')
                main.append('        const char* hang = "none"; Outcome r{}, w{}, c{};')
                main.append('        try { r = %s<%s>(%s, %du); } catch (const Hang&) { hang = "reader"; }' % (reader, flags, cur_pos_arg, entry))
                main.append('        try { w = writer_slice<%s>(%s, %du); } catch (const Hang&) { hang = "writer"; }' % (flags, cur_pos_arg, entry))
                main.append('        try { c = compute_qwen<%s>(%s, %du); } catch (const Hang&) { hang = "compute"; }' % (flags, cur_pos_arg, entry))
                main.append('        std::printf("P %s %s %u %%d %%u %%d %%u %%d %%u %%s %%d %%d %%d %%d %%d %%d\\n", r.skipped, r.cur_pos, w.skipped, w.cur_pos, '
                            'c.skipped, c.cur_pos, hang, PUSHED[8], PUSHED[15], POPPED[8], POPPED[15], DRAM_READS, L1_COPIES);'
                            % (label, reader, entry))
                main.append('    }')
    # The split each kernel then takes (rt_args_common, as every kernel calls it) at the runtime words.
    main.append('    const uint32_t positions[] = {%s};' % ', '.join(str(e - 1) for e in (2304, 16896, 33024, 65792, 98560, 131328)))
    main.append('    for (uint32_t cur_pos : positions) {')
    main.append('        auto chunk = get_dynamic_Sk_chunk_t<8, 8>(cur_pos) * 32;')
    main.append('        for (uint32_t core = 0; core < 16; ++core) {')
    main.append('            auto [pst, n, s, e, wu, wc] = get_workload_for_core(cur_pos, 0, core, 16, chunk);')
    main.append('            std::printf("W %u %u %u %u %u\\n", cur_pos, core, n, s, e);')
    main.append('        }')
    main.append('    }')
    main.append('    return 0;')
    main.append('}')
    return ''.join(parts) + NL.join(main) + NL


class CompiledTests(unittest.TestCase):
    def setUp(self):
        self.gxx = find_host_gxx()
        if self.gxx is None:
            self.skipTest('no host g++ (set QWEN_PF_GXX, or run this module in WSL or on the CI runner)')

    def test_w3_at_q_tile_start_0_writes_exactly_what_the_stock_function_writes(self):
        stock = function_text(read(STOCK['common']), TEMPLATE_LINE, 'write_partial_tiles_to_memory')
        ours = function_text(k64j(kernels.WRITER_SLICE), TEMPLATE_LINE, 'write_partial_tiles_sliced')
        output = compile_and_run(self.gxx, W3_PRELUDE + stock + ours + W3_MAIN)
        rows = [line.split() for line in output.splitlines() if line.startswith('C ')]
        self.assertGreater(len(rows), 1000)
        for row in rows:
            _c, pnht, vdht, kv_heads, per_kv, head, same, writes, advance, tiles = row
            self.assertEqual(same, '1', row)
            self.assertEqual(int(writes), 2 * int(vdht) * int(per_kv), row)    # two face-line writes per row and hidden tile
            self.assertEqual(advance, tiles, row)                              # the stock one's dead trailing increment

    def test_the_cur_pos_blocks_agree_skip_and_split_as_the_compile_time_call(self):
        output = compile_and_run(self.gxx, curpos_source(), {'rt_args_common.hpp': read(STOCK['rt_args'])})
        seen = {}
        for line in output.splitlines():
            if not line.startswith('P '):
                continue
            (_p, label, reader, entry, r_skip, r_pos, w_skip, w_pos, c_skip, c_pos, hang, pushed8, pushed15, popped8,
             popped15, dram, copies) = line.split()
            seen[(label, reader, int(entry))] = dict(
                r=(int(r_skip), int(r_pos)), w=(int(w_skip), int(w_pos)), c=(int(c_skip), int(c_pos)), hang=hang,
                pushed=(int(pushed8), int(pushed15)), popped=(int(popped8), int(popped15)), dram=int(dram), copies=int(copies))
        scenarios = {label: (causal, extent, share, arg, words) for label, causal, extent, share, arg, words in CURPOS_SCENARIOS}
        self.assertEqual(len(seen), sum(2 * len(words) for _l, _c, _e, _s, _a, words in CURPOS_SCENARIOS))
        for (label, reader, entry), got in sorted(seen.items()):
            causal, extent, share, arg, words = scenarios[label]
            with self.subTest(scenario=label, reader=reader, entry=entry):
                self.assertEqual(got['hang'], 'none')
                if arg is not None:                                  # a causal cur_pos vector: no CB traffic
                    expect, reads = arg, False
                elif extent or causal:
                    expect = words[0] if (extent and share) else words[entry]
                    reads = True
                else:                                                # K64i's non-causal programs: the capacity
                    expect, reads = 4104 * 32 - 1, False
                skipped = int(expect == UINT32_MAX)
                self.assertEqual(got['r'], (skipped, expect))
                self.assertEqual(got['w'], (skipped, expect))
                self.assertEqual(got['c'], (skipped, expect))
                self.assertEqual(got['pushed'], (1, 1) if reads else (0, 0))
                self.assertEqual(got['popped'], (1, 1) if reads else (0, 0))      # every pushed copy consumed
                self.assertEqual((got['dram'], got['copies']), (1, 1) if reads else (0, 0))
        # The split at every runtime word E - 1 is the compile-time call's at capacity E (split_model, which the P0
        # CompiledMirrorTests showed is the kernel code): same chunks, same per-core ranges.
        splits = {}
        for line in output.splitlines():
            if line.startswith('W '):
                _w, cur_pos, core, chunks, start, end = (int(v) if v != 'W' else v for v in line.split())
                splits.setdefault(cur_pos, []).append((chunks, start, end))
        self.assertEqual(len(splits), 6)
        for cur_pos, rows in splits.items():
            live = model.split(cur_pos, 16)
            self.assertEqual([(live['num_chunks'],) + tuple(r) for r in live['ranges']], [tuple(row) for row in rows])
            self.assertTrue(model.same_split(cur_pos, cur_pos + 1, 16))


# ---------------------------------------------------------------------------------------------
# The factory's own qwen blocks, executed.
# ---------------------------------------------------------------------------------------------

def between(text, start, end, include_end=True):
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j + len(end)] if include_end else text[i:j]


def factory_blocks():
    text = factory_text()
    flags = between(text, '    constexpr std::size_t kQwenMagicMask', 'program_config->q_chunk_size & 0xFFu) : 0u;\n')
    pnht = between(text, '    const uint32_t qwen_pnht_full', '    const uint32_t PNHt = qwen_q_slice ? qwen_slice_tiles : qwen_pnht_full;\n')
    checks = between(text, '    // ========== [QWEN-SDPA] preconditions ==========', '    // ========== Tree Reduction Setup ==========',
                     include_end=False)
    logs = between(text, '    if (qwen_mode) {\n        uint32_t qwen_cb_bytes = 0;', edit_new('F22')[len(factory.F22_ANCHOR):])
    reader = between(text, qwen_factory.F6[len(qwen_factory.F6_ANCHOR):], edit_new('F21 reader')[len(factory.F21_READER_ANCHOR):])
    writer = edit_new('F21 writer')
    compute = between(text, '    if (qwen_mode) {\n        compute_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_mask_tail));',
                      '    }\n')
    selection = slice_factory.F17R_NEW + factory.F21_KERNEL_NEW + qwen_factory.F8C_NEW
    for piece in (writer, selection):
        for line in piece.split(NL):
            if line.strip():
                assert line in text, line
    return flags, pnht, checks, logs, reader, writer, compute, selection


FACTORY_PRELUDE = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace tt { enum LogType { LogOp }; }
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;
enum class DataType { INT32, UINT32, BFLOAT16 };
enum class Layout { ROW_MAJOR, TILE };
struct Shape {
    std::vector<uint32_t> dims;
    uint32_t operator[](int index) const { return dims.at(index < 0 ? dims.size() + index : index); }
};
struct Tensor {
    DataType dt; Layout lay; Shape shape; bool sharded;
    DataType dtype() const { return dt; }
    Layout layout() const { return lay; }
    const Shape& padded_shape() const { return shape; }
};
struct ProgramConfig { std::size_t q_chunk_size; };
struct Fatal { std::string message; };
struct Arg { std::string dec, hex; };
template <typename T> Arg arg(const T& value) {
    if constexpr (std::is_same_v<T, bool>) {
        return {value ? "true" : "false", value ? "0x1" : "0x0"};
    } else {
        char buffer[40];
        std::snprintf(buffer, sizeof buffer, "0x%llx", static_cast<unsigned long long>(value));
        return {std::to_string(value), buffer};
    }
}
template <typename... A> std::string fmt(const char* format, const A&... values) {
    std::vector<Arg> args{arg(values)...};
    std::string out;
    size_t next = 0;
    for (const char* p = format; *p;) {
        if (p[0] == '{' && p[1] == '}') { out += args.at(next++).dec; p += 2; }
        else if (std::strncmp(p, "{:#x}", 5) == 0) { out += args.at(next++).hex; p += 5; }
        else { out += *p++; }
    }
    return out;
}
static std::vector<std::string> LOG;
template <typename... A> void log_info(tt::LogType, const char* format, const A&... values) { LOG.push_back(fmt(format, values...)); }
template <typename... A> [[noreturn]] void fatal(const char* format, const A&... values) { throw Fatal{fmt(format, values...)}; }
#define TT_FATAL(condition, ...) do { if (!(condition)) fatal(__VA_ARGS__); } while (0)

struct Case {
    const char* label; bool has_config; uint32_t q_chunk_size; bool is_causal; uint32_t B; uint32_t num_q_heads;
    uint32_t St; bool mask; uint32_t mask_width_t; bool cur_pos; DataType pos_dtype; Layout pos_layout; uint32_t pos_words;
    bool pos_sharded;
};

static void build(const Case& c) {
    LOG.clear();
    std::optional<ProgramConfig> program_config;
    if (c.has_config) program_config = ProgramConfig{c.q_chunk_size};
    const bool is_causal = c.is_causal;
    uint32_t B = c.B;
    uint32_t num_q_heads = c.num_q_heads;
    uint32_t num_kv_heads = 2;
    uint32_t PNH = ((num_q_heads + 31) / 32) * 32;
    uint32_t q_heads_parallel_factor = 1;
    const uint32_t St = c.St;
    const uint32_t Sk_chunk_t = 256 / TILE_HEIGHT;
    const bool use_attention_mask = c.mask;
    std::optional<Tensor> attn_mask;
    if (c.mask) attn_mask = Tensor{DataType::BFLOAT16, Layout::TILE, Shape{{B, 1, PNH, c.mask_width_t * TILE_WIDTH}}, false};
    std::optional<const Tensor> cur_pos_tensor;
    if (c.cur_pos) cur_pos_tensor.emplace(Tensor{c.pos_dtype, c.pos_layout, Shape{{c.pos_words}}, c.pos_sharded});
    const bool use_cur_pos_tensor = cur_pos_tensor.has_value();
    const bool is_cur_pos_tensor_sharded = use_cur_pos_tensor && cur_pos_tensor->sharded;
    const uint32_t cur_pos_stick_size = use_cur_pos_tensor ? 64 : 0;
    const uint32_t sliding_window_size = 0;
    const bool is_paged_attention = true, is_page_table_sharded = false, use_mla = false, is_q_sharded = false;
    const bool is_output_sharded = false, on_subcoregrid = false, use_attention_sink = false, tilize_q = false;
    struct { std::optional<int> v = 1; } tensor_args;
    const bool apply_geometry_override = false, has_block_padding = false;
    const uint32_t capacity_t = 0;
    uint32_t num_heads_per_core = 1;
    struct { uint32_t x, y; } grid_size{13, 10};
    const uint32_t num_cores_per_head = 16;
    const uint32_t num_cores_per_batch = num_cores_per_head * num_kv_heads;
    (void)use_mla; (void)num_heads_per_core;
FLAGS
PNHT
CHECKS
    struct CB { uint32_t total_size; };
    struct { std::vector<CB> cbs; } desc;
    desc.cbs.push_back({500000});
    if (use_cur_pos_tensor) { desc.cbs.push_back({cur_pos_stick_size}); desc.cbs.push_back({cur_pos_stick_size}); }
    const uint32_t out_tiles = PNHt * 8;
    const uint32_t intermed_output_tiles = (out_tiles + 2 * PNHt) * 4;
LOGS
    const uint32_t kv_ready_semaphore_id = 3;
    std::vector<uint32_t> reader_compile_time_args_common(44, 0u);
READER
    std::vector<uint32_t> writer_compile_time_args_common(29, 0u);
WRITER
    std::vector<uint32_t> compute_compile_time_args_common(32, 0u);
COMPUTE
    const std::string kernel_path = "";
    struct { std::string kernel_source; } reader_desc, writer_desc, compute_desc;
SELECTION
    std::printf("K %s reader=%s writer=%s compute=%s", c.label, reader_desc.kernel_source.c_str(),
                writer_desc.kernel_source.c_str(), compute_desc.kernel_source.c_str());
    std::printf(" rcta=");
    for (size_t i = 44; i < reader_compile_time_args_common.size(); ++i) std::printf("%u,", reader_compile_time_args_common[i]);
    std::printf(" wcta=");
    for (size_t i = 29; i < writer_compile_time_args_common.size(); ++i) std::printf("%u,", writer_compile_time_args_common[i]);
    std::printf(" ccta=");
    for (size_t i = 32; i < compute_compile_time_args_common.size(); ++i) std::printf("%u,", compute_compile_time_args_common[i]);
    std::printf("\n");
    for (const auto& line : LOG) std::printf("L %s %s\n", c.label, line.c_str());
}

int main() {
    const Case cases[] = {
CASES
    };
    for (const auto& c : cases) {
        try {
            build(c);
        } catch (const Fatal& error) {
            std::printf("F %s %s\n", c.label, error.message.c_str());
        }
    }
    return 0;
}
'''

MAGIC = 0x51DEC000
# (label, qwen flags or None for a legacy call, causal, B, rows per fold group, mask: None / 'narrow' / 'wide', cur_pos:
# None or (dtype, layout, words, sharded)); rows * 12 folded Q heads, 131,328 keys (St 4104).
FACTORY_CASES = (
    ('x21_g4b3', 0x21, False, 3, 4, 'narrow', ('INT32', 'ROW_MAJOR', 3, False)),
    ('x23_g4b3', 0x23, False, 3, 4, 'narrow', ('INT32', 'ROW_MAJOR', 3, False)),
    ('x21_g8b1', 0x21, False, 1, 8, 'narrow', ('INT32', 'ROW_MAJOR', 1, False)),
    ('x25_g8b1', 0x25, False, 1, 8, 'narrow', ('INT32', 'ROW_MAJOR', 1, False)),
    ('x27_g8b2', 0x27, False, 2, 8, 'narrow', ('INT32', 'ROW_MAJOR', 2, False)),
    ('x2f_g8b2', 0x2F, False, 2, 8, 'narrow', ('INT32', 'ROW_MAJOR', 2, False)),
    ('x2b_g8b2', 0x2B, False, 2, 8, 'narrow', ('INT32', 'ROW_MAJOR', 2, False)),
    ('k01_wide', 0x01, False, 3, 4, 'wide', None),
    ('k03_wide', 0x03, False, 3, 4, 'wide', None),
    ('k07_wide', 0x07, False, 2, 8, 'wide', None),
    ('k01_narrow', 0x01, False, 3, 4, 'narrow', None),
    ('legacy_causal', None, True, 3, 4, None, ('INT32', 'ROW_MAJOR', 3, False)),
    ('refuse_x10', 0x10, False, 3, 4, 'narrow', None),
    ('refuse_x11', 0x11, False, 3, 4, 'narrow', None),
    ('refuse_x31', 0x31, False, 3, 4, 'narrow', ('INT32', 'ROW_MAJOR', 3, False)),
    ('refuse_x40', 0x40, False, 3, 4, 'narrow', None),
    ('refuse_x20_notail', 0x20, False, 3, 4, 'narrow', ('INT32', 'ROW_MAJOR', 3, False)),
    ('refuse_x21_wide', 0x21, False, 3, 4, 'wide', ('INT32', 'ROW_MAJOR', 3, False)),
    ('refuse_x21_nopos', 0x21, False, 3, 4, 'narrow', None),
    ('refuse_x21_sharded', 0x21, False, 3, 4, 'narrow', ('INT32', 'ROW_MAJOR', 3, True)),
    ('refuse_x21_uint32', 0x21, False, 3, 4, 'narrow', ('UINT32', 'ROW_MAJOR', 3, False)),
    ('refuse_x21_tile', 0x21, False, 3, 4, 'narrow', ('INT32', 'TILE', 3, False)),
    ('refuse_x21_words', 0x21, False, 3, 4, 'narrow', ('INT32', 'ROW_MAJOR', 4, False)),
    ('refuse_x21_causal', 0x21, True, 3, 4, 'narrow', ('INT32', 'ROW_MAJOR', 3, False)),
    ('refuse_x01_pos', 0x01, False, 3, 4, 'narrow', ('INT32', 'ROW_MAJOR', 3, False)),
    ('refuse_x28', 0x28, False, 2, 8, 'narrow', ('INT32', 'ROW_MAJOR', 2, False)),
)


def factory_source():
    flags, pnht, checks, logs, reader, writer, compute, selection = factory_blocks()
    rows = []
    for label, qwen, causal, batch, group_rows, mask, pos in FACTORY_CASES:
        width = {None: 0, 'narrow': 8, 'wide': 4104}[mask]
        dtype, layout, words, sharded = pos or ('INT32', 'ROW_MAJOR', 0, False)
        rows.append('        {"%s", %s, 0x%Xu, %s, %du, %du, 4104u, %s, %du, %s, DataType::%s, Layout::%s, %du, %s},'
                    % (label, 'false' if qwen is None else 'true', MAGIC | (qwen or 0), 'true' if causal else 'false', batch,
                       group_rows * 12, 'true' if mask else 'false', width, 'true' if pos else 'false', dtype, layout, words,
                       'true' if sharded else 'false'))
    source = FACTORY_PRELUDE
    for name, value in (('FLAGS', flags), ('PNHT', pnht), ('CHECKS', checks), ('LOGS', logs), ('READER', reader),
                        ('WRITER', writer), ('COMPUTE', compute), ('SELECTION', selection), ('CASES', NL.join(rows))):
        source = source.replace(NL + name + NL, NL + value.rstrip(NL) + NL, 1)
    return source


class CompiledFactoryTests(unittest.TestCase):
    """F19-F22 in the factory's own text (generated from the vendored base), executed in a stub create_descriptor
    whose locals have the factory's declared types: the refusals, the compile-time args, the kernels selected and
    the log lines, for every flag set K64j builds and every precondition it refuses."""

    @classmethod
    def setUpClass(cls):
        gxx = find_host_gxx()
        if gxx is None:
            raise unittest.SkipTest('no host g++ (set QWEN_PF_GXX, or run this module in WSL or on the CI runner)')
        output = compile_and_run(gxx, factory_source())
        cls.built, cls.fatal, cls.logs = {}, {}, {}
        for line in output.splitlines():
            kind, label, rest = line.split(' ', 2)
            if kind == 'K':
                cls.built[label] = dict(part.split('=', 1) for part in rest.split())
            elif kind == 'F':
                cls.fatal[label] = rest
            else:
                cls.logs.setdefault(label, []).append(rest)

    def test_every_case_either_builds_or_is_refused(self):
        labels = [case[0] for case in FACTORY_CASES]
        self.assertEqual(sorted(set(self.built) | set(self.fatal)), sorted(labels))
        self.assertFalse(set(self.built) & set(self.fatal))
        for label in labels:
            self.assertEqual(label in self.fatal, label.startswith('refuse_'), (label, self.fatal.get(label)))

    def test_the_refusals_and_their_literals(self):
        expect = {
            'refuse_x10': '[QWEN-SDPA] unknown flags 0x10', 'refuse_x11': '[QWEN-SDPA] unknown flags 0x11',
            'refuse_x31': '[QWEN-SDPA] unknown flags 0x31', 'refuse_x40': '[QWEN-SDPA] unknown flags 0x40',
            'refuse_x20_notail': factory.EXTENT_MASK_REFUSAL, 'refuse_x21_wide': factory.EXTENT_MASK_REFUSAL,
            'refuse_x21_nopos': factory.EXTENT_CUR_POS_REFUSAL, 'refuse_x21_sharded': factory.EXTENT_CUR_POS_REFUSAL,
            'refuse_x21_uint32': factory.EXTENT_LAYOUT_REFUSAL, 'refuse_x21_tile': factory.EXTENT_LAYOUT_REFUSAL,
            'refuse_x21_words': factory.EXTENT_LAYOUT_REFUSAL, 'refuse_x21_causal': factory.CUR_POS_REFUSAL,
            'refuse_x01_pos': factory.CUR_POS_REFUSAL, 'refuse_x28': slice_factory.READAHEAD_REFUSAL}
        for label, needle in expect.items():
            with self.subTest(case=label):
                self.assertTrue(self.fatal[label].startswith(needle), self.fatal[label])
        self.assertEqual(self.fatal['refuse_x21_wide'], factory.EXTENT_MASK_REFUSAL + ' 8-tile mask, got flags 0x21 and mask width 4104')
        self.assertEqual(self.fatal['refuse_x21_words'], factory.EXTENT_LAYOUT_REFUSAL + '3 entries')

    def test_0x20_programs_run_the_qwen_kernels_and_the_slice_writer_with_the_word_at_every_kernels_offset(self):
        for label, reader, rcta, wcta in (
                ('x21_g4b3', 'dataflow/reader_decode_qwen.cpp', '1,8,0,3,1,', '2,24,20942,0,1,'),
                ('x23_g4b3', 'dataflow/reader_decode_qwen.cpp', '1,8,1,3,1,', '2,24,20942,0,1,'),
                ('x21_g8b1', 'dataflow/reader_decode_qwen.cpp', '1,8,0,3,1,', '3,48,20942,0,1,'),
                ('x25_g8b1', 'dataflow/reader_decode_qwen_slice.cpp', '1,8,0,3,3,48,0,1,20942,1,', '3,48,20942,1,1,'),
                ('x27_g8b2', 'dataflow/reader_decode_qwen_slice.cpp', '1,8,1,3,3,48,0,1,20942,1,', '3,48,20942,1,1,'),
                ('x2f_g8b2', 'dataflow/reader_decode_qwen_slice.cpp', '1,8,1,3,3,48,1,1,20942,1,', '3,48,20942,1,1,'),
                ('x2b_g8b2', 'dataflow/reader_decode_qwen_slice.cpp', '1,8,1,3,3,48,1,0,20942,1,', '3,48,20942,0,1,')):
            with self.subTest(case=label):
                built = self.built[label]
                self.assertEqual(built['reader'], reader)
                self.assertEqual(built['writer'], 'dataflow/writer_decode_qwen_slice.cpp')
                self.assertEqual(built['compute'], 'compute/sdpa_flash_decode_qwen.cpp')
                self.assertEqual(built['rcta'], rcta)
                self.assertEqual(built['wcta'], wcta)
                self.assertEqual(built['ccta'], '1,1,')
                values = [int(v) for v in built['rcta'].rstrip(',').split(',')]
                self.assertEqual(values[kernels.READER_EXTENT_OFFSET['dataflow/' + reader.split('/')[-1]]], 1)
                self.assertEqual(len(values) - 1, kernels.READER_EXTENT_OFFSET['dataflow/' + reader.split('/')[-1]])

    def test_k64i_programs_keep_their_kernels_and_args_with_the_word_off(self):
        for label, reader, writer, rcta, wcta in (
                ('k01_wide', 'dataflow/reader_decode_qwen.cpp', 'dataflow/writer_decode_all.cpp', '1,4104,0,3,0,', ''),
                ('k03_wide', 'dataflow/reader_decode_qwen.cpp', 'dataflow/writer_decode_all.cpp', '1,4104,1,3,0,', ''),
                ('k07_wide', 'dataflow/reader_decode_qwen_slice.cpp', 'dataflow/writer_decode_qwen_slice.cpp',
                 '1,4104,1,3,3,48,0,1,20942,0,', '3,48,20942,1,0,'),
                ('k01_narrow', 'dataflow/reader_decode_qwen.cpp', 'dataflow/writer_decode_all.cpp', '1,8,0,3,0,', '')):
            with self.subTest(case=label):
                built = self.built[label]
                self.assertEqual((built['reader'], built['writer'], built['rcta'], built['wcta'], built['ccta']),
                                 (reader, writer, rcta, wcta, '1,0,'))
                self.assertFalse([line for line in self.logs.get(label, []) if factory.EXTENT_LOG_MARKER in line])
        legacy = self.built['legacy_causal']
        self.assertEqual((legacy['reader'], legacy['writer'], legacy['compute'], legacy['rcta'], legacy['wcta'], legacy['ccta']),
                         ('dataflow/reader_decode_all.cpp', 'dataflow/writer_decode_all.cpp', 'compute/sdpa_flash_decode.cpp', '', '', ''))
        self.assertNotIn('legacy_causal', self.logs)

    def test_the_log_lines_one_f22_line_per_0x20_program_and_two_sticks_of_cb_bytes(self):
        for label, batch, share, q_slice in (('x21_g4b3', 3, 'false', 'false'), ('x23_g4b3', 3, 'true', 'false'),
                                             ('x25_g8b1', 1, 'false', 'true'), ('x27_g8b2', 2, 'true', 'true'),
                                             ('x2f_g8b2', 2, 'true', 'true')):
            with self.subTest(case=label):
                lines = self.logs[label]
                extent = [line for line in lines if line.startswith(factory.EXTENT_LOG_MARKER)]
                self.assertEqual(extent, ['%s%d kv_share=%s q_slice=%s writer=writer_decode_qwen_slice.cpp cur_pos_stick_bytes=64'
                                          % (factory.EXTENT_LOG_MARKER, batch, share, q_slice)])
                flags = [line for line in lines if line.startswith('[QWEN-SDPA] flags=')]
                self.assertEqual(len(flags), 1)
                self.assertIn('flags=0x%x ' % int(label[1:3], 16), flags[0])
                self.assertTrue(flags[0].endswith(' cb_bytes=%d' % (500000 + 2 * 64)), flags[0])
                self.assertEqual(len([line for line in lines if line.startswith('[QWEN-SDPA] q-slice ')]),
                                 1 if label in ('x25_g8b1', 'x27_g8b2', 'x2f_g8b2') else 0)
        self.assertTrue(self.logs['k03_wide'][0].endswith(' cb_bytes=500000'))


# ---------------------------------------------------------------------------------------------
# The stub syntax compile (sdpa_decode_slice/stubcheck): the K64j kernels at the factory's arg vectors.
# ---------------------------------------------------------------------------------------------

class StubCompileTests(unittest.TestCase):
    def test_the_k64j_readers_and_writer_compile_and_their_asserts_fire(self):
        import stub_compile
        gxx = stub_compile.find_gxx()
        if gxx is None:
            self.skipTest('no g++ (set QWEN_PF_GXX)')
        if not DUMP.is_file() or not (PREFILL_SRC / 'kernels').is_dir():
            self.skipTest('no dump or prefill tree for the kernel stubs')
        tag = slice_kernels.ABI_TAG
        qwen_reader = ('reader_decode_qwen.cpp', k64j(kernels.READER_QWEN).encode())
        slice_reader = ('reader_decode_qwen_slice.cpp', k64j(kernels.READER_SLICE).encode())
        writer = ('writer_decode_qwen_slice.cpp', k64j(kernels.WRITER_SLICE).encode())
        rct, wct = stub_compile.reader_ct, stub_compile.writer_ct
        narrow = dict(qwen=(1, 8, 1, 3))
        cases = (
            ('qwen reader 0x21', qwen_reader, rct(3, qwen=(1, 8, 0, 3), slice_suffix=(1,)), True),
            ('qwen reader 0x23', qwen_reader, rct(3, slice_suffix=(1,), **narrow), True),
            ('qwen reader 0x3 (K64i, word off)', qwen_reader, rct(3, slice_suffix=(0,)), True),
            ('slice reader 0x27', slice_reader, rct(2, slice_suffix=(3, 48, 0, 1, tag, 1), **narrow), True),
            ('slice reader 0x2F', slice_reader, rct(2, slice_suffix=(3, 48, 1, 1, tag, 1), **narrow), True),
            ('slice reader 0x2B', slice_reader, rct(3, slice_suffix=(3, 48, 1, 0, tag, 1), **narrow), True),
            ('slice reader 0x7 (K64i, word off)', slice_reader, rct(2, slice_suffix=(3, 48, 0, 1, tag, 0)), True),
            ('qwen reader 0x20 without the tail', qwen_reader, rct(3, qwen=(0, 8, 0, 3), slice_suffix=(1,)), 'needs the tail mask'),
            ('qwen reader 0x21 with a wide mask', qwen_reader, rct(3, slice_suffix=(1,)), 'narrow one-chunk mask'),
            ('qwen reader 0x21 causal', qwen_reader, rct(3, slice_suffix=(1,), is_causal=1, **narrow), 'non-causal mode'),
            ('qwen reader 0x21 sharded cur_pos', qwen_reader, rct(3, slice_suffix=(1,), is_cur_pos_tensor_sharded=1, **narrow),
             'interleaved cur_pos tensor'),
            ('writer 0x21 (no slice)', writer, wct(3, slice_suffix=(3, 48, tag, 0, 1)), True),
            ('writer 0x27 (slice)', writer, wct(2, slice_suffix=(3, 48, tag, 1, 1)), True),
            ('writer 0x7 (K64i, word off)', writer, wct(2, slice_suffix=(3, 48, tag, 1, 0)), True),
            ('writer MQA 0x21', writer, wct(1, slice_suffix=(1, 12, tag, 0, 1), num_kv_heads=1, num_q_heads=12,
                                           num_reducer_cores=2), True),
            ('writer neither flag', writer, wct(3, slice_suffix=(3, 48, tag, 0, 0)), '0x4 or 0x20 only'),
            ('writer 0x21 with a sliced PNHt', writer, wct(2, slice_suffix=(3, 48, tag, 0, 1)), "Q's own row tiles otherwise"),
            ('writer 0x21 causal', writer, wct(3, slice_suffix=(3, 48, tag, 0, 1), is_causal=1), 'non-causal mode'),
        )
        failed = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative, data in stub_compile.sources(DUMP, PREFILL_SRC).items():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_bytes(data)
            for label, (name, data), ct, expect in cases:
                code, output = stub_compile.compile_case(gxx, root, name, data, ct)
                if expect is True:
                    ok = code == 0
                else:
                    ok = code != 0 and 'static assertion failed' in output and expect in output
                if not ok:
                    failed.append((label, output[-800:]))
        self.assertEqual(failed, [])


class ContractTests(unittest.TestCase):
    def test_the_generated_files_are_lf(self):
        for path in list(KERNEL_DIR.rglob('*.cpp')) + list(HERE.glob('*.py')) + [FACTORY_BASE]:
            self.assertNotIn(b'\r', path.read_bytes(), path.name)

    def test_the_cpu_suite_runs_this_directory(self):
        self.assertIn("python -B -m unittest discover -s optimisation/ttnn-op/k64j -p 'test_*.py'", read(CPU_WORKFLOW))

    def test_the_probe_readme_points_here_with_the_recorded_shas(self):
        readme = read(PROBE / 'README.md')
        self.assertIn('optimisation/ttnn-op/k64j', readme)
        self.assertIn(factory.K64J_FACTORY[:16], readme)
        for name in kernels.KERNELS:
            self.assertIn(kernels.OUTPUTS[name][:8], readme, name)


if __name__ == '__main__':
    unittest.main()
