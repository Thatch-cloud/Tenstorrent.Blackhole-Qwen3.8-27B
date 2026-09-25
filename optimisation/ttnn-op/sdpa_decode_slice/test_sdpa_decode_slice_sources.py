"""CPU checks for the K64i sources (K1: the stage-4 decode factory and the two slice kernels); no device, no ttnn.

  - the committed kernels are their bases plus exactly their edits: reverting R9-R6 gives the committed stage-3
    reader (280a847f), reverting W3-W1 gives the stock writer (734c90c0); they hash to the recorded outputs, and
    regenerate from the sources dump;
  - apply_factory_slice refuses anything but the 3e0a69af base or the 06167779 stage-3 factory, reproduces the
    recorded stage-4 factory from both, and inverts to stage 3, stage 1 and the base;
  - the edits say what the design says: PNHt follows the slice only under 0x4; the reader suffix is appended
    under 0x4 OR 0x8 and the writer suffix under 0x4 only (the review's F16/F17 fix); 0x8 needs 0x2; R9's
    leader multicasts chunk n, reads its mask, reads n+1 (never past k_chunk_end), write-barriers, then VALID;
    write_partial_tiles_sliced is write_partial_tiles_to_memory with the L1 source rebased and nothing else;
  - every constant that crosses a file boundary agrees: the flags, the ABI tag and the CTA order between F16
    and R6/W1, the kernel names F17 selects, the recorded shas in the build script, the runner and the harness,
    the refusal texts and the log format the harness parses;
  - the index model (slice_index_model) over rows 1-16, 1-3 entries and both KV heads: the rule, Q / mask / output
    index math against the legacy path, every kept row written exactly once from the right L1 slot, and the CB
    bytes (the stage-3 test's table and the design's section-6 figures);
  - the K1b protocol model (share_protocol_model): no deadlock and no slot hazard for both leaders over a
    reduced grid, and the two broken variants are caught;
  - the build script (bash -n, LC_ALL=C, the inverted sdpa/ check, the image comparisons, the restore);
  - the stub compile (g++ -fsyntax-only of the kernels and factory blocks), when a compiler and the dumps exist.

The optional inputs are found at QWEN_SDPA_SOURCES_DUMP (the decode sources dump), QWEN_SDPA_FACTORY_BASE (a
3e0a69af factory) and QWEN_SDPA_PREFILL_SRC (probe-v25 src/device); the tests that need them skip without them.

    py -3.11 -B -m unittest test_sdpa_decode_slice_sources      (from this directory)
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
for path in (str(HERE), str(QWEN), str(HERE / 'stubcheck')):
    if path not in sys.path:
        sys.path.insert(0, path)

import apply_factory_qwen as qwen_factory  # noqa: E402
import apply_factory_slice as factory  # noqa: E402
import make_qwen_kernels as qwen_kernels  # noqa: E402
import make_slice_kernels as kernels  # noqa: E402
import sdpa_decode_slice_card_b as harness  # noqa: E402
import share_protocol_model as protocol  # noqa: E402
import slice_index_model as model  # noqa: E402
import test_sdpa_decode_qwen_card_m as card  # noqa: E402

NL = chr(10)
JOB = Path('C:/Users/liamb/.claude/jobs/8376c877/tmp')
DUMP = Path(os.environ.get('QWEN_SDPA_SOURCES_DUMP', JOB / 'sdpa-decode-sources.txt'))
FACTORY_BASE = Path(os.environ.get('QWEN_SDPA_FACTORY_BASE', JOB / 'sdpa_decode_program_factory.cpp.3e0a69af'))
PREFILL_SRC = Path(os.environ.get('QWEN_SDPA_PREFILL_SRC', JOB / 'probe-v25' / 'src' / 'device'))
BUILD = HERE / 'build_k64i.sh'
RUNNER = HERE / 'run_card_b.sh'
STAGE3_READER = QWEN / 'stage3' / qwen_kernels.READER_NAME


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return path.read_text(encoding='utf-8')


def find_bash():
    candidates = []
    if os.name == 'nt':
        for root in (os.environ.get('ProgramW6432'), os.environ.get('ProgramFiles'), 'C:/Program Files'):
            if root:
                candidates.append(Path(root) / 'Git' / 'bin' / 'bash.exe')
    found = shutil.which('bash')
    if found and not (os.name == 'nt' and ('system32' in found.lower() or 'windowsapps' in found.lower())):
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


BASH = find_bash()


def new_text(label):
    for name, _old, new in factory.STAGE4_EDITS:
        if name == label:
            return new
    raise KeyError(label)


def stage4_text():
    """The stage-4 factory's new text (every edit's replacement), without the base file."""
    return ''.join(new for _label, _old, new in factory.STAGE4_EDITS)


class KernelSourceTests(unittest.TestCase):
    def test_the_committed_kernels_are_their_bases_plus_exactly_the_edits(self):
        for name in (kernels.READER_SLICE_NAME, kernels.WRITER_SLICE_NAME):
            with self.subTest(kernel=name):
                built = (HERE / name).read_bytes()
                self.assertEqual(sha(built), kernels.OUTPUTS[name])
                self.assertNotIn(b'\r', built)
                self.assertEqual(sha(kernels.revert_edits(name, built)), kernels.BASES[name])
        reverted = kernels.revert_edits(kernels.READER_SLICE_NAME, (HERE / kernels.READER_SLICE_NAME).read_bytes())
        self.assertEqual(reverted, STAGE3_READER.read_bytes(), 'the slice reader is the committed stage-3 reader + R6-R9')

    def test_the_edits_apply_to_the_committed_stage_three_reader(self):
        built = kernels.apply_edits(kernels.READER_SLICE_NAME, STAGE3_READER.read_bytes())
        self.assertEqual(built, (HERE / kernels.READER_SLICE_NAME).read_bytes())
        with self.assertRaisesRegex(ValueError, 'base is'):
            kernels.apply_edits(kernels.READER_SLICE_NAME, STAGE3_READER.read_bytes() + b'// changed\n')

    def test_every_anchor_occurs_once_in_its_base(self):
        texts = {kernels.READER_SLICE_NAME: STAGE3_READER.read_text(encoding='utf-8'),
                 kernels.WRITER_SLICE_NAME: kernels.revert_edits(kernels.WRITER_SLICE_NAME,
                                                                  (HERE / kernels.WRITER_SLICE_NAME).read_bytes()).decode()}
        for name, edits in kernels.EDITS.items():
            for label, old, _new in edits:
                with self.subTest(kernel=name, edit=label):
                    self.assertEqual(texts[name].count(old), 1)

    def test_the_kernels_regenerate_from_the_dump(self):
        if not DUMP.is_file():
            self.skipTest('no sources dump at %s' % DUMP)
        with tempfile.TemporaryDirectory() as out:
            self.assertEqual(kernels.main(['--dump', str(DUMP), '--out', out]), 0)
            for name in (kernels.READER_SLICE_NAME, kernels.WRITER_SLICE_NAME):
                self.assertEqual(Path(out, name).read_bytes(), (HERE / name).read_bytes())
            self.assertEqual(kernels.main(['--dump', str(DUMP), '--check', str(HERE)]), 0)
            Path(out, kernels.WRITER_SLICE_NAME).write_bytes(b'x')
            self.assertEqual(kernels.main(['--dump', str(DUMP), '--check', out]), 1)

    def reader(self):
        return read(HERE / kernels.READER_SLICE_NAME)

    def test_r6_reads_the_suffix_at_the_factorys_offsets_and_asserts_the_tag(self):
        text = self.reader()
        offsets = {name: int(offset) for name, offset in
                   re.findall(r'constexpr \w+ (\w+) = get_compile_time_arg_val\(qwen_cta \+ ([0-9])\)', text)}
        self.assertEqual(offsets, dict(mask_tail=0, mask_width_t=1, kv_share=2, kv_ready_semaphore_id=3, pnht_full=4,
                                       rows_per_kv=5, kv_readahead=6, q_slice=7))
        self.assertIn('static_assert(get_compile_time_arg_val(qwen_cta + 8) == 0x51CE', text)
        self.assertEqual([name for name in factory.READER_SUFFIX], ['pnht_full', 'rows_per_kv', 'kv_readahead', 'q_slice', 'tag'])
        self.assertIn('const uint32_t q_tile_start = q_slice ? (cur_head_group * rows_per_kv) >> 5 : 0;', text)

    def test_r7_r8_use_the_full_layout_and_the_head_offset(self):
        text = self.reader()
        self.assertIn('const uint32_t q_batch_offset = cur_batch * pnht_full * DHt + q_tile_start * DHt;', text)
        self.assertIn('% Bmask) * pnht_full * mask_width_t + q_tile_start * mask_width_t;', text)
        self.assertNotIn('const uint32_t q_batch_offset = cur_batch * q_chunk_tiles;', text)
        # read_q still reads q_chunk_tiles (the slice's) tiles into cb_q_in; read_mask_chunk<PNHt> the slice's rows.
        self.assertIn('read_q<cb_q_in, cb_q_rm, q_tile_bytes, q_chunk_tiles,', text)
        self.assertEqual(text.count('read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>('), 3)

    def test_r9_is_the_designs_read_ahead_order_behind_its_flag(self):
        text = self.reader()
        start = text.index('if (kv_readahead && do_k_mcast) {')
        end = text.index('} else if (do_k_mcast) {', start)
        branch = text[start:end]
        order = [branch.index(marker) for marker in (
            'if (k_chunk == k_chunk_start) {',            # prologue: this core's first chunk
            'kv_ready.wait(num_dests);',                  # READY(n)
            'kv_ready.set(0);',
            'CoreLocalMem<uint32_t>(k_slot)',             # multicast K(n)
            'CoreLocalMem<uint32_t>(v_slot)',             # multicast V(n)
            'mask_start_tile_id = read_mask_chunk<',      # chunk n's mask before n+1's K/V
            'if (k_chunk + 1 < k_chunk_end) {',           # never past k_chunk_end
            'noc.async_write_barrier();',                 # n has landed on every twin
            'kv_valid.set(1);',                           # VALID(n)
            'kv_valid.set_multicast(')]
        self.assertEqual(order, sorted(order))
        self.assertEqual(branch.count('read_k<'), 2)
        self.assertEqual(branch.count('read_v<'), 2)
        self.assertEqual(branch.count('noc.async_write_barrier();'), 1)
        self.assertIn('const uint32_t next_row_num = k_chunk_start_row_num + Sk_chunk_t_dynamic;', branch)
        # The same leader helpers as stage 3 (no MLA multicast, explicit V), and the stage-3 leader is intact.
        self.assertEqual(branch.count('                                false,\n'), 4)
        stage3 = read(STAGE3_READER)
        leader = stage3[stage3.index('                    if (do_k_mcast) {'):stage3.index('                    } else {\n'
                                                                                          '                        // TWIN')]
        self.assertIn(leader.replace('                    if (do_k_mcast) {', '                    } else if (do_k_mcast) {', 1),
                      text)
        # The shared mask read skips the read-ahead leader, which read its mask inside its branch.
        self.assertIn('if ((!mask_tail || k_chunk == k_num_chunks - 1) && !(kv_readahead && do_k_mcast)) {', text)
        self.assertIn('static_assert(!kv_readahead || kv_share, "K/V read-ahead needs KV share");', text)

    def test_w3_is_write_partial_tiles_to_memory_with_the_l1_rebase_only(self):
        if not DUMP.is_file():
            self.skipTest('no sources dump at %s' % DUMP)
        common = qwen_kernels.from_dump(DUMP.read_text(encoding='utf-8'),
                                        qwen_kernels.DUMP_PREFIX + 'dataflow/dataflow_common.hpp').decode()

        def body(text, name):
            start = text.index('uint32_t %s(' % name)
            end = text.index('\n}\n', start)
            code = []
            for line in text[start:end].split(NL):
                line = line.split('//')[0].strip()
                if line:
                    code.append(line)
            return code

        stock = body(common, 'write_partial_tiles_to_memory')
        ours = body(read(HERE / kernels.WRITER_SLICE_NAME), 'write_partial_tiles_sliced')
        diff = [line for line in difflib.ndiff(stock, ours) if line[:1] in '+-']
        self.assertEqual(sorted(diff), sorted([
            '- uint32_t write_partial_tiles_to_memory(',
            '+ uint32_t write_partial_tiles_sliced(',
            '- uint32_t out_chunk_tiles) {',
            '+ uint32_t out_chunk_tiles,',
            '+ uint32_t q_tile_start) {',
            '- uint32_t tile_index = head_tile * num_hidden_tiles + hidden_tile;',
            '+ const uint32_t l1_tile_index = (head_tile - q_tile_start) * num_hidden_tiles + hidden_tile;',
            '+ const uint32_t dram_tile_id = out_tile_id + head_tile * num_hidden_tiles + hidden_tile;',
            '- uint32_t l1_read_addr_head = l1_base_addr + tile_index * tile_bytes + in_tile_offset;',
            '+ uint32_t l1_read_addr_head = l1_base_addr + l1_tile_index * tile_bytes + in_tile_offset;',
            '- const uint32_t dram_tile_id = out_tile_id + tile_index;',
            '- out_tile_id += out_chunk_tiles;']), diff)

    def test_the_writer_reads_its_suffix_and_uses_the_full_output_stride(self):
        text = read(HERE / kernels.WRITER_SLICE_NAME)
        self.assertIn('constexpr uint32_t qwen_slice_cta = out_args.next_compile_time_args_offset();', text)
        self.assertIn('constexpr uint32_t pnht_full = get_compile_time_arg_val(qwen_slice_cta + 0);', text)
        self.assertIn('constexpr uint32_t rows_per_kv = get_compile_time_arg_val(qwen_slice_cta + 1);', text)
        self.assertIn('static_assert(get_compile_time_arg_val(qwen_slice_cta + 2) == 0x51CE', text)
        self.assertIn('uint32_t out_tile_id = cur_batch * pnht_full * vDHt;', text)
        self.assertIn('const uint32_t q_tile_start = (cur_head_group * rows_per_kv) >> 5;', text)
        self.assertEqual(factory.WRITER_SUFFIX, ('pnht_full', 'rows_per_kv', 'tag'))
        # The tree traffic keeps the slice's PNHt (CTA 1): nothing else in the writer changed.
        self.assertEqual(text.count('write_partial_tiles_to_memory'), 1)   # the W3 comment only


class FactoryPatchTests(unittest.TestCase):
    def test_anything_but_the_base_or_stage_three_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'unexpected factory'):
            factory.patch(b'not a factory')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'f.cpp')
            path.write_bytes(b'nope')
            self.assertEqual(factory.main([str(path), '--out', str(Path(directory, 'o.cpp'))]), 1)

    def test_the_base_and_stage_three_patch_to_the_recorded_bytes_and_invert(self):
        if not FACTORY_BASE.is_file():
            self.skipTest('no 3e0a69af factory at %s' % FACTORY_BASE)
        base = FACTORY_BASE.read_bytes()
        self.assertEqual(sha(base), factory.BASE_FACTORY)
        stage3 = qwen_factory.patch(base, 3)
        for source in (base, stage3):
            built = factory.patch(source)
            self.assertEqual(sha(built), factory.STAGE4_FACTORY)
            self.assertEqual(factory.unpatch(built), stage3)
            self.assertEqual(sha(factory.unpatch(built, to_stage=1)), qwen_factory.QWEN_FACTORY)
            self.assertEqual(factory.unpatch(built, to_stage=0), base)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory, 'stage4.cpp')
            self.assertEqual(factory.main([str(FACTORY_BASE), '--out', str(out)]), 0)
            self.assertEqual(sha(out.read_bytes()), factory.STAGE4_FACTORY)
            self.assertEqual(factory.main([str(out), '--out', str(Path(directory, 'again.cpp'))]), 0)   # idempotent
            text = out.read_text(encoding='utf-8')
        for label, _old, new in factory.STAGE4_EDITS:
            self.assertEqual(text.count(new), 1, label)
        for marker in factory.STAGE4_MARKERS:
            self.assertIn(marker, text)
        self.assertNotIn(qwen_factory.STAGE1_SHARE_REFUSAL, text)

    def test_pnht_follows_the_slice_only_under_0x4(self):
        text = new_text('F14')
        self.assertIn('const uint32_t PNHt = qwen_q_slice ? qwen_slice_tiles : qwen_pnht_full;', text)
        self.assertIn('const uint32_t qwen_pnht_full = PNH / q_heads_parallel_factor / TILE_HEIGHT;', text)
        self.assertIn('const bool qwen_q_slice = (qwen_flags & kQwenQSlice) != 0;', text)
        # qwen_flags is 0 outside qwen mode (F1), so the legacy path keeps PNHt as it was.
        self.assertIn("const uint32_t qwen_flags = qwen_mode ? static_cast<uint32_t>(program_config->q_chunk_size & 0xFFu) : 0u;",
                      qwen_factory.F1)

    def test_the_review_fix_suffix_and_kernel_selection_conditions(self):
        self.assertIn('if (qwen_q_slice || qwen_kv_readahead) {', new_text('F16 reader'))
        self.assertIn('if (qwen_q_slice) {', new_text('F16 writer'))
        self.assertIn('(qwen_q_slice || qwen_kv_readahead) ? "dataflow/reader_decode_qwen_slice.cpp"', new_text('F17 reader'))
        self.assertIn('qwen_q_slice ? "dataflow/writer_decode_qwen_slice.cpp" : "dataflow/writer_decode_all.cpp"',
                      new_text('F17 writer'))
        pushes = re.findall(r'reader_compile_time_args_common\.push_back\(([^;]+)\);\s*// \+([0-9])', new_text('F16 reader'))
        self.assertEqual([offset for _value, offset in pushes], ['4', '5', '6', '7', '8'])
        self.assertEqual([value for value, _ in pushes],
                         ['qwen_pnht_full', 'qwen_rows_per_kv', 'static_cast<uint32_t>(qwen_kv_readahead)',
                          'static_cast<uint32_t>(qwen_q_slice)', 'kQwenSliceAbiTag'])
        writer = re.findall(r'writer_compile_time_args_common\.push_back\(([^;]+)\);', new_text('F16 writer'))
        self.assertEqual(writer, ['qwen_pnht_full', 'qwen_rows_per_kv', 'kQwenSliceAbiTag'])
        self.assertIn('const bool qwen_kv_readahead = (qwen_flags & kQwenKvReadahead) != 0 && qwen_kv_share;',
                      new_text('F15 decl'))

    def test_the_refusals_and_the_flag_set(self):
        text = new_text('F15')
        self.assertIn('(qwen_flags & ~(kQwenMaskTail | kQwenKvShare | kQwenQSlice | kQwenKvReadahead)) == 0', text)
        self.assertIn('(qwen_flags & kQwenKvReadahead) == 0 || (qwen_flags & kQwenKvShare) != 0', text)
        for needle in (factory.READAHEAD_REFUSAL, factory.NO_SAVING_REFUSAL, factory.COVER_REFUSAL,
                       factory.HEADS_REFUSAL, factory.MASK_REFUSAL):
            self.assertIn(needle, text)
        constants = dict(re.findall(r'constexpr uint32_t (kQwen\w+) = (0x[0-9A-F]+)u;', new_text('F13') + qwen_factory.F1))
        self.assertEqual({name: int(value, 16) for name, value in constants.items()},
                         dict(kQwenMaskTail=0x1, kQwenKvShare=0x2, kQwenQSlice=0x4, kQwenKvReadahead=0x8,
                              kQwenSliceAbiTag=0x51CE))

    def test_the_log_line_is_what_the_harness_parses(self):
        text = new_text('F18')
        self.assertIn('"[QWEN-SDPA] q-slice rows_per_kv={} pnht_full={} slice_tiles={} readahead={}"', text)
        self.assertIn('qwen_rows_per_kv, qwen_pnht_full, PNHt, qwen_kv_readahead);', text)
        sample = 'Op | INFO | [QWEN-SDPA] q-slice rows_per_kv=48 pnht_full=3 slice_tiles=2 readahead=true\n'
        self.assertEqual(harness.slice_lines(sample), [(48, 3, 2, True)])
        self.assertTrue(harness.SLICE_BINARY_MARKER.decode() in text)
        self.assertEqual(harness.SLICE_BINARY_MARKER.decode(), factory.SLICE_LOG_MARKER)
        # The F4 line itself is unchanged (stage 4 adds no second 'flags=' format), so the stage-3 parser and the
        # gate read stage-4 programs too.
        self.assertNotIn('flags=', stage4_text())


class ContractTests(unittest.TestCase):
    def test_the_recorded_shas_agree_everywhere(self):
        build, runner = read(BUILD), read(RUNNER)
        for text in (build, runner):
            self.assertIn('READER_SLICE=%s' % kernels.OUTPUTS[kernels.READER_SLICE_NAME], text)
            self.assertIn('WRITER_SLICE=%s' % kernels.OUTPUTS[kernels.WRITER_SLICE_NAME], text)
            self.assertIn('READER_QWEN=%s' % qwen_kernels.OUTPUTS_STAGE3[qwen_kernels.READER_NAME], text)
            self.assertIn('COMPUTE_QWEN=%s' % qwen_kernels.OUTPUTS_STAGE3[qwen_kernels.COMPUTE_NAME], text)
        self.assertIn('FACTORY_QWEN_STAGE4=%s' % factory.STAGE4_FACTORY, build)
        self.assertIn('FACTORY_QWEN_STAGE3=%s' % qwen_factory.QWEN_FACTORY_STAGE3, build)
        self.assertIn('FACTORY_BASE=%s' % factory.BASE_FACTORY, build)
        self.assertIn('WRITER_ALL=%s' % kernels.WRITER_BASE, build)
        self.assertIn('READER_ALL=%s' % qwen_kernels.READER_BASE, build)
        self.assertIn('K64G_TTNNCPP=%s' % harness.K64G_TTNNCPP_SHA256, build)
        self.assertIn('K64G_TTNNCPP_SHA256=%s' % harness.K64G_TTNNCPP_SHA256, runner)
        self.assertEqual(harness.SLICE_KERNELS, {'dataflow/' + name: kernels.OUTPUTS[name] for name in kernels.OUTPUTS})
        self.assertEqual(harness.KERNELS, {'dataflow/' + qwen_kernels.READER_NAME: qwen_kernels.OUTPUTS_STAGE3[qwen_kernels.READER_NAME],
                                           'compute/' + qwen_kernels.COMPUTE_NAME: qwen_kernels.OUTPUTS_STAGE3[qwen_kernels.COMPUTE_NAME]})
        import probe_k1_card_b as probe
        self.assertEqual(harness.KERNELS, probe.SERVED_KERNELS)

    def test_the_prefill_shas_are_build_k64gs(self):
        k64g = read(OPS / 'sdpa_prefill_chain' / 'build_k64g.sh')
        build = read(BUILD)
        for name in ('PF_BASE', 'PF_FACTORY', 'PF_READER_BASE', 'PF_READER', 'FACTORY_UNPATCHED', 'DATAFLOW_COMMON',
                     'RT_ARGS_COMMON', 'COMPUTE_ALL'):
            value = re.search(r'^%s=([0-9a-f]{64})$' % name, k64g, flags=re.M).group(1)
            self.assertIn('%s=%s' % (name, value), build, name)

    def test_the_kernel_names_agree(self):
        self.assertEqual((factory.READER_SLICE_NAME, factory.WRITER_SLICE_NAME),
                         (kernels.READER_SLICE_NAME, kernels.WRITER_SLICE_NAME))
        for name in (kernels.READER_SLICE_NAME, kernels.WRITER_SLICE_NAME):
            self.assertIn('"dataflow/%s"' % name, stage4_text())
            self.assertIn(name, read(BUILD))
            self.assertIn(name, read(RUNNER))

    def test_the_harness_refusal_needles_are_the_factorys_texts(self):
        text = stage4_text() + qwen_factory.F2 + qwen_factory.F9_NEW
        for name, _rows, _batches, _flags, needles, causal, mask_kind in harness.REFUSALS:
            with self.subTest(refusal=name):
                if mask_kind == 'short':
                    self.assertTrue(any(needle in text for needle in needles))   # F15's; validate's is the dump's
                    if DUMP.is_file():
                        self.assertIn('Expect same number of padded heads in mask as in Q', DUMP.read_text(encoding='utf-8'))
                else:
                    self.assertTrue(all(needle in text for needle in needles), needles)

    def test_the_abi_tag_and_the_flags_agree(self):
        self.assertEqual((factory.ABI_TAG, kernels.ABI_TAG), (0x51CE, 0x51CE))
        self.assertIn('kQwenSliceAbiTag = 0x51CEu', new_text('F13'))
        self.assertEqual((harness.TAIL, harness.SHARE, harness.SLICE, harness.READAHEAD),
                         (factory.FLAG_TAIL, factory.FLAG_SHARE, factory.FLAG_SLICE, factory.FLAG_READAHEAD))
        self.assertEqual(harness.MAGIC, 0x51DEC000)
        self.assertIn('constexpr std::size_t kQwenMagic = 0x51DEC000u;', qwen_factory.F1)

    def test_the_python_wiring_uses_the_factorys_rule_flags_and_marker(self):
        """scripts/ci/pooled_attention_replay.py (mounted by the arm) sends 0x4 exactly to the bundles the factory
        builds it for, 0x8 only with 0x2, and requires the stage-4 literal the factory prints."""
        ci = str(ROOT / 'scripts' / 'ci')
        if ci not in sys.path:
            sys.path.insert(0, ci)
        import pooled_attention_replay as replay
        self.assertEqual([rows for rows in range(1, 17) if replay.q_slice_saves(rows)],
                         [rows for rows in range(1, 17) if model.slice_applies(rows)])
        self.assertEqual((replay.QWEN_Q_SLICE, replay.QWEN_KV_READAHEAD), (factory.FLAG_SLICE, factory.FLAG_READAHEAD))
        self.assertEqual(replay.QWEN_SDPA_SLICE_MARKER, factory.SLICE_LOG_MARKER.encode())
        self.assertEqual(replay.QWEN_SDPA_SLICE_MARKER, harness.SLICE_BINARY_MARKER)
        self.assertEqual(replay.QWEN_DECODE_MAGIC, harness.MAGIC)
        served = frozenset({'tail', 'share', 'slice', 'readahead'})
        self.assertEqual(replay.mode_flags(served, 2, 8), 0xF)
        for rows in range(1, 9):
            for batches in (1, 2, 3):
                flags = replay.mode_flags(served, batches, rows)
                refusal = model.slice_rule(rows * 12)['refusal']
                self.assertEqual(bool(flags & 0x4), refusal is None, (rows, batches))
                self.assertFalse(flags & 0x8 and not flags & 0x2)

    def test_the_m3native_arm_and_gate_use_the_factorys_flags_literal_and_kernel_names(self):
        """scripts/ci/lever_n_m3native_gate.py demands the F4 line with every requested flag (0x7 for
        tail,share,slice) and F18's q-slice line; lever_n_m3native_run_arm.sh greps the graft binary for the same
        literal and requires both slice kernels by the names F17 selects."""
        ci = str(ROOT / 'scripts' / 'ci')
        if ci not in sys.path:
            sys.path.insert(0, ci)
        import lever_n_m3native_gate as gate
        self.assertEqual(gate.SDPA_SLICE_MARKER, factory.SLICE_LOG_MARKER)
        self.assertEqual(gate.SDPA_MODE_FLAGS, dict(tail=factory.FLAG_TAIL, share=factory.FLAG_SHARE,
                                                     slice=factory.FLAG_SLICE, readahead=factory.FLAG_READAHEAD))
        markers = gate.sdpa_mode_markers({'tail', 'share', 'slice'})
        self.assertEqual(markers[2:], ['[QWEN-SDPA] flags=0x7 ', factory.SLICE_LOG_MARKER])
        # The factory's own lines as fmt prints them ({:#x} and {} format as Python's do) carry both markers:
        # F4 is apply_factory_qwen's (stage 4 adds no second 'flags=' format), F18 the stage-4 one.
        f4 = re.search(r'"(\[QWEN-SDPA\] flags=[^"]*)"', qwen_factory.F4).group(1)
        f18 = re.search(r'"(\[QWEN-SDPA\] q-slice [^"]*)"', new_text('F18')).group(1)
        self.assertTrue(f4.format(0x7, 2, 2, 4104, 4104, 'true', 4, 837696).startswith(markers[2]))
        self.assertFalse(f4.format(0x3, 2, 3, 4104, 4104, 'true', 4, 837696).startswith(markers[2]))
        self.assertTrue(f18.format(48, 3, 2, 'false').startswith(markers[3]))
        arm = read(ROOT / 'scripts' / 'ci' / 'lever_n_m3native_run_arm.sh')
        self.assertIn("grep -a -q -F -- '%s' \"$KOPGRAFT64/_ttnncpp.so\"" % factory.SLICE_LOG_MARKER, arm)
        self.assertIn('sdpa_slice=(dataflow/%s dataflow/%s)' % (factory.READER_SLICE_NAME, factory.WRITER_SLICE_NAME), arm)

    def test_the_cpu_workflow_runs_both_suites(self):
        workflow = read(ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml')
        self.assertIn("python -B -m unittest discover -s optimisation/ttnn-op/sdpa_decode_slice -p 'test_sdpa_decode_slice_*.py'",
                      workflow)
        self.assertEqual(sorted(path.name for path in HERE.glob('test_sdpa_decode_slice_*.py')),
                         ['test_sdpa_decode_slice_card_b.py', 'test_sdpa_decode_slice_sources.py'])

    def test_the_runner_and_build_are_lf_and_the_build_is_byte_collated(self):
        for path in (BUILD, RUNNER):
            self.assertNotIn(b'\r', path.read_bytes())
        build = read(BUILD)
        self.assertIn('export LC_ALL=C', build)
        # K64g carries sdpa/: the base check is build_k64g.sh's inverted (required, and equal after the build).
        self.assertIn('for part in _ttnncpp.so _ttnn.so attn_prep nlp_concat_heads_decode sdpa_decode sdpa MANIFEST.sha256', build)
        self.assertIn('for op in attn_prep nlp_concat_heads_decode sdpa; do', build)
        self.assertNotIn('already carries sdpa/', build)
        # Every image named at step 0 is compared at step 7 (sdpa_decode/, sdpa/ and its binary's QWEN strings).
        self.assertIn('*" $GATE_IMAGE "*) ;;', build)
        self.assertIn('docker cp "$cid:$D" "$W/image-sdpa_decode"', build)
        self.assertIn('image_lost=$(comm -23 "$W/image-qwen-strings.txt" "$W/graft-qwen-strings.txt")', build)
        self.assertIn('restore_ttbuild', build)
        self.assertIn('need "graft sdpa_decode factory (on disk, audited)"', build)
        self.assertIn('GATE_IMAGE=sha256:70c27e7539db757e3b166f0d14ccdc32edf86532017b6f2f50310fcaad855c3f', build)
        self.assertIn('IMAGE=${IMAGE:-sha256:70c27e7539db757e3b166f0d14ccdc32edf86532017b6f2f50310fcaad855c3f}', read(RUNNER))

    @unittest.skipUnless(BASH, 'bash not found')
    def test_the_scripts_parse(self):
        for path in (BUILD, RUNNER):
            result = subprocess.run([BASH, '-n', path.as_posix()], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)


class IndexModelTests(unittest.TestCase):
    def test_the_rule_for_every_group_size(self):
        applies = [rows for rows in range(1, 17) if model.slice_applies(rows)]
        self.assertEqual([rows for rows in range(1, 9) if model.slice_applies(rows)], [6, 7, 8])
        for rows in range(1, 17):
            folded = rows * 12
            full = -(-folded // 32)
            G = folded // 2
            widest = max(-(-((h + 1) * G) // 32) - (h * G) // 32 for h in range(2))
            self.assertEqual(rows in applies, widest < full, rows)
        g8, g7, g4, g16 = (model.slice_rule(rows * 12) for rows in (8, 7, 4, 16))
        self.assertEqual((g8['pnht_full'], g8['slice_tiles'], g8['starts'], g8['refusal']), (3, 2, [0, 1], None))
        self.assertEqual((g7['rows_per_kv'], g7['starts'], g7['slice_tiles']), (42, [0, 1], 2))
        self.assertEqual(g4['refusal'], '[QWEN-SDPA] q-slice saves no tile')
        self.assertEqual((g16['pnht_full'], g16['slice_tiles'], g16['starts']), (6, 3, [0, 3]))
        self.assertEqual(model.slice_rule(96, 1)['refusal'], '[QWEN-SDPA] q-slice needs num_q_heads')
        self.assertEqual(model.slice_rule(95, 2)['refusal'], '[QWEN-SDPA] q-slice needs num_q_heads')
        self.assertEqual(model.program_pnht(96, 0x7), 2)
        self.assertEqual(model.program_pnht(96, 0xB), 3)
        with self.assertRaises(ValueError):
            model.program_pnht(48, 0x5)

    def shapes(self):
        for rows in range(1, 17):
            if model.slice_applies(rows):
                for batches in (1, 2, 3):
                    yield rows * 12, batches

    def test_every_kept_row_is_written_once_to_the_legacy_face_lines_from_its_own_row_tile(self):
        for num_q_heads, batches in self.shapes():
            with self.subTest(rows=num_q_heads, batches=batches):
                for entry in range(batches):
                    seen = {}
                    for head in range(2):
                        sliced = model.writer_writes(entry, head, num_q_heads, True)
                        legacy = model.legacy_writer_writes(entry, head, num_q_heads)
                        self.assertEqual([(row, dram, offset) for row, _slot, dram, offset in sliced],
                                         [(row, dram, offset) for row, _slot, dram, offset in legacy])
                        pnht = model.slice_rule(num_q_heads)['slice_tiles']
                        for row, slot, _dram, _offset in sliced:
                            self.assertTrue(0 <= slot < pnht * model.HEAD_DIM_TILES)
                            self.assertEqual(model.slot_row_tile(slot, head, num_q_heads, True), row // 32)
                            seen[row] = seen.get(row, 0) + 1
                    self.assertEqual(sorted(seen), list(range(num_q_heads)))
                    self.assertTrue(all(count == 2 * model.HEAD_DIM_TILES for count in seen.values()))

    def test_q_and_mask_tiles_are_the_legacy_tiles_of_the_heads_row_tiles(self):
        for num_q_heads, batches in self.shapes():
            rule = model.slice_rule(num_q_heads)
            full, pnht = rule['pnht_full'], rule['slice_tiles']
            width = 8448 // 32
            for entry in range(batches):
                for head in range(2):
                    start = rule['starts'][head]
                    legacy_q = model.q_tiles(entry, head, num_q_heads, False)
                    self.assertEqual(model.q_tiles(entry, head, num_q_heads, True),
                                     legacy_q[start * 8:(start + pnht) * 8])
                    legacy_mask = model.mask_tiles(entry, head, num_q_heads, False, width, bmask=batches)
                    self.assertEqual(model.mask_tiles(entry, head, num_q_heads, True, width, bmask=batches),
                                     legacy_mask[start * 8:(start + pnht) * 8])
                    self.assertEqual(len(legacy_q), full * 8)

    def test_the_bug_variants_are_what_the_card_controls_must_catch(self):
        rows, batches = 96, 2

        def value(entry, head, q_row, mask):
            return (entry, head, q_row, mask)

        good = model.output_rows(batches, rows, True, value)
        self.assertEqual(good, model.output_rows(batches, rows, False, value))
        self.assertEqual(sorted(good), [(b, r) for b in range(batches) for r in range(rows)])
        self.assertTrue(all(q_row == row and mask == (entry, row) and head == row // 48
                            for (entry, row), (_e, head, q_row, mask) in good.items()))
        for bug in ('r7', 'r8_row', 'r8_stride', 'w2', 'w3'):
            with self.subTest(bug=bug):
                self.assertNotEqual(model.output_rows(batches, rows, True, value, bugs=(bug,), garbage='G'), good)
        # R8's two slips: the row offset moves head 1's rows (both entries), the stride moves entry 1's only.
        row_slip = model.output_rows(batches, rows, True, value, bugs=('r8_row',))
        stride_slip = model.output_rows(batches, rows, True, value, bugs=('r8_stride',))
        self.assertEqual({key for key in good if row_slip[key] != good[key]}, {(b, r) for b in range(2) for r in range(48, 96)})
        self.assertEqual({key[0] for key in good if stride_slip[key] != good[key]}, {1})
        # Without the W3 rebase head 1 reads the wrong slot or past the CB (garbage).
        w3 = model.output_rows(batches, rows, True, value, bugs=('w3',), garbage='G')
        self.assertIn('G', [w3[(0, r)] for r in range(64, 96)])

    def test_the_cb_bytes_are_the_stage_three_tables_and_the_designs(self):
        for (capacity, pnht), expected in card.CB_BYTES.items():
            self.assertEqual(model.cb_bytes(capacity, pnht), expected)
        self.assertEqual(model.cb_bytes(131328, 3) - model.cb_bytes(131328, 2), 251904)        # section 0
        self.assertEqual(model.cb_bytes(131328, 6), 1804352)       # G16 unsliced (section 6): does not fit L1
        self.assertEqual(model.cb_bytes(33024, 6), 1798208)
        self.assertEqual(model.page_table_bytes(131328), 8256)
        self.assertEqual(model.page_table_bytes(33024), 2112)
        self.assertEqual(model.cb_bytes(131328, 2, rounds=5), 837696)    # roofline #3: 27 cores per head, 5 rounds


class ProtocolModelTests(unittest.TestCase):
    def test_both_leaders_are_deadlock_and_hazard_free(self):
        for label, result in protocol.sweep(masks=('tail', 'every'), shapes=[(2, 1), (2, 2), (2, 4), (3, 2)]):
            with self.subTest(case=label):
                self.assertTrue(result['complete'])
                self.assertIsNone(result['violation'])
                self.assertIsNone(result['deadlock'])
                self.assertGreater(result['finished'], 0)

    def test_the_broken_variants_are_caught(self):
        no_barrier = protocol.Model(2, 3, 'no_barrier', 'tail').explore()
        self.assertIn('compute expected chunk', no_barrier['violation'] or '')
        ready_first = protocol.Model(2, 3, 'readahead', 'tail', ready_first=True).explore()
        self.assertIn('has not reserved', ready_first['violation'] or '')

    def test_the_model_programs_are_the_kernels_order(self):
        leader = protocol.leader_program(3, 'readahead', protocol.masks_for('tail', 3))
        ops = [op[0] for op in leader]
        self.assertEqual(ops[:2], ['read', 'read'])                        # the prologue
        last = len(ops) - 1 - ops[::-1].index('mcast')
        self.assertEqual(ops[last:], ['mcast', 'mask', 'wbarrier', 'valid', 'wbarrier'])  # no read past the end
        self.assertEqual(sum(1 for op in leader if op[0] == 'read'), 6)


class StubCompileTests(unittest.TestCase):
    def test_the_kernels_and_factory_blocks_compile_against_the_stubs(self):
        import stub_compile
        gxx = stub_compile.find_gxx()
        if gxx is None:
            self.skipTest('no g++ (set QWEN_PF_GXX)')
        code, output = stub_compile.check_factory(gxx)
        self.assertEqual(code, 0, output)
        if not DUMP.is_file() or not (PREFILL_SRC / 'kernels').is_dir():
            self.skipTest('no dump or prefill tree for the kernel stubs')
        failed = [(label, detail) for label, ok, detail in stub_compile.check_kernels(gxx, DUMP, PREFILL_SRC) if not ok]
        self.assertEqual(failed, [])

    def test_a_runner_without_a_compiler_skips_rather_than_fails(self):
        """The CPU workflow's runner may have no g++: find_gxx then returns None (nothing on PATH, no QWEN_PF_GXX,
        no local arm-none-eabi-g++) and the compile test skips; the CLI says so and exits 2."""
        import io
        from contextlib import redirect_stderr
        from unittest import mock
        import stub_compile
        missing = str(Path(tempfile.gettempdir()) / 'no-such-compiler-dir' / 'g++')
        with mock.patch.dict(os.environ, {'QWEN_PF_GXX': ''}), mock.patch.object(stub_compile.shutil, 'which', return_value=None), \
                mock.patch.object(stub_compile, 'LOCAL_ARM_GXX', missing):
            self.assertIsNone(stub_compile.find_gxx())
            result = unittest.TestResult()
            StubCompileTests('test_the_kernels_and_factory_blocks_compile_against_the_stubs').run(result)
            self.assertEqual((result.errors, result.failures), ([], []))
            self.assertEqual([reason for _case, reason in result.skipped], ['no g++ (set QWEN_PF_GXX)'])
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(stub_compile.main([]), 2)
            self.assertIn('no g++ found', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
