"""CPU checks for the [QWEN-SDPA] stage-1 graft sources (no device, no ttnn).

  - the two committed kernels are the recorded originals plus exactly R1-R3 / C1-C2
    (reverting the edits reproduces the base sha), and hash to the recorded outputs;
  - apply_factory_qwen refuses anything but 3e0a69af and, given the base, reproduces the
    recorded F1-F8 factory and inverts back to it;
  - every constant that crosses a file boundary agrees: the sentinel and flag values, the
    binary marker, the factory's log format against the card-M parser and the gate marker,
    the TT_FATAL texts the card-M test expects, the shas build_k64e.sh enforces, the op
    directory the arm mounts;
  - the card-M test's host helpers (mask formula, mask/query builders, log parser).

The optional base inputs are found at QWEN_SDPA_SOURCES_DUMP (probe_sdpa_decode_sources.py
output) and QWEN_SDPA_FACTORY_BASE (a 3e0a69af factory); the tests that need them skip
without them.

    py -3.11 -B -m unittest test_sdpa_decode_qwen_sources      (from this directory)
"""

import hashlib
import os
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
CI = HERE.parents[2] / 'scripts' / 'ci'
for path in (str(HERE), str(CI)):
    if path not in sys.path:
        sys.path.insert(0, path)

import apply_factory_qwen as factory  # noqa: E402
import make_qwen_kernels as kernels  # noqa: E402
import test_sdpa_decode_qwen_card_m as card  # noqa: E402

NL = chr(10)
JOB = Path('C:/Users/liamb/.claude/jobs/8376c877/tmp')
DUMP = Path(os.environ.get('QWEN_SDPA_SOURCES_DUMP', JOB / 'sdpa-decode-sources.txt'))
FACTORY_BASE = Path(os.environ.get('QWEN_SDPA_FACTORY_BASE', JOB / 'sdpa_decode_program_factory.cpp.3e0a69af'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


class KernelSourceTests(unittest.TestCase):
    def test_the_committed_kernels_are_the_bases_plus_exactly_the_edits(self):
        for name in (kernels.READER_NAME, kernels.COMPUTE_NAME):
            with self.subTest(kernel=name):
                built = (HERE / name).read_bytes()
                self.assertEqual(sha(built), kernels.OUTPUTS[name])
                self.assertNotIn(b'\r', built)
                self.assertEqual(sha(kernels.revert_edits(name, built)), kernels.BASES[name])

    def test_the_edits_carry_the_tail_predicate_on_both_sides(self):
        reader = (HERE / kernels.READER_NAME).read_text(encoding='utf-8')
        compute = (HERE / kernels.COMPUTE_NAME).read_text(encoding='utf-8')
        predicate = 'if (!mask_tail || k_chunk == k_num_chunks - 1) {'
        self.assertEqual(reader.count(predicate), 1)
        self.assertEqual(compute.count(predicate), 1)
        self.assertIn('constexpr bool mask_tail = get_compile_time_arg_val(32) == 1;', compute)
        self.assertIn('constexpr uint32_t qwen_cta = attention_sink_args.next_compile_time_args_offset();', reader)
        # Every mask read in the reader goes through the predicate; the non-paged path is untouched.
        self.assertEqual(reader.count('read_mask_chunk<'), 1)
        self.assertIn('mask_width_t, Sk_chunk_t_dynamic, mask_chunk_tiles, mask_start_tile_id, mask_reader);', reader)
        self.assertEqual(compute.count('add_block_inplace<true>(cb_qk_im, cb_mask_in'), 1)

    def test_a_wrong_base_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'base is'):
            kernels.apply_edits(kernels.READER_NAME, b'// not the reader' + NL.encode())

    def test_the_kernels_regenerate_from_the_dump(self):
        if not DUMP.is_file():
            self.skipTest('no sources dump at %s' % DUMP)
        text = DUMP.read_bytes().decode('utf-8')
        for name, path in kernels.DUMP_PATHS.items():
            with self.subTest(kernel=name):
                base = kernels.from_dump(text, path)
                self.assertEqual(sha(base), kernels.BASES[name])
                self.assertEqual(kernels.apply_edits(name, base), (HERE / name).read_bytes())


class FactoryPatchTests(unittest.TestCase):
    def test_anything_but_the_tree_scratch_factory_is_refused(self):
        for source in (b'', b'int main() {}' + NL.encode()):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'unexpected factory'):
                factory.patch(source)

    def test_the_base_factory_patches_to_the_recorded_bytes_and_back(self):
        if not FACTORY_BASE.is_file():
            self.skipTest('no 3e0a69af factory at %s' % FACTORY_BASE)
        base = FACTORY_BASE.read_bytes()
        self.assertEqual(sha(base), factory.BASE_FACTORY)
        text = base.decode('utf-8')
        for label, old, new in factory.EDITS:
            with self.subTest(edit=label):
                self.assertEqual(text.count(old), 1)
        patched = factory.patch(base)
        self.assertEqual(sha(patched), factory.QWEN_FACTORY)
        self.assertEqual(factory.unpatch(patched), base)
        # F7 appends at index 32: the legacy compute arg list has exactly 32 entries.
        body = text[text.index('std::vector<uint32_t> compute_compile_time_args_common = {'):]
        body = body[body.index('{') + 1:body.index('};')]
        self.assertEqual(len([item for item in body.split(',') if item.strip()]), 32)
        # F6 appends after the last TensorAccessorArgs block of the reader list.
        reader = patched.decode('utf-8')
        f6 = reader.index('reader_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_mask_tail));')
        self.assertGreater(f6, reader.rindex('.append_to(reader_compile_time_args_common);'))

    def test_every_new_factory_branch_is_gated_on_qwen_mode(self):
        for label, old, new in factory.EDITS:
            added = new.replace(old, '', 1) if new.startswith(old) else new
            with self.subTest(edit=label):
                self.assertTrue('qwen_mode' in added or label in ('F2', 'F5') or 'kQwen' in added, label)
        self.assertIn('if (qwen_mode) {', factory.F2)
        self.assertIn('qwen_mode || (scratch_rounds_override', factory.F3_NEW)


class ContractTests(unittest.TestCase):
    """Values that cross file boundaries."""

    def f4_format(self):
        match = re.search(r'log_info\(tt::LogOp, "([^"]+)"', factory.F4)
        return match.group(1)

    def test_the_sentinel_and_flags_agree_everywhere(self):
        import pooled_attention_replay as replay
        magic = int(re.search(r'kQwenMagic = (0x[0-9A-F]+)u;', factory.F1).group(1), 16)
        self.assertEqual(magic, replay.QWEN_DECODE_MAGIC)
        self.assertEqual(int(re.search(r'kQwenMagicMask = (0x[0-9A-F]+)u;', factory.F1).group(1), 16), 0xFFFFFF00)
        self.assertEqual(int(re.search(r'kQwenMaskTail = (0x[0-9a-f]+)u;', factory.F1).group(1), 16), replay.QWEN_MASK_TAIL)
        self.assertEqual(int(re.search(r'kQwenKvShare = (0x[0-9a-f]+)u;', factory.F1).group(1), 16), replay.QWEN_KV_SHARE)
        self.assertEqual(card.TAIL, magic | replay.QWEN_MASK_TAIL)
        self.assertEqual(card.SHARE_STAGE3, magic | replay.QWEN_KV_SHARE)
        self.assertEqual(card.NO_FLAGS, magic)

    def test_the_log_line_the_factory_prints_is_what_the_parsers_and_the_gate_expect(self):
        import lever_n_m3native_gate as gate
        import pooled_attention_replay as replay
        text = self.f4_format()
        self.assertTrue(text.startswith(replay.QWEN_SDPA_BINARY_MARKER.decode()))
        self.assertTrue(text.startswith(card.BINARY_MARKER.decode()))
        # fmt's {:#x} and {} on these types format as Python's do; bool prints false/true.
        line = text.format(1, 3, 2, 1032, 1032, 'false', 4, 790592)
        self.assertEqual(line, '[QWEN-SDPA] flags=0x1 B=3 PNHt=2 St=1032 mask_width_t=1032 kv_share=false '
                               'scratch_slots=4 cb_bytes=790592')
        self.assertEqual(card.factory_lines('prefix ' + line + NL),
                         [dict(flags='0x1', B=3, PNHt=2, St=1032, mask_width_t=1032, kv_share='false',
                               scratch_slots=4, cb_bytes=790592)])
        self.assertIn(gate.SDPA_MODES_MARKERS[2], line + NL)
        self.assertIn('[QWEN-SDPA] flags=', factory.MARKERS)

    def test_the_refusals_the_card_m_test_expects_are_the_factorys_texts(self):
        source = inspect_source(card.timing_and_controls)
        needles = re.findall(r"SHARE_STAGE3, '([^']+)'|UNKNOWN_FLAG, '([^']+)'|TAIL, '([^']+)'|NO_FLAGS, '([^']+)'", source)
        needles = [next(part for part in group if part) for group in needles]
        self.assertEqual(len(needles), 5)
        for needle in needles:
            with self.subTest(needle=needle):
                self.assertIn(needle, factory.F2)

    def test_the_kernel_names_the_factory_selects_are_the_committed_files(self):
        self.assertIn('"dataflow/%s"' % kernels.READER_NAME, factory.F8R_NEW)
        self.assertIn('"compute/%s"' % kernels.COMPUTE_NAME, factory.F8C_NEW)

    def test_build_k64e_enforces_the_recorded_shas(self):
        script = (HERE / 'build_k64e.sh').read_text(encoding='utf-8')
        self.assertNotIn(chr(13), script)
        values = dict(re.findall(r'^([A-Z_]+)=([0-9a-f]{8,64})$', script, flags=re.M))
        self.assertEqual(values['FACTORY_BASE'], factory.BASE_FACTORY)
        self.assertEqual(values['FACTORY_QWEN'], factory.QWEN_FACTORY)
        self.assertEqual(values['READER_ALL'], kernels.READER_BASE)
        self.assertEqual(values['COMPUTE_ALL'], kernels.COMPUTE_BASE)
        self.assertEqual(values['READER_QWEN'], kernels.OUTPUTS[kernels.READER_NAME])
        self.assertEqual(values['COMPUTE_QWEN'], kernels.OUTPUTS[kernels.COMPUTE_NAME])
        import sdpa_tree_scratch
        self.assertEqual(values['FACTORY_BASE'], sdpa_tree_scratch.PATCHED_FACTORY_SHA256)
        self.assertEqual(values['FACTORY_UNPATCHED'], sdpa_tree_scratch.HASHES['sdpa_decode_program_factory.cpp'])
        self.assertEqual(values['WRITER_ALL'], sdpa_tree_scratch.HASHES['kernels/dataflow/writer_decode_all.cpp'])
        self.assertEqual(values['COMPUTE_ALL'], sdpa_tree_scratch.HASHES['kernels/compute/sdpa_flash_decode.cpp'])

    def test_build_k64e_leaves_build_release_on_the_restored_factory(self):
        """docker cp keeps the saved factory's old mtime: without a touch and a rebuild, ninja
        calls the unity TU up to date and the next ttbuild graft links the qwen factory."""
        script = (HERE / 'build_k64e.sh').read_text(encoding='utf-8')
        restore = script[script.index('restore_ttbuild() {'):script.index('on_exit() {')]
        self.assertIn('docker exec ttbuild touch "$F"', restore)
        self.assertLess(restore.index('ttbuild:$F'), restore.index('touch "$F"'), 'touch after the copy')
        staged = script[script.index('elif [ "$cur" = "$FACTORY_BASE" ]'):script.index('need "saved base factory"')]
        self.assertLess(staged.index('ttbuild:$F'), staged.index('touch "$F"'))
        step6 = script[script.index('# ---------- 6.'):script.index('# ---------- 7.')]
        ninja = 'ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so'
        self.assertLess(step6.index('restore_ttbuild'), step6.index(ninja), 'rebuild after the restore')
        self.assertIn("grep -caF -- '[QWEN-SDPA] flags='", step6)
        self.assertIn('rebuilt=1', step6)
        self.assertIn('trap on_exit EXIT', script)

    def test_the_arm_and_the_runner_mount_the_directory_build_k64e_assembles(self):
        target = '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode'
        build = (HERE / 'build_k64e.sh').read_text(encoding='utf-8')
        self.assertIn('OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer' + NL, build)
        self.assertIn('D=$OPS/sdpa_decode' + NL, build)
        self.assertIn('docker cp ttbuild:$D "$G/sdpa_decode"', build)
        self.assertIn('$KOPGRAFT64/sdpa_decode:%s:ro' % target, (CI / 'lever_n_m3native_run_arm.sh').read_text(encoding='utf-8'))
        self.assertIn('$G/sdpa_decode:%s:ro' % target, (HERE / 'run_card_m.sh').read_text(encoding='utf-8'))


def inspect_source(function):
    import inspect
    return inspect.getsource(function)


class CardMHelperTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_mask_positions_follow_the_refresh_kernel(self):
        """attention_mask_replay.cpp: position = start + offset + batch * rows + (head % (rows * 6)) / 6."""
        source = (CI / 'attention_mask_replay.cpp').read_text(encoding='utf-8')
        self.assertIn('const uint32_t position = start + offset + batch * rows + (head % (rows * 6)) / 6;', source)
        self.assertIn('cache_position > position ? 0xff80 : 0', source)
        positions = card.mask_positions(1000, (0, 4, 8))
        self.assertEqual(len(positions), 3)
        self.assertEqual(positions[0][:6], [1000] * 6)
        self.assertEqual(positions[0][6:12], [1001] * 6)
        self.assertEqual(positions[0][24:30], [1000] * 6, 'the second KV group repeats the tokens')
        self.assertEqual(positions[2][47], 1000 + 0 + 8 + 3)
        self.assertEqual(card.mask_positions(1000, (12,))[0][23], 1000 + 12 + 3)

    def test_the_masks_are_zero_but_the_tail_and_narrow_is_the_wide_tail(self):
        torch = self.torch
        capacity, start = 2304, 2304 - 256 + 7
        wide = card.build_mask(torch, capacity, start, (0, 4, 8))
        self.assertEqual(tuple(wide.shape), (3, 1, 48, capacity))
        self.assertEqual(wide.dtype, torch.bfloat16)
        self.assertTrue(bool((wide[..., :capacity - 256] == 0).all()))
        tail = wide[..., capacity - 256:].float()
        self.assertTrue(bool(((tail == 0) | (tail == float('-inf'))).all()))
        # entry 0, row 0 sees cache positions up to start exactly
        visible = (tail[0, 0, 0] == 0).nonzero().flatten()
        self.assertEqual(int(visible.max()) + capacity - 256, start)
        self.assertTrue(bool((tail[0, 0, 0, :7 + 1] == 0).all()))
        narrow = card.build_mask(torch, capacity, start, (0, 4, 8), width=256)
        self.assertTrue(torch.equal(narrow, wide[..., capacity - 256:]))
        half = card.build_mask(torch, capacity, start, (12,), width=512)
        self.assertEqual(tuple(half.shape), (1, 1, 48, 512))
        planted = card.build_mask(torch, capacity, start, (0, 4, 8), plant=True)
        self.assertTrue(bool((planted[..., 0].float() == float('-inf')).all()))
        self.assertTrue(torch.equal(planted[..., 1:], wide[..., 1:]))
        with self.assertRaises(ValueError):
            card.build_mask(torch, capacity, start, (0,), width=256, plant=True)

    def test_queries_and_the_pool_geometry(self):
        torch = self.torch
        self.assertEqual(card.num_blocks(33024), 33024 // 64 + 64)
        self.assertEqual(card.num_blocks(131328), 2052 + 64)
        for bad in (33000, 0, 33024.0):
            with self.assertRaises(ValueError):
                card.num_blocks(bad)
        normal = card.build_query(torch, 3, 0, 'normal')
        self.assertEqual(tuple(normal.shape), (1, 3, 48, 256))
        self.assertTrue(torch.equal(normal, card.build_query(torch, 3, 0, 'normal')), 'seeded')
        zero = card.build_query(torch, 1, 0, 'zeroq')
        self.assertTrue(bool((zero[0, :, ::4] == 0).all()))
        self.assertFalse(bool((zero[0, :, 1::4] == 0).all()))
        keys = torch.randn(10, 2, 64, 256).to(torch.bfloat16)
        table = torch.randperm(10)[:4].to(torch.int32)
        peaky = card.build_query(torch, 3, 0, 'peaky', keys, table).float()
        scores = torch.einsum('bhd,ntpd->bhntp', peaky[0], keys.float()) / 16
        self.assertGreater(float(scores.max()), 5.0, 'aligned keys dominate')
        with self.assertRaises(ValueError):
            card.build_query(torch, 3, 0, 'peaky')

    def test_the_factory_line_check(self):
        def line(capacity, batches, width, **changes):
            values = dict(flags='0x1', B=batches, PNHt=2, St=capacity // 32, mask_width_t=width, kv_share='false',
                          scratch_slots=4, cb_bytes=card.CB_BYTES[capacity])
            values.update(changes)
            return values

        complete = [line(c, b, w) for c in (33024, 131328) for b in (3, 1) for w in (c // 32, 8)]
        self.assertEqual(card.check_factory_lines(complete, (33024, 131328)), [])
        self.assertEqual(card.check_factory_lines(complete + complete, (33024, 131328)), [], 'cache off: repeats')
        self.assertEqual(len(card.check_factory_lines(complete[1:], (33024, 131328))), 1)
        self.assertEqual(card.check_factory_lines([entry for entry in complete if entry['mask_width_t'] != 8],
                                                  (33024, 131328), narrow=False), [])
        wrong = [dict(entry, scratch_slots=15) if index == 0 else entry for index, entry in enumerate(complete)]
        self.assertIn('scratch_slots', card.check_factory_lines(wrong, (33024, 131328))[0])
        self.assertEqual(card.check_factory_lines([line(33024, 3, 1032, cb_bytes=1)], (2304,))[-1],
                         '1 [QWEN-SDPA] lines for no tested capacity')



# The paged decode binding, sdpa_decode_nanobind.cpp:82-106 in the dump: four positionals,
# then kw_only(). A nanobind call with any other keyword raises TypeError before validate.
PAGED_POSITIONAL = ('input_tensor_q', 'input_tensor_k', 'input_tensor_v', 'page_table_tensor')
PAGED_KEYWORDS = ('is_causal', 'attn_mask', 'cur_pos_tensor', 'attention_sink', 'scale', 'sliding_window_size',
                  'memory_config', 'program_config', 'compute_kernel_config', 'paged_cache_geometry',
                  'cache_position_modulo')


class StrictTtnn:
    """A ttnn stand-in whose paged decode refuses what the real binding and validate refuse."""
    int32, bfloat16, ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'int32', 'bf16', 'rm', 'tile', 'dram'

    class Tensor:
        def __init__(self, dtype, layout, length):
            self.dtype, self.layout, self.length = dtype, layout, length

    def __init__(self):
        self.calls = []
        self.transformer = self

    def SDPAProgramConfig(self, **options):  # noqa: N802 - mirrors ttnn
        return options

    def from_torch(self, host, *, device, dtype, layout, memory_config):
        return self.Tensor(dtype, layout, host.shape[-1])

    def paged_scaled_dot_product_attention_decode(self, *args, **options):
        if len(args) != len(PAGED_POSITIONAL):
            raise TypeError('incompatible function arguments: %d positionals' % len(args))
        unknown = sorted(set(options) - set(PAGED_KEYWORDS))
        if unknown:
            raise TypeError('incompatible function arguments: %s' % unknown)
        if options.get('is_causal', True):
            position = options.get('cur_pos_tensor')
            if position is None:
                raise RuntimeError('Must have cur_pos tensor for paged attention in causal mode')
            if position.dtype != self.int32 or position.layout != self.ROW_MAJOR_LAYOUT:
                raise RuntimeError('Expect cur_pos to be INT32 ROW_MAJOR')
            if position.length != args[3].rows:
                raise RuntimeError('cur_pos must have batch size equal to Q')
            if (options['program_config']['q_chunk_size'] & 0xFFFFFF00) == (card.TAIL & 0xFFFFFF00):
                raise RuntimeError('TT_FATAL [QWEN-SDPA] modes are non-causal, full-window and take no cur_pos tensor')
        self.calls.append(options)
        return 'out'


class CardMCallTests(unittest.TestCase):
    """Case.call must be a call the real paged binding accepts (the card-M run on hardware found
    a list-valued cur_pos= keyword the binding does not have)."""

    def setUp(self):
        import torch
        self.torch = torch
        self.ttnn = StrictTtnn()
        grid = type('Grid', (), {'x': 11, 'y': 10})()
        device = type('Device', (), {'compute_with_storage_grid_size': lambda self: grid})()
        self.case = card.Case.__new__(card.Case)
        self.case.ttnn, self.case.torch, self.case.device, self.case.owned = self.ttnn, torch, device, []
        self.case.k = self.case.v = 'kv'
        self.case.tables = {}
        for rows in (3, 1):
            self.case.tables[rows] = self.ttnn.Tensor(self.ttnn.int32, self.ttnn.ROW_MAJOR_LAYOUT, 2)
            self.case.tables[rows].rows = rows
        self.query = type('Query', (), {'shape': (1, 3, 48, 256)})()

    def test_the_fake_binding_is_the_dumped_binding(self):
        if not DUMP.is_file():
            self.skipTest('no sources dump')
        text = DUMP.read_text(encoding='utf-8', errors='replace')
        begin = text.index('nb::arg("page_table_tensor").noconvert(),')
        end = text.index('nb::arg("cache_position_modulo") = nb::none());', begin)
        names = re.findall(r'nb::arg\("([a-z_]+)"\)', text[begin:end + 60])
        self.assertEqual(tuple(names), PAGED_POSITIONAL[3:] + PAGED_KEYWORDS)
        self.assertNotIn('cur_pos', names)

    def test_the_equality_and_timing_calls_are_accepted(self):
        for sentinel in (card.LEGACY, card.TAIL):
            for mask in (None, 'mask'):
                self.assertEqual(self.case.call(self.query, mask, sentinel), 'out')
        self.assertEqual(len(self.ttnn.calls), 4)
        self.assertEqual(self.ttnn.calls[0]['is_causal'], False)

    def test_the_causal_refusal_reaches_the_factorys_text(self):
        positions = self.case.upload(self.torch.full((3,), 33023, dtype=self.torch.int32), self.ttnn.int32)
        result = card.expect_refusal(self.case, self.query, None, card.TAIL, 'modes are non-causal',
                                     is_causal=True, cur_pos_tensor=positions)
        self.assertEqual((result['refused'], result['matched']), (True, True), result['message'])
        legacy = self.case.call(self.query, None, card.LEGACY, is_causal=True, cur_pos_tensor=positions)
        self.assertEqual(legacy, 'out', 'the same causal call without the sentinel is a valid call')

    def test_timing_and_controls_passes_a_cur_pos_tensor(self):
        source = inspect_source(card.timing_and_controls)
        self.assertNotIn('cur_pos=', source)
        self.assertIn('cur_pos_tensor=positions', source)
        self.assertIn('dtype=torch.int32), case.ttnn.int32)', source)


if __name__ == '__main__':
    unittest.main()
