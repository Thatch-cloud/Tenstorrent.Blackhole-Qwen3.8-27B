"""CPU tests of the three K-split kernels and their generators (make_ks_reader.py, make_ks_writer.py, make_ks_compute.py).

  - the committed kernels are what the generators make from their bases (the committed chain reader; the pinned writer and compute kernel, committed as fixtures), at the recorded shas;
  - the fixtures are the pinned tt-metal files at the shas the repository already records (compute_common.hpp and sdpa.cpp: scripts/ci/proposal_native_attention_gate.py);
  - the slice arithmetic is one text in the three kernels, and the C++ gives the Python planner's numbers (compiled and executed with g++ on a grid);
  - the compute kernel's call of the pinned ring loop matches the pinned template: the parameter lists by count, the positions that carry the semantics by name, and the pinned code the
    design relies on (the merge formula, the lse CBs, the even-count epilogue, the identity of the chunked K frame) is still there;
  - the order of the hand-off in the writer and the reader (data acked before VALID, credit before the write, dummy pair after the loop) and the reader's unchanged chain protocol;
  - no private names in any file (public repository), LF, ASCII outside the base reader's own characters.

Run from this directory:  python3 -B -m unittest test_ksplit_sources
"""

import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent / 'sdpa_prefill_chain'
CI = HERE.parents[2] / 'scripts' / 'ci'
for path in (HERE, CHAIN, CI):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ks_common  # noqa: E402
import ksplit_planner as planner  # noqa: E402
import make_ks_compute as mc  # noqa: E402
import make_ks_reader as mr  # noqa: E402
import make_ks_writer as mw  # noqa: E402

NL = chr(10)
FIX = HERE / 'fixtures'
READER = (HERE / mr.OUTPUT_NAME).read_text(encoding='utf-8')
WRITER = (HERE / mw.OUTPUT_NAME).read_text(encoding='utf-8')
COMPUTE = (HERE / mc.OUTPUT_NAME).read_text(encoding='utf-8')
COMPUTE_COMMON = (FIX / 'compute_common.3fb5da24.hpp').read_text(encoding='utf-8')
GXX = shutil.which('g++')


def sha(data):
    return hashlib.sha256(data).hexdigest()


class GeneratorTests(unittest.TestCase):
    def test_the_committed_kernels_are_the_generators_outputs_at_the_recorded_shas(self):
        for module, name in ((mr, mr.OUTPUT_NAME), (mw, mw.OUTPUT_NAME), (mc, mc.OUTPUT_NAME)):
            recorded = {mr: mr.READER_OUTPUT, mw: mw.WRITER_OUTPUT, mc: mc.COMPUTE_OUTPUT}[module]
            self.assertNotEqual(recorded, '__RECORD__')
            committed = (HERE / name).read_bytes()
            self.assertEqual(sha(committed), recorded, name)
            base = {mr: mr.CHAIN_READER, mw: mw.PINNED_WRITER, mc: mc.PINNED_COMPUTE}[module].read_bytes()
            self.assertEqual(module.patch(base), committed, name)
            self.assertEqual(module.unpatch(committed), base, name)

    def test_the_bases_are_the_recorded_files(self):
        self.assertEqual(sha(mr.CHAIN_READER.read_bytes()), mr.BASE_SHA)
        self.assertEqual(sha(mw.PINNED_WRITER.read_bytes()), mw.BASE_SHA)
        self.assertEqual(sha(mc.PINNED_COMPUTE.read_bytes()), mc.BASE_SHA)
        self.assertTrue(mw.PINNED_WRITER.name.startswith('writer_interleaved.' + mw.BASE_SHA[:8]))
        self.assertTrue(mc.PINNED_COMPUTE.name.startswith('sdpa.' + mc.BASE_SHA[:8]))

    def test_the_pinned_fixtures_carry_the_shas_the_repository_already_records(self):
        gate = (CI / 'proposal_native_attention_gate.py').read_text(encoding='utf-8')
        self.assertIn(sha((FIX / 'compute_common.3fb5da24.hpp').read_bytes()), gate)
        self.assertIn(sha((FIX / 'sdpa.a3f48af8.cpp').read_bytes()), gate)
        for name in sorted(FIX.iterdir()):
            digest = name.name.split('.')[-2]
            self.assertTrue(sha(name.read_bytes()).startswith(digest), name.name)

    def test_the_generators_refuse_other_inputs(self):
        for module, base in ((mr, b'x'), (mw, b'x'), (mc, b'x')):
            with self.assertRaises(module.Refusal):
                module.patch(base)

    def test_an_anchor_that_drifted_is_refused(self):
        drifted = mw.PINNED_WRITER.read_text(encoding='utf-8').replace('void kernel_main() {', NL + 'void kernel_main() {', 1).encode('utf-8')
        with self.assertRaises(mw.Refusal):
            mw.patch(drifted)
        base = mw.PINNED_WRITER.read_bytes()
        with self.assertRaises(mw.Refusal):
            mw.patch(base + b' ')                       # another sha

    def test_the_recorded_shas_in_the_card_runner_are_the_generators(self):
        text = (HERE / 'run_card_m_ks.sh').read_text(encoding='utf-8')
        for name, value in (('KS_READER_SHA', mr.READER_OUTPUT), ('KS_WRITER_SHA', mw.WRITER_OUTPUT), ('KS_COMPUTE_SHA', mc.COMPUTE_OUTPUT)):
            self.assertIn('%s=%s' % (name, value), text)
        build = (HERE / 'build_k64j_oq_ks.sh').read_text(encoding='utf-8')
        for name, value in (('KS_READER', mr.READER_OUTPUT), ('KS_WRITER', mw.WRITER_OUTPUT), ('KS_COMPUTE', mc.COMPUTE_OUTPUT)):
            self.assertIn('%s=%s' % (name, value), build)


class RangeTests(unittest.TestCase):
    def test_one_text_in_the_three_kernels(self):
        for text in (READER, WRITER, COMPUTE):
            self.assertEqual(text.count(ks_common.RANGE_BLOCK), 1)

    @unittest.skipIf(GXX is None, 'no g++')
    def test_the_cpp_gives_the_planners_numbers(self):
        program = ('#include <cstdint>\n#include <cstdio>\n#define FORCE_INLINE inline\n' + ks_common.RANGE_BLOCK
                   + 'int main() {\n  for (uint32_t blocks = 1; blocks <= 2100; ++blocks) for (uint32_t slices = 1; slices <= 4; ++slices) {\n'
                     '    const uint32_t eff = qwen_ks::effective_slices(blocks, slices);\n    std::printf("%u %u %u", blocks, slices, eff);\n'
                     '    for (uint32_t s = 0; s <= slices; ++s) std::printf(" %u", qwen_ks::slice_begin(blocks, s, eff));\n    std::printf("\\n");\n  }\n}\n')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'range.cpp'
            source.write_text(program)
            exe = Path(directory) / 'range'
            build = subprocess.run([GXX, '-std=c++17', '-O1', '-Wall', '-Werror', '-o', str(exe), str(source)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines()
        self.assertEqual(len(out), 2100 * 4)
        for line in out:
            blocks, slices, eff, *begins = (int(value) for value in line.split())
            self.assertEqual(eff, planner.effective_slices(blocks, slices))
            for s, begin in enumerate(begins):
                self.assertEqual(begin, planner.slice_begin(blocks, s, eff))
            self.assertEqual(begins[eff], blocks)


class CallTests(unittest.TestCase):
    """The compute kernel's call of the pinned ring loop against the pinned template."""

    @staticmethod
    def split_top(text, angle=True):
        parts, depth, current = [], 0, ''
        opening, closing = ('(<[{', ')>]}') if angle else ('([{', ')]}')
        for char in text:
            if char in opening:
                depth += 1
            elif char in closing:
                depth -= 1
            if char == ',' and depth == 0:
                parts.append(current.strip())
                current = ''
            else:
                current += char
        if current.strip():
            parts.append(current.strip())
        return parts

    @staticmethod
    def strip_comments(text):
        return re.sub(r'//[^\n]*', '', text)

    @classmethod
    def pinned_signature(cls):
        match = re.search(r'template <\s*SDPAType sdpa_type,(.*?)>\s*void sdpa_inner_loop\((.*?)\) \{', COMPUTE_COMMON, flags=re.S)
        template = cls.split_top('SDPAType sdpa_type,' + cls.strip_comments(match.group(1)))
        arguments = cls.split_top(cls.strip_comments(match.group(2)), angle=False)
        return [t.split('=')[0].strip().split()[-1] for t in template], [a.split('=')[0].strip().split()[-1] for a in arguments]

    @classmethod
    def our_call(cls):
        text = cls.strip_comments(COMPUTE)
        start = text.index('sdpa_inner_loop<')
        head_end = text.index('>(', start)
        template = cls.split_top(text[start + len('sdpa_inner_loop<'):head_end])
        depth, at = 1, head_end + 2
        while depth:
            depth += {'(': 1, ')': -1}.get(text[at], 0)
            at += 1
        arguments = cls.split_top(text[head_end + 2:at - 1], angle=False)
        return template, arguments

    def test_the_parameter_lists_have_the_pinned_counts(self):
        names_t, names_a = self.pinned_signature()
        template, arguments = self.our_call()
        self.assertEqual(len(template), len(names_t))
        # the trailing arguments with defaults (lw_mask ... chunked) are all passed
        self.assertEqual(len(arguments), len(names_a))
        self.assertEqual(names_a[-6:], ['lw_mask', 'is_causal', 'is_balanced', 'use_zigzag_balancing', 'is_last_ring_iter', 'chunked'])

    def test_the_semantic_positions(self):
        names_t, names_a = self.pinned_signature()
        template, arguments = self.our_call()
        by_name = dict(zip(names_a, arguments))
        expected = {'iter_q_start': 'ks_q_chunk', 'iter_q_end': 'ks_q_chunk + 1', 'q_num_chunks': 'q_num_chunks', 'iter_k_chunk_start': 'qwen_ks::slice_begin(ks_blocks, ks_slice, ks_eff)',
                    'iter_k_chunk_end': 'qwen_ks::slice_begin(ks_blocks, ks_slice + 1, ks_eff)', 'ring_iter': 'ks_slice > 0 ? 1 : 0', 'ring_id': '0',
                    'num_local_k_chunks': 'qwen_ks_ring::kNever', 'logical_nt': 'qwen_ks_ring::kNever', 'cb_lse_in': 'ks_cb_lse_in', 'cb_lse_out': 'ks_cb_lse_out',
                    'cb_prev_out': 'ks_cb_prev_out', 'cb_out': 'cb_out', 'is_causal': 'is_causal', 'is_balanced': 'false', 'use_zigzag_balancing': 'false',
                    'is_last_ring_iter': 'true', 'chunked': 'ks_chunked', 'cb_q_in': 'cb_q_in', 'cb_k_in': 'cb_k_in', 'cb_v_in': 'cb_v_in', 'cb_mask_in': 'cb_mask_in',
                    'cb_col_identity': 'cb_col_identity', 'cb_exp_max_diff': 'cb_exp_max_diff', 'q_chunk_tiles': 'q_chunk_tiles', 'out_chunk_tiles': 'out_chunk_tiles',
                    'ring_iter_needs_global_n_mask': 'false', 'ring_iter_needs_joint_n_mask': 'false', 'local_n_needs_masking': 'false'}
        for name, value in expected.items():
            self.assertEqual(by_name[name], value, name)
        self.assertEqual(dict(zip(names_t, template)), {
            'sdpa_type': 'RING', 'cb_qk_im': 'cb_qk_im', 'cb_identity_scale_in': 'cb_identity_scale_in', 'cb_attention_sink': '0', 'cb_scale_in': 'ks_cb_scale_in', 'Sq_chunk_t': 'Sq_chunk_t',
            'Sk_chunk_t': 'Sk_chunk_t', 'NH': '0', 'DHt': 'DHt', 'vDHt': 'vDHt', 'use_attention_sink': 'false', 'use_provided_mask': 'false', 'use_padded_mask': 'false',
            'use_joint_mask': 'false', 'is_chunked': 'false', 'scale_fp32': 'scale_fp32', 'sliding_window_size': '0', 'lightweight_mask_enabled': 'true', 'chunked_enabled': 'true',
            'chunked_q_local_padded_Nt': 'qwen_ks_ring::kKvIdentityTiles', 'chunked_chunk_size_t': 'qwen_ks_ring::kKvIdentityTiles'})

    def test_the_pinned_code_the_design_relies_on_is_there(self):
        for needle in ('out = prev_out - sig * (prev_out - cur_out)', 'lse = prev_lse - torch.logsigmoid(prev_lse - cur_lse)', 'cb_lse_in_obj.wait_front(Sq_chunk_t);',
                       'cb_prev_out_obj.wait_front(out_chunk_tiles);', 'sigmoid_sub(alias_cur_lse, cb_lse_in, alias_sig, Sq_chunk_t);', 'mul_block_bcast_scalar_inplace<cb_scale_in, Sq_chunk_t>(alias_prev_max);',
                       'if (KV_chunks_processed_in_iter % 2 == 0) {', 'copy_block(alias_prev_max, cb_lse_out, Sq_chunk_t);', 'if (ring_iter > 0) {',
                       'chunked.q_start_idx_t + chunked.ring_index * chunked_q_local_padded_Nt + q_chunk * Sq_chunk_t',
                       'kv_global_tile_for_local<true, 0, chunked_chunk_size_t, chunked_q_local_padded_Nt>(', 'uint32_t q_chunk = remap_q_index(q_iter, q_num_chunks, use_zigzag_balancing) % q_num_chunks;',
                       'for (uint32_t k_chunk = iter_k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {', 'k_chunk_end = iter_k_chunk_end;',
                       'if (!kv_chunk_is_joint && (kv_global_start_tile >= logical_nt)) {'):
            self.assertIn(needle, COMPUTE_COMMON, needle)
        # sdpa_ring ignores its own iter_k_chunk_start (it passes 0): the kernel calls sdpa_inner_loop directly for that reason
        ring = COMPUTE_COMMON[COMPUTE_COMMON.index('void sdpa_ring('):]
        self.assertRegex(ring, r'chunked_q_chunk_offset \(not used\)\n\s+0,\n\s+iter_k_chunk_end,')

    def test_the_chunked_k_frame_is_the_identity_below_the_constant(self):
        utils = (FIX / 'chunked_prefill_utils.9b662b10.hpp').read_text(encoding='utf-8')
        self.assertIn('return (local_tile_idx / q_local_padded_Nt) * chunk_size_t + ring_id * q_local_padded_Nt +\n           (local_tile_idx % q_local_padded_Nt);', utils)
        n = ks_common.KV_IDENTITY_TILES
        for tile in (0, 1, 4, 8000, 8192, n - 1):
            self.assertEqual((tile // n) * n + 0 * n + (tile % n), tile)
        self.assertGreater(n, 262144 // 32 + 64)            # more K tiles than the 262k pool has

    def test_the_stats_cbs_hold_the_lse_in_fp32_by_the_factory(self):
        # the lse is stored in alias_prev_max / alias_cur_max: the max CBs; the factory makes them fp32 under ksplit (PK4)
        self.assertIn('log_block(alias_prev_sum, alias_cur_max, Sq_chunk_t);', COMPUTE_COMMON)
        import apply_factory_pk as pk
        self.assertIn('stats_df = tt::DataFormat::Float32;', pk.PK4_NEW)

    def test_the_compute_kernel_returns_before_the_served_standard_loop(self):
        self.assertLess(COMPUTE.index('sdpa_inner_loop<'), COMPUTE.index('sdpa_standard<'))
        between = COMPUTE[COMPUTE.index('ks_chunked);'):COMPUTE.index('sdpa_standard<')]
        self.assertIn('return;', between)


class OrderTests(unittest.TestCase):
    def test_the_writers_handoff_order(self):
        namespace = WRITER[WRITER.index('namespace qwen_ks_xfer {'):WRITER.index('}  // namespace qwen_ks_xfer')]
        body = namespace[namespace.index('FORCE_INLINE void send_partial('):]
        order = ['cb_out.wait_front', 'cb_lse.wait_front', 'wd_wait_credit(credit_sem_id)', 'Semaphore<>(credit_sem_id).set(0)', 'noc.async_write(', 'noc.async_write(',
                 'noc.async_write_barrier()', 'relay_unicast', 'noc.async_writes_flushed()', 'cb_out.pop_front', 'cb_lse.pop_front']
        at = 0
        for needle in order:
            at = body.index(needle, at) + 1
        self.assertEqual(body.count('async_write('), 2)
        self.assertIn('WAYPOINT("QKDC")', namespace)
        self.assertIn('ASSERT(false)', namespace)

    def test_the_writer_skips_the_dram_write_unless_it_is_the_last_slice(self):
        unit = WRITER[WRITER.index('const uint32_t ks_q_high'):WRITER.index('}  // close phase')]
        self.assertLess(unit.index('if (ks_slice >= ks_eff)'), unit.index('send_partial'))
        self.assertLess(unit.index('send_partial'), unit.index('write_block('))
        self.assertEqual(unit.count('continue;'), 2)
        self.assertLess(unit.index('write_block('), unit.index('ks_cb_lse_drop.wait_front'))

    def test_the_readers_receive_order(self):
        unit = READER[READER.index('const uint32_t ks_blocks'):READER.index('}  // close global_q_iter')]
        order = ['effective_slices', 'ks_cb_prev.reserve_back', 'ks_cb_lse.reserve_back', 'Semaphore<>(ks_credit_sem).up(', 'noc.async_atomic_barrier()', 'for (uint32_t k_chunk = k_loop_start; k_chunk < k_loop_end',
                 '}  // close k_chunk', 'qwen_pf::wd_wait_valid(ks_valid_sem)', 'ks_cb_prev.push_back', 'ks_cb_lse.push_back', 'Semaphore<>(ks_valid_sem).set(INVALID)',
                 '% 2) == 0', 'cb_k.reserve_back(k_chunk_tiles);', 'cb_k.push_back(k_chunk_tiles);', 'cb_v.reserve_back(v_chunk_tiles);', 'cb_v.push_back(v_chunk_tiles);']
        at = 0
        for needle in order:
            at = unit.index(needle, at) + 1

    def test_the_chain_protocol_of_the_reader_is_untouched(self):
        base = mr.CHAIN_READER.read_text(encoding='utf-8')
        for needle in ('qwen_pf::credit_prev(', 'qwen_pf::wd_wait_valid(receiver_semaphore_id);', 'qwen_pf::forward_round(', 'Semaphore<>(valid_semaphore_id).set(VALID);',
                       'Semaphore<>(receiver_semaphore_id).set(INVALID);', 'noc.async_write_barrier();', 'qwen_pf::wd_wait_credit(', 'read_paged_chunk_with_padding<'):
            self.assertEqual(READER.count(needle), base.count(needle), needle)
        self.assertEqual(READER.count('noc.async_atomic_barrier();'), base.count('noc.async_atomic_barrier();') + 1)   # the hand-off credit's own, in K4

    def test_the_hang_flag_withholds_only_the_credit(self):
        self.assertIn('if (!ks_test_hang) {', READER)
        self.assertIn('(qwen_pf_flags & 0x400u) != 0', READER)

    def test_the_idle_slice_does_nothing(self):
        # an idle slice has an empty loop and no handshake: k_loop_end = k_loop_start, receiver false
        self.assertIn('const bool ks_receiver = ks_active && ks_slice > 0;', READER)
        self.assertIn('const uint32_t k_loop_end = ks_active ? qwen_ks::slice_begin(ks_blocks, ks_slice + 1, ks_eff) : k_loop_start;', READER)
        self.assertIn('if (ks_slice >= ks_eff) {' + NL + '                continue;', WRITER)
        self.assertIn('if (ks_slice >= ks_eff) {' + NL + '            return;', COMPUTE)


class TextTests(unittest.TestCase):
    FILES = ('reader_interleaved_qwen_ksplit.cpp', 'writer_interleaved_qwen_ksplit.cpp', 'sdpa_qwen_ksplit.cpp', 'ks_common.py', 'make_ks_reader.py', 'make_ks_writer.py',
             'make_ks_compute.py', 'apply_factory_pk.py', 'ksplit_planner.py', 'ks_protocol_model.py', 'ksplit_card_m.py', 'host_exec_ksplit.py', 'stub_compile_ksplit.py',
             'run_card_m_ks.sh', 'build_k64j_oq_ks.sh', 'README.md')

    def test_lf_and_nothing_private(self):
        private = re.compile(r'\d{1,3}(\.\d{1,3}){3}|thatch|blackhole-[0-9A-F]{8}|/home/|\bssh\b|ghp_|sha256:[0-9a-f]{16}|\.local\b', re.I)
        for name in self.FILES:
            path = HERE / name
            if not path.exists():
                continue
            raw = path.read_bytes()
            self.assertNotIn(b'\r', raw, name)
            text = raw.decode('utf-8')
            for number, line in enumerate(text.splitlines(), 1):
                match = private.search(line)
                self.assertIsNone(match, '%s:%d: %s' % (name, number, line.strip()[:160]))
            self.assertIsNone(re.search('[\U0001F300-\U0001FAFF\u2600-\u27BF]', text), name)

    def test_non_ascii_is_only_the_chain_readers_dash(self):
        for name in ('reader_interleaved_qwen_ksplit.cpp', 'writer_interleaved_qwen_ksplit.cpp', 'sdpa_qwen_ksplit.cpp'):
            text = (HERE / name).read_text(encoding='utf-8')
            base = {'reader_interleaved_qwen_ksplit.cpp': mr.CHAIN_READER, 'writer_interleaved_qwen_ksplit.cpp': mw.PINNED_WRITER, 'sdpa_qwen_ksplit.cpp': mc.PINNED_COMPUTE}[name]
            allowed = {char for char in base.read_text(encoding='utf-8') if ord(char) > 127}
            self.assertLessEqual({char for char in text if ord(char) > 127}, allowed, name)
        for name in ('make_ks_reader.py', 'make_ks_writer.py', 'make_ks_compute.py', 'ks_common.py', 'ksplit_planner.py', 'ks_protocol_model.py', 'ksplit_card_m.py', 'apply_factory_pk.py'):
            text = (HERE / name).read_text(encoding='utf-8')
            self.assertEqual({char for char in text if ord(char) > 127}, set(), name)


if __name__ == '__main__':
    unittest.main()
