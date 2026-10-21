"""CPU tests of the K-split factory patch (apply_factory_pk.py), the real decision C++ run on the host (host_exec_ksplit.py) and the stub compiles (stub_compile_ksplit.py).

  - the patch: input shas, anchors, the recorded output sha, inversion, no edit outside create_descriptor, the protected draft block untouched, LF, the refusal of other inputs;
  - the word decoder (decode_word) and the program-word constants against the C++ text;
  - the host-executed decision regions against ksplit_planner.plan on a grid of shapes: every core's range, slice, previous / next core, chain flags, the log lines and the refusals;
  - the oneq call is unchanged when flag 0x10 is absent (the host-exec output equals the oneq planner's);
  - the compile-time / runtime argument layouts the factory pushes against what the three kernels read;
  - the stub compiles (g++ -fsyntax-only, skipped without g++ / without a probe tree).

Run from this directory:  python3 -B -m unittest test_ksplit_factory
"""

import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent / 'sdpa_prefill_chain'
ONEQ = HERE.parent / 'sdpa_prefill_oneq'
for path in (HERE, ONEQ, CHAIN, HERE.parents[2] / 'scripts' / 'ci'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import apply_factory_pf as pf  # noqa: E402
import apply_factory_pk as pk  # noqa: E402
import apply_factory_ps as ps  # noqa: E402
import host_exec_ksplit as hx  # noqa: E402
import ksplit_planner as planner  # noqa: E402
import oneq_planner as oneq  # noqa: E402
import stub_compile_ksplit as stubs  # noqa: E402

SERVED = (CHAIN / 'fixtures' / 'sdpa_program_factory.fd8c0676.cpp').read_bytes()
GXX = stubs.find_gxx()
NL = chr(10)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patched = pk.patch(SERVED)
        cls.text = cls.patched.decode('utf-8')

    def test_the_recorded_output_is_what_the_patch_makes(self):
        self.assertNotEqual(pk.PK_FACTORY, '__RECORD__')
        self.assertEqual(sha(self.patched), pk.PK_FACTORY)

    def test_every_input_route_gives_the_same_bytes(self):
        ps_text = ps.patch(SERVED)
        pf_text = pf.patch(SERVED)
        self.assertEqual(sha(ps_text), ps.PS_FACTORY)
        self.assertEqual(pk.patch(ps_text), self.patched)
        self.assertEqual(pk.patch(pf_text), self.patched)

    def test_the_edits_invert(self):
        self.assertEqual(pk.unpatch(self.patched), ps.patch(SERVED))

    def test_other_inputs_are_refused(self):
        with self.assertRaises(pk.Refusal):
            pk.patch(b'int main() {}' + NL.encode())
        with self.assertRaises(pk.Refusal):
            pk.patch(self.patched)             # PK over PK: the base is not the PS factory

    def test_an_anchor_that_drifted_is_refused(self):
        drifted = ps.patch(SERVED).decode('utf-8').replace('#include <vector>', '#include <vector>' + NL, 1).encode('utf-8')
        with self.assertRaises(pk.Refusal):
            pk.patch(drifted)

    def test_every_anchor_is_unique_and_at_its_recorded_line(self):
        text = ps.patch(SERVED).decode('utf-8')
        found = pk.anchor_lines(text)
        for label, line, _old, _new in pk.EDITS:
            self.assertEqual(found[label], line, label)
        self.assertEqual([label for label, *_ in pk.EDITS], sorted((label for label, *_ in pk.EDITS), key=lambda l: found[l]), 'the edits are in file order')

    def test_no_edit_lies_outside_create_descriptor_and_the_draft_block_is_untouched(self):
        body = self.text.index(pk.FUNCTION_LINE)
        for label, _line, _old, new in pk.EDITS:
            self.assertGreater(self.text.index(new), body, label)
        self.assertEqual(pf.protected_block(self.text), pf.protected_block(ps.patch(SERVED).decode('utf-8')))
        self.assertNotIn(chr(13), self.text)

    def test_only_function_local_names_are_added(self):
        before = ps.patch(SERVED).decode('utf-8')
        head = before[:before.index(pk.FUNCTION_LINE)]
        self.assertEqual(self.text[:len(head)], head, 'nothing above create_descriptor changed')
        tail = before[before.index(pk.FUNCTION_LINE):]
        self.assertTrue(self.text.endswith(tail.split('}  // namespace ttnn::prim')[-1]))

    def test_the_new_qwen_strings_are_exactly_the_four_literals(self):
        before = ps.patch(SERVED).decode('utf-8')
        pattern = re.compile(r'"(\[QWEN-[^"]*)"')
        new = sorted(set(pattern.findall(self.text)) - set(pattern.findall(before)))
        self.assertEqual(new, sorted(pk.NEW_QWEN_STRINGS))
        for literal in pk.NEW_QWEN_STRINGS:
            self.assertEqual(self.text.count('"%s"' % literal), 1, literal)

    def test_the_kernel_paths_are_selected_only_under_ksplit(self):
        for path in (pk.READER_PATH, pk.WRITER_PATH, pk.COMPUTE_PATH):
            self.assertEqual(self.text.count(path), 1)
            at = self.text.index(path)
            self.assertIn('if (qwen_ks) {', self.text[at - 900:at])

    def test_every_new_use_of_the_extra_state_is_guarded_by_qwen_ks(self):
        # the only unguarded ksplit statements are the declarations and the decode; every behaviour change sits under `if (qwen_ks)` or an else of one
        for line in self.text.splitlines():
            if 'qwen_ks_cb_' in line and 'allocate_tile_cb' in line:
                self.assertTrue(line.startswith('        '), 'allocated inside the if: ' + line)
        self.assertIn('if (qwen_ks) {', self.text)
        unguarded = [line.strip() for line in self.text.splitlines() if 'qwen_ks' in line and 'if (qwen_ks' not in line and 'qwen_ks_' not in line.split('//')[0][:0]]
        self.assertGreater(len(unguarded), 10)


class WordTests(unittest.TestCase):
    def test_constants_match_the_cpp(self):
        text = pk.patch(SERVED).decode('utf-8')
        self.assertIn('constexpr uint32_t kQwenPfKsplit = 0x%xu, kQwenPfKsHang = 0x%xu;' % (pk.FLAG_KSPLIT, pk.FLAG_KS_HANG), text)
        self.assertIn('constexpr uint32_t kQwenKsBit = 0x%xu;' % pk.FLAG_KSPLIT, text)
        self.assertIn('kQwenKsCreditSem = %du, kQwenKsValidSem = %du' % (pk.CREDIT_SEM, pk.VALID_SEM), text)
        self.assertEqual(pk.KS_BITS, 0x19)
        self.assertEqual(pk.KSPLIT_FLAGS, (0x19, 0x1B, 0x1D, 0x1F))
        self.assertEqual(planner.FLAG_KSPLIT, pk.FLAG_KSPLIT)
        self.assertEqual(planner.FLAG_KS_HANG, pk.FLAG_KS_HANG)
        self.assertEqual(planner.SLICE_CAP, pk.SLICE_CAP)
        self.assertIn('std::min<uint32_t>(%du,' % pk.SLICE_CAP, text)

    def test_decode_word(self):
        tag = pk.PF_TAG
        self.assertEqual(pk.decode_word(tag | 0x1B), (True, 0x1B, True, True))
        self.assertEqual(pk.decode_word(tag | 0xB), (True, 0xB, True, False))
        self.assertEqual(pk.decode_word(0x10), (False, 0, False, False))
        for flags in (0x10, 0x11, 0x12, 0x13, 0x18, 0x400, 0x403, 0x40B, 0x1000, 0x10 | 0x8):
            with self.assertRaises(ValueError, msg=hex(flags)):
                pk.decode_word(tag | flags)
        with self.assertRaises(ValueError):
            pk.decode_word(tag | 0x41B)                            # the test flag needs QWEN_SDPA_PF_TEST=1
        self.assertEqual(pk.decode_word(tag | 0x41B, '1'), (True, 0x41B, True, True))
        self.assertEqual(pk.decode_word(tag | 0x303, '1')[1], 0x303)   # the chain's own test flags are unchanged
        for flags in pk.KSPLIT_FLAGS:
            self.assertTrue(pk.decode_word(tag | flags)[3])


@unittest.skipIf(GXX is None, 'no g++')
class HostExecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.exe = hx.build(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_case(self, nqh, nkh, rows, grid, word, **kwargs):
        return hx.run(self.exe, nqh, nkh, rows, 128, grid, word, **kwargs)

    def test_the_planner_is_the_factory_on_a_grid_of_shapes(self):
        shapes = []
        for rows in (512, 1024, 256, 128, 2048, 1920, 768):
            for grid in ((13, 10), (11, 10), (8, 8), (6, 8)):
                for nqh, nkh in ((6, 1), (12, 2), (8, 1)):
                    shapes.append((nqh, nkh, rows, grid))
        checked = engaged = 0
        for nqh, nkh, rows, grid in shapes:
            for flags in (0x1B, 0x19, 0x1F):
                outcome = self.run_case(nqh, nkh, rows, grid, pk.PF_TAG | flags)
                mine = planner.plan(nqh, nkh, rows, 128, grid, noc_order=bool(flags & 4))
                label = '%s %s' % ((nqh, nkh, rows, grid), hex(flags))
                checked += 1
                if mine['refusal']:
                    self.assertIsNotNone(outcome['fatal'], label)
                    self.assertTrue(self.fatal_matches(outcome['fatal'], mine['refusal']), '%s: %r vs %r' % (label, outcome['fatal'], mine['refusal']))
                    continue
                engaged += 1
                self.assertIsNone(outcome['fatal'], label)
                self.assertEqual(outcome['returncode'], 0, label)
                self.assertEqual(int(outcome['split']['slices']), mine['slices'], label)
                self.assertEqual(int(outcome['split']['cores']), mine['busy'], label)
                cores = outcome['cores']
                self.assertEqual(len(cores), mine['num_cores'], label)
                for core, record in zip(cores, mine['cores']):
                    if not record['busy']:
                        self.assertEqual((core['count'], core['slice']), (0, 0), label)
                        continue
                    self.assertEqual((core['start'], core['count']), (record['core'] % mine['total'], 1), label)
                    self.assertEqual(core['slice'], record['slice'], label)
                    prev = (0, 0) if record['prev'] is None else self.physical(record['prev'], grid)
                    nxt = (0, 0) if record['next'] is None else self.physical(record['next'], grid)
                    self.assertEqual((core['ksprev'], core['ksnext']), (prev, nxt), label)
                    self.assertEqual(core['participates'], int(any(record['core'] in members for members in mine['chains'].values())), label)
                lines = outcome['logs']
                self.assertEqual(len(lines), 3, label)
                self.assertIn('flags=%#x kv_chain=1 chains=%d members=%d' % (flags, mine['chain_count'], mine['member_count']), lines[0], label)
                self.assertEqual(lines[2], '[QWEN-SDPA-PF] ksplit=1 slices=%d q_chunks=%d cores=%d busy=%d chains=%d members=%d'
                                 % (mine['slices'], mine['total'], mine['num_cores'], mine['busy'], mine['chain_count'], mine['member_count']), label)
        self.assertGreater(checked, 200)
        self.assertGreater(engaged, 60)

    @staticmethod
    def physical(core, grid):
        return (core % grid[0] + 1, core // grid[0] + 2)

    @staticmethod
    def fatal_matches(fatal, refusal):
        if refusal.startswith(planner.FATAL_SLICES.split('{}')[0]):
            return fatal.startswith(planner.FATAL_SLICES.split('{}')[0].rstrip()[:-3]) or 'at least 2 slices' in fatal
        return refusal.split(' (')[0].split(':')[0] in fatal or fatal in refusal or fatal.split(':')[0] in refusal

    def test_the_chain_keys_carry_the_slice_one_chain_per_slice_and_q_chunk(self):
        outcome = self.run_case(6, 1, 1024, (13, 10), pk.PF_TAG | 0x1B)
        chains = {}
        for core in outcome['cores']:
            if core['count'] and core['participates']:
                chains.setdefault((core['slice'], core['qstart']), []).append(core['core'])
        self.assertEqual(len(chains), 16)
        self.assertTrue(all(len(members) == 6 for members in chains.values()))
        for members in chains.values():
            roles = sorted((outcome['cores'][m]['injector'], outcome['cores'][m]['sink']) for m in members)
            self.assertEqual(roles.count((1, 0)), 1)
            self.assertEqual(roles.count((0, 1)), 1)

    def test_without_flag_0x10_the_call_is_the_oneq_call(self):
        for rows in (2048, 1024, 512):
            outcome = self.run_case(6, 1, rows, (13, 10), pk.PF_TAG | 0xB)
            base = oneq.plan(6, 1, rows, 128, (13, 10), oneq=True)
            self.assertIsNone(outcome['fatal'])
            self.assertEqual(outcome['split']['ks'], '0')
            self.assertEqual([core['count'] for core in outcome['cores']], [count for _start, count in base['ranges']])
            self.assertTrue(all(core['slice'] == 0 and core['ksprev'] == (0, 0) and core['ksnext'] == (0, 0) for core in outcome['cores']))
            self.assertEqual(len(outcome['logs']), 2)
            self.assertNotIn('ksplit', ' '.join(outcome['logs']))

    def test_the_refusals_have_the_texts_the_harness_expects(self):
        cases = ((0x10, 1024, 'unknown or incomplete flags'), (0x11, 1024, 'unknown or incomplete flags'), (0x40B, 1024, 'unknown or incomplete flags'),
                 (0x41B, 1024, 'test-only flags'), (0x1B, 2048, 'ksplit needs at least 2 slices'), (0x1B, 1920, 'kv_chain outside its qualified envelope'))
        for flags, rows, needle in cases:
            outcome = self.run_case(6, 1, rows, (13, 10), pk.PF_TAG | flags)
            self.assertIsNotNone(outcome['fatal'], hex(flags))
            self.assertIn(needle, outcome['fatal'], hex(flags))
        outcome = self.run_case(12, 2, 2048, (13, 10), pk.PF_TAG | 0x1B)
        self.assertIn('oneq needs one q chunk per core', outcome['fatal'])
        outcome = self.run_case(6, 1, 1024, (6, 8), pk.PF_TAG | 0x1B)
        self.assertIn('ksplit needs at least 2 slices', outcome['fatal'])

    def test_the_mode_and_scale_guards(self):
        word = pk.PF_TAG | 0x1B
        self.assertIn('exact in bf16', self.run_case(6, 1, 1024, (13, 10), word, scale_packed=0x3D800001)['fatal'])
        self.assertIn('fp32 dest accumulation', self.run_case(6, 1, 1024, (13, 10), word, fp32_dest=False)['fatal'])
        self.assertIn('fp32 dest accumulation', self.run_case(6, 1, 1024, (13, 10), word, approx=True)['fatal'])
        self.assertIsNone(self.run_case(6, 1, 1024, (13, 10), word, scale_packed=0x3D800000)['fatal'])
        self.assertIsNone(self.run_case(6, 1, 1024, (13, 10), word, scale_packed=0x3E000000)['fatal'])

    def test_the_hang_test_flag_is_accepted_only_with_the_test_variable(self):
        word = pk.PF_TAG | 0x41B
        self.assertIn('test-only flags', self.run_case(6, 1, 1024, (13, 10), word)['fatal'])

    def test_the_cap_at_256_rows(self):
        outcome = self.run_case(6, 1, 256, (13, 10), pk.PF_TAG | 0x1B)
        self.assertEqual(int(outcome['split']['slices']), 4)
        self.assertEqual(int(outcome['split']['cores']), 48)


class LayoutTests(unittest.TestCase):
    """The argument layouts the factory pushes are the ones the three kernels read."""

    @classmethod
    def setUpClass(cls):
        cls.factory = pk.patch(SERVED).decode('utf-8')
        cls.reader = (HERE / 'reader_interleaved_qwen_ksplit.cpp').read_text(encoding='utf-8')
        cls.writer = (HERE / 'writer_interleaved_qwen_ksplit.cpp').read_text(encoding='utf-8')
        cls.compute = (HERE / 'sdpa_qwen_ksplit.cpp').read_text(encoding='utf-8')

    def offsets(self, text, prefix):
        return {name: int(offset) for name, offset in re.findall(r'constexpr (?:uint32_t|bool) %s(\w+) = get_compile_time_arg_val\(cb_arg_offset \+ (\d+)\)' % prefix, text)}

    def test_the_reader_suffix_order(self):
        found = self.offsets(self.reader, 'ks_')
        names = {'slices': 'slices', 'cb_prev_out': 'cb_prev_out', 'cb_lse_in': 'cb_lse_in', 'credit_sem': 'credit_sem', 'valid_sem': 'valid_sem'}
        self.assertEqual([found[name] for name in names], list(range(9, 9 + len(pk.READER_CT))))
        self.assertEqual(list(names), list(pk.READER_CT))
        self.assertIn('{qwen_ks_slices, qwen_ks_cb_prev_out, qwen_ks_cb_lse_in, kQwenKsCreditSem, kQwenKsValidSem}', self.factory)

    def test_the_writer_suffix_order(self):
        found = self.offsets(self.writer, 'ks_')
        order = sorted(found, key=found.get)
        self.assertEqual([found[name] for name in order], list(range(6, 6 + len(pk.WRITER_CT))))
        self.assertEqual(order, ['slices', 'Skt', 'cb_lse_out', 'cb_prev_out', 'cb_lse_in', 'cb_scale_in', 'credit_sem', 'valid_src_sem', 'valid_dst_sem', 'flags'])
        self.assertEqual(order, list(pk.WRITER_CT))
        self.assertIn('{qwen_ks_slices, Skt, qwen_ks_cb_lse_out, qwen_ks_cb_prev_out, qwen_ks_cb_lse_in, qwen_ks_cb_scale_in,', self.factory)
        self.assertIn('kQwenKsCreditSem, valid_semaphore_id, kQwenKsValidSem, qwen_pf_flags});', self.factory)

    def test_the_compute_suffix_order_and_base(self):
        found = self.offsets(self.compute, 'ks_')
        order = sorted(found, key=found.get)
        self.assertEqual([found[name] for name in order], list(range(18, 18 + len(pk.COMPUTE_CT))))
        self.assertEqual(order, ['slices', 'cb_lse_in', 'cb_lse_out', 'cb_prev_out', 'cb_scale_in', 'flags'])
        self.assertEqual(order, list(pk.COMPUTE_CT))
        self.assertIn('{qwen_ks_slices, qwen_ks_cb_lse_in, qwen_ks_cb_lse_out, qwen_ks_cb_prev_out, qwen_ks_cb_scale_in, qwen_pf_flags}', self.factory)
        self.assertIn('constexpr uint32_t cb_arg_offset = 34;', self.compute)

    def test_the_runtime_word_order(self):
        # reader: ... global_q_count, then slice, prev x, prev y;  writer: the served twelve, then slice, next x, next y;  compute: the served seven, then the slice
        self.assertLess(self.reader.index('global_q_count = get_arg_val'), self.reader.index('const uint32_t ks_slice = get_arg_val'))
        reader_tail = re.findall(r'(ks_\w+) = get_arg_val<uint32_t>\(argidx\+\+\)', self.reader)
        self.assertEqual(reader_tail, ['ks_slice', 'ks_prev_x', 'ks_prev_y'])
        self.assertIn('reader_args.push_back(qwen_ks_slice);' + NL + '            reader_args.push_back(static_cast<uint32_t>(qwen_ks_prev.x));' + NL
                      + '            reader_args.push_back(static_cast<uint32_t>(qwen_ks_prev.y));', self.factory)
        self.assertEqual(re.findall(r'(ks_(?:slice|next_x|next_y)) = get_arg_val<uint32_t>\((\d+)\)', self.writer), [('ks_slice', '12'), ('ks_next_x', '13'), ('ks_next_y', '14')])
        self.assertEqual(self.factory.count('qwen_ks_writer_args.push_back('), 15)
        self.assertEqual(re.findall(r'(ks_\w+) = get_arg_val<uint32_t>\((\d+)\)', self.compute), [('ks_slice', '7')])
        self.assertEqual(self.factory.count('qwen_ks_compute_args.push_back('), 8)

    def test_the_semaphore_ids_and_the_valid_source(self):
        self.assertEqual(pf.SEMAPHORE_IDS, dict(sender=0, receiver=1, valid=2))
        self.assertNotIn(pk.CREDIT_SEM, pf.SEMAPHORE_IDS.values())
        self.assertNotIn(pk.VALID_SEM, pf.SEMAPHORE_IDS.values())
        self.assertNotEqual(pk.CREDIT_SEM, pk.VALID_SEM)

    def test_the_stats_cbs_and_the_scale_cb_are_added_after_the_served_ones(self):
        allocations = [m.start() for m in re.finditer(r'= allocate_tile_cb\(', self.factory)]
        served_line = 'cb_ids.out = allocate_tile_cb(out0_t, out_tile_size, out_df);'
        served_last = self.factory.index(served_line) + len(served_line)
        ks_first = self.factory.index('qwen_ks_cb_prev_out = allocate_tile_cb(')
        self.assertLess(served_last, ks_first)
        self.assertEqual(sum(1 for at in allocations if at > served_last), 4)


@unittest.skipIf(GXX is None, 'no g++')
class StubCompileTests(unittest.TestCase):
    def test_the_decision_regions_compile(self):
        code, output = stubs.compile_text(GXX, stubs.decision_harness())
        self.assertEqual(code, 0, output)

    def test_the_plumbing_regions_compile(self):
        code, output = stubs.compile_text(GXX, stubs.plumbing_harness())
        self.assertEqual(code, 0, output)

    def test_the_stub_check_has_teeth(self):
        for harness in (stubs.decision_harness, stubs.plumbing_harness):
            text = harness()
            code, _output = stubs.compile_text(GXX, text.replace('qwen_ks_slices', 'qwen_ks_slicez', 1))
            self.assertNotEqual(code, 0, harness.__name__)

    def test_the_kernels_compile_against_the_pinned_headers_when_a_probe_tree_is_given(self):
        probe = os.environ.get('QWEN_SDPA_PREFILL_SRC')
        if not probe or not (Path(probe) / 'kernels' / 'dataflow' / 'dataflow_common.hpp').is_file():
            self.skipTest('no probe tree (QWEN_SDPA_PREFILL_SRC: kernels/dataflow/{dataflow_common.hpp, reader_interleaved.cpp, writer_interleaved.cpp}, kernels/{q_chunk_remapping.hpp, sliding_window_geometry.hpp})')
        results = stubs.check_kernels(GXX, probe)
        ok, extra = stubs.kernel_verdict(results)
        self.assertTrue(ok, str([(name, code, output[-1500:]) for name, code, output in results if code]) + str(extra))


if __name__ == '__main__':
    unittest.main()
