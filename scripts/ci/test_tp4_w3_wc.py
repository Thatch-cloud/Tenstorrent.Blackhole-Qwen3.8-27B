"""WC (gdn_wc, QWEN_FAST_TP4_GDN_WC): the lever's wiring, its kernels' text, the launch on a fake ttnn, the audit and the smoke rule.

test_gdn_wc holds the plan and the byte-level proof (the kernels' loops emulated against the V1 chain and the served per-user calls); this module holds
everything around it:

  * the flag grammar (strict, refused at the pair, the prerequisites, the audit rules) and that an unset flag leaves the V2 twin the pinned class call for call
    without importing the module;
  * the kernels' text: LF only, the compile-word and runtime-word indices they read against the module's tables, the circular-buffer indices against the plan,
    the transcribed window formulas, and that they call nothing the production kernels do not (no new API surface);
  * the launch on a fake ttnn: one generic_op, a program per chip with three kernels, the cores and their words, the common words, the circular buffers, the
    served compute source untouched, every fall-back (one logged line, nothing allocated, nothing advanced), the pinned-source refusal;
  * the stage and the binding: run_user_batched_projected with `prebuilt` builds neither the T2 windows nor the block stage, the V2 twin skips the split launch,
    asks WC after the state moves, and takes the V1 path (making the real pieces) when WC declines;
  * the audit (entries labelled, an audited layer without them fails, a difference raises) and the smoke rule;
  * the image copy lists and the CPU allowlist.

What no CPU test can show is that the compiled kernels compile and move those bytes: that is the card-M byte gate and the audit on a card.

Run at py 3.11 from scripts/ci: `python -B -m unittest test_tp4_w3_wc`.
"""

import ast
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gdn_block_conv_tp as block
import gdn_conv_gates_spread as spread
import gdn_wc
import tp4_vglue
import tp_shapes
import w3_wc_smoke
from test_gdn_conv_gates_spread import FakeTTNN, Kernel, Tensor, fake_mesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FOUR = {'QWEN_FAST_TP': '4'}
PARENT_COMMIT = '1cfc8412'     # tp4/er-drafters, the base of this branch
T2 = {'QWEN_FAST_VERIFY_T2': '1'}
LEVER = dict(FOUR, **T2, **{gdn_wc.FLAG: '1', 'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_TP4_GDN_BLOCK_CONV': '1'})
AUDITED = dict(LEVER, **{gdn_wc.AUDIT_FLAG: '1', 'QWEN_FAST_TP4_VGLUE_AUDIT': '1'})


def text(name):
    return (HERE / name).read_text(encoding='utf-8')


def plan4(users=4):
    """The core plan at four cards (the geometry is the process's width: set it, the plan would otherwise read the pair's)."""
    with patch.dict(os.environ, FOUR):
        return gdn_wc.plan(11, 10, users)


def env(**values):
    return patch.dict(os.environ, values)


def clean(**values):
    """A process environment holding only `values` (the runner's own variables cannot change a flag)."""
    return patch.dict(os.environ, values, clear=True)


# --- the flags --------------------------------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_unset_and_zero_are_off(self):
        with clean(**FOUR):
            self.assertFalse(gdn_wc.enabled())
        with clean(**FOUR, **{gdn_wc.FLAG: '0'}):
            self.assertFalse(gdn_wc.enabled())
            self.assertFalse(gdn_wc.audit_enabled())

    def test_one_is_on_with_four_cards_the_glue_the_block_conv_and_the_t2_windows(self):
        with clean(**LEVER):
            self.assertTrue(gdn_wc.enabled())

    def test_anything_else_raises(self):
        for value in ('2', 'true', '', ' 1'):
            with clean(**dict(LEVER, **{gdn_wc.FLAG: value})), self.assertRaises(ValueError):
                gdn_wc.enabled()

    def test_the_pair_refuses(self):
        with clean(**{gdn_wc.FLAG: '1'}), self.assertRaisesRegex(ValueError, 'TP4 lever'):
            gdn_wc.enabled()

    def test_it_needs_the_glue_and_the_block_conv(self):
        for missing in ('QWEN_FAST_TP4_GDN_GLUE', 'QWEN_FAST_TP4_GDN_BLOCK_CONV'):
            flags = dict(LEVER)
            del flags[missing]
            with clean(**flags), self.assertRaises(ValueError):
                gdn_wc.enabled()

    def test_it_needs_the_t2_windows_cut(self):
        without = {key: value for key, value in LEVER.items() if key != 'QWEN_FAST_VERIFY_T2'}
        with clean(**without), self.assertRaisesRegex(ValueError, 'windows cut'):
            gdn_wc.enabled()
        with clean(**dict(LEVER, QWEN_FAST_VERIFY_T2_SKIP='windows')), self.assertRaisesRegex(ValueError, 'windows cut'):
            gdn_wc.enabled()
        with clean(**dict(LEVER, QWEN_FAST_VERIFY_T2_SKIP='kv_chains')):
            self.assertTrue(gdn_wc.enabled())

    def test_the_t2_literals_are_verify_trace_t2s_and_agree_with_its_cut(self):
        import verify_trace_t2

        self.assertEqual((gdn_wc.T2_FLAG, gdn_wc.T2_SKIP_FLAG, gdn_wc.T2_WINDOWS_CUT),
                         (verify_trace_t2.FLAG, verify_trace_t2.SKIP_FLAG, verify_trace_t2.CUTS[0]))
        for flags in ({}, T2, dict(T2, QWEN_FAST_VERIFY_T2_SKIP='windows'), dict(T2, QWEN_FAST_VERIFY_T2_SKIP='kv_chains'),
                      dict(T2, QWEN_FAST_VERIFY_T2_SKIP='kv_chains, windows')):
            with self.subTest(flags=flags), clean(**flags):
                self.assertEqual(gdn_wc.t2_windows_cut(), verify_trace_t2.cut('windows'))

    def test_the_audit_needs_the_lever_and_the_vglue_audit(self):
        with clean(**dict(FOUR, **{gdn_wc.AUDIT_FLAG: '1'})), self.assertRaisesRegex(ValueError, 'audit compares'):
            gdn_wc.enabled()
        with clean(**dict(LEVER, **{gdn_wc.AUDIT_FLAG: '1'})), self.assertRaisesRegex(ValueError, 'VGLUE_AUDIT'):
            gdn_wc.audit_enabled()
        with clean(**AUDITED):
            self.assertTrue(gdn_wc.audit_enabled())

    def test_the_vglue_audit_needs_the_wc_audit_beside_the_lever(self):
        with clean(**dict(LEVER, QWEN_FAST_TP4_VGLUE_AUDIT='1')), self.assertRaisesRegex(ValueError, 'no V1 block entries'):
            gdn_wc.enabled()

    def test_f1_may_be_set_beside_it_but_not_its_audit(self):
        with clean(**dict(LEVER, **{spread.FLAG: '1'})):
            self.assertTrue(gdn_wc.enabled())
        with clean(**dict(AUDITED, **{spread.FLAG: '1'})):
            self.assertTrue(gdn_wc.audit_enabled())
        with clean(**dict(AUDITED, **{spread.FLAG: '1', spread.AUDIT_FLAG: '1'})), self.assertRaisesRegex(ValueError, 'fall-back launch'):
            gdn_wc.enabled()

    def test_the_literals_in_the_twin_and_the_smoke_rule_are_the_modules(self):
        import gdn_device_loop_state_tp as twin

        self.assertEqual((twin.WC_FLAG, twin.WC_AUDIT_FLAG), (gdn_wc.FLAG, gdn_wc.AUDIT_FLAG))
        self.assertEqual((w3_wc_smoke.FLAG, w3_wc_smoke.AUDIT_FLAG), (gdn_wc.FLAG, gdn_wc.AUDIT_FLAG))
        self.assertEqual((w3_wc_smoke.ENGAGED, w3_wc_smoke.FELL_BACK, w3_wc_smoke.AUDIT, w3_wc_smoke.MISMATCH),
                         (gdn_wc.ENGAGED, gdn_wc.FELL_BACK, gdn_wc.AUDIT_MARKER, gdn_wc.AUDIT_MISMATCH))

    def test_requested_is_false_and_imports_nothing_when_unset(self):
        import gdn_device_loop_state_tp as twin

        sys.modules.pop('gdn_wc', None)
        try:
            with clean(**FOUR):
                self.assertFalse(twin.wc_requested())
            self.assertNotIn('gdn_wc', sys.modules)
        finally:
            sys.modules['gdn_wc'] = gdn_wc

    def test_requested_validates_a_set_flag_strictly(self):
        import gdn_device_loop_state_tp as twin

        with clean(**LEVER):
            self.assertTrue(twin.wc_requested())
        with clean(**dict(LEVER, **{gdn_wc.FLAG: '2'})), self.assertRaises(ValueError):
            twin.wc_requested()
        with clean(**dict(FOUR, **{gdn_wc.AUDIT_FLAG: '1'})), self.assertRaises(ValueError):
            twin.wc_requested()


# --- the kernels' text --------------------------------------------------------------------------------------------------------------

def constants(source):
    """{name: value} of the `constexpr uint32_t NAME = expr;` lines (comma lists included), evaluated in order."""
    found = {}
    for statement in re.findall(r'constexpr uint32_t ([^;]+);', source):
        for piece in statement.split(','):
            if '=' not in piece:
                continue
            name, expression = (part.strip() for part in piece.split('=', 1))
            if 'get_' in expression or '(' in expression:
                continue
            try:
                found[name] = int(eval(expression, {}, dict(found)))
            except Exception:  # noqa: BLE001 - a non-constant right-hand side
                pass
    return found


def indices(source, pattern):
    return sorted({int(match) for match in re.findall(pattern, source)})


class KernelTextTests(unittest.TestCase):
    reader = text('gdn_wc_reader.cpp')
    writer = text('gdn_wc_writer.cpp')

    def test_the_kernel_files_are_lf_only_and_say_what_they_are(self):
        for name in ('gdn_wc_reader.cpp', 'gdn_wc_writer.cpp', 'gdn_wc.py', 'gdn_wc_emulate.py', 'w3_wc_smoke.py', 'test_gdn_wc.py'):
            with self.subTest(name=name):
                self.assertNotIn(b'\r', (HERE / name).read_bytes())
        for source in (self.reader, self.writer):
            self.assertIn('QWEN_FAST_TP4_GDN_WC', source)
            self.assertTrue(source.startswith('// SPDX-FileCopyrightText'))

    def test_the_runtime_words_the_kernels_read_are_below_rt_words(self):
        for source, role in ((self.reader, 'reader'), (self.writer, 'writer')):
            words = indices(source, r'get_arg_val<uint32_t>\((\d+)\)')
            self.assertEqual(words, list(range(6)), role)
            self.assertLess(max(words), gdn_wc.RT_WORDS[role])
            self.assertIn('static_assert(5 < RT_WORDS', source)

    def test_the_compile_words_the_kernels_read_are_the_modules(self):
        shape = plan4()['shape']
        reader = gdn_wc.compile_arguments('reader', shape, [])
        writer = gdn_wc.compile_arguments('writer', shape, [])
        self.assertEqual(indices(self.reader, r'get_compile_time_arg_val\((\d+)\)'), list(range(10)))
        self.assertIn('TensorAccessorArgs<10>()', self.reader)
        self.assertEqual(len(reader) - 1, 10, 'ten constants, then the accessors and RT_WORDS (the last word) after them')
        self.assertEqual(indices(self.writer, r'get_compile_time_arg_val\((\d+)\)'), list(range(5)))
        self.assertIn('TensorAccessorArgs<5>()', self.writer)
        self.assertEqual(len(writer) - 1, 5)
        # the names they are read into, in order
        names = re.findall(r'constexpr uint32_t (\w+) = get_compile_time_arg_val\((\d+)\);', self.reader)
        self.assertEqual([name for name, index in names[:10]],
                         ['K_ARG', 'Ct', 'Nvt', 'USERS', 'Wt', 'ACOL', 'BCOL', 'NV', 'ZFIRST', 'ZCOLS'])
        self.assertEqual([int(index) for name, index in names[:10]], list(range(10)))
        self.assertEqual(reader[:10], [4, shape['channel_tiles'], shape['gate_columns'], shape['users'], shape['width_tiles'], shape['a_col'],
                                       shape['b_col'], shape['heads'], shape['z_first'], shape['z_columns']])
        names = re.findall(r'constexpr uint32_t (\w+) = get_compile_time_arg_val\((\d+)\);', self.writer)
        self.assertEqual([name for name, index in names[:5]], ['K_ARG', 'Ct', 'Nvt', 'USERS', 'ZCOLS'])
        self.assertEqual(writer[:5], [4, shape['channel_tiles'], shape['gate_columns'], shape['users'], shape['z_columns']])

    def test_the_common_word_layout_is_the_modules(self):
        found = constants(self.reader)
        self.assertEqual((found['X_WORD'], found['H_WORD'], found['TAP_WORD'], found['DTB_WORD'], found['NEGA_WORD']), (0, 1, 13, 17, 18))
        self.assertEqual(found['NEGA_WORD'] + 1, gdn_wc.READER_COMMON_WORDS)
        addresses = dict(x=1000, history=[[2000 + 10 * u + i for i in range(4)] for u in range(4)], tap=[3000, 3001, 3002, 3003],
                         dt_bias=4000, neg_exp_A=4001)
        words = gdn_wc.reader_common(addresses, 4)
        self.assertEqual(words[found['X_WORD']], 1000)
        for user in range(4):
            for h in range(1, 4):
                self.assertEqual(words[found['H_WORD'] + found['history_used'] * user + (h - 1)], 2000 + 10 * user + h)
        self.assertEqual(words[found['TAP_WORD']:found['TAP_WORD'] + 4], [3000, 3001, 3002, 3003])
        self.assertEqual((words[found['DTB_WORD']], words[found['NEGA_WORD']]), (4000, 4001))
        wf = constants(self.writer)
        self.assertEqual((wf['CONV_WORD'], wf['BETA_WORD'], wf['G_WORD'], wf['Z_WORD'], wf['WINDOW_WORD']), (0, 4, 8, 12, 16))
        out = dict(conv=[100 + u for u in range(4)], beta=[110 + u for u in range(4)], g=[120 + u for u in range(4)],
                   z=[130 + u for u in range(4)], window=[[200 + 10 * u + s for s in range(4)] for u in range(4)])
        words = gdn_wc.writer_common(out, 4)
        self.assertEqual(words[wf['CONV_WORD'] + 3], 103)
        self.assertEqual(words[wf['BETA_WORD'] + 2], 112)
        self.assertEqual(words[wf['G_WORD'] + 1], 121)
        self.assertEqual(words[wf['Z_WORD'] + 3], 133)
        self.assertEqual(words[wf['WINDOW_WORD'] + 4 * 3 + 2], 232)
        self.assertEqual(wf['WINDOW_WORD'] + 16, gdn_wc.WRITER_COMMON_WORDS)

    def test_the_constants_are_the_geometry(self):
        found = constants(self.reader)
        self.assertEqual((found['page'], found['half_page'], found['face'], found['row'], found['span']), (2048, 1024, 512, 32, 64))
        self.assertEqual((found['page'], found['half_page'], found['face'], found['row'], found['span']),
                         (gdn_wc.PAGE, gdn_wc.HALF, gdn_wc.FACE, gdn_wc.ROW, gdn_wc.SPAN))
        self.assertEqual((found['history'], found['history_used'], found['K']), (gdn_wc.HISTORY, gdn_wc.HISTORY_USED, gdn_wc.K))
        writer = constants(self.writer)
        self.assertEqual((writer['page'], writer['half_page'], writer['K'], writer['user_pages']), (2048, 1024, 4, gdn_wc.USTAGE_PAGES))

    def test_the_circular_buffer_indices_are_the_plan_and_the_served_factorys(self):
        found = constants(self.reader)
        self.assertEqual((found['cb_win'], found['cb_tap'], found['cb_a'], found['cb_b'], found['cb_dtb'], found['cb_nega']), (0, 1, 6, 7, 8, 9))
        self.assertEqual((found['cb_gsrc'], found['cb_hist'], found['cb_z']), (gdn_wc.CB_GSRC, gdn_wc.CB_HIST, gdn_wc.CB_Z))
        writer = constants(self.writer)
        self.assertEqual((writer['cb_out'], writer['cb_shift'], writer['cb_beta'], writer['cb_g']), (4, 5, 11, 12))
        self.assertEqual((writer['cb_z'], writer['cb_ustage']), (gdn_wc.CB_Z, gdn_wc.CB_USTAGE))
        served = text('fixtures/gdn_conv_gates_served/kernels/compute/gdn_conv_gates.cpp')
        compute = constants(served)
        for name in ('cb_win', 'cb_tap', 'cb_out', 'cb_shift', 'cb_a', 'cb_b', 'cb_dtb', 'cb_nega', 'cb_beta', 'cb_g'):
            self.assertEqual((found if name in found else writer)[name], compute[name], name)
        self.assertEqual(sorted(gdn_wc.CB_PLAN), list(range(17)))

    def test_the_window_formulas_are_the_t2_row_map(self):
        self.assertIn('for (uint32_t r = 0; r + s < history; ++r)', self.reader)
        self.assertIn('local_copy(stage + ((ul * history_used + (r + s - 1)) * 2 + f) * span, destination + r * row, row);', self.reader)
        self.assertIn('local_copy(x_tile + face_index * face, destination + (history - s) * row, (16 - history + s) * row);', self.reader)
        self.assertIn('noc_async_read(hist.get_noc_addr(cc, f * face), stage + ((ul * history_used + (h - 1)) * 2 + f) * span, span);', self.reader)
        self.assertIn('for (uint32_t s = 1; s < history; ++s)', self.reader, 'slot 0 is never built')
        self.assertIn('canonicalise(x_tile + half_page, half_page);', self.reader, 'the second user half of x only')

    def test_the_canonical_rule_is_gdn_rows_dma_tps(self):
        rows = text('gdn_rows_dma_tp.cpp')
        for source in (self.reader, rows):
            self.assertIn('return (value & 0x7F80u) == 0 ? 0u : value;', source)
            self.assertIn('return value == 0x8000u ? 0u : value;', source)
            self.assertIn('#define CANON_DENORM 1', source)
        self.assertIn('canonical_half(value & 0xFFFFu) | (canonical_half(value >> 16) << 16)', self.reader)

    def test_no_new_api_surface(self):
        """Every function the kernels call is one a production kernel (the T2 windows, the row mover, F1) already calls."""
        production = ''.join(text(name) for name in ('gdn_conv_windows_packed.cpp', 'gdn_rows_dma_tp.cpp', 'gdn_conv_gates_spread_reader.cpp',
                                                      'gdn_conv_gates_spread_writer.cpp'))
        pattern = re.compile(r'\b((?:noc_|cb_|get_|invalidate_)\w+)\s*[<(]')
        local = {'get_compile_time_arg_val', 'get_arg_val', 'get_common_arg_val'}
        for name, source in (('reader', self.reader), ('writer', self.writer)):
            for call in sorted(set(pattern.findall(re.sub(r'//[^\n]*', '', source)))):
                with self.subTest(kernel=name, call=call):
                    self.assertIn(call, production, '%s is not called by any production kernel' % call)
        # invalidate_l1_cache is the one call only the non-default branch of the T2 windows kernel makes (its settings matrix ran on a card)
        self.assertIn('invalidate_l1_cache();', text('gdn_conv_windows_packed.cpp'))

    def test_no_bare_half_identifier(self):
        for source in (self.reader, self.writer):
            for line in source.splitlines():
                code = line.partition('//')[0]
                self.assertIsNone(re.search(r'\bhalf\b', code), line)

    def test_the_runtime_word_table_of_each_role_is_one_length(self):
        self.assertEqual(gdn_wc.RT_WORDS, dict(reader=8, writer=8, compute=2))
        self.assertEqual(gdn_wc.runtime_arguments('reader', dict(start=2, n_inst=2, g_n=0, gate_start=0, z_start=0, n_z=0)), [2, 2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(gdn_wc.runtime_arguments('compute', dict(start=2, n_inst=2, g_n=0, gate_start=0, z_start=0, n_z=0)), [2, 0])
        with self.assertRaises(ValueError):
            gdn_wc.runtime_arguments('unknown', {})

    def test_the_modules_that_ship_are_py37_syntax(self):
        for name in ('gdn_wc.py', 'w3_wc_smoke.py'):
            source = (HERE / name).read_text(encoding='utf-8')
            ast.parse(source, feature_version=(3, 7))

    def test_the_new_kernels_are_not_the_f1_kernels(self):
        digests = gdn_wc.source_sha256()
        self.assertEqual(sorted(digests), ['reader', 'writer'])
        self.assertNotEqual(digests, spread.source_sha256())


# --- the sources ------------------------------------------------------------------------------------------------------------------

class SourcesTests(unittest.TestCase):
    def setUp(self):
        gdn_wc._SOURCES.clear()
        self.addCleanup(gdn_wc._SOURCES.clear)
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root), True)
        target = self.root / spread.DIRECTORY
        shutil.copytree(str(HERE / 'fixtures' / 'gdn_conv_gates_served'), str(target))
        self.target = target

    def test_the_pinned_tree_gives_the_served_compute_and_our_two_kernels(self):
        found = gdn_wc.sources(self.root)
        self.assertEqual(sorted(found), ['compute', 'reader', 'writer'])
        self.assertEqual(found['compute'], (self.target / spread.COMPUTE_FILE).read_text(encoding='utf-8'))
        self.assertEqual(found['reader'], text('gdn_wc_reader.cpp'))
        self.assertEqual(found['writer'], text('gdn_wc_writer.cpp'))

    def test_any_changed_served_file_is_refused(self):
        for name in spread.SERVED:
            with self.subTest(name=name):
                path = self.target / name
                original = path.read_bytes()
                path.write_bytes(original + b'\n// changed\n')
                gdn_wc._SOURCES.clear()
                try:
                    with self.assertRaises(gdn_wc.SourceMismatch):
                        gdn_wc.sources(self.root)
                finally:
                    path.write_bytes(original)

    def test_a_missing_file_is_refused(self):
        (self.target / spread.COMPUTE_FILE).unlink()
        with self.assertRaises(gdn_wc.SourceMismatch):
            gdn_wc.sources(self.root)

    def test_the_defines_are_the_row_movers_canonical_rule_and_the_source_key(self):
        reader = dict(gdn_wc.kernel_defines('reader'))
        self.assertEqual(reader['CANON_DENORM'], '1')
        self.assertEqual(reader['WC_SRC_SHA'], '0x' + gdn_wc.source_sha256()['reader'][:8])
        self.assertEqual(gdn_wc.kernel_defines('compute'), [], 'the served source is compiled as served')
        self.assertEqual(dict(gdn_wc.kernel_defines('writer', canon_denorm=False))['CANON_DENORM'], '0')


# --- the launch on a fake ttnn -----------------------------------------------------------------------------------------------------

class WcKernel(Kernel):
    def __init__(self, kernel_source, source_type, core_ranges, compile_time_args, config, defines=None):
        Kernel.__init__(self, kernel_source, source_type, core_ranges, compile_time_args, config)
        self.defines, self.common_runtime_args = defines, None


class WcFakeTTNN(FakeTTNN):
    KernelDescriptor = WcKernel


SYNTHETIC = {'compute': 'compute source', 'reader': 'reader source', 'writer': 'writer source'}


class LaunchBase(object):
    def setUp(self):
        gdn_wc._NOTED.clear()
        self.lines = []
        for target in (patch.object(gdn_wc, 'log_line', side_effect=self.lines.append),
                       patch.object(gdn_wc, 'sources', return_value=SYNTHETIC)):
            target.start()
            self.addCleanup(target.stop)
        self.operations = WcFakeTTNN()
        with clean(**FOUR):
            self.found = tp_shapes.active()
        qkv = self.found.gdn_qkv
        self.projected = Tensor('projected', (1, 64, self.found.gdn_qkvzab_padded), 1000)
        self.histories = [[Tensor('history%d.%d' % (user, index), (1, 1, qkv), 2000 + 100 * user + 10 * index) for index in range(4)]
                          for user in range(4)]
        self.taps = [Tensor('tap%d' % slot, (1, 1, qkv), 3000 + slot) for slot in range(4)]
        self.dt_bias = Tensor('dt', (1, 1, self.found.gdn_nv), 4000)
        self.neg = Tensor('neg', (1, 1, self.found.gdn_nv), 4001)

    def call(self, flags=LEVER, **overrides):
        arguments = dict(operations=self.operations, mesh=fake_mesh(), projected=self.projected, histories=self.histories, taps=self.taps,
                         dt_bias=self.dt_bias, neg_exp_A=self.neg)
        arguments.update(overrides)
        with clean(**flags):
            return gdn_wc.launch(**arguments)


class LaunchTests(LaunchBase, unittest.TestCase):
    def test_one_generic_op_with_the_outputs_last(self):
        made = self.call()
        self.assertEqual([len(made[name]) for name in ('conv', 'beta', 'g', 'z')], [4, 4, 4, 4])
        self.assertEqual([len(user) for user in made['window']], [4, 4, 4, 4])
        self.assertEqual([tensor.shape for tensor in made['conv']], [(1, 16, 2560)] * 4)
        self.assertEqual([tensor.shape for tensor in made['beta'] + made['g']], [(1, 16, 12)] * 8)
        self.assertEqual([tensor.shape for tensor in made['z']], [(1, 16, 1536)] * 4)
        self.assertEqual([tensor.shape for user in made['window'] for tensor in user], [(1, 16, 2560)] * 16)
        every = [*made['conv'], *made['beta'], *made['g'], *made['z'], *[w for user in made['window'] for w in user]]
        self.assertTrue(all(t.memory == 'dram' and t.dtype == 'bf16' and t.layout == 'tile' for t in every))
        self.assertEqual(len(self.operations.launches), 1)
        names, program = self.operations.launches[0]
        inputs = ['projected'] + ['history%d.%d' % (u, i) for u in range(4) for i in range(4)] + ['tap%d' % s for s in range(4)] + ['dt', 'neg']
        self.assertEqual(names[:len(inputs)], inputs)
        self.assertEqual(names[len(inputs):], [tensor.name for tensor in self.operations.allocated])
        self.assertEqual(len(self.operations.allocated), 4 * 4 + 16)
        self.assertEqual(len(set(names)), len(names))

    def test_a_program_per_chip_with_three_kernels(self):
        self.call()
        program = self.operations.launches[0][1]
        self.assertEqual(sorted(program), [((0, chip), (0, chip)) for chip in range(4)])
        for descriptor in program.values():
            self.assertEqual([kernel.kernel_source for kernel in descriptor.kernels], ['reader source', 'writer source', 'compute source'])
            self.assertTrue(all(kernel.source_type == 'source' for kernel in descriptor.kernels))

    def test_the_cores_and_their_words(self):
        self.call()
        reader, writer, compute = self.operations.launches[0][1][((0, 2), (0, 2))].kernels
        work = plan4()
        for kernel, role in ((reader, 'reader'), (writer, 'writer'), (compute, 'compute')):
            table = kernel.runtime_args.table
            cores = sorted((x, y) for x in table for y in table[x])
            self.assertEqual(len(cores), 110, role)
            self.assertEqual({len(table[x][y]) for x, y in cores}, {gdn_wc.RT_WORDS[role]}, role)
        for item in work['work']:
            x, y = item['point']
            self.assertEqual(reader.runtime_args.table[x][y], gdn_wc.reader_runtime(item))
            self.assertEqual(writer.runtime_args.table[x][y], gdn_wc.reader_runtime(item))
            self.assertEqual(compute.runtime_args.table[x][y], [item['n_inst'], item['g_n']])
        self.assertEqual(reader.runtime_args.table[0][0], [0, 2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(reader.runtime_args.table[8][0], [0, 0, 1, 0, 0, 0, 0, 0], 'core 80: gate tile 0')
        self.assertEqual(reader.runtime_args.table[8][2][:6], [0, 0, 0, 0, 0, 4], 'core 82: the first z core takes four z tiles')

    def test_the_common_words_are_this_chips_addresses(self):
        self.call()
        program = self.operations.launches[0][1]
        outputs = self.operations.allocated
        for chip in range(4):
            reader, writer, compute = program[((0, chip), (0, chip))].kernels
            words = reader.common_runtime_args
            self.assertEqual(len(words), gdn_wc.READER_COMMON_WORDS)
            self.assertEqual(words[0], 1000 + chip)
            self.assertEqual(words[1:4], [2000 + 10 * i + chip for i in (1, 2, 3)], 'histories 1..3 of user 0')
            self.assertEqual(words[10:13], [2300 + 10 * i + chip for i in (1, 2, 3)], 'user 3')
            self.assertEqual(words[13:17], [3000 + s + chip for s in range(4)])
            self.assertEqual(words[17:], [4000 + chip, 4001 + chip])
            wwords = writer.common_runtime_args
            self.assertEqual(len(wwords), gdn_wc.WRITER_COMMON_WORDS)
            self.assertEqual(wwords, [tensor.address + chip for tensor in outputs[:16]] + [tensor.address + chip for tensor in outputs[16:]])
            self.assertIsNone(compute.common_runtime_args)

    def test_compile_words_defines_and_configs(self):
        self.call()
        reader, writer, compute = self.operations.launches[0][1][((0, 0), (0, 0))].kernels
        self.assertEqual(reader.compile_time_args[:10], [4, 80, 1, 4, 129, 4096, 4108, 12, 80, 48])
        self.assertEqual(len(reader.compile_time_args), 10 + 3 * 2 + 1)
        self.assertEqual(reader.compile_time_args[-1], gdn_wc.RT_WORDS['reader'])
        self.assertEqual(writer.compile_time_args[:5], [4, 80, 1, 4, 48])
        self.assertEqual(len(writer.compile_time_args), 5 + 2 + 1)
        self.assertEqual(writer.compile_time_args[-1], gdn_wc.RT_WORDS['writer'])
        self.assertEqual(compute.compile_time_args, [4, 1, 0x3F800000, 0x41A00000])
        self.assertEqual((reader.config, writer.config), (('movement', 'riscv1', 'noc1'), ('movement', 'riscv0', 'noc0')))
        self.assertEqual(compute.config, ('compute', 'hifi4', True, False), 'the served compute config')
        self.assertEqual(dict(reader.defines)['CANON_DENORM'], '1')
        self.assertIsNone(compute.defines)

    def test_the_circular_buffers_are_the_plan_on_every_core(self):
        self.call()
        cbs = self.operations.launches[0][1][((0, 0), (0, 0))].cbs
        plan = {cb['formats'][0][0]: (cb['total_size'], cb['formats'][0][1], cb['formats'][0][2]) for cb in cbs}
        self.assertEqual(sorted(plan), list(range(17)))
        for index in range(13):
            pages, kind = spread.CB_PLAN[index]
            page = 2048 if kind == 'bf16' else 4096
            self.assertEqual(plan[index], (pages * page, 'bf16' if kind == 'bf16' else 'fp32', page))
        self.assertEqual(plan[13], (4 * 2048, 'bf16', 2048))
        self.assertEqual(plan[14], (2048, 'bf16', 2048))
        self.assertEqual(plan[15], (2 * 2048, 'bf16', 2048))
        self.assertEqual(plan[16], (11 * 2048, 'bf16', 2048))
        self.assertEqual(sum(total for total, kind, page in plan.values()), gdn_wc.cb_bytes())
        self.assertEqual({cb['core_ranges'] for cb in cbs}, {cbs[0]['core_ranges']})

    def test_the_engaged_line_once(self):
        self.call()
        self.call()
        engaged = [line for line in self.lines if line.startswith(gdn_wc.ENGAGED)]
        self.assertEqual(engaged, [gdn_wc.ENGAGED + ' rows=64 conv_cores=80 gate_cores=2 z_cores=28 per_core=2 served_per_core=2 audit=0'])
        self.assertEqual(w3_wc_smoke.problems(dict(LEVER), '\n'.join(self.lines)), [])

    def test_two_users(self):
        made = self.call(histories=self.histories[:2], projected=Tensor('projected', (1, 32, self.found.gdn_qkvzab_padded), 1000))
        self.assertEqual(len(made['conv']), 2)
        self.assertIn('rows=32 conv_cores=80 gate_cores=1 z_cores=29', self.lines[-1])

    def test_an_odd_user_count_falls_back_with_nothing_allocated(self):
        self.assertIsNone(self.call(histories=self.histories[:3], projected=Tensor('projected', (1, 48, self.found.gdn_qkvzab_padded), 1000)))
        self.assertIn('3 users', self.lines[-1])
        self.assertEqual((self.operations.allocated, self.operations.launches), ([], []))

    def test_each_unfit_operand_falls_back(self):
        def change(tensor, **attributes):
            for key, value in attributes.items():
                setattr(tensor, key, value)

        for label, mutate, expected in (
                ('dtype', lambda: change(self.projected, dtype='fp32'), 'not bfloat16 TILE'),
                ('layout', lambda: change(self.histories[1][2], layout='row'), 'not bfloat16 TILE'),
                ('memory', lambda: change(self.histories[0][0], memory='sharded'), 'not interleaved'),
                ('mixed history placements', lambda: change(self.histories[2][1], memory='l1'), 'history placements are mixed'),
                ('rows', lambda: change(self.projected, shape=(1, 96, self.found.gdn_qkvzab_padded)), 'projection shape'),
                ('width', lambda: change(self.projected, shape=(1, 64, 100)), 'projection width'),
                ('history shape', lambda: change(self.histories[3][0], shape=(1, 1, 5120)), 'a history is not'),
                ('tap shape', lambda: change(self.taps[0], shape=(1, 1, 2592)), 'a tap is not'),
                ('tap placement', lambda: change(self.taps[1], memory='l1'), 'not in DRAM'),
                ('dt shape', lambda: change(self.neg, shape=(1, 1, 24)), 'dt_bias and neg_exp_A')):
            with self.subTest(label):
                self.setUp()
                mutate()
                self.assertIsNone(self.call())
                self.assertTrue(self.lines[-1].startswith(gdn_wc.FELL_BACK), self.lines)
                self.assertIn(expected, self.lines[-1])
                self.assertEqual((self.operations.allocated, self.operations.launches), ([], []))

    def test_missing_histories_or_taps_fall_back(self):
        self.assertIsNone(self.call(histories=[user[:3] for user in self.histories]))
        self.assertIn('histories per user', self.lines[-1])
        self.setUp()
        self.assertIsNone(self.call(taps=self.taps[:3]))
        self.assertIn('not 4 taps', self.lines[-1])

    def test_the_pair_falls_back(self):
        self.assertIsNone(self.call(flags={'QWEN_FAST_TP': '2'}))
        self.assertIn('not four cards', self.lines[-1])

    def test_a_reason_is_logged_once(self):
        self.projected.dtype = 'fp32'
        self.call()
        self.call()
        self.assertEqual(len([line for line in self.lines if line.startswith(gdn_wc.FELL_BACK)]), 1)

    def test_a_tree_that_is_not_the_pinned_one_falls_back(self):
        with patch.object(gdn_wc, 'sources', side_effect=gdn_wc.SourceMismatch('the conv-gates source x is not the pinned one')):
            self.assertIsNone(self.call())
        self.assertIn('not the pinned one', self.lines[-1])
        self.assertEqual(self.operations.allocated, [])

    def test_a_grid_without_room_falls_back(self):
        with clean(**LEVER):
            self.assertIsNone(gdn_wc.launch(self.operations, fake_mesh(82, 1), self.projected, self.histories, self.taps, self.dt_bias, self.neg))
        self.assertIn('no core is left', self.lines[-1])
        self.assertEqual(self.operations.allocated, [])

    def test_a_non_uniform_accessor_class_falls_back_and_frees_what_it_made(self):
        calls = {'count': 0}

        def accessor(tensor):
            calls['count'] += 1
            return SimpleNamespace(get_compile_time_args=lambda: [7, 9] if tensor.name != 'history2.1:3' else [7, 8])

        self.operations.TensorAccessorArgs = accessor
        self.assertIsNone(self.call())
        self.assertIn('different accessor arguments', self.lines[-1])
        self.assertEqual(sorted(self.operations.freed), sorted(tensor.name for tensor in self.operations.allocated))
        self.assertEqual(self.operations.launches, [])

    def test_a_chip_count_that_is_not_the_meshs_falls_back_and_frees_the_outputs(self):
        self.assertIsNone(self.call(chips=5))
        self.assertIn('every chip', self.lines[-1])
        self.assertEqual(sorted(self.operations.freed), sorted(tensor.name for tensor in self.operations.allocated))

    def test_a_failure_in_the_launch_frees_the_outputs_and_raises(self):
        def refuse(tensors, program):
            raise RuntimeError('device')

        self.operations.generic_op = refuse
        with self.assertRaises(RuntimeError):
            self.call()
        self.assertEqual(sorted(self.operations.freed), sorted(tensor.name for tensor in self.operations.allocated))


# --- the stage -------------------------------------------------------------------------------------------------------------------

class StageTests(LaunchBase, unittest.TestCase):
    def stage(self, flags=LEVER, groups=None, **overrides):
        arguments = dict(mesh=fake_mesh(), projected=self.projected, histories=self.histories, taps=self.taps, dt_bias=self.dt_bias,
                         neg_exp_A=self.neg, operations=self.operations, groups=groups)
        arguments.update(overrides)
        with clean(**flags):
            return gdn_wc.stage(**arguments)

    def test_the_stage_is_a_block_stage_with_the_sixteen_windows(self):
        found = self.stage()
        self.assertIsInstance(found, block.BlockStage)
        self.assertEqual(len(found.windows), 4)
        self.assertEqual([len(user) for user in found.windows], [4] * 4)
        self.assertEqual(len(found.users), 4)
        for (conv, beta, g), z in found.users:
            self.assertEqual((conv.shape, beta.shape, g.shape, z.shape), ((1, 16, 2560), (1, 16, 12), (1, 16, 12), (1, 16, 1536)))
        self.assertEqual((found.owned, found.held, found.entries), ([], [], []))
        flat = [tensor for (packed, z) in found.users for tensor in (*packed, z)] + [w for user in found.windows for w in user]
        self.assertEqual(len({id(tensor) for tensor in flat}), 32)

    def test_a_declined_call_is_none_and_says_so_once(self):
        notes = []
        self.projected.dtype = 'fp32'
        self.assertIsNone(self.stage(note_fallback=notes.append))
        self.assertEqual(notes, ['WC declined this layer'])
        self.assertEqual(len([line for line in self.lines if line.startswith(gdn_wc.FELL_BACK)]), 1)

    def test_the_audit_holds_the_served_path_beside_each_user_and_labels_the_entries(self):
        def hold(mesh, groups, windows, conv, beta, g, z, taps, dt, neg, operations, held):
            held.append('served-held')
            return [[dict(label='block conv user %d %s' % (user, what), mine=Tensor('m', (1,), 1), served=Tensor('s', (1,), 2))
                     for what in ('conv', 'beta', 'g', 'z') + tuple('window %d' % slot for slot in range(4))] for user in range(4)]

        with patch.object(block, 'hold_served', side_effect=hold) as held:
            found = self.stage(flags=AUDITED, groups=['group%d' % user for user in range(4)])
        held.assert_called_once()
        self.assertEqual(held.call_args[0][1], ['group0', 'group1', 'group2', 'group3'])
        self.assertEqual(found.held, ['served-held'])
        labels = [entry['label'] for user in found.entries for entry in user]
        self.assertEqual(len(labels), 32)
        self.assertEqual(labels[0], 'block conv user 0 wc conv')
        self.assertEqual(labels[-1], 'block conv user 3 wc window 3')
        self.assertTrue(all(label.startswith('block conv user ') and gdn_wc.WC_TAG in label for label in labels))
        for user in range(4):
            own = [label for label in labels if label.startswith('block conv user %d ' % user)]
            self.assertEqual(len(own), tp4_vglue.BLOCK_ENTRIES_PER_USER, 'the V1 audit still counts eight entries a user')

    def test_the_audit_without_the_pieces_raises_and_frees(self):
        with self.assertRaisesRegex(ValueError, 'needs the pieces'):
            self.stage(flags=AUDITED, groups=None)
        self.assertEqual(sorted(self.operations.freed), sorted(tensor.name for tensor in self.operations.allocated))

    def test_a_failing_audit_frees_everything(self):
        with patch.object(block, 'hold_served', side_effect=RuntimeError('audit')), self.assertRaises(RuntimeError):
            self.stage(flags=AUDITED, groups=['g'] * 4)
        self.assertEqual(sorted(self.operations.freed), sorted(tensor.name for tensor in self.operations.allocated))


# --- the binding: run_user_batched_projected(prebuilt=) and the V2 twin --------------------------------------------------------------

class PrebuiltTests(unittest.TestCase):
    """run_user_batched_projected with a prebuilt stage builds neither the T2 windows nor the block stage, and hands on the stage's tensors."""

    def setUp(self):
        patcher = patch.dict(os.environ, FOUR)
        patcher.start()
        self.addCleanup(patcher.stop)
        from test_gdn_user_batch_conv import fake_operations, user_group
        import verify_trace_t2

        verify_trace_t2.take()
        self.calls = []
        self.operations = fake_operations(self.calls)
        self.groups = [user_group(index, self.operations) for index in range(4)]
        self.execute_calls = []

    def stage(self):
        make = self.operations.make
        users = [((make('c%d' % index, (1, 16, 2560)), make('b%d' % index, (1, 16, 12)), make('g%d' % index, (1, 16, 12))),
                  make('z%d' % index, (1, 16, 1536))) for index in range(4)]
        stage = block.BlockStage(users, [], [], [])
        stage.windows = [[make('w%d.%d' % (user, slot), (1, 16, 2560)) for slot in range(4)] for user in range(4)]
        return stage

    def layer(self, prebuilt=None, block_stage=None, **flags):
        import gdn_seq_block
        import gdn_user_batch_conv

        def execute(mesh, inputs, kernels, operations, output_memory=None):
            self.execute_calls.append([tuple(value.name for value in user) for user in inputs])
            return [(self.operations.make('out%d' % index, (1, 16, 3072)), self.operations.make('states%d' % index, (16, 24, 128, 128)))
                    for index in range(len(inputs))]

        with env(**flags), patch.object(gdn_seq_block.batch, 'execute', side_effect=execute), \
                patch('gdn_conv_windows_packed.build_windows_packed') as packed, \
                patch('gdn_conv_windows.build_windows') as served, \
                patch('gdn_user_batch_conv.validate_projected', return_value=16):
            results = gdn_user_batch_conv.run_user_batched_projected(
                'mesh', self.groups, [self.operations.make('tap%d' % slot, (1, 1, 2560)) for slot in range(4)], 'dt', 'neg',
                self.operations.make('norm', (1, 1, 1536)), 'kernels', self.operations, block_stage=block_stage, prebuilt=prebuilt)
        return results, packed, served

    def test_the_prebuilt_stage_replaces_the_t2_windows_the_block_stage_and_the_per_user_calls(self):
        import verify_trace_t2

        stage = self.stage()
        block_stage = Mock()
        results, packed, served = self.layer(prebuilt=stage, block_stage=block_stage, QWEN_FAST_VERIFY_T2='1')
        packed.assert_not_called()
        served.assert_not_called()
        block_stage.assert_not_called()
        self.assertEqual([call for call in self.calls if call[0] in ('conv_gates', 'slice', 'windows')], [])
        self.assertEqual(self.execute_calls[0][0][:5], ('c0', 'b0', 'g0', 'initial0', 'z0'))
        for user, result in enumerate(results):
            self.assertEqual([window.name for window in result['packed_conv_states']], ['w%d.%d' % (user, slot) for slot in range(4)])
            names = [value.name for value in result['owned']]
            self.assertEqual(names[:4], ['w%d.%d' % (user, slot) for slot in range(4)])
            self.assertIn('c%d' % user, names)
            self.assertIn('z%d' % user, names)
        self.assertEqual(verify_trace_t2.take(), {'windows': 1}, 'the T2 windows count is noted once, as the T2 build notes it')

    def test_it_works_without_the_t2_flag_in_the_process_because_the_stage_made_the_windows(self):
        results, packed, served = self.layer(prebuilt=self.stage())
        packed.assert_not_called()
        self.assertEqual(len(results), 4)

    def test_the_stage_tensors_the_audit_holds_and_the_windows_are_freed_when_the_loop_fails(self):
        import gdn_seq_block
        import gdn_user_batch_conv

        stage = self.stage()
        stage.held = [self.operations.make('held', (1, 16, 2560))]
        with patch.object(gdn_seq_block.batch, 'execute', side_effect=RuntimeError('device')), \
                patch('gdn_conv_windows_packed.build_windows_packed'), patch('gdn_user_batch_conv.validate_projected', return_value=16):
            with self.assertRaises(RuntimeError):
                gdn_user_batch_conv.run_user_batched_projected(
                    'mesh', self.groups, [self.operations.make('tap%d' % slot, (1, 1, 2560)) for slot in range(4)], 'dt', 'neg',
                    self.operations.make('norm', (1, 1, 1536)), 'kernels', self.operations, prebuilt=stage)
        freed = {call[1] for call in self.calls if call[0] == 'free'}
        self.assertTrue({'held', 'w0.0', 'w3.3', 'c1', 'b2', 'g3', 'z0'} <= freed, sorted(freed))

    def test_without_it_the_loop_is_the_served_one(self):
        results, packed, served = self.layer(prebuilt=None, QWEN_FAST_VERIFY_T2='1')
        packed.assert_called_once()


class BindingTests(unittest.TestCase):
    """gdn_device_loop_state_tp over the packed-segments fixture's fakes: the split launch is skipped when WC is asked for, WC runs after the state moves,
    and a declined layer takes the V2 split and the V1 chain."""

    def setUp(self):
        import test_tp4_vglue_twin as twin_tests

        self.twin_tests = twin_tests
        self.base = twin_tests.TwinTests('test_flag_off_is_the_pinned_class_call_for_call')
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        tp4_vglue.take()

    def decode(self, flags, stage_result=None, decline=False, module=None):
        import gdn_device_loop_state_tp as twin
        import verify_trace_t2

        module = module or twin
        verify_trace_t2.take()
        base = self.base
        state, operations, layer, active, calls = base.build_state(module.DeviceLoopState)
        stage_calls, batched = [], []

        def launch(mesh, sources, destinations, tasks, **keywords):
            calls.append(('launch', tuple(source.name for source in sources), tuple(dest.name for dest in destinations)))
            base.launches.append((tuple(source.name for source in sources), tuple(dest.name for dest in destinations), tasks))

        def move(source, destination):
            calls.append(('copy_compact', source[0], destination[0]))

        def fake_stage(mesh, projected, histories, taps, dt_bias, neg_exp_A, operations_, groups=None, note_fallback=None, chips=None):
            calls.append(('wc_stage',))
            stage_calls.append(dict(histories=histories, taps=taps, groups=groups))
            return None if decline else stage_result

        def runner(mesh, users, taps, dt_bias, neg_exp_A, norm_w, kernels, operations_=None, **options):
            batched.append((tuple(type(piece).__name__ for piece, initial, history in users), tuple(getattr(piece, 'shape', None) for piece, i, h in users),
                            options.get('prebuilt')))
            calls.append(('batched',))
            return [dict(output=SimpleNamespace(shape=(1, piece.shape[1], 3072), name='out%d' % index, dtype='bf16', layout='tile',
                                                memory_config=lambda: 'l1'),
                         states=SimpleNamespace(shape=(piece.shape[1], 24, 128, 128)), conv_prefixes=[None] * piece.shape[1], owned=[],
                         packed_conv_states=[], packed_checkpoints=True, deferred_conv_publication=True, norm_batch=False,
                         prefix_zero_reuse=False, user_batched=True)
                    for index, (piece, initial, history) in enumerate(users)]

        with env(**flags), patch('gdn_rows_dma_tp.launch', side_effect=launch), patch('gdn_rows_dma_tp.problem', return_value=None), \
                patch('gdn_wc.stage', side_effect=fake_stage), \
                patch('gdn_device_loop_state.run_user_batched_projected', side_effect=runner), \
                patch.object(module, 'run_user_batched_projected', side_effect=runner), \
                patch('gdn_device_loop_state.restore_prefix'), patch('gdn_device_loop_state.release_owned'), \
                patch.object(module, 'release_owned'), \
                patch('gdn_device_loop_state.copy_compact', side_effect=move), patch.object(module, 'copy_compact', side_effect=move):
            packed = SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1')
            result = state.decode(packed, base.checkpoints, [0] * 4, segments=base.spans, slots=base.slots, deferred=True)
        return result, calls, stage_calls, batched

    def lever(self, **more):
        return dict(LEVER, **more)

    def fake_stage(self):
        stage = block.BlockStage([], [], [], [])
        stage.windows = [[] for user in range(4)]
        return stage

    def test_wc_on_skips_the_split_launch_and_runs_after_the_state_moves(self):
        stage = self.fake_stage()
        result, calls, stage_calls, batched = self.decode(self.lever(), stage_result=stage)
        self.assertEqual([entry for entry in calls if entry[0] == 'launch' and entry[1] == ('projected',)], [], 'no split launch')
        order = [entry[0] for entry in calls if entry[0] in ('copy_compact', 'wc_stage', 'batched')]
        self.assertEqual(order[-2:], ['wc_stage', 'batched'])
        self.assertTrue(all(name == 'copy_compact' for name in order[:-2]))
        self.assertEqual(len(stage_calls), 1)
        self.assertEqual(batched[0][0], ('WcPiece',) * 4)
        self.assertEqual(batched[0][1], ((1, 16, 8240),) * 4, 'stand-ins carry the (1, 16, W) shape the loop validates')
        self.assertIs(batched[0][2], stage, 'the stage reaches the batched loop as `prebuilt`')
        self.assertEqual(tp4_vglue.take(), {'gdn_glue': 1, 'gdn_merge': 1, 'gdn_block_conv': 1, 'gdn_wc': 1})
        self.assertEqual(stage_calls[0]['groups'], None)

    def test_the_stage_gets_each_users_conv_states_after_the_moves(self):
        result, calls, stage_calls, batched = self.decode(self.lever(), stage_result=self.fake_stage())
        histories = stage_calls[0]['histories']
        self.assertEqual(len(histories), 4)
        self.assertTrue(all(isinstance(user, list) and len(user) == 4 for user in histories), histories)

    def test_a_declined_layer_makes_the_real_pieces_after_the_moves_and_runs_the_v1_chain(self):
        result, calls, stage_calls, batched = self.decode(self.lever(), decline=True)
        splits = [entry for entry in calls if entry[0] == 'launch' and entry[1] == ('projected',)]
        self.assertEqual(len(splits), 1)
        names = [entry[0] for entry in calls]
        self.assertGreater(names.index('launch'), names.index('wc_stage'), 'the pieces are made after WC declined')
        self.assertEqual(batched[0][0], ('SimpleNamespace',) * 4, 'the loop gets the real pieces')
        self.assertIsNone(batched[0][2])
        self.assertEqual(tp4_vglue.take(), {'gdn_glue': 1, 'gdn_merge': 1})

    def test_the_audits_make_the_real_pieces_first_and_hand_them_to_the_stage(self):
        with patch('gdn_device_loop_state_tp.DeviceLoopState.hold_served_pieces', return_value=[[], [], [], []]), \
                patch('gdn_device_loop_state_tp.DeviceLoopState.hold_served_merge', return_value=[]):
            result, calls, stage_calls, batched = self.decode(dict(AUDITED), stage_result=self.fake_stage())
        self.assertEqual(batched[0][0], ('SimpleNamespace',) * 4, 'real pieces under the audit')
        self.assertEqual(len(stage_calls[0]['groups']), 4)
        self.assertEqual([len(group) for group in stage_calls[0]['groups']], [3, 3, 3, 3])
        self.assertLess([entry[0] for entry in calls].index('launch'), [entry[0] for entry in calls].index('wc_stage'))

    def test_flag_off_is_the_parents_twin_call_for_call_with_the_v1_levers_on(self):
        """The control: the same fixture through the twin as it was before this branch (git show of its base commit), the V2 glue, the V1 block conv and the T2
        windows on and WC unset. Every recorded op, split launch, hand-off to the batched loop and note is equal."""
        from test_dflash_proposal_trace import pinned_module

        parent = pinned_module('gdn_device_loop_state_tp.py', 'gdn_device_loop_state_tp_parent', commit=PARENT_COMMIT)
        if parent is None:
            raise unittest.SkipTest('no git history for %s' % PARENT_COMMIT)
        flags = dict(FOUR, **T2, QWEN_FAST_TP4_GDN_GLUE='1', QWEN_FAST_TP4_GDN_BLOCK_CONV='1')
        self.base.launches.clear()
        old_result, old_calls, old_stage, old_batched = self.decode(flags, module=parent)
        old_launches, old_notes = list(self.base.launches), tp4_vglue.take()
        self.base.launches.clear()
        new_result, new_calls, new_stage, new_batched = self.decode(flags)
        self.assertEqual(new_calls, old_calls)
        self.assertEqual(list(self.base.launches), old_launches)
        self.assertEqual(new_batched, old_batched)
        self.assertEqual(tp4_vglue.take(), old_notes)
        self.assertEqual(sorted(new_result), sorted(old_result))
        self.assertEqual((new_stage, old_stage), ([], []))

    def test_flag_off_the_stage_is_never_asked_and_the_split_runs(self):
        flags = dict(FOUR, **T2, QWEN_FAST_TP4_GDN_GLUE='1', QWEN_FAST_TP4_GDN_BLOCK_CONV='1')
        result, calls, stage_calls, batched = self.decode(flags, stage_result=None)
        self.assertEqual(stage_calls, [])
        self.assertEqual(len([entry for entry in calls if entry[0] == 'launch' and entry[1] == ('projected',)]), 1)
        self.assertEqual(batched[0][2], None)
        self.assertEqual(tp4_vglue.take(), {'gdn_glue': 1, 'gdn_merge': 1})


# --- the replay audit ------------------------------------------------------------------------------------------------------------

class RoundAuditTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        patcher = patch.object(gdn_wc, 'log_line', side_effect=self.lines.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        gdn_wc._AUDIT['rounds'] = 0

    def entries(self, users=4, wc=True, prefix='block conv user %d %s'):
        out = []
        for user in range(users):
            for what in ('conv', 'beta', 'g', 'z') + tuple('window %d' % slot for slot in range(4)):
                label = 'block conv user %d wc %s' % (user, what) if wc else 'block conv user %d %s' % (user, what)
                out.append(dict(label=label, mine=Tensor('m', (1,), 1), served=Tensor('s', (1,), 2)))
        return out

    def records(self, entries, layers=48):
        return [(None, dict(vglue_block_audit=list(entries))) for layer in range(layers)]

    def audit(self, entries, mismatches=None):
        operations = SimpleNamespace()
        with patch('tp4_vglue.compare_entry', side_effect=lambda ops, entry: list(mismatches or [])):
            return gdn_wc.audit_round(operations, self.records(entries), 1)

    def test_every_audited_layer_must_carry_the_users_entries_and_they_compare(self):
        import verify_trace_t2

        compared = self.audit(self.entries())
        self.assertEqual(compared, 32 * len(verify_trace_t2.audit_layers(1, 48)))
        self.assertTrue(self.lines[-1].startswith(gdn_wc.AUDIT_MARKER + ' 1 exact=True layers='))
        self.assertIn('entries=%d' % compared, self.lines[-1])

    def test_a_layer_without_wc_entries_fails_the_audit(self):
        with self.assertRaisesRegex(AssertionError, 'wc entries'):
            self.audit(self.entries(wc=False))
        self.assertTrue(self.lines[-1].startswith(gdn_wc.AUDIT_MISMATCH))

    def test_a_partial_set_fails(self):
        with self.assertRaisesRegex(AssertionError, 'positive multiple'):
            self.audit(self.entries()[:-1])

    def test_a_difference_logs_the_mismatch_marker_and_raises(self):
        with self.assertRaisesRegex(AssertionError, 'differ'):
            self.audit(self.entries(), mismatches=['conv chip 2: 5 of 100 elements differ'])
        self.assertTrue(self.lines[-1].startswith(gdn_wc.AUDIT_MISMATCH))

    def test_the_marker_formats_are_the_smoke_rules(self):
        self.audit(self.entries())
        self.assertEqual(w3_wc_smoke.problems(dict(AUDITED), '\n'.join([gdn_wc.engaged_line(plan4(), True)] + self.lines)), [])


# --- the smoke rule ------------------------------------------------------------------------------------------------------------------

ENGAGED_LINE = gdn_wc.ENGAGED + ' rows=64 conv_cores=80 gate_cores=2 z_cores=28 per_core=2 served_per_core=2 audit=0'
AUDIT_LINE = gdn_wc.AUDIT_MARKER + ' 3 exact=True layers=0,24 entries=64'


class SmokeRuleTests(unittest.TestCase):
    def test_without_the_flag_nothing_is_required_and_lines_are_refused(self):
        self.assertEqual(w3_wc_smoke.problems({}, 'nothing'), [])
        self.assertEqual(w3_wc_smoke.problems(None, 'nothing'), [])
        self.assertEqual(len(w3_wc_smoke.problems({}, ENGAGED_LINE)), 1)
        self.assertEqual(len(w3_wc_smoke.problems({}, AUDIT_LINE)), 1)

    def test_the_flag_needs_the_engaged_line_with_all_three_core_kinds(self):
        env_ = {gdn_wc.FLAG: '1'}
        self.assertEqual(len(w3_wc_smoke.problems(env_, 'no lines')), 1)
        self.assertEqual(w3_wc_smoke.problems(env_, ENGAGED_LINE), [])
        for field in ('conv_cores', 'gate_cores', 'z_cores'):
            line = re.sub(field + r'=\d+', field + '=0', ENGAGED_LINE)
            problems = w3_wc_smoke.problems(env_, line)
            self.assertEqual(len(problems), 1, field)
            self.assertIn(field, problems[0])

    def test_a_fall_back_line_fails_with_or_without_the_flag(self):
        line = gdn_wc.FELL_BACK + ' reason=3 users (WC takes 2 or 4)'
        self.assertEqual(len(w3_wc_smoke.problems({}, line)), 1)
        self.assertEqual(len(w3_wc_smoke.problems({gdn_wc.FLAG: '1'}, '\n'.join([ENGAGED_LINE, line]))), 1)

    def test_the_octo_block_s_decline_is_the_design_and_only_with_the_octo_block_on(self):
        """tp4/next-1: the ship profile runs the octo block (8 users of 8 rows) from six live seats; WC takes 2 or 4 users and declines it with one line, so that
        line is expected on a profile that names the octo block and a fault on any other; every other reason fails either way."""
        octo = gdn_wc.FELL_BACK + ' reason=8 users (WC takes 2 or 4)'
        other = gdn_wc.FELL_BACK + ' reason=3 users (WC takes 2 or 4)'
        on = {gdn_wc.FLAG: '1', 'QWEN_FAST_OCTO': 'live'}
        self.assertEqual(w3_wc_smoke.problems(on, '\n'.join([ENGAGED_LINE, octo])), [])
        self.assertEqual(len(w3_wc_smoke.problems(on, '\n'.join([ENGAGED_LINE, octo, other]))), 1)
        self.assertEqual(len(w3_wc_smoke.problems({gdn_wc.FLAG: '1'}, '\n'.join([ENGAGED_LINE, octo]))), 1)
        self.assertEqual(len(w3_wc_smoke.problems(dict(on, QWEN_FAST_OCTO='off'), '\n'.join([ENGAGED_LINE, octo]))), 1)

    def test_the_audit_flag_needs_a_passing_line(self):
        env_ = {gdn_wc.FLAG: '1', gdn_wc.AUDIT_FLAG: '1'}
        self.assertEqual(len(w3_wc_smoke.problems(env_, ENGAGED_LINE)), 1)
        self.assertEqual(w3_wc_smoke.problems(env_, '\n'.join([ENGAGED_LINE, AUDIT_LINE])), [])

    def test_a_mismatch_line_fails_even_beside_a_passing_one_and_is_not_a_passing_line(self):
        env_ = {gdn_wc.FLAG: '1', gdn_wc.AUDIT_FLAG: '1'}
        mismatch = gdn_wc.AUDIT_MISMATCH + ' round=3 layers=0 layer 0 conv chip 1: 2 of 9 elements differ'
        self.assertEqual(len(w3_wc_smoke.problems(env_, '\n'.join([ENGAGED_LINE, AUDIT_LINE, mismatch]))), 1)
        only = w3_wc_smoke.problems(env_, '\n'.join([ENGAGED_LINE, mismatch]))
        self.assertEqual(len(only), 2)


# --- shipment ----------------------------------------------------------------------------------------------------------------------

class ShipmentTests(unittest.TestCase):
    def test_the_module_and_its_kernels_ship_in_the_overlay_only(self):
        # New C2 modules go in docker/qwen-c2-overlay.txt ONLY: the P8 base lists are the production base's and stay as shipped.
        workflow = text('../../.github/workflows/qwen-fast-serving-image.yml')
        dockerfile = text('../../docker/qwen-fast-serving.Dockerfile')
        overlay = text('../../docker/qwen-c2-overlay.txt')
        for name in gdn_wc.RUNTIME_FILES:
            self.assertNotIn(name, workflow, name)
            self.assertNotIn(' scripts/ci/' + name, dockerfile, name)
            self.assertIn('scripts/ci/' + name, overlay.splitlines(), name)
        self.assertEqual(gdn_wc.RUNTIME_FILES, ('gdn_wc.py', 'gdn_wc_reader.cpp', 'gdn_wc_writer.cpp'))

    def test_the_replay_audit_is_called_from_the_packed_verifier_beside_f1s(self):
        source = text('packed_verifier.py')
        self.assertIn("os.environ.get('QWEN_FAST_TP4_GDN_WC_AUDIT', '0') != '0'", source)
        # audit_round_number is self.rounds + 1 unless an audit sampling arm thins the audited rounds (audit_sampling, gate profiles only)
        self.assertIn('gdn_wc.audit_round(self.operations, self.fixture.retained.records, audit_round_number)', source)
        self.assertLess(source.index('gdn_conv_gates_spread.audit_round('), source.index('gdn_wc.audit_round('))
        # the octo block (8 users of 8 rows) is never a WC block: its layers carry no WC entries, so the replay audit does not ask for them
        self.assertIn("os.environ.get('QWEN_FAST_TP4_GDN_WC_AUDIT', '0') != '0' and (self.users, self.rows_per_user) != (8, 8)", source)

    def test_the_emulation_and_the_smoke_rule_do_not_ship_in_the_image(self):
        overlay = text('../../docker/qwen-c2-overlay.txt').splitlines()
        self.assertNotIn('scripts/ci/gdn_wc_emulate.py', overlay)
        self.assertNotIn('scripts/ci/w3_wc_smoke.py', overlay)

    def test_these_tests_are_on_the_cpu_allowlist(self):
        workflow = text('../../.github/workflows/qwen-integration-cpu.yml')
        for name in ('test_gdn_wc', 'test_tp4_w3_wc'):
            self.assertRegex(workflow, r'python -B -m unittest [^\n]*\b%s\b' % name)
        self.assertIn("python -B -m unittest discover -s optimisation/ttnn-op/gdn_wc -p 'test_*.py'", workflow)


if __name__ == '__main__':
    unittest.main()
