"""F1 (gdn_conv_gates_spread, QWEN_FAST_TP4_CONV_GATES_SPREAD): the block conv-gates launch with the gate tiles on cores of their own.

Held here, all on the CPU:
  - the flag grammar (strict, refused at the pair, needs V1, the audit needs the lever and the vglue audit) and that an unset flag
    leaves the V1 stage's call the served one without importing the module;
  - the core plan: the served partition of the conv instances whenever the gate cores fit beside it, every conv instance and every gate
    tile owned exactly once, one gate tile per core, the plan growing instances per core (never dropping a tile) when they do not fit,
    and a refusal when the grid cannot hold them;
  - the GATHER: a transcription of the new reader's word gather (runs split at 16-column faces, word copies at even offsets, a 16-bit
    edge) against a transcription of the served reader's element-by-element gather, over every geometry the lever can meet (TP4's a at
    columns 0-11 and b at 12-23 of a tile, odd offsets, two source tiles, wider gates, fp32), byte for byte, with the word copies
    checked 4-byte aligned; the formulas are also looked up in the kernel text so the transcription cannot drift from it;
  - the kernels' text: runtime-argument counts against RT_WORDS (the generic_op cache lesson f945486e), the gate-start edits, and, when a
    tree with the served sources is at hand (GDN_CONV_ROOT or TT_METAL_HOME), that the conv loops are the served ones verbatim and the
    writer differs from the served one in exactly three places;
  - the launch on a fake ttnn: the per-chip programs (cores, runtime words, compile words, circular buffers, the compute kernel as served),
    one generic_op, the outputs, every fall-back (a refusal logs once and returns None with nothing allocated), the source pin, the audit
    entries and the replay audit;
  - the smoke rule, the image copy lists and the CPU allowlist.

What no CPU test can show is that the compiled kernels move those bytes: that is the audit (QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT) on a card.

Run at py 3.11: `py -3.11 -B -m unittest test_gdn_conv_gates_spread` from scripts/ci.
"""

import hashlib
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import c2_smoke_check
import gdn_block_conv_tp as block
import gdn_conv_gates_spread as spread
import tp4_vglue
import tp_shapes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FOUR = {'QWEN_FAST_TP': '4'}
LEVER = dict(FOUR, **{spread.FLAG: '1', 'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_TP4_GDN_BLOCK_CONV': '1'})
AUDITED = dict(LEVER, **{spread.AUDIT_FLAG: '1', 'QWEN_FAST_TP4_VGLUE_AUDIT': '1'})


def text(name):
    return (HERE / name).read_text(encoding='utf-8')


def env(**values):
    return patch.dict(os.environ, values)


def clean(**values):
    """A process environment holding only `values` (the CI runner's own variables cannot change a flag)."""
    return patch.dict(os.environ, values, clear=True)


# --- the flags ------------------------------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_unset_and_zero_are_off(self):
        with clean(**FOUR):
            self.assertFalse(spread.enabled())
            self.assertFalse(spread.audit_enabled())
        with clean(**dict(FOUR, **{spread.FLAG: '0'})):
            self.assertFalse(spread.enabled())

    def test_one_is_on_with_four_cards_and_v1(self):
        with clean(**LEVER):
            self.assertTrue(spread.enabled())

    def test_anything_else_raises(self):
        for value in ('2', 'true', 'on', ''):
            with self.subTest(value=value), clean(**dict(LEVER, **{spread.FLAG: value})):
                self.assertRaises(ValueError, spread.enabled)

    def test_the_pair_refuses(self):
        with clean(**{spread.FLAG: '1', 'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_TP4_GDN_BLOCK_CONV': '1'}):
            self.assertRaisesRegex(ValueError, 'TP4 lever', spread.enabled)

    def test_it_needs_the_block_conv(self):
        with clean(**dict(FOUR, **{spread.FLAG: '1'})):
            self.assertRaises(ValueError, spread.enabled)
        with clean(**dict(FOUR, **{spread.FLAG: '1', 'QWEN_FAST_TP4_GDN_BLOCK_CONV': '1'})):
            self.assertRaises(ValueError, spread.enabled)       # V1 itself needs V2

    def test_the_audit_needs_the_lever_and_the_vglue_audit(self):
        with clean(**dict(FOUR, **{spread.AUDIT_FLAG: '1'})):
            self.assertRaisesRegex(ValueError, 'needs', spread.enabled)
        with clean(**dict(LEVER, **{spread.AUDIT_FLAG: '1'})):
            self.assertRaisesRegex(ValueError, 'VGLUE_AUDIT', spread.audit_enabled)
        with clean(**AUDITED):
            self.assertTrue(spread.audit_enabled())
        with clean(**dict(AUDITED, **{spread.AUDIT_FLAG: '2'})):
            self.assertRaises(ValueError, spread.audit_enabled)

    def test_the_stage_literals_are_the_modules(self):
        self.assertEqual((block.SPREAD_FLAG, block.SPREAD_AUDIT_FLAG), (spread.FLAG, spread.AUDIT_FLAG))
        self.assertEqual((c2_smoke_check.SPREAD_FLAG, c2_smoke_check.SPREAD_AUDIT_FLAG), (spread.FLAG, spread.AUDIT_FLAG))
        self.assertEqual((c2_smoke_check.SPREAD_ENGAGED, c2_smoke_check.SPREAD_FELL_BACK, c2_smoke_check.SPREAD_AUDIT,
                          c2_smoke_check.SPREAD_MISMATCH),
                         (spread.ENGAGED, spread.FELL_BACK, spread.AUDIT_MARKER, spread.AUDIT_MISMATCH))
        self.assertTrue(spread.AUDIT_MISMATCH.startswith(spread.AUDIT_MARKER))


# --- the core plan --------------------------------------------------------------------------------------------------------------

def covered(work_plan):
    """(conv instances owned, gate tiles owned, cores used) from a plan: every list must hold each id once."""
    conv, gates, cores = [], [], []
    for point, start, n_inst, gate_start, g_n in work_plan['work']:
        conv.extend(range(start, start + n_inst))
        gates.extend(range(gate_start, gate_start + g_n))
        cores.append(point)
    return conv, gates, cores


class PlanTests(unittest.TestCase):
    def test_the_64_row_block_at_tp4(self):
        for grid in ((11, 10), (13, 10), (14, 10)):
            with self.subTest(grid=grid):
                found = spread.plan(grid[0], grid[1], 64, 64, 2560, 12)
                self.assertEqual((found['n_conv'], found['n_gate']), (160, 2))
                self.assertEqual(found['per_core'], found['served_per_core'], 'the served partition of the conv instances')
                conv, gates, cores = covered(found)
                self.assertEqual(sorted(conv), list(range(160)))
                self.assertEqual(sorted(gates), [0, 1])
                self.assertEqual(len(set(cores)), len(cores))

    def test_the_gate_tiles_are_alone_on_the_cores_after_the_conv_cores(self):
        found = spread.plan(11, 10, 64, 64, 2560, 12)
        work = found['work']
        conv_cores = found['conv_cores']
        self.assertEqual(conv_cores, 80)
        for index, (point, start, n_inst, gate_start, g_n) in enumerate(work):
            self.assertEqual(point, (index // 10, index % 10), 'the factory order: core c is (c // grid_y, c % grid_y)')
            if index < conv_cores:
                self.assertEqual((n_inst > 0, g_n), (True, 0))
            else:
                self.assertEqual((n_inst, g_n, gate_start), (0, 1, index - conv_cores))
        self.assertEqual(len(work), 82)

    def test_the_served_factory_puts_both_gate_tiles_on_one_core(self):
        per_core, conv_cores, own = spread.served_plan(11, 10, 64, 2560)
        self.assertEqual((per_core, conv_cores, own), (2, 80, True))      # one extra core carries gBt x Nvt = 2 tiles in the served plan

    def test_16_rows_is_the_served_single_gate_core(self):
        found = spread.plan(11, 10, 16, 16, 2560, 12)
        self.assertEqual((found['n_conv'], found['n_gate'], found['cores']), (80, 1, 81))

    def test_128_rows_grows_the_instances_per_core_when_the_gate_cores_do_not_fit(self):
        found = spread.plan(11, 10, 128, 128, 2560, 12)       # 320 instances: 3 a core is 107 cores, and 4 gate tiles make 111 of 110
        self.assertEqual((found['served_per_core'], found['per_core'], found['conv_cores']), (3, 4, 80))
        conv, gates, cores = covered(found)
        self.assertEqual(sorted(conv), list(range(320)))
        self.assertEqual(sorted(gates), [0, 1, 2, 3])
        self.assertLessEqual(len(cores), 110)
        wide = spread.plan(13, 10, 128, 128, 2560, 12)        # on 130 the gate tiles fit beside the served partition
        self.assertEqual(wide['per_core'], wide['served_per_core'])

    def test_rows_are_a_multiple_of_a_tile_in_gate_tiles(self):
        for rows, tiles in ((1, 1), (4, 1), (32, 1), (33, 2), (64, 2), (128, 4)):
            self.assertEqual(spread.plan(11, 10, 128, rows, 2560, 12)['n_gate'], tiles, rows)

    def test_more_heads_than_a_tile_make_more_gate_tiles(self):
        found = spread.plan(11, 10, 64, 64, 2560, 48)
        self.assertEqual(found['n_gate'], 4)
        self.assertEqual(sorted(covered(found)[1]), [0, 1, 2, 3])

    def test_a_grid_that_cannot_hold_the_gate_tiles_is_refused(self):
        self.assertRaises(ValueError, spread.plan, 1, 2, 64, 64, 2560, 12)
        for bad in ((0, 10, 64, 64, 2560, 12), (11, 10, 64, 64, 2561, 12), (11, 10, 64, 0, 2560, 12), (11, 10, 64, 64, 2560, 0)):
            self.assertRaises(ValueError, spread.plan, *bad)

    def test_a_random_sweep_owns_everything_once(self):
        generator = random.Random(5)
        for _ in range(60):
            grid = (generator.randint(4, 14), generator.randint(4, 10))
            bmax = generator.choice((16, 32, 64, 96, 128))
            rows = generator.randint(1, bmax)
            channels = 32 * generator.choice((8, 40, 80, 160))
            heads = generator.choice((1, 12, 24, 32, 40, 64))
            try:
                found = spread.plan(grid[0], grid[1], bmax, rows, channels, heads)
            except ValueError:
                continue
            conv, gates, cores = covered(found)
            self.assertEqual(sorted(conv), list(range(found['n_conv'])))
            self.assertEqual(sorted(gates), list(range(found['n_gate'])))
            self.assertEqual(len(set(cores)), len(cores))
            self.assertLessEqual(len(cores), grid[0] * grid[1])
            self.assertLessEqual(found['per_core'], found['served_per_core'] + 1)
            for point, start, n_inst, gate_start, g_n in found['work']:
                self.assertFalse(n_inst and g_n, 'a core is a conv core or a gate core, never both')
                self.assertLessEqual(g_n, 1)


# --- the gather -----------------------------------------------------------------------------------------------------------------

def tile_index(row, column):
    """Element index of (row, column) in a 32 x 32 tile: four 16 x 16 faces, face-major, row-major inside a face."""
    return ((row // 16) * 2 + column // 16) * 256 + (row % 16) * 16 + column % 16


def served_gate(pages, wt, col0, bt_g, t, heads):
    """The served reader's gather_gate: one element at a time into the zeroed tile."""
    destination = [0] * 1024
    h0 = t * 32
    h1 = min(h0 + 32, heads)
    for h in range(h0, h1):
        c = col0 + h
        source = pages[bt_g * wt + c // 32]
        for r in range(32):
            destination[tile_index(r, h - h0)] = source[tile_index(r, c % 32)]
    return destination


def spread_gate(pages, wt, col0, bt_g, t, heads, elem):
    """The new reader's gate gather for one of a and b: the source pages go into a four-slot scratch (a: slots 0-1, b: slots 2-3, the slot
    index (p - p0) from the first source page), the runs are cut at 16-column faces of both tiles, and a run moves as words when the
    element size is 2 and both columns are even. Returns (tile, how many word copies, how many 16-bit moves)."""
    h0 = t * 32
    h1 = min(h0 + 32, heads)
    p0 = (col0 + h0) // 32
    p1 = (col0 + h1 - 1) // 32
    scratch = {p - p0: pages[bt_g * wt + p] for p in range(p0, p1 + 1)}
    destination = [0] * 1024
    words = halves = 0
    h = h0
    while h < h1:
        c = col0 + h
        sc, dc = c % 32, h - h0
        run = min(h1 - h, 16 - sc % 16, 16 - dc % 16)
        source = scratch[c // 32 - p0]
        for r in range(32):
            se, de = tile_index(r, sc), tile_index(r, dc)
            if elem == 2 and ((sc | dc) & 1) == 0:
                assert (se * 2) % 4 == 0 and (de * 2) % 4 == 0
                for w in range(run // 2):
                    destination[de + 2 * w] = source[se + 2 * w]
                    destination[de + 2 * w + 1] = source[se + 2 * w + 1]
                    words += 1
                if run & 1:
                    destination[de + run - 1] = source[se + run - 1]
                    halves += 1
            else:
                for e in range(run):
                    destination[de + e] = source[se + e]
                    halves += 1
        h += run
    return destination, words, halves


def random_pages(generator, wt, rows_of_tiles):
    return {page: [generator.getrandbits(16) for _ in range(1024)] for page in range(wt * rows_of_tiles)}


class GatherTests(unittest.TestCase):
    GEOMETRIES = (
        # (a column, b column, heads): TP4's a at columns 0-11 of tile 128 and b at 12-23 of it
        (4096, 4108, 12),
        (0, 12, 12),
        (4096, 4120, 12),     # b straddles a tile edge
        (4097, 4109, 12),     # odd column offsets: element by element
        (4096, 4113, 12),     # an odd b offset
        (4100, 4130, 24),
        (4096, 4128, 32),
        (4093, 4125, 32),     # two source tiles for a 32-wide gate
        (4096, 4160, 40),     # two gate tiles (Nvt = 2): the second has 8 heads
        (31, 63, 5),
    )

    def test_the_word_gather_equals_the_element_gather_byte_for_byte(self):
        generator = random.Random(3)
        wt = 140
        pages = random_pages(generator, wt, 2)
        for a_col, b_col, heads in self.GEOMETRIES:
            for t in range(-(-heads // 32)):
                for bt_g in (0, 1):
                    for col0 in (a_col, b_col):
                        with self.subTest(a=a_col, b=b_col, heads=heads, t=t, bt=bt_g, col0=col0):
                            expected = served_gate(pages, wt, col0, bt_g, t, heads)
                            found, words, halves = spread_gate(pages, wt, col0, bt_g, t, heads, 2)
                            self.assertEqual(found, expected)

    def test_tp4_a_and_b_move_as_words(self):
        pages = random_pages(random.Random(4), 140, 2)
        found, words, halves = spread_gate(pages, 140, 4096, 0, 0, 12, 2)
        self.assertEqual((words, halves), (32 * 6, 0), 'a: six words a row, no 16-bit move')
        found, words, halves = spread_gate(pages, 140, 4108, 0, 0, 12, 2)
        self.assertEqual((words, halves), (32 * 6, 0), 'b: a 2-word run and a 4-word run a row')

    def test_the_move_count_beats_the_served_gather(self):
        pages = random_pages(random.Random(4), 140, 2)
        served_moves = 32 * 12
        for col0 in (4096, 4108):
            found, words, halves = spread_gate(pages, 140, col0, 0, 0, 12, 2)
            self.assertEqual(words * 2 + halves, served_moves)
            self.assertLess(words + halves, served_moves)

    def test_fp32_moves_element_by_element_like_the_served_reader(self):
        generator = random.Random(8)
        pages = random_pages(generator, 140, 1)
        expected = served_gate(pages, 140, 4108, 0, 0, 12)
        found, words, halves = spread_gate(pages, 140, 4108, 0, 0, 12, 4)
        self.assertEqual(found, expected)
        self.assertEqual(words, 0)

    def test_the_columns_past_the_heads_stay_zero(self):
        pages = {page: [0xFFFF] * 1024 for page in range(280)}
        found, words, halves = spread_gate(pages, 140, 4096, 0, 0, 12, 2)
        for r in range(32):
            for c in range(32):
                self.assertEqual(found[tile_index(r, c)], 0xFFFF if c < 12 else 0, (r, c))

    def test_a_row_of_a_tile_row_pair_is_gathered_from_its_own_tile_row(self):
        generator = random.Random(2)
        pages = random_pages(generator, 140, 2)
        first, _, _ = spread_gate(pages, 140, 4096, 1, 0, 12, 2)
        self.assertEqual(first, served_gate(pages, 140, 4096, 1, 0, 12))
        self.assertNotEqual(first, served_gate(pages, 140, 4096, 0, 0, 12))

    def test_the_kernel_text_holds_the_transcribed_formulas(self):
        reader = text('gdn_conv_gates_spread_reader.cpp')
        for formula in ('((r / 16) * 2 + (sc / 16)) * 256 + (r % 16) * 16 + (sc % 16)',
                        '((r / 16) * 2 + (dc / 16)) * 256 + (r % 16) * 16 + (dc % 16)',
                        'run = (16 - sc % 16 < run) ? 16 - sc % 16 : run;', 'run = (16 - dc % 16 < run) ? 16 - dc % 16 : run;',
                        'if (elem == 2 && ((sc | dc) & 1u) == 0)', 'const uint32_t words = run / 2;', 'if (run & 1u)',
                        'const uint32_t h1 = (h0 + 32 < NV) ? h0 + 32 : NV;',
                        'copy_run(sbase + (c / 32 - ap0) * tb, sc, dst_a, dc, run);',
                        'copy_run(sbase + (2 + c / 32 - bp0) * tb, sc, dst_b, dc, run);',
                        '{.page_id = bt_g * AWt + p}, {.offset_bytes = (p - ap0) * tb}',
                        '{.page_id = bt_g * BWt + p}, {.offset_bytes = (2 + p - bp0) * tb}',
                        '{.page_id = t}, {.offset_bytes = 0}'):
            self.assertIn(formula, reader)
        self.assertEqual(reader.count('async_read_barrier'), reader.count('noc.async_read_barrier();'))
        gate_body = reader[reader.index('auto gate = '):reader.index('// gates: one tile per instance')]
        self.assertEqual(gate_body.count('noc.async_read_barrier();'), 1, 'the four gate reads share ONE barrier')
        self.assertEqual(gate_body.count('noc.async_read('), 4, 'call sites: a and b (a loop each over their one or two source tiles), dt_bias, neg_exp_A')

    def test_the_scratch_ring_is_four_pages(self):
        self.assertEqual(spread.CB_PLAN[13], (4, 'bf16'))
        reader = text('gdn_conv_gates_spread_reader.cpp')
        self.assertIn('scb.reserve_back(4);', reader)
        self.assertIn('scb.push_back(4);', reader)
        self.assertIn('scb.pop_front(4);', reader)


# --- the kernels' text ----------------------------------------------------------------------------------------------------------

def highest_argument(source):
    return max(int(index) for index in re.findall(r'get_arg_val<uint32_t>\((\d+)\)', source))


class KernelTextTests(unittest.TestCase):
    def test_runtime_argument_counts_match_rt_words(self):
        reader, writer = text('gdn_conv_gates_spread_reader.cpp'), text('gdn_conv_gates_spread_writer.cpp')
        self.assertEqual(highest_argument(reader) + 1, spread.RT_WORDS['reader'])
        self.assertEqual(highest_argument(writer) + 1, spread.RT_WORDS['writer'])
        self.assertIn('static_assert(16 < RT_WORDS', reader)
        self.assertIn('static_assert(10 < RT_WORDS', writer)
        for source in (reader, writer):
            self.assertIn('constexpr uint32_t RT_WORDS = get_compile_time_arg_val(', source)

    def test_every_core_gets_the_same_number_of_words_per_role(self):
        addresses = dict(x=1, st=[2, 3, 4, 5], tap=[6, 7, 8, 9], dt_bias=10, neg_exp_A=11, conv=12, beta=13, g=14)
        for role in spread.ROLES:
            lengths = {len(spread.runtime_arguments(role, start, n, gate, g, addresses))
                       for start, n, gate, g in ((0, 2, 0, 0), (0, 0, 1, 1), (158, 2, 0, 0))}
            self.assertEqual(lengths, {spread.RT_WORDS[role]}, role)

    def test_gate_start_is_the_last_word_of_the_reader_and_the_writer(self):
        addresses = dict(x=1, st=[2, 3, 4, 5], tap=[6, 7, 8, 9], dt_bias=10, neg_exp_A=11, conv=12, beta=13, g=14)
        reader = spread.runtime_arguments('reader', 0, 0, 7, 1, addresses)
        writer = spread.runtime_arguments('writer', 0, 0, 7, 1, addresses)
        self.assertEqual((reader[-1], writer[-1]), (7, 7))
        self.assertEqual(reader[:3], [0, 0, 1])
        self.assertEqual(reader[3:16], [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 10, 11], 'the served order: x, st0-3, taps, a, b (the projection), dt_bias, neg_exp_A')
        self.assertEqual(writer[3:10], [12, 2, 3, 4, 5, 13, 14])

    def test_gates_start_at_the_cores_own_tile_and_pages_follow(self):
        reader, writer = text('gdn_conv_gates_spread_reader.cpp'), text('gdn_conv_gates_spread_writer.cpp')
        self.assertIn('for (uint32_t gi = gate_start; gi < gate_start + g_n; ++gi) {', reader)
        self.assertIn('for (uint32_t gi = gate_start; gi < gate_start + g_n; ++gi) {', writer)
        self.assertEqual(writer.count('{.page_id = gi}'), 2, 'beta and g go to page gi, which is the global tile')
        self.assertIn('const uint32_t t = gi % Nvt;', reader)
        self.assertIn('const uint32_t bt_g = gi / Nvt;', reader)

    def test_the_compile_words_end_with_rt_words_after_the_accessors(self):
        geometry = dict(channels=2560, heads=12, rows=64, x_rows=64, x_width=4120, a_col=4096, b_col=4108)
        reader = spread.compile_arguments('reader', geometry, [9] * 26)
        self.assertEqual(reader[:12], [4, 80, 1, 64, 2, 129, 1, 129, 4096, 129, 4108, 12])
        self.assertEqual(reader[-1], 17)
        self.assertEqual(len(reader), 12 + 26 + 1)
        writer = spread.compile_arguments('writer', geometry, [9] * 14)
        self.assertEqual((writer[:3], writer[-1]), ([4, 80, 1], 11))
        self.assertEqual(spread.compile_arguments('compute', geometry, []), [4, 1, 0x3F800000, 0x41A00000])

    def test_the_kernel_files_are_lf_only_and_say_what_they_are(self):
        for name in spread.SOURCES.values():
            data = (HERE / name).read_bytes()
            self.assertNotIn(b'\r', data)
            self.assertIn(b'QWEN_FAST_TP4_CONV_GATES_SPREAD', data)
        self.assertEqual(set(spread.source_sha256()), {'reader', 'writer'})

    def test_the_served_files_the_new_ones_derive_from(self):
        root = os.environ.get('GDN_CONV_ROOT') or os.environ.get('TT_METAL_HOME')
        base = Path(root) / spread.DIRECTORY if root else None
        if base is None or not (base / 'kernels/dataflow/reader_gdn_conv_gates.cpp').exists():
            self.skipTest('no tree with the served conv-gates sources (set GDN_CONV_ROOT to a tree root)')
        for name, expected in spread.SERVED.items():
            self.assertEqual(hashlib.sha256((base / name).read_bytes()).hexdigest(), expected, name)
        served_reader = (base / 'kernels/dataflow/reader_gdn_conv_gates.cpp').read_text()
        new_reader = text('gdn_conv_gates_spread_reader.cpp')
        loop_start = served_reader.index('    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {')
        loop_end = served_reader.index('    // Gather gate tile t')
        self.assertIn(served_reader[loop_start:loop_end], new_reader, 'the conv instance loop is the served one verbatim')
        served_writer = (base / 'kernels/dataflow/writer_gdn_conv_gates.cpp').read_text()
        new_writer = text('gdn_conv_gates_spread_writer.cpp')
        body = lambda source: source[source.index('#include "api/dataflow/dataflow_api.h"'):]
        left, right = body(served_writer).splitlines(), body(new_writer).splitlines()
        import difflib
        changes = [op for op in difflib.SequenceMatcher(None, left, right).get_opcodes() if op[0] != 'equal']
        self.assertEqual(len(changes), 3, changes)
        # the reader's prologue up to the loops is the served one except the two inserted lines (RT_WORDS, gate_start)
        served_head = served_reader[served_reader.index('#include "api/dataflow/dataflow_api.h"'):loop_start]
        new_head = new_reader[new_reader.index('#include "api/dataflow/dataflow_api.h"'):new_reader.index(
            '    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {')]
        changes = [op for op in difflib.SequenceMatcher(None, served_head.splitlines(), new_head.splitlines()).get_opcodes()
                   if op[0] != 'equal']
        self.assertEqual(len(changes), 2, changes)
        self.assertTrue(all(op[0] == 'insert' for op in changes))


# --- the launch, on a fake ttnn ---------------------------------------------------------------------------------------------------

class Tensor(object):
    def __init__(self, name, shape, address, memory='dram', dtype='bf16', layout='tile'):
        self.name, self.shape, self.address, self.memory, self.dtype, self.layout = name, tuple(shape), address, memory, dtype, layout

    def memory_config(self):
        return self.memory

    def buffer_address(self):
        return self.address


class Runtime(object):
    def __init__(self):
        self.table = {}

    def __getitem__(self, horizontal):
        row = self.table.setdefault(horizontal, {})
        return _Row(row)


class _Row(object):
    def __init__(self, row):
        self.row = row

    def __setitem__(self, vertical, value):
        self.row[vertical] = list(value)


class Kernel(object):
    class SourceType(object):
        SOURCE_CODE = 'source'
        FILE_PATH = 'file'

    def __init__(self, kernel_source, source_type, core_ranges, compile_time_args, config):
        self.kernel_source, self.source_type, self.core_ranges = kernel_source, source_type, core_ranges
        self.compile_time_args, self.config, self.runtime_args = list(compile_time_args), config, None


class FakeTTNN(object):
    bfloat16, float32 = 'bf16', 'fp32'
    TILE_LAYOUT = 'tile'
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = 'dram', 'l1'
    KernelDescriptor = Kernel
    MathFidelity = SimpleNamespace(HiFi4='hifi4')
    DataMovementProcessor = SimpleNamespace(RISCV_0='riscv0', RISCV_1='riscv1')
    NOC = SimpleNamespace(RISCV_0_default='noc0', RISCV_1_default='noc1')

    def __init__(self):
        self.allocated, self.freed, self.launches, self.clones = [], [], [], []
        self.address = 100000

    def empty(self, shape, dtype=None, layout=None, device=None, memory_config=None):
        tensor = Tensor('out%d' % len(self.allocated), shape, self.address, memory_config, dtype, layout)
        self.address += 4096
        self.allocated.append(tensor)
        return tensor

    def clone(self, tensor, memory_config=None):
        copy = Tensor('copy:' + tensor.name, tensor.shape, self.address, memory_config)
        self.address += 4096
        self.clones.append(copy)
        return copy

    def deallocate(self, tensor):
        self.freed.append(tensor.name)

    def get_device_tensors(self, tensor):
        return [Tensor('%s:%d' % (tensor.name, chip), tensor.shape, tensor.address + chip, tensor.memory, tensor.dtype, tensor.layout)
                for chip in range(4)]

    CoreCoord = staticmethod(lambda x, y: (x, y))
    CoreRange = staticmethod(lambda first, second: (first, second))
    CoreRangeSet = staticmethod(lambda ranges: tuple(ranges))
    Tile = staticmethod(lambda dimensions: tuple(dimensions))
    TileDescriptor = staticmethod(lambda tile: ('tile', tile))
    CBFormatDescriptor = staticmethod(lambda buffer_index, data_format, page_size, tile: (buffer_index, data_format, page_size, tile))
    CBDescriptor = staticmethod(lambda total_size, core_ranges, format_descriptors: dict(
        total_size=total_size, core_ranges=tuple(core_ranges), formats=tuple(format_descriptors)))
    TensorAccessorArgs = staticmethod(lambda tensor: SimpleNamespace(get_compile_time_args=lambda: [7, 9]))
    DataMovementConfigDescriptor = staticmethod(lambda processor, noc: ('movement', processor, noc))
    ComputeConfigDescriptor = staticmethod(lambda math_fidelity, fp32_dest_acc_en, math_approx_mode: (
        'compute', math_fidelity, fp32_dest_acc_en, math_approx_mode))
    RuntimeArgs = staticmethod(lambda: Runtime())
    MeshProgramDescriptor = staticmethod(lambda: {})
    MeshCoordinate = staticmethod(lambda row, chip: (row, chip))
    MeshCoordinateRange = staticmethod(lambda first, second: (first, second))
    ProgramDescriptor = staticmethod(lambda kernels, cbs: SimpleNamespace(kernels=list(kernels), cbs=list(cbs)))

    def generic_op(self, tensors, program):
        self.launches.append(([tensor.name for tensor in tensors], program))


def fake_mesh(x=11, y=10):
    return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=x, y=y))


SYNTHETIC = {'compute': 'compute source', 'reader': 'reader source', 'writer': 'writer source'}


class LaunchBase(object):
    def setUp(self):
        spread._NOTED.clear()
        spread._SOURCES.clear()
        self.lines = []
        patcher = patch.object(spread, 'log_line', side_effect=self.lines.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        sources = patch.object(spread, 'sources', return_value=SYNTHETIC)
        sources.start()
        self.addCleanup(sources.stop)
        self.operations = FakeTTNN()
        self.found = None
        with clean(**FOUR):
            self.found = tp_shapes.active()
        qkv = self.found.gdn_qkv
        self.x = Tensor('canon', (1, 64, self.found.gdn_qkvzab), 1000)
        self.windows = [Tensor('window%d' % slot, (1, 64, qkv), 2000 + slot) for slot in range(4)]
        self.taps = [Tensor('tap%d' % slot, (1, 1, qkv), 3000 + slot) for slot in range(4)]
        self.dt_bias = Tensor('dt', (1, 1, self.found.gdn_nv), 4000)
        self.neg = Tensor('neg', (1, 1, self.found.gdn_nv), 4001)

    def call(self, flags=LEVER, **overrides):
        arguments = dict(operations=self.operations, mesh=fake_mesh(), x=self.x, windows=self.windows, taps=self.taps,
                         dt_bias=self.dt_bias, neg_exp_A=self.neg, rows=64, channels=self.found.gdn_qkv,
                         a_col=self.found.gdn_a_col, b_col=self.found.gdn_b_col)
        arguments.update(overrides)
        with clean(**flags):
            return spread.launch(**arguments)


class LaunchTests(LaunchBase, unittest.TestCase):
    def test_one_generic_op_with_the_outputs_last(self):
        outputs = self.call()
        self.assertEqual(len(outputs), 3)
        self.assertEqual([tensor.shape for tensor in outputs], [(1, 64, 2560), (1, 64, 12), (1, 64, 12)])
        self.assertTrue(all(tensor.memory == 'dram' and tensor.dtype == 'bf16' and tensor.layout == 'tile' for tensor in outputs))
        self.assertEqual(len(self.operations.launches), 1)
        names, program = self.operations.launches[0]
        self.assertEqual(names, ['canon', 'window0', 'window1', 'window2', 'window3', 'tap0', 'tap1', 'tap2', 'tap3', 'dt', 'neg',
                                 'out0', 'out1', 'out2'], 'inputs, the windows advanced in place, then the three outputs; x once')

    def test_a_program_per_chip_with_three_kernels(self):
        self.call()
        program = self.operations.launches[0][1]
        self.assertEqual(sorted(program), [((0, chip), (0, chip)) for chip in range(4)])
        for descriptor in program.values():
            self.assertEqual([kernel.kernel_source for kernel in descriptor.kernels], ['reader source', 'writer source', 'compute source'])
            self.assertTrue(all(kernel.source_type == 'source' for kernel in descriptor.kernels))

    def test_the_cores_and_their_words(self):
        self.call()
        program = self.operations.launches[0][1]
        descriptor = program[((0, 2), (0, 2))]
        reader, writer, compute = descriptor.kernels
        found = spread.plan(11, 10, 64, 64, 2560, 12)
        self.assertEqual(found['conv_cores'], 80)
        for kernel, role in ((reader, 'reader'), (writer, 'writer'), (compute, 'compute')):
            table = kernel.runtime_args.table
            cores = sorted((x, y) for x in table for y in table[x])
            self.assertEqual(len(cores), 82)
            self.assertEqual({len(table[x][y]) for x, y in cores}, {spread.RT_WORDS[role]}, role)
        gate_cores = [(80 // 10, 80 % 10), (81 // 10, 81 % 10)]
        for index, (x, y) in enumerate(gate_cores):
            self.assertEqual(reader.runtime_args.table[x][y][:3], [0, 0, 1])
            self.assertEqual(reader.runtime_args.table[x][y][-1], index)
            self.assertEqual(writer.runtime_args.table[x][y][-1], index)
            self.assertEqual(compute.runtime_args.table[x][y], [0, 1])
        conv_core = reader.runtime_args.table[0][1]
        self.assertEqual(conv_core[:3], [2, 2, 0], 'core 1: instances 2 and 3, no gate tile')
        self.assertEqual(compute.runtime_args.table[0][1], [2, 0])

    def test_the_addresses_are_this_chips(self):
        self.call()
        program = self.operations.launches[0][1]
        for chip in range(4):
            reader = program[((0, chip), (0, chip))].kernels[0]
            words = reader.runtime_args.table[0][0]
            self.assertEqual(words[3], 1000 + chip, 'x')
            self.assertEqual(words[4:8], [2000 + slot + chip for slot in range(4)])
            self.assertEqual(words[8:12], [3000 + slot + chip for slot in range(4)])
            self.assertEqual(words[12:14], [1000 + chip, 1000 + chip], 'a and b are the projection itself')
            writer = program[((0, chip), (0, chip))].kernels[1]
            self.assertEqual(writer.runtime_args.table[0][0][3], self.operations.allocated[0].address + chip)

    def test_compile_words_and_configs(self):
        self.call()
        reader, writer, compute = self.operations.launches[0][1][((0, 0), (0, 0))].kernels
        self.assertEqual(reader.compile_time_args[:12], [4, 80, 1, 64, 2, 129, 1, 129, 4096, 129, 4108, 12])
        self.assertEqual(len(reader.compile_time_args), 12 + 13 * 2 + 1)
        self.assertEqual(reader.compile_time_args[-1], 17)
        self.assertEqual(writer.compile_time_args[:3] + writer.compile_time_args[-1:], [4, 80, 1, 11])
        self.assertEqual(len(writer.compile_time_args), 3 + 7 * 2 + 1)
        self.assertEqual(compute.compile_time_args, [4, 1, 0x3F800000, 0x41A00000])
        self.assertEqual((reader.config, writer.config), (('movement', 'riscv1', 'noc1'), ('movement', 'riscv0', 'noc0')))
        self.assertEqual(compute.config, ('compute', 'hifi4', True, False))

    def test_the_circular_buffers_are_the_factorys_with_a_four_page_scratch(self):
        self.call()
        cbs = self.operations.launches[0][1][((0, 0), (0, 0))].cbs
        plan = {cb['formats'][0][0]: (cb['total_size'], cb['formats'][0][1], cb['formats'][0][2]) for cb in cbs}
        self.assertEqual(sorted(plan), list(range(14)))
        for index in range(13):
            pages, kind = spread.CB_PLAN[index]
            page = 2048 if kind == 'bf16' else 4096
            self.assertEqual(plan[index], (pages * page, 'bf16' if kind == 'bf16' else 'fp32', page))
        self.assertEqual(plan[13], (4 * 2048, 'bf16', 2048))
        self.assertEqual({cb['core_ranges'] for cb in cbs}, {cbs[0]['core_ranges']})

    def test_the_engaged_line_once(self):
        self.call()
        self.call()
        engaged = [line for line in self.lines if line.startswith(spread.ENGAGED)]
        self.assertEqual(len(engaged), 1)
        self.assertIn('rows=64 conv_cores=80 gate_cores=2 per_core=2 served_per_core=2 audit=0', engaged[0])
        self.assertEqual(c2_smoke_check.spread_problems(dict(LEVER), '\n'.join(self.lines)), [])

    def test_a_missing_window_falls_back_with_nothing_allocated(self):
        self.windows.pop()
        self.assertIsNone(self.call())
        self.assertIn('not 4 windows and taps', self.lines[-1])
        self.assertEqual((self.operations.allocated, self.operations.launches), ([], []))

    def test_each_unfit_operand_falls_back(self):
        def change(tensor, **attributes):
            for key, value in attributes.items():
                setattr(tensor, key, value)

        for label, mutate, expected in (
                ('dtype', lambda: change(self.x, dtype='fp32'), 'not bfloat16 TILE'),
                ('layout', lambda: change(self.windows[2], layout='row'), 'not bfloat16 TILE'),
                ('memory', lambda: change(self.taps[1], memory='l1'), 'not interleaved DRAM'),
                ('narrow x', lambda: change(self.x, shape=(1, 64, 100)), 'does not hold'),
                ('window shape', lambda: change(self.windows[3], shape=(1, 64, 2592)), 'a window is not'),
                ('tap shape', lambda: change(self.taps[0], shape=(1, 1, 2592)), 'a tap is not'),
                ('heads', lambda: (change(self.x, shape=(1, 64, 6000)), change(self.dt_bias, shape=(1, 1, 33))), 'geometry'),
                ('dt shape', lambda: change(self.neg, shape=(1, 1, 24)), 'dt_bias and neg_exp_A'),
                ('x rows', lambda: change(self.x, shape=(1, 96, self.found.gdn_qkvzab)), 'more rows than the windows')):
            with self.subTest(label):
                self.setUp()
                mutate()
                self.assertIsNone(self.call())
                self.assertTrue(self.lines[-1].startswith(spread.FELL_BACK), self.lines)
                self.assertIn(expected, self.lines[-1])
                self.assertEqual((self.operations.allocated, self.operations.launches), ([], []))

    def test_the_pair_falls_back(self):
        self.assertIsNone(self.call(flags={'QWEN_FAST_TP': '2'}))
        self.assertIn('not four cards', self.lines[-1])

    def test_a_reason_is_logged_once(self):
        self.x.dtype = 'fp32'
        self.call()
        self.call()
        self.assertEqual(len([line for line in self.lines if line.startswith(spread.FELL_BACK)]), 1)

    def test_a_tree_that_is_not_the_pinned_one_falls_back(self):
        with patch.object(spread, 'sources', side_effect=spread.SourceMismatch('the conv-gates source x is not the pinned one')):
            self.assertIsNone(self.call())
        self.assertIn('not the pinned one', self.lines[-1])
        self.assertEqual(self.operations.allocated, [])

    def test_a_grid_without_room_falls_back(self):
        with clean(**LEVER):
            self.assertIsNone(spread.launch(self.operations, fake_mesh(1, 2), self.x, self.windows, self.taps, self.dt_bias, self.neg,
                                            64, 2560, 4096, 4108))
        self.assertIn('cannot hold', self.lines[-1])

    def test_a_failure_in_the_launch_frees_the_outputs(self):
        def refuse(tensors, program):
            raise RuntimeError('device')

        self.operations.generic_op = refuse
        with self.assertRaises(RuntimeError):
            self.call()
        self.assertEqual(sorted(self.operations.freed), sorted(tensor.name for tensor in self.operations.allocated))


class SourcesTests(unittest.TestCase):
    def setUp(self):
        spread._SOURCES.clear()
        self.addCleanup(spread._SOURCES.clear)

    def tree(self, directory, change=None):
        pins = {}
        for name in spread.SERVED:
            path = Path(directory) / spread.DIRECTORY / name
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = ('// synthetic %s\n' % name).encode()
            path.write_bytes(payload)
            pins[name] = hashlib.sha256(payload).hexdigest()
        if change:
            (Path(directory) / spread.DIRECTORY / change).write_bytes(b'// changed\n')
        return pins

    def test_the_pinned_tree_gives_the_served_compute_and_our_two_kernels(self):
        with tempfile.TemporaryDirectory() as directory:
            pins = self.tree(directory)
            with patch.object(spread, 'SERVED', pins):
                found = spread.sources(directory)
        self.assertEqual(found['compute'], '// synthetic %s\n' % spread.COMPUTE_FILE)
        self.assertEqual(found['reader'], text('gdn_conv_gates_spread_reader.cpp'))
        self.assertEqual(found['writer'], text('gdn_conv_gates_spread_writer.cpp'))

    def test_any_changed_served_file_is_refused(self):
        for name in spread.SERVED:
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                spread._SOURCES.clear()
                pins = self.tree(directory, change=name)
                with patch.object(spread, 'SERVED', pins):
                    self.assertRaises(spread.SourceMismatch, spread.sources, directory)

    def test_a_missing_file_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertRaises(spread.SourceMismatch, spread.sources, directory)

    def test_the_pins_are_the_direct_window_pins(self):
        import gdn_direct_window_device as direct

        self.assertEqual(spread.SERVED, direct.HASHES)
        self.assertEqual(spread.DIRECTORY, direct.DIRECTORY)


# --- the audit ---------------------------------------------------------------------------------------------------------------------

class AuditLaunchTests(LaunchBase, unittest.TestCase):
    def call(self, flags=AUDITED, **overrides):
        served_calls = []

        def conv_gates(operations, projected, windows, taps, dt_bias, neg_exp_A, rows):
            served_calls.append((projected.name, [window.name for window in windows], rows))
            return (Tensor('served-conv', (1, rows, 2560), 7000), Tensor('served-beta', (1, rows, 12), 7001),
                    Tensor('served-g', (1, rows, 12), 7002))

        self.served_calls = served_calls
        with patch('gdn_user_batch_conv.conv_gates', side_effect=conv_gates):
            return super().call(flags=flags, **overrides)

    def test_the_served_op_runs_first_on_clones_and_seven_entries_are_held(self):
        held, entries = [], []
        outputs = self.call(held=held, entries=entries)
        self.assertEqual(self.served_calls, [('canon', ['copy:window%d' % slot for slot in range(4)], 64)])
        self.assertEqual([entry['label'] for entry in entries], [spread.LABEL + name for name in (
            'conv', 'beta', 'g', 'window 0', 'window 1', 'window 2', 'window 3')])
        self.assertEqual(len(entries), spread.ENTRIES_PER_LAUNCH)
        self.assertEqual([entry['mine'].name for entry in entries[:3]], ['copy:out0', 'copy:out1', 'copy:out2'])
        self.assertEqual([entry['mine'].name for entry in entries[3:]], ['copy:window%d' % slot for slot in range(4)])
        self.assertEqual([entry['served'].name for entry in entries[:3]], ['served-conv', 'served-beta', 'served-g'])
        self.assertEqual([entry['served'].name for entry in entries[3:]], ['copy:window%d' % slot for slot in range(4)])
        self.assertEqual(len(held), 2 * len(entries))
        self.assertTrue(all(entry['mine'] is not entry['served'] for entry in entries))
        # the served op ran on windows cloned BEFORE the launch (the clones are the first allocations), and the launch advanced the real ones
        self.assertEqual([copy.name for copy in self.operations.clones[:4]], ['copy:window%d' % slot for slot in range(4)])
        self.assertEqual(self.operations.launches[0][0][1:5], ['window0', 'window1', 'window2', 'window3'])
        self.assertIn('audit=1', [line for line in self.lines if line.startswith(spread.ENGAGED)][0])

    def test_a_failure_after_the_served_op_frees_everything_made(self):
        def refuse(tensors, program):
            raise RuntimeError('device')

        self.operations.generic_op = refuse
        held, entries = [], []
        with self.assertRaises(RuntimeError):
            self.call(held=held, entries=entries)
        self.assertEqual((held, entries), ([], []))
        made = {tensor.name for tensor in self.operations.allocated} | {copy.name for copy in self.operations.clones}
        self.assertTrue(made <= set(self.operations.freed), (made, self.operations.freed))
        self.assertIn('served-conv', self.operations.freed)

    def test_a_fall_back_runs_no_served_op_and_makes_no_entries(self):
        self.x.dtype = 'fp32'
        entries = []
        self.assertIsNone(self.call(entries=entries))
        self.assertEqual((self.served_calls, entries, self.operations.clones), ([], [], []))


class RoundAuditTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        patcher = patch.object(spread, 'log_line', side_effect=self.lines.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        spread._AUDIT['rounds'] = 0

    def entries(self, count=spread.ENTRIES_PER_LAUNCH, label=spread.LABEL):
        return [dict(label=label + 'x%d' % index, mine='m%d' % index, served='s%d' % index) for index in range(count)]

    def records(self, per_layer):
        return [(None, {'vglue_block_audit': per_layer(layer)}, None) for layer in range(48)]

    def run_round(self, per_layer, compare=None):
        import verify_trace_t2

        with patch('tp4_vglue.compare_entry', side_effect=compare or (lambda operations, entry: [])):
            return spread.audit_round('ops', self.records(per_layer), 1)

    def test_every_audited_layer_must_carry_the_seven_and_they_compare(self):
        import verify_trace_t2

        layers = verify_trace_t2.audit_layers(1, 48)
        found = self.run_round(lambda layer: self.entries())
        self.assertEqual(found, len(layers) * spread.ENTRIES_PER_LAUNCH)
        self.assertTrue(self.lines[-1].startswith(spread.AUDIT_MARKER + ' 1 exact=True layers='), self.lines)

    def test_two_blocks_in_a_layer_are_two_launches(self):
        found = self.run_round(lambda layer: self.entries(2 * spread.ENTRIES_PER_LAUNCH))
        self.assertGreater(found, 0)

    def test_a_layer_without_the_entries_fails_the_audit(self):
        with self.assertRaises(AssertionError):
            self.run_round(lambda layer: [])
        self.assertIn(spread.AUDIT_MISMATCH, self.lines[-1])
        self.assertIn('spread entries', self.lines[-1])

    def test_a_partial_set_fails(self):
        with self.assertRaises(AssertionError):
            self.run_round(lambda layer: self.entries(spread.ENTRIES_PER_LAUNCH - 1))

    def test_other_entries_are_not_counted(self):
        with self.assertRaises(AssertionError):
            self.run_round(lambda layer: self.entries(label='block conv user 0 '))

    def test_a_difference_logs_the_mismatch_marker_and_raises(self):
        with self.assertRaises(AssertionError):
            self.run_round(lambda layer: self.entries(), compare=lambda operations, entry: ['%s chip 2: 3 of 64 elements differ' % entry['label']])
        self.assertTrue(self.lines[-1].startswith(spread.AUDIT_MISMATCH))
        self.assertIn('chip 2', self.lines[-1])

    def test_the_vglue_audit_does_not_count_these_entries_as_its_own(self):
        result = {'vglue_block_audit': self.entries()}
        with clean(**LEVER):
            counted = [entry['label'] for entry in tp4_vglue.audit_entries(result) if entry['label'].startswith('block conv user ')]
            missing = tp4_vglue.missing_entries(result)
        self.assertEqual(counted, [])
        self.assertIn('block conv user 0: 0 entries, expected 8', missing, 'the F1 entries are not not counted as the V1 stage own')


# --- the V1 stage ---------------------------------------------------------------------------------------------------------------

class StageTests(unittest.TestCase):
    """The V1 stage with the lever off makes the served call and never imports the module; with it on, makes the launch's call."""

    def setUp(self):
        import test_tp4_vglue_block as base

        self.base = base
        self.case = base.StageTests('test_three_launches_and_one_op_call_at_batch_64_on_the_canon_block')
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def test_off_it_is_the_served_call_and_the_module_is_not_imported(self):
        sys.modules.pop('gdn_conv_gates_spread', None)
        try:
            found, fallbacks = self.case.run_stage()
            self.assertNotIn('gdn_conv_gates_spread', sys.modules)
        finally:
            sys.modules['gdn_conv_gates_spread'] = spread
        self.assertEqual(self.case.conv_calls, [('empty0', ['empty1', 'empty2', 'empty3', 'empty4'], 64)])
        self.assertEqual(found.entries, [])

    def test_on_the_launch_replaces_the_served_call(self):
        calls = []

        def launch(operations, mesh, x, windows, taps, dt_bias, neg_exp_A, rows, channels, a_col, b_col, held=None, entries=None):
            calls.append((x.name, [window.name for window in windows], rows, channels, a_col, b_col))
            return (self.base.Tensor('conv', (1, rows, 2560)), self.base.Tensor('beta', (1, rows, 12)), self.base.Tensor('gate', (1, rows, 12)))

        with patch.object(spread, 'launch', side_effect=launch):
            found, fallbacks = self.case.run_stage(**LEVER)
        self.assertEqual(self.case.conv_calls, [])
        found_shapes = self.case.found
        self.assertEqual(calls, [('empty0', ['empty1', 'empty2', 'empty3', 'empty4'], 64, found_shapes.gdn_qkv, found_shapes.gdn_a_col,
                                  found_shapes.gdn_b_col)])
        self.assertEqual({tensor.name for tensor in found.owned}, {'empty0', 'empty1', 'empty2', 'empty3', 'empty4', 'conv', 'beta', 'gate'})

    def test_on_but_declined_it_is_the_served_call(self):
        with patch.object(spread, 'launch', return_value=None):
            found, fallbacks = self.case.run_stage(**LEVER)
        self.assertEqual(self.case.conv_calls, [('empty0', ['empty1', 'empty2', 'empty3', 'empty4'], 64)])
        self.assertEqual(fallbacks, [])

    def test_a_malformed_flag_raises_at_the_stage(self):
        with self.assertRaises(ValueError):
            self.case.run_stage(**dict(LEVER, **{spread.FLAG: '2'}))

    def test_the_audit_entries_ride_the_first_users_list(self):
        def launch(operations, mesh, x, windows, taps, dt_bias, neg_exp_A, rows, channels, a_col, b_col, held=None, entries=None):
            entries.extend(dict(label=spread.LABEL + str(index), mine=self.base.Tensor('m%d' % index, (1,)),
                                served=self.base.Tensor('s%d' % index, (1,))) for index in range(7))
            return (self.base.Tensor('conv', (1, rows, 2560)), self.base.Tensor('beta', (1, rows, 12)), self.base.Tensor('gate', (1, rows, 12)))

        with patch.object(spread, 'launch', side_effect=launch), patch('gdn_block_conv_tp.hold_served', return_value=[[], [], [], []]):
            found, fallbacks = self.case.run_stage(**AUDITED)
        self.assertEqual([len(user) for user in found.entries], [7, 0, 0, 0])


# --- the smoke rule -----------------------------------------------------------------------------------------------------------------

ENGAGED_LINE = spread.ENGAGED + ' rows=64 conv_cores=80 gate_cores=2 per_core=2 served_per_core=2 audit=0'
AUDIT_LINE = spread.AUDIT_MARKER + ' 3 exact=True layers=0,24 entries=14'


class SmokeRuleTests(unittest.TestCase):
    def test_without_the_flag_nothing_is_required_and_lines_are_refused(self):
        self.assertEqual(c2_smoke_check.spread_problems({}, ''), [])
        self.assertEqual(c2_smoke_check.spread_problems(None, ''), [])
        self.assertEqual(len(c2_smoke_check.spread_problems({}, ENGAGED_LINE)), 1)

    def test_the_flag_needs_the_engaged_line_with_a_gate_core(self):
        env_ = {spread.FLAG: '1'}
        self.assertEqual(len(c2_smoke_check.spread_problems(env_, '')), 1)
        self.assertEqual(c2_smoke_check.spread_problems(env_, ENGAGED_LINE), [])
        none = ENGAGED_LINE.replace('gate_cores=2', 'gate_cores=0')
        self.assertEqual(len(c2_smoke_check.spread_problems(env_, none)), 1)

    def test_a_fall_back_line_fails_with_or_without_the_flag(self):
        line = spread.FELL_BACK + ' reason=an operand is not bfloat16 TILE'
        self.assertEqual(len(c2_smoke_check.spread_problems({spread.FLAG: '1'}, ENGAGED_LINE + '\n' + line)), 1)
        self.assertEqual(len(c2_smoke_check.spread_problems({}, line)), 1)

    def test_the_audit_flag_needs_a_passing_line(self):
        env_ = {spread.FLAG: '1', spread.AUDIT_FLAG: '1'}
        self.assertEqual(len(c2_smoke_check.spread_problems(env_, ENGAGED_LINE)), 1)
        self.assertEqual(c2_smoke_check.spread_problems(env_, ENGAGED_LINE + '\n' + AUDIT_LINE), [])

    def test_a_mismatch_line_fails_even_beside_a_passing_one(self):
        env_ = {spread.FLAG: '1', spread.AUDIT_FLAG: '1'}
        mismatch = spread.AUDIT_MISMATCH + ' round=2 layers=0 layer 0 spread conv gates conv chip 1: 2 of 64 elements differ'
        problems = c2_smoke_check.spread_problems(env_, '\n'.join([ENGAGED_LINE, AUDIT_LINE, mismatch]))
        self.assertEqual(len(problems), 1)
        self.assertIn('found a difference', problems[0])

    def test_a_mismatch_line_is_not_a_passing_audit_line(self):
        env_ = {spread.FLAG: '1', spread.AUDIT_FLAG: '1'}
        mismatch = spread.AUDIT_MISMATCH + ' round=2 exact=True'
        problems = c2_smoke_check.spread_problems(env_, '\n'.join([ENGAGED_LINE, mismatch]))
        self.assertEqual(len(problems), 2)


# --- shipment -------------------------------------------------------------------------------------------------------------------

class ShipmentTests(unittest.TestCase):
    def test_the_module_and_its_kernels_are_in_all_three_copy_lists(self):
        workflow = text('../../.github/workflows/qwen-fast-serving-image.yml')
        dockerfile = text('../../docker/qwen-fast-serving.Dockerfile')
        overlay = text('../../docker/qwen-c2-overlay.txt')
        for name in spread.RUNTIME_FILES:
            self.assertIn(name, workflow, name)
            self.assertIn(' scripts/ci/' + name, dockerfile, name)
            self.assertIn('scripts/ci/' + name, overlay.splitlines(), name)
        self.assertEqual(spread.RUNTIME_FILES, ('gdn_conv_gates_spread.py', 'gdn_conv_gates_spread_reader.cpp', 'gdn_conv_gates_spread_writer.cpp'))

    def test_this_test_is_on_the_cpu_allowlist(self):
        self.assertIn('test_gdn_conv_gates_spread', text('../../.github/workflows/qwen-integration-cpu.yml'))


if __name__ == '__main__':
    unittest.main()
