"""WC (gdn_wc, QWEN_FAST_TP4_GDN_WC): the core plan, the kernels' byte movement emulated, and the proof on tiles that the one launch equals what it replaces.

Three layers, each failing if WC differs from the chain it replaces:

  * the plan: the cores, the conv/gate/z work and the pages every core reads and writes, held as invariants (each output page written exactly
    once, nothing outside the grid, fixed runtime-argument lengths, the circular buffers the kernels use);
  * the data: gdn_wc_emulate runs the kernels' loops (transcribed over face-order tile vectors) around a stand-in for the served compute kernel's
    contract, and the result is compared, bit for bit (-0, denormals, infinities and NaN payloads included), with (a) the four SERVED per-user calls
    (the pieces the served path cuts, the T2 windows the served window kernel builds, one conv_gates contract call each) and (b) the V1 chain's own planners
    (canon block, window stack, the unstack) run through the existing row-mover emulator;
  * the text of the kernels: the compile and runtime argument indices the kernels read are the module's, the loops are the transcribed ones.

What no CPU test shows: that the compiled kernels move those bytes. That is the card-M byte gate (optimisation/ttnn-op/gdn_wc) and the in-trace audit.

Run at py 3.11 from scripts/ci: `python -B -m unittest test_gdn_wc`.
"""

import ast
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import gdn_conv_windows_packed as windows_packed
import gdn_rows_dma_tp as rows_dma
import gdn_wc
import gdn_wc_emulate as emulate
import tp_shapes
from test_tp4_vglue_block import OpModel, finite, from_tiles, geometry, to_bits, to_float
from test_tp4_vglue_block import DataTests as BlockData
from test_tp4_vglue_gdn import canon, specials, to_tiles

HERE = Path(__file__).resolve().parent
GRID = (11, 10)


def four():
    return patch.dict(os.environ, {'QWEN_FAST_TP': '4'})


def text(name):
    return (HERE / name).read_text(encoding='utf-8')


# --- the plan ----------------------------------------------------------------------------------------------------------------------

class PlanTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        self.found = tp_shapes.active()

    def test_four_users_are_eighty_conv_cores_two_gate_cores_and_the_rest_z(self):
        found = gdn_wc.plan(*GRID, 4)
        self.assertEqual((found['conv_cores'], found['gate_cores'], found['z_cores'], found['cores']), (80, 2, 28, 110))
        self.assertEqual((found['per_core'], found['served_per_core']), (2, 2))
        shape = found['shape']
        self.assertEqual((shape['instances'], shape['gate_tiles'], shape['z_tiles'], shape['width_tiles']), (160, 2, 96, 129))
        self.assertEqual(sorted(item['n_z'] for item in found['work'] if item['n_z']), [3] * 16 + [4] * 12)

    def test_two_users_fill_a_smaller_plan(self):
        found = gdn_wc.plan(*GRID, 2)
        self.assertEqual((found['conv_cores'], found['gate_cores'], found['z_cores']), (80, 1, 29))
        self.assertEqual(found['shape']['z_tiles'], 48)
        self.assertEqual(sum(item['n_z'] for item in found['work']), 48)

    def test_an_odd_or_single_user_count_is_refused(self):
        for users in (1, 3, 5):
            with self.assertRaises(ValueError):
                gdn_wc.plan(*GRID, users)

    def test_a_grid_that_cannot_hold_conv_gate_and_z_is_refused(self):
        with self.assertRaises(ValueError):
            gdn_wc.plan(2, 2, 4)                  # no room for the gate tiles beside the conv instances
        with self.assertRaisesRegex(ValueError, 'no core is left'):
            gdn_wc.plan(82, 1, 4)                 # 80 conv cores and 2 gate cores fill it: the z tiles have nowhere to go

    def expected_outputs(self, users):
        found = self.found
        expected = set()
        for user in range(users):
            for page in range(80):
                expected.add(('conv', user, 0, page))
                for slot in range(4):
                    expected.add(('window', user, slot, page))
            expected.add(('beta', user, 0, 0))
            expected.add(('g', user, 0, 0))
            for page in range(48):
                expected.add(('z', user, 0, page))
        return expected

    def test_every_output_page_is_written_exactly_once(self):
        for users in (2, 4):
            found = gdn_wc.plan(*GRID, users)
            written = [page for item in found['work'] for page in gdn_wc.output_pages(item, found['shape'])]
            self.assertEqual(len(written), len(set(written)), 'a page is written twice')
            self.assertEqual(set(written), self.expected_outputs(users))

    def test_cores_are_distinct_and_inside_the_grid(self):
        found = gdn_wc.plan(*GRID, 4)
        points = [item['point'] for item in found['work']]
        self.assertEqual(len(points), len(set(points)))
        self.assertTrue(all(0 <= x < GRID[0] and 0 <= y < GRID[1] for x, y in points))

    def test_each_core_has_one_role_and_the_conv_instances_are_the_served_partition(self):
        found = gdn_wc.plan(*GRID, 4)
        starts = [(item['start'], item['n_inst']) for item in found['work'] if item['n_inst']]
        self.assertEqual(starts, [(2 * index, 2) for index in range(80)])
        for item in found['work']:
            self.assertEqual(sum(1 for key in ('n_inst', 'g_n', 'n_z') if item[key]), 1)

    def test_every_projection_page_the_work_needs_is_read(self):
        found = gdn_wc.plan(*GRID, 4)
        shape = found['shape']
        pages = {page for item in found['work'] for page in gdn_wc.input_pages(item, shape)}
        wanted = {bt * 129 + column for bt in range(2) for column in list(range(80)) + list(range(80, 128)) + [128]}
        self.assertEqual(pages, wanted, 'conv tiles, z tiles and the a/b tile of both tile rows')

    def test_runtime_words_have_the_same_length_on_every_core(self):
        found = gdn_wc.plan(*GRID, 4)
        for role in gdn_wc.ROLES:
            self.assertEqual({len(gdn_wc.runtime_arguments(role, item)) for item in found['work']}, {gdn_wc.RT_WORDS[role]})

    def test_the_common_words_are_fixed_length_whatever_the_user_count(self):
        addresses = dict(x=1, history=[[10 + 10 * u + i for i in range(4)] for u in range(4)], tap=[50, 51, 52, 53], dt_bias=60, neg_exp_A=61)
        out = dict(conv=[100 + u for u in range(4)], beta=[110 + u for u in range(4)], g=[120 + u for u in range(4)],
                   z=[130 + u for u in range(4)], window=[[200 + 10 * u + s for s in range(4)] for u in range(4)])
        for users in (2, 4):
            reader, writer = gdn_wc.reader_common(addresses, users), gdn_wc.writer_common(out, users)
            self.assertEqual((len(reader), len(writer)), (gdn_wc.READER_COMMON_WORDS, gdn_wc.WRITER_COMMON_WORDS))
        reader, writer = gdn_wc.reader_common(addresses, 4), gdn_wc.writer_common(out, 4)
        self.assertEqual(reader[:4], [1, 11, 12, 13], 'x, then history 1..3 of user 0')
        self.assertEqual(reader[1 + 3 * 3:1 + 3 * 4], [41, 42, 43], 'history 1..3 of user 3')
        self.assertEqual(reader[13:], [50, 51, 52, 53, 60, 61])
        self.assertEqual(writer[:4] + writer[12:16] + writer[16:20], [100, 101, 102, 103, 130, 131, 132, 133, 200, 201, 202, 203])
        self.assertEqual(writer[16 + 4 * 3 + 2], 232, 'window slot 2 of user 3')
        self.assertEqual(gdn_wc.reader_common(addresses, 2)[1 + 3 * 2:1 + 3 * 4], [0] * 6, 'past the users: zero')

    def test_the_circular_buffers_fit_with_room_to_spare(self):
        self.assertEqual(gdn_wc.cb_bytes(), 143360)
        self.assertLess(gdn_wc.cb_bytes(), 200 * 1024)
        self.assertEqual({index for index in gdn_wc.CB_PLAN} - set(gdn_wc.spread.CB_PLAN), {14, 15, 16})
        self.assertEqual(gdn_wc.CB_PLAN[16][0], gdn_wc.USTAGE_PAGES + 1, 'ten user pages and one zero source page')

    def test_the_compile_words_end_with_rt_words_after_the_accessors(self):
        shape = gdn_wc.plan(*GRID, 4)['shape']
        reader = gdn_wc.compile_arguments('reader', shape, [7, 9, 7, 9, 7, 9])
        self.assertEqual(reader[:10], [4, 80, 1, 4, 129, 4096, 4108, 12, 80, 48])
        self.assertEqual(reader[10:], [7, 9, 7, 9, 7, 9, gdn_wc.RT_WORDS['reader']])
        writer = gdn_wc.compile_arguments('writer', shape, [7, 9])
        self.assertEqual(writer, [4, 80, 1, 4, 48, 7, 9, gdn_wc.RT_WORDS['writer']])
        self.assertEqual(gdn_wc.compile_arguments('compute', shape, []), [4, 1, gdn_wc.spread.ONE_BITS, gdn_wc.spread.TWENTY_BITS])


# --- the data ----------------------------------------------------------------------------------------------------------------------

def pad_tile(vector_rows):
    """A (1, 2560)-shaped state row as the (32, 32) tiles of one history page set: row 0 the data, rows 1-31 garbage (a TILE tensor's padding
    holds whatever the allocation did; the kernels must read row 0 only)."""
    return vector_rows


class Chain(object):
    """One random layer: a block projection, per user four history tiles per channel page, taps and gate parameters; and the three computations
    over them (the served per-user calls, the V1 chain through the row-mover emulator, WC through the emulated kernels)."""

    def __init__(self, seed, users=4, edges=True):
        self.found = geometry()
        found = self.found
        self.users = users
        self.rows = users * 16
        self.generator = torch.Generator().manual_seed(seed)
        make = (lambda shape: specials(self.generator, shape)) if edges else (
            lambda shape: finite(torch.randint(-32768, 32768, shape, generator=self.generator, dtype=torch.int16)))
        self.columns = rows_dma.tile_columns(found.gdn_qkvzab)
        self.block = make((self.rows, self.columns * 32))
        self.block[:, found.gdn_qkvzab:] = 0
        if not edges:
            self.block = finite(self.block)
        # history[user][index]: (1, 2560) rows in (32, 2560) tensors whose rows 1-31 are garbage
        self.history = []
        for user in range(users):
            row = []
            for index in range(4):
                state = make((32, found.gdn_qkv))
                row.append(state)
            self.history.append(row)
        self.op = OpModel(self.generator)
        # the tiles the device holds are bf16: round the float parameters once, both ways use these
        self.op.taps = to_float(to_bits(self.op.taps))
        self.op.dt_bias = to_float(to_bits(self.op.dt_bias))
        self.op.neg_exp_A = to_float(to_bits(self.op.neg_exp_A))

    # -- pieces and the T2 windows, the served way --
    def piece(self, user):
        rows = self.block[16 * user:16 * user + 16]
        return canon(rows) if user % 2 else rows

    def t2_windows(self, user):
        """The four (16, 2560) windows the served window kernel builds (gdn_conv_windows_packed.reference_windows, the served loop transcribed)."""
        found = self.found
        piece_pages = torch.stack([emulate.tile_to_faces(torch.cat([self.piece(user), torch.zeros(16, self.columns * 32, dtype=torch.int16)])
                                                         [:, page * 32:(page + 1) * 32]) for page in range(80)])
        history_pages = [torch.stack([emulate.tile_to_faces(self.history[user][index][:, page * 32:(page + 1) * 32]) for page in range(80)])
                         for index in range(4)]
        built = windows_packed.reference_windows(piece_pages, history_pages)
        return [torch.cat([emulate.faces_to_tile(window[page])[:16] for page in range(80)], dim=1) for window in built]

    def served(self):
        out = []
        for user in range(self.users):
            piece = self.piece(user)
            conv, beta, g, advanced = self.op(piece, self.t2_windows(user), 16)
            out.append(dict(conv=conv, beta=beta, g=g, windows=advanced, z=piece[:, self.found.gdn_qkv:self.found.gdn_a_col]))
        return out

    # -- the V1 chain through the row-mover emulator --
    def v1(self):
        from test_tp4_vglue_gdn import execute
        import gdn_block_conv_tp as block

        found, columns = self.found, self.columns
        (canon_tiles,) = execute(rows_dma.canon_block(self.users, found.gdn_qkvzab), [to_tiles(self.block)], [{}])
        canon_block = from_tiles(canon_tiles, self.rows, columns * 32)
        flat = [to_tiles(window) for user in range(self.users) for window in self.t2_windows(user)]
        stacked = execute(block.plan_windows(self.users), flat, [{} for slot in range(4)])
        block_windows = [from_tiles(tiles, self.rows, found.gdn_qkv) for tiles in stacked]
        conv, beta, g, advanced = self.op(canon_block, block_windows, self.rows)
        sources = [to_tiles(conv), to_tiles(beta), to_tiles(g), canon_tiles] + [to_tiles(window) for window in advanced]
        return execute(block.plan_unstack(self.users, found.gdn_qkvzab), sources, [{} for index in range(8 * self.users)])

    # -- WC --
    def tiles_of(self, matrix):
        return {page: emulate.tile_to_faces(tile) for page, tile in to_tiles(matrix).items()}

    def wc(self, denorm=True, work_plan=None):
        found = self.found
        work_plan = work_plan or gdn_wc.plan(*GRID, self.users)
        x = self.tiles_of(self.block)
        history = [[{page: emulate.tile_to_faces(state[:, page * 32:(page + 1) * 32]) for page in range(80)} for state in row]
                   for row in self.history]
        taps = [{page: emulate.tile_to_faces(self.row_tile(self.op.taps[j][page * 32:(page + 1) * 32])) for page in range(80)} for j in range(4)]
        dt = {0: emulate.tile_to_faces(self.row_tile(self.op.dt_bias, 32))}
        nega = {0: emulate.tile_to_faces(self.row_tile(self.op.neg_exp_A, 32))}
        return emulate.emulate(work_plan, x, history, taps, dt, nega, emulate.Compute(), denorm), work_plan

    @staticmethod
    def row_tile(values, width=None):
        tile = torch.zeros(32, 32, dtype=torch.int16)
        data = to_bits(values)
        tile[0, :data.numel()] = data
        return tile


def matrix(written, name, user, slot, pages, rows=16):
    columns = []
    for page in pages:
        tile = emulate.faces_to_tile(written[(name, user, slot, page)])
        columns.append(tile[:rows])
    return torch.cat(columns, dim=1)


class DataTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_equal_to_served(self, chain, written):
        found = chain.found
        expected = chain.served()
        for user in range(chain.users):
            self.assertTrue(torch.equal(matrix(written, 'conv', user, 0, range(80)), expected[user]['conv']), 'conv user %d' % user)
            self.assertTrue(torch.equal(matrix(written, 'beta', user, 0, [0])[:, :found.gdn_nv], expected[user]['beta']), 'beta user %d' % user)
            self.assertTrue(torch.equal(matrix(written, 'g', user, 0, [0])[:, :found.gdn_nv], expected[user]['g']), 'g user %d' % user)
            self.assertTrue(torch.equal(matrix(written, 'z', user, 0, range(48)), expected[user]['z']), 'z user %d' % user)
            for slot in range(4):
                self.assertTrue(torch.equal(matrix(written, 'window', user, slot, range(80)), expected[user]['windows'][slot]),
                                'window %d user %d' % (slot, user))

    def test_wc_equals_the_four_served_calls_on_edge_values(self):
        for seed in (3, 4):
            chain = Chain(seed)
            written, work_plan = chain.wc()
            self.assert_equal_to_served(chain, written)

    def test_wc_equals_the_four_served_calls_on_ordinary_values(self):
        chain = Chain(5, edges=False)
        written, work_plan = chain.wc()
        self.assert_equal_to_served(chain, written)

    def test_two_users(self):
        chain = Chain(6, users=2)
        written, work_plan = chain.wc()
        self.assert_equal_to_served(chain, written)

    def test_wc_equals_the_v1_chain_tile_for_tile(self):
        """The V1 chain's destinations (its unstack) against WC's writes, page by page, padding included: half 1 of every page is zero in both."""
        chain = Chain(7)
        written, work_plan = chain.wc()
        built = chain.v1()
        counts = {'conv': 0, 'beta': 4, 'g': 8, 'z': 12}
        for (name, user, slot, page), vector in written.items():
            index = 16 + user * 4 + slot if name == 'window' else counts[name] + user
            tile = emulate.faces_to_tile(vector)
            if name in ('beta', 'g'):
                # The served compute runs on the whole gate tile: its columns past the real heads hold sigmoid(0) and the like in the device's tile,
                # which the row-parallel stand-ins model differently (zero in the V1 emulation). Both chains copy whatever the compute made, so only
                # the real heads' columns and the zero half are comparable here.
                self.assertTrue(torch.equal(tile[:, :chain.found.gdn_nv], built[index][page][:, :chain.found.gdn_nv]), (name, user, slot, page))
                self.assertFalse(bool(tile[16:].any()))
                continue
            self.assertTrue(torch.equal(tile, built[index][page]), (name, user, slot, page))
        self.assertEqual(len(written), sum(len(tiles) for tiles in built))

    def test_the_second_users_half_is_canonical_and_the_first_is_raw(self):
        chain = Chain(8)
        chain.block[:16, :2560] = -32768          # user 0: raw -0 everywhere in the conv channels
        chain.block[16:32, :2560] = -32768        # user 1: the canonical rule turns it to +0
        written, work_plan = chain.wc()
        shift_3 = {user: matrix(written, 'window', user, 3, range(80)) for user in range(4)}
        self.assertTrue(bool((shift_3[0] == -32768).all()), 'user 0 stays raw -0 (it passes through the served compute)')
        self.assertFalse(bool(shift_3[1].any()), 'user 1 is canonical +0')

    def test_a_missing_canonical_rule_is_caught(self):
        """The negative control: the same comparison fails when the reader forgets the canonical rule."""
        chain = Chain(9)
        with patch.object(emulate, 'canonical', side_effect=lambda vector, denorm=True: vector):
            written, work_plan = chain.wc()
        with self.assertRaises(AssertionError):
            self.assert_equal_to_served(chain, written)

    def test_a_wrong_history_row_is_caught(self):
        chain = Chain(10)
        written, work_plan = chain.wc()
        chain.history[0][2] = chain.history[0][2].clone()
        chain.history[0][2][0, :64] ^= 0x1111
        with self.assertRaises(AssertionError):
            self.assert_equal_to_served(chain, written)

    def test_the_history_padding_rows_are_never_read(self):
        chain = Chain(11)
        reference, work_plan = chain.wc()
        for row in chain.history:
            for state in row:
                state[1:] = state[1:] ^ 0x5A5A
        again, work_plan = chain.wc()
        for key in reference:
            self.assertTrue(torch.equal(reference[key], again[key]), key)

    def test_history_zero_is_never_read(self):
        """Slot 0 of the window is not built (the conv never reads it): the history it would use does not matter."""
        chain = Chain(12)
        reference, work_plan = chain.wc()
        for row in chain.history:
            row[0] = row[0] ^ 0x3333
        again, work_plan = chain.wc()
        for key in reference:
            self.assertTrue(torch.equal(reference[key], again[key]), key)

    def test_every_page_has_a_zero_second_half(self):
        chain = Chain(13)
        written, work_plan = chain.wc()
        for key, vector in written.items():
            self.assertFalse(bool(vector[emulate.HALF:].any()), key)


class KernelLoopTests(unittest.TestCase):
    """The loops the kernel text holds, transcribed as the emulation's formulas: the window row map is T2's window_copies."""

    def test_the_emulated_window_rows_are_window_copies(self):
        for slot in range(1, 4):
            copies = windows_packed.window_copies(slot)
            hist = [copy for copy in copies if copy[0] == 'hist']
            piece = [copy for copy in copies if copy[0] == 'piece']
            self.assertEqual([copy[2] for copy in hist], list(range(4 - slot)), 'history rows land in rows 0..3-s')
            self.assertEqual([copy[1] for copy in hist], [row + slot for row in range(4 - slot)], 'from history r + s')
            self.assertEqual(piece, [('piece', 0, 4 - slot, (12 + slot) * 32)], 'one block of 12 + s piece rows from row 4 - s')

    def test_history_zero_is_the_only_history_the_conv_does_not_use(self):
        used = {copy[1] for slot in range(1, 4) for copy in windows_packed.window_copies(slot) if copy[0] == 'hist'}
        self.assertEqual(used, {1, 2, 3})


if __name__ == '__main__':
    unittest.main()
