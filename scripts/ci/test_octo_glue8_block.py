"""QWEN_FAST_OCTO_GLUE8, site 'block_conv': V1 (QWEN_FAST_TP4_GDN_BLOCK_CONV) at the octo block's eight-row users: one conv-gates launch per layer against eight.

The same three layers of proof as the sixteen-row V1 (test_tp4_vglue_block), each failing if the block path differs from the per-user path it replaces:

  * data: the three launches' plans (canon block, window stack, unstack), run through the quarter-tile mover emulator (raw 2048-byte tiles) around an emulation of
    gdn_decode_conv_gates' contract (row-parallel FIR + silu + gates, windows advanced in place), equal the eight served per-user calls at batch 8 on every output and every
    advanced window, bit for bit, -0 and denormals included, for every user position;
  * orchestration: gdn_block_conv8_tp.stage with fakes makes the launches in order, calls the op once at batch 64 on the canon block and the block windows (the very call the M3
    block makes), returns per-user tensors in the served shapes, owns what it must and cleans up what it made when it cannot serve;
  * the sixteen-row stage (gdn_block_conv_tp) is untouched.

What no CPU test can show is that the op's per-element arithmetic is position-independent across tile rows and quarters; that is the in-trace audit (QWEN_FAST_TP4_VGLUE_AUDIT)
on a card, compared per user against the served call at batch 8.

Run: `python -m unittest test_octo_glue8_block` from scripts/ci.
"""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import gdn_block_conv8_tp as block8
import gdn_block_conv_tp as block16
import gdn_rows_dma8_tp as rows8
import tp4_vglue
import tp_shapes
from test_octo_glue8 import canon, execute, four, geometry, logical, raw_tiles, specials
from test_tp4_vglue_block import OpModel, Tensor, fake_operations, finite

HERE = Path(__file__).resolve().parent


def env(**values):
    return patch.dict(os.environ, values)


class DataTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        self.generator = torch.Generator().manual_seed(11)
        self.found = geometry()
        self.width = self.found.gdn_qkvzab
        self.columns = rows8.tile_columns(self.width)
        self.block = specials(self.generator, (64, self.columns * 32))
        self.block[:, self.width:] = 0
        self.block = finite(self.block)
        self.windows = [[finite(specials(self.generator, (8, self.found.gdn_qkv))) for slot in range(4)] for user in range(8)]
        self.op = OpModel(self.generator)

    def served(self):
        """Eight per-user calls on the served pieces: the users that start inside a tile canonical, each with its own windows, at the users' own 8 rows."""
        out = []
        for user in range(8):
            rows = self.block[8 * user:8 * user + 8]
            piece = canon(rows) if rows8.served_canon(user) else rows
            conv, beta, g, advanced = self.op(piece, self.windows[user], 8)
            out.append(dict(conv=conv, beta=beta, g=g, windows=advanced, z=piece[:, self.found.gdn_qkv:self.found.gdn_a_col]))
        return out

    def block_path(self):
        found, columns = self.found, self.columns
        (canon_tiles,) = execute(rows8.canon_block(8, self.width), [raw_tiles(self.block)], 1)
        canon_block = logical(canon_tiles, 64, columns * 32)
        flat = [raw_tiles(window) for user in self.windows for window in user]
        stacked = execute(block8.plan_windows(8), flat, 4)
        block_windows = [logical(tiles, 64, found.gdn_qkv) for tiles in stacked]
        conv, beta, g, advanced = self.op(canon_block, block_windows, 64)
        sources = [raw_tiles(conv), raw_tiles(beta), raw_tiles(g), canon_tiles] + [raw_tiles(window) for window in advanced]
        return execute(block8.plan_unstack(8, self.width), sources, 8 * 8)

    def test_the_block_launch_equals_the_eight_served_calls_bit_for_bit(self):
        found, expected, built = self.found, self.served(), self.block_path()
        for user in range(8):
            conv = logical(built[user], 8, found.gdn_qkv)
            beta = logical(built[8 + user], 8, 32)[:, :found.gdn_nv]
            g = logical(built[16 + user], 8, 32)[:, :found.gdn_nv]
            z = logical(built[24 + user], 8, found.gdn_a_col - found.gdn_qkv)
            self.assertTrue(torch.equal(conv, expected[user]['conv']), 'conv user %d' % user)
            self.assertTrue(torch.equal(beta, expected[user]['beta']), 'beta user %d' % user)
            self.assertTrue(torch.equal(g, expected[user]['g']), 'g user %d' % user)
            self.assertTrue(torch.equal(z, expected[user]['z']), 'z user %d' % user)
            for slot in range(4):
                window = logical(built[32 + user * 4 + slot], 8, found.gdn_qkv)
                self.assertTrue(torch.equal(window, expected[user]['windows'][slot]), 'window %d user %d' % (slot, user))

    def test_every_unstacked_tile_has_zero_rows_8_to_31_and_each_page_is_written_once(self):
        built = self.block_path()
        counts = [80] * 8 + [1] * 8 + [1] * 8 + [48] * 8 + [80] * 32
        self.assertEqual(len(built), len(counts))
        for index, (tiles, count) in enumerate(zip(built, counts)):
            self.assertEqual(sorted(tiles), list(range(count)), index)
            for raw in tiles.values():
                from test_octo_glue8 import from_raw

                self.assertFalse(bool(from_raw(raw)[8:].any()), index)

    def test_the_destination_indices_follow_the_documented_layout_at_any_user_count(self):
        for users in (8, 5):
            tasks = block8.plan_unstack(users, self.width)
            destinations = sorted({task[0] for task in tasks})
            self.assertEqual(destinations, list(range(4 * users + 4 * users)))
        windows = block8.plan_windows(8)
        self.assertEqual(sorted({task[0] for task in windows}), [0, 1, 2, 3])
        sources = {quarter[0] for task in windows for quarter in task[2] if quarter[2]}
        self.assertEqual(sources, set(range(32)))

    def test_a_model_that_mixed_rows_would_be_caught(self):
        """The negative control: an op whose row r depends on row r - 8 (a cross-user leak) fails the equality."""
        leaking = self.op

        class Leak(OpModel):
            def __call__(self, x, windows, batch):
                x = x.clone()
                if x.shape[0] > 8:
                    x[8:] = x[8:] + x[:-8]
                return OpModel.__call__(self, x, windows, batch)

        self.op = Leak.__new__(Leak)
        self.op.__dict__.update(leaking.__dict__)
        expected, built = self.served(), self.block_path()
        differs = any(not torch.equal(logical(built[user], 8, self.found.gdn_qkv), expected[user]['conv']) for user in range(8))
        self.assertTrue(differs)

    def test_a_canonicalisation_applied_to_the_wrong_users_would_be_caught(self):
        """The sixteen-row rule (users 1 and 3 canonical) applied to eight-row users is not the served one: the equality fails on the users it gets wrong."""
        planned = rows8.served_canon
        try:
            rows8.served_canon = lambda user: user % 2 == 1
            built = self.block_path()
        finally:
            rows8.served_canon = planned
        expected = self.served()
        wrong = [user for user in range(8)
                 if not torch.equal(logical(built[24 + user], 8, self.found.gdn_a_col - self.found.gdn_qkv), expected[user]['z'])]
        self.assertTrue(wrong, 'edge values reach z, so a wrong canonical set shows')
        self.assertEqual(set(wrong), {2, 6}, 'users 2 and 6 start inside a tile and are canonical; user parity says otherwise')


class StageTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = []
        self.operations = fake_operations(self.log)
        self.found = geometry()
        self.pieces = [Tensor('piece%d' % user, (1, 8, self.found.gdn_qkvzab), 'l1') for user in range(8)]
        self.histories = [[Tensor('history%d.%d' % (user, slot), (1, 8, 2560)) for slot in range(4)] for user in range(8)]
        self.windows = [[Tensor('window%d.%d' % (user, slot), (1, 8, 2560)) for slot in range(4)] for user in range(8)]
        self.groups = [(piece, 'initial%d' % user, history) for user, (piece, history) in enumerate(zip(self.pieces, self.histories))]
        self.projected = Tensor('projected', (1, 64, self.found.gdn_qkvzab), 'l1')
        self.launches = []
        self.conv_calls = []

    def run_stage(self, launch_error=None, windows='default', groups=None, **flags):
        def launch(mesh, sources, destinations, tasks, **keywords):
            if launch_error is not None and len(self.launches) == launch_error[0]:
                raise launch_error[1]
            self.launches.append(([s.name for s in sources], [d.name for d in destinations], tasks))

        def conv_gates(operations, projected, windows_, taps, dt_bias, neg_exp_A, rows):
            self.conv_calls.append((projected.name, [w.name for w in windows_], rows))
            return (Tensor('conv', (1, rows, 2560)), Tensor('beta', (1, rows, 12)), Tensor('gate', (1, rows, 12)))

        fallbacks = []
        with env(**flags), patch('gdn_rows_dma8_tp.launch', side_effect=launch), patch('gdn_user_batch_conv.conv_gates', side_effect=conv_gates):
            found = block8.stage('mesh', self.projected, groups or self.groups, self.windows if windows == 'default' else windows,
                                 ['tap%d' % slot for slot in range(4)], 'dt', 'neg', self.operations, note_fallback=fallbacks.append)
        return found, fallbacks

    def test_three_launches_and_one_op_call_at_batch_64_on_the_canon_block(self):
        found, fallbacks = self.run_stage()
        self.assertEqual(fallbacks, [])
        self.assertEqual(len(self.launches), 3)
        canon_launch, window_launch, unstack_launch = self.launches
        self.assertEqual(canon_launch[:2], (['projected'], ['empty0']))
        self.assertEqual(canon_launch[2], rows8.canon_block(8, self.found.gdn_qkvzab))
        self.assertEqual(window_launch[0], ['window%d.%d' % (user, slot) for user in range(8) for slot in range(4)])
        self.assertEqual(window_launch[1], ['empty1', 'empty2', 'empty3', 'empty4'])
        self.assertEqual(window_launch[2], block8.plan_windows(8))
        self.assertEqual(self.conv_calls, [('empty0', ['empty1', 'empty2', 'empty3', 'empty4'], 64)], 'the M3 block\'s own call: 64 rows, one op')
        self.assertEqual(unstack_launch[0], ['conv', 'beta', 'gate', 'empty0', 'empty1', 'empty2', 'empty3', 'empty4'])
        self.assertEqual(unstack_launch[1][32:], ['window%d.%d' % (user, slot) for user in range(8) for slot in range(4)],
                         'the advanced windows go back into the T2 windows themselves')
        self.assertEqual(unstack_launch[2], block8.plan_unstack(8, self.found.gdn_qkvzab))

    def test_the_users_get_the_served_shapes_in_dram(self):
        found, fallbacks = self.run_stage()
        shapes = {}
        for entry in self.log:
            if entry[0] == 'empty':
                shapes[entry[1]] = (entry[2], entry[3])
        self.assertEqual(shapes['empty0'], ((1, 64, 4120), 'dram'))
        self.assertEqual([shapes['empty%d' % index] for index in range(1, 5)], [((1, 64, 2560), 'dram')] * 4)
        for user in range(8):
            (conv, beta, g), z = found.users[user]
            self.assertEqual((conv.shape, beta.shape, g.shape, z.shape), ((1, 8, 2560), (1, 8, 12), (1, 8, 12), (1, 8, 1536)))
            self.assertTrue(all(tensor.memory == 'dram' for tensor in (conv, beta, g, z)))
        self.assertEqual(len({id(tensor) for (packed, z) in found.users for tensor in (*packed, z)}), 32)

    def test_ownership_the_block_temporaries_are_owned_and_the_user_tensors_are_not_repeated(self):
        found, fallbacks = self.run_stage()
        owned = {tensor.name for tensor in found.owned}
        self.assertEqual(owned, {'empty0', 'empty1', 'empty2', 'empty3', 'empty4', 'conv', 'beta', 'gate'})
        user_tensors = {tensor.name for (packed, z) in found.users for tensor in (*packed, z)}
        self.assertFalse(owned & user_tensors)
        self.assertEqual((found.held, found.entries), ([], []))

    def test_what_it_declines_and_why(self):
        found, fallbacks = self.run_stage(windows=None)
        self.assertIsNone(found)
        self.assertEqual(fallbacks, ['the T2 packed windows are not engaged'])
        self.assertEqual(self.launches, [])
        self.groups[2] = (Tensor('piece2', (1, 16, self.found.gdn_qkvzab), 'l1'), 'i', self.histories[2])
        found, fallbacks = self.run_stage()
        self.assertEqual(fallbacks[-1], 'a user is not an 8-row segment')
        found, fallbacks = self.run_stage(groups=self.groups[:4], windows=self.windows[:4])
        self.assertEqual(fallbacks[-1], 'not the octo block\'s 8 users')
        self.assertEqual(self.launches, [])
        self.assertEqual(self.conv_calls, [])

    def test_an_unsupported_launch_declines_and_frees_everything_it_made(self):
        for failing in (0, 1, 2):
            with self.subTest(launch=failing):
                self.log.clear()
                self.launches.clear()
                self.conv_calls.clear()
                found, fallbacks = self.run_stage(launch_error=(failing, rows8.Unsupported('no fit')))
                self.assertIsNone(found)
                self.assertEqual(fallbacks, ['no fit'])
                made = [entry[1] for entry in self.log if entry[0] == 'empty'] + (['conv', 'beta', 'gate'] if failing == 2 else [])
                freed = [entry[1] for entry in self.log if entry[0] == 'free']
                self.assertEqual(sorted(made), sorted(freed))
                self.assertEqual(len(self.conv_calls), 1 if failing == 2 else 0)

    def test_another_error_is_not_swallowed_and_frees_too(self):
        with self.assertRaises(RuntimeError):
            self.run_stage(launch_error=(1, RuntimeError('device')))
        made = [entry[1] for entry in self.log if entry[0] == 'empty']
        freed = [entry[1] for entry in self.log if entry[0] == 'free']
        self.assertEqual(sorted(made), sorted(freed))

    def test_the_audit_holds_the_served_call_at_the_users_own_eight_rows_beside_a_copy_of_each_user_tensor(self):
        served_calls = []

        def build_windows(mesh, piece, history):
            served_calls.append(('windows', piece.name))
            return [Tensor('served-window:%s.%d' % (piece.name, slot), (1, 8, 2560)) for slot in range(4)]

        with patch('gdn_conv_windows.build_windows', side_effect=build_windows):
            found, fallbacks = self.run_stage(QWEN_FAST_TP4_VGLUE_AUDIT='1', QWEN_FAST_TP4_GDN_GLUE='1')
        self.assertEqual(served_calls, [('windows', 'piece%d' % user) for user in range(8)])
        served = [call for call in self.conv_calls if call[0].startswith('piece')]
        self.assertEqual([call[0] for call in served], ['piece%d' % user for user in range(8)])
        self.assertTrue(all(call[2] == 8 for call in served), 'the served call at the users\' own 8 rows')
        self.assertEqual(len(found.entries), 8)
        for user, entries in enumerate(found.entries):
            self.assertEqual([entry['label'] for entry in entries],
                             ['block conv user %d %s' % (user, name) for name in ('conv', 'beta', 'g', 'z', 'window 0', 'window 1', 'window 2', 'window 3')])
            self.assertTrue(all(entry['mine'].name.startswith('copy:') for entry in entries))
        owned = {id(tensor) for tensor in found.owned}
        self.assertFalse(any(id(tensor) in owned for tensor in found.held))
        self.assertEqual(len(found.held), 8 * (4 + 3 + 1 + 8))
        # the audit's expected entries for eight users: eight per-user lists of eight, which tp4_vglue.missing_entries counts per user
        self.assertEqual(tp4_vglue.BLOCK_ENTRIES_PER_USER, 8)

    def test_the_sixteen_row_stage_is_untouched_and_still_refuses_eight_rows(self):
        self.assertEqual((block16.USER_ROWS, block16.SLOTS), (16, 4))
        self.assertEqual(block16.problem(self.groups, self.windows, self.operations), 'a user is not a 16-row segment')
        source = (HERE / 'gdn_block_conv_tp.py').read_text()
        self.assertIn('import gdn_rows_dma_tp as rows_dma', source)
        self.assertNotIn('gdn_rows_dma8', source)

    def test_remap_keeps_zero_sources_zero(self):
        quarters = ((3, 5, 1), rows8.Z, (2, 9, 5), rows8.Z)
        self.assertEqual(block8.remap_quarters(quarters, 0, 4, 2), ((14, 5, 1), rows8.Z, (10, 9, 5), rows8.Z))


if __name__ == '__main__':
    unittest.main()
