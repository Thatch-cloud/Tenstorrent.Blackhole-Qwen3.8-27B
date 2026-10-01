"""V1 (QWEN_FAST_TP4_GDN_BLOCK_CONV): one conv-gates launch per layer against four, on tiles.

Three layers of proof, each failing if the block path differs from the per-user path it replaces:

  * data: the four launches' plans (canon block, window stack, unstack), run through the row-mover emulator around an emulation
    of gdn_decode_conv_gates' contract (row-parallel FIR + silu + gates, windows advanced in place, rows past the batch zero),
    equal the four served per-user calls on every output and every advanced window, bit for bit, -0 and denormals included;
  * orchestration: gdn_block_conv_tp.stage with fakes makes the launches in order, calls the op once at batch 64 on the canon
    block and the block windows, returns per-user tensors in the served shapes, owns what it must and cleans up what it made
    when it cannot serve;
  * integration: run_user_batched_projected with a block_stage hands K5-A the staged tensors and the same windows, and without
    one (or when the stage declines) makes the served four conv_gates calls.

What no CPU test can show is that the op's per-element arithmetic is position-independent across tile rows and faces; that is
the in-trace audit (QWEN_FAST_TP4_VGLUE_AUDIT) on a card, compared per user against the served call.

Run at py 3.11: `py -3.11 -m unittest test_tp4_vglue_block` from scripts/ci.
"""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import gdn_block_conv_tp as block
import gdn_rows_dma_tp as rows_dma
import gdn_seq_block
import gdn_user_batch_conv
import tp4_vglue
import tp_shapes
from test_tp4_vglue_gdn import canon, execute, specials, to_tiles


def four():
    """QWEN_FAST_TP=4 without the address seam (these fakes are plain)."""
    return patch.dict(os.environ, {'QWEN_FAST_TP': '4'})


def env(**values):
    return patch.dict(os.environ, values)


def geometry():
    with four():
        return tp_shapes.active()


def finite(bits):
    """Replace infinities and NaNs (exponent all ones) with 1.0: the FIR emulation compares values, not payloads."""
    raw = bits.to(torch.int32) & 0xFFFF
    return torch.where((raw & 0x7F80) == 0x7F80, torch.full_like(bits, 0x3F80), bits)


def to_float(bits):
    return bits.contiguous().view(torch.bfloat16).to(torch.float32)


def to_bits(values):
    return values.to(torch.bfloat16).contiguous().view(torch.int16)


def from_tiles(tiles, rows, columns):
    """{page: tile} -> (rows, columns) int16 (columns a multiple of 32), page = tile row * tile columns + tile column."""
    tile_columns = columns // 32
    tile_rows = -(-rows // 32)
    matrix = torch.zeros(tile_rows * 32, columns, dtype=torch.int16)
    for page, tile in tiles.items():
        row, column = divmod(page, tile_columns)
        matrix[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32] = tile
    return matrix[:rows]


class OpModel(object):
    """gdn_decode_conv_gates' contract (gdn_conv_gates.hpp), row-parallel: window = [cs1, cs2, cs3, x]; conv = silu(sum
    window_j * tap_j); beta = sigmoid(b); g = neg_exp_A * softplus(a + dt_bias); the states advance to the window. Rows of x at or
    past `batch` are zero when they enter the window. Every value is computed per element in float32 and rounded to bf16, so
    any arrangement of rows gives the same bits - which is the property the block launch relies on and the op must show."""

    def __init__(self, generator):
        found = geometry()
        self.found = found
        self.taps = torch.randn(4, found.gdn_qkv, generator=generator)
        self.dt_bias = torch.randn(found.gdn_nv, generator=generator)
        self.neg_exp_A = -torch.rand(found.gdn_nv, generator=generator) - 0.1

    def __call__(self, x, windows, batch):
        found = self.found
        x = x.clone()
        x[batch:] = 0
        old = [to_float(window) for window in windows]
        current = to_float(x[:, :found.gdn_qkv])
        frames = [old[1], old[2], old[3], current]
        conv = torch.nn.functional.silu(sum(frame * tap for frame, tap in zip(frames, self.taps)))
        a = to_float(x[:, found.gdn_a_col:found.gdn_a_col + found.gdn_nv])
        b = to_float(x[:, found.gdn_b_col:found.gdn_b_col + found.gdn_nv])
        beta = torch.sigmoid(b)
        g = self.neg_exp_A * torch.nn.functional.softplus(a + self.dt_bias)
        advanced = [windows[1].clone(), windows[2].clone(), windows[3].clone(), x[:, :found.gdn_qkv].clone()]
        return to_bits(conv), to_bits(beta), to_bits(g), advanced


class DataTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        self.generator = torch.Generator().manual_seed(11)
        self.found = geometry()
        self.width = self.found.gdn_qkvzab
        self.columns = rows_dma.tile_columns(self.width)
        self.block = specials(self.generator, (64, self.columns * 32))
        self.block[:, self.width:] = 0
        self.block = finite(self.block)
        self.windows = [[finite(specials(self.generator, (16, self.found.gdn_qkv))) for slot in range(4)] for user in range(4)]
        self.op = OpModel(self.generator)

    def served(self):
        """Four per-user calls on the served pieces: users 1 and 3 canonical, each with its own windows."""
        out = []
        for user in range(4):
            rows = self.block[16 * user:16 * user + 16]
            piece = canon(rows) if user % 2 else rows
            conv, beta, g, advanced = self.op(piece, self.windows[user], 16)
            out.append(dict(conv=conv, beta=beta, g=g, windows=advanced, z=piece[:, self.found.gdn_qkv:self.found.gdn_a_col]))
        return out

    def block_path(self):
        found, columns = self.found, self.columns
        (canon_tiles,) = execute(rows_dma.canon_block(4, self.width), [to_tiles(self.block)], [{}])
        canon_block = from_tiles(canon_tiles, 64, columns * 32)
        flat = [to_tiles(window) for user in self.windows for window in user]
        stacked = execute(block.plan_windows(4), flat, [{} for slot in range(4)])
        block_windows = [from_tiles(tiles, 64, found.gdn_qkv) for tiles in stacked]
        conv, beta, g, advanced = self.op(canon_block, block_windows, 64)
        sources = [to_tiles(conv), to_tiles(beta), to_tiles(g), canon_tiles] + [to_tiles(window) for window in advanced]
        destinations = execute(block.plan_unstack(4, self.width), sources, [{} for index in range(32)])
        return destinations

    def test_the_block_launch_equals_the_four_served_calls_bit_for_bit(self):
        found, expected, built = self.found, self.served(), self.block_path()
        for user in range(4):
            conv = from_tiles(built[user], 16, found.gdn_qkv)
            beta = from_tiles(built[4 + user], 16, 32)[:, :found.gdn_nv]
            g = from_tiles(built[8 + user], 16, 32)[:, :found.gdn_nv]
            z = from_tiles(built[12 + user], 16, found.gdn_a_col - found.gdn_qkv)
            self.assertTrue(torch.equal(conv, expected[user]['conv']), 'conv user %d' % user)
            self.assertTrue(torch.equal(beta, expected[user]['beta']), 'beta user %d' % user)
            self.assertTrue(torch.equal(g, expected[user]['g']), 'g user %d' % user)
            self.assertTrue(torch.equal(z, expected[user]['z']), 'z user %d' % user)
            for slot in range(4):
                window = from_tiles(built[16 + user * 4 + slot], 16, found.gdn_qkv)
                self.assertTrue(torch.equal(window, expected[user]['windows'][slot]), 'window %d user %d' % (slot, user))

    def test_every_unstacked_tile_has_zero_rows_16_to_31_and_each_page_is_written_once(self):
        built = self.block_path()
        counts = [80] * 4 + [1] * 4 + [1] * 4 + [48] * 4 + [80] * 16
        for index, (tiles, count) in enumerate(zip(built, counts)):
            self.assertEqual(sorted(tiles), list(range(count)), index)
            for tile in tiles.values():
                self.assertFalse(bool(tile[16:].any()), index)

    def test_the_plans_fit_the_kernels_task_carrying_limit(self):
        for tasks in (block.plan_windows(4), block.plan_unstack(4, self.width), rows_dma.canon_block(4, self.width)):
            rows_dma.distribute(tasks, 110)
        self.assertEqual(len(block.plan_windows(4)), 4 * 2 * 80)
        self.assertEqual(len(block.plan_unstack(4, self.width)), 4 * 80 + 4 + 4 + 4 * 48 + 16 * 80)

    def test_a_model_that_mixed_rows_would_be_caught(self):
        """The negative control: an op whose row r depends on row r - 16 (a cross-user leak) fails the equality."""
        leaking = self.op

        class Leak(OpModel):
            def __call__(self, x, windows, batch):
                x = x.clone()
                if x.shape[0] > 16:
                    x[16:] = x[16:] + x[:-16]
                return OpModel.__call__(self, x, windows, batch)

        self.op = Leak.__new__(Leak)
        self.op.__dict__.update(leaking.__dict__)
        expected, built = self.served(), self.block_path()
        differs = any(not torch.equal(from_tiles(built[user], 16, self.found.gdn_qkv), expected[user]['conv'])
                      for user in range(4))
        self.assertTrue(differs)


# --- the stage, with fakes -------------------------------------------------------------------------------------------------

class Tensor(object):
    def __init__(self, name, shape, memory='dram'):
        self.name, self.shape, self.memory = name, tuple(shape), memory
        self.dtype, self.layout = 'bf16', 'tile'

    def memory_config(self):
        return self.memory


def fake_operations(log):
    counter = iter(range(1000))

    def empty(shape, dtype=None, layout=None, device=None, memory_config=None):
        tensor = Tensor('empty%d' % next(counter), shape, memory_config)
        log.append(('empty', tensor.name, tuple(shape), memory_config))
        return tensor

    operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1', bfloat16='bf16', TILE_LAYOUT='tile')
    operations.empty = empty
    operations.deallocate = lambda tensor: log.append(('free', tensor.name))
    operations.get_device_tensors = lambda tensor: [SimpleNamespace(buffer_address=lambda a=hash(tensor.name) + chip: a)
                                                    for chip in range(4)]
    operations.clone = lambda tensor, memory_config=None: Tensor('copy:' + tensor.name, tensor.shape, memory_config)
    operations.slice = lambda tensor, start, stop, memory_config=None: Tensor('slice:' + tensor.name,
                                                                              [b - a for a, b in zip(start, stop)])
    return operations


class StageTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = []
        self.operations = fake_operations(self.log)
        self.found = geometry()
        self.pieces = [Tensor('piece%d' % user, (1, 16, self.found.gdn_qkvzab), 'l1') for user in range(4)]
        self.histories = [[Tensor('history%d.%d' % (user, slot), (1, 16, 2560)) for slot in range(4)] for user in range(4)]
        self.windows = [[Tensor('window%d.%d' % (user, slot), (1, 16, 2560)) for slot in range(4)] for user in range(4)]
        self.groups = [(piece, 'initial%d' % user, history)
                       for user, (piece, history) in enumerate(zip(self.pieces, self.histories))]
        self.projected = Tensor('projected', (1, 64, self.found.gdn_qkvzab), 'l1')
        self.launches = []
        self.conv_calls = []

    def run_stage(self, launch_error=None, windows='default', **flags):
        def launch(mesh, sources, destinations, tasks, **keywords):
            if launch_error is not None and len(self.launches) == launch_error[0]:
                raise launch_error[1]
            self.launches.append(([s.name for s in sources], [d.name for d in destinations], tasks))

        def conv_gates(operations, projected, windows_, taps, dt_bias, neg_exp_A, rows):
            self.conv_calls.append((projected.name, [w.name for w in windows_], rows))
            return (Tensor('conv', (1, rows, 2560)), Tensor('beta', (1, rows, 12)), Tensor('gate', (1, rows, 12)))

        fallbacks = []
        with four(), env(**flags), patch('gdn_rows_dma_tp.launch', side_effect=launch), \
                patch('gdn_user_batch_conv.conv_gates', side_effect=conv_gates):
            found = block.stage('mesh', self.projected, self.groups, self.windows if windows == 'default' else windows,
                                ['tap%d' % slot for slot in range(4)], 'dt', 'neg', self.operations,
                                note_fallback=fallbacks.append)
        return found, fallbacks

    def test_three_launches_and_one_op_call_at_batch_64_on_the_canon_block(self):
        found, fallbacks = self.run_stage()
        self.assertEqual(fallbacks, [])
        self.assertEqual(len(self.launches), 3)
        canon_launch, window_launch, unstack_launch = self.launches
        self.assertEqual(canon_launch[:2], (['projected'], ['empty0']))
        self.assertEqual(canon_launch[2], rows_dma.canon_block(4, self.found.gdn_qkvzab))
        self.assertEqual(window_launch[0], ['window%d.%d' % (user, slot) for user in range(4) for slot in range(4)])
        self.assertEqual(window_launch[1], ['empty1', 'empty2', 'empty3', 'empty4'])
        self.assertEqual(window_launch[2], block.plan_windows(4))
        self.assertEqual(self.conv_calls, [('empty0', ['empty1', 'empty2', 'empty3', 'empty4'], 64)])
        self.assertEqual(unstack_launch[0], ['conv', 'beta', 'gate', 'empty0', 'empty1', 'empty2', 'empty3', 'empty4'])
        self.assertEqual(unstack_launch[1][16:], ['window%d.%d' % (user, slot) for user in range(4) for slot in range(4)],
                         'the advanced windows go back into the T2 windows themselves')
        self.assertEqual(unstack_launch[2], block.plan_unstack(4, self.found.gdn_qkvzab))

    def test_the_users_get_the_served_shapes_in_dram(self):
        found, fallbacks = self.run_stage()
        shapes = {}
        for entry in self.log:
            if entry[0] == 'empty':
                shapes[entry[1]] = (entry[2], entry[3])
        self.assertEqual(shapes['empty0'], ((1, 64, 4120), 'dram'))
        self.assertEqual([shapes['empty%d' % index] for index in range(1, 5)], [((1, 64, 2560), 'dram')] * 4)
        for user in range(4):
            (conv, beta, g), z = found.users[user]
            self.assertEqual((conv.shape, beta.shape, g.shape, z.shape),
                             ((1, 16, 2560), (1, 16, 12), (1, 16, 12), (1, 16, 1536)))
            self.assertTrue(all(tensor.memory == 'dram' for tensor in (conv, beta, g, z)))
        self.assertEqual(len({id(tensor) for (packed, z) in found.users for tensor in (*packed, z)}), 16)

    def test_ownership_the_block_temporaries_are_owned_and_the_user_tensors_are_not_repeated(self):
        found, fallbacks = self.run_stage()
        owned = {tensor.name for tensor in found.owned}
        self.assertEqual(owned, {'empty0', 'empty1', 'empty2', 'empty3', 'empty4', 'conv', 'beta', 'gate'})
        user_tensors = {tensor.name for (packed, z) in found.users for tensor in (*packed, z)}
        self.assertFalse(owned & user_tensors)
        self.assertEqual(found.held, [])
        self.assertEqual(found.entries, [])

    def test_no_window_stage_no_block(self):
        found, fallbacks = self.run_stage(windows=None)
        self.assertIsNone(found)
        self.assertEqual(fallbacks, ['the T2 packed windows are not engaged'])
        self.assertEqual(self.launches, [])
        self.assertEqual(self.conv_calls, [])

    def test_a_segment_that_is_not_sixteen_rows_declines(self):
        self.groups[2] = (Tensor('piece2', (1, 8, self.found.gdn_qkvzab), 'l1'), 'i', self.histories[2])
        found, fallbacks = self.run_stage()
        self.assertIsNone(found)
        self.assertEqual(fallbacks, ['a user is not a 16-row segment'])

    def test_an_unsupported_launch_declines_and_frees_everything_it_made(self):
        for failing in (0, 1, 2):
            with self.subTest(launch=failing):
                self.log.clear()
                self.launches.clear()
                self.conv_calls.clear()
                found, fallbacks = self.run_stage(launch_error=(failing, rows_dma.Unsupported('no fit')))
                self.assertIsNone(found)
                self.assertEqual(fallbacks, ['no fit'])
                made = [entry[1] for entry in self.log if entry[0] == 'empty'] + (['conv', 'beta', 'gate'] if failing == 2 else [])
                freed = [entry[1] for entry in self.log if entry[0] == 'free']
                self.assertEqual(sorted(made), sorted(freed))
                # the op ran only when its two feeding launches did, and never advanced a per-user window
                self.assertEqual(len(self.conv_calls), 1 if failing == 2 else 0)

    def test_another_error_is_not_swallowed_and_frees_too(self):
        with self.assertRaises(RuntimeError):
            self.run_stage(launch_error=(1, RuntimeError('device')))
        made = [entry[1] for entry in self.log if entry[0] == 'empty']
        freed = [entry[1] for entry in self.log if entry[0] == 'free']
        self.assertEqual(sorted(made), sorted(freed))

    def test_the_audit_holds_the_served_call_beside_a_copy_of_each_user_tensor(self):
        served_calls = []

        def build_windows(mesh, piece, history):
            served_calls.append(('windows', piece.name))
            return [Tensor('served-window:%s.%d' % (piece.name, slot), (1, 16, 2560)) for slot in range(4)]

        with patch('gdn_conv_windows.build_windows', side_effect=build_windows):
            found, fallbacks = self.run_stage(QWEN_FAST_TP4_VGLUE_AUDIT='1', QWEN_FAST_TP4_GDN_GLUE='1')
        self.assertEqual(served_calls, [('windows', 'piece%d' % user) for user in range(4)])
        served = [call for call in self.conv_calls if call[0].startswith('piece')]
        self.assertEqual([call[0] for call in served], ['piece0', 'piece1', 'piece2', 'piece3'])
        self.assertTrue(all(call[2] == 16 for call in served), 'the served call at the users\' own 16 rows')
        self.assertEqual(len(found.entries), 4)
        for user, entries in enumerate(found.entries):
            self.assertEqual([entry['label'] for entry in entries],
                             ['block conv user %d %s' % (user, name)
                              for name in ('conv', 'beta', 'g', 'z', 'window 0', 'window 1', 'window 2', 'window 3')])
            self.assertTrue(all(entry['mine'].name.startswith('copy:') for entry in entries))
        # everything held is outside `owned`
        owned = {id(tensor) for tensor in found.owned}
        self.assertFalse(any(id(tensor) in owned for tensor in found.held))
        self.assertEqual(len(found.held), 4 * (4 + 3 + 1 + 8))


# --- run_user_batched_projected with a block stage --------------------------------------------------------------------------

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        from test_gdn_user_batch_conv import fake_operations as conv_fake, user_group

        self.calls = []
        self.operations = conv_fake(self.calls)
        self.groups = [user_group(index, self.operations) for index in range(4)]
        self.execute_calls = []
        self.patchers = []

    def layer(self, stage, **flags):
        windows = [[self.operations.make('w%d.%d' % (user, slot), (1, 16, 5120)) for slot in range(4)] for user in range(4)]

        def execute(mesh, inputs, kernels, operations, output_memory=None):
            self.execute_calls.append([tuple(value.name for value in user) for user in inputs])
            return [(self.operations.make('out%d' % index, (1, 16, 3072)), self.operations.make('states%d' % index,
                                                                                               (16, 24, 128, 128)))
                    for index in range(len(inputs))]

        with env(**flags), patch.object(gdn_seq_block.batch, 'execute', side_effect=execute), \
                patch('gdn_conv_windows_packed.build_windows_packed', return_value=windows), \
                patch('gdn_user_batch_conv.validate_projected', return_value=16):
            results = gdn_user_batch_conv.run_user_batched_projected(
                'mesh', self.groups, [self.operations.make('tap%d' % slot, (1, 1, 2560)) for slot in range(4)], 'dt', 'neg',
                self.operations.make('norm', (1, 1, 1536)), 'kernels', self.operations, block_stage=stage)
        return results, windows

    def staged(self):
        users = [((self.operations.make('c%d' % index, (1, 16, 5120)), self.operations.make('b%d' % index, (1, 16, 24)),
                   self.operations.make('g%d' % index, (1, 16, 24))), self.operations.make('z%d' % index, (1, 16, 4096)))
                 for index in range(4)]
        return block.BlockStage(users, [self.operations.make('temp', (1, 64, 5120))], [], [])

    def test_the_stage_replaces_the_per_user_conv_gates_and_z_slices(self):
        stage = Mock(side_effect=lambda groups, windows: self.staged())
        results, windows = self.layer(stage, QWEN_FAST_VERIFY_T2='1')
        stage.assert_called_once()
        self.assertEqual([call for call in self.calls if call[0] in ('conv_gates', 'slice')], [])
        self.assertEqual(self.execute_calls[0][0][:5], ('c0', 'b0', 'g0', 'initial0', 'z0'))
        for user, result in enumerate(results):
            self.assertEqual([w.name for w in result['packed_conv_states']], ['w%d.%d' % (user, slot) for slot in range(4)])
        names = lambda result: [value.name for value in result['owned']]
        self.assertIn('temp', names(results[0]))
        self.assertNotIn('temp', names(results[1]))
        self.assertEqual(names(results[2])[:4], ['w2.0', 'w2.1', 'w2.2', 'w2.3'])

    def test_a_declining_stage_runs_the_served_loop(self):
        stage = Mock(return_value=None)
        results, windows = self.layer(stage, QWEN_FAST_VERIFY_T2='1')
        stage.assert_called_once()
        self.assertEqual([call[0] for call in self.calls if call[0] in ('conv_gates',)], ['conv_gates'] * 4)
        self.assertEqual(len([call for call in self.calls if call[0] == 'slice']), 4)

    def test_without_the_t2_windows_the_stage_is_not_called(self):
        stage = Mock(side_effect=lambda groups, windows: self.staged())
        with patch('gdn_conv_windows.build_windows', side_effect=lambda mesh, piece, history: [
                self.operations.make('sw', (1, 16, 5120)) for unused in range(4)]):
            results, windows = self.layer(stage)
        stage.assert_not_called()
        self.assertEqual(len([call for call in self.calls if call[0] == 'conv_gates']), 4)

    def test_no_stage_is_the_served_path_unchanged(self):
        results, windows = self.layer(None, QWEN_FAST_VERIFY_T2='1')
        self.assertEqual(len([call for call in self.calls if call[0] == 'conv_gates']), 4)
        self.assertEqual(self.calls[:0], [])


class TwinBlockTests(unittest.TestCase):
    def test_the_block_stage_is_passed_only_with_the_t2_windows_cut(self):
        import gdn_device_loop_state_tp as twin

        state = object.__new__(twin.DeviceLoopState)
        state.gdn = SimpleNamespace(mesh='mesh', tw=dict(conv_taps=['a'] * 4, dt_bias='dt', neg_exp_A='neg', norm_w='norm'))
        state.operations, state.kernels, state.prefix_zero_reuse = 'ops', 'kernels', False
        seen, lines = [], []
        with patch('gdn_device_loop_state_tp.run_user_batched_projected',
                   side_effect=lambda *args, **kwargs: seen.append(kwargs) or []), \
                patch('gdn_device_loop_state_tp.tp4_vglue.log_line', side_effect=lines.append):
            twin.DeviceLoopState.glue_fallbacks = set()
            with four(), env(QWEN_FAST_TP4_GDN_GLUE='1'):
                state.run_users('projected', [], [])
            with four(), env(QWEN_FAST_TP4_GDN_GLUE='1', QWEN_FAST_TP4_GDN_BLOCK_CONV='1'):
                state.run_users('projected', [], [])
            with four(), env(QWEN_FAST_TP4_GDN_GLUE='1', QWEN_FAST_TP4_GDN_BLOCK_CONV='1', QWEN_FAST_VERIFY_T2='1'):
                state.run_users('projected', [], [])
        self.assertEqual([sorted(kwargs) for kwargs in seen], [['prefix_zero_reuse'], ['prefix_zero_reuse'],
                                                               ['block_stage', 'prefix_zero_reuse']])
        self.assertTrue(any(line.startswith(tp4_vglue.FALLBACK) and 'site=block_conv' in line for line in lines))


if __name__ == '__main__':
    unittest.main()
