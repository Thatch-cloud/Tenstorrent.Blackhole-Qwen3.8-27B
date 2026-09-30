"""The four-card extent replay readers (extent_attention_replay_tp) and the fold / mask kernels' four-card siblings.

At one KV head a chip's six query heads fold token-major: a group of R tokens is (1, 1, R * 6, 256), its narrow tail mask
(B, 1, R * 6, 256), and K64j runs 0x23 (tail | share | extent; the slice needs a second KV head). The pinned reader stays
byte for byte what the TP2 evidence pins, so the checks here are: the twin at the pair agrees with the pinned module on
what does not depend on the width, the four-card mask is the token-major causal mask of each group (an oracle that is
not the transliteration), the kernels' index maps reproduce the host fold, the reader builds and calls on a four-chip
fake device with 0x23, the seam puts the twin where model_batch and packed_verifier import the pinned module, and every
chip loop covers four chips.
"""

from contextlib import ExitStack
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import attention_fold_dma_tp
import attention_mask_replay
import extent_attention_replay as pinned
import extent_attention_replay_tp as quad
import tp_addresses
import tp_kernels
import tp_shapes
from test_extent_attention_replay import (C, ENV, MESH, WIDTH, FakeDevice, FakeShard, FakeTensor, fake_ttnn, host_table,
                                          program_view)

HERE = Path(__file__).resolve().parent
K = pinned.K


def four():
    return patch.dict(os.environ, {'QWEN_FAST_TP': '4'})


def pair():
    return patch.dict(os.environ, {}, clear=True)


class FourChipDevice(FakeDevice):
    """test_extent_attention_replay.FakeDevice with four chips."""

    def __init__(self):
        FakeDevice.__init__(self)
        self.next = [0x1000 + chip * 0x800000 for chip in range(4)]

    def device_tensor(self, shape, dtype=None, layout=None, memory='dram', value=None, name='', origin=None):
        shards = []
        for chip in range(4):
            shards.append(FakeShard(self.next[chip]))
            self.next[chip] += 0x100
        tensor = FakeTensor(shape, dtype, layout, memory, shards, value, name, origin)
        self.live.append(tensor)
        return tensor


def lend(device, rows=16, width=WIDTH):
    pairs = []
    for bundle in pinned.LAYOUT(rows, 8):
        batches = len(bundle)
        pairs.append((device.device_tensor((batches, width), 'int32', 'row_major', 'dram',
                                           torch.zeros(batches, width, dtype=torch.int32), 'table'),
                      device.device_tensor((batches,), 'int32', 'row_major', 'dram',
                                           torch.zeros(batches, dtype=torch.int32), 'cur_pos')))
    return pairs


class Harness(object):
    def __init__(self, test, env=ENV):
        self.stack = ExitStack()
        self.logs, self.programs = [], []
        self.stack.enter_context(patch.dict(os.environ, dict(env, QWEN_FAST_TP='4'), clear=True))
        self.stack.enter_context(patch('pooled_attention_replay._binary_checked', []))
        self.stack.enter_context(patch('pooled_attention_replay._pindiag', side_effect=self.logs.append))
        self.stack.enter_context(patch('extent_attention_replay_tp._pindiag', side_effect=self.logs.append))
        self.stack.enter_context(patch('pooled_attention_replay.loaded_binary_has_modes',
                                       side_effect=lambda markers: ('/k64j/_ttnncpp.so', True)))

        def program(mesh, positions, mask, *, rows, batches, offset):
            self.programs.append(dict(positions=positions, mask=mask, rows=rows, batches=batches, offset=offset))
            return 'program%d' % len(self.programs)

        self.stack.enter_context(patch('extent_attention_replay_tp.prepare_narrow', side_effect=program))
        test.addCleanup(self.stack.close)


def segment(device, start=4200, rows=16, width=WIDTH, **options):
    pairs = lend(device, rows, width)
    options.setdefault('max_group_rows', 8)
    return quad.ExtentSegmentReader(device, MESH, rows, width, host_table(5, width), storage=pairs, start=start,
                                    **options), pairs


def causal_oracle(rows, start, capacity, head_rows=6):
    """The token-major folded causal mask of one group: row token * head_rows + head masks the keys past start + token."""
    mask = torch.zeros(rows * head_rows, capacity, dtype=torch.bfloat16)
    for token in range(rows):
        mask[token * head_rows:(token + 1) * head_rows, start + token + 1:] = float('-inf')
    return mask.reshape(1, 1, rows * head_rows, capacity)


class TwinAtThePairTests(unittest.TestCase):
    def test_the_geometry_free_helpers_are_the_pinned_objects(self):
        for name in ('extent', 'extent_values', 'accept_limit', 'admits', 'check_start', 'LAYOUT', 'K', 'MIN_LIVE_START',
                     'EXTENT_GROUP_ROWS', 'EXTENT_BUNDLE_ENTRIES', 'ENGAGED_MARKER', 'F22_MARKER', 'TREE_SCRATCH_ENV'):
            self.assertIs(getattr(quad, name), getattr(pinned, name), name)

    def test_at_the_pair_the_twin_computes_what_the_pinned_module_computes(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with pair():
            for rows, batches, offset in ((8, 2, 0), (8, 1, 8), (4, 3, 0)):
                for word in (0, 7, 130, 255):
                    for capacity in (256, 4352):
                        ours = quad.mask_tiles(word, rows, batches, offset, capacity)
                        theirs = pinned.mask_tiles(word, rows, batches, offset, capacity)
                        self.assertTrue(torch.equal(ours[0], theirs[0]))
                        self.assertTrue(torch.equal(ours[1], theirs[1]))
                    self.assertTrue(torch.equal(quad.narrow_mask_host(word, rows, batches, offset).view(torch.int16),
                                                pinned.narrow_mask_host(word, rows, batches, offset).view(torch.int16)))
            self.assertEqual(quad.head_rows(), 12)


class FourCardMaskTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_the_narrow_mask_is_the_token_major_causal_mask_of_each_group(self):
        self.assertEqual(quad.head_rows(), 6)
        for family in (256, 4352, C):
            for residue in (0, 7, 128, 240, 250, 255):
                start = family - K + residue
                narrow = quad.narrow_mask_host(residue, 8, 2, 0)
                self.assertEqual(tuple(narrow.shape), (2, 1, 48, 256))
                for batch in range(2):
                    oracle = causal_oracle(8, start + batch * 8, family)[0, 0, :, family - K:family]
                    with self.subTest(family=family, residue=residue, batch=batch):
                        self.assertTrue(torch.equal(narrow[batch, 0].view(torch.int16), oracle.view(torch.int16)))

    def test_the_narrow_mask_is_the_last_256_columns_of_the_wide_mask(self):
        for family in (256, 512, 4352, 16640):
            for residue in (0, 7, 127, 128, 240, 255):
                start = family - K + residue
                wide = quad.replay_mask_host(start, 8, 2, 0, family).view(torch.int16)
                narrow = quad.narrow_mask_host(start & 255, 8, 2, 0).view(torch.int16)
                self.assertTrue(torch.equal(wide[..., family - K:], narrow), (family, residue))
                self.assertFalse(bool(wide[..., :family - K].any()), (family, residue))

    def test_the_mask_grid_is_two_head_tiles_at_eight_rows(self):
        pages, tiles = quad.mask_tiles(0, 8, 2, 0, K)
        # 8 tokens x 6 rows = 48 head rows = two 32-row tiles; two batches; eight column tiles each
        self.assertEqual(pages.numel(), 2 * 2 * 8)
        self.assertEqual(sorted(pages.tolist()), list(range(32)))


class KernelIndexModelTests(unittest.TestCase):
    """The fold kernel's row maps (attention_fold_dma.cpp) with the head-row count as a parameter, against a torch fold
    (forward) and unfold (inverse). Written from the kernel's loops, so a change to the sibling's arithmetic fails here."""

    @staticmethod
    def forward_source(rows, head_rows, task_tile, target_row):
        head = task_tile * 32 + target_row
        if head >= rows * head_rows:
            return None
        remainder = head % (rows * 6)
        return remainder // 6, (head // (rows * 6)) * 6 + remainder % 6

    @staticmethod
    def inverse_source(rows, head_rows, token, target_row):
        if target_row >= head_rows:
            return None
        head = (target_row // 6) * rows * 6 + token * 6 + target_row % 6
        return head // 32, head % 32

    def test_the_forward_map_is_the_torch_fold_at_both_widths(self):
        for head_rows, kv in ((12, 2), (6, 1)):
            for rows in (1, 3, 8):
                query = torch.arange(rows * head_rows * 4, dtype=torch.float32).reshape(1, rows, head_rows, 4)
                # torch fold: (rows, kv, 6, .) -> (kv, rows, 6, .) -> (1, 1, rows * head_rows, .)
                folded = query.reshape(rows, kv, 6, 4).permute(1, 0, 2, 3).reshape(rows * head_rows, 4)
                built = torch.zeros_like(folded)
                for tile in range(-(-rows * head_rows // 32)):
                    for row in range(32):
                        source = self.forward_source(rows, head_rows, tile, row)
                        if source is not None:
                            built[tile * 32 + row] = query[0, source[0], source[1]]
                self.assertTrue(torch.equal(built, folded), (head_rows, rows))

    def test_the_inverse_map_is_the_torch_unfold_at_both_widths(self):
        for head_rows, kv in ((12, 2), (6, 1)):
            for rows in (1, 3, 8):
                folded = torch.arange(rows * head_rows * 4, dtype=torch.float32).reshape(rows * head_rows, 4)
                unfolded = folded.reshape(kv, rows, 6, 4).permute(1, 0, 2, 3).reshape(rows, head_rows, 4)
                built = torch.zeros_like(unfolded)
                for token in range(rows):
                    for row in range(32):
                        source = self.inverse_source(rows, head_rows, token, row)
                        if source is not None:
                            built[token, row] = folded[source[0] * 32 + source[1]]
                self.assertTrue(torch.equal(built, unfolded), (head_rows, rows))

    def test_at_one_kv_head_the_fold_is_a_reshape(self):
        query = torch.arange(5 * 6 * 4, dtype=torch.float32).reshape(1, 5, 6, 4)
        self.assertTrue(torch.equal(query.reshape(5, 1, 6, 4).permute(1, 0, 2, 3).reshape(30, 4), query.reshape(30, 4)))

    def test_source_row_agrees_with_the_pinned_oracle_at_the_pair_and_the_index_map_at_four_cards(self):
        import attention_fold_dma
        with pair():
            for rows in (1, 4, 8):
                for inverse in (False, True):
                    for index in range(rows * 12):
                        self.assertEqual(attention_fold_dma_tp.source_row(rows, index, inverse=inverse),
                                         attention_fold_dma.source_row(rows, index, inverse=inverse))
        with four():
            for rows in (1, 4, 8):
                self.assertEqual([attention_fold_dma_tp.source_row(rows, index) for index in range(rows * 6)],
                                 list(range(rows * 6)))
                self.assertEqual([attention_fold_dma_tp.source_row(rows, index, inverse=True)
                                  for index in range(rows * 6)], list(range(rows * 6)))

    def test_the_sibling_kernels_are_the_pinned_ones_with_the_twelve_replaced(self):
        cases = {'attention_mask_replay': [('rows * QWEN_FOLD_HEAD_ROWS', 'rows * 12')],
                 'attention_fold_dma': [('(rows * QWEN_FOLD_HEAD_ROWS + 31) / 32', '(rows * 12 + 31) / 32'),
                                        ('target_row >= QWEN_FOLD_HEAD_ROWS', 'target_row >= 12'),
                                        ('head >= rows * QWEN_FOLD_HEAD_ROWS', 'head >= rows * 12')]}
        for stem, subs in cases.items():
            with self.subTest(kernel=stem):
                original = (HERE / (stem + '.cpp')).read_text()
                sibling = (HERE / (stem + '_tp.cpp')).read_text()
                lines = sibling.split('\n')
                self.assertTrue(lines[0].startswith('// Four-card sibling of ' + stem))
                body = '\n'.join(lines[2:])
                guard = ('#ifndef QWEN_FOLD_HEAD_ROWS\n#error "QWEN_FOLD_HEAD_ROWS is defined by the launch builder '
                         '(tp_kernels.fold_defines)"\n#endif\n')
                self.assertTrue(body.startswith(guard))
                body = body[len(guard):]
                for macro, literal in subs:
                    body = body.replace(macro, literal)
                self.assertEqual(body, original)

    def test_kernel_source_and_defines_follow_the_width(self):
        with pair():
            self.assertEqual(tp_kernels.fold_defines(), [])
        with four():
            self.assertEqual(tp_kernels.fold_defines(), [('QWEN_FOLD_HEAD_ROWS', '6')])
            self.assertEqual(tp_kernels.source(HERE / 'attention_fold_dma.cpp'), str(HERE / 'attention_fold_dma_tp.cpp'))


def program_ttnn(chips):
    fake = fake_ttnn()
    fake.get_device_tensors = lambda tensor: tensor.shards
    return fake


class ProgramTests(unittest.TestCase):
    def tensors(self, chips, width=K, batches=2, rows=8, head_rows=6):
        shards = lambda base: [FakeShard(base + 0x8000 * chip) for chip in range(chips)]
        positions = FakeTensor((8,), 'int32', 'row_major', 'dram', shards(0x100))
        mask = FakeTensor((batches, 1, rows * head_rows, width), 'bf16', 'tile', 'dram', shards(0x200))
        return positions, mask

    def test_the_mask_program_covers_four_chips_with_the_sibling_kernel(self):
        with patch.dict(sys.modules, {'ttnn': program_ttnn(4)}), four():
            positions, mask = self.tensors(4)
            program = quad.prepare_narrow(MESH, positions, mask, rows=8, batches=2, offset=8)
        view = program_view(program)
        self.assertEqual(sorted(view), [((0, chip), (0, chip)) for chip in range(4)])
        self.assertEqual(len(view), 4)
        for chip, key in enumerate(sorted(view)):
            (kernel_source, cores, compile_args, config, runtime), buffers = view[key][0], view[key][-1]
            self.assertEqual(kernel_source, str(Path(attention_mask_replay.__file__).with_name('attention_mask_replay_tp.cpp')))
            tasks = [arguments for column in runtime.values() for arguments in column.values()]
            self.assertEqual(len(tasks), 2 * 2 * 8, 'two batches x two head tiles x eight column tiles')
            self.assertEqual({tuple(task[2:5]) for task in tasks}, {(8, K, 8)})
            self.assertEqual({task[0] for task in tasks}, {0x100 + 0x8000 * chip})
        kernels = [value.kernels[0] for value in program.values()]
        self.assertTrue(all(dict(kernel.defines) == {'QWEN_FOLD_HEAD_ROWS': '6'} for kernel in kernels))

    def test_the_pair_program_still_names_the_pinned_kernel_with_no_defines(self):
        with patch.dict(sys.modules, {'ttnn': program_ttnn(2)}), pair():
            positions, mask = self.tensors(2, head_rows=12)
            program = quad.prepare_narrow(MESH, positions, mask, rows=8, batches=2, offset=0)
        for value in program.values():
            self.assertEqual(value.kernels[0].kernel_source, str(Path(attention_mask_replay.__file__).with_suffix('.cpp')))
            self.assertEqual(value.kernels[0].defines, [])

    def test_a_pair_sized_mask_is_refused_at_four_cards(self):
        with patch.dict(sys.modules, {'ttnn': program_ttnn(4)}), four():
            positions, mask = self.tensors(4, head_rows=12)
            with self.assertRaisesRegex(ValueError, 'narrow attention mask'):
                quad.prepare_narrow(MESH, positions, mask, rows=8, batches=2, offset=0)
            positions, mask = self.tensors(2)
            with self.assertRaisesRegex(ValueError, 'Four chip-local metadata buffers required'):
                quad.prepare_narrow(MESH, positions, mask, rows=8, batches=2, offset=0)


class FoldLaunchTests(unittest.TestCase):
    def test_the_fold_launch_is_four_chip_programs_with_the_head_row_define(self):
        class Kernel(SimpleNamespace):
            pass

        record = []
        fake = fake_ttnn()
        fake.get_device_tensors = lambda tensor: tensor.shards
        fake.empty = lambda shape, **keywords: FakeTensor(shape, 'bf16', 'tile', 'dram',
                                                         [FakeShard(0x9000 + 0x100 * chip) for chip in range(4)])
        fake.generic_op = lambda tensors, program: record.append((tensors, program))
        source = FakeTensor((1, 16, 6, 256), 'bf16', 'tile', 'dram', [FakeShard(0x100 * (chip + 1)) for chip in range(4)])
        fake.L1_MEMORY_CONFIG = 'l1'
        owned = []
        with patch.dict(sys.modules, {'ttnn': fake}), four():
            output = attention_fold_dma_tp.device_layout_dma(MESH, source, 8, owned, offset=8)
        self.assertEqual(tuple(output.shape), (1, 1, 48, 256))
        program = record[0][1]
        self.assertEqual(len(program), 4)
        for chip, key in enumerate(sorted(program)):
            kernel = program[key].kernels[0]
            self.assertEqual(kernel.kernel_source, str(HERE / 'attention_fold_dma_tp.cpp'))
            self.assertEqual(dict(kernel.defines), {'QWEN_FOLD_HEAD_ROWS': '6'})
            tasks = [arguments for column in kernel.runtime_args.values() for arguments in column.values()]
            self.assertEqual(len(tasks), 16, 'two output tiles x eight column tiles')
            self.assertEqual({task[0] for task in tasks}, {0x100 * (chip + 1)})
            self.assertEqual({tuple(task[2:5]) for task in tasks}, {(8, 0, 8)})

    def test_a_pair_query_is_refused_at_four_cards(self):
        fake = fake_ttnn()
        source = FakeTensor((1, 16, 12, 256), 'bf16', 'tile', 'dram', [FakeShard(0x100 * (chip + 1)) for chip in range(4)])
        with patch.dict(sys.modules, {'ttnn': fake}), four():
            with self.assertRaisesRegex(ValueError, 'Native tiled TP4 query geometry'):
                attention_fold_dma_tp.device_layout_dma(MESH, source, 8, [], offset=0)


def fake_fold(device):
    def permute(mesh, source, rows, owned, *, inverse=False, offset=0):
        output = device.device_tensor((1, rows, 6, 256) if inverse else (1, 1, rows * 6, 256), 'bf16', 'tile', 'dram',
                                      name='fold', origin=('fold', source, inverse, offset))
        owned.append(output)
        return output
    return permute


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.device = FourChipDevice()
        self.harness = Harness(self)

    def test_construction_accepts_0x23_on_a_four_chip_mesh_and_stages_for_its_start(self):
        reader, ((table, cur_pos),) = segment(self.device, start=4200)
        word, position = pinned.extent_values(4200)
        self.assertEqual(reader.positions.value.tolist(), [word] + [0] * 7)
        self.assertEqual(cur_pos.value.tolist(), [position, position])
        self.assertEqual(tuple(reader.metadata[0][2].shape), (2, 1, 48, K), 'the narrow tail mask: 8 rows x 6 heads')
        self.assertEqual(list(reader.sdpa_modes_applied), [0x23])
        self.assertEqual(quad.EXTENT_FLAGS, 0x23)
        self.assertEqual(self.harness.programs[0]['rows'], 8)
        self.assertEqual(len(reader.positions.shards), 4)

    def test_the_slice_mode_is_dropped_at_one_kv_head_not_served(self):
        # QWEN_FAST_SDPA_MODES names slice (the pair's 0x27): at one KV head there is nothing to slice between, so the
        # flag is not built and the reader still qualifies at 0x23.
        reader, _ = segment(self.device, start=4200)
        self.assertEqual(list(reader.sdpa_modes_applied), [0x23])

    def test_a_pair_reader_geometry_is_refused_by_its_query_shape(self):
        reader, _ = segment(self.device, start=4200)
        with self.assertRaisesRegex(ValueError, 'Replay query geometry changed'):
            reader(SimpleNamespace(shape=(1, 16, 12, 256)), object(), object(), scale=1.0, memory_config='dram')
        self.assertTrue(reader.failed is False)

    def test_a_call_folds_every_group_and_reads_the_lent_cur_pos_and_the_narrow_mask(self):
        reader, ((table, cur_pos),) = segment(self.device, start=4200)
        query = self.device.device_tensor((1, 16, 6, 256), 'bf16', 'tile', 'dram', name='query')
        keys = self.device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys')
        with patch('extent_attention_replay_tp.device_layout_dma', side_effect=fake_fold(self.device)) as fold, \
                patch('extent_attention_replay_tp.attention_mask_replay.execute') as refresh:
            output = reader(query, keys, keys, scale=0.0625, memory_config='dram')
        self.assertEqual(refresh.call_count, 1)
        offsets = [call.kwargs.get('offset') for call in fold.call_args_list if not call.kwargs.get('inverse')]
        self.assertEqual(offsets, [0, 8])
        (options,) = self.device.sdpa
        self.assertIs(options['cur_pos_tensor'], cur_pos)
        self.assertIs(options['page_table_tensor'], reader.metadata[0][1])
        self.assertEqual(options['attn_mask'].shape, (2, 1, 48, K))
        self.assertEqual(tuple(output.shape), (1, 16, 6, 256))
        self.assertFalse(options['is_causal'])

    def test_the_block_reader_takes_a_six_head_query_and_slices_each_user(self):
        device = self.device
        segments = ((0, 16), (16, 32))
        lent = [lend(device), lend(device)]
        block = quad.PackedExtentReplayReader(device, MESH, segments, WIDTH, [host_table(1), host_table(2)],
                                              storage=lent, max_group_rows=8, starts=(4200, 9000))
        self.assertEqual(block.starts, (4200, 9000))
        query = device.device_tensor((1, 32, 6, 256), 'bf16', 'tile', 'dram', name='query')
        keys = device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys')
        with patch('extent_attention_replay_tp.device_layout_dma', side_effect=fake_fold(device)), \
                patch('extent_attention_replay_tp.attention_mask_replay.execute'):
            output = block(query, keys, keys, scale=0.0625, memory_config='dram')
        self.assertEqual(tuple(output.shape), (1, 32, 6, 256))
        self.assertEqual(len(device.sdpa), 2)
        with self.assertRaisesRegex(ValueError, 'Packed extent query geometry changed'):
            block(device.device_tensor((1, 32, 12, 256), 'bf16', 'tile', 'dram', name='q12'), keys, keys,
                  scale=0.0625, memory_config='dram')

    def test_independent_storage_is_checked_on_all_four_chips(self):
        first = self.device.device_tensor((1,), 'int32', 'row_major', 'dram')
        second = self.device.device_tensor((1,), 'int32', 'row_major', 'dram')
        with four():
            quad.independent(self.device, [first, second], 'Extent storage')
            second.shards[3] = first.shards[3]
            with self.assertRaisesRegex(ValueError, 'chip 3'):
                quad.independent(self.device, [first, second], 'Extent storage')


class SeamTests(unittest.TestCase):
    def test_install_puts_the_twin_where_the_lazy_imports_look_and_uninstall_puts_it_back(self):
        self.addCleanup(tp_addresses.uninstall)
        holder = SimpleNamespace(extent_attention_replay=pinned)
        sys.modules['tp_seam_holder'] = holder
        self.addCleanup(sys.modules.pop, 'tp_seam_holder', None)
        with four():
            tp_addresses.install()
            import extent_attention_replay as seen
            self.assertIs(seen, quad)
            self.assertIs(sys.modules['extent_attention_replay'], quad)
            self.assertTrue(hasattr(seen, 'PackedExtentReplayReader') and hasattr(seen, 'narrow_mask_host'))
            self.assertEqual(tp_addresses.install(), 0)
        tp_addresses.uninstall()
        self.assertIs(sys.modules['extent_attention_replay'], pinned)

    def test_the_pair_keeps_the_pinned_module(self):
        with pair():
            with self.assertRaises(ValueError):
                tp_addresses.install()
        self.assertIs(sys.modules['extent_attention_replay'], pinned)

    def test_every_name_the_serving_modules_read_from_the_module_exists_in_the_twin(self):
        for name in ('PackedExtentReplayReader', 'ExtentSegmentReader', 'extent', 'extent_values', 'admits', 'accept_limit',
                     'EXTENT_GROUP_ROWS', 'MIN_LIVE_START', 'narrow_mask_host', 'prepare_narrow', 'ENGAGED_MARKER',
                     'LAYOUT', 'EXTENT_BUNDLE_ENTRIES', 'device_layout_dma', 'attention_mask_replay', 'mask_tiles',
                     'replay_mask_host', 'execute_extent', 'validate_extent_storage'):
            self.assertTrue(hasattr(quad, name), name)


if __name__ == '__main__':
    unittest.main()
