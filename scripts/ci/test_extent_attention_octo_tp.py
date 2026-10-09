"""The octo-T8 extent readers (extent_attention_octo_tp): ONE eight-row group per bundle, K64j flags 0x21, on the four-chip fake device.

What this proves, and what it cannot. The pinned twin (extent_attention_replay_tp, held unedited by the four-card evidence) refuses an eight-row segment; the octo
reader accepts exactly that and nothing else, runs the SAME word / cur_pos / narrow-mask / staging / call code as the pinned segment reader, and asks the SDPA op for a
bundle of one entry at flags 0x21. The block fold (QWEN_FAST_TP4_ATTN_FOLD) is shown byte for byte equal to the served composition at eight segments of one group, by
the kernel-emulating oracle of test_tp4_vglue_attention. Nothing here is a kernel result: whether K64j's 0x21 program at a bundle of ONE eight-row group equals the native
row-by-row decode is the qualification job Q1 (k64j_card_b.py combo G8B1:0x21, extent_reader_card_b.py --octo)."""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import attention_block_fold_tp as fold
import extent_attention_fold_tp as folded
import extent_attention_octo_tp as octo
import extent_attention_replay as pinned
import extent_attention_replay_tp as quad
import pooled_attention_replay as pooled
import tp_addresses
from test_extent_attention_replay import MESH, WIDTH, host_table
from test_extent_attention_replay_tp import FourChipDevice, Harness, K, fake_fold, four, lend
from test_tp4_vglue_attention import (COLUMNS, HEAD, chunks_for, from_pages, run_launch, sdpa_stub, served, to_pages, user_of)

SEGMENTS = tuple((8 * user, 8 * user + 8) for user in range(8))


def segment(device, start=4200, **options):
    pairs = lend(device, 8, WIDTH)
    options.setdefault('max_group_rows', 8)
    return octo.OctoSegmentReader(device, MESH, 8, WIDTH, host_table(5, WIDTH), storage=pairs, start=start, **options), pairs


class GeometryTests(unittest.TestCase):
    def test_the_octo_segment_is_one_bundle_of_one_eight_row_group_and_the_m3_segment_two_groups(self):
        self.assertEqual(octo.bundles_of(8, 8), pinned.LAYOUT(8, 8))
        self.assertEqual([[group['rows'] for group in bundle] for bundle in octo.bundles_of(8, 8)], [[8]])
        self.assertEqual([[group['rows'] for group in bundle] for bundle in pinned.LAYOUT(16, 8)], [[8, 8]])
        with self.assertRaisesRegex(ValueError, 'G8B1 only'):
            octo.bundles_of(16, 8)
        with self.assertRaisesRegex(ValueError, 'G8B1 only'):
            octo.bundles_of(8, 4)

    def test_the_octo_flag_set_is_tail_and_extent_with_no_share_and_no_slice(self):
        self.assertEqual(octo.OCTO_FLAGS, 0x21)
        self.assertEqual(octo.OCTO_FLAGS, quad.EXTENT_FLAGS & ~pooled.QWEN_KV_SHARE, 'the pinned twin\'s 0x23 less the share bit')
        modes = frozenset({'tail', 'share', 'slice', 'extent'})
        with four():
            self.assertEqual(pooled.mode_flags(modes, 1, 8), 0x21, 'one entry has nothing to share and one KV head nothing to slice')
            self.assertEqual(pooled.mode_flags(modes, 2, 8), 0x23, 'the M3 bundle is unchanged')

    def test_a_block_is_octo_only_when_every_segment_is_eight_rows(self):
        self.assertTrue(octo.is_octo_block(SEGMENTS))
        for segments in (((0, 16), (16, 32), (32, 48), (48, 64)), ((0, 32), (32, 64)), ((0, 8), (8, 24)), (), None, 5):
            self.assertFalse(octo.is_octo_block(segments), segments)


class SegmentReaderTests(unittest.TestCase):
    def setUp(self):
        self.device = FourChipDevice()
        self.harness = Harness(self)

    def test_construction_runs_flags_0x21_on_a_bundle_of_one_and_stages_for_its_start(self):
        reader, ((table, cur_pos),) = segment(self.device, start=4200)
        word, position = pinned.extent_values(4200)
        self.assertEqual(reader.positions.value.tolist(), [word] + [0] * 7)
        self.assertEqual(cur_pos.value.tolist(), [position], 'one entry, one word')
        self.assertEqual((tuple(table.shape), tuple(cur_pos.shape)), ((1, WIDTH), (1,)))
        self.assertEqual(tuple(reader.metadata[0][2].shape), (1, 1, 48, K), 'the narrow tail mask of one eight-row group of six heads')
        self.assertEqual(list(reader.sdpa_modes_applied), [0x21])
        self.assertEqual((self.harness.programs[0]['rows'], self.harness.programs[0]['batches']), (8, 1))
        self.assertEqual(len(reader.positions.shards), 4, 'four chips')
        self.assertTrue(getattr(reader, 'runtime_extent', False), 'the attach admission asks every segment reader for it')

    def test_the_pinned_twin_still_refuses_an_eight_row_segment_and_its_bytes_are_untouched(self):
        pairs = lend(self.device, 8, WIDTH)
        with self.assertRaisesRegex(ValueError, r'qualified at G8B2 only \(0x23 at one KV head\): a 8-row segment bundles as \[\[8\]\]'):
            quad.ExtentSegmentReader(self.device, MESH, 8, WIDTH, host_table(5, WIDTH), storage=pairs, start=4200, max_group_rows=8)
        self.assertEqual(quad.EXTENT_FLAGS, 0x23)
        self.assertEqual(pinned.EXTENT_BUNDLE_ENTRIES, 2)

    def test_every_other_geometry_is_refused_before_anything_is_allocated(self):
        for name, rows, group_rows, text in (('M3 rows', 16, 8, 'explicit T8 segment'),
                                             ('T32 rows', 32, 8, 'explicit T8 segment'),
                                             ('four-row groups', 8, 4, 'eight-row groups only')):
            with self.subTest(name):
                device = FourChipDevice()
                pairs = lend(device, 16 if rows == 16 else 8, WIDTH)
                before = len(device.live)
                with self.assertRaisesRegex(ValueError, text):
                    octo.OctoSegmentReader(device, MESH, rows, WIDTH, host_table(5, WIDTH), storage=pairs, start=4200, max_group_rows=group_rows)
                self.assertEqual(len(device.live), before, 'nothing allocated')

    def test_the_g8_preconditions_are_the_pinned_readers(self):
        pairs = lend(self.device, 8, WIDTH)
        with patch.dict(os.environ, {pinned.TREE_SCRATCH_ENV: '0'}), self.assertRaisesRegex(ValueError, 'compact native scratch'):
            octo.OctoSegmentReader(self.device, MESH, 8, WIDTH, host_table(5, WIDTH), storage=pairs, start=4200, max_group_rows=8)
        with patch.dict(os.environ, {'QWEN_FAST_SDPA_MODES': 'share'}), self.assertRaisesRegex(ValueError, 'include tail'):
            octo.OctoSegmentReader(self.device, MESH, 8, WIDTH, host_table(5, WIDTH), storage=pairs, start=4200, max_group_rows=8)
        with self.assertRaisesRegex(ValueError, 'one complete native cache page table'.lower().replace('one', 'One')):
            octo.OctoSegmentReader(self.device, MESH, 8, WIDTH, torch.zeros(2, WIDTH, dtype=torch.int32), storage=pairs, start=4200, max_group_rows=8)

    def test_a_pair_sized_storage_for_two_groups_is_refused_for_a_bundle_of_one(self):
        pairs = lend(self.device, 16, WIDTH)         # (2, W) table and (2,) cur_pos: the M3 segment's
        with self.assertRaisesRegex(ValueError, r'Extent page table must be a row-major int32 \(1, %d\)' % WIDTH):
            octo.OctoSegmentReader(self.device, MESH, 8, WIDTH, host_table(5, WIDTH), storage=pairs, start=4200, max_group_rows=8)

    def test_a_call_folds_one_group_and_reads_the_lent_cur_pos_and_the_narrow_mask(self):
        reader, ((table, cur_pos),) = segment(self.device, start=4200)
        query = self.device.device_tensor((1, 8, 6, 256), 'bf16', 'tile', 'dram', name='query')
        keys = self.device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys')
        with patch('extent_attention_replay_tp.device_layout_dma', side_effect=fake_fold(self.device)) as folding, \
                patch('extent_attention_replay_tp.attention_mask_replay.execute') as refresh:
            output = reader(query, keys, keys, scale=0.0625, memory_config='dram')
        self.assertEqual(refresh.call_count, 1)
        offsets = [call.kwargs.get('offset') for call in folding.call_args_list if not call.kwargs.get('inverse')]
        self.assertEqual(offsets, [0])
        (options,) = self.device.sdpa
        self.assertIs(options['cur_pos_tensor'], cur_pos)
        self.assertIs(options['page_table_tensor'], reader.metadata[0][1])
        self.assertEqual(options['attn_mask'].shape, (1, 1, 48, K))
        self.assertEqual(tuple(output.shape), (1, 8, 6, 256))
        self.assertFalse(options['is_causal'])
        self.assertEqual(options['program_config'].q_chunk_size, pooled.QWEN_DECODE_MAGIC | 0x21, 'the sentinel carries the octo flags')

    def test_a_sixteen_row_query_is_refused(self):
        reader, _ = segment(self.device, start=4200)
        with self.assertRaisesRegex(ValueError, 'Replay query geometry changed'):
            reader(SimpleNamespace(shape=(1, 16, 6, 256)), object(), object(), scale=1.0, memory_config='dram')

    def test_staging_writes_the_word_and_one_cur_pos_and_the_tables(self):
        reader, ((table, cur_pos),) = segment(self.device, start=4200)
        reader.stage(70000, table=host_table(9, WIDTH))
        word, position = pinned.extent_values(70000)
        self.assertEqual(reader.positions.value.tolist()[0], word)
        self.assertEqual(cur_pos.value.tolist(), [position])
        self.assertEqual(table.value.tolist(), host_table(9, WIDTH)[:, :WIDTH].tolist())
        self.assertEqual(reader.start, 70000)


class PackedReaderTests(unittest.TestCase):
    def setUp(self):
        self.device = FourChipDevice()
        self.harness = Harness(self)

    def block(self, starts=(4200,) * 8, cls=folded.PackedExtentReplayReader, **options):
        lent = [lend(self.device, 8, WIDTH) for user in range(8)]
        return cls(self.device, MESH, SEGMENTS, WIDTH, [host_table(user + 1, WIDTH) for user in range(8)], storage=lent,
                   max_group_rows=8, starts=starts, **options), lent

    def test_the_fold_twin_builds_the_octo_readers_for_eight_row_segments_and_the_pinned_ones_for_sixteen(self):
        block, lent = self.block(starts=tuple(4200 + 100 * user for user in range(8)))
        self.assertEqual([type(reader) for reader in block.readers], [octo.OctoSegmentReader] * 8)
        self.assertEqual(block.starts, tuple(4200 + 100 * user for user in range(8)))
        self.assertEqual([list(reader.sdpa_modes_applied) for reader in block.readers], [[0x21]] * 8)
        self.assertEqual((block.rows, block.capacity, block.page_width), (64, WIDTH * 64, WIDTH))
        self.assertEqual(len(block.borrowed), 16, 'a table and a cur_pos per user')
        engaged = [line for line in self.harness.logs if line.startswith(pinned.ENGAGED_MARKER)]
        self.assertEqual(len(engaged), 1)
        self.assertIn('segments=8 flags=' + ','.join(['0x21'] * 8), engaged[0])
        self.assertIn('geometry=G8B1', engaged[0])
        import octo_judge

        self.assertTrue(octo_judge.ENGAGED_OCTO.search(engaged[0]), 'the line the judge reads is the line the reader writes: ' + engaged[0])
        # the M3 block's sixteen-row segments: the pinned constructor, the pinned flags
        m3 = folded.PackedExtentReplayReader(self.device, MESH, ((0, 16), (16, 32), (32, 48), (48, 64)), WIDTH,
                                             [host_table(1, WIDTH)] * 4, storage=[lend(self.device, 16, WIDTH) for user in range(4)],
                                             max_group_rows=8, starts=(4200,) * 4)
        self.assertEqual({type(reader) for reader in m3.readers}, {quad.ExtentSegmentReader})
        self.assertEqual([list(reader.sdpa_modes_applied) for reader in m3.readers], [[0x23]] * 4)

    def test_the_pinned_block_reader_refuses_the_octo_segments(self):
        with self.assertRaisesRegex(ValueError, 'qualified at G8B2 only'):
            self.block(cls=quad.PackedExtentReplayReader)

    def test_a_mixed_block_is_not_an_octo_block(self):
        lent = [lend(self.device, 8, WIDTH), lend(self.device, 16, WIDTH)]
        with self.assertRaises(ValueError):
            folded.PackedExtentReplayReader(self.device, MESH, ((0, 8), (8, 24)), WIDTH, [host_table(1, WIDTH)] * 2, storage=lent,
                                            max_group_rows=8, starts=(4200, 4200))

    def test_a_bad_start_or_storage_refuses_the_block_before_any_reader_is_built(self):
        lent = [lend(self.device, 8, WIDTH) for user in range(8)]
        before = len(self.device.live)
        with self.assertRaises(ValueError):
            folded.PackedExtentReplayReader(self.device, MESH, SEGMENTS, WIDTH, [host_table(1, WIDTH)] * 8, storage=lent,
                                            max_group_rows=8, starts=(4200,) * 7 + (WIDTH * 64,))
        self.assertEqual(len(self.device.live), before)
        lent[3] = lend(self.device, 16, WIDTH)
        before = len(self.device.live)
        with self.assertRaisesRegex(ValueError, 'Extent page table must be'):
            folded.PackedExtentReplayReader(self.device, MESH, SEGMENTS, WIDTH, [host_table(1, WIDTH)] * 8, storage=lent,
                                            max_group_rows=8, starts=(4200,) * 8)
        self.assertEqual(len(self.device.live), before)

    def test_a_block_call_slices_each_user_and_runs_eight_launches_of_one_entry(self):
        block, lent = self.block()
        query = self.device.device_tensor((1, 64, 6, 256), 'bf16', 'tile', 'dram', name='query')
        keys = self.device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys')
        with patch('extent_attention_replay_tp.device_layout_dma', side_effect=fake_fold(self.device)), \
                patch('extent_attention_replay_tp.attention_mask_replay.execute'):
            output = block(query, keys, keys, scale=0.0625, memory_config='dram')
        self.assertEqual(tuple(output.shape), (1, 64, 6, 256))
        self.assertEqual(len(self.device.sdpa), 8)
        self.assertEqual([options['attn_mask'].shape for options in self.device.sdpa], [(1, 1, 48, K)] * 8)
        self.assertEqual([options['cur_pos_tensor'] for options in self.device.sdpa], [pairs[0][1] for pairs in lent])
        with self.assertRaisesRegex(ValueError, 'Packed extent query geometry changed'):
            block(self.device.device_tensor((1, 64, 12, 256), 'bf16', 'tile', 'dram', name='q12'), keys, keys, scale=0.0625, memory_config='dram')

    def test_the_block_stages_each_segments_word_and_cur_pos_and_closes_clean(self):
        block, lent = self.block()
        block.stage(tuple(1000 + 256 * user for user in range(8)))
        for user, (reader, pairs) in enumerate(zip(block.readers, lent)):
            word, position = pinned.extent_values(1000 + 256 * user)
            self.assertEqual(reader.positions.value.tolist()[0], word)
            self.assertEqual(pairs[0][1].value.tolist(), [position])
        block.close()
        self.assertTrue(block.closed)
        self.assertTrue(all(reader.closed and reader.owned == [] for reader in block.readers), 'the readers\' own word and masks are released')
        block.close()                                   # closing twice is a no-op

    def test_the_audit_property_is_refused_like_the_pinned_readers(self):
        block, _ = self.block()
        self.assertIsNone(block.audit)
        with self.assertRaisesRegex(ValueError, 'takes no attention audit'):
            block.audit = object()


class BlockFoldTests(unittest.TestCase):
    """QWEN_FAST_TP4_ATTN_FOLD at eight segments of one group: the kernel's index map, emulated on the planner's own runtime arguments, against the served composition."""

    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_fold_accepts_bundles_of_one_and_two_eight_row_groups_and_nothing_else(self):
        chunks = chunks_for(SEGMENTS)
        self.assertEqual([chunk.batches for chunk in chunks], [1] * 8)
        ttnn = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram')
        query = SimpleNamespace(shape=(1, 64, HEAD, 256), dtype='bf16', layout='tile', memory_config=lambda: 'dram')
        self.assertIsNone(fold.problem(query, chunks, 'dram', ttnn))
        self.assertIsNone(fold.problem(query, chunks_for(((0, 16), (16, 32), (32, 48), (48, 64))), 'dram', ttnn))
        three = SimpleNamespace(shape=(1, 24, HEAD, 256), dtype='bf16', layout='tile', memory_config=lambda: 'dram')
        self.assertIn('one or two eight-row groups', fold.problem(three, [fold.Chunk(0, [(0, 8), (8, 8), (16, 8)])], 'dram', ttnn) or '')
        four_rows = SimpleNamespace(shape=(1, 4, HEAD, 256), dtype='bf16', layout='tile', memory_config=lambda: 'dram')
        self.assertIn('one or two eight-row groups', fold.problem(four_rows, [fold.Chunk(0, [(0, 4)])], 'dram', ttnn) or '')

    def test_the_kernel_composition_is_the_served_composition_byte_for_byte_at_eight_users_of_one_group(self):
        total = 64
        query = torch.arange(total * HEAD * 256, dtype=torch.float32).reshape(1, total, HEAD, 256) + 0.5
        chunks = chunks_for(SEGMENTS)
        expected = served(query, SEGMENTS, sdpa_stub)
        buffers = {1: to_pages(query)}
        stacked_addresses = [100 + index for index in range(len(chunks))]
        for address in stacked_addresses:
            buffers[address] = {}
        flat = fold.flat_tasks(fold.forward_tasks(chunks), [1] * len(chunks), stacked_addresses)
        run_launch(fold.runtime_arguments(False, fold.distribute(flat, 110)), buffers)
        for index, chunk in enumerate(chunks):
            stack, clean = from_pages(buffers[stacked_addresses[index]], chunk.batches, chunk.rows * HEAD)
            self.assertTrue(clean, 'the stacked query padding is zero, as the served fold leaves it')
            first, last = SEGMENTS[user_of(SEGMENTS, chunk)]
            served_stack = query[:, first:last].reshape(1, 1, 8 * HEAD, 256)
            self.assertTrue(torch.equal(stack, served_stack), index)
            buffers[200 + index] = to_pages(sdpa_stub(user_of(SEGMENTS, chunk), stack))
        buffers[300] = {}
        flat = fold.flat_tasks(fold.inverse_tasks(chunks), [200 + index for index in range(len(chunks))], [300] * len(chunks))
        run_launch(fold.runtime_arguments(True, fold.distribute(flat, 110)), buffers)
        built, clean = from_pages(buffers[300], total, HEAD)
        self.assertTrue(clean)
        self.assertTrue(torch.equal(built, expected))
        self.assertEqual(sorted(buffers[300]), list(range(total * COLUMNS)), 'every page of the output is written exactly once')

    def test_task_counts_match_the_m3_blocks_so_the_launch_costs_the_same(self):
        octo_tasks = [len(part) for part in fold.forward_tasks(chunks_for(SEGMENTS))], [len(part) for part in fold.inverse_tasks(chunks_for(SEGMENTS))]
        m3 = chunks_for(((0, 16), (16, 32), (32, 48), (48, 64)))
        m3_tasks = [len(part) for part in fold.forward_tasks(m3)], [len(part) for part in fold.inverse_tasks(m3)]
        self.assertEqual(sum(octo_tasks[0]), sum(m3_tasks[0]), 'the same 128 fold-in pages')
        self.assertEqual(sum(octo_tasks[1]), sum(m3_tasks[1]), 'the same 512 fold-out pages')


class SeamTests(unittest.TestCase):
    def test_install_binds_the_fold_twin_that_dispatches_to_the_octo_readers_when_the_octo_flag_is_set(self):
        self.addCleanup(tp_addresses.uninstall)
        with four(), patch.dict(os.environ, {'QWEN_FAST_OCTO': 'alternate'}):
            tp_addresses.install()
            import extent_attention_replay as seen

            self.assertIs(seen, quad)
            self.assertIs(seen.PackedExtentReplayReader, folded.PackedExtentReplayReader, 'model_batch reaches the fold twin, which dispatches by segment width')
        tp_addresses.uninstall()
        self.assertIs(sys.modules['extent_attention_replay'], pinned)

    def test_with_no_fold_flag_and_no_octo_flag_the_pinned_reader_stays_bound(self):
        self.addCleanup(tp_addresses.uninstall)
        with four(), patch.dict(os.environ, {'QWEN_FAST_OCTO': 'off'}):
            tp_addresses.install()
            import extent_attention_replay as seen

            self.assertIs(seen.PackedExtentReplayReader, quad.PackedExtentReplayReader)
            self.assertIsNot(seen.PackedExtentReplayReader, folded.PackedExtentReplayReader)
        tp_addresses.uninstall()

    def test_the_octo_flag_is_the_only_new_reason_to_bind_the_twin(self):
        self.assertTrue(tp_addresses._octo({'QWEN_FAST_OCTO': 'live'}))
        self.assertTrue(tp_addresses._octo({'QWEN_FAST_OCTO': 'alternate'}))
        self.assertFalse(tp_addresses._octo({}))
        self.assertFalse(tp_addresses._octo({'QWEN_FAST_OCTO': 'off'}))

    def test_the_octo_module_is_a_twin_and_does_not_edit_the_pinned_modules(self):
        import hashlib
        import json
        from pathlib import Path

        here = Path(__file__).resolve().parent
        evidence = json.loads((here / 'packed_any_evidence_tp4.json').read_text())
        recorded = evidence['sources']['extent_attention_replay_tp.py']
        digest = hashlib.sha256((here / 'extent_attention_replay_tp.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        self.assertEqual(digest, recorded, 'the reader twin the four-card evidence qualified is byte for byte what it was')


if __name__ == '__main__':
    unittest.main()
