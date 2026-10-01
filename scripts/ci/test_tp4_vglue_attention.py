"""V3a (QWEN_FAST_TP4_ATTN_FOLD): the block fold-in / fold-out launches against the served per-segment composition.

The kernel (attention_block_fold_tp.cpp) is emulated from its own loops over the very runtime arguments the host planner
writes, on 32x32 tiles in the TILE layout's page order, and compared with the served composition (query Slice, the fold,
the stacking Concat, the result Slices, the inverse fold, the Concats) computed on logical tensors. Nothing here can tell
the two apart except a wrong page, row or task: every value is distinct.

Run at py 3.11: `py -3.11 -m unittest test_tp4_vglue_attention` from scripts/ci.
"""

from collections import defaultdict
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import attention_block_fold_tp as fold
import extent_attention_replay as pinned
import extent_attention_fold_tp as folded
import extent_attention_replay_tp as quad
import tp4_vglue
from test_extent_attention_replay import WIDTH, host_table
from test_extent_attention_replay_tp import FourChipDevice, Harness, MESH, four, lend, pair

HERE = Path(__file__).resolve().parent
HEAD = 6
COLUMNS = 8


def env(**values):
    return patch.dict(os.environ, values)


# --- tiles -----------------------------------------------------------------------------------------------------------

def to_pages(tensor):
    """(1, B, H, 256) logical -> {page: 32x32 tile}, pages in the TILE layout's order (batch, tile row, tile column)."""
    _, batches, height, width = tensor.shape
    rows = -(-height // 32)
    pages = {}
    for batch in range(batches):
        padded = torch.zeros(rows * 32, width, dtype=tensor.dtype)
        padded[:height] = tensor[0, batch]
        for row in range(rows):
            for column in range(width // 32):
                pages[(batch * rows + row) * (width // 32) + column] = \
                    padded[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32].clone()
    return pages


def from_pages(pages, batches, height, width=256):
    rows = -(-height // 32)
    out = torch.zeros(1, batches, height, width)
    padding_clean = True
    for batch in range(batches):
        padded = torch.zeros(rows * 32, width)
        for row in range(rows):
            for column in range(width // 32):
                padded[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32] = \
                    pages[(batch * rows + row) * (width // 32) + column]
        out[0, batch] = padded[:height]
        padding_clean = padding_clean and not bool(padded[height:].any())
    return out, padding_clean


def kernel_task(inverse, rows, tiles_at, source_base, destination_base, task):
    """attention_block_fold_tp.cpp's per-task body, statement for statement: returns (page written, tile)."""
    column = task % 8
    source_tiles = (rows * HEAD + 31) // 32 if inverse else rows
    staged = [tiles_at((tile + source_base) * 8 + column) for tile in range(source_tiles)]
    output = torch.zeros(32, 32)
    for target_row in range(32):
        if inverse:
            if target_row >= HEAD:
                continue
            token = task // 8
            head = (target_row // 6) * rows * 6 + token * 6 + target_row % 6
            source_tile, source_head = head // 32, head % 32
        else:
            head = (task // 8) * 32 + target_row
            if head >= rows * HEAD:
                continue
            remainder = head % (rows * 6)
            source_tile, source_head = remainder // 6, (head // (rows * 6)) * 6 + remainder % 6
        output[target_row] = staged[source_tile][source_head]
    return destination_base + task, output


def run_launch(argument_lists, buffers):
    """Run every core's [inverse, tasks, six words per task] list against `buffers` {address: {page: tile}}."""
    for arguments in argument_lists:
        inverse, count = arguments[0], arguments[1]
        for index in range(count):
            source, destination, rows, source_base, destination_base, task = arguments[2 + index * 6:8 + index * 6]
            page, tile = kernel_task(bool(inverse), rows, buffers[source].__getitem__, source_base, destination_base, task)
            buffers[destination][page] = tile


# --- the served composition, on logical tensors -------------------------------------------------------------------------

def served(query, segments, sdpa):
    """PackedExtentReplayReader.__call__ (Slice, fold, Concat, SDPA, Slice, inverse fold, Concat, Concat) on logical
    tensors; the fold at one KV head is the token-major reshape (KernelIndexModelTests)."""
    users = []
    for user, (first, last) in enumerate(segments):
        rows = last - first
        chunk = query[:, first:last]
        outputs = []
        for bundle in pinned.LAYOUT(rows, 8):
            groups = [chunk[:, group['offset']:group['offset'] + group['rows']].reshape(1, 1, group['rows'] * HEAD, 256)
                      for group in bundle]
            result = sdpa(user, torch.cat(groups, dim=1))
            outputs += [result[:, index:index + 1].reshape(1, bundle[index]['rows'], HEAD, 256)
                        for index in range(len(bundle))]
        users.append(torch.cat(outputs, dim=1))
    return torch.cat(users, dim=1)


def sdpa_stub(user, stacked):
    """Row-mixing per batch and user so a swapped page, group or user shows: x * 1.5 + a per-batch, per-user offset."""
    offsets = torch.arange(stacked.shape[1], dtype=torch.float32).reshape(1, -1, 1, 1) * 0.25
    return stacked * 1.5 + offsets + user * 1000.0


def chunks_for(segments):
    bundles = [pinned.LAYOUT(last - first, 8) for first, last in segments]
    return fold.chunks_of(segments, bundles)


def user_of(segments, chunk):
    return next(user for user, (first, last) in enumerate(segments) if first == chunk.first)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_kernel_composition_is_the_served_composition_byte_for_byte(self):
        for segments in (((0, 16), (16, 32), (32, 48), (48, 64)), ((0, 16),), ((0, 32), (32, 48)), ((0, 32),)):
            total = segments[-1][1]
            query = torch.arange(total * HEAD * 256, dtype=torch.float32).reshape(1, total, HEAD, 256) + 0.5
            chunks = chunks_for(segments)
            expected = served(query, segments, sdpa_stub)

            # fold-in: one buffer per chunk, distinct addresses, the block query at its own
            buffers = {1: to_pages(query)}
            stacked_addresses = [100 + index for index in range(len(chunks))]
            for address in stacked_addresses:
                buffers[address] = {}
            flat = fold.flat_tasks(fold.forward_tasks(chunks), [1] * len(chunks), stacked_addresses)
            run_launch(fold.runtime_arguments(False, fold.distribute(flat, 110)), buffers)

            # SDPA on each chunk's stacked query, as the reader does, then fold-out
            for index, chunk in enumerate(chunks):
                stack, clean = from_pages(buffers[stacked_addresses[index]], chunk.batches, chunk.rows * HEAD)
                self.assertTrue(clean, 'the stacked query padding is zero, as the served fold leaves it')
                buffers[200 + index] = to_pages(sdpa_stub(user_of(segments, chunk), stack))
            buffers[300] = {}
            flat = fold.flat_tasks(fold.inverse_tasks(chunks), [200 + index for index in range(len(chunks))],
                                   [300] * len(chunks))
            run_launch(fold.runtime_arguments(True, fold.distribute(flat, 110)), buffers)
            built, clean = from_pages(buffers[300], total, HEAD)
            self.assertTrue(clean)
            self.assertTrue(torch.equal(built, expected), segments)
            # every page of the output is written exactly once
            self.assertEqual(sorted(buffers[300]), list(range(total * COLUMNS)))

    def test_the_fold_in_stacks_are_the_served_stacks(self):
        segments = ((0, 16), (16, 32), (32, 48), (48, 64))
        query = torch.randn(1, 64, HEAD, 256)
        chunks = chunks_for(segments)
        buffers = {1: to_pages(query)}
        addresses = [10, 11, 12, 13]
        for address in addresses:
            buffers[address] = {}
        flat = fold.flat_tasks(fold.forward_tasks(chunks), [1] * 4, addresses)
        run_launch(fold.runtime_arguments(False, fold.distribute(flat, 80)), buffers)
        for chunk, address, (first, last) in zip(chunks, addresses, segments):
            built, clean = from_pages(buffers[address], 2, 8 * HEAD)
            expected = torch.cat([query[:, first + offset:first + offset + 8].reshape(1, 1, 48, 256)
                                  for offset in (0, 8)], dim=1)
            self.assertTrue(clean)
            self.assertTrue(torch.equal(built, expected))

    def test_task_counts_and_core_distribution(self):
        chunks = chunks_for(((0, 16), (16, 32), (32, 48), (48, 64)))
        self.assertEqual([len(part) for part in fold.forward_tasks(chunks)], [32] * 4)
        self.assertEqual([len(part) for part in fold.inverse_tasks(chunks)], [128] * 4)
        for tasks in (128, 512):
            per_core = fold.distribute(list(range(tasks)), 110)
            self.assertEqual(len(per_core), 110)
            self.assertEqual(sorted(task for core in per_core for task in core), list(range(tasks)))
            self.assertLessEqual(max(map(len, per_core)) - min(map(len, per_core)), 1)
        with self.assertRaises(ValueError):
            fold.distribute([], 110)

    def test_bundles_that_do_not_tile_the_block_in_order_are_refused(self):
        bundles = [[[{'offset': 8, 'rows': 8}, {'offset': 0, 'rows': 8}]]]
        with self.assertRaises(ValueError):
            fold.chunks_of(((0, 16),), bundles)
        with self.assertRaises(ValueError):
            fold.chunks_of(((0, 16), (16, 32)), [pinned.LAYOUT(16, 8)])

    def test_the_kernel_text_is_the_served_fold_body(self):
        served_text = (HERE / 'attention_fold_dma_tp.cpp').read_text()
        block_text = (HERE / 'attention_block_fold_tp.cpp').read_text()

        def body(text):
            return re.sub(r'\s+', ' ', text[text.index('for (uint32_t target_row'):text.index('asm volatile')])

        self.assertEqual(body(served_text), body(block_text))
        self.assertIn('#ifndef QWEN_FOLD_HEAD_ROWS', block_text)


# --- the launch builder on a fake ttnn ---------------------------------------------------------------------------------

class FakeTensor(object):
    def __init__(self, shape, base, chips=4, memory='dram'):
        self.shape, self.base, self.chips, self.memory = tuple(shape), base, chips, memory
        self.dtype, self.layout = 'bf16', 'tile'

    def memory_config(self):
        return self.memory


class FakeShard(object):
    def __init__(self, address, layout):
        self.address, self.layout = address, layout

    def buffer_address(self):
        return self.address


def launch_ttnn(record, chips=4):
    def generic_op(tensors, program):
        record.append((list(tensors), program))

    addresses = iter(range(0x9000, 0x90000, 0x100))

    def empty(shape, dtype=None, layout=None, device=None, memory_config=None):
        return FakeTensor(shape, next(addresses), chips, memory_config)

    return SimpleNamespace(
        bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1', empty=empty,
        get_device_tensors=lambda tensor: [FakeShard(tensor.base + chip * 0x10000000, tensor.memory)
                                           for chip in range(chips)],
        CoreCoord=lambda x, y: (x, y), CoreRange=lambda a, b: (a, b), CoreRangeSet=lambda ranges: list(ranges),
        CBDescriptor=SimpleNamespace, CBFormatDescriptor=SimpleNamespace, TileDescriptor=lambda value: value, Tile=list,
        MeshProgramDescriptor=dict, ProgramDescriptor=SimpleNamespace, MeshCoordinate=lambda *v: v,
        MeshCoordinateRange=lambda *v: v, KernelDescriptor=SimpleNamespace,
        DataMovementConfigDescriptor=SimpleNamespace, DataMovementProcessor=SimpleNamespace(RISCV_0='r0'),
        NOC=SimpleNamespace(RISCV_0_default=0),
        TensorAccessorArgs=lambda shard: SimpleNamespace(get_compile_time_args=lambda: ['interleaved', shard.layout]),
        RuntimeArgs=lambda: defaultdict(dict), generic_op=generic_op)


def all_arguments(kernel):
    return [arguments for column in kernel.runtime_args.values() for arguments in column.values()]


class LaunchTests(unittest.TestCase):
    def build(self, memory='dram'):
        record = []
        ttnn = launch_ttnn(record)
        chunks = chunks_for(((0, 16), (16, 32), (32, 48), (48, 64)))
        query = FakeTensor((1, 64, HEAD, 256), 0x1000, 4, memory)
        owned = []
        with four(), patch.dict(sys.modules, {'ttnn': ttnn}):
            stacked = fold.fold_in(MESH, query, chunks, owned)
            results = [FakeTensor((1, 2, 48, 256), 0x5000 + 0x100 * index, 4, memory) for index in range(4)]
            output = fold.fold_out(MESH, results, chunks, memory, owned)
        return record, stacked, results, output, owned

    def test_two_launches_on_four_chips_with_the_fold_define_and_the_served_scratch(self):
        record, stacked, results, output, owned = self.build()
        self.assertEqual(len(record), 2)
        for (tensors, program), inverse in zip(record, (0, 1)):
            self.assertEqual(sorted(program), [((0, chip), (0, chip)) for chip in range(4)])
            for key, description in program.items():
                (kernel,) = description.kernels
                self.assertEqual(kernel.kernel_source, str(HERE / 'attention_block_fold_tp.cpp'))
                self.assertEqual(dict(kernel.defines), {'QWEN_FOLD_HEAD_ROWS': '6'})
                self.assertEqual(description.cbs[0].total_size, (8 + 1) * 2048)
                self.assertEqual(sum(arguments[1] for arguments in all_arguments(kernel)), 512 if inverse else 128)
                self.assertTrue(all(arguments[0] == inverse for arguments in all_arguments(kernel)))
        self.assertEqual(len(record[0][0]), 5, 'the block query, then the four stacked queries')
        self.assertIs(record[1][0][-1], output, 'the block output is the last io tensor')
        self.assertEqual(tuple(output.shape), (1, 64, HEAD, 256))
        self.assertEqual([tuple(stack.shape) for stack in stacked], [(1, 2, 48, 256)] * 4)
        self.assertEqual(len(owned), 5)

    def test_runtime_addresses_are_the_chips_own(self):
        record, stacked, results, output, owned = self.build()
        for chip in range(4):
            kernel = record[1][1][((0, chip), (0, chip))].kernels[0]
            for arguments in all_arguments(kernel):
                sources = {arguments[2 + i * 6] for i in range(arguments[1])}
                destinations = {arguments[3 + i * 6] for i in range(arguments[1])}
                self.assertTrue(all(address >> 28 == chip for address in sources | destinations))
                self.assertEqual(destinations, {output.base + chip * 0x10000000})

    def test_l1_and_dram_are_accepted_and_anything_else_is_a_problem(self):
        ttnn = launch_ttnn([])
        chunks = chunks_for(((0, 16), (16, 32), (32, 48), (48, 64)))
        with four():
            self.assertIsNone(fold.problem(FakeTensor((1, 64, HEAD, 256), 1, 4, 'l1'), chunks, 'l1', ttnn))
            self.assertIn('interleaved', fold.problem(FakeTensor((1, 64, HEAD, 256), 1, 4, 'sharded'), chunks, 'dram', ttnn))
            self.assertIn('interleaved', fold.problem(FakeTensor((1, 64, HEAD, 256), 1, 4, 'dram'), chunks, 'sharded', ttnn))
            self.assertIn('query', fold.problem(FakeTensor((1, 32, HEAD, 256), 1, 4), chunks, 'dram', ttnn))

    def test_mismatched_accessor_layouts_are_refused(self):
        record = []
        ttnn = launch_ttnn(record)
        chunks = chunks_for(((0, 16), (16, 32)))
        query = FakeTensor((1, 32, HEAD, 256), 0x1000)
        with four(), patch.dict(sys.modules, {'ttnn': ttnn}):
            owned = []
            fold.fold_in(MESH, query, chunks, owned)
            results = [FakeTensor((1, 2, 48, 256), 0x5000, 4, 'dram'), FakeTensor((1, 2, 48, 256), 0x5100, 4, 'l1')]
            with self.assertRaisesRegex(ValueError, 'accessor layout'):
                fold.fold_out(MESH, results, chunks, 'dram', owned)


# --- the reader hook ----------------------------------------------------------------------------------------------------

def block_reader(device, segments=((0, 16), (16, 32), (32, 48), (48, 64))):
    lent = [lend(device, last - first) for first, last in segments]
    starts = (4200, 9000, 4300, 5000)[:len(segments)]
    return folded.PackedExtentReplayReader(device, MESH, segments, WIDTH, [host_table(i + 1) for i in range(len(segments))],
                                         storage=lent, max_group_rows=8, starts=starts)


def fake_folds(device):
    def fold_in(mesh, query, chunks, owned):
        stacked = [device.device_tensor((1, chunk.batches, chunk.rows * HEAD, 256), 'bf16', 'tile', 'dram', name='stack',
                                        origin=('block-fold-in', query, chunk.first)) for chunk in chunks]
        owned.extend(stacked)
        return stacked

    def fold_out(mesh, results, chunks, memory_config, owned):
        total = chunks[-1].first + chunks[-1].groups[-1][0] + chunks[-1].groups[-1][1]
        output = device.device_tensor((1, total, HEAD, 256), 'bf16', 'tile', memory_config, name='block-out',
                                      origin=('block-fold-out', tuple(results)))
        owned.append(output)
        return output
    return fold_in, fold_out


def fake_served_fold(device):
    def permute(mesh, source, rows, owned, *, inverse=False, offset=0):
        output = device.device_tensor((1, rows, HEAD, 256) if inverse else (1, 1, rows * HEAD, 256), 'bf16', 'tile',
                                      'dram', name='fold')
        owned.append(output)
        return output
    return permute


class ReaderHookTests(unittest.TestCase):
    def setUp(self):
        self.device = FourChipDevice()
        self.harness = Harness(self)

    def call(self, memory='dram', **flags):
        device = self.device
        reader = block_reader(device)
        query = device.device_tensor((1, 64, HEAD, 256), 'bf16', 'tile', memory, name='query')
        keys = device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys')
        refreshed = []
        fold_in, fold_out = fake_folds(device)
        device.sdpa.clear()
        with env(**flags), patch('extent_attention_replay_tp.attention_mask_replay.execute',
                                 side_effect=lambda *a: refreshed.append(a)), \
                patch('attention_block_fold_tp.fold_in', side_effect=fold_in), \
                patch('attention_block_fold_tp.fold_out', side_effect=fold_out), \
                patch('extent_attention_replay_tp.device_layout_dma', side_effect=fake_served_fold(device)):
            output = reader(query, keys, keys, scale=0.0625, memory_config='dram')
        return reader, output, list(device.sdpa), refreshed, query, keys

    def test_flag_off_is_the_served_path(self):
        reader, output, sdpa, refreshed, query, keys = self.call()
        self.assertEqual(len(sdpa), 4)
        self.assertEqual(reader.calls, 1)
        self.assertEqual(output.origin[0], 'concat')

    def test_flag_on_makes_the_same_sdpa_calls_and_the_same_bookkeeping(self):
        served_reader, served_output, served_sdpa, served_refresh, *unused = self.call()
        reader, output, sdpa, refreshed, query, keys = self.call(QWEN_FAST_TP4_ATTN_FOLD='1')
        self.assertEqual(len(sdpa), len(served_sdpa))
        for left, right in zip(served_sdpa, sdpa):
            self.assertEqual(sorted(left), sorted(right))
            for name in ('is_causal', 'scale', 'memory_config'):
                self.assertEqual(left[name], right[name], name)
            self.assertEqual(vars(left['program_config']), vars(right['program_config']))
            self.assertEqual(tuple(left['attn_mask'].shape), tuple(right['attn_mask'].shape))
            self.assertEqual(left['is_causal'], False)
        # the same page tables, cur_pos words and masks, in the same bundle order, from the segment readers
        entries = [entry for segment in reader.readers for entry in segment.metadata]
        for (bundle, pages, mask, config), options in zip(entries, sdpa):
            self.assertIs(options['page_table_tensor'], pages)
            self.assertIs(options['attn_mask'], mask)
        self.assertEqual([options['cur_pos_tensor'] for options in sdpa],
                         [positions for segment in reader.readers for positions in segment.cur_pos])
        self.assertEqual(len(refreshed), len(served_refresh))
        self.assertEqual((reader.calls, [r.calls for r in reader.readers], reader.refresh_calls, reader.failed),
                         (served_reader.calls, [r.calls for r in served_reader.readers], served_reader.refresh_calls,
                          served_reader.failed))
        self.assertEqual(output.origin[0], 'block-fold-out')
        self.assertEqual(tuple(output.shape), tuple(served_output.shape))

    def test_the_sdpa_queries_are_the_folded_stacks_and_the_results_feed_the_fold_out(self):
        reader, output, sdpa, refreshed, query, keys = self.call(QWEN_FAST_TP4_ATTN_FOLD='1')
        self.assertEqual([options['program_config'].k_chunk_size for options in sdpa], [256] * 4)
        self.assertEqual(len(output.origin[1]), 4)
        for result in output.origin[1]:
            self.assertEqual(result.origin[0], 'sdpa')
            self.assertEqual(result.origin[1].origin[0], 'block-fold-in')
            self.assertIs(result.origin[1].origin[1], query)

    def test_intermediates_are_released_and_the_output_is_not(self):
        reader, output, sdpa, refreshed, query, keys = self.call(QWEN_FAST_TP4_ATTN_FOLD='1')
        freed = self.device.deallocated
        self.assertFalse(any(tensor is output for tensor in freed))
        self.assertFalse(any(tensor is query for tensor in freed))
        self.assertEqual(sum(1 for tensor in freed if tensor.name == 'stack'), 4)
        self.assertEqual(sum(1 for tensor in freed if tensor.name == 'sdpa'), 4)

    def test_a_failure_poisons_every_segment_reader_and_still_releases(self):
        device = self.device
        reader = block_reader(device)
        query = device.device_tensor((1, 64, HEAD, 256), 'bf16', 'tile', 'dram', name='query')
        keys = device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys')
        fold_in, fold_out = fake_folds(device)
        with env(QWEN_FAST_TP4_ATTN_FOLD='1'), patch('extent_attention_replay_tp.attention_mask_replay.execute'), \
                patch('attention_block_fold_tp.fold_in', side_effect=fold_in), \
                patch('attention_block_fold_tp.fold_out', side_effect=RuntimeError('device')):
            with self.assertRaises(RuntimeError):
                reader(query, keys, keys, scale=0.0625, memory_config='dram')
        self.assertTrue(all(r.failed for r in reader.readers))
        self.assertTrue(reader.failed)
        self.assertEqual(sum(1 for tensor in device.deallocated if tensor.name == 'stack'), 4)

    def test_the_shared_mask_budget_is_enforced_per_segment_reader(self):
        device = self.device
        reader = block_reader(device)
        query = device.device_tensor((1, 64, HEAD, 256), 'bf16', 'tile', 'dram', name='query')
        keys = device.device_tensor((10, 1, 64, 256), 'bf16', 'tile', 'dram', name='keys')
        fold_in, fold_out = fake_folds(device)
        refreshed = []
        with env(QWEN_FAST_TP4_ATTN_FOLD='1'), patch('extent_attention_replay_tp.attention_mask_replay.execute',
                                                     side_effect=lambda *a: refreshed.append(a)), \
                patch('attention_block_fold_tp.fold_in', side_effect=fold_in), \
                patch('attention_block_fold_tp.fold_out', side_effect=fold_out):
            with reader.shared_masks(2):
                after_scope_refresh = len(refreshed)
                reader(query, keys, keys, scale=0.0625, memory_config='dram')
                reader(query, keys, keys, scale=0.0625, memory_config='dram')
                self.assertEqual(len(refreshed), after_scope_refresh, 'no per-call refresh inside a shared-mask scope')
                with self.assertRaisesRegex(AssertionError, 'call budget'):
                    reader(query, keys, keys, scale=0.0625, memory_config='dram')

    def test_a_query_the_launch_is_not_written_for_falls_back_to_the_served_path(self):
        lines = []
        folded.PackedExtentReplayReader.fold_fallbacks = set()
        tp4_vglue.take()
        with patch('extent_attention_fold_tp.tp4_vglue.log_line', side_effect=lines.append):
            reader, output, sdpa, refreshed, query, keys = self.call(memory='sharded', QWEN_FAST_TP4_ATTN_FOLD='1')
        self.assertEqual(output.origin[0], 'concat')
        self.assertEqual(len(sdpa), 4)
        self.assertTrue(any(line.startswith(tp4_vglue.FALLBACK) for line in lines))
        self.assertEqual(tp4_vglue.take(), {'attn_fold_fallback': 1})

    def test_engaged_calls_are_counted_for_the_marker(self):
        tp4_vglue.take()
        self.call(QWEN_FAST_TP4_ATTN_FOLD='1')
        self.assertEqual(tp4_vglue.take(), {'attn_fold': 1})

    def test_the_flag_at_the_pair_raises(self):
        with pair(), env(QWEN_FAST_TP4_ATTN_FOLD='1'):
            with self.assertRaises(ValueError):
                tp4_vglue.enabled(tp4_vglue.ATTN_FOLD)


class PinnedReaderTests(unittest.TestCase):
    """extent_attention_replay_tp.py is the bytes CB2b qualified (packed_any_admission refuses the traffic profile on any other), so
    V3a lives in a subclass that tp_addresses binds in its place."""

    EVIDENCE = HERE / 'packed_any_evidence_tp4.json'

    @unittest.skipUnless((HERE / 'packed_any_evidence_tp4.json').is_file(), 'the evidence record is not in this tree (the image)')
    def test_the_reader_twin_still_has_the_bytes_the_evidence_qualified(self):
        import hashlib
        import json

        recorded = json.loads(self.EVIDENCE.read_text(encoding='utf-8'))['sources']['extent_attention_replay_tp.py']
        self.assertEqual(hashlib.sha256((HERE / 'extent_attention_replay_tp.py').read_bytes()).hexdigest(), recorded)

    def test_the_hook_is_not_in_the_pinned_module(self):
        text = (HERE / 'extent_attention_replay_tp.py').read_text(encoding='utf-8')
        self.assertNotIn('tp4_vglue', text)
        self.assertNotIn('attention_block_fold_tp', text)

    def test_the_subclass_is_bound_in_place_of_the_reader_at_four_cards(self):
        import tp_addresses

        entry = ('extent_attention_replay_tp', 'PackedExtentReplayReader', 'extent_attention_fold_tp', 'PackedExtentReplayReader')
        self.assertIn(entry, tp_addresses.TWINS)
        self.assertTrue(issubclass(folded.PackedExtentReplayReader, quad.PackedExtentReplayReader) or
                        folded.PackedExtentReplayReader.__mro__[1].__module__ == 'extent_attention_replay_tp')
        self.assertIn('extent_attention_fold_tp.py', tp4_vglue.RUNTIME_FILES)


if __name__ == '__main__':
    unittest.main()
