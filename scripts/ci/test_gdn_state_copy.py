"""The batched state-copy launch: same transfers, fewer dispatches.

`copy_compact_batch` exists to collapse the packed decode's per-user compact state
moves - three launches per GDN layer at four users, on every one of the 48 layers -
into one. These tests pin the two things that makes safe: the runtime argument
layout the kernel reads (`groups`, `worker`, then fifteen values per group), and the
refusal to batch groups whose tensor accessor arguments differ, since the kernel
resolves those at compile time and reuses one set for every group.
"""

from collections import defaultdict
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import gdn_state_copy


COMPACT = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
FULL = [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4
COUNTS = [384, 160, 160, 160, 160]


class FakeTensor:
    """One mesh tensor: two chip shards, each with its own buffer address."""

    def __init__(self, shape, address, accessor=('interleaved',)):
        self.shape = shape
        self.dtype = 'bf16'
        self.layout = 'tile'
        self.address = address
        self.accessor = accessor

    def memory_config(self):
        return 'dram'

    def shard(self, chip):
        return SimpleNamespace(buffer_address=lambda: self.address + chip,
                               accessor=self.accessor)


class FakeTtnn(SimpleNamespace):
    pass


def fake_ttnn(record):
    def generic_op(tensors, program):
        record.append((list(tensors), program))

    return FakeTtnn(
        bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
        get_device_tensors=lambda tensor: [tensor.shard(0), tensor.shard(1)],
        CoreRangeSet=list, CoreRange=lambda *values: values, CoreCoord=lambda *values: values,
        CBDescriptor=SimpleNamespace, CBFormatDescriptor=SimpleNamespace,
        TileDescriptor=lambda value: value, Tile=list,
        MeshProgramDescriptor=dict, ProgramDescriptor=SimpleNamespace,
        MeshCoordinate=lambda *values: values, MeshCoordinateRange=lambda *values: values,
        KernelDescriptor=SimpleNamespace,
        DataMovementConfigDescriptor=SimpleNamespace,
        DataMovementProcessor=SimpleNamespace(RISCV_0='writer'),
        NOC=SimpleNamespace(RISCV_0_default=0),
        TensorAccessorArgs=lambda shard: SimpleNamespace(
            get_compile_time_args=lambda: list(shard.accessor)),
        RuntimeArgs=lambda: defaultdict(dict),
        generic_op=generic_op)


def state(base, shapes=COMPACT, accessor=('interleaved',)):
    return [FakeTensor(shape, base + 100 * index, accessor) for index, shape in enumerate(shapes)]


def run(call, *arguments):
    record = []
    with patch.dict(sys.modules, {'ttnn': fake_ttnn(record)}):
        call(*arguments)
    return record


def core_args(program, chip=0):
    kernel = program[((0, chip), (0, chip))].kernels[0]
    return kernel.runtime_args, kernel.compile_time_args


class StateCopyBatchTests(unittest.TestCase):
    def test_single_transfer_layout_is_groups_worker_then_one_arg_block(self):
        source, destination = state(1000), state(9000)
        record = run(gdn_state_copy.copy_compact, source, destination)
        self.assertEqual(len(record), 1)
        tensors, program = record[0]
        self.assertEqual(len(tensors), 10)
        arguments, compile_args = core_args(program)
        self.assertEqual(compile_args, ['interleaved'] * 10)
        values = arguments[0][0]
        self.assertEqual(values[:2], [1, 0])
        self.assertEqual(len(values), 2 + 15)
        expected = []
        for slot, count in enumerate(COUNTS):
            expected.extend([1000 + 100 * slot, 9000 + 100 * slot, count])
        self.assertEqual(values[2:], expected)

    def test_worker_index_is_the_only_per_core_difference(self):
        record = run(gdn_state_copy.copy_compact, state(1000), state(9000))
        arguments, _ = core_args(record[0][1])
        for worker in range(48):
            values = arguments[worker % 8][worker // 8]
            self.assertEqual(values[1], worker)
            self.assertEqual(values[2:], arguments[0][0][2:])

    def test_three_transfers_become_one_launch_carrying_three_arg_blocks(self):
        transfers = [(state(1000 * (index + 1)), state(9000 + 1000 * index)) for index in range(3)]
        record = run(gdn_state_copy.copy_compact_batch, transfers)
        self.assertEqual(len(record), 1, 'one launch for all three transfers')
        tensors, program = record[0]
        self.assertEqual(len(tensors), 30)
        arguments, compile_args = core_args(program)
        # One accessor set, not one per group: the kernel resolves them at compile time.
        self.assertEqual(compile_args, ['interleaved'] * 10)
        values = arguments[0][0]
        self.assertEqual(values[:2], [3, 0])
        self.assertEqual(len(values), 2 + 45)
        # Each block is byte-for-byte what the separate call would have sent.
        for index, (source, destination) in enumerate(transfers):
            block = values[2 + 15 * index:2 + 15 * (index + 1)]
            separate = run(gdn_state_copy.copy_compact, source, destination)
            self.assertEqual(block, core_args(separate[0][1])[0][0][0][2:])

    def test_second_chip_carries_its_own_addresses(self):
        transfers = [(state(1000 * (index + 1)), state(9000 + 1000 * index)) for index in range(2)]
        record = run(gdn_state_copy.copy_compact_batch, transfers)
        blocks = [core_args(record[0][1], chip)[0][0][0] for chip in range(2)]
        self.assertEqual([block[:2] for block in blocks], [[2, 0], [2, 0]])
        for position, (left, right) in enumerate(zip(blocks[0][2:], blocks[1][2:])):
            # Positions 2, 5, 8, ... of each 15-value block are page counts, shared by
            # both chips; the addresses either side of them are the shard's own.
            expected = left if position % 3 == 2 else left + 1
            self.assertEqual(right, expected, 'value %d' % position)

    def test_mismatched_accessor_arguments_refuse_to_batch(self):
        transfers = [(state(1000), state(9000)),
                     (state(2000, accessor=('sharded', 7)), state(11000, accessor=('sharded', 7)))]
        with self.assertRaisesRegex(ValueError, 'one tensor accessor layout'):
            run(gdn_state_copy.copy_compact_batch, transfers)

    def test_aliasing_across_groups_is_caught(self):
        shared = state(1000)
        with self.assertRaisesRegex(ValueError, 'must not alias'):
            run(gdn_state_copy.copy_compact_batch, [(shared, state(9000)), (shared, state(11000))])

    def test_empty_batch_and_full_shapes_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'At least one transfer'):
            gdn_state_copy.copy_compact_batch([])
        with self.assertRaisesRegex(ValueError, 'transfer geometry'):
            run(gdn_state_copy.copy_compact_batch, [(state(1000), state(9000, shapes=FULL))])

    def test_copy_active_still_takes_the_single_group_path(self):
        record = run(gdn_state_copy.copy_active, state(1000), state(9000, shapes=FULL))
        arguments, _ = core_args(record[0][1])
        self.assertEqual(arguments[0][0][:2], [1, 0])


class BatchFlagTests(unittest.TestCase):
    def test_flag_defaults_off_and_rejects_anything_but_0_or_1(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(gdn_state_copy.batch_enabled())
        self.assertTrue(gdn_state_copy.batch_enabled({gdn_state_copy.FLAG: '1'}))
        self.assertFalse(gdn_state_copy.batch_enabled({gdn_state_copy.FLAG: '0'}))
        for bad in ('true', 'yes', '2', ''):
            with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                gdn_state_copy.batch_enabled({gdn_state_copy.FLAG: bad})


class KernelArgumentLayoutTests(unittest.TestCase):
    """The kernel half of the same contract, read from the source."""

    def source(self):
        return Path(__file__).with_name('gdn_state_copy.cpp').read_text(encoding='utf-8')

    def test_kernel_reads_groups_then_worker_and_loops_over_arg_blocks(self):
        source = self.source()
        self.assertIn('const uint32_t groups = get_arg_val<uint32_t>(0);', source)
        self.assertIn('const uint32_t worker = get_arg_val<uint32_t>(1);', source)
        self.assertIn('for (uint32_t group = 0; group < groups; ++group)', source)
        self.assertIn('const uint32_t base = 2 + group * 15;', source)
        for offset in ('base', 'base + 3', 'base + 6', 'base + 9', 'base + 12'):
            self.assertIn('>(%s, worker, scratch);' % offset, source)
        # The old fixed layout must be gone, not merely shadowed.
        self.assertNotIn('get_arg_val<uint32_t>(15)', source)


if __name__ == '__main__':
    unittest.main()
