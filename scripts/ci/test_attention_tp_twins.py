"""The sequential engine's attention and K/V writers at one KV head per chip, and the width-generic pieces beside them.

At the pair each twin builds what the pinned function builds (checked program for program where a program is built); at
QWEN_FAST_TP=4 they read the head layout from tp_shapes: a (N, 1, 64, 256) cache whose upstream update kernels carry
num_heads = 1 (reader compile argument 9, writer 10, compute 7), 6 folded rows per token, four chip programs.
"""

from collections import defaultdict, namedtuple
import itertools
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import attention_mask_replay
import attention_mask_replay_tp
import attention_parallel
import attention_parallel_tp
import attention_replay_tp
import ordered_cache
import ordered_cache_tp
import packed_ordered_cache
import serving_cache_owner
import tp_addresses
import tp_shapes

HERE = Path(__file__).resolve().parent
Core = namedtuple('Core', 'x y')


def four():
    return patch.dict(os.environ, {'QWEN_FAST_TP': '4'})


def pair():
    return patch.dict(os.environ, {}, clear=True)


class Shard:
    def __init__(self, address, padded=None):
        self.address, self.padded_shape = address, padded

    def buffer_address(self):
        return self.address

    def device(self):
        return SimpleNamespace(worker_core_from_logical_core=lambda core: Core(core.x + 1, core.y + 1))


class Tensor:
    def __init__(self, shape, chips, base, dtype='bf16', layout='tile', memory='dram'):
        self.shape, self.dtype, self.layout, self._memory = tuple(shape), dtype, layout, memory
        self.padded_shape = self.shape
        self.shards = [Shard(base + 0x1000 * chip) for chip in range(chips)]

    def memory_config(self):
        return self._memory


def fake_ttnn(record, chips):
    class Kernel(SimpleNamespace):
        class SourceType:
            SOURCE_CODE = 'source'

    def kernel(**keywords):
        return Kernel(**keywords)

    kernel.SourceType = Kernel.SourceType
    return SimpleNamespace(
        bfloat8_b='bf8', bfloat16='bf16', int32='int32', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='row',
        DRAM_MEMORY_CONFIG='dram',
        get_device_tensors=lambda tensor: tensor.shards,
        CoreRangeSet=list, CoreRange=lambda *values: values, CoreCoord=Core,
        CBDescriptor=lambda **keywords: tuple(sorted((k, str(v)) for k, v in keywords.items())),
        CBFormatDescriptor=lambda **keywords: tuple(sorted(keywords.items(), key=lambda item: item[0])),
        TileDescriptor=lambda value: value, Tile=list,
        SemaphoreDescriptor=lambda **keywords: tuple(sorted((k, str(v)) for k, v in keywords.items())),
        MeshProgramDescriptor=dict,
        ProgramDescriptor=lambda **keywords: SimpleNamespace(**keywords),
        MeshCoordinate=lambda *values: values, MeshCoordinateRange=lambda *values: values,
        KernelDescriptor=kernel, ComputeConfigDescriptor=lambda **keywords: ('compute', tuple(keywords.items())),
        DataMovementConfigDescriptor=lambda **keywords: ('movement', tuple(keywords.items())),
        DataMovementProcessor=SimpleNamespace(RISCV_0='riscv0', RISCV_1='riscv1'),
        NOC=SimpleNamespace(RISCV_0_default=0, RISCV_1_default=1),
        TensorAccessorArgs=lambda shard: SimpleNamespace(get_compile_time_args=lambda: [7]),
        RuntimeArgs=lambda: defaultdict(dict),
        generic_op=lambda tensors, program: record.append((tensors, program)))


MESH = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
KERNELS = dict(reader='r', writer='w', compute='c')


def view(program):
    return {key: [(kernel.compile_time_args, {x: dict(column) for x, column in kernel.runtime_args.items()})
                  for kernel in value.kernels] + [value.cbs] for key, value in program.items()}


class OrderedCacheTests(unittest.TestCase):
    def launch(self, function, chips, heads, rows=8):
        record = []
        cache = Tensor((100, heads, 64, 256), chips, 0x100, dtype='bf8')
        packed = Tensor((1, rows, 32, 256), chips, 0x200)
        positions = Tensor((rows,), chips, 0x300, dtype='int32', layout='row')
        pages = Tensor((rows, 64), chips, 0x400, dtype='int32', layout='row')
        with patch.dict(sys.modules, {'ttnn': fake_ttnn(record, chips)}):
            function(MESH, cache, packed, positions, pages, KERNELS)
        return record[0][1]

    def test_at_the_pair_the_twin_builds_the_pinned_program(self):
        with pair():
            theirs = self.launch(ordered_cache.update, 2, 2)
            ours = self.launch(ordered_cache_tp.update, 2, 2)
        self.assertEqual(view(theirs), view(ours))
        self.assertEqual(len(ours), 2)

    def test_at_four_cards_the_kernels_carry_one_head(self):
        with four():
            program = self.launch(ordered_cache_tp.update, 4, 1)
        self.assertEqual(len(program), 4)
        for chip, key in enumerate(sorted(program)):
            reader, writer, compute = program[key].kernels
            self.assertEqual(reader.compile_time_args[8:12], [1, 1, 64, 2], 'is_paged, num_heads, block_size, block_size_t')
            self.assertEqual(writer.compile_time_args[9:13], [1, 1, 64, 2])
            self.assertEqual(compute.compile_time_args, [0, 1, 24, 25, 26, 16, 8, 1])
            first = reader.runtime_args[0][0]
            self.assertEqual(first[0], 0x100 + 0x1000 * chip, 'this chip\'s cache buffer')

    def test_the_cache_geometry_follows_the_width(self):
        with pair():
            self.assertEqual(ordered_cache_tp.validate_shapes((8, 2, 64, 256), (1, 2, 32, 256), (2,), (2, 4)), 2)
            with self.assertRaisesRegex(ValueError, 'two-head'):
                ordered_cache_tp.validate_shapes((8, 1, 64, 256), (1, 2, 32, 256), (2,), (2, 4))
        with four():
            self.assertEqual(ordered_cache_tp.validate_shapes((8, 1, 64, 256), (1, 2, 32, 256), (2,), (2, 4)), 2)
            self.assertEqual(ordered_cache_tp.validate_shapes((2052, 1, 64, 256), (1, 32, 32, 256), (32,), (32, 2052)), 32)
            with self.assertRaisesRegex(ValueError, 'one-head'):
                ordered_cache_tp.validate_shapes((8, 2, 64, 256), (1, 2, 32, 256), (2,), (2, 4))

    def test_at_the_pair_the_twin_validates_what_the_pinned_module_does(self):
        cases = [((8, 2, 64, 256), (1, rows, 32, 256), (rows,), (rows, 4)) for rows in (1, 2, 4, 8, 16, 32)]
        cases += [((8, 1, 64, 256), (1, 2, 32, 256), (2,), (2, 4)), ((8, 2, 64, 256), (1, 3, 32, 256), (3,), (3, 4)),
                  ((8, 2, 64, 256), (1, 2, 32, 256), (2,), (2, 1025)), ((2051, 2, 64, 256), (1, 1, 32, 256), (1,), (1, 2052))]
        with pair():
            for arguments in cases:
                try:
                    expected = ('value', ordered_cache.validate_shapes(*arguments))
                except ValueError as error:
                    expected = ('error', str(error))
                try:
                    actual = ('value', ordered_cache_tp.validate_shapes(*arguments))
                except ValueError as error:
                    actual = ('error', str(error))
                self.assertEqual(actual, expected, arguments)


class PackedWriterTests(unittest.TestCase):
    def test_the_pair_arguments_are_the_literals_they_were(self):
        with pair():
            self.assertEqual(packed_ordered_cache.reader_compile_args(64, 2052),
                             [0, 1, 1, 2, 0, 8, 0, 256, 1, 2, 64, 2, 2052, 0, 8208, 3, 2, 0, 0])
            self.assertEqual(packed_ordered_cache.writer_compile_args(2052),
                             [16, 24, 25, 26, 1, 2, 0, 8, 512, 1, 2, 64, 2, 2052, 3, 2, 0, 0])
            self.assertEqual(packed_ordered_cache.compute_args(), (0, 1, 24, 25, 26, 16, 8, 2))
            self.assertEqual(packed_ordered_cache.validate_chained((8, 2, 64, 256), (1, 64, 32, 256), (64,), (64, 4), 64), 64)

    def test_four_cards_carry_one_head_in_all_three_kernels(self):
        with four():
            self.assertEqual(packed_ordered_cache.reader_compile_args(64, 2052)[9], 1)
            self.assertEqual(packed_ordered_cache.writer_compile_args(2052)[10], 1)
            self.assertEqual(packed_ordered_cache.compute_args()[-1], 1)
            self.assertEqual(packed_ordered_cache.validate_chained((8, 1, 64, 256), (1, 64, 32, 256), (64,), (64, 4), 64), 64)
            with self.assertRaises(ValueError):
                packed_ordered_cache.validate_chained((8, 2, 64, 256), (1, 64, 32, 256), (64,), (64, 4), 64)


class MaskProgramTests(unittest.TestCase):
    def launch(self, function, chips, head_rows, capacity=4352):
        record = []
        positions = Tensor((8,), chips, 0x100, dtype='int32', layout='row')
        mask = Tensor((2, 1, 8 * head_rows, capacity), chips, 0x200)
        with patch.dict(sys.modules, {'ttnn': fake_ttnn(record, chips)}):
            program = function(MESH, positions, mask, rows=8, batches=2, offset=0, capacity=capacity)
        return program

    def test_at_the_pair_the_twin_builds_the_pinned_program(self):
        with pair():
            theirs = self.launch(attention_mask_replay.prepare, 2, 12)
            ours = self.launch(attention_mask_replay_tp.prepare, 2, 12)
        self.assertEqual(view(theirs), view(ours))
        for value in ours.values():
            self.assertEqual(value.kernels[0].kernel_source, str(Path(attention_mask_replay.__file__).with_suffix('.cpp')))
            self.assertEqual(value.kernels[0].defines, [])

    def test_four_cards_program_the_sibling_kernel_on_four_chips(self):
        with four():
            program = self.launch(attention_mask_replay_tp.prepare, 4, 6)
        self.assertEqual(len(program), 4)
        for value in program.values():
            kernel = value.kernels[0]
            self.assertEqual(kernel.kernel_source, str(HERE / 'attention_mask_replay_tp.cpp'))
            self.assertEqual(dict(kernel.defines), {'QWEN_FOLD_HEAD_ROWS': '6'})
            self.assertEqual(sum(len(column) for column in kernel.runtime_args.values()), 2 * 2 * 8)
        with four(), self.assertRaises(ValueError):
            self.launch(attention_mask_replay_tp.prepare, 4, 12)

    def test_mask_position_is_the_token_of_the_head_row(self):
        with pair():
            for head in range(96):
                self.assertEqual(attention_mask_replay_tp.mask_position(100, 8, 1, head, 4),
                                 attention_mask_replay.mask_position(100, 8, 1, head, 4))
        with four():
            self.assertEqual([attention_mask_replay_tp.mask_position(100, 8, 1, head) for head in (0, 5, 6, 47)],
                             [108, 108, 109, 115])
            with self.assertRaises(ValueError):
                attention_mask_replay_tp.mask_position(100, 8, 1, 48)


class ParallelExecuteTests(unittest.TestCase):
    def run_execute(self, function, head_rows):
        calls = []
        counter = itertools.count(100, 10)

        def make(name, shape):
            return SimpleNamespace(name=name, shape=tuple(shape), address=next(counter))

        operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram')
        operations.concat = lambda values, dim, memory_config: make('concat', (1, len(values)) + values[0].shape[2:])
        operations.slice = lambda value, start, end, memory_config: (calls.append(('slice', tuple(end))) or make('slice', (1, 1, end[2], 256)))
        operations.transformer = SimpleNamespace(paged_scaled_dot_product_attention_decode=lambda *args, **kwargs: (
            calls.append(('sdpa', kwargs['page_table_tensor'])) or make('sdpa', (1, 2, 8 * head_rows, 256))))
        bundle = [dict(offset=0, rows=8, signature=(256, 4352)), dict(offset=8, rows=8, signature=(256, 4352))]
        metadata = [(bundle, 'pages', 'mask', 'config')]
        owned = []

        def fold(mesh, source, count, owned_list, *, inverse=False, offset=0):
            calls.append(('fold', inverse, offset))
            return make('fold', (1, count, head_rows, 256) if inverse else (1, 1, count * head_rows, 256))

        module = sys.modules[function.__module__]
        with patch.object(module, 'device_layout_dma', side_effect=fold):
            function('mesh', operations, make('query', (1, 16, head_rows, 256)), 'keys', 'values', metadata, owned,
                     scale=0.1, memory_config='dram')
        return calls

    def test_the_twin_calls_what_the_pinned_function_calls_at_the_pair(self):
        with pair():
            self.assertEqual(self.run_execute(attention_parallel.execute, 12), self.run_execute(attention_parallel_tp.execute, 12))

    def test_four_cards_slice_six_head_rows_per_token(self):
        with four():
            calls = self.run_execute(attention_parallel_tp.execute, 6)
        self.assertEqual([entry for entry in calls if entry[0] == 'slice'], [('slice', (1, 1, 48, 256)), ('slice', (1, 2, 48, 256))])


class ReplayReaderTests(unittest.TestCase):
    def test_the_reader_builds_and_checks_a_six_head_query_at_four_cards(self):
        import torch

        class Operations(SimpleNamespace):
            pass

        record = []
        made = []

        def upload(value, dtype=None, *args):
            made.append(tuple(value.shape))
            return Tensor(value.shape, 4, 0x100 * (len(made) + 1), dtype='int32' if dtype == 'int32' else 'bf16',
                          layout='row' if dtype == 'int32' else 'tile')

        operations = SimpleNamespace(int32='int32', SDPAProgramConfig=lambda **keywords: SimpleNamespace(**keywords))
        operations.get_device_tensors = lambda tensor: tensor.shards
        operations.deallocate = lambda value: None
        pages = torch.arange(2052, dtype=torch.int32).reshape(1, 2052)
        with four(), patch.dict(os.environ, {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}), \
                patch.dict(sys.modules, {'ttnn': fake_ttnn(record, 4)}):
            reader = attention_replay_tp.ReplayAttentionReader(operations, MESH, 16, 4352, pages, upload, max_group_rows=8)
            self.assertIn((2, 1, 48, 4352), made, 'the folded mask: 8 rows x 6 heads, two entries')
            with self.assertRaisesRegex(ValueError, 'Replay query geometry changed'):
                reader(SimpleNamespace(shape=(1, 16, 12, 256)), object(), object(), scale=1.0, memory_config='dram')
            reader.failed = False
            reader.close()

    def test_the_pooled_readers_take_the_twin_as_their_base_only_at_four_cards(self):
        code = ('import pooled_attention_replay as p, attention_replay as a, attention_replay_tp as t;'
                'print(p.PooledReplayAttentionReader.__mro__[1] is t.ReplayAttentionReader,'
                ' p.PooledReplayAttentionReader.__mro__[1] is a.ReplayAttentionReader)')
        outputs = {}
        for name, extra in (('pair', {}), ('four', {'QWEN_FAST_TP': '4'})):
            environment = {key: value for key, value in os.environ.items() if key != 'QWEN_FAST_TP'}
            environment.update(extra, PYTHONPATH=str(HERE), PYTHONDONTWRITEBYTECODE='1')
            outputs[name] = subprocess.run([sys.executable, '-c', code], env=environment, capture_output=True,
                                           text=True, timeout=120).stdout.strip()
        self.assertEqual(outputs, {'pair': 'False True', 'four': 'True False'})


class CacheOwnerTests(unittest.TestCase):
    def fixture(self, chips, heads):
        caches = [[SimpleNamespace(shape=(128, heads, 64, 256), dtype='bf8', addresses=tuple(index * 2 + side + 1 + 100 * chip
                                                                                       for chip in range(chips)))
                   for side in range(2)] for index in range(16)]
        layers = [SimpleNamespace(is_full_attention=True, attention=SimpleNamespace(
            paged_k=pair_[0], paged_v=pair_[1], use_paged=True)) for pair_ in caches]
        model = SimpleNamespace(num_devices=chips, args=SimpleNamespace(max_batch_size=8), _paged_kv_caches=caches,
                                layers=layers)
        runner = SimpleNamespace(model=SimpleNamespace(model=[model]), kv_caches=caches)
        operations = SimpleNamespace(bfloat8_b='bf8', get_device_tensors=lambda tensor: [
            SimpleNamespace(buffer_address=lambda value=value: value) for value in tensor.addresses])
        return operations, runner, model

    def test_the_owner_takes_the_width_the_process_serves_at(self):
        self.addCleanup(tp_addresses.uninstall)
        with four():
            tp_addresses.install()   # the owner reads addresses through the pinned helper: the startup seam
            owner = serving_cache_owner.ServingCacheOwner(*self.fixture(4, 1))
            owner.validate()
            with self.assertRaises(ValueError):
                serving_cache_owner.ServingCacheOwner(*self.fixture(2, 2))
            with self.assertRaisesRegex(ValueError, 'TP4'):
                serving_cache_owner.ServingCacheOwner(*self.fixture(4, 2))
        tp_addresses.uninstall()
        with pair():
            serving_cache_owner.ServingCacheOwner(*self.fixture(2, 2)).validate()
            with self.assertRaises(ValueError):
                serving_cache_owner.ServingCacheOwner(*self.fixture(4, 1))


class SeamTests(unittest.TestCase):
    def test_install_rebinds_the_sequential_attention_and_writer_helpers(self):
        import attention_fold_dma
        import attention_replay
        pinned = dict(execute=attention_parallel.execute, reader=attention_replay.ReplayAttentionReader,
                      prepare=attention_mask_replay.prepare, update=ordered_cache.update,
                      validate=ordered_cache.validate_shapes, fold=attention_fold_dma.device_layout_dma)
        self.addCleanup(tp_addresses.uninstall)
        with four():
            tp_addresses.install()
        import attention_fold_dma_tp
        self.assertIs(attention_parallel.execute, attention_parallel_tp.execute)
        self.assertIs(attention_replay.ReplayAttentionReader, attention_replay_tp.ReplayAttentionReader)
        self.assertIs(attention_mask_replay.prepare, attention_mask_replay_tp.prepare)
        self.assertIs(ordered_cache.update, ordered_cache_tp.update)
        self.assertIs(ordered_cache.validate_shapes, ordered_cache_tp.validate_shapes)
        self.assertIs(attention_fold_dma.device_layout_dma, attention_fold_dma_tp.device_layout_dma)
        tp_addresses.uninstall()
        self.assertIs(attention_parallel.execute, pinned['execute'])
        self.assertIs(attention_replay.ReplayAttentionReader, pinned['reader'])
        self.assertIs(ordered_cache.update, pinned['update'])
        self.assertIs(attention_fold_dma.device_layout_dma, pinned['fold'])


if __name__ == '__main__':
    unittest.main()
