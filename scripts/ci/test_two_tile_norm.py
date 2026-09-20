import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from two_tile_norm import (SHARDED_KEYS, TILE, TwoTileNormBinding, bind_two_tile_norms, is_decode, shard_height,
                           two_tile_memory_config, two_tile_norm_config, two_tile_program_config, validate_two_tile_rows)


class FakeTTNN:
    """The three ttnn constructors the override builds with, recording their arguments."""

    class TensorMemoryLayout:
        WIDTH_SHARDED, HEIGHT_SHARDED, INTERLEAVED = 'width', 'height', 'interleaved'

    class BufferType:
        L1, DRAM = 'l1', 'dram'

    L1_MEMORY_CONFIG = 'l1-interleaved'

    @staticmethod
    def ShardSpec(grid, shape, orientation):
        return SimpleNamespace(grid=grid, shape=list(shape), orientation=orientation)

    @staticmethod
    def MemoryConfig(memory_layout, buffer_type, shard_spec=None):
        return SimpleNamespace(memory_layout=memory_layout, buffer_type=buffer_type, shard_spec=shard_spec)

    @staticmethod
    def LayerNormShardedMultiCoreProgramConfig(*, compute_with_storage_grid_size, subblock_w, block_h, block_w, inplace):
        x, y = compute_with_storage_grid_size
        return SimpleNamespace(compute_with_storage_grid_size=SimpleNamespace(x=x, y=y), subblock_w=subblock_w,
                               block_h=block_h, block_w=block_w, inplace=inplace)


def one_tile_memory(ttnn, width=160, grid='cores-8x4', buffer='l1', orientation='row-major'):
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer, ttnn.ShardSpec(grid, [TILE, width], orientation))


def one_tile_program(ttnn, grid=(8, 4), subblock_w=1, block_w=5, inplace=False):
    return ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=grid, subblock_w=subblock_w,
                                                       block_h=1, block_w=block_w, inplace=inplace)


def native_config(ttnn, **extra):
    return dict(sharded_output_config=one_tile_memory(ttnn), sharded_program_config=one_tile_program(ttnn), **extra)


class Mode:
    """Stands in for tt_transformers Mode: a member named DECODE, one named PREFILL."""
    DECODE = SimpleNamespace(name='DECODE')
    PREFILL = SimpleNamespace(name='PREFILL')


class RowRuleTests(unittest.TestCase):
    def test_only_whole_tiles_beyond_one_take_the_path(self):
        self.assertEqual(validate_two_tile_rows(64), 2)
        self.assertEqual(validate_two_tile_rows(96), 3)
        for rows in (1, 2, 4, 8, 16, 31, 32, 33, 48, 65, 64.0, True, None, '64'):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, 'beyond one tile'):
                validate_two_tile_rows(rows)

    def test_decode_is_the_mode_member_or_its_string(self):
        self.assertTrue(is_decode(Mode.DECODE))
        self.assertTrue(is_decode('decode'))
        self.assertFalse(is_decode(Mode.PREFILL))
        self.assertFalse(is_decode('prefill'))
        self.assertFalse(is_decode(None))

    def test_shard_height_reads_only_sharded_configs(self):
        ttnn = FakeTTNN()
        self.assertEqual(shard_height(one_tile_memory(ttnn)), 32)
        self.assertIsNone(shard_height(ttnn.MemoryConfig('interleaved', 'dram')))
        self.assertIsNone(shard_height('dram'))
        self.assertIsNone(shard_height(SimpleNamespace(shard_spec=SimpleNamespace(shape='bad'))))


class MemoryConfigTests(unittest.TestCase):
    def test_the_shard_grows_to_the_rows_on_the_same_cores(self):
        ttnn = FakeTTNN()
        native = one_tile_memory(ttnn, width=160, grid='cores-8x4', buffer='l1', orientation='row-major')
        rebuilt = two_tile_memory_config(native, 64, ttnn)
        self.assertEqual((rebuilt.memory_layout, rebuilt.buffer_type), ('width', 'l1'))
        self.assertEqual((rebuilt.shard_spec.grid, rebuilt.shard_spec.shape, rebuilt.shard_spec.orientation),
                         ('cores-8x4', [64, 160], 'row-major'))
        # the native stays one tile: it is the 32-row block's
        self.assertEqual(native.shard_spec.shape, [32, 160])
        self.assertEqual(two_tile_memory_config(native, 96, ttnn).shard_spec.shape, [96, 160])

    def test_anything_but_a_one_tile_width_shard_is_refused(self):
        ttnn = FakeTTNN()
        with self.assertRaisesRegex(ValueError, 'width-sharded'):
            two_tile_memory_config(ttnn.MemoryConfig('interleaved', 'dram'), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'width-sharded'):
            two_tile_memory_config(ttnn.MemoryConfig('height', 'l1', ttnn.ShardSpec('g', [32, 160], 'rm')), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'one tile'):
            two_tile_memory_config(ttnn.MemoryConfig('width', 'l1', ttnn.ShardSpec('g', [64, 160], 'rm')), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            two_tile_memory_config(one_tile_memory(ttnn), 32, ttnn)


class ProgramConfigTests(unittest.TestCase):
    def test_block_h_becomes_the_tile_count_and_the_rest_is_kept(self):
        ttnn = FakeTTNN()
        native = one_tile_program(ttnn, grid=(8, 4), subblock_w=1, block_w=5, inplace=False)
        rebuilt = two_tile_program_config(native, 64, ttnn)
        self.assertEqual((rebuilt.compute_with_storage_grid_size.x, rebuilt.compute_with_storage_grid_size.y), (8, 4))
        self.assertEqual((rebuilt.subblock_w, rebuilt.block_h, rebuilt.block_w, rebuilt.inplace), (1, 2, 5, False))
        self.assertEqual(native.block_h, 1)
        self.assertEqual(two_tile_program_config(native, 96, ttnn).block_h, 3)
        kept = two_tile_program_config(one_tile_program(ttnn, grid=(11, 3), subblock_w=4, block_w=8, inplace=True), 64, ttnn)
        self.assertEqual((kept.compute_with_storage_grid_size.x, kept.compute_with_storage_grid_size.y,
                          kept.subblock_w, kept.block_w, kept.inplace), (11, 3, 4, 8, True))

    def test_a_program_that_is_not_one_tile_or_hides_a_field_is_refused(self):
        ttnn = FakeTTNN()
        two_tile = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(8, 4), subblock_w=1,
                                                                block_h=2, block_w=5, inplace=False)
        with self.assertRaisesRegex(ValueError, 'block_h 1'):
            two_tile_program_config(two_tile, 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'does not expose subblock_w, inplace'):
            two_tile_program_config(SimpleNamespace(compute_with_storage_grid_size=SimpleNamespace(x=8, y=4),
                                                    block_h=1, block_w=5), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            two_tile_program_config(one_tile_program(ttnn), 32, ttnn)


class NormConfigTests(unittest.TestCase):
    def test_the_two_sharded_values_are_rebuilt_and_the_rest_passes_through(self):
        ttnn = FakeTTNN()
        dram = ttnn.MemoryConfig('interleaved', 'dram')
        native = native_config(ttnn, output_mem_config=dram, distributed_output_mem_config=None, epsilon=1e-6)
        rebuilt = two_tile_norm_config(native, 64, ttnn)
        self.assertEqual(sorted(rebuilt), sorted(native))
        self.assertEqual(rebuilt['sharded_output_config'].shard_spec.shape, [64, 160])
        self.assertEqual(rebuilt['sharded_program_config'].block_h, 2)
        self.assertIs(rebuilt['output_mem_config'], dram)
        self.assertIsNone(rebuilt['distributed_output_mem_config'])
        self.assertEqual(rebuilt['epsilon'], 1e-6)
        # the native dict and its values are untouched
        self.assertEqual(native['sharded_output_config'].shard_spec.shape, [32, 160])
        self.assertEqual(native['sharded_program_config'].block_h, 1)
        self.assertEqual(SHARDED_KEYS, ('sharded_output_config', 'sharded_program_config'))

    def test_a_none_output_config_takes_the_interleaved_hand_off_and_a_set_one_is_kept(self):
        ttnn = FakeTTNN()
        # the framework's layer-norm dicts carry None: the wide block's hand-off fills it
        rebuilt = two_tile_norm_config(native_config(ttnn, output_mem_config=None), 64, ttnn, output_mem_config='l1')
        self.assertEqual(rebuilt['output_mem_config'], 'l1')
        # without a hand-off it stays None, as the 32-row block's would
        self.assertIsNone(two_tile_norm_config(native_config(ttnn, output_mem_config=None), 64, ttnn)['output_mem_config'])
        # a config the framework already set wins over the hand-off
        dram = ttnn.MemoryConfig('interleaved', 'dram')
        self.assertIs(two_tile_norm_config(native_config(ttnn, output_mem_config=dram), 64, ttnn, output_mem_config='l1')['output_mem_config'], dram)
        # a dict without the key gains it only from the hand-off
        self.assertEqual(two_tile_norm_config(native_config(ttnn), 64, ttnn, output_mem_config='l1')['output_mem_config'], 'l1')
        self.assertNotIn('output_mem_config', two_tile_norm_config(native_config(ttnn), 64, ttnn))

    def test_a_config_missing_the_sharded_keys_or_hiding_another_one_tile_shard_is_refused(self):
        ttnn = FakeTTNN()
        for missing in SHARDED_KEYS:
            config = native_config(ttnn)
            del config[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'no longer carries'):
                two_tile_norm_config(config, 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'no longer carries'):
            two_tile_norm_config(None, 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'sharded_input_config carry a one-tile shard'):
            two_tile_norm_config(native_config(ttnn, sharded_input_config=one_tile_memory(ttnn)), 64, ttnn)
        # a one-tile output_mem_config is the same refusal, hand-off or not
        with self.assertRaisesRegex(ValueError, 'output_mem_config carry a one-tile shard'):
            two_tile_norm_config(native_config(ttnn, output_mem_config=one_tile_memory(ttnn)), 64, ttnn, output_mem_config='l1')
        # a stale two-tile shard elsewhere is not ours to judge; a one-tile one is
        two_tile_norm_config(native_config(ttnn, other=ttnn.MemoryConfig('width', 'l1', ttnn.ShardSpec('g', [64, 8], 'rm'))), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            two_tile_norm_config(native_config(ttnn), 32, ttnn)


class BindingTests(unittest.TestCase):
    def model(self, ttnn, layers=64):
        args = SimpleNamespace()
        args.get_norm_config = Mock(side_effect=lambda name, mode: native_config(ttnn, name=name, mode=mode, output_mem_config=None))
        return SimpleNamespace(args=args, layers=[object()] * layers)

    def test_decode_calls_come_back_two_tiles_high_and_interleaved_and_prefill_ones_untouched(self):
        ttnn = FakeTTNN()
        model = self.model(ttnn)
        binding = bind_two_tile_norms(model, 64, ttnn)
        self.assertIsInstance(binding, TwoTileNormBinding)
        self.assertEqual(binding.label, 'decode norm')
        self.assertEqual(binding.binding[:2], (model.args, 'get_norm_config'))
        self.assertIs(binding.binding[2], binding)
        self.assertEqual(binding.bindings, [binding.binding])
        self.assertEqual((binding.rows, binding.tiles, binding.expected_calls, binding.calls), (64, 2, 129, 0))
        decode = binding('attn', Mode.DECODE)
        self.assertEqual((decode['name'], decode['mode']), ('attn', Mode.DECODE))
        self.assertEqual(decode['sharded_output_config'].shard_spec.shape, [64, 160])
        self.assertEqual(decode['sharded_program_config'].block_h, 2)
        self.assertEqual(decode['output_mem_config'], ttnn.L1_MEMORY_CONFIG)
        self.assertEqual(binding.calls, 1)
        prefill = binding('ff', Mode.PREFILL)
        self.assertEqual(prefill['sharded_output_config'].shard_spec.shape, [32, 160])
        self.assertEqual(prefill['sharded_program_config'].block_h, 1)
        self.assertIsNone(prefill['output_mem_config'])
        self.assertEqual(binding.calls, 1)
        binding('lm_head', Mode.DECODE)
        self.assertEqual(binding.calls, 2)
        self.assertEqual(model.args.get_norm_config.call_count, 3)
        model.args.get_norm_config.assert_any_call('lm_head', Mode.DECODE)

    def test_the_binding_reaches_the_native_getter_through_instance_overrides(self):
        from model_batch import instance_overrides

        ttnn = FakeTTNN()
        model = self.model(ttnn, layers=2)
        native = model.args.get_norm_config
        binding = bind_two_tile_norms(model, 64, ttnn)
        self.assertEqual(binding.expected_calls, 5)
        with instance_overrides(binding.bindings):
            self.assertIs(model.args.get_norm_config, binding)
            self.assertEqual(model.args.get_norm_config('attn', Mode.DECODE)['sharded_program_config'].block_h, 2)
        self.assertIs(model.args.get_norm_config, native)
        self.assertEqual(model.args.get_norm_config('attn', Mode.DECODE)['sharded_program_config'].block_h, 1)
        self.assertEqual(binding.calls, 1)

    def test_a_block_within_one_tile_or_a_model_without_the_getter_is_refused(self):
        ttnn = FakeTTNN()
        for rows in (16, 32):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, 'beyond one tile'):
                bind_two_tile_norms(self.model(ttnn), rows, ttnn)
        with self.assertRaisesRegex(ValueError, 'no get_norm_config'):
            bind_two_tile_norms(SimpleNamespace(args=SimpleNamespace(), layers=[object()] * 64), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'layers and args'):
            bind_two_tile_norms(SimpleNamespace(args=SimpleNamespace()), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'interleaved L1'):
            bind_two_tile_norms(self.model(ttnn), 64, SimpleNamespace(L1_MEMORY_CONFIG=None))


if __name__ == '__main__':
    unittest.main()
