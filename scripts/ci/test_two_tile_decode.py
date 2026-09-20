import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from model_batch import instance_overrides
from two_tile_decode import (TwoTileAttentionBinding, TwoTileAttentionDecode, TwoTileMLPBinding, TwoTileMLPForward,
                             bind_two_tile_attention, bind_two_tile_mlp, two_tile_matmul_1d_progcfg,
                             validate_two_tile_rows)


class FakeTTNN:
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = 'dram', 'l1'

    class TensorMemoryLayout:
        WIDTH_SHARDED = 'width'

    class BufferType:
        L1, DRAM = 'l1', 'dram'

    def __init__(self):
        self.deallocated = []
        self.transformer = SimpleNamespace(attn_decode_prep=Mock(return_value=('q', 'gate', 'k_sh', 'v_sh')))

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

    @staticmethod
    def MatmulMultiCoreReuseMultiCast1DProgramConfig(*, compute_with_storage_grid_size, in0_block_w, out_subblock_h,
                                                     out_subblock_w, per_core_M, per_core_N, fuse_batch, fused_activation,
                                                     mcast_in0):
        x, y = compute_with_storage_grid_size
        return SimpleNamespace(compute_with_storage_grid_size=SimpleNamespace(x=x, y=y), in0_block_w=in0_block_w,
                               out_subblock_h=out_subblock_h, out_subblock_w=out_subblock_w, per_core_M=per_core_M,
                               per_core_N=per_core_N, fuse_batch=fuse_batch, fused_activation=fused_activation,
                               mcast_in0=mcast_in0)

    def deallocate(self, value):
        self.deallocated.append(value)


def one_tile_progcfg(ttnn, grid=(8, 8), in0_block_w=8, out_subblock_w=1, per_core_N=2, fused_activation=None,
                     per_core_M=1, out_subblock_h=1, fuse_batch=True, mcast_in0=True):
    """tp_common.create_matmul_1d_decode_progcfg at M = 1, as model_config builds attn_qkv_decode_1d_progcfg."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=in0_block_w, out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w, per_core_M=per_core_M, per_core_N=per_core_N, fuse_batch=fuse_batch,
        fused_activation=fused_activation, mcast_in0=mcast_in0)


def fake_attention(args, rows=64, prep_layout=True):
    attention = SimpleNamespace(args=args, use_paged=True, _fused_qkv=True, tw={'q_norm': 'qn', 'k_norm': 'kn'},
                                NH=12, NKV=2, HD=256, rope_dim=64)
    attention._qkv_raw_decode = Mock(return_value='qkv_raw')
    attention._kv_shard_cfg = Mock(side_effect=lambda batch: 'kv-shard-%d' % batch)
    attention._decode_from_prep = Mock(return_value='attention-output')
    return attention


def fake_mlp(devices=2):
    mlp = SimpleNamespace(num_devices=devices)
    mlp._fuse_gateup_agmm = True
    mlp.seen = []

    def _forward_tp(x):
        mlp.seen.append(mlp._fuse_gateup_agmm)
        return ('mlp-output', x)

    mlp._forward_tp = _forward_tp
    mlp.forward = Mock(return_value='native-mlp-forward')
    return mlp


def fake_model(ttnn, layers=64, full=(3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63), progcfg=None):
    args = SimpleNamespace(proj_1d_decode=True, attn_qkv_decode_1d_progcfg=progcfg or one_tile_progcfg(ttnn))
    args.get_norm_config = Mock(side_effect=lambda name, mode: dict(
        sharded_output_config=ttnn.MemoryConfig('width', 'l1', ttnn.ShardSpec('g', [32, 160], 'rm')),
        sharded_program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 4), subblock_w=1, block_h=1, block_w=5, inplace=False),
        output_mem_config=None))
    model = SimpleNamespace(args=args, layers=[])
    for index in range(layers):
        is_full = index in full
        model.layers.append(SimpleNamespace(is_full_attention=is_full,
                                            attention=fake_attention(args) if is_full else SimpleNamespace(args=args),
                                            feed_forward=fake_mlp()))
    return model


class ProgramConfigTests(unittest.TestCase):
    def test_per_core_m_becomes_the_tile_count_and_the_subblock_follows_the_builders_rule(self):
        ttnn = FakeTTNN()
        native = one_tile_progcfg(ttnn, grid=(8, 8), in0_block_w=8, out_subblock_w=1, per_core_N=2, fused_activation='silu')
        rebuilt = two_tile_matmul_1d_progcfg(native, 64, ttnn)
        self.assertEqual((rebuilt.compute_with_storage_grid_size.x, rebuilt.compute_with_storage_grid_size.y), (8, 8))
        self.assertEqual((rebuilt.in0_block_w, rebuilt.per_core_N, rebuilt.fused_activation), (8, 2, 'silu'))
        self.assertEqual((rebuilt.per_core_M, rebuilt.out_subblock_h, rebuilt.out_subblock_w), (2, 2, 1))
        self.assertTrue(rebuilt.fuse_batch is True and rebuilt.mcast_in0 is True)
        # the native is the 32-row block's and stays one tile
        self.assertEqual((native.per_core_M, native.out_subblock_h), (1, 1))
        # out_subblock_h * out_subblock_w stays within the fp32 cap of 4, as the builder keeps it
        self.assertEqual(two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn, out_subblock_w=2), 64, ttnn).out_subblock_h, 2)
        self.assertEqual(two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn, out_subblock_w=3), 64, ttnn).out_subblock_h, 1)
        self.assertEqual(two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn, out_subblock_w=4), 64, ttnn).out_subblock_h, 1)
        three = two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn, out_subblock_w=1), 96, ttnn)
        self.assertEqual((three.per_core_M, three.out_subblock_h), (3, 3))
        self.assertEqual(two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn, out_subblock_w=2), 96, ttnn).out_subblock_h, 1)

    def test_anything_but_a_one_tile_fused_batch_mcast_config_is_refused(self):
        ttnn = FakeTTNN()
        with self.assertRaisesRegex(ValueError, 'per_core_M 2, out_subblock_h 1'):
            two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn, per_core_M=2), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'per_core_M 1, out_subblock_h 2'):
            two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn, out_subblock_h=2), 64, ttnn)
        for name in ('fuse_batch', 'mcast_in0'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'multicast in0 with fused batch'):
                two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn, **{name: False}), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'outside the fp32 cap'):
            two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn, out_subblock_w=5), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'does not expose in0_block_w, per_core_N'):
            two_tile_matmul_1d_progcfg(SimpleNamespace(compute_with_storage_grid_size=SimpleNamespace(x=8, y=8),
                                                       out_subblock_h=1, out_subblock_w=1, per_core_M=1, fuse_batch=True,
                                                       fused_activation=None, mcast_in0=True), 64, ttnn)
        for rows in (16, 32, 48):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, 'beyond one tile'):
                two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn), rows, ttnn)
        self.assertEqual(validate_two_tile_rows(64), 2)


class AttentionForwardTests(unittest.TestCase):
    def test_the_forward_is_the_models_prep_path_at_the_blocks_rows(self):
        ttnn = FakeTTNN()
        args = SimpleNamespace(proj_1d_decode=True)
        attention = fake_attention(args)
        progcfg = two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn), 64, ttnn)
        args.attn_qkv_decode_1d_progcfg = progcfg
        forward = TwoTileAttentionDecode(attention, 64, ttnn, progcfg)
        x = SimpleNamespace(shape=(1, 1, 64, 5120))
        self.assertEqual(forward(x, 'positions', 'cos', 'sin', page_table='pages'), 'attention-output')
        attention._qkv_raw_decode.assert_called_once_with(x)
        attention._kv_shard_cfg.assert_called_once_with(64)
        ttnn.transformer.attn_decode_prep.assert_called_once_with(
            'qkv_raw', 'cos', 'sin', 'qn', 'kn', 12, 2, 256, 64, 'kv-shard-64', batch=64, memory_config='dram')
        self.assertEqual(ttnn.deallocated, ['qkv_raw'])
        attention._decode_from_prep.assert_called_once_with('q', 'gate', 'k_sh', 'v_sh', 'positions', 'pages', 64)
        self.assertEqual(forward.calls, 1)
        # the prep tail is whatever is bound on the instance at call time: the block's writer and readers
        attention._decode_from_prep = Mock(return_value='bound-tail')
        self.assertEqual(forward(x, 'positions', 'cos', 'sin', page_table='pages'), 'bound-tail')
        self.assertEqual(forward.calls, 2)

    def test_a_forward_refuses_the_wrong_rows_an_unpaged_call_or_an_unbound_config_and_frees_the_projection(self):
        ttnn = FakeTTNN()
        args = SimpleNamespace(proj_1d_decode=True)
        attention = fake_attention(args)
        progcfg = two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn), 64, ttnn)
        args.attn_qkv_decode_1d_progcfg = progcfg
        forward = TwoTileAttentionDecode(attention, 64, ttnn, progcfg)
        with self.assertRaisesRegex(ValueError, 'bound for 64 rows'):
            forward(SimpleNamespace(shape=(1, 1, 32, 5120)), 'p', 'c', 's', page_table='pages')
        with self.assertRaisesRegex(ValueError, 'paged decode only'):
            forward(SimpleNamespace(shape=(1, 1, 64, 5120)), 'p', 'c', 's')
        args.attn_qkv_decode_1d_progcfg = one_tile_progcfg(ttnn)
        with self.assertRaisesRegex(AssertionError, 'not bound on the model args'):
            forward(SimpleNamespace(shape=(1, 1, 64, 5120)), 'p', 'c', 's', page_table='pages')
        attention._qkv_raw_decode.assert_not_called()
        self.assertEqual(forward.calls, 0)
        args.attn_qkv_decode_1d_progcfg = progcfg
        ttnn.transformer.attn_decode_prep = Mock(side_effect=RuntimeError('prep failed'))
        with self.assertRaisesRegex(RuntimeError, 'prep failed'):
            forward(SimpleNamespace(shape=(1, 1, 64, 5120)), 'p', 'c', 's', page_table='pages')
        self.assertEqual(ttnn.deallocated, ['qkv_raw'])
        self.assertEqual(forward.calls, 0)

    def test_construction_needs_the_paged_fused_1d_attention_the_serving_model_runs(self):
        ttnn = FakeTTNN()
        progcfg = one_tile_progcfg(ttnn)
        for name, value, message in (('use_paged', False, 'paged fused-QKV'), ('_fused_qkv', False, 'paged fused-QKV'),
                                     ('tw', {'q_norm': 'qn'}, 'q_norm and k_norm'), ('NH', 0, 'integer NH'),
                                     ('_decode_from_prep', None, 'no longer exposes _decode_from_prep')):
            attention = fake_attention(SimpleNamespace(proj_1d_decode=True))
            setattr(attention, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                TwoTileAttentionDecode(attention, 64, ttnn, progcfg)
        with self.assertRaisesRegex(ValueError, '1D decode projection'):
            TwoTileAttentionDecode(fake_attention(SimpleNamespace(proj_1d_decode=False)), 64, ttnn, progcfg)
        with self.assertRaisesRegex(ValueError, 'attn_decode_prep is required'):
            TwoTileAttentionDecode(fake_attention(SimpleNamespace(proj_1d_decode=True)), 64, SimpleNamespace(), progcfg)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            TwoTileAttentionDecode(fake_attention(SimpleNamespace(proj_1d_decode=True)), 32, ttnn, progcfg)


class AttentionBindingTests(unittest.TestCase):
    def test_the_binding_is_the_rebuilt_config_on_the_args_and_one_forward_per_full_attention_layer(self):
        ttnn = FakeTTNN()
        model = fake_model(ttnn)
        native = model.args.attn_qkv_decode_1d_progcfg
        binding = bind_two_tile_attention(model, 64, ttnn)
        self.assertIsInstance(binding, TwoTileAttentionBinding)
        self.assertEqual(binding.label, 'full-attention forward')
        self.assertEqual((binding.rows, binding.expected_calls, binding.calls), (64, 16, 0))
        self.assertEqual(binding.progcfg.per_core_M, 2)
        self.assertEqual(binding.bindings[0], (model.args, 'attn_qkv_decode_1d_progcfg', binding.progcfg))
        full = [layer.attention for layer in model.layers if layer.is_full_attention]
        self.assertEqual([(instance, name) for instance, name, value in binding.bindings[1:]],
                         [(attention, 'forward_decode') for attention in full])
        self.assertTrue(all(isinstance(value, TwoTileAttentionDecode) for instance, name, value in binding.bindings[1:]))
        with instance_overrides(binding.bindings):
            self.assertIs(model.args.attn_qkv_decode_1d_progcfg, binding.progcfg)
            for attention in full:
                attention.forward_decode(SimpleNamespace(shape=(1, 1, 64, 5120)), 'p', 'c', 's', page_table='pages')
            self.assertEqual(binding.calls, 16)
        self.assertIs(model.args.attn_qkv_decode_1d_progcfg, native)
        self.assertFalse(any('forward_decode' in attention.__dict__ for attention in full))

    def test_a_model_without_full_attention_layers_or_with_foreign_args_is_refused(self):
        ttnn = FakeTTNN()
        with self.assertRaisesRegex(ValueError, 'at least one full-attention layer'):
            bind_two_tile_attention(fake_model(ttnn, layers=4, full=()), 64, ttnn)
        model = fake_model(ttnn, layers=8, full=(3, 7))
        model.layers[7].attention.args = SimpleNamespace(proj_1d_decode=True, attn_qkv_decode_1d_progcfg=one_tile_progcfg(ttnn))
        with self.assertRaisesRegex(ValueError, 'share the model args'):
            bind_two_tile_attention(model, 64, ttnn)
        model = fake_model(ttnn, layers=8, full=(3, 7))
        model.args.attn_qkv_decode_1d_progcfg = None
        with self.assertRaisesRegex(ValueError, 'no attn_qkv_decode_1d_progcfg'):
            bind_two_tile_attention(model, 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            bind_two_tile_attention(fake_model(ttnn, layers=8, full=(3, 7)), 32, ttnn)


class MLPForwardTests(unittest.TestCase):
    def test_the_fusion_is_off_for_exactly_the_call(self):
        mlp = fake_mlp()
        forward = TwoTileMLPForward(mlp, 64)
        x = SimpleNamespace(shape=(1, 1, 64, 5120))
        self.assertEqual(forward(x), ('mlp-output', x))
        self.assertEqual(mlp.seen, [False])
        self.assertTrue(mlp._fuse_gateup_agmm is True)
        self.assertEqual(forward.calls, 1)
        mlp.forward.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'bound for 64 rows'):
            forward(SimpleNamespace(shape=(1, 1, 32, 5120)))
        self.assertEqual((mlp.seen, forward.calls), ([False], 1))

        def failing(value):
            mlp.seen.append(mlp._fuse_gateup_agmm)
            raise RuntimeError('matmul failed')

        mlp._forward_tp = failing
        with self.assertRaisesRegex(RuntimeError, 'matmul failed'):
            forward(x)
        self.assertEqual(mlp.seen, [False, False])
        self.assertTrue(mlp._fuse_gateup_agmm is True, 'the flag is restored when the forward fails')
        self.assertEqual(forward.calls, 1)

    def test_construction_needs_the_tensor_parallel_mlp_with_its_fusion_switch(self):
        with self.assertRaisesRegex(ValueError, 'tensor-parallel MLP'):
            TwoTileMLPForward(fake_mlp(devices=1), 64)
        without = fake_mlp()
        del without._forward_tp
        with self.assertRaisesRegex(ValueError, 'tensor-parallel MLP'):
            TwoTileMLPForward(without, 64)

        class ClassFlag:
            num_devices = 2
            _fuse_gateup_agmm = True

            def _forward_tp(self, x):
                return x

        with self.assertRaisesRegex(ValueError, 'on the instance'):
            TwoTileMLPForward(ClassFlag(), 64)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            TwoTileMLPForward(fake_mlp(), 32)


class MLPBindingTests(unittest.TestCase):
    def test_one_forward_per_layer_bound_on_feed_forward(self):
        ttnn = FakeTTNN()
        model = fake_model(ttnn, layers=6, full=(3,))
        binding = bind_two_tile_mlp(model, 64)
        self.assertIsInstance(binding, TwoTileMLPBinding)
        self.assertEqual((binding.label, binding.rows, binding.expected_calls, binding.calls), ('MLP forward', 64, 6, 0))
        self.assertEqual([(instance, name) for instance, name, value in binding.bindings],
                         [(layer.feed_forward, 'forward') for layer in model.layers])
        natives = [layer.feed_forward.forward for layer in model.layers]
        with instance_overrides(binding.bindings):
            for layer in model.layers:
                self.assertEqual(layer.feed_forward.forward(SimpleNamespace(shape=(1, 1, 64, 5120)))[0], 'mlp-output')
            self.assertEqual(binding.calls, 6)
        self.assertEqual([layer.feed_forward.forward for layer in model.layers], natives)
        self.assertTrue(all(layer.feed_forward._fuse_gateup_agmm is True for layer in model.layers))
        with self.assertRaisesRegex(ValueError, 'every layer carries a feed_forward'):
            bind_two_tile_mlp(SimpleNamespace(layers=[SimpleNamespace(feed_forward=None)]), 64)
        with self.assertRaisesRegex(ValueError, 'every layer carries a feed_forward'):
            bind_two_tile_mlp(SimpleNamespace(layers=[]), 64)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            bind_two_tile_mlp(model, 32)


class ModelBatchWiringTests(unittest.TestCase):
    """model_batch builds the three binders only beyond one tile and checks every count per forward."""

    def test_within_one_tile_there_are_no_bindings_and_the_modules_are_never_imported(self):
        from model_batch import two_tile_bindings

        with patch.dict(sys.modules, {'two_tile_norm': None, 'two_tile_decode': None}):
            for rows in (1, 2, 4, 8, 16, 32):
                with self.subTest(rows=rows):
                    self.assertEqual(two_tile_bindings(rows, SimpleNamespace(), FakeTTNN()), ())
        with self.assertRaises(ValueError):
            two_tile_bindings(48, SimpleNamespace(), FakeTTNN())

    def test_beyond_one_tile_the_three_binders_are_built_in_order(self):
        from model_batch import two_tile_bindings
        from two_tile_norm import TwoTileNormBinding

        ttnn = FakeTTNN()
        model = fake_model(ttnn)
        norm, attention, mlp = two_tile_bindings(64, model, ttnn)
        self.assertIsInstance(norm, TwoTileNormBinding)
        self.assertIsInstance(attention, TwoTileAttentionBinding)
        self.assertIsInstance(mlp, TwoTileMLPBinding)
        self.assertEqual([binder.expected_calls for binder in (norm, attention, mlp)], [129, 16, 64])
        self.assertEqual([binder.label for binder in (norm, attention, mlp)], ['decode norm', 'full-attention forward', 'MLP forward'])

    def fixture(self, two_tile):
        from model_batch import ModelBatch

        fixture = ModelBatch.__new__(ModelBatch)
        fixture.retained = None
        fixture.gdn_calls = fixture.norm_batch_calls = 0
        fixture.norm_batch = fixture.compact_gdn = fixture.attention_replay = fixture.attention_mask_once = False
        fixture.working_states, fixture.writers, fixture.readers = [], [], []
        fixture.tokens, fixture.cos, fixture.sin, fixture.positions, fixture.pages = range(5)
        fixture.two_tile = two_tile
        fixture.bindings = [binding for binder in two_tile for binding in binder.bindings]
        return fixture

    def test_run_demands_every_two_tile_count_and_only_for_a_wide_block(self):
        from model_batch import two_tile_bindings

        ttnn = FakeTTNN()
        model = fake_model(ttnn, layers=3, full=(1,))
        two_tile = two_tile_bindings(64, model, ttnn)
        norm, attention, mlp = two_tile
        self.assertEqual([binder.expected_calls for binder in two_tile], [7, 1, 3])
        fixture = self.fixture(two_tile)
        decode = SimpleNamespace(name='DECODE')
        seen = dict(block_h=[], fusion=[])

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            for layer in model.layers:
                for unused in range(2):
                    seen['block_h'].append(model.args.get_norm_config('attn', decode)['sharded_program_config'].block_h)
                if layer.is_full_attention:
                    layer.attention.forward_decode(SimpleNamespace(shape=(1, 1, 64, 5120)), 'p', 'c', 's', page_table='pages')
                layer.feed_forward.forward(SimpleNamespace(shape=(1, 1, 64, 5120)))
                seen['fusion'].append(layer.feed_forward.seen[-1])
            seen['block_h'].append(model.args.get_norm_config('lm_head', decode)['sharded_program_config'].block_h)
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        self.assertEqual(fixture.run(), 'logits')
        self.assertEqual(seen, dict(block_h=[2] * 7, fusion=[False] * 3))
        self.assertEqual([binder.calls for binder in two_tile], [7, 1, 3])
        # everything is off again outside the forward
        self.assertEqual(model.args.get_norm_config('attn', decode)['sharded_program_config'].block_h, 1)
        self.assertEqual(model.args.attn_qkv_decode_1d_progcfg.per_core_M, 1)
        self.assertTrue(all(layer.feed_forward._fuse_gateup_agmm is True for layer in model.layers))
        # a forward that skips one MLP is refused by count, naming the binder
        short = Mock(side_effect=lambda *a, **k: (setattr(fixture, 'gdn_calls', fixture.gdn_calls + 48),
                                                   [model.args.get_norm_config('attn', decode) for unused in range(7)],
                                                   model.layers[1].attention.forward_decode(SimpleNamespace(shape=(1, 1, 64, 5120)), 'p', 'c', 's', page_table='pages'),
                                                   [layer.feed_forward.forward(SimpleNamespace(shape=(1, 1, 64, 5120))) for layer in model.layers[:2]]))
        fixture.model = SimpleNamespace(_forward_decode=short)
        with self.assertRaisesRegex(AssertionError, 'Every MLP forward of the wide block must take its two-tile form: 2 engaged, 3 expected'):
            fixture.run()
        # and one that skips the attention forward
        short = Mock(side_effect=lambda *a, **k: (setattr(fixture, 'gdn_calls', fixture.gdn_calls + 48),
                                                   [model.args.get_norm_config('attn', decode) for unused in range(7)],
                                                   [layer.feed_forward.forward(SimpleNamespace(shape=(1, 1, 64, 5120))) for layer in model.layers]))
        fixture.model = SimpleNamespace(_forward_decode=short)
        with self.assertRaisesRegex(AssertionError, 'Every full-attention forward of the wide block must take its two-tile form: 0 engaged, 1 expected'):
            fixture.run()
        # an unbound (one-tile) fixture asks nothing of any of them
        narrow = self.fixture(())
        narrow.model = SimpleNamespace(_forward_decode=Mock(side_effect=lambda *a, **k: (setattr(narrow, 'gdn_calls', 48), 'logits')[1]))
        self.assertEqual(narrow.run(), 'logits')


if __name__ == '__main__':
    unittest.main()
