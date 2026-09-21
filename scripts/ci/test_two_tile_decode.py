import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from model_batch import instance_overrides
from two_tile_decode import (NATIVE_ATTN_ENV, TwoTileAttentionBinding, TwoTileAttentionDecode, TwoTileConcatHeads,
                             TwoTileGDNOutputBinding, TwoTileMLPBinding, TwoTileMLPForward, TwoTileNativeAttnGuard,
                             TwoTileProjectionSplit, bind_two_tile_attention, bind_two_tile_gdn_output,
                             bind_two_tile_mlp, native_attn_enabled, prep_by_tile, prep_native,
                             two_tile_matmul_1d_progcfg, validate_two_tile_rows)


class FakeTTNN:
    """Device tensors carry a shape, a name and a placement, so the per-tile prep and the
    two-half head concat can be checked call by call, free by free."""

    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = 'dram', 'l1'

    class TensorMemoryLayout:
        WIDTH_SHARDED = 'width'

    class BufferType:
        L1, DRAM = 'l1', 'dram'

    def __init__(self, prep_limit=32):
        self.deallocated, self.calls = [], []
        self.prep_limit = prep_limit
        self.transformer = SimpleNamespace(attn_decode_prep=Mock(side_effect=self.prep))

    def tensor(self, shape, name, memory='dram'):
        return SimpleNamespace(shape=tuple(shape), name=name, memory_config=lambda: memory)

    def prep(self, qkv, cos, sin, q_norm, k_norm, NH, NKV, HD, rope_dim, config, *, batch, memory_config):
        """The model's op at one tile: q and the gate (1, B, NH, HD) interleaved, K and V
        (1, B, 32, HD) height-sharded per `config`. At batch 64 the stock kernel hung
        (v51); `prep_limit=None` stands in for the K64 graft's fixed kernel (proved
        bit-exact at B=64), which raises nothing here regardless of batch."""
        if self.prep_limit is not None and batch > self.prep_limit:
            raise RuntimeError('attn_decode_prep hangs at batch %d' % batch)
        tag = qkv.name
        return (self.tensor((1, batch, NH, HD), 'q(%s)' % tag, memory_config),
                self.tensor((1, batch, NH, HD), 'gate(%s)' % tag, memory_config),
                self.tensor((1, batch, 32, HD), 'k_sh(%s)' % tag, config),
                self.tensor((1, batch, 32, HD), 'v_sh(%s)' % tag, config))

    def slice(self, value, start, stop, memory_config=None):
        shape = tuple(b - a for a, b in zip(start, stop))
        axes = [index for index, (a, b) in enumerate(zip(start, stop)) if (a, b) != (0, value.shape[index])]
        axis = axes[0] if axes else 0
        self.calls.append(('slice', value.name, axis, start[axis], stop[axis], memory_config))
        return self.tensor(shape, '%s[%d:%d]' % (value.name, start[axis], stop[axis]), memory_config)

    def concat(self, parts, dim, memory_config=None):
        shape = list(parts[0].shape)
        shape[dim] = sum(part.shape[dim] for part in parts)
        self.calls.append(('concat', [part.name for part in parts], dim, memory_config))
        return self.tensor(shape, 'cat(%s)' % ','.join(part.name for part in parts), memory_config)

    def sharded_to_interleaved(self, value, memory_config):
        self.calls.append(('interleave', value.name, memory_config))
        return self.tensor(value.shape, 'il(%s)' % value.name, memory_config)

    def deallocate(self, value):
        if any(value is seen for seen in self.deallocated):
            raise AssertionError('Double free of %s' % getattr(value, 'name', value))
        self.deallocated.append(value)

    def freed(self):
        return [getattr(value, 'name', value) for value in self.deallocated]

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


def one_tile_progcfg(ttnn, grid=(8, 8), in0_block_w=8, out_subblock_w=1, per_core_N=2, fused_activation=None,
                     per_core_M=1, out_subblock_h=1, fuse_batch=True, mcast_in0=True):
    """tp_common.create_matmul_1d_decode_progcfg at M = 1, as model_config builds attn_qkv_decode_1d_progcfg."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=in0_block_w, out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w, per_core_M=per_core_M, per_core_N=per_core_N, fuse_batch=fuse_batch,
        fused_activation=fused_activation, mcast_in0=mcast_in0)


class FakeAttention:
    """The TPAttention surface the two-tile forward touches. `_decode_from_prep` stands for the
    block-bound tail: it calls the head concat and the output projection on `self`, as the
    model's does (attention/tp.py:749, :7xx)."""

    def __init__(self, args, ttnn, *, nlp_heads=True, concat_limit=32):
        self.args, self.ttnn = args, ttnn
        self.use_paged, self._fused_qkv, self._use_nlp_decode_heads = True, True, nlp_heads
        self.tw = {'q_norm': 'qn', 'k_norm': 'kn'}
        self.NH, self.NKV, self.HD, self.rope_dim = 12, 2, 256, 64
        self.concat_limit = concat_limit
        self._qkv_raw_decode = Mock(side_effect=lambda x: ttnn.tensor((1, 1, x.shape[-2], 7168), 'qkv_raw'))
        self._kv_shard_cfg = Mock(side_effect=lambda batch: 'kv-shard-%d' % batch)
        self._decode_from_prep = Mock(side_effect=self.tail)
        self.native_concats = []
        self.wo_calls = []

    def tail(self, q, gate, k_sh, v_sh, positions, pages, B):
        gated = self.ttnn.tensor((1, B, self.NH, self.HD), 'gated', 'l1')
        flat = self._concat_heads_decode(gated, B)
        projected = self._wo_proj(flat, 'wo-weight')
        self.ttnn.deallocate(flat)
        self.ttnn.deallocate(projected)
        return 'attention-output'

    def _concat_heads_decode(self, gated, B):
        """The model's method: consumes and frees its input, emits (1, B, NH * HD) in L1; its
        op refuses more than one tile of users (v51's host raise) unless `concat_limit` is
        raised past it (None stands in for the K64 graft's fixed kernel, proved bit-exact
        at B=64, which accepts any batch here)."""
        if self.concat_limit is not None and B > self.concat_limit:
            raise RuntimeError('TT_FATAL nlp_concat_heads_decode_device_operation.cpp:39: input_shape[1] <= 32')
        self.native_concats.append((gated.name, B))
        self.ttnn.deallocate(gated)
        return self.ttnn.tensor((1, B, self.NH * self.HD), 'heads(%s)' % gated.name, 'l1')

    def _wo_proj(self, flat, weight):
        """The model's method: does NOT free its input - attention/tp.py's `_decode_from_prep`
        frees `gated_flat` itself right after calling this (the same contract `_row_proj` has
        with gdn_multitoken_conv.finish_output). Above one tile it takes a slow prefill arm
        rather than refusing outright, but the two-tile wrapper must never reach it there."""
        if flat.shape[1] > 32:
            raise RuntimeError('wo_proj called above one tile: the two-tile split did not engage')
        self.wo_calls.append((flat.name, tuple(flat.shape), weight))
        return self.ttnn.tensor((1, flat.shape[1], 5120), 'wo(%s)' % flat.name, flat.memory_config())


def fake_attention(args, ttnn, **options):
    return FakeAttention(args, ttnn, **options)


def fake_mlp(ttnn, devices=2):
    mlp = SimpleNamespace(num_devices=devices)
    mlp._fuse_gateup_agmm = True
    mlp.seen = []

    def _forward_tp(x):
        """The model's method: reads its activation for both w1 and w3, then frees it -
        the same consume-and-free contract as `_concat_heads_decode`."""
        if x.shape[-2] > 32:
            raise RuntimeError('_forward_tp called above one tile: the two-tile split did not engage')
        mlp.seen.append((x.name, tuple(x.shape), x.memory_config(), mlp._fuse_gateup_agmm))
        result = ttnn.tensor((1, 1, x.shape[-2], 5120), 'mlp(%s)' % x.name, x.memory_config())
        ttnn.deallocate(x)
        return result

    mlp._forward_tp = _forward_tp
    mlp.forward = Mock(return_value='native-mlp-forward')
    return mlp


class FakeGDN:
    """The GDN surface the two-tile output-projection binding touches. `_row_proj` is looked
    up by attribute on this instance at call time by gdn_multitoken_conv.finish_output
    (FROZEN, not exercised directly here); it does not free its input, the same contract as
    `_wo_proj`."""

    def __init__(self, args, ttnn):
        self.args, self.ttnn = args, ttnn
        self.row_proj_calls = []

    def _row_proj(self, output, weight):
        if output.shape[1] > 32:
            raise RuntimeError('_row_proj called above one tile: the two-tile split did not engage')
        self.row_proj_calls.append((output.name, tuple(output.shape), weight))
        return self.ttnn.tensor((1, output.shape[1], 5120), 'row(%s)' % output.name, output.memory_config())


def fake_gdn(args, ttnn):
    return FakeGDN(args, ttnn)


def fake_model(ttnn, layers=64, full=(3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63), progcfg=None,
              native_m3=False, concat_limit=32):
    args = SimpleNamespace(proj_1d_decode=True, attn_qkv_decode_1d_progcfg=progcfg or one_tile_progcfg(ttnn))
    args.get_norm_config = Mock(side_effect=lambda name, mode: dict(
        sharded_output_config=ttnn.MemoryConfig('width', 'l1', ttnn.ShardSpec('g', [32, 160], 'rm')),
        sharded_program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 4), subblock_w=1, block_h=1, block_w=5, inplace=False),
        output_mem_config=None))
    if native_m3:
        # The Lever N M3native graft's model_config.py attributes: a distinct M=64
        # config object per projection, so tests can assert identity (the runtime
        # rebuild is skipped in favour of these) rather than shape alone.
        args.attn_qkv_decode_1d_progcfg_64 = one_tile_progcfg(ttnn, per_core_M=2, out_subblock_h=2)
        args.attn_wo_decode_1d_progcfg_64 = one_tile_progcfg(ttnn, per_core_M=2, out_subblock_h=2)
        args.gdn_qkvz_decode_1d_progcfg_64 = one_tile_progcfg(ttnn, per_core_M=2, out_subblock_h=2)
        args.gdn_out_decode_1d_progcfg_64 = one_tile_progcfg(ttnn, per_core_M=2, out_subblock_h=2)
        args.mlp_w1_decode_1d_progcfg_64 = one_tile_progcfg(ttnn, per_core_M=2, out_subblock_h=2)
        args.mlp_w3_decode_1d_progcfg_64 = one_tile_progcfg(ttnn, per_core_M=2, out_subblock_h=2)
        args.mlp_w2_decode_1d_progcfg_64 = one_tile_progcfg(ttnn, per_core_M=2, out_subblock_h=2)
    model = SimpleNamespace(args=args, layers=[])
    for index in range(layers):
        is_full = index in full
        model.layers.append(SimpleNamespace(is_full_attention=is_full,
                                            attention=fake_attention(args, ttnn, concat_limit=concat_limit) if is_full
                                            else fake_gdn(args, ttnn),
                                            feed_forward=fake_mlp(ttnn)))
    return model


def block_inputs(ttnn, rows=64):
    return (SimpleNamespace(shape=(1, 1, rows, 5120)), 'positions',
            ttnn.tensor((1, rows, 1, 64), 'cos'), ttnn.tensor((1, rows, 1, 64), 'sin'))


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


class NativeAttnDetectionTests(unittest.TestCase):
    """QWEN_FAST_NATIVE_ATTN gates native_attn_enabled - a RUNTIME/ENV fact (the K64
    kernel graft patches no Python source), never inferred from the model the way
    native_m3's hasattr check is."""

    def test_only_the_exact_value_one_engages_it(self):
        self.assertEqual(NATIVE_ATTN_ENV, 'QWEN_FAST_NATIVE_ATTN')
        self.assertFalse(native_attn_enabled({}))
        for off in ('0', 'true', 'True', 'yes', ' 1', '1 '):
            with self.subTest(value=off):
                self.assertFalse(native_attn_enabled({NATIVE_ATTN_ENV: off}))
        self.assertTrue(native_attn_enabled({NATIVE_ATTN_ENV: '1'}))

    def test_defaults_to_reading_the_real_process_environment(self):
        with patch.dict(os.environ, {NATIVE_ATTN_ENV: '1'}):
            self.assertTrue(native_attn_enabled())
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(native_attn_enabled())


class PrepNativeTests(unittest.TestCase):
    """ttnn.transformer.attn_decode_prep as ONE call at the block's full row count
    (QWEN_FAST_NATIVE_ATTN), once the K64 graft's kernel accepts batch > 32: no slicing,
    no per-tile join, `_kv_shard_cfg(rows)` rather than `_kv_shard_cfg(TILE)`."""

    def test_one_call_over_the_whole_projection_and_tables_no_slice_or_join(self):
        ttnn = FakeTTNN(prep_limit=None)
        attention = fake_attention(SimpleNamespace(proj_1d_decode=True), ttnn, concat_limit=None)
        qkv_raw = ttnn.tensor((1, 1, 64, 7168), 'qkv_raw')
        cos, sin = ttnn.tensor((1, 64, 1, 64), 'cos'), ttnn.tensor((1, 64, 1, 64), 'sin')
        q, gate, k_sh, v_sh = prep_native(ttnn, attention, qkv_raw, cos, sin, 64)
        attention._kv_shard_cfg.assert_called_once_with(64)
        prep = ttnn.transformer.attn_decode_prep
        prep.assert_called_once_with(qkv_raw, cos, sin, 'qn', 'kn', 12, 2, 256, 64, 'kv-shard-64',
                                     batch=64, memory_config='dram')
        self.assertEqual([(value.shape, value.name) for value in (q, gate, k_sh, v_sh)],
                         [((1, 64, 12, 256), 'q(qkv_raw)'), ((1, 64, 12, 256), 'gate(qkv_raw)'),
                          ((1, 64, 32, 256), 'k_sh(qkv_raw)'), ((1, 64, 32, 256), 'v_sh(qkv_raw)')])
        # unlike prep_by_tile, nothing is sliced, joined, owned or freed here
        self.assertEqual(ttnn.calls, [])
        self.assertEqual(ttnn.deallocated, [])

    def test_refuses_the_wrong_projection_or_table_geometry(self):
        ttnn = FakeTTNN(prep_limit=None)
        attention = fake_attention(SimpleNamespace(proj_1d_decode=True), ttnn)
        cos, sin = ttnn.tensor((1, 64, 1, 64), 'cos'), ttnn.tensor((1, 64, 1, 64), 'sin')
        with self.assertRaisesRegex(ValueError, 'fused QKV projection of a 64-row block'):
            prep_native(ttnn, attention, ttnn.tensor((1, 1, 32, 7168), 'qkv_raw'), cos, sin, 64)
        with self.assertRaisesRegex(ValueError, 'rotary tables of a 64-row block'):
            prep_native(ttnn, attention, ttnn.tensor((1, 1, 64, 7168), 'qkv_raw'), ttnn.tensor((1, 32, 1, 64), 'cos'), sin, 64)
        with self.assertRaisesRegex(ValueError, 'rotary tables of a 64-row block'):
            prep_native(ttnn, attention, ttnn.tensor((1, 1, 64, 7168), 'qkv_raw'), cos, ttnn.tensor((1, 64, 1, 32), 'sin'), 64)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            prep_native(ttnn, attention, ttnn.tensor((1, 1, 32, 7168), 'qkv_raw'), cos, sin, 32)
        ttnn.transformer.attn_decode_prep.assert_not_called()


class AttentionForwardTests(unittest.TestCase):
    """The forward at 64 rows: the projection once, the prep per 32-row tile, the outputs joined,
    the tail once with the joined outputs, and the tail's head concat as two halves."""

    def forward(self, ttnn=None, rows=64, **options):
        ttnn = ttnn or FakeTTNN()
        args = SimpleNamespace(proj_1d_decode=True)
        attention = fake_attention(args, ttnn, **options)
        progcfg = two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn), 64, ttnn)
        args.attn_qkv_decode_1d_progcfg = progcfg
        return ttnn, attention, TwoTileAttentionDecode(attention, rows, ttnn, progcfg)

    def test_the_forward_is_the_models_prep_path_once_per_tile_then_its_tail_with_the_joined_outputs(self):
        ttnn, attention, forward = self.forward()
        x, positions, cos, sin = block_inputs(ttnn)
        with instance_overrides([(attention, '_concat_heads_decode', forward.concat), (attention, '_wo_proj', forward.wo)]):
            self.assertEqual(forward(x, positions, cos, sin, page_table='pages'), 'attention-output')
        attention._qkv_raw_decode.assert_called_once_with(x)
        # the K/V shard is the 32-row block's (8x4, one user per core), never a 64-user one
        attention._kv_shard_cfg.assert_called_once_with(32)
        prep = ttnn.transformer.attn_decode_prep
        self.assertEqual(prep.call_count, 2)
        for call, (first, last) in zip(prep.call_args_list, ((0, 32), (32, 64))):
            piece, cos_piece, sin_piece = call.args[:3]
            self.assertEqual((piece.shape, piece.name), ((1, 1, 32, 7168), 'qkv_raw[%d:%d]' % (first, last)))
            self.assertEqual((cos_piece.shape, cos_piece.name), ((1, 32, 1, 64), 'cos[%d:%d]' % (first, last)))
            self.assertEqual((sin_piece.shape, sin_piece.name), ((1, 32, 1, 64), 'sin[%d:%d]' % (first, last)))
            self.assertEqual(call.args[3:], ('qn', 'kn', 12, 2, 256, 64, 'kv-shard-32'))
            self.assertEqual(call.kwargs, dict(batch=32, memory_config='dram'))
        # every slice is whole tiles: the projection at its tile rows, the tables, the gated
        # SDPA output and the wo-projection's activation, each per 32-row/user tile
        flat_name = 'heads(gated[0:32]),heads(gated[32:64])'
        flat_name = 'cat(%s)' % flat_name
        self.assertEqual([entry for entry in ttnn.calls if entry[0] == 'slice'],
                         [('slice', 'qkv_raw', 2, 0, 32, 'dram'), ('slice', 'cos', 1, 0, 32, 'dram'),
                          ('slice', 'sin', 1, 0, 32, 'dram'), ('slice', 'qkv_raw', 2, 32, 64, 'dram'),
                          ('slice', 'cos', 1, 32, 64, 'dram'), ('slice', 'sin', 1, 32, 64, 'dram'),
                          ('slice', 'gated', 1, 0, 32, 'l1'), ('slice', 'gated', 1, 32, 64, 'l1'),
                          ('slice', flat_name, 1, 0, 32, 'l1'), ('slice', flat_name, 1, 32, 64, 'l1')])
        # K and V back to interleaved DRAM per tile, then every output joined on the user axis
        self.assertEqual([entry for entry in ttnn.calls if entry[0] == 'interleave'],
                         [('interleave', 'k_sh(qkv_raw[0:32])', 'dram'), ('interleave', 'v_sh(qkv_raw[0:32])', 'dram'),
                          ('interleave', 'k_sh(qkv_raw[32:64])', 'dram'), ('interleave', 'v_sh(qkv_raw[32:64])', 'dram')])
        wo_names = ['wo(%s[%d:%d])' % (flat_name, first, first + 32) for first in (0, 32)]
        self.assertEqual([entry for entry in ttnn.calls if entry[0] == 'concat'],
                         [('concat', ['q(qkv_raw[0:32])', 'q(qkv_raw[32:64])'], 1, 'dram'),
                          ('concat', ['gate(qkv_raw[0:32])', 'gate(qkv_raw[32:64])'], 1, 'dram'),
                          ('concat', ['il(k_sh(qkv_raw[0:32]))', 'il(k_sh(qkv_raw[32:64]))'], 1, 'dram'),
                          ('concat', ['il(v_sh(qkv_raw[0:32]))', 'il(v_sh(qkv_raw[32:64]))'], 1, 'dram'),
                          ('concat', ['heads(gated[0:32])', 'heads(gated[32:64])'], 1, 'l1'),
                          ('concat', wo_names, 1, 'l1')])
        tail = attention._decode_from_prep
        tail.assert_called_once()
        q, gate, k, v, positions_given, pages, batch = tail.call_args.args
        self.assertEqual([(value.shape, value.memory_config()) for value in (q, gate, k, v)],
                         [((1, 64, 12, 256), 'dram'), ((1, 64, 12, 256), 'dram'),
                          ((1, 64, 32, 256), 'dram'), ((1, 64, 32, 256), 'dram')])
        self.assertEqual((positions_given, pages, batch), ('positions', 'pages', 64))
        # the tail's head concat ran as two 32-user halves of the model's own method, and its
        # output projection ran as two 32-row halves of the model's own method
        self.assertEqual(attention.native_concats, [('gated[0:32]', 32), ('gated[32:64]', 32)])
        self.assertEqual([(name, shape) for name, shape, weight in attention.wo_calls],
                         [(flat_name + '[0:32]', (1, 32, 3072)), (flat_name + '[32:64]', (1, 32, 3072))])
        self.assertTrue(all(weight == 'wo-weight' for name, shape, weight in attention.wo_calls))
        self.assertEqual((forward.calls, forward.concat.calls, forward.wo.calls), (1, 1, 1))
        # freed here: the projection, its tile pieces, the table pieces, every per-tile prep output
        # and interleaved half, the concat's input/halves/half-outputs, and the wo projection's
        # halves and half-outputs; the joined heads and the joined wo output are the tail's (the
        # fake tail frees both, as _decode_from_prep and its all_reduce would)
        wo_joined_name = 'cat(%s)' % ','.join(wo_names)
        expected_freed = ('qkv_raw', 'qkv_raw[0:32]', 'qkv_raw[32:64]', 'cos[0:32]', 'cos[32:64]', 'sin[0:32]', 'sin[32:64]',
                          'q(qkv_raw[0:32])', 'gate(qkv_raw[0:32])', 'k_sh(qkv_raw[0:32])', 'v_sh(qkv_raw[0:32])',
                          'q(qkv_raw[32:64])', 'gate(qkv_raw[32:64])', 'k_sh(qkv_raw[32:64])', 'v_sh(qkv_raw[32:64])',
                          'il(k_sh(qkv_raw[0:32]))', 'il(v_sh(qkv_raw[0:32]))', 'il(k_sh(qkv_raw[32:64]))',
                          'il(v_sh(qkv_raw[32:64]))', 'gated', 'gated[0:32]', 'gated[32:64]',
                          'heads(gated[0:32])', 'heads(gated[32:64])', flat_name,
                          flat_name + '[0:32]', flat_name + '[32:64]', wo_names[0], wo_names[1], wo_joined_name)
        freed = ttnn.freed()
        for name in expected_freed:
            self.assertIn(name, freed)
        # nothing else was freed: exactly these, once each (no leak, no double free)
        self.assertEqual(sorted(freed), sorted(expected_freed))

    def test_a_tail_that_skips_the_head_concat_is_refused(self):
        ttnn, attention, forward = self.forward()
        attention._decode_from_prep = Mock(return_value='no-concat')
        x, positions, cos, sin = block_inputs(ttnn)
        with self.assertRaisesRegex(AssertionError, 'two-tile head concat exactly once; 0 taken'):
            forward(x, positions, cos, sin, page_table='pages')
        self.assertEqual(forward.calls, 0)

    def test_a_failed_second_prep_frees_the_first_tiles_outputs_the_pieces_and_the_projection(self):
        ttnn, attention, forward = self.forward()
        seen = []

        def prep(qkv, *args, **kwargs):
            seen.append(qkv.name)
            if len(seen) == 2:
                raise RuntimeError('second tile refused')
            return ttnn.prep(qkv, *args, **kwargs)

        ttnn.transformer.attn_decode_prep = Mock(side_effect=prep)
        x, positions, cos, sin = block_inputs(ttnn)
        with self.assertRaisesRegex(RuntimeError, 'second tile refused'):
            forward(x, positions, cos, sin, page_table='pages')
        freed = ttnn.freed()
        for name in ('qkv_raw', 'qkv_raw[0:32]', 'qkv_raw[32:64]', 'cos[0:32]', 'cos[32:64]', 'sin[0:32]', 'sin[32:64]',
                     'q(qkv_raw[0:32])', 'gate(qkv_raw[0:32])', 'k_sh(qkv_raw[0:32])', 'v_sh(qkv_raw[0:32])',
                     'il(k_sh(qkv_raw[0:32]))', 'il(v_sh(qkv_raw[0:32]))'):
            self.assertIn(name, freed)
        self.assertFalse(any(entry[0] == 'concat' for entry in ttnn.calls))
        attention._decode_from_prep.assert_not_called()
        self.assertEqual(forward.calls, 0)

    def test_a_failed_join_frees_what_was_joined_so_far(self):
        ttnn, attention, forward = self.forward()
        original = ttnn.concat

        def concat(parts, dim, memory_config=None):
            if parts[0].name.startswith('gate('):
                raise RuntimeError('join refused')
            return original(parts, dim, memory_config)

        ttnn.concat = concat
        x, positions, cos, sin = block_inputs(ttnn)
        with self.assertRaisesRegex(RuntimeError, 'join refused'):
            forward(x, positions, cos, sin, page_table='pages')
        self.assertIn('cat(q(qkv_raw[0:32]),q(qkv_raw[32:64]))', ttnn.freed())
        self.assertIn('qkv_raw', ttnn.freed())
        attention._decode_from_prep.assert_not_called()
        self.assertEqual(forward.calls, 0)

    def test_prep_by_tile_refuses_the_wrong_projection_or_table_geometry(self):
        ttnn, attention, forward = self.forward()
        cos, sin = ttnn.tensor((1, 64, 1, 64), 'cos'), ttnn.tensor((1, 64, 1, 64), 'sin')
        with self.assertRaisesRegex(ValueError, 'fused QKV projection of a 64-row block'):
            prep_by_tile(ttnn, attention, ttnn.tensor((1, 1, 32, 7168), 'qkv_raw'), cos, sin, 64)
        with self.assertRaisesRegex(ValueError, 'rotary tables of a 64-row block'):
            prep_by_tile(ttnn, attention, ttnn.tensor((1, 1, 64, 7168), 'qkv_raw'), ttnn.tensor((1, 32, 1, 64), 'cos'), sin, 64)
        with self.assertRaisesRegex(ValueError, 'rotary tables of a 64-row block'):
            prep_by_tile(ttnn, attention, ttnn.tensor((1, 1, 64, 7168), 'qkv_raw'), cos, ttnn.tensor((1, 64, 1, 32), 'sin'), 64)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            prep_by_tile(ttnn, attention, ttnn.tensor((1, 1, 32, 7168), 'qkv_raw'), cos, sin, 32)
        ttnn.transformer.attn_decode_prep.assert_not_called()
        self.assertEqual(ttnn.deallocated, [])

    def test_a_forward_refuses_the_wrong_rows_an_unpaged_call_or_an_unbound_config_before_any_op(self):
        ttnn, attention, forward = self.forward()
        x, positions, cos, sin = block_inputs(ttnn)
        with self.assertRaisesRegex(ValueError, 'bound for 64 rows'):
            forward(SimpleNamespace(shape=(1, 1, 32, 5120)), positions, cos, sin, page_table='pages')
        with self.assertRaisesRegex(ValueError, 'paged decode only'):
            forward(x, positions, cos, sin)
        attention.args.attn_qkv_decode_1d_progcfg = one_tile_progcfg(ttnn)
        with self.assertRaisesRegex(AssertionError, 'not bound on the model args'):
            forward(x, positions, cos, sin, page_table='pages')
        attention._qkv_raw_decode.assert_not_called()
        ttnn.transformer.attn_decode_prep.assert_not_called()
        self.assertEqual(forward.calls, 0)

    def test_construction_needs_the_paged_fused_1d_nlp_heads_attention_the_serving_model_runs(self):
        ttnn = FakeTTNN()
        progcfg = one_tile_progcfg(ttnn)
        for name, value, message in (('use_paged', False, 'paged fused-QKV'), ('_fused_qkv', False, 'paged fused-QKV'),
                                     ('tw', {'q_norm': 'qn'}, 'q_norm and k_norm'), ('NH', 0, 'integer NH'),
                                     ('_decode_from_prep', None, 'no longer exposes _decode_from_prep'),
                                     ('_use_nlp_decode_heads', False, 'nlp decode-heads path')):
            attention = fake_attention(SimpleNamespace(proj_1d_decode=True), ttnn)
            setattr(attention, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                TwoTileAttentionDecode(attention, 64, ttnn, progcfg)
        with self.assertRaisesRegex(ValueError, '1D decode projection'):
            TwoTileAttentionDecode(fake_attention(SimpleNamespace(proj_1d_decode=False), ttnn), 64, ttnn, progcfg)
        with self.assertRaisesRegex(ValueError, 'attn_decode_prep is required'):
            TwoTileAttentionDecode(fake_attention(SimpleNamespace(proj_1d_decode=True), ttnn), 64, SimpleNamespace(), progcfg)
        bare = SimpleNamespace(transformer=ttnn.transformer, slice=ttnn.slice, concat=ttnn.concat)
        with self.assertRaisesRegex(ValueError, 'slice, concat and sharded_to_interleaved'):
            TwoTileAttentionDecode(fake_attention(SimpleNamespace(proj_1d_decode=True), ttnn), 64, bare, progcfg)

        class NoConcat(FakeAttention):
            _concat_heads_decode = None

        with self.assertRaisesRegex(ValueError, 'no longer defines _concat_heads_decode'):
            TwoTileAttentionDecode(NoConcat(SimpleNamespace(proj_1d_decode=True), ttnn), 64, ttnn, progcfg)

        class NoWo(FakeAttention):
            _wo_proj = None

        with self.assertRaisesRegex(ValueError, 'no longer defines _wo_proj'):
            TwoTileAttentionDecode(NoWo(SimpleNamespace(proj_1d_decode=True), ttnn), 64, ttnn, progcfg)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            TwoTileAttentionDecode(fake_attention(SimpleNamespace(proj_1d_decode=True), ttnn), 32, ttnn, progcfg)


class NativeAttnAttentionForwardTests(unittest.TestCase):
    """QWEN_FAST_NATIVE_ATTN: the forward at 64 rows runs the K64 graft's kernels once
    each - one attn_decode_prep at batch 64, one model-native _concat_heads_decode at 64
    users - instead of the sliced two-call forms; `_wo_proj` two-tile splitting (native_m3
    territory) is unaffected either way."""

    def forward(self, ttnn=None, rows=64, native_m3=False, **options):
        ttnn = ttnn or FakeTTNN(prep_limit=None)
        args = SimpleNamespace(proj_1d_decode=True)
        attention = fake_attention(args, ttnn, concat_limit=None, **options)
        progcfg = two_tile_matmul_1d_progcfg(one_tile_progcfg(ttnn), 64, ttnn)
        args.attn_qkv_decode_1d_progcfg = progcfg
        return ttnn, attention, TwoTileAttentionDecode(attention, rows, ttnn, progcfg,
                                                       native_m3=native_m3, native_attn=True)

    def test_one_native_prep_call_and_the_models_own_concat_at_the_full_batch(self):
        ttnn, attention, forward = self.forward()
        self.assertTrue(forward.native_attn)
        x, positions, cos, sin = block_inputs(ttnn)
        with instance_overrides([(attention, '_wo_proj', forward.wo)]):
            self.assertEqual(forward(x, positions, cos, sin, page_table='pages'), 'attention-output')
        attention._qkv_raw_decode.assert_called_once_with(x)
        # the 8x8 (64-user) shard config, never the 32-user one prep_by_tile takes
        attention._kv_shard_cfg.assert_called_once_with(64)
        prep = ttnn.transformer.attn_decode_prep
        prep.assert_called_once()
        qkv_arg, cos_arg, sin_arg = prep.call_args.args[:3]
        # the whole projection and tables, unsliced - qkv_raw's own name/shape, not a
        # 32-row piece of it
        self.assertEqual((qkv_arg.name, qkv_arg.shape), ('qkv_raw', (1, 1, 64, 7168)))
        self.assertIs(cos_arg, cos)
        self.assertIs(sin_arg, sin)
        self.assertEqual(prep.call_args.kwargs, dict(batch=64, memory_config='dram'))
        # no per-tile slicing or joining of the projection, tables or gated SDPA output -
        # exact-name match, so wo's own two-tile split (native_m3 territory, unaffected
        # by native_attn) over its DIFFERENTLY-named 'heads(gated)' activation is not
        # mistaken for a leak here
        disallowed = {'qkv_raw', 'cos', 'sin', 'gated'}
        for entry in ttnn.calls:
            if entry[0] == 'slice':
                self.assertNotIn(entry[1], disallowed)
            elif entry[0] == 'concat':
                self.assertFalse(disallowed.intersection(entry[1]))
        tail = attention._decode_from_prep
        tail.assert_called_once()
        # the model's own concat ran once, at the full 64-user batch, never through the
        # retired TwoTileConcatHeads wrapper
        self.assertEqual(attention.native_concats, [('gated', 64)])
        self.assertEqual(attention.wo_calls[0][1], (1, 32, 3072))  # wo stays two-tile, unaffected
        self.assertEqual(len(attention.wo_calls), 2)
        self.assertEqual((forward.calls, forward.prep_calls, forward.concat.calls, forward.wo.calls), (1, 0, 0, 1))

    def test_legacy_slicing_never_engages_even_though_the_wrapper_objects_still_exist(self):
        ttnn, attention, forward = self.forward()
        self.assertIsInstance(forward.concat, TwoTileConcatHeads, 'kept as an inert leak counter')
        x, positions, cos, sin = block_inputs(ttnn)
        with instance_overrides([(attention, '_wo_proj', forward.wo)]):
            forward(x, positions, cos, sin, page_table='pages')
        # the retired wrapper was never bound, so it was never called either
        self.assertFalse('_concat_heads_decode' in attention.__dict__)
        self.assertEqual(forward.concat.calls, 0)
        self.assertEqual(forward.prep_calls, 0)

    def test_a_tail_that_skips_the_concat_is_not_flagged_here_under_native_attn(self):
        """Without native_attn a skipped concat fails loudly inside the forward itself
        (test_a_tail_that_skips_the_head_concat_is_refused); under native_attn the
        concat is the model's own unbound method, so this forward has nothing local left
        to check - model_batch.run()'s prep/concat guards catch a leak instead, at the
        binder level. native_m3=True here too, so there is no wo check left to confound
        this: native_attn's own concat check is what is under test."""
        ttnn, attention, forward = self.forward(native_m3=True)
        attention._decode_from_prep = Mock(return_value='no-concat')
        x, positions, cos, sin = block_inputs(ttnn)
        self.assertEqual(forward(x, positions, cos, sin, page_table='pages'), 'no-concat')
        self.assertEqual(forward.calls, 1)


class ConcatHeadsTests(unittest.TestCase):
    """The head concat at 64 users: two 32-user halves of the model's own method, whose op
    refuses more than one tile (v51), joined on the user axis in L1."""

    def concat(self, ttnn=None, cls=FakeAttention):
        ttnn = ttnn or FakeTTNN()
        attention = cls(SimpleNamespace(proj_1d_decode=True), ttnn)
        return ttnn, attention, TwoTileConcatHeads(attention, 64, ttnn)

    def test_two_halves_of_the_models_method_joined_on_the_user_axis(self):
        ttnn, attention, concat = self.concat()
        gated = ttnn.tensor((1, 64, 12, 256), 'gated', 'l1')
        joined = concat(gated, 64)
        self.assertEqual((joined.shape, joined.memory_config()), ((1, 64, 3072), 'l1'))
        self.assertEqual([entry for entry in ttnn.calls if entry[0] == 'slice'],
                         [('slice', 'gated', 1, 0, 32, 'l1'), ('slice', 'gated', 1, 32, 64, 'l1')])
        self.assertEqual(attention.native_concats, [('gated[0:32]', 32), ('gated[32:64]', 32)])
        self.assertEqual([entry for entry in ttnn.calls if entry[0] == 'concat'],
                         [('concat', ['heads(gated[0:32])', 'heads(gated[32:64])'], 1, 'l1')])
        # consumed: the input (the native contract), each half (by the native method), each
        # half's output once joined; the joined output is the caller's
        self.assertEqual(ttnn.freed(), ['gated', 'gated[0:32]', 'gated[32:64]', 'heads(gated[0:32])', 'heads(gated[32:64])'])
        self.assertEqual(concat.calls, 1)

    def test_the_batch_and_the_shape_must_be_the_blocks(self):
        ttnn, attention, concat = self.concat()
        for shape, batch in (((1, 32, 12, 256), 32), ((1, 64, 12, 256), 32), ((1, 32, 12, 256), 64), ((64, 12, 256), 64)):
            with self.subTest(shape=shape, batch=batch), self.assertRaisesRegex(ValueError, 'bound for 64 users'):
                concat(ttnn.tensor(shape, 'gated', 'l1'), batch)
        self.assertEqual((ttnn.calls, ttnn.deallocated, attention.native_concats, concat.calls), ([], [], [], 0))

    def test_a_native_failure_frees_only_the_halves_not_yet_handed_over_and_never_twice(self):
        class FailsFirst(FakeAttention):
            def _concat_heads_decode(self, gated, B):
                raise RuntimeError('concat refused')

        ttnn, attention, concat = self.concat(cls=FailsFirst)
        with self.assertRaisesRegex(RuntimeError, 'concat refused'):
            concat(ttnn.tensor((1, 64, 12, 256), 'gated', 'l1'), 64)
        # the input (consumed) and the second half (never handed over); the failing half's
        # state is the native method's to know, so it is left alone
        self.assertEqual(ttnn.freed(), ['gated', 'gated[32:64]'])
        self.assertEqual(concat.calls, 0)

        class FailsSecond(FakeAttention):
            def _concat_heads_decode(self, gated, B):
                if self.native_concats:
                    raise RuntimeError('second concat refused')
                return super()._concat_heads_decode(gated, B)

        ttnn, attention, concat = self.concat(cls=FailsSecond)
        with self.assertRaisesRegex(RuntimeError, 'second concat refused'):
            concat(ttnn.tensor((1, 64, 12, 256), 'gated', 'l1'), 64)
        # the first half's output is freed on the way out; nothing was joined
        self.assertEqual(ttnn.freed(), ['gated', 'gated[0:32]', 'heads(gated[0:32])'])
        self.assertFalse(any(entry[0] == 'concat' for entry in ttnn.calls))
        self.assertEqual(concat.calls, 0)

    def test_construction_needs_the_nlp_heads_path_and_the_native_method(self):
        ttnn = FakeTTNN()
        with self.assertRaisesRegex(ValueError, 'nlp decode-heads path'):
            TwoTileConcatHeads(FakeAttention(SimpleNamespace(), ttnn, nlp_heads=False), 64, ttnn)

        class NoConcat(FakeAttention):
            _concat_heads_decode = None

        with self.assertRaisesRegex(ValueError, 'no longer defines _concat_heads_decode'):
            TwoTileConcatHeads(NoConcat(SimpleNamespace(), ttnn), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            TwoTileConcatHeads(FakeAttention(SimpleNamespace(), ttnn), 32, ttnn)


class ProjectionSplitTests(unittest.TestCase):
    """The (activation, weight) -> tensor split shared by `_row_proj` (GDN) and `_wo_proj`
    (attention): two 32-row calls to the model's own method, joined on the row axis.
    Neither native method frees the activation it is given - `_row_proj`'s frozen caller
    (gdn_multitoken_conv.finish_output) frees it right after calling it - so unlike
    TwoTileConcatHeads and TwoTileMLPForward, the whole activation is left alone here."""

    def split(self, ttnn=None, cls=FakeGDN, name='_row_proj', rows=64):
        ttnn = ttnn or FakeTTNN()
        instance = cls(SimpleNamespace(), ttnn)
        return ttnn, instance, TwoTileProjectionSplit(instance, name, rows, ttnn)

    def test_two_halves_of_the_models_method_joined_on_the_row_axis(self):
        ttnn, gdn, split = self.split()
        output = ttnn.tensor((1, 64, 3072), 'output', 'l1')
        joined = split(output, 'out-weight')
        self.assertEqual((joined.shape, joined.memory_config()), ((1, 64, 5120), 'l1'))
        self.assertEqual([entry for entry in ttnn.calls if entry[0] == 'slice'],
                         [('slice', 'output', 1, 0, 32, 'l1'), ('slice', 'output', 1, 32, 64, 'l1')])
        self.assertEqual([(name, shape) for name, shape, weight in gdn.row_proj_calls],
                         [('output[0:32]', (1, 32, 3072)), ('output[32:64]', (1, 32, 3072))])
        self.assertTrue(all(weight == 'out-weight' for name, shape, weight in gdn.row_proj_calls))
        self.assertEqual([entry for entry in ttnn.calls if entry[0] == 'concat'],
                         [('concat', ['row(output[0:32])', 'row(output[32:64])'], 1, 'l1')])
        # the whole activation is left alone - its frozen or model-owned caller frees it -
        # only the halves and their native outputs are freed here; the joined result is
        # the caller's
        freed = ttnn.freed()
        self.assertNotIn('output', freed)
        self.assertEqual(sorted(freed), sorted(['output[0:32]', 'output[32:64]', 'row(output[0:32])', 'row(output[32:64])']))
        self.assertEqual(split.calls, 1)

    def test_the_batch_and_the_shape_must_be_the_blocks(self):
        ttnn, gdn, split = self.split()
        for shape in ((1, 32, 3072), (1, 64), (64, 3072), (1, 64, 3072, 1)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'bound for 64 rows'):
                split(ttnn.tensor(shape, 'bad', 'l1'), 'w')
        self.assertEqual((gdn.row_proj_calls, ttnn.calls, ttnn.deallocated), ([], [], []))

    def test_a_native_failure_leaves_the_failing_half_alone(self):
        class FailsFirst(FakeGDN):
            def _row_proj(self, output, weight):
                raise RuntimeError('projection refused')

        ttnn, gdn, split = self.split(cls=FailsFirst)
        output = ttnn.tensor((1, 64, 3072), 'output', 'l1')
        with self.assertRaisesRegex(RuntimeError, 'projection refused'):
            split(output, 'w')
        # only the one half already cut is freed; the half handed to the failing call and
        # the untouched second half are left alone
        self.assertEqual(ttnn.freed(), ['output[0:32]'])
        self.assertFalse(any(entry[0] == 'concat' for entry in ttnn.calls))
        self.assertEqual(split.calls, 0)

        class FailsSecond(FakeGDN):
            def _row_proj(self, output, weight):
                if self.row_proj_calls:
                    raise RuntimeError('second half refused')
                return super()._row_proj(output, weight)

        ttnn, gdn, split = self.split(cls=FailsSecond)
        output = ttnn.tensor((1, 64, 3072), 'output', 'l1')
        with self.assertRaisesRegex(RuntimeError, 'second half refused'):
            split(output, 'w')
        self.assertEqual(sorted(ttnn.freed()), sorted(['output[0:32]', 'row(output[0:32])', 'output[32:64]']))
        self.assertFalse(any(entry[0] == 'concat' for entry in ttnn.calls))
        self.assertEqual(split.calls, 0)

    def test_a_failed_join_frees_both_halves_native_outputs(self):
        ttnn, gdn, split = self.split()
        output = ttnn.tensor((1, 64, 3072), 'output', 'l1')

        def concat(parts, dim, memory_config=None):
            raise RuntimeError('join refused')

        ttnn.concat = concat
        with self.assertRaisesRegex(RuntimeError, 'join refused'):
            split(output, 'w')
        self.assertEqual(sorted(ttnn.freed()),
                         sorted(['output[0:32]', 'output[32:64]', 'row(output[0:32])', 'row(output[32:64])']))
        self.assertEqual(split.calls, 0)

    def test_construction_needs_the_named_method_and_whole_tiles_beyond_one(self):
        ttnn = FakeTTNN()

        class NoProj(FakeGDN):
            _row_proj = None

        with self.assertRaisesRegex(ValueError, 'no longer defines _row_proj'):
            TwoTileProjectionSplit(NoProj(SimpleNamespace(), ttnn), '_row_proj', 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            TwoTileProjectionSplit(FakeGDN(SimpleNamespace(), ttnn), '_row_proj', 32, ttnn)


class AttentionBindingTests(unittest.TestCase):
    def test_the_binding_is_the_rebuilt_config_on_the_args_and_a_forward_and_a_head_concat_per_full_attention_layer(self):
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
        expected = []
        for attention in full:
            expected += [(attention, 'forward_decode'), (attention, '_concat_heads_decode'), (attention, '_wo_proj')]
        self.assertEqual([(instance, name) for instance, name, value in binding.bindings[1:]], expected)
        forwards = [value for instance, name, value in binding.bindings[1:] if name == 'forward_decode']
        concats = [value for instance, name, value in binding.bindings[1:] if name == '_concat_heads_decode']
        wos = [value for instance, name, value in binding.bindings[1:] if name == '_wo_proj']
        self.assertTrue(all(isinstance(value, TwoTileAttentionDecode) for value in forwards))
        self.assertEqual(concats, [forward.concat for forward in forwards])
        self.assertEqual(wos, [forward.wo for forward in forwards])
        self.assertTrue(all(isinstance(value, TwoTileConcatHeads) for value in concats))
        self.assertTrue(all(isinstance(value, TwoTileProjectionSplit) for value in wos))
        with instance_overrides(binding.bindings):
            self.assertIs(model.args.attn_qkv_decode_1d_progcfg, binding.progcfg)
            for attention in full:
                x, positions, cos, sin = block_inputs(ttnn)
                attention.forward_decode(x, positions, cos, sin, page_table='pages')
            self.assertEqual(binding.calls, 16)
            self.assertEqual(sum(forward.concat.calls for forward in forwards), 16)
            self.assertEqual(sum(forward.wo.calls for forward in forwards), 16)
        self.assertIs(model.args.attn_qkv_decode_1d_progcfg, native)
        self.assertFalse(any(name in attention.__dict__ for attention in full
                             for name in ('forward_decode', '_concat_heads_decode', '_wo_proj')))
        # every layer's tail concatenated its heads as two halves of the model's method, and
        # projected its output as two 32-row halves of the model's own method
        self.assertTrue(all(attention.native_concats == [('gated[0:32]', 32), ('gated[32:64]', 32)] for attention in full))
        self.assertTrue(all(len(attention.wo_calls) == 2 for attention in full))

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


class NativeM3AttentionBindingTests(unittest.TestCase):
    """Lever N M3native: once model_batch.two_tile_bindings passes native_m3=True (a
    model whose args carry attn_wo_decode_1d_progcfg_64), the attention binder uses the
    graft's own M=64 config directly - no runtime rebuild - and drops its _wo_proj
    wrapping; _wo_proj is native at the block's rows once the graft's patched
    attention/tp.py widens its own gate."""

    def test_the_binding_uses_the_grafts_own_64_config_without_a_runtime_rebuild_and_drops_wo(self):
        ttnn = FakeTTNN()
        model = fake_model(ttnn, native_m3=True)
        native_64 = model.args.attn_qkv_decode_1d_progcfg_64
        original_progcfg = model.args.attn_qkv_decode_1d_progcfg
        binding = bind_two_tile_attention(model, 64, ttnn, native_m3=True)
        self.assertIsInstance(binding, TwoTileAttentionBinding)
        self.assertIs(binding.progcfg, native_64, "native_m3 uses the graft's own M=64 config directly, no rebuild")
        self.assertEqual((binding.rows, binding.expected_calls, binding.calls), (64, 16, 0))
        full = [layer.attention for layer in model.layers if layer.is_full_attention]
        expected = []
        for attention in full:
            expected += [(attention, 'forward_decode'), (attention, '_concat_heads_decode')]
        self.assertEqual([(instance, name) for instance, name, value in binding.bindings[1:]], expected,
                         'no _wo_proj binding under native_m3: the model source is native there')
        forwards = [value for instance, name, value in binding.bindings[1:] if name == 'forward_decode']
        self.assertTrue(forwards and all(forward.wo is None for forward in forwards))
        # bound and unbound cleanly, even with no _wo_proj entry to restore
        with instance_overrides(binding.bindings):
            self.assertIs(model.args.attn_qkv_decode_1d_progcfg, native_64)
            self.assertTrue(all(attention.forward_decode is forward
                                for attention, forward in zip(full, forwards)))
        self.assertIs(model.args.attn_qkv_decode_1d_progcfg, original_progcfg)
        self.assertFalse(any(name in attention.__dict__ for attention in full
                             for name in ('forward_decode', '_concat_heads_decode', '_wo_proj')))

    def test_native_m3_requires_the_grafts_64_config_on_the_model_args(self):
        ttnn = FakeTTNN()
        model = fake_model(ttnn)  # no _64 attrs at all
        with self.assertRaisesRegex(ValueError, 'attn_qkv_decode_1d_progcfg_64'):
            bind_two_tile_attention(model, 64, ttnn, native_m3=True)

    def test_native_m3_false_is_unaffected_even_when_the_64_attrs_happen_to_be_present(self):
        """The switch is explicit (model_batch's hasattr detection passes it down), not
        inferred by the binder itself from the attrs' mere presence."""
        ttnn = FakeTTNN()
        model = fake_model(ttnn, native_m3=True)
        binding = bind_two_tile_attention(model, 64, ttnn)
        self.assertIsNot(binding.progcfg, model.args.attn_qkv_decode_1d_progcfg_64)
        self.assertEqual(binding.progcfg.per_core_M, 2)
        full = [layer.attention for layer in model.layers if layer.is_full_attention]
        expected = []
        for attention in full:
            expected += [(attention, 'forward_decode'), (attention, '_concat_heads_decode'), (attention, '_wo_proj')]
        self.assertEqual([(instance, name) for instance, name, value in binding.bindings[1:]], expected)


class NativeAttnAttentionBindingTests(unittest.TestCase):
    """QWEN_FAST_NATIVE_ATTN: the attention binder drops its `_concat_heads_decode`
    binding (the model's own method, native at the block's rows once the K64 graft's
    kernel is mounted, runs unbound), independent of native_m3's `_wo_proj` retirement.
    `prep_guard`/`concat_guard` (TwoTileNativeAttnGuard) expose the per-layer legacy-path
    counters for model_batch.run()'s generic per-binder loop: expected_calls 0 under
    native_attn, loud on a leak; unchanged (one per forward) otherwise."""

    def test_concat_binding_drops_wo_binding_stays_prep_and_concat_guards_expect_zero(self):
        ttnn = FakeTTNN(prep_limit=None)
        model = fake_model(ttnn, layers=6, full=(3,), concat_limit=None)
        binding = bind_two_tile_attention(model, 64, ttnn, native_attn=True)
        self.assertTrue(binding.native_attn)
        full = [layer.attention for layer in model.layers if layer.is_full_attention]
        expected = []
        for attention in full:
            expected += [(attention, 'forward_decode'), (attention, '_wo_proj')]
        self.assertEqual([(instance, name) for instance, name, value in binding.bindings[1:]], expected,
                         'no _concat_heads_decode binding under native_attn: the model source runs it natively')
        self.assertEqual((binding.prep_guard.expected_calls, binding.concat_guard.expected_calls), (0, 0))
        self.assertEqual((binding.prep_guard.calls, binding.concat_guard.calls), (0, 0))
        self.assertEqual((binding.prep_guard.label, binding.concat_guard.label),
                         ('sliced attn_decode_prep', 'two-tile head concat'))
        self.assertEqual(binding.prep_guard.bindings, ())
        self.assertEqual(binding.concat_guard.bindings, ())

        # driven for real: one native prep call and one native concat call per layer,
        # never through the retired wrappers
        with instance_overrides(binding.bindings):
            for attention in full:
                x, positions, cos, sin = block_inputs(ttnn)
                attention.forward_decode(x, positions, cos, sin, page_table='pages')
        self.assertEqual(binding.calls, 1)
        self.assertEqual((binding.prep_guard.calls, binding.concat_guard.calls), (0, 0),
                         'no leak: the legacy paths never ran')
        self.assertTrue(all(attention.native_concats == [('gated', 64)] for attention in full))

    def test_a_leaked_call_through_either_retired_wrapper_shows_up_on_its_guard(self):
        ttnn = FakeTTNN(prep_limit=None)
        model = fake_model(ttnn, layers=4, full=(1, 3), concat_limit=None)
        binding = bind_two_tile_attention(model, 64, ttnn, native_attn=True)
        # simulate a future bug that still reaches the retired per-layer wrapper objects
        # (they are built as inert counters purely so a leak like this is visible)
        binding.forwards[0].prep_calls += 1
        self.assertEqual(binding.prep_guard.calls, 1)
        self.assertEqual(binding.concat_guard.calls, 0)
        gated = ttnn.tensor((1, 64, 12, 256), 'gated', 'l1')
        binding.forwards[1].concat(gated, 64)
        self.assertEqual(binding.concat_guard.calls, 1)

    def test_native_attn_false_is_unchanged(self):
        ttnn = FakeTTNN()
        model = fake_model(ttnn, layers=6, full=(3,))
        binding = bind_two_tile_attention(model, 64, ttnn)
        self.assertFalse(binding.native_attn)
        full = [layer.attention for layer in model.layers if layer.is_full_attention]
        expected = []
        for attention in full:
            expected += [(attention, 'forward_decode'), (attention, '_concat_heads_decode'), (attention, '_wo_proj')]
        self.assertEqual([(instance, name) for instance, name, value in binding.bindings[1:]], expected)
        self.assertEqual((binding.prep_guard.expected_calls, binding.concat_guard.expected_calls), (1, 1))
        with instance_overrides(binding.bindings):
            for attention in full:
                x, positions, cos, sin = block_inputs(ttnn)
                attention.forward_decode(x, positions, cos, sin, page_table='pages')
        self.assertEqual((binding.prep_guard.calls, binding.concat_guard.calls), (1, 1))

    def test_native_m3_and_native_attn_together_leave_only_forward_decode_bound(self):
        """Both grafts at once: `_wo_proj` is native (native_m3) and `_concat_heads_decode`
        is native (native_attn), so the attention binder's only per-layer binding left is
        `forward_decode` itself - it still has to run the prep op and pick the right
        progcfg, so it cannot retire the way MLP/GDN-output's bindings do."""
        ttnn = FakeTTNN(prep_limit=None)
        model = fake_model(ttnn, layers=4, full=(1, 3), native_m3=True, concat_limit=None)
        binding = bind_two_tile_attention(model, 64, ttnn, native_m3=True, native_attn=True)
        full = [layer.attention for layer in model.layers if layer.is_full_attention]
        self.assertEqual([(instance, name) for instance, name, value in binding.bindings[1:]],
                         [(attention, 'forward_decode') for attention in full])


class NativeM3MLPAndGDNOutputBindingTests(unittest.TestCase):
    """Lever N M3native retires the MLP and GDN-output two-tile wrappers entirely: their
    bindings are empty (nothing overrides the now-native feed_forward.forward / _row_proj)
    and expected_calls is 0, while the per-layer wrapper objects are still built as inert
    call counters for model_batch.run()'s tightened zero-call assertion."""

    def test_mlp_binding_drops_its_bindings_and_expects_zero_calls(self):
        ttnn = FakeTTNN()
        model = fake_model(ttnn, layers=6, full=(3,), native_m3=True)
        binding = bind_two_tile_mlp(model, 64, ttnn, native_m3=True)
        self.assertEqual(binding.bindings, ())
        self.assertEqual(binding.expected_calls, 0)
        self.assertEqual(len(binding.forwards), 6, 'still built, as an inert call counter')
        self.assertEqual(binding.calls, 0)
        natives = [layer.feed_forward.forward for layer in model.layers]
        with instance_overrides(binding.bindings):
            pass
        self.assertEqual([layer.feed_forward.forward for layer in model.layers], natives,
                         'nothing was ever overridden - there was nothing to restore either')

    def test_gdn_output_binding_drops_its_bindings_and_expects_zero_calls(self):
        ttnn = FakeTTNN()
        model = fake_model(ttnn, layers=6, full=(3,), native_m3=True)
        binding = bind_two_tile_gdn_output(model, 64, ttnn, native_m3=True)
        self.assertEqual(binding.bindings, ())
        self.assertEqual(binding.expected_calls, 0)
        self.assertEqual(len(binding.projections), 5)
        self.assertEqual(binding.calls, 0)

    def test_without_native_m3_both_bindings_are_unchanged(self):
        ttnn = FakeTTNN()
        model = fake_model(ttnn, layers=6, full=(3,), native_m3=True)
        mlp = bind_two_tile_mlp(model, 64, ttnn)
        gdn_output = bind_two_tile_gdn_output(model, 64, ttnn)
        self.assertEqual(mlp.expected_calls, 6)
        self.assertEqual(gdn_output.expected_calls, 5)
        self.assertTrue(mlp.bindings and gdn_output.bindings)


class MLPForwardTests(unittest.TestCase):
    """The forward at 64 rows: two 32-row calls to `_forward_tp`, concatenated on the row
    axis, instead of one 64-row call with the fusion switch off."""

    def test_two_32_row_calls_to_forward_tp_concatenated_on_the_row_axis(self):
        ttnn = FakeTTNN()
        mlp = fake_mlp(ttnn)
        forward = TwoTileMLPForward(mlp, 64, ttnn)
        x = ttnn.tensor((1, 1, 64, 5120), 'x', 'l1')
        joined = forward(x)
        self.assertEqual(mlp.seen, [('x[0:32]', (1, 1, 32, 5120), 'l1', True), ('x[32:64]', (1, 1, 32, 5120), 'l1', True)])
        self.assertTrue(mlp._fuse_gateup_agmm is True, '32 rows never reaches the prefill arm; no override is needed')
        self.assertEqual((joined.shape, joined.memory_config()), ((1, 1, 64, 5120), 'l1'))
        self.assertEqual([entry for entry in ttnn.calls if entry[0] == 'slice'],
                         [('slice', 'x', 2, 0, 32, 'l1'), ('slice', 'x', 2, 32, 64, 'l1')])
        self.assertEqual([entry for entry in ttnn.calls if entry[0] == 'concat'],
                         [('concat', ['mlp(x[0:32])', 'mlp(x[32:64])'], 2, 'l1')])
        # freed: the whole activation (never handed to _forward_tp whole any more), both
        # slices and both per-tile outputs; the joined result is the caller's
        freed = ttnn.freed()
        for name in ('x', 'x[0:32]', 'x[32:64]', 'mlp(x[0:32])', 'mlp(x[32:64])'):
            self.assertIn(name, freed)
        self.assertNotIn('cat(mlp(x[0:32]),mlp(x[32:64]))', freed)
        self.assertEqual(forward.calls, 1)
        mlp.forward.assert_not_called()

    def test_the_wrong_rows_are_refused_before_any_op(self):
        ttnn = FakeTTNN()
        mlp = fake_mlp(ttnn)
        forward = TwoTileMLPForward(mlp, 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'bound for 64 rows'):
            forward(ttnn.tensor((1, 1, 32, 5120), 'x', 'l1'))
        self.assertEqual((mlp.seen, forward.calls, ttnn.calls, ttnn.deallocated), ([], 0, [], []))

    def test_a_failed_second_call_frees_the_whole_activation_and_the_first_output(self):
        ttnn = FakeTTNN()
        mlp = fake_mlp(ttnn)
        forward = TwoTileMLPForward(mlp, 64, ttnn)
        x = ttnn.tensor((1, 1, 64, 5120), 'x', 'l1')
        native, seen = mlp._forward_tp, []

        def failing(value):
            seen.append(value.name)
            if len(seen) == 2:
                raise RuntimeError('second tile refused')
            return native(value)

        mlp._forward_tp = failing
        with self.assertRaisesRegex(RuntimeError, 'second tile refused'):
            forward(x)
        freed = ttnn.freed()
        # the whole activation, the first half (freed by its own successful _forward_tp
        # call) and its output; the second half was handed to the failing call, whose own
        # state is not ours to know, so - like TwoTileConcatHeads - it is left alone
        for name in ('x', 'x[0:32]', 'mlp(x[0:32])'):
            self.assertIn(name, freed)
        self.assertNotIn('x[32:64]', freed, 'never freed: the failing call may already own it')
        self.assertFalse(any(entry[0] == 'concat' for entry in ttnn.calls))
        self.assertEqual(forward.calls, 0)

    def test_a_failed_join_frees_both_per_tile_outputs(self):
        ttnn = FakeTTNN()
        mlp = fake_mlp(ttnn)
        forward = TwoTileMLPForward(mlp, 64, ttnn)
        x = ttnn.tensor((1, 1, 64, 5120), 'x', 'l1')

        def concat(parts, dim, memory_config=None):
            raise RuntimeError('join refused')

        ttnn.concat = concat
        with self.assertRaisesRegex(RuntimeError, 'join refused'):
            forward(x)
        freed = ttnn.freed()
        for name in ('x', 'x[0:32]', 'x[32:64]', 'mlp(x[0:32])', 'mlp(x[32:64])'):
            self.assertIn(name, freed)
        self.assertEqual(forward.calls, 0)

    def test_construction_needs_the_tensor_parallel_mlp_with_its_fusion_switch(self):
        ttnn = FakeTTNN()
        with self.assertRaisesRegex(ValueError, 'tensor-parallel MLP'):
            TwoTileMLPForward(fake_mlp(ttnn, devices=1), 64, ttnn)
        without = fake_mlp(ttnn)
        del without._forward_tp
        with self.assertRaisesRegex(ValueError, 'tensor-parallel MLP'):
            TwoTileMLPForward(without, 64, ttnn)

        class ClassFlag:
            num_devices = 2
            _fuse_gateup_agmm = True

            def _forward_tp(self, x):
                return x

        with self.assertRaisesRegex(ValueError, 'on the instance'):
            TwoTileMLPForward(ClassFlag(), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            TwoTileMLPForward(fake_mlp(ttnn), 32, ttnn)


class MLPBindingTests(unittest.TestCase):
    def test_one_forward_per_layer_bound_on_feed_forward(self):
        ttnn = FakeTTNN()
        model = fake_model(ttnn, layers=6, full=(3,))
        binding = bind_two_tile_mlp(model, 64, ttnn)
        self.assertIsInstance(binding, TwoTileMLPBinding)
        self.assertEqual((binding.label, binding.rows, binding.expected_calls, binding.calls), ('MLP forward', 64, 6, 0))
        self.assertEqual([(instance, name) for instance, name, value in binding.bindings],
                         [(layer.feed_forward, 'forward') for layer in model.layers])
        natives = [layer.feed_forward.forward for layer in model.layers]
        with instance_overrides(binding.bindings):
            for index, layer in enumerate(model.layers):
                joined = layer.feed_forward.forward(ttnn.tensor((1, 1, 64, 5120), 'x%d' % index, 'l1'))
                self.assertEqual((joined.shape, joined.memory_config()), ((1, 1, 64, 5120), 'l1'))
            self.assertEqual(binding.calls, 6)
        self.assertEqual([layer.feed_forward.forward for layer in model.layers], natives)
        self.assertTrue(all(layer.feed_forward._fuse_gateup_agmm is True for layer in model.layers))
        with self.assertRaisesRegex(ValueError, 'every layer carries a feed_forward'):
            bind_two_tile_mlp(SimpleNamespace(layers=[SimpleNamespace(feed_forward=None)]), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'every layer carries a feed_forward'):
            bind_two_tile_mlp(SimpleNamespace(layers=[]), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            bind_two_tile_mlp(model, 32, ttnn)


class GDNOutputBindingTests(unittest.TestCase):
    def test_one_projection_per_gdn_layer_bound_on_the_gdn_instance(self):
        ttnn = FakeTTNN()
        model = fake_model(ttnn, layers=6, full=(3,))
        binding = bind_two_tile_gdn_output(model, 64, ttnn)
        self.assertIsInstance(binding, TwoTileGDNOutputBinding)
        self.assertEqual((binding.label, binding.rows, binding.expected_calls, binding.calls),
                         ('GDN output projection', 64, 5, 0))
        gdn_layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
        self.assertEqual([(instance, name) for instance, name, value in binding.bindings],
                         [(gdn, '_row_proj') for gdn in gdn_layers])
        self.assertTrue(all(isinstance(value, TwoTileProjectionSplit) for instance, name, value in binding.bindings))
        natives = [gdn._row_proj for gdn in gdn_layers]
        with instance_overrides(binding.bindings):
            for index, gdn in enumerate(gdn_layers):
                output = ttnn.tensor((1, 64, 3072), 'gdn-out%d' % index, 'l1')
                joined = gdn._row_proj(output, 'out-weight')
                self.assertEqual((joined.shape, joined.memory_config()), ((1, 64, 5120), 'l1'))
            self.assertEqual(binding.calls, 5)
        self.assertEqual([gdn._row_proj for gdn in gdn_layers], natives)
        self.assertTrue(all(len(gdn.row_proj_calls) == 2 for gdn in gdn_layers))
        with self.assertRaisesRegex(ValueError, 'at least one GDN layer'):
            bind_two_tile_gdn_output(fake_model(ttnn, layers=4, full=(0, 1, 2, 3)), 64, ttnn)
        with self.assertRaisesRegex(ValueError, 'beyond one tile'):
            bind_two_tile_gdn_output(model, 32, ttnn)


DECODE = SimpleNamespace(name='DECODE')


def decode_mode_module():
    """The model's Mode enum module, as two_tile_norm imports it at bind."""
    return patch.dict(sys.modules, {'models.tt_transformers.tt.common': SimpleNamespace(Mode=SimpleNamespace(DECODE=DECODE))})


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

    def test_beyond_one_tile_the_four_binders_are_built_in_order(self):
        from model_batch import two_tile_bindings
        from two_tile_norm import TwoTileNormBinding

        ttnn = FakeTTNN()
        model = fake_model(ttnn)
        with decode_mode_module():
            norm, attention, mlp, gdn_output = two_tile_bindings(64, model, ttnn)
        self.assertIsInstance(norm, TwoTileNormBinding)
        self.assertIsInstance(attention, TwoTileAttentionBinding)
        self.assertIsInstance(mlp, TwoTileMLPBinding)
        self.assertIsInstance(gdn_output, TwoTileGDNOutputBinding)
        self.assertEqual([binder.expected_calls for binder in (norm, attention, mlp, gdn_output)], [129, 16, 64, 48])
        self.assertEqual([binder.label for binder in (norm, attention, mlp, gdn_output)],
                         ['decode norm', 'full-attention forward', 'MLP forward', 'GDN output projection'])

    def test_native_m3_still_builds_all_four_binders_but_retires_mlp_and_gdn_output(self):
        """Lever N M3native: detected by hasattr on the model args (the graft's own
        attn_wo_decode_1d_progcfg_64 attribute, mounted alongside the patched sources -
        never an env var). All four binders are still built (norm and attention are
        needed regardless), but MLP and GDN-output retire their wrapping entirely and
        the [PINDIAG] native_m3 marker fires once."""
        from model_batch import two_tile_bindings

        ttnn = FakeTTNN()
        model = fake_model(ttnn, native_m3=True)
        with decode_mode_module(), patch('dflash_device.pindiag') as marker:
            norm, attention, mlp, gdn_output = two_tile_bindings(64, model, ttnn)
        marker.assert_called_once()
        self.assertIn('native_m3 engaged', marker.call_args.args[0])
        self.assertEqual([binder.expected_calls for binder in (norm, attention, mlp, gdn_output)], [129, 16, 0, 0])
        self.assertEqual((mlp.bindings, gdn_output.bindings), ((), ()))
        self.assertTrue(norm.bindings and attention.bindings)
        self.assertIs(attention.progcfg, model.args.attn_qkv_decode_1d_progcfg_64)

    def test_without_the_64_attrs_native_m3_is_not_inferred(self):
        from model_batch import two_tile_bindings

        ttnn = FakeTTNN()
        model = fake_model(ttnn)
        with decode_mode_module(), patch('dflash_device.pindiag') as marker:
            norm, attention, mlp, gdn_output = two_tile_bindings(64, model, ttnn)
        marker.assert_not_called()
        self.assertEqual([binder.expected_calls for binder in (norm, attention, mlp, gdn_output)], [129, 16, 64, 48])

    def test_native_attn_appends_its_two_guards_and_fires_its_own_marker(self):
        """QWEN_FAST_NATIVE_ATTN (a RUNTIME/ENV fact, unlike native_m3's hasattr check):
        the tuple grows from four to six, the two extras being the attention binding's
        own prep_guard/concat_guard, appended (never inserted) so every existing
        positional unpack of the first four stays valid."""
        from model_batch import two_tile_bindings

        ttnn = FakeTTNN(prep_limit=None)
        model = fake_model(ttnn, concat_limit=None)
        with decode_mode_module(), patch.dict(os.environ, {NATIVE_ATTN_ENV: '1'}), \
             patch('dflash_device.pindiag') as marker:
            two_tile = two_tile_bindings(64, model, ttnn)
        self.assertEqual(len(two_tile), 6)
        norm, attention, mlp, gdn_output, prep_guard, concat_guard = two_tile
        self.assertIsInstance(attention, TwoTileAttentionBinding)
        self.assertIs(prep_guard, attention.prep_guard)
        self.assertIs(concat_guard, attention.concat_guard)
        self.assertEqual([binder.expected_calls for binder in two_tile], [129, 16, 64, 48, 0, 0])
        self.assertEqual(marker.call_count, 1, 'one native_attn marker line, no native_m3 line (not engaged)')
        self.assertIn('native_attn engaged', marker.call_args.args[0])

    def test_native_attn_off_keeps_the_four_tuple_shape(self):
        from model_batch import two_tile_bindings

        ttnn = FakeTTNN()
        model = fake_model(ttnn)
        with decode_mode_module(), patch.dict(os.environ):
            os.environ.pop(NATIVE_ATTN_ENV, None)
            two_tile = two_tile_bindings(64, model, ttnn)
        self.assertEqual(len(two_tile), 4)

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
        with decode_mode_module():
            two_tile = two_tile_bindings(64, model, ttnn)
        norm, attention, mlp, gdn_output = two_tile
        self.assertEqual([binder.expected_calls for binder in two_tile], [7, 1, 3, 2])
        fixture = self.fixture(two_tile)
        decode = DECODE
        seen = dict(block_h=[], fusion=[])

        def attend(layer):
            x, positions, cos, sin = block_inputs(ttnn)
            return layer.attention.forward_decode(x, positions, cos, sin, page_table='pages')

        def project_gdn_output(layer, index):
            output = ttnn.tensor((1, 64, 3072), 'gdn-out%d' % index, 'l1')
            return layer.attention._row_proj(output, 'out-weight')

        def mlp_input(index):
            return ttnn.tensor((1, 1, 64, 5120), 'mlp-in%d' % index, 'l1')

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            for index, layer in enumerate(model.layers):
                for unused in range(2):
                    seen['block_h'].append(model.args.get_norm_config('attn', decode)['sharded_program_config'].block_h)
                if layer.is_full_attention:
                    attend(layer)
                else:
                    project_gdn_output(layer, index)
                layer.feed_forward.forward(mlp_input(index))
                seen['fusion'].append(layer.feed_forward._fuse_gateup_agmm)
            seen['block_h'].append(model.args.get_norm_config('lm_head', decode)['sharded_program_config'].block_h)
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        self.assertEqual(fixture.run(), 'logits')
        self.assertEqual(seen, dict(block_h=[2] * 7, fusion=[True] * 3))
        self.assertEqual([binder.calls for binder in two_tile], [7, 1, 3, 2])
        # the attention tail took the two-half head concat and output projection under the binding
        self.assertEqual(model.layers[1].attention.native_concats, [('gated[0:32]', 32), ('gated[32:64]', 32)])
        self.assertEqual(len(model.layers[1].attention.wo_calls), 2)
        # the two GDN layers each took their output projection as two 32-row halves
        self.assertEqual([len(layer.attention.row_proj_calls) for layer in model.layers if not layer.is_full_attention],
                         [2, 2])
        # everything is off again outside the forward
        self.assertEqual(model.args.get_norm_config('attn', decode)['sharded_program_config'].block_h, 1)
        self.assertEqual(model.args.attn_qkv_decode_1d_progcfg.per_core_M, 1)
        self.assertTrue(all(layer.feed_forward._fuse_gateup_agmm is True for layer in model.layers))
        self.assertFalse('_concat_heads_decode' in model.layers[1].attention.__dict__)
        self.assertFalse('_wo_proj' in model.layers[1].attention.__dict__)
        self.assertFalse(any('_row_proj' in layer.attention.__dict__ for layer in model.layers if not layer.is_full_attention))
        # a forward that skips one MLP is refused by count, naming the binder (mlp comes before
        # the GDN output binder, so its count is checked first)
        short = Mock(side_effect=lambda *a, **k: (setattr(fixture, 'gdn_calls', fixture.gdn_calls + 48),
                                                   [model.args.get_norm_config('attn', decode) for unused in range(7)],
                                                   attend(model.layers[1]),
                                                   [layer.feed_forward.forward(mlp_input(index))
                                                    for index, layer in enumerate(model.layers[:2])]))
        fixture.model = SimpleNamespace(_forward_decode=short)
        with self.assertRaisesRegex(AssertionError, 'Every MLP forward of the wide block must take its two-tile form: 2 engaged, 3 expected'):
            fixture.run()
        # and one that skips the attention forward
        short = Mock(side_effect=lambda *a, **k: (setattr(fixture, 'gdn_calls', fixture.gdn_calls + 48),
                                                   [model.args.get_norm_config('attn', decode) for unused in range(7)],
                                                   [layer.feed_forward.forward(mlp_input(index))
                                                    for index, layer in enumerate(model.layers)]))
        fixture.model = SimpleNamespace(_forward_decode=short)
        with self.assertRaisesRegex(AssertionError, 'Every full-attention forward of the wide block must take its two-tile form: 0 engaged, 1 expected'):
            fixture.run()
        # and one that skips a GDN output projection (checked last, after norm/attention/mlp)
        short = Mock(side_effect=lambda *a, **k: (setattr(fixture, 'gdn_calls', fixture.gdn_calls + 48),
                                                   [model.args.get_norm_config('attn', decode) for unused in range(7)],
                                                   attend(model.layers[1]),
                                                   project_gdn_output(model.layers[0], 0),
                                                   [layer.feed_forward.forward(mlp_input(index))
                                                    for index, layer in enumerate(model.layers)]))
        fixture.model = SimpleNamespace(_forward_decode=short)
        with self.assertRaisesRegex(AssertionError, 'Every GDN output projection of the wide block must take its two-tile form: 1 engaged, 2 expected'):
            fixture.run()
        # an unbound (one-tile) fixture asks nothing of any of them
        narrow = self.fixture(())
        narrow.model = SimpleNamespace(_forward_decode=Mock(side_effect=lambda *a, **k: (setattr(narrow, 'gdn_calls', 48), 'logits')[1]))
        self.assertEqual(narrow.run(), 'logits')


if __name__ == '__main__':
    unittest.main()
