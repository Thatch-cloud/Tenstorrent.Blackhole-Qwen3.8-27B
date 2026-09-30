"""The drafter at four cards (S2T-07, host side): its shards, heads and shapes at 8 query / 2 KV heads per chip.

The drafter is TP-sharded (dflash_device.py: heads, MLP columns, embedding width and vocabulary are split over the chips).
The pair's modules carry 16 / 4 heads, 2,560-column taps and two chips; the four-card widths come from tp_shapes. Held here:

  - feature_projection_tp: the pinned tap packing at the pair, four 1,280-column shards at four cards;
  - draft_attention_tp: validate_attention / draft_sdpa equal to the frozen draft_attention's at the pair, the (1, 8, 32, 128)
    query on (1, 2, L, 128) keys at four cards, the pinned composed path still refusing there, and the seam rebinding the
    unedited importers;
  - draft_head_layout, draft_kv_projection, draft_attention_branch weights, draft_mlp: shapes and shard counts;
  - pair_row_exact at 8 / 2 heads: the folded pair's rows are bit-identical to each user's own SDPA (the property the pair's
    test holds at 16 / 4), on the CPU stand-in for the ttnn surface test_pair_row_exact builds;
  - draft_kv_history_tp: the sibling class over 2 KV heads, and the pair's class untouched;
  - serving_buffer_pool's draft shapes, dflash_device's tap and shard checks.

Nothing here runs on a card: the drafter's SDPA at 8 / 2 heads, the matmul partitions and the collectives are what HW-B shows."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import draft_attention
import draft_attention_tp
import draft_head_layout
import draft_kv_history
import draft_kv_history_tp
import draft_mlp
import feature_projection
import feature_projection_tp
import pair_row_exact
import pair_row_exact_tp
import serving_buffer_pool
import tp_addresses
import tp_shapes
from tp_test_support import four_cards
from test_pair_row_exact import Device, TorchOps, keep

HERE = Path(__file__).resolve().parent
# the pinned function objects, before any test installs the seam that rebinds the module's name to the twin
PINNED_VALIDATE_ATTENTION = draft_attention.validate_attention


def four():
    return four_cards()


def pair():
    return patch.dict(os.environ, {}, clear=True)


class Tensor:
    def __init__(self, shape, dtype='bf16', layout='tile', memory='dram'):
        self.shape, self.dtype, self.layout, self._memory = tuple(shape), dtype, layout, memory

    def memory_config(self):
        return self._memory


ATTENTION_OPS = SimpleNamespace(bfloat16='bf16')


class FeatureProjectionTests(unittest.TestCase):
    def weight(self):
        generator = torch.Generator().manual_seed(1)
        return torch.randn(64, 5 * 5120, generator=generator).bfloat16()

    def test_at_the_pair_the_twin_is_the_pinned_packing(self):
        weight = self.weight()
        with pair():
            ours, theirs = feature_projection_tp.projection_shards(weight), feature_projection.projection_shards(weight)
        self.assertEqual(len(ours), 2)
        for left, right in zip(ours, theirs):
            self.assertTrue(torch.equal(left, right))

    def test_four_cards_shard_every_tap_into_four_1280_column_slices(self):
        weight = self.weight()
        with four():
            shards = feature_projection_tp.projection_shards(weight)
        self.assertEqual(len(shards), 4)
        grouped = weight.reshape(64, 5, 5120)
        for chip, shard in enumerate(shards):
            self.assertEqual(tuple(shard.shape), (5 * 1280, 64))
            expected = grouped[:, :, chip * 1280:(chip + 1) * 1280].reshape(64, 5 * 1280).T
            self.assertTrue(torch.equal(shard, expected))
        # the four shards together are the whole weight
        rebuilt = torch.stack([shard.T.reshape(64, 5, 1280) for shard in shards], dim=2).reshape(64, 5, 5120)
        self.assertTrue(torch.equal(rebuilt, grouped))

    def test_the_local_features_are_hidden_over_tp_wide(self):
        operations = SimpleNamespace(concat=lambda values, dim, memory_config: ('joined', len(values), dim),
                                     DRAM_MEMORY_CONFIG='dram')
        for env, width in ((pair, 2560), (four, 1280)):
            with env():
                taps = [Tensor((1, 1, 32, width)) for _ in range(5)]
                self.assertEqual(feature_projection_tp.concatenate_local_features(operations, taps), ('joined', 5, -1))
                with self.assertRaises(ValueError):
                    feature_projection_tp.concatenate_local_features(operations, [Tensor((1, 1, 32, 5120))] * 5)
        with pair():
            with self.assertRaisesRegex(ValueError, 'Matching chip-local \\[1,1,T,hidden/2\\] features required'):
                feature_projection_tp.concatenate_local_features(operations, [Tensor((1, 1, 32, 1280))] * 5)


class AttentionTests(unittest.TestCase):
    def operands(self, heads, kv, rows=64, bf16='bf16'):
        return (Tensor((1, heads, 32, 128), bf16), Tensor((1, kv, rows, 128), bf16), Tensor((1, kv, rows, 128), bf16),
                Tensor((1, 1, 32, rows), bf16))

    def outcome(self, function, *arguments):
        try:
            return ('value', function(*arguments))
        except ValueError as error:
            return ('error', str(error))

    def test_at_the_pair_validation_is_the_pinned_functions(self):
        cases = [(16, 4), (8, 2), (16, 2), (12, 4)]
        with pair():
            for heads, kv in cases:
                operands = self.operands(heads, kv)
                self.assertEqual(self.outcome(draft_attention_tp.validate_attention, ATTENTION_OPS, *operands),
                                 self.outcome(draft_attention.validate_attention, ATTENTION_OPS, *operands), (heads, kv))
            bad_dtype = self.operands(16, 4, bf16='fp32')
            self.assertEqual(self.outcome(draft_attention_tp.validate_attention, ATTENTION_OPS, *bad_dtype),
                             self.outcome(draft_attention.validate_attention, ATTENTION_OPS, *bad_dtype))
            unaligned = list(self.operands(16, 4, rows=48))
            unaligned[1] = Tensor((1, 4, 48, 128))
            self.assertEqual(self.outcome(draft_attention_tp.validate_attention, ATTENTION_OPS, *unaligned),
                             self.outcome(draft_attention.validate_attention, ATTENTION_OPS, *unaligned))

    def test_four_cards_take_eight_query_and_two_kv_heads_and_refuse_the_pairs(self):
        with four():
            draft_attention_tp.validate_attention(ATTENTION_OPS, *self.operands(8, 2))
            for heads, kv in ((16, 4), (8, 4), (16, 2)):
                with self.assertRaisesRegex(ValueError, 'TP4 draft GQA uses 8 query and two KV heads'):
                    draft_attention_tp.validate_attention(ATTENTION_OPS, *self.operands(heads, kv))
            # the frozen module's own check (the composed path's) still refuses the four-card heads
            with self.assertRaises(ValueError):
                PINNED_VALIDATE_ATTENTION(ATTENTION_OPS, *self.operands(8, 2))

    def test_draft_sdpa_is_the_same_call_with_the_width_s_heads(self):
        calls = []

        class Operations:
            bfloat16 = 'bf16'
            MathFidelity = SimpleNamespace(HiFi4='hifi4')
            DRAM_MEMORY_CONFIG = 'dram'

            def WormholeComputeKernelConfig(self, **options):
                return ('kernel', tuple(sorted(options.items())))

            def SDPAProgramConfig(self, **options):
                return ('program', tuple(sorted(options.items())))

            transformer = SimpleNamespace(scaled_dot_product_attention=lambda *args, **kwargs: calls.append(
                (tuple(getattr(a, 'shape', a) for a in args),
                 tuple(sorted((key, getattr(value, 'shape', value)) for key, value in kwargs.items())))) or 'out')

        operations = Operations()
        with pair():
            draft_attention_tp.draft_sdpa(operations, *self.operands(16, 4))
            draft_attention.draft_sdpa(operations, *self.operands(16, 4))
        self.assertEqual(calls[0], calls[1])
        with four():
            self.assertEqual(draft_attention_tp.draft_sdpa(operations, *self.operands(8, 2), key_chunk_size=64), 'out')
        self.assertEqual(dict(calls[2][1])['scale'], 128 ** -0.5)

    def test_the_seam_rebinds_the_unedited_importers(self):
        import dflash_t16_native_attention
        import proposal_native_attention
        before = (dflash_t16_native_attention.draft_sdpa, proposal_native_attention.validate_attention,
                  draft_attention.draft_sdpa)
        self.addCleanup(tp_addresses.uninstall)
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            tp_addresses.install()
        self.assertIs(dflash_t16_native_attention.draft_sdpa, draft_attention_tp.draft_sdpa)
        self.assertIs(proposal_native_attention.validate_attention, draft_attention_tp.validate_attention)
        self.assertIs(draft_attention.draft_sdpa, draft_attention_tp.draft_sdpa)
        tp_addresses.uninstall()
        self.assertEqual((dflash_t16_native_attention.draft_sdpa, proposal_native_attention.validate_attention,
                          draft_attention.draft_sdpa), before)


class HeadLayoutTests(unittest.TestCase):
    def test_split_projected_heads_calls_the_op_with_the_width_s_head_counts(self):
        seen = {}

        class Operations:
            bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'

            def pad(self, value, padding, fill):
                return Tensor((1, 1, value.shape[2] + padding[2][1], value.shape[3]))

            def concat(self, values, dim, memory_config):
                shape = list(values[0].shape)
                shape[dim] = sum(value.shape[dim] for value in values)
                return Tensor(shape)

            def slice(self, value, start, end):
                return Tensor(tuple(b - a for a, b in zip(start, end)))

        operations = Operations()
        operations.experimental = SimpleNamespace(nlp_create_qkv_heads=lambda q, kv, **options: (
            seen.update(options) or (Tensor((1, options['num_heads'], 64, 128)),
                                     Tensor((1, options['num_kv_heads'], 64, 128)),
                                     Tensor((1, options['num_kv_heads'], 64, 128)))))
        for env, (query_width, key_width, heads, kv) in ((pair, (2048, 512, 16, 4)), (four, (1024, 256, 8, 2))):
            with env():
                found = draft_head_layout.split_projected_heads(
                    operations, Tensor((1, 1, 32, query_width)), Tensor((1, 1, 64, key_width)),
                    Tensor((1, 1, 64, key_width)), lambda value: value)
                self.assertEqual((seen['num_heads'], seen['num_kv_heads']), (heads, kv))
                self.assertEqual(tuple(found['q'].shape), (1, heads, 32, 128))
                with self.assertRaisesRegex(ValueError, 'for %d query and %s KV heads' % (heads, {4: 'four', 2: 'two'}[kv])):
                    draft_head_layout.split_projected_heads(operations, Tensor((1, 1, 32, 3)), Tensor((1, 1, 64, key_width)),
                                                            Tensor((1, 1, 64, key_width)), lambda value: value)
        with four():
            operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
                                         experimental=SimpleNamespace(nlp_concat_heads=lambda value, memory_config: 'merged'))
            self.assertEqual(draft_head_layout.concatenate_query_heads(operations, Tensor((1, 8, 32, 128)), lambda v: v), 'merged')
            with self.assertRaises(ValueError):
                draft_head_layout.concatenate_query_heads(operations, Tensor((1, 16, 32, 128)), lambda v: v)


class MlpTests(unittest.TestCase):
    def test_the_mlp_is_split_over_the_chips(self):
        generator = torch.Generator().manual_seed(2)
        gate = torch.randn(128, 64, generator=generator).bfloat16()
        up = torch.randn(128, 64, generator=generator).bfloat16()
        down = torch.randn(64, 128, generator=generator).bfloat16()
        for env, chips in ((pair, 2), (four, 4)):
            with env():
                shards = draft_mlp.split_mlp_weights(gate, up, down)
                self.assertEqual(len(shards), chips)
                self.assertEqual(tuple(shards[0][0].shape), (64, 128 // chips))
                self.assertEqual(tuple(shards[0][2].shape), (128 // chips, 64))
                self.assertTrue(torch.equal(torch.cat([shard[0].T for shard in shards], dim=0), gate))
                self.assertTrue(torch.equal(torch.cat([shard[2].T for shard in shards], dim=1), down))
        with four():
            with self.assertRaises(ValueError):
                draft_mlp.split_mlp_weights(torch.zeros(6, 4).bfloat16(), torch.zeros(6, 4).bfloat16(),
                                            torch.zeros(4, 6).bfloat16())

    def test_the_gate_up_partition_is_ceil_tiles_over_80_cores(self):
        import draft_mlp_branch
        shard = lambda tiles: [[SimpleNamespace(shape=(5120, tiles * 32))] * 3]
        self.assertEqual(draft_mlp_branch.gate_up_columns_of(dict(shards=shard(272))), 4, 'the pair: 8,704 columns')
        self.assertEqual(draft_mlp_branch.gate_up_columns_of(dict(shards=shard(136))), 2, 'four cards: 4,352 columns')
        with pair():
            self.assertEqual(draft_mlp_branch.gate_up_columns_of({}), 4)
        with four():
            self.assertEqual(draft_mlp_branch.gate_up_columns_of({}), 2)


class BranchPreparationTests(unittest.TestCase):
    def test_the_attention_weights_are_sharded_over_the_chips_by_head(self):
        import draft_attention_branch as branch
        uploads = []

        class Operations:
            bfloat16, bfloat8_b = 'bf16', 'bf8'
            ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'row', 'tile', 'dram'
            MathFidelity = SimpleNamespace(HiFi4='hifi4')

            def from_torch(self, value, **options):
                uploads.append((tuple(value.shape), options['mesh_mapper']))
                return SimpleNamespace(shape=tuple(value.shape))

            def ShardTensorToMesh(self, mesh, dim):
                return ('shard', dim)

            def ReplicateTensorToMesh(self, mesh):
                return ('replicate',)

            def WormholeComputeKernelConfig(self, **options):
                return 'kernel'

        generator = torch.Generator().manual_seed(3)

        def weights():
            return {'layers.0.self_attn.q_proj.weight': torch.randn(4096, 5120, generator=generator).bfloat16(),
                    'layers.0.self_attn.k_proj.weight': torch.randn(1024, 5120, generator=generator).bfloat16(),
                    'layers.0.self_attn.v_proj.weight': torch.randn(1024, 5120, generator=generator).bfloat16(),
                    'layers.0.self_attn.o_proj.weight': torch.randn(5120, 4096, generator=generator).bfloat16(),
                    'layers.0.self_attn.q_norm.weight': torch.randn(128).bfloat16(),
                    'layers.0.self_attn.k_norm.weight': torch.randn(128).bfloat16(),
                    'layers.0.input_layernorm.weight': torch.randn(5120).bfloat16()}

        convolution = {'layers.0.input_layernorm.weight': torch.randn(5120).bfloat16(),
                       'layers.0.attention_conv.kernel_projection.weight': torch.randn(1280, 5120).bfloat16(),
                       'layers.0.attention_conv.base_kernel': torch.randn(2, 2, 5120).bfloat16()}
        for env, chips in ((pair, 2), (four, 4)):
            del uploads[:]
            with env():
                branch.prepare_attention_branch(Operations(), object(), weights(), convolution, lambda value: value,
                                                native_head_layout=True, block_rows=16)
            sharded = [shape for shape, mapper in uploads if mapper == ('shard', 0)]
            # q, k, v projections (input x output, stacked over the chips along dim 0) and the output projection
            self.assertEqual(sharded[0], (chips * 5120, 4096 // chips))
            self.assertEqual(sharded[1], (chips * 5120, 1024 // chips))
            self.assertEqual(sharded[3], (chips * (4096 // chips), 5120))


# ---- the folded pair SDPA at 8 / 2 heads -------------------------------------------------------------------------------

class TPPairFixture:
    """test_pair_row_exact.PairFixture at the width's heads: two users' cached history, live block and query, as the pair's
    branch assembles them and as each user's single-user trace does."""

    def __init__(self, heads, kv, seed=0):
        from dflash_batched_mask import key_value_plan

        generator = torch.Generator().manual_seed(seed)

        def normal(*shape):
            return torch.randn(*shape, generator=generator).bfloat16()

        self.heads, self.kv = heads, kv
        self.cache = {user: {name: normal(1, kv, 2048, 128) for name in 'kv'} for user in 'ab'}
        self.live = {user: {name: normal(1, kv, 16, 128) for name in 'kv'} for user in 'ab'}
        self.live_pad = {user: {name: normal(1, kv, 16, 128) for name in 'kv'} for user in 'ab'}
        self.query = {user: normal(1, heads, 16, 128) for user in 'ab'}
        self.query_pad = {user: normal(1, heads, 16, 128) for user in 'ab'}
        block = {name: torch.cat([self.live['a'][name], self.live['b'][name]], dim=2) for name in 'kv'}
        self.pair_query = torch.cat([self.query['a'], self.query['b']], dim=2)
        plan, spans, key_rows = key_value_plan([2048, 2048], 16)
        self.pair_keys = {}
        for name in 'kv':
            pieces = []
            for part in plan:
                if part['kind'] == 'cached':
                    pieces.append(self.cache['ab'[part['user']]][name])
                    continue
                start = part['source'].start if part['kind'] == 'live' else 0
                pieces.append(block[name][:, :, start:start + part['rows']])
            self.pair_keys[name] = torch.cat(pieces, dim=2)
        assert key_rows == 4160 and self.pair_keys['k'].shape == (1, kv, 4160, 128)

    def single(self, user):
        keys = {name: torch.cat([self.cache[user][name], self.live[user][name], self.live_pad[user][name]], dim=2)
                for name in 'kv'}
        return torch.cat([self.query[user], self.query_pad[user]], dim=2), keys['k'], keys['v']


def same_bits(left, right):
    return (tuple(left.shape) == tuple(right.shape) and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16)))


class PairRowExactAtFourCardsTests(unittest.TestCase):
    def test_the_pairs_constants_and_the_width_s_heads(self):
        self.assertEqual((pair_row_exact.QUERY_HEADS, pair_row_exact.KEY_HEADS), (16, 4))
        with pair():
            self.assertEqual(pair_row_exact_tp.heads(), (16, 4, 4, 32, 8))
        with four():
            self.assertEqual(pair_row_exact_tp.heads(), (8, 2, 4, 16, 4))

    def test_the_fold_index_map_at_eight_query_heads(self):
        with four():
            query = torch.arange(1 * 8 * 32 * 4, dtype=torch.float32).reshape(1, 8, 32, 4)
            # widen the last dim to 128 by repetition so the shape check passes
            query = query.repeat(1, 1, 1, 32).bfloat16() if False else torch.randn(1, 8, 32, 128).bfloat16()
            owned = []
            operations = TorchOps()
            folded = pair_row_exact.fold_query(operations, Device(query), keep(owned)).value
            self.assertEqual(tuple(folded.shape), (1, 16, 32, 128))
            shifted = torch.cat([query[:, :, 16:], query[:, :, :16]], dim=2)
            for h in range(2):
                for u in range(2):
                    for j in range(4):
                        expected = (query if u == 0 else shifted)[0, 4 * h + j]
                        self.assertTrue(torch.equal(folded[0, 8 * h + 4 * u + j], expected), (h, u, j))
            # and the unfold takes each user's rows back
            restored = pair_row_exact.unfold_output(operations, Device(folded), keep(owned)).value
            self.assertTrue(torch.equal(restored, query))

    def test_the_folded_rows_are_each_users_single_user_rows_bit_for_bit(self):
        with four():
            fixture = TPPairFixture(8, 2)
            mask = pair_row_exact.fold_mask([2048, 2048])
            mask_device = Device(mask)
            operations = TorchOps()
            owned = []
            folded = pair_row_exact.fold_attention(
                operations, Device(fixture.pair_query), Device(fixture.pair_keys['k']), Device(fixture.pair_keys['v']),
                mask_device, keep(owned), mask_validated=True).value
            self.assertEqual(tuple(folded.shape), (1, 8, 32, 128))
            self.assertTrue(any(call[0] == 'sdpa' and call[1] == (1, 16, 32, 128) and call[2] == (1, 4, 2080, 128)
                                for call in operations.calls), 'the SDPA runs at 16 query and 4 KV heads')
            for index, user in enumerate('ab'):
                query, keys, values = fixture.single(user)
                alone = operations.sdpa(Device(query), Device(keys), Device(values), attn_mask=mask_device,
                                        is_causal=False, scale=128 ** -0.5, program_config=None,
                                        compute_kernel_config=None, memory_config=None).value
                rows = slice(16 * index, 16 * index + 16)
                self.assertTrue(same_bits(folded[:, :, rows], alone[:, :, :16]), user)

    def test_the_operands_are_validated_against_the_width(self):
        operations = TorchOps()
        mask = Device(pair_row_exact.fold_mask([2048, 2048]))
        fixture = TPPairFixture(16, 4)
        with four():
            with self.assertRaisesRegex(ValueError, 'takes the packed \\(1, 8, 32, 128\\) query, the two-segment \\(1, 2, 4160'):
                pair_row_exact.validate_fold(operations, Device(fixture.pair_query), Device(fixture.pair_keys['k']),
                                             Device(fixture.pair_keys['v']), mask)


# ---- the sibling draft K/V history ---------------------------------------------------------------------------------------

class HistoryOps(TorchOps):
    """TorchOps plus what DraftKVHistory calls: pad, copy, zeros_like, upload, synchronize."""

    int32, uint32 = 'int32', 'uint32'
    ROW_MAJOR_LAYOUT = 'row'

    def __init__(self):
        TorchOps.__init__(self)
        self.copies = 0

    def pad(self, tensor, padding, value):
        pads = []
        for low, high in reversed(padding):
            pads += [low, high]
        return Device(torch.nn.functional.pad(tensor.value.float(), pads, value=value).to(tensor.value.dtype), tensor.dtype)

    def zeros_like(self, tensor):
        return Device(torch.zeros_like(tensor.value), tensor.dtype)

    def copy(self, source, destination):
        destination.value = source.value.clone()
        self.copies += 1

    def synchronize_device(self, mesh):
        pass

    def deallocate(self, tensor):
        pass

    def from_torch(self, value, **options):
        return Device(value.clone(), 'bf16')

    def ReplicateTensorToMesh(self, mesh):
        return ('replicate', mesh)

    def get_device_tensors(self, tensor):
        return [SimpleNamespace(buffer_address=lambda chip=chip: id(tensor) + chip)
                for chip in range(tp_shapes.chip_count())]

    def to_torch(self, shard):
        raise AssertionError('not read here')


class HistoryTests(unittest.TestCase):
    def build(self, cls, kv, query_width, layers=2):
        self.addCleanup(tp_addresses.uninstall)
        if tp_shapes.chip_count() != tp_shapes.PAIR:
            tp_addresses.install()   # draft_kv_history reads addresses through the pinned helper: the startup seam
        operations = HistoryOps()
        rows = 64

        def project(operations_, inputs, query, tables, retain, parameters):
            value = torch.arange(kv * 32 * 128, dtype=torch.float32).reshape(1, kv, 32, 128).bfloat16()
            padded = torch.cat([value] * (inputs.shape[2] // 32), dim=2)
            return dict(q=Device(torch.zeros(1, 1, 32, query_width)), k=Device(padded), v=Device(padded + 1))

        features = Device(torch.randn(1, 1, rows, 5120).bfloat16())
        module = sys.modules[cls.__module__]
        with patch.object(module, 'project_key_value', project):
            cache = cls(operations, 'mesh', [object()] * layers, features, position=rows, history_rows=rows)
            prepared = cache.prepare(features, 16, position=rows)
        return operations, cache, prepared

    def test_the_sibling_holds_two_kv_heads_and_the_pairs_class_four(self):
        with four():
            operations, cache, prepared = self.build(draft_kv_history_tp.DraftKVHistory, 2, 1024)
            self.assertEqual(len(cache.active), 2)
            self.assertEqual(tuple(cache.active[0]['k'].shape), (1, 2, 2048, 128))
            self.assertEqual(tuple(cache.spare[0]['k'].shape), (1, 2, 2048, 128))
            self.assertEqual((prepared.prefix, prepared.rows), (16, 80))
            self.assertEqual(tuple(cache.query.shape), (1, 1, 32, 1024))
            slices = [call for call in operations.calls if call[0] == 'slice' and len(call[2]) == 4 and call[3][1] in (2, 4)]
            self.assertTrue(slices and all(call[3][1] == 2 for call in slices), slices[:3])
        with pair():
            operations, cache, prepared = self.build(draft_kv_history.DraftKVHistory, 4, 2048)
            self.assertEqual(tuple(cache.active[0]['k'].shape), (1, 4, 2048, 128))

    def test_the_lent_banks_and_query_are_checked_at_the_width(self):
        operations = HistoryOps()
        with four():
            bank = lambda: {side: {name: Device(torch.zeros(1, 2, 2048, 128).bfloat16()) for name in 'kv'}
                            for side in ('active', 'spare')}
            self.assertEqual(len(draft_kv_history_tp.validate_storage(operations, [bank(), bank()], 2)), 2)
            wrong = {side: {name: Device(torch.zeros(1, 4, 2048, 128).bfloat16()) for name in 'kv'}
                     for side in ('active', 'spare')}
            with self.assertRaisesRegex(ValueError, '\\(1, 2, 2048, 128\\) K/V bank per learned layer'):
                draft_kv_history_tp.validate_storage(operations, [wrong, wrong], 2)
            query = Device(torch.zeros(1, 1, 32, 1024).bfloat16())
            self.assertIs(draft_kv_history_tp.validate_query(operations, query), query)
            with self.assertRaises(ValueError):
                draft_kv_history_tp.validate_query(operations, Device(torch.zeros(1, 1, 32, 2048).bfloat16()))

    def test_the_pairs_history_source_is_untouched(self):
        """draft_kv_slide_adapter text-patches DraftKVHistory.prepare's exact source: it keeps its (1, 4, ...) slices."""
        text = (HERE / 'draft_kv_history.py').read_text()
        self.assertIn('(1, 4, self.history_rows, 128)))\n                    accepted = retain(operations.slice(result[name], (0, 0, 0, 0), (1, 4, prefix, 128)))', text)
        self.assertNotIn('tp_shapes', text)


class PoolShapesTests(unittest.TestCase):
    def test_the_draft_banks_follow_the_width(self):
        with pair():
            self.assertEqual((serving_buffer_pool.kv_shape(), serving_buffer_pool.query_shape()),
                             (serving_buffer_pool.KV_SHAPE, serving_buffer_pool.QUERY_SHAPE))
            self.assertEqual(serving_buffer_pool.kv_shape(), (1, 4, 2048, 128))
        with four():
            self.assertEqual(serving_buffer_pool.kv_shape(), (1, 2, 2048, 128))
            self.assertEqual(serving_buffer_pool.query_shape(), (1, 1, 32, 1024))
            self.assertEqual(draft_kv_history_tp.kv_shape(), serving_buffer_pool.kv_shape())
            self.assertEqual(draft_kv_history_tp.query_shape(), serving_buffer_pool.query_shape())


class DeviceChecksTests(unittest.TestCase):
    def test_project_features_validates_the_taps_at_the_width(self):
        import dflash_device
        device = dflash_device.DFlashDevice.__new__(dflash_device.DFlashDevice)
        device.operations = SimpleNamespace(bfloat16='bf16')
        for env, good, bad in ((pair, 2560, 1280), (four, 1280, 2560)):
            with env():
                taps = tuple(Tensor((1, 1, 32, bad)) for _ in range(5))
                with self.assertRaisesRegex(ValueError, 'Five complete ordered BF16 local feature taps required'):
                    device.project_features(taps, 16)

    def test_the_device_takes_the_target_of_the_served_width(self):
        import dflash_device
        with four():
            bad = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True, mesh_device=None)
            with self.assertRaisesRegex(ValueError, 'Pinned TP4 target, all five DFlash2 layers and bounded prefill required'):
                dflash_device.DFlashDevice(SimpleNamespace(), bad, None, [], None, None, [], position=64)
        with pair():
            bad = SimpleNamespace(num_devices=4, vocab_size=248320, _lmhead_vocab_sharded=True, mesh_device=None)
            with self.assertRaisesRegex(ValueError, 'Pinned TP2 target'):
                dflash_device.DFlashDevice(SimpleNamespace(), bad, None, [], None, None, [], position=64)


if __name__ == '__main__':
    unittest.main()
