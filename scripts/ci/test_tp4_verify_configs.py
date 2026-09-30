"""The M = 64 verify matmul configs and verify-trace T1 #11 at four cards (S2T-08).

The model graft (docker/qwen-c2-graft/graft/model_config.py) builds seven 1D decode matmul configs at M = 64 for the
packed block and, under QWEN_FAST_VERIFY_T1=1, re-partitions two of them (attn_qkv, the MLP gate) through the same
builder, raising if anything but the N partition or the subblock would move. Every width in `_init_tp_config` is already
`// tp` (test_tp4_model_widths), so nothing in the graft changes for four cards; this holds that claim over the real
builder's arithmetic (transcribed in test_verify_trace_t1_graft from tp_common.py): the same seven configs build at
(1, 4), each K is whole tiles with an in0_block_w that divides it, every output tile of every config is computed by exactly
one core, and T1 #11 changes exactly the two configs it changes at the pair with the kept fields intact.

What only hardware can show - that the wider partition is exact at four cards, and how fast - is the T1 byte compare
(verify_t1_device_compare.py) and the S2 timing arm. The core counts (44 / 88 / 33 / 64) are the pair's, untuned at
four cards' N (S2T-O3).
"""

import sys
import types
import unittest
from unittest.mock import patch

import test_tp4_model_widths as widths
from test_verify_trace_t1_graft import create_matmul_1d_decode_progcfg, k_schedule, tile_owners

DECODE_NAMES = ('mlp_w1_decode_1d_progcfg_64', 'mlp_w3_decode_1d_progcfg_64', 'mlp_w2_decode_1d_progcfg_64',
                'attn_qkv_decode_1d_progcfg_64', 'gdn_qkvz_decode_1d_progcfg_64', 'attn_wo_decode_1d_progcfg_64',
                'gdn_out_decode_1d_progcfg_64')


def build(tp, batch, environ):
    """_init_tp_config over the real builder's arithmetic at `tp` devices: (args, log lines)."""
    graft, recorder, stubs = widths.load_graft()
    logged = []
    builder = types.SimpleNamespace(
        TILE_SIZE=32, create_matmul_1d_decode_progcfg=create_matmul_1d_decode_progcfg,
        create_dram_sharded_mem_config=lambda k, n: ('dram-memcfg', k, n),
        create_dram_sharded_matmul_program_config=lambda m, k, n, num_cores=None: ('dram-progcfg', m, k, n),
        prefill_grid_default=lambda: (8, 10), prefill_tuning=lambda tp_: 'tuning',
        create_prefill_matmul_program_config=lambda *args, **options: 'prefill',
        create_activation_shard_config=lambda k: ('activation', k))
    stubs['models.demos.blackhole.qwen36.tt'].tp_common = builder
    loguru = types.ModuleType('loguru')
    loguru.logger = types.SimpleNamespace(info=lambda *a: logged.append(('info',) + a),
                                          warning=lambda *a: logged.append(('warning',) + a))
    args = types.SimpleNamespace(
        n_heads=24, n_kv_heads=4, head_dim=256, dim=5120, hidden_dim=17408, max_batch_size=batch,
        linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, linear_q_dim=16 * 128, linear_k_dim=16 * 128, linear_v_dim=48 * 128,
        num_devices=tp)
    with patch.dict(sys.modules, dict(stubs, loguru=loguru)), patch.dict('os.environ', environ, clear=True):
        graft.Qwen36ModelArgs._init_tp_config(args, widths.Mesh(tp))
    return args, logged


def configs(args):
    return {name: getattr(args, name) for name in DECODE_NAMES}


class VerifyConfigsAtFourCardsTests(unittest.TestCase):
    def setUp(self):
        self.pair, _ = build(2, 4, {})
        self.quad, _ = build(4, 8, {})

    def test_the_same_seven_configs_build_at_both_widths(self):
        for args in (self.pair, self.quad):
            found = configs(args)
            self.assertEqual(sorted(found), sorted(DECODE_NAMES))
            for name, config in found.items():
                self.assertEqual(config.per_core_M, 2, name)
                self.assertTrue(config.fuse_batch and config.mcast_in0, name)

    def test_every_k_is_whole_tiles_with_a_block_width_that_divides_it(self):
        ks = dict(mlp_w1_decode_1d_progcfg_64=5120, mlp_w3_decode_1d_progcfg_64=5120, mlp_w2_decode_1d_progcfg_64=4352,
                  attn_qkv_decode_1d_progcfg_64=5120, gdn_qkvz_decode_1d_progcfg_64=5120,
                  attn_wo_decode_1d_progcfg_64=1536, gdn_out_decode_1d_progcfg_64=1536)
        for name, config in configs(self.quad).items():
            k_tiles = ks[name] // 32
            self.assertEqual(ks[name] % 32, 0, name)
            self.assertEqual(k_tiles % config.in0_block_w, 0, name)
            self.assertLessEqual(config.in0_block_w, 8, name)

    def test_the_output_partition_covers_every_tile_exactly_once(self):
        ns = dict(mlp_w1_decode_1d_progcfg_64=4352, mlp_w3_decode_1d_progcfg_64=4352, mlp_w2_decode_1d_progcfg_64=5120,
                  attn_qkv_decode_1d_progcfg_64=3584, gdn_qkvz_decode_1d_progcfg_64=4120,
                  attn_wo_decode_1d_progcfg_64=5120, gdn_out_decode_1d_progcfg_64=5120)
        for name, config in configs(self.quad).items():
            n_tiles = -(-ns[name] // 32)
            owners = tile_owners(config, n_tiles)
            self.assertEqual(sorted(owners), list(range(n_tiles)), name)
            cores = config.compute_with_storage_grid_size[0] * config.compute_with_storage_grid_size[1]
            self.assertGreaterEqual(cores * config.per_core_N, n_tiles, name)

    def test_the_n_tile_counts_at_four_cards_against_the_pairs(self):
        # the widths the configs are built for: half at four cards, except the outputs back to the residual
        self.assertEqual(self.pair.attn_qkv_fused_dim_tp, 7168)
        self.assertEqual(self.quad.attn_qkv_fused_dim_tp, 3584)
        self.assertEqual((self.pair.hidden_dim // 2, self.quad.hidden_dim // 4), (8704, 4352))

    def test_t1_11_changes_exactly_the_two_configs_and_keeps_the_reduction(self):
        before, unused = build(4, 8, {})
        after, logged = build(4, 8, {'QWEN_FAST_VERIFY_T1': '1'})
        old, new = configs(before), configs(after)
        changed = sorted(name for name in old if vars(old[name]) != vars(new[name]))
        self.assertEqual(changed, ['attn_qkv_decode_1d_progcfg_64', 'mlp_w1_decode_1d_progcfg_64'])
        for name in changed:
            for field in ('in0_block_w', 'per_core_M', 'fuse_batch', 'mcast_in0', 'fused_activation'):
                self.assertEqual(getattr(old[name], field), getattr(new[name], field), (name, field))
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0][0], 'info')
        self.assertIn('site=matmul_configs', logged[0][1])
        # the K schedule of every output tile, and one owner per tile, in the wider partition
        for name, n in (('attn_qkv_decode_1d_progcfg_64', 3584), ('mlp_w1_decode_1d_progcfg_64', 4352)):
            self.assertEqual(k_schedule(old[name], 160), k_schedule(new[name], 160), name)
            self.assertEqual(sorted(tile_owners(new[name], n // 32)), list(range(n // 32)), name)

    def test_the_wider_partition_at_four_cards_halves_the_gate_per_core_columns(self):
        # 136 gate tiles: 44 cores -> 4 tiles each, 88 cores -> 2; attn_qkv's 112: 64 cores -> 2, 44 -> 3
        before, unused = build(4, 8, {})
        after, unused = build(4, 8, {'QWEN_FAST_VERIFY_T1': '1'})
        self.assertEqual((before.mlp_w1_decode_1d_progcfg_64.per_core_N, after.mlp_w1_decode_1d_progcfg_64.per_core_N), (4, 2))
        self.assertEqual((before.attn_qkv_decode_1d_progcfg_64.per_core_N, after.attn_qkv_decode_1d_progcfg_64.per_core_N),
                         (2, 3))

    def test_the_pair_is_untouched_by_the_width_generic_graft(self):
        # the pair's seven configs: the TP2 figures test_verify_trace_t1_graft holds
        found = configs(self.pair)
        self.assertEqual(vars(found['mlp_w1_decode_1d_progcfg_64'])['per_core_N'], 7)
        self.assertEqual(vars(found['attn_qkv_decode_1d_progcfg_64'])['compute_with_storage_grid_size'], (8, 8))

    def test_a_reduction_change_is_still_refused_at_four_cards(self):
        original = create_matmul_1d_decode_progcfg

        def moved(m, k, n, num_cores, **options):
            config = original(m, k, n, num_cores, **options)
            if num_cores == 88:
                config.in0_block_w = 4
            return config

        with patch(__name__ + '.create_matmul_1d_decode_progcfg', side_effect=moved):
            with self.assertRaisesRegex(ValueError, 'mlp_w1 M=64 config would change in0_block_w'):
                build(4, 8, {'QWEN_FAST_VERIFY_T1': '1'})


if __name__ == '__main__':
    unittest.main()
