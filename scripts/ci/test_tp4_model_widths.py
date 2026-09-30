"""The image's model graft at TP4: every per-device width Qwen36ModelArgs._init_tp_config computes for the four-card
(1, 4) mesh, from the Qwen3.8-27B text config, held against the TP2 values the pair serves today.

The graft's model_config.py is loaded with stand-ins for ttnn, tt_transformers' ModelArgs and tp_common (the
builders record what they are asked for), so this runs on the CPU; the numbers are the model's
(docs/decode-payload-bound.md: 24 query / 4 KV heads of 256, 16 / 48 GDN key / value heads of 128, MLP 17,408,
hidden 5,120)."""

import importlib.util
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GRAFT = os.path.join(ROOT, 'docker', 'qwen-c2-graft', 'graft', 'model_config.py')
TILE = 32


class Recorder(object):
    """tp_common's builders: each call is recorded as (builder, args) and returned as the record."""

    TILE_SIZE = TILE

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith('create_'):
            def build(*args, **kwargs):
                record = (name, args, tuple(sorted(kwargs.items())))
                self.calls.append(record)
                return record
            return build
        raise AttributeError(name)

    def prefill_grid_default(self):
        return (8, 10)

    def prefill_tuning(self, tp):
        return dict(tp=tp)


class Grid(object):
    x, y = 11, 10


class Mesh(object):
    def __init__(self, cols):
        self.shape = (1, cols)

    def get_num_devices(self):
        return self.shape[0] * self.shape[1]

    def compute_with_storage_grid_size(self):
        return Grid()


def load_graft():
    stubs = {}

    def module(name, **attributes):
        stub = types.ModuleType(name)
        stub.__dict__.update(attributes)
        stubs[name] = stub
        return stub

    module('models')
    module('models.tt_transformers')
    module('models.tt_transformers.tt')
    module('models.tt_transformers.tt.model_config', ModelArgs=object)
    module('models.demos')
    module('models.demos.blackhole')
    module('models.demos.blackhole.qwen36')
    recorder = Recorder()
    module('models.demos.blackhole.qwen36.tt', tp_common=recorder)
    module('models.demos.blackhole.qwen36.tt.tp_common')
    simple = types.SimpleNamespace
    module('ttnn', UnaryOpType=simple(SILU='silu'), CoreGrid=lambda x, y: ('grid', x, y),
           ShardStrategy=simple(HEIGHT='height'), ShardOrientation=simple(ROW_MAJOR='row_major'),
           create_sharded_memory_config=lambda **kwargs: ('kv_shard', tuple(sorted(kwargs.items()))))
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location('qwen_c2_graft_model_config', GRAFT)
        graft = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graft)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return graft, recorder, stubs


def tp_args(tp, batch):
    """_init_tp_config on the 27B text config at `tp` devices (stand-ins for everything but the arithmetic)."""
    graft, recorder, stubs = load_graft()
    args = types.SimpleNamespace(
        n_heads=24, n_kv_heads=4, head_dim=256, dim=5120, hidden_dim=17408, max_batch_size=batch,
        linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, linear_q_dim=16 * 128, linear_k_dim=16 * 128, linear_v_dim=48 * 128,
        num_devices=tp)
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    environ = os.environ.pop('QWEN_FAST_VERIFY_T1', None)
    try:
        graft.Qwen36ModelArgs._init_tp_config(args, Mesh(tp))
    finally:
        if environ is not None:
            os.environ['QWEN_FAST_VERIFY_T1'] = environ
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return args, recorder.calls


WIDTHS = ('n_local_heads', 'n_local_kv_heads', 'kv_replication', 'gdn_nk_tp', 'gdn_nv_tp', 'gdn_qkv_dim_tp',
          'gdn_z_dim_tp', 'gdn_qkvz_dim_tp', 'gdn_qkvzab_dim_tp', 'gdn_value_dim_tp', 'gdn_key_dim_tp',
          'attn_out_dim_tp', 'attn_qkv_fused_dim_tp')


class Tp4WidthTests(unittest.TestCase):
    def test_the_pair_is_what_it_serves_today(self):
        args, _ = tp_args(2, 4)
        self.assertEqual({name: getattr(args, name) for name in WIDTHS}, dict(
            n_local_heads=12, n_local_kv_heads=2, kv_replication=False, gdn_nk_tp=8, gdn_nv_tp=24,
            gdn_qkv_dim_tp=5120, gdn_z_dim_tp=3072, gdn_qkvz_dim_tp=8192, gdn_qkvzab_dim_tp=8240,
            gdn_value_dim_tp=3072, gdn_key_dim_tp=1024, attn_out_dim_tp=3072, attn_qkv_fused_dim_tp=7168))

    def test_four_cards_halve_every_width_with_one_kv_head_each(self):
        args, _ = tp_args(4, 8)
        self.assertEqual({name: getattr(args, name) for name in WIDTHS}, dict(
            n_local_heads=6, n_local_kv_heads=1, kv_replication=False, gdn_nk_tp=4, gdn_nv_tp=12,
            gdn_qkv_dim_tp=2560, gdn_z_dim_tp=1536, gdn_qkvz_dim_tp=4096, gdn_qkvzab_dim_tp=4120,
            gdn_value_dim_tp=1536, gdn_key_dim_tp=512, attn_out_dim_tp=1536, attn_qkv_fused_dim_tp=3584))
        self.assertEqual(args.cluster_shape, [1, 4])

    def test_every_decode_and_prefill_k_is_whole_tiles_at_tp4(self):
        _, calls = tp_args(4, 8)
        for name, positional, _ in calls:
            if name in ('create_dram_sharded_matmul_program_config', 'create_matmul_1d_decode_progcfg'):
                m, k, n = positional[:3]
                self.assertEqual(k % TILE, 0, (name, positional))
            if name == 'create_dram_sharded_mem_config':
                k, n = positional
                self.assertEqual(k % TILE, 0, (name, positional))
        widths = sorted(set(call[1][2] for call in calls if call[0] == 'create_matmul_1d_decode_progcfg'))
        self.assertEqual(widths, [3584, 4120, 4352, 5120])
        self.assertIn(4352, widths, 'the MLP shard, 17408 / 4')
        self.assertIn(4120, widths, 'the fused GDN in-projection, padded by the builder as at TP2 (8240)')

    def test_the_gdn_heads_fit_one_core_wave_at_eight_seats(self):
        """gotchas: batched GDN runs B x Nv_tp (user, head) pairs, one per core, on the 110-core grid."""
        args, _ = tp_args(4, 8)
        self.assertLessEqual(8 * args.gdn_nv_tp, Grid.x * Grid.y)
        pair, _ = tp_args(2, 8)
        self.assertGreater(8 * pair.gdn_nv_tp, Grid.x * Grid.y, 'why the pair serves four seats and TP4 eight')

    def test_the_kv_update_shard_spans_the_seats(self):
        args, _ = tp_args(4, 8)
        self.assertEqual(dict(args.kv_update_shard_cfg[1])['core_grid'], ('grid', 8, 1))
        self.assertEqual(dict(args.kv_update_shard_cfg[1])['shape'], (TILE, 256))

    def test_the_prefill_conv_op_takes_the_tp4_shard(self):
        import gdn_prefill_conv_exact as pcx

        args, _ = tp_args(4, 8)
        C, kd = args.gdn_qkv_dim_tp, args.gdn_key_dim_tp
        self.assertTrue(C % TILE == 0 and kd % TILE == 0 and 0 < 2 * kd < C)
        self.assertIn((1, 4), pcx.SUPPORTED_MESHES)

    def test_the_weight_cache_is_keyed_by_the_mesh(self):
        with open(GRAFT, encoding='utf-8') as handle:
            source = handle.read()
        self.assertIn('suffix += "_mesh" + "x".join(str(d) for d in self.cluster_shape)', source)
        self.assertEqual('_mesh' + 'x'.join(str(d) for d in [1, 4]), '_mesh1x4',
                         'a TP4 cache never reuses the pair\'s _mesh1x2 files')


if __name__ == '__main__':
    unittest.main()
