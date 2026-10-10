"""WP4 (op-fusion programme): the TP4 MLP gate/up levers, QWEN_FAST_MLP_CFG (F-D2 and the streaming config) and the binder that carries both. CPU only.

Held here:

  flags      strict, one spelling per configuration, `g` and `u` refused beside the fused op, `p` refused without it, an audit needs its lever, the pair serves none;
  shapes     the TP4 per-chip decode shapes, the bytes a launch streams and the GB/s the profile reports, the builder against the graft's own transcription, the
             core counts of the profile on both grids (68 / 34 on 11x10, 68 / 46 on 13x10);
  K loop     every config the sweep proposes keeps in0_block_w, per_core_M, fuse_batch and mcast_in0; every output tile has one owner and the same K schedule; an
             emulation of the 1D mcast matmul at the real N gives bit-identical outputs for the served and the named partitions (and a different block does not);
  plan       the served config objects when nothing is named, new objects with the named partitions otherwise, a refused K loop, width and grid errors;
  forward    the lever's arm run beside the REAL grafted Qwen36MLP._forward_tp over one fake ttnn: the same op sequence but for the multiply's memory config, the
             same values, the same all-reduce call; the audit holds and compares the SwiGLU product and the layer output and serves the served result;
  binder     off = the tuple model_batch returned before (the module is not even imported); on = one more binder, one forward per layer, one call per layer per forward;
  files      every manifest entry exists, the job templates parse, nothing names a rig.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types
import unittest
from unittest import mock

import torch

import tp4_mlp_gateup as lever
import test_verify_trace_t1_graft as graft

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GRAFT_MLP = (REPO / 'docker' / 'qwen-c2-graft' / 'graft' / 'mlp.py').read_text(encoding='utf-8')
SHAPES = lever.shapes(4)
GATE, UP, DOWN = SHAPES['mlp_w1'], SHAPES['mlp_w3'], SHAPES['mlp_w2']
FOUR = {'QWEN_FAST_TP': '4'}


def env(**values):
    base = dict((key, value) for key, value in os.environ.items() if not key.startswith('QWEN_FAST_'))
    base.update(FOUR)
    base.update(values)
    return mock.patch.dict(os.environ, base, clear=True)


# ---------------------------------------------------------------------------------------------------------------------------
# Flags.
# ---------------------------------------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_nothing_set_is_off(self):
        with env():
            selection = lever.resolve()
            self.assertEqual((selection.route, selection.name, selection.audit), (None, None, False))
            self.assertFalse(lever.requested())

    def test_the_fused_flag_is_strict(self):
        with env(QWEN_FAST_MLP_GATEUP='1'):
            selection = lever.resolve()
            self.assertEqual((selection.route, selection.name, selection.pairs), ('fused', 'p3', 3))
            self.assertTrue(lever.requested())
        with env(QWEN_FAST_MLP_GATEUP='0'):
            self.assertIsNone(lever.resolve().route)
        for bad in ('2', 'true', '', ' 1', 'on'):
            with env(QWEN_FAST_MLP_GATEUP=bad), self.assertRaises(ValueError):
                lever.resolve()

    def test_the_cfg_names(self):
        with env(QWEN_FAST_MLP_CFG='l1'):
            selection = lever.resolve()
            self.assertEqual((selection.route, selection.name), ('cfg', 'l1'))
            self.assertEqual(selection.cfg, lever.Cfg('l1', None, None, None, None, None))
        with env(QWEN_FAST_MLP_CFG='g2u3d5'):
            self.assertEqual(lever.resolve().cfg, lever.Cfg('g2u3d5', 2, 3, 5, None, None))
        with env(QWEN_FAST_MLP_CFG='u4'):
            self.assertEqual(lever.resolve().cfg.u, 4)
        with env(QWEN_FAST_MLP_CFG='0'):
            self.assertIsNone(lever.resolve().route)
        for bad in ('', 'g', 'g0', 'g02', 'x3', 'u3g2', 'g2g3', 'g2 u3', 'G2', 'g2u3d5w', 'l1g2', 'true', 'p6'):
            with self.subTest(name=bad), env(QWEN_FAST_MLP_CFG=bad), self.assertRaises(ValueError):
                lever.resolve()

    def test_one_spelling_per_configuration(self):
        for g in (None, 2, 17):
            for u in (None, 3):
                for d in (None, 5):
                    for w in (None, 12):
                        name = lever.canonical_name(g=g, u=u, d=d, w=w)
                        if name == 'l1':
                            continue
                        self.assertEqual(lever.parse_cfg(name), lever.Cfg(name, g, u, d, None, w))
        self.assertEqual(lever.canonical_name(), 'l1')
        self.assertEqual(lever.canonical_name(g=2, u=3, d=5, p=4, w=13), 'g2u3d5p4w13')

    def test_g_and_u_are_refused_beside_the_fused_op_and_p_without_it(self):
        for name in ('g2', 'u3', 'g2u3d5'):
            with env(QWEN_FAST_MLP_GATEUP='1', QWEN_FAST_MLP_CFG=name), self.assertRaises(ValueError):
                lever.resolve()
        with env(QWEN_FAST_MLP_CFG='p3'), self.assertRaises(ValueError):
            lever.resolve()
        with env(QWEN_FAST_MLP_GATEUP='1', QWEN_FAST_MLP_CFG='d5p4'):
            selection = lever.resolve()
            self.assertEqual((selection.route, selection.name, selection.pairs), ('fused', 'd5p4', 4))
        with env(QWEN_FAST_MLP_GATEUP='1', QWEN_FAST_MLP_CFG='d5'):
            self.assertEqual(lever.resolve().name, 'd5p3', 'the fused route always spells its pairs per worker')
        for pairs in (2, 3, 4, 5, 7):
            with env(QWEN_FAST_MLP_GATEUP='1', QWEN_FAST_MLP_CFG='p%d' % pairs):
                self.assertEqual(lever.resolve().pairs, pairs)
        for pairs in (1, 6, 8, 9):
            with env(QWEN_FAST_MLP_GATEUP='1', QWEN_FAST_MLP_CFG='p%d' % pairs), self.assertRaises(ValueError):
                lever.resolve()

    def test_an_audit_needs_its_lever(self):
        with env(QWEN_FAST_MLP_GATEUP_AUDIT='1'), self.assertRaises(ValueError):
            lever.resolve()
        with env(QWEN_FAST_MLP_CFG_AUDIT='1'), self.assertRaises(ValueError):
            lever.resolve()
        with env(QWEN_FAST_MLP_CFG='l1', QWEN_FAST_MLP_CFG_AUDIT='1'):
            self.assertTrue(lever.resolve().audit)
        with env(QWEN_FAST_MLP_GATEUP='1', QWEN_FAST_MLP_GATEUP_AUDIT='1'):
            self.assertTrue(lever.resolve().audit)
        with env(QWEN_FAST_MLP_GATEUP='1', QWEN_FAST_MLP_CFG_AUDIT='1'), self.assertRaises(ValueError):
            lever.resolve()
        with env(QWEN_FAST_MLP_CFG='l1', QWEN_FAST_MLP_CFG_AUDIT='2'), self.assertRaises(ValueError):
            lever.resolve()

    def test_the_audit_stride_is_a_positive_integer(self):
        with env(QWEN_FAST_MLP_CFG='l1'):
            self.assertEqual(lever.resolve().stride, 4)
        with env(QWEN_FAST_MLP_CFG='l1', QWEN_FAST_MLP_AUDIT_STRIDE='1'):
            self.assertEqual(lever.resolve().stride, 1)
        for bad in ('0', '-1', '1.5', 'x', '', '1000'):
            with env(QWEN_FAST_MLP_CFG='l1', QWEN_FAST_MLP_AUDIT_STRIDE=bad), self.assertRaises(ValueError):
                lever.resolve()

    def test_the_pair_serves_none_of_it(self):
        for name, value in (('QWEN_FAST_MLP_GATEUP', '1'), ('QWEN_FAST_MLP_CFG', 'l1'), ('QWEN_FAST_MLP_AUDIT_STRIDE', '2')):
            with mock.patch.dict(os.environ, {name: value}, clear=True), self.assertRaises(ValueError):
                lever.resolve()
            with mock.patch.dict(os.environ, {name: value, 'QWEN_FAST_TP': '2'}, clear=True), self.assertRaises(ValueError):
                lever.resolve()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(lever.resolve().route)
        with mock.patch.dict(os.environ, {'QWEN_FAST_MLP_GATEUP': '0'}, clear=True):
            self.assertIsNone(lever.resolve().route)


# ---------------------------------------------------------------------------------------------------------------------------
# Shapes, bytes, the builder.
# ---------------------------------------------------------------------------------------------------------------------------

class ShapeTests(unittest.TestCase):
    def test_the_per_chip_shapes_at_four_chips(self):
        self.assertEqual((GATE.k, GATE.n, GATE.dtype, GATE.silu, GATE.requested_cores), (5120, 4352, 'bfp4', True, 88))
        self.assertEqual((UP.k, UP.n, UP.dtype, UP.silu, UP.requested_cores), (5120, 4352, 'bfp4', False, 44))
        self.assertEqual((DOWN.k, DOWN.n, DOWN.dtype, DOWN.requested_cores), (4352, 5120, 'bfp8', 33))
        self.assertEqual(lever.tiles(GATE.k), 160)
        self.assertEqual(lever.tiles(GATE.n), 136)
        self.assertEqual(lever.tiles(DOWN.n), 160)

    def test_the_other_projections_are_the_ones_the_profile_names(self):
        self.assertEqual(sorted(SHAPES), ['attn_in', 'attn_wo', 'gdn_in', 'gdn_out', 'mlp_w1', 'mlp_w2', 'mlp_w3'])
        self.assertEqual((SHAPES['attn_in'].n, SHAPES['gdn_in'].n, SHAPES['attn_wo'].k, SHAPES['gdn_out'].k), (3584, 4120, 1536, 1536))

    def test_the_bytes_and_the_bandwidth_are_the_profile_s(self):
        self.assertEqual(lever.weight_bytes(GATE), 12533760)
        self.assertEqual(lever.weight_bytes(DOWN), 23674880)
        # M676: gate 53.14 us = 236 GB/s, up 44.14 = 284, down 66.27 = 357
        self.assertAlmostEqual(lever.gbps(GATE, 53.14), 236.0, delta=0.5)
        self.assertAlmostEqual(lever.gbps(UP, 44.14), 284.0, delta=0.5)
        self.assertAlmostEqual(lever.gbps(DOWN, 66.27), 357.0, delta=0.5)
        self.assertAlmostEqual(lever.gbps(SHAPES['gdn_in'], 62.10), 362.0, delta=1.0)
        self.assertEqual(lever.PEAK_GBPS, 512.0)

    def test_the_builder_equals_the_graft_s_transcription_on_both_grids(self):
        for grid_x in (11, 13):
            for key, shape in SHAPES.items():
                for cores in (shape.requested_cores, 52, 88):
                    with self.subTest(grid=grid_x, shape=key, cores=cores):
                        mine = lever.builder_config(shape, grid_x, cores)
                        theirs = graft.create_matmul_1d_decode_progcfg(64, shape.k, shape.n, cores, grid_w=grid_x)
                        self.assertEqual(mine.grid, tuple(theirs.compute_with_storage_grid_size))
                        for field in ('in0_block_w', 'per_core_M', 'per_core_N', 'out_subblock_h', 'out_subblock_w'):
                            self.assertEqual(getattr(mine, field), getattr(theirs, field), field)

    def test_the_profile_s_core_counts_on_both_grids(self):
        gate11, up11, down11 = (lever.builder_config(shape, 11) for shape in (GATE, UP, DOWN))
        self.assertEqual([lever.active_cores(c, s) for c, s in ((gate11, GATE), (up11, UP), (down11, DOWN))], [68, 34, 32])
        gate13, up13, down13 = (lever.builder_config(shape, 13) for shape in (GATE, UP, DOWN))
        self.assertEqual([lever.active_cores(c, s) for c, s in ((gate13, GATE), (up13, UP), (down13, DOWN))], [68, 46, 32],
                         'v678: the up moved 34 -> 46 cores, the down stayed at 32')
        self.assertEqual((gate13.grid, up13.grid, down13.grid), ((13, 7), (13, 4), (13, 3)))
        self.assertEqual((gate13.per_core_N, up13.per_core_N, down13.per_core_N), (2, 3, 5))
        self.assertEqual({c.in0_block_w for c in (gate11, up11, down11, gate13, up13, down13)}, {8})

    def test_a_named_config_is_the_minimal_rectangle_with_the_served_block(self):
        served = lever.builder_config(GATE, 13)
        named = lever.named_config(GATE, 2, 13)
        self.assertEqual((named.grid, named.per_core_N, named.in0_block_w, named.per_core_M, named.out_subblock_h, named.out_subblock_w),
                         ((13, 6), 2, served.in0_block_w, served.per_core_M, served.out_subblock_h, served.out_subblock_w))
        self.assertEqual(lever.active_cores(named, GATE), lever.active_cores(served, GATE))
        self.assertEqual(lever.named_config(UP, 4, 11).grid, (11, 4))
        for bad in (0, 137, -1):
            with self.assertRaises(ValueError):
                lever.named_config(GATE, bad, 13)

    def test_the_subblock_rule_is_the_builders(self):
        self.assertEqual([lever.subblock(n) for n in (1, 2, 3, 4, 5, 6, 7, 8)],
                         [(2, 1), (2, 2), (1, 3), (1, 4), (2, 1), (1, 3), (2, 1), (1, 4)])

    def test_the_choices_fit_the_grid_and_leave_nothing_out(self):
        for grid_x, limit in ((13, 130), (11, 110)):
            choices = lever.per_core_n_choices(GATE, grid_x, 10)
            self.assertEqual(choices[0], 2, 'pcn 1 is 136 cores: more than either grid')
            cores = [-(-136 // pcn) for pcn in choices]
            self.assertEqual(cores, sorted(set(cores), reverse=True))
            self.assertTrue(all(8 <= count <= limit for count in cores))
            # a core count the grid holds is a choice unless a smaller per_core_N gives the same count
            for pcn in range(1, 137):
                count = -(-136 // pcn)
                if 8 <= count <= limit and pcn not in choices:
                    self.assertIn(-(-136 // (pcn - 1)) if pcn > 1 else 0, cores + [0], pcn)


# ---------------------------------------------------------------------------------------------------------------------------
# The K loop: what the partition may not change.
# ---------------------------------------------------------------------------------------------------------------------------

def namespace_of(config, silu=False):
    return types.SimpleNamespace(compute_with_storage_grid_size=config.grid, in0_block_w=config.in0_block_w, per_core_M=config.per_core_M,
                                 per_core_N=config.per_core_N, out_subblock_h=config.out_subblock_h, out_subblock_w=config.out_subblock_w,
                                 fuse_batch=True, mcast_in0=True, fused_activation='silu' if silu else None)


class KLoopTests(unittest.TestCase):
    def test_same_k_loop_names_what_moved(self):
        a = namespace_of(lever.builder_config(GATE, 13))
        b = namespace_of(lever.named_config(GATE, 3, 13))
        self.assertEqual(lever.same_k_loop(a, b), [])
        for field, value in (('in0_block_w', 4), ('per_core_M', 1), ('fuse_batch', False), ('mcast_in0', False)):
            moved = types.SimpleNamespace(**dict(vars(b), **{field: value}))
            self.assertEqual([item[0] for item in lever.same_k_loop(a, moved)], [field])

    def test_every_choice_keeps_the_k_schedule_and_one_owner_per_output_tile(self):
        for grid_x in (11, 13):
            for key in ('mlp_w1', 'mlp_w3', 'mlp_w2'):
                shape = SHAPES[key]
                served = namespace_of(lever.builder_config(shape, grid_x))
                reference = graft.k_schedule(served, lever.tiles(shape.k))
                for pcn in lever.per_core_n_choices(shape, grid_x, 10):
                    config = namespace_of(lever.named_config(shape, pcn, grid_x))
                    with self.subTest(grid=grid_x, shape=key, pcn=pcn):
                        owners = graft.tile_owners(config, lever.tiles(shape.n))
                        self.assertEqual(sorted(owners), list(range(lever.tiles(shape.n))))
                        self.assertEqual(graft.k_schedule(config, lever.tiles(shape.k)), reference)
                        self.assertEqual(config.in0_block_w, 8)

    # The real M (64 rows) and the real N, the real block (8 tiles); K cut to two blocks so the packer's L1 accumulation is exercised.
    K = 16 * 32

    def test_emulated_outputs_are_bit_identical_across_partitions_at_the_real_n(self):
        generator = torch.Generator().manual_seed(21)
        x = torch.randn(64, self.K, generator=generator).to(torch.bfloat16)
        weight = (torch.randn(self.K, GATE.n, generator=generator) * 0.05).to(torch.bfloat16)
        reference = None
        for pcn in (2, 3, 7):
            config = namespace_of(lever.named_config(GATE, pcn, 13), silu=True)
            out = graft.emulate(x, weight, config, silu=True)
            self.assertEqual(tuple(out.shape), (64, GATE.n))
            if reference is None:
                reference = out
            self.assertTrue(torch.equal(out.view(torch.int16), reference.view(torch.int16)), pcn)
        served = graft.emulate(x, weight, namespace_of(lever.builder_config(GATE, 13), silu=True), silu=True)
        self.assertTrue(torch.equal(served.view(torch.int16), reference.view(torch.int16)))

    def test_the_emulation_sees_a_regrouped_k_loop(self):
        generator = torch.Generator().manual_seed(22)
        x = torch.randn(64, self.K, generator=generator).to(torch.bfloat16)
        weight = (torch.randn(self.K, 512, generator=generator) * 0.05).to(torch.bfloat16)
        config = namespace_of(lever.named_config(UP, 4, 13))
        regrouped = types.SimpleNamespace(**dict(vars(config), in0_block_w=4))
        self.assertFalse(torch.equal(graft.emulate(x, weight, config).view(torch.int16), graft.emulate(x, weight, regrouped).view(torch.int16)))


# ---------------------------------------------------------------------------------------------------------------------------
# A fake ttnn and the REAL graft mlp.py over it.
# ---------------------------------------------------------------------------------------------------------------------------

class Fake(object):
    """ttnn for the grafted mlp.py and for the twin: bfloat16 torch tensors, every call recorded by name."""

    def __init__(self):
        self.calls = []
        self.names = {}
        self.freed = []
        self.pending = {}
        self.ttnn = types.SimpleNamespace(
            Tensor=object, TILE_SIZE=32, DRAM_MEMORY_CONFIG='DRAM', L1_MEMORY_CONFIG='L1', L1_WIDTH_SHARDED_MEMORY_CONFIG='L1WS',
            UnaryOpType=types.SimpleNamespace(SILU='SILU'), linear=self.linear, mul=self.mul, to_memory_config=self.to_memory_config,
            deallocate=self.deallocate, clone=self.clone, get_device_tensors=lambda tensor: [tensor], to_torch=lambda tensor: tensor,
            MatmulMultiCoreReuseMultiCast1DProgramConfig=lambda **options: types.SimpleNamespace(**options))

    def name(self, tensor):
        return self.names.get(id(tensor), 'act')

    def describe(self, config):
        if config is None or isinstance(config, str):
            return config
        return ('cfg', lever.grid_of(config), config.per_core_N, config.in0_block_w, config.fused_activation)

    def linear(self, x, weight, compute_kernel_config=None, program_config=None, memory_config=None, activation=None):
        self.calls.append(('linear', tuple(x.shape)[-2], self.name(weight), self.describe(program_config), memory_config, compute_kernel_config))
        out = x.float() @ weight.float()
        if getattr(program_config, 'fused_activation', None) in ('SILU', 'silu'):
            out = out * torch.sigmoid(out)
        return out.to(torch.bfloat16)

    def mul(self, a, b, memory_config=None):
        self.calls.append(('mul', tuple(a.shape)[-2], memory_config))
        return (a.float() * b.float()).to(torch.bfloat16)

    def to_memory_config(self, x, memory_config):
        self.calls.append(('to_memory_config', memory_config))
        return x

    def deallocate(self, tensor):
        self.freed.append(id(tensor))

    def clone(self, tensor, memory_config=None):
        self.calls.append(('clone', memory_config))
        return tensor.clone()

    def all_reduce(self, partial, device, tt_ccl, **options):
        self.calls.append(('all_reduce', device, tt_ccl, tuple(sorted(options.items()))))
        return partial

    def modules(self):
        ccl = types.ModuleType('models.tt_transformers.tt.ccl')
        ccl.tt_all_reduce = self.all_reduce
        tpc = types.SimpleNamespace(TILE_SIZE=32, mlp_gateup_agmm_enabled=lambda count: count > 1)
        loguru = types.ModuleType('loguru')
        loguru.logger = types.SimpleNamespace(info=lambda *a: None)
        found = {'ttnn': self.ttnn, 'loguru': loguru, 'models.tt_transformers.tt.ccl': ccl}
        for name in ('models', 'models.demos', 'models.demos.blackhole', 'models.demos.blackhole.qwen36', 'models.demos.blackhole.qwen36.tt',
                     'models.tt_transformers', 'models.tt_transformers.tt'):
            found[name] = types.ModuleType(name)
        found['models.demos.blackhole.qwen36.tt'].tp_common = tpc
        return mock.patch.dict(sys.modules, found)


def served_args(grid_x=13):
    """The graft's 64-row configs as the model_config builds them (the image's builder, T1 on: the gate at 88 requested cores)."""
    def build(shape, silu):
        return graft.create_matmul_1d_decode_progcfg(64, shape.k, shape.n, shape.requested_cores, fused_activation='SILU' if silu else None,
                                                     grid_w=grid_x)
    return types.SimpleNamespace(
        dim=5120, decode_grid_w=grid_x, ccl_topology=lambda: 'ring',
        mlp_w1_decode_1d_progcfg_64=build(GATE, True), mlp_w3_decode_1d_progcfg_64=build(UP, False), mlp_w2_decode_1d_progcfg_64=build(DOWN, False))


ROWS, DIM, HIDDEN = 64, 5120, 96


def make_mlp(fake, args=None, seed=0):
    """The real grafted Qwen36MLP (mlp.py executed over the fake), built the way the tests of the graft build it; small tensors, real arithmetic."""
    with fake.modules():
        namespace = {}
        exec(compile(GRAFT_MLP, 'mlp.py', 'exec'), namespace)
    cls = namespace['Qwen36MLP']
    mlp = cls.__new__(cls)
    generator = torch.Generator().manual_seed(seed)
    w1 = (torch.randn(64, HIDDEN, generator=generator) * 0.2).to(torch.bfloat16)
    w3 = (torch.randn(64, HIDDEN, generator=generator) * 0.2).to(torch.bfloat16)
    w2 = (torch.randn(HIDDEN, 64, generator=generator) * 0.2).to(torch.bfloat16)
    for tensor, label in ((w1, 'w1'), (w2, 'w2'), (w3, 'w3')):
        fake.names[id(tensor)] = label
    mlp.weights = types.SimpleNamespace(w1=w1, w2=w2, w3=w3, w_gate_up=None)
    mlp.args = args or served_args()
    mlp.args.dim = 64
    mlp.tt_ccl, mlp.device = 'ccl', 'mesh'
    mlp.compute_kernel_config, mlp.compute_kernel_config_decode = 'ckc', 'ckc_decode'
    mlp.compute_kernel_config_agmm = 'ckc_agmm'
    mlp._mlp_1d_decode, mlp._dram_sharded, mlp._fuse_gateup_agmm = True, False, False
    mlp.num_devices = 4
    return mlp, namespace


def activation(seed=1, rows=ROWS):
    return (torch.randn(1, 1, rows, 64, generator=torch.Generator().manual_seed(seed)) * 0.5).to(torch.bfloat16)


def selection_of(**values):
    with env(**values):
        return lever.resolve()


def plan_of(fake, mlp, **values):
    selection = selection_of(**values)
    return lever.build_plan(selection, mlp.args, fake.ttnn, 13, 10)


# ---------------------------------------------------------------------------------------------------------------------------
# The plan.
# ---------------------------------------------------------------------------------------------------------------------------

class PlanTests(unittest.TestCase):
    def setUp(self):
        self.fake = Fake()
        self.mlp, unused = make_mlp(self.fake)

    def test_l1_keeps_the_graft_s_own_config_objects(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='l1')
        args = self.mlp.args
        self.assertIs(plan.gate, args.mlp_w1_decode_1d_progcfg_64)
        self.assertIs(plan.up, args.mlp_w3_decode_1d_progcfg_64)
        self.assertIs(plan.down, args.mlp_w2_decode_1d_progcfg_64)
        self.assertEqual((plan.route, plan.name), ('cfg', 'l1'))

    def test_named_partitions_are_new_objects_with_the_served_k_loop(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='g3u2d4')
        self.assertEqual([c.per_core_N for c in (plan.gate, plan.up, plan.down)], [3, 2, 4])
        self.assertEqual([tuple(c.compute_with_storage_grid_size) for c in (plan.gate, plan.up, plan.down)], [(13, 4), (13, 6), (13, 4)])
        args = self.mlp.args
        for built, served in ((plan.gate, args.mlp_w1_decode_1d_progcfg_64), (plan.up, args.mlp_w3_decode_1d_progcfg_64),
                              (plan.down, args.mlp_w2_decode_1d_progcfg_64)):
            self.assertEqual(lever.same_k_loop(served, built), [])
        self.assertEqual(plan.gate.fused_activation, 'SILU', 'the gate keeps its fused SiLU')
        self.assertIsNone(plan.up.fused_activation)

    def test_only_the_named_matmul_moves(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='d4')
        self.assertIs(plan.gate, self.mlp.args.mlp_w1_decode_1d_progcfg_64)
        self.assertIs(plan.up, self.mlp.args.mlp_w3_decode_1d_progcfg_64)
        self.assertEqual(plan.down.per_core_N, 4)

    def test_the_fused_route_replaces_the_gate_and_up(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_GATEUP='1')
        self.assertEqual((plan.route, plan.gate, plan.up, plan.fused_pairs), ('fused', None, None, 3))
        self.assertIs(plan.down, self.mlp.args.mlp_w2_decode_1d_progcfg_64)
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_GATEUP='1', QWEN_FAST_MLP_CFG='d4p5')
        self.assertEqual((plan.down.per_core_N, plan.fused_pairs), (4, 5))

    def test_a_width_alone_re_lays_the_served_partition(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='w11')
        served = self.mlp.args
        self.assertEqual(plan.gate.per_core_N, served.mlp_w1_decode_1d_progcfg_64.per_core_N)
        self.assertEqual(tuple(plan.gate.compute_with_storage_grid_size), (11, 7))
        self.assertEqual(tuple(plan.up.compute_with_storage_grid_size), (11, 5))

    def test_a_config_that_would_move_the_k_loop_is_refused_by_name(self):
        fake = Fake()
        fake.ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig = lambda **options: types.SimpleNamespace(**dict(options, in0_block_w=4))
        with self.assertRaisesRegex(ValueError, r'mlp_w1.*K loop.*in0_block_w 8 -> 4'):
            lever.build_plan(selection_of(QWEN_FAST_MLP_CFG='g3'), self.mlp.args, fake.ttnn, 13, 10)

    def test_grid_and_width_errors(self):
        with self.assertRaisesRegex(ValueError, 'outside the device grid width'):
            lever.build_plan(selection_of(QWEN_FAST_MLP_CFG='w14'), self.mlp.args, self.fake.ttnn, 13, 10)
        with self.assertRaisesRegex(ValueError, 'outside 1..136'):
            lever.build_plan(selection_of(QWEN_FAST_MLP_CFG='g137'), self.mlp.args, self.fake.ttnn, 13, 10)
        with self.assertRaisesRegex(ValueError, 'needs a'):
            lever.build_plan(selection_of(QWEN_FAST_MLP_CFG='g1'), self.mlp.args, self.fake.ttnn, 13, 5)

    def test_a_model_without_the_graft_s_64_row_configs_is_refused(self):
        args = served_args()
        del args.mlp_w2_decode_1d_progcfg_64
        with self.assertRaisesRegex(ValueError, 'Lever N native graft'):
            lever.build_plan(selection_of(QWEN_FAST_MLP_CFG='l1'), args, self.fake.ttnn, 13, 10)


# ---------------------------------------------------------------------------------------------------------------------------
# The forward, beside the real grafted _forward_tp.
# ---------------------------------------------------------------------------------------------------------------------------

def run_served(fake, mlp, x):
    with fake.modules():
        fake.calls.clear()
        out = mlp._forward_tp(x)
        return out, list(fake.calls)


def run_twin(fake, mlp, x, plan, index=0, fused=None):
    forward = lever.Tp4MlpForward(mlp, index, ROWS, fake.ttnn, plan, fused=fused)
    with fake.modules():
        fake.calls.clear()
        out = forward(x)
        return out, list(fake.calls), forward


def fake_fused(fake, mlp):
    """The fused launch stand-in: silu(x w1) * (x w3) in one call, the rounding points of the unfused ops."""
    def run(x):
        fake.calls.append(('fused', tuple(x.shape)[-2]))
        w = mlp.weights
        gate = x.float() @ w.w1.float()
        gate = (gate * torch.sigmoid(gate)).to(torch.bfloat16)
        up = (x.float() @ w.w3.float()).to(torch.bfloat16)
        return (gate.float() * up.float()).to(torch.bfloat16)
    return run


class ForwardTests(unittest.TestCase):
    def setUp(self):
        lever.forget_logged()
        self.fake = Fake()
        self.mlp, unused = make_mlp(self.fake)
        self.x = activation()
        self.served, self.served_calls = run_served(self.fake, self.mlp, self.x)

    def test_the_served_arm_is_the_five_launches(self):
        kinds = [call[0] for call in self.served_calls]
        self.assertEqual(kinds, ['to_memory_config', 'linear', 'linear', 'mul', 'linear', 'all_reduce'])
        self.assertEqual(self.served_calls[3], ('mul', 64, 'DRAM'), 'the multiply writes DRAM at 64 rows')
        self.assertEqual([call[2] for call in self.served_calls if call[0] == 'linear'], ['w1', 'w3', 'w2'])
        self.assertEqual(self.served_calls[1][5], 'ckc_decode')

    def test_l1_is_the_served_sequence_with_the_multiply_in_l1_and_the_same_values(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='l1')
        out, calls, forward = run_twin(self.fake, self.mlp, self.x, plan)
        expected = [('mul', 64, 'L1') if call[0] == 'mul' else call for call in self.served_calls]
        self.assertEqual(calls, expected)
        self.assertTrue(torch.equal(out.view(torch.int16), self.served.view(torch.int16)))
        self.assertEqual((forward.calls, forward.engaged, forward.fell_back), (1, 1, 0))

    def test_named_partitions_change_the_configs_and_nothing_else(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='g3u3d4')
        out, calls, forward = run_twin(self.fake, self.mlp, self.x, plan)
        linears = [call for call in calls if call[0] == 'linear']
        self.assertEqual([(call[2], call[3][2]) for call in linears], [('w1', 3), ('w3', 3), ('w2', 4)])
        self.assertEqual([call[3][1] for call in linears], [(13, 4), (13, 4), (13, 4)])
        self.assertEqual([call[4] for call in linears], ['L1', 'L1', 'L1'], 'every projection writes L1, as served')
        rest = [call for call in calls if call[0] != 'linear']
        served_rest = [call for call in self.served_calls if call[0] != 'linear']
        self.assertEqual(rest, [('mul', 64, 'L1') if call[0] == 'mul' else call for call in served_rest])
        self.assertTrue(torch.equal(out.view(torch.int16), self.served.view(torch.int16)))

    def test_the_compute_kernel_config_follows_the_served_arm_s_rule(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='l1')
        out, calls, forward = run_twin(self.fake, self.mlp, self.x, plan)
        self.assertEqual({call[5] for call in calls if call[0] == 'linear'}, {'ckc_decode'})

    def test_the_all_reduce_is_the_process_s_at_call_time(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='l1')
        later = Fake()
        forward = lever.Tp4MlpForward(self.mlp, 0, ROWS, self.fake.ttnn, plan)
        with later.modules():            # a wrapper installed after the binding (tile_collective_tp.install) is the one reached
            forward(self.x)
        self.assertEqual([call[0] for call in later.calls], ['all_reduce'])
        self.assertEqual(later.calls[0][1:3], ('mesh', 'ccl'))

    def test_the_fused_route_is_one_launch_for_the_gate_and_up(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_GATEUP='1')
        out, calls, forward = run_twin(self.fake, self.mlp, self.x, plan, fused=fake_fused(self.fake, self.mlp))
        self.assertEqual([call[0] for call in calls], ['to_memory_config', 'fused', 'linear', 'all_reduce'])
        self.assertEqual(calls[2][2:5], ('w2', ('cfg', (13, 3), 5, 8, None), 'L1'))
        self.assertTrue(torch.equal(out.view(torch.int16), self.served.view(torch.int16)),
                        'the stand-in rounds where the unfused ops round: the structure is what is held here')

    def test_the_engaged_line_is_logged_once_with_the_route_and_the_configs(self):
        lines = []
        with mock.patch.object(lever, 'log_line', lines.append):
            plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='g3u3d4')
            lever.Tp4MlpBinding(types.SimpleNamespace(layers=[types.SimpleNamespace(feed_forward=self.mlp)] * 2), ROWS, self.fake.ttnn, plan)
            for _ in range(3):
                run_twin(self.fake, self.mlp, self.x, plan)
        engaged = [line for line in lines if line.startswith(lever.ENGAGED)]
        self.assertEqual(len(engaged), 1)
        self.assertIn('route=cfg name=g3u3d4 rows=64 layers=2 gate=13x4/pcn3/blk8/sub1x3', engaged[0])
        self.assertIn('l1_multiply=1 audit=0', engaged[0])
        self.assertFalse([line for line in lines if lever.FALLBACK in line])

    def test_the_engaged_line_reads_a_ttnn_style_grid_with_x_and_y(self):
        class CoreCoord(object):
            def __init__(self, x, y):
                self.x, self.y = x, y

            def __getitem__(self, index):
                raise TypeError('CoreCoord is not subscriptable')

        args = self.mlp.args
        for name in ('mlp_w1_decode_1d_progcfg_64', 'mlp_w3_decode_1d_progcfg_64', 'mlp_w2_decode_1d_progcfg_64'):
            config = getattr(args, name)
            grid = config.compute_with_storage_grid_size
            config.compute_with_storage_grid_size = CoreCoord(*grid)
        lines = []
        with mock.patch.object(lever, 'log_line', lines.append):
            plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='l1')
            run_twin(self.fake, self.mlp, self.x, plan)
        self.assertIn('gate=13x7/pcn2/blk8/sub2x2', lines[0])

    def test_a_log_line_that_cannot_be_built_never_takes_the_round(self):
        lines = []
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='l1')
        with mock.patch.object(lever, 'log_line', lines.append), mock.patch.object(lever.Plan, 'describe', side_effect=RuntimeError('boom')):
            out, calls, forward = run_twin(self.fake, self.mlp, self.x, plan)
        self.assertTrue(torch.equal(out.view(torch.int16), self.served.view(torch.int16)))
        self.assertEqual(len(lines), 1)
        self.assertIn('describe=RuntimeError', lines[0])
        self.assertTrue(lines[0].startswith(lever.ENGAGED + ' route=cfg name=l1'))

    def test_a_call_the_lever_cannot_take_runs_the_served_arm_and_says_why(self):
        lines = []
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='l1')
        with mock.patch.object(lever, 'log_line', lines.append):
            for x, why in ((activation(rows=32), 'activation rows 32, bound for 64'), (torch.zeros(1, 64, 64, dtype=torch.bfloat16), 'rank 3'),
                           (torch.zeros(1, 1, 64, 32, dtype=torch.bfloat16), 'not the full K')):
                forward = lever.Tp4MlpForward(self.mlp, 0, ROWS, self.fake.ttnn, plan)
                with self.fake.modules():
                    self.fake.calls.clear()
                    try:
                        forward(x)
                    except Exception:   # the served arm may refuse the odd shapes in this stub; the line is what is held
                        pass
                self.assertEqual((forward.engaged, forward.fell_back), (0, 1), why)
        reasons = ' | '.join(lines)
        for fragment in ('activation rows 32, bound for 64', 'activation rank 3', 'not the full K'):
            self.assertIn(fragment, reasons)
        self.assertTrue(all(line.startswith(lever.FALLBACK) for line in lines))

    def test_the_mlp_off_the_1d_decode_arm_falls_back(self):
        self.mlp._mlp_1d_decode = False
        self.assertIn('1D decode', lever.refusal(self.mlp, self.x, ROWS))
        self.mlp._mlp_1d_decode, self.mlp._dram_sharded = True, True
        self.assertIn('1D decode', lever.refusal(self.mlp, self.x, ROWS))
        self.mlp._dram_sharded = False
        self.assertIsNone(lever.refusal(self.mlp, self.x, ROWS))


# ---------------------------------------------------------------------------------------------------------------------------
# The audit.
# ---------------------------------------------------------------------------------------------------------------------------

class AuditTests(unittest.TestCase):
    def setUp(self):
        lever.forget_logged()
        self.fake = Fake()
        self.mlp, unused = make_mlp(self.fake)
        self.x = activation()
        self.served, self.served_calls = run_served(self.fake, self.mlp, self.x)

    def audited(self, **values):
        values.setdefault('QWEN_FAST_MLP_CFG_AUDIT', '1')
        values.setdefault('QWEN_FAST_MLP_CFG', 'g3u3d4')
        plan = plan_of(self.fake, self.mlp, **values)
        return plan, run_twin(self.fake, self.mlp, self.x, plan)

    def test_it_serves_the_served_result_and_holds_two_pairs_with_one_all_reduce(self):
        plan, (out, calls, forward) = self.audited()
        self.assertTrue(torch.equal(out.view(torch.int16), self.served.view(torch.int16)))
        self.assertEqual([(pair['kind'], pair['layer'], pair['route']) for pair in lever._HELD], [('hidden', 0, 'cfg'), ('down', 0, 'cfg')])
        kinds = [call[0] for call in calls]
        self.assertEqual(kinds.count('all_reduce'), 1, 'the block counts the layer\'s all-reduces (tile_collective_tp.block_scope) and the ring\'s parity matters')
        self.assertEqual(kinds.count('clone'), 5, 'a copy of x for the served arm, four DRAM copies of the products and down outputs')
        self.assertEqual([call for call in calls if call[0] == 'mul'], [('mul', 64, 'L1'), ('mul', 64, 'DRAM')], 'the lever\'s, then the served one')
        linears = [(call[2], call[3]) for call in calls if call[0] == 'linear']
        self.assertEqual([name for name, unused in linears], ['w1', 'w3', 'w2', 'w1', 'w3', 'w2'])
        served = self.mlp.args
        self.assertEqual([config for name, config in linears[3:]],
                         [self.fake.describe(served.mlp_w1_decode_1d_progcfg_64), self.fake.describe(served.mlp_w3_decode_1d_progcfg_64),
                          self.fake.describe(served.mlp_w2_decode_1d_progcfg_64)], 'the served half uses the graft\'s own configs')

    def test_the_served_half_of_the_audit_is_the_graft_s_arm_op_for_op(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='g3u3d4', QWEN_FAST_MLP_CFG_AUDIT='1')
        forward = lever.Tp4MlpForward(self.mlp, 0, ROWS, self.fake.ttnn, plan)
        with self.fake.modules():
            self.fake.calls.clear()
            x_il = self.fake.ttnn.to_memory_config(self.x, 'L1')
            product = forward.served_hidden(x_il, 'ckc_decode')
            partial = forward.down(product, 'ckc_decode', self.mlp.args.mlp_w2_decode_1d_progcfg_64)
            ours = list(self.fake.calls)
        served = [call for call in self.served_calls if call[0] != 'all_reduce']
        self.assertEqual(ours, served)
        self.assertTrue(torch.equal(partial, self.served))        # the all-reduce is the identity in the fake

    def test_the_stride_picks_the_layers(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_CFG='l1', QWEN_FAST_MLP_CFG_AUDIT='1', QWEN_FAST_MLP_AUDIT_STRIDE='4')
        held = []
        for index in range(8):
            run_twin(self.fake, self.mlp, self.x, plan, index=index)
            held.append(len(lever._HELD))
        self.assertEqual(held, [2, 2, 2, 2, 4, 4, 4, 4], 'layers 0 and 4 of 8')

    def test_claim_replay_round_release(self):
        plan, unused = self.audited()
        lines = []
        owner = object()
        self.assertEqual(lever.audit_claim(owner, 'capture'), 2)
        self.assertEqual(lever.audit_claim(object(), 'capture'), 0, 'claimed once')
        lever.audit_replayed(owner)
        self.assertEqual(lever.audit_round(self.fake.ttnn, owner, 1, log=lines.append), 2)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(lever.AUDIT_MARKER) and 'exact=True' in lines[0] and 'route=cfg' in lines[0] and 'name=g3u3d4' in lines[0])
        self.assertEqual(len(lever.audit_held_of(owner)), 4)
        self.assertEqual(lever.audit_release(self.fake.ttnn, owner), 2)
        self.assertEqual(lever._HELD, [])
        self.assertEqual(lever.audit_round(self.fake.ttnn, owner, 2), 0, 'nothing held: the audit is off for this owner')

    def test_a_differing_element_is_a_mismatch_line_and_an_error(self):
        self.audited()
        owner = object()
        lever.audit_claim(owner)
        lever.audit_replayed(owner)
        lever._HELD[1]['mine'].view(torch.int16).flatten()[5] ^= 1
        lines = []
        with self.assertRaisesRegex(AssertionError, 'audit mismatch'):
            lever.audit_round(self.fake.ttnn, owner, 1, log=lines.append)
        self.assertTrue(lines[0].startswith(lever.AUDIT_MISMATCH) and 'exact=False' in lines[0] and 'layer=0 down' in lines[0], lines)
        self.assertNotIn('exact=True', lines[0])

    def test_minus_zero_and_plus_zero_differ(self):
        self.audited()
        owner = object()
        lever.audit_claim(owner)
        lever.audit_replayed(owner)
        pair = lever._HELD[0]
        pair['mine'].view(torch.int16).flatten()[0] = 0
        pair['served'].view(torch.int16).flatten()[0] = -32768
        with self.assertRaises(AssertionError):
            lever.audit_round(self.fake.ttnn, owner, 1, log=lambda line: None)

    def test_a_round_read_after_another_block_replayed_is_refused(self):
        self.audited()
        owner, other = object(), object()
        lever.audit_claim(owner)
        lever.audit_replayed(other)
        with self.assertRaisesRegex(AssertionError, 'after another block replayed'):
            lever.audit_round(self.fake.ttnn, owner, 1, log=lambda line: None)

    def test_the_fused_route_audit_labels_its_route(self):
        plan = plan_of(self.fake, self.mlp, QWEN_FAST_MLP_GATEUP='1', QWEN_FAST_MLP_GATEUP_AUDIT='1')
        out, calls, forward = run_twin(self.fake, self.mlp, self.x, plan, fused=fake_fused(self.fake, self.mlp))
        self.assertEqual({pair['route'] for pair in lever._HELD}, {'fused'})
        self.assertEqual({pair['name'] for pair in lever._HELD}, {'p3'})
        owner = object()
        lever.audit_claim(owner)
        lever.audit_replayed(owner)
        lines = []
        lever.audit_round(self.fake.ttnn, owner, 1, log=lines.append)
        self.assertIn('route=fused name=p3', lines[0])


# ---------------------------------------------------------------------------------------------------------------------------
# The binder and the hook.
# ---------------------------------------------------------------------------------------------------------------------------

def model_of(fake, layers=3, grid=(13, 10), native=True):
    mlps = []
    for index in range(layers):
        mlp, unused = make_mlp(fake, seed=index)
        mlps.append(mlp)
    args = mlps[0].args
    for mlp in mlps:
        mlp.args = args
        mlp.compute_kernel_config_decode = types.SimpleNamespace(math_approx_mode=True)
    if not native:
        del args.mlp_w1_decode_1d_progcfg_64
    return types.SimpleNamespace(args=args, layers=[types.SimpleNamespace(feed_forward=mlp) for mlp in mlps],
                                 mesh_device=types.SimpleNamespace(compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=grid[0], y=grid[1])))


class BinderTests(unittest.TestCase):
    def setUp(self):
        lever.forget_logged()
        self.fake = Fake()

    def test_nothing_set_binds_nothing(self):
        with env():
            self.assertEqual(lever.bindings(model_of(self.fake), 64, self.fake.ttnn), ())

    def test_a_lever_binds_one_forward_per_layer_and_counts_one_call_per_layer(self):
        model = model_of(self.fake, layers=3)
        with env(QWEN_FAST_MLP_CFG='g3u3d4'):
            (binder,) = lever.bindings(model, 64, self.fake.ttnn)
        self.assertEqual((binder.label, binder.expected_calls, binder.rows), ('MLP gate/up lever', 3, 64))
        self.assertEqual([(instance is layer.feed_forward, name) for (instance, name, value), layer in zip(binder.bindings, model.layers)],
                         [(True, 'forward')] * 3)
        self.assertTrue(all(isinstance(value, lever.Tp4MlpForward) for unused, unused2, value in binder.bindings))
        # model_batch's instance_overrides + per-forward count
        from model_batch import instance_overrides
        before = binder.calls
        with instance_overrides(binder.bindings), self.fake.modules():
            for layer in model.layers:
                layer.feed_forward.forward(activation())
        self.assertEqual(binder.calls - before, binder.expected_calls)
        self.assertNotIn('forward', model.layers[0].feed_forward.__dict__, 'the override is gone after the forward')

    def test_a_block_that_is_not_64_rows_or_not_native_binds_nothing_after_one_fell_back_line_each(self):
        lines = []
        with mock.patch.object(lever, 'log_line', lines.append), env(QWEN_FAST_MLP_CFG='l1'):
            self.assertEqual(lever.bindings(model_of(self.fake), 128, self.fake.ttnn), ())
            self.assertEqual(lever.bindings(model_of(self.fake), 64, self.fake.ttnn, native_m3=False), ())
            self.assertEqual(lever.bindings(model_of(self.fake, native=False), 64, self.fake.ttnn), ())
        self.assertEqual(len([line for line in lines if lever.FALLBACK in line]), 3)
        self.assertIn('64-row block', lines[0])
        self.assertIn('Lever N native graft', lines[1])

    def test_a_refused_k_loop_binds_nothing_with_its_line(self):
        model = model_of(self.fake)
        lines = []
        self.fake.ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig = lambda **options: types.SimpleNamespace(**dict(options, in0_block_w=2))
        with mock.patch.object(lever, 'log_line', lines.append), env(QWEN_FAST_MLP_CFG='d4'):
            self.assertEqual(lever.bindings(model, 64, self.fake.ttnn), ())
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(lever.FALLBACK + ' reason=') and 'K loop' in lines[0], lines)

    def test_a_fused_launch_that_cannot_be_built_binds_nothing(self):
        model = model_of(self.fake, layers=2)
        lines = []

        class Broken(object):
            def __init__(self, *args, **options):
                raise OSError('the native compute kernel source is not there')

        module = types.ModuleType('tp4_mlp_fused')
        module.FusedGateUp = Broken
        with mock.patch.dict(sys.modules, {'tp4_mlp_fused': module}), mock.patch.object(lever, 'log_line', lines.append), env(QWEN_FAST_MLP_GATEUP='1'):
            self.assertEqual(lever.bindings(model, 64, self.fake.ttnn), ())
        self.assertEqual(lines, [lever.FALLBACK + ' reason=the native compute kernel source is not there'])

    def test_a_bad_flag_combination_still_raises_at_the_warm(self):
        with env(QWEN_FAST_MLP_GATEUP='1', QWEN_FAST_MLP_CFG='g2'), self.assertRaises(ValueError):
            lever.bindings(model_of(self.fake), 64, self.fake.ttnn)

    def test_the_fused_route_builds_one_op_per_layer_with_the_served_approximation_mode(self):
        model = model_of(self.fake, layers=2)
        built = []

        class Op(object):
            def __init__(self, operations, mesh, w1, w3, **options):
                built.append((mesh, w1, w3, options))

        module = types.ModuleType('tp4_mlp_fused')
        module.FusedGateUp = Op
        with mock.patch.dict(sys.modules, {'tp4_mlp_fused': module}), env(QWEN_FAST_MLP_GATEUP='1', QWEN_FAST_MLP_CFG='d4p4w12'):
            (binder,) = lever.bindings(model, 64, self.fake.ttnn)
        self.assertEqual(len(built), 2)
        self.assertEqual(built[0][0], 'mesh')
        self.assertIs(built[0][1], model.layers[0].feed_forward.weights.w1)
        self.assertEqual(built[0][3], dict(pairs_per_worker=4, grid=(13, 10), width=12, math_approx_mode=True))
        self.assertEqual(binder.plan.fused_pairs, 4)
        self.assertEqual(binder.plan.down.per_core_N, 4)


class HookTests(unittest.TestCase):
    """two_tile_decode.bind_two_tile_mlp (reached through model_batch.two_tile_bindings, which is not edited) hands the lever's binder the MLP's place when a flag
    asks and binds nothing new otherwise."""

    def setUp(self):
        import test_two_tile_decode as tests
        self.tests = tests

    def bind(self, **values):
        from model_batch import two_tile_bindings
        ttnn = self.tests.FakeTTNN()
        model = self.tests.fake_model(ttnn, native_m3=True)
        with self.tests.decode_mode_module(), mock.patch('dflash_device.pindiag'), env(**values):
            return two_tile_bindings(64, model, ttnn), model, ttnn

    def test_without_a_flag_the_tuple_is_the_four_binders_and_the_module_is_never_called(self):
        with mock.patch.object(lever, 'bindings') as called:
            tuple_, unused, unused2 = self.bind()
        called.assert_not_called()
        self.assertEqual([binder.label for binder in tuple_], ['decode norm', 'full-attention forward', 'MLP forward', 'GDN output projection'])
        self.assertEqual([binder.expected_calls for binder in tuple_], [129, 16, 0, 0])

    def test_with_the_module_absent_and_no_flag_nothing_imports_it(self):
        with mock.patch.dict(sys.modules, {'tp4_mlp_gateup': None}):
            tuple_, unused, unused2 = self.bind()
        self.assertEqual(len(tuple_), 4)

    def test_a_flag_puts_the_lever_s_binder_in_the_mlp_s_place_and_keeps_the_tuple_s_shape(self):
        sentinel = types.SimpleNamespace(label='MLP gate/up lever', bindings=[], expected_calls=64, calls=0)
        with mock.patch.object(lever, 'bindings', return_value=(sentinel,)) as called:
            tuple_, model, ttnn = self.bind(QWEN_FAST_MLP_CFG='l1')
        self.assertEqual(len(tuple_), 4, 'norm, attention, MLP, GDN output: every positional unpack keeps working')
        self.assertIs(tuple_[2], sentinel)
        called.assert_called_once()
        self.assertEqual(called.call_args.args[1], 64)
        self.assertIs(called.call_args.kwargs['native_m3'], True)

    def test_a_lever_that_binds_nothing_leaves_the_binder_that_was_there(self):
        with mock.patch.object(lever, 'bindings', return_value=()):
            tuple_, model, ttnn = self.bind(QWEN_FAST_MLP_CFG='l1')
        self.assertEqual(tuple_[2].label, 'MLP forward')
        self.assertEqual(tuple_[2].expected_calls, 0)

    def test_the_audit_flags_and_the_stride_alone_also_reach_the_module_so_a_stray_one_raises(self):
        for name, value in (('QWEN_FAST_MLP_GATEUP_AUDIT', '1'), ('QWEN_FAST_MLP_CFG_AUDIT', '1'), ('QWEN_FAST_MLP_AUDIT_STRIDE', '2')):
            with self.subTest(flag=name), self.assertRaises(ValueError):
                self.bind(**{name: value})

    def test_the_flags_the_hook_watches_are_the_modules(self):
        import two_tile_decode
        self.assertEqual(tuple(two_tile_decode.MLP_LEVER_FLAGS), tuple(lever.ALL_FLAGS))

    def test_the_lever_s_binder_runs_inside_the_blocks_per_forward_check(self):
        """model_batch.run counts binder.calls against expected_calls; the lever's binder satisfies the contract end to end over the real Tp4MlpForward."""
        fake = Fake()
        model = model_of(fake, layers=3)
        with env(QWEN_FAST_MLP_CFG='l1'):
            binder = lever.bindings(model, 64, fake.ttnn)[0]
        from model_batch import instance_overrides
        before = binder.calls
        with instance_overrides(binder.bindings), fake.modules():
            for layer in model.layers:
                layer.feed_forward.forward(activation())
        self.assertEqual(binder.calls - before, binder.expected_calls)


# ---------------------------------------------------------------------------------------------------------------------------
# The smoke rules.
# ---------------------------------------------------------------------------------------------------------------------------

import tp4_mlp_gateup_smoke as smoke


def engaged_line(route='cfg', name='g3u3d4', rows=64, layers=64, audit=0):
    return ('[PINDIAG] tp4 mlp gateup engaged route=%s name=%s rows=%d layers=%d gate=13x4/pcn3/blk8/sub1x3 up=13x4/pcn3/blk8/sub1x3 '
            'down=13x4/pcn4/blk8/sub1x4 l1_multiply=1 audit=%d stride=4' % (route, name, rows, layers, audit))


def audit_line(route='cfg', name='g3u3d4', exact='True'):
    return '[PINDIAG] tp4 mlp gateup audit 1 exact=%s route=%s name=%s owner=capture1 round=1 layers=16 pairs=32 chips=4 elements=1' % (exact, route, name)


class SmokeTests(unittest.TestCase):
    ENV = {'QWEN_FAST_TP': '4', 'QWEN_FAST_MLP_CFG': 'g3u3d4'}

    def check(self, env, *lines):
        return smoke.problems(dict(env), '\n'.join(('2026-10-10 INFO ' + line) for line in lines))

    def test_a_clean_timed_arm(self):
        self.assertEqual(self.check(self.ENV, engaged_line()), [])

    def test_a_clean_audited_arm(self):
        env = dict(self.ENV, QWEN_FAST_MLP_CFG_AUDIT='1')
        self.assertEqual(self.check(env, engaged_line(audit=1), audit_line()), [])

    def test_the_fused_route_and_its_canonical_name(self):
        env = {'QWEN_FAST_TP': '4', 'QWEN_FAST_MLP_GATEUP': '1'}
        self.assertEqual(self.check(env, engaged_line('fused', 'p3')), [])
        found = self.check(env, engaged_line('cfg', 'l1'))
        self.assertTrue(found and 'route=fused name=p3' in found[0], found)
        env = dict(env, QWEN_FAST_MLP_CFG='d4', QWEN_FAST_MLP_GATEUP_AUDIT='1')
        self.assertEqual(self.check(env, engaged_line('fused', 'd4p3'), audit_line('fused', 'd4p3')), [])

    def test_lines_without_a_lever_are_a_problem(self):
        found = self.check({'QWEN_FAST_TP': '4'}, engaged_line())
        self.assertEqual(len(found), 1)
        self.assertIn('no MLP lever is on', found[0])
        self.assertEqual(smoke.problems({'QWEN_FAST_TP': '4'}, 'nothing about the MLP'), [])

    def test_no_engaged_line_or_another_configuration_is_a_problem(self):
        self.assertIn('no engaged line', self.check(self.ENV, 'unrelated')[0])
        found = self.check(self.ENV, engaged_line(name='g2'))
        self.assertTrue(found and 'engaged lines seen' in found[0] and 'name=g2' in found[0], found)

    def test_the_block_the_layers_and_the_l1_multiply_are_checked(self):
        self.assertIn('64-row block', self.check(self.ENV, engaged_line(rows=32))[0])
        self.assertIn('all 64 layers', self.check(self.ENV, engaged_line(layers=48))[0])
        broken = engaged_line().replace('l1_multiply=1', 'l1_multiply=0')
        self.assertIn('multiply was written to L1', self.check(self.ENV, broken)[0])

    def test_a_fell_back_line_fails_even_beside_an_engaged_one(self):
        found = self.check(self.ENV, engaged_line(), '[PINDIAG] tp4 mlp gateup fell back reason=activation rows 32, bound for 64')
        self.assertEqual(len(found), 1)
        self.assertIn('fell back', found[0])

    def test_an_audit_flag_needs_a_passing_line_for_its_own_configuration(self):
        env = dict(self.ENV, QWEN_FAST_MLP_CFG_AUDIT='1')
        self.assertIn('no passing audit line', self.check(env, engaged_line(audit=1))[0])
        self.assertIn('no passing audit line', self.check(env, engaged_line(audit=1), audit_line(name='l1'))[0])
        found = self.check(env, engaged_line(audit=1), audit_line(exact='False'))
        self.assertTrue(any('no passing audit line' in item for item in found) and any('found a difference' in item for item in found), found)

    def test_a_mismatch_fails_whatever_else_is_in_the_log(self):
        env = dict(self.ENV, QWEN_FAST_MLP_CFG_AUDIT='1')
        mismatch = '[PINDIAG] tp4 mlp gateup audit mismatch round=1 route=cfg name=g3u3d4 exact=False layer=4 down chip 2: 3 of 327680 elements differ'
        found = self.check(env, engaged_line(audit=1), audit_line(), mismatch)
        self.assertTrue(any('found a difference' in item for item in found), found)

    def test_an_audit_line_without_the_audit_flag_is_a_problem(self):
        found = self.check(self.ENV, engaged_line(), audit_line())
        self.assertEqual(len(found), 1)
        self.assertIn('no audit flag', found[0])

    def test_a_flag_combination_the_lever_refuses_is_reported_not_raised(self):
        found = smoke.problems({'QWEN_FAST_TP': '4', 'QWEN_FAST_MLP_CFG': 'u3g2'}, '')
        self.assertTrue(found and 'not a configuration' in found[0])

    def test_the_markers_are_the_modules(self):
        self.assertEqual(smoke.PREFIX, '[PINDIAG] tp4 mlp gateup')
        for marker in (lever.ENGAGED, lever.FALLBACK, lever.AUDIT_MARKER, lever.AUDIT_MISMATCH):
            self.assertTrue(marker.startswith(smoke.PREFIX))


# ---------------------------------------------------------------------------------------------------------------------------
# Files: the manifest, the job templates, the public text.
# ---------------------------------------------------------------------------------------------------------------------------

FOLDER = HERE / 'references' / 'fusion-jobs' / 'WP4'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|zot\.|@[A-Z0-9_]+@')


class FileTests(unittest.TestCase):
    def test_the_runtime_files_exist_and_the_manifest_names_them(self):
        for name in lever.RUNTIME_FILES:
            self.assertTrue((HERE / name).is_file(), name)
        manifest = json.loads((HERE / 'fusion-wp' / 'WP4.json').read_text(encoding='utf-8'))
        listed = [item if isinstance(item, str) else item['path'] for item in manifest['image_files']]
        for name in lever.RUNTIME_FILES + ('two_tile_decode.py',):
            self.assertIn('scripts/ci/' + name, listed)

    def test_the_manifest_levers_are_the_flags_of_this_module(self):
        manifest = json.loads((HERE / 'fusion-wp' / 'WP4.json').read_text(encoding='utf-8'))
        by_id = dict((lever_['id'], lever_) for lever_ in manifest['levers'])
        self.assertEqual((by_id['mlpcfg']['flag'], by_id['mlpcfg']['value'], by_id['mlpcfg']['audit_flag']),
                         (lever.CFG, 'l1', lever.CFG_AUDIT))
        self.assertEqual((by_id['mlpgu']['flag'], by_id['mlpgu']['value'], by_id['mlpgu']['audit_flag']),
                         (lever.GATEUP, '1', lever.GATEUP_AUDIT))
        for entry in by_id.values():
            self.assertTrue(entry['engaged'].startswith(lever.ENGAGED), entry['engaged'])
            self.assertEqual(entry['fell_back'], lever.FALLBACK)
            self.assertEqual(entry['audit'], lever.AUDIT_MARKER)
        # the profile twins the generator would write parse through this module
        for entry in by_id.values():
            with env(**{entry['flag']: entry['value']}):
                self.assertIsNotNone(lever.resolve().route)
            with env(**{entry['flag']: entry['value'], entry['audit_flag']: '1'}):
                self.assertTrue(lever.resolve().audit)

    def test_every_job_template_parses_through_the_job_reader_and_is_in_the_order(self):
        import c2_serving_job as job
        with open(HERE / 'qwen_c2_profiles.json', encoding='utf-8') as handle:
            names = sorted(json.load(handle)['profiles'])
        order = [line.split() for line in (FOLDER / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]
        self.assertEqual(sorted(row[0] for row in order), sorted(path.stem for path in FOLDER.glob('*.env')))
        for row in order:
            self.assertEqual(len(row), 4)
            self.assertIn(row[1], ('stop', 'soft'))
            text = (FOLDER / (row[0] + '.env')).read_text(encoding='utf-8')
            values = job.parse_env(text)
            parsed = job.read_job(values, names)          # raises JobError on anything the workflow must not run
            self.assertIsNotNone(parsed)
            self.assertEqual(values['C2_ACTIONS'], 'cardm')
            self.assertEqual(values['C2_CARDM_HARNESS'], 'optimisation/ttnn-op/mlp_gateup/run_card_m.sh')
            self.assertEqual(values['C2_CARDS'], 'pair')
            self.assertIsNone(BANNED.search(text))
            result = subprocess.run([sys.executable, '-s', str(HERE / 'c2_serving_job.py'), str(FOLDER / (row[0] + '.env')),
                                     str(HERE / 'qwen_c2_profiles.json')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('C2_TT_GRID=10,9', (FOLDER / 'C2-mlp-sweep-grid11.env').read_text(encoding='utf-8'))
        self.assertNotIn('C2_TT_GRID', (FOLDER / 'C1-mlp-sweep.env').read_text(encoding='utf-8'))

    def test_the_first_job_is_the_sweep_with_the_composition(self):
        order = [line.split() for line in (FOLDER / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]
        self.assertEqual(order[0][:2], ['C1-mlp-sweep', 'stop'])
        text = (FOLDER / 'C1-mlp-sweep.env').read_text(encoding='utf-8')
        self.assertIn('C2_CARDM_ARGS=--shapes mlp_w1,mlp_w3,mlp_w2 --arms sweep,compose', text)

    def test_new_source_files_name_no_rig_address_registry_or_digest(self):
        for name in lever.RUNTIME_FILES + ('tp4_mlp_gateup_smoke.py', 'test_tp4_mlp_gateup.py', 'test_tp4_mlp_fused.py'):
            text = (HERE / name).read_text(encoding='utf-8')
            if not name.startswith('test_'):
                self.assertIsNone(BANNED.search(text), name)


if __name__ == '__main__':
    unittest.main()
