"""Verify-trace T1, wave 1 (#11) and its wiring: the graft, the gate and the arm.

#11 re-partitions two of the seven M = 64 decode matmul configs under QWEN_FAST_VERIFY_T1=1
(lever_n_m3native_patch section A2). The claim is class B*: only which core computes each
output tile and the subblock shape move, never how that tile is reduced over K. Held here:

  - the grafted _init_tp_config, executed over a transcription of the image's builder, builds
    every config exactly as before with the flag unset, and with it set changes exactly two,
    keeping in0_block_w, per_core_M, fuse_batch, mcast_in0 and the fused activation;
  - every output tile keeps its K-block schedule and is computed by exactly one core;
  - an emulation of the 1D mcast matmul (sequential fp32 accumulation within a K block, the
    packer adding each block's partial into L1, SiLU on the gate) at the real shapes gives
    bit-identical bf16 outputs for the old and new partitions - and a different in0_block_w,
    the T-class change this cut must never make, does not (the negative control).

What only hardware can show - that the FPU does what the emulation assumes - is the G0 byte
compare, verify_t1_device_compare.py, which must pass before any token gate.
"""

import ast
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

import lever_n_m3native_patch as patcher
from test_lever_n_m3native_patch import MODEL_CONFIG

HERE = Path(__file__).parent
ARM = HERE / 'lever_n_m3native_run_arm.sh'


# ---------------------------------------------------------------------------------------
# The image's builder, transcribed: tp_common.create_matmul_1d_decode_progcfg and
# _find_largest_divisor from the image A''' dump (IMG/tp_common.py:75-79 and :137-166).
# tp_common.py is sha-pinned on the serving path and never grafted; nothing here edits it.
# ---------------------------------------------------------------------------------------

def _find_largest_divisor(n, max_div=8):
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def create_matmul_1d_decode_progcfg(m, k, n, num_cores, fused_activation=None, fp32_acc=True, grid_w=8):
    cols = min(grid_w, num_cores)
    rows = math.ceil(num_cores / cols)
    m_tiles = math.ceil(m / 32)
    k_tiles = math.ceil(k / 32)
    n_tiles = math.ceil(n / 32)
    per_core_k = _find_largest_divisor(k_tiles)
    per_core_n = math.ceil(n_tiles / (cols * rows))
    cap = 4 if fp32_acc else 8
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_tiles % i == 0 and i * sub_w <= cap)
    return types.SimpleNamespace(compute_with_storage_grid_size=(cols, rows), in0_block_w=per_core_k,
                                 out_subblock_h=sub_h, out_subblock_w=sub_w, per_core_M=m_tiles,
                                 per_core_N=per_core_n, fuse_batch=True, fused_activation=fused_activation,
                                 mcast_in0=True)


# Qwen3.8-27B at TP2 (the dims the probe dump's _init_tp_config reads).
QWEN_27B = dict(linear_num_key_heads=16, linear_key_head_dim=128, linear_num_value_heads=48,
                linear_value_head_dim=128, linear_conv_kernel_dim=4, linear_q_dim=2048, linear_k_dim=2048,
                linear_v_dim=6144, n_heads=24, n_kv_heads=4, head_dim=256, dim=5120, hidden_dim=17408,
                max_batch_size=8, num_devices=2)


def build_configs(source, environ, grid=(11, 10)):
    """Execute a (grafted) model_config.py's _init_tp_config over stubs; returns (args, logged)."""
    logged = []
    ttnn = types.SimpleNamespace(
        UnaryOpType=types.SimpleNamespace(SILU='silu'), CoreGrid=lambda x, y: (x, y),
        ShardStrategy=types.SimpleNamespace(HEIGHT='height'), ShardOrientation=types.SimpleNamespace(ROW_MAJOR='rm'),
        create_sharded_memory_config=lambda **options: ('sharded', tuple(sorted(options))))
    tpc = types.SimpleNamespace(
        TILE_SIZE=32, create_matmul_1d_decode_progcfg=create_matmul_1d_decode_progcfg,
        create_dram_sharded_mem_config=lambda k, n: ('dram-memcfg', k, n),
        create_dram_sharded_matmul_program_config=lambda m, k, n, num_cores=None: ('dram-progcfg', m, k, n),
        prefill_grid_default=lambda: (8, 10), prefill_tuning=lambda tp: 'tuning',
        create_prefill_matmul_program_config=lambda *args, **options: 'prefill',
        create_activation_shard_config=lambda k: ('activation', k))
    loguru = types.ModuleType('loguru')
    loguru.logger = types.SimpleNamespace(info=lambda *a: logged.append(('info',) + a),
                                          warning=lambda *a: logged.append(('warning',) + a))
    base = types.ModuleType('models.tt_transformers.tt.model_config')
    base.ModelArgs = type('ModelArgs', (), {})
    modules = {'loguru': loguru, 'ttnn': ttnn, 'models.tt_transformers.tt.model_config': base}
    for name in ('models', 'models.demos', 'models.demos.blackhole', 'models.demos.blackhole.qwen36',
                 'models.demos.blackhole.qwen36.tt', 'models.tt_transformers', 'models.tt_transformers.tt'):
        modules[name] = types.ModuleType(name)
    modules['models.demos.blackhole.qwen36.tt'].tp_common = tpc
    namespace = {}
    mesh = types.SimpleNamespace(shape=(1, 2),
                                 compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=grid[0], y=grid[1]))
    with patch.dict(sys.modules, modules), patch.dict('os.environ', environ, clear=True):
        exec(compile(source, 'model_config.py', 'exec'), namespace)
        args = object.__new__(namespace['Qwen36ModelArgs'])
        args.__dict__.update(QWEN_27B)
        args._init_tp_config(mesh)
    return args, logged


def decode_configs(args):
    return {name: vars(value) for name, value in vars(args).items() if name.endswith(('_decode_1d_progcfg',
                                                                                      '_decode_1d_progcfg_64'))}


def without_t1(source):
    """The graft as it was before section A2: the flag-gated block and its helper removed."""
    stripped = source.replace(patcher.VERIFY_T1_BLOCK, '').replace(patcher.VERIFY_T1_HELPER, '')
    if stripped == source or 'verify_t1' in stripped.lower():
        raise AssertionError('the T1 text did not strip cleanly')
    ast.parse(stripped)
    return stripped


GRAFTED = patcher.patch_model_config(MODEL_CONFIG)


class MatmulConfigGraftTests(unittest.TestCase):
    def test_flag_off_every_decode_config_is_exactly_what_the_graft_built_before(self):
        before, unused = build_configs(without_t1(GRAFTED), {})
        for environ in ({}, {'QWEN_FAST_VERIFY_T1': '0'}, {'QWEN_FAST_VERIFY_T1': 'true'}):
            with self.subTest(environ=environ):
                after, logged = build_configs(GRAFTED, environ)
                self.assertEqual(decode_configs(after), decode_configs(before))
                self.assertEqual(len(decode_configs(after)), 14, 'seven M = 1 and seven M = 64 configs')
                self.assertEqual(logged, [])

    def test_flag_on_changes_exactly_the_attn_qkv_and_gate_m64_configs(self):
        before, unused = build_configs(without_t1(GRAFTED), {})
        after, logged = build_configs(GRAFTED, {'QWEN_FAST_VERIFY_T1': '1'})
        old, new = decode_configs(before), decode_configs(after)
        changed = sorted(name for name in old if old[name] != new[name])
        self.assertEqual(changed, ['attn_qkv_decode_1d_progcfg_64', 'mlp_w1_decode_1d_progcfg_64'])
        self.assertEqual(logged, [('info', patcher.VERIFY_T1_MARKER + ' site=matmul_configs attn_qkv_64_cores=44 '
                                   'mlp_w1_64_cores=88 grid_w={}', 11)])

    def test_only_the_n_partition_and_the_subblock_move(self):
        before, unused = build_configs(without_t1(GRAFTED), {})
        after, logged = build_configs(GRAFTED, {'QWEN_FAST_VERIFY_T1': '1'})
        expected = {
            # name: ((grid, per_core_N, subblock) before, the same after)
            'attn_qkv_decode_1d_progcfg_64': (((8, 8), 4, (1, 4)), ((11, 4), 6, (1, 3))),
            'mlp_w1_decode_1d_progcfg_64': (((11, 4), 7, (2, 1)), ((11, 8), 4, (1, 4))),
        }
        for name, (shape_before, shape_after) in expected.items():
            old, new = vars(getattr(before, name)), vars(getattr(after, name))
            with self.subTest(config=name):
                for field in ('in0_block_w', 'per_core_M', 'fuse_batch', 'mcast_in0', 'fused_activation'):
                    self.assertEqual(old[field], new[field], field)
                self.assertEqual(new['in0_block_w'], 8, 'K = 5120: 160 tiles in blocks of 8, both ways')
                self.assertEqual((old['compute_with_storage_grid_size'], old['per_core_N'],
                                  (old['out_subblock_h'], old['out_subblock_w'])), shape_before)
                self.assertEqual((new['compute_with_storage_grid_size'], new['per_core_N'],
                                  (new['out_subblock_h'], new['out_subblock_w'])), shape_after)
        self.assertEqual(getattr(after, 'mlp_w1_decode_1d_progcfg_64').fused_activation, 'silu')
        self.assertIsNone(getattr(after, 'attn_qkv_decode_1d_progcfg_64').fused_activation)

    def test_skipping_matmul_configs_builds_every_config_as_before(self):
        before, unused = build_configs(without_t1(GRAFTED), {})
        for skip in ('matmul_configs', 'mask_once, matmul_configs ', 'matmul_configs,shard_argmax'):
            with self.subTest(skip=skip):
                after, logged = build_configs(GRAFTED, {'QWEN_FAST_VERIFY_T1': '1', 'QWEN_FAST_VERIFY_T1_SKIP': skip})
                self.assertEqual(decode_configs(after), decode_configs(before))
                self.assertEqual(logged, [])
        # naming only other cuts leaves #11 on
        after, logged = build_configs(GRAFTED, {'QWEN_FAST_VERIFY_T1': '1',
                                                'QWEN_FAST_VERIFY_T1_SKIP': 'mask_once,direct_carry,coalesce,shard_argmax'})
        self.assertEqual(len(logged), 1)
        self.assertNotEqual(decode_configs(after), decode_configs(before))

    def test_a_config_whose_k_reduction_would_change_is_refused_at_construction(self):
        """The helper's guard: a builder that moved in0_block_w (a T-class change) raises."""
        original = create_matmul_1d_decode_progcfg

        def moved(m, k, n, num_cores, **options):
            config = original(m, k, n, num_cores, **options)
            if num_cores == 88:
                config.in0_block_w = 4
            return config

        with patch(__name__ + '.create_matmul_1d_decode_progcfg', side_effect=moved):
            with self.assertRaisesRegex(ValueError, 'mlp_w1 M=64 config would change in0_block_w'):
                build_configs(GRAFTED, {'QWEN_FAST_VERIFY_T1': '1'})

    def test_a_grid_too_small_for_88_cores_keeps_the_configs_and_says_so(self):
        before, unused = build_configs(without_t1(GRAFTED), {}, grid=(11, 7))
        after, logged = build_configs(GRAFTED, {'QWEN_FAST_VERIFY_T1': '1'}, grid=(11, 7))
        self.assertEqual(decode_configs(after), decode_configs(before))
        self.assertEqual(logged, [('warning', patcher.VERIFY_T1_SKIPPED + ': {}',
                                   'worker grid 11x7 cannot hold 88 cores 11 wide')])

    def test_the_block_sits_after_the_seven_m64_configs_and_the_helper_before_the_class(self):
        body = GRAFTED[GRAFTED.index('def _init_tp_config'):]
        self.assertLess(body.index('self.gdn_out_decode_1d_progcfg_64 = '), body.index('QWEN_FAST_VERIFY_T1'))
        self.assertLess(body.index('QWEN_FAST_VERIFY_T1'), body.index('def _set_hf_params'))
        self.assertLess(GRAFTED.index('def _qwen_verify_t1_configs('), GRAFTED.index('class Qwen36ModelArgs('))
        self.assertEqual(GRAFTED.count('os.environ.get("QWEN_FAST_VERIFY_T1") == "1"'), 1)

    def test_tp_common_is_not_grafted_and_no_call_site_changes(self):
        """The fidelity and accumulation live in the call sites' compute configs; #11 touches
        only model_config.py, and tp_common.py (sha-pinned at attach) is never grafted."""
        from test_lever_n_m3native_patch import ATTENTION_TP, GDN_TP, MLP

        self.assertNotIn('tp_common.py', patcher.PATCHES)
        for name, source, patch_function in (('attention/tp.py', ATTENTION_TP, patcher.patch_attention_tp),
                                             ('gdn/tp.py', GDN_TP, patcher.patch_gdn_tp),
                                             ('mlp.py', MLP, patcher.patch_mlp)):
            with self.subTest(graft=name):
                self.assertNotIn('VERIFY_T1', patch_function(source))


# ---------------------------------------------------------------------------------------
# The K schedule of every output tile, and an emulation at the real shapes.
# ---------------------------------------------------------------------------------------

def tile_owners(config, n_tiles):
    """{output column tile: (core, subblock column index)} for a 1D mcast_in0 config."""
    cores = config.compute_with_storage_grid_size[0] * config.compute_with_storage_grid_size[1]
    owners = {}
    for core in range(cores):
        for local in range(config.per_core_N):
            column = core * config.per_core_N + local
            if column < n_tiles:
                if column in owners:
                    raise AssertionError('tile %d computed twice' % column)
                owners[column] = (core, local // config.out_subblock_w)
    return owners


def k_schedule(config, k_tiles):
    return [tuple(range(start, start + config.in0_block_w)) for start in range(0, k_tiles, config.in0_block_w)]


def emulate(x, weight, config, silu=False):
    """The 1D mcast_in0 matmul as the partition computes it: each core its own per_core_N
    columns (zero padding beyond N), each K block accumulated in fp32 one K element at a time
    (dest), each block's partial added into the running output (the packer's L1 accumulation),
    SiLU on the final fp32 value when fused, then bf16. Elementwise ops only, so an element's
    value depends on its own accumulation sequence alone - never on how many columns share a
    core."""
    rows, k = x.shape
    n = weight.shape[1]
    cores_used = math.ceil(n / (config.per_core_N * 32))
    width = config.per_core_N * 32
    padded = torch.zeros(k, cores_used * width, dtype=torch.float32)
    padded[:, :n] = weight.float()
    blocks = padded.reshape(k, cores_used, width).permute(1, 0, 2)  # [core, k, width]
    a = x.float()
    out = None
    step = config.in0_block_w * 32
    for start in range(0, k, step):
        dest = torch.zeros(cores_used, rows, width, dtype=torch.float32)
        for index in range(start, start + step):
            dest = dest + a[:, index].reshape(1, rows, 1) * blocks[:, index, :].reshape(cores_used, 1, width)
        out = dest if out is None else out + dest
    if silu:
        out = out * torch.sigmoid(out)
    return out.permute(1, 0, 2).reshape(rows, cores_used * width)[:, :n].to(torch.bfloat16)


class KReductionTests(unittest.TestCase):
    def configs(self):
        before, unused = build_configs(without_t1(GRAFTED), {})
        after, unused = build_configs(GRAFTED, {'QWEN_FAST_VERIFY_T1': '1'})
        return before, after

    def test_every_output_tile_keeps_its_k_schedule_and_one_owner(self):
        before, after = self.configs()
        for name, n in (('attn_qkv_decode_1d_progcfg_64', 7168), ('mlp_w1_decode_1d_progcfg_64', 8704)):
            old, new = getattr(before, name), getattr(after, name)
            with self.subTest(config=name):
                old_owners, new_owners = tile_owners(old, n // 32), tile_owners(new, n // 32)
                self.assertEqual(sorted(old_owners), list(range(n // 32)), 'every tile computed exactly once')
                self.assertEqual(sorted(new_owners), list(range(n // 32)))
                self.assertNotEqual(old_owners, new_owners, 'the partition does move')
                self.assertEqual(k_schedule(old, 160), k_schedule(new, 160))
                self.assertEqual(len(k_schedule(new, 160)), 20)

    # The real M (64 rows) and N (7168, 8704) with the real in0_block_w (8 tiles); K is cut from
    # 160 tiles to 16 (two K blocks, so the packer's L1 accumulation is exercised) because the
    # emulation steps one K element at a time and 160 tiles would take minutes on CPU. The K
    # schedule at the real K is test_every_output_tile_keeps_its_k_schedule_and_one_owner's.
    K = 16 * 32

    def test_emulated_outputs_are_bit_identical_at_the_real_rows_and_widths(self):
        before, after = self.configs()
        generator = torch.Generator().manual_seed(11)
        x = torch.randn(64, self.K, generator=generator).to(torch.bfloat16)
        for name, n, silu in (('attn_qkv_decode_1d_progcfg_64', 7168, False),
                              ('mlp_w1_decode_1d_progcfg_64', 8704, True)):
            weight = (torch.randn(self.K, n, generator=generator) * 0.05).to(torch.bfloat16)
            with self.subTest(config=name):
                old = emulate(x, weight, getattr(before, name), silu=silu)
                new = emulate(x, weight, getattr(after, name), silu=silu)
                self.assertEqual(tuple(new.shape), (64, n))
                self.assertTrue(torch.equal(old.view(torch.int16), new.view(torch.int16)))

    def test_negative_control_a_different_in0_block_w_is_not_bit_identical(self):
        """The emulation can see what #11 must not do: regroup the K reduction."""
        before, unused = self.configs()
        generator = torch.Generator().manual_seed(12)
        x = torch.randn(64, self.K, generator=generator).to(torch.bfloat16)
        weight = (torch.randn(self.K, 7168, generator=generator) * 0.05).to(torch.bfloat16)
        config = getattr(before, 'attn_qkv_decode_1d_progcfg_64')
        regrouped = types.SimpleNamespace(**dict(vars(config), in0_block_w=4))
        self.assertFalse(torch.equal(emulate(x, weight, config).view(torch.int16),
                                     emulate(x, weight, regrouped).view(torch.int16)))


# ---------------------------------------------------------------------------------------
# The gate and the arm.
# ---------------------------------------------------------------------------------------

class GateTests(unittest.TestCase):
    def report(self, environ, log, users=4):
        from lever_n_m3native_gate import flag_marker_report
        return flag_marker_report(environ, users, log)

    def test_the_gate_texts_are_the_image_module_and_the_grafts(self):
        import lever_n_m3native_gate as gate
        import verify_trace_t1 as image
        self.assertEqual((gate.VERIFY_T1_FLAG, gate.VERIFY_T1_AUDIT_FLAG, gate.VERIFY_T1_MARKER,
                          gate.VERIFY_T1_AUDIT_MARKER, gate.VERIFY_T1_AUDIT_MISMATCH),
                         (image.FLAG, image.AUDIT_FLAG, image.MARKER, image.AUDIT_MARKER, image.AUDIT_MISMATCH))
        self.assertEqual((patcher.VERIFY_T1_FLAG, patcher.VERIFY_T1_MARKER), (image.FLAG, image.MARKER))
        self.assertEqual((gate.VERIFY_T1_SKIP_FLAG, gate.VERIFY_T1_KEPT_SAMPLER, gate.VERIFY_T1_CUTS,
                          gate.VERIFY_T1_WAVE2_CUTS),
                         (image.SKIP_FLAG, image.KEPT_SAMPLER, image.CUTS, image.WAVE2_CUTS))
        self.assertEqual(patcher.VERIFY_T1_SKIP_FLAG, image.SKIP_FLAG)
        self.assertIn(patcher.VERIFY_T1_GRAFT_CUT, image.CUTS)
        self.assertNotIn(patcher.VERIFY_T1_GRAFT_CUT, image.WAVE2_CUTS)

    def test_the_flag_requires_the_engaged_marker(self):
        from lever_n_m3native_gate import VERIFY_T1_MARKER, required_flag_markers
        environ = {'QWEN_FAST_VERIFY_T1': '1'}
        self.assertEqual(required_flag_markers(environ, 4), {'QWEN_FAST_VERIFY_T1': [VERIFY_T1_MARKER]})
        self.assertEqual(required_flag_markers({'QWEN_FAST_VERIFY_T1': '0'}, 4), {})
        self.assertEqual(required_flag_markers({'QWEN_FAST_VERIFY_T1_AUDIT': '1'}, 4), {},
                         'the audit does nothing without the flag')
        self.assertEqual(self.report(environ, '[PINDIAG] draft weights lent', users=1)['missing'],
                         ['QWEN_FAST_VERIFY_T1: ' + VERIFY_T1_MARKER])
        self.assertEqual(self.report(environ, '[PINDIAG] draft weights lent')['missing'][0],
                         'QWEN_FAST_VERIFY_T1: ' + VERIFY_T1_MARKER)

    GRAFT = '2026-09-24 | INFO | [PINDIAG] verify t1 engaged site=matmul_configs attn_qkv_64_cores=44 ' \
            'mlp_w1_64_cores=88 grid_w=11'
    BATCHED = '[PINDIAG] gdn user_batched calls this captured forward: 48 of 48 GDN layers'
    WAVE2 = {'QWEN_FAST_VERIFY_T1': '1', 'QWEN_FAST_GDN_USER_BATCH': '1'}
    WAVE1 = {'QWEN_FAST_VERIFY_T1': '1', 'QWEN_FAST_VERIFY_T1_SKIP': 'mask_once,direct_carry,coalesce,shard_argmax'}

    @staticmethod
    def packed(**changes):
        counts = dict(audit=0, coalesce_fallback=0, coalesced=48, direct_carry=48, last_carry=48, mask_once=1,
                      shard_argmax=1)
        counts.update(changes)
        return '[PINDIAG] verify t1 engaged site=packed_verify ' + ' '.join(
            '%s=%d' % (name, counts[name]) for name in sorted(counts))

    def wave2_log(self, *lines):
        return chr(10).join((self.GRAFT, self.BATCHED) + lines)

    def test_a_wave_2_arm_passes_when_every_cut_engaged_and_the_report_names_the_sites(self):
        report = self.report(self.WAVE2, self.wave2_log(self.packed()))
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['verify_t1_sites'], ['matmul_configs', 'packed_verify'])
        self.assertEqual(report['verify_t1_packed'], [dict(audit=0, coalesce_fallback=0, coalesced=48, direct_carry=48,
                                                           last_carry=48, mask_once=1, shard_argmax=1)])
        self.assertIsNone(self.report({}, self.packed())['verify_t1_sites'])

    def test_the_graft_line_alone_fails_a_four_user_arm_unless_it_declares_wave_1(self):
        """A wave-2 case pointed at an image without wave 2 logs only the graft's line."""
        missing = self.report(self.WAVE2, self.wave2_log())['missing']
        self.assertEqual(len(missing), 1)
        self.assertIn('site=packed_verify', missing[0])
        self.assertIn('QWEN_FAST_VERIFY_T1_SKIP=mask_once,direct_carry,coalesce,shard_argmax', missing[0])
        report = self.report(self.WAVE1, self.GRAFT)
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['verify_t1_sites'], ['matmul_configs'])
        self.assertEqual(self.report({'QWEN_FAST_VERIFY_T1': '1'}, self.GRAFT, users=1)['missing'], [],
                         'no packed verify below four users')

    def test_a_cut_that_did_not_engage_fails(self):
        for changes, field in ((dict(mask_once=0), 'mask_once=0'), (dict(direct_carry=0), 'direct_carry=0'),
                               (dict(last_carry=0), 'last_carry=0'), (dict(coalesced=0, coalesce_fallback=48),
                                                                       'coalesced=0'),
                               (dict(coalesced=47), 'coalesced=47')):
            with self.subTest(changes=changes):
                missing = self.report(self.WAVE2, self.wave2_log(self.packed(**changes)))['missing']
                self.assertEqual(len(missing), 1)
                self.assertIn('capture 1 engaged', missing[0])
                self.assertIn(field + ' (expected', missing[0])
        # every capture is held to it, not only the first
        missing = self.report(self.WAVE2, self.wave2_log(self.packed(), self.packed(mask_once=0)))['missing']
        self.assertEqual(len(missing), 1)
        self.assertIn('capture 2 engaged mask_once=0', missing[0])

    def test_the_pinned_sampler_kept_fails_unless_shard_argmax_is_skipped(self):
        kept = '[PINDIAG] verify t1 kept the pinned sampler: penalties or log-probabilities are active'
        missing = self.report(self.WAVE2, self.wave2_log(self.packed(shard_argmax=0), kept))['missing']
        self.assertEqual(len(missing), 2)
        self.assertIn('shard_argmax=0 (expected 1)', missing[0])
        self.assertIn('per-shard argmax engaged (' + kept, missing[1])
        skip = dict(self.WAVE2, QWEN_FAST_VERIFY_T1_SKIP='shard_argmax')
        self.assertEqual(self.report(skip, self.wave2_log(self.packed(shard_argmax=0)))['missing'], [])

    def test_the_per_layer_cuts_need_the_user_batch(self):
        environ = {'QWEN_FAST_VERIFY_T1': '1'}
        missing = self.report(environ, self.packed(direct_carry=0, last_carry=0, coalesced=0))['missing']
        self.assertEqual(missing, ['QWEN_FAST_VERIFY_T1: direct_carry,coalesce engage only under '
                                   'QWEN_FAST_GDN_USER_BATCH=1 (or skip them)'])
        skip = dict(environ, QWEN_FAST_VERIFY_T1_SKIP='direct_carry,coalesce')
        self.assertEqual(self.report(skip, self.packed(direct_carry=0, last_carry=0, coalesced=0))['missing'], [])

    def test_a_skipped_cut_must_not_engage_and_the_skip_names_only_cuts(self):
        skip = dict(self.WAVE2, QWEN_FAST_VERIFY_T1_SKIP='last_carry')
        self.assertEqual(self.report(skip, self.wave2_log(self.packed(last_carry=0)))['missing'], [])
        missing = self.report(skip, self.wave2_log(self.packed()))['missing']
        self.assertEqual(len(missing), 1)
        self.assertIn('last_carry=48 (expected 0)', missing[0])
        bad = dict(self.WAVE2, QWEN_FAST_VERIFY_T1_SKIP='last_carry,masks')
        missing = self.report(bad, self.wave2_log(self.packed(last_carry=0)))['missing']
        self.assertEqual(len(missing), 1)
        self.assertTrue(missing[0].startswith('QWEN_FAST_VERIFY_T1_SKIP: names only cuts (masks is none of'))

    def test_the_audit_needs_its_first_line_at_four_users_and_fails_on_a_mismatch(self):
        from lever_n_m3native_gate import VERIFY_T1_AUDIT_MARKER, VERIFY_T1_MARKER, required_flag_markers
        environ = {'QWEN_FAST_VERIFY_T1': '1', 'QWEN_FAST_VERIFY_T1_AUDIT': '1'}
        self.assertEqual(required_flag_markers(environ, 4),
                         {'QWEN_FAST_VERIFY_T1': [VERIFY_T1_MARKER],
                          'QWEN_FAST_VERIFY_T1_AUDIT': [VERIFY_T1_AUDIT_MARKER + ' 1 exact=True']})
        self.assertEqual(required_flag_markers(environ, 1), {'QWEN_FAST_VERIFY_T1': [VERIFY_T1_MARKER]})
        environ = dict(environ, QWEN_FAST_GDN_USER_BATCH='1')
        engaged = chr(10).join([self.BATCHED, self.packed(audit=1)])
        good = chr(10).join([engaged, VERIFY_T1_AUDIT_MARKER + ' 1 exact=True rows=64',
                             VERIFY_T1_AUDIT_MARKER + ' 2 exact=True rows=128'])
        self.assertEqual(self.report(environ, good)['missing'], [])
        self.assertEqual(self.report(environ, engaged)['missing'],
                         ['QWEN_FAST_VERIFY_T1_AUDIT: ' + VERIFY_T1_AUDIT_MARKER + ' 1 exact=True'])
        bad = good + chr(10) + '[PINDIAG] verify t1 audit mismatch round=3 rows=[5] shard=[7] sampler=[9]'
        missing = self.report(environ, bad)['missing']
        self.assertEqual(len(missing), 1)
        self.assertTrue(missing[0].startswith('QWEN_FAST_VERIFY_T1_AUDIT: no mismatch ([PINDIAG] verify t1 audit '
                                              'mismatch round=3'))


def arm_lines():
    return ARM.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10)).split(chr(10))


class ArmTests(unittest.TestCase):
    LINE = '${M3NATIVE_VERIFY_T1:+-e QWEN_FAST_VERIFY_T1=1}'
    AUDIT = '${M3NATIVE_VERIFY_T1_AUDIT:+-e QWEN_FAST_VERIFY_T1_AUDIT=1}'
    SKIP = '${M3NATIVE_VERIFY_T1_SKIP:+-e QWEN_FAST_VERIFY_T1_SKIP=$M3NATIVE_VERIFY_T1_SKIP}'

    def test_the_flag_crosses_on_its_own_line_right_after_the_round_b1_lines(self):
        lines = arm_lines()
        text = chr(10).join(lines)
        self.assertEqual(text.count(self.LINE), 1)
        index = next(number for number, line in enumerate(lines) if self.LINE in line)
        self.assertEqual(lines[index - 1].strip(), '${M3NATIVE_ROUND_B1_AUDIT:+-e QWEN_FAST_ROUND_B1_AUDIT=1} ' + chr(92))
        self.assertEqual(lines[index - 2].strip(), '${M3NATIVE_ROUND_B1:+-e QWEN_FAST_ROUND_B1=1} ' + chr(92))
        self.assertEqual(lines[index].strip(), self.LINE + ' ' + chr(92), 'nothing else on the continued line')
        self.assertEqual(lines[index + 1].strip(), self.AUDIT + ' ' + chr(92))
        self.assertEqual(lines[index + 2].strip(), self.SKIP + ' ' + chr(92))
        self.assertLess(text.index(self.SKIP), text.index('--entrypoint python3'))

    def test_no_comment_sits_inside_the_continued_docker_command(self):
        lines = arm_lines()
        start = next(number for number, line in enumerate(lines) if line.startswith('timeout -k 30 2200 docker run'))
        end = next(number for number in range(start, len(lines)) if not lines[number].rstrip().endswith(chr(92)))
        self.assertFalse([line for line in lines[start:end + 1] if line.strip().startswith('#')])

    def test_the_modules_that_read_it_read_that_name(self):
        image = (HERE / 'verify_trace_t1.py').read_text(encoding='utf-8')
        self.assertIn("os.environ.get('QWEN_FAST_VERIFY_T1') == '1'", image)
        self.assertIn("os.environ.get('QWEN_FAST_VERIFY_T1_AUDIT') == '1'", image)
        self.assertIn("os.environ.get('QWEN_FAST_VERIFY_T1_SKIP', '')", image)
        self.assertIn('os.environ.get("QWEN_FAST_VERIFY_T1_SKIP", "")', GRAFTED)
        self.assertIn('os.environ.get("QWEN_FAST_VERIFY_T1") == "1"', GRAFTED)
        for module in ('packed_verifier.py', 'gdn_device_loop_state.py', 'gdn_user_batch.py'):
            with self.subTest(module=module):
                self.assertIn('verify_t', (HERE / module).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
