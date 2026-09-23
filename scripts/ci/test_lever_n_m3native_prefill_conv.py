"""Lever #2 graft: gdn/tp.py forward_prefill under QWEN_FAST_GDN_PREFILL_CONV, the one mount table,
the arm, the workflow staging and the gate.

The fixture is the image's own forward_prefill (graft artifact of run 35816715775, gdn/tp.py.orig
lines 466-653) and the two module helpers it calls, verbatim, so the anchors the graft matches
are the real text. The patched method is EXECUTED against a recording fake ttnn: with the flag
unset it must issue exactly the original's calls, in order, with the same arguments.
"""

import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch  # noqa: F401 - imported once, before any sys.modules patching (a re-import segfaults)

import gdn_prefill_conv_exact as pcx
import lever_n_m3native_patch as patcher

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
ARM = HERE / 'lever_n_m3native_run_arm.sh'
WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-lever-n-m3native-gate.yml'
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'

GDN_TP_PREFILL = '''import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
    create_chunk_masks_seq,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.tt_transformers.tt.ccl import tt_all_reduce


def _softplus_add(a, bias):
    """g-gate: softplus(a + bias) fused into one op (softplus as a post-activation on the add)."""
    return ttnn.add(a, bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])


def _silu_mul(x, z, memory_config):
    """out-gate: x * silu(z). NOT fused into one op: fusing silu via input_tensor_b_activations
    overflows to NaN in the real layer for large-magnitude z (op-level PCC hid it — small inputs)."""
    return ttnn.multiply(x, ttnn.silu(z, memory_config=memory_config), memory_config=memory_config)


class TPGatedDeltaNet:
    """Standalone TP GDN decode (per-device value-head recurrence + all-reduce)."""

    def reset_state(self):
        return None

    def forward_prefill(self, x, chunk_size=128, valid_len=None, capture_state=False, return_state=False):
        """Causal chunk-prefill from scratch. x [1,1,T,dim]: K-sharded (dim/tp per device) when the
        fused in-proj AG-matmul path is active (``_fuse_agmm`` and T>TILE — the norm skips its
        post-AG); replicated otherwise. Output reduce-scattered.

        valid_len: real token count (rest is padding). capture_state: save rec/conv state for decode.
        return_state: when True (per-user batched prefill), return
        ``(output, final_state, conv_new_state)`` for one user's from-scratch B=1
        pass and skip all self.* writeback; the caller stitches per-user states via
        assemble_batched_state(). Single-sequence behavior is unchanged when False.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        T = x.shape[1]
        # Pass the RAW valid_len (may be None) to the conv-FIR / seq kernels below — NOT a
        # `valid_len or T` coercion. A full chunk (valid_len is None) must take the kernels'
        # valid_len-None path (a static last-(K-1) slice for the conv state), which is trace-safe;
        # the valid_len-set path builds a one-hot via ttnn.from_torch (a host write) that TT_FATALs
        # ("Writes are not supported during trace capture") inside the captured chunk-outer trace.
        # Masked buckets still pass a real valid_len (< T) so their exact masking is unchanged, and
        # for a full chunk the None slice and the valid_len==T one-hot select the identical rows.

        # Cross-chunk carry (chunk-outer prefill): when _stable_state, the recurrent + conv
        # state continue from the persistent buffers (zeroed at sequence start by
        # reset_state_inplace, so a from-scratch single pass reads zeros == None). The demo
        # path (_stable_state False) is unchanged: no carry, reassign state.
        # Per-user prefill (return_state) is always from scratch: must not carry the shared
        # batched buffer (other users' state) as its initial recurrent/conv state.
        carry = self._stable_state and not return_state
        if carry and self.conv_carry is None:
            self.reset_state()

        # Prefill qkvzab in L1: keeps proj + q/k/v/z/a/b resident for conv+gate prep.
        qkv, z, a, b = self._project_qkvzab(x, T, out_mc=ttnn.L1_MEMORY_CONFIG)

        # FIR conv1d; conv_state = previous chunk's last K-1 inputs (None/zero from scratch)
        _cstate = self.conv_carry if carry else None
        if self._gdn_conv1d and valid_len is None:
            # Native depthwise ttnn.conv1d (masked buckets keep the MAC FIR: valid_len new_state differs)
            conv, conv_new_state = self._conv1d_prefill(qkv, T, _cstate)
        else:
            conv, conv_new_state = _causal_conv1d_fir(
                qkv,
                None,
                None,
                self.K,
                self.mesh,
                # Conv in L1 (output freed before chunk kernel; new_state lands in DRAM internally)
                memory_config=ttnn.L1_MEMORY_CONFIG,
                conv_state=_cstate,
                weight_taps=tw["conv_taps"],
                bias_dev=None,
                valid_len=valid_len,
            )
        ttnn.deallocate(qkv)

        # q/k/v/beta/g stay DRAM — alive across chunk kernel; L1 crashes it.
        kd = self.key_dim_tp
        if self._gdn_flat_qkv:
            # Flat q/k/v: adapter splits heads inside untilize
            q = ttnn.slice(conv, (0, 0, 0), (1, T, kd))
            k = ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd))
            v = ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp))
            _qkv_head_dims = (Nk, Dk, Nv, Dv)
        else:
            q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
            k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd)), (1, T, Nk, Dk))
            v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp)), (1, T, Nv, Dv))
            _qkv_head_dims = None
        ttnn.deallocate(conv)
        # GQA late-expand: adapter L2-norms at Nk, expands to Nv after
        beta = ttnn.reshape(ttnn.sigmoid(b), (1, T, Nv))
        ttnn.deallocate(b)
        g = ttnn.reshape(ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"])), (1, T, Nv))
        ttnn.deallocate(a)

        # Fused chunk_gated_delta_rule; also used for masked valid_len.
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import (
            chunk_gated_delta_rule_fused_adapter,
            fused_chunk_enabled,
        )

        _use_fused = fused_chunk_enabled()
        _delta_fn = chunk_gated_delta_rule_fused_adapter if _use_fused else chunk_gated_delta_rule_seq_adapter
        # const_tiles only applies to the fused op; the seq adapter has no such param.
        _extra = {"const_tiles": self._fused_const_tiles} if _use_fused else {}
        o, final_state = _delta_fn(
            q,
            k,
            v,
            beta,
            g,
            chunk_size=chunk_size,
            scale=self.scale,
            initial_state=self.rec_state if carry else None,
            device=self.mesh,
            cached_masks=self.chunk_seq_masks,
            valid_len=valid_len,
            qkv_head_dims=_qkv_head_dims,
            return_o_bh=self._gdn_fuse_out,
            **_extra,
        )
        B, D = 1, self.qkv_dim_tp
        captured = None
        if return_state:
            # Per-user prefill: return this user's state for assemble_batched_state to stitch
            # into the batched buffers. No self.* writeback; tensors are not deallocated here.
            captured = (final_state, conv_new_state)
        else:
            # ---- Carry recurrent + conv state for the NEXT chunk (chunk-outer prefill). ----
            # In place (ttnn.copy) when _stable_state so the addresses the prefill/decode traces
            # baked in stay valid across execute_trace replays and across sequences.
            if carry:
                ttnn.copy(final_state, self.rec_state)
                ttnn.deallocate(final_state)
                ttnn.copy(conv_new_state, self.conv_carry)  # [1, K-1, D] last-K-1 conv inputs
            else:
                self.rec_state = final_state
            # ---- Finalize the decode conv window (last chunk / short prompt). ----
            # conv_states[1..K-1] = the last K-1 real conv inputs; [0] is the (shifted-out) zero.
            # Harmless to refresh every chunk — the last chunk's values are the ones decode reads.
            if capture_state:
                if self.conv_states is None:
                    self.reset_state()
                if self._zero_conv0 is not None:
                    ttnn.copy(self._zero_conv0, self.conv_states[0])
                else:
                    zero = ttnn.from_torch(
                        torch.zeros(1, B, D, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                    )
                    ttnn.copy(zero, self.conv_states[0])
                    ttnn.deallocate(zero)
                for j in range(self.K - 1):
                    src = ttnn.reshape(ttnn.slice(conv_new_state, (0, j, 0), (1, j + 1, D)), (1, B, D))
                    ttnn.copy(src, self.conv_states[j + 1])
            ttnn.deallocate(conv_new_state)
        # Gated RMSNorm + SiLU(z); norm/flatten in L1, gated output in DRAM for out-proj
        _L1 = ttnn.L1_MEMORY_CONFIG
        if self._gdn_fuse_out:
            # Fuse adapter relayout with per-head rms_norm + head-flatten.
            # TILE-native head->token relayout (transpose + fold), dropping the
            # TILE->ROW_MAJOR->TILE round-trip. o is head-major (1,Nv,T,Dv).
            n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
            ttnn.deallocate(o)
            n = ttnn.reshape(n, (1, Nv, T, Dv))
            # Fused head->token relayout: [1,Nv,T,Dv] -> [1,1,T,Nv*Dv].
            n = ttnn.experimental.nlp_concat_heads(n, memory_config=_L1)
            out_f = ttnn.reshape(n, (1, T, self.value_dim_tp))
        else:
            out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
            ttnn.deallocate(o)
            out_f = ttnn.reshape(out_n, (1, T, self.value_dim_tp), memory_config=_L1)
            ttnn.deallocate(out_n)
        gated = _silu_mul(out_f, z, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        # Prefill: fused out-proj matmul + reduce-scatter (matmul_reduce_scatter_async), flag-gated.
        if self._fuse_out_mmrs_prefill:
            x_out = ttnn.reshape(gated, (1, 1, T, gated.shape[-1]))
            # fp32 output is load-bearing: o_proj is row-parallel, so the RS SUMS 4 per-device partials
            # across devices — bf16 there tanks PCC to ~0.69 even at ISL 2048 (test_oproj_dtype_isl). Keep fp32.
            out = tpc.matmul_reduce_scatter_prefill(
                x_out, tw["out"], self.tt_ccl, self.cfg, self.args.ccl_topology(), self.args.num_devices, ttnn.float32
            )
            ttnn.deallocate(gated)
            if return_state:
                return out, captured[0], captured[1]
            return out
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        out = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if return_state:
            return out, captured[0], captured[1]
        return out

    def forward_prefill_collect(self, x, chunk_size=128, valid_len=None):
        return None
'''

FLAG = patcher.PREFILL_CONV_FLAG
AUDIT = patcher.PREFILL_CONV_AUDIT_FLAG


def text(path):
    return path.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))


# ---------------------------------------------------------------------------------------------
# A recording fake of everything forward_prefill touches.
# ---------------------------------------------------------------------------------------------

class T:
    def __init__(self, label, shape):
        self.label, self.shape = label, tuple(shape)

    def __repr__(self):
        return self.label


class Run:
    def __init__(self, reason=None, exact=True):
        self.events, self.logs = [], []
        self.op_imports = []      # attempts to import the op module while it is NOT mounted
        self.count = 0
        self.reason, self.exact = reason, exact

    def tensor(self, shape):
        self.count += 1
        return T('t%d' % self.count, shape)

    def describe(self, value):
        if isinstance(value, T):
            return value.label
        if isinstance(value, (list, tuple)):
            return tuple(self.describe(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((key, self.describe(item)) for key, item in value.items()))
        if isinstance(value, types.ModuleType) or callable(value):
            return getattr(value, '__name__', type(value).__name__)
        return value

    def record(self, name, args, kwargs):
        self.events.append((name, self.describe(args), self.describe(kwargs)))

    def op(self, name, shape=None):
        def call(*args, **kwargs):
            self.record(name, args, kwargs)
            if shape is None:
                return None
            return self.tensor(shape(*args, **kwargs))
        call.__name__ = name
        return call


def first_shape(*args, **kwargs):
    return args[0].shape


def build_modules(run, T_rows, C, kd, Nv, Dv, op_mounted=True):
    """The fake import tree. op_mounted False is the arm without M3NATIVE_GDN_PREFILL_CONV: the op
    module is not in the tree, and any attempt to import it is recorded in run.op_imports and
    fails the way the container's import would (ImportError)."""
    ttnn = types.ModuleType('ttnn')
    ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG = 'L1', 'DRAM'
    ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.float32 = 'TILE', 'bf16', 'f32'
    ttnn.reshape = run.op('ttnn.reshape', lambda t, shape, **k: shape)
    ttnn.slice = run.op('ttnn.slice', lambda t, start, end, **k: tuple(e - s for s, e in zip(start, end)))
    ttnn.deallocate = run.op('ttnn.deallocate')
    ttnn.copy = run.op('ttnn.copy')
    ttnn.sigmoid = run.op('ttnn.sigmoid', first_shape)
    ttnn.multiply = run.op('ttnn.multiply', lambda a, b, **k: b.shape)
    ttnn.add = run.op('ttnn.add', first_shape)
    ttnn.silu = run.op('ttnn.silu', first_shape)
    ttnn.UnaryWithParam = lambda *a: ('UnaryWithParam',) + a
    ttnn.UnaryOpType = types.SimpleNamespace(SOFTPLUS='SOFTPLUS')
    ttnn.rms_norm = run.op('ttnn.rms_norm', first_shape)
    ttnn.from_torch = run.op('ttnn.from_torch', lambda *a, **k: (1, 1, C))
    ttnn.ReplicateTensorToMesh = run.op('ttnn.ReplicateTensorToMesh', lambda *a, **k: (1,))
    ttnn.experimental = types.SimpleNamespace(nlp_concat_heads=run.op('ttnn.experimental.nlp_concat_heads', first_shape))

    tpc = types.ModuleType('tp_common')
    tpc.TILE_SIZE = 32
    tpc.matmul_reduce_scatter_prefill = run.op('tpc.matmul_reduce_scatter_prefill', lambda x, *a, **k: (1, 1, T_rows, 5120))

    def _causal_conv1d_fir(*args, **kwargs):
        run.record('_causal_conv1d_fir', args, kwargs)
        return run.tensor((1, T_rows, C)), run.tensor((1, 3, C))

    def fused_adapter(*args, **kwargs):
        run.record('chunk_gated_delta_rule_fused_adapter', args, kwargs)
        return run.tensor((1, Nv, T_rows, Dv)), run.tensor((1, Nv, 128, Dv))

    def seq_adapter(*args, **kwargs):
        run.record('chunk_gated_delta_rule_seq_adapter', args, kwargs)
        return run.tensor((1, Nv, T_rows, Dv)), run.tensor((1, Nv, 128, Dv))

    op_module = types.ModuleType('gdn_prefill_conv_exact')

    def unsupported(*args, **kwargs):
        run.record('pcx.unsupported', args, kwargs)
        return run.reason

    def conv_op(*args, **kwargs):
        run.record('pcx.gdn_prefill_conv_exact', args, kwargs)
        return (run.tensor((1, T_rows, kd)), run.tensor((1, T_rows, kd)), run.tensor((1, T_rows, C - 2 * kd)),
                run.tensor((1, 3, C)))

    def audit(*args, **kwargs):
        run.record('pcx.audit_against_fir', args, kwargs)
        return dict(exact=run.exact, mismatches={} if run.exact else {'q/chip0': 3})

    op_module.unsupported, op_module.gdn_prefill_conv_exact, op_module.audit_against_fir = unsupported, conv_op, audit

    logger = types.SimpleNamespace(
        info=lambda message, *a: run.logs.append(('info', message.format(*a))),
        warning=lambda message, *a: run.logs.append(('warning', message.format(*a))))
    loguru = types.ModuleType('loguru')
    loguru.logger = logger

    def package(name, **attributes):
        module = types.ModuleType(name)
        module.__path__ = []
        for key, value in attributes.items():
            setattr(module, key, value)
        return module

    fused = types.ModuleType('fused_chunk')
    fused.chunk_gated_delta_rule_fused_adapter = fused_adapter
    fused.fused_chunk_enabled = lambda: True
    seq = types.ModuleType('ttnn_delta_rule_seq')
    seq.chunk_gated_delta_rule_seq_adapter = seq_adapter
    seq.create_chunk_masks_seq = lambda *a, **k: None
    deltanet = types.ModuleType('ttnn_gated_deltanet')
    deltanet._causal_conv1d_fir = _causal_conv1d_fir
    ccl = types.ModuleType('ccl')
    ccl.tt_all_reduce = run.op('tt_all_reduce', first_shape)
    if op_mounted:
        gdn = package('models.demos.blackhole.qwen36.tt.gdn', fused_chunk=fused, gdn_prefill_conv_exact=op_module)
    else:
        gdn = package('models.demos.blackhole.qwen36.tt.gdn', fused_chunk=fused)

        def missing(name):
            if name == 'gdn_prefill_conv_exact':
                run.op_imports.append(name)
            raise AttributeError(name)
        gdn.__getattr__ = missing
    modules = {
        'ttnn': ttnn, 'loguru': loguru,
        'models': package('models'), 'models.demos': package('models.demos'),
        'models.demos.blackhole': package('models.demos.blackhole'),
        'models.demos.blackhole.qwen36': package('models.demos.blackhole.qwen36'),
        'models.demos.blackhole.qwen36.tt': package('models.demos.blackhole.qwen36.tt', tp_common=tpc, gdn=gdn),
        'models.demos.blackhole.qwen36.tt.tp_common': tpc,
        'models.demos.blackhole.qwen36.tt.gdn': gdn,
        'models.demos.blackhole.qwen36.tt.gdn.fused_chunk': fused,
        'models.demos.blackhole.qwen36.tt.gdn.gdn_prefill_conv_exact': op_module,
        'models.experimental': package('models.experimental'),
        'models.experimental.gated_attention_gated_deltanet': package('models.experimental.gated_attention_gated_deltanet'),
        'models.experimental.gated_attention_gated_deltanet.tt': package('models.experimental.gated_attention_gated_deltanet.tt'),
        'models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq': seq,
        'models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet': deltanet,
        'models.tt_transformers': package('models.tt_transformers'),
        'models.tt_transformers.tt': package('models.tt_transformers.tt'),
        'models.tt_transformers.tt.ccl': ccl,
    }
    if not op_mounted:
        del modules['models.demos.blackhole.qwen36.tt.gdn.gdn_prefill_conv_exact']
    return modules


def make_layer(cls, run, *, T_rows, conv1d, stable, flat, fuse_out, C=5120, kd=1024):
    layer = cls.__new__(cls)
    layer.mesh = 'mesh'
    layer.args = types.SimpleNamespace(ccl_topology=lambda: 'linear', num_devices=2)
    layer.tt_ccl = 'ccl'
    layer.cfg = 'hifi2'
    layer.tw = dict(conv_taps=[run.tensor((1, 1, C)) for _ in range(4)], neg_exp_A=run.tensor((1, 1, 24)),
                    dt_bias=run.tensor((1, 1, 24)), norm_w=run.tensor((1, 1, 128)), out=run.tensor((3072, 5120)))
    layer.Nk, layer.Nv, layer.Dk, layer.Dv = 8, 24, 128, 128
    layer.key_dim_tp, layer.qkv_dim_tp, layer.value_dim_tp = kd, C, 3072
    layer.K, layer.scale = 4, 128 ** -0.5
    layer._gdn_conv1d, layer._stable_state, layer._gdn_flat_qkv = conv1d, stable, flat
    layer._gdn_fuse_out, layer._fuse_out_mmrs_prefill = True, fuse_out
    layer.conv_carry, layer.rec_state = run.tensor((1, 3, C)), run.tensor((1, 24, 128, 128))
    layer.chunk_seq_masks, layer._fused_const_tiles = 'masks', 'consts'
    layer.conv_states = [run.tensor((1, 32, C)) for _ in range(4)]
    layer._zero_conv0 = run.tensor((1, 32, C))

    def project(x, S, out_mc=None):
        run.record('self._project_qkvzab', (x, S), dict(out_mc=out_mc))
        return run.tensor((1, S, C)), run.tensor((1, S, 3072)), run.tensor((1, S, 24)), run.tensor((1, S, 24))

    def conv1d_prefill(qkv, T_, state):
        run.record('self._conv1d_prefill', (qkv, T_, state), {})
        return run.tensor((1, T_, C)), run.tensor((1, 3, C))

    def row_proj(x, weight):
        run.record('self._row_proj', (x, weight), {})
        return run.tensor((1, T_rows, 5120))

    layer._project_qkvzab, layer._conv1d_prefill, layer._row_proj = project, conv1d_prefill, row_proj
    return layer


def load(source, modules):
    namespace = {'__name__': 'gdn_tp_fixture'}
    with patch.dict(sys.modules, modules):
        exec(compile(source, 'gdn_tp_fixture', 'exec'), namespace)
    return namespace


def drive(source, environ, *, T_rows=2048, valid_len=1288, conv1d=True, stable=True, flat=True, fuse_out=False,
          capture=False, return_state=False, layers=1, forwards=1, reason=None, exact=True, op_mounted=True):
    """Run forward_prefill `forwards` times over `layers` layer objects; return (run, results).
    op_mounted False leaves the op module out of the import tree (the flag-off arm mounts nothing)."""
    run = Run(reason=reason, exact=exact)
    modules = build_modules(run, T_rows, 5120, 1024, 24, 128, op_mounted=op_mounted)
    namespace = load(source, modules)
    cls = namespace['TPGatedDeltaNet']
    stack = [make_layer(cls, run, T_rows=T_rows, conv1d=conv1d, stable=stable, flat=flat, fuse_out=fuse_out)
             for _ in range(layers)]
    results = []
    clean = {key: value for key, value in os.environ.items() if key not in (FLAG, AUDIT)}
    clean.update(environ)
    with patch.dict(sys.modules, modules), patch.dict(os.environ, clean, clear=True):
        for _ in range(forwards):
            for layer in stack:
                x = run.tensor((1, 1, T_rows, 2560))
                results.append(layer.forward_prefill(x, chunk_size=128, valid_len=valid_len,
                                                     capture_state=capture, return_state=return_state))
    return run, results


ORIGINAL = GDN_TP_PREFILL
PATCHED = patcher.patch_gdn_tp_prefill_conv(ORIGINAL)


# ---------------------------------------------------------------------------------------------
# The patch itself.
# ---------------------------------------------------------------------------------------------

class PatchShapeTests(unittest.TestCase):
    def test_the_fixture_is_the_image_source_shape(self):
        self.assertEqual(ORIGINAL.count('\n\nclass TPGatedDeltaNet:\n'), 1)
        for anchor in (patcher.PREFILL_CONV_A_OLD, patcher.PREFILL_CONV_B_OLD, patcher.PREFILL_CONV_C_OLD):
            self.assertEqual(ORIGINAL.count(anchor), 1)
        ast.parse(ORIGINAL)

    def test_three_edits_and_the_helpers_land_once(self):
        for new in (patcher.PREFILL_CONV_A_NEW, patcher.PREFILL_CONV_B_NEW, patcher.PREFILL_CONV_C_NEW):
            self.assertEqual(PATCHED.count(new), 1)
        self.assertEqual(PATCHED.count('def _qwen_prefill_conv_on('), 1)
        self.assertEqual(PATCHED.count('def _qwen_prefill_conv('), 1)
        self.assertLess(PATCHED.index('def _qwen_prefill_conv('), PATCHED.index('class TPGatedDeltaNet:'))
        ast.parse(PATCHED)

    def test_a_second_pass_refuses(self):
        with self.assertRaisesRegex(ValueError, 'already grafted'):
            patcher.patch_gdn_tp_prefill_conv(PATCHED)

    def test_each_anchor_is_required(self):
        drifts = ((patcher.PREFILL_CONV_A_OLD, 'valid_len is None', 'valid_len is  None'),
                  (patcher.PREFILL_CONV_B_OLD, 'kd = self', 'kd  = self'),
                  (patcher.PREFILL_CONV_C_OLD, 'deallocate(conv)', 'deallocate( conv)'),
                  ('\n\nclass TPGatedDeltaNet:\n', 'TPGatedDeltaNet:', 'TPGatedDeltaNet :'))
        for old, before, after in drifts:
            with self.subTest(anchor=old[:40]):
                drifted = ORIGINAL.replace(old, old.replace(before, after, 1))
                self.assertNotEqual(drifted, ORIGINAL)
                ast.parse(drifted)
                with self.assertRaises(ValueError):
                    patcher.patch_gdn_tp_prefill_conv(drifted)

    def test_no_top_level_import_of_the_op(self):
        tree = ast.parse(PATCHED)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.assertNotIn('gdn_prefill_conv_exact', ast.unparse(node))

    def test_the_fir_call_and_the_slices_are_untouched(self):
        start = ORIGINAL.index('            conv, conv_new_state = _causal_conv1d_fir(')
        end = ORIGINAL.index('        ttnn.deallocate(qkv)\n', start)
        fir = ORIGINAL[start:end]
        self.assertEqual(PATCHED.count(fir), 1)
        flat = ORIGINAL[ORIGINAL.index('            # Flat q/k/v: adapter splits heads inside untilize'):
                        ORIGINAL.index('            _qkv_head_dims = None\n')]
        self.assertEqual(PATCHED.count(flat), 1)

    def test_the_ordered_calls_of_forward_prefill_are_the_originals_plus_the_two_helpers(self):
        def calls(source):
            tree = ast.parse(source)
            method = [node for cls in tree.body if isinstance(cls, ast.ClassDef) for node in cls.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'forward_prefill'][0]
            found = sorted((node.lineno, node.col_offset, ast.unparse(node.func)) for node in ast.walk(method)
                           if isinstance(node, ast.Call))
            return [name for _, _, name in found]
        original, patched = calls(ORIGINAL), calls(PATCHED)
        self.assertEqual([name for name in patched if not name.startswith('_qwen_prefill_conv')], original)
        self.assertEqual(sorted(set(patched) - set(original)), ['_qwen_prefill_conv', '_qwen_prefill_conv_on'])

    def test_the_table_composes_the_decode_graft_then_lever_2(self):
        self.assertIs(patcher.PATCHES['gdn/tp.py'], patcher.patch_gdn_tp_full)
        self.assertIs(patcher.SOURCES['gdn/tp.py'][1], patcher.patch_gdn_tp_full)
        import test_lever_n_m3native_patch as m3
        decode_only = patcher.patch_gdn_tp(m3.GDN_TP)
        with self.assertRaises(ValueError):
            patcher.patch_gdn_tp_full(m3.GDN_TP)   # the decode fixture's forward_prefill is a stub
        self.assertNotIn(FLAG, decode_only)

    def test_the_helper_reads_the_flag_names_the_gate_and_the_arm_use(self):
        import lever_n_m3native_gate as gate
        self.assertEqual(gate.PREFILL_CONV_FLAG, FLAG)
        self.assertEqual(gate.PREFILL_CONV_AUDIT_FLAG, AUDIT)
        self.assertEqual(gate.PREFILL_CONV_MARKER, patcher.MARKER_PREFILL_CONV)
        self.assertEqual(gate.PREFILL_CONV_FALLBACK, patcher.MARKER_PREFILL_CONV_FALLBACK)
        self.assertEqual(gate.PREFILL_CONV_AUDIT_MARKER, patcher.MARKER_PREFILL_CONV_AUDIT)
        self.assertIn('os.environ.get("%s") != "1"' % FLAG, PATCHED)
        self.assertIn('os.environ.get("%s", "0")' % AUDIT, PATCHED)
        self.assertEqual(patcher.PREFILL_CONV_MODULE.rsplit('.', 1)[1], Path(pcx.__file__).stem)


# ---------------------------------------------------------------------------------------------
# Executing the patched method.
# ---------------------------------------------------------------------------------------------

class FlagOffIdentityTests(unittest.TestCase):
    def test_flag_off_runs_exactly_the_original_calls(self):
        cases = 0
        for valid_len in (None, 1288, 2048):
            for conv1d in (True, False):
                for stable in (True, False):
                    for capture in (False, True):
                        for return_state in (False, True):
                            for flat in (True, False):
                                for environ in ({}, {FLAG: '0'}, {FLAG: 'yes'}):
                                    kwargs = dict(valid_len=valid_len, conv1d=conv1d, stable=stable, capture=capture,
                                                  return_state=return_state, flat=flat)
                                    with self.subTest(environ=environ, **kwargs):
                                        before, _ = drive(ORIGINAL, environ, op_mounted=False, **kwargs)
                                        # The flag-off arm mounts no op file: an import would raise here.
                                        after, _ = drive(PATCHED, environ, op_mounted=False, **kwargs)
                                        self.assertEqual(after.events, before.events)
                                        self.assertEqual(after.logs, [])
                                        self.assertEqual(after.op_imports, [])
                                        self.assertFalse(any(e[0].startswith('pcx.') for e in after.events))
                                    cases += 1
        self.assertEqual(cases, 288)

    def test_flag_on_leaves_the_valid_len_none_paths_alone(self):
        """conv1d (valid_len None) keeps priority; with conv1d off, the valid_len-None FIR (the
        trace-safe static slice) is not replaced either, and the op module is never imported."""
        for conv1d in (True, False):
            with self.subTest(conv1d=conv1d):
                before, _ = drive(ORIGINAL, {}, valid_len=None, conv1d=conv1d)
                after, _ = drive(PATCHED, {FLAG: '1'}, valid_len=None, conv1d=conv1d, op_mounted=False)
                self.assertEqual(after.events, before.events)
                self.assertEqual(after.logs, [])
                self.assertEqual(after.op_imports, [])

    def test_the_unmounted_op_is_seen_when_the_patch_does_import_it(self):
        """The guard above can fail: with the flag on and a valid_len, the patch imports the op, and
        an unmounted op raises (a flag-off import would crash every flag-off arm the same way)."""
        with self.assertRaises(ImportError):
            drive(PATCHED, {FLAG: '1'}, op_mounted=False)
        early = PATCHED.replace(
            '    if os.environ.get("%s") != "1":\n' % FLAG,
            '    from models.demos.blackhole.qwen36.tt.gdn import gdn_prefill_conv_exact as _early\n'
            '    if os.environ.get("%s") != "1":\n' % FLAG, 1)
        self.assertNotEqual(early, PATCHED)
        with self.assertRaises(ImportError):
            drive(early, {}, op_mounted=False)

    def test_the_fuse_out_branch_is_unchanged_too(self):
        before, _ = drive(ORIGINAL, {}, fuse_out=True)
        after, _ = drive(PATCHED, {}, fuse_out=True)
        self.assertEqual(after.events, before.events)


class FlagOnTests(unittest.TestCase):
    def names(self, run):
        return [event[0] for event in run.events]

    def test_the_op_replaces_the_fir_and_the_three_slices(self):
        before, _ = drive(ORIGINAL, {}, capture=True)
        after, _ = drive(PATCHED, {FLAG: '1'}, capture=True)
        names = self.names(after)
        self.assertNotIn('_causal_conv1d_fir', names)
        self.assertEqual(names.count('pcx.gdn_prefill_conv_exact'), 1)
        (op,) = [e for e in after.events if e[0] == 'pcx.gdn_prefill_conv_exact']
        (fir,) = [e for e in before.events if e[0] == '_causal_conv1d_fir']
        fir_kwargs = dict(fir[2])
        # The same qkv, carry and taps the FIR got; valid_len and the key width passed through.
        self.assertEqual(op[1], ('mesh', fir[1][0], fir_kwargs['conv_state'], fir_kwargs['weight_taps']))
        self.assertEqual(fir_kwargs['valid_len'], 1288)
        self.assertEqual(dict(op[2]), dict(valid_len=1288, key_dim_tp=1024))
        # Everything but the FIR, its three slices and the conv deallocate is the same sequence.
        fir_index = self.names(before).index('_causal_conv1d_fir')
        slices = [i for i, e in enumerate(before.events) if e[0] == 'ttnn.slice' and i > fir_index][:3]
        conv_dealloc = [i for i, e in enumerate(before.events) if e[0] == 'ttnn.deallocate' and i > slices[-1]][0]
        self.assertEqual(before.events[conv_dealloc][1][0], before.events[slices[0]][1][0])
        dropped = {fir_index, conv_dealloc, *slices}
        kept = [e[0] for i, e in enumerate(before.events) if i not in dropped]
        self.assertEqual([name for name in names if not name.startswith('pcx.')], kept)

    def test_q_k_v_and_the_carry_come_from_the_op(self):
        run, _ = drive(PATCHED, {FLAG: '1'}, capture=True)
        (index,) = [i for i, e in enumerate(run.events) if e[0] == 'pcx.gdn_prefill_conv_exact']
        qkv, carry = run.events[index][1][1], run.events[index][1][2]
        delta = [e for e in run.events if e[0] == 'chunk_gated_delta_rule_fused_adapter'][0]
        q, k, v = delta[1][:3]
        self.assertEqual(dict(delta[2])['qkv_head_dims'], (8, 128, 24, 128))
        copies = [e for e in run.events if e[0] == 'ttnn.copy']
        state, target = copies[1][1]          # rec_state first, then the conv carry
        self.assertEqual(target, carry)
        numbers = [int(label[1:]) for label in (q, k, v, state)]
        self.assertEqual(numbers, list(range(numbers[0], numbers[0] + 4)))   # the op's four outputs, in order
        self.assertGreater(numbers[0], int(qkv[1:]))
        deallocated = [e[1][0] for e in run.events if e[0] == 'ttnn.deallocate']
        self.assertIn(state, deallocated)
        self.assertNotIn(None, deallocated)
        self.assertIn(qkv, deallocated)       # qkv is still freed after the op
        # capture_state slices the op's state for the decode conv window, as it did the FIR's.
        windows = [e for e in run.events if e[0] == 'ttnn.slice' and e[1][0] == state]
        self.assertEqual(len(windows), 3)

    def test_one_marker_per_chunk_with_the_previous_chunks_calls(self):
        run, _ = drive(PATCHED, {FLAG: '1'}, layers=3, forwards=4)
        markers = [message for kind, message in run.logs if patcher.MARKER_PREFILL_CONV in message]
        self.assertEqual(len(markers), 4)
        self.assertEqual([re.search('previous_chunk_calls=([0-9]+)', m).group(1) for m in markers], ['0', '3', '3', '3'])
        self.assertEqual([re.search(' chunk ([0-9]+) ', m).group(1) for m in markers], ['1', '2', '3', '4'])
        self.assertTrue(markers[0].startswith(patcher.MARKER_PREFILL_CONV + ': chunk 1 previous_chunk_calls=0 calls=1 '))
        complete = [message for kind, message in run.logs if patcher.MARKER_PREFILL_CONV_COMPLETE in message]
        self.assertEqual(complete, [patcher.MARKER_PREFILL_CONV_COMPLETE + ': chunk %d calls=3' % n for n in (2, 3, 4)])
        import lever_n_m3native_gate as gate
        self.assertEqual(gate.PREFILL_CONV_COMPLETE_MARKER, patcher.MARKER_PREFILL_CONV_COMPLETE)
        log = chr(10).join(message for _, message in run.logs)
        self.assertEqual(gate.prefill_conv_chunk_calls(log), [3, 3, 3])
        self.assertEqual(gate.flag_marker_report({FLAG: '1'}, 1, log, prompt_tokens=4 * 2048, gdn_layers=3)['missing'], [])
        # At the model's 48 GDN layers, three-layer chunks are a partial replacement.
        self.assertEqual(len(gate.flag_marker_report({FLAG: '1'}, 1, log)['missing']), 3)
        # A fifth prompt chunk that never engaged, and a last chunk cut short, both fail.
        self.assertEqual(len(gate.flag_marker_report({FLAG: '1'}, 1, log, prompt_tokens=5 * 2048, gdn_layers=3)['missing']), 1)
        short, _ = drive(PATCHED, {FLAG: '1'}, layers=3, forwards=2)
        cut = chr(10).join(m for _, m in short.logs) + chr(10) + chr(10).join(
            m for _, m in run.logs if 'chunk 3 ' in m and patcher.MARKER_PREFILL_CONV in m)
        missing = gate.flag_marker_report({FLAG: '1'}, 1, cut, gdn_layers=3)['missing']
        self.assertTrue(any('the last chunk (3)' in m for m in missing), missing)

    def test_an_unsupported_input_falls_back_to_the_fir_and_says_so(self):
        before, _ = drive(ORIGINAL, {})
        after, _ = drive(PATCHED, {FLAG: '1'}, reason='qkv not bf16 TILE interleaved')
        self.assertEqual([e for e in after.events if not e[0].startswith('pcx.')], before.events)
        self.assertEqual([e[0] for e in after.events if e[0].startswith('pcx.')], ['pcx.unsupported'])
        (unsupported,) = [e for e in after.events if e[0] == 'pcx.unsupported']
        self.assertEqual(dict(unsupported[2]), dict(flat=True, kernel_size=4))
        self.assertEqual(after.logs, [('warning', patcher.MARKER_PREFILL_CONV_FALLBACK
                                       + ': qkv not bf16 TILE interleaved (T=2048 valid_len=1288)')])

    def test_the_audit_runs_for_the_first_n_calls_and_raises_on_a_difference(self):
        run, _ = drive(PATCHED, {FLAG: '1', AUDIT: '2'}, layers=2, forwards=2)
        audits = [e for e in run.events if e[0] == 'pcx.audit_against_fir']
        self.assertEqual(len(audits), 2)
        self.assertEqual(audits[0][1][1], '_causal_conv1d_fir')
        lines = [m for _, m in run.logs if m.startswith(patcher.MARKER_PREFILL_CONV_AUDIT)]
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith(patcher.MARKER_PREFILL_CONV_AUDIT + ' 1 exact=True'))
        with self.assertRaisesRegex(AssertionError, 'differs from the FIR'):
            drive(PATCHED, {FLAG: '1', AUDIT: '1'}, exact=False)
        quiet, _ = drive(PATCHED, {FLAG: '1'})
        self.assertFalse(any(e[0] == 'pcx.audit_against_fir' for e in quiet.events))


# ---------------------------------------------------------------------------------------------
# One mount table: the graft stages it, the arm mounts it.
# ---------------------------------------------------------------------------------------------

class MountTableTests(unittest.TestCase):
    def test_the_table_is_the_module_and_its_three_kernels_beside_gdn_tp(self):
        self.assertEqual(patcher.PREFILL_CONV_FILES, {
            'gdn/gdn_prefill_conv_exact.py': 'gdn_prefill_conv_exact.py',
            'gdn/gdn_prefill_conv_exact_reader.cpp': 'gdn_prefill_conv_exact_reader.cpp',
            'gdn/gdn_prefill_conv_exact_compute.cpp': 'gdn_prefill_conv_exact_compute.cpp',
            'gdn/gdn_prefill_conv_exact_writer.cpp': 'gdn_prefill_conv_exact_writer.cpp'})
        for relative, source in patcher.PREFILL_CONV_FILES.items():
            self.assertTrue((HERE / source).is_file(), source)
        self.assertEqual(set(pcx.RUNTIME_FILES), set(patcher.PREFILL_CONV_FILES.values()))
        self.assertEqual(sorted(set(patcher.PREFILL_CONV_FILES) & set(patcher.with_lever_n())), [])

    def test_the_patched_module_imports_the_mounted_name(self):
        for relative in patcher.PREFILL_CONV_FILES:
            if relative.endswith('.py'):
                dotted = patcher.MODEL_ROOT.replace('/opt/tt-metal/', '').replace('/', '.') + '.' + relative[:-3].replace('/', '.')
                self.assertEqual(dotted, patcher.PREFILL_CONV_MODULE)

    def _arm_block(self, environ, staged):
        bash = shutil.which('bash')
        if bash is None or shutil.which('python3') is None:
            self.skipTest('no bash / python3')
        arm = text(ARM)
        start = arm.index('prefill_conv_mounts=()')
        end = arm.index(chr(10) + 'fi' + chr(10), start) + 4
        script = ('set -euo pipefail' + chr(10) + 'root=/opt/tt-metal/models/demos/blackhole/qwen36/tt' + chr(10)
                  + arm[start:end] + 'printf "RESULT|%s" "${prefill_conv_mounts[*]:-}"' + chr(10))
        with tempfile.TemporaryDirectory() as directory:
            ci = Path(directory, 'scripts', 'ci')
            ci.mkdir(parents=True)
            for name in ('lever_n_m3native_patch.py', 'gdn_prefill_conv_exact.py'):
                shutil.copy(HERE / name, ci / name)
            for relative in staged:
                target = Path(directory, 'graft', relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('staged', encoding='utf-8')
            env = dict(os.environ)
            env.pop('M3NATIVE_GDN_PREFILL_CONV', None)
            env.update(environ)
            try:
                result = subprocess.run([bash, '-c', script], env=env, cwd=directory, capture_output=True,
                                        text=True, timeout=120)
            except OSError as error:
                self.skipTest('bash unusable: %s' % error)
        return result

    def test_the_arm_mounts_every_file_of_the_table_one_by_one_and_only_on_the_flag(self):
        files = sorted(patcher.PREFILL_CONV_FILES)
        result = self._arm_block({}, files)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split('RESULT|')[-1], '')
        result = self._arm_block({'M3NATIVE_GDN_PREFILL_CONV': '1'}, files)
        if result.returncode != 0 and 'No module named' in result.stderr:
            self.skipTest('python3 here cannot import from the temp tree: %s' % result.stderr.strip())
        self.assertEqual(result.returncode, 0, result.stderr)
        mounts = result.stdout.split('RESULT|')[-1].split(' --mount ')
        mounts = [m.replace('--mount ', '') for m in mounts]
        self.assertEqual(len(mounts), len(files))
        for mount, relative in zip(mounts, files):
            self.assertRegex(mount, '^type=bind,src=.*/graft/%s,dst=/opt/tt-metal/models/demos/blackhole/qwen36/tt/%s,readonly$'
                             % (re.escape(relative), re.escape(relative)))

    def test_the_arm_refuses_a_file_the_graft_did_not_stage(self):
        files = sorted(patcher.PREFILL_CONV_FILES)
        result = self._arm_block({'M3NATIVE_GDN_PREFILL_CONV': '1'}, files[1:])
        if 'No module named' in result.stderr:
            self.skipTest('python3 here cannot import from the temp tree')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('graft/%s was not staged' % files[0], result.stderr)

    def test_the_arm_wires_the_mounts_and_the_two_env_vars_into_the_docker_run(self):
        arm = text(ARM)
        run = arm[arm.index('timeout -k 30 2200 docker run'):arm.index('--entrypoint python3')]
        self.assertEqual(run.count('"${prefill_conv_mounts[@]}"'), 1)
        self.assertIn('${M3NATIVE_GDN_PREFILL_CONV:+-e QWEN_FAST_GDN_PREFILL_CONV=1}', run)
        self.assertIn('${M3NATIVE_GDN_PREFILL_CONV_AUDIT:+-e QWEN_FAST_GDN_PREFILL_CONV_AUDIT=$M3NATIVE_GDN_PREFILL_CONV_AUDIT}', run)
        # Single files only: never a directory over gdn/ or over the baked evidence tree.
        self.assertNotIn('dst=$root/gdn,', arm)
        self.assertNotRegex(arm, r'dst=/experiment-scripts/ci[,"]')
        self.assertIn('p.PREFILL_CONV_FILES', arm)

    def test_the_workflow_stages_the_same_table_and_hashes_it(self):
        workflow = text(WORKFLOW)
        self.assertIn('for relative, source in sorted(patcher.PREFILL_CONV_FILES.items()):', workflow)
        stage = workflow[workflow.index('graft/prefill-conv-manifest.txt'):]
        stage = stage[:stage.index('cat experiment-results/graft.sha256')]
        self.assertIn('cp "scripts/ci/$source" "graft/$relative"', stage)
        self.assertIn('sha256sum "graft/$relative" >> experiment-results/graft.sha256', stage)
        # Staged in the graft job, before its artifact upload (which the arm job downloads).
        self.assertLess(workflow.index('graft/prefill-conv-manifest.txt'), workflow.index('name: m3native-graft-${{ github.run_id }}'))

    def test_the_new_tests_are_allowlisted_in_the_cpu_suite(self):
        cpu = text(CPU_WORKFLOW)
        for name in ('test_gdn_prefill_conv_exact', 'test_lever_n_m3native_prefill_conv'):
            self.assertRegex(cpu, r'python -B -m unittest [^\n]*\b%s\b' % name)
        self.assertIn("python -B -m unittest discover -s optimisation/ttnn-op/gdn_prefill_conv -p 'test_*.py'", cpu)


# ---------------------------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------------------------

class GateTests(unittest.TestCase):
    def report(self, environ, log, users=1, prompt_tokens=None):
        import lever_n_m3native_gate as gate
        return gate.flag_marker_report(environ, users, log, prompt_tokens=prompt_tokens)

    def marker(self, chunk, previous):
        return '%s: chunk %d previous_chunk_calls=%d calls=%d T=2048 valid_len=2048 carry=True' % (
            patcher.MARKER_PREFILL_CONV, chunk, previous, 1 + 48 * (chunk - 1))

    def complete(self, chunk, calls=48):
        return '%s: chunk %d calls=%d' % (patcher.MARKER_PREFILL_CONV_COMPLETE, chunk, calls)

    def chunks(self, count, last_calls=48):
        lines = []
        for n in range(1, count + 1):
            lines.append(self.marker(n, 0 if n == 1 else 48))
            if n > 1 and (n < count or last_calls is not None):
                lines.append(self.complete(n, 48 if n < count else last_calls))
        return lines

    def test_the_flag_requires_its_marker(self):
        self.assertEqual(self.report({FLAG: '1'}, '')['missing'], [FLAG + ': ' + patcher.MARKER_PREFILL_CONV])
        self.assertEqual(self.report({}, '')['missing'], [])
        log = chr(10).join(self.chunks(16))
        report = self.report({FLAG: '1'}, log, prompt_tokens=32768)
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['prefill_conv_chunk_calls'], [48] * 15)
        self.assertEqual((report['prefill_conv']['engaged_chunks'], report['prefill_conv']['required_chunks'],
                          report['prefill_conv']['last_chunk_calls']), (16, 16, 48))

    def test_a_fallback_or_an_uneven_chunk_fails(self):
        base = self.chunks(2)
        fallback = base + [patcher.MARKER_PREFILL_CONV_FALLBACK + ': q/k/v not flat (T=2048 valid_len=5)']
        self.assertEqual(len(self.report({FLAG: '1'}, chr(10).join(fallback))['missing']), 1)
        uneven = base + [self.marker(3, 47), self.complete(3)]
        missing = self.report({FLAG: '1'}, chr(10).join(uneven))['missing']
        self.assertEqual(len(missing), 1)
        self.assertIn('[47, 48]', missing[0])

    def test_equal_but_partial_chunks_fail(self):
        """Every chunk engaging the same 47 layers used to pass (equal counts)."""
        log = [self.marker(1, 0)] + [self.marker(n, 47) for n in (2, 3)] + [self.complete(n, 47) for n in (2, 3)]
        missing = self.report({FLAG: '1'}, chr(10).join(log))['missing']
        self.assertTrue(any('all 48 GDN layers' in m and '[47]' in m for m in missing), missing)
        self.assertTrue(any('completion lines report [47]' in m for m in missing), missing)

    def test_only_the_tail_chunks_engaging_fails_the_chunk_floor(self):
        """Two users of 131072 tokens are 128 chunks; if only each user's tail engaged (full chunks
        on another path), the markers are present and every count is 48 - and the run must fail."""
        tails = chr(10).join(self.chunks(2))
        missing = self.report({FLAG: '1'}, tails, users=2, prompt_tokens=131072)['missing']
        self.assertEqual(len(missing), 1)
        self.assertIn('2 engaged prefill chunks, at least 128 expected', missing[0])
        full = chr(10).join(self.chunks(128))
        self.assertEqual(self.report({FLAG: '1'}, full, users=2, prompt_tokens=131072)['missing'], [])
        self.assertEqual(len(self.report({FLAG: '1'}, full, users=2, prompt_tokens=131073)['missing']), 1)

    def test_the_last_chunk_must_complete(self):
        for last_calls in (None, 47):
            with self.subTest(last_calls=last_calls):
                missing = self.report({FLAG: '1'}, chr(10).join(self.chunks(4, last_calls=last_calls)))['missing']
                self.assertTrue(any('the last chunk (4) never completed 48' in m for m in missing), missing)
        self.assertEqual(self.report({FLAG: '1'}, self.marker(1, 0))['missing'], [])   # one chunk: nothing later to count

    def test_the_arm_gate_passes_the_prompt_length(self):
        import lever_n_m3native_gate as gate
        source = Path(gate.__file__).read_text(encoding='utf-8')
        self.assertIn('flag_marker_report(os.environ, streams, log_text,' + chr(10)
                      + '                                                    prompt_tokens=options.prompt_tokens)', source)
        self.assertEqual(gate.prefill_conv_required_chunks(4, 32768), 64)
        self.assertEqual(gate.prefill_conv_required_chunks(1, 2049), 2)
        self.assertEqual(gate.prefill_conv_required_chunks(4, None), 0)

    def test_the_audit_flag_requires_an_exact_first_audit(self):
        log = chr(10).join([self.marker(1, 0), patcher.MARKER_PREFILL_CONV_AUDIT + ' 1 exact=True T=2048 valid_len=2048'])
        self.assertEqual(self.report({FLAG: '1', AUDIT: '64'}, log)['missing'], [])
        self.assertEqual(len(self.report({FLAG: '1', AUDIT: '64'}, self.marker(1, 0))['missing']), 1)
        self.assertEqual(self.report({FLAG: '1', AUDIT: '0'}, self.marker(1, 0))['missing'], [])


if __name__ == '__main__':
    unittest.main()
