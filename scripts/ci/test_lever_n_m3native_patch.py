"""Each of the four M3native patches must land on its one real anchor, or fail loudly.

Fixtures are the real dumped lines (probe run 35503727180) for the methods under test,
wrapped in the real class each lives in so function_span's AST sibling-order scoping
sees the same shape the real files do. gdn/tp.py's _project_qkvzab and
_project_qkvzab_raw read alike outside their own functions (both build the fused
qkvzab projection through the same 1D decode arm), which is exactly the ambiguity
lever_n_model_patch.py's own tests hold scoping honest against.
"""

import ast
import unittest
from unittest.mock import patch

import lever_n_m3native_patch as patcher

from lever_n_m3native_patch import (function_span, patch_attention_tp, patch_gdn_tp,
                                    patch_mlp, patch_model_config, replace_once)

# model_config.py, class Qwen36ModelArgs: real __init__ and _init_tp_config verbatim,
# plus the real trailing siblings that bound _init_tp_config's AST span.
MODEL_CONFIG = '''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

from models.tt_transformers.tt.model_config import ModelArgs

GDN_CONV1D_L1_SMALL_SIZE = 24576


class Qwen36ModelArgs(ModelArgs):
    """Qwen3.5-9B ModelArgs for Blackhole P150."""

    SUPPORTS_KV_REPLICATION = True

    def __init__(
        self,
        mesh_device=None,
        max_batch_size=1,
        max_seq_len=2048,
        **kwargs,
    ):
        hf_model = os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.6-27B")
        super().__init__(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)
        self.num_devices = mesh_device.get_num_devices() if mesh_device is not None else 1
        if mesh_device is not None and self.num_devices > 1:
            self._init_tp_config(mesh_device)

    def _init_tp_config(self, mesh_device):
        """Per-device sharded dims + DRAM matmul/mem configs for TP (num_devices>1)."""
        import ttnn
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        tp = self.num_devices
        self.cluster_shape = list(mesh_device.shape)

        self.gdn_nk = self.linear_num_key_heads
        self.gdn_dk = self.linear_key_head_dim
        self.gdn_nv = self.linear_num_value_heads
        self.gdn_dv = self.linear_value_head_dim
        self.gdn_conv_kernel_size = self.linear_conv_kernel_dim
        self.gdn_key_dim = self.linear_q_dim
        self.gdn_value_dim = self.linear_v_dim
        self.gdn_qkv_dim = self.linear_q_dim + self.linear_k_dim + self.linear_v_dim
        self.gdn_z_dim = self.linear_v_dim
        self.gdn_chunk_size = 128

        assert self.n_heads % tp == 0, f"n_heads {self.n_heads} not divisible by TP={tp}"
        assert self.gdn_nk % tp == 0 and self.gdn_nv % tp == 0, "GDN head counts must divide by TP"
        self.n_local_heads = self.n_heads // tp
        self.n_local_kv_heads = max(1, self.n_kv_heads // tp)
        self.kv_replication = tp > self.n_kv_heads
        self.gdn_nk_tp = self.gdn_nk // tp
        self.gdn_nv_tp = self.gdn_nv // tp
        self.gdn_qkv_dim_tp = self.gdn_qkv_dim // tp
        self.gdn_z_dim_tp = self.gdn_z_dim // tp
        self.gdn_qkvz_dim_tp = (self.gdn_qkv_dim + self.gdn_z_dim) // tp
        self.gdn_qkvzab_dim_tp = self.gdn_qkvz_dim_tp + 2 * self.gdn_nv_tp
        self.gdn_value_dim_tp = self.gdn_value_dim // tp
        self.gdn_key_dim_tp = self.gdn_key_dim // tp
        self.attn_out_dim_tp = (self.n_heads * self.head_dim) // tp
        kv_dim_per_device = self.n_local_kv_heads * self.head_dim

        self.gdn_qkvz_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.gdn_qkvz_dim_tp)
        self.gdn_qkvzab_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.gdn_qkvzab_dim_tp)
        self.attn_qg_weight_memcfg = tpc.create_dram_sharded_mem_config(
            self.dim, self.n_local_heads * self.head_dim * 2
        )
        self.attn_k_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, kv_dim_per_device)
        self.attn_v_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, kv_dim_per_device)
        self.attn_qkv_fused_dim_tp = self.n_local_heads * self.head_dim * 2 + 2 * kv_dim_per_device
        self.attn_qkv_fused_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.attn_qkv_fused_dim_tp)
        self.mlp_w1_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.hidden_dim // tp)
        self.mlp_w3_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.hidden_dim // tp)
        self.gdn_out_weight_memcfg = None
        self.attn_wo_weight_memcfg = None
        self.mlp_w2_weight_memcfg = tpc.create_dram_sharded_mem_config(self.hidden_dim // tp, self.dim)

        M = 1
        self.gdn_qkvz_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.gdn_qkvz_dim_tp)
        self.gdn_qkvzab_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.gdn_qkvzab_dim_tp)
        self.gdn_out_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.gdn_value_dim_tp, self.dim)
        self.attn_qg_progcfg = tpc.create_dram_sharded_matmul_program_config(
            M, self.dim, self.n_local_heads * self.head_dim * 2
        )
        self.attn_k_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, kv_dim_per_device)
        self.attn_v_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, kv_dim_per_device)
        self.attn_qkv_fused_progcfg = tpc.create_dram_sharded_matmul_program_config(
            M, self.dim, self.attn_qkv_fused_dim_tp
        )
        self.attn_wo_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.attn_out_dim_tp, self.dim)
        self.mlp_w1_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.hidden_dim // tp)
        self.mlp_w3_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.hidden_dim // tp)
        self.mlp_w2_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.hidden_dim // tp, self.dim)

        self.decode_grid_w = mesh_device.compute_with_storage_grid_size().x
        self.mlp_1d_decode = True
        self.mlp_w1_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.dim,
            self.hidden_dim // tp,
            num_cores=44,
            fused_activation=ttnn.UnaryOpType.SILU,
            grid_w=self.decode_grid_w,
        )
        self.mlp_w3_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.hidden_dim // tp, num_cores=44, grid_w=self.decode_grid_w
        )
        self.mlp_w2_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.hidden_dim // tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )

        self.proj_1d_decode = True
        self.attn_qkv_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.attn_qkv_fused_dim_tp, num_cores=64
        )
        self.gdn_qkvz_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.gdn_qkvzab_dim_tp, num_cores=44, grid_w=self.decode_grid_w
        )
        self.attn_wo_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.attn_out_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )
        self.gdn_out_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.gdn_value_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )

        self._prefill_grid = tpc.prefill_grid_default()
        self.prefill_tuning = tpc.prefill_tuning(tp)
        self.prefill_progcfg = lambda seq_len, k, n: tpc.create_prefill_matmul_program_config(
            seq_len, k, n, grid_size=self._prefill_grid, tuning=self.prefill_tuning
        )

        self.act_shard_hidden = tpc.create_activation_shard_config(self.dim)
        self.act_shard_gdn_value = tpc.create_activation_shard_config(self.gdn_value_dim_tp)
        self.act_shard_attn_out = tpc.create_activation_shard_config(self.attn_out_dim_tp)

        _B = max(1, self.max_batch_size)
        _cols = next(c for c in range(min(8, _B), 0, -1) if _B % c == 0)
        _rows = _B // _cols
        self.kv_update_shard_cfg = ttnn.create_sharded_memory_config(
            shape=(tpc.TILE_SIZE, self.head_dim),
            core_grid=ttnn.CoreGrid(x=_cols, y=_rows),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _set_hf_params(self, checkpoint_dir):
        self.trust_remote_code_hf = True
        super()._set_hf_params(checkpoint_dir)

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return self.attention_type_list[layer_idx] == "full_attention"
'''

# attention/tp.py, class TPAttention: _qkv_raw_decode, _qkv, _col_proj, _wo_proj and a
# trimmed forward_decode (its real _prep gate, verbatim, with the unrelated body either
# side stubbed to a valid but minimal shape) verbatim from the dump.
ATTENTION_TP = '''import os

import ttnn

from models.demos.blackhole.qwen36.tt import tp_common as tpc


class TPAttention:
    def _qkv_raw_decode(self, x):
        """Decode-only: the fused [q|k|v|gate] projection output, unsliced (for attn_decode_prep)."""
        tw = self.tw
        if getattr(self.args, "proj_1d_decode", False):
            return tpc.matmul_1d_decode(
                x,
                tw["wqkv_fused"],
                self.args.attn_qkv_decode_1d_progcfg,
                self.compute_cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._col_proj(x, tw["wqkv_fused"], self.args.attn_qkv_fused_progcfg)

    def _qkv(self, x):
        """Q+gate/K/V projections -> (qg, kp, vp). Fused path: one matmul, then slice."""
        tw = self.tw
        if not self._fused_qkv:
            return (
                self._col_proj(x, tw["wqkv"], self.args.attn_qg_progcfg),
                self._col_proj(x, tw["wk"], self.args.attn_k_progcfg),
                self._col_proj(x, tw["wv"], self.args.attn_v_progcfg),
            )
        if self._fuse_agmm and x.shape[-2] > tpc.TILE_SIZE:
            qkv = tpc.all_gather_matmul_prefill(
                x, tw["wqkv_fused"], self.tt_ccl, self.compute_cfg, self.args.ccl_topology()
            )
        elif getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: small-grid 1D matmul (interleaved weight). Output DRAM so _make_heads_decode's
            # to_memory_config(.,L1) stays a real copy before it deallocates the source.
            qkv = tpc.matmul_1d_decode(
                x,
                tw["wqkv_fused"],
                self.args.attn_qkv_decode_1d_progcfg,
                self.compute_cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            qkv = self._col_proj(x, tw["wqkv_fused"], self.args.attn_qkv_fused_progcfg)
        qkv3_dim = self.NH * self.HD + 2 * self.NKV * self.HD
        gate_dim = self.NH * self.HD
        sh = list(qkv.shape)
        _qkv3_mc = ttnn.L1_MEMORY_CONFIG if sh[2] > tpc.TILE_SIZE else ttnn.DRAM_MEMORY_CONFIG
        qkv3 = ttnn.slice(qkv, (0, 0, 0, 0), (sh[0], sh[1], sh[2], qkv3_dim), memory_config=_qkv3_mc)
        gate = ttnn.slice(qkv, (0, 0, 0, qkv3_dim), (sh[0], sh[1], sh[2], qkv3_dim + gate_dim))
        ttnn.deallocate(qkv)
        return qkv3, gate, None

    def _col_proj(self, x, weight, decode_progcfg):
        """Column-parallel projection; DRAM-sharded decode matmul when enabled."""
        if not self._dram_sharded:
            return ttnn.linear(x, weight, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.compute_cfg,
            decode_progcfg,
            self.args.act_shard_hidden,
            self.args.prefill_progcfg,
            self.args.dim,
        )

    def _wo_proj(self, x, weight):
        """Row-parallel output projection: DRAM-sharded decode/prefill matmul (K=attn_out_dim_tp),
        matching the in-proj. Falls back to plain interleaved when no sharded memcfg."""
        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.
            return tpc.matmul_1d_decode(
                x,
                weight,
                self.args.attn_wo_decode_1d_progcfg,
                self.compute_cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if not self._wo_sharded:
            if x.shape[-2] > tpc.TILE_SIZE:
                pc = tpc.create_prefill_mlp_matmul_program_config(
                    x.shape[-2],
                    weight.shape[-2],
                    weight.shape[-1],
                    max_cols=getattr(self.args, "decode_grid_w", 8),
                    tuning=getattr(self.args, "prefill_tuning", None),
                )
                return ttnn.linear(
                    x,
                    weight,
                    compute_kernel_config=self.compute_cfg,
                    program_config=pc,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            return ttnn.linear(x, weight, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.compute_cfg,
            self.args.attn_wo_progcfg,
            self.args.act_shard_attn_out,
            self.args.prefill_progcfg,
            self.args.attn_out_dim_tp,
        )

    def _make_heads(self, qg, kp, vp, S):
        return None

    def forward_decode(self, x, cur_pos_tt, cos_tt, sin_tt, page_table=None):
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        B = x.shape[-2]
        _L1 = ttnn.L1_MEMORY_CONFIG
        use_paged = self.use_paged and page_table is not None
        if not use_paged and self.k_caches is None:
            self.reset_state()

        _prep = (
            os.environ.get("QWEN_ATTN_PREP", "0") == "1"
            and use_paged
            and self._fused_qkv
            and x.shape[-2] <= ttnn.TILE_SIZE
        )
        if _prep:
            qkv_raw = self._qkv_raw_decode(x)
            return self._decode_from_prep(qkv_raw, cur_pos_tt, page_table, B)
        return None

    def _decode_from_prep(self, q, gate, k_sh, v_sh, cur_pos_tt, page_table, B):
        return None
'''

# gdn/tp.py, class TPGatedDeltaNet: _row_proj, _project_qkvzab and _project_qkvzab_raw
# verbatim from the dump - the ambiguity function_span must not fall for.
GDN_TP = '''import os

import ttnn

from models.demos.blackhole.qwen36.tt import tp_common as tpc


class TPGatedDeltaNet:
    def _row_proj(self, x, weight):
        """Row-parallel out projection: DRAM-sharded decode/prefill matmul (K=gdn_value_dim_tp),
        matching the in-proj. Falls back to plain interleaved on single device (no sharded memcfg)."""
        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.
            return tpc.matmul_1d_decode(
                x, weight, self.args.gdn_out_decode_1d_progcfg, self.cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        if not self._out_sharded:
            if x.shape[-2] > tpc.TILE_SIZE:
                pc = tpc.create_prefill_mlp_matmul_program_config(
                    x.shape[-2],
                    weight.shape[-2],
                    weight.shape[-1],
                    max_cols=getattr(self.args, "decode_grid_w", 8),
                    tuning=getattr(self.args, "prefill_tuning", None),
                )
                return ttnn.linear(
                    x, weight, compute_kernel_config=self.cfg, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            return ttnn.linear(x, weight, compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.cfg,
            self.args.gdn_out_progcfg,
            self.args.act_shard_gdn_value,
            self.args.prefill_progcfg,
            self.args.gdn_value_dim_tp,
        )

    def _project_qkvzab(self, x, S, out_mc=None):
        """Project x -> (qkv, z, a, b). Fused path: one [qkv|z|a|b] matmul then slice."""
        Nv, qz, az = self.Nv, self.qkv_dim_tp, self.qkvz_dim_tp
        _proj_mc = out_mc if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG
        if self._fuse_ab:
            if self._fuse_agmm and S > tpc.TILE_SIZE:
                qkvzab = tpc.all_gather_matmul_prefill(
                    x,
                    self.tw["qkvz"],
                    self.tt_ccl,
                    self.cfg,
                    self.args.ccl_topology(),
                    out_memory_config=_proj_mc,
                )
                qkvzab = ttnn.reshape(qkvzab, (1, S, qkvzab.shape[-1]))
            elif getattr(self.args, "proj_1d_decode", False) and S <= tpc.TILE_SIZE:
                # Decode: small-grid 1D matmul on the interleaved fused weight (beats the DRAM-sharded grid).
                qkvzab = tpc.matmul_1d_decode(
                    x,
                    self.tw["qkvz"],
                    self.args.gdn_qkvz_decode_1d_progcfg,
                    self.cfg,
                    out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                qkvzab = self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvzab_progcfg, out_memory_config=_proj_mc)
            qkv = ttnn.slice(qkvzab, (0, 0, 0), (1, S, qz), memory_config=out_mc)
            _z_mc = ttnn.DRAM_MEMORY_CONFIG if (self._fuse_agmm and S > tpc.TILE_SIZE) else out_mc
            z = ttnn.slice(qkvzab, (0, 0, qz), (1, S, az), memory_config=_z_mc)
            _ab_end = min(az + -(-2 * Nv // tpc.TILE_SIZE) * tpc.TILE_SIZE, qkvzab.shape[-1])
            ab = ttnn.slice(qkvzab, (0, 0, az), (1, S, _ab_end), memory_config=out_mc)
            ttnn.deallocate(qkvzab)
            a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv), memory_config=out_mc)
            b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv), memory_config=out_mc)
            ttnn.deallocate(ab)
            return qkv, z, a, b
        qkvz = self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvz_progcfg)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, S, qz))
        z = ttnn.slice(qkvz, (0, 0, qz), (1, S, az))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, self.tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv))
        ttnn.deallocate(ab)
        return qkv, z, a, b

    def forward_prefill(self, x, chunk_size=128, valid_len=None, capture_state=False, return_state=False):
        return None

    def _project_qkvzab_raw(self, x, S, out_mc):
        """Decode-only: the fused [qkv|z|a|b] projection output, unsliced (milestone 4)."""
        _proj_mc = out_mc if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG
        if getattr(self.args, "proj_1d_decode", False) and S <= tpc.TILE_SIZE:
            return tpc.matmul_1d_decode(
                x,
                self.tw["qkvz"],
                self.args.gdn_qkvz_decode_1d_progcfg,
                self.cfg,
                out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvzab_progcfg, out_memory_config=_proj_mc)

    def _conv_gates_enabled(self):
        return os.environ.get("QWEN_GDN_CONV_GATES", "0") == "1"
'''

# mlp.py, class Qwen36MLP: _forward_tp verbatim (both the w1/w3 site and the w2 site).
MLP = '''import ttnn

from models.demos.blackhole.qwen36.tt import tp_common as tpc


class Qwen36MLP:
    def _forward_tp(self, x):
        """TP forward: replicated input; reduce-scatter output fractured on hidden dim."""
        from models.tt_transformers.tt.ccl import tt_all_reduce

        w = self.weights
        args = self.args
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        mc = ttnn.DRAM_MEMORY_CONFIG
        _silu_fused = False
        _fused_gu = self._fuse_gateup_agmm and x.shape[-2] > ttnn.TILE_SIZE and w.w_gate_up is not None
        if _fused_gu:
            hidden = tpc.all_gather_swiglu_prefill(
                x, w.w_gate_up, self.tt_ccl, self.compute_kernel_config_agmm, args.ccl_topology()
            )
            _silu_fused = True
        elif getattr(self, "_dram_sharded", False) and x.shape[-2] <= ttnn.TILE_SIZE:
            x_sh = ttnn.to_memory_config(x, args.act_shard_hidden)
            w1_out = ttnn.linear(
                x_sh,
                w.w1,
                compute_kernel_config=ckc,
                program_config=args.mlp_w1_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_sh,
                w.w3,
                compute_kernel_config=ckc,
                program_config=args.mlp_w3_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_sh)
            w1_out = ttnn.to_memory_config(w1_out, ttnn.L1_MEMORY_CONFIG)
            w3_out = ttnn.to_memory_config(w3_out, ttnn.L1_MEMORY_CONFIG)
        elif self._mlp_1d_decode and x.shape[-2] <= ttnn.TILE_SIZE:
            # 1D mcast decode matmuls on a small explicit grid, silu fused in the w1 progcfg.
            # mcast_in0 needs interleaved in0, but ff-norm hands us a width-shard -> interleave first.
            x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            w1_out = ttnn.linear(
                x_il,
                w.w1,
                compute_kernel_config=ckc,
                program_config=args.mlp_w1_decode_1d_progcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_il,
                w.w3,
                compute_kernel_config=ckc,
                program_config=args.mlp_w3_decode_1d_progcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_il)
            _silu_fused = True
        elif x.shape[-2] > ttnn.TILE_SIZE:
            seq = x.shape[-2]
            _gw = getattr(args, "decode_grid_w", 8)
            _pt = getattr(args, "prefill_tuning", None)
            pc_gate = tpc.create_prefill_mlp_matmul_program_config(
                seq, args.dim, w.w1.shape[-1], fused_activation=ttnn.UnaryOpType.SILU, max_cols=_gw, tuning=_pt
            )
            pc_up = tpc.create_prefill_mlp_matmul_program_config(
                seq, args.dim, w.w3.shape[-1], max_cols=_gw, tuning=_pt
            )
            w1_out = ttnn.linear(
                x, w.w1, compute_kernel_config=ckc, program_config=pc_gate, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            w3_out = ttnn.linear(
                x, w.w3, compute_kernel_config=ckc, program_config=pc_up, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            _silu_fused = True
        else:
            w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
            w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
            _silu_fused = True

        _prefill_tuned = x.shape[-2] > ttnn.TILE_SIZE and _silu_fused
        if not _fused_gu:
            mc_out = ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= ttnn.TILE_SIZE else mc
            if _silu_fused:
                hidden = ttnn.mul(w1_out, w3_out, memory_config=mc_out)
                ttnn.deallocate(w1_out)
            else:
                w1_act = ttnn.silu(w1_out, memory_config=mc_out)
                ttnn.deallocate(w1_out)
                hidden = ttnn.mul(w1_act, w3_out, memory_config=mc_out)
                ttnn.deallocate(w1_act)
            ttnn.deallocate(w3_out)
        w2_pc = None
        if self._mlp_1d_decode and hidden.shape[-2] <= ttnn.TILE_SIZE:
            # 1D mcast decode down-proj on a small explicit grid (~16 cores).
            w2_pc = args.mlp_w2_decode_1d_progcfg
        elif hidden.shape[-2] > ttnn.TILE_SIZE:
            w2_pc = tpc.create_prefill_mlp_matmul_program_config(
                hidden.shape[-2],
                hidden.shape[-1],
                w.w2.shape[-1],
                max_cols=getattr(args, "decode_grid_w", 8),
                tuning=getattr(args, "prefill_tuning", None),
            )
        mc_w2_out = ttnn.L1_MEMORY_CONFIG if (x.shape[-2] <= ttnn.TILE_SIZE or _prefill_tuned) else mc
        partial = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc_w2_out, program_config=w2_pc)
        ttnn.deallocate(hidden)

        out = tt_all_reduce(
            partial,
            self.device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out
'''


class ModelConfigPatchTests(unittest.TestCase):
    def test_seven_native_64_configs_appended_after_kv_update_shard_cfg(self):
        out = patch_model_config(MODEL_CONFIG)
        ast.parse(out)
        for name in ('mlp_w1_decode_1d_progcfg_64', 'mlp_w3_decode_1d_progcfg_64',
                    'mlp_w2_decode_1d_progcfg_64', 'attn_qkv_decode_1d_progcfg_64',
                    'gdn_qkvz_decode_1d_progcfg_64', 'attn_wo_decode_1d_progcfg_64',
                    'gdn_out_decode_1d_progcfg_64'):
            self.assertIn('self.%s = tpc.create_matmul_1d_decode_progcfg(' % name, out)
        self.assertIn('M64 = 64', out)
        # appended AFTER kv_update_shard_cfg, inside _init_tp_config
        self.assertLess(out.index('self.kv_update_shard_cfg = ttnn.create_sharded_memory_config('),
                        out.index('M64 = 64'))
        self.assertLess(out.index('M64 = 64'), out.index('def _set_hf_params'))

    def test_every_m_equals_1_config_is_byte_identical(self):
        out = patch_model_config(MODEL_CONFIG)
        for line in MODEL_CONFIG.splitlines():
            if 'decode_1d_progcfg' in line and '_64' not in line:
                self.assertEqual(out.count(line), 1, line)

    def test_the_seven_new_configs_use_the_same_arguments_as_their_m_equals_1_originals(self):
        out = patch_model_config(MODEL_CONFIG)
        pairs = (
            ('self.mlp_w1_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(\n'
             '            M,\n'
             '            self.dim,\n'
             '            self.hidden_dim // tp,\n'
             '            num_cores=44,\n'
             '            fused_activation=ttnn.UnaryOpType.SILU,\n'
             '            grid_w=self.decode_grid_w,\n'
             '        )',
             'self.mlp_w1_decode_1d_progcfg_64 = tpc.create_matmul_1d_decode_progcfg(\n'
             '            M64,\n'
             '            self.dim,\n'
             '            self.hidden_dim // tp,\n'
             '            num_cores=44,\n'
             '            fused_activation=ttnn.UnaryOpType.SILU,\n'
             '            grid_w=self.decode_grid_w,\n'
             '        )'),
            ('self.attn_wo_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(\n'
             '            M, self.attn_out_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w\n'
             '        )',
             'self.attn_wo_decode_1d_progcfg_64 = tpc.create_matmul_1d_decode_progcfg(\n'
             '            M64, self.attn_out_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w\n'
             '        )'),
        )
        for original, native in pairs:
            self.assertIn(original, out)
            self.assertIn(native, out)

    def test_applies_exactly_once(self):
        with self.assertRaises(ValueError):
            patch_model_config(patch_model_config(MODEL_CONFIG))

    def test_a_missing_anchor_fails_loudly(self):
        broken = MODEL_CONFIG.replace(
            '        self.kv_update_shard_cfg = ttnn.create_sharded_memory_config(\n'
            '            shape=(tpc.TILE_SIZE, self.head_dim),\n'
            '            core_grid=ttnn.CoreGrid(x=_cols, y=_rows),\n'
            '            strategy=ttnn.ShardStrategy.HEIGHT,\n'
            '            orientation=ttnn.ShardOrientation.ROW_MAJOR,\n'
            '            use_height_and_width_as_shard_shape=True,\n'
            '        )\n',
            '        pass\n', 1)
        with self.assertRaises(ValueError):
            patch_model_config(broken)

    def test_a_duplicated_anchor_within_the_function_fails_loudly(self):
        anchor_start = MODEL_CONFIG.index('        self.kv_update_shard_cfg = ttnn.create_sharded_memory_config(')
        anchor_end = MODEL_CONFIG.index('        )\n', anchor_start) + len('        )\n')
        anchor = MODEL_CONFIG[anchor_start:anchor_end]
        duplicated = MODEL_CONFIG[:anchor_end] + anchor + MODEL_CONFIG[anchor_end:]
        with self.assertRaises(ValueError):
            patch_model_config(duplicated)


class AttentionTpPatchTests(unittest.TestCase):
    def test_qkv_gate_widens_and_selects_the_64_config_above_one_tile(self):
        out = patch_attention_tp(ATTENTION_TP)
        ast.parse(out)
        self.assertIn('elif getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= 2 * tpc.TILE_SIZE:', out)
        self.assertIn('self.args.attn_qkv_decode_1d_progcfg_64 if x.shape[-2] > tpc.TILE_SIZE', out)
        self.assertIn('else self.args.attn_qkv_decode_1d_progcfg,', out)

    def test_wo_proj_gate_widens_and_selects_the_64_config_above_one_tile(self):
        out = patch_attention_tp(ATTENTION_TP)
        self.assertIn('if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= 2 * tpc.TILE_SIZE:', out)
        self.assertIn('self.args.attn_wo_decode_1d_progcfg_64 if x.shape[-2] > tpc.TILE_SIZE', out)
        self.assertIn('else self.args.attn_wo_decode_1d_progcfg,', out)

    def test_forward_decode_prep_gate_widens(self):
        out = patch_attention_tp(ATTENTION_TP)
        self.assertIn('and x.shape[-2] <= 2 * ttnn.TILE_SIZE\n        )', out)
        self.assertNotIn('and x.shape[-2] <= ttnn.TILE_SIZE\n        )', out)

    def test_rows_up_to_32_still_read_as_the_original_one_tile_branch(self):
        """Reading the widened source AT rows<=32 must still route to the M=1 config -
        the ternary's else arm is exactly the original expression, byte for byte."""
        out = patch_attention_tp(ATTENTION_TP)
        self.assertIn('else self.args.attn_qkv_decode_1d_progcfg,\n', out)
        self.assertIn('else self.args.attn_wo_decode_1d_progcfg,\n', out)

    def test_applies_exactly_once(self):
        with self.assertRaises(ValueError):
            patch_attention_tp(patch_attention_tp(ATTENTION_TP))

    def test_a_missing_qkv_anchor_fails_loudly(self):
        broken = ATTENTION_TP.replace(
            'elif getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:',
            'elif False:', 1)
        with self.assertRaises(ValueError):
            patch_attention_tp(broken)

    def test_a_duplicated_wo_proj_anchor_fails_loudly(self):
        span = function_span(ATTENTION_TP, '_wo_proj')
        lines = ATTENTION_TP.splitlines(keepends=True)
        region = ''.join(lines[span[0]:span[1]])
        anchor_start = region.index('        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:')
        anchor_end = region.index('            )\n', anchor_start) + len('            )\n')
        anchor = region[anchor_start:anchor_end]
        duplicated_region = region[:anchor_end] + anchor + region[anchor_end:]
        lines[span[0]:span[1]] = duplicated_region.splitlines(keepends=True)
        with self.assertRaises(ValueError):
            patch_attention_tp(''.join(lines))


class GdnTpPatchTests(unittest.TestCase):
    def test_row_proj_gate_widens_and_selects_the_64_config(self):
        out = patch_gdn_tp(GDN_TP)
        ast.parse(out)
        self.assertIn('if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= 2 * tpc.TILE_SIZE:', out)
        self.assertIn('_row_proj_cfg = (self.args.gdn_out_decode_1d_progcfg_64 if x.shape[-2] > tpc.TILE_SIZE', out)
        self.assertIn('else self.args.gdn_out_decode_1d_progcfg)', out)

    def test_project_qkvzab_gate_widens_and_selects_the_64_config(self):
        out = patch_gdn_tp(GDN_TP)
        self.assertIn('elif getattr(self.args, "proj_1d_decode", False) and S <= 2 * tpc.TILE_SIZE:', out)
        self.assertIn('self.args.gdn_qkvz_decode_1d_progcfg_64 if S > tpc.TILE_SIZE', out)

    def test_project_qkvzab_raw_gate_widens_and_selects_the_64_config(self):
        """The ambiguity function_span must resolve correctly: _project_qkvzab and
        _project_qkvzab_raw both read the same qkvz weight through the same 1D arm."""
        out = patch_gdn_tp(GDN_TP)
        span = function_span(out, '_project_qkvzab_raw')
        region = ''.join(out.splitlines(keepends=True)[span[0]:span[1]])
        self.assertIn('if getattr(self.args, "proj_1d_decode", False) and S <= 2 * tpc.TILE_SIZE:', region)
        self.assertIn('self.args.gdn_qkvz_decode_1d_progcfg_64 if S > tpc.TILE_SIZE', region)
        self.assertEqual(region.count('gdn_qkvz_decode_1d_progcfg_64'), 1)

    def test_project_qkvzab_and_its_raw_sibling_are_each_patched_exactly_once(self):
        out = patch_gdn_tp(GDN_TP)
        self.assertEqual(out.count('S <= 2 * tpc.TILE_SIZE'), 2, 'one in _project_qkvzab, one in its _raw sibling')

    def test_applies_exactly_once(self):
        with self.assertRaises(ValueError):
            patch_gdn_tp(patch_gdn_tp(GDN_TP))

    def test_a_missing_project_qkvzab_raw_anchor_fails_loudly(self):
        broken = GDN_TP.replace(
            '        if getattr(self.args, "proj_1d_decode", False) and S <= tpc.TILE_SIZE:\n'
            '            return tpc.matmul_1d_decode(\n'
            '                x,\n'
            '                self.tw["qkvz"],\n'
            '                self.args.gdn_qkvz_decode_1d_progcfg,\n'
            '                self.cfg,\n'
            '                out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,\n'
            '            )\n',
            '        return None\n', 1)
        with self.assertRaises(ValueError):
            patch_gdn_tp(broken)


class MlpPatchTests(unittest.TestCase):
    def test_w1_w3_gate_widens_and_selects_the_64_configs(self):
        out = patch_mlp(MLP)
        ast.parse(out)
        self.assertIn('elif self._mlp_1d_decode and x.shape[-2] <= 2 * ttnn.TILE_SIZE:', out)
        self.assertIn('_w1_1d_cfg = (args.mlp_w1_decode_1d_progcfg_64 if x.shape[-2] > ttnn.TILE_SIZE', out)
        self.assertIn('_w3_1d_cfg = (args.mlp_w3_decode_1d_progcfg_64 if x.shape[-2] > ttnn.TILE_SIZE', out)
        self.assertIn('program_config=_w1_1d_cfg,', out)
        self.assertIn('program_config=_w3_1d_cfg,', out)

    def test_w2_gate_widens_and_selects_the_64_config(self):
        out = patch_mlp(MLP)
        self.assertIn('if self._mlp_1d_decode and hidden.shape[-2] <= 2 * ttnn.TILE_SIZE:', out)
        self.assertIn('w2_pc = (args.mlp_w2_decode_1d_progcfg_64 if hidden.shape[-2] > ttnn.TILE_SIZE', out)
        self.assertIn('else args.mlp_w2_decode_1d_progcfg)', out)

    def test_applies_exactly_once(self):
        with self.assertRaises(ValueError):
            patch_mlp(patch_mlp(MLP))

    def test_a_missing_w2_anchor_fails_loudly(self):
        broken = MLP.replace(
            '        if self._mlp_1d_decode and hidden.shape[-2] <= ttnn.TILE_SIZE:\n'
            '            # 1D mcast decode down-proj on a small explicit grid (~16 cores).\n'
            '            w2_pc = args.mlp_w2_decode_1d_progcfg\n',
            '        if False:\n            w2_pc = None\n', 1)
        with self.assertRaises(ValueError):
            patch_mlp(broken)

    def test_a_duplicated_w1_w3_anchor_fails_loudly(self):
        span = function_span(MLP, '_forward_tp')
        lines = MLP.splitlines(keepends=True)
        region = ''.join(lines[span[0]:span[1]])
        anchor_start = region.index('        elif self._mlp_1d_decode and x.shape[-2] <= ttnn.TILE_SIZE:')
        anchor_end = region.index('            _silu_fused = True\n', anchor_start) + len('            _silu_fused = True\n')
        anchor = region[anchor_start:anchor_end]
        duplicated_region = region[:anchor_end] + anchor + region[anchor_end:]
        lines[span[0]:span[1]] = duplicated_region.splitlines(keepends=True)
        with self.assertRaises(ValueError):
            patch_mlp(''.join(lines))


class FusedPrefillLayoutGuardTests(unittest.TestCase):
    """Run 35558196643: at 64 rows every fused all-gather + matmul PREFILL branch was
    selected on rows alone and refused the replicated decode input (K=10240 vs
    K_w=5120). Each branch now also requires x narrower than the weight K."""

    def test_attention_qkv_fused_branch_requires_the_k_sharded_input(self):
        out = patch_attention_tp(ATTENTION_TP)
        self.assertIn('if self._fuse_agmm and x.shape[-2] > tpc.TILE_SIZE and x.shape[-1] < tw["wqkv_fused"].shape[-2]:', out)
        self.assertEqual(out.count('if self._fuse_agmm and x.shape[-2] > tpc.TILE_SIZE'), 1)
        self.assertEqual(out.count('qkv = tpc.all_gather_matmul_prefill('), 1)

    def test_gdn_project_qkvzab_fused_branch_requires_the_k_sharded_input(self):
        out = patch_gdn_tp(GDN_TP)
        self.assertIn('if self._fuse_agmm and S > tpc.TILE_SIZE and x.shape[-1] < self.tw["qkvz"].shape[-2]:', out)
        self.assertEqual(out.count('qkvzab = tpc.all_gather_matmul_prefill('), 1)

    def test_mlp_fused_gate_up_requires_the_k_sharded_input(self):
        out = patch_mlp(MLP)
        self.assertIn('_fused_gu = (self._fuse_gateup_agmm and x.shape[-2] > ttnn.TILE_SIZE and w.w_gate_up is not None', out)
        self.assertIn('and x.shape[-1] < w.w_gate_up.shape[-2])', out)
        self.assertEqual(out.count('_fused_gu = '), 1)
        self.assertEqual(out.count('hidden = tpc.all_gather_swiglu_prefill('), 1)

    def test_the_guard_is_absent_before_patching(self):
        for source in (ATTENTION_TP, GDN_TP, MLP):
            self.assertNotIn('x.shape[-1] < ', source)


class GraftFileSetTests(unittest.TestCase):
    """The graft's file list used to live in four places - the docker cp lines, two
    sha256sum lines and PATCHES - and the 65536 hardware lane shipped a runner nothing
    staged because of exactly that duplication. SOURCES is now the single table the
    workflow derives all of them from, so these tests pin its shape.
    """

    def test_sources_is_the_decode_side_four_under_the_model_root(self):
        self.assertEqual(sorted(patcher.SOURCES),
                         ['attention/tp.py', 'gdn/tp.py', 'mlp.py', 'model_config.py'])
        for relative, (directory, apply) in patcher.SOURCES.items():
            self.assertEqual(directory, patcher.MODEL_ROOT, relative)
            self.assertTrue(callable(apply), relative)

    def test_sources_and_patches_cannot_drift_apart(self):
        self.assertEqual(sorted(patcher.SOURCES), sorted(patcher.PATCHES))
        for relative, patch in patcher.PATCHES.items():
            self.assertIs(patcher.SOURCES[relative][1], patch)

    def test_with_lever_n_adds_the_three_m1_files_from_their_real_trees(self):
        full = patcher.with_lever_n()
        self.assertEqual(sorted(set(full) - set(patcher.SOURCES)),
                         ['model.py', 'platform.py', 'qwen36_vllm.py'])
        # platform.py is the one that does NOT live under the model root.
        self.assertEqual(full['platform.py'][0], patcher.PLUGIN_ROOT)
        self.assertEqual(full['model.py'][0], patcher.MODEL_ROOT)
        self.assertEqual(full['qwen36_vllm.py'][0], patcher.MODEL_ROOT)

    def test_with_lever_n_leaves_the_decode_side_four_untouched(self):
        full = patcher.with_lever_n()
        for relative, entry in patcher.SOURCES.items():
            self.assertEqual(full[relative], entry, relative)

    def test_an_overlap_between_the_two_grafts_is_refused(self):
        """Two patchers rewriting one file would silently drop one of them."""
        with patch.dict(patcher.SOURCES, {'model.py': (patcher.MODEL_ROOT, lambda s: s)}):
            with self.assertRaisesRegex(ValueError, 'graft the same file'):
                patcher.with_lever_n()


if __name__ == '__main__':
    unittest.main()
