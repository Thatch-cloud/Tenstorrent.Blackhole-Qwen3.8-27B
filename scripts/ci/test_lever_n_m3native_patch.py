"""Each of the four M3native patches must land on its one real anchor, or fail loudly.

Fixtures are the real dumped lines (probe run 35503727180) for the methods under test,
wrapped in the real class each lives in so function_span's AST sibling-order scoping
sees the same shape the real files do. gdn/tp.py's _project_qkvzab and
_project_qkvzab_raw read alike outside their own functions (both build the fused
qkvzab projection through the same 1D decode arm), which is exactly the ambiguity
lever_n_model_patch.py's own tests hold scoping honest against.
"""

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import lever_n_m3native_patch as patcher

from lever_n_m3native_patch import (function_span, module_function_span, patch_attention_tp,
                                    patch_gdn_tp, patch_mlp, patch_mlp_full, patch_mlp_single_gateup,
                                    patch_layer, patch_model_config, replace_once)

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

    def remap_slots(self, remap):
        """Reindex the batched decode state after a vLLM batch condense: slot i takes the state
        previously at slot remap[i] (identity entries are no-ops)."""
        idx = [int(remap[i]) for i in range(self.B)]
        if all(idx[i] == i for i in range(self.B)):
            return
        self._gather_indices(self.rec_state, idx, dim=0)
        for m in range(self.K):
            self._gather_indices(self.conv_states[m], idx, dim=1)
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

    def test_sources_is_the_decode_side_five_under_the_model_root(self):
        self.assertEqual(sorted(patcher.SOURCES),
                         ['attention/tp.py', 'gdn/tp.py', 'layer.py', 'mlp.py', 'model_config.py'])
        for relative, (directory, apply) in patcher.SOURCES.items():
            self.assertEqual(directory, patcher.MODEL_ROOT, relative)
            self.assertTrue(callable(apply), relative)

    def test_sources_and_patches_cannot_drift_apart(self):
        self.assertEqual(sorted(patcher.SOURCES), sorted(patcher.PATCHES))
        for relative, patch in patcher.PATCHES.items():
            self.assertIs(patcher.SOURCES[relative][1], patch)

    def test_with_lever_n_adds_the_lever_n_files_from_their_real_trees(self):
        """Five now. scheduler.py is M2 item 2, one prefill in
        flight - after run 35690327326 proved the overlay route cannot deliver it:
        serving_one_in_flight.install sets scheduler_config.scheduler_cls and the
        plugin's platform.check_and_update_config overwrites it afterwards."""
        full = patcher.with_lever_n()
        self.assertEqual(sorted(set(full) - set(patcher.SOURCES)),
                         ['lane_scheduler.py', 'model.py', 'platform.py',
                          'qwen36_vllm.py', 'scheduler.py'])
        # The three that do NOT live under the model root are the plugin's own.
        self.assertEqual(full['platform.py'][0], patcher.PLUGIN_ROOT)
        self.assertEqual(full['scheduler.py'][0], patcher.PLUGIN_ROOT)
        self.assertEqual(full['lane_scheduler.py'][0], patcher.PLUGIN_ROOT)
        self.assertEqual(full['model.py'][0], patcher.MODEL_ROOT)
        self.assertEqual(full['qwen36_vllm.py'][0], patcher.MODEL_ROOT)

    def test_every_lever_n_file_is_mounted_by_the_arm(self):
        """A graft the arm does not mount is a file patched into an artifact and never
        served - which is exactly how steps 1/2/4, 5 and 8 sat unused for five runs."""
        arm = (Path(__file__).parent / 'lever_n_m3native_run_arm.sh').read_text(encoding='utf-8')
        for relative in sorted(set(patcher.with_lever_n()) - set(patcher.SOURCES)):
            with self.subTest(graft=relative):
                self.assertIn('graft/%s,dst=' % relative, arm)

    def test_with_lever_n_leaves_the_decode_side_four_untouched(self):
        full = patcher.with_lever_n()
        for relative, entry in patcher.SOURCES.items():
            self.assertEqual(full[relative], entry, relative)

    def test_an_overlap_between_the_two_grafts_is_refused(self):
        """Two patchers rewriting one file would silently drop one of them."""
        with patch.dict(patcher.SOURCES, {'model.py': (patcher.MODEL_ROOT, lambda s: s)}):
            with self.assertRaisesRegex(ValueError, 'graft the same file'):
                patcher.with_lever_n()


# mlp.py's load_mlp_weights head, verbatim from the image (graft artifact of run
# 35801010447, mlp.py.orig lines 45-78), closed off so it parses and can be executed.
LOAD_MLP_WEIGHTS = '''import os


def load_mlp_weights(mesh_device, state_dict, tensor_cache_path=None, args=None) -> MLPWeights:
    """Per-layer MLP state: gate_proj, down_proj, up_proj weights."""
    tp = getattr(args, "num_devices", 1) if args is not None else 1

    if tp > 1:
        # TP: w1/w3 column-parallel (shard out dim), w2 row-parallel (shard in dim).
        # DRAM-sharded memcfgs from args.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        # w1/w3 DRAM-WIDTH_SHARDED for decode (M=1 tile, ~+10% tok/s); w2 interleaved.
        # Cache uses `.dramshard` suffix - layout incompatible with interleaved cache
        # (as_tensor ignores requested memcfg on reload). Fallback if memcfgs absent.
        # 1D-decode (default) uses interleaved weights (its mcast decode matmul needs them).
        dram_sharded = (
            args is not None
            and getattr(args, "mlp_w1_weight_memcfg", None) is not None
            and not getattr(args, "mlp_1d_decode", False)
        )

        def cache(name, tag=""):
            return str(tensor_cache_path / f"mlp.{name}.weight{tag}.tp") if tensor_cache_path else None

        # Prefill-only packed [gate|up] AGMM weight (decode keeps w1/w3; extra DRAM ~w1+w3/layer).
        wgu = (
            _build_gate_up(
                state_dict["gate_proj.weight"],
                state_dict["up_proj.weight"],
                mesh_device,
                tp,
                cache("gate_up", ".swiglu"),
            )
            if tpc.mlp_gateup_agmm_enabled(tp)
            else None
        )
        return wgu
    return None


'''

# Qwen36MLP.__init__'s switch lines, verbatim (mlp.py.orig lines 174-178), in front of the
# fixture's _forward_tp so function_span sees the real sibling order.
MLP_INIT = '''    def __init__(self, mesh_device, state_dict, tensor_cache_path=None, args=None, tt_ccl=None):
        self.num_devices = getattr(args, "num_devices", 1) if args is not None else 1
        # Prefill fused-swiglu AGMM (ff_norm skips its AG; layer.py sets _fuse_ff_agmm to match).
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._fuse_gateup_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)

'''

MLP_MODULE = LOAD_MLP_WEIGHTS + MLP.replace('class Qwen36MLP:\n', 'class Qwen36MLP:\n' + MLP_INIT, 1)

# layer.py (probe run 35503727180, sha256 586429a11078f91d): Qwen36DecoderLayer.__init__
# lines 25-77 verbatim, module imports dropped so it executes without the model tree.
LAYER = '''class Qwen36DecoderLayer:
    """Single transformer layer with hybrid attention dispatch."""

    def __init__(self, mesh_device, args, state_dict, layer_num, tensor_cache_path=None, tt_ccl=None):
        self.layer_num = layer_num
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1)
        self.is_full_attention = args.is_full_attention_layer(layer_num)

        prefix = f"layers.{layer_num}"

        self._fuse_norm_agmm = self.num_devices > 1 and (
            (not self.is_full_attention and getattr(args, "gdn_qkvz_weight_memcfg", None) is not None)
            or (self.is_full_attention and getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None)
        )
        self.attention_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "input_layernorm",
            tensor_cache_path,
            tt_ccl,
            "attention_norm",
            enable_all_gather=not self._fuse_norm_agmm,
        )
        # Prefill: ff_norm skips AG (fused into gate/up AGMM); decode gathers pre-norm so this is a no-op there.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
        self.ffn_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "post_attention_layernorm",
            tensor_cache_path,
            tt_ccl,
            "ff_norm",
            enable_all_gather=not self._fuse_ff_agmm,
        )

    def _make_norm(self, *args, enable_all_gather=True):
        return enable_all_gather
'''


def _stubbed(agmm, environ, extra=None):
    """sys.modules and os.environ for executing a grafted model file without the model tree."""
    import sys
    import types

    logged = []
    tpc = types.SimpleNamespace(mlp_gateup_agmm_enabled=lambda n: agmm and n > 1)
    loguru = types.ModuleType('loguru')
    loguru.logger = types.SimpleNamespace(info=lambda *a: logged.append(a))
    modules = {'loguru': loguru, 'ttnn': types.SimpleNamespace(TILE_SIZE=32)}
    for name in ('models', 'models.demos', 'models.demos.blackhole', 'models.demos.blackhole.qwen36',
                 'models.demos.blackhole.qwen36.tt'):
        modules[name] = types.ModuleType(name)
    modules['models.demos.blackhole.qwen36.tt'].tp_common = tpc
    return logged, patch.dict(sys.modules, modules), patch.dict('os.environ', environ, clear=True)


def _run_load(source, agmm, environ):
    """Execute a (patched) load_mlp_weights with the model tree stubbed out."""
    import types
    built = []
    logged, modules, env = _stubbed(agmm, environ)
    namespace = {'MLPWeights': object, '_build_gate_up': lambda *a: built.append(a) or 'wgu'}
    with modules, env:
        exec(compile(source, 'mlp.py', 'exec'), namespace)
        result = namespace['load_mlp_weights'](None, {'gate_proj.weight': 1, 'up_proj.weight': 2},
                                               None, types.SimpleNamespace(num_devices=2))
        mlp = namespace['Qwen36MLP'](None, {}, None, types.SimpleNamespace(num_devices=2))
    return result, built, logged, mlp._fuse_gateup_agmm


def _run_layer(source, agmm, environ, devices=2):
    import types
    logged, modules, env = _stubbed(agmm, environ)
    namespace = {}
    args = types.SimpleNamespace(num_devices=devices, is_full_attention_layer=lambda n: False)
    with modules, env:
        exec(compile(source, 'layer.py', 'exec'), namespace)
        layer = namespace['Qwen36DecoderLayer'](None, args, {}, 7)
    return layer._fuse_ff_agmm, layer.ffn_norm, logged


def _strip_c1(text):
    """Remove every C1 insertion: flag-gated if-blocks, the local import and the flag term."""
    kept = []
    skipping = None
    dropping_helper = False
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if skipping is not None:
            if line.strip() and len(line) - len(line.lstrip()) <= skipping:
                skipping = None
            else:
                continue
        stripped = line.lstrip()
        # C1d's and C1e's branches are elifs keyed on the attribute their own flag-gated blocks set.
        gated = 'QWEN_FAST_SINGLE_GATEUP' in line or (stripped.startswith('elif ') and (
            '_qwen_c1_agmm' in line or '"_qwen_c1e"' in line))
        if gated and stripped.startswith(('if ', 'elif ')) and line.rstrip().endswith(':'):
            skipping = len(line) - len(stripped)
            continue
        if stripped.startswith('# Lever N M3native C1'):
            continue
        if stripped.startswith('_qwen_linear = '):
            continue
        if line.startswith('_QWEN_C1_SLICE_ROWS = '):
            dropping_helper = True
        if dropping_helper:
            if line.startswith('class '):
                dropping_helper = False
            else:
                continue
        if stripped.startswith('# gate/up copy and never fuses the gather'):
            continue
        if stripped == 'import os\n' and len(line) - len(stripped) > 0:
            if index + 1 < len(lines) and not lines[index + 1].strip():
                lines[index + 1] = '\0'
            continue
        if line == '\0':
            continue
        kept.append(line.replace(' and os.environ.get("QWEN_FAST_SINGLE_GATEUP") != "1"', '')
                    .replace('= _qwen_linear(', '= ttnn.linear('))
    return ''.join(kept)


class SingleGateUpGraftTests(unittest.TestCase):
    """C1 of the 4 x 131k DRAM plan: QWEN_FAST_SINGLE_GATEUP=1 drops the packed w_gate_up
    copy (~3.21 GB per chip). Unset, every grafted line must behave exactly as before."""

    def test_without_the_flag_the_weight_is_built_the_branch_fuses_and_nothing_is_logged(self):
        patched = patch_mlp_full(MLP_MODULE)
        for environ in ({}, {'QWEN_FAST_SINGLE_GATEUP': '0'}):
            with self.subTest(environ=environ):
                result, built, logged, fused = _run_load(patched, True, environ)
                self.assertEqual((result, len(built), logged, fused), ('wgu', 1, [], True))
        # A switch that already says no, with the flag unset, is some other configuration.
        self.assertEqual(_run_load(patched, False, {})[2:], ([], False))

    def test_with_the_flag_no_copy_is_built_the_branch_is_off_and_the_marker_fires(self):
        result, built, logged, fused = _run_load(patch_mlp_full(MLP_MODULE), True, {'QWEN_FAST_SINGLE_GATEUP': '1'})
        self.assertIsNone(result)
        self.assertEqual(built, [])
        self.assertFalse(fused)
        self.assertEqual(len(logged), 1)
        self.assertIn('[PINDIAG] single gate/up copy: w_gate_up not built', logged[0][0])

    def test_the_ff_norm_keeps_its_gather_exactly_when_the_mlp_stops_fusing(self):
        patched = patch_layer(LAYER)
        self.assertEqual(_run_layer(patched, True, {}), (True, False, []))
        self.assertEqual(_run_layer(LAYER, True, {}), _run_layer(patched, True, {}))
        fused, gathers, logged = _run_layer(patched, True, {'QWEN_FAST_SINGLE_GATEUP': '1'})
        self.assertEqual((fused, gathers), (False, True))
        self.assertEqual(len(logged), 1)
        self.assertIn('[PINDIAG] single gate/up copy: ff_norm gathers its own input', logged[0][0])
        # One device never fused, so the flag has nothing to say there.
        self.assertEqual(_run_layer(patched, True, {'QWEN_FAST_SINGLE_GATEUP': '1'}, devices=1), (False, True, []))

    def test_layer_and_mlp_flip_together(self):
        """A ff_norm that skips its gather in front of an MLP that no longer fuses one would
        hand the w1/w3 matmul a K-sharded input."""
        for environ in ({}, {'QWEN_FAST_SINGLE_GATEUP': '1'}):
            with self.subTest(environ=environ):
                layer_fused = _run_layer(patch_layer(LAYER), True, environ)[0]
                mlp_fused = _run_load(patch_mlp_full(MLP_MODULE), True, environ)[3]
                self.assertEqual(layer_fused, mlp_fused)

    def test_the_prefill_marker_sits_inside_the_2d_branch_and_is_flag_gated(self):
        patched = patch_mlp_full(MLP_MODULE)
        span = function_span(patched, '_forward_tp')
        region = ''.join(patched.splitlines(keepends=True)[span[0]:span[1]])
        branch = region.index('        elif x.shape[-2] > ttnn.TILE_SIZE:\n')
        marker = region.index('[PINDIAG] prefill MLP via w1/w3 2D branch')
        next_branch = region.index('        else:\n', branch)
        self.assertLess(branch, marker)
        self.assertLess(marker, next_branch)

    def test_removing_the_c1_edits_gives_back_the_originals(self):
        """Flag-off equivalence, structurally: C1 (and C1c/C1d/C1e) only insert gated blocks and a flag term."""
        self.assertEqual(_strip_c1(patch_mlp_full(MLP_MODULE)), patch_mlp(MLP_MODULE))
        self.assertEqual(_strip_c1(patch_layer(LAYER)), LAYER)
        self.assertEqual(_strip_c1(patcher.patch_layer_c1_exact(patch_layer(LAYER))), LAYER)

    def test_each_c1_patch_applies_exactly_once_and_fails_loudly_on_drift(self):
        with self.assertRaises(ValueError):
            patch_mlp_single_gateup(patch_mlp_single_gateup(MLP_MODULE))
        with self.assertRaises(ValueError):
            patch_layer(patch_layer(LAYER))
        with self.assertRaises(ValueError):
            patch_layer(LAYER.replace('tpc.mlp_gateup_agmm_enabled(self.num_devices)', 'True'))
        with self.assertRaisesRegex(ValueError, 'no module function'):
            patch_mlp_single_gateup(MLP_MODULE.replace('def load_mlp_weights', 'def load_weights'))

    def test_module_function_span_ends_at_the_next_top_level_statement(self):
        start, end = module_function_span(MLP_MODULE, 'load_mlp_weights')
        lines = MLP_MODULE.splitlines()
        self.assertTrue(lines[start].startswith('def load_mlp_weights'))
        self.assertTrue(lines[end].startswith('import ttnn'))

    def test_the_table_grafts_mlp_and_layer_but_never_tp_common(self):
        self.assertIs(patcher.PATCHES['mlp.py'], patch_mlp_full)
        # patch_layer_full = patch_layer (C1, C1d) then the M2 prefill profile flush hook.
        self.assertIs(patcher.PATCHES['layer.py'], patcher.patch_layer_full)
        self.assertNotIn('tp_common.py', patcher.PATCHES)

    def test_no_graft_touches_a_source_the_serving_path_qualification_pins(self):
        """Run 35802496949 (v95): a grafted tp_common.py died at attach, 'Simulator-qualified
        source changed', because mlp_down_grid_gate pins its sha256. The pinned set is read
        from the gates dflash_combined_request actually enters, so a new pin is seen here."""
        import re
        here = Path(__file__).parent
        request = (here / 'dflash_combined_request.py').read_text(encoding='utf-8')
        gates = sorted(set(re.findall(r'^from (\w+_gate) import qualify', request, re.M)))
        self.assertIn('mlp_down_grid_gate', gates)
        pinned = set()
        for gate in gates:
            text = (here / (gate + '.py')).read_text(encoding='utf-8')
            pinned |= set(re.findall(r'models/demos/blackhole/qwen36/tt/([A-Za-z0-9_/]+\.py)', text))
        self.assertIn('tp_common.py', pinned)
        self.assertEqual(sorted(pinned & set(patcher.PATCHES)), [])

    def test_every_decode_side_file_is_mounted_by_the_arm(self):
        """A patched file the arm does not mount is patched and never served."""
        arm = (Path(__file__).parent / 'lever_n_m3native_run_arm.sh').read_text(encoding='utf-8')
        for relative in sorted(patcher.SOURCES):
            with self.subTest(graft=relative):
                self.assertIn('src=$PWD/graft/%s,dst=$root/%s,readonly' % (relative, relative), arm)
        self.assertNotIn('graft/tp_common.py', arm)




class SingleGateUpSlicingTests(unittest.TestCase):
    """At TP2 the unfused 2D gate/up program does not fit L1 at 2048 rows: v100 (run
    35807762937) hit its L1 outputs, v102 (run 35808212287) its own circular buffers. Under
    the flag the branch runs 1024-row slices with DRAM outputs, joined on rows."""

    def _helper(self, rank=4):
        """Execute the grafted module's helper against a fake ttnn and tp_common."""
        import types
        calls = []

        class Tensor:
            def __init__(self, shape, name):
                self.shape, self.name = tuple(shape), name

        ttnn = types.SimpleNamespace(
            UnaryOpType=types.SimpleNamespace(SILU='SILU'), DRAM_MEMORY_CONFIG='DRAM',
            slice=lambda x, b, e: calls.append(('slice', b[rank - 2], e[rank - 2])) or Tensor(
                x.shape[:rank - 2] + (e[rank - 2] - b[rank - 2], x.shape[-1]), 'slice'),
            linear=lambda x, weight, **k: calls.append(('linear', x.shape[-2], weight.name, k['program_config'],
                                                        k['memory_config'])) or Tensor(x.shape[:-1] + (8704,), 'out'),
            concat=lambda parts, dim, memory_config: calls.append(('concat', len(parts), dim, memory_config)) or Tensor(
                parts[0].shape[:rank - 2] + (sum(p.shape[-2] for p in parts), 8704), 'joined'),
            deallocate=lambda t: calls.append(('free', t.name)))
        tpc = types.SimpleNamespace(create_prefill_mlp_matmul_program_config=lambda rows, k, n, **o: (rows, k, n, o.get(
            'fused_activation'), o.get('max_cols'), o.get('tuning')))
        source = patch_mlp_full(MLP_MODULE)
        start = source.index('_QWEN_C1_SLICE_ROWS = ')
        end = source.index('class Qwen36MLP:')
        namespace = {'ttnn': ttnn}
        exec(source[start:end], namespace)
        w = types.SimpleNamespace(w1=Tensor((5120, 8704), 'w1'), w3=Tensor((5120, 8704), 'w3'))
        args = types.SimpleNamespace(dim=5120)
        return namespace['_qwen_c1_linear'](args, tpc, w, 11, 'tune'), w, Tensor, calls

    def test_2048_rows_run_as_two_1024_row_slices_joined_in_dram(self):
        linear, w, Tensor, calls = self._helper()
        x = Tensor((1, 1, 2048, 5120), 'x')
        out = linear(x, w.w1, compute_kernel_config='ckc', program_config='ignored', memory_config='L1')
        self.assertEqual(out.shape, (1, 1, 2048, 8704))
        self.assertEqual(calls, [
            ('slice', 0, 1024), ('linear', 1024, 'w1', (1024, 5120, 8704, 'SILU', 11, 'tune'), 'DRAM'), ('free', 'slice'),
            ('slice', 1024, 2048), ('linear', 1024, 'w1', (1024, 5120, 8704, 'SILU', 11, 'tune'), 'DRAM'), ('free', 'slice'),
            ('concat', 2, 2, 'DRAM'), ('free', 'out'), ('free', 'out')])

    def test_up_carries_no_activation(self):
        linear, w, Tensor, calls = self._helper()
        linear(Tensor((1, 1, 2048, 5120), 'x'), w.w3, compute_kernel_config='ckc')
        self.assertEqual({c[3][3] for c in calls if c[0] == 'linear'}, {None})

    def test_up_to_1024_rows_is_one_unsliced_call_with_the_branchs_own_builder_arguments(self):
        linear, w, Tensor, calls = self._helper()
        x = Tensor((1, 1, 512, 5120), 'x')
        out = linear(x, w.w1, compute_kernel_config='ckc')
        self.assertEqual(calls, [('linear', 512, 'w1', (512, 5120, 8704, 'SILU', 11, 'tune'), 'DRAM')])
        self.assertEqual(out.name, 'out')

    def test_a_ragged_tail_gets_its_own_short_slice(self):
        linear, w, Tensor, calls = self._helper()
        linear(Tensor((1, 1, 1536, 5120), 'x'), w.w3, compute_kernel_config='ckc')
        self.assertEqual([c[1:] for c in calls if c[0] == 'slice'], [(0, 1024), (1024, 1536)])

    def test_the_branch_selects_plain_ttnn_linear_unless_the_flag_is_one(self):
        import os
        import types
        line = [l.strip() for l in patch_mlp_full(MLP_MODULE).splitlines() if l.strip().startswith('_qwen_linear = ')][0]
        ttnn = types.SimpleNamespace(linear='ttnn.linear')
        for environ, expected in (({}, 'ttnn.linear'), ({'QWEN_FAST_SINGLE_GATEUP': '0'}, 'ttnn.linear'),
                                  ({'QWEN_FAST_SINGLE_GATEUP': '1'}, 'sliced')):
            with self.subTest(environ=environ), patch.dict('os.environ', environ, clear=True):
                namespace = dict(ttnn=ttnn, os=os, args=None, tpc=None, w=None, _gw=None, _pt=None,
                                 _qwen_c1_linear=lambda *a: 'sliced')
                exec(line, namespace)
                self.assertEqual(namespace['_qwen_linear'], expected)

    def test_both_2d_calls_go_through_the_selection(self):
        patched = patch_mlp_full(MLP_MODULE)
        span = function_span(patched, '_forward_tp')
        region = ''.join(patched.splitlines(keepends=True)[span[0]:span[1]])
        self.assertEqual(region.count('_out = _qwen_linear('), 2)
        self.assertLess(region.index('_qwen_linear = '), region.index('w1_out = _qwen_linear('))
        self.assertLess(region.index('        elif x.shape[-2] > ttnn.TILE_SIZE:\n'), region.index('_qwen_linear = '))


# ---------------------------------------------------------------------------------
# C1c / C1d (prefill ranking levers 3a and 3) and the M2 prefill profile flush hook.
# ---------------------------------------------------------------------------------

def _interleave(w1, w3):
    """The served per-chip packing (prepare_for_fused_swiglu's shard): column tile 2t = w1 tile t, 2t+1 = w3 tile t."""
    import numpy as np

    k, n = w1.shape
    return np.stack((w1.reshape(k, n // 32, 32), w3.reshape(k, n // 32, 32)), axis=-2).reshape(k, 2 * n)


def _mlp_env(environ, *, numeric=True, corrupt_pack=False):
    """A fake ttnn / tp_common / ccl / mlp_c1e_pack for executing a grafted mlp.py's _forward_tp.

    Tensors are numpy float32 arrays, so the fakes also compute: linear is x @ w with the
    program config's fused SiLU, mul is elementwise, slice and concat are numpy's, the fused
    SwiGLU AGMM gathers a K-sharded x (the fake's two chips hold the same shard) and forms
    silu(gate) * up from a tile-pair-interleaved weight, and the C1e pack writes that
    interleave of w1/w3 into the scratch (one element off with corrupt_pack). Every call is
    recorded (with names, not ids) so a test can compare the op sequence of two sources, or
    the values two branches produce."""
    import sys
    import types
    import numpy as np

    calls = []
    names = {}

    def name(t):
        return names.get(id(t), 'act')

    def silu(a):
        return a / (1.0 + np.exp(-a))

    def linear(x, weight, compute_kernel_config=None, program_config=None, memory_config=None, activation=None):
        calls.append(('linear', x.shape[-2], name(weight), program_config, memory_config, compute_kernel_config))
        out = np.matmul(x, weight)
        act = program_config[3] if isinstance(program_config, tuple) else activation
        return silu(out) if act in ('SILU', 'silu') else out

    def slice_(x, begins, ends, memory_config=None):
        calls.append(('slice', begins[-2], ends[-2]))
        return x[tuple(slice(b, e) for b, e in zip(begins, ends))].copy()

    def concat(parts, dim, memory_config=None):
        calls.append(('concat', len(parts), dim, memory_config))
        return np.concatenate(parts, axis=dim)

    def mul(a, b, memory_config=None):
        calls.append(('mul', a.shape[-2], memory_config))
        return a * b

    ttnn = types.SimpleNamespace(
        TILE_SIZE=32, DRAM_MEMORY_CONFIG='DRAM', L1_MEMORY_CONFIG='L1', L1_WIDTH_SHARDED_MEMORY_CONFIG='L1WS',
        UnaryOpType=types.SimpleNamespace(SILU='SILU'),
        linear=linear, slice=slice_, concat=concat, mul=mul,
        silu=lambda a, memory_config=None: calls.append(('silu',)) or silu(a),
        to_memory_config=lambda x, mc: calls.append(('to_memory_config', mc)) or x,
        deallocate=lambda t: None)

    def agmm(x, weight, tt_ccl, compute_cfg, topology, grid=(7, 9), cluster_axis=1, fused_activation=None,
             out_memory_config='DRAM'):
        calls.append(('agmm', x.shape[-2], x.shape[-1], name(weight), compute_cfg, topology, fused_activation,
                      out_memory_config))
        return np.zeros(x.shape[:-1] + (weight.shape[-1],), dtype=np.float32)

    def swiglu_agmm(x, weight, tt_ccl, compute_cfg, topology, **options):
        calls.append(('agmm_swiglu', x.shape[-2], name(weight), compute_cfg, tt_ccl, topology, tuple(sorted(options))))
        if weight.shape[-1] % 64 or 2 * x.shape[-1] != weight.shape[-2]:
            return np.zeros(x.shape[:-1] + (weight.shape[-1] // 2,), dtype=np.float32)
        product = np.matmul(np.concatenate([x, x], axis=-1), weight)
        pairs = product.reshape(product.shape[:-1] + (weight.shape[-1] // 64, 2, 32))
        return (silu(pairs[..., 0, :]) * pairs[..., 1, :]).reshape(product.shape[:-1] + (weight.shape[-1] // 2,))

    def pack_gate_up(mesh, w1, w3, scratch):
        calls.append(('c1e_pack', mesh, name(w1), name(w3), name(scratch)))
        scratch[...] = _interleave(w1, w3)
        if corrupt_pack:
            scratch[0, 0] += 1.0
        return scratch

    def audit_pairs(operations, mesh, packed, w1, w3):
        calls.append(('c1e_audit', mesh, name(packed), name(w1), name(w3)))
        wrong = int(np.count_nonzero(packed != _interleave(w1, w3)))
        return dict(exact=wrong == 0, pages=w1.size // 1024, mismatched_words=wrong)

    c1e = types.ModuleType('models.demos.blackhole.qwen36.tt.mlp_c1e_pack')
    c1e.pack_gate_up, c1e.audit_pairs = pack_gate_up, audit_pairs
    c1e.traffic_bytes = lambda shape: (2 * shape[0] * shape[1] // 1024 * 576,) * 2

    tpc = types.SimpleNamespace(
        TILE_SIZE=32,
        create_prefill_mlp_matmul_program_config=lambda m, k, n, fused_activation=None, max_cols=None, tuning=None: (
            m, k, n, fused_activation, max_cols, tuning),
        all_gather_matmul_prefill=agmm, all_gather_swiglu_prefill=swiglu_agmm,
        mlp_gateup_agmm_enabled=lambda n: n > 1)
    ccl = types.ModuleType('models.tt_transformers.tt.ccl')
    ccl.tt_all_reduce = lambda partial, *a, **k: calls.append(('all_reduce',)) or partial
    logged = []
    loguru = types.ModuleType('loguru')
    loguru.logger = types.SimpleNamespace(info=lambda *a: logged.append(a))
    modules = {'ttnn': ttnn, 'loguru': loguru, 'models.tt_transformers.tt.ccl': ccl,
               'models.demos.blackhole.qwen36.tt.mlp_c1e_pack': c1e}
    for module in ('models', 'models.demos', 'models.demos.blackhole', 'models.demos.blackhole.qwen36',
                   'models.demos.blackhole.qwen36.tt', 'models.tt_transformers', 'models.tt_transformers.tt'):
        modules[module] = types.ModuleType(module)
    modules['models.demos.blackhole.qwen36.tt'].tp_common = tpc
    modules['models.demos.blackhole.qwen36.tt'].mlp_c1e_pack = c1e
    return calls, names, logged, patch.dict(sys.modules, modules), patch.dict('os.environ', environ, clear=True)


def _run_mlp(source, environ, x, *, fuse_agmm=False, packed=False, c1_agmm=None, seed=0, hidden=48, c1e=False,
             repeat=1, corrupt_pack=False):
    """Execute `source`'s Qwen36MLP._forward_tp on x (`repeat` times, one module instance); return
    (last output, calls, logged). packed='interleave' makes w_gate_up the served packing of w1/w3 (the
    fake SwiGLU needs hidden a multiple of 32); c1e=True gives the MLP the C1e scratch its __init__ would."""
    import types
    import numpy as np

    calls, names, logged, modules, env = _mlp_env(environ, corrupt_pack=corrupt_pack)
    rng = np.random.default_rng(seed)
    dim = 64
    w1 = rng.standard_normal((dim, hidden)).astype(np.float32)
    w3 = rng.standard_normal((dim, hidden)).astype(np.float32)
    w2 = rng.standard_normal((hidden, dim)).astype(np.float32)
    if packed == 'interleave':
        wgu = _interleave(w1, w3)
    else:
        wgu = rng.standard_normal((dim, 2 * hidden)).astype(np.float32) if packed else None
    scratch = np.zeros((dim, 2 * hidden), dtype=np.float32) if c1e else None
    for tensor, label in ((w1, 'w1'), (w2, 'w2'), (w3, 'w3'), (wgu, 'w_gate_up'), (scratch, 'scratch')):
        if tensor is not None:
            names[id(tensor)] = label
    with modules, env:
        namespace = {'MLPWeights': object, '_build_gate_up': lambda *a: None}
        exec(compile(source, 'mlp.py', 'exec'), namespace)
        cls = namespace['Qwen36MLP']
        mlp = cls.__new__(cls)
        mlp.weights = types.SimpleNamespace(w1=w1, w2=w2, w3=w3, w_gate_up=wgu)
        mlp.args = types.SimpleNamespace(
            dim=dim, decode_grid_w=11, prefill_tuning='tune', ccl_topology=lambda: 'ring',
            mlp_w1_decode_1d_progcfg='w1_1d', mlp_w3_decode_1d_progcfg='w3_1d', mlp_w2_decode_1d_progcfg='w2_1d',
            mlp_w1_decode_1d_progcfg_64='w1_1d_64', mlp_w3_decode_1d_progcfg_64='w3_1d_64',
            mlp_w2_decode_1d_progcfg_64='w2_1d_64')
        mlp.tt_ccl, mlp.device = 'ccl', 'mesh'
        mlp.compute_kernel_config, mlp.compute_kernel_config_decode = 'ckc', 'ckc_decode'
        mlp.compute_kernel_config_agmm = 'ckc_agmm'
        mlp._mlp_1d_decode, mlp._dram_sharded, mlp._fuse_gateup_agmm = True, False, fuse_agmm
        if c1_agmm is not None:
            mlp._qwen_c1_agmm = c1_agmm
        if c1e:
            mlp._qwen_c1e = scratch
        for _ in range(repeat):
            out = mlp._forward_tp(x)
    return out, calls, logged


def _x(rows, width=64, seed=1):
    import numpy as np
    return np.random.default_rng(seed).standard_normal((1, 1, rows, width)).astype(np.float32)


SINGLE = {'QWEN_FAST_SINGLE_GATEUP': '1'}
LEGACY = {'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_C1_LEGACY': '1'}
C1D = {'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_C1_AGMM': '1'}


class C1cGraftTests(unittest.TestCase):
    """C1c: slice x once per 1024 rows for both w1 and w3, multiply per slice, join only the
    product. Default whenever QWEN_FAST_SINGLE_GATEUP=1; must be bit-identical to C1."""

    def setUp(self):
        self.full = patch_mlp_full(MLP_MODULE)

    def test_c1c_is_bit_identical_to_c1_on_the_values_the_ops_produce(self):
        import numpy as np
        for rows in (2048, 1536, 1024, 512):
            with self.subTest(rows=rows):
                x = _x(rows)
                c1c = _run_mlp(self.full, SINGLE, x)[0]
                c1 = _run_mlp(self.full, LEGACY, x)[0]
                self.assertTrue(np.array_equal(c1c, c1), 'C1c changed a value against C1')

    def test_c1c_runs_exactly_c1s_matmuls_one_slice_per_1024_rows_and_one_concat(self):
        x = _x(2048)
        c1c = _run_mlp(self.full, SINGLE, x)[1]
        c1 = _run_mlp(self.full, LEGACY, x)[1]
        matmuls = lambda calls: sorted(c for c in calls if c[0] == 'linear' and c[2] in ('w1', 'w3'))
        # Same programs (builder arguments), same rows, same compute config, same DRAM output.
        self.assertEqual(matmuls(c1c), matmuls(c1))
        self.assertEqual(matmuls(c1c), sorted([
            ('linear', 1024, 'w1', (1024, 64, 48, 'SILU', 11, 'tune'), 'DRAM', 'ckc_decode'),
            ('linear', 1024, 'w1', (1024, 64, 48, 'SILU', 11, 'tune'), 'DRAM', 'ckc_decode'),
            ('linear', 1024, 'w3', (1024, 64, 48, None, 11, 'tune'), 'DRAM', 'ckc_decode'),
            ('linear', 1024, 'w3', (1024, 64, 48, None, 11, 'tune'), 'DRAM', 'ckc_decode')]))
        count = lambda calls, op: sum(1 for c in calls if c[0] == op)
        self.assertEqual((count(c1, 'slice'), count(c1, 'concat'), count(c1, 'mul')), (4, 2, 1))
        self.assertEqual((count(c1c, 'slice'), count(c1c, 'concat'), count(c1c, 'mul')), (2, 1, 2))
        self.assertEqual([c for c in c1c if c[0] == 'mul'], [('mul', 1024, 'DRAM'), ('mul', 1024, 'DRAM')])
        self.assertEqual([c for c in c1c if c[0] == 'concat'], [('concat', 2, 2, 'DRAM')])
        # Order: each slice feeds w1 then w3, then their product, before the next slice.
        head = [c[0] if c[0] != 'linear' else c[2] for c in c1c][:5]
        self.assertEqual(head, ['slice', 'w1', 'w3', 'mul', 'slice'])

    def test_the_down_projection_after_c1c_is_the_one_after_c1(self):
        """_silu_fused stays True in C1c, so the w2 output keeps C1's L1 placement."""
        x = _x(2048)
        tail = lambda calls: [c for c in calls if c[0] in ('linear', 'all_reduce') and c[2:3] != ('w1',)
                              and c[2:3] != ('w3',)]
        self.assertEqual(tail(_run_mlp(self.full, SINGLE, x)[1]), tail(_run_mlp(self.full, LEGACY, x)[1]))
        self.assertIn(('linear', 2048, 'w2', (2048, 48, 64, None, 11, 'tune'), 'L1', 'ckc_decode'),
                      _run_mlp(self.full, SINGLE, x)[1])

    def test_the_c1c_marker_fires_once_inside_the_branch_and_legacy_keeps_c1s_marker(self):
        logged = _run_mlp(self.full, SINGLE, _x(2048))[2]
        self.assertEqual([entry[0].split(':')[0] for entry in logged], ['[PINDIAG] prefill MLP C1c'])
        logged = _run_mlp(self.full, LEGACY, _x(2048))[2]
        self.assertEqual([entry[0].split(':')[0] for entry in logged], ['[PINDIAG] prefill MLP via w1/w3 2D branch'])

    def test_decode_rows_never_reach_c1c(self):
        """Rows 1-64 keep the 1D decode branch under the flag, exactly as under C1."""
        for rows in (32, 64):
            with self.subTest(rows=rows):
                self.assertEqual(_run_mlp(self.full, SINGLE, _x(rows))[1], _run_mlp(self.full, LEGACY, _x(rows))[1])
                self.assertFalse(any(c[0] in ('slice', 'concat') for c in _run_mlp(self.full, SINGLE, _x(rows))[1]))


class FlagOffEquivalenceTests(unittest.TestCase):
    """With no flag set the fully grafted mlp.py runs the op sequence patch_mlp alone runs."""

    def test_every_branch_runs_the_same_ops_as_before_c1(self):
        full, base = patch_mlp_full(MLP_MODULE), patch_mlp(MLP_MODULE)
        cases = (
            dict(x=_x(2048, width=32), fuse_agmm=True, packed=True),    # v98 fused prefill (K-sharded)
            dict(x=_x(2048), fuse_agmm=False),                          # unfused 2D prefill
            dict(x=_x(64), fuse_agmm=True, packed=True),                # 64-row native decode
            dict(x=_x(32), fuse_agmm=True, packed=True),                # one-tile decode
        )
        for case in cases:
            for environ in ({}, {'QWEN_FAST_C1_AGMM': '1'}, {'QWEN_FAST_C1_LEGACY': '1'}, {'QWEN_FAST_C1_EXACT': '1'},
                            {'QWEN_FAST_C1_EXACT': '1', 'QWEN_FAST_C1_EXACT_AUDIT': '4'}):
                with self.subTest(rows=case['x'].shape[-2], environ=environ):
                    kwargs = dict(case)
                    x = kwargs.pop('x')
                    self.assertEqual(_run_mlp(full, environ, x, **kwargs)[1], _run_mlp(base, environ, x, **kwargs)[1])


class C1dGraftTests(unittest.TestCase):
    """C1d (QWEN_FAST_C1_AGMM=1 with QWEN_FAST_SINGLE_GATEUP=1): two all_gather_matmul_prefill
    calls (w1 with SiLU fused, then w3) and one mul, with the ff_norm skipping its gather."""

    def setUp(self):
        self.full = patch_mlp_full(MLP_MODULE)

    def test_k_sharded_prefill_runs_two_agmms_and_a_mul(self):
        out, calls, logged = _run_mlp(self.full, C1D, _x(2048, width=32), c1_agmm=True)
        self.assertEqual(calls[:3], [
            ('agmm', 2048, 32, 'w1', 'ckc_agmm', 'ring', 'SILU', 'DRAM'),
            ('agmm', 2048, 32, 'w3', 'ckc_agmm', 'ring', None, 'DRAM'),
            ('mul', 2048, 'DRAM')])
        self.assertFalse(any(c[0] in ('slice', 'concat', 'agmm_swiglu') for c in calls))
        # The down projection keeps the fused path's L1 output (_silu_fused), then the reduce.
        self.assertEqual(calls[3][:3], ('linear', 2048, 'w2'))
        self.assertEqual(calls[3][4], 'L1')
        self.assertEqual(calls[4], ('all_reduce',))
        self.assertEqual(out.shape, (1, 1, 2048, 64))
        self.assertEqual([entry[0].split(':')[0] for entry in logged], ['[PINDIAG] prefill MLP C1d'])

    def test_a_full_k_input_never_takes_the_agmm_branch(self):
        """The 64-row native decode input is replicated at full K (run 35558196643)."""
        for rows in (32, 64):
            with self.subTest(rows=rows):
                calls = _run_mlp(self.full, C1D, _x(rows), c1_agmm=True)[1]
                self.assertFalse(any(c[0] == 'agmm' for c in calls))
                self.assertEqual(calls[0], ('to_memory_config', 'L1'))

    def test_c1d_is_inert_without_single_gateup(self):
        """Alone, QWEN_FAST_C1_AGMM sets nothing: the __init__ block needs both flags."""
        source = patch_mlp_full(MLP_MODULE)
        for environ, expected in (({'QWEN_FAST_C1_AGMM': '1'}, False), (C1D, True), (SINGLE, False), ({}, False)):
            with self.subTest(environ=environ):
                loaded = _run_load(source, True, environ)
                import types
                logged, modules, env = _stubbed(True, environ)
                namespace = {'MLPWeights': object, '_build_gate_up': lambda *a: 'wgu'}
                with modules, env:
                    exec(compile(source, 'mlp.py', 'exec'), namespace)
                    mlp = namespace['Qwen36MLP'](None, {}, None, types.SimpleNamespace(num_devices=2))
                self.assertEqual(getattr(mlp, '_qwen_c1_agmm', False), expected)
                # The packed copy is still never built under SINGLE_GATEUP (C1d uses w1/w3).
                self.assertEqual(loaded[1] == [], 'QWEN_FAST_SINGLE_GATEUP' in environ)

    def test_the_ff_norm_skips_its_gather_exactly_when_the_mlp_fuses_one(self):
        """Either the packed SwiGLU AGMM (v98) or C1d's two AGMMs; never neither with a skip."""
        import types
        for environ in ({}, SINGLE, C1D, {'QWEN_FAST_C1_AGMM': '1'}, LEGACY):
            with self.subTest(environ=environ):
                layer_fused, gathers, logged = _run_layer(patch_layer(LAYER), True, environ)
                result, built, _, mlp_fused = _run_load(patch_mlp_full(MLP_MODULE), True, environ)
                logged2, modules, env = _stubbed(True, environ)
                namespace = {'MLPWeights': object, '_build_gate_up': lambda *a: 'wgu'}
                with modules, env:
                    exec(compile(patch_mlp_full(MLP_MODULE), 'mlp.py', 'exec'), namespace)
                    mlp = namespace['Qwen36MLP'](None, {}, None, types.SimpleNamespace(num_devices=2))
                mlp_gathers = mlp_fused or getattr(mlp, '_qwen_c1_agmm', False)
                self.assertEqual(layer_fused, mlp_gathers)
                self.assertEqual(gathers, not layer_fused)

    def test_the_layer_markers_name_what_the_ff_norm_does(self):
        fused, gathers, logged = _run_layer(patch_layer(LAYER), True, C1D)
        self.assertEqual((fused, gathers), (True, False))
        self.assertEqual([entry[0] for entry in logged],
                         ['[PINDIAG] C1d: ff_norm skips its all-gather for the MLP AGMM (layer {})'])
        # One device: nothing is fused and nothing is logged, as under C1.
        self.assertEqual(_run_layer(patch_layer(LAYER), True, C1D, devices=1), (False, True, []))

    def test_c1c_c1d_patch_applies_once_and_needs_c1_first(self):
        with_c1 = patch_mlp_single_gateup(patch_mlp(MLP_MODULE))
        with self.assertRaisesRegex(ValueError, 'already grafted'):
            patcher.patch_mlp_c1_fused(patcher.patch_mlp_c1_fused(with_c1))
        with self.assertRaisesRegex(ValueError, 'needs the C1 graft first'):
            patcher.patch_mlp_c1_fused(patch_mlp(MLP_MODULE))

    def test_the_gate_requires_under_c1d_exactly_the_markers_the_graft_emits(self):
        """Under C1d the ff_norm no longer gathers, so the gate must not demand C1's
        gathers-for-itself marker, and must demand the two C1d markers instead."""
        from lever_n_m3native_gate import SINGLE_GATEUP_MARKERS, flag_marker_report, required_flag_markers
        c1d = required_flag_markers(C1D, 4)['QWEN_FAST_SINGLE_GATEUP']
        self.assertNotIn(patcher.MARKER_FF_NORM_GATHER, c1d)
        for marker in c1d[len(SINGLE_GATEUP_MARKERS) - 1:]:
            self.assertTrue(patcher.MARKER_C1D.startswith(marker) or patcher.MARKER_C1D_FF_NORM.startswith(marker),
                            marker)
        # Default SINGLE_GATEUP proves C1c ran; LEGACY proves C1's 2D branch ran (never the other).
        self.assertEqual(required_flag_markers(SINGLE, 4)['QWEN_FAST_SINGLE_GATEUP'],
                         list(SINGLE_GATEUP_MARKERS) + ['[PINDIAG] prefill MLP C1c: one slice of x per 1024 rows'])
        self.assertEqual(required_flag_markers(LEGACY, 4)['QWEN_FAST_SINGLE_GATEUP'],
                         list(SINGLE_GATEUP_MARKERS) + ['[PINDIAG] prefill MLP via w1/w3 2D branch'])
        # A C1d run's log: the three C1 markers that still fire plus what the graft logs for C1d.
        log = chr(10).join([m + ' 64 layers' for m in SINGLE_GATEUP_MARKERS[:3]]
                           + [patcher.MARKER_C1D_FF_NORM + ' (layer 0)', patcher.MARKER_C1D + ': rows=2048 k_local=2560'])
        self.assertEqual(flag_marker_report(C1D, 4, log)['missing'], [])
        # Read as a default SINGLE_GATEUP run, the same log lacks the ff_norm marker and C1c's.
        self.assertEqual(flag_marker_report(SINGLE, 4, log)['missing'], sorted(
            'QWEN_FAST_SINGLE_GATEUP: ' + m for m in (patcher.MARKER_FF_NORM_GATHER,
                                                       '[PINDIAG] prefill MLP C1c: one slice of x per 1024 rows')))

    def test_the_gate_requires_the_marker_of_the_mlp_branch_that_ran(self):
        """C1c is the default under SINGLE_GATEUP: a graft that silently fell through to C1 (or
        a LEGACY run that took C1c) must fail the gate. The required text is a prefix of what the
        executed branch logs, checked against the logger calls the fakes record."""
        from lever_n_m3native_gate import SINGLE_GATEUP_MARKERS, flag_marker_report, required_flag_markers
        base = [m + ' 64 layers' for m in SINGLE_GATEUP_MARKERS]
        for environ, other in ((SINGLE, LEGACY), (LEGACY, SINGLE)):
            with self.subTest(environ=environ):
                logged = _run_mlp(self.full, environ, _x(2048))[2]
                emitted = [entry[0] for entry in logged]
                wrong = [entry[0] for entry in _run_mlp(self.full, other, _x(2048))[2]]
                required = required_flag_markers(environ, 4)['QWEN_FAST_SINGLE_GATEUP'][-1]
                self.assertTrue(any(line.startswith(required) for line in emitted), (required, emitted))
                self.assertFalse(any(line.startswith(required) for line in wrong))
                self.assertEqual(flag_marker_report(environ, 4, chr(10).join(base + emitted))['missing'], [])
                missing = flag_marker_report(environ, 4, chr(10).join(base + wrong))['missing']
                self.assertEqual(missing, ['QWEN_FAST_SINGLE_GATEUP: ' + required])

    def test_the_gate_requires_the_flush_hooks_first_flush_marker(self):
        from lever_n_m3native_gate import PREFILL_FLUSH_MARKER, flag_marker_report, required_flag_markers
        flag = {'QWEN_PREFILL_PROFILE_FLUSH': '1'}
        self.assertEqual(required_flag_markers(flag, 1), {'QWEN_PREFILL_PROFILE_FLUSH': [PREFILL_FLUSH_MARKER]})
        self.assertEqual(required_flag_markers({}, 1), {})
        events = _run_layer_forward(patcher.patch_layer_full(LAYER_MODULE), flag, _chunks(1))
        logs = [e[1] for e in events if e[0] == 'log']
        self.assertTrue(any(line.startswith(PREFILL_FLUSH_MARKER) for line in logs), logs)
        self.assertEqual(flag_marker_report(flag, 1, chr(10).join(logs))['missing'], [])
        self.assertEqual(flag_marker_report(flag, 1, '')['missing'],
                         ['QWEN_PREFILL_PROFILE_FLUSH: ' + PREFILL_FLUSH_MARKER])

    def test_tp_common_is_still_never_grafted(self):
        self.assertNotIn('tp_common.py', patcher.PATCHES)
        self.assertIn('tpc.all_gather_matmul_prefill(', patch_mlp_full(MLP_MODULE))


# layer.py (image dump; graft artifact of run 35816715775, layer.py.orig sha256 586429a11078f91d),
# Qwen36DecoderLayer.forward lines 153-259 verbatim, appended to the fixture class above.
LAYER_FORWARD = '''    def forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,  # = GDN long_prefill_chunk_size; the only size the chunk-seq prefill kernel supports
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
        gdn_collect=False,
    ):
        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        if self.num_devices > 1:
            # TP: DistributedNorm uses the framework's per-norm memory configs.
            _attn_norm_config = self.args.get_norm_config("attn", _norm_mode)
            # PREFILL: distributed rmsnorm outputs in L1 so the fused in-proj AGMM gathers from L1, not DRAM.
            if _norm_mode == Mode.PREFILL:
                _attn_norm_config = {**_attn_norm_config, "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG}
            # DECODE ff_norm uses the attn_norm layout (act_shard_hidden, 32-core) so Qwen36MLP's input reshard is a no-op and the norm runs on 32 cores not 8; PREFILL keeps the framework ff config.
            if _norm_mode == Mode.DECODE:
                _ff_norm_config = self.args.get_norm_config("attn", _norm_mode)
            else:
                # ff_norm output stays DRAM: L1 keeps the full-width norm resident across the whole MLP,
                # clashing with each matmul's CBs (w1/w3/w2) for no gain. Verified dead end; keep DRAM.
                _ff_norm_config = self.args.get_norm_config("ff", _norm_mode)
        else:
            # In decode the norm output stays in L1 (as the old rms_norm_ttnn(memory_config=L1) did);
            # in prefill the framework RMSNorm returns interleaved DRAM (matches the old None default).
            _attn_norm_config = _ff_norm_config = (
                {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None
            )
        attn_input = self.attention_norm(x, mode=_norm_mode, norm_config=_attn_norm_config)

        if self.num_devices > 1:
            # TP modules: input is the gathered (full-dim) norm output [1,1,B/S,dim];
            # output is fractured along dim=3. cos/sin are in rope_tp format.
            if self.is_full_attention:
                if mode == "prefill":
                    # Contract/vLLM path supplies a page_table → paged KV prefill; the
                    # demo path (no page_table) uses the internal concat caches.
                    if page_table is not None:
                        attn_output = self.attention.forward_prefill_paged(
                            attn_input,
                            cos,
                            sin,
                            page_table,
                            chunk_page_table=chunk_page_table,
                            chunk_start_idx=chunk_start_idx if chunk_start_idx is not None else 0,
                            chunk_start_idx_tensor=chunk_start_idx_tensor,
                        )
                    else:
                        attn_output = self.attention.forward_prefill(attn_input, cos, sin)
                else:
                    attn_output = self.attention.forward_decode(
                        attn_input, position_tensor, cos, sin, page_table=page_table
                    )
            else:
                # GDN carries its recurrent/conv state internally (capture_state on
                # prefill, read on decode); it has no paged KV, so page_table is N/A.
                if mode == "prefill":
                    if gdn_collect:
                        # Batched per-user prefill: stash this user's from-scratch state for
                        # assembly into row u of the batched buffers (finalize_pending later).
                        attn_output = self.attention.forward_prefill_collect(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len
                        )
                    else:
                        attn_output = self.attention.forward_prefill(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len, capture_state=True
                        )
                else:
                    attn_output = self.attention.forward_decode(attn_input)
        elif self.is_full_attention:
            attn_output = self.attention.forward(
                attn_input,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(
                attn_input, mode=deltanet_mode, chunk_size=chunk_size, valid_len=valid_len
            )
        ttnn.deallocate(attn_input)

        h = ttnn.add(x, attn_output)
        ttnn.deallocate(attn_output)

        ff_input = self.ffn_norm(h, mode=_norm_mode, norm_config=_ff_norm_config)

        ff_output = self.feed_forward.forward(ff_input)
        ttnn.deallocate(ff_input)

        output = ttnn.add(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        return output
'''

LAYER_MODULE = 'import ttnn\n\n\n' + LAYER + '\n' + LAYER_FORWARD


def _run_layer_forward(source, environ, schedule, *, n_layers=64, tracy=True):
    """Drive the (patched) forward through `schedule`: a list of (mode, layer_num) or
    (mode, layer_num, chunk_start_idx) calls (chunk_start_idx None when omitted).

    Returns the recorded events: ('add'), ('sync', layer), ('read', layer), ('signpost', label),
    plus every logged line."""
    import sys
    import types

    events = []
    fake = types.SimpleNamespace(
        L1_MEMORY_CONFIG='L1',
        add=lambda a, b: events.append(('add',)) or 'sum',
        deallocate=lambda t: None,
        synchronize_device=lambda device: events.append(('sync', current[0])),
        ReadDeviceProfiler=lambda device: events.append(('read', current[0])))
    current = [None]
    loguru = types.ModuleType('loguru')
    loguru.logger = types.SimpleNamespace(info=lambda *a: events.append(('log',) + a))
    modules = {'ttnn': fake, 'loguru': loguru}
    if tracy:
        module = types.ModuleType('tracy')
        module.signpost = lambda label: events.append(('signpost', label))
        modules['tracy'] = module
    else:
        modules['tracy'] = None  # import tracy raises ImportError
    mode_type = types.SimpleNamespace(PREFILL='prefill', DECODE='decode')
    args = types.SimpleNamespace(n_layers=n_layers, get_norm_config=lambda kind, mode: {})
    attention = types.SimpleNamespace(
        forward_prefill=lambda a, chunk_size=None, valid_len=None, capture_state=None: 'attn',
        forward_decode=lambda a: 'attn')
    x = types.SimpleNamespace(shape=(1, 1, 2048, 2560))
    with patch.dict(sys.modules, modules), patch.dict('os.environ', environ, clear=True):
        namespace = {}
        exec(compile(source, 'layer.py', 'exec'), namespace)
        namespace['Mode'] = mode_type
        cls = namespace['Qwen36DecoderLayer']
        layers = {}
        for entry in schedule:
            mode, number = entry[:2]
            start = entry[2] if len(entry) > 2 else None
            layer = layers.get(number)
            if layer is None:
                layer = layers[number] = cls.__new__(cls)
                layer.layer_num, layer.device, layer.args, layer.num_devices = number, 'mesh', args, 2
                layer.is_full_attention = False
                layer.attention_norm = layer.ffn_norm = lambda t, mode=None, norm_config=None: 'normed'
                layer.attention = attention
                layer.feed_forward = types.SimpleNamespace(forward=lambda t: 'ff')
            current[0] = number
            layer.forward(x, mode=mode, chunk_start_idx=start)
    return events


def _chunks(count, n_layers=64, rows=2048):
    """One prompt of `count` chunks, as the served paged path calls it (chunk_start_idx per chunk)."""
    return [('prefill', number, chunk * rows) for chunk in range(count) for number in range(n_layers)]


def _strip_profile(text):
    """Remove the M2 edits: the helper block and the two gated forward blocks."""
    kept, skipping, helper = [], None, False
    for line in text.splitlines(keepends=True):
        if skipping is not None:
            if line.strip() and len(line) - len(line.lstrip()) <= skipping:
                skipping = None
            else:
                continue
        if line.lstrip().startswith('if mode == "prefill" and _qwen_prefill_profile_on():'):
            skipping = len(line) - len(line.lstrip())
            continue
        if line.startswith('# Lever N M2 prefill profile hook'):
            helper = True
        if helper:
            if line.startswith('class '):
                helper = False
            else:
                continue
        kept.append(line)
    return ''.join(kept)


class PrefillProfileFlushTests(unittest.TestCase):
    """M2: under QWEN_PREFILL_PROFILE_FLUSH=1 every 16th prefill layer synchronises and reads the
    device profiler; layer 0 counts chunks; chunks 0, 1, 31, 32, 62, 63 are signposted."""

    def setUp(self):
        self.full = patcher.patch_layer_full(LAYER_MODULE)

    def test_flag_off_the_forward_does_exactly_what_it_did(self):
        schedule = _chunks(2) + [('decode', n) for n in range(64)]
        for environ in ({}, {'QWEN_PREFILL_PROFILE_FLUSH': '0'}):
            with self.subTest(environ=environ):
                events = _run_layer_forward(self.full, environ, schedule)
                self.assertEqual(events, _run_layer_forward(patch_layer(LAYER_MODULE), environ, schedule))
                self.assertEqual(events, _run_layer_forward(LAYER_MODULE, environ, schedule))
                self.assertEqual({e[0] for e in events}, {'add'})

    def test_removing_the_m2_edits_gives_back_the_c1_graft(self):
        # patch_layer_full = C1 (and C1d), then C1e's ff_norm switch (section J), then the M2 hook.
        self.assertEqual(_strip_profile(self.full), patcher.patch_layer_c1_exact(patch_layer(LAYER_MODULE)))
        self.assertEqual(_strip_c1(_strip_profile(self.full)), LAYER_MODULE)

    def test_every_16th_prefill_layer_syncs_then_reads(self):
        events = _run_layer_forward(self.full, {'QWEN_PREFILL_PROFILE_FLUSH': '1'},
                                    _chunks(2) + [('decode', n) for n in range(64)])
        device = [e for e in events if e[0] in ('sync', 'read')]
        expected = [(kind, layer) for _ in range(2) for layer in (15, 31, 47, 63) for kind in ('sync', 'read')]
        self.assertEqual(device, expected, 'decode calls must never flush')

    def test_the_last_layer_flushes_even_off_the_16_grid(self):
        events = _run_layer_forward(self.full, {'QWEN_PREFILL_PROFILE_FLUSH': '1'}, _chunks(1, n_layers=40),
                                    n_layers=40)
        self.assertEqual([e[1] for e in events if e[0] == 'read'], [15, 31, 39])

    def test_chunks_0_1_31_32_62_63_are_signposted_and_nothing_else(self):
        events = _run_layer_forward(self.full, {'QWEN_PREFILL_PROFILE_FLUSH': '1'}, _chunks(66))
        labels = [e[1] for e in events if e[0] == 'signpost']
        expected = []
        for chunk in (0, 1, 31, 32, 62, 63):
            expected += ['qwen_prefill_p1_chunk_%d_begin' % chunk, 'qwen_prefill_p1_chunk_%d_end' % chunk]
        self.assertEqual(labels, expected)
        # The end signpost sits after the last layer's sync and before its read.
        index = events.index(('signpost', 'qwen_prefill_p1_chunk_0_end'))
        self.assertEqual(events[index - 1], ('sync', 63))
        self.assertEqual(events[index + 1], ('read', 63))

    def test_the_chunk_counter_counts_layer_0_prefill_calls_only(self):
        schedule = ([('decode', 0)] * 3 + [('prefill', n) for n in range(1, 64)] + _chunks(2))
        events = _run_layer_forward(self.full, {'QWEN_PREFILL_PROFILE_FLUSH': '1'}, schedule)
        begins = [e for e in events if e[0] == 'log' and 'begin rows' in e[1]]
        # (prompt, chunk, rows, chunk_start_idx, absolute layer-0 prefill call)
        self.assertEqual([e[2:] for e in begins], [(1, 0, 2048, 0, 1), (1, 1, 2048, 2048, 2)])

    def test_warm_up_prefills_do_not_shift_the_prompt_chunk_index(self):
        """Warm-up / probe prefills earlier in the process (an unchunked forward, then a short
        chunked prompt) must not move the real prompt's signposts off chunks 0, 1, 62, 63."""
        warm = [('prefill', n) for n in range(64)] + _chunks(3)
        events = _run_layer_forward(self.full, {'QWEN_PREFILL_PROFILE_FLUSH': '1'}, warm + _chunks(64))
        labels = [e[1] for e in events if e[0] == 'signpost' and e[1].startswith('qwen_prefill_p3_')]
        expected = []
        for chunk in (0, 1, 31, 32, 62, 63):
            expected += ['qwen_prefill_p3_chunk_%d_begin' % chunk, 'qwen_prefill_p3_chunk_%d_end' % chunk]
        self.assertEqual(labels, expected)
        begins = [e for e in events if e[0] == 'log' and 'begin rows' in e[1]]
        # The last prompt's chunk 63 is the process's 68th layer-0 prefill call (1 + 3 + 64).
        self.assertEqual(begins[-1][2:], (3, 63, 2048, 63 * 2048, 68))

    def test_markers_prove_the_hook_executed(self):
        events = _run_layer_forward(self.full, {'QWEN_PREFILL_PROFILE_FLUSH': '1'}, _chunks(3))
        logs = [e[1] for e in events if e[0] == 'log']
        self.assertEqual(sum('first flush' in line for line in logs), 1)
        self.assertEqual(sum('begin rows' in line for line in logs), 2)
        self.assertTrue(all(line.startswith('[PINDIAG] prefill profile flush') for line in logs))

    def test_a_missing_tracy_still_flushes_and_says_so_once(self):
        events = _run_layer_forward(self.full, {'QWEN_PREFILL_PROFILE_FLUSH': '1'}, _chunks(2), tracy=False)
        self.assertEqual(sum(1 for e in events if e[0] == 'read'), 8)
        self.assertFalse(any(e[0] == 'signpost' for e in events))
        self.assertEqual(sum(1 for e in events if e[0] == 'log' and 'signpost unavailable' in e[1]), 1)

    def test_the_profile_patch_applies_once_and_fails_loudly_on_drift(self):
        with self.assertRaisesRegex(ValueError, 'already grafted'):
            patcher.patch_layer_profile_flush(self.full)
        with self.assertRaises(ValueError):
            patcher.patch_layer_profile_flush(LAYER_MODULE.replace('        return output\n', '        return h\n'))
        with self.assertRaises(ValueError):
            patcher.patch_layer_profile_flush(LAYER_MODULE.replace('import ttnn\n\n\n', ''))

    def test_the_flag_name_the_arm_passes_is_the_one_the_graft_reads(self):
        arm = (Path(__file__).parent / 'lever_n_m3native_run_arm.sh').read_text(encoding='utf-8')
        self.assertIn('${M3NATIVE_PROFILE_FLUSH:+-e QWEN_PREFILL_PROFILE_FLUSH=1}', arm)
        self.assertIn('_QWEN_PREFILL_PROFILE_FLAG = "QWEN_PREFILL_PROFILE_FLUSH"', self.full)


# ---------------------------------------------------------------------------------
# C1e (section J): the served fused gate/up on a per-layer rebuild of its packed weight.
# ---------------------------------------------------------------------------------

C1E = {'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_C1_EXACT': '1'}
TP_ARGS = dict(dim=5120, hidden_dim=17408)


def _c1e_module_env(environ, *, agmm=True, exact=True):
    """sys.modules / os.environ for constructing a grafted Qwen36MLP and running its load_mlp_weights
    under C1e: loguru, ttnn, tp_common (with shard_w) and mlp_c1e_pack fakes that record every call."""
    import sys
    import types

    events, logged, scratches = [], [], {}
    tpc = types.SimpleNamespace(
        mlp_gateup_agmm_enabled=lambda n: agmm and n > 1,
        shard_w=lambda weight, mesh, dim, memory_config, cache_path, dtype: events.append(
            ('shard_w', weight, mesh, dim, memory_config, cache_path, dtype)) or ('loaded', cache_path))
    ttnn = types.SimpleNamespace(TILE_SIZE=32, DRAM_MEMORY_CONFIG='DRAM', bfloat4_b='bf4',
                                 deallocate=lambda tensor: events.append(('deallocate', tensor)))
    c1e = types.ModuleType('models.demos.blackhole.qwen36.tt.mlp_c1e_pack')
    c1e.has_scratch = lambda mesh, shape: (mesh, tuple(shape)) in scratches

    def allocate_scratch(operations, mesh, shape):
        key = (mesh, tuple(shape))
        if key not in scratches:
            events.append(('allocate_scratch', operations is ttnn, mesh, tuple(shape)))
            scratches[key] = types.SimpleNamespace(shape=(shape[0], 2 * shape[1]))
        return scratches[key]

    def audit_pairs(operations, mesh, packed, w1, w3):
        events.append(('audit_pairs', mesh, packed, w1, w3))
        return dict(exact=exact, pages=43520, mismatched_words=0 if exact else 9)

    c1e.allocate_scratch, c1e.audit_pairs = allocate_scratch, audit_pairs
    c1e.traffic_bytes = lambda shape: (50135040, 50135040)
    loguru = types.ModuleType('loguru')
    loguru.logger = types.SimpleNamespace(info=lambda *a: logged.append(a))
    modules = {'loguru': loguru, 'ttnn': ttnn, 'models.demos.blackhole.qwen36.tt.mlp_c1e_pack': c1e}
    for name in ('models', 'models.demos', 'models.demos.blackhole', 'models.demos.blackhole.qwen36',
                 'models.demos.blackhole.qwen36.tt'):
        modules[name] = types.ModuleType(name)
    modules['models.demos.blackhole.qwen36.tt'].tp_common = tpc
    modules['models.demos.blackhole.qwen36.tt'].mlp_c1e_pack = c1e
    return events, logged, patch.dict(sys.modules, modules), patch.dict('os.environ', environ, clear=True)


def _c1e_construct(source, environ, *, devices=2, count=1, agmm=True):
    """Construct `count` grafted Qwen36MLPs on one mesh; (mlps, events, logged)."""
    import types
    events, logged, modules, env = _c1e_module_env(environ, agmm=agmm)
    namespace = {'MLPWeights': object, '_build_gate_up': lambda *a: 'wgu'}
    args = types.SimpleNamespace(num_devices=devices, **TP_ARGS)
    with modules, env:
        exec(compile(source, 'mlp.py', 'exec'), namespace)
        mlps = [namespace['Qwen36MLP']('mesh', {}, None, args) for _ in range(count)]
    return mlps, events, logged


def _c1e_load(source, environ, layers=3, *, exact=True, dram_sharded=False):
    """Run the grafted load_mlp_weights for `layers` layers; (results, events, logged)."""
    import types
    from pathlib import PurePosixPath

    events, logged, modules, env = _c1e_module_env(environ, exact=exact)

    def build_gate_up(gate, up, mesh, tp, cache_path):
        events.append(('build_gate_up', gate, up, mesh, tp, cache_path))
        return ('served', cache_path)

    namespace = {'MLPWeights': object, '_build_gate_up': build_gate_up}
    args = types.SimpleNamespace(num_devices=2, **({'mlp_w1_weight_memcfg': 'sharded'} if dram_sharded else {}))
    results = []
    with modules, env:
        exec(compile(source, 'mlp.py', 'exec'), namespace)
        for layer in range(layers):
            results.append(namespace['load_mlp_weights'](
                'mesh', {'gate_proj.weight': 'G%d' % layer, 'up_proj.weight': 'U%d' % layer},
                PurePosixPath('/cache/layers.%d' % layer), args))
    return results, events, logged


def _formatted(logged):
    return [entry[0].format(*entry[1:]) for entry in logged]


class C1eGraftTests(unittest.TestCase):
    """C1e (QWEN_FAST_C1_EXACT=1 with QWEN_FAST_SINGLE_GATEUP=1): the prefill MLP makes the served fused
    call again, on one shared scratch rewritten per layer as that layer's served packing of w1/w3."""

    def setUp(self):
        self.full = patch_mlp_full(MLP_MODULE)
        self.layer = patcher.patch_layer_c1_exact(patch_layer(LAYER))

    def forward_region(self):
        span = function_span(self.full, '_forward_tp')
        return ''.join(self.full.splitlines(keepends=True)[span[0]:span[1]])

    # --- the patch itself -------------------------------------------------------------------

    def test_the_patch_lands_once_on_its_anchors_and_needs_c1_c1c_c1d_first(self):
        fused = patcher.patch_mlp_c1_fused(patch_mlp_single_gateup(patch_mlp(MLP_MODULE)))
        self.assertEqual(patcher.patch_mlp_c1_exact(fused), self.full)
        with self.assertRaisesRegex(ValueError, 'already grafted'):
            patcher.patch_mlp_c1_exact(self.full)
        with self.assertRaisesRegex(ValueError, 'needs the C1, C1c and C1d grafts first'):
            patcher.patch_mlp_c1_exact(patch_mlp_single_gateup(patch_mlp(MLP_MODULE)))
        with self.assertRaisesRegex(ValueError, 'already grafted'):
            patcher.patch_layer_c1_exact(self.layer)
        with self.assertRaisesRegex(ValueError, 'needs the C1/C1d graft first'):
            patcher.patch_layer_c1_exact(LAYER)
        for text in (patcher.C1E_INIT_BLOCK, patcher.C1E_LOAD_BLOCK, patcher.C1E_BRANCH):
            self.assertEqual(self.full.count(text), 1)
        self.assertEqual(self.layer.count(patcher.C1E_LAYER_BLOCK), 1)
        self.assertIs(patcher.PATCHES['mlp.py'], patch_mlp_full)
        self.assertIs(patcher.PATCHES['layer.py'], patcher.patch_layer_full)
        # The helpers sit with C1/C1c's, ahead of the class; tp_common is still never grafted.
        self.assertLess(self.full.index('def _qwen_c1_swiglu('), self.full.index('def _qwen_c1e_pack('))
        self.assertLess(self.full.index('def _qwen_c1e_pack('), self.full.index('class Qwen36MLP:'))
        self.assertNotIn('tp_common.py', patcher.PATCHES)

    def test_the_branch_sits_right_after_the_served_one_and_before_c1d(self):
        region = self.forward_region()
        served = region.index('        if _fused_gu:\n')
        c1e = region.index('        elif getattr(self, "_qwen_c1e", None) is not None')
        c1d = region.index('        elif getattr(self, "_qwen_c1_agmm", False)')
        self.assertLess(served, c1e)
        self.assertLess(c1e, c1d)
        self.assertEqual(region[served:c1e].count('\n        elif '), 0)

    def test_the_call_is_the_served_branchs_with_only_the_weight_renamed(self):
        """The same function, the same positional arguments in the same order, the same keywords (none):
        only w.w_gate_up becomes the scratch the pack returned."""
        served_call = ('            hidden = tpc.all_gather_swiglu_prefill(\n'
                       '                x, w.w_gate_up, self.tt_ccl, self.compute_kernel_config_agmm, args.ccl_topology()\n'
                       '            )\n')
        self.assertIn(served_call, MLP)
        self.assertIn(served_call.replace('w.w_gate_up', '_qwen_c1e_weight'), patcher.C1E_BRANCH)
        self.assertIn('            _qwen_c1e_weight = _qwen_c1e_pack(self, w, x)\n', patcher.C1E_BRANCH)
        # The served branch sets _silu_fused (and its condition is _fused_gu): C1e sets both.
        self.assertTrue(patcher.C1E_BRANCH.endswith('            _silu_fused = True\n            _fused_gu = True\n'))

    def test_no_top_level_import_of_the_op(self):
        head = self.full[:self.full.index('class Qwen36MLP:')]
        for line in head.splitlines():
            if 'mlp_c1e_pack' in line and 'import' in line:
                self.assertTrue(line.startswith('    '), line)   # inside the helpers only
        dotted = patcher.MODEL_ROOT.replace('/opt/tt-metal/', '').replace('/', '.') + '.mlp_c1e_pack'
        self.assertEqual(dotted, patcher.C1E_MODULE)
        self.assertIn('from models.demos.blackhole.qwen36.tt import mlp_c1e_pack as _c1e', self.full)

    # --- construction ---------------------------------------------------------------------

    def test_the_first_mlp_allocates_the_one_scratch_and_every_later_one_shares_it(self):
        mlps, events, logged = _c1e_construct(self.full, C1E, count=3)
        self.assertEqual(events, [('allocate_scratch', True, 'mesh', (5120, 8704))])
        self.assertTrue(all(mlp._qwen_c1e is mlps[0]._qwen_c1e for mlp in mlps))
        self.assertEqual(mlps[0]._qwen_c1e.shape, (5120, 17408))
        self.assertEqual(_formatted(logged), ['[PINDIAG] C1e scratch allocated: per chip (5120, 17408) for w1/w3 '
                                              '(5120, 8704), 50135040 bytes; audit=0'])
        # The served fused branch stays off (no packed copy exists under SINGLE_GATEUP).
        self.assertFalse(any(mlp._fuse_gateup_agmm for mlp in mlps))
        self.assertFalse(any(hasattr(mlp, '_qwen_c1_agmm') for mlp in mlps))

    def test_c1e_refuses_the_other_prefill_paths_and_a_bad_audit_count(self):
        for extra, pattern in (({'QWEN_FAST_C1_AGMM': '1'}, 'excludes QWEN_FAST_C1_AGMM=1'),
                               ({'QWEN_FAST_C1_LEGACY': '1'}, 'excludes QWEN_FAST_C1_LEGACY=1'),
                               ({'QWEN_FAST_C1_EXACT_AUDIT': '65'}, 'must be an integer 0..64'),
                               ({'QWEN_FAST_C1_EXACT_AUDIT': '-1'}, 'must be an integer 0..64'),
                               ({'QWEN_FAST_C1_EXACT_AUDIT': 'two'}, 'must be an integer 0..64')):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, pattern):
                _c1e_construct(self.full, dict(C1E, **extra))
        for audit in ('', '0', '64', ' 3 '):
            with self.subTest(audit=audit):
                mlps, events, _ = _c1e_construct(self.full, dict(C1E, QWEN_FAST_C1_EXACT_AUDIT=audit))
                self.assertIsNotNone(mlps[0]._qwen_c1e)

    def test_c1e_is_inert_without_single_gateup_and_where_nothing_ever_fused(self):
        for environ in ({}, {'QWEN_FAST_C1_EXACT': '1'}, {'QWEN_FAST_C1_EXACT': '1', 'QWEN_FAST_C1_EXACT_AUDIT': '3'},
                        {'QWEN_FAST_SINGLE_GATEUP': '1'}):
            with self.subTest(environ=environ):
                mlps, events, logged = _c1e_construct(self.full, environ)
                self.assertFalse(hasattr(mlps[0], '_qwen_c1e'))
                self.assertEqual((events, logged), ([], []))
        # One device (or a switch that never fused): the attribute exists and is None, so no branch.
        for devices, agmm in ((1, True), (2, False)):
            with self.subTest(devices=devices, agmm=agmm):
                mlps, events, logged = _c1e_construct(self.full, C1E, devices=devices, agmm=agmm)
                self.assertIsNone(mlps[0]._qwen_c1e)
                self.assertEqual((events, logged), ([], []))

    # --- the load: the premise audit ------------------------------------------------------

    def test_the_premise_audit_checks_the_first_n_layers_served_weight_at_load(self):
        results, events, logged = _c1e_load(self.full, dict(C1E, QWEN_FAST_C1_EXACT_AUDIT='2'), layers=3)
        self.assertEqual(results, [None, None, None])     # still no packed copy is built for the model
        expected = []
        for layer in (0, 1):
            root = '/cache/layers.%d/' % layer
            served = ('served', root + 'mlp.gate_up.weight.swiglu.tp')
            w1, w3 = ('loaded', root + 'mlp.gate_proj.weight.tp'), ('loaded', root + 'mlp.up_proj.weight.tp')
            expected += [('build_gate_up', 'G%d' % layer, 'U%d' % layer, 'mesh', 2, served[1]),
                         ('shard_w', 'G%d' % layer, 'mesh', -1, 'DRAM', w1[1], 'bf4'),
                         ('shard_w', 'U%d' % layer, 'mesh', -1, 'DRAM', w3[1], 'bf4'),
                         ('audit_pairs', 'mesh', served, w1, w3),
                         ('deallocate', served), ('deallocate', w1), ('deallocate', w3)]
        self.assertEqual(events, expected)
        lines = _formatted(logged)
        self.assertEqual([line for line in lines if line.startswith(patcher.MARKER_C1E_PREMISE)], [
            '[PINDIAG] C1e premise audit 1 exact=True layer=1 pages=43520 mismatched_words=0',
            '[PINDIAG] C1e premise audit 2 exact=True layer=2 pages=43520 mismatched_words=0'])
        self.assertEqual(sum(line.startswith(patcher.MARKER_SINGLE_GATEUP) for line in lines), 3)

    def test_a_premise_difference_raises_after_freeing_what_it_loaded(self):
        import types
        from pathlib import PurePosixPath

        events, logged, modules, env = _c1e_module_env(dict(C1E, QWEN_FAST_C1_EXACT_AUDIT='1'), exact=False)
        namespace = {'MLPWeights': object, '_build_gate_up': lambda *a: ('served', a[-1])}
        with modules, env:
            exec(compile(self.full, 'mlp.py', 'exec'), namespace)
            with self.assertRaisesRegex(AssertionError, 'not the interleave of w1/w3'):
                namespace['load_mlp_weights']('mesh', {'gate_proj.weight': 'G', 'up_proj.weight': 'U'},
                                              PurePosixPath('/c'), types.SimpleNamespace(num_devices=2))
        self.assertEqual([event[0] for event in events], ['shard_w', 'shard_w', 'audit_pairs', 'deallocate',
                                                          'deallocate', 'deallocate'])
        self.assertIn('[PINDIAG] C1e premise audit 1 exact=False layer=1 pages=43520 mismatched_words=9',
                      _formatted(logged))

    def test_the_premise_refuses_dram_sharded_weights_and_is_silent_without_the_audit(self):
        with self.assertRaisesRegex(ValueError, 'DRAM-interleaved'):
            _c1e_load(self.full, dict(C1E, QWEN_FAST_C1_EXACT_AUDIT='1'), layers=1, dram_sharded=True)
        results, events, logged = _c1e_load(self.full, C1E, layers=2)
        self.assertEqual(events, [])
        self.assertFalse(any(line.startswith(patcher.MARKER_C1E_PREMISE) for line in _formatted(logged)))
        # Without SINGLE_GATEUP the served copy is built and nothing C1e runs, audit or not.
        results, events, logged = _c1e_load(self.full, {'QWEN_FAST_C1_EXACT': '1', 'QWEN_FAST_C1_EXACT_AUDIT': '2'},
                                            layers=2)
        self.assertEqual([e[0] for e in events], ['build_gate_up', 'build_gate_up'])
        self.assertEqual(logged, [])

    # --- the forward ----------------------------------------------------------------------

    def test_k_sharded_prefill_packs_then_makes_the_served_call_and_gets_the_served_values(self):
        import numpy as np
        for rows in (2048, 1056, 128):
            with self.subTest(rows=rows):
                x = _x(rows, width=32)
                served_out, served, served_logged = _run_mlp(self.full, {}, x, fuse_agmm=True, packed='interleave',
                                                             hidden=64)
                out, calls, logged = _run_mlp(self.full, C1E, x, hidden=64, c1e=True)
                self.assertEqual(calls[0], ('c1e_pack', 'mesh', 'w1', 'w3', 'scratch'))
                rename = lambda call: tuple('w_gate_up' if part == 'scratch' else part for part in call)
                self.assertEqual([rename(call) for call in calls[1:]], served)
                self.assertEqual(served[0], ('agmm_swiglu', rows, 'w_gate_up', 'ckc_agmm', 'ccl', 'ring', ()))
                self.assertTrue(np.array_equal(out, served_out))
                self.assertEqual(served_logged, [])
                self.assertEqual([entry[0].split(':')[0] for entry in logged], ['[PINDIAG] prefill MLP C1e'])

    def test_each_prefill_call_packs_once_and_the_marker_fires_once(self):
        calls, logged = _run_mlp(self.full, C1E, _x(2048, width=32), hidden=64, c1e=True, repeat=3)[1:]
        self.assertEqual([c for c in calls if c[0] == 'c1e_pack'], [('c1e_pack', 'mesh', 'w1', 'w3', 'scratch')] * 3)
        self.assertEqual([c[0] for c in calls if c[0] != 'linear'],
                         ['c1e_pack', 'agmm_swiglu', 'all_reduce'] * 3)
        self.assertEqual(len(logged), 1)
        self.assertFalse(any(c[0] in ('slice', 'concat', 'mul', 'agmm', 'c1e_audit') for c in calls))

    def test_decode_rows_never_pack(self):
        """Rows 1-64 are replicated at full K: the 1D decode branch, exactly as under C1c."""
        for rows in (32, 64):
            with self.subTest(rows=rows):
                c1e = _run_mlp(self.full, C1E, _x(rows), hidden=64, c1e=True)[1]
                self.assertEqual(c1e, _run_mlp(self.full, SINGLE, _x(rows), hidden=64)[1])
                self.assertFalse(any(c[0] == 'c1e_pack' for c in c1e))

    def test_a_full_k_prefill_input_falls_through_to_c1c_and_logs_it(self):
        """What the gate forbids under C1e: a prefill MLP that did not take the C1e branch logs C1c's marker."""
        calls, logged = _run_mlp(self.full, C1E, _x(2048), hidden=64, c1e=True)[1:]
        self.assertFalse(any(c[0] == 'c1e_pack' for c in calls))
        self.assertEqual([entry[0].split(':')[0] for entry in logged], ['[PINDIAG] prefill MLP C1c'])

    def test_the_audit_checks_the_first_n_packs_against_w1_w3(self):
        calls, logged = _run_mlp(self.full, dict(C1E, QWEN_FAST_C1_EXACT_AUDIT='2'), _x(2048, width=32), hidden=64,
                                 c1e=True, repeat=3)[1:]
        self.assertEqual([c[0] for c in calls if c[0] in ('c1e_pack', 'c1e_audit')],
                         ['c1e_pack', 'c1e_audit', 'c1e_pack', 'c1e_audit', 'c1e_pack'])
        self.assertIn(('c1e_audit', 'mesh', 'scratch', 'w1', 'w3'), calls)
        audits = [line for line in _formatted(logged) if line.startswith(patcher.MARKER_C1E_AUDIT)]
        self.assertEqual(audits, ['[PINDIAG] C1e audit 1 exact=True pages=4 mismatched_words=0 rows=2048',
                                  '[PINDIAG] C1e audit 2 exact=True pages=4 mismatched_words=0 rows=2048'])

    def test_an_audited_difference_raises(self):
        with self.assertRaisesRegex(AssertionError, 'packed scratch differs'):
            _run_mlp(self.full, dict(C1E, QWEN_FAST_C1_EXACT_AUDIT='1'), _x(2048, width=32), hidden=64, c1e=True,
                     corrupt_pack=True)
        # Unaudited, the same corruption is only a wrong value (which the M + A gate's byte compare sees).
        out = _run_mlp(self.full, C1E, _x(2048, width=32), hidden=64, c1e=True, corrupt_pack=True)[0]
        served = _run_mlp(self.full, {}, _x(2048, width=32), fuse_agmm=True, packed='interleave', hidden=64)[0]
        self.assertFalse((out == served).all())

    # --- layer.py -------------------------------------------------------------------------

    def test_the_ff_norm_skips_its_gather_exactly_when_c1e_fuses_it(self):
        self.assertEqual(_run_layer(self.layer, True, {}), (True, False, []))
        fused, gathers, logged = _run_layer(self.layer, True, C1E)
        self.assertEqual((fused, gathers), (True, False))
        self.assertEqual([entry[0] for entry in logged], [patcher.MARKER_C1E_FF_NORM + ' (layer {})'])
        # SINGLE_GATEUP alone keeps C1/C1c's gather; one device never fused; C1_EXACT alone is the served path.
        self.assertEqual(_run_layer(self.layer, True, SINGLE)[:2], (False, True))
        self.assertEqual(_run_layer(self.layer, True, C1E, devices=1), (False, True, []))
        self.assertEqual(_run_layer(self.layer, True, {'QWEN_FAST_C1_EXACT': '1'}), (True, False, []))

    def test_layer_and_mlp_flip_together(self):
        """The ff_norm skips its gather iff the MLP fuses one: the served copy, C1d's two AGMMs or C1e's."""
        for environ in ({}, SINGLE, C1E, {'QWEN_FAST_C1_EXACT': '1'}, LEGACY, C1D):
            with self.subTest(environ=environ):
                layer_fused, gathers, _ = _run_layer(self.layer, True, environ)
                mlp = _c1e_construct(self.full, environ)[0][0]
                fuses = (mlp._fuse_gateup_agmm or getattr(mlp, '_qwen_c1_agmm', False)
                         or getattr(mlp, '_qwen_c1e', None) is not None)
                self.assertEqual(layer_fused, fuses)
                self.assertEqual(gathers, not layer_fused)

    # --- the gate -------------------------------------------------------------------------

    def graft_log(self, environ, rows=2048):
        """The lines this graft logs in one C1e process: construction, load, the layer and the forward."""
        mlp_lines = _formatted(_c1e_construct(self.full, environ)[2])
        load_lines = _formatted(_c1e_load(self.full, environ, layers=4)[2])
        layer_lines = _formatted(_run_layer(self.layer, True, environ)[2])
        forward_lines = _formatted(_run_mlp(self.full, environ, _x(rows, width=32), hidden=64, c1e=True, repeat=4)[2])
        from lever_n_m3native_gate import SINGLE_GATEUP_MARKERS
        serving = [m + ' 64 layers' for m in SINGLE_GATEUP_MARKERS[:2]]
        return chr(10).join(serving + mlp_lines + load_lines + layer_lines + forward_lines)

    def test_the_gate_requires_exactly_the_markers_the_graft_emits(self):
        from lever_n_m3native_gate import SINGLE_GATEUP_MARKERS, flag_marker_report, required_flag_markers
        required = required_flag_markers(C1E, 4)['QWEN_FAST_SINGLE_GATEUP']
        self.assertNotIn(patcher.MARKER_FF_NORM_GATHER, required)
        self.assertEqual(required[:3], list(SINGLE_GATEUP_MARKERS[:3]))
        for marker, emitted in zip(required[3:], (patcher.MARKER_C1E_SCRATCH, patcher.MARKER_C1E_FF_NORM,
                                                  patcher.MARKER_C1E)):
            self.assertTrue(emitted.startswith(marker), (marker, emitted))
        self.assertEqual(flag_marker_report(C1E, 4, self.graft_log(C1E))['missing'], [])
        # Read as a C1c run (SINGLE_GATEUP alone), the same log lacks C1c's marker and the ff_norm gather.
        self.assertEqual(flag_marker_report(SINGLE, 4, self.graft_log(C1E))['missing'], sorted(
            'QWEN_FAST_SINGLE_GATEUP: ' + m for m in (patcher.MARKER_FF_NORM_GATHER,
                                                       '[PINDIAG] prefill MLP C1c: one slice of x per 1024 rows')))

    def test_the_gate_requires_the_nth_audit_lines(self):
        from lever_n_m3native_gate import flag_marker_report, required_flag_markers
        audited = dict(C1E, QWEN_FAST_C1_EXACT_AUDIT='3')
        self.assertEqual(required_flag_markers(audited, 4)['QWEN_FAST_SINGLE_GATEUP'][-2:],
                         ['[PINDIAG] C1e premise audit 3 exact=True', '[PINDIAG] C1e audit 3 exact=True'])
        self.assertEqual(flag_marker_report(audited, 4, self.graft_log(audited))['missing'], [])
        # A log from an unaudited run does not satisfy an audited arm.
        self.assertEqual(flag_marker_report(audited, 4, self.graft_log(C1E))['missing'], sorted([
            'QWEN_FAST_SINGLE_GATEUP: [PINDIAG] C1e audit 3 exact=True',
            'QWEN_FAST_SINGLE_GATEUP: [PINDIAG] C1e premise audit 3 exact=True']))

    def test_the_gate_fails_a_fall_through_and_the_refused_combinations(self):
        from lever_n_m3native_gate import flag_marker_report
        log = self.graft_log(C1E)
        fell = log + chr(10) + '[PINDIAG] prefill MLP C1c: one slice of x per 1024 rows, product-only concat: rows=2048'
        self.assertEqual(flag_marker_report(C1E, 4, fell)['missing'], [
            'QWEN_FAST_C1_EXACT: every prefill MLP takes the C1e branch '
            '([PINDIAG] prefill MLP C1c: one slice of x per 1024 rows logged)'])
        self.assertIn('QWEN_FAST_C1_EXACT: needs QWEN_FAST_SINGLE_GATEUP=1 (alone it is inert and the served path runs)',
                      flag_marker_report({'QWEN_FAST_C1_EXACT': '1'}, 4, log)['missing'])
        for other in ('QWEN_FAST_C1_AGMM', 'QWEN_FAST_C1_LEGACY'):
            with self.subTest(other=other):
                self.assertIn('QWEN_FAST_C1_EXACT: excludes %s=1 (the graft refuses the pair)' % other,
                              flag_marker_report(dict(C1E, **{other: '1'}), 4, log)['missing'])
        self.assertIn("QWEN_FAST_C1_EXACT_AUDIT: an integer 0..64, not '65'",
                      flag_marker_report(dict(C1E, QWEN_FAST_C1_EXACT_AUDIT='65'), 4, log)['missing'])
        # Without C1_EXACT nothing C1e is asked for, and a C1c run is not failed by C1e's rules.
        self.assertEqual(flag_marker_report({}, 4, log)['missing'], [])

    # --- the mount table, the arm, the workflow, the CPU suite -----------------------------

    def test_the_table_is_the_op_module_and_its_two_kernels_beside_mlp_py(self):
        import mlp_c1e_pack
        self.assertEqual(patcher.C1E_FILES, {name: name for name in mlp_c1e_pack.RUNTIME_FILES})
        self.assertEqual(sorted(patcher.C1E_FILES), ['mlp_c1e_pack.cpp', 'mlp_c1e_pack.py', 'packed_weight_check.cpp'])
        here = Path(__file__).parent
        for relative, source in patcher.C1E_FILES.items():
            self.assertTrue((here / source).is_file(), source)
        self.assertEqual(sorted(set(patcher.C1E_FILES) & set(patcher.with_lever_n())), [])
        self.assertEqual(sorted(set(patcher.C1E_FILES) & set(patcher.PREFILL_CONV_FILES)), [])

    def _arm_block(self, environ, staged):
        import os
        import shutil
        import subprocess
        import tempfile
        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        arm = (Path(__file__).parent / 'lever_n_m3native_run_arm.sh').read_text(encoding='utf-8')
        start = arm.index('c1e_mounts=()')
        end = arm.index(chr(10) + 'fi' + chr(10), start) + 4
        script = ('set -euo pipefail' + chr(10) + 'root=/opt/tt-metal/models/demos/blackhole/qwen36/tt' + chr(10)
                  + arm[start:end] + 'printf "RESULT|%s" "${c1e_mounts[*]:-}"' + chr(10))
        with tempfile.TemporaryDirectory() as directory:
            ci = Path(directory, 'scripts', 'ci')
            ci.mkdir(parents=True)
            for name in ('lever_n_m3native_patch.py', 'gdn_prefill_conv_exact.py'):
                shutil.copy(Path(__file__).parent / name, ci / name)
            for relative in staged:
                target = Path(directory, 'graft', relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('staged', encoding='utf-8')
            env = {k: v for k, v in os.environ.items() if not k.startswith('M3NATIVE_')}
            env.update(environ)
            try:
                return subprocess.run([bash, '-c', script], env=env, cwd=directory, capture_output=True, text=True,
                                      timeout=120)
            except OSError as error:
                self.skipTest('bash unusable: %s' % error)

    def test_the_arm_refuses_what_the_graft_would_before_any_docker_run(self):
        files = sorted(patcher.C1E_FILES)
        cases = (
            ({'M3NATIVE_C1_EXACT': '1'}, 'needs M3NATIVE_SINGLE_GATEUP=1'),
            ({'M3NATIVE_C1_EXACT': 'yes', 'M3NATIVE_SINGLE_GATEUP': '1'}, 'must be 1 or unset'),
            ({'M3NATIVE_C1_EXACT': '1', 'M3NATIVE_SINGLE_GATEUP': '1', 'M3NATIVE_C1_AGMM': '1'}, 'excludes'),
            ({'M3NATIVE_C1_EXACT': '1', 'M3NATIVE_SINGLE_GATEUP': '1', 'M3NATIVE_C1_LEGACY': '1'}, 'excludes'),
            ({'M3NATIVE_C1_EXACT': '1', 'M3NATIVE_SINGLE_GATEUP': '1', 'M3NATIVE_C1_EXACT_AUDIT': '65'}, '0..64'),
            ({'M3NATIVE_C1_EXACT': '1', 'M3NATIVE_SINGLE_GATEUP': '1', 'M3NATIVE_C1_EXACT_AUDIT': 'x'}, '0..64'),
            ({'M3NATIVE_C1_EXACT_AUDIT': '2'}, 'without M3NATIVE_C1_EXACT=1 audits nothing'),
        )
        for environ, message in cases:
            with self.subTest(environ=environ):
                result = self._arm_block(environ, files)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(message, result.stderr)
        result = self._arm_block({}, files)
        self.assertEqual((result.returncode, result.stdout.split('RESULT|')[-1]), (0, ''), result.stderr)

    def test_the_arm_mounts_every_file_of_the_table_one_by_one(self):
        import re
        import shutil
        if shutil.which('python3') is None:
            self.skipTest('no python3')
        files = sorted(patcher.C1E_FILES)
        flags = {'M3NATIVE_C1_EXACT': '1', 'M3NATIVE_SINGLE_GATEUP': '1', 'M3NATIVE_C1_EXACT_AUDIT': '64'}
        result = self._arm_block(flags, files)
        if result.returncode != 0 and ('No module named' in result.stderr or 'No such file' in result.stderr):
            self.skipTest('python3 here cannot import from the temp tree: %s' % result.stderr.strip())
        self.assertEqual(result.returncode, 0, result.stderr)
        mounts = [m.replace('--mount ', '') for m in result.stdout.split('RESULT|')[-1].split(' --mount ')]
        self.assertEqual(len(mounts), len(files))
        for mount, relative in zip(mounts, files):
            self.assertRegex(mount, '^type=bind,src=.*/graft/%s,dst=/opt/tt-metal/models/demos/blackhole/qwen36/tt/%s,'
                                    'readonly$' % (re.escape(relative), re.escape(relative)))
        missing = self._arm_block(flags, files[1:])
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('graft/%s was not staged' % files[0], missing.stderr)

    def test_the_arm_wires_the_mounts_and_both_env_vars_into_the_docker_run(self):
        arm = (Path(__file__).parent / 'lever_n_m3native_run_arm.sh').read_text(encoding='utf-8')
        run = arm[arm.index('timeout -k 30 2200 docker run'):arm.index('--entrypoint python3')]
        lines = [line.strip() for line in run.splitlines()]
        mlp = lines.index('--mount "type=bind,src=$PWD/graft/mlp.py,dst=$root/mlp.py,readonly" ' + chr(92))
        self.assertEqual(lines[mlp + 1], '"${c1e_mounts[@]}" ' + chr(92))
        agmm = lines.index('${M3NATIVE_C1_AGMM:+-e QWEN_FAST_C1_AGMM=1} ' + chr(92))
        self.assertEqual(lines[agmm + 1:agmm + 4], [
            '${M3NATIVE_C1_EXACT:+-e QWEN_FAST_C1_EXACT=1} ' + chr(92),
            '${M3NATIVE_C1_EXACT_AUDIT:+-e QWEN_FAST_C1_EXACT_AUDIT=$M3NATIVE_C1_EXACT_AUDIT} ' + chr(92),
            '${M3NATIVE_C1_LEGACY:+-e QWEN_FAST_C1_LEGACY=1} ' + chr(92)])
        self.assertEqual(run.count('c1e_mounts'), 1)
        self.assertIn('p.C1E_FILES', arm)
        self.assertNotIn('dst=$root,', arm)
        graft = (Path(__file__).parent / 'lever_n_m3native_patch.py').read_text(encoding='utf-8')
        self.assertIn("C1_EXACT_FLAG = 'QWEN_FAST_C1_EXACT'", graft)
        self.assertIn("C1_EXACT_AUDIT_FLAG = 'QWEN_FAST_C1_EXACT_AUDIT'", graft)

    def test_the_workflow_stages_the_same_table_and_hashes_it(self):
        workflow = (Path(__file__).parent.parent.parent / '.github' / 'workflows'
                    / 'qwen-lever-n-m3native-gate.yml').read_text(encoding='utf-8')
        self.assertIn('for relative, source in sorted(patcher.C1E_FILES.items()):', workflow)
        stage = workflow[workflow.index('graft/c1e-manifest.txt'):]
        stage = stage[:stage.index('cat experiment-results/graft.sha256')]
        self.assertIn('cp "scripts/ci/$source" "graft/$relative"', stage)
        self.assertIn('sha256sum "graft/$relative" >> experiment-results/graft.sha256', stage)
        self.assertLess(workflow.index('graft/c1e-manifest.txt'),
                        workflow.index('name: m3native-graft-${{ github.run_id }}'))

    def test_the_c1e_tests_are_allowlisted_in_the_cpu_suite(self):
        cpu = (Path(__file__).parent.parent.parent / '.github' / 'workflows'
               / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        for name in ('test_lever_n_m3native_patch', 'test_mlp_c1e_pack'):
            self.assertRegex(cpu, r'python -B -m unittest [^\n]*\b%s\b' % name)
        self.assertIn("python -B -m unittest discover -s optimisation/ttnn-op/c1e_gateup -p 'test_*.py'", cpu)


if __name__ == '__main__':
    unittest.main()
